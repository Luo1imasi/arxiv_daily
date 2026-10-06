import json
import re
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from loguru import logger
from openai import (
    APIConnectionError,
    APIError as OpenAIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)
from openai.types.chat import ChatCompletionMessageParam

from .config import get_config_value
from .utils import retry_call


PROMPT_VERSIONS = {
    "extract": "keywords-v2",
    "summarize": "tldr-v2",
    "judge": "judge-v1",
    "profile": "profile-v1",
    "qa": "qa-v1",
}

_client_cache: dict[tuple[Any, ...], OpenAI] = {}
_client_cache_lock = threading.Lock()
_llm_request_semaphore_lock = threading.Lock()
_llm_request_semaphores: dict[int, threading.BoundedSemaphore] = {}
_usage_lock = threading.Lock()
_MAX_CACHE_SIZE = 10

RETRY_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OpenAIError,
    OSError,
)


class LLMCallError(Exception):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = True,
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after = retry_after


@dataclass
class LLMUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_requests: int = 0
    error_count: int = 0
    warnings: list[str] = field(default_factory=list)


_usage = LLMUsage()


def reset_llm_usage() -> None:
    global _usage
    with _usage_lock:
        _usage = LLMUsage()


def note_llm_error(message: str) -> None:
    text = str(message or "").strip()
    with _usage_lock:
        _usage.error_count += 1
        if text and text not in _usage.warnings and len(_usage.warnings) < 8:
            _usage.warnings.append(text)


def note_llm_warning(message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    with _usage_lock:
        if text not in _usage.warnings and len(_usage.warnings) < 8:
            _usage.warnings.append(text)


def get_llm_usage() -> dict[str, Any]:
    with _usage_lock:
        return {
            "llm_prompt_tokens": int(_usage.prompt_tokens),
            "llm_completion_tokens": int(_usage.completion_tokens),
            "llm_api_requests": int(_usage.api_requests),
            "llm_error_count": int(_usage.error_count),
            "llm_warning": "；".join(_usage.warnings),
        }


def _record_usage(prompt_tokens: int, completion_tokens: int) -> None:
    with _usage_lock:
        _usage.api_requests += 1
        _usage.prompt_tokens += max(0, int(prompt_tokens or 0))
        _usage.completion_tokens += max(0, int(completion_tokens or 0))


def resolve_model(config: dict[str, Any], role: str) -> str:
    model_role = {"profile": "judge", "qa": "summarize"}.get(role, role)
    models = get_config_value(config, "llm.models", {}) or {}
    if isinstance(models, dict):
        configured = models.get(model_role) or models.get(role)
        if configured:
            return str(configured)
    return str(get_config_value(config, "llm.model"))


def make_llm_cache_key(config: dict[str, Any], role: str) -> str:
    version = PROMPT_VERSIONS.get(role, role)
    return "|".join(
        [
            str(get_config_value(config, "llm.base_url", "")),
            resolve_model(config, role),
            str(get_config_value(config, "llm.language", "")),
            role,
            version,
        ]
    )


def _remove_think_tags(content: str) -> str:
    if not content:
        return content
    content = re.sub(r"<think.*?>.*?</think\s*>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"<\|think\|>.*?<\|/think\|>", "", content, flags=re.DOTALL)
    return content.strip()


def _content_was_think_only(content: str) -> bool:
    """Check if the response contained only reasoning/think tags (common with reasoning models)."""
    if not content:
        return False
    stripped = _remove_think_tags(content)
    return bool(content) and not stripped


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(getattr(item, "text", "") or ""))
        return "\n".join(part for part in parts if part)
    return str(value)


def message_text(message: Any) -> str:
    """Prefer visible content, then reasoning_content when the body is empty."""
    raw_content = _coerce_text(getattr(message, "content", None))
    if isinstance(message, dict):
        raw_content = _coerce_text(message.get("content"))
    stripped = _remove_think_tags(raw_content)
    if stripped:
        return stripped

    reasoning = getattr(message, "reasoning_content", None)
    if not reasoning and isinstance(message, dict):
        reasoning = message.get("reasoning_content")
    if not reasoning:
        extra = getattr(message, "model_extra", None) or {}
        if isinstance(extra, dict):
            reasoning = extra.get("reasoning_content")
    if not reasoning:
        try:
            dumped = message.model_dump()
        except Exception:
            dumped = {}
        if isinstance(dumped, dict):
            reasoning = dumped.get("reasoning_content")
    return _remove_think_tags(_coerce_text(reasoning))


def _get_client(config: dict[str, Any]) -> OpenAI:
    global _client_cache
    cache_key = (
        get_config_value(config, "llm.api_key"),
        get_config_value(config, "llm.base_url"),
        float(get_config_value(config, "llm.timeout", 60)),
    )

    with _client_cache_lock:
        if cache_key not in _client_cache:
            if len(_client_cache) >= _MAX_CACHE_SIZE:
                oldest_key = next(iter(_client_cache))
                del _client_cache[oldest_key]
                logger.debug("Cleared oldest LLM client cache entry")

            _client_cache[cache_key] = OpenAI(
                api_key=get_config_value(config, "llm.api_key"),
                base_url=get_config_value(config, "llm.base_url"),
                timeout=float(get_config_value(config, "llm.timeout", 60)),
            )
        return _client_cache[cache_key]


def _get_llm_request_semaphore(config: dict[str, Any]) -> threading.BoundedSemaphore:
    max_concurrent = max(
        1,
        int(get_config_value(config, "llm.max_concurrent_requests", 2)),
    )
    with _llm_request_semaphore_lock:
        semaphore = _llm_request_semaphores.get(max_concurrent)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(max_concurrent)
            _llm_request_semaphores[max_concurrent] = semaphore
        return semaphore


def _header_value(headers: Any, name: str) -> str | None:
    if headers is None:
        return None
    try:
        value = headers.get(name)
    except Exception:
        value = None
    if value:
        return str(value)
    lowered = name.lower()
    try:
        for key, item in headers.items():
            if str(key).lower() == lowered and item:
                return str(item)
    except Exception:
        return None
    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    raw = _header_value(headers, "retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _status_code(exc: Exception) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    return None


def _is_token_param_error(exc: Exception) -> bool:
    if _status_code(exc) not in {None, 400}:
        return False
    text = str(exc).lower()
    return "max_tokens" in text or "max_completion_tokens" in text


def _is_temperature_error(exc: Exception) -> bool:
    if _status_code(exc) not in {None, 400}:
        return False
    return "temperature" in str(exc).lower()


def _is_response_format_error(exc: Exception) -> bool:
    if _status_code(exc) not in {None, 400}:
        return False
    return "response_format" in str(exc).lower()


def _translate_status_error(exc: APIStatusError) -> LLMCallError:
    status = _status_code(exc)
    if status in {400, 401, 403}:
        return LLMCallError(str(exc), status_code=status, retryable=False)
    if status == 429:
        return LLMCallError(
            str(exc),
            status_code=429,
            retryable=True,
            retry_after=_retry_after_seconds(exc),
        )
    return LLMCallError(str(exc), status_code=status, retryable=True)


def _create_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[ChatCompletionMessageParam],
    max_tokens: int,
    temperature: float | None,
    response_format: dict[str, Any] | None,
) -> Any:
    token_fields = ("max_tokens", "max_completion_tokens")
    temperatures: list[float | None] = [temperature]
    if temperature is not None:
        temperatures.append(None)
    formats: list[dict[str, Any] | None] = [response_format, None] if response_format else [None]
    last_error: Exception | None = None
    for token_field in token_fields:
        for temp in temperatures:
            for fmt in formats:
                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    token_field: max_tokens,
                }
                if temp is not None:
                    kwargs["temperature"] = temp
                if fmt is not None:
                    kwargs["response_format"] = fmt
                try:
                    return client.chat.completions.create(**kwargs)
                except APIStatusError as exc:
                    last_error = exc
                    status = _status_code(exc)
                    if status in {401, 403} or (status == 400 and not (
                        _is_token_param_error(exc)
                        or _is_temperature_error(exc)
                        or _is_response_format_error(exc)
                    )):
                        raise _translate_status_error(exc) from exc
                    if token_field == "max_tokens" and _is_token_param_error(exc):
                        break
                    if temp is not None and _is_temperature_error(exc):
                        continue
                    if fmt is not None and _is_response_format_error(exc):
                        continue
                    if status == 429:
                        raise _translate_status_error(exc) from exc
                    raise _translate_status_error(exc) from exc
            else:
                continue
            break
    if isinstance(last_error, APIStatusError):
        raise _translate_status_error(last_error) from last_error
    if last_error:
        raise last_error
    raise LLMCallError("LLM request failed before a response was received")


def _call_llm_api(
    config: dict[str, Any],
    messages: list[ChatCompletionMessageParam],
    max_tokens: int = 4096,
    *,
    role: str = "summarize",
    temperature: float | None = 0,
    response_format: dict[str, Any] | None = None,
) -> str:
    client = _get_client(config)
    model = resolve_model(config, role)
    semaphore = _get_llm_request_semaphore(config)
    try:
        with semaphore:
            response = _create_completion(
                client,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format=response_format,
            )
    except LLMCallError:
        raise
    except APIStatusError as exc:
        raise _translate_status_error(exc) from exc
    except (APIConnectionError, APITimeoutError, TimeoutError, ConnectionError) as exc:
        raise LLMCallError(str(exc), retryable=True) from exc

    usage = getattr(response, "usage", None)
    _record_usage(
        int(getattr(usage, "prompt_tokens", 0) or 0),
        int(getattr(usage, "completion_tokens", 0) or 0),
    )
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise LLMCallError("LLM returned no choices", retryable=True)
    text = message_text(choices[0].message)
    if not text:
        raise LLMCallError("LLM returned an empty response", retryable=True)
    return text


def _call_llm_api_with_retry(
    config: dict[str, Any],
    messages: list[ChatCompletionMessageParam],
    max_tokens: int = 4096,
    *,
    role: str = "summarize",
    temperature: float | None = 0,
    response_format: dict[str, Any] | None = None,
) -> str:
    return retry_call(
        lambda: _call_llm_api(
            config,
            messages,
            max_tokens,
            role=role,
            temperature=temperature,
            response_format=response_format,
        ),
        max_retries=int(get_config_value(config, "llm.max_retries", 3)),
        base_delay=float(get_config_value(config, "llm.base_delay", 1.0)),
        max_delay=120.0,
        exceptions=RETRY_EXCEPTIONS + (LLMCallError,),
    )


def _parse_json_list(text: str, max_items: int = 20) -> list[str]:
    match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if match:
        items = json.loads(match.group(0))
        return [str(k).strip() for k in items if k][:max_items]
    return []


_CODE_FENCE_RE = re.compile(r"```(?:json|JSON)?[ \t]*\r?\n?([\s\S]*?)```", re.IGNORECASE)


def _strip_code_fences(content: str) -> str:
    """Drop markdown fences wherever they appear, keeping the fenced text."""
    content = content.strip()
    if "```" not in content:
        return content
    stripped = _CODE_FENCE_RE.sub(lambda match: match.group(1), content)
    stripped = re.sub(r"```(?:json|JSON)?", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _prepare_json_text(content: str) -> str:
    return _strip_code_fences(_remove_think_tags(content or ""))


def _scan_balanced_json(content: str, start: int) -> str | None:
    opener = content[start]
    if opener not in "{[":
        return None
    closers = {"{": "}", "[": "]"}
    stack = [opener]
    in_string = False
    escape = False
    for index in range(start + 1, len(content)):
        char = content[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char in "{[":
            stack.append(char)
            continue
        if char in "}]":
            if not stack or closers[stack[-1]] != char:
                return None
            stack.pop()
            if not stack:
                return content[start : index + 1]
    return None


def _iter_json_values(content: str):
    prepared = _prepare_json_text(content)
    if not prepared:
        return
    try:
        whole = json.loads(prepared)
    except json.JSONDecodeError:
        whole = None
    if isinstance(whole, (dict, list)):
        yield whole
        return
    for index, char in enumerate(prepared):
        if char not in "{[":
            continue
        candidate = _scan_balanced_json(prepared, index)
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            yield value


def _extract_json_value(content: str) -> dict[str, Any] | list[Any] | None:
    for value in _iter_json_values(content):
        return value
    return None


def _extract_json_object(content: str) -> dict[str, Any] | None:
    for value in _iter_json_values(content):
        if isinstance(value, dict):
            return value
    return None


def _clip(value: str | None, limit: int) -> str:
    text = re.sub(r"\s+", " ", (value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _language(config: dict[str, Any]) -> str:
    return str(get_config_value(config, "llm.language", "Chinese"))


def _string_list(value: Any, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
        if len(result) >= limit:
            break
    return result


def _paper_id(paper: Any, index: int) -> str:
    url = str(getattr(paper, "url", "") or "")
    return url or f"paper-{index + 1}"


def check_connectivity(config: dict[str, Any]) -> tuple[bool, str]:
    """Confirm the OpenAI-compatible endpoint without generating text."""
    api_key = str(get_config_value(config, "llm.api_key", "") or "")
    base_url = str(get_config_value(config, "llm.base_url", "") or "").rstrip("/")
    if not api_key:
        return False, "未配置 LLM API key，已跳过生成"
    if not base_url:
        return False, "未配置 LLM base_url，已跳过生成"
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            status = getattr(response, "status", 200)
            if status != 200:
                return False, f"LLM 连通性检查失败：HTTP {status}"
            return True, ""
    except urllib.error.HTTPError as exc:
        if exc.code in {401, 403}:
            return False, f"LLM 连通性检查失败：HTTP {exc.code}，密钥被拒绝"
        return False, f"LLM 连通性检查失败：HTTP {exc.code}"
    except Exception as exc:
        return False, f"LLM 连通性检查失败：{exc}"


def test_connection(config: dict[str, Any]) -> bool:
    ok, message = check_connectivity(config)
    if not ok:
        logger.error(message or "LLM connection test failed")
    return ok


def _call_json(
    config: dict[str, Any],
    *,
    role: str,
    system: str,
    user: str,
    max_tokens: int,
) -> dict[str, Any] | None:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        content = _call_llm_api_with_retry(
            config,
            messages,
            max_tokens=max_tokens,
            role=role,
            temperature=0,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        note_llm_error(f"{role} 调用失败：{exc}")
        logger.warning(f"LLM {role} call failed: {exc}")
        return None
    parsed = _extract_json_value(content)
    # Some models emit a bare array instead of {"items": [...]}.
    if isinstance(parsed, list):
        parsed = {"items": parsed}
    if not isinstance(parsed, dict):
        note_llm_error(f"{role} 返回的 JSON 无法解析")
        logger.warning(f"Failed to parse {role} JSON. Raw response: {content[:500]!r}")
        return None
    return parsed


def build_interest_profile(
    papers: list[dict[str, str]],
    feedback: list[dict[str, str]],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    language = _language(config)
    lines = []
    for paper in papers[:80]:
        lines.append(
            f"- { _clip(paper.get('title'), 180) }: {_clip(paper.get('abstract'), 220)}"
        )
    positive = [
        item.get("title", "")
        for item in feedback
        if item.get("vote") == "relevant" and item.get("title")
    ][:12]
    negative = [
        item.get("title", "")
        for item in feedback
        if item.get("vote") == "irrelevant" and item.get("title")
    ][:12]
    prompt = (
        "根据下面的论文库和阅读反馈，生成结构化兴趣画像。\n"
        "canonical_terms 必须是英文规范术语，适合放进 arXiv 的 abs:/ti: 查询，"
        "合并同义词，不要堆近义说法。\n"
        f"summary、topics、methods、not_interested、representative_papers 用{language}。\n\n"
        "论文库：\n"
        + "\n".join(lines)
        + "\n\n标为相关的反馈：\n"
        + ("\n".join(f"- {title}" for title in positive) or "无")
        + "\n\n标为不相关的反馈：\n"
        + ("\n".join(f"- {title}" for title in negative) or "无")
        + "\n\n只返回 JSON："
        '{"summary":"","topics":[],"methods":[],"not_interested":[],'
        '"representative_papers":[],"canonical_terms":[]}'
    )
    parsed = _call_json(
        config,
        role="judge",
        system="你维护一位研究者的兴趣画像，只返回 JSON。",
        user=prompt,
        max_tokens=1800,
    )
    if not parsed:
        return None
    profile = {
        "summary": _clip(str(parsed.get("summary") or ""), 400),
        "topics": _string_list(parsed.get("topics"), 8),
        "methods": _string_list(parsed.get("methods"), 8),
        "not_interested": _string_list(parsed.get("not_interested"), 8),
        "representative_papers": _string_list(parsed.get("representative_papers"), 6),
        "canonical_terms": _string_list(parsed.get("canonical_terms"), 12),
    }
    if not profile["canonical_terms"]:
        profile["canonical_terms"] = _string_list(
            profile["topics"] + profile["methods"], 12
        )
    if not profile["topics"] and not profile["canonical_terms"]:
        note_llm_error("兴趣画像缺少主题和检索术语")
        return None
    return profile


def judge_papers(
    papers: list[Any],
    profile: dict[str, Any] | None,
    config: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if not papers:
        return []
    language = _language(config)
    profile = profile or {}
    profile_text = json.dumps(
        {
            "summary": profile.get("summary") or "",
            "topics": profile.get("topics") or [],
            "methods": profile.get("methods") or [],
            "not_interested": profile.get("not_interested") or [],
        },
        ensure_ascii=False,
    )
    blocks = []
    for index, paper in enumerate(papers):
        blocks.append(
            "\n".join(
                [
                    f"id: {_paper_id(paper, index)}",
                    f"title: {_clip(getattr(paper, 'title', ''), 240)}",
                    f"abstract: {_clip(getattr(paper, 'abstract', ''), 900)}",
                ]
            )
        )
    prompt = (
        "你是论文推荐裁判。兴趣画像如下：\n"
        f"{profile_text}\n\n"
        "只评估下面这些候选。relevance 是 1 到 5 的整数绝对分，不要做当天归一化。\n"
        f"reason 用{language}，点名最接近的库内方向，或说明为什么不相关。\n"
        "keep 为 true 表示值得进入当天短名单。\n\n"
        + "\n\n".join(blocks)
        + '\n\n只返回 JSON：{"items":[{"id":"","relevance":1,"reason":"","keep":false}]}'
    )
    parsed = _call_json(
        config,
        role="judge",
        system=f"你只返回 JSON。reason 使用{language}。",
        user=prompt,
        max_tokens=2400,
    )
    if not parsed:
        return None
    items = parsed.get("items")
    if not isinstance(items, list):
        note_llm_error("裁判结果缺少 items 数组")
        return None
    normalized = []
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            relevance = int(round(float(item.get("relevance"))))
        except (TypeError, ValueError):
            continue
        relevance = max(1, min(5, relevance))
        keep = item.get("keep")
        if isinstance(keep, str):
            keep = keep.strip().lower() in {"1", "true", "yes", "keep"}
        if keep is None:
            keep = relevance >= 3
        if relevance < 3:
            keep = False
        normalized.append(
            {
                "id": str(item.get("id") or "").strip(),
                "relevance": relevance,
                "reason": _clip(str(item.get("reason") or ""), 240),
                "keep": bool(keep),
            }
        )
    if not normalized:
        note_llm_error("裁判结果没有可用条目")
        return None
    return normalized


def _tldr_prompt(
    papers: list[Any], profile: dict[str, Any], language: str
) -> str:
    blocks = []
    for index, paper in enumerate(papers):
        blocks.append(
            "\n".join(
                [
                    f"id: {_paper_id(paper, index)}",
                    f"title: {_clip(getattr(paper, 'title', ''), 240)}",
                    f"abstract: {_clip(getattr(paper, 'abstract', ''), 1000)}",
                ]
            )
        )
    interest = _clip(str(profile.get("summary") or ""), 300) or "、".join(
        _string_list(profile.get("topics"), 6)
    )
    return (
        f"为下面每一篇论文写结构化中文短评。语言：{language}。\n"
        "每篇都要有 tldr、method、evidence、why_for_me，各用一句，大约 40 到 80 个汉字。\n"
        f"读者兴趣：{interest or '未提供'}\n\n"
        + "\n\n".join(blocks)
        + '\n\n只返回 JSON：{"items":[{"id":"","tldr":"","method":"","evidence":"","why_for_me":""}]}'
    )


def _collect_tldr_items(parsed: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    if not isinstance(parsed, dict):
        return {}
    items = parsed.get("items")
    if not isinstance(items, list):
        if parsed.get("tldr"):
            items = [parsed]
        else:
            return {}
    result: dict[str, dict[str, str]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        item_id = str(item.get("id") or "").strip()
        tldr = _clip(str(item.get("tldr") or ""), 220)
        if not tldr:
            continue
        if not item_id:
            item_id = f"__anon_{len(result)}"
        result[item_id] = {
            "tldr": tldr,
            "method": _clip(str(item.get("method") or ""), 220),
            "evidence": _clip(str(item.get("evidence") or ""), 220),
            "why_for_me": _clip(str(item.get("why_for_me") or ""), 220),
        }
    return result


def _tldr_max_tokens(paper_count: int) -> int:
    # A 10-paper batch at 2800 tokens stopped inside the reasoning trace.
    return min(12000, 1200 + 700 * max(1, paper_count))


def _request_tldr_items(
    papers: list[Any], profile: dict[str, Any], config: dict[str, Any]
) -> dict[str, dict[str, str]]:
    language = _language(config)
    parsed = _call_json(
        config,
        role="summarize",
        system=f"你为研究者写短评，只返回 JSON，使用{language}。",
        user=_tldr_prompt(papers, profile, language),
        max_tokens=_tldr_max_tokens(len(papers)),
    )
    return _collect_tldr_items(parsed)


def generate_tldrs_batch(
    papers: list[Any],
    profile: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, dict[str, str]] | None:
    if not papers:
        return {}
    profile = profile or {}
    result = _request_tldr_items(papers, profile, config)
    if len(papers) > 1:
        missing = [
            (index, paper)
            for index, paper in enumerate(papers)
            if _paper_id(paper, index) not in result
        ]
        if missing:
            logger.info(f"Retrying {len(missing)} TLDR item(s) individually")
            for index, paper in missing:
                paper_id = _paper_id(paper, index)
                one = _request_tldr_items([paper], profile, config)
                payload = one.get(paper_id) or one.get(_paper_id(paper, 0))
                if payload is None and len(one) == 1:
                    payload = next(iter(one.values()))
                if payload and payload.get("tldr"):
                    result[paper_id] = payload
    if not result:
        note_llm_error("TLDR 结果没有可用条目")
        return None
    return result


def generate_tldr(paper: Any, config: dict[str, Any]) -> str:
    language = _language(config)
    has_content = paper.title or paper.abstract
    if not has_content:
        logger.warning(f"No summary text available for {paper.url}")
        return "暂不提供摘要"

    batch = generate_tldrs_batch([paper], None, config)
    paper_id = _paper_id(paper, 0)
    if batch and paper_id in batch:
        return batch[paper_id]["tldr"]
    if batch:
        first = next(iter(batch.values()), None)
        if first:
            return first["tldr"]

    prompt = (
        f"Given the following paper, write a one-sentence TLDR summary in {language}.\n\n"
    )
    if paper.title:
        prompt += f"Title: {paper.title}\n\n"
    if paper.abstract:
        prompt += f"Abstract: {paper.abstract}\n\n"
    prompt += (
        "Return ONLY a valid JSON object with exactly this format:\n"
        '{"tldr": "your one-sentence summary here"}\n\n'
        "Do not include any text outside the JSON object."
    )
    parsed = _call_json(
        config,
        role="summarize",
        system=(
            f"You are an assistant who summarizes scientific papers. "
            f"Return only valid JSON with a single 'tldr' key in {language}."
        ),
        user=prompt,
        max_tokens=500,
    )
    if not parsed:
        return ""
    tldr = str(parsed.get("tldr") or "").strip()
    if not tldr:
        note_llm_error(f"TLDR 为空：{getattr(paper, 'url', '')}")
    return tldr


def extract_keywords_from_paper(
    title: str, abstract: str, config: dict[str, Any], max_keywords: int = 5
) -> list[str]:
    prompt = f"""Given the following paper title and abstract, extract the most important research keywords/phrases.

Title: {title}

Abstract: {abstract}

Focus on:
1. Specific technical terms and methods
2. Research domains and applications
3. Key concepts

Return {max_keywords} keywords as a JSON list of strings, ordered by importance.
Only return JSON."""

    parsed = _call_json(
        config,
        role="extract",
        system="You extract research keywords. Return only a JSON object with a keywords array.",
        user=prompt + '\n{"keywords":["keyword1","keyword2"]}',
        max_tokens=600,
    )
    if parsed and isinstance(parsed.get("keywords"), list):
        return [str(item).strip() for item in parsed["keywords"] if str(item).strip()][:max_keywords]
    try:
        text = _call_llm_api_with_retry(
            config,
            [
                {
                    "role": "system",
                    "content": "You are a research assistant specialized in extracting key topics from papers. Return only a Python list of keywords.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=600,
            role="extract",
            temperature=0,
        )
        return _parse_json_list(text, max_keywords)
    except Exception as exc:
        note_llm_error(f"关键词抽取失败：{exc}")
        logger.warning(f"Failed to extract keywords from paper '{title}': {exc}")
        return []


def extract_keywords_from_corpus(
    papers_text: list[str], config: dict[str, Any], max_keywords: int = 20
) -> list[str]:
    combined_text = "\n\n---\n\n".join(papers_text[:50])
    prompt = f"""Given the following collection of research paper titles and abstracts, extract the most important and representative research keywords/phrases.

Focus on:
1. Specific technical terms and methods
2. Research domains and applications
3. Key concepts that appear across multiple papers

Return {max_keywords} keywords as a JSON list of strings, ordered by importance.

Papers:
{combined_text[:10000]}
"""
    parsed = _call_json(
        config,
        role="extract",
        system="You extract research topics. Return only a JSON object with a keywords array.",
        user=prompt + '\n{"keywords":["keyword1"]}',
        max_tokens=800,
    )
    if not parsed:
        return []
    return [str(item).strip() for item in parsed.get("keywords") or [] if str(item).strip()][
        :max_keywords
    ]


def answer_question(
    question: str,
    contexts: list[dict[str, str]],
    config: dict[str, Any],
) -> str:
    language = _language(config)
    blocks = []
    for index, item in enumerate(contexts, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] {item.get('title') or ''}",
                    f"url: {item.get('url') or ''}",
                    f"date: {item.get('date') or ''}",
                    f"score: {item.get('score') or ''}",
                    f"tldr: {_clip(item.get('tldr'), 240)}",
                    f"reason: {_clip(item.get('reason'), 240)}",
                    f"abstract: {_clip(item.get('abstract'), 700)}",
                ]
            )
        )
    prompt = (
        f"只根据下面的本地资料回答问题，使用{language}。\n"
        "不要编造资料里没有的论文。回答里用 [编号] 指出依据。\n"
        "如果资料不够，直接说不够。\n\n"
        f"问题：{question}\n\n资料：\n\n" + "\n\n".join(blocks)
    )
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": f"你是本地论文库的只读助手，只依据给定资料用{language}回答。",
        },
        {"role": "user", "content": prompt},
    ]
    try:
        return _call_llm_api_with_retry(
            config,
            messages,
            max_tokens=900,
            role="qa",
            temperature=0,
        )
    except Exception as exc:
        note_llm_error(f"问答失败：{exc}")
        logger.warning(f"QA call failed: {exc}")
        return ""
