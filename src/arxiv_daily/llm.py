import re
import json
import threading
from typing import Any
from loguru import logger
from openai import OpenAI, APIError as OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from .config import get_config_value
from .utils import retry_call


def _remove_think_tags(content: str) -> str:
    if not content:
        return content
    content = re.sub(r"<think.*?>.*?</think\s*>", "", content, flags=re.DOTALL)
    content = re.sub(r"<\|think\|>.*?<\|/think\|>", "", content, flags=re.DOTALL)
    content = content.strip()
    return content


_client_cache: dict[tuple[Any, Any], OpenAI] = {}
_client_cache_lock = threading.Lock()
_llm_request_semaphore_lock = threading.Lock()
_llm_request_semaphores: dict[int, threading.BoundedSemaphore] = {}
_MAX_CACHE_SIZE = 10

RETRY_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OpenAIError,
)


def _get_client(config: dict[str, Any]) -> OpenAI:
    global _client_cache
    cache_key = (
        get_config_value(config, "llm.api_key"),
        get_config_value(config, "llm.base_url"),
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
                timeout=float(get_config_value(config, "llm.timeout")),
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


def _call_llm_api(
    config: dict[str, Any], messages: list[ChatCompletionMessageParam], max_tokens: int = 4096
) -> str:
    client = _get_client(config)
    semaphore = _get_llm_request_semaphore(config)
    with semaphore:
        response = client.chat.completions.create(
            messages=messages,
            model=get_config_value(config, "llm.model"),
            max_tokens=max_tokens,
        )
    return _remove_think_tags(response.choices[0].message.content or "")


def _call_llm_api_with_retry(
    config: dict[str, Any], messages: list[ChatCompletionMessageParam], max_tokens: int = 4096
) -> str:
    return retry_call(
        lambda: _call_llm_api(config, messages, max_tokens),
        max_retries=int(get_config_value(config, "llm.max_retries")),
        base_delay=float(get_config_value(config, "llm.base_delay")),
        exceptions=RETRY_EXCEPTIONS,
    )


def _parse_json_list(text: str, max_items: int = 20) -> list[str]:
    match = re.search(r"\[.*?\]", text, flags=re.DOTALL)
    if match:
        items = json.loads(match.group(0))
        return [str(k).strip() for k in items if k][:max_items]
    return []


def _strip_code_fences(content: str) -> str:
    content = content.strip()
    if not content.startswith("```"):
        return content

    lines = content.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content


def _extract_json_object(content: str) -> dict[str, Any] | None:
    if not content:
        return None

    content = _strip_code_fences(content)

    try:
        result = json.loads(content)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    for start in range(len(content)):
        if content[start] != "{":
            continue

        depth = 0
        in_string = False
        escape = False

        for end in range(start, len(content)):
            char = content[end]

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
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = content[start : end + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        break

    return None


def generate_tldr(paper: Any, config: dict[str, Any]) -> str:
    language = get_config_value(config, "llm.language")

    has_content = paper.title or paper.abstract
    if not has_content:
        logger.warning(f"No summary text available for {paper.url}")
        return "暂不提供摘要"

    prompt = f"Given the following paper, write a one-sentence TLDR summary in {language}.\n\n"
    if paper.title:
        prompt += f"Title: {paper.title}\n\n"
    if paper.abstract:
        prompt += f"Abstract: {paper.abstract}\n\n"
    prompt += (
        "Return ONLY a valid JSON object with exactly this format:\n"
        '{"tldr": "your one-sentence summary here"}\n\n'
        "Do not include any text outside the JSON object."
    )

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                f"You are an assistant who summarizes scientific papers. "
                f"Return only valid JSON with a single 'tldr' key in {language}."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        content = _call_llm_api_with_retry(config, messages, max_tokens=256)
        result = _extract_json_object(content)
        if result:
            return result.get("tldr", "")
        logger.warning(f"Failed to parse TLDR JSON from response for {paper.url}")
        return ""
    except Exception as e:
        logger.warning(f"Failed to generate TLDR for {paper.url}: {e}")
        return ""
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

Return {max_keywords} keywords as a Python list, ordered by importance.
Format: ["keyword1", "keyword2", ...]

Only return the list, no explanation."""

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a research assistant specialized in extracting key topics from papers. Return only a Python list of keywords.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        text = _call_llm_api_with_retry(config, messages, max_tokens=1024)
        return _parse_json_list(text, max_keywords)
    except Exception as e:
        logger.warning(f"Failed to extract keywords from paper '{title}': {e}")
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

Return {max_keywords} keywords/phrases as a Python list, ordered by importance.
Format: ["keyword1", "keyword2", ...]

Papers:
{combined_text[:10000]}

Only return the list, no explanation."""

    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": "You are a research assistant specialized in extracting key research topics from scientific papers. Return only a Python list of keywords.",
        },
        {"role": "user", "content": prompt},
    ]

    try:
        text = _call_llm_api_with_retry(config, messages, max_tokens=2048)
        return _parse_json_list(text, max_keywords)
    except Exception as e:
        logger.warning(f"Failed to extract keywords: {e}")
        return []


def test_connection(config: dict[str, Any]) -> bool:
    try:
        _call_llm_api(config, [{"role": "user", "content": "Hi"}], max_tokens=10)
        return True
    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return False
