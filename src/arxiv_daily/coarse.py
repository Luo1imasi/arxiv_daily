"""LLM coarse screening. Scores are 0-10; a failed pass falls back to BM25."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from .config import get_config_value
from .lexical import diverse_indices, order_by_bm25
from .llm import _call_json, _clip, note_llm_error

JsonCaller = Callable[..., dict[str, Any] | None]


@dataclass
class CoarseOutcome:
    scores: dict[str, float] | None
    requests: int
    failed_batches: int


def coarse_pool_size(config: dict[str, Any]) -> int:
    max_paper_num = max(1, int(get_config_value(config, "executor.max_paper_num", 10)))
    judge_pool = int(get_config_value(config, "executor.judge_pool_size", 24))
    return min(30, max(20, judge_pool, min(max_paper_num, 30)))


def coarse_batch_size(config: dict[str, Any]) -> int:
    size = int(get_config_value(config, "llm.coarse_batch_size", 40))
    return min(50, max(40, size))


def coarse_concurrency(config: dict[str, Any]) -> int:
    requested = int(get_config_value(config, "llm.coarse_concurrency", 2))
    global_cap = int(get_config_value(config, "llm.max_concurrent_requests", 2))
    return min(3, max(1, requested), max(1, global_cap))


def _abstract_chars(config: dict[str, Any]) -> int:
    return max(80, int(get_config_value(config, "llm.coarse_abstract_chars", 280)))


def _max_tokens(config: dict[str, Any]) -> int:
    return max(256, int(get_config_value(config, "llm.coarse_max_tokens", 2200)))


def _profile_text(profile: dict[str, Any] | None) -> str:
    profile = profile or {}
    payload = {
        "summary": _clip(str(profile.get("summary") or ""), 300),
        "topics": list(profile.get("topics") or [])[:8],
        "methods": list(profile.get("methods") or [])[:8],
        "not_interested": list(profile.get("not_interested") or [])[:8],
    }
    return json.dumps(payload, ensure_ascii=False)


def _paper_block(local_id: str, paper: Any, abstract_chars: int) -> str:
    title = _clip(str(getattr(paper, "title", "") or ""), 180)
    abstract = _clip(str(getattr(paper, "abstract", "") or ""), abstract_chars)
    return f"{local_id}. {title}\n{abstract}"


def _parse_score(value: Any) -> float | None:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if score != score:
        return None
    return max(0.0, min(10.0, score))


def _scores_from_response(
    parsed: dict[str, Any] | None,
    id_to_url: dict[str, str],
) -> dict[str, float]:
    if not isinstance(parsed, dict):
        return {}
    items = parsed.get("items")
    if not isinstance(items, list):
        return {}
    scores: dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        local_id = str(item.get("id") or "").strip()
        url = id_to_url.get(local_id)
        score = _parse_score(item.get("score"))
        if url and score is not None:
            scores[url] = score
    return scores


def _score_batch(
    papers: list[Any],
    profile: dict[str, Any] | None,
    config: dict[str, Any],
    caller: JsonCaller,
) -> dict[str, float]:
    abstract_chars = _abstract_chars(config)
    id_to_url: dict[str, str] = {}
    blocks: list[str] = []
    for index, paper in enumerate(papers, start=1):
        local_id = str(index)
        url = str(getattr(paper, "url", "") or "")
        if not url:
            continue
        id_to_url[local_id] = url
        blocks.append(_paper_block(local_id, paper, abstract_chars))
    if not blocks:
        return {}
    prompt = (
        "对照兴趣画像，给每篇论文一个 0 到 10 的整数分。"
        "10 分表示和画像高度相关，0 分表示无关或属于不感兴趣的方向。"
        "不要写理由。只返回 JSON："
        '{"items":[{"id":"1","score":0}]}\n\n'
        f"兴趣画像：\n{_profile_text(profile)}\n\n"
        "论文：\n\n" + "\n\n".join(blocks)
    )
    parsed = caller(
        config,
        role="coarse",
        system="你是论文粗筛员。只返回 JSON，不要解释。",
        user=prompt,
        max_tokens=_max_tokens(config),
    )
    return _scores_from_response(parsed, id_to_url)


def score_coarse_papers(
    papers: list[Any],
    profile: dict[str, Any] | None,
    config: dict[str, Any],
    *,
    caller: JsonCaller | None = None,
) -> CoarseOutcome:
    if not papers:
        return CoarseOutcome(scores={}, requests=0, failed_batches=0)
    caller = caller or _call_json
    size = coarse_batch_size(config)
    batches = [papers[start : start + size] for start in range(0, len(papers), size)]
    workers = min(coarse_concurrency(config), len(batches))
    scores: dict[str, float] = {}
    failed = 0
    logger.info(
        f"Coarse-screening {len(papers)} papers in {len(batches)} batches "
        f"(size={size}, concurrency={workers})"
    )
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="coarse") as pool:
        futures = [
            pool.submit(_score_batch, batch, profile, config, caller) for batch in batches
        ]
        for future in as_completed(futures):
            try:
                batch_scores = future.result()
            except Exception as exc:
                failed += 1
                note_llm_error(f"LLM 粗筛批次失败：{exc}")
                logger.warning(f"Coarse batch failed: {exc}")
                continue
            if not batch_scores:
                failed += 1
                note_llm_error("LLM 粗筛批次没有返回可用分数")
                continue
            scores.update(batch_scores)
    if not scores or failed * 2 >= len(batches):
        note_llm_error("LLM 粗筛失败，已退回 BM25 排序")
        logger.warning(
            f"Coarse screening failed ({failed}/{len(batches)} batches, "
            f"{len(scores)} scores); falling back to BM25"
        )
        return CoarseOutcome(scores=None, requests=len(batches), failed_batches=failed)
    if failed:
        note_llm_error(f"LLM 粗筛有 {failed} 个批次失败，已使用成功批次的分数")
    logger.info(f"Coarse screening scored {len(scores)} of {len(papers)} papers")
    return CoarseOutcome(scores=scores, requests=len(batches), failed_batches=failed)


def diverse_shortlist(
    papers: list[Any],
    scores: dict[str, float],
    config: dict[str, Any],
) -> list[Any]:
    if not papers:
        return []
    limit = coarse_pool_size(config)
    lam = float(get_config_value(config, "reranker.mmr_lambda", 0.78))
    duplicate_jaccard = float(get_config_value(config, "llm.coarse_duplicate_jaccard", 0.8))
    texts = [str(getattr(paper, "title", "") or "") for paper in papers]
    raw_scores = [float(scores.get(str(getattr(paper, "url", "") or ""), 0.0)) for paper in papers]
    chosen_indexes = diverse_indices(
        texts,
        raw_scores,
        limit,
        lam=lam,
        duplicate_jaccard=duplicate_jaccard,
    )
    chosen: list[Any] = []
    for index in chosen_indexes:
        papers[index].score = raw_scores[index]
        chosen.append(papers[index])
    return chosen


__all__ = [
    "CoarseOutcome",
    "coarse_batch_size",
    "coarse_concurrency",
    "coarse_pool_size",
    "diverse_shortlist",
    "order_by_bm25",
    "score_coarse_papers",
]
