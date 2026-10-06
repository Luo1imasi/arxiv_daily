import os
import asyncio
import copy
import hashlib
import json
import threading
import time
import concurrent.futures
from datetime import date, datetime
from loguru import logger

from collections.abc import Sequence
from typing import Any

from .protocol import CorpusPaper, Paper
from .webdav import fetch_corpus
from .retriever import get_retriever_cls
from .reranker import get_reranker_cls
from .llm import (
    PROMPT_VERSIONS,
    build_interest_profile,
    check_connectivity,
    generate_tldrs_batch,
    get_llm_usage,
    judge_papers,
    make_llm_cache_key,
    note_llm_error,
    note_llm_warning,
    reset_llm_usage,
    resolve_model,
)
from . import database as db
from .config import get_config_value
from .utils import make_content_key
from .business_date import (
    business_date_range_between as _business_date_range_between,
    business_date_range_until as _business_date_range_until,
    config_for_business_date as _config_for_business_date,
    get_business_date_string as _get_business_date,
    normalize_business_date as _normalize_business_date,
)

_executor_pool: concurrent.futures.ThreadPoolExecutor | None = None
_executor_pool_lock = threading.Lock()


def get_executor_pool(config: dict[str, Any] | None = None) -> concurrent.futures.ThreadPoolExecutor:
    global _executor_pool
    with _executor_pool_lock:
        if _executor_pool is None:
            workers = 8
            if config is not None:
                workers = int(get_config_value(config, "executor.thread_pool_workers", 8))
            _executor_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(2, workers),
                thread_name_prefix="arxiv-daily",
            )
        return _executor_pool


def _fallback_tldr(p: Paper):
    # Avoid presenting raw abstracts as TLDR text in the UI.
    p.tldr = None
    p.method = None
    p.evidence = None
    p.why_for_me = None


def _get_llm_cache_key(config: dict[str, Any], role: str = "summarize") -> str:
    return make_llm_cache_key(config, role)


def _default_llm_metrics(*, enabled: bool, target_count: int = 0) -> dict[str, object]:
    return {
        "llm_enabled": enabled,
        "llm_error_count": 0,
        "llm_target_count": target_count,
        "llm_cache_hits": 0,
        "llm_request_count": 0,
        "tldr_cache_hits": 0,
        "tldr_request_count": 0,
        "judge_cache_hits": 0,
        "judge_request_count": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_warning": "",
    }


def _merge_llm_usage(metrics: dict[str, object]) -> None:
    usage = get_llm_usage()
    metrics["llm_error_count"] = usage["llm_error_count"]
    metrics["llm_prompt_tokens"] = usage["llm_prompt_tokens"]
    metrics["llm_completion_tokens"] = usage["llm_completion_tokens"]
    metrics["llm_warning"] = usage["llm_warning"]
    metrics["llm_request_count"] = usage["llm_api_requests"]


def _content_key(paper: Paper) -> str:
    return make_content_key(paper.title, paper.abstract or "")


def _profile_signatures(
    corpus: Sequence[object], feedback: list[dict[str, Any]]
) -> tuple[str, str]:
    corpus_signature = hashlib.sha1(
        "\n".join(
            sorted(
                make_content_key(
                    getattr(paper, "title", "") or "",
                    getattr(paper, "abstract", "") or "",
                )
                for paper in corpus
            )
        ).encode("utf-8")
    ).hexdigest()
    feedback_signature = hashlib.sha1(
        "\n".join(
            sorted(
                f"{row.get('url')}|{row.get('vote')}|{row.get('updated_at') or ''}"
                for row in feedback
            )
        ).encode("utf-8")
    ).hexdigest()
    return corpus_signature, feedback_signature


def _copy_cached_tldr(paper: Paper, cached: dict[str, Any]) -> None:
    paper.tldr = cached.get("tldr")
    paper.method = cached.get("method")
    paper.evidence = cached.get("evidence")
    paper.why_for_me = cached.get("why_for_me")


def _copy_cached_judgment(paper: Paper, cached: dict[str, Any]) -> None:
    relevance = cached.get("judge_relevance")
    paper.judge_relevance = float(relevance) if relevance is not None else None
    paper.judge_reason = cached.get("judge_reason")
    if cached.get("judge_keep") is None:
        paper.judge_keep = None
    else:
        paper.judge_keep = bool(cached.get("judge_keep"))


def _select_judged_papers(papers: list[Paper], max_num: int) -> list[Paper]:
    kept = [paper for paper in papers if paper.judge_keep]
    if len(kept) >= max_num:
        return kept[:max_num]
    rest = [paper for paper in papers if not paper.judge_keep]
    rest.sort(
        key=lambda paper: (
            -(paper.judge_relevance if paper.judge_relevance is not None else -1),
        )
    )
    return (kept + rest)[:max_num]


def _attach_judgments(papers: list[Paper], judgments: list[dict[str, Any]]) -> None:
    by_id = {str(item.get("id") or ""): item for item in judgments}
    missing = 0
    for index, paper in enumerate(papers):
        item = by_id.get(paper.url or "")
        if item is None and len(judgments) == len(papers):
            item = judgments[index]
        if not item:
            missing += 1
            continue
        paper.judge_relevance = float(item["relevance"])
        paper.judge_reason = item.get("reason") or ""
        paper.judge_keep = bool(item.get("keep"))
    if missing:
        note_llm_error(f"{missing} 篇论文没有裁判结果")


async def _ensure_interest_profile(
    config: dict[str, Any], corpus: Sequence[object], *, llm_ready: bool
) -> dict[str, Any] | None:
    feedback = await db.list_feedback()
    corpus_signature, feedback_signature = _profile_signatures(corpus, feedback)
    cached = await db.load_interest_profile()
    model = resolve_model(config, "profile")
    prompt_version = PROMPT_VERSIONS["profile"]
    cached_profile = (cached or {}).get("profile") if cached else None
    if (
        cached
        and cached.get("corpus_signature") == corpus_signature
        and cached.get("feedback_signature") == feedback_signature
        and cached.get("model") == model
        and cached.get("prompt_version") == prompt_version
        and cached_profile
    ):
        logger.info("Using cached interest profile")
        return cached_profile
    if not llm_ready:
        if cached_profile:
            note_llm_warning("兴趣画像未更新，沿用上次缓存")
            return cached_profile
        return None

    briefs = [
        {"title": getattr(paper, "title", "") or "", "abstract": getattr(paper, "abstract", "") or ""}
        for paper in corpus
    ]
    loop = asyncio.get_running_loop()
    profile = await loop.run_in_executor(
        get_executor_pool(config),
        lambda: build_interest_profile(
            briefs,
            [
                {"title": row.get("title") or "", "vote": row.get("vote") or ""}
                for row in feedback
            ],
            config,
        ),
    )
    if not profile:
        if cached_profile:
            note_llm_warning("兴趣画像生成失败，沿用上次缓存")
            return cached_profile
        return None
    await db.save_interest_profile(
        profile,
        corpus_signature=corpus_signature,
        feedback_signature=feedback_signature,
        model=model,
        prompt_version=prompt_version,
    )
    logger.info("Refreshed interest profile")
    return profile


async def _apply_judge(
    papers: list[Paper],
    profile: dict[str, Any] | None,
    config: dict[str, Any],
    metrics: dict[str, object],
    *,
    max_num: int | None = None,
) -> list[Paper]:
    if max_num is None:
        max_num = max(1, int(get_config_value(config, "executor.max_paper_num")))
    if not papers:
        return []
    cache_key = _get_llm_cache_key(config, "judge")
    cached_enrichments = await db.load_candidate_enrichments(
        [paper.url for paper in papers if paper.url]
    )
    pending: list[Paper] = []
    hits = 0
    for paper in papers:
        cached = cached_enrichments.get(paper.url or "")
        if (
            cached
            and cached.get("content_key") == _content_key(paper)
            and cached.get("judge_cache_key") == cache_key
            and cached.get("judge_relevance") is not None
        ):
            _copy_cached_judgment(paper, cached)
            hits += 1
        else:
            pending.append(paper)
    metrics["judge_cache_hits"] = hits
    metrics["judge_request_count"] = 1 if pending else 0
    if pending:
        logger.info(f"Judging {len(pending)} shortlisted papers")
        loop = asyncio.get_running_loop()
        judgments = await loop.run_in_executor(
            get_executor_pool(config),
            lambda: judge_papers(pending, profile, config),
        )
        if not judgments:
            logger.warning("Judge failed; keeping MMR order")
            chosen = papers[:max_num]
        else:
            _attach_judgments(pending, judgments)
            chosen = _select_judged_papers(papers, max_num)
    else:
        chosen = _select_judged_papers(papers, max_num)

    entries = []
    for paper in papers:
        if not paper.url or paper.judge_relevance is None:
            continue
        entries.append(
            {
                "url": paper.url,
                "pdf_url": paper.pdf_url,
                "content_key": _content_key(paper),
                "judge_relevance": paper.judge_relevance,
                "judge_reason": paper.judge_reason,
                "judge_keep": int(bool(paper.judge_keep)),
                "judge_cache_key": cache_key,
            }
        )
    await db.save_candidate_enrichments(entries)
    return chosen


async def _apply_tldr_enrichment(
    papers: list[Paper],
    config: dict[str, Any],
    metrics: dict[str, object],
    profile: dict[str, Any] | None = None,
) -> None:
    llm_config = config.get("llm", {})
    metrics["llm_enabled"] = bool(llm_config.get("api_key"))
    metrics["llm_target_count"] = len(papers)
    if not llm_config.get("api_key"):
        logger.info("No LLM API key configured, skipping TLDR generation")
        for paper in papers:
            _fallback_tldr(paper)
        metrics["llm_enabled"] = False
        return

    logger.info("Generating structured TLDRs for selected papers...")
    llm_cache_key = _get_llm_cache_key(config, "summarize")
    cached_enrichments = await db.load_candidate_enrichments(
        [paper.url for paper in papers if paper.url]
    )
    tldr_cache_hits = 0
    papers_for_tldr = []
    for paper in papers:
        cached = cached_enrichments.get(paper.url or "")
        if (
            cached
            and cached.get("content_key") == _content_key(paper)
            and cached.get("llm_cache_key") == llm_cache_key
            and (cached.get("tldr") or "").strip()
        ):
            _copy_cached_tldr(paper, cached)
            tldr_cache_hits += 1
        else:
            papers_for_tldr.append(paper)

    metrics["llm_cache_hits"] = tldr_cache_hits
    metrics["tldr_cache_hits"] = tldr_cache_hits
    metrics["tldr_request_count"] = 1 if papers_for_tldr else 0
    generated: dict[str, dict[str, str]] = {}
    if papers_for_tldr:
        loop = asyncio.get_running_loop()
        batch = await loop.run_in_executor(
            get_executor_pool(config),
            lambda: generate_tldrs_batch(papers_for_tldr, profile, config),
        )
        generated = batch or {}
        for index, paper in enumerate(papers_for_tldr):
            payload = generated.get(paper.url or "")
            if payload is None and len(generated) == 1 and len(papers_for_tldr) == 1:
                payload = next(iter(generated.values()))
            if not payload or not payload.get("tldr"):
                note_llm_error(f"TLDR 为空或解析失败：{paper.url or paper.title}")
                _fallback_tldr(paper)
                continue
            paper.tldr = payload["tldr"]
            paper.method = payload.get("method") or None
            paper.evidence = payload.get("evidence") or None
            paper.why_for_me = payload.get("why_for_me") or None

    cache_entries = []
    for paper in papers:
        content_key = _content_key(paper)
        cached = cached_enrichments.get(paper.url or "", {})
        entry = {
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "content_key": content_key,
            "tldr": None,
            "method": None,
            "evidence": None,
            "why_for_me": None,
            "llm_cache_key": None,
        }
        if paper.tldr:
            entry.update(
                {
                    "tldr": paper.tldr,
                    "method": paper.method,
                    "evidence": paper.evidence,
                    "why_for_me": paper.why_for_me,
                    "llm_cache_key": llm_cache_key,
                }
            )
        elif (
            cached.get("content_key") == content_key
            and cached.get("llm_cache_key") == llm_cache_key
            and cached.get("tldr")
        ):
            entry.update(
                {
                    "tldr": cached.get("tldr"),
                    "method": cached.get("method"),
                    "evidence": cached.get("evidence"),
                    "why_for_me": cached.get("why_for_me"),
                    "llm_cache_key": cached.get("llm_cache_key"),
                }
            )
        cache_entries.append(entry)

    await db.save_candidate_enrichments(cache_entries)
    _merge_llm_usage(metrics)


async def _filter_seen_papers(
    papers: list[Paper], corpus: Sequence[object], business_date: str
) -> list[Paper]:
    seen_urls = await db.get_seen_paper_urls(before_date=business_date)
    seen_content_keys = await db.get_seen_paper_content_keys(before_date=business_date)
    corpus_content_keys = {
        make_content_key(getattr(paper, "title", ""), getattr(paper, "abstract", "") or "")
        for paper in corpus
        if getattr(paper, "title", None)
    }
    filtered = []
    batch_keys = set()

    for paper in papers:
        content_key = make_content_key(paper.title, paper.abstract or "")
        identity = paper.url or content_key
        if not identity or identity in batch_keys:
            continue
        if paper.url and paper.url in seen_urls:
            continue
        if content_key and content_key in seen_content_keys:
            continue
        if content_key and content_key in corpus_content_keys:
            continue

        batch_keys.add(identity)
        filtered.append(paper)

    return filtered


class Executor:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.last_run_metrics: dict[str, object] = {
            "status": "initialized",
            "sources": [],
        }

    async def run_for_date(self, business_date: str | date, skip_tldr: bool = False) -> list[Paper]:
        scoped = Executor(_config_for_business_date(self.config, _normalize_business_date(business_date)))
        result = await scoped.run(skip_tldr=skip_tldr)
        self.last_run_metrics = dict(scoped.last_run_metrics)
        return result

    async def run_until_date(self, until_date: str | date, skip_tldr: bool = False) -> dict[str, object]:
        dates = _business_date_range_until(self.config, until_date)
        return await self._run_backfill_dates(
            dates,
            start_date=dates[-1],
            end_date=dates[0],
            skip_tldr=skip_tldr,
        )

    async def run_between_dates(
        self, start_date: str | date, end_date: str | date, skip_tldr: bool = False
    ) -> dict[str, object]:
        dates = _business_date_range_between(self.config, start_date, end_date)
        return await self._run_backfill_dates(
            dates,
            start_date=dates[0],
            end_date=dates[-1],
            skip_tldr=skip_tldr,
        )

    async def _run_backfill_dates(
        self,
        dates: list[str],
        *,
        start_date: str,
        end_date: str,
        skip_tldr: bool = False,
    ) -> dict[str, object]:
        summary: list[dict[str, object]] = []
        total_recommendations = 0
        self.last_run_metrics = {
            "status": "running",
            "mode": "backfill",
            "start_date": start_date,
            "end_date": end_date,
            "until_date": end_date,
            "date_count": len(dates),
            "dates": [],
        }

        for business_date in dates:
            logger.info(f"Backfill generating recommendations for {business_date}")
            papers = await self.run_for_date(business_date, skip_tldr=skip_tldr)
            day_metrics = dict(self.last_run_metrics)
            day_summary = {
                "date": business_date,
                "status": day_metrics.get("status"),
                "recommendations": len(papers),
            }
            summary.append(day_summary)
            total_recommendations += len(papers)

        self.last_run_metrics = {
            "status": "completed",
            "mode": "backfill",
            "start_date": start_date,
            "end_date": end_date,
            "until_date": end_date,
            "date_count": len(dates),
            "final_recommendations": total_recommendations,
            "dates": summary,
        }
        return self.last_run_metrics

    async def fetch_corpus(self, force_refresh: bool = False) -> list[CorpusPaper]:
        logger.info("Fetching corpus...")
        from .webdav import build_corpus_manifest, load_corpus_manifest, manifest_content_equal

        cached = None
        cached_manifest = None
        if not force_refresh:
            cached = await db.load_corpus_cache()
            if cached and len(cached) > 0:
                valid = True
                for paper in cached:
                    if not paper.file_path or not os.path.exists(paper.file_path):
                        valid = False
                        logger.warning(f"Cache file missing: {paper.file_path}")
                        break
                if valid:
                    cached_manifest = load_corpus_manifest(self.config["webdav"])
                    manifest_ttl_minutes = self.config.get("executor", {}).get(
                        "manifest_ttl_minutes", 30
                    )
                    manifest_fresh = False
                    generated_at = (cached_manifest or {}).get("generated_at")
                    if generated_at:
                        try:
                            manifest_age_seconds = (
                                datetime.now() - datetime.fromisoformat(generated_at)
                            ).total_seconds()
                            manifest_fresh = (
                                manifest_age_seconds < manifest_ttl_minutes * 60
                            )
                        except ValueError:
                            manifest_fresh = False

                    if manifest_fresh:
                        logger.info(
                            "Using cached corpus manifest within freshness window"
                        )
                    else:
                        current_manifest = build_corpus_manifest(self.config["webdav"])
                        if not manifest_content_equal(current_manifest, cached_manifest):
                            valid = False
                            logger.info("Corpus source changed, refreshing corpus cache...")
                if valid:
                    logger.info(f"Using cached corpus: {len(cached)} papers")
                    return cached
                else:
                    logger.info("Cache files missing, refreshing corpus...")

        loop = asyncio.get_event_loop()
        previous_corpus = cached if cached else None
        previous_manifest = cached_manifest
        return await loop.run_in_executor(
            get_executor_pool(self.config),
            lambda: fetch_corpus(
                self.config["webdav"],
                self.config.get("executor", {}),
                previous_corpus=previous_corpus,
                previous_manifest=previous_manifest,
            ),
        )

    async def run(self, skip_tldr: bool = False) -> list[Paper]:
        business_date = _get_business_date(self.config)
        stage_seconds: dict[str, float] = {}

        def _finish_stage(name: str, started: float) -> None:
            stage_seconds[name] = round(time.perf_counter() - started, 3)
            self.last_run_metrics["stage_seconds"] = dict(stage_seconds)

        stage_started = time.perf_counter()
        corpus = await self.fetch_corpus()
        _finish_stage("corpus", stage_started)
        self.last_run_metrics.update(
            {
                "status": "running",
                "skip_tldr": skip_tldr,
                "corpus_count": len(corpus),
            }
        )
        if not corpus:
            logger.error(
                "No papers found. Please check your WebDAV or local path settings."
            )
            self.last_run_metrics.update(
                {
                    "status": "no_corpus",
                    "final_recommendations": 0,
                }
            )
            return []

        await db.save_corpus_cache(corpus)

        sources = self.config.get("executor", {}).get("source", ["arxiv"])
        loop = asyncio.get_running_loop()
        source_metrics: list[dict[str, object]] = []
        reset_llm_usage()
        use_llm = bool(self.config.get("llm", {}).get("api_key")) and not skip_tldr
        llm_ready = False
        if use_llm:
            llm_ready, llm_message = await loop.run_in_executor(
                get_executor_pool(self.config),
                lambda: check_connectivity(self.config),
            )
            if not llm_ready:
                logger.error(llm_message)
                note_llm_error(llm_message)
        elif not skip_tldr and not self.config.get("llm", {}).get("api_key"):
            note_llm_warning("未配置 LLM API key，已跳过画像、裁判和 TLDR")
        stage_started = time.perf_counter()
        profile = await _ensure_interest_profile(self.config, corpus, llm_ready=llm_ready)
        _finish_stage("profile", stage_started)
        retrieval_config = copy.deepcopy(self.config)
        if profile:
            retrieval_config["interest_profile"] = profile

        async def _retrieve_source(source: str) -> tuple[str, list[Paper]]:
            logger.info(f"Retrieving {source} papers...")
            retriever = get_retriever_cls(source)(retrieval_config)
            papers = await loop.run_in_executor(
                get_executor_pool(self.config),
                lambda retriever=retriever: retriever.retrieve_papers(corpus),
            )
            return source, papers

        stage_started = time.perf_counter()
        source_results = await asyncio.gather(*[_retrieve_source(source) for source in sources])
        _finish_stage("retrieve", stage_started)
        from .retriever.arxiv_retriever import LAST_RETRIEVAL_STATS

        self.last_run_metrics["retrieval_stats"] = dict(LAST_RETRIEVAL_STATS)

        all_papers = []
        for source, papers in source_results:
            logger.info(f"Retrieved {len(papers)} {source} papers")
            source_metrics.append({"name": source, "retrieved": len(papers)})
            all_papers.extend(papers)
        self.last_run_metrics["sources"] = source_metrics

        if not all_papers:
            logger.info("No new papers found today")
            self.last_run_metrics.update(
                {
                    "status": "no_candidates",
                    "retrieved_candidates": 0,
                    "filtered_candidates": 0,
                    "final_recommendations": 0,
                }
            )
            _merge_llm_usage(self.last_run_metrics)
            return []

        self.last_run_metrics["retrieved_candidates"] = len(all_papers)
        all_papers = await _filter_seen_papers(all_papers, corpus, business_date)
        self.last_run_metrics["filtered_candidates"] = len(all_papers)
        if not all_papers:
            logger.info("All retrieved papers were already in corpus or previously recommended")
            self.last_run_metrics.update(
                {
                    "status": "all_seen",
                    "final_recommendations": 0,
                }
            )
            _merge_llm_usage(self.last_run_metrics)
            return []

        logger.info(f"Total {len(all_papers)} papers from all sources")

        logger.info("Reranking with local reranker...")
        reranker = get_reranker_cls("local")(self.config)
        stage_started = time.perf_counter()
        reranked = await loop.run_in_executor(
            get_executor_pool(self.config), lambda: reranker.rerank(all_papers, corpus)
        )
        _finish_stage("rerank", stage_started)
        self.last_run_metrics["reranked_candidates"] = len(reranked)
        max_num = int(get_config_value(self.config, "executor.max_paper_num"))

        if use_llm and llm_ready:
            self.last_run_metrics.update(
                _default_llm_metrics(enabled=True, target_count=min(len(reranked), max_num))
            )
            stage_started = time.perf_counter()
            reranked = await _apply_judge(
                reranked, profile, self.config, self.last_run_metrics
            )
            _finish_stage("judge", stage_started)
            stage_started = time.perf_counter()
            await _apply_tldr_enrichment(
                reranked, self.config, self.last_run_metrics, profile
            )
            _finish_stage("tldr", stage_started)
        else:
            reranked = reranked[:max_num]
            self.last_run_metrics.update(
                _default_llm_metrics(enabled=False, target_count=len(reranked))
            )
            for paper in reranked:
                _fallback_tldr(paper)
            if use_llm and not llm_ready:
                _merge_llm_usage(self.last_run_metrics)
                self.last_run_metrics["llm_enabled"] = True

        tz_name = get_config_value(self.config, "executor.timezone")
        await db.save_papers(reranked, business_date)
        self.last_run_metrics.update(
            {
                "status": "completed",
                "saved_date": business_date,
                "business_timezone": tz_name,
                "final_recommendations": len(reranked),
            }
        )
        logger.info(f"Saved {len(reranked)} recommended papers for {business_date}")
        if use_llm:
            self.last_run_metrics["llm_models"] = {
                "extract": resolve_model(self.config, "extract"),
                "summarize": resolve_model(self.config, "summarize"),
                "judge": resolve_model(self.config, "judge"),
            }
            _merge_llm_usage(self.last_run_metrics)

        return reranked

    async def enrich_saved_dates(self, dates: list[str]) -> dict[str, object]:
        """Fill TLDR and judge fields for papers already stored. Does not call arXiv."""
        reset_llm_usage()
        self.last_run_metrics = {"status": "running", "mode": "enrich", "dates": []}
        if not self.config.get("llm", {}).get("api_key"):
            note_llm_error("未配置 LLM API key，无法补全已入库论文")
            _merge_llm_usage(self.last_run_metrics)
            self.last_run_metrics["status"] = "llm_unavailable"
            return self.last_run_metrics

        corpus = await self.fetch_corpus()
        loop = asyncio.get_running_loop()
        llm_ready, llm_message = await loop.run_in_executor(
            get_executor_pool(self.config),
            lambda: check_connectivity(self.config),
        )
        if not llm_ready:
            note_llm_error(llm_message)
            _merge_llm_usage(self.last_run_metrics)
            self.last_run_metrics["status"] = "llm_unavailable"
            return self.last_run_metrics

        profile = await _ensure_interest_profile(self.config, corpus, llm_ready=True)
        summaries: list[dict[str, object]] = []
        for business_date in dates:
            rows = await db.get_papers_by_date(business_date)
            papers = [_paper_from_row(row) for row in rows]
            if not papers:
                summaries.append({"date": business_date, "papers": 0, "tldr": 0})
                continue
            metrics = _default_llm_metrics(enabled=True, target_count=len(papers))
            papers = await _apply_judge(
                papers, profile, self.config, metrics, max_num=len(papers)
            )
            await _apply_tldr_enrichment(papers, self.config, metrics, profile)
            await db.save_papers(papers, business_date)
            summaries.append(
                {
                    "date": business_date,
                    "papers": len(papers),
                    "tldr": sum(1 for paper in papers if paper.tldr),
                    "judged": sum(1 for paper in papers if paper.judge_relevance is not None),
                }
            )
        self.last_run_metrics = {
            "status": "completed",
            "mode": "enrich",
            "dates": summaries,
            "final_recommendations": sum(int(item["papers"]) for item in summaries),
        }
        _merge_llm_usage(self.last_run_metrics)
        return self.last_run_metrics

    async def backfill_tldrs(self, plan: list[tuple[str, bool]]) -> dict[str, object]:
        """Regenerate TLDRs for papers already stored. Does not fetch arXiv or re-judge."""
        reset_llm_usage()
        summaries: list[dict[str, object]] = []
        if not self.config.get("llm", {}).get("api_key"):
            note_llm_error("未配置 LLM API key，无法补写 TLDR")
            return {"status": "llm_unavailable", "dates": summaries, **get_llm_usage()}

        profile_row = await db.load_interest_profile()
        profile = (profile_row or {}).get("profile") if profile_row else None
        loop = asyncio.get_running_loop()
        for business_date, only_empty in plan:
            rows = await db.get_papers_by_date(business_date)
            papers = [_paper_from_row(row) for row in rows]
            if only_empty:
                papers = [paper for paper in papers if not (paper.tldr or "").strip()]
            if not papers:
                summaries.append(
                    {"date": business_date, "selected": 0, "updated": 0, "only_empty": only_empty}
                )
                continue
            generated = await loop.run_in_executor(
                get_executor_pool(self.config),
                lambda papers=papers: generate_tldrs_batch(papers, profile, self.config),
            )
            generated = generated or {}
            updated_papers: list[Paper] = []
            for index, paper in enumerate(papers):
                payload = generated.get(paper.url or "") or generated.get(f"paper-{index + 1}")
                if payload is None and len(papers) == 1 and len(generated) == 1:
                    payload = next(iter(generated.values()))
                if not payload or not payload.get("tldr"):
                    continue
                paper.tldr = payload["tldr"]
                paper.method = payload.get("method") or None
                paper.evidence = payload.get("evidence") or None
                paper.why_for_me = payload.get("why_for_me") or None
                updated_papers.append(paper)
            count = await db.update_paper_summaries(business_date, updated_papers)
            cache_key = _get_llm_cache_key(self.config, "summarize")
            await db.save_candidate_enrichments(
                [
                    {
                        "url": paper.url,
                        "pdf_url": paper.pdf_url,
                        "content_key": _content_key(paper),
                        "tldr": paper.tldr,
                        "method": paper.method,
                        "evidence": paper.evidence,
                        "why_for_me": paper.why_for_me,
                        "llm_cache_key": cache_key,
                    }
                    for paper in updated_papers
                    if paper.url
                ]
            )
            summaries.append(
                {
                    "date": business_date,
                    "selected": len(papers),
                    "updated": count,
                    "only_empty": only_empty,
                }
            )
            logger.info(
                f"Backfilled TLDR for {count}/{len(papers)} saved papers on {business_date}"
            )
        result: dict[str, object] = {
            "status": "completed",
            "dates": summaries,
            "updated": sum(int(item["updated"]) for item in summaries),
        }
        _merge_llm_usage(result)
        return result


def plan_tldr_backfill(
    explicit_dates: list[str],
    recent_empty_dates: list[str],
    *,
    only_empty: bool = False,
) -> list[tuple[str, bool]]:
    """Return (date, only_empty) for a TLDR backfill.

    Dates passed explicitly are regenerated unless only_empty is set.
    Recent dates only fill recommendations whose TLDR is still empty.
    """
    plan: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for value in explicit_dates:
        day = _normalize_business_date(value)
        if day in seen:
            continue
        seen.add(day)
        plan.append((day, only_empty))
    for value in recent_empty_dates:
        day = _normalize_business_date(value)
        if day in seen:
            continue
        seen.add(day)
        plan.append((day, True))
    plan.sort(key=lambda item: item[0])
    return plan


def _paper_from_row(row: dict[str, Any]) -> Paper:
    authors = row.get("authors")
    if isinstance(authors, str):
        try:
            authors = json.loads(authors)
        except json.JSONDecodeError:
            authors = []
    keep = row.get("judge_keep")
    return Paper(
        source=row.get("source") or "arxiv",
        title=row.get("title") or "",
        authors=authors or [],
        abstract=row.get("abstract") or "",
        url=row.get("url") or "",
        pdf_url=row.get("pdf_url"),
        code_url=row.get("code_url"),
        tldr=row.get("tldr"),
        method=row.get("method"),
        evidence=row.get("evidence"),
        why_for_me=row.get("why_for_me"),
        score=row.get("score"),
        judge_relevance=row.get("judge_relevance"),
        judge_reason=row.get("judge_reason"),
        judge_keep=None if keep is None else bool(keep),
        date=row.get("date"),
    )
