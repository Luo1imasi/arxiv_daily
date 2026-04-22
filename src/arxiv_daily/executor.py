import os
import asyncio
import concurrent.futures
from datetime import datetime
from zoneinfo import ZoneInfo
from loguru import logger

from .protocol import Paper
from .webdav import fetch_corpus
from .retriever import get_retriever_cls
from .reranker import get_reranker_cls
from .llm import generate_tldr
from . import database as db
from .config import get_config_value
from .utils import make_content_key

_executor_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _fallback_tldr(p: Paper):
    # Avoid presenting raw abstracts as TLDR text in the UI.
    p.tldr = None


def _get_business_timezone(config: dict) -> ZoneInfo:
    tz_name = get_config_value(config, "executor.timezone")
    return ZoneInfo(tz_name)


def _get_business_date(config: dict) -> str:
    return datetime.now(_get_business_timezone(config)).strftime("%Y-%m-%d")


def _get_llm_cache_key(config: dict) -> str:
    return "|".join(
        [
            get_config_value(config, "llm.base_url"),
            get_config_value(config, "llm.model"),
            get_config_value(config, "llm.language"),
        ]
    )


def _default_llm_metrics(*, enabled: bool, target_count: int = 0) -> dict[str, object]:
    return {
        "llm_enabled": enabled,
        "llm_error_count": 0,
        "llm_target_count": target_count,
        "llm_cache_hits": 0,
        "llm_request_count": 0,
        "tldr_cache_hits": 0,
        "tldr_request_count": 0,
    }


async def _apply_tldr_enrichment(
    papers: list[Paper], config: dict, metrics: dict[str, object]
) -> None:
    llm_config = config.get("llm", {})
    metrics.update(
        _default_llm_metrics(
            enabled=bool(llm_config.get("api_key")),
            target_count=len(papers),
        )
    )
    if not llm_config.get("api_key"):
        logger.info("No LLM API key configured, skipping TLDR generation")
        for paper in papers:
            _fallback_tldr(paper)
        metrics.update(_default_llm_metrics(enabled=False))
        return

    logger.info("Generating TLDRs for selected papers...")
    errors = []
    failed_urls = set()
    llm_cache_key = _get_llm_cache_key(config)
    cached_enrichments = await db.load_candidate_enrichments(
        [paper.url for paper in papers if paper.url]
    )
    tldr_cache_hits = 0
    papers_for_tldr = []
    for paper in papers:
        cached = cached_enrichments.get(paper.url or "")
        content_key = make_content_key(paper.title, paper.abstract or "")
        if (
            cached
            and cached.get("content_key") == content_key
            and cached.get("llm_cache_key") == llm_cache_key
            and cached.get("tldr")
        ):
            paper.tldr = cached["tldr"]
            tldr_cache_hits += 1
        else:
            papers_for_tldr.append(paper)

    metrics["llm_cache_hits"] = tldr_cache_hits
    metrics["tldr_cache_hits"] = tldr_cache_hits
    metrics["llm_request_count"] = len(papers_for_tldr)
    metrics["tldr_request_count"] = len(papers_for_tldr)

    for paper in papers_for_tldr:
        try:
            paper.tldr = generate_tldr(paper, config)
            if not paper.tldr:
                _fallback_tldr(paper)
        except Exception as e:
            errors.append((paper, e))
            if getattr(paper, "url", None):
                failed_urls.add(paper.url)
            _fallback_tldr(paper)

    for paper, error in errors:
        logger.warning(f"Error processing {paper.title}: {error}")

    cache_entries = []
    for paper in papers:
        content_key = make_content_key(paper.title, paper.abstract or "")
        cached = cached_enrichments.get(paper.url or "", {})
        entry = {
            "url": paper.url,
            "pdf_url": paper.pdf_url,
            "content_key": content_key,
        }
        if paper.url not in failed_urls and paper.tldr:
            entry.update(
                {
                    "tldr": paper.tldr,
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
                    "llm_cache_key": cached.get("llm_cache_key"),
                }
            )
        cache_entries.append(entry)

    await db.save_candidate_enrichments(cache_entries)
    metrics["llm_error_count"] = len(errors)


async def _filter_seen_papers(
    papers: list[Paper], corpus: list[Paper] | list, business_date: str
) -> list[Paper]:
    seen_urls = await db.get_seen_paper_urls(exclude_date=business_date)
    seen_content_keys = await db.get_seen_paper_content_keys(exclude_date=business_date)
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
    def __init__(self, config: dict):
        self.config = config
        self.last_run_metrics: dict[str, object] = {
            "status": "initialized",
            "sources": [],
        }

    async def fetch_corpus(self, force_refresh: bool = False) -> list:
        logger.info("Fetching corpus...")
        from .webdav import build_corpus_manifest, load_corpus_manifest

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
                        if current_manifest != cached_manifest:
                            valid = False
                            logger.info("Corpus source changed, refreshing corpus cache...")
                if valid:
                    logger.info(f"Using cached corpus: {len(cached)} papers")
                    return cached
                else:
                    logger.info("Cache files missing, refreshing corpus...")

        loop = asyncio.get_event_loop()
        previous_corpus = cached if 'cached' in locals() and cached else None
        previous_manifest = cached_manifest if 'cached_manifest' in locals() else None
        return await loop.run_in_executor(
            _executor_pool,
            lambda: fetch_corpus(
                self.config["webdav"],
                self.config.get("executor", {}),
                previous_corpus=previous_corpus,
                previous_manifest=previous_manifest,
            ),
        )

    async def run(self, skip_tldr: bool = False) -> list[Paper]:
        business_date = _get_business_date(self.config)
        corpus = await self.fetch_corpus()
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
        loop = asyncio.get_event_loop()

        async def _retrieve_source(source: str) -> tuple[str, list[Paper]]:
            logger.info(f"Retrieving {source} papers...")
            retriever = get_retriever_cls(source)(self.config)
            papers = await loop.run_in_executor(
                _executor_pool,
                lambda retriever=retriever: retriever.retrieve_papers(corpus),
            )
            return source, papers

        source_results = await asyncio.gather(*[_retrieve_source(source) for source in sources])

        all_papers = []
        for source, papers in source_results:
            logger.info(f"Retrieved {len(papers)} {source} papers")
            self.last_run_metrics["sources"].append(
                {"name": source, "retrieved": len(papers)}
            )
            all_papers.extend(papers)

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
            return []

        logger.info(f"Total {len(all_papers)} papers from all sources")

        logger.info("Reranking with local reranker...")
        reranker = get_reranker_cls("local")(self.config)
        reranked = await loop.run_in_executor(
            _executor_pool, lambda: reranker.rerank(all_papers, corpus)
        )

        max_num = int(get_config_value(self.config, "executor.max_paper_num"))
        reranked = reranked[:max_num]
        self.last_run_metrics["reranked_candidates"] = len(reranked)

        if not skip_tldr:
            await _apply_tldr_enrichment(reranked, self.config, self.last_run_metrics)
        else:
            self.last_run_metrics.update(_default_llm_metrics(enabled=False))

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

        return reranked
