import asyncio
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import cast, override
from unittest.mock import patch

from arxiv_daily import database as db
from arxiv_daily import webdav as webdav_module
from arxiv_daily.executor import (
    Executor,
    _apply_tldr_enrichment,
    _default_llm_metrics,
    _ensure_interest_profile,
    _fallback_tldr,
    _filter_seen_papers,
    _profile_signatures,
    _select_judged_papers,
    plan_tldr_backfill,
)
from arxiv_daily.llm import PROMPT_VERSIONS
from arxiv_daily.protocol import CorpusPaper
from arxiv_daily.protocol import Paper
from arxiv_daily.retriever.base import BaseRetriever, register_retriever
from arxiv_daily.utils import make_content_key
from arxiv_daily.webdav import save_corpus_manifest


class FallbackTldrTests(unittest.TestCase):
    def test_fallback_leaves_tldr_empty(self):
        paper = Paper(
            source="arxiv",
            title="Test Paper",
            authors=["Alice"],
            abstract="This is an English abstract that should not appear as TLDR.",
            url="https://example.com/paper",
        )

        _fallback_tldr(paper)

        self.assertIsNone(paper.tldr)


class DefaultLlmMetricsTests(unittest.TestCase):
    def test_metrics_only_keep_tldr_related_fields(self):
        metrics = _default_llm_metrics(enabled=True, target_count=3)

        self.assertEqual(
            metrics,
            {
                "llm_enabled": True,
                "llm_error_count": 0,
                "llm_target_count": 3,
                "llm_cache_hits": 0,
                "llm_request_count": 0,
                "tldr_cache_hits": 0,
                "tldr_request_count": 0,
                "judge_cache_hits": 0,
                "judge_request_count": 0,
                "coarse_cache_hits": 0,
                "coarse_request_count": 0,
                "coarse_fallback": False,
                "llm_prompt_tokens": 0,
                "llm_completion_tokens": 0,
                "llm_usage_by_model": {},
                "llm_warning": "",
            },
        )


class FilterSeenPapersTests(unittest.TestCase):
    def test_filter_excludes_previously_recommended_same_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                asyncio.run(db.init_db())
                historical = Paper(
                    source="arxiv",
                    title="Shared Title",
                    authors=["Alice"],
                    abstract="Shared abstract",
                    url="https://example.com/old",
                )
                asyncio.run(db.save_papers([historical], "2026-04-21"))

                candidates = [
                    Paper(
                        source="arxiv",
                        title="Shared Title",
                        authors=["Bob"],
                        abstract="Shared abstract",
                        url="https://example.com/new",
                    ),
                    Paper(
                        source="arxiv",
                        title="Fresh Title",
                        authors=["Carol"],
                        abstract="Fresh abstract",
                        url="https://example.com/fresh",
                    ),
                ]

                filtered = asyncio.run(
                    _filter_seen_papers(candidates, [], "2026-04-22")
                )

                self.assertEqual([paper.title for paper in filtered], ["Fresh Title"])
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir

    def test_filter_ignores_future_recommendations_during_backfill(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                asyncio.run(db.init_db())
                future = Paper(
                    source="arxiv",
                    title="Shared Title",
                    authors=["Alice"],
                    abstract="Shared abstract",
                    url="https://example.com/future",
                )
                past = Paper(
                    source="arxiv",
                    title="Past Title",
                    authors=["Bob"],
                    abstract="Past abstract",
                    url="https://example.com/past",
                )
                asyncio.run(db.save_papers([future], "2026-04-24"))
                asyncio.run(db.save_papers([past], "2026-04-20"))

                candidates = [
                    Paper(
                        source="arxiv",
                        title="Shared Title",
                        authors=["Carol"],
                        abstract="Shared abstract",
                        url="https://example.com/shared-new",
                    ),
                    Paper(
                        source="arxiv",
                        title="Past Title",
                        authors=["Dan"],
                        abstract="Past abstract",
                        url="https://example.com/past-new",
                    ),
                ]

                filtered = asyncio.run(
                    _filter_seen_papers(candidates, [], "2026-04-22")
                )

                self.assertEqual([paper.title for paper in filtered], ["Shared Title"])
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


class KeywordCacheTests(unittest.TestCase):
    def test_load_keywords_for_papers_uses_content_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                asyncio.run(db.init_db())
                paper = CorpusPaper(
                    title="Graph Attention for Agents",
                    abstract="We study agent planning with graph attention.",
                    added_date=datetime(2026, 4, 22),
                )
                asyncio.run(
                    db.save_keyword_cache(
                        paper.title,
                        paper.abstract,
                        ["graph attention", "agent planning"],
                    )
                )

                cached = asyncio.run(db.load_keywords_for_papers([paper]))

                self.assertEqual(
                    cached,
                    {
                        make_content_key(paper.title, paper.abstract): [
                            "graph attention",
                            "agent planning",
                        ]
                    },
                )
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


class SavePapersOverwriteTests(unittest.TestCase):
    def test_save_papers_replaces_existing_date_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                asyncio.run(db.init_db())
                old_paper = Paper(
                    source="arxiv",
                    title="Old Paper",
                    authors=["Alice"],
                    abstract="Old abstract",
                    url="https://example.com/old",
                    score=4.0,
                )
                new_paper = Paper(
                    source="arxiv",
                    title="New Paper",
                    authors=["Bob"],
                    abstract="New abstract",
                    url="https://example.com/new",
                    score=9.0,
                )

                asyncio.run(db.save_papers([old_paper], "2026-04-22"))
                asyncio.run(db.save_papers([new_paper], "2026-04-22"))
                papers = asyncio.run(db.get_papers_by_date("2026-04-22"))

                self.assertEqual([paper["title"] for paper in papers], ["New Paper"])
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


class CorpusManifestTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_corpus_ignores_manifest_generated_at_when_files_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                pdf_path = Path(tmpdir) / "cached.pdf"
                pdf_path.write_text("cached", encoding="utf-8")
                source_path = str(Path(tmpdir) / "source.pdf")
                cached = CorpusPaper(
                    title="Cached Paper",
                    abstract="Cached abstract",
                    file_path=str(pdf_path),
                    source_path=source_path,
                    added_date=datetime(2026, 4, 22),
                )
                await db.save_corpus_cache([cached])
                webdav_config = {"local_path": str(Path(tmpdir) / "corpus")}
                save_corpus_manifest(
                    webdav_config,
                    {
                        "count": 1,
                        "generated_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                        "files": [
                            {
                                "path": source_path,
                                "name": "source.pdf",
                                "size": 123,
                                "modified": "2026-04-22T00:00:00",
                            }
                        ],
                    },
                )

                old_build = webdav_module.build_corpus_manifest
                old_fetch = webdav_module.fetch_corpus
                webdav_module.build_corpus_manifest = lambda _: {
                    "count": 1,
                    "generated_at": datetime.now().isoformat(),
                    "files": [
                        {
                            "path": source_path,
                            "name": "source.pdf",
                            "size": 123,
                            "modified": "2026-04-22T00:00:00",
                        }
                    ],
                }

                def fail_fetch_corpus(*args: object, **kwargs: object) -> list[CorpusPaper]:
                    raise AssertionError("corpus should be reused")

                webdav_module.fetch_corpus = fail_fetch_corpus
                try:
                    executor = Executor(
                        {
                            "webdav": webdav_config,
                            "executor": {"manifest_ttl_minutes": 0},
                        }
                    )
                    result = await executor.fetch_corpus()
                finally:
                    webdav_module.build_corpus_manifest = old_build
                    webdav_module.fetch_corpus = old_fetch

                self.assertEqual([paper.title for paper in result], ["Cached Paper"])
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


class JudgeSelectionTests(unittest.TestCase):
    def test_keeps_mmr_order_and_fills_when_too_few_are_marked(self):
        papers = [
            Paper(source="arxiv", title=f"P{index}", authors=[], abstract="a", url=f"https://example.com/{index}", judge_relevance=float(index), judge_keep=index >= 3)
            for index in range(5)
        ]

        chosen = _select_judged_papers(papers, 3)

        self.assertEqual([paper.url for paper in chosen], [
            "https://example.com/3",
            "https://example.com/4",
            "https://example.com/2",
        ])


class ProfileSignatureTests(unittest.TestCase):
    def test_feedback_changes_the_profile_signature(self):
        corpus = [CorpusPaper(title="Agents", abstract="Planning", added_date=datetime(2026, 10, 6))]
        before = _profile_signatures(corpus, [])
        after = _profile_signatures(
            corpus,
            [{"url": "https://example.com/a", "vote": "irrelevant", "updated_at": "2026-10-06T00:00:00+00:00"}],
        )

        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])


class InterestProfileCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_profile_rebuilds_and_falls_back_when_llm_is_down(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                corpus = [
                    CorpusPaper(
                        title="Humanoid whole-body control",
                        abstract="Loco-manipulation on rough terrain.",
                        added_date=datetime(2026, 10, 6),
                    )
                ]
                corpus_signature, feedback_signature = _profile_signatures(corpus, [])
                legacy = {
                    "summary": "旧画像",
                    "topics": ["人形控制"],
                    "methods": [],
                    "not_interested": ["通用强化学习理论"],
                    "representative_papers": [],
                    "canonical_terms": ["humanoid whole-body control"],
                }
                await db.save_interest_profile(
                    legacy,
                    corpus_signature=corpus_signature,
                    feedback_signature=feedback_signature,
                    model="grok-4.7",
                    prompt_version="profile-v1",
                )
                config = {
                    "llm": {
                        "model": "grok-4.7",
                        "language": "Chinese",
                        "models": {"profile": "grok-4.7"},
                    }
                }
                fresh = {
                    "summary": "新画像",
                    "topics": ["人形控制"],
                    "methods": [],
                    "not_interested": ["通用强化学习理论"],
                    "representative_papers": [],
                    "canonical_terms": ["humanoid whole-body control", "reinforcement learning"],
                    "core_terms": ["humanoid whole-body control"],
                    "broad_terms": ["reinforcement learning"],
                    "negative_terms": ["ankle mechanism"],
                }
                with patch(
                    "arxiv_daily.executor.build_interest_profile", return_value=fresh
                ) as build:
                    rebuilt = await _ensure_interest_profile(config, corpus, llm_ready=True)
                build.assert_called_once()
                self.assertEqual(rebuilt["negative_terms"], ["ankle mechanism"])
                self.assertEqual(rebuilt["core_terms"], ["humanoid whole-body control"])

                with patch(
                    "arxiv_daily.executor.build_interest_profile", return_value=fresh
                ) as build_again:
                    cached = await _ensure_interest_profile(config, corpus, llm_ready=True)
                build_again.assert_not_called()
                self.assertEqual(cached["negative_terms"], ["ankle mechanism"])

                await db.save_interest_profile(
                    legacy,
                    corpus_signature=corpus_signature,
                    feedback_signature=feedback_signature,
                    model="grok-4.7",
                    prompt_version=PROMPT_VERSIONS["profile"],
                )
                with patch(
                    "arxiv_daily.executor.build_interest_profile", return_value=fresh
                ) as offline:
                    fallback = await _ensure_interest_profile(config, corpus, llm_ready=False)
                offline.assert_not_called()
                self.assertEqual(fallback["canonical_terms"], ["humanoid whole-body control"])
                self.assertNotIn("negative_terms", fallback)
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


class CandidateEnrichmentCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_candidate_enrichment_clears_stale_tldr_when_explicitly_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                await db.save_candidate_enrichments(
                    [
                        {
                            "url": "https://example.com/paper",
                            "pdf_url": "https://example.com/paper.pdf",
                            "content_key": "old-content",
                            "tldr": "old summary",
                            "llm_cache_key": "old-llm",
                        }
                    ]
                )
                await db.save_candidate_enrichments(
                    [
                        {
                            "url": "https://example.com/paper",
                            "pdf_url": "https://example.com/paper.pdf",
                            "content_key": "new-content",
                            "tldr": None,
                            "llm_cache_key": None,
                        }
                    ]
                )

                cached = await db.load_candidate_enrichments(["https://example.com/paper"])

                self.assertEqual(cached["https://example.com/paper"]["content_key"], "new-content")
                self.assertIsNone(cached["https://example.com/paper"]["tldr"])
                self.assertIsNone(cached["https://example.com/paper"]["llm_cache_key"])
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir

    async def test_old_or_empty_tldr_cache_does_not_hit(self):
        from unittest.mock import patch

        from arxiv_daily.llm import make_llm_cache_key

        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                config = {
                    "llm": {
                        "api_key": "test-key",
                        "base_url": "http://127.0.0.1:9/v1",
                        "model": "deepseek-flash",
                        "language": "Chinese",
                        "models": {"summarize": "deepseek-flash"},
                    }
                }
                paper = Paper(
                    source="arxiv",
                    title="Cached Paper",
                    authors=["Ada"],
                    abstract="An abstract about retrieval.",
                    url="https://example.com/cached",
                )
                content_key = make_content_key(paper.title, paper.abstract or "")
                await db.save_candidate_enrichments(
                    [
                        {
                            "url": paper.url,
                            "content_key": content_key,
                            "tldr": "old empty-looking summary",
                            "llm_cache_key": "base|old-model|Chinese",
                        }
                    ]
                )
                metrics: dict[str, object] = {}
                payload = {
                    paper.url: {
                        "tldr": "中文总结",
                        "method": "方法",
                        "evidence": "证据",
                        "why_for_me": "适合我",
                    }
                }
                with patch("arxiv_daily.executor.generate_tldrs_batch", return_value=payload) as generate:
                    await _apply_tldr_enrichment([paper], config, metrics, None)

                generate.assert_called_once()
                self.assertEqual(paper.tldr, "中文总结")
                self.assertEqual(paper.method, "方法")
                self.assertEqual(metrics["tldr_cache_hits"], 0)

                fresh = Paper(
                    source="arxiv",
                    title="Cached Paper",
                    authors=["Ada"],
                    abstract="An abstract about retrieval.",
                    url="https://example.com/cached",
                )
                hit_metrics: dict[str, object] = {}
                with patch("arxiv_daily.executor.generate_tldrs_batch", return_value=payload) as generate_again:
                    await _apply_tldr_enrichment([fresh], config, hit_metrics, None)

                generate_again.assert_not_called()
                self.assertEqual(fresh.tldr, "中文总结")
                self.assertEqual(hit_metrics["tldr_cache_hits"], 1)
                self.assertIn("tldr-v3", str(make_llm_cache_key(config, "summarize")))
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


@register_retriever("scope-spy")
class ScopeSpyRetriever(BaseRetriever):
    seen_business_dates: list[str] = []

    @override
    def _retrieve_raw_papers(self) -> list[dict[str, str]]:
        business_date = str(self.config["executor"]["business_date"])
        self.seen_business_dates.append(business_date)
        return [
            {
                "source": "scope-spy",
                "title": f"Paper {business_date}",
                "abstract": "Abstract",
                "url": f"https://example.com/{business_date}",
            }
        ]

    @override
    def convert_to_paper(self, raw_paper: dict[str, str]) -> Paper:
        return Paper(
            source="scope-spy",
            title=raw_paper["title"],
            authors=[],
            abstract=raw_paper["abstract"],
            url=raw_paper["url"],
        )


class BackfillRetrieverScopeTests(unittest.IsolatedAsyncioTestCase):
    async def test_backfill_passes_each_business_date_to_retriever(self):
        ScopeSpyRetriever.seen_business_dates = []
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                executor = Executor(
                    {
                        "webdav": {"local_path": str(Path(tmpdir) / "corpus")},
                        "executor": {
                            "business_date": "2026-04-24",
                            "timezone": "UTC",
                            "source": ["scope-spy"],
                            "max_paper_num": 10,
                        },
                        "llm": {"api_key": ""},
                    }
                )
                old_fetch_corpus = Executor.fetch_corpus

                async def fake_fetch_corpus(
                    self: Executor, force_refresh: bool = False
                ) -> list[CorpusPaper]:
                    return [
                        CorpusPaper(
                            title="Corpus",
                            abstract="Corpus abstract",
                            added_date=datetime(2026, 4, 20),
                        )
                    ]

                Executor.fetch_corpus = fake_fetch_corpus
                try:
                    await executor.run_between_dates("2026-04-22", "2026-04-24", skip_tldr=True)
                finally:
                    Executor.fetch_corpus = old_fetch_corpus

                self.assertEqual(
                    ScopeSpyRetriever.seen_business_dates,
                    ["2026-04-22", "2026-04-23", "2026-04-24"],
                )
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


class BackfillRunTests(unittest.IsolatedAsyncioTestCase):
    class StubExecutor(Executor):
        def __init__(self, config: dict[str, object]):
            super().__init__(config)
            self.dates: list[str] = []

        @override
        async def run_for_date(
            self, business_date: str | date, skip_tldr: bool = False
        ) -> list[Paper]:
            date_value = str(business_date)
            self.dates.append(date_value)
            self.last_run_metrics = {
                "status": "completed",
                "saved_date": date_value,
                "final_recommendations": 1,
            }
            return [
                Paper(
                    source="arxiv",
                    title=f"Paper {date_value}",
                    authors=[],
                    abstract="Abstract",
                    url=f"https://example.com/{date_value}",
                )
            ]

    def make_executor(self):
        return self.StubExecutor(
            {"executor": {"business_date": "2026-04-24", "timezone": "UTC"}}
        )

    async def test_run_until_date_runs_oldest_to_newest(self):
        executor = self.make_executor()

        metrics = await executor.run_until_date("2026-04-22", skip_tldr=True)

        self.assertEqual(executor.dates, ["2026-04-22", "2026-04-23", "2026-04-24"])
        self.assertEqual(metrics["date_count"], 3)
        self.assertEqual(metrics["final_recommendations"], 3)
        self.assertEqual(metrics["start_date"], "2026-04-24")
        self.assertEqual(metrics["end_date"], "2026-04-22")
        date_metrics = cast(list[dict[str, object]], metrics["dates"])
        self.assertIsInstance(date_metrics, list)
        self.assertEqual(
            [item["date"] for item in date_metrics],
            ["2026-04-22", "2026-04-23", "2026-04-24"],
        )

    async def test_run_between_dates_runs_selected_range_oldest_to_newest(self):
        executor = self.make_executor()

        metrics = await executor.run_between_dates("2026-04-21", "2026-04-23", skip_tldr=True)

        self.assertEqual(executor.dates, ["2026-04-21", "2026-04-22", "2026-04-23"])
        self.assertEqual(metrics["start_date"], "2026-04-21")
        self.assertEqual(metrics["end_date"], "2026-04-23")
        self.assertEqual(metrics["date_count"], 3)
        date_metrics = cast(list[dict[str, object]], metrics["dates"])
        self.assertEqual(
            [item["date"] for item in date_metrics],
            ["2026-04-21", "2026-04-22", "2026-04-23"],
        )

    async def test_run_between_dates_rejects_future_dates(self):
        executor = self.make_executor()

        with self.assertRaisesRegex(ValueError, "cannot be later"):
            await executor.run_between_dates("2026-04-25", "2026-04-24")

    async def test_run_between_dates_rejects_start_after_end(self):
        executor = self.make_executor()

        with self.assertRaisesRegex(ValueError, "older date"):
            await executor.run_between_dates("2026-04-23", "2026-04-21")


class TldrBackfillPlanTests(unittest.TestCase):
    def test_explicit_date_is_regenerated_and_recent_empty_dates_are_filled(self):
        plan = plan_tldr_backfill(
            ["2026-10-06", "2026-10-06"],
            ["2026-10-06", "2026-10-05", "2026-10-04"],
        )

        self.assertEqual(
            plan,
            [
                ("2026-10-04", True),
                ("2026-10-05", True),
                ("2026-10-06", False),
            ],
        )

    def test_only_empty_flag_applies_to_explicit_dates(self):
        self.assertEqual(
            plan_tldr_backfill(["2026-10-06"], [], only_empty=True),
            [("2026-10-06", True)],
        )


class SavedTldrBackfillTests(unittest.IsolatedAsyncioTestCase):
    async def test_backfill_updates_tldr_without_rewriting_the_day(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                kept = Paper(
                    source="arxiv",
                    title="Already summarized",
                    authors=["Ada"],
                    abstract="Keep this row.",
                    url="https://example.com/kept",
                    tldr="已有短评",
                    score=1.5,
                    judge_relevance=4,
                    judge_reason="相关",
                    judge_keep=True,
                )
                empty = Paper(
                    source="arxiv",
                    title="Needs a summary",
                    authors=["Ada"],
                    abstract="Empty TLDR.",
                    url="https://example.com/empty",
                    score=2.5,
                    judge_relevance=5,
                    judge_reason="很相关",
                    judge_keep=True,
                )
                today = Paper(
                    source="arxiv",
                    title="Regenerate me",
                    authors=["Ada"],
                    abstract="Old summary should be replaced.",
                    url="https://example.com/today",
                    tldr="旧短评",
                    score=3.5,
                    judge_relevance=3,
                    judge_reason="一般",
                    judge_keep=False,
                )
                await db.save_papers([kept, empty], "2026-10-05")
                await db.save_papers([today], "2026-10-06")
                seen: list[list[str]] = []

                def fake_generate(papers, profile, config):
                    del profile, config
                    seen.append([paper.url for paper in papers])
                    return {
                        paper.url: {
                            "tldr": f"新短评 {paper.url.rsplit('/', 1)[-1]}",
                            "method": "方法",
                            "evidence": "证据",
                            "why_for_me": "相关",
                        }
                        for paper in papers
                    }

                executor = Executor(
                    {
                        "llm": {
                            "api_key": "test-key",
                            "base_url": "http://127.0.0.1:8080/v1",
                            "model": "glm-5.3-flash",
                            "language": "Chinese",
                            "models": {"summarize": "glm-5.3-flash"},
                        }
                    }
                )
                with (
                    patch("arxiv_daily.executor.fetch_corpus", side_effect=AssertionError("arxiv")),
                    patch("arxiv_daily.executor.db.save_papers", side_effect=AssertionError("rewrite")),
                    patch("arxiv_daily.executor.generate_tldrs_batch", side_effect=fake_generate),
                ):
                    result = await executor.backfill_tldrs(
                        [("2026-10-05", True), ("2026-10-06", False)]
                    )

                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["updated"], 2)
                self.assertEqual(seen, [["https://example.com/empty"], ["https://example.com/today"]])
                older = {row["url"]: row for row in await db.get_papers_by_date("2026-10-05")}
                current = {row["url"]: row for row in await db.get_papers_by_date("2026-10-06")}
                self.assertEqual(older["https://example.com/kept"]["tldr"], "已有短评")
                self.assertEqual(older["https://example.com/kept"]["judge_reason"], "相关")
                self.assertEqual(older["https://example.com/empty"]["tldr"], "新短评 empty")
                self.assertEqual(older["https://example.com/empty"]["judge_relevance"], 5)
                self.assertEqual(older["https://example.com/empty"]["score"], 2.5)
                self.assertEqual(current["https://example.com/today"]["tldr"], "新短评 today")
                self.assertEqual(current["https://example.com/today"]["method"], "方法")
                self.assertEqual(current["https://example.com/today"]["judge_reason"], "一般")
                self.assertEqual(current["https://example.com/today"]["score"], 3.5)
                cached = await db.load_candidate_enrichments(["https://example.com/today"])
                self.assertEqual(cached["https://example.com/today"]["tldr"], "新短评 today")
                self.assertIn("glm-5.3-flash", cached["https://example.com/today"]["llm_cache_key"])
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


if __name__ == "__main__":
    unittest.main()
