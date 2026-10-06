import os
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from typing import Any, cast
from unittest.mock import patch
import arxiv_daily.retriever.arxiv_retriever as arxiv_retriever_module
from arxiv_daily import database as db
from arxiv_daily.business_date import announcement_window_utc, submitted_not_after_utc
from arxiv_daily.executor import Executor, _filter_seen_papers
from arxiv_daily.main import main
from arxiv_daily.protocol import CorpusPaper, Paper
from arxiv_daily.retriever.arxiv_retriever import (
    ArxivRetriever,
    _ARXIV_MIN_REQUEST_INTERVAL_SECONDS,
    _run_arxiv_call,
)


def _shanghai_config() -> dict[str, Any]:
    return {"executor": {"timezone": "Asia/Shanghai", "business_date": "2026-10-06"}}


class AnnouncementWindowTests(unittest.TestCase):
    def test_weekday_windows_follow_the_previous_weekday_cutoff(self):
        cases = {
            # Monday mailing is Thursday 14:00 through Friday 14:00 ET.
            "2026-09-14": ("2026-09-10T18:00:00+00:00", "2026-09-11T18:00:00+00:00", "announcement"),
            # Tuesday mailing covers the weekend: Friday 14:00 through Monday 14:00 ET.
            "2026-09-15": ("2026-09-11T18:00:00+00:00", "2026-09-14T18:00:00+00:00", "announcement"),
            "2026-09-16": ("2026-09-14T18:00:00+00:00", "2026-09-15T18:00:00+00:00", "announcement"),
            "2026-09-17": ("2026-09-15T18:00:00+00:00", "2026-09-16T18:00:00+00:00", "announcement"),
            "2026-09-18": ("2026-09-16T18:00:00+00:00", "2026-09-17T18:00:00+00:00", "announcement"),
        }
        for business_date, (start, end, kind) in cases.items():
            window_start, window_end, window_kind, source = announcement_window_utc(business_date)
            self.assertEqual(window_start.isoformat(), start, business_date)
            self.assertEqual(window_end.isoformat(), end, business_date)
            self.assertEqual(window_kind, kind, business_date)
            self.assertEqual(source, business_date)

    def test_weekend_reuses_friday_window(self):
        friday = announcement_window_utc("2026-09-11")
        for business_date in ("2026-09-12", "2026-09-13"):
            window = announcement_window_utc(business_date)
            self.assertEqual(window[0], friday[0], business_date)
            self.assertEqual(window[1], friday[1], business_date)
            self.assertEqual(window[2], "reused", business_date)
            self.assertEqual(window[3], "2026-09-11", business_date)

    def test_labor_day_2026_defers_tuesday_morning_and_widens_wednesday(self):
        monday = announcement_window_utc("2026-09-07")
        self.assertEqual(monday[2], "announcement")
        self.assertEqual(monday[0].isoformat(), "2026-09-03T18:00:00+00:00")
        self.assertEqual(monday[1].isoformat(), "2026-09-04T18:00:00+00:00")

        tuesday = announcement_window_utc("2026-09-08")
        self.assertEqual(tuesday[2], "reused")
        self.assertEqual(tuesday[3], "2026-09-07")
        self.assertEqual(tuesday[0], monday[0])
        self.assertEqual(tuesday[1], monday[1])

        wednesday = announcement_window_utc("2026-09-09")
        self.assertEqual(wednesday[2], "deferred")
        self.assertEqual(wednesday[0].isoformat(), "2026-09-04T18:00:00+00:00")
        self.assertEqual(wednesday[1].isoformat(), "2026-09-08T18:00:00+00:00")

    def test_new_year_deferred_window_uses_eastern_standard_time(self):
        start, end, kind, source = announcement_window_utc("2026-01-05")
        self.assertEqual(kind, "deferred")
        self.assertEqual(source, "2026-01-05")
        self.assertEqual(start.isoformat(), "2025-12-31T19:00:00+00:00")
        self.assertEqual(end.isoformat(), "2026-01-02T19:00:00+00:00")

    def test_window_never_includes_submissions_after_the_business_date(self):
        config = _shanghai_config()
        day = date(2026, 1, 1)
        while day <= date(2026, 10, 6):
            start, end, _kind, source = announcement_window_utc(day)
            cap = submitted_not_after_utc(config, day)
            self.assertLess(start, end, day.isoformat())
            self.assertLessEqual(end, cap, day.isoformat())
            self.assertLessEqual(source, day.isoformat())
            day += timedelta(days=1)


class ArxivPolitenessTests(unittest.TestCase):
    def test_request_slot_is_at_least_three_seconds(self):
        self.assertGreaterEqual(_ARXIV_MIN_REQUEST_INTERVAL_SECONDS, 3)

    def test_retryable_status_uses_exponential_backoff(self):
        calls = {"n": 0}
        delays: list[float] = []

        def func() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429 too many requests")
            return "ok"

        old_wait = arxiv_retriever_module._wait_for_arxiv_request_slot
        old_sleep = arxiv_retriever_module._sleep_for_retry
        arxiv_retriever_module._wait_for_arxiv_request_slot = lambda: None
        arxiv_retriever_module._sleep_for_retry = delays.append
        try:
            result = _run_arxiv_call(func, description="search", max_attempts=5)
        finally:
            arxiv_retriever_module._wait_for_arxiv_request_slot = old_wait
            arxiv_retriever_module._sleep_for_retry = old_sleep

        self.assertEqual(result, "ok")
        self.assertEqual(delays, [20, 40])

    def test_service_unavailable_is_retried(self):
        calls = {"n": 0}

        def func() -> str:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("503 service unavailable")
            return "ok"

        old_wait = arxiv_retriever_module._wait_for_arxiv_request_slot
        old_sleep = arxiv_retriever_module._sleep_for_retry
        arxiv_retriever_module._wait_for_arxiv_request_slot = lambda: None
        arxiv_retriever_module._sleep_for_retry = lambda _seconds: None
        try:
            result = _run_arxiv_call(func, description="search", max_attempts=3)
        finally:
            arxiv_retriever_module._wait_for_arxiv_request_slot = old_wait
            arxiv_retriever_module._sleep_for_retry = old_sleep

        self.assertEqual(result, "ok")
        self.assertEqual(calls["n"], 2)


class AnnouncementFetchTests(unittest.TestCase):
    def _retriever(self) -> ArxivRetriever:
        return ArxivRetriever(
            {
                "executor": {
                    "business_date": "2026-09-14",
                    "timezone": "Asia/Shanghai",
                    "announcement_backfill": True,
                },
                "source": {
                    "arxiv": {
                        "category": ["cs.AI", "cs.CV", "cs.LG", "cs.CL", "cs.RO", "cs.SY"],
                        "recent_days": 1,
                        "recent_max_results": 10,
                        "include_cross_list": False,
                        "announcement_page_size": 200,
                        "announcement_category_max_results": 500,
                        "keyword_query_max_results": 80,
                        "max_keywords": 4,
                    }
                },
            }
        )

    def test_pool_pages_small_and_drops_papers_submitted_after_the_date(self):
        retriever = self._retriever()
        retriever._active_terms = ["humanoid locomotion"]
        calls: list[dict[str, object]] = []

        def fake_search(*args: object, **kwargs: object) -> list[dict[str, object]]:
            calls.append({"args": args, "kwargs": kwargs})
            if len(calls) > 1:
                return []
            return [
                {
                    "title": "In window",
                    "authors": ["Ada"],
                    "summary": "humanoid locomotion controller",
                    "entry_id": "http://arxiv.org/abs/2609.00001v1",
                    "published": "2026-09-10T20:00:00+00:00",
                    "primary_category": "cs.RO",
                    "categories": ["cs.RO"],
                },
                {
                    "title": "After the business date",
                    "authors": ["Ada"],
                    "summary": "humanoid locomotion later",
                    "entry_id": "http://arxiv.org/abs/2609.00002v1",
                    "published": "2026-09-15T00:30:00+00:00",
                    "primary_category": "cs.RO",
                    "categories": ["cs.RO"],
                },
            ]

        retriever._search_arxiv = fake_search  # type: ignore[method-assign]
        window_start, window_end, _kind, _source = announcement_window_utc("2026-09-14")
        # Widen the end so the second paper is inside the query window and the
        # business-date cap is what removes it.
        wide_end = datetime(2026, 9, 16, tzinfo=timezone.utc)
        not_after = submitted_not_after_utc(retriever.config, date(2026, 9, 14))
        papers = retriever._fetch_announcement_pool(
            ["cs.AI", "cs.CV", "cs.LG", "cs.CL", "cs.RO", "cs.SY"],
            window_start,
            wide_end,
            not_after,
        )

        self.assertEqual([paper["entry_id"] for paper in papers], ["http://arxiv.org/abs/2609.00001v1"])
        category_calls = [
            call for call in calls if str(call["args"][0]).startswith("cat:")
        ]
        self.assertEqual(len(category_calls), 6)
        for call in category_calls:
            self.assertLessEqual(int(cast(int, call["args"][1])), 100)
            self.assertLessEqual(int(cast(int, call["kwargs"]["page_size"])), 50)
            self.assertIn("submittedDate:", str(call["args"][0]))
            self.assertGreaterEqual(int(cast(int, call["kwargs"]["max_attempts"])), 3)
        self.assertTrue(any("humanoid locomotion" in str(call["args"][0]) for call in calls))
        self.assertLessEqual(window_end, not_after)

    def test_backfill_retrieval_uses_the_api_window_and_not_rss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                import asyncio

                asyncio.run(db.init_db())
                retriever = self._retriever()
                retriever._corpus = [
                    CorpusPaper(title="Library", abstract="humanoid", added_date=datetime(2026, 9, 1))
                ]
                ranked: list[list[str]] = []

                async def fake_terms() -> tuple[list[str], list[str]]:
                    return ["humanoid locomotion"], []

                def fake_pool(*_args: object, **_kwargs: object) -> list[dict[str, str]]:
                    return [
                        {
                            "title": "Candidate",
                            "summary": "humanoid locomotion",
                            "entry_id": "http://arxiv.org/abs/2609.00003v1",
                            "published": "2026-09-10T20:00:00+00:00",
                        }
                    ]

                def fake_rank(papers: list[dict[str, str]], keywords: list[str]) -> list[dict[str, str]]:
                    ranked.append(keywords)
                    return papers

                retriever._resolve_search_terms = fake_terms  # type: ignore[method-assign]
                retriever._fetch_announcement_pool = fake_pool  # type: ignore[method-assign]
                retriever._rank_candidate_pool = fake_rank  # type: ignore[method-assign]
                retriever._fetch_rss_papers = lambda _categories: (_ for _ in ()).throw(  # type: ignore[method-assign]
                    AssertionError("rss")
                )
                retriever._retrieve_with_adaptive_lookback = lambda *_args, **_kwargs: (_ for _ in ()).throw(  # type: ignore[method-assign]
                    AssertionError("lookback")
                )

                papers = retriever._retrieve_raw_papers()
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir

        self.assertEqual(papers[0]["entry_id"], "http://arxiv.org/abs/2609.00003v1")
        self.assertEqual(ranked, [["humanoid locomotion"]])
        self.assertEqual(
            arxiv_retriever_module.LAST_RETRIEVAL_STATS["announcement_kind"],
            "announcement",
        )
        self.assertEqual(
            arxiv_retriever_module.LAST_RETRIEVAL_STATS["announcement_source_date"],
            "2026-09-14",
        )


class SeenAcrossDatesTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_dates_scope_drops_later_recommendations_and_keeps_corpus(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                await db.save_papers(
                    [
                        Paper(
                            source="arxiv",
                            title="Later",
                            authors=["Ada"],
                            abstract="Later abstract",
                            url="https://arxiv.org/abs/2609.10001v1",
                        )
                    ],
                    "2026-09-20",
                )
                await db.save_papers(
                    [
                        Paper(
                            source="arxiv",
                            title="Earlier",
                            authors=["Ada"],
                            abstract="Earlier abstract",
                            url="http://arxiv.org/abs/2609.10002v1",
                        )
                    ],
                    "2026-09-01",
                )
                candidates = [
                    Paper(
                        source="arxiv",
                        title="Later copy",
                        authors=["Bea"],
                        abstract="Different",
                        url="http://arxiv.org/abs/2609.10001v1",
                    ),
                    Paper(
                        source="arxiv",
                        title="Earlier copy",
                        authors=["Bea"],
                        abstract="Different",
                        url="https://arxiv.org/abs/2609.10002v1",
                    ),
                    Paper(
                        source="arxiv",
                        title="Library paper",
                        authors=["Bea"],
                        abstract="In zotero",
                        url="http://arxiv.org/abs/2609.10003v1",
                    ),
                    Paper(
                        source="arxiv",
                        title="Fresh",
                        authors=["Bea"],
                        abstract="New abstract",
                        url="http://arxiv.org/abs/2609.10004v1",
                    ),
                ]
                corpus = [CorpusPaper(title="Library paper", abstract="In zotero", added_date=datetime(2026, 9, 1))]
                filtered = await _filter_seen_papers(candidates, corpus, "2026-09-14", scope="all")
                self.assertEqual([paper.title for paper in filtered], ["Fresh"])
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir


class BackfillDaysTests(unittest.IsolatedAsyncioTestCase):
    class StubExecutor(Executor):
        ran: list[tuple[str, bool, bool]] = []

        def __init__(self, config: dict[str, Any]):
            super().__init__(config)

        async def run_for_date(self, business_date: str | date, skip_tldr: bool = False) -> list[Paper]:
            del skip_tldr
            day = str(business_date)
            executor_config = self.config.get("executor") or {}
            self.__class__.ran.append(
                (
                    day,
                    bool(executor_config.get("announcement_backfill")),
                    bool(executor_config.get("dedup_all_dates")),
                )
            )
            if day == "2026-07-26":
                raise RuntimeError("arxiv down")
            self.last_run_metrics = {
                "status": "completed",
                "final_recommendations": 1,
                "retrieved_candidates": 4,
                "filtered_candidates": 3,
                "retrieval_stats": {},
            }
            return [
                Paper(
                    source="arxiv",
                    title=f"Paper {day}",
                    authors=[],
                    abstract="Abstract",
                    url=f"https://example.com/{day}",
                )
            ]

    def _executor(self) -> "BackfillDaysTests.StubExecutor":
        return self.StubExecutor(_shanghai_config())

    async def test_runs_oldest_first_skips_existing_and_continues_after_failure(self):
        self.StubExecutor.ran = []
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                await db.save_papers(
                    [
                        Paper(
                            source="arxiv",
                            title="Already",
                            authors=[],
                            abstract="Kept",
                            url="https://example.com/kept",
                        )
                    ],
                    "2026-07-09",
                )
                executor = self._executor()
                result = await executor.backfill_days(
                    ["2026-08-11", "2026-07-26", "2026-07-09", "2026-07-26"]
                )
                kept = await db.get_papers_by_date("2026-07-09")
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir

        self.assertEqual(kept[0]["title"], "Already")
        self.assertEqual(
            self.StubExecutor.ran,
            [("2026-07-26", True, True), ("2026-08-11", True, True)],
        )
        self.assertEqual(result["status"], "partial")
        dates = cast(list[dict[str, object]], result["dates"])
        self.assertEqual([item["date"] for item in dates], ["2026-07-09", "2026-07-26", "2026-08-11"])
        self.assertEqual(dates[0]["status"], "skipped")
        self.assertEqual(dates[0]["recommendations"], 1)
        self.assertEqual(dates[1]["status"], "failed")
        self.assertIn("arxiv down", str(dates[1]["error"]))
        self.assertEqual(dates[2]["status"], "completed")
        self.assertEqual(dates[2]["recommendations"], 1)

    async def test_force_reruns_and_future_dates_are_rejected(self):
        self.StubExecutor.ran = []
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                await db.init_db()
                await db.save_papers(
                    [
                        Paper(
                            source="arxiv",
                            title="Already",
                            authors=[],
                            abstract="Kept",
                            url="https://example.com/kept",
                        )
                    ],
                    "2026-07-09",
                )
                executor = self._executor()
                await executor.backfill_days(["2026-07-09"], force=True)
                with self.assertRaisesRegex(ValueError, "cannot be later"):
                    await executor.backfill_days(["2026-10-07"])
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir

        self.assertEqual(self.StubExecutor.ran, [("2026-07-09", True, True)])


class BackfillDayCliTests(unittest.TestCase):
    def test_cli_passes_repeated_dates_and_force(self):
        recorded: dict[str, object] = {}

        class FakeExecutor:
            def __init__(self, config: dict[str, Any]):
                recorded["config"] = config

            async def backfill_days(self, dates: list[str], force: bool = False) -> dict[str, object]:
                recorded["dates"] = list(dates)
                recorded["force"] = force
                return {"status": "completed", "dates": []}

        with (
            patch("arxiv_daily.main.load_config", return_value={"executor": {"timezone": "Asia/Shanghai"}}),
            patch("arxiv_daily.main.Executor", FakeExecutor),
        ):
            main(["backfill-day", "--date", "2026-07-26", "--date", "2026-07-09", "--force"])

        self.assertEqual(recorded["dates"], ["2026-07-26", "2026-07-09"])
        self.assertIs(recorded["force"], True)

    def test_cli_rejects_a_bad_date(self):
        with self.assertRaises(SystemExit):
            main(["backfill-day", "--date", "07-09"])

    def test_cli_exits_when_a_day_failed(self):
        class FakeExecutor:
            def __init__(self, config: dict[str, Any]):
                del config

            async def backfill_days(self, dates: list[str], force: bool = False) -> dict[str, object]:
                del dates, force
                return {"status": "partial", "dates": []}

        with (
            patch("arxiv_daily.main.load_config", return_value={}),
            patch("arxiv_daily.main.Executor", FakeExecutor),
            self.assertRaises(SystemExit),
        ):
            main(["backfill-day", "--date", "2026-07-09"])
