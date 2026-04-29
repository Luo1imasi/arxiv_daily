import unittest
import asyncio
import os
import tempfile
from typing import Any, cast, override

from datetime import datetime, timezone

import arxiv_daily.retriever.arxiv_retriever as arxiv_retriever_module
from arxiv_daily import database as db
from arxiv_daily.protocol import CorpusPaper
from arxiv_daily.retriever.arxiv_retriever import (
    ArxivRetriever,
    RawPaper,
    _business_window_utc,
    _extract_local_keywords_from_corpus,
    _looks_like_oversized_arxiv_request,
    _matches_category_policy,
    _published_in_window,
)


def _raw_paper(**values: object) -> RawPaper:
    return dict(values)


class StubArxivRetriever(ArxivRetriever):
    def __init__(
        self,
        config: dict[str, Any],
        windows: dict[tuple[int, int | None], list[RawPaper]],
        rankings: dict[tuple[str, ...], list[RawPaper]],
    ):
        super().__init__(config)
        self.windows = windows
        self.rankings = rankings
        self.calls: list[tuple[int, int | None]] = []

    @override
    def _collect_candidate_pool(
        self,
        categories: list[str],
        *,
        start_days_ago: int = 0,
        end_days_ago: int | None = None,
    ) -> list[RawPaper]:
        key = (start_days_ago, end_days_ago)
        self.calls.append(key)
        return list(self.windows.get(key, []))

    @override
    def _rank_candidate_pool(self, papers: list[RawPaper], keywords: list[str]) -> list[RawPaper]:
        paper_ids = tuple(sorted(cast(str, paper["entry_id"]) for paper in papers))
        return list(self.rankings.get(paper_ids, []))


class AdaptiveLookbackTests(unittest.TestCase):
    def test_business_window_uses_configured_timezone(self):
        config = {
            "executor": {"business_date": "2026-04-22", "timezone": "Asia/Shanghai"},
            "source": {"arxiv": {"category": ["cs.AI"]}},
        }

        window_start, window_end = _business_window_utc(config, start_days_ago=0, end_days_ago=1)

        self.assertEqual(window_start.isoformat(), "2026-04-21T16:00:00+00:00")
        self.assertEqual(window_end.isoformat(), "2026-04-22T16:00:00+00:00")

    def test_published_window_is_half_open(self):
        window_start = datetime(2026, 4, 22, tzinfo=timezone.utc)
        window_end = datetime(2026, 4, 23, tzinfo=timezone.utc)

        self.assertTrue(_published_in_window(window_start, window_start, window_end))
        self.assertFalse(_published_in_window(window_end, window_start, window_end))

    def test_category_policy_excludes_cross_list_without_primary_match(self):
        paper = _raw_paper(
            entry_id="cross",
            primary_category="math.OC",
            categories=["math.OC", "cs.AI"],
        )

        self.assertFalse(_matches_category_policy(paper, ["cs.AI"], include_cross=False))
        self.assertTrue(_matches_category_policy(paper, ["cs.AI"], include_cross=True))

    def test_candidate_cache_key_includes_business_date(self):
        config = {
            "executor": {"business_date": "2026-04-24"},
            "source": {"arxiv": {"category": ["cs.AI"]}},
        }
        first = StubArxivRetriever(config, {}, {})
        second = StubArxivRetriever(
            {
                "executor": {"business_date": "2026-04-23"},
                "source": {"arxiv": {"category": ["cs.AI"]}},
            },
            {},
            {},
        )

        self.assertNotEqual(
            first._candidate_cache_key(["cs.AI"], start_days_ago=0, end_days_ago=7),
            second._candidate_cache_key(["cs.AI"], start_days_ago=0, end_days_ago=7),
        )

    def test_expands_lookback_when_top_scores_are_low(self):
        config = {
            "source": {
                "arxiv": {
                    "category": ["cs.AI"],
                    "recent_days": 7,
                    "max_recent_days": 21,
                    "lookback_score_threshold": 7.0,
                }
            }
        }
        windows: dict[tuple[int, int | None], list[RawPaper]] = {
            (0, 7): [_raw_paper(entry_id="a", title="A", summary="...")],
            (7, 14): [_raw_paper(entry_id="b", title="B", summary="...")],
        }
        rankings: dict[tuple[str, ...], list[RawPaper]] = {
            ("a",): [_raw_paper(entry_id="a", title="A", summary="...", lookback_score=3.0)],
            ("b",): [_raw_paper(entry_id="b", title="B", summary="...", lookback_score=8.0)],
        }

        retriever = StubArxivRetriever(config, windows, rankings)
        ranked = retriever._retrieve_with_adaptive_lookback(["cs.AI"], ["agent"], windows[(0, 7)])

        self.assertEqual(retriever.calls, [(7, 14)])
        self.assertEqual([paper["entry_id"] for paper in ranked], ["b"])

    def test_skips_low_quality_window_in_following_expansion(self):
        config = {
            "source": {
                "arxiv": {
                    "category": ["cs.AI"],
                    "recent_days": 7,
                    "max_recent_days": 28,
                    "lookback_score_threshold": 7.0,
                }
            }
        }
        windows: dict[tuple[int, int | None], list[RawPaper]] = {
            (0, 7): [_raw_paper(entry_id="a", title="A", summary="...")],
            (7, 14): [_raw_paper(entry_id="b", title="B", summary="...")],
            (14, 21): [_raw_paper(entry_id="c", title="C", summary="...")],
        }
        rankings: dict[tuple[str, ...], list[RawPaper]] = {
            ("a",): [_raw_paper(entry_id="a", title="A", summary="...", lookback_score=3.0)],
            ("b",): [_raw_paper(entry_id="b", title="B", summary="...", lookback_score=5.0)],
            ("c",): [_raw_paper(entry_id="c", title="C", summary="...", lookback_score=8.5)],
        }

        retriever = StubArxivRetriever(config, windows, rankings)
        ranked = retriever._retrieve_with_adaptive_lookback(["cs.AI"], ["agent"], windows[(0, 7)])

        self.assertEqual(retriever.calls, [(7, 14), (14, 21)])
        self.assertEqual([paper["entry_id"] for paper in ranked], ["c"])

    def test_stops_without_expanding_when_initial_scores_are_good(self):
        config = {
            "source": {
                "arxiv": {
                    "category": ["cs.AI"],
                    "recent_days": 7,
                    "max_recent_days": 21,
                    "lookback_score_threshold": 7.0,
                }
            }
        }
        windows: dict[tuple[int, int | None], list[RawPaper]] = {
            (0, 7): [_raw_paper(entry_id="a", title="A", summary="...")],
        }
        rankings: dict[tuple[str, ...], list[RawPaper]] = {
            ("a",): [_raw_paper(entry_id="a", title="A", summary="...", lookback_score=8.0)],
        }

        retriever = StubArxivRetriever(config, windows, rankings)
        ranked = retriever._retrieve_with_adaptive_lookback(["cs.AI"], ["agent"], windows[(0, 7)])

        self.assertEqual(retriever.calls, [])
        self.assertEqual([paper["entry_id"] for paper in ranked], ["a"])

    def test_single_strong_top_paper_can_stop_lookback(self):
        config = {
            "source": {
                "arxiv": {
                    "category": ["cs.AI"],
                    "recent_days": 7,
                    "max_recent_days": 21,
                    "lookback_score_threshold": 7.0,
                }
            }
        }
        retriever = ArxivRetriever(config)

        should_expand = retriever._should_expand_lookback(
            [_raw_paper(entry_id="a", title="A", summary="...", lookback_score=8.0)]
        )

        self.assertFalse(should_expand)

    def test_expands_when_too_few_top_papers_are_strong(self):
        config = {
            "source": {
                "arxiv": {
                    "category": ["cs.AI"],
                    "recent_days": 7,
                    "max_recent_days": 21,
                    "lookback_score_threshold": 7.0,
                }
            }
        }
        windows: dict[tuple[int, int | None], list[RawPaper]] = {
            (0, 7): [
                _raw_paper(entry_id="a", title="A", summary="..."),
                _raw_paper(entry_id="b", title="B", summary="..."),
                _raw_paper(entry_id="c", title="C", summary="..."),
                _raw_paper(entry_id="d", title="D", summary="..."),
            ],
            (7, 14): [_raw_paper(entry_id="e", title="E", summary="...")],
        }
        rankings: dict[tuple[str, ...], list[RawPaper]] = {
            ("a", "b", "c", "d"): [
                _raw_paper(entry_id="a", title="A", summary="...", lookback_score=10.0),
                _raw_paper(entry_id="b", title="B", summary="...", lookback_score=6.8),
                _raw_paper(entry_id="c", title="C", summary="...", lookback_score=6.8),
                _raw_paper(entry_id="d", title="D", summary="...", lookback_score=6.8),
            ],
            ("e",): [_raw_paper(entry_id="e", title="E", summary="...", lookback_score=8.0)],
        }

        retriever = StubArxivRetriever(config, windows, rankings)
        ranked = retriever._retrieve_with_adaptive_lookback(["cs.AI"], ["agent"], windows[(0, 7)])

        self.assertEqual(retriever.calls, [(7, 14)])
        self.assertEqual([paper["entry_id"] for paper in ranked], ["e"])

    def test_rss_candidates_are_filtered_to_business_date_window(self):
        config = {
            "executor": {"business_date": "2026-04-22", "timezone": "UTC"},
            "source": {
                "arxiv": {
                    "category": ["cs.AI"],
                    "recent_days": 1,
                    "max_recent_days": 7,
                }
            },
        }
        retriever = ArxivRetriever(config)
        current_paper = _raw_paper(
            title="Current paper",
            authors=[],
            summary="Current abstract",
            entry_id="https://arxiv.org/abs/current",
            pdf_url=None,
            code_url=None,
            published=datetime(2026, 4, 22, 12, tzinfo=timezone.utc).isoformat(),
        )
        future_paper = _raw_paper(
            title="Future paper",
            authors=[],
            summary="Future abstract",
            entry_id="https://arxiv.org/abs/future",
            pdf_url=None,
            code_url=None,
            published=datetime(2026, 4, 24, 12, tzinfo=timezone.utc).isoformat(),
        )

        retriever._fetch_rss_paper_ids = lambda categories: ["current", "future"]
        retriever._fetch_papers_by_ids = lambda paper_ids: cast(
            Any, [current_paper, future_paper]
        )
        retriever._fetch_recent_category_papers = lambda *args, **kwargs: []

        papers = retriever._collect_candidate_pool(["cs.AI"])

        self.assertEqual([paper["entry_id"] for paper in papers], ["https://arxiv.org/abs/current"])

    def test_recent_category_query_uses_submitted_date_window_and_cross_filter(self):
        config = {
            "executor": {"business_date": "2026-04-22", "timezone": "UTC"},
            "source": {
                "arxiv": {
                    "category": ["cs.AI"],
                    "recent_days": 1,
                    "recent_max_results": 10,
                    "include_cross_list": False,
                }
            },
        }
        retriever = ArxivRetriever(config)
        captured: dict[str, object] = {}

        class SortCriterion:
            SubmittedDate = "submittedDate"

        class SortOrder:
            Descending = "descending"

        class Search:
            def __init__(self, **kwargs: object):
                captured["search"] = kwargs

        class Client:
            def __init__(self, **kwargs: object):
                captured["client"] = kwargs

            def results(self, search: object):
                del search
                return iter(
                    [
                        _raw_paper(
                            title="Primary match",
                            authors=[],
                            summary="Abstract",
                            entry_id="primary",
                            published="2026-04-22T12:00:00+00:00",
                            primary_category="cs.AI",
                            categories=["cs.AI"],
                        ),
                        _raw_paper(
                            title="Cross-listed",
                            authors=[],
                            summary="Abstract",
                            entry_id="cross",
                            published="2026-04-22T12:00:00+00:00",
                            primary_category="math.OC",
                            categories=["math.OC", "cs.AI"],
                        ),
                    ]
                )

        old_arxiv = arxiv_retriever_module._ARXIV
        old_wait = arxiv_retriever_module._wait_for_arxiv_request_slot
        arxiv_retriever_module._ARXIV = cast(
            Any,
            type(
                "ArxivStub",
                (),
                {
                    "Search": Search,
                    "Client": Client,
                    "SortCriterion": SortCriterion,
                    "SortOrder": SortOrder,
                },
            ),
        )
        arxiv_retriever_module._wait_for_arxiv_request_slot = lambda: None
        try:
            papers = retriever._fetch_recent_category_papers(["cs.AI"])
        finally:
            arxiv_retriever_module._ARXIV = old_arxiv
            arxiv_retriever_module._wait_for_arxiv_request_slot = old_wait

        search_kwargs = cast(dict[str, object], captured["search"])
        self.assertIn("submittedDate:[202604220000 TO 202604230000]", cast(str, search_kwargs["query"]))
        self.assertEqual(search_kwargs["sort_order"], "descending")
        self.assertEqual([paper["entry_id"] for paper in papers], ["primary"])

    def test_recent_category_query_does_not_exceed_scoped_business_date(self):
        captured_queries: list[str] = []

        class SortCriterion:
            SubmittedDate = "submittedDate"

        class SortOrder:
            Descending = "descending"

        class Search:
            def __init__(self, **kwargs: object):
                captured_queries.append(cast(str, kwargs["query"]))

        class Client:
            def __init__(self, **kwargs: object):
                pass

            def results(self, search: object):
                del search
                return iter([])

        old_arxiv = arxiv_retriever_module._ARXIV
        old_wait = arxiv_retriever_module._wait_for_arxiv_request_slot
        arxiv_retriever_module._ARXIV = cast(
            Any,
            type(
                "ArxivStub",
                (),
                {
                    "Search": Search,
                    "Client": Client,
                    "SortCriterion": SortCriterion,
                    "SortOrder": SortOrder,
                },
            ),
        )
        arxiv_retriever_module._wait_for_arxiv_request_slot = lambda: None
        try:
            for business_date in ["2026-04-21", "2026-04-22"]:
                retriever = ArxivRetriever(
                    {
                        "executor": {"business_date": business_date, "timezone": "UTC"},
                        "source": {
                            "arxiv": {
                                "category": ["cs.AI"],
                                "recent_days": 1,
                                "recent_max_results": 10,
                            }
                        },
                    }
                )
                retriever._fetch_recent_category_papers(["cs.AI"])
        finally:
            arxiv_retriever_module._ARXIV = old_arxiv
            arxiv_retriever_module._wait_for_arxiv_request_slot = old_wait

        self.assertIn("submittedDate:[202604210000 TO 202604220000]", captured_queries[0])
        self.assertIn("submittedDate:[202604220000 TO 202604230000]", captured_queries[1])

    def test_empty_candidate_cache_is_cache_hit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_data_dir = os.environ.get("ARXIV_DAILY_DATA")
            os.environ["ARXIV_DAILY_DATA"] = tmpdir
            try:
                asyncio.run(db.init_db())
                config = {
                    "executor": {"business_date": "2026-04-22", "timezone": "UTC"},
                    "source": {
                        "arxiv": {
                            "category": ["cs.AI"],
                            "recent_days": 1,
                            "candidate_cache_ttl_minutes": 30,
                        }
                    },
                }
                retriever = ArxivRetriever(config)
                calls = {"recent": 0}

                def fake_recent(*args: object, **kwargs: object) -> list[RawPaper]:
                    del args, kwargs
                    calls["recent"] += 1
                    return []

                retriever._fetch_rss_paper_ids = lambda categories: []
                retriever._fetch_recent_category_papers = fake_recent

                first = retriever._collect_candidate_pool(["cs.AI"])
                second = retriever._collect_candidate_pool(["cs.AI"])

                self.assertEqual(first, [])
                self.assertEqual(second, [])
                self.assertEqual(calls["recent"], 1)
            finally:
                if old_data_dir is None:
                    os.environ.pop("ARXIV_DAILY_DATA", None)
                else:
                    os.environ["ARXIV_DAILY_DATA"] = old_data_dir

    def test_fetch_papers_by_ids_uses_configurable_large_batches(self):
        batch_lengths: list[int] = []
        max_results_values: list[int] = []
        wait_calls = {"count": 0}

        class Search:
            def __init__(self, id_list: list[str], max_results: int):
                self.id_list = id_list
                self.max_results = max_results

        class Client:
            def __init__(self, **kwargs: object):
                pass

            def results(self, search: Search):
                batch_lengths.append(len(search.id_list))
                max_results_values.append(search.max_results)
                return iter([])

        old_arxiv = arxiv_retriever_module._ARXIV
        old_wait = arxiv_retriever_module._wait_for_arxiv_request_slot
        arxiv_retriever_module._ARXIV = cast(
            Any,
            type("ArxivStub", (), {"Search": Search, "Client": Client}),
        )
        arxiv_retriever_module._wait_for_arxiv_request_slot = lambda: wait_calls.__setitem__(
            "count", wait_calls["count"] + 1
        )
        try:
            retriever = ArxivRetriever(
                {
                    "executor": {"arxiv_id_batch_size": 200},
                    "source": {"arxiv": {"category": ["cs.AI"]}},
                }
            )
            retriever._fetch_papers_by_ids([str(index) for index in range(450)])
        finally:
            arxiv_retriever_module._ARXIV = old_arxiv
            arxiv_retriever_module._wait_for_arxiv_request_slot = old_wait

        self.assertEqual(batch_lengths, [200, 200, 50])
        self.assertEqual(max_results_values, [200, 200, 50])
        self.assertEqual(wait_calls["count"], 3)

    def test_fetch_papers_by_ids_splits_failed_batches(self):
        batch_ids: list[list[str]] = []
        wait_calls = {"count": 0}

        class Search:
            def __init__(self, id_list: list[str], max_results: int):
                self.id_list = id_list
                self.max_results = max_results

        class Client:
            def __init__(self, **kwargs: object):
                pass

            def results(self, search: Search):
                batch_ids.append(list(search.id_list))
                if len(search.id_list) > 2:
                    raise RuntimeError("414 URL too long")
                return iter(search.id_list)

        old_arxiv = arxiv_retriever_module._ARXIV
        old_wait = arxiv_retriever_module._wait_for_arxiv_request_slot
        arxiv_retriever_module._ARXIV = cast(
            Any,
            type("ArxivStub", (), {"Search": Search, "Client": Client}),
        )
        arxiv_retriever_module._wait_for_arxiv_request_slot = lambda: wait_calls.__setitem__(
            "count", wait_calls["count"] + 1
        )
        try:
            retriever = ArxivRetriever(
                {
                    "executor": {"arxiv_id_batch_size": 4},
                    "source": {"arxiv": {"category": ["cs.AI"]}},
                }
            )
            papers = retriever._fetch_papers_by_ids(["a", "b", "c", "d"])
        finally:
            arxiv_retriever_module._ARXIV = old_arxiv
            arxiv_retriever_module._wait_for_arxiv_request_slot = old_wait

        self.assertEqual(batch_ids, [["a", "b", "c", "d"], ["a", "b"], ["c", "d"]])
        self.assertEqual(papers, ["a", "b", "c", "d"])
        self.assertEqual(wait_calls["count"], 3)

    def test_fetch_papers_by_ids_does_not_split_regular_upstream_failures(self):
        batch_ids: list[list[str]] = []
        wait_calls = {"count": 0}

        class Search:
            def __init__(self, id_list: list[str], max_results: int):
                self.id_list = id_list
                self.max_results = max_results

        class Client:
            def __init__(self, **kwargs: object):
                pass

            def results(self, search: Search):
                batch_ids.append(list(search.id_list))
                raise RuntimeError("503 Service Unavailable")

        old_arxiv = arxiv_retriever_module._ARXIV
        old_wait = arxiv_retriever_module._wait_for_arxiv_request_slot
        arxiv_retriever_module._ARXIV = cast(
            Any,
            type("ArxivStub", (), {"Search": Search, "Client": Client}),
        )
        arxiv_retriever_module._wait_for_arxiv_request_slot = lambda: wait_calls.__setitem__(
            "count", wait_calls["count"] + 1
        )
        try:
            retriever = ArxivRetriever(
                {
                    "executor": {"arxiv_id_batch_size": 4},
                    "source": {"arxiv": {"category": ["cs.AI"]}},
                }
            )
            papers = retriever._fetch_papers_by_ids(["a", "b", "c", "d"])
        finally:
            arxiv_retriever_module._ARXIV = old_arxiv
            arxiv_retriever_module._wait_for_arxiv_request_slot = old_wait

        self.assertEqual(batch_ids, [["a", "b", "c", "d"]])
        self.assertEqual(papers, [])
        self.assertEqual(wait_calls["count"], 1)

    def test_oversized_arxiv_request_detection_matches_url_length_errors(self):
        self.assertTrue(_looks_like_oversized_arxiv_request(RuntimeError("414")))
        self.assertTrue(_looks_like_oversized_arxiv_request(RuntimeError("URI Too Long")))
        self.assertTrue(_looks_like_oversized_arxiv_request(RuntimeError("URL too long")))
        self.assertFalse(_looks_like_oversized_arxiv_request(RuntimeError("503 Service Unavailable")))


class LocalKeywordExtractionTests(unittest.TestCase):
    def test_extract_local_keywords_handles_first_seen_terms(self):
        corpus = [
            CorpusPaper(
                title="Propose Better Agents",
                abstract="We propose a better planning agent architecture.",
                added_date=datetime(2026, 4, 22),
            )
        ]

        keywords = _extract_local_keywords_from_corpus(corpus, limit=10)

        self.assertIn("propose", keywords)


if __name__ == "__main__":
    unittest.main()
