import unittest
from typing import Any, cast, override

from datetime import datetime

from arxiv_daily.protocol import CorpusPaper
from arxiv_daily.retriever.arxiv_retriever import (
    ArxivRetriever,
    RawPaper,
    _extract_local_keywords_from_corpus,
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
