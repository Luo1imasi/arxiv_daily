import unittest

from arxiv_daily.retriever.arxiv_retriever import ArxivRetriever


class StubArxivRetriever(ArxivRetriever):
    def __init__(self, config, windows, rankings):
        super().__init__(config)
        self.windows = windows
        self.rankings = rankings
        self.calls = []

    def _collect_candidate_pool(self, categories, *, start_days_ago=0, end_days_ago=None):
        key = (start_days_ago, end_days_ago)
        self.calls.append(key)
        return list(self.windows.get(key, []))

    def _rank_candidate_pool(self, papers, keywords):
        paper_ids = tuple(sorted(paper["entry_id"] for paper in papers))
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
        windows = {
            (0, 7): [{"entry_id": "a", "title": "A", "summary": "..."}],
            (7, 14): [{"entry_id": "b", "title": "B", "summary": "..."}],
        }
        rankings = {
            ("a",): [{"entry_id": "a", "title": "A", "summary": "...", "lookback_score": 3.0}],
            ("b",): [{"entry_id": "b", "title": "B", "summary": "...", "lookback_score": 8.0}],
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
        windows = {
            (0, 7): [{"entry_id": "a", "title": "A", "summary": "..."}],
            (7, 14): [{"entry_id": "b", "title": "B", "summary": "..."}],
            (14, 21): [{"entry_id": "c", "title": "C", "summary": "..."}],
        }
        rankings = {
            ("a",): [{"entry_id": "a", "title": "A", "summary": "...", "lookback_score": 3.0}],
            ("b",): [{"entry_id": "b", "title": "B", "summary": "...", "lookback_score": 5.0}],
            ("c",): [{"entry_id": "c", "title": "C", "summary": "...", "lookback_score": 8.5}],
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
        windows = {
            (0, 7): [{"entry_id": "a", "title": "A", "summary": "..."}],
        }
        rankings = {
            ("a",): [{"entry_id": "a", "title": "A", "summary": "...", "lookback_score": 8.0}],
        }

        retriever = StubArxivRetriever(config, windows, rankings)
        ranked = retriever._retrieve_with_adaptive_lookback(["cs.AI"], ["agent"], windows[(0, 7)])

        self.assertEqual(retriever.calls, [])
        self.assertEqual([paper["entry_id"] for paper in ranked], ["a"])


if __name__ == "__main__":
    unittest.main()
