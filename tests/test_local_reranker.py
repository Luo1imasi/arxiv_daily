import unittest

from arxiv_daily.protocol import Paper
from arxiv_daily.reranker.local import LocalReranker


class RecencyScoreTests(unittest.TestCase):
    def test_recency_score_uses_configured_business_date(self):
        reranker = LocalReranker(
            {
                "executor": {"business_date": "2026-04-24"},
                "reranker": {"recency_half_life_days": 7.0},
            }
        )

        same_day = reranker._get_recency_score("2026-04-24")
        previous_week = reranker._get_recency_score("2026-04-17")

        self.assertGreater(same_day, 0.99)
        self.assertLess(previous_week, same_day)


class LexicalRerankTests(unittest.TestCase):
    def test_rerank_orders_by_bm25_without_embeddings(self):
        reranker = LocalReranker({"executor": {"max_paper_num": 10, "judge_pool_size": 24}})
        papers = [
            Paper(source="arxiv", title="low", authors=[], abstract="", url="low", bm25_score=0.1),
            Paper(source="arxiv", title="high", authors=[], abstract="", url="high", bm25_score=0.9),
        ]

        ranked = reranker.rerank(papers, [])

        self.assertEqual([paper.url for paper in ranked], ["high", "low"])


if __name__ == "__main__":
    unittest.main()
