import re
import threading
import time
import unittest

from arxiv_daily.coarse import diverse_shortlist, order_by_bm25, score_coarse_papers
from arxiv_daily.lexical import diverse_indices, rank_items_by_query
from arxiv_daily.llm import get_llm_usage, reset_llm_usage
from arxiv_daily.protocol import Paper


def _config(**llm: object) -> dict:
    merged = {
        "model": "fallback",
        "coarse_batch_size": 40,
        "coarse_concurrency": 2,
        "max_concurrent_requests": 2,
        "coarse_abstract_chars": 80,
        "coarse_max_tokens": 400,
        "coarse_duplicate_jaccard": 0.8,
        "models": {"coarse": "grok-4.3"},
    }
    merged.update(llm)
    return {
        "executor": {"max_paper_num": 10, "judge_pool_size": 24},
        "reranker": {"mmr_lambda": 0.78},
        "llm": merged,
    }


def _paper(index: int, title: str | None = None, bm25: float = 0.0) -> Paper:
    return Paper(
        source="arxiv",
        title=title or f"Distinct topic {index} about widgets",
        authors=[],
        abstract=f"Abstract {index} discusses a separate method.",
        url=f"http://arxiv.org/abs/{index}",
        bm25_score=bm25,
    )


def _ids_from_prompt(user: str) -> list[str]:
    return re.findall(r"(?m)^(\d+)\. ", user)


class CoarseScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_llm_usage()

    def test_batches_run_with_a_concurrency_cap(self):
        state = {"current": 0, "peak": 0}
        lock = threading.Lock()

        def caller(config, **kwargs):
            del config
            with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            time.sleep(0.15)
            with lock:
                state["current"] -= 1
            return {"items": [{"id": local_id, "score": 4} for local_id in _ids_from_prompt(kwargs["user"])]}

        papers = [_paper(index) for index in range(120)]
        outcome = score_coarse_papers(papers, {"summary": "robots"}, _config(), caller=caller)

        self.assertEqual(outcome.requests, 3)
        self.assertEqual(outcome.failed_batches, 0)
        self.assertEqual(state["peak"], 2)
        self.assertIsNotNone(outcome.scores)
        self.assertEqual(len(outcome.scores or {}), 120)
        self.assertEqual((outcome.scores or {})["http://arxiv.org/abs/0"], 4.0)

    def test_total_failure_falls_back_and_records_an_llm_error(self):
        def caller(config, **kwargs):
            del config, kwargs
            return None

        outcome = score_coarse_papers([_paper(1), _paper(2)], {}, _config(), caller=caller)

        self.assertIsNone(outcome.scores)
        self.assertGreaterEqual(get_llm_usage()["llm_error_count"], 1)
        self.assertIn("BM25", get_llm_usage()["llm_warning"])

    def test_one_failed_batch_keeps_the_successful_scores(self):
        calls = {"count": 0}

        def caller(config, **kwargs):
            del config
            calls["count"] += 1
            if calls["count"] == 1:
                return None
            return {"items": [{"id": local_id, "score": 8} for local_id in _ids_from_prompt(kwargs["user"])]}

        papers = [_paper(index) for index in range(120)]
        outcome = score_coarse_papers(papers, {}, _config(), caller=caller)

        self.assertIsNotNone(outcome.scores)
        self.assertEqual(outcome.failed_batches, 1)
        self.assertEqual(len(outcome.scores or {}), 80)

    def test_scores_are_clamped(self):
        def caller(config, **kwargs):
            del config
            local_id = _ids_from_prompt(kwargs["user"])[0]
            return {"items": [{"id": local_id, "score": 42}]}

        outcome = score_coarse_papers([_paper(3)], {}, _config(), caller=caller)

        self.assertEqual((outcome.scores or {})["http://arxiv.org/abs/3"], 10.0)


class DiversityTests(unittest.TestCase):
    def test_title_similarity_skips_a_near_duplicate(self):
        texts = [
            "Humanoid whole body control with balance",
            "Humanoid whole body control with balance",
            "Protein folding language model",
        ]
        order = diverse_indices(texts, [10.0, 9.0, 6.0], 2, lam=0.78, duplicate_jaccard=0.8)

        self.assertEqual(order, [0, 2])

    def test_bm25_fallback_orders_by_bm25_score(self):
        papers = [_paper(1, bm25=0.2), _paper(2, bm25=0.9), _paper(3, bm25=0.4)]

        chosen = order_by_bm25(papers, 2)

        self.assertEqual([paper.url for paper in chosen], [
            "http://arxiv.org/abs/2",
            "http://arxiv.org/abs/3",
        ])

    def test_shortlist_uses_coarse_scores(self):
        papers = [_paper(index, title=f"Separate subject {index} zeta") for index in range(3)]
        scores = {paper.url: float(index + 1) for index, paper in enumerate(papers)}

        chosen = diverse_shortlist(papers, scores, _config())

        self.assertEqual(chosen[0].url, papers[-1].url)
        self.assertEqual(chosen[0].score, 3.0)


class QueryRankTests(unittest.TestCase):
    def test_bm25_recall_prefers_the_matching_item(self):
        items = [
            {"title": "Cooking pasta", "abstract": "Boil water and salt the pot."},
            {"title": "Humanoid locomotion", "abstract": "A robot walks with a whole body controller."},
        ]

        ranked = rank_items_by_query("humanoid robot walking controller", items, top_k=1)

        self.assertEqual(ranked[0][0], 1)
        self.assertGreater(ranked[0][1], 0)


if __name__ == "__main__":
    unittest.main()
