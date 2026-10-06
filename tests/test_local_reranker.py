import unittest
from types import SimpleNamespace

import numpy as np

from arxiv_daily.reranker import local as local_module
from arxiv_daily.reranker.local import LocalReranker, _display_scores


class DisplayScoreTests(unittest.TestCase):
    def test_display_scores_expand_relative_differences(self):
        scores = _display_scores([0.21, 0.20, 0.19])

        self.assertEqual(scores[0], 10.0)
        self.assertAlmostEqual(scores[1], 7.0, places=5)
        self.assertEqual(scores[2], 4.0)

    def test_display_scores_use_neutral_value_when_all_scores_match(self):
        self.assertEqual(_display_scores([0.2, 0.2]), [5.0, 5.0])

    def test_display_scores_use_neutral_value_for_single_score(self):
        self.assertEqual(_display_scores([0.2]), [5.0])


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


class FeatureAlignmentTests(unittest.TestCase):
    def test_missing_embedding_does_not_drop_rows(self):
        reranker = LocalReranker({"reranker": {"model": "stub", "encode_kwargs": {}}})
        items = [
            SimpleNamespace(title="one", abstract="a"),
            SimpleNamespace(title="two", abstract="b"),
            SimpleNamespace(title="three", abstract="c"),
        ]

        def load_cached(cached_items, model_key):
            del cached_items, model_key
            return [0], [np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)]

        class Encoder:
            def encode(self, texts, **kwargs):
                del kwargs
                return np.ones((max(len(texts) - 1, 0), 4), dtype=np.float32)

        old_load = local_module._load_cached_features
        local_module._load_cached_features = load_cached
        try:
            features = reranker._get_item_features(
                Encoder(),
                items,
                ["one", "two", "three"],
                "stub",
                {},
                log_prefix="candidate",
            )
        finally:
            local_module._load_cached_features = old_load

        self.assertEqual(features.shape, (3, 4))
        self.assertAlmostEqual(float(np.linalg.norm(features[2])), 0.0)


if __name__ == "__main__":
    unittest.main()
