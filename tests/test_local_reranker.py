import unittest

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


if __name__ == "__main__":
    unittest.main()
