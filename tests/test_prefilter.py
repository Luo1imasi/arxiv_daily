import unittest
from types import SimpleNamespace

from unittest.mock import patch

from arxiv_daily.executor import _arxiv_url_aliases, _config_with_retrieval_context
from arxiv_daily.lexical import (
    BROAD_TERM_WEIGHT,
    CORE_TERM_WEIGHT,
    assign_term_weights,
    author_overlap_bonus,
    corpus_author_keys,
    feedback_negative_terms,
    negative_overlap_penalty,
    phrase_bm25_scores,
    sanitize_negative_terms,
)
from arxiv_daily.llm import build_interest_profile, profile_has_rank_tiers
from arxiv_daily.protocol import CorpusPaper
from arxiv_daily.retriever.arxiv_retriever import ArxivRetriever
from datetime import datetime


def _paper(**values: object) -> dict[str, object]:
    base: dict[str, object] = {
        "title": "",
        "summary": "",
        "authors": [],
        "published": "2026-04-22T12:00:00+00:00",
        "primary_category": "cs.RO",
        "categories": ["cs.RO"],
    }
    base.update(values)
    return base


def _retriever(**arxiv: object) -> ArxivRetriever:
    source: dict[str, object] = {
        "category": ["cs.RO"],
        "use_bm25_scoring": False,
        "min_keyword_matches": 1,
        "llm_prefilter_limit": 3,
        "keyword_fallback_min_results": 10,
        "recency_half_life_days": 14,
        "author_overlap_weight": 0,
        "category_preference_weight": 0,
    }
    source.update(arxiv)
    return ArxivRetriever(
        {
            "executor": {"business_date": "2026-04-22", "timezone": "UTC"},
            "source": {"arxiv": source},
        }
    )


class NegativeTermTests(unittest.TestCase):
    def test_generic_negative_terms_are_dropped(self):
        cleaned = sanitize_negative_terms(
            ["learning", "model", "robot", "control", "data", "ankle mechanism", "机器人"]
        )

        self.assertEqual(cleaned, ["ankle mechanism"])

    def test_feedback_titles_keep_specific_tokens_only(self):
        terms = feedback_negative_terms(
            [
                "Learning a robot model for control",
                "Design of an ankle mechanism",
            ]
        )

        self.assertIn("ankle", terms)
        self.assertIn("mechanism", terms)
        for generic in ("learning", "robot", "model", "control", "data"):
            self.assertNotIn(generic, terms)

    def test_penalty_shrinks_as_core_evidence_grows(self):
        self.assertEqual(negative_overlap_penalty(4.0, 0.0), 4.0)
        self.assertAlmostEqual(negative_overlap_penalty(2.0, 2.0), 1.5)
        self.assertEqual(negative_overlap_penalty(4.0, 8.0), 0.0)
        self.assertEqual(negative_overlap_penalty(0.0, 0.0), 0.0)

    def test_weak_positive_hit_is_pushed_below_a_clean_core_hit(self):
        retriever = _retriever(llm_prefilter_limit=2)
        retriever.config["interest_profile"] = {
            "core_terms": ["humanoid locomotion"],
            "broad_terms": ["reinforcement learning"],
            "negative_terms": ["ankle mechanism"],
        }
        papers = [
            _paper(
                entry_id="dirty",
                title="Humanoid locomotion with an ankle mechanism",
                summary="balance",
            ),
            _paper(
                entry_id="clean",
                title="Humanoid locomotion on rough ground",
                summary="balance",
            ),
        ]

        ranked = retriever._rank_candidate_pool(papers, ["humanoid locomotion"])

        self.assertEqual([paper["entry_id"] for paper in ranked], ["clean", "dirty"])

    def test_strong_core_paper_survives_an_exclusion_phrase(self):
        retriever = _retriever(llm_prefilter_limit=2)
        core = [
            "humanoid locomotion",
            "loco-manipulation",
            "perceptive locomotion",
            "motion tracking",
        ]
        retriever.config["interest_profile"] = {
            "core_terms": core,
            "broad_terms": [],
            "negative_terms": ["ankle mechanism"],
        }
        papers = [
            _paper(entry_id="weak", title="Humanoid locomotion", summary="balance"),
            _paper(
                entry_id="strong",
                title="Humanoid locomotion loco-manipulation perceptive locomotion motion tracking",
                summary="Also mentions an ankle mechanism in related work",
            ),
        ]

        ranked = retriever._rank_candidate_pool(papers, core)

        self.assertEqual(ranked[0]["entry_id"], "strong")

    def test_negative_phrases_do_not_match_a_component_word(self):
        retriever = _retriever(llm_prefilter_limit=2)
        retriever.config["interest_profile"] = {
            "core_terms": ["perceptive locomotion"],
            "broad_terms": [],
            "negative_terms": ["visual slam"],
        }
        papers = [
            _paper(
                entry_id="visual",
                title="Perceptive locomotion from visual observations",
                summary="Onboard depth",
            ),
            _paper(
                entry_id="slam",
                title="Perceptive locomotion for visual slam mapping",
                summary="Pose graph",
            ),
        ]

        ranked = retriever._rank_candidate_pool(papers, ["perceptive locomotion"])

        self.assertEqual([paper["entry_id"] for paper in ranked], ["visual", "slam"])


class AuthorMatchTests(unittest.TestCase):
    def _paper(self, *authors: str) -> SimpleNamespace:
        return SimpleNamespace(authors=list(authors), file_path="")

    def test_full_name_key_ignores_surname_only_and_common_surnames(self):
        keys = corpus_author_keys(
            [
                self._paper("Wei Wang"),
                self._paper("Wei Wang; Xue Bin Peng"),
                self._paper("Zhang"),
                self._paper("Anonymous Submission"),
                self._paper("xiaopang"),
            ]
        )

        self.assertEqual(keys["w|wang"], 2)
        self.assertEqual(keys["x|peng"], 1)
        self.assertNotIn("z|zhang", keys)
        self.assertEqual(author_overlap_bonus(["Lei Wang"], keys, 0.6), 0.0)
        self.assertEqual(author_overlap_bonus(["Ann Zhang"], keys, 0.0), 0.0)
        self.assertGreater(author_overlap_bonus(["Wei Wang"], keys, 0.6), 0.0)
        self.assertGreater(
            author_overlap_bonus(["Wei Wang"], keys, 0.6),
            author_overlap_bonus(["Xue Peng"], keys, 0.6),
        )

    def test_normalizes_last_first_accents_and_hyphens(self):
        western = corpus_author_keys([self._paper("Peng, Xue Bin")])
        self.assertIn("x|peng", western)
        self.assertGreater(author_overlap_bonus(["Xue Bin Peng"], western, 0.6), 0.0)
        self.assertEqual(author_overlap_bonus(["Ann Peng"], western, 0.6), 0.0)

        accented = corpus_author_keys([self._paper("José García")])
        self.assertIn("j|garcia", accented)
        self.assertGreater(author_overlap_bonus(["Jose Garcia"], accented, 0.6), 0.0)

        hyphen = corpus_author_keys([self._paper("Hong-Xing Yu")])
        self.assertGreater(author_overlap_bonus(["Hong Xing Yu"], hyphen, 0.6), 0.0)

        umlaut = corpus_author_keys([self._paper("Moritz Bächer")])
        self.assertGreater(author_overlap_bonus(["Moritz Bacher"], umlaut, 0.6), 0.0)

    def test_bonus_cap_stays_at_three_weights(self):
        surnames = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]
        library = {f"a|{surname.lower()}": 5 for surname in surnames}
        authors = [f"Ann {surname}" for surname in surnames]

        bonus = author_overlap_bonus(authors, library, 0.6)

        self.assertAlmostEqual(bonus, 1.8)
        self.assertLessEqual(bonus, 3 * 0.6)

    def test_repeated_library_author_outranks_a_shared_surname(self):
        retriever = _retriever(author_overlap_weight=0.6, llm_prefilter_limit=4)
        retriever._corpus = [
            CorpusPaper(
                title="One",
                abstract="humanoid",
                added_date=datetime(2026, 4, 1),
                authors=["Wei Wang"],
            ),
            CorpusPaper(
                title="Two",
                abstract="humanoid",
                added_date=datetime(2026, 4, 2),
                authors=["Wei Wang"],
            ),
            CorpusPaper(
                title="Three",
                abstract="humanoid",
                added_date=datetime(2026, 4, 3),
                authors=["Xue Bin Peng"],
            ),
            CorpusPaper(
                title="Surname only",
                abstract="humanoid",
                added_date=datetime(2026, 4, 4),
                authors=["Zhang"],
            ),
        ]
        papers = [
            _paper(entry_id="lei", title="Alpha beta", summary="robot", authors=["Lei Wang"]),
            _paper(entry_id="ann", title="Alpha beta", summary="robot", authors=["Ann Zhang"]),
            _paper(entry_id="wei", title="Alpha beta", summary="robot", authors=["Wei Wang"]),
            _paper(entry_id="xue", title="Alpha beta", summary="robot", authors=["X. Peng"]),
        ]

        ranked = retriever._rank_candidate_pool(papers, ["alpha", "beta"])

        self.assertEqual(
            [paper["entry_id"] for paper in ranked],
            ["wei", "xue", "lei", "ann"],
        )


class BroadTermTests(unittest.TestCase):
    def test_phrase_bm25_ignores_a_lone_generic_token(self):
        scores = phrase_bm25_scores(
            [
                "Reinforcement learning for humanoid control",
                "Learning a dynamics model of a robot",
                "Whole-body control on rough terrain",
            ],
            ["reinforcement learning", "humanoid whole-body control"],
        )

        self.assertGreater(scores[0], 0.0)
        self.assertEqual(scores[1], 0.0)
        self.assertGreater(scores[2], 0.0)

    def test_generic_phrase_cannot_be_promoted_to_core(self):
        weights = assign_term_weights(
            ["reinforcement learning", "loco-manipulation"],
            ["placeholder"] * 5,
            core_terms=["reinforcement learning", "loco-manipulation"],
            broad_terms=[],
        )

        self.assertEqual(weights["reinforcement learning"], BROAD_TERM_WEIGHT)
        self.assertEqual(weights["loco-manipulation"], CORE_TERM_WEIGHT)

    def test_high_document_frequency_terms_are_demoted_without_profile_tiers(self):
        texts = ["sim-to-real transfer study"] * 40 + ["loco-manipulation of a box"] * 8
        texts += ["unrelated geometry proof"] * 12
        weights = assign_term_weights(
            ["sim-to-real", "loco-manipulation"],
            texts,
        )

        self.assertEqual(weights["sim-to-real"], BROAD_TERM_WEIGHT)
        self.assertEqual(weights["loco-manipulation"], CORE_TERM_WEIGHT)

    def test_broad_only_papers_do_not_fill_the_prefilter_when_core_papers_exist(self):
        retriever = _retriever(llm_prefilter_limit=2, use_bm25_scoring=True)
        retriever.config["interest_profile"] = {
            "core_terms": ["loco-manipulation"],
            "broad_terms": ["reinforcement learning", "sim-to-real"],
            "negative_terms": [],
        }
        papers = [
            _paper(
                entry_id="broad-new",
                title="Reinforcement learning for sim-to-real transfer",
                summary="A general learning benchmark",
                published="2026-04-22T18:00:00+00:00",
            ),
            _paper(
                entry_id="core-old",
                title="Loco-manipulation on a humanoid",
                summary="Whole body contact",
                published="2026-04-10T12:00:00+00:00",
            ),
            _paper(
                entry_id="core-mid",
                title="Dataset for loco-manipulation",
                summary="Humanoid trials",
                published="2026-04-18T12:00:00+00:00",
            ),
            _paper(
                entry_id="broad-old",
                title="Model predictive control survey",
                summary="Reinforcement learning is mentioned once",
                published="2026-04-12T12:00:00+00:00",
            ),
        ]

        ranked = retriever._rank_candidate_pool(
            papers,
            ["loco-manipulation", "reinforcement learning", "sim-to-real"],
        )

        self.assertEqual([paper["entry_id"] for paper in ranked], ["core-mid", "core-old"])

    def test_generic_only_papers_do_not_pad_the_prefilter(self):
        retriever = _retriever(llm_prefilter_limit=5, use_bm25_scoring=True)
        retriever.config["interest_profile"] = {
            "core_terms": ["humanoid whole-body control", "loco-manipulation"],
            "broad_terms": ["reinforcement learning", "sim-to-real"],
            "negative_terms": [],
        }
        papers = [
            _paper(
                entry_id="phrase",
                title="Loco-manipulation on a humanoid",
                summary="Whole body contact",
                published="2026-04-12T12:00:00+00:00",
            ),
            _paper(
                entry_id="token",
                title="A humanoid keeps its balance",
                summary="Flat ground walking",
                published="2026-04-20T12:00:00+00:00",
            ),
            _paper(
                entry_id="broad",
                title="Reinforcement learning for sim-to-real transfer",
                summary="A general benchmark",
                published="2026-04-22T12:00:00+00:00",
            ),
            _paper(
                entry_id="blank",
                title="Graph transformers for language models",
                summary="No robotics content",
                published="2026-04-22T18:00:00+00:00",
            ),
        ]

        ranked = retriever._rank_candidate_pool(
            papers,
            [
                "humanoid whole-body control",
                "loco-manipulation",
                "reinforcement learning",
                "sim-to-real",
            ],
        )

        self.assertEqual(
            [paper["entry_id"] for paper in ranked],
            ["phrase", "token", "broad"],
        )
        self.assertEqual(
            [paper["prefilter_tier"] for paper in ranked],
            ["core", "core", "broad"],
        )
        self.assertEqual(retriever._last_rank_stats["broad_backfill_count"], 1)

    def test_broad_backfill_stays_behind_core_and_respects_the_cap(self):
        retriever = _retriever(
            llm_prefilter_limit=10,
            broad_backfill_limit=3,
            use_bm25_scoring=True,
        )
        retriever.config["interest_profile"] = {
            "core_terms": ["loco-manipulation"],
            "broad_terms": ["reinforcement learning", "sim-to-real", "imitation learning"],
            "negative_terms": ["visual slam"],
        }
        papers = [
            _paper(
                entry_id="core",
                title="Loco-manipulation with a humanoid",
                summary="Contact rich",
                published="2026-04-01T12:00:00+00:00",
            ),
            _paper(
                entry_id="broad-strong",
                title="Reinforcement learning and sim-to-real imitation learning",
                summary="Policy transfer",
                published="2026-04-02T12:00:00+00:00",
            ),
            _paper(
                entry_id="broad-weak",
                title="Imitation learning and reinforcement learning",
                summary="A toy gridworld",
                published="2026-04-22T12:00:00+00:00",
            ),
            _paper(
                entry_id="excluded",
                title="Reinforcement learning for visual slam",
                summary="Mapping",
                published="2026-04-22T18:00:00+00:00",
            ),
            _paper(
                entry_id="blank",
                title="Graph transformers",
                summary="Language only",
                published="2026-04-22T18:00:00+00:00",
            ),
            _paper(
                entry_id="broad-extra",
                title="Sim-to-real robot transfer",
                summary="Domain gap",
                published="2026-04-21T12:00:00+00:00",
            ),
        ]

        ranked = retriever._rank_candidate_pool(
            papers,
            [
                "loco-manipulation",
                "reinforcement learning",
                "sim-to-real",
                "imitation learning",
            ],
        )

        self.assertEqual([paper["entry_id"] for paper in ranked], ["core", "broad-strong", "broad-weak"])
        self.assertEqual(
            [paper["prefilter_tier"] for paper in ranked],
            ["core", "broad", "broad"],
        )
        self.assertEqual(retriever._last_rank_stats["core_count"], 1)
        self.assertEqual(retriever._last_rank_stats["broad_backfill_count"], 2)
        self.assertLessEqual(len(ranked), 3)


class SeenBeforeLimitTests(unittest.TestCase):
    def test_seen_papers_do_not_consume_prefilter_slots(self):
        retriever = _retriever(llm_prefilter_limit=2)
        seen_id = "http://arxiv.org/abs/2604.10001v1"
        retriever.config["seen_paper_urls"] = sorted(
            _arxiv_url_aliases("https://arxiv.org/abs/2604.10001v1")
        )
        papers = [
            _paper(
                entry_id=seen_id,
                title="Alpha beta gamma delta epsilon",
                summary="best match already recommended",
            ),
            _paper(entry_id="second", title="Alpha beta gamma delta", summary="next"),
            _paper(entry_id="third", title="Alpha beta gamma", summary="third"),
            _paper(entry_id="fourth", title="Alpha beta", summary="fourth"),
        ]

        ranked = retriever._rank_candidate_pool(
            papers, ["alpha", "beta", "gamma", "delta", "epsilon"]
        )

        self.assertEqual([paper["entry_id"] for paper in ranked], ["second", "third"])
        self.assertEqual(retriever._last_rank_stats["final_count"], 2)
        self.assertGreater(retriever._last_rank_stats["seen_skipped"], 0)

    def test_retrieval_config_carries_seen_identities(self):
        config = _config_with_retrieval_context(
            {"executor": {"max_paper_num": 10}},
            {"canonical_terms": ["loco-manipulation"]},
            _arxiv_url_aliases("http://arxiv.org/abs/2604.10001v1"),
            {"content-key"},
        )

        self.assertIn("https://arxiv.org/abs/2604.10001v1", config["seen_paper_urls"])
        self.assertIn("http://arxiv.org/abs/2604.10001v1", config["seen_paper_urls"])
        self.assertEqual(config["seen_content_keys"], ["content-key"])
        self.assertEqual(config["interest_profile"]["canonical_terms"], ["loco-manipulation"])


class ProfileTierTests(unittest.TestCase):
    def test_legacy_profile_is_not_ready_for_ranking(self):
        self.assertFalse(
            profile_has_rank_tiers(
                {
                    "canonical_terms": ["humanoid whole-body control"],
                    "not_interested": ["通用强化学习理论"],
                }
            )
        )

    def test_profile_drops_generic_negative_terms(self):
        payload = {
            "summary": "关注人形全身控制",
            "topics": ["人形控制"],
            "methods": ["运动跟踪"],
            "not_interested": ["并联踝关节机构"],
            "representative_papers": ["GMT"],
            "canonical_terms": ["reinforcement learning", "humanoid whole-body control"],
            "core_terms": ["humanoid whole-body control", "loco-manipulation"],
            "broad_terms": ["reinforcement learning", "sim-to-real"],
            "negative_terms": ["learning", "model", "ankle mechanism", "data", "visual slam"],
        }
        with patch("arxiv_daily.llm._call_json", return_value=payload):
            profile = build_interest_profile(
                [],
                [],
                {"llm": {"language": "Chinese"}},
            )

        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(
            profile["negative_terms"],
            ["ankle mechanism", "visual slam"],
        )
        self.assertEqual(profile["core_terms"][0], "humanoid whole-body control")
        self.assertEqual(profile["canonical_terms"][0], "humanoid whole-body control")
        self.assertIn("reinforcement learning", profile["broad_terms"])
        self.assertTrue(profile_has_rank_tiers(profile))


if __name__ == "__main__":
    unittest.main()
