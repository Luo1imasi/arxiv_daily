import asyncio
import os
import tempfile
import unittest
from datetime import datetime

from arxiv_daily import database as db
from arxiv_daily.executor import _default_llm_metrics, _fallback_tldr, _filter_seen_papers
from arxiv_daily.protocol import CorpusPaper
from arxiv_daily.protocol import Paper
from arxiv_daily.utils import make_content_key


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


if __name__ == "__main__":
    unittest.main()
