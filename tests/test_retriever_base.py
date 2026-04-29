import time
import unittest
from typing import Any, override

from arxiv_daily.protocol import Paper
from arxiv_daily.retriever.base import BaseRetriever


class ReversingTimingRetriever(BaseRetriever):
    @override
    def _retrieve_raw_papers(self) -> list[Any]:
        return ["slow", "fast"]

    @override
    def convert_to_paper(self, raw_paper: Any) -> Paper | None:
        if raw_paper == "slow":
            time.sleep(0.02)
        title = str(raw_paper)
        return Paper(source="test", title=title, authors=[], abstract="", url=title)


class FilteringRetriever(BaseRetriever):
    @override
    def _retrieve_raw_papers(self) -> list[Any]:
        return [1, 2, 3, 4]

    @override
    def convert_to_paper(self, raw_paper: Any) -> Paper | None:
        if raw_paper == 2:
            raise ValueError("bad paper")
        if raw_paper == 3:
            return None
        title = f"paper-{raw_paper}"
        return Paper(source="test", title=title, authors=[], abstract="", url=title)


class BaseRetrieverOrderingTests(unittest.TestCase):
    def test_parallel_conversion_preserves_raw_order(self):
        retriever = ReversingTimingRetriever({"executor": {"retriever_workers": 2}})

        papers = retriever.retrieve_papers([])

        self.assertEqual([paper.title for paper in papers], ["slow", "fast"])

    def test_parallel_conversion_skips_failures_and_none_results(self):
        retriever = FilteringRetriever({"executor": {"retriever_workers": 4}})

        papers = retriever.retrieve_papers([])

        self.assertEqual([paper.title for paper in papers], ["paper-1", "paper-4"])


if __name__ == "__main__":
    unittest.main()
