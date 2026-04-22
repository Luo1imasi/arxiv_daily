from abc import ABC, abstractmethod
from typing import Type
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger
from tqdm import tqdm
from ..protocol import Paper, CorpusPaper


class BaseRetriever(ABC):
    name: str

    def __init__(self, config: dict):
        self.config = config
        self._corpus: list[CorpusPaper] = []

    def set_corpus(self, corpus: list[CorpusPaper]):
        self._corpus = corpus

    @abstractmethod
    def _retrieve_raw_papers(self) -> list:
        pass

    @abstractmethod
    def convert_to_paper(self, raw_paper) -> Paper | None:
        pass

    def retrieve_papers(self, corpus: list[CorpusPaper] | None = None) -> list[Paper]:
        if corpus is not None:
            self._corpus = corpus
        raw_papers = self._retrieve_raw_papers()
        logger.info(f"Processing {len(raw_papers)} raw papers...")

        max_workers = self.config.get("executor", {}).get("retriever_workers", 4)
        papers = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.convert_to_paper, p): p for p in raw_papers}
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Converting papers"
            ):
                try:
                    paper = future.result()
                    if paper is not None:
                        papers.append(paper)
                except Exception as exc:
                    logger.warning(f"Skipping paper: {exc}")

        return papers


_registered_retrievers: dict[str, Type[BaseRetriever]] = {}


def register_retriever(name: str):
    def decorator(cls):
        _registered_retrievers[name] = cls
        cls.name = name
        return cls

    return decorator


def get_retriever_cls(name: str) -> Type[BaseRetriever]:
    if name not in _registered_retrievers:
        raise ValueError(
            f"Unknown retriever: {name}. Available: {list(_registered_retrievers.keys())}"
        )
    return _registered_retrievers[name]
