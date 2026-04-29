from abc import ABC, abstractmethod
from typing import Any, Type
from loguru import logger
from ..protocol import Paper, CorpusPaper
from ..utils import parallel_execute


class BaseRetriever(ABC):
    name: str

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self._corpus: list[CorpusPaper] = []

    def set_corpus(self, corpus: list[CorpusPaper]) -> None:
        self._corpus = corpus

    @abstractmethod
    def _retrieve_raw_papers(self) -> list[Any]:
        pass

    @abstractmethod
    def convert_to_paper(self, raw_paper: Any) -> Paper | None:
        pass

    def retrieve_papers(self, corpus: list[CorpusPaper] | None = None) -> list[Paper]:
        if corpus is not None:
            self._corpus = corpus
        raw_papers = self._retrieve_raw_papers()
        logger.info(f"Processing {len(raw_papers)} raw papers...")

        executor_config = self.config.get("executor", {})
        max_workers = 4
        if isinstance(executor_config, dict):
            max_workers = int(executor_config.get("retriever_workers", 4) or 4)

        def convert(raw_paper: Any) -> Paper | None:
            try:
                return self.convert_to_paper(raw_paper)
            except Exception as exc:
                logger.warning(f"Skipping paper: {exc}")
                return None

        return [
            paper
            for paper in parallel_execute(
                convert,
                raw_papers,
                max_workers=max_workers,
                desc="Converting papers",
            )
            if paper is not None
        ]


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
