from abc import ABC, abstractmethod
from typing import Type
import numpy as np

from ..protocol import Paper, CorpusPaper


class BaseReranker(ABC):
    def __init__(self, config: dict):
        self.config = config

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        if not candidates:
            return []
        if not corpus:
            for candidate in candidates:
                candidate.score = 0.0
            return candidates

        sim = self.get_similarity_score(
            candidates,
            [f"{c.title}\n{c.abstract}" for c in candidates],
            corpus,
            [f"{c.title}\n{c.abstract}" for c in corpus],
        )
        if sim.ndim != 2 or sim.shape[0] != len(candidates):
            raise ValueError(
                f"Unexpected similarity matrix shape {sim.shape} for {len(candidates)} candidates"
            )

        top_k = min(
            max(1, self.config.get("reranker", {}).get("top_k", 20)),
            sim.shape[1],
        )
        top_scores = np.partition(sim, sim.shape[1] - top_k, axis=1)[:, -top_k:]
        scores = top_scores.mean(axis=1) * 10
        for s, c in zip(scores, candidates):
            c.score = float(s)
        candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        return candidates

    @abstractmethod
    def get_similarity_score(
        self,
        candidates: list[Paper],
        candidate_texts: list[str],
        corpus: list[CorpusPaper],
        corpus_texts: list[str],
    ) -> np.ndarray:
        raise NotImplementedError


_registered_rerankers: dict[str, Type[BaseReranker]] = {}


def register_reranker(name: str):
    def decorator(cls):
        _registered_rerankers[name] = cls
        return cls

    return decorator


def get_reranker_cls(name: str) -> Type[BaseReranker]:
    if name not in _registered_rerankers:
        raise ValueError(
            f"Unknown reranker: {name}. Available: {list(_registered_rerankers.keys())}"
        )
    return _registered_rerankers[name]
