"""Lexical stand-in for the old embedding reranker.

The daily pipeline no longer calls this. Query ranking lives in ``lexical``.
Nothing in this module imports torch or sentence-transformers.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from .base import BaseReranker, register_reranker
from ..business_date import reference_datetime as _reference_datetime
from ..config import get_config_value
from ..lexical import order_by_bm25, rank_items_by_query as _rank_items_by_query


@register_reranker("local")
class LocalReranker(BaseReranker):
    def rerank(self, candidates: list, corpus: list) -> list:
        del corpus
        if not candidates:
            return []
        return order_by_bm25(candidates, len(candidates))

    def get_similarity_score(self, candidates, candidate_texts, corpus, corpus_texts):
        import numpy as np

        from ..lexical import jaccard, token_set

        del candidates, corpus
        left = [token_set(text) for text in candidate_texts]
        right = [token_set(text) for text in corpus_texts]
        if not left or not right:
            return np.zeros((len(left), len(right)), dtype=np.float32)
        rows = [[jaccard(tokens, other) for other in right] for tokens in left]
        return np.asarray(rows, dtype=np.float32)

    def _get_recency_score(self, published_date: str | None) -> float:
        if not published_date:
            return 0.0
        try:
            published = datetime.strptime(published_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            age_days = max(
                (_reference_datetime(self.config).date() - published.date()).days,
                0,
            )
            half_life = max(
                float(get_config_value(self.config, "reranker.recency_half_life_days")),
                1.0,
            )
            return float(math.exp(-age_days / half_life))
        except ValueError:
            return 0.0


def rank_items_by_query(config: dict, query: str, items: list, top_k: int = 8) -> list[tuple[int, float]]:
    del config
    return _rank_items_by_query(query, items, top_k=top_k)
