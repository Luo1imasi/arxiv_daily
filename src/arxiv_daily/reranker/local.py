import asyncio
import hashlib
import json
import logging
import warnings
from datetime import datetime, timezone

import numpy as np
from loguru import logger

from .base import BaseReranker, register_reranker
from .. import database as db
from ..config import get_config_value
from ..utils import make_content_key
from ..business_date import reference_datetime as _reference_datetime

_model_cache: dict[str, object] = {}
_corpus_feature_cache: dict[str, dict[str, object]] = {}


def _get_encoder(model_name: str):
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Local reranker dependencies are not installed. Reinstall the project dependencies and try again."
            ) from exc

        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning)

        logger.info(f"Loading embedding model: {model_name}")
        _model_cache[model_name] = SentenceTransformer(
            model_name, trust_remote_code=True
        )
    else:
        logger.info(f"Using cached embedding model: {model_name}")
    return _model_cache[model_name]


def _to_numpy(features) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return array


def _normalize_rows(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def _load_cached_features(items: list, model_key: str) -> tuple[list[int], list]:
    return asyncio.run(db.load_embeddings(items, model_key))


def _save_cached_features(items: list, features: np.ndarray, model_key: str) -> None:
    asyncio.run(db.save_embeddings(items, features, model_key))


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    min_score = float(np.min(scores))
    max_score = float(np.max(scores))
    if max_score <= min_score:
        return np.zeros(scores.shape[0], dtype=np.float32)
    return (scores - min_score) / (max_score - min_score)


def _display_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []

    raw = np.asarray(scores, dtype=np.float32)
    normalized = _normalize_scores(raw)
    if float(np.max(normalized)) <= 0.0:
        return [5.0 for _ in scores]

    return [float(4.0 + value * 6.0) for value in normalized]


def _compute_topk_stats(
    candidate_features: np.ndarray,
    corpus_features: np.ndarray,
    top_k: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean_scores = np.zeros(candidate_features.shape[0], dtype=np.float32)
    max_scores = np.zeros(candidate_features.shape[0], dtype=np.float32)

    for start in range(0, candidate_features.shape[0], batch_size):
        end = min(start + batch_size, candidate_features.shape[0])
        sim = candidate_features[start:end] @ corpus_features.T
        local_top_k = min(top_k, sim.shape[1])
        top_scores = np.partition(sim, sim.shape[1] - local_top_k, axis=1)[:, -local_top_k:]
        mean_scores[start:end] = top_scores.mean(axis=1)
        max_scores[start:end] = sim.max(axis=1)

    return mean_scores, max_scores


@register_reranker("local")
class LocalReranker(BaseReranker):
    def rerank(self, candidates: list, corpus: list) -> list:
        if not candidates:
            return []
        if not corpus:
            for candidate in candidates:
                candidate.score = 0.0
            return candidates

        model_name = get_config_value(self.config, "reranker.model")
        encode_kwargs = get_config_value(self.config, "reranker.encode_kwargs")
        embedding_cache_key = (
            f"{model_name}|{json.dumps(encode_kwargs, sort_keys=True, ensure_ascii=True)}"
        )

        candidate_texts = [f"{c.title}\n{c.abstract}" for c in candidates]
        corpus_texts = [f"{c.title}\n{c.abstract}" for c in corpus]

        encoder = _get_encoder(model_name)
        candidate_features = self._get_item_features(
            encoder,
            candidates,
            candidate_texts,
            embedding_cache_key,
            encode_kwargs,
            log_prefix="candidate",
        )

        corpus_features = self._get_corpus_features(
            encoder,
            corpus,
            corpus_texts,
            embedding_cache_key,
            encode_kwargs,
        )

        top_k = min(
            max(1, int(get_config_value(self.config, "reranker.top_k"))),
            corpus_features.shape[0],
        )
        batch_size = max(1, int(get_config_value(self.config, "reranker.score_batch_size")))
        relevance_scores, max_scores = _compute_topk_stats(
            candidate_features,
            corpus_features,
            top_k,
            batch_size,
        )

        novelty_scores = np.clip(1.0 - max_scores, 0.0, 1.0)
        recency_scores = np.array(
            [self._get_recency_score(getattr(candidate, "date", None)) for candidate in candidates],
            dtype=np.float32,
        )

        relevance_weight = float(get_config_value(self.config, "reranker.relevance_weight"))
        novelty_weight = float(get_config_value(self.config, "reranker.novelty_weight"))
        recency_weight = float(get_config_value(self.config, "reranker.recency_weight"))
        retrieval_weight = float(get_config_value(self.config, "reranker.retrieval_weight"))
        retrieval_scores = _normalize_scores(
            np.array(
                [float(getattr(candidate, "retrieval_score", 0.0) or 0.0) for candidate in candidates],
                dtype=np.float32,
            )
        )
        base_scores = (
            relevance_scores * relevance_weight
            + novelty_scores * novelty_weight
            + recency_scores * recency_weight
            + retrieval_scores * retrieval_weight
        )

        max_paper_num = max(1, int(get_config_value(self.config, "executor.max_paper_num")))
        pre_mmr_factor = max(
            1,
            int(get_config_value(self.config, "reranker.pre_mmr_candidate_factor")),
        )
        pre_mmr_limit = min(len(candidates), max(max_paper_num * pre_mmr_factor, max_paper_num))
        if pre_mmr_limit < len(candidates):
            selected_indices = np.argsort(-base_scores)[:pre_mmr_limit]
            selected_indices = selected_indices.tolist()
            candidates = [candidates[i] for i in selected_indices]
            candidate_features = candidate_features[selected_indices]
            base_scores = base_scores[selected_indices]

        candidate_sim = candidate_features @ candidate_features.T
        order, mmr_scores = self._mmr_order(base_scores, candidate_sim, max_paper_num)

        reranked = [candidates[i] for i in order]
        display_scores = _display_scores(mmr_scores)
        for rank, idx in enumerate(order):
            candidates[idx].score = display_scores[rank]
        return reranked

    def _get_corpus_features(
        self,
        encoder,
        corpus: list,
        corpus_texts: list[str],
        embedding_cache_key: str,
        encode_kwargs: dict,
    ) -> np.ndarray:
        corpus_keys = [make_content_key(p.title, p.abstract or "") for p in corpus]
        cache_signature = hashlib.sha1("|".join(corpus_keys).encode("utf-8")).hexdigest()
        cache_key = f"{embedding_cache_key}|{cache_signature}"
        cached = _corpus_feature_cache.get(cache_key)
        if cached is not None:
            logger.info(f"Using in-memory cached corpus embeddings: {len(corpus)} papers")
            return cached["features"]

        feature_matrix = self._get_item_features(
            encoder,
            corpus,
            corpus_texts,
            embedding_cache_key,
            encode_kwargs,
            log_prefix="corpus",
        )
        _corpus_feature_cache[cache_key] = {"features": feature_matrix}
        return feature_matrix

    def _get_item_features(
        self,
        encoder,
        items: list,
        texts: list[str],
        embedding_cache_key: str,
        encode_kwargs: dict,
        log_prefix: str,
    ) -> np.ndarray:
        cached_indices, cached_embeddings = _load_cached_features(items, embedding_cache_key)
        logger.info(f"Found {len(cached_indices)} cached embeddings for {log_prefix} papers")

        all_indices = set(range(len(items)))
        uncached_indices = sorted(all_indices - set(cached_indices))
        if uncached_indices:
            uncached_texts = [texts[i] for i in uncached_indices]
            logger.info(f"Encoding {len(uncached_texts)} uncached {log_prefix} papers...")
            uncached_features = _to_numpy(
                encoder.encode(
                    uncached_texts,
                    **encode_kwargs,
                    show_progress_bar=True,
                )
            )
            _save_cached_features(
                [items[i] for i in uncached_indices],
                uncached_features,
                embedding_cache_key,
            )
        else:
            uncached_features = np.zeros((0, 0), dtype=np.float32)

        item_features = [None] * len(items)
        for i, emb in zip(cached_indices, cached_embeddings):
            item_features[i] = np.asarray(emb, dtype=np.float32)
        for i, emb in zip(uncached_indices, uncached_features):
            item_features[i] = np.asarray(emb, dtype=np.float32)

        valid_features = [feature for feature in item_features if feature is not None]
        if len(valid_features) != len(item_features):
            missing = len(item_features) - len(valid_features)
            logger.warning(f"Missing embeddings for {missing} {log_prefix} papers, dropping them")
        if not valid_features:
            raise ValueError(f"No {log_prefix} embeddings available after cache/encoding")
        return _normalize_rows(_to_numpy(valid_features))

    def _get_recency_score(self, published_date: str | None) -> float:
        if not published_date:
            return 0.0
        try:
            published = datetime.strptime(published_date, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            )
            age_days = max(
                (_reference_datetime(self.config).date() - published.date()).days,
                0,
            )
            half_life = max(
                float(get_config_value(self.config, "reranker.recency_half_life_days")),
                1.0,
            )
            return float(np.exp(-age_days / half_life))
        except ValueError:
            return 0.0

    def _mmr_order(
        self,
        base_scores: np.ndarray,
        candidate_similarity: np.ndarray,
        limit: int | None = None,
    ) -> tuple[list[int], list[float]]:
        remaining = set(range(len(base_scores)))
        selected = []
        selected_scores = []
        mmr_lambda = float(get_config_value(self.config, "reranker.mmr_lambda"))

        target_size = min(limit or len(base_scores), len(base_scores))

        while remaining and len(selected) < target_size:
            best_idx = None
            best_score = None
            for idx in remaining:
                diversity_penalty = 0.0
                if selected:
                    diversity_penalty = max(candidate_similarity[idx, chosen] for chosen in selected)
                score = mmr_lambda * float(base_scores[idx]) - (1.0 - mmr_lambda) * float(
                    diversity_penalty
                )
                if best_score is None or score > best_score:
                    best_idx = idx
                    best_score = score
            remaining.remove(best_idx)
            selected.append(best_idx)
            selected_scores.append(best_score if best_score is not None else 0.0)

        return selected, selected_scores

    def get_similarity_score(
        self,
        candidates: list,
        candidate_texts: list[str],
        corpus: list,
        corpus_texts: list[str],
    ) -> np.ndarray:
        model_name = get_config_value(self.config, "reranker.model")
        encode_kwargs = get_config_value(self.config, "reranker.encode_kwargs")
        embedding_cache_key = (
            f"{model_name}|{json.dumps(encode_kwargs, sort_keys=True, ensure_ascii=True)}"
        )

        encoder = _get_encoder(model_name)

        candidate_features = self._get_item_features(
            encoder,
            candidates,
            candidate_texts,
            embedding_cache_key,
            encode_kwargs,
            log_prefix="candidate",
        )
        corpus_features = self._get_corpus_features(
            encoder,
            corpus,
            corpus_texts,
            embedding_cache_key,
            encode_kwargs,
        )
        return candidate_features @ corpus_features.T
