"""Lexical scoring and diversity. This module must not load embedding models."""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any

TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9+\-\.]{1,}")
_AUTHOR_SPLIT = re.compile(r"\s*(?:;|\band\b|,|&)\s*", re.IGNORECASE)
_SURNAME_STOP = {
    "al",
    "anonymous",
    "author",
    "authors",
    "da",
    "de",
    "dr",
    "et",
    "jr",
    "la",
    "le",
    "mr",
    "sr",
    "submission",
    "van",
    "von",
}
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "for",
    "from",
    "into",
    "that",
    "the",
    "this",
    "with",
}

# Hint words are matched as whole tokens against the local library.
CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "cs.RO": ("robot", "humanoid", "locomotion", "manipulation", "quadruped", "legged"),
    "cs.CV": ("vision", "image", "video", "detection", "segmentation"),
    "cs.CL": ("language", "linguistic", "translation", "dialogue"),
    "cs.LG": ("learning", "reinforcement", "policy", "gradient"),
    "cs.AI": ("agent", "planning", "reasoning"),
    "cs.SY": ("control", "dynamical", "stability"),
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall((text or "").lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    if union <= 0:
        return 0.0
    return len(left & right) / union


def author_surnames(names: list[str]) -> set[str]:
    found: set[str] = set()
    for raw in names:
        for part in _AUTHOR_SPLIT.split(str(raw or "")):
            tokens = re.findall(r"[A-Za-z][A-Za-z'\-]{1,}", part)
            if not tokens:
                continue
            surname = tokens[-1].lower().strip("-'")
            if len(surname) < 3 or surname in _SURNAME_STOP:
                continue
            found.add(surname)
    return found


def _pdf_author_string(pdf_path: str) -> str:
    if not pdf_path or not os.path.isfile(pdf_path):
        return ""
    try:
        from .webdav import _load_meta_cache, _save_meta_cache, extract_metadata_from_pdf
    except Exception:
        return ""
    try:
        cached = _load_meta_cache(pdf_path) or {}
    except Exception:
        cached = {}
    if isinstance(cached, dict) and "author" in cached:
        return str(cached.get("author") or "")
    try:
        author = str(extract_metadata_from_pdf(pdf_path).get("author") or "")
    except Exception:
        return ""
    if isinstance(cached, dict) and cached:
        updated = dict(cached)
        updated["author"] = author
        try:
            _save_meta_cache(pdf_path, updated)
        except Exception:
            pass
    return author


def corpus_author_surnames(corpus: list[Any]) -> set[str]:
    names: list[str] = []
    for paper in corpus:
        authors = list(getattr(paper, "authors", None) or [])
        if not authors:
            author_string = _pdf_author_string(str(getattr(paper, "file_path", "") or ""))
            if author_string:
                authors = [author_string]
        names.extend(str(author) for author in authors)
    return author_surnames(names)


def author_overlap_bonus(authors: list[str], library_surnames: set[str], weight: float) -> float:
    if weight <= 0 or not library_surnames or not authors:
        return 0.0
    hits = len(author_surnames(authors) & library_surnames)
    if hits <= 0:
        return 0.0
    return min(hits, 3) * weight


def category_weights_from_corpus(corpus: list[Any]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for paper in corpus:
        tokens = token_set(
            f"{getattr(paper, 'title', '') or ''} {getattr(paper, 'abstract', '') or ''}"
        )
        if not tokens:
            continue
        for category, hints in CATEGORY_HINTS.items():
            hits = sum(1 for hint in hints if hint in tokens)
            if hits:
                counts[category] += hits
    if not counts:
        return {}
    peak = max(counts.values())
    if peak <= 0:
        return {}
    return {category: value / peak for category, value in counts.items()}


def category_preference_bonus(
    primary: str | None,
    categories: list[str],
    weights: dict[str, float],
    scale: float,
) -> float:
    if scale <= 0 or not weights:
        return 0.0
    score = float(weights.get(primary or "", 0.0))
    for category in categories:
        if category == primary:
            continue
        score = max(score, 0.5 * float(weights.get(category, 0.0)))
    return score * scale


def bm25_scores(texts: list[str], query: str) -> list[float]:
    if not texts or not query.strip():
        return [0.0 for _ in texts]
    tokenized = [tokenize(text) for text in texts]
    query_counts = Counter(tokenize(query))
    if not query_counts:
        return [0.0 for _ in texts]
    doc_freq: Counter[str] = Counter()
    for tokens in tokenized:
        doc_freq.update(set(tokens))
    total_docs = len(tokenized)
    avgdl = sum(len(tokens) for tokens in tokenized) / max(total_docs, 1)
    k1 = 1.5
    b = 0.75
    scores: list[float] = []
    for tokens in tokenized:
        if not tokens:
            scores.append(0.0)
            continue
        tf = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for token, qf in query_counts.items():
            df = doc_freq.get(token, 0)
            if not df or not tf.get(token, 0):
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            freq = tf[token]
            denom = freq + k1 * (1 - b + b * doc_len / max(avgdl, 1e-6))
            score += qf * idf * (freq * (k1 + 1)) / denom
        scores.append(score)
    return scores


def rank_items_by_query(query: str, items: list[Any], top_k: int = 8) -> list[tuple[int, float]]:
    if not str(query or "").strip() or not items:
        return []
    texts = []
    for item in items:
        if isinstance(item, dict):
            title = str(item.get("title") or "")
            abstract = str(item.get("abstract") or item.get("summary") or "")
        else:
            title = str(getattr(item, "title", "") or "")
            abstract = str(getattr(item, "abstract", "") or getattr(item, "summary", "") or "")
        texts.append(f"{title}\n{abstract}")
    scores = bm25_scores(texts, query)
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    limit = max(1, int(top_k))
    return [(index, float(scores[index])) for index in order if scores[index] > 0][:limit]


def diverse_indices(
    texts: list[str],
    scores: list[float],
    limit: int,
    *,
    lam: float = 0.78,
    duplicate_jaccard: float = 0.8,
) -> list[int]:
    count = min(len(texts), len(scores))
    if count <= 0 or limit <= 0:
        return []
    target = min(count, limit)
    lo = min(scores[:count])
    hi = max(scores[:count])
    if hi <= lo:
        normalized = [1.0 for _ in range(count)]
    else:
        normalized = [(score - lo) / (hi - lo) for score in scores[:count]]
    token_sets = [token_set(texts[index]) for index in range(count)]
    selected: list[int] = []
    remaining = set(range(count))
    lam = min(1.0, max(0.0, float(lam)))
    while remaining and len(selected) < target:
        slots_left = target - len(selected)
        best_index = None
        best_value = None
        for index in remaining:
            similarity = 0.0
            if selected:
                similarity = max(jaccard(token_sets[index], token_sets[chosen]) for chosen in selected)
            if similarity >= duplicate_jaccard and len(remaining) > slots_left:
                continue
            value = lam * normalized[index] - (1.0 - lam) * similarity
            if best_value is None or value > best_value:
                best_index = index
                best_value = value
        if best_index is None:
            best_index = max(remaining, key=lambda index: (scores[index], -index))
        remaining.remove(best_index)
        selected.append(best_index)
    return selected


def order_by_bm25(papers: list[Any], limit: int) -> list[Any]:
    if limit <= 0 or not papers:
        return []
    ranked = sorted(
        papers,
        key=lambda paper: (
            -float(getattr(paper, "bm25_score", 0.0) or 0.0),
            -float(getattr(paper, "retrieval_score", 0.0) or 0.0),
            str(getattr(paper, "title", "") or ""),
        ),
    )
    seen: set[str] = set()
    chosen: list[Any] = []
    for paper in ranked:
        title_key = " ".join(str(getattr(paper, "title", "") or "").lower().split())
        if title_key and title_key in seen:
            continue
        if title_key:
            seen.add(title_key)
        chosen.append(paper)
        if len(chosen) >= limit:
            break
    peak = max((float(getattr(paper, "bm25_score", 0.0) or 0.0) for paper in chosen), default=0.0)
    if peak <= 0:
        peak = 1.0
    for paper in chosen:
        paper.score = 10.0 * float(getattr(paper, "bm25_score", 0.0) or 0.0) / peak
    return chosen
