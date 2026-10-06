"""Lexical scoring and diversity. This module must not load embedding models."""

from __future__ import annotations

import math
import os
import re
import unicodedata
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
# Bare "robot" / "learning" / "control" fire on almost every candidate, so the
# preference uses the more specific leftovers of each category.
CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "cs.RO": ("humanoid", "locomotion", "manipulation", "quadruped", "legged"),
    "cs.CV": ("vision", "image", "video", "detection", "segmentation"),
    "cs.CL": ("language", "linguistic", "translation", "dialogue"),
    "cs.LG": ("reinforcement", "policy", "gradient"),
    "cs.AI": ("agent", "planning", "reasoning"),
    "cs.SY": ("dynamical", "stability", "controller"),
}

# Tokens that are too common to count as a standalone query hit or as a
# negative term. A phrase is generic only when every token is in this set.
GENERIC_TERMS = {
    "adaptive",
    "algorithm",
    "algorithms",
    "analysis",
    "approach",
    "approaches",
    "based",
    "control",
    "controlled",
    "controller",
    "controllers",
    "controls",
    "data",
    "dataset",
    "datasets",
    "deep",
    "dynamic",
    "dynamics",
    "efficient",
    "framework",
    "frameworks",
    "general",
    "inference",
    "large",
    "learn",
    "learned",
    "learning",
    "method",
    "methods",
    "model",
    "modeling",
    "models",
    "multi",
    "network",
    "networks",
    "neural",
    "novel",
    "paper",
    "policies",
    "policy",
    "real",
    "reinforcement",
    "results",
    "robot",
    "robotic",
    "robotics",
    "robots",
    "robust",
    "simple",
    "single",
    "study",
    "system",
    "systems",
    "task",
    "tasks",
    "time",
    "towards",
    "training",
    "using",
    "via",
    "world",
}

PHRASE_STOPWORDS = STOPWORDS | {
    "among",
    "based",
    "into",
    "such",
    "their",
    "these",
    "using",
    "via",
}

CORE_TERM_WEIGHT = 1.0
BROAD_TERM_WEIGHT = 0.25

_NAME_STOP = _SURNAME_STOP | {
    "del",
    "della",
    "den",
    "der",
    "di",
    "dos",
    "ii",
    "iii",
    "iv",
    "ms",
    "phd",
    "prof",
}

_PERSON_SPLIT = re.compile(r"\s*(?:;|&|\band\b)\s*", re.IGNORECASE)
_NAME_TOKEN = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)*")


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


def _strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _name_tokens(part: str) -> list[str]:
    cleaned = _strip_accents(part).replace("'", "").replace("’", "")
    tokens: list[str] = []
    for raw in _NAME_TOKEN.findall(cleaned):
        token = raw.lower().strip("-")
        if not token or token in _NAME_STOP:
            continue
        tokens.append(token)
    while tokens and tokens[-1] in _NAME_STOP:
        tokens.pop()
    return tokens


def _people_token_lists(raw: str) -> list[list[str]]:
    """Split an author string into per-person token lists.

    Handles "First Last", "Last, First", semicolon/and separators, and
    hyphenated names. Surname-only fragments are returned too; the match key
    drops them later.
    """
    text = str(raw or "").strip()
    if not text:
        return []
    people: list[list[str]] = []
    for chunk in _PERSON_SPLIT.split(text):
        chunk = chunk.strip()
        if not chunk:
            continue
        people.extend(_split_comma_names(chunk))
    return people


def _split_comma_names(chunk: str) -> list[list[str]]:
    if "," not in chunk:
        tokens = _name_tokens(chunk)
        return [tokens] if tokens else []
    parts = [part.strip() for part in chunk.split(",") if part.strip()]
    token_lists = [tokens for tokens in (_name_tokens(part) for part in parts) if tokens]
    if not token_lists:
        return []
    if len(token_lists) == 2 and len(token_lists[0]) == 1 and 1 <= len(token_lists[1]) <= 3:
        return [token_lists[1] + token_lists[0]]
    if all(len(tokens) >= 2 for tokens in token_lists):
        return token_lists
    if all(len(tokens) == 1 for tokens in token_lists) and len(token_lists) % 2 == 0:
        return [
            token_lists[index + 1] + token_lists[index]
            for index in range(0, len(token_lists), 2)
        ]
    people: list[list[str]] = []
    index = 0
    while index < len(token_lists):
        current = token_lists[index]
        nxt = token_lists[index + 1] if index + 1 < len(token_lists) else None
        if nxt is not None and len(current) == 1 and len(nxt) <= 3:
            people.append(nxt + current)
            index += 2
            continue
        people.append(current)
        index += 1
    return people


def author_match_key(tokens: list[str]) -> str | None:
    """Return ``given-initial|surname`` or None when the name is surname-only."""
    if len(tokens) < 2:
        return None
    surname = tokens[-1]
    given = tokens[0]
    if len(surname) < 2 or surname in _NAME_STOP or not given or given in _NAME_STOP:
        return None
    return f"{given[0]}|{surname}"


def iter_author_keys(names: list[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in names:
        for tokens in _people_token_lists(str(raw or "")):
            key = author_match_key(tokens)
            if not key or key in seen:
                continue
            seen.add(key)
            found.append(key)
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


def _corpus_author_strings(paper: Any) -> list[str]:
    authors = list(getattr(paper, "authors", None) or [])
    if authors:
        return [str(author) for author in authors if str(author or "").strip()]
    author_string = _pdf_author_string(str(getattr(paper, "file_path", "") or ""))
    if author_string:
        return [author_string]
    return []


def corpus_author_keys(corpus: list[Any]) -> dict[str, int]:
    """Map ``initial|surname`` to the number of library papers that name them.

    Surname-only metadata (a bare "Wang" from a PDF) does not enter the map.
    """
    counts: Counter[str] = Counter()
    for paper in corpus:
        counts.update(set(iter_author_keys(_corpus_author_strings(paper))))
    return {key: int(value) for key, value in counts.items()}


def _author_frequency_factor(count: int) -> float:
    if count >= 3:
        return 1.0
    if count == 2:
        return 0.75
    return 0.5


def author_overlap_bonus(
    authors: list[str], library_keys: dict[str, int], weight: float
) -> float:
    """Bonus for normalized full-name overlap. Capped at ``3 * weight``."""
    if weight <= 0 or not library_keys or not authors:
        return 0.0
    score = 0.0
    for key in iter_author_keys(authors):
        count = library_keys.get(key, 0)
        if count <= 0:
            continue
        score += weight * _author_frequency_factor(count)
    if score <= 0:
        return 0.0
    return min(score, 3.0 * weight)


def phrase_tokens(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall((text or "").lower())
        if token not in PHRASE_STOPWORDS and len(token) > 1
    ]


# These tokens are real words inside core phrases, but alone they match
# unrelated papers (motion capture, visual tracking, adversarial training).
# They still count inside the full phrase or a mixed bigram.
WEAK_SUPPORT_TOKENS = {
    "adversarial",
    "motion",
    "prior",
    "priors",
    "tracking",
}


def is_generic_phrase(term: str) -> bool:
    tokens = phrase_tokens(term)
    return bool(tokens) and all(token in GENERIC_TERMS for token in tokens)


def term_support_units(term: str) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    """Full phrase, mixed bigrams, and distinctive unigrams for one query term.

    A 3-word core phrase such as "humanoid whole-body control" rarely appears
    verbatim. The mixed bigrams ("humanoid whole-body", "whole-body control")
    and the distinctive leftovers ("humanoid", "whole-body") are enough to
    show the paper is on the specific topic. Generic leftovers are omitted.
    """
    tokens = phrase_tokens(term)
    if not tokens:
        return "", (), ()
    full = " ".join(tokens)
    bigrams: list[str] = []
    if len(tokens) >= 3:
        for left, right in zip(tokens, tokens[1:]):
            if left in GENERIC_TERMS and right in GENERIC_TERMS:
                continue
            bigrams.append(f"{left} {right}")
    unigrams = tuple(
        token
        for token in tokens
        if token not in GENERIC_TERMS
        and token not in WEAK_SUPPORT_TOKENS
        and len(token) > 3
    )
    return full, tuple(bigrams), unigrams


def sanitize_negative_terms(terms: list[str], limit: int = 10) -> list[str]:
    """Keep specific English exclusion phrases and drop generic single words."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in terms:
        text = " ".join(str(raw or "").split())
        key = text.lower()
        if not text or key in seen or not re.search(r"[A-Za-z]", text):
            continue
        tokens = phrase_tokens(text)
        if not tokens or all(token in GENERIC_TERMS for token in tokens):
            continue
        seen.add(key)
        cleaned.append(text)
        if len(cleaned) >= limit:
            break
    return cleaned


def feedback_negative_terms(titles: list[str], limit: int = 12) -> list[str]:
    """Specific tokens from irrelevant-feedback titles, without generic words."""
    counts: Counter[str] = Counter()
    for title in titles:
        counts.update(phrase_tokens(title))
    picked: list[str] = []
    for term, _count in counts.most_common():
        if term in GENERIC_TERMS or len(term) < 4:
            continue
        picked.append(term)
        if len(picked) >= limit:
            break
    return picked


def negative_overlap_penalty(negative_score: float, positive_score: float) -> float:
    """Scale an exclusion penalty down as core evidence gets stronger.

    ``positive_score`` is the unweighted core match score (one exact core phrase
    is 2). The raw penalty is capped at 4. It is kept in full when the core
    score is 0 and fades to 0 at a core score of 8 (about four exact core hits),
    so a strong paper that also brushes an exclusion phrase is not dropped.
    """
    if negative_score <= 0:
        return 0.0
    raw = min(float(negative_score), 4.0)
    strength = max(0.0, float(positive_score))
    return raw * max(0.0, 1.0 - strength / 8.0)


def _adjacent_count(tokens: list[str], gram: tuple[str, ...]) -> int:
    width = len(gram)
    if width <= 0 or len(tokens) < width:
        return 0
    count = 0
    for index in range(len(tokens) - width + 1):
        if tuple(tokens[index : index + width]) == gram:
            count += 1
    return count


def _bm25_term(tf: int, df: int, total_docs: int, doc_len: int, avgdl: float) -> float:
    if tf <= 0 or df <= 0 or total_docs <= 0:
        return 0.0
    k1 = 1.5
    b = 0.75
    idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
    denom = tf + k1 * (1 - b + b * doc_len / max(avgdl, 1e-6))
    return idf * (tf * (k1 + 1)) / denom


def phrase_bm25_scores(
    texts: list[str],
    phrases: list[str],
    weights: dict[str, float] | None = None,
) -> list[float]:
    """BM25 over whole phrases and, for longer phrases, adjacent bigrams.

    Component words of a multi-word phrase do not score on their own, so a
    generic token such as "learning" cannot carry a paper by itself.
    """
    if not texts or not phrases:
        return [0.0 for _ in texts]
    tokenized = [phrase_tokens(text) for text in texts]
    units: list[tuple[tuple[str, ...], tuple[tuple[str, ...], ...], float]] = []
    for phrase in phrases:
        tokens = tuple(phrase_tokens(phrase))
        if not tokens:
            continue
        if len(tokens) == 1 and tokens[0] in GENERIC_TERMS:
            continue
        weight = CORE_TERM_WEIGHT
        if weights:
            weight = float(weights.get(phrase, weights.get(phrase.lower(), weight)))
        if weight <= 0:
            continue
        bigrams: tuple[tuple[str, ...], ...] = ()
        if len(tokens) >= 3:
            bigrams = tuple(
                bigram
                for bigram in zip(tokens, tokens[1:])
                if not all(token in GENERIC_TERMS for token in bigram)
            )
        units.append((tokens, bigrams, weight))
    if not units:
        return [0.0 for _ in texts]

    grams: set[tuple[str, ...]] = set()
    for tokens, bigrams, _weight in units:
        grams.add(tokens)
        grams.update(bigrams)
    doc_freq: Counter[tuple[str, ...]] = Counter()
    for tokens in tokenized:
        if not tokens:
            continue
        present = {gram for gram in grams if _adjacent_count(tokens, gram) > 0}
        doc_freq.update(present)
    total_docs = len(tokenized)
    avgdl = sum(len(tokens) for tokens in tokenized) / max(total_docs, 1)
    scores: list[float] = []
    for tokens in tokenized:
        if not tokens:
            scores.append(0.0)
            continue
        doc_len = len(tokens)
        score = 0.0
        for phrase_tokens_, bigrams, weight in units:
            full_tf = _adjacent_count(tokens, phrase_tokens_)
            if full_tf:
                score += weight * _bm25_term(
                    full_tf, doc_freq.get(phrase_tokens_, 0), total_docs, doc_len, avgdl
                )
                continue
            for bigram in bigrams:
                bigram_tf = _adjacent_count(tokens, bigram)
                if not bigram_tf:
                    continue
                score += 0.5 * weight * _bm25_term(
                    bigram_tf, doc_freq.get(bigram, 0), total_docs, doc_len, avgdl
                )
        scores.append(score)
    return scores


def phrase_document_frequency(texts: list[str], phrases: list[str]) -> dict[str, int]:
    tokenized = [phrase_tokens(text) for text in texts]
    frequencies: dict[str, int] = {}
    for phrase in phrases:
        gram = tuple(phrase_tokens(phrase))
        key = phrase.lower()
        if not gram:
            frequencies[key] = 0
            continue
        frequencies[key] = sum(1 for tokens in tokenized if _adjacent_count(tokens, gram) > 0)
    return frequencies


def assign_term_weights(
    terms: list[str],
    texts: list[str],
    *,
    core_terms: list[str] | None = None,
    broad_terms: list[str] | None = None,
) -> dict[str, float]:
    """Weight query phrases. Missing tiers fall back to candidate-pool DF.

    A phrase whose tokens are all generic is always broad, so it cannot pass
    the core-hit gate by itself. When the profile has no tiers and the pool is
    large, the most common third of the phrases are demoted the same way.
    """
    core = {str(term).strip().lower() for term in (core_terms or []) if str(term).strip()}
    broad = {str(term).strip().lower() for term in (broad_terms or []) if str(term).strip()}
    has_tiers = bool(core or broad)
    weights: dict[str, float] = {}
    if not terms:
        return weights

    def _store(term: str, weight: float) -> None:
        weights[term] = weight
        weights[term.lower()] = weight

    if has_tiers:
        for term in terms:
            key = term.lower()
            if is_generic_phrase(term) or key in broad:
                _store(term, BROAD_TERM_WEIGHT)
            elif key in core:
                _store(term, CORE_TERM_WEIGHT)
            else:
                _store(term, CORE_TERM_WEIGHT)
        return weights

    if len(texts) < 40:
        for term in terms:
            _store(term, BROAD_TERM_WEIGHT if is_generic_phrase(term) else CORE_TERM_WEIGHT)
        return weights

    frequencies = phrase_document_frequency(texts, terms)
    total_docs = max(len(texts), 1)
    ranked = sorted(terms, key=lambda term: (-frequencies.get(term.lower(), 0), term.lower()))
    demote_count = max(1, len(terms) // 3) if len(terms) >= 2 else 0
    high_df = set(ranked[:demote_count])
    for term in terms:
        ratio = frequencies.get(term.lower(), 0) / total_docs
        if is_generic_phrase(term) or (term in high_df and ratio >= 0.05):
            _store(term, BROAD_TERM_WEIGHT)
        else:
            _store(term, CORE_TERM_WEIGHT)
    if terms and not any(weights.get(term, 0.0) >= 0.75 for term in terms):
        promoted = 0
        target = max(1, len(terms) // 2)
        for term in reversed(ranked):
            if is_generic_phrase(term):
                continue
            _store(term, CORE_TERM_WEIGHT)
            promoted += 1
            if promoted >= target:
                break
    return weights


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
