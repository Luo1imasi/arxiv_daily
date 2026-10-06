import asyncio
import hashlib
import math
import re
import threading
import time as time_module
from datetime import datetime, timezone
from typing import Any, cast, override

import feedparser
import arxiv
from arxiv import Result as ArxivResult
from loguru import logger
from tqdm import tqdm

from .base import BaseRetriever, register_retriever
from ..protocol import Paper, CorpusPaper
from ..config import get_config_value
from ..lexical import (
    assign_term_weights,
    author_overlap_bonus,
    category_preference_bonus,
    category_weights_from_corpus,
    corpus_author_keys,
    feedback_negative_terms,
    negative_overlap_penalty,
    phrase_bm25_scores,
    sanitize_negative_terms,
    term_support_units,
)
from .. import database as db
from ..business_date import announcement_window_utc as _announcement_window_utc
from ..business_date import business_window_utc as _business_window_utc
from ..business_date import get_business_date as _business_date
from ..business_date import submitted_not_after_utc as _submitted_not_after_utc
from ..utils import make_content_key

RawPaper = dict[str, object]
_ARXIV = cast(Any, arxiv)
_FEEDPARSER = cast(Any, feedparser)
_ARXIV_REQUEST_LOCK = threading.Lock()
_last_arxiv_request_at = 0.0
_ARXIV_MIN_REQUEST_INTERVAL_SECONDS = 5.0
_export_call_count = 0
LAST_RETRIEVAL_STATS: dict[str, object] = {}
_RSS_ABSTRACT_PREFIX = re.compile(
    r"^arXiv:\S+\s+Announce Type:\s*\S+\s+Abstract:\s*",
    re.IGNORECASE,
)
_ARXIV_ID_IN_ABS = re.compile(r"arxiv\.org/abs/([^?#\s]+)", re.IGNORECASE)

TOKEN_PATTERN = re.compile(r"[a-z][a-z0-9+\-\.]{1,}")
STOPWORDS = {
    "about",
    "after",
    "among",
    "also",
    "approach",
    "based",
    "between",
    "from",
    "into",
    "method",
    "model",
    "paper",
    "results",
    "show",
    "study",
    "such",
    "their",
    "these",
    "this",
    "using",
    "with",
}


def _match_with_word_boundary(text: str, keyword: str) -> bool:
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
    return bool(re.search(pattern, text.lower()))


def _paper_title(paper: RawPaper) -> str:
    return str(paper.get("title", ""))


def _paper_summary(paper: RawPaper) -> str:
    return str(paper.get("summary", ""))


def _paper_entry_id(paper: RawPaper) -> str:
    return str(paper.get("entry_id", ""))


def _paper_published(paper: RawPaper):
    published = paper.get("published")
    if not published:
        return None
    if isinstance(published, datetime):
        dt = published
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    try:
        dt = datetime.fromisoformat(str(published))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _published_in_window(
    published: datetime | None,
    window_start: datetime,
    window_end: datetime,
) -> bool:
    if published is None:
        return False
    published_utc = (
        published.replace(tzinfo=timezone.utc)
        if published.tzinfo is None
        else published.astimezone(timezone.utc)
    )
    return window_start <= published_utc < window_end


def _arxiv_date(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M")


def _wait_for_arxiv_request_slot() -> None:
    global _last_arxiv_request_at
    with _ARXIV_REQUEST_LOCK:
        now = time_module.monotonic()
        elapsed = now - _last_arxiv_request_at
        if elapsed < _ARXIV_MIN_REQUEST_INTERVAL_SECONDS:
            time_module.sleep(_ARXIV_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
        _last_arxiv_request_at = time_module.monotonic()


def _sleep_for_retry(seconds: float) -> None:
    time_module.sleep(max(0.0, seconds))


def _is_retryable_arxiv_failure(exc: Exception) -> bool:
    status = getattr(exc, "status", None) or getattr(exc, "status_code", None)
    if status in {406, 429, 503}:
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "429",
            "503",
            "406",
            "too many requests",
            "service unavailable",
            "not acceptable",
        )
    )


def _arxiv_retry_after(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else getattr(exc, "headers", None)
    if headers is None:
        return None
    raw = None
    try:
        raw = headers.get("Retry-After") or headers.get("retry-after")
    except Exception:
        raw = None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


def _run_arxiv_call(func, *, description: str, max_attempts: int = 3):
    max_attempts = max(1, int(max_attempts))
    last_exception: Exception | None = None
    for attempt in range(max_attempts):
        try:
            _wait_for_arxiv_request_slot()
            _note_export_call()
            return func()
        except Exception as exc:
            last_exception = exc
            if _looks_like_oversized_arxiv_request(exc) or not _is_retryable_arxiv_failure(exc):
                raise
            if attempt >= max_attempts - 1:
                raise
            delay = _arxiv_retry_after(exc)
            if delay is None:
                delay = min(20 * (2**attempt), 120)
            logger.warning(
                f"arXiv {description} failed ({exc}). Retrying in {delay:.0f}s "
                f"(attempt {attempt + 1}/{max_attempts})"
            )
            _sleep_for_retry(delay)
    if last_exception is not None:
        raise last_exception
    raise RuntimeError(f"arXiv {description} failed")


def _looks_like_oversized_arxiv_request(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "414",
            "uri too long",
            "url too long",
            "request-uri too long",
            "request uri too long",
        )
    )


def _raw_paper_text(value: object | None) -> str | None:
    return value if isinstance(value, str) else None


def _raw_paper_authors(paper: RawPaper) -> list[str]:
    authors = paper.get("authors", [])
    if not isinstance(authors, list):
        return []
    return [str(author) for author in authors]


def _raw_paper_float(value: object | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _paper_retrieval_score(paper: RawPaper) -> float:
    return _raw_paper_float(paper.get("retrieval_score"), 0.0)


def _paper_lookback_score(paper: RawPaper) -> float:
    return _raw_paper_float(paper.get("lookback_score"), 0.0)


def _to_raw_paper(paper: ArxivResult | RawPaper) -> RawPaper:
    if isinstance(paper, dict):
        return dict(paper)

    code_url = None
    for link in getattr(paper, "links", []):
        if "github.com" in str(link):
            code_url = str(link)
            break

    published = getattr(paper, "published", None)
    primary_category = getattr(getattr(paper, "primary_category", None), "term", None)
    categories = [
        str(getattr(category, "term", category))
        for category in getattr(paper, "categories", [])
    ]
    return {
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "summary": paper.summary,
        "entry_id": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "code_url": code_url,
        "published": published.isoformat() if published else None,
        "primary_category": primary_category,
        "categories": categories,
        "retrieval_score": float(getattr(paper, "retrieval_score", 0.0) or 0.0),
        "lookback_score": float(getattr(paper, "lookback_score", 0.0) or 0.0),
    }


def _raw_paper_primary_category(paper: RawPaper) -> str | None:
    value = paper.get("primary_category")
    return value if isinstance(value, str) and value else None


def _raw_paper_categories(paper: RawPaper) -> list[str]:
    values = paper.get("categories", [])
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if value]


def _note_export_call() -> None:
    global _export_call_count
    _export_call_count += 1


def _entry_value(entry: Any, key: str, default: Any = None) -> Any:
    if isinstance(entry, dict):
        return entry.get(key, default)
    getter = getattr(entry, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(entry, key, default)


def _rss_short_id(entry: Any) -> str:
    raw = str(_entry_value(entry, "id", "") or "").strip()
    if raw.startswith("oai:arXiv.org:"):
        return raw.removeprefix("oai:arXiv.org:").strip()
    match = _ARXIV_ID_IN_ABS.search(raw)
    if match:
        return match.group(1).strip()
    link = str(_entry_value(entry, "link", "") or "")
    match = _ARXIV_ID_IN_ABS.search(link)
    if match:
        return match.group(1).strip()
    if raw and " " not in raw and not raw.startswith("http"):
        return raw
    return ""


def _rss_authors(entry: Any) -> list[str]:
    raw_names: list[str] = []
    authors = _entry_value(entry, "authors", []) or []
    if isinstance(authors, list):
        for author in authors:
            if isinstance(author, dict):
                raw_names.append(str(author.get("name") or ""))
            else:
                raw_names.append(str(getattr(author, "name", author)))
    if not raw_names:
        author = _entry_value(entry, "author", "")
        if author:
            raw_names.append(str(author))
    names: list[str] = []
    seen: set[str] = set()
    for raw in raw_names:
        parts = [part.strip() for part in str(raw).split(",")] if "," in str(raw) else [str(raw).strip()]
        for part in parts:
            key = part.lower()
            if not part or key in seen:
                continue
            seen.add(key)
            names.append(part)
    return names


def _rss_categories(entry: Any) -> tuple[str | None, list[str]]:
    categories: list[str] = []
    tags = _entry_value(entry, "tags", []) or []
    if isinstance(tags, list):
        for tag in tags:
            if isinstance(tag, dict):
                term = str(tag.get("term") or "").strip()
            else:
                term = str(getattr(tag, "term", "") or "").strip()
            if term and term not in categories:
                categories.append(term)
    primary = categories[0] if categories else None
    return primary, categories


def _clean_rss_summary(summary: str) -> str:
    text = str(summary or "").replace("\r\n", "\n").strip()
    text = _RSS_ABSTRACT_PREFIX.sub("", text).strip()
    text = re.split(r"\n(?:Journal-ref|DOI|Proxy|MSC classes)\s*:", text, maxsplit=1)[0].strip()
    return text


def _rss_published_iso(entry: Any) -> str | None:
    """Announcement time from Atom <published>, not the feed <updated> stamp.

    rss.arxiv.org sets <published> to midnight US/Eastern on the announcement
    day. <updated> is the feed generation time and is nearly identical for
    every entry, so it must not drive the business window or freshness score.
    """
    raw = _entry_value(entry, "published", None)
    if not raw:
        return None
    try:
        published = datetime.fromisoformat(str(raw).strip())
    except ValueError:
        return None
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    return published.isoformat()


def _rss_pdf_url(entry: Any, short_id: str) -> str | None:
    links = _entry_value(entry, "links", []) or []
    if isinstance(links, list):
        for link in links:
            if not isinstance(link, dict):
                continue
            href = str(link.get("href") or "")
            title = str(link.get("title") or "").lower()
            link_type = str(link.get("type") or "").lower()
            if href and ("pdf" in title or "pdf" in link_type or "/pdf/" in href):
                return href
    if short_id:
        return f"http://arxiv.org/pdf/{short_id}"
    return None


def _rss_code_url(entry: Any) -> str | None:
    links = _entry_value(entry, "links", []) or []
    if not isinstance(links, list):
        return None
    for link in links:
        href = str(link.get("href") if isinstance(link, dict) else link)
        if "github.com" in href:
            return href
    return None


def _raw_paper_from_rss_entry(entry: Any) -> RawPaper | None:
    short_id = _rss_short_id(entry)
    if not short_id:
        return None
    primary_category, categories = _rss_categories(entry)
    return {
        "title": " ".join(str(_entry_value(entry, "title", "") or "").split()),
        "authors": _rss_authors(entry),
        "summary": _clean_rss_summary(str(_entry_value(entry, "summary", "") or "")),
        "entry_id": f"http://arxiv.org/abs/{short_id}",
        "pdf_url": _rss_pdf_url(entry, short_id),
        "code_url": _rss_code_url(entry),
        "published": _rss_published_iso(entry),
        "primary_category": primary_category,
        "categories": categories,
        "announce_type": str(_entry_value(entry, "arxiv_announce_type", "") or "new"),
        "retrieval_score": 0.0,
        "lookback_score": 0.0,
    }


def _rss_paper_missing_critical_fields(paper: RawPaper | None) -> bool:
    if paper is None:
        return True
    if not _paper_title(paper).strip():
        return True
    if not _paper_summary(paper).strip():
        return True
    if not _paper_entry_id(paper).strip():
        return True
    if not _raw_paper_authors(paper):
        return True
    if _paper_published(paper) is None:
        return True
    if not _raw_paper_primary_category(paper) and not _raw_paper_categories(paper):
        return True
    return False


def _matches_category_policy(
    paper: RawPaper, categories: list[str], *, include_cross: bool
) -> bool:
    category_set = set(categories)
    if include_cross:
        paper_categories = set(_raw_paper_categories(paper))
        primary_category = _raw_paper_primary_category(paper)
        if primary_category:
            paper_categories.add(primary_category)
        return not paper_categories or bool(paper_categories & category_set)

    primary_category = _raw_paper_primary_category(paper)
    return primary_category is None or primary_category in category_set


def _paper_with_scores(
    paper: RawPaper,
    retrieval_score: float,
    lookback_score: float,
    bm25_score: float = 0.0,
) -> RawPaper:
    snapshot = dict(paper)
    snapshot["retrieval_score"] = float(retrieval_score)
    snapshot["lookback_score"] = float(lookback_score)
    snapshot["bm25_score"] = float(bm25_score)
    return snapshot


def _compute_keyword_match_score(
    paper: RawPaper,
    keywords: list[str],
    weights: dict[str, float] | None = None,
    *,
    partial: bool = True,
) -> tuple[float, float, float, float]:
    """Return weighted score, weighted hits, core score, and core hits.

    Core hits ignore broad phrases (weight below 0.75). The core score is
    unweighted so one exact core phrase is always worth 2 when the penalty
    taper is applied. Partial support (a mixed bigram or a distinctive
    unigram) is only for positive core terms. Exclusion phrases stay exact,
    so "visual slam" does not fire on the word "visual".
    """
    text = f"{_paper_title(paper)} {_paper_summary(paper)}".lower()
    score = 0.0
    match_count = 0.0
    core_score = 0.0
    core_hits = 0.0
    credited_bigrams: set[str] = set()
    credited_unigrams: set[str] = set()

    for kw in keywords:
        kw_lower = kw.lower().strip()
        if not kw_lower:
            continue
        weight = 1.0
        if weights:
            weight = float(weights.get(kw, weights.get(kw_lower, 1.0)))
        if weight <= 0:
            continue
        core = weight >= 0.75
        full, bigrams, unigrams = term_support_units(kw_lower)
        phrase = full or kw_lower
        if _match_with_word_boundary(text, phrase):
            score += 2.0 * weight
            match_count += weight
            if core:
                core_score += 2.0
                core_hits += 1.0
                credited_bigrams.update(bigrams)
                credited_unigrams.update(unigrams)
        elif phrase in text:
            score += 0.5 * weight
            match_count += 0.5 * weight
            if core:
                core_score += 0.5
                core_hits += 0.5
                credited_bigrams.update(bigrams)
                credited_unigrams.update(unigrams)
        elif core and partial:
            bigram = next(
                (
                    unit
                    for unit in bigrams
                    if unit not in credited_bigrams and _match_with_word_boundary(text, unit)
                ),
                None,
            )
            if bigram:
                credited_bigrams.add(bigram)
                score += 1.0 * weight
                match_count += 0.5 * weight
                core_score += 1.0
                core_hits += 1.0
                continue
            unigram = next(
                (
                    unit
                    for unit in unigrams
                    if unit not in credited_unigrams and _match_with_word_boundary(text, unit)
                ),
                None,
            )
            if unigram:
                credited_unigrams.add(unigram)
                score += 0.75 * weight
                match_count += 0.35 * weight
                core_score += 0.75
                core_hits += 1.0

    return score, match_count, core_score, core_hits


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN_PATTERN.findall((text or "").lower())
        if token not in STOPWORDS and len(token) > 2
    ]


def _extract_local_keywords_from_corpus(corpus: list[CorpusPaper], limit: int) -> list[str]:
    if not corpus:
        return []

    phrase_doc_freq: dict[str, int] = {}
    phrase_scores: dict[str, float] = {}
    for paper in corpus:
        title_tokens = _tokenize(paper.title)
        abstract_tokens = _tokenize(paper.abstract or "")
        doc_terms = set(title_tokens)
        doc_terms.update(abstract_tokens)

        title_bigrams = list(zip(title_tokens, title_tokens[1:]))
        abstract_bigrams = list(zip(abstract_tokens, abstract_tokens[1:]))
        doc_phrases = {
            f"{a} {b}"
            for a, b in title_bigrams + abstract_bigrams
            if a != b and a not in STOPWORDS and b not in STOPWORDS
        }

        for term in doc_terms:
            phrase_doc_freq[term] = phrase_doc_freq.get(term, 0) + 1
            phrase_scores[term] = phrase_scores.get(term, 0.0) + (
                1.0 + (1.5 if term in title_tokens else 0.0)
            )
        for phrase in doc_phrases:
            phrase_doc_freq[phrase] = phrase_doc_freq.get(phrase, 0) + 1
            phrase_scores[phrase] = phrase_scores.get(phrase, 0.0) + (
                2.0 + (1.5 if phrase in " ".join(title_tokens) else 0.0)
            )

    total_docs = max(len(corpus), 1)
    scored: list[tuple[float, int, str]] = []
    for phrase, doc_freq in phrase_doc_freq.items():
        if total_docs >= 10 and doc_freq / total_docs > 0.6:
            continue
        idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1.0
        scored.append((phrase_scores[phrase] * idf, doc_freq, phrase))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return [phrase for _, _, phrase in scored[:limit]]


def _compute_bm25_scores(
    papers: list[RawPaper],
    query_terms: list[str],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    if not papers or not query_terms:
        return {}
    texts = [f"{_paper_title(paper)} {_paper_summary(paper)}" for paper in papers]
    scores = phrase_bm25_scores(texts, query_terms, weights)
    return {
        _paper_entry_id(paper): float(score)
        for paper, score in zip(papers, scores)
    }


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    max_score = max(values)
    min_score = min(values)
    if max_score <= min_score:
        return {key: 0.0 for key in scores}
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}


def _dedupe_arxiv_results(papers: object) -> list[ArxivResult | RawPaper]:
    if not isinstance(papers, list):
        return []
    seen_ids = set()
    deduped: list[ArxivResult | RawPaper] = []
    for paper in papers:
        if not isinstance(paper, (dict, ArxivResult)):
            continue
        paper_id = _paper_entry_id(_to_raw_paper(paper))
        if not paper_id or paper_id in seen_ids:
            continue
        seen_ids.add(paper_id)
        deduped.append(paper)
    return deduped


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    name = "arxiv"

    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        self.source_config: dict[str, Any] = cast(
            dict[str, Any], get_config_value(config, "source.arxiv")
        )
        self.executor_config: dict[str, Any] = cast(
            dict[str, Any], get_config_value(config, "executor", {})
        )
        if not self.source_config.get("category"):
            raise ValueError("arxiv category must be specified in config")
        self.use_keyword_search = bool(get_config_value(config, "source.arxiv.use_keyword_search"))
        self.max_keywords = int(get_config_value(config, "source.arxiv.max_keywords"))
        self.keyword_corpus_limit = int(get_config_value(config, "source.arxiv.keyword_corpus_limit"))
        self.keyword_max_doc_freq = float(get_config_value(config, "source.arxiv.keyword_max_doc_freq"))
        self.candidate_cache_ttl_minutes = int(
            get_config_value(config, "source.arxiv.candidate_cache_ttl_minutes")
        )
        self.keyword_fallback_min_results = int(
            get_config_value(config, "source.arxiv.keyword_fallback_min_results")
        )
        self.pre_rerank_limit = int(get_config_value(config, "source.arxiv.pre_rerank_limit"))
        self.initial_recent_days = max(1, int(get_config_value(config, "source.arxiv.recent_days")))
        self.max_recent_days = max(
            self.initial_recent_days,
            int(get_config_value(config, "source.arxiv.max_recent_days")),
        )
        max_paper_num = max(1, int(get_config_value(config, "executor.max_paper_num")))
        self.lookback_top_n = max(1, min(8, max(3, max_paper_num // 4)))
        self.lookback_min_strong_papers = max(1, min(3, self.lookback_top_n // 2))
        self.lookback_score_threshold = float(
            get_config_value(config, "source.arxiv.lookback_score_threshold")
        )
        self.recency_half_life_days = max(
            1.0,
            float(get_config_value(config, "source.arxiv.recency_half_life_days")),
        )
        self.business_date = _business_date(config)
        self._active_terms: list[str] = []
        self._negative_terms: list[str] = []
        self._reset_retrieval_counters()
        if "arxiv_fetch_workers" in self.executor_config:
            logger.warning(
                "executor.arxiv_fetch_workers is deprecated and ignored; "
                "use executor.arxiv_id_batch_size to tune arXiv ID lookup batches"
            )

    def _reset_retrieval_counters(self) -> None:
        self.rss_calls = 0
        self.id_fallback_count = 0
        self._rss_announced = 0
        self._rss_direct = 0
        self._term_query_count = 0
        self._lookback_rounds = 0
        self._rss_seconds = 0.0
        self._search_seconds = 0.0
        self._rank_seconds = 0.0
        self._semantic_seconds = 0.0
        self._last_rank_stats: dict[str, int] = {}
        self._announcement_meta: dict[str, object] = {}
        self._author_keys_cache: dict[str, int] | None = None
        self._category_weights_cache: dict[str, float] | None = None

    def _publish_retrieval_stats(self, export_calls: int) -> None:
        rank_stats = dict(self._last_rank_stats)
        LAST_RETRIEVAL_STATS.clear()
        LAST_RETRIEVAL_STATS.update(
            {
                "rss_seconds": round(self._rss_seconds, 3),
                "search_seconds": round(self._search_seconds, 3),
                "rank_seconds": round(self._rank_seconds, 3),
                "semantic_seconds": round(self._semantic_seconds, 3),
                "export_calls": export_calls,
                "rss_calls": self.rss_calls,
                "id_fallback_count": self.id_fallback_count,
                "rss_announced": self._rss_announced,
                "rss_direct": self._rss_direct,
                "term_query_count": self._term_query_count,
                "lookback_rounds": self._lookback_rounds,
                **rank_stats,
                **self._announcement_meta,
            }
        )

    def _candidate_cache_key(
        self,
        categories: list[str],
        *,
        start_days_ago: int = 0,
        end_days_ago: int | None = None,
    ) -> str:
        lookback_end = self.initial_recent_days if end_days_ago is None else int(end_days_ago)
        payload = {
            "categories": list(categories),
            "include_cross_list": bool(get_config_value(self.config, "source.arxiv.include_cross_list")),
            "business_date": self.business_date.isoformat(),
            "timezone": str(get_config_value(self.config, "executor.timezone")),
            "start_days_ago": int(start_days_ago),
            "recent_days": lookback_end,
            "recent_max_results": int(get_config_value(self.config, "source.arxiv.recent_max_results")),
            "query_terms": [term.lower() for term in self._active_terms],
        }
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

    @override
    def _retrieve_raw_papers(self) -> list[RawPaper]:
        categories = self.source_config["category"]
        if isinstance(categories, str):
            categories = [categories]
        elif isinstance(categories, list):
            categories = [str(category) for category in categories]
        else:
            raise ValueError("arxiv category must be a string or list of strings")

        global _export_call_count
        export_started = _export_call_count
        self._reset_retrieval_counters()
        try:
            if bool(self.executor_config.get("announcement_backfill")):
                logger.info("Using announcement backfill mode")
                return self._announcement_backfill_retrieval(categories)
            if self.use_keyword_search:
                logger.info("Using keyword-based search mode")
                return self._keyword_based_retrieval(categories)
            logger.info("Using RSS feed search mode")
            return self._rss_based_retrieval(categories)
        finally:
            self._publish_retrieval_stats(_export_call_count - export_started)

    def _keyword_based_retrieval(self, categories: list[str]) -> list[RawPaper]:
        keywords: list[str] = []
        if self._corpus:
            keywords, negative_terms = asyncio.run(self._resolve_search_terms())
            self._negative_terms = negative_terms
        else:
            logger.warning("No corpus for keyword extraction, using category retrieval")
        self._active_terms = keywords
        try:
            candidate_pool = self._collect_candidate_pool(
                categories,
                start_days_ago=0,
                end_days_ago=self.initial_recent_days,
            )
            if not candidate_pool:
                logger.warning("No candidate papers collected from arXiv")
                return []
            if not keywords:
                return candidate_pool
            logger.info(f"Using {len(keywords)} keywords: {keywords[:10]}...")
            return self._retrieve_with_adaptive_lookback(categories, keywords, candidate_pool)
        finally:
            self._active_terms = []

    def _retrieve_with_adaptive_lookback(
        self,
        categories: list[str],
        keywords: list[str],
        candidate_pool: list[RawPaper],
    ) -> list[RawPaper]:
        ranked_papers = self._rank_candidate_pool(candidate_pool, keywords)
        if not self._should_expand_lookback(ranked_papers):
            return ranked_papers

        start_days_ago = self.initial_recent_days
        end_days_ago = min(self.initial_recent_days * 2, self.max_recent_days)

        while start_days_ago < self.max_recent_days and start_days_ago < end_days_ago:
            self._lookback_rounds += 1
            logger.info(
                f"Top recommendations look weak, shifting arXiv lookback window to {start_days_ago}-{end_days_ago} days ago"
            )
            candidate_pool = self._collect_candidate_pool(
                categories,
                start_days_ago=start_days_ago,
                end_days_ago=end_days_ago,
            )
            ranked_papers = self._rank_candidate_pool(candidate_pool, keywords)
            if not self._should_expand_lookback(ranked_papers):
                return ranked_papers
            if end_days_ago >= self.max_recent_days:
                break
            start_days_ago = end_days_ago
            end_days_ago = min(end_days_ago + self.initial_recent_days, self.max_recent_days)

        return ranked_papers

    def _should_expand_lookback(self, ranked_papers: list[RawPaper]) -> bool:
        if not ranked_papers:
            return True

        top_n = min(len(ranked_papers), self.lookback_top_n)
        top_scores = [_paper_lookback_score(paper) for paper in ranked_papers[:top_n]]
        avg_top_score = sum(top_scores) / top_n
        strong_paper_count = sum(score >= self.lookback_score_threshold for score in top_scores)
        required_strong_papers = min(self.lookback_min_strong_papers, top_n)
        if avg_top_score < self.lookback_score_threshold:
            logger.info(
                f"Average lookback score of top {top_n} candidates is {avg_top_score:.2f}, below threshold {self.lookback_score_threshold:.2f}"
            )
            return True
        if strong_paper_count < required_strong_papers:
            logger.info(
                f"Only {strong_paper_count} of top {top_n} candidates exceed lookback threshold {self.lookback_score_threshold:.2f}; need {required_strong_papers}"
            )
            return True
        return False

    async def _resolve_search_terms(self) -> tuple[list[str], list[str]]:
        profile = self.config.get("interest_profile") or {}
        if not isinstance(profile, dict):
            profile = {}
        terms = [
            str(term).strip()
            for term in profile.get("canonical_terms") or []
            if str(term).strip()
        ]
        profile_negative = [
            str(term).strip()
            for term in profile.get("negative_terms") or []
            if str(term).strip()
        ]
        if not profile_negative:
            profile_negative = [
                str(term).strip()
                for term in profile.get("not_interested") or []
                if re.search(r"[A-Za-z]", str(term))
            ]
        negative = sanitize_negative_terms(profile_negative)
        feedback = await db.list_feedback()
        feedback_titles = [
            str(row.get("title") or "")
            for row in feedback
            if row.get("vote") == "irrelevant"
        ]
        existing = {term.lower() for term in negative}
        for term in feedback_negative_terms(feedback_titles):
            if term.lower() not in existing:
                negative.append(term)
                existing.add(term.lower())
        if terms:
            limited = terms[: self.max_keywords]
            logger.info(f"Using {len(limited)} canonical terms from the interest profile")
            return limited, negative

        local_keywords = _extract_local_keywords_from_corpus(
            self._select_corpus_for_keywords(),
            limit=self.max_keywords,
        )
        logger.info(f"Using {len(local_keywords)} local fallback keywords")
        return local_keywords, negative

    def _select_corpus_for_keywords(self) -> list[CorpusPaper]:
        if not self._corpus:
            return []

        def _sort_key(paper: CorpusPaper) -> float:
            added_date = getattr(paper, "added_date", None)
            if added_date is None:
                return 0.0
            try:
                return added_date.timestamp()
            except (AttributeError, OSError, OverflowError, ValueError):
                return 0.0

        ranked = sorted(
            self._corpus,
            key=_sort_key,
            reverse=True,
        )
        if self.keyword_corpus_limit <= 0:
            return ranked
        return ranked[: self.keyword_corpus_limit]

    def _load_rss_entries(self, categories: list[str]) -> list[Any]:
        query = "+".join(categories)
        _wait_for_arxiv_request_slot()
        self.rss_calls += 1
        feed = _FEEDPARSER.parse(f"https://rss.arxiv.org/atom/{query}")
        title = ""
        feed_meta = getattr(feed, "feed", None)
        if isinstance(feed_meta, dict):
            title = str(feed_meta.get("title", ""))
        elif feed_meta is not None:
            title = str(getattr(feed_meta, "get", lambda *_: "")("title", "") or "")
        if "Feed error for query" in title:
            raise Exception(f"Invalid arxiv query: {query}")
        return list(getattr(feed, "entries", []) or [])

    def _fetch_rss_papers(self, categories: list[str]) -> list[RawPaper]:
        entries = self._load_rss_entries(categories)
        include_cross = bool(get_config_value(self.config, "source.arxiv.include_cross_list"))
        allowed_types = {"new", "cross"} if include_cross else {"new"}
        papers: list[RawPaper] = []
        fallback_ids: list[str] = []
        seen_fallback: set[str] = set()
        announced = 0
        for entry in entries:
            announce = str(_entry_value(entry, "arxiv_announce_type", "") or "new")
            if announce not in allowed_types:
                continue
            announced += 1
            paper = _raw_paper_from_rss_entry(entry)
            if _rss_paper_missing_critical_fields(paper):
                short_id = _rss_short_id(entry)
                if short_id and short_id not in seen_fallback:
                    seen_fallback.add(short_id)
                    fallback_ids.append(short_id)
                continue
            papers.append(cast(RawPaper, paper))

        if bool(self.executor_config.get("debug", False)):
            papers = papers[:10]
            fallback_ids = fallback_ids[:10]

        direct_count = len(papers)
        if fallback_ids:
            self.id_fallback_count += len(fallback_ids)
            logger.info(
                f"RSS entries missing critical fields; ID lookup for {len(fallback_ids)} papers"
            )
            for result in self._fetch_papers_by_ids(fallback_ids):
                papers.append(_to_raw_paper(result))

        self._rss_announced += announced
        self._rss_direct += direct_count
        logger.info(
            f"Parsed {direct_count} RSS papers directly "
            f"({announced} announced, {len(fallback_ids)} ID fallbacks)"
        )
        return papers

    def _fetch_rss_paper_ids(self, categories: list[str]) -> list[str]:
        return [
            _paper_entry_id(paper).rsplit("/", 1)[-1]
            for paper in self._fetch_rss_papers(categories)
            if _paper_entry_id(paper)
        ]

    def _collect_candidate_pool(
        self,
        categories: list[str],
        *,
        start_days_ago: int = 0,
        end_days_ago: int | None = None,
    ) -> list[RawPaper]:
        cache_key = self._candidate_cache_key(
            categories,
            start_days_ago=start_days_ago,
            end_days_ago=end_days_ago,
        )
        cached_pool = asyncio.run(db.load_candidate_cache(cache_key))
        if cached_pool is not None:
            logger.info(f"Using cached candidate pool: {len(cached_pool)} papers")
            return [dict(paper) for paper in cached_pool]

        window_start, window_end = _business_window_utc(
            self.config,
            start_days_ago=start_days_ago,
            end_days_ago=end_days_ago or self.initial_recent_days,
        )

        def _fetch_rss_candidates(_: str) -> tuple[str, list[ArxivResult | RawPaper]]:
            started = time_module.perf_counter()
            rss_papers = self._fetch_rss_papers(categories)
            self._rss_seconds += time_module.perf_counter() - started
            return "rss", [
                paper
                for paper in rss_papers
                if _published_in_window(
                    _paper_published(_to_raw_paper(paper)),
                    window_start,
                    window_end,
                )
            ]

        def _fetch_recent_candidates(_: str) -> tuple[str, list[ArxivResult | RawPaper]]:
            started = time_module.perf_counter()
            recent_papers: list[ArxivResult | RawPaper] = list(self._fetch_recent_category_papers(
                categories,
                start_days_ago=start_days_ago,
                end_days_ago=end_days_ago,
            ))
            self._search_seconds += time_module.perf_counter() - started
            return "recent", recent_papers

        source_labels = ["recent"]
        if start_days_ago <= 0:
            source_labels.insert(0, "rss")

        candidate_sources = [
            _fetch_rss_candidates(source) if source == "rss" else _fetch_recent_candidates(source)
            for source in source_labels
        ]
        source_results: dict[str, list[ArxivResult | RawPaper]] = {
            label: papers for label, papers in candidate_sources
        }
        rss_papers = source_results.get("rss", [])
        recent_papers = source_results.get("recent", [])
        candidate_pool = [
            _to_raw_paper(paper)
            for paper in _dedupe_arxiv_results(rss_papers + recent_papers)
        ]
        asyncio.run(
            db.save_candidate_cache(
                cache_key,
                candidate_pool,
                ttl_minutes=self.candidate_cache_ttl_minutes,
            )
        )
        logger.info(
            f"Collected {len(candidate_pool)} candidate papers ({len(rss_papers)} RSS + {len(recent_papers)} recent)"
        )
        return candidate_pool

    def _fetch_recent_category_papers(
        self,
        categories: list[str],
        *,
        start_days_ago: int = 0,
        end_days_ago: int | None = None,
    ) -> list[RawPaper]:
        lookback_days = self.initial_recent_days if end_days_ago is None else int(end_days_ago)
        max_results = int(get_config_value(self.config, "source.arxiv.recent_max_results"))
        start_days_ago = max(0, int(start_days_ago))
        if lookback_days <= 0 or max_results <= 0 or start_days_ago >= lookback_days:
            return []

        window_start, window_end = _business_window_utc(
            self.config,
            start_days_ago=start_days_ago,
            end_days_ago=lookback_days,
        )
        date_clause = (
            f"submittedDate:[{_arxiv_date(window_start)} TO {_arxiv_date(window_end)}]"
        )
        include_cross = bool(get_config_value(self.config, "source.arxiv.include_cross_list"))
        term_queries = self._term_search_queries(self._active_terms)
        self._term_query_count += len(term_queries)
        papers: list[RawPaper] = []
        if term_queries:
            per_query = min(
                max_results,
                int(get_config_value(self.config, "source.arxiv.keyword_query_max_results", 80)),
            )
            failures: list[Exception] = []
            for term_query in term_queries:
                try:
                    papers.extend(
                        self._search_arxiv(
                            f"{term_query} AND {date_clause}",
                            per_query,
                            window_start,
                            window_end,
                            categories,
                            include_cross=include_cross,
                        )
                    )
                except Exception as exc:
                    failures.append(exc)
                    logger.warning(f"arXiv term query failed: {exc}")
            papers = [
                _to_raw_paper(paper)
                for paper in _dedupe_arxiv_results(papers)
            ]
            if not papers and failures:
                raise failures[-1]
            if not papers:
                logger.warning(
                    "Canonical term queries returned no papers; using one capped category query"
                )
                category_query = " OR ".join(f"cat:{category}" for category in categories)
                papers = self._search_arxiv(
                    f"({category_query}) AND {date_clause}",
                    min(max_results, 100),
                    window_start,
                    window_end,
                    categories,
                    include_cross=include_cross,
                )
        else:
            category_query = " OR ".join(f"cat:{category}" for category in categories)
            papers = self._search_arxiv(
                f"({category_query}) AND {date_clause}",
                max_results,
                window_start,
                window_end,
                categories,
                include_cross=include_cross,
            )

        if bool(self.executor_config.get("debug", False)):
            papers = papers[:20]
        logger.info(f"Fetched {len(papers)} recent papers from category search")
        return papers

    def _term_clause(self, term: str) -> str:
        if " " in term:
            return f'abs:"{term}" OR ti:"{term}"'
        return f"abs:{term} OR ti:{term}"

    def _term_group_query(self, terms: list[str]) -> str:
        return "(" + " OR ".join(self._term_clause(term) for term in terms) + ")"

    def _term_search_queries(self, terms: list[str]) -> list[str]:
        group_size = max(1, int(get_config_value(self.config, "source.arxiv.keyword_query_group_size", 6)))
        max_groups = max(1, int(get_config_value(self.config, "source.arxiv.keyword_query_max_groups", 3)))
        max_chars = max(
            1,
            int(get_config_value(self.config, "source.arxiv.keyword_query_max_chars", 1100)),
        )
        cleaned: list[str] = []
        seen: set[str] = set()
        for term in terms:
            safe = re.sub(r'["\\]', "", str(term)).strip()
            key = safe.lower()
            if len(safe) < 3 or key in seen:
                continue
            seen.add(key)
            cleaned.append(safe)
        if self.max_keywords > 0:
            cleaned = cleaned[: self.max_keywords]
        if not cleaned:
            return []

        # Prefer the configured group size, but grow it so every term still fits
        # in about max_groups requests. A group that would be too long is split
        # instead of dropping the remaining terms.
        target_size = max(group_size, math.ceil(len(cleaned) / max_groups))
        queries: list[str] = []
        index = 0
        while index < len(cleaned):
            size = min(target_size, len(cleaned) - index)
            while size > 1 and len(self._term_group_query(cleaned[index : index + size])) > max_chars:
                size -= 1
            query = self._term_group_query(cleaned[index : index + size])
            if len(query) > max_chars:
                logger.warning(
                    f"arXiv term query is {len(query)} chars, above the {max_chars} budget"
                )
            queries.append(query)
            index += size
        logger.info(f"Built {len(queries)} arXiv term queries covering {len(cleaned)} keywords")
        return queries

    def _search_arxiv(
        self,
        query: str,
        max_results: int,
        window_start: datetime,
        window_end: datetime,
        categories: list[str],
        *,
        include_cross: bool,
        page_size: int | None = None,
        max_attempts: int | None = None,
    ) -> list[RawPaper]:
        if max_results <= 0:
            return []
        search = _ARXIV.Search(
            query=query,
            max_results=max_results,
            sort_by=_ARXIV.SortCriterion.SubmittedDate,
            sort_order=_ARXIV.SortOrder.Descending,
        )
        resolved_page = min(max_results, 200) if page_size is None else int(page_size)
        resolved_page = max(1, min(resolved_page, max_results))
        client = _ARXIV.Client(num_retries=2, delay_seconds=5, page_size=resolved_page)

        def _fetch() -> list[RawPaper]:
            found: list[RawPaper] = []
            for result in client.results(search):
                raw_paper = _to_raw_paper(result)
                published = _paper_published(raw_paper)
                if not _published_in_window(published, window_start, window_end):
                    if published is None:
                        continue
                    published_utc = (
                        published.replace(tzinfo=timezone.utc)
                        if published.tzinfo is None
                        else published.astimezone(timezone.utc)
                    )
                    if published_utc < window_start:
                        break
                    continue
                if not _matches_category_policy(raw_paper, categories, include_cross=include_cross):
                    continue
                found.append(raw_paper)
            return found

        return _run_arxiv_call(
            _fetch,
            description="search",
            max_attempts=3 if max_attempts is None else max_attempts,
        )

    def _announcement_limits(self) -> tuple[int, int, int]:
        page_size = int(get_config_value(self.config, "source.arxiv.announcement_page_size", 50))
        per_category = int(
            get_config_value(self.config, "source.arxiv.announcement_category_max_results", 100)
        )
        max_attempts = int(get_config_value(self.config, "source.arxiv.announcement_max_attempts", 5))
        # arXiv asks for at least 3 seconds between requests. The shared slot
        # wait is 5 seconds, and each page stays at or below 50 results.
        page_size = max(1, min(page_size, 50))
        per_category = max(1, min(per_category, 100))
        return page_size, per_category, max(1, max_attempts)

    def _announcement_cache_key(
        self,
        categories: list[str],
        window_start: datetime,
        window_end: datetime,
    ) -> str:
        payload = {
            "mode": "announcement",
            "categories": list(categories),
            "include_cross_list": bool(get_config_value(self.config, "source.arxiv.include_cross_list")),
            "business_date": self.business_date.isoformat(),
            "window_start": window_start.astimezone(timezone.utc).strftime("%Y%m%d%H%M"),
            "window_end": window_end.astimezone(timezone.utc).strftime("%Y%m%d%H%M"),
            "query_terms": [term.lower() for term in self._active_terms],
            "page_size": self._announcement_limits()[0],
            "per_category": self._announcement_limits()[1],
        }
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

    def _paper_submitted_in_bounds(
        self,
        paper: RawPaper,
        window_start: datetime,
        window_end: datetime,
        not_after: datetime,
    ) -> bool:
        published = _paper_published(paper)
        if not _published_in_window(published, window_start, window_end):
            return False
        if published is None:
            return False
        published_utc = (
            published.replace(tzinfo=timezone.utc)
            if published.tzinfo is None
            else published.astimezone(timezone.utc)
        )
        return published_utc < not_after

    def _fetch_announcement_pool(
        self,
        categories: list[str],
        window_start: datetime,
        window_end: datetime,
        not_after: datetime,
    ) -> list[RawPaper]:
        """One announcement batch: category listings plus profile term queries.

        Requests stay sequential. The shared arXiv slot enforces the pause
        between calls, 429/503 retry with exponential backoff, and each page
        is capped by announcement_page_size.
        """
        page_size, per_category, max_attempts = self._announcement_limits()
        include_cross = bool(get_config_value(self.config, "source.arxiv.include_cross_list"))
        date_clause = (
            f"submittedDate:[{_arxiv_date(window_start)} TO {_arxiv_date(window_end)}]"
        )
        papers: list[RawPaper] = []
        failures: list[Exception] = []
        for category in categories:
            try:
                papers.extend(
                    self._search_arxiv(
                        f"cat:{category} AND {date_clause}",
                        per_category,
                        window_start,
                        window_end,
                        categories,
                        include_cross=include_cross,
                        page_size=page_size,
                        max_attempts=max_attempts,
                    )
                )
            except Exception as exc:
                failures.append(exc)
                logger.warning(f"arXiv announcement query failed for {category}: {exc}")

        term_queries = self._term_search_queries(self._active_terms)
        self._term_query_count += len(term_queries)
        per_query = min(
            per_category,
            int(get_config_value(self.config, "source.arxiv.keyword_query_max_results", 80)),
        )
        for term_query in term_queries:
            try:
                papers.extend(
                    self._search_arxiv(
                        f"{term_query} AND {date_clause}",
                        per_query,
                        window_start,
                        window_end,
                        categories,
                        include_cross=include_cross,
                        page_size=page_size,
                        max_attempts=max_attempts,
                    )
                )
            except Exception as exc:
                failures.append(exc)
                logger.warning(f"arXiv announcement term query failed: {exc}")

        if not papers and failures:
            raise failures[-1]
        deduped = [
            _to_raw_paper(paper)
            for paper in _dedupe_arxiv_results(papers)
            if self._paper_submitted_in_bounds(
                _to_raw_paper(paper), window_start, window_end, not_after
            )
        ]
        if bool(self.executor_config.get("debug", False)):
            deduped = deduped[:20]
        logger.info(
            f"Fetched {len(deduped)} announcement papers "
            f"({len(categories)} categories, {len(term_queries)} term queries)"
        )
        return deduped

    def _announcement_backfill_retrieval(self, categories: list[str]) -> list[RawPaper]:
        keywords: list[str] = []
        if self._corpus:
            keywords, negative_terms = asyncio.run(self._resolve_search_terms())
            self._negative_terms = negative_terms
        else:
            logger.warning("No corpus for keyword extraction, using category retrieval")
        self._active_terms = keywords
        try:
            window_start, window_end, kind, source_date = _announcement_window_utc(self.business_date)
            not_after = _submitted_not_after_utc(self.config, self.business_date)
            if window_end > not_after:
                logger.info(
                    f"Clipping announcement window end from {window_end.isoformat()} "
                    f"to {not_after.isoformat()}"
                )
                window_end = not_after
            self._announcement_meta = {
                "announcement_kind": kind,
                "announcement_source_date": source_date,
                "announcement_window_start": window_start.isoformat(),
                "announcement_window_end": window_end.isoformat(),
            }
            logger.info(
                f"Announcement window for {self.business_date.isoformat()} "
                f"({kind}, source {source_date}): "
                f"{window_start.isoformat()} -> {window_end.isoformat()}"
            )
            cache_key = self._announcement_cache_key(categories, window_start, window_end)
            cached_pool = asyncio.run(db.load_candidate_cache(cache_key))
            if cached_pool:
                logger.info(f"Using cached announcement pool: {len(cached_pool)} papers")
                candidate_pool = [dict(paper) for paper in cached_pool]
            else:
                candidate_pool = self._fetch_announcement_pool(
                    categories, window_start, window_end, not_after
                )
                if candidate_pool:
                    asyncio.run(
                        db.save_candidate_cache(
                            cache_key,
                            candidate_pool,
                            ttl_minutes=self.candidate_cache_ttl_minutes,
                        )
                    )
            if not candidate_pool:
                logger.warning(
                    f"No announcement papers for {self.business_date.isoformat()} ({kind})"
                )
                return []
            if not keywords:
                limit = int(get_config_value(self.config, "source.arxiv.broad_backfill_limit", 100))
                return candidate_pool[:limit] if limit > 0 else candidate_pool
            return self._rank_candidate_pool(candidate_pool, keywords)
        finally:
            self._active_terms = []

    def _library_author_keys(self) -> dict[str, int]:
        if self._author_keys_cache is None:
            self._author_keys_cache = corpus_author_keys(self._corpus)
            logger.info(f"Library author keys: {len(self._author_keys_cache)}")
        return self._author_keys_cache

    def _seen_identities(self) -> tuple[set[str], set[str]]:
        urls = self.config.get("seen_paper_urls") or []
        keys = self.config.get("seen_content_keys") or []
        return {str(url) for url in urls if url}, {str(key) for key in keys if key}

    def _paper_is_seen(self, paper: RawPaper, seen_urls: set[str], seen_keys: set[str]) -> bool:
        if not seen_urls and not seen_keys:
            return False
        entry_id = _paper_entry_id(paper)
        if entry_id and entry_id in seen_urls:
            return True
        title = _paper_title(paper).strip()
        if title and seen_keys:
            if make_content_key(title, _paper_summary(paper)) in seen_keys:
                return True
        return False

    def _negative_terms_for_rank(self) -> list[str]:
        if self._negative_terms:
            return list(self._negative_terms)
        profile = self.config.get("interest_profile") or {}
        if not isinstance(profile, dict):
            return []
        terms = [str(term).strip() for term in profile.get("negative_terms") or [] if str(term).strip()]
        if not terms:
            terms = [
                str(term).strip()
                for term in profile.get("not_interested") or []
                if re.search(r"[A-Za-z]", str(term))
            ]
        return sanitize_negative_terms(terms)

    def _ranking_keywords(self, keywords: list[str]) -> list[str]:
        profile = self.config.get("interest_profile") or {}
        if not isinstance(profile, dict):
            profile = {}
        merged: list[str] = []
        seen: set[str] = set()
        extras = list(profile.get("core_terms") or []) + list(profile.get("broad_terms") or [])
        for term in list(keywords) + extras:
            text = str(term).strip()
            key = text.lower()
            if len(text) < 3 or key in seen:
                continue
            seen.add(key)
            merged.append(text)
        return merged

    def _library_category_weights(self) -> dict[str, float]:
        if self._category_weights_cache is None:
            self._category_weights_cache = category_weights_from_corpus(self._corpus)
        return self._category_weights_cache

    def _rank_candidate_pool(
        self, papers: list[RawPaper], keywords: list[str]
    ) -> list[RawPaper]:
        started = time_module.perf_counter()
        try:
            return self._rank_candidate_pool_impl(papers, keywords)
        finally:
            self._rank_seconds += time_module.perf_counter() - started

    def _rank_candidate_pool_impl(
        self, papers: list[RawPaper], keywords: list[str]
    ) -> list[RawPaper]:
        if not keywords or not papers:
            self._last_rank_stats = {
                "candidate_count": len(papers),
                "prefilter_count": 0,
                "semantic_input_count": 0,
                "held_back_count": len(papers),
                "matched_before_backfill": 0,
                "backfilled": 0,
                "tail_kept": 0,
                "final_count": 0,
            }
            return []

        keywords = self._ranking_keywords(keywords)
        logger.info(
            f"Ranking {len(papers)} candidate papers with {len(keywords)} keywords..."
        )
        logger.debug(f"Keywords: {keywords[:10]}...")

        min_match_count = float(get_config_value(self.config, "source.arxiv.min_keyword_matches"))
        use_bm25 = bool(get_config_value(self.config, "source.arxiv.use_bm25_scoring"))
        profile = self.config.get("interest_profile") or {}
        if not isinstance(profile, dict):
            profile = {}
        term_texts = [f"{_paper_title(paper)} {_paper_summary(paper)}" for paper in papers]
        term_weights = assign_term_weights(
            keywords,
            term_texts,
            core_terms=list(profile.get("core_terms") or []),
            broad_terms=list(profile.get("broad_terms") or []),
        )
        bm25_scores = _compute_bm25_scores(papers, keywords, term_weights) if use_bm25 else {}
        bm25_scores = _normalize_scores(bm25_scores)
        prefilter_limit = int(get_config_value(self.config, "source.arxiv.llm_prefilter_limit", 200))
        broad_backfill_limit = int(
            get_config_value(self.config, "source.arxiv.broad_backfill_limit", 100)
        )
        author_weight = float(get_config_value(self.config, "source.arxiv.author_overlap_weight", 0.6))
        category_scale = float(
            get_config_value(self.config, "source.arxiv.category_preference_weight", 0.8)
        )
        library_authors = self._library_author_keys() if author_weight > 0 else {}
        category_weights = self._library_category_weights() if category_scale > 0 else {}
        negative_terms = self._negative_terms_for_rank()
        seen_urls, seen_keys = self._seen_identities()

        def _recency_bonus(published: datetime | None) -> float:
            if published is None:
                return 0.0
            published_utc = (
                published.replace(tzinfo=timezone.utc)
                if published.tzinfo is None
                else published.astimezone(timezone.utc)
            )
            age_days = max(
                (_business_window_utc(self.config)[1] - published_utc).total_seconds() / 86400.0,
                0.0,
            )
            return float(math.exp(-age_days / self.recency_half_life_days))

        # hybrid, lexical, match, bm25, match_count, recency, retrieval, core_hits, negative_count, paper
        cheap_rows: list[tuple[float, float, float, float, float, float, float, float, float, RawPaper]] = []
        seen_skipped = 0
        for paper in papers:
            if self._paper_is_seen(paper, seen_urls, seen_keys):
                seen_skipped += 1
                continue
            match_score, match_count, core_score, core_hits = _compute_keyword_match_score(
                paper, keywords, term_weights
            )
            bm25_score = bm25_scores.get(_paper_entry_id(paper), 0.0)
            negative_score, negative_count, _, _ = (
                _compute_keyword_match_score(paper, negative_terms, partial=False)
                if negative_terms
                else (0.0, 0.0, 0.0, 0.0)
            )
            recency_bonus = _recency_bonus(_paper_published(paper))
            text_score = match_score + bm25_score * 4.0
            negative_penalty = (
                negative_overlap_penalty(negative_score, core_score) if negative_count else 0.0
            )
            overlap_bonus = author_overlap_bonus(
                _raw_paper_authors(paper), library_authors, author_weight
            )
            category_bonus = category_preference_bonus(
                _raw_paper_primary_category(paper),
                _raw_paper_categories(paper),
                category_weights,
                category_scale,
            )
            # Admission uses the text score only. Recency still breaks ties below.
            retrieval_score = text_score - negative_penalty
            lexical_hybrid = retrieval_score + recency_bonus * 0.75
            hybrid_score = lexical_hybrid + overlap_bonus + category_bonus
            cheap_rows.append(
                (
                    hybrid_score,
                    lexical_hybrid,
                    match_score,
                    bm25_score,
                    match_count,
                    recency_bonus,
                    retrieval_score,
                    core_hits,
                    float(negative_count),
                    paper,
                )
            )
        cheap_rows.sort(key=lambda row: (-row[0], -row[2], -row[3], -row[4]))

        # Core hits occupy the list first. Broad-only papers may fill what is
        # left, and the configured cap stops the list growing past that.
        if prefilter_limit <= 0:
            limit = len(cheap_rows)
        else:
            limit = prefilter_limit
        if broad_backfill_limit > 0:
            limit = min(limit, broad_backfill_limit)
        scored_papers: list[tuple[float, float, float, float, float, RawPaper]] = []
        seen_ids: set[str] = set()
        for (
            hybrid_score,
            _lexical_hybrid,
            match_score,
            bm25_score,
            match_count,
            recency_bonus,
            retrieval_score,
            core_hits,
            _negative_count,
            paper,
        ) in cheap_rows:
            # A broad-only hit never satisfies the gate. Recency is not part of it.
            passes_gate = core_hits >= min_match_count or (
                core_hits > 0 and retrieval_score >= 1.5
            )
            if not passes_gate:
                continue
            paper_id = _paper_entry_id(paper)
            if paper_id in seen_ids:
                continue
            seen_ids.add(paper_id)
            lookback_score = hybrid_score * (1.0 + min(match_count, 3.0))
            kept = _paper_with_scores(paper, retrieval_score, lookback_score, bm25_score)
            kept["prefilter_tier"] = "core"
            scored_papers.append(
                (
                    hybrid_score,
                    match_score,
                    bm25_score,
                    match_count,
                    recency_bonus,
                    kept,
                )
            )
            if len(scored_papers) >= limit:
                break

        matched_papers = [paper for *_, paper in scored_papers]
        matched_before_backfill = len(matched_papers)
        broad_backfill_count = 0

        # Coarse ranking is cheap, so a short core list is topped up with
        # broad hits that did not touch an exclusion phrase. They stay behind
        # every core paper and are ordered by text score alone.
        if len(matched_papers) < limit:
            broad_rows: list[tuple[float, float, RawPaper]] = []
            for (
                _hybrid_score,
                _lexical_hybrid,
                match_score,
                row_bm25,
                match_count,
                _row_recency,
                retrieval_score,
                core_hits,
                negative_count,
                paper,
            ) in cheap_rows:
                if core_hits > 0 or match_count <= 0 or negative_count > 0:
                    continue
                paper_id = _paper_entry_id(paper)
                if paper_id in seen_ids:
                    continue
                kept = _paper_with_scores(paper, retrieval_score, retrieval_score, row_bm25)
                kept["prefilter_tier"] = "broad"
                broad_rows.append((retrieval_score, match_score, kept))
            broad_rows.sort(key=lambda item: (-item[0], -item[1]))
            for _retrieval_score, _match_score, paper in broad_rows:
                paper_id = _paper_entry_id(paper)
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                matched_papers.append(paper)
                broad_backfill_count += 1
                if len(matched_papers) >= limit:
                    break

        backfilled_count = broad_backfill_count
        if not matched_papers:
            logger.warning(
                "Keyword ranking yielded no matches, falling back to BM25/recency ranking"
            )
            fallback_scored: list[tuple[float, float, float, RawPaper]] = []
            for paper in papers:
                if self._paper_is_seen(paper, seen_urls, seen_keys):
                    continue
                bm25_score = bm25_scores.get(_paper_entry_id(paper), 0.0)
                recency_bonus = _recency_bonus(_paper_published(paper))
                retrieval_score = bm25_score * 4.0
                fallback_score = retrieval_score + recency_bonus
                lookback_score = fallback_score * 0.75
                fallback_scored.append(
                    (
                        fallback_score,
                        bm25_score,
                        recency_bonus,
                        _paper_with_scores(paper, retrieval_score, lookback_score, bm25_score),
                    )
                )
            fallback_scored.sort(key=lambda item: (-item[0], -item[1], -item[2]))
            backfilled: list[RawPaper] = []
            existing_ids: set[str] = set()
            for _, _, _, paper in fallback_scored:
                paper_id = _paper_entry_id(paper)
                if paper_id in existing_ids:
                    continue
                existing_ids.add(paper_id)
                backfilled.append(paper)
                if len(backfilled) >= limit:
                    break
            backfilled_count = len(backfilled)
            if backfilled_count:
                logger.info(f"Backfilled {backfilled_count} additional BM25/recency papers")
            matched_papers = backfilled

        self._last_rank_stats = {
            "candidate_count": len(papers),
            "prefilter_count": len(matched_papers),
            "semantic_input_count": len(matched_papers),
            "held_back_count": max(len(papers) - len(matched_papers), 0),
            "matched_before_backfill": matched_before_backfill,
            "core_count": matched_before_backfill,
            "broad_backfill_count": broad_backfill_count,
            "backfilled": backfilled_count,
            "tail_kept": backfilled_count,
            "final_count": len(matched_papers),
            "seen_skipped": seen_skipped,
            "broad_backfill_limit": broad_backfill_limit,
        }
        logger.info(
            f"Cheap prefilter kept {len(matched_papers)} of {len(papers)} candidates "
            f"for LLM coarse ranking (limit={limit}, core={matched_before_backfill}, "
            f"broad_backfill={broad_backfill_count}, min_matches={min_match_count}, "
            f"bm25={use_bm25})"
        )
        if matched_papers and logger.level("DEBUG").no >= logger.level("INFO").no:
            logger.info("Top matches:")
            for index, (score, lexical, bm25, count, recency, paper) in enumerate(scored_papers[:5]):
                logger.info(
                    f"  {index + 1}. [score={score:.2f}, lexical={lexical:.2f}, bm25={bm25:.2f}, "
                    f"matches={count:.1f}, recency={recency:.2f}] {_paper_title(paper)[:60]}..."
                )

        return matched_papers

    def _fetch_papers_by_ids(self, paper_ids: list[str]) -> list[ArxivResult]:
        if not paper_ids:
            return []

        batch_size = max(
            1,
            int(get_config_value(self.config, "executor.arxiv_id_batch_size", 50)),
        )
        arxiv_client = _ARXIV.Client(
            num_retries=2, delay_seconds=5, page_size=batch_size
        )

        def fetch_batch(batch_ids: list[str]) -> list[ArxivResult]:
            search = _ARXIV.Search(id_list=batch_ids, max_results=len(batch_ids))
            return _run_arxiv_call(
                lambda: list(arxiv_client.results(search)),
                description="id lookup",
            )

        def fetch_batch_with_fallback(batch_ids: list[str]) -> list[ArxivResult]:
            try:
                return fetch_batch(batch_ids)
            except Exception as exc:
                if not _looks_like_oversized_arxiv_request(exc):
                    logger.warning(
                        f"Failed to fetch {len(batch_ids)} arxiv paper details: {exc}"
                    )
                    return []
                if len(batch_ids) <= 1:
                    logger.warning(
                        f"Failed to fetch arxiv paper details for {batch_ids[0]}: {exc}"
                    )
                    return []

                midpoint = len(batch_ids) // 2
                logger.warning(
                    f"Failed to fetch {len(batch_ids)} arxiv paper details; "
                    f"retrying as batches of {midpoint} and {len(batch_ids) - midpoint}: {exc}"
                )
                return fetch_batch_with_fallback(
                    batch_ids[:midpoint]
                ) + fetch_batch_with_fallback(batch_ids[midpoint:])
        batches = [paper_ids[i : i + batch_size] for i in range(0, len(paper_ids), batch_size)]
        results: list[list[ArxivResult]] = []
        for batch in tqdm(batches, desc="Fetching paper details"):
            results.append(fetch_batch_with_fallback(batch))

        raw_papers: list[ArxivResult] = []
        for batch_papers in results:
            if batch_papers:
                raw_papers.extend(batch_papers)

        return raw_papers

    def _rss_based_retrieval(self, categories: list[str]) -> list[RawPaper]:
        return self._collect_candidate_pool(categories)

    @override
    def convert_to_paper(self, raw_paper: RawPaper) -> Paper:
        title = _paper_title(raw_paper)
        authors = _raw_paper_authors(raw_paper)
        abstract = _paper_summary(raw_paper)
        pdf_url = _raw_paper_text(raw_paper.get("pdf_url"))
        code_url = _raw_paper_text(raw_paper.get("code_url"))
        published = _paper_published(raw_paper)
        retrieval_score = raw_paper.get("retrieval_score")

        return Paper(
            source="arxiv",
            title=title,
            authors=authors,
            abstract=abstract,
            url=_paper_entry_id(raw_paper),
            pdf_url=pdf_url,
            code_url=code_url,
            retrieval_score=_raw_paper_float(retrieval_score) if retrieval_score is not None else None,
            bm25_score=(
                _raw_paper_float(raw_paper.get("bm25_score"))
                if raw_paper.get("bm25_score") is not None
                else None
            ),
            date=published.strftime("%Y-%m-%d") if published else None,
        )
