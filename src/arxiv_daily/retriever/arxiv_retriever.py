import asyncio
import hashlib
import math
import re
import threading
import time as time_module
from collections import Counter
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
from .. import database as db
from ..business_date import business_window_utc as _business_window_utc
from ..business_date import get_business_date as _business_date

RawPaper = dict[str, object]
_ARXIV = cast(Any, arxiv)
_FEEDPARSER = cast(Any, feedparser)
_ARXIV_REQUEST_LOCK = threading.Lock()
_last_arxiv_request_at = 0.0
_ARXIV_MIN_REQUEST_INTERVAL_SECONDS = 5.0

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


def _run_arxiv_call(func, *, description: str):
    max_attempts = 3
    last_exception: Exception | None = None
    for attempt in range(max_attempts):
        try:
            _wait_for_arxiv_request_slot()
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
) -> RawPaper:
    snapshot = dict(paper)
    snapshot["retrieval_score"] = float(retrieval_score)
    snapshot["lookback_score"] = float(lookback_score)
    return snapshot


def _compute_keyword_match_score(
    paper: RawPaper, keywords: list[str]
) -> tuple[float, float]:
    text = f"{_paper_title(paper)} {_paper_summary(paper)}".lower()
    exact_matches = 0
    fuzzy_match_units = 0.0

    for kw in keywords:
        kw_lower = kw.lower().strip()
        if not kw_lower:
            continue

        if _match_with_word_boundary(text, kw_lower):
            exact_matches += 1
        elif kw_lower in text:
            fuzzy_match_units += 0.5

    score = exact_matches * 2.0 + fuzzy_match_units
    match_count = exact_matches + fuzzy_match_units
    return score, match_count


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


def _compute_bm25_scores(papers: list[RawPaper], query_terms: list[str]) -> dict[str, float]:
    if not papers or not query_terms:
        return {}

    tokenized_docs: list[tuple[str, list[str]]] = []
    doc_freq: Counter[str] = Counter()
    for paper in papers:
        tokens = _tokenize(f"{_paper_title(paper)} {_paper_summary(paper)}")
        tokenized_docs.append((_paper_entry_id(paper), tokens))
        for token in set(tokens):
            doc_freq[token] += 1

    avgdl = sum(len(tokens) for _, tokens in tokenized_docs) / max(len(tokenized_docs), 1)
    query_token_counts: Counter[str] = Counter()
    for term in query_terms:
        query_token_counts.update(_tokenize(term))
    if not query_token_counts:
        return {}

    k1 = 1.5
    b = 0.75
    total_docs = len(tokenized_docs)
    scores = {}
    for paper_id, tokens in tokenized_docs:
        if not tokens:
            scores[paper_id] = 0.0
            continue
        tf = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for token, qf in query_token_counts.items():
            df = doc_freq.get(token, 0)
            if not df:
                continue
            idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            freq = tf.get(token, 0)
            if not freq:
                continue
            denom = freq + k1 * (1 - b + b * doc_len / max(avgdl, 1e-6))
            score += qf * idf * (freq * (k1 + 1)) / denom
        scores[paper_id] = score
    return scores


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
        if "arxiv_fetch_workers" in self.executor_config:
            logger.warning(
                "executor.arxiv_fetch_workers is deprecated and ignored; "
                "use executor.arxiv_id_batch_size to tune arXiv ID lookup batches"
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

        if self.use_keyword_search:
            logger.info("Using keyword-based search mode")
            return self._keyword_based_retrieval(categories)
        else:
            logger.info("Using RSS feed search mode")
            return self._rss_based_retrieval(categories)

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
        negative = [
            str(term).strip()
            for term in profile.get("not_interested") or []
            if str(term).strip()
        ]
        feedback = await db.list_feedback()
        negative_counts: Counter[str] = Counter()
        for row in feedback:
            if row.get("vote") != "irrelevant":
                continue
            negative_counts.update(_tokenize(str(row.get("title") or "")))
        for term, _count in negative_counts.most_common(12):
            if term not in negative:
                negative.append(term)
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

    def _fetch_rss_paper_ids(self, categories: list[str]) -> list[str]:
        query = "+".join(categories)
        include_cross = bool(get_config_value(self.config, "source.arxiv.include_cross_list"))

        feed = _FEEDPARSER.parse(f"https://rss.arxiv.org/atom/{query}")
        if "Feed error for query" in feed.feed.get("title", ""):
            raise Exception(f"Invalid arxiv query: {query}")

        allowed_types = {"new", "cross"} if include_cross else {"new"}
        paper_ids = [
            entry.id.removeprefix("oai:arXiv.org:")
            for entry in feed.entries
            if entry.get("arxiv_announce_type", "new") in allowed_types
        ]

        if bool(self.executor_config.get("debug", False)):
            paper_ids = paper_ids[:10]

        logger.info(f"Found {len(paper_ids)} paper IDs from arxiv RSS feed")
        return paper_ids

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
            _wait_for_arxiv_request_slot()
            rss_paper_ids = self._fetch_rss_paper_ids(categories)
            rss_papers = self._fetch_papers_by_ids(rss_paper_ids) if rss_paper_ids else []
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
            recent_papers: list[ArxivResult | RawPaper] = list(self._fetch_recent_category_papers(
                categories,
                start_days_ago=start_days_ago,
                end_days_ago=end_days_ago,
            ))
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

    def _term_search_queries(self, terms: list[str]) -> list[str]:
        group_size = max(1, int(get_config_value(self.config, "source.arxiv.keyword_query_group_size", 3)))
        max_groups = max(1, int(get_config_value(self.config, "source.arxiv.keyword_query_max_groups", 3)))
        cleaned: list[str] = []
        seen: set[str] = set()
        for term in terms:
            safe = re.sub(r'["\\]', "", str(term)).strip()
            key = safe.lower()
            if len(safe) < 3 or key in seen:
                continue
            seen.add(key)
            cleaned.append(safe)
        cleaned = cleaned[: group_size * max_groups]
        queries = []
        for start in range(0, len(cleaned), group_size):
            parts = []
            for term in cleaned[start : start + group_size]:
                if " " in term:
                    parts.append(f'abs:"{term}"')
                    parts.append(f'ti:"{term}"')
                else:
                    parts.append(f"abs:{term}")
                    parts.append(f"ti:{term}")
            if parts:
                queries.append("(" + " OR ".join(parts) + ")")
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
    ) -> list[RawPaper]:
        if max_results <= 0:
            return []
        search = _ARXIV.Search(
            query=query,
            max_results=max_results,
            sort_by=_ARXIV.SortCriterion.SubmittedDate,
            sort_order=_ARXIV.SortOrder.Descending,
        )
        client = _ARXIV.Client(num_retries=2, delay_seconds=5, page_size=min(max_results, 200))

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

        return _run_arxiv_call(_fetch, description="search")

    def _semantic_scores(self, papers: list[RawPaper]) -> dict[str, float]:
        if not papers or not self._corpus:
            return {}
        try:
            from ..reranker.local import semantic_scores_for_raw_papers

            profile = self.config.get("interest_profile") or {}
            query = ""
            if isinstance(profile, dict):
                query = str(profile.get("summary") or "")
                if not query:
                    query = " ".join(str(item) for item in (profile.get("topics") or []))
            return semantic_scores_for_raw_papers(
                self.config, papers, self._corpus, query or None
            )
        except Exception as exc:
            logger.warning(f"Semantic coarse ranking skipped: {exc}")
            return {}

    def _rank_candidate_pool(
        self, papers: list[RawPaper], keywords: list[str]
    ) -> list[RawPaper]:
        if not keywords or not papers:
            return []

        logger.info(
            f"Ranking {len(papers)} candidate papers with {len(keywords)} keywords..."
        )
        logger.debug(f"Keywords: {keywords[:10]}...")

        min_match_count = float(get_config_value(self.config, "source.arxiv.min_keyword_matches"))
        use_bm25 = bool(get_config_value(self.config, "source.arxiv.use_bm25_scoring"))
        bm25_scores = _compute_bm25_scores(papers, keywords) if use_bm25 else {}
        bm25_scores = _normalize_scores(bm25_scores)
        semantic_scores = self._semantic_scores(papers)
        semantic_values = list(semantic_scores.values())
        semantic_min = min(semantic_values) if semantic_values else 0.0
        semantic_max = max(semantic_values) if semantic_values else 0.0
        semantic_mix = float(get_config_value(self.config, "source.arxiv.semantic_mix", 0.45))
        fallback_target = max(self.keyword_fallback_min_results, self.pre_rerank_limit)

        def _semantic_norm(paper_id: str) -> float:
            if not semantic_scores or semantic_max <= semantic_min:
                return 0.0
            return (semantic_scores.get(paper_id, semantic_min) - semantic_min) / (
                semantic_max - semantic_min
            )

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

        scored_papers: list[tuple[float, float, float, float, float, RawPaper]] = []
        seen_ids: set[str] = set()
        for paper in papers:
            match_score, match_count = _compute_keyword_match_score(paper, keywords)
            paper_id = _paper_entry_id(paper)
            bm25_score = bm25_scores.get(paper_id, 0.0)
            semantic_norm = _semantic_norm(paper_id)
            negative_score, negative_count = (
                _compute_keyword_match_score(paper, self._negative_terms)
                if self._negative_terms
                else (0.0, 0.0)
            )

            published = _paper_published(paper)
            recency_bonus = _recency_bonus(published)

            text_score = match_score + bm25_score * 4.0
            if semantic_scores:
                retrieval_score = (1.0 - semantic_mix) * text_score + semantic_mix * (
                    semantic_norm * 4.0
                )
            else:
                retrieval_score = text_score
            if negative_count and match_count == 0:
                retrieval_score -= min(negative_score, 4.0)
            hybrid_score = retrieval_score + recency_bonus * 0.75
            lookback_score = hybrid_score * (1.0 + min(match_count, 3.0))
            if match_count >= min_match_count or hybrid_score >= 1.5 or semantic_norm >= 0.72:
                if paper_id not in seen_ids:
                    seen_ids.add(paper_id)
                    scored_papers.append(
                        (
                            hybrid_score,
                            match_score,
                            bm25_score,
                            match_count,
                            recency_bonus,
                            _paper_with_scores(paper, retrieval_score, lookback_score),
                        )
                    )

        scored_papers.sort(key=lambda x: (-x[0], -x[1], -x[2], -x[3]))

        matched_papers = [p for *_, p in scored_papers]

        fallback_scored: list[tuple[float, float, float, RawPaper]] = []
        for paper in papers:
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
                    _paper_with_scores(paper, retrieval_score, lookback_score),
                )
            )

        fallback_scored.sort(key=lambda item: (-item[0], -item[1], -item[2]))

        if not matched_papers:
            logger.warning(
                "Keyword ranking yielded no matches, falling back to BM25/recency ranking"
            )
            matched_papers = [paper for *_, paper in fallback_scored[:fallback_target]]
        elif len(matched_papers) < fallback_target:
            existing_ids = {_paper_entry_id(paper) for paper in matched_papers}
            backfilled = list(matched_papers)
            for _, _, _, paper in fallback_scored:
                paper_id = _paper_entry_id(paper)
                if paper_id in existing_ids:
                    continue
                existing_ids.add(paper_id)
                backfilled.append(paper)
                if len(backfilled) >= fallback_target:
                    break
            if len(backfilled) > len(matched_papers):
                logger.info(
                    f"Backfilled {len(backfilled) - len(matched_papers)} additional BM25/recency papers"
                )
            matched_papers = backfilled

        if len(matched_papers) > self.pre_rerank_limit > 0:
            matched_papers = matched_papers[: self.pre_rerank_limit]

        logger.info(
            f"Found {len(matched_papers)} papers matching keywords "
            f"(min_matches={min_match_count}, bm25={use_bm25})"
        )
        if matched_papers and logger.level("DEBUG").no >= logger.level("INFO").no:
            logger.info("Top matches:")
            for i, (score, lexical, bm25, count, recency, paper) in enumerate(scored_papers[:5]):
                logger.info(
                    f"  {i + 1}. [score={score:.2f}, lexical={lexical:.2f}, bm25={bm25:.2f}, matches={count:.1f}, recency={recency:.2f}] {_paper_title(paper)[:60]}..."
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
            date=published.strftime("%Y-%m-%d") if published else None,
        )
