import asyncio
import re
import math
import hashlib
from collections import Counter
from datetime import datetime, timedelta, timezone

import feedparser
import arxiv
from arxiv import Result as ArxivResult
from loguru import logger
from tqdm import tqdm

from .base import BaseRetriever, register_retriever
from ..protocol import Paper, CorpusPaper
from ..config import get_config_value
from ..utils import parallel_execute
from ..llm import extract_keywords_from_paper
from .. import database as db

RawPaper = dict[str, object]

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


def _paper_retrieval_score(paper: RawPaper) -> float:
    return float(paper.get("retrieval_score", 0.0) or 0.0)


def _paper_lookback_score(paper: RawPaper) -> float:
    return float(paper.get("lookback_score", 0.0) or 0.0)


def _to_raw_paper(paper: ArxivResult | RawPaper) -> RawPaper:
    if isinstance(paper, dict):
        return dict(paper)

    code_url = None
    for link in getattr(paper, "links", []):
        if "github.com" in str(link):
            code_url = str(link)
            break

    published = getattr(paper, "published", None)
    return {
        "title": paper.title,
        "authors": [a.name for a in paper.authors],
        "summary": paper.summary,
        "entry_id": paper.entry_id,
        "pdf_url": paper.pdf_url,
        "code_url": code_url,
        "published": published.isoformat() if published else None,
        "retrieval_score": float(getattr(paper, "retrieval_score", 0.0) or 0.0),
        "lookback_score": float(getattr(paper, "lookback_score", 0.0) or 0.0),
    }


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

    phrase_doc_freq = Counter()
    phrase_scores = Counter()
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
            phrase_doc_freq[term] += 1
            phrase_scores[term] += 1.0 + (1.5 if term in title_tokens else 0.0)
        for phrase in doc_phrases:
            phrase_doc_freq[phrase] += 1
            phrase_scores[phrase] += 2.0 + (1.5 if phrase in " ".join(title_tokens) else 0.0)

    total_docs = max(len(corpus), 1)
    scored = []
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

    tokenized_docs = []
    doc_freq = Counter()
    for paper in papers:
        tokens = _tokenize(f"{_paper_title(paper)} {_paper_summary(paper)}")
        tokenized_docs.append((_paper_entry_id(paper), tokens))
        for token in set(tokens):
            doc_freq[token] += 1

    avgdl = sum(len(tokens) for _, tokens in tokenized_docs) / max(len(tokenized_docs), 1)
    query_token_counts = Counter()
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


def _dedupe_arxiv_results(papers: list[RawPaper]) -> list[RawPaper]:
    seen_ids = set()
    deduped = []
    for paper in papers:
        paper_id = _paper_entry_id(paper)
        if not paper_id or paper_id in seen_ids:
            continue
        seen_ids.add(paper_id)
        deduped.append(paper)
    return deduped


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    name = "arxiv"

    def __init__(self, config: dict):
        super().__init__(config)
        self.source_config = get_config_value(config, "source.arxiv")
        self.executor_config = get_config_value(config, "executor")
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
        self.lookback_score_threshold = float(
            get_config_value(config, "source.arxiv.lookback_score_threshold")
        )
        self.recency_half_life_days = max(
            1.0,
            float(get_config_value(config, "source.arxiv.recency_half_life_days")),
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
            "start_days_ago": int(start_days_ago),
            "recent_days": lookback_end,
            "recent_max_results": int(get_config_value(self.config, "source.arxiv.recent_max_results")),
        }
        return hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()

    def _retrieve_raw_papers(self) -> list[RawPaper]:
        categories = self.source_config["category"]
        if isinstance(categories, str):
            categories = [categories]

        if self.use_keyword_search:
            logger.info("Using keyword-based search mode")
            return self._keyword_based_retrieval(categories)
        else:
            logger.info("Using RSS feed search mode")
            return self._rss_based_retrieval(categories)

    def _keyword_based_retrieval(self, categories: list[str]) -> list[RawPaper]:
        candidate_pool = self._collect_candidate_pool(
            categories,
            start_days_ago=0,
            end_days_ago=self.initial_recent_days,
        )
        if not candidate_pool:
            logger.warning("No candidate papers collected from arXiv")
            return []

        if not self._corpus:
            logger.warning("No corpus for keyword extraction, returning all collected papers")
            return candidate_pool

        keywords = asyncio.run(self._get_aggregated_keywords())

        if not keywords:
            logger.warning("Failed to aggregate keywords, returning all collected papers")
            return candidate_pool

        logger.info(f"Using {len(keywords)} keywords: {keywords[:10]}...")
        return self._retrieve_with_adaptive_lookback(categories, keywords, candidate_pool)

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
        avg_top_score = sum(_paper_lookback_score(paper) for paper in ranked_papers[:top_n]) / top_n
        if avg_top_score < self.lookback_score_threshold:
            logger.info(
                f"Average lookback score of top {top_n} candidates is {avg_top_score:.2f}, below threshold {self.lookback_score_threshold:.2f}"
            )
            return True
        return False

    async def _get_aggregated_keywords(self) -> list[str]:
        corpus_for_keywords = self._select_corpus_for_keywords()
        logger.info(f"Loading keywords for {len(corpus_for_keywords)} corpus papers...")
        cached_keywords = await db.load_keywords_for_papers(corpus_for_keywords)
        logger.info(f"Loaded {len(cached_keywords)} cached keywords")

        local_keywords = _extract_local_keywords_from_corpus(
            corpus_for_keywords,
            limit=max(self.max_keywords * 3, 30),
        )

        llm_enabled = bool(self.config.get("llm", {}).get("api_key"))
        papers_to_extract = []
        if llm_enabled:
            for paper in corpus_for_keywords:
                cache_key = f"{paper.title}|{paper.abstract[:50] if paper.abstract else ''}"
                if cache_key not in cached_keywords:
                    papers_to_extract.append(paper)

        if papers_to_extract:
            logger.info(
                f"Extracting keywords for {len(papers_to_extract)} new papers..."
            )
            import concurrent.futures

            def extract_single(paper):
                keywords = extract_keywords_from_paper(
                    paper.title, paper.abstract or "", self.config, max_keywords=5
                )
                return paper, keywords

            extracted_results = []
            max_workers = int(get_config_value(self.config, "executor.llm_workers"))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(extract_single, p): p for p in papers_to_extract}
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Extracting keywords",
                ):
                    try:
                        paper, keywords = future.result()
                        if keywords:
                            extracted_results.append((paper, keywords))
                    except Exception as e:
                        logger.warning(f"Failed to extract keywords: {e}")

            for paper, keywords in extracted_results:
                cache_key = (
                    f"{paper.title}|{paper.abstract[:50] if paper.abstract else ''}"
                )
                await db.save_keyword_cache(paper.title, paper.abstract or "", keywords)
                cached_keywords[cache_key] = keywords

        total_docs = max(len(corpus_for_keywords), 1)
        keyword_doc_freq = {}
        keyword_weight = {}
        for keyword in local_keywords:
            keyword_doc_freq[keyword] = keyword_doc_freq.get(keyword, 0) + 1
            keyword_weight[keyword] = keyword_weight.get(keyword, 0.0) + 1.25

        for cache_key, kw_list in cached_keywords.items():
            for kw in {k.lower().strip() for k in kw_list}:
                kw_lower = kw.lower().strip()
                if kw_lower and len(kw_lower) > 2:
                    keyword_doc_freq[kw_lower] = keyword_doc_freq.get(kw_lower, 0) + 1
                    keyword_weight[kw_lower] = keyword_weight.get(kw_lower, 0.0) + 1.0

        keyword_scores = []
        for keyword, doc_freq in keyword_doc_freq.items():
            if total_docs >= 10 and doc_freq / total_docs > self.keyword_max_doc_freq:
                continue
            idf = math.log((total_docs + 1) / (doc_freq + 1)) + 1.0
            score = keyword_weight.get(keyword, 0.0) * idf
            keyword_scores.append((keyword, score, doc_freq))

        sorted_keywords = sorted(keyword_scores, key=lambda item: (-item[1], -item[2], item[0]))
        top_keywords = [kw for kw, _, _ in sorted_keywords[: self.max_keywords]]

        logger.info(
            f"Aggregated {len(top_keywords)} keywords from {len(cached_keywords)} papers"
        )
        if top_keywords:
            logger.debug(f"Top keywords: {top_keywords}")
        return top_keywords

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

        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
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
        if cached_pool:
            logger.info(f"Using cached candidate pool: {len(cached_pool)} papers")
            return [dict(paper) for paper in cached_pool]

        def _fetch_rss_candidates(_: str) -> tuple[str, list[ArxivResult]]:
            rss_paper_ids = self._fetch_rss_paper_ids(categories)
            return "rss", self._fetch_papers_by_ids(rss_paper_ids) if rss_paper_ids else []

        def _fetch_recent_candidates(_: str) -> tuple[str, list[ArxivResult]]:
            return "recent", self._fetch_recent_category_papers(
                categories,
                start_days_ago=start_days_ago,
                end_days_ago=end_days_ago,
            )

        source_labels = ["recent"]
        if start_days_ago <= 0:
            source_labels.insert(0, "rss")

        candidate_sources = parallel_execute(
            lambda source: _fetch_rss_candidates(source)
            if source == "rss"
            else _fetch_recent_candidates(source),
            source_labels,
            max_workers=len(source_labels),
            desc="Collecting arXiv candidates",
        )
        source_results = {label: papers for label, papers in candidate_sources}
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
    ) -> list[ArxivResult]:
        lookback_days = self.initial_recent_days if end_days_ago is None else int(end_days_ago)
        max_results = int(get_config_value(self.config, "source.arxiv.recent_max_results"))
        start_days_ago = max(0, int(start_days_ago))
        if lookback_days <= 0 or max_results <= 0 or start_days_ago >= lookback_days:
            return []

        query = " OR ".join(f"cat:{category}" for category in categories)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )
        client = arxiv.Client(num_retries=5, delay_seconds=5)
        now = datetime.now(timezone.utc)
        earliest = now - timedelta(days=lookback_days)
        latest = now - timedelta(days=start_days_ago)

        papers = []
        for result in client.results(search):
            published = getattr(result, "published", None)
            if published is None:
                continue
            if published.tzinfo is None:
                published = published.replace(tzinfo=timezone.utc)
            if published > latest:
                continue
            if published < earliest:
                break
            papers.append(result)

        if bool(self.executor_config.get("debug", False)):
            papers = papers[:20]
        logger.info(f"Fetched {len(papers)} recent papers from category search")
        return papers

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
        fallback_target = max(self.keyword_fallback_min_results, self.pre_rerank_limit)

        def _recency_bonus(published) -> float:
            if published is None:
                return 0.0
            published_utc = (
                published.replace(tzinfo=timezone.utc)
                if published.tzinfo is None
                else published.astimezone(timezone.utc)
            )
            age_days = max(
                (datetime.now(timezone.utc) - published_utc).total_seconds() / 86400.0,
                0.0,
            )
            return float(math.exp(-age_days / self.recency_half_life_days))

        scored_papers = []
        seen_ids = set()
        for paper in papers:
            match_score, match_count = _compute_keyword_match_score(paper, keywords)
            bm25_score = bm25_scores.get(_paper_entry_id(paper), 0.0)

            published = _paper_published(paper)
            recency_bonus = _recency_bonus(published)

            retrieval_score = match_score + bm25_score * 4.0
            hybrid_score = retrieval_score + recency_bonus * 0.75
            lookback_score = hybrid_score * (1.0 + min(match_count, 3.0))
            if match_count >= min_match_count or hybrid_score >= 1.5:
                paper_id = _paper_entry_id(paper)
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

        fallback_scored = []
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

        arxiv_client = arxiv.Client(num_retries=5, delay_seconds=5)

        def fetch_batch(batch_ids):
            search = arxiv.Search(id_list=batch_ids)
            return list(arxiv_client.results(search))

        batches = [paper_ids[i : i + 20] for i in range(0, len(paper_ids), 20)]
        max_workers = int(get_config_value(self.config, "executor.arxiv_fetch_workers"))
        max_workers = min(max_workers, len(batches)) or 1

        results = parallel_execute(
            fetch_batch, batches, max_workers=max_workers, desc="Fetching paper details"
        )

        raw_papers = []
        for batch_papers in results:
            if batch_papers:
                raw_papers.extend(batch_papers)

        return raw_papers

    def _rss_based_retrieval(self, categories: list[str]) -> list[RawPaper]:
        return self._collect_candidate_pool(categories)

    def convert_to_paper(self, raw_paper: RawPaper) -> Paper:
        title = _paper_title(raw_paper)
        authors = [str(author) for author in raw_paper.get("authors", [])]
        abstract = _paper_summary(raw_paper)
        pdf_url = raw_paper.get("pdf_url")
        code_url = raw_paper.get("code_url")
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
            retrieval_score=float(retrieval_score) if retrieval_score is not None else None,
            date=published.strftime("%Y-%m-%d") if published else None,
        )
