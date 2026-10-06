from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: str | None = None
    code_url: str | None = None
    tldr: str | None = None
    method: str | None = None
    evidence: str | None = None
    why_for_me: str | None = None
    score: float | None = None
    judge_relevance: float | None = None
    judge_reason: str | None = None
    judge_keep: bool | None = None
    retrieval_score: float | None = None
    bm25_score: float | None = None
    date: str | None = None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    file_path: str = ""
    source_path: str = ""
    paths: list[str] = field(default_factory=list)
    authors: list[str] = field(default_factory=list)
