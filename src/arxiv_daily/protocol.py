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
    score: float | None = None
    retrieval_score: float | None = None
    date: str | None = None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    file_path: str = ""
    source_path: str = ""
    paths: list[str] = field(default_factory=list)
