import os
import json
import aiosqlite
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional
from pathlib import Path
from loguru import logger

from .protocol import CorpusPaper
from .utils import make_content_key


_INIT_DB_SQL = """
    CREATE TABLE IF NOT EXISTS papers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT NOT NULL,
        source TEXT NOT NULL,
        title TEXT NOT NULL,
        authors TEXT,
        abstract TEXT,
        url TEXT,
        pdf_url TEXT,
        code_url TEXT,
        tldr TEXT,
        score REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(url, date)
    );
    CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(date);
    CREATE INDEX IF NOT EXISTS idx_papers_url ON papers(url);

    CREATE TABLE IF NOT EXISTS corpus_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        abstract TEXT,
        file_path TEXT,
        paths TEXT,
        added_date TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS keyword_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        abstract TEXT,
        content_key TEXT,
        keywords TEXT NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(title, abstract)
    );
    CREATE INDEX IF NOT EXISTS idx_keyword_cache_title ON keyword_cache(title);

    CREATE TABLE IF NOT EXISTS embedding_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        abstract TEXT,
        content_key TEXT,
        model TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(title, abstract, model)
    );
    CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(model);

    CREATE TABLE IF NOT EXISTS task_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_name TEXT NOT NULL,
        trigger TEXT NOT NULL,
        status TEXT NOT NULL,
        metadata TEXT,
        metrics TEXT,
        error TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_task_runs_started_at ON task_runs(started_at DESC);
    CREATE INDEX IF NOT EXISTS idx_task_runs_status ON task_runs(status);

    CREATE TABLE IF NOT EXISTS candidate_cache (
        cache_key TEXT PRIMARY KEY,
        payload TEXT NOT NULL,
        created_at TEXT NOT NULL,
        expires_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_candidate_cache_expires_at ON candidate_cache(expires_at);

    CREATE TABLE IF NOT EXISTS candidate_enrichment_cache (
        url TEXT PRIMARY KEY,
        pdf_url TEXT,
        content_key TEXT,
        tldr TEXT,
        llm_cache_key TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_candidate_enrichment_cache_updated_at ON candidate_enrichment_cache(updated_at);
"""


@asynccontextmanager
async def _connect(db_path: Optional[str] = None, *, row_factory: bool = False):
    db = await aiosqlite.connect(str(_get_db_path(db_path)))
    if row_factory:
        db.row_factory = aiosqlite.Row
    try:
        yield db
    finally:
        await db.close()


def _iter_chunks(values: list[str], size: int = 500):
    for start in range(0, len(values), size):
        yield values[start : start + size]


async def _select_rows_by_values(
    db: aiosqlite.Connection,
    query_prefix: str,
    values: list[str],
) -> list[aiosqlite.Row]:
    rows = []
    for chunk in _iter_chunks(values):
        placeholders = ", ".join("?" for _ in chunk)
        cursor = await db.execute(f"{query_prefix} ({placeholders})", chunk)
        rows.extend(await cursor.fetchall())
    return rows
def _corpus_paper_from_row(row: aiosqlite.Row) -> CorpusPaper:
    return CorpusPaper(
        title=row["title"],
        abstract=row["abstract"],
        added_date=datetime.fromisoformat(row["added_date"]) if row["added_date"] else datetime.now(),
        file_path=row["file_path"],
        source_path=row["source_path"] or "",
        paths=json.loads(row["paths"]) if row["paths"] else [],
    )


def _get_default_db_path() -> Path:
    data_dir = os.environ.get("ARXIV_DAILY_DATA", "")
    if data_dir:
        return Path(data_dir) / "arxiv_daily.db"
    return Path.home() / ".arxiv_daily" / "arxiv_daily.db"


def _get_db_path(config_path: Optional[str] = None) -> Path:
    if config_path:
        return Path(config_path)
    return _get_default_db_path()


async def init_db(db_path: Optional[str] = None):
    path = _get_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with _connect(db_path) as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.executescript(_INIT_DB_SQL)
        await _ensure_cache_columns(db)
        await _ensure_paper_columns(db)
        await _backfill_content_keys(db, "keyword_cache")
        await _backfill_content_keys(db, "embedding_cache")
        await db.commit()
    logger.info(f"Database initialized at {path}")


def _utcnow_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _json_dumps(value: Optional[dict[str, Any]]) -> Optional[str]:
    if not value:
        return None
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _json_loads(value: Optional[str]) -> Optional[dict[str, Any]]:
    if not value:
        return None
    return json.loads(value)


def _decorate_task_run(row: aiosqlite.Row | None) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    result = dict(row)
    result["metadata"] = _json_loads(result.get("metadata"))
    result["metrics"] = _json_loads(result.get("metrics"))

    started_at = result.get("started_at")
    finished_at = result.get("finished_at")
    duration_seconds = None
    if started_at and finished_at:
        try:
            duration_seconds = int(
                (datetime.fromisoformat(finished_at) - datetime.fromisoformat(started_at)).total_seconds()
            )
        except ValueError:
            duration_seconds = None
    result["duration_seconds"] = duration_seconds
    return result


async def _column_exists(db: aiosqlite.Connection, table: str, column: str) -> bool:
    cursor = await db.execute(f"PRAGMA table_info({table})")
    rows = await cursor.fetchall()
    return any(row[1] == column for row in rows)


async def _ensure_cache_columns(db: aiosqlite.Connection):
    if not await _column_exists(db, "keyword_cache", "content_key"):
        await db.execute("ALTER TABLE keyword_cache ADD COLUMN content_key TEXT")
    if not await _column_exists(db, "embedding_cache", "content_key"):
        await db.execute("ALTER TABLE embedding_cache ADD COLUMN content_key TEXT")
    if not await _column_exists(db, "candidate_enrichment_cache", "content_key"):
        await db.execute("ALTER TABLE candidate_enrichment_cache ADD COLUMN content_key TEXT")
    if not await _column_exists(db, "corpus_cache", "source_path"):
        await db.execute("ALTER TABLE corpus_cache ADD COLUMN source_path TEXT")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_keyword_cache_content_key ON keyword_cache(content_key)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_embedding_cache_key_model ON embedding_cache(content_key, model)"
    )


async def _ensure_paper_columns(db: aiosqlite.Connection):
    if not await _column_exists(db, "papers", "tldr"):
        await db.execute("ALTER TABLE papers ADD COLUMN tldr TEXT")
    if not await _column_exists(db, "papers", "score"):
        await db.execute("ALTER TABLE papers ADD COLUMN score REAL")
    if not await _column_exists(db, "candidate_enrichment_cache", "tldr"):
        await db.execute("ALTER TABLE candidate_enrichment_cache ADD COLUMN tldr TEXT")
    if not await _column_exists(db, "candidate_enrichment_cache", "llm_cache_key"):
        await db.execute(
            "ALTER TABLE candidate_enrichment_cache ADD COLUMN llm_cache_key TEXT"
        )


async def _backfill_content_keys(db: aiosqlite.Connection, table: str):
    cursor = await db.execute(
        f"SELECT id, title, abstract FROM {table} WHERE content_key IS NULL OR content_key = ''"
    )
    rows = list(await cursor.fetchall())
    if not rows:
        return

    await db.executemany(
        f"UPDATE {table} SET content_key = ? WHERE id = ?",
        [(make_content_key(row[1], row[2] or ""), row[0]) for row in rows],
    )
    logger.info(f"Backfilled {len(rows)} content keys for {table}")


async def save_papers(papers: list[Any], date: str, db_path: Optional[str] = None) -> None:
    async with _connect(db_path) as db:
        await db.execute("DELETE FROM papers WHERE date = ?", (date,))
        data = []
        for p in papers:
            data.append(
                (
                    date,
                    p.source,
                    p.title,
                    json.dumps(p.authors, ensure_ascii=False),
                    p.abstract,
                    p.url,
                    p.pdf_url,
                    p.code_url,
                    p.tldr,
                    p.score,
                )
            )
        await db.executemany(
            """INSERT OR REPLACE INTO papers
               (date, source, title, authors, abstract, url, pdf_url, code_url, tldr, score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            data,
        )
        await db.commit()
    logger.info(f"Saved {len(papers)} papers for {date}")


async def create_task_run(
    task_name: str,
    trigger: str,
    metadata: Optional[dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> int:
    started_at = _utcnow_iso()
    async with _connect(db_path) as db:
        cursor = await db.execute(
            """INSERT INTO task_runs (task_name, trigger, status, metadata, started_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (task_name, trigger, "running", _json_dumps(metadata), started_at, started_at),
        )
        await db.commit()
        return int(cursor.lastrowid or 0)


async def complete_task_run(
    run_id: int,
    status: str,
    error: Optional[str] = None,
    metrics: Optional[dict[str, Any]] = None,
    db_path: Optional[str] = None,
):
    finished_at = _utcnow_iso()
    async with _connect(db_path) as db:
        await db.execute(
            """UPDATE task_runs
               SET status = ?, error = ?, metrics = ?, finished_at = ?, updated_at = ?
               WHERE id = ?""",
            (status, error, _json_dumps(metrics), finished_at, finished_at, run_id),
        )
        await db.commit()


async def get_latest_task_run(db_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    async with _connect(db_path, row_factory=True) as db:
        cursor = await db.execute(
            "SELECT * FROM task_runs ORDER BY started_at DESC, id DESC LIMIT 1"
        )
        row = await cursor.fetchone()
        return _decorate_task_run(row)


async def load_candidate_cache(
    cache_key: str, db_path: Optional[str] = None
) -> Optional[list[dict[str, Any]]]:
    now_iso = _utcnow_iso()
    async with _connect(db_path) as db:
        cursor = await db.execute(
            "SELECT payload, expires_at FROM candidate_cache WHERE cache_key = ?",
            (cache_key,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        payload, expires_at = row
        if expires_at <= now_iso:
            await db.execute("DELETE FROM candidate_cache WHERE cache_key = ?", (cache_key,))
            await db.commit()
            return None
        return json.loads(payload)


async def save_candidate_cache(
    cache_key: str,
    payload: list[dict[str, Any]],
    ttl_minutes: int,
    db_path: Optional[str] = None,
):
    created_at = _utcnow_iso()
    from datetime import timedelta

    expires_at = (
        datetime.utcnow().replace(microsecond=0)
        + timedelta(minutes=max(ttl_minutes, 1))
    ).isoformat()
    async with _connect(db_path) as db:
        await db.execute(
            """INSERT OR REPLACE INTO candidate_cache (cache_key, payload, created_at, expires_at)
               VALUES (?, ?, ?, ?)""",
            (cache_key, json.dumps(payload, ensure_ascii=False), created_at, expires_at),
        )
        await db.commit()


async def load_candidate_enrichments(
    urls: list[str], db_path: Optional[str] = None
) -> dict[str, dict[str, Any]]:
    if not urls:
        return {}

    unique_urls = [url for url in dict.fromkeys(urls) if url]
    if not unique_urls:
        return {}

    async with _connect(db_path, row_factory=True) as db:
        result: dict[str, dict[str, Any]] = {}
        rows = await _select_rows_by_values(
            db,
            "SELECT * FROM candidate_enrichment_cache WHERE url IN",
            unique_urls,
        )
        for row in rows:
            record = dict(row)
            result[record["url"]] = record
        return result


async def save_candidate_enrichments(
    entries: list[dict[str, Any]], db_path: Optional[str] = None
):
    if not entries:
        return

    now_iso = _utcnow_iso()
    async with _connect(db_path, row_factory=True) as db:
        data = []
        for entry in entries:
            url = entry.get("url")
            if not url:
                continue
            data.append(entry)

        if not data:
            return

        existing: dict[str, dict[str, Any]] = {}
        urls = [entry["url"] for entry in data if entry.get("url")]
        rows = await _select_rows_by_values(
            db,
            "SELECT * FROM candidate_enrichment_cache WHERE url IN",
            urls,
        )
        for row in rows:
            existing[row["url"]] = dict(row)

        payload = []
        for entry in data:
            url = entry["url"]
            previous = existing.get(url, {})
            created_at = previous.get("created_at") or now_iso
            payload.append(
                (
                    url,
                    entry.get("pdf_url", previous.get("pdf_url")),
                    entry.get("content_key", previous.get("content_key")),
                    entry.get("tldr", previous.get("tldr")),
                    entry.get("llm_cache_key", previous.get("llm_cache_key")),
                    created_at,
                    now_iso,
                )
            )

        await db.executemany(
            """INSERT OR REPLACE INTO candidate_enrichment_cache
               (url, pdf_url, content_key, tldr, llm_cache_key, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            payload,
        )
        await db.commit()
    logger.info(f"Cached {len(payload)} candidate enrichment entries")


async def get_papers_by_date(date: str, db_path: Optional[str] = None) -> list[dict[str, Any]]:
    async with _connect(db_path, row_factory=True) as db:
        cursor = await db.execute(
            "SELECT * FROM papers WHERE date = ? ORDER BY score DESC",
            (date,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_all_dates(db_path: Optional[str] = None) -> list[str]:
    async with _connect(db_path) as db:
        cursor = await db.execute("SELECT DISTINCT date FROM papers ORDER BY date DESC")
        rows = await cursor.fetchall()
        return [r[0] for r in rows]


async def get_paper_count(db_path: Optional[str] = None) -> int:
    async with _connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM papers")
        row = await cursor.fetchone()
        return int(row[0]) if row else 0


async def get_seen_paper_urls(
    db_path: Optional[str] = None,
    *,
    exclude_date: Optional[str] = None,
) -> set[str]:
    async with _connect(db_path) as db:
        if exclude_date:
            cursor = await db.execute(
                "SELECT DISTINCT url FROM papers WHERE url IS NOT NULL AND url != '' AND date != ?",
                (exclude_date,),
            )
        else:
            cursor = await db.execute(
                "SELECT DISTINCT url FROM papers WHERE url IS NOT NULL AND url != ''"
            )
        rows = await cursor.fetchall()
        return {row[0] for row in rows if row and row[0]}


async def get_seen_paper_content_keys(
    db_path: Optional[str] = None,
    *,
    exclude_date: Optional[str] = None,
) -> set[str]:
    async with _connect(db_path) as db:
        if exclude_date:
            cursor = await db.execute(
                "SELECT title, abstract FROM papers WHERE date != ?",
                (exclude_date,),
            )
        else:
            cursor = await db.execute("SELECT title, abstract FROM papers")
        rows = await cursor.fetchall()
        return {
            make_content_key(row[0] or "", row[1] or "")
            for row in rows
            if row and (row[0] or row[1])
        }


async def get_corpus_count(db_path: Optional[str] = None) -> int:
    """获取corpus缓存数量，避免加载全部数据"""
    async with _connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM corpus_cache")
        row = await cursor.fetchone()
        return row[0] if row else 0


async def save_corpus_cache(corpus: list[CorpusPaper], db_path: Optional[str] = None) -> None:
    async with _connect(db_path) as db:
        await db.execute("DELETE FROM corpus_cache")
        data = [
            (
                c.title,
                c.abstract,
                c.file_path,
                c.source_path,
                json.dumps(c.paths, ensure_ascii=False),
                c.added_date.isoformat()
                if hasattr(c.added_date, "isoformat")
                else str(c.added_date),
            )
            for c in corpus
        ]
        await db.executemany(
            """INSERT INTO corpus_cache (title, abstract, file_path, source_path, paths, added_date)
               VALUES (?, ?, ?, ?, ?, ?)""",
            data,
        )
        await db.commit()
    logger.info(f"Cached {len(corpus)} corpus papers")


async def save_keyword_cache(
    title: str, abstract: str, keywords: list[str], db_path: Optional[str] = None
):
    async with _connect(db_path) as db:
        await db.execute(
            """INSERT OR REPLACE INTO keyword_cache (title, abstract, content_key, keywords)
               VALUES (?, ?, ?, ?)""",
            (
                title,
                abstract,
                make_content_key(title, abstract),
                json.dumps(keywords, ensure_ascii=False),
            ),
        )
        await db.commit()
    logger.debug(f"Cached keywords for: {title[:50]}...")


async def load_keywords_for_papers(
    papers: list[CorpusPaper], db_path: Optional[str] = None
) -> dict[str, list[str]]:
    if not papers:
        return {}

    async with _connect(db_path, row_factory=True) as db:
        result: dict[str, list[str]] = {}
        key_map = {
            make_content_key(paper.title, paper.abstract or ""): paper
            for paper in papers
        }
        content_keys = list(key_map.keys())

        rows = await _select_rows_by_values(
            db,
            "SELECT content_key, title, abstract, keywords FROM keyword_cache WHERE content_key IN",
            content_keys,
        )
        for row in rows:
            content_key = row["content_key"] or make_content_key(
                row["title"], row["abstract"] or ""
            )
            paper = key_map.get(content_key)
            if not paper:
                continue
            result[content_key] = json.loads(row["keywords"])
        return result


async def get_all_cached_keywords(
    db_path: Optional[str] = None,
) -> list[tuple[str, list[str]]]:
    async with _connect(db_path, row_factory=True) as db:
        cursor = await db.execute("SELECT title, keywords FROM keyword_cache")
        rows = await cursor.fetchall()
        return [(r["title"], json.loads(r["keywords"])) for r in rows]


async def load_corpus_cache(db_path: Optional[str] = None) -> list[CorpusPaper]:
    async with _connect(db_path, row_factory=True) as db:
        cursor = await db.execute("SELECT * FROM corpus_cache")
        rows = await cursor.fetchall()
        return [_corpus_paper_from_row(row) for row in rows]


async def save_embeddings(
    papers: list[CorpusPaper],
    embeddings: list[Any],
    model: str,
    db_path: Optional[str] = None,
) -> None:
    import pickle

    async with _connect(db_path) as db:
        data = [
            (
                p.title,
                p.abstract,
                make_content_key(p.title, p.abstract or ""),
                model,
                pickle.dumps(e),
            )
            for p, e in zip(papers, embeddings)
        ]
        await db.executemany(
            """INSERT OR REPLACE INTO embedding_cache (title, abstract, content_key, model, embedding)
               VALUES (?, ?, ?, ?, ?)""",
            data,
        )
        await db.commit()
    logger.info(f"Cached {len(papers)} embeddings for model {model}")


async def load_embeddings(
    papers: list[CorpusPaper], model: str, db_path: Optional[str] = None
) -> tuple[list[int], list[Any]]:
    import pickle

    if not papers:
        return [], []

    async with _connect(db_path, row_factory=True) as db:
        key_to_indices: dict[str, list[int]] = {}
        for i, paper in enumerate(papers):
            content_key = make_content_key(paper.title, paper.abstract or "")
            key_to_indices.setdefault(content_key, []).append(i)

        indices = []
        embeddings: list[Any] = []
        content_keys = list(key_to_indices.keys())
        for chunk in _iter_chunks(content_keys):
            placeholders = ", ".join("?" for _ in chunk)
            cursor = await db.execute(
                f"SELECT content_key, embedding FROM embedding_cache "
                f"WHERE model = ? AND content_key IN ({placeholders})",
                [model, *chunk],
            )
            rows = await cursor.fetchall()
            for row in rows:
                for idx in key_to_indices.get(row["content_key"], []):
                    indices.append(idx)
                    embeddings.append(pickle.loads(row["embedding"]))
        return indices, embeddings
