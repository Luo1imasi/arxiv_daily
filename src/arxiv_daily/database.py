import os
import json
import aiosqlite
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
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
        method TEXT,
        evidence TEXT,
        why_for_me TEXT,
        judge_relevance REAL,
        judge_reason TEXT,
        judge_keep INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(url, date)
    );
    CREATE INDEX IF NOT EXISTS idx_papers_date ON papers(date);
    CREATE INDEX IF NOT EXISTS idx_papers_url ON papers(url);

    CREATE TABLE IF NOT EXISTS zotero_exports (
        library_id TEXT NOT NULL,
        arxiv_id TEXT NOT NULL,
        payload TEXT NOT NULL,
        PRIMARY KEY (library_id, arxiv_id)
    );

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
        model TEXT,
        prompt_version TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(title, abstract)
    );
    CREATE INDEX IF NOT EXISTS idx_keyword_cache_title ON keyword_cache(title);

    CREATE TABLE IF NOT EXISTS embedding_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL,
        abstract TEXT,
        content_key TEXT NOT NULL,
        model TEXT NOT NULL,
        embedding BLOB NOT NULL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(content_key, model)
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
        method TEXT,
        evidence TEXT,
        why_for_me TEXT,
        judge_relevance REAL,
        judge_reason TEXT,
        judge_keep INTEGER,
        llm_cache_key TEXT,
        judge_cache_key TEXT,
        coarse_score REAL,
        coarse_cache_key TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_candidate_enrichment_cache_updated_at ON candidate_enrichment_cache(updated_at);

    CREATE TABLE IF NOT EXISTS interest_profile (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        corpus_signature TEXT NOT NULL,
        feedback_signature TEXT NOT NULL,
        model TEXT,
        prompt_version TEXT,
        profile_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS paper_feedback (
        url TEXT PRIMARY KEY,
        date TEXT,
        title TEXT,
        tldr TEXT,
        vote TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );
"""


@asynccontextmanager
async def _connect(db_path: Optional[str] = None, *, row_factory: bool = False):
    db = await aiosqlite.connect(str(_get_db_path(db_path)))
    await db.execute("PRAGMA busy_timeout = 5000")
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
        embedding_migrated = await _migrate_embedding_cache_unique(db)
        deleted_candidates = await _delete_expired_candidate_cache(db)
        await db.commit()
        cursor = await db.execute(
            "SELECT value FROM schema_meta WHERE key = 'vacuum_revision'"
        )
        row = await cursor.fetchone()
        needs_vacuum = (embedding_migrated or deleted_candidates) and (
            not row or row[0] != "20261006-1"
        )
    if needs_vacuum:
        _vacuum_database(path, "20261006-1")
    logger.info(f"Database initialized at {path}")


def _utcnow_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


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
    for table, column, definition in (
        ("keyword_cache", "content_key", "TEXT"),
        ("keyword_cache", "model", "TEXT"),
        ("keyword_cache", "prompt_version", "TEXT"),
        ("embedding_cache", "content_key", "TEXT"),
        ("candidate_enrichment_cache", "content_key", "TEXT"),
        ("candidate_enrichment_cache", "method", "TEXT"),
        ("candidate_enrichment_cache", "evidence", "TEXT"),
        ("candidate_enrichment_cache", "why_for_me", "TEXT"),
        ("candidate_enrichment_cache", "judge_relevance", "REAL"),
        ("candidate_enrichment_cache", "judge_reason", "TEXT"),
        ("candidate_enrichment_cache", "judge_keep", "INTEGER"),
        ("candidate_enrichment_cache", "judge_cache_key", "TEXT"),
        ("candidate_enrichment_cache", "coarse_score", "REAL"),
        ("candidate_enrichment_cache", "coarse_cache_key", "TEXT"),
        ("corpus_cache", "source_path", "TEXT"),
    ):
        if not await _column_exists(db, table, column):
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_keyword_cache_content_key ON keyword_cache(content_key)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_embedding_cache_key_model ON embedding_cache(content_key, model)"
    )


async def _ensure_paper_columns(db: aiosqlite.Connection):
    for column, definition in (
        ("tldr", "TEXT"),
        ("score", "REAL"),
        ("method", "TEXT"),
        ("evidence", "TEXT"),
        ("why_for_me", "TEXT"),
        ("judge_relevance", "REAL"),
        ("judge_reason", "TEXT"),
        ("judge_keep", "INTEGER"),
    ):
        if not await _column_exists(db, "papers", column):
            await db.execute(f"ALTER TABLE papers ADD COLUMN {column} {definition}")
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


async def _migrate_embedding_cache_unique(db: aiosqlite.Connection) -> bool:
    cursor = await db.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'embedding_cache'"
    )
    row = await cursor.fetchone()
    table_sql = (row[0] or "") if row else ""
    normalized = "".join(table_sql.split()).lower()
    if "unique(content_key,model)" in normalized and "unique(title,abstract,model)" not in normalized:
        return False

    await db.execute(
        """
        DELETE FROM embedding_cache
        WHERE id NOT IN (
            SELECT MAX(id) FROM embedding_cache
            WHERE content_key IS NOT NULL AND content_key != ''
            GROUP BY content_key, model
        )
        """
    )
    await db.execute(
        """
        CREATE TABLE embedding_cache_v2 (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            abstract TEXT,
            content_key TEXT NOT NULL,
            model TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(content_key, model)
        )
        """
    )
    await db.execute(
        """
        INSERT INTO embedding_cache_v2
            (id, title, abstract, content_key, model, embedding, created_at)
        SELECT id, title, abstract, content_key, model, embedding, created_at
        FROM embedding_cache
        WHERE content_key IS NOT NULL AND content_key != ''
        """
    )
    await db.execute("DROP TABLE embedding_cache")
    await db.execute("ALTER TABLE embedding_cache_v2 RENAME TO embedding_cache")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_embedding_cache_model ON embedding_cache(model)"
    )
    logger.info("Rebuilt embedding_cache with UNIQUE(content_key, model)")
    return True


async def _delete_expired_candidate_cache(db: aiosqlite.Connection) -> int:
    cursor = await db.execute(
        "DELETE FROM candidate_cache WHERE expires_at <= ?",
        (_utcnow_iso(),),
    )
    deleted = int(cursor.rowcount or 0)
    if deleted:
        logger.info(f"Deleted {deleted} expired candidate cache rows")
    return deleted


def _vacuum_database(path: Path, revision: str) -> None:
    import sqlite3

    logger.info(f"Vacuuming database ({revision})")
    connection = sqlite3.connect(str(path))
    try:
        connection.isolation_level = None
        connection.execute("VACUUM")
        connection.isolation_level = ""
        connection.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('vacuum_revision', ?)",
            (revision,),
        )
        connection.commit()
    finally:
        connection.close()
    logger.info("Database vacuum finished")


async def purge_expired_candidate_cache(db_path: Optional[str] = None) -> int:
    async with _connect(db_path) as db:
        deleted = await _delete_expired_candidate_cache(db)
        await db.commit()
        return deleted


async def fail_orphaned_running_tasks(
    reason: str, db_path: Optional[str] = None
) -> int:
    finished_at = _utcnow_iso()
    async with _connect(db_path) as db:
        cursor = await db.execute(
            """UPDATE task_runs
               SET status = 'failed', error = ?, finished_at = ?, updated_at = ?
               WHERE status = 'running'""",
            (reason, finished_at, finished_at),
        )
        await db.commit()
        return int(cursor.rowcount or 0)


async def count_task_runs_started_on(
    business_date: str,
    *,
    task_name: str,
    timezone_name: str,
    db_path: Optional[str] = None,
) -> dict[str, int]:
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(timezone_name)
    async with _connect(db_path, row_factory=True) as db:
        cursor = await db.execute(
            "SELECT status, started_at FROM task_runs WHERE task_name = ?",
            (task_name,),
        )
        rows = await cursor.fetchall()
    counts = {"total": 0, "succeeded": 0, "failed": 0}
    for row in rows:
        started_at = row["started_at"]
        if not started_at:
            continue
        try:
            started = datetime.fromisoformat(started_at)
        except ValueError:
            continue
        if started.tzinfo is None:
            started = started.replace(tzinfo=UTC)
        if started.astimezone(tz).date().isoformat() != business_date:
            continue
        counts["total"] += 1
        if row["status"] == "succeeded":
            counts["succeeded"] += 1
        elif row["status"] == "failed":
            counts["failed"] += 1
    return counts


def _field(entry: dict[str, Any], previous: dict[str, Any], key: str) -> Any:
    if key in entry:
        return entry.get(key)
    return previous.get(key)


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
                    getattr(p, "method", None),
                    getattr(p, "evidence", None),
                    getattr(p, "why_for_me", None),
                    getattr(p, "judge_relevance", None),
                    getattr(p, "judge_reason", None),
                    None
                    if getattr(p, "judge_keep", None) is None
                    else int(bool(getattr(p, "judge_keep"))),
                )
            )
        await db.executemany(
            """INSERT OR REPLACE INTO papers
               (date, source, title, authors, abstract, url, pdf_url, code_url, tldr, score,
                method, evidence, why_for_me, judge_relevance, judge_reason, judge_keep)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
    expires_at = (
        datetime.now(UTC).replace(microsecond=0)
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
                    _field(entry, previous, "pdf_url"),
                    _field(entry, previous, "content_key"),
                    _field(entry, previous, "tldr"),
                    _field(entry, previous, "method"),
                    _field(entry, previous, "evidence"),
                    _field(entry, previous, "why_for_me"),
                    _field(entry, previous, "judge_relevance"),
                    _field(entry, previous, "judge_reason"),
                    _field(entry, previous, "judge_keep"),
                    _field(entry, previous, "llm_cache_key"),
                    _field(entry, previous, "judge_cache_key"),
                    _field(entry, previous, "coarse_score"),
                    _field(entry, previous, "coarse_cache_key"),
                    created_at,
                    now_iso,
                )
            )

        await db.executemany(
            """INSERT OR REPLACE INTO candidate_enrichment_cache
               (url, pdf_url, content_key, tldr, method, evidence, why_for_me,
                judge_relevance, judge_reason, judge_keep, llm_cache_key, judge_cache_key,
                coarse_score, coarse_cache_key, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            payload,
        )
        await db.commit()
    logger.info(f"Cached {len(payload)} candidate enrichment entries")


async def update_paper_summaries(
    date: str, papers: list[Any], db_path: Optional[str] = None
) -> int:
    """Write TLDR fields for papers already stored on a date. Other columns stay."""
    if not papers:
        return 0
    updated = 0
    async with _connect(db_path) as db:
        for paper in papers:
            url = getattr(paper, "url", None)
            tldr = getattr(paper, "tldr", None)
            if not url or not str(tldr or "").strip():
                continue
            cursor = await db.execute(
                """UPDATE papers
                   SET tldr = ?, method = ?, evidence = ?, why_for_me = ?
                   WHERE date = ? AND url = ?""",
                (
                    tldr,
                    getattr(paper, "method", None),
                    getattr(paper, "evidence", None),
                    getattr(paper, "why_for_me", None),
                    date,
                    url,
                ),
            )
            updated += int(cursor.rowcount or 0)
        await db.commit()
    logger.info(f"Updated TLDR for {updated} papers on {date}")
    return updated


async def list_dates_with_empty_tldr(
    since: str | None = None, db_path: Optional[str] = None
) -> list[str]:
    async with _connect(db_path) as db:
        if since:
            cursor = await db.execute(
                """SELECT DISTINCT date FROM papers
                   WHERE date >= ? AND (tldr IS NULL OR TRIM(tldr) = '')
                   ORDER BY date ASC""",
                (since,),
            )
        else:
            cursor = await db.execute(
                """SELECT DISTINCT date FROM papers
                   WHERE tldr IS NULL OR TRIM(tldr) = ''
                   ORDER BY date ASC"""
            )
        rows = await cursor.fetchall()
        return [row[0] for row in rows]


async def get_papers_by_date(date: str, db_path: Optional[str] = None) -> list[dict[str, Any]]:
    async with _connect(db_path, row_factory=True) as db:
        cursor = await db.execute(
            """SELECT papers.*, paper_feedback.vote AS feedback_vote
               FROM papers
               LEFT JOIN paper_feedback ON paper_feedback.url = papers.url
               WHERE papers.date = ?
               ORDER BY CASE WHEN papers.judge_relevance IS NULL THEN 0 ELSE 1 END DESC,
                        papers.judge_relevance DESC,
                        papers.score DESC,
                        papers.id ASC""",
            (date,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def get_papers_between(
    start_date: str, end_date: str, db_path: Optional[str] = None
) -> list[dict[str, Any]]:
    async with _connect(db_path, row_factory=True) as db:
        cursor = await db.execute(
            """SELECT * FROM papers
               WHERE date >= ? AND date <= ?
               ORDER BY date DESC, score DESC""",
            (start_date, end_date),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


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
    before_date: Optional[str] = None,
) -> set[str]:
    async with _connect(db_path) as db:
        if before_date:
            cursor = await db.execute(
                "SELECT DISTINCT url FROM papers WHERE url IS NOT NULL AND url != '' AND date < ?",
                (before_date,),
            )
        elif exclude_date:
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
    before_date: Optional[str] = None,
) -> set[str]:
    async with _connect(db_path) as db:
        if before_date:
            cursor = await db.execute(
                "SELECT title, abstract FROM papers WHERE date < ?",
                (before_date,),
            )
        elif exclude_date:
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
    """Count the corpus including persistent Zotero exports, without duplicates."""
    return len(await load_corpus_cache(db_path))


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
    title: str,
    abstract: str,
    keywords: list[str],
    db_path: Optional[str] = None,
    *,
    model: str = "",
    prompt_version: str = "",
):
    async with _connect(db_path) as db:
        await db.execute(
            """INSERT OR REPLACE INTO keyword_cache
               (title, abstract, content_key, keywords, model, prompt_version)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                title,
                abstract,
                make_content_key(title, abstract),
                json.dumps(keywords, ensure_ascii=False),
                model,
                prompt_version,
            ),
        )
        await db.commit()
    logger.debug(f"Cached keywords for: {title[:50]}...")


async def load_keywords_for_papers(
    papers: list[CorpusPaper],
    db_path: Optional[str] = None,
    *,
    model: str | None = None,
    prompt_version: str | None = None,
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
            """SELECT content_key, title, abstract, keywords, model, prompt_version
               FROM keyword_cache WHERE content_key IN""",
            content_keys,
        )
        for row in rows:
            if model is not None and (row["model"] or "") != model:
                continue
            if prompt_version is not None and (row["prompt_version"] or "") != prompt_version:
                continue
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
        papers = [_corpus_paper_from_row(row) for row in rows]
    return await merge_exported_corpus(papers, db_path)


async def get_paper_by_id(paper_id: int, db_path: str | None = None) -> dict[str, Any] | None:
    async with _connect(db_path, row_factory=True) as conn:
        cursor = await conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,))
        row = await cursor.fetchone()
        return dict(row) if row else None


async def load_zotero_exports(db_path: str | None = None) -> list[dict[str, Any]]:
    async with _connect(db_path) as conn:
        cursor = await conn.execute("SELECT payload FROM zotero_exports")
        return [json.loads(row[0]) for row in await cursor.fetchall()]


async def save_zotero_export(record: dict[str, Any], db_path: str | None = None) -> None:
    async with _connect(db_path) as conn:
        await conn.execute(
            "INSERT INTO zotero_exports (library_id, arxiv_id, payload) VALUES (?, ?, ?) "
            "ON CONFLICT(library_id, arxiv_id) DO UPDATE SET payload = excluded.payload",
            (record["library_id"], record["arxiv_id"], json.dumps(record, ensure_ascii=False)),
        )
        await conn.commit()


async def merge_exported_corpus(
    papers: list[CorpusPaper], db_path: str | None = None
) -> list[CorpusPaper]:
    for record in await load_zotero_exports(db_path):
        if record.get("status") != "saved":
            continue
        paper = CorpusPaper(
            title=record["title"],
            abstract=record["abstract"],
            authors=record["authors"],
            added_date=datetime.fromisoformat(record["added_date"]),
            file_path=record["file_path"],
            source_path=record["source_path"],
        )
        key = make_content_key(paper.title, paper.abstract)
        papers = [
            p for p in papers
            if p.source_path.lstrip("/") != paper.source_path.lstrip("/")
            and make_content_key(p.title, p.abstract) != key
        ]
        papers.append(paper)
    return papers


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
            seen_keys: set[str] = set()
            for row in rows:
                content_key = row["content_key"]
                if content_key in seen_keys:
                    continue
                seen_keys.add(content_key)
                for idx in key_to_indices.get(content_key, []):
                    indices.append(idx)
                    embeddings.append(pickle.loads(row["embedding"]))
        return indices, embeddings


async def load_interest_profile(db_path: Optional[str] = None) -> Optional[dict[str, Any]]:
    async with _connect(db_path, row_factory=True) as db:
        cursor = await db.execute("SELECT * FROM interest_profile WHERE id = 1")
        row = await cursor.fetchone()
        if not row:
            return None
        record = dict(row)
        record["profile"] = _json_loads(record.get("profile_json")) or {}
        return record


async def save_interest_profile(
    profile: dict[str, Any],
    *,
    corpus_signature: str,
    feedback_signature: str,
    model: str,
    prompt_version: str,
    db_path: Optional[str] = None,
) -> None:
    async with _connect(db_path) as db:
        await db.execute(
            """INSERT OR REPLACE INTO interest_profile
               (id, corpus_signature, feedback_signature, model, prompt_version, profile_json, updated_at)
               VALUES (1, ?, ?, ?, ?, ?, ?)""",
            (
                corpus_signature,
                feedback_signature,
                model,
                prompt_version,
                _json_dumps(profile),
                _utcnow_iso(),
            ),
        )
        await db.commit()


async def list_feedback(db_path: Optional[str] = None) -> list[dict[str, Any]]:
    async with _connect(db_path, row_factory=True) as db:
        cursor = await db.execute(
            "SELECT * FROM paper_feedback ORDER BY updated_at DESC, url ASC"
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def upsert_feedback(
    *,
    url: str,
    vote: str,
    date: str = "",
    title: str = "",
    tldr: str = "",
    db_path: Optional[str] = None,
) -> None:
    if vote not in {"relevant", "irrelevant"}:
        raise ValueError("vote must be relevant or irrelevant")
    now_iso = _utcnow_iso()
    async with _connect(db_path) as db:
        cursor = await db.execute(
            "SELECT created_at FROM paper_feedback WHERE url = ?",
            (url,),
        )
        previous = await cursor.fetchone()
        created_at = previous[0] if previous and previous[0] else now_iso
        await db.execute(
            """INSERT OR REPLACE INTO paper_feedback
               (url, date, title, tldr, vote, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (url, date, title, tldr, vote, created_at, now_iso),
        )
        await db.commit()
