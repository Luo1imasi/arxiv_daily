import os
import sys
import json
import hmac
import copy
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler

import jinja2

from . import database as db
from .config import (
    build_override_config,
    deep_merge,
    get_config_value,
    get_default_config,
    load_config,
    save_config,
)
from .executor import Executor
from .task_runner import TaskRunner
from .zotero import ExportError, arxiv_id, export_paper, settings as zotero_settings
from .business_date import (
    business_date_range_between,
    get_business_date,
    get_business_timezone_info,
)

BASE_DIR = Path(__file__).parent
SENSITIVE_KEYS = {"password", "api_key", "key", "admin_password"}
ADMIN_PASSWORD_ENV = "ARXIV_DAILY_ADMIN_PASSWORD"
_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(BASE_DIR / "templates")),
    autoescape=jinja2.select_autoescape(["html"]),
)


def _from_json_filter(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


_jinja_env.filters["from_json"] = _from_json_filter

_scheduler: AsyncIOScheduler | None = None
_app_config: dict[str, Any] = {}
_task_runner = TaskRunner()


def _configure_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )


def _mask_password(d: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _mask_password(v)
        elif k in SENSITIVE_KEYS and isinstance(v, str) and v:
            result[k] = "****"
        else:
            result[k] = v
    return result


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    """Settings a reader can see. Secrets and filesystem paths stay out."""
    arxiv_config = (config.get("source") or {}).get("arxiv") or {}
    llm_config = config.get("llm") or {}
    executor_config = config.get("executor") or {}
    reranker_config = config.get("reranker") or {}
    models = llm_config.get("models") if isinstance(llm_config.get("models"), dict) else {}
    return {
        "source": {
            "arxiv": {
                "category": arxiv_config.get("category") or [],
                "include_cross_list": bool(arxiv_config.get("include_cross_list")),
                "use_keyword_search": bool(arxiv_config.get("use_keyword_search")),
                "recent_days": arxiv_config.get("recent_days"),
                "max_keywords": arxiv_config.get("max_keywords"),
            }
        },
        "llm": {
            "language": llm_config.get("language"),
            "model": llm_config.get("model"),
            "models": {
                "extract": models.get("extract"),
                "coarse": models.get("coarse"),
                "summarize": models.get("summarize"),
                "judge": models.get("judge"),
                "profile": models.get("profile"),
                "qa": models.get("qa"),
            },
            "configured": bool(llm_config.get("api_key")),
        },
        "executor": {
            "max_paper_num": executor_config.get("max_paper_num"),
            "timezone": executor_config.get("timezone"),
            "schedule_hour": executor_config.get("schedule_hour"),
            "schedule_minute": executor_config.get("schedule_minute"),
            "source": executor_config.get("source"),
        },
        "reranker": {"model": reranker_config.get("model")},
    }


def admin_config_view(config: dict[str, Any]) -> dict[str, Any]:
    """Full settings for an authenticated operator, with secrets removed."""
    redacted = copy.deepcopy(config)
    for section_name in ("webdav", "llm", "server", "zotero"):
        section = redacted.get(section_name)
        if not isinstance(section, dict):
            continue
        for key in list(section):
            if key in SENSITIVE_KEYS:
                section[f"{key}_set"] = bool(section.get(key))
                section.pop(key, None)
    return redacted


def _get_schedule_settings(config: dict[str, Any]) -> tuple[str, ZoneInfo, int, int]:
    tz_name, tz = get_business_timezone_info(config)
    return (
        tz_name,
        tz,
        int(get_config_value(config, "executor.schedule_hour")),
        int(get_config_value(config, "executor.schedule_minute")),
    )


def _format_run_timestamp(value: str | None, tz: ZoneInfo) -> str | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S")


def _get_admin_password() -> str:
    env_password = os.environ.get(ADMIN_PASSWORD_ENV)
    if env_password is not None:
        return env_password
    return str(get_config_value(_app_config, "server.admin_password", "admin"))


async def _require_admin_password(request: Request) -> None:
    expected = _get_admin_password()
    provided = request.headers.get("X-Admin-Password", "")
    if not expected or not hmac.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Invalid operation password")


def _decorate_timestamp_fields(record: dict[str, Any], tz: ZoneInfo) -> dict[str, Any]:
    for field in ("started_at", "finished_at", "created_at", "updated_at"):
        if field in record:
            raw_value = record.get(field)
            record[f"{field}_raw"] = raw_value
            record[f"{field}_display"] = _format_run_timestamp(raw_value, tz)
            record[field] = record[f"{field}_display"]
    return record


def _decorate_latest_run_for_display(
    latest_run: dict[str, Any] | None, config: dict[str, Any]
) -> dict[str, Any] | None:
    if not latest_run:
        return latest_run
    tz_name, tz = get_business_timezone_info(config)
    decorated = _decorate_timestamp_fields(dict(latest_run), tz)
    decorated["display_timezone"] = tz_name
    return decorated


async def _build_page_context(current_date: str | None) -> dict[str, Any]:
    dates = await db.get_all_dates()
    papers = await db.get_papers_by_date(current_date) if current_date else []
    library_id = zotero_settings(_app_config)["user_id"]
    exports = {
        r["arxiv_id"]: r for r in await db.load_zotero_exports()
        if r["library_id"] == library_id
    }
    for paper in papers:
        try:
            paper["zotero_status"] = exports.get(arxiv_id(paper["url"]), {}).get("status", "")
        except ExportError:
            paper["zotero_status"] = ""
    corpus_count = await db.get_corpus_count()
    status = await _task_runner.get_status()
    latest_run = _decorate_latest_run_for_display(status.get("latest_run"), _app_config)
    return {
        "papers": papers,
        "dates": dates,
        "current_date": current_date,
        "config": public_config(_app_config),
        "has_corpus": corpus_count > 0,
        "task_running": status["running"],
        "latest_run": latest_run,
    }


async def _start_executor_run(skip_tldr: bool = False, trigger: str = "manual") -> int:
    executor = Executor(_app_config)
    return await _task_runner.start(
        "daily recommendation",
        lambda: executor.run(skip_tldr=skip_tldr),
        trigger=trigger,
        metadata={"skip_tldr": skip_tldr},
        result_to_metrics=lambda _result: dict(executor.last_run_metrics),
    )


async def _start_backfill_run(
    start_date: str, end_date: str, skip_tldr: bool = False, trigger: str = "manual"
) -> int:
    executor = Executor(_app_config)
    return await _task_runner.start(
        "recommendation backfill",
        lambda: executor.run_between_dates(start_date, end_date, skip_tldr=skip_tldr),
        trigger=trigger,
        metadata={"skip_tldr": skip_tldr, "start_date": start_date, "end_date": end_date},
        result_to_metrics=lambda _result: dict(executor.last_run_metrics),
    )


def _parse_backfill_date(value: object, field_name: str) -> date:
    raw_value = str(value or "").strip()
    if not raw_value:
        raise HTTPException(status_code=400, detail=f"{field_name} is required")
    try:
        return date.fromisoformat(raw_value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be YYYY-MM-DD") from exc


def _current_business_date(config: dict[str, Any]) -> date:
    return get_business_date(config)


def _reschedule_daily_run(config: dict[str, Any]) -> None:
    if not _scheduler:
        return

    tz_name, tz, hour, minute = _get_schedule_settings(config)
    _scheduler.reschedule_job(
        "daily_run", trigger="cron", hour=hour, minute=minute, timezone=tz
    )
    logger.info(f"Rescheduled to {hour:02d}:{minute:02d} ({tz_name})")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _scheduler, _app_config
    _configure_logging()

    await db.init_db()
    interrupted = await db.fail_orphaned_running_tasks(
        "Process restarted before the task finished"
    )
    if interrupted:
        logger.warning(f"Marked {interrupted} interrupted task(s) as failed")

    config = load_config()
    _app_config = config

    _scheduler = AsyncIOScheduler()
    tz_name, tz, hour, minute = _get_schedule_settings(config)
    _scheduler.add_job(
        scheduled_run, "cron", hour=hour, minute=minute, id="daily_run", timezone=tz
    )
    _scheduler.add_job(
        scheduled_catchup,
        "cron",
        hour="11,15,20",
        minute=15,
        id="daily_catchup",
        timezone=tz,
    )
    _scheduler.add_job(
        scheduled_cache_cleanup,
        "cron",
        hour=3,
        minute=40,
        id="cache_cleanup",
        timezone=tz,
    )
    _scheduler.start()
    logger.info(f"Scheduler started: daily run at {hour:02d}:{minute:02d} ({tz_name})")
    logger.info(f"Same-day catch-up scheduled at 11:15, 15:15 and 20:15 ({tz_name})")
    logger.info(f"Expired candidate cache cleanup scheduled at 03:40 ({tz_name})")

    yield

    if _scheduler:
        _scheduler.shutdown()

async def scheduled_run():
    if _task_runner.is_running():
        logger.info("A task is already running, skipping scheduled run")
        return
    logger.info("Starting scheduled daily run...")
    await _start_executor_run(skip_tldr=False, trigger="scheduler")
    try:
        await _task_runner.wait()
    except Exception as e:
        logger.error(f"Scheduled run failed: {e}")


async def scheduled_cache_cleanup():
    deleted = await db.purge_expired_candidate_cache()
    if deleted:
        logger.info(f"Periodic cleanup removed {deleted} expired candidate cache rows")


async def scheduled_catchup():
    if _task_runner.is_running():
        logger.info("A task is already running, skipping same-day catch-up")
        return
    today = get_business_date(_app_config).isoformat()
    if await db.get_papers_by_date(today):
        return
    tz_name, _tz = get_business_timezone_info(_app_config)
    counts = await db.count_task_runs_started_on(
        today,
        task_name="daily recommendation",
        timezone_name=tz_name,
    )
    if counts["succeeded"] or counts["total"] >= 3:
        logger.info(
            f"Skipping same-day catch-up for {today}: "
            f"{counts['succeeded']} succeeded, {counts['total']} attempts"
        )
        return
    logger.info(f"Starting same-day catch-up for {today}")
    await _start_executor_run(skip_tldr=False, trigger="catchup")
    try:
        await _task_runner.wait()
    except Exception as exc:
        logger.error(f"Same-day catch-up failed: {exc}")


app = FastAPI(title="arXiv Daily", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _render(template_name: str, context: dict[str, Any]) -> HTMLResponse:
    tmpl = _jinja_env.get_template(template_name)
    return HTMLResponse(tmpl.render(**context))


@app.get("/", response_class=HTMLResponse)
async def index():
    dates = await db.get_all_dates()
    current_date = dates[0] if dates else None
    return _render("index.html", await _build_page_context(current_date))


@app.get("/date/{date}", response_class=HTMLResponse)
async def papers_by_date(date: str):
    return _render("index.html", await _build_page_context(date))


@app.post("/api/run")
async def trigger_run(request: Request):
    await _require_admin_password(request)
    if _task_runner.is_running():
        raise HTTPException(status_code=409, detail="A task is already running")

    await _start_executor_run(skip_tldr=False, trigger="manual")
    return JSONResponse({"status": "started", "message": "Task started"})


@app.post("/api/run-until")
async def trigger_run_until(request: Request):
    await _require_admin_password(request)
    if _task_runner.is_running():
        raise HTTPException(status_code=409, detail="A task is already running")

    body = await request.json()
    start_date = _parse_backfill_date(body.get("start_date"), "start_date")
    end_date = _parse_backfill_date(body.get("end_date"), "end_date")
    today = _current_business_date(_app_config)
    if start_date > today or end_date > today:
        raise HTTPException(status_code=400, detail="backfill dates cannot be later than today")
    if start_date > end_date:
        raise HTTPException(
            status_code=400,
            detail="start_date must be the older date and cannot be later than end_date",
        )

    await _start_backfill_run(
        start_date.isoformat(),
        end_date.isoformat(),
        skip_tldr=bool(body.get("skip_tldr", False)),
    )
    return JSONResponse({"status": "started", "message": "Backfill task started"})


@app.get("/api/status")
async def task_status():
    status = await _task_runner.get_status()
    status["latest_run"] = _decorate_latest_run_for_display(status.get("latest_run"), _app_config)
    return JSONResponse(status)


@app.get("/api/config")
async def get_config():
    return JSONResponse(public_config(_app_config))


def _public_read_headers(request: Request) -> dict[str, str]:
    headers = {"Cache-Control": "public, max-age=60", "Vary": "Origin"}
    origin = request.headers.get("origin")
    if origin in get_config_value(_app_config, "server.widget_allowed_origins"):
        headers["Access-Control-Allow-Origin"] = origin
    return headers


@app.get("/api/widget")
async def latest_widget(request: Request, limit: int = Query(3, ge=1, le=5)):
    dates = await db.get_all_dates()
    latest_date = dates[0] if dates else None
    rows = await db.get_papers_by_date(latest_date) if latest_date else []
    return JSONResponse({
        "date": latest_date,
        "total": len(rows),
        "papers": [{
            "id": row["id"],
            "title": row["title"],
            "tldr": str(row.get("tldr") or "")[:320],
            "score": row.get("judge_relevance") if row.get("judge_relevance") is not None else row.get("score"),
            "score_max": 5 if row.get("judge_relevance") is not None else 10,
            "detail_url": f"https://arxiv.luolimasi.xyz/date/{latest_date}#paper-{row['id']}",
        } for row in rows[:limit]],
    }, headers=_public_read_headers(request))


@app.get("/api/terminal")
async def terminal_papers(
    request: Request,
    date: str | None = Query(None, pattern=r"^\d{4}-\d{2}-\d{2}$"),
    q: str | None = Query(None, min_length=1, max_length=120),
    limit: int = Query(5, ge=1, le=10),
):
    headers = _public_read_headers(request)
    dates = await db.get_all_dates()
    if q is not None and (not q.strip() or date is not None):
        return JSONResponse({"detail": "Search requires a nonempty query without a date"}, status_code=400, headers=headers)
    if date is not None and date not in dates:
        return JSONResponse({"detail": "No recommendations for this date"}, status_code=404, headers=headers)
    selected_date = date or (dates[0] if dates else None)
    if q is not None:
        rows = await db.search_recommended_papers(q.strip(), limit)
        total = rows[0]["matched_count"] if rows else 0
        selected_date = None
    else:
        rows = await db.get_papers_by_date(selected_date) if selected_date else []
        total = len(rows)
        rows = rows[:limit]
    papers = []
    for row in rows:
        authors = row.get("authors") or []
        if isinstance(authors, str):
            try:
                authors = json.loads(authors)
            except json.JSONDecodeError:
                authors = []
        pdf_url = row.get("pdf_url")
        try:
            arxiv_id(pdf_url or "")
        except ExportError:
            pdf_url = None
        papers.append({
            "id": row["id"], "date": row["date"], "title": row["title"],
            "authors": authors if isinstance(authors, list) else [],
            "abstract": str(row.get("abstract") or "")[:6000],
            "tldr": str(row.get("tldr") or "")[:3200],
            "reason": str(row.get("judge_reason") or "")[:2000],
            "why_for_me": str(row.get("why_for_me") or "")[:2000],
            "score": row.get("judge_relevance") if row.get("judge_relevance") is not None else row.get("score"),
            "score_max": 5 if row.get("judge_relevance") is not None else 10,
            "pdf_url": pdf_url,
            "detail_url": f"https://arxiv.luolimasi.xyz/date/{row['date']}#paper-{row['id']}",
        })
    return JSONResponse({"date": selected_date, "dates": dates[:30], "total": total,
                         "papers": papers}, headers=headers)


@app.get("/api/config/admin")
async def get_admin_config(request: Request):
    await _require_admin_password(request)
    return JSONResponse(admin_config_view(_app_config))


@app.post("/api/config")
async def update_config(request: Request):
    global _app_config
    await _require_admin_password(request)
    body = await request.json()

    merged = deep_merge(_app_config, body)

    def is_masked(v: str) -> bool:
        return v == "****" or (v.endswith("****") and len(v) > 4)

    for section_key in ("webdav", "llm", "zotero"):
        old_section = _app_config.get(section_key, {})
        new_section = body.get(section_key, {})
        for k in SENSITIVE_KEYS:
            new_val = new_section.get(k, "")
            old_val = old_section.get(k, "")
            if k in old_section and old_val:
                if not new_val or is_masked(new_val):
                    merged[section_key][k] = old_val

    _app_config = merged
    save_config(build_override_config(get_default_config(), merged))

    _reschedule_daily_run(merged)

    return JSONResponse({"status": "ok"})


@app.post("/api/webdav/test")
async def test_webdav(request: Request):
    await _require_admin_password(request)
    body = await request.json()
    local_path = body.get("local_path", "")
    if local_path:
        ok = os.path.isdir(local_path)
        return JSONResponse({"success": ok, "mode": "local"})

    from .webdav import WebDAVClient

    client = WebDAVClient(
        url=body.get("url", ""),
        username=body.get("username", ""),
        password=body.get("password", ""),
        base_path=body.get("path", "/papers"),
    )
    ok = client.test_connection()
    return JSONResponse({"success": ok, "mode": "webdav"})


@app.post("/api/llm/test")
async def test_llm(request: Request):
    await _require_admin_password(request)
    body = await request.json()
    from .llm import test_connection

    saved = _app_config.get("llm") or {}
    api_key = str(body.get("api_key") or "")
    if not api_key or api_key == "****":
        api_key = str(saved.get("api_key") or "")
    test_config = {
        "llm": {
            **saved,
            **{key: value for key, value in body.items() if value not in (None, "")},
            "api_key": api_key,
        }
    }
    ok = test_connection(test_config)
    return JSONResponse({"success": ok})


@app.post("/api/corpus/reload")
async def reload_corpus(request: Request):
    await _require_admin_password(request)
    if _task_runner.is_running():
        raise HTTPException(status_code=409, detail="A task is already running")

    async def _reload():
        corpus = await Executor(_app_config).fetch_corpus(force_refresh=True)
        await db.save_corpus_cache(corpus)
        logger.info(f"Reloaded corpus: {len(corpus)} papers")
        return corpus

    await _task_runner.start(
        "corpus reload",
        _reload,
        trigger="manual",
        result_to_metrics=lambda corpus: {"corpus_count": len(corpus)},
    )
    return JSONResponse({"status": "started", "message": "Corpus reload started"})


@app.post("/api/feedback")
async def save_feedback(request: Request):
    await _require_admin_password(request)
    body = await request.json()
    url = str(body.get("url") or "").strip()
    vote = str(body.get("vote") or "").strip()
    if not url or vote not in {"relevant", "irrelevant"}:
        raise HTTPException(status_code=400, detail="url and a relevant/irrelevant vote are required")
    await db.upsert_feedback(
        url=url,
        vote=vote,
        date=str(body.get("date") or ""),
        title=str(body.get("title") or ""),
        tldr=str(body.get("tldr") or ""),
    )
    return JSONResponse({"status": "ok", "vote": vote})


@app.post("/api/papers/{paper_id}/zotero")
async def save_to_zotero(paper_id: int, request: Request):
    await _require_admin_password(request)
    paper = await db.get_paper_by_id(paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    try:
        record = await export_paper(paper, copy.deepcopy(_app_config))
    except ExportError as error:
        raise HTTPException(status_code=400, detail=str(error)) from None
    return JSONResponse({
        "status": record["status"],
        "item_key": record["paper_key"],
        "corpus_size": await db.get_corpus_count(),
    })


@app.post("/api/enrich")
async def enrich_saved(request: Request):
    await _require_admin_password(request)
    if _task_runner.is_running():
        raise HTTPException(status_code=409, detail="A task is already running")
    body = await request.json()
    start_date = _parse_backfill_date(body.get("start_date"), "start_date")
    end_date = _parse_backfill_date(body.get("end_date"), "end_date")
    today = _current_business_date(_app_config)
    if start_date > today or end_date > today or start_date > end_date:
        raise HTTPException(status_code=400, detail="invalid enrich date range")
    dates = business_date_range_between(_app_config, start_date, end_date)
    executor = Executor(_app_config)
    await _task_runner.start(
        "enrich saved papers",
        lambda: executor.enrich_saved_dates(dates),
        trigger="manual",
        metadata={"start_date": start_date.isoformat(), "end_date": end_date.isoformat()},
        result_to_metrics=lambda result: dict(result) if isinstance(result, dict) else {},
    )
    return JSONResponse({"status": "started", "message": "Enrichment started"})


@app.post("/api/qa")
async def ask_question(request: Request):
    await _require_admin_password(request)
    if _task_runner.is_running():
        raise HTTPException(status_code=409, detail="A task is already running")
    body = await request.json()
    question = str(body.get("question") or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is required")
    if len(question) > 1000:
        raise HTTPException(status_code=400, detail="question is too long")

    import asyncio

    from .lexical import rank_items_by_query
    from .llm import answer_question as generate_answer
    from .llm import reset_llm_usage

    today = _current_business_date(_app_config)
    papers = await db.get_papers_between((today - timedelta(days=21)).isoformat(), today.isoformat())
    corpus = await db.load_corpus_cache()
    items: list[dict[str, str]] = []
    for paper in corpus:
        items.append(
            {
                "title": paper.title,
                "abstract": paper.abstract or "",
                "url": "",
                "date": "",
                "tldr": "",
                "reason": "语料库",
                "score": "",
            }
        )
    for paper in papers:
        items.append(
            {
                "title": str(paper.get("title") or ""),
                "abstract": str(paper.get("abstract") or ""),
                "url": str(paper.get("url") or ""),
                "date": str(paper.get("date") or ""),
                "tldr": str(paper.get("tldr") or ""),
                "reason": str(paper.get("judge_reason") or ""),
                "score": str(paper.get("judge_relevance") or paper.get("score") or ""),
            }
        )
    loop = asyncio.get_running_loop()
    ranked = await loop.run_in_executor(
        None, lambda: rank_items_by_query(question, items, top_k=8)
    )
    contexts = []
    for index, score in ranked:
        item = dict(items[index])
        item["score"] = f"{score:.3f}"
        contexts.append(item)
    if not contexts:
        return JSONResponse({"answer": "本地库里没有可检索的论文。", "sources": []})
    reset_llm_usage()
    answer = await loop.run_in_executor(
        None, lambda: generate_answer(question, contexts, _app_config)
    )
    if not answer:
        raise HTTPException(status_code=502, detail="LLM 没有返回答案")
    sources = [
        {
            "title": item["title"],
            "url": item["url"],
            "date": item["date"],
            "score": item["score"],
        }
        for item in contexts
    ]
    return JSONResponse({"answer": answer, "sources": sources})


@app.get("/api/stats")
async def get_stats():
    dates = await db.get_all_dates()
    total = await db.get_paper_count()
    corpus = await db.get_corpus_count()
    latest_run = _decorate_latest_run_for_display(await db.get_latest_task_run(), _app_config)
    return JSONResponse(
        {
            "total_dates": len(dates),
            "total_papers": total,
            "corpus_size": corpus,
            "latest_date": dates[0] if dates else None,
            "latest_run": latest_run,
        }
    )


def _cli_backfill_tldr(args: Any) -> None:
    import asyncio

    from .executor import plan_tldr_backfill

    config = load_config()
    explicit = [str(value) for value in (args.date or [])]
    recent: list[str] = []
    if args.recent_days:
        if args.recent_days < 1:
            raise SystemExit("--recent-days must be at least 1")
        start = (get_business_date(config) - timedelta(days=args.recent_days)).isoformat()
        recent = asyncio.run(db.list_dates_with_empty_tldr(start))
    if not explicit and not recent:
        raise SystemExit("pass --date and/or --recent-days")
    plan = plan_tldr_backfill(explicit, recent, only_empty=bool(args.only_empty))
    result = asyncio.run(Executor(config).backfill_tldrs(plan))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "completed":
        raise SystemExit(1)


def _cli_backfill_day(args: Any) -> None:
    import asyncio

    config = load_config()
    dates = [str(value) for value in (args.date or [])]
    if not dates:
        raise SystemExit("pass --date")
    for value in dates:
        try:
            date.fromisoformat(value)
        except ValueError as exc:
            raise SystemExit(f"invalid date {value}: expected YYYY-MM-DD") from exc
    result = asyncio.run(Executor(config).backfill_days(dates, force=bool(args.force)))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result.get("status") != "completed":
        raise SystemExit(1)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(prog="arxiv-daily")
    sub = parser.add_subparsers(dest="command")
    backfill = sub.add_parser(
        "backfill-tldr",
        help="Regenerate TLDRs for recommendations already saved in the database",
    )
    backfill.add_argument(
        "--date",
        action="append",
        default=[],
        help="Business date to regenerate, YYYY-MM-DD. Can be repeated.",
    )
    backfill.add_argument(
        "--recent-days",
        type=int,
        default=0,
        help="Also fill empty TLDRs on dates within this many days of the business date.",
    )
    backfill.add_argument(
        "--only-empty",
        action="store_true",
        help="On --date, skip papers that already have a TLDR.",
    )
    backfill_day = sub.add_parser(
        "backfill-day",
        help="Generate recommendations for historical business dates from the arXiv API",
    )
    backfill_day.add_argument(
        "--date",
        action="append",
        default=[],
        help="Business date to generate, YYYY-MM-DD. Can be repeated.",
    )
    backfill_day.add_argument(
        "--force",
        action="store_true",
        help="Regenerate a date even when recommendations are already saved.",
    )
    args = parser.parse_args(argv)
    if args.command == "backfill-tldr":
        _cli_backfill_tldr(args)
        return
    if args.command == "backfill-day":
        _cli_backfill_day(args)
        return

    import uvicorn

    config = load_config()
    host = get_config_value(config, "server.host")
    port = int(get_config_value(config, "server.port"))
    uvicorn.run("arxiv_daily.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
