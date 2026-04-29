import os
import sys
import json
import hmac
from datetime import date, datetime, timezone
from pathlib import Path
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Request, HTTPException
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
from .business_date import get_business_date, get_business_timezone_info

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
    corpus_count = await db.get_corpus_count()
    status = await _task_runner.get_status()
    latest_run = _decorate_latest_run_for_display(status.get("latest_run"), _app_config)
    return {
        "papers": papers,
        "dates": dates,
        "current_date": current_date,
        "config": _mask_password(_app_config),
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

    config = load_config()
    _app_config = config

    _scheduler = AsyncIOScheduler()
    tz_name, tz, hour, minute = _get_schedule_settings(config)
    _scheduler.add_job(
        scheduled_run, "cron", hour=hour, minute=minute, id="daily_run", timezone=tz
    )
    _scheduler.start()
    logger.info(f"Scheduler started: daily run at {hour:02d}:{minute:02d} ({tz_name})")

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
    return JSONResponse(_mask_password(_app_config))


@app.post("/api/config")
async def update_config(request: Request):
    global _app_config
    await _require_admin_password(request)
    body = await request.json()

    merged = deep_merge(_app_config, body)

    def is_masked(v: str) -> bool:
        return v == "****" or (v.endswith("****") and len(v) > 4)

    for section_key in ("webdav", "llm"):
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

    test_config = {"llm": body}
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


def main():
    import uvicorn

    config = load_config()
    host = get_config_value(config, "server.host")
    port = int(get_config_value(config, "server.port"))
    uvicorn.run("arxiv_daily.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
