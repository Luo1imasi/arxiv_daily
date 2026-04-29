import copy
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from .config import get_config_value


def get_business_timezone(config: dict[str, Any]) -> ZoneInfo:
    return ZoneInfo(str(get_config_value(config, "executor.timezone")))


def get_business_timezone_info(config: dict[str, Any]) -> tuple[str, ZoneInfo]:
    tz_name = str(get_config_value(config, "executor.timezone"))
    return tz_name, ZoneInfo(tz_name)


def normalize_business_date(value: str | date) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)).isoformat()


def get_business_date(config: dict[str, Any]) -> date:
    configured_date = get_config_value(config, "executor.business_date", None)
    if configured_date:
        return date.fromisoformat(normalize_business_date(configured_date))
    return datetime.now(get_business_timezone(config)).date()


def get_business_date_string(config: dict[str, Any]) -> str:
    return get_business_date(config).isoformat()


def business_date_range_until(config: dict[str, Any], until_date: str | date) -> list[str]:
    current = get_business_date(config)
    target = date.fromisoformat(normalize_business_date(until_date))
    if target > current:
        raise ValueError("until_date cannot be later than the current business date")
    days = (current - target).days
    return [(target + timedelta(days=offset)).isoformat() for offset in range(days + 1)]


def business_date_range_between(
    config: dict[str, Any], start_date: str | date, end_date: str | date
) -> list[str]:
    current = get_business_date(config)
    start = date.fromisoformat(normalize_business_date(start_date))
    end = date.fromisoformat(normalize_business_date(end_date))
    if start > current or end > current:
        raise ValueError("backfill dates cannot be later than the current business date")
    if start > end:
        raise ValueError("start_date must be the older date and cannot be later than end_date")
    days = (end - start).days
    return [(start + timedelta(days=offset)).isoformat() for offset in range(days + 1)]


def config_for_business_date(config: dict[str, Any], business_date: str | date) -> dict[str, Any]:
    scoped = copy.deepcopy(config)
    scoped.setdefault("executor", {})["business_date"] = normalize_business_date(business_date)
    return scoped


def business_window_utc(
    config: dict[str, Any],
    *,
    start_days_ago: int = 0,
    end_days_ago: int | None = None,
) -> tuple[datetime, datetime]:
    tz = get_business_timezone(config)
    target_date = get_business_date(config)
    start_days_ago = max(0, int(start_days_ago))
    lookback_days = int(end_days_ago) if end_days_ago is not None else start_days_ago + 1
    if lookback_days <= 0 or start_days_ago >= lookback_days:
        raise ValueError("invalid arXiv lookback window")

    window_end_local = datetime.combine(
        target_date + timedelta(days=1 - start_days_ago), time.min, tzinfo=tz
    )
    window_start_local = datetime.combine(
        target_date + timedelta(days=1 - lookback_days), time.min, tzinfo=tz
    )
    return (
        window_start_local.astimezone(timezone.utc),
        window_end_local.astimezone(timezone.utc),
    )


def reference_datetime(config: dict[str, Any]) -> datetime:
    configured_date = get_config_value(config, "executor.business_date", None)
    if not configured_date:
        return datetime.now(timezone.utc)
    return datetime.combine(get_business_date(config), time.max, tzinfo=timezone.utc)
