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


_ARXIV_ET = ZoneInfo("America/New_York")

# Mornings (Asia/Shanghai business dates) that receive no new arXiv mailing.
# The previous public mailing's window is reused. Sources: arXiv holiday posts
# for New Year 2026, Juneteenth 2026, Independence Day 2026, and Labor Day 2026.
_NO_ANNOUNCEMENT_MORNINGS = {
    date(2026, 1, 1),
    date(2026, 1, 2),
    date(2026, 1, 3),
    date(2026, 1, 4),
    date(2026, 6, 22),
    date(2026, 7, 6),
    date(2026, 9, 8),
}

# Mornings that receive a deferred mailing. Bounds are 14:00 ET submission
# cutoffs: [start, end). The end date is exclusive at 14:00 ET.
_DEFERRED_WINDOWS = {
    date(2026, 1, 5): (date(2025, 12, 31), date(2026, 1, 2)),
    date(2026, 6, 23): (date(2026, 6, 18), date(2026, 6, 22)),
    date(2026, 7, 7): (date(2026, 7, 2), date(2026, 7, 6)),
    date(2026, 9, 9): (date(2026, 9, 4), date(2026, 9, 8)),
}


def _as_date(value: str | date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _previous_weekday(day: date) -> date:
    cursor = day - timedelta(days=1)
    while cursor.weekday() >= 5:
        cursor -= timedelta(days=1)
    return cursor


def _cutoff_utc(day: date) -> datetime:
    """14:00 US/Eastern on `day`, as UTC. arXiv's submission deadline."""
    return datetime.combine(day, time(14, 0), tzinfo=_ARXIV_ET).astimezone(timezone.utc)


def submitted_not_after_utc(config: dict[str, Any], business_day: str | date) -> datetime:
    """Exclusive upper bound: submissions from the start of the next business day.

    Papers submitted on business date D are allowed. Later ones are not.
    """
    day = _as_date(business_day)
    tz = get_business_timezone(config)
    return datetime.combine(day + timedelta(days=1), time.min, tzinfo=tz).astimezone(timezone.utc)


def announcement_window_utc(business_day: str | date) -> tuple[datetime, datetime, str, str]:
    """Submitted-date window for the arXiv mailing read on business date D.

    Weekday D's mailing closes at 14:00 ET on the previous weekday. The window
    opens at the cutoff before that, so Tuesday covers Friday 14:00 through
    Monday 14:00 and the other weekdays cover a single weekday-to-weekday span.
    The interval is half-open: [start, end).

    Saturday, Sunday, and known holidays with no mailing reuse the nearest
    earlier mailing. A deferred holiday mailing uses the widened cutoff pair
    arXiv announced for that morning.

    Returns (start_utc, end_utc, kind, source_date). `kind` is "announcement",
    "reused", or "deferred". `source_date` is the morning whose mailing this is.
    """
    day = _as_date(business_day)
    start, end, kind, source = _announcement_window(day, depth=0)
    return start, end, kind, source


def _announcement_window(day: date, *, depth: int) -> tuple[datetime, datetime, str, str]:
    if depth > 21:
        raise ValueError(f"no arXiv announcement window within 21 days of {day.isoformat()}")
    widened = _DEFERRED_WINDOWS.get(day)
    if widened is not None:
        start_day, end_day = widened
        return _cutoff_utc(start_day), _cutoff_utc(end_day), "deferred", day.isoformat()
    if day.weekday() >= 5 or day in _NO_ANNOUNCEMENT_MORNINGS:
        anchor = day - timedelta(days=1)
        while anchor.weekday() >= 5 or anchor in _NO_ANNOUNCEMENT_MORNINGS:
            anchor -= timedelta(days=1)
        start, end, _kind, _source = _announcement_window(anchor, depth=depth + 1)
        return start, end, "reused", anchor.isoformat()
    end_day = _previous_weekday(day)
    start_day = _previous_weekday(end_day)
    return _cutoff_utc(start_day), _cutoff_utc(end_day), "announcement", day.isoformat()
