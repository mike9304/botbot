"""Date resolution helpers."""

from __future__ import annotations

from datetime import datetime, timedelta


def resolve_date(date_str: str) -> str:
    """Convert a date alias to YYYY-MM-DD string.

    Supported aliases: "today", "yesterday", "this_week", "this_month".
    Also passes through YYYY-MM-DD strings unchanged.
    """
    today = datetime.now()
    if date_str == "today":
        return today.strftime("%Y-%m-%d")
    if date_str == "yesterday":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    return date_str


def date_range(start: str, end: str) -> list[str]:
    """Return all YYYY-MM-DD strings between start and end (inclusive)."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    dates: list[str] = []
    current = s
    while current <= e:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates
