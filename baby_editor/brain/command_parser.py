"""Parse Korean natural language commands into structured parameters.

Simple keyword matching — no LLM needed.
"""

from __future__ import annotations

import re


def parse_command(command: str | None) -> dict:
    """Convert a Korean natural-language command to editing parameters.

    Returns a dict with keys: date, target, style, focus, format, mode.
    """
    params: dict = {
        "date": "today",
        "target": "auto",
        "style": "story_arc",
        "focus": "all",
        "format": "horizontal",
        "mode": "edit",
    }
    if not command:
        return params

    # --- Date ---
    if "오늘" in command:
        params["date"] = "today"
    elif "어제" in command:
        params["date"] = "yesterday"
    elif "이번 주" in command or "주간" in command:
        params["date"] = "this_week"
    elif "이번 달" in command or "월간" in command:
        params["date"] = "this_month"

    # --- Duration ---
    total_sec = 0
    m = re.search(r"(\d+)\s*분", command)
    if m:
        total_sec += int(m.group(1)) * 60
    s = re.search(r"(\d+)\s*초", command)
    if s:
        total_sec += int(s.group(1))
    if total_sec > 0:
        params["target"] = total_sec

    # --- Focus ---
    focus_map = {
        "웃": "laughing",
        "미소": "laughing",
        "목욕": "bathing",
        "밥": "eating",
        "먹": "eating",
        "이유식": "eating",
        "자는": "sleeping",
        "잠": "sleeping",
        "놀": "playing",
        "쌍둥이": "twins_together",
        "같이": "twins_together",
        "걸음마": "milestone",
        "첫": "milestone",
    }
    for kw, val in focus_map.items():
        if kw in command:
            params["focus"] = val
            break

    # --- Format ---
    if any(k in command for k in ("릴스", "숏츠", "세로")):
        params["format"] = "vertical"

    # --- Style ---
    if "하이라이트" in command:
        params["style"] = "highlight_best_first"
    elif "시간순" in command:
        params["style"] = "chronological"

    # --- Mode ---
    if any(k in command for k in ("찾아", "검색")):
        params["mode"] = "search"

    return params
