"""Parse and validate LLM JSON responses for each editing stage."""

from __future__ import annotations

import logging

logger = logging.getLogger("baby_editor.brain")


def parse_selection(response: dict | None) -> list[dict] | None:
    """Extract selected clips from Stage A response.

    Returns list of clip dicts or None if invalid.
    """
    if response is None:
        return None
    try:
        sel = response["selection"]
        clips = sel["clips"]
        if not isinstance(clips, list) or len(clips) == 0:
            logger.warning("Selection response has empty clips list")
            return None
        return clips
    except (KeyError, TypeError) as exc:
        logger.warning("Failed to parse selection response: %s", exc)
        return None


def parse_narrative(response: dict | None) -> dict | None:
    """Extract narrative timeline from Stage B response.

    Returns narrative dict with 'timeline' key or None.
    """
    if response is None:
        return None
    try:
        nar = response["narrative"]
        timeline = nar["timeline"]
        if not isinstance(timeline, list) or len(timeline) == 0:
            logger.warning("Narrative response has empty timeline")
            return None
        return nar
    except (KeyError, TypeError) as exc:
        logger.warning("Failed to parse narrative response: %s", exc)
        return None


def parse_validation(response: dict | None) -> dict:
    """Extract validation result from Stage C response.

    Returns validation dict; defaults to {"passed": True} on failure.
    """
    if response is None:
        return {"passed": True, "checks": {}}
    try:
        val = response["validation"]
        return val
    except (KeyError, TypeError) as exc:
        logger.warning("Failed to parse validation response: %s", exc)
        return {"passed": True, "checks": {}}
