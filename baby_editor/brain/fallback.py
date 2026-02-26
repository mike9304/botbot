"""Rule-based fallback engine when LLM is unavailable or fails.

Guarantees a rough cut is always produced even without an LLM.
"""

from __future__ import annotations


def fallback_select(
    clips: list[dict],
    target_duration: float,
) -> list[dict]:
    """Select clips by score until target duration is met (±15%)."""
    sorted_clips = sorted(clips, key=lambda c: c["peak_score"], reverse=True)
    selected: list[dict] = []
    total = 0.0

    for clip in sorted_clips:
        if total + clip["duration"] <= target_duration * 1.15:
            selected.append(clip)
            total += clip["duration"]

    return selected


def fallback_order(clips: list[dict]) -> list[dict]:
    """Order clips chronologically with simple transitions."""
    ordered = sorted(clips, key=lambda c: (c["video"], c["start"]))

    for i, clip in enumerate(ordered):
        clip["position"] = i + 1
        clip["role"] = "opening" if i == 0 else ("closing" if i == len(ordered) - 1 else "middle")
        clip["transition_in"] = {
            "type": "fade_black" if i == 0 else "dissolve",
            "duration": 0.5,
        }
        clip["transition_out"] = {"type": "dissolve", "duration": 0.5}
        clip["placement_reason"] = "Fallback: chronological order by score"

    return ordered


def fallback_full(
    clips: list[dict],
    target_duration: float,
) -> dict:
    """Run full fallback pipeline: select + order.

    Returns a narrative dict compatible with the LLM output format.
    """
    selected = fallback_select(clips, target_duration)
    ordered = fallback_order(selected)
    actual_dur = sum(c["duration"] for c in ordered)

    return {
        "title": "Baby Highlights (auto)",
        "total_duration": round(actual_dur, 1),
        "timeline": ordered,
    }
