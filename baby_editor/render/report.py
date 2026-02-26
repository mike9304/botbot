"""Generate a human-readable JSON report documenting all editing decisions."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


def generate_report(
    edit_plan: dict,
    score_matrix: dict,
    output_path: str,
) -> str:
    """Write a report.json summarising how the edit was constructed.

    Returns the output path.
    """
    daily = score_matrix.get("daily_summary", {})
    params = edit_plan.get("parameters", {})
    actual_dur = params.get("actual_duration", 1)
    total_input = daily.get("total_duration_sec", 1)
    ratio = total_input / actual_dur if actual_dur else 0

    report = {
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "date": params.get("date", ""),
            "input_videos": daily.get("total_videos", 0),
            "input_duration_sec": total_input,
            "output_duration_sec": actual_dur,
            "compression_ratio": f"{ratio:.1f}:1",
            "clips_selected": len(edit_plan.get("timeline", [])),
            "segments_deleted_pct": daily.get("deleted_pct", 0),
            "llm_provider": params.get("llm_provider", "unknown"),
        },
        "timeline_details": [
            {
                "position": clip.get("position", i + 1),
                "source": clip.get("source_video", clip.get("video", "")),
                "time_range": (
                    f"{clip.get('in_point', clip.get('in', clip.get('start', 0))):.1f}s"
                    f" - {clip.get('out_point', clip.get('out', clip.get('end', 0))):.1f}s"
                ),
                "duration": f"{clip.get('duration', 0):.1f}s",
                "score": clip.get("score", 0),
                "tags": clip.get("tags", []),
                "transcript": clip.get("transcript", ""),
                "why_selected": clip.get("placement_reason", ""),
                "transition": clip.get("transition_in", {}).get("type", "cut"),
            }
            for i, clip in enumerate(edit_plan.get("timeline", []))
        ],
        "validation": edit_plan.get("validation", {}),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )

    return output_path
