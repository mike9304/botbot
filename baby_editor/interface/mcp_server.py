"""MCP Server exposing baby video editing tools to Claude Desktop.

Tools: analyze_video, create_roughcut, create_reel, get_report, get_status.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger("baby_editor.mcp")

# Lazy import — mcp may not be installed
_app = None

VIDEO_BASE = "/videos"
OUTPUT_BASE = "/output"


def _resolve_date(date_str: str) -> str:
    today = datetime.now()
    if date_str == "today":
        return today.strftime("%Y-%m-%d")
    if date_str == "yesterday":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    return date_str


def _get_app():
    global _app
    if _app is not None:
        return _app

    from mcp.server import Server

    _app = Server("baby-video-editor")

    @_app.tool()
    async def analyze_video(date: str = "today") -> str:
        """Analyze video files for a given date.

        Args:
            date: "today", "yesterday", or "YYYY-MM-DD"
        """
        resolved = _resolve_date(date)
        input_folder = os.path.join(VIDEO_BASE, resolved)

        if not os.path.exists(input_folder):
            return f"No video folder found for {resolved}"

        from baby_editor.analyze import AnalysisPipeline

        pipeline = AnalysisPipeline({})
        score_matrix = await asyncio.to_thread(pipeline.analyze_folder, input_folder)

        summary = score_matrix.get("daily_summary", {})
        highlights = summary.get("highlights", [])

        lines = [
            f"Analysis complete for {resolved}:",
            f"  Videos: {summary.get('total_videos', 0)}",
            f"  Total duration: {summary.get('total_duration_sec', 0) / 60:.1f} min",
            f"  Segments analyzed: {summary.get('total_segments', 0)}",
            f"  Auto-deleted: {summary.get('deleted_pct', 0):.0f}%",
            f"  Candidates: {summary.get('candidate_pct', 0):.0f}%",
        ]
        if highlights:
            h = highlights[0]
            lines.append(
                f"  Top highlight: {h['video']} "
                f"[{h['start']}-{h['end']}s] "
                f"(score {h['score']}, {h['top_tag']})"
            )
        return "\n".join(lines)

    @_app.tool()
    async def create_roughcut(
        date: str = "today",
        duration: str = "auto",
        focus: str = "all",
        style: str = "story_arc",
    ) -> str:
        """Create a rough cut highlight video.

        Args:
            date: "today", "yesterday", or "YYYY-MM-DD"
            duration: "auto", "60", "1분", "30초"
            focus: "all", "laughing", "playing", "bathing", "twins_together"
            style: "story_arc", "chronological", "highlight_best_first"
        """
        resolved = _resolve_date(date)
        input_folder = os.path.join(VIDEO_BASE, resolved)
        output_folder = os.path.join(OUTPUT_BASE, resolved)
        os.makedirs(output_folder, exist_ok=True)

        parts = [date]
        if duration != "auto":
            parts.append(duration)
        if focus != "all":
            parts.append(focus)
        if style != "story_arc":
            parts.append(style)
        user_command = " ".join(parts)

        from baby_editor.pipeline import run_full_pipeline

        results = await asyncio.to_thread(
            run_full_pipeline, input_folder, output_folder, user_command, {}
        )

        return (
            f"Rough cut created!\n"
            f"  Video: {results.get('rough_cut', 'N/A')}\n"
            f"  Duration: {results.get('duration', 0):.0f}s\n"
            f"  Clips: {results.get('clip_count', 0)}\n"
            f"  Timeline XML: {results.get('timeline_xml', 'N/A')}\n"
            f"  Report: {results.get('report', 'N/A')}"
        )

    @_app.tool()
    async def create_reel(
        date: str = "today",
        duration: str = "30초",
        focus: str = "all",
    ) -> str:
        """Create a vertical short-form video for Reels/Shorts.

        Args:
            date: Target date
            duration: Max 60초
            focus: Content focus filter
        """
        resolved = _resolve_date(date)
        output_folder = os.path.join(OUTPUT_BASE, resolved)
        os.makedirs(output_folder, exist_ok=True)

        user_command = f"{date} {focus} {duration} 릴스"

        from baby_editor.pipeline import run_full_pipeline

        results = await asyncio.to_thread(
            run_full_pipeline,
            os.path.join(VIDEO_BASE, resolved),
            output_folder, user_command, {}
        )
        return f"Reel created: {results.get('reels', 'N/A')}"

    @_app.tool()
    async def get_report(date: str = "today") -> str:
        """View the editing report for a given date."""
        resolved = _resolve_date(date)
        report_path = os.path.join(OUTPUT_BASE, resolved, "report.json")
        if os.path.exists(report_path):
            with open(report_path) as f:
                report = json.load(f)
            return json.dumps(report.get("summary", {}), ensure_ascii=False, indent=2)
        return f"No report found for {resolved}"

    return _app


def run_mcp_server(config: dict | None = None) -> None:
    """Start the MCP server."""
    global VIDEO_BASE, OUTPUT_BASE
    if config:
        iface = config.get("interface", {})
        VIDEO_BASE = iface.get("video_base_path", VIDEO_BASE)
        OUTPUT_BASE = iface.get("output_base_path", OUTPUT_BASE)

    import mcp
    mcp.run(_get_app())
