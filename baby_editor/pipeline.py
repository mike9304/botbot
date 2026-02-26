"""Full pipeline orchestrator: Phase 1 → Phase 2 → Phase 3.

Ties together analysis, LLM editing brain, and rendering into a single call.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("baby_editor.pipeline")


def run_full_pipeline(
    input_dir: str,
    output_dir: str,
    user_command: str | None = None,
    config: dict | None = None,
) -> dict:
    """Run the complete Phase 1 → 2 → 3 pipeline.

    Args:
        input_dir: Folder containing .mp4/.mov video files.
        output_dir: Where to write all outputs.
        user_command: Optional Korean NL editing command.
        config: Full config dict (from config.yaml).

    Returns:
        dict with paths to all output files + summary stats.
    """
    config = config or {}
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Analysis → score_matrix.json
    # ------------------------------------------------------------------
    logger.info("Phase 1: Analyzing videos in %s", input_dir)
    from baby_editor.analyze import AnalysisPipeline

    analyzer = AnalysisPipeline(config)
    score_matrix = analyzer.analyze_folder(input_dir)

    score_path = os.path.join(output_dir, "score_matrix.json")
    Path(score_path).write_text(
        json.dumps(score_matrix, ensure_ascii=False, indent=2)
    )
    logger.info("Phase 1 complete: %s", score_path)

    # ------------------------------------------------------------------
    # Phase 2: LLM Brain → edit_plan.json
    # ------------------------------------------------------------------
    logger.info("Phase 2: Running editing brain")
    from baby_editor.brain.orchestrator import BrainOrchestrator

    brain = BrainOrchestrator(config)
    edit_plan_path = os.path.join(output_dir, "edit_plan.json")
    edit_plan = brain.run(
        score_matrix,
        user_command=user_command,
        output_path=edit_plan_path,
    )
    logger.info("Phase 2 complete: %s", edit_plan_path)

    timeline = edit_plan.get("timeline", [])

    if not timeline:
        logger.warning("Empty timeline — skipping rendering")
        return {
            "score_matrix": score_path,
            "edit_plan": edit_plan_path,
            "rough_cut": None,
            "timeline_xml": None,
            "report": None,
            "duration": 0,
            "clip_count": 0,
        }

    # ------------------------------------------------------------------
    # Phase 3: Rendering → rough_cut.mp4, timeline.xml, report.json
    # ------------------------------------------------------------------
    logger.info("Phase 3: Rendering %d clips", len(timeline))
    from baby_editor.render.ffmpeg_renderer import FFmpegRenderer
    from baby_editor.render.fcp_xml import FCPXMLGenerator
    from baby_editor.render.report import generate_report

    # 3a. Render rough cut
    rough_cut_path = os.path.join(output_dir, "rough_cut.mp4")
    renderer = FFmpegRenderer(config)
    try:
        renderer.render(timeline, rough_cut_path)
    finally:
        renderer.cleanup()

    # 3b. Generate FCP XML timeline
    xml_path = os.path.join(output_dir, "timeline.xml")
    fcp = FCPXMLGenerator(fps=30.0)
    fcp.generate(edit_plan, xml_path)

    # 3c. Generate report
    report_path = os.path.join(output_dir, "report.json")
    generate_report(edit_plan, score_matrix, report_path)

    # 3d. Optional vertical reframe
    results: dict = {
        "score_matrix": score_path,
        "edit_plan": edit_plan_path,
        "rough_cut": rough_cut_path,
        "timeline_xml": xml_path,
        "report": report_path,
        "duration": edit_plan.get("parameters", {}).get("actual_duration", 0),
        "clip_count": len(timeline),
    }

    fmt = edit_plan.get("parameters", {}).get("format", "horizontal")
    if fmt == "vertical":
        from baby_editor.render.vertical_reframe import render_vertical

        reels_path = os.path.join(output_dir, "reels.mp4")
        vert_cfg = config.get("render", {}).get("vertical", {})
        render_vertical(
            rough_cut_path,
            reels_path,
            width=vert_cfg.get("width", 1080),
            height=vert_cfg.get("height", 1920),
        )
        results["reels"] = reels_path

    logger.info("Phase 3 complete")
    return results
