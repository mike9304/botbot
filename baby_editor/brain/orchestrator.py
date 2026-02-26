"""Phase 2 orchestrator: Stage A (Selection) → B (Narrative) → C (Validation).

Runs the LLM pipeline with automatic fallback to rule-based editing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from baby_editor.brain.command_parser import parse_command
from baby_editor.brain.fallback import fallback_full
from baby_editor.brain.llm_client import LLMClient
from baby_editor.brain.prompt_builder import (
    build_narrative_prompt,
    build_selection_prompt,
    build_validation_prompt,
)
from baby_editor.brain.response_parser import (
    parse_narrative,
    parse_selection,
    parse_validation,
)
from baby_editor.brain.segment_merger import merge_segments

logger = logging.getLogger("baby_editor.brain")


class BrainOrchestrator:
    """Drives the 3-stage LLM editing pipeline."""

    def __init__(self, config: dict):
        brain_cfg = config.get("brain", {})
        llm_cfg = brain_cfg.get("llm", {})
        editing_cfg = brain_cfg.get("editing", {})

        provider = llm_cfg.get("default_provider", "ollama")
        provider_cfg = llm_cfg.get(provider, {})
        model = provider_cfg.get("model", "qwen2.5:32b")

        self.llm = LLMClient(provider=provider, model=model, config=provider_cfg)
        self.editing = editing_cfg
        self.fallback_enabled = llm_cfg.get("fallback_enabled", True)
        self.provider = provider
        self.model = model

    def run(
        self,
        score_matrix: dict,
        user_command: str | None = None,
        output_path: str | None = None,
    ) -> dict:
        """Execute the full Phase 2 pipeline.

        Args:
            score_matrix: Output of Phase 1.
            user_command: Optional Korean NL command string.
            output_path: Where to write edit_plan.json (optional).

        Returns:
            edit_plan dict.
        """
        params = parse_command(user_command)

        # --- collect all segments across videos ---
        all_segments: list[dict] = []
        for video in score_matrix.get("videos", []):
            for seg in video.get("segments", []):
                seg_copy = dict(seg)
                seg_copy["video"] = video["filename"]
                all_segments.append(seg_copy)

        # --- merge into clips ---
        baseline = self.editing.get("selection_baseline", 3)
        handle = self.editing.get("handle_frames_sec", 0.5)
        min_clip = self.editing.get("min_clip_duration", 1.5)
        max_clip = self.editing.get("max_clip_duration", 15.0)

        clips = merge_segments(
            all_segments,
            baseline=baseline,
            handle=handle,
            min_dur=min_clip,
            max_dur=max_clip,
        )
        logger.info("Merged %d segments into %d clips", len(all_segments), len(clips))

        if not clips:
            logger.warning("No candidate clips found — nothing to edit")
            return self._empty_plan(params, user_command)

        # --- resolve target duration ---
        total_input_dur = score_matrix.get("daily_summary", {}).get(
            "total_duration_sec", 600
        )
        target = params.get("target", "auto")
        if target == "auto":
            ratio = self.editing.get("default_target_ratio", 0.17)
            target_dur = max(
                self.editing.get("min_target_sec", 15),
                min(
                    total_input_dur * ratio,
                    self.editing.get("max_target_sec", 300),
                ),
            )
        else:
            target_dur = float(target)

        # --- Stage A: Selection ---
        sys_a, usr_a = build_selection_prompt(
            clips,
            target_dur,
            style=params["style"],
            focus=params["focus"],
            date=params["date"],
        )
        resp_a = self.llm.generate(sys_a, usr_a)
        selected_clips = parse_selection(resp_a)

        if selected_clips is None:
            if self.fallback_enabled:
                logger.warning("Stage A failed — using fallback")
                narrative = fallback_full(clips, target_dur)
                return self._build_plan(
                    narrative, {"passed": True}, params, user_command
                )
            return self._empty_plan(params, user_command)

        # --- Stage B: Narrative ---
        sys_b, usr_b = build_narrative_prompt(
            selected_clips, target_dur, style=params["style"]
        )
        resp_b = self.llm.generate(sys_b, usr_b)
        narrative = parse_narrative(resp_b)

        if narrative is None:
            if self.fallback_enabled:
                logger.warning("Stage B failed — using fallback ordering")
                narrative = fallback_full(clips, target_dur)
            else:
                return self._empty_plan(params, user_command)

        # --- Stage C: Validation ---
        daily_summary = score_matrix.get("daily_summary", {})
        sys_c, usr_c = build_validation_prompt(narrative, daily_summary)
        resp_c = self.llm.generate(sys_c, usr_c)
        validation = parse_validation(resp_c)

        # Apply corrections if validation failed
        if not validation.get("passed", True) and "corrected_timeline" in validation:
            narrative["timeline"] = validation["corrected_timeline"]

        plan = self._build_plan(narrative, validation, params, user_command)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(
                json.dumps(plan, ensure_ascii=False, indent=2)
            )
            logger.info("Edit plan written to %s", output_path)

        return plan

    # ------------------------------------------------------------------

    def _build_plan(
        self,
        narrative: dict,
        validation: dict,
        params: dict,
        user_command: str | None,
    ) -> dict:
        timeline = narrative.get("timeline", [])
        actual_dur = sum(c.get("duration", 0) for c in timeline)

        return {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "user_command": user_command or "",
            "parameters": {
                "date": params.get("date", ""),
                "target_duration": params.get("target", "auto"),
                "actual_duration": round(actual_dur, 1),
                "style": params.get("style", "story_arc"),
                "focus": params.get("focus", "all"),
                "format": params.get("format", "horizontal"),
                "llm_provider": self.provider,
                "llm_model": self.model,
            },
            "timeline": timeline,
            "validation": validation,
        }

    def _empty_plan(self, params: dict, user_command: str | None) -> dict:
        return self._build_plan(
            {"timeline": []},
            {"passed": True, "checks": {}},
            params,
            user_command,
        )
