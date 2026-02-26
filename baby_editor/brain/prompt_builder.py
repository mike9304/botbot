"""Build LLM prompts for each editing stage (Selection, Narrative, Validation)."""

from __future__ import annotations

import json

# ---------------------------------------------------------------------------
# Stage A: Selection
# ---------------------------------------------------------------------------

SELECTION_SYSTEM = """\
You are a baby video editor AI. You receive analyzed segment data with scores \
and tags, and select the best clips for a highlight reel. Output ONLY valid JSON. \
No markdown, no explanation outside JSON.

Rules:
1. NEVER remove segments tagged "first_word", "first_steps", "first_crawl", \
"first_stand" — these are ALWAYS included regardless of score.
2. When multiple segments show similar content (same video, same activity, close \
timestamps, similar scores): pick ONLY the highest-scored one. Explain why you \
dropped the others in dropped_reasoning.
3. Do not select more than 3 consecutive segments from the same video file (variety).
4. Total selected duration must be within ±15% of target_duration.
5. If a segment has negative delete scores BUT high positive scores (e.g., \
shake=-5 but laugh=+8): keep it if the positive value outweighs the negative. \
Laughter and milestones outweigh camera shake.
6. For twins: ensure neither twin appears in less than 35% of total face time. \
Flag imbalance."""

SELECTION_OUTPUT_FORMAT = """\
{"selection":{"target_duration":60,"actual_duration":58,"clips":[\
{"id":"...","video":"...","start":0.0,"end":8.0,"duration":8.0,"score":19,\
"priority":"highlight","tags":[...],"select_reason":"..."}],\
"dropped_reasoning":[{"id":"...","reason":"..."}],\
"twin_balance":{"baby_a_pct":52,"baby_b_pct":48,"balanced":true}}}"""

# ---------------------------------------------------------------------------
# Stage B: Narrative
# ---------------------------------------------------------------------------

NARRATIVE_SYSTEM = """\
You are a baby video narrative director. You arrange selected clips into a \
compelling sequence. Output ONLY valid JSON.

Narrative principles:
1. OPENING: Start with an engaging moment — smile, eye contact, or energetic scene. \
Never start quiet/static.
2. RISING ACTION (first 30%): Energetic, playful moments. Build interest.
3. CLIMAX (60-70% mark): Place highest-scoring clips here. Emotional peak.
4. FALLING ACTION: Transition to calmer moments — feeding, cuddling, quiet play.
5. CLOSING: End with warmth — sleeping, hugging, peaceful scene. If unavailable, \
use second-best smile.
6. VARIETY: Never 2 clips from same location back-to-back. Alternate activities.
7. PACING: Alternate short energetic clips (3-5s) with longer calm clips (6-10s).
8. TRANSITIONS:
   - Same scene, quick: "cut" (0s)
   - Different location, same mood: "dissolve" (0.5s)
   - Major scene change: "fade_black" (0.8s)
   - Different day: "fade_black" (1.2s)
9. TWIN RULE: Never show same twin for 3+ consecutive clips without the other."""

NARRATIVE_OUTPUT_FORMAT = """\
{"narrative":{"title":"...","total_duration":58.5,"timeline":[\
{"position":1,"role":"opening","clip_id":"...","video":"...","in":14.0,\
"out":20.0,"duration":6.0,"transition_in":{"type":"fade_black","duration":0.5},\
"transition_out":{"type":"dissolve","duration":0.5},"placement_reason":"..."}]}}"""

# ---------------------------------------------------------------------------
# Stage C: Validation
# ---------------------------------------------------------------------------

VALIDATION_SYSTEM = """\
You are a quality control editor reviewing a baby video edit plan. Check for \
problems and fix them. Output ONLY valid JSON.

Checks:
1. DURATION: Within ±15% of target? If over → suggest clip to shorten/remove. \
If under → suggest clip to recover.
2. PACING: 3+ short clips (<3s) in a row? → Fix. 2+ long clips (>10s) in a row? → Fix.
3. VARIETY: 2+ clips from same location back-to-back? → Reorder.
4. DUPLICATES: Overlapping timestamps in same video? → Remove lower score.
5. MOOD: Starts low energy? → Swap opening. Ends abruptly high? → Swap closing.
6. TWIN BALANCE: Either twin <35% face time? → Suggest swaps.
7. OPENING: Has face? CLOSING: Feels conclusive?

If ALL checks pass → output plan unchanged with "passed": true.
If any fail → output corrected plan with explanations."""

VALIDATION_OUTPUT_FORMAT = """\
{"validation":{"passed":true,"checks":{"duration":"OK","pacing":"OK",\
"variety":"OK","duplicates":"OK","mood_arc":"OK","twin_balance":"OK",\
"opening":"OK","closing":"OK"}}}"""


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_selection_prompt(
    clips: list[dict],
    target_duration: float,
    style: str = "story_arc",
    focus: str = "all",
    date: str = "",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Stage A."""
    condensed = json.dumps(clips, ensure_ascii=False)
    user = (
        f"Target: {target_duration}s | Style: {style} | Focus: {focus} | Date: {date}\n"
        f"Clips: {condensed}\n\n"
        f"Output format:\n{SELECTION_OUTPUT_FORMAT}"
    )
    return SELECTION_SYSTEM, user


def build_narrative_prompt(
    selected_clips: list[dict],
    target_duration: float,
    style: str = "story_arc",
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Stage B."""
    selected_json = json.dumps(selected_clips, ensure_ascii=False)
    user = (
        f"Selected clips: {selected_json}\n"
        f"Target: {target_duration}s | Style: {style}\n\n"
        f"Output format:\n{NARRATIVE_OUTPUT_FORMAT}"
    )
    return NARRATIVE_SYSTEM, user


def build_validation_prompt(
    narrative: dict,
    daily_summary: dict,
) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for Stage C."""
    user = (
        f"Plan: {json.dumps(narrative, ensure_ascii=False)}\n"
        f"Daily summary: {json.dumps(daily_summary, ensure_ascii=False)}\n\n"
        f"Output format:\n{VALIDATION_OUTPUT_FORMAT}"
    )
    return VALIDATION_SYSTEM, user
