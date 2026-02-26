"""Pre-LLM processing: merge adjacent candidate segments into clips.

Reduces ~600 segments to ~40-60 clips before sending to the LLM.
No AI involved — pure Python logic.
"""

from __future__ import annotations


def merge_segments(
    segments: list[dict],
    baseline: int = 3,
    handle: float = 0.5,
    min_dur: float = 1.5,
    max_dur: float = 15.0,
) -> list[dict]:
    """Merge adjacent candidate segments into continuous clips.

    Algorithm:
      1. Filter segments with score >= baseline
      2. Sort by (video, start)
      3. Merge contiguous segments from the same video
      4. Add handles (±0.5 s), trim long clips to peak ±5 s
      5. Drop clips < min_dur
    """
    candidates = [s for s in segments if s["total_score"] >= baseline]
    sorted_segs = sorted(candidates, key=lambda s: (s["video"], s["start"]))

    clips: list[dict] = []
    current: dict | None = None

    for seg in sorted_segs:
        if (
            current
            and seg["video"] == current["video"]
            and seg["start"] == current["_raw_end"]
        ):
            current["_segments"].append(seg)
            current["_raw_end"] = seg["end"]
        else:
            if current:
                clips.append(_finalize_clip(current, handle, max_dur))
            current = {
                "video": seg["video"],
                "_raw_start": seg["start"],
                "_raw_end": seg["end"],
                "_segments": [seg],
            }

    if current:
        clips.append(_finalize_clip(current, handle, max_dur))

    return [c for c in clips if c["duration"] >= min_dur]


def _finalize_clip(current: dict, handle: float, max_dur: float) -> dict:
    segs = current["_segments"]
    scores = [s["total_score"] for s in segs]
    all_tags: set[str] = set()
    for s in segs:
        all_tags.update(s.get("tags", []))

    start = max(0, current["_raw_start"] - handle)
    end = current["_raw_end"] + handle
    duration = end - start

    # Trim long clips to peak ± window
    if duration > max_dur:
        peak_idx = scores.index(max(scores))
        peak_seg = segs[peak_idx]
        start = max(0, peak_seg["start"] - 3.0)
        end = peak_seg["end"] + 5.0
        duration = end - start

    # Find best transcript
    best_transcript = ""
    for s in segs:
        t = s.get("metrics", {}).get("transcript", "")
        if len(t) > len(best_transcript):
            best_transcript = t

    return {
        "id": f"{current['video'].replace('.mp4', '').replace('.mov', '')}_seg{segs[0]['index']}",
        "video": current["video"],
        "start": round(start, 1),
        "end": round(end, 1),
        "duration": round(duration, 1),
        "score": round(sum(scores) / len(scores), 1),
        "peak_score": max(scores),
        "tags": sorted(all_tags),
        "transcript": best_transcript[:80],
        "face_count": max(
            s.get("metrics", {}).get("face_count", 0) for s in segs
        ),
        "laugh_conf": max(
            s.get("metrics", {}).get("laugh_confidence", 0) for s in segs
        ),
    }
