"""Reframe horizontal video to 9:16 vertical for Instagram Reels / YouTube Shorts.

Sprint 1: simple center crop.
Sprint 2 upgrade: face-following dynamic crop.
"""

from __future__ import annotations

import logging
import subprocess

logger = logging.getLogger("baby_editor.render")


def render_vertical(
    input_path: str,
    output_path: str,
    width: int = 1080,
    height: int = 1920,
    codec: str = "h264_nvenc",
    preset: str = "p4",
    cq: int = 23,
) -> str:
    """Create a 9:16 vertical crop of the input video.

    Returns the output path.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vf", f"crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale={width}:{height}",
        "-c:v", codec, "-preset", preset, "-cq", str(cq),
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]

    logger.info("Rendering vertical reframe: %s → %s", input_path, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Vertical reframe failed: {result.stderr[:500]}")

    return output_path
