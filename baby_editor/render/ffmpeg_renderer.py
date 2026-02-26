"""FFmpeg-based video renderer.

Two-pass approach:
  Pass 1: Extract each clip as a separate temp file (with handles)
  Pass 2: Concatenate all clips with transitions into final output

GPU acceleration via NVENC on RTX 5080.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile

logger = logging.getLogger("baby_editor.render")


class FFmpegRenderer:
    def __init__(self, config: dict):
        render_cfg = config.get("render", {})
        vid = render_cfg.get("video", {})
        aud = render_cfg.get("audio", {})

        self.codec = vid.get("codec", "h264_nvenc")
        self.preset = vid.get("preset", "p4")
        self.cq = vid.get("cq", 23)
        self.audio_codec = aud.get("codec", "aac")
        self.audio_bitrate = aud.get("bitrate", "192k")
        self.fade_dur = aud.get("fade_duration", 0.3)
        self.normalize_lufs = aud.get("normalize_target_lufs", -18)
        self.temp_dir = tempfile.mkdtemp(prefix="baby_edit_")

    def render(self, timeline: list[dict], output_path: str) -> str:
        """Render the full rough cut from a timeline.

        Returns the path to the output file.
        """
        if not timeline:
            raise ValueError("Empty timeline — nothing to render")

        # Pass 1: extract individual clips
        clip_paths: list[str] = []
        for i, clip in enumerate(timeline):
            clip_path = self._extract_clip(clip, i)
            clip_paths.append(clip_path)

        # Pass 2: concatenate with transitions
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        self._concatenate(clip_paths, timeline, output_path)

        return output_path

    def cleanup(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # ------------------------------------------------------------------

    def _extract_clip(self, clip: dict, index: int) -> str:
        output = os.path.join(self.temp_dir, f"clip_{index:03d}.mp4")
        in_point = clip.get("in_point", clip.get("in", clip.get("start", 0)))
        duration = clip["duration"]
        source = clip.get("source_path", clip.get("video", ""))
        fade_out_start = max(0, duration - self.fade_dur)

        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-ss", str(in_point),
            "-i", source,
            "-t", str(duration),
            "-af",
            (
                f"afade=t=in:d={self.fade_dur},"
                f"afade=t=out:st={fade_out_start}:d={self.fade_dur},"
                f"loudnorm=I={self.normalize_lufs}:LRA=7:TP=-1"
            ),
            "-c:v", self.codec,
            "-preset", self.preset,
            "-cq", str(self.cq),
            "-c:a", self.audio_codec, "-b:a", self.audio_bitrate,
            "-movflags", "+faststart",
            output,
        ]

        logger.info(
            "Extracting clip %d: %s [%.1f–%.1f]",
            index, os.path.basename(source), in_point, in_point + duration,
        )
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logger.error("FFmpeg clip extraction failed: %s", result.stderr[:500])
            raise RuntimeError(f"Clip extraction failed for {source}")

        return output

    def _concatenate(
        self,
        clip_paths: list[str],
        timeline: list[dict],
        output_path: str,
    ) -> None:
        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], output_path)
            return

        # Try complex filter with xfade transitions
        try:
            self._concat_xfade(clip_paths, timeline, output_path)
        except Exception:
            logger.warning("xfade concat failed — falling back to simple concat")
            self._concat_simple(clip_paths, output_path)

    def _concat_xfade(
        self,
        clip_paths: list[str],
        timeline: list[dict],
        output_path: str,
    ) -> None:
        n = len(clip_paths)
        inputs = []
        for p in clip_paths:
            inputs.extend(["-i", p])

        # Build xfade filter chain
        filter_parts: list[str] = []
        current_label = "[0:v]"
        offset = 0.0

        for i in range(1, n):
            trans = timeline[i].get("transition_in", {})
            trans_type = trans.get("type", "cut")
            trans_dur = trans.get("duration", 0)
            prev_dur = self._probe_duration(clip_paths[i - 1])
            offset += prev_dur - trans_dur

            next_label = f"[v{i}]"
            if trans_type == "dissolve" and trans_dur > 0:
                filter_parts.append(
                    f"{current_label}[{i}:v]xfade=transition=dissolve"
                    f":duration={trans_dur}:offset={offset}{next_label}"
                )
            elif trans_type == "fade_black" and trans_dur > 0:
                filter_parts.append(
                    f"{current_label}[{i}:v]xfade=transition=fadeblack"
                    f":duration={trans_dur}:offset={offset}{next_label}"
                )
            else:
                filter_parts.append(
                    f"{current_label}[{i}:v]concat=n=2:v=1:a=0{next_label}"
                )
            current_label = next_label

        # Audio: simple concat
        audio_labels = "".join(f"[{i}:a]" for i in range(n))
        audio_filter = f"{audio_labels}concat=n={n}:v=0:a=1[aout]"

        full_filter = ";".join(filter_parts) + f";{audio_filter}"

        cmd = [
            "ffmpeg", "-y",
            *inputs,
            "-filter_complex", full_filter,
            "-map", current_label,
            "-map", "[aout]",
            "-c:v", self.codec, "-preset", self.preset, "-cq", str(self.cq),
            "-c:a", self.audio_codec, "-b:a", self.audio_bitrate,
            output_path,
        ]

        logger.info("Concatenating %d clips with xfade transitions", n)
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(result.stderr[:500])

    def _concat_simple(self, clip_paths: list[str], output_path: str) -> None:
        list_file = os.path.join(self.temp_dir, "concat_list.txt")
        with open(list_file, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-c:v", self.codec, "-preset", self.preset, "-cq", str(self.cq),
            "-c:a", self.audio_codec, "-b:a", self.audio_bitrate,
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Simple concat failed: {result.stderr[:500]}")

    def _probe_duration(self, path: str) -> float:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0", path,
            ],
            capture_output=True, text=True,
        )
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 5.0  # safe default
