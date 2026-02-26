"""Telegram notification bot for baby_editor pipeline completion."""

from __future__ import annotations

import logging
import os
import subprocess

import httpx

logger = logging.getLogger("baby_editor.telegram")


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str):
        self.token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def notify_complete(self, date: str, results: dict) -> None:
        """Send completion notification with optional thumbnail."""
        rough_cut = results.get("rough_cut", "")
        thumb_path = rough_cut.replace(".mp4", "_thumb.jpg") if rough_cut else ""

        # Generate thumbnail
        if rough_cut and os.path.exists(rough_cut):
            try:
                subprocess.run(
                    [
                        "ffmpeg", "-y", "-i", rough_cut,
                        "-ss", "2", "-vframes", "1",
                        "-vf", "scale=480:-1",
                        thumb_path,
                    ],
                    capture_output=True, check=True,
                )
            except Exception:
                thumb_path = ""

        duration = results.get("duration", 0)
        clip_count = results.get("clip_count", 0)

        message = (
            f"Baby Edit Complete — {date}\n"
            f"Duration: {duration:.0f}s\n"
            f"Clips: {clip_count}"
        )

        try:
            if thumb_path and os.path.exists(thumb_path):
                with open(thumb_path, "rb") as f:
                    httpx.post(
                        f"{self.base_url}/sendPhoto",
                        data={"chat_id": self.chat_id, "caption": message},
                        files={"photo": f},
                        timeout=30,
                    )
            else:
                httpx.post(
                    f"{self.base_url}/sendMessage",
                    json={"chat_id": self.chat_id, "text": message},
                    timeout=30,
                )
        except Exception:
            logger.exception("Failed to send Telegram notification")
