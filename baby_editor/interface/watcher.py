"""Watch a folder for new video files and automatically trigger the pipeline.

Uses the watchdog library.  Waits for a configurable debounce period after the
last file creation event before starting the pipeline (batch upload detection).
"""

from __future__ import annotations

import logging
import os
import time

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = logging.getLogger("baby_editor.watcher")

_VIDEO_EXTENSIONS = (".mp4", ".mov")


class _VideoFolderHandler(FileSystemEventHandler):
    def __init__(self, debounce_seconds: int = 30):
        self.debounce = debounce_seconds
        self.last_event_time: float = 0
        self.pending_dates: set[str] = set()

    def on_created(self, event) -> None:  # type: ignore[override]
        if event.is_directory:
            return
        if event.src_path.lower().endswith(_VIDEO_EXTENSIONS):
            date_folder = os.path.basename(os.path.dirname(event.src_path))
            self.pending_dates.add(date_folder)
            self.last_event_time = time.time()
            logger.info("New video detected: %s", event.src_path)


def run_watcher(
    video_base: str = "/videos",
    debounce: int = 30,
    output_base: str = "/output",
) -> None:
    """Start the folder watcher (blocking)."""
    handler = _VideoFolderHandler(debounce)
    observer = Observer()
    observer.schedule(handler, video_base, recursive=True)
    observer.start()

    logger.info("Watching %s for new videos...", video_base)

    try:
        while True:
            time.sleep(5)
            if (
                handler.pending_dates
                and (time.time() - handler.last_event_time > debounce)
            ):
                dates = handler.pending_dates.copy()
                handler.pending_dates.clear()
                for date_str in dates:
                    logger.info("Triggering pipeline for %s", date_str)
                    from baby_editor.pipeline import run_full_pipeline

                    try:
                        run_full_pipeline(
                            os.path.join(video_base, date_str),
                            os.path.join(output_base, date_str),
                            "오늘 러프컷 만들어줘",
                            {},
                        )
                    except Exception:
                        logger.exception("Pipeline failed for %s", date_str)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
