"""Command-line interface for baby_editor.

Supports modes: edit, analyze, auto, mcp, watch.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import yaml

logger = logging.getLogger("baby_editor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Baby Video Auto-Editor")
    parser.add_argument("--input", "-i", help="Input video folder path")
    parser.add_argument("--output", "-o", help="Output folder path")
    parser.add_argument(
        "--date", "-d", default="today",
        help="Date: today, yesterday, YYYY-MM-DD",
    )
    parser.add_argument(
        "--duration", default="auto",
        help="Target duration: auto, 60, 1분, 30초",
    )
    parser.add_argument(
        "--focus", default="all",
        help="Focus: all, laughing, playing, bathing, twins_together",
    )
    parser.add_argument(
        "--style", default="story_arc",
        help="Style: story_arc, chronological, highlight_best_first",
    )
    parser.add_argument(
        "--mode", default="edit",
        choices=["edit", "analyze", "auto", "mcp", "watch"],
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Config file path",
    )
    return parser


def resolve_date(date_str: str) -> str:
    """Convert date alias to YYYY-MM-DD string."""
    today = datetime.now()
    if date_str == "today":
        return today.strftime("%Y-%m-%d")
    if date_str == "yesterday":
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    return date_str


def cli_main() -> None:
    """Entry point for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler("baby_editor.log"),
            logging.StreamHandler(),
        ],
    )

    parser = build_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text())
    else:
        config = {}

    date_str = resolve_date(args.date)
    interface_cfg = config.get("interface", {})
    video_base = interface_cfg.get("video_base_path", "/videos")
    output_base = interface_cfg.get("output_base_path", "/output")

    input_dir = args.input or os.path.join(video_base, date_str)
    output_dir = args.output or os.path.join(output_base, date_str)
    os.makedirs(output_dir, exist_ok=True)

    # --- MCP server mode ---
    if args.mode == "mcp":
        from baby_editor.interface.mcp_server import run_mcp_server
        run_mcp_server(config)
        return

    # --- Folder watcher mode ---
    if args.mode == "watch":
        from baby_editor.interface.watcher import run_watcher
        run_watcher(video_base)
        return

    user_command = f"{args.date} {args.focus} {args.duration} {args.style}"

    # --- Analyze only ---
    if args.mode == "analyze":
        from baby_editor.analyze import AnalysisPipeline

        pipeline = AnalysisPipeline(config)
        score_matrix = pipeline.analyze_folder(input_dir)
        out_path = os.path.join(output_dir, "score_matrix.json")
        Path(out_path).write_text(json.dumps(score_matrix, ensure_ascii=False, indent=2))
        logger.info("Analysis complete: %s", out_path)
        return

    # --- Full pipeline (edit / auto) ---
    from baby_editor.pipeline import run_full_pipeline

    results = run_full_pipeline(input_dir, output_dir, user_command, config)
    logger.info("Pipeline complete: %s", json.dumps(results, indent=2))

    # Telegram notification
    tg = interface_cfg.get("telegram", {})
    if tg.get("enabled") and tg.get("bot_token") and tg.get("chat_id"):
        from baby_editor.interface.telegram_bot import TelegramNotifier
        notifier = TelegramNotifier(tg["bot_token"], tg["chat_id"])
        notifier.notify_complete(date_str, results)
