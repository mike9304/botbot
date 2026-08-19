#!/usr/bin/env python3
"""Fetch public BTCUSDT OHLCV and backtest Strategy A v1.1 + 5 aggressive variants.

Research / paper only. Does not place orders and does not use exchange API keys.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from research.strategy_a_aggressive.engine import VARIANTS, run_backtest
from research.strategy_a_aggressive.fetch_ohlcv import (
    fetch_binance_vision_spot,
    fetch_bitget_futures,
    load_csv,
    save_csv,
    to_columns,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

WINDOWS = {
    "last_12m": ("2025-08-19", "2026-08-19"),
    "from_2022": ("2022-01-01", "2026-08-19"),
    "full_2020": ("2020-01-01", "2026-08-19"),
}


def parse_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def iso(ts_ms: float) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def fetch_or_load(tf: str, start_ms: int, end_ms: int, force: bool) -> tuple[np.ndarray, str]:
    path = DATA / f"BTCUSDT_{tf}_bitget.csv"
    if path.exists() and not force:
        arr = load_csv(path)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr, "bitget_usdt_futures_cache"
    print(f"Fetching Bitget USDT-FUTURES {tf} …", flush=True)
    try:
        arr = fetch_bitget_futures("BTCUSDT", tf, start_ms, end_ms)
        if len(arr) >= 200:
            save_csv(path, arr)
            return arr, "bitget_usdt_futures"
        print(f"  Bitget {tf} too short ({len(arr)} bars), trying Binance Vision spot …", flush=True)
    except Exception as exc:
        print(f"  Bitget {tf} failed: {exc}", flush=True)
    path2 = DATA / f"BTCUSDT_{tf}_binance_vision_spot.csv"
    if path2.exists() and not force:
        arr = load_csv(path2)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr, "binance_vision_spot_cache"
    arr = fetch_binance_vision_spot("BTCUSDT", tf, start_ms, end_ms)
    if len(arr) == 0:
        raise RuntimeError(f"Could not fetch {tf} OHLCV from Bitget or Binance Vision")
    save_csv(path2, arr)
    return arr, "binance_vision_spot"


def slim_result(result) -> dict:
    s = result.summary()
    s.pop("trade_log", None)
    s["first_bar_iso"] = iso(result.first_bar) if result.first_bar else None
    s["last_bar_iso"] = iso(result.last_bar) if result.last_bar else None
    s["profit_factor"] = None if result.profit_factor >= 999 else round(result.profit_factor, 3)
    for key in ("ret_pct", "max_dd_pct", "win_rate", "avg_win_pct", "avg_loss_pct"):
        s[key] = round(s[key], 4)
    s["end_equity"] = round(s["end_equity"], 2)
    return s


def main() -> int:
    parser = argparse.ArgumentParser(description="Strategy A aggressive paper study")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--start", default="2019-06-01", help="history start YYYY-MM-DD (warmup)")
    args = parser.parse_args()

    start_ms = parse_ms(args.start)
    end_ms = parse_ms("2026-08-20")

    needed = sorted({v.timeframe for v in VARIANTS} | {"1D"})
    series: dict[str, tuple[np.ndarray, str]] = {}
    fetch_errors: dict[str, str] = {}
    for tf in needed:
        try:
            series[tf] = fetch_or_load(tf, start_ms, end_ms, args.force_fetch)
            arr, src = series[tf]
            print(
                f"  {tf}: {len(arr)} bars  {iso(arr[0, 0])} → {iso(arr[-1, 0])}  source={src}",
                flush=True,
            )
        except Exception as exc:
            fetch_errors[tf] = str(exc)
            print(f"  {tf}: FETCH FAILED — {exc}", flush=True)

    if "1D" not in series:
        print("FATAL: daily candles required for EMA50/200 regime.", flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "study.json").write_text(
            json.dumps({"ok": False, "reason": "daily data missing", "errors": fetch_errors}, indent=2)
        )
        return 2

    daily = to_columns(series["1D"][0])
    rows = []
    for variant in VARIANTS:
        if variant.timeframe not in series:
            for window in WINDOWS:
                rows.append(
                    {
                        "variant_id": variant.id,
                        "window": window,
                        "ok": False,
                        "reason": f"no {variant.timeframe} data: {fetch_errors.get(variant.timeframe)}",
                    }
                )
            continue
        arr, src = series[variant.timeframe]
        bars = to_columns(arr)
        for window, (w0, w1) in WINDOWS.items():
            result = run_backtest(
                bars,
                daily,
                variant,
                source=src,
                window=window,
                window_start_ms=float(parse_ms(w0)),
                window_end_ms=float(parse_ms(w1)),
            )
            slim = slim_result(result)
            slim["ok"] = True
            slim["variant_id"] = variant.id
            rows.append(slim)
            print(
                f"{variant.id:3} {window:10}  ret={result.ret_pct:+7.2f}%  "
                f"WR={result.win_rate:5.1f}%  n={result.trades:4}  "
                f"MDD={result.max_dd_pct:5.2f}%  src={src}",
                flush=True,
            )

    payload = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "Paper research only. Next-open fill, 0.20% RT, isolated, lev<=3, one position.",
        "windows": WINDOWS,
        "variants": [v.__dict__ for v in VARIANTS],
        "fetch_errors": fetch_errors,
        "results": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "study.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
