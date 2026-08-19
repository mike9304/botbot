#!/usr/bin/env python3
"""Follow-up paper study after the 2026-08-19 zero-trade range day.

Three NEW tests (not in results/study.json):
1. ADX threshold sweep 20/22/24/25 on the official A v1.1 config
   (today's 4H ADX was ~24.33, just under the official >25 gate).
2. Round-trip cost sensitivity 0.10%-0.40% for A11 and V5.
3. Range-day mean-reversion overlay: fade z-score extremes when ADX(14)<20
   (exactly the bars where A/V5 cannot trade).

Research / paper only. No orders, no API keys.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

from research.strategy_a_aggressive.engine import VARIANTS, Variant, run_backtest
from research.strategy_a_aggressive.engine_mr import MRVariant, run_mr_backtest
from research.strategy_a_aggressive.fetch_ohlcv import to_columns
from research.strategy_a_aggressive.run_study import WINDOWS, fetch_or_load, parse_ms

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

A11 = VARIANTS[0]
V5 = [v for v in VARIANTS if v.id == "V5"][0]

ADX_SWEEP: list[Variant] = [
    replace(A11, id=f"A11_ADX{int(th)}", name=f"A v1.1 with ADX>{int(th)}", adx_min=th)
    for th in (20.0, 22.0, 24.0, 25.0)
]

COST_GRID = [0.0010, 0.0015, 0.0020, 0.0030, 0.0040]
COST_SWEEP: list[Variant] = [
    replace(base, id=f"{base.id}_RT{int(cost * 10000)}bp", rt_cost=cost)
    for base in (A11, V5)
    for cost in COST_GRID
]

MR_VARIANTS: list[MRVariant] = [
    MRVariant(
        id="MR2H",
        name="2H z20 fade +-2.0, ADX<20, both sides, R0.5%, ATR1.5, hold<=20",
        timeframe="2H",
    ),
    MRVariant(
        id="MR2H_RG",
        name="2H z20 fade, ADX<20, daily-regime direction only",
        timeframe="2H",
        regime_gate=True,
    ),
    MRVariant(
        id="MR4H",
        name="4H z20 fade +-2.0, ADX<20, both sides",
        timeframe="4H",
    ),
]


def trend_row(res) -> dict:
    return {
        "variant_id": res.variant.id,
        "kind": "trend",
        "window": res.window,
        "ret_pct": round(res.ret_pct, 4),
        "win_rate": round(res.win_rate, 2),
        "trades": res.trades,
        "max_dd_pct": round(res.max_dd_pct, 4),
        "profit_factor": None if res.profit_factor >= 999 else round(res.profit_factor, 3),
        "source": res.source,
        "spec": asdict(res.variant),
    }


def mr_row(res) -> dict:
    return {
        "variant_id": res.variant.id,
        "kind": "mean_reversion",
        "window": res.window,
        "ret_pct": round(res.ret_pct, 4),
        "win_rate": round(res.win_rate, 2),
        "trades": res.trades,
        "max_dd_pct": round(res.max_dd_pct, 4),
        "profit_factor": None if res.profit_factor >= 999 else round(res.profit_factor, 3),
        "source": res.source,
        "spec": asdict(res.variant),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Range-day follow-up paper study")
    parser.add_argument("--force-fetch", action="store_true")
    parser.add_argument("--start", default="2019-06-01")
    args = parser.parse_args()

    start_ms = parse_ms(args.start)
    end_ms = parse_ms("2026-08-20")

    needed = {"1D", "4H", "2H"}
    series = {}
    fetch_errors: dict[str, str] = {}
    for tf in sorted(needed):
        try:
            series[tf] = fetch_or_load(tf, start_ms, end_ms, args.force_fetch)
            arr, src = series[tf]
            print(f"  {tf}: {len(arr)} bars source={src}", flush=True)
        except Exception as exc:  # data failure must be reported, never invented
            fetch_errors[tf] = str(exc)
            print(f"  {tf}: FETCH FAILED — {exc}", flush=True)

    if "1D" not in series or "4H" not in series or "2H" not in series:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "followup_study.json").write_text(
            json.dumps({"ok": False, "reason": "OHLCV fetch failed", "errors": fetch_errors}, indent=2)
        )
        print("FATAL: missing data, wrote failure marker.", flush=True)
        return 2

    daily = to_columns(series["1D"][0])
    rows: list[dict] = []

    print("\n== 1) ADX threshold sweep (official A v1.1 otherwise) ==", flush=True)
    for variant in ADX_SWEEP:
        arr, src = series[variant.timeframe]
        bars = to_columns(arr)
        for window, (w0, w1) in WINDOWS.items():
            res = run_backtest(
                bars, daily, variant, src, window,
                float(parse_ms(w0)), float(parse_ms(w1)),
            )
            rows.append(trend_row(res))
            print(
                f"{variant.id:12} {window:10} ret={res.ret_pct:+7.2f}% WR={res.win_rate:5.1f}% "
                f"n={res.trades:4} MDD={res.max_dd_pct:5.2f}%",
                flush=True,
            )

    print("\n== 2) Round-trip cost sensitivity ==", flush=True)
    for variant in COST_SWEEP:
        arr, src = series[variant.timeframe]
        bars = to_columns(arr)
        for window, (w0, w1) in WINDOWS.items():
            res = run_backtest(
                bars, daily, variant, src, window,
                float(parse_ms(w0)), float(parse_ms(w1)),
            )
            rows.append(trend_row(res))
            print(
                f"{variant.id:14} {window:10} ret={res.ret_pct:+7.2f}% WR={res.win_rate:5.1f}% "
                f"n={res.trades:4} MDD={res.max_dd_pct:5.2f}%",
                flush=True,
            )

    print("\n== 3) Range-day mean-reversion overlay (ADX<20 bars only) ==", flush=True)
    for variant in MR_VARIANTS:
        arr, src = series[variant.timeframe]
        bars = to_columns(arr)
        for window, (w0, w1) in WINDOWS.items():
            res = run_mr_backtest(
                bars, daily, variant, src, window,
                float(parse_ms(w0)), float(parse_ms(w1)),
            )
            rows.append(mr_row(res))
            print(
                f"{variant.id:9} {window:10} ret={res.ret_pct:+7.2f}% WR={res.win_rate:5.1f}% "
                f"n={res.trades:4} MDD={res.max_dd_pct:5.2f}% PF={res.profit_factor:5.2f}",
                flush=True,
            )

    payload = {
        "ok": True,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Paper research only. Next-open fill, isolated, lev<=3, one position. "
            "Trend rows use rt_cost per spec (default 0.20% RT); MR rows 0.20% RT."
        ),
        "windows": WINDOWS,
        "fetch_errors": fetch_errors,
        "results": rows,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "followup_study.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
