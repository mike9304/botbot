"""Range-day mean-reversion overlay backtest (paper research only).

Motivation: on 2026-08-19 the official Strategy A v1.1 and the V5 paper loop
both produced zero trades because ADX stayed below threshold and no Donchian
band broke. This engine tests the opposite trade: when ADX(14) < ``adx_max``
(no trend), fade z-score extremes back to the mean.

Fill model matches ``engine.py``: signal on bar close, fill at next bar open,
0.20% round-trip cost, isolated, one position, effective leverage capped at 3x,
ATR stop with gap-through fills at the open. Exits: z-score returns to 0
(signal on close, fill next open), ATR stop, or a time stop.

No live orders, no API keys.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .indicators import adx_wilder, align_daily_ema_regime, atr_wilder


@dataclass(frozen=True)
class MRVariant:
    id: str
    name: str
    timeframe: str
    z_period: int = 20
    z_entry: float = 2.0
    adx_max: float = 20.0
    risk_pct: float = 0.005
    atr_stop_mult: float = 1.5
    max_hold_bars: int = 20
    regime_gate: bool = False  # True: only fade in the daily EMA50/200 direction
    atr_period: int = 14
    adx_period: int = 14
    max_leverage: float = 3.0
    rt_cost: float = 0.002


@dataclass
class MRTrade:
    side: int
    entry_ts: float
    exit_ts: float
    entry: float
    exit: float
    pnl: float
    reason: str


@dataclass
class MRResult:
    variant: MRVariant
    source: str
    window: str
    ret_pct: float
    max_dd_pct: float
    trades: int
    wins: int
    win_rate: float
    profit_factor: float
    end_equity: float
    trade_log: list[MRTrade] = field(default_factory=list)


def zscore(close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(period - 1, n):
        w = close[i - period + 1 : i + 1]
        sd = float(np.std(w))
        if sd > 0:
            out[i] = (close[i] - float(np.mean(w))) / sd
    return out


def run_mr_backtest(
    bars: dict[str, np.ndarray],
    daily: dict[str, np.ndarray],
    variant: MRVariant,
    source: str,
    window: str,
    window_start_ms: float,
    window_end_ms: float,
    start_equity: float = 10_000.0,
) -> MRResult:
    ts = bars["ts"]
    o = bars["open"]
    h = bars["high"]
    l = bars["low"]
    c = bars["close"]
    n = len(c)

    z = zscore(c, variant.z_period)
    atr = atr_wilder(h, l, c, variant.atr_period)
    adx = adx_wilder(h, l, c, variant.adx_period)
    regime = align_daily_ema_regime(ts, daily["ts"], daily["close"])

    equity = start_equity
    peak = start_equity
    max_dd = 0.0
    realized = 0.0
    trades: list[MRTrade] = []

    pos_side = 0
    pos_qty = 0.0
    pos_entry = 0.0
    pos_entry_ts = 0.0
    pos_notional = 0.0
    pos_stop = 0.0
    pos_bars = 0
    pending_side = 0
    pending_atr = 0.0
    pending_exit = False

    def close_position(fill: float, fill_ts: float, reason: str) -> None:
        nonlocal realized, equity, pos_side, pos_qty, pos_entry, pos_notional
        nonlocal pos_stop, pos_entry_ts, pos_bars, peak, max_dd
        if pos_side == 0:
            return
        gross = pos_qty * pos_side * (fill - pos_entry)
        pnl = gross - pos_notional * variant.rt_cost
        realized += pnl
        equity = start_equity + realized
        trades.append(MRTrade(pos_side, pos_entry_ts, fill_ts, pos_entry, fill, pnl, reason))
        pos_side = 0
        pos_qty = 0.0
        pos_entry = 0.0
        pos_notional = 0.0
        pos_stop = 0.0
        pos_entry_ts = 0.0
        pos_bars = 0
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)

    def open_position(side: int, fill: float, fill_ts: float, stop_atr: float) -> None:
        nonlocal pos_side, pos_qty, pos_entry, pos_entry_ts, pos_notional, pos_stop, pos_bars
        if side == 0 or stop_atr <= 0:
            return
        stop_dist = variant.atr_stop_mult * stop_atr
        qty = equity * variant.risk_pct / stop_dist
        notional = qty * fill
        max_notional = equity * variant.max_leverage
        if notional > max_notional:
            qty = max_notional / fill
            notional = max_notional
        if qty <= 0:
            return
        pos_side = side
        pos_qty = qty
        pos_entry = fill
        pos_entry_ts = fill_ts
        pos_notional = notional
        pos_stop = fill - side * stop_dist
        pos_bars = 0

    in_window = (ts >= window_start_ms) & (ts <= window_end_ms)

    for i in range(n - 1):
        # Fill pending exit (z returned to mean / time stop) at this open.
        if pending_exit and pos_side != 0:
            close_position(o[i], ts[i], "mean_or_time_next_open")
        pending_exit = False

        # Fill pending entry at this open.
        if pending_side != 0 and ts[i] >= window_start_ms:
            if pos_side == 0:
                open_position(pending_side, o[i], ts[i], pending_atr)
        pending_side = 0
        pending_atr = 0.0

        if pos_side != 0:
            pos_bars += 1
            liq_move = (1.0 / variant.max_leverage) - 0.005
            liq_price = pos_entry * (1.0 - pos_side * liq_move)
            hit_liq = (pos_side > 0 and l[i] <= liq_price) or (pos_side < 0 and h[i] >= liq_price)
            if hit_liq:
                fill = o[i] if (pos_side > 0 and o[i] <= liq_price) or (
                    pos_side < 0 and o[i] >= liq_price
                ) else liq_price
                close_position(fill, ts[i], "isolated_liq")
            elif pos_side > 0:
                if o[i] <= pos_stop:
                    close_position(o[i], ts[i], "stop_gap")
                elif l[i] <= pos_stop:
                    close_position(pos_stop, ts[i], "stop")
            else:
                if o[i] >= pos_stop:
                    close_position(o[i], ts[i], "stop_gap")
                elif h[i] >= pos_stop:
                    close_position(pos_stop, ts[i], "stop")

        eq = start_equity + realized + (pos_qty * pos_side * (c[i] - pos_entry) if pos_side else 0.0)
        if ts[i] >= window_start_ms:
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak if peak > 0 else 0.0)

        # Exit signal on close: z back through 0, or time stop.
        if pos_side != 0 and not np.isnan(z[i]):
            back_to_mean = (pos_side > 0 and z[i] >= 0.0) or (pos_side < 0 and z[i] <= 0.0)
            if back_to_mean or pos_bars >= variant.max_hold_bars:
                pending_exit = True
            continue

        # Entry signal on close (flat only, inside window, range regime only).
        if ts[i] < window_start_ms or ts[i] > window_end_ms:
            continue
        if np.isnan(z[i]) or np.isnan(atr[i]) or atr[i] <= 0 or np.isnan(adx[i]):
            continue
        if adx[i] >= variant.adx_max:
            continue
        want = 0
        if z[i] <= -variant.z_entry:
            want = 1
        elif z[i] >= variant.z_entry:
            want = -1
        if want == 0:
            continue
        if variant.regime_gate and regime[i] != want:
            continue
        pending_side = want
        pending_atr = atr[i]

    if pos_side != 0:
        close_position(c[n - 1], ts[n - 1], "eod_flatten")

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    wp = sum(t.pnl for t in wins)
    lp = abs(sum(t.pnl for t in losses))
    pf = wp / lp if lp > 0 else (999.0 if wp > 0 else 0.0)
    end_equity = start_equity + realized
    return MRResult(
        variant=variant,
        source=source,
        window=window,
        ret_pct=(end_equity / start_equity - 1.0) * 100.0,
        max_dd_pct=max_dd * 100.0,
        trades=len(trades),
        wins=len(wins),
        win_rate=(len(wins) / len(trades) * 100.0) if trades else 0.0,
        profit_factor=pf,
        end_equity=end_equity,
        trade_log=trades,
    )
