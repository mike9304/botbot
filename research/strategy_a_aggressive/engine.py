"""Strategy A v1.1 + aggressive paper variants.

Fill model (matches the official paper spec notes):
- Signal on bar close, fill at next bar open
- 0.20% round-trip cost on entry notional
- Isolated, one position, effective leverage capped at 3x
- Stop: ATR multiple; trail: Donchian opposite band (ratchet only)
- Gap through stop fills at the open
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .indicators import (
    adx_wilder,
    align_daily_ema_regime,
    atr_wilder,
    donchian_prior,
    ema,
    roc,
)


@dataclass(frozen=True)
class Variant:
    id: str
    name: str
    timeframe: str
    donchian: int
    adx_min: Optional[float]
    risk_pct: float
    atr_stop_mult: float
    trail_n: int = 10
    use_ema_regime: bool = True
    momentum_roc: Optional[int] = None
    ema_fast: Optional[int] = None
    atr_period: int = 14
    adx_period: int = 14
    max_leverage: float = 3.0
    rt_cost: float = 0.002
    isolated: bool = True
    one_position: bool = True


# Official baseline + 5 aggressive, still risk-capped variants.
VARIANTS: list[Variant] = [
    Variant(
        id="A11",
        name="Strategy A v1.1 official (validation)",
        timeframe="4H",
        donchian=55,
        adx_min=25.0,
        risk_pct=0.005,
        atr_stop_mult=2.0,
    ),
    Variant(
        id="V1",
        name="Fast Donchian20 + weak ADX + R1% + ATR1.5",
        timeframe="4H",
        donchian=20,
        adx_min=20.0,
        risk_pct=0.01,
        atr_stop_mult=1.5,
    ),
    Variant(
        id="V2",
        name="Donchian20 + no ADX + R1% + ATR1.5",
        timeframe="4H",
        donchian=20,
        adx_min=None,
        risk_pct=0.01,
        atr_stop_mult=1.5,
    ),
    Variant(
        id="V3",
        name="Tight Donchian10 + weak ADX + R1% + ATR1.5",
        timeframe="4H",
        donchian=10,
        adx_min=20.0,
        risk_pct=0.01,
        atr_stop_mult=1.5,
    ),
    Variant(
        id="V4",
        name="1H Donchian20 + weak ADX + R1% + ATR1.5",
        timeframe="1H",
        donchian=20,
        adx_min=20.0,
        risk_pct=0.01,
        atr_stop_mult=1.5,
    ),
    Variant(
        id="V5",
        name="2H Donchian20 + ROC overlay + weak ADX + R1% + ATR1.5",
        timeframe="2H",
        donchian=20,
        adx_min=20.0,
        risk_pct=0.01,
        atr_stop_mult=1.5,
        momentum_roc=12,
        ema_fast=20,
    ),
]


@dataclass
class Trade:
    side: int
    entry_ts: float
    exit_ts: float
    entry: float
    exit: float
    qty: float
    notional: float
    leverage_used: float
    pnl: float
    pnl_pct_equity: float
    reason: str


@dataclass
class BacktestResult:
    variant: Variant
    source: str
    window: str
    start_ts: float
    end_ts: float
    start_equity: float
    end_equity: float
    ret_pct: float
    max_dd_pct: float
    trades: int
    wins: int
    win_rate: float
    profit_factor: float
    avg_win_pct: float
    avg_loss_pct: float
    bars: int
    first_bar: float
    last_bar: float
    trade_log: list[Trade] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d["variant"] = asdict(self.variant)
        d["trade_log"] = [
            {
                "side": "LONG" if t.side > 0 else "SHORT",
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "entry": t.entry,
                "exit": t.exit,
                "qty": t.qty,
                "notional": t.notional,
                "leverage_used": t.leverage_used,
                "pnl": t.pnl,
                "pnl_pct_equity": t.pnl_pct_equity,
                "reason": t.reason,
            }
            for t in self.trade_log
        ]
        return d


def run_backtest(
    bars: dict[str, np.ndarray],
    daily: dict[str, np.ndarray],
    variant: Variant,
    source: str,
    window: str,
    window_start_ms: float,
    window_end_ms: float,
    start_equity: float = 10_000.0,
) -> BacktestResult:
    ts = bars["ts"]
    o = bars["open"]
    h = bars["high"]
    l = bars["low"]
    c = bars["close"]
    n = len(c)

    dc_up, dc_lo = donchian_prior(h, l, variant.donchian)
    tr_up, tr_lo = donchian_prior(h, l, variant.trail_n)
    atr = atr_wilder(h, l, c, variant.atr_period)
    adx = adx_wilder(h, l, c, variant.adx_period)
    regime = (
        align_daily_ema_regime(ts, daily["ts"], daily["close"])
        if variant.use_ema_regime
        else np.ones(n, dtype=np.int8)
    )
    mom = roc(c, variant.momentum_roc) if variant.momentum_roc else None
    fast_ema = ema(c, variant.ema_fast) if variant.ema_fast else None

    equity = start_equity
    peak = start_equity
    max_dd = 0.0
    realized = 0.0
    trades: list[Trade] = []

    pos_side = 0
    pos_qty = 0.0
    pos_entry = 0.0
    pos_entry_ts = 0.0
    pos_notional = 0.0
    pos_stop = 0.0
    pos_lev = 0.0
    pending_side = 0
    pending_atr = 0.0
    pending_signal_i = -1

    def mtm_equity(price: float) -> float:
        if pos_side == 0:
            return start_equity + realized
        return start_equity + realized + pos_qty * pos_side * (price - pos_entry)

    def close_position(fill: float, fill_ts: float, reason: str) -> None:
        nonlocal equity, realized, pos_side, pos_qty, pos_entry, pos_notional
        nonlocal pos_stop, pos_lev, pos_entry_ts, peak, max_dd
        if pos_side == 0:
            return
        gross = pos_qty * pos_side * (fill - pos_entry)
        cost = pos_notional * variant.rt_cost
        pnl = gross - cost
        realized += pnl
        equity = start_equity + realized
        trades.append(
            Trade(
                side=pos_side,
                entry_ts=pos_entry_ts,
                exit_ts=fill_ts,
                entry=pos_entry,
                exit=fill,
                qty=pos_qty,
                notional=pos_notional,
                leverage_used=pos_lev,
                pnl=pnl,
                pnl_pct_equity=pnl / (equity - pnl) if equity != pnl else 0.0,
                reason=reason,
            )
        )
        pos_side = 0
        pos_qty = 0.0
        pos_entry = 0.0
        pos_notional = 0.0
        pos_stop = 0.0
        pos_lev = 0.0
        pos_entry_ts = 0.0
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)

    def open_position(side: int, fill: float, fill_ts: float, stop_atr: float) -> None:
        nonlocal pos_side, pos_qty, pos_entry, pos_entry_ts, pos_notional
        nonlocal pos_stop, pos_lev, equity
        if side == 0 or stop_atr <= 0:
            return
        stop_dist = variant.atr_stop_mult * stop_atr
        risk_cash = equity * variant.risk_pct
        qty = risk_cash / stop_dist
        notional = qty * fill
        max_notional = equity * variant.max_leverage
        if notional > max_notional:
            qty = max_notional / fill
            notional = max_notional
        if qty <= 0 or notional <= 0:
            return
        pos_side = side
        pos_qty = qty
        pos_entry = fill
        pos_entry_ts = fill_ts
        pos_notional = notional
        pos_lev = notional / equity
        pos_stop = fill - side * stop_dist

    in_window = (ts >= window_start_ms) & (ts <= window_end_ms)
    first_tradeable = int(np.argmax(in_window)) if in_window.any() else n
    last_i = n - 1

    for i in range(n - 1):
        # Fill pending entry at this bar's open (signal was on previous close).
        if pending_side != 0 and i > first_tradeable - 1 and ts[i] >= window_start_ms:
            if pos_side == 0:
                open_position(pending_side, o[i], ts[i], pending_atr)
            pending_side = 0
            pending_atr = 0.0
            pending_signal_i = -1
        else:
            pending_side = 0
            pending_atr = 0.0

        if pos_side != 0:
            # Isolated approx liquidation (MMR 0.5%, leverage setting = 3x).
            liq_move = (1.0 / variant.max_leverage) - 0.005
            liq_price = pos_entry * (1.0 - pos_side * liq_move)
            hit_liq = (pos_side > 0 and l[i] <= liq_price) or (pos_side < 0 and h[i] >= liq_price)
            if hit_liq:
                fill = o[i] if (pos_side > 0 and o[i] <= liq_price) or (
                    pos_side < 0 and o[i] >= liq_price
                ) else liq_price
                close_position(fill, ts[i], "isolated_liq")
            else:
                # Stop / trail: gap at open first, else pierce during bar.
                if pos_side > 0:
                    if o[i] <= pos_stop:
                        close_position(o[i], ts[i], "stop_gap")
                    elif l[i] <= pos_stop:
                        close_position(pos_stop, ts[i], "stop")
                else:
                    if o[i] >= pos_stop:
                        close_position(o[i], ts[i], "stop_gap")
                    elif h[i] >= pos_stop:
                        close_position(pos_stop, ts[i], "stop")

        # Ratchet trail on close if still in position (uses prior-N Donchian).
        if pos_side != 0 and not np.isnan(tr_lo[i]) and not np.isnan(tr_up[i]):
            if pos_side > 0:
                pos_stop = max(pos_stop, tr_lo[i])
            else:
                pos_stop = min(pos_stop, tr_up[i])

        eq = mtm_equity(c[i])
        if ts[i] >= window_start_ms:
            peak = max(peak, eq)
            max_dd = max(max_dd, (peak - eq) / peak if peak > 0 else 0.0)

        # New signal on this close for next-open fill. One position: ignore if flat pending/in.
        if pos_side != 0:
            continue
        if ts[i] < window_start_ms or ts[i] > window_end_ms:
            continue
        if np.isnan(dc_up[i]) or np.isnan(atr[i]) or atr[i] <= 0:
            continue
        if variant.adx_min is not None and (np.isnan(adx[i]) or adx[i] <= variant.adx_min):
            continue

        long_ok = c[i] > dc_up[i]
        short_ok = c[i] < dc_lo[i]
        if variant.use_ema_regime:
            long_ok = long_ok and regime[i] >= 0  # bull or unknown after warmup
            short_ok = short_ok and regime[i] <= 0
            # After EMA200 is live, unknown (0) means skip both.
            if regime[i] == 0:
                long_ok = False
                short_ok = False
        if mom is not None:
            if np.isnan(mom[i]):
                continue
            long_ok = long_ok and mom[i] > 0
            short_ok = short_ok and mom[i] < 0
        if fast_ema is not None:
            if np.isnan(fast_ema[i]):
                continue
            long_ok = long_ok and c[i] > fast_ema[i]
            short_ok = short_ok and c[i] < fast_ema[i]

        if long_ok and not short_ok:
            pending_side = 1
            pending_atr = atr[i]
            pending_signal_i = i
        elif short_ok and not long_ok:
            pending_side = -1
            pending_atr = atr[i]
            pending_signal_i = i

    # Flatten at last window close (paper — no live order).
    if pos_side != 0:
        close_position(c[last_i], ts[last_i], "eod_flatten")

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_pnl = sum(t.pnl for t in wins)
    loss_pnl = abs(sum(t.pnl for t in losses))
    pf = win_pnl / loss_pnl if loss_pnl > 0 else (float("inf") if win_pnl > 0 else 0.0)
    end_equity = start_equity + realized
    return BacktestResult(
        variant=variant,
        source=source,
        window=window,
        start_ts=window_start_ms,
        end_ts=window_end_ms,
        start_equity=start_equity,
        end_equity=end_equity,
        ret_pct=(end_equity / start_equity - 1.0) * 100.0,
        max_dd_pct=max_dd * 100.0,
        trades=len(trades),
        wins=len(wins),
        win_rate=(len(wins) / len(trades) * 100.0) if trades else 0.0,
        profit_factor=pf if pf != float("inf") else 999.0,
        avg_win_pct=(np.mean([t.pnl_pct_equity for t in wins]) * 100.0) if wins else 0.0,
        avg_loss_pct=(np.mean([t.pnl_pct_equity for t in losses]) * 100.0) if losses else 0.0,
        bars=int(in_window.sum()),
        first_bar=float(ts[in_window][0]) if in_window.any() else 0.0,
        last_bar=float(ts[in_window][-1]) if in_window.any() else 0.0,
        trade_log=trades,
    )
