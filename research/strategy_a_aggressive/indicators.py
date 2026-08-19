"""Wilder / EMA / Donchian indicators for Strategy A research.

BotBot's live strategies in ``src/strategies`` use SMA-smoothed ATR/ADX
(``momentum.py``, ``advanced.py``). Official Strategy A v1.1 matches
TradingView-style Wilder RMA, so this module is standalone and does not
import the simulation stack (also avoids a pandas dependency).
"""
from __future__ import annotations

import numpy as np


def wilder_rma(values: np.ndarray, period: int) -> np.ndarray:
    """Wilder moving average. Seed = SMA of the first finite ``period`` window."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    start = None
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        if np.all(np.isfinite(window)):
            start = i
            out[i] = float(np.mean(window))
            break
    if start is None:
        return out
    for i in range(start + 1, len(values)):
        if not np.isfinite(values[i]) or not np.isfinite(out[i - 1]):
            continue
        out[i] = (out[i - 1] * (period - 1) + values[i]) / period
    return out


def ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA with SMA seed. ``k = 2 / (period + 1)``."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1.0)
    out[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        out[i] = values[i] * k + out[i - 1] * (1.0 - k)
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    return np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    return wilder_rma(true_range(high, low, close), period)


def adx_wilder(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average Directional Index (Wilder). Returns ADX only."""
    n = len(close)
    plus_dm = np.zeros(n, dtype=np.float64)
    minus_dm = np.zeros(n, dtype=np.float64)
    up_move = np.empty(n, dtype=np.float64)
    down_move = np.empty(n, dtype=np.float64)
    up_move[0] = 0.0
    down_move[0] = 0.0
    up_move[1:] = high[1:] - high[:-1]
    down_move[1:] = low[:-1] - low[1:]
    plus_dm[1:] = np.where((up_move[1:] > down_move[1:]) & (up_move[1:] > 0), up_move[1:], 0.0)
    minus_dm[1:] = np.where((down_move[1:] > up_move[1:]) & (down_move[1:] > 0), down_move[1:], 0.0)

    atr = atr_wilder(high, low, close, period)
    plus_di = 100.0 * wilder_rma(plus_dm, period) / atr
    minus_di = 100.0 * wilder_rma(minus_dm, period) / atr
    di_sum = plus_di + minus_di
    dx = np.divide(
        100.0 * np.abs(plus_di - minus_di),
        di_sum,
        out=np.full_like(di_sum, np.nan),
        where=di_sum > 0,
    )
    # DX is defined only after the first ATR/DI seed.
    return wilder_rma(dx, period)


def donchian_prior(high: np.ndarray, low: np.ndarray, period: int) -> tuple[np.ndarray, np.ndarray]:
    """Prior-N Donchian bands (excludes the current bar — no lookahead)."""
    n = len(high)
    upper = np.full(n, np.nan, dtype=np.float64)
    lower = np.full(n, np.nan, dtype=np.float64)
    if n <= period:
        return upper, lower
    # rolling max/min of high[i-period:i] / low[i-period:i]
    for i in range(period, n):
        upper[i] = np.max(high[i - period : i])
        lower[i] = np.min(low[i - period : i])
    return upper, lower


def rolling_max_prior(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for i in range(period, len(values)):
        out[i] = np.max(values[i - period : i])
    return out


def roc(close: np.ndarray, period: int) -> np.ndarray:
    out = np.full(close.shape, np.nan, dtype=np.float64)
    if period <= 0 or len(close) <= period:
        return out
    out[period:] = (close[period:] / close[:-period] - 1.0) * 100.0
    return out


def align_daily_ema_regime(
    bar_ts_ms: np.ndarray,
    daily_ts_ms: np.ndarray,
    daily_close: np.ndarray,
    fast: int = 50,
    slow: int = 200,
) -> np.ndarray:
    """+1 bull (EMA50>EMA200), -1 bear, 0 unknown.

    Uses the last *completed* UTC daily bar before the current bar's day,
    so an intraday 4H/1H/2H bar never sees the same day's close.
    """
    ema_fast = ema(daily_close, fast)
    ema_slow = ema(daily_close, slow)
    regime_daily = np.zeros(len(daily_close), dtype=np.int8)
    valid = ~np.isnan(ema_fast) & ~np.isnan(ema_slow)
    regime_daily[valid & (ema_fast > ema_slow)] = 1
    regime_daily[valid & (ema_fast < ema_slow)] = -1

    # Day start of each bar (UTC midnight).
    day_ms = 86_400_000
    bar_day_start = (bar_ts_ms // day_ms) * day_ms
    # Last daily bar with ts < bar_day_start
    idx = np.searchsorted(daily_ts_ms, bar_day_start, side="left") - 1
    out = np.zeros(len(bar_ts_ms), dtype=np.int8)
    good = idx >= 0
    out[good] = regime_daily[idx[good]]
    return out
