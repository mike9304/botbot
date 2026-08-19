"""Offline unit checks for Strategy A research engine (no network)."""
from __future__ import annotations

import numpy as np

from research.strategy_a_aggressive.engine import Variant, run_backtest
from research.strategy_a_aggressive.indicators import adx_wilder, atr_wilder, donchian_prior, ema


def test_ema_seed_and_update() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    e = ema(x, 3)
    assert np.isnan(e[1])
    assert abs(e[2] - 2.0) < 1e-12  # SMA seed
    k = 2.0 / 4.0
    assert abs(e[3] - (4.0 * k + 2.0 * (1 - k))) < 1e-12


def test_donchian_excludes_current_bar() -> None:
    high = np.array([1.0, 3.0, 2.0, 10.0, 4.0])
    low = np.array([0.5, 1.0, 0.8, 2.0, 1.5])
    up, lo = donchian_prior(high, low, 2)
    assert np.isnan(up[1])
    assert up[3] == 3.0  # max(high[1:3]) = max(3,2), not 10
    assert lo[3] == 0.8


def test_atr_positive_on_range() -> None:
    h = np.linspace(11, 20, 30)
    l = h - 2.0
    c = h - 0.5
    a = atr_wilder(h, l, c, 14)
    assert a[13] > 0
    assert not np.isnan(a[-1])


def test_adx_finite_after_warmup() -> None:
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, 80))
    high = close + 1.0
    low = close - 1.0
    adx = adx_wilder(high, low, close, 14)
    assert np.isnan(adx[20])  # 2*period seed
    assert np.isfinite(adx[-1])
    assert 0 <= adx[-1] <= 100


def test_leverage_cap_and_one_position() -> None:
    n = 400
    ts = np.arange(n) * 3_600_000.0 + 1_600_000_000_000.0
    # Flat range, then a clean upside break, then a downside break.
    close = np.full(n, 100.0)
    close[80:90] = np.linspace(100, 130, 10)
    close[90:200] = 130.0
    close[200:220] = np.linspace(130, 85, 20)
    close[220:] = 85.0
    high = close + 0.4
    high[85] = 136.0  # breakout high
    low = close - 0.4
    low[210] = 80.0
    open_ = np.r_[close[0], close[:-1]]
    bars = {"ts": ts, "open": open_, "high": high, "low": low, "close": close, "volume": np.ones(n)}
    daily_n = 220
    dts = ts[0] - np.arange(daily_n, 0, -1) * 86_400_000.0
    dclose = np.linspace(80, 130, daily_n)
    daily = {"ts": dts, "open": dclose, "high": dclose, "low": dclose, "close": dclose, "volume": np.ones(daily_n)}
    variant = Variant(
        id="T",
        name="unit",
        timeframe="1H",
        donchian=10,
        adx_min=None,
        risk_pct=0.01,
        atr_stop_mult=1.5,
        use_ema_regime=False,
    )
    result = run_backtest(bars, daily, variant, "synth", "unit", ts[50], ts[-1])
    assert result.trades >= 1
    assert all(t.leverage_used <= 3.0001 for t in result.trade_log)
    # No overlapping positions: exit <= next entry
    entries = [t.entry_ts for t in result.trade_log]
    exits = [t.exit_ts for t in result.trade_log]
    for i in range(len(entries) - 1):
        assert exits[i] <= entries[i + 1]


if __name__ == "__main__":
    test_ema_seed_and_update()
    test_donchian_excludes_current_bar()
    test_atr_positive_on_range()
    test_adx_finite_after_warmup()
    test_leverage_cap_and_one_position()
    print("ok")
