"""Tests for trading strategies."""
import numpy as np
import pytest

from src.core.models import Candle, Side
from src.data.market_data import generate_synthetic_data
from src.strategies.advanced import (
    MarketRegimeStrategy,
    OrderFlowImbalanceStrategy,
    SmartMoneyConceptStrategy,
)
from src.strategies.base import HybridStrategy
from src.strategies.momentum import (
    MeanReversionStrategy,
    MomentumBreakoutStrategy,
    RSITrendMomentumStrategy,
)
from src.strategies.technical import (
    BollingerBreakoutStrategy,
    EMATripleCrossStrategy,
    FibonacciRetracementStrategy,
    RSIMACDStrategy,
    VolumeProfileStrategy,
)


def _make_candles(prices: list[float], volumes: list[float] | None = None) -> list[Candle]:
    """Helper to create candle list from close prices."""
    candles = []
    for i, price in enumerate(prices):
        vol = volumes[i] if volumes else 1000
        candles.append(Candle(
            timestamp=float(i * 3600),
            open=price * 0.999,
            high=price * 1.005,
            low=price * 0.995,
            close=price,
            volume=vol,
            symbol="BTCUSDT",
        ))
    return candles


class TestRSIMACDStrategy:
    def test_default_params(self):
        strategy = RSIMACDStrategy()
        params = strategy.default_params()
        assert params["rsi_period"] == 14
        assert params["macd_fast"] == 12

    def test_analyze_returns_none_when_insufficient_data(self):
        strategy = RSIMACDStrategy()
        candles = _make_candles([50000] * 10)
        result = strategy.analyze(candles)
        assert result is None  # Not enough data

    def test_analyze_with_synthetic_data(self):
        strategy = RSIMACDStrategy()
        candles = generate_synthetic_data("BTCUSDT", n_candles=200, volatility=0.03)
        result = strategy.analyze(candles)
        # May or may not generate signal - just check it doesn't crash
        if result:
            assert result.symbol == "BTCUSDT"
            assert result.side in (Side.LONG, Side.SHORT)
            assert 0 < result.confidence <= 1.0


class TestBollingerBreakoutStrategy:
    def test_default_params(self):
        strategy = BollingerBreakoutStrategy()
        assert strategy.params["bb_period"] == 20

    def test_analyze_no_crash(self):
        strategy = BollingerBreakoutStrategy()
        candles = generate_synthetic_data("BTCUSDT", n_candles=200, volatility=0.03)
        result = strategy.analyze(candles)
        if result:
            assert result.strategy_name == "bollinger_breakout"


class TestMomentumBreakoutStrategy:
    def test_analyze_no_crash(self):
        strategy = MomentumBreakoutStrategy()
        candles = generate_synthetic_data("BTCUSDT", n_candles=200, volatility=0.05)
        result = strategy.analyze(candles)
        if result:
            assert "ATR" in result.reason


class TestMeanReversionStrategy:
    def test_analyze_no_crash(self):
        strategy = MeanReversionStrategy()
        candles = generate_synthetic_data("BTCUSDT", n_candles=200, volatility=0.03)
        result = strategy.analyze(candles)
        if result:
            assert "z=" in result.reason


class TestHybridStrategy:
    def test_hybrid_creation(self):
        strategies = [
            (RSIMACDStrategy(), 0.5),
            (BollingerBreakoutStrategy(), 0.5),
        ]
        hybrid = HybridStrategy(strategies)
        assert hybrid.name == "hybrid"
        assert len(hybrid.strategies) == 2

    def test_hybrid_analyze_no_crash(self):
        strategies = [
            (RSIMACDStrategy(), 0.5),
            (MomentumBreakoutStrategy(), 0.5),
        ]
        hybrid = HybridStrategy(strategies)
        candles = generate_synthetic_data("BTCUSDT", n_candles=200, volatility=0.03)
        result = hybrid.analyze(candles)
        # Just ensure no crash
        if result:
            assert result.strategy_name == "hybrid"


class TestAllStrategies:
    """Run all strategies through synthetic data to ensure no crashes."""

    @pytest.fixture
    def candles(self):
        return generate_synthetic_data("BTCUSDT", n_candles=300, volatility=0.025)

    def test_ema_triple_cross(self, candles):
        s = EMATripleCrossStrategy()
        s.analyze(candles)

    def test_fibonacci(self, candles):
        s = FibonacciRetracementStrategy()
        s.analyze(candles)

    def test_volume_profile(self, candles):
        s = VolumeProfileStrategy()
        s.analyze(candles)

    def test_rsi_trend_momentum(self, candles):
        s = RSITrendMomentumStrategy()
        s.analyze(candles)

    def test_order_flow(self, candles):
        s = OrderFlowImbalanceStrategy()
        s.analyze(candles)

    def test_smc(self, candles):
        s = SmartMoneyConceptStrategy()
        s.analyze(candles)

    def test_market_regime(self, candles):
        s = MarketRegimeStrategy()
        s.analyze(candles)
