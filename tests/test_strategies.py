"""Tests for trading strategies."""
import numpy as np
import pytest

from tests.conftest import make_candles

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


class TestRSIMACDStrategy:
    def test_default_params(self):
        strategy = RSIMACDStrategy()
        params = strategy.default_params()
        assert params["rsi_period"] == 14
        assert params["macd_fast"] == 12

    def test_analyze_returns_none_when_insufficient_data(self):
        strategy = RSIMACDStrategy()
        candles = make_candles([50000] * 10)
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

    def test_ema_triple_cross(self, sample_candles):
        s = EMATripleCrossStrategy()
        s.analyze(sample_candles)

    def test_fibonacci(self, sample_candles):
        s = FibonacciRetracementStrategy()
        s.analyze(sample_candles)

    def test_volume_profile(self, sample_candles):
        s = VolumeProfileStrategy()
        s.analyze(sample_candles)

    def test_rsi_trend_momentum(self, sample_candles):
        s = RSITrendMomentumStrategy()
        s.analyze(sample_candles)

    def test_order_flow(self, sample_candles):
        s = OrderFlowImbalanceStrategy()
        s.analyze(sample_candles)

    def test_smc(self, sample_candles):
        s = SmartMoneyConceptStrategy()
        s.analyze(sample_candles)

    def test_market_regime(self, sample_candles):
        s = MarketRegimeStrategy()
        s.analyze(sample_candles)
