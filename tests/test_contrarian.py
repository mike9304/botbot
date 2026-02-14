"""Tests for contrarian, psychology, and counter-indicator strategies."""
import pytest

from src.core.models import Candle, Side, TradeSignal
from src.core.exchange import VirtualExchange
from src.data.market_data import generate_synthetic_data
from src.strategies.contrarian import (
    FearGreedContrarianStrategy,
    FundingRateContrarianStrategy,
    RetailSentimentFaderStrategy,
    WyckoffPsychologyStrategy,
)
from src.strategies.counter_indicator import (
    CounterIndicatorStrategy,
    StrategyTypeCounterIndicatorStrategy,
)
from src.agents.base_agent import AgentConfig, AgentGroup
from src.agents.counter_agent import CounterIndicatorAgent, StrategyTypeCounterAgent


class TestFearGreedContrarian:
    def test_default_params(self):
        s = FearGreedContrarianStrategy()
        assert s.params["extreme_greed_threshold"] == 78
        assert s.params["extreme_fear_threshold"] == 22

    def test_analyze_no_crash(self):
        s = FearGreedContrarianStrategy()
        candles = generate_synthetic_data("BTCUSDT", n_candles=200, volatility=0.03)
        result = s.analyze(candles)
        if result:
            assert result.strategy_name == "fear_greed_contrarian"
            assert "CONTRARIAN" in result.reason

    def test_custom_params(self):
        s = FearGreedContrarianStrategy({
            "extreme_greed_threshold": 70,
            "extreme_fear_threshold": 30,
        })
        assert s.params["extreme_greed_threshold"] == 70
        assert s.params["rsi_period"] == 14  # Default preserved


class TestRetailFader:
    def test_analyze_no_crash(self):
        s = RetailSentimentFaderStrategy()
        candles = generate_synthetic_data("BTCUSDT", n_candles=200, volatility=0.04)
        result = s.analyze(candles)
        if result:
            assert result.strategy_name == "retail_fader"


class TestFundingRateContrarian:
    def test_analyze_no_crash(self):
        s = FundingRateContrarianStrategy()
        candles = generate_synthetic_data("BTCUSDT", n_candles=200, volatility=0.03)
        result = s.analyze(candles)
        if result:
            assert "FUNDING" in result.reason


class TestWyckoffPsychology:
    def test_analyze_no_crash(self):
        s = WyckoffPsychologyStrategy()
        candles = generate_synthetic_data("BTCUSDT", n_candles=200, volatility=0.02)
        result = s.analyze(candles)
        if result:
            assert "WYCKOFF" in result.reason


class TestCounterIndicatorStrategy:
    def test_tracking(self):
        counter = CounterIndicatorStrategy()

        # Simulate agent signals
        for i in range(15):
            signal = TradeSignal(
                symbol="BTCUSDT",
                side=Side.LONG,
                confidence=0.7,
                strategy_name="bad_strategy",
            )
            # Register with progressively higher price (signal was wrong to go long if price drops)
            counter.register_signal("bad_agent", "bad_strategy", signal, 50000 - i * 100)

        report = counter.get_tracking_report()
        assert len(report) == 1
        assert report[0]["agent_id"] == "bad_agent"

    def test_inversion_candidate(self):
        counter = CounterIndicatorStrategy()
        tracker = counter.tracked_agents.setdefault(
            "loser", type("T", (), {
                "agent_id": "loser", "strategy_name": "bad",
                "total_signals": 20, "correct_signals": 3, "wrong_signals": 17,
                "recent_results": [], "pending_signal": None, "pending_price": 0,
            })()
        )
        # Can't directly test with mock, but let's test the actual tracker
        from src.strategies.counter_indicator import AgentTracker
        at = AgentTracker(agent_id="loser_real", strategy_name="bad")
        for i in range(20):
            at.record_result(i < 4)  # 4 correct, 16 wrong = 20% accuracy
        assert at.accuracy == 0.2
        assert at.is_reliable_loser
        assert at.inversion_confidence == 0.8

    def test_generate_inverted_signal(self):
        counter = CounterIndicatorStrategy()
        from src.strategies.counter_indicator import AgentTracker
        tracker = AgentTracker(agent_id="loser", strategy_name="bad")
        for _ in range(20):
            tracker.record_result(False)  # 0% accuracy = perfect reverse indicator
        counter.tracked_agents["loser"] = tracker

        source_signal = TradeSignal(
            symbol="BTCUSDT", side=Side.LONG, confidence=0.7,
            strategy_name="bad",
        )
        inverted = counter.generate_counter_signal("loser", source_signal)
        assert inverted is not None
        assert inverted.side == Side.SHORT  # Inverted!
        assert "COUNTER-INDICATOR" in inverted.reason


class TestStrategyTypeCounter:
    def test_type_tracking(self):
        counter = StrategyTypeCounterIndicatorStrategy()
        for i in range(15):
            signal = TradeSignal(
                symbol="BTCUSDT", side=Side.LONG, confidence=0.7,
                strategy_name="test",
            )
            counter.register_strategy_type_signal("momentum", signal, 50000 - i * 200)

        report = counter.get_type_report()
        assert len(report) == 1
        assert report[0]["strategy_type"] == "momentum"


class TestCounterIndicatorAgent:
    def test_agent_creation(self):
        exchange = VirtualExchange()
        config = AgentConfig(
            name="test_counter", group="counter", strategy=CounterIndicatorStrategy(),
            symbols=["BTCUSDT"],
        )
        agent = CounterIndicatorAgent(config, exchange)
        assert agent.id is not None
        summary = agent.get_summary()
        assert summary["observed_signals"] == 0

    def test_signal_observation(self):
        exchange = VirtualExchange()
        config = AgentConfig(
            name="test_counter", group="counter", strategy=CounterIndicatorStrategy(),
            symbols=["BTCUSDT"],
        )
        agent = CounterIndicatorAgent(config, exchange)

        candle = Candle(
            timestamp=1000, open=50000, high=50100, low=49900,
            close=50000, volume=100, symbol="BTCUSDT",
        )
        exchange.update_price("BTCUSDT", candle)

        signal = TradeSignal(
            symbol="BTCUSDT", side=Side.LONG, confidence=0.7,
            strategy_name="test",
        )
        agent.on_other_agent_signal("other_agent", "test_strat", signal, 50000)
        assert agent._observed_count == 1


class TestAllContrarianStrategies:
    """Smoke test all contrarian strategies with varied synthetic data."""

    @pytest.fixture
    def candles_volatile(self):
        return generate_synthetic_data("ETHUSDT", n_candles=300, volatility=0.04)

    @pytest.fixture
    def candles_calm(self):
        return generate_synthetic_data("BTCUSDT", n_candles=300, volatility=0.01)

    def test_fear_greed_volatile(self, candles_volatile):
        s = FearGreedContrarianStrategy()
        s.analyze(candles_volatile)

    def test_fear_greed_calm(self, candles_calm):
        s = FearGreedContrarianStrategy()
        s.analyze(candles_calm)

    def test_retail_fader_volatile(self, candles_volatile):
        s = RetailSentimentFaderStrategy()
        s.analyze(candles_volatile)

    def test_funding_contrarian_volatile(self, candles_volatile):
        s = FundingRateContrarianStrategy()
        s.analyze(candles_volatile)

    def test_wyckoff_calm(self, candles_calm):
        s = WyckoffPsychologyStrategy()
        s.analyze(candles_calm)
