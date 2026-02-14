"""Tests for genetic evolution system."""
import pytest

from src.agents.base_agent import AgentConfig, AgentGroup, TradingAgent
from src.core.exchange import VirtualExchange
from src.core.models import Candle, Side, TradeSignal
from src.data.market_data import generate_synthetic_data
from src.evolution.genetic import EvolutionConfig, GeneticEvolver
from src.evolution.reinforcement import RLEnhancedEvolution, RLState, SimpleQLearner
from src.strategies.technical import RSIMACDStrategy


class TestGeneticEvolver:
    def setup_method(self):
        self.exchange = VirtualExchange()
        self.group = AgentGroup("Test Group", "Test", "test")

        # Create some agents
        for i in range(5):
            config = AgentConfig(
                name=f"agent_{i}",
                group="test",
                strategy=RSIMACDStrategy(),
                symbols=["BTCUSDT"],
            )
            agent = TradingAgent(config, self.exchange)
            self.group.add_agent(agent)

    def test_evolve_requires_min_trades(self):
        evolver = GeneticEvolver(EvolutionConfig(min_trades_for_eval=5))
        result = evolver.evolve_group(self.group, self.exchange)
        # No agents have trades, so evolution should skip
        assert result.agents_fired == 0
        assert result.agents_hired == 0

    def test_evolve_with_trades(self):
        # Simulate some trades for each agent
        candle = Candle(
            timestamp=1000, open=50000, high=50100, low=49900,
            close=50000, volume=100, symbol="BTCUSDT",
        )
        self.exchange.update_price("BTCUSDT", candle)

        for agent in self.group.agents:
            account = self.exchange.accounts[agent.id]
            # Manually add trade history
            account.total_trades = 10
            account.win_count = 6
            account.loss_count = 4
            account.total_pnl = float(hash(agent.id) % 200 - 100)

        evolver = GeneticEvolver(EvolutionConfig(
            min_trades_for_eval=5,
            fire_ratio=0.2,
            elite_ratio=0.2,
        ))
        result = evolver.evolve_group(self.group, self.exchange)
        assert result.generation == 1
        assert result.agents_fired >= 1
        assert result.agents_hired >= 1


class TestSimpleQLearner:
    def test_creation(self):
        learner = SimpleQLearner(state_dim=10, n_actions=4)
        assert learner.weights.shape == (10, 4)
        assert learner.epsilon == 1.0

    def test_select_action(self):
        learner = SimpleQLearner(state_dim=10, n_actions=4)
        state = RLState(price_returns=[0.01] * 3).to_array()
        # Pad to match state_dim
        import numpy as np
        state = np.zeros(10)
        action = learner.select_action(state)
        assert 0 <= action < 4

    def test_get_stats(self):
        learner = SimpleQLearner()
        stats = learner.get_stats()
        assert "epsilon" in stats
        assert "total_steps" in stats


class TestRLEnhancedEvolution:
    def test_build_state(self):
        rl = RLEnhancedEvolution(state_dim=17, n_returns=10)
        state = rl.build_state(
            prices=[50000, 50100, 50200, 50300, 50400, 50500, 50600, 50700, 50800, 50900, 51000],
            volume=1000,
            avg_volume=800,
            has_position=False,
            position_pnl_pct=0.0,
            equity_pct=1.0,
            drawdown=0.0,
        )
        assert isinstance(state, RLState)
        arr = state.to_array()
        assert len(arr) == 17

    def test_decide(self):
        rl = RLEnhancedEvolution(state_dim=17, n_returns=10)
        state = rl.build_state(
            prices=[50000 + i * 100 for i in range(12)],
            volume=1000, avg_volume=800,
            has_position=False, position_pnl_pct=0.0,
            equity_pct=1.0, drawdown=0.0,
        )
        action = rl.decide(state)
        assert 0 <= action < 4
