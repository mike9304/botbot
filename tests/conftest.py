"""Shared test fixtures for the BotBot test suite.

Centralizes common setup patterns: exchange, agents, candle generation,
and trade simulation helpers used across multiple test modules.
"""
from __future__ import annotations

import pytest

from src.agents.base_agent import AgentConfig, AgentGroup, TradingAgent
from src.core.exchange import VirtualExchange
from src.core.models import Candle
from src.data.market_data import generate_synthetic_data
from src.strategies.technical import RSIMACDStrategy


@pytest.fixture
def exchange() -> VirtualExchange:
    """Fresh VirtualExchange instance."""
    return VirtualExchange()


@pytest.fixture
def agent(exchange: VirtualExchange) -> TradingAgent:
    """Single agent registered on the exchange."""
    config = AgentConfig(
        name="test_agent",
        group="test",
        strategy=RSIMACDStrategy(),
        symbols=["BTCUSDT"],
    )
    return TradingAgent(config, exchange)


@pytest.fixture
def group_with_agents(exchange: VirtualExchange) -> AgentGroup:
    """AgentGroup with 5 agents for evolution / teacher tests."""
    group = AgentGroup(name="Test Group", description="Test", category="test")
    for i in range(5):
        config = AgentConfig(
            name=f"agent_{i}",
            group="test",
            strategy=RSIMACDStrategy(),
            symbols=["BTCUSDT"],
        )
        a = TradingAgent(config, exchange)
        group.add_agent(a)
    return group


@pytest.fixture
def sample_candles() -> list[Candle]:
    """300 synthetic BTC candles for strategy smoke tests."""
    return generate_synthetic_data("BTCUSDT", n_candles=300, volatility=0.025)


def make_candles(prices: list[float], volumes: list[float] | None = None) -> list[Candle]:
    """Create candle list from close prices (deterministic)."""
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


def simulate_trades(
    agent: TradingAgent,
    n_wins: int = 3,
    n_losses: int = 2,
) -> None:
    """Simulate wins/losses on an agent's account for testing."""
    account = agent.account
    if not account:
        return
    for _ in range(n_wins):
        account.win_count += 1
        account.total_trades += 1
        account.total_pnl += 100
        account.equity += 100
        account.balance += 100
    for _ in range(n_losses):
        account.loss_count += 1
        account.total_trades += 1
        account.total_pnl -= 50
        account.equity -= 50
        account.balance -= 50
