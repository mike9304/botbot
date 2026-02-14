"""Base trading agent and agent group system."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional

from src.core.exchange import VirtualExchange
from src.core.models import AccountState, Candle, Side, TradeSignal
from src.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    name: str
    group: str
    strategy: BaseStrategy
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT"])
    initial_balance: float = 10000.0
    max_positions: int = 3
    risk_per_trade: float = 0.02  # 2% of equity per trade
    max_daily_trades: int = 10
    cooldown_candles: int = 5  # Minimum candles between trades


class TradingAgent:
    """Individual trading agent that uses a strategy to trade on a virtual exchange."""

    def __init__(self, config: AgentConfig, exchange: VirtualExchange):
        self.id = f"{config.group}_{config.name}_{uuid.uuid4().hex[:6]}"
        self.config = config
        self.exchange = exchange
        self.strategy = config.strategy
        self.daily_trades = 0
        self.candles_since_last_trade = 999
        self.active = True

        # Register on exchange
        exchange.register_agent(self.id, config.initial_balance)

    @property
    def account(self) -> Optional[AccountState]:
        return self.exchange.accounts.get(self.id)

    def on_candle(self, candle: Candle):
        """Process new candle data."""
        if not self.active:
            return

        if candle.symbol not in self.config.symbols:
            return

        self.candles_since_last_trade += 1

        # Check existing position management
        account = self.account
        if not account:
            return

        # Strategy analysis
        signal = self.strategy.update(candle)

        if signal and self._can_trade(account, signal):
            order = self.exchange.execute_signal(self.id, signal)
            if order:
                self.daily_trades += 1
                self.candles_since_last_trade = 0

    def _can_trade(self, account: AccountState, signal: TradeSignal) -> bool:
        if self.daily_trades >= self.config.max_daily_trades:
            return False
        if self.candles_since_last_trade < self.config.cooldown_candles:
            return False
        if len(account.positions) >= self.config.max_positions:
            return False
        if account.equity < account.balance * 0.1:  # Equity too low
            return False
        return True

    def reset_daily_counter(self):
        self.daily_trades = 0

    def get_summary(self) -> dict:
        base = self.exchange.get_account_summary(self.id)
        base.update({
            "name": self.config.name,
            "group": self.config.group,
            "strategy": self.strategy.name,
            "active": self.active,
            "signals_generated": self.strategy.signals_generated,
        })
        return base

    @property
    def fitness(self) -> float:
        """Calculate fitness score for genetic evolution.

        Fitness considers:
        - Total PnL (primary)
        - Win rate (secondary)
        - Max drawdown (penalty)
        - Sharpe-like ratio
        """
        account = self.account
        if not account or account.total_trades == 0:
            return 0.0

        pnl_score = account.pnl_percent
        win_rate_bonus = (account.win_rate - 0.5) * 20  # Bonus for >50% WR
        drawdown_penalty = account.max_drawdown * 50  # Penalize drawdown
        trade_frequency = min(account.total_trades / 10, 2.0)  # Reward active trading

        return pnl_score + win_rate_bonus - drawdown_penalty + trade_frequency


class AgentGroup:
    """A group of agents sharing a strategy category or theme."""

    def __init__(self, name: str, description: str, category: str):
        self.name = name
        self.description = description
        self.category = category
        self.agents: list[TradingAgent] = []

    def add_agent(self, agent: TradingAgent):
        self.agents.append(agent)

    def on_candle(self, candle: Candle):
        for agent in self.agents:
            agent.on_candle(candle)

    def reset_daily_counters(self):
        for agent in self.agents:
            agent.reset_daily_counter()

    def get_rankings(self) -> list[dict]:
        summaries = [a.get_summary() for a in self.agents]
        return sorted(summaries, key=lambda x: x.get("total_pnl", 0), reverse=True)

    def get_best_agent(self) -> Optional[TradingAgent]:
        if not self.agents:
            return None
        return max(self.agents, key=lambda a: a.fitness)

    def get_worst_agent(self) -> Optional[TradingAgent]:
        if not self.agents:
            return None
        return min(self.agents, key=lambda a: a.fitness)

    @property
    def total_pnl(self) -> float:
        return sum(
            a.account.total_pnl for a in self.agents if a.account
        )

    @property
    def avg_win_rate(self) -> float:
        rates = [a.account.win_rate for a in self.agents if a.account and a.account.total_trades > 0]
        return sum(rates) / len(rates) if rates else 0.0
