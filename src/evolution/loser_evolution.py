"""Loser Evolution System - Evolve the worst agents into reliable reverse indicators.

Core concept: Instead of discarding bottom performers, we give them their own
evolution track. They breed among themselves, creating offspring that are
*also* likely to be wrong — which makes them BETTER reverse indicators.

Think of it like breeding the world's worst fortune tellers. Their predictions
are so consistently wrong that doing the opposite is incredibly profitable.

The system:
1. Collect the worst agents from all groups (bottom 20%)
2. Put them in a "Loser League" where they evolve separately
3. Use genetic algorithm to breed them: crossover + mutation
4. Their evolved offspring are used exclusively as REVERSE indicators
5. A companion "Inverse Loser" group takes all their signals and flips them

The key insight: Random agents have 50% accuracy (useless).
Consistently wrong agents have 30% accuracy (=70% accuracy when inverted).
EVOLVED consistently wrong agents may reach 20% accuracy (=80% inverted!).

References:
- "Anti-Portfolio" concept from VC (Bessemer's famous anti-portfolio)
- "Contrarian Diversification" - using negatively correlated signals
- Tournament selection for WORST fitness in genetic algorithms
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.agents.base_agent import AgentConfig, AgentGroup, TradingAgent
from src.core.exchange import VirtualExchange
from src.core.models import Candle, Side, TradeSignal
from src.strategies.base import BaseStrategy, HybridStrategy

logger = logging.getLogger(__name__)


@dataclass
class LoserEvolutionConfig:
    """Config for evolving the worst performers."""

    collection_threshold: float = 0.3  # Bottom 30% are "losers"
    min_trades_for_collection: int = 8
    loser_pool_size: int = 15  # Max losers to maintain
    evolution_elite_ratio: float = 0.3  # Keep top 30% of losers (worst fitness = best for inversion)
    mutation_rate: float = 0.4  # Higher mutation for more diversity
    mutation_strength: float = 0.25
    crossover_rate: float = 0.6
    breed_count: int = 5  # New losers per generation


class LoserLeague:
    """Manages the evolution of the worst-performing agents.

    The Loser League is a separate evolution track where:
    - Agents are selected for being BAD (low fitness)
    - They breed to create offspring that are also likely bad
    - Their signals are inverted by companion agents for profit
    """

    def __init__(self, config: Optional[LoserEvolutionConfig] = None):
        self.config = config or LoserEvolutionConfig()
        self.loser_pool: list[TradingAgent] = []
        self.generation = 0
        self.history: list[dict] = []

    def collect_losers(self, groups: list[AgentGroup]) -> list[TradingAgent]:
        """Collect the worst agents from all groups.

        Returns agents sorted by fitness (worst first).
        """
        all_agents = []
        for group in groups:
            # Skip counter-indicator groups to avoid circular dependency
            if group.category in ("counter_indicator",):
                continue
            for agent in group.agents:
                if (
                    agent.account
                    and agent.active
                    and agent.account.total_trades >= self.config.min_trades_for_collection
                ):
                    all_agents.append(agent)

        if not all_agents:
            return []

        # Sort by fitness ascending (worst first)
        all_agents.sort(key=lambda a: a.fitness)

        # Take bottom N%
        n_losers = max(1, int(len(all_agents) * self.config.collection_threshold))
        losers = all_agents[:n_losers]

        logger.info(
            f"LoserLeague: Collected {len(losers)} losers from {len(all_agents)} agents. "
            f"Worst fitness: {losers[0].fitness:.2f}, "
            f"Best loser fitness: {losers[-1].fitness:.2f}"
        )

        return losers

    def evolve_losers(
        self,
        source_losers: list[TradingAgent],
        exchange: VirtualExchange,
        loser_group: AgentGroup,
    ) -> list[TradingAgent]:
        """Evolve the loser pool to breed more reliably wrong agents.

        Unlike normal evolution which keeps the BEST, we keep the WORST
        and breed them to create offspring that are also likely wrong.
        """
        self.generation += 1

        if len(source_losers) < 2:
            return []

        # Select "elite losers" (the WORST performers)
        n_elite = max(2, int(len(source_losers) * self.config.evolution_elite_ratio))
        elite_losers = source_losers[:n_elite]  # Already sorted worst-first

        new_agents = []
        for i in range(self.config.breed_count):
            # Tournament selection — pick the WORST from random sample
            parent1 = self._inverse_tournament_select(elite_losers)
            parent2 = self._inverse_tournament_select(elite_losers)

            # Crossover and mutate
            child_strategy = self._breed_loser(parent1.strategy, parent2.strategy)

            child_config = AgentConfig(
                name=f"loser_gen{self.generation}_{i}",
                group="loser_league",
                strategy=child_strategy,
                symbols=parent1.config.symbols,
                initial_balance=10000.0,
            )
            child = TradingAgent(child_config, exchange)
            new_agents.append(child)
            loser_group.add_agent(child)

        # Track history
        self.history.append({
            "generation": self.generation,
            "source_losers": len(source_losers),
            "elite_losers": n_elite,
            "new_bred": len(new_agents),
            "worst_fitness": source_losers[0].fitness if source_losers else 0,
            "avg_loser_fitness": np.mean([a.fitness for a in source_losers]),
        })

        logger.info(
            f"LoserLeague Gen {self.generation}: "
            f"Bred {len(new_agents)} new losers from {n_elite} elite losers"
        )

        return new_agents

    def _inverse_tournament_select(self, agents: list[TradingAgent]) -> TradingAgent:
        """Select the WORST agent from a random tournament (inverse of normal)."""
        k = min(3, len(agents))
        tournament = random.sample(agents, k)
        return min(tournament, key=lambda a: a.fitness)  # Min = worst = what we want

    def _breed_loser(self, s1: BaseStrategy, s2: BaseStrategy) -> BaseStrategy:
        """Breed two loser strategies. Higher mutation than normal evolution."""
        if type(s1) is type(s2):
            child = s1.clone()
            if random.random() < self.config.crossover_rate:
                p1, p2 = s1.get_params(), s2.get_params()
                child_params = {}
                for key in set(p1.keys()) | set(p2.keys()):
                    if key in p1 and key in p2:
                        if isinstance(p1[key], (int, float)):
                            alpha = random.uniform(0.2, 0.8)
                            val = alpha * p1[key] + (1 - alpha) * p2[key]
                            child_params[key] = type(p1[key])(val)
                        else:
                            child_params[key] = random.choice([p1[key], p2[key]])
                    else:
                        child_params[key] = p1.get(key, p2.get(key))
                child.set_params(child_params)
        else:
            # Different types → hybrid of two losers
            child = HybridStrategy(
                [(s1.clone(), random.uniform(0.3, 0.7)),
                 (s2.clone(), random.uniform(0.3, 0.7))],
                {"confidence_threshold": random.uniform(0.4, 0.6)},
            )

        # Aggressive mutation
        if random.random() < self.config.mutation_rate:
            params = child.get_params()
            for key, val in params.items():
                if random.random() > self.config.mutation_rate:
                    continue
                if isinstance(val, float):
                    delta = val * self.config.mutation_strength * random.uniform(-1, 1)
                    params[key] = max(0.001, val + delta)
                elif isinstance(val, int) and key not in ("max_history", "min_candles"):
                    delta = max(1, int(val * self.config.mutation_strength))
                    params[key] = max(1, val + random.randint(-delta, delta))
            child.set_params(params)

        return child


class InverseLoserAgent(TradingAgent):
    """Agent that watches the Loser League and inverts all their signals.

    This is the profit-making companion to the Loser League.
    Every signal the losers generate gets flipped.
    """

    def __init__(self, config: AgentConfig, exchange: VirtualExchange):
        super().__init__(config, exchange)
        self._inverted_count = 0
        self._observed_count = 0

    def on_loser_signal(
        self, loser_id: str, signal: TradeSignal, current_price: float
    ):
        """Receive and invert a signal from a Loser League agent."""
        if not self.active:
            return

        self._observed_count += 1

        account = self.account
        if not account:
            return

        # Invert the signal
        inverted_side = Side.SHORT if signal.side == Side.LONG else Side.LONG
        inverted_signal = TradeSignal(
            symbol=signal.symbol,
            side=inverted_side,
            confidence=signal.confidence,
            strategy_name="inverse_loser",
            leverage=min(signal.leverage, 5),
            stop_loss_pct=signal.stop_loss_pct or 0.02,
            take_profit_pct=signal.take_profit_pct or 0.04,
            reason=f"INVERSE LOSER: flipped {loser_id}'s {signal.side.value} → {inverted_side.value}",
        )

        if self._can_trade(account, inverted_signal):
            order = self.exchange.execute_signal(self.id, inverted_signal)
            if order:
                self._inverted_count += 1
                self.daily_trades += 1
                self.candles_since_last_trade = 0

    def on_candle(self, candle: Candle) -> Optional[TradeSignal]:
        if not self.active:
            return None
        self.candles_since_last_trade += 1
        return None

    def get_summary(self) -> dict:
        base = super().get_summary()
        base.update({
            "observed_loser_signals": self._observed_count,
            "inverted_trades": self._inverted_count,
        })
        return base
