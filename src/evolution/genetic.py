"""Genetic Algorithm evolution system for trading agents.

Implements a "hire and fire" system where:
- Top performing agents survive and reproduce (crossover + mutation)
- Bottom performing agents get replaced by offspring of top agents
- Population evolves over generations to find optimal strategies/parameters

References:
- "Genetic Algorithms in Search, Optimization, and Machine Learning" - Goldberg (1989)
- "NeuroEvolution of Augmenting Topologies" (NEAT) - Stanley & Miikkulainen (2002)
- "Genetic Programming for Financial Trading" - Chen (2002)
"""
from __future__ import annotations

import copy
import logging
import random
from dataclasses import dataclass, field

import numpy as np

from src.agents.base_agent import AgentConfig, AgentGroup, TradingAgent
from src.core.exchange import VirtualExchange
from src.strategies.base import BaseStrategy, HybridStrategy

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    population_size: int = 30
    elite_ratio: float = 0.2  # Top 20% survive unchanged
    mutation_rate: float = 0.3  # 30% chance of parameter mutation
    mutation_strength: float = 0.2  # Max 20% parameter change
    crossover_rate: float = 0.5  # 50% chance of crossover
    tournament_size: int = 3  # Tournament selection size
    fire_ratio: float = 0.2  # Bottom 20% get replaced
    min_trades_for_eval: int = 5  # Minimum trades before evaluation
    prize_pool: float = 1000.0  # USDT prize pool per generation
    prize_distribution: list[float] = field(
        default_factory=lambda: [0.4, 0.25, 0.15, 0.1, 0.05, 0.05]  # Top 6 distribution
    )


class GeneticEvolver:
    """Manages the genetic evolution of trading agents."""

    def __init__(self, config: EvolutionConfig | None = None):
        self.config = config or EvolutionConfig()
        self.generation = 0
        self.history: list[GenerationResult] = []

    def evolve_group(
        self, group: AgentGroup, exchange: VirtualExchange
    ) -> GenerationResult:
        """Run one generation of evolution on an agent group.

        Steps:
        1. Evaluate fitness of all agents
        2. Select elites (top performers - keep unchanged)
        3. Select parents via tournament selection
        4. Create offspring through crossover and mutation
        5. Replace worst performers with new offspring
        6. Award prizes to top performers
        """
        self.generation += 1
        logger.info(f"Generation {self.generation} - Group: {group.name}")

        # Step 1: Evaluate
        agents_with_fitness = [
            (agent, agent.fitness)
            for agent in group.agents
            if agent.account and agent.account.total_trades >= self.config.min_trades_for_eval
        ]

        if len(agents_with_fitness) < 3:
            logger.info("Not enough agents with sufficient trades for evolution")
            return GenerationResult(
                generation=self.generation,
                group_name=group.name,
                best_fitness=0,
                avg_fitness=0,
                agents_fired=0,
                agents_hired=0,
            )

        agents_with_fitness.sort(key=lambda x: x[1], reverse=True)

        # Step 2: Identify elites and worst
        n_elite = max(1, int(len(agents_with_fitness) * self.config.elite_ratio))
        n_fire = max(1, int(len(agents_with_fitness) * self.config.fire_ratio))

        elites = [a for a, _ in agents_with_fitness[:n_elite]]
        fired = [a for a, _ in agents_with_fitness[-n_fire:]]

        # Log performance
        best = agents_with_fitness[0]
        worst = agents_with_fitness[-1]
        avg_fitness = np.mean([f for _, f in agents_with_fitness])

        logger.info(f"  Best: {best[0].config.name} fitness={best[1]:.2f}")
        logger.info(f"  Worst: {worst[0].config.name} fitness={worst[1]:.2f}")
        logger.info(f"  Avg fitness: {avg_fitness:.2f}")
        logger.info(f"  Firing {len(fired)} agents, keeping {len(elites)} elites")

        # Step 3-4: Create offspring to replace fired agents
        new_agents = []
        for _ in range(len(fired)):
            parent1 = self._tournament_select(agents_with_fitness)
            parent2 = self._tournament_select(agents_with_fitness)
            child_strategy = self._crossover_and_mutate(
                parent1.strategy, parent2.strategy
            )
            child_config = AgentConfig(
                name=f"gen{self.generation}_{random.randint(1000,9999)}",
                group=parent1.config.group,
                strategy=child_strategy,
                symbols=parent1.config.symbols,
                initial_balance=10000.0,  # Fresh start
            )
            child = TradingAgent(child_config, exchange)
            new_agents.append(child)

        # Step 5: Replace fired with new
        for agent in fired:
            agent.active = False
            group.agents.remove(agent)

        for agent in new_agents:
            group.add_agent(agent)

        # Step 6: Award prizes
        prizes = self._distribute_prizes(agents_with_fitness)

        result = GenerationResult(
            generation=self.generation,
            group_name=group.name,
            best_fitness=best[1],
            avg_fitness=avg_fitness,
            agents_fired=len(fired),
            agents_hired=len(new_agents),
            prizes=prizes,
            elite_names=[a.config.name for a in elites],
            fired_names=[a.config.name for a in fired],
        )
        self.history.append(result)
        return result

    def _tournament_select(
        self, agents_with_fitness: list[tuple[TradingAgent, float]]
    ) -> TradingAgent:
        """Select a parent using tournament selection."""
        tournament = random.sample(
            agents_with_fitness,
            min(self.config.tournament_size, len(agents_with_fitness)),
        )
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]

    def _crossover_and_mutate(
        self, strategy1: BaseStrategy, strategy2: BaseStrategy
    ) -> BaseStrategy:
        """Create a child strategy through crossover and mutation.

        Strategy-aware crossover (Chen & Navet, 2007):
        - Same type → parameter crossover (blend/swap params)
        - Same category → light mutation of the better parent
        - Different category → heavy mutation only (no crossover)
        This prevents nonsensical hybrids and preserves strategy coherence.
        """
        if type(strategy1) is type(strategy2):
            # Same exact strategy type → standard parameter crossover
            child = strategy1.clone()
            if random.random() < self.config.crossover_rate:
                child_params = self._crossover_params(
                    strategy1.get_params(), strategy2.get_params()
                )
                child.set_params(child_params)
            # Light mutation
            if random.random() < self.config.mutation_rate:
                self._mutate_params(child)

        elif (
            hasattr(strategy1, 'category') and hasattr(strategy2, 'category')
            and strategy1.category == strategy2.category
        ):
            # Same category but different type → clone better, light mutation
            child = strategy1.clone()
            if random.random() < self.config.mutation_rate:
                self._mutate_params(child)

        elif isinstance(strategy1, HybridStrategy) or isinstance(strategy2, HybridStrategy):
            child = strategy1.clone()
            if random.random() < self.config.mutation_rate:
                self._mutate_params(child)
        else:
            # Different categories → heavy mutation only (no crossover)
            # Pick one parent and apply aggressive mutation
            child = random.choice([strategy1, strategy2]).clone()
            heavy_strength = self.config.mutation_strength * 2.5
            params = child.get_params()
            for key, value in params.items():
                if isinstance(value, float):
                    delta = value * heavy_strength * random.uniform(-1, 1)
                    params[key] = max(0.001, value + delta)
                elif isinstance(value, int) and key not in ("max_history", "min_candles"):
                    delta = max(1, int(value * heavy_strength))
                    params[key] = max(1, value + random.randint(-delta, delta))
            child.set_params(params)

        return child

    def _crossover_params(self, params1: dict, params2: dict) -> dict:
        """Uniform crossover of parameters."""
        child_params = {}
        all_keys = set(params1.keys()) | set(params2.keys())
        for key in all_keys:
            if key in params1 and key in params2:
                # Randomly pick from either parent or blend
                if isinstance(params1[key], (int, float)) and isinstance(params2[key], (int, float)):
                    if random.random() < 0.5:
                        # Blend
                        alpha = random.uniform(0.3, 0.7)
                        val = alpha * params1[key] + (1 - alpha) * params2[key]
                        child_params[key] = type(params1[key])(val)
                    else:
                        child_params[key] = random.choice([params1[key], params2[key]])
                else:
                    child_params[key] = random.choice([params1[key], params2[key]])
            elif key in params1:
                child_params[key] = params1[key]
            else:
                child_params[key] = params2[key]
        return child_params

    def _mutate_params(self, strategy: BaseStrategy):
        """Mutate strategy parameters randomly."""
        params = strategy.get_params()
        for key, value in params.items():
            if random.random() > self.config.mutation_rate:
                continue
            if isinstance(value, float):
                delta = value * self.config.mutation_strength * random.uniform(-1, 1)
                params[key] = max(0.001, value + delta)
            elif isinstance(value, int) and key != "max_history" and key != "min_candles":
                delta = max(1, int(value * self.config.mutation_strength))
                params[key] = max(1, value + random.randint(-delta, delta))
        strategy.set_params(params)

    def _distribute_prizes(
        self, agents_with_fitness: list[tuple[TradingAgent, float]]
    ) -> dict[str, float]:
        """Distribute prize pool to top performers."""
        prizes = {}
        pool = self.config.prize_pool
        distribution = self.config.prize_distribution

        for i, (agent, fitness) in enumerate(agents_with_fitness):
            if i >= len(distribution):
                break
            prize = pool * distribution[i]
            prizes[agent.config.name] = prize
            # Add prize to agent balance
            if agent.account:
                agent.account.balance += prize

        return prizes


@dataclass
class GenerationResult:
    generation: int
    group_name: str
    best_fitness: float
    avg_fitness: float
    agents_fired: int
    agents_hired: int
    prizes: dict[str, float] = field(default_factory=dict)
    elite_names: list[str] = field(default_factory=list)
    fired_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "group_name": self.group_name,
            "best_fitness": round(self.best_fitness, 2),
            "avg_fitness": round(self.avg_fitness, 2),
            "agents_fired": self.agents_fired,
            "agents_hired": self.agents_hired,
            "prizes": {k: round(v, 2) for k, v in self.prizes.items()},
            "elites": self.elite_names,
            "fired": self.fired_names,
        }
