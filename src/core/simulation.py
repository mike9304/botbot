"""Main simulation engine that ties everything together.

Runs the trading simulation with:
- Multiple agent groups
- Virtual exchange
- Genetic evolution
- Real-time data broadcasting via WebSocket
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.agents.base_agent import AgentGroup, TradingAgent
from src.agents.group_factory import DEFAULT_SYMBOLS, create_all_groups
from src.core.exchange import VirtualExchange
from src.core.models import Candle
from src.data.market_data import generate_multi_symbol_data
from src.evolution.genetic import EvolutionConfig, GeneticEvolver

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    symbols: list[str] = field(default_factory=lambda: DEFAULT_SYMBOLS[:5])
    n_candles: int = 1000
    evolution_interval: int = 100  # Evolve every N candles
    speed_multiplier: float = 1.0  # Simulation speed (1.0 = real-time-ish)
    use_real_data: bool = False
    broadcast_interval: int = 1  # Broadcast state every N candles


class SimulationEngine:
    """Main simulation engine."""

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.exchange = VirtualExchange()
        self.groups: list[AgentGroup] = []
        self.evolver = GeneticEvolver(EvolutionConfig())
        self.current_candle_idx = 0
        self.running = False
        self.market_data: dict[str, list[Candle]] = {}
        self.subscribers: list[asyncio.Queue] = []
        self.rankings_history: list[dict] = []

    def setup(self):
        """Initialize the simulation."""
        logger.info("Setting up simulation...")

        # Create agent groups
        self.groups = create_all_groups(self.exchange)
        total_agents = sum(len(g.agents) for g in self.groups)
        logger.info(f"Created {len(self.groups)} groups with {total_agents} total agents")

        # Generate or fetch market data
        logger.info(f"Generating market data for {self.config.symbols}...")
        self.market_data = generate_multi_symbol_data(
            self.config.symbols, self.config.n_candles
        )
        logger.info(f"Generated {self.config.n_candles} candles per symbol")

    async def run(self):
        """Run the simulation loop."""
        self.running = True
        self.setup()

        n_candles = self.config.n_candles
        logger.info(f"Starting simulation: {n_candles} candles")

        for i in range(n_candles):
            if not self.running:
                break

            self.current_candle_idx = i

            # Feed candles to exchange and agents
            for symbol in self.config.symbols:
                candles = self.market_data.get(symbol, [])
                if i >= len(candles):
                    continue
                candle = candles[i]

                # Update exchange price
                self.exchange.update_price(symbol, candle)

                # Feed to all groups
                for group in self.groups:
                    group.on_candle(candle)

            # Periodic evolution
            if i > 0 and i % self.config.evolution_interval == 0:
                self._run_evolution()

            # Broadcast state
            if i % self.config.broadcast_interval == 0:
                state = self._build_state_snapshot(i)
                await self._broadcast(state)

            # Simulation speed control
            if self.config.speed_multiplier < 100:
                await asyncio.sleep(0.01 / self.config.speed_multiplier)

        # Final results
        final = self._build_final_results()
        await self._broadcast({"type": "simulation_complete", "data": final})
        self.running = False
        return final

    def _run_evolution(self):
        """Run genetic evolution on all groups."""
        logger.info(f"Running evolution at candle {self.current_candle_idx}")
        for group in self.groups:
            result = self.evolver.evolve_group(group, self.exchange)
            logger.info(
                f"  {group.name}: best={result.best_fitness:.2f} "
                f"avg={result.avg_fitness:.2f} fired={result.agents_fired}"
            )

    def _build_state_snapshot(self, candle_idx: int) -> dict:
        """Build a state snapshot for broadcasting."""
        # Current prices
        prices = {s: round(p, 2) for s, p in self.exchange.current_prices.items()}

        # Group summaries
        group_summaries = []
        for group in self.groups:
            rankings = group.get_rankings()
            group_summaries.append({
                "name": group.name,
                "category": group.category,
                "total_agents": len(group.agents),
                "total_pnl": round(group.total_pnl, 2),
                "avg_win_rate": round(group.avg_win_rate * 100, 1),
                "rankings": rankings[:5],  # Top 5 per group
            })

        # Global leaderboard
        all_agents = []
        for group in self.groups:
            for agent in group.agents:
                summary = agent.get_summary()
                all_agents.append(summary)
        all_agents.sort(key=lambda x: x.get("total_pnl", 0), reverse=True)

        return {
            "type": "state_update",
            "data": {
                "candle_index": candle_idx,
                "total_candles": self.config.n_candles,
                "progress_pct": round(candle_idx / self.config.n_candles * 100, 1),
                "prices": prices,
                "groups": group_summaries,
                "global_leaderboard": all_agents[:20],  # Top 20 global
                "evolution_generation": self.evolver.generation,
            },
        }

    def _build_final_results(self) -> dict:
        """Build final simulation results."""
        all_agents = []
        for group in self.groups:
            for agent in group.agents:
                summary = agent.get_summary()
                summary["fitness"] = round(agent.fitness, 2)
                all_agents.append(summary)

        all_agents.sort(key=lambda x: x.get("total_pnl", 0), reverse=True)

        # Group rankings
        group_rankings = []
        for group in self.groups:
            group_rankings.append({
                "name": group.name,
                "category": group.category,
                "total_pnl": round(group.total_pnl, 2),
                "avg_win_rate": round(group.avg_win_rate * 100, 1),
                "total_agents": len(group.agents),
                "best_agent": group.get_best_agent().get_summary() if group.get_best_agent() else None,
            })
        group_rankings.sort(key=lambda x: x["total_pnl"], reverse=True)

        return {
            "global_leaderboard": all_agents,
            "group_rankings": group_rankings,
            "evolution_history": [r.to_dict() for r in self.evolver.history],
            "total_generations": self.evolver.generation,
            "simulation_candles": self.config.n_candles,
        }

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time state updates."""
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue):
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def _broadcast(self, data: dict):
        """Broadcast state to all subscribers."""
        message = json.dumps(data, default=str)
        for queue in self.subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass  # Skip if subscriber is slow

    def stop(self):
        self.running = False
