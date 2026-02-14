"""Main simulation engine that ties everything together.

Runs the trading simulation with:
- Multiple agent groups (19 groups, ~70+ agents)
- Virtual Binance Futures exchange
- Genetic evolution (hire & fire system)
- Loser League (evolve worst agents as reverse indicators)
- AI Teacher & Risk Manager (LLM-powered oversight)
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
from src.agents.counter_agent import CounterIndicatorAgent, StrategyTypeCounterAgent
from src.agents.group_factory import DEFAULT_SYMBOLS, create_all_groups
from src.ai_teacher.llm_client import LLMConfig, LLMProvider
from src.ai_teacher.teacher import AITeacher, TeacherConfig
from src.core.exchange import VirtualExchange
from src.core.models import Candle, TradeSignal
from src.data.market_data import generate_multi_symbol_data
from src.evolution.genetic import EvolutionConfig, GeneticEvolver
from src.evolution.loser_evolution import InverseLoserAgent, LoserLeague

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    symbols: list[str] = field(default_factory=lambda: DEFAULT_SYMBOLS[:5])
    n_candles: int = 1000
    evolution_interval: int = 100  # Evolve every N candles
    risk_check_interval: int = 25  # Quick risk check every N candles
    speed_multiplier: float = 1.0  # Simulation speed (1.0 = real-time-ish)
    use_real_data: bool = False
    broadcast_interval: int = 1  # Broadcast state every N candles

    # AI Teacher settings
    ai_teacher_enabled: bool = True
    llm_provider: str = "offline"  # "offline", "deepseek", "openai", "anthropic", etc.
    llm_api_key: str = ""
    max_daily_llm_cost: float = 5.0


class SimulationEngine:
    """Main simulation engine with AI Teacher oversight."""

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

        # Counter-indicator tracking
        self.counter_agents: list[CounterIndicatorAgent] = []
        self.type_counter_agents: list[StrategyTypeCounterAgent] = []

        # Loser League system
        self.loser_league = LoserLeague()
        self.loser_group: Optional[AgentGroup] = None
        self.inverse_loser_agents: list[InverseLoserAgent] = []

        # AI Teacher
        self.teacher: Optional[AITeacher] = None
        if self.config.ai_teacher_enabled:
            self._setup_teacher()

    def _setup_teacher(self):
        """Initialize the AI Teacher with configured LLM provider."""
        provider_map = {
            "offline": LLMProvider.OFFLINE,
            "anthropic": LLMProvider.ANTHROPIC,
            "openai": LLMProvider.OPENAI,
            "deepseek": LLMProvider.DEEPSEEK,
            "google": LLMProvider.GOOGLE,
            "groq": LLMProvider.GROQ,
        }
        provider = provider_map.get(self.config.llm_provider, LLMProvider.OFFLINE)

        llm_config = LLMConfig(
            provider=provider,
            api_key=self.config.llm_api_key,
            max_cost_per_day_usd=self.config.max_daily_llm_cost,
        )
        teacher_config = TeacherConfig(
            llm_config=llm_config,
            evaluate_every_n_candles=self.config.evolution_interval,
            quick_check_every_n_candles=self.config.risk_check_interval,
        )
        self.teacher = AITeacher(teacher_config)
        logger.info(f"AI Teacher initialized: provider={provider.value}")

    def setup(self) -> None:
        """Initialize the simulation."""
        logger.info("Setting up simulation...")

        # Create agent groups
        self.groups = create_all_groups(self.exchange)
        total_agents = sum(len(g.agents) for g in self.groups)
        logger.info(f"Created {len(self.groups)} groups with {total_agents} total agents")

        # Collect special agent types
        for group in self.groups:
            for agent in group.agents:
                if isinstance(agent, CounterIndicatorAgent):
                    self.counter_agents.append(agent)
                elif isinstance(agent, StrategyTypeCounterAgent):
                    self.type_counter_agents.append(agent)
                elif isinstance(agent, InverseLoserAgent):
                    self.inverse_loser_agents.append(agent)

            # Find loser league group
            if group.category == "loser_league":
                self.loser_group = group

        if self.counter_agents or self.type_counter_agents:
            logger.info(
                f"Counter-indicator system: {len(self.counter_agents)} agent-level, "
                f"{len(self.type_counter_agents)} type-level counters active"
            )

        if self.loser_group:
            logger.info(
                f"Loser League: {len(self.loser_group.agents)} seed agents, "
                f"{len(self.inverse_loser_agents)} inverse loser agents"
            )

        if self.teacher:
            logger.info("AI Teacher: ACTIVE — will evaluate agents every evolution cycle")

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

                # Feed to all groups and collect signals
                for group in self.groups:
                    signals = group.on_candle(candle)

                    # Forward signals to counter-indicator agents
                    for agent, signal in signals:
                        current_price = self.exchange.current_prices.get(signal.symbol, 0)
                        if current_price <= 0:
                            continue

                        # Counter-indicator agents
                        for counter in self.counter_agents:
                            counter.on_other_agent_signal(
                                agent.id, agent.strategy.name,
                                signal, current_price,
                            )
                        for type_counter in self.type_counter_agents:
                            type_counter.on_strategy_type_signal(
                                agent.strategy.category, signal, current_price,
                            )

                        # Forward loser league signals to inverse loser agents
                        if group.category == "loser_league":
                            for inv_agent in self.inverse_loser_agents:
                                inv_agent.on_loser_signal(
                                    agent.id, signal, current_price,
                                )

            # Quick risk check (every 25 candles)
            if self.teacher and i > 0 and i % self.config.risk_check_interval == 0:
                await self.teacher.quick_risk_check(self.groups)

            # Periodic evolution + teacher evaluation (every 100 candles)
            if i > 0 and i % self.config.evolution_interval == 0:
                await self._run_evolution_with_teacher()

            # Broadcast state
            if i % self.config.broadcast_interval == 0:
                state = self._build_state_snapshot(i)
                await self._broadcast(state)

            # Simulation speed control
            if self.config.speed_multiplier < 100:
                await asyncio.sleep(0.01 / self.config.speed_multiplier)

        # Final results
        final = await self._build_final_results()
        await self._broadcast({"type": "simulation_complete", "data": final})
        self.running = False

        # Clean up AI Teacher
        if self.teacher:
            await self.teacher.close()

        return final

    async def _run_evolution_with_teacher(self):
        """Run evolution cycle with AI Teacher oversight."""
        logger.info(f"Running evolution at candle {self.current_candle_idx}")

        # 1. AI Teacher evaluation (before evolution)
        teacher_report = None
        if self.teacher:
            recent_candles = self._get_recent_candles(50)
            teacher_report = await self.teacher.evaluate_generation(
                self.groups, self.current_candle_idx, recent_candles
            )
            # Broadcast teacher report
            await self._broadcast({
                "type": "teacher_report",
                "data": {
                    "generation": teacher_report.generation,
                    "summary": teacher_report.overall_summary,
                    "market_regime": teacher_report.market_regime,
                    "risk_summary": teacher_report.risk_summary,
                    "llm_cost": teacher_report.llm_cost_usd,
                    "group_grades": {
                        g.group_name: g.overall_grade
                        for g in teacher_report.group_reports
                    },
                },
            })

        # 2. Regular genetic evolution
        for group in self.groups:
            result = self.evolver.evolve_group(group, self.exchange)
            logger.info(
                f"  {group.name}: best={result.best_fitness:.2f} "
                f"avg={result.avg_fitness:.2f} fired={result.agents_fired}"
            )

        # 3. Loser League evolution
        if self.loser_group:
            source_losers = self.loser_league.collect_losers(self.groups)
            if len(source_losers) >= 2:
                new_losers = self.loser_league.evolve_losers(
                    source_losers, self.exchange, self.loser_group,
                )
                if new_losers:
                    logger.info(
                        f"  Loser League: bred {len(new_losers)} new losers "
                        f"(gen {self.loser_league.generation})"
                    )

    def _get_recent_candles(self, n: int) -> dict[str, list[Candle]]:
        """Get recent candles for market regime detection."""
        result = {}
        for symbol, candles in self.market_data.items():
            start = max(0, self.current_candle_idx - n)
            end = self.current_candle_idx + 1
            result[symbol] = candles[start:end]
        return result

    def _build_state_snapshot(self, candle_idx: int) -> dict:
        """Build a state snapshot for broadcasting."""
        prices = {s: round(p, 2) for s, p in self.exchange.current_prices.items()}

        group_summaries = []
        for group in self.groups:
            rankings = group.get_rankings()
            summary = {
                "name": group.name,
                "category": group.category,
                "total_agents": len(group.agents),
                "total_pnl": round(group.total_pnl, 2),
                "avg_win_rate": round(group.avg_win_rate * 100, 1),
                "rankings": rankings[:5],
            }

            # Add teacher grade if available
            if self.teacher and self.teacher.reports:
                latest = self.teacher.get_latest_report()
                if latest:
                    for g in latest.group_reports:
                        if g.group_name == group.name:
                            summary["grade"] = g.overall_grade
                            summary["teacher_comment"] = g.teacher_comment

            group_summaries.append(summary)

        # Global leaderboard
        all_agents = []
        for group in self.groups:
            for agent in group.agents:
                summary = agent.get_summary()
                all_agents.append(summary)
        all_agents.sort(key=lambda x: x.get("total_pnl", 0), reverse=True)

        # Risk and teacher data
        extra = {}
        if self.teacher:
            latest = self.teacher.get_latest_report()
            if latest:
                extra["market_regime"] = latest.market_regime
                extra["risk_level"] = latest.risk_summary.get("risk_level", "LOW")
                extra["teacher_generation"] = latest.generation
            extra["llm_costs"] = self.teacher.get_cost_report()

        return {
            "type": "state_update",
            "data": {
                "candle_index": candle_idx,
                "total_candles": self.config.n_candles,
                "progress_pct": round(candle_idx / self.config.n_candles * 100, 1),
                "prices": prices,
                "groups": group_summaries,
                "global_leaderboard": all_agents[:20],
                "evolution_generation": self.evolver.generation,
                "loser_league_generation": self.loser_league.generation,
                **extra,
            },
        }

    async def _build_final_results(self) -> dict:
        """Build final simulation results."""
        all_agents = []
        for group in self.groups:
            for agent in group.agents:
                summary = agent.get_summary()
                summary["fitness"] = round(agent.fitness, 2)
                all_agents.append(summary)

        all_agents.sort(key=lambda x: x.get("total_pnl", 0), reverse=True)

        group_rankings = []
        for group in self.groups:
            group_data = {
                "name": group.name,
                "category": group.category,
                "total_pnl": round(group.total_pnl, 2),
                "avg_win_rate": round(group.avg_win_rate * 100, 1),
                "total_agents": len(group.agents),
                "best_agent": (
                    group.get_best_agent().get_summary() if group.get_best_agent() else None
                ),
            }
            # Add teacher grade
            if self.teacher and self.teacher.reports:
                latest = self.teacher.get_latest_report()
                if latest:
                    for g in latest.group_reports:
                        if g.group_name == group.name:
                            group_data["grade"] = g.overall_grade
                            group_data["teacher_comment"] = g.teacher_comment
            group_rankings.append(group_data)

        group_rankings.sort(key=lambda x: x["total_pnl"], reverse=True)

        result = {
            "global_leaderboard": all_agents,
            "group_rankings": group_rankings,
            "evolution_history": [r.to_dict() for r in self.evolver.history],
            "total_generations": self.evolver.generation,
            "loser_league_history": self.loser_league.history,
            "simulation_candles": self.config.n_candles,
        }

        # Add teacher reports
        if self.teacher:
            result["teacher_reports"] = [
                {
                    "generation": r.generation,
                    "market_regime": r.market_regime,
                    "summary": r.overall_summary,
                    "risk_summary": r.risk_summary,
                    "llm_cost": r.llm_cost_usd,
                }
                for r in self.teacher.reports
            ]
            result["total_llm_cost"] = self.teacher.get_cost_report()

        return result

    def subscribe(self) -> asyncio.Queue:
        """Subscribe to real-time state updates."""
        queue: asyncio.Queue = asyncio.Queue()
        self.subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        if queue in self.subscribers:
            self.subscribers.remove(queue)

    async def _broadcast(self, data: dict):
        """Broadcast state to all subscribers."""
        message = json.dumps(data, default=str)
        for queue in self.subscribers:
            try:
                queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

    def stop(self) -> None:
        self.running = False
