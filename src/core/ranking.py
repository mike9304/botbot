"""Ranking and reward system for trading agents.

Provides:
- Daily / Weekly / Monthly performance rankings
- Prize distribution based on rankings
- Performance metrics aggregation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.agents.base_agent import AgentGroup, TradingAgent


@dataclass
class RankingEntry:
    rank: int
    agent_id: str
    agent_name: str
    group: str
    strategy: str
    total_pnl: float
    pnl_percent: float
    win_rate: float
    total_trades: int
    max_drawdown: float
    fitness: float
    prize: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "group": self.group,
            "strategy": self.strategy,
            "total_pnl": round(self.total_pnl, 2),
            "pnl_percent": round(self.pnl_percent, 2),
            "win_rate": round(self.win_rate * 100, 1),
            "total_trades": self.total_trades,
            "max_drawdown": round(self.max_drawdown * 100, 2),
            "fitness": round(self.fitness, 2),
            "prize": round(self.prize, 2),
        }


@dataclass
class PrizeConfig:
    """Prize pool configuration.

    Daily prize pool: 100 USDT
    Weekly prize pool: 500 USDT
    Monthly prize pool: 2000 USDT

    Distribution:
    1st: 30%, 2nd: 20%, 3rd: 15%, 4th: 10%, 5th: 8%,
    6th-10th: 3.4% each
    """

    daily_pool: float = 100.0
    weekly_pool: float = 500.0
    monthly_pool: float = 2000.0
    distribution: list[float] = field(
        default_factory=lambda: [0.30, 0.20, 0.15, 0.10, 0.08, 0.034, 0.034, 0.034, 0.034, 0.034]
    )


class RankingSystem:
    """Manages rankings and prize distribution."""

    def __init__(self, config: Optional[PrizeConfig] = None):
        self.config = config or PrizeConfig()
        self.daily_rankings: list[list[RankingEntry]] = []
        self.weekly_rankings: list[list[RankingEntry]] = []
        self.monthly_rankings: list[list[RankingEntry]] = []

    def calculate_rankings(
        self, groups: list[AgentGroup], period: str = "daily"
    ) -> list[RankingEntry]:
        """Calculate rankings across all groups."""
        all_entries = []
        for group in groups:
            for agent in group.agents:
                if not agent.account or not agent.active:
                    continue
                entry = RankingEntry(
                    rank=0,
                    agent_id=agent.id,
                    agent_name=agent.config.name,
                    group=agent.config.group,
                    strategy=agent.strategy.name,
                    total_pnl=agent.account.total_pnl,
                    pnl_percent=agent.account.pnl_percent,
                    win_rate=agent.account.win_rate,
                    total_trades=agent.account.total_trades,
                    max_drawdown=agent.account.max_drawdown,
                    fitness=agent.fitness,
                )
                all_entries.append(entry)

        # Sort by PnL (primary) and fitness (secondary)
        all_entries.sort(key=lambda e: (e.total_pnl, e.fitness), reverse=True)

        # Assign ranks
        for i, entry in enumerate(all_entries):
            entry.rank = i + 1

        # Distribute prizes
        pool = {
            "daily": self.config.daily_pool,
            "weekly": self.config.weekly_pool,
            "monthly": self.config.monthly_pool,
        }.get(period, self.config.daily_pool)

        for i, entry in enumerate(all_entries):
            if i < len(self.config.distribution):
                entry.prize = pool * self.config.distribution[i]

        # Store in history
        if period == "daily":
            self.daily_rankings.append(all_entries)
        elif period == "weekly":
            self.weekly_rankings.append(all_entries)
        elif period == "monthly":
            self.monthly_rankings.append(all_entries)

        return all_entries

    def get_group_rankings(self, groups: list[AgentGroup]) -> list[dict]:
        """Rank groups by total performance."""
        group_stats = []
        for group in groups:
            total_pnl = group.total_pnl
            avg_wr = group.avg_win_rate
            best = group.get_best_agent()
            group_stats.append({
                "name": group.name,
                "category": group.category,
                "total_pnl": round(total_pnl, 2),
                "avg_win_rate": round(avg_wr * 100, 1),
                "best_agent": best.config.name if best else "-",
                "best_agent_pnl": round(best.account.total_pnl, 2) if best and best.account else 0,
                "total_agents": len(group.agents),
            })
        group_stats.sort(key=lambda x: x["total_pnl"], reverse=True)
        for i, gs in enumerate(group_stats):
            gs["rank"] = i + 1
        return group_stats

    def get_latest_rankings(self, period: str = "daily") -> list[dict]:
        history = {
            "daily": self.daily_rankings,
            "weekly": self.weekly_rankings,
            "monthly": self.monthly_rankings,
        }.get(period, self.daily_rankings)

        if not history:
            return []
        return [e.to_dict() for e in history[-1]]
