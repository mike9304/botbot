"""Counter-Indicator Strategy System.

Core idea: Some strategies/agents are CONSISTENTLY WRONG. Instead of discarding
them, we can use them as valuable "reverse signals" — when they say BUY, we SELL.

This is mathematically sound: an agent with 30% win rate is actually a
GREAT indicator — just inverted. A perfectly random agent has 50% win rate.
An agent that consistently loses has found a pattern, just backwards.

The Counter-Indicator system:
1. Monitors all other agents' signals in real-time
2. Tracks each agent's historical accuracy
3. Identifies "reliable losers" (agents with <40% win rate over 20+ trades)
4. When a reliable loser generates a signal, take the OPPOSITE trade
5. Continuously learns: if a former loser starts winning, stop inverting them

References:
- "Contrarian Investment Strategies" - David Dreman (1998)
- "The Wisdom of Crowds... Inverted" - concept from multi-agent systems research
- "Using Losing Strategies to Win" - game theory / information aggregation
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.core.models import Candle, Side, TradeSignal

from .base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class AgentTracker:
    """Tracks another agent's signal performance."""

    agent_id: str
    strategy_name: str
    total_signals: int = 0
    correct_signals: int = 0
    wrong_signals: int = 0
    recent_results: deque = field(default_factory=lambda: deque(maxlen=50))
    # Track their signals and what happened after
    pending_signal: Optional[TradeSignal] = None
    pending_price: float = 0.0

    @property
    def accuracy(self) -> float:
        total = self.correct_signals + self.wrong_signals
        if total == 0:
            return 0.5  # Unknown = assume random
        return self.correct_signals / total

    @property
    def is_reliable_loser(self) -> bool:
        """An agent is a reliable loser if they have enough trades and low accuracy."""
        total = self.correct_signals + self.wrong_signals
        return total >= 10 and self.accuracy < 0.40

    @property
    def is_reliable_winner(self) -> bool:
        total = self.correct_signals + self.wrong_signals
        return total >= 10 and self.accuracy > 0.60

    @property
    def inversion_confidence(self) -> float:
        """How confident we are in inverting this agent's signals.

        Lower accuracy → higher inversion confidence.
        30% accuracy → 0.70 confidence in opposite.
        20% accuracy → 0.80 confidence in opposite.
        """
        if not self.is_reliable_loser:
            return 0.0
        return 1.0 - self.accuracy

    def record_result(self, was_correct: bool):
        if was_correct:
            self.correct_signals += 1
        else:
            self.wrong_signals += 1
        self.recent_results.append(was_correct)


class CounterIndicatorStrategy(BaseStrategy):
    """Inverts signals from consistently losing agents/strategies.

    This strategy doesn't analyze price directly. Instead, it:
    1. Collects signals from other agents
    2. Evaluates their track record
    3. Inverts signals from agents with <40% accuracy

    The counter-indicator agent gets SMARTER over time as it learns
    which agents to invert and which to follow.
    """

    name = "counter_indicator"
    description = "Invert signals from consistently wrong agents"
    category = "counter_indicator"

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.tracked_agents: dict[str, AgentTracker] = {}
        self.pending_evaluations: list[tuple[str, TradeSignal, float]] = []

    def default_params(self) -> dict:
        return {
            "min_trades_for_tracking": 10,
            "loser_accuracy_threshold": 0.40,
            "min_inversion_confidence": 0.55,
            "eval_candles_ahead": 10,  # Evaluate signal after N candles
            "max_history": 500,
            "min_candles": 1,
            "leverage": 5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }

    def register_signal(self, agent_id: str, strategy_name: str, signal: TradeSignal, price: float):
        """Register a signal from another agent for tracking."""
        if agent_id not in self.tracked_agents:
            self.tracked_agents[agent_id] = AgentTracker(
                agent_id=agent_id, strategy_name=strategy_name
            )

        tracker = self.tracked_agents[agent_id]
        tracker.total_signals += 1

        # If there's a pending signal, evaluate it
        if tracker.pending_signal and tracker.pending_price > 0:
            self._evaluate_pending(tracker, price)

        # Store new signal for future evaluation
        tracker.pending_signal = signal
        tracker.pending_price = price

    def _evaluate_pending(self, tracker: AgentTracker, current_price: float):
        """Evaluate if a previous signal was correct."""
        signal = tracker.pending_signal
        if not signal:
            return

        entry_price = tracker.pending_price
        if entry_price <= 0:
            return

        price_change = (current_price - entry_price) / entry_price

        if signal.side == Side.LONG:
            was_correct = price_change > 0.005  # At least 0.5% profit
        else:
            was_correct = price_change < -0.005

        tracker.record_result(was_correct)
        tracker.pending_signal = None

    def get_inversion_candidates(self) -> list[AgentTracker]:
        """Get agents whose signals should be inverted."""
        candidates = []
        for tracker in self.tracked_agents.values():
            if tracker.is_reliable_loser:
                candidates.append(tracker)
        return sorted(candidates, key=lambda t: t.accuracy)  # Worst first

    def generate_counter_signal(
        self, source_agent_id: str, source_signal: TradeSignal
    ) -> Optional[TradeSignal]:
        """Generate an inverted signal if the source agent is a reliable loser."""
        tracker = self.tracked_agents.get(source_agent_id)
        if not tracker or not tracker.is_reliable_loser:
            return None

        inv_confidence = tracker.inversion_confidence
        if inv_confidence < self.params["min_inversion_confidence"]:
            return None

        # INVERT the signal
        inverted_side = Side.SHORT if source_signal.side == Side.LONG else Side.LONG

        return TradeSignal(
            symbol=source_signal.symbol,
            side=inverted_side,
            confidence=inv_confidence,
            strategy_name=self.name,
            leverage=self.params["leverage"],
            stop_loss_pct=self.params["stop_loss_pct"],
            take_profit_pct=self.params["take_profit_pct"],
            reason=(
                f"COUNTER-INDICATOR: Inverted {tracker.agent_id} "
                f"(accuracy={tracker.accuracy:.0%}, inverted→{inverted_side.value})"
            ),
        )

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        """Not used directly - counter signals are generated via generate_counter_signal."""
        return None

    def get_tracking_report(self) -> list[dict]:
        """Get report of all tracked agents and their accuracy."""
        report = []
        for tracker in self.tracked_agents.values():
            total = tracker.correct_signals + tracker.wrong_signals
            report.append({
                "agent_id": tracker.agent_id,
                "strategy": tracker.strategy_name,
                "total_signals": tracker.total_signals,
                "evaluated": total,
                "accuracy": round(tracker.accuracy * 100, 1),
                "is_reliable_loser": tracker.is_reliable_loser,
                "inversion_confidence": round(tracker.inversion_confidence * 100, 1),
            })
        report.sort(key=lambda x: x["accuracy"])
        return report


class StrategyTypeCounterIndicatorStrategy(BaseStrategy):
    """Counter-indicator at the STRATEGY TYPE level.

    Instead of tracking individual agents, this tracks entire strategy categories
    and learns which TYPES of strategies are losing in the current market regime.

    Example: If all momentum strategies are losing → invert all momentum signals.
    This captures regime changes faster than individual agent tracking.

    The insight: Different strategy types fail in different market conditions:
    - Momentum fails in ranging markets
    - Mean reversion fails in trending markets
    - Breakout strategies fail in choppy markets

    By detecting which type is currently failing, we can systematically
    trade against that type's signals.
    """

    name = "strategy_type_counter"
    description = "Track losing strategy TYPES and invert their signals"
    category = "counter_indicator"

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.type_trackers: dict[str, AgentTracker] = {}

    def default_params(self) -> dict:
        return {
            "min_trades_for_tracking": 8,
            "loser_accuracy_threshold": 0.40,
            "min_inversion_confidence": 0.55,
            "max_history": 500,
            "min_candles": 1,
            "leverage": 4,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }

    def register_strategy_type_signal(
        self, strategy_type: str, signal: TradeSignal, price: float
    ):
        """Register a signal under a strategy type category."""
        if strategy_type not in self.type_trackers:
            self.type_trackers[strategy_type] = AgentTracker(
                agent_id=strategy_type, strategy_name=strategy_type
            )

        tracker = self.type_trackers[strategy_type]
        tracker.total_signals += 1

        if tracker.pending_signal and tracker.pending_price > 0:
            entry = tracker.pending_price
            change = (price - entry) / entry
            was_correct = (
                (change > 0.005) if tracker.pending_signal.side == Side.LONG else (change < -0.005)
            )
            tracker.record_result(was_correct)

        tracker.pending_signal = signal
        tracker.pending_price = price

    def generate_counter_signal(
        self, strategy_type: str, source_signal: TradeSignal
    ) -> Optional[TradeSignal]:
        tracker = self.type_trackers.get(strategy_type)
        if not tracker or not tracker.is_reliable_loser:
            return None

        inv_confidence = tracker.inversion_confidence
        if inv_confidence < self.params["min_inversion_confidence"]:
            return None

        inverted_side = Side.SHORT if source_signal.side == Side.LONG else Side.LONG

        return TradeSignal(
            symbol=source_signal.symbol,
            side=inverted_side,
            confidence=inv_confidence,
            strategy_name=self.name,
            leverage=self.params["leverage"],
            stop_loss_pct=self.params["stop_loss_pct"],
            take_profit_pct=self.params["take_profit_pct"],
            reason=(
                f"TYPE COUNTER: {strategy_type} type accuracy={tracker.accuracy:.0%} "
                f"→ invert to {inverted_side.value}"
            ),
        )

    def get_losing_strategy_types(self) -> list[str]:
        """Get list of strategy types that are currently losing."""
        losers = []
        for stype, tracker in self.type_trackers.items():
            if tracker.is_reliable_loser:
                losers.append(stype)
        return losers

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        return None

    def get_type_report(self) -> list[dict]:
        report = []
        for stype, tracker in self.type_trackers.items():
            total = tracker.correct_signals + tracker.wrong_signals
            report.append({
                "strategy_type": stype,
                "total_signals": tracker.total_signals,
                "evaluated": total,
                "accuracy": round(tracker.accuracy * 100, 1),
                "is_losing": tracker.is_reliable_loser,
                "recent_trend": list(tracker.recent_results)[-10:] if tracker.recent_results else [],
            })
        report.sort(key=lambda x: x["accuracy"])
        return report
