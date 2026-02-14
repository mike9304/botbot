"""Counter-Indicator Agent that monitors other agents and inverts loser signals.

This agent acts as a meta-learner:
- It observes all other agents' signals and outcomes
- Identifies "reliable losers" (consistently wrong agents)
- Inverts their signals to generate profitable trades
- Continuously updates its model of who to invert

Think of it as a "talent scout in reverse" — finding the worst players
and betting against everything they do.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.core.exchange import VirtualExchange
from src.core.models import Candle, Side, TradeSignal
from src.core.signal_bus import FireObserver, SignalObserver
from src.strategies.counter_indicator import (
    CounterIndicatorStrategy,
    StrategyTypeCounterIndicatorStrategy,
)

from .base_agent import AgentConfig, TradingAgent

logger = logging.getLogger(__name__)


class CounterIndicatorAgent(TradingAgent, SignalObserver, FireObserver):
    """An agent that monitors other agents and inverts signals from consistent losers.

    Implements SignalObserver and FireObserver so it self-registers with the SignalBus
    instead of requiring isinstance() checks in SimulationEngine.

    Flow:
    1. on_signal(): Called via SignalBus when any agent generates a signal
    2. The CounterIndicatorStrategy tracks that agent's accuracy
    3. If the source agent is a "reliable loser", generate an inverted signal
    4. Execute the inverted signal on the exchange
    """

    def __init__(self, config: AgentConfig, exchange: VirtualExchange):
        super().__init__(config, exchange)
        self.counter_strategy: CounterIndicatorStrategy = config.strategy  # type: ignore
        self._observed_count = 0
        self._inverted_count = 0
        self._feedback_loop_resets = 0
        self._current_regime = "unknown"

    def on_signal(
        self,
        source_agent_id: str,
        strategy_name: str,
        strategy_category: str,
        signal: TradeSignal,
        current_price: float,
    ) -> None:
        """SignalObserver interface — called via SignalBus when any agent signals."""
        self.on_other_agent_signal(source_agent_id, strategy_name, signal, current_price)

    def on_other_agent_signal(
        self,
        source_agent_id: str,
        source_strategy_name: str,
        signal: TradeSignal,
        current_price: float,
    ):
        """Process a signal from another agent."""
        if not self.active:
            return

        self._observed_count += 1

        # Register the signal for tracking
        self.counter_strategy.register_signal(
            source_agent_id, source_strategy_name, signal, current_price
        )

        # Check if we should invert this signal (regime-dependent)
        counter_signal = self.counter_strategy.generate_counter_signal(
            source_agent_id, signal, market_regime=self._current_regime
        )

        if counter_signal and self._can_trade(self.account, counter_signal):
            order = self.exchange.execute_signal(self.id, counter_signal)
            if order:
                self._inverted_count += 1
                self.daily_trades += 1
                self.candles_since_last_trade = 0
                logger.info(
                    f"COUNTER-INDICATOR {self.id}: Inverted {source_agent_id}'s "
                    f"{signal.side.value} → {counter_signal.side.value} on {signal.symbol}"
                )

    def on_candle(self, candle: Candle):
        """Process candle - mainly for updating price and existing positions."""
        if not self.active:
            return
        self.candles_since_last_trade += 1

    def get_summary(self) -> dict:
        base = super().get_summary()
        base.update({
            "observed_signals": self._observed_count,
            "inverted_signals": self._inverted_count,
            "tracked_agents": len(self.counter_strategy.tracked_agents),
            "reliable_losers": len(self.counter_strategy.get_inversion_candidates()),
        })
        return base

    def set_market_regime(self, regime: str):
        """Update current market regime for regime-dependent inversion."""
        self._current_regime = regime

    def on_agent_fired(self, fired_agent_id: str):
        """Handle feedback loop when an inverted agent gets fired.

        From Bao & Liu (2019): When counter-indicators succeed, the agents
        they invert get replaced, erasing the counter-indicator's edge.
        Halve the track record so the agent quickly re-learns from the replacement.
        """
        tracker = self.counter_strategy.tracked_agents.get(fired_agent_id)
        if tracker:
            logger.info(
                f"FEEDBACK LOOP: {fired_agent_id} fired "
                f"(was accuracy={tracker.accuracy:.0%}). Halving tracker."
            )
            tracker.correct_signals = tracker.correct_signals // 2
            tracker.wrong_signals = tracker.wrong_signals // 2
            tracker.pending_signal = None
            self._feedback_loop_resets += 1

    def get_tracking_report(self) -> list[dict]:
        return self.counter_strategy.get_tracking_report()


class StrategyTypeCounterAgent(TradingAgent, SignalObserver):
    """Counter agent at the strategy TYPE level.

    Implements SignalObserver so it self-registers with the SignalBus.

    Instead of tracking individual agents, this tracks strategy categories.
    When an entire category (e.g., "momentum") is losing, it inverts ALL
    signals from that category.

    This is more aggressive but captures regime changes faster.
    """

    def __init__(self, config: AgentConfig, exchange: VirtualExchange):
        super().__init__(config, exchange)
        self.type_counter: StrategyTypeCounterIndicatorStrategy = config.strategy  # type: ignore
        self._observed_count = 0
        self._inverted_count = 0
        self._current_regime = "unknown"

    def on_signal(
        self,
        source_agent_id: str,
        strategy_name: str,
        strategy_category: str,
        signal: TradeSignal,
        current_price: float,
    ) -> None:
        """SignalObserver interface — routes to strategy type tracking."""
        self.on_strategy_type_signal(strategy_category, signal, current_price)

    def on_strategy_type_signal(
        self,
        strategy_type: str,
        signal: TradeSignal,
        current_price: float,
    ):
        """Process a signal categorized by strategy type."""
        if not self.active:
            return

        self._observed_count += 1

        self.type_counter.register_strategy_type_signal(
            strategy_type, signal, current_price
        )

        counter_signal = self.type_counter.generate_counter_signal(
            strategy_type, signal, market_regime=self._current_regime
        )

        if counter_signal and self._can_trade(self.account, counter_signal):
            order = self.exchange.execute_signal(self.id, counter_signal)
            if order:
                self._inverted_count += 1
                self.daily_trades += 1
                self.candles_since_last_trade = 0
                logger.info(
                    f"TYPE-COUNTER {self.id}: Inverted {strategy_type} "
                    f"{signal.side.value} → {counter_signal.side.value} on {signal.symbol}"
                )

    def on_candle(self, candle: Candle):
        if not self.active:
            return
        self.candles_since_last_trade += 1

    def set_market_regime(self, regime: str):
        self._current_regime = regime

    def get_summary(self) -> dict:
        base = super().get_summary()
        base.update({
            "observed_signals": self._observed_count,
            "inverted_signals": self._inverted_count,
            "tracked_types": len(self.type_counter.type_trackers),
            "losing_types": self.type_counter.get_losing_strategy_types(),
        })
        return base
