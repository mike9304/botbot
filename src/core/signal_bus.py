"""Signal Bus — decouples signal producers from signal observers.

Instead of SimulationEngine hard-coding isinstance() checks for each meta-agent type,
agents self-register as signal observers. When any agent generates a signal,
the bus routes it to all registered observers.

This eliminates:
- isinstance() checks in SimulationEngine
- Direct coupling between engine and counter/loser agents
- Need to modify engine when adding new observer types

Usage:
    bus = SignalBus()
    bus.register_signal_observer(counter_agent)
    bus.register_fire_observer(counter_agent)
    # ... later, in simulation loop:
    bus.publish_signal(agent_id, strategy_name, strategy_category, signal, price)
    bus.publish_agent_fired(fired_agent_id)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

from src.core.models import TradeSignal

logger = logging.getLogger(__name__)


class SignalObserver(ABC):
    """Interface for agents that observe other agents' signals."""

    @abstractmethod
    def on_signal(
        self,
        source_agent_id: str,
        strategy_name: str,
        strategy_category: str,
        signal: TradeSignal,
        current_price: float,
    ) -> None:
        """Called when any agent generates a trading signal."""
        ...


class FireObserver(ABC):
    """Interface for agents that need to know when other agents are fired."""

    @abstractmethod
    def on_agent_fired(self, fired_agent_id: str) -> None:
        """Called when an agent is removed during evolution."""
        ...


class LoserSignalObserver(ABC):
    """Interface for agents that observe loser league signals specifically."""

    @abstractmethod
    def on_loser_signal(
        self,
        loser_id: str,
        signal: TradeSignal,
        current_price: float,
    ) -> None:
        """Called when a loser league agent generates a signal."""
        ...


class SignalBus:
    """Central event bus for routing signals between agents.

    Observers register themselves; the simulation engine just calls publish().
    """

    def __init__(self) -> None:
        self._signal_observers: list[SignalObserver] = []
        self._fire_observers: list[FireObserver] = []
        self._loser_signal_observers: list[LoserSignalObserver] = []

    def register_signal_observer(self, observer: SignalObserver) -> None:
        self._signal_observers.append(observer)

    def register_fire_observer(self, observer: FireObserver) -> None:
        self._fire_observers.append(observer)

    def register_loser_signal_observer(self, observer: LoserSignalObserver) -> None:
        self._loser_signal_observers.append(observer)

    def publish_signal(
        self,
        source_agent_id: str,
        strategy_name: str,
        strategy_category: str,
        signal: TradeSignal,
        current_price: float,
    ) -> None:
        """Notify all signal observers of a new signal."""
        for observer in self._signal_observers:
            observer.on_signal(
                source_agent_id, strategy_name, strategy_category,
                signal, current_price,
            )

    def publish_loser_signal(
        self,
        loser_id: str,
        signal: TradeSignal,
        current_price: float,
    ) -> None:
        """Notify all loser signal observers."""
        for observer in self._loser_signal_observers:
            observer.on_loser_signal(loser_id, signal, current_price)

    def publish_agent_fired(self, fired_agent_id: str) -> None:
        """Notify all fire observers when an agent is removed."""
        for observer in self._fire_observers:
            observer.on_agent_fired(fired_agent_id)

    @property
    def stats(self) -> dict:
        return {
            "signal_observers": len(self._signal_observers),
            "fire_observers": len(self._fire_observers),
            "loser_signal_observers": len(self._loser_signal_observers),
        }
