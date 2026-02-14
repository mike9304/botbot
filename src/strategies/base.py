"""Base strategy interface and common utilities."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from src.core.models import Candle, Side, TimeFrame, TradeSignal


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    name: str = "base"
    description: str = ""
    category: str = "unknown"  # technical, onchain, sentiment, statistical, ml

    def __init__(self, params: Optional[dict] = None):
        defaults = self.default_params()
        if params:
            defaults.update(params)
        self.params = defaults
        self.candle_history: list[Candle] = []
        self.signals_generated: int = 0

    @abstractmethod
    def default_params(self) -> dict:
        """Return default parameters for this strategy."""
        return {}

    @abstractmethod
    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        """Analyze candles and return a trade signal if conditions are met."""
        pass

    def update(self, candle: Candle) -> Optional[TradeSignal]:
        """Update with new candle and check for signals."""
        self.candle_history.append(candle)
        max_history = self.params.get("max_history", 500)
        if len(self.candle_history) > max_history:
            self.candle_history = self.candle_history[-max_history:]
        if len(self.candle_history) < self.params.get("min_candles", 20):
            return None
        signal = self.analyze(self.candle_history)
        if signal:
            self.signals_generated += 1
        return signal

    def to_dataframe(self, candles: list[Candle]) -> pd.DataFrame:
        """Convert candle list to DataFrame for easier analysis."""
        data = {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
        return pd.DataFrame(data)

    def get_params(self) -> dict:
        return self.params.copy()

    def set_params(self, params: dict):
        self.params.update(params)

    def clone(self) -> BaseStrategy:
        """Create a copy of this strategy with same params."""
        return self.__class__(params=self.params.copy())


class HybridStrategy(BaseStrategy):
    """Combines multiple strategies with weighted voting."""

    name = "hybrid"
    description = "Weighted combination of multiple strategies"
    category = "hybrid"

    def __init__(self, strategies: list[tuple[BaseStrategy, float]], params: Optional[dict] = None):
        """
        Args:
            strategies: list of (strategy, weight) tuples
        """
        super().__init__(params)
        self.strategies = strategies
        total_weight = sum(w for _, w in strategies)
        self.strategies = [(s, w / total_weight) for s, w in strategies]

    def default_params(self) -> dict:
        return {
            "confidence_threshold": 0.6,
            "max_history": 500,
            "min_candles": 50,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        long_score = 0.0
        short_score = 0.0
        reasons = []

        for strategy, weight in self.strategies:
            signal = strategy.analyze(candles)
            if signal:
                if signal.side == Side.LONG:
                    long_score += signal.confidence * weight
                else:
                    short_score += signal.confidence * weight
                reasons.append(f"{strategy.name}:{signal.side.value}({signal.confidence:.2f})")

        threshold = self.params["confidence_threshold"]
        if long_score > threshold and long_score > short_score:
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=long_score,
                strategy_name=self.name,
                leverage=self._calc_leverage(long_score),
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                reason=" | ".join(reasons),
            )
        elif short_score > threshold and short_score > long_score:
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=short_score,
                strategy_name=self.name,
                leverage=self._calc_leverage(short_score),
                stop_loss_pct=0.02,
                take_profit_pct=0.04,
                reason=" | ".join(reasons),
            )
        return None

    def _calc_leverage(self, confidence: float) -> int:
        # Higher confidence -> higher leverage (capped)
        return min(int(confidence * 10) + 1, 20)
