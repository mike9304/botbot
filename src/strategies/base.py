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
        self._cached_df: Optional[pd.DataFrame] = None
        self._cached_df_len: int = 0

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
        """Convert candle list to DataFrame for easier analysis.

        Uses simple length-based cache to avoid recreating the full DataFrame
        on every analyze() call when the candle list has not changed.
        """
        n = len(candles)
        if self._cached_df is not None and self._cached_df_len == n:
            return self._cached_df
        data = {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        }
        self._cached_df = pd.DataFrame(data)
        self._cached_df_len = n
        return self._cached_df

    def get_params(self) -> dict:
        return self.params.copy()

    # Parameter constraints: keys that have known valid ranges.
    # Subclasses can override this to add strategy-specific constraints.
    PARAM_CONSTRAINTS: dict[str, tuple[float, float]] = {
        "leverage": (1, 125),
        "stop_loss_pct": (0.001, 0.5),
        "take_profit_pct": (0.001, 1.0),
        "confidence_threshold": (0.01, 1.0),
        "rsi_overbought": (50, 99),
        "rsi_oversold": (1, 50),
        "lookback_period": (3, 500),
        "z_score_entry": (0.1, 10.0),
    }

    def set_params(self, params: dict) -> None:
        """Update params with validation — clamp values to valid ranges."""
        for key, value in params.items():
            if key in self.PARAM_CONSTRAINTS and isinstance(value, (int, float)):
                lo, hi = self.PARAM_CONSTRAINTS[key]
                params[key] = type(value)(max(lo, min(hi, value)))
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
