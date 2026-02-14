"""Momentum and mean reversion strategies.

References:
- "Momentum Strategies in Cryptocurrency Markets" - Grobys et al. (2020)
- "Mean Reversion in Cryptocurrency Markets" - Caporale & Plastun (2019)
- "Time Series Momentum in Crypto" - Liu & Tsyvinski (2021)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.models import Candle, Side, TradeSignal

from .base import BaseStrategy


class MomentumBreakoutStrategy(BaseStrategy):
    """Multi-timeframe momentum breakout strategy.

    Whale insight: Top futures traders identify breakouts using ATR-normalized
    price movement. When price moves >2 ATR in a session with increasing volume,
    they ride the momentum with trailing stops.

    Paper: "Time-Series Momentum" - Moskowitz, Ooi, Pedersen (2012)
    """

    name = "momentum_breakout"
    description = "ATR-normalized momentum breakout with volume surge"
    category = "momentum"

    def default_params(self) -> dict:
        return {
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "volume_surge_mult": 2.0,
            "momentum_lookback": 10,
            "min_candles": 30,
            "max_history": 500,
            "leverage": 5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.06,
        }

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        atr = self._calculate_atr(df, self.params["atr_period"])

        current_close = df["close"].iloc[-1]
        lookback = self.params["momentum_lookback"]
        lookback_close = df["close"].iloc[-lookback]
        price_change = current_close - lookback_close
        atr_move = abs(price_change) / atr.iloc[-1] if atr.iloc[-1] > 0 else 0

        # Volume surge check
        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        current_vol = df["volume"].iloc[-1]
        vol_surge = current_vol / avg_vol if avg_vol > 0 else 0

        if atr_move < self.params["atr_multiplier"]:
            return None
        if vol_surge < self.params["volume_surge_mult"]:
            return None

        confidence = min(0.5 + atr_move * 0.1 + vol_surge * 0.05, 0.9)

        if price_change > 0:
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"Momentum breakout UP ({atr_move:.1f} ATR, {vol_surge:.1f}x vol)",
            )
        else:
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"Momentum breakout DOWN ({atr_move:.1f} ATR, {vol_surge:.1f}x vol)",
            )


class MeanReversionStrategy(BaseStrategy):
    """Statistical mean reversion strategy.

    Whale insight: Market makers and statistical arb traders exploit short-term
    deviations from the mean. When price deviates > 2 standard deviations from
    a rolling mean, they take contrarian positions expecting reversion.

    Paper: "Mean Reversion in Bitcoin Markets" - Caporale & Plastun (2019)
    """

    name = "mean_reversion"
    description = "Z-score based mean reversion on price deviations"
    category = "statistical"

    def default_params(self) -> dict:
        return {
            "lookback_period": 30,
            "z_score_entry": 2.0,
            "z_score_exit": 0.5,
            "min_candles": 40,
            "max_history": 500,
            "leverage": 3,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.02,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        closes = df["close"]

        period = self.params["lookback_period"]
        rolling_mean = closes.rolling(window=period).mean()
        rolling_std = closes.rolling(window=period).std()

        z_score = (closes - rolling_mean) / rolling_std
        current_z = z_score.iloc[-1]
        prev_z = z_score.iloc[-2]

        entry_threshold = self.params["z_score_entry"]

        # Price is extremely high -> expect reversion down
        if current_z > entry_threshold and current_z < prev_z:  # Starting to revert
            confidence = min(0.5 + (current_z - entry_threshold) * 0.15, 0.85)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"Mean reversion SHORT (z={current_z:.2f})",
            )

        # Price is extremely low -> expect reversion up
        if current_z < -entry_threshold and current_z > prev_z:  # Starting to revert
            confidence = min(0.5 + (abs(current_z) - entry_threshold) * 0.15, 0.85)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"Mean reversion LONG (z={current_z:.2f})",
            )

        return None


class RSITrendMomentumStrategy(BaseStrategy):
    """RSI trend-following momentum strategy.

    Whale insight: Unlike mean reversion RSI, momentum traders use RSI
    differently. In strong trends, RSI stays in the 40-80 range (uptrend) or
    20-60 range (downtrend). Top traders enter when RSI bounces off the
    trend-zone floor.

    Paper: "RSI as Trend Indicator" - Constance Brown (2012)
    """

    name = "rsi_trend_momentum"
    description = "RSI trend-zone momentum following"
    category = "momentum"

    def default_params(self) -> dict:
        return {
            "rsi_period": 14,
            "trend_ema_period": 50,
            "uptrend_rsi_floor": 40,
            "downtrend_rsi_ceiling": 60,
            "min_candles": 55,
            "max_history": 500,
            "leverage": 4,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        closes = df["close"]

        # Calculate RSI
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.params["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.params["rsi_period"]).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        # Trend filter
        ema = closes.ewm(span=self.params["trend_ema_period"], adjust=False).mean()
        uptrend = closes.iloc[-1] > ema.iloc[-1]

        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        if uptrend:
            # Buy when RSI bounces off uptrend floor
            floor = self.params["uptrend_rsi_floor"]
            if prev_rsi <= floor + 5 and current_rsi > prev_rsi and current_rsi > floor:
                confidence = min(0.55 + (current_rsi - floor) * 0.01, 0.8)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.LONG,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"RSI trend bounce in uptrend (rsi={current_rsi:.1f})",
                )
        else:
            # Sell when RSI hits downtrend ceiling
            ceiling = self.params["downtrend_rsi_ceiling"]
            if prev_rsi >= ceiling - 5 and current_rsi < prev_rsi and current_rsi < ceiling:
                confidence = min(0.55 + (ceiling - current_rsi) * 0.01, 0.8)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.SHORT,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"RSI trend rejection in downtrend (rsi={current_rsi:.1f})",
                )

        return None
