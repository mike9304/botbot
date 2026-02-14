"""Technical analysis strategies based on classic indicators.

References:
- "Technical Analysis of the Financial Markets" - John J. Murphy
- "A Multi-Strategy Approach to Trading" - Kakushadze & Serur (2018)
- Top whale traders commonly combine RSI divergence with volume confirmation
  and Bollinger Band breakouts for entry timing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.models import Candle, Side, TradeSignal

from .base import BaseStrategy


class RSIMACDStrategy(BaseStrategy):
    """RSI + MACD confluence strategy.

    Whale insight: Top traders use RSI divergences combined with MACD crossovers
    as primary entry signals. RSI overbought/oversold alone has low win rate,
    but when confirmed by MACD histogram direction change, win rate jumps to ~60%.
    """

    name = "rsi_macd"
    description = "RSI divergence + MACD crossover confluence"
    category = "technical"

    def default_params(self) -> dict:
        return {
            "rsi_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "min_candles": 50,
            "max_history": 500,
            "leverage": 5,
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.03,
        }

    def _calculate_rsi(self, closes: pd.Series, period: int = 14) -> pd.Series:
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.inf)
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, closes: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = closes.ewm(span=fast, adjust=False).mean()
        ema_slow = closes.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        closes = df["close"]

        rsi = self._calculate_rsi(closes, self.params["rsi_period"])
        macd_line, signal_line, histogram = self._calculate_macd(
            closes, self.params["macd_fast"], self.params["macd_slow"], self.params["macd_signal"]
        )

        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        current_hist = histogram.iloc[-1]
        prev_hist = histogram.iloc[-2]

        # LONG: RSI coming out of oversold + MACD histogram turning positive
        if (
            prev_rsi < self.params["rsi_oversold"]
            and current_rsi > self.params["rsi_oversold"]
            and prev_hist < 0
            and current_hist > 0
        ):
            confidence = min((self.params["rsi_oversold"] - prev_rsi) / 20 + 0.5, 0.95)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"RSI oversold recovery ({prev_rsi:.1f}->{current_rsi:.1f}) + MACD bullish cross",
            )

        # SHORT: RSI coming out of overbought + MACD histogram turning negative
        if (
            prev_rsi > self.params["rsi_overbought"]
            and current_rsi < self.params["rsi_overbought"]
            and prev_hist > 0
            and current_hist < 0
        ):
            confidence = min((prev_rsi - self.params["rsi_overbought"]) / 20 + 0.5, 0.95)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"RSI overbought reversal ({prev_rsi:.1f}->{current_rsi:.1f}) + MACD bearish cross",
            )

        return None


class BollingerBreakoutStrategy(BaseStrategy):
    """Bollinger Bands breakout with volume confirmation.

    Whale insight: Smart money enters on Bollinger Band squeezes (low bandwidth)
    followed by breakouts with above-average volume. The squeeze indicates
    consolidation, and the breakout direction with volume confirms the move.
    """

    name = "bollinger_breakout"
    description = "Bollinger Band squeeze breakout with volume confirmation"
    category = "technical"

    def default_params(self) -> dict:
        return {
            "bb_period": 20,
            "bb_std": 2.0,
            "volume_multiplier": 1.5,
            "squeeze_threshold": 0.03,  # Bandwidth threshold for squeeze
            "min_candles": 30,
            "max_history": 500,
            "leverage": 3,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        closes = df["close"]
        volumes = df["volume"]

        period = self.params["bb_period"]
        std_mult = self.params["bb_std"]

        sma = closes.rolling(window=period).mean()
        std = closes.rolling(window=period).std()
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        bandwidth = (upper - lower) / sma

        avg_volume = volumes.rolling(window=period).mean()

        current_close = closes.iloc[-1]
        prev_close = closes.iloc[-2]
        current_bw = bandwidth.iloc[-1]
        prev_bw = bandwidth.iloc[-2]
        current_vol = volumes.iloc[-1]
        avg_vol = avg_volume.iloc[-1]

        # Check for squeeze (low bandwidth)
        was_squeeze = prev_bw < self.params["squeeze_threshold"]
        vol_confirmed = current_vol > avg_vol * self.params["volume_multiplier"]

        # Breakout above upper band after squeeze
        if was_squeeze and current_close > upper.iloc[-1] and vol_confirmed:
            confidence = min(0.6 + (current_vol / avg_vol - 1) * 0.1, 0.9)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"BB squeeze breakout UP (bw={current_bw:.4f}, vol={current_vol/avg_vol:.1f}x)",
            )

        # Breakout below lower band after squeeze
        if was_squeeze and current_close < lower.iloc[-1] and vol_confirmed:
            confidence = min(0.6 + (current_vol / avg_vol - 1) * 0.1, 0.9)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"BB squeeze breakout DOWN (bw={current_bw:.4f}, vol={current_vol/avg_vol:.1f}x)",
            )

        return None


class EMATripleCrossStrategy(BaseStrategy):
    """Triple EMA crossover with trend filter.

    Whale insight: Institutional traders use EMA 9/21/55 alignment as a trend
    confirmation tool. When all three align (golden cross), they enter on
    pullbacks to the 21 EMA. This is one of the most consistent whale patterns.
    """

    name = "ema_triple_cross"
    description = "Triple EMA (9/21/55) crossover with trend alignment"
    category = "technical"

    def default_params(self) -> dict:
        return {
            "ema_fast": 9,
            "ema_mid": 21,
            "ema_slow": 55,
            "min_candles": 60,
            "max_history": 500,
            "leverage": 3,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        closes = df["close"]

        ema_fast = closes.ewm(span=self.params["ema_fast"], adjust=False).mean()
        ema_mid = closes.ewm(span=self.params["ema_mid"], adjust=False).mean()
        ema_slow = closes.ewm(span=self.params["ema_slow"], adjust=False).mean()

        # Current alignment
        fast_above_mid = ema_fast.iloc[-1] > ema_mid.iloc[-1]
        mid_above_slow = ema_mid.iloc[-1] > ema_slow.iloc[-1]
        prev_fast_above_mid = ema_fast.iloc[-2] > ema_mid.iloc[-2]

        # Bullish alignment just formed
        if fast_above_mid and mid_above_slow and not prev_fast_above_mid:
            trend_strength = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
            confidence = min(0.5 + abs(trend_strength) * 10, 0.85)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"EMA bullish alignment (9>{self.params['ema_mid']}>{self.params['ema_slow']})",
            )

        # Bearish alignment
        fast_below_mid = not fast_above_mid
        mid_below_slow = not mid_above_slow
        prev_fast_below_mid = not prev_fast_above_mid

        if fast_below_mid and mid_below_slow and not prev_fast_below_mid:
            trend_strength = (ema_slow.iloc[-1] - ema_fast.iloc[-1]) / ema_slow.iloc[-1]
            confidence = min(0.5 + abs(trend_strength) * 10, 0.85)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"EMA bearish alignment (9<{self.params['ema_mid']}<{self.params['ema_slow']})",
            )

        return None


class FibonacciRetracementStrategy(BaseStrategy):
    """Fibonacci retracement entry strategy.

    Whale insight: Top traders identify swing highs/lows, then place limit orders
    at key Fibonacci levels (0.382, 0.5, 0.618). The 0.618 level (golden ratio)
    is the most commonly used by institutional traders.
    """

    name = "fibonacci_retracement"
    description = "Fibonacci level bounce entries at key retracement zones"
    category = "technical"

    FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786]

    def default_params(self) -> dict:
        return {
            "swing_lookback": 50,
            "tolerance": 0.005,  # 0.5% tolerance around fib levels
            "min_candles": 60,
            "max_history": 500,
            "leverage": 4,
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.03,
        }

    def _find_swing_points(self, df: pd.DataFrame) -> tuple[float, float]:
        lookback = self.params["swing_lookback"]
        recent = df.tail(lookback)
        swing_high = recent["high"].max()
        swing_low = recent["low"].min()
        return swing_high, swing_low

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        swing_high, swing_low = self._find_swing_points(df)
        current_close = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        diff = swing_high - swing_low
        if diff <= 0:
            return None

        # Determine trend direction
        mid = (swing_high + swing_low) / 2
        uptrend = current_close > mid

        tolerance = self.params["tolerance"]

        for fib_level in self.FIB_LEVELS:
            if uptrend:
                # In uptrend, look for bounces at retracement levels from swing low
                fib_price = swing_high - diff * fib_level
                near_fib = abs(current_close - fib_price) / fib_price < tolerance
                bouncing_up = current_close > prev_close

                if near_fib and bouncing_up:
                    confidence = 0.5 + (1 - fib_level) * 0.4  # Higher fib = higher confidence
                    return TradeSignal(
                        symbol=candles[-1].symbol,
                        side=Side.LONG,
                        confidence=min(confidence, 0.85),
                        strategy_name=self.name,
                        leverage=self.params["leverage"],
                        stop_loss_pct=self.params["stop_loss_pct"],
                        take_profit_pct=self.params["take_profit_pct"],
                        reason=f"Fib {fib_level} bounce in uptrend @ {fib_price:.2f}",
                    )
            else:
                # In downtrend, look for rejections at retracement levels from swing high
                fib_price = swing_low + diff * fib_level
                near_fib = abs(current_close - fib_price) / fib_price < tolerance
                rejecting_down = current_close < prev_close

                if near_fib and rejecting_down:
                    confidence = 0.5 + (1 - fib_level) * 0.4
                    return TradeSignal(
                        symbol=candles[-1].symbol,
                        side=Side.SHORT,
                        confidence=min(confidence, 0.85),
                        strategy_name=self.name,
                        leverage=self.params["leverage"],
                        stop_loss_pct=self.params["stop_loss_pct"],
                        take_profit_pct=self.params["take_profit_pct"],
                        reason=f"Fib {fib_level} rejection in downtrend @ {fib_price:.2f}",
                    )

        return None


class VolumeProfileStrategy(BaseStrategy):
    """Volume Profile / VWAP-based strategy.

    Whale insight: Institutional traders heavily rely on VWAP (Volume Weighted
    Average Price) and volume profile nodes. They buy at VWAP support in uptrends
    and sell at VWAP resistance in downtrends. High volume nodes act as
    support/resistance.
    """

    name = "volume_profile"
    description = "VWAP and volume profile node analysis"
    category = "technical"

    def default_params(self) -> dict:
        return {
            "vwap_period": 20,
            "volume_node_bins": 50,
            "proximity_threshold": 0.003,  # 0.3% from VWAP
            "min_candles": 30,
            "max_history": 500,
            "leverage": 3,
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.03,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)

        # Calculate VWAP
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cumulative_tp_vol = (typical_price * df["volume"]).rolling(
            window=self.params["vwap_period"]
        ).sum()
        cumulative_vol = df["volume"].rolling(window=self.params["vwap_period"]).sum()
        vwap = cumulative_tp_vol / cumulative_vol

        current_close = df["close"].iloc[-1]
        current_vwap = vwap.iloc[-1]
        prev_close = df["close"].iloc[-2]
        prev_vwap = vwap.iloc[-2]

        proximity = abs(current_close - current_vwap) / current_vwap

        if proximity > self.params["proximity_threshold"]:
            return None

        # Price bouncing off VWAP from below -> LONG
        if prev_close < prev_vwap and current_close > current_vwap:
            vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
            confidence = min(0.55 + vol_ratio * 0.1, 0.85)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"VWAP bounce bullish (vwap={current_vwap:.2f})",
            )

        # Price rejecting VWAP from above -> SHORT
        if prev_close > prev_vwap and current_close < current_vwap:
            vol_ratio = df["volume"].iloc[-1] / df["volume"].rolling(20).mean().iloc[-1]
            confidence = min(0.55 + vol_ratio * 0.1, 0.85)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"VWAP rejection bearish (vwap={current_vwap:.2f})",
            )

        return None
