"""Contrarian and crowd psychology strategies.

Core philosophy: "Be fearful when others are greedy, be greedy when others are fearful."
                                                        - Warren Buffett

These strategies analyze crowd/retail trader psychology indicators and trade
in the OPPOSITE direction. When the crowd is euphoric, we short. When the
crowd is in maximum fear, we buy.

References:
- "Contrarian Investment, Extrapolation, and Risk" - Lakonishok, Shleifer, Vishny (1994)
- "Fear and Greed Index" methodology (CNN/Alternative.me)
- "Sentiment Analysis for Cryptocurrency Trading" - Kraaijeveld & De Smedt (2020)
- "The Disposition Effect in Securities Trading" - Shefrin & Statman (1985)
- "Noise Trader Risk in Financial Markets" - De Long, Shleifer, Summers, Waldmann (1990)

Whale insight: The top 1% of crypto traders consistently fade retail sentiment
extremes. When funding rates spike positive (everyone is long), whales short.
When social media panic peaks, whales accumulate. This is arguably the single
highest-alpha signal in crypto markets.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.models import Candle, Side, TradeSignal

from .base import BaseStrategy


class FearGreedContrarianStrategy(BaseStrategy):
    """Synthetic Fear & Greed Index contrarian strategy.

    Constructs a fear/greed index from price action and trades against it:
    - Extreme Greed (>80) → SHORT (crowd is euphoric, top is near)
    - Extreme Fear (<20) → LONG (crowd is panicking, bottom is near)

    The index is built from:
    1. Price momentum (deviation from SMA)
    2. Volatility (high vol = fear)
    3. Volume trend (panic selling = high volume on red candles)
    4. Consecutive candle direction (streak of green = greed)
    5. RSI extremes

    Whale insight: Funding rate on Binance Futures is the #1 contrarian signal.
    When funding is >0.1%, it means longs are paying shorts heavily → too many
    longs → short squeeze incoming is LESS likely, dump is MORE likely.
    """

    name = "fear_greed_contrarian"
    description = "Synthetic Fear/Greed index contrarian - fade the crowd"
    category = "contrarian"

    def default_params(self) -> dict:
        return {
            "lookback": 30,
            "extreme_greed_threshold": 78,
            "extreme_fear_threshold": 22,
            "sma_period": 20,
            "rsi_period": 14,
            "min_candles": 40,
            "max_history": 500,
            "leverage": 5,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.05,
        }

    def _calculate_fear_greed_index(self, df: pd.DataFrame) -> pd.Series:
        """Build synthetic fear/greed index (0-100).

        Components (each 0-100, equally weighted):
        1. Momentum: price vs SMA deviation
        2. Volatility: inverse of realized vol (high vol = fear)
        3. Volume pattern: buying vs selling volume ratio
        4. Streak: consecutive candle direction
        5. RSI: direct mapping
        """
        closes = df["close"]
        volumes = df["volume"]
        lookback = self.params["lookback"]

        # 1. Momentum score: how far above/below SMA
        sma = closes.rolling(self.params["sma_period"]).mean()
        deviation = (closes - sma) / sma
        # Normalize to 0-100: +5% deviation = 100, -5% = 0
        momentum_score = ((deviation / 0.05) * 50 + 50).clip(0, 100)

        # 2. Volatility score: HIGH volatility = FEAR, LOW = complacency/greed
        returns = closes.pct_change()
        rolling_vol = returns.rolling(lookback).std()
        vol_percentile = rolling_vol.rank(pct=True) * 100
        volatility_score = 100 - vol_percentile  # Invert: high vol = low score (fear)

        # 3. Volume pattern: buying pressure vs selling pressure
        bar_range = (df["high"] - df["low"]).replace(0, np.nan)
        buy_ratio = (df["close"] - df["low"]) / bar_range
        buy_ratio = buy_ratio.fillna(0.5)
        vol_pattern_score = buy_ratio.rolling(lookback).mean() * 100

        # 4. Streak score: consecutive green/red candles
        is_green = (closes > df["open"]).astype(float)
        # Rolling sum of green candles in last N
        green_ratio = is_green.rolling(lookback).mean()
        streak_score = green_ratio * 100

        # 5. RSI
        delta = closes.diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.params["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.params["rsi_period"]).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        # Composite index (equal weight)
        index = (momentum_score + volatility_score + vol_pattern_score + streak_score + rsi) / 5.0

        return index

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        fg_index = self._calculate_fear_greed_index(df)

        current_fg = fg_index.iloc[-1]
        prev_fg = fg_index.iloc[-2]

        if pd.isna(current_fg):
            return None

        # EXTREME GREED → SHORT (crowd is too bullish, reversal incoming)
        if current_fg > self.params["extreme_greed_threshold"]:
            # Additional confirmation: greed is starting to decline
            turning_down = current_fg < prev_fg
            if turning_down:
                intensity = (current_fg - self.params["extreme_greed_threshold"]) / 22
                confidence = min(0.55 + intensity * 0.3, 0.9)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.SHORT,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"CONTRARIAN SHORT: Extreme Greed ({current_fg:.0f}/100) turning down",
                )

        # EXTREME FEAR → LONG (crowd is panicking, bottom is near)
        if current_fg < self.params["extreme_fear_threshold"]:
            turning_up = current_fg > prev_fg
            if turning_up:
                intensity = (self.params["extreme_fear_threshold"] - current_fg) / 22
                confidence = min(0.55 + intensity * 0.3, 0.9)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.LONG,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"CONTRARIAN LONG: Extreme Fear ({current_fg:.0f}/100) turning up",
                )

        return None


class RetailSentimentFaderStrategy(BaseStrategy):
    """Fade retail trader positioning strategy.

    Retail traders tend to:
    1. Buy AFTER prices have already risen significantly (FOMO)
    2. Sell AFTER prices have already dropped (panic)
    3. Hold losers too long, sell winners too early (disposition effect)
    4. Increase position size at the worst times (averaging down into trends)

    This strategy detects these retail patterns and trades AGAINST them.

    Paper: "The Disposition Effect in Securities Trading" - Shefrin & Statman (1985)
    Paper: "Noise Trader Risk" - De Long et al. (1990)

    Detection method:
    - Rapid price increase + volume surge = retail FOMO buying → SHORT
    - Rapid price decrease + volume surge = retail panic selling → LONG
    - Extended one-direction move + decreasing volume = exhaustion → fade
    """

    name = "retail_fader"
    description = "Detect and fade retail trader FOMO/panic patterns"
    category = "contrarian"

    def default_params(self) -> dict:
        return {
            "fomo_lookback": 5,
            "fomo_price_threshold": 0.05,  # 5% move in lookback period
            "panic_price_threshold": -0.05,
            "volume_surge_mult": 2.0,
            "exhaustion_lookback": 10,
            "exhaustion_vol_decline": 0.6,  # Volume must decline to 60% of peak
            "min_candles": 20,
            "max_history": 500,
            "leverage": 5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        closes = df["close"]
        volumes = df["volume"]

        lookback = self.params["fomo_lookback"]
        if len(df) < lookback + 10:
            return None

        # Recent price change
        recent_return = (closes.iloc[-1] - closes.iloc[-lookback]) / closes.iloc[-lookback]
        avg_volume = volumes.rolling(20).mean().iloc[-1]
        recent_avg_vol = volumes.iloc[-lookback:].mean()
        vol_ratio = recent_avg_vol / avg_volume if avg_volume > 0 else 1

        # Pattern 1: FOMO detection (rapid rise + volume surge)
        if (
            recent_return > self.params["fomo_price_threshold"]
            and vol_ratio > self.params["volume_surge_mult"]
        ):
            # Check if move is decelerating (exhaustion)
            last_return = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
            prev_return = (closes.iloc[-2] - closes.iloc[-3]) / closes.iloc[-3]
            decelerating = abs(last_return) < abs(prev_return)

            if decelerating:
                confidence = min(0.55 + vol_ratio * 0.05 + recent_return * 2, 0.85)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.SHORT,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=(
                        f"FADE FOMO: +{recent_return:.1%} in {lookback} bars, "
                        f"{vol_ratio:.1f}x vol, decelerating"
                    ),
                )

        # Pattern 2: Panic detection (rapid drop + volume surge)
        if (
            recent_return < self.params["panic_price_threshold"]
            and vol_ratio > self.params["volume_surge_mult"]
        ):
            last_return = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
            prev_return = (closes.iloc[-2] - closes.iloc[-3]) / closes.iloc[-3]
            decelerating = abs(last_return) < abs(prev_return)

            if decelerating:
                confidence = min(0.55 + vol_ratio * 0.05 + abs(recent_return) * 2, 0.85)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.LONG,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=(
                        f"FADE PANIC: {recent_return:.1%} in {lookback} bars, "
                        f"{vol_ratio:.1f}x vol, decelerating"
                    ),
                )

        # Pattern 3: Trend exhaustion (price trending but volume declining)
        exh_lookback = self.params["exhaustion_lookback"]
        if len(df) >= exh_lookback + 5:
            price_trend = (closes.iloc[-1] - closes.iloc[-exh_lookback]) / closes.iloc[-exh_lookback]
            vol_start = volumes.iloc[-exh_lookback : -exh_lookback + 3].mean()
            vol_end = volumes.iloc[-3:].mean()
            vol_decline = vol_end / vol_start if vol_start > 0 else 1

            if abs(price_trend) > 0.03 and vol_decline < self.params["exhaustion_vol_decline"]:
                if price_trend > 0:
                    return TradeSignal(
                        symbol=candles[-1].symbol,
                        side=Side.SHORT,
                        confidence=min(0.55 + abs(price_trend) * 3, 0.8),
                        strategy_name=self.name,
                        leverage=max(self.params["leverage"] - 1, 2),
                        stop_loss_pct=self.params["stop_loss_pct"] * 1.2,
                        take_profit_pct=self.params["take_profit_pct"],
                        reason=(
                            f"EXHAUSTION SHORT: +{price_trend:.1%} trend, "
                            f"vol declined to {vol_decline:.0%}"
                        ),
                    )
                else:
                    return TradeSignal(
                        symbol=candles[-1].symbol,
                        side=Side.LONG,
                        confidence=min(0.55 + abs(price_trend) * 3, 0.8),
                        strategy_name=self.name,
                        leverage=max(self.params["leverage"] - 1, 2),
                        stop_loss_pct=self.params["stop_loss_pct"] * 1.2,
                        take_profit_pct=self.params["take_profit_pct"],
                        reason=(
                            f"EXHAUSTION LONG: {price_trend:.1%} trend, "
                            f"vol declined to {vol_decline:.0%}"
                        ),
                    )

        return None


class FundingRateContrarianStrategy(BaseStrategy):
    """Simulated funding rate contrarian strategy.

    In real Binance Futures, funding rate reflects long/short imbalance:
    - High positive funding → too many longs → SHORT signal
    - High negative funding → too many shorts → LONG signal

    Since we don't have real funding data, we SIMULATE it using:
    - Price deviation from VWAP (proxy for directional bias)
    - Volume imbalance (buy vs sell pressure)
    - Recent price action direction

    Whale insight: When funding rate exceeds ±0.05%, the top traders
    consistently take the opposite side. This is the single most reliable
    contrarian indicator in crypto futures.

    Paper: "Perpetual Futures and Funding Rate Dynamics" - various (2021)
    """

    name = "funding_rate_contrarian"
    description = "Simulated funding rate contrarian - fade crowded trades"
    category = "contrarian"

    def default_params(self) -> dict:
        return {
            "lookback": 20,
            "extreme_threshold": 0.7,  # Synthetic funding extremity (0-1)
            "confirmation_candles": 3,
            "min_candles": 30,
            "max_history": 500,
            "leverage": 5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }

    def _synthetic_funding_rate(self, df: pd.DataFrame) -> pd.Series:
        """Estimate funding rate from price action.

        Positive = longs dominating (bullish crowd), Negative = shorts dominating.
        Scale: -1 to +1
        """
        closes = df["close"]
        lookback = self.params["lookback"]

        # Component 1: Price vs VWAP deviation
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vwap = (typical * df["volume"]).rolling(lookback).sum() / df["volume"].rolling(lookback).sum()
        price_deviation = (closes - vwap) / vwap

        # Component 2: Directional bias from candle bodies
        body = df["close"] - df["open"]
        body_ratio = body / (df["high"] - df["low"]).replace(0, np.nan)
        body_ratio = body_ratio.fillna(0)
        directional_bias = body_ratio.rolling(lookback).mean()

        # Component 3: Buy/sell volume imbalance
        bar_range = (df["high"] - df["low"]).replace(0, np.nan)
        buy_pct = ((df["close"] - df["low"]) / bar_range).fillna(0.5)
        imbalance = (buy_pct - 0.5).rolling(lookback).mean() * 2  # Scale to -1,1

        # Composite synthetic funding: -1 (extreme short bias) to +1 (extreme long bias)
        funding = (price_deviation * 10 + directional_bias + imbalance) / 3
        funding = funding.clip(-1, 1)

        return funding

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        funding = self._synthetic_funding_rate(df)

        current_funding = funding.iloc[-1]
        if pd.isna(current_funding):
            return None

        threshold = self.params["extreme_threshold"]

        # Check for confirmation: funding was extreme for N candles
        confirm_n = self.params["confirmation_candles"]
        recent_funding = funding.iloc[-confirm_n:]
        if recent_funding.isna().any():
            return None

        # Extreme long bias → SHORT (everyone is long, expect reversal)
        if all(f > threshold for f in recent_funding):
            # Check for early reversal sign
            declining = current_funding < funding.iloc[-2]
            if declining:
                confidence = min(0.6 + (current_funding - threshold) * 0.5, 0.9)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.SHORT,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=(
                        f"FUNDING CONTRARIAN SHORT: synthetic rate {current_funding:.2f} "
                        f"(extreme long bias, declining)"
                    ),
                )

        # Extreme short bias → LONG (everyone is short, expect squeeze)
        if all(f < -threshold for f in recent_funding):
            rising = current_funding > funding.iloc[-2]
            if rising:
                confidence = min(0.6 + (abs(current_funding) - threshold) * 0.5, 0.9)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.LONG,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=(
                        f"FUNDING CONTRARIAN LONG: synthetic rate {current_funding:.2f} "
                        f"(extreme short bias, rising → short squeeze)"
                    ),
                )

        return None


class WyckoffPsychologyStrategy(BaseStrategy):
    """Wyckoff method-inspired crowd psychology strategy.

    Richard Wyckoff's market cycle (1930s):
    1. Accumulation: Smart money buying while retail is fearful
    2. Markup: Trend up (follow)
    3. Distribution: Smart money selling while retail is euphoric
    4. Markdown: Trend down (follow)

    We detect phase transitions to trade AHEAD of the crowd:
    - Accumulation→Markup transition: LONG
    - Distribution→Markdown transition: SHORT

    Key principle: "Composite Man" (institutional traders) manipulates
    price to trigger retail emotions, then trades against them.

    Reference: "Studies in Tape Reading" - Richard Wyckoff (1910)
    """

    name = "wyckoff_psychology"
    description = "Wyckoff cycle phase detection - trade ahead of the crowd"
    category = "contrarian"

    def default_params(self) -> dict:
        return {
            "range_lookback": 30,
            "range_threshold": 0.04,  # Price range within 4% = ranging
            "volume_decline_pct": 0.5,
            "breakout_volume_mult": 1.8,
            "spring_threshold": 0.002,  # 0.2% below range low = spring
            "upthrust_threshold": 0.002,
            "min_candles": 40,
            "max_history": 500,
            "leverage": 5,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.05,
        }

    def _detect_phase(self, df: pd.DataFrame) -> tuple[str, float]:
        """Detect current Wyckoff phase.

        Returns: (phase_name, confidence)
        """
        lookback = self.params["range_lookback"]
        recent = df.tail(lookback)
        closes = recent["close"]
        volumes = recent["volume"]

        range_high = recent["high"].max()
        range_low = recent["low"].min()
        range_pct = (range_high - range_low) / range_low if range_low > 0 else 0

        current_close = df["close"].iloc[-1]
        current_low = df["low"].iloc[-1]
        current_high = df["high"].iloc[-1]

        is_ranging = range_pct < self.params["range_threshold"]

        # Volume trend (declining volume in range = accumulation/distribution)
        vol_first_half = volumes.iloc[: lookback // 2].mean()
        vol_second_half = volumes.iloc[lookback // 2 :].mean()
        vol_declining = vol_second_half < vol_first_half * self.params["volume_decline_pct"]

        if not is_ranging:
            return "trending", 0.0

        # Price near range low = potential accumulation
        near_low = (current_close - range_low) / (range_high - range_low) < 0.3 if range_high != range_low else False
        near_high = (current_close - range_low) / (range_high - range_low) > 0.7 if range_high != range_low else False

        # SPRING: Brief dip below range low then reclaim (accumulation sign)
        spring_level = range_low * (1 - self.params["spring_threshold"])
        spring_detected = current_low < spring_level and current_close > range_low

        # UPTHRUST: Brief push above range high then fail (distribution sign)
        upthrust_level = range_high * (1 + self.params["upthrust_threshold"])
        upthrust_detected = current_high > upthrust_level and current_close < range_high

        if spring_detected:
            return "accumulation_spring", 0.8

        if upthrust_detected:
            return "distribution_upthrust", 0.8

        if near_low and vol_declining:
            return "accumulation", 0.5

        if near_high and vol_declining:
            return "distribution", 0.5

        return "ranging", 0.0

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        phase, confidence = self._detect_phase(df)

        if confidence < 0.5:
            return None

        # Accumulation spring → LONG (smart money finished buying, markup coming)
        if phase == "accumulation_spring":
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason="WYCKOFF SPRING: Accumulation phase spring detected → markup expected",
            )

        # Distribution upthrust → SHORT (smart money finished selling, markdown coming)
        if phase == "distribution_upthrust":
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason="WYCKOFF UPTHRUST: Distribution phase upthrust → markdown expected",
            )

        # General accumulation zone
        if phase == "accumulation":
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=max(self.params["leverage"] - 2, 2),
                stop_loss_pct=self.params["stop_loss_pct"] * 1.5,
                take_profit_pct=self.params["take_profit_pct"],
                reason="WYCKOFF ACCUMULATION: Smart money accumulating near range low",
            )

        # General distribution zone
        if phase == "distribution":
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=max(self.params["leverage"] - 2, 2),
                stop_loss_pct=self.params["stop_loss_pct"] * 1.5,
                take_profit_pct=self.params["take_profit_pct"],
                reason="WYCKOFF DISTRIBUTION: Smart money distributing near range high",
            )

        return None
