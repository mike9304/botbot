"""Online sentiment analysis strategies.

Simulates reading online community sentiment (Reddit, Twitter/X, Telegram, etc.)
and translates crowd emotion into trading signals.

Two competing approaches:
1. Sentiment FOLLOWER: Trade WITH the crowd sentiment
   (crowd is bullish → LONG, crowd is bearish → SHORT)
2. Sentiment CONTRARIAN: Trade AGAINST the crowd sentiment
   (crowd is euphoric → SHORT, crowd is panicking → LONG)

The question: Which approach makes more money? The simulation will tell us.

Sentiment is SIMULATED from price action because:
- Real sentiment APIs require paid access
- Price action IS a reflection of crowd sentiment
- We can model known behavioral patterns accurately

Models of crowd psychology used:
- Herding behavior: Crowds follow trends with delay
- Overreaction: Crowds overshoot on news
- Anchoring: Crowds fixate on round numbers
- Recency bias: Recent moves weigh more heavily
- FOMO/FUD cycles: Greed peaks at tops, fear peaks at bottoms

References:
- "Irrational Exuberance" - Robert Shiller (2000)
- "Sentiment Analysis for Cryptocurrency" - Kraaijeveld & De Smedt (2020)
- "Twitter mood predicts the stock market" - Bollen, Mao, Zeng (2011)
- "Crypto Fear & Greed Index" - alternative.me methodology
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.models import Candle, Side, TradeSignal

from .base import BaseStrategy


class SentimentSimulator:
    """Simulates online community sentiment from price action.

    Generates a sentiment score from -100 (extreme bearish) to +100 (extreme bullish).

    The model captures:
    1. Herding: Sentiment follows price with a LAG (crowds react late)
    2. Amplification: Crowds exaggerate moves (2x actual price change)
    3. Stickiness: Sentiment doesn't flip instantly (exponential smoothing)
    4. Volume confirms: High volume amplifies sentiment
    5. Novelty decay: Sentiment decays without new stimuli
    """

    def __init__(
        self,
        lag_periods: int = 3,
        amplification: float = 2.0,
        smoothing: float = 0.3,
        volume_weight: float = 0.3,
        decay_rate: float = 0.95,
    ):
        self.lag_periods = lag_periods
        self.amplification = amplification
        self.smoothing = smoothing
        self.volume_weight = volume_weight
        self.decay_rate = decay_rate

    def calculate_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """Calculate simulated community sentiment score (-100 to +100)."""
        closes = df["close"]
        volumes = df["volume"]

        # 1. Lagged price momentum (crowd reacts LATE)
        lagged_return = closes.pct_change(self.lag_periods).shift(1)  # Extra shift for lag
        price_sentiment = lagged_return * self.amplification * 100

        # 2. Volume amplification
        avg_vol = volumes.rolling(20).mean()
        vol_ratio = volumes / avg_vol.replace(0, 1)
        vol_factor = 1 + (vol_ratio - 1) * self.volume_weight

        # 3. Trend persistence (crowd anchors to recent trend)
        sma_short = closes.rolling(5).mean()
        sma_long = closes.rolling(20).mean()
        trend = ((sma_short - sma_long) / sma_long * 100).fillna(0)

        # 4. Consecutive candles (winning streak = euphoria)
        is_green = (closes > df["open"]).astype(float)
        streak = is_green.rolling(5).mean() * 2 - 1  # -1 to +1
        streak_sentiment = streak * 20

        # 5. Composite raw sentiment
        raw = (price_sentiment * vol_factor + trend * 0.5 + streak_sentiment) / 2.5

        # 6. Exponential smoothing (sentiment is sticky)
        smoothed = raw.ewm(span=int(1 / self.smoothing), adjust=False).mean()

        # 7. Clip to -100, +100
        return smoothed.clip(-100, 100)

    def get_sentiment_zone(self, score: float) -> str:
        """Classify sentiment into zones."""
        if score > 70:
            return "extreme_bullish"
        elif score > 30:
            return "bullish"
        elif score > -30:
            return "neutral"
        elif score > -70:
            return "bearish"
        else:
            return "extreme_bearish"


class SentimentFollowerStrategy(BaseStrategy):
    """Trade WITH the crowd sentiment.

    Philosophy: "The trend is your friend" / "Don't fight the tape"
    When the crowd is bullish → go LONG
    When the crowd is bearish → go SHORT

    This works in trending markets where sentiment reflects genuine momentum.
    Fails when sentiment reaches extremes and reverses.
    """

    name = "sentiment_follower"
    description = "Follow crowd sentiment - bullish crowd = LONG"
    category = "sentiment_follow"

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.simulator = SentimentSimulator(
            lag_periods=self.params.get("lag_periods", 3),
            amplification=self.params.get("amplification", 2.0),
        )

    def default_params(self) -> dict:
        return {
            "bullish_threshold": 30,
            "bearish_threshold": -30,
            "momentum_confirmation": True,
            "lag_periods": 3,
            "amplification": 2.0,
            "min_candles": 30,
            "max_history": 500,
            "leverage": 4,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        sentiment = self.simulator.calculate_sentiment(df)

        current_sent = sentiment.iloc[-1]
        prev_sent = sentiment.iloc[-2]

        if pd.isna(current_sent):
            return None

        # Optional: require momentum confirmation
        if self.params["momentum_confirmation"]:
            ema_fast = df["close"].ewm(span=9, adjust=False).mean()
            ema_slow = df["close"].ewm(span=21, adjust=False).mean()
            trend_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        else:
            trend_bullish = None

        bull_threshold = self.params["bullish_threshold"]
        bear_threshold = self.params["bearish_threshold"]

        # FOLLOW: Bullish sentiment crossing up → LONG
        if (
            current_sent > bull_threshold
            and prev_sent <= bull_threshold
            and (trend_bullish is None or trend_bullish)
        ):
            intensity = min(current_sent / 100, 1.0)
            confidence = 0.5 + intensity * 0.3
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=min(confidence, 0.85),
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"FOLLOW SENTIMENT: Bullish ({current_sent:.0f}/100) → LONG",
            )

        # FOLLOW: Bearish sentiment crossing down → SHORT
        if (
            current_sent < bear_threshold
            and prev_sent >= bear_threshold
            and (trend_bullish is None or not trend_bullish)
        ):
            intensity = min(abs(current_sent) / 100, 1.0)
            confidence = 0.5 + intensity * 0.3
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=min(confidence, 0.85),
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"FOLLOW SENTIMENT: Bearish ({current_sent:.0f}/100) → SHORT",
            )

        return None


class SentimentContrarianStrategy(BaseStrategy):
    """Trade AGAINST the crowd sentiment at extremes.

    Philosophy: "Be fearful when others are greedy, greedy when fearful"
    When crowd is EXTREMELY bullish → SHORT (top is near)
    When crowd is EXTREMELY bearish → LONG (bottom is near)

    This works at market extremes where sentiment has overshot.
    Fails in strong trends where sentiment stays extreme for extended periods.
    """

    name = "sentiment_contrarian"
    description = "Fade extreme crowd sentiment - euphoria = SHORT"
    category = "sentiment_fade"

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.simulator = SentimentSimulator(
            lag_periods=self.params.get("lag_periods", 3),
            amplification=self.params.get("amplification", 2.0),
        )

    def default_params(self) -> dict:
        return {
            "extreme_bullish_threshold": 65,
            "extreme_bearish_threshold": -65,
            "require_divergence": True,  # Price making new high but sentiment declining
            "use_statistical_extreme": True,  # Kristoufek (2015): use 2σ threshold
            "sigma_threshold": 2.0,  # Only trade at >2 standard deviations
            "sigma_lookback": 30,  # Lookback for calculating σ
            "lag_periods": 3,
            "amplification": 2.0,
            "min_candles": 30,
            "max_history": 500,
            "leverage": 5,
            "stop_loss_pct": 0.025,
            "take_profit_pct": 0.05,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        sentiment = self.simulator.calculate_sentiment(df)
        closes = df["close"]

        current_sent = sentiment.iloc[-1]
        prev_sent = sentiment.iloc[-2]

        if pd.isna(current_sent):
            return None

        # Statistical extreme detection (Kristoufek, 2015)
        # Only signal when sentiment is >2σ from its moving average
        if self.params.get("use_statistical_extreme", False):
            lookback = self.params["sigma_lookback"]
            sigma_thresh = self.params["sigma_threshold"]
            if len(sentiment.dropna()) >= lookback:
                sent_ma = sentiment.rolling(lookback).mean().iloc[-1]
                sent_std = sentiment.rolling(lookback).std().iloc[-1]
                if pd.notna(sent_std) and sent_std > 0:
                    z_score = (current_sent - sent_ma) / sent_std
                    # Must exceed σ threshold to trade
                    if abs(z_score) < sigma_thresh:
                        return None  # Not extreme enough — stay flat

        bull_extreme = self.params["extreme_bullish_threshold"]
        bear_extreme = self.params["extreme_bearish_threshold"]

        # CONTRARIAN: Extreme bullish AND turning down → SHORT
        if current_sent > bull_extreme:
            turning_down = current_sent < prev_sent

            # Optional divergence: price still rising but sentiment declining
            if self.params["require_divergence"]:
                price_up = closes.iloc[-1] > closes.iloc[-3]
                sent_down = current_sent < sentiment.iloc[-3] if not pd.isna(sentiment.iloc[-3]) else False
                divergence = price_up and sent_down
                condition = turning_down or divergence
            else:
                condition = turning_down

            if condition:
                intensity = (current_sent - bull_extreme) / (100 - bull_extreme)
                confidence = min(0.55 + intensity * 0.3, 0.9)
                reason = "FADE EUPHORIA"
                if self.params["require_divergence"] and 'divergence' in dir() and divergence:
                    reason += " + DIVERGENCE"
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.SHORT,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"{reason}: sentiment={current_sent:.0f}/100 → SHORT",
                )

        # CONTRARIAN: Extreme bearish AND turning up → LONG
        if current_sent < bear_extreme:
            turning_up = current_sent > prev_sent

            if self.params["require_divergence"]:
                price_down = closes.iloc[-1] < closes.iloc[-3]
                sent_up = current_sent > sentiment.iloc[-3] if not pd.isna(sentiment.iloc[-3]) else False
                divergence = price_down and sent_up
                condition = turning_up or divergence
            else:
                condition = turning_up

            if condition:
                intensity = (bear_extreme - current_sent) / (100 + bear_extreme)
                confidence = min(0.55 + intensity * 0.3, 0.9)
                reason = "FADE PANIC"
                if self.params["require_divergence"] and 'divergence' in dir() and divergence:
                    reason += " + DIVERGENCE"
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.LONG,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"{reason}: sentiment={current_sent:.0f}/100 → LONG",
                )

        return None


class SentimentMomentumStrategy(BaseStrategy):
    """Sentiment momentum - trade when sentiment is accelerating.

    Not just the level of sentiment but the RATE OF CHANGE matters.
    Rapidly increasing bullish sentiment = market mania forming → can ride it
    but with tight exits.

    This strategy tries to catch the "middle" of sentiment swings
    where the trend is strongest.
    """

    name = "sentiment_momentum"
    description = "Trade sentiment acceleration (rate of change)"
    category = "sentiment_follow"

    def __init__(self, params: Optional[dict] = None):
        super().__init__(params)
        self.simulator = SentimentSimulator()

    def default_params(self) -> dict:
        return {
            "acceleration_threshold": 15,  # Sentiment change per period
            "min_sentiment_level": 20,  # Don't trade in neutral zone
            "momentum_periods": 3,
            "use_second_derivative": True,  # Trade sentiment acceleration (not just velocity)
            "jerk_confirmation": True,  # Confirm with 2nd derivative sign
            "min_candles": 30,
            "max_history": 500,
            "leverage": 4,
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.03,
        }

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        sentiment = self.simulator.calculate_sentiment(df)

        if sentiment.iloc[-1] is None or pd.isna(sentiment.iloc[-1]):
            return None

        periods = self.params["momentum_periods"]
        if len(sentiment) < periods + 2:
            return None

        # Sentiment acceleration (1st derivative)
        sent_change = sentiment.iloc[-1] - sentiment.iloc[-periods - 1]
        current_sent = sentiment.iloc[-1]

        # 2nd derivative: is acceleration itself increasing? (jerk)
        if self.params.get("use_second_derivative") and len(sentiment) > periods * 2 + 2:
            prev_change = sentiment.iloc[-periods - 1] - sentiment.iloc[-2 * periods - 1]
            jerk = sent_change - prev_change  # Acceleration of acceleration
            jerk_confirms_bull = jerk > 0
            jerk_confirms_bear = jerk < 0
        else:
            jerk_confirms_bull = True
            jerk_confirms_bear = True

        threshold = self.params["acceleration_threshold"]
        min_level = self.params["min_sentiment_level"]

        # Strong bullish acceleration + already bullish + jerk confirms
        jerk_ok_bull = not self.params.get("jerk_confirmation") or jerk_confirms_bull
        if sent_change > threshold and current_sent > min_level and jerk_ok_bull:
            confidence = min(0.5 + abs(sent_change) / 100, 0.8)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=(
                    f"SENTIMENT ACCELERATING UP: "
                    f"Δ={sent_change:+.0f} in {periods} bars, level={current_sent:.0f}"
                ),
            )

        # Strong bearish acceleration + already bearish + jerk confirms
        jerk_ok_bear = not self.params.get("jerk_confirmation") or jerk_confirms_bear
        if sent_change < -threshold and current_sent < -min_level and jerk_ok_bear:
            confidence = min(0.5 + abs(sent_change) / 100, 0.8)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=(
                    f"SENTIMENT ACCELERATING DOWN: "
                    f"Δ={sent_change:+.0f} in {periods} bars, level={current_sent:.0f}"
                ),
            )

        return None
