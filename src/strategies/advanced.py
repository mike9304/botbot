"""Advanced strategies: order flow, sentiment, and market microstructure.

References:
- "Advances in Financial Machine Learning" - Marcos Lopez de Prado (2018)
- "Order Flow Analysis for Crypto Markets" - Various (adapted from CME research)
- "Sentiment Analysis for Cryptocurrency Trading" - Kraaijeveld & De Smedt (2020)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.core.models import Candle, Side, TradeSignal

from .base import BaseStrategy


class OrderFlowImbalanceStrategy(BaseStrategy):
    """Order flow imbalance detection strategy.

    Whale insight: Large traders leave footprints in the order flow. By analyzing
    buying vs selling volume (approximated via close position within candle),
    we can detect accumulation and distribution phases. When buy volume dominates
    for several candles, it signals whale accumulation.

    Reference: "Order Flow Trading for Fun and Profit" - daemon.xyz (2020)
    """

    name = "order_flow_imbalance"
    description = "Buy/sell volume imbalance detection"
    category = "orderflow"

    def default_params(self) -> dict:
        return {
            "imbalance_period": 10,
            "imbalance_threshold": 0.65,  # 65% buy or sell volume
            "trend_confirmation_period": 20,
            "min_candles": 25,
            "max_history": 500,
            "leverage": 4,
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.03,
        }

    def _estimate_buy_sell_volume(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Approximate buy/sell volume using close position within bar.

        Close near high = more buying pressure, close near low = more selling.
        """
        bar_range = df["high"] - df["low"]
        bar_range = bar_range.replace(0, np.nan)

        buy_ratio = (df["close"] - df["low"]) / bar_range
        buy_ratio = buy_ratio.fillna(0.5)

        buy_volume = df["volume"] * buy_ratio
        sell_volume = df["volume"] * (1 - buy_ratio)
        return buy_volume, sell_volume

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        buy_vol, sell_vol = self._estimate_buy_sell_volume(df)

        period = self.params["imbalance_period"]
        total_buy = buy_vol.rolling(period).sum()
        total_sell = sell_vol.rolling(period).sum()
        total_vol = total_buy + total_sell
        buy_ratio = total_buy / total_vol.replace(0, np.nan)

        current_ratio = buy_ratio.iloc[-1]
        if pd.isna(current_ratio):
            return None

        threshold = self.params["imbalance_threshold"]

        # Trend confirmation
        trend_period = self.params["trend_confirmation_period"]
        ema = df["close"].ewm(span=trend_period, adjust=False).mean()
        uptrend = df["close"].iloc[-1] > ema.iloc[-1]

        # Strong buy imbalance in uptrend
        if current_ratio > threshold and uptrend:
            confidence = min(0.5 + (current_ratio - threshold) * 2, 0.85)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.LONG,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"Buy flow imbalance ({current_ratio:.1%} buy vol, uptrend)",
            )

        # Strong sell imbalance in downtrend
        if current_ratio < (1 - threshold) and not uptrend:
            confidence = min(0.5 + ((1 - threshold) - current_ratio) * 2, 0.85)
            return TradeSignal(
                symbol=candles[-1].symbol,
                side=Side.SHORT,
                confidence=confidence,
                strategy_name=self.name,
                leverage=self.params["leverage"],
                stop_loss_pct=self.params["stop_loss_pct"],
                take_profit_pct=self.params["take_profit_pct"],
                reason=f"Sell flow imbalance ({1-current_ratio:.1%} sell vol, downtrend)",
            )

        return None


class SmartMoneyConceptStrategy(BaseStrategy):
    """Smart Money Concepts (SMC) / ICT strategy.

    Whale insight: ICT (Inner Circle Trader) methodology identifies:
    - Order blocks: Last bearish candle before a bullish move (demand zone)
    - Fair value gaps: Imbalanced price areas that get revisited
    - Liquidity sweeps: Stop hunts above/below recent highs/lows

    This strategy is extremely popular among top crypto futures traders.
    """

    name = "smart_money_concept"
    description = "ICT-based order blocks, FVG, and liquidity sweeps"
    category = "orderflow"

    def default_params(self) -> dict:
        return {
            "swing_lookback": 20,
            "fvg_min_gap_pct": 0.003,  # 0.3% minimum fair value gap
            "liquidity_sweep_threshold": 0.001,  # 0.1% beyond swing high/low
            "min_candles": 30,
            "max_history": 500,
            "leverage": 5,
            "stop_loss_pct": 0.015,
            "take_profit_pct": 0.03,
        }

    def _find_swing_highs_lows(self, df: pd.DataFrame, lookback: int = 5):
        """Find recent swing high and swing low points."""
        highs = df["high"]
        lows = df["low"]

        swing_highs = []
        swing_lows = []

        for i in range(lookback, len(df) - lookback):
            if highs.iloc[i] == highs.iloc[i - lookback : i + lookback + 1].max():
                swing_highs.append((i, highs.iloc[i]))
            if lows.iloc[i] == lows.iloc[i - lookback : i + lookback + 1].min():
                swing_lows.append((i, lows.iloc[i]))

        return swing_highs, swing_lows

    def _detect_fvg(self, df: pd.DataFrame) -> list[dict]:
        """Detect Fair Value Gaps (imbalanced candles)."""
        gaps = []
        for i in range(2, len(df)):
            # Bullish FVG: gap between candle[i-2] high and candle[i] low
            if df["low"].iloc[i] > df["high"].iloc[i - 2]:
                gap_size = (df["low"].iloc[i] - df["high"].iloc[i - 2]) / df["close"].iloc[i]
                if gap_size > self.params["fvg_min_gap_pct"]:
                    gaps.append({
                        "type": "bullish",
                        "top": df["low"].iloc[i],
                        "bottom": df["high"].iloc[i - 2],
                        "index": i,
                    })
            # Bearish FVG
            if df["high"].iloc[i] < df["low"].iloc[i - 2]:
                gap_size = (df["low"].iloc[i - 2] - df["high"].iloc[i]) / df["close"].iloc[i]
                if gap_size > self.params["fvg_min_gap_pct"]:
                    gaps.append({
                        "type": "bearish",
                        "top": df["low"].iloc[i - 2],
                        "bottom": df["high"].iloc[i],
                        "index": i,
                    })
        return gaps

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        current_close = df["close"].iloc[-1]
        current_low = df["low"].iloc[-1]
        current_high = df["high"].iloc[-1]

        swing_highs, swing_lows = self._find_swing_highs_lows(
            df.iloc[:-3], lookback=5  # Exclude last 3 candles for swing detection
        )

        if not swing_highs or not swing_lows:
            return None

        # Get most recent swing points
        last_swing_high = swing_highs[-1][1] if swing_highs else None
        last_swing_low = swing_lows[-1][1] if swing_lows else None

        threshold = self.params["liquidity_sweep_threshold"]

        # Liquidity sweep below swing low then reversal (bullish)
        if last_swing_low:
            swept_below = current_low < last_swing_low * (1 - threshold)
            closed_above = current_close > last_swing_low
            if swept_below and closed_above:
                confidence = 0.7
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.LONG,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"Liquidity sweep below {last_swing_low:.2f} + reversal",
                )

        # Liquidity sweep above swing high then reversal (bearish)
        if last_swing_high:
            swept_above = current_high > last_swing_high * (1 + threshold)
            closed_below = current_close < last_swing_high
            if swept_above and closed_below:
                confidence = 0.7
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.SHORT,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"Liquidity sweep above {last_swing_high:.2f} + reversal",
                )

        # Check for FVG fill entry
        fvgs = self._detect_fvg(df.iloc[:-1])
        if fvgs:
            last_fvg = fvgs[-1]
            if last_fvg["type"] == "bullish":
                # Price filling bullish FVG -> buy
                in_gap = last_fvg["bottom"] <= current_close <= last_fvg["top"]
                if in_gap:
                    return TradeSignal(
                        symbol=candles[-1].symbol,
                        side=Side.LONG,
                        confidence=0.65,
                        strategy_name=self.name,
                        leverage=self.params["leverage"],
                        stop_loss_pct=self.params["stop_loss_pct"],
                        take_profit_pct=self.params["take_profit_pct"],
                        reason=f"Bullish FVG fill @ {last_fvg['bottom']:.2f}-{last_fvg['top']:.2f}",
                    )

        return None


class MarketRegimeStrategy(BaseStrategy):
    """Market regime detection strategy.

    Whale insight: Successful traders first identify the market regime
    (trending, ranging, volatile) before applying strategies. This strategy
    classifies regimes and trades accordingly:
    - Trending: follow the trend
    - Ranging: mean revert at boundaries
    - Volatile: widen stops, reduce size

    Paper: "Hidden Markov Models for Regime Detection" - Bulla (2011)
    """

    name = "market_regime"
    description = "Market regime classification with adaptive strategy"
    category = "statistical"

    def default_params(self) -> dict:
        return {
            "regime_lookback": 30,
            "adx_period": 14,
            "adx_trend_threshold": 25,
            "volatility_lookback": 20,
            "vol_high_percentile": 80,
            "min_candles": 40,
            "max_history": 500,
            "leverage": 3,
            "stop_loss_pct": 0.02,
            "take_profit_pct": 0.04,
        }

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Average Directional Index for trend strength."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di

    def analyze(self, candles: list[Candle]) -> Optional[TradeSignal]:
        df = self.to_dataframe(candles)
        adx, plus_di, minus_di = self._calculate_adx(df, self.params["adx_period"])

        current_adx = adx.iloc[-1]
        if pd.isna(current_adx):
            return None

        # Volatility regime
        returns = df["close"].pct_change()
        current_vol = returns.rolling(self.params["volatility_lookback"]).std().iloc[-1]
        historical_vol = returns.rolling(100).std()
        vol_percentile = (historical_vol < current_vol).mean() * 100

        trending = current_adx > self.params["adx_trend_threshold"]
        high_volatility = vol_percentile > self.params["vol_high_percentile"]

        if trending and not high_volatility:
            # Trend following
            bullish = plus_di.iloc[-1] > minus_di.iloc[-1]
            prev_bullish = plus_di.iloc[-2] > minus_di.iloc[-2]

            # DI crossover
            if bullish and not prev_bullish:
                confidence = min(0.5 + (current_adx - 25) * 0.01, 0.8)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.LONG,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"Regime: TRENDING (ADX={current_adx:.0f}), +DI crossover",
                )
            if not bullish and prev_bullish:
                confidence = min(0.5 + (current_adx - 25) * 0.01, 0.8)
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.SHORT,
                    confidence=confidence,
                    strategy_name=self.name,
                    leverage=self.params["leverage"],
                    stop_loss_pct=self.params["stop_loss_pct"],
                    take_profit_pct=self.params["take_profit_pct"],
                    reason=f"Regime: TRENDING (ADX={current_adx:.0f}), -DI crossover",
                )

        elif not trending:
            # Mean reversion in ranging market
            period = self.params["regime_lookback"]
            upper = df["close"].rolling(period).mean() + 2 * df["close"].rolling(period).std()
            lower = df["close"].rolling(period).mean() - 2 * df["close"].rolling(period).std()

            if df["close"].iloc[-1] < lower.iloc[-1]:
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.LONG,
                    confidence=0.6,
                    strategy_name=self.name,
                    leverage=max(self.params["leverage"] - 1, 1),
                    stop_loss_pct=self.params["stop_loss_pct"] * 1.5,
                    take_profit_pct=self.params["take_profit_pct"] * 0.7,
                    reason=f"Regime: RANGING (ADX={current_adx:.0f}), mean reversion LONG",
                )
            if df["close"].iloc[-1] > upper.iloc[-1]:
                return TradeSignal(
                    symbol=candles[-1].symbol,
                    side=Side.SHORT,
                    confidence=0.6,
                    strategy_name=self.name,
                    leverage=max(self.params["leverage"] - 1, 1),
                    stop_loss_pct=self.params["stop_loss_pct"] * 1.5,
                    take_profit_pct=self.params["take_profit_pct"] * 0.7,
                    reason=f"Regime: RANGING (ADX={current_adx:.0f}), mean reversion SHORT",
                )

        return None
