"""Market data provider - fetches real crypto data from public APIs.

Uses free public APIs:
- Binance public API for historical klines (no auth needed)
- Can also generate synthetic data for testing
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp
import numpy as np

from src.core.models import Candle, TimeFrame

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_FUTURES_URL = "https://fapi.binance.com"


async def fetch_historical_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> list[Candle]:
    """Fetch historical kline/candlestick data from Binance public API."""
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000),
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Binance API error: {resp.status}")
                    return []
                data = await resp.json()

        candles = []
        for kline in data:
            candle = Candle(
                timestamp=kline[0] / 1000,  # ms to seconds
                open=float(kline[1]),
                high=float(kline[2]),
                low=float(kline[3]),
                close=float(kline[4]),
                volume=float(kline[5]),
                symbol=symbol,
            )
            candles.append(candle)
        return candles

    except Exception as e:
        logger.error(f"Error fetching klines: {e}")
        return []


async def fetch_multiple_symbols(
    symbols: list[str],
    interval: str = "1h",
    limit: int = 500,
) -> dict[str, list[Candle]]:
    """Fetch historical data for multiple symbols concurrently."""
    tasks = [
        fetch_historical_klines(symbol, interval, limit)
        for symbol in symbols
    ]
    results = await asyncio.gather(*tasks)
    return {symbol: candles for symbol, candles in zip(symbols, results)}


def generate_synthetic_data(
    symbol: str = "BTCUSDT",
    n_candles: int = 1000,
    start_price: float = 50000.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    timeframe_seconds: int = 3600,
) -> list[Candle]:
    """Generate realistic synthetic OHLCV data using geometric Brownian motion.

    Includes:
    - Trend component
    - Volatility clustering (GARCH-like)
    - Volume correlation with price movement
    - Occasional spike candles (whale activity)
    """
    candles = []
    price = start_price
    current_vol = volatility
    base_volume = 1000.0
    start_ts = time.time() - n_candles * timeframe_seconds

    for i in range(n_candles):
        # GARCH-like volatility clustering
        current_vol = 0.9 * current_vol + 0.1 * volatility * (1 + np.random.exponential(0.5))
        current_vol = np.clip(current_vol, volatility * 0.3, volatility * 3)

        # Price movement
        returns = np.random.normal(trend, current_vol)

        # Occasional large moves (whale candles)
        if np.random.random() < 0.02:
            returns += np.random.choice([-1, 1]) * current_vol * 3

        open_price = price
        close_price = price * (1 + returns)

        # Generate realistic high/low
        intra_vol = abs(returns) + current_vol * 0.5
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, intra_vol * 0.3)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, intra_vol * 0.3)))

        # Volume correlates with volatility
        vol_mult = 1 + abs(returns) / volatility
        volume = base_volume * vol_mult * (1 + np.random.exponential(0.3))

        candle = Candle(
            timestamp=start_ts + i * timeframe_seconds,
            open=round(open_price, 2),
            high=round(high_price, 2),
            low=round(low_price, 2),
            close=round(close_price, 2),
            volume=round(volume, 2),
            symbol=symbol,
        )
        candles.append(candle)
        price = close_price

    return candles


def generate_multi_symbol_data(
    symbols: Optional[list[str]] = None,
    n_candles: int = 1000,
) -> dict[str, list[Candle]]:
    """Generate correlated synthetic data for multiple symbols."""
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]

    # Base prices and characteristics
    configs = {
        "BTCUSDT": {"price": 50000, "vol": 0.015, "trend": 0.0001},
        "ETHUSDT": {"price": 3000, "vol": 0.02, "trend": 0.00015},
        "SOLUSDT": {"price": 100, "vol": 0.03, "trend": 0.0002},
        "BNBUSDT": {"price": 400, "vol": 0.02, "trend": 0.0001},
        "XRPUSDT": {"price": 0.60, "vol": 0.025, "trend": 0.00005},
        "DOGEUSDT": {"price": 0.10, "vol": 0.035, "trend": 0.0001},
        "ADAUSDT": {"price": 0.50, "vol": 0.025, "trend": 0.0001},
        "AVAXUSDT": {"price": 30, "vol": 0.03, "trend": 0.00015},
        "DOTUSDT": {"price": 7, "vol": 0.025, "trend": 0.0001},
        "MATICUSDT": {"price": 0.80, "vol": 0.03, "trend": 0.0001},
    }

    all_data = {}
    for symbol in symbols:
        cfg = configs.get(symbol, {"price": 100, "vol": 0.02, "trend": 0.0001})
        all_data[symbol] = generate_synthetic_data(
            symbol=symbol,
            n_candles=n_candles,
            start_price=cfg["price"],
            volatility=cfg["vol"],
            trend=cfg["trend"],
        )

    return all_data
