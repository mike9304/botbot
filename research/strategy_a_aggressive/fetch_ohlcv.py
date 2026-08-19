"""Public OHLCV fetchers for paper research. No API keys, no trading.

Primary: Bitget USDT-FUTURES public candles (the user's venue).
Fallback: Binance Vision public spot klines (no auth; used only if Bitget fails).
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

BITGET_HISTORY = "https://api.bitget.com/api/v2/mix/market/history-candles"
BINANCE_VISION = "https://data-api.binance.vision/api/v3/klines"
UA = "botbot-strategy-a-research/1.0 (paper backtest; no trading)"

GRANULARITY_MS = {
    "1H": 3_600_000,
    "2H": 7_200_000,
    "4H": 14_400_000,
    "1D": 86_400_000,
}


def _http_get_json(url: str, timeout: int = 30, retries: int = 5) -> object:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            time.sleep(0.4 * (2**attempt))
    raise RuntimeError(f"GET failed after {retries} retries: {url} ({last_err})")


def fetch_bitget_futures(
    symbol: str,
    granularity: str,
    start_ms: int,
    end_ms: int,
    pause_s: float = 0.12,
) -> np.ndarray:
    """Return structured array ts, open, high, low, close, volume (oldest first)."""
    rows: list[list[float]] = []
    cursor = end_ms
    seen: set[int] = set()
    pages = 0
    while cursor > start_ms:
        url = (
            f"{BITGET_HISTORY}?symbol={symbol}&productType=USDT-FUTURES"
            f"&granularity={granularity}&limit=200&endTime={cursor}"
        )
        payload = _http_get_json(url)
        if not isinstance(payload, dict) or payload.get("code") != "00000":
            raise RuntimeError(f"Bitget error: {payload}")
        data = payload.get("data") or []
        if not data:
            break
        page_ts = []
        for item in data:
            ts = int(item[0])
            if ts in seen or ts < start_ms or ts > end_ms:
                continue
            seen.add(ts)
            page_ts.append(ts)
            rows.append(
                [
                    float(ts),
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[5]),
                ]
            )
        pages += 1
        oldest = min(int(item[0]) for item in data)
        if oldest >= cursor:
            break
        cursor = oldest  # API returns ts < endTime; no overlap in probes
        if pages > 800:
            break
        time.sleep(pause_s)

    if not rows:
        return _empty_ohlcv()
    arr = np.array(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    # de-dupe timestamps
    _, uniq = np.unique(arr[:, 0], return_index=True)
    return arr[np.sort(uniq)]


def fetch_binance_vision_spot(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    pause_s: float = 0.12,
) -> np.ndarray:
    """Binance Vision public spot klines (no API key). Fallback only."""
    mapping = {"1H": "1h", "2H": "2h", "4H": "4h", "1D": "1d"}
    binance_interval = mapping[interval]
    rows: list[list[float]] = []
    cursor = start_ms
    pages = 0
    while cursor < end_ms:
        url = (
            f"{BINANCE_VISION}?symbol={symbol}&interval={binance_interval}"
            f"&limit=1000&startTime={cursor}&endTime={end_ms}"
        )
        data = _http_get_json(url)
        if not isinstance(data, list) or not data:
            break
        for item in data:
            rows.append(
                [
                    float(item[0]),
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[5]),
                ]
            )
        last_ts = int(data[-1][0])
        nxt = last_ts + 1
        if nxt <= cursor:
            break
        cursor = nxt
        pages += 1
        if pages > 200 or len(data) < 1000:
            break
        time.sleep(pause_s)
    if not rows:
        return _empty_ohlcv()
    arr = np.array(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    _, uniq = np.unique(arr[:, 0], return_index=True)
    return arr[np.sort(uniq)]


def _empty_ohlcv() -> np.ndarray:
    return np.empty((0, 6), dtype=np.float64)


def to_columns(arr: np.ndarray) -> dict[str, np.ndarray]:
    if arr.size == 0:
        z = np.array([], dtype=np.float64)
        return {"ts": z, "open": z, "high": z, "low": z, "close": z, "volume": z}
    return {
        "ts": arr[:, 0],
        "open": arr[:, 1],
        "high": arr[:, 2],
        "low": arr[:, 3],
        "close": arr[:, 4],
        "volume": arr[:, 5],
    }


def save_csv(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "ts_ms,open,high,low,close,volume"
    np.savetxt(path, arr, delimiter=",", header=header, comments="", fmt="%.8f")


def load_csv(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
