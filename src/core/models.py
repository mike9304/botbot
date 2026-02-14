"""Core data models for the trading simulation system.

Uses a hybrid approach:
- Candle: stays as @dataclass for hot-path performance (~10k+ instances/sim)
- TradeSignal, AccountState: Pydantic v2 BaseModel for validation & serialization
- Position, Order: @dataclass (mutated frequently by exchange)
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    LIQUIDATED = "LIQUIDATED"


class TimeFrame(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"


# ── Hot-path models (dataclass for zero-overhead construction) ──────────────

@dataclass
class Candle:
    """OHLCV candle. Kept as dataclass for performance in hot loops."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str = "BTCUSDT"
    timeframe: TimeFrame = TimeFrame.M1

    def __post_init__(self) -> None:
        if self.high < self.low:
            self.high, self.low = self.low, self.high
        if self.volume < 0:
            self.volume = 0.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


@dataclass
class Order:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str = "BTCUSDT"
    side: Side = Side.LONG
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0
    price: float = 0.0
    leverage: int = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    timestamp: float = field(default_factory=time.time)
    filled_price: float = 0.0
    fee: float = 0.0
    pnl: float = 0.0


@dataclass
class Position:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    symbol: str = "BTCUSDT"
    side: Side = Side.LONG
    entry_price: float = 0.0
    quantity: float = 0.0
    leverage: int = 1
    unrealized_pnl: float = 0.0
    liquidation_price: float = 0.0
    margin: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.quantity < 0:
            self.quantity = 0.0
        if self.leverage < 1:
            self.leverage = 1

    def calculate_pnl(self, current_price: float) -> float:
        if self.side == Side.LONG:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
        return self.unrealized_pnl

    def calculate_liquidation_price(self) -> float:
        maintenance_margin_rate = 0.005  # 0.5% Binance default
        if self.side == Side.LONG:
            self.liquidation_price = self.entry_price * (1 - 1 / self.leverage + maintenance_margin_rate)
        else:
            self.liquidation_price = self.entry_price * (1 + 1 / self.leverage - maintenance_margin_rate)
        return self.liquidation_price


# ── Validated models (Pydantic v2 for validation & serialization) ───────────

class TradeSignal(BaseModel):
    """A trading signal with built-in validation.

    Pydantic v2 gives us:
    - Automatic confidence clamping to [0, 1]
    - Leverage range enforcement (1-125)
    - JSON serialization via .model_dump()
    """
    model_config = {"frozen": False}

    symbol: str
    side: Side
    confidence: float = Field(ge=0.0, le=1.0)
    strategy_name: str
    leverage: int = Field(default=1, ge=1, le=125)
    stop_loss_pct: Optional[float] = Field(default=None, ge=0.001, le=0.5)
    take_profit_pct: Optional[float] = Field(default=None, ge=0.001, le=1.0)
    reason: str = ""
    timestamp: float = Field(default_factory=time.time)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class AccountState(BaseModel):
    """Account state with computed properties.

    Pydantic v2 gives us:
    - Automatic serialization for API responses
    - Validation on construction
    - Computed properties via @property
    """
    model_config = {"frozen": False, "arbitrary_types_allowed": True}

    balance: float = 10000.0
    equity: float = 10000.0
    available_margin: float = 10000.0
    initial_balance: float = 10000.0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    total_trades: int = 0
    positions: dict[str, Position] = Field(default_factory=dict)
    order_history: list[Order] = Field(default_factory=list)
    peak_equity: float = 10000.0
    max_drawdown: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.win_count / self.total_trades

    @property
    def pnl_percent(self) -> float:
        if self.initial_balance <= 0:
            return 0.0
        return (self.total_pnl / self.initial_balance) * 100

    def update_drawdown(self) -> None:
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity
        current_dd = (self.peak_equity - self.equity) / self.peak_equity
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd
