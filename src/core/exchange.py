"""Virtual exchange engine simulating Binance Futures."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .models import (
    AccountState,
    Candle,
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TradeSignal,
)

logger = logging.getLogger(__name__)


@dataclass
class BinanceFeeModel:
    """Binance Futures fee structure.

    Default VIP0 tier:
    - Maker: 0.0200% (with BNB discount: 0.0180%)
    - Taker: 0.0500% (with BNB discount: 0.0450%)

    Using standard taker fees since most simulated trades are market orders.
    """

    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0005  # 0.05%
    # Funding rate applied every 8 hours
    funding_rate: float = 0.0001  # 0.01% default

    def calculate_fee(self, notional_value: float, is_maker: bool = False) -> float:
        rate = self.maker_fee if is_maker else self.taker_fee
        return notional_value * rate


class VirtualExchange:
    """Simulates a Binance-like futures exchange with realistic fee model."""

    # Maximum leverage per symbol (simplified)
    MAX_LEVERAGE = {
        "BTCUSDT": 125,
        "ETHUSDT": 100,
        "SOLUSDT": 75,
        "BNBUSDT": 75,
        "XRPUSDT": 75,
        "DOGEUSDT": 75,
        "ADAUSDT": 75,
        "AVAXUSDT": 50,
        "DOTUSDT": 50,
        "MATICUSDT": 50,
    }

    TICK_SIZE = {
        "BTCUSDT": 0.10,
        "ETHUSDT": 0.01,
        "SOLUSDT": 0.001,
        "BNBUSDT": 0.01,
        "XRPUSDT": 0.0001,
        "DOGEUSDT": 0.00001,
        "ADAUSDT": 0.0001,
        "AVAXUSDT": 0.01,
        "DOTUSDT": 0.001,
        "MATICUSDT": 0.0001,
    }

    def __init__(self, fee_model: Optional[BinanceFeeModel] = None):
        self.fee_model = fee_model or BinanceFeeModel()
        self.accounts: dict[str, AccountState] = {}
        self.current_prices: dict[str, float] = {}
        self.pending_orders: dict[str, list[Order]] = {}  # agent_id -> orders

    def register_agent(self, agent_id: str, initial_balance: float = 10000.0):
        self.accounts[agent_id] = AccountState(
            balance=initial_balance,
            equity=initial_balance,
            available_margin=initial_balance,
            peak_equity=initial_balance,
        )
        self.pending_orders[agent_id] = []

    def update_price(self, symbol: str, candle: Candle):
        self.current_prices[symbol] = candle.close
        self._check_liquidations(symbol, candle)
        self._process_pending_orders(symbol, candle)
        self._update_all_equity()

    def execute_signal(self, agent_id: str, signal: TradeSignal) -> Optional[Order]:
        account = self.accounts.get(agent_id)
        if not account:
            logger.warning(f"Agent {agent_id} not registered")
            return None

        current_price = self.current_prices.get(signal.symbol)
        if not current_price:
            logger.warning(f"No price data for {signal.symbol}")
            return None

        # Check if agent already has a position in this symbol
        existing = account.positions.get(signal.symbol)
        if existing:
            # Close existing position first if opposite direction
            if existing.side != signal.side:
                self._close_position(agent_id, signal.symbol, current_price)
            else:
                logger.info(f"Agent {agent_id} already has {signal.side} position on {signal.symbol}")
                return None

        # Calculate position size based on available margin and leverage
        max_leverage = self.MAX_LEVERAGE.get(signal.symbol, 20)
        leverage = min(signal.leverage, max_leverage)

        # Risk 2% of equity per trade (position sizing)
        risk_amount = account.equity * 0.02
        notional_value = risk_amount * leverage
        quantity = notional_value / current_price
        margin_required = notional_value / leverage

        if margin_required > account.available_margin:
            logger.info(f"Agent {agent_id}: insufficient margin")
            return None

        # Create and fill market order
        fee = self.fee_model.calculate_fee(notional_value)
        order = Order(
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price,
            leverage=leverage,
            status=OrderStatus.FILLED,
            filled_price=current_price,
            fee=fee,
        )

        # Calculate stop loss and take profit prices
        if signal.stop_loss_pct:
            if signal.side == Side.LONG:
                order.stop_loss = current_price * (1 - signal.stop_loss_pct)
            else:
                order.stop_loss = current_price * (1 + signal.stop_loss_pct)

        if signal.take_profit_pct:
            if signal.side == Side.LONG:
                order.take_profit = current_price * (1 + signal.take_profit_pct)
            else:
                order.take_profit = current_price * (1 - signal.take_profit_pct)

        # Open position
        position = Position(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=current_price,
            quantity=quantity,
            leverage=leverage,
            margin=margin_required,
        )
        position.calculate_liquidation_price()

        account.positions[signal.symbol] = position
        account.balance -= fee
        account.total_fees += fee
        account.available_margin -= margin_required
        account.order_history.append(order)

        # Set up SL/TP as pending orders
        if order.stop_loss:
            sl_order = Order(
                symbol=signal.symbol,
                side=Side.SHORT if signal.side == Side.LONG else Side.LONG,
                order_type=OrderType.STOP_LOSS,
                quantity=quantity,
                price=order.stop_loss,
                leverage=leverage,
            )
            self.pending_orders[agent_id].append(sl_order)

        if order.take_profit:
            tp_order = Order(
                symbol=signal.symbol,
                side=Side.SHORT if signal.side == Side.LONG else Side.LONG,
                order_type=OrderType.TAKE_PROFIT,
                quantity=quantity,
                price=order.take_profit,
                leverage=leverage,
            )
            self.pending_orders[agent_id].append(tp_order)

        logger.info(
            f"Agent {agent_id}: {signal.side.value} {signal.symbol} "
            f"qty={quantity:.6f} @ {current_price} lev={leverage}x fee={fee:.4f}"
        )
        return order

    def close_position(self, agent_id: str, symbol: str) -> Optional[Order]:
        current_price = self.current_prices.get(symbol)
        if not current_price:
            return None
        return self._close_position(agent_id, symbol, current_price)

    def _close_position(
        self, agent_id: str, symbol: str, close_price: float
    ) -> Optional[Order]:
        account = self.accounts.get(agent_id)
        if not account:
            return None

        position = account.positions.get(symbol)
        if not position:
            return None

        pnl = position.calculate_pnl(close_price)
        notional_value = close_price * position.quantity
        fee = self.fee_model.calculate_fee(notional_value)

        close_side = Side.SHORT if position.side == Side.LONG else Side.LONG
        order = Order(
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=position.quantity,
            price=close_price,
            leverage=position.leverage,
            status=OrderStatus.FILLED,
            filled_price=close_price,
            fee=fee,
            pnl=pnl,
        )

        # Update account
        account.balance += pnl - fee
        account.total_pnl += pnl
        account.total_fees += fee
        account.total_trades += 1
        if pnl > 0:
            account.win_count += 1
        else:
            account.loss_count += 1

        account.available_margin += position.margin
        account.order_history.append(order)
        del account.positions[symbol]

        # Cancel related pending orders
        self.pending_orders[agent_id] = [
            o for o in self.pending_orders[agent_id] if o.symbol != symbol
        ]

        logger.info(
            f"Agent {agent_id}: CLOSE {symbol} @ {close_price} PnL={pnl:.2f} fee={fee:.4f}"
        )
        return order

    def _check_liquidations(self, symbol: str, candle: Candle):
        for agent_id, account in self.accounts.items():
            position = account.positions.get(symbol)
            if not position:
                continue

            liquidated = False
            if position.side == Side.LONG:
                liquidated = candle.low <= position.liquidation_price
            else:
                liquidated = candle.high >= position.liquidation_price

            if liquidated:
                logger.warning(f"Agent {agent_id}: LIQUIDATED {symbol} @ {position.liquidation_price}")
                # Liquidation: lose entire margin
                account.balance -= position.margin
                account.total_pnl -= position.margin
                account.total_trades += 1
                account.loss_count += 1
                account.available_margin += position.margin  # margin was already deducted
                del account.positions[symbol]
                self.pending_orders[agent_id] = [
                    o for o in self.pending_orders[agent_id] if o.symbol != symbol
                ]

    def _process_pending_orders(self, symbol: str, candle: Candle):
        for agent_id in list(self.pending_orders.keys()):
            orders = self.pending_orders[agent_id]
            remaining = []
            for order in orders:
                if order.symbol != symbol:
                    remaining.append(order)
                    continue

                triggered = False
                if order.order_type == OrderType.STOP_LOSS:
                    if order.side == Side.SHORT:  # SL for long position
                        triggered = candle.low <= order.price
                    else:  # SL for short position
                        triggered = candle.high >= order.price
                elif order.order_type == OrderType.TAKE_PROFIT:
                    if order.side == Side.SHORT:  # TP for long position
                        triggered = candle.high >= order.price
                    else:  # TP for short position
                        triggered = candle.low <= order.price

                if triggered:
                    self._close_position(agent_id, symbol, order.price)
                else:
                    remaining.append(order)

            self.pending_orders[agent_id] = remaining

    def _update_all_equity(self):
        for agent_id, account in self.accounts.items():
            total_unrealized = 0.0
            for symbol, position in account.positions.items():
                price = self.current_prices.get(symbol, position.entry_price)
                total_unrealized += position.calculate_pnl(price)

            account.equity = account.balance + total_unrealized
            account.update_drawdown()

    def get_account_summary(self, agent_id: str) -> dict:
        account = self.accounts.get(agent_id)
        if not account:
            return {}
        return {
            "agent_id": agent_id,
            "balance": round(account.balance, 2),
            "equity": round(account.equity, 2),
            "total_pnl": round(account.total_pnl, 2),
            "pnl_percent": round(account.pnl_percent, 2),
            "total_fees": round(account.total_fees, 2),
            "win_rate": round(account.win_rate * 100, 1),
            "total_trades": account.total_trades,
            "max_drawdown": round(account.max_drawdown * 100, 2),
            "open_positions": len(account.positions),
        }
