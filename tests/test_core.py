"""Tests for core trading engine components."""
import pytest

from src.core.exchange import BinanceFeeModel, VirtualExchange
from src.core.models import (
    AccountState,
    Candle,
    Order,
    OrderStatus,
    Position,
    Side,
    TradeSignal,
)


class TestModels:
    def test_candle_mid(self):
        candle = Candle(
            timestamp=1000, open=100, high=110, low=90, close=105, volume=1000
        )
        assert candle.mid == 100.0

    def test_position_long_pnl(self):
        pos = Position(
            symbol="BTCUSDT", side=Side.LONG, entry_price=50000, quantity=0.1, leverage=10
        )
        pnl = pos.calculate_pnl(51000)
        assert pnl == 100.0  # (51000-50000) * 0.1

    def test_position_short_pnl(self):
        pos = Position(
            symbol="BTCUSDT", side=Side.SHORT, entry_price=50000, quantity=0.1, leverage=10
        )
        pnl = pos.calculate_pnl(49000)
        assert pnl == 100.0  # (50000-49000) * 0.1

    def test_position_liquidation_long(self):
        pos = Position(
            symbol="BTCUSDT", side=Side.LONG, entry_price=50000, quantity=0.1, leverage=10
        )
        liq = pos.calculate_liquidation_price()
        # Should be ~45250 (1 - 1/10 + 0.005 = 0.905 -> 50000 * 0.905)
        assert 45000 < liq < 46000

    def test_position_liquidation_short(self):
        pos = Position(
            symbol="BTCUSDT", side=Side.SHORT, entry_price=50000, quantity=0.1, leverage=10
        )
        liq = pos.calculate_liquidation_price()
        # Should be ~54750 (1 + 1/10 - 0.005 = 1.095 -> 50000 * 1.095)
        assert 54000 < liq < 56000

    def test_account_win_rate(self):
        acc = AccountState(win_count=6, loss_count=4, total_trades=10)
        assert acc.win_rate == 0.6

    def test_account_pnl_percent(self):
        acc = AccountState(total_pnl=500)
        assert acc.pnl_percent == 5.0  # 500/10000 * 100


class TestBinanceFeeModel:
    def test_taker_fee(self):
        model = BinanceFeeModel()
        fee = model.calculate_fee(10000, is_maker=False)
        assert fee == 5.0  # 10000 * 0.0005

    def test_maker_fee(self):
        model = BinanceFeeModel()
        fee = model.calculate_fee(10000, is_maker=True)
        assert fee == 2.0  # 10000 * 0.0002


class TestVirtualExchange:
    def setup_method(self):
        self.exchange = VirtualExchange()
        self.exchange.register_agent("test_agent", 10000.0)

    def test_register_agent(self):
        assert "test_agent" in self.exchange.accounts
        assert self.exchange.accounts["test_agent"].balance == 10000.0

    def test_update_price(self):
        candle = Candle(
            timestamp=1000, open=50000, high=50100, low=49900,
            close=50050, volume=100, symbol="BTCUSDT",
        )
        self.exchange.update_price("BTCUSDT", candle)
        assert self.exchange.current_prices["BTCUSDT"] == 50050

    def test_execute_long_signal(self):
        # Set up price
        candle = Candle(
            timestamp=1000, open=50000, high=50100, low=49900,
            close=50050, volume=100, symbol="BTCUSDT",
        )
        self.exchange.update_price("BTCUSDT", candle)

        signal = TradeSignal(
            symbol="BTCUSDT",
            side=Side.LONG,
            confidence=0.8,
            strategy_name="test",
            leverage=5,
            stop_loss_pct=0.02,
            take_profit_pct=0.04,
        )

        order = self.exchange.execute_signal("test_agent", signal)
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.side == Side.LONG
        assert "BTCUSDT" in self.exchange.accounts["test_agent"].positions

    def test_close_position(self):
        # Open position
        candle1 = Candle(
            timestamp=1000, open=50000, high=50100, low=49900,
            close=50000, volume=100, symbol="BTCUSDT",
        )
        self.exchange.update_price("BTCUSDT", candle1)

        signal = TradeSignal(
            symbol="BTCUSDT", side=Side.LONG, confidence=0.8,
            strategy_name="test", leverage=5,
        )
        self.exchange.execute_signal("test_agent", signal)

        # Price goes up
        candle2 = Candle(
            timestamp=2000, open=50000, high=51100, low=50000,
            close=51000, volume=100, symbol="BTCUSDT",
        )
        self.exchange.update_price("BTCUSDT", candle2)

        # Close position
        order = self.exchange.close_position("test_agent", "BTCUSDT")
        assert order is not None
        assert order.pnl > 0  # Should be profitable
        assert "BTCUSDT" not in self.exchange.accounts["test_agent"].positions

    def test_account_summary(self):
        summary = self.exchange.get_account_summary("test_agent")
        assert summary["balance"] == 10000.0
        assert summary["agent_id"] == "test_agent"
