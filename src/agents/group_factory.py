"""Factory for creating predefined agent groups.

Creates groups based on:
1. Single-strategy groups (pure technical, momentum, orderflow, etc.)
2. Hybrid groups (combinations of strategies)
3. Each group has multiple agents with varied parameters for diversity.
"""
from __future__ import annotations

from src.core.exchange import VirtualExchange
from src.strategies.advanced import (
    MarketRegimeStrategy,
    OrderFlowImbalanceStrategy,
    SmartMoneyConceptStrategy,
)
from src.strategies.base import HybridStrategy
from src.strategies.momentum import (
    MeanReversionStrategy,
    MomentumBreakoutStrategy,
    RSITrendMomentumStrategy,
)
from src.strategies.technical import (
    BollingerBreakoutStrategy,
    EMATripleCrossStrategy,
    FibonacciRetracementStrategy,
    RSIMACDStrategy,
    VolumeProfileStrategy,
)

from .base_agent import AgentConfig, AgentGroup, TradingAgent

# Supported trading symbols (by market cap / volume)
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
]


def create_all_groups(exchange: VirtualExchange) -> list[AgentGroup]:
    """Create all agent groups for the simulation."""
    groups = []

    # === PURE STRATEGY GROUPS ===

    # Group 1: Technical Analysis Team
    groups.append(_create_technical_group(exchange))

    # Group 2: Momentum Traders
    groups.append(_create_momentum_group(exchange))

    # Group 3: Mean Reversion / Statistical
    groups.append(_create_mean_reversion_group(exchange))

    # Group 4: Order Flow / Smart Money
    groups.append(_create_orderflow_group(exchange))

    # Group 5: Market Regime Adaptive
    groups.append(_create_regime_group(exchange))

    # === HYBRID STRATEGY GROUPS ===

    # Group 6: Technical + Momentum Hybrid
    groups.append(_create_hybrid_tech_momentum(exchange))

    # Group 7: OrderFlow + Technical Hybrid
    groups.append(_create_hybrid_orderflow_tech(exchange))

    # Group 8: Full Ensemble (all strategies)
    groups.append(_create_hybrid_ensemble(exchange))

    # Group 9: Conservative Hybrid (mean reversion + regime)
    groups.append(_create_hybrid_conservative(exchange))

    # Group 10: Aggressive Hybrid (momentum + SMC)
    groups.append(_create_hybrid_aggressive(exchange))

    return groups


def _create_technical_group(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="Technical Analysts",
        description="Pure technical analysis using RSI, MACD, Bollinger, EMA, Fibonacci, VWAP",
        category="technical",
    )
    strategies = [
        ("rsi_macd_default", RSIMACDStrategy()),
        ("rsi_macd_aggressive", RSIMACDStrategy({"rsi_overbought": 65, "rsi_oversold": 35, "leverage": 8})),
        ("bollinger_default", BollingerBreakoutStrategy()),
        ("bollinger_tight", BollingerBreakoutStrategy({"bb_std": 1.5, "squeeze_threshold": 0.02})),
        ("ema_cross_default", EMATripleCrossStrategy()),
        ("fibonacci_default", FibonacciRetracementStrategy()),
        ("vwap_default", VolumeProfileStrategy()),
    ]
    for name, strategy in strategies:
        agent = TradingAgent(
            AgentConfig(name=name, group="technical", strategy=strategy, symbols=DEFAULT_SYMBOLS[:5]),
            exchange,
        )
        group.add_agent(agent)
    return group


def _create_momentum_group(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="Momentum Riders",
        description="Momentum breakout and trend-following strategies",
        category="momentum",
    )
    strategies = [
        ("momentum_default", MomentumBreakoutStrategy()),
        ("momentum_sensitive", MomentumBreakoutStrategy({"atr_multiplier": 1.5, "volume_surge_mult": 1.5})),
        ("rsi_trend_default", RSITrendMomentumStrategy()),
        ("rsi_trend_tight", RSITrendMomentumStrategy({"uptrend_rsi_floor": 45, "downtrend_rsi_ceiling": 55})),
    ]
    for name, strategy in strategies:
        agent = TradingAgent(
            AgentConfig(name=name, group="momentum", strategy=strategy, symbols=DEFAULT_SYMBOLS[:5]),
            exchange,
        )
        group.add_agent(agent)
    return group


def _create_mean_reversion_group(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="Mean Reverters",
        description="Statistical mean reversion strategies",
        category="statistical",
    )
    strategies = [
        ("mr_default", MeanReversionStrategy()),
        ("mr_tight", MeanReversionStrategy({"z_score_entry": 1.5, "lookback_period": 20})),
        ("mr_wide", MeanReversionStrategy({"z_score_entry": 2.5, "lookback_period": 50})),
    ]
    for name, strategy in strategies:
        agent = TradingAgent(
            AgentConfig(name=name, group="mean_reversion", strategy=strategy, symbols=DEFAULT_SYMBOLS[:5]),
            exchange,
        )
        group.add_agent(agent)
    return group


def _create_orderflow_group(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="Smart Money Trackers",
        description="Order flow imbalance and Smart Money Concepts (ICT)",
        category="orderflow",
    )
    strategies = [
        ("orderflow_default", OrderFlowImbalanceStrategy()),
        ("orderflow_sensitive", OrderFlowImbalanceStrategy({"imbalance_threshold": 0.6})),
        ("smc_default", SmartMoneyConceptStrategy()),
        ("smc_tight", SmartMoneyConceptStrategy({"fvg_min_gap_pct": 0.002})),
    ]
    for name, strategy in strategies:
        agent = TradingAgent(
            AgentConfig(name=name, group="orderflow", strategy=strategy, symbols=DEFAULT_SYMBOLS[:5]),
            exchange,
        )
        group.add_agent(agent)
    return group


def _create_regime_group(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="Regime Adapters",
        description="Market regime detection with adaptive strategy switching",
        category="regime",
    )
    strategies = [
        ("regime_default", MarketRegimeStrategy()),
        ("regime_sensitive", MarketRegimeStrategy({"adx_trend_threshold": 20})),
        ("regime_strict", MarketRegimeStrategy({"adx_trend_threshold": 30})),
    ]
    for name, strategy in strategies:
        agent = TradingAgent(
            AgentConfig(name=name, group="regime", strategy=strategy, symbols=DEFAULT_SYMBOLS[:5]),
            exchange,
        )
        group.add_agent(agent)
    return group


def _create_hybrid_tech_momentum(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="Tech-Momentum Hybrids",
        description="Combination of technical analysis + momentum strategies",
        category="hybrid",
    )
    combos = [
        ("rsi_macd+momentum", [(RSIMACDStrategy(), 0.5), (MomentumBreakoutStrategy(), 0.5)]),
        ("bollinger+rsi_trend", [(BollingerBreakoutStrategy(), 0.6), (RSITrendMomentumStrategy(), 0.4)]),
        ("ema+momentum+vwap", [
            (EMATripleCrossStrategy(), 0.4),
            (MomentumBreakoutStrategy(), 0.3),
            (VolumeProfileStrategy(), 0.3),
        ]),
    ]
    for name, strats in combos:
        hybrid = HybridStrategy(strats, {"confidence_threshold": 0.55})
        agent = TradingAgent(
            AgentConfig(name=name, group="hybrid_tech_mom", strategy=hybrid, symbols=DEFAULT_SYMBOLS[:5]),
            exchange,
        )
        group.add_agent(agent)
    return group


def _create_hybrid_orderflow_tech(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="OrderFlow-Tech Hybrids",
        description="Smart Money + Technical analysis combination",
        category="hybrid",
    )
    combos = [
        ("smc+rsi_macd", [(SmartMoneyConceptStrategy(), 0.6), (RSIMACDStrategy(), 0.4)]),
        ("orderflow+bollinger", [(OrderFlowImbalanceStrategy(), 0.5), (BollingerBreakoutStrategy(), 0.5)]),
        ("smc+fib+vwap", [
            (SmartMoneyConceptStrategy(), 0.4),
            (FibonacciRetracementStrategy(), 0.3),
            (VolumeProfileStrategy(), 0.3),
        ]),
    ]
    for name, strats in combos:
        hybrid = HybridStrategy(strats, {"confidence_threshold": 0.55})
        agent = TradingAgent(
            AgentConfig(name=name, group="hybrid_of_tech", strategy=hybrid, symbols=DEFAULT_SYMBOLS[:5]),
            exchange,
        )
        group.add_agent(agent)
    return group


def _create_hybrid_ensemble(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="Full Ensemble",
        description="All strategies combined with equal/weighted voting",
        category="hybrid",
    )
    all_strats_equal = [
        (RSIMACDStrategy(), 1.0),
        (BollingerBreakoutStrategy(), 1.0),
        (MomentumBreakoutStrategy(), 1.0),
        (MeanReversionStrategy(), 1.0),
        (OrderFlowImbalanceStrategy(), 1.0),
        (SmartMoneyConceptStrategy(), 1.0),
        (MarketRegimeStrategy(), 1.0),
    ]
    all_strats_weighted = [
        (SmartMoneyConceptStrategy(), 2.0),
        (MomentumBreakoutStrategy(), 1.5),
        (RSIMACDStrategy(), 1.0),
        (BollingerBreakoutStrategy(), 1.0),
        (OrderFlowImbalanceStrategy(), 1.5),
        (MarketRegimeStrategy(), 1.5),
        (VolumeProfileStrategy(), 1.0),
    ]
    agent1 = TradingAgent(
        AgentConfig(
            name="equal_ensemble",
            group="ensemble",
            strategy=HybridStrategy(all_strats_equal, {"confidence_threshold": 0.5}),
            symbols=DEFAULT_SYMBOLS[:5],
        ),
        exchange,
    )
    agent2 = TradingAgent(
        AgentConfig(
            name="weighted_ensemble",
            group="ensemble",
            strategy=HybridStrategy(all_strats_weighted, {"confidence_threshold": 0.5}),
            symbols=DEFAULT_SYMBOLS[:5],
        ),
        exchange,
    )
    group.add_agent(agent1)
    group.add_agent(agent2)
    return group


def _create_hybrid_conservative(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="Conservative Hybrids",
        description="Low-risk combination: mean reversion + regime detection",
        category="hybrid",
    )
    combos = [
        ("mr+regime", [(MeanReversionStrategy(), 0.5), (MarketRegimeStrategy(), 0.5)]),
        ("mr+vwap+fib", [
            (MeanReversionStrategy(), 0.4),
            (VolumeProfileStrategy(), 0.3),
            (FibonacciRetracementStrategy(), 0.3),
        ]),
    ]
    for name, strats in combos:
        hybrid = HybridStrategy(strats, {"confidence_threshold": 0.65})
        agent = TradingAgent(
            AgentConfig(
                name=name, group="hybrid_conservative", strategy=hybrid,
                symbols=DEFAULT_SYMBOLS[:5], max_daily_trades=5,
            ),
            exchange,
        )
        group.add_agent(agent)
    return group


def _create_hybrid_aggressive(exchange: VirtualExchange) -> AgentGroup:
    group = AgentGroup(
        name="Aggressive Hybrids",
        description="High-risk combination: momentum + SMC with higher leverage",
        category="hybrid",
    )
    combos = [
        ("momentum+smc", [(MomentumBreakoutStrategy(), 0.5), (SmartMoneyConceptStrategy(), 0.5)]),
        ("momentum+rsi_trend+orderflow", [
            (MomentumBreakoutStrategy(), 0.4),
            (RSITrendMomentumStrategy(), 0.3),
            (OrderFlowImbalanceStrategy(), 0.3),
        ]),
    ]
    for name, strats in combos:
        hybrid = HybridStrategy(strats, {"confidence_threshold": 0.5})
        agent = TradingAgent(
            AgentConfig(
                name=name, group="hybrid_aggressive", strategy=hybrid,
                symbols=DEFAULT_SYMBOLS[:5], max_daily_trades=15, risk_per_trade=0.03,
            ),
            exchange,
        )
        group.add_agent(agent)
    return group
