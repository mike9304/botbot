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
from src.strategies.contrarian import (
    FearGreedContrarianStrategy,
    FundingRateContrarianStrategy,
    RetailSentimentFaderStrategy,
    WyckoffPsychologyStrategy,
)
from src.strategies.counter_indicator import (
    CounterIndicatorStrategy,
    StrategyTypeCounterIndicatorStrategy,
)
from src.strategies.proven_bots import (
    CombinedBinHClucStrategy,
    MultiTimeframeTrendStrategy,
    NostalgiaForInfinityStrategy,
    ScalpingMomentumStrategy,
)
from src.strategies.sentiment import (
    SentimentContrarianStrategy,
    SentimentFollowerStrategy,
    SentimentMomentumStrategy,
)
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
from .counter_agent import CounterIndicatorAgent, StrategyTypeCounterAgent
from src.evolution.loser_evolution import InverseLoserAgent

# Supported trading symbols (by market cap / volume)
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT",
]


def _build_agents_from_strategies(
    group: AgentGroup,
    group_key: str,
    strategies: list[tuple[str, object]],
    exchange: VirtualExchange,
    *,
    agent_cls: type = TradingAgent,
    symbols: list[str] | None = None,
    **config_overrides,
) -> list:
    """Helper to create agents from a list of (name, strategy) pairs and add to group.

    Returns the created agents (useful for counter-indicator groups).
    """
    symbols = symbols or DEFAULT_SYMBOLS[:5]
    agents = []
    for name, strategy in strategies:
        config = AgentConfig(
            name=name, group=group_key, strategy=strategy,
            symbols=symbols, **config_overrides,
        )
        agent = agent_cls(config, exchange)
        group.add_agent(agent)
        agents.append(agent)
    return agents


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

    # === CONTRARIAN / PSYCHOLOGY GROUPS ===

    # Group 11: Crowd Psychology Contrarians
    groups.append(_create_contrarian_group(exchange))

    # Group 12: Contrarian + Technical Hybrids
    groups.append(_create_hybrid_contrarian_tech(exchange))

    # === COUNTER-INDICATOR GROUPS ===

    # Group 13: Counter-Indicator Agents (invert losing agents)
    counter_group, counter_agents = _create_counter_indicator_group(exchange)
    groups.append(counter_group)

    # Group 14: Strategy-Type Counter Agents (invert losing strategy types)
    type_counter_group, type_counter_agents = _create_type_counter_group(exchange)
    groups.append(type_counter_group)

    # === SENTIMENT ANALYSIS GROUPS (FOLLOWER vs CONTRARIAN) ===

    # Group 15: Sentiment Followers (trade WITH the crowd)
    groups.append(_create_sentiment_follower_group(exchange))

    # Group 16: Sentiment Faders (trade AGAINST the crowd)
    groups.append(_create_sentiment_contrarian_group(exchange))

    # === PROVEN BOT STRATEGIES ===

    # Group 17: Proven Bots (adapted from top real-world trading bots)
    groups.append(_create_proven_bots_group(exchange))

    # === LOSER LEAGUE ===

    # Group 18: Loser League (evolved worst agents as reverse indicators)
    groups.append(_create_loser_league_group(exchange))

    # Group 19: Inverse Losers (profit from loser signals)
    groups.append(_create_inverse_loser_group(exchange))

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
    _build_agents_from_strategies(group, "technical", strategies, exchange)
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
    _build_agents_from_strategies(group, "momentum", strategies, exchange)
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
    _build_agents_from_strategies(group, "mean_reversion", strategies, exchange)
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
    _build_agents_from_strategies(group, "orderflow", strategies, exchange)
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
    _build_agents_from_strategies(group, "regime", strategies, exchange)
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


# === CONTRARIAN / PSYCHOLOGY GROUPS ===


def _create_contrarian_group(exchange: VirtualExchange) -> AgentGroup:
    """Group 11: Crowd Psychology Contrarians.

    Agents that analyze crowd/retail psychology and trade AGAINST the herd.
    """
    group = AgentGroup(
        name="Crowd Psychology Contrarians",
        description="Fear/Greed index, retail FOMO fading, funding rate contrarian, Wyckoff psychology",
        category="contrarian",
    )
    strategies = [
        ("fear_greed_default", FearGreedContrarianStrategy()),
        ("fear_greed_sensitive", FearGreedContrarianStrategy({
            "extreme_greed_threshold": 72, "extreme_fear_threshold": 28,
        })),
        ("retail_fader_default", RetailSentimentFaderStrategy()),
        ("retail_fader_sensitive", RetailSentimentFaderStrategy({
            "fomo_price_threshold": 0.03, "panic_price_threshold": -0.03,
            "volume_surge_mult": 1.5,
        })),
        ("funding_contrarian_default", FundingRateContrarianStrategy()),
        ("funding_contrarian_sensitive", FundingRateContrarianStrategy({
            "extreme_threshold": 0.6, "confirmation_candles": 2,
        })),
        ("wyckoff_default", WyckoffPsychologyStrategy()),
        ("wyckoff_sensitive", WyckoffPsychologyStrategy({
            "range_threshold": 0.05, "spring_threshold": 0.003,
        })),
    ]
    _build_agents_from_strategies(group, "contrarian", strategies, exchange)
    return group


def _create_hybrid_contrarian_tech(exchange: VirtualExchange) -> AgentGroup:
    """Group 12: Contrarian + Technical Hybrids.

    Combine contrarian signals with technical confirmation for higher confidence.
    """
    group = AgentGroup(
        name="Contrarian-Tech Hybrids",
        description="Contrarian psychology combined with technical confirmation",
        category="hybrid",
    )
    combos = [
        ("fear_greed+rsi_macd", [
            (FearGreedContrarianStrategy(), 0.6),
            (RSIMACDStrategy(), 0.4),
        ]),
        ("retail_fader+bollinger", [
            (RetailSentimentFaderStrategy(), 0.5),
            (BollingerBreakoutStrategy(), 0.5),
        ]),
        ("wyckoff+smc", [
            (WyckoffPsychologyStrategy(), 0.5),
            (SmartMoneyConceptStrategy(), 0.5),
        ]),
        ("funding+regime+vwap", [
            (FundingRateContrarianStrategy(), 0.4),
            (MarketRegimeStrategy(), 0.3),
            (VolumeProfileStrategy(), 0.3),
        ]),
    ]
    for name, strats in combos:
        hybrid = HybridStrategy(strats, {"confidence_threshold": 0.55})
        agent = TradingAgent(
            AgentConfig(
                name=name, group="hybrid_contrarian", strategy=hybrid,
                symbols=DEFAULT_SYMBOLS[:5],
            ),
            exchange,
        )
        group.add_agent(agent)
    return group


def _create_counter_indicator_group(
    exchange: VirtualExchange,
) -> tuple[AgentGroup, list[CounterIndicatorAgent]]:
    """Group 13: Counter-Indicator Agents.

    These agents DON'T analyze price. Instead, they monitor OTHER agents'
    signals and invert signals from agents who are consistently wrong.
    """
    group = AgentGroup(
        name="Counter-Indicators",
        description="Monitor losing agents and trade OPPOSITE to their signals",
        category="counter_indicator",
    )
    agents = []
    configs = [
        ("counter_default", CounterIndicatorStrategy()),
        ("counter_aggressive", CounterIndicatorStrategy({
            "loser_accuracy_threshold": 0.45,
            "min_inversion_confidence": 0.50,
            "leverage": 7,
        })),
        ("counter_conservative", CounterIndicatorStrategy({
            "loser_accuracy_threshold": 0.35,
            "min_inversion_confidence": 0.60,
            "min_trades_for_tracking": 15,
            "leverage": 3,
        })),
    ]
    for name, strategy in configs:
        agent = CounterIndicatorAgent(
            AgentConfig(
                name=name, group="counter_indicator", strategy=strategy,
                symbols=DEFAULT_SYMBOLS[:5], max_daily_trades=20,
            ),
            exchange,
        )
        group.add_agent(agent)
        agents.append(agent)
    return group, agents


def _create_type_counter_group(
    exchange: VirtualExchange,
) -> tuple[AgentGroup, list[StrategyTypeCounterAgent]]:
    """Group 14: Strategy-Type Counter Agents.

    Track entire strategy CATEGORIES and invert the losing ones.
    More aggressive approach that captures regime changes faster.
    """
    group = AgentGroup(
        name="Strategy-Type Counters",
        description="Track losing STRATEGY TYPES and systematically invert their signals",
        category="counter_indicator",
    )
    agents = []
    configs = [
        ("type_counter_default", StrategyTypeCounterIndicatorStrategy()),
        ("type_counter_aggressive", StrategyTypeCounterIndicatorStrategy({
            "loser_accuracy_threshold": 0.45,
            "min_inversion_confidence": 0.50,
            "leverage": 6,
        })),
    ]
    for name, strategy in configs:
        agent = StrategyTypeCounterAgent(
            AgentConfig(
                name=name, group="type_counter", strategy=strategy,
                symbols=DEFAULT_SYMBOLS[:5], max_daily_trades=20,
            ),
            exchange,
        )
        group.add_agent(agent)
        agents.append(agent)
    return group, agents


# === SENTIMENT ANALYSIS GROUPS ===


def _create_sentiment_follower_group(exchange: VirtualExchange) -> AgentGroup:
    """Group 15: Sentiment Followers.

    Trade WITH the crowd — when online sentiment is bullish, go LONG.
    Tests the hypothesis that crowds are sometimes right (in trending markets).
    """
    group = AgentGroup(
        name="Sentiment Followers",
        description="Trade WITH crowd sentiment — bullish crowd = LONG",
        category="sentiment_follow",
    )
    strategies = [
        ("sent_follow_default", SentimentFollowerStrategy()),
        ("sent_follow_sensitive", SentimentFollowerStrategy({
            "bullish_threshold": 20, "bearish_threshold": -20,
        })),
        ("sent_momentum", SentimentMomentumStrategy()),
        ("sent_momentum_tight", SentimentMomentumStrategy({
            "acceleration_threshold": 10, "min_sentiment_level": 15,
        })),
    ]
    _build_agents_from_strategies(group, "sentiment_follow", strategies, exchange)
    return group


def _create_sentiment_contrarian_group(exchange: VirtualExchange) -> AgentGroup:
    """Group 16: Sentiment Faders.

    Trade AGAINST the crowd — when euphoria peaks, SHORT.
    Tests the contrarian hypothesis that crowds are wrong at extremes.
    """
    group = AgentGroup(
        name="Sentiment Faders",
        description="Trade AGAINST crowd sentiment — euphoria = SHORT",
        category="sentiment_fade",
    )
    strategies = [
        ("sent_contra_default", SentimentContrarianStrategy()),
        ("sent_contra_aggressive", SentimentContrarianStrategy({
            "extreme_bullish_threshold": 55, "extreme_bearish_threshold": -55,
            "require_divergence": False,
        })),
        ("sent_contra_careful", SentimentContrarianStrategy({
            "extreme_bullish_threshold": 75, "extreme_bearish_threshold": -75,
            "require_divergence": True,
        })),
    ]
    _build_agents_from_strategies(group, "sentiment_fade", strategies, exchange)
    return group


# === PROVEN BOT STRATEGIES ===


def _create_proven_bots_group(exchange: VirtualExchange) -> AgentGroup:
    """Group 17: Proven Bots.

    Strategies adapted from top real-world trading bots
    (NostalgiaForInfinity, CombinedBinHCluc, scalpers, multi-TF).
    """
    group = AgentGroup(
        name="Proven Bots",
        description="Adapted from top real-world trading bots (Freqtrade, 3Commas)",
        category="proven_bot",
    )
    strategies = [
        ("nfi_default", NostalgiaForInfinityStrategy()),
        ("nfi_aggressive", NostalgiaForInfinityStrategy({
            "leverage": 6, "stop_loss_pct": 0.025, "take_profit_pct": 0.06,
        })),
        ("binhcluc_default", CombinedBinHClucStrategy()),
        ("binhcluc_tight", CombinedBinHClucStrategy({
            "close_to_bb_ratio": 0.995, "rsi_buy": 35,
        })),
        ("scalper_default", ScalpingMomentumStrategy()),
        ("scalper_fast", ScalpingMomentumStrategy({
            "ema_ultra_fast": 2, "ema_fast": 5, "leverage": 10,
            "stop_loss_pct": 0.005, "take_profit_pct": 0.01,
        })),
        ("mtf_trend_default", MultiTimeframeTrendStrategy()),
    ]
    _build_agents_from_strategies(group, "proven_bot", strategies, exchange)
    return group


# === LOSER LEAGUE ===


def _create_loser_league_group(exchange: VirtualExchange) -> AgentGroup:
    """Group 18: Loser League.

    Placeholder group — agents are added dynamically during evolution
    when the worst performers from other groups are collected here.
    Initially empty; populated by LoserLeague.evolve_losers().
    """
    group = AgentGroup(
        name="Loser League",
        description="Evolved worst agents — used as reverse indicators",
        category="loser_league",
    )
    # Start with seed losers (intentionally bad params)
    bad_strategies = [
        ("loser_seed_rsi", RSIMACDStrategy({"rsi_overbought": 50, "rsi_oversold": 50})),
        ("loser_seed_mr", MeanReversionStrategy({"z_score_entry": 0.5, "lookback_period": 5})),
    ]
    _build_agents_from_strategies(group, "loser_league", bad_strategies, exchange)
    return group


def _create_inverse_loser_group(exchange: VirtualExchange) -> AgentGroup:
    """Group 19: Inverse Losers.

    Agents that watch the Loser League and invert all their signals.
    This is how we profit from consistently bad agents.
    """
    group = AgentGroup(
        name="Inverse Losers",
        description="Invert Loser League signals for profit",
        category="inverse_loser",
    )
    configs = [
        "inverse_loser_default",
        "inverse_loser_conservative",
        "inverse_loser_aggressive",
    ]
    for name in configs:
        agent = InverseLoserAgent(
            AgentConfig(
                name=name, group="inverse_loser",
                strategy=RSIMACDStrategy(),  # Placeholder; signals come from losers
                symbols=DEFAULT_SYMBOLS[:5],
            ),
            exchange,
        )
        group.add_agent(agent)
    return group
