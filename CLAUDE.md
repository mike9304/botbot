# BotBot - Crypto Trading Agent Arena

## Project Overview
Multi-agent cryptocurrency futures trading simulation system with genetic evolution and reinforcement learning. Agents compete in a virtual Binance Futures environment, evolving through a "hire and fire" system.

## Architecture

```
botbot/
├── main.py                  # Entry point (web dashboard or headless mode)
├── src/
│   ├── core/
│   │   ├── models.py        # Data models (Candle, Order, Position, etc.)
│   │   ├── exchange.py      # Virtual Binance Futures exchange with fee model
│   │   ├── simulation.py    # Main simulation engine
│   │   └── ranking.py       # Ranking & prize distribution system
│   ├── agents/
│   │   ├── base_agent.py    # TradingAgent & AgentGroup classes
│   │   ├── group_factory.py # Creates 14 predefined agent groups
│   │   └── counter_agent.py # Counter-indicator agents (invert losers)
│   ├── strategies/
│   │   ├── base.py              # BaseStrategy & HybridStrategy
│   │   ├── technical.py         # RSI/MACD, Bollinger, EMA, Fibonacci, VWAP
│   │   ├── momentum.py          # Momentum breakout, mean reversion, RSI trend
│   │   ├── advanced.py          # Order flow, Smart Money (ICT), market regime
│   │   ├── contrarian.py        # Fear/Greed, Retail Fader, Wyckoff, Funding Rate
│   │   └── counter_indicator.py # Invert losing agents/strategy types
│   ├── evolution/
│   │   ├── genetic.py       # Genetic algorithm (crossover, mutation, selection)
│   │   └── reinforcement.py # Q-Learning RL component
│   ├── data/
│   │   └── market_data.py   # Binance API fetcher + synthetic data generator
│   └── visualization/
│       └── api.py           # FastAPI server with WebSocket real-time updates
├── frontend/
│   └── public/
│       └── index.html       # Single-page dashboard (vanilla JS)
├── tests/                   # Pytest test suite
├── docs/research/           # Trading strategy research documents
└── configs/                 # Configuration files
```

## Key Commands

```bash
# Run web dashboard (default)
python main.py

# Run headless simulation
python main.py --headless --candles 2000 --speed 50

# Run tests
python -m pytest tests/ -v

# Custom symbols and speed
python main.py --candles 5000 --speed 20 --symbols "BTCUSDT,ETHUSDT,SOLUSDT"
```

## Development Conventions

- **Python 3.11+** required
- Line length: 100 chars (ruff)
- Type hints on all public functions
- Dataclasses for models, ABC for strategy interface
- Async for I/O operations (aiohttp, WebSocket)
- Tests in `tests/` with pytest

## Agent Groups (14 total)

### Pure Strategy Groups (5):
1. **Technical Analysts** - RSI/MACD, Bollinger, EMA, Fibonacci, VWAP
2. **Momentum Riders** - ATR breakout, RSI trend momentum
3. **Mean Reverters** - Z-score mean reversion (tight/default/wide)
4. **Smart Money Trackers** - Order flow imbalance, ICT/SMC
5. **Regime Adapters** - ADX-based market regime detection

### Hybrid Groups (5):
6. **Tech-Momentum Hybrids** - Technical + Momentum combos
7. **OrderFlow-Tech Hybrids** - SMC + Technical combos
8. **Full Ensemble** - All strategies combined (equal & weighted)
9. **Conservative Hybrids** - Mean reversion + Regime (low risk)
10. **Aggressive Hybrids** - Momentum + SMC (high risk)

### Contrarian / Psychology Groups (2):
11. **Crowd Psychology Contrarians** - Fear/Greed index, retail FOMO fading, Wyckoff, funding rate contrarian
12. **Contrarian-Tech Hybrids** - Contrarian signals confirmed by technical analysis

### Counter-Indicator Groups (2):
13. **Counter-Indicators** - Monitor losing agents and invert their signals
14. **Strategy-Type Counters** - Track losing strategy CATEGORIES and systematically invert them

## Binance Futures Fee Model
- Maker: 0.02%, Taker: 0.05%
- Leverage: up to 125x BTC, 100x ETH, 75x others
- Liquidation with maintenance margin rate 0.5%
- Funding rate: 0.01% per 8 hours

## Evolution System
- **Genetic Algorithm**: Tournament selection, crossover, mutation
- **Hire/Fire**: Bottom 20% replaced by offspring of top 20%
- **RL Component**: Q-Learning for position sizing and exit timing
- **Prizes**: Distributed to top performers each generation

## Contrarian & Counter-Indicator System
- **Fear/Greed Index**: Synthetic index from momentum, volatility, volume, RSI, streak
- **Retail Fader**: Detects FOMO buying and panic selling, then fades them
- **Funding Rate**: Simulates Binance funding rate, shorts extreme longs and vice versa
- **Wyckoff Psychology**: Detects accumulation/distribution phases with spring/upthrust patterns
- **Counter-Indicator**: Tracks individual agents' accuracy; inverts signals from agents with <40% win rate
- **Type Counter**: Tracks entire strategy categories; when "momentum" type is losing, inverts ALL momentum signals
