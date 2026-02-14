# BotBot - Crypto Trading Agent Arena

## Project Overview
Multi-agent cryptocurrency futures trading simulation system with genetic evolution, reinforcement learning, and AI Teacher oversight. 19 agent groups (~70+ agents) compete in a virtual Binance Futures environment, evolving through a "hire and fire" system with an AI Teacher that evaluates, grades, and guides them.

## Architecture

```
botbot/
├── main.py                  # Entry point (web dashboard or headless mode)
├── src/
│   ├── core/
│   │   ├── models.py        # Data models (Candle, Order, Position, etc.)
│   │   ├── exchange.py      # Virtual Binance Futures exchange with fee model
│   │   ├── simulation.py    # Main simulation engine (orchestrates everything)
│   │   └── ranking.py       # Ranking & prize distribution system
│   ├── agents/
│   │   ├── base_agent.py    # TradingAgent & AgentGroup classes
│   │   ├── group_factory.py # Creates 19 predefined agent groups
│   │   └── counter_agent.py # Counter-indicator agents (invert losers)
│   ├── strategies/
│   │   ├── base.py              # BaseStrategy & HybridStrategy
│   │   ├── technical.py         # RSI/MACD, Bollinger, EMA, Fibonacci, VWAP
│   │   ├── momentum.py          # Momentum breakout, mean reversion, RSI trend
│   │   ├── advanced.py          # Order flow, Smart Money (ICT), market regime
│   │   ├── contrarian.py        # Fear/Greed, Retail Fader, Wyckoff, Funding Rate
│   │   ├── counter_indicator.py # Invert losing agents/strategy types
│   │   ├── sentiment.py         # Online sentiment follower vs contrarian
│   │   └── proven_bots.py       # NostalgiaForInfinity, BinHCluc, scalpers, MTF
│   ├── evolution/
│   │   ├── genetic.py           # Genetic algorithm (crossover, mutation, selection)
│   │   ├── reinforcement.py     # Q-Learning RL component
│   │   └── loser_evolution.py   # Loser League (evolve worst → reverse indicators)
│   ├── ai_teacher/
│   │   ├── __init__.py          # AI Teacher package
│   │   ├── llm_client.py        # Multi-provider LLM client (Claude, GPT, DeepSeek, Gemini, Groq)
│   │   ├── risk_manager.py      # Risk monitoring, drawdown alerts, correlation checks
│   │   └── teacher.py           # AI Teacher core — evaluates, grades, guides evolution
│   ├── data/
│   │   └── market_data.py   # Binance API fetcher + synthetic data generator
│   └── visualization/
│       └── api.py           # FastAPI server with WebSocket real-time updates
├── frontend/
│   └── public/
│       └── index.html       # Single-page dashboard (vanilla JS)
├── tests/                   # Pytest test suite (96+ tests)
├── docs/research/           # Trading strategy research documents
└── configs/                 # Configuration files
```

## Key Commands

```bash
# Run web dashboard (default, AI Teacher in offline mode)
python main.py

# Run with AI Teacher using DeepSeek API (cheapest)
python main.py --llm-provider deepseek

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

## Agent Groups (19 total)

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

### Sentiment Analysis Groups (2):
15. **Sentiment Followers** - Trade WITH crowd sentiment (bullish crowd = LONG)
16. **Sentiment Faders** - Trade AGAINST extreme sentiment (euphoria = SHORT)

### Proven Bot Strategies (1):
17. **Proven Bots** - NostalgiaForInfinity, CombinedBinHCluc, Scalping Momentum, Multi-TF Trend

### Loser League (2):
18. **Loser League** - Evolve worst agents as reverse indicators (breed among themselves)
19. **Inverse Losers** - Invert all Loser League signals for profit

## AI Teacher System (NEW)

The AI Teacher acts as a "선생님" (teacher/mentor) overseeing all agent groups:

### Architecture
- **Risk Manager** (`risk_manager.py`): Monitors drawdowns, liquidation proximity, leverage, and portfolio correlation
- **LLM Client** (`llm_client.py`): Multi-provider client supporting 6 API providers
- **Teacher** (`teacher.py`): Core evaluation engine with report cards, grades (A+ to F-), and evolution guidance

### Supported LLM Providers
| Provider | Default Model | Input/MTok | Output/MTok | Monthly Cost* |
|---|---|---:|---:|---:|
| DeepSeek | V3.2 | $0.28 | $0.42 | ~$0.01 |
| Gemini | 2.5 Flash | $0.15 | $0.60 | ~$0.01 |
| GPT-4o-mini | gpt-4o-mini | $0.15 | $0.60 | ~$0.01 |
| Groq | Llama 3.3 70B | $0.59 | $0.99 | ~$0.02 |
| Claude | Haiku 4.5 | $1.00 | $5.00 | ~$0.06 |
| Claude | Sonnet 4.5 | $3.00 | $15.00 | ~$0.19 |

*Cost based on 1 simulation/day, 70 agents, 20 evaluations/sim

### Teacher Capabilities
- **Agent Grading**: A+ through F- based on PnL, win rate, drawdown, trade frequency
- **Risk Alerts**: CRITICAL (auto-disable), HIGH (warning), MEDIUM (advisory)
- **Market Regime Detection**: Trending up/down, ranging, volatile — adjusts evolution guidance
- **Evolution Guidance**: Which categories to preserve, which need aggressive mutation
- **Report Cards**: Detailed per-agent and per-group evaluation with strengths/weaknesses
- **Offline Mode**: Full functionality without API keys (rule-based evaluation)

### Configuration
```python
# In SimulationConfig:
ai_teacher_enabled = True       # Enable AI Teacher
llm_provider = "deepseek"       # "offline", "deepseek", "openai", "anthropic", "google", "groq"
llm_api_key = ""                # From env var or direct
max_daily_llm_cost = 5.0        # Safety cap in USD
```

## Binance Futures Fee Model
- Maker: 0.02%, Taker: 0.05%
- Leverage: up to 125x BTC, 100x ETH, 75x others
- Liquidation with maintenance margin rate 0.5%
- Funding rate: 0.01% per 8 hours

## Evolution System
- **Genetic Algorithm**: Tournament selection, crossover, mutation
- **Hire/Fire**: Bottom 20% replaced by offspring of top 20%
- **Loser League**: Bottom agents breed separately → inverted for profit
- **RL Component**: Q-Learning for position sizing and exit timing
- **Prizes**: Distributed to top performers each generation
- **AI Teacher**: Guides evolution direction based on market regime

## Contrarian & Counter-Indicator System
- **Fear/Greed Index**: Synthetic index from momentum, volatility, volume, RSI, streak
- **Retail Fader**: Detects FOMO buying and panic selling, then fades them
- **Funding Rate**: Simulates Binance funding rate, shorts extreme longs and vice versa
- **Wyckoff Psychology**: Detects accumulation/distribution phases with spring/upthrust patterns
- **Counter-Indicator**: Tracks individual agents' accuracy; inverts signals from agents with <40% win rate
- **Type Counter**: Tracks entire strategy categories; when "momentum" type is losing, inverts ALL momentum signals

## Sentiment Analysis System
- **Sentiment Simulator**: Generates crowd sentiment (-100 to +100) from price action using herding, amplification, and decay models
- **Follower vs Contrarian**: Two groups compete — one trades WITH the crowd, the other AGAINST extremes
- **Sentiment Momentum**: Trades sentiment acceleration (rate of change, not just level)

## Proven Bot Strategies
Adapted from real-world top-performing open-source trading bots:
- **NostalgiaForInfinityX**: Multi-condition entry (EMA + RSI + Bollinger + MFI)
- **CombinedBinHCluc**: Dual Bollinger Band approach (BinH + Cluc detection)
- **Scalping Momentum**: EMA 3x8 cross + RSI(7) + volume spike
- **Multi-Timeframe Trend**: 3-layer EMA alignment (short/medium/long)
