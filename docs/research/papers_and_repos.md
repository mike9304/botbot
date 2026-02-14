# Trading Research: Papers, Repos & Key Insights

## Academic Papers

### Multi-Agent Trading Systems

**1. "An Agent-Based Model of the Flash Crash of May 6, 2010" — Paddrik et al. (2012)**
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2182600
- Agent diversity and interaction effects matter more than individual sophistication.
- Taxonomy of agent types (fundamental, momentum, market maker, noise) maps to BotBot's 19-group architecture.

**2. "Multi-Agent Reinforcement Learning for Liquidation Strategy Analysis" — Bao & Liu (2019)**
- URL: https://arxiv.org/abs/1906.03107
- Multiple RL agents competing create non-stationary environments.
- Counter-indicator and loser league mechanics must account for agents adapting to each other.

### Genetic Algorithms for Trading

**3. "Genetic Programming for Financial Trading: A Survey" — Chen & Navet (2007)**
- URL: https://hal.inria.fr/inria-00169566
- Tournament selection with elitism consistently outperforms other methods (BotBot already uses this).
- **Key insight**: Crossover between compatible strategy types produces better offspring than random crossover — validates BotBot's group-based evolution.

**4. "Evolving Trading Strategies With GP: A Two-Market Study" — Lohpetch & Corne (2010)**
- URL: https://doi.org/10.1145/1830483.1830580
- Strategies evolved on multiple market conditions generalize 3-4x better than single-regime ones.
- AI Teacher's market regime detection should guide which conditions to evolve against.

### Reinforcement Learning for Trading

**5. "Deep RL for Automated Stock Trading: An Ensemble Strategy" — Yang et al. (2020)**
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996
- Ensemble of PPO, A2C, DDPG agents switches based on market regime (turbulence index).
- **Actionable**: Add turbulence index (Mahalanobis distance of returns) to enhance ADX-based regime detection.

**6. "Model-based RL for Limit Order Books" — Wei et al. (2019)**
- URL: https://arxiv.org/abs/1910.03743
- Learning a transition model of market states improves sample efficiency 5-10x over model-free Q-learning.

### Crowd Psychology & Sentiment in Crypto

**7. "What Are the Main Drivers of the Bitcoin Price?" — Kristoufek (2015)**
- URL: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0123923
- Sentiment is a **leading** indicator at extremes but **lagging** in normal conditions.
- Validates BotBot's dual approach: Followers work in trends, Faders work at extremes.
- "Extreme" = >2 standard deviations from 30-day sentiment MA.

**8. "Contrarian and Momentum Strategies in the Cryptocurrency Market" — Tzouvanas et al. (2020)**
- URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3594582
- Short-term (1-week) contrarian + medium-term (3-4 week) momentum both generate alpha.
- **Low volatility → momentum; High volatility → contrarian**.
- AI Teacher should shift agent mix accordingly.

---

## GitHub Repositories

### Top Freqtrade Strategies

**1. NostalgiaForInfinityX**
- URL: https://github.com/iterativv/NostalgiaForInfinity
- 40+ buy conditions with independent sell logic per entry type.
- Key innovation: tiered exit system + protection system (cooldown after losses).

**2. Freqtrade Community Strategies**
- URL: https://github.com/freqtrade/freqtrade-strategies
- Combined BinHV2 + ClucMay72018 and dozens of variations.
- "Informative pair" concept: use BTC signals to filter altcoin trades.

### RL-Based Trading Frameworks

**3. FinRL — Deep RL for Quantitative Finance**
- URL: https://github.com/AI4Finance-Foundation/FinRL (10,000+ stars)
- State: [balance, shares_held, MACD, RSI, CCI, ADX] — more expressive than discrete Q-learning.
- Ensemble method: train 3 agents, pick best per period.

**4. TensorTrade — Modular RL Trading**
- URL: https://github.com/tensortrade-org/tensortrade (4,500+ stars)
- **Key insight**: Use Sortino ratio as RL reward instead of raw PnL to prevent excessive risk-taking.

### Multi-Agent Simulation

**5. ABIDES — JP Morgan Agent-Based Simulator**
- URL: https://github.com/jpmorganchase/abides-jpmc-public
- Realistic order book with heterogeneous agents.
- Market impact model: when 70 agents trade, their actions affect prices for each other.

**6. Mesa — Agent-Based Modeling Framework**
- URL: https://github.com/projectmesa/mesa (2,500+ stars)
- DataCollector for systematic metric recording; BatchRunner for hyperparameter sweeps.

### Crypto-Specific

**7. Crypto-Signal**
- URL: https://github.com/CryptoSignal/Crypto-Signal (4,800+ stars)
- Battle-tested indicator calculations for Binance data (handles missing candles, gaps).

**8. Freqtrade**
- URL: https://github.com/freqtrade/freqtrade (30,000+ stars)
- Protection plugins (StoplossGuard, MaxDrawdown, CooldownPeriod) map to BotBot's risk manager.
- Hyperopt module uses Optuna — alternative to genetic evolution for parameter tuning.

---

## Actionable Improvements for BotBot

### Evolution System
1. **Strategy-aware crossover**: Only crossover between compatible types within groups; cross-group uses mutation-heavy operators.
2. **Multi-regime evaluation**: Each generation evaluated across trending + ranging + volatile data to prevent overfitting.

### Reinforcement Learning
3. **Risk-adjusted rewards**: Replace raw PnL with Sortino ratio in Q-Learning to discourage high-variance strategies.
4. **Turbulence index**: Use Mahalanobis distance of returns for more robust regime detection than ADX alone.

### Counter-Indicator & Loser League
5. **Regime-dependent inversion**: Invert more aggressively in high-volatility (contrarian works) and less in trending (momentum works).
6. **Feedback loop awareness**: When counter-indicators win, the agents they invert get fired, erasing the counter-indicator's edge.

### Sentiment Analysis
7. **Extreme-only contrarian**: Faders should only signal at >2σ extremes; flat at normal sentiment levels.
8. **Sentiment acceleration > level**: Trading rate-of-change of sentiment precedes price momentum.

### AI Teacher
9. **Regime-relative grading**: -2% in a crash ≠ -2% in a bull market. Grades should be benchmarked against regime expectations.
10. **Crowding detection**: With 70 agents, detect when too many converge on same positions and force diversification.
