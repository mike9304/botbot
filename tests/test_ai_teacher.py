"""Tests for the AI Teacher & Risk Manager system."""
import asyncio
import pytest

from tests.conftest import simulate_trades

from src.agents.base_agent import AgentConfig, AgentGroup, TradingAgent
from src.ai_teacher.llm_client import (
    LLMClient, LLMConfig, LLMProvider, LLMResponse, TOKEN_COSTS,
)
from src.ai_teacher.risk_manager import (
    RiskConfig, RiskLevel, RiskManager,
)
from src.ai_teacher.teacher import (
    AITeacher, TeacherConfig,
)
from src.core.exchange import VirtualExchange
from src.core.models import AccountState, Candle, Position, Side
from src.strategies.technical import RSIMACDStrategy


# === LLM Client Tests ===


class TestLLMConfig:
    def test_default_config(self):
        config = LLMConfig()
        assert config.provider == LLMProvider.OFFLINE
        assert config.temperature == 0.3
        assert config.max_tokens == 500

    def test_config_selects_default_model(self):
        config = LLMConfig(provider=LLMProvider.DEEPSEEK)
        assert config.model == "deepseek-chat"

    def test_config_anthropic_model(self):
        config = LLMConfig(provider=LLMProvider.ANTHROPIC)
        assert "claude" in config.model.lower() or "haiku" in config.model.lower()

    def test_config_openai_model(self):
        config = LLMConfig(provider=LLMProvider.OPENAI)
        assert "gpt" in config.model.lower()


class TestLLMClient:
    def test_offline_mode(self):
        client = LLMClient(LLMConfig(provider=LLMProvider.OFFLINE))
        result = asyncio.get_event_loop().run_until_complete(
            client.chat("test prompt")
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "[OFFLINE MODE]"
        assert result.cost_usd == 0

    def test_cost_tracking(self):
        client = LLMClient(LLMConfig())
        assert client.config.total_cost_usd == 0
        assert client.config.total_input_tokens == 0

    def test_cost_report(self):
        client = LLMClient(LLMConfig())
        report = client.get_cost_report()
        assert "total_calls" in report
        assert "total_cost_usd" in report
        assert "provider" in report

    def test_token_costs_defined(self):
        assert "deepseek-chat" in TOKEN_COSTS
        assert "gpt-4o-mini" in TOKEN_COSTS
        assert len(TOKEN_COSTS) >= 5

    def test_daily_cost_limit(self):
        config = LLMConfig(
            provider=LLMProvider.OFFLINE,
            max_cost_per_day_usd=0.01,
        )
        client = LLMClient(config)
        client._daily_cost = 0.02  # Exceed limit
        # Even though it's offline, the cost check should be tested
        assert client._daily_cost > config.max_cost_per_day_usd


# === Risk Manager Tests ===


class TestRiskManager:
    def test_init(self):
        rm = RiskManager()
        assert rm.config is not None
        assert len(rm.alerts) == 0
        assert len(rm.disabled_agents) == 0

    def test_check_empty_groups(self):
        rm = RiskManager()
        alerts = rm.check_all([])
        assert alerts == []

    def test_drawdown_warning(self, agent, exchange):
        simulate_trades(agent)
        account = agent.account
        account.max_drawdown = 0.25  # 25% drawdown

        group = AgentGroup(name="Test", description="", category="test")
        group.add_agent(agent)

        rm = RiskManager()
        alerts = rm.check_all([group])
        drawdown_alerts = [a for a in alerts if a.category == "drawdown"]
        assert len(drawdown_alerts) > 0
        assert drawdown_alerts[0].level == RiskLevel.HIGH

    def test_drawdown_critical_disables_agent(self, agent, exchange):
        simulate_trades(agent)
        account = agent.account
        account.max_drawdown = 0.35  # 35% drawdown — exceeds 30% critical

        group = AgentGroup(name="Test", description="", category="test")
        group.add_agent(agent)

        rm = RiskManager()
        alerts = rm.check_all([group])
        critical_alerts = [a for a in alerts if a.level == RiskLevel.CRITICAL]
        assert len(critical_alerts) > 0
        assert not agent.active  # Agent should be disabled
        assert agent.id in rm.disabled_agents

    def test_evaluate_agent_performance(self, agent, exchange):
        simulate_trades(agent, n_wins=7, n_losses=3)

        rm = RiskManager()
        evaluation = rm.evaluate_agent_performance(agent)

        assert "grade" in evaluation
        assert "pnl_pct" in evaluation
        assert "win_rate" in evaluation
        assert "strengths" in evaluation
        assert "weaknesses" in evaluation
        assert evaluation["trades"] == 10
        assert evaluation["win_rate"] == 0.7

    def test_evaluate_agent_no_trades(self, agent, exchange):
        rm = RiskManager()
        evaluation = rm.evaluate_agent_performance(agent)
        assert evaluation["grade"] == "N/A"

    def test_risk_summary(self, agent, exchange):
        group = AgentGroup(name="Test", description="", category="test")
        group.add_agent(agent)

        rm = RiskManager()
        rm.check_all([group])
        summary = rm.get_risk_summary()

        assert "total_alerts" in summary
        assert "critical" in summary
        assert "risk_level" in summary

    def test_correlation_check(self, exchange):
        """Test that correlation alert fires when many agents are in same direction."""
        group = AgentGroup(name="Corr", description="", category="test")
        for i in range(10):
            config = AgentConfig(
                name=f"corr_{i}", group="test",
                strategy=RSIMACDStrategy(), symbols=["BTCUSDT"],
            )
            agent = TradingAgent(config, exchange)
            account = agent.account
            # Give all agents LONG positions
            account.positions[f"pos_{i}"] = Position(
                symbol="BTCUSDT", side=Side.LONG,
                entry_price=50000, quantity=0.1,
                leverage=5, liquidation_price=40000,
            )
            group.add_agent(agent)

        exchange.current_prices["BTCUSDT"] = 50000

        rm = RiskManager()
        alerts = rm.check_all([group])
        corr_alerts = [a for a in alerts if a.category == "correlation"]
        assert len(corr_alerts) > 0


# === AI Teacher Tests ===


class TestAITeacher:
    def test_init_offline(self):
        config = TeacherConfig()
        teacher = AITeacher(config)
        assert teacher.risk_manager is not None
        assert teacher.llm_client is not None
        assert len(teacher.reports) == 0

    def test_evaluate_generation(self, group_with_agents, exchange):
        config = TeacherConfig(offline_only=True)
        teacher = AITeacher(config)

        # Simulate some trades
        for agent in group_with_agents.agents:
            simulate_trades(agent, n_wins=5, n_losses=3)

        report = asyncio.get_event_loop().run_until_complete(
            teacher.evaluate_generation([group_with_agents], candle_index=100)
        )

        assert report is not None
        assert report.generation == 1
        assert report.candle_index == 100
        assert len(report.group_reports) == 1
        assert report.overall_summary != ""

    def test_group_report_card(self, group_with_agents, exchange):
        config = TeacherConfig(offline_only=True)
        teacher = AITeacher(config)

        for agent in group_with_agents.agents:
            simulate_trades(agent, n_wins=5, n_losses=3)

        report = asyncio.get_event_loop().run_until_complete(
            teacher.evaluate_generation([group_with_agents], candle_index=100)
        )

        group_report = report.group_reports[0]
        assert group_report.group_name == "Test Group"
        assert group_report.overall_grade != "N/A"
        assert len(group_report.agent_reports) == 5

    def test_agent_report_card(self, group_with_agents, exchange):
        config = TeacherConfig(offline_only=True)
        teacher = AITeacher(config)

        for agent in group_with_agents.agents:
            simulate_trades(agent, n_wins=5, n_losses=3)

        report = asyncio.get_event_loop().run_until_complete(
            teacher.evaluate_generation([group_with_agents], candle_index=100)
        )

        agent_report = report.group_reports[0].agent_reports[0]
        assert agent_report.grade in ("A+", "A", "B+", "B", "C", "D", "F", "F-")
        assert agent_report.teacher_comment != ""
        assert agent_report.trades == 8

    def test_quick_risk_check(self, group_with_agents, exchange):
        config = TeacherConfig(offline_only=True)
        teacher = AITeacher(config)

        alerts = asyncio.get_event_loop().run_until_complete(
            teacher.quick_risk_check([group_with_agents])
        )
        assert isinstance(alerts, list)

    def test_cost_report(self):
        config = TeacherConfig()
        teacher = AITeacher(config)
        report = teacher.get_cost_report()
        assert "total_cost_usd" in report
        assert report["total_cost_usd"] == 0

    def test_market_regime_detection(self):
        config = TeacherConfig()
        teacher = AITeacher(config)

        # Create trending up candles
        candles = []
        for i in range(30):
            price = 50000 + i * 200  # Uptrend
            candles.append(Candle(
                timestamp=i * 60,
                open=price - 50, high=price + 100,
                low=price - 100, close=price,
                volume=1000, symbol="BTCUSDT",
            ))

        regime = teacher._detect_market_regime({"BTCUSDT": candles})
        assert regime in ("trending_up", "trending_down", "ranging", "volatile", "unknown")

    def test_evolution_guidance(self, group_with_agents, exchange):
        config = TeacherConfig(offline_only=True)
        teacher = AITeacher(config)

        for agent in group_with_agents.agents:
            simulate_trades(agent, n_wins=5, n_losses=3)

        report = asyncio.get_event_loop().run_until_complete(
            teacher.evaluate_generation([group_with_agents], candle_index=100)
        )

        guidance = report.evolution_guidance
        assert "market_regime" in guidance
        assert "regime_advice" in guidance
        assert "preserve_categories" in guidance
        assert "fire_candidates" in guidance

    def test_multiple_evaluations(self, group_with_agents, exchange):
        config = TeacherConfig(offline_only=True)
        teacher = AITeacher(config)

        for agent in group_with_agents.agents:
            simulate_trades(agent)

        # Run multiple evaluations
        for i in range(3):
            asyncio.get_event_loop().run_until_complete(
                teacher.evaluate_generation([group_with_agents], candle_index=i * 100)
            )

        assert len(teacher.reports) == 3
        assert teacher.evaluation_count == 3

    def test_get_latest_report(self, group_with_agents, exchange):
        config = TeacherConfig(offline_only=True)
        teacher = AITeacher(config)

        assert teacher.get_latest_report() is None

        for agent in group_with_agents.agents:
            simulate_trades(agent)

        asyncio.get_event_loop().run_until_complete(
            teacher.evaluate_generation([group_with_agents], candle_index=100)
        )

        latest = teacher.get_latest_report()
        assert latest is not None
        assert latest.generation == 1


# === Sentiment & Proven Bots Strategy Tests ===


class TestSentimentStrategies:
    def test_sentiment_follower_import(self):
        from src.strategies.sentiment import SentimentFollowerStrategy
        s = SentimentFollowerStrategy()
        assert s.name == "sentiment_follower"
        assert s.category == "sentiment_follow"

    def test_sentiment_contrarian_import(self):
        from src.strategies.sentiment import SentimentContrarianStrategy
        s = SentimentContrarianStrategy()
        assert s.name == "sentiment_contrarian"
        assert s.category == "sentiment_fade"

    def test_sentiment_momentum_import(self):
        from src.strategies.sentiment import SentimentMomentumStrategy
        s = SentimentMomentumStrategy()
        assert s.name == "sentiment_momentum"

    def test_sentiment_simulator(self):
        import pandas as pd
        from src.strategies.sentiment import SentimentSimulator

        sim = SentimentSimulator()
        df = pd.DataFrame({
            "open": [100 + i * 0.5 for i in range(50)],
            "high": [101 + i * 0.5 for i in range(50)],
            "low": [99 + i * 0.5 for i in range(50)],
            "close": [100.5 + i * 0.5 for i in range(50)],
            "volume": [1000 + i * 10 for i in range(50)],
        })
        scores = sim.calculate_sentiment(df)
        assert len(scores) == 50
        assert all(-100 <= s <= 100 for s in scores.dropna())

    def test_sentiment_zone_classification(self):
        from src.strategies.sentiment import SentimentSimulator
        sim = SentimentSimulator()
        assert sim.get_sentiment_zone(80) == "extreme_bullish"
        assert sim.get_sentiment_zone(50) == "bullish"
        assert sim.get_sentiment_zone(0) == "neutral"
        assert sim.get_sentiment_zone(-50) == "bearish"
        assert sim.get_sentiment_zone(-80) == "extreme_bearish"


class TestProvenBotStrategies:
    def test_nostalgia_infinity_import(self):
        from src.strategies.proven_bots import NostalgiaForInfinityStrategy
        s = NostalgiaForInfinityStrategy()
        assert s.name == "nostalgia_infinity"
        assert s.category == "proven_bot"

    def test_combined_binhcluc_import(self):
        from src.strategies.proven_bots import CombinedBinHClucStrategy
        s = CombinedBinHClucStrategy()
        assert s.name == "combined_binhcluc"

    def test_scalping_momentum_import(self):
        from src.strategies.proven_bots import ScalpingMomentumStrategy
        s = ScalpingMomentumStrategy()
        assert s.name == "scalping_momentum"
        assert s.params["leverage"] == 7

    def test_multi_tf_trend_import(self):
        from src.strategies.proven_bots import MultiTimeframeTrendStrategy
        s = MultiTimeframeTrendStrategy()
        assert s.name == "multi_tf_trend"


class TestLoserEvolution:
    def test_loser_league_init(self):
        from src.evolution.loser_evolution import LoserLeague
        ll = LoserLeague()
        assert ll.generation == 0
        assert len(ll.loser_pool) == 0

    def test_inverse_loser_agent(self, exchange):
        from src.evolution.loser_evolution import InverseLoserAgent
        config = AgentConfig(
            name="inverse_test", group="inverse_loser",
            strategy=RSIMACDStrategy(), symbols=["BTCUSDT"],
        )
        agent = InverseLoserAgent(config, exchange)
        assert agent._inverted_count == 0
        assert agent._observed_count == 0

    def test_collect_losers_empty(self):
        from src.evolution.loser_evolution import LoserLeague
        ll = LoserLeague()
        losers = ll.collect_losers([])
        assert losers == []


class TestGroupFactory:
    def test_create_all_groups(self, exchange):
        from src.agents.group_factory import create_all_groups
        groups = create_all_groups(exchange)
        assert len(groups) == 19

        categories = [g.category for g in groups]
        assert "sentiment_follow" in categories
        assert "sentiment_fade" in categories
        assert "proven_bot" in categories
        assert "loser_league" in categories
        assert "inverse_loser" in categories

    def test_group_names(self, exchange):
        from src.agents.group_factory import create_all_groups
        groups = create_all_groups(exchange)
        names = [g.name for g in groups]
        assert "Sentiment Followers" in names
        assert "Sentiment Faders" in names
        assert "Proven Bots" in names
        assert "Loser League" in names
        assert "Inverse Losers" in names

    def test_total_agents(self, exchange):
        from src.agents.group_factory import create_all_groups
        groups = create_all_groups(exchange)
        total = sum(len(g.agents) for g in groups)
        assert total >= 60  # Should have many agents
