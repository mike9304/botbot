"""AI Teacher — the intelligent overseer of the Trading Agent Arena.

The AI Teacher is a "선생님" (teacher) that:
1. Evaluates each agent's trading performance with detailed feedback
2. Identifies what agents are doing right/wrong
3. Suggests parameter adjustments based on market conditions
4. Guides the genetic evolution process (which traits to preserve/discard)
5. Provides risk management alerts and enforces discipline
6. Creates a "report card" for each agent group
7. Detects market regime changes and advises strategy adjustments

Architecture:
- Works in two modes:
  a) OFFLINE: Rule-based evaluation (no API cost, always available)
  b) ONLINE: LLM-powered analysis for deeper insights (requires API key)
- The offline mode handles 80% of evaluations
- LLM is only called for complex situations (regime changes, anomalies)

Usage:
    teacher = AITeacher(TeacherConfig(
        llm_config=LLMConfig(provider=LLMProvider.DEEPSEEK)  # Cheapest option
    ))
    # During simulation:
    report = await teacher.evaluate_generation(groups, market_context)
    # Teacher provides feedback to evolution system
    adjustments = teacher.suggest_evolution_adjustments(groups)
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.agents.base_agent import AgentGroup, TradingAgent
from src.core.models import Candle

from .llm_client import LLMClient, LLMConfig, LLMProvider
from .risk_manager import RiskAlert, RiskConfig, RiskLevel, RiskManager

logger = logging.getLogger(__name__)


TEACHER_SYSTEM_PROMPT = """You are an expert cryptocurrency futures trading teacher and risk manager.
You are overseeing a multi-agent trading simulation arena where different agent groups compete.

Your role:
1. Evaluate agent performance honestly — praise strengths, criticize weaknesses
2. Provide specific, actionable advice (not vague suggestions)
3. Identify market regime changes and how strategies should adapt
4. Flag dangerous behavior (excessive leverage, drawdown, correlation)
5. Suggest parameter adjustments based on the data

Communication style:
- Be direct and concise (Korean trading mentor style)
- Use trading terminology correctly
- Quantify everything — give specific numbers
- Grade agents: A+ (exceptional) through F- (terrible)
- Don't sugarcoat poor performance

Output format: JSON with keys: summary, agent_grades, advice, risk_alerts, market_regime"""


@dataclass
class TeacherConfig:
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    risk_config: RiskConfig = field(default_factory=RiskConfig)

    # Evaluation frequency
    evaluate_every_n_candles: int = 100  # Match evolution interval
    quick_check_every_n_candles: int = 25  # Lightweight risk check

    # When to use LLM vs offline rules
    use_llm_for_evaluations: bool = True  # Full eval → LLM
    use_llm_for_risk_only: bool = False  # If True, only use LLM for risk alerts
    offline_only: bool = False  # Force offline mode (no API calls)

    # Teacher personality
    strictness: float = 0.7  # 0=lenient, 1=harsh
    verbosity: int = 2  # 1=brief, 2=normal, 3=detailed


@dataclass
class AgentReportCard:
    """Report card for a single agent."""
    agent_id: str
    agent_name: str
    group_name: str
    strategy_name: str
    grade: str
    pnl_pct: float
    win_rate: float
    max_drawdown: float
    trades: int
    fitness: float
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    teacher_comment: str = ""
    parameter_suggestions: dict = field(default_factory=dict)


@dataclass
class GroupReportCard:
    """Report card for an entire group."""
    group_name: str
    category: str
    overall_grade: str
    total_pnl: float
    avg_win_rate: float
    best_agent: str
    worst_agent: str
    agent_reports: list[AgentReportCard] = field(default_factory=list)
    teacher_comment: str = ""
    evolution_advice: str = ""


@dataclass
class TeacherReport:
    """Full teacher evaluation report for one generation."""
    generation: int
    candle_index: int
    market_regime: str  # "trending_up", "trending_down", "ranging", "volatile"
    overall_summary: str
    group_reports: list[GroupReportCard] = field(default_factory=list)
    risk_alerts: list[RiskAlert] = field(default_factory=list)
    risk_summary: dict = field(default_factory=dict)
    evolution_guidance: dict = field(default_factory=dict)
    llm_response: str = ""
    llm_cost_usd: float = 0.0


class AITeacher:
    """The AI Teacher — oversees, evaluates, and guides trading agents.

    Like a real teacher:
    - Gives grades (A+ to F-)
    - Praises good work, criticizes poor performance
    - Provides actionable feedback
    - Manages classroom discipline (risk management)
    - Guides student development (evolution direction)
    """

    def __init__(self, config: Optional[TeacherConfig] = None):
        self.config = config or TeacherConfig()
        self.risk_manager = RiskManager(self.config.risk_config)
        self.llm_client = LLMClient(self.config.llm_config)
        self.reports: list[TeacherReport] = []
        self.evaluation_count = 0
        self._current_regime = "unknown"

    async def evaluate_generation(
        self,
        groups: list[AgentGroup],
        candle_index: int,
        recent_candles: Optional[dict[str, list[Candle]]] = None,
    ) -> TeacherReport:
        """Run a full evaluation of all agents and groups.

        This is called every evolution cycle (e.g., every 100 candles).
        """
        self.evaluation_count += 1
        logger.info(f"AI Teacher: Evaluation #{self.evaluation_count} at candle {candle_index}")

        # 1. Detect market regime FIRST (needed for regime-relative grading)
        market_regime = self._detect_market_regime(recent_candles)
        self._current_regime = market_regime

        # 2. Risk check (including turbulence + crowding)
        risk_alerts = self.risk_manager.check_all(groups)
        risk_alerts.extend(self.risk_manager.check_symbol_crowding(groups))
        risk_summary = self.risk_manager.get_risk_summary()

        # 3. Evaluate each group and agent (regime-relative grading)
        group_reports = []
        for group in groups:
            report = self._evaluate_group(group)
            group_reports.append(report)

        # 4. Generate evolution guidance
        evolution_guidance = self._generate_evolution_guidance(group_reports, market_regime)

        # 5. Build overall summary (offline)
        summary = self._build_offline_summary(group_reports, risk_summary, market_regime)

        # 6. Optionally enhance with LLM
        llm_response = ""
        llm_cost = 0.0
        if (
            not self.config.offline_only
            and self.config.use_llm_for_evaluations
            and self.config.llm_config.provider != LLMProvider.OFFLINE
        ):
            llm_result = await self._get_llm_evaluation(
                group_reports, risk_alerts, market_regime, candle_index
            )
            llm_response = llm_result.content
            llm_cost = llm_result.cost_usd

            # Parse LLM insights and enhance reports
            self._enhance_with_llm(group_reports, llm_response)

        report = TeacherReport(
            generation=self.evaluation_count,
            candle_index=candle_index,
            market_regime=market_regime,
            overall_summary=summary,
            group_reports=group_reports,
            risk_alerts=risk_alerts,
            risk_summary=risk_summary,
            evolution_guidance=evolution_guidance,
            llm_response=llm_response,
            llm_cost_usd=llm_cost,
        )

        self.reports.append(report)
        self._log_report(report)
        return report

    async def quick_risk_check(self, groups: list[AgentGroup]) -> list[RiskAlert]:
        """Lightweight risk check (no LLM, just rules)."""
        return self.risk_manager.check_all(groups)

    def _evaluate_group(self, group: AgentGroup) -> GroupReportCard:
        """Evaluate a single group's performance."""
        agent_reports = []
        for agent in group.agents:
            report = self._evaluate_agent(agent, group.name)
            agent_reports.append(report)

        # Group-level stats
        pnl_values = [r.pnl_pct for r in agent_reports if r.trades > 0]
        win_rates = [r.win_rate for r in agent_reports if r.trades > 0]

        if not pnl_values:
            overall_grade = "N/A"
            teacher_comment = "No trading activity yet."
        else:
            avg_pnl = sum(pnl_values) / len(pnl_values)
            avg_wr = sum(win_rates) / len(win_rates) if win_rates else 0
            overall_grade = self._calculate_group_grade(avg_pnl, avg_wr)
            teacher_comment = self._generate_group_comment(
                group.name, group.category, avg_pnl, avg_wr, agent_reports
            )

        best = max(agent_reports, key=lambda r: r.pnl_pct) if agent_reports else None
        worst = min(agent_reports, key=lambda r: r.pnl_pct) if agent_reports else None

        return GroupReportCard(
            group_name=group.name,
            category=group.category,
            overall_grade=overall_grade,
            total_pnl=round(group.total_pnl, 2),
            avg_win_rate=round(group.avg_win_rate, 4),
            best_agent=best.agent_name if best else "",
            worst_agent=worst.agent_name if worst else "",
            agent_reports=agent_reports,
            teacher_comment=teacher_comment,
            evolution_advice=self._generate_evolution_advice(group.category, agent_reports),
        )

    def _evaluate_agent(self, agent: TradingAgent, group_name: str) -> AgentReportCard:
        """Evaluate a single agent's performance with regime-relative grading.

        From multiple papers: -2% in a crash ≠ -2% in a bull market.
        Grades are adjusted based on the current market regime so agents
        aren't unfairly penalized for conditions beyond their control.
        """
        evaluation = self.risk_manager.evaluate_agent_performance(agent)

        # Regime-relative grade adjustment
        grade = evaluation["grade"]
        grade = self._adjust_grade_for_regime(
            grade, evaluation["pnl_pct"], agent.strategy.category
            if hasattr(agent.strategy, 'category') else "unknown"
        )
        evaluation["grade"] = grade

        # Teacher comment based on grade
        comment = self._generate_agent_comment(
            agent.config.name, evaluation["grade"],
            evaluation.get("strengths", []), evaluation.get("weaknesses", []),
        )

        # Parameter suggestions
        suggestions = self._suggest_parameters(agent, evaluation)

        return AgentReportCard(
            agent_id=agent.id,
            agent_name=agent.config.name,
            group_name=group_name,
            strategy_name=agent.strategy.name,
            grade=evaluation["grade"],
            pnl_pct=evaluation["pnl_pct"],
            win_rate=evaluation["win_rate"],
            max_drawdown=evaluation["max_drawdown"],
            trades=evaluation["trades"],
            fitness=evaluation["fitness"],
            strengths=evaluation.get("strengths", []),
            weaknesses=evaluation.get("weaknesses", []),
            teacher_comment=comment,
            parameter_suggestions=suggestions,
        )

    def _generate_agent_comment(
        self, name: str, grade: str, strengths: list, weaknesses: list,
    ) -> str:
        """Generate a teacher's comment for an agent (offline mode)."""
        strictness = self.config.strictness

        if grade in ("A+", "A"):
            comments = [
                f"{name}: Excellent work! Keep this strategy and parameters.",
                f"{name}: Top performer. This is how it's done.",
                f"{name}: Outstanding. Protect this agent's genes in evolution.",
            ]
        elif grade in ("B+", "B"):
            comments = [
                f"{name}: Good performance but room for improvement.",
                f"{name}: Solid results. Fine-tune parameters for better consistency.",
                f"{name}: Above average. Focus on reducing drawdown.",
            ]
        elif grade == "C":
            comments = [
                f"{name}: Mediocre. Not losing money but not making enough either.",
                f"{name}: Average performance. Needs parameter optimization.",
                f"{name}: Breaking even isn't enough. Push for better entries.",
            ]
        elif grade == "D":
            if strictness > 0.5:
                comments = [
                    f"{name}: Poor performance. On probation — improve or get fired.",
                    f"{name}: Underperforming badly. Strategy needs a major rethink.",
                ]
            else:
                comments = [
                    f"{name}: Struggling. Consider adjusting risk parameters.",
                    f"{name}: Below expectations. May need different market conditions.",
                ]
        else:  # F, F-
            if strictness > 0.5:
                comments = [
                    f"{name}: Terrible. Candidate for firing in next evolution cycle.",
                    f"{name}: Consistently wrong — might be useful as a counter-indicator.",
                    f"{name}: Failed. Genes should NOT be passed to next generation.",
                ]
            else:
                comments = [
                    f"{name}: Very poor results. Major changes needed.",
                    f"{name}: Consider moving to the Loser League for inversion.",
                ]

        comment = random.choice(comments)

        # Add specific feedback
        if weaknesses and strictness > 0.3:
            comment += f" Issues: {'; '.join(weaknesses[:2])}"

        return comment

    def _adjust_grade_for_regime(
        self, grade: str, pnl_pct: float, strategy_category: str,
    ) -> str:
        """Adjust grade based on market regime (regime-relative grading).

        From research: an agent losing 2% in a crash should be graded
        differently than one losing 2% in a bull market.
        Also: momentum agents are expected to do well in trends,
        mean reversion in ranges — grade relative to expected performance.
        """
        regime = self._current_regime
        grade_order = ["F-", "F", "D", "C", "B", "B+", "A", "A+"]

        def shift_grade(g: str, steps: int) -> str:
            try:
                idx = grade_order.index(g)
            except ValueError:
                return g
            new_idx = max(0, min(len(grade_order) - 1, idx + steps))
            return grade_order[new_idx]

        # Regime-based adjustment
        if regime == "volatile":
            # Volatile market: losing is expected, be lenient
            if pnl_pct < 0 and pnl_pct > -10:
                grade = shift_grade(grade, 1)  # Bump up one grade
            # Contrarian strategies expected to shine
            if strategy_category in ("contrarian", "counter_indicator", "sentiment_fade"):
                if pnl_pct < 0:
                    grade = shift_grade(grade, -1)  # Expect better from you

        elif regime in ("trending_up", "trending_down"):
            # Trending: momentum should do well, mean reversion should struggle
            if strategy_category in ("momentum", "sentiment_follow"):
                if pnl_pct < 0:
                    grade = shift_grade(grade, -1)  # Should be winning
            elif strategy_category in ("statistical", "regime"):
                if pnl_pct < 0:
                    grade = shift_grade(grade, 1)  # Expected to struggle

        elif regime == "ranging":
            # Range-bound: mean reversion expected to do well
            if strategy_category in ("statistical",):
                if pnl_pct < 0:
                    grade = shift_grade(grade, -1)
            elif strategy_category in ("momentum",):
                if pnl_pct < 0:
                    grade = shift_grade(grade, 1)

        return grade

    def _suggest_parameters(self, agent: TradingAgent, evaluation: dict) -> dict:
        """Suggest parameter adjustments based on performance."""
        suggestions = {}
        params = agent.strategy.get_params()

        # If win rate is low, suggest tighter entry conditions
        if evaluation["win_rate"] < 0.40 and evaluation["trades"] > 5:
            if "confidence_threshold" in params:
                suggestions["confidence_threshold"] = min(
                    params["confidence_threshold"] + 0.05, 0.8
                )
            if "leverage" in params and params["leverage"] > 3:
                suggestions["leverage"] = max(params["leverage"] - 2, 2)

        # If drawdown is high, reduce risk
        if evaluation["max_drawdown"] > 0.20:
            if "leverage" in params:
                suggestions["leverage"] = max(params["leverage"] // 2, 1)
            if "stop_loss_pct" in params:
                suggestions["stop_loss_pct"] = max(
                    params["stop_loss_pct"] * 0.7, 0.005
                )

        # If profitable but low trade count, loosen conditions
        if evaluation["pnl_pct"] > 3 and evaluation["trades"] < 5:
            if "confidence_threshold" in params:
                suggestions["confidence_threshold"] = max(
                    params["confidence_threshold"] - 0.05, 0.4
                )

        return suggestions

    def _calculate_group_grade(self, avg_pnl: float, avg_wr: float) -> str:
        """Calculate overall group grade."""
        score = avg_pnl * 0.6 + (avg_wr - 0.5) * 100 * 0.4
        if score >= 8:
            return "A+"
        elif score >= 5:
            return "A"
        elif score >= 3:
            return "B+"
        elif score >= 1:
            return "B"
        elif score >= -1:
            return "C"
        elif score >= -5:
            return "D"
        else:
            return "F"

    def _generate_group_comment(
        self, name: str, category: str, avg_pnl: float, avg_wr: float,
        agents: list[AgentReportCard],
    ) -> str:
        """Generate a teacher's comment for a group."""
        n_profitable = sum(1 for a in agents if a.pnl_pct > 0)
        n_total = len([a for a in agents if a.trades > 0])

        if avg_pnl > 5:
            tone = "Excellent group performance!"
        elif avg_pnl > 0:
            tone = "Positive overall, but inconsistent."
        elif avg_pnl > -5:
            tone = "Struggling. Needs adjustment."
        else:
            tone = "Severely underperforming. Consider strategy overhaul."

        return (
            f"[{name}] {tone} "
            f"Avg PnL: {avg_pnl:+.1f}%, Win rate: {avg_wr:.0%}, "
            f"Profitable agents: {n_profitable}/{n_total}"
        )

    def _generate_evolution_advice(
        self, category: str, agents: list[AgentReportCard],
    ) -> str:
        """Advise on how to evolve this group."""
        if not agents:
            return "No data yet."

        profitable = [a for a in agents if a.pnl_pct > 0]
        losing = [a for a in agents if a.pnl_pct < 0 and a.trades > 3]

        if len(profitable) > len(losing):
            return (
                f"Evolution: PRESERVE top performers. "
                f"Crossover {profitable[0].agent_name}'s params with others. "
                f"Mutation rate can be LOW (these genes work)."
            )
        else:
            return (
                f"Evolution: AGGRESSIVE mutation needed. "
                f"Current params aren't working for {category}. "
                f"Try wider parameter ranges and more crossover diversity."
            )

    def _detect_market_regime(
        self, recent_candles: Optional[dict[str, list[Candle]]] = None,
    ) -> str:
        """Detect current market regime from recent price data."""
        if not recent_candles:
            return "unknown"

        # Use BTC as market proxy
        btc_candles = recent_candles.get("BTCUSDT", [])
        if len(btc_candles) < 20:
            return "unknown"

        closes = [c.close for c in btc_candles[-20:]]
        if len(closes) < 20:
            return "unknown"

        returns = np.diff(closes) / closes[:-1]

        trend = (closes[-1] - closes[0]) / closes[0]
        volatility = np.std(returns)

        if abs(trend) > 0.05 and volatility < 0.03:
            return "trending_up" if trend > 0 else "trending_down"
        elif volatility > 0.04:
            return "volatile"
        else:
            return "ranging"

    def _generate_evolution_guidance(
        self, group_reports: list[GroupReportCard], market_regime: str,
    ) -> dict:
        """Generate guidance for the genetic evolution system."""
        guidance = {
            "market_regime": market_regime,
            "regime_advice": "",
            "preserve_categories": [],
            "mutate_heavily": [],
            "fire_candidates": [],
            "promising_hybrids": [],
        }

        # Regime-specific advice
        regime_advice = {
            "trending_up": (
                "Trending UP market. Favor momentum and trend-following strategies. "
                "Reduce mean reversion weight. Increase leverage for trend riders."
            ),
            "trending_down": (
                "Trending DOWN market. Short-bias strategies should outperform. "
                "Contrarian long entries need higher thresholds. "
                "Counter-indicators may struggle (everyone is short)."
            ),
            "volatile": (
                "Volatile/choppy market. Reduce leverage across all groups. "
                "Mean reversion and scalping should do well. "
                "Trend followers will get whipsawed — tighten stops."
            ),
            "ranging": (
                "Range-bound market. Mean reversion is king. "
                "Momentum strategies will generate false signals. "
                "Reduce trade frequency for breakout strategies."
            ),
            "unknown": "Insufficient data to determine regime. Maintain balanced exposure.",
        }
        guidance["regime_advice"] = regime_advice.get(market_regime, regime_advice["unknown"])

        # Categorize groups
        for report in group_reports:
            if report.overall_grade in ("A+", "A"):
                guidance["preserve_categories"].append(report.category)
            elif report.overall_grade in ("D", "F", "F-"):
                guidance["mutate_heavily"].append(report.category)

            # Find agents to fire
            for agent in report.agent_reports:
                if agent.grade in ("F", "F-") and agent.trades > 5:
                    guidance["fire_candidates"].append({
                        "name": agent.agent_name,
                        "group": report.group_name,
                        "reason": f"Grade {agent.grade}, PnL {agent.pnl_pct:+.1f}%",
                    })

        return guidance

    def _build_offline_summary(
        self, group_reports: list[GroupReportCard],
        risk_summary: dict, market_regime: str,
    ) -> str:
        """Build overall summary without LLM (rule-based)."""
        total_groups = len(group_reports)
        profitable_groups = sum(1 for g in group_reports if g.total_pnl > 0)
        risk_level = risk_summary.get("risk_level", "LOW")

        lines = [
            f"=== AI Teacher Report (Generation {self.evaluation_count}) ===",
            f"Market Regime: {market_regime.upper()}",
            f"Risk Level: {risk_level}",
            f"Groups: {profitable_groups}/{total_groups} profitable",
            "",
        ]

        # Top and bottom groups
        sorted_groups = sorted(group_reports, key=lambda g: g.total_pnl, reverse=True)
        if sorted_groups:
            best = sorted_groups[0]
            worst = sorted_groups[-1]
            lines.append(f"Best Group: {best.group_name} ({best.overall_grade}) PnL: ${best.total_pnl:+.2f}")
            lines.append(f"Worst Group: {worst.group_name} ({worst.overall_grade}) PnL: ${worst.total_pnl:+.2f}")

        # Risk alerts
        if risk_summary.get("critical", 0) > 0:
            lines.append(f"\n⚠ CRITICAL ALERTS: {risk_summary['critical']}")
        if risk_summary.get("disabled_agents", 0) > 0:
            lines.append(f"Disabled agents: {risk_summary['disabled_agents']}")

        return "\n".join(lines)

    async def _get_llm_evaluation(
        self, group_reports: list[GroupReportCard],
        risk_alerts: list[RiskAlert], market_regime: str,
        candle_index: int,
    ):
        """Get enhanced evaluation from LLM."""
        # Build concise prompt with key data
        data = {
            "generation": self.evaluation_count,
            "candle": candle_index,
            "regime": market_regime,
            "groups": [],
        }

        for g in group_reports:
            group_data = {
                "name": g.group_name,
                "category": g.category,
                "grade": g.overall_grade,
                "pnl": g.total_pnl,
                "wr": round(g.avg_win_rate, 2),
                "best": g.best_agent,
                "worst": g.worst_agent,
            }
            # Include top/bottom agent details
            if g.agent_reports:
                sorted_agents = sorted(g.agent_reports, key=lambda a: a.pnl_pct, reverse=True)
                group_data["top_agent"] = {
                    "name": sorted_agents[0].agent_name,
                    "pnl": sorted_agents[0].pnl_pct,
                    "wr": sorted_agents[0].win_rate,
                }
                group_data["bottom_agent"] = {
                    "name": sorted_agents[-1].agent_name,
                    "pnl": sorted_agents[-1].pnl_pct,
                    "wr": sorted_agents[-1].win_rate,
                }
            data["groups"].append(group_data)

        # Add risk alerts
        if risk_alerts:
            data["risk_alerts"] = [
                {"level": a.level.value, "msg": a.message}
                for a in risk_alerts[:5]  # Limit to top 5
            ]

        prompt = (
            f"Evaluate this trading simulation state and provide specific advice.\n"
            f"Data:\n{json.dumps(data, indent=2)}\n\n"
            f"Respond in JSON with: summary (2-3 sentences), "
            f"top_advice (3 specific actions), "
            f"regime_strategy (what to do given {market_regime} market), "
            f"fire_list (agents that should be replaced)"
        )

        return await self.llm_client.chat(prompt, system=TEACHER_SYSTEM_PROMPT)

    def _enhance_with_llm(
        self, group_reports: list[GroupReportCard], llm_response: str,
    ):
        """Parse LLM response and enhance group reports."""
        try:
            # Try to parse JSON from the response
            # Handle markdown code blocks
            content = llm_response.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            insights = json.loads(content)

            # Add LLM summary to each group if available
            if "top_advice" in insights:
                advice = insights["top_advice"]
                if isinstance(advice, list):
                    for i, report in enumerate(group_reports):
                        if i < len(advice):
                            report.teacher_comment += f" | AI: {advice[i]}"

        except (json.JSONDecodeError, KeyError, IndexError):
            # LLM response wasn't valid JSON — use as raw text
            logger.debug(f"Could not parse LLM response as JSON: {llm_response[:200]}")

    def _log_report(self, report: TeacherReport):
        """Log the teacher's report."""
        logger.info(report.overall_summary)
        for alert in report.risk_alerts:
            if alert.level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
                logger.warning(f"Risk: {alert.message}")

    def get_cost_report(self) -> dict:
        """Get API cost report from the LLM client."""
        return self.llm_client.get_cost_report()

    def get_latest_report(self) -> Optional[TeacherReport]:
        """Get the most recent evaluation report."""
        return self.reports[-1] if self.reports else None

    async def close(self):
        """Clean up resources."""
        await self.llm_client.close()
