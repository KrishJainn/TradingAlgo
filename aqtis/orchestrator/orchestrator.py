"""
AQTIS Multi-Agent Orchestrator.

Coordinates all agents and manages trading workflows:
- Pre-trade: signal analysis -> backtest -> risk check -> execute
- Post-trade: outcome recording -> post-mortem -> strategy updates
- Daily: research scan -> model check -> rolling backtests
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from aqtis.memory.memory_layer import MemoryLayer
from aqtis.llm.base import LLMProvider
from aqtis.agents.strategy_generator import StrategyGeneratorAgent
from aqtis.agents.backtester import BacktestingAgent
from aqtis.agents.risk_manager import RiskManagementAgent
from aqtis.agents.researcher import ResearchAgent
from aqtis.agents.post_mortem import PostMortemAgent
from aqtis.agents.prediction_tracker import PredictionTrackingAgent

logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    Coordinates all AQTIS agents through structured workflows.
    """

    def __init__(
        self,
        memory: MemoryLayer,
        llm: Optional[LLMProvider] = None,
        config: Dict = None,
    ):
        self.memory = memory
        self.llm = llm
        self.config = config or {}

        # Initialize all agents
        self.agents = {
            "strategy_generator": StrategyGeneratorAgent(memory, llm),
            "backtester": BacktestingAgent(memory, llm),
            "risk_manager": RiskManagementAgent(
                memory, llm,
                risk_limits=config.get("risk_limits") if config else None,
            ),
            "researcher": ResearchAgent(memory, llm),
            "post_mortem": PostMortemAgent(memory, llm),
            "prediction_tracker": PredictionTrackingAgent(memory, llm),
        }

    # ─────────────────────────────────────────────────────────────────
    # PRE-TRADE WORKFLOW
    # ─────────────────────────────────────────────────────────────────

    def pre_trade_workflow(self, market_signal: Dict) -> Dict:
        """
        Orchestrate pre-trade decision making.

        Flow:
        1. Strategy Generator: identify opportunity
        2. Backtester: instant backtest
        3. Risk Manager: validate trade
        4. Prediction Tracker: log prediction
        5. Return decision
        """
        logger.info("Starting pre-trade workflow")

        # 1. Strategy Generator identifies opportunity
        opportunity = self.agents["strategy_generator"].run({
            "action": "analyze_signal",
            "signal": market_signal,
        })

        if opportunity.get("error"):
            return {"decision": "error", "reason": opportunity["error"]}

        if not opportunity.get("should_trade", False):
            return {
                "decision": "skip",
                "reason": opportunity.get("reason", "No opportunity"),
            }

        strategy = opportunity.get("strategy", {})

        # 2. Backtest the opportunity
        backtest_result = self.agents["backtester"].run({
            "action": "instant",
            "strategy": strategy,
            "signal": market_signal,
        })

        if backtest_result.get("insufficient_data"):
            logger.warning("Insufficient backtest data, proceeding with caution")

        # 3. Risk validation
        portfolio_value = self.config.get("portfolio_value", 100000)
        confidence = backtest_result.get("confidence", 0.5)

        proposed_trade = {
            "asset": market_signal.get("asset", ""),
            "strategy_id": strategy.get("strategy_id", ""),
            "action": market_signal.get("action", "BUY"),
            "entry_price": market_signal.get("price", 0),
            "confidence": confidence,
            "predicted_return": backtest_result.get("expected_return", 0),
            "portfolio_value": portfolio_value,
        }

        risk_check = self.agents["risk_manager"].run({
            "action": "validate",
            "trade": proposed_trade,
        })

        if not risk_check.get("approved", False):
            return {
                "decision": "reject",
                "reason": risk_check.get("rejection_reasons", ["Risk check failed"]),
                "checks": risk_check.get("checks", {}),
            }

        # 4. Calculate position size
        position_result = self.agents["risk_manager"].run({
            "action": "position_size",
            "prediction": {
                "predicted_confidence": confidence,
                "predicted_return": backtest_result.get("expected_return", 0),
                "asset": market_signal.get("asset", ""),
            },
            "portfolio_value": portfolio_value,
        })

        position_size = position_result.get("position_size", 0)

        # 5. Log prediction
        prediction_result = self.agents["prediction_tracker"].run({
            "action": "record_prediction",
            "prediction": {
                "strategy_id": strategy.get("strategy_id", ""),
                "asset": market_signal.get("asset", ""),
                "predicted_return": backtest_result.get("expected_return", 0),
                "predicted_confidence": confidence,
                "win_probability": backtest_result.get("win_probability", 0.5),
                "predicted_hold_seconds": int(backtest_result.get("expected_hold_seconds", 0)),
                "primary_model": "ensemble",
                "market_features": market_signal,
            },
        })

        prediction_id = prediction_result.get("prediction_id")

        return {
            "decision": "execute",
            "prediction_id": prediction_id,
            "position_size": position_size,
            "strategy": strategy,
            "backtest": backtest_result,
            "risk_check": risk_check,
            "details": {
                "asset": market_signal.get("asset"),
                "action": market_signal.get("action", "BUY"),
                "confidence": confidence,
                "expected_return": backtest_result.get("expected_return", 0),
            },
        }

    # ─────────────────────────────────────────────────────────────────
    # POST-TRADE WORKFLOW
    # ─────────────────────────────────────────────────────────────────

    def post_trade_workflow(self, trade_id: str) -> Dict:
        """
        Orchestrate post-trade analysis.

        Flow:
        1. Record prediction outcome
        2. Deep post-mortem analysis
        3. Strategy updates if actionable insights found
        """
        logger.info(f"Starting post-trade workflow for {trade_id}")

        trade = self.memory.get_trade(trade_id)
        if not trade:
            return {"error": f"Trade {trade_id} not found"}

        # 1. Record prediction outcome
        if trade.get("prediction_id"):
            self.agents["prediction_tracker"].run({
                "action": "record_outcome",
                "prediction_id": trade["prediction_id"],
                "outcome": {
                    "actual_return": trade.get("pnl_percent", 0),
                    "actual_hold_seconds": trade.get("hold_duration_seconds", 0),
                    "actual_max_drawdown": trade.get("max_adverse_excursion", 0),
                },
            })

        # 2. Deep analysis
        analysis = self.agents["post_mortem"].run({
            "action": "analyze_trade",
            "trade_id": trade_id,
        })

        # 3. Strategy updates if needed
        insights = analysis.get("insights")
        if insights and isinstance(insights, dict) and insights.get("actionable_changes"):
            logger.info(f"Actionable changes found for strategy {trade.get('strategy_id')}")
            # Queue strategy improvement (don't auto-apply)

        return {
            "trade_id": trade_id,
            "analysis": analysis,
            "prediction_updated": bool(trade.get("prediction_id")),
        }

    # ─────────────────────────────────────────────────────────────────
    # DAILY ROUTINE
    # ─────────────────────────────────────────────────────────────────

    def daily_routine(self) -> Dict:
        """
        Run daily maintenance tasks.

        Flow:
        1. Research scan
        2. Model degradation check
        3. Rolling backtests for active strategies
        4. Weekly review (if Monday)
        """
        logger.info("Starting daily routine")
        results = {}

        # 1. Research scan
        research = self.agents["researcher"].run({"action": "scan"})
        results["research"] = {
            "papers_scanned": research.get("papers_scanned", 0),
            "relevant_papers": research.get("relevant_papers", 0),
        }

        # 2. Model degradation check
        degradation = self.agents["prediction_tracker"].run({
            "action": "detect_degradation",
        })
        results["degradation"] = degradation.get("degradation_alerts", [])

        if results["degradation"]:
            logger.warning(f"Model degradation detected: {results['degradation']}")

        # 3. Rolling backtests for active strategies
        strategies = self.memory.get_active_strategies()
        backtest_results = {}
        for strategy in strategies:
            bt = self.agents["backtester"].run({
                "action": "rolling",
                "strategy": strategy,
            })
            backtest_results[strategy.get("strategy_id", "")] = {
                "degradation_detected": bt.get("degradation_detected", False),
                "recent_sharpe": bt.get("recent_sharpe", 0),
            }
        results["backtests"] = backtest_results

        # 4. Weekly review (Monday)
        if datetime.now().weekday() == 0:
            review = self.agents["post_mortem"].run({"action": "weekly_review"})
            results["weekly_review"] = review

        return results

    # ─────────────────────────────────────────────────────────────────
    # STATUS
    # ─────────────────────────────────────────────────────────────────

    def get_agent_statuses(self) -> Dict:
        """Get status of all agents."""
        return {name: agent.get_status() for name, agent in self.agents.items()}

    def get_system_health(self) -> Dict:
        """Get overall system health."""
        statuses = self.get_agent_statuses()
        memory_stats = self.memory.get_stats()

        return {
            "agents": statuses,
            "memory": memory_stats,
            "circuit_breaker": self.agents["risk_manager"].circuit_breaker_active,
            "timestamp": datetime.now().isoformat(),
        }
