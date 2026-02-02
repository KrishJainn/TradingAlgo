"""
AQTIS Post-Mortem Agent.

Analyzes completed trades to extract learnings, update strategy
parameters, and generate natural language insights.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAgent

logger = logging.getLogger(__name__)


class PostMortemAgent(BaseAgent):
    """
    Deep analysis of completed trades.

    Capabilities:
    - Compare prediction vs actual outcome
    - Identify what went right/wrong
    - Extract patterns from wins and losses
    - Generate natural language insights for memory
    - Weekly performance reviews
    """

    def __init__(self, memory, llm=None):
        super().__init__(name="post_mortem", memory=memory, llm=llm)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute post-mortem action."""
        action = context.get("action", "analyze_trade")

        if action == "analyze_trade":
            return self.analyze_trade(context["trade_id"])
        elif action == "weekly_review":
            return self.weekly_performance_review()
        elif action == "extract_lessons":
            return self.extract_lessons(context.get("lookback_days", 30))
        else:
            return {"error": f"Unknown action: {action}"}

    # ─────────────────────────────────────────────────────────────────
    # TRADE ANALYSIS
    # ─────────────────────────────────────────────────────────────────

    def analyze_trade(self, trade_id: str) -> Dict:
        """Deep analysis of a completed trade."""
        trade = self.memory.get_trade(trade_id)
        if not trade:
            return {"error": f"Trade {trade_id} not found"}

        # Get prediction if available
        prediction = None
        if trade.get("prediction_id"):
            prediction = self.memory.get_prediction(trade["prediction_id"])

        # Calculate errors
        errors = self._calculate_errors(trade, prediction)

        # Find similar trades for comparison
        similar = self.memory.get_similar_trades(trade, top_k=20)

        # Statistical comparison
        stats = self._compare_with_similar(trade, similar)

        # Generate LLM insights if available
        insights = None
        if self.llm and self.llm.is_available():
            insights = self._generate_insights(trade, prediction, errors, similar)

        analysis = {
            "trade_id": trade_id,
            "trade": {
                "asset": trade.get("asset"),
                "strategy_id": trade.get("strategy_id"),
                "action": trade.get("action"),
                "pnl": trade.get("pnl"),
                "pnl_percent": trade.get("pnl_percent"),
                "market_regime": trade.get("market_regime"),
            },
            "errors": errors,
            "comparison_stats": stats,
            "insights": insights,
            "outcome": "win" if (trade.get("pnl") or 0) > 0 else "loss",
        }

        # Store analysis back to memory as a trade pattern
        self._store_lessons(trade_id, analysis)

        return analysis

    def _calculate_errors(self, trade: Dict, prediction: Optional[Dict]) -> Dict:
        """Calculate prediction errors."""
        if not prediction:
            return {"no_prediction": True}

        actual_return = trade.get("pnl_percent", 0) or 0
        predicted_return = prediction.get("predicted_return", 0) or 0
        predicted_confidence = prediction.get("predicted_confidence", 0.5)

        return {
            "return_error": abs(predicted_return - actual_return),
            "direction_correct": (predicted_return > 0) == (actual_return > 0),
            "confidence_error": abs(predicted_confidence - (1.0 if actual_return > 0 else 0.0)),
            "predicted_return": predicted_return,
            "actual_return": actual_return,
        }

    def _compare_with_similar(self, trade: Dict, similar: List[Dict]) -> Dict:
        """Compare trade outcome with similar historical trades."""
        if not similar:
            return {"no_similar_trades": True}

        pnls = [t.get("pnl_percent", 0) or 0 for t in similar]
        trade_pnl = trade.get("pnl_percent", 0) or 0

        return {
            "similar_trades_count": len(similar),
            "avg_similar_return": float(np.mean(pnls)),
            "median_similar_return": float(np.median(pnls)),
            "similar_win_rate": sum(1 for p in pnls if p > 0) / len(pnls),
            "trade_vs_average": trade_pnl - float(np.mean(pnls)),
            "percentile": float(sum(1 for p in pnls if p < trade_pnl) / len(pnls)),
        }

    def _generate_insights(
        self, trade: Dict, prediction: Optional[Dict], errors: Dict, similar: List[Dict]
    ) -> Dict:
        """Use LLM to extract nuanced insights."""
        similar_summary = []
        for t in similar[:5]:
            similar_summary.append({
                "asset": t.get("asset"),
                "pnl_percent": t.get("pnl_percent"),
                "regime": t.get("market_regime"),
                "strategy": t.get("strategy_id"),
            })

        prompt = f"""Analyze this completed trade and extract learnings.

TRADE:
Asset: {trade.get('asset')}
Strategy: {trade.get('strategy_id')}
Action: {trade.get('action')}
Entry: {trade.get('entry_price')} -> Exit: {trade.get('exit_price')}
P&L: {trade.get('pnl_percent', 0):.2%}
Regime: {trade.get('market_regime')}

PREDICTION ERRORS:
{json.dumps(errors, indent=2, default=str)}

SIMILAR TRADES:
{json.dumps(similar_summary, indent=2, default=str)}

Respond in JSON:
{{
    "outcome_summary": "Why the trade worked/failed",
    "primary_factors": ["factor1", "factor2"],
    "error_attribution": {{"model": 0.5, "execution": 0.2, "randomness": 0.3}},
    "lessons_learned": ["lesson1", "lesson2"],
    "actionable_changes": ["change1", "change2"]
}}"""

        return self.llm.generate_json(prompt)

    def _store_lessons(self, trade_id: str, analysis: Dict):
        """Store lessons learned back to memory."""
        trade = analysis.get("trade", {})
        insights = analysis.get("insights")
        lessons = ""
        if insights and isinstance(insights, dict):
            lessons = ". ".join(insights.get("lessons_learned", []))

        description = (
            f"Post-mortem: {trade.get('action', '')} {trade.get('asset', '')} "
            f"using {trade.get('strategy_id', '')}. "
            f"Outcome: {analysis.get('outcome', 'unknown')}. "
            f"P&L: {trade.get('pnl_percent', 0):.2%}. "
            f"Regime: {trade.get('market_regime', 'unknown')}. "
            f"Lessons: {lessons}"
        )

        self.memory.vectors.add_trade_pattern({
            "trade_id": f"analysis_{trade_id}",
            "text": description,
            "metadata": {
                "type": "post_mortem",
                "trade_id": trade_id,
                "outcome": analysis.get("outcome", "unknown"),
                "strategy_id": trade.get("strategy_id", ""),
            },
        })

    # ─────────────────────────────────────────────────────────────────
    # WEEKLY REVIEW
    # ─────────────────────────────────────────────────────────────────

    def weekly_performance_review(self) -> Dict:
        """Aggregate learnings from past week's trades."""
        cutoff = (datetime.now() - timedelta(days=7)).isoformat()
        trades = self.memory.get_trades(start_date=cutoff)

        if not trades:
            return {"message": "No trades in the past week"}

        # Group by strategy
        by_strategy: Dict[str, List] = {}
        for trade in trades:
            sid = trade.get("strategy_id", "unknown")
            by_strategy.setdefault(sid, []).append(trade)

        # Analyze each strategy
        strategy_reviews = {}
        for sid, strades in by_strategy.items():
            pnls = [t.get("pnl_percent", 0) or 0 for t in strades]
            wins = sum(1 for p in pnls if p > 0)

            strategy_reviews[sid] = {
                "total_trades": len(strades),
                "wins": wins,
                "losses": len(strades) - wins,
                "win_rate": wins / len(strades),
                "total_pnl": sum(t.get("pnl", 0) or 0 for t in strades),
                "avg_return": float(np.mean(pnls)),
            }

        # Overall stats
        all_pnls = [t.get("pnl_percent", 0) or 0 for t in trades]
        overall = {
            "total_trades": len(trades),
            "total_pnl": sum(t.get("pnl", 0) or 0 for t in trades),
            "avg_return": float(np.mean(all_pnls)),
            "win_rate": sum(1 for p in all_pnls if p > 0) / len(all_pnls),
        }

        # LLM synthesis
        synthesis = None
        if self.llm and self.llm.is_available():
            prompt = f"""Weekly trading performance review.

STRATEGY PERFORMANCE:
{json.dumps(strategy_reviews, indent=2, default=str)}

OVERALL:
{json.dumps(overall, indent=2, default=str)}

Provide strategic recommendations in JSON:
{{
    "best_strategy": "...",
    "worst_strategy": "...",
    "key_insights": ["insight1", "insight2"],
    "focus_next_week": ["action1", "action2"],
    "regime_observations": "..."
}}"""
            synthesis = self.llm.generate_json(prompt)

        return {
            "period": f"Last 7 days (from {cutoff[:10]})",
            "overall": overall,
            "strategy_reviews": strategy_reviews,
            "synthesis": synthesis,
        }

    # ─────────────────────────────────────────────────────────────────
    # LESSON EXTRACTION
    # ─────────────────────────────────────────────────────────────────

    def extract_lessons(self, lookback_days: int = 30) -> Dict:
        """Extract key lessons from recent trading activity."""
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        trades = self.memory.get_trades(start_date=cutoff)

        if not trades:
            return {"message": f"No trades in the past {lookback_days} days"}

        # Separate wins and losses
        wins = [t for t in trades if (t.get("pnl") or 0) > 0]
        losses = [t for t in trades if (t.get("pnl") or 0) <= 0]

        # Common patterns
        win_regimes = [t.get("market_regime", "unknown") for t in wins]
        loss_regimes = [t.get("market_regime", "unknown") for t in losses]

        win_strategies = [t.get("strategy_id", "unknown") for t in wins]
        loss_strategies = [t.get("strategy_id", "unknown") for t in losses]

        return {
            "period_days": lookback_days,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_regimes": self._count_items(win_regimes),
            "loss_regimes": self._count_items(loss_regimes),
            "win_strategies": self._count_items(win_strategies),
            "loss_strategies": self._count_items(loss_strategies),
        }

    def _count_items(self, items: List[str]) -> Dict[str, int]:
        """Count occurrences of items."""
        counts: Dict[str, int] = {}
        for item in items:
            counts[item] = counts.get(item, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
