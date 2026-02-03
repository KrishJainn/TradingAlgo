"""
AQTIS Post-Mortem Agent.

Analyzes completed trades to extract learnings, update strategy
parameters, and generate natural language insights.

Enhanced with 5-player coach model capabilities:
- Structured mistake taxonomy (15 types from PostMarketAnalyzer)
- Indicator contribution scoring per trade
- Coach session recording for cross-run learning
- Regime-specific diagnosis
"""

import json
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAgent

logger = logging.getLogger(__name__)


class MistakeType(str, Enum):
    """Structured mistake taxonomy from 5-player coach model."""
    WRONG_DIRECTION = "wrong_direction"
    EARLY_EXIT = "early_exit"
    LATE_EXIT = "late_exit"
    OVERSIZED = "oversized"
    UNDERSIZED = "undersized"
    BAD_ENTRY_TIMING = "bad_entry_timing"
    IGNORED_REGIME = "ignored_regime"
    COUNTER_TREND = "counter_trend"
    LOW_VOLUME_ENTRY = "low_volume_entry"
    STOP_TOO_TIGHT = "stop_too_tight"
    STOP_TOO_WIDE = "stop_too_wide"
    OVERTRADING = "overtrading"
    REVENGE_TRADE = "revenge_trade"
    CHASED_MOVE = "chased_move"
    IGNORED_INDICATOR = "ignored_indicator"


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
        """
        Deep analysis of a completed trade.

        Enhanced with 5-player coach model capabilities:
        - Structured mistake classification
        - Indicator contribution scoring
        - Coach session recording for cross-run learning
        - Regime-aware diagnosis
        """
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

        # Classify mistakes (from 5-player coach model)
        mistakes = self._classify_mistakes(trade, prediction, errors)

        # Score indicators for this trade
        indicator_scores = self._score_indicators(trade)

        # Get regime-specific context
        regime = trade.get("market_regime", "unknown")
        regime_indicators = self.memory.get_top_indicators_for_regime(regime, limit=5)

        # Generate LLM insights if available (enhanced with diagnosis)
        insights = None
        if self.llm and self.llm.is_available():
            insights = self._generate_insights(
                trade, prediction, errors, similar, mistakes, indicator_scores
            )

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
            "mistakes": mistakes,
            "indicator_scores": indicator_scores,
            "comparison_stats": stats,
            "regime_indicators": regime_indicators,
            "insights": insights,
            "outcome": "win" if (trade.get("pnl") or 0) > 0 else "loss",
        }

        # Store analysis back to memory as a trade pattern
        self._store_lessons(trade_id, analysis)

        # Record coach session for cross-run learning
        strategy_id = trade.get("strategy_id", "")
        if strategy_id and insights:
            advice = ""
            if isinstance(insights, dict):
                changes = insights.get("actionable_changes", [])
                if changes:
                    advice = "; ".join(changes[:3])
            self.memory.record_coach_session(
                strategy_id=strategy_id,
                regime=regime,
                advice=advice or "No actionable changes",
                patch_json=json.dumps(insights) if insights else "{}",
                pre_sharpe=0.0,
                post_sharpe=0.0,
            )

        # Store indicator scores for cross-run intelligence
        if indicator_scores:
            self.memory.store_indicator_scores(trade_id, indicator_scores)

        # Update indicator-regime stats
        if strategy_id:
            self.memory.update_indicator_regime_stats(strategy_id)

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

    def _classify_mistakes(
        self, trade: Dict, prediction: Optional[Dict], errors: Dict
    ) -> List[Dict]:
        """
        Classify trade mistakes using the 5-player coach model's taxonomy.

        Returns a list of identified mistakes with severity scores.
        """
        mistakes = []
        pnl = trade.get("pnl", 0) or 0
        pnl_pct = trade.get("pnl_percent", 0) or 0

        if pnl >= 0:
            return mistakes  # No mistakes to classify for winning trades

        # Wrong direction
        if prediction and errors.get("direction_correct") is False:
            mistakes.append({
                "type": MistakeType.WRONG_DIRECTION.value,
                "severity": min(abs(pnl_pct) / 5.0, 1.0),
                "detail": "Predicted wrong direction",
            })

        # Ignored regime
        regime = trade.get("market_regime", "unknown")
        if regime != "unknown":
            regime_stats = self.memory.get_indicator_regime_stats(regime)
            if regime_stats:
                avg_wr = np.mean([s.get("win_rate", 0.5) for s in regime_stats]) if regime_stats else 0.5
                if avg_wr < 0.35:
                    mistakes.append({
                        "type": MistakeType.IGNORED_REGIME.value,
                        "severity": 0.7,
                        "detail": f"Traded in unfavorable regime ({regime}, avg WR={avg_wr:.0%})",
                    })

        # Stop too tight / too wide
        max_adverse = trade.get("max_adverse_excursion", 0) or 0
        if max_adverse != 0 and pnl < 0:
            sl_distance = trade.get("stop_loss_distance", 0) or 0
            if sl_distance > 0 and abs(max_adverse) < sl_distance * 0.3:
                mistakes.append({
                    "type": MistakeType.STOP_TOO_TIGHT.value,
                    "severity": 0.5,
                    "detail": "Stop loss triggered too early; price may have recovered",
                })

        # Oversized position
        if trade.get("position_size") and trade.get("portfolio_value"):
            pos_frac = (
                trade["position_size"] * (trade.get("entry_price", 1) or 1)
            ) / trade["portfolio_value"]
            if pos_frac > 0.08:
                mistakes.append({
                    "type": MistakeType.OVERSIZED.value,
                    "severity": min(pos_frac / 0.15, 1.0),
                    "detail": f"Position was {pos_frac:.1%} of portfolio (recommended <8%)",
                })

        return mistakes

    def _score_indicators(self, trade: Dict) -> List[Dict]:
        """
        Score indicator contributions for a trade.

        Uses signal data to determine which indicators contributed
        to the trade decision and whether they were correct.
        """
        scores = []
        signals = trade.get("signals", trade.get("indicator_signals", {}))
        if not signals or not isinstance(signals, dict):
            return scores

        pnl = trade.get("pnl", 0) or 0
        was_correct = pnl > 0
        action = trade.get("action", "BUY")

        for indicator_name, signal_value in signals.items():
            if not isinstance(signal_value, (int, float)):
                continue

            # Determine if indicator agreed with the trade direction
            if action == "BUY":
                agreed_with_trade = signal_value > 0
            else:
                agreed_with_trade = signal_value < 0

            # Score: positive if indicator and outcome aligned
            if agreed_with_trade and was_correct:
                contribution = abs(signal_value) * 0.5  # Correct signal, correct trade
            elif not agreed_with_trade and not was_correct:
                contribution = abs(signal_value) * 0.3  # Warned us, we ignored
            elif agreed_with_trade and not was_correct:
                contribution = -abs(signal_value) * 0.4  # Bad signal led to loss
            else:
                contribution = abs(signal_value) * 0.1  # Disagreed but trade still lost

            scores.append({
                "indicator": indicator_name,
                "signal_value": signal_value,
                "agreed_with_trade": agreed_with_trade,
                "trade_was_correct": was_correct,
                "contribution_score": round(contribution, 4),
            })

        return scores

    def _generate_insights(
        self, trade: Dict, prediction: Optional[Dict], errors: Dict,
        similar: List[Dict], mistakes: List[Dict] = None,
        indicator_scores: List[Dict] = None,
    ) -> Dict:
        """Use LLM to extract nuanced insights with full diagnosis context."""
        similar_summary = []
        for t in similar[:5]:
            similar_summary.append({
                "asset": t.get("asset"),
                "pnl_percent": t.get("pnl_percent"),
                "regime": t.get("market_regime"),
                "strategy": t.get("strategy_id"),
            })

        # Query knowledge base for relevant theory
        strategy_type = trade.get("strategy_id", "trading")
        knowledge = self.memory.search_knowledge(
            f"{strategy_type} strategy analysis risk management",
            top_k=3,
        )
        knowledge_context = [k.get("text", "")[:200] for k in knowledge] if knowledge else []

        # Build memory context
        strategy_id = trade.get("strategy_id", "")
        memory_context = self.memory.build_coach_context(strategy_id) if strategy_id else ""

        # Format mistakes
        mistake_text = ""
        if mistakes:
            parts = [f"  {m['type']}: {m['detail']} (severity={m['severity']:.2f})" for m in mistakes]
            mistake_text = "\n".join(parts)

        # Format indicator scores
        indicator_text = ""
        if indicator_scores:
            sorted_scores = sorted(indicator_scores, key=lambda x: x["contribution_score"], reverse=True)
            parts = [
                f"  {s['indicator']}: score={s['contribution_score']:.3f} "
                f"({'agreed' if s['agreed_with_trade'] else 'disagreed'})"
                for s in sorted_scores[:5]
            ]
            indicator_text = "\n".join(parts)

        prompt = f"""Analyze this completed trade and extract learnings. Use the structured diagnosis data to provide precise, actionable feedback.

TRADE:
Asset: {trade.get('asset')}
Strategy: {trade.get('strategy_id')}
Action: {trade.get('action')}
Entry: {trade.get('entry_price')} -> Exit: {trade.get('exit_price')}
P&L: {trade.get('pnl_percent', 0):.2%}
Regime: {trade.get('market_regime')}

PREDICTION ERRORS:
{json.dumps(errors, indent=2, default=str)}

CLASSIFIED MISTAKES:
{mistake_text or "No mistakes classified (winning trade or insufficient data)"}

INDICATOR CONTRIBUTIONS:
{indicator_text or "No indicator score data"}

HISTORICAL CONTEXT (from memory):
{memory_context or "No historical context"}

SIMILAR TRADES:
{json.dumps(similar_summary, indent=2, default=str)}

RELEVANT THEORY (from knowledge base):
{json.dumps(knowledge_context, indent=2)}

Respond in JSON:
{{
    "outcome_summary": "Why the trade worked/failed",
    "primary_factors": ["factor1", "factor2"],
    "mistake_analysis": "Analysis of classified mistakes and root causes",
    "indicator_diagnosis": "Which indicators helped/hurt and recommended weight changes",
    "error_attribution": {{"model": 0.5, "execution": 0.2, "regime": 0.2, "randomness": 0.1}},
    "lessons_learned": ["lesson1", "lesson2"],
    "actionable_changes": ["change1", "change2"],
    "weight_recommendations": {{"indicator_name": 0.05}}
}}"""

        return self.llm.generate_json(prompt)

    def _store_lessons(self, trade_id: str, analysis: Dict):
        """Store lessons learned back to memory."""
        trade = analysis.get("trade", {})
        insights = analysis.get("insights")
        lessons = ""
        if insights and isinstance(insights, dict):
            lessons = ". ".join(insights.get("lessons_learned", []))

        pnl_pct = trade.get('pnl_percent') or 0
        description = (
            f"Post-mortem: {trade.get('action', '')} {trade.get('asset', '')} "
            f"using {trade.get('strategy_id', '')}. "
            f"Outcome: {analysis.get('outcome', 'unknown')}. "
            f"P&L: {pnl_pct:.2%}. "
            f"Regime: {trade.get('market_regime', 'unknown')}. "
            f"Lessons: {lessons}"
        )

        try:
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
        except Exception:
            pass  # Vector store is optional

    # ─────────────────────────────────────────────────────────────────
    # WEEKLY REVIEW
    # ─────────────────────────────────────────────────────────────────

    def weekly_performance_review(self) -> Dict:
        """
        Aggregate learnings from past week's trades.

        Enhanced with cross-run intelligence and indicator-regime analysis.
        """
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

        # Get cross-run P&L trend
        pnl_trend = self.memory.get_cross_run_pnl_trend()

        # Get indicator-regime intelligence
        indicator_stats = self.memory.get_indicator_regime_stats()
        top_indicators = []
        if indicator_stats:
            for s in indicator_stats[:5]:
                if s.get("total_trades", 0) >= 5:
                    top_indicators.append({
                        "indicator": s["indicator"],
                        "regime": s["regime"],
                        "win_rate": s.get("win_rate", 0),
                        "avg_pnl": s.get("avg_pnl", 0),
                    })

        # Get coach advice effectiveness
        coach_reviews = {}
        for sid in by_strategy:
            effectiveness = self.memory.get_coach_advice_effectiveness(sid)
            if effectiveness:
                helped = sum(1 for e in effectiveness if e.get("helped"))
                coach_reviews[sid] = {
                    "total_patches": len(effectiveness),
                    "helped": helped,
                    "hurt": len(effectiveness) - helped,
                }

        # LLM synthesis
        synthesis = None
        if self.llm and self.llm.is_available():
            trend_str = ""
            if pnl_trend:
                trend_str = " -> ".join(f"${r['team_pnl']:,.0f}" for r in pnl_trend[-5:])

            prompt = f"""Weekly trading performance review with cross-run intelligence.

STRATEGY PERFORMANCE:
{json.dumps(strategy_reviews, indent=2, default=str)}

OVERALL:
{json.dumps(overall, indent=2, default=str)}

CROSS-RUN P&L TREND:
{trend_str or "First run"}

TOP INDICATOR-REGIME COMBOS:
{json.dumps(top_indicators, indent=2, default=str) if top_indicators else "No data"}

COACH ADVICE EFFECTIVENESS:
{json.dumps(coach_reviews, indent=2, default=str) if coach_reviews else "No coach data"}

Provide strategic recommendations in JSON:
{{
    "best_strategy": "...",
    "worst_strategy": "...",
    "key_insights": ["insight1", "insight2"],
    "indicator_recommendations": {{"keep": [], "remove": [], "increase_weight": []}},
    "regime_observations": "...",
    "coach_assessment": "Are the learning patches helping or hurting?",
    "focus_next_week": ["action1", "action2"]
}}"""
            synthesis = self.llm.generate_json(prompt)

        return {
            "period": f"Last 7 days (from {cutoff[:10]})",
            "overall": overall,
            "strategy_reviews": strategy_reviews,
            "pnl_trend": pnl_trend,
            "top_indicators": top_indicators,
            "coach_effectiveness": coach_reviews,
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
