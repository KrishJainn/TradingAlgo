"""
AQTIS Strategy Generator Agent.

Proposes new quantitative strategies and parameter variations
using LLM reasoning combined with historical performance data.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import BaseAgent

logger = logging.getLogger(__name__)


class StrategyGeneratorAgent(BaseAgent):
    """
    Generates and improves trading strategies using LLM + data analysis.

    Capabilities:
    - Analyze existing strategy performance to find improvements
    - Generate parameter variations for A/B testing
    - Propose entirely new strategies from research insights
    - Combine successful elements from multiple strategies
    """

    def __init__(self, memory, llm=None):
        super().__init__(name="strategy_generator", memory=memory, llm=llm)

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute strategy generation action."""
        action = context.get("action", "analyze_signal")

        if action == "analyze_signal":
            return self.analyze_signal(context.get("signal", {}))
        elif action == "improve":
            return self.propose_strategy_improvement(context["strategy_id"])
        elif action == "generate_variants":
            return {
                "variants": self.generate_parameter_variants(
                    context["strategy_id"],
                    context.get("n_variants", 5),
                )
            }
        elif action == "propose_new":
            return self.propose_new_strategy(context.get("constraints", {}))
        else:
            return {"error": f"Unknown action: {action}"}

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────

    def analyze_signal(self, market_signal: Dict) -> Dict:
        """
        Analyze a market signal to determine if a trade should be taken.

        Enhanced with 5-player coach model intelligence:
        - Regime-specific indicator effectiveness from historical data
        - Memory context from cross-run learning
        - Knowledge base lookup for strategy-specific rules
        """
        # Get current market regime
        regime = self.memory.get_market_regime()
        regime_name = regime.get("vol_regime", "unknown") if regime else "unknown"

        # Get active strategies
        strategies = self.memory.get_active_strategies()

        if not strategies:
            return {"should_trade": False, "reason": "No active strategies"}

        # Get regime-specific indicator intelligence from memory
        top_indicators = self.memory.get_top_indicators_for_regime(regime_name, limit=5)

        # Find best strategy for current regime (enhanced scoring)
        best_strategy = None
        best_score = -float("inf")

        for strategy in strategies:
            regime_perf = (strategy.get("performance_by_regime") or {})
            regime_score = regime_perf.get(regime_name, 0)
            overall_score = strategy.get("sharpe_ratio", 0) or 0

            # Boost score if strategy uses indicators that perform well in this regime
            indicator_boost = 0.0
            if top_indicators:
                strategy_indicators = set(
                    (strategy.get("parameters") or {}).get("indicators", [])
                )
                for ind in top_indicators:
                    if ind.get("indicator") in strategy_indicators:
                        indicator_boost += ind.get("contribution_score", 0) * 0.1

            score = regime_score * 0.5 + overall_score * 0.3 + indicator_boost * 0.2
            if score > best_score:
                best_score = score
                best_strategy = strategy

        if not best_strategy:
            return {"should_trade": False, "reason": "No suitable strategy for current regime"}

        # Use LLM for deeper analysis if available
        if self.llm and self.llm.is_available():
            llm_analysis = self._llm_analyze_signal(
                market_signal, best_strategy, regime_name, top_indicators
            )
            return {
                "should_trade": llm_analysis.get("should_trade", True),
                "strategy": best_strategy,
                "regime": regime_name,
                "top_indicators": top_indicators,
                "analysis": llm_analysis,
                "reason": llm_analysis.get("reasoning", "LLM approved"),
            }

        return {
            "should_trade": best_score > 0,
            "strategy": best_strategy,
            "regime": regime_name,
            "top_indicators": top_indicators,
            "score": best_score,
            "reason": "Rule-based analysis",
        }

    def _llm_analyze_signal(
        self, signal: Dict, strategy: Dict, regime: str, top_indicators: List = None
    ) -> Dict:
        """Use LLM to analyze trading signal with full memory context."""
        # Get similar historical trades for context
        similar = self.memory.get_similar_trades(signal, top_k=10)
        similar_summary = []
        for t in similar[:5]:
            similar_summary.append({
                "asset": t.get("asset"),
                "pnl_percent": t.get("pnl_percent"),
                "regime": t.get("market_regime"),
            })

        # Build memory context from cross-run learning
        strategy_id = strategy.get("strategy_id", "")
        memory_context = self.memory.build_pre_trade_context(signal)

        # Get knowledge base context for this regime + strategy type
        knowledge = self.memory.search_knowledge(
            f"{regime} regime {strategy.get('strategy_type', 'trading')} strategy rules",
            top_k=3,
        )
        knowledge_snippets = [k.get("text", "")[:200] for k in knowledge] if knowledge else []

        # Format indicator intelligence
        indicator_info = ""
        if top_indicators:
            ind_parts = [
                f"  {i.get('indicator')}: WR={i.get('win_rate', 0):.0%}, "
                f"score={i.get('contribution_score', 0):.2f}"
                for i in top_indicators
            ]
            indicator_info = "\n".join(ind_parts)

        # Determine if this is a bootstrap scenario (no historical data)
        is_bootstrap = (
            not similar_summary
            and not memory_context
            and not knowledge_snippets
            and not indicator_info
        )

        bootstrap_note = ""
        if is_bootstrap:
            bootstrap_note = """
IMPORTANT: This is an EARLY BOOTSTRAP run with no historical data yet.
You should APPROVE trades that have reasonable signal scores (>0.15) to
build up the historical trade database. Be moderately permissive — the
system needs trade data to start learning. Default to should_trade=true
unless the signal is clearly weak or contradictory.
"""

        prompt = f"""You are a quantitative strategy analyst. Analyze this trading signal and decide whether to trade.
{bootstrap_note}
CURRENT SIGNAL:
{json.dumps(signal, indent=2, default=str)}

STRATEGY: {strategy.get('strategy_name', strategy.get('strategy_id'))}
Type: {strategy.get('strategy_type')}
Win Rate: {strategy.get('win_rate')}
Sharpe: {strategy.get('sharpe_ratio')}

MARKET REGIME: {regime}

BEST INDICATORS FOR THIS REGIME:
{indicator_info or "No indicator-regime data available yet (bootstrap)"}

MEMORY CONTEXT (from historical runs):
{memory_context or "No historical context yet (bootstrap run)"}

SIMILAR HISTORICAL TRADES:
{json.dumps(similar_summary, indent=2, default=str) if similar_summary else "No similar trades yet (bootstrap run)"}

KNOWLEDGE BASE:
{json.dumps(knowledge_snippets, indent=2) if knowledge_snippets else "No relevant knowledge yet"}

Respond in JSON format:
{{
    "should_trade": true/false,
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation referencing regime, indicators, and history",
    "suggested_adjustments": "Any parameter tweaks",
    "regime_alignment": "How well strategy aligns with current regime"
}}"""

        return self.llm.generate_json(prompt)

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY IMPROVEMENT
    # ─────────────────────────────────────────────────────────────────

    def propose_strategy_improvement(self, strategy_id: str) -> Dict:
        """
        Analyze strategy and propose improvements.

        Enhanced with 5-player coach model intelligence:
        - Full optimization context from memory (cross-run learning)
        - Indicator-regime effectiveness data
        - Coach advice history and effectiveness
        - Best historical strategy snapshots
        """
        strategy = self.memory.get_strategy(strategy_id)
        if not strategy:
            return {"error": f"Strategy {strategy_id} not found"}

        performance = self.memory.get_strategy_performance(strategy_id)

        # Get recent failed trades
        failed_trades = self.memory.get_trades(
            strategy_id=strategy_id, outcome="loss", limit=20
        )

        # Search for relevant research
        research = self.memory.search_research(
            f"improving {strategy.get('strategy_type', 'trading')} strategies",
            top_k=3,
        )

        # Query knowledge base for strategy-type-specific concepts
        knowledge = self.memory.search_knowledge(
            f"{strategy.get('strategy_type', 'trading')} strategy implementation best practices",
            top_k=3,
        )

        # Get full optimization context from cross-run memory
        optimization_context = self.memory.build_full_optimization_context(strategy_id)

        # Get best historical strategy snapshot
        best_snapshot = self.memory.get_best_strategy_snapshot(strategy_id)

        # Get indicator-regime effectiveness
        indicator_stats = self.memory.get_indicator_regime_stats()
        top_ind_summary = []
        if indicator_stats:
            for s in indicator_stats[:5]:
                if s.get("total_trades", 0) >= 10:
                    top_ind_summary.append(
                        f"{s['indicator']} in {s['regime']}: "
                        f"WR={s.get('win_rate', 0):.0%}, avg P&L=${s.get('avg_pnl', 0):.0f}"
                    )

        # Get coach advice effectiveness
        coach_effectiveness = self.memory.get_coach_advice_effectiveness(strategy_id)

        if not self.llm or not self.llm.is_available():
            return {
                "strategy_id": strategy_id,
                "performance": performance,
                "failed_trades_count": len(failed_trades),
                "research_found": len(research),
                "knowledge_found": len(knowledge),
                "best_snapshot": best_snapshot,
                "indicator_intelligence": top_ind_summary,
                "improvements": "LLM not available for detailed analysis",
            }

        knowledge_context = [k.get("text", "")[:200] for k in knowledge] if knowledge else []

        # Use LLM to generate improvement proposal with full context
        prompt = f"""You are a quantitative strategy designer. Analyze this strategy and propose improvements.

STRATEGY: {strategy.get('strategy_name')}
Type: {strategy.get('strategy_type')}
Parameters: {json.dumps(strategy.get('parameters', {}), indent=2, default=str)}

PERFORMANCE:
Total Trades: {performance.get('total_trades', 0)}
Win Rate: {performance.get('win_rate', 0):.2%}
Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}
Max Drawdown: {performance.get('max_drawdown', 0):.2%}

CROSS-RUN MEMORY (learned from all previous simulation runs):
{optimization_context}

BEST HISTORICAL CONFIG:
{json.dumps(best_snapshot, indent=2, default=str) if best_snapshot else "No historical snapshots"}

INDICATOR-REGIME INTELLIGENCE (which indicators work in which regimes):
{chr(10).join(top_ind_summary) if top_ind_summary else "No indicator data yet"}

COACH ADVICE HISTORY:
{json.dumps(coach_effectiveness[-3:], indent=2, default=str) if coach_effectiveness else "No coach history"}

RECENT FAILED TRADES ({len(failed_trades)}):
Common regimes: {self._count_regimes(failed_trades)}

RELEVANT RESEARCH:
{json.dumps([r.get('metadata', {}).get('title', r.get('text', '')[:100]) for r in research], indent=2)}

KNOWLEDGE BASE (best practices for this strategy type):
{json.dumps(knowledge_context, indent=2)}

Propose specific improvements in JSON format:
{{
    "analysis": "What's working and what isn't, considering cross-run history",
    "parameter_changes": {{"param_name": "new_value"}},
    "indicator_changes": {{"add": [], "remove": [], "reweight": {{}}}},
    "regime_specific_rules": {{"regime_name": "rule"}},
    "new_rules": ["Rule 1", "Rule 2"],
    "expected_improvement": "Brief prediction with rationale from historical data"
}}"""

        return self.llm.generate_json(prompt)

    # ─────────────────────────────────────────────────────────────────
    # PARAMETER VARIANTS
    # ─────────────────────────────────────────────────────────────────

    def generate_parameter_variants(self, strategy_id: str, n_variants: int = 5) -> List[Dict]:
        """Create parameter variations for shadow testing."""
        strategy = self.memory.get_strategy(strategy_id)
        if not strategy:
            return []

        params = strategy.get("parameters", {})
        if not params:
            return []

        if self.llm and self.llm.is_available():
            prompt = f"""Generate {n_variants} parameter variations for this trading strategy.

Strategy: {strategy.get('strategy_name')}
Current Parameters: {json.dumps(params, indent=2, default=str)}
Recent Performance - Win Rate: {strategy.get('win_rate')}, Sharpe: {strategy.get('sharpe_ratio')}

Generate {n_variants} variations as a JSON array. Each variation should adjust 1-3 parameters.
Focus on parameters most likely to improve performance.

[{{"variant_name": "...", "parameters": {{...}}, "rationale": "..."}}]"""

            result = self.llm.generate_json(prompt)
            if isinstance(result, list):
                return result
            return result.get("variants", [result]) if isinstance(result, dict) else []

        # Rule-based variants
        import numpy as np
        variants = []
        for i in range(n_variants):
            variant = {"variant_name": f"variant_{i}", "parameters": {}}
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    factor = 1.0 + np.random.uniform(-0.2, 0.2)
                    new_val = value * factor
                    variant["parameters"][key] = type(value)(new_val)
                else:
                    variant["parameters"][key] = value
            variants.append(variant)

        return variants

    # ─────────────────────────────────────────────────────────────────
    # NEW STRATEGY PROPOSAL
    # ─────────────────────────────────────────────────────────────────

    def propose_new_strategy(self, constraints: Dict = None) -> Dict:
        """Propose a new trading strategy."""
        if not self.llm or not self.llm.is_available():
            return {"error": "LLM required for strategy proposal"}

        # Get context
        existing = self.memory.get_active_strategies()
        research = self.memory.search_research("quantitative trading strategies", top_k=5)
        regime = self.memory.get_market_regime()

        # Query knowledge base for strategy concepts
        knowledge = self.memory.search_knowledge("trading strategies momentum mean reversion", top_k=5)
        knowledge_context = [k.get("text", "")[:200] for k in knowledge] if knowledge else []

        prompt = f"""You are a quantitative strategy designer. Propose a new trading strategy.

EXISTING STRATEGIES:
{json.dumps([s.get('strategy_name') for s in existing], indent=2)}

CURRENT MARKET REGIME:
{json.dumps(regime, indent=2, default=str) if regime else "Unknown"}

RESEARCH INSIGHTS:
{json.dumps([r.get('text', '')[:200] for r in research], indent=2)}

KNOWLEDGE BASE CONTEXT:
{json.dumps(knowledge_context, indent=2)}

CONSTRAINTS:
{json.dumps(constraints or {}, indent=2)}

Propose a new strategy in JSON format:
{{
    "strategy_name": "...",
    "strategy_type": "mean_reversion/momentum/pairs_trading/etc",
    "description": "Detailed description",
    "parameters": {{...}},
    "entry_rules": ["Rule 1", "Rule 2"],
    "exit_rules": ["Rule 1", "Rule 2"],
    "expected_performance": {{
        "target_win_rate": 0.55,
        "target_sharpe": 1.5,
        "suitable_regimes": ["trending_up", "mean_reverting"]
    }}
}}"""

        return self.llm.generate_json(prompt)

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _count_regimes(self, trades: List[Dict]) -> Dict[str, int]:
        """Count market regimes in trade list."""
        counts: Dict[str, int] = {}
        for t in trades:
            regime = t.get("market_regime", "unknown")
            counts[regime] = counts.get(regime, 0) + 1
        return counts
