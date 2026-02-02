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
        """
        # Get current market regime
        regime = self.memory.get_market_regime()
        regime_name = regime.get("vol_regime", "unknown") if regime else "unknown"

        # Get active strategies
        strategies = self.memory.get_active_strategies()

        if not strategies:
            return {"should_trade": False, "reason": "No active strategies"}

        # Find best strategy for current regime
        best_strategy = None
        best_score = -float("inf")

        for strategy in strategies:
            regime_perf = (strategy.get("performance_by_regime") or {})
            regime_score = regime_perf.get(regime_name, 0)
            overall_score = strategy.get("sharpe_ratio", 0) or 0

            score = regime_score * 0.6 + overall_score * 0.4
            if score > best_score:
                best_score = score
                best_strategy = strategy

        if not best_strategy:
            return {"should_trade": False, "reason": "No suitable strategy for current regime"}

        # Use LLM for deeper analysis if available
        if self.llm and self.llm.is_available():
            llm_analysis = self._llm_analyze_signal(market_signal, best_strategy, regime_name)
            return {
                "should_trade": llm_analysis.get("should_trade", True),
                "strategy": best_strategy,
                "regime": regime_name,
                "analysis": llm_analysis,
                "reason": llm_analysis.get("reasoning", "LLM approved"),
            }

        return {
            "should_trade": best_score > 0,
            "strategy": best_strategy,
            "regime": regime_name,
            "score": best_score,
            "reason": "Rule-based analysis",
        }

    def _llm_analyze_signal(self, signal: Dict, strategy: Dict, regime: str) -> Dict:
        """Use LLM to analyze trading signal."""
        # Get similar historical trades for context
        similar = self.memory.get_similar_trades(signal, top_k=10)
        similar_summary = []
        for t in similar[:5]:
            similar_summary.append({
                "asset": t.get("asset"),
                "pnl_percent": t.get("pnl_percent"),
                "regime": t.get("market_regime"),
            })

        prompt = f"""You are a quantitative strategy analyst. Analyze this trading signal and decide whether to trade.

CURRENT SIGNAL:
{json.dumps(signal, indent=2, default=str)}

STRATEGY: {strategy.get('strategy_name', strategy.get('strategy_id'))}
Type: {strategy.get('strategy_type')}
Win Rate: {strategy.get('win_rate')}
Sharpe: {strategy.get('sharpe_ratio')}

MARKET REGIME: {regime}

SIMILAR HISTORICAL TRADES:
{json.dumps(similar_summary, indent=2, default=str)}

Respond in JSON format:
{{
    "should_trade": true/false,
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation",
    "suggested_adjustments": "Any parameter tweaks"
}}"""

        return self.llm.generate_json(prompt)

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY IMPROVEMENT
    # ─────────────────────────────────────────────────────────────────

    def propose_strategy_improvement(self, strategy_id: str) -> Dict:
        """Analyze strategy and propose improvements."""
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

        if not self.llm or not self.llm.is_available():
            return {
                "strategy_id": strategy_id,
                "performance": performance,
                "failed_trades_count": len(failed_trades),
                "research_found": len(research),
                "improvements": "LLM not available for detailed analysis",
            }

        # Use LLM to generate improvement proposal
        prompt = f"""You are a quantitative strategy designer. Analyze this strategy and propose improvements.

STRATEGY: {strategy.get('strategy_name')}
Type: {strategy.get('strategy_type')}
Parameters: {json.dumps(strategy.get('parameters', {}), indent=2, default=str)}

PERFORMANCE:
Total Trades: {performance.get('total_trades', 0)}
Win Rate: {performance.get('win_rate', 0):.2%}
Sharpe Ratio: {performance.get('sharpe_ratio', 0):.2f}
Max Drawdown: {performance.get('max_drawdown', 0):.2%}

RECENT FAILED TRADES ({len(failed_trades)}):
Common regimes: {self._count_regimes(failed_trades)}

RELEVANT RESEARCH:
{json.dumps([r.get('metadata', {}).get('title', r.get('text', '')[:100]) for r in research], indent=2)}

Propose specific improvements in JSON format:
{{
    "analysis": "What's working and what isn't",
    "parameter_changes": {{"param_name": "new_value"}},
    "new_rules": ["Rule 1", "Rule 2"],
    "expected_improvement": "Brief prediction"
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

        prompt = f"""You are a quantitative strategy designer. Propose a new trading strategy.

EXISTING STRATEGIES:
{json.dumps([s.get('strategy_name') for s in existing], indent=2)}

CURRENT MARKET REGIME:
{json.dumps(regime, indent=2, default=str) if regime else "Unknown"}

RESEARCH INSIGHTS:
{json.dumps([r.get('text', '')[:200] for r in research], indent=2)}

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
