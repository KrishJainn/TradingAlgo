"""
LLM Analyst - Memory-Informed Post-Mortem and Config Selection.

The LLM's role is REDEFINED: instead of generating strategies from scratch,
it reads from dual memory (knowledge + experience) and:
  1. Analyzes what went wrong/right (post-mortem)
  2. Selects from PROVEN configs instead of inventing new ones
  3. Suggests small parameter adjustments grounded in historical data
"""

import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMAnalyst:
    """
    LLM reads from dual memory for analysis and selection.
    Never invents strategies from scratch.
    """

    def __init__(self, llm_provider, dual_memory):
        """
        Args:
            llm_provider: GeminiProvider or any LLMProvider
            dual_memory: DualMemoryContext instance
        """
        self.llm = llm_provider
        self.memory = dual_memory

    def post_mortem(self, run_results: Dict, regime: str) -> str:
        """
        Analyze a completed run using dual memory context.

        Returns a structured analysis string.
        """
        if not self.llm:
            return ""

        players = run_results.get("players", {})
        team_pnl = run_results.get("team_total_pnl", 0)

        # Build player summary
        player_lines = []
        for pid, pdata in players.items():
            player_lines.append(
                f"{pdata.get('label', pid)}: PnL {pdata.get('pnl', 0):+.0f}, "
                f"WR {pdata.get('win_rate', 0):.0%}, "
                f"Sharpe {pdata.get('sharpe', 0):.2f}, "
                f"{pdata.get('trades', 0)} trades"
            )

        # Get memory context
        context = self.memory.get_context_for_coach(
            situation=f"Post-mortem analysis for run in {regime} regime",
            regime=regime,
            max_tokens=1500,
        )

        prompt = f"""You are analyzing a completed trading system run. Be concise.

RESULTS:
- Regime: {regime}
- Team P&L: {team_pnl:+.0f}
- Players:
{chr(10).join('  ' + l for l in player_lines)}

{context}

ANALYZE (respond in 3-5 bullet points):
1. Which player was best/worst matched to the {regime} regime and why?
2. What entry/exit patterns led to the biggest wins and losses?
3. What specific parameter change would you recommend for each underperforming player?

Keep each bullet to 1-2 sentences max."""

        try:
            response = self.llm._call(prompt, temperature=0.3)
            return response.strip() if response else ""
        except Exception as e:
            logger.error(f"[LLMAnalyst] Post-mortem failed: {e}")
            return ""

    def select_config(
        self,
        player_id: str,
        player_label: str,
        regime: str,
        current_config: Dict,
    ) -> Optional[Dict]:
        """
        Select from PROVEN configs in memory instead of inventing from scratch.

        The LLM chooses between:
          (a) One of the top historical configs for this regime
          (b) A blended/modified version of a proven config
          (c) Keeping the current config with minor tweaks

        Returns:
            Modified config dict, or None if no historical data.
        """
        if not self.llm:
            return None

        proven = self.memory.get_proven_configs(player_id, regime, top_k=3)
        if not proven:
            return None

        # Format proven configs for the prompt
        config_descriptions = []
        for i, p in enumerate(proven):
            cfg = p.get("config", {})
            weights = cfg.get("weights", {})
            top_3 = sorted(weights.items(), key=lambda x: -x[1])[:3]
            top_str = ", ".join(f"{k}={v:.2f}" for k, v in top_3)
            config_descriptions.append(
                f"Config {i}: Sharpe {p['team_sharpe']:.2f}, PnL {p['team_pnl']:+.0f}, "
                f"entry={cfg.get('entry_threshold', 0.25):.2f}, "
                f"top indicators: {top_str}, "
                f"{len(weights)} total indicators"
            )

        # Current config summary
        cur_weights = current_config.get("weights", {})
        cur_top = sorted(cur_weights.items(), key=lambda x: -x[1])[:3]
        cur_str = ", ".join(f"{k}={v:.2f}" for k, v in cur_top)

        prompt = f"""You are selecting a trading configuration for the {player_label} player in {regime} regime.

PROVEN CONFIGS (from past runs in this regime):
{chr(10).join(config_descriptions)}

CURRENT CONFIG:
entry={current_config.get('entry_threshold', 0.25):.2f}, top indicators: {cur_str}

Choose the best approach. Return ONLY valid JSON:
{{"action": "select", "config_index": 0}}
OR
{{"action": "blend", "config_indices": [0, 1], "blend_weight": 0.7}}
OR
{{"action": "keep", "adjustments": {{"entry_threshold": 0.26}}}}"""

        try:
            response = self.llm._call(prompt, temperature=0.2)
            if not response:
                return None

            decision = _parse_json(response)
            if not decision:
                return None

            action = decision.get("action", "keep")

            if action == "select":
                idx = decision.get("config_index", 0)
                if 0 <= idx < len(proven):
                    return proven[idx]["config"]

            elif action == "blend":
                indices = decision.get("config_indices", [0, 1])
                blend_w = decision.get("blend_weight", 0.5)
                configs_to_blend = [
                    proven[i]["config"]
                    for i in indices
                    if 0 <= i < len(proven)
                ]
                if len(configs_to_blend) >= 2:
                    return _blend_configs(configs_to_blend[0], configs_to_blend[1], blend_w)

            elif action == "keep":
                adjustments = decision.get("adjustments", {})
                modified = dict(current_config)
                for k, v in adjustments.items():
                    if k in modified:
                        modified[k] = v
                return modified

        except Exception as e:
            logger.error(f"[LLMAnalyst] Config selection failed: {e}")

        return None

    def generate_coaching_notes(
        self,
        player_id: str,
        player_label: str,
        regime: str,
        pnl: float,
        win_rate: float,
    ) -> str:
        """Generate concise coaching notes using memory context."""
        if not self.llm:
            return ""

        context = self.memory.get_player_context(
            player_id=player_id,
            player_label=player_label,
            regime=regime,
            max_tokens=1000,
        )

        prompt = f"""Brief coaching note for {player_label} player.
Regime: {regime}, PnL: {pnl:+.0f}, WR: {win_rate:.0%}

{context}

Write 2-3 sentences of actionable coaching advice based on the history above."""

        try:
            response = self.llm._call(prompt, temperature=0.3)
            return response.strip() if response else ""
        except Exception:
            return ""


def _parse_json(text: str) -> Optional[Dict]:
    """Extract JSON from LLM response."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None
    return None


def _blend_configs(config1: Dict, config2: Dict, weight: float = 0.5) -> Dict:
    """
    Blend two configs by weighted averaging their weights.

    Args:
        config1, config2: Config dicts
        weight: Weight for config1 (1-weight for config2)
    """
    from copy import deepcopy

    blended = deepcopy(config1)
    w1 = config1.get("weights", {})
    w2 = config2.get("weights", {})

    all_inds = set(w1.keys()) | set(w2.keys())
    blended_weights = {}
    for ind in all_inds:
        v1 = w1.get(ind, 0)
        v2 = w2.get(ind, 0)
        blended_weights[ind] = v1 * weight + v2 * (1 - weight)
        if blended_weights[ind] < 0.05:
            continue  # Skip very low weights
        blended_weights[ind] = max(0.05, min(1.0, blended_weights[ind]))

    # Remove near-zero weights
    blended["weights"] = {k: v for k, v in blended_weights.items() if v >= 0.05}

    # Blend thresholds
    blended["entry_threshold"] = (
        config1.get("entry_threshold", 0.25) * weight
        + config2.get("entry_threshold", 0.25) * (1 - weight)
    )
    blended["exit_threshold"] = (
        config1.get("exit_threshold", -0.10) * weight
        + config2.get("exit_threshold", -0.10) * (1 - weight)
    )

    return blended
