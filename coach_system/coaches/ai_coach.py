"""
AI-Powered Coach for 5-Player Trading System.

GEMINI IS THE BRAIN - It makes ALL optimization decisions:
1. Which indicators to add/remove for each player
2. What weights to assign to each indicator
3. Entry/exit thresholds
4. Learning from trade history to become a better trader

Each player gets INDEPENDENT optimization based on their style and performance.
"""

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

# Dynamically load indicators from the actual IndicatorUniverse
def _load_available_indicators():
    """Load available indicators from the actual IndicatorUniverse."""
    try:
        from trading_evolution.indicators.universe import IndicatorUniverse
        universe = IndicatorUniverse()
        universe.load_all()
        return universe.get_all()
    except ImportError:
        # Fallback to hardcoded list if universe not available
        return [
            # Momentum
            "RSI_7", "RSI_14", "RSI_21", "STOCH_5_3", "STOCH_14_3", "STOCH_21_5",
            "MACD_12_26_9", "MACD_8_17_9", "MACD_5_35_5", "CCI_14", "CCI_20",
            "CMO_14", "CMO_20", "MOM_10", "MOM_20", "ROC_10", "ROC_20",
            "WILLR_14", "WILLR_28", "TSI_13_25", "UO_7_14_28", "AO_5_34",
            "KST", "COPPOCK",
            # Trend
            "ADX_14", "ADX_20", "AROON_14", "AROON_25",
            "SUPERTREND_7_3", "SUPERTREND_10_2", "SUPERTREND_20_3", "PSAR",
            "VORTEX_14", "LINREG_SLOPE_14", "LINREG_SLOPE_25",
            # Volatility
            "ATR_14", "ATR_20", "NATR_14", "NATR_20",
            "BBANDS_20_2", "KC_20_2",
            "DONCHIAN_20", "DONCHIAN_50", "TRUERANGE", "MASS_INDEX",
            # Volume
            "OBV", "AD", "ADOSC_3_10", "CMF_20", "CMF_21", "MFI_14", "MFI_20",
            "EFI_13", "EFI_20", "NVI", "PVI",
            # Overlap/Moving Averages
            "EMA_10", "EMA_20", "EMA_50", "EMA_100", "EMA_200",
            "SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200",
            "WMA_10", "WMA_20", "DEMA_10", "DEMA_20", "TEMA_10", "TEMA_20",
            "HMA_9", "HMA_16", "VWMA_10", "VWMA_20",
            "KAMA_10", "KAMA_20", "T3_5", "T3_10",
            # Other
            "ZSCORE_20", "ZSCORE_50",
        ]

# Load indicators at module level
AVAILABLE_INDICATORS = _load_available_indicators()

# Player style profiles
PLAYER_STYLES = {
    "Aggressive": {
        "description": "High risk tolerance, short holding periods, momentum-focused",
        "preferred_categories": ["momentum", "volatility"],
        "risk_tolerance": "high",
    },
    "Conservative": {
        "description": "Low risk tolerance, longer holds, trend-following",
        "preferred_categories": ["trend", "overlap"],
        "risk_tolerance": "low",
    },
    "Balanced": {
        "description": "Medium risk, diversified indicators",
        "preferred_categories": ["momentum", "trend", "volume"],
        "risk_tolerance": "medium",
    },
    "VolBreakout": {
        "description": "Volatility breakout specialist, catches big moves",
        "preferred_categories": ["volatility", "trend"],
        "risk_tolerance": "high",
    },
    "Momentum": {
        "description": "Pure momentum player, rides trends",
        "preferred_categories": ["momentum", "overlap"],
        "risk_tolerance": "medium-high",
    },
}


@dataclass
class PlayerAnalysis:
    """Analysis results for a single player."""
    player_id: str
    player_label: str

    # Performance metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0

    # Gemini's decisions
    new_weights: Dict[str, float] = field(default_factory=dict)
    indicators_to_add: Dict[str, float] = field(default_factory=dict)
    indicators_to_remove: List[str] = field(default_factory=list)
    new_entry_threshold: float = 0.25
    new_exit_threshold: float = -0.10
    new_min_hold_bars: int = 4

    # For display
    weight_changes: Dict[str, float] = field(default_factory=dict)
    best_indicators: List[str] = field(default_factory=list)
    worst_indicators: List[str] = field(default_factory=list)

    # LLM reasoning
    llm_analysis: str = ""
    llm_recommendations: List[str] = field(default_factory=list)


class AICoach:
    """
    Gemini-Powered AI Coach - The Brain of the Trading System.

    Gemini makes ALL decisions:
    - Analyzes trade history to understand what's working
    - Decides which indicators to use and their weights
    - Sets entry/exit thresholds
    - Learns and improves over time
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_provider=None,
    ):
        self.use_llm = use_llm
        self.llm = llm_provider

        # Track learning history across sessions
        self.learning_history: Dict[str, List[Dict]] = {}

    def analyze_player(
        self,
        player_id: str,
        player_label: str,
        trades: List[Dict],
        current_weights: Dict[str, float],
        current_config: Dict,
        market_regime: str = "unknown",
    ) -> PlayerAnalysis:
        """
        Have Gemini analyze a player and decide on optimizations.

        Gemini receives:
        - Player's style and current config
        - Recent trade history with outcomes
        - Market regime
        - Available indicators

        Gemini outputs:
        - New indicator weights (JSON)
        - Indicators to add/remove
        - New thresholds
        - Reasoning
        """
        analysis = PlayerAnalysis(
            player_id=player_id,
            player_label=player_label,
        )

        # Calculate basic metrics
        if trades:
            analysis.total_trades = len(trades)
            wins = [t for t in trades if t.get("pnl", 0) > 0]
            losses = [t for t in trades if t.get("pnl", 0) <= 0]
            analysis.wins = len(wins)
            analysis.losses = len(losses)
            analysis.total_pnl = sum(t.get("pnl", 0) for t in trades)
            analysis.win_rate = len(wins) / len(trades) if trades else 0

            win_pnls = [t["pnl"] for t in wins]
            loss_pnls = [t["pnl"] for t in losses]
            analysis.avg_win = np.mean(win_pnls) if win_pnls else 0
            analysis.avg_loss = np.mean(loss_pnls) if loss_pnls else 0

            total_wins = sum(win_pnls) if win_pnls else 0
            total_losses = abs(sum(loss_pnls)) if loss_pnls else 1
            analysis.profit_factor = total_wins / total_losses if total_losses > 0 else 0

        # If no LLM or not enough trades, use fallback
        if not self.use_llm or not self.llm or len(trades) < 3:
            return self._fallback_analysis(analysis, current_weights, current_config)

        # === GEMINI MAKES THE DECISIONS ===
        try:
            gemini_decision = self._get_gemini_decision(
                analysis=analysis,
                trades=trades,
                current_weights=current_weights,
                current_config=current_config,
                market_regime=market_regime,
            )

            if gemini_decision:
                # Apply Gemini's decisions
                analysis.new_weights = gemini_decision.get("weights", current_weights)
                analysis.indicators_to_add = gemini_decision.get("add_indicators", {})
                analysis.indicators_to_remove = gemini_decision.get("remove_indicators", [])
                analysis.new_entry_threshold = gemini_decision.get("entry_threshold", current_config.get("entry_threshold", 0.25))
                analysis.new_exit_threshold = gemini_decision.get("exit_threshold", current_config.get("exit_threshold", -0.10))
                analysis.new_min_hold_bars = gemini_decision.get("min_hold_bars", current_config.get("min_hold_bars", 4))
                analysis.llm_analysis = gemini_decision.get("reasoning", "")
                analysis.llm_recommendations = gemini_decision.get("recommendations", [])
                analysis.best_indicators = gemini_decision.get("best_indicators", [])
                analysis.worst_indicators = gemini_decision.get("worst_indicators", [])

                # Calculate weight changes for display
                for ind, new_w in analysis.new_weights.items():
                    old_w = current_weights.get(ind, 0)
                    if abs(new_w - old_w) > 0.01:
                        analysis.weight_changes[ind] = new_w

                print(f"[Gemini] {player_id}: Made {len(analysis.weight_changes)} weight changes, "
                      f"+{len(analysis.indicators_to_add)} -{len(analysis.indicators_to_remove)} indicators")
            else:
                return self._fallback_analysis(analysis, current_weights, current_config)

        except Exception as e:
            logger.error(f"Gemini decision failed for {player_id}: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_analysis(analysis, current_weights, current_config)

        return analysis

    def _get_gemini_decision(
        self,
        analysis: PlayerAnalysis,
        trades: List[Dict],
        current_weights: Dict[str, float],
        current_config: Dict,
        market_regime: str,
    ) -> Optional[Dict]:
        """
        Ask Gemini to make optimization decisions.
        Returns structured JSON with weights, indicators, thresholds.
        """
        # Build trade summary
        trade_details = []
        for t in trades[-20:]:  # Last 20 trades
            trade_details.append({
                "symbol": t.get("symbol", "?"),
                "direction": t.get("direction", "?"),
                "pnl": round(t.get("pnl", 0), 2),
                "exit_reason": t.get("exit_reason", "?"),
                "bars_held": t.get("bars_held", 0),
            })

        # Get learning history for this player
        history = self.learning_history.get(analysis.player_id, [])
        history_summary = ""
        if history:
            recent = history[-3:]
            history_summary = f"\nPREVIOUS SESSIONS ({len(history)} total):\n"
            for h in recent:
                history_summary += f"- WR: {h.get('win_rate', 0):.1%}, PnL: ${h.get('pnl', 0):+.0f}, Changes: {h.get('changes', 'none')}\n"

        # Compact prompt for JSON output
        current_inds = list(current_weights.keys())

        # Build valid indicator list from categories appropriate for player style
        valid_inds_for_prompt = []
        try:
            from trading_evolution.indicators.universe import IndicatorUniverse
            universe = IndicatorUniverse()
            universe.load_all()

            # Get category-specific indicators based on player style
            if analysis.player_label in ["Aggressive", "Momentum"]:
                valid_inds_for_prompt = universe.get_by_category("momentum")[:15]
            elif analysis.player_label == "VolBreakout":
                valid_inds_for_prompt = universe.get_by_category("volatility")[:10] + universe.get_by_category("momentum")[:5]
            elif analysis.player_label == "Conservative":
                valid_inds_for_prompt = universe.get_by_category("trend")[:10] + universe.get_by_category("overlap")[:5]
            else:  # Balanced
                valid_inds_for_prompt = (universe.get_by_category("momentum")[:5] +
                                         universe.get_by_category("trend")[:5] +
                                         universe.get_by_category("volatility")[:5])
        except ImportError:
            valid_inds_for_prompt = ["RSI_7", "RSI_14", "STOCH_5_3", "MACD_12_26_9", "CCI_14", "CMO_14",
                                     "ADX_14", "SUPERTREND_7_3", "ATR_14", "NATR_14", "BBANDS_20_2",
                                     "OBV", "CMF_20", "MFI_14", "EMA_20", "DEMA_20"]

        valid_inds_str = ", ".join(valid_inds_for_prompt[:20])  # Limit to 20 for prompt size

        prompt = f"""Optimize {analysis.player_label} trader. Stats: {analysis.total_trades} trades, {analysis.win_rate:.0%} WR, ${analysis.total_pnl:+.0f}. Regime: {market_regime}. Current entry={current_config.get('entry_threshold')}.

Trades: {json.dumps(trade_details[:5])}

Current indicators: {', '.join(current_inds[:8])}
{history_summary}

Return complete JSON object with ALL these fields:
- weights: object with 8-10 indicator names as keys, float values 0.1-1.0
- entry_threshold: float between 0.15-0.40
- exit_threshold: float between -0.20 to -0.05
- min_hold_bars: int 2-8
- best_indicators: array of 2-3 indicator names
- worst_indicators: array of 0-2 indicator names
- add_indicators: object with indicator names to add and their weights
- remove_indicators: array of indicator names to remove from current set
- recommendations: array of exactly 2 short strings
- reasoning: one short sentence

IMPORTANT: Only use indicators from this list: {valid_inds_str}

Rules: Low WR (<40%) means higher entry threshold. High WR (>55%) means lower threshold. {analysis.player_label} style = {'momentum/volatility focus' if analysis.player_label in ['Aggressive', 'Momentum', 'VolBreakout'] else 'trend/stability focus'}"""

        try:
            response = self.llm._call(prompt, temperature=0.3, max_tokens=4000, json_mode=True)

            if not response:
                print(f"[Gemini] Empty response for {analysis.player_id}")
                return None

            # Parse JSON from response
            # Try to extract JSON from the response
            json_str = response.strip()

            # Handle markdown code blocks
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                json_str = json_str.split("```")[1].split("```")[0].strip()

            # Find JSON object
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = json_str[start:end]

            decision = json.loads(json_str)

            # Validate and clean the decision
            decision = self._validate_decision(decision, current_weights)

            # Store in learning history
            if analysis.player_id not in self.learning_history:
                self.learning_history[analysis.player_id] = []
            self.learning_history[analysis.player_id].append({
                "timestamp": datetime.now().isoformat(),
                "win_rate": analysis.win_rate,
                "pnl": analysis.total_pnl,
                "changes": f"+{len(decision.get('add_indicators', {}))} -{len(decision.get('remove_indicators', []))} inds",
            })

            return decision

        except json.JSONDecodeError as e:
            print(f"[Gemini] JSON parse error for {analysis.player_id}: {e}")
            print(f"[Gemini] Response was: {response[:500] if response else 'None'}...")
            return None
        except Exception as e:
            print(f"[Gemini] Error for {analysis.player_id}: {e}")
            return None

    def _validate_decision(self, decision: Dict, current_weights: Dict) -> Dict:
        """Validate and clean Gemini's decision."""
        # Ensure weights dict exists and has valid values
        weights = decision.get("weights", {})
        if not weights:
            weights = current_weights.copy()

        # Clean weights
        clean_weights = {}
        for ind, w in weights.items():
            if ind in AVAILABLE_INDICATORS:
                clean_weights[ind] = max(0.1, min(1.0, float(w)))

        # Add new indicators
        add_inds = decision.get("add_indicators", {})
        clean_add = {}
        for ind, w in add_inds.items():
            if ind in AVAILABLE_INDICATORS and ind not in clean_weights:
                clean_add[ind] = max(0.1, min(1.0, float(w)))

        # Remove indicators
        remove_inds = decision.get("remove_indicators", [])
        clean_remove = [ind for ind in remove_inds if ind in clean_weights]

        # Validate thresholds
        entry = decision.get("entry_threshold", 0.25)
        entry = max(0.15, min(0.40, float(entry)))

        exit_t = decision.get("exit_threshold", -0.10)
        exit_t = max(-0.20, min(-0.05, float(exit_t)))

        hold = decision.get("min_hold_bars", 4)
        hold = max(2, min(10, int(hold)))

        return {
            "weights": clean_weights,
            "add_indicators": clean_add,
            "remove_indicators": clean_remove,
            "entry_threshold": entry,
            "exit_threshold": exit_t,
            "min_hold_bars": hold,
            "best_indicators": decision.get("best_indicators", [])[:3],
            "worst_indicators": decision.get("worst_indicators", [])[:3],
            "recommendations": decision.get("recommendations", [])[:5],
            "reasoning": decision.get("reasoning", ""),
        }

    def _fallback_analysis(
        self,
        analysis: PlayerAnalysis,
        current_weights: Dict[str, float],
        current_config: Dict,
    ) -> PlayerAnalysis:
        """Fallback when Gemini is not available - simple statistical adjustments."""
        analysis.new_weights = current_weights.copy()
        analysis.new_entry_threshold = current_config.get("entry_threshold", 0.25)
        analysis.new_exit_threshold = current_config.get("exit_threshold", -0.10)
        analysis.new_min_hold_bars = current_config.get("min_hold_bars", 4)

        # Simple threshold adjustment based on win rate
        if analysis.win_rate < 0.40 and analysis.total_trades >= 5:
            analysis.new_entry_threshold = min(0.40, analysis.new_entry_threshold + 0.02)
        elif analysis.win_rate > 0.55 and analysis.total_trades >= 5:
            analysis.new_entry_threshold = max(0.20, analysis.new_entry_threshold - 0.02)

        analysis.llm_analysis = "Using statistical fallback (Gemini not available)"
        return analysis

    def apply_recommendations(
        self,
        config: Dict,
        analysis: PlayerAnalysis,
    ) -> Dict:
        """Apply Gemini's decisions to a player's config."""
        new_config = deepcopy(config)

        # Start with Gemini's new weights
        new_weights = analysis.new_weights.copy()

        # Add new indicators
        for ind, weight in analysis.indicators_to_add.items():
            new_weights[ind] = weight

        # Remove indicators
        for ind in analysis.indicators_to_remove:
            new_weights.pop(ind, None)

        new_config["weights"] = new_weights
        new_config["entry_threshold"] = analysis.new_entry_threshold
        new_config["exit_threshold"] = analysis.new_exit_threshold
        new_config["min_hold_bars"] = analysis.new_min_hold_bars

        return new_config

    def get_coach_summary(self, analysis: PlayerAnalysis) -> str:
        """Generate a human-readable summary of coaching changes."""
        parts = []

        parts.append(f"WR: {analysis.win_rate:.1%}, P&L: ${analysis.total_pnl:+,.0f}")

        if analysis.weight_changes:
            parts.append(f"Adjusted {len(analysis.weight_changes)} weights")

        if analysis.indicators_to_add:
            added = list(analysis.indicators_to_add.keys())
            parts.append(f"Add: {', '.join(added[:3])}")

        if analysis.indicators_to_remove:
            parts.append(f"Remove: {', '.join(analysis.indicators_to_remove[:3])}")

        parts.append(f"Entry: {analysis.new_entry_threshold:.2f}")

        return " | ".join(parts)
