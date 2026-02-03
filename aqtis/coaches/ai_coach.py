"""
AI-Powered Coach for 5-Player Trading System.

Each player gets INDEPENDENT optimization based on:
1. Their unique indicator weights and strategy style
2. Their individual trade history and performance
3. Per-indicator contribution analysis
4. Gemini LLM insights for strategy improvement

Key Features:
- Tracks indicator effectiveness per player
- Correlates indicator signals with trade outcomes
- Adds/removes indicators strategically per player
- Uses LLM for deeper pattern analysis
"""

import json
import logging
import os
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# All available indicators by category
INDICATOR_CATEGORIES = {
    "momentum": [
        "RSI_7", "RSI_14", "RSI_21", "STOCH_5_3", "STOCH_14_3",
        "MACD_12_26_9", "MACD_8_17_9", "CCI_14", "CCI_20",
        "CMO_14", "MOM_10", "MOM_20", "ROC_10", "ROC_20",
        "WILLR_14", "WILLR_21", "TSI_13_25", "UO", "AO",
        "PPO_12_26_9", "TRIX_15", "KST", "COPPOCK",
    ],
    "trend": [
        "ADX_14", "ADX_21", "AROON_14", "AROON_25",
        "SUPERTREND_7_3", "SUPERTREND_10_2", "PSAR",
        "DPO_20", "VORTEX_14",
    ],
    "volatility": [
        "ATR_14", "ATR_21", "NATR_14", "NATR_21",
        "BBANDS_20_2", "KELTNER_20_1.5", "KELTNER_20_2",
        "DONCHIAN_20", "TRUERANGE",
    ],
    "volume": [
        "OBV", "AD", "CMF_20", "CMF_21", "MFI_14", "MFI_20",
        "EFI_13", "NVI", "PVI", "VWAP",
    ],
    "overlap": [
        "EMA_9", "EMA_20", "EMA_50", "SMA_10", "SMA_20", "SMA_50",
        "WMA_10", "WMA_20", "DEMA_20", "TEMA_20",
        "HMA_9", "HMA_16", "VWMA_10", "VWMA_20",
        "KAMA_10", "T3_5", "T3_10",
    ],
    "other": ["ZSCORE_20", "ZSCORE_50"],
}

# Player style profiles for targeted optimization
PLAYER_STYLES = {
    "Aggressive": {
        "preferred_categories": ["momentum", "volatility"],
        "risk_tolerance": "high",
        "hold_preference": "short",
        "entry_range": (0.20, 0.35),
    },
    "Conservative": {
        "preferred_categories": ["trend", "overlap"],
        "risk_tolerance": "low",
        "hold_preference": "long",
        "entry_range": (0.30, 0.45),
    },
    "Balanced": {
        "preferred_categories": ["momentum", "trend", "volume"],
        "risk_tolerance": "medium",
        "hold_preference": "medium",
        "entry_range": (0.25, 0.38),
    },
    "VolBreakout": {
        "preferred_categories": ["volatility", "trend", "volume"],
        "risk_tolerance": "high",
        "hold_preference": "short",
        "entry_range": (0.20, 0.32),
    },
    "Momentum": {
        "preferred_categories": ["momentum", "overlap"],
        "risk_tolerance": "medium-high",
        "hold_preference": "short",
        "entry_range": (0.22, 0.35),
    },
}


@dataclass
class IndicatorScore:
    """Track indicator performance for a specific player."""
    indicator: str
    trades_with_signal: int = 0
    wins_with_signal: int = 0
    total_pnl: float = 0.0
    avg_signal_at_entry: float = 0.0

    @property
    def win_rate(self) -> float:
        if self.trades_with_signal == 0:
            return 0.0
        return self.wins_with_signal / self.trades_with_signal

    @property
    def effectiveness(self) -> float:
        """Score combining win rate and profitability."""
        if self.trades_with_signal < 3:
            return 0.5  # Neutral for insufficient data
        wr_score = self.win_rate
        pnl_score = 0.5 + min(0.5, max(-0.5, self.total_pnl / 1000))
        return 0.6 * wr_score + 0.4 * pnl_score


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

    # Indicator analysis
    indicator_scores: Dict[str, IndicatorScore] = field(default_factory=dict)
    best_indicators: List[str] = field(default_factory=list)
    worst_indicators: List[str] = field(default_factory=list)

    # Recommendations
    weight_changes: Dict[str, float] = field(default_factory=dict)
    indicators_to_add: Dict[str, float] = field(default_factory=dict)
    indicators_to_remove: List[str] = field(default_factory=list)
    threshold_changes: Dict[str, float] = field(default_factory=dict)

    # LLM insights
    llm_analysis: str = ""
    llm_recommendations: List[str] = field(default_factory=list)


class AICoach:
    """
    AI-Powered Coach that optimizes each player independently.

    Uses:
    1. Statistical analysis of per-indicator performance
    2. Player style awareness for targeted improvements
    3. Optional Gemini LLM for deeper insights
    """

    def __init__(
        self,
        use_llm: bool = True,
        llm_provider=None,
        max_weight_change: float = 0.15,
        min_weight: float = 0.10,
        max_weight: float = 1.0,
        max_indicators: int = 12,
        min_indicators: int = 6,
        min_trades_for_analysis: int = 5,
    ):
        self.use_llm = use_llm
        self.llm = llm_provider
        self.max_weight_change = max_weight_change
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_indicators = max_indicators
        self.min_indicators = min_indicators
        self.min_trades = min_trades_for_analysis

        # Per-player historical tracking
        self.player_histories: Dict[str, List[PlayerAnalysis]] = {}

    def analyze_player(
        self,
        player_id: str,
        player_label: str,
        trades: List[Dict],
        current_weights: Dict[str, float],
        current_config: Dict,
        indicator_signals_at_trades: Optional[Dict[str, Dict]] = None,
        market_regime: str = "unknown",
    ) -> PlayerAnalysis:
        """
        Perform deep analysis for a single player.

        Args:
            player_id: Player identifier
            player_label: Player style label (Aggressive, Conservative, etc.)
            trades: List of trade dicts with pnl, direction, symbol, etc.
            current_weights: Current indicator weights for this player
            current_config: Full player config (thresholds, etc.)
            indicator_signals_at_trades: Optional dict mapping trade_id to indicator values
            market_regime: Current market regime

        Returns:
            PlayerAnalysis with recommendations specific to this player
        """
        analysis = PlayerAnalysis(
            player_id=player_id,
            player_label=player_label,
        )

        if not trades or len(trades) < self.min_trades:
            logger.info(f"{player_id}: Insufficient trades ({len(trades)}) for analysis")
            return analysis

        # Basic performance metrics
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
        analysis.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # Per-indicator analysis
        self._analyze_indicators(analysis, trades, current_weights, indicator_signals_at_trades)

        # Generate recommendations based on player style
        player_style = PLAYER_STYLES.get(player_label, PLAYER_STYLES["Balanced"])
        self._generate_weight_recommendations(analysis, current_weights, player_style)
        self._generate_indicator_changes(analysis, current_weights, player_style, market_regime)
        self._generate_threshold_recommendations(analysis, current_config, player_style)

        # Optional LLM analysis
        if self.use_llm and self.llm:
            self._get_llm_insights(analysis, trades, current_weights, market_regime)

        # Store in history
        if player_id not in self.player_histories:
            self.player_histories[player_id] = []
        self.player_histories[player_id].append(analysis)

        return analysis

    def _analyze_indicators(
        self,
        analysis: PlayerAnalysis,
        trades: List[Dict],
        current_weights: Dict[str, float],
        indicator_signals: Optional[Dict[str, Dict]],
    ):
        """Analyze which indicators contributed to wins vs losses."""
        # Initialize scores for all current indicators
        for indicator in current_weights.keys():
            analysis.indicator_scores[indicator] = IndicatorScore(indicator=indicator)

        # If we have indicator signals at trade time, use them
        if indicator_signals:
            for trade in trades:
                trade_id = trade.get("trade_id", str(id(trade)))
                signals = indicator_signals.get(trade_id, {})
                pnl = trade.get("pnl", 0)
                is_win = pnl > 0

                for indicator, signal_value in signals.items():
                    if indicator in analysis.indicator_scores:
                        score = analysis.indicator_scores[indicator]
                        score.trades_with_signal += 1
                        if is_win:
                            score.wins_with_signal += 1
                        score.total_pnl += pnl
                        score.avg_signal_at_entry = (
                            (score.avg_signal_at_entry * (score.trades_with_signal - 1) + signal_value)
                            / score.trades_with_signal
                        )
        else:
            # Without detailed signals, estimate based on overall performance
            for indicator in current_weights.keys():
                score = analysis.indicator_scores[indicator]
                # Use overall stats as proxy
                score.trades_with_signal = analysis.total_trades
                score.wins_with_signal = analysis.wins
                score.total_pnl = analysis.total_pnl / max(1, len(current_weights))

        # Rank indicators
        ranked = sorted(
            analysis.indicator_scores.items(),
            key=lambda x: x[1].effectiveness,
            reverse=True
        )

        analysis.best_indicators = [ind for ind, _ in ranked[:3]]
        analysis.worst_indicators = [ind for ind, _ in ranked[-3:] if ranked[-3:]]

    def _generate_weight_recommendations(
        self,
        analysis: PlayerAnalysis,
        current_weights: Dict[str, float],
        player_style: Dict,
    ):
        """Generate weight adjustment recommendations for this player."""
        for indicator, current_weight in current_weights.items():
            score = analysis.indicator_scores.get(indicator)
            if not score or score.trades_with_signal < 3:
                # Insufficient data - small random exploration
                change = random.uniform(-0.02, 0.02)
            else:
                effectiveness = score.effectiveness

                # High performers get boosted, low performers get reduced
                if effectiveness > 0.6:
                    change = random.uniform(0.03, 0.10)
                elif effectiveness > 0.5:
                    change = random.uniform(-0.02, 0.05)
                elif effectiveness > 0.4:
                    change = random.uniform(-0.05, 0.02)
                else:
                    change = random.uniform(-0.10, -0.03)

                # Boost if it's a preferred category for this player style
                for cat, indicators in INDICATOR_CATEGORIES.items():
                    if indicator in indicators and cat in player_style["preferred_categories"]:
                        change += 0.02
                        break

            # Apply bounded change
            change = max(-self.max_weight_change, min(self.max_weight_change, change))
            new_weight = max(self.min_weight, min(self.max_weight, current_weight + change))

            if abs(new_weight - current_weight) > 0.01:
                analysis.weight_changes[indicator] = new_weight

    def _generate_indicator_changes(
        self,
        analysis: PlayerAnalysis,
        current_weights: Dict[str, float],
        player_style: Dict,
        market_regime: str,
    ):
        """Decide which indicators to add or remove for this player."""
        current_count = len(current_weights)

        # REMOVAL: Only if performing poorly and above minimum
        if analysis.win_rate < 0.42 and current_count > self.min_indicators:
            worst = analysis.worst_indicators[:2]
            for indicator in worst:
                score = analysis.indicator_scores.get(indicator)
                if score and score.effectiveness < 0.4:
                    weight = current_weights.get(indicator, 0)
                    if weight < 0.35 and random.random() < 0.5:
                        analysis.indicators_to_remove.append(indicator)
                        if current_count - len(analysis.indicators_to_remove) <= self.min_indicators:
                            break

        # ADDITION: If below max and could benefit from new indicators
        if current_count < self.max_indicators:
            # Find underrepresented categories for this player
            current_cats = {}
            for ind in current_weights.keys():
                for cat, indicators in INDICATOR_CATEGORIES.items():
                    if ind in indicators:
                        current_cats[cat] = current_cats.get(cat, 0) + 1
                        break

            # Prioritize player's preferred categories
            preferred = player_style["preferred_categories"]

            # Regime-based additions
            regime_boost = {
                "trending": ["trend", "momentum"],
                "volatile": ["volatility", "volume"],
                "volatile_bullish": ["momentum", "volatility"],
                "volatile_bearish": ["volatility", "trend"],
                "ranging": ["momentum", "other"],
            }
            regime_cats = regime_boost.get(market_regime.lower(), [])

            # Find best category to add from
            candidates = []
            for cat in preferred + regime_cats:
                if current_cats.get(cat, 0) < 3:
                    available = [
                        ind for ind in INDICATOR_CATEGORIES.get(cat, [])
                        if ind not in current_weights
                    ]
                    candidates.extend(available)

            if candidates:
                # Add 1-2 indicators
                num_to_add = min(2, self.max_indicators - current_count)
                chosen = random.sample(candidates, min(num_to_add, len(candidates)))
                for ind in chosen:
                    # Initial weight based on player style
                    if player_style["risk_tolerance"] == "high":
                        init_weight = random.uniform(0.45, 0.65)
                    elif player_style["risk_tolerance"] == "low":
                        init_weight = random.uniform(0.35, 0.50)
                    else:
                        init_weight = random.uniform(0.40, 0.55)
                    analysis.indicators_to_add[ind] = init_weight

    def _generate_threshold_recommendations(
        self,
        analysis: PlayerAnalysis,
        current_config: Dict,
        player_style: Dict,
    ):
        """Generate threshold adjustments based on player performance and style."""
        entry = current_config.get("entry_threshold", 0.30)
        exit_thresh = current_config.get("exit_threshold", -0.10)
        min_hold = current_config.get("min_hold_bars", 4)

        entry_min, entry_max = player_style["entry_range"]

        # Entry threshold adjustment
        if analysis.win_rate < 0.38:
            # Too many losing trades - be more selective
            new_entry = min(entry_max, entry + 0.03)
        elif analysis.win_rate < 0.45:
            new_entry = min(entry_max, entry + 0.02)
        elif analysis.win_rate > 0.58:
            # Winning a lot - can be more aggressive
            new_entry = max(entry_min, entry - 0.02)
        elif analysis.win_rate > 0.52:
            new_entry = max(entry_min, entry - 0.01)
        else:
            new_entry = entry

        # Exit threshold adjustment
        if analysis.avg_loss < -50 and analysis.profit_factor < 1.0:
            # Big losses - tighten stops
            new_exit = max(-0.18, exit_thresh - 0.02)
        elif analysis.profit_factor > 1.5:
            # Profitable - can let winners run
            new_exit = min(-0.05, exit_thresh + 0.01)
        else:
            new_exit = exit_thresh

        # Min hold adjustment
        if player_style["hold_preference"] == "short":
            target_hold = 3
        elif player_style["hold_preference"] == "long":
            target_hold = 6
        else:
            target_hold = 4

        if analysis.win_rate < 0.40:
            new_hold = min(8, min_hold + 1)
        elif analysis.win_rate > 0.55:
            new_hold = max(2, min_hold - 1)
        else:
            new_hold = min_hold

        # Bias toward style preference
        new_hold = int((new_hold + target_hold) / 2)

        analysis.threshold_changes = {
            "entry_threshold": round(new_entry, 3),
            "exit_threshold": round(new_exit, 3),
            "min_hold_bars": new_hold,
        }

    def _get_llm_insights(
        self,
        analysis: PlayerAnalysis,
        trades: List[Dict],
        current_weights: Dict[str, float],
        market_regime: str,
    ):
        """Use Gemini LLM to analyze patterns and provide recommendations."""
        if not self.llm:
            return

        try:
            # Build context for LLM
            best_inds = ", ".join(analysis.best_indicators) if analysis.best_indicators else "None identified"
            worst_inds = ", ".join(analysis.worst_indicators) if analysis.worst_indicators else "None identified"

            # Sample recent trades for context
            recent_trades = trades[-10:] if len(trades) > 10 else trades
            trade_summary = []
            for t in recent_trades:
                trade_summary.append(
                    f"- {t.get('symbol', '?')}: {t.get('direction', '?')} -> P&L: {t.get('pnl', 0):+.2f}"
                )

            prompt = f"""Analyze this trading player's performance and provide specific recommendations.

PLAYER: {analysis.player_label} (ID: {analysis.player_id})
STYLE: {PLAYER_STYLES.get(analysis.player_label, {})}
MARKET REGIME: {market_regime}

PERFORMANCE:
- Trades: {analysis.total_trades} (W: {analysis.wins}, L: {analysis.losses})
- Win Rate: {analysis.win_rate:.1%}
- Total P&L: ${analysis.total_pnl:+,.2f}
- Avg Win: ${analysis.avg_win:+,.2f}
- Avg Loss: ${analysis.avg_loss:+,.2f}
- Profit Factor: {analysis.profit_factor:.2f}

CURRENT INDICATORS ({len(current_weights)}):
Top performers: {best_inds}
Underperformers: {worst_inds}

RECENT TRADES:
{chr(10).join(trade_summary)}

Provide 3-5 specific, actionable recommendations to improve this player's strategy.
Focus on:
1. Which indicators to prioritize or remove
2. Entry/exit threshold adjustments
3. Position sizing or hold time changes
4. Any pattern you see in the recent trades

Be concise and specific to THIS player's style ({analysis.player_label})."""

            response = self.llm._call(prompt, temperature=0.4, max_tokens=500)

            if response:
                analysis.llm_analysis = response
                # Extract recommendations (simple split)
                lines = response.strip().split('\n')
                analysis.llm_recommendations = [
                    line.strip() for line in lines
                    if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('-'))
                ][:5]

        except Exception as e:
            logger.warning(f"LLM analysis failed for {analysis.player_id}: {e}")

    def apply_recommendations(
        self,
        config: Dict,
        analysis: PlayerAnalysis,
    ) -> Dict:
        """Apply coach recommendations to a player's config."""
        new_config = deepcopy(config)
        weights = new_config.get("weights", {})

        # Apply weight changes
        for indicator, new_weight in analysis.weight_changes.items():
            if indicator in weights:
                weights[indicator] = new_weight

        # Add new indicators
        for indicator, weight in analysis.indicators_to_add.items():
            if indicator not in weights:
                weights[indicator] = weight

        # Remove indicators
        for indicator in analysis.indicators_to_remove:
            weights.pop(indicator, None)

        new_config["weights"] = weights

        # Apply threshold changes
        for key, value in analysis.threshold_changes.items():
            new_config[key] = value

        return new_config

    def get_coach_summary(self, analysis: PlayerAnalysis) -> str:
        """Generate a human-readable summary of coaching changes."""
        parts = []

        parts.append(f"WR: {analysis.win_rate:.1%}, P&L: ${analysis.total_pnl:+,.0f}")

        if analysis.weight_changes:
            changes = len(analysis.weight_changes)
            parts.append(f"Adjusted {changes} weights")

        if analysis.indicators_to_add:
            added = list(analysis.indicators_to_add.keys())
            parts.append(f"Add: {', '.join(added)}")

        if analysis.indicators_to_remove:
            parts.append(f"Remove: {', '.join(analysis.indicators_to_remove)}")

        if analysis.threshold_changes:
            new_entry = analysis.threshold_changes.get("entry_threshold")
            if new_entry:
                parts.append(f"Entry: {new_entry:.2f}")

        return " | ".join(parts)


def coach_session(
    players: Dict,  # player_id -> PlayerState with trades, config, etc.
    day_num: int,
    market_regime: str = "unknown",
    use_llm: bool = True,
    llm_provider=None,
) -> Tuple[Dict[str, Dict], Dict[str, PlayerAnalysis]]:
    """
    Run a coaching session for all players.

    Each player is analyzed and optimized INDEPENDENTLY.

    Returns:
        Tuple of (new_configs dict, analyses dict)
    """
    coach = AICoach(use_llm=use_llm, llm_provider=llm_provider)

    new_configs = {}
    analyses = {}

    for pid, state in players.items():
        config = deepcopy(state.config)
        label = config.get("label", "Balanced")

        # Get recent trades
        recent_trades = state.trades[-50:] if len(state.trades) > 50 else state.trades

        if not recent_trades:
            new_configs[pid] = config
            analyses[pid] = PlayerAnalysis(player_id=pid, player_label=label)
            continue

        # Analyze this player
        analysis = coach.analyze_player(
            player_id=pid,
            player_label=label,
            trades=recent_trades,
            current_weights=config.get("weights", {}),
            current_config=config,
            market_regime=market_regime,
        )

        # Apply recommendations
        new_config = coach.apply_recommendations(config, analysis)
        new_configs[pid] = new_config
        analyses[pid] = analysis

        # Log summary
        summary = coach.get_coach_summary(analysis)
        print(f"  {pid} ({label}): {summary}")

    return new_configs, analyses
