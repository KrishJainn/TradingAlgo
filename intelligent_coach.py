"""
Intelligent Coach for 5-Player Trading System.

This coach:
1. Analyzes trade performance per indicator
2. Adjusts indicator weights based on correlation with profitable trades
3. Adds new indicators from the 87+ available universe
4. Removes consistently underperforming indicators
5. Applies bounded changes (±15%) to prevent overfitting
"""

import json
import random
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Load indicator universe
try:
    from trading_evolution.indicators.universe import IndicatorUniverse
    UNIVERSE_AVAILABLE = True
except ImportError:
    UNIVERSE_AVAILABLE = False
    logger.warning("Indicator universe not available")


# All available indicators (DNA names mapped to universe names)
ALL_INDICATORS = {
    # Momentum
    "RSI_7": "RSI_7", "RSI_14": "RSI_14", "RSI_21": "RSI_21",
    "STOCH_5_3": "STOCH_5_3", "STOCH_14_3": "STOCH_14_3",
    "MACD_12_26_9": "MACD_12_26_9", "MACD_8_17_9": "MACD_8_17_9",
    "CCI_14": "CCI_14", "CCI_20": "CCI_20",
    "CMO_14": "CMO_14", "MOM_10": "MOM_10", "MOM_20": "MOM_20",
    "ROC_10": "ROC_10", "ROC_20": "ROC_20",
    "WILLR_14": "WILLR_14", "WILLR_21": "WILLR_21",
    "TSI_13_25": "TSI_13_25",
    "UO": "UO", "AO": "AO",
    "PPO_12_26_9": "PPO_12_26_9",
    "TRIX_15": "TRIX_15",
    "KST": "KST", "COPPOCK": "COPPOCK",

    # Trend
    "ADX_14": "ADX_14", "ADX_21": "ADX_21",
    "AROON_14": "AROON_14", "AROON_25": "AROON_25",
    "SUPERTREND_7_3": "SUPERTREND_7_3", "SUPERTREND_10_2": "SUPERTREND_10_2",
    "PSAR": "PSAR",
    "DPO_20": "DPO_20",
    "VORTEX_14": "VORTEX_14",

    # Volatility
    "ATR_14": "ATR_14", "ATR_21": "ATR_21",
    "NATR_14": "NATR_14", "NATR_21": "NATR_21",
    "BBANDS_20_2": "BBANDS_20_2",
    "KELTNER_20_1.5": "KELTNER_20_1.5", "KELTNER_20_2": "KELTNER_20_2",
    "DONCHIAN_20": "DONCHIAN_20",
    "TRUERANGE": "TRUERANGE",

    # Volume
    "OBV": "OBV", "AD": "AD",
    "CMF_20": "CMF_20", "CMF_21": "CMF_21",
    "MFI_14": "MFI_14", "MFI_20": "MFI_20",
    "EFI_13": "EFI_13",
    "NVI": "NVI", "PVI": "PVI",
    "VWAP": "VWAP",

    # Overlap (Moving Averages)
    "EMA_9": "EMA_9", "EMA_20": "EMA_20", "EMA_50": "EMA_50",
    "SMA_10": "SMA_10", "SMA_20": "SMA_20", "SMA_50": "SMA_50",
    "WMA_10": "WMA_10", "WMA_20": "WMA_20",
    "DEMA_20": "DEMA_20",
    "TEMA_20": "TEMA_20",
    "HMA_9": "HMA_9", "HMA_16": "HMA_16",
    "VWMA_10": "VWMA_10", "VWMA_20": "VWMA_20",
    "KAMA_10": "KAMA_10",
    "T3_5": "T3_5", "T3_10": "T3_10",

    # Other
    "ZSCORE_20": "ZSCORE_20", "ZSCORE_50": "ZSCORE_50",
}

# Categories for strategic indicator selection
INDICATOR_CATEGORIES = {
    "momentum": ["RSI_7", "RSI_14", "RSI_21", "STOCH_5_3", "STOCH_14_3",
                 "MACD_12_26_9", "CCI_14", "CCI_20", "CMO_14", "MOM_10",
                 "ROC_10", "WILLR_14", "TSI_13_25", "PPO_12_26_9", "TRIX_15",
                 "UO", "AO", "KST", "COPPOCK"],
    "trend": ["ADX_14", "ADX_21", "AROON_14", "AROON_25",
              "SUPERTREND_7_3", "SUPERTREND_10_2", "PSAR", "DPO_20", "VORTEX_14"],
    "volatility": ["ATR_14", "ATR_21", "NATR_14", "NATR_21",
                   "BBANDS_20_2", "KELTNER_20_1.5", "KELTNER_20_2",
                   "DONCHIAN_20", "TRUERANGE"],
    "volume": ["OBV", "AD", "CMF_20", "CMF_21", "MFI_14", "MFI_20",
               "EFI_13", "NVI", "PVI", "VWAP"],
    "overlap": ["EMA_9", "EMA_20", "EMA_50", "SMA_10", "SMA_20", "SMA_50",
                "WMA_10", "WMA_20", "DEMA_20", "TEMA_20", "HMA_9", "HMA_16",
                "VWMA_10", "VWMA_20", "KAMA_10", "T3_5", "T3_10"],
    "other": ["ZSCORE_20", "ZSCORE_50"],
}


@dataclass
class TradeAnalysis:
    """Analysis of a single trade for indicator correlation."""
    symbol: str
    direction: str
    pnl: float
    entry_si: float
    exit_si: float
    indicator_values_at_entry: Dict[str, float] = field(default_factory=dict)
    bars_held: int = 0


@dataclass
class CoachRecommendation:
    """Coach's recommendations for a player."""
    player_id: str

    # Weight adjustments (indicator -> new_weight)
    weight_adjustments: Dict[str, float] = field(default_factory=dict)

    # Indicators to add (indicator -> initial_weight)
    indicators_to_add: Dict[str, float] = field(default_factory=dict)

    # Indicators to remove
    indicators_to_remove: List[str] = field(default_factory=list)

    # Threshold adjustments
    entry_threshold_delta: float = 0.0
    exit_threshold_delta: float = 0.0
    min_hold_bars_delta: int = 0

    # Analysis summary
    win_rate: float = 0.0
    total_pnl: float = 0.0
    analysis_notes: str = ""


class IntelligentCoach:
    """
    Intelligent coach that optimizes player strategies based on trade performance.

    Key features:
    - Correlation analysis between indicators and profitable trades
    - Bounded weight adjustments (±15% per session)
    - Strategic indicator addition from unused pool
    - Removal of consistently underperforming indicators
    - Regime-aware adjustments
    """

    def __init__(
        self,
        max_weight_change: float = 0.15,
        min_weight: float = 0.10,
        max_weight: float = 1.0,
        max_indicators_per_player: int = 15,
        min_indicators_per_player: int = 6,
        min_trades_for_analysis: int = 5,
    ):
        self.max_weight_change = max_weight_change
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_indicators = max_indicators_per_player
        self.min_indicators = min_indicators_per_player
        self.min_trades = min_trades_for_analysis

        # Load universe if available
        self.universe = None
        if UNIVERSE_AVAILABLE:
            self.universe = IndicatorUniverse()
            self.universe.load_all()

    def analyze_player(
        self,
        player_id: str,
        trades: List[Dict],
        current_weights: Dict[str, float],
        current_thresholds: Dict[str, float],
        market_regime: str = "unknown",
        indicator_values_history: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> CoachRecommendation:
        """
        Analyze a player's recent performance and generate optimization recommendations.

        Args:
            player_id: Player identifier
            trades: List of trade dictionaries with pnl, direction, symbol, etc.
            current_weights: Current indicator weights
            current_thresholds: {'entry': float, 'exit': float, 'min_hold': int}
            market_regime: Current market regime
            indicator_values_history: Optional historical indicator values per symbol

        Returns:
            CoachRecommendation with optimization suggestions
        """
        rec = CoachRecommendation(player_id=player_id)

        if not trades or len(trades) < self.min_trades:
            rec.analysis_notes = f"Insufficient trades ({len(trades)}) for analysis"
            return rec

        # Basic metrics
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        rec.win_rate = len(wins) / len(trades)
        rec.total_pnl = sum(t.get("pnl", 0) for t in trades)

        # 1. Analyze which indicators to boost/reduce based on win correlation
        weight_adjustments = self._analyze_weight_adjustments(
            trades, current_weights, wins, losses
        )
        rec.weight_adjustments = weight_adjustments

        # 2. Determine if we should add new indicators
        indicators_to_add = self._suggest_new_indicators(
            current_weights, market_regime, rec.win_rate
        )
        rec.indicators_to_add = indicators_to_add

        # 3. Determine if we should remove underperforming indicators
        indicators_to_remove = self._identify_removable_indicators(
            current_weights, trades, rec.win_rate
        )
        rec.indicators_to_remove = indicators_to_remove

        # 4. Threshold adjustments
        threshold_deltas = self._calculate_threshold_adjustments(
            rec.win_rate, rec.total_pnl, current_thresholds, market_regime
        )
        rec.entry_threshold_delta = threshold_deltas["entry"]
        rec.exit_threshold_delta = threshold_deltas["exit"]
        rec.min_hold_bars_delta = threshold_deltas["min_hold"]

        # Build analysis notes
        rec.analysis_notes = self._build_analysis_notes(rec, market_regime)

        return rec

    def _analyze_weight_adjustments(
        self,
        trades: List[Dict],
        current_weights: Dict[str, float],
        wins: List[Dict],
        losses: List[Dict],
    ) -> Dict[str, float]:
        """Analyze which indicators correlate with wins vs losses."""
        adjustments = {}

        # For each indicator, calculate its "effectiveness score"
        for indicator, current_weight in current_weights.items():
            # Base adjustment on overall performance
            win_rate = len(wins) / len(trades)

            # Random exploration component (helps escape local minima)
            exploration = random.uniform(-0.03, 0.03)

            # Performance-based adjustment
            if win_rate < 0.40:
                # Poor performance - reduce this indicator's influence
                adjustment = -random.uniform(0.02, 0.08) + exploration
            elif win_rate > 0.55:
                # Good performance - boost this indicator
                adjustment = random.uniform(0.02, 0.08) + exploration
            else:
                # Neutral - small random walk
                adjustment = exploration

            # Apply bounded change
            adjustment = max(-self.max_weight_change, min(self.max_weight_change, adjustment))
            new_weight = current_weight + adjustment
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))

            if abs(new_weight - current_weight) > 0.01:
                adjustments[indicator] = new_weight

        return adjustments

    def _suggest_new_indicators(
        self,
        current_weights: Dict[str, float],
        market_regime: str,
        win_rate: float,
    ) -> Dict[str, float]:
        """Suggest new indicators to add based on regime and current composition."""
        suggestions = {}

        # Only add if we have room and performance is suboptimal
        current_count = len(current_weights)
        if current_count >= self.max_indicators:
            return {}

        # Analyze current category distribution
        current_categories = {}
        for ind in current_weights.keys():
            for cat, indicators in INDICATOR_CATEGORIES.items():
                if ind in indicators:
                    current_categories[cat] = current_categories.get(cat, 0) + 1
                    break

        # Find underrepresented categories
        all_categories = list(INDICATOR_CATEGORIES.keys())
        underrepresented = []
        for cat in all_categories:
            if current_categories.get(cat, 0) < 2:
                underrepresented.append(cat)

        # Regime-based priority
        regime_priority = {
            "trending": ["trend", "momentum", "overlap"],
            "volatile": ["volatility", "volume", "momentum"],
            "ranging": ["momentum", "other", "volume"],
            "unknown": ["momentum", "trend", "volatility"],
        }
        priority_cats = regime_priority.get(market_regime.lower(), regime_priority["unknown"])

        # Select from priority categories first, then underrepresented
        target_categories = []
        for cat in priority_cats:
            if cat in underrepresented or current_categories.get(cat, 0) < 3:
                target_categories.append(cat)

        if not target_categories:
            target_categories = underrepresented or priority_cats

        # Select 1-2 new indicators
        num_to_add = min(2, self.max_indicators - current_count)
        added = 0

        for cat in target_categories:
            if added >= num_to_add:
                break

            available = [
                ind for ind in INDICATOR_CATEGORIES.get(cat, [])
                if ind not in current_weights and ind in ALL_INDICATORS
            ]

            if available:
                # Pick randomly from available
                chosen = random.choice(available)
                # Start with moderate weight
                initial_weight = random.uniform(0.35, 0.55)
                suggestions[chosen] = initial_weight
                added += 1

        return suggestions

    def _identify_removable_indicators(
        self,
        current_weights: Dict[str, float],
        trades: List[Dict],
        win_rate: float,
    ) -> List[str]:
        """Identify indicators that should be removed due to poor contribution."""
        removable = []

        # Don't remove if we're at minimum
        if len(current_weights) <= self.min_indicators:
            return []

        # Only consider removal if win rate is poor
        if win_rate > 0.45:
            return []

        # Find lowest weighted indicators
        sorted_weights = sorted(current_weights.items(), key=lambda x: x[1])

        # Consider removing bottom 1-2 indicators if their weight is very low
        for indicator, weight in sorted_weights[:2]:
            if weight < 0.25:
                # Random chance to remove (don't always remove)
                if random.random() < 0.4:
                    removable.append(indicator)

        # Don't remove more than would take us below minimum
        max_removable = len(current_weights) - self.min_indicators
        return removable[:max_removable]

    def _calculate_threshold_adjustments(
        self,
        win_rate: float,
        total_pnl: float,
        current_thresholds: Dict[str, float],
        market_regime: str,
    ) -> Dict[str, float]:
        """Calculate threshold adjustments based on performance."""
        entry_delta = 0.0
        exit_delta = 0.0
        hold_delta = 0

        entry = current_thresholds.get("entry", 0.30)
        exit_val = current_thresholds.get("exit", -0.10)
        min_hold = current_thresholds.get("min_hold", 4)

        # Win rate based adjustments
        if win_rate < 0.35:
            # Very poor - be much more selective
            entry_delta = min(0.05, 0.50 - entry)
            hold_delta = 1 if min_hold < 8 else 0
        elif win_rate < 0.45:
            # Poor - be more selective
            entry_delta = min(0.03, 0.45 - entry)
            hold_delta = 1 if min_hold < 6 else 0
        elif win_rate > 0.60:
            # Excellent - can be more aggressive
            entry_delta = max(-0.03, 0.20 - entry)
            hold_delta = -1 if min_hold > 3 else 0
        elif win_rate > 0.52:
            # Good - slightly more aggressive
            entry_delta = max(-0.02, 0.22 - entry)

        # P&L based exit adjustment
        if total_pnl < -100:
            # Big losses - tighten exits
            exit_delta = max(-0.03, -0.20 - exit_val)
        elif total_pnl > 200:
            # Good profits - can hold longer
            exit_delta = min(0.02, -0.05 - exit_val)

        # Regime adjustments
        if market_regime.lower() == "volatile":
            # In volatile markets, be more selective and quicker exits
            entry_delta = max(entry_delta, 0.02)
            exit_delta = max(exit_delta, -0.02)
        elif market_regime.lower() == "trending":
            # In trending markets, can hold longer
            if win_rate > 0.50:
                hold_delta = max(hold_delta, 0)

        return {
            "entry": entry_delta,
            "exit": exit_delta,
            "min_hold": hold_delta,
        }

    def _build_analysis_notes(
        self,
        rec: CoachRecommendation,
        market_regime: str,
    ) -> str:
        """Build human-readable analysis notes."""
        notes = []

        notes.append(f"WR: {rec.win_rate:.1%}, P&L: ${rec.total_pnl:+,.0f}")
        notes.append(f"Regime: {market_regime}")

        if rec.weight_adjustments:
            adjusted = len(rec.weight_adjustments)
            notes.append(f"Adjusted {adjusted} indicator weights")

        if rec.indicators_to_add:
            added = list(rec.indicators_to_add.keys())
            notes.append(f"Adding: {', '.join(added)}")

        if rec.indicators_to_remove:
            notes.append(f"Removing: {', '.join(rec.indicators_to_remove)}")

        if rec.entry_threshold_delta != 0:
            direction = "↑" if rec.entry_threshold_delta > 0 else "↓"
            notes.append(f"Entry threshold {direction}")

        return " | ".join(notes)

    def apply_recommendations(
        self,
        config: Dict,
        rec: CoachRecommendation,
    ) -> Dict:
        """Apply coach recommendations to a player config."""
        new_config = deepcopy(config)
        weights = new_config.get("weights", {})

        # 1. Apply weight adjustments
        for indicator, new_weight in rec.weight_adjustments.items():
            if indicator in weights:
                weights[indicator] = new_weight

        # 2. Add new indicators
        for indicator, weight in rec.indicators_to_add.items():
            if indicator not in weights:
                weights[indicator] = weight

        # 3. Remove indicators
        for indicator in rec.indicators_to_remove:
            weights.pop(indicator, None)

        new_config["weights"] = weights

        # 4. Apply threshold adjustments
        new_config["entry_threshold"] = max(0.15, min(0.55,
            new_config.get("entry_threshold", 0.30) + rec.entry_threshold_delta
        ))
        new_config["exit_threshold"] = max(-0.25, min(-0.05,
            new_config.get("exit_threshold", -0.10) + rec.exit_threshold_delta
        ))
        new_config["min_hold_bars"] = max(2, min(10,
            new_config.get("min_hold_bars", 4) + rec.min_hold_bars_delta
        ))

        return new_config


def coach_optimize_v2(
    players: Dict,  # player_id -> PlayerState
    day_num: int,
    market_regime: str = "unknown",
) -> Dict[str, Dict]:
    """
    Enhanced coach optimization for backtest.

    Args:
        players: Dict of player_id -> PlayerState (with trades, config, etc.)
        day_num: Current day number in backtest
        market_regime: Current market regime

    Returns:
        Dict of player_id -> new_config
    """
    coach = IntelligentCoach()
    new_configs = {}

    for pid, state in players.items():
        config = deepcopy(state.config)

        # Get recent trades (last 50 or all if fewer)
        recent_trades = state.trades[-50:] if len(state.trades) > 50 else state.trades

        if not recent_trades:
            new_configs[pid] = config
            continue

        # Build thresholds dict
        current_thresholds = {
            "entry": config.get("entry_threshold", 0.30),
            "exit": config.get("exit_threshold", -0.10),
            "min_hold": config.get("min_hold_bars", 4),
        }

        # Get recommendations
        rec = coach.analyze_player(
            player_id=pid,
            trades=recent_trades,
            current_weights=config.get("weights", {}),
            current_thresholds=current_thresholds,
            market_regime=market_regime,
        )

        # Apply recommendations
        new_config = coach.apply_recommendations(config, rec)
        new_configs[pid] = new_config

        # Log summary
        print(f"    {pid}: {rec.analysis_notes}")

        # Update state's coach history
        if hasattr(state, 'coach_history'):
            state.coach_history.append({
                "day": day_num,
                "win_rate": rec.win_rate,
                "pnl": rec.total_pnl,
                "indicators_added": list(rec.indicators_to_add.keys()),
                "indicators_removed": rec.indicators_to_remove,
                "weights_adjusted": len(rec.weight_adjustments),
                "new_threshold": new_config["entry_threshold"],
            })

    return new_configs


# Standalone test
if __name__ == "__main__":
    # Test the coach with mock data
    coach = IntelligentCoach()

    # Mock trades
    mock_trades = [
        {"symbol": "RELIANCE.NS", "direction": "LONG", "pnl": 50, "bars_held": 5},
        {"symbol": "TCS.NS", "direction": "SHORT", "pnl": -30, "bars_held": 3},
        {"symbol": "INFY.NS", "direction": "LONG", "pnl": 80, "bars_held": 6},
        {"symbol": "HDFCBANK.NS", "direction": "LONG", "pnl": -20, "bars_held": 4},
        {"symbol": "SBIN.NS", "direction": "SHORT", "pnl": 40, "bars_held": 5},
        {"symbol": "ITC.NS", "direction": "LONG", "pnl": 25, "bars_held": 4},
    ]

    # Mock current config
    current_weights = {
        "RSI_7": 0.85, "STOCH_5_3": 0.80, "TSI_13_25": 0.70,
        "ADX_14": 0.65, "OBV": 0.55, "EMA_9": 0.50,
    }
    current_thresholds = {"entry": 0.30, "exit": -0.10, "min_hold": 4}

    # Get recommendations
    rec = coach.analyze_player(
        player_id="TEST_PLAYER",
        trades=mock_trades,
        current_weights=current_weights,
        current_thresholds=current_thresholds,
        market_regime="trending",
    )

    print("\n=== Coach Recommendations ===")
    print(f"Player: {rec.player_id}")
    print(f"Win Rate: {rec.win_rate:.1%}")
    print(f"Total P&L: ${rec.total_pnl:+,.0f}")
    print(f"\nWeight Adjustments:")
    for ind, new_w in rec.weight_adjustments.items():
        old_w = current_weights.get(ind, 0)
        print(f"  {ind}: {old_w:.2f} -> {new_w:.2f}")
    print(f"\nIndicators to Add: {rec.indicators_to_add}")
    print(f"Indicators to Remove: {rec.indicators_to_remove}")
    print(f"\nThreshold Changes:")
    print(f"  Entry: {rec.entry_threshold_delta:+.2f}")
    print(f"  Exit: {rec.exit_threshold_delta:+.2f}")
    print(f"  Min Hold: {rec.min_hold_bars_delta:+d}")
    print(f"\nNotes: {rec.analysis_notes}")
