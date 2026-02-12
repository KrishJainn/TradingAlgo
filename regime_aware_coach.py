"""
Regime-Aware Coach for the 5-Player Closed-Loop Trading System.

Wraps the existing AICoach with regime detection so that every coaching
cycle is *conditioned* on the current market regime:

  1. At each coaching interval, calls RegimeDetector.predict() for the
     current regime + confidence + duration.
  2. Uses get_regime_weights() to scale player capital allocations.
  3. Applies per-player regime adjustments via RegimeAwarePlayer.
  4. Feeds risk overrides (from RegimeDetector.get_risk_adjustments())
     to PortfolioRiskManager.
  5. Passes slippage_multiplier to TransactionCostModel.
  6. Embeds regime context into the Gemini prompt (regime, confidence,
     duration, last-5 transitions, per-player regime P&L).
  7. Tracks regime-conditional performance for all players.
  8. Logs regime transitions.

Usage — drop-in replacement for the coaching block in continuous_backtest.py:

    from regime_aware_coach import RegimeAwareCoach

    rac = RegimeAwareCoach(
        ai_coach=coach,             # existing AICoach instance
        regime_detector=rd,         # RegimeDetector (fit already called)
        nifty_ohlcv=nifty_df,       # daily Nifty 50 OHLCV for predict()
    )

    # Inside the day loop, at coaching intervals:
    coaching_output = rac.run_coaching_cycle(
        players_data=players_data,
        market_data=data,
    )
    for pid, adj_cfg in coaching_output.adjusted_configs.items():
        players[pid].config = adj_cfg
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Optional imports — degrade gracefully
try:
    from regime_detector import RegimeDetector, load_nifty_data
    REGIME_DETECTOR_AVAILABLE = True
except ImportError:
    REGIME_DETECTOR_AVAILABLE = False
    logger.info("regime_detector not available; RegimeAwareCoach will use fallback regime.")

try:
    from regime_aware_player import (
        RegimeAwarePlayer, REGIME_PLAYER_REGISTRY, build_registry, PLAYER_LABELS,
    )
    REGIME_PLAYER_AVAILABLE = True
except ImportError:
    REGIME_PLAYER_AVAILABLE = False
    logger.info("regime_aware_player not available; no player-level adjustments.")

try:
    from portfolio_risk_manager import PortfolioRiskManager, RiskLimits
    RISK_MANAGER_AVAILABLE = True
except ImportError:
    RISK_MANAGER_AVAILABLE = False

try:
    from transaction_costs import TransactionCostModel
    COST_MODEL_AVAILABLE = True
except ImportError:
    COST_MODEL_AVAILABLE = False


# ---------------------------------------------------------------------------
# Coaching output dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoachingOutput:
    """Structured result of a single coaching cycle."""

    regime: str = "Sideways"
    regime_confidence: float = 0.0
    regime_duration: int = 0
    regime_probabilities: Dict[str, float] = field(default_factory=dict)

    # Per-player adjusted configs (ready to assign to PlayerState.config)
    adjusted_configs: Dict[str, Dict] = field(default_factory=dict)

    # Regime-specific overrides that the caller can forward to the risk manager
    risk_overrides: Dict[str, float] = field(default_factory=dict)

    # Slippage multiplier for TransactionCostModel
    slippage_multiplier: float = 1.0

    # Player allocation weights (from RegimeDetector.get_regime_weights)
    player_weights: Dict[str, float] = field(default_factory=dict)

    # Transition flag
    regime_changed: bool = False
    previous_regime: str = "Sideways"

    # Timestamp
    timestamp: str = ""

    def summary(self) -> str:
        conf = self.regime_probabilities.get(self.regime, 0)
        changed_str = " [TRANSITION]" if self.regime_changed else ""
        return (
            f"Regime={self.regime} ({conf:.0%}, {self.regime_duration}d){changed_str} | "
            f"Slippage={self.slippage_multiplier:.1f}x | "
            f"Players adjusted={len(self.adjusted_configs)}"
        )


# ---------------------------------------------------------------------------
# Regime transition tracker
# ---------------------------------------------------------------------------

class RegimeTransitionTracker:
    """
    Track regime transitions over time.

    Provides the last-N transitions and duration in current regime.
    """

    def __init__(self, max_history: int = 50):
        self.max_history = max_history
        self._history: List[Dict[str, Any]] = []
        self._current_regime: str = "Sideways"

    @property
    def current_regime(self) -> str:
        return self._current_regime

    def update(self, regime: str, confidence: float = 0.0) -> bool:
        """
        Update with a new regime observation.

        Returns True if the regime changed (transition).
        """
        changed = regime != self._current_regime and len(self._history) > 0
        prev = self._current_regime
        self._current_regime = regime

        self._history.append({
            "regime": regime,
            "previous": prev,
            "confidence": round(confidence, 4),
            "timestamp": datetime.now().isoformat(),
            "changed": changed,
        })

        # Trim history
        if len(self._history) > self.max_history:
            self._history = self._history[-self.max_history:]

        if changed:
            print(f"[RegimeCoach] Regime transition: {prev} → {regime} (conf={confidence:.0%})")

        return changed

    def last_n_transitions(self, n: int = 5) -> List[Dict[str, Any]]:
        """Return the last N actual transitions (where regime changed)."""
        transitions = [h for h in self._history if h.get("changed", False)]
        return transitions[-n:]

    def transition_summary(self, n: int = 5) -> str:
        """One-line string for LLM prompt embedding."""
        transitions = self.last_n_transitions(n)
        if not transitions:
            return "No regime transitions recorded yet."
        parts = []
        for t in transitions:
            ts = t.get("timestamp", "?")[:10]
            parts.append(f"{t['previous']}→{t['regime']}({ts})")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# RegimeAwareCoach
# ---------------------------------------------------------------------------

class RegimeAwareCoach:
    """
    Wraps AICoach with regime detection for regime-conditional coaching.

    This is the top-level integration class.  Drop it into the backtest
    runner and call ``run_coaching_cycle()`` at each coaching interval.
    """

    # Persistence path for regime stats
    _REGIME_STATS_PATH = Path(__file__).parent / "regime_coach_stats.json"

    def __init__(
        self,
        ai_coach=None,
        regime_detector: Optional[Any] = None,
        nifty_ohlcv=None,
        risk_manager: Optional[Any] = None,
        cost_model: Optional[Any] = None,
        *,
        player_registry: Optional[Dict[str, RegimeAwarePlayer]] = None,
        auto_apply_risk_overrides: bool = True,
        auto_apply_slippage: bool = True,
    ):
        """
        Args:
            ai_coach:       Existing AICoach instance (or None for standalone usage).
            regime_detector: Fitted RegimeDetector instance.
            nifty_ohlcv:    Daily Nifty OHLCV DataFrame for regime prediction.
            risk_manager:   PortfolioRiskManager instance (optional).
            cost_model:     TransactionCostModel instance (optional).
            player_registry: Dict[player_id → RegimeAwarePlayer].  If None,
                             builds from the default REGIME_PLAYER_REGISTRY.
            auto_apply_risk_overrides: Whether to patch risk_manager limits
                automatically after detecting a regime.
            auto_apply_slippage: Whether to patch cost_model slippage
                automatically.
        """
        self.ai_coach = ai_coach
        self.regime_detector = regime_detector
        self.nifty_ohlcv = nifty_ohlcv
        self.risk_manager = risk_manager
        self.cost_model = cost_model

        self.auto_apply_risk = auto_apply_risk_overrides
        self.auto_apply_slippage = auto_apply_slippage

        # Player-level regime adapters
        if player_registry is not None:
            self._players = player_registry
        elif REGIME_PLAYER_AVAILABLE:
            self._players = build_registry()
        else:
            self._players = {}

        # Transition tracker
        self._transition_tracker = RegimeTransitionTracker()

        # Coaching cycle counter
        self._cycle_count = 0

        # Load persisted regime stats into player records
        self._load_regime_stats()

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def run_coaching_cycle(
        self,
        players_data: List[Dict],
        market_data: Optional[Dict] = None,
    ) -> CoachingOutput:
        """
        Execute a full regime-aware coaching cycle.

        Steps:
          1. Detect current regime (or fallback to "Sideways").
          2. Track transition.
          3. Apply risk overrides and slippage multiplier.
          4. Build enriched Gemini prompt context.
          5. Call AICoach.analyze_players_batch() with regime info.
          6. Apply per-player regime adjustments via RegimeAwarePlayer.
          7. Record results and return CoachingOutput.

        Args:
            players_data: List of dicts with keys:
                player_id, player_label, trades, weights, config
            market_data: Optional dict of symbol → OHLCV DataFrame.

        Returns:
            CoachingOutput with adjusted configs and metadata.
        """
        self._cycle_count += 1
        output = CoachingOutput(timestamp=datetime.now().isoformat())

        # ------- 1. Detect regime -------
        regime, probs, duration = self._detect_regime()
        output.regime = regime
        output.regime_probabilities = probs
        output.regime_duration = duration
        output.regime_confidence = probs.get(regime, 0.0)

        # ------- 2. Track transition -------
        prev_regime = self._transition_tracker.current_regime  # capture BEFORE update
        changed = self._transition_tracker.update(regime, output.regime_confidence)
        output.regime_changed = changed
        output.previous_regime = prev_regime

        # ------- 3. Get regime-conditional parameters -------
        if self.regime_detector is not None:
            output.risk_overrides = self.regime_detector.get_risk_adjustments(regime)
            output.slippage_multiplier = output.risk_overrides.get("slippage_multiplier", 1.0)
            output.player_weights = self.regime_detector.get_regime_weights(regime)
        else:
            output.risk_overrides = {}
            output.slippage_multiplier = 1.0
            output.player_weights = {}

        # ------- 4. Apply risk overrides to PortfolioRiskManager -------
        if self.auto_apply_risk and self.risk_manager is not None and output.risk_overrides:
            self._apply_risk_overrides(output.risk_overrides)

        # ------- 5. Apply slippage multiplier to TransactionCostModel -------
        if self.auto_apply_slippage and self.cost_model is not None:
            self._apply_slippage_multiplier(output.slippage_multiplier)

        # ------- 6. Call AICoach with enriched regime context -------
        batch_results = {}
        if self.ai_coach is not None and players_data:
            # Enrich players_data with regime context for the Gemini prompt
            enriched = self._enrich_players_data(players_data, regime, probs, duration)
            batch_results = self.ai_coach.analyze_players_batch(
                enriched,
                market_regime=regime,
                market_data=market_data,
            )

        # ------- 7. Per-player regime adjustments -------
        for p in players_data:
            pid = p["player_id"]
            base_config = p.get("config", {})

            # Start from the coach's recommended config (if coach ran)
            if pid in batch_results and self.ai_coach is not None:
                coached_config = self.ai_coach.apply_recommendations(
                    base_config, batch_results[pid]
                )
            else:
                coached_config = deepcopy(base_config)

            # Apply regime-aware player adjustments on top
            rap = self._players.get(pid)
            if rap is not None:
                adjusted = rap.adjust_config(coached_config, regime)
            else:
                adjusted = coached_config

            output.adjusted_configs[pid] = adjusted

        print(f"[RegimeCoach] Cycle #{self._cycle_count}: {output.summary()}")
        return output

    def record_run_results(
        self,
        regime: str,
        player_results: Dict[str, Dict],
    ) -> None:
        """
        Record per-player run results for regime-conditional tracking.

        Call after a backtest run completes.

        Args:
            regime:         The regime label for this run.
            player_results: Dict of player_id → {"pnl": float, "sharpe": float, "win_rate": float, ...}
        """
        for pid, pdata in player_results.items():
            rap = self._players.get(pid)
            if rap is not None:
                rap.record_result(
                    regime=regime,
                    pnl=pdata.get("pnl", 0.0),
                    sharpe=pdata.get("sharpe", 0.0),
                    win_rate=pdata.get("win_rate", 0.0),
                )
        self._save_regime_stats()

    def get_regime_context_for_prompt(
        self,
        regime: str,
        probs: Dict[str, float],
        duration: int,
        player_id: Optional[str] = None,
    ) -> str:
        """
        Build a regime context block suitable for injection into the Gemini prompt.

        Contains:
          - Current regime + confidence + duration
          - Last 5 transitions
          - Per-player regime performance (if player_id given)
        """
        lines = []
        lines.append(f"MARKET REGIME: {regime} (confidence={probs.get(regime, 0):.0%}, duration={duration} days)")

        # Transition history
        trans_summary = self._transition_tracker.transition_summary(5)
        if trans_summary:
            lines.append(f"RECENT TRANSITIONS: {trans_summary}")

        # Player-specific regime history
        if player_id and player_id in self._players:
            rap = self._players[player_id]
            perf = rap.get_regime_performance_summary()
            if perf:
                lines.append(f"YOUR REGIME HISTORY: {perf}")

        return "\n".join(lines)

    def get_player_allocation_weights(self, regime: str) -> Dict[str, float]:
        """
        Return player capital-allocation weights for the given regime.

        Keys are player_id (PLAYER_1, etc.), values are weights (sum ≈ 1.0).
        """
        if self.regime_detector is None:
            return {pid: 0.20 for pid in self._players}

        # regime_detector returns label-keyed weights; convert to player_id
        label_weights = self.regime_detector.get_regime_weights(regime)

        pid_weights = {}
        for pid, rap in self._players.items():
            pid_weights[pid] = label_weights.get(rap.player_label, 0.20)
        return pid_weights

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_regime(self) -> Tuple[str, Dict[str, float], int]:
        """Call RegimeDetector.predict() or return Sideways fallback."""
        if self.regime_detector is not None and self.nifty_ohlcv is not None:
            try:
                return self.regime_detector.predict(self.nifty_ohlcv)
            except Exception as e:
                logger.warning(f"Regime prediction failed: {e}")

        return "Sideways", {"Bull": 0.33, "Bear": 0.33, "Sideways": 0.34}, 0

    def _enrich_players_data(
        self,
        players_data: List[Dict],
        regime: str,
        probs: Dict[str, float],
        duration: int,
    ) -> List[Dict]:
        """
        Inject regime context into each player's data dict.

        The AICoach prompt-builder can read these extra keys.
        """
        enriched = []
        # Compute allocation weights once (identical for all players)
        alloc = self.get_player_allocation_weights(regime)
        for p in players_data:
            ep = deepcopy(p)
            pid = p["player_id"]

            # Add regime context that will appear in the prompt
            regime_ctx = self.get_regime_context_for_prompt(
                regime, probs, duration, player_id=pid,
            )
            ep["regime_context"] = regime_ctx

            # Add player weight (coach can use for allocation suggestions)
            ep["regime_allocation_weight"] = alloc.get(pid, 0.20)

            # Add sizing multiplier info
            rap = self._players.get(pid)
            if rap is not None:
                ep["position_sizing_multiplier"] = rap.get_sizing_multiplier(regime)

            enriched.append(ep)
        return enriched

    def _apply_risk_overrides(self, overrides: Dict[str, float]) -> None:
        """Patch PortfolioRiskManager.limits with regime-based overrides."""
        rm = self.risk_manager
        if rm is None or not hasattr(rm, "limits"):
            return

        limits = rm.limits
        mapping = {
            "max_single_stock_pct": "max_single_stock_pct",
            "max_sector_pct": "max_sector_pct",
            "max_gross_exposure_pct": "max_gross_exposure_pct",
        }
        for override_key, limit_attr in mapping.items():
            if override_key in overrides and hasattr(limits, limit_attr):
                old = getattr(limits, limit_attr)
                new_val = overrides[override_key]
                setattr(limits, limit_attr, new_val)
                if abs(old - new_val) > 0.001:
                    logger.info(f"[RiskOverride] {limit_attr}: {old:.2f} → {new_val:.2f}")

    def _apply_slippage_multiplier(self, multiplier: float) -> None:
        """Patch TransactionCostModel base_slippage_pct with regime multiplier."""
        cm = self.cost_model
        if cm is None:
            return

        # Store the original base_slippage_pct once so we can always scale
        # from it (avoids compounding drift across repeated calls).
        if not hasattr(cm, "_original_base_slippage_pct"):
            cm._original_base_slippage_pct = cm.base_slippage_pct
        cm.base_slippage_pct = cm._original_base_slippage_pct * multiplier
        logger.info(f"[SlippageOverride] base_slippage_pct → {cm.base_slippage_pct:.6f} ({multiplier:.1f}x)")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_regime_stats(self) -> None:
        """Persist per-player regime stats to JSON (full history lists)."""
        try:
            data = {}
            for pid, rap in self._players.items():
                player_data = {}
                for regime, rec in rap._regime_stats.items():
                    player_data[regime] = {
                        "pnl_history": rec.pnl_history,
                        "sharpe_history": rec.sharpe_history,
                        "win_rate_history": rec.win_rate_history,
                    }
                data[pid] = player_data
            with open(self._REGIME_STATS_PATH, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save regime stats: {e}")

    def _load_regime_stats(self) -> None:
        """Load persisted regime stats back into player records (full history)."""
        if not self._REGIME_STATS_PATH.exists():
            return
        try:
            with open(self._REGIME_STATS_PATH, "r") as f:
                data = json.load(f)
            for pid, regimes in data.items():
                rap = self._players.get(pid)
                if rap is None:
                    continue
                for regime, stats in regimes.items():
                    # New format: full history lists
                    if "pnl_history" in stats:
                        from regime_aware_player import RegimePerformanceRecord
                        rec = RegimePerformanceRecord(
                            pnl_history=list(stats["pnl_history"]),
                            sharpe_history=list(stats["sharpe_history"]),
                            win_rate_history=list(stats["win_rate_history"]),
                        )
                        rap._regime_stats[regime.strip().title()] = rec
                    else:
                        # Legacy format: single summary — replay N copies
                        n = stats.get("n_samples", 0)
                        if n > 0:
                            for _ in range(n):
                                rap.record_result(
                                    regime,
                                    pnl=stats.get("avg_pnl", 0.0),
                                    sharpe=stats.get("avg_sharpe", 0.0),
                                    win_rate=stats.get("avg_win_rate", 0.0),
                                )
            logger.info(f"Loaded regime stats from {self._REGIME_STATS_PATH}")
        except Exception as e:
            logger.warning(f"Failed to load regime stats: {e}")


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Regime-Aware Coach — Demo")
    print("=" * 70)

    # ------ Build components ------
    # We test the coach in isolation (no actual AICoach / RegimeDetector).
    # This demonstrates the wiring and data flow.

    # Mock regime detector
    class MockRegimeDetector:
        def predict(self, ohlcv, lookback=5):
            return "Bull", {"Bull": 0.78, "Bear": 0.07, "Sideways": 0.15}, 12

        def get_regime_weights(self, regime):
            weights = {
                "Bull": {"Momentum": 0.30, "Aggressive": 0.25, "Balanced": 0.20, "VolBreakout": 0.10, "Conservative": 0.15},
                "Bear": {"Momentum": 0.10, "Aggressive": 0.10, "Balanced": 0.15, "VolBreakout": 0.30, "Conservative": 0.35},
                "Sideways": {"Momentum": 0.15, "Aggressive": 0.15, "Balanced": 0.30, "VolBreakout": 0.20, "Conservative": 0.20},
            }
            return weights.get(regime, weights["Sideways"])

        def get_risk_adjustments(self, regime):
            if regime == "Bull":
                return {"max_single_stock_pct": 0.25, "max_sector_pct": 0.40,
                        "max_gross_exposure_pct": 2.0, "slippage_multiplier": 0.8}
            elif regime == "Bear":
                return {"max_single_stock_pct": 0.15, "max_sector_pct": 0.30,
                        "max_gross_exposure_pct": 1.2, "slippage_multiplier": 1.5}
            return {"max_single_stock_pct": 0.20, "max_sector_pct": 0.35,
                    "max_gross_exposure_pct": 1.5, "slippage_multiplier": 1.0}

        def get_indicator_adjustments(self, regime):
            return {}

    detector = MockRegimeDetector()
    import pandas as pd
    dummy_ohlcv = pd.DataFrame()  # predict() is mocked

    rac = RegimeAwareCoach(
        ai_coach=None,           # no real coach — we test wiring only
        regime_detector=detector,
        nifty_ohlcv=dummy_ohlcv,
    )

    # ------ Simulate a coaching cycle ------
    print("\n[Demo] Running coaching cycle with mock regime detector...")

    demo_configs = {
        "PLAYER_1": {
            "label": "Aggressive",
            "weights": {"RSI_7": 0.90, "STOCH_5_3": 0.85, "ADX_14": 0.50, "SUPERTREND_7_3": 0.60, "BBANDS_20_2": 0.55},
            "entry_threshold": 0.25,
            "exit_threshold": -0.10,
            "min_hold_bars": 4,
        },
        "PLAYER_2": {
            "label": "Conservative",
            "weights": {"ADX_14": 0.90, "SUPERTREND_7_3": 0.85, "EMA_50": 0.80, "RSI_14": 0.60, "BBANDS_20_2": 0.55},
            "entry_threshold": 0.30,
            "exit_threshold": -0.15,
            "min_hold_bars": 5,
        },
        "PLAYER_3": {
            "label": "Balanced",
            "weights": {"RSI_14": 0.85, "BBANDS_20_2": 0.80, "DEMA_20": 0.50, "ADX_14": 0.45, "ATR_14": 0.40},
            "entry_threshold": 0.25,
            "exit_threshold": -0.10,
            "min_hold_bars": 4,
        },
        "PLAYER_4": {
            "label": "VolBreakout",
            "weights": {"NATR_14": 0.95, "KC_20_2": 0.90, "ADX_14": 0.85, "BBANDS_20_2": 0.75, "ATR_14": 0.70},
            "entry_threshold": 0.22,
            "exit_threshold": -0.08,
            "min_hold_bars": 3,
        },
        "PLAYER_5": {
            "label": "Momentum",
            "weights": {"RSI_7": 0.95, "TSI_13_25": 0.90, "MACD_12_26_9": 0.85, "CMO_14": 0.80, "STOCH_5_3": 0.75},
            "entry_threshold": 0.23,
            "exit_threshold": -0.10,
            "min_hold_bars": 4,
        },
    }

    players_data = [
        {
            "player_id": pid,
            "player_label": cfg["label"],
            "trades": [{"pnl": 100, "symbol": "RELIANCE.NS", "direction": "LONG", "exit_reason": "signal_exit", "bars_held": 5}],
            "weights": cfg["weights"],
            "config": cfg,
        }
        for pid, cfg in demo_configs.items()
    ]

    result = rac.run_coaching_cycle(players_data)

    print(f"\n  Regime:      {result.regime} ({result.regime_confidence:.0%})")
    print(f"  Duration:    {result.regime_duration} days")
    print(f"  Changed:     {result.regime_changed}")
    print(f"  Slippage:    {result.slippage_multiplier:.1f}x")
    print(f"  Risk limits: single={result.risk_overrides.get('max_single_stock_pct', '?')}, "
          f"sector={result.risk_overrides.get('max_sector_pct', '?')}")

    # Show player allocation weights
    print(f"\n  Player allocation weights (for regime '{result.regime}'):")
    for pid, w in sorted(result.player_weights.items(), key=lambda x: -x[1]):
        print(f"    {pid:14s}  {w:.0%}")

    # Show adjusted configs
    print(f"\n  Adjusted configs:")
    for pid, cfg in result.adjusted_configs.items():
        sizing = cfg.get("position_sizing_multiplier", 1.0)
        entry = cfg.get("entry_threshold", "?")
        exit_t = cfg.get("exit_threshold", "?")
        print(f"    {pid}  sizing={sizing:.1f}x  entry={entry}  exit={exit_t}")

    # ------ Test regime transition ------
    print(f"\n[Demo] Simulating regime transition to Bear...")
    detector_bear = MockRegimeDetector()
    detector_bear.predict = lambda ohlcv, lookback=5: ("Bear", {"Bull": 0.05, "Bear": 0.82, "Sideways": 0.13}, 3)
    rac.regime_detector = detector_bear
    result2 = rac.run_coaching_cycle(players_data)
    print(f"  Transition detected: {result2.regime_changed}  ({result2.previous_regime} → {result2.regime})")

    # ------ Record and query regime stats ------
    print(f"\n[Demo] Recording run results and querying stats...")
    rac.record_run_results("Bull", {
        "PLAYER_1": {"pnl": 1500, "sharpe": 2.1, "win_rate": 0.52},
        "PLAYER_2": {"pnl": 800, "sharpe": 1.2, "win_rate": 0.48},
        "PLAYER_3": {"pnl": 600, "sharpe": 0.9, "win_rate": 0.45},
        "PLAYER_4": {"pnl": 200, "sharpe": 0.3, "win_rate": 0.40},
        "PLAYER_5": {"pnl": 1800, "sharpe": 2.5, "win_rate": 0.55},
    })

    for pid, rap in rac._players.items():
        stats = rap.get_all_regime_stats()
        if stats:
            print(f"    {pid} ({rap.player_label}): {stats}")

    # ------ Test prompt context ------
    print(f"\n[Demo] Prompt context for PLAYER_1:")
    ctx = rac.get_regime_context_for_prompt(
        "Bull", {"Bull": 0.78, "Bear": 0.07, "Sideways": 0.15}, 12, "PLAYER_1"
    )
    for line in ctx.split("\n"):
        print(f"    {line}")

    print(f"\n{'=' * 70}")
    print("  All demos complete.")
    print(f"{'=' * 70}")
