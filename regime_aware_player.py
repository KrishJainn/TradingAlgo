"""
Regime-Aware Player Layer for the 5-Player Coach Trading System.

Wraps each player's config with regime-conditional adjustments:
  - Position sizing multipliers per regime
  - Indicator weight adjustments per regime
  - Entry/exit threshold tightening or loosening per regime
  - Per-player, per-regime performance tracking (Sharpe, hit-rate, avg P&L)

This module does NOT replace the existing PlayerState or BacktestEngine logic.
Instead it provides a lightweight adapter that the coach (or backtest runner)
calls *before* executing a player's trades for a given day.

Usage:
    from regime_aware_player import RegimeAwarePlayer, REGIME_PLAYER_REGISTRY

    rap = RegimeAwarePlayer.from_registry("PLAYER_1")  # Aggressive
    adjusted_config = rap.adjust_config(base_config, regime="Bear")
    rap.record_result(regime="Bear", pnl=1200.0, sharpe=1.8, win_rate=0.48)
    stats = rap.get_regime_stats("Bear")
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Player-level regime config
# ---------------------------------------------------------------------------

# Player ID → label mapping (redesigned for forced diversity)
PLAYER_LABELS: Dict[str, str] = {
    "PLAYER_1": "TrendFollower",
    "PLAYER_2": "MeanReversion",
    "PLAYER_3": "VolumeFlow",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "MultiMomentum",
}

# Position-sizing multipliers applied to the base position value.
# > 1.0 means larger positions, < 1.0 means smaller.
_SIZING_MULTIPLIERS: Dict[str, Dict[str, float]] = {
    "TrendFollower": {
        "Bull":     1.5,     # full weight in bull — trend-following shines
        "Bear":     0.2,     # minimal in bear
        "Sideways": 0.5,     # half weight — trends are weak
    },
    "MeanReversion": {
        "Bull":     0.3,     # small — mean-reversion fights trends
        "Bear":     0.3,     # small — reversals are dangerous in bear
        "Sideways": 1.2,     # full weight — sideways is MR territory
    },
    "VolumeFlow": {
        "Bull":     1.0,     # volume signals are regime-agnostic
        "Bear":     1.0,
        "Sideways": 1.0,
    },
    "VolBreakout": {
        "Bull":     0.8,     # good but trends already running
        "Bear":     0.5,     # catch vol expansion
        "Sideways": 1.3,     # best for breakout from range
    },
    "MultiMomentum": {
        "Bull":     1.5,     # full weight — ride momentum
        "Bear":     0.1,     # near-zero — momentum is bearish
        "Sideways": 0.5,     # half weight
    },
}

# Entry-threshold adjustment (additive delta).
# Positive = more selective (raise threshold), negative = looser (lower it).
_ENTRY_THRESHOLD_DELTA: Dict[str, Dict[str, float]] = {
    "TrendFollower": {
        "Bull":     -0.05,   # looser in bull — ride the trend
        "Bear":     +0.10,   # very selective in bear
        "Sideways": +0.05,   # tighter — fewer good trends
    },
    "MeanReversion": {
        "Bull":     +0.08,   # very selective — MR struggles in trends
        "Bear":     +0.05,
        "Sideways": -0.05,   # looser — this is MR's regime
    },
    "VolumeFlow": {
        "Bull":      0.00,
        "Bear":     +0.03,
        "Sideways":  0.00,
    },
    "VolBreakout": {
        "Bull":     +0.03,   # tighter — false breakouts in calm trends
        "Bear":     -0.02,   # looser — vol expansion = opportunity
        "Sideways": -0.03,   # looser — breakouts from range
    },
    "MultiMomentum": {
        "Bull":     -0.05,   # ride all momentum
        "Bear":     +0.10,   # very selective
        "Sideways": +0.05,
    },
}

# Exit-threshold adjustment (additive delta).
# More negative = tighter stop (exit sooner), positive = looser.
_EXIT_THRESHOLD_DELTA: Dict[str, Dict[str, float]] = {
    "TrendFollower": {
        "Bull":     +0.03,   # let winners run
        "Bear":     -0.05,   # cut losers fast
        "Sideways": -0.02,
    },
    "MeanReversion": {
        "Bull":     -0.03,   # quick exits — MR trades are short
        "Bear":     -0.03,
        "Sideways": +0.02,   # let reversions play out
    },
    "VolumeFlow": {
        "Bull":      0.00,
        "Bear":     -0.02,
        "Sideways":  0.00,
    },
    "VolBreakout": {
        "Bull":     -0.02,   # quick profits in calm markets
        "Bear":     +0.02,   # give vol trades room
        "Sideways":  0.00,
    },
    "MultiMomentum": {
        "Bull":     +0.03,   # let trends breathe
        "Bear":     -0.05,   # strict cut
        "Sideways": -0.02,
    },
}


# ---------------------------------------------------------------------------
# Per-regime performance tracker
# ---------------------------------------------------------------------------

@dataclass
class RegimePerformanceRecord:
    """Accumulated stats for a single (player, regime) pair."""
    pnl_history: List[float] = field(default_factory=list)
    sharpe_history: List[float] = field(default_factory=list)
    win_rate_history: List[float] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.pnl_history)

    @property
    def avg_pnl(self) -> float:
        return float(np.mean(self.pnl_history)) if self.pnl_history else 0.0

    @property
    def avg_sharpe(self) -> float:
        return float(np.mean(self.sharpe_history)) if self.sharpe_history else 0.0

    @property
    def avg_win_rate(self) -> float:
        return float(np.mean(self.win_rate_history)) if self.win_rate_history else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "avg_pnl": round(self.avg_pnl, 2),
            "avg_sharpe": round(self.avg_sharpe, 4),
            "avg_win_rate": round(self.avg_win_rate, 4),
            "last_pnl": round(self.pnl_history[-1], 2) if self.pnl_history else 0.0,
            "last_sharpe": round(self.sharpe_history[-1], 4) if self.sharpe_history else 0.0,
        }


# ---------------------------------------------------------------------------
# RegimeAwarePlayer
# ---------------------------------------------------------------------------

class RegimeAwarePlayer:
    """
    Adapter that applies regime-conditional adjustments to a player's config.

    It does NOT own or mutate the player's state — it only computes a
    *derived* config dict that the caller can use for the current bar or day.

    Typical flow:
        rap = RegimeAwarePlayer("PLAYER_1", "Aggressive")
        adj_cfg = rap.adjust_config(player_state.config, regime="Bull")
        # use adj_cfg for signal generation / position sizing
    """

    def __init__(
        self,
        player_id: str,
        player_label: str,
        *,
        indicator_adjustments: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Args:
            player_id:   e.g. "PLAYER_1"
            player_label: e.g. "Aggressive"
            indicator_adjustments: Optional per-regime indicator weight multipliers.
                If None, pulled from regime_detector._INDICATOR_ADJUSTMENTS at call time.
        """
        self.player_id = player_id
        self.player_label = player_label
        self._indicator_adjustments = indicator_adjustments  # lazy-loaded

        # Per-regime performance tracking
        self._regime_stats: Dict[str, RegimePerformanceRecord] = {}

    # ----- factory -----
    @classmethod
    def from_registry(cls, player_id: str, **kwargs) -> "RegimeAwarePlayer":
        """Create from the global PLAYER_LABELS registry."""
        label = PLAYER_LABELS.get(player_id)
        if label is None:
            raise ValueError(f"Unknown player_id: {player_id}. Expected one of {list(PLAYER_LABELS)}")
        return cls(player_id, label, **kwargs)

    # ----- core API -----

    def adjust_config(
        self,
        base_config: Dict,
        regime: str,
        *,
        apply_sizing: bool = True,
        apply_indicator_weights: bool = True,
        apply_thresholds: bool = True,
    ) -> Dict:
        """
        Return a copy of *base_config* with regime-conditional adjustments.

        The returned config has the same shape as the original:
            {"label": str, "weights": {...}, "entry_threshold": float,
             "exit_threshold": float, "min_hold_bars": int}

        Additional key added:
            "position_sizing_multiplier": float  (for position-sizing code)

        Args:
            base_config: The player's current config dict.
            regime:      "Bull", "Bear", or "Sideways".
            apply_sizing: Whether to include a position_sizing_multiplier key.
            apply_indicator_weights: Whether to multiply indicator weights by
                regime-specific adjustments.
            apply_thresholds: Whether to adjust entry/exit thresholds.

        Returns:
            A deep copy with adjustments applied.
        """
        cfg = deepcopy(base_config)

        # Normalise regime label to title-case
        regime = regime.strip().title() if regime else "Sideways"
        if regime not in ("Bull", "Bear", "Sideways"):
            regime = "Sideways"

        # 1. Position sizing multiplier
        if apply_sizing:
            sizing_table = _SIZING_MULTIPLIERS.get(self.player_label, {})
            cfg["position_sizing_multiplier"] = sizing_table.get(regime, 1.0)

        # 2. Indicator weight adjustments
        if apply_indicator_weights:
            adjustments = self._get_indicator_adjustments(regime)
            weights = cfg.get("weights", {})
            adjusted_weights = {}
            for ind, w in weights.items():
                mult = adjustments.get(ind, 1.0)
                adjusted_weights[ind] = round(max(0.05, min(1.0, w * mult)), 4)
            cfg["weights"] = adjusted_weights

        # 3. Threshold adjustments
        if apply_thresholds:
            entry_delta = _ENTRY_THRESHOLD_DELTA.get(self.player_label, {}).get(regime, 0.0)
            exit_delta = _EXIT_THRESHOLD_DELTA.get(self.player_label, {}).get(regime, 0.0)

            base_entry = cfg.get("entry_threshold", 0.25)
            base_exit = cfg.get("exit_threshold", -0.10)

            cfg["entry_threshold"] = round(max(0.10, min(0.50, base_entry + entry_delta)), 4)
            cfg["exit_threshold"] = round(max(-0.30, min(-0.03, base_exit + exit_delta)), 4)

        # Tag with regime so downstream code knows which regime produced this config
        cfg["_regime"] = regime

        return cfg

    def get_sizing_multiplier(self, regime: str) -> float:
        """Return position-sizing multiplier for the given regime."""
        regime = regime.strip().title() if regime else "Sideways"
        table = _SIZING_MULTIPLIERS.get(self.player_label, {})
        return table.get(regime, 1.0)

    # ----- performance tracking -----

    def record_result(
        self,
        regime: str,
        pnl: float,
        sharpe: float = 0.0,
        win_rate: float = 0.0,
    ) -> None:
        """Record a run's result for the (player, regime) pair."""
        regime = regime.strip().title() if regime else "Sideways"
        if regime not in self._regime_stats:
            self._regime_stats[regime] = RegimePerformanceRecord()
        rec = self._regime_stats[regime]
        rec.pnl_history.append(pnl)
        rec.sharpe_history.append(sharpe)
        rec.win_rate_history.append(win_rate)

    def get_regime_stats(self, regime: str) -> Dict[str, Any]:
        """Return accumulated performance stats for one regime."""
        regime = regime.strip().title() if regime else "Sideways"
        rec = self._regime_stats.get(regime)
        if rec is None:
            return {"n_samples": 0, "avg_pnl": 0.0, "avg_sharpe": 0.0, "avg_win_rate": 0.0}
        return rec.to_dict()

    def get_all_regime_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return performance stats for every regime this player has seen."""
        return {r: rec.to_dict() for r, rec in self._regime_stats.items()}

    def get_regime_performance_summary(self) -> str:
        """
        One-line summary suitable for embedding in LLM prompts.

        Example:
            "Bull(3 runs): avgPnL=+1200, Sharpe=1.8 | Bear(2 runs): avgPnL=-400, Sharpe=-0.3"
        """
        parts = []
        for regime in ("Bull", "Bear", "Sideways"):
            rec = self._regime_stats.get(regime)
            if rec and rec.n_samples > 0:
                parts.append(
                    f"{regime}({rec.n_samples}): "
                    f"avgPnL={rec.avg_pnl:+,.0f}, "
                    f"Sharpe={rec.avg_sharpe:.2f}, "
                    f"WR={rec.avg_win_rate:.0%}"
                )
        return " | ".join(parts) if parts else ""

    # ----- internal helpers -----

    def _get_indicator_adjustments(self, regime: str) -> Dict[str, float]:
        """
        Get per-indicator multipliers for *regime*.

        Tries the explicit dict passed at init, then falls back to
        the regime_detector module's _INDICATOR_ADJUSTMENTS table.
        """
        if self._indicator_adjustments is not None:
            return self._indicator_adjustments.get(regime, {})

        # Lazy import: avoids hard dependency on regime_detector at import time
        try:
            from regime_detector import _INDICATOR_ADJUSTMENTS
            return dict(_INDICATOR_ADJUSTMENTS.get(regime, {}))
        except ImportError:
            return {}

    def __repr__(self) -> str:
        return f"RegimeAwarePlayer({self.player_id!r}, {self.player_label!r})"


# ---------------------------------------------------------------------------
# Registry of all 5 players (convenience)
# ---------------------------------------------------------------------------

def build_registry() -> Dict[str, RegimeAwarePlayer]:
    """
    Build and return a dict mapping player_id → RegimeAwarePlayer.

    This is the main entry-point: call once at system startup, then use
    the returned dict throughout the backtest.

        registry = build_registry()
        adj_cfg = registry["PLAYER_1"].adjust_config(cfg, "Bull")
    """
    return {pid: RegimeAwarePlayer(pid, label) for pid, label in PLAYER_LABELS.items()}


# Pre-built constant for convenience
REGIME_PLAYER_REGISTRY: Dict[str, RegimeAwarePlayer] = build_registry()


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("  Regime-Aware Player — Demo")
    print("=" * 70)

    # Demo config (resembles PLAYER_1 Aggressive)
    demo_config = {
        "label": "Aggressive",
        "weights": {
            "RSI_7": 0.90, "STOCH_5_3": 0.85, "TSI_13_25": 0.75,
            "CMO_14": 0.70, "SUPERTREND_7_3": 0.60, "ADX_14": 0.50,
            "BBANDS_20_2": 0.55, "MFI_14": 0.55, "DEMA_20": 0.45,
            "NATR_14": 0.40,
        },
        "entry_threshold": 0.25,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    }

    registry = build_registry()

    for pid, rap in registry.items():
        print(f"\n{'─' * 60}")
        print(f"  {pid} — {rap.player_label}")
        print(f"{'─' * 60}")

        for regime in ("Bull", "Bear", "Sideways"):
            adj = rap.adjust_config(demo_config, regime)
            sizing = adj.get("position_sizing_multiplier", 1.0)
            entry = adj["entry_threshold"]
            exit_t = adj["exit_threshold"]

            # Count how many weights changed
            orig_w = demo_config["weights"]
            changed = sum(1 for k in adj["weights"] if abs(adj["weights"][k] - orig_w.get(k, 0)) > 0.001)

            print(f"  {regime:8s}  sizing={sizing:.1f}x  entry={entry:.3f}  exit={exit_t:.3f}  "
                  f"weights_changed={changed}/{len(orig_w)}")

    # Performance tracking demo
    print(f"\n{'─' * 60}")
    print("  Performance Tracking Demo")
    print(f"{'─' * 60}")

    rap = registry["PLAYER_1"]
    rap.record_result("Bull", pnl=1500, sharpe=2.1, win_rate=0.52)
    rap.record_result("Bull", pnl=1800, sharpe=2.4, win_rate=0.55)
    rap.record_result("Bear", pnl=-400, sharpe=-0.5, win_rate=0.35)
    rap.record_result("Sideways", pnl=200, sharpe=0.3, win_rate=0.42)

    print(f"\n  {rap.player_label} regime stats:")
    for regime in ("Bull", "Bear", "Sideways"):
        stats = rap.get_regime_stats(regime)
        print(f"    {regime:8s}  samples={stats['n_samples']}  "
              f"avgPnL={stats['avg_pnl']:+,.0f}  "
              f"avgSharpe={stats['avg_sharpe']:.2f}  "
              f"avgWR={stats['avg_win_rate']:.0%}")

    print(f"\n  Prompt-ready summary: {rap.get_regime_performance_summary()}")

    print(f"\n{'=' * 70}")
    print("  All demos complete.")
    print(f"{'=' * 70}")
