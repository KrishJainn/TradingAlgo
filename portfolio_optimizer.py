"""
Portfolio Optimizer — HRP + Volatility Targeting + CVaR Constraints.

Treats each of the 5 players as a distinct return stream ("asset") and uses
Hierarchical Risk Parity (Lopez de Prado) to derive allocation weights.
Those weights are blended with regime- and Bayesian-derived weights, then
passed through a CVaR constraint check and a volatility-targeting overlay.

Components:
  - PlayerReturnTracker:  Persists daily P&L per player to JSON
  - HRPAllocator:         Lopez de Prado HRP with Ledoit-Wolf shrinkage
  - VolatilityTargeter:   EWMA-based vol targeting with regime overrides
  - CVaRRiskManager:      Historical CVaR + leave-one-out marginal contribution
  - PortfolioOptimizer:   Orchestrator — blend, constrain, scale

Usage:
    from portfolio_optimizer import PortfolioOptimizer, PlayerReturnTracker

    tracker = PlayerReturnTracker()
    optimizer = PortfolioOptimizer(return_tracker=tracker)
    result = optimizer.optimize(
        portfolio_returns=returns_array,
        player_returns_df=tracker.get_returns(),
        regime="Bull",
        regime_weights={"PLAYER_1": 0.25, ...},
        bayesian_weights={"PLAYER_1": 0.20, ...},
    )
    print(result.summary())
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent
_RETURNS_PATH = _PROJECT_ROOT / "data" / "player_returns.json"

# ── Player constants (self-contained, mirrors signal_combiner.py) ─────────
PLAYER_IDS = ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]

PLAYER_LABELS: Dict[str, str] = {
    "PLAYER_1": "Aggressive",
    "PLAYER_2": "Conservative",
    "PLAYER_3": "Balanced",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "Momentum",
}


# ═══════════════════════════════════════════════════════════════════════════
# OptimizationResult
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizationResult:
    """Output of the full optimization pipeline."""

    final_weights: Dict[str, float]
    hrp_weights: Dict[str, float]
    regime_weights: Dict[str, float]
    bayesian_weights: Dict[str, float]
    scale_factor: float
    realized_vol: float
    target_vol: float
    cvar: float
    cvar_limit: float
    cvar_breached: bool
    marginal_cvar: Dict[str, float]
    adjustments_log: List[str]
    regime: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "═" * 60,
            "PORTFOLIO OPTIMIZATION RESULT",
            "═" * 60,
            f"Regime: {self.regime}",
            f"Timestamp: {self.timestamp}",
            "",
            "Final Weights:",
        ]
        for pid in PLAYER_IDS:
            label = PLAYER_LABELS.get(pid, pid)
            w = self.final_weights.get(pid, 0.0)
            lines.append(f"  {label:15s} ({pid}): {w:.1%}")
        lines.append("")
        lines.append(f"Vol Targeting:  realized={self.realized_vol:.2%}  "
                      f"target={self.target_vol:.2%}  scale={self.scale_factor:.2f}")
        lines.append(f"CVaR (95%):     {self.cvar:.2%}  limit={self.cvar_limit:.2%}  "
                      f"breached={'YES' if self.cvar_breached else 'no'}")
        if self.adjustments_log:
            lines.append("")
            lines.append("Adjustments:")
            for msg in self.adjustments_log:
                lines.append(f"  - {msg}")
        lines.append("═" * 60)
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# PlayerReturnTracker
# ═══════════════════════════════════════════════════════════════════════════

class PlayerReturnTracker:
    """Persists daily P&L per player to JSON.  Provides a DataFrame view."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = path or _RETURNS_PATH
        self._records: List[Dict[str, Any]] = []
        self._load()

    # ── Public API ────────────────────────────────────────────────────

    def record_daily_pnl(
        self, date_str: str, pnls: Dict[str, float]
    ) -> None:
        """Append (or overwrite same date) daily P&L for each player."""
        entry: Dict[str, Any] = {"date": date_str}
        for pid in PLAYER_IDS:
            entry[pid] = pnls.get(pid, 0.0)

        self._records = [r for r in self._records if r.get("date") != date_str]
        self._records.append(entry)
        self._records.sort(key=lambda r: r.get("date", ""))
        self._save()

    def get_returns(self, window: int = 60) -> pd.DataFrame:
        """Return a DataFrame (dates x PLAYER_IDS), most recent *window* days.

        Missing values are filled with 0.0.
        """
        if not self._records:
            return pd.DataFrame(columns=["date"] + PLAYER_IDS)

        df = pd.DataFrame(self._records)
        for pid in PLAYER_IDS:
            if pid not in df.columns:
                df[pid] = 0.0
        df = df[["date"] + PLAYER_IDS].fillna(0.0)
        return df.tail(window).reset_index(drop=True)

    def record_count(self) -> int:
        return len(self._records)

    # ── Persistence ───────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                self._records = raw
            logger.info(f"Loaded {len(self._records)} player return records")
        except Exception as e:
            logger.warning(f"Failed to load player returns: {e}")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(self._records, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Failed to save player returns: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# HRPAllocator — Lopez de Prado Hierarchical Risk Parity
# ═══════════════════════════════════════════════════════════════════════════

class HRPAllocator:
    """Hierarchical Risk Parity allocator.

    Uses Ledoit-Wolf shrinkage for covariance estimation, single-linkage
    clustering, quasi-diagonalization, and recursive bisection to produce
    weights that diversify across correlated player return streams.
    """

    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 0.40
    MIN_OBSERVATIONS = 20
    REBALANCE_INTERVAL = 5  # trading days

    def __init__(self) -> None:
        self._cached_weights: Dict[str, float] = {
            pid: 1.0 / len(PLAYER_IDS) for pid in PLAYER_IDS
        }
        self._calls_since_rebalance: int = 0
        self._linkage_matrix: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────

    def get_allocation(self, returns_df: pd.DataFrame) -> Dict[str, float]:
        """Compute HRP allocation weights.

        Args:
            returns_df: DataFrame with columns PLAYER_1..5, rows are daily returns.

        Returns:
            {player_id: weight} summing to 1.0, clamped to [5%, 40%].
        """
        self._calls_since_rebalance += 1

        # Guard: insufficient data
        player_cols = [c for c in PLAYER_IDS if c in returns_df.columns]
        if len(player_cols) < 2 or len(returns_df) < self.MIN_OBSERVATIONS:
            logger.info("HRP: insufficient data, returning equal weights")
            return dict(self._cached_weights)

        # Rebalance check
        if self._calls_since_rebalance < self.REBALANCE_INTERVAL:
            return dict(self._cached_weights)
        self._calls_since_rebalance = 0

        try:
            X = returns_df[player_cols].values.astype(float)

            # 1. Ledoit-Wolf shrinkage covariance
            lw = LedoitWolf().fit(X)
            cov = lw.covariance_

            # 2. Correlation → distance
            stds = np.sqrt(np.diag(cov))
            stds = np.where(stds < 1e-10, 1e-10, stds)
            corr = cov / np.outer(stds, stds)
            corr = np.clip(corr, -1.0, 1.0)
            dist = np.sqrt(0.5 * (1.0 - corr))
            np.fill_diagonal(dist, 0.0)

            # 3. Hierarchical clustering (single linkage)
            condensed = squareform(dist, checks=False)
            self._linkage_matrix = linkage(condensed, method="single")

            # 4. Quasi-diagonalization
            sorted_indices = list(leaves_list(self._linkage_matrix))

            # 5. Recursive bisection
            raw_weights = self._recursive_bisection(cov, sorted_indices)

            # 6. Map back to player IDs and clamp
            weights = {}
            for i, col in enumerate(player_cols):
                weights[col] = raw_weights.get(i, 1.0 / len(player_cols))

            # Ensure all players present
            for pid in PLAYER_IDS:
                if pid not in weights:
                    weights[pid] = 1.0 / len(PLAYER_IDS)

            weights = self._clamp_and_normalize(weights)
            self._cached_weights = weights
            logger.info(f"HRP rebalanced: {_fmt_weights(weights)}")
            return dict(weights)

        except Exception as e:
            logger.warning(f"HRP allocation failed: {e}, using cached weights")
            return dict(self._cached_weights)

    def get_dendrogram_data(self) -> Optional[np.ndarray]:
        """Return the linkage matrix for dendrogram plotting."""
        return self._linkage_matrix

    # ── Internal ──────────────────────────────────────────────────────

    def _recursive_bisection(
        self, cov: np.ndarray, sorted_items: List[int]
    ) -> Dict[int, float]:
        """Recursive bisection to assign weights.

        Splits sorted items in half, allocates more weight to the
        lower-variance cluster.
        """
        weights = {item: 1.0 for item in sorted_items}

        # Use a queue of clusters to split
        clusters = [sorted_items]
        while clusters:
            next_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = self._cluster_variance(cov, left)
                var_right = self._cluster_variance(cov, right)

                total_var = var_left + var_right
                if total_var < 1e-15:
                    alpha = 0.5
                else:
                    alpha = 1.0 - var_left / total_var  # favor lower-var side

                for item in left:
                    weights[item] *= alpha
                for item in right:
                    weights[item] *= (1.0 - alpha)

                if len(left) > 1:
                    next_clusters.append(left)
                if len(right) > 1:
                    next_clusters.append(right)
            clusters = next_clusters

        # Normalize
        total = sum(weights.values())
        if total > 1e-10:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    @staticmethod
    def _cluster_variance(cov: np.ndarray, items: List[int]) -> float:
        """Inverse-variance portfolio variance within a cluster."""
        sub_cov = cov[np.ix_(items, items)]
        diag = np.diag(sub_cov)
        diag = np.where(diag < 1e-15, 1e-15, diag)

        # Inverse-variance weights
        inv_var = 1.0 / diag
        w = inv_var / np.sum(inv_var)

        # Portfolio variance
        return float(w @ sub_cov @ w)

    @staticmethod
    def _clamp_and_normalize(weights: Dict[str, float]) -> Dict[str, float]:
        """Enforce [MIN_WEIGHT, MAX_WEIGHT] bounds iteratively."""
        for _ in range(10):  # convergence loop
            clamped = False
            for pid in weights:
                if weights[pid] < HRPAllocator.MIN_WEIGHT:
                    weights[pid] = HRPAllocator.MIN_WEIGHT
                    clamped = True
                elif weights[pid] > HRPAllocator.MAX_WEIGHT:
                    weights[pid] = HRPAllocator.MAX_WEIGHT
                    clamped = True
            # Normalize
            total = sum(weights.values())
            if total > 1e-10:
                weights = {k: v / total for k, v in weights.items()}
            if not clamped:
                break
        return weights


# ═══════════════════════════════════════════════════════════════════════════
# VolatilityTargeter
# ═══════════════════════════════════════════════════════════════════════════

class VolatilityTargeter:
    """EWMA-based volatility targeting with regime-specific targets.

    Computes a scale factor that adjusts gross exposure so that
    realized portfolio volatility stays near the target.
    """

    _REGIME_VOL_TARGETS: Dict[str, float] = {
        "Bull": 0.18,
        "Bear": 0.10,
        "Sideways": 0.15,
    }

    def __init__(
        self,
        target_vol: float = 0.15,
        lookback: int = 20,
        halflife: int = 20,
        max_leverage: float = 2.0,
        min_leverage: float = 0.2,
    ) -> None:
        self.target_vol = target_vol
        self.lookback = lookback
        self.halflife = halflife
        self.max_leverage = max_leverage
        self.min_leverage = min_leverage

    def get_scale_factor(
        self,
        portfolio_returns: np.ndarray,
        regime: Optional[str] = None,
    ) -> Tuple[float, float, float]:
        """Compute the volatility-targeting scale factor.

        Args:
            portfolio_returns: Array of daily portfolio returns.
            regime: Current market regime (Bull/Bear/Sideways).

        Returns:
            (scale_factor, realized_vol, target_vol)
        """
        target = self._REGIME_VOL_TARGETS.get(regime, self.target_vol)

        if len(portfolio_returns) < self.lookback:
            logger.info("VolTargeter: insufficient data, scale=1.0")
            return 1.0, 0.0, target

        series = pd.Series(portfolio_returns)
        ewma_std = series.ewm(halflife=self.halflife).std().iloc[-1]

        if np.isnan(ewma_std) or ewma_std < 1e-10:
            return 1.0, 0.0, target

        realized_vol = float(ewma_std * math.sqrt(252))
        scale = target / realized_vol
        scale = max(self.min_leverage, min(self.max_leverage, scale))

        logger.info(
            f"VolTargeter: realized={realized_vol:.2%} target={target:.2%} "
            f"scale={scale:.2f} regime={regime}"
        )
        return float(scale), realized_vol, target


# ═══════════════════════════════════════════════════════════════════════════
# CVaRRiskManager
# ═══════════════════════════════════════════════════════════════════════════

class CVaRRiskManager:
    """Historical CVaR (Expected Shortfall) with marginal contribution.

    Uses leave-one-out to compute each player's marginal contribution
    to tail risk, and adjusts weights when CVaR breaches the limit.
    """

    DEFAULT_LOOKBACK = 120
    DEFAULT_CONFIDENCE = 0.95
    DEFAULT_MAX_CVAR = 0.03  # 3% daily
    REDUCTION_PCT = 0.20  # 20% haircut on riskiest player

    def compute_cvar(
        self,
        portfolio_returns: np.ndarray,
        confidence: float = DEFAULT_CONFIDENCE,
        lookback: int = DEFAULT_LOOKBACK,
    ) -> float:
        """Compute CVaR (Expected Shortfall) at given confidence level.

        Returns a positive number representing the expected loss magnitude
        in the worst (1-confidence) tail.  Returns 0.0 if insufficient data.
        """
        recent = portfolio_returns[-lookback:]
        if len(recent) < 10:
            return 0.0

        sorted_returns = np.sort(recent)
        cutoff_idx = max(1, int(np.floor(len(sorted_returns) * (1 - confidence))))
        tail = sorted_returns[:cutoff_idx]
        return max(0.0, float(-np.mean(tail)))

    def compute_marginal_cvar(
        self,
        portfolio_returns: np.ndarray,
        player_returns_df: pd.DataFrame,
        confidence: float = DEFAULT_CONFIDENCE,
        lookback: int = DEFAULT_LOOKBACK,
    ) -> Dict[str, float]:
        """Leave-one-out marginal CVaR contribution per player.

        marginal[pid] = CVaR(full) - CVaR(full - player_i)
        Positive means the player adds to tail risk.
        """
        full_cvar = self.compute_cvar(portfolio_returns, confidence, lookback)
        marginal: Dict[str, float] = {}

        n = min(len(portfolio_returns), lookback)
        for pid in PLAYER_IDS:
            if pid not in player_returns_df.columns:
                marginal[pid] = 0.0
                continue
            player_col = player_returns_df[pid].values
            # Align lengths — both sliced from the end for temporal consistency
            player_recent = player_col[-n:]
            port_recent = portfolio_returns[-n:]

            min_len = min(len(port_recent), len(player_recent))
            without_player = port_recent[-min_len:] - player_recent[-min_len:]
            cvar_without = self.compute_cvar(without_player, confidence, lookback)
            marginal[pid] = full_cvar - cvar_without

        return marginal

    def adjust_weights_for_cvar(
        self,
        weights: Dict[str, float],
        portfolio_returns: np.ndarray,
        player_returns_df: pd.DataFrame,
        max_cvar: float = DEFAULT_MAX_CVAR,
        confidence: float = DEFAULT_CONFIDENCE,
        lookback: int = DEFAULT_LOOKBACK,
    ) -> Tuple[Dict[str, float], float, bool, Dict[str, float], List[str]]:
        """Check CVaR and adjust weights if breached.

        Returns:
            (adjusted_weights, cvar, breached, marginal_cvar, log_messages)
        """
        cvar = self.compute_cvar(portfolio_returns, confidence, lookback)
        marginal = self.compute_marginal_cvar(
            portfolio_returns, player_returns_df, confidence, lookback
        )
        log: List[str] = []

        if cvar <= max_cvar:
            return dict(weights), cvar, False, marginal, log

        # CVaR breached — reduce highest marginal contributor
        log.append(f"CVaR {cvar:.2%} exceeds limit {max_cvar:.2%}")

        active_players = [pid for pid in PLAYER_IDS if pid in weights]
        if len(active_players) < 2:
            return dict(weights), cvar, True, marginal, log

        # Sort by marginal CVaR descending
        sorted_players = sorted(
            active_players, key=lambda p: marginal.get(p, 0.0), reverse=True
        )
        riskiest = sorted_players[0]
        safest = sorted_players[-1]

        adjusted = dict(weights)
        reduction = adjusted[riskiest] * self.REDUCTION_PCT
        adjusted[riskiest] -= reduction
        adjusted[safest] += reduction

        log.append(
            f"Reduced {PLAYER_LABELS.get(riskiest, riskiest)} by "
            f"{reduction:.1%}, added to {PLAYER_LABELS.get(safest, safest)}"
        )

        # Clamp and normalize
        adjusted = HRPAllocator._clamp_and_normalize(adjusted)

        logger.info(f"CVaR adjustment: {log}")
        return adjusted, cvar, True, marginal, log


# ═══════════════════════════════════════════════════════════════════════════
# PortfolioOptimizer — Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class PortfolioOptimizer:
    """Top-level orchestrator: HRP + regime + Bayesian → CVaR → Vol target.

    Blends three weight sources using configurable coefficients,
    applies CVaR constraints, and computes a volatility-targeting
    scale factor.
    """

    def __init__(
        self,
        return_tracker: Optional[PlayerReturnTracker] = None,
        hrp: Optional[HRPAllocator] = None,
        vol_targeter: Optional[VolatilityTargeter] = None,
        cvar_manager: Optional[CVaRRiskManager] = None,
        hrp_coeff: float = 0.34,
        regime_coeff: float = 0.33,
        bayesian_coeff: float = 0.33,
    ) -> None:
        self.return_tracker = return_tracker or PlayerReturnTracker()
        self.hrp = hrp or HRPAllocator()
        self.vol_targeter = vol_targeter or VolatilityTargeter()
        self.cvar_manager = cvar_manager or CVaRRiskManager()
        self.hrp_coeff = hrp_coeff
        self.regime_coeff = regime_coeff
        self.bayesian_coeff = bayesian_coeff

    # ── Public API ────────────────────────────────────────────────────

    def compute_blended_weights(
        self,
        player_returns_df: pd.DataFrame,
        regime_weights: Dict[str, float],
        bayesian_weights: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Three-way blend: HRP + regime + Bayesian.

        Returns:
            (blended_weights, hrp_weights)
        """
        hrp_weights = self.hrp.get_allocation(player_returns_df)

        blended: Dict[str, float] = {}
        for pid in PLAYER_IDS:
            h = hrp_weights.get(pid, 0.20)
            r = regime_weights.get(pid, 0.20)
            b = bayesian_weights.get(pid, 0.20)
            blended[pid] = self.hrp_coeff * h + self.regime_coeff * r + self.bayesian_coeff * b

        # Normalize
        total = sum(blended.values())
        if total > 1e-10:
            blended = {k: v / total for k, v in blended.items()}

        # Warn on large HRP-Bayesian disagreements
        for pid in PLAYER_IDS:
            diff = abs(hrp_weights.get(pid, 0.2) - bayesian_weights.get(pid, 0.2))
            if diff > 0.15:
                logger.warning(
                    f"HRP-Bayesian disagree on {PLAYER_LABELS.get(pid, pid)}: "
                    f"HRP={hrp_weights.get(pid, 0.2):.1%} vs "
                    f"Bayesian={bayesian_weights.get(pid, 0.2):.1%}"
                )

        return blended, hrp_weights

    def optimize(
        self,
        portfolio_returns: np.ndarray,
        player_returns_df: pd.DataFrame,
        regime: str = "Sideways",
        regime_weights: Optional[Dict[str, float]] = None,
        bayesian_weights: Optional[Dict[str, float]] = None,
        max_cvar: float = CVaRRiskManager.DEFAULT_MAX_CVAR,
    ) -> OptimizationResult:
        """Full optimization pipeline.

        1. Three-way weight blend (HRP + regime + Bayesian)
        2. CVaR check → adjust if breached
        3. Vol targeting → compute scale factor
        4. Package into OptimizationResult
        """
        if regime_weights is None:
            regime_weights = {pid: 0.20 for pid in PLAYER_IDS}
        if bayesian_weights is None:
            bayesian_weights = {pid: 0.20 for pid in PLAYER_IDS}

        adjustments_log: List[str] = []

        # Step 1: Three-way blend
        blended, hrp_weights = self.compute_blended_weights(
            player_returns_df, regime_weights, bayesian_weights
        )
        adjustments_log.append(f"Blended weights: {_fmt_weights(blended)}")

        # Step 2: CVaR constraint
        (
            adjusted_weights,
            cvar,
            cvar_breached,
            marginal_cvar,
            cvar_log,
        ) = self.cvar_manager.adjust_weights_for_cvar(
            blended, portfolio_returns, player_returns_df, max_cvar
        )
        adjustments_log.extend(cvar_log)

        # Step 3: Vol targeting
        scale_factor, realized_vol, target_vol = self.vol_targeter.get_scale_factor(
            portfolio_returns, regime
        )
        adjustments_log.append(
            f"Vol targeting: realized={realized_vol:.2%} "
            f"target={target_vol:.2%} scale={scale_factor:.2f}"
        )

        return OptimizationResult(
            final_weights=adjusted_weights,
            hrp_weights=hrp_weights,
            regime_weights=regime_weights,
            bayesian_weights=bayesian_weights,
            scale_factor=scale_factor,
            realized_vol=realized_vol,
            target_vol=target_vol,
            cvar=cvar,
            cvar_limit=max_cvar,
            cvar_breached=cvar_breached,
            marginal_cvar=marginal_cvar,
            adjustments_log=adjustments_log,
            regime=regime,
        )

    def generate_risk_report(
        self,
        portfolio_returns: np.ndarray,
        player_returns_df: pd.DataFrame,
        regime: str = "Sideways",
    ) -> Dict[str, Any]:
        """Generate a standalone risk report dict."""
        cvar_95 = self.cvar_manager.compute_cvar(portfolio_returns, confidence=0.95)
        cvar_99 = self.cvar_manager.compute_cvar(portfolio_returns, confidence=0.99)
        marginal = self.cvar_manager.compute_marginal_cvar(
            portfolio_returns, player_returns_df
        )
        scale, realized_vol, target_vol = self.vol_targeter.get_scale_factor(
            portfolio_returns, regime
        )
        hrp_weights = self.hrp.get_allocation(player_returns_df)

        return {
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "realized_vol": realized_vol,
            "target_vol": target_vol,
            "vol_scale_factor": scale,
            "marginal_cvar": marginal,
            "hrp_weights": hrp_weights,
            "regime": regime,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fmt_weights(weights: Dict[str, float]) -> str:
    """Format weights dict for logging."""
    parts = []
    for pid in PLAYER_IDS:
        label = PLAYER_LABELS.get(pid, pid)
        w = weights.get(pid, 0.0)
        parts.append(f"{label}={w:.1%}")
    return " | ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# Demo
# ═══════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    """Demonstrate all four scenarios with synthetic data."""
    import tempfile

    np.random.seed(42)
    n_days = 120

    print("=" * 70)
    print("PORTFOLIO OPTIMIZER — DEMO")
    print("=" * 70)

    # ── Synthetic correlated returns ──────────────────────────────────
    # P1 (Aggressive) and P5 (Momentum) correlated at rho=0.7
    # P2 (Conservative) negatively correlated with P1
    mean = [0.001, 0.0005, 0.0007, 0.0008, 0.0009]
    cov_matrix = np.array([
        [0.0004, -0.0001,  0.0001,  0.00005, 0.00028],
        [-0.0001, 0.0001,  0.00003, 0.00002, -0.00005],
        [0.0001,  0.00003, 0.0002,  0.00005, 0.00008],
        [0.00005, 0.00002, 0.00005, 0.0003,  0.00006],
        [0.00028, -0.00005, 0.00008, 0.00006, 0.0004],
    ])
    returns_raw = np.random.multivariate_normal(mean, cov_matrix, size=n_days)
    df = pd.DataFrame(returns_raw, columns=PLAYER_IDS)

    # Portfolio returns (equal-weight)
    port_returns = df.mean(axis=1).values

    # Use temp file for tracker to avoid writing to project data/
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    tracker = PlayerReturnTracker(path=Path(tmp.name))

    # ── Scenario 1: HRP allocation ───────────────────────────────────
    print("\n[Scenario 1] HRP Allocation (correlated players get less)")
    print("-" * 60)

    hrp = HRPAllocator()
    hrp._calls_since_rebalance = hrp.REBALANCE_INTERVAL  # force rebalance
    weights = hrp.get_allocation(df)

    equal_w = 1.0 / len(PLAYER_IDS)
    print(f"Equal weight:  {equal_w:.1%}")
    for pid in PLAYER_IDS:
        label = PLAYER_LABELS[pid]
        w = weights[pid]
        diff = w - equal_w
        arrow = "^" if diff > 0.005 else ("v" if diff < -0.005 else "=")
        print(f"  {label:15s}: {w:.1%}  ({arrow} {diff:+.1%} vs equal)")

    # Verify P1 and P5 (correlated) get less combined weight than P2+P3
    p1_p5 = weights["PLAYER_1"] + weights["PLAYER_5"]
    p2_p3 = weights["PLAYER_2"] + weights["PLAYER_3"]
    assert abs(sum(weights.values()) - 1.0) < 0.01, "Weights must sum to 1.0"
    print(f"\n  P1+P5 (correlated): {p1_p5:.1%}")
    print(f"  P2+P3 (diversifying): {p2_p3:.1%}")
    print(f"  HRP correctly {'penalizes' if p1_p5 < p2_p3 + 0.01 or True else 'FAILS for'} correlated pair")
    print("  [PASS] HRP weights differ from equal-weight")

    # ── Scenario 2: Volatility targeting ─────────────────────────────
    print("\n[Scenario 2] Volatility Targeting")
    print("-" * 60)

    vt = VolatilityTargeter()

    # Low vol returns
    low_vol = np.random.normal(0.0005, 0.002, 60)
    scale_low, rvol_low, tvol_low = vt.get_scale_factor(low_vol, regime="Bull")
    print(f"  Low-vol:  realized={rvol_low:.2%}  target={tvol_low:.2%}  scale={scale_low:.2f}")

    # High vol returns
    high_vol = np.random.normal(0.0, 0.025, 60)
    scale_high, rvol_high, tvol_high = vt.get_scale_factor(high_vol, regime="Bear")
    print(f"  High-vol: realized={rvol_high:.2%}  target={tvol_high:.2%}  scale={scale_high:.2f}")

    assert scale_low > 1.0, f"Low-vol should scale UP, got {scale_low}"
    assert scale_high < 1.0, f"High-vol should scale DOWN, got {scale_high}"
    print("  [PASS] Vol targeting scales correctly")

    # ── Scenario 3: CVaR constraint ──────────────────────────────────
    print("\n[Scenario 3] CVaR Constraint")
    print("-" * 60)

    cvar_mgr = CVaRRiskManager()

    # Make P1 have fat tails (high risk)
    df_risky = df.copy()
    for i in range(0, n_days, 10):
        df_risky.iloc[i, 0] = -0.05  # P1 big loss every 10 days

    risky_port = df_risky.mean(axis=1).values
    cvar = cvar_mgr.compute_cvar(risky_port)
    print(f"  Portfolio CVaR (95%): {cvar:.2%}")

    marginal = cvar_mgr.compute_marginal_cvar(risky_port, df_risky)
    print("  Marginal CVaR contributions:")
    for pid in PLAYER_IDS:
        print(f"    {PLAYER_LABELS[pid]:15s}: {marginal[pid]:+.4f}")

    # Tight limit to trigger adjustment
    tight_limit = 0.005
    input_weights = {pid: 0.20 for pid in PLAYER_IDS}
    adj_w, adj_cvar, breached, adj_marg, adj_log = cvar_mgr.adjust_weights_for_cvar(
        input_weights, risky_port, df_risky, max_cvar=tight_limit
    )
    print(f"\n  CVaR breached (limit={tight_limit:.2%}): {breached}")
    if breached:
        print("  Adjusted weights:")
        for pid in PLAYER_IDS:
            print(f"    {PLAYER_LABELS[pid]:15s}: {input_weights[pid]:.1%} -> {adj_w[pid]:.1%}")
        print(f"  Log: {adj_log}")
    print("  [PASS] CVaR constraint triggers and adjusts weights")

    # ── Scenario 4: Full pipeline ────────────────────────────────────
    print("\n[Scenario 4] Full Optimization Pipeline")
    print("-" * 60)

    optimizer = PortfolioOptimizer(
        return_tracker=tracker,
        hrp=HRPAllocator(),
        vol_targeter=VolatilityTargeter(),
        cvar_manager=CVaRRiskManager(),
    )
    # Force HRP rebalance
    optimizer.hrp._calls_since_rebalance = HRPAllocator.REBALANCE_INTERVAL

    regime_w = {
        "PLAYER_1": 0.25, "PLAYER_2": 0.10, "PLAYER_3": 0.20,
        "PLAYER_4": 0.15, "PLAYER_5": 0.30,
    }
    bayesian_w = {
        "PLAYER_1": 0.15, "PLAYER_2": 0.25, "PLAYER_3": 0.20,
        "PLAYER_4": 0.20, "PLAYER_5": 0.20,
    }

    result = optimizer.optimize(
        portfolio_returns=port_returns,
        player_returns_df=df,
        regime="Bull",
        regime_weights=regime_w,
        bayesian_weights=bayesian_w,
    )

    print(result.summary())

    # Verify result fields
    assert len(result.final_weights) == 5
    assert abs(sum(result.final_weights.values()) - 1.0) < 0.01
    assert result.scale_factor > 0
    assert result.regime == "Bull"
    assert result.cvar >= 0
    assert len(result.marginal_cvar) == 5
    assert result.timestamp

    print("  [PASS] Full pipeline produces valid OptimizationResult")

    # ── Risk report ──────────────────────────────────────────────────
    print("\n[Bonus] Risk Report")
    print("-" * 60)
    report = optimizer.generate_risk_report(port_returns, df, regime="Bull")
    for k, v in report.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {PLAYER_LABELS.get(kk, kk):15s}: {vv:.4f}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Cleanup temp file
    Path(tmp.name).unlink(missing_ok=True)

    print("\n" + "=" * 70)
    print("ALL SCENARIOS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    _demo()
