"""
Hierarchical Risk Parity (HRP) portfolio allocator.

Implements the full HRP pipeline from Lopez de Prado (2016):
    "Building Diversified Portfolios that Outperform Out-of-Sample",
    Journal of Portfolio Management, 42(4), 59-69.

Algorithm overview:
    1. Compute a correlation-based distance matrix:
       d(i,j) = sqrt(0.5 * (1 - rho(i,j)))
    2. Apply Ward's agglomerative clustering on the distance matrix.
    3. Quasi-diagonalize (seriate) the covariance matrix using the
       dendrogram leaf ordering.
    4. Recursively bisect the sorted assets, allocating inverse-variance
       weights at each split to build the final risk-budget weights.

This avoids the instability of mean-variance optimization by never
inverting the covariance matrix, relying only on its diagonal and the
hierarchical cluster structure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RiskBudget:
    """Output of the HRP allocation.

    Attributes:
        weights: Mapping from asset name to portfolio weight (sums to 1.0).
        linkage_matrix: The (n-1, 4) linkage matrix from Ward's clustering.
        sorted_order: Asset names in quasi-diagonalized (seriated) order.
    """

    weights: Dict[str, float]
    linkage_matrix: np.ndarray
    sorted_order: List[str]

    def to_series(self) -> pd.Series:
        """Return weights as a named pandas Series sorted by weight descending."""
        s = pd.Series(self.weights, name="hrp_weight")
        return s.sort_values(ascending=False)

    def effective_n(self) -> float:
        """Compute the effective number of independent bets (1 / sum(w^2)).

        Also known as the Herfindahl inverse or portfolio diversity score.
        """
        w = np.array(list(self.weights.values()))
        return 1.0 / np.sum(w ** 2)


# ═══════════════════════════════════════════════════════════════════════════
# Main allocator
# ═══════════════════════════════════════════════════════════════════════════

class RiskAllocator:
    """Hierarchical Risk Parity allocator.

    Implements Lopez de Prado (2016) with Ledoit-Wolf shrinkage for
    covariance estimation, Ward's linkage for clustering, and iterative
    weight clamping to respect position limits.

    Parameters:
        min_weight: Floor for any single asset weight (default 0.05).
        max_weight: Ceiling for any single asset weight (default 0.40).
        linkage_method: Linkage criterion for agglomerative clustering.
            One of 'ward', 'single', 'complete', 'average' (default 'ward').

    Reference:
        Lopez de Prado, M. (2016). "Building Diversified Portfolios that
        Outperform Out-of-Sample." Journal of Portfolio Management, 42(4).
    """

    def __init__(
        self,
        min_weight: float = 0.05,
        max_weight: float = 0.40,
        linkage_method: str = "ward",
    ) -> None:
        if min_weight < 0 or min_weight >= max_weight:
            raise ValueError(
                f"Invalid bounds: min_weight={min_weight}, max_weight={max_weight}. "
                "Require 0 <= min_weight < max_weight."
            )
        if linkage_method not in ("ward", "single", "complete", "average"):
            raise ValueError(
                f"Unknown linkage method '{linkage_method}'. "
                "Choose from: ward, single, complete, average."
            )
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.linkage_method = linkage_method

    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────

    def allocate(self, returns: pd.DataFrame) -> RiskBudget:
        """Run the full HRP pipeline on a returns DataFrame.

        Parameters:
            returns: T x N DataFrame of asset returns, where columns are
                asset names and rows are time observations.  Must have at
                least 2 assets and more observations than assets.

        Returns:
            RiskBudget with weights, linkage matrix, and sorted asset order.

        Raises:
            ValueError: If returns has fewer than 2 columns or insufficient rows.

        Reference:
            Lopez de Prado, M. (2016). Building Diversified Portfolios
            that Outperform Out-of-Sample. *J. Portfolio Management*, 42(4).
        """
        if returns.shape[1] < 2:
            raise ValueError(
                f"HRP requires >= 2 assets, got {returns.shape[1]}."
            )
        if returns.shape[0] <= returns.shape[1]:
            logger.warning(
                "Fewer observations (%d) than assets (%d). "
                "Ledoit-Wolf shrinkage will help, but results may be noisy.",
                returns.shape[0],
                returns.shape[1],
            )

        names = list(returns.columns)

        # Step 0: Robust covariance estimation via Ledoit-Wolf shrinkage
        lw = LedoitWolf().fit(returns.values)
        cov = lw.covariance_
        logger.debug(
            "Ledoit-Wolf shrinkage coefficient: %.4f", lw.shrinkage_
        )

        return self.allocate_from_covariance(cov, names)

    def allocate_from_covariance(
        self, cov: np.ndarray, names: List[str]
    ) -> RiskBudget:
        """Run HRP from a pre-computed covariance matrix.

        Useful when the caller already has a shrunk or custom covariance
        matrix (e.g., from an exponentially-weighted estimator).

        Parameters:
            cov: N x N covariance matrix (must be symmetric PSD).
            names: List of N asset names matching the rows/columns of cov.

        Returns:
            RiskBudget with weights, linkage matrix, and sorted asset order.

        Raises:
            ValueError: If dimensions are inconsistent or matrix is invalid.
        """
        n = len(names)
        if cov.shape != (n, n):
            raise ValueError(
                f"Covariance shape {cov.shape} does not match "
                f"{n} asset names."
            )

        # Step 1: Correlation-based distance matrix
        # d(i,j) = sqrt(0.5 * (1 - rho(i,j)))  — Lopez de Prado Eq. (2)
        corr = self._cov_to_corr(cov)
        dist = np.sqrt(0.5 * (1.0 - corr))
        np.fill_diagonal(dist, 0.0)

        # Step 2: Hierarchical clustering (Ward's linkage)
        condensed_dist = squareform(dist, checks=False)
        link = linkage(condensed_dist, method=self.linkage_method)

        # Step 3: Quasi-diagonalization (matrix seriation)
        sorted_indices = list(leaves_list(link).astype(int))
        sorted_names = [names[i] for i in sorted_indices]

        # Step 4: Recursive bisection with inverse-variance weights
        raw_weights = self._recursive_bisection(cov, sorted_indices)

        # Step 5: Clamp and normalize weights
        weight_dict = {
            names[i]: w for i, w in zip(sorted_indices, raw_weights)
        }
        clamped = self._clamp_and_normalize(weight_dict)

        logger.info(
            "HRP allocation complete: %d assets, effective N=%.1f",
            n,
            1.0 / sum(w ** 2 for w in clamped.values()),
        )

        return RiskBudget(
            weights=clamped,
            linkage_matrix=link,
            sorted_order=sorted_names,
        )

    # ───────────────────────────────────────────────────────────────────
    # Internal helpers
    # ───────────────────────────────────────────────────────────────────

    @staticmethod
    def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
        """Convert covariance matrix to correlation matrix.

        Handles zero-variance assets by replacing NaNs with zero
        correlation (conservative fallback).
        """
        std = np.sqrt(np.diag(cov))
        # Avoid division by zero for constant-return assets
        std[std == 0] = 1.0
        corr = cov / np.outer(std, std)
        # Numerical cleanup: clamp to [-1, 1]
        corr = np.clip(corr, -1.0, 1.0)
        np.fill_diagonal(corr, 1.0)
        return corr

    def _recursive_bisection(
        self, cov: np.ndarray, sorted_indices: List[int]
    ) -> List[float]:
        """Recursive bisection step of HRP.

        Splits the sorted asset list into two halves at each level,
        allocating capital proportional to the inverse cluster variance.
        This is the core of Lopez de Prado's top-down allocation.

        Parameters:
            cov: Full N x N covariance matrix.
            sorted_indices: Asset indices in quasi-diagonalized order.

        Returns:
            List of raw weights in the same order as sorted_indices.
        """
        n = len(sorted_indices)
        weights = np.ones(n)

        # Stack-based iterative implementation (avoids recursion depth issues)
        # Each item on the stack is (start_idx, end_idx) into sorted_indices
        stack: List[tuple[int, int]] = [(0, n)]

        while stack:
            start, end = stack.pop()
            if end - start <= 1:
                continue

            mid = (start + end) // 2
            left_indices = sorted_indices[start:mid]
            right_indices = sorted_indices[mid:end]

            # Cluster variance = w' * Cov * w, where w = inverse-variance within cluster
            left_var = self._cluster_variance(cov, left_indices)
            right_var = self._cluster_variance(cov, right_indices)

            # Allocation factor: proportion to left cluster
            # alpha = 1 - left_var / (left_var + right_var)
            total_var = left_var + right_var
            if total_var < 1e-16:
                alpha = 0.5
            else:
                alpha = 1.0 - left_var / total_var

            # Scale weights
            weights[start:mid] *= alpha
            weights[mid:end] *= (1.0 - alpha)

            # Recurse into both halves
            stack.append((start, mid))
            stack.append((mid, end))

        return weights.tolist()

    @staticmethod
    def _cluster_variance(cov: np.ndarray, indices: List[int]) -> float:
        """Compute the variance of an inverse-variance-weighted cluster.

        Within each cluster, assets are weighted by their inverse variance
        (diagonal of the covariance matrix). The cluster variance is then:
            Var_cluster = w' * Cov_sub * w

        This gives a scalar measure of the cluster's risk contribution.

        Parameters:
            cov: Full covariance matrix.
            indices: Row/column indices of assets in this cluster.

        Returns:
            Scalar cluster variance.
        """
        sub_cov = cov[np.ix_(indices, indices)]
        diag = np.diag(sub_cov)

        # Inverse-variance weights (handle zero-variance gracefully)
        inv_var = np.where(diag > 1e-16, 1.0 / diag, 0.0)
        inv_var_sum = inv_var.sum()
        if inv_var_sum < 1e-16:
            # Fallback to equal weights if all variances are zero
            w = np.ones(len(indices)) / len(indices)
        else:
            w = inv_var / inv_var_sum

        # Cluster variance: w' * Cov * w
        cluster_var = float(w @ sub_cov @ w)
        return cluster_var

    def _clamp_and_normalize(
        self, weights: Dict[str, float], max_iterations: int = 50
    ) -> Dict[str, float]:
        """Iteratively clamp weights to [min_weight, max_weight] and renormalize.

        After recursive bisection, some weights may fall outside the
        acceptable bounds. This method iteratively clamps violators and
        redistributes the excess/deficit to unclamped assets, repeating
        until convergence or max_iterations is reached.

        Parameters:
            weights: Raw weight dictionary from recursive bisection.
            max_iterations: Safety limit on iteration count.

        Returns:
            Clamped and normalized weight dictionary summing to 1.0.
        """
        w = dict(weights)
        n = len(w)

        # Edge case: if n * min_weight > 1.0, the constraints are infeasible
        if n * self.min_weight > 1.0:
            logger.warning(
                "min_weight=%.3f with %d assets is infeasible (sum > 1.0). "
                "Falling back to equal weights.",
                self.min_weight,
                n,
            )
            equal_w = 1.0 / n
            return {name: equal_w for name in w}

        for iteration in range(max_iterations):
            total = sum(w.values())
            if total < 1e-16:
                # Degenerate case: reset to equal weights
                equal_w = 1.0 / n
                return {name: equal_w for name in w}

            # Normalize to sum to 1.0
            w = {name: val / total for name, val in w.items()}

            # Clamp
            clamped_names: set = set()
            clamped_total = 0.0
            free_total = 0.0

            for name, val in w.items():
                if val < self.min_weight:
                    w[name] = self.min_weight
                    clamped_names.add(name)
                    clamped_total += self.min_weight
                elif val > self.max_weight:
                    w[name] = self.max_weight
                    clamped_names.add(name)
                    clamped_total += self.max_weight
                else:
                    free_total += val

            if not clamped_names:
                # All weights within bounds — done
                break

            # Redistribute: scale free weights so total = 1.0
            remaining = 1.0 - clamped_total
            if remaining <= 0 or free_total < 1e-16:
                # All assets are clamped; normalize the clamped set
                total_clamped = sum(w.values())
                w = {name: val / total_clamped for name, val in w.items()}
                break

            scale = remaining / free_total
            for name in w:
                if name not in clamped_names:
                    w[name] *= scale
        else:
            logger.warning(
                "Weight clamping did not converge in %d iterations.",
                max_iterations,
            )
            # Final normalization
            total = sum(w.values())
            w = {name: val / total for name, val in w.items()}

        return w
