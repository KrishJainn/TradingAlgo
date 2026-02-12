"""Risk estimation: volatility, covariance, and regime-adjusted measures.

Implements:
1. EWMA (Exponentially Weighted Moving Average) volatility —
   RiskMetrics (1996) technical document by J.P. Morgan/Reuters.
2. Ledoit-Wolf shrinkage covariance estimator —
   Ledoit, O. & Wolf, M. (2004) "A well-conditioned estimator for
   large-dimensional covariance matrices", Journal of Multivariate Analysis,
   88(2), 365-411.
3. Regime-adjusted volatility scaling for Bull / Bear / Sideways markets.

References
----------
J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*, 4th ed.
Ledoit, O. & Wolf, M. (2004). J. Multivariate Anal. 88(2):365-411.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)

# Default regime multipliers grounded in empirical observation:
# Bear-market realised vol is roughly 1.5-2x bull-market vol for equity
# indices (see Ang & Chen 2002, "Asymmetric correlations of equity portfolios").
_DEFAULT_REGIME_MULTIPLIERS: dict[str, float] = {
    "bull": 0.8,
    "bear": 1.5,
    "sideways": 1.0,
}


@dataclass(frozen=True)
class RiskParams:
    """Configuration parameters for the risk estimator.

    Attributes
    ----------
    ewma_lambda : float
        Decay factor for EWMA volatility.  The RiskMetrics (1996) standard
        is 0.94 for daily data and 0.97 for monthly data.
    min_observations : int
        Minimum number of return observations required before estimation.
    regime_multipliers : dict[str, float]
        Mapping of regime label -> volatility scaling factor.
    annualisation_factor : float
        Factor to annualise daily volatility (sqrt(252) ≈ 15.87).
    """

    ewma_lambda: float = 0.94
    min_observations: int = 30
    regime_multipliers: dict[str, float] | None = None
    annualisation_factor: float = np.sqrt(252)

    def __post_init__(self) -> None:
        if not 0 < self.ewma_lambda < 1:
            raise ValueError(
                f"ewma_lambda must be in (0, 1), got {self.ewma_lambda}"
            )


class RiskEstimator:
    """Estimate volatility, covariance, and regime-adjusted risk measures.

    Parameters
    ----------
    params : RiskParams, optional
        Configuration dataclass.  Uses defaults if not provided.

    Examples
    --------
    >>> import numpy as np
    >>> re = RiskEstimator()
    >>> returns = np.random.randn(100) * 0.01
    >>> vol_series = re.ewma_volatility(returns)
    >>> vol_series.shape
    (100,)
    """

    def __init__(self, params: Optional[RiskParams] = None) -> None:
        self.params: RiskParams = params or RiskParams()
        self._regime_mult: dict[str, float] = (
            self.params.regime_multipliers
            if self.params.regime_multipliers is not None
            else dict(_DEFAULT_REGIME_MULTIPLIERS)
        )

    # ------------------------------------------------------------------
    # EWMA Volatility
    # ------------------------------------------------------------------

    def ewma_volatility(
        self,
        returns: np.ndarray,
        lam: float | None = None,
    ) -> np.ndarray:
        """Compute EWMA volatility series.

        Formula (RiskMetrics 1996)
        --------------------------
            sigma_t^2 = lambda * sigma_{t-1}^2 + (1 - lambda) * r_{t-1}^2

        where ``lambda`` (``lam``) is the decay factor.  A higher lambda gives
        more weight to historical variance; the RiskMetrics daily standard is
        lambda = 0.94.

        The half-life in days is:  h = -ln(2) / ln(lambda).
        For lambda = 0.94 this gives h ≈ 11 days.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of simple or log returns.
        lam : float, optional
            Decay factor in (0, 1).  Defaults to ``self.params.ewma_lambda``.

        Returns
        -------
        np.ndarray
            1-D array of EWMA volatility estimates (standard deviations),
            same length as ``returns``.  The first element is seeded with
            the absolute value of the first return.

        Reference
        ---------
        J.P. Morgan/Reuters (1996). *RiskMetrics — Technical Document*, 4th ed.
        Section 5.2, Equation 5.12.
        """
        if lam is None:
            lam = self.params.ewma_lambda

        if not 0 < lam < 1:
            raise ValueError(f"lam must be in (0, 1), got {lam}")

        returns = np.asarray(returns, dtype=np.float64).ravel()
        n: int = len(returns)

        if n == 0:
            return np.array([], dtype=np.float64)

        variance = np.empty(n, dtype=np.float64)

        # Seed with the squared first return (unbiased seed is not critical
        # because EWMA is adaptive and the seed washes out quickly).
        variance[0] = returns[0] ** 2

        for t in range(1, n):
            variance[t] = lam * variance[t - 1] + (1.0 - lam) * returns[t - 1] ** 2

        # Return volatility (standard deviation), not variance.
        vol: np.ndarray = np.sqrt(variance)
        return vol

    # ------------------------------------------------------------------
    # Ledoit-Wolf Shrinkage Covariance
    # ------------------------------------------------------------------

    def ledoit_wolf_covariance(
        self,
        returns: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Estimate covariance matrix using Ledoit-Wolf shrinkage.

        Shrinks the sample covariance S toward a structured target F
        (scaled identity) using the optimal shrinkage intensity alpha*:

            Sigma_shrunk = alpha* * F + (1 - alpha*) * S

        The shrinkage intensity is chosen to minimise the expected Frobenius
        loss ||Sigma_shrunk - Sigma_true||_F^2 under mild assumptions.

        Parameters
        ----------
        returns : np.ndarray
            2-D array of shape (T, N) where T = number of observations and
            N = number of assets.

        Returns
        -------
        tuple[np.ndarray, float]
            (covariance_matrix, shrinkage_intensity) where
            ``covariance_matrix`` has shape (N, N) and ``shrinkage_intensity``
            is the optimal alpha* in [0, 1].

        Raises
        ------
        ValueError
            If fewer than ``min_observations`` rows are provided.

        Reference
        ---------
        Ledoit, O. & Wolf, M. (2004). "A well-conditioned estimator for
        large-dimensional covariance matrices." *J. Multivariate Anal.*,
        88(2), 365-411.
        """
        returns = np.asarray(returns, dtype=np.float64)

        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        t_obs, n_assets = returns.shape

        if t_obs < self.params.min_observations:
            raise ValueError(
                f"Need at least {self.params.min_observations} observations, "
                f"got {t_obs}."
            )

        lw = LedoitWolf(assume_centered=False)
        lw.fit(returns)

        cov_matrix: np.ndarray = lw.covariance_
        shrinkage: float = float(lw.shrinkage_)

        logger.debug(
            "Ledoit-Wolf shrinkage: %.4f for (%d obs, %d assets)",
            shrinkage,
            t_obs,
            n_assets,
        )

        return cov_matrix, shrinkage

    # ------------------------------------------------------------------
    # Correlation Matrix (with Ledoit-Wolf shrinkage)
    # ------------------------------------------------------------------

    def correlation_matrix(
        self,
        returns: pd.DataFrame,
    ) -> np.ndarray:
        """Compute correlation matrix from Ledoit-Wolf shrunk covariance.

        Steps:
            1. Estimate shrunk covariance C via ``ledoit_wolf_covariance``.
            2. Convert to correlation:  R_ij = C_ij / sqrt(C_ii * C_jj).

        Using the shrunk covariance for correlation avoids the noisy
        eigenvalue spectrum of the sample correlation matrix, which is
        especially problematic when N / T is not small (Marchenko-Pastur
        regime).

        Parameters
        ----------
        returns : pd.DataFrame
            DataFrame of shape (T, N) with asset returns in columns.

        Returns
        -------
        np.ndarray
            Correlation matrix of shape (N, N) with ones on the diagonal.

        Reference
        ---------
        Ledoit, O. & Wolf, M. (2004). J. Multivariate Anal. 88(2):365-411.
        """
        cov, _shrinkage = self.ledoit_wolf_covariance(returns.values)

        # D = diag(1 / sqrt(diag(cov)))
        std_devs = np.sqrt(np.diag(cov))
        # Guard against zero std devs (constant columns)
        std_devs = np.where(std_devs > 0, std_devs, 1.0)

        d_inv = np.diag(1.0 / std_devs)
        corr: np.ndarray = d_inv @ cov @ d_inv

        # Numerical cleanup: enforce exact ones on diagonal and symmetry.
        np.fill_diagonal(corr, 1.0)
        corr = (corr + corr.T) / 2.0

        return corr

    # ------------------------------------------------------------------
    # Regime-Adjusted Volatility
    # ------------------------------------------------------------------

    def regime_adjusted_vol(
        self,
        base_vol: float,
        regime: str,
    ) -> float:
        """Scale volatility by a regime-dependent multiplier.

        Empirical equity-index data shows that realised volatility is
        roughly 1.5-2x higher in bear markets than in bull markets
        (Ang & Chen 2002).  This method applies a simple multiplicative
        adjustment:

            sigma_adjusted = base_vol * multiplier(regime)

        Default multipliers:
            - ``"bull"``:     0.8   (vol tends to compress in uptrends)
            - ``"bear"``:     1.5   (vol expands in downtrends)
            - ``"sideways"``: 1.0   (neutral)

        Parameters
        ----------
        base_vol : float
            Unconditional or EWMA volatility estimate.
        regime : str
            Market regime label (case-insensitive).  Must be one of the
            keys in the configured ``regime_multipliers``.

        Returns
        -------
        float
            Regime-scaled volatility.

        Raises
        ------
        ValueError
            If ``regime`` is not a recognised label.

        Reference
        ---------
        Ang, A. & Chen, J. (2002). "Asymmetric correlations of equity
        portfolios." *J. Financial Economics*, 63(3), 443-494.
        """
        regime_lower = regime.strip().lower()

        if regime_lower not in self._regime_mult:
            valid = list(self._regime_mult.keys())
            raise ValueError(
                f"Unknown regime '{regime}'. Valid regimes: {valid}"
            )

        multiplier: float = self._regime_mult[regime_lower]
        adjusted: float = base_vol * multiplier

        logger.debug(
            "Regime '%s': base_vol=%.6f * multiplier=%.2f -> %.6f",
            regime,
            base_vol,
            multiplier,
            adjusted,
        )

        return adjusted
