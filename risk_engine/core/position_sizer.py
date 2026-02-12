"""Position sizing via Fractional Kelly and Ergodicity-corrected Kelly.

Implements the Kelly criterion and its practical variants for determining
optimal bet / position sizes given estimated edge and risk.

Key references
--------------
- Kelly, J.L. (1956). "A New Interpretation of Information Rate."
  *Bell System Technical Journal*, 35(4), 917-926.
- Thorp, E.O. (2008). "The Kelly Criterion in Blackjack, Sports Betting
  and the Stock Market." In: *Handbook of Asset and Liability Management*,
  Zenios & Ziemba (eds.), North-Holland.
- Peters, O. (2019). "The ergodicity problem in economics."
  *Nature Physics*, 15, 1216-1221.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------


@dataclass(frozen=True)
class PositionSizeResult:
    """Output of the position-sizing calculation.

    Attributes
    ----------
    raw_fraction : float
        The Kelly (or ergodic-Kelly) optimal fraction *before* any
        clamping or shrinkage.
    clamped_fraction : float
        The fraction after applying ``kelly_fraction`` scaling, signal
        confidence adjustment, regime adjustment, and position caps.
    expected_growth_rate : float
        Estimated log-growth rate g(f) at the clamped fraction:
            g(f) = log(1 + f * mu) - 0.5 * f^2 * sigma^2
        (continuous approximation).
    method_used : str
        Label indicating which sizing method produced the result,
        e.g. ``"fractional_kelly"`` or ``"ergodic_kelly"``.
    """

    raw_fraction: float
    clamped_fraction: float
    expected_growth_rate: float
    method_used: str


# ------------------------------------------------------------------
# Regime multipliers for position shrinkage
# ------------------------------------------------------------------

_REGIME_POSITION_SCALE: dict[str, float] = {
    "bull": 1.0,
    "bear": 0.5,
    "sideways": 0.75,
}


# ------------------------------------------------------------------
# Main class
# ------------------------------------------------------------------


class PositionSizer:
    """Compute optimal position sizes using Kelly-criterion variants.

    Parameters
    ----------
    kelly_fraction : float
        Fractional Kelly multiplier in (0, 1].  A value of 0.25 means
        we bet one-quarter of the full Kelly amount — a common choice in
        practice to reduce variance of terminal wealth at the cost of
        slightly lower geometric growth.  Thorp (2008) recommends
        fractions in the 0.1–0.5 range.
    max_position_pct : float
        Hard cap on the position size as a fraction of portfolio NAV.
        Default 0.20 (20 %).

    Reference
    ---------
    Thorp, E.O. (2008). Section 4, "Practical implementation".
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        max_position_pct: float = 0.20,
    ) -> None:
        if not 0 < kelly_fraction <= 1.0:
            raise ValueError(
                f"kelly_fraction must be in (0, 1], got {kelly_fraction}"
            )
        if not 0 < max_position_pct <= 1.0:
            raise ValueError(
                f"max_position_pct must be in (0, 1], got {max_position_pct}"
            )

        self.kelly_fraction: float = kelly_fraction
        self.max_position_pct: float = max_position_pct

    # ------------------------------------------------------------------
    # Fractional Kelly
    # ------------------------------------------------------------------

    def fractional_kelly(
        self,
        mu: float,
        sigma: float,
        fraction: float | None = None,
    ) -> float:
        """Compute the fractional Kelly optimal bet size.

        Formula
        -------
            f* = fraction * (mu / sigma^2)

        where ``mu`` is the expected excess return and ``sigma`` is the
        return volatility.  The full Kelly fraction (fraction = 1) maximises
        the expected logarithm of wealth; using fraction < 1 trades a small
        amount of growth for substantially lower variance.

        Parameters
        ----------
        mu : float
            Expected excess return (e.g. annualised alpha).
        sigma : float
            Return volatility (same units as mu).
        fraction : float, optional
            Override for ``self.kelly_fraction``.

        Returns
        -------
        float
            Optimal position fraction.  Can be negative (short) when
            ``mu < 0``.  Returns 0.0 when ``sigma`` is effectively zero.

        Reference
        ---------
        Kelly (1956); Thorp (2008), Eq. 5.
        """
        if fraction is None:
            fraction = self.kelly_fraction

        if sigma <= 0 or np.isnan(sigma):
            logger.warning(
                "sigma is non-positive or NaN (%.6g); returning 0.0", sigma
            )
            return 0.0

        full_kelly: float = mu / (sigma ** 2)
        result: float = fraction * full_kelly

        logger.debug(
            "fractional_kelly: mu=%.6f, sigma=%.6f, full=%.4f, "
            "fraction=%.2f -> %.4f",
            mu,
            sigma,
            full_kelly,
            fraction,
            result,
        )
        return result

    # ------------------------------------------------------------------
    # Ergodic Kelly
    # ------------------------------------------------------------------

    def ergodic_kelly(
        self,
        mu: float,
        sigma: float,
    ) -> float:
        """Compute the ergodicity-corrected Kelly fraction.

        Background
        ----------
        Peters (2019) highlights that the ensemble-average (expected value)
        and the time-average (experienced by a single agent) of
        multiplicative wealth dynamics diverge.  Maximising the
        *time-average growth rate* rather than the *expected value* leads
        to the Kelly criterion naturally.

        For a continuous-time geometric Brownian motion model the
        time-average growth rate as a function of leverage f is:

            g(f) = f * mu  -  0.5 * f^2 * sigma^2

        Maximising g(f) over f gives:

            f* = mu / sigma^2

        which coincides with the classical Kelly formula.  The key
        distinction is *interpretive*: the Kelly bet is not a risk-
        preference — it is the *dynamically optimal* strategy for any
        ergodic multiplicative process.

        For the *discrete* Kelly with binary outcomes (win amount ``a``
        with probability ``p``, lose amount ``b`` with probability q=1-p):

            f* = p/a - q/b

        Parameters
        ----------
        mu : float
            Expected excess return per period.
        sigma : float
            Return volatility per period.

        Returns
        -------
        float
            Ergodic-optimal fraction.  Returns 0.0 when sigma is
            non-positive.

        Reference
        ---------
        Peters, O. (2019). "The ergodicity problem in economics."
        *Nature Physics*, 15, 1216-1221.  Equation (5).
        """
        if sigma <= 0 or np.isnan(sigma):
            logger.warning(
                "sigma is non-positive or NaN (%.6g); returning 0.0", sigma
            )
            return 0.0

        f_star: float = mu / (sigma ** 2)

        logger.debug(
            "ergodic_kelly: mu=%.6f, sigma=%.6f -> f*=%.4f",
            mu,
            sigma,
            f_star,
        )
        return f_star

    # ------------------------------------------------------------------
    # Full sizing pipeline
    # ------------------------------------------------------------------

    def size(
        self,
        mu: float,
        sigma: float,
        signal_direction: float,
        signal_confidence: float,
        regime: str,
    ) -> PositionSizeResult:
        """Compute a clamped, regime-aware position size.

        Pipeline
        --------
        1. Compute raw fraction via ``fractional_kelly(mu, sigma)``.
        2. Scale by ``signal_direction`` (sign) and ``signal_confidence``
           (magnitude in [0, 1]).
        3. Apply regime-dependent shrinkage (Bull=1.0, Bear=0.5,
           Sideways=0.75).
        4. Clamp to [-max_position_pct, +max_position_pct].
        5. Compute expected log-growth at the clamped fraction.

        Parameters
        ----------
        mu : float
            Expected excess return.
        sigma : float
            Return volatility.
        signal_direction : float
            +1.0 for long, -1.0 for short, or any float to express
            directional conviction.
        signal_confidence : float
            Confidence score in [0, 1].  Scales the position linearly.
        regime : str
            Market regime label (``"bull"`` / ``"bear"`` / ``"sideways"``).

        Returns
        -------
        PositionSizeResult
        """
        raw: float = self.fractional_kelly(mu, sigma)

        # Apply direction and confidence
        scaled: float = raw * signal_direction * max(0.0, min(1.0, signal_confidence))

        # Regime adjustment
        regime_lower = regime.strip().lower()
        regime_scale: float = _REGIME_POSITION_SCALE.get(regime_lower, 1.0)
        scaled *= regime_scale

        # Clamp
        clamped: float = float(
            np.clip(scaled, -self.max_position_pct, self.max_position_pct)
        )

        # Expected log-growth at the clamped fraction:
        #   g(f) = f * mu - 0.5 * f^2 * sigma^2
        # (continuous-time approximation for small f)
        if sigma > 0:
            growth: float = clamped * mu - 0.5 * (clamped ** 2) * (sigma ** 2)
        else:
            growth = 0.0

        return PositionSizeResult(
            raw_fraction=raw,
            clamped_fraction=clamped,
            expected_growth_rate=growth,
            method_used="fractional_kelly",
        )

    # ------------------------------------------------------------------
    # Multi-asset Kelly
    # ------------------------------------------------------------------

    def multi_asset_kelly(
        self,
        mu_vector: np.ndarray,
        cov_matrix: np.ndarray,
    ) -> np.ndarray:
        """Compute multi-asset Kelly-optimal allocations.

        Formula
        -------
            f* = Sigma^{-1} @ mu

        where ``mu`` is the (N,) vector of expected excess returns and
        ``Sigma`` is the (N, N) covariance matrix.  This is the
        continuous-time result for geometric Brownian motion and is
        equivalent to the mean-variance tangency portfolio when a
        risk-free rate exists.

        In practice the result is scaled by ``self.kelly_fraction`` and
        each element is clamped to [-max_position_pct, +max_position_pct].

        Parameters
        ----------
        mu_vector : np.ndarray
            Expected excess return vector of shape (N,).
        cov_matrix : np.ndarray
            Covariance matrix of shape (N, N).  Must be positive-definite
            (or at least positive semi-definite — a pseudo-inverse is used
            as fallback).

        Returns
        -------
        np.ndarray
            Fractional Kelly allocation vector of shape (N,).

        Raises
        ------
        ValueError
            If shapes are incompatible.

        Reference
        ---------
        Thorp, E.O. (2008). "The Kelly Criterion in Blackjack, Sports
        Betting and the Stock Market." Section 6, Multi-asset extension.
        """
        mu_vector = np.asarray(mu_vector, dtype=np.float64).ravel()
        cov_matrix = np.asarray(cov_matrix, dtype=np.float64)

        n = mu_vector.shape[0]

        if cov_matrix.shape != (n, n):
            raise ValueError(
                f"Shape mismatch: mu_vector has {n} elements but "
                f"cov_matrix has shape {cov_matrix.shape}."
            )

        # Attempt standard inverse; fall back to pseudo-inverse for
        # singular / near-singular matrices.
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            logger.warning(
                "Covariance matrix is singular; using pseudo-inverse."
            )
            cov_inv = np.linalg.pinv(cov_matrix)

        full_kelly_vec: np.ndarray = cov_inv @ mu_vector

        # Apply fractional Kelly shrinkage
        scaled: np.ndarray = self.kelly_fraction * full_kelly_vec

        # Per-asset position cap
        clamped: np.ndarray = np.clip(
            scaled, -self.max_position_pct, self.max_position_pct
        )

        logger.debug(
            "multi_asset_kelly: %d assets, max_abs_weight=%.4f",
            n,
            float(np.max(np.abs(clamped))),
        )

        return clamped
