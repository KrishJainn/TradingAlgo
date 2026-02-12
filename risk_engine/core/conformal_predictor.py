"""Conformal prediction intervals for position sizing.

Conformal prediction provides distribution-free, finite-sample valid
prediction intervals.  Unlike parametric approaches (e.g. Gaussian
confidence intervals), conformal methods require only exchangeability
of the calibration data and guarantee marginal coverage at the specified
level regardless of the underlying distribution.

This module implements:

1. **Split conformal prediction** (Vovk et al. 2005):
   - Train a model on a proper training set.
   - Compute nonconformity scores on a held-out calibration set:
         alpha_i = |y_i - hat{y}_i|
   - For a new point with prediction hat{y}, the (1 - alpha) interval is:
         [hat{y} - q, hat{y} + q]
     where q is the ceil((n+1)*(1-alpha)) / n quantile of the calibration
     scores.  This guarantees P(Y in interval) >= 1 - alpha.

2. **Adaptive conformal inference** (Gibbs & Candes 2021):
   - Online update of the miscoverage level:
         alpha_{t+1} = alpha_t + eta * (alpha_target - 1{y_t in C_t})
     where eta is the step size and 1{.} is the indicator function.
   - Adapts to distribution shift: if the model is miscalibrated, alpha
     increases (intervals widen); if over-covering, alpha decreases.

3. **Position sizing adjustment**:
   - Wider intervals signal higher forecast uncertainty -> reduce position.
   - Narrower intervals -> maintain or increase position (up to base).
   - Formula: adj_size = base_size * (median_width / current_width)
   - Clamped to [0.25 * base_size, base_size] to avoid extremes.

Key formulae
-------------
Quantile index for coverage (1 - alpha):

    k = ceil((n + 1) * (1 - alpha))

Prediction interval:

    C(x) = [hat{y}(x) - q_k, hat{y}(x) + q_k]

where q_k is the k-th smallest absolute residual in the calibration set.

Adaptive update (Gibbs & Candes 2021):

    alpha_{t+1} = alpha_t + eta * (alpha_target - 1{y_t in C_t(x_t)})

References
----------
Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in
    a Random World*. Springer. Chapter 2.
Shafer, G. & Vovk, V. (2008). "A Tutorial on Conformal Prediction."
    *Journal of Machine Learning Research*, 9, 371-421.
Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J. & Wasserman, L. (2018).
    "Distribution-Free Predictive Inference for Regression." *JASA*,
    113(523), 1094-1111.
Gibbs, I. & Candes, E. (2021). "Adaptive Conformal Inference Under
    Distribution Shift." *NeurIPS 2021*.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from risk_engine.config.constants import RISK_DEFAULTS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ConformalResult:
    """Output of conformal prediction interval computation.

    Attributes
    ----------
    lower : float
        Lower bound of the prediction interval:
            lower = point_forecast - quantile_residual.
    upper : float
        Upper bound of the prediction interval:
            upper = point_forecast + quantile_residual.
    coverage : float
        Target marginal coverage probability (1 - alpha).  The conformal
        guarantee is P(Y in [lower, upper]) >= coverage for exchangeable
        calibration data (Vovk et al. 2005, Theorem 2.2).
    alpha : float
        Current miscoverage level, i.e. alpha = 1 - coverage.  May differ
        from the initial setting after adaptive updates.
    n_calibration : int
        Number of calibration residuals used to compute the quantile.
    """

    lower: float
    upper: float
    coverage: float
    alpha: float
    n_calibration: int

    @property
    def width(self) -> float:
        """Width of the prediction interval: upper - lower."""
        return self.upper - self.lower


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════


class ConformalPredictor:
    """Split conformal predictor with adaptive online updates.

    Provides distribution-free prediction intervals that can be used to
    adjust position sizes: wider intervals indicate higher uncertainty and
    trigger position reduction.

    Parameters
    ----------
    coverage : float
        Target marginal coverage probability in (0, 1).  Default is read
        from ``RISK_DEFAULTS["conformal_coverage"]`` (typically 0.90).

    Raises
    ------
    ValueError
        If ``coverage`` is not in (0, 1).

    References
    ----------
    Vovk et al. (2005). *Algorithmic Learning in a Random World*, Ch. 2.
    Shafer & Vovk (2008). *JMLR*, 9, 371-421.
    Gibbs & Candes (2021). *NeurIPS 2021*.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> residuals = rng.standard_normal(200)
    >>> cp = ConformalPredictor(coverage=0.90)
    >>> cp.calibrate(residuals)
    >>> result = cp.predict_interval(point_forecast=0.5)
    >>> result.lower < 0.5 < result.upper
    True
    """

    def __init__(self, coverage: float | None = None) -> None:
        if coverage is None:
            coverage = float(RISK_DEFAULTS.get("conformal_coverage", 0.90))

        if not 0.0 < coverage < 1.0:
            raise ValueError(
                f"coverage must be in (0, 1), got {coverage}"
            )

        self._target_coverage: float = coverage
        self._alpha: float = 1.0 - coverage

        # Calibration state
        self._sorted_residuals: np.ndarray | None = None
        self._median_width: float | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        """Current miscoverage level (1 - coverage)."""
        return self._alpha

    @property
    def coverage(self) -> float:
        """Current target coverage (1 - alpha)."""
        return 1.0 - self._alpha

    @property
    def is_calibrated(self) -> bool:
        """Whether ``calibrate()`` has been called with valid residuals."""
        return self._sorted_residuals is not None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, residuals: np.ndarray) -> None:
        """Calibrate the conformal predictor using held-out residuals.

        Stores the sorted absolute residuals for quantile lookup during
        ``predict_interval()``.

        Nonconformity score (Lei et al. 2018):

            alpha_i = |y_i - hat{y}_i|

        The sorted residuals define the empirical CDF from which the
        conformal quantile is extracted.

        Parameters
        ----------
        residuals : np.ndarray
            1-D array of residuals ``y_i - hat{y}_i`` from the calibration
            set.  Both positive and negative residuals are accepted; their
            absolute values are used as nonconformity scores.

        Raises
        ------
        ValueError
            If ``residuals`` is empty.

        Reference
        ---------
        Lei et al. (2018). *JASA*, 113(523), 1094-1111. Section 3.
        Vovk et al. (2005). *Algorithmic Learning in a Random World*, Ch. 2.
        """
        residuals = np.asarray(residuals, dtype=np.float64).ravel()

        if len(residuals) == 0:
            raise ValueError("Cannot calibrate with an empty residual array.")

        abs_residuals: np.ndarray = np.abs(residuals)
        self._sorted_residuals = np.sort(abs_residuals)

        # Pre-compute median width for position sizing normalisation
        # Median width = 2 * median(|residual|)
        self._median_width = 2.0 * float(np.median(abs_residuals))

        logger.info(
            "Conformal predictor calibrated: n=%d, median_|r|=%.6f, "
            "target_coverage=%.2f",
            len(residuals),
            self._median_width / 2.0,
            self.coverage,
        )

    # ------------------------------------------------------------------
    # Prediction interval
    # ------------------------------------------------------------------

    def predict_interval(self, point_forecast: float) -> ConformalResult:
        """Compute a conformal prediction interval around a point forecast.

        Quantile calculation (Vovk et al. 2005, Proposition 2.2):

            k = ceil((n + 1) * (1 - alpha))

        If k > n, the interval is set to [-inf, +inf] (guaranteed coverage
        but vacuous).  Otherwise the k-th smallest absolute residual gives
        the half-width of the symmetric interval:

            C(x) = [hat{y} - q_k, hat{y} + q_k]

        Parameters
        ----------
        point_forecast : float
            The model's point prediction hat{y}(x) for a new observation.

        Returns
        -------
        ConformalResult
            Prediction interval with coverage guarantee.

        Raises
        ------
        RuntimeError
            If ``calibrate()`` has not been called.

        Reference
        ---------
        Vovk et al. (2005). *Algorithmic Learning in a Random World*,
        Proposition 2.2, page 28.
        """
        if self._sorted_residuals is None:
            raise RuntimeError(
                "ConformalPredictor has not been calibrated. "
                "Call calibrate() first."
            )

        n: int = len(self._sorted_residuals)

        # Quantile index: ceil((n+1) * (1 - alpha)) — 1-based
        k: int = math.ceil((n + 1) * (1.0 - self._alpha))

        if k > n:
            # Cannot guarantee coverage with current calibration size;
            # return vacuous interval.
            logger.warning(
                "Conformal quantile index k=%d exceeds n=%d. "
                "Returning vacuous interval.",
                k,
                n,
            )
            q: float = float("inf")
        elif k < 1:
            # alpha very close to 1 (no coverage requested)
            q = 0.0
        else:
            # Convert to 0-based index
            q = float(self._sorted_residuals[k - 1])

        lower: float = point_forecast - q
        upper: float = point_forecast + q

        result = ConformalResult(
            lower=lower,
            upper=upper,
            coverage=self.coverage,
            alpha=self._alpha,
            n_calibration=n,
        )

        logger.debug(
            "Conformal interval: [%.6f, %.6f] (width=%.6f, alpha=%.4f, k=%d/%d)",
            lower,
            upper,
            result.width,
            self._alpha,
            k,
            n,
        )

        return result

    # ------------------------------------------------------------------
    # Adaptive online update
    # ------------------------------------------------------------------

    def adaptive_update(
        self,
        new_residual: float,
        step_size: float = 0.01,
    ) -> None:
        """Perform an adaptive conformal inference update (Gibbs & Candes 2021).

        Online update rule for the miscoverage level:

            alpha_{t+1} = alpha_t + step_size * (alpha_target - 1{y_t in C_t})

        Interpretation:
        - If the true observation fell *outside* the interval (miss),
          ``1{y_t in C_t} = 0`` and alpha *increases*, widening future
          intervals.
        - If it fell *inside* (hit), ``1{y_t in C_t} = 1`` and alpha
          *decreases*, tightening future intervals.

        The step_size ``eta`` controls the adaptation speed.  Gibbs & Candes
        (2021) show that for a fixed step size, the time-averaged coverage
        converges to the target level even under distribution shift.

        Parameters
        ----------
        new_residual : float
            The new observed residual ``y_t - hat{y}_t``.
        step_size : float
            Learning rate ``eta`` for the alpha update.  Default 0.01.
            Smaller values give more stable but slower adaptation.

        Raises
        ------
        RuntimeError
            If ``calibrate()`` has not been called.
        ValueError
            If ``step_size`` is not in (0, 1).

        Reference
        ---------
        Gibbs, I. & Candes, E. (2021). "Adaptive Conformal Inference Under
        Distribution Shift." *NeurIPS 2021*, Theorem 1.
        """
        if self._sorted_residuals is None:
            raise RuntimeError(
                "ConformalPredictor has not been calibrated. "
                "Call calibrate() first."
            )
        if not 0.0 < step_size < 1.0:
            raise ValueError(
                f"step_size must be in (0, 1), got {step_size}"
            )

        # Determine whether the new observation was covered
        abs_residual: float = abs(new_residual)
        n: int = len(self._sorted_residuals)
        k: int = math.ceil((n + 1) * (1.0 - self._alpha))

        if k > n:
            # Vacuous interval covers everything
            covered: float = 1.0
        elif k < 1:
            covered = 0.0
        else:
            threshold: float = float(self._sorted_residuals[k - 1])
            covered = 1.0 if abs_residual <= threshold else 0.0

        # Target miscoverage is alpha_target = 1 - target_coverage
        alpha_target: float = 1.0 - self._target_coverage

        # Update: alpha_{t+1} = alpha_t + eta * (alpha_target - covered)
        # Note: when covered=0 (miss), alpha increases; when covered=1 (hit),
        # alpha decreases.  The sign convention follows Gibbs & Candes (2021).
        self._alpha += step_size * (alpha_target - covered)

        # Clamp alpha to (0, 1) for numerical safety
        self._alpha = float(np.clip(self._alpha, 1e-6, 1.0 - 1e-6))

        # Optionally add the new residual to the calibration set
        # (growing window; keeps all history)
        self._sorted_residuals = np.sort(
            np.append(self._sorted_residuals, abs_residual)
        )

        # Update median width for position sizing
        self._median_width = 2.0 * float(np.median(self._sorted_residuals))

        logger.debug(
            "Adaptive update: |r|=%.6f, covered=%d, alpha=%.6f -> coverage=%.4f, "
            "n_cal=%d",
            abs_residual,
            int(covered),
            self._alpha,
            self.coverage,
            len(self._sorted_residuals),
        )

    # ------------------------------------------------------------------
    # Position sizing adjustment
    # ------------------------------------------------------------------

    def position_adjustment(
        self,
        base_size: float,
        interval: ConformalResult,
    ) -> float:
        """Scale position size inversely to prediction interval width.

        Wider conformal intervals indicate higher model uncertainty, so
        positions should be reduced.  Narrower intervals indicate the model
        is confident, so positions can approach the base size.

        Formula:

            adj_size = base_size * (median_width / current_width)

        where ``median_width`` is the median interval width observed during
        calibration and ``current_width`` is the width of the given
        ``interval``.  The result is clamped to:

            [0.25 * base_size, base_size]

        to prevent excessive leverage (never exceed base) or excessively
        small positions (floor at 25 % of base).

        Parameters
        ----------
        base_size : float
            The baseline position size (e.g. from Kelly criterion or fixed
            allocation).  Must be > 0.
        interval : ConformalResult
            A prediction interval from ``predict_interval()``.

        Returns
        -------
        float
            Adjusted position size in [0.25 * base_size, base_size].

        Raises
        ------
        RuntimeError
            If ``calibrate()`` has not been called (median width unavailable).
        ValueError
            If ``base_size`` is not positive.
        """
        if self._median_width is None:
            raise RuntimeError(
                "ConformalPredictor has not been calibrated. "
                "Call calibrate() first."
            )
        if base_size <= 0.0:
            raise ValueError(
                f"base_size must be positive, got {base_size}"
            )

        current_width: float = interval.width

        # Guard against zero or negative width (degenerate interval)
        if current_width <= 0.0:
            logger.warning(
                "Degenerate interval width=%.6f; returning base_size.",
                current_width,
            )
            return base_size

        # Guard against infinite width (vacuous interval)
        if not np.isfinite(current_width):
            logger.warning(
                "Infinite interval width; returning minimum position."
            )
            return 0.25 * base_size

        # Scale inversely to width, normalised by median
        ratio: float = self._median_width / current_width
        adjusted: float = base_size * ratio

        # Clamp to [0.25 * base_size, base_size]
        floor: float = 0.25 * base_size
        ceiling: float = base_size
        adjusted = float(np.clip(adjusted, floor, ceiling))

        logger.debug(
            "Position adjustment: base=%.4f, median_w=%.6f, current_w=%.6f, "
            "ratio=%.4f, adjusted=%.4f",
            base_size,
            self._median_width,
            current_width,
            ratio,
            adjusted,
        )

        return adjusted
