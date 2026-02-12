"""
Deflated Sharpe Ratio (DSR).

Implements the Deflated Sharpe Ratio from Bailey & Lopez de Prado (2014),
which corrects the observed Sharpe Ratio for the multiplicity of trials
(strategy selection bias), non-normality of returns (skewness and kurtosis),
and finite sample length.

Core Formulae
-------------

1. **False Strategy Theorem** --- expected maximum Sharpe Ratio under the
   null of zero true Sharpe across *N* independent trials of length *T*:

       E[max(SR)] ~ (1 - gamma) * Phi^{-1}(1 - 1/N)
                   + gamma * Phi^{-1}(1 - 1/(N*e))

   where gamma = 0.5772... is the Euler--Mascheroni constant, Phi^{-1} is
   the standard normal quantile function, and *e* is Euler's number.

2. **Skewness / kurtosis correction** for the standard error of the Sharpe
   Ratio (Lo, 2002; Bailey & Lopez de Prado, 2012):

       SE(SR) = sqrt( (1 - S*SR + ((K-1)/4)*SR^2) / T )

   where S = sample skewness, K = sample excess kurtosis, and T = sample
   length.

3. **Deflated Sharpe Ratio**:

       DSR = Phi( (SR_obs - E[max(SR)]) / SE(SR) )

   A DSR close to 1 means the observed Sharpe is unlikely to be a
   statistical fluke; close to 0 means it is likely explained by selection
   bias alone.

4. **Haircut** --- multiplicative discount:

       haircut = 1 - E[max(SR)] / SR_obs        (if SR_obs > 0)

   Values below ~0.5 suggest the Sharpe is largely attributable to trial
   multiplicity.

References
----------
Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and Non-Normality."
    *Journal of Portfolio Management*, 40(5), 94--107.

Bailey, D. H. & Lopez de Prado, M. (2012). "The Sharpe Ratio Efficient
    Frontier." *Journal of Risk*, 15(2), 3--44.

Lo, A. W. (2002). "The Statistics of Sharpe Ratios." *Financial Analysts
    Journal*, 58(4), 36--52.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

from risk_engine.config.constants import (
    ANNUALIZATION_FACTOR,
    EULER_MASCHERONI,
    TRADING_DAYS_PER_YEAR,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class DSRResult:
    """Outcome of a Deflated Sharpe Ratio test.

    Attributes
    ----------
    dsr : float
        Deflated Sharpe Ratio --- probability that the observed Sharpe is
        genuine after correcting for multiple testing and non-normality.
        Range [0, 1]; values above 0.95 are conventionally significant.
    observed_sr : float
        Annualised Sharpe Ratio of the strategy's return stream.
    expected_max_sr : float
        Expected maximum Sharpe Ratio under the null of zero true Sharpe,
        derived from the False Strategy Theorem.
    haircut : float
        Multiplicative haircut factor in [0, 1].  A haircut of 0.7 means
        only 30 % of the observed Sharpe survives adjustment.
    is_significant : bool
        ``True`` when ``dsr >= 0.95`` (conventional 5 % significance level).
    """

    dsr: float
    observed_sr: float
    expected_max_sr: float
    haircut: float
    is_significant: bool


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════


class DeflatedSharpe:
    """Deflated Sharpe Ratio --- Bailey & Lopez de Prado (2014).

    Corrects the observed Sharpe Ratio for:

    * **Selection bias** (multiple strategies tested).
    * **Non-normality** (skewness and excess kurtosis of returns).
    * **Finite sample length** (small *T*).

    Parameters
    ----------
    n_trials : int
        Number of independent strategy variants tested during research.
        This is the *N* in the False Strategy Theorem.
    T : int
        Number of return observations used for evaluation.  Typically
        the number of trading days in the backtest.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> returns = rng.normal(0.0005, 0.01, size=504)
    >>> ds = DeflatedSharpe(n_trials=50, T=len(returns))
    >>> result = ds.compute(returns)
    >>> result.dsr  # doctest: +SKIP
    0.312...
    """

    def __init__(self, n_trials: int, T: int) -> None:
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if T < 2:
            raise ValueError(f"T must be >= 2, got {T}")

        self.n_trials: int = n_trials
        self.T: int = T

    # ------------------------------------------------------------------
    # False Strategy Theorem
    # ------------------------------------------------------------------

    @staticmethod
    def expected_max_sr(n_trials: int, T: int) -> float:
        """Expected maximum Sharpe Ratio under the null hypothesis.

        Uses the False Strategy Theorem (Bailey & Lopez de Prado, 2014):

            E[max(SR)] ~ (1 - gamma) * Z(1 - 1/N)
                        + gamma       * Z(1 - 1/(N*e))

        where:
            - gamma = 0.5772... (Euler--Mascheroni constant)
            - Z(p) = Phi^{-1}(p) (standard normal quantile / ppf)
            - N = n_trials
            - e = 2.71828... (Euler's number)

        The result is on the *per-observation* scale (not annualised).
        Multiply by ``sqrt(252)`` to annualise if daily returns are used.

        Parameters
        ----------
        n_trials : int
            Number of independent strategy trials.
        T : int
            Number of return observations per trial.

        Returns
        -------
        float
            Expected maximum annualised Sharpe Ratio under the null.

        Notes
        -----
        When ``n_trials == 1`` the formula reduces to ~ 0, as expected ---
        no selection bias exists for a single strategy.
        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}")
        if T < 2:
            raise ValueError(f"T must be >= 2, got {T}")

        gamma: float = EULER_MASCHERONI
        n: int = n_trials

        # Quantile arguments --- clip to avoid ppf(0) = -inf or ppf(1) = +inf
        p1: float = max(1.0e-15, 1.0 - 1.0 / n) if n > 1 else 0.5
        p2: float = max(1.0e-15, 1.0 - 1.0 / (n * math.e)) if n > 1 else 0.5

        z1: float = float(sp_stats.norm.ppf(p1))
        z2: float = float(sp_stats.norm.ppf(p2))

        # Expected max of N independent standard normal draws
        e_max_z: float = (1.0 - gamma) * z1 + gamma * z2

        # Under the null (true SR = 0), each annualised sample Sharpe is
        # distributed approximately N(0, TRADING_DAYS / T).  The standard
        # deviation of the annualised Sharpe is therefore:
        #   sigma_SR = sqrt(TRADING_DAYS_PER_YEAR / T)
        #
        # The expected max annualised Sharpe across N trials is:
        #   E[max SR_annual] = E[max Z] * sigma_SR
        #                    = E[max Z] * sqrt(252 / T)
        e_max_sr_annual: float = e_max_z * math.sqrt(
            TRADING_DAYS_PER_YEAR / T
        )

        return e_max_sr_annual

    # ------------------------------------------------------------------
    # Compute DSR
    # ------------------------------------------------------------------

    def compute(
        self,
        returns: np.ndarray,
        n_trials: int | None = None,
    ) -> DSRResult:
        """Compute the Deflated Sharpe Ratio for a return stream.

        Steps:
            1. Estimate annualised Sharpe, skewness *S*, excess kurtosis *K*.
            2. Compute the standard error of the Sharpe Ratio adjusted for
               non-normality (Lo, 2002):

                   SE(SR) = sqrt( (1 - S*SR + ((K-1)/4)*SR^2) / T )

            3. Compute expected max SR via the False Strategy Theorem.
            4. DSR = Phi( (SR_obs - E[max SR]) / SE(SR) ).

        Parameters
        ----------
        returns : np.ndarray
            1-D array of simple (arithmetic) returns.
        n_trials : int, optional
            Override the number of trials set at init.

        Returns
        -------
        DSRResult
            Named container with ``dsr``, ``observed_sr``,
            ``expected_max_sr``, ``haircut``, ``is_significant``.

        Raises
        ------
        ValueError
            If the return array has fewer than 2 observations or contains
            all-zero / constant values (zero variance).
        """
        returns = np.asarray(returns, dtype=np.float64).ravel()
        T: int = len(returns)
        n: int = n_trials if n_trials is not None else self.n_trials

        if T < 2:
            raise ValueError(f"Need at least 2 return observations, got {T}")

        # ---- Moment estimates ----
        mu: float = float(np.mean(returns))
        sigma: float = float(np.std(returns, ddof=1))

        if sigma < 1.0e-15:
            raise ValueError(
                "Return series has near-zero standard deviation; "
                "cannot compute a meaningful Sharpe Ratio."
            )

        # Daily (per-observation) Sharpe
        sr_daily: float = mu / sigma

        # Annualised Sharpe
        sr_annual: float = sr_daily * ANNUALIZATION_FACTOR

        # Sample skewness and excess kurtosis
        skew: float = float(sp_stats.skew(returns, bias=False))
        excess_kurt: float = float(sp_stats.kurtosis(returns, bias=False))

        # ---- Standard error of SR (adjusted for non-normality) ----
        # Lo (2002), Equation 4; Bailey & Lopez de Prado (2012):
        #   Var(SR) ~ (1 - S*SR + ((K-1)/4)*SR^2) / T
        # Here SR is the per-observation Sharpe (sr_daily).
        var_sr: float = (
            1.0
            - skew * sr_daily
            + ((excess_kurt - 1.0) / 4.0) * sr_daily**2
        ) / T
        se_sr: float = math.sqrt(max(var_sr, 1.0e-20))

        # ---- Expected max SR ----
        e_max_sr: float = self.expected_max_sr(n, T)

        # ---- DSR = Phi( (SR_obs - E[max SR]) / SE(SR) ) ----
        # We work in annualised space for interpretability.
        se_sr_annual: float = se_sr * ANNUALIZATION_FACTOR
        z_score: float = (sr_annual - e_max_sr) / se_sr_annual if se_sr_annual > 1.0e-15 else 0.0
        dsr: float = float(sp_stats.norm.cdf(z_score))

        # ---- Haircut ----
        haircut: float = self.haircut_sharpe(sr_annual, n, T)

        is_sig: bool = dsr >= 0.95

        logger.debug(
            "DSR=%.4f | SR_obs=%.4f | E[max SR]=%.4f | haircut=%.4f | sig=%s",
            dsr,
            sr_annual,
            e_max_sr,
            haircut,
            is_sig,
        )

        return DSRResult(
            dsr=round(dsr, 6),
            observed_sr=round(sr_annual, 6),
            expected_max_sr=round(e_max_sr, 6),
            haircut=round(haircut, 6),
            is_significant=is_sig,
        )

    # ------------------------------------------------------------------
    # Haircut
    # ------------------------------------------------------------------

    @staticmethod
    def haircut_sharpe(observed_sr: float, n_trials: int, T: int) -> float:
        """Compute the multiplicative haircut factor for the Sharpe Ratio.

        The haircut quantifies how much of the observed Sharpe survives
        after adjusting for multiple testing:

            haircut = max(0, 1 - E[max(SR)] / SR_obs)

        A haircut of 0.7 means 30 % of the observed Sharpe is attributable
        to selection bias; only 70 % is *real*.

        Parameters
        ----------
        observed_sr : float
            Observed annualised Sharpe Ratio.
        n_trials : int
            Number of independent strategy variants tested.
        T : int
            Number of return observations.

        Returns
        -------
        float
            Haircut factor in [0, 1].  Values near 1 indicate almost all
            of the Sharpe is genuine; values near 0 indicate heavy erosion
            by selection bias.

        Notes
        -----
        Returns 0.0 if ``observed_sr <= 0`` (negative Sharpe cannot be
        meaningfully haircutted).
        """
        if observed_sr <= 0.0:
            return 0.0

        e_max: float = DeflatedSharpe.expected_max_sr(n_trials, T)
        raw_haircut: float = 1.0 - e_max / observed_sr
        return max(0.0, raw_haircut)
