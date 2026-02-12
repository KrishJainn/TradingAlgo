"""Copula-based tail dependence estimation for HRP distance adjustment.

Tail dependence captures the probability that two assets experience joint
extreme losses (or gains) simultaneously.  Standard correlation underestimates
this co-movement in the tails, leading HRP to under-penalise pairs that crash
together.  This module estimates tail dependence coefficients and inflates the
HRP distance matrix accordingly.

Key formulae
------------
Clayton copula lower tail dependence (Nelsen 2006, Theorem 5.4.2):

    lambda_L = 2^{-1/theta},   theta > 0

Gumbel copula upper tail dependence (Nelsen 2006, Theorem 5.4.6):

    lambda_U = 2 - 2^{1/theta},   theta >= 1

Empirical tail dependence (Schmidt & Stadtmuller 2006):

    hat{lambda}_L(q) = P(U <= q | V <= q)
                     = #{(u_i, v_i) : u_i <= q and v_i <= q} / #{v_i <= q}

    hat{lambda}_U(q) = P(U > 1-q | V > 1-q)
                     = #{(u_i, v_i) : u_i > 1-q and v_i > 1-q} / #{v_i > 1-q}

where (U, V) are rank-transformed (pseudo-copula) observations.

Adjusted HRP distance (original formulation in this module):

    d_adj(i,j) = d_std(i,j) * (1 + tail_penalty(i,j))

where ``d_std`` is the standard correlation-based distance
``sqrt(0.5 * (1 - rho))`` from Lopez de Prado (2016) and
``tail_penalty = max(lambda_L, lambda_U)`` penalises pairs with
high tail co-dependence so that HRP clusters them together and
allocates less combined weight.

References
----------
McNeil, A.J., Frey, R. & Embrechts, P. (2005). *Quantitative Risk
    Management: Concepts, Techniques and Tools*. Princeton University Press.
    Chapter 5: Copulas and Dependence.
Nelsen, R.B. (2006). *An Introduction to Copulas*, 2nd ed. Springer.
    Theorems 5.4.2 and 5.4.6.
Schmidt, R. & Stadtmuller, U. (2006). "Non-parametric estimation of tail
    dependence." *Scandinavian Journal of Statistics*, 33(2), 307-335.
Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform
    Out-of-Sample." *Journal of Portfolio Management*, 42(4), 59-69.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import rankdata

from risk_engine.config.constants import RISK_DEFAULTS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class TailDependenceResult:
    """Output of pairwise tail dependence estimation.

    Attributes
    ----------
    lower_tail : np.ndarray
        (N, N) matrix of lower tail dependence coefficients lambda_L(i,j).
        Each entry is in [0, 1].  lambda_L = 0 means asymptotic tail
        independence in the lower tail; lambda_L = 1 means perfect lower
        tail dependence.
    upper_tail : np.ndarray
        (N, N) matrix of upper tail dependence coefficients lambda_U(i,j).
        Interpretation is symmetric to ``lower_tail`` for the right tail.
    adjusted_distance : np.ndarray
        (N, N) HRP distance matrix adjusted for tail dependence:
            d_adj(i,j) = d_std(i,j) * (1 + max(lambda_L(i,j), lambda_U(i,j)))
        where d_std = sqrt(0.5 * (1 - rho(i,j))).  The diagonal is zero.
    """

    lower_tail: np.ndarray
    upper_tail: np.ndarray
    adjusted_distance: np.ndarray


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════


class TailDependence:
    """Estimate tail dependence and produce adjusted HRP distance matrices.

    Supports three estimation methods:

    * ``"empirical"`` -- non-parametric estimation from rank-transformed data
      (Schmidt & Stadtmuller 2006).  No distributional assumptions.
    * ``"clayton"`` -- fits a Clayton copula via maximum likelihood and
      derives ``lambda_L = 2^{-1/theta}`` (lower tail only).
    * ``"gumbel"`` -- fits a Gumbel copula via maximum likelihood and
      derives ``lambda_U = 2 - 2^{1/theta}`` (upper tail only).

    Parameters
    ----------
    method : str
        One of ``"empirical"``, ``"clayton"``, ``"gumbel"``.
        Default ``"empirical"``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported values.

    References
    ----------
    McNeil, Frey & Embrechts (2005). *QRM*, Chapter 5.
    Nelsen (2006). *An Introduction to Copulas*, 2nd ed.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(42)
    >>> df = pd.DataFrame(rng.standard_normal((500, 3)), columns=["A","B","C"])
    >>> td = TailDependence(method="empirical")
    >>> result = td.pairwise_tail_dependence(df)
    >>> result.adjusted_distance.shape
    (3, 3)
    """

    _VALID_METHODS: frozenset[str] = frozenset({"empirical", "clayton", "gumbel"})

    def __init__(self, method: str = "empirical") -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"Unknown method '{method}'. "
                f"Choose from: {sorted(self._VALID_METHODS)}"
            )
        self.method: str = method
        self._correlation_lookback: int = int(
            RISK_DEFAULTS.get("correlation_lookback", 60)
        )

    # ------------------------------------------------------------------
    # Empirical tail dependence
    # ------------------------------------------------------------------

    def empirical_tail_dependence(
        self,
        u: np.ndarray,
        v: np.ndarray,
        threshold: float = 0.05,
    ) -> tuple[float, float]:
        """Estimate lower and upper tail dependence from pseudo-observations.

        The estimator follows Schmidt & Stadtmuller (2006).  Given
        pseudo-observations ``(u_i, v_i)`` in [0, 1] (rank-transformed),
        the lower and upper tail dependence are estimated as:

            hat{lambda}_L(q) = #{u_i <= q and v_i <= q} / #{v_i <= q}
            hat{lambda}_U(q) = #{u_i > 1-q and v_i > 1-q} / #{v_i > 1-q}

        Parameters
        ----------
        u : np.ndarray
            1-D array of pseudo-observations for asset 1, in (0, 1).
        v : np.ndarray
            1-D array of pseudo-observations for asset 2, in (0, 1).
            Must have the same length as ``u``.
        threshold : float
            Quantile threshold ``q`` defining the tails.  Default 0.05
            means the bottom/top 5 % of observations.

        Returns
        -------
        tuple[float, float]
            ``(lambda_L, lambda_U)`` -- lower and upper tail dependence
            estimates, each in [0, 1].

        Reference
        ---------
        Schmidt, R. & Stadtmuller, U. (2006). "Non-parametric estimation
        of tail dependence." *Scand. J. Stat.*, 33(2), 307-335.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()

        if len(u) != len(v):
            raise ValueError(
                f"Length mismatch: u has {len(u)} elements, "
                f"v has {len(v)} elements."
            )
        if not 0.0 < threshold < 0.5:
            raise ValueError(
                f"threshold must be in (0, 0.5), got {threshold}"
            )

        n: int = len(u)

        # Lower tail: P(U <= q | V <= q)
        lower_mask_v: np.ndarray = v <= threshold
        n_lower_v: int = int(np.sum(lower_mask_v))
        if n_lower_v > 0:
            joint_lower: int = int(np.sum((u <= threshold) & lower_mask_v))
            lambda_l: float = joint_lower / n_lower_v
        else:
            lambda_l = 0.0

        # Upper tail: P(U > 1-q | V > 1-q)
        upper_thresh: float = 1.0 - threshold
        upper_mask_v: np.ndarray = v > upper_thresh
        n_upper_v: int = int(np.sum(upper_mask_v))
        if n_upper_v > 0:
            joint_upper: int = int(np.sum((u > upper_thresh) & upper_mask_v))
            lambda_u: float = joint_upper / n_upper_v
        else:
            lambda_u = 0.0

        logger.debug(
            "Empirical tail dep (n=%d, q=%.3f): lambda_L=%.4f, lambda_U=%.4f",
            n,
            threshold,
            lambda_l,
            lambda_u,
        )

        return lambda_l, lambda_u

    # ------------------------------------------------------------------
    # Clayton copula
    # ------------------------------------------------------------------

    @staticmethod
    def clayton_tail_dependence(theta: float) -> float:
        """Compute the lower tail dependence coefficient for a Clayton copula.

        Formula (Nelsen 2006, Theorem 5.4.2)
        -------------------------------------
            lambda_L = 2^{-1/theta}

        The Clayton copula C(u, v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}
        has lower tail dependence for theta > 0 and is asymptotically
        independent in the upper tail (lambda_U = 0).

        Parameters
        ----------
        theta : float
            Clayton copula parameter.  Must be > 0 for positive dependence
            and non-trivial lower tail dependence.

        Returns
        -------
        float
            Lower tail dependence coefficient in (0, 1).

        Raises
        ------
        ValueError
            If theta <= 0.

        Reference
        ---------
        Nelsen, R.B. (2006). *An Introduction to Copulas*, Theorem 5.4.2.
        McNeil, Frey & Embrechts (2005). *QRM*, Section 5.2.3.
        """
        if theta <= 0.0:
            raise ValueError(
                f"Clayton parameter theta must be > 0, got {theta}"
            )
        return float(2.0 ** (-1.0 / theta))

    # ------------------------------------------------------------------
    # Gumbel copula
    # ------------------------------------------------------------------

    @staticmethod
    def gumbel_tail_dependence(theta: float) -> float:
        """Compute the upper tail dependence coefficient for a Gumbel copula.

        Formula (Nelsen 2006, Theorem 5.4.6)
        -------------------------------------
            lambda_U = 2 - 2^{1/theta}

        The Gumbel copula C(u, v) = exp(-((- ln u)^theta + (- ln v)^theta)^{1/theta})
        has upper tail dependence for theta > 1 and is asymptotically
        independent in the lower tail (lambda_L = 0).  At theta = 1 the
        copula reduces to the independence copula (Pi).

        Parameters
        ----------
        theta : float
            Gumbel copula parameter.  Must be >= 1.

        Returns
        -------
        float
            Upper tail dependence coefficient in [0, 1).
            Returns 0.0 when theta = 1 (independence).

        Raises
        ------
        ValueError
            If theta < 1.

        Reference
        ---------
        Nelsen, R.B. (2006). *An Introduction to Copulas*, Theorem 5.4.6.
        McNeil, Frey & Embrechts (2005). *QRM*, Section 5.2.4.
        """
        if theta < 1.0:
            raise ValueError(
                f"Gumbel parameter theta must be >= 1, got {theta}"
            )
        return float(2.0 - 2.0 ** (1.0 / theta))

    # ------------------------------------------------------------------
    # MLE fitting: Clayton
    # ------------------------------------------------------------------

    def fit_clayton_theta(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> float:
        """Fit the Clayton copula parameter theta via maximum likelihood.

        The Clayton copula density is:

            c(u, v; theta) = (1 + theta) * (u * v)^{-(1 + theta)}
                             * (u^{-theta} + v^{-theta} - 1)^{-(2 + 1/theta)}

        The log-likelihood for n observations {(u_i, v_i)} is:

            LL(theta) = n * ln(1 + theta)
                        - (1 + theta) * sum_i [ln(u_i) + ln(v_i)]
                        - (2 + 1/theta) * sum_i ln(u_i^{-theta} + v_i^{-theta} - 1)

        We maximise LL(theta) over theta > 0 using bounded scalar
        optimisation (Brent's method).

        Parameters
        ----------
        u : np.ndarray
            1-D array of pseudo-observations for asset 1, in (0, 1).
        v : np.ndarray
            1-D array of pseudo-observations for asset 2, in (0, 1).

        Returns
        -------
        float
            Maximum likelihood estimate of theta.

        Reference
        ---------
        McNeil, Frey & Embrechts (2005). *QRM*, Section 5.5.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()

        # Clip to avoid log(0) and numerical blow-up
        eps: float = 1e-10
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)
        n: int = len(u)

        def neg_log_likelihood(theta: float) -> float:
            """Negative log-likelihood of the Clayton copula."""
            if theta <= 0.0:
                return 1e12
            term1: float = n * np.log(1.0 + theta)
            term2: float = -(1.0 + theta) * float(np.sum(np.log(u) + np.log(v)))
            inner: np.ndarray = u ** (-theta) + v ** (-theta) - 1.0
            # Guard against non-positive inner values (can happen at boundary)
            inner = np.clip(inner, eps, None)
            term3: float = -(2.0 + 1.0 / theta) * float(np.sum(np.log(inner)))
            return -(term1 + term2 + term3)

        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(1e-4, 50.0),
            method="bounded",
        )

        theta_hat: float = float(result.x)
        logger.debug(
            "Clayton MLE: theta=%.4f (converged=%s, nfev=%d)",
            theta_hat,
            result.success if hasattr(result, "success") else "N/A",
            result.nfev,
        )

        return theta_hat

    # ------------------------------------------------------------------
    # MLE fitting: Gumbel
    # ------------------------------------------------------------------

    def fit_gumbel_theta(
        self,
        u: np.ndarray,
        v: np.ndarray,
    ) -> float:
        """Fit the Gumbel copula parameter theta via maximum likelihood.

        The Gumbel copula density is:

            c(u, v; theta) = C(u,v) / (u * v)
                * ((-ln u)^theta + (-ln v)^theta)^{2/theta - 2}
                * (ln u * ln v)^{theta - 1}
                * [1 + (theta - 1) * ((-ln u)^theta + (-ln v)^theta)^{-1/theta}]

        where C(u,v) = exp(-((-ln u)^theta + (-ln v)^theta)^{1/theta}).

        We maximise the log-likelihood numerically over theta >= 1.

        Parameters
        ----------
        u : np.ndarray
            1-D array of pseudo-observations for asset 1, in (0, 1).
        v : np.ndarray
            1-D array of pseudo-observations for asset 2, in (0, 1).

        Returns
        -------
        float
            Maximum likelihood estimate of theta.

        Reference
        ---------
        McNeil, Frey & Embrechts (2005). *QRM*, Section 5.5.
        Genest, C. & Rivest, L.-P. (1993). "Statistical inference procedures
        for bivariate Archimedean copulas." *JASA*, 88(423), 1034-1043.
        """
        u = np.asarray(u, dtype=np.float64).ravel()
        v = np.asarray(v, dtype=np.float64).ravel()

        eps: float = 1e-10
        u = np.clip(u, eps, 1.0 - eps)
        v = np.clip(v, eps, 1.0 - eps)

        log_u: np.ndarray = -np.log(u)
        log_v: np.ndarray = -np.log(v)

        def neg_log_likelihood(theta: float) -> float:
            """Negative log-likelihood of the Gumbel copula."""
            if theta < 1.0:
                return 1e12

            a: np.ndarray = log_u ** theta + log_v ** theta
            a_safe: np.ndarray = np.clip(a, eps, None)

            # C(u,v) = exp(-a^{1/theta})
            a_inv_theta: np.ndarray = a_safe ** (1.0 / theta)
            log_C: np.ndarray = -a_inv_theta

            # log-density terms
            # term 1: log C(u,v)
            t1: float = float(np.sum(log_C))
            # term 2: -ln(u) - ln(v) (already positive: log_u, log_v)
            t2: float = -float(np.sum(np.log(u) + np.log(v)))
            # term 3: (2/theta - 2) * ln(a)
            t3: float = (2.0 / theta - 2.0) * float(np.sum(np.log(a_safe)))
            # term 4: (theta - 1) * ln(log_u * log_v)
            product: np.ndarray = np.clip(log_u * log_v, eps, None)
            t4: float = (theta - 1.0) * float(np.sum(np.log(product)))
            # term 5: ln(1 + (theta-1) * a^{-1/theta})
            bracket: np.ndarray = 1.0 + (theta - 1.0) * a_safe ** (-1.0 / theta)
            bracket = np.clip(bracket, eps, None)
            t5: float = float(np.sum(np.log(bracket)))

            ll: float = t1 + t2 + t3 + t4 + t5
            return -ll

        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(1.0, 50.0),
            method="bounded",
        )

        theta_hat: float = float(result.x)
        logger.debug(
            "Gumbel MLE: theta=%.4f (nfev=%d)",
            theta_hat,
            result.nfev,
        )

        return theta_hat

    # ------------------------------------------------------------------
    # Rank transformation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pseudo_observations(x: np.ndarray) -> np.ndarray:
        """Convert raw observations to pseudo-observations (rank / (n+1)).

        The division by ``(n+1)`` instead of ``n`` avoids boundary values
        of exactly 0 or 1, which would cause issues with copula likelihoods
        (Genest, Ghoudi & Rivest 1995).

        Parameters
        ----------
        x : np.ndarray
            1-D array of raw observations.

        Returns
        -------
        np.ndarray
            Pseudo-observations in (0, 1).

        Reference
        ---------
        Genest, C., Ghoudi, K. & Rivest, L.-P. (1995). "A semiparametric
        estimation procedure of dependence parameters in multivariate
        families of distributions." *Biometrika*, 82(3), 543-552.
        """
        n: int = len(x)
        return rankdata(x, method="ordinal") / (n + 1.0)

    # ------------------------------------------------------------------
    # Pairwise tail dependence matrix
    # ------------------------------------------------------------------

    def pairwise_tail_dependence(
        self,
        returns: pd.DataFrame,
        threshold: float = 0.05,
    ) -> TailDependenceResult:
        """Compute pairwise tail dependence and adjusted HRP distance matrix.

        Pipeline
        --------
        1. Rank-transform each column to pseudo-observations in (0, 1).
        2. For each pair (i, j), estimate lower and upper tail dependence
           using the configured ``method``.
        3. Compute the standard HRP distance matrix from Pearson correlation:
               d_std(i,j) = sqrt(0.5 * (1 - rho(i,j)))
        4. Adjust for tail dependence:
               tail_penalty(i,j) = max(lambda_L(i,j), lambda_U(i,j))
               d_adj(i,j) = d_std(i,j) * (1 + tail_penalty(i,j))
           This inflates distances for pairs that co-crash, causing HRP
           to treat them as more distinct and reduce their combined weight.

        Parameters
        ----------
        returns : pd.DataFrame
            (T, N) DataFrame of asset returns.  T >= 30 recommended for
            stable tail estimates.  Columns are asset names.
        threshold : float
            Quantile threshold for empirical tail estimation (default 0.05).
            Ignored when method is ``"clayton"`` or ``"gumbel"`` (the copula
            parameter determines tail dependence analytically).

        Returns
        -------
        TailDependenceResult
            Contains ``lower_tail``, ``upper_tail``, and ``adjusted_distance``
            matrices, all of shape (N, N).

        Reference
        ---------
        Lopez de Prado (2016), Eq. (2) for d_std.
        McNeil, Frey & Embrechts (2005), Chapter 5 for tail dependence.
        """
        if returns.shape[0] < 10:
            raise ValueError(
                f"Need at least 10 observations for tail dependence, "
                f"got {returns.shape[0]}."
            )

        n_assets: int = returns.shape[1]
        values: np.ndarray = returns.values

        # Step 1: Rank-transform to pseudo-observations
        pseudo: np.ndarray = np.column_stack(
            [self._to_pseudo_observations(values[:, j]) for j in range(n_assets)]
        )

        # Step 2: Pairwise tail dependence
        lower_tail: np.ndarray = np.zeros((n_assets, n_assets), dtype=np.float64)
        upper_tail: np.ndarray = np.zeros((n_assets, n_assets), dtype=np.float64)

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                u_i: np.ndarray = pseudo[:, i]
                v_j: np.ndarray = pseudo[:, j]

                if self.method == "empirical":
                    lam_l, lam_u = self.empirical_tail_dependence(
                        u_i, v_j, threshold=threshold
                    )
                elif self.method == "clayton":
                    theta: float = self.fit_clayton_theta(u_i, v_j)
                    lam_l = self.clayton_tail_dependence(theta)
                    lam_u = 0.0  # Clayton has no upper tail dependence
                elif self.method == "gumbel":
                    theta = self.fit_gumbel_theta(u_i, v_j)
                    lam_l = 0.0  # Gumbel has no lower tail dependence
                    lam_u = self.gumbel_tail_dependence(theta)
                else:
                    # Defensive: should never reach here due to __init__ check
                    lam_l, lam_u = 0.0, 0.0

                lower_tail[i, j] = lam_l
                lower_tail[j, i] = lam_l
                upper_tail[i, j] = lam_u
                upper_tail[j, i] = lam_u

        # Step 3: Standard HRP distance from Pearson correlation
        # d_std(i,j) = sqrt(0.5 * (1 - rho(i,j)))
        corr_matrix: np.ndarray = np.corrcoef(values, rowvar=False)
        # Numerical cleanup
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        np.fill_diagonal(corr_matrix, 1.0)

        standard_distance: np.ndarray = np.sqrt(0.5 * (1.0 - corr_matrix))
        np.fill_diagonal(standard_distance, 0.0)

        # Step 4: Adjusted distance = d_std * (1 + tail_penalty)
        tail_penalty: np.ndarray = np.maximum(lower_tail, upper_tail)
        adjusted_distance: np.ndarray = standard_distance * (1.0 + tail_penalty)
        np.fill_diagonal(adjusted_distance, 0.0)

        # Ensure symmetry (guard against floating point drift)
        adjusted_distance = (adjusted_distance + adjusted_distance.T) / 2.0
        np.fill_diagonal(adjusted_distance, 0.0)

        logger.info(
            "Tail dependence computed for %d assets (method=%s): "
            "mean lambda_L=%.4f, mean lambda_U=%.4f, "
            "mean d_adj=%.4f vs mean d_std=%.4f",
            n_assets,
            self.method,
            float(np.mean(lower_tail[np.triu_indices(n_assets, k=1)])),
            float(np.mean(upper_tail[np.triu_indices(n_assets, k=1)])),
            float(np.mean(adjusted_distance[np.triu_indices(n_assets, k=1)])),
            float(np.mean(standard_distance[np.triu_indices(n_assets, k=1)])),
        )

        return TailDependenceResult(
            lower_tail=lower_tail,
            upper_tail=upper_tail,
            adjusted_distance=adjusted_distance,
        )
