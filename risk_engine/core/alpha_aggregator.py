"""Inverse-variance weighted alpha combination.

Implements the optimal linear combination of multiple alpha signals using
inverse-variance weighting, as described in Grinold & Kahn (1999)
"Active Portfolio Management", Chapter 14.

The core idea: signals with lower estimation variance receive higher weight,
producing a combined alpha with minimum variance among all linear combinations.

Reference
---------
Grinold, R.C. & Kahn, R.N. (1999). *Active Portfolio Management: A Quantitative
Approach for Producing Superior Returns and Controlling Risk*, 2nd ed., McGraw-Hill.
Chapter 14: Portfolio Construction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregationResult:
    """Result container for alpha aggregation.

    Attributes
    ----------
    combined_alpha : float
        The inverse-variance weighted combined alpha.
    weights : dict[str, float]
        The weight assigned to each input signal.
    effective_n : float
        Effective number of independent signals (1 / sum(w_i^2)).
    """

    combined_alpha: float
    weights: dict[str, float]
    effective_n: float


class AlphaAggregator:
    """Combine alpha signals from multiple agents using inverse-variance weighting.

    Given K alpha signals {alpha_1, ..., alpha_K} with estimation variances
    {sigma_1^2, ..., sigma_K^2}, the optimal linear combination is:

        w_i = (1 / sigma_i^2) / sum_j(1 / sigma_j^2)

        alpha_combined = sum_i(w_i * alpha_i)

    This produces the minimum-variance unbiased estimator of the true alpha
    when signals are uncorrelated with a common mean.

    Parameters
    ----------
    min_variance : float
        Floor for variance values to prevent division-by-zero or
        numerically unstable weights. Default 1e-10.
    max_single_weight : float
        Maximum weight any single signal can receive (cap for
        concentration risk). Default 1.0 (no cap).

    Reference
    ---------
    Grinold & Kahn (1999), Chapter 14, Equation 14.3.
    """

    def __init__(
        self,
        min_variance: float = 1e-10,
        max_single_weight: float = 1.0,
    ) -> None:
        if min_variance <= 0:
            raise ValueError(f"min_variance must be positive, got {min_variance}")
        if not 0 < max_single_weight <= 1.0:
            raise ValueError(
                f"max_single_weight must be in (0, 1], got {max_single_weight}"
            )
        self.min_variance: float = min_variance
        self.max_single_weight: float = max_single_weight

    def combine(
        self,
        alphas: dict[str, float],
        variances: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Combine alphas using inverse-variance weighting.

        Formula
        -------
            w_i = (1 / sigma_i^2) / sum_j(1 / sigma_j^2)
            alpha_combined = sum_i(w_i * alpha_i)

        Parameters
        ----------
        alphas : dict[str, float]
            Mapping of signal_name -> alpha forecast.
        variances : dict[str, float]
            Mapping of signal_name -> estimation variance (sigma_i^2).
            Must contain all keys present in ``alphas``.

        Returns
        -------
        tuple[float, dict[str, float]]
            (combined_alpha, weights_used) where weights_used maps each
            signal name to its normalised weight.

        Raises
        ------
        ValueError
            If ``alphas`` is empty or keys mismatch with ``variances``.

        Reference
        ---------
        Grinold & Kahn (1999), Chapter 14, Eq. 14.3.
        """
        if not alphas:
            raise ValueError("alphas dict must not be empty")

        missing_keys = set(alphas.keys()) - set(variances.keys())
        if missing_keys:
            raise ValueError(
                f"Variance entries missing for signals: {missing_keys}"
            )

        # --- Single-agent shortcut ---
        if len(alphas) == 1:
            key = next(iter(alphas))
            return alphas[key], {key: 1.0}

        # --- Compute inverse-variance weights ---
        inv_var: dict[str, float] = {}
        for name in alphas:
            var_i = max(variances[name], self.min_variance)
            inv_var[name] = 1.0 / var_i

        total_inv_var: float = sum(inv_var.values())

        if total_inv_var <= 0:
            # Degenerate case: uniform weighting fallback
            logger.warning(
                "Total inverse-variance is non-positive; falling back to "
                "equal weights."
            )
            n = len(alphas)
            weights = {name: 1.0 / n for name in alphas}
        else:
            weights = {name: iv / total_inv_var for name, iv in inv_var.items()}

        # --- Apply concentration cap ---
        weights = self._cap_weights(weights)

        # --- Combine ---
        combined: float = sum(weights[name] * alphas[name] for name in alphas)

        # Handle all-zero alphas explicitly (result is 0.0, which is correct
        # mathematically, but worth logging).
        if all(v == 0.0 for v in alphas.values()):
            logger.info("All alpha signals are zero; combined alpha is 0.0.")

        return combined, weights

    def combine_with_confidence(
        self,
        alphas: dict[str, float],
        confidences: dict[str, float],
        variances: dict[str, float],
    ) -> tuple[float, dict[str, float]]:
        """Combine alphas scaled by confidence, then inverse-variance weighted.

        Each alpha is first scaled by its associated confidence score before
        the standard inverse-variance combination is applied:

            alpha_i_scaled = confidence_i * alpha_i

        This allows agents to self-report their conviction level, shrinking
        low-confidence forecasts toward zero.

        Parameters
        ----------
        alphas : dict[str, float]
            Mapping of signal_name -> raw alpha forecast.
        confidences : dict[str, float]
            Mapping of signal_name -> confidence in [0, 1].
        variances : dict[str, float]
            Mapping of signal_name -> estimation variance.

        Returns
        -------
        tuple[float, dict[str, float]]
            (combined_alpha, weights_used) after confidence scaling.

        Raises
        ------
        ValueError
            If keys mismatch or confidence values are out of range.
        """
        missing_conf = set(alphas.keys()) - set(confidences.keys())
        if missing_conf:
            raise ValueError(
                f"Confidence entries missing for signals: {missing_conf}"
            )

        for name, conf in confidences.items():
            if name in alphas and not (0.0 <= conf <= 1.0):
                raise ValueError(
                    f"Confidence for '{name}' must be in [0, 1], got {conf}"
                )

        scaled_alphas: dict[str, float] = {
            name: confidences[name] * alpha for name, alpha in alphas.items()
        }

        return self.combine(scaled_alphas, variances)

    def aggregate(
        self,
        alphas: dict[str, float],
        variances: dict[str, float],
    ) -> AggregationResult:
        """Full aggregation returning an ``AggregationResult`` with metadata.

        Parameters
        ----------
        alphas : dict[str, float]
            Signal name -> alpha forecast.
        variances : dict[str, float]
            Signal name -> estimation variance.

        Returns
        -------
        AggregationResult
        """
        combined, weights = self.combine(alphas, variances)

        # Effective N: reciprocal of Herfindahl index of weights
        hhi = sum(w ** 2 for w in weights.values())
        effective_n = 1.0 / hhi if hhi > 0 else 0.0

        return AggregationResult(
            combined_alpha=combined,
            weights=weights,
            effective_n=effective_n,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cap_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Iteratively cap weights at ``max_single_weight`` and renormalise.

        After capping, excess weight is redistributed proportionally among
        uncapped signals.  The loop terminates when no weight exceeds the cap
        (guaranteed to converge because each iteration fixes at least one
        weight).
        """
        if self.max_single_weight >= 1.0:
            return weights

        capped: dict[str, float] = dict(weights)
        for _ in range(len(weights)):
            excess = 0.0
            uncapped_names: list[str] = []
            for name, w in capped.items():
                if w > self.max_single_weight:
                    excess += w - self.max_single_weight
                    capped[name] = self.max_single_weight
                else:
                    uncapped_names.append(name)

            if excess <= 0:
                break

            if not uncapped_names:
                break

            uncapped_total = sum(capped[n] for n in uncapped_names)
            if uncapped_total <= 0:
                break

            for name in uncapped_names:
                capped[name] += excess * (capped[name] / uncapped_total)

        return capped
