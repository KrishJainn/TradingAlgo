"""
Meucci Entropy Pooling View Combiner.

Combines a prior distribution (market-implied) with subjective views
(agent forecasts) by finding the posterior distribution that is closest
to the prior in the Kullback-Leibler sense while satisfying the view
constraints exactly.

Reference:
    Meucci, A. (2008). "Fully Flexible Views: Theory and Practice."
    Risk, 21(10), 97-102.
    Available at SSRN: https://ssrn.com/abstract=1213325

Optimization problem:
    min_q  KL(q || p)  =  sum_j  q_j * ln(q_j / p_j)
    s.t.   H' q = h           (view constraints)
           1' q = 1           (probabilities sum to 1)
           q_j >= 0  for all j  (non-negativity)

Where:
    p = prior probability vector (J scenarios)
    q = posterior probability vector (J scenarios)
    H = view matrix (K views x J scenarios)
    h = view values (K x 1)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class EntropyPoolingResult:
    """Result of the entropy pooling view combination.

    Attributes:
        posterior_probs: Posterior probability vector over scenarios.
            Shape (n_scenarios,). Sums to 1.0.
        combined_alpha: Single scalar alpha (expected return) under the
            posterior distribution, combining all agent views.
        kl_divergence: Kullback-Leibler divergence KL(posterior || prior).
            Measures how far the posterior moved from the prior.
            Lower = views are more consistent with prior.
    """

    posterior_probs: np.ndarray
    combined_alpha: float
    kl_divergence: float


class EntropyPooling:
    """Entropy pooling view combiner following Meucci (2008).

    This class implements the Fully Flexible Views framework for combining
    a discrete prior distribution over market scenarios with linear view
    constraints from agent forecasts.

    The key advantage over Black-Litterman is generality: views can be
    expressed on any function of the scenarios (not just expected returns),
    and the prior can be any discrete distribution (not just Gaussian).

    Algorithm:
    1. Start with J equally-weighted (or market-implied) scenario probabilities.
    2. Express agent views as linear constraints on the scenario probabilities:
       E_q[h_k(X)] = view_k, i.e., H' q = h.
    3. Find q* that minimizes KL(q || p) subject to the view constraints.
    4. The dual problem has an analytical structure that makes SLSQP efficient.

    Reference:
        Meucci, A. (2008). "Fully Flexible Views: Theory and Practice."

    Example:
        >>> ep = EntropyPooling()
        >>> prior = np.ones(100) / 100  # uniform over 100 scenarios
        >>> H = np.random.randn(2, 100)  # 2 view portfolios
        >>> h = np.array([0.01, -0.005])  # expected view values
        >>> posterior = ep.fit(prior, H, h)
        >>> abs(posterior.sum() - 1.0) < 1e-10
        True
    """

    def __init__(self, max_iter: int = 1000, tol: float = 1e-10) -> None:
        """Initialize the entropy pooling optimizer.

        Args:
            max_iter: Maximum iterations for the SLSQP optimizer.
            tol: Convergence tolerance for the optimizer.
        """
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def _kl_divergence(q: np.ndarray, p: np.ndarray) -> float:
        """Compute KL(q || p) with numerical safeguards.

        KL(q || p) = sum_j q_j * ln(q_j / p_j)

        Args:
            q: Posterior probabilities. Shape (J,).
            p: Prior probabilities. Shape (J,).

        Returns:
            KL divergence (non-negative scalar).
        """
        # Avoid log(0) by clipping.
        q_safe = np.clip(q, 1e-15, None)
        p_safe = np.clip(p, 1e-15, None)
        return float(np.sum(q_safe * np.log(q_safe / p_safe)))

    def fit(
        self,
        prior_probs: np.ndarray,
        view_matrix: np.ndarray,
        view_values: np.ndarray,
    ) -> np.ndarray:
        """Solve the entropy pooling optimization.

        Finds the posterior distribution q* closest to the prior p in
        KL-divergence while satisfying the linear view constraints H'q = h.

        Args:
            prior_probs: Prior probability vector of shape (J,).
                Must be non-negative and sum to 1.
            view_matrix: View constraint matrix of shape (K, J), where
                K = number of views, J = number of scenarios.
                Row k defines the portfolio/function for view k.
            view_values: View target values of shape (K,).
                The posterior must satisfy E_q[H_k] = h_k.

        Returns:
            Posterior probability vector of shape (J,) that minimizes
            KL(q || p) subject to H'q = h and sum(q) = 1.

        Raises:
            ValueError: If input shapes are inconsistent or prior is invalid.
        """
        prior_probs = np.asarray(prior_probs, dtype=np.float64)
        view_matrix = np.asarray(view_matrix, dtype=np.float64)
        view_values = np.asarray(view_values, dtype=np.float64)

        n_scenarios = len(prior_probs)

        if prior_probs.ndim != 1:
            raise ValueError(f"prior_probs must be 1-D, got shape {prior_probs.shape}")
        if abs(prior_probs.sum() - 1.0) > 1e-6:
            raise ValueError(
                f"prior_probs must sum to 1.0, got {prior_probs.sum():.6f}"
            )
        if np.any(prior_probs < 0):
            raise ValueError("prior_probs must be non-negative")

        # Handle 1-D view (single view) case.
        if view_matrix.ndim == 1:
            view_matrix = view_matrix.reshape(1, -1)
        if view_values.ndim == 0:
            view_values = view_values.reshape(1)

        n_views, n_scen_v = view_matrix.shape
        if n_scen_v != n_scenarios:
            raise ValueError(
                f"view_matrix has {n_scen_v} columns but prior has {n_scenarios} scenarios"
            )
        if len(view_values) != n_views:
            raise ValueError(
                f"view_values has {len(view_values)} elements but view_matrix has {n_views} rows"
            )

        # --- Objective: minimize KL(q || p) ---
        p = prior_probs.copy()

        def objective(q: np.ndarray) -> float:
            return self._kl_divergence(q, p)

        def objective_grad(q: np.ndarray) -> np.ndarray:
            q_safe = np.clip(q, 1e-15, None)
            p_safe = np.clip(p, 1e-15, None)
            return np.log(q_safe / p_safe) + 1.0

        # --- Constraints ---
        constraints = []

        # Sum-to-one constraint: sum(q) = 1.
        constraints.append({
            "type": "eq",
            "fun": lambda q: np.sum(q) - 1.0,
            "jac": lambda q: np.ones(n_scenarios),
        })

        # View constraints: H @ q = h -> H[k, :] @ q - h[k] = 0 for each k.
        for k in range(n_views):
            row_k = view_matrix[k, :]
            val_k = view_values[k]
            constraints.append({
                "type": "eq",
                "fun": lambda q, r=row_k, v=val_k: float(r @ q - v),
                "jac": lambda q, r=row_k: r,
            })

        # Bounds: q_j >= 0.
        bounds = [(1e-15, 1.0) for _ in range(n_scenarios)]

        # Initial guess: start from the prior.
        q0 = p.copy()

        result = minimize(
            fun=objective,
            x0=q0,
            jac=objective_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        # Normalize to ensure exact summation to 1.
        posterior = np.clip(result.x, 1e-15, None)
        posterior /= posterior.sum()

        return posterior

    def combine_agent_views(
        self,
        agent_alphas: Dict[str, float],
        agent_confidences: Dict[str, float],
        prior_returns: np.ndarray,
    ) -> EntropyPoolingResult:
        """Combine agent views using entropy pooling.

        Convenience method that translates agent alpha forecasts and
        confidences into the view matrix / value formulation and runs
        the entropy pooling optimizer.

        Each agent's view is expressed as a constraint on the expected
        return under the posterior distribution, weighted by the agent's
        confidence. Specifically, the combined view is:

            E_q[returns] = sum_i (confidence_i * alpha_i) / sum_i(confidence_i)

        This is implemented as a single linear constraint on the posterior.

        Args:
            agent_alphas: Mapping of agent_name -> alpha forecast.
                Alpha is the agent's expected return signal.
            agent_confidences: Mapping of agent_name -> confidence in [0, 1].
                Must have the same keys as agent_alphas.
            prior_returns: Array of scenario returns, shape (J,).
                Each element is the return for one scenario in the
                prior distribution.

        Returns:
            EntropyPoolingResult with posterior probabilities, combined alpha,
            and KL divergence from prior.

        Raises:
            ValueError: If agent keys mismatch or inputs are invalid.
        """
        if set(agent_alphas.keys()) != set(agent_confidences.keys()):
            raise ValueError(
                "agent_alphas and agent_confidences must have identical keys"
            )
        if len(agent_alphas) == 0:
            raise ValueError("Must provide at least one agent view")

        prior_returns = np.asarray(prior_returns, dtype=np.float64)
        n_scenarios = len(prior_returns)

        # Uniform prior over scenarios.
        prior_probs = np.ones(n_scenarios, dtype=np.float64) / n_scenarios

        # Confidence-weighted consensus alpha.
        total_confidence = sum(agent_confidences.values())
        if total_confidence <= 0:
            # No confidence — return prior unchanged.
            return EntropyPoolingResult(
                posterior_probs=prior_probs,
                combined_alpha=float(prior_probs @ prior_returns),
                kl_divergence=0.0,
            )

        combined_view = sum(
            agent_confidences[agent] * agent_alphas[agent]
            for agent in agent_alphas
        ) / total_confidence

        # View constraint: E_q[returns] = combined_view.
        # H is a (1, J) matrix where H[0, j] = returns[j].
        # Then H @ q = sum_j returns[j] * q[j] = E_q[returns] = combined_view.
        view_matrix = prior_returns.reshape(1, -1)
        view_values = np.array([combined_view])

        # Run entropy pooling.
        posterior = self.fit(prior_probs, view_matrix, view_values)

        # Compute outputs.
        combined_alpha = float(posterior @ prior_returns)
        kl_div = self._kl_divergence(posterior, prior_probs)

        return EntropyPoolingResult(
            posterior_probs=posterior,
            combined_alpha=round(combined_alpha, 6),
            kl_divergence=round(kl_div, 6),
        )
