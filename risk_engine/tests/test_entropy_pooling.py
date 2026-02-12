"""Tests for the EntropyPooling view combiner (Meucci 2008).

Validates:
- fit() produces valid posterior probabilities
- KL divergence properties (non-negative, zero when posterior == prior)
- combine_agent_views() integrates agent alphas with confidence weighting
- Input validation (shape mismatches, invalid priors)
"""

import numpy as np
import pytest

from risk_engine.core.entropy_pooling import EntropyPooling, EntropyPoolingResult


class TestEntropyPoolingFit:
    """Tests for EntropyPooling.fit()."""

    def test_trivial_view_minimal_kl(self):
        """When view is already satisfied by the prior, KL divergence should be minimal."""
        ep = EntropyPooling()
        n = 100
        prior = np.ones(n) / n
        # View: expected value of a constant vector = that constant.
        # The prior already satisfies E_prior[0.5] = 0.5, so the optimizer
        # should return a posterior very close to the prior (small KL).
        scenarios = np.ones(n) * 0.5
        view_matrix = scenarios.reshape(1, -1)
        view_values = np.array([0.5])
        posterior = ep.fit(prior, view_matrix, view_values)
        assert posterior.shape == (n,)
        assert sum(posterior) == pytest.approx(1.0, abs=1e-6)
        # KL divergence should be very small since the prior already satisfies the view
        kl = EntropyPooling._kl_divergence(posterior, prior)
        assert kl < 0.01

    def test_posterior_sums_to_one(self):
        """Posterior probabilities must always sum to 1."""
        ep = EntropyPooling()
        n = 50
        prior = np.ones(n) / n
        # View: the expected value of the first 25 scenarios should be 0.03
        rng = np.random.RandomState(42)
        scenarios = rng.randn(n) * 0.02
        view_matrix = scenarios.reshape(1, -1)
        view_values = np.array([0.005])
        posterior = ep.fit(prior, view_matrix, view_values)
        assert sum(posterior) == pytest.approx(1.0, abs=1e-6)

    def test_posterior_non_negative(self):
        """All posterior probabilities must be non-negative."""
        ep = EntropyPooling()
        n = 80
        prior = np.ones(n) / n
        rng = np.random.RandomState(123)
        scenarios = rng.randn(n) * 0.01
        view_matrix = scenarios.reshape(1, -1)
        view_values = np.array([0.002])
        posterior = ep.fit(prior, view_matrix, view_values)
        assert np.all(posterior >= 0)

    def test_kl_divergence_non_negative(self):
        """KL(posterior || prior) must be >= 0 for any valid posterior."""
        ep = EntropyPooling()
        n = 50
        prior = np.ones(n) / n
        rng = np.random.RandomState(42)
        scenarios = rng.randn(n) * 0.02
        view_matrix = scenarios.reshape(1, -1)
        view_values = np.array([0.005])
        posterior = ep.fit(prior, view_matrix, view_values)
        kl = EntropyPooling._kl_divergence(posterior, prior)
        assert kl >= -1e-10  # allow tiny numerical error

    def test_kl_divergence_zero_when_no_shift(self):
        """KL divergence should be ~0 when posterior == prior."""
        ep = EntropyPooling()
        n = 50
        prior = np.ones(n) / n
        kl = EntropyPooling._kl_divergence(prior, prior)
        assert kl == pytest.approx(0.0, abs=1e-10)

    def test_view_constraint_satisfied(self):
        """The posterior should satisfy the view constraint E_q[X] = target."""
        ep = EntropyPooling()
        n = 100
        prior = np.ones(n) / n
        rng = np.random.RandomState(7)
        scenarios = rng.randn(n) * 0.02
        target = 0.01
        view_matrix = scenarios.reshape(1, -1)
        view_values = np.array([target])
        posterior = ep.fit(prior, view_matrix, view_values)
        realized = float(posterior @ scenarios)
        assert realized == pytest.approx(target, abs=1e-3)

    def test_multiple_views(self):
        """fit() should accept multiple simultaneous view constraints."""
        ep = EntropyPooling()
        n = 100
        prior = np.ones(n) / n
        rng = np.random.RandomState(42)
        # Two independent scenario dimensions
        s1 = rng.randn(n) * 0.02
        s2 = rng.randn(n) * 0.01
        view_matrix = np.vstack([s1.reshape(1, -1), s2.reshape(1, -1)])
        view_values = np.array([0.005, -0.002])
        posterior = ep.fit(prior, view_matrix, view_values)
        assert sum(posterior) == pytest.approx(1.0, abs=1e-6)
        assert float(posterior @ s1) == pytest.approx(0.005, abs=1e-3)
        assert float(posterior @ s2) == pytest.approx(-0.002, abs=1e-3)

    def test_1d_view_matrix_auto_reshaped(self):
        """A 1-D view_matrix should be automatically reshaped to (1, n)."""
        ep = EntropyPooling()
        n = 40
        prior = np.ones(n) / n
        rng = np.random.RandomState(42)
        scenarios = rng.randn(n) * 0.02
        view_values = np.array([0.003])
        # Pass 1-D instead of (1, n)
        posterior = ep.fit(prior, scenarios, view_values)
        assert posterior.shape == (n,)
        assert sum(posterior) == pytest.approx(1.0, abs=1e-6)

    def test_invalid_prior_not_sum_to_one(self):
        """Should raise ValueError if prior doesn't sum to 1."""
        ep = EntropyPooling()
        prior = np.array([0.5, 0.3])  # sums to 0.8
        with pytest.raises(ValueError, match="sum to 1"):
            ep.fit(prior, np.ones((1, 2)), np.array([1.0]))

    def test_invalid_prior_negative(self):
        """Should raise ValueError if prior has negative entries."""
        ep = EntropyPooling()
        prior = np.array([1.5, -0.5])  # negative entry
        with pytest.raises(ValueError, match="non-negative"):
            ep.fit(prior, np.ones((1, 2)), np.array([1.0]))

    def test_shape_mismatch_raises(self):
        """Should raise ValueError if view_matrix columns != prior length."""
        ep = EntropyPooling()
        prior = np.ones(10) / 10
        view_matrix = np.ones((1, 5))  # 5 != 10
        with pytest.raises(ValueError, match="columns"):
            ep.fit(prior, view_matrix, np.array([1.0]))


class TestEntropyPoolingCombineViews:
    """Tests for EntropyPooling.combine_agent_views()."""

    def test_basic_combine(self):
        """combine_agent_views returns an EntropyPoolingResult."""
        ep = EntropyPooling()
        rng = np.random.RandomState(42)
        prior_returns = rng.randn(100) * 0.02
        alphas = {"agent_A": 0.01, "agent_B": -0.005}
        confidences = {"agent_A": 0.8, "agent_B": 0.6}
        result = ep.combine_agent_views(alphas, confidences, prior_returns)
        assert isinstance(result, EntropyPoolingResult)
        assert result.posterior_probs.shape == (100,)
        assert sum(result.posterior_probs) == pytest.approx(1.0, abs=1e-6)

    def test_combined_alpha_is_float(self):
        """combined_alpha must be a scalar float."""
        ep = EntropyPooling()
        rng = np.random.RandomState(42)
        prior_returns = rng.randn(50) * 0.01
        alphas = {"a1": 0.005}
        confidences = {"a1": 0.9}
        result = ep.combine_agent_views(alphas, confidences, prior_returns)
        assert isinstance(result.combined_alpha, float)

    def test_kl_divergence_in_result(self):
        """KL divergence in the result must be non-negative."""
        ep = EntropyPooling()
        rng = np.random.RandomState(42)
        prior_returns = rng.randn(80) * 0.02
        alphas = {"a1": 0.01, "a2": 0.002}
        confidences = {"a1": 0.7, "a2": 0.5}
        result = ep.combine_agent_views(alphas, confidences, prior_returns)
        assert result.kl_divergence >= -1e-10

    def test_zero_confidence_returns_prior(self):
        """When all confidences are zero, posterior should match prior."""
        ep = EntropyPooling()
        rng = np.random.RandomState(42)
        n = 50
        prior_returns = rng.randn(n) * 0.01
        alphas = {"a1": 0.01}
        confidences = {"a1": 0.0}
        result = ep.combine_agent_views(alphas, confidences, prior_returns)
        expected_prior = np.ones(n) / n
        np.testing.assert_allclose(result.posterior_probs, expected_prior, atol=1e-8)
        assert result.kl_divergence == pytest.approx(0.0, abs=1e-10)

    def test_mismatched_keys_raises(self):
        """Should raise ValueError if alpha and confidence keys differ."""
        ep = EntropyPooling()
        prior_returns = np.random.randn(50) * 0.01
        alphas = {"a1": 0.01}
        confidences = {"a2": 0.8}
        with pytest.raises(ValueError, match="identical keys"):
            ep.combine_agent_views(alphas, confidences, prior_returns)

    def test_empty_views_raises(self):
        """Should raise ValueError if no agent views are provided."""
        ep = EntropyPooling()
        prior_returns = np.random.randn(50) * 0.01
        with pytest.raises(ValueError, match="at least one"):
            ep.combine_agent_views({}, {}, prior_returns)
