"""Tests for copula-based TailDependence estimation.

Validates:
- Clayton copula formula: lambda_L = 2^{-1/theta}
- Gumbel copula formula: lambda_U = 2 - 2^{1/theta}
- Empirical tail dependence returns valid [0, 1] coefficients
- Pairwise computation produces correctly shaped matrices
- Adjusted distance matrix properties (symmetry, non-negative off-diagonal)
- Clayton/Gumbel theta fitting via MLE

References:
    Nelsen, R.B. (2006). An Introduction to Copulas, Theorems 5.4.2 and 5.4.6.
    Schmidt, R. & Stadtmuller, U. (2006). Scand. J. Stat., 33(2), 307-335.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from risk_engine.core.tail_dependence import TailDependence, TailDependenceResult


class TestCopulaFormulae:
    """Tests for the closed-form copula tail dependence coefficients."""

    def test_clayton_formula_theta_2(self):
        """lambda_L = 2^{-1/theta} for theta=2 should be 2^{-0.5}."""
        result = TailDependence.clayton_tail_dependence(theta=2.0)
        expected = 2 ** (-1.0 / 2.0)
        assert result == pytest.approx(expected)

    def test_clayton_formula_theta_1(self):
        """lambda_L = 2^{-1/1} = 0.5 for theta=1."""
        result = TailDependence.clayton_tail_dependence(theta=1.0)
        assert result == pytest.approx(0.5)

    def test_clayton_large_theta_approaches_one(self):
        """As theta -> inf, lambda_L -> 1 (perfect lower tail dependence)."""
        result = TailDependence.clayton_tail_dependence(theta=100.0)
        assert result > 0.99

    def test_clayton_small_theta_approaches_zero(self):
        """As theta -> 0+, lambda_L -> 0 (tail independence)."""
        result = TailDependence.clayton_tail_dependence(theta=0.01)
        assert result < 0.01

    def test_clayton_invalid_theta_raises(self):
        """theta <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="theta must be > 0"):
            TailDependence.clayton_tail_dependence(theta=0.0)
        with pytest.raises(ValueError, match="theta must be > 0"):
            TailDependence.clayton_tail_dependence(theta=-1.0)

    def test_gumbel_formula_theta_2(self):
        """lambda_U = 2 - 2^{1/theta} for theta=2 should be 2 - sqrt(2)."""
        result = TailDependence.gumbel_tail_dependence(theta=2.0)
        expected = 2 - 2 ** (1.0 / 2.0)
        assert result == pytest.approx(expected)

    def test_gumbel_theta_1_independence(self):
        """At theta=1, Gumbel reduces to independence: lambda_U = 0."""
        result = TailDependence.gumbel_tail_dependence(theta=1.0)
        assert result == pytest.approx(0.0)

    def test_gumbel_large_theta_approaches_one(self):
        """As theta -> inf, lambda_U -> 1 (perfect upper tail dependence)."""
        result = TailDependence.gumbel_tail_dependence(theta=100.0)
        assert result > 0.99

    def test_gumbel_invalid_theta_raises(self):
        """theta < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="theta must be >= 1"):
            TailDependence.gumbel_tail_dependence(theta=0.5)


class TestEmpiricalTailDependence:
    """Tests for the non-parametric empirical tail dependence estimator."""

    def test_returns_tuple_of_two_floats(self):
        """empirical_tail_dependence should return (lambda_L, lambda_U)."""
        td = TailDependence(method="empirical")
        rng = np.random.RandomState(42)
        u = rng.uniform(0, 1, 500)
        v = rng.uniform(0, 1, 500)
        result = td.empirical_tail_dependence(u, v, threshold=0.1)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_values_in_zero_one_range(self):
        """Both coefficients must be in [0, 1]."""
        td = TailDependence(method="empirical")
        rng = np.random.RandomState(42)
        u = rng.uniform(0, 1, 500)
        v = rng.uniform(0, 1, 500)
        lower, upper = td.empirical_tail_dependence(u, v, threshold=0.1)
        assert 0.0 <= lower <= 1.0
        assert 0.0 <= upper <= 1.0

    def test_independent_data_low_tail_dep(self):
        """Independent uniform data should show low tail dependence."""
        td = TailDependence(method="empirical")
        rng = np.random.RandomState(42)
        u = rng.uniform(0, 1, 2000)
        v = rng.uniform(0, 1, 2000)
        lower, upper = td.empirical_tail_dependence(u, v, threshold=0.05)
        # For independent data, empirical tail dep should be near the threshold
        assert lower < 0.3
        assert upper < 0.3

    def test_perfectly_comonotone_data(self):
        """Perfectly comonotone data (u=v) should have high tail dependence."""
        td = TailDependence(method="empirical")
        n = 1000
        u = np.linspace(0.001, 0.999, n)
        v = u.copy()
        lower, upper = td.empirical_tail_dependence(u, v, threshold=0.05)
        assert lower > 0.9
        assert upper > 0.9

    def test_length_mismatch_raises(self):
        """Mismatched array lengths should raise ValueError."""
        td = TailDependence(method="empirical")
        with pytest.raises(ValueError, match="Length mismatch"):
            td.empirical_tail_dependence(np.array([0.1, 0.2]), np.array([0.1]))

    def test_invalid_threshold_raises(self):
        """Threshold outside (0, 0.5) should raise ValueError."""
        td = TailDependence(method="empirical")
        u = np.array([0.1, 0.5, 0.9])
        v = np.array([0.2, 0.4, 0.8])
        with pytest.raises(ValueError, match="threshold"):
            td.empirical_tail_dependence(u, v, threshold=0.6)


class TestPairwiseTailDependence:
    """Tests for the pairwise tail dependence matrix computation."""

    def test_returns_correct_type(self):
        """pairwise_tail_dependence should return a TailDependenceResult."""
        td = TailDependence(method="empirical")
        rng = np.random.RandomState(42)
        returns = pd.DataFrame(rng.randn(200, 3) * 0.02, columns=["A", "B", "C"])
        result = td.pairwise_tail_dependence(returns)
        assert isinstance(result, TailDependenceResult)

    def test_matrix_shapes(self):
        """All output matrices should be (n_assets, n_assets)."""
        td = TailDependence(method="empirical")
        rng = np.random.RandomState(42)
        returns = pd.DataFrame(rng.randn(200, 4) * 0.02, columns=["A", "B", "C", "D"])
        result = td.pairwise_tail_dependence(returns)
        assert result.lower_tail.shape == (4, 4)
        assert result.upper_tail.shape == (4, 4)
        assert result.adjusted_distance.shape == (4, 4)

    def test_adjusted_distance_diagonal_zero(self):
        """Diagonal of the adjusted distance matrix should be zero."""
        td = TailDependence(method="empirical")
        rng = np.random.RandomState(42)
        returns = pd.DataFrame(rng.randn(200, 3) * 0.02, columns=["A", "B", "C"])
        result = td.pairwise_tail_dependence(returns)
        for i in range(3):
            assert result.adjusted_distance[i, i] == pytest.approx(0.0)

    def test_adjusted_distance_off_diagonal_non_negative(self):
        """Off-diagonal entries of adjusted_distance must be >= 0."""
        td = TailDependence(method="empirical")
        rng = np.random.RandomState(42)
        returns = pd.DataFrame(rng.randn(200, 3) * 0.02, columns=["A", "B", "C"])
        result = td.pairwise_tail_dependence(returns)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert result.adjusted_distance[i, j] >= 0

    def test_adjusted_distance_symmetric(self):
        """Adjusted distance matrix should be symmetric."""
        td = TailDependence(method="empirical")
        rng = np.random.RandomState(42)
        returns = pd.DataFrame(rng.randn(200, 3) * 0.02, columns=["A", "B", "C"])
        result = td.pairwise_tail_dependence(returns)
        np.testing.assert_allclose(
            result.adjusted_distance, result.adjusted_distance.T, atol=1e-10
        )

    def test_tail_matrices_symmetric(self):
        """Lower and upper tail dependence matrices should be symmetric."""
        td = TailDependence(method="empirical")
        rng = np.random.RandomState(42)
        returns = pd.DataFrame(rng.randn(200, 3) * 0.02, columns=["A", "B", "C"])
        result = td.pairwise_tail_dependence(returns)
        np.testing.assert_allclose(
            result.lower_tail, result.lower_tail.T, atol=1e-10
        )
        np.testing.assert_allclose(
            result.upper_tail, result.upper_tail.T, atol=1e-10
        )

    def test_too_few_observations_raises(self):
        """Should raise ValueError with fewer than 10 observations."""
        td = TailDependence(method="empirical")
        returns = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
        with pytest.raises(ValueError, match="at least 10"):
            td.pairwise_tail_dependence(returns)


class TestThetaFitting:
    """Tests for Clayton and Gumbel MLE theta fitting."""

    def test_clayton_theta_positive(self):
        """Fitted Clayton theta should be positive for positively correlated data."""
        td = TailDependence(method="clayton")
        rng = np.random.RandomState(42)
        n = 500
        x = rng.randn(n)
        y = 0.7 * x + 0.3 * rng.randn(n)
        u = rankdata(x) / (n + 1)
        v = rankdata(y) / (n + 1)
        theta = td.fit_clayton_theta(u, v)
        assert theta > 0

    def test_gumbel_theta_at_least_one(self):
        """Fitted Gumbel theta should be >= 1."""
        td = TailDependence(method="gumbel")
        rng = np.random.RandomState(42)
        n = 500
        x = rng.randn(n)
        y = 0.7 * x + 0.3 * rng.randn(n)
        u = rankdata(x) / (n + 1)
        v = rankdata(y) / (n + 1)
        theta = td.fit_gumbel_theta(u, v)
        assert theta >= 1.0

    def test_invalid_method_raises(self):
        """Unknown method should raise ValueError at construction."""
        with pytest.raises(ValueError, match="Unknown method"):
            TailDependence(method="student_t")
