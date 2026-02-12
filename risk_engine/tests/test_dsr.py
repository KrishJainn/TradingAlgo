"""Tests for DeflatedSharpe.

Validates the Deflated Sharpe Ratio implementation from
Bailey & Lopez de Prado (2014), including the False Strategy Theorem,
the DSR statistic itself, and the haircut function.
"""

import numpy as np
import pytest
from risk_engine.validation.deflated_sharpe import DeflatedSharpe, DSRResult


class TestDeflatedSharpe:
    def test_single_trial_no_bias(self):
        """With n_trials=1, expected max SR should be ~0 (no selection bias)."""
        expected = DeflatedSharpe.expected_max_sr(n_trials=1, T=252)
        assert expected == pytest.approx(0.0, abs=0.01)

    def test_more_trials_higher_expected_max(self):
        """More trials should increase the expected max SR under the null."""
        e1 = DeflatedSharpe.expected_max_sr(n_trials=10, T=252)
        e2 = DeflatedSharpe.expected_max_sr(n_trials=100, T=252)
        assert e2 > e1

    def test_compute_returns_result(self):
        """compute() should return a DSRResult with dsr in [0, 1]."""
        rng = np.random.RandomState(42)
        returns = rng.randn(252) * 0.01 + 0.0003
        dsr = DeflatedSharpe(n_trials=10, T=252)
        result = dsr.compute(returns)
        assert isinstance(result, DSRResult)
        assert 0.0 <= result.dsr <= 1.0

    def test_high_sharpe_is_significant(self):
        """A very high Sharpe (e.g., SR > 3) should survive deflation with few trials."""
        rng = np.random.RandomState(42)
        returns = rng.randn(252) * 0.01 + 0.003  # very high alpha
        dsr = DeflatedSharpe(n_trials=5, T=252)
        result = dsr.compute(returns)
        assert result.dsr > 0.5  # should still be significant

    def test_random_noise_not_significant(self):
        """Pure noise should not be significant after multiple testing correction."""
        rng = np.random.RandomState(42)
        returns = rng.randn(252) * 0.01  # zero mean
        dsr = DeflatedSharpe(n_trials=100, T=252)
        result = dsr.compute(returns)
        assert result.dsr < 0.5

    def test_haircut_between_zero_and_one(self):
        """Haircut factor should always lie in [0, 1]."""
        h = DeflatedSharpe.haircut_sharpe(observed_sr=2.0, n_trials=50, T=252)
        assert 0.0 <= h <= 1.0

    def test_haircut_negative_sr_returns_zero(self):
        """Negative observed SR should yield a haircut of 0.0."""
        h = DeflatedSharpe.haircut_sharpe(observed_sr=-1.0, n_trials=10, T=252)
        assert h == 0.0

    def test_haircut_zero_sr_returns_zero(self):
        """Zero observed SR should yield a haircut of 0.0."""
        h = DeflatedSharpe.haircut_sharpe(observed_sr=0.0, n_trials=10, T=252)
        assert h == 0.0

    def test_haircut_decreases_with_more_trials(self):
        """More trials should erode the haircut (more selection bias)."""
        h_few = DeflatedSharpe.haircut_sharpe(observed_sr=2.0, n_trials=5, T=252)
        h_many = DeflatedSharpe.haircut_sharpe(observed_sr=2.0, n_trials=500, T=252)
        assert h_few > h_many

    def test_expected_max_increases_with_trials(self):
        """Expected max SR should be monotonically increasing in n_trials."""
        prev = DeflatedSharpe.expected_max_sr(n_trials=1, T=252)
        for n in [2, 5, 10, 50, 100, 500]:
            curr = DeflatedSharpe.expected_max_sr(n_trials=n, T=252)
            assert curr >= prev, f"expected_max_sr decreased from n={n-1} to n={n}"
            prev = curr

    def test_longer_sample_reduces_expected_max(self):
        """Longer sample (larger T) should reduce expected max SR (more precise estimates)."""
        e_short = DeflatedSharpe.expected_max_sr(n_trials=50, T=100)
        e_long = DeflatedSharpe.expected_max_sr(n_trials=50, T=1000)
        assert e_short > e_long

    def test_result_has_all_fields(self):
        """DSRResult should contain all expected attributes."""
        rng = np.random.RandomState(42)
        returns = rng.randn(252) * 0.01 + 0.0005
        dsr = DeflatedSharpe(n_trials=10, T=252)
        result = dsr.compute(returns)
        assert hasattr(result, "dsr")
        assert hasattr(result, "observed_sr")
        assert hasattr(result, "expected_max_sr")
        assert hasattr(result, "haircut")
        assert hasattr(result, "is_significant")

    def test_is_significant_flag(self):
        """is_significant should be True when dsr >= 0.95."""
        rng = np.random.RandomState(42)
        # Very strong signal, few trials -> should be significant
        returns = rng.randn(504) * 0.01 + 0.005
        dsr = DeflatedSharpe(n_trials=2, T=504)
        result = dsr.compute(returns)
        assert result.is_significant is True
        assert result.dsr >= 0.95

    def test_invalid_n_trials_raises(self):
        """n_trials < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            DeflatedSharpe(n_trials=0, T=252)

    def test_invalid_T_raises(self):
        """T < 2 should raise ValueError."""
        with pytest.raises(ValueError):
            DeflatedSharpe(n_trials=10, T=1)

    def test_constant_returns_raises(self):
        """All-constant returns (zero variance) should raise ValueError."""
        returns = np.ones(100) * 0.001
        dsr = DeflatedSharpe(n_trials=10, T=100)
        with pytest.raises(ValueError, match="near-zero standard deviation"):
            dsr.compute(returns)
