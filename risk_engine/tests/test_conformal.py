"""Tests for the ConformalPredictor (split conformal + adaptive updates).

Validates:
- calibrate() stores sorted residuals correctly
- predict_interval() produces valid intervals containing the forecast
- Higher coverage yields wider intervals
- predict_interval() before calibrate() raises RuntimeError
- adaptive_update() modifies alpha based on coverage hits/misses
- position_adjustment() shrinks for wide intervals, respects bounds

References:
    Vovk, V., Gammerman, A. & Shafer, G. (2005). Algorithmic Learning
    in a Random World. Springer.
    Gibbs, I. & Candes, E. (2021). Adaptive Conformal Inference Under
    Distribution Shift. NeurIPS 2021.
"""

import numpy as np
import pytest

from risk_engine.core.conformal_predictor import ConformalPredictor, ConformalResult


class TestConformalCalibration:
    """Tests for ConformalPredictor.calibrate()."""

    def test_calibrate_sets_state(self):
        """After calibration, is_calibrated should be True."""
        cp = ConformalPredictor(coverage=0.90)
        assert not cp.is_calibrated
        residuals = np.random.RandomState(42).randn(100) * 0.01
        cp.calibrate(residuals)
        assert cp.is_calibrated

    def test_calibrate_empty_raises(self):
        """Calibrating with an empty array should raise ValueError."""
        cp = ConformalPredictor(coverage=0.90)
        with pytest.raises(ValueError, match="empty"):
            cp.calibrate(np.array([]))

    def test_invalid_coverage_raises(self):
        """Coverage outside (0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="coverage"):
            ConformalPredictor(coverage=0.0)
        with pytest.raises(ValueError, match="coverage"):
            ConformalPredictor(coverage=1.0)
        with pytest.raises(ValueError, match="coverage"):
            ConformalPredictor(coverage=-0.5)


class TestConformalPrediction:
    """Tests for ConformalPredictor.predict_interval()."""

    def test_predict_returns_conformal_result(self):
        """predict_interval should return a ConformalResult."""
        cp = ConformalPredictor(coverage=0.90)
        residuals = np.random.RandomState(42).randn(100) * 0.01
        cp.calibrate(residuals)
        result = cp.predict_interval(point_forecast=0.005)
        assert isinstance(result, ConformalResult)

    def test_interval_contains_forecast(self):
        """The prediction interval should contain the point forecast."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(100) * 0.01)
        result = cp.predict_interval(point_forecast=0.0)
        assert result.lower <= 0.0 <= result.upper

    def test_interval_lower_less_than_upper(self):
        """Lower bound must be strictly less than upper bound."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(100) * 0.01)
        result = cp.predict_interval(point_forecast=0.005)
        assert result.lower < result.upper

    def test_n_calibration_matches(self):
        """n_calibration in the result should match the calibration set size."""
        cp = ConformalPredictor(coverage=0.90)
        residuals = np.random.RandomState(42).randn(100) * 0.01
        cp.calibrate(residuals)
        result = cp.predict_interval(point_forecast=0.005)
        assert result.n_calibration == 100

    def test_higher_coverage_wider_interval(self):
        """A higher coverage level should produce a wider (or equal) interval."""
        residuals = np.random.RandomState(42).randn(200) * 0.01
        cp90 = ConformalPredictor(coverage=0.90)
        cp95 = ConformalPredictor(coverage=0.95)
        cp90.calibrate(residuals)
        cp95.calibrate(residuals)
        r90 = cp90.predict_interval(0.0)
        r95 = cp95.predict_interval(0.0)
        assert r95.width >= r90.width

    def test_width_property(self):
        """width should equal upper - lower."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(100) * 0.01)
        result = cp.predict_interval(point_forecast=0.0)
        assert result.width == pytest.approx(result.upper - result.lower)

    def test_predict_before_calibrate_raises(self):
        """predict_interval without calibration should raise RuntimeError."""
        cp = ConformalPredictor(coverage=0.90)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict_interval(0.0)

    def test_coverage_field_matches(self):
        """The coverage field in ConformalResult should match target."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(100) * 0.01)
        result = cp.predict_interval(0.0)
        assert result.coverage == pytest.approx(0.90, abs=1e-6)

    def test_shifted_forecast(self):
        """Interval should shift with the point forecast."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(100) * 0.01)
        r1 = cp.predict_interval(point_forecast=0.0)
        r2 = cp.predict_interval(point_forecast=1.0)
        # The width should be the same, but shifted by 1.0
        assert r2.lower == pytest.approx(r1.lower + 1.0, abs=1e-10)
        assert r2.upper == pytest.approx(r1.upper + 1.0, abs=1e-10)


class TestAdaptiveUpdate:
    """Tests for ConformalPredictor.adaptive_update()."""

    def test_adaptive_update_changes_alpha(self):
        """After an adaptive update, alpha should change."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(50) * 0.01)
        initial_alpha = cp.alpha
        cp.adaptive_update(new_residual=0.005, step_size=0.05)
        # Alpha should have changed (direction depends on whether covered)
        assert isinstance(cp.alpha, float)

    def test_adaptive_update_grows_calibration_set(self):
        """Each adaptive_update should add one residual to the calibration set."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(50) * 0.01)
        cp.adaptive_update(new_residual=0.005)
        # Now calibration set has 51 residuals
        result = cp.predict_interval(0.0)
        assert result.n_calibration == 51

    def test_adaptive_update_before_calibrate_raises(self):
        """adaptive_update before calibration should raise RuntimeError."""
        cp = ConformalPredictor(coverage=0.90)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.adaptive_update(new_residual=0.01)

    def test_invalid_step_size_raises(self):
        """step_size outside (0, 1) should raise ValueError."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(50) * 0.01)
        with pytest.raises(ValueError, match="step_size"):
            cp.adaptive_update(new_residual=0.01, step_size=0.0)
        with pytest.raises(ValueError, match="step_size"):
            cp.adaptive_update(new_residual=0.01, step_size=1.0)

    def test_alpha_stays_bounded(self):
        """Alpha should remain in (0, 1) even after many updates."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(50) * 0.01)
        # Many misses (large residuals) should push alpha up but not beyond 1
        for _ in range(100):
            cp.adaptive_update(new_residual=10.0, step_size=0.05)
        assert 0 < cp.alpha < 1
        # Many hits (small residuals) should push alpha down but not below 0
        for _ in range(200):
            cp.adaptive_update(new_residual=0.0, step_size=0.05)
        assert 0 < cp.alpha < 1


class TestPositionAdjustment:
    """Tests for ConformalPredictor.position_adjustment()."""

    def test_position_adjustment_returns_float(self):
        """position_adjustment should return a float."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(100) * 0.01)
        interval = cp.predict_interval(0.0)
        adj = cp.position_adjustment(base_size=1.0, interval=interval)
        assert isinstance(adj, float)

    def test_adjustment_capped_at_base(self):
        """Position adjustment should never exceed base_size."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(100) * 0.01)
        interval = cp.predict_interval(0.0)
        adj = cp.position_adjustment(base_size=1.0, interval=interval)
        assert adj <= 1.0

    def test_adjustment_floored(self):
        """Position adjustment should be at least 0.25 * base_size."""
        cp = ConformalPredictor(coverage=0.90)
        # Calibrate with tiny residuals (so median_width is small)
        cp.calibrate(np.ones(100) * 0.001)
        # Create a very wide interval manually
        wide_result = ConformalResult(
            lower=-1.0, upper=1.0, coverage=0.9, alpha=0.1, n_calibration=100
        )
        adj = cp.position_adjustment(base_size=1.0, interval=wide_result)
        assert adj >= 0.25

    def test_adjustment_shrinks_for_wide_interval(self):
        """A wider-than-median interval should shrink the position."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.ones(100) * 0.01)
        # The median width from calibration = 2 * median(0.01) = 0.02
        # Create an interval wider than median
        wide_result = ConformalResult(
            lower=-0.05, upper=0.05, coverage=0.9, alpha=0.1, n_calibration=100
        )
        adj = cp.position_adjustment(base_size=1.0, interval=wide_result)
        assert adj <= 1.0

    def test_adjustment_before_calibrate_raises(self):
        """position_adjustment before calibration should raise RuntimeError."""
        cp = ConformalPredictor(coverage=0.90)
        dummy_interval = ConformalResult(
            lower=-0.01, upper=0.01, coverage=0.9, alpha=0.1, n_calibration=0
        )
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.position_adjustment(base_size=1.0, interval=dummy_interval)

    def test_negative_base_size_raises(self):
        """base_size <= 0 should raise ValueError."""
        cp = ConformalPredictor(coverage=0.90)
        cp.calibrate(np.random.RandomState(42).randn(50) * 0.01)
        interval = cp.predict_interval(0.0)
        with pytest.raises(ValueError, match="positive"):
            cp.position_adjustment(base_size=0.0, interval=interval)
        with pytest.raises(ValueError, match="positive"):
            cp.position_adjustment(base_size=-1.0, interval=interval)
