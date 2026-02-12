"""Tests for PositionSizer.ergodic_kelly() and size()."""

import numpy as np
import pytest

from risk_engine.core.position_sizer import PositionSizer, PositionSizeResult


class TestErgodicKelly:
    def test_ergodic_equals_full_kelly(self):
        """Ergodic kelly is f* = mu/sigma^2 (same as full Kelly in continuous time)"""
        ps = PositionSizer()
        result = ps.ergodic_kelly(mu=0.005, sigma=0.02)
        expected = 0.005 / 0.02**2
        assert result == pytest.approx(expected)

    def test_zero_sigma(self):
        ps = PositionSizer()
        assert ps.ergodic_kelly(mu=0.01, sigma=0.0) == 0.0


class TestSize:
    def test_size_pipeline(self):
        ps = PositionSizer(kelly_fraction=0.25, max_position_pct=0.20)
        result = ps.size(mu=0.003, sigma=0.02, signal_direction=1.0, signal_confidence=0.8, regime="bull")
        assert isinstance(result, PositionSizeResult)
        assert result.method_used == "fractional_kelly"
        assert -0.20 <= result.clamped_fraction <= 0.20

    def test_bear_regime_shrinks(self):
        # Use small mu so that fractional kelly stays below the cap,
        # allowing the bear regime shrinkage (0.5x) to be visible.
        ps = PositionSizer(kelly_fraction=0.25, max_position_pct=0.5)
        bull = ps.size(mu=0.0001, sigma=0.02, signal_direction=1.0, signal_confidence=1.0, regime="bull")
        bear = ps.size(mu=0.0001, sigma=0.02, signal_direction=1.0, signal_confidence=1.0, regime="bear")
        assert abs(bear.clamped_fraction) < abs(bull.clamped_fraction)

    def test_position_cap(self):
        ps = PositionSizer(kelly_fraction=1.0, max_position_pct=0.10)
        result = ps.size(mu=0.05, sigma=0.02, signal_direction=1.0, signal_confidence=1.0, regime="bull")
        assert abs(result.clamped_fraction) <= 0.10 + 1e-10

    def test_growth_rate_positive_for_good_trade(self):
        ps = PositionSizer(kelly_fraction=0.25, max_position_pct=0.5)
        result = ps.size(mu=0.01, sigma=0.02, signal_direction=1.0, signal_confidence=1.0, regime="bull")
        assert result.expected_growth_rate > 0


class TestMultiAssetKelly:
    def test_two_assets(self):
        ps = PositionSizer(kelly_fraction=0.25, max_position_pct=0.20)
        mu = np.array([0.01, 0.005])
        cov = np.array([[0.04, 0.01], [0.01, 0.02]])
        result = ps.multi_asset_kelly(mu, cov)
        assert result.shape == (2,)
        assert all(abs(r) <= 0.20 + 1e-10 for r in result)

    def test_shape_mismatch_raises(self):
        ps = PositionSizer()
        with pytest.raises(ValueError):
            ps.multi_asset_kelly(np.array([0.01, 0.02]), np.eye(3))
