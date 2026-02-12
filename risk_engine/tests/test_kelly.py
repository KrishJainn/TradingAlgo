"""Tests for PositionSizer.fractional_kelly()."""

import numpy as np
import pytest

from risk_engine.core.position_sizer import PositionSizer, PositionSizeResult


class TestFractionalKelly:
    def test_basic_kelly(self):
        """f* = fraction * mu / sigma^2"""
        ps = PositionSizer(kelly_fraction=0.25, max_position_pct=0.20)
        # mu=0.01, sigma=0.02 => full kelly = 0.01/0.0004 = 25, frac=6.25
        result = ps.fractional_kelly(mu=0.01, sigma=0.02)
        assert result == pytest.approx(0.25 * 0.01 / 0.02**2)

    def test_zero_sigma_returns_zero(self):
        ps = PositionSizer()
        assert ps.fractional_kelly(mu=0.01, sigma=0.0) == 0.0

    def test_negative_mu_returns_negative(self):
        ps = PositionSizer(kelly_fraction=1.0)
        result = ps.fractional_kelly(mu=-0.005, sigma=0.02)
        assert result < 0

    def test_fraction_override(self):
        ps = PositionSizer(kelly_fraction=0.25)
        result = ps.fractional_kelly(mu=0.01, sigma=0.02, fraction=0.5)
        assert result == pytest.approx(0.5 * 0.01 / 0.02**2)

    def test_nan_sigma_returns_zero(self):
        ps = PositionSizer()
        assert ps.fractional_kelly(mu=0.01, sigma=float('nan')) == 0.0

    def test_constructor_validation(self):
        with pytest.raises(ValueError):
            PositionSizer(kelly_fraction=0.0)
        with pytest.raises(ValueError):
            PositionSizer(kelly_fraction=1.5)
        with pytest.raises(ValueError):
            PositionSizer(max_position_pct=0.0)
