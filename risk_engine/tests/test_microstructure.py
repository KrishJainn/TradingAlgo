"""Tests for CostModel.get_microstructure_multiplier().

Validates intraday microstructure slippage multipliers based on
NSE trading session patterns and expiry-day adjustments.
"""

import pytest
from risk_engine.core.cost_model import CostModel


class TestMicrostructure:
    def setup_method(self):
        self.cm = CostModel()

    def test_midday_is_lowest(self):
        """11:00 AM - 2:00 PM should have multiplier 1.0 (calm zone)."""
        mult = self.cm.get_microstructure_multiplier(hour=12.0)
        assert mult == pytest.approx(1.0)

    def test_morning_higher_than_midday(self):
        """9:15-10:00 AM (morning) should have higher multiplier than midday."""
        morning = self.cm.get_microstructure_multiplier(hour=9.5)
        midday = self.cm.get_microstructure_multiplier(hour=12.0)
        assert morning > midday

    def test_pre_open_highest_non_expiry(self):
        """Before 9:15 AM (pre-open) should have multiplier 3.0."""
        pre_open = self.cm.get_microstructure_multiplier(hour=9.0)
        assert pre_open == pytest.approx(3.0)

    def test_closing_multiplier(self):
        """2:30-3:30 PM closing session should have multiplier 2.0."""
        closing = self.cm.get_microstructure_multiplier(hour=15.0)
        assert closing == pytest.approx(2.0)

    def test_expiry_day_adds_multiplier(self):
        """Expiry day should increase the multiplier versus a normal day."""
        normal = self.cm.get_microstructure_multiplier(hour=12.0, is_expiry=False)
        expiry = self.cm.get_microstructure_multiplier(hour=12.0, is_expiry=True)
        assert expiry > normal

    def test_expiry_last_30min(self):
        """Last 30 min on expiry day: closing_mult * expiry_last_30min_mult = 2.0 * 2.5 = 5.0."""
        expiry_close = self.cm.get_microstructure_multiplier(hour=15.0, is_expiry=True)
        assert expiry_close == pytest.approx(2.0 * 2.5)

    def test_late_morning(self):
        """10:00-11:00 AM should have late_morning_mult = 1.5."""
        mult = self.cm.get_microstructure_multiplier(hour=10.5)
        assert mult == pytest.approx(1.5)

    def test_pre_close(self):
        """2:00-2:30 PM should have pre_close_multiplier = 1.3."""
        mult = self.cm.get_microstructure_multiplier(hour=14.25)
        assert mult == pytest.approx(1.3)

    def test_morning_multiplier_value(self):
        """9:15-10:00 AM should have morning_multiplier = 2.5."""
        mult = self.cm.get_microstructure_multiplier(hour=9.5)
        assert mult == pytest.approx(2.5)

    def test_after_hours_uses_closing(self):
        """After 3:30 PM should fall back to closing_multiplier = 2.0."""
        mult = self.cm.get_microstructure_multiplier(hour=16.0)
        assert mult == pytest.approx(2.0)

    def test_expiry_midday_uses_expiry_day_multiplier(self):
        """Midday on expiry (before 3 PM): midday_mult * expiry_day_mult = 1.0 * 1.5."""
        mult = self.cm.get_microstructure_multiplier(hour=12.0, is_expiry=True)
        assert mult == pytest.approx(1.0 * 1.5)

    def test_expiry_morning_uses_expiry_day_multiplier(self):
        """Morning on expiry: morning_mult * expiry_day_mult = 2.5 * 1.5."""
        mult = self.cm.get_microstructure_multiplier(hour=9.5, is_expiry=True)
        assert mult == pytest.approx(2.5 * 1.5)

    def test_boundary_at_9_25(self):
        """Exactly 9.25 should be morning (not pre-open)."""
        mult = self.cm.get_microstructure_multiplier(hour=9.25)
        assert mult == pytest.approx(2.5)

    def test_boundary_at_10(self):
        """Exactly 10.0 should be late_morning (not morning)."""
        mult = self.cm.get_microstructure_multiplier(hour=10.0)
        assert mult == pytest.approx(1.5)

    def test_boundary_at_11(self):
        """Exactly 11.0 should be midday (not late_morning)."""
        mult = self.cm.get_microstructure_multiplier(hour=11.0)
        assert mult == pytest.approx(1.0)

    def test_boundary_at_14(self):
        """Exactly 14.0 should be pre_close (not midday)."""
        mult = self.cm.get_microstructure_multiplier(hour=14.0)
        assert mult == pytest.approx(1.3)

    def test_boundary_at_14_5(self):
        """Exactly 14.5 should be closing (not pre_close)."""
        mult = self.cm.get_microstructure_multiplier(hour=14.5)
        assert mult == pytest.approx(2.0)

    def test_multiplier_always_at_least_one(self):
        """All multipliers across the day should be >= 1.0."""
        for hour in [9.0, 9.25, 9.5, 10.0, 10.5, 11.0, 12.0, 13.0, 14.0, 14.5, 15.0, 15.5]:
            mult = self.cm.get_microstructure_multiplier(hour=hour)
            assert mult >= 1.0, f"Multiplier at hour={hour} is {mult}, expected >= 1.0"
