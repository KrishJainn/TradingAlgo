"""Tests for CostModel (NSE transaction costs)."""

import pytest

from risk_engine.core.cost_model import CostModel, CostBreakdown


class TestCostModel:
    def setup_method(self):
        self.cm = CostModel()

    def test_one_way_cost_returns_breakdown(self):
        result = self.cm.one_way_cost(price=2000.0, quantity=100, side="buy")
        assert isinstance(result, CostBreakdown)
        assert result.total > 0
        assert result.total_bps > 0

    def test_buy_has_stamp_duty(self):
        buy = self.cm.one_way_cost(price=2000.0, quantity=100, side="buy")
        sell = self.cm.one_way_cost(price=2000.0, quantity=100, side="sell")
        assert buy.stamp_duty > 0
        assert sell.stamp_duty == 0.0

    def test_intraday_stt_only_on_sell(self):
        buy = self.cm.one_way_cost(price=1000.0, quantity=500, side="buy", trade_type="intraday")
        sell = self.cm.one_way_cost(price=1000.0, quantity=500, side="sell", trade_type="intraday")
        assert buy.stt == 0.0
        assert sell.stt > 0

    def test_delivery_stt_on_both_sides(self):
        buy = self.cm.one_way_cost(price=1000.0, quantity=500, side="buy", trade_type="delivery")
        sell = self.cm.one_way_cost(price=1000.0, quantity=500, side="sell", trade_type="delivery")
        assert buy.stt > 0
        assert sell.stt > 0

    def test_round_trip_positive(self):
        rt = self.cm.round_trip_cost_bps(price=2000.0, quantity=100)
        assert rt > 0

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            self.cm.one_way_cost(price=100, quantity=10, side="hold")

    def test_alpha_threshold(self):
        result = CostModel.alpha_threshold(round_trip_bps=10.0, safety_margin=2.0)
        assert result == pytest.approx(20.0)

    def test_gst_computation(self):
        result = self.cm.one_way_cost(price=1000.0, quantity=100, side="buy")
        expected_gst = (result.brokerage + result.exchange) * 0.18
        assert result.gst == pytest.approx(expected_gst, rel=1e-6)
