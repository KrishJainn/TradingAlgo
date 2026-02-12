"""Integration tests tying the full risk engine pipeline together.

Validates:
- CoachInterface.process_signals() returns a valid CoachDecision
- CoachDecision fields are well-formed (exposure >= 0, weights present)
- Custom RiskParams are respected
- Online weight updates work after initial processing
- get_risk_report() returns expected keys
- AgentSignal validation (direction/confidence bounds)
- Sizing module convenience API (size_positions / reset)
"""

import numpy as np
import pandas as pd
import pytest

from risk_engine.integration.agent_interface import AgentSignal
from risk_engine.integration.coach_interface import CoachInterface, CoachDecision
from risk_engine.config.risk_params import RiskParams


class TestAgentSignal:
    """Tests for the AgentSignal dataclass."""

    def test_valid_signal_creation(self):
        """A valid signal should be constructable and retain its fields."""
        sig = AgentSignal(
            agent_id="PLAYER_1",
            direction=0.5,
            confidence=0.8,
            alpha_estimate=0.001,
            volatility_estimate=0.02,
            metadata={"regime": "Bull"},
        )
        assert sig.agent_id == "PLAYER_1"
        assert sig.direction == 0.5
        assert sig.confidence == 0.8

    def test_conviction_alpha(self):
        """conviction_alpha should be direction * confidence * alpha_estimate."""
        sig = AgentSignal(
            agent_id="PLAYER_1",
            direction=0.5,
            confidence=0.8,
            alpha_estimate=0.001,
            volatility_estimate=0.02,
        )
        expected = 0.5 * 0.8 * 0.001
        assert sig.conviction_alpha == pytest.approx(expected)

    def test_variance_estimate(self):
        """variance_estimate should be volatility_estimate**2."""
        sig = AgentSignal(
            agent_id="PLAYER_1",
            direction=0.5,
            confidence=0.8,
            alpha_estimate=0.001,
            volatility_estimate=0.02,
        )
        assert sig.variance_estimate == pytest.approx(0.02 ** 2)

    def test_is_actionable(self):
        """A signal with sufficient direction, confidence, alpha, and vol is actionable."""
        sig = AgentSignal(
            agent_id="PLAYER_1",
            direction=0.5,
            confidence=0.8,
            alpha_estimate=0.001,
            volatility_estimate=0.02,
        )
        assert sig.is_actionable is True

    def test_not_actionable_low_direction(self):
        """A signal with near-zero direction is not actionable."""
        sig = AgentSignal(
            agent_id="PLAYER_1",
            direction=0.01,
            confidence=0.8,
            alpha_estimate=0.001,
            volatility_estimate=0.02,
        )
        assert sig.is_actionable is False

    def test_direction_out_of_range_raises(self):
        """direction outside [-1, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="direction"):
            AgentSignal(
                agent_id="PLAYER_1",
                direction=1.5,
                confidence=0.8,
                alpha_estimate=0.001,
                volatility_estimate=0.02,
            )

    def test_confidence_out_of_range_raises(self):
        """confidence outside [0, 1] should raise ValueError."""
        with pytest.raises(ValueError, match="confidence"):
            AgentSignal(
                agent_id="PLAYER_1",
                direction=0.5,
                confidence=1.5,
                alpha_estimate=0.001,
                volatility_estimate=0.02,
            )


class TestCoachInterface:
    """Tests for the CoachInterface pipeline."""

    @staticmethod
    def _make_signals(n_agents=5, rng=None):
        """Helper to create a list of valid, actionable AgentSignals."""
        if rng is None:
            rng = np.random.RandomState(42)
        signals = []
        for i in range(n_agents):
            direction = rng.uniform(0.2, 0.9)  # ensure actionable (|dir| > 0.05)
            signals.append(AgentSignal(
                agent_id=f"PLAYER_{i + 1}",
                direction=direction,
                confidence=rng.uniform(0.3, 0.9),
                alpha_estimate=abs(direction) * 0.002,
                volatility_estimate=0.02,
                metadata={"regime": "Bull"},
            ))
        return signals

    @staticmethod
    def _make_market_data(n_bars=100, rng=None):
        """Helper to create synthetic OHLCV market data."""
        if rng is None:
            rng = np.random.RandomState(42)
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="D")
        close = 100 * np.exp(np.cumsum(rng.randn(n_bars) * 0.01))
        return pd.DataFrame({
            "Close": close,
            "Open": close * (1 + rng.randn(n_bars) * 0.001),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Volume": rng.randint(100000, 1000000, n_bars),
        }, index=dates)

    def test_process_signals_returns_decision(self):
        """process_signals should return a CoachDecision."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = coach.process_signals(
            signals=signals,
            market_data=market_data,
            portfolio_value=10_000_000,
        )
        assert isinstance(decision, CoachDecision)

    def test_decision_has_agent_weights(self):
        """CoachDecision should contain non-empty agent_weights."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 10_000_000)
        assert len(decision.agent_weights) > 0

    def test_decision_total_exposure_is_float(self):
        """total_exposure should be a numeric value."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 10_000_000)
        assert isinstance(decision.total_exposure, (int, float))

    def test_decision_has_position_sizes(self):
        """CoachDecision should contain position_sizes for active agents."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 10_000_000)
        assert isinstance(decision.position_sizes, dict)

    def test_decision_cost_estimate_non_negative(self):
        """cost_estimate should be >= 0."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 10_000_000)
        assert decision.cost_estimate >= 0

    def test_decision_conformal_interval_tuple(self):
        """conformal_interval should be a tuple of (lower, upper)."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 10_000_000)
        assert isinstance(decision.conformal_interval, tuple)
        assert len(decision.conformal_interval) == 2

    def test_decision_timestamp_is_string(self):
        """timestamp should be an ISO-format string."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 10_000_000)
        assert isinstance(decision.timestamp, str)
        assert "T" in decision.timestamp  # ISO-8601 format

    def test_decision_to_dict(self):
        """to_dict() should return a plain dictionary."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 10_000_000)
        d = decision.to_dict()
        assert isinstance(d, dict)
        assert "agent_weights" in d
        assert "position_sizes" in d

    def test_flat_decision_when_no_actionable(self):
        """When no signals are actionable, should return a flat decision."""
        coach = CoachInterface()
        # Create signals with near-zero direction (not actionable)
        signals = [
            AgentSignal(
                agent_id="PLAYER_1",
                direction=0.01,  # below actionability threshold
                confidence=0.1,
                alpha_estimate=0.0001,
                volatility_estimate=0.02,
            )
        ]
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 10_000_000)
        assert decision.total_exposure == 0.0

    def test_coach_with_custom_params(self):
        """CoachInterface should accept custom RiskParams."""
        params = RiskParams(kelly_fraction=0.10, max_single_position_pct=0.15)
        coach = CoachInterface(params=params)
        signals = self._make_signals(n_agents=3)
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 5_000_000)
        assert isinstance(decision, CoachDecision)

    def test_update_online_weights(self):
        """update_online_weights should not raise for valid input."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        coach.process_signals(signals, market_data, 10_000_000)
        # Update with realized returns
        returns = {f"PLAYER_{i + 1}": np.random.uniform(-0.01, 0.01) for i in range(5)}
        coach.update_online_weights(returns)
        # No exception means success

    def test_update_online_weights_empty_noop(self):
        """update_online_weights with empty dict should be a no-op."""
        coach = CoachInterface()
        coach.update_online_weights({})
        # No exception means success

    def test_risk_report_keys(self):
        """get_risk_report should return a dict with expected keys."""
        coach = CoachInterface()
        report = coach.get_risk_report()
        assert isinstance(report, dict)
        assert "online_weights" in report
        assert "hrp_weights" in report
        assert "params" in report
        assert "daily_pnl" in report
        assert "peak_nav" in report
        assert "bar_count" in report

    def test_gross_exposure_property(self):
        """CoachDecision.gross_exposure should be sum of abs(position_sizes)."""
        coach = CoachInterface()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = coach.process_signals(signals, market_data, 10_000_000)
        expected_gross = sum(abs(v) for v in decision.position_sizes.values())
        assert decision.gross_exposure == pytest.approx(expected_gross, abs=1e-10)


class TestSizingModule:
    """Tests for the risk_engine.sizing convenience API."""

    @staticmethod
    def _make_signals(n_agents=5, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        signals = []
        for i in range(n_agents):
            direction = rng.uniform(0.2, 0.9)
            signals.append(AgentSignal(
                agent_id=f"PLAYER_{i + 1}",
                direction=direction,
                confidence=rng.uniform(0.3, 0.9),
                alpha_estimate=abs(direction) * 0.002,
                volatility_estimate=0.02,
                metadata={"regime": "Bull"},
            ))
        return signals

    @staticmethod
    def _make_market_data(n_bars=100, rng=None):
        if rng is None:
            rng = np.random.RandomState(42)
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="D")
        close = 100 * np.exp(np.cumsum(rng.randn(n_bars) * 0.01))
        return pd.DataFrame({
            "Close": close,
            "Open": close * (1 + rng.randn(n_bars) * 0.001),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Volume": rng.randint(100000, 1000000, n_bars),
        }, index=dates)

    def test_size_positions(self):
        """size_positions should return a CoachDecision."""
        from risk_engine.sizing import size_positions, reset
        reset()
        signals = self._make_signals()
        market_data = self._make_market_data()
        decision = size_positions(signals, market_data, 10_000_000)
        assert isinstance(decision, CoachDecision)

    def test_reset_clears_state(self):
        """reset() should clear the module-level singleton."""
        from risk_engine.sizing import size_positions, reset
        reset()
        signals = self._make_signals()
        market_data = self._make_market_data()
        # First call initializes internal coach
        size_positions(signals, market_data, 10_000_000)
        # Reset and call again -- should not raise
        reset()
        decision = size_positions(signals, market_data, 10_000_000)
        assert isinstance(decision, CoachDecision)

    def test_get_risk_report(self):
        """get_risk_report should return a dict after initialization."""
        from risk_engine.sizing import get_risk_report, reset
        reset()
        # Calling get_risk_report on uninitialized coach should still work
        # (it initializes lazily)
        report = get_risk_report()
        assert isinstance(report, dict)

    def test_update_weights(self):
        """update_weights should not raise for valid input."""
        from risk_engine.sizing import size_positions, update_weights, reset
        reset()
        signals = self._make_signals()
        market_data = self._make_market_data()
        size_positions(signals, market_data, 10_000_000)
        returns = {f"PLAYER_{i + 1}": 0.001 for i in range(5)}
        update_weights(returns)
        # No exception means success
