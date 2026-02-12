"""Tests for DisagreementSignal.

Validates the Shannon entropy-based disagreement measure and the
confidence-weighted dispersion calculation for agent consensus detection.
"""

import math
import pytest
from risk_engine.core.disagreement_signal import DisagreementSignal, DisagreementResult


class TestDisagreement:
    def test_unanimous_bullish_low_entropy(self):
        """All agents bullish -> low entropy, not high disagreement, multiplier 1.0."""
        signal = DisagreementSignal()
        directions = {"p1": 0.8, "p2": 0.7, "p3": 0.9}
        confidences = {"p1": 0.9, "p2": 0.8, "p3": 0.85}
        result = signal.compute(directions, confidences)
        assert result.entropy < 0.5
        assert result.is_high_disagreement is False
        assert result.position_size_multiplier == 1.0
        # All in strong_bull bin (>= 0.5)
        assert result.n_clusters == 1

    def test_mixed_signals_high_entropy(self):
        """Agents in different bins -> higher entropy and dispersion."""
        signal = DisagreementSignal()
        # p1=0.8 -> strong_bull, p2=-0.6 -> strong_bear, p3=0.1 -> mild_bull
        directions = {"p1": 0.8, "p2": -0.6, "p3": 0.1}
        confidences = {"p1": 0.9, "p2": 0.7, "p3": 0.5}
        result = signal.compute(directions, confidences)
        assert result.n_clusters >= 2
        assert result.dispersion > 0

    def test_full_disagreement(self):
        """Agents spread across all bins -> is_high_disagreement=True, multiplier=0.5."""
        signal = DisagreementSignal()
        # 0.8 -> strong_bull, 0.3 -> mild_bull, 0.0 -> neutral,
        # -0.3 -> mild_bear, -0.8 -> strong_bear
        directions = {"p1": 0.8, "p2": 0.3, "p3": 0.0, "p4": -0.3, "p5": -0.8}
        confidences = {"p1": 0.8, "p2": 0.8, "p3": 0.8, "p4": 0.8, "p5": 0.8}
        result = signal.compute(directions, confidences)
        assert result.is_high_disagreement is True
        assert result.position_size_multiplier == 0.5

    def test_empty_directions_raises(self):
        """Empty directions dict should raise ValueError."""
        signal = DisagreementSignal()
        with pytest.raises(ValueError):
            signal.compute({}, {})

    def test_mismatched_keys_raises(self):
        """Mismatched keys between directions and confidences should raise ValueError."""
        signal = DisagreementSignal()
        with pytest.raises(ValueError):
            signal.compute({"p1": 0.5}, {"p2": 0.5})

    def test_returns_disagreement_result(self):
        """compute() should return a DisagreementResult instance."""
        signal = DisagreementSignal()
        directions = {"p1": 0.3, "p2": -0.2}
        confidences = {"p1": 0.7, "p2": 0.6}
        result = signal.compute(directions, confidences)
        assert isinstance(result, DisagreementResult)

    def test_single_agent_zero_entropy(self):
        """Single agent should produce zero entropy (only one bin occupied)."""
        signal = DisagreementSignal()
        directions = {"only_agent": 0.5}
        confidences = {"only_agent": 0.9}
        result = signal.compute(directions, confidences)
        assert result.entropy == pytest.approx(0.0)
        assert result.n_clusters == 1
        assert result.is_high_disagreement is False

    def test_single_agent_zero_dispersion(self):
        """Single agent should have zero dispersion (no spread of opinions)."""
        signal = DisagreementSignal()
        directions = {"solo": 0.6}
        confidences = {"solo": 0.8}
        result = signal.compute(directions, confidences)
        assert result.dispersion == pytest.approx(0.0)

    def test_two_agents_same_bin_zero_entropy(self):
        """Two agents in the same bin should produce zero entropy."""
        signal = DisagreementSignal()
        # Both in strong_bull
        directions = {"a": 0.6, "b": 0.9}
        confidences = {"a": 0.8, "b": 0.7}
        result = signal.compute(directions, confidences)
        assert result.entropy == pytest.approx(0.0)
        assert result.n_clusters == 1

    def test_two_agents_opposite_bins(self):
        """Two agents in opposite bins should have non-zero entropy."""
        signal = DisagreementSignal()
        # a -> strong_bull, b -> strong_bear
        directions = {"a": 0.9, "b": -0.9}
        confidences = {"a": 0.9, "b": 0.9}
        result = signal.compute(directions, confidences)
        assert result.entropy > 0
        assert result.n_clusters == 2
        # Two equally sized bins: entropy = log(2)
        assert result.entropy == pytest.approx(math.log(2), abs=1e-4)

    def test_max_entropy_all_bins_equal(self):
        """Equal distribution across all 5 bins should produce max entropy = log(5)."""
        signal = DisagreementSignal()
        # Carefully place one agent in each bin
        directions = {
            "sb": -0.8,   # strong_bear (< -0.5)
            "mb": -0.3,   # mild_bear ([-0.5, -0.1))
            "n": 0.0,     # neutral ([-0.1, 0.1))
            "mbu": 0.3,   # mild_bull ([0.1, 0.5))
            "sbu": 0.8,   # strong_bull (>= 0.5)
        }
        confidences = {k: 0.8 for k in directions}
        result = signal.compute(directions, confidences)
        assert result.n_clusters == 5
        assert result.entropy == pytest.approx(math.log(5), abs=1e-4)

    def test_high_disagreement_threshold(self):
        """High disagreement flag should trigger when entropy > 0.7 * max_entropy."""
        signal = DisagreementSignal()
        max_ent = math.log(5)
        threshold = 0.7 * max_ent
        # Two bins with equal distribution: entropy = log(2) = 0.693
        # threshold ~ 0.7 * 1.609 = 1.127
        # log(2) < threshold, so not high disagreement
        directions = {"a": 0.8, "b": -0.8}
        confidences = {"a": 0.9, "b": 0.9}
        result = signal.compute(directions, confidences)
        assert result.entropy == pytest.approx(math.log(2), abs=1e-4)
        assert result.entropy < threshold
        assert result.is_high_disagreement is False

    def test_bin_boundaries_strong_bull(self):
        """Direction >= 0.5 should map to strong_bull."""
        signal = DisagreementSignal()
        directions = {"a": 0.5, "b": 1.0}
        confidences = {"a": 0.8, "b": 0.8}
        result = signal.compute(directions, confidences)
        assert result.n_clusters == 1  # both in strong_bull

    def test_bin_boundaries_mild_bull(self):
        """Direction in [0.1, 0.5) should map to mild_bull."""
        signal = DisagreementSignal()
        directions = {"a": 0.1, "b": 0.49}
        confidences = {"a": 0.8, "b": 0.8}
        result = signal.compute(directions, confidences)
        assert result.n_clusters == 1  # both in mild_bull

    def test_bin_boundaries_neutral(self):
        """Direction in [-0.1, 0.1) should map to neutral."""
        signal = DisagreementSignal()
        directions = {"a": -0.1, "b": 0.09}
        confidences = {"a": 0.8, "b": 0.8}
        result = signal.compute(directions, confidences)
        assert result.n_clusters == 1  # both in neutral
