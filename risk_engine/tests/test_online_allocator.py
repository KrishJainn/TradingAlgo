"""Tests for the OnlineAllocator (Multiplicative Weights Update).

Validates:
- Initial weights are uniform across agents
- MWU update rule increases weight for winners
- Weights always sum to 1 after any number of updates
- Minimum weight floor is respected
- Blending with HRP weights produces valid allocations
- Input validation (n_agents, learning_rate, name mismatches)

Reference:
    Cesa-Bianchi, N., & Lugosi, G. (2006). "Prediction, Learning, and Games."
"""

import numpy as np
import pytest

from risk_engine.core.online_allocator import OnlineAllocator


class TestOnlineAllocatorInit:
    """Tests for OnlineAllocator initialization."""

    def test_initial_weights_equal(self):
        """All agents should start with equal weight (1/n)."""
        oa = OnlineAllocator(n_agents=3)
        oa.set_agent_names(["A", "B", "C"])
        weights = oa.get_weights()
        assert len(weights) == 3
        for w in weights.values():
            assert w == pytest.approx(1 / 3, abs=1e-6)

    def test_n_agents_init(self):
        """Initialization with n_agents should create the correct count."""
        oa = OnlineAllocator(n_agents=4)
        weights = oa.get_weights()
        assert len(weights) == 4

    def test_default_agent_names(self):
        """Default agent names should be agent_0, agent_1, etc."""
        oa = OnlineAllocator(n_agents=3)
        weights = oa.get_weights()
        assert set(weights.keys()) == {"agent_0", "agent_1", "agent_2"}

    def test_set_agent_names(self):
        """set_agent_names should update the weight keys."""
        oa = OnlineAllocator(n_agents=3)
        oa.set_agent_names(["alpha", "beta", "gamma"])
        weights = oa.get_weights()
        assert set(weights.keys()) == {"alpha", "beta", "gamma"}

    def test_set_agent_names_wrong_count_raises(self):
        """Should raise ValueError if number of names doesn't match n_agents."""
        oa = OnlineAllocator(n_agents=3)
        with pytest.raises(ValueError, match="Expected 3"):
            oa.set_agent_names(["A", "B"])

    def test_invalid_n_agents_raises(self):
        """n_agents < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_agents"):
            OnlineAllocator(n_agents=0)

    def test_invalid_learning_rate_raises(self):
        """learning_rate <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="learning_rate"):
            OnlineAllocator(n_agents=3, learning_rate=0.0)


class TestOnlineAllocatorUpdate:
    """Tests for OnlineAllocator.update()."""

    def test_update_increases_winner_weight(self):
        """Agent with highest return should gain weight after update."""
        oa = OnlineAllocator(n_agents=3, learning_rate=0.5)
        oa.set_agent_names(["A", "B", "C"])
        initial_a = oa.get_weights()["A"]
        oa.update({"A": 0.05, "B": -0.01, "C": -0.02})
        updated_a = oa.get_weights()["A"]
        assert updated_a > initial_a

    def test_update_decreases_loser_weight(self):
        """Agent with worst return should lose weight after update."""
        oa = OnlineAllocator(n_agents=3, learning_rate=0.5)
        oa.set_agent_names(["A", "B", "C"])
        initial_c = oa.get_weights()["C"]
        oa.update({"A": 0.05, "B": 0.01, "C": -0.05})
        updated_c = oa.get_weights()["C"]
        assert updated_c < initial_c

    def test_weights_sum_to_one_after_update(self):
        """Weights must always sum to 1 after any update."""
        oa = OnlineAllocator(n_agents=3)
        oa.set_agent_names(["A", "B", "C"])
        for _ in range(10):
            oa.update({"A": 0.01, "B": -0.005, "C": 0.002})
        weights = oa.get_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-6)

    def test_weights_sum_to_one_many_updates(self):
        """Weights remain valid after many sequential updates."""
        rng = np.random.RandomState(42)
        oa = OnlineAllocator(n_agents=5, learning_rate=0.2)
        names = [f"P{i}" for i in range(5)]
        oa.set_agent_names(names)
        for _ in range(100):
            returns = {n: rng.uniform(-0.03, 0.03) for n in names}
            oa.update(returns)
        weights = oa.get_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-5)

    def test_min_weight_floor(self):
        """No agent weight should fall below min_weight after heavy losses."""
        oa = OnlineAllocator(n_agents=3, min_weight=0.05, learning_rate=0.5)
        oa.set_agent_names(["A", "B", "C"])
        # Repeatedly punish B
        for _ in range(50):
            oa.update({"A": 0.1, "B": -0.1, "C": 0.05})
        weights = oa.get_weights()
        # Account for rounding: the floor guarantees >= min_weight within tolerance
        assert weights["B"] >= 0.05 - 1e-6

    def test_auto_detect_names_on_first_update(self):
        """On first update, agent names should be auto-detected from keys."""
        oa = OnlineAllocator(n_agents=3)
        oa.update({"X": 0.01, "Y": -0.01, "Z": 0.005})
        weights = oa.get_weights()
        assert set(weights.keys()) == {"X", "Y", "Z"}

    def test_mismatched_keys_raises_after_naming(self):
        """update() should raise if keys don't match registered agents."""
        oa = OnlineAllocator(n_agents=2)
        oa.set_agent_names(["A", "B"])
        with pytest.raises(ValueError, match="don't match"):
            oa.update({"A": 0.01, "C": -0.01})

    def test_n_steps_increments(self):
        """n_steps property should track the number of updates."""
        oa = OnlineAllocator(n_agents=2)
        oa.set_agent_names(["A", "B"])
        assert oa.n_steps == 0
        oa.update({"A": 0.01, "B": -0.01})
        assert oa.n_steps == 1
        oa.update({"A": 0.02, "B": 0.01})
        assert oa.n_steps == 2

    def test_history_recorded(self):
        """History should contain one snapshot per update call."""
        oa = OnlineAllocator(n_agents=2)
        oa.set_agent_names(["A", "B"])
        oa.update({"A": 0.01, "B": -0.01})
        oa.update({"A": -0.005, "B": 0.02})
        assert len(oa.history) == 2
        assert oa.history[0].step == 0
        assert oa.history[1].step == 1


class TestOnlineAllocatorBlend:
    """Tests for OnlineAllocator.blend_with_hrp()."""

    def test_blend_with_hrp_sums_to_one(self):
        """Blended weights must sum to 1."""
        oa = OnlineAllocator(n_agents=3)
        oa.set_agent_names(["A", "B", "C"])
        hrp = {"A": 0.5, "B": 0.3, "C": 0.2}
        blended = oa.blend_with_hrp(hrp, tactical_ratio=0.3)
        assert sum(blended.values()) == pytest.approx(1.0, abs=1e-6)

    def test_blend_pure_hrp(self):
        """With tactical_ratio=0, result should match HRP weights."""
        oa = OnlineAllocator(n_agents=3)
        oa.set_agent_names(["A", "B", "C"])
        hrp = {"A": 0.5, "B": 0.3, "C": 0.2}
        blended = oa.blend_with_hrp(hrp, tactical_ratio=0.0)
        for key in hrp:
            assert blended[key] == pytest.approx(hrp[key], abs=1e-4)

    def test_blend_pure_online(self):
        """With tactical_ratio=1, result should match online weights."""
        oa = OnlineAllocator(n_agents=3)
        oa.set_agent_names(["A", "B", "C"])
        online = oa.get_weights()
        hrp = {"A": 0.5, "B": 0.3, "C": 0.2}
        blended = oa.blend_with_hrp(hrp, tactical_ratio=1.0)
        for key in online:
            assert blended[key] == pytest.approx(online[key], abs=1e-4)

    def test_blend_invalid_ratio_raises(self):
        """tactical_ratio outside [0, 1] should raise ValueError."""
        oa = OnlineAllocator(n_agents=2)
        oa.set_agent_names(["A", "B"])
        with pytest.raises(ValueError, match="tactical_ratio"):
            oa.blend_with_hrp({"A": 0.5, "B": 0.5}, tactical_ratio=1.5)

    def test_blend_missing_hrp_keys_raises(self):
        """HRP weights missing agents should raise ValueError."""
        oa = OnlineAllocator(n_agents=2)
        oa.set_agent_names(["A", "B"])
        with pytest.raises(ValueError, match="missing"):
            oa.blend_with_hrp({"A": 1.0}, tactical_ratio=0.3)
