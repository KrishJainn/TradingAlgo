"""Tests for RiskAllocator (Hierarchical Risk Parity)."""

import numpy as np
import pandas as pd
import pytest

from risk_engine.core.risk_allocator import RiskAllocator, RiskBudget


class TestHRP:
    def _make_returns(self, n_assets=5, n_obs=100, seed=42):
        rng = np.random.RandomState(seed)
        data = rng.randn(n_obs, n_assets) * 0.02
        cols = [f"Asset_{i}" for i in range(n_assets)]
        return pd.DataFrame(data, columns=cols)

    def test_weights_sum_to_one(self):
        alloc = RiskAllocator(min_weight=0.05, max_weight=0.40)
        returns = self._make_returns()
        budget = alloc.allocate(returns)
        assert sum(budget.weights.values()) == pytest.approx(1.0, abs=1e-8)

    def test_weights_within_bounds(self):
        alloc = RiskAllocator(min_weight=0.05, max_weight=0.40)
        returns = self._make_returns()
        budget = alloc.allocate(returns)
        for w in budget.weights.values():
            assert w >= 0.05 - 1e-8
            assert w <= 0.40 + 1e-8

    def test_two_assets_minimum(self):
        alloc = RiskAllocator()
        returns = self._make_returns(n_assets=2)
        budget = alloc.allocate(returns)
        assert len(budget.weights) == 2

    def test_single_asset_raises(self):
        alloc = RiskAllocator()
        returns = pd.DataFrame(np.random.randn(100, 1), columns=["A"])
        with pytest.raises(ValueError):
            alloc.allocate(returns)

    def test_sorted_order_length(self):
        alloc = RiskAllocator()
        returns = self._make_returns(n_assets=5)
        budget = alloc.allocate(returns)
        assert len(budget.sorted_order) == 5

    def test_effective_n(self):
        alloc = RiskAllocator()
        returns = self._make_returns()
        budget = alloc.allocate(returns)
        eff_n = budget.effective_n()
        assert eff_n >= 1.0
        assert eff_n <= 5.0

    def test_allocate_from_covariance(self):
        alloc = RiskAllocator()
        cov = np.array([[0.04, 0.01, 0.005],
                        [0.01, 0.03, 0.008],
                        [0.005, 0.008, 0.02]])
        names = ["A", "B", "C"]
        budget = alloc.allocate_from_covariance(cov, names)
        assert sum(budget.weights.values()) == pytest.approx(1.0, abs=1e-8)

    def test_invalid_linkage_method(self):
        with pytest.raises(ValueError):
            RiskAllocator(linkage_method="invalid")
