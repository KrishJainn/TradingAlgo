"""Tests for MetaLabeler.

Validates the meta-labeling secondary model from Lopez de Prado (2018),
including fit/predict lifecycle, purged cross-validation, and edge cases.
"""

import numpy as np
import pytest
from risk_engine.core.meta_labeler import MetaLabeler, MetaLabelResult


class TestMetaLabeler:
    def _make_data(self, n=200, n_features=5, seed=42):
        """Generate synthetic data with a learnable pattern.

        Positive primary signals combined with positive X[:,0]
        tend to produce positive returns, giving the meta-labeler
        something to learn.
        """
        rng = np.random.RandomState(seed)
        X = rng.randn(n, n_features)
        primary_signals = np.sign(rng.randn(n))
        # Ensure no zero signals (np.sign can return 0)
        primary_signals[primary_signals == 0] = 1.0
        returns = primary_signals * (0.001 + 0.002 * X[:, 0]) + rng.randn(n) * 0.005
        return X, primary_signals, returns

    def test_fit_returns_none(self):
        """fit() should return None (fits in place)."""
        ml = MetaLabeler(n_folds=3)
        X, signals, returns = self._make_data()
        result = ml.fit(X, signals, returns, purge_bars=2)
        assert result is None

    def test_predict_returns_meta_label_result(self):
        """predict() should return a MetaLabelResult instance."""
        ml = MetaLabeler(n_folds=3)
        X, signals, returns = self._make_data()
        ml.fit(X, signals, returns, purge_bars=2)
        result = ml.predict(X[-1:], primary_direction=1.0)
        assert isinstance(result, MetaLabelResult)

    def test_confidence_in_range(self):
        """Predicted confidence should be in [0, 1]."""
        ml = MetaLabeler(n_folds=3)
        X, signals, returns = self._make_data()
        ml.fit(X, signals, returns, purge_bars=2)
        result = ml.predict(X[-1:], primary_direction=1.0)
        assert 0.0 <= result.confidence <= 1.0

    def test_bet_size_in_range(self):
        """Bet size should be in [0, 1]."""
        ml = MetaLabeler(n_folds=3)
        X, signals, returns = self._make_data()
        ml.fit(X, signals, returns, purge_bars=2)
        result = ml.predict(X[-1:], primary_direction=1.0)
        assert 0.0 <= result.bet_size <= 1.0

    def test_predict_before_fit_raises(self):
        """predict() before fit() should raise RuntimeError."""
        ml = MetaLabeler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            ml.predict(np.array([[1, 2, 3, 4, 5]]), primary_direction=1.0)

    def test_primary_direction_passed_through(self):
        """The primary_direction should be stored in the result unchanged."""
        ml = MetaLabeler(n_folds=3)
        X, signals, returns = self._make_data()
        ml.fit(X, signals, returns, purge_bars=2)

        result_long = ml.predict(X[-1:], primary_direction=1.0)
        assert result_long.primary_direction == 1.0

        result_short = ml.predict(X[-1:], primary_direction=-1.0)
        assert result_short.primary_direction == -1.0

    def test_should_trade_based_on_threshold(self):
        """should_trade should reflect whether confidence >= threshold."""
        ml = MetaLabeler(n_folds=3, threshold=0.5)
        X, signals, returns = self._make_data()
        ml.fit(X, signals, returns, purge_bars=2)
        result = ml.predict(X[-1:], primary_direction=1.0)
        if result.confidence >= 0.5:
            assert result.should_trade is True
        else:
            assert result.should_trade is False

    def test_cv_scores_populated_after_fit(self):
        """Cross-validation scores should be populated after fit."""
        ml = MetaLabeler(n_folds=3)
        X, signals, returns = self._make_data()
        ml.fit(X, signals, returns, purge_bars=2)
        assert len(ml.cv_scores) > 0
        for score in ml.cv_scores:
            assert 0.0 <= score <= 1.0

    def test_mean_cv_score_is_valid(self):
        """Mean CV score should be a float in [0, 1] after fit."""
        ml = MetaLabeler(n_folds=3)
        X, signals, returns = self._make_data()
        ml.fit(X, signals, returns, purge_bars=2)
        mean_score = ml.mean_cv_score
        assert mean_score is not None
        assert 0.0 <= mean_score <= 1.0

    def test_mean_cv_score_none_before_fit(self):
        """Mean CV score should be None before fit."""
        ml = MetaLabeler()
        assert ml.mean_cv_score is None

    def test_shape_mismatch_raises(self):
        """Mismatched X, signals, returns shapes should raise ValueError."""
        ml = MetaLabeler()
        X = np.random.randn(100, 5)
        signals = np.ones(50)  # wrong length
        returns = np.random.randn(100) * 0.01
        with pytest.raises(ValueError, match="Shape mismatch"):
            ml.fit(X, signals, returns)

    def test_predict_1d_input(self):
        """predict() should accept 1-D feature input (single sample)."""
        ml = MetaLabeler(n_folds=3)
        X, signals, returns = self._make_data()
        ml.fit(X, signals, returns, purge_bars=2)
        # Pass 1-D instead of 2-D
        result = ml.predict(X[-1], primary_direction=1.0)
        assert isinstance(result, MetaLabelResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_fit_with_larger_dataset(self):
        """fit() should work with a larger dataset (500 samples)."""
        ml = MetaLabeler(n_folds=5)
        X, signals, returns = self._make_data(n=500, n_features=10, seed=99)
        ml.fit(X, signals, returns, purge_bars=5)
        result = ml.predict(X[-1:], primary_direction=-1.0)
        assert isinstance(result, MetaLabelResult)
        assert 0.0 <= result.bet_size <= 1.0

    def test_bet_size_calibration(self):
        """Bet size should be 2*confidence - 1, clipped to [0, 1]."""
        ml = MetaLabeler(n_folds=3)
        X, signals, returns = self._make_data()
        ml.fit(X, signals, returns, purge_bars=2)
        result = ml.predict(X[-1:], primary_direction=1.0)
        expected_bet = max(0.0, min(1.0, 2.0 * result.confidence - 1.0))
        assert result.bet_size == pytest.approx(expected_bet, abs=1e-3)

    def test_different_seeds_give_different_results(self):
        """Different random seeds in data generation should vary results."""
        results = []
        for seed in [10, 20, 30]:
            ml = MetaLabeler(n_folds=3)
            X, signals, returns = self._make_data(n=200, seed=seed)
            ml.fit(X, signals, returns, purge_bars=2)
            result = ml.predict(X[-1:], primary_direction=1.0)
            results.append(result.confidence)
        # Not all should be identical
        assert len(set(round(r, 4) for r in results)) > 1
