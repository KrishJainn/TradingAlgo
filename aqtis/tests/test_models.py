"""
Tests for AQTIS ML Models.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aqtis.models.rf_model import RandomForestPredictor
from aqtis.models.linear_model import LinearPredictor
from aqtis.models.regime_detector import RegimeDetector
from aqtis.models.ensemble import ModelEnsemble


def _generate_sample_data(n=500):
    """Generate sample features and targets."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")

    features = pd.DataFrame({
        "RSI_14": np.random.uniform(20, 80, n),
        "MACD_12_26_9": np.random.randn(n) * 0.5,
        "ADX_14": np.random.uniform(10, 50, n),
        "ATR_14": np.random.uniform(1, 5, n),
        "SMA_20": np.random.uniform(95, 105, n),
    }, index=dates)

    # Targets: forward 1-day return with some signal
    targets = (features["RSI_14"] - 50) * -0.001 + np.random.randn(n) * 0.01
    targets = pd.Series(targets.values, index=dates, name="target")

    return features, targets


def _generate_price_series(n=500):
    """Generate sample price series."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates, name="close")


class TestRandomForest:
    def test_train_and_predict(self):
        features, targets = _generate_sample_data()
        model = RandomForestPredictor(n_estimators=50, max_depth=5)

        result = model.train(features, targets)
        assert result["samples"] == len(features)
        assert "cv_r2_mean" in result

        preds = model.predict(features)
        assert len(preds) == len(features)

    def test_predict_single(self):
        features, targets = _generate_sample_data()
        model = RandomForestPredictor(n_estimators=50)
        model.train(features, targets)

        single = features.iloc[0].to_dict()
        pred = model.predict_single(single)
        assert isinstance(pred, float)

    def test_feature_importance(self):
        features, targets = _generate_sample_data()
        model = RandomForestPredictor(n_estimators=50)
        model.train(features, targets)

        importance = model.get_feature_importance()
        assert len(importance) == len(features.columns)
        assert all(v >= 0 for v in importance.values())

    def test_insufficient_data(self):
        features = pd.DataFrame({"a": [1, 2, 3]})
        targets = pd.Series([0.01, -0.01, 0.02])
        model = RandomForestPredictor()

        result = model.train(features, targets)
        assert "error" in result


class TestLinearModel:
    def test_train_and_predict(self):
        features, targets = _generate_sample_data()
        model = LinearPredictor(alpha=1.0)

        result = model.train(features, targets)
        assert result["samples"] == len(features)

        preds = model.predict(features)
        assert len(preds) == len(features)

    def test_lasso(self):
        features, targets = _generate_sample_data()
        model = LinearPredictor(alpha=0.1, model_type="lasso")

        result = model.train(features, targets)
        assert "cv_r2_mean" in result


class TestRegimeDetector:
    def test_rule_based_regime(self):
        prices = _generate_price_series()
        detector = RegimeDetector()
        regime = detector.detect_regime(prices)

        assert "regime_name" in regime
        assert regime["regime_name"] in [
            "trending_up", "trending_down", "mean_reverting",
            "high_volatility", "low_volatility", "unknown",
        ]

    def test_fit_and_detect(self):
        prices = _generate_price_series(n=300)
        detector = RegimeDetector(n_regimes=3)
        detector.fit(prices)

        regime = detector.detect_regime(prices)
        assert "regime_name" in regime
        assert "method" in regime

    def test_regime_history(self):
        prices = _generate_price_series(n=300)
        detector = RegimeDetector(n_regimes=3)

        history = detector.get_regime_history(prices)
        if not history.empty:
            assert "regime_name" in history.columns


class TestModelEnsemble:
    def test_train_all(self):
        features, targets = _generate_sample_data()
        ensemble = ModelEnsemble()

        results = ensemble.train_all(features, targets)
        assert "random_forest" in results
        assert "linear_regression" in results

    def test_predict(self):
        features, targets = _generate_sample_data()
        ensemble = ModelEnsemble()
        ensemble.train_all(features, targets)

        result = ensemble.predict(features)
        assert "ensemble_prediction" in result
        assert "direction" in result
        assert "individual_predictions" in result

    def test_predict_single(self):
        features, targets = _generate_sample_data()
        ensemble = ModelEnsemble()
        ensemble.train_all(features, targets)

        single = features.iloc[-1].to_dict()
        result = ensemble.predict_single(single)
        assert "ensemble_prediction" in result

    def test_update_weights(self):
        ensemble = ModelEnsemble()
        original_weights = ensemble.get_weights()

        ensemble.update_weights({
            "random_forest": 0.8,
            "linear_regression": 0.6,
            "lstm": 0.4,
            "rules_based": 0.7,
        })

        new_weights = ensemble.get_weights()
        # Weights should have changed
        assert new_weights != original_weights


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
