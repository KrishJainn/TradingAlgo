"""
AQTIS Random Forest Predictor.

Ensemble tree-based model for return prediction.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RandomForestPredictor:
    """
    Random Forest model for predicting trade returns.

    Uses technical indicator features to predict:
    - Direction (up/down)
    - Expected return magnitude
    """

    def __init__(self, n_estimators: int = 200, max_depth: int = 10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self._model = None
        self._feature_names: List[str] = []
        self._fitted = False

    def train(self, features: pd.DataFrame, targets: pd.Series) -> Dict:
        """
        Train the Random Forest model.

        Args:
            features: DataFrame of technical indicator features.
            targets: Series of forward returns (the prediction target).

        Returns:
            Training metrics.
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import cross_val_score

        # Drop NaN rows
        valid = features.dropna().index.intersection(targets.dropna().index)
        X = features.loc[valid].values
        y = targets.loc[valid].values

        if len(X) < 50:
            return {"error": "Insufficient training data", "samples": len(X)}

        self._feature_names = list(features.columns)

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

        # Cross-validation score
        cv_scores = cross_val_score(self._model, X, y, cv=5, scoring="r2")

        # Fit on full data
        self._model.fit(X, y)
        self._fitted = True

        # Feature importance
        importances = dict(zip(self._feature_names, self._model.feature_importances_))

        logger.info(
            f"RF trained: {len(X)} samples, CV R2={np.mean(cv_scores):.4f} "
            f"(+/- {np.std(cv_scores):.4f})"
        )

        return {
            "samples": len(X),
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std": float(np.std(cv_scores)),
            "feature_importance": {
                k: round(v, 4)
                for k, v in sorted(importances.items(), key=lambda x: -x[1])[:10]
            },
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict returns for given features.

        Returns:
            Array of predicted returns.
        """
        if not self._fitted:
            logger.warning("Model not fitted, returning zeros")
            return np.zeros(len(features))

        X = features[self._feature_names].values if self._feature_names else features.values
        return self._model.predict(X)

    def predict_single(self, features: Dict[str, float]) -> float:
        """Predict return for a single observation."""
        if not self._fitted:
            return 0.0

        X = np.array([[features.get(f, 0) for f in self._feature_names]])
        return float(self._model.predict(X)[0])

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance rankings."""
        if not self._fitted:
            return {}
        return dict(zip(self._feature_names, self._model.feature_importances_))

    def save(self, path: str):
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "feature_names": self._feature_names,
                "fitted": self._fitted,
            }, f)

    def load(self, path: str):
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._feature_names = data["feature_names"]
        self._fitted = data["fitted"]
