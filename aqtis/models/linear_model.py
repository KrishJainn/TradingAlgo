"""
AQTIS Linear Model (Ridge/Lasso).

Simple linear baseline for return prediction.
"""

import logging
import pickle
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LinearPredictor:
    """
    Linear regression baseline using Ridge regularization.

    Serves as a simple, interpretable baseline for the ensemble.
    """

    def __init__(self, alpha: float = 1.0, model_type: str = "ridge"):
        self.alpha = alpha
        self.model_type = model_type
        self._model = None
        self._feature_names: List[str] = []
        self._fitted = False

    def train(self, features: pd.DataFrame, targets: pd.Series) -> Dict:
        """Train the linear model."""
        from sklearn.linear_model import Ridge, Lasso
        from sklearn.model_selection import cross_val_score
        from sklearn.preprocessing import StandardScaler

        valid = features.dropna().index.intersection(targets.dropna().index)
        X = features.loc[valid].values
        y = targets.loc[valid].values

        if len(X) < 30:
            return {"error": "Insufficient training data", "samples": len(X)}

        self._feature_names = list(features.columns)

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        if self.model_type == "lasso":
            self._model = Lasso(alpha=self.alpha, random_state=42)
        else:
            self._model = Ridge(alpha=self.alpha)

        cv_scores = cross_val_score(self._model, X_scaled, y, cv=5, scoring="r2")
        self._model.fit(X_scaled, y)
        self._fitted = True

        # Coefficients
        coefs = dict(zip(self._feature_names, self._model.coef_))

        logger.info(
            f"Linear model trained: {len(X)} samples, "
            f"CV R2={np.mean(cv_scores):.4f}"
        )

        return {
            "samples": len(X),
            "cv_r2_mean": float(np.mean(cv_scores)),
            "cv_r2_std": float(np.std(cv_scores)),
            "top_coefficients": {
                k: round(v, 6)
                for k, v in sorted(coefs.items(), key=lambda x: -abs(x[1]))[:10]
            },
        }

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Predict returns."""
        if not self._fitted:
            return np.zeros(len(features))

        X = features[self._feature_names].values if self._feature_names else features.values
        X_scaled = self._scaler.transform(X)
        return self._model.predict(X_scaled)

    def predict_single(self, features: Dict[str, float]) -> float:
        """Predict for a single observation."""
        if not self._fitted:
            return 0.0
        X = np.array([[features.get(f, 0) for f in self._feature_names]])
        X_scaled = self._scaler.transform(X)
        return float(self._model.predict(X_scaled)[0])

    def get_feature_importance(self) -> Dict[str, float]:
        """Get absolute coefficient values as importance."""
        if not self._fitted:
            return {}
        return {
            name: abs(coef)
            for name, coef in zip(self._feature_names, self._model.coef_)
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "scaler": self._scaler,
                "feature_names": self._feature_names,
                "fitted": self._fitted,
            }, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self._scaler = data["scaler"]
        self._feature_names = data["feature_names"]
        self._fitted = data["fitted"]
