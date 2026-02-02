"""
AQTIS Model Ensemble.

Combines multiple predictive models with dynamic weighting
based on recent accuracy tracked by the PredictionTrackingAgent.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .rf_model import RandomForestPredictor
from .linear_model import LinearPredictor
from .lstm_model import LSTMPredictor
from .regime_detector import RegimeDetector

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "random_forest": 0.30,
    "linear_regression": 0.20,
    "lstm": 0.20,
    "rules_based": 0.30,
}


class ModelEnsemble:
    """
    Ensemble of predictive models with dynamic weighting.

    Models:
    - Random Forest: Good for non-linear feature interactions
    - Linear Regression: Interpretable baseline
    - LSTM: Captures sequential patterns
    - Rules-based: Simple momentum/mean-reversion signals

    Weights are updated based on recent prediction accuracy.
    """

    def __init__(
        self,
        weights: Dict[str, float] = None,
        sequence_length: int = 60,
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.sequence_length = sequence_length

        # Initialize models
        self.models = {
            "random_forest": RandomForestPredictor(n_estimators=200, max_depth=10),
            "linear_regression": LinearPredictor(alpha=1.0),
            "lstm": LSTMPredictor(sequence_length=sequence_length),
        }

        self.regime_detector = RegimeDetector()
        self._trained = False
        self._training_results: Dict = {}

    def train_all(self, features: pd.DataFrame, targets: pd.Series) -> Dict:
        """
        Train all models in the ensemble.

        Args:
            features: DataFrame of technical indicator features.
            targets: Series of forward returns.

        Returns:
            Training results for each model.
        """
        results = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            try:
                result = model.train(features, targets)
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                results[name] = {"error": str(e)}

        # Fit regime detector on price data if available
        if "close" in features.columns:
            try:
                self.regime_detector.fit(features["close"])
                results["regime_detector"] = {"fitted": True}
            except Exception as e:
                results["regime_detector"] = {"error": str(e)}

        self._trained = True
        self._training_results = results

        logger.info("Ensemble training complete")
        return results

    def predict(self, features: pd.DataFrame) -> Dict:
        """
        Generate ensemble prediction.

        Args:
            features: DataFrame of indicator features.

        Returns:
            Dict with ensemble prediction and individual model predictions.
        """
        predictions = {}
        confidences = {}

        # Get individual model predictions
        for name, model in self.models.items():
            try:
                pred = model.predict(features)
                if len(pred) > 0:
                    predictions[name] = float(pred[-1])  # Latest prediction
                else:
                    predictions[name] = 0.0
            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                predictions[name] = 0.0

        # Rules-based prediction
        rules_pred = self._rules_based_prediction(features)
        predictions["rules_based"] = rules_pred

        # Weighted ensemble
        ensemble_pred = 0.0
        total_weight = 0.0
        for name, pred in predictions.items():
            weight = self.weights.get(name, 0)
            ensemble_pred += pred * weight
            total_weight += weight

        if total_weight > 0:
            ensemble_pred /= total_weight

        # Confidence estimate
        pred_values = list(predictions.values())
        agreement = 1.0 - np.std(pred_values) / (abs(np.mean(pred_values)) + 1e-6)
        confidence = max(0.0, min(1.0, agreement))

        # Regime detection
        regime = None
        if "close" in features.columns:
            try:
                regime = self.regime_detector.detect_regime(features["close"])
            except Exception:
                pass

        return {
            "ensemble_prediction": ensemble_pred,
            "direction": "LONG" if ensemble_pred > 0 else "SHORT",
            "confidence": confidence,
            "individual_predictions": predictions,
            "weights": self.weights,
            "regime": regime,
        }

    def predict_single(self, features: Dict[str, float]) -> Dict:
        """Predict for a single observation."""
        predictions = {}

        for name, model in self.models.items():
            if name == "lstm":
                continue  # LSTM needs sequence
            try:
                predictions[name] = model.predict_single(features)
            except Exception:
                predictions[name] = 0.0

        predictions["rules_based"] = self._rules_based_single(features)

        ensemble_pred = sum(
            predictions.get(name, 0) * self.weights.get(name, 0)
            for name in predictions
        )
        total_weight = sum(self.weights.get(name, 0) for name in predictions if name in self.weights)

        if total_weight > 0:
            ensemble_pred /= total_weight

        return {
            "ensemble_prediction": ensemble_pred,
            "direction": "LONG" if ensemble_pred > 0 else "SHORT",
            "individual_predictions": predictions,
        }

    def update_weights(self, accuracy_by_model: Dict[str, float]):
        """
        Update model weights based on recent accuracy.

        Args:
            accuracy_by_model: Dict mapping model name to recent accuracy.
        """
        if not accuracy_by_model:
            return

        # Softmax-style weighting
        total = sum(accuracy_by_model.values())
        if total <= 0:
            return

        new_weights = {}
        for name in self.weights:
            if name in accuracy_by_model:
                new_weights[name] = accuracy_by_model[name] / total
            else:
                new_weights[name] = self.weights[name]

        # Smooth update (blend old and new)
        blend = 0.3
        for name in self.weights:
            if name in new_weights:
                self.weights[name] = (1 - blend) * self.weights[name] + blend * new_weights[name]

        # Normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info(f"Updated ensemble weights: {self.weights}")

    def get_weights(self) -> Dict[str, float]:
        """Get current ensemble weights."""
        return self.weights.copy()

    # ─────────────────────────────────────────────────────────────────
    # RULES-BASED MODEL
    # ─────────────────────────────────────────────────────────────────

    def _rules_based_prediction(self, features: pd.DataFrame) -> float:
        """
        Simple rules-based prediction using momentum and mean reversion signals.
        """
        if features.empty:
            return 0.0

        last = features.iloc[-1]
        signals = []

        # RSI mean reversion
        rsi = last.get("RSI_14", 50)
        if rsi < 30:
            signals.append(0.02)  # Oversold -> bullish
        elif rsi > 70:
            signals.append(-0.02)  # Overbought -> bearish

        # MACD momentum
        macd = last.get("MACD_12_26_9", 0)
        if macd > 0:
            signals.append(0.01)
        elif macd < 0:
            signals.append(-0.01)

        # ADX trend strength
        adx = last.get("ADX_14", 0)
        if adx > 25:
            # Strong trend - amplify momentum signal
            if signals:
                signals[-1] *= 1.5

        return float(np.mean(signals)) if signals else 0.0

    def _rules_based_single(self, features: Dict[str, float]) -> float:
        """Rules-based prediction for a single observation."""
        signals = []

        rsi = features.get("RSI_14", 50)
        if rsi < 30:
            signals.append(0.02)
        elif rsi > 70:
            signals.append(-0.02)

        macd = features.get("MACD_12_26_9", 0)
        if macd > 0:
            signals.append(0.01)
        elif macd < 0:
            signals.append(-0.01)

        return float(np.mean(signals)) if signals else 0.0

    # ─────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────

    def save(self, directory: str):
        """Save all models to a directory."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        for name, model in self.models.items():
            try:
                model.save(str(path / f"{name}.pkl"))
            except Exception as e:
                logger.warning(f"Failed to save {name}: {e}")

        # Save weights
        import json
        with open(path / "weights.json", "w") as f:
            json.dump(self.weights, f)

    def load(self, directory: str):
        """Load all models from a directory."""
        path = Path(directory)
        if not path.exists():
            return

        for name, model in self.models.items():
            model_path = path / f"{name}.pkl"
            if model_path.exists():
                try:
                    model.load(str(model_path))
                except Exception as e:
                    logger.warning(f"Failed to load {name}: {e}")

        # Load weights
        import json
        weights_path = path / "weights.json"
        if weights_path.exists():
            with open(weights_path) as f:
                self.weights = json.load(f)

        self._trained = True
