"""
AQTIS Prediction Tracking Agent.

Tracks every prediction vs actual outcome, calibrates confidence,
monitors model accuracy, and detects model degradation.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAgent

logger = logging.getLogger(__name__)


class PredictionTrackingAgent(BaseAgent):
    """
    Monitors prediction accuracy and calibrates model confidence.

    Responsibilities:
    - Track every prediction vs actual outcome
    - Confidence calibration (adjust model confidence to match reality)
    - Model performance attribution
    - Detect when models are degrading
    """

    def __init__(self, memory, llm=None):
        super().__init__(name="prediction_tracker", memory=memory, llm=llm)
        self._calibration_bins = self._initialize_bins()

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute based on context action."""
        action = context.get("action", "status")

        if action == "record_prediction":
            return self.record_prediction(context["prediction"])
        elif action == "record_outcome":
            return self.record_outcome(
                context["prediction_id"], context["outcome"]
            )
        elif action == "calibrate":
            return self.get_calibration_report()
        elif action == "detect_degradation":
            return {"degradation_alerts": self.detect_model_degradation()}
        elif action == "llm_calibration_analysis":
            return self.llm_calibration_analysis(
                context.get("metrics", {}),
                context.get("regime_breakdown", {}),
            )
        else:
            return self.get_calibration_report()

    # ─────────────────────────────────────────────────────────────────
    # CORE OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def record_prediction(self, prediction: Dict) -> Dict:
        """
        Log a new prediction. Returns prediction_id.
        """
        prediction_id = prediction.get("prediction_id", str(uuid.uuid4()))
        prediction["prediction_id"] = prediction_id
        prediction["timestamp"] = prediction.get("timestamp", datetime.now().isoformat())

        self.memory.store_prediction(prediction)
        self.logger.info(
            f"Recorded prediction {prediction_id}: "
            f"{prediction.get('asset')} {prediction.get('predicted_return', 0):.2%} "
            f"conf={prediction.get('predicted_confidence', 0):.2f}"
        )
        return {"prediction_id": prediction_id}

    def record_outcome(self, prediction_id: str, outcome: Dict) -> Dict:
        """
        Record actual outcome and update calibration.
        """
        prediction = self.memory.get_prediction(prediction_id)
        if not prediction:
            return {"error": f"Prediction {prediction_id} not found"}

        # Calculate errors
        actual_return = outcome.get("actual_return", 0)
        predicted_return = prediction.get("predicted_return", 0)
        predicted_confidence = prediction.get("predicted_confidence", 0.5)

        direction_correct = (predicted_return > 0) == (actual_return > 0) if predicted_return != 0 else False

        errors = {
            "actual_return": actual_return,
            "was_profitable": 1 if actual_return > 0 else 0,
            "return_prediction_error": abs(predicted_return - actual_return),
            "direction_correct": 1 if direction_correct else 0,
            "confidence_calibration_error": abs(
                predicted_confidence - (1.0 if actual_return > 0 else 0.0)
            ),
        }

        if outcome.get("actual_hold_seconds"):
            errors["actual_hold_seconds"] = outcome["actual_hold_seconds"]
        if outcome.get("actual_max_drawdown"):
            errors["actual_max_drawdown"] = outcome["actual_max_drawdown"]

        # Update in memory
        self.memory.db.update_prediction(prediction_id, errors)

        # Update calibration bins
        self._update_calibration(predicted_confidence, actual_return > 0)

        self.logger.info(
            f"Outcome for {prediction_id}: actual={actual_return:.2%}, "
            f"direction_correct={direction_correct}"
        )

        return {"prediction_id": prediction_id, "errors": errors}

    # ─────────────────────────────────────────────────────────────────
    # CONFIDENCE CALIBRATION
    # ─────────────────────────────────────────────────────────────────

    def _initialize_bins(self) -> Dict[str, Dict]:
        """Create confidence bins for calibration."""
        bins = {}
        for i in range(0, 100, 10):
            key = f"{i}-{i+10}%"
            bins[key] = {"predictions": 0, "wins": 0}
        return bins

    def _update_calibration(self, confidence: float, was_win: bool):
        """Update calibration bins with a new data point."""
        bin_idx = min(int(confidence * 10), 9)
        key = f"{bin_idx * 10}-{bin_idx * 10 + 10}%"
        self._calibration_bins[key]["predictions"] += 1
        if was_win:
            self._calibration_bins[key]["wins"] += 1

    def get_calibrated_confidence(self, raw_confidence: float, context: Dict = None) -> float:
        """
        Adjust model confidence based on historical calibration.

        Args:
            raw_confidence: Model's raw confidence score (0-1).
            context: Optional dict with market_regime, strategy_id, etc.

        Returns:
            Calibrated confidence that better reflects actual win probability.
        """
        bin_idx = min(int(raw_confidence * 10), 9)
        key = f"{bin_idx * 10}-{bin_idx * 10 + 10}%"
        bin_data = self._calibration_bins[key]

        if bin_data["predictions"] < 10:
            return raw_confidence

        actual_win_rate = bin_data["wins"] / bin_data["predictions"]

        # Blend raw confidence with observed rate (shrinkage)
        n = bin_data["predictions"]
        shrinkage = min(n / 50, 1.0)
        calibrated = shrinkage * actual_win_rate + (1 - shrinkage) * raw_confidence

        return max(0.0, min(1.0, calibrated))

    def get_calibration_report(self) -> Dict:
        """Generate calibration report."""
        report = {"bins": {}}
        total_preds = 0
        total_correct = 0

        for key, data in self._calibration_bins.items():
            n = data["predictions"]
            wins = data["wins"]
            actual_rate = wins / n if n > 0 else 0.0

            report["bins"][key] = {
                "predictions": n,
                "wins": wins,
                "actual_win_rate": round(actual_rate, 4),
            }
            total_preds += n
            total_correct += wins

        # Also pull from database for persistence
        db_accuracy = self.memory.get_prediction_accuracy_history(lookback_days=30)
        report["recent_accuracy"] = db_accuracy
        report["total_predictions"] = total_preds
        report["overall_accuracy"] = total_correct / total_preds if total_preds > 0 else 0.0

        return report

    # ─────────────────────────────────────────────────────────────────
    # MODEL WEIGHTS
    # ─────────────────────────────────────────────────────────────────

    def get_model_weights(self, lookback_days: int = 30) -> Dict[str, float]:
        """
        Calculate current ensemble weights based on recent accuracy.
        """
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        predictions = self.memory.get_predictions(start_date=cutoff)

        by_model: Dict[str, Dict] = {}
        for pred in predictions:
            model = pred.get("primary_model", "unknown")
            if model not in by_model:
                by_model[model] = {"correct": 0, "total": 0}
            by_model[model]["total"] += 1
            if pred.get("direction_correct"):
                by_model[model]["correct"] += 1

        if not by_model:
            return {}

        accuracies = {}
        for model, stats in by_model.items():
            accuracies[model] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.5

        # Softmax-like weighting
        total = sum(accuracies.values())
        if total == 0:
            return {m: 1.0 / len(accuracies) for m in accuracies}

        return {model: acc / total for model, acc in accuracies.items()}

    # ─────────────────────────────────────────────────────────────────
    # DEGRADATION DETECTION
    # ─────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────
    # LLM-POWERED CALIBRATION ANALYSIS
    # ─────────────────────────────────────────────────────────────────

    def llm_calibration_analysis(
        self,
        metrics: Dict,
        regime_breakdown: Dict,
    ) -> Dict:
        """
        Use Gemini to analyze prediction calibration and recommend adjustments.

        Examines calibration bins, regime-specific accuracy, and model
        degradation to produce confidence scaling recommendations.

        Returns:
            Dict with confidence_adjustments, regime_overrides, and reasoning.
        """
        if not self.llm:
            return {"error": "No LLM available", "confidence_adjustments": {}}

        # Build calibration snapshot
        cal_report = self.get_calibration_report()
        bins_summary = []
        for key, data in cal_report.get("bins", {}).items():
            n = data.get("predictions", 0)
            if n > 0:
                bins_summary.append(
                    f"{key}: {n} preds, actual WR={data.get('actual_win_rate', 0):.0%}"
                )

        # Degradation alerts
        degradation = self.detect_model_degradation()
        deg_str = "; ".join(
            f"{a['model']}: {a['recent_accuracy']:.0%} vs {a['historical_accuracy']:.0%}"
            for a in degradation
        ) if degradation else "None"

        # Regime performance
        regime_str = ", ".join(
            f"{r}: {s.get('trades', 0)}t/{s.get('wins', 0)}w"
            for r, s in regime_breakdown.items()
        ) if regime_breakdown else "N/A"

        overall_acc = cal_report.get("overall_accuracy", 0)
        total_preds = cal_report.get("total_predictions", 0)

        prompt = f"""You are AQTIS prediction calibration analyst. Analyze prediction accuracy and recommend confidence adjustments.

CALIBRATION BINS:
{chr(10).join(bins_summary) if bins_summary else "No calibration data yet"}

OVERALL: {total_preds} predictions, accuracy={overall_acc:.0%}
DEGRADATION ALERTS: {deg_str}
REGIME BREAKDOWN: {regime_str}

PERFORMANCE: WR={metrics.get('win_rate', 0):.0%}, Sharpe={metrics.get('sharpe_ratio', 0):.2f}

Respond in JSON:
{{
  "confidence_scaling": float (0.5-1.5, multiply raw confidence by this),
  "regime_overrides": {{"regime_name": scaling_float}},
  "min_confidence_threshold": float (0.3-0.8),
  "reasoning": "2-3 sentences on calibration state"
}}

If predictions are well-calibrated, confidence_scaling should be 1.0."""

        try:
            result = self.llm.generate_json(prompt)
            if not isinstance(result, dict):
                return {"confidence_scaling": 1.0, "reasoning": "LLM returned non-dict"}

            self.logger.info(
                f"LLM calibration analysis: {result.get('reasoning', '')[:100]}"
            )
            return result
        except Exception as e:
            self.logger.warning(f"LLM calibration analysis failed: {e}")
            return {"confidence_scaling": 1.0, "error": str(e)}

    # ─────────────────────────────────────────────────────────────────
    # DEGRADATION DETECTION
    # ─────────────────────────────────────────────────────────────────

    def detect_model_degradation(self, lookback_days: int = 30) -> List[Dict]:
        """
        Check if any models are significantly degrading.
        """
        now = datetime.now()
        recent_cutoff = (now - timedelta(days=lookback_days)).isoformat()
        historical_start = (now - timedelta(days=lookback_days * 3)).isoformat()
        historical_end = recent_cutoff

        recent = self.memory.get_predictions(start_date=recent_cutoff)
        historical = self.memory.get_predictions(
            start_date=historical_start, end_date=historical_end
        )

        def calc_accuracy(preds):
            if not preds:
                return 0.5
            correct = sum(1 for p in preds if p.get("direction_correct"))
            return correct / len(preds)

        # Group by model
        models = set()
        for p in recent + historical:
            if p.get("primary_model"):
                models.add(p["primary_model"])

        alerts = []
        for model in models:
            recent_model = [p for p in recent if p.get("primary_model") == model]
            hist_model = [p for p in historical if p.get("primary_model") == model]

            if len(recent_model) < 5 or len(hist_model) < 5:
                continue

            recent_acc = calc_accuracy(recent_model)
            hist_acc = calc_accuracy(hist_model)

            if hist_acc > 0 and recent_acc < hist_acc * 0.8:
                alerts.append({
                    "model": model,
                    "recent_accuracy": round(recent_acc, 4),
                    "historical_accuracy": round(hist_acc, 4),
                    "degradation_pct": round((hist_acc - recent_acc) / hist_acc, 4),
                    "recommendation": "Consider retraining or reducing weight",
                })

        return alerts
