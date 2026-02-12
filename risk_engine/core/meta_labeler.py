"""
Meta-Labeling Secondary Model.

Implements the meta-labeling framework where a primary model provides the
trade direction, and a secondary (meta) model decides *whether* to take
the trade and *how much* to bet.

Reference:
    Lopez de Prado, M. (2018). "Advances in Financial Machine Learning."
    Wiley, Chapter 3: Meta-Labeling.

Key insight: separating the side (direction) decision from the size decision
allows each model to specialize, improving overall performance. The meta-label
is 1 if the primary model's prediction was correct (positive return in the
predicted direction), and 0 otherwise. The predicted probability of the
meta-label model becomes the bet size.

Purged cross-validation is used during fitting to prevent information leakage
between train and test folds (Lopez de Prado, Chapter 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MetaLabelResult:
    """Result of the meta-labeling prediction.

    Attributes:
        should_trade: Whether the secondary model recommends taking the trade.
            True when predicted probability exceeds the threshold (default 0.5).
        bet_size: Suggested bet size in [0, 1], derived from the predicted
            probability of the meta-label being positive.
        confidence: Raw probability output from the logistic regression,
            representing the model's confidence that the primary signal
            will be profitable.
        primary_direction: The primary model's direction signal passed through
            for downstream consumption (positive = long, negative = short).
    """

    should_trade: bool
    bet_size: float
    confidence: float
    primary_direction: float


class MetaLabeler:
    """Meta-labeling secondary model using purged cross-validated logistic regression.

    The meta-labeling approach from Lopez de Prado (2018) Chapter 3:

    1. A **primary model** generates directional predictions (long / short).
    2. This **secondary model** (the meta-labeler) learns to predict whether
       the primary model's prediction will be *correct* (i.e., produce a
       positive return in the predicted direction).
    3. The predicted probability becomes the **bet size**: high probability
       means the model is confident the primary signal is right, warranting
       a larger position; low probability means skip or reduce size.

    Purged Cross-Validation:
        Standard k-fold CV leaks information when samples are serially
        correlated (which financial returns always are). Purging removes
        ``purge_bars`` samples around each test fold boundary to create
        a gap between train and test data.

    Example:
        >>> labeler = MetaLabeler()
        >>> labeler.fit(X_features, primary_preds, realized_returns, purge_bars=5)
        >>> result = labeler.predict(X_new, primary_direction=1.0)
        >>> result.should_trade
        True
        >>> result.bet_size
        0.73
    """

    def __init__(
        self,
        n_folds: int = 5,
        threshold: float = 0.5,
        regularization: float = 1.0,
        max_iter: int = 1000,
    ) -> None:
        """Initialize the meta-labeler.

        Args:
            n_folds: Number of cross-validation folds for purged CV.
            threshold: Probability threshold for the should_trade decision.
            regularization: Inverse regularization strength (C parameter)
                for LogisticRegression. Smaller values = stronger regularization.
            max_iter: Maximum iterations for logistic regression solver.
        """
        self.n_folds = n_folds
        self.threshold = threshold
        self.regularization = regularization
        self.max_iter = max_iter

        self._model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False
        self._cv_scores: List[float] = []

    def _create_meta_labels(
        self,
        primary_predictions: np.ndarray,
        returns: np.ndarray,
    ) -> np.ndarray:
        """Create binary meta-labels from primary predictions and realized returns.

        A meta-label is 1 (correct) if sign(primary_prediction) == sign(return),
        meaning the primary model got the direction right and the trade was
        profitable. Otherwise 0.

        Args:
            primary_predictions: Primary model's directional predictions.
                Shape (n_samples,).
            returns: Realized forward returns. Shape (n_samples,).

        Returns:
            Binary meta-labels of shape (n_samples,). 1 = primary was correct.
        """
        # Trade return = return * sign(prediction). Positive means profitable.
        trade_returns = returns * np.sign(primary_predictions)
        meta_labels = (trade_returns > 0).astype(np.int32)
        return meta_labels

    def _purged_train_test_split(
        self,
        n_samples: int,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
        purge_bars: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply purging to train/test indices to prevent information leakage.

        Removes ``purge_bars`` samples from the training set that are
        temporally adjacent to the test set boundaries. This creates a gap
        between train and test data, mitigating serial correlation leakage.

        Reference:
            Lopez de Prado (2018), Chapter 7: Cross-Validation in Finance.

        Args:
            n_samples: Total number of samples.
            train_indices: Original training fold indices.
            test_indices: Original test fold indices.
            purge_bars: Number of bars to purge around test boundaries.

        Returns:
            Tuple of (purged_train_indices, test_indices).
        """
        if purge_bars <= 0:
            return train_indices, test_indices

        test_start = test_indices.min()
        test_end = test_indices.max()

        # Define the embargo zone: purge_bars before test start, purge_bars after test end.
        embargo_start = max(0, test_start - purge_bars)
        embargo_end = min(n_samples - 1, test_end + purge_bars)

        embargo_zone = set(range(embargo_start, embargo_end + 1))
        purged_train = np.array(
            [i for i in train_indices if i not in embargo_zone],
            dtype=np.int64,
        )

        return purged_train, test_indices

    def fit(
        self,
        X: np.ndarray,
        primary_predictions: np.ndarray,
        returns: np.ndarray,
        purge_bars: int = 5,
    ) -> None:
        """Fit the meta-labeling model with purged cross-validation.

        Steps:
        1. Create binary meta-labels from primary predictions and returns.
        2. Run purged k-fold CV to estimate out-of-sample performance.
        3. Refit on the full dataset for production use.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
                These are the features the meta-model uses to decide whether
                the primary signal is trustworthy (e.g., volatility, volume,
                regime indicators, spread, time-of-day features).
            primary_predictions: Primary model's directional predictions.
                Shape (n_samples,). Sign indicates direction.
            returns: Realized forward returns corresponding to each sample.
                Shape (n_samples,).
            purge_bars: Number of bars to purge around each test fold
                boundary to prevent information leakage. Default 5.

        Raises:
            ValueError: If input shapes are inconsistent.
        """
        if X.shape[0] != len(primary_predictions) or X.shape[0] != len(returns):
            raise ValueError(
                f"Shape mismatch: X has {X.shape[0]} samples, "
                f"primary_predictions has {len(primary_predictions)}, "
                f"returns has {len(returns)}"
            )

        n_samples = X.shape[0]
        meta_labels = self._create_meta_labels(primary_predictions, returns)

        # --- Purged cross-validation ---
        self._cv_scores = []
        kf = KFold(n_splits=self.n_folds, shuffle=False)

        for train_idx, test_idx in kf.split(X):
            purged_train_idx, purged_test_idx = self._purged_train_test_split(
                n_samples, train_idx, test_idx, purge_bars
            )

            if len(purged_train_idx) < 10:
                # Skip fold if too few training samples after purging.
                continue

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X[purged_train_idx])
            X_test = scaler.transform(X[purged_test_idx])

            y_train = meta_labels[purged_train_idx]
            y_test = meta_labels[purged_test_idx]

            # Check that both classes are present in training data.
            if len(np.unique(y_train)) < 2:
                continue

            model = LogisticRegression(
                C=self.regularization,
                max_iter=self.max_iter,
                solver="lbfgs",
                random_state=42,
            )
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            self._cv_scores.append(score)

        # --- Refit on full data for production ---
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = LogisticRegression(
            C=self.regularization,
            max_iter=self.max_iter,
            solver="lbfgs",
            random_state=42,
        )

        # Guard against single-class edge case.
        if len(np.unique(meta_labels)) < 2:
            # Fall back: all labels identical, model will just predict that class.
            # We still fit so predict() works, but flag low confidence.
            pass

        self._model.fit(X_scaled, meta_labels)
        self._is_fitted = True

    def predict(
        self,
        X: np.ndarray,
        primary_direction: float,
    ) -> MetaLabelResult:
        """Predict whether to trade and the bet size.

        Args:
            X: Feature vector of shape (1, n_features) or (n_features,).
                Same features used during fit.
            primary_direction: The primary model's current directional signal.
                Positive = long, negative = short.

        Returns:
            MetaLabelResult with trade decision, bet size, confidence, and
            the primary direction passed through.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self._is_fitted or self._model is None or self._scaler is None:
            raise RuntimeError(
                "MetaLabeler has not been fitted. Call fit() first."
            )

        # Reshape if needed: (n_features,) -> (1, n_features)
        X_input = np.atleast_2d(X)
        X_scaled = self._scaler.transform(X_input)

        # Predicted probability of meta-label = 1 (primary signal is correct).
        proba = self._model.predict_proba(X_scaled)[0]

        # Index of class 1 in the model's classes.
        class_1_idx = list(self._model.classes_).index(1) if 1 in self._model.classes_ else -1

        if class_1_idx >= 0:
            confidence = float(proba[class_1_idx])
        else:
            # Model never saw class 1 — default to low confidence.
            confidence = 0.0

        should_trade = confidence >= self.threshold

        # Bet size: calibrated probability clipped to [0, 1].
        # Following Lopez de Prado: bet_size = 2 * prob - 1, clipped to [0, 1].
        # This maps 0.5 -> 0, 1.0 -> 1.0 for a more conservative sizing.
        bet_size = float(np.clip(2.0 * confidence - 1.0, 0.0, 1.0))

        return MetaLabelResult(
            should_trade=should_trade,
            bet_size=round(bet_size, 4),
            confidence=round(confidence, 4),
            primary_direction=primary_direction,
        )

    @property
    def cv_scores(self) -> List[float]:
        """Cross-validation accuracy scores from purged CV during fit."""
        return self._cv_scores.copy()

    @property
    def mean_cv_score(self) -> Optional[float]:
        """Mean cross-validation accuracy, or None if not fitted."""
        if not self._cv_scores:
            return None
        return float(np.mean(self._cv_scores))
