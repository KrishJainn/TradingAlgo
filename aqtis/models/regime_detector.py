"""
AQTIS Market Regime Detector.

Classifies market regimes using Hidden Markov Models or clustering.
Regimes: trending_up, trending_down, mean_reverting, high_volatility, low_volatility.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIME_NAMES = {
    0: "low_volatility",
    1: "trending_up",
    2: "mean_reverting",
    3: "trending_down",
    4: "high_volatility",
}


class RegimeDetector:
    """
    Market regime detection using statistical features and clustering.

    Supports two backends:
    - HMM (hmmlearn) for probabilistic regime detection
    - K-Means clustering as fallback
    """

    def __init__(self, n_regimes: int = 5, lookback_days: int = 60):
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self._model = None
        self._fitted = False
        self._scaler = None

    def _compute_features(self, prices: pd.Series) -> pd.DataFrame:
        """
        Compute regime detection features from price series.

        Features:
        - Returns (1d, 5d, 20d)
        - Volatility (20d realized)
        - Trend strength (SMA ratio)
        - Mean reversion score
        """
        df = pd.DataFrame(index=prices.index)

        # Returns at different horizons
        df["ret_1d"] = prices.pct_change(1)
        df["ret_5d"] = prices.pct_change(5)
        df["ret_20d"] = prices.pct_change(20)

        # Realized volatility
        df["vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)

        # Trend strength: ratio of 20-day SMA to 60-day SMA
        sma_20 = prices.rolling(20).mean()
        sma_60 = prices.rolling(60).mean()
        df["trend_strength"] = (sma_20 / sma_60 - 1) * 100

        # Mean reversion score: how far from rolling mean
        df["mean_rev_score"] = (prices - sma_20) / sma_20 * 100

        return df.dropna()

    def fit(self, prices: pd.Series):
        """
        Fit the regime detector on historical price data.

        Args:
            prices: Close price series.
        """
        features_df = self._compute_features(prices)
        if features_df.empty or len(features_df) < 60:
            logger.warning("Insufficient data for regime detection")
            return

        from sklearn.preprocessing import StandardScaler
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(features_df.values)

        try:
            from hmmlearn.hmm import GaussianHMM
            model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=200,
                random_state=42,
            )
            model.fit(X)
            self._model = model
            self._fitted = True
            logger.info(f"HMM regime detector fitted with {self.n_regimes} regimes")
        except ImportError:
            logger.info("hmmlearn not available, using KMeans fallback")
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=self.n_regimes, random_state=42, n_init=10)
            model.fit(X)
            self._model = model
            self._fitted = True
            logger.info(f"KMeans regime detector fitted with {self.n_regimes} regimes")

    def detect_regime(self, prices: pd.Series) -> Dict:
        """
        Detect current market regime.

        Args:
            prices: Recent close price series (at least lookback_days).

        Returns:
            Dict with regime name, index, and probabilities.
        """
        if not self._fitted:
            # Auto-fit on provided data
            self.fit(prices)

        if not self._fitted:
            return self._rule_based_regime(prices)

        features_df = self._compute_features(prices)
        if features_df.empty:
            return self._rule_based_regime(prices)

        X = self._scaler.transform(features_df.values)

        # Get regime for most recent data point
        try:
            from hmmlearn.hmm import GaussianHMM
            if isinstance(self._model, GaussianHMM):
                states = self._model.predict(X)
                current_state = int(states[-1])
                probs = self._model.predict_proba(X)[-1]
                return {
                    "regime_index": current_state,
                    "regime_name": REGIME_NAMES.get(current_state, f"regime_{current_state}"),
                    "probabilities": {
                        REGIME_NAMES.get(i, f"regime_{i}"): float(p)
                        for i, p in enumerate(probs)
                    },
                    "method": "hmm",
                }
        except (ImportError, AttributeError):
            pass

        # KMeans fallback
        labels = self._model.predict(X)
        current_label = int(labels[-1])
        return {
            "regime_index": current_label,
            "regime_name": REGIME_NAMES.get(current_label, f"regime_{current_label}"),
            "probabilities": {},
            "method": "kmeans",
        }

    def get_regime_history(self, prices: pd.Series) -> pd.DataFrame:
        """Get regime classification over time."""
        if not self._fitted:
            self.fit(prices)

        if not self._fitted:
            return pd.DataFrame()

        features_df = self._compute_features(prices)
        if features_df.empty:
            return pd.DataFrame()

        X = self._scaler.transform(features_df.values)

        try:
            from hmmlearn.hmm import GaussianHMM
            if isinstance(self._model, GaussianHMM):
                states = self._model.predict(X)
            else:
                states = self._model.predict(X)
        except (ImportError, AttributeError):
            states = self._model.predict(X)

        result = pd.DataFrame(index=features_df.index)
        result["regime_index"] = states
        result["regime_name"] = [REGIME_NAMES.get(s, f"regime_{s}") for s in states]

        return result

    def _rule_based_regime(self, prices: pd.Series) -> Dict:
        """Simple rule-based regime detection as fallback."""
        if len(prices) < 20:
            return {"regime_name": "unknown", "regime_index": -1, "method": "rule_based"}

        ret_20 = (prices.iloc[-1] / prices.iloc[-20] - 1)
        vol_20 = prices.pct_change().tail(20).std() * np.sqrt(252)

        if vol_20 > 0.4:
            regime = "high_volatility"
            idx = 4
        elif vol_20 < 0.1:
            regime = "low_volatility"
            idx = 0
        elif ret_20 > 0.05:
            regime = "trending_up"
            idx = 1
        elif ret_20 < -0.05:
            regime = "trending_down"
            idx = 3
        else:
            regime = "mean_reverting"
            idx = 2

        return {
            "regime_name": regime,
            "regime_index": idx,
            "method": "rule_based",
            "return_20d": float(ret_20),
            "vol_20d": float(vol_20),
        }
