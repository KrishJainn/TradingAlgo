"""
AQTIS Indicator Bridge.

Wraps trading_evolution's IndicatorCalculator and IndicatorNormalizer
to provide feature engineering for ML models and agents.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class IndicatorBridge:
    """
    Bridge to trading_evolution's indicator calculation engine.

    Provides normalized technical features for ML models and trading agents.
    """

    def __init__(self):
        self._calculator = None
        self._feature_names: Optional[List[str]] = None

    @property
    def calculator(self):
        """Lazy-initialize the IndicatorCalculator."""
        if self._calculator is None:
            try:
                from trading_evolution.indicators.calculator import IndicatorCalculator
                self._calculator = IndicatorCalculator()
                logger.info("Using trading_evolution IndicatorCalculator")
            except ImportError:
                logger.warning(
                    "trading_evolution not available, using standalone calculator"
                )
                self._calculator = _StandaloneCalculator()
        return self._calculator

    def calculate_features(
        self,
        ohlcv: pd.DataFrame,
        normalize: bool = True,
        indicators: List[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate technical indicator features from OHLCV data.

        Args:
            ohlcv: DataFrame with columns [open, high, low, close, volume].
            normalize: Whether to normalize to [-1, 1] range.
            indicators: Specific indicators to calculate (default: all).

        Returns:
            DataFrame with indicator features.
        """
        result = self.calculator.calculate_all(
            ohlcv,
            indicators=indicators,
            normalize=normalize,
        )

        # Rename to DNA names if using full calculator
        if hasattr(self.calculator, "rename_to_dna_names"):
            result = self.calculator.rename_to_dna_names(result)

        self._feature_names = list(result.columns)
        return result

    def calculate_at_timestamp(
        self,
        ohlcv: pd.DataFrame,
        timestamp: pd.Timestamp,
        indicators: List[str] = None,
        normalize: bool = True,
    ) -> Dict[str, float]:
        """
        Calculate indicators at a specific timestamp.

        Uses only data available up to that timestamp (no lookahead).
        """
        if hasattr(self.calculator, "calculate_at_timestamp"):
            indicator_names = indicators or self.get_feature_names()
            return self.calculator.calculate_at_timestamp(
                ohlcv, indicator_names, timestamp, normalize=normalize
            )

        # Fallback: calculate on historical slice
        historical = ohlcv.loc[:timestamp].copy()
        result = self.calculate_features(historical, normalize=normalize, indicators=indicators)
        if result.empty or timestamp not in result.index:
            return {}
        return result.loc[timestamp].dropna().to_dict()

    def get_feature_names(self) -> List[str]:
        """Get list of feature names from last calculation."""
        if self._feature_names:
            return self._feature_names

        if hasattr(self.calculator, "universe"):
            return self.calculator.universe.get_all()

        return []

    def get_feature_importance(self, weights: Dict[str, float]) -> List[Dict]:
        """
        Rank features by importance (absolute weight).

        Args:
            weights: Dict mapping indicator name to weight.

        Returns:
            Sorted list of dicts with name, weight, abs_weight.
        """
        items = [
            {"name": name, "weight": w, "abs_weight": abs(w)}
            for name, w in weights.items()
        ]
        return sorted(items, key=lambda x: x["abs_weight"], reverse=True)


class _StandaloneCalculator:
    """Minimal indicator calculator when trading_evolution is unavailable."""

    def calculate_all(self, ohlcv: pd.DataFrame, indicators=None, normalize=False):
        try:
            from ta import momentum, trend, volatility
        except ImportError:
            logger.error("'ta' library required: pip install ta")
            return pd.DataFrame(index=ohlcv.index)

        df = ohlcv.copy()
        df.columns = df.columns.str.lower()
        result = pd.DataFrame(index=df.index)

        result["RSI_14"] = momentum.RSIIndicator(df["close"], 14).rsi()
        m = trend.MACD(df["close"])
        result["MACD_12_26_9"] = m.macd()
        result["ADX_14"] = trend.ADXIndicator(df["high"], df["low"], df["close"], 14).adx()
        result["ATR_14"] = volatility.AverageTrueRange(df["high"], df["low"], df["close"], 14).average_true_range()
        result["SMA_20"] = trend.SMAIndicator(df["close"], 20).sma_indicator()
        result["EMA_20"] = trend.EMAIndicator(df["close"], 20).ema_indicator()
        result["BB_PCT"] = volatility.BollingerBands(df["close"], 20, 2).bollinger_pband()

        if normalize:
            for col in result.columns:
                expanding_mean = result[col].expanding(20).mean()
                expanding_std = result[col].expanding(20).std().replace(0, np.nan)
                z = (result[col] - expanding_mean) / expanding_std
                result[col] = np.tanh(z * 0.5).clip(-1, 1)

        result = result.bfill().ffill()
        return result
