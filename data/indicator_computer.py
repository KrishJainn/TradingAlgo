"""
AQTIS Indicator Pre-computation.

Computes all technical indicators for cached OHLCV data,
bridging to the trading_evolution indicator engine when available
and falling back to the `ta` library otherwise.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Allow importing trading_evolution
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class IndicatorComputer:
    """Pre-computes all indicators for cached OHLCV data."""

    def __init__(self):
        """Initialize -- tries to load full trading_evolution calculator."""
        self._full_calculator = None
        self._normalizer = None
        self._has_full = False

        try:
            from trading_evolution.indicators.calculator import IndicatorCalculator
            from trading_evolution.indicators.normalizer import IndicatorNormalizer
            self._full_calculator = IndicatorCalculator()
            self._normalizer = IndicatorNormalizer()
            self._has_full = True
            logger.info("IndicatorComputer: using trading_evolution full calculator (87+ indicators)")
        except ImportError:
            logger.info("IndicatorComputer: trading_evolution not available, using ta-lib fallback")

    def compute_all_indicators(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators for a single symbol's OHLCV data.

        Args:
            ohlcv_df: DataFrame with columns Open, High, Low, Close, Volume.

        Returns:
            DataFrame with all indicator columns (same index as input).
        """
        if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 30:
            return pd.DataFrame(index=ohlcv_df.index if ohlcv_df is not None else [])

        if self._has_full:
            return self._compute_full(ohlcv_df)
        return self._compute_fallback(ohlcv_df)

    # ------------------------------------------------------------------
    # Full calculator (87+ indicators via trading_evolution)
    # ------------------------------------------------------------------

    def _compute_full(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Compute using the full trading_evolution indicator engine."""
        df = ohlcv_df.copy()
        col_lower = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_lower)

        try:
            raw = self._full_calculator.calculate_all(df)
            if hasattr(self._full_calculator, "rename_to_dna_names"):
                raw = self._full_calculator.rename_to_dna_names(raw)

            # Also compute normalized version
            if self._normalizer and "close" in df.columns:
                normalized = self._normalizer.normalize_all(raw, price_series=df["close"])
                norm_cols = {c: f"{c}_norm" for c in normalized.columns}
                normalized = normalized.rename(columns=norm_cols)
                raw = raw.join(normalized, how="left")

            raw = raw.ffill(limit=3).bfill(limit=3)
            return raw

        except Exception as e:
            logger.warning(f"Full indicator computation failed, falling back: {e}")
            return self._compute_fallback(ohlcv_df)

    # ------------------------------------------------------------------
    # Fallback calculator (ta library)
    # ------------------------------------------------------------------

    def _compute_fallback(self, ohlcv_df: pd.DataFrame) -> pd.DataFrame:
        """Compute indicators using the `ta` library."""
        try:
            from ta import momentum, trend, volatility, volume as ta_volume
        except ImportError:
            logger.error("'ta' library required: pip install ta")
            return pd.DataFrame(index=ohlcv_df.index)

        df = ohlcv_df.copy()
        col_map = {c: c.lower() for c in df.columns}
        df = df.rename(columns=col_map)

        o, h, l, c, v = df["open"], df["high"], df["low"], df["close"], df["volume"]
        result = pd.DataFrame(index=df.index)

        # --- RSI ---
        result["RSI_14"] = momentum.RSIIndicator(c, 14).rsi()
        result["RSI_7"] = momentum.RSIIndicator(c, 7).rsi()
        result["RSI_21"] = momentum.RSIIndicator(c, 21).rsi()

        # --- MACD ---
        macd = trend.MACD(c, window_slow=26, window_fast=12, window_sign=9)
        result["MACD_12_26_9"] = macd.macd()
        result["MACD_signal"] = macd.macd_signal()
        result["MACD_hist"] = macd.macd_diff()

        # --- Bollinger Bands ---
        bb = volatility.BollingerBands(c, window=20, window_dev=2)
        result["BB_high"] = bb.bollinger_hband()
        result["BB_low"] = bb.bollinger_lband()
        result["BB_mid"] = bb.bollinger_mavg()
        result["BB_pct"] = bb.bollinger_pband()
        result["BB_width"] = bb.bollinger_wband()

        # --- ATR ---
        result["ATR_14"] = volatility.AverageTrueRange(h, l, c, 14).average_true_range()

        # --- ADX ---
        adx_ind = trend.ADXIndicator(h, l, c, 14)
        result["ADX_14"] = adx_ind.adx()
        result["DI_plus_14"] = adx_ind.adx_pos()
        result["DI_minus_14"] = adx_ind.adx_neg()

        # --- SMA ---
        result["SMA_20"] = trend.SMAIndicator(c, 20).sma_indicator()
        result["SMA_50"] = trend.SMAIndicator(c, 50).sma_indicator()
        if len(c) >= 200:
            result["SMA_200"] = trend.SMAIndicator(c, 200).sma_indicator()

        # --- EMA ---
        result["EMA_9"] = trend.EMAIndicator(c, 9).ema_indicator()
        result["EMA_21"] = trend.EMAIndicator(c, 21).ema_indicator()
        result["EMA_55"] = trend.EMAIndicator(c, 55).ema_indicator()

        # --- VWAP (simplified cumulative) ---
        typical_price = (h + l + c) / 3
        result["VWAP"] = (typical_price * v).cumsum() / v.cumsum()

        # --- OBV ---
        result["OBV"] = ta_volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()

        # --- Stochastic ---
        stoch = momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
        result["STOCH_k"] = stoch.stoch()
        result["STOCH_d"] = stoch.stoch_signal()

        # --- CCI ---
        result["CCI_20"] = trend.CCIIndicator(h, l, c, 20).cci()

        # --- Williams %R ---
        result["WILLR_14"] = momentum.WilliamsRIndicator(h, l, c, 14).williams_r()

        # --- MFI ---
        result["MFI_14"] = ta_volume.MFIIndicator(h, l, c, v, 14).money_flow_index()

        # --- Keltner Channel ---
        kc = volatility.KeltnerChannel(h, l, c, window=20)
        result["KC_high"] = kc.keltner_channel_hband()
        result["KC_low"] = kc.keltner_channel_lband()
        result["KC_mid"] = kc.keltner_channel_mband()

        # --- Donchian Channel ---
        dc = volatility.DonchianChannel(h, l, c, window=20)
        result["DC_high"] = dc.donchian_channel_hband()
        result["DC_low"] = dc.donchian_channel_lband()

        # --- CMF ---
        result["CMF_20"] = ta_volume.ChaikinMoneyFlowIndicator(h, l, c, v, 20).chaikin_money_flow()

        # --- Aroon ---
        aroon = trend.AroonIndicator(h, l, 25)
        result["AROON_up"] = aroon.aroon_up()
        result["AROON_down"] = aroon.aroon_down()

        # --- PSAR ---
        psar = trend.PSARIndicator(h, l, c)
        result["PSAR"] = psar.psar()

        # --- Normalized versions (z-score + tanh) ---
        for col in list(result.columns):
            series = result[col]
            expanding_mean = series.expanding(20).mean()
            expanding_std = series.expanding(20).std().replace(0, np.nan)
            z = (series - expanding_mean) / expanding_std
            result[f"{col}_norm"] = np.tanh(z * 0.5).clip(-1, 1)

        result = result.ffill(limit=3).bfill(limit=3)
        return result
