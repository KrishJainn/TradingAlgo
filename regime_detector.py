"""
Market Regime Detection Layer for the 5-Player Coach Trading System.

Uses a 3-state Gaussian Hidden Markov Model trained on daily Nifty 50
features to classify the market as Bull, Bear, or Sideways.  Provides
regime-conditional parameter overrides for:

    - Player capital allocation weights
    - Technical indicator emphasis
    - PortfolioRiskManager limits
    - TransactionCostModel slippage multipliers

Usage:
    from regime_detector import RegimeDetector, load_nifty_data

    df = load_nifty_data("2020-01-01", "2025-12-31")
    rd = RegimeDetector()
    rd.fit(df)
    regime, probs, duration = rd.predict(df.tail(60))
    risk_adj = rd.get_risk_adjustments(regime)
"""

from __future__ import annotations

import logging
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REGIME_LABELS = ("Bull", "Bear", "Sideways")

_PROJECT_ROOT = Path(__file__).parent
_MODEL_PATH = _PROJECT_ROOT / "regime_hmm_model.joblib"
_CACHE_DIR = _PROJECT_ROOT / "data" / "nifty_cache"

# Player IDs used by the 5-player coach system
PLAYER_IDS = {
    "PLAYER_1": "Aggressive",
    "PLAYER_2": "Conservative",
    "PLAYER_3": "Balanced",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "Momentum",
}


# ---------------------------------------------------------------------------
# Regime-conditional lookup tables
# ---------------------------------------------------------------------------

# Requirement 5 — player allocation weights per regime
_PLAYER_WEIGHTS: Dict[str, Dict[str, float]] = {
    "Bull": {
        "Momentum":     0.30,
        "Aggressive":   0.25,
        "Balanced":     0.20,
        "VolBreakout":  0.10,   # mapped from "Contrarian" role
        "Conservative": 0.15,
    },
    "Bear": {
        "Momentum":     0.10,
        "Aggressive":   0.10,
        "Balanced":     0.15,
        "VolBreakout":  0.30,
        "Conservative": 0.35,
    },
    "Sideways": {
        "Momentum":     0.15,
        "Aggressive":   0.15,
        "Balanced":     0.30,
        "VolBreakout":  0.20,
        "Conservative": 0.20,
    },
}

# Requirement 6 — indicator emphasis multipliers per regime
# >1 means upweight, <1 means downweight, 1.0 means no change.
_INDICATOR_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "Bull": {
        # Trend-following up
        "SUPERTREND_7_3": 1.4, "MACD_12_26_9": 1.3, "ADX_14": 1.3,
        "EMA_50": 1.2, "AROON_14": 1.2, "DEMA_20": 1.2, "HMA_9": 1.1,
        # Mean-reversion down
        "BBANDS_20_2": 0.7, "RSI_14": 0.8, "RSI_7": 0.8,
        "STOCH_14_3": 0.7, "STOCH_5_3": 0.7, "ZSCORE_20": 0.7,
        # Volatility neutral
        "ATR_14": 1.0, "NATR_14": 1.0,
    },
    "Bear": {
        # Volatility indicators up — only reliable signals in bear markets
        "ATR_14": 1.5, "NATR_14": 1.5, "BBANDS_20_2": 1.4, "KC_20_2": 1.3,
        # Momentum indicators SUPPRESSED — give false signals in bear
        "RSI_7": 0.2, "RSI_14": 0.2, "STOCH_5_3": 0.2, "STOCH_14_3": 0.2,
        "CCI_14": 0.2, "WILLR_14": 0.3, "CMO_14": 0.2, "MFI_14": 0.3,
        "UO_7_14_28": 0.2, "TSI_13_25": 0.3, "MACD_8_17_9": 0.3,
        # Trend-following down
        "SUPERTREND_7_3": 0.5, "MACD_12_26_9": 0.6, "ADX_14": 0.7,
        "EMA_50": 0.6, "AROON_14": 0.6, "DEMA_20": 0.7,
        # PSAR/Donchian neutral
        "PSAR": 0.8, "DONCHIAN_20": 0.9,
    },
    "Sideways": {
        # Range-bound up
        "RSI_14": 1.3, "STOCH_14_3": 1.3, "BBANDS_20_2": 1.3,
        "RSI_7": 1.2, "STOCH_5_3": 1.2, "ZSCORE_20": 1.2,
        "MFI_14": 1.1, "MFI_20": 1.1, "CMF_20": 1.1,
        # Trend-following down
        "SUPERTREND_7_3": 0.7, "MACD_12_26_9": 0.8, "ADX_14": 0.8,
        "EMA_50": 0.8, "AROON_14": 0.8, "DEMA_20": 0.9,
        # Volatility neutral
        "ATR_14": 1.0, "NATR_14": 1.0,
    },
}

# Requirement 7 — risk parameter overrides per regime
@dataclass
class RegimeRiskOverrides:
    """Risk-manager parameter overrides for a given regime."""
    max_single_stock_pct: float
    max_sector_pct: float
    max_gross_exposure_pct: float
    trailing_stop_largecap_pct: float
    trailing_stop_midcap_pct: float
    slippage_multiplier: float          # for TransactionCostModel

    def to_dict(self) -> Dict[str, float]:
        return {
            "max_single_stock_pct": self.max_single_stock_pct,
            "max_sector_pct": self.max_sector_pct,
            "max_gross_exposure_pct": self.max_gross_exposure_pct,
            "trailing_stop_largecap_pct": self.trailing_stop_largecap_pct,
            "trailing_stop_midcap_pct": self.trailing_stop_midcap_pct,
            "slippage_multiplier": self.slippage_multiplier,
        }


_RISK_OVERRIDES: Dict[str, RegimeRiskOverrides] = {
    "Bull": RegimeRiskOverrides(
        max_single_stock_pct=0.25,
        max_sector_pct=0.40,
        max_gross_exposure_pct=2.00,
        trailing_stop_largecap_pct=0.025,   # looser
        trailing_stop_midcap_pct=0.035,
        slippage_multiplier=0.8,            # good liquidity
    ),
    "Bear": RegimeRiskOverrides(
        max_single_stock_pct=0.15,
        max_sector_pct=0.30,
        max_gross_exposure_pct=1.20,
        trailing_stop_largecap_pct=0.012,   # much tighter
        trailing_stop_midcap_pct=0.018,
        slippage_multiplier=1.5,            # poor liquidity, wide spreads
    ),
    "Sideways": RegimeRiskOverrides(
        max_single_stock_pct=0.20,
        max_sector_pct=0.35,
        max_gross_exposure_pct=1.50,
        trailing_stop_largecap_pct=0.018,
        trailing_stop_midcap_pct=0.025,
        slippage_multiplier=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_nifty_data(
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    symbol: str = "^NSEI",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download Nifty 50 index daily OHLCV from yfinance, with local CSV cache.

    Args:
        start_date: ISO date string.
        end_date:   ISO date string (None = today).
        symbol:     Yahoo Finance ticker (^NSEI for Nifty 50).
        cache:      If True, read/write a CSV cache under data/nifty_cache/.

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume
        (Date is also set as the index.)
    """
    import yfinance as yf

    end_date = end_date or pd.Timestamp.now().strftime("%Y-%m-%d")

    # Cache key uses only start_date + symbol (NOT end_date) when end_date
    # was left as the default (today).  This way yesterday's cache is still
    # a hit — we just append any new rows after loading.
    cache_key = hashlib.md5(f"{symbol}_{start_date}".encode()).hexdigest()[:12]
    cache_path = _CACHE_DIR / f"{cache_key}.csv"

    if cache and cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["Date"], index_col="Date")
        last_cached = df.index.max().strftime("%Y-%m-%d") if len(df) else start_date

        # If the cache already covers up to end_date (or close), reuse it
        if last_cached >= end_date:
            logger.info(f"Loaded cached Nifty data: {len(df)} rows from {cache_path.name}")
            return df

        # Otherwise, download only the missing tail and append
        try:
            new_start = (pd.Timestamp(last_cached) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            raw_new = yf.download(symbol, start=new_start, end=end_date, progress=False)
            if not raw_new.empty:
                if isinstance(raw_new.columns, pd.MultiIndex):
                    raw_new.columns = raw_new.columns.get_level_values(0)
                new_rows = raw_new[["Open", "High", "Low", "Close", "Volume"]].copy()
                new_rows.index.name = "Date"
                new_rows = new_rows.dropna(how="all")
                df = pd.concat([df, new_rows])
                df = df[~df.index.duplicated(keep="last")]
                df.to_csv(cache_path)
                logger.info(f"Updated cache with {len(new_rows)} new rows → {len(df)} total")
            return df
        except Exception:
            # If incremental update fails, return what we have
            logger.info(f"Loaded cached Nifty data: {len(df)} rows from {cache_path.name}")
            return df

    logger.info(f"Downloading {symbol} data from {start_date} to {end_date} …")
    raw = yf.download(symbol, start=start_date, end=end_date, progress=False)

    if raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")

    # Flatten MultiIndex columns if present (yfinance >= 0.2.31)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "Date"

    # Remove any rows that are all-NaN (weekends that crept in)
    df = df.dropna(how="all")

    if cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
        logger.info(f"Cached {len(df)} rows to {cache_path}")

    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 4 HMM feature columns from an OHLCV DataFrame.

    Features:
        1. log_return       — daily log return of Close
        2. realized_vol     — 20-day annualised realised volatility
        3. volume_ratio     — Volume / 20-day SMA(Volume)
        4. momentum_5d      — Close / Close_5days_ago − 1

    Args:
        df: DataFrame with at least Close and Volume columns (or lowercase).

    Returns:
        DataFrame with the 4 feature columns (NaN rows from rolling windows
        are dropped).
    """
    close_col = "Close" if "Close" in df.columns else "close"
    vol_col = "Volume" if "Volume" in df.columns else "volume"

    close = df[close_col].astype(float)
    volume = df[vol_col].astype(float)

    feat = pd.DataFrame(index=df.index)

    # 1. Daily log return
    feat["log_return"] = np.log(close / close.shift(1))

    # 2. 20-day annualised realised volatility
    feat["realized_vol"] = feat["log_return"].rolling(20).std() * np.sqrt(252)

    # 3. Volume ratio (current / 20-day average)
    vol_sma = volume.rolling(20).mean()
    feat["volume_ratio"] = volume / vol_sma

    # 4. 5-day momentum
    feat["momentum_5d"] = close / close.shift(5) - 1.0

    # Replace inf/-inf (e.g. from zero-volume division) before dropping
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna()
    return feat


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    3-state Gaussian Hidden Markov Model for Nifty market regime detection.

    States are auto-labelled after fitting:
        - **Bull**     – highest mean log-return
        - **Bear**     – lowest mean log-return
        - **Sideways** – middle

    The detector also exposes regime-conditional parameter tables for the
    coach, players, risk manager, and cost model.
    """

    def __init__(
        self,
        n_states: int = 3,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
        model_path: Path | str = _MODEL_PATH,
    ):
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model_path = Path(model_path)

        self._model = None          # GaussianHMM instance
        self._state_map: Dict[int, str] = {}   # HMM-state-int → label
        self._fitted = False
        self._feature_means: Optional[np.ndarray] = None   # for diagnostics
        self._feature_stds: Optional[np.ndarray] = None

        # Try loading a saved model
        self._load()

    # ------------------------------------------------------------------
    # fit()
    # ------------------------------------------------------------------

    def fit(self, ohlcv_df: pd.DataFrame) -> "RegimeDetector":
        """
        Fit the HMM on daily OHLCV data.

        Args:
            ohlcv_df: DataFrame with Date index, Open/High/Low/Close/Volume.

        Returns:
            self (for chaining).
        """
        from hmmlearn.hmm import GaussianHMM

        features_df = compute_features(ohlcv_df)
        if len(features_df) < 60:
            raise ValueError(
                f"Need ≥60 daily observations after rolling windows; got {len(features_df)}"
            )

        X = features_df.values
        self._feature_means = X.mean(axis=0)
        self._feature_stds = X.std(axis=0)

        # Normalise features for numerical stability
        X_norm = (X - self._feature_means) / (self._feature_stds + 1e-10)

        # Fit with fallback covariance types
        fitted = False
        for cov_type in [self.covariance_type, "diag", "spherical"]:
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=cov_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
                tol=0.01,
            )
            try:
                model.fit(X_norm)
                self._model = model
                fitted = True
                print(f"[RegimeDetector] HMM fitted ({cov_type}), "
                      f"{len(features_df)} observations, "
                      f"score={model.score(X_norm):.1f}")
                break
            except Exception as exc:
                logger.warning(f"HMM fit failed with {cov_type}: {exc}")

        if not fitted:
            raise RuntimeError("HMM fitting failed with all covariance types.")

        # Auto-label states
        self._auto_label_states(X_norm, features_df)
        self._fitted = True
        self._save()
        return self

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    def predict(
        self,
        recent_ohlcv: pd.DataFrame,
        lookback: int = 5,
    ) -> Tuple[str, Dict[str, float], int]:
        """
        Predict the current regime from recent OHLCV data.

        Args:
            recent_ohlcv: Recent daily OHLCV (≥25 rows recommended so rolling
                          windows can populate).
            lookback:     Number of recent states to use when computing
                          regime duration.

        Returns:
            (current_regime, regime_probabilities, regime_duration)
            - current_regime:  "Bull", "Bear", or "Sideways"
            - regime_probabilities: {"Bull": 0.82, "Bear": 0.05, "Sideways": 0.13}
            - regime_duration: consecutive days in the current regime
        """
        if not self._fitted or self._model is None:
            return "Sideways", {"Bull": 0.33, "Bear": 0.33, "Sideways": 0.34}, 0

        feat_df = compute_features(recent_ohlcv)
        if feat_df.empty:
            return "Sideways", {"Bull": 0.33, "Bear": 0.33, "Sideways": 0.34}, 0

        X = feat_df.values
        X_norm = (X - self._feature_means) / (self._feature_stds + 1e-10)

        states = self._model.predict(X_norm)
        probs = self._model.predict_proba(X_norm)

        current_state = int(states[-1])
        current_regime = self._state_map.get(current_state, "Sideways")

        # Build probability dict keyed by regime label.
        # Guard against a partial _state_map: accumulate probabilities per
        # label so that even if two HMM ints somehow share a label the
        # output always has exactly one entry per regime that appeared.
        prob_dict: Dict[str, float] = {}
        for i in range(self.n_states):
            label = self._state_map.get(i, "Sideways")
            prob_dict[label] = round(
                prob_dict.get(label, 0.0) + float(probs[-1, i]), 4
            )
        # Ensure all canonical regime labels are present
        for r in REGIME_LABELS:
            prob_dict.setdefault(r, 0.0)

        # Regime duration: count consecutive days in current state from the end
        duration = 0
        for s in reversed(states):
            if int(s) == current_state:
                duration += 1
            else:
                break

        return current_regime, prob_dict, duration

    # ------------------------------------------------------------------
    # Regime-conditional parameter getters
    # ------------------------------------------------------------------

    def get_regime_weights(self, regime: str) -> Dict[str, float]:
        """
        Return recommended player capital-allocation weights.

        Keys are player labels (Momentum, Aggressive, Balanced,
        VolBreakout, Conservative).  Values sum to 1.0.
        """
        return dict(_PLAYER_WEIGHTS.get(regime, _PLAYER_WEIGHTS["Sideways"]))

    def get_indicator_adjustments(self, regime: str) -> Dict[str, float]:
        """
        Return per-indicator weight multipliers for the given regime.

        >1 = upweight, <1 = downweight, missing = 1.0 (no change).
        """
        return dict(_INDICATOR_ADJUSTMENTS.get(regime, _INDICATOR_ADJUSTMENTS["Sideways"]))

    def get_risk_adjustments(self, regime: str) -> Dict[str, Any]:
        """
        Return risk-manager overrides and slippage multiplier for the regime.

        The returned dict is ready to patch into ``RiskLimits`` fields and
        ``TransactionCostModel`` slippage scaling.
        """
        overrides = _RISK_OVERRIDES.get(regime, _RISK_OVERRIDES["Sideways"])
        return overrides.to_dict()

    # ------------------------------------------------------------------
    # State labelling
    # ------------------------------------------------------------------

    def get_state_summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return fitted state statistics (in original feature space).

        Returns:
            {regime_label: {feature_name: mean_value, ...}, ...}
        """
        if not self._fitted or self._model is None:
            return {}

        feature_names = ["log_return", "realized_vol", "volume_ratio", "momentum_5d"]
        summary = {}
        for state_int, label in self._state_map.items():
            means_norm = self._model.means_[state_int]
            means_orig = means_norm * (self._feature_stds + 1e-10) + self._feature_means
            summary[label] = {
                name: round(float(val), 6) for name, val in zip(feature_names, means_orig)
            }
        return summary

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_regimes(
        self,
        ohlcv_df: pd.DataFrame,
        save_path: str | Path | None = None,
        figsize: Tuple[int, int] = (16, 7),
    ) -> None:
        """
        Plot Nifty Close price with colour-coded regime backgrounds.

        Colours:
            - Bull:     green (#c8f7c5)
            - Bear:     red   (#f7c5c5)
            - Sideways: yellow (#f7f0c5)

        Args:
            ohlcv_df:  Full OHLCV DataFrame used for fit/predict.
            save_path: If given, save chart as PNG (otherwise plt.show()).
            figsize:   Figure size.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not self._fitted or self._model is None:
            print("[RegimeDetector] Model not fitted — cannot plot.")
            return

        feat_df = compute_features(ohlcv_df)
        X = feat_df.values
        X_norm = (X - self._feature_means) / (self._feature_stds + 1e-10)
        states = self._model.predict(X_norm)

        close_col = "Close" if "Close" in ohlcv_df.columns else "close"
        price = ohlcv_df.loc[feat_df.index, close_col]

        colours = {
            "Bull":     "#c8f7c5",
            "Bear":     "#f7c5c5",
            "Sideways": "#f7f0c5",
        }

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(price.index, price.values, color="black", linewidth=0.8, label="Nifty 50 Close")

        # Shade backgrounds
        dates = price.index
        for i in range(len(states)):
            label = self._state_map.get(int(states[i]), "Sideways")
            x_start = dates[i]
            # Last day: extend by 1 calendar day so the span has non-zero width
            if i + 1 < len(dates):
                x_end = dates[i + 1]
            else:
                x_end = dates[i] + pd.Timedelta(days=1)
            ax.axvspan(x_start, x_end, alpha=0.35, color=colours.get(label, "#f7f0c5"))

        # Legend patches
        from matplotlib.patches import Patch
        legend_patches = [
            Patch(facecolor=colours[r], edgecolor="grey", label=r, alpha=0.5)
            for r in REGIME_LABELS
        ]
        legend_patches.insert(0, plt.Line2D([0], [0], color="black", linewidth=0.8, label="Nifty 50"))
        ax.legend(handles=legend_patches, loc="upper left", fontsize=9)

        ax.set_title("Nifty 50 — HMM Regime Detection", fontsize=14, fontweight="bold")
        ax.set_ylabel("Price (INR)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        fig.autofmt_xdate()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        if save_path:
            fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
            print(f"[RegimeDetector] Saved plot to {save_path}")
        else:
            plt.show()

        plt.close(fig)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        """Save fitted model + metadata with joblib."""
        payload = {
            "model": self._model,
            "state_map": self._state_map,
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
            "n_states": self.n_states,
        }
        try:
            joblib.dump(payload, self.model_path)
            logger.info(f"Saved HMM model to {self.model_path}")
        except Exception as exc:
            logger.warning(f"Failed to save model: {exc}")

    def _load(self) -> None:
        """Load a previously saved model."""
        if not self.model_path.exists():
            return
        try:
            payload = joblib.load(self.model_path)
            self._model = payload["model"]
            self._state_map = payload["state_map"]
            self._feature_means = payload["feature_means"]
            self._feature_stds = payload["feature_stds"]
            self._fitted = self._model is not None
            if self._fitted:
                logger.info(f"Loaded HMM model from {self.model_path}")
        except Exception as exc:
            logger.warning(f"Failed to load model: {exc}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_label_states(
        self,
        X_norm: np.ndarray,
        feat_df: pd.DataFrame,
    ) -> None:
        """
        Label states by sorting on mean log-return in original space.

        Highest mean return → Bull, lowest → Bear, middle → Sideways.
        """
        # Un-normalise the state means back to original feature space
        means_orig = (
            self._model.means_ * (self._feature_stds + 1e-10) + self._feature_means
        )
        # Column 0 is log_return
        return_means = means_orig[:, 0]
        order = np.argsort(return_means)  # ascending

        bull_state = int(order[-1])
        bear_state = int(order[0])

        # Guard: if highest and lowest resolve to the same state (degenerate
        # case with identical means), force distinct assignments.
        if bull_state == bear_state:
            bull_state = int(order[-1])
            bear_state = int(order[-2]) if len(order) > 1 else int(order[0])

        labels = {
            bull_state: "Bull",
            bear_state: "Bear",
        }
        for i in range(self.n_states):
            if i not in labels:
                labels[i] = "Sideways"

        self._state_map = labels

        # Print summary
        feature_names = ["log_return", "realized_vol", "volume_ratio", "momentum_5d"]
        for state_int, label in sorted(labels.items()):
            vals = ", ".join(
                f"{fn}={means_orig[state_int, j]:.5f}"
                for j, fn in enumerate(feature_names)
            )
            print(f"  State {state_int} → {label:8s}  [{vals}]")


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def _main():
    """
    End-to-end demo:
      1. Download 5 years of Nifty 50 data
      2. Fit the HMM
      3. Predict current regime
      4. Print risk adjustments showing integration points
      5. Plot regime history
      6. Save model
    """
    print("=" * 70)
    print("  Regime Detector — HMM-Based Market Regime Classification")
    print("=" * 70)

    # 1. Load data
    print("\n[1] Loading Nifty 50 daily data (5 years) …")
    df = load_nifty_data(start_date="2020-01-01")
    print(f"    {len(df)} trading days  |  "
          f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}")

    # 2. Fit
    print("\n[2] Fitting 3-state Gaussian HMM …")
    rd = RegimeDetector()
    rd.fit(df)

    # State summary
    print("\n    State statistics (original feature space):")
    for label, stats in rd.get_state_summary().items():
        vol_ann = stats['realized_vol']
        print(f"      {label:8s}  ret={stats['log_return']:+.5f}  "
              f"vol={vol_ann:.3f}  vol_ratio={stats['volume_ratio']:.3f}  "
              f"mom5d={stats['momentum_5d']:+.4f}")

    # 3. Predict current regime
    print("\n[3] Current regime prediction …")
    regime, probs, duration = rd.predict(df.tail(60))
    print(f"    Regime:    {regime}")
    print(f"    Duration:  {duration} days")
    print(f"    Probabilities:")
    for r, p in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(p * 40)
        print(f"      {r:8s}  {p:.1%}  {bar}")

    # 4. Regime-conditional parameters
    print(f"\n[4] Regime-conditional parameters for '{regime}':")

    # Player weights
    weights = rd.get_regime_weights(regime)
    print(f"\n    Player allocation weights:")
    for player, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"      {player:14s}  {w:.0%}")

    # Indicator adjustments (top 5 up, top 5 down)
    adj = rd.get_indicator_adjustments(regime)
    up = sorted([(k, v) for k, v in adj.items() if v > 1.0], key=lambda x: -x[1])[:5]
    down = sorted([(k, v) for k, v in adj.items() if v < 1.0], key=lambda x: x[1])[:5]
    print(f"\n    Indicator adjustments (top changes):")
    print(f"      Upweighted:   {', '.join(f'{k}={v:.1f}x' for k, v in up)}")
    print(f"      Downweighted: {', '.join(f'{k}={v:.1f}x' for k, v in down)}")

    # Risk adjustments
    risk = rd.get_risk_adjustments(regime)
    print(f"\n    Risk parameter overrides (→ PortfolioRiskManager):")
    print(f"      max_single_stock:   {risk['max_single_stock_pct']:.0%}")
    print(f"      max_sector:         {risk['max_sector_pct']:.0%}")
    print(f"      max_gross_exposure: {risk['max_gross_exposure_pct']:.0%}")
    print(f"      trailing_stop (LC): {risk['trailing_stop_largecap_pct']:.1%}")
    print(f"      trailing_stop (MC): {risk['trailing_stop_midcap_pct']:.1%}")
    print(f"      slippage_mult (→ TransactionCostModel): {risk['slippage_multiplier']:.1f}x")

    # 5. Plot
    plot_path = _PROJECT_ROOT / "regime_plot.png"
    print(f"\n[5] Plotting regime chart → {plot_path}")
    rd.plot_regimes(df, save_path=plot_path)

    # 6. Model is already saved by fit(), confirm
    print(f"\n[6] Model saved to {rd.model_path}")
    print(f"    Model file size: {rd.model_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 70)
    print("  Done.  Import and use:")
    print("    from regime_detector import RegimeDetector, load_nifty_data")
    print("=" * 70)


if __name__ == "__main__":
    _main()
