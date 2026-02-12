"""
Backtest Exit Comparison — Baseline (SL/TP/signal-decay) vs SmartExitEngine.

Runs both exit strategies on identical data and shared entry signals,
then produces a side-by-side comparison with MFE/MAE analysis.

Architecture
============
Phase 1 — Shared Signal Generation:
    Walk 60 days x 30 symbols x 5 players once.  Record every entry
    signal crossing and per-day signal intensity values.

Phase 2 — Parallel Exit Replay:
    Replay entry events through two independent simulators:
    - BASELINE:   SL/TP/signal-decay exits (matching continuous_backtest.py)
    - SMART_EXIT: SmartExitEngine 10-signal weighted voting + staged exits

Both modes independently gate entries by their own position/capital
state (max 5 positions, 20% sizing).  MFE/MAE tracked in both.

Usage
=====
    python3 backtest_exit_comparison.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Project root on sys.path ──────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Imports from project modules ──────────────────────────────────────

from trading_evolution.indicators.normalizer import IndicatorNormalizer
from smart_exit_engine import SmartExitEngine, PositionState, ExitDecision

try:
    from mfe_mae_analyzer import MFEMAEAnalyzer
    _HAS_MFE = True
except ImportError:
    _HAS_MFE = False

try:
    from regime_detector import RegimeDetector
    _HAS_REGIME = True
except ImportError:
    try:
        from coach_system.coaches.regime_detector import RegimeDetector
        _HAS_REGIME = True
    except ImportError:
        _HAS_REGIME = False

# ── Inline utilities (originally in continuous_backtest.py) ───────────

FIXED_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS",
    "POWERGRID.NS", "NTPC.NS", "NESTLEIND.NS", "BRITANNIA.NS", "M&M.NS",
    "ONGC.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIPORTS.NS", "TECHM.NS",
]

PLAYERS_CONFIG = {
    "PLAYER_1": {
        "label": "Aggressive",
        "weights": {
            "RSI_7": 0.90, "STOCH_5_3": 0.85, "TSI_13_25": 0.75,
            "CMO_14": 0.70, "WILLR_14": 0.65, "OBV": 0.60,
            "MFI_14": 0.55, "ADX_14": 0.50, "DEMA_20": 0.45,
            "NATR_14": 0.40,
        },
        "entry_threshold": 0.25, "exit_threshold": -0.10, "min_hold_bars": 4,
    },
    "PLAYER_2": {
        "label": "Conservative",
        "weights": {
            "ADX_14": 0.90, "SUPERTREND_7_3": 0.85, "EMA_50": 0.80,
            "AROON_14": 0.70, "CMF_20": 0.65, "RSI_14": 0.60,
            "BBANDS_20_2": 0.55, "OBV": 0.50, "VWMA_20": 0.45,
            "HMA_9": 0.40, "ATR_14": 0.35,
        },
        "entry_threshold": 0.30, "exit_threshold": -0.15, "min_hold_bars": 5,
    },
    "PLAYER_3": {
        "label": "Balanced",
        "weights": {
            "MACD_12_26_9": 0.85, "RSI_14": 0.80, "ADX_14": 0.75,
            "BBANDS_20_2": 0.70, "OBV": 0.65, "MFI_14": 0.60,
            "EMA_20": 0.55, "STOCH_14_3": 0.50, "CMF_20": 0.45,
            "NATR_14": 0.40,
        },
        "entry_threshold": 0.25, "exit_threshold": -0.12, "min_hold_bars": 4,
    },
    "PLAYER_4": {
        "label": "VolBreakout",
        "weights": {
            "NATR_14": 0.90, "BBANDS_20_2": 0.85, "ATR_14": 0.80,
            "ADX_14": 0.70, "OBV": 0.60, "CMO_14": 0.55,
            "RSI_14": 0.50, "MACD_12_26_9": 0.45, "MFI_14": 0.40,
        },
        "entry_threshold": 0.30, "exit_threshold": -0.15, "min_hold_bars": 3,
    },
    "PLAYER_5": {
        "label": "Momentum",
        "weights": {
            "RSI_7": 0.90, "STOCH_5_3": 0.85, "MACD_12_26_9": 0.80,
            "CMO_14": 0.70, "WILLR_14": 0.65, "TSI_13_25": 0.60,
            "ADX_14": 0.55, "OBV": 0.50, "DEMA_20": 0.45,
        },
        "entry_threshold": 0.20, "exit_threshold": -0.08, "min_hold_bars": 3,
    },
}

_LOCAL_CACHE = None


def _get_local_cache():
    """Get or create local cache singleton."""
    global _LOCAL_CACHE
    if _LOCAL_CACHE is None:
        try:
            from data.local_cache import get_cache
            _LOCAL_CACHE = get_cache()
        except Exception:
            try:
                from data_cache.local_cache import get_cache
                _LOCAL_CACHE = get_cache()
            except Exception:
                _LOCAL_CACHE = False
    return _LOCAL_CACHE if _LOCAL_CACHE else None


def _fetch_all_data(symbols_tuple, interval: str, days: int) -> Dict:
    """Fetch OHLCV data for all symbols from local cache."""
    symbols = list(symbols_tuple)
    data = {}
    cache = _get_local_cache()

    for sym in symbols:
        df = None
        # Try local cache
        if cache:
            try:
                df = cache.get_data(sym, interval=interval)
            except Exception:
                pass

        # Try pickle cache
        if df is None or (hasattr(df, 'empty') and df.empty):
            pkl_path = _PROJECT_ROOT / "data" / "cache" / "5m" / f"{sym}.pkl"
            if pkl_path.exists():
                try:
                    df = pd.read_pickle(pkl_path)
                except Exception:
                    pass

        # Fallback to yfinance
        if df is None or (hasattr(df, 'empty') and df.empty):
            try:
                import yfinance as yf
                from datetime import timedelta
                end = datetime.now()
                start = end - timedelta(days=days + 10)
                ticker = yf.Ticker(sym)
                df = ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval=interval,
                )
            except Exception:
                pass

        if df is not None and not df.empty:
            # Normalize column names to lowercase
            col_map = {}
            for c in df.columns:
                cl = c.lower()
                if cl in ['open', 'high', 'low', 'close', 'volume']:
                    col_map[c] = cl
            if col_map:
                df = df.rename(columns=col_map)
            data[sym] = df

    print(f"  [Data] Loaded {len(data)}/{len(symbols)} symbols")
    return data


def _load_cached_indicators(symbols: List[str]) -> Dict:
    """Load pre-computed indicators from cache."""
    indicator_data = {}
    cache_dir = _PROJECT_ROOT / "data" / "cache" / "indicators"

    for sym in symbols:
        cache_path = cache_dir / f"{sym}.pkl"
        if cache_path.exists():
            try:
                indicator_data[sym] = pd.read_pickle(cache_path)
            except Exception:
                pass

    print(f"  [Indicators] Loaded {len(indicator_data)}/{len(symbols)} from cache")
    return indicator_data


def raw_weighted_average(normalized_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Compute weighted average of normalized indicators."""
    weighted_sum = pd.Series(0.0, index=normalized_df.index)
    total_weight = 0.0
    for name, weight in weights.items():
        if name not in normalized_df.columns or weight == 0:
            continue
        weighted_sum += normalized_df[name].fillna(0) * weight
        total_weight += abs(weight)
    if total_weight == 0:
        return pd.Series(0.0, index=normalized_df.index)
    return (weighted_sum / total_weight).clip(-1.0, 1.0)


def load_evolved_configs() -> Optional[Dict]:
    """Load evolved player configs from disk."""
    cfg_path = _PROJECT_ROOT / "evolved_player_configs.json"
    try:
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            # Handle both formats: {pid: config} or {pid: {runs, ...config}}
            configs = {}
            for pid, val in raw.items():
                if pid.startswith("PLAYER_") or pid.startswith("total_"):
                    if isinstance(val, dict) and "weights" in val:
                        configs[pid] = val
                    elif isinstance(val, dict):
                        # Nested format: extract config fields
                        inner = {}
                        for k, v in val.items():
                            if k in ("weights", "entry_threshold", "exit_threshold",
                                     "min_hold_bars", "label"):
                                inner[k] = v
                        if "weights" in inner:
                            configs[pid] = inner
            if configs:
                return configs
    except Exception:
        pass
    return None

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════

COMPARISON_DAYS = 60
BARS_PER_DAY = 26
INITIAL_CAPITAL = 100_000  # per player
MAX_POSITIONS = 5
POSITION_SIZE_PCT = 0.20
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.0

REGIME_MAP = {
    "bull_low_vol": "bull",
    "bear_high_vol": "bear",
    "sideways": "sideways",
    # Fallbacks from simple detector
    "volatile_bullish": "bull",
    "volatile_bearish": "bear",
    "volatile": "bear",
    "ranging": "sideways",
    "trending_up": "bull",
    "trending_down": "bear",
    "unknown": "sideways",
}

RESULTS_PATH = Path("data/exit_comparison_results.json")


# ═══════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class EntryEvent:
    """A candidate entry signal recorded during Phase 1."""
    day: int
    symbol: str
    player_id: str
    direction: str        # "LONG" or "SHORT"
    entry_price: float
    signal_intensity: float
    atr: float
    regime: str


@dataclass
class TrackedPosition:
    """Position with MFE/MAE tracking for both modes."""
    symbol: str
    player_id: str
    direction: str
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_day: int
    entry_signal: float
    entry_atr: float
    entry_regime: str
    bars_held: int = 0
    peak_price: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    remaining_fraction: float = 1.0
    staged_exits_done: int = 0
    original_quantity: int = 0
    # For SmartExit mode: the PositionState wrapper
    smart_state: Optional[PositionState] = None

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price
        if self.original_quantity == 0:
            self.original_quantity = self.quantity

    def update_mfe_mae(self, current_price: float, current_high: float,
                       current_low: float):
        """Update MFE/MAE tracking each bar."""
        if self.direction == "LONG":
            self.peak_price = max(self.peak_price, current_high)
            pnl_pct = (current_price - self.entry_price) / self.entry_price
        else:
            self.peak_price = min(self.peak_price, current_low)
            pnl_pct = (self.entry_price - current_price) / self.entry_price
        self.mfe = max(self.mfe, pnl_pct)
        if pnl_pct < 0:
            self.mae = max(self.mae, abs(pnl_pct))
        self.bars_held += 1


@dataclass
class SimPlayerState:
    """Per-player state during simulation."""
    player_id: str
    config: Dict[str, Any]
    equity: float
    positions: Dict[str, TrackedPosition] = field(default_factory=dict)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    daily_pnl: List[float] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    partial_exit_count: int = 0


# ═══════════════════════════════════════════════════════════════════════
# Phase 0: Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_data() -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Load OHLCV data and indicator data for all symbols."""
    data = _fetch_all_data(
        tuple(FIXED_SYMBOLS), "5m", COMPARISON_DAYS + 20,
    )
    indicator_data = _load_cached_indicators(FIXED_SYMBOLS)
    return data, indicator_data


def compute_regime_sequence(
    data: Dict[str, pd.DataFrame],
    total_days: int,
) -> Tuple[List[str], Optional[Any], Dict[str, int]]:
    """Pre-compute regime for each day using RegimeDetector.

    Returns:
        (regime_sequence, hmm_model, regime_state_map)
        - regime_sequence: list of regime labels per day
        - hmm_model: fitted HMM model (or None)
        - regime_state_map: {"Bull": 0, "Bear": 1, ...} (or empty)
    """
    if not _HAS_REGIME:
        return ["sideways"] * total_days, None, {}

    try:
        # Use the RegimeDetector's actual API: fit on Nifty OHLCV, predict
        from regime_detector import load_nifty_data
        detector = RegimeDetector()

        # If no pre-trained model loaded, fit on Nifty data
        if not detector._fitted:
            nifty_df = load_nifty_data()
            if nifty_df is not None and len(nifty_df) > 60:
                detector.fit(nifty_df)
            else:
                print("  [RegimeDetector] Nifty data too short, using 'sideways' fallback")
                return ["sideways"] * total_days, None, {}

        # Extract HMM model and state map for SmartExitEngine
        hmm_model = detector._model if detector._fitted else None
        regime_state_map: Dict[str, int] = {}
        if detector._fitted and hasattr(detector, '_state_map'):
            regime_state_map = {
                label: state_int
                for state_int, label in detector._state_map.items()
            }

        # Predict regime for each day using Nifty data
        nifty_df = load_nifty_data()
        if nifty_df is not None and len(nifty_df) > 30:
            regime_label, probs, duration = detector.predict(nifty_df)
            # Map title-case to lowercase for consistency
            regime_lower = REGIME_MAP.get(
                regime_label.lower().replace(" ", "_"),
                regime_label.lower()
            )
            # Use a single regime for all days (HMM predicts current state)
            regimes = [regime_lower] * total_days
            print(f"  [RegimeDetector] Current regime: {regime_label} "
                  f"(Bull={probs.get('Bull', 0):.0%}, Bear={probs.get('Bear', 0):.0%}, "
                  f"Sideways={probs.get('Sideways', 0):.0%})")
        else:
            regimes = ["sideways"] * total_days

        return regimes, hmm_model, regime_state_map

    except Exception as e:
        print(f"  [RegimeDetector] Failed: {e} — using 'sideways' fallback")
        return ["sideways"] * total_days, None, {}


# ═══════════════════════════════════════════════════════════════════════
# Phase 1: Entry Signal Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_entry_signals(
    data: Dict[str, pd.DataFrame],
    indicator_data: Dict[str, pd.DataFrame],
    player_configs: Dict[str, Dict],
    regime_sequence: List[str],
    total_days: int,
) -> Tuple[List[EntryEvent], Dict[str, Dict[str, Dict[int, float]]]]:
    """Walk all days and record entry signals + per-day signal intensities.

    Returns
    -------
    entry_events : list of EntryEvent
        Every signal crossing, chronologically ordered.
    signal_map : {player_id: {symbol: {day: last_si}}}
        Per-day signal intensity for exit evaluation.
    """
    normalizer = IndicatorNormalizer()
    entry_events: List[EntryEvent] = []
    signal_map: Dict[str, Dict[str, Dict[int, float]]] = {}

    for pid, config in player_configs.items():
        signal_map[pid] = {}

    for day in range(total_days):
        start_bar = day * BARS_PER_DAY
        end_bar = start_bar + BARS_PER_DAY
        regime = regime_sequence[day]

        for pid, config in player_configs.items():
            weights = config["weights"]
            entry_thresh = config.get("entry_threshold", 0.25)

            for sym, df in data.items():
                if sym not in indicator_data:
                    continue

                raw = indicator_data[sym]
                close_col = "close" if "close" in df.columns else "Close"

                if end_bar > len(df) or end_bar > len(raw):
                    continue

                day_df = df.iloc[start_bar:end_bar]
                day_raw = raw.iloc[start_bar:end_bar]

                if day_df.empty or day_raw.empty:
                    continue

                active = [i for i in weights.keys() if i in day_raw.columns]
                if not active:
                    continue

                try:
                    norm = normalizer.normalize_all(
                        day_raw[active],
                        price_series=day_df[close_col],
                    )
                    if norm.empty:
                        continue

                    si = raw_weighted_average(
                        norm, {k: weights[k] for k in active},
                    )
                    if si.empty:
                        continue

                    last_si = float(si.iloc[-1])
                    last_price = float(day_df[close_col].iloc[-1])
                    atr = (
                        float(day_raw["ATR_14"].iloc[-1])
                        if "ATR_14" in day_raw.columns
                        else last_price * 0.02
                    )

                    # Store signal for exit evaluation
                    if sym not in signal_map[pid]:
                        signal_map[pid][sym] = {}
                    signal_map[pid][sym][day] = last_si

                    # Record entry if threshold crossed
                    if last_si > entry_thresh:
                        direction = "LONG"
                    elif last_si < -entry_thresh:
                        direction = "SHORT"
                    else:
                        continue

                    entry_events.append(EntryEvent(
                        day=day,
                        symbol=sym,
                        player_id=pid,
                        direction=direction,
                        entry_price=last_price,
                        signal_intensity=last_si,
                        atr=atr,
                        regime=regime,
                    ))

                except Exception:
                    continue

    return entry_events, signal_map


# ═══════════════════════════════════════════════════════════════════════
# Phase 2: Simulation Engine
# ═══════════════════════════════════════════════════════════════════════

def _close_position(
    state: SimPlayerState,
    sym: str,
    exit_price: float,
    exit_reason: str,
) -> float:
    """Close a position fully, return PnL."""
    pos = state.positions[sym]
    if pos.direction == "LONG":
        pnl = (exit_price - pos.entry_price) * pos.quantity
    else:
        pnl = (pos.entry_price - exit_price) * pos.quantity

    return_pct = pnl / (pos.entry_price * pos.quantity) if pos.entry_price > 0 else 0.0
    state.equity += pnl

    state.trades.append({
        "symbol": sym,
        "player_id": pos.player_id,
        "direction": pos.direction,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "quantity": pos.quantity,
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 6),
        "bars_held": pos.bars_held,
        "exit_reason": exit_reason,
        "entry_regime": pos.entry_regime,
        "mfe": round(pos.mfe, 6),
        "mae": round(pos.mae, 6),
        "peak_price": round(pos.peak_price, 2),
        "entry_signal": pos.entry_signal,
        "entry_volatility": pos.entry_atr / pos.entry_price * math.sqrt(252) if pos.entry_price > 0 else 0.0,
        "partial": False,
        "staged_exits_done": pos.staged_exits_done,
    })

    del state.positions[sym]
    return pnl


def _partial_close(
    state: SimPlayerState,
    sym: str,
    exit_price: float,
    exit_fraction: float,
    exit_reason: str,
) -> float:
    """Partial close: exit a fraction, keep position open."""
    pos = state.positions[sym]
    shares_to_exit = max(1, int(pos.quantity * exit_fraction))

    if shares_to_exit >= pos.quantity:
        return _close_position(state, sym, exit_price, exit_reason)

    if pos.direction == "LONG":
        pnl = (exit_price - pos.entry_price) * shares_to_exit
    else:
        pnl = (pos.entry_price - exit_price) * shares_to_exit

    return_pct = pnl / (pos.entry_price * shares_to_exit) if pos.entry_price > 0 else 0.0
    state.equity += pnl

    state.trades.append({
        "symbol": sym,
        "player_id": pos.player_id,
        "direction": pos.direction,
        "entry_price": pos.entry_price,
        "exit_price": exit_price,
        "quantity": shares_to_exit,
        "pnl": round(pnl, 2),
        "return_pct": round(return_pct, 6),
        "bars_held": pos.bars_held,
        "exit_reason": exit_reason,
        "entry_regime": pos.entry_regime,
        "mfe": round(pos.mfe, 6),
        "mae": round(pos.mae, 6),
        "peak_price": round(pos.peak_price, 2),
        "entry_signal": pos.entry_signal,
        "entry_volatility": pos.entry_atr / pos.entry_price * math.sqrt(252) if pos.entry_price > 0 else 0.0,
        "partial": True,
        "staged_exits_done": pos.staged_exits_done,
    })

    pos.quantity -= shares_to_exit
    pos.remaining_fraction = pos.quantity / pos.original_quantity
    pos.staged_exits_done += 1
    state.partial_exit_count += 1
    return pnl


def simulate(
    mode: str,
    entry_events: List[EntryEvent],
    data: Dict[str, pd.DataFrame],
    indicator_data: Dict[str, pd.DataFrame],
    signal_map: Dict[str, Dict[str, Dict[int, float]]],
    player_configs: Dict[str, Dict],
    regime_sequence: List[str],
    total_days: int,
    exit_threshold: float = 0.55,
    hmm_model: Optional[Any] = None,
    regime_state_map: Optional[Dict[str, int]] = None,
    regime_config_override: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, SimPlayerState]:
    """Run simulation in either 'baseline' or 'smart_exit' mode."""

    # Init players
    players: Dict[str, SimPlayerState] = {}
    for pid, config in player_configs.items():
        players[pid] = SimPlayerState(
            player_id=pid,
            config=deepcopy(config),
            equity=INITIAL_CAPITAL,
        )
        players[pid].equity_curve.append(INITIAL_CAPITAL)

    # SmartExitEngine (only for smart_exit mode)
    engine: Optional[SmartExitEngine] = None
    if mode == "smart_exit":
        engine = SmartExitEngine(
            hmm_model=hmm_model,
            regime_state_map=regime_state_map or {},
            regime_config_override=regime_config_override,
            exit_threshold_override=exit_threshold,
        )

    # Index entry events by (day, player_id)
    entry_index: Dict[Tuple[int, str], List[EntryEvent]] = {}
    for ev in entry_events:
        key = (ev.day, ev.player_id)
        entry_index.setdefault(key, []).append(ev)

    # ── Day-by-day simulation ─────────────────────────────────────
    for day in range(total_days):
        start_bar = day * BARS_PER_DAY
        end_bar = start_bar + BARS_PER_DAY
        regime = regime_sequence[day]

        for pid, state in players.items():
            config = state.config
            exit_thresh_cfg = config.get("exit_threshold", -0.10)
            min_hold = config.get("min_hold_bars", 4)
            day_pnl = 0.0

            # ── EXITS: evaluate all open positions ────────────
            for sym in list(state.positions.keys()):
                pos = state.positions[sym]

                if sym not in data or sym not in indicator_data:
                    continue

                df = data[sym]
                raw = indicator_data[sym]
                close_col = "close" if "close" in df.columns else "Close"
                high_col = "high" if "high" in df.columns else "High"
                low_col = "low" if "low" in df.columns else "Low"
                vol_col = "volume" if "volume" in df.columns else "Volume"

                if end_bar > len(df):
                    continue

                # Get day-level signal intensity
                day_si = signal_map.get(pid, {}).get(sym, {}).get(day, 0.0)

                if mode == "baseline":
                    # ── Baseline: end-of-day check only ───────
                    last_bar = end_bar - 1
                    if last_bar >= len(df):
                        continue
                    last_price = float(df[close_col].iloc[last_bar])
                    last_high = float(df[high_col].iloc[last_bar])
                    last_low = float(df[low_col].iloc[last_bar])

                    # Update MFE/MAE
                    pos.update_mfe_mae(last_price, last_high, last_low)

                    exit_signal = False
                    exit_reason = ""

                    if pos.direction == "LONG":
                        if last_price <= pos.stop_loss:
                            exit_signal, exit_reason = True, "stop_loss"
                        elif last_price >= pos.take_profit:
                            exit_signal, exit_reason = True, "take_profit"
                        elif day_si < exit_thresh_cfg and pos.bars_held >= min_hold:
                            exit_signal, exit_reason = True, "signal_exit"
                    else:
                        if last_price >= pos.stop_loss:
                            exit_signal, exit_reason = True, "stop_loss"
                        elif last_price <= pos.take_profit:
                            exit_signal, exit_reason = True, "take_profit"
                        elif day_si > -exit_thresh_cfg and pos.bars_held >= min_hold:
                            exit_signal, exit_reason = True, "signal_exit"

                    if exit_signal:
                        pnl = _close_position(state, sym, last_price, exit_reason)
                        day_pnl += pnl

                elif mode == "smart_exit" and engine is not None:
                    # ── SmartExit: bar-by-bar evaluation ──────
                    if pos.smart_state is None:
                        continue

                    exited = False
                    for bar_idx in range(start_bar, min(end_bar, len(df))):
                        if sym not in state.positions:
                            break  # already exited

                        if bar_idx >= len(raw):
                            break

                        cur_price = float(df[close_col].iloc[bar_idx])
                        cur_low = float(df[low_col].iloc[bar_idx])
                        cur_high = float(df[high_col].iloc[bar_idx])
                        cur_vol = float(df[vol_col].iloc[bar_idx]) if vol_col in df.columns else 0.0

                        atr = (
                            float(raw["ATR_14"].iloc[bar_idx])
                            if "ATR_14" in raw.columns and bar_idx < len(raw)
                            else cur_price * 0.02
                        )

                        # Vol forecast: NATR / 100 as proxy
                        natr_val = 0.20
                        if "NATR_14" in raw.columns and bar_idx < len(raw):
                            natr_val = float(raw["NATR_14"].iloc[bar_idx]) / 100.0
                            if natr_val <= 0:
                                natr_val = 0.20

                        # Update our MFE/MAE tracker (parallel to engine's)
                        pos.update_mfe_mae(cur_price, cur_high, cur_low)

                        try:
                            decision = engine.evaluate(
                                position=pos.smart_state,
                                current_price=cur_price,
                                current_low=cur_low,
                                current_volume=cur_vol,
                                atr=atr,
                                current_signal=day_si,
                                current_agreement=0.5,
                                current_vol_forecast=natr_val,
                                regime=regime,
                                current_high=cur_high,
                            )
                        except Exception:
                            continue

                        if decision.should_exit:
                            exit_reason = decision.reason.value if hasattr(decision.reason, 'value') else str(decision.reason)
                            exit_frac = decision.exit_fraction

                            if exit_frac < 1.0 and pos.quantity > 1:
                                pnl = _partial_close(
                                    state, sym, cur_price,
                                    exit_frac, exit_reason,
                                )
                                day_pnl += pnl
                            else:
                                pnl = _close_position(
                                    state, sym, cur_price, exit_reason,
                                )
                                day_pnl += pnl
                                break  # fully exited

            # ── ENTRIES: accept entry events for this day ──────
            day_entries = entry_index.get((day, pid), [])
            for ev in day_entries:
                if ev.symbol in state.positions:
                    continue
                if len(state.positions) >= MAX_POSITIONS:
                    continue

                max_pos_value = state.equity * POSITION_SIZE_PCT
                qty = max(1, int(max_pos_value / ev.entry_price))
                cost = qty * ev.entry_price

                if cost > state.equity * 0.25:
                    continue

                stop_dist = ev.atr * SL_ATR_MULT if ev.atr > 0 else ev.entry_price * 0.02
                if ev.direction == "LONG":
                    sl = ev.entry_price - stop_dist
                    tp = ev.entry_price + ev.atr * TP_ATR_MULT
                else:
                    sl = ev.entry_price + stop_dist
                    tp = ev.entry_price - ev.atr * TP_ATR_MULT

                pos = TrackedPosition(
                    symbol=ev.symbol,
                    player_id=pid,
                    direction=ev.direction,
                    entry_price=ev.entry_price,
                    quantity=qty,
                    stop_loss=sl,
                    take_profit=tp,
                    entry_day=day,
                    entry_signal=ev.signal_intensity,
                    entry_atr=ev.atr,
                    entry_regime=ev.regime,
                )

                if mode == "smart_exit":
                    pos.smart_state = PositionState(
                        symbol=ev.symbol,
                        direction=ev.direction,
                        entry_price=ev.entry_price,
                        entry_date=str(day),
                        entry_signal=ev.signal_intensity,
                        entry_agreement=0.5,
                        entry_volatility=(
                            ev.atr / ev.entry_price * math.sqrt(252)
                            if ev.entry_price > 0 else 0.20
                        ),
                        entry_regime=ev.regime,
                    )

                state.positions[ev.symbol] = pos

            state.daily_pnl.append(day_pnl)
            state.equity_curve.append(state.equity)

    return players


# ═══════════════════════════════════════════════════════════════════════
# Metrics Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(
    players: Dict[str, SimPlayerState],
    total_days: int,
    mode_name: str,
) -> Dict[str, Any]:
    """Compute aggregate metrics for a simulation mode."""

    all_trades: List[Dict] = []
    total_partial = 0
    team_equity_curves: List[List[float]] = []

    for pid, state in players.items():
        all_trades.extend(state.trades)
        total_partial += state.partial_exit_count
        team_equity_curves.append(state.equity_curve)

    n_trade_events = len(all_trades)

    # ── Consolidate partial exits into logical trades ─────────────
    # Group partials + final close by (symbol, player_id, entry_price, direction)
    # to count each position as 1 logical trade with aggregated PnL.
    position_groups: Dict[tuple, Dict] = {}
    for t in all_trades:
        key = (t["symbol"], t["player_id"], t["entry_price"], t["direction"])
        if key not in position_groups:
            position_groups[key] = {
                "total_pnl": 0.0,
                "total_quantity": 0,
                "bars_held": 0,
                "mfe": t.get("mfe", 0.0),
                "mae": t.get("mae", 0.0),
            }
        g = position_groups[key]
        g["total_pnl"] += t["pnl"]
        g["total_quantity"] += t.get("quantity", 0)
        # bars_held: take the max (final close has the full duration)
        g["bars_held"] = max(g["bars_held"], t.get("bars_held", 0))
        # MFE/MAE: take the extremes across all partial records
        g["mfe"] = max(g["mfe"], t.get("mfe", 0.0))
        g["mae"] = min(g["mae"], t.get("mae", 0.0))

    logical_trades = list(position_groups.values())
    n_trades = len(logical_trades)

    # ── Returns ───────────────────────────────────────────────────
    total_initial = INITIAL_CAPITAL * len(players)
    total_final = sum(s.equity for s in players.values())
    total_return = (total_final - total_initial) / total_initial

    # Annualized (assume 252 trading days/year)
    ann_return = total_return * (252 / total_days) if total_days > 0 else 0.0

    # ── Daily returns for Sharpe ──────────────────────────────────
    # Combine all players' daily PnL into a team daily return series
    n_days = min(len(s.daily_pnl) for s in players.values()) if players else 0
    daily_returns = []
    for d in range(n_days):
        team_pnl = sum(s.daily_pnl[d] for s in players.values())
        daily_returns.append(team_pnl / total_initial)

    daily_arr = np.array(daily_returns) if daily_returns else np.array([0.0])
    # Risk-free rate: Indian 10Y ≈ 6.5% annual → daily
    rf_daily = 0.065 / 252
    excess_returns = daily_arr - rf_daily
    sharpe = (
        float(np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252))
        if np.std(excess_returns) > 0 else 0.0
    )

    # ── Max drawdown (team equity curve) ──────────────────────────
    team_eq = []
    for d in range(n_days + 1):
        team_eq.append(sum(
            s.equity_curve[d] if d < len(s.equity_curve) else s.equity
            for s in players.values()
        ))
    eq_series = pd.Series(team_eq) if team_eq else pd.Series([total_initial])
    peak = eq_series.expanding().max()
    dd = (eq_series - peak) / peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    # ── Win rate / profit factor (using logical trades) ───────────
    logical_winners = [t for t in logical_trades if t["total_pnl"] > 0]
    logical_losers = [t for t in logical_trades if t["total_pnl"] <= 0]
    win_rate = len(logical_winners) / n_trades if n_trades > 0 else 0.0

    gross_profit = sum(t["total_pnl"] for t in logical_winners) if logical_winners else 0.0
    gross_loss = abs(sum(t["total_pnl"] for t in logical_losers)) if logical_losers else 0.0
    if gross_loss > 0:
        profit_factor = min(gross_profit / gross_loss, 999.0)
    else:
        profit_factor = 999.0 if gross_profit > 0 else 0.0

    avg_win = gross_profit / len(logical_winners) if logical_winners else 0.0
    avg_loss = gross_loss / len(logical_losers) if logical_losers else 0.0

    # ── Avg holding period (using logical trades) ─────────────────
    avg_hold = (
        sum(t["bars_held"] for t in logical_trades) / n_trades
        if n_trades > 0 else 0.0
    )

    # ── Exit reason distribution ──────────────────────────────────
    exit_reasons: Dict[str, int] = {}
    for t in all_trades:
        reason = t.get("exit_reason", "unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    # ── MFE/MAE analysis ──────────────────────────────────────────
    avg_capture = 0.0
    avg_edge = 0.0
    mfe_report = {}
    if _HAS_MFE and n_trades >= 5:
        try:
            analyzer = MFEMAEAnalyzer(min_trades_for_report=5)
            mfe_report = analyzer.generate_tuning_report(all_trades)
            if mfe_report.get("status") == "ok":
                avg_capture = mfe_report.get("avg_capture_ratio", 0.0)
                avg_edge = mfe_report.get("avg_edge_ratio", 0.0)
        except Exception:
            pass

    return {
        "mode": mode_name,
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_hold_bars": round(avg_hold, 1),
        "trade_count": n_trades,
        "trade_event_count": n_trade_events,
        "partial_exit_count": total_partial,
        "avg_capture_ratio": round(avg_capture, 3),
        "avg_edge_ratio": round(avg_edge, 3),
        "exit_reason_dist": exit_reasons,
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "all_trades": all_trades,
        "mfe_report": mfe_report,
        "equity_curve": team_eq,
    }


# ═══════════════════════════════════════════════════════════════════════
# Output Formatters
# ═══════════════════════════════════════════════════════════════════════

def _fmt_pct(val: float) -> str:
    """Format percentage with sign."""
    return f"{val:+.2f}%" if val != 0 else "0.00%"


def _fmt_delta(baseline_val: float, smart_val: float, is_pct: bool = False) -> str:
    """Format the delta between two values."""
    delta = smart_val - baseline_val
    if is_pct:
        return f"{delta:+.2f}%"
    return f"{delta:+.2f}"


def print_comparison_table(
    baseline: Dict, smart: Dict,
    variants: Optional[List[Dict]] = None,
) -> None:
    """Print rich ASCII comparison table."""
    print()
    header = f"  EXIT STRATEGY COMPARISON \u2014 {COMPARISON_DAYS}-Day Backtest"
    print(f"\u2550" * 70)
    print(header)
    print(f"\u2550" * 70)
    print()

    rows = [
        ("Total Return",      "total_return_pct",   True),
        ("Annualized Return",  "annualized_return_pct", True),
        ("Sharpe Ratio",       "sharpe",             False),
        ("Max Drawdown",       "max_drawdown_pct",   True),
        ("Win Rate",           "win_rate",           True),
        ("Profit Factor",      "profit_factor",      False),
        ("Avg Hold (bars)",    "avg_hold_bars",      False),
        ("Trade Count",        "trade_count",        False),
        ("Partial Exits",      "partial_exit_count", False),
        ("Capture Ratio",      "avg_capture_ratio",  False),
        ("Edge Ratio",         "avg_edge_ratio",     False),
        ("Avg Winner",         "avg_win",            False),
        ("Avg Loser",          "avg_loss",           False),
    ]

    # Header row
    print(f"  {'Metric':<22} {'Baseline':>12} {'SmartExit':>12} {'Delta':>10}")
    print(f"  {'─' * 22} {'─' * 12} {'─' * 12} {'─' * 10}")

    for label, key, is_pct in rows:
        bv = baseline.get(key, 0)
        sv = smart.get(key, 0)
        delta = sv - bv

        if is_pct:
            b_str = f"{bv:+.2f}%"
            s_str = f"{sv:+.2f}%"
            d_str = f"{delta:+.2f}%"
        elif isinstance(bv, int):
            b_str = f"{bv}"
            s_str = f"{sv}"
            d_str = f"{delta:+d}"
        else:
            b_str = f"{bv:.2f}"
            s_str = f"{sv:.2f}"
            d_str = f"{delta:+.2f}"

        print(f"  {label:<22} {b_str:>12} {s_str:>12} {d_str:>10}")

    print(f"  {'─' * 58}")

    # Threshold variants
    if variants:
        print()
        print("  THRESHOLD SENSITIVITY:")
        print(f"  {'Threshold':<14} {'Sharpe':>8} {'Return':>10} {'MaxDD':>10} {'Capture':>10}")
        print(f"  {'─' * 54}")
        for v in [smart] + variants:
            th = v["mode"].split("_")[-1] if "_" in v["mode"] else "0.55"
            print(
                f"  {th:<14} {v['sharpe']:>8.2f} "
                f"{v['total_return_pct']:>+9.2f}% "
                f"{v['max_drawdown_pct']:>+9.2f}% "
                f"{v['avg_capture_ratio']:>10.3f}"
            )
        print(f"  {'─' * 54}")

    print()


def print_exit_reason_chart(baseline: Dict, smart: Dict) -> None:
    """Print ASCII horizontal bar chart of exit reason distribution."""
    for mode_name, metrics in [("BASELINE", baseline), ("SMART_EXIT", smart)]:
        dist = metrics.get("exit_reason_dist", {})
        total = sum(dist.values()) or 1
        print(f"  {mode_name} Exit Reasons:")

        # Sort by count descending
        sorted_reasons = sorted(dist.items(), key=lambda x: -x[1])
        max_count = max(dist.values()) if dist else 1

        for reason, count in sorted_reasons:
            pct = count / total * 100
            bar_len = int(count / max_count * 30)
            bar = "\u2588" * bar_len + "\u2591" * (30 - bar_len)
            print(f"    {reason:<22} {bar} {count:>4} ({pct:>5.1f}%)")

        print()


def print_mfe_scatter_analysis(baseline: Dict, smart: Dict) -> None:
    """Print MFE capture analysis by bucket."""
    print("  MFE CAPTURE ANALYSIS:")
    print(f"  {'MFE Bucket':<14} {'Baseline':>18} {'SmartExit':>18}")
    print(f"  {'─' * 52}")

    buckets = [
        ("0-1%",  0.0,  0.01),
        ("1-2%",  0.01, 0.02),
        ("2-5%",  0.02, 0.05),
        ("5%+",   0.05, 1.0),
    ]

    for label, lo, hi in buckets:
        for mode_name, metrics in [("baseline", baseline), ("smart", smart)]:
            trades = metrics.get("all_trades", [])
            bucket_trades = [
                t for t in trades
                if lo <= abs(t.get("mfe", 0)) < hi
            ]
            if bucket_trades:
                # Compute capture for this bucket
                captures = []
                for t in bucket_trades:
                    mfe = abs(t.get("mfe", 0))
                    if mfe > 0.001:
                        capture = t.get("return_pct", 0) / mfe
                        captures.append(max(-1.0, min(1.5, capture)))
                avg_cap = np.mean(captures) if captures else 0.0
                n = len(bucket_trades)
                if mode_name == "baseline":
                    b_str = f"{avg_cap:.2f} (n={n})"
                else:
                    s_str = f"{avg_cap:.2f} (n={n})"
            else:
                if mode_name == "baseline":
                    b_str = "  --"
                else:
                    s_str = "  --"

        print(f"  {label:<14} {b_str:>18} {s_str:>18}")

    print(f"  {'─' * 52}")

    # Summary: % of trades capturing >50% of MFE
    for mode_name, metrics in [("Baseline", baseline), ("SmartExit", smart)]:
        trades = metrics.get("all_trades", [])
        with_mfe = [t for t in trades if abs(t.get("mfe", 0)) > 0.001]
        good_captures = 0
        for t in with_mfe:
            mfe = abs(t.get("mfe", 0))
            capture = t.get("return_pct", 0) / mfe if mfe > 0.001 else 0
            if capture > 0.5:
                good_captures += 1
        pct = good_captures / len(with_mfe) * 100 if with_mfe else 0
        print(f"  {mode_name}: {good_captures}/{len(with_mfe)} trades captured >50% of MFE ({pct:.1f}%)")

    print()


def save_results(
    baseline: Dict, smart: Dict,
    variants: Optional[List[Dict]] = None,
    regime_sequence: Optional[List[str]] = None,
) -> None:
    """Save results to JSON."""
    # Strip non-serializable data
    def _clean(metrics: Dict) -> Dict:
        clean = {k: v for k, v in metrics.items() if k not in ("all_trades", "mfe_report", "equity_curve")}
        # Keep trade summaries (not full trade dicts for size)
        clean["trade_count"] = metrics.get("trade_count", 0)
        return clean

    result = {
        "generated_at": datetime.now().isoformat(),
        "comparison_days": COMPARISON_DAYS,
        "bars_per_day": BARS_PER_DAY,
        "total_symbols": len(FIXED_SYMBOLS),
        "baseline": _clean(baseline),
        "smart_exit_0.55": _clean(smart),
    }

    if variants:
        for v in variants:
            key = v["mode"].lower().replace(" ", "_")
            result[key] = _clean(v)

    # Determine winner
    if smart["sharpe"] > baseline["sharpe"]:
        result["winner"] = smart["mode"]
    else:
        best = smart
        if variants:
            for v in variants:
                if v["sharpe"] > best["sharpe"]:
                    best = v
        result["winner"] = best["mode"] if best["sharpe"] > baseline["sharpe"] else baseline["mode"]

    if regime_sequence:
        result["regime_sequence"] = regime_sequence

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")


def print_success_check(baseline: Dict, smart: Dict) -> None:
    """Check and print success criteria."""
    print("  SUCCESS CRITERIA CHECK:")
    print(f"  {'─' * 50}")

    sharpe_delta = smart["sharpe"] - baseline["sharpe"]
    dd_improvement = baseline["max_drawdown_pct"] - smart["max_drawdown_pct"]
    capture = smart["avg_capture_ratio"]

    checks = [
        (
            f"Sharpe delta >= 0.30",
            sharpe_delta >= 0.30,
            f"{sharpe_delta:+.2f}",
        ),
        (
            f"SmartExit MaxDD < Baseline MaxDD",
            smart["max_drawdown_pct"] > baseline["max_drawdown_pct"],  # less negative = better
            f"{smart['max_drawdown_pct']:.2f}% vs {baseline['max_drawdown_pct']:.2f}%",
        ),
        (
            f"Capture ratio > 0.45",
            capture > 0.45,
            f"{capture:.3f}",
        ),
    ]

    all_pass = True
    for label, passed, detail in checks:
        status = "\u2705" if passed else "\u274c"
        all_pass = all_pass and passed
        print(f"  {status} {label:<38} {detail}")

    print(f"  {'─' * 50}")
    if all_pass:
        print("  \u2705 ALL CRITERIA MET \u2014 SmartExitEngine validated!")
    else:
        print("  \u26a0\ufe0f  Some criteria not met \u2014 consider tuning signal weights")
    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Main entry point — orchestrates the full comparison."""
    t0 = time.time()

    print("=" * 70)
    print("  EXIT STRATEGY COMPARISON: Baseline vs SmartExitEngine")
    print("=" * 70)

    # 1. Load data
    print("\n[1/7] Loading data and indicators...")
    data, indicator_data = load_data()
    print(f"       {len(data)} symbols with OHLCV, {len(indicator_data)} with indicators")

    if len(data) < 5:
        print("ERROR: Not enough data loaded. Need cached 5m data.")
        return

    # 2. Load player configs
    print("[2/7] Loading player configs...")
    configs = load_evolved_configs()
    if not configs:
        configs = deepcopy(PLAYERS_CONFIG)
        print("       Using default configs (no evolved configs found)")
    else:
        print(f"       Loaded evolved configs for {len(configs)} players")

    # 3. Compute regime + total days
    sample_df = list(data.values())[0]
    total_bars = len(sample_df)
    total_days = min(COMPARISON_DAYS, total_bars // BARS_PER_DAY)
    print(f"[3/7] Computing regime sequence ({total_days} days)...")
    regime_sequence, hmm_model, regime_state_map = compute_regime_sequence(data, total_days)

    regime_counts = {}
    for r in regime_sequence:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    print(f"       Regimes: {regime_counts}")
    if hmm_model is not None:
        print(f"       HMM model loaded — regime_transition signal enabled")
    else:
        print(f"       HMM model not available — regime_transition signal returns 0.0")

    # 4. Generate entry signals (Phase 1)
    print("[4/7] Generating entry signals (shared across both modes)...")
    entry_events, signal_map = generate_entry_signals(
        data, indicator_data, configs, regime_sequence, total_days,
    )
    print(f"       {len(entry_events)} entry signals recorded across {total_days} days")

    # 5. Run BASELINE
    print("[5/7] Running BASELINE simulation...")
    t_base = time.time()
    baseline_players = simulate(
        "baseline", entry_events, data, indicator_data,
        signal_map, configs, regime_sequence, total_days,
    )
    baseline_metrics = compute_metrics(baseline_players, total_days, "BASELINE")
    print(
        f"       {baseline_metrics['trade_count']} trades, "
        f"return={baseline_metrics['total_return_pct']:+.2f}%, "
        f"sharpe={baseline_metrics['sharpe']:.2f} "
        f"({time.time() - t_base:.1f}s)"
    )

    # 6. Run SMART_EXIT
    print("[6/7] Running SMART_EXIT simulation (threshold=0.55)...")
    t_smart = time.time()
    smart_players = simulate(
        "smart_exit", entry_events, data, indicator_data,
        signal_map, configs, regime_sequence, total_days,
        exit_threshold=0.55,
        hmm_model=hmm_model,
        regime_state_map=regime_state_map,
    )
    smart_metrics = compute_metrics(smart_players, total_days, "SMART_EXIT_0.55")
    print(
        f"       {smart_metrics['trade_count']} trades, "
        f"return={smart_metrics['total_return_pct']:+.2f}%, "
        f"sharpe={smart_metrics['sharpe']:.2f}, "
        f"partial_exits={smart_metrics['partial_exit_count']} "
        f"({time.time() - t_smart:.1f}s)"
    )

    # 7. Threshold sensitivity (if SmartExit underperforms)
    smart_variants: List[Dict] = []
    if smart_metrics["sharpe"] < baseline_metrics["sharpe"]:
        print("       SmartExit Sharpe < Baseline — testing threshold variants...")
        for threshold in [0.50, 0.60]:
            variant_players = simulate(
                "smart_exit", entry_events, data, indicator_data,
                signal_map, configs, regime_sequence, total_days,
                exit_threshold=threshold,
                hmm_model=hmm_model,
                regime_state_map=regime_state_map,
            )
            variant_metrics = compute_metrics(
                variant_players, total_days, f"SMART_EXIT_{threshold}",
            )
            smart_variants.append(variant_metrics)
            print(
                f"       threshold={threshold}: "
                f"sharpe={variant_metrics['sharpe']:.2f}, "
                f"return={variant_metrics['total_return_pct']:+.2f}%"
            )

    # Output
    print(f"\n[7/7] Generating comparison report...\n")
    print_comparison_table(baseline_metrics, smart_metrics, smart_variants or None)
    print_exit_reason_chart(baseline_metrics, smart_metrics)
    print_mfe_scatter_analysis(baseline_metrics, smart_metrics)
    print_success_check(baseline_metrics, smart_metrics)

    save_results(baseline_metrics, smart_metrics, smart_variants or None, regime_sequence)
    print(f"  Results saved to {RESULTS_PATH}")
    print(f"  Total time: {time.time() - t0:.1f}s")
    print()


if __name__ == "__main__":
    main()
