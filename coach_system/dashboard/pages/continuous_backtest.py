"""
Continuous 5-Player Backtest Runner with AI Coach.

FIXED SETTINGS (No Configuration):
  - 50 trading days
  - Coach every 3 days
  - Nifty 30 stocks
  - 100,000 capital per player

Features:
  1. Runs the full 5-player trading system (Aggressive, Conservative, Balanced, VolBreakout, Momentum)
  2. Each player has independent strategy and gets personalized coaching
  3. AI Coach optimizes each player based on their individual performance
  4. Tracks performance across multiple runs
"""

import json
import os
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import streamlit as st
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    raise ImportError("streamlit and plotly required: pip install streamlit plotly")

from coach_system.dashboard.theme import COACH_COLORS as AQTIS_COLORS

# ============================================================================
# FIXED SETTINGS - DO NOT CHANGE
# ============================================================================
FIXED_DAYS = 50
FIXED_COACH_INTERVAL = 3
FIXED_CAPITAL = 100000
FIXED_SYMBOLS = [
    # Nifty 30 stocks
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
    "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "WIPRO.NS", "HCLTECH.NS", "ULTRACEMCO.NS",
    "POWERGRID.NS", "NTPC.NS", "NESTLEIND.NS", "TATAMOTORS.NS", "M&M.NS",
    "ONGC.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "ADANIPORTS.NS", "TECHM.NS",
]

# Player colors for charts
PLAYER_COLORS = {
    "PLAYER_1": "#ff6b6b",  # Red - Aggressive
    "PLAYER_2": "#4dabf7",  # Blue - Conservative
    "PLAYER_3": "#51cf66",  # Green - Balanced
    "PLAYER_4": "#ffd43b",  # Yellow - VolBreakout
    "PLAYER_5": "#da77f2",  # Purple - Momentum
}

PLAYER_LABELS = {
    "PLAYER_1": "Aggressive",
    "PLAYER_2": "Conservative",
    "PLAYER_3": "Balanced",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "Momentum",
}

# Default 5-player configurations
PLAYERS_CONFIG = {
    "PLAYER_1": {
        "label": "Aggressive",
        "weights": {
            "RSI_7": 0.90, "STOCH_5_3": 0.85, "TSI_13_25": 0.75,
            "CMO_14": 0.70, "WILLR_14": 0.65, "OBV": 0.60,
            "MFI_14": 0.55, "ADX_14": 0.50, "DEMA_20": 0.45,
            "NATR_14": 0.40,
        },
        "entry_threshold": 0.25,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_2": {
        "label": "Conservative",
        "weights": {
            "ADX_14": 0.90, "SUPERTREND_7_3": 0.85, "EMA_50": 0.80,
            "AROON_14": 0.70, "CMF_20": 0.65, "RSI_14": 0.60,
            "BBANDS_20_2": 0.55, "OBV": 0.50, "VWMA_20": 0.45,
            "HMA_9": 0.40, "ATR_14": 0.35,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.15,
        "min_hold_bars": 5,
    },
    "PLAYER_3": {
        "label": "Balanced",
        "weights": {
            "RSI_14": 0.85, "BBANDS_20_2": 0.80, "STOCH_14_3": 0.75,
            "CMF_20": 0.65, "ZSCORE_20": 0.60, "MFI_20": 0.55,
            "DEMA_20": 0.50, "ADX_14": 0.45, "ATR_14": 0.40,
            "TEMA_20": 0.35,
        },
        "entry_threshold": 0.25,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_4": {
        "label": "VolBreakout",
        "weights": {
            "NATR_14": 0.95, "KC_20_2": 0.90, "ADX_14": 0.85,
            "BBANDS_20_2": 0.75, "ATR_14": 0.70, "CCI_14": 0.60,
            "RSI_7": 0.55, "OBV": 0.50, "CMF_20": 0.45,
            "WILLR_14": 0.40,
        },
        "entry_threshold": 0.22,
        "exit_threshold": -0.08,
        "min_hold_bars": 3,
    },
    "PLAYER_5": {
        "label": "Momentum",
        "weights": {
            "RSI_7": 0.95, "TSI_13_25": 0.90, "MACD_12_26_9": 0.85,
            "CMO_14": 0.80, "STOCH_5_3": 0.75, "COPPOCK": 0.65,
            "ROC_20": 0.60, "ROC_10": 0.55, "MOM_10": 0.50,
            "DEMA_20": 0.45,
        },
        "entry_threshold": 0.23,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
}


@dataclass
class PlayerState:
    """State for a single player during backtest."""
    player_id: str
    config: Dict
    equity: float = 100000.0
    positions: Dict = field(default_factory=dict)
    bars_held: Dict = field(default_factory=dict)
    prev_si: Dict = field(default_factory=dict)
    trades: List = field(default_factory=list)
    daily_pnl: List = field(default_factory=list)
    equity_curve: List = field(default_factory=list)
    coach_history: List = field(default_factory=list)


EVOLVED_CONFIGS_PATH = Path(__file__).parent.parent.parent.parent / "evolved_player_configs.json"


def save_evolved_configs(configs: Dict, performance: Dict = None):
    """Save evolved configs - each player's BEST config is saved independently."""
    try:
        existing_data = {}
        if EVOLVED_CONFIGS_PATH.exists():
            with open(EVOLVED_CONFIGS_PATH, "r") as f:
                existing_data = json.load(f)

        existing_configs = existing_data.get("configs", {})
        player_best_pnl = existing_data.get("player_best_pnl", {})
        total_runs = existing_data.get("total_runs", 0) + 1

        updates = []
        if performance and performance.get("players"):
            for pid, pdata in performance["players"].items():
                current_pnl = pdata.get("pnl", 0)
                existing_best = player_best_pnl.get(pid, float("-inf"))

                if current_pnl > existing_best:
                    existing_configs[pid] = configs[pid]
                    player_best_pnl[pid] = current_pnl
                    updates.append(f"{PLAYER_LABELS.get(pid, pid)}: ₹{current_pnl:+,.0f}")
                else:
                    if pid not in existing_configs:
                        existing_configs[pid] = configs[pid]
                        player_best_pnl[pid] = current_pnl

        for pid in configs:
            if pid not in existing_configs:
                existing_configs[pid] = configs[pid]
                player_best_pnl[pid] = 0

        save_data = {
            "saved_at": datetime.now().isoformat(),
            "configs": existing_configs,
            "player_best_pnl": player_best_pnl,
            "total_runs": total_runs,
            "last_update": datetime.now().isoformat(),
        }

        with open(EVOLVED_CONFIGS_PATH, "w") as f:
            json.dump(save_data, f, indent=2)

    except Exception as e:
        print(f"[Config] Failed to save: {e}")


def load_evolved_configs() -> Optional[Dict]:
    """Load the BEST evolved configs for each player."""
    try:
        if EVOLVED_CONFIGS_PATH.exists():
            with open(EVOLVED_CONFIGS_PATH, "r") as f:
                data = json.load(f)
            return data.get("configs")
    except Exception:
        pass
    return None


def get_player_best_pnls() -> Dict[str, float]:
    """Get each player's best P&L for display."""
    try:
        if EVOLVED_CONFIGS_PATH.exists():
            with open(EVOLVED_CONFIGS_PATH, "r") as f:
                data = json.load(f)
            return data.get("player_best_pnl", {})
    except Exception:
        pass
    return {}


def init_session_state():
    """Initialize session state variables."""
    if "continuous_running" not in st.session_state:
        st.session_state.continuous_running = False
    if "continuous_results" not in st.session_state:
        st.session_state.continuous_results = []
    if "run_count" not in st.session_state:
        st.session_state.run_count = 0
    if "last_run_time" not in st.session_state:
        st.session_state.last_run_time = None
    if "player_configs" not in st.session_state:
        loaded = load_evolved_configs()
        if loaded:
            st.session_state.player_configs = loaded
            st.session_state.configs_evolved = True
        else:
            st.session_state.player_configs = deepcopy(PLAYERS_CONFIG)
            st.session_state.configs_evolved = False
    if "configs_evolved" not in st.session_state:
        st.session_state.configs_evolved = False


# Cache data fetching for speed
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_all_data(symbols_tuple, interval: str, days: int) -> Dict:
    """Fetch and cache OHLCV data for all symbols."""
    symbols = list(symbols_tuple)
    data = {}

    for sym in symbols:
        df = _fetch_symbol_data(sym, interval, days)
        if df is not None and not df.empty:
            data[sym] = df

    return data


def _fetch_symbol_data(symbol: str, interval: str = "5m", days: int = 60) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a symbol."""
    df = None

    try:
        from data.local_cache import get_cache
        cache = get_cache()
        df = cache.get_data(symbol, interval=interval)
    except Exception:
        pass

    if df is None or df.empty:
        try:
            import yfinance as yf
            end = datetime.now()
            start = end - timedelta(days=days + 10)
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=interval,
            )
        except Exception:
            pass

    if df is not None and not df.empty:
        col_map = {}
        for c in df.columns:
            cl = c.lower()
            if cl in ['open', 'high', 'low', 'close', 'volume']:
                col_map[c] = cl
        if col_map:
            df = df.rename(columns=col_map)
        return df

    return None


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


def run_5player_backtest(player_configs: Dict) -> Dict:
    """
    Run a full 5-player backtest with AI coach optimization.
    Uses FIXED settings: 50 days, coach every 3 days, Nifty 30 stocks.
    """
    symbols = FIXED_SYMBOLS
    days = FIXED_DAYS
    coach_interval = FIXED_COACH_INTERVAL
    initial_capital = FIXED_CAPITAL

    # Load data (cached)
    data = _fetch_all_data(tuple(symbols), "5m", days + 20)

    if not data:
        return {"error": "No data fetched"}

    # Calculate indicators
    try:
        from trading_evolution.indicators.calculator import IndicatorCalculator
        from trading_evolution.indicators.normalizer import IndicatorNormalizer
        from trading_evolution.indicators.universe import IndicatorUniverse

        universe = IndicatorUniverse()
        universe.load_all()
        calculator = IndicatorCalculator(universe=universe)
        normalizer = IndicatorNormalizer()

        indicator_data = {}
        for sym, df in data.items():
            try:
                raw = calculator.calculate_all(df)
                raw = calculator.rename_to_dna_names(raw)
                indicator_data[sym] = raw
            except Exception:
                pass
    except ImportError:
        return {"error": "Indicator modules not available"}

    if not indicator_data:
        return {"error": "Could not compute indicators"}

    # Initialize players
    players = {}
    for pid, config in player_configs.items():
        cfg = deepcopy(config)
        players[pid] = PlayerState(
            player_id=pid,
            config=cfg,
            equity=initial_capital,
        )
        players[pid].equity_curve.append(initial_capital)

    # Simulate
    bars_per_day = 26
    sample_df = list(data.values())[0]
    total_bars = len(sample_df)
    total_days = min(days, total_bars // bars_per_day)

    coach_sessions = []

    def detect_regime(day_idx: int) -> str:
        """Detect market regime from recent price action."""
        try:
            sym = list(data.keys())[0]
            df = data[sym]
            close_col = "close" if "close" in df.columns else "Close"

            end_bar = (day_idx + 1) * bars_per_day
            start_bar = max(0, end_bar - bars_per_day * 5)

            if end_bar > len(df):
                return "unknown"

            recent = df.iloc[start_bar:end_bar]
            if recent.empty or close_col not in recent.columns:
                return "unknown"

            close = recent[close_col]
            returns = close.pct_change().dropna()

            if len(returns) < 10:
                return "unknown"

            vol = returns.std() * np.sqrt(252)
            trend = (close.iloc[-1] - close.iloc[0]) / close.iloc[0]

            if vol > 0.25:
                if trend > 0.02:
                    return "volatile_bullish"
                elif trend < -0.02:
                    return "volatile_bearish"
                return "volatile"
            elif abs(trend) < 0.01:
                return "ranging"
            elif trend > 0:
                return "trending_up"
            return "trending_down"
        except Exception:
            return "unknown"

    # Day simulation loop
    for day in range(total_days):
        start_bar = day * bars_per_day
        end_bar = start_bar + bars_per_day

        # Each player trades independently
        for pid, state in players.items():
            config = state.config
            weights = config["weights"]
            entry_thresh = config["entry_threshold"]
            exit_thresh = config["exit_threshold"]
            min_hold = config["min_hold_bars"]

            day_pnl = 0.0

            for sym, df in data.items():
                if sym not in indicator_data:
                    continue

                raw = indicator_data[sym]
                close_col = "close" if "close" in df.columns else "Close"

                if end_bar > len(df):
                    continue

                day_df = df.iloc[start_bar:end_bar]
                day_raw = raw.iloc[start_bar:end_bar]

                if day_df.empty or day_raw.empty:
                    continue

                active = [i for i in weights.keys() if i in day_raw.columns]
                if not active:
                    continue

                try:
                    norm = normalizer.normalize_all(day_raw[active], price_series=day_df[close_col])
                    if norm.empty:
                        continue

                    si = raw_weighted_average(norm, {k: weights[k] for k in active})
                    if si.empty:
                        continue

                    last_si = float(si.iloc[-1])
                    last_price = float(day_df[close_col].iloc[-1])
                    atr = float(day_raw["ATR_14"].iloc[-1]) if "ATR_14" in day_raw.columns else last_price * 0.02

                    # Check exits first
                    if sym in state.positions:
                        pos = state.positions[sym]
                        state.bars_held[sym] = state.bars_held.get(sym, 0) + 1

                        exit_signal = False
                        exit_reason = ""

                        if pos["direction"] == "LONG":
                            if last_price <= pos["stop_loss"]:
                                exit_signal, exit_reason = True, "stop_loss"
                            elif last_price >= pos["take_profit"]:
                                exit_signal, exit_reason = True, "take_profit"
                            elif last_si < exit_thresh and state.bars_held[sym] >= min_hold:
                                exit_signal, exit_reason = True, "signal_exit"
                        else:
                            if last_price >= pos["stop_loss"]:
                                exit_signal, exit_reason = True, "stop_loss"
                            elif last_price <= pos["take_profit"]:
                                exit_signal, exit_reason = True, "take_profit"
                            elif last_si > -exit_thresh and state.bars_held[sym] >= min_hold:
                                exit_signal, exit_reason = True, "signal_exit"

                        if exit_signal:
                            entry_price = pos["entry_price"]
                            qty = pos["quantity"]
                            if pos["direction"] == "LONG":
                                pnl = (last_price - entry_price) * qty
                            else:
                                pnl = (entry_price - last_price) * qty

                            state.equity += pnl
                            day_pnl += pnl

                            state.trades.append({
                                "symbol": sym,
                                "direction": pos["direction"],
                                "entry_price": entry_price,
                                "exit_price": last_price,
                                "quantity": qty,
                                "pnl": pnl,
                                "exit_reason": exit_reason,
                                "bars_held": state.bars_held[sym],
                            })

                            del state.positions[sym]
                            state.bars_held.pop(sym, None)

                    # Check entries
                    if sym not in state.positions and len(state.positions) < 5:
                        if last_si > entry_thresh:
                            direction = "LONG"
                        elif last_si < -entry_thresh:
                            direction = "SHORT"
                        else:
                            direction = None

                        if direction:
                            max_position_value = state.equity * 0.20
                            qty = max(1, int(max_position_value / last_price))
                            cost = qty * last_price
                            stop_dist = atr * 2 if atr > 0 else last_price * 0.02

                            if cost <= state.equity * 0.25:
                                if direction == "LONG":
                                    sl = last_price - stop_dist
                                    tp = last_price + atr * 3
                                else:
                                    sl = last_price + stop_dist
                                    tp = last_price - atr * 3

                                state.positions[sym] = {
                                    "direction": direction,
                                    "entry_price": last_price,
                                    "quantity": qty,
                                    "stop_loss": sl,
                                    "take_profit": tp,
                                }
                                state.bars_held[sym] = 0

                    state.prev_si[sym] = last_si

                except Exception:
                    continue

            state.daily_pnl.append(day_pnl)
            state.equity_curve.append(state.equity)

        # Coach optimization - every 3 days
        if (day + 1) % coach_interval == 0 and day < total_days - 1:
            regime = detect_regime(day)

            try:
                from coach_system.coaches.ai_coach import AICoach
                from coach_system.llm.gemini_provider import GeminiProvider
                import dotenv

                env_path = Path(__file__).parent.parent.parent.parent / ".env"
                dotenv.load_dotenv(env_path)

                llm = None
                try:
                    llm = GeminiProvider()
                    if not llm.is_available():
                        print("[Coach] Gemini API key not found - using fallback")
                        llm = None
                    else:
                        print(f"[Coach] Gemini provider ready with model: {llm.model}")
                except Exception as e:
                    print(f"[Coach] Failed to initialize Gemini: {e}")
                    llm = None

                coach = AICoach(use_llm=(llm is not None), llm_provider=llm)

                session = {"day": day + 1, "regime": regime, "updates": {}, "llm_used": llm is not None}

                for pid, state in players.items():
                    recent_trades = state.trades[-50:] if len(state.trades) > 50 else state.trades

                    if not recent_trades:
                        continue

                    analysis = coach.analyze_player(
                        player_id=pid,
                        player_label=state.config.get("label", "Balanced"),
                        trades=recent_trades,
                        current_weights=state.config.get("weights", {}),
                        current_config=state.config,
                        market_regime=regime,
                    )

                    new_config = coach.apply_recommendations(state.config, analysis)
                    state.config = new_config

                    state.coach_history.append({
                        "day": day + 1,
                        "win_rate": analysis.win_rate,
                        "pnl": analysis.total_pnl,
                        "num_indicators": len(new_config.get("weights", {})),
                    })

                    session["updates"][pid] = {
                        "label": state.config.get("label"),
                        "win_rate": analysis.win_rate,
                        "pnl": analysis.total_pnl,
                        "indicators_added": list(analysis.indicators_to_add.keys()),
                        "indicators_removed": analysis.indicators_to_remove,
                    }

                coach_sessions.append(session)

            except ImportError:
                pass

    llm_was_used = any(s.get("llm_used", False) for s in coach_sessions) if coach_sessions else False

    # Compile results
    results = {
        "run_time": datetime.now().isoformat(),
        "total_days": total_days,
        "symbols": len(data),
        "coach_interval": coach_interval,
        "coach_sessions": len(coach_sessions),
        "llm_used": llm_was_used,
        "market_regime": detect_regime(total_days - 1),
        "players": {},
        "team_total_pnl": 0,
        "team_total_trades": 0,
    }

    for pid, state in players.items():
        trades = len(state.trades)
        wins = sum(1 for t in state.trades if t["pnl"] > 0)
        win_rate = wins / trades if trades > 0 else 0
        pnl = state.equity - initial_capital

        win_pnls = [t["pnl"] for t in state.trades if t["pnl"] > 0]
        loss_pnls = [t["pnl"] for t in state.trades if t["pnl"] <= 0]
        total_wins = sum(win_pnls) if win_pnls else 0
        total_losses = abs(sum(loss_pnls)) if loss_pnls else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        if state.daily_pnl:
            returns = np.array(state.daily_pnl) / initial_capital
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        eq_curve = state.equity_curve
        if len(eq_curve) > 1:
            peak = pd.Series(eq_curve).expanding().max()
            dd = (pd.Series(eq_curve) - peak) / peak
            max_dd = float(dd.min())
        else:
            max_dd = 0

        results["players"][pid] = {
            "label": state.config.get("label", "Unknown"),
            "trades": trades,
            "wins": wins,
            "losses": trades - wins,
            "win_rate": win_rate,
            "pnl": pnl,
            "sharpe": sharpe,
            "profit_factor": profit_factor,
            "max_drawdown": max_dd,
            "final_equity": state.equity,
            "final_threshold": state.config.get("entry_threshold", 0.30),
            "num_indicators": len(state.config.get("weights", {})),
        }

        results["team_total_pnl"] += pnl
        results["team_total_trades"] += trades

    # Update stored configs
    updated_configs = {}
    for pid, state in players.items():
        updated_configs[pid] = state.config

    return results, updated_configs


def render_continuous_backtest(memory=None):
    """Render the Continuous 5-Player Backtest page."""
    init_session_state()

    st.header("5-Player Backtest with AI Coach")

    # Fixed settings info
    st.info(f"**Fixed Settings:** {FIXED_DAYS} days | Coach every {FIXED_COACH_INTERVAL} days | {len(FIXED_SYMBOLS)} Nifty stocks | ₹{FIXED_CAPITAL:,} per player")

    # Show each player's best P&L
    try:
        if EVOLVED_CONFIGS_PATH.exists():
            with open(EVOLVED_CONFIGS_PATH, "r") as f:
                saved_data = json.load(f)

            player_best_pnl = saved_data.get("player_best_pnl", {})
            total_runs = saved_data.get("total_runs", 0)
            team_best = sum(player_best_pnl.values())

            st.markdown("### Personal Best (Per Player)")

            pcols = st.columns(5)
            for i, pid in enumerate(["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]):
                best = player_best_pnl.get(pid, 0)
                label = PLAYER_LABELS.get(pid, pid)
                color = "normal" if best >= 0 else "inverse"
                pcols[i].metric(label, f"₹{best:+,.0f}", delta_color=color)

            st.caption(f"Combined Best: ₹{team_best:+,.0f} | Total Runs: {total_runs}")
            st.divider()
    except Exception:
        pass

    # Control buttons
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("Run Backtest", type="primary", use_container_width=True):
            progress = st.progress(0, text="Fetching data...")

            try:
                progress.progress(10, text="Fetching market data...")

                result, new_configs = run_5player_backtest(
                    player_configs=st.session_state.player_configs,
                )

                progress.progress(90, text="Saving results...")

                if "error" not in result:
                    result["run_number"] = st.session_state.run_count + 1
                    st.session_state.run_count += 1
                    st.session_state.continuous_results.append(result)
                    st.session_state.last_run_time = datetime.now()

                    # Save configs
                    save_evolved_configs(new_configs, result)
                    best_configs = load_evolved_configs()
                    if best_configs:
                        st.session_state.player_configs = best_configs
                    st.session_state.configs_evolved = True

                    progress.progress(100, text="Complete!")
                    time.sleep(0.5)
                    progress.empty()

                    st.success(f"Run #{result['run_number']} complete! Team P&L: ₹{result['team_total_pnl']:+,.0f}")
                else:
                    progress.empty()
                    st.error(result["error"])

            except Exception as e:
                progress.empty()
                st.error(f"Error: {e}")

    with col2:
        if st.button("Clear Results", use_container_width=True):
            st.session_state.continuous_results = []
            st.session_state.run_count = 0
            st.rerun()

    with col3:
        if st.button("Reset Configs", use_container_width=True):
            st.session_state.player_configs = deepcopy(PLAYERS_CONFIG)
            st.session_state.configs_evolved = False
            if EVOLVED_CONFIGS_PATH.exists():
                EVOLVED_CONFIGS_PATH.unlink()
            st.success("Reset to defaults")
            st.rerun()

    # Display Results
    st.divider()

    results = st.session_state.continuous_results

    if not results:
        st.info("Click 'Run Backtest' to start.")
        return

    # Latest run
    latest = results[-1]

    st.subheader(f"Run #{latest.get('run_number', '?')} Results")
    st.caption(f"Regime: {latest.get('market_regime', 'unknown')} | LLM: {'Active' if latest.get('llm_used') else 'Off'}")

    # Per-player results
    cols = st.columns(5)
    for i, (pid, pdata) in enumerate(latest.get("players", {}).items()):
        with cols[i]:
            pnl = pdata.get("pnl", 0)
            color = "normal" if pnl >= 0 else "inverse"
            st.metric(
                pdata.get("label", pid),
                f"₹{pnl:+,.0f}",
                f"WR: {pdata.get('win_rate', 0):.0%}",
                delta_color=color
            )
            st.caption(f"{pdata.get('trades', 0)} trades | {pdata.get('num_indicators', 0)} inds")

    # Team total
    st.metric("Team Total P&L", f"₹{latest['team_total_pnl']:+,.0f}")

    # Charts if multiple runs
    if len(results) > 1:
        st.subheader("Performance Across Runs")

        player_data = {pid: [] for pid in PLAYER_LABELS.keys()}
        team_pnls = []

        for r in results:
            team_pnls.append(r["team_total_pnl"])
            for pid, pdata in r.get("players", {}).items():
                if pid in player_data:
                    player_data[pid].append(pdata.get("pnl", 0))

        # Cumulative P&L chart
        fig = go.Figure()
        run_numbers = list(range(1, len(results) + 1))

        for pid, pnls in player_data.items():
            if pnls:
                fig.add_trace(go.Scatter(
                    x=run_numbers,
                    y=np.cumsum(pnls),
                    mode="lines+markers",
                    name=PLAYER_LABELS.get(pid, pid),
                    line=dict(color=PLAYER_COLORS.get(pid, "#888")),
                ))

        fig.update_layout(
            title="Cumulative P&L by Player",
            xaxis_title="Run #",
            yaxis_title="Cumulative P&L (₹)",
            height=350,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Team P&L bar chart
        colors = [AQTIS_COLORS["green"] if p >= 0 else AQTIS_COLORS["red"] for p in team_pnls]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=run_numbers, y=team_pnls, marker_color=colors))
        fig2.update_layout(
            title="Team P&L per Run",
            xaxis_title="Run #",
            yaxis_title="P&L (₹)",
            height=250,
            template="plotly_white",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Summary stats
    if len(results) > 1:
        st.subheader("Summary")
        team_pnls = [r["team_total_pnl"] for r in results]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg P&L", f"₹{np.mean(team_pnls):+,.0f}")
        c2.metric("Total P&L", f"₹{sum(team_pnls):+,.0f}")
        c3.metric("Best Run", f"₹{max(team_pnls):+,.0f}")
        c4.metric("Worst Run", f"₹{min(team_pnls):+,.0f}")


def _auto_run():
    try:
        st.set_page_config(page_title="5-Player Backtest", page_icon="📈", layout="wide")
    except Exception:
        pass
    render_continuous_backtest(None)


try:
    _ctx = st.runtime.scriptrunner.get_script_run_ctx()
    if _ctx is not None:
        _auto_run()
except Exception:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
        if _get_ctx() is not None:
            _auto_run()
    except Exception:
        pass
