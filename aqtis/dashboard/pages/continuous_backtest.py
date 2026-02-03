"""
Continuous 5-Player Backtest Runner with AI Coach.

Features:
  1. Runs the full 5-player trading system (Aggressive, Conservative, Balanced, VolBreakout, Momentum)
  2. Each player has independent strategy and gets personalized coaching
  3. AI Coach optimizes each player based on their individual performance
  4. Tracks performance across multiple runs
  5. Auto-run capability for continuous testing
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

from aqtis.dashboard.theme import AQTIS_COLORS

try:
    from data.symbols import NIFTY_50_SYMBOLS
except ImportError:
    try:
        from data_cache.symbols import NIFTY_50_SYMBOLS
    except ImportError:
        NIFTY_50_SYMBOLS = [
            "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
            "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
            "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
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

# Default 5-player configurations (using available indicators)
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


def save_evolved_configs(configs: Dict):
    """Save evolved configs to file for persistence across sessions."""
    try:
        # Convert to serializable format
        save_data = {
            "saved_at": datetime.now().isoformat(),
            "configs": configs,
        }
        with open(EVOLVED_CONFIGS_PATH, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"[Config] Saved evolved configs to {EVOLVED_CONFIGS_PATH}")
    except Exception as e:
        print(f"[Config] Failed to save configs: {e}")


def load_evolved_configs() -> Optional[Dict]:
    """Load evolved configs from file if available."""
    try:
        if EVOLVED_CONFIGS_PATH.exists():
            with open(EVOLVED_CONFIGS_PATH, "r") as f:
                data = json.load(f)
            print(f"[Config] Loaded evolved configs from {data.get('saved_at', 'unknown')}")
            return data.get("configs")
    except Exception as e:
        print(f"[Config] Failed to load configs: {e}")
    return None


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
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = False
    if "player_configs" not in st.session_state:
        # Try to load evolved configs from file, otherwise use defaults
        loaded = load_evolved_configs()
        if loaded:
            st.session_state.player_configs = loaded
            st.session_state.configs_evolved = True
        else:
            st.session_state.player_configs = deepcopy(PLAYERS_CONFIG)
            st.session_state.configs_evolved = False
    if "configs_evolved" not in st.session_state:
        st.session_state.configs_evolved = False


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
        # Normalize column names to lowercase
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


def run_5player_backtest(
    symbols: List[str],
    days: int,
    coach_interval: int,
    initial_capital: float,
    player_configs: Dict,
    use_ai_coach: bool = True,
) -> Dict:
    """
    Run a full 5-player backtest with AI coach optimization.

    Each player trades independently with their own strategy.
    Coach analyzes each player individually and provides personalized recommendations.
    """
    # Load data
    data = {}
    for sym in symbols:
        df = _fetch_symbol_data(sym, interval="5m", days=days + 20)
        if df is not None and not df.empty:
            data[sym] = df

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

    # Debug info
    _debug = False  # Set to True for verbose output
    _trades_entered = 0

    # Day simulation loop
    for day in range(total_days):
        start_bar = day * bars_per_day
        end_bar = start_bar + bars_per_day

        if _debug and day < 3:
            print(f"Day {day}: bars {start_bar}-{end_bar}")

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

                # Get active indicators for this player
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
                                exit_signal = True
                                exit_reason = "stop_loss"
                            elif last_price >= pos["take_profit"]:
                                exit_signal = True
                                exit_reason = "take_profit"
                            elif last_si < exit_thresh and state.bars_held[sym] >= min_hold:
                                exit_signal = True
                                exit_reason = "signal_exit"
                        else:
                            if last_price >= pos["stop_loss"]:
                                exit_signal = True
                                exit_reason = "stop_loss"
                            elif last_price <= pos["take_profit"]:
                                exit_signal = True
                                exit_reason = "take_profit"
                            elif last_si > -exit_thresh and state.bars_held[sym] >= min_hold:
                                exit_signal = True
                                exit_reason = "signal_exit"

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
                            # Position sizing: max 20% of equity per position
                            max_position_value = state.equity * 0.20
                            qty = max(1, int(max_position_value / last_price))
                            cost = qty * last_price

                            # Calculate stop based on ATR
                            stop_dist = atr * 2 if atr > 0 else last_price * 0.02

                            if _debug and day < 3:
                                print(f"  {pid} {sym}: {direction} signal SI={last_si:.3f}, qty={qty}, cost={cost:.0f}")

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
                                _trades_entered += 1

                                if _debug:
                                    print(f"    -> ENTERED {direction} {qty} shares @ {last_price:.2f}")

                    state.prev_si[sym] = last_si

                except Exception as e:
                    # Log errors for debugging
                    import traceback
                    print(f"Error in {pid} {sym}: {e}")
                    traceback.print_exc()
                    continue

            state.daily_pnl.append(day_pnl)
            state.equity_curve.append(state.equity)

        # Coach optimization - each player analyzed INDEPENDENTLY
        if (day + 1) % coach_interval == 0 and day < total_days - 1:
            regime = detect_regime(day)

            if use_ai_coach:
                try:
                    from aqtis.coaches.ai_coach import AICoach
                    from aqtis.llm.gemini_provider import GeminiProvider
                    import dotenv

                    # Load env from project root (New Project/.env)
                    # __file__ = .../aqtis/dashboard/pages/continuous_backtest.py
                    # Need: .../New Project/.env (4 parents up)
                    env_path = Path(__file__).parent.parent.parent.parent / ".env"
                    dotenv.load_dotenv(env_path)

                    # Initialize Gemini LLM
                    llm = None
                    try:
                        llm = GeminiProvider()
                        if llm.is_available():
                            print(f"[Coach Day {day+1}] Using Gemini LLM: {llm.model}")
                        else:
                            llm = None
                    except Exception as e:
                        print(f"[Coach] LLM error: {e}")
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
                            "profit_factor": analysis.profit_factor,
                            "weights_changed": len(analysis.weight_changes),
                            "indicators_added": list(analysis.indicators_to_add.keys()),
                            "indicators_removed": analysis.indicators_to_remove,
                            "best_indicators": analysis.best_indicators,
                            "worst_indicators": analysis.worst_indicators,
                            "new_threshold": new_config.get("entry_threshold", 0.30),
                            "num_indicators": len(new_config.get("weights", {})),
                        })

                        session["updates"][pid] = {
                            "label": state.config.get("label"),
                            "win_rate": analysis.win_rate,
                            "pnl": analysis.total_pnl,
                            "weights_changed": len(analysis.weight_changes),
                            "indicators_added": list(analysis.indicators_to_add.keys()),
                            "indicators_removed": analysis.indicators_to_remove,
                            "best_indicators": analysis.best_indicators,
                            "worst_indicators": analysis.worst_indicators,
                            "llm_recommendations": analysis.llm_recommendations,
                            "llm_analysis": analysis.llm_analysis[:500] if analysis.llm_analysis else "",
                        }

                        # Print coach summary
                        summary = coach.get_coach_summary(analysis)
                        print(f"  [Coach] {pid} ({state.config.get('label')}): {summary}")

                    coach_sessions.append(session)

                except ImportError:
                    # Fallback: basic threshold adjustment
                    for pid, state in players.items():
                        trades = state.trades[-30:]
                        if trades:
                            wins = sum(1 for t in trades if t["pnl"] > 0)
                            wr = wins / len(trades)
                            if wr < 0.40:
                                state.config["entry_threshold"] = min(0.45, state.config["entry_threshold"] + 0.02)
                            elif wr > 0.55:
                                state.config["entry_threshold"] = max(0.20, state.config["entry_threshold"] - 0.02)

    # Check if LLM was used in any session
    llm_was_used = any(s.get("llm_used", False) for s in coach_sessions) if coach_sessions else False

    # Compile results
    results = {
        "run_time": datetime.now().isoformat(),
        "total_days": total_days,
        "symbols": len(data),
        "coach_interval": coach_interval,
        "coach_sessions": len(coach_sessions),
        "coach_sessions_detail": coach_sessions,  # Full details for display
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

        # Max drawdown
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
            "coach_sessions": len(state.coach_history),
            "equity_curve": state.equity_curve[-10:],  # Last 10 points for charts
        }

        results["team_total_pnl"] += pnl
        results["team_total_trades"] += trades

    # Update stored configs for next run (learning persists)
    updated_configs = {}
    for pid, state in players.items():
        updated_configs[pid] = state.config

    return results, updated_configs


def render_continuous_backtest(memory=None):
    """Render the Continuous 5-Player Backtest page."""
    init_session_state()

    st.header("Continuous 5-Player Backtest with AI Coach")
    st.markdown(
        "Run the **full 5-player trading system** with independent AI coaching for each player. "
        "Strategies evolve across runs as the coach learns from each player's performance."
    )

    # Sidebar Configuration
    st.sidebar.subheader("Backtest Settings")

    # Symbol Selection
    symbol_mode = st.sidebar.radio("Symbols", ["NIFTY 50 Top 15", "Custom"], index=0)
    if symbol_mode == "NIFTY 50 Top 15":
        symbols = NIFTY_50_SYMBOLS[:15]
    else:
        custom = st.sidebar.text_area("Enter symbols (one per line)", "RELIANCE.NS\nTCS.NS\nINFY.NS\nHDFCBANK.NS\nICICIBANK.NS")
        symbols = [s.strip() for s in custom.strip().split("\n") if s.strip()]

    st.sidebar.markdown("**Parameters**")
    days = st.sidebar.slider("Trading Days per Run", 10, 60, 30, 5)
    coach_interval = st.sidebar.slider("Coach Interval (days)", 1, 10, 3, 1)
    capital = st.sidebar.number_input("Capital per Player", 50000, 500000, 100000, 10000)

    use_ai_coach = st.sidebar.checkbox("Use AI Coach", value=True, help="Enable per-player AI optimization")
    persist_learning = st.sidebar.checkbox("Persist Learning", value=True, help="Keep learned configs across runs")

    # Auto-run settings
    st.sidebar.markdown("**Auto-Run**")
    auto_interval = st.sidebar.slider("Run interval (seconds)", 30, 300, 120, 30)
    max_runs = st.sidebar.number_input("Max runs (0=unlimited)", 0, 50, 5, 1)

    # Reset configs button
    if st.sidebar.button("Reset All Player Configs"):
        st.session_state.player_configs = deepcopy(PLAYERS_CONFIG)
        st.session_state.configs_evolved = False
        # Delete saved file
        if EVOLVED_CONFIGS_PATH.exists():
            EVOLVED_CONFIGS_PATH.unlink()
        st.success("Player configs reset to defaults")

    # Control buttons
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Run Single Backtest", type="primary"):
            with st.spinner("Running 5-player backtest with AI coach..."):
                result, new_configs = run_5player_backtest(
                    symbols=symbols,
                    days=days,
                    coach_interval=coach_interval,
                    initial_capital=capital,
                    player_configs=st.session_state.player_configs,
                    use_ai_coach=use_ai_coach,
                )

                if "error" not in result:
                    result["run_number"] = st.session_state.run_count + 1
                    # Store evolved configs in result for tracking
                    result["evolved_configs"] = new_configs
                    st.session_state.run_count += 1
                    st.session_state.continuous_results.append(result)
                    st.session_state.last_run_time = datetime.now()

                    if persist_learning:
                        st.session_state.player_configs = new_configs
                        st.session_state.configs_evolved = True
                        # Save to file for persistence
                        save_evolved_configs(new_configs)

                    st.success(f"Run #{result['run_number']} complete! Team P&L: ₹{result['team_total_pnl']:+,.0f}")
                else:
                    st.error(result["error"])

    with col2:
        auto_run = st.checkbox("Auto-Run", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_run

    with col3:
        if st.button("Clear Results"):
            st.session_state.continuous_results = []
            st.session_state.run_count = 0
            st.session_state.player_configs = deepcopy(PLAYERS_CONFIG)
            st.rerun()

    with col4:
        st.metric("Total Runs", st.session_state.run_count)

    # Auto-run logic
    if st.session_state.auto_refresh:
        should_run = False
        if st.session_state.last_run_time is None:
            should_run = True
        else:
            elapsed = (datetime.now() - st.session_state.last_run_time).total_seconds()
            if elapsed >= auto_interval:
                should_run = True

        if max_runs > 0 and st.session_state.run_count >= max_runs:
            st.session_state.auto_refresh = False
            st.info(f"Reached max runs ({max_runs}). Auto-run stopped.")
        elif should_run:
            with st.spinner(f"Auto-running backtest #{st.session_state.run_count + 1}..."):
                result, new_configs = run_5player_backtest(
                    symbols=symbols,
                    days=days,
                    coach_interval=coach_interval,
                    initial_capital=capital,
                    player_configs=st.session_state.player_configs,
                    use_ai_coach=use_ai_coach,
                )

                if "error" not in result:
                    result["run_number"] = st.session_state.run_count + 1
                    result["evolved_configs"] = new_configs
                    st.session_state.run_count += 1
                    st.session_state.continuous_results.append(result)
                    st.session_state.last_run_time = datetime.now()

                    if persist_learning:
                        st.session_state.player_configs = new_configs
                        st.session_state.configs_evolved = True
                        save_evolved_configs(new_configs)

        time.sleep(1)
        st.rerun()

    # Display Results
    st.divider()

    results = st.session_state.continuous_results
    if not results:
        st.info("No backtest results yet. Click 'Run Single Backtest' or enable 'Auto-Run' to start.")

        # Show current player configs with evolution status
        evolved = st.session_state.get("configs_evolved", False)
        st.subheader(f"Current Player Configurations {'🧬 (Evolved)' if evolved else '📋 (Default)'}")

        if evolved:
            st.success("Using evolved configurations from previous learning. These will continue to evolve with each run.")

        for pid, config in st.session_state.player_configs.items():
            original = PLAYERS_CONFIG.get(pid, {})
            orig_inds = len(original.get("weights", {}))
            curr_inds = len(config.get("weights", {}))
            orig_thresh = original.get("entry_threshold", 0.30)
            curr_thresh = config.get("entry_threshold", 0.30)

            # Show evolution delta
            ind_delta = curr_inds - orig_inds
            thresh_delta = curr_thresh - orig_thresh

            label = PLAYER_LABELS.get(pid, pid)
            delta_str = ""
            if evolved and (ind_delta != 0 or abs(thresh_delta) > 0.001):
                delta_str = f" | Δ: {ind_delta:+d} inds, {thresh_delta:+.2f} thresh"

            with st.expander(f"{label}: Entry {curr_thresh:.2f}, {curr_inds} indicators{delta_str}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Current Config:**")
                    st.json({
                        "entry_threshold": config.get("entry_threshold"),
                        "exit_threshold": config.get("exit_threshold"),
                        "min_hold_bars": config.get("min_hold_bars"),
                        "num_indicators": curr_inds,
                        "indicators": list(config.get("weights", {}).keys()),
                    })
                with col2:
                    if evolved:
                        st.markdown("**Original (Default):**")
                        st.json({
                            "entry_threshold": original.get("entry_threshold"),
                            "exit_threshold": original.get("exit_threshold"),
                            "min_hold_bars": original.get("min_hold_bars"),
                            "num_indicators": orig_inds,
                            "indicators": list(original.get("weights", {}).keys()),
                        })
        return

    # Team summary
    st.subheader(f"Team Results Summary ({len(results)} runs)")

    team_pnls = [r["team_total_pnl"] for r in results]
    team_trades = [r["team_total_trades"] for r in results]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Team P&L", f"₹{np.mean(team_pnls):+,.0f}")
    c2.metric("Total Team P&L", f"₹{sum(team_pnls):+,.0f}")
    c3.metric("Total Trades", sum(team_trades))
    c4.metric("Best Run", f"₹{max(team_pnls):+,.0f}")
    c5.metric("Worst Run", f"₹{min(team_pnls):+,.0f}")

    # Per-player performance across runs
    st.subheader("Player Performance Across Runs")

    # Build player data
    player_data = {pid: {"pnls": [], "win_rates": [], "trades": []} for pid in PLAYER_LABELS.keys()}

    for r in results:
        for pid, pdata in r.get("players", {}).items():
            if pid in player_data:
                player_data[pid]["pnls"].append(pdata.get("pnl", 0))
                player_data[pid]["win_rates"].append(pdata.get("win_rate", 0))
                player_data[pid]["trades"].append(pdata.get("trades", 0))

    # Player summary table
    player_summary = []
    for pid, data in player_data.items():
        if data["pnls"]:
            player_summary.append({
                "Player": PLAYER_LABELS.get(pid, pid),
                "Avg P&L": f"₹{np.mean(data['pnls']):+,.0f}",
                "Total P&L": f"₹{sum(data['pnls']):+,.0f}",
                "Avg WR": f"{np.mean(data['win_rates']):.1%}",
                "Total Trades": sum(data["trades"]),
            })

    if player_summary:
        st.dataframe(pd.DataFrame(player_summary), use_container_width=True)

    # P&L chart by player
    st.subheader("P&L by Player Across Runs")

    fig = go.Figure()
    run_numbers = list(range(1, len(results) + 1))

    for pid, data in player_data.items():
        if data["pnls"]:
            fig.add_trace(go.Scatter(
                x=run_numbers,
                y=np.cumsum(data["pnls"]),
                mode="lines+markers",
                name=PLAYER_LABELS.get(pid, pid),
                line=dict(color=PLAYER_COLORS.get(pid, "#888")),
            ))

    fig.update_layout(
        title="Cumulative P&L by Player",
        xaxis_title="Run #",
        yaxis_title="Cumulative P&L (₹)",
        height=400,
        template="plotly_dark",
        paper_bgcolor=AQTIS_COLORS["background"],
        plot_bgcolor=AQTIS_COLORS["card_bg"],
        font=dict(color=AQTIS_COLORS["text"]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Team P&L chart
    st.subheader("Team P&L Across Runs")

    fig2 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         subplot_titles=("P&L per Run", "Cumulative P&L"),
                         vertical_spacing=0.1)

    colors = [AQTIS_COLORS["green"] if p >= 0 else AQTIS_COLORS["red"] for p in team_pnls]
    fig2.add_trace(go.Bar(x=run_numbers, y=team_pnls, marker_color=colors, name="P&L"), row=1, col=1)
    fig2.add_trace(go.Scatter(x=run_numbers, y=np.cumsum(team_pnls), mode="lines+markers",
                              line=dict(color=AQTIS_COLORS["blue"]), name="Cumulative"), row=2, col=1)

    fig2.update_layout(
        height=400,
        showlegend=False,
        template="plotly_dark",
        paper_bgcolor=AQTIS_COLORS["background"],
        plot_bgcolor=AQTIS_COLORS["card_bg"],
        font=dict(color=AQTIS_COLORS["text"]),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Latest run details
    if results:
        latest = results[-1]
        st.subheader(f"Latest Run Details (Run #{latest.get('run_number', '?')})")

        llm_used = latest.get("llm_used", False)
        st.markdown(f"**Market Regime:** {latest.get('market_regime', 'unknown')} | "
                    f"**Days:** {latest.get('total_days', 0)} | "
                    f"**Coach Sessions:** {latest.get('coach_sessions', 0)} | "
                    f"**LLM Coach:** {'✅ Active' if llm_used else '❌ Disabled'}")

        # Per-player details
        cols = st.columns(5)
        for i, (pid, pdata) in enumerate(latest.get("players", {}).items()):
            with cols[i % 5]:
                color = "normal" if pdata.get("pnl", 0) >= 0 else "inverse"
                st.metric(
                    pdata.get("label", pid),
                    f"₹{pdata.get('pnl', 0):+,.0f}",
                    f"WR: {pdata.get('win_rate', 0):.0%}",
                    delta_color=color
                )
                st.caption(f"Trades: {pdata.get('trades', 0)} | Sharpe: {pdata.get('sharpe', 0):.2f}")
                st.caption(f"Indicators: {pdata.get('num_indicators', 0)} | Entry: {pdata.get('final_threshold', 0.30):.2f}")

        # Coach Optimization Details
        st.subheader("🧠 AI Coach Optimization Details")

        coach_sessions_data = latest.get("coach_sessions_detail", [])
        if coach_sessions_data:
            for session in coach_sessions_data[-3:]:  # Last 3 sessions
                st.markdown(f"**Day {session.get('day', '?')} | Regime: {session.get('regime', 'unknown')} | LLM: {'✅' if session.get('llm_used') else '❌'}**")
                updates = session.get("updates", {})

                if updates:
                    coach_cols = st.columns(len(updates))
                    for idx, (pid, update) in enumerate(updates.items()):
                        with coach_cols[idx]:
                            label = update.get("label", pid)
                            st.markdown(f"**{label}**")

                            # Performance
                            wr = update.get("win_rate", 0)
                            pnl = update.get("pnl", 0)
                            st.caption(f"WR: {wr:.1%} | P&L: ₹{pnl:+,.0f}")

                            # Changes made
                            weights_changed = update.get("weights_changed", 0)
                            added = update.get("indicators_added", [])
                            removed = update.get("indicators_removed", [])

                            if weights_changed > 0:
                                st.caption(f"⚖️ {weights_changed} weights adjusted")
                            if added:
                                st.caption(f"➕ Added: {', '.join(added[:3])}")
                            if removed:
                                st.caption(f"➖ Removed: {', '.join(removed[:3])}")

                            # LLM recommendations
                            llm_recs = update.get("llm_recommendations", [])
                            if llm_recs:
                                with st.expander("LLM Insights"):
                                    for rec in llm_recs[:3]:
                                        st.markdown(f"• {rec}")
                st.divider()
        else:
            st.info("No coach optimization sessions recorded yet. Run more days or decrease coach interval.")

    # Strategy Evolution Tracker
    st.subheader("🧬 Strategy Evolution Tracker")

    # Show how each player's config has evolved from original
    evolved = st.session_state.get("configs_evolved", False)
    if evolved:
        evolution_data = []
        for pid in ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]:
            original = PLAYERS_CONFIG.get(pid, {})
            current = st.session_state.player_configs.get(pid, {})

            orig_inds = set(original.get("weights", {}).keys())
            curr_inds = set(current.get("weights", {}).keys())

            added = curr_inds - orig_inds
            removed = orig_inds - curr_inds
            kept = orig_inds & curr_inds

            # Calculate weight changes for kept indicators
            weight_changes = 0
            for ind in kept:
                orig_w = original.get("weights", {}).get(ind, 0)
                curr_w = current.get("weights", {}).get(ind, 0)
                if abs(curr_w - orig_w) > 0.01:
                    weight_changes += 1

            evolution_data.append({
                "Player": PLAYER_LABELS.get(pid, pid),
                "Original Inds": len(orig_inds),
                "Current Inds": len(curr_inds),
                "Added": len(added),
                "Removed": len(removed),
                "Weights Changed": weight_changes,
                "Entry Δ": f"{current.get('entry_threshold', 0.30) - original.get('entry_threshold', 0.30):+.2f}",
                "Current Entry": f"{current.get('entry_threshold', 0.30):.2f}",
            })

        st.dataframe(pd.DataFrame(evolution_data), use_container_width=True)

        # Detailed view per player
        with st.expander("View Detailed Evolution (Indicators Added/Removed)"):
            for pid in ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]:
                original = PLAYERS_CONFIG.get(pid, {})
                current = st.session_state.player_configs.get(pid, {})

                orig_inds = set(original.get("weights", {}).keys())
                curr_inds = set(current.get("weights", {}).keys())

                added = curr_inds - orig_inds
                removed = orig_inds - curr_inds

                st.markdown(f"**{PLAYER_LABELS.get(pid, pid)}:**")
                if added:
                    st.markdown(f"  ➕ Added: {', '.join(sorted(added))}")
                if removed:
                    st.markdown(f"  ➖ Removed: {', '.join(sorted(removed))}")
                if not added and not removed:
                    st.markdown(f"  ∅ No indicator changes (only weight adjustments)")
    else:
        st.info("Configs haven't evolved yet. Run backtests to see how the AI coach optimizes each player's strategy.")

    # Run history table
    st.subheader("Run History")
    history_data = []
    for r in results:
        row = {
            "Run": r.get("run_number", "?"),
            "Time": r.get("run_time", "")[:19],
            "Team P&L": f"₹{r.get('team_total_pnl', 0):+,.0f}",
            "Trades": r.get("team_total_trades", 0),
            "Regime": r.get("market_regime", "?"),
        }
        for pid in ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]:
            pdata = r.get("players", {}).get(pid, {})
            row[PLAYER_LABELS.get(pid, pid)[:3]] = f"₹{pdata.get('pnl', 0):+,.0f}"
        history_data.append(row)

    st.dataframe(pd.DataFrame(history_data), use_container_width=True, height=250)

    # Export
    st.subheader("Export")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download All Results (JSON)",
            data=json.dumps(results, indent=2, default=str),
            file_name=f"5player_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    with col_dl2:
        if history_data:
            st.download_button(
                "Download Summary (CSV)",
                data=pd.DataFrame(history_data).to_csv(index=False),
                file_name=f"5player_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# Auto-run when Streamlit discovers this as a page
def _auto_run():
    try:
        st.set_page_config(page_title="Continuous 5-Player Backtest", page_icon="🔄", layout="wide")
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
