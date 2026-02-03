#!/usr/bin/env python3
"""
Standalone 5-Player Backtest Analysis Dashboard.

Run with: streamlit run backtest_dashboard.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from copy import deepcopy

import numpy as np
import pandas as pd

_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    print("Required: pip install streamlit plotly pandas numpy")
    sys.exit(1)

# Theme colors
COLORS = {
    "background": "#0e1117",
    "card_bg": "#1a1f2e",
    "green": "#00d26a",
    "red": "#ff4757",
    "blue": "#4dabf7",
    "yellow": "#ffd43b",
}

# Player colors
PLAYER_COLORS = {
    "PLAYER_1": "#ff6b6b",
    "PLAYER_2": "#4dabf7",
    "PLAYER_3": "#51cf66",
    "PLAYER_4": "#ffd43b",
    "PLAYER_5": "#da77f2",
}

PLAYER_LABELS = {
    "PLAYER_1": "Aggressive",
    "PLAYER_2": "Conservative",
    "PLAYER_3": "Balanced",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "Momentum",
}

# Default player configs
PLAYERS_CONFIG = {
    "PLAYER_1": {
        "label": "Aggressive",
        "weights": {
            "RSI_7": 0.90, "STOCH_5_3": 0.85, "TSI_13_25": 0.75,
            "CMO_14": 0.70, "WILLR_14": 0.65, "OBV": 0.60,
            "MFI_14": 0.55, "ADX_14": 0.50, "EMA_9": 0.45,
            "NATR_14": 0.40,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_2": {
        "label": "Conservative",
        "weights": {
            "ADX_14": 0.90, "SUPERTREND_7_3": 0.85, "EMA_20": 0.80,
            "AROON_14": 0.70, "CMF_20": 0.65, "RSI_14": 0.60,
            "BBANDS_20_2": 0.55, "OBV": 0.50, "VWMA_10": 0.45,
            "HMA_9": 0.40, "ATR_14": 0.35,
        },
        "entry_threshold": 0.35,
        "exit_threshold": -0.15,
        "min_hold_bars": 5,
    },
    "PLAYER_3": {
        "label": "Balanced",
        "weights": {
            "RSI_14": 0.85, "BBANDS_20_2": 0.80, "STOCH_14_3": 0.75,
            "CMF_20": 0.65, "ZSCORE_20": 0.60, "MFI_20": 0.55,
            "EMA_9": 0.50, "ADX_14": 0.45, "ATR_14": 0.40,
            "TEMA_20": 0.35,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_4": {
        "label": "VolBreakout",
        "weights": {
            "NATR_14": 0.95, "KELTNER_20_1.5": 0.90, "ADX_14": 0.85,
            "BBANDS_20_2": 0.75, "ATR_14": 0.70, "CCI_20": 0.60,
            "RSI_7": 0.55, "OBV": 0.50, "CMF_20": 0.45,
            "WILLR_14": 0.40,
        },
        "entry_threshold": 0.25,
        "exit_threshold": -0.08,
        "min_hold_bars": 3,
    },
    "PLAYER_5": {
        "label": "Momentum",
        "weights": {
            "RSI_7": 0.95, "TSI_13_25": 0.90, "MACD_12_26_9": 0.85,
            "CMO_14": 0.80, "STOCH_5_3": 0.75, "TRIX_15": 0.65,
            "PPO_12_26_9": 0.60, "ROC_10": 0.55, "MOM_10": 0.50,
            "EMA_9": 0.45,
        },
        "entry_threshold": 0.28,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
}


@dataclass
class PlayerState:
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


def raw_weighted_average(normalized_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Weighted average of normalised indicators."""
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


def coach_optimize(players: Dict[str, PlayerState], day_num: int) -> Dict[str, Dict]:
    """Coach analyzes recent trades and optimizes configs."""
    import random

    new_configs = {}

    for pid, state in players.items():
        config = deepcopy(state.config)
        weights = config["weights"]

        recent_trades = state.trades[-50:] if len(state.trades) > 50 else state.trades

        if not recent_trades:
            new_configs[pid] = config
            continue

        wins = sum(1 for t in recent_trades if t["pnl"] > 0)
        win_rate = wins / len(recent_trades) if recent_trades else 0
        total_pnl = sum(t["pnl"] for t in recent_trades)

        old_thresh = config["entry_threshold"]

        if win_rate < 0.40:
            config["entry_threshold"] = min(0.50, config["entry_threshold"] + 0.03)
            config["min_hold_bars"] = min(8, config["min_hold_bars"] + 1)
        elif win_rate > 0.55:
            config["entry_threshold"] = max(0.20, config["entry_threshold"] - 0.02)

        if total_pnl < 0:
            config["exit_threshold"] = max(-0.20, config["exit_threshold"] - 0.02)

        for k in weights:
            perturbation = random.uniform(-0.05, 0.05)
            weights[k] = max(0.1, min(1.0, weights[k] + perturbation))

        config["weights"] = weights
        new_configs[pid] = config

        state.coach_history.append({
            "day": day_num,
            "win_rate": win_rate,
            "pnl": total_pnl,
            "old_threshold": old_thresh,
            "new_threshold": config["entry_threshold"],
        })

    return new_configs


def simulate_day(
    day_data: Dict[str, pd.DataFrame],
    indicator_data: Dict[str, pd.DataFrame],
    players: Dict[str, PlayerState],
    normalizer,
    day_idx: int,
    bars_per_day: int = 26,
):
    """Simulate one trading day."""
    start_bar = day_idx * bars_per_day
    end_bar = start_bar + bars_per_day

    for pid, state in players.items():
        config = state.config
        weights = config["weights"]
        entry_thresh = config["entry_threshold"]
        exit_thresh = config["exit_threshold"]
        min_hold = config["min_hold_bars"]

        day_pnl = 0.0

        for sym, df in day_data.items():
            if sym not in indicator_data:
                continue

            raw = indicator_data[sym]

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
                norm = normalizer.normalize_all(day_raw[active], price_series=day_df["close"])
                if norm.empty:
                    continue
                si_series = raw_weighted_average(norm, weights).fillna(0.0)
            except:
                continue

            for bar_idx in range(len(day_df)):
                si = si_series.iloc[bar_idx]
                price = day_df.iloc[bar_idx]["close"]
                prev_si = state.prev_si.get(sym, 0.0)

                if sym in state.positions:
                    state.bars_held[sym] = state.bars_held.get(sym, 0) + 1
                    pos = state.positions[sym]
                    held = state.bars_held[sym]

                    should_exit = False
                    if pos["direction"] == "LONG" and held >= min_hold and si < exit_thresh:
                        should_exit = True
                    elif pos["direction"] == "SHORT" and held >= min_hold and si > abs(exit_thresh):
                        should_exit = True

                    if bar_idx == len(day_df) - 1:
                        should_exit = True

                    if should_exit:
                        entry_price = pos["entry_price"]
                        if pos["direction"] == "LONG":
                            pnl = (price - entry_price) / entry_price * 1000
                        else:
                            pnl = (entry_price - price) / entry_price * 1000

                        state.equity += pnl
                        day_pnl += pnl

                        state.trades.append({
                            "symbol": sym,
                            "direction": pos["direction"],
                            "entry_price": entry_price,
                            "exit_price": price,
                            "pnl": pnl,
                            "bars_held": held,
                            "day": day_idx,
                        })

                        del state.positions[sym]
                        state.bars_held.pop(sym, None)

                else:
                    if bar_idx < len(day_df) - 2:
                        if si > entry_thresh and prev_si > entry_thresh * 0.7:
                            state.positions[sym] = {
                                "direction": "LONG",
                                "entry_price": price,
                            }
                            state.bars_held[sym] = 0
                        elif si < -entry_thresh and prev_si < -entry_thresh * 0.7:
                            state.positions[sym] = {
                                "direction": "SHORT",
                                "entry_price": price,
                            }
                            state.bars_held[sym] = 0

                state.prev_si[sym] = si

        state.daily_pnl.append(day_pnl)
        state.equity_curve.append(state.equity)


def run_backtest(data_dir: Path, coach_interval: int = 3, progress_callback=None) -> Dict:
    """Run full backtest with coach optimization."""
    from trading_evolution.indicators.universe import IndicatorUniverse
    from trading_evolution.indicators.calculator import IndicatorCalculator
    from trading_evolution.indicators.normalizer import IndicatorNormalizer

    data = {}
    for f in data_dir.glob("*_60d_15m.csv"):
        sym = f.stem.replace("_60d_15m", "") + ".NS"
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        data[sym] = df

    if not data:
        return {"error": "No data found in backtest_data/"}

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
        except:
            pass

    players = {}
    for pid, config in PLAYERS_CONFIG.items():
        players[pid] = PlayerState(
            player_id=pid,
            config=deepcopy(config),
        )
        players[pid].equity_curve.append(100000.0)

    bars_per_day = 26
    sample_df = list(data.values())[0]
    total_bars = len(sample_df)
    total_days = total_bars // bars_per_day

    coach_sessions = []

    for day in range(total_days):
        simulate_day(data, indicator_data, players, normalizer, day, bars_per_day)

        if progress_callback:
            progress_callback((day + 1) / total_days)

        if (day + 1) % coach_interval == 0 and day < total_days - 1:
            new_configs = coach_optimize(players, day + 1)

            session = {"day": day + 1, "updates": {}}
            for pid, config in new_configs.items():
                session["updates"][pid] = {
                    "win_rate": players[pid].coach_history[-1]["win_rate"] if players[pid].coach_history else 0,
                    "pnl": players[pid].coach_history[-1]["pnl"] if players[pid].coach_history else 0,
                    "threshold": config["entry_threshold"],
                }
                players[pid].config = config
            coach_sessions.append(session)

    results = {
        "total_days": total_days,
        "symbols": len(data),
        "coach_interval": coach_interval,
        "coach_sessions": coach_sessions,
        "players": {},
    }

    for pid, state in players.items():
        trades = len(state.trades)
        wins = sum(1 for t in state.trades if t["pnl"] > 0)
        win_rate = wins / trades if trades > 0 else 0
        pnl = state.equity - 100000

        if state.daily_pnl:
            returns = np.array(state.daily_pnl) / 100000
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        results["players"][pid] = {
            "label": state.config["label"],
            "trades": trades,
            "wins": wins,
            "win_rate": win_rate,
            "pnl": pnl,
            "sharpe": sharpe,
            "final_threshold": state.config["entry_threshold"],
            "equity_curve": state.equity_curve,
            "daily_pnl": state.daily_pnl,
            "coach_history": state.coach_history,
            "final_config": state.config,
        }

    total_pnl = sum(p["pnl"] for p in results["players"].values())
    results["total_pnl"] = total_pnl

    return results


def create_equity_curves_chart(results: Dict) -> go.Figure:
    """Create equity curves for all players."""
    fig = go.Figure()

    for pid, pdata in results["players"].items():
        equity = pdata["equity_curve"]
        days = list(range(len(equity)))

        fig.add_trace(go.Scatter(
            x=days,
            y=equity,
            name=f"{pid} ({pdata['label']})",
            line=dict(color=PLAYER_COLORS.get(pid, "#888"), width=2),
            mode="lines",
        ))

    team_equity = []
    max_len = max(len(p["equity_curve"]) for p in results["players"].values())
    for i in range(max_len):
        total = sum(
            p["equity_curve"][i] if i < len(p["equity_curve"]) else p["equity_curve"][-1]
            for p in results["players"].values()
        )
        team_equity.append(total)

    fig.add_trace(go.Scatter(
        x=list(range(len(team_equity))),
        y=team_equity,
        name="Team Total",
        line=dict(color="white", width=3, dash="dot"),
        mode="lines",
    ))

    fig.update_layout(
        title="Player Equity Curves",
        xaxis_title="Day",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def create_pnl_comparison_chart(results: Dict) -> go.Figure:
    """Create P&L comparison bar chart."""
    player_ids = list(results["players"].keys())
    pnls = [results["players"][p]["pnl"] for p in player_ids]
    labels = [f"{p}<br>{PLAYER_LABELS.get(p, '')}" for p in player_ids]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=pnls,
        marker_color=[COLORS["green"] if p > 0 else COLORS["red"] for p in pnls],
        text=[f"${p:+,.0f}" for p in pnls],
        textposition="outside",
    ))

    fig.update_layout(
        title="Total P&L by Player",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        height=300,
        showlegend=False,
    )

    return fig


def create_sharpe_chart(results: Dict) -> go.Figure:
    """Create Sharpe ratio comparison chart."""
    player_ids = list(results["players"].keys())
    sharpes = [results["players"][p]["sharpe"] for p in player_ids]
    labels = [f"{p}<br>{PLAYER_LABELS.get(p, '')}" for p in player_ids]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=sharpes,
        marker_color=[COLORS["blue"] if s > 0 else COLORS["red"] for s in sharpes],
        text=[f"{s:.2f}" for s in sharpes],
        textposition="outside",
    ))

    fig.update_layout(
        title="Sharpe Ratio by Player",
        yaxis_title="Sharpe Ratio",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        height=300,
        showlegend=False,
    )

    return fig


def create_coach_threshold_chart(results: Dict) -> go.Figure:
    """Create threshold evolution chart from coach sessions."""
    fig = go.Figure()

    for pid, pdata in results["players"].items():
        history = pdata.get("coach_history", [])
        if history:
            days = [h["day"] for h in history]
            thresholds = [h["new_threshold"] for h in history]

            fig.add_trace(go.Scatter(
                x=days,
                y=thresholds,
                name=f"{pid} ({pdata['label']})",
                line=dict(color=PLAYER_COLORS.get(pid, "#888"), width=2),
                mode="lines+markers",
            ))

    fig.update_layout(
        title="Entry Threshold Evolution (Coach Adjustments)",
        xaxis_title="Day",
        yaxis_title="Entry Threshold",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        height=350,
    )

    return fig


def main():
    st.set_page_config(
        page_title="5-Player Backtest Analysis",
        page_icon="📊",
        layout="wide",
    )

    st.title("5-Player Backtest Analysis")

    data_dir = Path("backtest_data")

    # Check for data
    if not data_dir.exists() or not list(data_dir.glob("*_60d_15m.csv")):
        st.warning("No backtest data found. Download 60-day 15m data first.")

        if st.button("Download Data (30 Nifty Stocks)"):
            with st.spinner("Downloading 60-day 15m data..."):
                try:
                    import yfinance as yf

                    symbols = [
                        "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
                        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
                        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
                        "SUNPHARMA", "ULTRACEMCO", "BAJFINANCE", "WIPRO", "HCLTECH",
                        "NESTLEIND", "ONGC", "NTPC", "POWERGRID", "TECHM",
                        "BAJAJFINSV", "INDUSINDBK", "JSWSTEEL", "M&M", "ADANIPORTS",
                    ]

                    data_dir.mkdir(exist_ok=True)

                    for sym in symbols:
                        ticker = yf.Ticker(f"{sym}.NS")
                        df = ticker.history(period="60d", interval="15m")
                        if not df.empty:
                            df.to_csv(data_dir / f"{sym}_60d_15m.csv")

                    st.success(f"Downloaded data for {len(symbols)} symbols!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Download failed: {e}")
        return

    csv_files = list(data_dir.glob("*_60d_15m.csv"))
    st.info(f"Found {len(csv_files)} symbols in backtest_data/")

    # Sidebar controls
    st.sidebar.subheader("Backtest Settings")

    coach_interval = st.sidebar.slider(
        "Coach Interval (days)",
        min_value=1,
        max_value=10,
        value=3,
    )

    if "backtest_results" not in st.session_state:
        st.session_state.backtest_results = None

    if st.sidebar.button("Run Backtest", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(pct):
            progress_bar.progress(pct)
            status_text.text(f"Simulating... {pct*100:.0f}%")

        with st.spinner("Running backtest..."):
            status_text.text("Loading data and calculating indicators...")
            results = run_backtest(data_dir, coach_interval, update_progress)
            st.session_state.backtest_results = results

        progress_bar.empty()
        status_text.empty()
        st.success("Backtest complete!")
        st.rerun()

    results = st.session_state.backtest_results

    if not results:
        st.markdown("""
        ### How to Use

        1. **Run Backtest**: Click the button in the sidebar
        2. **Adjust Coach Interval**: Change how often the coach optimizes
        3. **Analyze Results**: View equity curves, P&L, Sharpe ratios
        """)
        return

    if "error" in results:
        st.error(results["error"])
        return

    # Summary metrics
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Days", results["total_days"])

    with col2:
        st.metric("Symbols", results["symbols"])

    with col3:
        st.metric("Coach Sessions", len(results["coach_sessions"]))

    with col4:
        total_pnl = results["total_pnl"]
        st.metric("Total P&L", f"${total_pnl:+,.0f}")

    # Equity curves
    st.markdown("---")
    fig = create_equity_curves_chart(results)
    st.plotly_chart(fig, use_container_width=True)

    # P&L and Sharpe
    col1, col2 = st.columns(2)

    with col1:
        fig = create_pnl_comparison_chart(results)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = create_sharpe_chart(results)
        st.plotly_chart(fig, use_container_width=True)

    # Player table
    st.subheader("Player Performance Summary")

    rows = []
    for pid, pdata in results["players"].items():
        rows.append({
            "Player": pid,
            "Label": pdata["label"],
            "Trades": pdata["trades"],
            "Win Rate": f"{pdata['win_rate']:.0%}",
            "P&L": f"${pdata['pnl']:+,.0f}",
            "Sharpe": f"{pdata['sharpe']:.2f}",
            "Final Threshold": f"{pdata['final_threshold']:.2f}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Coach history
    st.markdown("---")
    st.subheader("Coach Optimization History")

    fig = create_coach_threshold_chart(results)
    st.plotly_chart(fig, use_container_width=True)

    # Player details
    st.markdown("---")
    st.subheader("Player Details")

    selected_player = st.selectbox(
        "Select Player",
        list(results["players"].keys()),
        format_func=lambda x: f"{x} ({PLAYER_LABELS.get(x, '')})"
    )

    if selected_player:
        pdata = results["players"][selected_player]

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Trades", pdata["trades"])
        with col2:
            st.metric("Win Rate", f"{pdata['win_rate']:.1%}")
        with col3:
            st.metric("Sharpe Ratio", f"{pdata['sharpe']:.2f}")

        if pdata["daily_pnl"]:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(range(len(pdata["daily_pnl"]))),
                y=pdata["daily_pnl"],
                marker_color=[COLORS["green"] if p > 0 else COLORS["red"] for p in pdata["daily_pnl"]],
            ))
            fig.update_layout(
                title=f"{selected_player} Daily P&L",
                xaxis_title="Day",
                yaxis_title="P&L ($)",
                template="plotly_dark",
                paper_bgcolor=COLORS["background"],
                plot_bgcolor=COLORS["card_bg"],
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Final Configuration"):
            st.json(pdata["final_config"])

    # Save config button
    st.sidebar.markdown("---")
    if st.sidebar.button("Save Optimized Configs"):
        output = {}
        for pid, pdata in results["players"].items():
            output[pid] = pdata["final_config"]

        with open("optimized_player_configs.json", "w") as f:
            json.dump(output, f, indent=2)

        st.sidebar.success("Saved to optimized_player_configs.json")


if __name__ == "__main__":
    main()
