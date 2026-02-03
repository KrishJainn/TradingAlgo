"""
Live Paper Trading Dashboard — Monitor 5-Player Coach System in Real-Time.

Displays:
- Team and per-player equity curves
- Current positions across all 5 players
- Today's trades with P&L
- Market regime detection
- Coach session history with weight patches
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

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
from aqtis.memory.memory_layer import MemoryLayer

# Path to live state file
LIVE_STATE_FILE = Path("live_5player_state.json")

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


# ───────────────────────────────────────────────────────────────────────
# DATA LOADING
# ───────────────────────────────────────────────────────────────────────
def load_live_state() -> Optional[Dict]:
    """Load current live trading state from file."""
    if not LIVE_STATE_FILE.exists():
        return None

    try:
        with open(LIVE_STATE_FILE) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load state: {e}")
        return None


def get_trader_instance():
    """Get or create LiveFivePlayerTrader instance (no caching - always fresh)."""
    try:
        from live_5player_trader import LiveFivePlayerTrader
        return LiveFivePlayerTrader()
    except ImportError:
        return None


def fetch_current_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices directly from yfinance."""
    import yfinance as yf
    prices = {}
    if not symbols:
        return prices

    try:
        data = yf.download(symbols, period="1d", interval="15m", progress=False)
        if data.empty:
            return prices

        for sym in symbols:
            try:
                if len(symbols) == 1:
                    # Single symbol - no MultiIndex
                    close_val = data["Close"].iloc[-1]
                elif isinstance(data.columns, pd.MultiIndex):
                    # Multi-symbol - MultiIndex columns
                    close_val = data[("Close", sym)].iloc[-1]
                else:
                    close_val = data["Close"].iloc[-1]

                # Handle both scalar and Series
                if hasattr(close_val, 'iloc'):
                    close_val = close_val.iloc[0]
                prices[sym] = float(close_val)
            except Exception:
                pass
    except Exception as e:
        st.warning(f"Price fetch error: {e}")

    return prices


def refresh_prices_from_yfinance() -> bool:
    """Refresh current prices from yfinance and update state file."""
    trader = get_trader_instance()
    if trader:
        return trader.refresh_prices()
    return False


# ───────────────────────────────────────────────────────────────────────
# CHARTS
# ───────────────────────────────────────────────────────────────────────
def create_equity_curves(equity_history: List[Dict]) -> go.Figure:
    """Create multi-player equity curve chart."""
    if not equity_history:
        fig = go.Figure()
        fig.add_annotation(text="No equity history yet", showarrow=False)
        return fig

    fig = go.Figure()

    # Add trace for each player
    for player_id, color in PLAYER_COLORS.items():
        dates = []
        equities = []

        for record in equity_history:
            if player_id in record:
                dates.append(record["date"])
                equities.append(record[player_id])

        if dates:
            fig.add_trace(go.Scatter(
                x=dates,
                y=equities,
                name=f"{player_id} ({PLAYER_LABELS.get(player_id, '')})",
                line=dict(color=color, width=2),
                mode="lines",
            ))

    # Add team total
    dates = []
    team_equity = []
    for record in equity_history:
        if "team" in record:
            dates.append(record["date"])
            team_equity.append(record["team"])

    if dates:
        fig.add_trace(go.Scatter(
            x=dates,
            y=team_equity,
            name="Team Total",
            line=dict(color="white", width=3, dash="dot"),
            mode="lines",
        ))

    fig.update_layout(
        title="Player Equity Curves",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        paper_bgcolor=AQTIS_COLORS["background"],
        plot_bgcolor=AQTIS_COLORS["card_bg"],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
    )

    return fig


def create_pnl_bar_chart(players_data: Dict) -> go.Figure:
    """Create day P&L bar chart for all players."""
    player_ids = list(players_data.keys())
    pnls = [players_data[p].get("day_pnl", 0) for p in player_ids]
    colors = [PLAYER_COLORS.get(p, "#888") for p in player_ids]
    labels = [f"{p}<br>{PLAYER_LABELS.get(p, '')}" for p in player_ids]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=pnls,
        marker_color=[AQTIS_COLORS["green"] if p > 0 else AQTIS_COLORS["red"] for p in pnls],
        text=[f"${p:+,.0f}" for p in pnls],
        textposition="outside",
    ))

    fig.update_layout(
        title="Today's P&L by Player",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        paper_bgcolor=AQTIS_COLORS["background"],
        plot_bgcolor=AQTIS_COLORS["card_bg"],
        height=300,
        showlegend=False,
    )

    return fig


# ───────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ───────────────────────────────────────────────────────────────────────
def render_live_paper_trading(memory: MemoryLayer = None):
    """Render the Live Paper Trading dashboard page."""

    st.header("Live 5-Player Paper Trading")

    # Load state first
    state = load_live_state()

    # Fetch fresh prices directly from yfinance and update state
    if state and state.get("players"):
        # Collect all symbols from positions
        all_symbols = set()
        for pdata in state["players"].values():
            for pos in pdata.get("positions", []):
                sym = pos.get("symbol")
                if sym:
                    all_symbols.add(sym)

        if all_symbols:
            # Fetch live prices
            current_prices = fetch_current_prices(list(all_symbols))

            # Update positions with fresh prices
            if current_prices:
                for pdata in state["players"].values():
                    for pos in pdata.get("positions", []):
                        sym = pos.get("symbol")
                        if sym in current_prices:
                            new_price = current_prices[sym]
                            pos["current_price"] = new_price

                            # Recalculate unrealized P&L
                            entry = pos.get("entry_price", 0)
                            qty = pos.get("quantity", 0)
                            direction = pos.get("direction", "LONG")
                            if direction == "LONG":
                                pos["unrealized_pnl"] = (new_price - entry) * qty
                            else:
                                pos["unrealized_pnl"] = (entry - new_price) * qty

                # Update timestamp to show fresh data
                state["last_updated"] = datetime.now().strftime("%H:%M:%S")
                st.success(f"Live prices updated for {len(current_prices)} symbols")

    if not state:
        st.warning("No live trading state found. Start the live trader first.")

        st.markdown("""
        ### Getting Started

        1. **Start the live trader** from command line:
        ```bash
        python live_5player_trader.py --continuous
        ```

        2. Or run a **single scan**:
        ```bash
        python live_5player_trader.py --scan
        ```

        3. The dashboard will automatically update when trading state is available.
        """)

        # Show manual controls
        st.sidebar.subheader("Manual Controls")

        if st.sidebar.button("Run Single Scan"):
            trader = get_trader_instance()
            if trader:
                with st.spinner("Running market scan..."):
                    result = trader.run_scan()
                    st.success("Scan complete!")
                    st.rerun()
            else:
                st.error("Could not initialize trader")

        return

    # ── STATUS BAR ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        market_status = state.get("market_status", "Unknown")
        status_color = "🟢" if market_status == "OPEN" else "🔴"
        st.metric("Market Status", f"{status_color} {market_status}")

    with col2:
        regime = state.get("market_regime", "unknown")
        st.metric("Market Regime", regime.upper())

    with col3:
        last_scan = state.get("last_scan_time", "Never")
        if last_scan:
            try:
                dt = datetime.fromisoformat(last_scan.replace("Z", "+00:00"))
                last_scan = dt.strftime("%H:%M:%S")
            except:
                pass
        st.metric("Last Scan", last_scan)

    with col4:
        updated = state.get("last_updated", "")
        if updated:
            try:
                dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                updated = dt.strftime("%H:%M")
            except:
                pass
        st.metric("Updated", updated)

    # ── TEAM METRICS ──
    st.markdown("---")

    team_equity = state.get("team_equity", 500000)
    team_day_pnl = state.get("team_day_pnl", 0)
    total_positions = state.get("total_positions", 0)
    total_trades_today = state.get("total_trades_today", 0)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pnl_pct = (team_day_pnl / 500000 * 100) if team_equity > 0 else 0
        st.metric(
            "Team Equity",
            f"${team_equity:,.0f}",
            delta=f"${team_day_pnl:+,.0f} ({pnl_pct:+.2f}%)" if team_day_pnl != 0 else None,
        )

    with col2:
        st.metric("Day P&L", f"${team_day_pnl:+,.0f}")

    with col3:
        st.metric("Open Positions", total_positions)

    with col4:
        st.metric("Trades Today", total_trades_today)

    # ── EQUITY CURVES ──
    st.markdown("---")

    equity_history = state.get("equity_history", [])
    if equity_history:
        fig = create_equity_curves(equity_history)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Equity history will appear after the first day of trading")

    # ── PLAYER STATUS TABLE ──
    st.subheader("Player Status")

    players_data = state.get("players", {})

    if players_data:
        # Build DataFrame
        rows = []
        for pid, pdata in players_data.items():
            rows.append({
                "Player": pid,
                "Label": pdata.get("label", ""),
                "Equity": f"${pdata.get('equity', 0):,.0f}",
                "Day P&L": f"${pdata.get('day_pnl', 0):+,.0f}",
                "Positions": pdata.get("num_positions", 0),
                "Win Rate": f"{pdata.get('win_rate', 0):.0%}",
                "Total Trades": pdata.get("total_trades", 0),
                "Version": pdata.get("strategy_version", "v1.0"),
            })

        df = pd.DataFrame(rows)

        # Style the dataframe
        def highlight_pnl(val):
            if "$-" in str(val):
                return f"color: {AQTIS_COLORS['red']}"
            elif "$+" in str(val) or (str(val).startswith("$") and "-" not in str(val)):
                return f"color: {AQTIS_COLORS['green']}"
            return ""

        styled = df.style.map(highlight_pnl, subset=["Day P&L"])
        st.dataframe(styled, use_container_width=True, hide_index=True)

        # Day P&L chart
        fig = create_pnl_bar_chart(players_data)
        st.plotly_chart(fig, use_container_width=True)

    # ── OPEN POSITIONS ──
    with st.expander(f"Open Positions ({total_positions})", expanded=total_positions > 0):
        all_positions = []
        for pid, pdata in players_data.items():
            for pos in pdata.get("positions", []):
                pos["Player"] = pid
                all_positions.append(pos)

        if all_positions:
            pos_df = pd.DataFrame(all_positions)

            # Reorder columns
            cols = ["Player", "symbol", "direction", "entry_price", "current_price",
                    "unrealized_pnl", "bars_held"]
            available_cols = [c for c in cols if c in pos_df.columns]
            pos_df = pos_df[available_cols]

            # Rename columns
            pos_df.columns = ["Player", "Symbol", "Direction", "Entry", "Current", "Unrealized P&L", "Bars Held"]

            # Format currency
            pos_df["Entry"] = pos_df["Entry"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            pos_df["Current"] = pos_df["Current"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
            pos_df["Unrealized P&L"] = pos_df["Unrealized P&L"].apply(
                lambda x: f"${x:+,.2f}" if pd.notna(x) else ""
            )

            st.dataframe(pos_df, use_container_width=True, hide_index=True)
        else:
            st.info("No open positions")

    # ── TODAY'S TRADES ──
    with st.expander(f"Today's Trades ({total_trades_today})", expanded=total_trades_today > 0):
        all_trades = []
        for pid, pdata in players_data.items():
            for trade in pdata.get("todays_trades", []):
                trade["Player"] = pid
                all_trades.append(trade)

        if all_trades:
            trades_df = pd.DataFrame(all_trades)

            # Select columns
            cols = ["Player", "symbol", "side", "entry_price", "exit_price", "pnl", "exit_reason"]
            available_cols = [c for c in cols if c in trades_df.columns]
            trades_df = trades_df[available_cols]

            # Rename
            col_names = {
                "symbol": "Symbol", "side": "Direction",
                "entry_price": "Entry", "exit_price": "Exit",
                "pnl": "P&L", "exit_reason": "Exit Reason"
            }
            trades_df = trades_df.rename(columns=col_names)

            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades today")

    # ── SIDEBAR CONTROLS ──
    st.sidebar.subheader("Live Trading Controls")

    if st.sidebar.button("🔄 Refresh Page"):
        st.rerun()

    if st.sidebar.button("💹 Update Prices"):
        with st.spinner("Fetching latest prices from yfinance..."):
            if refresh_prices_from_yfinance():
                st.sidebar.success("Prices updated!")
                st.rerun()
            else:
                st.sidebar.error("Failed to refresh prices")

    if st.sidebar.button("📊 Run Scan"):
        trader = get_trader_instance()
        if trader:
            with st.spinner("Running market scan..."):
                result = trader.run_scan()
                st.sidebar.success("Scan complete!")
                st.rerun()

    if st.sidebar.button("🎓 Run Coach"):
        trader = get_trader_instance()
        if trader:
            with st.spinner("Running coach analysis..."):
                result = trader.run_eod_coach()
                st.sidebar.success("Coach analysis complete!")
                st.rerun()

    # Auto-refresh option
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

    # ── PLAYER DETAILS ──
    st.sidebar.subheader("Player Details")

    selected_player = st.sidebar.selectbox(
        "Select Player",
        list(players_data.keys()) if players_data else ["No players"],
    )

    if selected_player and selected_player in players_data:
        pdata = players_data[selected_player]

        st.sidebar.markdown(f"**{pdata.get('label', '')}**")
        st.sidebar.write(f"Entry Threshold: {pdata.get('entry_threshold', 0):.2f}")
        st.sidebar.write(f"Exit Threshold: {pdata.get('exit_threshold', 0):.2f}")

        # Show top weights
        weights = pdata.get("weights", {})
        if weights:
            st.sidebar.markdown("**Top Indicators:**")
            sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for ind, w in sorted_weights:
                st.sidebar.write(f"  {ind}: {w:.2f}")


# ───────────────────────────────────────────────────────────────────────
# STANDALONE EXECUTION
# ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    st.set_page_config(
        page_title="Live Paper Trading - AQTIS",
        page_icon="📈",
        layout="wide",
    )
    render_live_paper_trading()
