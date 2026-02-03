#!/usr/bin/env python3
"""
Standalone Live 5-Player Paper Trading Dashboard.

Run with: streamlit run live_dashboard.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    print("Required: pip install streamlit plotly pandas")
    sys.exit(1)

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

# Theme colors
COLORS = {
    "background": "#0e1117",
    "card_bg": "#1a1f2e",
    "green": "#00d26a",
    "red": "#ff4757",
    "blue": "#4dabf7",
    "yellow": "#ffd43b",
}


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
    """Get LiveFivePlayerTrader instance."""
    try:
        from live_5player_trader import LiveFivePlayerTrader
        return LiveFivePlayerTrader()
    except Exception as e:
        st.error(f"Could not initialize trader: {e}")
        return None


def create_equity_curves(equity_history: List[Dict]) -> go.Figure:
    """Create multi-player equity curve chart."""
    fig = go.Figure()

    if not equity_history:
        fig.add_annotation(text="No equity history yet", showarrow=False)
        return fig

    for player_id, color in PLAYER_COLORS.items():
        dates = []
        equities = []
        for record in equity_history:
            if player_id in record:
                dates.append(record.get("date", ""))
                equities.append(record[player_id])
        if dates:
            fig.add_trace(go.Scatter(
                x=dates, y=equities,
                name=f"{player_id} ({PLAYER_LABELS.get(player_id, '')})",
                line=dict(color=color, width=2),
                mode="lines",
            ))

    # Team total
    dates = [r.get("date", "") for r in equity_history if "team" in r]
    team_equity = [r["team"] for r in equity_history if "team" in r]
    if dates:
        fig.add_trace(go.Scatter(
            x=dates, y=team_equity,
            name="Team Total",
            line=dict(color="white", width=3, dash="dot"),
            mode="lines",
        ))

    fig.update_layout(
        title="Player Equity Curves",
        xaxis_title="Date",
        yaxis_title="Equity ($)",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        height=400,
    )
    return fig


def create_pnl_bar_chart(players_data: Dict) -> go.Figure:
    """Create day P&L bar chart."""
    player_ids = list(players_data.keys())
    pnls = [players_data[p].get("day_pnl", 0) for p in player_ids]
    labels = [f"{p}<br>{PLAYER_LABELS.get(p, '')}" for p in player_ids]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=pnls,
        marker_color=[COLORS["green"] if p > 0 else COLORS["red"] for p in pnls],
        text=[f"${p:+,.0f}" for p in pnls],
        textposition="outside",
    ))
    fig.update_layout(
        title="Today's P&L by Player",
        yaxis_title="P&L ($)",
        template="plotly_dark",
        paper_bgcolor=COLORS["background"],
        plot_bgcolor=COLORS["card_bg"],
        height=300,
        showlegend=False,
    )
    return fig


def main():
    st.set_page_config(
        page_title="Live 5-Player Paper Trading",
        page_icon="📈",
        layout="wide",
    )

    st.title("Live 5-Player Paper Trading")

    # Load state
    state = load_live_state()

    if not state:
        st.warning("No live trading state found. Run a scan first.")

        st.markdown("""
        ### Getting Started

        Click **Run Single Scan** in the sidebar to fetch market data and generate trading signals.

        Or start from command line:
        ```bash
        python live_5player_trader.py --scan
        ```
        """)

        # Sidebar controls
        st.sidebar.subheader("Controls")
        if st.sidebar.button("Run Single Scan", type="primary"):
            trader = get_trader_instance()
            if trader:
                with st.spinner("Running market scan..."):
                    result = trader.run_scan()
                    st.success("Scan complete!")
                    st.rerun()
        return

    # ── STATUS BAR ──
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        market_status = state.get("market_status", "Unknown")
        status_icon = "🟢" if market_status == "OPEN" else "🔴"
        st.metric("Market Status", f"{status_icon} {market_status}")

    with col2:
        regime = state.get("market_regime", "unknown")
        st.metric("Market Regime", regime.upper())

    with col3:
        last_scan = state.get("last_scan_time", "Never")
        if last_scan and last_scan != "Never":
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
        st.info("Equity history will appear after trading activity")

    # ── PLAYER STATUS TABLE ──
    st.subheader("Player Status")

    players_data = state.get("players", {})

    if players_data:
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
        st.dataframe(df, use_container_width=True, hide_index=True)

        # P&L chart
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
            st.dataframe(trades_df, use_container_width=True, hide_index=True)
        else:
            st.info("No trades today")

    # ── SIDEBAR CONTROLS ──
    st.sidebar.subheader("Controls")

    if st.sidebar.button("🔄 Refresh"):
        st.rerun()

    if st.sidebar.button("📊 Run Scan", type="primary"):
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
                st.sidebar.success("Coach complete!")
                st.rerun()

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)")
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()

    # Player details
    st.sidebar.markdown("---")
    st.sidebar.subheader("Player Details")

    selected_player = st.sidebar.selectbox(
        "Select Player",
        list(players_data.keys()) if players_data else ["No players"],
    )

    if selected_player and selected_player in players_data:
        pdata = players_data[selected_player]
        st.sidebar.markdown(f"**{pdata.get('label', '')}**")
        st.sidebar.write(f"Entry: {pdata.get('entry_threshold', 0):.2f}")
        st.sidebar.write(f"Exit: {pdata.get('exit_threshold', 0):.2f}")

        weights = pdata.get("weights", {})
        if weights:
            st.sidebar.markdown("**Top Indicators:**")
            sorted_w = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            for ind, w in sorted_w:
                st.sidebar.write(f"  {ind}: {w:.2f}")


if __name__ == "__main__":
    main()
