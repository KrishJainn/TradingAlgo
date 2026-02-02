"""
AQTIS Streamlit Dashboard.

Visual monitoring and analysis interface for the AQTIS system.

Run with: streamlit run aqtis/dashboard/app.py
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Required package not installed: {e}")
    print("Install with: pip install streamlit plotly pandas numpy")
    sys.exit(1)

from aqtis.config.settings import load_config
from aqtis.memory.memory_layer import MemoryLayer


@st.cache_resource
def get_memory():
    """Initialize memory layer (cached across reruns)."""
    config = load_config()
    return MemoryLayer(
        db_path=str(config.system.db_path),
        vector_path=str(config.system.vector_db_path),
    )


def main():
    st.set_page_config(
        page_title="AQTIS Dashboard",
        page_icon="📊",
        layout="wide",
    )

    st.title("AQTIS - Adaptive Quantitative Trading Intelligence")

    memory = get_memory()

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "Overview",
            "Strategy Performance",
            "Prediction Analysis",
            "Risk Monitor",
            "Research",
            "Memory Explorer",
        ],
    )

    if page == "Overview":
        render_overview(memory)
    elif page == "Strategy Performance":
        render_strategies(memory)
    elif page == "Prediction Analysis":
        render_predictions(memory)
    elif page == "Risk Monitor":
        render_risk(memory)
    elif page == "Research":
        render_research(memory)
    elif page == "Memory Explorer":
        render_memory(memory)


def render_overview(memory: MemoryLayer):
    """Overview dashboard page."""
    st.header("System Overview")

    # Key metrics
    stats = memory.get_stats()
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Trades", stats.get("trades", 0))
    col2.metric("Predictions", stats.get("predictions", 0))
    col3.metric("Strategies", stats.get("strategies", 0))
    col4.metric("Market States", stats.get("market_states", 0))
    col5.metric("Risk Events", stats.get("risk_events", 0))

    # Recent trades
    st.subheader("Recent Trades")
    trades = memory.get_trades(limit=20)
    if trades:
        df = pd.DataFrame(trades)
        display_cols = ["timestamp", "asset", "strategy_id", "action", "pnl", "pnl_percent", "market_regime"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

        # Equity curve
        if "pnl" in df.columns:
            df_sorted = df.sort_values("timestamp")
            df_sorted["cumulative_pnl"] = df_sorted["pnl"].fillna(0).cumsum()
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(len(df_sorted))),
                y=df_sorted["cumulative_pnl"],
                mode="lines",
                name="Cumulative P&L",
            ))
            fig.update_layout(title="Cumulative P&L", xaxis_title="Trade #", yaxis_title="P&L")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No trades recorded yet.")

    # Vector store stats
    vector_stats = stats.get("vector_collections", {})
    st.subheader("Knowledge Base")
    vcol1, vcol2 = st.columns(2)
    vcol1.metric("Research Papers", vector_stats.get("trading_research", 0))
    vcol2.metric("Trade Patterns", vector_stats.get("trade_patterns", 0))


def render_strategies(memory: MemoryLayer):
    """Strategy performance page."""
    st.header("Strategy Performance")

    strategies = memory.get_active_strategies()
    if not strategies:
        st.info("No strategies found.")
        return

    # Strategy selector
    strategy_ids = [s["strategy_id"] for s in strategies]
    selected = st.selectbox("Select Strategy", strategy_ids)

    strategy = next((s for s in strategies if s["strategy_id"] == selected), None)
    if not strategy:
        return

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Trades", strategy.get("total_trades", 0))
    col2.metric("Win Rate", f"{(strategy.get('win_rate') or 0):.2%}")
    col3.metric("Sharpe Ratio", f"{(strategy.get('sharpe_ratio') or 0):.2f}")
    col4.metric("Max Drawdown", f"{(strategy.get('max_drawdown') or 0):.2%}")

    # Strategy details
    st.json(strategy.get("parameters") or {})

    # Strategy trades
    st.subheader("Strategy Trades")
    trades = memory.get_trades(strategy_id=selected, limit=50)
    if trades:
        df = pd.DataFrame(trades)
        st.dataframe(df[["timestamp", "asset", "action", "pnl", "pnl_percent", "market_regime"]].dropna(how="all"), use_container_width=True)


def render_predictions(memory: MemoryLayer):
    """Prediction analysis page."""
    st.header("Prediction Analysis")

    # Accuracy over time
    accuracy = memory.get_prediction_accuracy_history(lookback_days=30)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", accuracy.get("total", 0))
    col2.metric("Directional Accuracy", f"{accuracy.get('accuracy', 0):.2%}")
    col3.metric("Avg Return Error", f"{accuracy.get('avg_return_error', 0):.4f}")

    # Recent predictions
    st.subheader("Recent Predictions")
    predictions = memory.get_predictions(limit=30)
    if predictions:
        df = pd.DataFrame(predictions)
        display_cols = [
            "timestamp", "asset", "strategy_id", "predicted_return",
            "predicted_confidence", "actual_return", "direction_correct",
        ]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

        # Calibration chart
        if "predicted_confidence" in df.columns and "direction_correct" in df.columns:
            df_cal = df.dropna(subset=["predicted_confidence", "direction_correct"])
            if not df_cal.empty:
                df_cal["conf_bin"] = pd.cut(df_cal["predicted_confidence"], bins=10)
                cal_data = df_cal.groupby("conf_bin").agg(
                    predicted=("predicted_confidence", "mean"),
                    actual=("direction_correct", "mean"),
                    count=("direction_correct", "count"),
                ).reset_index()

                if not cal_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=cal_data["predicted"],
                        y=cal_data["actual"],
                        mode="markers+lines",
                        name="Actual Win Rate",
                    ))
                    fig.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode="lines",
                        name="Perfect Calibration",
                        line=dict(dash="dash"),
                    ))
                    fig.update_layout(
                        title="Confidence Calibration",
                        xaxis_title="Predicted Confidence",
                        yaxis_title="Actual Win Rate",
                    )
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions recorded yet.")


def render_risk(memory: MemoryLayer):
    """Risk monitor page."""
    st.header("Risk Monitor")

    # Risk events
    events = memory.get_risk_events(days=30)
    if events:
        st.subheader(f"Risk Events (Last 30 Days): {len(events)}")
        for event in events[:10]:
            with st.expander(f"{event.get('event_type', '')} - {event.get('timestamp', '')[:16]}"):
                st.write(f"Reason: {event.get('reason', 'N/A')}")
                if event.get("details"):
                    st.json(json.loads(event["details"]) if isinstance(event["details"], str) else event["details"])
    else:
        st.success("No risk events in the last 30 days.")

    # Daily P&L
    st.subheader("Recent Daily P&L")
    today = datetime.now().strftime("%Y-%m-%d")
    daily_pnl = memory.db.get_daily_pnl(today)
    st.metric("Today's P&L", f"${daily_pnl:,.2f}")


def render_research(memory: MemoryLayer):
    """Research dashboard page."""
    st.header("Research Database")

    vector_stats = memory.vectors.get_stats()
    st.metric("Papers in Database", vector_stats.get("trading_research", 0))

    # Search
    query = st.text_input("Search Research", placeholder="volatility trading strategies...")
    if query:
        results = memory.search_research(query, top_k=10)
        for r in results:
            title = r.get("metadata", {}).get("title", "Untitled")
            relevance = r.get("distance", "N/A")
            with st.expander(f"{title} (distance: {relevance})"):
                st.write(r.get("text", "")[:500])
                st.json(r.get("metadata", {}))


def render_memory(memory: MemoryLayer):
    """Memory explorer page."""
    st.header("Memory Explorer")

    # Search
    query = st.text_input("Search Similar Trades", placeholder="BUY RELIANCE.NS in trending market...")
    if query:
        trades = memory.get_similar_trades({"notes": query, "asset": query}, top_k=10)
        if trades:
            st.write(f"Found {len(trades)} similar trades")
            for t in trades:
                pnl = t.get("pnl_percent", t.get("pnl", "N/A"))
                st.write(
                    f"- {t.get('asset', 'N/A')} | {t.get('action', 'N/A')} | "
                    f"P&L: {pnl} | Regime: {t.get('market_regime', 'N/A')}"
                )
        else:
            st.info("No similar trades found.")

    # Database stats
    st.subheader("Database Statistics")
    stats = memory.get_stats()
    st.json(stats)


if __name__ == "__main__":
    main()
