"""
Interactive Backtest Visualization Dashboard.

Integrated into the AQTIS Streamlit app as the "Backtest Analysis" page.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    import streamlit as st
    import plotly.graph_objects as go
except ImportError:
    raise ImportError("streamlit and plotly are required: pip install streamlit plotly")

from aqtis.dashboard.components.charts import (
    create_candlestick_with_trades,
    create_drawdown_chart,
    create_equity_curve,
    create_regime_breakdown,
    create_returns_histogram,
    create_strategy_comparison,
    create_time_heatmap,
)
from aqtis.dashboard.components.metrics import (
    BacktestMetrics,
    calculate_drawdown_series,
    calculate_metrics,
    calculate_rolling_metrics,
)
from aqtis.dashboard.components.trade_inspector import TradeInspector
from aqtis.dashboard.theme import AQTIS_COLORS

try:
    from data.symbols import NIFTY_50_SYMBOLS
except ImportError:
    from data_cache.symbols import NIFTY_50_SYMBOLS


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _find_result_files() -> list:
    """Scan common directories for backtest result JSON files."""
    search_dirs = [
        Path("aqtis_data"),
        Path("backtest_results"),
        Path("trading_results"),
        Path("paper_trades"),
    ]
    files = []
    for d in search_dirs:
        if d.exists():
            files.extend(d.glob("*.json"))

    # Also pick up simulation_*.json and multi_player_*.json in project root
    root = Path(".")
    files.extend(root.glob("simulation_*.json"))
    files.extend(root.glob("multi_player_*.json"))

    # Sort by modification time (newest first), deduplicate
    files = sorted(set(files), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:50]


def _load_result(path: Path) -> dict:
    """Load a backtest result JSON file. Normalizes various formats to dict."""
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        # Could be: list of trade dicts OR list of run-summary dicts
        if not data:
            return {"trades": []}

        first = data[0] if data else {}
        if isinstance(first, dict):
            # Run-summary format: each item has 'run', 'pnl', 'trades' (int)
            if "run" in first and isinstance(first.get("trades"), (int, float)):
                return {"runs": data, "_format": "run_summaries"}
            # List of actual trade dicts (have entry_price, exit_price, etc.)
            if any(k in first for k in ("entry_price", "entry_time", "symbol", "action")):
                return {"trades": data}
        # Fallback: wrap as runs
        return {"runs": data, "_format": "run_summaries"}

    return data


def _is_run_summary_format(results: dict) -> bool:
    """Check if results are multi-run summary format."""
    return results.get("_format") == "run_summaries" or (
        "runs" in results
        and isinstance(results["runs"], list)
        and results["runs"]
        and isinstance(results["runs"][0].get("trades"), (int, float))
    )


def _run_summaries_to_trades_df(runs: list) -> pd.DataFrame:
    """Convert run-summary dicts into a pseudo-trades DataFrame.

    Each run becomes a 'trade' row with the run's aggregate stats.
    """
    rows = []
    for r in runs:
        rows.append({
            "symbol": f"Run #{r.get('run', '?')}",
            "action": "RUN",
            "pnl": r.get("pnl", 0),
            "pnl_percent": r.get("return_pct", 0),
            "win_rate": r.get("win_rate", 0),
            "sharpe": r.get("sharpe", 0),
            "trades_count": r.get("trades", 0),
            "wins": r.get("wins", 0),
            "losses": r.get("losses", 0),
        })
    return pd.DataFrame(rows)


def _results_to_trades_df(results) -> pd.DataFrame:
    """Convert backtest results to a trades DataFrame.

    Handles: dict with 'trades' list, dict with 'runs', list of trades, etc.
    """
    if isinstance(results, list):
        trades = results
    elif isinstance(results, dict):
        # Run-summary format (multi_run_results.json, 50_runs_results.json)
        if _is_run_summary_format(results):
            return _run_summaries_to_trades_df(results.get("runs", []))

        trades = results.get("trades", [])

        # trades might be an int (count) not a list
        if isinstance(trades, (int, float)):
            return pd.DataFrame()

        if not trades:
            if "iteration_history" in results:
                trades = []
                for it in results["iteration_history"]:
                    if isinstance(it, dict):
                        t = it.get("trades", [])
                        if isinstance(t, list):
                            trades.extend(t)
            elif "runs" in results and isinstance(results["runs"], list):
                trades = []
                for run in results["runs"]:
                    if isinstance(run, dict):
                        t = run.get("trades", [])
                        if isinstance(t, list):
                            trades.extend(t)
    else:
        trades = []

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    for col in ("entry_time", "exit_time", "timestamp"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _results_to_equity(results, initial_capital: float = 100000) -> pd.Series:
    """Build equity curve from trades, runs, or pre-computed curve."""
    if isinstance(results, list):
        results = {"runs": results, "_format": "run_summaries"}

    if not isinstance(results, dict):
        return pd.Series(dtype=float)

    # Pre-computed equity curve
    ec = results.get("equity_curve")
    if ec is not None:
        if isinstance(ec, dict):
            s = pd.Series(ec)
            s.index = pd.to_datetime(s.index, errors="coerce")
            return s
        if isinstance(ec, list):
            if ec and isinstance(ec[0], dict):
                # List of {date, equity, ...} dicts
                eq_df = pd.DataFrame(ec)
                if "equity" in eq_df.columns:
                    s = eq_df["equity"]
                    if "date" in eq_df.columns:
                        s.index = pd.to_datetime(eq_df["date"], errors="coerce")
                    return s
            return pd.Series(ec)

    # Run-summary format: build from cumulative P&L
    if _is_run_summary_format(results):
        runs = results.get("runs", [])
        if runs:
            cum_pnl = []
            running = initial_capital
            for r in runs:
                running += r.get("pnl", 0)
                cum_pnl.append(running)
            return pd.Series(cum_pnl)
        return pd.Series(dtype=float)

    # Build from individual trade P&Ls
    trades_df = _results_to_trades_df(results)
    if trades_df.empty or "pnl" not in trades_df.columns:
        return pd.Series(dtype=float)

    time_col = "exit_time" if "exit_time" in trades_df.columns else "timestamp"
    if time_col not in trades_df.columns:
        trades_sorted = trades_df
    else:
        trades_sorted = trades_df.sort_values(time_col)
    cumulative = trades_sorted["pnl"].fillna(0).cumsum() + initial_capital
    if time_col in trades_sorted.columns:
        cumulative.index = trades_sorted[time_col]
    return cumulative


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------


def render_backtest_dashboard(memory=None):
    """Main Streamlit page function for Backtest Analysis."""

    st.header("Backtest Analysis Dashboard")

    # --- Sidebar controls ---
    st.sidebar.subheader("Backtest Settings")

    source = st.sidebar.radio(
        "Data Source",
        ["Load from File", "Latest in Memory"],
        index=0,
    )

    results = None
    if source == "Load from File":
        result_files = _find_result_files()
        if result_files:
            selected_file = st.sidebar.selectbox(
                "Select Result File",
                result_files,
                format_func=lambda p: p.name,
            )
            if selected_file:
                results = _load_result(selected_file)
        else:
            st.sidebar.warning("No result files found.")
    else:
        # Try loading from memory layer
        if memory:
            trades = memory.get_trades(limit=500)
            if trades:
                results = {"trades": trades}

    if results is None:
        st.info("Select a backtest result file from the sidebar to begin.")
        return

    trades_df = _results_to_trades_df(results)
    equity = _results_to_equity(results)
    metadata = results.get("metadata", {}) if isinstance(results, dict) else {}

    if trades_df.empty:
        st.warning("No trades found in the selected result.")
        return

    # --- Symbol filter ---
    available_symbols = sorted(trades_df["symbol"].dropna().unique()) if "symbol" in trades_df.columns else []
    symbol_filter = st.sidebar.multiselect("Filter Symbols", available_symbols)
    if symbol_filter:
        trades_df = trades_df[trades_df["symbol"].isin(symbol_filter)]

    # --- Tabs ---
    tab_chart, tab_perf, tab_analysis, tab_inspect, tab_evo = st.tabs(
        ["Trade Chart", "Performance", "Analysis", "Trade Inspector", "Evolution"]
    )

    # ===================================================================
    # TAB 1: Trade Chart
    # ===================================================================
    with tab_chart:
        _render_trade_chart_tab(trades_df, metadata)

    # ===================================================================
    # TAB 2: Performance
    # ===================================================================
    with tab_perf:
        _render_performance_tab(trades_df, equity)

    # ===================================================================
    # TAB 3: Analysis
    # ===================================================================
    with tab_analysis:
        _render_analysis_tab(trades_df)

    # ===================================================================
    # TAB 4: Trade Inspector
    # ===================================================================
    with tab_inspect:
        _render_inspector_tab(trades_df, memory)

    # ===================================================================
    # TAB 5: Evolution
    # ===================================================================
    with tab_evo:
        _render_evolution_tab(results)


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_trade_chart_tab(trades_df: pd.DataFrame, metadata: dict):
    """Tab 1: Candlestick chart with trade signals."""
    st.subheader("Price Chart with Trade Signals")

    symbols = sorted(trades_df["symbol"].dropna().unique()) if "symbol" in trades_df.columns else []
    if not symbols:
        st.info("No symbol data available for charting.")
        return

    selected_sym = st.selectbox("Symbol", symbols, key="chart_sym")

    # Try loading cached OHLCV
    ohlcv = _load_ohlcv_for_symbol(selected_sym, metadata.get("interval", "5m"))
    sym_trades = trades_df[trades_df["symbol"] == selected_sym] if "symbol" in trades_df.columns else trades_df

    if ohlcv is not None and not ohlcv.empty:
        indicators = st.multiselect(
            "Overlay Indicators",
            [c for c in ohlcv.columns if c not in ("Open", "High", "Low", "Close", "Volume")],
            default=[],
            key="chart_ind",
        )
        fig = create_candlestick_with_trades(ohlcv, sym_trades, indicators)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No OHLCV data cached for {selected_sym}. Run `python -m data_cache.daily_updater --full` first.")


def _render_performance_tab(trades_df: pd.DataFrame, equity: pd.Series):
    """Tab 2: Equity curve, drawdown, metrics."""

    # Calculate metrics
    metrics = calculate_metrics(trades_df, equity)

    # Key metrics row
    st.subheader("Key Metrics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trades", metrics.total_trades)
    c2.metric("Win Rate", f"{metrics.win_rate:.1%}")
    c3.metric("Sharpe", f"{metrics.sharpe_ratio:.2f}")
    c4.metric("Sortino", f"{metrics.sortino_ratio:.2f}")
    c5.metric("Max DD", f"{metrics.max_drawdown:.1%}")
    c6.metric("Profit Factor", f"{metrics.profit_factor:.2f}")

    c7, c8, c9, c10, c11, c12 = st.columns(6)
    c7.metric("Total Return", f"{metrics.total_return:.1%}")
    c8.metric("CAGR", f"{metrics.cagr:.1%}")
    c9.metric("Avg Win", f"{metrics.avg_win:,.2f}")
    c10.metric("Avg Loss", f"{metrics.avg_loss:,.2f}")
    c11.metric("Expectancy", f"{metrics.expectancy:,.2f}")
    c12.metric("Calmar", f"{metrics.calmar_ratio:.2f}")

    # Equity curve
    if len(equity) > 1:
        st.subheader("Equity Curve")
        fig_eq = create_equity_curve(equity)
        st.plotly_chart(fig_eq, use_container_width=True)

        st.subheader("Drawdown")
        fig_dd = create_drawdown_chart(equity)
        st.plotly_chart(fig_dd, use_container_width=True)

    # Win/loss distribution
    if "pnl" in trades_df.columns:
        st.subheader("Trade Return Distribution")
        fig_hist = create_returns_histogram(trades_df["pnl"].dropna())
        st.plotly_chart(fig_hist, use_container_width=True)


def _render_analysis_tab(trades_df: pd.DataFrame):
    """Tab 3: Heatmaps, regime analysis."""
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Time Heatmap")
        if "entry_time" in trades_df.columns and "pnl" in trades_df.columns:
            fig_hm = create_time_heatmap(trades_df, metric="pnl", grouping="hour_dow")
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("Need entry_time and pnl columns for heatmap.")

    with col_right:
        st.subheader("Regime Breakdown")
        if "market_regime" in trades_df.columns:
            fig_regime = create_regime_breakdown(trades_df)
            st.plotly_chart(fig_regime, use_container_width=True)
        else:
            st.info("No market_regime data available.")

    # Rolling metrics
    if "pnl" in trades_df.columns and len(trades_df) >= 30:
        st.subheader("Rolling Metrics (30-trade window)")
        rolling = calculate_rolling_metrics(trades_df, window=30)
        if not rolling.empty:
            from aqtis.dashboard.theme import apply_theme
            fig_roll = go.Figure()
            fig_roll.add_trace(
                go.Scatter(
                    x=rolling["trade_num"],
                    y=rolling["sharpe"],
                    name="Rolling Sharpe",
                    line=dict(color=AQTIS_COLORS["blue"]),
                )
            )
            fig_roll.add_trace(
                go.Scatter(
                    x=rolling["trade_num"],
                    y=rolling["win_rate"] * 100,
                    name="Rolling Win Rate %",
                    yaxis="y2",
                    line=dict(color=AQTIS_COLORS["green"], dash="dot"),
                )
            )
            fig_roll.update_layout(
                title="Rolling Performance",
                yaxis=dict(title="Sharpe"),
                yaxis2=dict(title="Win Rate %", overlaying="y", side="right"),
                height=350,
            )
            st.plotly_chart(apply_theme(fig_roll), use_container_width=True)


def _render_inspector_tab(trades_df: pd.DataFrame, memory=None):
    """Tab 4: Individual trade drill-down."""
    st.subheader("Trade Inspector")

    if trades_df.empty:
        st.info("No trades to inspect.")
        return

    # Trade selector
    trades_df = trades_df.reset_index(drop=True)
    display_col = []
    for c in ("symbol", "action", "pnl", "entry_time"):
        if c in trades_df.columns:
            display_col.append(c)

    labels = [
        f"#{i} | {' | '.join(str(trades_df.iloc[i].get(c, '')) for c in display_col)}"
        for i in range(len(trades_df))
    ]
    selected_idx = st.selectbox("Select Trade", range(len(trades_df)), format_func=lambda i: labels[i])

    trade = trades_df.iloc[selected_idx].to_dict()
    inspector = TradeInspector(trade, memory_layer=memory)

    # Summary
    summary = inspector.get_summary()
    cols = st.columns(4)
    items = list(summary.items())
    for i, (k, v) in enumerate(items):
        cols[i % 4].metric(k, v)

    # Try loading OHLCV for mini chart
    sym = trade.get("symbol")
    if sym:
        ohlcv = _load_ohlcv_for_symbol(sym)
        if ohlcv is not None:
            inspector.ohlcv = ohlcv
            fig = inspector.create_trade_chart()
            st.plotly_chart(fig, use_container_width=True)

    # Indicator values at entry
    ind_vals = inspector.get_indicators_at_entry()
    if ind_vals:
        with st.expander("Indicator Values at Entry"):
            st.json(ind_vals)

    # Similar trades
    similar = inspector.get_similar_trades()
    if similar:
        with st.expander(f"Similar Historical Trades ({len(similar)})"):
            st.dataframe(pd.DataFrame(similar), use_container_width=True)


def _render_evolution_tab(results):
    """Tab 5: Algorithm evolution view."""
    st.subheader("Algorithm Evolution")

    # Check for multi-run results
    runs = None
    if isinstance(results, list):
        runs = results
    elif isinstance(results, dict):
        if "iteration_history" in results:
            runs = results["iteration_history"]
        elif isinstance(results.get("runs"), list):
            runs = results["runs"]
        elif isinstance(results.get("trades"), list) and len(results.get("trades", [])) > 0:
            # Single-run result — treat the entire result as one "run"
            runs = [results]

    if not runs:
        st.info("No evolution data found in this result file.")
        return

    if not runs:
        st.info("No evolution data available.")
        return

    run_nums = list(range(1, len(runs) + 1))
    sharpes = [r.get("sharpe", r.get("metrics", {}).get("sharpe_ratio", 0)) for r in runs]
    pnls = [r.get("pnl", r.get("metrics", {}).get("pnl", 0)) for r in runs]
    win_rates = [
        r.get("win_rate", r.get("metrics", {}).get("win_rate", 0)) for r in runs
    ]

    from aqtis.dashboard.theme import apply_theme

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=run_nums,
            y=sharpes,
            name="Sharpe Ratio",
            mode="lines+markers",
            line=dict(color=AQTIS_COLORS["blue"]),
        )
    )
    # Cumulative P&L on secondary axis
    cum_pnl = np.cumsum(pnls)
    fig.add_trace(
        go.Scatter(
            x=run_nums,
            y=cum_pnl,
            name="Cumulative P&L",
            mode="lines",
            yaxis="y2",
            line=dict(color=AQTIS_COLORS["green"], dash="dot"),
        )
    )
    fig.update_layout(
        title="Evolution Across Runs",
        xaxis_title="Run #",
        yaxis=dict(title="Sharpe Ratio"),
        yaxis2=dict(title="Cumulative P&L", overlaying="y", side="right"),
        height=450,
    )
    st.plotly_chart(apply_theme(fig), use_container_width=True)

    # Summary table
    st.subheader("Run Summary")
    summary_df = pd.DataFrame(
        {
            "Run": run_nums,
            "Sharpe": sharpes,
            "P&L": pnls,
            "Win Rate": win_rates,
            "Cum P&L": cum_pnl,
        }
    )
    st.dataframe(summary_df, use_container_width=True)


# ---------------------------------------------------------------------------
# OHLCV loading helper
# ---------------------------------------------------------------------------


def _load_ohlcv_for_symbol(symbol: str, interval: str = "5m") -> pd.DataFrame:
    """Try loading OHLCV from the data cache (new pickle-based or legacy Parquet)."""
    # Try new LocalDataCache first
    try:
        from data.local_cache import get_cache
        cache = get_cache()
        df = cache.load_from_cache(symbol, interval)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Fallback to legacy Parquet CacheManager
    try:
        from data_cache.cache_manager import CacheManager
        cm = CacheManager()
        df = cm.get_full_data(symbol, interval)
        return df
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auto-execute when Streamlit discovers this as a page
# ---------------------------------------------------------------------------

def _auto_run():
    """Called when Streamlit runs this file directly as a page."""
    try:
        st.set_page_config(page_title="Backtest Analysis", page_icon="📊", layout="wide")
    except Exception:
        pass  # Already set by app.py

    try:
        from aqtis.config.settings import load_config
        from aqtis.memory.memory_layer import MemoryLayer

        @st.cache_resource
        def _get_memory():
            config = load_config()
            return MemoryLayer(
                db_path=str(config.system.db_path),
                vector_path=str(config.system.vector_db_path),
            )

        _memory = _get_memory()
    except Exception:
        _memory = None

    render_backtest_dashboard(_memory)


# When Streamlit auto-discovers this file as a page, it executes it directly.
# We detect this by checking if streamlit is actively running.
try:
    _ctx = st.runtime.scriptrunner.get_script_run_ctx()
    if _ctx is not None:
        _auto_run()
except Exception:
    try:
        # Older streamlit versions
        from streamlit.runtime.scriptrunner import get_script_run_ctx as _get_ctx
        if _get_ctx() is not None:
            _auto_run()
    except Exception:
        pass
