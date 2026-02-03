"""Reusable Plotly chart components for AQTIS dashboard."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from aqtis.dashboard.theme import AQTIS_COLORS, apply_theme


def create_candlestick_with_trades(
    ohlcv: pd.DataFrame,
    trades: pd.DataFrame = None,
    indicators: list = None,
) -> go.Figure:
    """
    Create interactive candlestick chart with trade markers and volume.

    Args:
        ohlcv: DataFrame with Open, High, Low, Close, Volume.
        trades: DataFrame with entry_time, exit_time, entry_price,
                exit_price, pnl, pnl_percent, action.
        indicators: Indicator column names to overlay from ohlcv.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.8, 0.2],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=ohlcv.index,
            open=ohlcv["Open"],
            high=ohlcv["High"],
            low=ohlcv["Low"],
            close=ohlcv["Close"],
            name="Price",
            increasing_line_color=AQTIS_COLORS["green"],
            decreasing_line_color=AQTIS_COLORS["red"],
        ),
        row=1,
        col=1,
    )

    # Volume bars
    colors = [
        AQTIS_COLORS["red"] if c < o else AQTIS_COLORS["green"]
        for c, o in zip(ohlcv["Close"], ohlcv["Open"])
    ]
    fig.add_trace(
        go.Bar(
            x=ohlcv.index,
            y=ohlcv["Volume"],
            marker_color=colors,
            opacity=0.4,
            name="Volume",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Trade markers
    if trades is not None and not trades.empty:
        # Buy entries
        buys = trades[trades.get("action", pd.Series(dtype=str)).str.upper() == "BUY"]
        if not buys.empty and "entry_time" in buys.columns and "entry_price" in buys.columns:
            fig.add_trace(
                go.Scatter(
                    x=buys["entry_time"],
                    y=buys["entry_price"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color=AQTIS_COLORS["green"],
                        line=dict(width=1, color="white"),
                    ),
                    name="Buy",
                    hovertemplate="<b>BUY</b><br>%{x}<br>Price: %{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        # Sell exits
        if "exit_time" in trades.columns and "exit_price" in trades.columns:
            exits = trades.dropna(subset=["exit_time", "exit_price"])
            if not exits.empty:
                fig.add_trace(
                    go.Scatter(
                        x=exits["exit_time"],
                        y=exits["exit_price"],
                        mode="markers",
                        marker=dict(
                            symbol="triangle-down",
                            size=12,
                            color=AQTIS_COLORS["red"],
                            line=dict(width=1, color="white"),
                        ),
                        name="Sell",
                        hovertemplate="<b>SELL</b><br>%{x}<br>Price: %{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # Trade spans (shaded win/loss regions)
        for _, trade in trades.iterrows():
            if pd.notna(trade.get("entry_time")) and pd.notna(trade.get("exit_time")):
                pnl = trade.get("pnl", 0)
                color = "rgba(0,210,106,0.08)" if pnl > 0 else "rgba(255,107,107,0.08)"
                fig.add_vrect(
                    x0=trade["entry_time"],
                    x1=trade["exit_time"],
                    fillcolor=color,
                    layer="below",
                    line_width=0,
                    row=1,
                    col=1,
                )

    # Indicator overlays
    if indicators:
        line_colors = [
            AQTIS_COLORS["blue"],
            AQTIS_COLORS["yellow"],
            AQTIS_COLORS["purple"],
            "#ff9ff3",
            "#48dbfb",
        ]
        for idx, ind in enumerate(indicators):
            if ind in ohlcv.columns:
                fig.add_trace(
                    go.Scatter(
                        x=ohlcv.index,
                        y=ohlcv[ind],
                        name=ind,
                        line=dict(
                            width=1,
                            color=line_colors[idx % len(line_colors)],
                        ),
                    ),
                    row=1,
                    col=1,
                )

    fig.update_layout(
        title="Price Chart with Trade Signals",
        xaxis_rangeslider_visible=False,
        height=650,
    )
    return apply_theme(fig)


def create_equity_curve(
    equity: pd.Series,
    benchmark: pd.Series = None,
    show_drawdown: bool = True,
) -> go.Figure:
    """Create equity curve with optional benchmark and drawdown shading."""
    rows = 2 if show_drawdown else 1
    heights = [0.7, 0.3] if show_drawdown else [1.0]
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=heights,
    )

    # Portfolio equity
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity,
            name="Portfolio",
            line=dict(color=AQTIS_COLORS["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(77,171,247,0.1)",
        ),
        row=1,
        col=1,
    )

    # Benchmark
    if benchmark is not None and len(benchmark) > 0:
        # Normalize to same starting value
        norm_bench = benchmark / benchmark.iloc[0] * equity.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=norm_bench.index,
                y=norm_bench,
                name="Benchmark (NIFTY 50)",
                line=dict(color=AQTIS_COLORS["muted"], width=1, dash="dash"),
            ),
            row=1,
            col=1,
        )

    # Drawdown subplot
    if show_drawdown:
        peak = equity.expanding().max()
        dd_pct = (equity - peak) / peak * 100
        fig.add_trace(
            go.Scatter(
                x=dd_pct.index,
                y=dd_pct,
                name="Drawdown %",
                fill="tozeroy",
                fillcolor="rgba(255,107,107,0.3)",
                line=dict(color=AQTIS_COLORS["red"], width=1),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    fig.update_layout(title="Equity Curve", height=550)
    fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
    return apply_theme(fig)


def create_drawdown_chart(equity: pd.Series) -> go.Figure:
    """Dedicated drawdown analysis chart."""
    peak = equity.expanding().max()
    dd = equity - peak
    dd_pct = dd / peak * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd_pct.index,
            y=dd_pct,
            fill="tozeroy",
            fillcolor="rgba(255,107,107,0.3)",
            line=dict(color=AQTIS_COLORS["red"]),
            name="Drawdown %",
        )
    )

    # Annotate max drawdown
    max_dd_idx = dd_pct.idxmin()
    max_dd_val = dd_pct.min()
    fig.add_annotation(
        x=max_dd_idx,
        y=max_dd_val,
        text=f"Max DD: {max_dd_val:.1f}%",
        showarrow=True,
        arrowhead=2,
        font=dict(color=AQTIS_COLORS["red"]),
    )

    fig.update_layout(title="Drawdown Analysis", yaxis_title="Drawdown %", height=350)
    return apply_theme(fig)


def create_returns_histogram(returns: pd.Series) -> go.Figure:
    """Win/loss distribution histogram."""
    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    fig = go.Figure()
    if len(wins):
        fig.add_trace(
            go.Histogram(
                x=wins,
                name="Wins",
                marker_color=AQTIS_COLORS["green"],
                opacity=0.7,
            )
        )
    if len(losses):
        fig.add_trace(
            go.Histogram(
                x=losses,
                name="Losses",
                marker_color=AQTIS_COLORS["red"],
                opacity=0.7,
            )
        )

    fig.update_layout(
        title="Trade Return Distribution",
        xaxis_title="P&L",
        yaxis_title="Count",
        barmode="overlay",
        height=350,
    )

    # Annotations
    fig.add_vline(
        x=returns.mean(),
        line_dash="dash",
        line_color=AQTIS_COLORS["yellow"],
        annotation_text=f"Mean: {returns.mean():.2f}",
    )

    return apply_theme(fig)


def create_time_heatmap(
    trades: pd.DataFrame,
    metric: str = "pnl",
    grouping: str = "hour_dow",
) -> go.Figure:
    """
    Time-based performance heatmap.

    grouping: 'hour_dow' (hour vs day-of-week) or 'monthly'.
    """
    if trades.empty:
        return go.Figure()

    trades = trades.copy()

    if "entry_time" in trades.columns:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    else:
        return go.Figure()

    if grouping == "hour_dow":
        trades["hour"] = trades["entry_time"].dt.hour
        trades["dow"] = trades["entry_time"].dt.day_name()
        pivot = trades.pivot_table(
            values=metric,
            index="hour",
            columns="dow",
            aggfunc="mean",
        )
        # Reorder days
        day_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
        ]
        pivot = pivot[[d for d in day_order if d in pivot.columns]]
        title = f"Average {metric} by Hour & Day of Week"
    else:
        trades["month"] = trades["entry_time"].dt.month
        trades["year"] = trades["entry_time"].dt.year
        pivot = trades.pivot_table(
            values=metric,
            index="month",
            columns="year",
            aggfunc="sum",
        )
        title = f"Monthly {metric}"

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.astype(str),
            y=pivot.index.astype(str),
            colorscale="RdYlGn",
            zmid=0,
            hoverongaps=False,
        )
    )
    fig.update_layout(title=title, height=400)
    return apply_theme(fig)


def create_strategy_comparison(strategies_metrics: dict) -> go.Figure:
    """Bar chart comparing strategies across key metrics."""
    names = list(strategies_metrics.keys())
    sharpes = [s.get("sharpe", 0) for s in strategies_metrics.values()]
    win_rates = [s.get("win_rate", 0) * 100 for s in strategies_metrics.values()]
    returns = [s.get("total_return", 0) * 100 for s in strategies_metrics.values()]

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["Sharpe Ratio", "Win Rate %", "Total Return %"],
    )

    fig.add_trace(
        go.Bar(x=names, y=sharpes, marker_color=AQTIS_COLORS["blue"], name="Sharpe"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(x=names, y=win_rates, marker_color=AQTIS_COLORS["green"], name="Win Rate"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(x=names, y=returns, marker_color=AQTIS_COLORS["purple"], name="Return"),
        row=1,
        col=3,
    )

    fig.update_layout(title="Strategy Comparison", height=350, showlegend=False)
    return apply_theme(fig)


def create_regime_breakdown(trades: pd.DataFrame) -> go.Figure:
    """Performance breakdown by market regime."""
    if trades.empty or "market_regime" not in trades.columns:
        return go.Figure()

    regime_stats = trades.groupby("market_regime").agg(
        count=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        avg_pnl=("pnl", "mean"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
    ).reset_index()

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}]],
        subplot_titles=["Trade Distribution", "Win Rate by Regime"],
    )

    fig.add_trace(
        go.Pie(
            labels=regime_stats["market_regime"],
            values=regime_stats["count"],
            hole=0.4,
            marker=dict(
                colors=[
                    AQTIS_COLORS["green"],
                    AQTIS_COLORS["red"],
                    AQTIS_COLORS["blue"],
                    AQTIS_COLORS["yellow"],
                    AQTIS_COLORS["purple"],
                ]
            ),
        ),
        row=1,
        col=1,
    )

    colors = [
        AQTIS_COLORS["green"] if wr > 0.5 else AQTIS_COLORS["red"]
        for wr in regime_stats["win_rate"]
    ]
    fig.add_trace(
        go.Bar(
            x=regime_stats["market_regime"],
            y=regime_stats["win_rate"] * 100,
            marker_color=colors,
            name="Win Rate %",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(title="Regime Analysis", height=400)
    return apply_theme(fig)
