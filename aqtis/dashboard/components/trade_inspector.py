"""Individual trade inspection component."""

from datetime import timedelta

import pandas as pd
import plotly.graph_objects as go

from aqtis.dashboard.theme import AQTIS_COLORS, apply_theme


class TradeInspector:
    """Detailed inspection of individual trades."""

    def __init__(self, trade: dict, ohlcv: pd.DataFrame = None, memory_layer=None):
        """
        Args:
            trade: Single trade dict with keys like entry_time, exit_time,
                   entry_price, exit_price, pnl, symbol, etc.
            ohlcv: Full OHLCV data for the symbol.
            memory_layer: AQTIS memory layer for similar trades lookup.
        """
        self.trade = trade
        self.ohlcv = ohlcv
        self.memory = memory_layer

    def get_summary(self) -> dict:
        """Return trade summary as a flat dict for display."""
        t = self.trade
        pnl = t.get("pnl", 0)
        return {
            "Symbol": t.get("symbol", "N/A"),
            "Action": t.get("action", "N/A"),
            "Entry Price": f"{t.get('entry_price', 0):.2f}",
            "Exit Price": f"{t.get('exit_price', 0):.2f}",
            "P&L": f"{pnl:+.2f}",
            "P&L %": f"{t.get('pnl_percent', 0):+.2f}%",
            "Entry Time": str(t.get("entry_time", ""))[:19],
            "Exit Time": str(t.get("exit_time", ""))[:19],
            "Regime": t.get("market_regime", "N/A"),
            "Confidence": f"{t.get('confidence', 0):.2f}",
            "Strategy": t.get("strategy_id", "N/A"),
        }

    def create_trade_chart(self, context_bars: int = 50) -> go.Figure:
        """
        Create a mini candlestick chart for this trade's time window.

        Args:
            context_bars: Number of bars before/after the trade to show.
        """
        if self.ohlcv is None or self.ohlcv.empty:
            return go.Figure()

        entry_time = pd.Timestamp(self.trade.get("entry_time"))
        exit_time = pd.Timestamp(self.trade.get("exit_time"))

        # Find the window
        idx = self.ohlcv.index.get_indexer([entry_time], method="nearest")[0]
        start = max(0, idx - context_bars)
        end = min(len(self.ohlcv), idx + context_bars)
        window = self.ohlcv.iloc[start:end]

        fig = go.Figure()
        fig.add_trace(
            go.Candlestick(
                x=window.index,
                open=window["Open"],
                high=window["High"],
                low=window["Low"],
                close=window["Close"],
                name="Price",
                increasing_line_color=AQTIS_COLORS["green"],
                decreasing_line_color=AQTIS_COLORS["red"],
            )
        )

        # Entry marker
        entry_price = self.trade.get("entry_price")
        if entry_price:
            fig.add_trace(
                go.Scatter(
                    x=[entry_time],
                    y=[entry_price],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-up",
                        size=15,
                        color=AQTIS_COLORS["green"],
                    ),
                    name="Entry",
                )
            )

        # Exit marker
        exit_price = self.trade.get("exit_price")
        if exit_price and pd.notna(exit_time):
            fig.add_trace(
                go.Scatter(
                    x=[exit_time],
                    y=[exit_price],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=15,
                        color=AQTIS_COLORS["red"],
                    ),
                    name="Exit",
                )
            )

        # Shade trade region
        pnl = self.trade.get("pnl", 0)
        color = "rgba(0,210,106,0.12)" if pnl > 0 else "rgba(255,107,107,0.12)"
        if pd.notna(entry_time) and pd.notna(exit_time):
            fig.add_vrect(
                x0=entry_time,
                x1=exit_time,
                fillcolor=color,
                layer="below",
                line_width=0,
            )

        sym = self.trade.get("symbol", "")
        fig.update_layout(
            title=f"Trade Detail: {sym}",
            xaxis_rangeslider_visible=False,
            height=400,
        )
        return apply_theme(fig)

    def get_indicators_at_entry(self) -> dict:
        """Return indicator values at trade entry time."""
        if self.ohlcv is None or self.ohlcv.empty:
            return {}

        entry_time = pd.Timestamp(self.trade.get("entry_time"))
        idx = self.ohlcv.index.get_indexer([entry_time], method="nearest")[0]
        row = self.ohlcv.iloc[idx]

        # Filter to indicator columns (exclude OHLCV)
        ohlcv_cols = {"Open", "High", "Low", "Close", "Volume"}
        indicator_vals = {
            col: round(float(row[col]), 4)
            for col in row.index
            if col not in ohlcv_cols and pd.notna(row[col])
        }
        return indicator_vals

    def get_similar_trades(self, top_k: int = 5) -> list:
        """Find similar historical trades via memory layer."""
        if self.memory is None:
            return []
        try:
            return self.memory.get_similar_trades(self.trade, top_k=top_k)
        except Exception:
            return []
