"""
Live Backtest Runner — Run custom backtests from the AQTIS dashboard.

Allows users to:
  1. Pick any equity (NIFTY 50 or custom ticker)
  2. Select an AQTIS evolved strategy / DNA
  3. Configure backtest parameters (days, capital, etc.)
  4. Run the backtest and see real-time results
"""

import json
import os
import sys
import time
import traceback
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
except ImportError:
    raise ImportError("streamlit and plotly required: pip install streamlit plotly")

from aqtis.dashboard.theme import AQTIS_COLORS

try:
    from data.symbols import NIFTY_50_SYMBOLS
except ImportError:
    from data_cache.symbols import NIFTY_50_SYMBOLS


# ---------------------------------------------------------------------------
# DNA / Strategy Loading
# ---------------------------------------------------------------------------

def _find_available_dna() -> List[dict]:
    """Scan reports/ for evolved best_dna.json files."""
    dna_list = []
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return dna_list

    for run_dir in sorted(reports_dir.iterdir(), reverse=True):
        dna_path = run_dir / "best_dna.json"
        if dna_path.exists():
            try:
                with open(dna_path) as f:
                    data = json.load(f)
                entries = data.get("entries", [data] if "dna_id" in data else [])
                if entries:
                    best = entries[0]
                    dna_list.append({
                        "run": run_dir.name,
                        "path": str(dna_path),
                        "dna_id": best.get("dna_id", "unknown"),
                        "sharpe": best.get("sharpe_ratio", 0),
                        "win_rate": best.get("win_rate", 0),
                        "profit": best.get("net_profit", 0),
                        "generation": best.get("generation", 0),
                        "genes": best.get("genes", {}),
                        "fitness": best.get("fitness_score", 0),
                    })
            except Exception:
                pass

    return dna_list


def _load_dna_weights(dna_entry: dict) -> Dict[str, float]:
    """Extract indicator weights from a DNA entry."""
    genes = dna_entry.get("genes", {})
    weights = {}
    for name, gene in genes.items():
        if isinstance(gene, dict) and gene.get("active", True):
            weights[name] = gene.get("weight", 0.0)
        elif isinstance(gene, (int, float)):
            weights[name] = float(gene)
    return weights


# ---------------------------------------------------------------------------
# Lightweight Backtest Engine (runs in-process, no LLM needed)
# ---------------------------------------------------------------------------

class QuickBacktester:
    """
    Lightweight backtest engine that runs AQTIS signal logic
    without needing Gemini LLM. Uses pure indicator-based scoring
    with evolved DNA weights.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        initial_capital: float = 100_000,
        entry_threshold: float = 0.15,
        max_positions: int = 5,
        stop_loss_atr_mult: float = 2.0,
        take_profit_atr_mult: float = 3.0,
        max_hold_days: int = 5,
    ):
        self.weights = weights
        self.initial_capital = initial_capital
        self.entry_threshold = entry_threshold
        self.max_positions = max_positions
        self.sl_mult = stop_loss_atr_mult
        self.tp_mult = take_profit_atr_mult
        self.max_hold_days = max_hold_days

        # State
        self.capital = initial_capital
        self.positions: Dict[str, dict] = {}
        self.trades: List[dict] = []
        self.equity_curve: List[dict] = []

    def run(
        self,
        symbol_data: Dict[str, pd.DataFrame],
        days: int = 60,
        progress_callback=None,
    ) -> dict:
        """
        Run backtest over the provided data.

        Args:
            symbol_data: {symbol: DataFrame with OHLCV + indicators}
            days: Number of trading days to simulate.
            progress_callback: Optional callable(pct, msg) for Streamlit progress.

        Returns:
            Result dict with trades, metrics, equity curve.
        """
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

        # Collect all unique trading dates across symbols
        all_dates = set()
        for df in symbol_data.values():
            if df is not None and not df.empty:
                all_dates.update(df.index.normalize().unique())

        sim_dates = sorted(all_dates)[-days:] if len(all_dates) > days else sorted(all_dates)

        total_days = len(sim_dates)
        for day_idx, date in enumerate(sim_dates):
            if progress_callback:
                progress_callback(
                    (day_idx + 1) / total_days,
                    f"Day {day_idx + 1}/{total_days} — {date.strftime('%Y-%m-%d')}"
                )

            # Process each symbol
            for symbol, df in symbol_data.items():
                if df is None or df.empty:
                    continue

                # Get data up to this date
                mask = df.index.normalize() == date
                day_data = df[mask]
                if day_data.empty:
                    continue

                # Use last bar of the day for signal
                bar = day_data.iloc[-1]
                bar_idx = df.index.get_loc(day_data.index[-1])

                # Check exits first
                if symbol in self.positions:
                    self._check_exit(symbol, bar, date, bar_idx)

                # Check entries
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    self._check_entry(symbol, df, bar, bar_idx, date)

            # Record equity
            unrealized = self._calc_unrealized(symbol_data, date)
            self.equity_curve.append({
                "date": date.isoformat(),
                "equity": self.capital + unrealized,
                "cash": self.capital,
                "positions": len(self.positions),
            })

        # Close all remaining positions at last available prices
        self._close_all(symbol_data, sim_dates[-1] if sim_dates else datetime.now())

        return self._compile_results(sim_dates)

    def _compute_signal(self, df: pd.DataFrame, bar_idx: int) -> float:
        """Compute weighted signal score from indicators."""
        if bar_idx < 1:
            return 0.0

        row = df.iloc[bar_idx]
        score = 0.0
        total_weight = 0.0

        for indicator, weight in self.weights.items():
            # Check if indicator exists in data (try exact match and normalized)
            val = None
            norm_key = f"{indicator}_norm"

            if norm_key in df.columns:
                val = row.get(norm_key)
            elif indicator in df.columns:
                val = row.get(indicator)

            if val is not None and not pd.isna(val):
                score += val * weight
                total_weight += abs(weight)

        if total_weight > 0:
            score /= total_weight

        return float(np.clip(score, -1.0, 1.0))

    def _check_entry(self, symbol: str, df: pd.DataFrame, bar: pd.Series, bar_idx: int, date):
        """Check for entry signal."""
        signal = self._compute_signal(df, bar_idx)

        if abs(signal) < self.entry_threshold:
            return

        price = float(bar.get("Close", bar.get("close", 0)))
        if price <= 0:
            return

        atr = float(bar.get("ATR_14", bar.get("atr_14", price * 0.02)))
        if pd.isna(atr) or atr <= 0:
            atr = price * 0.02

        action = "BUY" if signal > 0 else "SELL"

        # Position sizing: risk 2% of capital per trade
        risk_per_trade = self.capital * 0.02
        stop_distance = atr * self.sl_mult
        if stop_distance <= 0:
            return
        shares = max(1, int(risk_per_trade / stop_distance))
        cost = shares * price

        if cost > self.capital * 0.3:  # Max 30% per position
            shares = max(1, int(self.capital * 0.3 / price))
            cost = shares * price

        if cost > self.capital:
            return

        if action == "BUY":
            sl = price - stop_distance
            tp = price + atr * self.tp_mult
        else:
            sl = price + stop_distance
            tp = price - atr * self.tp_mult

        self.positions[symbol] = {
            "symbol": symbol,
            "action": action,
            "entry_price": price,
            "shares": shares,
            "stop_loss": sl,
            "take_profit": tp,
            "entry_date": date.isoformat() if hasattr(date, "isoformat") else str(date),
            "entry_bar": bar_idx,
            "signal_score": signal,
            "atr": atr,
        }
        self.capital -= cost

    def _check_exit(self, symbol: str, bar: pd.Series, date, bar_idx: int):
        """Check exit conditions for an open position."""
        pos = self.positions[symbol]
        price = float(bar.get("Close", bar.get("close", 0)))
        if price <= 0:
            return

        action = pos["action"]
        entry = pos["entry_price"]
        reason = None

        # Stop loss
        if action == "BUY" and price <= pos["stop_loss"]:
            reason = "stop_loss"
        elif action == "SELL" and price >= pos["stop_loss"]:
            reason = "stop_loss"

        # Take profit
        if action == "BUY" and price >= pos["take_profit"]:
            reason = "take_profit"
        elif action == "SELL" and price <= pos["take_profit"]:
            reason = "take_profit"

        # Max hold time
        entry_date = pd.Timestamp(pos["entry_date"])
        if hasattr(date, "date"):
            current_date = date
        else:
            current_date = pd.Timestamp(date)
        hold_days = (current_date - entry_date).days
        if hold_days >= self.max_hold_days:
            reason = "max_hold"

        if reason:
            self._close_position(symbol, price, date, reason)

    def _close_position(self, symbol: str, exit_price: float, date, reason: str):
        """Close a position and record the trade."""
        pos = self.positions.pop(symbol)
        entry_cost = pos["entry_price"] * pos["shares"]

        if pos["action"] == "BUY":
            pnl = (exit_price - pos["entry_price"]) * pos["shares"]
            # We subtracted entry_cost on entry. Now get back exit proceeds.
            self.capital += exit_price * pos["shares"]
        else:
            pnl = (pos["entry_price"] - exit_price) * pos["shares"]
            # We subtracted entry_cost on entry (as collateral). Get it back + pnl.
            self.capital += entry_cost + pnl

        self.trades.append({
            "symbol": symbol,
            "action": pos["action"],
            "entry_price": pos["entry_price"],
            "exit_price": exit_price,
            "shares": pos["shares"],
            "pnl": round(pnl, 2),
            "pnl_percent": round(pnl / (pos["entry_price"] * pos["shares"]) * 100, 2),
            "entry_time": pos["entry_date"],
            "exit_time": date.isoformat() if hasattr(date, "isoformat") else str(date),
            "exit_reason": reason,
            "signal_score": pos["signal_score"],
            "hold_days": (pd.Timestamp(date) - pd.Timestamp(pos["entry_date"])).days,
        })

    def _close_all(self, symbol_data: Dict[str, pd.DataFrame], last_date):
        """Close all open positions at last available prices."""
        for symbol in list(self.positions.keys()):
            df = symbol_data.get(symbol)
            if df is not None and not df.empty:
                price = float(df.iloc[-1].get("Close", df.iloc[-1].get("close", 0)))
                self._close_position(symbol, price, last_date, "end_of_sim")

    def _calc_unrealized(self, symbol_data: Dict[str, pd.DataFrame], date) -> float:
        """Calculate unrealized P&L for open positions."""
        unrealized = 0.0
        for symbol, pos in self.positions.items():
            df = symbol_data.get(symbol)
            if df is None or df.empty:
                continue
            mask = df.index.normalize() <= date
            if not mask.any():
                continue
            latest = df[mask].iloc[-1]
            price = float(latest.get("Close", latest.get("close", 0)))
            if pos["action"] == "BUY":
                unrealized += (price - pos["entry_price"]) * pos["shares"]
            else:
                unrealized += (pos["entry_price"] - price) * pos["shares"]
        return unrealized

    def _compile_results(self, sim_dates: list) -> dict:
        """Compile final results."""
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        # Metrics
        total_trades = len(self.trades)
        if total_trades > 0:
            pnls = [t["pnl"] for t in self.trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]

            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            total_pnl = sum(pnls)
            profit_factor = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

            # Sharpe from equity curve
            if len(self.equity_curve) > 1:
                eq_values = [e["equity"] for e in self.equity_curve]
                returns = np.diff(eq_values) / eq_values[:-1]
                sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if np.std(returns) > 0 else 0
            else:
                sharpe = 0

            # Max drawdown
            if len(self.equity_curve) > 1:
                eq = pd.Series([e["equity"] for e in self.equity_curve])
                peak = eq.expanding().max()
                dd = (eq - peak) / peak
                max_dd = float(dd.min())
            else:
                max_dd = 0
        else:
            win_rate = avg_win = avg_loss = total_pnl = sharpe = max_dd = 0
            profit_factor = 0

        final_equity = self.equity_curve[-1]["equity"] if self.equity_curve else self.initial_capital

        return {
            "trades": self.trades,
            "equity_curve": self.equity_curve,
            "capital": {
                "initial": self.initial_capital,
                "final": round(final_equity, 2),
                "pnl": round(final_equity - self.initial_capital, 2),
                "return_pct": round((final_equity - self.initial_capital) / self.initial_capital * 100, 2),
            },
            "metrics": {
                "total_trades": total_trades,
                "wins": len([t for t in self.trades if t["pnl"] > 0]),
                "losses": len([t for t in self.trades if t["pnl"] <= 0]),
                "win_rate": round(win_rate, 4),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "total_pnl": round(total_pnl, 2),
                "sharpe_ratio": round(sharpe, 2),
                "max_drawdown": round(max_dd, 4),
                "profit_factor": round(profit_factor, 2),
            },
            "date_range": {
                "start": sim_dates[0].isoformat() if sim_dates else "",
                "end": sim_dates[-1].isoformat() if sim_dates else "",
            },
        }


# ---------------------------------------------------------------------------
# Data fetching for custom symbols
# ---------------------------------------------------------------------------

def _fetch_symbol_data(symbol: str, interval: str = "5m", days: int = 60) -> Optional[pd.DataFrame]:
    """Fetch OHLCV + indicators for a symbol."""
    # Try local cache first
    try:
        from data.local_cache import get_cache
        cache = get_cache()
        df = cache.get_data(symbol, interval=interval)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    # Fallback: fetch fresh from yfinance
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
        if df is not None and not df.empty:
            # Standardize columns
            col_map = {}
            for c in df.columns:
                lower = c.lower()
                if lower == "open": col_map[c] = "Open"
                elif lower == "high": col_map[c] = "High"
                elif lower == "low": col_map[c] = "Low"
                elif lower == "close": col_map[c] = "Close"
                elif lower == "volume": col_map[c] = "Volume"
            df = df.rename(columns=col_map)
            keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
            df = df[keep].dropna()

            # Compute indicators
            try:
                from data.indicator_computer import IndicatorComputer
                ic = IndicatorComputer()
                indicators = ic.compute_all_indicators(df)
                if indicators is not None and not indicators.empty:
                    df = df.join(indicators, how="left")
            except Exception:
                pass

            return df
    except Exception as e:
        st.warning(f"Failed to fetch {symbol}: {e}")

    return None


# ---------------------------------------------------------------------------
# Default indicator weights (fallback if no DNA selected)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "RSI_14": -0.15,
    "MACD_12_26_9": 0.20,
    "ADX_14": 0.10,
    "BB_pct": -0.12,
    "ATR_14": 0.05,
    "STOCH_k": -0.10,
    "CCI_20": 0.08,
    "OBV": 0.06,
    "MFI_14": -0.08,
    "EMA_9": 0.15,
    "EMA_21": 0.12,
    "SMA_20": 0.10,
    "SMA_50": 0.08,
    "WILLR_14": -0.07,
    "CMF_20": 0.09,
    "AROON_up": 0.06,
    "AROON_down": -0.06,
    "KC_mid": 0.04,
    "DC_high": 0.03,
    "DC_low": -0.03,
}


# ---------------------------------------------------------------------------
# Main page renderer
# ---------------------------------------------------------------------------

def render_live_backtest(memory=None):
    """Render the Live Backtest page in the AQTIS dashboard."""

    st.header("Live Backtest Runner")

    # ===================================================================
    # SIDEBAR: Configuration
    # ===================================================================
    st.sidebar.subheader("Backtest Configuration")

    # --- Engine Mode ---
    engine_mode = st.sidebar.radio(
        "Backtest Engine",
        ["Quick (Indicator Signals)", "Full AQTIS (All Agents + Gemini LLM)"],
        index=0,
        help="Quick mode uses evolved DNA weights for fast scoring. "
             "Full AQTIS runs the complete multi-agent system with Gemini."
    )

    is_full_aqtis = engine_mode.startswith("Full")

    if is_full_aqtis:
        st.markdown(
            "Running the **full AQTIS system** with all 6 agents "
            "(PostMortem, Risk, Strategy, Backtest, Prediction, Research) "
            "and **Gemini LLM** for high-conviction analysis."
        )
    else:
        st.markdown(
            "Run a backtest on **any equity** using AQTIS evolved strategies. "
            "No API key needed — uses pure indicator-based signal scoring."
        )

    # --- Symbol Selection ---
    st.sidebar.markdown("**Symbols**")
    symbol_mode = st.sidebar.radio(
        "Symbol selection",
        ["NIFTY 50 Preset", "Custom Symbols"],
        index=1,
    )

    if symbol_mode == "NIFTY 50 Preset":
        selected_symbols = st.sidebar.multiselect(
            "Select NIFTY 50 Stocks",
            NIFTY_50_SYMBOLS,
            default=["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"],
        )
    else:
        custom_input = st.sidebar.text_area(
            "Enter tickers (one per line)",
            value="RELIANCE.NS\nTCS.NS\nINFY.NS",
            help="Use Yahoo Finance format. Indian stocks end with .NS",
        )
        selected_symbols = [s.strip() for s in custom_input.strip().split("\n") if s.strip()]

    # --- Strategy Selection (Quick mode only) ---
    weights = DEFAULT_WEIGHTS.copy()
    if not is_full_aqtis:
        st.sidebar.markdown("**Strategy / DNA**")
        available_dna = _find_available_dna()

        strategy_options = ["Default Weights (20 indicators)"]
        dna_map = {}
        for dna in available_dna[:15]:
            label = f"{dna['run']} — Sharpe {dna['sharpe']:.2f}, WR {dna['win_rate']:.1%}"
            strategy_options.append(label)
            dna_map[label] = dna

        selected_strategy = st.sidebar.selectbox(
            "Select Strategy",
            strategy_options,
            index=0,
        )

        if selected_strategy == "Default Weights (20 indicators)":
            weights = DEFAULT_WEIGHTS.copy()
            st.sidebar.caption(f"Using {len(weights)} default indicator weights")
        else:
            dna = dna_map[selected_strategy]
            weights = _load_dna_weights(dna)
            st.sidebar.caption(
                f"DNA: {dna['dna_id']} | Gen {dna['generation']} | "
                f"{len(weights)} active genes"
            )

    # --- Full AQTIS Settings ---
    llm_budget = 80
    if is_full_aqtis:
        st.sidebar.markdown("**AQTIS Agent Settings**")
        llm_budget = st.sidebar.slider("LLM Budget (Gemini calls)", 20, 300, 80, 10)
        st.sidebar.caption(
            "Uses Gemini for: pre-trade analysis, post-mortems, "
            "periodic reviews, weight mutations"
        )

    # --- Backtest Parameters ---
    st.sidebar.markdown("**Parameters**")
    days = st.sidebar.slider("Trading Days", 10, 60, 30, 5)
    capital = st.sidebar.number_input("Initial Capital (₹)", 10_000, 10_000_000, 100_000, 10_000)
    interval = st.sidebar.selectbox("Data Interval", ["5m", "15m"], index=0)

    if not is_full_aqtis:
        col_a, col_b = st.sidebar.columns(2)
        entry_threshold = col_a.number_input("Entry Threshold", 0.05, 0.50, 0.15, 0.05)
        max_positions = col_b.number_input("Max Positions", 1, 20, 5, 1)

        col_c, col_d = st.sidebar.columns(2)
        sl_mult = col_c.number_input("SL (ATR mult)", 0.5, 5.0, 2.0, 0.5)
        tp_mult = col_d.number_input("TP (ATR mult)", 0.5, 8.0, 3.0, 0.5)

        max_hold = st.sidebar.slider("Max Hold Days", 1, 20, 5, 1)
    else:
        entry_threshold = 0.15
        max_positions = 5
        sl_mult = 2.0
        tp_mult = 3.0
        max_hold = 5

    # ===================================================================
    # RUN BUTTON
    # ===================================================================
    btn_label = "Run Full AQTIS Backtest" if is_full_aqtis else "Run Quick Backtest"
    run_clicked = st.button(btn_label, type="primary", use_container_width=True)

    if run_clicked and selected_symbols:
        if is_full_aqtis:
            _execute_full_aqtis_backtest(
                symbols=selected_symbols,
                days=days,
                capital=capital,
                llm_budget=llm_budget,
                memory=memory,
            )
        else:
            _execute_backtest(
                symbols=selected_symbols,
                weights=weights,
                days=days,
                capital=capital,
                interval=interval,
                entry_threshold=entry_threshold,
                max_positions=max_positions,
                sl_mult=sl_mult,
                tp_mult=tp_mult,
                max_hold=max_hold,
            )
    elif run_clicked and not selected_symbols:
        st.warning("Please select at least one symbol.")

    # Show previous results if stored in session
    elif "backtest_result" in st.session_state:
        _display_results(st.session_state["backtest_result"])


def _execute_full_aqtis_backtest(symbols, days, capital, llm_budget, memory=None):
    """Execute a full AQTIS backtest using PaperTrader + all agents + Gemini."""

    st.subheader("Full AQTIS Backtest")
    status = st.empty()
    progress = st.progress(0, text="Initializing AQTIS system...")

    try:
        from dotenv import load_dotenv
        load_dotenv()

        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not gemini_key:
            st.error(
                "Gemini API key not found. Set `GEMINI_API_KEY` in your `.env` file.\n\n"
                "Get a key at: https://aistudio.google.com/apikey"
            )
            return

        progress.progress(0.1, text="Loading config...")
        from aqtis.config.settings import load_config
        config = load_config("aqtis_config.yaml")

        progress.progress(0.2, text="Initializing memory layer...")
        from aqtis.memory.memory_layer import MemoryLayer
        if memory is None:
            memory = MemoryLayer(
                db_path=str(config.system.db_path),
                vector_path=str(config.system.vector_db_path),
            )

        progress.progress(0.3, text="Connecting to Gemini LLM...")
        from aqtis.llm.gemini_provider import GeminiProvider
        llm = GeminiProvider(api_key=gemini_key)

        progress.progress(0.4, text="Creating multi-agent orchestrator...")
        from aqtis.orchestrator.orchestrator import MultiAgentOrchestrator
        orchestrator = MultiAgentOrchestrator(
            memory=memory,
            llm=llm,
            config={"portfolio_value": capital},
        )

        progress.progress(0.5, text="Setting up PaperTrader...")
        from aqtis.backtest.paper_trader import PaperTrader

        # Use local cache as data provider — it now has get_historical() method
        # compatible with PaperTrader's expected interface
        data_provider = None
        try:
            from data.local_cache import get_cache
            cache = get_cache()
            # Pre-fetch data for all symbols to ensure cache is warm
            # force_refresh=False means it only downloads from yfinance if data
            # is missing or stale (>24 hours). This avoids re-downloading on
            # every Streamlit rerun.
            progress.progress(0.5, text=f"Pre-fetching data for {len(symbols)} symbols...")
            for i, sym in enumerate(symbols):
                try:
                    cache.fetch_and_cache(sym, interval="5m", force_refresh=False)
                except Exception as fetch_err:
                    st.warning(f"Failed to fetch {sym}: {fetch_err}")
                progress.progress(
                    0.5 + 0.1 * ((i + 1) / len(symbols)),
                    text=f"Fetched {sym} ({i+1}/{len(symbols)})..."
                )
            data_provider = cache
        except Exception as dp_err:
            st.warning(f"Local cache unavailable ({dp_err}), using default MarketDataProvider")

        progress.progress(0.65, text="Setting up PaperTrader...")
        trader = PaperTrader(
            memory=memory,
            orchestrator=orchestrator,
            config=config,
            data_provider=data_provider,
            llm_budget=llm_budget,
        )

        progress.progress(0.7, text=f"Running {days}-day backtest with {len(symbols)} symbols...")
        status.info(
            f"Running full AQTIS backtest: {len(symbols)} symbols, {days} days, "
            f"LLM budget: {llm_budget} calls. This may take a few minutes..."
        )

        # Enable logging to capture PaperTrader output
        import logging
        logging.basicConfig(level=logging.INFO)
        pt_logger = logging.getLogger("aqtis.backtest.paper_trader")
        pt_logger.setLevel(logging.INFO)

        # Run the actual backtest
        result = trader.run(
            symbols=symbols,
            days=days,
            initial_capital=capital,
        )

        progress.progress(1.0, text="Complete!")

        # Show diagnostic info
        if result:
            trade_count = result.get("trade_count", 0)
            days_sim = result.get("days_simulated", 0)
            llm_used = result.get("llm_usage", {}).get("calls_used", 0)
            data_syms = len(result.get("symbols", []))
            status.success(
                f"Full AQTIS backtest complete! "
                f"{days_sim} days, {data_syms} symbols, "
                f"{trade_count} trades, {llm_used} LLM calls used."
            )
        else:
            status.warning("Backtest returned no results.")

        # Normalize result to our display format
        if result:
            display_result = _normalize_aqtis_result(result, capital)
            st.session_state["backtest_result"] = display_result
            _display_results(display_result)
        else:
            st.warning("Backtest returned no results.")

    except ImportError as e:
        st.error(
            f"Missing dependency: {e}\n\n"
            "Make sure all AQTIS packages are installed:\n"
            "`pip install -r requirements.txt`"
        )
    except Exception as e:
        st.error(f"Full AQTIS backtest failed: {e}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())


def _normalize_aqtis_result(result: dict, initial_capital: float) -> dict:
    """Normalize PaperTrader result dict to our display format."""
    cap = result.get("capital", {})
    metrics = result.get("metrics", {})
    agent_log = result.get("agent_log", [])

    # Extract trades from the result — PaperTrader now includes _all_trades
    raw_trades = result.get("trades", [])
    if isinstance(raw_trades, (int, float)):
        raw_trades = []

    # Normalize trade dicts to have consistent field names for display
    display_trades = []
    for t in raw_trades:
        if not isinstance(t, dict):
            continue
        # PaperTrader uses 'asset' instead of 'symbol'
        display_trades.append({
            "symbol": t.get("asset", t.get("symbol", "?")),
            "action": t.get("action", "?"),
            "entry_price": t.get("entry_price", 0),
            "exit_price": t.get("exit_price", 0),
            "shares": t.get("position_size", t.get("shares", 0)),
            "pnl": round(t.get("pnl", 0), 2) if t.get("pnl") is not None else 0,
            "pnl_percent": round(t.get("pnl_percent", 0), 2) if t.get("pnl_percent") is not None else 0,
            "exit_reason": t.get("exit_reason", ""),
            "entry_time": t.get("entry_date", t.get("timestamp", "")),
            "exit_time": t.get("timestamp", ""),
            "signal_score": round(t.get("signal_score", t.get("confidence", 0)), 3),
            "hold_days": 0,
        })

    # Count only completed trades (ones with exit_price)
    completed_trades = [t for t in display_trades if t.get("exit_price", 0) > 0]

    return {
        "trades": completed_trades if completed_trades else display_trades,
        "equity_curve": result.get("equity_curve", []),
        "capital": {
            "initial": initial_capital,
            "final": cap.get("final", initial_capital),
            "pnl": cap.get("pnl", 0),
            "return_pct": cap.get("return_pct", 0),
        },
        "metrics": {
            "total_trades": metrics.get("total_trades", len(completed_trades)),
            "wins": metrics.get("wins", 0),
            "losses": metrics.get("losses", 0),
            "win_rate": metrics.get("win_rate", 0),
            "avg_win": metrics.get("avg_win", 0),
            "avg_loss": metrics.get("avg_loss", 0),
            "total_pnl": cap.get("pnl", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "profit_factor": metrics.get("profit_factor", 0),
        },
        "date_range": result.get("date_range", {}),
        "learning": result.get("learning", {}),
        "agent_log": agent_log,
    }


def _execute_backtest(
    symbols, weights, days, capital, interval,
    entry_threshold, max_positions, sl_mult, tp_mult, max_hold
):
    """Execute the backtest and display results."""

    # --- Data Fetch Phase ---
    st.subheader("Fetching Market Data")
    data_progress = st.progress(0, text="Loading symbols...")
    symbol_data = {}

    for i, sym in enumerate(symbols):
        data_progress.progress(
            (i + 1) / len(symbols),
            text=f"Fetching {sym} ({i+1}/{len(symbols)})..."
        )
        df = _fetch_symbol_data(sym, interval=interval, days=days + 20)
        if df is not None and not df.empty:
            symbol_data[sym] = df

    data_progress.progress(1.0, text=f"Loaded {len(symbol_data)}/{len(symbols)} symbols")

    if not symbol_data:
        st.error("No data could be fetched for any symbol. Check your tickers.")
        return

    # Show data summary
    with st.expander(f"Data Summary ({len(symbol_data)} symbols loaded)"):
        summary_rows = []
        for sym, df in symbol_data.items():
            n_indicators = len([c for c in df.columns if c not in ("Open", "High", "Low", "Close", "Volume")])
            summary_rows.append({
                "Symbol": sym,
                "Bars": len(df),
                "Start": str(df.index.min())[:10],
                "End": str(df.index.max())[:10],
                "Indicators": n_indicators,
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    # --- Backtest Phase ---
    st.subheader("Running Backtest")
    bt_progress = st.progress(0, text="Initializing...")
    status_text = st.empty()

    backtester = QuickBacktester(
        weights=weights,
        initial_capital=capital,
        entry_threshold=entry_threshold,
        max_positions=max_positions,
        stop_loss_atr_mult=sl_mult,
        take_profit_atr_mult=tp_mult,
        max_hold_days=max_hold,
    )

    def progress_cb(pct, msg):
        bt_progress.progress(pct, text=msg)
        status_text.text(msg)

    try:
        result = backtester.run(symbol_data, days=days, progress_callback=progress_cb)
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return

    bt_progress.progress(1.0, text="Complete!")
    status_text.empty()

    # Store in session state for persistence
    st.session_state["backtest_result"] = result

    # Display results
    _display_results(result)


def _display_results(result: dict):
    """Display backtest results with charts and metrics."""
    from aqtis.dashboard.theme import apply_theme

    st.divider()
    st.subheader("Results")

    metrics = result["metrics"]
    cap = result["capital"]

    # --- Hero Metrics ---
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    pnl_color = "normal" if cap["pnl"] >= 0 else "inverse"
    c1.metric("Final P&L", f"₹{cap['pnl']:,.0f}", f"{cap['return_pct']:+.1f}%", delta_color=pnl_color)
    c2.metric("Total Trades", metrics["total_trades"])
    c3.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    c4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    c5.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}")
    c6.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")

    c7, c8, c9, c10 = st.columns(4)
    c7.metric("Wins", metrics["wins"])
    c8.metric("Losses", metrics["losses"])
    c9.metric("Avg Win", f"₹{metrics['avg_win']:,.0f}")
    c10.metric("Avg Loss", f"₹{metrics['avg_loss']:,.0f}")

    # --- Equity Curve ---
    equity_data = result.get("equity_curve", [])
    if len(equity_data) > 1:
        st.subheader("Equity Curve")
        eq_df = pd.DataFrame(equity_data)
        eq_df["date"] = pd.to_datetime(eq_df["date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq_df["date"],
            y=eq_df["equity"],
            mode="lines",
            name="Portfolio Value",
            line=dict(color=AQTIS_COLORS["blue"], width=2),
            fill="tozeroy",
            fillcolor="rgba(77, 171, 247, 0.1)",
        ))
        fig.add_hline(
            y=cap["initial"],
            line_dash="dash",
            line_color=AQTIS_COLORS["muted"],
            annotation_text="Initial Capital",
        )
        fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            height=400,
            template="plotly_dark",
            paper_bgcolor=AQTIS_COLORS["background"],
            plot_bgcolor=AQTIS_COLORS["card_bg"],
            font=dict(color=AQTIS_COLORS["text"]),
        )
        st.plotly_chart(apply_theme(fig), use_container_width=True)

        # Drawdown chart
        eq_series = eq_df.set_index("date")["equity"]
        peak = eq_series.expanding().max()
        drawdown = (eq_series - peak) / peak * 100

        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode="lines",
            name="Drawdown %",
            line=dict(color=AQTIS_COLORS["red"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 107, 107, 0.15)",
        ))
        fig_dd.update_layout(
            title="Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown %",
            height=250,
            template="plotly_dark",
            paper_bgcolor=AQTIS_COLORS["background"],
            plot_bgcolor=AQTIS_COLORS["card_bg"],
            font=dict(color=AQTIS_COLORS["text"]),
        )
        st.plotly_chart(apply_theme(fig_dd), use_container_width=True)

    # --- Trades Table ---
    trades = result.get("trades", [])
    if trades:
        st.subheader(f"Trade Log ({len(trades)} trades)")

        trades_df = pd.DataFrame(trades)

        # Color-code P&L
        def color_pnl(val):
            color = AQTIS_COLORS["green"] if val > 0 else AQTIS_COLORS["red"]
            return f"color: {color}"

        display_cols = ["symbol", "action", "entry_price", "exit_price", "shares", "pnl", "pnl_percent", "exit_reason", "hold_days", "signal_score"]
        available_cols = [c for c in display_cols if c in trades_df.columns]

        st.dataframe(
            trades_df[available_cols].style.applymap(color_pnl, subset=["pnl"] if "pnl" in available_cols else []),
            use_container_width=True,
            height=min(400, 35 * len(trades_df) + 38),
        )

        # P&L by symbol
        if "symbol" in trades_df.columns and "pnl" in trades_df.columns:
            st.subheader("P&L by Symbol")
            symbol_pnl = trades_df.groupby("symbol")["pnl"].agg(["sum", "count", "mean"]).round(2)
            symbol_pnl.columns = ["Total P&L", "Trades", "Avg P&L"]
            symbol_pnl = symbol_pnl.sort_values("Total P&L", ascending=False)

            fig_bar = go.Figure()
            colors = [AQTIS_COLORS["green"] if v >= 0 else AQTIS_COLORS["red"] for v in symbol_pnl["Total P&L"]]
            fig_bar.add_trace(go.Bar(
                x=symbol_pnl.index,
                y=symbol_pnl["Total P&L"],
                marker_color=colors,
                text=[f"₹{v:,.0f}" for v in symbol_pnl["Total P&L"]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                title="P&L by Symbol",
                xaxis_title="Symbol",
                yaxis_title="Total P&L (₹)",
                height=350,
                template="plotly_dark",
                paper_bgcolor=AQTIS_COLORS["background"],
                plot_bgcolor=AQTIS_COLORS["card_bg"],
                font=dict(color=AQTIS_COLORS["text"]),
            )
            st.plotly_chart(apply_theme(fig_bar), use_container_width=True)

        # Exit reason breakdown
        if "exit_reason" in trades_df.columns:
            st.subheader("Exit Reasons")
            exit_counts = trades_df["exit_reason"].value_counts()
            fig_pie = go.Figure(data=[go.Pie(
                labels=exit_counts.index,
                values=exit_counts.values,
                marker=dict(colors=[
                    AQTIS_COLORS["green"],
                    AQTIS_COLORS["red"],
                    AQTIS_COLORS["yellow"],
                    AQTIS_COLORS["blue"],
                    AQTIS_COLORS["purple"],
                ]),
                hole=0.4,
            )])
            fig_pie.update_layout(
                title="Exit Reason Distribution",
                height=300,
                template="plotly_dark",
                paper_bgcolor=AQTIS_COLORS["background"],
                font=dict(color=AQTIS_COLORS["text"]),
            )
            st.plotly_chart(apply_theme(fig_pie), use_container_width=True)

    # --- Agent Activity Log (Full AQTIS mode) ---
    agent_log = result.get("agent_log", [])
    if agent_log:
        with st.expander(f"Agent Activity Log ({len(agent_log)} events)", expanded=False):
            log_df = pd.DataFrame(agent_log)
            display_cols = [c for c in ["day", "agent", "action", "symbol", "detail"] if c in log_df.columns]
            if display_cols:
                st.dataframe(log_df[display_cols], use_container_width=True, height=300)

    # --- Learning Info (Full AQTIS mode) ---
    learning = result.get("learning", {})
    if learning:
        with st.expander("Learning & Weight Mutations", expanded=False):
            lc1, lc2, lc3 = st.columns(3)
            lc1.metric("Weight Mutations", learning.get("weight_mutations", 0))
            lc2.metric("Final Entry Threshold", f"{learning.get('final_entry_threshold', 0.15):.3f}")
            lc3.metric("Confidence Scaling", f"{learning.get('final_confidence_scaling', 1.0):.2f}")

            top_weights = learning.get("final_weights_sample", {})
            if top_weights:
                st.caption("Top 10 Final Indicator Weights:")
                st.json(top_weights)

    # --- Download Results ---
    st.subheader("Export")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "Download Results (JSON)",
            data=json.dumps(result, indent=2, default=str),
            file_name=f"aqtis_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )
    with col_dl2:
        if trades:
            csv = pd.DataFrame(trades).to_csv(index=False)
            st.download_button(
                "Download Trades (CSV)",
                data=csv,
                file_name=f"aqtis_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Auto-execute when Streamlit discovers this as a page
# ---------------------------------------------------------------------------

def _auto_run():
    """Called when Streamlit runs this file directly as a page."""
    try:
        st.set_page_config(page_title="Run Backtest", page_icon="▶️", layout="wide")
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

    render_live_backtest(_memory)


# When Streamlit auto-discovers this file as a page, it executes it directly.
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
