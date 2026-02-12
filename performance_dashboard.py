"""
Performance Dashboard — Comprehensive quant-grade monitoring for the
5-Player Coach Trading System.

Streamlit + Plotly app reading from MetricsStore. 7 sections:
    A. Portfolio Overview
    B. Player Performance Cards
    C. Regime Analysis
    D. Signal Quality
    E. Risk Monitor
    F. Coaching & Evolution Log
    G. Equity Curve (full-width)

Run:
    streamlit run performance_dashboard.py
"""

from __future__ import annotations

import math
import random
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

# Project root on path
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from metrics_store import MetricsStore, MetricsCalculator

# ═══════════════════════════════════════════════════════════════════════════
# Theme & constants
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "bg": "#0e1117",
    "card": "#1a1e2e",
    "text": "#fafafa",
    "green": "#00c853",
    "red": "#ff1744",
    "yellow": "#ffd600",
    "blue": "#2979ff",
    "purple": "#7c4dff",
    "muted": "#90a4ae",
    "border": "#37474f",
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zeroline=False),
    margin=dict(l=50, r=20, t=40, b=40),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

PLAYER_IDS = ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]

PLAYER_LABELS = {
    "PLAYER_1": "TrendFollower",
    "PLAYER_2": "MeanReversion",
    "PLAYER_3": "VolumeFlow",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "MultiMomentum",
}

PLAYER_COLORS = {
    "PLAYER_1": "#ff6b6b",
    "PLAYER_2": "#4dabf7",
    "PLAYER_3": "#51cf66",
    "PLAYER_4": "#ffd43b",
    "PLAYER_5": "#da77f2",
}

REGIME_COLORS = {"Bull": "#00c853", "Bear": "#ff1744", "Sideways": "#ffd600"}

SECTORS = [
    "IT", "Banking", "Pharma", "Auto", "FMCG",
    "Energy", "Metals", "Telecom", "Infra", "Other",
]


def _t(fig: go.Figure) -> go.Figure:
    """Apply dark theme to a Plotly figure."""
    fig.update_layout(**PLOTLY_TEMPLATE)
    return fig


def _inr(val: float) -> str:
    if abs(val) >= 1e7:
        return f"₹{val / 1e7:+,.2f} Cr"
    if abs(val) >= 1e5:
        return f"₹{val / 1e5:+,.2f} L"
    return f"₹{val:+,.0f}"


def _pct(val: float) -> str:
    return f"{val:+.2f}%"


def _color(val: float) -> str:
    return COLORS["green"] if val >= 0 else COLORS["red"]


# ═══════════════════════════════════════════════════════════════════════════
# Demo data generator (90 days of synthetic data)
# ═══════════════════════════════════════════════════════════════════════════

def _generate_demo_data(store: MetricsStore, days: int = 90) -> None:
    """
    Generate realistic synthetic data for all dashboard sections.

    Simulates: 5 players with varying performance, regime transitions,
    drawdown periods, coaching sessions, and evolution events.
    """
    random.seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Build a regime sequence with transitions
    regimes = []
    current_regime = "Bull"
    for i in range(days):
        # Regime transition probabilities
        if i < 30:
            current_regime = "Bull"
        elif i < 50:
            current_regime = "Bear"
        elif i < 70:
            current_regime = "Sideways"
        else:
            current_regime = "Bull"
        # Add some noise
        if rng.random() < 0.08:
            current_regime = str(rng.choice(["Bull", "Bear", "Sideways"]))
        regimes.append(current_regime)

    # Player personality biases
    player_biases = {
        "PLAYER_1": {"sharpe_base": 1.6, "wr_base": 0.55, "pnl_scale": 1.2},
        "PLAYER_2": {"sharpe_base": 1.1, "wr_base": 0.58, "pnl_scale": 0.8},
        "PLAYER_3": {"sharpe_base": 0.9, "wr_base": 0.50, "pnl_scale": 1.0},
        "PLAYER_4": {"sharpe_base": 1.3, "wr_base": 0.48, "pnl_scale": 1.5},
        "PLAYER_5": {"sharpe_base": 1.4, "wr_base": 0.52, "pnl_scale": 1.1},
    }

    equity = 500_000.0
    peak_equity = equity
    cum_cost = 0.0
    cum_pnl = 0.0

    sample_stocks = [
        "RELIANCE", "HDFCBANK", "TCS", "INFY", "ICICIBANK",
        "SBIN", "BHARTIARTL", "ITC", "TATASTEEL", "MARUTI",
        "BAJFINANCE", "WIPRO", "SUNPHARMA", "TITAN", "HINDALCO",
    ]
    sample_indicators = ["RSI_7", "MACD_12_26", "EMA_20", "ADX_14",
                         "BBANDS_20", "ATR_14", "VWAP", "OBV"]

    # Evolution log entries
    evolution_entries = [
        {"generation": 1, "date_idx": 15, "changes": "PLAYER_3 swapped RSI_7 for VWAP"},
        {"generation": 2, "date_idx": 30, "changes": "PLAYER_1 crossover with PLAYER_5 indicators"},
        {"generation": 3, "date_idx": 50, "changes": "PLAYER_4 mutated: ATR weight 0.6→0.8"},
        {"generation": 4, "date_idx": 70, "changes": "PLAYER_2 randomized after 3 negative windows"},
        {"generation": 5, "date_idx": 85, "changes": "PLAYER_5 gained BBANDS_20, dropped OBV"},
    ]

    indicator_survival = {ind: int(rng.integers(1, 6)) for ind in sample_indicators}

    for i in range(days):
        d = (date.today() - timedelta(days=days - 1 - i)).isoformat()
        regime = regimes[i]

        # Daily return varies by regime
        regime_base = {"Bull": 0.003, "Bear": -0.002, "Sideways": 0.0005}[regime]
        # Add a drawdown period
        if 40 <= i <= 55:
            regime_base -= 0.004  # forced drawdown
        daily_ret = regime_base + float(rng.normal(0, 0.012))

        daily_pnl_gross = equity * daily_ret
        daily_cost = abs(daily_pnl_gross) * float(rng.uniform(0.015, 0.035))
        daily_pnl_net = daily_pnl_gross - daily_cost
        equity += daily_pnl_net
        peak_equity = max(peak_equity, equity)
        cum_cost += daily_cost
        cum_pnl += daily_pnl_net

        # Regime probabilities
        probs = {"Bull": 0.15, "Bear": 0.15, "Sideways": 0.15}
        probs[regime] = 0.55 + float(rng.uniform(0, 0.15))
        total_p = sum(probs.values())
        probs = {k: round(v / total_p, 3) for k, v in probs.items()}

        # ── Players ──────────────────────────────────────────────────
        players: Dict[str, Any] = {}
        player_cum_pnl_today: Dict[str, float] = {}
        for pid in PLAYER_IDS:
            bias = player_biases[pid]
            noise = float(rng.normal(0, 0.3))
            regime_mod = {"Bull": 0.3, "Bear": -0.4, "Sideways": 0.0}[regime]

            p_sharpe = max(-2, min(5, bias["sharpe_base"] + noise + regime_mod * 0.5))
            p_sortino = max(-2, min(8, p_sharpe * 1.3 + float(rng.normal(0, 0.2))))
            p_wr = max(0.25, min(0.80, bias["wr_base"] + noise * 0.05 + regime_mod * 0.03))
            p_trades = int(rng.integers(1, 20))
            p_pnl = daily_pnl_net * bias["pnl_scale"] * float(rng.uniform(0.05, 0.4))
            if rng.random() < 0.35:
                p_pnl = -abs(p_pnl)
            player_cum_pnl_today[pid] = p_pnl

            # Bayesian weights (regime-dependent)
            w = 0.20 + float(rng.uniform(-0.08, 0.08))
            if regime == "Bull" and pid in ("PLAYER_1", "PLAYER_5"):
                w += 0.05
            elif regime == "Bear" and pid == "PLAYER_2":
                w += 0.07

            # HRP weight
            hrp_w = 0.20 + float(rng.uniform(-0.06, 0.06))

            # Best/worst trade
            best_t = abs(p_pnl) * float(rng.uniform(1.5, 4.0))
            worst_t = -abs(p_pnl) * float(rng.uniform(1.0, 3.0))

            players[pid] = {
                "sharpe": round(p_sharpe, 3),
                "sortino": round(p_sortino, 3),
                "win_rate": round(p_wr, 3),
                "avg_win_loss": round(max(0.4, 1.3 + noise * 0.2), 2),
                "num_trades": p_trades,
                "pnl": round(p_pnl, 2),
                "cumulative_pnl": round(p_pnl * (i + 1) * 0.15, 2),
                "bayesian_weight": round(max(0.05, min(0.40, w)), 3),
                "hrp_weight": round(max(0.05, min(0.40, hrp_w)), 3),
                "best_trade": round(best_t, 2),
                "worst_trade": round(worst_t, 2),
                "active_rules": [
                    f"Avoid {'low-vol' if rng.random() > 0.5 else 'high-spread'} entries",
                    f"Tighten stops in {rng.choice(['Bear', 'Sideways'])} regime",
                    f"{'Scale up' if regime == 'Bull' else 'Scale down'} position sizing",
                ][:3],
                "regime_performance": {
                    "Bull": round(max(-1, min(4, bias["sharpe_base"] + 0.5 + float(rng.normal(0, 0.2)))), 3),
                    "Bear": round(max(-2, min(2, bias["sharpe_base"] - 1.5 + float(rng.normal(0, 0.2)))), 3),
                    "Sideways": round(max(-1, min(3, bias["sharpe_base"] - 0.3 + float(rng.normal(0, 0.2)))), 3),
                },
            }

        # ── Positions ────────────────────────────────────────────────
        n_pos = int(rng.integers(3, 8))
        positions = []
        for j in range(n_pos):
            sym = sample_stocks[j % len(sample_stocks)]
            direction = "LONG" if rng.random() > 0.35 else "SHORT"
            positions.append({
                "symbol": sym,
                "direction": direction,
                "size": int(rng.integers(10, 300)),
                "unrealized_pnl": round(float(rng.uniform(-5000, 8000)), 2),
                "player_id": PLAYER_IDS[j % 5],
            })

        # ── Trades today ─────────────────────────────────────────────
        n_trades = int(rng.integers(0, 8))
        trades_today = []
        for _ in range(n_trades):
            trades_today.append({
                "symbol": str(rng.choice(sample_stocks)),
                "direction": "LONG" if rng.random() > 0.4 else "SHORT",
                "pnl": round(float(rng.uniform(-3000, 4000)), 2),
                "player_id": str(rng.choice(PLAYER_IDS)),
                "cost": round(float(rng.uniform(10, 120)), 2),
            })

        # ── Signals ──────────────────────────────────────────────────
        agreement = max(0.25, min(1.0, 0.60 + float(rng.uniform(-0.20, 0.20))))
        signals = {
            "agreement_rate": round(agreement, 3),
            "pre_debate_accuracy": round(max(0.3, min(0.80, 0.52 + float(rng.uniform(-0.12, 0.12)))), 3),
            "post_debate_accuracy": round(max(0.35, min(0.85, 0.60 + float(rng.uniform(-0.10, 0.10)))), 3),
            "conversion_rate": round(max(0.35, min(0.95, 0.70 + float(rng.uniform(-0.15, 0.15)))), 3),
            "false_signal_rate": {
                pid: round(max(0.03, min(0.40, 0.15 + float(rng.uniform(-0.08, 0.10)))), 3)
                for pid in PLAYER_IDS
            },
        }

        # ── Risk ─────────────────────────────────────────────────────
        current_dd = max(0.0, (peak_equity - equity) / peak_equity)
        sector_exp = {}
        for sec in SECTORS:
            sector_exp[sec] = round(float(rng.uniform(0, 0.18)), 4)

        # CVaR
        cvar_val = max(0.01, min(0.06, 0.025 + float(rng.normal(0, 0.005))))
        cvar_limit = 0.03

        # Vol targeting
        vol_scale = max(0.2, min(2.0, 1.0 + float(rng.normal(0, 0.15))))

        cb_triggers: List[str] = []
        if current_dd > 0.10:
            cb_triggers.append(f"Peak drawdown breached 10% on {d}")
        if i == 48:
            cb_triggers.append(f"Intraday circuit breaker triggered on {d}: -5.2% intraday loss")

        risk = {
            "max_drawdown": round(max(current_dd, 0.03 + float(rng.uniform(0, 0.08))), 4),
            "current_drawdown": round(current_dd, 4),
            "sector_exposure": sector_exp,
            "cost_drag_pct": round(cum_cost / max(1, equity) * 100, 3),
            "circuit_breaker_triggers": cb_triggers,
            "cvar_95": round(cvar_val, 4),
            "cvar_limit": cvar_limit,
            "cvar_breached": cvar_val > cvar_limit,
            "vol_scale_factor": round(vol_scale, 3),
        }

        # ── Coaching ─────────────────────────────────────────────────
        indicator_weights: Dict[str, Dict[str, float]] = {}
        for pid in PLAYER_IDS:
            indicator_weights[pid] = {
                ind: round(float(rng.uniform(0.2, 1.0)), 3)
                for ind in sample_indicators
            }

        coaching = {
            "last_session_summary": (
                f"Regime={regime}. Adjusted configs for all players. "
                f"{'Tightened risk limits due to drawdown.' if current_dd > 0.05 else 'Normal operations.'} "
                f"Boosted {'momentum' if regime == 'Bull' else 'mean-reversion'} indicators."
            ),
            "indicator_weights": indicator_weights,
            "reflection_highlights": [
                f"{PLAYER_LABELS[rng.choice(PLAYER_IDS)]} {'overtraded' if rng.random() > 0.5 else 'missed entries'} in {regime}",
                f"Sector rotation from {rng.choice(SECTORS)} to {rng.choice(SECTORS)} detected",
                f"Stop-loss {'tightened' if regime == 'Bear' else 'widened'} by {int(rng.integers(5, 25))}%",
            ],
        }

        # ── Evolution ────────────────────────────────────────────────
        evo_entry = None
        for e in evolution_entries:
            if e["date_idx"] == i:
                evo_entry = e
                break

        evolution = {
            "generation": max(1, i // 20),
            "latest_change": evo_entry["changes"] if evo_entry else "No changes",
            "indicator_survival": indicator_survival,
        }

        # ── Assemble snapshot ────────────────────────────────────────
        snap = {
            "date": d,
            "portfolio": {
                "gross_pnl": round(daily_pnl_gross, 2),
                "net_pnl": round(daily_pnl_net, 2),
                "cumulative_pnl": round(cum_pnl, 2),
                "gross_exposure": round(equity * float(rng.uniform(0.5, 1.0)), 2),
                "net_exposure": round(equity * float(rng.uniform(0.05, 0.5)), 2),
                "largest_concentration": round(float(rng.uniform(0.06, 0.22)), 4),
                "equity_curve_value": round(equity, 2),
                "daily_return": round(daily_ret, 6),
            },
            "positions": positions,
            "trades_today": trades_today,
            "regime": {
                "current": regime,
                "probabilities": probs,
                "duration_days": int(rng.integers(1, 20)),
            },
            "players": players,
            "signals": signals,
            "risk": risk,
            "coaching": coaching,
            "evolution": evolution,
        }
        store.record_daily_snapshot(snap)


# ═══════════════════════════════════════════════════════════════════════════
# Section A: Portfolio Overview
# ═══════════════════════════════════════════════════════════════════════════

def _render_portfolio_overview(
    latest: Dict, snapshots: List[Dict], selected_players: List[str],
) -> None:
    port = latest.get("portfolio", {})
    regime_data = latest.get("regime", {})

    # ── KPI row ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        gross = port.get("gross_pnl", 0)
        st.metric("Gross P&L (Today)", _inr(gross))
    with c2:
        net = port.get("net_pnl", 0)
        st.metric("Net P&L (Today)", _inr(net))
    with c3:
        cum = port.get("cumulative_pnl", 0)
        st.metric("Cumulative P&L", _inr(cum))
    with c4:
        gr_exp = port.get("gross_exposure", 0)
        st.metric("Gross Exposure", _inr(gr_exp))
    with c5:
        conc = port.get("largest_concentration", 0)
        st.metric("Max Concentration", f"{conc:.1%}")

    # ── Risk metrics row ─────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        net_exp = port.get("net_exposure", 0)
        st.metric("Net Exposure", _inr(net_exp))
    with c2:
        regime = regime_data.get("current", "Unknown")
        regime_color = REGIME_COLORS.get(regime, COLORS["muted"])
        dur = regime_data.get("duration_days", 0)
        probs = regime_data.get("probabilities", {})
        prob_pct = probs.get(regime, 0)
        st.markdown(
            f'<div style="background:{regime_color};color:#000;padding:8px 16px;'
            f'border-radius:8px;font-weight:bold;font-size:18px;text-align:center;">'
            f'{regime} ({prob_pct:.0%}) — {dur}d</div>',
            unsafe_allow_html=True,
        )
    with c3:
        risk = latest.get("risk", {})
        st.metric("Current Drawdown", f"{risk.get('current_drawdown', 0):.2%}")
    with c4:
        st.metric("Equity", _inr(port.get("equity_curve_value", 0)))

    st.markdown("---")

    # ── Positions table ──────────────────────────────────────────────
    positions = latest.get("positions", [])
    if positions:
        st.subheader("Current Positions")
        df = pd.DataFrame(positions)
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        st.dataframe(df, hide_index=True, use_container_width=True)

    # ── Today's trades ───────────────────────────────────────────────
    trades = latest.get("trades_today", [])
    if trades:
        st.subheader("Today's Trades")
        df = pd.DataFrame(trades)
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        st.dataframe(df, hide_index=True, use_container_width=True)
    else:
        st.info("No trades today.")


# ═══════════════════════════════════════════════════════════════════════════
# Section B: Player Performance Cards
# ═══════════════════════════════════════════════════════════════════════════

def _render_player_performance(
    latest: Dict, snapshots: List[Dict], selected_players: List[str],
) -> None:
    players = latest.get("players", {})
    if not players:
        st.info("No player data available.")
        return

    active = [p for p in PLAYER_IDS if p in selected_players]

    # ── Cards row ────────────────────────────────────────────────────
    cols = st.columns(len(active) if active else 1)
    for idx, pid in enumerate(active):
        pdata = players.get(pid, {})
        label = PLAYER_LABELS.get(pid, pid)
        color = PLAYER_COLORS.get(pid, COLORS["muted"])

        with cols[idx]:
            st.markdown(
                f'<div style="border-left:4px solid {color};padding:4px 10px;">'
                f'<b style="font-size:15px;">{label}</b></div>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Sharpe", f"{pdata.get('sharpe', 0):.2f}")
                st.metric("Win Rate", f"{pdata.get('win_rate', 0):.1%}")
                st.metric("Trades", str(pdata.get("num_trades", 0)))
            with m2:
                st.metric("Sortino", f"{pdata.get('sortino', 0):.2f}")
                st.metric("Avg W/L", f"{pdata.get('avg_win_loss', 0):.2f}")
                st.metric("P&L", _inr(pdata.get("pnl", 0)))

            # Weight bars
            bw = pdata.get("bayesian_weight", 0.20)
            hw = pdata.get("hrp_weight", 0.20)
            st.caption(f"Bayesian: {bw:.0%}  |  HRP: {hw:.0%}")
            st.progress(min(1.0, bw / 0.40))

            with st.expander("Details"):
                st.write(f"**Best Trade:** {_inr(pdata.get('best_trade', 0))}")
                st.write(f"**Worst Trade:** {_inr(pdata.get('worst_trade', 0))}")
                rules = pdata.get("active_rules", [])
                if rules:
                    st.write("**Active Rules:**")
                    for r in rules[:3]:
                        st.write(f"- {r}")
                rp = pdata.get("regime_performance", {})
                if rp:
                    st.write("**Regime Sharpe:**")
                    for reg, val in rp.items():
                        rc = REGIME_COLORS.get(reg, COLORS["muted"])
                        st.markdown(f'<span style="color:{rc}">{reg}: {val:+.2f}</span>',
                                    unsafe_allow_html=True)

    # ── Cumulative P&L sparklines ────────────────────────────────────
    st.subheader("Cumulative P&L by Player")
    fig = go.Figure()
    dates = [s["date"] for s in snapshots]
    for pid in active:
        cum_vals = []
        running = 0.0
        for s in snapshots:
            running += s.get("players", {}).get(pid, {}).get("pnl", 0)
            cum_vals.append(running)
        fig.add_trace(go.Scatter(
            x=dates, y=cum_vals,
            mode="lines", name=PLAYER_LABELS.get(pid, pid),
            line=dict(color=PLAYER_COLORS.get(pid), width=2),
        ))
    fig.update_layout(title="", yaxis_title="Cumulative P&L (₹)")
    _t(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Bayesian weight bar chart ────────────────────────────────────
    st.subheader("Current Bayesian Weights")
    bw_vals = [players.get(pid, {}).get("bayesian_weight", 0.2) for pid in active]
    fig = go.Figure(go.Bar(
        x=[PLAYER_LABELS.get(p, p) for p in active],
        y=bw_vals,
        marker_color=[PLAYER_COLORS.get(p) for p in active],
        text=[f"{v:.0%}" for v in bw_vals],
        textposition="outside",
    ))
    fig.update_layout(yaxis_title="Weight", yaxis_range=[0, 0.5])
    _t(fig)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Section C: Regime Analysis
# ═══════════════════════════════════════════════════════════════════════════

def _render_regime_analysis(latest: Dict, snapshots: List[Dict]) -> None:
    regime_data = latest.get("regime", {})

    c1, c2 = st.columns(2)

    # ── Regime probability bar chart ─────────────────────────────────
    with c1:
        probs = regime_data.get("probabilities", {})
        if probs:
            fig = go.Figure(go.Bar(
                x=list(probs.keys()),
                y=list(probs.values()),
                marker_color=[REGIME_COLORS.get(r, COLORS["muted"]) for r in probs],
                text=[f"{v:.0%}" for v in probs.values()],
                textposition="outside",
            ))
            fig.update_layout(title="Current Regime Probabilities",
                              yaxis_title="Probability", yaxis_range=[0, 1])
            _t(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── Time in regime (pie) ─────────────────────────────────────────
    with c2:
        regime_counts: Dict[str, int] = {"Bull": 0, "Bear": 0, "Sideways": 0}
        for s in snapshots:
            r = s.get("regime", {}).get("current", "Sideways")
            if r in regime_counts:
                regime_counts[r] += 1
        fig = go.Figure(go.Pie(
            labels=list(regime_counts.keys()),
            values=list(regime_counts.values()),
            marker=dict(colors=[REGIME_COLORS.get(r) for r in regime_counts]),
            textinfo="label+percent",
            hole=0.4,
        ))
        fig.update_layout(title="Time in Each Regime")
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Regime history timeline ──────────────────────────────────────
    dates = [s["date"] for s in snapshots]
    regime_labels = [s.get("regime", {}).get("current", "Sideways") for s in snapshots]
    if dates:
        regime_num = [{"Bull": 1, "Sideways": 0, "Bear": -1}.get(r, 0) for r in regime_labels]
        # Color-coded bar segments
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dates,
            y=[1] * len(dates),
            marker_color=[REGIME_COLORS.get(r, COLORS["muted"]) for r in regime_labels],
            text=regime_labels,
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        ))
        fig.update_layout(
            title="Regime History Timeline",
            yaxis=dict(visible=False),
            bargap=0,
            height=120,
        )
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Performance by regime table ──────────────────────────────────
    calc = MetricsCalculator
    rows = []
    for regime in ("Bull", "Bear", "Sideways"):
        regime_snaps = [s for s in snapshots if s.get("regime", {}).get("current") == regime]
        if not regime_snaps:
            continue
        rets = np.array([s.get("portfolio", {}).get("daily_return", 0) for s in regime_snaps])
        pnls = np.array([s.get("portfolio", {}).get("net_pnl", 0) for s in regime_snaps])
        rows.append({
            "Regime": regime,
            "Days": len(regime_snaps),
            "Total P&L": _inr(float(np.sum(pnls))),
            "Avg Return": f"{float(np.mean(rets)):.4f}",
            "Sharpe": f"{calc.compute_sharpe(rets):.2f}",
            "Win Rate": f"{calc.compute_win_rate(pnls):.0%}",
        })
    if rows:
        st.subheader("Performance by Regime")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Section D: Signal Quality
# ═══════════════════════════════════════════════════════════════════════════

def _render_signal_quality(
    latest: Dict, snapshots: List[Dict], selected_players: List[str],
) -> None:
    signals = latest.get("signals", {})
    dates = [s["date"] for s in snapshots]

    # ── KPI row ──────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Agreement Rate", f"{signals.get('agreement_rate', 0):.0%}")
    with c2:
        st.metric("Pre-Debate Acc", f"{signals.get('pre_debate_accuracy', 0):.0%}")
    with c3:
        st.metric("Post-Debate Acc", f"{signals.get('post_debate_accuracy', 0):.0%}")
    with c4:
        st.metric("Conversion Rate", f"{signals.get('conversion_rate', 0):.0%}")

    c1, c2 = st.columns(2)

    # ── Agreement rate over time ─────────────────────────────────────
    with c1:
        agree_vals = [s.get("signals", {}).get("agreement_rate", 0) for s in snapshots]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=agree_vals,
            mode="lines", fill="tozeroy",
            line=dict(color=COLORS["blue"], width=2),
            fillcolor="rgba(41,121,255,0.1)",
            name="Agreement",
        ))
        fig.add_hline(y=0.60, line_dash="dash", line_color=COLORS["yellow"],
                      annotation_text="3/5 players = 60%")
        fig.update_layout(title="Player Agreement Rate Over Time",
                          yaxis_title="Rate", yaxis_range=[0, 1])
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Pre vs post debate accuracy ──────────────────────────────────
    with c2:
        pre_vals = [s.get("signals", {}).get("pre_debate_accuracy", 0) for s in snapshots]
        post_vals = [s.get("signals", {}).get("post_debate_accuracy", 0) for s in snapshots]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=pre_vals,
            mode="lines", name="Pre-Debate",
            line=dict(color=COLORS["red"], width=2, dash="dot"),
        ))
        fig.add_trace(go.Scatter(
            x=dates, y=post_vals,
            mode="lines", name="Post-Debate",
            line=dict(color=COLORS["green"], width=2),
        ))
        fig.update_layout(title="Signal Accuracy: Pre vs Post Debate",
                          yaxis_title="Accuracy", yaxis_range=[0, 1])
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── False signal rate per player ─────────────────────────────────
    false_rates = signals.get("false_signal_rate", {})
    if false_rates:
        active = [p for p in PLAYER_IDS if p in selected_players]
        fig = go.Figure(go.Bar(
            x=[PLAYER_LABELS.get(p, p) for p in active],
            y=[false_rates.get(p, 0) for p in active],
            marker_color=[PLAYER_COLORS.get(p) for p in active],
            text=[f"{false_rates.get(p, 0):.0%}" for p in active],
            textposition="outside",
        ))
        fig.update_layout(title="False Signal Rate by Player",
                          yaxis_title="Rate", yaxis_range=[0, 0.5])
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Conversion rate over time ────────────────────────────────────
    conv_vals = [s.get("signals", {}).get("conversion_rate", 0) for s in snapshots]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=conv_vals,
        mode="lines+markers", fill="tozeroy",
        line=dict(color=COLORS["purple"], width=2),
        fillcolor="rgba(124,77,255,0.1)",
        marker=dict(size=3),
        name="Conversion",
    ))
    fig.update_layout(title="Signal-to-Trade Conversion Rate",
                      yaxis_title="Rate", yaxis_range=[0, 1])
    _t(fig)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Section E: Risk Monitor
# ═══════════════════════════════════════════════════════════════════════════

def _render_risk_monitor(latest: Dict, snapshots: List[Dict]) -> None:
    risk = latest.get("risk", {})
    dates = [s["date"] for s in snapshots]

    # ── KPI row ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Max Drawdown", f"{risk.get('max_drawdown', 0):.2%}")
    with c2:
        st.metric("Current DD", f"{risk.get('current_drawdown', 0):.2%}")
    with c3:
        st.metric("Cost Drag", f"{risk.get('cost_drag_pct', 0):.2f}%")
    with c4:
        cvar = risk.get("cvar_95", 0)
        cvar_limit = risk.get("cvar_limit", 0.03)
        breach_txt = "BREACHED" if cvar > cvar_limit else "OK"
        st.metric(f"CVaR 95% ({breach_txt})", f"{cvar:.2%}")
    with c5:
        triggers = risk.get("circuit_breaker_triggers", [])
        st.metric("Circuit Breakers", str(len(triggers)))

    # ── Drawdown curve (underwater plot) ─────────────────────────────
    equity_vals = [s.get("portfolio", {}).get("equity_curve_value", 0) for s in snapshots]
    if equity_vals:
        eq_arr = np.array(equity_vals)
        peak = np.maximum.accumulate(eq_arr)
        dd_pct = np.where(peak > 0, (eq_arr - peak) / peak * 100, 0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=dd_pct.tolist(),
            mode="lines", fill="tozeroy",
            line=dict(color=COLORS["red"], width=2),
            fillcolor="rgba(255,23,68,0.15)",
            name="Drawdown %",
        ))
        fig.add_hline(y=-12, line_dash="dash", line_color=COLORS["yellow"],
                      annotation_text="Circuit Breaker (-12%)")
        fig.update_layout(title="Underwater Plot (Drawdown from Peak)",
                          yaxis_title="Drawdown %")
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    # ── Sector exposure heatmap ──────────────────────────────────────
    with c1:
        sector_exp = risk.get("sector_exposure", {})
        if sector_exp:
            secs = list(sector_exp.keys())
            vals = [sector_exp[s] * 100 for s in secs]
            fig = go.Figure(go.Bar(
                x=secs, y=vals,
                marker_color=[COLORS["blue"] if v < 15 else COLORS["yellow"] for v in vals],
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
            ))
            fig.update_layout(title="Sector Exposure (%)", yaxis_title="%")
            _t(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── Position concentration ───────────────────────────────────────
    with c2:
        positions = latest.get("positions", [])
        if positions:
            syms = [p["symbol"] for p in positions]
            sizes = [abs(p.get("unrealized_pnl", 0)) for p in positions]
            dirs = [p.get("direction", "LONG") for p in positions]
            bar_colors = [COLORS["green"] if d == "LONG" else COLORS["red"] for d in dirs]
            fig = go.Figure(go.Bar(
                x=syms, y=sizes, marker_color=bar_colors,
                text=[_inr(s) for s in sizes], textposition="outside",
            ))
            fig.update_layout(title="Position Concentration (|Unrealized P&L|)",
                              yaxis_title="₹")
            _t(fig)
            st.plotly_chart(fig, use_container_width=True)

    # ── Cost drag over time ──────────────────────────────────────────
    cost_vals = [s.get("risk", {}).get("cost_drag_pct", 0) for s in snapshots]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=cost_vals,
        mode="lines+markers",
        line=dict(color=COLORS["yellow"], width=2),
        marker=dict(size=3),
        name="Cost Drag %",
    ))
    fig.update_layout(title="Cumulative Transaction Cost Drag Over Time",
                      yaxis_title="%")
    _t(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)

    # ── CVaR gauge ───────────────────────────────────────────────────
    with c1:
        cvar_val = risk.get("cvar_95", 0)
        cvar_lim = risk.get("cvar_limit", 0.03)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=cvar_val * 100,
            title=dict(text="CVaR 95% vs Limit"),
            number=dict(suffix="%"),
            gauge=dict(
                axis=dict(range=[0, 6]),
                bar=dict(color=COLORS["red"] if cvar_val > cvar_lim else COLORS["green"]),
                threshold=dict(line=dict(color=COLORS["yellow"], width=4),
                               thickness=0.75, value=cvar_lim * 100),
                steps=[
                    dict(range=[0, cvar_lim * 100], color="rgba(0,200,83,0.15)"),
                    dict(range=[cvar_lim * 100, 6], color="rgba(255,23,68,0.15)"),
                ],
            ),
        ))
        fig.update_layout(height=250)
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Vol targeting scale factor over time ─────────────────────────
    with c2:
        vol_vals = [s.get("risk", {}).get("vol_scale_factor", 1.0) for s in snapshots]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=vol_vals,
            mode="lines",
            line=dict(color=COLORS["purple"], width=2),
            name="Vol Scale Factor",
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["muted"],
                      annotation_text="Neutral (1.0x)")
        fig.update_layout(title="Volatility Targeting Scale Factor",
                          yaxis_title="Scale Factor")
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    # ── Circuit breaker log ──────────────────────────────────────────
    all_triggers: List[str] = []
    for s in reversed(snapshots):
        for t in s.get("risk", {}).get("circuit_breaker_triggers", []):
            all_triggers.append(t)
    if all_triggers:
        with st.expander(f"Circuit Breaker Log ({len(all_triggers)} events)"):
            for t in all_triggers[:30]:
                st.write(f"- {t}")
    else:
        st.success("No circuit breaker triggers in the observed period.")


# ═══════════════════════════════════════════════════════════════════════════
# Section F: Coaching & Evolution Log
# ═══════════════════════════════════════════════════════════════════════════

def _render_coaching_log(
    latest: Dict, snapshots: List[Dict], selected_players: List[str],
) -> None:
    coaching = latest.get("coaching", {})
    evolution = latest.get("evolution", {})

    # ── Last coaching session ────────────────────────────────────────
    summary = coaching.get("last_session_summary", "No coaching session recorded.")
    st.markdown(f"**Last Coaching Session:** {summary}")

    # ── Reflection highlights ────────────────────────────────────────
    highlights = coaching.get("reflection_highlights", [])
    if highlights:
        st.subheader("Reflection Highlights")
        for h in highlights:
            st.write(f"- {h}")

    st.markdown("---")

    # ── Evolution log ────────────────────────────────────────────────
    st.subheader("Evolution Log")
    gen = evolution.get("generation", 0)
    latest_change = evolution.get("latest_change", "No changes")
    st.write(f"**Generation:** {gen}")
    st.write(f"**Latest:** {latest_change}")

    # ── Indicator survival rates ─────────────────────────────────────
    survival = evolution.get("indicator_survival", {})
    if survival:
        st.subheader("Indicator Survival Rates")
        fig = go.Figure(go.Bar(
            x=list(survival.keys()),
            y=list(survival.values()),
            marker_color=COLORS["blue"],
            text=[f"{v}" for v in survival.values()],
            textposition="outside",
        ))
        fig.update_layout(title="Generations Survived per Indicator",
                          yaxis_title="Generations")
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Indicator weight trends per player ───────────────────────────
    st.subheader("Indicator Weight Trends")
    active = [p for p in PLAYER_IDS if p in selected_players]
    color_cycle = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    ]

    for pid in active:
        label = PLAYER_LABELS.get(pid, pid)
        with st.expander(f"{label} — Indicator Weights"):
            plot_dates: List[str] = []
            indicator_series: Dict[str, List[float]] = {}

            for s in snapshots:
                d = s.get("date", "")
                weights = s.get("coaching", {}).get("indicator_weights", {}).get(pid, {})
                if not weights:
                    continue
                plot_dates.append(d)
                for ind in list(indicator_series.keys()) + [k for k in weights if k not in indicator_series]:
                    indicator_series.setdefault(ind, [None] * (len(plot_dates) - 1))
                    if ind in weights:
                        indicator_series[ind].append(weights[ind])
                    else:
                        indicator_series[ind].append(None)

            if plot_dates and indicator_series:
                fig = go.Figure()
                for idx, (ind, vals) in enumerate(indicator_series.items()):
                    fig.add_trace(go.Scatter(
                        x=plot_dates, y=vals, connectgaps=True,
                        mode="lines", name=ind,
                        line=dict(color=color_cycle[idx % len(color_cycle)], width=2),
                    ))
                fig.update_layout(
                    yaxis_title="Weight", yaxis_range=[0, 1.1],
                    legend=dict(orientation="h", yanchor="bottom", y=-0.3),
                    height=350,
                )
                _t(fig)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No indicator weight history available.")


# ═══════════════════════════════════════════════════════════════════════════
# Section G: Equity Curve (full-width)
# ═══════════════════════════════════════════════════════════════════════════

def _render_equity_curve(latest: Dict, snapshots: List[Dict], store: MetricsStore) -> None:
    dates = [s["date"] for s in snapshots]
    equity_vals = [s.get("portfolio", {}).get("equity_curve_value", 0) for s in snapshots]

    if not equity_vals:
        st.info("No equity data available.")
        return

    # ── Main equity curve with benchmark ─────────────────────────────
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Cumulative P&L vs Nifty 50 Benchmark", "Rolling 30-Day Sharpe"),
    )

    # Normalize to base 100 for comparison
    base = equity_vals[0] if equity_vals[0] > 0 else 1.0
    norm_equity = [v / base * 100 for v in equity_vals]

    # Synthetic Nifty benchmark (baseline + noise)
    np.random.seed(99)
    nifty_base = 100.0
    nifty_vals = [nifty_base]
    for i in range(1, len(dates)):
        nifty_base *= (1 + np.random.normal(0.0003, 0.01))
        nifty_vals.append(nifty_base)

    fig.add_trace(go.Scatter(
        x=dates, y=norm_equity,
        mode="lines", name="Portfolio",
        line=dict(color=COLORS["blue"], width=2.5),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=nifty_vals,
        mode="lines", name="Nifty 50 (Synthetic)",
        line=dict(color=COLORS["muted"], width=1.5, dash="dot"),
    ), row=1, col=1)

    # ── Rolling 30-day Sharpe subplot ────────────────────────────────
    rets = np.array([s.get("portfolio", {}).get("daily_return", 0) for s in snapshots])
    rolling_sharpe = MetricsCalculator.compute_rolling_sharpe(rets, window=min(30, max(5, len(rets) - 1)))

    if len(rolling_sharpe) > 0:
        sharpe_dates = dates[len(dates) - len(rolling_sharpe):]
        fig.add_trace(go.Scatter(
            x=sharpe_dates, y=rolling_sharpe.tolist(),
            mode="lines", name="Rolling Sharpe",
            line=dict(color=COLORS["purple"], width=2),
        ), row=2, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color=COLORS["muted"], row=2, col=1)
        fig.add_hline(y=1.5, line_dash="dash", line_color=COLORS["green"],
                      annotation_text="Target (1.5)", row=2, col=1)

    fig.update_layout(height=600)
    fig.update_yaxes(title_text="Indexed (Base=100)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=2, col=1)
    _t(fig)
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly returns heatmap ──────────────────────────────────────
    st.subheader("Monthly Returns Heatmap")
    monthly = store.compute_monthly_returns(days=365)
    if monthly:
        # Build matrix: rows=years, cols=months
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        all_years = sorted(set(ym[:4] for ym in monthly.keys()))

        z_data: List[List[float]] = []
        hover_data: List[List[str]] = []
        for year in all_years:
            row = []
            hover_row = []
            for m_idx in range(1, 13):
                ym_key = f"{year}-{m_idx:02d}"
                pnl = monthly.get(ym_key, {}).get("pnl", 0)
                row.append(pnl)
                hover_row.append(f"{month_names[m_idx-1]} {year}: {_inr(pnl)}")
            z_data.append(row)
            hover_data.append(hover_row)

        fig = go.Figure(go.Heatmap(
            z=z_data,
            x=month_names,
            y=all_years,
            text=hover_data,
            hoverinfo="text",
            colorscale=[
                [0, COLORS["red"]],
                [0.5, "#1a1e2e"],
                [1, COLORS["green"]],
            ],
            zmid=0,
            texttemplate="%{z:,.0f}",
            textfont=dict(size=10),
        ))
        fig.update_layout(title="", height=max(200, len(all_years) * 60 + 80))
        _t(fig)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def _render_sidebar(store: MetricsStore, snapshots: List[Dict]) -> Tuple[int, List[str], bool]:
    """Render sidebar controls and key metrics. Returns (lookback, selected_players, auto_refresh)."""
    st.sidebar.title("📊 Dashboard Controls")

    # Lookback selector
    lookback = st.sidebar.selectbox(
        "Lookback Period",
        [7, 14, 30, 60, 90, 180],
        index=2,
        format_func=lambda x: f"{x} days",
    )

    # Player filter
    selected_players = st.sidebar.multiselect(
        "Players",
        PLAYER_IDS,
        default=PLAYER_IDS,
        format_func=lambda x: PLAYER_LABELS.get(x, x),
    )

    # Regime filter
    regime_filter = st.sidebar.multiselect(
        "Regime Filter",
        ["Bull", "Bear", "Sideways"],
        default=["Bull", "Bear", "Sideways"],
    )

    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (60s)", value=False)

    # Export
    st.sidebar.markdown("---")
    if st.sidebar.button("📥 Export CSV"):
        csv_data = store.export_csv(days=lookback)
        st.sidebar.download_button(
            "Download CSV",
            data=csv_data,
            file_name=f"metrics_export_{date.today().isoformat()}.csv",
            mime="text/csv",
        )

    # ── Key Metrics sidebar ──────────────────────────────────────────
    st.sidebar.markdown("---")
    st.sidebar.subheader("Key Metrics")

    if snapshots:
        rets = np.array([s.get("portfolio", {}).get("daily_return", 0) for s in snapshots])
        eq = np.array([s.get("portfolio", {}).get("equity_curve_value", 0) for s in snapshots])
        net_pnls = np.array([s.get("portfolio", {}).get("net_pnl", 0) for s in snapshots])
        gross_pnls = np.array([s.get("portfolio", {}).get("gross_pnl", 0) for s in snapshots])

        calc = MetricsCalculator
        sharpe = calc.compute_sharpe(rets)
        sortino = calc.compute_sortino(rets)
        max_dd = calc.compute_max_drawdown(eq) if len(eq) > 1 else 0
        calmar = calc.compute_calmar(rets)
        pf = calc.compute_profit_factor(net_pnls)
        wl = calc.compute_avg_win_loss_ratio(net_pnls)
        wr = calc.compute_win_rate(net_pnls)
        cd = calc.compute_cost_drag(float(np.sum(gross_pnls)), float(np.sum(net_pnls)))

        metrics_list = [
            ("Sharpe", f"{sharpe:.2f}", sharpe >= 1.5),
            ("Sortino", f"{sortino:.2f}", sortino >= 2.0),
            ("Calmar", f"{calmar:.2f}", calmar >= 1.0),
            ("Max DD", f"{max_dd:.1%}", max_dd < 0.15),
            ("Profit Factor", f"{pf:.2f}", pf >= 1.5),
            ("Win Rate", f"{wr:.0%}", wr >= 0.50),
            ("Avg W/L", f"{wl:.2f}", wl >= 1.3),
            ("Cost Drag", f"{cd:.1f}%", cd < 5.0),
        ]

        for name, val_str, is_good in metrics_list:
            color = COLORS["green"] if is_good else COLORS["red"]
            st.sidebar.markdown(
                f'<span style="color:{COLORS["muted"]};font-size:11px;">{name}</span><br>'
                f'<span style="color:{color};font-weight:bold;font-size:15px;">{val_str}</span>',
                unsafe_allow_html=True,
            )
            st.sidebar.markdown("")

    # Last updated
    st.sidebar.markdown("---")
    latest = store.get_latest()
    st.sidebar.caption(f"Last snapshot: {latest.get('date', 'N/A')}")
    st.sidebar.caption(f"Snapshots loaded: {len(snapshots)}")

    return lookback, selected_players, auto_refresh


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="5-Player Coach — Performance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Dark theme CSS
    st.markdown("""
    <style>
        .stApp { background-color: #0e1117; }
        [data-testid="stSidebar"] { background-color: #161b22; }
        [data-testid="stMetricValue"] { font-weight: bold; }
        .stTabs [data-baseweb="tab"] { font-weight: 600; }
        section[data-testid="stSidebar"] .stMarkdown { color: #c9d1d9; }
    </style>
    """, unsafe_allow_html=True)

    st.title("5-Player Coach — Performance Dashboard")

    store = MetricsStore()

    # ── Demo mode: generate data if empty ────────────────────────────
    if store.snapshot_count() == 0:
        with st.spinner("No metrics data found. Generating 90 days of demo data..."):
            _generate_demo_data(store, days=90)
        st.toast("Demo data generated — 90 days of synthetic trading data.", icon="✅")

    # ── Sidebar ──────────────────────────────────────────────────────
    # Pre-load all snapshots to pass to sidebar
    all_snaps_initial = store.get_all_snapshots(days=90)
    lookback, selected_players, auto_refresh = _render_sidebar(store, all_snaps_initial)

    # Reload with actual lookback
    snapshots = store.get_all_snapshots(days=lookback)
    latest = store.get_latest()

    if not latest:
        st.warning("No metric snapshots available.")
        return

    # Manual refresh
    if st.button("🔄 Refresh Dashboard"):
        st.rerun()

    # ── Main tabs ────────────────────────────────────────────────────
    tab_a, tab_b, tab_c, tab_d, tab_e, tab_f, tab_g = st.tabs([
        "📋 Portfolio Overview",
        "👥 Player Performance",
        "📈 Regime Analysis",
        "🎯 Signal Quality",
        "🛡️ Risk Monitor",
        "🧠 Coaching & Evolution",
        "📊 Equity Curve",
    ])

    with tab_a:
        _render_portfolio_overview(latest, snapshots, selected_players)
    with tab_b:
        _render_player_performance(latest, snapshots, selected_players)
    with tab_c:
        with st.expander("Regime Analysis", expanded=True):
            _render_regime_analysis(latest, snapshots)
    with tab_d:
        with st.expander("Signal Quality", expanded=True):
            _render_signal_quality(latest, snapshots, selected_players)
    with tab_e:
        with st.expander("Risk Monitor", expanded=True):
            _render_risk_monitor(latest, snapshots)
    with tab_f:
        with st.expander("Coaching & Evolution Log", expanded=True):
            _render_coaching_log(latest, snapshots, selected_players)
    with tab_g:
        _render_equity_curve(latest, snapshots, store)

    # ── Auto-refresh ─────────────────────────────────────────────────
    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == "__main__":
    main()
