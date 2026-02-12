"""
Paper Trading Pipeline — Configuration.

Central configuration for the daily paper trading pipeline.
All runtime parameters, file paths, and universe definitions live here.

Usage:
    from paper_trading_config import CONFIG, UNIVERSE, PLAYER_IDS
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

# ═══════════════════════════════════════════════════════════════════════════
# Project root
# ═══════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(__file__).parent

# ═══════════════════════════════════════════════════════════════════════════
# Player definitions (must match the rest of the system)
# ═══════════════════════════════════════════════════════════════════════════

PLAYER_IDS = ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]

PLAYER_LABELS: Dict[str, str] = {
    "PLAYER_1": "TrendFollower",
    "PLAYER_2": "MeanReversion",
    "PLAYER_3": "VolumeFlow",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "MultiMomentum",
}

# ═══════════════════════════════════════════════════════════════════════════
# Trading universe — All 50 Nifty 50 stocks
# ═══════════════════════════════════════════════════════════════════════════

UNIVERSE: List[str] = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "HINDUNILVR.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "ITC.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "AXISBANK.NS",
    "BAJFINANCE.NS",
    "ASIANPAINT.NS",
    "MARUTI.NS",
    "TITAN.NS",
    "SUNPHARMA.NS",
    "ULTRACEMCO.NS",
    "NESTLEIND.NS",
    "WIPRO.NS",
    "TATAMOTORS.NS",
    "BAJAJFINSV.NS",
    "NTPC.NS",
    "TATASTEEL.NS",
    "POWERGRID.NS",
    "M&M.NS",
    "ADANIENT.NS",
    "HCLTECH.NS",
    "ONGC.NS",
    "COALINDIA.NS",
    "JSWSTEEL.NS",
    "ADANIPORTS.NS",
    "TECHM.NS",
    "DRREDDY.NS",
    "INDUSINDBK.NS",
    "CIPLA.NS",
    "GRASIM.NS",
    "DIVISLAB.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "EICHERMOT.NS",
    "APOLLOHOSP.NS",
    "TATACONSUM.NS",
    "SBILIFE.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS",
    "HINDALCO.NS",
    "LTIM.NS",
    "SHRIRAMFIN.NS",
]

# ═══════════════════════════════════════════════════════════════════════════
# Master configuration dict
# ═══════════════════════════════════════════════════════════════════════════

CONFIG: Dict[str, Any] = {
    # ── Capital & Sizing ──────────────────────────────────────────────
    "starting_capital": 500_000,             # ₹5 lakh (₹1L × 5 players)
    "num_players": 5,                       # 5 independent players
    "capital_per_player": 100_000,          # ₹1 lakh per player
    "max_positions": 15,                    # max simultaneous open positions
    "max_single_stock_pct": 0.25,           # 25% max concentration per stock

    # ── Universe ──────────────────────────────────────────────────────
    "universe": UNIVERSE,

    # ── LLM / AI ──────────────────────────────────────────────────────
    "gemini_model": "gemini-2.5-flash",     # for debate, reflection, coaching
    "gemini_timeout": 30,                   # seconds per API call
    "max_api_calls_per_run": 50,            # budget for Gemini calls per day

    # ── Schedule — 2 daily runs (IST) ──────────────────────────────
    "schedule": {
        "premarket": "09:00",               # Signals + debates + execute trades
        "postmarket": "15:45",              # Update prices + stops + reflect/coach
    },
    "run_time": "15:45",                    # Legacy — kept for backward compat
    "timezone": "Asia/Kolkata",

    # ── Data lookback ─────────────────────────────────────────────────
    "ohlcv_lookback_days": 120,             # days of OHLCV for regime/signals
    "regime_lookback": 60,                  # bars for regime prediction
    "sentiment_lookback_days": 5,           # days for news sentiment

    # ── Risk management ───────────────────────────────────────────────
    "trailing_stop_pct": 0.02,              # 2% trailing stop-loss
    "max_drawdown_pct": 0.12,               # 12% portfolio circuit breaker
    "volatility_target": 0.15,              # 15% annualised vol target

    # ── Reflection / coaching / evolution triggers ────────────────────
    "reflection_every_n_trades": 5,         # reflect after every 5 trades
    "coaching_min_trades": 10,              # min trades before coaching cycle
    "evolution_interval_days": 120,         # evolve player genomes every 120d

    # ── File paths ────────────────────────────────────────────────────
    "metrics_dir": str(PROJECT_ROOT / "metrics_data"),
    "positions_file": str(PROJECT_ROOT / "paper_data" / "positions.json"),
    "journal_file": str(PROJECT_ROOT / "paper_data" / "trade_journal.json"),
    "state_file": str(PROJECT_ROOT / "paper_data" / "pipeline_state.json"),
    "evolved_configs_file": str(PROJECT_ROOT / "evolved_player_configs.json"),
    "signal_combiner_state_dir": str(PROJECT_ROOT / "signal_combiner_state"),
    "daily_reports_dir": str(PROJECT_ROOT / "paper_data" / "daily_reports"),
    "log_file": str(PROJECT_ROOT / "paper_data" / "paper_trading.log"),

    # ── Transaction costs (Indian market) ─────────────────────────────
    "brokerage_pct": 0.0003,                # 0.03% brokerage
    "avg_daily_volume": 5_000_000,          # assumed avg volume for cost calc
    "signal_type": "momentum",              # default signal type for cost model

    # ── Paper execution ───────────────────────────────────────────────
    "paper_mode": True,                     # paper trading (no real orders)
    "slippage_bps": 5,                      # 5 bps additional slippage in paper

    # ── EVOLVED+ENSEMBLE — evolution & consensus sizing ──────────────
    "evolution": {
        "calendar_evolution_days": 25,          # Baseline evolution frequency
        "min_days_between_evolution": 10,        # Cooldown after regime-triggered evolution
        "min_trading_days_for_evolution": 15,    # Minimum data before evolving
        "ensemble_lonely_multiplier": 0.5,      # Position size for minority signals (0-1 agree)
        "ensemble_standard_multiplier": 1.0,    # Position size for standard consensus (2 agree)
        "ensemble_strong_multiplier": 1.2,      # Position size for strong majority (3+ agree)
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Helper — ensure data directories exist
# ═══════════════════════════════════════════════════════════════════════════

def ensure_directories() -> None:
    """Create all required data directories if they don't exist."""
    dirs = [
        Path(CONFIG["metrics_dir"]),
        Path(CONFIG["positions_file"]).parent,
        Path(CONFIG["daily_reports_dir"]),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
