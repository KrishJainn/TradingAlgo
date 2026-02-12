"""
All mathematical constants, NSE cost parameters, and system-wide defaults.

These are *fixed* values that do not change at runtime.
For tunable parameters, see risk_params.py.
"""

from __future__ import annotations

import math
from typing import Dict, Any

# ═══════════════════════════════════════════════════════════════════════════
# Player / Regime constants (mirrors signal_combiner.py)
# ═══════════════════════════════════════════════════════════════════════════

PLAYER_IDS: list[str] = [
    "PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5",
]

PLAYER_LABELS: Dict[str, str] = {
    "PLAYER_1": "TrendFollower",
    "PLAYER_2": "MeanReversion",
    "PLAYER_3": "VolumeFlow",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "MultiMomentum",
}

REGIMES: list[str] = ["Bull", "Bear", "Sideways"]

# ═══════════════════════════════════════════════════════════════════════════
# Mathematical constants
# ═══════════════════════════════════════════════════════════════════════════

EULER_MASCHERONI: float = 0.5772156649015329

TRADING_DAYS_PER_YEAR: int = 252

ANNUALIZATION_FACTOR: float = math.sqrt(TRADING_DAYS_PER_YEAR)

# India 10-year benchmark yield (risk-free proxy)
RISK_FREE_RATE_ANNUAL: float = 0.065

RISK_FREE_RATE_DAILY: float = RISK_FREE_RATE_ANNUAL / TRADING_DAYS_PER_YEAR

# ═══════════════════════════════════════════════════════════════════════════
# NSE Transaction Cost Parameters
# ═══════════════════════════════════════════════════════════════════════════

NSE_COSTS: Dict[str, float] = {
    "brokerage_flat": 20,              # ₹20 flat per order (discount broker)
    "brokerage_pct": 0.0003,           # 0.03% (whichever is lower)
    "stt_intraday_sell": 0.00025,      # 0.025% on sell side
    "stt_delivery_both": 0.001,        # 0.1% on both sides
    "exchange_txn_nse": 0.0000325,     # 0.00325%
    "sebi_fee": 0.000002,             # 0.0002%
    "stamp_duty": 0.00003,            # 0.003% (approximate, varies by state)
    "gst_on_brokerage": 0.18,          # 18% GST on brokerage
    "avg_slippage_liquid_bps": 3,      # 3 bps for NIFTY stocks
    "avg_slippage_midcap_bps": 10,     # 10 bps for mid-caps
    "slippage_highvol_multiplier": 2.5,
}

# ═══════════════════════════════════════════════════════════════════════════
# NSE Market Microstructure — Intraday Slippage Multipliers
# ═══════════════════════════════════════════════════════════════════════════

NSE_MICROSTRUCTURE: Dict[str, float] = {
    "pre_open_multiplier": 3.0,        # 9:15 AM pre-open auction
    "morning_multiplier": 2.5,         # 9:15-10:00 AM
    "late_morning_mult": 1.5,          # 10:00-11:00 AM
    "midday_multiplier": 1.0,          # 11:00 AM - 2:00 PM (calm zone)
    "pre_close_multiplier": 1.3,       # 2:00-2:30 PM
    "closing_multiplier": 2.0,         # 2:30-3:30 PM
    "expiry_day_multiplier": 1.5,      # additional on Thursdays
    "expiry_last_30min_mult": 2.5,     # last 30 min on expiry day
}

# ═══════════════════════════════════════════════════════════════════════════
# Risk Defaults
# ═══════════════════════════════════════════════════════════════════════════

RISK_DEFAULTS: Dict[str, Any] = {
    "kelly_fraction": 0.25,            # Quarter Kelly — conservative
    "max_single_position_pct": 0.20,   # 20% max per instrument
    "max_net_exposure": 1.0,           # 100% net long max
    "max_gross_exposure": 1.5,         # 150% gross max
    "max_daily_loss_pct": 0.02,        # 2% daily loss limit → kill switch
    "max_drawdown_pct": 0.10,          # 10% drawdown → reduce 50%
    "ewma_lambda": 0.94,              # EWMA decay for daily vol
    "vol_lookback": 60,               # days for volatility estimation
    "ic_lookback": 60,                # days for IC calculation
    "cost_safety_margin": 2.0,         # only trade if alpha > 2× cost
    "min_agent_sharpe": -1.0,          # kill agent if rolling SR < -1
    "correlation_lookback": 60,        # days for HRP correlation
    "rebalance_frequency": "daily",    # HRP recomputation frequency
    "tactical_blend": 0.3,             # online vs HRP blend ratio
    "conformal_coverage": 0.90,        # conformal prediction target
    "meta_label_purge": 5,             # bars to purge in CV
}
