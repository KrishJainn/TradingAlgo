#!/usr/bin/env python3
"""
LIVE 5-PLAYER PAPER TRADING SYSTEM

Adapts the proven 5-Player + Coach architecture for live paper trading
with 15-minute polling intervals.

Features:
- 5 independent players with diverse strategies (Aggressive, Conservative, etc.)
- Real-time market data via yfinance polling
- 85+ technical indicators
- EOD coach analysis with Gemini LLM
- Bounded strategy patches (±10% weight/threshold changes)
- State persistence for recovery

Usage:
    python live_5player_trader.py --scan          # Single market scan
    python live_5player_trader.py --continuous    # Continuous trading
    python live_5player_trader.py --coach         # Run EOD coach analysis
    python live_5player_trader.py --status        # Show current status
"""

import argparse
import json
import logging
import sys
import time
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, time as dt_time, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import threading

import numpy as np
import pandas as pd
import pytz

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# Core components
from trading_evolution.data.intraday import IntradayDataFetcher, IntradayConfig
from trading_evolution.data.cache import DataCache
from trading_evolution.indicators.universe import IndicatorUniverse
from trading_evolution.indicators.calculator import IndicatorCalculator
from trading_evolution.indicators.normalizer import IndicatorNormalizer
from trading_evolution.super_indicator.dna import (
    SuperIndicatorDNA, IndicatorGene, create_dna_from_weights,
)
from trading_evolution.super_indicator.core import SuperIndicator
from trading_evolution.super_indicator.signals import SignalType, PositionState
from trading_evolution.player.trader import Player
from trading_evolution.player.portfolio import Portfolio, Trade
from trading_evolution.player.risk_manager import RiskManager, RiskParameters
from trading_evolution.player.execution import ExecutionEngine
from trading_evolution.ai_config import AIConfig

# Coach components
try:
    from trading_evolution.ai_coach.post_market_analyzer import PostMarketAnalyzer
    COACH_AVAILABLE = True
except ImportError:
    COACH_AVAILABLE = False

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Intelligent Coach
try:
    from intelligent_coach import IntelligentCoach
    INTELLIGENT_COACH_AVAILABLE = True
except ImportError:
    INTELLIGENT_COACH_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
for _lib in ("httpx", "google_genai", "urllib3", "yfinance"):
    logging.getLogger(_lib).setLevel(logging.ERROR)
logger = logging.getLogger("live_5player")

# ───────────────────────────────────────────────────────────────────────
# TIMEZONE
# ───────────────────────────────────────────────────────────────────────
IST = pytz.timezone("Asia/Kolkata")

# ───────────────────────────────────────────────────────────────────────
# RAW SI CALCULATOR — bypasses double-tanh compression
# ───────────────────────────────────────────────────────────────────────
def raw_weighted_average(normalized_df: pd.DataFrame,
                         weights: Dict[str, float]) -> pd.Series:
    """Weighted average of normalised indicators WITHOUT final tanh."""
    weighted_sum = pd.Series(0.0, index=normalized_df.index)
    total_weight = 0.0
    for name, weight in weights.items():
        if name not in normalized_df.columns or weight == 0:
            continue
        weighted_sum += normalized_df[name].fillna(0) * weight
        total_weight += abs(weight)
    if total_weight == 0:
        return pd.Series(0.0, index=normalized_df.index)
    return (weighted_sum / total_weight).clip(-1.0, 1.0)


# ───────────────────────────────────────────────────────────────────────
# 5 PLAYER CONFIGURATIONS (proven from backtesting)
# ───────────────────────────────────────────────────────────────────────
PLAYERS_CONFIG = {
    "PLAYER_1": {
        "dna_id": "intra_agg1",
        "label": "Aggressive",
        "weights": {
            "RSI_7": 0.90, "STOCH_5_3": 0.85, "TSI_13_25": 0.75,
            "CMO_14": 0.70, "WILLR_14": 0.65, "OBV": 0.60,
            "MFI_14": 0.55, "ADX_14": 0.50, "EMA_9": 0.45,
            "NATR_14": 0.40,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_2": {
        "dna_id": "intra_con2",
        "label": "Conservative",
        "weights": {
            "ADX_14": 0.90, "SUPERTREND_7_3": 0.85, "EMA_20": 0.80,
            "AROON_14": 0.70, "CMF_20": 0.65, "RSI_14": 0.60,
            "BBANDS_20_2": 0.55, "OBV": 0.50, "VWMA_10": 0.45,
            "HMA_9": 0.40, "ATR_14": 0.35,
        },
        "entry_threshold": 0.35,
        "exit_threshold": -0.15,
        "min_hold_bars": 5,
    },
    "PLAYER_3": {
        "dna_id": "intra_bal3",
        "label": "Balanced",
        "weights": {
            "RSI_14": 0.85, "BBANDS_20_2": 0.80, "STOCH_14_3": 0.75,
            "CMF_20": 0.65, "ZSCORE_20": 0.60, "MFI_20": 0.55,
            "EMA_9": 0.50, "ADX_14": 0.45, "ATR_14": 0.40,
            "TEMA_20": 0.35,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_4": {
        "dna_id": "intra_vol4",
        "label": "VolBreakout",
        "weights": {
            "NATR_14": 0.90, "ATR_14": 0.80, "BBANDS_20_2": 0.75,
            "ADX_14": 0.70, "SUPERTREND_7_3": 0.65, "OBV": 0.60,
            "CMF_20": 0.55, "RSI_7": 0.50, "AROON_14": 0.45,
            "PSAR": 0.40,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_5": {
        "dna_id": "intra_mom5",
        "label": "Momentum",
        "weights": {
            "RSI_7": 0.95, "STOCH_5_3": 0.90, "TSI_13_25": 0.80,
            "CMO_14": 0.75, "UO_7_14_28": 0.65, "MFI_14": 0.60,
            "OBV": 0.55, "HMA_9": 0.50, "NATR_14": 0.40,
        },
        "entry_threshold": 0.25,
        "exit_threshold": -0.08,
        "min_hold_bars": 3,
    },
}

# NIFTY 50 symbols
NIFTY25_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS",
    "BAJFINANCE.NS", "WIPRO.NS", "HCLTECH.NS", "NTPC.NS", "POWERGRID.NS",
    "ULTRACEMCO.NS", "TATASTEEL.NS", "ONGC.NS", "JSWSTEEL.NS", "TECHM.NS",
]

# Trading hours
MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)
NO_ENTRY_AFTER = dt_time(14, 45)
EOD_FLATTEN_TIME = dt_time(15, 15)

# Risk parameters
INITIAL_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.015   # 1.5% risk per trade
MAX_POSITION_PCT = 1.0   # 100% - no limit on position size
MAX_CONCURRENT = 100     # Effectively unlimited - player takes all signals
ATR_STOP_MULT = 5.0

# State file
STATE_FILE = Path("live_5player_state.json")


# ───────────────────────────────────────────────────────────────────────
# PLAYER STATE
# ───────────────────────────────────────────────────────────────────────
@dataclass
class LivePlayerState:
    """Per-player mutable state for live trading."""
    player_id: str
    dna_id: str
    label: str
    weights: Dict[str, float]
    entry_threshold: float
    exit_threshold: float
    min_hold_bars: int = 4

    # Computed objects (not serialized)
    dna: SuperIndicatorDNA = field(default=None, repr=False)
    si: SuperIndicator = field(default=None, repr=False)
    player: Player = field(default=None, repr=False)

    # Per-position state
    bars_held: Dict[str, int] = field(default_factory=dict)
    prev_si: Dict[str, float] = field(default_factory=dict)

    # Tracking
    equity: float = INITIAL_CAPITAL
    day_pnl: float = 0.0
    todays_trades: List[Dict] = field(default_factory=list)
    all_trades: List[Dict] = field(default_factory=list)
    equity_history: List[Dict] = field(default_factory=list)
    patches_applied: List[Dict] = field(default_factory=list)
    strategy_version: str = "v1.0"

    def to_dict(self) -> Dict:
        """Serialize to dict for JSON storage."""
        # Serialize portfolio positions if player exists
        positions_data = {}
        if self.player is not None:
            for sym, pos in self.player.portfolio.positions.items():
                positions_data[sym] = {
                    "symbol": pos.symbol,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "quantity": pos.quantity,
                    "stop_loss": getattr(pos, "stop_loss", None),
                    "take_profit": getattr(pos, "take_profit", None),
                    "entry_time": str(pos.entry_time) if pos.entry_time else None,
                    "atr_at_entry": getattr(pos, "atr_at_entry", None),
                    "signal_at_entry": getattr(pos, "signal_at_entry", None),
                }

        return {
            "player_id": self.player_id,
            "dna_id": self.dna_id,
            "label": self.label,
            "weights": self.weights,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "min_hold_bars": self.min_hold_bars,
            "bars_held": self.bars_held,
            "prev_si": self.prev_si,
            "equity": self.equity,
            "day_pnl": self.day_pnl,
            "todays_trades": self.todays_trades,
            "all_trades": self.all_trades[-100:],  # Keep last 100
            "equity_history": self.equity_history[-60:],  # Keep last 60 days
            "patches_applied": self.patches_applied[-20:],  # Keep last 20
            "strategy_version": self.strategy_version,
            "positions": positions_data,  # Save positions!
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "LivePlayerState":
        """Deserialize from dict."""
        return cls(
            player_id=data["player_id"],
            dna_id=data["dna_id"],
            label=data["label"],
            weights=data["weights"],
            entry_threshold=data["entry_threshold"],
            exit_threshold=data["exit_threshold"],
            min_hold_bars=data.get("min_hold_bars", 4),
            bars_held=data.get("bars_held", {}),
            prev_si=data.get("prev_si", {}),
            equity=data.get("equity", INITIAL_CAPITAL),
            day_pnl=data.get("day_pnl", 0.0),
            todays_trades=data.get("todays_trades", []),
            all_trades=data.get("all_trades", []),
            equity_history=data.get("equity_history", []),
            patches_applied=data.get("patches_applied", []),
            strategy_version=data.get("strategy_version", "v1.0"),
        )


# ───────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────
def trade_to_dict(trade: Trade, si_value: float = 0.0) -> Dict:
    """Convert Trade to dict format."""
    return {
        "trade_id": trade.trade_id,
        "symbol": trade.symbol,
        "side": trade.direction,
        "entry_price": trade.entry_price,
        "exit_price": trade.exit_price,
        "pnl": trade.net_pnl,
        "gross_pnl": trade.gross_pnl,
        "si_value": trade.signal_at_entry,
        "exit_si": trade.signal_at_exit,
        "timestamp": str(trade.entry_time),
        "exit_time": str(trade.exit_time),
        "exit_reason": trade.exit_reason,
        "atr": trade.atr_at_entry,
    }


def is_market_open() -> bool:
    """Check if Indian market is open."""
    now = datetime.now(IST)
    if now.weekday() >= 5:  # Weekend
        return False
    current_time = now.time()
    return MARKET_OPEN <= current_time <= MARKET_CLOSE


def get_market_status() -> str:
    """Get human-readable market status."""
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return "CLOSED (Weekend)"
    current_time = now.time()
    if current_time < MARKET_OPEN:
        return "PRE-MARKET"
    elif current_time > MARKET_CLOSE:
        return "CLOSED"
    elif current_time > EOD_FLATTEN_TIME:
        return "EOD FLATTEN"
    elif current_time > NO_ENTRY_AFTER:
        return "NO NEW ENTRIES"
    else:
        return "OPEN"


# ───────────────────────────────────────────────────────────────────────
# LIVE FIVE PLAYER TRADER
# ───────────────────────────────────────────────────────────────────────
class LiveFivePlayerTrader:
    """
    Live paper trading engine with 5 independent players.

    Polls yfinance every 15 minutes during market hours,
    executes trades based on indicator signals, and runs
    EOD coach analysis for strategy improvement.
    """

    def __init__(
        self,
        symbols: List[str] = None,
        poll_interval: int = 15,
        state_file: Path = STATE_FILE,
    ):
        self.symbols = symbols or NIFTY25_SYMBOLS[:15]
        self.poll_interval = poll_interval
        self.state_file = state_file

        # Initialize infrastructure
        self.universe = IndicatorUniverse()
        self.universe.load_all()
        self.calculator = IndicatorCalculator(universe=self.universe)
        self.normalizer = IndicatorNormalizer()

        self.intraday = IntradayDataFetcher(
            config=IntradayConfig(interval="15m"),
            cache=DataCache("data_cache"),
        )

        # Coach
        self.coach = PostMarketAnalyzer(config=AIConfig()) if COACH_AVAILABLE else None

        # Players
        self.players: Dict[str, LivePlayerState] = {}

        # Market data cache
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.indicator_data: Dict[str, pd.DataFrame] = {}

        # State
        self.last_scan_time: Optional[datetime] = None
        self.market_regime: str = "unknown"
        self.running: bool = False

        # Load or initialize state
        self._load_or_init_state()

    def _load_or_init_state(self):
        """Load state from file or initialize fresh."""
        saved_positions = {}  # Store positions to restore after Player init

        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)

                # Try to load from _player_states first (raw state)
                player_states = data.get("_player_states", {})
                if not player_states:
                    # Fallback to players (dashboard format)
                    player_states = data.get("players", {})

                for pid, pdata in player_states.items():
                    self.players[pid] = LivePlayerState.from_dict(pdata)
                    # Store positions for later restoration
                    if "positions" in pdata:
                        saved_positions[pid] = pdata["positions"]

                self.last_scan_time = data.get("last_scan_time")
                if self.last_scan_time:
                    self.last_scan_time = datetime.fromisoformat(self.last_scan_time)

                self.market_regime = data.get("market_regime", "unknown")
                logger.info(f"Loaded state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state: {e}, initializing fresh")
                self._init_players()
        else:
            self._init_players()

        # Initialize player objects (creates fresh Portfolio)
        self._init_player_objects()

        # Restore positions into Portfolio
        self._restore_positions(saved_positions)

    def _restore_positions(self, saved_positions: Dict[str, Dict]):
        """Restore saved positions into Player portfolios."""
        from trading_evolution.player.portfolio import Position

        for pid, positions_data in saved_positions.items():
            if pid not in self.players:
                continue

            st = self.players[pid]
            if st.player is None:
                continue

            restored_count = 0
            for sym, pos_data in positions_data.items():
                try:
                    # Parse entry_time if it's a string
                    entry_time = pos_data.get("entry_time")
                    if entry_time and isinstance(entry_time, str):
                        try:
                            entry_time = datetime.fromisoformat(entry_time.replace("Z", "+00:00"))
                        except:
                            entry_time = datetime.now(IST)
                    elif not entry_time:
                        entry_time = datetime.now(IST)

                    # Calculate default stop loss if not provided
                    entry_price = pos_data["entry_price"]
                    stop_loss = pos_data.get("stop_loss")
                    if not stop_loss:
                        # Default to 2% stop
                        if pos_data["direction"] == "LONG":
                            stop_loss = entry_price * 0.98
                        else:
                            stop_loss = entry_price * 1.02

                    pos = Position(
                        symbol=pos_data["symbol"],
                        direction=pos_data["direction"],
                        entry_price=entry_price,
                        quantity=pos_data["quantity"],
                        entry_time=entry_time,
                        stop_loss=stop_loss,
                    )

                    # Restore optional fields
                    if pos_data.get("take_profit"):
                        pos.take_profit = pos_data["take_profit"]
                    if pos_data.get("atr_at_entry"):
                        pos.atr_at_entry = pos_data["atr_at_entry"]
                    if pos_data.get("signal_at_entry"):
                        pos.signal_strength = pos_data["signal_at_entry"]

                    # Add to portfolio
                    st.player.portfolio.positions[sym] = pos
                    restored_count += 1
                    logger.debug(f"[{pid}] Restored position: {sym} {pos.direction}")
                except Exception as e:
                    logger.warning(f"[{pid}] Failed to restore position {sym}: {e}")

            if restored_count > 0:
                logger.info(f"[{pid}] Restored {restored_count} positions")

    def _init_players(self):
        """Initialize fresh player states from config."""
        for pid, cfg in PLAYERS_CONFIG.items():
            self.players[pid] = LivePlayerState(
                player_id=pid,
                dna_id=cfg["dna_id"],
                label=cfg["label"],
                weights=cfg["weights"].copy(),
                entry_threshold=cfg["entry_threshold"],
                exit_threshold=cfg["exit_threshold"],
                min_hold_bars=cfg.get("min_hold_bars", 4),
            )

    def _init_player_objects(self):
        """Initialize Player objects for each player state."""
        for pid, st in self.players.items():
            # Create DNA using the utility function
            st.dna = create_dna_from_weights(st.weights)
            st.si = SuperIndicator(dna=st.dna)

            # Create Player
            risk_params = RiskParameters(
                max_risk_per_trade=RISK_PER_TRADE,
                max_position_pct=MAX_POSITION_PCT,
                max_concurrent_positions=MAX_CONCURRENT,
                atr_stop_multiplier=ATR_STOP_MULT,
            )

            portfolio = Portfolio(initial_capital=st.equity)
            risk_mgr = RiskManager(risk_params)
            exec_eng = ExecutionEngine(slippage_pct=0.001, commission_pct=0.0005)

            st.player = Player(
                portfolio=portfolio,
                risk_manager=risk_mgr,
                execution=exec_eng,
            )

    def _save_state(self):
        """Save current state to file."""
        # Use get_live_state() for richer data including positions
        live_state = self.get_live_state()

        # Also preserve serializable state for restore
        data = {
            "last_updated": datetime.now(IST).isoformat(),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "market_regime": self.market_regime,
            "market_status": get_market_status(),
            "team_equity": live_state.get("team_equity", 500000),
            "team_day_pnl": live_state.get("team_day_pnl", 0),
            "total_positions": live_state.get("total_positions", 0),
            "total_trades_today": live_state.get("total_trades_today", 0),
            "players": live_state.get("players", {}),
            "equity_history": live_state.get("equity_history", []),
            # Also save raw state for restore
            "_player_states": {pid: st.to_dict() for pid, st in self.players.items()},
        }

        with open(self.state_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.debug(f"State saved to {self.state_file}")

    def _fetch_data(self) -> bool:
        """Fetch latest market data for all symbols."""
        logger.info(f"Fetching data for {len(self.symbols)} symbols...")

        try:
            raw_data = self.intraday.fetch_multiple(
                symbols=self.symbols,
                days=5,  # 5 days for indicator calculation
            )
        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return False

        if not raw_data:
            logger.warning("No data fetched")
            return False

        # Normalize columns
        for sym, df in raw_data.items():
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.columns = [c.lower() for c in df.columns]
            for col in ["dividends", "stock splits", "capital gains"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            self.market_data[sym] = df

        # Calculate indicators
        logger.info("Calculating indicators...")
        for sym, df in self.market_data.items():
            try:
                raw = self.calculator.calculate_all(df)
                raw = self.calculator.rename_to_dna_names(raw)
                self.indicator_data[sym] = raw
            except Exception as e:
                logger.warning(f"Indicator calc failed for {sym}: {e}")

        logger.info(f"Data ready for {len(self.indicator_data)} symbols")
        return len(self.indicator_data) > 0

    def _detect_regime(self) -> str:
        """Detect current market regime based on NIFTY breadth."""
        if not self.indicator_data:
            return "unknown"

        up_count = 0
        down_count = 0
        high_vol = 0

        for sym, raw in self.indicator_data.items():
            if raw.empty:
                continue

            # Get latest values
            latest = raw.iloc[-1] if len(raw) > 0 else None
            if latest is None:
                continue

            # RSI check
            rsi = latest.get("RSI_14", 50)
            if pd.notna(rsi):
                if rsi > 55:
                    up_count += 1
                elif rsi < 45:
                    down_count += 1

            # ADX for volatility
            adx = latest.get("ADX_14", 20)
            if pd.notna(adx) and adx > 25:
                high_vol += 1

        total = len(self.indicator_data)
        if total == 0:
            return "unknown"

        up_pct = up_count / total
        down_pct = down_count / total
        vol_pct = high_vol / total

        if vol_pct > 0.6:
            if up_pct > 0.5:
                return "volatile_bullish"
            elif down_pct > 0.5:
                return "volatile_bearish"
            return "volatile"
        elif up_pct > 0.6:
            return "trending_up"
        elif down_pct > 0.6:
            return "trending_down"
        else:
            return "ranging"

    def _process_signals(self):
        """Process signals for all players across all symbols."""
        today = datetime.now(IST).date()
        current_time = datetime.now(IST).time()

        for sym in self.indicator_data:
            df = self.market_data.get(sym)
            raw = self.indicator_data.get(sym)

            if df is None or raw is None or df.empty or raw.empty:
                continue

            # Get latest bar
            latest_bar = df.iloc[-1]
            latest_ts = df.index[-1]
            close_price = float(latest_bar["close"])

            # ATR for stops
            atr_val = 1.0
            if "ATR_14" in raw.columns:
                a = raw.iloc[-1].get("ATR_14")
                if pd.notna(a) and a > 0:
                    atr_val = float(a)
                else:
                    atr_val = close_price * 0.02
            else:
                atr_val = close_price * 0.02

            # Trend filter
            close_series = df["close"]
            trend_bull = True
            trend_bear = True
            if len(close_series) >= 60:
                ema50 = close_series.ewm(span=50, min_periods=50).mean()
                valid_ema = ema50.dropna()
                if len(valid_ema) >= 10:
                    ema_now = valid_ema.iloc[-1]
                    ema_prev = valid_ema.iloc[-10]
                    price_now = close_series.iloc[-1]
                    slope_up = ema_now > ema_prev
                    price_above = price_now > ema_now
                    if slope_up and price_above:
                        trend_bear = False
                    elif (not slope_up) and (not price_above):
                        trend_bull = False

            # Process each player
            for pid, st in self.players.items():
                # Get active indicators
                active = [i for i in st.dna.get_active_indicators()
                          if i in raw.columns]
                if not active:
                    continue

                # Normalize
                try:
                    norm = self.normalizer.normalize_all(
                        raw[active], price_series=close_series,
                    )
                    if norm.empty:
                        continue
                except Exception:
                    continue

                # Calculate SI
                try:
                    si_series = raw_weighted_average(norm, st.weights).fillna(0.0)
                    si_val = float(si_series.iloc[-1])
                except Exception:
                    continue

                # Get previous SI
                prev_si = st.prev_si.get(sym, 0.0)
                st.prev_si[sym] = si_val

                # Position state
                pos = st.player.portfolio.get_position(sym)
                pos_state = PositionState.FLAT
                if pos:
                    pos_state = (PositionState.LONG
                                 if pos.direction == "LONG"
                                 else PositionState.SHORT)

                # Track holding bars
                if pos_state != PositionState.FLAT:
                    st.bars_held[sym] = st.bars_held.get(sym, 0) + 1
                else:
                    st.bars_held.pop(sym, None)

                held = st.bars_held.get(sym, 0)

                # Signal logic
                signal = SignalType.HOLD
                entry_thresh = st.entry_threshold
                exit_thresh = st.exit_threshold

                if pos_state == PositionState.FLAT:
                    # No entries after cutoff or during flatten
                    if current_time > NO_ENTRY_AFTER:
                        signal = SignalType.HOLD
                    elif (si_val > entry_thresh
                          and prev_si > entry_thresh * 0.7
                          and trend_bull):
                        signal = SignalType.LONG_ENTRY
                    elif (si_val < -entry_thresh
                          and prev_si < -entry_thresh * 0.7
                          and trend_bear):
                        signal = SignalType.SHORT_ENTRY

                elif pos_state == PositionState.LONG:
                    if held >= st.min_hold_bars and si_val < exit_thresh:
                        signal = SignalType.LONG_EXIT

                elif pos_state == PositionState.SHORT:
                    if held >= st.min_hold_bars and si_val > abs(exit_thresh):
                        signal = SignalType.SHORT_EXIT

                # Execute signal
                if signal != SignalType.HOLD:
                    trade = st.player.process_signal(
                        symbol=sym,
                        signal=signal,
                        current_price=close_price,
                        timestamp=latest_ts,
                        high=float(latest_bar["high"]),
                        low=float(latest_bar["low"]),
                        atr=atr_val,
                        si_value=si_val,
                    )

                    if trade:
                        td = trade_to_dict(trade, si_val)
                        st.todays_trades.append(td)
                        st.all_trades.append(td)
                        st.day_pnl += trade.net_pnl
                        st.equity = st.player.portfolio.get_equity()

                        logger.info(
                            f"[{pid}] {trade.direction} {sym} @ {close_price:.2f} "
                            f"SI={si_val:.3f} | PnL: {trade.net_pnl:+.2f}"
                        )

                        if signal in (SignalType.LONG_EXIT, SignalType.SHORT_EXIT):
                            st.bars_held.pop(sym, None)

    def _eod_flatten(self):
        """Close all positions at EOD."""
        logger.info("EOD Flatten: Closing all positions...")

        for pid, st in self.players.items():
            if st.player.portfolio.num_positions == 0:
                continue

            for sym in list(st.player.portfolio.positions.keys()):
                df = self.market_data.get(sym)
                if df is None or df.empty:
                    continue

                close_price = float(df.iloc[-1]["close"])
                pos = st.player.portfolio.get_position(sym)

                if pos:
                    signal = (SignalType.LONG_EXIT if pos.direction == "LONG"
                              else SignalType.SHORT_EXIT)

                    trade = st.player.process_signal(
                        symbol=sym,
                        signal=signal,
                        current_price=close_price,
                        timestamp=datetime.now(IST),
                        high=close_price,
                        low=close_price,
                        atr=close_price * 0.02,
                        si_value=0.0,
                    )

                    if trade:
                        trade.exit_reason = "eod_flatten"
                        td = trade_to_dict(trade)
                        st.todays_trades.append(td)
                        st.all_trades.append(td)
                        st.day_pnl += trade.net_pnl
                        st.equity = st.player.portfolio.get_equity()

                        logger.info(
                            f"[{pid}] EOD FLATTEN {sym} | PnL: {trade.net_pnl:+.2f}"
                        )

    def _record_daily_equity(self):
        """Record daily equity for all players."""
        today = datetime.now(IST).date().isoformat()

        for pid, st in self.players.items():
            st.equity_history.append({
                "date": today,
                "equity": st.equity,
                "day_pnl": st.day_pnl,
                "trades": len(st.todays_trades),
                "version": st.strategy_version,
            })

    def refresh_prices(self) -> bool:
        """Fetch current prices from yfinance without running full scan.

        This is a lightweight operation that only updates current prices
        for dashboard display, without processing signals or trades.
        """
        logger.info("Refreshing current prices from yfinance...")

        try:
            import yfinance as yf

            # Fetch latest prices for all symbols with positions
            symbols_to_fetch = set(self.symbols)

            # Add symbols that have open positions
            for st in self.players.values():
                if st.player:
                    for sym in st.player.portfolio.positions.keys():
                        symbols_to_fetch.add(sym)

            if not symbols_to_fetch:
                return True

            symbols_list = list(symbols_to_fetch)
            logger.info(f"Fetching prices for {len(symbols_list)} symbols...")

            # Fetch using yfinance download
            data = yf.download(
                symbols_list,
                period="1d",
                interval="15m",
                progress=False,
                threads=True,
            )

            if data.empty:
                logger.warning("No price data returned")
                return False

            # Update market_data with latest prices
            updated_count = 0
            for sym in symbols_list:
                try:
                    if len(symbols_list) == 1:
                        # Single symbol - data is already flat
                        sym_data = data.copy()
                    else:
                        # Multi-symbol: yfinance returns MultiIndex (Price, Symbol)
                        if isinstance(data.columns, pd.MultiIndex):
                            # Extract data for this symbol
                            sym_data = data.loc[:, (slice(None), sym)].copy()
                            # Flatten column names - remove the symbol level
                            sym_data.columns = sym_data.columns.droplevel(1)
                        else:
                            sym_data = data.copy()

                    if sym_data is not None and not sym_data.empty:
                        # Store as DataFrame with lowercase columns
                        sym_data.columns = [c.lower() for c in sym_data.columns]
                        self.market_data[sym] = sym_data
                        updated_count += 1

                        # Log latest price
                        if "close" in sym_data.columns:
                            latest_price = sym_data["close"].iloc[-1]
                            logger.debug(f"{sym}: {latest_price:.2f}")

                except Exception as e:
                    logger.warning(f"Failed to get price for {sym}: {e}")

            logger.info(f"Prices refreshed for {updated_count}/{len(symbols_list)} symbols")

            # Update state file with new prices
            self._save_state()

            return updated_count > 0

        except Exception as e:
            logger.error(f"Price refresh failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    def run_scan(self) -> Dict:
        """Run a single market scan."""
        scan_start = datetime.now(IST)
        logger.info(f"Starting scan at {scan_start.strftime('%H:%M:%S')}")

        # Fetch data
        if not self._fetch_data():
            return {"error": "Data fetch failed", "timestamp": scan_start.isoformat()}

        # Detect regime
        self.market_regime = self._detect_regime()
        logger.info(f"Market regime: {self.market_regime}")

        # Process signals
        self._process_signals()

        # Check for EOD flatten
        current_time = datetime.now(IST).time()
        if current_time >= EOD_FLATTEN_TIME:
            self._eod_flatten()
            self._record_daily_equity()

        # Update state
        self.last_scan_time = datetime.now(IST)
        self._save_state()

        # Build result
        result = self.get_live_state()
        result["scan_duration_seconds"] = (datetime.now(IST) - scan_start).total_seconds()

        return result

    def run_continuous(self, interval_minutes: int = None):
        """Run continuous market scanning."""
        interval = interval_minutes or self.poll_interval

        logger.info(f"Starting continuous trading (poll every {interval} minutes)")
        logger.info("Press Ctrl+C to stop")

        self.running = True

        try:
            while self.running:
                # Check market status
                status = get_market_status()

                if status in ("OPEN", "NO NEW ENTRIES", "EOD FLATTEN"):
                    result = self.run_scan()
                    self._print_status()
                else:
                    logger.info(f"Market {status} - skipping scan")

                # Wait for next interval
                logger.info(f"Next scan in {interval} minutes...")
                time.sleep(interval * 60)

        except KeyboardInterrupt:
            logger.info("\nStopping live trader...")
            self._save_state()
            logger.info("State saved. Goodbye!")

    def run_eod_coach(self) -> Dict:
        """Run EOD coach analysis and apply patches using intelligent coach."""
        logger.info("Running EOD Coach Analysis (Intelligent Coach)...")

        results = {}

        # Use intelligent coach if available
        if INTELLIGENT_COACH_AVAILABLE:
            coach = IntelligentCoach()

            for pid, st in self.players.items():
                # Use all trades (today's + historical) for analysis
                all_trades = st.all_trades if st.all_trades else st.todays_trades

                if not all_trades or len(all_trades) < 3:
                    logger.info(f"[{pid}] Insufficient trades for analysis")
                    continue

                try:
                    # Get current config
                    current_thresholds = {
                        "entry": st.entry_threshold,
                        "exit": st.exit_threshold,
                        "min_hold": st.min_hold_bars,
                    }

                    # Get recommendations from intelligent coach
                    rec = coach.analyze_player(
                        player_id=pid,
                        trades=all_trades[-50:],  # Last 50 trades
                        current_weights=st.weights,
                        current_thresholds=current_thresholds,
                        market_regime=self.market_regime,
                    )

                    # Apply recommendations
                    old_config = {
                        "weights": dict(st.weights),
                        "entry_threshold": st.entry_threshold,
                        "exit_threshold": st.exit_threshold,
                        "min_hold_bars": st.min_hold_bars,
                    }
                    new_config = coach.apply_recommendations(old_config, rec)

                    # Update player state
                    st.weights = new_config["weights"]
                    st.entry_threshold = new_config["entry_threshold"]
                    st.exit_threshold = new_config["exit_threshold"]
                    st.min_hold_bars = new_config["min_hold_bars"]

                    # Rebuild DNA and SI with new weights
                    st.dna = create_dna_from_weights(st.weights)
                    st.si = SuperIndicator(dna=st.dna)

                    # Increment strategy version
                    version_parts = st.strategy_version.replace("v", "").split(".")
                    minor = int(version_parts[1]) + 1
                    st.strategy_version = f"v{version_parts[0]}.{minor}"

                    # Record patch
                    patch_record = {
                        "date": datetime.now(IST).isoformat(),
                        "regime": self.market_regime,
                        "win_rate": rec.win_rate,
                        "total_pnl": rec.total_pnl,
                        "indicators_added": list(rec.indicators_to_add.keys()),
                        "indicators_removed": rec.indicators_to_remove,
                        "weights_adjusted": len(rec.weight_adjustments),
                        "new_entry_threshold": st.entry_threshold,
                        "new_version": st.strategy_version,
                    }
                    st.patches_applied.append(patch_record)

                    results[pid] = {
                        "success": True,
                        "win_rate": rec.win_rate,
                        "total_pnl": rec.total_pnl,
                        "indicators_added": list(rec.indicators_to_add.keys()),
                        "indicators_removed": rec.indicators_to_remove,
                        "weights_adjusted": len(rec.weight_adjustments),
                        "new_version": st.strategy_version,
                        "notes": rec.analysis_notes,
                    }

                    logger.info(f"[{pid}] Coach: {rec.analysis_notes}")
                    if rec.indicators_to_add:
                        logger.info(f"[{pid}]   Added: {list(rec.indicators_to_add.keys())}")
                    if rec.indicators_to_remove:
                        logger.info(f"[{pid}]   Removed: {rec.indicators_to_remove}")

                except Exception as e:
                    logger.error(f"[{pid}] Coach analysis failed: {e}")
                    results[pid] = {"error": str(e)}
        else:
            # Fallback to basic analysis
            for pid, st in self.players.items():
                if not st.todays_trades:
                    continue

                trades_summary = {
                    "total": len(st.todays_trades),
                    "wins": sum(1 for t in st.todays_trades if t.get("pnl", 0) > 0),
                    "total_pnl": sum(t.get("pnl", 0) for t in st.todays_trades),
                }
                win_rate = trades_summary["wins"] / max(1, trades_summary["total"])

                results[pid] = {
                    "win_rate": win_rate,
                    "total_pnl": trades_summary["total_pnl"],
                    "note": "Basic analysis (intelligent coach not available)",
                }

                logger.info(
                    f"[{pid}] Basic: {trades_summary['total']} trades, "
                    f"WR: {win_rate:.1%}, P&L: ${trades_summary['total_pnl']:+.2f}"
                )

        # Reset daily counters
        for st in self.players.values():
            st.todays_trades = []
            st.day_pnl = 0.0

        self._save_state()

        return results

    def get_live_state(self) -> Dict:
        """Get current state for dashboard."""
        team_equity = sum(st.equity for st in self.players.values())
        team_day_pnl = sum(st.day_pnl for st in self.players.values())
        total_positions = sum(
            st.player.portfolio.num_positions for st in self.players.values()
        )
        total_trades_today = sum(
            len(st.todays_trades) for st in self.players.values()
        )

        players_data = {}
        for pid, st in self.players.items():
            positions = []
            for sym, pos in st.player.portfolio.positions.items():
                current_price = 0
                if sym in self.market_data:
                    df = self.market_data[sym]
                    if not df.empty:
                        current_price = float(df.iloc[-1]["close"])

                unrealized = 0
                if current_price > 0:
                    if pos.direction == "LONG":
                        unrealized = (current_price - pos.entry_price) * pos.quantity
                    else:
                        unrealized = (pos.entry_price - current_price) * pos.quantity

                positions.append({
                    "symbol": sym,
                    "direction": pos.direction,
                    "entry_price": pos.entry_price,
                    "current_price": current_price,
                    "quantity": pos.quantity,
                    "unrealized_pnl": unrealized,
                    "bars_held": st.bars_held.get(sym, 0),
                })

            # Calculate win rate
            all_trades = st.all_trades
            wins = sum(1 for t in all_trades if t.get("pnl", 0) > 0)
            win_rate = wins / max(1, len(all_trades))

            players_data[pid] = {
                "label": st.label,
                "equity": st.equity,
                "day_pnl": st.day_pnl,
                "positions": positions,
                "num_positions": len(positions),
                "todays_trades": st.todays_trades,
                "win_rate": win_rate,
                "total_trades": len(all_trades),
                "strategy_version": st.strategy_version,
                "weights": st.weights,
                "entry_threshold": st.entry_threshold,
                "exit_threshold": st.exit_threshold,
            }

        return {
            "timestamp": datetime.now(IST).isoformat(),
            "last_scan_time": self.last_scan_time.isoformat() if self.last_scan_time else None,
            "market_status": get_market_status(),
            "market_regime": self.market_regime,
            "team_equity": team_equity,
            "team_day_pnl": team_day_pnl,
            "total_positions": total_positions,
            "total_trades_today": total_trades_today,
            "players": players_data,
            "equity_history": self._get_team_equity_history(),
        }

    def _get_team_equity_history(self) -> List[Dict]:
        """Get combined equity history for all players."""
        history = {}

        for pid, st in self.players.items():
            for record in st.equity_history:
                date = record["date"]
                if date not in history:
                    history[date] = {"date": date}
                history[date][pid] = record["equity"]
                history[date]["team"] = history[date].get("team", 0) + record["equity"]

        return sorted(history.values(), key=lambda x: x["date"])

    def _print_status(self):
        """Print current status to console."""
        state = self.get_live_state()

        print("\n" + "=" * 70)
        print(f"LIVE 5-PLAYER PAPER TRADING | {state['timestamp']}")
        print(f"Market: {state['market_status']} | Regime: {state['market_regime']}")
        print("=" * 70)

        print(f"\nTeam Equity: ${state['team_equity']:,.2f}")
        print(f"Day P&L: ${state['team_day_pnl']:+,.2f}")
        print(f"Open Positions: {state['total_positions']}")
        print(f"Trades Today: {state['total_trades_today']}")

        print("\n" + "-" * 70)
        print(f"{'Player':<12} {'Label':<14} {'Equity':>12} {'Day P&L':>10} {'Pos':>4} {'WR':>6}")
        print("-" * 70)

        for pid, pdata in state["players"].items():
            print(
                f"{pid:<12} {pdata['label']:<14} "
                f"${pdata['equity']:>10,.0f} "
                f"${pdata['day_pnl']:>+8,.0f} "
                f"{pdata['num_positions']:>4} "
                f"{pdata['win_rate']:>5.0%}"
            )

        # Show positions
        all_positions = []
        for pid, pdata in state["players"].items():
            for pos in pdata["positions"]:
                pos["player"] = pid
                all_positions.append(pos)

        if all_positions:
            print("\n" + "-" * 70)
            print("Open Positions:")
            for pos in all_positions:
                pnl_str = f"${pos['unrealized_pnl']:+,.2f}"
                print(
                    f"  {pos['player']:<10} {pos['symbol']:<12} {pos['direction']:<5} "
                    f"@ {pos['entry_price']:.2f} -> {pos['current_price']:.2f} "
                    f"{pnl_str:>10} ({pos['bars_held']} bars)"
                )

        print("=" * 70 + "\n")


# ───────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Live 5-Player Paper Trading System")
    parser.add_argument("--scan", action="store_true", help="Run single market scan")
    parser.add_argument("--continuous", action="store_true", help="Run continuous trading")
    parser.add_argument("--interval", type=int, default=15, help="Poll interval in minutes")
    parser.add_argument("--coach", action="store_true", help="Run EOD coach analysis")
    parser.add_argument("--status", action="store_true", help="Show current status")
    parser.add_argument("--reset", action="store_true", help="Reset all state")
    parser.add_argument("--symbols", type=int, default=15, help="Number of symbols to trade")

    args = parser.parse_args()

    symbols = NIFTY25_SYMBOLS[:args.symbols]
    trader = LiveFivePlayerTrader(symbols=symbols, poll_interval=args.interval)

    if args.reset:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        print("State reset complete!")
        return

    if args.status:
        trader._print_status()
        return

    if args.coach:
        results = trader.run_eod_coach()
        print(json.dumps(results, indent=2, default=str))
        return

    if args.continuous:
        trader.run_continuous()
        return

    # Default: single scan
    result = trader.run_scan()
    trader._print_status()


if __name__ == "__main__":
    main()
