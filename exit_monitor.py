"""
Exit Monitor — Standalone SmartExitEngine running every 30 seconds.

Runs independently of the main pipeline. During NSE market hours
(09:15–15:30 IST), fetches live prices and evaluates all open positions
using the rule-based SmartExitEngine.  No LLM / Gemini calls — pure math.

Hard stops, staged profit-taking, and weighted voting exits fire
immediately when triggered.  Gemini consultation stays in the 3 daily
pipeline runs only.

Usage:
    python exit_monitor.py              # start continuous monitor
    python exit_monitor.py --once       # run one evaluation cycle and exit

Architecture:
    - Reloads positions from disk every cycle (file-based coordination
      with the main pipeline)
    - Reads regime from pipeline_state.json (set by daily pipeline)
    - Computes ATR inline from yfinance OHLCV (cached, refreshed every 5 min)
    - Uses entry_signal / entry_agreement as current signal fallback
      (no player signal generation)
"""

from __future__ import annotations

import json
import logging
import math
import signal
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# ── Local imports ────────────────────────────────────────────────────────

from paper_trading_config import CONFIG
from position_tracker import PositionTracker
from trade_journal import TradeJournal
from smart_exit_engine import SmartExitEngine, PositionState
from scheduler import is_trading_day

# ── yfinance ─────────────────────────────────────────────────────────────

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

# ── Logging ──────────────────────────────────────────────────────────────

logger = logging.getLogger("exit_monitor")

# ── Constants ────────────────────────────────────────────────────────────

TICK_INTERVAL = 30          # seconds between evaluations
OHLCV_CACHE_TTL = 300       # seconds — refresh full OHLCV every 5 minutes
OHLCV_LOOKBACK = "25d"      # enough for 14-period ATR + buffer

# IST market hours (as UTC offsets for comparison)
_IST_OFFSET = timedelta(hours=5, minutes=30)
_IST = timezone(_IST_OFFSET)
MARKET_OPEN_IST = (9, 15)   # 09:15 IST
MARKET_CLOSE_IST = (15, 30) # 15:30 IST


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _now_ist() -> datetime:
    """Current datetime in IST."""
    return datetime.now(_IST)


def _is_market_open() -> bool:
    """Check if NSE is currently in regular trading hours (09:15–15:30 IST)."""
    now = _now_ist()
    open_time = now.replace(hour=MARKET_OPEN_IST[0], minute=MARKET_OPEN_IST[1],
                            second=0, microsecond=0)
    close_time = now.replace(hour=MARKET_CLOSE_IST[0], minute=MARKET_CLOSE_IST[1],
                             second=0, microsecond=0)
    return open_time <= now <= close_time


def _seconds_until_market_open() -> float:
    """Seconds until next market open (09:15 IST), skipping weekends/holidays."""
    now = _now_ist()
    target = now.replace(hour=MARKET_OPEN_IST[0], minute=MARKET_OPEN_IST[1],
                         second=0, microsecond=0)
    if now >= target:
        # Already past today's open — aim for tomorrow
        target += timedelta(days=1)
    # Skip weekends and holidays
    while not is_trading_day(target.date()):
        target += timedelta(days=1)
    return (target - now).total_seconds()


def _load_regime() -> str:
    """Read last known regime from pipeline_state.json."""
    state_file = Path(CONFIG["state_file"])
    try:
        if state_file.exists():
            data = json.loads(state_file.read_text(encoding="utf-8"))
            return data.get("last_regime", "sideways").lower()
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Could not read pipeline state: {e}")
    return "sideways"


def _compute_atr(ohlcv: pd.DataFrame, period: int = 14) -> float:
    """Compute ATR from OHLCV DataFrame (last `period` bars).

    Uses True Range = max(H-L, |H-prevC|, |L-prevC|).
    Returns ATR as a float, or 1.5% of last close as fallback.
    """
    if ohlcv is None or len(ohlcv) < period + 1:
        # Not enough data — return fallback
        if ohlcv is not None and len(ohlcv) > 0:
            return float(ohlcv["Close"].iloc[-1]) * 0.015
        return 0.0

    high = ohlcv["High"].values
    low = ohlcv["Low"].values
    close = ohlcv["Close"].values

    tr_list = []
    for i in range(1, len(ohlcv)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr_list.append(max(hl, hc, lc))

    if len(tr_list) < period:
        return float(np.mean(tr_list)) if tr_list else close[-1] * 0.015

    return float(np.mean(tr_list[-period:]))


# ═══════════════════════════════════════════════════════════════════════════
# OHLCV Cache — batch download, refreshed every 5 minutes
# ═══════════════════════════════════════════════════════════════════════════

class OHLCVCache:
    """Caches OHLCV data per symbol. Full refresh every OHLCV_CACHE_TTL seconds."""

    def __init__(self):
        self._data: Dict[str, pd.DataFrame] = {}   # symbol -> DataFrame
        self._last_refresh: float = 0.0             # monotonic timestamp

    def get(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._data.get(symbol)

    def needs_refresh(self) -> bool:
        return (time.monotonic() - self._last_refresh) > OHLCV_CACHE_TTL

    def refresh(self, symbols: List[str]) -> Dict[str, float]:
        """Download OHLCV for all symbols. Returns {symbol: last_close}."""
        prices: Dict[str, float] = {}
        if not _HAS_YF or not symbols:
            return prices

        try:
            data = yf.download(
                symbols,
                period=OHLCV_LOOKBACK,
                progress=False,
                threads=True,
            )

            if data is None or data.empty:
                logger.warning("yfinance returned no data")
                return prices

            for sym in symbols:
                try:
                    if len(symbols) == 1:
                        df = data[["Open", "High", "Low", "Close", "Volume"]].copy()
                    else:
                        df = pd.DataFrame({
                            "Open": data["Open"][sym],
                            "High": data["High"][sym],
                            "Low": data["Low"][sym],
                            "Close": data["Close"][sym],
                            "Volume": data["Volume"][sym],
                        })
                    df = df.dropna()
                    if len(df) > 0:
                        self._data[sym] = df
                        prices[sym] = float(df["Close"].iloc[-1])
                except Exception as e:
                    logger.debug(f"Failed to extract {sym}: {e}")

            self._last_refresh = time.monotonic()
            logger.info(f"OHLCV cache refreshed: {len(prices)}/{len(symbols)} symbols")

        except Exception as e:
            logger.error(f"yfinance batch download failed: {e}")

        return prices

    def fetch_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Quick price fetch (latest close). Uses cached data if OHLCV
        was refreshed recently, otherwise does a lightweight 1d download."""
        if not self.needs_refresh():
            # Use cached last close
            return {
                sym: float(df["Close"].iloc[-1])
                for sym, df in self._data.items()
                if sym in symbols and len(df) > 0
            }
        # Full refresh needed
        return self.refresh(symbols)


# ═══════════════════════════════════════════════════════════════════════════
# Exit Monitor — main evaluation loop
# ═══════════════════════════════════════════════════════════════════════════

class ExitMonitor:
    """Standalone exit engine that evaluates positions every 30 seconds."""

    def __init__(self):
        self._exit_engine = SmartExitEngine()
        self._ohlcv_cache = OHLCVCache()
        self._running = True

        # Log file for exit decisions
        self._log_dir = Path(__file__).parent / "logs"
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def _load_tracker(self) -> PositionTracker:
        """Create a fresh PositionTracker from disk each cycle."""
        return PositionTracker(
            capital=CONFIG["starting_capital"],
            num_players=CONFIG["num_players"],
            capital_per_player=CONFIG["capital_per_player"],
            positions_file=CONFIG["positions_file"],
            trailing_stop_pct=CONFIG.get("trailing_stop_pct", 0.02),
        )

    def _load_journal(self) -> TradeJournal:
        """Create a fresh TradeJournal from disk."""
        return TradeJournal(journal_file=CONFIG["journal_file"])

    def _build_position_state(self, pos) -> PositionState:
        """Convert a Position object to a PositionState for the exit engine."""
        return PositionState(
            symbol=pos.symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            entry_date=pos.entry_date,
            entry_signal=pos.entry_signal,
            entry_agreement=pos.entry_agreement,
            entry_volatility=pos.entry_volatility,
            entry_regime=pos.entry_regime,
            remaining_fraction=pos.remaining_fraction,
            staged_exits_done=pos.staged_exits_done,
            max_favorable_excursion=pos.max_favorable_excursion,
            max_adverse_excursion=pos.max_adverse_excursion,
            peak_price=pos.high_water_mark,
            signal_history=list(pos.signal_history),
            agreement_history=list(pos.agreement_history),
            volume_history=list(pos.volume_history),
            bars_since_entry=pos.bars_held,
        )

    def _log_exit_decision(self, symbol: str, decision, details: Dict) -> None:
        """Append exit decision to JSONL log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "should_exit": decision.should_exit,
            "exit_fraction": decision.exit_fraction,
            "reason": decision.reason.value,
            "urgency": decision.urgency,
            "weighted_score": details.get("weighted_score"),
            "signal_scores": details.get("signal_scores", {}),
            "source": "exit_monitor",
        }
        try:
            log_file = self._log_dir / "exit_monitor.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass

    def evaluate_once(self) -> Dict[str, Any]:
        """Run one evaluation cycle on all open positions.

        Returns a summary dict with cycle stats.
        """
        cycle_start = time.monotonic()
        today = date.today().isoformat()

        # Load state
        tracker = self._load_tracker()
        journal = self._load_journal()
        regime = _load_regime()

        n_positions = len(tracker.positions)
        if n_positions == 0:
            logger.info("No open positions — nothing to evaluate")
            return {"positions": 0, "exits": 0, "elapsed_ms": 0}

        # Get symbols from open positions
        symbols = list(set(pos.symbol for pos in tracker.positions.values()))

        # Fetch prices (uses cache or refreshes)
        prices = self._ohlcv_cache.fetch_latest_prices(symbols)

        if not prices:
            logger.warning("No prices fetched — skipping cycle")
            return {"positions": n_positions, "exits": 0, "elapsed_ms": 0,
                    "error": "no_prices"}

        exits_triggered = 0
        trades_executed: List[Dict] = []

        # Evaluate each position
        position_items = list(tracker.positions.items())
        for pid, pos in position_items:
            sym = pos.symbol
            price = prices.get(sym)
            if price is None:
                continue

            # Get OHLCV for ATR
            ohlcv = self._ohlcv_cache.get(sym)
            atr = _compute_atr(ohlcv)
            if atr <= 0:
                atr = price * 0.015  # fallback: 1.5% of price

            # Current bar data from OHLCV
            if ohlcv is not None and len(ohlcv) > 0:
                last_bar = ohlcv.iloc[-1]
                current_low = float(last_bar.get("Low", price * 0.99))
                current_high = float(last_bar.get("High", price * 1.01))
                current_volume = float(last_bar.get("Volume", 0))
            else:
                current_low = price * 0.99
                current_high = price * 1.01
                current_volume = 0

            # Vol forecast from ATR
            current_vol = (atr / price) * math.sqrt(252) if price > 0 else 0.2

            # Build PositionState
            pos_state = self._build_position_state(pos)

            # Evaluate — pure math, no Gemini
            decision = self._exit_engine.evaluate(
                position=pos_state,
                current_price=price,
                current_low=current_low,
                current_volume=current_volume,
                atr=atr,
                current_signal=pos.entry_signal,       # no fresh signals
                current_agreement=pos.entry_agreement,  # no fresh consensus
                current_vol_forecast=current_vol,
                regime=regime,
                current_high=current_high,
                eval_date=date.today(),
            )

            # Log the decision
            self._log_exit_decision(sym, decision, decision.details)

            # Execute exit if triggered (before state sync — close deletes position)
            if decision.should_exit:
                reason_str = decision.reason.value
                exits_triggered += 1

                if decision.exit_fraction < 1.0:
                    # Partial exit
                    shares_to_exit = max(1, int(pos.quantity * decision.exit_fraction))
                    trade = tracker.partial_close_position(
                        pid, shares_to_exit, price,
                        exit_date=today, reason=reason_str,
                    )
                else:
                    # Full exit
                    trade = tracker.close_position(
                        pid, price, exit_date=today, reason=reason_str,
                    )

                if trade:
                    trade["regime"] = regime
                    trade["exit_urgency"] = round(decision.urgency, 2)
                    trade["exit_source"] = "exit_monitor"
                    journal.record_trade(trade)
                    trades_executed.append(trade)

                    pnl = trade.get("pnl", 0)
                    logger.info(
                        f"EXIT {reason_str}: {pos.direction} {sym} "
                        f"@ {price:,.2f} | P&L=₹{pnl:+,.2f} | "
                        f"fraction={decision.exit_fraction:.0%} | "
                        f"urgency={decision.urgency:.2f} | "
                        f"[{pos.player_id}]"
                    )
            else:
                # No exit — sync state back to position (MFE/MAE, peak)
                # Only for positions that are still open (not closed above)
                if pid in tracker.positions:
                    pos.high_water_mark = pos_state.peak_price
                    pos.max_favorable_excursion = pos_state.max_favorable_excursion
                    pos.max_adverse_excursion = pos_state.max_adverse_excursion
                    pos.remaining_fraction = pos_state.remaining_fraction
                    pos.staged_exits_done = pos_state.staged_exits_done

        # Save updated positions (MFE/MAE synced even if no exits)
        tracker._save()

        elapsed_ms = int((time.monotonic() - cycle_start) * 1000)

        logger.info(
            f"Cycle: {n_positions} positions | "
            f"{len(prices)} prices | "
            f"{exits_triggered} exits | "
            f"regime={regime} | "
            f"{elapsed_ms}ms"
        )

        return {
            "positions": n_positions,
            "prices_fetched": len(prices),
            "exits": exits_triggered,
            "trades": trades_executed,
            "regime": regime,
            "elapsed_ms": elapsed_ms,
        }

    def run(self) -> None:
        """Main loop — evaluate every 30 seconds during market hours."""
        logger.info("=" * 60)
        logger.info("  Exit Monitor started")
        logger.info(f"  Tick interval: {TICK_INTERVAL}s")
        logger.info(f"  OHLCV cache TTL: {OHLCV_CACHE_TTL}s")
        logger.info(f"  Market hours: {MARKET_OPEN_IST[0]:02d}:{MARKET_OPEN_IST[1]:02d}"
                     f"–{MARKET_CLOSE_IST[0]:02d}:{MARKET_CLOSE_IST[1]:02d} IST")
        logger.info("=" * 60)

        while self._running:
            today = date.today()

            # Skip non-trading days
            if not is_trading_day(today):
                now = _now_ist()
                logger.info(
                    f"Not a trading day ({today}, {now.strftime('%A')}). "
                    f"Sleeping until tomorrow..."
                )
                # Sleep until next day 09:00 IST
                sleep_secs = _seconds_until_market_open()
                self._interruptible_sleep(sleep_secs)
                continue

            # Check market hours
            if not _is_market_open():
                now = _now_ist()
                close_time = now.replace(hour=MARKET_CLOSE_IST[0],
                                         minute=MARKET_CLOSE_IST[1],
                                         second=0, microsecond=0)
                if now >= close_time:
                    # After close — sleep until tomorrow
                    logger.info(
                        f"Market closed for today ({now.strftime('%H:%M')} IST). "
                        f"Sleeping until tomorrow..."
                    )
                    sleep_secs = _seconds_until_market_open()
                    self._interruptible_sleep(sleep_secs)
                else:
                    # Before open — sleep until open
                    sleep_secs = _seconds_until_market_open()
                    mins = int(sleep_secs / 60)
                    logger.info(
                        f"Pre-market ({now.strftime('%H:%M')} IST). "
                        f"Market opens in {mins} minutes..."
                    )
                    self._interruptible_sleep(min(sleep_secs, 60))
                continue

            # ── Market is open — run evaluation ──
            try:
                self.evaluate_once()
            except Exception as e:
                logger.error(f"Evaluation cycle failed: {e}", exc_info=True)

            # Sleep until next tick
            self._interruptible_sleep(TICK_INTERVAL)

        logger.info("Exit Monitor stopped.")

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep that can be interrupted by shutdown signal."""
        end = time.monotonic() + seconds
        while self._running and time.monotonic() < end:
            time.sleep(min(1.0, end - time.monotonic()))

    def shutdown(self) -> None:
        """Signal the monitor to stop."""
        logger.info("Shutdown signal received...")
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy loggers
    for name in ("urllib3", "yfinance", "peewee", "filelock"):
        logging.getLogger(name).setLevel(logging.WARNING)

    monitor = ExitMonitor()

    # Graceful shutdown
    def _handle_signal(signum, frame):
        monitor.shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    if "--once" in sys.argv:
        # Single evaluation cycle (for testing)
        result = monitor.evaluate_once()
        print(json.dumps(result, indent=2, default=str))
    else:
        monitor.run()


if __name__ == "__main__":
    main()
