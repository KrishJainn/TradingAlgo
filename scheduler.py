"""
Scheduler for the Paper Trading Pipeline — 3 daily runs.

Automated intraday execution using the `schedule` library.
Schedules 3 jobs per trading day:

    09:00 IST  →  "intelligence"   Morning brief + heatmaps
    12:30 IST  →  "refresh"        News + breadth refresh
    17:30 IST  →  "full"           Full pipeline (signals + trades + metrics)

All times are IST (UTC+5:30). The schedule library uses local system clock,
so times are converted from IST → local timezone at startup.

Features:
    - 3 intraday scheduled runs with distinct pipeline modes
    - IST → local timezone conversion (works on any machine)
    - Shared DailyRunner instance across runs (avoids re-init)
    - Graceful shutdown on Ctrl+C / SIGTERM
    - Weekend / NSE holiday skipping
    - Logging of each run
    - Manual trigger support (--run-now --mode <mode>)

Usage (Python):
    from scheduler import start_scheduler
    start_scheduler()              # runs forever, 3 triggers per day

Usage (CLI):
    python scheduler.py                         # start 3-job scheduler
    python scheduler.py --run-now               # run full pipeline now
    python scheduler.py --run-now --mode intelligence   # run morning brief now
    python scheduler.py --run-now --mode refresh        # run midday refresh now

Usage (crontab — alternative to running this script):
    crontab -e
    # 09:00 IST = 03:30 UTC on weekdays:
    30 3 * * 1-5 cd /path/to/project && python3 daily_runner.py --mode intelligence >> paper_trading.log 2>&1
    # 12:30 IST = 07:00 UTC on weekdays:
    0  7 * * 1-5 cd /path/to/project && python3 daily_runner.py --mode refresh >> paper_trading.log 2>&1
    # 17:30 IST = 12:00 UTC on weekdays:
    0 12 * * 1-5 cd /path/to/project && python3 daily_runner.py --mode full >> paper_trading.log 2>&1
"""

from __future__ import annotations

import logging
import signal
import sys
import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Optional import — schedule library
try:
    import schedule
    _HAS_SCHEDULE = True
except ImportError:
    _HAS_SCHEDULE = False

# Optional import — timezone support
try:
    from zoneinfo import ZoneInfo
    _HAS_ZONEINFO = True
except ImportError:
    try:
        from backports.zoneinfo import ZoneInfo
        _HAS_ZONEINFO = True
    except ImportError:
        _HAS_ZONEINFO = False


from paper_trading_config import CONFIG
from daily_runner import DailyRunner


# ═══════════════════════════════════════════════════════════════════════════
# Schedule configuration — 3 daily runs (IST times)
# ═══════════════════════════════════════════════════════════════════════════

SCHEDULE_JOBS: list = [
    {
        "ist_time": "09:00",
        "mode": "intelligence",
        "label": "Morning Intelligence (brief + heatmaps)",
    },
    {
        "ist_time": "12:30",
        "mode": "refresh",
        "label": "Midday Refresh (news + breadth)",
    },
    {
        "ist_time": "17:30",
        "mode": "full",
        "label": "Full Pipeline (signals + trades + metrics)",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
# IST → local time conversion
# ═══════════════════════════════════════════════════════════════════════════

_IST_OFFSET = timedelta(hours=5, minutes=30)
_IST = timezone(_IST_OFFSET)


def ist_to_local(ist_time_str: str) -> str:
    """Convert an IST time string (HH:MM) to local system time (HH:MM).

    Uses zoneinfo if available for DST-aware conversion,
    otherwise falls back to fixed UTC+5:30 offset math.
    """
    hour, minute = map(int, ist_time_str.split(":"))

    if _HAS_ZONEINFO:
        # DST-aware: build a datetime in IST, convert to system local
        ist_tz = ZoneInfo("Asia/Kolkata")
        today = date.today()
        ist_dt = datetime(today.year, today.month, today.day, hour, minute, tzinfo=ist_tz)
        local_dt = ist_dt.astimezone()  # system local timezone (DST-aware)
        return local_dt.strftime("%H:%M")
    else:
        # Fallback: fixed UTC+5:30 offset math (no DST awareness)
        ist_dt = datetime(2025, 1, 1, hour, minute, tzinfo=_IST)
        local_dt = ist_dt.astimezone()  # system local timezone
        return local_dt.strftime("%H:%M")


# ═══════════════════════════════════════════════════════════════════════════
# Weekend / Holiday check
# ═══════════════════════════════════════════════════════════════════════════

# Major NSE holidays (fixed dates — update annually)
NSE_HOLIDAYS: set = {
    # ── 2025 ──────────────────────────────────────────────────────
    "2025-01-26",  # Republic Day
    "2025-02-26",  # Maha Shivaratri
    "2025-03-14",  # Holi
    "2025-03-31",  # Id-ul-Fitr
    "2025-04-10",  # Shri Mahavir Jayanti
    "2025-04-14",  # Dr. Ambedkar Jayanti
    "2025-04-18",  # Good Friday
    "2025-05-01",  # Maharashtra Day
    "2025-06-07",  # Bakrid (Eid ul-Adha)
    "2025-08-15",  # Independence Day
    "2025-08-16",  # Parsi New Year
    "2025-08-27",  # Ganesh Chaturthi
    "2025-10-02",  # Mahatma Gandhi Jayanti
    "2025-10-20",  # Diwali (Laxmi Pujan)
    "2025-10-21",  # Diwali Balipratipada
    "2025-11-05",  # Guru Nanak Jayanti
    "2025-12-25",  # Christmas
    # ── 2026 ──────────────────────────────────────────────────────
    "2026-01-26",  # Republic Day
    "2026-02-17",  # Maha Shivaratri
    "2026-03-04",  # Holi
    "2026-03-20",  # Id-ul-Fitr (tentative)
    "2026-03-25",  # Shri Mahavir Jayanti
    "2026-04-03",  # Good Friday
    "2026-04-14",  # Dr. Ambedkar Jayanti
    "2026-05-01",  # Maharashtra Day
    "2026-05-27",  # Bakrid (tentative)
    "2026-08-15",  # Independence Day
    "2026-08-26",  # Janmashtami
    "2026-10-02",  # Mahatma Gandhi Jayanti
    "2026-10-09",  # Dussehra
    "2026-10-29",  # Diwali (Laxmi Pujan)
    "2026-11-24",  # Guru Nanak Jayanti
    "2026-12-25",  # Christmas
}

# Track which years have holiday data
_HOLIDAY_YEARS_COVERED = {2025, 2026}


def is_trading_day(d: Optional[date] = None) -> bool:
    """Check if the given date is a valid NSE trading day.

    Returns False for weekends and known holidays.
    Logs a warning if the year's holiday calendar is not loaded.
    """
    d = d or date.today()

    # Skip weekends
    if d.weekday() >= 5:  # Saturday=5, Sunday=6
        return False

    # Warn if holiday data is missing for this year
    if d.year not in _HOLIDAY_YEARS_COVERED:
        logger.warning(
            f"No holiday calendar for {d.year} — update NSE_HOLIDAYS in scheduler.py. "
            f"Only weekends will be skipped."
        )

    # Skip known holidays
    if d.isoformat() in NSE_HOLIDAYS:
        return False

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Shared DailyRunner instance (reused across all 3 runs per day)
# ═══════════════════════════════════════════════════════════════════════════

_runner: Optional[DailyRunner] = None


def _get_runner() -> DailyRunner:
    """Get or create the shared DailyRunner instance."""
    global _runner
    if _runner is None:
        _runner = DailyRunner()
    return _runner


# ═══════════════════════════════════════════════════════════════════════════
# Job functions (one per mode)
# ═══════════════════════════════════════════════════════════════════════════

def _run_mode(mode: str, label: str) -> None:
    """Execute a pipeline run in the given mode, skipping non-trading days."""
    today = date.today()

    if not is_trading_day(today):
        logger.info(f"  Skipping {today} — not a trading day (weekend/holiday)")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"  SCHEDULED: {label}")
    logger.info(f"  Date: {today.isoformat()}")
    logger.info(f"{'='*60}")

    try:
        runner = _get_runner()
        report = runner.run(mode=mode)

        if report["success"]:
            logger.info(f"  ✅ {label} completed successfully")
            _log_mode_summary(mode, report)
        else:
            logger.error(f"  ❌ {label} had {len(report['errors'])} errors:")
            for err in report["errors"]:
                logger.error(f"    • {err}")

    except Exception as e:
        logger.error(f"  FATAL ERROR in {label}: {e}", exc_info=True)


def _log_mode_summary(mode: str, report: Dict) -> None:
    """Log mode-specific summary after a successful run."""
    if mode == "full" and report.get("summary"):
        s = report["summary"]
        logger.info(f"  Equity: ₹{s['equity']:,.2f}  |  "
                     f"Win Rate: {s['win_rate']:.1f}%  |  "
                     f"P&L: ₹{s['total_pnl']:+,.2f}")

    elif mode == "intelligence" and report.get("morning_brief"):
        b = report["morning_brief"]
        logger.info(f"  Regime: {b['regime']} (conf {b['regime_confidence']:.0%})")
        logger.info(f"  Top Bull: {', '.join(s['symbol'] for s in b['top_bullish'])}")
        logger.info(f"  Top Bear: {', '.join(s['symbol'] for s in b['top_bearish'])}")

    elif mode == "refresh":
        breadth = report.get("breadth", {})
        deltas = report.get("sentiment_deltas", [])
        logger.info(f"  Breadth: {breadth.get('advancing', 0)}↑ "
                     f"{breadth.get('declining', 0)}↓ "
                     f"({breadth.get('signal', 'n/a')})")
        logger.info(f"  Sentiment shifts: {len(deltas)}")


def run_intelligence() -> None:
    """09:00 IST — Morning intelligence (brief + heatmaps)."""
    _run_mode("intelligence", "Morning Intelligence (brief + heatmaps)")


def run_refresh() -> None:
    """12:30 IST — Midday refresh (news + breadth)."""
    _run_mode("refresh", "Midday Refresh (news + breadth)")


def run_full_pipeline() -> None:
    """17:30 IST — Full pipeline (signals + trades + metrics)."""
    _run_mode("full", "Full Pipeline (signals + trades + metrics)")


# ═══════════════════════════════════════════════════════════════════════════
# Scheduler — 3 daily jobs
# ═══════════════════════════════════════════════════════════════════════════

def start_scheduler() -> None:
    """
    Start the 3-job daily scheduler.

    Schedules:
        09:00 IST  →  Morning Intelligence (brief + heatmaps)
        12:30 IST  →  Midday Refresh (news + breadth)
        17:30 IST  →  Full Pipeline (signals + trades + metrics)

    Times are converted from IST to your local system timezone.
    The scheduler runs in an infinite loop, checking every 30 seconds.
    Press Ctrl+C to stop gracefully.
    """
    if not _HAS_SCHEDULE:
        print("ERROR: 'schedule' library is required.")
        print("Install with: pip install schedule")
        sys.exit(1)

    # Clear any jobs from previous calls (prevents accumulation)
    schedule.clear()

    # Map modes to their job functions
    mode_to_func = {
        "intelligence": run_intelligence,
        "refresh": run_refresh,
        "full": run_full_pipeline,
    }

    logger.info(f"{'='*60}")
    logger.info(f"  Paper Trading Scheduler — 3 Daily Jobs")
    logger.info(f"{'='*60}")

    for job in SCHEDULE_JOBS:
        ist_time = job["ist_time"]
        local_time = ist_to_local(ist_time)
        func = mode_to_func[job["mode"]]
        label = job["label"]

        schedule.every().day.at(local_time).do(func)

        logger.info(f"  {ist_time} IST ({local_time} local) → {label}")

    logger.info(f"{'─'*60}")
    logger.info(f"  Press Ctrl+C to stop")
    logger.info(f"{'='*60}")

    # Handle graceful shutdown
    running = True

    def _shutdown(signum, frame):
        nonlocal running
        logger.info("\n  Received shutdown signal. Stopping scheduler...")
        running = False

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Main loop
    while running:
        schedule.run_pending()
        time.sleep(30)

    logger.info("  Scheduler stopped.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI entry point for the scheduler.

    Usage:
        python scheduler.py                            # start 3-job scheduler
        python scheduler.py --run-now                  # run full pipeline now
        python scheduler.py --run-now --mode intelligence
        python scheduler.py --run-now --mode refresh
        python scheduler.py --run-now --mode full
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if "--run-now" in sys.argv:
        # Determine mode
        mode = "full"
        if "--mode" in sys.argv:
            idx = sys.argv.index("--mode")
            if idx + 1 < len(sys.argv):
                mode = sys.argv[idx + 1]

        mode_funcs = {
            "intelligence": run_intelligence,
            "refresh": run_refresh,
            "full": run_full_pipeline,
        }

        if mode not in mode_funcs:
            print(f"ERROR: Unknown mode '{mode}'. Use: intelligence, refresh, full")
            sys.exit(1)

        label = {"intelligence": "Morning Intelligence",
                 "refresh": "Midday Refresh",
                 "full": "Full Pipeline"}[mode]
        print(f"Running {label} immediately...\n")
        mode_funcs[mode]()
    else:
        start_scheduler()


if __name__ == "__main__":
    main()
