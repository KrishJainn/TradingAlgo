#!/usr/bin/env python3
"""
AQTIS 5-Player Live Paper Trading Launcher.

Starts the live trading engine and Streamlit dashboard.

Usage:
    python start_live_trading.py              # Start both trader and dashboard
    python start_live_trading.py --trader     # Start only the trader
    python start_live_trading.py --dashboard  # Start only the dashboard
"""

import argparse
import subprocess
import sys
import threading
import time
import signal
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def start_trader(symbols: list = None, poll_interval: int = 15):
    """Start the live trading engine."""
    from live_5player_trader import LiveFivePlayerTrader

    default_symbols = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    ]

    symbols = symbols or default_symbols

    print(f"\n{'='*60}")
    print("  AQTIS 5-Player Live Paper Trading Engine")
    print(f"{'='*60}")
    print(f"  Symbols: {len(symbols)}")
    print(f"  Poll Interval: {poll_interval} minutes")
    print(f"{'='*60}\n")

    trader = LiveFivePlayerTrader(
        symbols=symbols,
        poll_interval=poll_interval,
    )

    # Run continuous trading
    trader.run_continuous()


def start_dashboard():
    """Start the Streamlit dashboard."""
    dashboard_path = Path(__file__).parent / "aqtis" / "dashboard" / "app.py"

    print(f"\n{'='*60}")
    print("  Starting AQTIS Dashboard")
    print(f"  URL: http://localhost:8501")
    print(f"{'='*60}\n")

    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.headless", "true",
    ])


def main():
    parser = argparse.ArgumentParser(
        description="AQTIS 5-Player Live Paper Trading Launcher"
    )
    parser.add_argument(
        "--trader",
        action="store_true",
        help="Start only the trading engine",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start only the dashboard",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        help="Poll interval in minutes (default: 15)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        help="List of symbols to trade (default: NSE top 10)",
    )

    args = parser.parse_args()

    # Handle signals for graceful shutdown
    stop_event = threading.Event()

    def signal_handler(signum, frame):
        print("\n\nShutting down...")
        stop_event.set()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start components based on args
    if args.trader:
        start_trader(args.symbols, args.poll_interval)
    elif args.dashboard:
        start_dashboard()
    else:
        # Start both: trader in background thread, dashboard in foreground
        print("\n" + "="*60)
        print("  AQTIS 5-Player Live Paper Trading System")
        print("="*60)
        print("\nStarting trading engine in background...")
        print("Starting dashboard in foreground...")
        print("\nPress Ctrl+C to stop both.\n")

        # Start trader in background thread
        trader_thread = threading.Thread(
            target=start_trader,
            args=(args.symbols, args.poll_interval),
            daemon=True,
        )
        trader_thread.start()

        # Give trader a moment to initialize
        time.sleep(2)

        # Start dashboard (this blocks)
        start_dashboard()


if __name__ == "__main__":
    main()
