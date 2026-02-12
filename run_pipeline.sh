#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Paper Trading Pipeline Runner
# Wraps daily_runner.py with caffeinate to prevent sleep during run
# ═══════════════════════════════════════════════════════════════

MODE="${1:-full}"
DIR="$(cd "$(dirname "$0")" && pwd)"

# caffeinate -i prevents idle sleep for the duration of the python process
exec /usr/bin/caffeinate -i /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 "$DIR/daily_runner.py" --mode "$MODE"
