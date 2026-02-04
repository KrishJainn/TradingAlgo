#!/usr/bin/env python3
"""
Run the 5-Player Trading Coach Dashboard.

Usage:
    python run_dashboard.py
    # OR
    streamlit run coach_system/dashboard/app.py
"""

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    dashboard_path = Path(__file__).parent / "coach_system" / "dashboard" / "app.py"

    print("🎯 Starting 5-Player Trading Coach Dashboard...")
    print(f"   Dashboard: {dashboard_path}")
    print("   Press Ctrl+C to stop\n")

    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
