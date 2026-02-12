"""
Paper Trading Pipeline — Targeted Test Suite (12 tests).

Tests:
    1.  PositionTracker open/close cycle — P&L math correct (LONG + SHORT)
    2.  PositionTracker with 0 positions — portfolio_summary doesn't crash
    3.  Stop loss check — position at 100, stop 5%, price 94 → triggers
    4.  Trailing stop update — price moves up, stop follows
    5.  TradeJournal record + filter — 10 trades, filter by player, count
    6.  TradeJournal export_csv — valid CSV, readable with csv module
    7.  DailyRunner on Republic Day (Jan 26) — scheduler skips
    8.  DailyRunner with Gemini unavailable — fallback mode works
    9.  Daily report formatting — 0 trades, all losses, no crash
   10.  Metrics recording — 3 days → 3 snapshots
   11.  Position persistence — open, reload, verify
   12.  Cumulative P&L across days — matches sum of daily P&Ls

Run:
    python3 tests/test_paper_trading.py
"""

from __future__ import annotations

import csv
import io
import json
import math
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from position_tracker import PositionTracker
from trade_journal import TradeJournal
from metrics_store import MetricsStore
from daily_runner import DailyRunner
from paper_trading_config import CONFIG, UNIVERSE, PLAYER_IDS
from scheduler import is_trading_day

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

PASS_COUNT = 0
FAIL_COUNT = 0


def _check(name: str, condition: bool, detail: str = "") -> None:
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  ✅ PASS: {name}")
    else:
        FAIL_COUNT += 1
        msg = f"  ❌ FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def _make_test_config(tmpdir: str) -> Dict[str, Any]:
    """Build a config dict pointing to a temp directory."""
    cfg = dict(CONFIG)
    cfg["starting_capital"] = 1_000_000
    cfg["metrics_dir"] = str(Path(tmpdir) / "metrics")
    cfg["positions_file"] = str(Path(tmpdir) / "positions.json")
    cfg["journal_file"] = str(Path(tmpdir) / "journal.json")
    cfg["state_file"] = str(Path(tmpdir) / "state.json")
    cfg["daily_reports_dir"] = str(Path(tmpdir) / "reports")
    cfg["signal_combiner_state_dir"] = str(Path(tmpdir) / "combiner")
    cfg["max_positions"] = 8
    cfg["position_size_pct"] = 0.08
    return cfg


def _make_prices(day_offset: int, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed + day_offset)
    base = {
        "RELIANCE.NS": 2500, "HDFCBANK.NS": 1600, "INFY.NS": 1800,
        "TCS.NS": 4000, "ICICIBANK.NS": 1200, "SBIN.NS": 750,
        "BHARTIARTL.NS": 1500, "ITC.NS": 450, "TATASTEEL.NS": 160,
        "MARUTI.NS": 12000, "AXISBANK.NS": 1100, "BAJFINANCE.NS": 6500,
        "SUNPHARMA.NS": 1700, "LT.NS": 3500, "WIPRO.NS": 480,
    }
    prices = {}
    for sym in UNIVERSE:
        b = base.get(sym, 1000.0)
        drift = float(rng.normal(0.001, 0.012))
        prices[sym] = round(b * (1 + drift) ** (day_offset + 1), 2)
    return prices


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: PositionTracker open/close cycle — P&L math
# ═══════════════════════════════════════════════════════════════════════════

def test_01_open_close_pnl():
    """P&L math correct for LONG and SHORT positions."""
    print("\n[Test 1] PositionTracker open/close P&L math")

    with tempfile.TemporaryDirectory() as d:
        t = PositionTracker(capital=1_000_000, positions_file=f"{d}/p.json")

        # ── LONG: buy 200 @ ₹1000, sell @ ₹1080 ──
        pos = t.open_position("REL.NS", "PLAYER_1", "LONG", 200, 1000.0, "2025-01-10")
        expected_capital_after_buy = 1_000_000 - 200 * 1000
        _check("LONG: capital reduced correctly",
               abs(t.current_capital - expected_capital_after_buy) < 0.01,
               f"expected {expected_capital_after_buy}, got {t.current_capital}")

        trade = t.close_position(pos.position_id, exit_price=1080.0)
        expected_pnl = (1080 - 1000) * 200  # +16,000
        _check("LONG: P&L = ₹16,000",
               abs(trade["pnl"] - expected_pnl) < 0.01,
               f"got {trade['pnl']}")
        _check("LONG: return_pct correct",
               abs(trade["return_pct"] - 8.0) < 0.01,
               f"got {trade['return_pct']}")

        capital_after_long = 1_000_000 + expected_pnl  # 1,016,000
        _check("LONG: capital after close = initial + profit",
               abs(t.current_capital - capital_after_long) < 0.01,
               f"got {t.current_capital}")

        # ── SHORT: sell 100 @ ₹2000, cover @ ₹1900 ──
        pos2 = t.open_position("TCS.NS", "PLAYER_2", "SHORT", 100, 2000.0, "2025-01-10")
        # SHORT open: cash increases by entry_price * qty
        expected_after_short = capital_after_long + 100 * 2000
        _check("SHORT: capital increases on open",
               abs(t.current_capital - expected_after_short) < 0.01,
               f"expected {expected_after_short}, got {t.current_capital}")

        trade2 = t.close_position(pos2.position_id, exit_price=1900.0)
        expected_pnl_short = (2000 - 1900) * 100  # +10,000
        _check("SHORT: P&L = ₹10,000",
               abs(trade2["pnl"] - expected_pnl_short) < 0.01,
               f"got {trade2['pnl']}")

        # Final capital should be initial + LONG profit + SHORT profit
        final_expected = 1_000_000 + 16_000 + 10_000
        _check("Final capital = initial + all profits",
               abs(t.current_capital - final_expected) < 0.01,
               f"expected {final_expected}, got {t.current_capital}")

        # Equity with no open positions = cash
        s = t.portfolio_summary()
        _check("Equity = cash (no open pos)",
               abs(s["equity"] - t.current_capital) < 0.01,
               f"equity={s['equity']}, cash={t.current_capital}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: PositionTracker with 0 positions — no crash
# ═══════════════════════════════════════════════════════════════════════════

def test_02_zero_positions():
    """portfolio_summary, mark_to_market, check_stops all work with 0 positions."""
    print("\n[Test 2] PositionTracker with 0 positions")

    t = PositionTracker(capital=500_000)

    # portfolio_summary should not crash
    s = t.portfolio_summary()
    _check("Equity = initial capital", abs(s["equity"] - 500_000) < 0.01)
    _check("open_positions = 0", s["open_positions"] == 0)
    _check("unrealised_pnl = 0", abs(s["unrealised_pnl"]) < 0.01)
    _check("daily_return = 0", abs(s["daily_return"]) < 0.0001)
    _check("gross_exposure = 0", abs(s["gross_exposure"]) < 0.01)

    # mark_to_market should not crash
    t.mark_to_market({"RELIANCE.NS": 2500.0})
    _check("MTM with 0 positions OK", True)

    # check_stops should return empty list
    stopped = t.check_stops({"RELIANCE.NS": 2500.0})
    _check("check_stops returns []", stopped == [])

    # get_closed_today should be empty
    _check("closed_today empty", t.get_closed_today() == [])


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: Stop loss — position at 100, stop 5%, price 94 → triggers
# ═══════════════════════════════════════════════════════════════════════════

def test_03_stop_loss_triggers():
    """Explicit stop-loss trigger test: 5% trailing stop fires at 94."""
    print("\n[Test 3] Stop loss check — triggers at correct price")

    t = PositionTracker(capital=500_000, trailing_stop_pct=0.05)

    pos = t.open_position("TEST.NS", "PLAYER_1", "LONG", 100, 100.0, "2025-01-10")

    # HWM = 100 (entry price). Stop at 100 * 0.95 = 95.
    # Price 96 → should NOT trigger
    stopped = t.check_stops({"TEST.NS": 96.0})
    _check("Price 96 → no stop (stop at 95)", len(stopped) == 0)

    # Price 95 → should trigger (price <= stop_price)
    stopped = t.check_stops({"TEST.NS": 95.0})
    _check("Price 95 → stop triggers", len(stopped) == 1)

    # Re-open and test with 94
    pos2 = t.open_position("TEST2.NS", "PLAYER_2", "LONG", 100, 100.0, "2025-01-10")
    stopped2 = t.check_stops({"TEST2.NS": 94.0})
    _check("Price 94 → stop triggers", len(stopped2) == 1)
    _check("Stop P&L = -600 (94-100)*100",
           abs(stopped2[0]["pnl"] - (-600)) < 0.01,
           f"got {stopped2[0].get('pnl')}")
    _check("Stop reason = trailing_stop",
           stopped2[0].get("close_reason") == "trailing_stop")


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Trailing stop update — price moves up, stop follows
# ═══════════════════════════════════════════════════════════════════════════

def test_04_trailing_stop_follows():
    """Trailing stop ratchets up with price for LONG positions."""
    print("\n[Test 4] Trailing stop update — stop follows price up")

    t = PositionTracker(capital=500_000, trailing_stop_pct=0.05)

    pos = t.open_position("TEST.NS", "PLAYER_1", "LONG", 100, 100.0, "2025-01-10")

    # Price rises to 120 → HWM should update to 120
    t.mark_to_market({"TEST.NS": 120.0})
    pos = t.get_position(pos.position_id)
    _check("HWM updated to 120", abs(pos.high_water_mark - 120.0) < 0.01)

    # New stop = 120 * 0.95 = 114. Price 115 should NOT trigger.
    stopped = t.check_stops({"TEST.NS": 115.0})
    _check("Price 115 → no stop (stop at 114)", len(stopped) == 0)

    # Price 113 → should trigger (< 114)
    stopped = t.check_stops({"TEST.NS": 113.0})
    _check("Price 113 → stop triggers", len(stopped) == 1)
    _check("Stop P&L = 1300 (113-100)*100",
           abs(stopped[0]["pnl"] - 1300) < 0.01,
           f"got {stopped[0].get('pnl')}")

    # SHORT: trailing stop moves down
    pos2 = t.open_position("SHORT.NS", "PLAYER_3", "SHORT", 50, 200.0, "2025-01-10")
    # Price drops to 180 → HWM (lowest for short) should be 180
    t.mark_to_market({"SHORT.NS": 180.0})
    pos2 = t.get_position(pos2.position_id)
    _check("SHORT HWM tracks lowest = 180", abs(pos2.high_water_mark - 180.0) < 0.01)

    # Stop for short = 180 * 1.05 = 189. Price 190 → triggers
    stopped2 = t.check_stops({"SHORT.NS": 190.0})
    _check("SHORT stop triggers at 190", len(stopped2) == 1)
    # P&L = (200 - 190) * 50 = 500
    _check("SHORT stop P&L = 500",
           abs(stopped2[0]["pnl"] - 500) < 0.01,
           f"got {stopped2[0].get('pnl')}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: TradeJournal record + filter — 10 trades, filter by player
# ═══════════════════════════════════════════════════════════════════════════

def test_05_journal_filter():
    """Record 10 trades, filter by player, verify counts."""
    print("\n[Test 5] TradeJournal record + filter")

    with tempfile.TemporaryDirectory() as d:
        j = TradeJournal(journal_file=f"{d}/j.json")

        # Record 10 trades: 3 for PLAYER_1, 3 for PLAYER_2, 2 for PLAYER_3,
        #                    1 for PLAYER_4, 1 for PLAYER_5
        player_dist = ["PLAYER_1"] * 3 + ["PLAYER_2"] * 3 + ["PLAYER_3"] * 2 + \
                      ["PLAYER_4"] * 1 + ["PLAYER_5"] * 1
        symbols = UNIVERSE[:10]

        for i in range(10):
            j.record_trade({
                "symbol": symbols[i],
                "player_id": player_dist[i],
                "direction": "LONG" if i % 2 == 0 else "SHORT",
                "quantity": 100,
                "entry_price": 1000 + i * 50,
                "exit_price": 1000 + i * 50 + (50 if i % 3 != 0 else -30),
                "entry_date": "2025-01-10",
                "exit_date": "2025-01-15",
                "pnl": 5000 if i % 3 != 0 else -3000,
                "return_pct": 5.0 if i % 3 != 0 else -3.0,
                "bars_held": 5,
                "close_reason": "signal",
            })

        _check("Total trades = 10", j.trade_count == 10)

        # Filter by player
        p1 = j.filter_trades(player_id="PLAYER_1")
        _check("PLAYER_1 has 3 trades", len(p1) == 3)

        p2 = j.filter_trades(player_id="PLAYER_2")
        _check("PLAYER_2 has 3 trades", len(p2) == 3)

        p3 = j.filter_trades(player_id="PLAYER_3")
        _check("PLAYER_3 has 2 trades", len(p3) == 2)

        # Filter by direction
        longs = j.filter_trades(direction="LONG")
        shorts = j.filter_trades(direction="SHORT")
        _check("LONG + SHORT = 10", len(longs) + len(shorts) == 10)

        # Filter by min_pnl
        winners = j.filter_trades(min_pnl=0.01)
        _check("Winners filtered correctly", all(t["pnl"] > 0 for t in winners))

        # Performance summary
        s = j.performance_summary()
        _check("Summary total_trades = 10", s["total_trades"] == 10)
        _check("Summary has profit_factor", s["profit_factor"] > 0)


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: TradeJournal CSV export — valid, readable
# ═══════════════════════════════════════════════════════════════════════════

def test_06_csv_export():
    """CSV export is valid RFC 4180 and can be parsed back."""
    print("\n[Test 6] TradeJournal export_csv")

    with tempfile.TemporaryDirectory() as d:
        j = TradeJournal(journal_file=f"{d}/j.json")

        # Trade with a comma and quote in the reasoning field
        j.record_trade({
            "symbol": "RELIANCE.NS",
            "player_id": "PLAYER_1",
            "direction": "LONG",
            "quantity": 100,
            "entry_price": 2500,
            "exit_price": 2600,
            "pnl": 10000,
            "bars_held": 5,
            "reasoning": 'Bullish on RELIANCE, confidence=0.85, "strong buy"',
            "entry_date": "2025-01-10",
            "exit_date": "2025-01-15",
            "close_reason": "signal",
        })
        j.record_trade({
            "symbol": "TCS.NS",
            "player_id": "PLAYER_2",
            "direction": "SHORT",
            "quantity": 50,
            "entry_price": 4000,
            "exit_price": 3900,
            "pnl": 5000,
            "bars_held": 3,
            "entry_date": "2025-01-11",
            "exit_date": "2025-01-14",
            "close_reason": "stop",
        })

        csv_str = j.export_csv()

        # Parse with csv module — should not raise
        reader = csv.reader(io.StringIO(csv_str))
        rows = list(reader)

        _check("CSV has header + 2 data rows", len(rows) == 3,
               f"got {len(rows)} rows")

        header = rows[0]
        _check("Header contains 'pnl'", "pnl" in header)
        _check("Header contains 'symbol'", "symbol" in header)

        # Check that the comma in reasoning was properly escaped
        pnl_idx = header.index("pnl")
        _check("Row 1 pnl = 10000",
               rows[1][pnl_idx] == "10000",
               f"got '{rows[1][pnl_idx]}'")

        # Check that the quote in reasoning was properly escaped
        if "reasoning" in header:
            reason_idx = header.index("reasoning")
            reasoning_val = rows[1][reason_idx]
            _check("Reasoning with comma/quote survived CSV round-trip",
                   "strong buy" in reasoning_val,
                   f"got '{reasoning_val}'")

        # Empty journal CSV
        j2 = TradeJournal()
        csv_empty = j2.export_csv()
        _check("Empty CSV has header", "pnl" in csv_empty)


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: DailyRunner on Republic Day — scheduler skips
# ═══════════════════════════════════════════════════════════════════════════

def test_07_holiday_skip():
    """Republic Day (Jan 26 2025) is NOT a trading day."""
    print("\n[Test 7] DailyRunner on Republic Day — scheduler skips")

    republic_day = date(2025, 1, 26)  # Sunday in 2025, but also a holiday
    _check("Jan 26 2025 is NOT trading day", not is_trading_day(republic_day))

    # Independence Day
    independence_day = date(2025, 8, 15)
    _check("Aug 15 2025 is NOT trading day", not is_trading_day(independence_day))

    # Normal weekday
    normal_monday = date(2025, 1, 20)
    _check("Jan 20 2025 (Monday) IS trading day", is_trading_day(normal_monday))

    # Weekend
    saturday = date(2025, 1, 25)
    _check("Jan 25 2025 (Saturday) is NOT trading day", not is_trading_day(saturday))

    # 2026 Republic Day should also be blocked (Bug 9 fix)
    republic_day_2026 = date(2026, 1, 26)
    _check("Jan 26 2026 is NOT trading day (2026 holidays added)",
           not is_trading_day(republic_day_2026))


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: DailyRunner with Gemini unavailable — fallback mode
# ═══════════════════════════════════════════════════════════════════════════

def test_08_no_gemini_fallback():
    """Pipeline completes successfully without Gemini API key."""
    print("\n[Test 8] DailyRunner with Gemini unavailable — fallback mode")

    import os
    # Ensure no Gemini key is set
    old_key = os.environ.pop("GEMINI_API_KEY", None)
    old_gkey = os.environ.pop("GOOGLE_API_KEY", None)

    try:
        with tempfile.TemporaryDirectory() as d:
            cfg = _make_test_config(d)
            runner = DailyRunner(config=cfg)
            prices = _make_prices(0)

            report = runner.run(sim_date="2025-02-03", sim_prices=prices)

            _check("Pipeline succeeds without Gemini",
                   report.get("success", False))

            # Debate should be skipped
            steps = report.get("steps_completed", [])
            debate_steps = [s for s in steps if "debate" in s.lower()]
            _check("Debate skipped or fallback",
                   len(debate_steps) > 0 and any("skip" in s.lower() or "no api" in s.lower()
                                                   for s in debate_steps),
                   f"debate steps: {debate_steps}")

            # Sentiment should work (rule-based fallback or skip)
            sentiment_steps = [s for s in steps if "sentiment" in s.lower()]
            _check("Sentiment step completed",
                   len(sentiment_steps) > 0)

            # All 13 steps should be present
            _check("All 13 steps ran", len(steps) >= 13,
                   f"got {len(steps)} steps")
    finally:
        if old_key:
            os.environ["GEMINI_API_KEY"] = old_key
        if old_gkey:
            os.environ["GOOGLE_API_KEY"] = old_gkey


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: Daily report — 0 trades, all losses, no crash
# ═══════════════════════════════════════════════════════════════════════════

def test_09_report_edge_cases():
    """Daily report formatting with 0 trades and all-losses journal."""
    print("\n[Test 9] Daily report formatting — edge cases")

    with tempfile.TemporaryDirectory() as d:
        # ── Case A: 0 trades ──
        cfg = _make_test_config(d)
        cfg["max_positions"] = 0  # no trades
        runner = DailyRunner(config=cfg)
        prices = _make_prices(0)

        report = runner.run(sim_date="2025-03-05", sim_prices=prices)
        _check("0 trades: pipeline succeeds", report.get("success", False))
        _check("0 trades: summary exists", "summary" in report)
        _check("0 trades: win_rate = 0",
               report.get("summary", {}).get("win_rate", -1) == 0.0)

    with tempfile.TemporaryDirectory() as d:
        # ── Case B: all losses ──
        j = TradeJournal(journal_file=f"{d}/journal.json")
        for i in range(5):
            j.record_trade({
                "symbol": UNIVERSE[i], "player_id": "PLAYER_1",
                "direction": "LONG", "quantity": 100,
                "entry_price": 1000, "exit_price": 950,
                "pnl": -5000, "return_pct": -5.0,
                "bars_held": 3, "close_reason": "stop",
                "entry_date": "2025-01-10", "exit_date": "2025-01-13",
            })

        s = j.performance_summary()
        _check("All-loss: win_rate = 0%", s["win_rate"] == 0.0)
        _check("All-loss: total_pnl < 0", s["total_pnl"] < 0)
        _check("All-loss: profit_factor = 0", s["profit_factor"] == 0.0)
        _check("All-loss: avg_win = 0", s["avg_win"] == 0.0)
        _check("All-loss: avg_win_loss_ratio = 0",
               s["avg_win_loss_ratio"] == 0.0)
        _check("All-loss: sharpe < 0", s["sharpe"] < 0,
               f"got {s['sharpe']}")


# ═══════════════════════════════════════════════════════════════════════════
# Test 10: Metrics recording — 3 days → 3 snapshots
# ═══════════════════════════════════════════════════════════════════════════

def test_10_metrics_3_days():
    """Run 3 days, verify MetricsStore has exactly 3 snapshots."""
    print("\n[Test 10] Metrics recording — 3 days → 3 snapshots")

    with tempfile.TemporaryDirectory() as d:
        cfg = _make_test_config(d)
        runner = DailyRunner(config=cfg)

        base = date(2025, 3, 10)
        for day in range(3):
            sim_date = (base + timedelta(days=day)).isoformat()
            prices = _make_prices(day, seed=100 + day)
            runner.run(sim_date=sim_date, sim_prices=prices)

        store = MetricsStore(metrics_dir=Path(cfg["metrics_dir"]))
        snapshots = store.get_all_snapshots()

        _check("3 snapshots recorded", len(snapshots) == 3,
               f"got {len(snapshots)}")

        dates = [s.get("date") for s in snapshots]
        _check("Date 2025-03-10 present", "2025-03-10" in dates)
        _check("Date 2025-03-11 present", "2025-03-11" in dates)
        _check("Date 2025-03-12 present", "2025-03-12" in dates)

        # Each snapshot has required fields
        for snap in snapshots:
            _check(f"Snapshot {snap['date']} has portfolio",
                   "portfolio" in snap)
            _check(f"Snapshot {snap['date']} has equity > 0",
                   snap.get("portfolio", {}).get("equity_curve_value", 0) > 0)
            _check(f"Snapshot {snap['date']} has regime",
                   snap.get("regime", {}).get("current") in ("Bull", "Bear", "Sideways"))
            _check(f"Snapshot {snap['date']} has all 5 players",
                   len(snap.get("players", {})) == 5)


# ═══════════════════════════════════════════════════════════════════════════
# Test 11: Position persistence — open, reload, verify
# ═══════════════════════════════════════════════════════════════════════════

def test_11_persistence():
    """Open positions, create new tracker from same file, verify loaded."""
    print("\n[Test 11] Position persistence")

    with tempfile.TemporaryDirectory() as d:
        pfile = f"{d}/positions.json"

        # Tracker 1: open 2 positions
        t1 = PositionTracker(capital=1_000_000, positions_file=pfile,
                             trailing_stop_pct=0.03)

        t1.open_position("RELIANCE.NS", "PLAYER_1", "LONG", 100, 2500.0, "2025-01-10")
        t1.open_position("TCS.NS", "PLAYER_2", "SHORT", 50, 4000.0, "2025-01-10")

        t1.mark_to_market({"RELIANCE.NS": 2550.0, "TCS.NS": 3950.0})

        saved_capital = t1.current_capital
        saved_count = t1.open_position_count

        # Tracker 2: load from same file
        t2 = PositionTracker(capital=1_000_000, positions_file=pfile,
                             trailing_stop_pct=0.03)

        _check("Loaded position count matches",
               t2.open_position_count == saved_count,
               f"expected {saved_count}, got {t2.open_position_count}")

        _check("Loaded capital matches",
               abs(t2.current_capital - saved_capital) < 0.01,
               f"expected {saved_capital}, got {t2.current_capital}")

        # Verify position details survived round-trip
        rel_pos = t2.get_positions_for_symbol("RELIANCE.NS")
        _check("RELIANCE position loaded", len(rel_pos) == 1)
        if rel_pos:
            _check("RELIANCE direction = LONG", rel_pos[0].direction == "LONG")
            _check("RELIANCE quantity = 100", rel_pos[0].quantity == 100)
            _check("RELIANCE entry_price = 2500",
                   abs(rel_pos[0].entry_price - 2500.0) < 0.01)

        tcs_pos = t2.get_positions_for_symbol("TCS.NS")
        _check("TCS position loaded", len(tcs_pos) == 1)
        if tcs_pos:
            _check("TCS direction = SHORT", tcs_pos[0].direction == "SHORT")


# ═══════════════════════════════════════════════════════════════════════════
# Test 12: Cumulative P&L across days — matches sum of daily P&Ls
# ═══════════════════════════════════════════════════════════════════════════

def test_12_cumulative_pnl():
    """Run 5 days, verify cumulative P&L from journal matches equity change."""
    print("\n[Test 12] Cumulative P&L across multiple days")

    with tempfile.TemporaryDirectory() as d:
        cfg = _make_test_config(d)
        initial_capital = cfg["starting_capital"]
        runner = DailyRunner(config=cfg)

        base = date(2025, 4, 7)
        equities = []

        for day in range(5):
            sim_date = (base + timedelta(days=day)).isoformat()
            prices = _make_prices(day, seed=200 + day)
            report = runner.run(sim_date=sim_date, sim_prices=prices)
            eq = report.get("summary", {}).get("equity", 0)
            equities.append(eq)

        # Journal: total realised P&L from closed trades
        j = TradeJournal(journal_file=cfg["journal_file"])
        realised_pnl = sum(t.get("pnl", 0) for t in j.get_all_trades())

        # Tracker: remaining unrealised P&L
        tracker = PositionTracker(
            capital=initial_capital,
            positions_file=cfg["positions_file"],
        )
        unrealised_pnl = sum(p.unrealised_pnl for p in tracker.positions.values())

        # Total P&L = realised + unrealised
        total_pnl = realised_pnl + unrealised_pnl

        # Equity change = final equity - initial capital
        final_equity = equities[-1]
        equity_change = final_equity - initial_capital

        _check("Equity change matches total P&L",
               abs(equity_change - total_pnl) < 100,  # small float tolerance
               f"equity_change={equity_change:.2f}, total_pnl={total_pnl:.2f}")

        # Equity should be positive and reasonable
        _check("Final equity > 0", final_equity > 0)
        _check("Equity within reasonable range",
               500_000 < final_equity < 2_000_000,
               f"got {final_equity}")

        # All equities should be positive
        _check("All daily equities > 0", all(e > 0 for e in equities))


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Paper Trading — 12-Test Suite (with bug fixes)")
    print("=" * 70)

    test_01_open_close_pnl()
    test_02_zero_positions()
    test_03_stop_loss_triggers()
    test_04_trailing_stop_follows()
    test_05_journal_filter()
    test_06_csv_export()
    test_07_holiday_skip()
    test_08_no_gemini_fallback()
    test_09_report_edge_cases()
    test_10_metrics_3_days()
    test_11_persistence()
    test_12_cumulative_pnl()

    print(f"\n{'=' * 70}")
    print(f"  RESULTS: {PASS_COUNT} PASSED, {FAIL_COUNT} FAILED "
          f"(out of {PASS_COUNT + FAIL_COUNT})")
    print(f"{'=' * 70}")

    if FAIL_COUNT > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
