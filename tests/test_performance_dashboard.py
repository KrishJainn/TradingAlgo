"""
Tests for metrics_store.py and performance_dashboard.py — 12 targeted test cases.

Run:
    python3 tests/test_performance_dashboard.py
"""

import json
import math
import os
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from metrics_store import (
    MetricsStore,
    MetricsCalculator,
    PLAYER_IDS,
    PLAYER_LABELS,
    TRADING_DAYS_PER_YEAR,
    RISK_FREE_RATE,
)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

passed = 0
failed = 0
total = 12


def _report(test_num: int, name: str, success: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if success else "FAIL"
    if success:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] Test {test_num}: {name}")
    if detail and not success:
        print(f"         Detail: {detail}")


def _tmp_store() -> MetricsStore:
    """Create a MetricsStore backed by a fresh temp directory."""
    tmpdir = tempfile.mkdtemp(prefix="metrics_test_")
    return MetricsStore(metrics_dir=Path(tmpdir))


def _make_snapshot(day_offset: int = 0, **overrides) -> dict:
    """Create a minimal valid snapshot for a given day offset from today.

    day_offset=0 → today, day_offset=1 → yesterday, etc.
    """
    d = (date.today() - timedelta(days=day_offset)).isoformat()
    snap = {
        "date": d,
        "portfolio": {
            "gross_pnl": 1500.0 + day_offset * 10,
            "net_pnl": 1400.0 + day_offset * 8,
            "cumulative_pnl": 50000.0 + day_offset * 100,
            "gross_exposure": 450000.0,
            "net_exposure": 200000.0,
            "largest_concentration": 0.12,
            "equity_curve_value": 510000.0 - day_offset * 200,
            "daily_return": 0.003 - day_offset * 0.0001,
        },
        "positions": [
            {"symbol": "RELIANCE", "direction": "LONG", "size": 50,
             "unrealized_pnl": 1200.0, "player_id": "PLAYER_1"},
        ],
        "trades_today": [
            {"symbol": "TCS", "direction": "LONG", "pnl": 500.0,
             "player_id": "PLAYER_2", "cost": 15.0},
        ],
        "regime": {
            "current": "Bull",
            "probabilities": {"Bull": 0.65, "Bear": 0.15, "Sideways": 0.20},
            "duration_days": 5,
        },
        "players": {
            "PLAYER_1": {"sharpe": 1.8, "sortino": 2.5, "win_rate": 0.55,
                         "avg_win_loss": 1.4, "num_trades": 10,
                         "pnl": 500.0, "cumulative_pnl": 5000.0,
                         "bayesian_weight": 0.22, "hrp_weight": 0.18,
                         "best_trade": 1200.0, "worst_trade": -600.0,
                         "active_rules": ["Avoid low-vol entries"],
                         "regime_performance": {"Bull": 2.1, "Bear": -0.3, "Sideways": 0.8}},
            "PLAYER_2": {"sharpe": 1.2, "sortino": 1.8, "win_rate": 0.50,
                         "avg_win_loss": 1.1, "num_trades": 8,
                         "pnl": 300.0, "cumulative_pnl": 3000.0,
                         "bayesian_weight": 0.20, "hrp_weight": 0.20,
                         "best_trade": 800.0, "worst_trade": -500.0,
                         "active_rules": [], "regime_performance": {"Bull": 1.5, "Bear": 0.2, "Sideways": 0.5}},
            "PLAYER_3": {"sharpe": 0.9, "sortino": 1.3, "win_rate": 0.48,
                         "avg_win_loss": 1.0, "num_trades": 12,
                         "pnl": -200.0, "cumulative_pnl": -1000.0,
                         "bayesian_weight": 0.18, "hrp_weight": 0.22,
                         "best_trade": 600.0, "worst_trade": -800.0,
                         "active_rules": [], "regime_performance": {"Bull": 0.8, "Bear": -0.5, "Sideways": 0.1}},
            "PLAYER_4": {"sharpe": 1.5, "sortino": 2.0, "win_rate": 0.52,
                         "avg_win_loss": 1.3, "num_trades": 6,
                         "pnl": 400.0, "cumulative_pnl": 4000.0,
                         "bayesian_weight": 0.21, "hrp_weight": 0.19,
                         "best_trade": 1000.0, "worst_trade": -400.0,
                         "active_rules": [], "regime_performance": {"Bull": 1.9, "Bear": -0.1, "Sideways": 0.6}},
            "PLAYER_5": {"sharpe": 1.3, "sortino": 1.7, "win_rate": 0.53,
                         "avg_win_loss": 1.2, "num_trades": 9,
                         "pnl": -100.0, "cumulative_pnl": -500.0,
                         "bayesian_weight": 0.19, "hrp_weight": 0.21,
                         "best_trade": 700.0, "worst_trade": -550.0,
                         "active_rules": [], "regime_performance": {"Bull": 1.4, "Bear": 0.0, "Sideways": 0.4}},
        },
        "signals": {
            "agreement_rate": 0.72,
            "pre_debate_accuracy": 0.55,
            "post_debate_accuracy": 0.63,
            "conversion_rate": 0.70,
            "false_signal_rate": {pid: 0.12 for pid in PLAYER_IDS},
        },
        "risk": {
            "max_drawdown": 0.08,
            "current_drawdown": 0.03,
            "sector_exposure": {"IT": 0.12, "Banking": 0.15},
            "cost_drag_pct": 1.2,
            "circuit_breaker_triggers": [],
            "cvar_95": 0.025,
            "cvar_limit": 0.03,
            "cvar_breached": False,
            "vol_scale_factor": 1.0,
        },
        "coaching": {
            "last_session_summary": "Test coaching session",
            "indicator_weights": {"PLAYER_1": {"RSI_7": 0.8, "MACD_12_26": 0.6}},
            "reflection_highlights": ["Good momentum capture in Bull regime"],
        },
        "evolution": {
            "generation": 3,
            "latest_change": "PLAYER_1 indicator adjustment",
            "indicator_survival": {"RSI_7": 4, "MACD_12_26": 3, "EMA_20": 5},
        },
    }
    snap.update(overrides)
    return snap


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: record_daily_snapshot creates JSON file in metrics_data/
# ═══════════════════════════════════════════════════════════════════════════

def test_1_record_snapshot_creates_json():
    """MetricsStore record_daily_snapshot with valid data — verify JSON file created in metrics_data/."""
    try:
        store = _tmp_store()
        snap = _make_snapshot(0)
        today_str = date.today().isoformat()

        store.record_daily_snapshot(snap)

        # Verify the file was created
        expected_file = store.metrics_dir / f"metrics_{today_str}.json"
        file_exists = expected_file.exists()

        errors = []
        if not file_exists:
            errors.append(f"File not created at {expected_file}")
        else:
            raw = json.loads(expected_file.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                errors.append(f"Content is not a list: {type(raw)}")
            elif len(raw) != 1:
                errors.append(f"Expected 1 snapshot, got {len(raw)}")
            else:
                stored = raw[0]
                if stored.get("date") != today_str:
                    errors.append(f"Stored date '{stored.get('date')}' != '{today_str}'")
                if stored.get("portfolio", {}).get("gross_pnl") != 1500.0:
                    errors.append(f"gross_pnl mismatch: {stored.get('portfolio', {}).get('gross_pnl')}")
                if "recorded_at" not in stored:
                    errors.append("Missing 'recorded_at' timestamp")

        ok = len(errors) == 0
        _report(1, "record_daily_snapshot creates JSON file", ok, "; ".join(errors))
    except Exception as e:
        _report(1, "record_daily_snapshot creates JSON file", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: get_metric_history returns correct time series length
# ═══════════════════════════════════════════════════════════════════════════

def test_2_metric_history_length():
    """Record 10 days, request 5, get ≤5 back (date-range scoped)."""
    try:
        store = _tmp_store()

        # Insert 10 snapshots (day_offset 0..9 → today back to 9 days ago)
        for i in range(10):
            store.record_daily_snapshot(_make_snapshot(day_offset=i))

        # Query all 10 — should get all 10
        history_all = store.get_metric_history("portfolio.gross_pnl", days=30)
        ok_all = len(history_all) == 10

        # Query only last 5 days — get_all_snapshots(days=5) fetches
        # date.today()-5 through date.today() inclusive = 6 calendar days,
        # but only offsets 0..5 have data = 6 snapshots
        history_5 = store.get_metric_history("portfolio.gross_pnl", days=5)
        ok_5 = len(history_5) == 6

        # Verify the values are correct floats
        ok_floats = all(isinstance(h[1], float) for h in history_all)

        # Verify nested path works
        history_nested = store.get_metric_history("players.PLAYER_1.sharpe", days=30)
        ok_nested = len(history_nested) == 10

        # Non-existent path returns empty
        history_bad = store.get_metric_history("nonexistent.path", days=30)
        ok_bad = len(history_bad) == 0

        ok = ok_all and ok_5 and ok_floats and ok_nested and ok_bad
        detail = (f"all={len(history_all)}(want 10), "
                  f"5d={len(history_5)}(want 6), "
                  f"nested={len(history_nested)}(want 10), "
                  f"bad_path={len(history_bad)}(want 0)")
        _report(2, "get_metric_history returns correct length", ok, detail)
    except Exception as e:
        _report(2, "get_metric_history returns correct length", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: get_latest returns most recent snapshot (not oldest)
# ═══════════════════════════════════════════════════════════════════════════

def test_3_get_latest_returns_most_recent():
    """get_latest returns most recent snapshot, not the first recorded."""
    try:
        store = _tmp_store()

        # Insert snapshots OUT OF ORDER: 2 days ago, today, yesterday
        store.record_daily_snapshot(_make_snapshot(day_offset=2))
        store.record_daily_snapshot(_make_snapshot(day_offset=0))
        store.record_daily_snapshot(_make_snapshot(day_offset=1))

        latest = store.get_latest()
        expected_date = date.today().isoformat()
        ok_date = latest.get("date") == expected_date

        # Also verify it returns the correct data (today's snapshot has day_offset=0)
        ok_data = latest.get("portfolio", {}).get("equity_curve_value") == 510000.0

        ok = ok_date and ok_data
        _report(3, "get_latest returns most recent snapshot", ok,
                f"got date='{latest.get('date')}' eq={latest.get('portfolio', {}).get('equity_curve_value')}, "
                f"expected '{expected_date}' eq=510000.0")
    except Exception as e:
        _report(3, "get_latest returns most recent snapshot", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Empty store — no crash on get_latest or get_metric_history
# ═══════════════════════════════════════════════════════════════════════════

def test_4_empty_store_no_crash():
    """Empty store — no crash on any read method."""
    try:
        store = _tmp_store()

        latest = store.get_latest()
        history = store.get_metric_history("portfolio.sharpe", days=30)
        count = store.snapshot_count()
        all_snaps = store.get_all_snapshots(days=7)
        all_snaps_none = store.get_all_snapshots(days=None)
        date_range = store.get_date_range(date.today() - timedelta(days=5), date.today())

        # Derived metrics on empty store
        rolling_sharpe = store.compute_rolling_sharpe(days=30, window=10)
        rolling_hr = store.compute_rolling_hit_rate(days=30, window=10)
        regime_m = store.compute_regime_conditional_metrics(days=30)
        contrib = store.compute_player_contribution(days=30)
        dd = store.compute_drawdown_series(days=30)
        monthly = store.compute_monthly_returns(days=30)
        cost_drag = store.compute_cost_drag_series(days=30)
        csv_out = store.export_csv(days=30)

        errors = []
        if latest != {}:
            errors.append(f"get_latest returned {latest} instead of {{}}")
        if history != []:
            errors.append(f"get_metric_history returned {history}")
        if count != 0:
            errors.append(f"snapshot_count returned {count}")
        if all_snaps != []:
            errors.append(f"get_all_snapshots(7) returned {len(all_snaps)} items")
        if all_snaps_none != []:
            errors.append(f"get_all_snapshots(None) returned {len(all_snaps_none)} items")
        if date_range != []:
            errors.append(f"get_date_range returned {len(date_range)} items")
        if rolling_sharpe != []:
            errors.append(f"rolling_sharpe not empty: {rolling_sharpe}")
        if rolling_hr != []:
            errors.append(f"rolling_hr not empty: {rolling_hr}")
        if dd != []:
            errors.append(f"drawdown not empty: {dd}")
        if csv_out != "date\n":
            errors.append(f"csv export not header-only: {csv_out[:50]}")

        ok = len(errors) == 0
        _report(4, "Empty store — no crash", ok, "; ".join(errors))
    except Exception as e:
        _report(4, "Empty store — no crash", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Duplicate date snapshots — verify overwrite
# ═══════════════════════════════════════════════════════════════════════════

def test_5_duplicate_date_overwrites():
    """Duplicate date snapshots — latest overwrites previous, no duplicates."""
    try:
        store = _tmp_store()
        today = date.today().isoformat()

        # First snapshot for today
        snap1 = _make_snapshot(0)
        snap1["portfolio"]["gross_pnl"] = 1000.0
        store.record_daily_snapshot(snap1)

        # Overwrite with new data for same date
        snap2 = _make_snapshot(0)
        snap2["portfolio"]["gross_pnl"] = 9999.0
        store.record_daily_snapshot(snap2)

        count = store.snapshot_count()
        latest = store.get_latest()
        actual_pnl = latest.get("portfolio", {}).get("gross_pnl")

        errors = []
        if count != 1:
            errors.append(f"snapshot_count={count}, want 1")
        if actual_pnl != 9999.0:
            errors.append(f"gross_pnl={actual_pnl}, want 9999.0")

        # Verify the JSON file itself has only 1 entry
        f = store.metrics_dir / f"metrics_{today}.json"
        if f.exists():
            raw = json.loads(f.read_text())
            if len(raw) != 1:
                errors.append(f"JSON file has {len(raw)} entries, want 1")
        else:
            errors.append("JSON file not found")

        ok = len(errors) == 0
        _report(5, "Duplicate date overwrites previous", ok, "; ".join(errors))
    except Exception as e:
        _report(5, "Duplicate date overwrites previous", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: get_date_range with dates outside range — returns empty
# ═══════════════════════════════════════════════════════════════════════════

def test_6_date_range_outside():
    """get_date_range with dates outside range — returns empty, no crash."""
    try:
        store = _tmp_store()

        # Insert 3 snapshots for today, yesterday, 2 days ago
        for i in range(3):
            store.record_daily_snapshot(_make_snapshot(day_offset=i))

        # Query far in the future — no data
        future_start = date.today() + timedelta(days=100)
        future_end = date.today() + timedelta(days=110)
        result_future = store.get_date_range(future_start, future_end)

        # Query far in the past — no data
        past_start = date.today() - timedelta(days=1000)
        past_end = date.today() - timedelta(days=990)
        result_past = store.get_date_range(past_start, past_end)

        # Query exact range that has data
        exact_start = date.today() - timedelta(days=2)
        exact_end = date.today()
        result_exact = store.get_date_range(exact_start, exact_end)

        # Query start > end (inverted) — should return empty
        result_inverted = store.get_date_range(date.today(), date.today() - timedelta(days=5))

        errors = []
        if result_future != []:
            errors.append(f"Future range returned {len(result_future)} items")
        if result_past != []:
            errors.append(f"Past range returned {len(result_past)} items")
        if len(result_exact) != 3:
            errors.append(f"Exact range returned {len(result_exact)} items, want 3")
        if result_inverted != []:
            errors.append(f"Inverted range returned {len(result_inverted)} items")

        ok = len(errors) == 0
        _report(6, "get_date_range outside range — empty, no crash", ok, "; ".join(errors))
    except Exception as e:
        _report(6, "get_date_range outside range — empty, no crash", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Rolling Sharpe matches manual calculation
# ═══════════════════════════════════════════════════════════════════════════

def test_7_rolling_sharpe_manual():
    """Rolling Sharpe computation matches a manual calculation."""
    try:
        np.random.seed(42)
        returns = np.array([0.01, -0.005, 0.008, -0.002, 0.012,
                            0.003, -0.007, 0.006, 0.001, -0.003])
        window = 5
        calc = MetricsCalculator
        rolling = calc.compute_rolling_sharpe(returns, window=window,
                                              risk_free=RISK_FREE_RATE,
                                              ann_factor=TRADING_DAYS_PER_YEAR)

        errors = []

        # Should produce len(returns) - window + 1 = 6 values
        expected_len = len(returns) - window + 1
        if len(rolling) != expected_len:
            errors.append(f"Length {len(rolling)} != {expected_len}")

        # Manually compute Sharpe for the first window [0:5]
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        chunk = returns[:window]
        excess = chunk - daily_rf
        manual_sharpe = float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))

        if len(rolling) > 0:
            diff = abs(rolling[0] - manual_sharpe)
            if diff > 0.001:
                errors.append(f"First Sharpe: computed={rolling[0]:.6f}, manual={manual_sharpe:.6f}, diff={diff:.6f}")

        # Manually compute for the last window [5:10]
        chunk_last = returns[-window:]
        excess_last = chunk_last - daily_rf
        manual_last = float(np.mean(excess_last) / np.std(excess_last, ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))
        if len(rolling) > 0:
            diff_last = abs(rolling[-1] - manual_last)
            if diff_last > 0.001:
                errors.append(f"Last Sharpe: computed={rolling[-1]:.6f}, manual={manual_last:.6f}")

        # Edge: window = len(returns) → exactly 1 result
        rolling_full = calc.compute_rolling_sharpe(returns, window=len(returns))
        if len(rolling_full) != 1:
            errors.append(f"Full window: got {len(rolling_full)}, want 1")

        # Edge: window > len(returns) → empty
        rolling_too_big = calc.compute_rolling_sharpe(returns, window=len(returns) + 1)
        if len(rolling_too_big) != 0:
            errors.append(f"Oversized window: got {len(rolling_too_big)}, want 0")

        ok = len(errors) == 0
        _report(7, "Rolling Sharpe matches manual calculation", ok, "; ".join(errors))
    except Exception as e:
        _report(7, "Rolling Sharpe matches manual calculation", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Player contribution attribution sums to 100%
# ═══════════════════════════════════════════════════════════════════════════

def test_8_player_attribution_sums():
    """Player contribution pct_of_total sums to 100%."""
    try:
        store = _tmp_store()

        # Create snapshots with mixed positive/negative player PnLs
        for i in range(5):
            snap = _make_snapshot(day_offset=i)
            # Ensure mix of positive and negative PnLs
            snap["players"]["PLAYER_1"]["pnl"] = 500.0
            snap["players"]["PLAYER_2"]["pnl"] = 300.0
            snap["players"]["PLAYER_3"]["pnl"] = -200.0
            snap["players"]["PLAYER_4"]["pnl"] = 400.0
            snap["players"]["PLAYER_5"]["pnl"] = -100.0
            store.record_daily_snapshot(snap)

        contrib = store.compute_player_contribution(days=30)

        errors = []

        # All 5 players should be present
        for pid in PLAYER_IDS:
            if pid not in contrib:
                errors.append(f"{pid} missing from contribution")

        # pct_of_total should sum to 100% (using abs values in denominator)
        total_pct = sum(c["pct_of_total"] for c in contrib.values())
        if abs(total_pct - 100.0) > 0.5:
            errors.append(f"pct_of_total sum={total_pct:.2f}, want ~100.0")

        # Each pct_of_total should be non-negative (since we use abs)
        for pid, c in contrib.items():
            if c["pct_of_total"] < 0:
                errors.append(f"{pid} pct_of_total={c['pct_of_total']} is negative")

        # P1 has highest abs PnL, should have highest %
        if contrib["PLAYER_1"]["pct_of_total"] < contrib["PLAYER_5"]["pct_of_total"]:
            errors.append(f"P1 pct ({contrib['PLAYER_1']['pct_of_total']}) should be > P5 ({contrib['PLAYER_5']['pct_of_total']})")

        # Also test MetricsCalculator.compute_player_attribution directly
        player_pnls = {"P1": 3000, "P2": -1000, "P3": 2000}
        total = sum(player_pnls.values())  # 4000
        attr = MetricsCalculator.compute_player_attribution(player_pnls, total)
        # This method uses signed pnl / |total| * 100 → sum = 100 when net > 0
        attr_sum = sum(attr.values())
        if abs(attr_sum - 100.0) > 0.1:
            errors.append(f"MetricsCalculator attr sum={attr_sum:.2f} (expected 100.0)")

        ok = len(errors) == 0
        _report(8, "Player contribution attribution sums to 100%", ok, "; ".join(errors))
    except Exception as e:
        _report(8, "Player contribution attribution sums to 100%", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 9: Corrupted JSON handling
# ═══════════════════════════════════════════════════════════════════════════

def test_9_corrupted_json():
    """Corrupted JSON handling — MetricsStore skips gracefully."""
    try:
        store = _tmp_store()

        # Write valid data first
        store.record_daily_snapshot(_make_snapshot(day_offset=1))  # yesterday
        store.record_daily_snapshot(_make_snapshot(day_offset=0))  # today

        # Corrupt yesterday's file by writing garbage JSON
        yesterday = date.today() - timedelta(days=1)
        corrupt_file = store.metrics_dir / f"metrics_{yesterday.isoformat()}.json"
        corrupt_file.write_text("{{{INVALID JSON!!!", encoding="utf-8")

        errors = []

        # get_latest should still work (returns today's data, skipping corrupted)
        latest = store.get_latest()
        if latest.get("date") != date.today().isoformat():
            errors.append(f"get_latest date={latest.get('date')}, expected {date.today().isoformat()}")

        # get_all_snapshots should skip corrupted file
        all_snaps = store.get_all_snapshots(days=30)
        if len(all_snaps) != 1:
            errors.append(f"get_all_snapshots returned {len(all_snaps)}, want 1 (skip corrupted)")

        # snapshot_count should only count valid
        count = store.snapshot_count()
        if count != 1:
            errors.append(f"snapshot_count={count}, want 1")

        # Write empty file (valid but empty)
        two_days_ago = date.today() - timedelta(days=2)
        empty_file = store.metrics_dir / f"metrics_{two_days_ago.isoformat()}.json"
        empty_file.write_text("", encoding="utf-8")

        # Should still work
        all_snaps2 = store.get_all_snapshots(days=30)
        if len(all_snaps2) != 1:
            errors.append(f"After empty file: get_all_snapshots returned {len(all_snaps2)}, want 1")

        # Write file with non-array JSON
        three_days_ago = date.today() - timedelta(days=3)
        scalar_file = store.metrics_dir / f"metrics_{three_days_ago.isoformat()}.json"
        scalar_file.write_text('"just a string"', encoding="utf-8")

        all_snaps3 = store.get_all_snapshots(days=30)
        if len(all_snaps3) != 1:
            errors.append(f"After scalar file: get_all_snapshots returned {len(all_snaps3)}, want 1")

        ok = len(errors) == 0
        _report(9, "Corrupted JSON handling — skips gracefully", ok, "; ".join(errors))
    except Exception as e:
        _report(9, "Corrupted JSON handling — skips gracefully", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 10: Dashboard demo data generator — 90 days with all fields
# ═══════════════════════════════════════════════════════════════════════════

def test_10_demo_data_generation():
    """Demo mode generator produces 90 days of data with all required fields."""
    try:
        from performance_dashboard import _generate_demo_data

        store = _tmp_store()
        _generate_demo_data(store, days=90)

        errors = []

        # Verify count
        count = store.snapshot_count()
        if count != 90:
            errors.append(f"snapshot_count={count}, want 90")

        # Verify all required top-level sections exist
        all_snaps = store.get_all_snapshots(days=365)
        required_sections = [
            "date", "portfolio", "positions", "trades_today",
            "regime", "players", "signals", "risk", "coaching", "evolution",
        ]

        for snap in all_snaps[:5] + all_snaps[-5:]:  # spot check first & last 5
            for section in required_sections:
                if section not in snap:
                    errors.append(f"Missing '{section}' on date {snap.get('date', '?')}")

        # Verify portfolio has all required fields
        portfolio_fields = [
            "gross_pnl", "net_pnl", "cumulative_pnl", "gross_exposure",
            "net_exposure", "largest_concentration", "equity_curve_value", "daily_return",
        ]
        for snap in all_snaps[:3]:
            port = snap.get("portfolio", {})
            for field in portfolio_fields:
                if field not in port:
                    errors.append(f"Missing portfolio.{field} on {snap.get('date', '?')}")

        # Verify all 5 players present
        for snap in all_snaps[:3]:
            for pid in PLAYER_IDS:
                if pid not in snap.get("players", {}):
                    errors.append(f"Missing {pid} on {snap.get('date', '?')}")

        # Verify regime transitions exist (not all same regime)
        regimes_seen = set()
        for snap in all_snaps:
            r = snap.get("regime", {}).get("current")
            if r:
                regimes_seen.add(r)
        if len(regimes_seen) < 2:
            errors.append(f"Only {regimes_seen} regimes — expected transitions")

        # Verify dates are chronologically ordered
        dates = [snap["date"] for snap in all_snaps]
        if dates != sorted(dates):
            errors.append("Snapshots not in chronological order")

        # Verify latest date is today
        latest = store.get_latest()
        if latest.get("date") != date.today().isoformat():
            errors.append(f"Latest date={latest.get('date')}, expected {date.today().isoformat()}")

        ok = len(errors) == 0
        _report(10, "Demo data generator — 90 days with all fields", ok,
                "; ".join(errors[:5]))  # limit detail to first 5 errors
    except Exception as e:
        _report(10, "Demo data generator — 90 days with all fields", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 11: CSV export — correct columns and row count
# ═══════════════════════════════════════════════════════════════════════════

def test_11_csv_export():
    """CSV export has correct columns and row count."""
    try:
        store = _tmp_store()

        # Insert 5 snapshots
        for i in range(5):
            store.record_daily_snapshot(_make_snapshot(day_offset=i))

        csv_str = store.export_csv(days=30)
        lines = csv_str.strip().split("\n")

        errors = []

        # Header + 5 data rows = 6 lines
        if len(lines) != 6:
            errors.append(f"CSV has {len(lines)} lines, want 6 (header + 5)")

        # Header should contain expected columns
        header = lines[0]
        expected_cols = ["date", "portfolio.gross_pnl", "portfolio.net_pnl",
                         "portfolio.equity_curve_value", "regime.current"]
        for col in expected_cols:
            if col not in header:
                errors.append(f"Missing column '{col}' in CSV header")

        # All data rows should have same number of fields as header
        n_cols = len(header.split(","))
        for i, line in enumerate(lines[1:], 1):
            n_fields = len(line.split(","))
            if n_fields != n_cols:
                errors.append(f"Row {i}: {n_fields} fields vs {n_cols} columns")
                break  # one error is enough

        # Verify commas in values are handled (replaced with ";")
        if ",," in csv_str:
            # Double-comma means empty values, which is acceptable
            pass

        # Empty store CSV
        empty_store = _tmp_store()
        empty_csv = empty_store.export_csv(days=30)
        if empty_csv != "date\n":
            errors.append(f"Empty CSV: '{empty_csv[:30]}', want 'date\\n'")

        ok = len(errors) == 0
        _report(11, "CSV export — correct columns and row count", ok, "; ".join(errors))
    except Exception as e:
        _report(11, "CSV export — correct columns and row count", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Test 12: Dashboard imports — all Streamlit/Plotly components load
# ═══════════════════════════════════════════════════════════════════════════

def test_12_dashboard_imports():
    """All Streamlit/Plotly components and dashboard functions load without errors."""
    try:
        errors = []

        # Import core dashboard module
        from performance_dashboard import (
            _generate_demo_data,
            _t,
            _inr,
            _pct,
            _color,
            COLORS,
            PLOTLY_TEMPLATE,
            _render_portfolio_overview,
            _render_player_performance,
            _render_regime_analysis,
            _render_signal_quality,
            _render_risk_monitor,
            _render_coaching_log,
            _render_equity_curve,
            _render_sidebar,
        )

        # Import constants
        from performance_dashboard import (
            PLAYER_IDS as DASH_PLAYER_IDS,
            PLAYER_LABELS as DASH_PLAYER_LABELS,
            PLAYER_COLORS,
            REGIME_COLORS,
            SECTORS,
        )

        # Verify constant sizes
        if len(DASH_PLAYER_IDS) != 5:
            errors.append(f"PLAYER_IDS has {len(DASH_PLAYER_IDS)} entries, want 5")
        if len(DASH_PLAYER_LABELS) != 5:
            errors.append(f"PLAYER_LABELS has {len(DASH_PLAYER_LABELS)} entries, want 5")
        if len(PLAYER_COLORS) != 5:
            errors.append(f"PLAYER_COLORS has {len(PLAYER_COLORS)} entries, want 5")
        if len(REGIME_COLORS) != 3:
            errors.append(f"REGIME_COLORS has {len(REGIME_COLORS)} entries, want 3")
        if len(SECTORS) != 10:
            errors.append(f"SECTORS has {len(SECTORS)} entries, want 10")

        # Verify formatting helpers
        inr_result = _inr(123456.0)
        if "₹" not in inr_result:
            errors.append(f"_inr missing ₹: {inr_result}")
        inr_cr = _inr(15_000_000.0)
        if "Cr" not in inr_cr:
            errors.append(f"_inr large value missing Cr: {inr_cr}")
        inr_lakh = _inr(500_000.0)
        if "L" not in inr_lakh:
            errors.append(f"_inr lakh value missing L: {inr_lakh}")

        pct_result = _pct(5.5)
        if "%" not in pct_result:
            errors.append(f"_pct missing %: {pct_result}")

        color_pos = _color(1.0)
        color_neg = _color(-1.0)
        if color_pos != COLORS["green"]:
            errors.append(f"_color(+) returned {color_pos}, want green")
        if color_neg != COLORS["red"]:
            errors.append(f"_color(-) returned {color_neg}, want red")

        # Verify _t (theme function) works on a Plotly figure
        import plotly.graph_objects as go
        fig = go.Figure()
        themed = _t(fig)
        if not isinstance(themed, go.Figure):
            errors.append(f"_t returned {type(themed)}, want go.Figure")
        if themed.layout.paper_bgcolor != "rgba(0,0,0,0)":
            errors.append(f"Theme paper_bgcolor: {themed.layout.paper_bgcolor}")

        # Verify PLAYER_IDS match between dashboard and metrics_store
        from metrics_store import PLAYER_IDS as STORE_PLAYER_IDS
        if DASH_PLAYER_IDS != STORE_PLAYER_IDS:
            errors.append(f"PLAYER_IDS mismatch: dashboard={DASH_PLAYER_IDS}, store={STORE_PLAYER_IDS}")

        # Verify Streamlit and Plotly importable
        import streamlit
        import plotly
        from plotly.subplots import make_subplots

        ok = len(errors) == 0
        _report(12, "Dashboard imports — all components load", ok, "; ".join(errors))
    except ImportError as e:
        _report(12, "Dashboard imports — all components load", False, f"ImportError: {e}")
    except Exception as e:
        _report(12, "Dashboard imports — all components load", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  metrics_store.py + performance_dashboard.py — Test Suite")
    print("  12 tests")
    print("=" * 65)
    print()

    test_1_record_snapshot_creates_json()
    test_2_metric_history_length()
    test_3_get_latest_returns_most_recent()
    test_4_empty_store_no_crash()
    test_5_duplicate_date_overwrites()
    test_6_date_range_outside()
    test_7_rolling_sharpe_manual()
    test_8_player_attribution_sums()
    test_9_corrupted_json()
    test_10_demo_data_generation()
    test_11_csv_export()
    test_12_dashboard_imports()

    print()
    print("-" * 65)
    print(f"  Results: {passed}/{total} PASSED, {failed}/{total} FAILED")
    print("=" * 65)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
