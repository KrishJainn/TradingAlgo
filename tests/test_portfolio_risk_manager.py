"""
Test suite for portfolio_risk_manager.py

Runs 6 deterministic tests against the PortfolioRiskManager:
  1. Crowding filter: 5 players long same stock → 40% haircut applied
  2. Single-stock 25% cap: reject a trade that would breach it
  3. Sector 40% cap: reject a trade that would breach it
  4. Circuit breaker (intraday -6%): triggers flatten_to_50
  5. Circuit breaker (peak -13%): triggers flatten_to_50
  6. Clean pass: a small diversified set of signals all approved

Each test is independent — no shared mutable state between tests.
"""

import sys
import os

# Ensure project root is on the path so we can import the module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_risk_manager import (
    PortfolioRiskManager,
    PositionRecord,
    RiskLimits,
    get_sector,
    Sector,
)


# ── Helpers ──────────────────────────────────────────────────────────────

passed = 0
failed = 0


def check(test_name: str, condition: bool, detail: str = ""):
    """Print PASS/FAIL and accumulate counts."""
    global passed, failed
    status = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}]  {test_name}{suffix}")


# ── Test 1: Crowding filter ──────────────────────────────────────────────

def test_crowding_filter():
    """
    5 players all go LONG RELIANCE.NS.
    With 5 agreeing, the 40% crowding haircut should apply.
    Each player's approved size should be original * 0.6.
    """
    print("\nTest 1: Crowding filter (5 players → 40% haircut)")
    print("-" * 55)

    cap = 1_000_000  # Rs 10 lakh — large enough that no limit binds
    rm = PortfolioRiskManager(total_capital=cap)

    # Each player proposes Rs 50k LONG RELIANCE — well under 25% cap
    # after haircut: 50k * 0.6 = 30k each, total = 150k = 15% < 25%
    signals = {
        f"PLAYER_{i}": {"RELIANCE.NS": ("LONG", 50_000, 0.90 - i * 0.01)}
        for i in range(1, 6)
    }

    approved, rejected = rm.filter_signals(signals, [], cap)

    # All 5 should be approved (total after haircut = 150k = 15%)
    total_approved = sum(
        len(sigs) for sigs in approved.values()
    )
    check("All 5 RELIANCE signals approved", total_approved == 5,
          f"got {total_approved}")

    # Verify every approved size is 60% of original 50k = 30,000
    all_sizes = [
        sz for player_sigs in approved.values()
        for sym, (d, sz, c) in player_sigs.items()
    ]
    all_haircut_correct = all(abs(s - 30_000) < 0.01 for s in all_sizes)
    check("Each approved size = 30,000 (60% of 50k)", all_haircut_correct,
          f"sizes: {all_sizes}")

    total_rejected = sum(len(s) for s in rejected.values())
    check("No rejections", total_rejected == 0,
          f"got {total_rejected} rejected")


# ── Test 2: Single-stock 25% cap ────────────────────────────────────────

def test_single_stock_cap():
    """
    Pre-load 20% in RELIANCE, then propose another 10%.
    20% + 10% = 30% > 25% → must reject.
    """
    print("\nTest 2: Single-stock 25% cap breach")
    print("-" * 55)

    cap = 500_000
    rm = PortfolioRiskManager(total_capital=cap)

    # Existing position: 20% of capital in RELIANCE
    existing = [
        PositionRecord(
            symbol="RELIANCE.NS", player="PLAYER_1", direction="LONG",
            quantity=40, entry_price=2500, current_price=2500,
        ),  # 40 * 2500 = 100,000 = 20% of 500k
    ]

    # One player proposes another 10% RELIANCE
    signals = {
        "PLAYER_2": {
            "RELIANCE.NS": ("LONG", 50_000, 0.85),  # would push to 30%
        },
    }

    approved, rejected = rm.filter_signals(signals, existing, cap)

    reliance_rejected = (
        "PLAYER_2" in rejected
        and "RELIANCE.NS" in rejected["PLAYER_2"]
    )
    check("RELIANCE rejected (would reach 30%)", reliance_rejected)

    if reliance_rejected:
        reason = rejected["PLAYER_2"]["RELIANCE.NS"][2]
        check("Reason mentions 'Single-stock limit'",
              "Single-stock limit" in reason, reason)
    else:
        check("Reason mentions 'Single-stock limit'", False,
              "trade was not rejected")

    total_approved = sum(len(s) for s in approved.values())
    check("Nothing approved", total_approved == 0, f"got {total_approved}")


# ── Test 3: Sector 40% cap ──────────────────────────────────────────────

def test_sector_cap():
    """
    Pre-load 35% in Banking sector (spread across stocks so no single-
    stock breach), then propose a new Banking trade that pushes past 40%.
    """
    print("\nTest 3: Sector 40% cap breach (Banking)")
    print("-" * 55)

    cap = 1_000_000
    rm = PortfolioRiskManager(total_capital=cap)

    # Existing Banking positions: 35% of capital across 3 stocks
    existing = [
        PositionRecord(
            symbol="HDFCBANK.NS", player="PLAYER_1", direction="LONG",
            quantity=75, entry_price=1600, current_price=1600,
        ),  # 75 * 1600 = 120,000 = 12%
        PositionRecord(
            symbol="ICICIBANK.NS", player="PLAYER_2", direction="LONG",
            quantity=100, entry_price=1100, current_price=1100,
        ),  # 100 * 1100 = 110,000 = 11%
        PositionRecord(
            symbol="SBIN.NS", player="PLAYER_3", direction="LONG",
            quantity=150, entry_price=800, current_price=800,
        ),  # 150 * 800 = 120,000 = 12%
    ]
    # Total Banking = 350,000 = 35%

    # New signal: another 80k in KOTAKBANK (Banking) → 35% + 8% = 43% > 40%
    signals = {
        "PLAYER_4": {
            "KOTAKBANK.NS": ("LONG", 80_000, 0.78),
        },
    }

    approved, rejected = rm.filter_signals(signals, existing, cap)

    kotak_rejected = (
        "PLAYER_4" in rejected
        and "KOTAKBANK.NS" in rejected["PLAYER_4"]
    )
    check("KOTAKBANK rejected (Banking would reach 43%)", kotak_rejected)

    if kotak_rejected:
        reason = rejected["PLAYER_4"]["KOTAKBANK.NS"][2]
        check("Reason mentions 'Sector limit'",
              "Sector limit" in reason, reason)
    else:
        check("Reason mentions 'Sector limit'", False,
              "trade was not rejected")


# ── Test 4: Circuit breaker — intraday -6% ──────────────────────────────

def test_circuit_breaker_intraday():
    """
    Start the day at 500k.  Equity drops to 470k (−6%).
    Intraday limit is 5%  →  circuit breaker must trigger.
    """
    print("\nTest 4: Circuit breaker — intraday -6% drawdown")
    print("-" * 55)

    cap = 500_000
    rm = PortfolioRiskManager(total_capital=cap)

    # Simulate day start
    rm.update_equity(cap, is_new_day=True)

    # Equity drops 6%
    rm.update_equity(cap * 0.94)

    cb = rm.check_circuit_breaker()

    check("Circuit breaker triggered", cb["triggered"] is True)
    check("Action is flatten_to_50", cb["action"] == "flatten_to_50")
    check("Intraday DD reported ≈ 6%",
          abs(cb["intraday_drawdown_pct"] - 0.06) < 0.001,
          f"got {cb['intraday_drawdown_pct']:.4f}")
    check("Reason mentions 'Intraday drawdown'",
          cb["reason"] is not None and "Intraday drawdown" in cb["reason"],
          str(cb["reason"]))


# ── Test 5: Circuit breaker — peak -13% ─────────────────────────────────

def test_circuit_breaker_peak():
    """
    Peak was 500k, now equity is 435k (−13% from peak), but intraday
    only dropped 3% (day started at 448k).
    Peak limit is 12%  →  must trigger.
    """
    print("\nTest 5: Circuit breaker — 13% from peak")
    print("-" * 55)

    cap = 500_000
    rm = PortfolioRiskManager(total_capital=cap)

    # Set peak
    rm.update_equity(cap, is_new_day=True)

    # Next day starts at 448k (within intraday tolerance)
    rm.update_equity(448_000, is_new_day=True)

    # Drop to 435k during the day — 3% intraday (under 5% limit)
    # but 13% from 500k peak (over 12% limit)
    rm.update_equity(435_000)

    cb = rm.check_circuit_breaker()

    intraday_dd = (448_000 - 435_000) / 448_000   # ≈ 2.9%
    peak_dd = (500_000 - 435_000) / 500_000        # = 13.0%

    check("Intraday DD under 5%",
          cb["intraday_drawdown_pct"] < 0.05,
          f"got {cb['intraday_drawdown_pct']:.4f}")
    check("Peak DD over 12%",
          cb["peak_drawdown_pct"] >= 0.12,
          f"got {cb['peak_drawdown_pct']:.4f}")
    check("Circuit breaker triggered", cb["triggered"] is True)
    check("Action is flatten_to_50", cb["action"] == "flatten_to_50")
    check("Reason mentions 'Peak drawdown'",
          cb["reason"] is not None and "Peak drawdown" in cb["reason"],
          str(cb["reason"]))


# ── Test 6: Clean pass — everything approved ─────────────────────────────

def test_clean_pass():
    """
    Small diversified signals from 3 players, no crowding, no breaches.
    Everything should be approved.
    """
    print("\nTest 6: Clean pass — diversified signals all approved")
    print("-" * 55)

    cap = 1_000_000
    rm = PortfolioRiskManager(total_capital=cap)

    signals = {
        "PLAYER_1": {
            "RELIANCE.NS": ("LONG", 80_000, 0.85),    # Energy 8%
        },
        "PLAYER_2": {
            "TCS.NS":      ("LONG", 70_000, 0.80),    # IT 7%
        },
        "PLAYER_3": {
            "HDFCBANK.NS": ("LONG", 90_000, 0.78),    # Banking 9%
            "SUNPHARMA.NS": ("LONG", 40_000, 0.72),   # Pharma 4%
        },
    }

    approved, rejected = rm.filter_signals(signals, [], cap)

    total_approved = sum(len(s) for s in approved.values())
    total_rejected = sum(len(s) for s in rejected.values())

    check("All 4 signals approved", total_approved == 4,
          f"got {total_approved}")
    check("Zero rejected", total_rejected == 0,
          f"got {total_rejected}")

    # No crowding haircut (max 1 player per stock) — sizes should be unchanged
    for player, sigs in approved.items():
        for sym, (d, sz, c) in sigs.items():
            original = signals[player][sym][1]
            check(f"  {sym} size unchanged ({original:,.0f})",
                  abs(sz - original) < 0.01,
                  f"got {sz:,.0f}")


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  portfolio_risk_manager.py — Test Suite")
    print("=" * 60)

    test_crowding_filter()
    test_single_stock_cap()
    test_sector_cap()
    test_circuit_breaker_intraday()
    test_circuit_breaker_peak()
    test_clean_pass()

    print("\n" + "=" * 60)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {failed} TEST(S) FAILED")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
