"""
Comprehensive test script for regime-aware integration (v2).

Tests:
  1. Import RegimeDetector, RegimeAwareCoach, and all 5 regime-aware players
  2. Verify each player has indicator adjustments for all 3 regimes
  3. Verify specific position sizing multipliers
  4. Mock bear detector → tighter risk limits on PortfolioRiskManager
  5. Mock bull detector → slippage_multiplier=0.8 on TransactionCostModel
  6. Regime weights from coach sum to 1.0 for each regime
  7. Transition logging: bull→bear→sideways, all 3 logged with timestamps
  8. Player performance tracking stores separate metrics per regime
  9. Gemini prompt template includes regime context fields
 10. Coach doesn't crash on low-confidence predictions (~0.33)
"""

import sys
import os
import json
import tempfile
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    tag = "PASS" if condition else "FAIL"
    if condition:
        passed += 1
    else:
        failed += 1
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}]  {name}{suffix}")


# ── Mock helpers ─────────────────────────────────────────────────────────────

class MockDetector:
    """Configurable mock for RegimeDetector."""
    def __init__(self, regime="Bull", conf=0.80, duration=10):
        self._regime = regime
        self._conf = conf
        self._duration = duration

    def predict(self, ohlcv, lookback=5):
        probs = {"Bull": 0.0, "Bear": 0.0, "Sideways": 0.0}
        probs[self._regime] = self._conf
        remaining = 1.0 - self._conf
        others = [r for r in probs if r != self._regime]
        for r in others:
            probs[r] = round(remaining / len(others), 4)
        return self._regime, probs, self._duration

    def get_regime_weights(self, regime):
        W = {
            "Bull":     {"Momentum": 0.30, "Aggressive": 0.25, "Balanced": 0.20,
                         "VolBreakout": 0.10, "Conservative": 0.15},
            "Bear":     {"Momentum": 0.10, "Aggressive": 0.10, "Balanced": 0.15,
                         "VolBreakout": 0.30, "Conservative": 0.35},
            "Sideways": {"Momentum": 0.15, "Aggressive": 0.15, "Balanced": 0.30,
                         "VolBreakout": 0.20, "Conservative": 0.20},
        }
        return W.get(regime, W["Sideways"])

    def get_risk_adjustments(self, regime):
        R = {
            "Bull":     {"max_single_stock_pct": 0.25, "max_sector_pct": 0.40,
                         "max_gross_exposure_pct": 2.0,  "slippage_multiplier": 0.8},
            "Bear":     {"max_single_stock_pct": 0.15, "max_sector_pct": 0.30,
                         "max_gross_exposure_pct": 1.2,  "slippage_multiplier": 1.5},
            "Sideways": {"max_single_stock_pct": 0.20, "max_sector_pct": 0.35,
                         "max_gross_exposure_pct": 1.5,  "slippage_multiplier": 1.0},
        }
        return R.get(regime, R["Sideways"])

    def get_indicator_adjustments(self, regime):
        return {}


@dataclass
class MockRiskLimits:
    max_single_stock_pct: float = 0.25
    max_sector_pct: float = 0.40
    max_gross_exposure_pct: float = 2.0

class MockRiskManager:
    def __init__(self):
        self.limits = MockRiskLimits()

class MockCostModel:
    def __init__(self):
        self.base_slippage_pct = 0.0005


def _players_data():
    cfgs = {
        "PLAYER_1": {"label": "Aggressive",   "weights": {"RSI_7": 0.9, "ADX_14": 0.5, "SUPERTREND_7_3": 0.6, "BBANDS_20_2": 0.55, "STOCH_5_3": 0.85}, "entry_threshold": 0.25, "exit_threshold": -0.10, "min_hold_bars": 4},
        "PLAYER_2": {"label": "Conservative", "weights": {"ADX_14": 0.9, "SUPERTREND_7_3": 0.85, "RSI_14": 0.6, "BBANDS_20_2": 0.55, "EMA_50": 0.8},   "entry_threshold": 0.30, "exit_threshold": -0.15, "min_hold_bars": 5},
        "PLAYER_3": {"label": "Balanced",     "weights": {"RSI_14": 0.85, "BBANDS_20_2": 0.8, "DEMA_20": 0.5, "ADX_14": 0.45, "ATR_14": 0.4},           "entry_threshold": 0.25, "exit_threshold": -0.10, "min_hold_bars": 4},
        "PLAYER_4": {"label": "VolBreakout",  "weights": {"NATR_14": 0.95, "ADX_14": 0.85, "BBANDS_20_2": 0.75, "ATR_14": 0.7, "KC_20_2": 0.9},        "entry_threshold": 0.22, "exit_threshold": -0.08, "min_hold_bars": 3},
        "PLAYER_5": {"label": "Momentum",     "weights": {"RSI_7": 0.95, "MACD_12_26_9": 0.85, "CMO_14": 0.8, "STOCH_5_3": 0.75, "TSI_13_25": 0.9},    "entry_threshold": 0.23, "exit_threshold": -0.10, "min_hold_bars": 4},
    }
    return [
        {"player_id": pid, "player_label": c["label"],
         "trades": [{"pnl": 100, "symbol": "RELIANCE.NS", "direction": "LONG",
                      "exit_reason": "signal_exit", "bars_held": 5}],
         "weights": c["weights"], "config": c}
        for pid, c in cfgs.items()
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Test 1: Imports
# ═════════════════════════════════════════════════════════════════════════════

def test_1_imports():
    print("\nTest 1: Import RegimeDetector, RegimeAwareCoach, all 5 players")
    print("-" * 60)

    try:
        from regime_detector import RegimeDetector
        check("Import RegimeDetector", True)
    except Exception as e:
        check("Import RegimeDetector", False, str(e))

    try:
        from regime_aware_coach import RegimeAwareCoach
        check("Import RegimeAwareCoach", True)
    except Exception as e:
        check("Import RegimeAwareCoach", False, str(e))

    try:
        from regime_aware_player import (
            RegimeAwarePlayer, build_registry, REGIME_PLAYER_REGISTRY, PLAYER_LABELS,
        )
        check("Import RegimeAwarePlayer + registry", True)
    except Exception as e:
        check("Import RegimeAwarePlayer + registry", False, str(e))

    from regime_aware_player import PLAYER_LABELS
    expected = {"PLAYER_1": "Aggressive", "PLAYER_2": "Conservative",
                "PLAYER_3": "Balanced", "PLAYER_4": "VolBreakout",
                "PLAYER_5": "Momentum"}
    check("All 5 player labels present",
          PLAYER_LABELS == expected,
          f"got {PLAYER_LABELS}")

    from regime_aware_player import build_registry
    reg = build_registry()
    check("Registry has 5 entries", len(reg) == 5, f"got {len(reg)}")
    for pid, label in expected.items():
        check(f"  {pid} → {label}",
              pid in reg and reg[pid].player_label == label)


# ═════════════════════════════════════════════════════════════════════════════
# Test 2: Indicator adjustments defined for all 3 regimes
# ═════════════════════════════════════════════════════════════════════════════

def test_2_indicator_adjustments():
    print("\nTest 2: Each player has indicator adjustments for Bull/Bear/Sideways")
    print("-" * 60)

    from regime_aware_player import build_registry
    from regime_detector import _INDICATOR_ADJUSTMENTS

    reg = build_registry()

    for regime in ("Bull", "Bear", "Sideways"):
        adj = _INDICATOR_ADJUSTMENTS.get(regime)
        check(f"_INDICATOR_ADJUSTMENTS['{regime}'] exists and non-empty",
              adj is not None and len(adj) > 0,
              f"len={len(adj) if adj else 0}")

    # Verify every player can call _get_indicator_adjustments for each regime
    # without error, and gets a non-empty dict back
    sample_cfg = {
        "label": "Test", "weights": {"RSI_7": 0.9, "SUPERTREND_7_3": 0.6,
                                      "ADX_14": 0.5, "BBANDS_20_2": 0.55},
        "entry_threshold": 0.25, "exit_threshold": -0.10, "min_hold_bars": 4,
    }
    for pid, rap in reg.items():
        for regime in ("Bull", "Bear", "Sideways"):
            adj = rap._get_indicator_adjustments(regime)
            check(f"  {rap.player_label}/{regime}: got {len(adj)} multipliers",
                  isinstance(adj, dict) and len(adj) > 0)

    # Verify adjustments actually change weights when applied
    for regime in ("Bull", "Bear", "Sideways"):
        rap = reg["PLAYER_1"]  # Aggressive
        adj_cfg = rap.adjust_config(sample_cfg, regime)
        orig_w = sample_cfg["weights"]
        adj_w = adj_cfg["weights"]
        n_changed = sum(1 for k in orig_w if abs(adj_w.get(k, 0) - orig_w[k]) > 0.001)
        check(f"  Aggressive/{regime}: {n_changed}/{len(orig_w)} weights actually changed",
              n_changed > 0)


# ═════════════════════════════════════════════════════════════════════════════
# Test 3: Specific position sizing multipliers
# ═════════════════════════════════════════════════════════════════════════════

def test_3_sizing_multipliers():
    print("\nTest 3: Position sizing multipliers match specification")
    print("-" * 60)

    from regime_aware_player import build_registry
    reg = build_registry()

    cases = [
        ("PLAYER_5", "Momentum",     "Bull", 1.5),
        ("PLAYER_5", "Momentum",     "Bear", 0.2),
        ("PLAYER_2", "Conservative", "Bear", 0.5),
        ("PLAYER_1", "Aggressive",   "Bull", 1.5),
        ("PLAYER_1", "Aggressive",   "Bear", 0.3),
    ]

    for pid, label, regime, expected in cases:
        rap = reg[pid]
        actual = rap.get_sizing_multiplier(regime)
        ok = abs(actual - expected) < 0.001
        check(f"{label} {regime} = {expected}x",
              ok, f"got {actual}")

    # Also check via adjust_config path
    dummy_cfg = {"label": "x", "weights": {"RSI_7": 0.5}, "entry_threshold": 0.25,
                 "exit_threshold": -0.10, "min_hold_bars": 4}
    for pid, label, regime, expected in cases:
        rap = reg[pid]
        adj = rap.adjust_config(dummy_cfg, regime)
        actual = adj.get("position_sizing_multiplier", -1)
        ok = abs(actual - expected) < 0.001
        check(f"  adjust_config {label}/{regime} = {expected}x",
              ok, f"got {actual}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 4: Bear detector → tighter risk limits
# ═════════════════════════════════════════════════════════════════════════════

def test_4_bear_risk_limits():
    print("\nTest 4: Bear regime → tighter PortfolioRiskManager limits")
    print("-" * 60)

    from regime_aware_coach import RegimeAwareCoach

    detector = MockDetector(regime="Bear", conf=0.85)
    rm = MockRiskManager()

    rac = RegimeAwareCoach(
        ai_coach=None,
        regime_detector=detector,
        nifty_ohlcv=pd.DataFrame(),
        risk_manager=rm,
        auto_apply_risk_overrides=True,
    )
    # Isolate from persisted stats
    rac._REGIME_STATS_PATH = Path(tempfile.mktemp(suffix=".json"))
    for rap in rac._players.values():
        rap._regime_stats = {}

    result = rac.run_coaching_cycle(_players_data())

    check("Risk overrides present in output",
          bool(result.risk_overrides))
    check("max_single_stock_pct = 0.15 (Bear, tighter than 0.25)",
          abs(rm.limits.max_single_stock_pct - 0.15) < 0.001,
          f"got {rm.limits.max_single_stock_pct}")
    check("max_sector_pct = 0.30 (Bear, tighter than 0.40)",
          abs(rm.limits.max_sector_pct - 0.30) < 0.001,
          f"got {rm.limits.max_sector_pct}")
    check("max_gross_exposure_pct = 1.20 (Bear, tighter than 2.0)",
          abs(rm.limits.max_gross_exposure_pct - 1.20) < 0.001,
          f"got {rm.limits.max_gross_exposure_pct}")
    check("All 3 Bear limits are strictly tighter than Bull defaults",
          rm.limits.max_single_stock_pct < 0.25
          and rm.limits.max_sector_pct < 0.40
          and rm.limits.max_gross_exposure_pct < 2.0)


# ═════════════════════════════════════════════════════════════════════════════
# Test 5: Bull detector → slippage_multiplier = 0.8
# ═════════════════════════════════════════════════════════════════════════════

def test_5_bull_slippage():
    print("\nTest 5: Bull regime → slippage_multiplier=0.8 on TransactionCostModel")
    print("-" * 60)

    from regime_aware_coach import RegimeAwareCoach

    detector = MockDetector(regime="Bull", conf=0.80)
    cm = MockCostModel()
    original_slippage = cm.base_slippage_pct  # 0.0005

    rac = RegimeAwareCoach(
        ai_coach=None,
        regime_detector=detector,
        nifty_ohlcv=pd.DataFrame(),
        cost_model=cm,
        auto_apply_slippage=True,
    )
    rac._REGIME_STATS_PATH = Path(tempfile.mktemp(suffix=".json"))
    for rap in rac._players.values():
        rap._regime_stats = {}

    result = rac.run_coaching_cycle(_players_data())

    check("CoachingOutput.slippage_multiplier = 0.8",
          abs(result.slippage_multiplier - 0.8) < 0.001,
          f"got {result.slippage_multiplier}")

    expected = original_slippage * 0.8
    check(f"CostModel.base_slippage_pct = {expected} (0.0005 * 0.8)",
          abs(cm.base_slippage_pct - expected) < 1e-8,
          f"got {cm.base_slippage_pct}")

    # Verify original is preserved internally
    check("Original base_slippage_pct preserved",
          hasattr(cm, "_original_base_slippage_pct")
          and abs(cm._original_base_slippage_pct - original_slippage) < 1e-8)

    # Switch to Bear, re-run, verify slippage resets from original (no drift)
    detector._regime = "Bear"
    detector._conf = 0.85
    result2 = rac.run_coaching_cycle(_players_data())
    expected_bear = original_slippage * 1.5
    check(f"After Bear switch: slippage = {expected_bear} (no compounding)",
          abs(cm.base_slippage_pct - expected_bear) < 1e-8,
          f"got {cm.base_slippage_pct}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 6: Regime weights sum to 1.0
# ═════════════════════════════════════════════════════════════════════════════

def test_6_regime_weights_sum():
    print("\nTest 6: Regime weights sum to 1.0 for each regime")
    print("-" * 60)

    from regime_aware_coach import RegimeAwareCoach

    for regime in ("Bull", "Bear", "Sideways"):
        detector = MockDetector(regime=regime)
        rac = RegimeAwareCoach(
            ai_coach=None,
            regime_detector=detector,
            nifty_ohlcv=pd.DataFrame(),
        )
        rac._REGIME_STATS_PATH = Path(tempfile.mktemp(suffix=".json"))
        for rap in rac._players.values():
            rap._regime_stats = {}

        weights = rac.get_player_allocation_weights(regime)
        total = sum(weights.values())
        check(f"{regime}: sum={total:.4f} (expect 1.0)",
              abs(total - 1.0) < 0.001)
        check(f"  {regime}: 5 players present",
              len(weights) == 5, f"got {len(weights)}")

        # Verify no player has weight <= 0
        all_positive = all(v > 0 for v in weights.values())
        check(f"  {regime}: all weights > 0", all_positive)

    # Also verify via coaching output
    detector = MockDetector(regime="Bear", conf=0.90)
    rac = RegimeAwareCoach(
        ai_coach=None, regime_detector=detector,
        nifty_ohlcv=pd.DataFrame(),
    )
    rac._REGIME_STATS_PATH = Path(tempfile.mktemp(suffix=".json"))
    for rap in rac._players.values():
        rap._regime_stats = {}
    result = rac.run_coaching_cycle(_players_data())
    output_sum = sum(result.player_weights.values())
    check(f"CoachingOutput.player_weights sum = {output_sum:.4f}",
          abs(output_sum - 1.0) < 0.001)


# ═════════════════════════════════════════════════════════════════════════════
# Test 7: Transition logging: Bull → Bear → Sideways
# ═════════════════════════════════════════════════════════════════════════════

def test_7_transition_logging():
    print("\nTest 7: Regime transitions Bull→Bear→Sideways all logged with timestamps")
    print("-" * 60)

    from regime_aware_coach import RegimeAwareCoach

    detector = MockDetector(regime="Bull", conf=0.80, duration=10)
    rac = RegimeAwareCoach(
        ai_coach=None, regime_detector=detector,
        nifty_ohlcv=pd.DataFrame(),
    )
    rac._REGIME_STATS_PATH = Path(tempfile.mktemp(suffix=".json"))
    for rap in rac._players.values():
        rap._regime_stats = {}

    pd_list = _players_data()

    # Cycle 1: Bull (initial, no transition)
    r1 = rac.run_coaching_cycle(pd_list)
    check("Cycle 1 (Bull): no transition", r1.regime_changed is False)
    check("Cycle 1: regime=Bull", r1.regime == "Bull")

    # Cycle 2: Bull → Bear
    detector._regime = "Bear"
    detector._conf = 0.85
    r2 = rac.run_coaching_cycle(pd_list)
    check("Cycle 2 (Bull→Bear): transition detected", r2.regime_changed is True)
    check("Cycle 2: previous_regime=Bull", r2.previous_regime == "Bull")
    check("Cycle 2: regime=Bear", r2.regime == "Bear")

    # Cycle 3: Bear → Sideways
    detector._regime = "Sideways"
    detector._conf = 0.70
    r3 = rac.run_coaching_cycle(pd_list)
    check("Cycle 3 (Bear→Sideways): transition detected", r3.regime_changed is True)
    check("Cycle 3: previous_regime=Bear", r3.previous_regime == "Bear")
    check("Cycle 3: regime=Sideways", r3.regime == "Sideways")

    # Verify tracker has exactly 2 transitions
    tracker = rac._transition_tracker
    transitions = tracker.last_n_transitions(10)
    check("Tracker recorded exactly 2 transitions",
          len(transitions) == 2, f"got {len(transitions)}")

    # Verify both transitions have timestamps
    for i, t in enumerate(transitions):
        has_ts = "timestamp" in t and len(t["timestamp"]) >= 10
        check(f"  Transition {i+1} has timestamp",
              has_ts, t.get("timestamp", "MISSING")[:19])

    check("Transition 1: Bull→Bear",
          transitions[0]["previous"] == "Bull" and transitions[0]["regime"] == "Bear")
    check("Transition 2: Bear→Sideways",
          transitions[1]["previous"] == "Bear" and transitions[1]["regime"] == "Sideways")

    # Summary string has both arrows
    summary = tracker.transition_summary(5)
    check("Summary contains 'Bull→Bear'", "Bull→Bear" in summary, summary[:80])
    check("Summary contains 'Bear→Sideways'", "Bear→Sideways" in summary)


# ═════════════════════════════════════════════════════════════════════════════
# Test 8: Player performance tracking — separate metrics per regime
# ═════════════════════════════════════════════════════════════════════════════

def test_8_performance_per_regime():
    print("\nTest 8: Performance tracking stores separate metrics per regime")
    print("-" * 60)

    from regime_aware_player import RegimeAwarePlayer

    rap = RegimeAwarePlayer("PLAYER_1", "Aggressive")

    # Record distinct data in each regime
    rap.record_result("Bull", pnl=2000, sharpe=2.5, win_rate=0.55)
    rap.record_result("Bull", pnl=1500, sharpe=2.0, win_rate=0.52)
    rap.record_result("Bear", pnl=-500, sharpe=-0.8, win_rate=0.30)
    rap.record_result("Bear", pnl=-300, sharpe=-0.4, win_rate=0.35)
    rap.record_result("Bear", pnl=-100, sharpe=-0.1, win_rate=0.38)
    rap.record_result("Sideways", pnl=300, sharpe=0.4, win_rate=0.42)

    bull = rap.get_regime_stats("Bull")
    bear = rap.get_regime_stats("Bear")
    sw   = rap.get_regime_stats("Sideways")

    check("Bull: 2 samples", bull["n_samples"] == 2)
    check("Bear: 3 samples", bear["n_samples"] == 3)
    check("Sideways: 1 sample", sw["n_samples"] == 1)

    # Verify averages are computed independently
    check("Bull avg_pnl = 1750",
          abs(bull["avg_pnl"] - 1750) < 0.1, f"got {bull['avg_pnl']}")
    check("Bear avg_pnl = -300",
          abs(bear["avg_pnl"] - (-300)) < 0.1, f"got {bear['avg_pnl']}")
    check("Sideways avg_pnl = 300",
          abs(sw["avg_pnl"] - 300) < 0.1, f"got {sw['avg_pnl']}")

    check("Bull avg_sharpe = 2.25",
          abs(bull["avg_sharpe"] - 2.25) < 0.01)
    check("Bear avg_win_rate ≈ 0.343",
          abs(bear["avg_win_rate"] - (0.30 + 0.35 + 0.38) / 3) < 0.01,
          f"got {bear['avg_win_rate']}")

    # Verify regimes are truly isolated — Bull stats unchanged by Bear recording
    rap.record_result("Bear", pnl=-800, sharpe=-1.5, win_rate=0.25)
    bull_after = rap.get_regime_stats("Bull")
    check("Bull unchanged after adding Bear data",
          bull_after["n_samples"] == 2 and abs(bull_after["avg_pnl"] - 1750) < 0.1)

    # get_all_regime_stats has all 3
    all_stats = rap.get_all_regime_stats()
    check("get_all_regime_stats has Bull, Bear, Sideways",
          set(all_stats.keys()) == {"Bull", "Bear", "Sideways"})

    # Persistence round-trip through coach
    from regime_aware_coach import RegimeAwareCoach
    rac = RegimeAwareCoach(ai_coach=None, regime_detector=MockDetector(),
                           nifty_ohlcv=pd.DataFrame())
    tmp_path = Path(tempfile.mktemp(suffix=".json"))
    rac._REGIME_STATS_PATH = tmp_path
    for r in rac._players.values():
        r._regime_stats = {}

    rac.record_run_results("Bull", {
        "PLAYER_1": {"pnl": 1000, "sharpe": 1.5, "win_rate": 0.50},
        "PLAYER_2": {"pnl": 500,  "sharpe": 0.8, "win_rate": 0.45},
    })
    rac.record_run_results("Bull", {
        "PLAYER_1": {"pnl": 2000, "sharpe": 2.0, "win_rate": 0.55},
    })

    # Reload in a new coach instance
    rac2 = RegimeAwareCoach(ai_coach=None, regime_detector=MockDetector(),
                            nifty_ohlcv=pd.DataFrame())
    rac2._REGIME_STATS_PATH = tmp_path
    for r in rac2._players.values():
        r._regime_stats = {}
    rac2._load_regime_stats()

    p1 = rac2._players["PLAYER_1"]
    reloaded = p1.get_regime_stats("Bull")
    check("Persistence: PLAYER_1 Bull n_samples=2 after reload",
          reloaded["n_samples"] == 2, f"got {reloaded['n_samples']}")
    check("Persistence: avg_pnl = 1500 after reload",
          abs(reloaded["avg_pnl"] - 1500) < 0.1, f"got {reloaded['avg_pnl']}")

    # Clean up
    try:
        tmp_path.unlink()
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# Test 9: Gemini prompt template includes regime context
# ═════════════════════════════════════════════════════════════════════════════

def test_9_prompt_context():
    print("\nTest 9: Gemini prompt includes regime, duration, transition history")
    print("-" * 60)

    from regime_aware_coach import RegimeAwareCoach

    detector = MockDetector(regime="Bull", conf=0.78, duration=12)
    rac = RegimeAwareCoach(
        ai_coach=None, regime_detector=detector,
        nifty_ohlcv=pd.DataFrame(),
    )
    rac._REGIME_STATS_PATH = Path(tempfile.mktemp(suffix=".json"))
    for rap in rac._players.values():
        rap._regime_stats = {}

    # Record history so it shows up in prompt
    rac.record_run_results("Bull", {
        "PLAYER_1": {"pnl": 1200, "sharpe": 1.8, "win_rate": 0.50},
    })

    # Simulate a transition so there's history
    rac._transition_tracker.update("Bull", 0.78)
    rac._transition_tracker.update("Bear", 0.85)
    rac._transition_tracker.update("Bull", 0.78)

    ctx = rac.get_regime_context_for_prompt(
        "Bull", {"Bull": 0.78, "Bear": 0.07, "Sideways": 0.15},
        duration=12, player_id="PLAYER_1",
    )

    check("Contains 'MARKET REGIME:'", "MARKET REGIME:" in ctx)
    check("Contains current regime 'Bull'", "Bull" in ctx)
    check("Contains 'confidence='", "confidence=" in ctx)
    check("Contains 'duration=12'", "duration=12" in ctx)
    check("Contains 'RECENT TRANSITIONS:'", "RECENT TRANSITIONS:" in ctx)
    check("Contains transition arrows (→)", "→" in ctx)
    check("Contains 'YOUR REGIME HISTORY:'", "YOUR REGIME HISTORY:" in ctx)
    check("Contains player performance data (avgPnL)", "avgPnL" in ctx)

    # Verify prompt does NOT contain stale "No regime history yet." string
    # (Bug 4 fix)
    check("No 'No regime history yet.' pollution",
          "No regime history yet." not in ctx)

    # Verify enriched players_data includes regime_context
    pd_list = _players_data()
    enriched = rac._enrich_players_data(pd_list, "Bull",
                                         {"Bull": 0.78, "Bear": 0.07, "Sideways": 0.15}, 12)
    for ep in enriched:
        check(f"  {ep['player_id']} enriched with regime_context",
              "regime_context" in ep and len(ep["regime_context"]) > 0)
        check(f"  {ep['player_id']} enriched with regime_allocation_weight",
              "regime_allocation_weight" in ep and ep["regime_allocation_weight"] > 0)
        check(f"  {ep['player_id']} enriched with position_sizing_multiplier",
              "position_sizing_multiplier" in ep and ep["position_sizing_multiplier"] > 0)

    # Verify a player with NO history does NOT add the YOUR REGIME HISTORY line
    ctx_no_hist = rac.get_regime_context_for_prompt(
        "Bear", {"Bull": 0.10, "Bear": 0.80, "Sideways": 0.10},
        duration=5, player_id="PLAYER_3",  # no results recorded for PLAYER_3
    )
    check("Player with no history: no 'YOUR REGIME HISTORY' line",
          "YOUR REGIME HISTORY" not in ctx_no_hist,
          f"ctx contains: {ctx_no_hist[-80:]}")


# ═════════════════════════════════════════════════════════════════════════════
# Test 10: Low-confidence predictions (~0.33) don't crash
# ═════════════════════════════════════════════════════════════════════════════

def test_10_low_confidence():
    print("\nTest 10: Coach handles low-confidence predictions (~0.33)")
    print("-" * 60)

    from regime_aware_coach import RegimeAwareCoach

    class LowConfDetector:
        """Returns near-uniform probabilities — maximum uncertainty."""
        def predict(self, ohlcv, lookback=5):
            return "Sideways", {"Bull": 0.34, "Bear": 0.33, "Sideways": 0.33}, 1

        def get_regime_weights(self, regime):
            return {"Momentum": 0.15, "Aggressive": 0.15, "Balanced": 0.30,
                    "VolBreakout": 0.20, "Conservative": 0.20}

        def get_risk_adjustments(self, regime):
            return {"max_single_stock_pct": 0.20, "max_sector_pct": 0.35,
                    "max_gross_exposure_pct": 1.5, "slippage_multiplier": 1.0}

    rm = MockRiskManager()
    cm = MockCostModel()
    rac = RegimeAwareCoach(
        ai_coach=None,
        regime_detector=LowConfDetector(),
        nifty_ohlcv=pd.DataFrame(),
        risk_manager=rm,
        cost_model=cm,
    )
    rac._REGIME_STATS_PATH = Path(tempfile.mktemp(suffix=".json"))
    for rap in rac._players.values():
        rap._regime_stats = {}

    # Run coaching cycle — should not crash
    try:
        result = rac.run_coaching_cycle(_players_data())
        no_crash = True
    except Exception as e:
        no_crash = False
        result = None
        check("Coaching cycle did not crash", False, str(e))

    if no_crash:
        check("Coaching cycle completed without error", True)
        check("Regime = Sideways (low-confidence fallback)",
              result.regime == "Sideways")
        check("Confidence ≈ 0.33",
              abs(result.regime_confidence - 0.33) < 0.02,
              f"got {result.regime_confidence}")
        check("Duration = 1", result.regime_duration == 1)
        check("5 adjusted configs produced",
              len(result.adjusted_configs) == 5)

        # All configs should have valid values
        for pid, cfg in result.adjusted_configs.items():
            ok = (isinstance(cfg.get("weights"), dict)
                  and 0.10 <= cfg.get("entry_threshold", 0) <= 0.50
                  and -0.30 <= cfg.get("exit_threshold", 0) <= -0.03
                  and cfg.get("position_sizing_multiplier", 0) > 0)
            check(f"  {pid} config valid", ok)

    # Run multiple low-confidence cycles back-to-back
    try:
        for _ in range(5):
            rac.run_coaching_cycle(_players_data())
        check("5 consecutive low-confidence cycles: no crash", True)
    except Exception as e:
        check("5 consecutive low-confidence cycles: no crash", False, str(e))

    # Record results under low confidence
    try:
        rac.record_run_results("Sideways", {
            "PLAYER_1": {"pnl": 100, "sharpe": 0.1, "win_rate": 0.40},
        })
        check("record_run_results under low confidence: no crash", True)
    except Exception as e:
        check("record_run_results under low confidence: no crash", False, str(e))

    # Prompt context with low confidence
    try:
        ctx = rac.get_regime_context_for_prompt(
            "Sideways", {"Bull": 0.34, "Bear": 0.33, "Sideways": 0.33},
            duration=1, player_id="PLAYER_1",
        )
        check("Prompt context generation: no crash", True)
        check("Prompt shows low confidence (33%)", "33%" in ctx, ctx[:60])
    except Exception as e:
        check("Prompt context generation: no crash", False, str(e))


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  Regime-Aware Integration — Comprehensive Test Suite (v2)")
    print("=" * 65)

    test_1_imports()
    test_2_indicator_adjustments()
    test_3_sizing_multipliers()
    test_4_bear_risk_limits()
    test_5_bull_slippage()
    test_6_regime_weights_sum()
    test_7_transition_logging()
    test_8_performance_per_regime()
    test_9_prompt_context()
    test_10_low_confidence()

    print("\n" + "=" * 65)
    total = passed + failed
    print(f"  Results: {passed}/{total} passed, {failed}/{total} failed")
    if failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {failed} TEST(S) FAILED")
    print("=" * 65)

    sys.exit(0 if failed == 0 else 1)
