#!/usr/bin/env python3
"""
Comprehensive test suite for the Dynamic Signal Combination System.

Covers:
  1. ConfidenceWeightedVoter — basic combination
  2. ConfidenceWeightedVoter — confidence scaling
  3. AgreementAnalyzer — 5/5 unanimous (bullish)
  4. AgreementAnalyzer — 4/1 strong signal
  5. AgreementAnalyzer — 3/2 split → reduced size
  6. AgreementAnalyzer — no clear majority → skip
  7. BayesianPlayerWeights — initialization
  8. BayesianPlayerWeights — correct signal → upweight
  9. BayesianPlayerWeights — poor performer → downweighted
  10. BayesianPlayerWeights — regime-conditional separation
  11. BayesianPlayerWeights — weight constraints [0.05, 0.40]
  12. RegimeAwareCombiner — full 5-player bullish
  13. RegimeAwareCombiner — 3-2 split
  14. RegimeAwareCombiner — regime change shifts weights
  15. CombinerDashboard — summary output
  16. CombinationLogger — accuracy tracking
  17. Integration — Bayesian downweight after bad trades

Run: python3 tests/test_signal_combiner.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signal_combiner import (
    AgreementAnalyzer,
    AgreementResult,
    BayesianPlayerWeights,
    CombinationLogger,
    CombinedSignal,
    CombinerDashboard,
    ConfidenceWeightedVoter,
    PlayerSignal,
    RegimeAwareCombiner,
    TradeOutcome,
    PLAYER_IDS,
    PLAYER_LABELS,
)

# ---------------------------------------------------------------------------
_pass = 0
_fail = 0


def check(condition: bool, label: str) -> None:
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  \u2705 {label}")
    else:
        _fail += 1
        print(f"  \u274c {label}")


def tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="test_combiner_"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bullish_signals(n: int = 5) -> list:
    """All 5 players bullish with varying confidence."""
    confs = [0.90, 0.80, 0.85, 0.70, 0.95]
    dirs = [0.80, 0.60, 0.70, 0.55, 0.90]
    return [
        PlayerSignal(
            player_id=PLAYER_IDS[i],
            direction=dirs[i],
            confidence=confs[i],
            reasoning=f"Bullish based on indicator set {i+1}",
        )
        for i in range(n)
    ]


def make_split_signals_3_2() -> list:
    """3 bullish, 2 bearish."""
    return [
        PlayerSignal("PLAYER_1", 0.70, 0.85, "Momentum strong"),
        PlayerSignal("PLAYER_2", 0.60, 0.75, "Trend continuation"),
        PlayerSignal("PLAYER_3", 0.50, 0.80, "Balanced bullish"),
        PlayerSignal("PLAYER_4", -0.60, 0.70, "Volatility breakdown"),
        PlayerSignal("PLAYER_5", -0.55, 0.90, "Momentum fading"),
    ]


def make_no_majority_signals() -> list:
    """2 bullish, 2 bearish, 1 neutral."""
    return [
        PlayerSignal("PLAYER_1", 0.70, 0.80, "Bullish breakout"),
        PlayerSignal("PLAYER_2", 0.50, 0.70, "Mild bullish"),
        PlayerSignal("PLAYER_3", -0.60, 0.85, "Bearish reversal"),
        PlayerSignal("PLAYER_4", -0.50, 0.75, "Vol contraction"),
        PlayerSignal("PLAYER_5", 0.00, 0.30, "No clear direction"),
    ]


# ---------------------------------------------------------------------------
# Test 1: ConfidenceWeightedVoter — basic combination
# ---------------------------------------------------------------------------

def test_1_voter_basic():
    print("\n--- Test 1: ConfidenceWeightedVoter \u2014 basic ---")
    voter = ConfidenceWeightedVoter()
    signals = [
        PlayerSignal("PLAYER_1", 0.80, 0.90, "bullish"),
        PlayerSignal("PLAYER_2", 0.60, 0.80, "bullish"),
    ]
    weights = {"PLAYER_1": 0.50, "PLAYER_2": 0.50}

    final, influence = voter.combine(signals, weights)

    check(final > 0, f"Final signal is positive: {final:.3f}")
    check(abs(final) <= 1.0, f"Signal in [-1, 1]: {final:.3f}")

    # With equal weights, higher confidence player should have more influence
    check(
        abs(influence["PLAYER_1"]) > abs(influence["PLAYER_2"]),
        f"PLAYER_1 (conf=0.90) more influential than PLAYER_2 (conf=0.80)"
    )

    # Manual calc: num = 0.80*0.90*0.50 + 0.60*0.80*0.50 = 0.36 + 0.24 = 0.60
    #              den = 0.90*0.50 + 0.80*0.50 = 0.45 + 0.40 = 0.85
    #              final = 0.60 / 0.85 ≈ 0.7059
    check(abs(final - 0.7059) < 0.01, f"Manual calc matches: {final:.4f} ≈ 0.7059")


# ---------------------------------------------------------------------------
# Test 2: ConfidenceWeightedVoter — confidence scaling
# ---------------------------------------------------------------------------

def test_2_voter_confidence():
    print("\n--- Test 2: ConfidenceWeightedVoter \u2014 confidence scaling ---")
    voter = ConfidenceWeightedVoter()

    # Same direction but one player has much higher confidence
    signals = [
        PlayerSignal("PLAYER_1", 0.90, 1.00, "very confident bullish"),
        PlayerSignal("PLAYER_2", 0.10, 0.10, "barely bullish, low confidence"),
    ]
    weights = {"PLAYER_1": 0.50, "PLAYER_2": 0.50}

    final, influence = voter.combine(signals, weights)

    check(
        final > 0.80,
        f"High-confidence player dominates: final={final:.3f} > 0.80"
    )

    # Now test opposing directions
    signals2 = [
        PlayerSignal("PLAYER_1", 0.80, 0.95, "bullish, high conf"),
        PlayerSignal("PLAYER_2", -0.80, 0.20, "bearish, low conf"),
    ]
    final2, _ = voter.combine(signals2, weights)
    check(
        final2 > 0,
        f"High-confidence bullish outweighs low-confidence bearish: {final2:.3f}"
    )


# ---------------------------------------------------------------------------
# Test 3: AgreementAnalyzer — 5/5 unanimous
# ---------------------------------------------------------------------------

def test_3_agreement_unanimous():
    print("\n--- Test 3: AgreementAnalyzer \u2014 5/5 unanimous ---")
    analyzer = AgreementAnalyzer()
    signals = make_bullish_signals()

    result = analyzer.analyze(signals)

    check(result.pattern == "unanimous", f"Pattern: {result.pattern}")
    check(result.agreement_score == 1.0, f"Agreement: {result.agreement_score}")
    check(result.majority_direction == 1.0, f"Majority dir: {result.majority_direction}")
    check(result.majority_count == 5, f"Majority count: {result.majority_count}")
    check(result.dissent_count == 0, f"Dissent count: {result.dissent_count}")
    check(result.should_trade is True, "Should trade: True")
    check(result.crowding_flag is True, "Crowding flag: True (5/5)")
    check(result.position_size_multiplier == 1.0, f"Size mult: {result.position_size_multiplier}")


# ---------------------------------------------------------------------------
# Test 4: AgreementAnalyzer — 4/1 strong
# ---------------------------------------------------------------------------

def test_4_agreement_strong():
    print("\n--- Test 4: AgreementAnalyzer \u2014 4/1 strong ---")
    analyzer = AgreementAnalyzer()
    signals = [
        PlayerSignal("PLAYER_1", 0.70, 0.85, "bullish"),
        PlayerSignal("PLAYER_2", 0.60, 0.80, "bullish"),
        PlayerSignal("PLAYER_3", 0.50, 0.75, "bullish"),
        PlayerSignal("PLAYER_4", 0.55, 0.70, "bullish"),
        PlayerSignal("PLAYER_5", -0.40, 0.60, "bearish dissenter"),
    ]

    result = analyzer.analyze(signals)

    check(result.pattern == "strong", f"Pattern: {result.pattern}")
    check(result.majority_count == 4, f"Majority: {result.majority_count}")
    check(result.dissent_count == 1, f"Dissenters: {result.dissent_count}")
    check("PLAYER_5" in result.dissenters, "PLAYER_5 is the dissenter")
    check(result.should_trade is True, "Should trade")
    check(result.crowding_flag is False, "No crowding with 4/5")
    check(result.position_size_multiplier == 1.0, "Full position size for 4/1")
    check("Momentum" in result.dissent_analysis, "Dissent analysis names player")


# ---------------------------------------------------------------------------
# Test 5: AgreementAnalyzer — 3/2 split → reduced size
# ---------------------------------------------------------------------------

def test_5_agreement_3_2_split():
    print("\n--- Test 5: AgreementAnalyzer \u2014 3/2 split ---")
    analyzer = AgreementAnalyzer()
    signals = make_split_signals_3_2()

    result = analyzer.analyze(signals)

    check(result.pattern == "moderate", f"Pattern: {result.pattern}")
    check(result.majority_count == 3, f"Majority: {result.majority_count}")
    check(result.dissent_count == 2, f"Dissenters: {result.dissent_count}")
    check(result.should_trade is True, "Should still trade")
    check(
        result.position_size_multiplier == 0.80,
        f"Position size reduced by 20%: {result.position_size_multiplier}"
    )
    check(
        len(result.dissenters) == 2,
        f"2 dissenters: {result.dissenters}"
    )


# ---------------------------------------------------------------------------
# Test 6: AgreementAnalyzer — no clear majority
# ---------------------------------------------------------------------------

def test_6_agreement_no_majority():
    print("\n--- Test 6: AgreementAnalyzer \u2014 no majority ---")
    analyzer = AgreementAnalyzer()
    signals = make_no_majority_signals()

    result = analyzer.analyze(signals)

    check(result.pattern == "no_majority", f"Pattern: {result.pattern}")
    check(result.should_trade is False, "Should NOT trade")
    check(result.position_size_multiplier == 0.0, "Size multiplier = 0")


# ---------------------------------------------------------------------------
# Test 7: BayesianPlayerWeights — initialization
# ---------------------------------------------------------------------------

def test_7_bayesian_init():
    print("\n--- Test 7: BayesianPlayerWeights \u2014 initialization ---")
    d = tmp()
    try:
        bw = BayesianPlayerWeights(state_dir=d)

        for regime in ["Bull", "Bear", "Sideways"]:
            weights = bw.get_weights(regime)
            check(len(weights) == 5, f"{regime}: 5 players")
            total = sum(weights.values())
            check(
                abs(total - 1.0) < 0.001,
                f"{regime}: weights sum to {total:.4f} ≈ 1.0"
            )
            for pid, w in weights.items():
                check(
                    abs(w - 0.20) < 0.001,
                    f"{regime} {pid}: initial weight = {w:.4f} ≈ 0.20"
                )
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 8: BayesianPlayerWeights — correct signal → upweight
# ---------------------------------------------------------------------------

def test_8_bayesian_correct_upweight():
    print("\n--- Test 8: BayesianPlayerWeights \u2014 correct → upweight ---")
    d = tmp()
    try:
        bw = BayesianPlayerWeights(state_dir=d)
        initial_w = bw.get_weights("Bull")["PLAYER_1"]

        # P1 makes a correct bullish call with good PnL
        outcome = TradeOutcome(
            player_id="PLAYER_1",
            signal_direction=0.80,
            actual_pnl=300.0,
            regime="Bull",
        )
        new_weights = bw.update(outcome)

        check(
            new_weights["PLAYER_1"] > initial_w,
            f"PLAYER_1 upweighted: {initial_w:.4f} → {new_weights['PLAYER_1']:.4f}"
        )
        total = sum(new_weights.values())
        check(
            abs(total - 1.0) < 0.001,
            f"Weights still sum to 1.0: {total:.6f}"
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 9: BayesianPlayerWeights — poor performer downweighted
# ---------------------------------------------------------------------------

def test_9_bayesian_poor_performer():
    print("\n--- Test 9: BayesianPlayerWeights \u2014 poor performer downweighted ---")
    d = tmp()
    try:
        bw = BayesianPlayerWeights(state_dir=d)

        # PLAYER_3 makes repeated wrong calls in Bull
        for _ in range(10):
            bw.update(TradeOutcome("PLAYER_3", 0.70, -250.0, "Bull"))

        # Meanwhile PLAYER_1 makes good calls
        for _ in range(10):
            bw.update(TradeOutcome("PLAYER_1", 0.70, 300.0, "Bull"))

        weights = bw.get_weights("Bull")

        check(
            weights["PLAYER_1"] > weights["PLAYER_3"],
            f"PLAYER_1 ({weights['PLAYER_1']:.4f}) > PLAYER_3 ({weights['PLAYER_3']:.4f})"
        )
        check(
            weights["PLAYER_3"] >= BayesianPlayerWeights.MIN_WEIGHT - 1e-6,
            f"PLAYER_3 weight ≥ {BayesianPlayerWeights.MIN_WEIGHT}: {weights['PLAYER_3']:.6f}"
        )
        check(
            weights["PLAYER_1"] <= BayesianPlayerWeights.MAX_WEIGHT + 1e-6,
            f"PLAYER_1 weight ≤ {BayesianPlayerWeights.MAX_WEIGHT}: {weights['PLAYER_1']:.6f}"
        )

        total = sum(weights.values())
        check(abs(total - 1.0) < 0.001, f"Sum = {total:.6f} ≈ 1.0")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 10: BayesianPlayerWeights — regime-conditional
# ---------------------------------------------------------------------------

def test_10_bayesian_regime_conditional():
    print("\n--- Test 10: BayesianPlayerWeights \u2014 regime-conditional ---")
    d = tmp()
    try:
        bw = BayesianPlayerWeights(state_dir=d)

        # PLAYER_5 good in Bull, bad in Bear
        for _ in range(8):
            bw.update(TradeOutcome("PLAYER_5", 0.80, 400.0, "Bull"))
            bw.update(TradeOutcome("PLAYER_5", 0.80, -300.0, "Bear"))

        bull_w = bw.get_weights("Bull")["PLAYER_5"]
        bear_w = bw.get_weights("Bear")["PLAYER_5"]

        check(
            bull_w > bear_w,
            f"PLAYER_5 weight in Bull ({bull_w:.4f}) > Bear ({bear_w:.4f})"
        )

        # Sideways should be unaffected (still ~0.20)
        side_w = bw.get_weights("Sideways")["PLAYER_5"]
        check(
            abs(side_w - 0.20) < 0.01,
            f"Sideways unaffected: {side_w:.4f} ≈ 0.20"
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 11: BayesianPlayerWeights — constraints [0.05, 0.40]
# ---------------------------------------------------------------------------

def test_11_bayesian_constraints():
    print("\n--- Test 11: BayesianPlayerWeights \u2014 weight constraints ---")
    d = tmp()
    try:
        bw = BayesianPlayerWeights(state_dir=d)

        # Extreme: PLAYER_1 is incredibly good, everyone else terrible
        for _ in range(50):
            bw.update(TradeOutcome("PLAYER_1", 0.90, 500.0, "Bull"))
            for pid in PLAYER_IDS[1:]:
                bw.update(TradeOutcome(pid, 0.80, -400.0, "Bull"))

        weights = bw.get_weights("Bull")

        check(
            weights["PLAYER_1"] <= 0.40 + 1e-6,
            f"Max cap: PLAYER_1 = {weights['PLAYER_1']:.6f} ≤ 0.40"
        )

        for pid in PLAYER_IDS[1:]:
            check(
                weights[pid] >= 0.05 - 1e-6,
                f"Min floor: {pid} = {weights[pid]:.6f} ≥ 0.05"
            )

        total = sum(weights.values())
        check(abs(total - 1.0) < 0.01, f"Sum = {total:.4f} ≈ 1.0")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 12: RegimeAwareCombiner — 5 players all bullish
# ---------------------------------------------------------------------------

def test_12_combiner_all_bullish():
    print("\n--- Test 12: RegimeAwareCombiner \u2014 5 bullish ---")
    d = tmp()
    try:
        combiner = RegimeAwareCombiner(state_dir=d)
        signals = make_bullish_signals()

        result = combiner.combine(signals, regime="Bull")

        check(result.should_trade is True, "Should trade")
        check(result.final_signal > 0, f"Bullish signal: {result.final_signal:.3f}")
        check(
            result.final_signal > 0.50,
            f"High conviction: {result.final_signal:.3f} > 0.50"
        )
        check(
            result.position_size_multiplier == 1.0,
            f"Full position size: {result.position_size_multiplier}"
        )
        check(
            result.signal_metadata.get("crowding_flag") is True,
            "Crowding flag set (5/5)"
        )
        check(
            result.signal_metadata["agreement"]["pattern"] == "unanimous",
            "Pattern: unanimous"
        )

        # Check influence map exists and has all players
        infl = result.signal_metadata.get("influence_map", {})
        check(len(infl) == 5, f"Influence map has 5 entries")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 13: RegimeAwareCombiner — 3-2 split
# ---------------------------------------------------------------------------

def test_13_combiner_3_2_split():
    print("\n--- Test 13: RegimeAwareCombiner \u2014 3/2 split ---")
    d = tmp()
    try:
        combiner = RegimeAwareCombiner(state_dir=d)
        signals = make_split_signals_3_2()

        result = combiner.combine(signals, regime="Sideways")

        check(result.should_trade is True, "Should trade (3/2 is moderate)")
        check(
            result.position_size_multiplier == 0.80,
            f"Position size reduced 20%: {result.position_size_multiplier}"
        )
        # Signal direction should still be positive (3 bulls > 2 bears)
        check(
            result.final_signal > 0,
            f"Net bullish signal: {result.final_signal:.3f}"
        )
        check(
            result.signal_metadata["agreement"]["pattern"] == "moderate",
            "Pattern: moderate"
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 14: RegimeAwareCombiner — regime change shifts weights
# ---------------------------------------------------------------------------

def test_14_combiner_regime_change():
    print("\n--- Test 14: RegimeAwareCombiner \u2014 regime weight shifts ---")
    d = tmp()
    try:
        combiner = RegimeAwareCombiner(state_dir=d)

        # Use regime_detector-style weights
        bull_weights = {
            "PLAYER_1": 0.25,  # Aggressive
            "PLAYER_2": 0.15,  # Conservative
            "PLAYER_3": 0.20,  # Balanced
            "PLAYER_4": 0.10,  # VolBreakout
            "PLAYER_5": 0.30,  # Momentum
        }
        bear_weights = {
            "PLAYER_1": 0.10,
            "PLAYER_2": 0.35,
            "PLAYER_3": 0.15,
            "PLAYER_4": 0.30,
            "PLAYER_5": 0.10,
        }

        signals = make_bullish_signals()

        # Bull regime
        result_bull = combiner.combine(signals, regime="Bull", regime_weights=bull_weights)
        blended_bull = result_bull.signal_metadata["blended_weights"]

        # Bear regime
        result_bear = combiner.combine(signals, regime="Bear", regime_weights=bear_weights)
        blended_bear = result_bear.signal_metadata["blended_weights"]

        # In Bull, Momentum (P5) should have more weight
        check(
            blended_bull["PLAYER_5"] > blended_bear["PLAYER_5"],
            f"Momentum weight: Bull={blended_bull['PLAYER_5']:.3f} > Bear={blended_bear['PLAYER_5']:.3f}"
        )

        # In Bear, Conservative (P2) should have more weight
        check(
            blended_bear["PLAYER_2"] > blended_bull["PLAYER_2"],
            f"Conservative weight: Bear={blended_bear['PLAYER_2']:.3f} > Bull={blended_bull['PLAYER_2']:.3f}"
        )

        # In Bear, VolBreakout (P4) should have more weight
        check(
            blended_bear["PLAYER_4"] > blended_bull["PLAYER_4"],
            f"VolBreakout weight: Bear={blended_bear['PLAYER_4']:.3f} > Bull={blended_bull['PLAYER_4']:.3f}"
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 15: CombinerDashboard
# ---------------------------------------------------------------------------

def test_15_dashboard():
    print("\n--- Test 15: CombinerDashboard \u2014 summary ---")
    d = tmp()
    try:
        combiner = RegimeAwareCombiner(state_dir=d)

        # Generate some data
        signals = make_bullish_signals()
        combiner.combine(signals, regime="Bull")
        combiner.combine(make_split_signals_3_2(), regime="Sideways")

        # Record some outcomes
        combiner.record_trade_outcome(TradeOutcome("PLAYER_1", 0.80, 200.0, "Bull"))
        combiner.record_trade_outcome(TradeOutcome("PLAYER_2", 0.60, -100.0, "Bull"))

        dashboard = CombinerDashboard(combiner)
        summary = dashboard.summary()

        check("bayesian_weights_by_regime" in summary, "Has Bayesian weights")
        check("rolling_accuracy" in summary, "Has rolling accuracy")
        check("regime_accuracy" in summary, "Has regime accuracy")
        check("agreement_patterns_last_50" in summary, "Has agreement patterns")
        check("most_influential_player" in summary, "Has most influential")
        check("least_influential_player" in summary, "Has least influential")

        # Verify structure
        bw = summary["bayesian_weights_by_regime"]
        check("Bull" in bw and "Bear" in bw and "Sideways" in bw, "Has all 3 regimes")

        ra = summary["rolling_accuracy"]
        check("last_20" in ra and "last_50" in ra and "last_100" in ra, "Has all accuracy windows")

        patterns = summary["agreement_patterns_last_50"]
        check(patterns["unanimous"] >= 1, f"1+ unanimous: {patterns['unanimous']}")
        check(patterns["moderate"] >= 1, f"1+ moderate: {patterns['moderate']}")

        check(summary["total_combinations_logged"] == 2, "2 combinations logged")
        check(summary["total_outcomes_tracked"] == 2, "2 outcomes tracked")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 16: CombinationLogger — accuracy tracking
# ---------------------------------------------------------------------------

def test_16_logger_accuracy():
    print("\n--- Test 16: CombinationLogger \u2014 accuracy ---")
    d = tmp()
    try:
        lgr = CombinationLogger(state_dir=d)

        # 7 correct, 3 wrong = 70% accuracy
        for i in range(7):
            lgr.record_outcome(0.5, 100.0, "Bull")  # correct
        for i in range(3):
            lgr.record_outcome(0.5, -100.0, "Bull")  # wrong

        acc = lgr.rolling_accuracy(last_n=10)
        check(abs(acc - 0.70) < 0.001, f"Overall accuracy = {acc:.2f} ≈ 0.70")

        # Regime-specific
        lgr.record_outcome(-0.5, -100.0, "Bear")  # correct (bearish + loss)
        lgr.record_outcome(-0.5, 100.0, "Bear")   # wrong
        # Wait — bearish signal (-0.5) with actual_pnl = -100 means we shorted and lost.
        # Actually: correct = (dir > 0 and pnl > 0) or (dir < 0 and pnl < 0)
        # So: dir=-0.5, pnl=-100 → correct. dir=-0.5, pnl=100 → wrong (signal said short but made money going long?)
        # The logic is: if you signaled short and PnL was negative, it's "correct" that the bearish prediction matched negative outcome

        bear_acc = lgr.regime_accuracy("Bear")
        check(abs(bear_acc - 0.50) < 0.001, f"Bear accuracy = {bear_acc:.2f} ≈ 0.50")

        bull_acc = lgr.regime_accuracy("Bull")
        check(abs(bull_acc - 0.70) < 0.001, f"Bull accuracy = {bull_acc:.2f} ≈ 0.70")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 17: Integration — Bayesian downweight after bad trades
# ---------------------------------------------------------------------------

def test_17_integration_bayesian_downweight():
    print("\n--- Test 17: Integration \u2014 Bayesian downweight over time ---")
    d = tmp()
    try:
        combiner = RegimeAwareCombiner(state_dir=d)

        # Initial: uniform weights
        initial_w = combiner.bayesian_weights.get_weights("Bull")
        check(
            abs(initial_w["PLAYER_5"] - 0.20) < 0.001,
            f"PLAYER_5 starts at 0.20"
        )

        # PLAYER_5 (Momentum) makes 15 bad calls in Bull
        for _ in range(15):
            combiner.record_trade_outcome(
                TradeOutcome("PLAYER_5", 0.80, -300.0, "Bull")
            )

        # Other players make good calls
        for _ in range(15):
            for pid in ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4"]:
                combiner.record_trade_outcome(
                    TradeOutcome(pid, 0.70, 200.0, "Bull")
                )

        # Now check P5 is downweighted
        new_w = combiner.bayesian_weights.get_weights("Bull")
        check(
            new_w["PLAYER_5"] < 0.15,
            f"PLAYER_5 downweighted: {new_w['PLAYER_5']:.4f} < 0.15"
        )
        check(
            new_w["PLAYER_5"] >= 0.05 - 1e-6,
            f"PLAYER_5 above floor: {new_w['PLAYER_5']:.6f} ≥ 0.05"
        )

        # Now combine signals — PLAYER_5 should have less influence
        # Give all players same direction magnitude and confidence
        # so weight is the sole differentiator
        signals = [
            PlayerSignal("PLAYER_1", 0.70, 0.80, "bullish"),
            PlayerSignal("PLAYER_2", 0.70, 0.80, "bullish"),
            PlayerSignal("PLAYER_3", 0.70, 0.80, "bullish"),
            PlayerSignal("PLAYER_4", 0.70, 0.80, "bullish"),
            PlayerSignal("PLAYER_5", -0.70, 0.80, "BEARISH — Momentum fading"),
        ]

        result = combiner.combine(signals, regime="Bull")

        # Despite P5's bearish call, 4 bulls should dominate
        check(
            result.final_signal > 0,
            f"P5's bearish signal overridden: final={result.final_signal:.3f}"
        )

        # P5 has lowest blended weight → lowest absolute influence
        # (same |direction| and same confidence, so weight is the only variable)
        infl = result.signal_metadata.get("influence_map", {})
        p5_infl = abs(infl.get("PLAYER_5", 0))
        p1_infl = abs(infl.get("PLAYER_1", 0))
        check(
            p5_infl < p1_infl,
            f"PLAYER_5 influence ({p5_infl:.4f}) < PLAYER_1 ({p1_infl:.4f})"
        )
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Bonus: Voter with empty signals
# ---------------------------------------------------------------------------

def test_bonus_voter_empty():
    print("\n--- Bonus: Empty signals ---")
    voter = ConfidenceWeightedVoter()
    final, infl = voter.combine([], {})
    check(final == 0.0, f"Empty signals → 0.0: {final}")

    analyzer = AgreementAnalyzer()
    result = analyzer.analyze([])
    check(result.should_trade is False, "Empty → no trade")
    check(result.pattern == "no_majority", "Empty → no_majority")


# ---------------------------------------------------------------------------
# Bonus: PlayerSignal clamping
# ---------------------------------------------------------------------------

def test_bonus_clamping():
    print("\n--- Bonus: PlayerSignal clamping ---")
    s = PlayerSignal("P1", 2.5, 1.5, "extreme")
    check(s.direction == 1.0, f"Direction clamped to 1.0: {s.direction}")
    check(s.confidence == 1.0, f"Confidence clamped to 1.0: {s.confidence}")

    s2 = PlayerSignal("P2", -3.0, -0.5, "negative")
    check(s2.direction == -1.0, f"Direction clamped to -1.0: {s2.direction}")
    check(s2.confidence == 0.0, f"Confidence clamped to 0.0: {s2.confidence}")


# ---------------------------------------------------------------------------
# Bonus: Persistence round-trip
# ---------------------------------------------------------------------------

def test_bonus_persistence():
    print("\n--- Bonus: Bayesian persistence round-trip ---")
    d = tmp()
    try:
        bw1 = BayesianPlayerWeights(state_dir=d)
        for _ in range(5):
            bw1.update(TradeOutcome("PLAYER_1", 0.80, 300.0, "Bull"))

        weights_before = bw1.get_weights("Bull")

        # Create new instance (should load from disk)
        bw2 = BayesianPlayerWeights(state_dir=d)
        weights_after = bw2.get_weights("Bull")

        for pid in PLAYER_IDS:
            check(
                abs(weights_before[pid] - weights_after[pid]) < 0.001,
                f"{pid}: {weights_before[pid]:.4f} ≈ {weights_after[pid]:.4f}"
            )
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Bonus: No-majority skip in combiner
# ---------------------------------------------------------------------------

def test_bonus_combiner_skip():
    print("\n--- Bonus: Combiner skips on no majority ---")
    d = tmp()
    try:
        combiner = RegimeAwareCombiner(state_dir=d)
        signals = make_no_majority_signals()

        result = combiner.combine(signals, regime="Sideways")

        check(result.should_trade is False, "Trade skipped")
        check(result.final_signal == 0.0, f"Signal = 0: {result.final_signal}")
        check(result.position_size_multiplier == 0.0, "Size = 0")
        check("No clear majority" in result.signal_metadata.get("reason", ""), "Reason explains skip")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Edge case: All confidences = 0 → no division by zero
# ---------------------------------------------------------------------------

def test_edge_all_zero_confidence():
    print("\n--- Edge: All confidences = 0 ---")
    voter = ConfidenceWeightedVoter()
    signals = [
        PlayerSignal("PLAYER_1", 0.80, 0.0, "zero conf"),
        PlayerSignal("PLAYER_2", -0.60, 0.0, "zero conf"),
        PlayerSignal("PLAYER_3", 0.50, 0.0, "zero conf"),
        PlayerSignal("PLAYER_4", -0.70, 0.0, "zero conf"),
        PlayerSignal("PLAYER_5", 0.30, 0.0, "zero conf"),
    ]
    weights = {p: 0.20 for p in PLAYER_IDS}
    final, influence = voter.combine(signals, weights)

    check(final == 0.0, f"Zero confidence → final=0.0: {final}")
    for pid in PLAYER_IDS:
        check(influence[pid] == 0.0, f"{pid} influence = 0.0")


# ---------------------------------------------------------------------------
# Edge case: Novel regime string (e.g. "Correction") → no KeyError
# ---------------------------------------------------------------------------

def test_edge_novel_regime():
    print("\n--- Edge: Novel regime string ---")
    d = tmp()
    try:
        bw = BayesianPlayerWeights(state_dir=d)

        # get_weights should NOT KeyError on unknown regime
        weights = bw.get_weights("Correction")
        check(len(weights) == 5, f"Novel regime returns 5 weights: {len(weights)}")
        total = sum(weights.values())
        check(abs(total - 1.0) < 0.001, f"Novel regime weights sum to 1.0: {total:.4f}")

        # update should NOT KeyError on unknown regime
        new_w = bw.update(TradeOutcome("PLAYER_1", 0.70, 200.0, "Correction"))
        check(new_w["PLAYER_1"] > 0.20, f"Update in novel regime works: P1={new_w['PLAYER_1']:.4f}")

        # RegimeAwareCombiner should handle novel regime too
        combiner = RegimeAwareCombiner(state_dir=d)
        signals = make_bullish_signals()
        result = combiner.combine(signals, regime="Correction")
        check(result.should_trade is True, "Novel regime combine works")
        check(result.final_signal > 0, f"Novel regime signal positive: {result.final_signal:.3f}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Edge case: All 5 players submit identical signals
# ---------------------------------------------------------------------------

def test_edge_identical_signals():
    print("\n--- Edge: All 5 identical signals ---")
    analyzer = AgreementAnalyzer()
    # All 5 same direction, same confidence
    signals = [PlayerSignal(pid, 0.50, 0.80, "same") for pid in PLAYER_IDS]

    result = analyzer.analyze(signals)
    check(result.pattern == "unanimous", f"Identical → unanimous: {result.pattern}")
    check(result.majority_count == 5, f"All 5 in majority: {result.majority_count}")
    check(result.crowding_flag is True, "Crowding flag set")
    check(result.dissent_count == 0, f"No dissenters: {result.dissent_count}")

    # Also test with voter: identical signals should give back the same direction
    voter = ConfidenceWeightedVoter()
    weights = {p: 0.20 for p in PLAYER_IDS}
    final, _ = voter.combine(signals, weights)
    check(abs(final - 0.50) < 0.001, f"Identical signals → direction preserved: {final:.4f}")


# ---------------------------------------------------------------------------
# Edge case: Neutral direction (0.0) in record_outcome
# ---------------------------------------------------------------------------

def test_edge_neutral_direction_outcome():
    print("\n--- Edge: Neutral direction in record_outcome ---")
    d = tmp()
    try:
        lgr = CombinationLogger(state_dir=d)

        # Neutral signal should be marked as incorrect (no directional call)
        lgr.record_outcome(0.0, 500.0, "Bull")
        lgr.record_outcome(0.0, -500.0, "Bull")
        acc = lgr.rolling_accuracy(2)
        check(acc == 0.0, f"Neutral predictions → 0% accuracy: {acc:.2f}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Edge case: Persistence with missing player_id backfill
# ---------------------------------------------------------------------------

def test_edge_persistence_missing_player():
    print("\n--- Edge: Persistence load with missing player backfill ---")
    d = tmp()
    try:
        # Create initial state with only 3 players
        import json
        d.mkdir(parents=True, exist_ok=True)
        partial_data = {
            "weights": {
                "Bull": {"PLAYER_1": 0.30, "PLAYER_2": 0.30, "PLAYER_3": 0.40},
            },
            "scores": {
                "Bull": {"PLAYER_1": 1.0, "PLAYER_2": 0.5, "PLAYER_3": -0.5},
            },
            "history": [],
        }
        with open(d / "bayesian_weights.json", "w") as f:
            json.dump(partial_data, f)

        # Load — should backfill PLAYER_4 and PLAYER_5
        bw = BayesianPlayerWeights(state_dir=d)
        weights = bw.get_weights("Bull")
        check("PLAYER_4" in weights, "PLAYER_4 backfilled in weights")
        check("PLAYER_5" in weights, "PLAYER_5 backfilled in weights")
        check(len(weights) == 5, f"All 5 players present: {len(weights)}")

        # Scores should also be backfilled — update should work
        new_w = bw.update(TradeOutcome("PLAYER_4", 0.60, 200.0, "Bull"))
        check("PLAYER_4" in new_w, "PLAYER_4 update works after backfill")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Dynamic Signal Combination System \u2014 Test Suite")
    print("=" * 70)

    test_1_voter_basic()
    test_2_voter_confidence()
    test_3_agreement_unanimous()
    test_4_agreement_strong()
    test_5_agreement_3_2_split()
    test_6_agreement_no_majority()
    test_7_bayesian_init()
    test_8_bayesian_correct_upweight()
    test_9_bayesian_poor_performer()
    test_10_bayesian_regime_conditional()
    test_11_bayesian_constraints()
    test_12_combiner_all_bullish()
    test_13_combiner_3_2_split()
    test_14_combiner_regime_change()
    test_15_dashboard()
    test_16_logger_accuracy()
    test_17_integration_bayesian_downweight()
    test_bonus_voter_empty()
    test_bonus_clamping()
    test_bonus_persistence()
    test_bonus_combiner_skip()
    test_edge_all_zero_confidence()
    test_edge_novel_regime()
    test_edge_identical_signals()
    test_edge_neutral_direction_outcome()
    test_edge_persistence_missing_player()

    print("\n" + "=" * 70)
    print(f"RESULTS: {_pass} passed, {_fail} failed out of {_pass + _fail} checks")
    print("=" * 70)

    if _fail > 0:
        sys.exit(1)
