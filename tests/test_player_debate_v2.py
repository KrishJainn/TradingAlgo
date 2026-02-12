#!/usr/bin/env python3
"""
Player Debate System — Targeted Test Suite v2

Tests the 10 specific scenarios requested:
  1.  All 5 players agree (same direction) — debate SKIPPED
  2.  All players confidence < 0.3 — debate SKIPPED
  3.  3-2 split (3 bullish, 2 bearish) — debate triggers, challenges issued
  4.  Post-debate signals have valid direction [-1,1] and confidence [0,1]
  5.  API cost tracking counts correctly (max 10 calls per debate)
  6.  DebateOutcome has all required fields
  7.  Player reverses position after challenge — changed=True
  8.  Debate analytics tracks which players change most often
  9.  Malformed LLM JSON — graceful fallback (keep original signal)
  10. Debate transcripts stored for later analysis

Run: python3 tests/test_player_debate_v2.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach_system.llm.base import MockLLMProvider

from player_debate import (
    ChallengeEntry,
    CostTracker,
    DebateAnalytics,
    DebateOutcome,
    DebateRound,
    DebateSignal,
    DebateStore,
    ResponseEntry,
    create_debate_system,
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
        print(f"  PASS  {label}")
    else:
        _fail += 1
        print(f"  FAIL  {label}")


def tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="test_debate_v2_"))


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------

# Per-player challenges for the 3-2 split scenario
MOCK_P1_CHALLENGE = json.dumps({
    "challenged_player": "PLAYER_2",
    "challenge": "Your SuperTrend bearish flip is a lagging signal.",
    "revised_direction": 0.78,
    "revised_confidence": 0.85,
    "stance": "maintain",
})

MOCK_P2_CHALLENGE = json.dumps({
    "challenged_player": "PLAYER_1",
    "challenge": "RSI_7 above 70 is overbought territory, not breakout.",
    "revised_direction": -0.55,
    "revised_confidence": 0.72,
    "stance": "maintain",
})

MOCK_P3_CHALLENGE = json.dumps({
    "challenged_player": "PLAYER_4",
    "challenge": "Keltner squeeze could resolve upward given momentum.",
    "revised_direction": 0.20,
    "revised_confidence": 0.50,
    "stance": "soften",
})

MOCK_P4_CHALLENGE = json.dumps({
    "challenged_player": "PLAYER_5",
    "challenge": "Momentum indicators are lagging in this regime.",
    "revised_direction": -0.40,
    "revised_confidence": 0.62,
    "stance": "maintain",
})

MOCK_P5_CHALLENGE = json.dumps({
    "challenged_player": "PLAYER_2",
    "challenge": "EMA_50 is a 50-period lagging average.",
    "revised_direction": 0.82,
    "revised_confidence": 0.88,
    "stance": "maintain",
})

# Response where player CONCEDES (changed=True)
MOCK_RESPONSE_CONCEDE = json.dumps({
    "response": "You raise a valid point. Reducing my bearish conviction.",
    "final_direction": -0.20,
    "final_confidence": 0.45,
    "changed": True,
})

# Response where player HOLDS FIRM (changed=False)
MOCK_RESPONSE_HOLD = json.dumps({
    "response": "I disagree. My indicators still support my position.",
    "final_direction": 0.75,
    "final_confidence": 0.82,
    "changed": False,
})

# Response where player REVERSES position (direction sign flips)
MOCK_RESPONSE_REVERSE = json.dumps({
    "response": "After seeing the momentum data, I'm now bullish.",
    "final_direction": 0.35,
    "final_confidence": 0.55,
    "changed": True,
})


# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------

def make_all_agree_bullish():
    """All 5 players bullish — should skip."""
    return {
        "PLAYER_1": DebateSignal("PLAYER_1", "Aggressive", "RELIANCE.NS", 0.70, 0.82,
                                 "RSI breakout confirmed", {"RSI_7": 0.85}),
        "PLAYER_2": DebateSignal("PLAYER_2", "Conservative", "RELIANCE.NS", 0.30, 0.65,
                                 "EMA support holding", {"EMA_50": 0.30}),
        "PLAYER_3": DebateSignal("PLAYER_3", "Balanced", "RELIANCE.NS", 0.40, 0.55,
                                 "Multiple signals aligned", {"BBANDS": 0.20}),
        "PLAYER_4": DebateSignal("PLAYER_4", "VolBreakout", "RELIANCE.NS", 0.55, 0.70,
                                 "Keltner breakout upward", {"KC_20": 0.50}),
        "PLAYER_5": DebateSignal("PLAYER_5", "Momentum", "RELIANCE.NS", 0.80, 0.90,
                                 "TSI/MACD/ROC all bullish", {"TSI": 0.75}),
    }


def make_all_agree_bearish():
    """All 5 players bearish — should skip."""
    return {
        "PLAYER_1": DebateSignal("PLAYER_1", "Aggressive", "TCS.NS", -0.60, 0.75,
                                 "RSI breakdown", {"RSI_7": -0.70}),
        "PLAYER_2": DebateSignal("PLAYER_2", "Conservative", "TCS.NS", -0.40, 0.60,
                                 "EMA resistance", {"EMA_50": -0.40}),
        "PLAYER_3": DebateSignal("PLAYER_3", "Balanced", "TCS.NS", -0.25, 0.50,
                                 "Bearish tilt", {"CCI_14": -0.30}),
        "PLAYER_4": DebateSignal("PLAYER_4", "VolBreakout", "TCS.NS", -0.70, 0.80,
                                 "Keltner squeeze down", {"KC_20": -0.65}),
        "PLAYER_5": DebateSignal("PLAYER_5", "Momentum", "TCS.NS", -0.50, 0.65,
                                 "Momentum fading", {"TSI": -0.55}),
    }


def make_all_low_conf():
    """Mixed directions but all confidence < 0.3."""
    return {
        "PLAYER_1": DebateSignal("PLAYER_1", "Aggressive", "INFY.NS", 0.60, 0.25,
                                 "Weak RSI signal", {"RSI_7": 0.50}),
        "PLAYER_2": DebateSignal("PLAYER_2", "Conservative", "INFY.NS", -0.40, 0.20,
                                 "Weak bearish EMA", {"EMA_50": -0.30}),
        "PLAYER_3": DebateSignal("PLAYER_3", "Balanced", "INFY.NS", 0.10, 0.15,
                                 "Almost no signal", {"BBANDS": 0.05}),
        "PLAYER_4": DebateSignal("PLAYER_4", "VolBreakout", "INFY.NS", -0.30, 0.28,
                                 "Low vol squeeze", {"KC_20": -0.20}),
        "PLAYER_5": DebateSignal("PLAYER_5", "Momentum", "INFY.NS", 0.20, 0.22,
                                 "Barely bullish", {"TSI": 0.15}),
    }


def make_3_2_split():
    """3 bullish, 2 bearish — should trigger debate."""
    return {
        "PLAYER_1": DebateSignal("PLAYER_1", "Aggressive", "RELIANCE.NS",
                                 0.75, 0.82, "Strong RSI_7 breakout",
                                 {"RSI_7": 0.85, "MACD": 0.72, "OBV": 0.60}),
        "PLAYER_2": DebateSignal("PLAYER_2", "Conservative", "RELIANCE.NS",
                                 -0.60, 0.70, "SuperTrend bearish",
                                 {"SUPERTREND": -0.80, "EMA_50": -0.45, "CMF": -0.55}),
        "PLAYER_3": DebateSignal("PLAYER_3", "Balanced", "RELIANCE.NS",
                                 0.15, 0.45, "Slightly bullish",
                                 {"RSI_14": 0.10, "BBANDS": -0.05}),
        "PLAYER_4": DebateSignal("PLAYER_4", "VolBreakout", "RELIANCE.NS",
                                 -0.45, 0.65, "Keltner squeeze resolving down",
                                 {"KC_20": -0.70, "ADX": 0.60}),
        "PLAYER_5": DebateSignal("PLAYER_5", "Momentum", "RELIANCE.NS",
                                 0.80, 0.85, "TSI crossover bullish",
                                 {"TSI": 0.75, "MACD": 0.80, "ROC": 0.65}),
    }


# ---------------------------------------------------------------------------
# Test 1: All 5 players agree (same direction) — debate SKIPPED
# ---------------------------------------------------------------------------

def test_1_all_agree_skip():
    print("\n--- Test 1: All 5 agree — debate SKIPPED ---")
    d = tmp()
    try:
        mock = MockLLMProvider(default_response=MOCK_P1_CHALLENGE)
        debate = DebateRound(llm=mock, debate_dir=d)

        # All bullish
        signals = make_all_agree_bullish()
        outcome = debate.run_debate(signals)

        check(outcome.skipped is True, "Bullish agreement: debate skipped")
        check(outcome.skip_reason == "all_agree_on_direction",
              f"Skip reason: {outcome.skip_reason}")
        check(outcome.api_calls_made == 0, f"Zero API calls: {outcome.api_calls_made}")
        check(len(outcome.challenges) == 0, "No challenges issued")
        check(len(outcome.responses) == 0, "No responses issued")
        check(len(mock._call_log) == 0, "LLM never called")

        # Verify post-signals = pre-signals exactly
        for pid in PLAYER_IDS:
            pre_d = outcome.pre_signals[pid].direction
            post_d = outcome.post_signals[pid].direction
            pre_c = outcome.pre_signals[pid].confidence
            post_c = outcome.post_signals[pid].confidence
            check(pre_d == post_d, f"{pid} direction unchanged: {pre_d}")
            check(pre_c == post_c, f"{pid} confidence unchanged: {pre_c}")

        # All bearish
        signals_bear = make_all_agree_bearish()
        outcome_bear = debate.run_debate(signals_bear)
        check(outcome_bear.skipped is True, "Bearish agreement: debate skipped")
        check(outcome_bear.skip_reason == "all_agree_on_direction",
              f"Bearish skip reason: {outcome_bear.skip_reason}")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: All players confidence < 0.3 — debate SKIPPED
# ---------------------------------------------------------------------------

def test_2_low_confidence_skip():
    print("\n--- Test 2: All confidence < 0.3 — debate SKIPPED ---")
    d = tmp()
    try:
        mock = MockLLMProvider(default_response=MOCK_P1_CHALLENGE)
        debate = DebateRound(llm=mock, debate_dir=d)

        signals = make_all_low_conf()
        outcome = debate.run_debate(signals)

        check(outcome.skipped is True, "Low confidence: debate skipped")
        check(outcome.skip_reason == "all_low_confidence",
              f"Skip reason: {outcome.skip_reason}")
        check(outcome.api_calls_made == 0, f"Zero API calls: {outcome.api_calls_made}")
        check(len(outcome.challenges) == 0, "No challenges")
        check(len(outcome.responses) == 0, "No responses")

        # Verify pre == post
        for pid in PLAYER_IDS:
            check(
                outcome.pre_signals[pid].direction == outcome.post_signals[pid].direction,
                f"{pid} signal unchanged after skip"
            )

        # Edge: one player at exactly 0.3 should NOT trigger skip
        signals["PLAYER_3"] = DebateSignal(
            "PLAYER_3", "Balanced", "INFY.NS", 0.10, 0.30, "threshold", {}
        )
        outcome2 = debate.run_debate(signals)
        check(outcome2.skipped is False,
              f"conf=0.30 NOT low enough to skip: skipped={outcome2.skipped}")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: 3-2 split — debate triggers, challenges issued
# ---------------------------------------------------------------------------

def test_3_split_debate_triggers():
    print("\n--- Test 3: 3-2 split — debate triggers ---")
    d = tmp()
    try:
        mock = MockLLMProvider(
            responses={
                "you are the aggressive": MOCK_P1_CHALLENGE,
                "you are the conservative": MOCK_P2_CHALLENGE,
                "you are the balanced": MOCK_P3_CHALLENGE,
                "you are the volbreakout": MOCK_P4_CHALLENGE,
                "you are the momentum": MOCK_P5_CHALLENGE,
                "challenge from": MOCK_RESPONSE_CONCEDE,
            },
            default_response=MOCK_P1_CHALLENGE,
        )
        debate = DebateRound(llm=mock, debate_dir=d)
        signals = make_3_2_split()

        outcome = debate.run_debate(signals)

        check(outcome.skipped is False, "3-2 split: debate NOT skipped")
        check(len(outcome.challenges) >= 3, f"Multiple challenges issued: {len(outcome.challenges)}")
        check(len(outcome.responses) >= 1, f"At least 1 response: {len(outcome.responses)}")

        # Verify each challenge has valid structure
        challengers_seen = set()
        targets_seen = set()
        for ch in outcome.challenges:
            challengers_seen.add(ch.challenger_id)
            targets_seen.add(ch.challenged_id)
            check(ch.challenger_id != ch.challenged_id,
                  f"{ch.challenger_label} didn't self-challenge")
            check(ch.challenged_id in signals,
                  f"Target {ch.challenged_id} exists in signals")
            check(len(ch.challenge_text) > 0, f"Challenge text non-empty for {ch.challenger_label}")

        check(len(challengers_seen) >= 3, f"At least 3 distinct challengers: {len(challengers_seen)}")

        # Verify bulls challenge bears and vice versa
        bull_ids = {"PLAYER_1", "PLAYER_3", "PLAYER_5"}
        bear_ids = {"PLAYER_2", "PLAYER_4"}
        cross_challenges = sum(
            1 for ch in outcome.challenges
            if (ch.challenger_id in bull_ids and ch.challenged_id in bear_ids)
            or (ch.challenger_id in bear_ids and ch.challenged_id in bull_ids)
        )
        check(cross_challenges >= 2, f"Cross-camp challenges: {cross_challenges}")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: Post-debate signals valid range [-1,1] and [0,1]
# ---------------------------------------------------------------------------

def test_4_post_signals_valid_range():
    print("\n--- Test 4: Post-debate signals valid range ---")
    d = tmp()
    try:
        # Use a mock that returns extreme values to stress-test clamping
        extreme_challenge = json.dumps({
            "challenged_player": "PLAYER_2",
            "challenge": "You are wrong.",
            "revised_direction": 2.5,     # Out of range
            "revised_confidence": 1.5,     # Out of range
            "stance": "maintain",
        })
        extreme_response = json.dumps({
            "response": "I concede completely.",
            "final_direction": -3.0,       # Out of range
            "final_confidence": -0.5,      # Out of range
            "changed": True,
        })

        mock = MockLLMProvider(
            responses={"challenge from": extreme_response},
            default_response=extreme_challenge,
        )
        debate = DebateRound(llm=mock, debate_dir=d)
        signals = make_3_2_split()

        outcome = debate.run_debate(signals)

        check(outcome.skipped is False, "Debate ran with extreme values")

        # Verify ALL post-signals are clamped
        for pid, sig in outcome.post_signals.items():
            check(-1.0 <= sig.direction <= 1.0,
                  f"{pid} post direction in [-1,1]: {sig.direction}")
            check(0.0 <= sig.confidence <= 1.0,
                  f"{pid} post confidence in [0,1]: {sig.confidence}")

        # Also check challenges have clamped values
        for ch in outcome.challenges:
            check(-1.0 <= ch.revised_direction <= 1.0,
                  f"Challenge revised_dir clamped: {ch.revised_direction}")
            check(0.0 <= ch.revised_confidence <= 1.0,
                  f"Challenge revised_conf clamped: {ch.revised_confidence}")

        # Check responses have clamped values
        for resp in outcome.responses:
            check(-1.0 <= resp.final_direction <= 1.0,
                  f"Response final_dir clamped: {resp.final_direction}")
            check(0.0 <= resp.final_confidence <= 1.0,
                  f"Response final_conf clamped: {resp.final_confidence}")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: API cost tracking (max 10 calls per debate)
# ---------------------------------------------------------------------------

def test_5_api_cost_tracking():
    print("\n--- Test 5: API cost tracking (max 10 calls) ---")
    d = tmp()
    try:
        mock = MockLLMProvider(
            responses={"challenge from": MOCK_RESPONSE_CONCEDE},
            default_response=MOCK_P1_CHALLENGE,
        )

        # Default max_api_calls=10
        debate = DebateRound(llm=mock, debate_dir=d)
        signals = make_3_2_split()

        outcome = debate.run_debate(signals)

        check(outcome.api_calls_made <= 10,
              f"API calls within cap: {outcome.api_calls_made} <= 10")
        check(outcome.api_calls_made > 0,
              f"At least 1 API call: {outcome.api_calls_made}")
        check(outcome.estimated_cost_usd > 0,
              f"Cost tracked: ${outcome.estimated_cost_usd:.6f}")

        # Verify LLM was actually called the right number of times
        actual_llm_calls = len(mock._call_log)
        check(actual_llm_calls == outcome.api_calls_made,
              f"LLM calls match tracker: {actual_llm_calls} == {outcome.api_calls_made}")

        # Test with explicit max_api_calls=4
        mock2 = MockLLMProvider(
            responses={"challenge from": MOCK_RESPONSE_CONCEDE},
            default_response=MOCK_P1_CHALLENGE,
        )
        debate2 = DebateRound(llm=mock2, max_api_calls=4, debate_dir=d)
        outcome2 = debate2.run_debate(make_3_2_split())

        check(outcome2.api_calls_made <= 4,
              f"Custom cap 4 enforced: {outcome2.api_calls_made} <= 4")
        check(len(mock2._call_log) <= 4,
              f"LLM called at most 4 times: {len(mock2._call_log)}")

        # Test cumulative cost tracking across debates
        tracker = debate.cost_tracker
        check(tracker.cumulative_calls == outcome.api_calls_made,
              f"Cumulative after 1 debate: {tracker.cumulative_calls}")

        # Run a second debate — cumulative should grow
        mock._call_log.clear()
        outcome3 = debate.run_debate(make_3_2_split())
        check(tracker.cumulative_calls == outcome.api_calls_made + outcome3.api_calls_made,
              f"Cumulative after 2 debates: {tracker.cumulative_calls}")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 6: DebateOutcome has all required fields
# ---------------------------------------------------------------------------

def test_6_outcome_required_fields():
    print("\n--- Test 6: DebateOutcome has all required fields ---")
    d = tmp()
    try:
        mock = MockLLMProvider(
            responses={"challenge from": MOCK_RESPONSE_CONCEDE},
            default_response=MOCK_P1_CHALLENGE,
        )
        debate = DebateRound(llm=mock, debate_dir=d)
        signals = make_3_2_split()
        outcome = debate.run_debate(signals)

        # Required fields
        check(isinstance(outcome.debate_id, str) and len(outcome.debate_id) > 0,
              f"debate_id present: {outcome.debate_id}")
        check(isinstance(outcome.timestamp, str) and len(outcome.timestamp) > 0,
              f"timestamp present: {outcome.timestamp[:20]}")
        check(outcome.symbol == "RELIANCE.NS", f"symbol: {outcome.symbol}")
        check(isinstance(outcome.skipped, bool), f"skipped is bool: {outcome.skipped}")

        # Pre-signals
        check(isinstance(outcome.pre_signals, dict), "pre_signals is dict")
        check(len(outcome.pre_signals) == 5, f"pre_signals has 5 entries: {len(outcome.pre_signals)}")
        for pid in PLAYER_IDS:
            check(pid in outcome.pre_signals, f"pre_signals has {pid}")
            check(isinstance(outcome.pre_signals[pid], DebateSignal),
                  f"pre_signals[{pid}] is DebateSignal")

        # Post-signals
        check(isinstance(outcome.post_signals, dict), "post_signals is dict")
        check(len(outcome.post_signals) == 5, f"post_signals has 5 entries: {len(outcome.post_signals)}")
        for pid in PLAYER_IDS:
            check(pid in outcome.post_signals, f"post_signals has {pid}")

        # Challenges
        check(isinstance(outcome.challenges, list), "challenges is list")
        for ch in outcome.challenges:
            check(isinstance(ch, ChallengeEntry), "challenge is ChallengeEntry")
            check(hasattr(ch, 'challenger_id'), "challenge has challenger_id")
            check(hasattr(ch, 'challenged_id'), "challenge has challenged_id")
            check(hasattr(ch, 'challenge_text'), "challenge has challenge_text")
            check(hasattr(ch, 'stance'), "challenge has stance")

        # Responses
        check(isinstance(outcome.responses, list), "responses is list")
        for resp in outcome.responses:
            check(isinstance(resp, ResponseEntry), "response is ResponseEntry")
            check(hasattr(resp, 'responder_id'), "response has responder_id")
            check(hasattr(resp, 'changed'), "response has changed")

        # Numeric fields
        check(isinstance(outcome.signals_changed, int), f"signals_changed is int: {outcome.signals_changed}")
        check(0.0 <= outcome.consensus_strength <= 1.0,
              f"consensus_strength in [0,1]: {outcome.consensus_strength:.3f}")
        check(isinstance(outcome.debate_summary, str) and len(outcome.debate_summary) > 0,
              f"debate_summary present: {outcome.debate_summary[:50]}")
        check(isinstance(outcome.api_calls_made, int), f"api_calls_made is int: {outcome.api_calls_made}")
        check(isinstance(outcome.estimated_cost_usd, float),
              f"estimated_cost_usd is float: {outcome.estimated_cost_usd}")

        # Serialization round-trip
        d_dict = outcome.to_dict()
        restored = DebateOutcome.from_dict(d_dict)
        check(restored.debate_id == outcome.debate_id, "Round-trip debate_id")
        check(restored.symbol == outcome.symbol, "Round-trip symbol")
        check(len(restored.challenges) == len(outcome.challenges), "Round-trip challenges count")
        check(len(restored.responses) == len(outcome.responses), "Round-trip responses count")
        check(len(restored.pre_signals) == len(outcome.pre_signals), "Round-trip pre_signals count")
        check(len(restored.post_signals) == len(outcome.post_signals), "Round-trip post_signals count")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 7: Player reverses position after challenge — changed=True
# ---------------------------------------------------------------------------

def test_7_player_reversal():
    print("\n--- Test 7: Player reverses position — changed=True ---")
    d = tmp()
    try:
        # Setup: PLAYER_1 (bullish +0.60) challenges PLAYER_2 (bearish -0.50)
        # PLAYER_2 responds with REVERSAL: direction flips from -0.50 to +0.35
        mock = MockLLMProvider(
            responses={
                "you are the aggressive": json.dumps({
                    "challenged_player": "PLAYER_2",
                    "challenge": "Your bearish signal misses the momentum breakout.",
                    "revised_direction": 0.65,
                    "revised_confidence": 0.85,
                    "stance": "maintain",
                }),
                "challenge from": MOCK_RESPONSE_REVERSE,   # reverses to +0.35
            },
            default_response=json.dumps({
                "challenged_player": "PLAYER_1",
                "challenge": "Generic challenge.",
                "revised_direction": -0.50,
                "revised_confidence": 0.70,
                "stance": "maintain",
            }),
        )
        debate = DebateRound(llm=mock, debate_dir=d)

        signals = {
            "PLAYER_1": DebateSignal("PLAYER_1", "Aggressive", "SBIN.NS",
                                     0.60, 0.80, "RSI bullish", {"RSI_7": 0.70}),
            "PLAYER_2": DebateSignal("PLAYER_2", "Conservative", "SBIN.NS",
                                     -0.50, 0.65, "SuperTrend bearish", {"SUPERTREND": -0.60}),
        }

        outcome = debate.run_debate(signals)

        check(outcome.skipped is False, "Debate triggered for disagreement")

        # Find PLAYER_2's response
        p2_responses = [r for r in outcome.responses if r.responder_id == "PLAYER_2"]
        check(len(p2_responses) > 0, f"PLAYER_2 responded: {len(p2_responses)} responses")

        if p2_responses:
            resp = p2_responses[0]
            check(resp.changed is True, f"PLAYER_2 changed=True: {resp.changed}")
            check(resp.final_direction > 0, f"PLAYER_2 reversed to bullish: {resp.final_direction:+.2f}")

            # Verify post-signal reflects the reversal
            p2_pre = outcome.pre_signals["PLAYER_2"].direction
            p2_post = outcome.post_signals["PLAYER_2"].direction
            check(p2_pre < 0, f"Pre-debate was bearish: {p2_pre:+.2f}")
            check(p2_post > 0, f"Post-debate is bullish: {p2_post:+.2f}")
            check(p2_pre * p2_post < 0, "Direction sign actually flipped (true reversal)")

        # Verify signals_changed counts this reversal
        check(outcome.signals_changed >= 1,
              f"At least 1 signal changed: {outcome.signals_changed}")

        # Verify debate_summary mentions the shift
        check("Signal shifts" in outcome.debate_summary or "signal" in outcome.debate_summary.lower(),
              f"Summary mentions change: {outcome.debate_summary[:60]}")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 8: Analytics tracks which players change most often
# ---------------------------------------------------------------------------

def test_8_analytics_change_tracking():
    print("\n--- Test 8: Analytics tracks player change frequency ---")
    d = tmp()
    try:
        store = DebateStore(debate_dir=d)
        analytics = DebateAnalytics(store=store)

        # Simulate 5 debates with known concession patterns:
        #   PLAYER_2 concedes in 4 out of 5 debates  (most adaptable)
        #   PLAYER_4 concedes in 2 out of 5 debates
        #   PLAYER_1 concedes in 0 debates            (never changes)
        #   PLAYER_3 concedes in 1 debate
        #   PLAYER_5 concedes in 0 debates

        debate_responses = [
            # Debate 0: P2 concedes, P4 concedes
            [ResponseEntry("PLAYER_2", "Conservative", "PLAYER_1", "ok", -0.20, 0.40, True),
             ResponseEntry("PLAYER_4", "VolBreakout", "PLAYER_5", "ok", -0.15, 0.35, True)],
            # Debate 1: P2 concedes
            [ResponseEntry("PLAYER_2", "Conservative", "PLAYER_5", "ok", -0.25, 0.45, True),
             ResponseEntry("PLAYER_1", "Aggressive", "PLAYER_2", "no", 0.70, 0.80, False)],
            # Debate 2: P2 concedes, P4 concedes
            [ResponseEntry("PLAYER_2", "Conservative", "PLAYER_1", "ok", -0.10, 0.35, True),
             ResponseEntry("PLAYER_4", "VolBreakout", "PLAYER_3", "ok", -0.20, 0.40, True)],
            # Debate 3: P2 concedes, P3 concedes
            [ResponseEntry("PLAYER_2", "Conservative", "PLAYER_5", "ok", -0.15, 0.40, True),
             ResponseEntry("PLAYER_3", "Balanced", "PLAYER_1", "ok", 0.25, 0.55, True)],
            # Debate 4: nobody concedes
            [ResponseEntry("PLAYER_2", "Conservative", "PLAYER_1", "no", -0.50, 0.65, False),
             ResponseEntry("PLAYER_5", "Momentum", "PLAYER_4", "no", 0.80, 0.85, False)],
        ]

        for i, responses in enumerate(debate_responses):
            outcome = DebateOutcome(
                debate_id=f"debate_{i}",
                timestamp="2026-02-06T12:00:00",
                symbol="RELIANCE.NS",
                responses=responses,
            )
            store.save_debate(outcome)

        # Mind changers
        mc = analytics.mind_changers(last_n=10)
        check(mc["PLAYER_2"] == 4, f"PLAYER_2 changed 4 times: {mc['PLAYER_2']}")
        check(mc["PLAYER_4"] == 2, f"PLAYER_4 changed 2 times: {mc['PLAYER_4']}")
        check(mc["PLAYER_3"] == 1, f"PLAYER_3 changed 1 time: {mc['PLAYER_3']}")
        check(mc["PLAYER_1"] == 0, f"PLAYER_1 never changed: {mc['PLAYER_1']}")
        check(mc["PLAYER_5"] == 0, f"PLAYER_5 never changed: {mc['PLAYER_5']}")

        # Most adaptable = PLAYER_2
        most_adaptable = max(mc, key=mc.get)
        check(most_adaptable == "PLAYER_2",
              f"Most adaptable: {PLAYER_LABELS.get(most_adaptable)} ({most_adaptable})")

        # Impactful challengers
        ic = analytics.impactful_challengers(last_n=10)
        # PLAYER_1 caused concessions in debates 0, 2 (P2) = 2
        # PLAYER_5 caused concessions in debates 1, 3 (P2) = 2
        # PLAYER_3 caused concession in debate 2 (P4) = 1
        # PLAYER_1 also: debate 3 (P3) = 1 more → total 3
        check(ic["PLAYER_1"] == 3, f"PLAYER_1 impact: {ic['PLAYER_1']} concessions caused")
        check(ic["PLAYER_5"] == 3, f"PLAYER_5 impact: {ic['PLAYER_5']} concessions caused")

        # Challenge frequency
        cf = analytics.challenge_frequency(last_n=10)
        check(isinstance(cf, dict), "challenge_frequency returns dict")

        # Summary aggregation
        summary = analytics.summary(last_n=10)
        check("mind_changers" in summary, "Summary includes mind_changers")
        check("impactful_challengers" in summary, "Summary includes impactful_challengers")
        check("average_signal_shift" in summary, "Summary includes average_signal_shift")
        check("skip_rate" in summary, "Summary includes skip_rate")
        check("cost" in summary, "Summary includes cost")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 9: Malformed LLM JSON — graceful fallback
# ---------------------------------------------------------------------------

def test_9_malformed_json_fallback():
    print("\n--- Test 9: Malformed LLM JSON — graceful fallback ---")
    d = tmp()
    try:
        # Test several kinds of malformed responses
        malformed_responses = [
            "This is not JSON at all",
            "```json\n{broken json\n```",
            "{ incomplete",
            "",
            "null",
            "<html><body>Error 500</body></html>",
        ]

        for i, bad_response in enumerate(malformed_responses):
            mock = MockLLMProvider(default_response=bad_response)
            debate = DebateRound(llm=mock, debate_dir=d)
            signals = make_3_2_split()

            outcome = debate.run_debate(signals)

            # Should not crash
            check(outcome.skipped is False,
                  f"Malformed #{i}: debate attempted (not skipped)")
            check(len(outcome.challenges) == 0,
                  f"Malformed #{i}: no valid challenges")
            check(len(outcome.responses) == 0,
                  f"Malformed #{i}: no responses needed")

            # Post-signals should match pre-signals exactly
            for pid in signals:
                pre_d = outcome.pre_signals[pid].direction
                post_d = outcome.post_signals[pid].direction
                check(abs(pre_d - post_d) < 0.001,
                      f"Malformed #{i}: {pid} signal preserved ({pre_d:+.2f})")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 10: Debate transcripts stored for later analysis
# ---------------------------------------------------------------------------

def test_10_transcripts_stored():
    print("\n--- Test 10: Debate transcripts stored ---")
    d = tmp()
    try:
        mock = MockLLMProvider(
            responses={
                "you are the aggressive": MOCK_P1_CHALLENGE,
                "you are the conservative": MOCK_P2_CHALLENGE,
                "you are the balanced": MOCK_P3_CHALLENGE,
                "you are the volbreakout": MOCK_P4_CHALLENGE,
                "you are the momentum": MOCK_P5_CHALLENGE,
                "challenge from": MOCK_RESPONSE_CONCEDE,
            },
            default_response=MOCK_P1_CHALLENGE,
        )

        system = create_debate_system(llm=mock, debate_dir=d)
        debate_round = system["debate_round"]
        store = system["store"]
        analytics = system["analytics"]

        # Run 3 debates and save each
        debate_ids = []
        for i in range(3):
            signals = make_3_2_split()
            outcome = debate_round.run_debate(signals)
            store.save_debate(outcome)
            debate_ids.append(outcome.debate_id)

        # Verify all 3 stored
        all_debates = store.get_all_debates()
        check(len(all_debates) == 3, f"3 debates stored: {len(all_debates)}")

        # Verify debate IDs match
        stored_ids = [db.debate_id for db in all_debates]
        for did in debate_ids:
            check(did in stored_ids, f"Debate {did} found in store")

        # Verify transcripts have full content
        for db in all_debates:
            check(len(db.challenges) > 0,
                  f"{db.debate_id}: has challenges ({len(db.challenges)})")
            check(db.symbol == "RELIANCE.NS",
                  f"{db.debate_id}: symbol preserved")
            check(len(db.pre_signals) == 5,
                  f"{db.debate_id}: pre_signals preserved ({len(db.pre_signals)})")
            check(len(db.post_signals) == 5,
                  f"{db.debate_id}: post_signals preserved ({len(db.post_signals)})")

        # Verify last_n retrieval
        last_2 = store.get_debates(last_n=2)
        check(len(last_2) == 2, f"get_debates(last_n=2) returns 2: {len(last_2)}")
        check(last_2[-1].debate_id == debate_ids[-1],
              f"Most recent debate is last: {last_2[-1].debate_id}")

        # Verify JSON file exists on disk
        history_path = d / "debate_history.json"
        check(history_path.exists(), "debate_history.json exists on disk")

        # Verify JSON is valid and parseable
        with open(history_path, "r") as f:
            raw = json.load(f)
        check("debates" in raw, "JSON has 'debates' key")
        check(len(raw["debates"]) == 3, f"JSON has 3 entries: {len(raw['debates'])}")

        # Verify challenge texts are preserved in transcript
        first_debate = all_debates[0]
        for ch in first_debate.challenges:
            check(len(ch.challenge_text) > 0,
                  f"Challenge text preserved: '{ch.challenge_text[:40]}...'")

        # Verify performance tracking (JSONL file)
        analytics.record_outcome("d1", "PLAYER_1", 0.75, 0.78, 150.0)
        analytics.record_outcome("d1", "PLAYER_2", -0.60, -0.20, 150.0)

        perf_path = d / "debate_performance.jsonl"
        check(perf_path.exists(), "debate_performance.jsonl exists")

        with open(perf_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]
        check(len(lines) == 2, f"2 performance records: {len(lines)}")

        record = json.loads(lines[0])
        check(record["player_id"] == "PLAYER_1", f"Record player_id: {record['player_id']}")
        check(record["pre_direction"] == 0.75, f"Record pre_dir: {record['pre_direction']}")
        check(record["post_correct"] is True, "Post-debate direction was correct")

    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Player Debate System — Targeted Test Suite v2")
    print("=" * 70)

    test_1_all_agree_skip()
    test_2_low_confidence_skip()
    test_3_split_debate_triggers()
    test_4_post_signals_valid_range()
    test_5_api_cost_tracking()
    test_6_outcome_required_fields()
    test_7_player_reversal()
    test_8_analytics_change_tracking()
    test_9_malformed_json_fallback()
    test_10_transcripts_stored()

    print("\n" + "=" * 70)
    print(f"RESULTS: {_pass} passed, {_fail} failed out of {_pass + _fail} checks")
    print("=" * 70)

    if _fail > 0:
        sys.exit(1)
