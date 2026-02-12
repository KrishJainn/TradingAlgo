#!/usr/bin/env python3
"""
Comprehensive test suite for the Player Self-Reflection and Memory System.

13 test categories, all using MockLLMProvider (no API calls).
Run: python tests/test_player_reflection.py
"""

import json
import shutil
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from coach_system.llm.base import MockLLMProvider
from player_reflection import (
    PlayerMemoryContext,
    ReflectionResult,
    ReflectionRule,
    ReflectionScheduler,
    ReflectionStore,
    RuleManager,
    TradeReflectionEngine,
    _normalize_regime,
    _rule_id_from_text,
    create_reflection_system,
)

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

_pass = 0
_fail = 0


def check(condition: bool, label: str) -> None:
    global _pass, _fail
    if condition:
        _pass += 1
        print(f"  ✅ {label}")
    else:
        _fail += 1
        print(f"  ❌ {label}")


# ---------------------------------------------------------------------------
# Shared mock data
# ---------------------------------------------------------------------------

VALID_REFLECTION_JSON = json.dumps({
    "reflection_summary": "Momentum strategy struggled in sideways market. Too many false breakout signals.",
    "rules": [
        "Avoid LONG entries when RSI_7 > 80 and ADX_14 < 20 in Sideways regime",
        "Reduce position size by 50% when regime is Sideways",
        "Require MACD histogram positive slope before LONG entry",
    ],
    "indicator_adjustments": {"RSI_7": -0.10, "ADX_14": 0.15, "MACD_12_26_9": 0.05},
    "confidence_calibration": "over_confident",
    "key_insight": "Momentum signals generate many false positives in sideways markets",
})


def make_mock_trades(n: int = 10, win_ratio: float = 0.4) -> list:
    trades = []
    symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]
    for i in range(n):
        is_win = i < int(n * win_ratio)
        pnl = 200.0 + i * 10 if is_win else -(150.0 + i * 5)
        entry = 1000.0 + i * 50
        exit_p = entry + pnl / 4  # rough calc
        trades.append({
            "symbol": symbols[i % len(symbols)],
            "direction": "LONG",
            "entry_price": entry,
            "exit_price": exit_p,
            "quantity": 4,
            "pnl": pnl,
            "exit_reason": "take_profit" if is_win else "stop_loss",
            "bars_held": 3 + i,
        })
    return trades


def make_temp_dir() -> Path:
    return Path(tempfile.mkdtemp(prefix="test_reflection_"))


# ---------------------------------------------------------------------------
# Test 1: Prompt construction
# ---------------------------------------------------------------------------

def test_1_prompt_construction():
    print("\n--- Test 1: TradeReflectionEngine — prompt construction ---")
    tmp = make_temp_dir()
    try:
        mock = MockLLMProvider(default_response=VALID_REFLECTION_JSON)
        engine = TradeReflectionEngine(llm=mock)

        trades = make_mock_trades(5, win_ratio=0.4)
        result = engine.reflect("PLAYER_5", "Momentum", trades, "Sideways")

        check(len(mock._call_log) == 1, "LLM called exactly once")

        prompt = mock._call_log[0]
        check("Momentum" in prompt, "Prompt contains player name 'Momentum'")
        check("Sideways" in prompt, "Prompt contains regime 'Sideways'")
        check("RELIANCE.NS" in prompt, "Prompt contains trade symbol")
        check("LOSING TRADES:" in prompt, "Prompt has losing trades section")
        check("WINNING TRADES:" in prompt, "Prompt has winning trades section")

        check(result.player_id == "PLAYER_5", "Result player_id correct")
        check(result.player_label == "Momentum", "Result player_label correct")
        check(result.regime == "Sideways", "Result regime correct")
        check(len(result.rules) == 3, f"Result has 3 rules (got {len(result.rules)})")
        check(result.confidence_calibration == "over_confident", "Confidence calibration parsed")
        check(result.key_insight != "", "Key insight non-empty")
        check(result.trade_count == 5, "Trade count correct")
        check(len(result.trade_ids) == 5, "Trade IDs correct length")
        check("RSI_7" in result.indicator_adjustments, "Indicator adjustments parsed")
        check(result.indicator_adjustments["RSI_7"] == -0.10, "RSI_7 adjustment value correct")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 2: Malformed LLM response
# ---------------------------------------------------------------------------

def test_2_malformed_llm_response():
    print("\n--- Test 2: TradeReflectionEngine — malformed LLM response ---")
    tmp = make_temp_dir()
    try:
        mock = MockLLMProvider(default_response="This is not valid JSON at all!!!")
        engine = TradeReflectionEngine(llm=mock)
        trades = make_mock_trades(3)

        result = engine.reflect("PLAYER_5", "Momentum", trades, "Bull")
        check(isinstance(result, ReflectionResult), "Returns ReflectionResult (no crash)")
        check(result.rules == [], "Rules list is empty on parse failure")
        check(result.indicator_adjustments == {}, "Indicator adjustments empty on failure")
        check("parse_error" in result.raw_llm_response, "raw_llm_response contains parse_error")
        check(result.trade_count == 3, "Trade count still correct")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 3: Scheduler trigger at N trades
# ---------------------------------------------------------------------------

def test_3_scheduler_trigger():
    print("\n--- Test 3: ReflectionScheduler — trigger after N trades ---")
    tmp = make_temp_dir()
    try:
        mock = MockLLMProvider(default_response=VALID_REFLECTION_JSON)
        engine = TradeReflectionEngine(llm=mock)
        store = ReflectionStore(reflections_dir=tmp)
        rule_mgr = RuleManager(reflections_dir=tmp)
        scheduler = ReflectionScheduler(
            engine=engine,
            rule_manager=rule_mgr,
            reflection_store=store,
            trigger_every_n_trades=5,
        )

        trades_4 = make_mock_trades(4)
        r1 = scheduler.check_and_reflect("PLAYER_5", "Momentum", trades_4, "Bull")
        check(r1 is None, "No reflection with 4 trades (threshold=5)")

        trades_5 = make_mock_trades(5)
        r2 = scheduler.check_and_reflect("PLAYER_5", "Momentum", trades_5, "Bull")
        check(r2 is not None, "Reflection triggered with 5 trades")
        check(r2.trade_count == 5, "Reflected on 5 trades")

        # Now trades 0-4 are reflected. Add 4 more (total 9)
        trades_9 = make_mock_trades(9)
        r3 = scheduler.check_and_reflect("PLAYER_5", "Momentum", trades_9, "Bull")
        check(r3 is None, "No reflection with only 4 new trades (9-5=4)")

        # Add 1 more (total 10 = 5 new unreflected)
        trades_10 = make_mock_trades(10)
        r4 = scheduler.check_and_reflect("PLAYER_5", "Momentum", trades_10, "Bull")
        check(r4 is not None, "Reflection triggered with 5 new trades")
        check(r4.trade_count == 5, "Only reflected on 5 new trades")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 4: No double-reflection
# ---------------------------------------------------------------------------

def test_4_no_double_reflection():
    print("\n--- Test 4: ReflectionScheduler — no double-reflection ---")
    tmp = make_temp_dir()
    try:
        mock = MockLLMProvider(default_response=VALID_REFLECTION_JSON)
        engine = TradeReflectionEngine(llm=mock)
        store = ReflectionStore(reflections_dir=tmp)
        rule_mgr = RuleManager(reflections_dir=tmp)
        scheduler = ReflectionScheduler(
            engine=engine,
            rule_manager=rule_mgr,
            reflection_store=store,
            trigger_every_n_trades=5,
        )

        trades = make_mock_trades(5)
        scheduler.check_and_reflect("PLAYER_5", "Momentum", trades, "Bull")

        call_count_before = len(mock._call_log)
        r2 = scheduler.check_and_reflect("PLAYER_5", "Momentum", trades, "Bull")
        check(r2 is None, "No reflection on same 5 trades")
        check(
            len(mock._call_log) == call_count_before,
            "LLM not called again"
        )
        check(
            scheduler.get_unreflected_count("PLAYER_5", 5) == 0,
            "Unreflected count is 0"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 5: Force reflect
# ---------------------------------------------------------------------------

def test_5_force_reflect():
    print("\n--- Test 5: ReflectionScheduler — force_reflect ---")
    tmp = make_temp_dir()
    try:
        mock = MockLLMProvider(default_response=VALID_REFLECTION_JSON)
        engine = TradeReflectionEngine(llm=mock)
        store = ReflectionStore(reflections_dir=tmp)
        rule_mgr = RuleManager(reflections_dir=tmp)
        scheduler = ReflectionScheduler(
            engine=engine,
            rule_manager=rule_mgr,
            reflection_store=store,
            trigger_every_n_trades=10,
        )

        trades = make_mock_trades(3)
        result = scheduler.force_reflect("PLAYER_5", "Momentum", trades, "Bear")
        check(result is not None, "force_reflect works below threshold")
        check(result.trade_count == 3, "Reflected on 3 trades")

        # Verify the last_reflected_index was NOT advanced
        unreflected = scheduler.get_unreflected_count("PLAYER_5", 3)
        check(
            unreflected == 3,
            f"Index NOT advanced after force_reflect (unreflected={unreflected})"
        )

        # Verify reflection was stored
        stored = store.get_reflections("PLAYER_5", last_n=1)
        check(len(stored) == 1, "Reflection stored despite force_reflect")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 6: Rule cap enforcement
# ---------------------------------------------------------------------------

def test_6_rule_cap():
    print("\n--- Test 6: RuleManager — rule cap enforcement ---")
    tmp = make_temp_dir()
    try:
        rule_mgr = RuleManager(max_rules_per_player=15, reflections_dir=tmp)

        # Add 4 reflections × 5 rules each = 20 unique rules
        for batch in range(4):
            refl = ReflectionResult(
                reflection_id=f"refl_test_{batch}",
                player_id="PLAYER_5",
                player_label="Momentum",
                regime="Sideways",
                timestamp=_now_iso(),
                trade_count=10,
                rules=[f"Rule {batch}_{i}: Do something specific {batch}_{i}" for i in range(5)],
            )
            rule_mgr.add_rules_from_reflection("PLAYER_5", refl)

        active = rule_mgr.get_active_rules("PLAYER_5")
        check(
            len(active) == 15,
            f"Active rules capped at 15 (got {len(active)})"
        )

        # The 5 oldest/worst-scoring should be deprecated
        all_rules = rule_mgr._rules["PLAYER_5"]
        deprecated = [r for r in all_rules if r.deprecated]
        check(
            len(deprecated) == 5,
            f"5 rules deprecated (got {len(deprecated)})"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _now_iso():
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Test 7: Rule deprecation
# ---------------------------------------------------------------------------

def test_7_rule_deprecation():
    print("\n--- Test 7: RuleManager — deprecation check ---")
    tmp = make_temp_dir()
    try:
        rule_mgr = RuleManager(deprecation_days=30, reflections_dir=tmp)
        rule_mgr._rules["PLAYER_5"] = []

        old_date = (datetime.now() - timedelta(days=31)).isoformat()
        new_date = (datetime.now() - timedelta(days=5)).isoformat()

        # Rule 1: Old + negative lift → should deprecate
        rule_mgr._rules["PLAYER_5"].append(ReflectionRule(
            rule_id="rule_bad_old",
            text="Bad old rule",
            created_at=old_date,
            created_in_regime="Sideways",
            source_reflection_id="refl_test",
            trades_following=3,
            trades_following_pnl=-100.0,
            trades_ignoring=2,
            trades_ignoring_pnl=50.0,
        ))

        # Rule 2: Old + positive lift → should NOT deprecate
        rule_mgr._rules["PLAYER_5"].append(ReflectionRule(
            rule_id="rule_good_old",
            text="Good old rule",
            created_at=old_date,
            created_in_regime="Bull",
            source_reflection_id="refl_test",
            trades_following=5,
            trades_following_pnl=500.0,
            trades_ignoring=3,
            trades_ignoring_pnl=-60.0,
        ))

        # Rule 3: Old + untested → should deprecate
        rule_mgr._rules["PLAYER_5"].append(ReflectionRule(
            rule_id="rule_untested_old",
            text="Untested old rule",
            created_at=old_date,
            created_in_regime="Bear",
            source_reflection_id="refl_test",
        ))

        # Rule 4: New + negative lift → should NOT deprecate (too young)
        rule_mgr._rules["PLAYER_5"].append(ReflectionRule(
            rule_id="rule_bad_new",
            text="Bad new rule",
            created_at=new_date,
            created_in_regime="Sideways",
            source_reflection_id="refl_test",
            trades_following=2,
            trades_following_pnl=-50.0,
            trades_ignoring=1,
            trades_ignoring_pnl=10.0,
        ))

        deprecated = rule_mgr.run_deprecation_check("PLAYER_5")

        check("rule_bad_old" in deprecated, "Old negative-lift rule deprecated")
        check("rule_good_old" not in deprecated, "Old positive-lift rule kept")
        check("rule_untested_old" in deprecated, "Old untested rule deprecated")
        check("rule_bad_new" not in deprecated, "New rule kept (too young)")

        active = rule_mgr.get_active_rules("PLAYER_5")
        active_ids = {r.rule_id for r in active}
        check("rule_good_old" in active_ids, "Good old rule still active")
        check("rule_bad_new" in active_ids, "Bad new rule still active")
        check(len(active) == 2, f"2 rules remain active (got {len(active)})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 8: Duplicate rule detection
# ---------------------------------------------------------------------------

def test_8_duplicate_rules():
    print("\n--- Test 8: RuleManager — duplicate detection ---")
    tmp = make_temp_dir()
    try:
        rule_mgr = RuleManager(reflections_dir=tmp)

        refl1 = ReflectionResult(
            reflection_id="refl_1",
            player_id="PLAYER_5",
            player_label="Momentum",
            regime="Bull",
            timestamp=_now_iso(),
            trade_count=10,
            rules=["Avoid LONG when RSI > 80", "Use tight stops in Bear regime"],
        )
        rule_mgr.add_rules_from_reflection("PLAYER_5", refl1)
        check(
            len(rule_mgr.get_active_rules("PLAYER_5")) == 2,
            "2 rules added initially"
        )

        # Same text, different case
        refl2 = ReflectionResult(
            reflection_id="refl_2",
            player_id="PLAYER_5",
            player_label="Momentum",
            regime="Bull",
            timestamp=_now_iso(),
            trade_count=10,
            rules=["AVOID LONG WHEN RSI > 80", "Brand new unique rule"],
        )
        rule_mgr.add_rules_from_reflection("PLAYER_5", refl2)
        active = rule_mgr.get_active_rules("PLAYER_5")
        check(
            len(active) == 3,
            f"Only 3 rules (duplicate skipped, got {len(active)})"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 9: ReflectionStore save/load round-trip
# ---------------------------------------------------------------------------

def test_9_store_roundtrip():
    print("\n--- Test 9: ReflectionStore — save/load round-trip ---")
    tmp = make_temp_dir()
    try:
        store = ReflectionStore(reflections_dir=tmp)

        original = ReflectionResult(
            reflection_id="refl_PLAYER_5_20260206_120000",
            player_id="PLAYER_5",
            player_label="Momentum",
            regime="Sideways",
            timestamp="2026-02-06T12:00:00",
            trade_count=10,
            trade_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            reflection_summary="Test summary",
            rules=["rule A", "rule B"],
            indicator_adjustments={"RSI_7": -0.10, "ADX_14": 0.15},
            confidence_calibration="over_confident",
            key_insight="Key insight here",
            raw_llm_response={"test": True},
        )

        store.save_reflection("PLAYER_5", original)
        loaded = store.get_reflections("PLAYER_5", last_n=1)

        check(len(loaded) == 1, "Loaded 1 reflection")
        r = loaded[0]
        check(r.reflection_id == original.reflection_id, "reflection_id matches")
        check(r.player_id == "PLAYER_5", "player_id matches")
        check(r.player_label == "Momentum", "player_label matches")
        check(r.regime == "Sideways", "regime matches")
        check(r.trade_count == 10, "trade_count matches")
        check(r.trade_ids == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "trade_ids match")
        check(r.reflection_summary == "Test summary", "reflection_summary matches")
        check(r.rules == ["rule A", "rule B"], "rules match")
        check(r.indicator_adjustments == {"RSI_7": -0.10, "ADX_14": 0.15}, "adjustments match")
        check(r.confidence_calibration == "over_confident", "calibration matches")
        check(r.key_insight == "Key insight here", "key_insight matches")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 10: Store keeps last 50
# ---------------------------------------------------------------------------

def test_10_store_max_50():
    print("\n--- Test 10: ReflectionStore — keeps last 50 ---")
    tmp = make_temp_dir()
    try:
        store = ReflectionStore(reflections_dir=tmp)

        for i in range(55):
            r = ReflectionResult(
                reflection_id=f"refl_{i}",
                player_id="PLAYER_5",
                player_label="Momentum",
                regime="Bull",
                timestamp=f"2026-01-{(i % 28) + 1:02d}T12:00:00",
                trade_count=10,
                rules=[f"rule_{i}"],
            )
            store.save_reflection("PLAYER_5", r)

        all_reflections = store.get_all_reflections("PLAYER_5")
        check(
            len(all_reflections) == 50,
            f"Store keeps max 50 (got {len(all_reflections)})"
        )
        check(
            all_reflections[0].reflection_id == "refl_5",
            "Oldest kept is refl_5 (first 5 pruned)"
        )
        check(
            all_reflections[-1].reflection_id == "refl_54",
            "Newest is refl_54"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 11: PlayerMemoryContext output
# ---------------------------------------------------------------------------

def test_11_memory_context_output():
    print("\n--- Test 11: PlayerMemoryContext — build_context output ---")
    tmp = make_temp_dir()
    try:
        store = ReflectionStore(reflections_dir=tmp)
        rule_mgr = RuleManager(reflections_dir=tmp)

        # Add 3 reflections
        for i in range(3):
            r = ReflectionResult(
                reflection_id=f"refl_{i}",
                player_id="PLAYER_5",
                player_label="Momentum",
                regime="Sideways",
                timestamp=f"2026-02-0{i+1}T12:00:00",
                trade_count=10,
                reflection_summary=f"Summary {i}",
                rules=[f"Rule from refl {i}: specific rule text {i}"],
                indicator_adjustments={"RSI_7": -0.05 * (i + 1)},
                confidence_calibration="over_confident",
                key_insight=f"Insight {i}",
            )
            store.save_reflection("PLAYER_5", r)
            rule_mgr.add_rules_from_reflection("PLAYER_5", r)

        ctx = PlayerMemoryContext(
            reflection_store=store,
            rule_manager=rule_mgr,
        )

        output = ctx.build_context("PLAYER_5", "Momentum", "Sideways")

        check("SELF-REFLECTION HISTORY" in output, "Contains reflection history section")
        check("ACTIVE RULES" in output, "Contains active rules section")
        check("INDICATOR ADJUSTMENTS" in output, "Contains indicator adjustments section")
        check("Insight 0" in output, "Contains key insight from reflection")
        check("over_confident" in output, "Contains confidence calibration")

        # Rules should be numbered
        check("1." in output, "Rules are numbered")

        # Verify rule count in header
        active = rule_mgr.get_active_rules("PLAYER_5")
        check(f"({len(active)}/15)" in output, "Rule count shown in header")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 12: Empty state
# ---------------------------------------------------------------------------

def test_12_empty_state():
    print("\n--- Test 12: PlayerMemoryContext — empty state ---")
    tmp = make_temp_dir()
    try:
        store = ReflectionStore(reflections_dir=tmp)
        rule_mgr = RuleManager(reflections_dir=tmp)
        ctx = PlayerMemoryContext(
            reflection_store=store,
            rule_manager=rule_mgr,
        )

        output = ctx.build_context("PLAYER_5", "Momentum", "Bull")
        check(output == "", f"Empty state returns empty string (got '{output[:50]}')")
        check(not output, "Empty string is falsy (no prompt pollution)")

        # Also test individual methods
        rules_text = ctx.get_rules_for_prompt("PLAYER_5")
        check(rules_text == "", "get_rules_for_prompt returns empty string")

        summary = ctx.get_reflection_summary("PLAYER_5")
        check(summary == "", "get_reflection_summary returns empty string")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Test 13: Full integration cycle
# ---------------------------------------------------------------------------

def test_13_full_integration():
    print("\n--- Test 13: Full integration cycle ---")
    tmp = make_temp_dir()
    try:
        mock = MockLLMProvider(default_response=VALID_REFLECTION_JSON)

        system = create_reflection_system(
            llm=mock,
            reflections_dir=tmp,
            trigger_every=10,
        )

        scheduler = system["scheduler"]
        store = system["store"]
        rule_mgr = system["rule_manager"]
        memory_ctx = system["memory_context_factory"]()

        # Feed 10 trades to trigger reflection
        trades = make_mock_trades(10, win_ratio=0.3)
        result = scheduler.check_and_reflect("PLAYER_5", "Momentum", trades, "Sideways")

        check(result is not None, "Reflection triggered after 10 trades")
        check(result.trade_count == 10, f"Reflected on 10 trades (got {result.trade_count})")
        check(len(result.rules) == 3, f"3 rules generated (got {len(result.rules)})")

        # Verify rules were stored
        active_rules = rule_mgr.get_active_rules("PLAYER_5")
        check(len(active_rules) == 3, f"3 active rules in manager (got {len(active_rules)})")

        # Verify reflection was stored
        stored = store.get_reflections("PLAYER_5", last_n=1)
        check(len(stored) == 1, "Reflection stored")
        check(stored[0].reflection_id == result.reflection_id, "Stored reflection ID matches")

        # Verify memory context builds
        context = memory_ctx.build_context("PLAYER_5", "Momentum", "Sideways")
        check(context != "", "Memory context is non-empty")
        check("SELF-REFLECTION HISTORY" in context, "Context has reflection history")
        check("ACTIVE RULES" in context, "Context has rules")
        check("INDICATOR ADJUSTMENTS" in context, "Context has indicator adjustments")
        check("Momentum signals generate many false positives" in context, "Context contains key insight")

        # Verify unreflected count
        unreflected = scheduler.get_unreflected_count("PLAYER_5", 10)
        check(unreflected == 0, f"0 unreflected trades (got {unreflected})")

        # Feed 5 more → should not trigger
        trades_15 = make_mock_trades(15)
        r2 = scheduler.check_and_reflect("PLAYER_5", "Momentum", trades_15, "Sideways")
        check(r2 is None, "No reflection with only 5 new trades")

        # Feed 10 more (total 20) → should trigger on trades 10-19
        trades_20 = make_mock_trades(20)
        r3 = scheduler.check_and_reflect("PLAYER_5", "Momentum", trades_20, "Sideways")
        check(r3 is not None, "Reflection triggered for second batch")

        # Rules should now be 3 (duplicates from same mock response skipped)
        active_rules_2 = rule_mgr.get_active_rules("PLAYER_5")
        check(
            len(active_rules_2) == 3,
            f"Still 3 rules (duplicates skipped, got {len(active_rules_2)})"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Bonus tests
# ---------------------------------------------------------------------------

def test_bonus_helpers():
    print("\n--- Bonus: Helper functions ---")

    check(_normalize_regime("bull") == "Bull", "Normalizes 'bull' → 'Bull'")
    check(_normalize_regime("  BEAR ") == "Bear", "Normalizes '  BEAR ' → 'Bear'")
    check(_normalize_regime("sideways") == "Sideways", "Normalizes 'sideways' → 'Sideways'")
    check(_normalize_regime("") == "Sideways", "Empty string → 'Sideways'")
    check(_normalize_regime(None) == "Sideways", "None → 'Sideways'")

    rid = _rule_id_from_text("Test Rule")
    check(rid.startswith("rule_"), "Rule ID starts with 'rule_'")
    check(len(rid) == 13, f"Rule ID is 13 chars (rule_ + 8 hex, got {len(rid)})")

    # Same text, different case → same ID
    rid2 = _rule_id_from_text("test rule")
    check(rid == rid2, "Same text different case → same rule_id")


def test_bonus_empty_trades():
    print("\n--- Bonus: Empty trade list ---")
    mock = MockLLMProvider(default_response=VALID_REFLECTION_JSON)
    engine = TradeReflectionEngine(llm=mock)
    result = engine.reflect("PLAYER_5", "Momentum", [], "Bull")

    check(result.trade_count == 0, "Trade count is 0")
    check(result.reflection_summary == "No trades to reflect on.", "Summary is placeholder")
    check(len(mock._call_log) == 0, "LLM NOT called for empty trades")


def test_bonus_rule_stats_update():
    print("\n--- Bonus: Rule stats update ---")
    tmp = make_temp_dir()
    try:
        rule_mgr = RuleManager(reflections_dir=tmp)
        rule_mgr._rules["PLAYER_5"] = [
            ReflectionRule(
                rule_id="rule_test_1",
                text="Test rule",
                created_at=_now_iso(),
                created_in_regime="Bull",
                source_reflection_id="refl_1",
            )
        ]

        rule_mgr.update_rule_stats("PLAYER_5", "rule_test_1", followed=True, trade_pnl=100.0)
        rule_mgr.update_rule_stats("PLAYER_5", "rule_test_1", followed=True, trade_pnl=-30.0)
        rule_mgr.update_rule_stats("PLAYER_5", "rule_test_1", followed=False, trade_pnl=50.0)

        rule = rule_mgr._rules["PLAYER_5"][0]
        check(rule.trades_following == 2, f"Following count = 2 (got {rule.trades_following})")
        check(rule.trades_following_pnl == 70.0, f"Following PnL = 70 (got {rule.trades_following_pnl})")
        check(rule.trades_ignoring == 1, f"Ignoring count = 1 (got {rule.trades_ignoring})")
        check(rule.trades_ignoring_pnl == 50.0, f"Ignoring PnL = 50 (got {rule.trades_ignoring_pnl})")

        score = rule.score()
        # avg_following = 70/2 = 35, avg_ignoring = 50/1 = 50, lift = 35 - 50 = -15
        check(abs(score - (-15.0)) < 0.01, f"Rule score = -15.0 (got {score:.2f})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bonus_regime_filter():
    print("\n--- Bonus: Regime filtering ---")
    tmp = make_temp_dir()
    try:
        store = ReflectionStore(reflections_dir=tmp)
        rule_mgr = RuleManager(reflections_dir=tmp)

        for regime in ["Bull", "Bear", "Sideways"]:
            r = ReflectionResult(
                reflection_id=f"refl_{regime}",
                player_id="PLAYER_5",
                player_label="Momentum",
                regime=regime,
                timestamp=_now_iso(),
                trade_count=10,
                reflection_summary=f"Summary for {regime}",
                rules=[f"Rule for {regime}"],
                key_insight=f"Insight for {regime}",
            )
            store.save_reflection("PLAYER_5", r)
            rule_mgr.add_rules_from_reflection("PLAYER_5", r)

        # Filter reflections by regime
        bull_only = store.get_reflections("PLAYER_5", last_n=10, regime="Bull")
        check(len(bull_only) == 1, f"1 Bull reflection (got {len(bull_only)})")
        check(bull_only[0].regime == "Bull", "Filtered reflection is Bull")

        # Filter rules by regime
        bear_rules = rule_mgr.get_active_rules("PLAYER_5", regime="Bear")
        check(len(bear_rules) == 1, f"1 Bear rule (got {len(bear_rules)})")
        check(bear_rules[0].created_in_regime == "Bear", "Rule is Bear-tagged")

        # All rules (no filter)
        all_rules = rule_mgr.get_active_rules("PLAYER_5")
        check(len(all_rules) == 3, f"3 total rules (got {len(all_rules)})")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Player Self-Reflection & Memory System — Test Suite")
    print("=" * 70)

    test_1_prompt_construction()
    test_2_malformed_llm_response()
    test_3_scheduler_trigger()
    test_4_no_double_reflection()
    test_5_force_reflect()
    test_6_rule_cap()
    test_7_rule_deprecation()
    test_8_duplicate_rules()
    test_9_store_roundtrip()
    test_10_store_max_50()
    test_11_memory_context_output()
    test_12_empty_state()
    test_13_full_integration()
    test_bonus_helpers()
    test_bonus_empty_trades()
    test_bonus_rule_stats_update()
    test_bonus_regime_filter()

    print("\n" + "=" * 70)
    print(f"RESULTS: {_pass} passed, {_fail} failed out of {_pass + _fail} checks")
    print("=" * 70)

    if _fail > 0:
        sys.exit(1)
