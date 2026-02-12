"""
Player Self-Reflection and Memory System for the 5-Player Trading System.

Enables players to learn from their own trading history through LLM-powered
post-hoc analysis. Each player accumulates rules and adjustments over time,
creating an evolving "trading wisdom" that is injected into future prompts.

Components:
  - TradeReflectionEngine: LLM-powered post-trade analysis
  - ReflectionScheduler:   Triggers reflections after every N trades
  - RuleManager:           Accumulates, scores, and deprecates trading rules
  - ReflectionStore:       JSON persistence for reflection history
  - PlayerMemoryContext:   Builds reflection context for prompt injection
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from coach_system.llm.base import LLMProvider
except ImportError:
    LLMProvider = None  # type: ignore[assignment,misc]

REFLECTIONS_DIR = Path(__file__).parent / "player_reflections"

PLAYER_LABELS = {
    "PLAYER_1": "Aggressive",
    "PLAYER_2": "Conservative",
    "PLAYER_3": "Balanced",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "Momentum",
}

_DEFAULT_PLAYER_DESCRIPTIONS: Dict[str, str] = {
    "Aggressive": (
        "High risk tolerance, short holding periods, momentum-focused. "
        "Fast momentum and reversal signals."
    ),
    "Conservative": (
        "Low risk tolerance, longer holds, trend-following. "
        "Stable trend-following with low drawdown."
    ),
    "Balanced": (
        "Medium risk, diversified indicators. "
        "Mix of momentum and mean-reversion."
    ),
    "VolBreakout": (
        "Volatility breakout specialist, catches big moves."
    ),
    "Momentum": (
        "Pure momentum player, rides trends. Momentum-based trading."
    ),
}

_VALID_CALIBRATIONS = {"over_confident", "under_confident", "well_calibrated"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReflectionRule:
    """A single rule generated from a reflection cycle."""

    rule_id: str
    text: str
    created_at: str
    created_in_regime: str
    source_reflection_id: str
    trades_following: int = 0
    trades_following_pnl: float = 0.0
    trades_ignoring: int = 0
    trades_ignoring_pnl: float = 0.0
    deprecated: bool = False
    deprecated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "text": self.text,
            "created_at": self.created_at,
            "created_in_regime": self.created_in_regime,
            "source_reflection_id": self.source_reflection_id,
            "trades_following": self.trades_following,
            "trades_following_pnl": self.trades_following_pnl,
            "trades_ignoring": self.trades_ignoring,
            "trades_ignoring_pnl": self.trades_ignoring_pnl,
            "deprecated": self.deprecated,
            "deprecated_at": self.deprecated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReflectionRule":
        return cls(
            rule_id=d["rule_id"],
            text=d["text"],
            created_at=d["created_at"],
            created_in_regime=d.get("created_in_regime", "Sideways"),
            source_reflection_id=d.get("source_reflection_id", ""),
            trades_following=d.get("trades_following", 0),
            trades_following_pnl=d.get("trades_following_pnl", 0.0),
            trades_ignoring=d.get("trades_ignoring", 0),
            trades_ignoring_pnl=d.get("trades_ignoring_pnl", 0.0),
            deprecated=d.get("deprecated", False),
            deprecated_at=d.get("deprecated_at"),
        )

    def score(self) -> float:
        """Compute the rule's effectiveness lift score."""
        if self.trades_following == 0:
            return -1.0  # untested
        avg_following = self.trades_following_pnl / self.trades_following
        avg_ignoring = (
            (self.trades_ignoring_pnl / self.trades_ignoring)
            if self.trades_ignoring > 0
            else 0.0
        )
        return avg_following - avg_ignoring


@dataclass
class ReflectionResult:
    """Output from a single reflection cycle."""

    reflection_id: str
    player_id: str
    player_label: str
    regime: str
    timestamp: str
    trade_count: int
    trade_ids: List[int] = field(default_factory=list)
    reflection_summary: str = ""
    rules: List[str] = field(default_factory=list)
    indicator_adjustments: Dict[str, float] = field(default_factory=dict)
    confidence_calibration: str = "unknown"
    key_insight: str = ""
    raw_llm_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reflection_id": self.reflection_id,
            "player_id": self.player_id,
            "player_label": self.player_label,
            "regime": self.regime,
            "timestamp": self.timestamp,
            "trade_count": self.trade_count,
            "trade_ids": self.trade_ids,
            "reflection_summary": self.reflection_summary,
            "rules": self.rules,
            "indicator_adjustments": self.indicator_adjustments,
            "confidence_calibration": self.confidence_calibration,
            "key_insight": self.key_insight,
            "raw_llm_response": self.raw_llm_response,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ReflectionResult":
        return cls(
            reflection_id=d["reflection_id"],
            player_id=d["player_id"],
            player_label=d.get("player_label", ""),
            regime=d.get("regime", "Sideways"),
            timestamp=d.get("timestamp", ""),
            trade_count=d.get("trade_count", 0),
            trade_ids=d.get("trade_ids", []),
            reflection_summary=d.get("reflection_summary", ""),
            rules=d.get("rules", []),
            indicator_adjustments=d.get("indicator_adjustments", {}),
            confidence_calibration=d.get("confidence_calibration", "unknown"),
            key_insight=d.get("key_insight", ""),
            raw_llm_response=d.get("raw_llm_response", {}),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_regime(regime: str) -> str:
    if not regime or not isinstance(regime, str):
        return "Sideways"
    return regime.strip().title()


def _rule_id_from_text(text: str) -> str:
    h = hashlib.md5(text.strip().lower().encode()).hexdigest()[:8]
    return f"rule_{h}"


def _now_iso() -> str:
    return datetime.now().isoformat()


# ---------------------------------------------------------------------------
# ReflectionStore — persistence layer
# ---------------------------------------------------------------------------

class ReflectionStore:
    """JSON persistence for reflection history."""

    _MAX_STORED = 50

    def __init__(self, reflections_dir: Optional[Path] = None):
        self._dir = reflections_dir or REFLECTIONS_DIR

    def save_reflection(self, player_id: str, result: ReflectionResult) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{player_id}_reflections.json"

        existing = self._load_raw(path)
        reflections = existing.get("reflections", [])
        reflections.append(result.to_dict())

        # Keep last N
        reflections = reflections[-self._MAX_STORED:]

        data = {
            "player_id": player_id,
            "reflections": reflections,
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[ReflectionStore] Failed to save {player_id}: {e}")

    def get_reflections(
        self,
        player_id: str,
        last_n: int = 3,
        regime: Optional[str] = None,
    ) -> List[ReflectionResult]:
        path = self._dir / f"{player_id}_reflections.json"
        raw = self._load_raw(path)
        reflections = raw.get("reflections", [])

        if regime:
            regime = _normalize_regime(regime)
            reflections = [r for r in reflections if r.get("regime") == regime]

        reflections = reflections[-last_n:]
        return [ReflectionResult.from_dict(r) for r in reflections]

    def get_all_reflections(self, player_id: str) -> List[ReflectionResult]:
        path = self._dir / f"{player_id}_reflections.json"
        raw = self._load_raw(path)
        return [ReflectionResult.from_dict(r) for r in raw.get("reflections", [])]

    def _load_raw(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[ReflectionStore] Corrupt file {path}: {e}")
            return {}


# ---------------------------------------------------------------------------
# RuleManager — accumulation, scoring, deprecation
# ---------------------------------------------------------------------------

class RuleManager:
    """Manages the lifecycle of trading rules per player."""

    def __init__(
        self,
        max_rules_per_player: int = 15,
        deprecation_days: int = 30,
        reflections_dir: Optional[Path] = None,
    ):
        self._max_rules = max_rules_per_player
        self._deprecation_days = deprecation_days
        self._dir = reflections_dir or REFLECTIONS_DIR
        self._rules: Dict[str, List[ReflectionRule]] = {}

    # -- Public API --

    def add_rules_from_reflection(
        self,
        player_id: str,
        reflection: ReflectionResult,
    ) -> List[ReflectionRule]:
        self._ensure_loaded(player_id)
        existing_texts = {
            r.text.strip().lower() for r in self._rules[player_id] if not r.deprecated
        }

        new_rules: List[ReflectionRule] = []
        for rule_text in reflection.rules:
            if not rule_text or not isinstance(rule_text, str):
                continue
            normalized = rule_text.strip().lower()
            if normalized in existing_texts:
                continue  # duplicate

            rule = ReflectionRule(
                rule_id=_rule_id_from_text(rule_text),
                text=rule_text.strip(),
                created_at=_now_iso(),
                created_in_regime=_normalize_regime(reflection.regime),
                source_reflection_id=reflection.reflection_id,
            )
            self._rules[player_id].append(rule)
            existing_texts.add(normalized)
            new_rules.append(rule)

        self._enforce_cap(player_id)
        self._save_rules(player_id)
        return new_rules

    def get_active_rules(
        self,
        player_id: str,
        regime: Optional[str] = None,
    ) -> List[ReflectionRule]:
        self._ensure_loaded(player_id)
        active = [r for r in self._rules[player_id] if not r.deprecated]
        if regime:
            regime = _normalize_regime(regime)
            active = [r for r in active if r.created_in_regime == regime]
        # newest first
        active.sort(key=lambda r: r.created_at, reverse=True)
        return active

    def update_rule_stats(
        self,
        player_id: str,
        rule_id: str,
        followed: bool,
        trade_pnl: float,
    ) -> None:
        self._ensure_loaded(player_id)
        for rule in self._rules[player_id]:
            if rule.rule_id == rule_id:
                if followed:
                    rule.trades_following += 1
                    rule.trades_following_pnl += trade_pnl
                else:
                    rule.trades_ignoring += 1
                    rule.trades_ignoring_pnl += trade_pnl
                self._save_rules(player_id)
                return

    def run_deprecation_check(self, player_id: str) -> List[str]:
        self._ensure_loaded(player_id)
        now = datetime.now()
        deprecated_ids: List[str] = []

        for rule in self._rules[player_id]:
            if rule.deprecated:
                continue
            try:
                created = datetime.fromisoformat(rule.created_at)
            except (ValueError, TypeError):
                continue
            age_days = (now - created).days
            if age_days <= self._deprecation_days:
                continue

            # Old enough to check
            if rule.trades_following == 0:
                # Never tested in 30+ days → deprecate
                rule.deprecated = True
                rule.deprecated_at = _now_iso()
                deprecated_ids.append(rule.rule_id)
            else:
                lift = rule.score()
                if lift <= 0:
                    rule.deprecated = True
                    rule.deprecated_at = _now_iso()
                    deprecated_ids.append(rule.rule_id)

        if deprecated_ids:
            self._save_rules(player_id)
        return deprecated_ids

    # -- Internal --

    def _enforce_cap(self, player_id: str) -> None:
        rules = self._rules[player_id]

        # Remove deprecated entirely
        active = [r for r in rules if not r.deprecated]

        if len(active) <= self._max_rules:
            self._rules[player_id] = rules  # keep deprecated in file for history
            return

        # Too many active: deprecate worst-scoring
        active.sort(key=lambda r: r.score())
        to_remove = len(active) - self._max_rules
        for i in range(to_remove):
            active[i].deprecated = True
            active[i].deprecated_at = _now_iso()

        # Rebuild: all deprecated removed, active kept
        self._rules[player_id] = [r for r in rules if not r.deprecated] + [
            r for r in rules if r.deprecated
        ]

    def _ensure_loaded(self, player_id: str) -> None:
        if player_id not in self._rules:
            self._rules[player_id] = self._load_rules(player_id)

    def _load_rules(self, player_id: str) -> List[ReflectionRule]:
        path = self._dir / f"{player_id}_rules.json"
        if not path.exists():
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return [ReflectionRule.from_dict(r) for r in data.get("rules", [])]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[RuleManager] Corrupt file {path}: {e}")
            return []

    def _save_rules(self, player_id: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{player_id}_rules.json"
        data = {
            "player_id": player_id,
            "rules": [r.to_dict() for r in self._rules.get(player_id, [])],
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[RuleManager] Failed to save {player_id}: {e}")


# ---------------------------------------------------------------------------
# TradeReflectionEngine — LLM-powered post-trade analysis
# ---------------------------------------------------------------------------

_REFLECTION_PROMPT_TEMPLATE = """You are the {player_name} player in a multi-agent trading system.
Your trading personality: {player_description}
Current market regime: {current_regime}

Review these recent trades:
{trade_details_formatted}

For each losing trade, analyze:
  - Which indicators gave false signals?
  - Was the regime appropriate for this trade type?
  - What market condition did you miss?

For each winning trade, analyze:
  - Which indicators were most predictive?
  - Could you have sized larger or held longer?

Then generate:
  - A list of 3-5 specific RULES to follow going forward (e.g., "In sideways regime, do not enter momentum trades with RSI > 60")
  - Indicator weight adjustment suggestions (e.g., {{"RSI_7": -0.10}} means decrease RSI_7 weight by 10%)
  - A confidence calibration note (are you over-confident or under-confident based on results?)

Respond in JSON format:
{{
  "reflection_summary": "...",
  "rules": ["rule1", "rule2", ...],
  "indicator_adjustments": {{"indicator_name": adjustment_pct, ...}},
  "confidence_calibration": "over_confident" | "under_confident" | "well_calibrated",
  "key_insight": "single most important lesson"
}}"""


class TradeReflectionEngine:
    """Calls LLM to analyze trades post-hoc and produce structured reflections."""

    def __init__(
        self,
        llm: Any,  # LLMProvider
        player_descriptions: Optional[Dict[str, str]] = None,
    ):
        self._llm = llm
        self._descriptions = player_descriptions or _DEFAULT_PLAYER_DESCRIPTIONS

    def reflect(
        self,
        player_id: str,
        player_label: str,
        trades: List[Dict],
        regime: str,
        trade_start_index: int = 0,
    ) -> ReflectionResult:
        regime = _normalize_regime(regime)
        now = _now_iso()
        reflection_id = f"refl_{player_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if not trades:
            return ReflectionResult(
                reflection_id=reflection_id,
                player_id=player_id,
                player_label=player_label,
                regime=regime,
                timestamp=now,
                trade_count=0,
                trade_ids=[],
                reflection_summary="No trades to reflect on.",
            )

        trade_ids = list(range(trade_start_index, trade_start_index + len(trades)))

        prompt = self._build_prompt(
            player_name=player_label,
            player_description=self._descriptions.get(
                player_label, "General trading player."
            ),
            regime=regime,
            trades=trades,
        )

        parsed = self._call_llm_json(prompt)

        # Handle parse failure
        if "parse_error" in parsed:
            logger.warning(
                f"[Reflection] LLM returned unparseable response for {player_id}: "
                f"{parsed.get('parse_error')}"
            )
            return ReflectionResult(
                reflection_id=reflection_id,
                player_id=player_id,
                player_label=player_label,
                regime=regime,
                timestamp=now,
                trade_count=len(trades),
                trade_ids=trade_ids,
                raw_llm_response=parsed,
            )

        # Extract fields with safe defaults
        rules = parsed.get("rules", [])
        if not isinstance(rules, list):
            rules = []
        rules = [str(r) for r in rules if r]

        adjustments = parsed.get("indicator_adjustments", {})
        if not isinstance(adjustments, dict):
            adjustments = {}
        # Ensure values are floats
        clean_adj: Dict[str, float] = {}
        for k, v in adjustments.items():
            try:
                clean_adj[str(k)] = float(v)
            except (ValueError, TypeError):
                continue

        calibration = str(parsed.get("confidence_calibration", "unknown"))
        if calibration not in _VALID_CALIBRATIONS:
            calibration = "unknown"

        return ReflectionResult(
            reflection_id=reflection_id,
            player_id=player_id,
            player_label=player_label,
            regime=regime,
            timestamp=now,
            trade_count=len(trades),
            trade_ids=trade_ids,
            reflection_summary=str(parsed.get("reflection_summary", "")),
            rules=rules,
            indicator_adjustments=clean_adj,
            confidence_calibration=calibration,
            key_insight=str(parsed.get("key_insight", "")),
            raw_llm_response=parsed,
        )

    def _call_llm_json(self, prompt: str) -> Dict[str, Any]:
        """Call LLM and parse JSON from response.

        Uses generate() + custom parsing to avoid the generate_json() bug
        where nested arrays inside objects cause mis-extraction.
        """
        try:
            raw = self._llm.generate(prompt)
        except Exception as e:
            return {"raw_response": str(e), "parse_error": str(e)}

        text = raw.strip()

        # Strip markdown code fences
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        # Find the outermost { ... } (object only — we always expect an object)
        brace_start = text.find("{")
        if brace_start == -1:
            return {"raw_response": raw, "parse_error": "No JSON object found"}

        depth = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            if depth == 0:
                text = text[brace_start : i + 1]
                break

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"[Reflection] JSON parse error: {e}")
            return {"raw_response": raw, "parse_error": str(e)}

    def _build_prompt(
        self,
        player_name: str,
        player_description: str,
        regime: str,
        trades: List[Dict],
    ) -> str:
        lines: List[str] = []
        winners = [t for t in trades if t.get("pnl", 0) > 0]
        losers = [t for t in trades if t.get("pnl", 0) <= 0]

        if losers:
            lines.append("LOSING TRADES:")
            for t in losers:
                lines.append(self._format_trade(t))
        if winners:
            lines.append("WINNING TRADES:")
            for t in winners:
                lines.append(self._format_trade(t))

        trade_details = "\n".join(lines)

        return _REFLECTION_PROMPT_TEMPLATE.format(
            player_name=player_name,
            player_description=player_description,
            current_regime=regime,
            trade_details_formatted=trade_details,
        )

    @staticmethod
    def _format_trade(t: Dict) -> str:
        direction = t.get("direction", "LONG")
        symbol = t.get("symbol", "UNKNOWN")
        entry = t.get("entry_price", 0)
        exit_ = t.get("exit_price", 0)
        pnl = t.get("pnl", 0)
        bars = t.get("bars_held", 0)
        reason = t.get("exit_reason", "unknown")
        return (
            f"  {direction} {symbol}: entry {entry:.2f} -> exit {exit_:.2f}, "
            f"PnL {pnl:+.2f}, held {bars} bars, exit: {reason}"
        )


# ---------------------------------------------------------------------------
# ReflectionScheduler — triggers reflections at the right cadence
# ---------------------------------------------------------------------------

class ReflectionScheduler:
    """Triggers reflections after every N completed trades per player."""

    _STATE_FILE = "_scheduler_state.json"

    def __init__(
        self,
        engine: TradeReflectionEngine,
        rule_manager: RuleManager,
        reflection_store: ReflectionStore,
        trigger_every_n_trades: int = 10,
    ):
        self._engine = engine
        self._rule_manager = rule_manager
        self._store = reflection_store
        self._trigger_every = trigger_every_n_trades
        self._last_reflected_index: Dict[str, int] = {}
        self._load_state()

    def check_and_reflect(
        self,
        player_id: str,
        player_label: str,
        all_trades: List[Dict],
        regime: str,
    ) -> Optional[ReflectionResult]:
        last_idx = self._last_reflected_index.get(player_id, 0)
        new_trades = all_trades[last_idx:]

        if len(new_trades) < self._trigger_every:
            return None

        result = self._engine.reflect(
            player_id=player_id,
            player_label=player_label,
            trades=new_trades,
            regime=regime,
            trade_start_index=last_idx,
        )

        self._last_reflected_index[player_id] = len(all_trades)
        self._store.save_reflection(player_id, result)
        self._rule_manager.add_rules_from_reflection(player_id, result)
        self._rule_manager.run_deprecation_check(player_id)
        self._save_state()
        return result

    def force_reflect(
        self,
        player_id: str,
        player_label: str,
        trades: List[Dict],
        regime: str,
    ) -> ReflectionResult:
        result = self._engine.reflect(
            player_id=player_id,
            player_label=player_label,
            trades=trades,
            regime=regime,
            trade_start_index=0,
        )
        self._store.save_reflection(player_id, result)
        self._rule_manager.add_rules_from_reflection(player_id, result)
        # NOTE: force_reflect does NOT advance _last_reflected_index
        return result

    def get_unreflected_count(self, player_id: str, total_trades: int) -> int:
        last_idx = self._last_reflected_index.get(player_id, 0)
        return max(0, total_trades - last_idx)

    def _load_state(self) -> None:
        path = self._store._dir / self._STATE_FILE
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self._last_reflected_index = data.get("last_reflected_index", {})
        except (json.JSONDecodeError, Exception):
            pass

    def _save_state(self) -> None:
        self._store._dir.mkdir(parents=True, exist_ok=True)
        path = self._store._dir / self._STATE_FILE
        data = {"last_reflected_index": self._last_reflected_index}
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"[ReflectionScheduler] Failed to save state: {e}")


# ---------------------------------------------------------------------------
# PlayerMemoryContext — builds reflection context for prompt injection
# ---------------------------------------------------------------------------

class PlayerMemoryContext:
    """Builds a reflection-aware context string for player prompt injection."""

    def __init__(
        self,
        reflection_store: ReflectionStore,
        rule_manager: RuleManager,
        regime_player: Optional[Any] = None,
    ):
        self._store = reflection_store
        self._rules = rule_manager
        self._regime_player = regime_player

    def build_context(
        self,
        player_id: str,
        player_label: str,
        regime: str,
        current_symbols: Optional[List[str]] = None,
        max_tokens: int = 1500,
    ) -> str:
        regime = _normalize_regime(regime)
        sections: List[str] = []

        # Section 1: Reflection history
        refl_summary = self.get_reflection_summary(player_id, last_n=3)
        if refl_summary:
            sections.append(f"=== SELF-REFLECTION HISTORY ===\n{refl_summary}")

        # Section 2: Active rules
        rules_text = self.get_rules_for_prompt(player_id, regime=None)
        if rules_text:
            active_count = len(self._rules.get_active_rules(player_id))
            max_count = self._rules._max_rules
            sections.append(
                f"=== ACTIVE RULES ({active_count}/{max_count}) ===\n{rules_text}"
            )

        # Section 3: Latest indicator adjustments
        recent = self._store.get_reflections(player_id, last_n=1)
        if recent and recent[0].indicator_adjustments:
            adj = recent[0].indicator_adjustments
            adj_lines = [f"  {k}: {v:+.2f}" for k, v in adj.items()]
            sections.append(
                "=== INDICATOR ADJUSTMENTS (latest reflection) ===\n"
                + "\n".join(adj_lines)
            )

        # Section 4: Regime performance stats
        if self._regime_player is not None:
            try:
                perf = self._regime_player.get_regime_performance_summary()
                if perf:
                    sections.append(f"=== REGIME PERFORMANCE STATS ===\n{perf}")
            except Exception:
                pass

        if not sections:
            return ""

        full_text = "\n\n".join(sections)

        # Rough token budget trim (1 token ≈ 4 chars)
        max_chars = max_tokens * 4
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n[...truncated]"

        return full_text

    def get_rules_for_prompt(
        self,
        player_id: str,
        regime: Optional[str] = None,
    ) -> str:
        active = self._rules.get_active_rules(player_id, regime=regime)
        if not active:
            return ""
        lines = []
        for i, rule in enumerate(active, 1):
            lines.append(f"  {i}. [{rule.created_in_regime}] {rule.text}")
        return "\n".join(lines)

    def get_reflection_summary(self, player_id: str, last_n: int = 3) -> str:
        reflections = self._store.get_reflections(player_id, last_n=last_n)
        if not reflections:
            return ""
        blocks: List[str] = []
        for r in reflections:
            ts = r.timestamp[:10] if len(r.timestamp) >= 10 else r.timestamp
            block = (
                f"  [{ts}] Regime: {r.regime} | "
                f"Confidence: {r.confidence_calibration}\n"
                f"    Insight: {r.key_insight}\n"
                f"    Summary: {r.reflection_summary[:200]}"
            )
            blocks.append(block)
        return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

def create_reflection_system(
    llm: Any,
    reflections_dir: Optional[Path] = None,
    trigger_every: int = 10,
) -> Dict[str, Any]:
    """Factory function to create all reflection components.

    Returns dict with keys: engine, store, rule_manager, scheduler, memory_context_factory.
    """
    store = ReflectionStore(reflections_dir=reflections_dir)
    rule_mgr = RuleManager(reflections_dir=reflections_dir)
    engine = TradeReflectionEngine(llm=llm)
    scheduler = ReflectionScheduler(
        engine=engine,
        rule_manager=rule_mgr,
        reflection_store=store,
        trigger_every_n_trades=trigger_every,
    )

    def make_memory_context(regime_player=None):
        return PlayerMemoryContext(
            reflection_store=store,
            rule_manager=rule_mgr,
            regime_player=regime_player,
        )

    return {
        "engine": engine,
        "store": store,
        "rule_manager": rule_mgr,
        "scheduler": scheduler,
        "memory_context_factory": make_memory_context,
    }
