"""
Player Debate System for the 5-Player Trading Coach.

When players disagree on trade direction, a structured LLM-powered debate
refines signals before they reach the signal combiner. Each player can
challenge one other player's reasoning, and challenged players respond.

Flow:
  Players -> Signals -> **Debate** -> Revised Signals -> Combiner -> Trade

Components:
  - DebateRound:      Orchestrates a 2-round challenge/response debate
  - DebateOutcome:    Structured output capturing pre/post signals and analytics
  - DebateAnalytics:  Tracks mind-changers, impactful challengers, performance
  - DebateStore:      JSON persistence for debate transcripts
  - CostTracker:      Estimates and tracks LLM API cost per debate
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEBATE_DIR = Path(__file__).parent / "player_debates"

PLAYER_IDS = ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]

PLAYER_LABELS: Dict[str, str] = {
    "PLAYER_1": "Aggressive",
    "PLAYER_2": "Conservative",
    "PLAYER_3": "Balanced",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "Momentum",
}

_DEFAULT_PERSONALITIES: Dict[str, str] = {
    "Aggressive": (
        "High risk tolerance, short holding periods, momentum-focused. "
        "You believe in fast entries and exits, and that hesitation costs money."
    ),
    "Conservative": (
        "Low risk tolerance, longer holds, trend-following with strict risk management. "
        "You prioritize capital preservation over capturing every move."
    ),
    "Balanced": (
        "Medium risk, diversified indicators, mix of momentum and mean-reversion. "
        "You seek the middle ground and weigh multiple signals carefully."
    ),
    "VolBreakout": (
        "Volatility breakout specialist. You look for expanding ranges and "
        "Bollinger/Keltner squeezes. You wait for the right moment then act decisively."
    ),
    "Momentum": (
        "Pure momentum rider. You ride trends using MACD, RSI, and rate-of-change "
        "indicators. You believe the trend is your friend until proven otherwise."
    ),
}

# Gemini Flash pricing (per 1M tokens)
_GEMINI_COST_PER_MILLION = {
    "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-3-flash-preview": {"input": 0.10, "output": 0.40},
    "default": {"input": 0.10, "output": 0.40},
}

_CHARS_PER_TOKEN = 4

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CHALLENGE_PROMPT = """You are the {player_name} player in a 5-player trading system.
Your personality: {player_personality}

=== CURRENT SIGNALS FOR {symbol} ===
{all_signals_formatted}

You see that the players disagree. Review all signals above.

YOUR current signal: {own_direction_label} (direction={own_direction:+.2f}, confidence={own_confidence:.2f})
Your reasoning: {own_reasoning}

TASK: Issue ONE challenge to the player whose signal you most disagree with.
Explain why their reasoning is flawed given the current indicators.
You may also revise your own signal if seeing others' reasoning changes your view.

Respond in JSON:
{{
  "challenged_player": "PLAYER_X",
  "challenge": "Your specific challenge to that player's reasoning (2-3 sentences)",
  "revised_direction": <your updated direction float [-1.0 to 1.0]>,
  "revised_confidence": <your updated confidence float [0.0 to 1.0]>,
  "stance": "maintain" | "soften" | "reverse"
}}"""

_RESPONSE_PROMPT = """You are the {player_name} player in a 5-player trading system.
Your personality: {player_personality}

=== ORIGINAL SIGNALS FOR {symbol} ===
{all_signals_formatted}

YOUR original signal: {own_direction_label} (direction={own_direction:+.2f}, confidence={own_confidence:.2f})
Your reasoning: {own_reasoning}

=== CHALLENGE FROM {challenger_name} ===
"{challenge_text}"

The {challenger_name} player ({challenger_personality_short}) challenges your signal.

TASK: Respond to this challenge. Either defend your position or concede and adjust.
Be specific about which indicators support your response.

Respond in JSON:
{{
  "response": "Your defense or concession (2-3 sentences)",
  "final_direction": <your final direction float [-1.0 to 1.0]>,
  "final_confidence": <your final confidence float [0.0 to 1.0]>,
  "changed": true | false
}}"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class DebateSignal:
    """A player's signal as input to the debate."""

    player_id: str
    player_label: str
    symbol: str
    direction: float
    confidence: float
    reasoning: str = ""
    indicators: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.direction = max(-1.0, min(1.0, float(self.direction)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def direction_label(self) -> str:
        if self.direction > 0.1:
            return "BULLISH"
        elif self.direction < -0.1:
            return "BEARISH"
        return "NEUTRAL"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "player_label": self.player_label,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "indicators": dict(self.indicators),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DebateSignal":
        return cls(
            player_id=d["player_id"],
            player_label=d.get("player_label", ""),
            symbol=d.get("symbol", ""),
            direction=d.get("direction", 0.0),
            confidence=d.get("confidence", 0.5),
            reasoning=d.get("reasoning", ""),
            indicators=d.get("indicators", {}),
        )


@dataclass
class ChallengeEntry:
    """Round 1 output: a player's challenge to another player."""

    challenger_id: str
    challenger_label: str
    challenged_id: str
    challenged_label: str
    challenge_text: str
    revised_direction: float
    revised_confidence: float
    stance: str  # "maintain", "soften", "reverse"
    raw_llm_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenger_id": self.challenger_id,
            "challenger_label": self.challenger_label,
            "challenged_id": self.challenged_id,
            "challenged_label": self.challenged_label,
            "challenge_text": self.challenge_text,
            "revised_direction": self.revised_direction,
            "revised_confidence": self.revised_confidence,
            "stance": self.stance,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChallengeEntry":
        return cls(
            challenger_id=d["challenger_id"],
            challenger_label=d.get("challenger_label", ""),
            challenged_id=d.get("challenged_id", ""),
            challenged_label=d.get("challenged_label", ""),
            challenge_text=d.get("challenge_text", ""),
            revised_direction=d.get("revised_direction", 0.0),
            revised_confidence=d.get("revised_confidence", 0.5),
            stance=d.get("stance", "maintain"),
        )


@dataclass
class ResponseEntry:
    """Round 2 output: a challenged player's response."""

    responder_id: str
    responder_label: str
    challenger_id: str
    response_text: str
    final_direction: float
    final_confidence: float
    changed: bool
    raw_llm_response: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "responder_id": self.responder_id,
            "responder_label": self.responder_label,
            "challenger_id": self.challenger_id,
            "response_text": self.response_text,
            "final_direction": self.final_direction,
            "final_confidence": self.final_confidence,
            "changed": self.changed,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ResponseEntry":
        return cls(
            responder_id=d["responder_id"],
            responder_label=d.get("responder_label", ""),
            challenger_id=d.get("challenger_id", ""),
            response_text=d.get("response_text", ""),
            final_direction=d.get("final_direction", 0.0),
            final_confidence=d.get("final_confidence", 0.5),
            changed=d.get("changed", False),
        )


@dataclass
class DebateOutcome:
    """Complete output from one debate round."""

    debate_id: str
    timestamp: str
    symbol: str
    skipped: bool = False
    skip_reason: str = ""

    pre_signals: Dict[str, DebateSignal] = field(default_factory=dict)
    post_signals: Dict[str, DebateSignal] = field(default_factory=dict)
    challenges: List[ChallengeEntry] = field(default_factory=list)
    responses: List[ResponseEntry] = field(default_factory=list)

    signals_changed: int = 0
    consensus_strength: float = 0.0
    debate_summary: str = ""

    api_calls_made: int = 0
    estimated_cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "debate_id": self.debate_id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "pre_signals": {k: v.to_dict() for k, v in self.pre_signals.items()},
            "post_signals": {k: v.to_dict() for k, v in self.post_signals.items()},
            "challenges": [c.to_dict() for c in self.challenges],
            "responses": [r.to_dict() for r in self.responses],
            "signals_changed": self.signals_changed,
            "consensus_strength": self.consensus_strength,
            "debate_summary": self.debate_summary,
            "api_calls_made": self.api_calls_made,
            "estimated_cost_usd": self.estimated_cost_usd,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DebateOutcome":
        return cls(
            debate_id=d["debate_id"],
            timestamp=d.get("timestamp", ""),
            symbol=d.get("symbol", ""),
            skipped=d.get("skipped", False),
            skip_reason=d.get("skip_reason", ""),
            pre_signals={
                k: DebateSignal.from_dict(v)
                for k, v in d.get("pre_signals", {}).items()
            },
            post_signals={
                k: DebateSignal.from_dict(v)
                for k, v in d.get("post_signals", {}).items()
            },
            challenges=[
                ChallengeEntry.from_dict(c) for c in d.get("challenges", [])
            ],
            responses=[
                ResponseEntry.from_dict(r) for r in d.get("responses", [])
            ],
            signals_changed=d.get("signals_changed", 0),
            consensus_strength=d.get("consensus_strength", 0.0),
            debate_summary=d.get("debate_summary", ""),
            api_calls_made=d.get("api_calls_made", 0),
            estimated_cost_usd=d.get("estimated_cost_usd", 0.0),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_signals_block(signals: Dict[str, DebateSignal]) -> str:
    """Format all player signals into a readable block for prompts."""
    lines: List[str] = []
    for pid in PLAYER_IDS:
        if pid not in signals:
            continue
        sig = signals[pid]
        dir_label = sig.direction_label()
        ind_str = ""
        if sig.indicators:
            top_inds = sorted(
                sig.indicators.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]
            ind_str = " | Top indicators: " + ", ".join(
                f"{k}={v:+.2f}" for k, v in top_inds
            )
        lines.append(
            f"  {sig.player_label} ({pid}): {dir_label} "
            f"dir={sig.direction:+.2f} conf={sig.confidence:.2f} "
            f"-- {sig.reasoning}{ind_str}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Estimates and tracks LLM API cost per debate and cumulatively."""

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self._model = model_name
        self._pricing = _GEMINI_COST_PER_MILLION.get(
            model_name, _GEMINI_COST_PER_MILLION["default"]
        )
        self._debate_calls: int = 0
        self._debate_cost: float = 0.0
        self._cumulative_calls: int = 0
        self._cumulative_cost: float = 0.0

    def estimate_call_cost(self, prompt: str, response: str) -> float:
        """Estimate cost of a single LLM call in USD."""
        input_tokens = len(prompt) / _CHARS_PER_TOKEN
        output_tokens = len(response) / _CHARS_PER_TOKEN
        cost = (
            (input_tokens / 1_000_000) * self._pricing["input"]
            + (output_tokens / 1_000_000) * self._pricing["output"]
        )
        return cost

    def record_call(self, prompt: str, response: str) -> float:
        """Record a call and return its estimated cost."""
        cost = self.estimate_call_cost(prompt, response)
        self._debate_calls += 1
        self._debate_cost += cost
        self._cumulative_calls += 1
        self._cumulative_cost += cost
        return cost

    def reset_debate(self) -> Tuple[int, float]:
        """Reset per-debate counters, return (calls, cost) for just-ended debate."""
        calls, cost = self._debate_calls, self._debate_cost
        self._debate_calls = 0
        self._debate_cost = 0.0
        return calls, cost

    @property
    def debate_calls(self) -> int:
        return self._debate_calls

    @property
    def debate_cost(self) -> float:
        return self._debate_cost

    @property
    def cumulative_calls(self) -> int:
        return self._cumulative_calls

    @property
    def cumulative_cost(self) -> float:
        return self._cumulative_cost

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self._model,
            "current_debate_calls": self._debate_calls,
            "current_debate_cost_usd": round(self._debate_cost, 6),
            "cumulative_calls": self._cumulative_calls,
            "cumulative_cost_usd": round(self._cumulative_cost, 6),
        }


# ---------------------------------------------------------------------------
# DebateRound — core engine
# ---------------------------------------------------------------------------

class DebateRound:
    """Orchestrates a structured 2-round debate among players.

    Round 1 -- Challenge: Each player sees all signals, issues ONE challenge.
    Round 2 -- Response: Challenged players defend or concede.
    """

    def __init__(
        self,
        llm: Any,
        player_personalities: Optional[Dict[str, str]] = None,
        max_api_calls: int = 10,
        debate_dir: Optional[Path] = None,
    ):
        self._llm = llm
        self._personalities = player_personalities or dict(_DEFAULT_PERSONALITIES)
        self._max_api_calls = max_api_calls
        self._dir = debate_dir or DEBATE_DIR
        self._cost_tracker = CostTracker(
            model_name=getattr(llm, "model", "gemini-2.5-flash")
        )

    @property
    def cost_tracker(self) -> CostTracker:
        return self._cost_tracker

    # -- Public API --

    def run_debate(
        self,
        signals: Dict[str, DebateSignal],
        symbol: Optional[str] = None,
    ) -> DebateOutcome:
        """Run a full 2-round debate.

        Args:
            signals: Dict mapping player_id -> DebateSignal.
            symbol: Optional symbol override (inferred from signals if not given).

        Returns:
            DebateOutcome with pre/post signals, challenges, responses, analytics.
        """
        now = datetime.now()
        debate_id = f"debate_{now.strftime('%Y%m%d_%H%M%S')}"
        timestamp = now.isoformat()

        if symbol is None:
            first = next(iter(signals.values()), None)
            symbol = first.symbol if first else "UNKNOWN"

        # Populate player labels if missing
        for pid, sig in signals.items():
            if not sig.player_label:
                sig.player_label = PLAYER_LABELS.get(pid, pid)

        # Check skip conditions
        should_skip, skip_reason = self._should_skip(signals)
        if should_skip:
            return DebateOutcome(
                debate_id=debate_id,
                timestamp=timestamp,
                symbol=symbol,
                skipped=True,
                skip_reason=skip_reason,
                pre_signals=dict(signals),
                post_signals=dict(signals),
                consensus_strength=self._compute_consensus(signals),
            )

        # Deep copy pre-signals
        pre_signals = {
            pid: DebateSignal.from_dict(sig.to_dict())
            for pid, sig in signals.items()
        }

        # === ROUND 1: CHALLENGES ===
        challenges = self._run_challenge_round(signals, symbol)

        # === ROUND 2: RESPONSES ===
        challenged_by: Dict[str, ChallengeEntry] = {}
        for ch in challenges:
            if ch.challenged_id not in challenged_by:
                challenged_by[ch.challenged_id] = ch

        responses = self._run_response_round(signals, challenged_by, symbol)

        # === BUILD POST-SIGNALS ===
        post_signals = self._build_post_signals(signals, challenges, responses)

        # Count changes
        signals_changed = 0
        for pid in post_signals:
            if pid in pre_signals:
                pre_dir = pre_signals[pid].direction
                post_dir = post_signals[pid].direction
                if (pre_dir * post_dir < 0) or abs(pre_dir - post_dir) > 0.2:
                    signals_changed += 1

        # Cost tracking
        calls, cost = self._cost_tracker.reset_debate()

        return DebateOutcome(
            debate_id=debate_id,
            timestamp=timestamp,
            symbol=symbol,
            skipped=False,
            pre_signals=pre_signals,
            post_signals=post_signals,
            challenges=challenges,
            responses=responses,
            signals_changed=signals_changed,
            consensus_strength=self._compute_consensus(post_signals),
            debate_summary=self._build_summary(
                pre_signals, post_signals, challenges, responses
            ),
            api_calls_made=calls,
            estimated_cost_usd=cost,
        )

    # -- Skip logic --

    def _should_skip(
        self, signals: Dict[str, DebateSignal]
    ) -> Tuple[bool, str]:
        """Determine if debate should be skipped."""
        if len(signals) < 2:
            return True, "fewer_than_2_signals"

        directions = [s.direction for s in signals.values()]
        confidences = [s.confidence for s in signals.values()]

        all_bullish = all(d > 0.0 for d in directions)
        all_bearish = all(d < 0.0 for d in directions)
        if all_bullish or all_bearish:
            return True, "all_agree_on_direction"

        if all(c < 0.3 for c in confidences):
            return True, "all_low_confidence"

        return False, ""

    # -- Round 1: Challenges --

    def _run_challenge_round(
        self,
        signals: Dict[str, DebateSignal],
        symbol: str,
    ) -> List[ChallengeEntry]:
        """Round 1: Each player issues ONE challenge."""
        challenges: List[ChallengeEntry] = []
        signals_block = _format_signals_block(signals)

        for pid in PLAYER_IDS:
            if pid not in signals:
                continue
            if self._cost_tracker.debate_calls >= self._max_api_calls:
                logger.warning(
                    f"[Debate] API call cap reached ({self._max_api_calls}), "
                    f"stopping challenge round."
                )
                break

            sig = signals[pid]
            label = sig.player_label
            personality = self._personalities.get(label, "General trader.")

            prompt = _CHALLENGE_PROMPT.format(
                player_name=label,
                player_personality=personality,
                symbol=symbol,
                all_signals_formatted=signals_block,
                own_direction_label=sig.direction_label(),
                own_direction=sig.direction,
                own_confidence=sig.confidence,
                own_reasoning=sig.reasoning or "No reasoning provided",
            )

            parsed = self._call_llm_json(prompt)

            if "parse_error" in parsed:
                logger.warning(
                    f"[Debate] Challenge parse error for {pid}: "
                    f"{parsed.get('parse_error')}"
                )
                continue

            challenged_pid = str(parsed.get("challenged_player", ""))
            if challenged_pid not in signals or challenged_pid == pid:
                logger.warning(
                    f"[Debate] {pid} challenged invalid target "
                    f"'{challenged_pid}', skipping."
                )
                continue

            challenged_label = signals[challenged_pid].player_label

            stance = str(parsed.get("stance", "maintain")).lower()
            if stance not in ("maintain", "soften", "reverse"):
                stance = "maintain"

            entry = ChallengeEntry(
                challenger_id=pid,
                challenger_label=label,
                challenged_id=challenged_pid,
                challenged_label=challenged_label,
                challenge_text=str(parsed.get("challenge", "")),
                revised_direction=max(
                    -1.0,
                    min(1.0, float(parsed.get("revised_direction", sig.direction))),
                ),
                revised_confidence=max(
                    0.0,
                    min(1.0, float(parsed.get("revised_confidence", sig.confidence))),
                ),
                stance=stance,
                raw_llm_response=parsed,
            )
            challenges.append(entry)

        return challenges

    # -- Round 2: Responses --

    def _run_response_round(
        self,
        signals: Dict[str, DebateSignal],
        challenged_by: Dict[str, ChallengeEntry],
        symbol: str,
    ) -> List[ResponseEntry]:
        """Round 2: Challenged players respond."""
        responses: List[ResponseEntry] = []
        signals_block = _format_signals_block(signals)

        for challenged_pid, challenge in challenged_by.items():
            if self._cost_tracker.debate_calls >= self._max_api_calls:
                logger.warning(
                    f"[Debate] API call cap reached ({self._max_api_calls}), "
                    f"stopping response round."
                )
                break

            sig = signals[challenged_pid]
            label = sig.player_label
            personality = self._personalities.get(label, "General trader.")
            challenger_label = challenge.challenger_label
            challenger_personality_short = self._personalities.get(
                challenger_label, ""
            ).split(".")[0]

            prompt = _RESPONSE_PROMPT.format(
                player_name=label,
                player_personality=personality,
                symbol=symbol,
                all_signals_formatted=signals_block,
                own_direction_label=sig.direction_label(),
                own_direction=sig.direction,
                own_confidence=sig.confidence,
                own_reasoning=sig.reasoning or "No reasoning provided",
                challenger_name=challenger_label,
                challenger_personality_short=challenger_personality_short,
                challenge_text=challenge.challenge_text,
            )

            parsed = self._call_llm_json(prompt)

            if "parse_error" in parsed:
                logger.warning(
                    f"[Debate] Response parse error for {challenged_pid}: "
                    f"{parsed.get('parse_error')}"
                )
                continue

            entry = ResponseEntry(
                responder_id=challenged_pid,
                responder_label=label,
                challenger_id=challenge.challenger_id,
                response_text=str(parsed.get("response", "")),
                final_direction=max(
                    -1.0,
                    min(1.0, float(parsed.get("final_direction", sig.direction))),
                ),
                final_confidence=max(
                    0.0,
                    min(1.0, float(parsed.get("final_confidence", sig.confidence))),
                ),
                changed=bool(parsed.get("changed", False)),
                raw_llm_response=parsed,
            )
            responses.append(entry)

        return responses

    # -- Post-signal building --

    def _build_post_signals(
        self,
        original_signals: Dict[str, DebateSignal],
        challenges: List[ChallengeEntry],
        responses: List[ResponseEntry],
    ) -> Dict[str, DebateSignal]:
        """Build post-debate signals.

        Priority: response final > challenge revision > original.
        """
        post: Dict[str, DebateSignal] = {}

        # Start with originals
        for pid, sig in original_signals.items():
            post[pid] = DebateSignal.from_dict(sig.to_dict())

        # Apply challenger revisions
        for ch in challenges:
            cid = ch.challenger_id
            if cid in post:
                post[cid].direction = ch.revised_direction
                post[cid].confidence = ch.revised_confidence

        # Apply responder finals (overrides challenger revision if same player)
        for resp in responses:
            rid = resp.responder_id
            if rid in post:
                post[rid].direction = resp.final_direction
                post[rid].confidence = resp.final_confidence

        return post

    # -- Metrics --

    def _compute_consensus(self, signals: Dict[str, DebateSignal]) -> float:
        """Compute consensus strength [0.0, 1.0]."""
        if not signals:
            return 0.0
        directions = [s.direction for s in signals.values()]
        confidences = [s.confidence for s in signals.values()]

        total_weight = sum(confidences) or 1.0
        weighted_dir = sum(
            d * c for d, c in zip(directions, confidences)
        ) / total_weight

        avg_conf = sum(confidences) / len(confidences)
        return min(1.0, abs(weighted_dir) * avg_conf)

    def _build_summary(
        self,
        pre: Dict[str, DebateSignal],
        post: Dict[str, DebateSignal],
        challenges: List[ChallengeEntry],
        responses: List[ResponseEntry],
    ) -> str:
        """Build a human-readable debate summary."""
        lines: List[str] = []
        lines.append(
            f"Debate: {len(challenges)} challenges, {len(responses)} responses."
        )

        changed: List[str] = []
        for pid in post:
            if pid in pre:
                pre_dir = pre[pid].direction
                post_dir = post[pid].direction
                if (pre_dir * post_dir < 0) or abs(pre_dir - post_dir) > 0.2:
                    label = post[pid].player_label
                    changed.append(f"{label}: {pre_dir:+.2f} -> {post_dir:+.2f}")

        if changed:
            lines.append(f"Signal shifts: {', '.join(changed)}.")
        else:
            lines.append("No significant signal changes.")

        concessions = [r for r in responses if r.changed]
        if concessions:
            names = [r.responder_label for r in concessions]
            lines.append(f"Conceded: {', '.join(names)}.")

        return " ".join(lines)

    # -- LLM JSON call --

    def _call_llm_json(self, prompt: str) -> Dict[str, Any]:
        """Call LLM and parse JSON from response.

        Replicates pattern from player_reflection.py _call_llm_json.
        """
        try:
            raw = self._llm.generate(prompt)
        except Exception as e:
            return {"raw_response": str(e), "parse_error": str(e)}

        # Record cost
        self._cost_tracker.record_call(prompt, raw)

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

        # Find outermost { ... }
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
                text = text[brace_start: i + 1]
                break

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"[Debate] JSON parse error: {e}")
            return {"raw_response": raw, "parse_error": str(e)}


# ---------------------------------------------------------------------------
# DebateStore — persistence
# ---------------------------------------------------------------------------

class DebateStore:
    """JSON persistence for debate transcripts."""

    _MAX_STORED = 100

    def __init__(self, debate_dir: Optional[Path] = None):
        self._dir = debate_dir or DEBATE_DIR

    def save_debate(self, outcome: DebateOutcome) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / "debate_history.json"

        existing = self._load_raw(path)
        debates = existing.get("debates", [])
        debates.append(outcome.to_dict())
        debates = debates[-self._MAX_STORED:]

        try:
            with open(path, "w") as f:
                json.dump({"debates": debates}, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[DebateStore] Failed to save: {e}")

    def get_debates(self, last_n: int = 10) -> List[DebateOutcome]:
        path = self._dir / "debate_history.json"
        raw = self._load_raw(path)
        debates = raw.get("debates", [])[-last_n:]
        return [DebateOutcome.from_dict(d) for d in debates]

    def get_all_debates(self) -> List[DebateOutcome]:
        path = self._dir / "debate_history.json"
        raw = self._load_raw(path)
        return [DebateOutcome.from_dict(d) for d in raw.get("debates", [])]

    def _load_raw(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[DebateStore] Corrupt file {path}: {e}")
            return {}


# ---------------------------------------------------------------------------
# DebateAnalytics
# ---------------------------------------------------------------------------

class DebateAnalytics:
    """Tracks debate effectiveness metrics over time."""

    def __init__(self, store: DebateStore):
        self._store = store

    def mind_changers(self, last_n: int = 50) -> Dict[str, int]:
        """Count how many times each player changed signal after being challenged."""
        debates = self._store.get_debates(last_n=last_n)
        counts: Dict[str, int] = {pid: 0 for pid in PLAYER_IDS}
        for debate in debates:
            if debate.skipped:
                continue
            for resp in debate.responses:
                if resp.changed:
                    counts[resp.responder_id] = (
                        counts.get(resp.responder_id, 0) + 1
                    )
        return counts

    def impactful_challengers(self, last_n: int = 50) -> Dict[str, int]:
        """Count how many times each player's challenge caused a concession."""
        debates = self._store.get_debates(last_n=last_n)
        counts: Dict[str, int] = {pid: 0 for pid in PLAYER_IDS}
        for debate in debates:
            if debate.skipped:
                continue
            for resp in debate.responses:
                if resp.changed:
                    counts[resp.challenger_id] = (
                        counts.get(resp.challenger_id, 0) + 1
                    )
        return counts

    def challenge_frequency(self, last_n: int = 50) -> Dict[str, Dict[str, int]]:
        """Who challenges whom most often."""
        debates = self._store.get_debates(last_n=last_n)
        freq: Dict[str, Dict[str, int]] = {}
        for debate in debates:
            for ch in debate.challenges:
                if ch.challenger_id not in freq:
                    freq[ch.challenger_id] = {}
                target = ch.challenged_id
                freq[ch.challenger_id][target] = (
                    freq[ch.challenger_id].get(target, 0) + 1
                )
        return freq

    def average_signal_shift(self, last_n: int = 50) -> float:
        """Average absolute direction change across all debates."""
        debates = self._store.get_debates(last_n=last_n)
        shifts: List[float] = []
        for debate in debates:
            if debate.skipped:
                continue
            for pid in debate.post_signals:
                if pid in debate.pre_signals:
                    pre_d = debate.pre_signals[pid].direction
                    post_d = debate.post_signals[pid].direction
                    shifts.append(abs(post_d - pre_d))
        return sum(shifts) / len(shifts) if shifts else 0.0

    def skip_rate(self, last_n: int = 50) -> float:
        """Fraction of debates that were skipped."""
        debates = self._store.get_debates(last_n=last_n)
        if not debates:
            return 0.0
        skipped = sum(1 for d in debates if d.skipped)
        return skipped / len(debates)

    def cost_summary(self, last_n: int = 50) -> Dict[str, float]:
        """Aggregate cost across recent debates."""
        debates = self._store.get_debates(last_n=last_n)
        total_cost = sum(d.estimated_cost_usd for d in debates)
        total_calls = sum(d.api_calls_made for d in debates)
        active_debates = sum(1 for d in debates if not d.skipped)
        return {
            "total_debates": len(debates),
            "active_debates": active_debates,
            "total_api_calls": total_calls,
            "total_cost_usd": round(total_cost, 6),
            "avg_cost_per_debate_usd": (
                round(total_cost / active_debates, 6) if active_debates else 0.0
            ),
        }

    def record_outcome(
        self,
        debate_id: str,
        player_id: str,
        pre_direction: float,
        post_direction: float,
        actual_pnl: float,
    ) -> None:
        """Record actual trade outcome for pre vs post debate comparison."""
        self._store._dir.mkdir(parents=True, exist_ok=True)
        path = self._store._dir / "debate_performance.jsonl"
        record = {
            "debate_id": debate_id,
            "player_id": player_id,
            "pre_direction": pre_direction,
            "post_direction": post_direction,
            "actual_pnl": actual_pnl,
            "pre_correct": (
                (pre_direction > 0 and actual_pnl > 0)
                or (pre_direction < 0 and actual_pnl < 0)
            ),
            "post_correct": (
                (post_direction > 0 and actual_pnl > 0)
                or (post_direction < 0 and actual_pnl < 0)
            ),
            "timestamp": datetime.now().isoformat(),
        }
        try:
            with open(path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            logger.warning(f"[DebateAnalytics] Failed to record outcome: {e}")

    def debate_accuracy_lift(self) -> Optional[float]:
        """Compare pre-debate vs post-debate signal accuracy.

        Returns percentage point improvement, or None if no data.
        """
        path = self._store._dir / "debate_performance.jsonl"
        if not path.exists():
            return None

        pre_correct = 0
        post_correct = 0
        total = 0
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    total += 1
                    if record.get("pre_correct"):
                        pre_correct += 1
                    if record.get("post_correct"):
                        post_correct += 1
        except Exception:
            return None

        if total == 0:
            return None

        pre_rate = pre_correct / total
        post_rate = post_correct / total
        return (post_rate - pre_rate) * 100

    def summary(self, last_n: int = 50) -> Dict[str, Any]:
        """Full analytics summary."""
        return {
            "mind_changers": self.mind_changers(last_n),
            "impactful_challengers": self.impactful_challengers(last_n),
            "average_signal_shift": round(self.average_signal_shift(last_n), 4),
            "skip_rate": round(self.skip_rate(last_n), 2),
            "cost": self.cost_summary(last_n),
            "accuracy_lift_pct": self.debate_accuracy_lift(),
        }


# ---------------------------------------------------------------------------
# Factory and convenience functions
# ---------------------------------------------------------------------------

def create_debate_system(
    llm: Any,
    debate_dir: Optional[Path] = None,
    max_api_calls: int = 10,
    player_personalities: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Factory to create all debate components.

    Returns dict with keys: debate_round, store, analytics, cost_tracker.
    """
    d = debate_dir or DEBATE_DIR
    store = DebateStore(debate_dir=d)
    debate_round = DebateRound(
        llm=llm,
        player_personalities=player_personalities,
        max_api_calls=max_api_calls,
        debate_dir=d,
    )
    analytics = DebateAnalytics(store=store)

    return {
        "debate_round": debate_round,
        "store": store,
        "analytics": analytics,
        "cost_tracker": debate_round.cost_tracker,
    }


def debate_signals(
    signals: Dict[str, Dict[str, Any]],
    llm: Any,
    debate_dir: Optional[Path] = None,
) -> Tuple[Dict[str, DebateSignal], Optional[DebateOutcome]]:
    """Convenience: run debate and return revised signals.

    Args:
        signals: {player_id: {symbol, direction, confidence, reasoning, indicators}}
        llm: LLMProvider (Flash model recommended).

    Returns:
        (revised_signals_dict, debate_outcome)
    """
    debate_map: Dict[str, DebateSignal] = {}
    for pid, raw in signals.items():
        label = PLAYER_LABELS.get(pid, pid)
        debate_map[pid] = DebateSignal(
            player_id=pid,
            player_label=label,
            symbol=raw.get("symbol", "UNKNOWN"),
            direction=raw.get("direction", 0.0),
            confidence=raw.get("confidence", 0.5),
            reasoning=raw.get("reasoning", ""),
            indicators=raw.get("indicators", {}),
        )

    system = create_debate_system(llm=llm, debate_dir=debate_dir)
    debate_round = system["debate_round"]
    store = system["store"]

    outcome = debate_round.run_debate(debate_map)

    if not outcome.skipped:
        store.save_debate(outcome)

    return outcome.post_signals, outcome
