"""
Gemini Exit Reasoning — LLM adjudication for ambiguous exit decisions.

Sits on top of the rule-based SmartExitEngine.  When the engine's weighted
score falls in an ambiguous zone (0.30–0.70 by default), Gemini 2.5 Flash
is called to make a holistic EXIT vs HOLD decision.  Clear-cut cases
(score < 0.30 or > 0.70) bypass Gemini entirely — zero cost.

Key design choices
==================
- response_mime_type="application/json" forces Gemini to return valid JSON
  (no markdown fences to strip — cleaner than the coach implementation)
- system_instruction separates role context from per-position data
- Exponential backoff retry (max 2 retries) on API/parse errors
- Audit log: every evaluation → logs/exit_decisions.jsonl
- Token usage tracked from response.usage_metadata (real counts, not estimates)

Cost Control
============
- Only called in the ambiguous zone (~30-40% of positions on any given day)
- Per-ticker-per-day caching prevents duplicate calls
- Hard daily_call_budget (default 50) enforced
- Any API failure → silent fallback to rule-based decision
- Expected cost: ~$0.01/day at 50 calls with Gemini Flash pricing

Integration
===========
    from claude_exit_reasoning import GeminiExitReasoner

    reasoner = GeminiExitReasoner(api_key=os.getenv("GEMINI_API_KEY"))
    result = reasoner.evaluate_with_reasoning(
        position=pos,
        market_data={"atr": 35.0, "current_volume": 1e6, "avg_volume_20d": 8e5, ...},
        player_signals=[{"player_id": "PLAYER_1", "direction": 0.6, "confidence": 0.7}, ...],
        regime="bull",
        rule_decision=exit_decision,  # from SmartExitEngine
        portfolio_context={"equity": 504000, "cash": 242000, "n_positions": 15},
        fii_dii_data=None,
    )
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

_CHARS_PER_TOKEN = 4  # same heuristic used throughout the codebase

# Gemini Flash pricing (per 1M tokens)
GEMINI_COST_PER_MILLION = {
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "default": {"input": 0.15, "output": 0.60},
}

# ── System Instruction (role context — sent once via system_instruction) ──

EXIT_SYSTEM_INSTRUCTION = """You are the Exit Decision Engine for a 5-player AI trading system on NIFTY 50 stocks. Your ONLY job: decide EXIT, HOLD, or PARTIAL EXIT.

Architecture: 5 players (TrendFollower, MeanReversion, VolumeFlow, VolBreakout, MultiMomentum) → Coach synthesis → Rule-based SmartExitEngine → YOU adjudicate ambiguous cases.

The SmartExitEngine evaluates 9 independent exit signals:
1. Alpha Decay — how much of the entry signal has been consumed
2. Profit Target — ATR-normalized distance to take-profit level
3. Trailing Stop — bar LOW (longs) or HIGH (shorts) vs ratcheting stop
4. Time Decay — proximity to maximum holding period
5. Agreement Decay — whether player consensus has eroded since entry
6. Drawdown from Peak — how much of best unrealized profit was given back
7. Volatility Shift — realized vol spike vs entry vol
8. Volume Climax — abnormally high volume signaling exhaustion
9. Event Calendar — proximity to scheduled high-impact events (RBI MPC, Union Budget, F&O expiry, US Fed FOMC, US CPI/NFP, earnings season)

Decision rules:
- BEAR regime: aggressive profit taking, tight stops, don't let winners become losers
- BULL regime: let winners run longer, wider trailing stops, but protect against giveback > 40% of peak
- Multiple signals firing simultaneously (score > 0.5) = convergent evidence → EXIT
- Rule composite > 0.7 → almost always agree to exit
- Rule composite < 0.3 → almost always agree to hold
- Agreement decay + high volatility + drawdown = GET OUT regardless of P&L
- Never hold when entry thesis is broken (alpha_decay near 1.0)

India-specific event intelligence:
- RBI MPC at 10 AM IST → tighten trailing stops the day before
- Union Budget Feb 1 → close or reduce ALL positions — single-day 5%+ moves happen
- Monthly F&O expiry (last Thursday) → institutional gamma hedging = whipsaws
- US Fed FOMC → impacts Indian market NEXT MORNING. If profitable going into Fed night, take at least Tranche 1
- US CPI/NFP → can shift FII flow sentiment overnight

Think HOLISTICALLY: a position with high profit_target but low alpha_decay and low drawdown may be worth holding. Moderate signals across many dimensions may warrant exit even if no single signal is critical.

Consider the P&L trajectory (MFE vs current P&L), regime context, per-player breakdown, portfolio-level exposure, and upcoming scheduled events when making your decision."""


# ═══════════════════════════════════════════════════════════════════════════════
# Cost Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiCostTracker:
    """Estimates and tracks Gemini API cost."""

    def __init__(self, model_name: str = "gemini-2.5-flash") -> None:
        self._model = model_name
        self._pricing = GEMINI_COST_PER_MILLION.get(
            model_name, GEMINI_COST_PER_MILLION["default"]
        )
        self._total_calls: int = 0
        self._total_cost: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    def record_call(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Record a call with real token counts and return estimated cost."""
        cost = (
            (input_tokens / 1_000_000) * self._pricing["input"]
            + (output_tokens / 1_000_000) * self._pricing["output"]
        )
        self._total_calls += 1
        self._total_cost += cost
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        return cost

    def record_call_from_chars(self, prompt: str, response: str) -> float:
        """Fallback: estimate tokens from character count."""
        input_tokens = len(prompt) // _CHARS_PER_TOKEN
        output_tokens = len(response) // _CHARS_PER_TOKEN
        return self.record_call(input_tokens, output_tokens)

    def summary(self) -> Dict[str, Any]:
        return {
            "model": self._model,
            "total_calls": self._total_calls,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_cost_usd": round(self._total_cost, 6),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Gemini Exit Reasoner (v2 — full rewrite)
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiExitReasoner:
    """LLM reasoning layer for ambiguous exit decisions.

    Only calls Gemini when the SmartExitEngine's weighted_score falls in
    the ambiguous zone.  All other cases use rule-based logic directly.

    Parameters
    ----------
    api_key : str, optional
        Gemini API key.  Falls back to GEMINI_API_KEY env var.
    model : str
        Gemini model to use.
    ambiguous_zone : tuple
        (low, high) bounds.  Scores outside this range skip Gemini.
    daily_call_budget : int
        Maximum Gemini API calls per calendar day.
    log_dir : str
        Directory for audit log (exit_decisions.jsonl).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        ambiguous_zone: Tuple[float, float] = (0.30, 0.70),
        daily_call_budget: int = 50,
        log_dir: str = "logs",
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._model = model
        self._ambiguous_low, self._ambiguous_high = ambiguous_zone
        self._daily_call_budget = daily_call_budget
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Lazy-initialized client
        self._client: Optional[Any] = None

        # Daily state
        self._calls_today: int = 0
        self._date: date = date.today()
        self._cache: Dict[str, Dict[str, Any]] = {}  # "{symbol}_{date}" → result

        # Cost tracking
        self._cost_tracker = GeminiCostTracker(model)

        # Audit log: every evaluation (AI or rule-based) stored here
        self._decision_log: List[Dict[str, Any]] = []

        # Daily stats
        self._stats = self._fresh_stats()

        logger.info(
            f"[GeminiExit] Initialised (model={model}, "
            f"zone={ambiguous_zone}, budget={daily_call_budget}/day)"
        )

    def _fresh_stats(self) -> Dict[str, int]:
        return {
            "total_evaluations": 0,
            "ai_calls": 0,
            "rule_only": 0,
            "exits_triggered": 0,
            "holds_triggered": 0,
            "ai_overrides": 0,
            "cache_hits": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }

    # ── Client management ──────────────────────────────────────────────

    def _ensure_client(self) -> None:
        """Lazy-initialise the Gemini client."""
        if self._client is not None:
            return
        if not self._api_key:
            logger.warning("[GeminiExit] No API key — will use rule-based fallback")
            return
        try:
            from google import genai
            self._client = genai.Client(api_key=self._api_key)
            logger.info("[GeminiExit] Gemini client initialised")
        except ImportError:
            logger.warning("[GeminiExit] google-genai package not installed")
        except Exception as e:
            logger.warning(f"[GeminiExit] Client init failed: {e}")

    def _reset_if_new_day(self) -> None:
        """Reset daily counters and cache at the start of a new day."""
        today = date.today()
        if today != self._date:
            if self._calls_today > 0:
                summary = self._cost_tracker.summary()
                logger.info(
                    f"[GeminiExit] Day ended: {self._calls_today} calls, "
                    f"${summary['total_cost_usd']:.4f} est. cost"
                )
            # Save yesterday's audit log before resetting
            self.save_decision_log()
            self._calls_today = 0
            self._cache.clear()
            self._date = today
            self._decision_log.clear()
            self._stats = self._fresh_stats()

    # ── Public API: main entry point ──────────────────────────────────

    def evaluate_with_reasoning(
        self,
        position: Any,
        market_data: Dict[str, Any],
        player_signals: List[Dict[str, Any]],
        regime: str,
        rule_decision: Any,
        portfolio_context: Optional[Dict[str, Any]] = None,
        fii_dii_data: Optional[Dict[str, Any]] = None,
        event_context: Optional[str] = None,
        mfe_mae_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate whether to exit, using Gemini for ambiguous cases.

        Accepts the ALREADY-COMPUTED rule_decision from SmartExitEngine.
        Checks if weighted_score is in ambiguous zone. If yes and budget
        allows → call Gemini, return AI decision. Otherwise → format
        rule_decision as standard dict and return.

        Parameters
        ----------
        position : PositionState or Position
            The position being evaluated.
        market_data : dict
            Keys: atr, current_volume, avg_volume_20d, current_price,
            current_low, current_high, garch_vol, entry_vol.
        player_signals : list of dict
            Per-player breakdown: [{player_id, direction, confidence, player_label}, ...].
        regime : str
            Current market regime ("bull", "bear", "sideways").
        rule_decision : ExitDecision
            Pre-computed result from SmartExitEngine.evaluate().
        portfolio_context : dict, optional
            Keys: equity, cash, n_positions, gross_exposure_pct.
        fii_dii_data : dict, optional
            Keys: fii_net, dii_net, fii_selling_streak.
        event_context : str, optional
            Pre-built text block from EventCalendar.get_event_context_for_prompt().
        mfe_mae_context : str, optional
            Pre-built text block from MFEMAEAnalyzer.format_for_gemini_prompt().

        Returns
        -------
        dict
            {should_exit, exit_fraction, reason, urgency, reasoning,
             confidence, ai_consulted: bool, rule_based_score: float,
             override_rule_based: bool, source: "gemini"|"rule_based"}
        """
        try:
            self._reset_if_new_day()
            self._stats["total_evaluations"] += 1

            # Extract weighted_score from rule_decision
            details = getattr(rule_decision, "details", {}) or {}
            weighted_score = float(details.get("weighted_score", 0.0))
            signal_scores = details.get("signal_scores", {})
            exit_threshold = details.get("exit_threshold", 0.55)
            unrealised_pnl_pct = float(details.get("unrealised_pnl_pct", 0.0))
            mfe = float(details.get("mfe", 0.0))
            mae = float(details.get("mae", 0.0))

            # Extract position info
            symbol = getattr(position, "symbol", "UNKNOWN")
            direction = getattr(position, "direction", "LONG")
            entry_price = float(getattr(position, "entry_price", 0.0))
            bars_held = int(getattr(position, "bars_since_entry",
                            getattr(position, "bars_held", 0)))
            remaining_fraction = float(getattr(position, "remaining_fraction", 1.0))
            staged_exits_done = int(getattr(position, "staged_exits_done", 0))
            entry_signal = float(getattr(position, "entry_signal", 0.0))
            entry_agreement = float(getattr(position, "entry_agreement", 0.0))

            current_price = float(market_data.get("current_price", entry_price))
            atr = float(market_data.get("atr", current_price * 0.015))

            # Build the rule-based result (always returned as baseline)
            rule_result = self._format_rule_decision(rule_decision, weighted_score)

            # ── Clear-cut cases: skip Gemini ────────────────────────
            if weighted_score >= self._ambiguous_high:
                result = rule_result
                result["_skip_reason"] = "clear_exit"
                self._stats["rule_only"] += 1
                self._log_evaluation(symbol, result, weighted_score, False)
                return result

            if weighted_score <= self._ambiguous_low:
                result = rule_result
                result["_skip_reason"] = "clear_hold"
                self._stats["rule_only"] += 1
                self._log_evaluation(symbol, result, weighted_score, False)
                return result

            # ── Check cache ─────────────────────────────────────────
            cache_key = f"{symbol}_{self._date.isoformat()}"
            if cache_key in self._cache:
                logger.debug(f"[GeminiExit] Cache hit: {symbol}")
                self._stats["cache_hits"] += 1
                return self._cache[cache_key]

            # ── Budget check ────────────────────────────────────────
            if self._calls_today >= self._daily_call_budget:
                logger.warning(
                    f"[GeminiExit] Daily budget exhausted "
                    f"({self._calls_today}/{self._daily_call_budget})"
                )
                result = rule_result
                result["_skip_reason"] = "budget_exhausted"
                self._stats["rule_only"] += 1
                self._log_evaluation(symbol, result, weighted_score, False)
                return result

            # ── Build prompt and call Gemini ─────────────────────────
            prompt = self._build_prompt(
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                current_price=current_price,
                unrealised_pnl_pct=unrealised_pnl_pct,
                bars_held=bars_held,
                remaining_fraction=remaining_fraction,
                staged_exits_done=staged_exits_done,
                entry_signal=entry_signal,
                entry_agreement=entry_agreement,
                regime=regime,
                signal_scores=signal_scores,
                weighted_score=weighted_score,
                exit_threshold=exit_threshold,
                atr=atr,
                mfe=mfe,
                mae=mae,
                market_data=market_data,
                player_signals=player_signals,
                portfolio_context=portfolio_context,
                fii_dii_data=fii_dii_data,
                event_context=event_context,
                mfe_mae_context=mfe_mae_context,
            )

            parsed = self._call_gemini_with_retry(prompt)

            if parsed is None or "parse_error" in parsed:
                err = parsed.get("parse_error", "unknown") if parsed else "null response"
                logger.warning(f"[GeminiExit] {symbol} API/parse failed: {err}")
                result = rule_result
                result["_skip_reason"] = f"api_error: {err}"
                self._stats["rule_only"] += 1
                self._log_evaluation(symbol, result, weighted_score, False)
                return result

            # ── Build result from Gemini's response ──────────────────
            should_exit = bool(parsed.get("should_exit", False))
            exit_fraction = max(0.0, min(1.0, float(parsed.get("exit_fraction", 0.0))))
            reason = str(parsed.get("reason", "ai_override_hold"))
            urgency = max(0.0, min(1.0, float(parsed.get("urgency", 0.5))))
            reasoning = str(parsed.get("reasoning", ""))
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            override_rule = bool(parsed.get("override_rule_based", False))

            # Detect if Gemini disagrees with rule-based
            rule_should_exit = rule_result["should_exit"]
            is_override = (should_exit != rule_should_exit)
            if is_override:
                self._stats["ai_overrides"] += 1

            result: Dict[str, Any] = {
                "should_exit": should_exit,
                "exit_fraction": exit_fraction if should_exit else 0.0,
                "reason": reason,
                "urgency": urgency,
                "reasoning": reasoning,
                "confidence": confidence,
                "ai_consulted": True,
                "rule_based_score": weighted_score,
                "override_rule_based": is_override,
                "source": "gemini",
            }

            # Update stats
            self._stats["ai_calls"] += 1
            if should_exit:
                self._stats["exits_triggered"] += 1
            else:
                self._stats["holds_triggered"] += 1

            # Cache the result
            self._cache[cache_key] = result

            # Console log
            prefix = "\u26a1OVERRIDE" if is_override else ""
            icon = "\U0001f916 AI"
            action = "EXIT" if should_exit else "HOLD"
            logger.info(
                f"[GeminiExit] {icon} {symbol} {action} "
                f"frac={exit_fraction:.2f} reason={reason} "
                f"urgency={urgency:.2f} conf={confidence:.2f} "
                f"{prefix} "
                f"| calls={self._calls_today}/{self._daily_call_budget}"
            )

            self._log_evaluation(symbol, result, weighted_score, True, signal_scores)
            return result

        except Exception as e:
            # NEVER crash the pipeline
            logger.warning(f"[GeminiExit] Unexpected error for "
                           f"{getattr(position, 'symbol', '?')}: {e}")
            result = self._format_rule_decision(rule_decision, 0.0)
            result["_skip_reason"] = f"exception: {e}"
            return result

    # ── Legacy API (backward compat) ───────────────────────────────────

    def should_exit(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        current_price: float,
        unrealised_pnl_pct: float,
        bars_held: int,
        regime: str,
        signal_scores: Dict[str, float],
        weighted_score: float,
        atr: float,
        mfe: float,
        mae: float,
        exit_threshold: float = 0.55,
    ) -> Dict[str, Any]:
        """Legacy API — wraps evaluate_with_reasoning for backward compat."""

        # Build a minimal mock position
        class _MinPos:
            pass
        pos = _MinPos()
        pos.symbol = symbol
        pos.direction = direction
        pos.entry_price = entry_price
        pos.bars_held = bars_held
        pos.remaining_fraction = 1.0
        pos.staged_exits_done = 0
        pos.entry_signal = 0.0
        pos.entry_agreement = 0.0

        # Build a minimal mock rule_decision
        class _MinDecision:
            pass
        rd = _MinDecision()
        rd.should_exit = weighted_score >= exit_threshold
        rd.exit_fraction = min(1.0, 0.5 + (weighted_score - exit_threshold) * 2.0) if rd.should_exit else 0.0
        rd.reason = type("R", (), {"value": "weighted_vote"})()
        rd.urgency = min(1.0, abs(weighted_score - exit_threshold) / max(exit_threshold, 0.01))
        rd.details = {
            "weighted_score": weighted_score,
            "signal_scores": signal_scores,
            "exit_threshold": exit_threshold,
            "unrealised_pnl_pct": unrealised_pnl_pct,
            "mfe": mfe,
            "mae": mae,
        }

        market_data = {
            "atr": atr,
            "current_price": current_price,
            "current_volume": 0,
            "avg_volume_20d": 0,
        }

        result = self.evaluate_with_reasoning(
            position=pos,
            market_data=market_data,
            player_signals=[],
            regime=regime,
            rule_decision=rd,
        )

        return result

    # ── Internal: Format rule-based decision ───────────────────────────

    def _format_rule_decision(
        self,
        rule_decision: Any,
        weighted_score: float,
    ) -> Dict[str, Any]:
        """Convert ExitDecision into the standard return dict."""
        should_exit = getattr(rule_decision, "should_exit", False)
        exit_fraction = float(getattr(rule_decision, "exit_fraction", 0.0))
        reason_obj = getattr(rule_decision, "reason", None)
        reason_str = getattr(reason_obj, "value", str(reason_obj)) if reason_obj else "weighted_vote"
        urgency = float(getattr(rule_decision, "urgency", 0.5))

        return {
            "should_exit": should_exit,
            "exit_fraction": exit_fraction,
            "reason": reason_str,
            "urgency": urgency,
            "reasoning": f"Rule-based: weighted_score={weighted_score:.3f}",
            "confidence": min(1.0, abs(weighted_score - 0.55) / 0.55),
            "ai_consulted": False,
            "rule_based_score": weighted_score,
            "override_rule_based": False,
            "source": "rule_based",
        }

    # ── Internal: Prompt builder ───────────────────────────────────────

    def _build_prompt(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        current_price: float,
        unrealised_pnl_pct: float,
        bars_held: int,
        remaining_fraction: float,
        staged_exits_done: int,
        entry_signal: float,
        entry_agreement: float,
        regime: str,
        signal_scores: Dict[str, float],
        weighted_score: float,
        exit_threshold: float,
        atr: float,
        mfe: float,
        mae: float,
        market_data: Dict[str, Any],
        player_signals: List[Dict[str, Any]],
        portfolio_context: Optional[Dict[str, Any]] = None,
        fii_dii_data: Optional[Dict[str, Any]] = None,
        event_context: Optional[str] = None,
        mfe_mae_context: Optional[str] = None,
    ) -> str:
        """Build compact prompt to minimize tokens."""

        # P&L context
        pnl_sign = "+" if unrealised_pnl_pct >= 0 else ""
        giveback = mfe - unrealised_pnl_pct if mfe > 0 else 0.0
        giveback_pct = (giveback / mfe * 100) if mfe > 0.001 else 0.0

        # Signal scores as text bar chart
        signal_lines = []
        for name, score in sorted(signal_scores.items()):
            bar = "\u2588" * int(score * 10) + "\u2591" * (10 - int(score * 10))
            signal_lines.append(f"  {name:<22} {bar} {score:.3f}")
        signals_block = "\n".join(signal_lines) if signal_lines else "  (no signal scores)"

        # Dominant signal
        dominant = max(signal_scores, key=signal_scores.get) if signal_scores else "none"
        dominant_score = signal_scores.get(dominant, 0.0)

        # Rule recommendation
        rule_rec = "EXIT" if weighted_score >= exit_threshold else "HOLD"

        # Volume context
        current_volume = market_data.get("current_volume", 0)
        avg_vol_20d = market_data.get("avg_volume_20d", 0)
        vol_ratio_str = ""
        if avg_vol_20d and avg_vol_20d > 0:
            vol_ratio = current_volume / avg_vol_20d
            vol_ratio_str = f"Volume vs 20d avg: {vol_ratio:.2f}x"
        garch_vol = market_data.get("garch_vol", 0)
        entry_vol = market_data.get("entry_vol", 0)

        # Current signal vs entry signal
        current_signal = market_data.get("current_signal", 0.0)
        current_agreement = market_data.get("current_agreement", 0.0)

        # Per-player breakdown
        player_lines = []
        for ps in player_signals:
            pid = ps.get("player_id", "?")
            label = ps.get("player_label", pid)
            pdir = ps.get("direction", 0.0)
            pconf = ps.get("confidence", 0.0)
            player_lines.append(f"  {label:<18} dir={pdir:+.3f}  conf={pconf:.2f}")
        players_block = "\n".join(player_lines) if player_lines else "  (no player data)"

        # Portfolio context
        portfolio_block = ""
        if portfolio_context:
            eq = portfolio_context.get("equity", 0)
            cash = portfolio_context.get("cash", 0)
            n_pos = portfolio_context.get("n_positions", 0)
            gross_exp = portfolio_context.get("gross_exposure_pct", 0)
            portfolio_block = f"""
=== PORTFOLIO ===
Equity: {eq:,.0f} | Cash: {cash:,.0f} | Positions: {n_pos} | Gross exposure: {gross_exp:.1f}%"""

        # FII/DII data
        fii_block = ""
        if fii_dii_data:
            fii_net = fii_dii_data.get("fii_net", 0)
            dii_net = fii_dii_data.get("dii_net", 0)
            streak = fii_dii_data.get("fii_selling_streak", 0)
            fii_block = f"""
=== FII / DII ===
FII net: {fii_net:+,.0f} cr | DII net: {dii_net:+,.0f} cr | FII selling streak: {streak} days"""

        # Event calendar context
        event_block = ""
        if event_context:
            event_block = f"""
=== UPCOMING EVENTS ===
{event_context}"""

        # MFE/MAE historical analysis context
        mfe_block = ""
        if mfe_mae_context:
            mfe_block = f"""
=== EXIT QUALITY STATS (historical) ===
{mfe_mae_context}"""

        prompt = f"""=== POSITION ===
{symbol} | {direction} | Entry: {entry_price:,.2f} → Current: {current_price:,.2f} | P&L: {pnl_sign}{unrealised_pnl_pct:.2%}
Bars held: {bars_held} | Remaining: {remaining_fraction:.0%} | Staged exits done: {staged_exits_done}

=== SIGNALS ===
Entry signal: {entry_signal:.3f} → Current: {current_signal:.3f}
Entry agreement: {entry_agreement:.3f} → Current: {current_agreement:.3f}

=== PER-PLAYER BREAKDOWN ===
{players_block}

=== MARKET ===
Regime: {regime.upper()} | ATR: {atr:.2f}
{vol_ratio_str}
GARCH vol: {garch_vol:.4f} | Entry vol: {entry_vol:.4f}

=== MFE / MAE ===
MFE: {mfe:.4f} ({mfe:.2%}) | MAE: {mae:.4f} ({mae:.2%})
Giveback from peak: {giveback:.4f} ({giveback_pct:.1f}% of MFE)

=== RULE SCORES (0.0 = no exit, 1.0 = strong exit) ===
{signals_block}

Composite: {weighted_score:.4f} | Threshold: {exit_threshold} | Dominant: {dominant} ({dominant_score:.3f})
Rule recommendation: {rule_rec}
Status: AMBIGUOUS ZONE ({self._ambiguous_low:.2f} – {self._ambiguous_high:.2f})
{portfolio_block}{fii_block}{event_block}{mfe_block}

MAKE YOUR EXIT DECISION (JSON only):"""

        return prompt

    # ── Internal: Gemini API call with retry ───────────────────────────

    def _call_gemini_with_retry(
        self,
        prompt: str,
        max_retries: int = 2,
    ) -> Optional[Dict[str, Any]]:
        """Call Gemini API with retry + exponential backoff.

        Uses response_mime_type="application/json" for guaranteed valid JSON.
        Retries on API errors with 1s exponential backoff.
        """
        self._ensure_client()
        if self._client is None:
            return {"parse_error": "Gemini client unavailable"}

        last_error = ""
        for attempt in range(1 + max_retries):
            try:
                from google.genai import types
                config = types.GenerateContentConfig(
                    system_instruction=EXIT_SYSTEM_INSTRUCTION,
                    temperature=0.1,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                )
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=prompt,
                    config=config,
                )
                raw = response.text or ""

                # Track token usage from response metadata
                usage = getattr(response, "usage_metadata", None)
                if usage:
                    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) or 0
                    cost = self._cost_tracker.record_call(input_tokens, output_tokens)
                    self._stats["total_input_tokens"] += input_tokens
                    self._stats["total_output_tokens"] += output_tokens
                else:
                    cost = self._cost_tracker.record_call_from_chars(prompt, raw)

                self._calls_today += 1
                logger.debug(
                    f"[GeminiExit] Call #{self._calls_today}, "
                    f"est. cost=${cost:.5f}"
                )

                # Parse JSON (should already be valid thanks to response_mime_type)
                parsed = self._parse_response(raw)
                if "parse_error" not in parsed:
                    return parsed

                # JSON parse failed even with mime type — retry
                last_error = parsed.get("parse_error", "unknown parse error")
                logger.warning(
                    f"[GeminiExit] Parse failed attempt {attempt + 1}: {last_error}"
                )

            except Exception as e:
                last_error = str(e)
                logger.warning(
                    f"[GeminiExit] API error attempt {attempt + 1}: {e}"
                )

            # Exponential backoff before retry
            if attempt < max_retries:
                backoff = 1.0 * (2 ** attempt)
                time.sleep(backoff)

        return {"parse_error": f"All {1 + max_retries} attempts failed: {last_error}"}

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse JSON from Gemini's response.

        With response_mime_type="application/json", the response should
        already be valid JSON. We still have a safety net that strips
        fences and finds outermost braces.
        """
        text = raw.strip()

        # First try direct parse (most common with response_mime_type)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: strip markdown fences
        if "```json" in text:
            try:
                start = text.index("```json") + 7
                end = text.index("```", start)
                text = text[start:end].strip()
            except ValueError:
                pass
        elif "```" in text:
            try:
                start = text.index("```") + 3
                end = text.index("```", start)
                text = text[start:end].strip()
            except ValueError:
                pass

        # Find outermost { ... }
        brace_start = text.find("{")
        if brace_start == -1:
            return {"raw_response": raw, "parse_error": "No JSON object found"}

        depth = 0
        brace_end = len(text) - 1
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            if depth == 0:
                brace_end = i
                break

        json_str = text[brace_start:brace_end + 1]

        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"[GeminiExit] JSON parse error: {e}")
            return {"raw_response": raw, "parse_error": str(e)}

    # ── Audit logging ─────────────────────────────────────────────────

    def _log_evaluation(
        self,
        symbol: str,
        result: Dict[str, Any],
        weighted_score: float,
        ai_consulted: bool,
        signal_scores: Optional[Dict[str, float]] = None,
    ) -> None:
        """Log an evaluation to the decision log."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "weighted_score": round(weighted_score, 4),
            "ai_consulted": ai_consulted,
            "should_exit": result.get("should_exit", False),
            "exit_fraction": round(result.get("exit_fraction", 0.0), 4),
            "reason": result.get("reason", ""),
            "urgency": round(result.get("urgency", 0.0), 3),
            "confidence": round(result.get("confidence", 0.0), 3),
            "source": result.get("source", "rule_based"),
            "override": result.get("override_rule_based", False),
        }
        if signal_scores:
            entry["signal_scores"] = {k: round(v, 3) for k, v in signal_scores.items()}
        self._decision_log.append(entry)

    def save_decision_log(self, filepath: Optional[str] = None) -> None:
        """Append today's decision log to JSONL file."""
        if not self._decision_log:
            return
        path = Path(filepath) if filepath else self._log_dir / "exit_decisions.jsonl"
        try:
            with open(path, "a") as f:
                for entry in self._decision_log:
                    f.write(json.dumps(entry) + "\n")
            logger.debug(
                f"[GeminiExit] Saved {len(self._decision_log)} decisions to {path}"
            )
        except Exception as e:
            logger.warning(f"[GeminiExit] Failed to save decision log: {e}")

    # ── Diagnostics ────────────────────────────────────────────────────

    def get_daily_stats(self) -> Dict[str, Any]:
        """Return daily usage and decision stats."""
        cost = self._cost_tracker.summary()
        return {
            "date": self._date.isoformat(),
            "calls_today": self._calls_today,
            "daily_budget": self._daily_call_budget,
            "cache_size": len(self._cache),
            "cost": cost,
            **self._stats,
            "estimated_cost_usd": cost["total_cost_usd"],
        }

    def get_summary(self) -> Dict[str, Any]:
        """Alias for backward compat."""
        return self.get_daily_stats()


# ── Backward compatibility aliases ────────────────────────────────────────
ClaudeExitReasoner = GeminiExitReasoner
ClaudeCostTracker = GeminiCostTracker
