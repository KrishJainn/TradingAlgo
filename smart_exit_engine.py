"""
Smart Exit Engine — Multi-Signal Weighted Voting Exit System.

Replaces the simple trailing-stop exit logic with an institutional-grade
exit framework that evaluates 8 independent exit signals, weights them
by regime, and combines via a voting mechanism.

Mathematical Foundations
========================

1. **Alpha Decay** — Measures how much of the entry signal (alpha) has been
   consumed since entry.  Modelled as exponential consumption:

       α(t) = α₀ × e^(−λt)

   where λ is regime-dependent (higher in Bear = faster decay expectation).
   When current_signal / entry_signal falls below a threshold, the alpha
   that justified the position no longer exists → exit.

2. **Profit Target** — Uses ATR-normalised distance to avoid fixed-₹ or
   fixed-% targets that ignore volatility:

       target = entry ± k × ATR      (k varies by regime & signal strength)

   A position reaching 3× ATR profit in Bull is "statistically extended";
   in Bear the bar is lower (1.2× ATR) because mean-reversion is faster.

3. **Trailing Stop (LOW-based)** — Classic ratcheting stop, but checked
   against the bar's LOW price, not the close.  This catches intraday
   violations that close-only stops miss:

       if low ≤ peak_price × (1 − stop%)  →  exit

4. **Staged Exits (Kelly-inspired)** — Optimal position management says
   exit in tranches: 1/3 at Tranche 1, 1/3 at Tranche 2, let the final
   1/3 ride with a trailing stop.  This balances capture-ratio vs.
   letting winners run.  MFE capture target: 0.50-0.60 (institutional).

5. **MFE/MAE Tracking** — Maximum Favourable Excursion (best unrealised
   P&L) and Maximum Adverse Excursion (worst drawdown) are tracked per
   position to enable drawdown-from-peak and capture-ratio analytics.

6. **Regime Transition** — Uses the HMM transition matrix to assess
   regime instability.  If P(stay in current regime) is declining and/or
   P(adverse regime) is rising, the exit score increases.  Combines:

       instability × 0.5 + p_adverse × 0.3 + regime_mismatch × 0.2

   where regime_mismatch = 0.3 if current regime ≠ entry regime.

7. **Weighted Voting** — Each of 10 signals returns a score ∈ [0, 1].
   A regime-dependent weight vector W is applied:

       exit_score = Σ(signal_i × W_i)

   If exit_score ≥ EXIT_THRESHOLD (0.55), the position exits.  The
   dominant signal (highest weighted contribution) becomes the ExitReason.

Usage
=====
    from smart_exit_engine import SmartExitEngine, PositionState

    engine = SmartExitEngine()
    decision = engine.evaluate(
        position=pos,
        current_price=1500.0,
        current_low=1492.0,
        current_volume=2_500_000,
        atr=28.5,
        current_signal=0.35,
        current_agreement=0.6,
        current_vol_forecast=0.22,
        regime="bull",
    )

    if decision.should_exit:
        if decision.exit_fraction < 1.0:
            # partial exit (staged profit taking)
            ...
        else:
            # full exit
            ...
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════

class ExitReason(Enum):
    """All possible reasons for exiting a position.

    Each reason maps to exactly one of the 8 exit signals evaluated by
    the SmartExitEngine, plus two special reasons for staged profit taking
    and hard risk limits.
    """
    ALPHA_DECAY = "alpha_decay"
    PROFIT_TARGET = "profit_target"
    TRAILING_STOP = "trailing_stop"
    TIME_DECAY = "time_decay"
    AGREEMENT_DECAY = "agreement_decay"
    DRAWDOWN_FROM_PEAK = "drawdown_from_peak"
    VOLATILITY_SPIKE = "volatility_spike"
    VOLUME_CLIMAX = "volume_climax"
    EVENT_RISK = "event_risk"
    REGIME_TRANSITION = "regime_transition"
    STAGED_PROFIT = "staged_profit"
    RISK_LIMIT = "risk_limit"
    BAYESIAN_EXIT = "bayesian_exit"


# ═══════════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExitDecision:
    """Result of the SmartExitEngine evaluation for a single position.

    Attributes:
        should_exit:    Whether the position should be exited (fully or partially).
        exit_fraction:  Fraction of remaining position to exit (0.0–1.0).
                        1.0 = full exit, 0.33 = one tranche of staged exit.
        reason:         The dominant signal that triggered the exit.
        urgency:        How urgent the exit is (0.0 = low, 1.0 = immediate).
                        Hard stops always get urgency=1.0.
        details:        Dict of all signal scores and metadata for logging.
    """
    should_exit: bool
    exit_fraction: float
    reason: ExitReason
    urgency: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionState:
    """Extended position state for the SmartExitEngine.

    Extends the basic position tracking (symbol, direction, entry price)
    with fields needed by the exit signals: signal history, agreement
    history, volume history, MFE/MAE, and staged-exit bookkeeping.

    Attributes:
        symbol:                 Ticker symbol (e.g. "RELIANCE.NS").
        direction:              "LONG" or "SHORT".
        entry_price:            Price at which the position was opened.
        entry_date:             ISO date string of entry.
        entry_signal:           Signal strength at entry (e.g. 0.7 = strong bullish).
        entry_agreement:        Player agreement at entry (0–1, where 1 = unanimous).
        entry_volatility:       GARCH vol forecast at entry (annualised, e.g. 0.20).
        entry_regime:           Market regime at entry ("bull", "bear", "sideways").
        remaining_fraction:     Fraction of original position still open (1.0 at start).
        staged_exits_done:      Number of staged profit-taking tranches completed (0–3).
        max_favorable_excursion: Best unrealised P&L % since entry (MFE).
        max_adverse_excursion:  Worst unrealised P&L % since entry (MAE, stored positive).
        peak_price:             Highest price for LONG / lowest for SHORT since entry.
        signal_history:         List of daily signal values since entry.
        agreement_history:      List of daily agreement values since entry.
        volume_history:         List of daily volume values since entry.
        bars_since_entry:       Number of trading days since entry.
    """
    symbol: str
    direction: str  # "LONG" or "SHORT"
    entry_price: float
    entry_date: str
    entry_signal: float
    entry_agreement: float
    entry_volatility: float
    entry_regime: str
    remaining_fraction: float = 1.0
    staged_exits_done: int = 0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    peak_price: float = 0.0
    signal_history: List[float] = field(default_factory=list)
    agreement_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)
    bars_since_entry: int = 0

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price


# ═══════════════════════════════════════════════════════════════════════════════
# SmartExitEngine
# ═══════════════════════════════════════════════════════════════════════════════

class SmartExitEngine:
    """Multi-signal weighted voting exit system.

    Evaluates 10 independent exit signals per bar, weights them by the
    current market regime, and produces an ExitDecision via a voting
    mechanism.

    Parameters
    ----------
    garch_model : optional
        A fitted GARCH model object with a `forecast()` method.  If
        provided, the engine can generate its own vol forecasts; otherwise
        the caller must supply `current_vol_forecast`.
    trade_history : list, optional
        Historical closed trades for computing Bayesian priors on exit
        timing.  Each entry should be a dict with keys: pnl, bars_held,
        regime, exit_reason.
    hmm_model : optional
        A fitted HMM model (hmmlearn GaussianHMM) with a `transmat_`
        attribute.  Used by the regime_transition signal to assess
        regime stability via the transition probability matrix.
    regime_state_map : dict, optional
        Mapping from regime label ("Bull", "Bear", "Sideways") to the
        HMM state integer index.  Required for regime_transition signal.
    """

    # ── Regime-dependent configuration ────────────────────────────────────

    REGIME_CONFIGS: Dict[str, Dict[str, float]] = {
        "bull": {
            "tp_atr_mult": 3.0,           # Take-profit at 3× ATR
            "sl_atr_mult": 2.5,            # Stop-loss at 2.5× ATR
            "partial_exit_1_atr": 2.0,     # Tranche 1 at 2× ATR profit
            "partial_exit_2_atr": 2.5,     # Tranche 2 at 2.5× ATR profit
            "max_hold_days": 15,           # Maximum holding period
            "time_decay_start_day": 8,     # Time decay kicks in at day 8
            "alpha_decay_threshold": 0.60, # Exit when 60% of alpha consumed
            "agreement_decay_threshold": 0.50,
            "drawdown_from_peak_pct": 0.40,  # Max 40% giveback from MFE
            "trailing_stop_pct": 0.025,    # 2.5% trailing stop
            "vol_expansion_exit_mult": 2.5,  # Exit if vol doubles+
        },
        "bear": {
            "tp_atr_mult": 1.2,
            "sl_atr_mult": 1.0,
            "partial_exit_1_atr": 0.8,
            "partial_exit_2_atr": 1.0,
            "max_hold_days": 5,
            "time_decay_start_day": 3,
            "alpha_decay_threshold": 0.35,
            "agreement_decay_threshold": 0.35,
            "drawdown_from_peak_pct": 0.20,
            "trailing_stop_pct": 0.012,
            "vol_expansion_exit_mult": 1.8,
        },
        "sideways": {
            "tp_atr_mult": 1.5,
            "sl_atr_mult": 1.5,
            "partial_exit_1_atr": 1.0,
            "partial_exit_2_atr": 1.3,
            "max_hold_days": 8,
            "time_decay_start_day": 5,
            "alpha_decay_threshold": 0.50,
            "agreement_decay_threshold": 0.45,
            "drawdown_from_peak_pct": 0.30,
            "trailing_stop_pct": 0.018,
            "vol_expansion_exit_mult": 2.0,
        },
    }

    SIGNAL_WEIGHTS: Dict[str, Dict[str, float]] = {
        "bull": {
            "alpha_decay": 0.13,
            "profit_target": 0.10,
            "trailing_stop": 0.13,
            "time_decay": 0.10,
            "agreement_decay": 0.13,
            "drawdown_from_peak": 0.13,
            "volatility_shift": 0.10,
            "volume_climax": 0.10,
            "event_calendar": 0.04,
            "regime_transition": 0.04,       # HMM transition matrix signal
        },
        "bear": {
            "alpha_decay": 0.10,
            "profit_target": 0.19,
            "trailing_stop": 0.19,
            "time_decay": 0.13,
            "agreement_decay": 0.10,
            "drawdown_from_peak": 0.09,
            "volatility_shift": 0.10,
            "volume_climax": 0.04,
            "event_calendar": 0.04,
            "regime_transition": 0.02,       # lower in bear (already defensive)
        },
        "sideways": {
            "alpha_decay": 0.13,
            "profit_target": 0.13,
            "trailing_stop": 0.10,
            "time_decay": 0.13,
            "agreement_decay": 0.13,
            "drawdown_from_peak": 0.10,
            "volatility_shift": 0.10,
            "volume_climax": 0.10,
            "event_calendar": 0.04,
            "regime_transition": 0.04,       # HMM transition matrix signal
        },
    }

    EXIT_THRESHOLD: float = 0.55

    # ── Constructor ───────────────────────────────────────────────────────

    def __init__(
        self,
        garch_model: Optional[Any] = None,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        event_calendar: Optional[Any] = None,
        hmm_model: Optional[Any] = None,
        regime_state_map: Optional[Dict[str, int]] = None,
        regime_config_override: Optional[Dict[str, Dict[str, float]]] = None,
        exit_threshold_override: Optional[float] = None,
    ) -> None:
        # ── Instance-level copies of class configs ─────────────────────
        # Deep-copy so evolution / overrides don't mutate class defaults
        self.REGIME_CONFIGS = deepcopy(self.__class__.REGIME_CONFIGS)
        self.SIGNAL_WEIGHTS = deepcopy(self.__class__.SIGNAL_WEIGHTS)
        self.EXIT_THRESHOLD = self.__class__.EXIT_THRESHOLD

        # ── Apply overrides (from genetic evolution) ───────────────────
        if regime_config_override:
            for regime, params in regime_config_override.items():
                if regime in self.REGIME_CONFIGS:
                    self.REGIME_CONFIGS[regime].update(params)
            logger.debug(
                f"[SmartExit] Applied regime_config_override for "
                f"{list(regime_config_override.keys())}"
            )
        elif exit_threshold_override is None:
            # No explicit override → try loading evolved params from disk
            self._load_evolved_params()

        if exit_threshold_override is not None:
            self.EXIT_THRESHOLD = exit_threshold_override

        # ── Standard fields ────────────────────────────────────────────
        self._garch_model = garch_model
        self._trade_history = trade_history or []
        self._event_calendar = event_calendar  # EventCalendar instance (or None)

        # ── HMM regime transition signal ──────────────────────────────
        self._hmm_model = hmm_model
        # regime_state_map: {"Bull": 0, "Bear": 1, "Sideways": 2}
        self._regime_state_map: Dict[str, int] = regime_state_map or {}
        # Invert: {0: "Bull", 1: "Bear", 2: "Sideways"}
        self._state_regime_map: Dict[int, str] = {v: k for k, v in self._regime_state_map.items()}
        # 5-day sliding window of p_stay values for trend detection
        self._p_stay_history: List[float] = []

        # Pre-compute Bayesian priors from trade history if available
        self._bayesian_priors: Dict[str, Dict[str, float]] = {}
        if self._trade_history:
            self._compute_bayesian_priors()

    # ══════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════

    def evaluate(
        self,
        position: PositionState,
        current_price: float,
        current_low: float,
        current_volume: float,
        atr: float,
        current_signal: float,
        current_agreement: float,
        current_vol_forecast: float,
        regime: str,
        current_high: Optional[float] = None,
        eval_date: Optional[Any] = None,
    ) -> ExitDecision:
        """Evaluate whether a position should be exited.

        This is the main entry point.  It runs through the full exit
        decision pipeline:

        1. Update position state (MFE, MAE, peak, histories)
        2. Compute unrealised P&L %
        3. Check hard stops (always override — checked FIRST)
        4. Check staged profit taking
        5. Evaluate all 10 signals
        6. Weighted voting → ExitDecision

        Parameters
        ----------
        position : PositionState
            The extended position state.  MUTATED in place (histories
            appended, MFE/MAE updated, bars_since_entry incremented).
        current_price : float
            Current bar's close price.
        current_low : float
            Current bar's low price (for trailing stop checking).
        current_volume : float
            Current bar's volume.
        atr : float
            Current ATR value (14-period recommended).
        current_signal : float
            Current signal strength from the player (same scale as
            entry_signal).
        current_agreement : float
            Current player agreement score (0–1).
        current_vol_forecast : float
            Current GARCH volatility forecast (annualised).
        regime : str
            Current market regime: "bull", "bear", or "sideways".
        current_high : float, optional
            Current bar's high price (for SHORT stop checking).
            If not provided, defaults to current_price.
        eval_date : date, optional
            Current evaluation date.  Used by the event_calendar signal
            to check proximity to scheduled events.  If not provided,
            the event_calendar signal returns 0.0.

        Returns
        -------
        ExitDecision
            Contains should_exit, exit_fraction, reason, urgency, and
            a details dict with all signal scores.
        """
        if current_high is None:
            current_high = current_price
        regime = regime.lower()
        if regime not in self.REGIME_CONFIGS:
            regime = "sideways"

        cfg = self.REGIME_CONFIGS[regime]
        weights = self.SIGNAL_WEIGHTS[regime]

        # ── Step 1: Update position state ─────────────────────────────
        self._update_position_state(
            position, current_price, current_low, current_high,
            current_volume, current_signal, current_agreement, atr,
        )

        # ── Step 2: Unrealised P&L % ─────────────────────────────────
        if abs(position.entry_price) < 0.01:
            unrealised_pnl_pct = 0.0
        elif position.direction == "LONG":
            unrealised_pnl_pct = (current_price - position.entry_price) / position.entry_price
        else:
            unrealised_pnl_pct = (position.entry_price - current_price) / position.entry_price

        unrealised_pnl_atr = 0.0
        if atr > 0:
            if position.direction == "LONG":
                unrealised_pnl_atr = (current_price - position.entry_price) / atr
            else:
                unrealised_pnl_atr = (position.entry_price - current_price) / atr

        # ── Step 3: Check hard stops (FIRST — always override) ────────
        hard_stop = self._check_hard_stops(
            position, current_price, current_low, current_high, atr, cfg, regime,
        )
        if hard_stop is not None:
            return hard_stop

        # ── Step 4: Check staged profit taking ────────────────────────
        staged_decision = self._check_staged_exits(
            position, unrealised_pnl_atr, atr, cfg, regime,
        )
        if staged_decision is not None:
            return staged_decision

        # ── Step 5: Evaluate all 10 signals ───────────────────────────
        signals: Dict[str, float] = {
            "alpha_decay": self._signal_alpha_decay(
                position, current_signal, cfg,
            ),
            "profit_target": self._signal_profit_target(
                position, unrealised_pnl_atr, cfg,
            ),
            "trailing_stop": self._signal_trailing_stop(
                position, current_price, current_low, current_high, cfg,
            ),
            "time_decay": self._signal_time_decay(
                position, cfg,
            ),
            "agreement_decay": self._signal_agreement_decay(
                position, current_agreement, cfg,
            ),
            "drawdown_from_peak": self._signal_drawdown_from_peak(
                position, unrealised_pnl_pct, cfg,
            ),
            "volatility_shift": self._signal_volatility_shift(
                position, current_vol_forecast, cfg,
            ),
            "volume_climax": self._signal_volume_climax(
                position, current_volume,
            ),
            "event_calendar": self._signal_event_calendar(
                eval_date, unrealised_pnl_pct,
            ),
            "regime_transition": self._signal_regime_transition(
                position, regime,
            ),
        }

        # ── Step 6: Weighted voting ───────────────────────────────────
        weighted_score = sum(
            signals[name] * weights[name] for name in signals
        )

        # Build details dict for logging / debugging
        details: Dict[str, Any] = {
            "regime": regime,
            "unrealised_pnl_pct": round(unrealised_pnl_pct, 4),
            "unrealised_pnl_atr": round(unrealised_pnl_atr, 2),
            "bars_held": position.bars_since_entry,
            "mfe": round(position.max_favorable_excursion, 4),
            "mae": round(position.max_adverse_excursion, 4),
            "peak_price": round(position.peak_price, 2),
            "weighted_score": round(weighted_score, 4),
            "threshold": self.EXIT_THRESHOLD,
            "signal_scores": {k: round(v, 3) for k, v in signals.items()},
            "weighted_contributions": {
                k: round(signals[k] * weights[k], 4) for k in signals
            },
        }

        if weighted_score >= self.EXIT_THRESHOLD:
            # Determine dominant reason (highest weighted contribution)
            contributions = {k: signals[k] * weights[k] for k in signals}
            dominant = max(contributions, key=contributions.get)

            reason_map = {
                "alpha_decay": ExitReason.ALPHA_DECAY,
                "profit_target": ExitReason.PROFIT_TARGET,
                "trailing_stop": ExitReason.TRAILING_STOP,
                "time_decay": ExitReason.TIME_DECAY,
                "agreement_decay": ExitReason.AGREEMENT_DECAY,
                "drawdown_from_peak": ExitReason.DRAWDOWN_FROM_PEAK,
                "volatility_shift": ExitReason.VOLATILITY_SPIKE,
                "volume_climax": ExitReason.VOLUME_CLIMAX,
                "event_calendar": ExitReason.EVENT_RISK,
                "regime_transition": ExitReason.REGIME_TRANSITION,
            }

            # Exit fraction scales with how far above threshold
            exit_fraction = min(1.0, 0.5 + (weighted_score - self.EXIT_THRESHOLD) * 2.0)
            urgency = min(1.0, weighted_score)

            logger.info(
                f"[SmartExit] {position.symbol} EXIT — score={weighted_score:.3f} "
                f"reason={dominant} fraction={exit_fraction:.2f} "
                f"pnl={unrealised_pnl_pct:+.2%}"
            )

            return ExitDecision(
                should_exit=True,
                exit_fraction=exit_fraction,
                reason=reason_map.get(dominant, ExitReason.ALPHA_DECAY),
                urgency=urgency,
                details=details,
            )

        # No exit
        return ExitDecision(
            should_exit=False,
            exit_fraction=0.0,
            reason=ExitReason.ALPHA_DECAY,  # placeholder
            urgency=0.0,
            details=details,
        )

    # ══════════════════════════════════════════════════════════════════════
    # POSITION STATE UPDATER
    # ══════════════════════════════════════════════════════════════════════

    def _update_position_state(
        self,
        pos: PositionState,
        current_price: float,
        current_low: float,
        current_high: float,
        current_volume: float,
        current_signal: float,
        current_agreement: float,
        atr: float,
    ) -> None:
        """Update position state with current bar data.

        Called at the start of every evaluate() call.  Appends to
        histories, updates peak price, MFE, MAE, and increments the
        bar counter.

        MFE (Maximum Favourable Excursion)
        -----------------------------------
        The best unrealised P&L % the position has seen.  Used by the
        drawdown-from-peak signal to detect profit giveback.

        MAE (Maximum Adverse Excursion)
        --------------------------------
        The worst unrealised P&L % (stored as positive).  Used for
        post-trade analytics and Bayesian priors.

        Peak Price
        ----------
        For LONG positions: highest price seen since entry (uses HIGH).
        For SHORT positions: lowest price seen since entry (uses LOW).
        Used by the trailing stop to ratchet the stop level.
        """
        # Append histories (capped at 60 entries — see Bug 8)
        pos.signal_history.append(current_signal)
        pos.agreement_history.append(current_agreement)
        pos.volume_history.append(current_volume)
        if len(pos.signal_history) > 60:
            pos.signal_history = pos.signal_history[-60:]
        if len(pos.agreement_history) > 60:
            pos.agreement_history = pos.agreement_history[-60:]
        if len(pos.volume_history) > 60:
            pos.volume_history = pos.volume_history[-60:]

        # Update peak price (for trailing stop ratcheting)
        if pos.direction == "LONG":
            # For longs, use HIGH of bar for peak tracking
            pos.peak_price = max(pos.peak_price, current_high)
        else:
            # For shorts, peak is the lowest price (best for short seller)
            # Use LOW of bar for peak tracking
            pos.peak_price = min(pos.peak_price, current_low)

        # Compute current unrealised P&L %
        if abs(pos.entry_price) < 0.01:
            current_pnl_pct = 0.0
        elif pos.direction == "LONG":
            current_pnl_pct = (current_price - pos.entry_price) / pos.entry_price
        else:
            current_pnl_pct = (pos.entry_price - current_price) / pos.entry_price

        # Update MFE (best P&L % seen)
        pos.max_favorable_excursion = max(
            pos.max_favorable_excursion, current_pnl_pct,
        )

        # Update MAE (worst P&L %, stored positive)
        if current_pnl_pct < 0:
            pos.max_adverse_excursion = max(
                pos.max_adverse_excursion, abs(current_pnl_pct),
            )

        # Increment bar counter
        pos.bars_since_entry += 1

    # ══════════════════════════════════════════════════════════════════════
    # STAGED PROFIT TAKING
    # ══════════════════════════════════════════════════════════════════════

    def _check_staged_exits(
        self,
        pos: PositionState,
        unrealised_pnl_atr: float,
        atr: float,
        cfg: Dict[str, float],
        regime: str,
    ) -> Optional[ExitDecision]:
        """Check if a staged profit-taking tranche should trigger.

        Staged exits follow optimal Kelly-inspired sizing:
        - Tranche 1 (33%): when profit >= partial_exit_1_atr × ATR
        - Tranche 2 (33%): when profit >= partial_exit_2_atr × ATR
        - Tranche 3 (34%): runs with trailing stop (handled by main flow)

        This balances the capture ratio (locking in profit) against
        letting winners run (the remaining fraction stays open).

        Returns None if no staged exit is triggered.
        """
        if pos.remaining_fraction <= 0.34:
            # Only the final tranche remains — let it ride with stops
            return None

        if unrealised_pnl_atr <= 0:
            # Not in profit — no staged exits
            return None

        # Tranche 1: 33% of original → 33% of remaining (remaining=1.0)
        if pos.staged_exits_done == 0 and unrealised_pnl_atr >= cfg["partial_exit_1_atr"]:
            pos.staged_exits_done = 1
            # exit_fraction = fraction of REMAINING position to sell
            exit_frac_of_remaining = 0.33 / pos.remaining_fraction if pos.remaining_fraction > 0 else 1.0
            exit_frac_of_remaining = min(1.0, exit_frac_of_remaining)
            pos.remaining_fraction -= 0.33

            logger.info(
                f"[SmartExit] {pos.symbol} STAGED EXIT T1 — "
                f"profit={unrealised_pnl_atr:.1f}×ATR, "
                f"exiting {exit_frac_of_remaining:.0%} of remaining, "
                f"remaining={pos.remaining_fraction:.0%}"
            )
            return ExitDecision(
                should_exit=True,
                exit_fraction=exit_frac_of_remaining,
                reason=ExitReason.STAGED_PROFIT,
                urgency=0.5,
                details={
                    "tranche": 1,
                    "profit_atr": round(unrealised_pnl_atr, 2),
                    "threshold_atr": cfg["partial_exit_1_atr"],
                    "remaining_fraction": round(pos.remaining_fraction, 2),
                    "exit_fraction_of_remaining": round(exit_frac_of_remaining, 4),
                    "regime": regime,
                },
            )

        # Tranche 2: another 33% of original → ~49% of remaining (remaining≈0.67)
        if pos.staged_exits_done == 1 and unrealised_pnl_atr >= cfg["partial_exit_2_atr"]:
            pos.staged_exits_done = 2
            exit_frac_of_remaining = 0.33 / pos.remaining_fraction if pos.remaining_fraction > 0 else 1.0
            exit_frac_of_remaining = min(1.0, exit_frac_of_remaining)
            pos.remaining_fraction -= 0.33

            logger.info(
                f"[SmartExit] {pos.symbol} STAGED EXIT T2 — "
                f"profit={unrealised_pnl_atr:.1f}×ATR, "
                f"exiting {exit_frac_of_remaining:.0%} of remaining, "
                f"remaining={pos.remaining_fraction:.0%}"
            )
            return ExitDecision(
                should_exit=True,
                exit_fraction=exit_frac_of_remaining,
                reason=ExitReason.STAGED_PROFIT,
                urgency=0.5,
                details={
                    "tranche": 2,
                    "profit_atr": round(unrealised_pnl_atr, 2),
                    "threshold_atr": cfg["partial_exit_2_atr"],
                    "remaining_fraction": round(pos.remaining_fraction, 2),
                    "exit_fraction_of_remaining": round(exit_frac_of_remaining, 4),
                    "regime": regime,
                },
            )

        return None

    # ══════════════════════════════════════════════════════════════════════
    # HARD STOPS (always override)
    # ══════════════════════════════════════════════════════════════════════

    def _check_hard_stops(
        self,
        pos: PositionState,
        current_price: float,
        current_low: float,
        current_high: float,
        atr: float,
        cfg: Dict[str, float],
        regime: str,
    ) -> Optional[ExitDecision]:
        """Check unconditional hard stop conditions.

        Two hard stops that always trigger regardless of voting:

        1. **ATR-based stop-loss using LOW price** — If the bar's low
           breaches entry ∓ sl_atr_mult × ATR, exit immediately.
           Using LOW (not close) catches intraday violations.

        2. **Maximum holding period** — If bars_since_entry exceeds
           max_hold_days, exit.  Holding too long = dead capital.

        Returns None if no hard stop is hit.
        """
        # Hard stop 1: ATR-based stop-loss (checks LOW for longs, HIGH for shorts)
        if atr > 0:
            if pos.direction == "LONG":
                stop_price = pos.entry_price - cfg["sl_atr_mult"] * atr
                if current_low <= stop_price:
                    logger.info(
                        f"[SmartExit] {pos.symbol} HARD STOP — "
                        f"low={current_low:.2f} <= stop={stop_price:.2f} "
                        f"({cfg['sl_atr_mult']}×ATR)"
                    )
                    return ExitDecision(
                        should_exit=True,
                        exit_fraction=1.0,
                        reason=ExitReason.RISK_LIMIT,
                        urgency=1.0,
                        details={
                            "trigger": "atr_stop_loss",
                            "stop_price": round(stop_price, 2),
                            "current_low": current_low,
                            "sl_atr_mult": cfg["sl_atr_mult"],
                            "regime": regime,
                        },
                    )
            else:  # SHORT
                stop_price = pos.entry_price + cfg["sl_atr_mult"] * atr
                # For shorts, check HIGH of bar against stop (adverse move is up)
                if current_high >= stop_price:
                    logger.info(
                        f"[SmartExit] {pos.symbol} HARD STOP (SHORT) — "
                        f"high={current_high:.2f} >= stop={stop_price:.2f}"
                    )
                    return ExitDecision(
                        should_exit=True,
                        exit_fraction=1.0,
                        reason=ExitReason.RISK_LIMIT,
                        urgency=1.0,
                        details={
                            "trigger": "atr_stop_loss",
                            "stop_price": round(stop_price, 2),
                            "current_high": current_high,
                            "sl_atr_mult": cfg["sl_atr_mult"],
                            "regime": regime,
                        },
                    )

        # Hard stop 2: Maximum holding period
        if pos.bars_since_entry >= cfg["max_hold_days"]:
            logger.info(
                f"[SmartExit] {pos.symbol} MAX HOLD — "
                f"{pos.bars_since_entry} bars >= {cfg['max_hold_days']}"
            )
            return ExitDecision(
                should_exit=True,
                exit_fraction=1.0,
                reason=ExitReason.TIME_DECAY,
                urgency=0.9,
                details={
                    "trigger": "max_hold_days",
                    "bars_held": pos.bars_since_entry,
                    "max_hold": int(cfg["max_hold_days"]),
                    "regime": regime,
                },
            )

        return None

    # ══════════════════════════════════════════════════════════════════════
    # INDIVIDUAL EXIT SIGNALS (each returns 0.0–1.0)
    # ══════════════════════════════════════════════════════════════════════

    def _signal_alpha_decay(
        self,
        pos: PositionState,
        current_signal: float,
        cfg: Dict[str, float],
    ) -> float:
        """Evaluate alpha decay — how much entry signal has been consumed.

        The alpha (expected return) that justified the trade decays over
        time.  We measure this as:

            alpha_consumed = 1 − (current_signal / entry_signal)

        Special cases:
        - If the signal has **reversed direction** (entry was bullish,
          now bearish), return 1.0 immediately.
        - If entry_signal ≈ 0, return 0.0 (no alpha to decay).

        The score is scaled linearly against alpha_decay_threshold:
            score = alpha_consumed / threshold

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        if abs(pos.entry_signal) < 0.01:
            return 0.0

        # Check for signal reversal (direction flip)
        if pos.entry_signal > 0 and current_signal < 0:
            return 1.0
        if pos.entry_signal < 0 and current_signal > 0:
            return 1.0

        # Compute alpha consumed
        signal_ratio = current_signal / pos.entry_signal
        alpha_consumed = max(0.0, 1.0 - signal_ratio)

        threshold = cfg["alpha_decay_threshold"]
        if threshold <= 0:
            return 0.0

        score = alpha_consumed / threshold
        return min(1.0, max(0.0, score))

    def _signal_profit_target(
        self,
        pos: PositionState,
        unrealised_pnl_atr: float,
        cfg: Dict[str, float],
    ) -> float:
        """Evaluate profit target — ATR-normalised distance to target.

        Uses ATR to normalise the profit target so it adapts to current
        volatility.  The target in ATR multiples is scaled by entry
        signal confidence:

            effective_target = tp_atr_mult × (0.5 + 0.5 × |entry_signal|)

        This means high-conviction entries get a higher target (let them
        run), while low-conviction entries take profit earlier.

        Scoring:
        - profit_atr >= target       → 1.0
        - profit_atr >= 80% target   → 0.7
        - profit_atr >= 50% target   → 0.3
        - below 50%                  → linear 0.0–0.3

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        if unrealised_pnl_atr <= 0:
            return 0.0

        # Scale target by entry signal confidence
        confidence_scale = 0.5 + 0.5 * min(1.0, abs(pos.entry_signal))
        effective_target = cfg["tp_atr_mult"] * confidence_scale

        if effective_target <= 0:
            return 0.0

        ratio = unrealised_pnl_atr / effective_target

        if ratio >= 1.0:
            return 1.0
        elif ratio >= 0.8:
            return 0.7
        elif ratio >= 0.5:
            return 0.3
        else:
            # Linear scale from 0 to 0.3 as ratio goes from 0 to 0.5
            return 0.3 * (ratio / 0.5)

    def _signal_trailing_stop(
        self,
        pos: PositionState,
        current_price: float,
        current_low: float,
        current_high: float,
        cfg: Dict[str, float],
    ) -> float:
        """Evaluate trailing stop using bar LOW price.

        The trailing stop ratchets upward (for longs) with the peak
        price.  Crucially, it checks the bar's LOW price, not the
        close, to catch intraday violations:

            LONG:  stop_level = peak_price × (1 − trailing_stop_pct)
                   if low ≤ stop_level → score = 1.0

            SHORT: stop_level = peak_price × (1 + trailing_stop_pct)
                   if price ≥ stop_level → score = 1.0

        When price is approaching the stop (within 50% of the gap),
        the score ramps from 0.0 to 1.0 linearly.

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        stop_pct = cfg["trailing_stop_pct"]

        if pos.direction == "LONG":
            stop_level = pos.peak_price * (1.0 - stop_pct)
            if current_low <= stop_level:
                return 1.0

            # How close is current_low to the stop?
            gap = pos.peak_price - stop_level  # total gap
            if gap <= 0:
                return 0.0
            distance_from_stop = current_low - stop_level
            proximity = 1.0 - (distance_from_stop / gap)
            # Only signal when within 50% of the gap
            if proximity > 0.5:
                return min(1.0, (proximity - 0.5) * 2.0)
            return 0.0

        else:  # SHORT
            stop_level = pos.peak_price * (1.0 + stop_pct)
            # For shorts, check HIGH of bar (adverse move is upward)
            if current_high >= stop_level:
                return 1.0

            gap = stop_level - pos.peak_price
            if gap <= 0:
                return 0.0
            distance_from_stop = stop_level - current_high
            proximity = 1.0 - (distance_from_stop / gap)
            if proximity > 0.5:
                return min(1.0, (proximity - 0.5) * 2.0)
            return 0.0

    def _signal_time_decay(
        self,
        pos: PositionState,
        cfg: Dict[str, float],
    ) -> float:
        """Evaluate time decay — linear ramp toward max hold days.

        Holding periods have diminishing expected returns.  This signal
        linearly ramps from 0.0 to 1.0 between `time_decay_start_day`
        and `max_hold_days`:

            if bars < start_day:  score = 0.0
            if bars >= max_days:  score = 1.0
            else:  score = (bars - start_day) / (max_days - start_day)

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        start = cfg["time_decay_start_day"]
        end = cfg["max_hold_days"]
        bars = pos.bars_since_entry

        if bars < start:
            return 0.0
        if bars >= end:
            return 1.0
        if end <= start:
            return 1.0

        return (bars - start) / (end - start)

    def _signal_agreement_decay(
        self,
        pos: PositionState,
        current_agreement: float,
        cfg: Dict[str, float],
    ) -> float:
        """Evaluate agreement decay — consensus eroding since entry.

        If the players agreed strongly at entry (say 0.8) and now only
        0.4 agree, the consensus that justified the trade has eroded.

            decay_ratio = (entry_agreement − current_agreement) / entry_agreement

        Scaled against the agreement_decay_threshold:
            score = decay_ratio / threshold

        If entry_agreement was very low (< 0.1), return 0.0 — there
        was no consensus to decay.

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        if pos.entry_agreement < 0.1:
            return 0.0

        decay_ratio = (pos.entry_agreement - current_agreement) / pos.entry_agreement
        decay_ratio = max(0.0, decay_ratio)  # ignore improvement

        threshold = cfg["agreement_decay_threshold"]
        if threshold <= 0:
            return 0.0

        score = decay_ratio / threshold
        return min(1.0, max(0.0, score))

    def _signal_drawdown_from_peak(
        self,
        pos: PositionState,
        unrealised_pnl_pct: float,
        cfg: Dict[str, float],
    ) -> float:
        """Evaluate drawdown from peak — profit giveback detection.

        Only active when the position has been profitable (MFE > 0).
        Measures how much of the best unrealised profit has been given
        back:

            giveback_ratio = (MFE − current_pnl) / MFE

        A giveback_ratio of 0.4 means 40% of the peak profit has been
        lost.  Scaled against drawdown_from_peak_pct threshold.

        This is the "don't let a winner turn into a loser" signal.

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        mfe = pos.max_favorable_excursion
        if mfe <= 0.005:  # less than 0.5% peak profit — not meaningful
            return 0.0

        giveback = mfe - unrealised_pnl_pct
        if giveback <= 0:
            return 0.0  # still at or near peak, no giveback

        giveback_ratio = giveback / mfe
        threshold = cfg["drawdown_from_peak_pct"]
        if threshold <= 0:
            return 0.0

        score = giveback_ratio / threshold
        return min(1.0, max(0.0, score))

    def _signal_volatility_shift(
        self,
        pos: PositionState,
        current_vol_forecast: float,
        cfg: Dict[str, float],
    ) -> float:
        """Evaluate volatility shift — sudden vol expansion.

        A spike in volatility often precedes adverse moves.  We compare
        current vol to entry vol:

            vol_ratio = current_vol / entry_vol

        Scoring:
        - vol_ratio >= expansion_mult → 1.0 (full signal)
        - vol_ratio >= 70% of expansion_mult → 0.5
        - Linear interpolation between 50% and 70%

        If entry_volatility was near zero, return 0.0.

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        if pos.entry_volatility < 0.01:
            return 0.0

        vol_ratio = current_vol_forecast / pos.entry_volatility

        expansion_mult = cfg["vol_expansion_exit_mult"]
        if expansion_mult <= 0:
            return 0.0

        if vol_ratio >= expansion_mult:
            return 1.0

        # Threshold at 70% of expansion_mult
        threshold_70 = expansion_mult * 0.70
        threshold_50 = expansion_mult * 0.50

        if vol_ratio >= threshold_70:
            return 0.5 + 0.5 * ((vol_ratio - threshold_70) / (expansion_mult - threshold_70))
        elif vol_ratio >= threshold_50:
            return 0.5 * ((vol_ratio - threshold_50) / (threshold_70 - threshold_50))

        return 0.0

    def _signal_volume_climax(
        self,
        pos: PositionState,
        current_volume: float,
    ) -> float:
        """Evaluate volume climax — abnormally high volume.

        A volume climax often signals exhaustion (the end of a move).
        We compare current volume to the 20-day average volume from
        the position's history:

        Scoring:
        - volume >= 3.0× avg → 0.8 (strong climax signal)
        - volume >= 2.5× avg → 0.4
        - volume >= 2.0× avg → 0.2
        - below 2.0×          → 0.0

        Requires at least 5 bars of volume history to compute a
        meaningful average.

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        if len(pos.volume_history) < 20:
            return 0.0

        # Use up to 20 bars of history for average
        lookback = pos.volume_history[-20:]
        avg_volume = sum(lookback) / len(lookback)

        if avg_volume <= 0:
            return 0.0

        vol_ratio = current_volume / avg_volume

        if vol_ratio >= 3.0:
            return 0.8
        elif vol_ratio >= 2.5:
            return 0.4
        elif vol_ratio >= 2.0:
            return 0.2
        return 0.0

    def _signal_event_calendar(
        self,
        eval_date: Optional[Any],
        unrealised_pnl_pct: float,
    ) -> float:
        """Evaluate event calendar — upcoming scheduled event risk.

        Delegates to EventCalendar.exit_signal_score() which considers
        both event proximity/severity AND position P&L:
        - Profitable positions (>1%) get strong tightening (protect gains)
        - Losing positions get moderate tightening
        - Flat positions get mild tightening

        If no EventCalendar is configured or no eval_date is provided,
        returns 0.0 (no event risk signal).

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        if self._event_calendar is None or eval_date is None:
            return 0.0

        try:
            score = self._event_calendar.exit_signal_score(eval_date, unrealised_pnl_pct)
            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.warning(f"[SmartExit] Event calendar signal error: {e}")
            return 0.0

    def _signal_regime_transition(
        self,
        position: PositionState,
        regime: str,
    ) -> float:
        """Evaluate regime transition probability using HMM transition matrix.

        Uses the fitted HMM's transition probability matrix (transmat_) to
        assess how stable the current regime is and whether an adverse
        regime transition is likely.

        The score combines three components:

            score = instability × 0.5 + p_adverse × 0.3 + regime_mismatch × 0.2

        where:
            - instability = 1.0 − P(stay in current regime)
            - p_adverse = P(transition to adverse regime)
                For LONG positions: adverse = "Bear"
                For SHORT positions: adverse = "Bull"
            - regime_mismatch = 0.3 if current regime ≠ position's entry regime

        Additionally tracks p_stay over a 5-day sliding window.  If p_stay
        is declining (trend < 0), the score gets a 20% boost to signal
        increasing regime instability.

        Returns
        -------
        float
            Signal score ∈ [0.0, 1.0].
        """
        if self._hmm_model is None or not hasattr(self._hmm_model, 'transmat_'):
            return 0.0

        if not self._regime_state_map:
            return 0.0

        # Normalise regime label for lookup — the map uses title-case ("Bull")
        # but SmartExitEngine evaluate() normalises to lowercase ("bull")
        regime_title = regime.capitalize()  # "bull" → "Bull", "bear" → "Bear"

        current_state_idx = self._regime_state_map.get(regime_title)
        if current_state_idx is None:
            return 0.0

        try:
            transmat = self._hmm_model.transmat_
        except Exception:
            return 0.0

        if transmat.ndim != 2 or transmat.shape[0] != transmat.shape[1]:
            return 0.0
        n_states = transmat.shape[0]
        if current_state_idx >= n_states:
            return 0.0

        # ── P(stay in current regime) ──────────────────────────────────
        p_stay = float(transmat[current_state_idx, current_state_idx])
        instability = 1.0 - p_stay

        # ── P(adverse regime) ──────────────────────────────────────────
        # For LONG: adverse is Bear.  For SHORT: adverse is Bull.
        if position.direction == "LONG":
            adverse_regime = "Bear"
        else:
            adverse_regime = "Bull"

        adverse_state_idx = self._regime_state_map.get(adverse_regime)
        if adverse_state_idx is not None and adverse_state_idx < n_states:
            p_adverse = float(transmat[current_state_idx, adverse_state_idx])
        else:
            p_adverse = 0.0

        # ── Regime mismatch ────────────────────────────────────────────
        # If entry_regime is unknown, skip mismatch penalty (don't penalize missing data)
        if position.entry_regime:
            entry_regime_title = position.entry_regime.capitalize()
            regime_mismatch = 0.3 if regime_title != entry_regime_title else 0.0
        else:
            regime_mismatch = 0.0

        # ── Combined score ─────────────────────────────────────────────
        score = instability * 0.5 + p_adverse * 0.3 + regime_mismatch * 0.2

        # ── 5-day sliding window trend detection ───────────────────────
        self._p_stay_history.append(p_stay)
        if len(self._p_stay_history) > 5:
            self._p_stay_history = self._p_stay_history[-5:]

        if len(self._p_stay_history) >= 3:
            # Simple linear trend: compare first half to second half
            mid = len(self._p_stay_history) // 2
            first_avg = sum(self._p_stay_history[:mid]) / mid
            second_avg = sum(self._p_stay_history[mid:]) / (len(self._p_stay_history) - mid)
            trend = second_avg - first_avg  # negative = p_stay declining
            if trend < 0:
                # p_stay declining → 20% boost to score
                score *= 1.20

        return min(1.0, max(0.0, score))

    # ══════════════════════════════════════════════════════════════════════
    # EVOLVED PARAMETER LOADING
    # ══════════════════════════════════════════════════════════════════════

    def _load_evolved_params(self) -> None:
        """Load evolved exit parameters from disk if available.

        Checks for `data/evolved_exit_params.json` relative to this file.
        If found, overrides REGIME_CONFIGS and EXIT_THRESHOLD with the
        evolved values.  Falls back silently if file doesn't exist.
        """
        params_path = Path(__file__).parent / "data" / "evolved_exit_params.json"

        if not params_path.exists():
            return

        try:
            with open(params_path, "r") as f:
                data = json.load(f)

            # Override REGIME_CONFIGS
            if "regime_configs" in data:
                for regime, params in data["regime_configs"].items():
                    if regime in self.REGIME_CONFIGS:
                        self.REGIME_CONFIGS[regime].update(params)

            # Override EXIT_THRESHOLD
            if "exit_threshold" in data:
                self.EXIT_THRESHOLD = data["exit_threshold"]

            logger.info(
                f"[SmartExit] Loaded evolved params "
                f"(threshold={self.EXIT_THRESHOLD:.3f}, "
                f"fitness={data.get('fitness_score', '?')})"
            )

        except Exception as e:
            logger.warning(f"[SmartExit] Failed to load evolved params: {e}")

    # ══════════════════════════════════════════════════════════════════════
    # BAYESIAN PRIORS (from trade history)
    # ══════════════════════════════════════════════════════════════════════

    def _compute_bayesian_priors(self) -> None:
        """Compute Bayesian priors from historical trade outcomes.

        Analyses the trade_history to learn:
        - Average profitable holding period per regime
        - Average losing holding period per regime
        - P(profitable | regime, bars_held)

        These priors modulate the time_decay signal: if historical
        winners in Bull tend to close by day 7, holding to day 12
        is increasingly unlikely to be optimal.
        """
        if not self._trade_history:
            return

        for regime in ("bull", "bear", "sideways"):
            regime_trades = [
                t for t in self._trade_history
                if t.get("regime", "").lower() == regime
            ]
            if not regime_trades:
                continue

            winners = [t for t in regime_trades if t.get("pnl", 0) > 0]
            losers = [t for t in regime_trades if t.get("pnl", 0) <= 0]

            avg_win_bars = (
                sum(t.get("bars_held", 0) for t in winners) / len(winners)
                if winners else 10
            )
            avg_loss_bars = (
                sum(t.get("bars_held", 0) for t in losers) / len(losers)
                if losers else 5
            )
            win_rate = len(winners) / len(regime_trades) if regime_trades else 0.5

            self._bayesian_priors[regime] = {
                "avg_win_bars": avg_win_bars,
                "avg_loss_bars": avg_loss_bars,
                "win_rate": win_rate,
                "sample_size": len(regime_trades),
            }

        logger.info(
            f"[SmartExit] Computed Bayesian priors from "
            f"{len(self._trade_history)} trades"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_smart_exit_engine(
    garch_model: Optional[Any] = None,
    trade_history: Optional[List[Dict[str, Any]]] = None,
    event_calendar: Optional[Any] = None,
    hmm_model: Optional[Any] = None,
    regime_state_map: Optional[Dict[str, int]] = None,
    regime_config_override: Optional[Dict[str, Dict[str, float]]] = None,
    exit_threshold_override: Optional[float] = None,
) -> SmartExitEngine:
    """Create a SmartExitEngine with optional overrides."""
    return SmartExitEngine(
        garch_model=garch_model,
        trade_history=trade_history,
        event_calendar=event_calendar,
        hmm_model=hmm_model,
        regime_state_map=regime_state_map,
        regime_config_override=regime_config_override,
        exit_threshold_override=exit_threshold_override,
    )
