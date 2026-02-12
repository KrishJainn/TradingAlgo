"""
Dynamic Signal Combination System for the 5-Player Trading Coach.

Replaces the simple weighted-average signal combination with a
confidence-weighted, Bayesian-updated, regime-aware voting system.

Components:
  - ConfidenceWeightedVoter:  Combines player signals weighted by confidence × player weight
  - BayesianPlayerWeights:    Maintains and updates per-player weights via Bayesian updating
  - AgreementAnalyzer:        Analyses consensus / dissent patterns among players
  - RegimeAwareCombiner:      Top-level orchestrator tying everything together
  - CombinerDashboard:        Generates summary dicts for monitoring
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

COMBINER_DIR = Path(__file__).parent / "signal_combiner_state"

PLAYER_IDS = ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]

PLAYER_LABELS: Dict[str, str] = {
    "PLAYER_1": "TrendFollower",
    "PLAYER_2": "MeanReversion",
    "PLAYER_3": "VolumeFlow",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "MultiMomentum",
}

REGIMES = ["Bull", "Bear", "Sideways"]

# Signal filter thresholds
# NOTE: MIN_CONFIDENCE_THRESHOLD lowered from 0.7 to 0.25.
# The old 0.7 threshold blocked ~97% of signals because tanh-squashed
# weighted averages rarely exceed 0.4.  The per-player entry_threshold
# (read from evolved_player_configs.json) is the REAL gatekeeper.
MIN_CONFIDENCE_THRESHOLD = 0.25   # Skip trade if |final_signal| < this
SIGNAL_FLIP_THRESHOLD = 0.3     # Don't flip direction unless reversal exceeds this

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PlayerSignal:
    """A single player's signal submission."""

    player_id: str
    direction: float        # [-1.0, 1.0]: negative = bearish, positive = bullish
    confidence: float       # [0.0, 1.0]: how confident the player is
    reasoning: str = ""

    def __post_init__(self):
        self.direction = max(-1.0, min(1.0, float(self.direction)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))


@dataclass
class AgreementResult:
    """Output from the AgreementAnalyzer."""

    agreement_score: float          # [0.0, 1.0]
    pattern: str                    # "unanimous", "strong", "moderate", "no_majority"
    majority_direction: float       # Signed: +1 bullish majority, -1 bearish
    majority_count: int             # How many in majority
    dissent_count: int
    dissenters: List[str]           # player_ids that disagree
    dissent_analysis: str           # Human-readable analysis
    should_trade: bool              # False if no_majority
    position_size_multiplier: float # 1.0 normally, 0.8 for moderate, 0 for skip
    crowding_flag: bool             # True if 5/5 unanimous


@dataclass
class CombinedSignal:
    """Final output from the RegimeAwareCombiner."""

    final_signal: float                 # [-1.0, 1.0]
    position_size_multiplier: float     # [0.0, 1.0+]
    should_trade: bool
    signal_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TradeOutcome:
    """Resolved trade result for Bayesian weight updating."""

    player_id: str
    signal_direction: float     # What the player signaled [-1, 1]
    actual_pnl: float           # Realized PnL
    regime: str                 # Regime at signal time


# ---------------------------------------------------------------------------
# ConfidenceWeightedVoter
# ---------------------------------------------------------------------------

class ConfidenceWeightedVoter:
    """Combines player signals weighted by confidence and player weight.

    Formula:
        final = sum(direction_i * confidence_i * weight_i)
               / sum(confidence_i * weight_i)
    """

    def combine(
        self,
        signals: List[PlayerSignal],
        player_weights: Dict[str, float],
    ) -> Tuple[float, Dict[str, float]]:
        """Combine signals into a single direction value.

        Args:
            signals: List of PlayerSignal from each player.
            player_weights: Dict mapping player_id → weight.

        Returns:
            (final_signal, influence_map) where influence_map shows
            each player's contribution to the final signal.
        """
        numerator = 0.0
        denominator = 0.0
        influence: Dict[str, float] = {}

        for sig in signals:
            w = player_weights.get(sig.player_id, 0.20)
            contrib = sig.confidence * w
            numerator += sig.direction * contrib
            denominator += contrib
            influence[sig.player_id] = sig.direction * contrib

        if denominator < 1e-9:
            return 0.0, {s.player_id: 0.0 for s in signals}

        final = numerator / denominator
        final = max(-1.0, min(1.0, final))

        # Normalize influence to show share of final
        for pid in influence:
            influence[pid] = influence[pid] / denominator

        return final, influence


# ---------------------------------------------------------------------------
# BayesianPlayerWeights
# ---------------------------------------------------------------------------

class BayesianPlayerWeights:
    """Maintains per-player weights via Bayesian updating with exponential decay.

    Regime-conditional: separate weight sets for each regime.
    Constraints: each weight ∈ [0.05, 0.40], all 5 sum to 1.0.
    """

    MIN_WEIGHT = 0.05
    MAX_WEIGHT = 0.40
    HALF_LIFE_DAYS = 20
    # Decay factor per update: λ = 2^(-1/half_life)
    DECAY = 2.0 ** (-1.0 / HALF_LIFE_DAYS)
    # Likelihood scaling: controls how aggressively weights shift
    LIKELIHOOD_SCALE = 2.0

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        player_ids: Optional[List[str]] = None,
    ):
        self._dir = state_dir or COMBINER_DIR
        self._player_ids = player_ids or PLAYER_IDS

        # regime → {player_id → weight}
        self._weights: Dict[str, Dict[str, float]] = {}
        # regime → {player_id → cumulative decayed score}
        self._scores: Dict[str, Dict[str, float]] = {}
        # History for dashboard
        self._update_history: List[Dict[str, Any]] = []

        self._init_all_regimes()
        self._load()

    def get_weights(self, regime: str) -> Dict[str, float]:
        regime = self._norm_regime(regime)
        if regime not in self._weights:
            # Novel regime (e.g. "Correction") — initialize on the fly
            self._ensure_regime(regime)
        return dict(self._weights[regime])

    def update(self, outcome: TradeOutcome) -> Dict[str, float]:
        """Update weights after a trade resolves.

        Uses a soft Bayesian approach:
          1. Compute likelihood from trade correctness.
          2. Apply exponential decay to existing scores.
          3. Add new evidence.
          4. Convert scores to weights via softmax-like normalization.
          5. Clamp and re-normalize.

        Returns:
            Updated weight dict for the outcome's regime.
        """
        regime = self._norm_regime(outcome.regime)
        pid = outcome.player_id

        # Ensure regime exists (handles novel regimes like "Correction")
        if regime not in self._scores:
            self._ensure_regime(regime)

        if pid not in self._scores[regime]:
            return self.get_weights(regime)

        # 1. Compute likelihood: did the player's signal direction match PnL?
        signal_correct = (outcome.signal_direction > 0 and outcome.actual_pnl > 0) or \
                         (outcome.signal_direction < 0 and outcome.actual_pnl < 0)
        # Magnitude: how much PnL relative to a reference (₹200)
        magnitude = min(abs(outcome.actual_pnl) / 200.0, 3.0)
        likelihood = magnitude if signal_correct else -magnitude

        # 2. Decay all scores in this regime
        for p in self._scores[regime]:
            self._scores[regime][p] *= self.DECAY

        # 3. Add evidence for this player
        self._scores[regime][pid] += likelihood * self.LIKELIHOOD_SCALE

        # 4. Convert scores → weights via exp-normalize (softmax)
        self._scores_to_weights(regime)

        # 5. Record
        self._update_history.append({
            "timestamp": datetime.now().isoformat(),
            "player_id": pid,
            "regime": regime,
            "signal_direction": outcome.signal_direction,
            "actual_pnl": outcome.actual_pnl,
            "signal_correct": signal_correct,
            "new_weights": dict(self._weights[regime]),
        })

        self._save()
        return dict(self._weights[regime])

    def get_update_history(self, last_n: int = 50) -> List[Dict]:
        return self._update_history[-last_n:]

    # -- Internal --

    def _scores_to_weights(self, regime: str) -> None:
        scores = self._scores[regime]
        # Ensure all player_ids have a score (handles loaded data with missing players)
        for p in self._player_ids:
            if p not in scores:
                scores[p] = 0.0
        # Softmax-like: w_i = exp(s_i) / sum(exp(s_j))
        max_s = max(scores[p] for p in self._player_ids) if self._player_ids else 0.0
        exp_scores = {p: math.exp(scores[p] - max_s) for p in self._player_ids}
        total = sum(exp_scores.values())

        if total < 1e-9:
            n = len(self._player_ids)
            self._weights[regime] = {p: 1.0 / n for p in self._player_ids}
            return

        raw = {p: exp_scores[p] / total for p in self._player_ids}
        self._weights[regime] = self._clamp_and_normalize(raw)

    def _clamp_and_normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Clamp each weight to [MIN, MAX] then re-normalize to sum=1.

        Uses iterative projection that preserves bounds:
        fix players that exceed bounds, redistribute among free players.
        """
        result = dict(weights)
        lo, hi = self.MIN_WEIGHT, self.MAX_WEIGHT
        n = len(result)

        for _ in range(20):
            # Identify players that violate bounds
            fixed: Dict[str, float] = {}
            free: Dict[str, float] = {}
            for p, w in result.items():
                if w >= hi:
                    fixed[p] = hi
                elif w <= lo:
                    fixed[p] = lo
                else:
                    free[p] = w

            if not free:
                # All hit bounds. Distribute 1.0 proportionally but respecting bounds.
                # Assign hi to max player, lo to all others, then fill remainder
                # into mid-range players. For 5 players [0.05, 0.40]:
                # max feasible: 0.40 + 4*0.15 = 1.0 or 0.05*5=0.25 not enough...
                # We need a proportional split within bounds.
                raw_total = sum(weights.values())
                if raw_total < 1e-9:
                    return {p: 1.0 / n for p in result}
                proportional = {p: weights[p] / raw_total for p in result}
                # Clamp proportional and redo
                result = {p: max(lo, min(hi, w)) for p, w in proportional.items()}
                # Redistribute excess/deficit
                clamped_sum = sum(result.values())
                diff = 1.0 - clamped_sum
                # Distribute diff among all players that aren't at their limit
                # in the direction of diff
                adjustable = []
                for p in result:
                    if diff > 0 and result[p] < hi:
                        adjustable.append(p)
                    elif diff < 0 and result[p] > lo:
                        adjustable.append(p)
                if adjustable:
                    share = diff / len(adjustable)
                    for p in adjustable:
                        result[p] = max(lo, min(hi, result[p] + share))
                total = sum(result.values())
                if abs(total - 1.0) > 1e-6:
                    result = {p: w / total for p, w in result.items()}
                return result

            # Redistribute: free players share (1.0 - sum(fixed))
            fixed_sum = sum(fixed.values())
            target = 1.0 - fixed_sum
            free_sum = sum(free.values())

            if free_sum < 1e-9:
                share = target / len(free)
                for p in free:
                    result[p] = max(lo, min(hi, share))
            else:
                scale = target / free_sum
                for p in free:
                    result[p] = max(lo, min(hi, free[p] * scale))

            # Re-set fixed
            for p, w in fixed.items():
                result[p] = w

            # Check convergence
            total = sum(result.values())
            all_ok = all(lo - 1e-9 <= result[p] <= hi + 1e-9 for p in result)
            if abs(total - 1.0) < 1e-6 and all_ok:
                break

        return result

    def _ensure_regime(self, regime: str) -> None:
        """Ensure a regime has initialized weights and scores."""
        n = len(self._player_ids)
        if regime not in self._weights:
            self._weights[regime] = {p: 1.0 / n for p in self._player_ids}
        if regime not in self._scores:
            self._scores[regime] = {p: 0.0 for p in self._player_ids}

    def _init_all_regimes(self) -> None:
        for regime in REGIMES:
            self._ensure_regime(regime)

    @staticmethod
    def _norm_regime(regime: str) -> str:
        if not regime or not isinstance(regime, str):
            return "Sideways"
        return regime.strip().title()

    def _load(self) -> None:
        path = self._dir / "bayesian_weights.json"
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            n = len(self._player_ids)
            for regime in REGIMES:
                if regime in data.get("weights", {}):
                    loaded_w = data["weights"][regime]
                    # Backfill any missing player_ids (e.g. player added after persistence)
                    for p in self._player_ids:
                        if p not in loaded_w:
                            loaded_w[p] = 1.0 / n
                    self._weights[regime] = loaded_w
                if regime in data.get("scores", {}):
                    loaded_s = data["scores"][regime]
                    for p in self._player_ids:
                        if p not in loaded_s:
                            loaded_s[p] = 0.0
                    self._scores[regime] = loaded_s
            self._update_history = data.get("history", [])[-200:]
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"[BayesianWeights] Failed to load: {e}")

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / "bayesian_weights.json"
        data = {
            "weights": self._weights,
            "scores": self._scores,
            "history": self._update_history[-200:],
        }
        try:
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[BayesianWeights] Failed to save: {e}")


# ---------------------------------------------------------------------------
# AgreementAnalyzer
# ---------------------------------------------------------------------------

class AgreementAnalyzer:
    """Analyses consensus and dissent patterns among player signals."""

    # Threshold: a signal is considered "bullish" if direction > this
    DIRECTION_THRESHOLD = 0.0

    def analyze(self, signals: List[PlayerSignal]) -> AgreementResult:
        if not signals:
            return AgreementResult(
                agreement_score=0.0, pattern="no_majority",
                majority_direction=0.0, majority_count=0, dissent_count=0,
                dissenters=[], dissent_analysis="No signals provided.",
                should_trade=False, position_size_multiplier=0.0,
                crowding_flag=False,
            )

        n = len(signals)
        bullish = [s for s in signals if s.direction > self.DIRECTION_THRESHOLD]
        bearish = [s for s in signals if s.direction < -self.DIRECTION_THRESHOLD]
        neutral = [s for s in signals if abs(s.direction) <= self.DIRECTION_THRESHOLD]

        bull_count = len(bullish)
        bear_count = len(bearish)

        if bull_count > bear_count:
            majority_dir = 1.0
            majority_count = bull_count
            dissenters = [s.player_id for s in bearish + neutral]
        elif bear_count > bull_count:
            majority_dir = -1.0
            majority_count = bear_count
            dissenters = [s.player_id for s in bullish + neutral]
        else:
            # Tied or all neutral
            majority_dir = 0.0
            majority_count = max(bull_count, bear_count)
            dissenters = [s.player_id for s in signals]

        dissent_count = n - majority_count

        # Classify pattern
        if majority_count == n and n >= 2:
            pattern = "unanimous"
            agreement_score = 1.0
            size_mult = 1.0
            should_trade = True
            crowding = True
        elif majority_count >= n - 1 and majority_count >= 4:
            pattern = "strong"
            agreement_score = majority_count / n
            size_mult = 1.0
            should_trade = True
            crowding = False
        elif majority_count >= 3 and n >= 5:
            pattern = "moderate"
            agreement_score = majority_count / n
            size_mult = 0.80  # reduce by 20%
            should_trade = True
            crowding = False
        else:
            pattern = "no_majority"
            agreement_score = majority_count / n if n > 0 else 0.0
            size_mult = 0.0
            should_trade = False
            crowding = False

        # Build dissent analysis
        dissent_parts: List[str] = []
        for s in signals:
            if s.player_id in dissenters:
                label = PLAYER_LABELS.get(s.player_id, s.player_id)
                dir_str = "bullish" if s.direction > 0 else "bearish" if s.direction < 0 else "neutral"
                reason = s.reasoning or "no reasoning given"
                dissent_parts.append(
                    f"{label} ({s.player_id}): {dir_str} "
                    f"(dir={s.direction:+.2f}, conf={s.confidence:.2f}) — {reason}"
                )

        dissent_text = "; ".join(dissent_parts) if dissent_parts else "Full agreement."

        return AgreementResult(
            agreement_score=agreement_score,
            pattern=pattern,
            majority_direction=majority_dir,
            majority_count=majority_count,
            dissent_count=dissent_count,
            dissenters=dissenters,
            dissent_analysis=dissent_text,
            should_trade=should_trade,
            position_size_multiplier=size_mult,
            crowding_flag=crowding,
        )


# ---------------------------------------------------------------------------
# Signal combination logging
# ---------------------------------------------------------------------------

@dataclass
class CombinationLog:
    """Record of one signal-combination event."""

    timestamp: str
    regime: str
    signals: List[Dict[str, Any]]
    agreement: Dict[str, Any]
    player_weights: Dict[str, float]
    influence_map: Dict[str, float]
    final_signal: float
    position_size_multiplier: float
    should_trade: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "regime": self.regime,
            "signals": self.signals,
            "agreement": self.agreement,
            "player_weights": self.player_weights,
            "influence_map": self.influence_map,
            "final_signal": self.final_signal,
            "position_size_multiplier": self.position_size_multiplier,
            "should_trade": self.should_trade,
        }


class CombinationLogger:
    """Logs every signal combination and tracks accuracy."""

    def __init__(self, state_dir: Optional[Path] = None):
        self._dir = state_dir or COMBINER_DIR
        self._logs: List[CombinationLog] = []
        # Track combined signal accuracy:  list of (predicted_direction, actual_pnl, regime)
        self._accuracy_records: List[Dict[str, Any]] = []

    def log_combination(self, entry: CombinationLog) -> None:
        self._logs.append(entry)
        # Keep bounded
        if len(self._logs) > 500:
            self._logs = self._logs[-500:]

    def record_outcome(
        self,
        predicted_direction: float,
        actual_pnl: float,
        regime: str,
    ) -> None:
        # Neutral signal (direction=0) is never "correct" or "incorrect" — mark as neutral
        if abs(predicted_direction) < 1e-9:
            correct = False  # Neutral prediction — no directional call made
        else:
            correct = (predicted_direction > 0 and actual_pnl > 0) or \
                      (predicted_direction < 0 and actual_pnl < 0)
        self._accuracy_records.append({
            "predicted_direction": predicted_direction,
            "actual_pnl": actual_pnl,
            "regime": regime,
            "correct": correct,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self._accuracy_records) > 500:
            self._accuracy_records = self._accuracy_records[-500:]

    def rolling_accuracy(self, last_n: int = 50) -> float:
        records = self._accuracy_records[-last_n:]
        if not records:
            return 0.0
        return sum(1 for r in records if r["correct"]) / len(records)

    def regime_accuracy(self, regime: str, last_n: int = 50) -> float:
        records = [r for r in self._accuracy_records if r["regime"] == regime][-last_n:]
        if not records:
            return 0.0
        return sum(1 for r in records if r["correct"]) / len(records)

    def most_influential_player(self, last_n: int = 20) -> Optional[str]:
        recent = self._logs[-last_n:]
        if not recent:
            return None
        totals: Dict[str, float] = {}
        for entry in recent:
            for pid, infl in entry.influence_map.items():
                totals[pid] = totals.get(pid, 0.0) + abs(infl)
        if not totals:
            return None
        return max(totals, key=totals.get)  # type: ignore[arg-type]

    def least_influential_player(self, last_n: int = 20) -> Optional[str]:
        recent = self._logs[-last_n:]
        if not recent:
            return None
        totals: Dict[str, float] = {}
        for entry in recent:
            for pid, infl in entry.influence_map.items():
                totals[pid] = totals.get(pid, 0.0) + abs(infl)
        if not totals:
            return None
        return min(totals, key=totals.get)  # type: ignore[arg-type]

    def get_logs(self, last_n: int = 20) -> List[Dict]:
        return [log.to_dict() for log in self._logs[-last_n:]]

    def get_accuracy_records(self, last_n: int = 50) -> List[Dict]:
        return self._accuracy_records[-last_n:]

    def agreement_pattern_counts(self, last_n: int = 50) -> Dict[str, int]:
        recent = self._logs[-last_n:]
        counts: Dict[str, int] = {"unanimous": 0, "strong": 0, "moderate": 0, "no_majority": 0}
        for entry in recent:
            pat = entry.agreement.get("pattern", "no_majority")
            counts[pat] = counts.get(pat, 0) + 1
        return counts


# ---------------------------------------------------------------------------
# RegimeAwareCombiner — top-level orchestrator
# ---------------------------------------------------------------------------

class RegimeAwareCombiner:
    """Ties together all components for intelligent signal combination.

    Combines regime-based weights with Bayesian performance-based weights:
        final_player_weight = regime_blend * regime_weight + (1 - regime_blend) * bayesian_weight
    Then applies confidence weighting and agreement analysis.
    """

    def __init__(
        self,
        bayesian_weights: Optional[BayesianPlayerWeights] = None,
        state_dir: Optional[Path] = None,
        regime_blend: float = 0.5,
    ):
        self._dir = state_dir or COMBINER_DIR
        self._bayesian = bayesian_weights or BayesianPlayerWeights(state_dir=self._dir)
        self._voter = ConfidenceWeightedVoter()
        self._agreement = AgreementAnalyzer()
        self._logger = CombinationLogger(state_dir=self._dir)
        self._regime_blend = regime_blend
        self._previous_signal: float = 0.0  # Track last signal for flip threshold

    @property
    def bayesian_weights(self) -> BayesianPlayerWeights:
        return self._bayesian

    @property
    def logger(self) -> CombinationLogger:
        return self._logger

    @property
    def agreement_analyzer(self) -> AgreementAnalyzer:
        return self._agreement

    def combine(
        self,
        signals: List[PlayerSignal],
        regime: str,
        regime_weights: Optional[Dict[str, float]] = None,
    ) -> CombinedSignal:
        """Combine player signals into a final trading signal.

        Args:
            signals: One PlayerSignal per player.
            regime: Current market regime ("Bull", "Bear", "Sideways").
            regime_weights: Optional regime-based weights (player_id → weight).
                            If None, uses uniform 0.20.

        Returns:
            CombinedSignal with final_signal, position_size_multiplier, should_trade,
            and full signal_metadata.
        """
        regime = _norm_regime(regime)

        # 1. Agreement analysis
        agreement = self._agreement.analyze(signals)

        if not agreement.should_trade:
            return CombinedSignal(
                final_signal=0.0,
                position_size_multiplier=0.0,
                should_trade=False,
                signal_metadata={
                    "regime": regime,
                    "agreement": _agreement_to_dict(agreement),
                    "reason": "No clear majority — trade skipped.",
                },
            )

        # 2. Compute blended player weights
        bayesian_w = self._bayesian.get_weights(regime)
        if regime_weights is None:
            regime_w = {p: 0.20 for p in PLAYER_IDS}
        else:
            regime_w = regime_weights

        blended = {}
        rb = self._regime_blend
        for pid in PLAYER_IDS:
            rw = regime_w.get(pid, 0.20)
            bw = bayesian_w.get(pid, 0.20)
            blended[pid] = rb * rw + (1.0 - rb) * bw

        # Normalize blended weights
        total = sum(blended.values())
        if total > 0:
            blended = {p: w / total for p, w in blended.items()}

        # 3. Confidence-weighted voting
        final_signal, influence = self._voter.combine(signals, blended)

        # 4. Confidence filter: skip if signal too weak
        if abs(final_signal) < MIN_CONFIDENCE_THRESHOLD:
            self._previous_signal = final_signal
            return CombinedSignal(
                final_signal=final_signal,
                position_size_multiplier=0.0,
                should_trade=False,
                signal_metadata={
                    "regime": regime,
                    "agreement": _agreement_to_dict(agreement),
                    "reason": f"Signal {final_signal:.3f} below confidence threshold {MIN_CONFIDENCE_THRESHOLD}.",
                },
            )

        # 4b. Signal flip threshold: don't reverse unless change is large enough
        prev = self._previous_signal
        if prev != 0.0 and final_signal != 0.0:
            # Check if direction is flipping
            if (prev > 0 and final_signal < 0) or (prev < 0 and final_signal > 0):
                reversal_magnitude = abs(final_signal - prev)
                if reversal_magnitude < SIGNAL_FLIP_THRESHOLD:
                    return CombinedSignal(
                        final_signal=final_signal,
                        position_size_multiplier=0.0,
                        should_trade=False,
                        signal_metadata={
                            "regime": regime,
                            "agreement": _agreement_to_dict(agreement),
                            "reason": (
                                f"Signal flip {prev:.3f}→{final_signal:.3f} "
                                f"(reversal {reversal_magnitude:.3f}) below flip threshold {SIGNAL_FLIP_THRESHOLD}."
                            ),
                        },
                    )

        self._previous_signal = final_signal

        # 5. Position sizing: start with agreement multiplier
        size_mult = agreement.position_size_multiplier

        # 6. Build metadata
        metadata = {
            "regime": regime,
            "agreement": _agreement_to_dict(agreement),
            "bayesian_weights": bayesian_w,
            "regime_weights": dict(regime_w),
            "blended_weights": blended,
            "influence_map": influence,
            "crowding_flag": agreement.crowding_flag,
        }

        # 6. Log
        log_entry = CombinationLog(
            timestamp=datetime.now().isoformat(),
            regime=regime,
            signals=[
                {"player_id": s.player_id, "direction": s.direction,
                 "confidence": s.confidence, "reasoning": s.reasoning}
                for s in signals
            ],
            agreement=_agreement_to_dict(agreement),
            player_weights=blended,
            influence_map=influence,
            final_signal=final_signal,
            position_size_multiplier=size_mult,
            should_trade=True,
        )
        self._logger.log_combination(log_entry)

        return CombinedSignal(
            final_signal=final_signal,
            position_size_multiplier=size_mult,
            should_trade=True,
            signal_metadata=metadata,
        )

    def record_trade_outcome(self, outcome: TradeOutcome) -> None:
        """Record a resolved trade for both Bayesian updating and accuracy tracking."""
        self._bayesian.update(outcome)
        self._logger.record_outcome(
            predicted_direction=outcome.signal_direction,
            actual_pnl=outcome.actual_pnl,
            regime=outcome.regime,
        )


# ---------------------------------------------------------------------------
# CombinerDashboard
# ---------------------------------------------------------------------------

class CombinerDashboard:
    """Generates monitoring summaries for the signal combiner."""

    def __init__(self, combiner: RegimeAwareCombiner):
        self._combiner = combiner

    def summary(self) -> Dict[str, Any]:
        b = self._combiner.bayesian_weights
        lgr = self._combiner.logger

        # Per-regime Bayesian weights
        regime_weights = {}
        for regime in REGIMES:
            regime_weights[regime] = b.get_weights(regime)

        # Rolling accuracy
        accuracy = {
            "last_20": lgr.rolling_accuracy(20),
            "last_50": lgr.rolling_accuracy(50),
            "last_100": lgr.rolling_accuracy(100),
        }

        # Per-regime accuracy
        regime_accuracy = {}
        for regime in REGIMES:
            regime_accuracy[regime] = lgr.regime_accuracy(regime)

        # Agreement patterns
        patterns = lgr.agreement_pattern_counts(last_n=50)

        # Most / least influential
        most = lgr.most_influential_player(20)
        least = lgr.least_influential_player(20)

        return {
            "bayesian_weights_by_regime": regime_weights,
            "rolling_accuracy": accuracy,
            "regime_accuracy": regime_accuracy,
            "agreement_patterns_last_50": patterns,
            "most_influential_player": most,
            "most_influential_label": PLAYER_LABELS.get(most, "") if most else "",
            "least_influential_player": least,
            "least_influential_label": PLAYER_LABELS.get(least, "") if least else "",
            "total_combinations_logged": len(lgr._logs),
            "total_outcomes_tracked": len(lgr._accuracy_records),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm_regime(regime: str) -> str:
    if not regime or not isinstance(regime, str):
        return "Sideways"
    return regime.strip().title()


def _agreement_to_dict(a: AgreementResult) -> Dict[str, Any]:
    return {
        "agreement_score": a.agreement_score,
        "pattern": a.pattern,
        "majority_direction": a.majority_direction,
        "majority_count": a.majority_count,
        "dissent_count": a.dissent_count,
        "dissenters": a.dissenters,
        "dissent_analysis": a.dissent_analysis,
        "should_trade": a.should_trade,
        "position_size_multiplier": a.position_size_multiplier,
        "crowding_flag": a.crowding_flag,
    }
