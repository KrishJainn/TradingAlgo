"""
Agent Disagreement Entropy Signal.

Measures how much agents disagree on market direction using Shannon entropy
of the direction distribution and confidence-weighted dispersion.

Reference:
    Hong, H., & Stein, J.C. (2007). "Disagreement and the Stock Market."
    Journal of Economic Perspectives, 21(2), 109-128.

When agents strongly disagree, it indicates genuine uncertainty in the market.
High disagreement should reduce position sizing as a risk management measure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class DisagreementResult:
    """Result of the disagreement entropy computation.

    Attributes:
        entropy: Shannon entropy of the direction bin distribution.
            Range [0, log(n_bins)]. Higher means more disagreement.
        dispersion: Standard deviation of direction_i * confidence_i.
            Captures confidence-weighted spread of opinions.
        n_clusters: Number of non-empty direction bins occupied by agents.
        is_high_disagreement: True when entropy > 0.7 * max_entropy,
            indicating agents have fundamentally divergent views.
        position_size_multiplier: Risk scaling factor. 0.5 when high
            disagreement (halve exposure), 1.0 otherwise.
    """

    entropy: float
    dispersion: float
    n_clusters: int
    is_high_disagreement: bool
    position_size_multiplier: float


class DisagreementSignal:
    """Computes agent disagreement using Shannon entropy and dispersion.

    The signal quantifies how much a panel of agents disagree on market
    direction. Disagreement is measured two ways:

    1. **Shannon Entropy** of the direction-bin distribution:
       H = -sum(p_i * log(p_i)) for each occupied bin.
       Bins partition the direction space into five regimes:
         - strong_bull:  direction > 0.5
         - mild_bull:    0.0 <= direction <= 0.5
         - neutral:      -0.1 < direction < 0.0  (overlap resolved by order)
         - mild_bear:    -0.5 <= direction < -0.1
         - strong_bear:  direction < -0.5

       Note: the neutral band is narrow (-0.1, 0.1) by design — most agents
       should have a directional opinion.

    2. **Confidence-weighted dispersion**: std(direction_i * confidence_i),
       capturing how spread out the conviction-adjusted signals are.

    Reference:
        Hong, H., & Stein, J.C. (2007). "Disagreement and the Stock Market."
        Journal of Economic Perspectives, 21(2), 109-128.

    Example:
        >>> signal = DisagreementSignal()
        >>> directions = {"momentum": 0.8, "mean_rev": -0.6, "ml": 0.2}
        >>> confidences = {"momentum": 0.9, "mean_rev": 0.7, "ml": 0.5}
        >>> result = signal.compute(directions, confidences)
        >>> result.is_high_disagreement
        True
    """

    # Direction bins: (name, lower_bound_inclusive, upper_bound_exclusive)
    # Order matters — first match wins for boundary values.
    BINS: List[tuple[str, float, float]] = [
        ("strong_bear", -float("inf"), -0.5),
        ("mild_bear", -0.5, -0.1),
        ("neutral", -0.1, 0.1),
        ("mild_bull", 0.1, 0.5),
        ("strong_bull", 0.5, float("inf")),
    ]

    # Number of bins determines maximum possible entropy.
    N_BINS: int = len(BINS)
    MAX_ENTROPY: float = math.log(N_BINS)

    # Threshold ratio for high-disagreement flag (Hong & Stein recommend
    # being conservative — 70% of max entropy is a reasonable cutoff).
    HIGH_DISAGREEMENT_RATIO: float = 0.7

    def _classify_direction(self, direction: float) -> str:
        """Assign a direction value to one of the five bins.

        Args:
            direction: Scalar direction signal, typically in [-1, 1].

        Returns:
            Bin name string.
        """
        for name, lo, hi in self.BINS:
            if lo <= direction < hi:
                return name
        # Edge case: direction == +inf or exactly upper bound of last bin.
        return "strong_bull"

    def _compute_entropy(self, bin_counts: Dict[str, int], n_total: int) -> float:
        """Compute Shannon entropy from bin counts.

        H = -sum(p_i * log(p_i)) where p_i = count_i / n_total.
        Convention: 0 * log(0) = 0.

        Args:
            bin_counts: Mapping of bin name to agent count.
            n_total: Total number of agents.

        Returns:
            Shannon entropy in nats (natural log).
        """
        if n_total == 0:
            return 0.0

        entropy = 0.0
        for count in bin_counts.values():
            if count > 0:
                p_i = count / n_total
                entropy -= p_i * math.log(p_i)
        return entropy

    def compute(
        self,
        directions: Dict[str, float],
        confidences: Dict[str, float],
    ) -> DisagreementResult:
        """Compute disagreement metrics across agents.

        Args:
            directions: Mapping of agent_name -> direction signal.
                Direction is a float where positive = bullish, negative = bearish.
                Typical range [-1, 1] but not enforced.
            confidences: Mapping of agent_name -> confidence in [0, 1].
                Must have the same keys as ``directions``.

        Returns:
            DisagreementResult with entropy, dispersion, cluster count,
            high-disagreement flag, and position size multiplier.

        Raises:
            ValueError: If directions and confidences have different keys,
                or if either is empty.
        """
        if not directions:
            raise ValueError("directions must be non-empty")
        if set(directions.keys()) != set(confidences.keys()):
            raise ValueError(
                "directions and confidences must have identical keys. "
                f"directions keys: {set(directions.keys())}, "
                f"confidences keys: {set(confidences.keys())}"
            )

        agent_names = list(directions.keys())
        n_agents = len(agent_names)

        # --- Step 1: Bin each agent's direction ---
        bin_counts: Dict[str, int] = {name: 0 for name, _, _ in self.BINS}
        for agent in agent_names:
            bin_name = self._classify_direction(directions[agent])
            bin_counts[bin_name] += 1

        # --- Step 2: Shannon entropy of the bin distribution ---
        entropy = self._compute_entropy(bin_counts, n_agents)

        # --- Step 3: Number of non-empty clusters ---
        n_clusters = sum(1 for c in bin_counts.values() if c > 0)

        # --- Step 4: Confidence-weighted dispersion ---
        # dispersion = std(direction_i * confidence_i)
        weighted_signals = np.array(
            [directions[a] * confidences[a] for a in agent_names],
            dtype=np.float64,
        )
        dispersion = float(np.std(weighted_signals, ddof=0))

        # --- Step 5: High disagreement flag ---
        threshold = self.HIGH_DISAGREEMENT_RATIO * self.MAX_ENTROPY
        is_high_disagreement = entropy > threshold

        # --- Step 6: Position size multiplier ---
        # Halve exposure under high disagreement (risk-off).
        position_size_multiplier = 0.5 if is_high_disagreement else 1.0

        return DisagreementResult(
            entropy=round(entropy, 6),
            dispersion=round(dispersion, 6),
            n_clusters=n_clusters,
            is_high_disagreement=is_high_disagreement,
            position_size_multiplier=position_size_multiplier,
        )
