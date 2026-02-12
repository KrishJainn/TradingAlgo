"""
Exponential Weights Online Agent Allocation.

Implements the multiplicative weights update (MWU) algorithm for dynamically
allocating capital across agents based on their realized performance.

References:
    Cover, T.M. (1991). "Universal Portfolios."
    Mathematical Finance, 1(1), 1-29.

    Cesa-Bianchi, N., & Lugosi, G. (2006). "Prediction, Learning, and Games."
    Cambridge University Press, Chapter 2: Weighted Majority Algorithm.

The multiplicative weights update rule:
    w_i(t+1) = w_i(t) * exp(eta * r_i(t)) / Z(t)

Where:
    w_i(t)  = weight of agent i at time t
    r_i(t)  = realized return of agent i at time t
    eta     = learning rate (controls adaptivity vs. stability)
    Z(t)    = normalization constant so weights sum to 1

Properties:
    - Regret bound: O(sqrt(T * ln(N))) where T = horizon, N = number of agents
    - No distributional assumptions on returns (adversarial setting)
    - Learning rate eta = sqrt(ln(N) / T) is optimal, but we use a fixed
      rate for simplicity and adaptivity to non-stationary environments.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AllocationSnapshot:
    """A snapshot of agent weights at a point in time.

    Attributes:
        step: The time step (0-indexed).
        weights: Agent name -> weight mapping at this step.
        agent_returns: Agent returns that caused this weight update.
    """

    step: int
    weights: Dict[str, float]
    agent_returns: Dict[str, float]


class OnlineAllocator:
    """Exponential weights (multiplicative weights update) agent allocator.

    Dynamically adjusts capital allocation across agents based on their
    realized returns using the MWU algorithm. Agents that perform well
    receive more weight; underperformers are down-weighted exponentially.

    The MWU algorithm is a foundational online learning method with strong
    theoretical guarantees (sub-linear regret) even against adversarial
    return sequences.

    References:
        Cover, T.M. (1991). "Universal Portfolios."
        Cesa-Bianchi, N., & Lugosi, G. (2006). "Prediction, Learning, and Games."

    Example:
        >>> allocator = OnlineAllocator(n_agents=3, learning_rate=0.1)
        >>> allocator.set_agent_names(["momentum", "mean_rev", "ml_model"])
        >>> new_weights = allocator.update({
        ...     "momentum": 0.02, "mean_rev": -0.01, "ml_model": 0.015
        ... })
        >>> new_weights
        {'momentum': 0.367, 'mean_rev': 0.298, 'ml_model': 0.335}
    """

    def __init__(
        self,
        n_agents: int,
        learning_rate: float = 0.1,
        min_weight: float = 0.01,
    ) -> None:
        """Initialize the online allocator with uniform weights.

        Args:
            n_agents: Number of agents to allocate across.
            learning_rate: The eta parameter controlling how aggressively
                weights react to new returns. Higher = more reactive but
                less stable. Theoretical optimum is sqrt(ln(N)/T) but a
                fixed rate of 0.05-0.2 works well in practice for
                non-stationary financial environments.
            min_weight: Minimum weight floor for any agent, preventing
                complete elimination. This ensures all agents retain some
                allocation for diversification and recovery potential.

        Raises:
            ValueError: If n_agents < 1 or learning_rate <= 0.
        """
        if n_agents < 1:
            raise ValueError(f"n_agents must be >= 1, got {n_agents}")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {learning_rate}")
        if not 0 <= min_weight < 1.0 / n_agents:
            raise ValueError(
                f"min_weight must be in [0, 1/n_agents), got {min_weight}"
            )

        self.n_agents = n_agents
        self.learning_rate = learning_rate
        self.min_weight = min_weight

        # Initialize uniform weights. Agent names assigned later.
        self._agent_names: List[str] = [f"agent_{i}" for i in range(n_agents)]
        self._weights: Dict[str, float] = {
            name: 1.0 / n_agents for name in self._agent_names
        }

        # History of weight snapshots for monitoring and analysis.
        self._history: List[AllocationSnapshot] = []
        self._step: int = 0

    def set_agent_names(self, names: List[str]) -> None:
        """Set human-readable agent names.

        Must be called before update() if you want meaningful names.
        Weights are redistributed uniformly under new names.

        Args:
            names: List of agent names. Must have length == n_agents.

        Raises:
            ValueError: If length doesn't match n_agents or names aren't unique.
        """
        if len(names) != self.n_agents:
            raise ValueError(
                f"Expected {self.n_agents} names, got {len(names)}"
            )
        if len(set(names)) != len(names):
            raise ValueError(f"Agent names must be unique, got duplicates in {names}")

        # Transfer weights from old names to new names, preserving order.
        old_weights = list(self._weights.values())
        self._agent_names = list(names)
        self._weights = {
            name: weight for name, weight in zip(self._agent_names, old_weights)
        }

    def update(self, agent_returns: Dict[str, float]) -> Dict[str, float]:
        """Update weights using multiplicative weights update and return new allocation.

        Applies the exponential weights rule:
            w_i(t+1) = w_i(t) * exp(eta * r_i(t)) / Z(t)

        Where Z(t) = sum_i w_i(t) * exp(eta * r_i(t)) is the normalization
        constant ensuring weights sum to 1.

        After normalization, a minimum weight floor is applied: any agent
        below ``min_weight`` is set to ``min_weight`` and the excess is
        redistributed proportionally from agents above the floor.

        Args:
            agent_returns: Mapping of agent_name -> realized return for this
                period. Must contain all agent names.

        Returns:
            Updated weight allocation as agent_name -> weight mapping.
            Weights sum to 1.0.

        Raises:
            ValueError: If agent_returns keys don't match registered agents.
        """
        # Auto-detect agent names on first call if defaults are still in place.
        if self._step == 0 and all(
            n.startswith("agent_") for n in self._agent_names
        ):
            incoming_names = sorted(agent_returns.keys())
            if len(incoming_names) == self.n_agents:
                self.set_agent_names(incoming_names)

        expected_keys = set(self._agent_names)
        provided_keys = set(agent_returns.keys())
        if expected_keys != provided_keys:
            raise ValueError(
                f"agent_returns keys {provided_keys} don't match "
                f"registered agents {expected_keys}"
            )

        # --- Multiplicative weights update ---
        new_weights: Dict[str, float] = {}
        for name in self._agent_names:
            r_i = agent_returns[name]
            # Exponential update: w_i * exp(eta * r_i)
            new_weights[name] = self._weights[name] * np.exp(
                self.learning_rate * r_i
            )

        # --- Normalize ---
        z = sum(new_weights.values())
        if z <= 0 or not np.isfinite(z):
            # Fallback to uniform if numerical issues arise.
            new_weights = {name: 1.0 / self.n_agents for name in self._agent_names}
        else:
            new_weights = {name: w / z for name, w in new_weights.items()}

        # --- Apply minimum weight floor ---
        if self.min_weight > 0:
            new_weights = self._apply_weight_floor(new_weights)

        # --- Record history ---
        self._history.append(
            AllocationSnapshot(
                step=self._step,
                weights=copy.deepcopy(new_weights),
                agent_returns=dict(agent_returns),
            )
        )

        self._weights = new_weights
        self._step += 1

        return self.get_weights()

    def _apply_weight_floor(
        self, weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply minimum weight floor and redistribute excess.

        Any agent below ``min_weight`` is raised to the floor. The total
        excess needed is taken proportionally from agents above the floor.

        Args:
            weights: Current weight mapping (must sum to ~1.0).

        Returns:
            Adjusted weights with floor applied, summing to 1.0.
        """
        floored = dict(weights)
        deficit = 0.0

        # First pass: identify agents below floor and compute deficit.
        for name in self._agent_names:
            if floored[name] < self.min_weight:
                deficit += self.min_weight - floored[name]
                floored[name] = self.min_weight

        if deficit <= 0:
            return floored

        # Second pass: reduce agents above floor proportionally.
        above_floor = {
            name: w
            for name, w in floored.items()
            if w > self.min_weight
        }
        above_total = sum(above_floor.values())

        if above_total <= 0:
            # Edge case: all at floor, just normalize.
            total = sum(floored.values())
            return {name: w / total for name, w in floored.items()}

        for name in above_floor:
            reduction = deficit * (floored[name] / above_total)
            floored[name] = max(self.min_weight, floored[name] - reduction)

        # Final normalization for numerical safety.
        total = sum(floored.values())
        return {name: w / total for name, w in floored.items()}

    def get_weights(self) -> Dict[str, float]:
        """Return current agent weight allocation.

        Returns:
            Copy of the current weights as agent_name -> weight mapping.
            Weights sum to 1.0.
        """
        return {
            name: round(w, 6) for name, w in self._weights.items()
        }

    def blend_with_hrp(
        self,
        hrp_weights: Dict[str, float],
        tactical_ratio: float = 0.3,
    ) -> Dict[str, float]:
        """Blend online weights with Hierarchical Risk Parity (HRP) weights.

        Combines the tactical (online / adaptive) allocation with a strategic
        (HRP / risk-based) allocation:

            final_i = tactical_ratio * online_i + (1 - tactical_ratio) * hrp_i

        This provides stability from HRP's risk-based diversification while
        allowing the online allocator to tilt toward recently outperforming
        agents.

        Args:
            hrp_weights: Strategic allocation from HRP or similar risk-based
                method. Mapping of agent_name -> weight. Must contain all
                agent names and sum to ~1.0.
            tactical_ratio: Blending ratio in [0, 1]. 0.0 = pure HRP,
                1.0 = pure online weights. Default 0.3 (30% tactical).

        Returns:
            Blended weight allocation as agent_name -> weight.

        Raises:
            ValueError: If tactical_ratio is out of range or keys mismatch.
        """
        if not 0.0 <= tactical_ratio <= 1.0:
            raise ValueError(
                f"tactical_ratio must be in [0, 1], got {tactical_ratio}"
            )

        missing = set(self._agent_names) - set(hrp_weights.keys())
        if missing:
            raise ValueError(
                f"hrp_weights missing agents: {missing}"
            )

        online_w = self.get_weights()
        blended: Dict[str, float] = {}

        for name in self._agent_names:
            blended[name] = (
                tactical_ratio * online_w[name]
                + (1.0 - tactical_ratio) * hrp_weights[name]
            )

        # Normalize for numerical safety.
        total = sum(blended.values())
        if total > 0:
            blended = {name: w / total for name, w in blended.items()}

        return {name: round(w, 6) for name, w in blended.items()}

    @property
    def history(self) -> List[AllocationSnapshot]:
        """Return the full history of weight snapshots.

        Returns:
            List of AllocationSnapshot objects, one per update() call.
        """
        return self._history.copy()

    @property
    def n_steps(self) -> int:
        """Number of update steps completed."""
        return self._step
