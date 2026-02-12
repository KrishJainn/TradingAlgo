"""Standard interface that agents (players) must implement to interact with the risk engine.

Every agent in the multi-player coach system exposes its signal through the
``AgentSignal`` dataclass and conforms to the ``AgentInterface`` protocol.
This decouples the risk engine from any specific strategy implementation,
allowing new agents to be added without modifying downstream risk code.

The signal schema follows the alpha-and-variance decomposition from
Grinold & Kahn (1999), *Active Portfolio Management*, Chapter 6:

    Signal = direction * confidence * alpha_estimate

where:
    - ``direction`` encodes the sign and magnitude of the directional view,
    - ``confidence`` represents the agent's self-assessed conviction,
    - ``alpha_estimate`` is the expected excess return per bar.

Reference
---------
Grinold, R.C. & Kahn, R.N. (1999). *Active Portfolio Management*, 2nd ed.
    McGraw-Hill.  Chapter 6: Signal Weighting and Refinement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from risk_engine.config.constants import PLAYER_IDS, RISK_DEFAULTS


# ---------------------------------------------------------------------------
# Agent signal container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSignal:
    """Standardised signal emitted by an agent for a single bar.

    This dataclass captures every piece of information the risk engine
    needs to size, allocate, and constrain a position originating from
    a particular agent.

    Attributes
    ----------
    agent_id : str
        Unique identifier for the agent.  Must be one of the registered
        ``PLAYER_IDS`` from ``risk_engine.config.constants``.
    direction : float
        Directional view in [-1, +1].
        * +1.0 = maximum-conviction long.
        * -1.0 = maximum-conviction short.
        * 0.0  = flat / no opinion.
        Intermediate values express partial conviction.
    confidence : float
        Self-assessed confidence in [0, 1].  Multiplied with the raw
        Kelly fraction to shrink positions when the agent is unsure.
        A confidence of 0 means "do not trade"; 1 means "full conviction".
    alpha_estimate : float
        Expected excess return per bar (e.g. per 5-minute bar or per day).
        This is the *mu* input to the Kelly criterion:

            f* = mu / sigma^2

        (Kelly 1956; Thorp 2008).
    volatility_estimate : float
        Estimated standard deviation of the signal's return stream over
        the same horizon as ``alpha_estimate``.  This is the *sigma*
        input to Kelly sizing.  Must be strictly positive when
        ``alpha_estimate != 0``.
    metadata : dict[str, Any]
        Optional extra information for downstream consumers.  Common keys:
        * ``"regime"``: current market regime label (``"Bull"``/``"Bear"``/``"Sideways"``).
        * ``"strategy_type"``: strategy family (``"trend"``, ``"mean_rev"``, etc.).
        * ``"lookback"``: number of bars the signal was computed over.
        * ``"features"``: dict of feature values used for meta-labeling.

    Raises
    ------
    ValueError
        If ``direction`` is outside [-1, 1] or ``confidence`` is outside
        [0, 1] (enforced in ``__post_init__``).

    Example
    -------
    >>> signal = AgentSignal(
    ...     agent_id="PLAYER_1",
    ...     direction=0.85,
    ...     confidence=0.72,
    ...     alpha_estimate=0.0012,
    ...     volatility_estimate=0.018,
    ...     metadata={"regime": "Bull", "strategy_type": "trend"},
    ... )
    >>> signal.direction
    0.85
    """

    agent_id: str
    direction: float
    confidence: float
    alpha_estimate: float
    volatility_estimate: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate signal bounds after construction."""
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError(
                f"direction must be in [-1, 1], got {self.direction}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be in [0, 1], got {self.confidence}"
            )

    # -- Derived helpers ----------------------------------------------------

    @property
    def conviction_alpha(self) -> float:
        """Return the confidence-scaled, direction-adjusted alpha.

        Formula
        -------
            conviction_alpha = direction * confidence * alpha_estimate

        This is the effective alpha fed into the position sizer after
        accounting for directional view and self-assessed confidence.
        """
        return self.direction * self.confidence * self.alpha_estimate

    @property
    def variance_estimate(self) -> float:
        """Return the variance of the signal (sigma^2).

        Used as the denominator in the Kelly criterion:

            f* = mu / sigma^2
        """
        return self.volatility_estimate ** 2

    @property
    def is_actionable(self) -> bool:
        """Check whether the signal warrants a trade.

        A signal is considered actionable when:
        1. The agent has a non-trivial directional view (|direction| > 0.05).
        2. Confidence exceeds a minimum threshold (> 0.1).
        3. The alpha estimate exceeds zero.
        4. Volatility estimate is strictly positive.

        Returns
        -------
        bool
            True if the signal passes all actionability checks.
        """
        return (
            abs(self.direction) > 0.05
            and self.confidence > 0.1
            and abs(self.alpha_estimate) > 0.0
            and self.volatility_estimate > 0.0
        )


# ---------------------------------------------------------------------------
# Agent protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentInterface(Protocol):
    """Protocol that every trading agent must satisfy.

    The risk engine interacts with agents exclusively through this
    interface.  Any class that provides the required properties and
    methods is considered a valid agent, regardless of its inheritance
    hierarchy (structural sub-typing).

    Using ``Protocol`` (PEP 544) rather than an abstract base class
    keeps the coupling minimal: agents do not need to import or
    inherit from anything in the risk engine package.

    Required API
    ------------
    * ``agent_id`` property  -- unique string identifier.
    * ``generate_signal(data)`` -- produce an ``AgentSignal`` from market data.
    * ``get_returns(lookback)`` -- historical return series for performance tracking.
    * ``get_sharpe(lookback)`` -- rolling Sharpe ratio for health monitoring.
    * ``is_active()`` -- whether the agent is currently enabled.

    Example
    -------
    >>> class MyAgent:
    ...     @property
    ...     def agent_id(self) -> str:
    ...         return "PLAYER_1"
    ...     def generate_signal(self, data: pd.DataFrame) -> AgentSignal: ...
    ...     def get_returns(self, lookback: int = 60) -> np.ndarray: ...
    ...     def get_sharpe(self, lookback: int = 60) -> float: ...
    ...     def is_active(self) -> bool: ...
    >>> isinstance(MyAgent(), AgentInterface)
    True
    """

    @property
    def agent_id(self) -> str:
        """Unique identifier for this agent.

        Must correspond to one of the entries in
        ``risk_engine.config.constants.PLAYER_IDS``.

        Returns
        -------
        str
            Agent identifier string (e.g. ``"PLAYER_1"``).
        """
        ...

    def generate_signal(self, data: pd.DataFrame) -> AgentSignal:
        """Generate a trading signal from current market data.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with at least the columns ``Open``, ``High``,
            ``Low``, ``Close``, ``Volume``, plus any computed indicators.
            Index is a ``DatetimeIndex``.

        Returns
        -------
        AgentSignal
            The agent's current directional view, confidence, and alpha
            forecast packed into a standardised signal container.
        """
        ...

    def get_returns(self, lookback: int = 60) -> np.ndarray:
        """Return the agent's historical signal-return series.

        The risk engine uses this series for:
        * Rolling Sharpe ratio computation (health monitoring).
        * Information coefficient (IC) tracking.
        * Covariance estimation across agents (HRP allocation).
        * Online allocator weight updates.

        Parameters
        ----------
        lookback : int
            Number of most-recent bars to include.  Defaults to
            ``RISK_DEFAULTS["vol_lookback"]`` (60).

        Returns
        -------
        np.ndarray
            1-D array of per-bar returns, length <= ``lookback``.
            Returns an empty array if the agent has no history.
        """
        ...

    def get_sharpe(self, lookback: int = 60) -> float:
        """Compute the agent's rolling Sharpe ratio.

        Formula
        -------
            SR = mean(r) / std(r) * sqrt(annualisation_factor)

        where ``r`` is the vector of per-bar returns over the lookback
        window, and ``annualisation_factor`` converts per-bar Sharpe to
        an annualised figure.

        The Sharpe ratio is the primary input to the agent health check
        in ``Constraints.check_agent_health``.  If it falls below
        ``RISK_DEFAULTS["min_agent_sharpe"]`` (default -1.0), the agent
        is disabled.

        Parameters
        ----------
        lookback : int
            Number of bars for the rolling window.

        Returns
        -------
        float
            Annualised Sharpe ratio.  Returns 0.0 if insufficient data
            or zero-variance returns.

        Reference
        ---------
        Sharpe, W.F. (1966). "Mutual Fund Performance." *Journal of
        Business*, 39(1), 119-138.
        """
        ...

    def is_active(self) -> bool:
        """Check whether the agent is currently enabled for trading.

        An inactive agent's signal is excluded from the aggregation
        pipeline.  Agents may be deactivated by:
        * The constraint engine (Sharpe kill switch).
        * Manual override from the coach.
        * Self-disabling logic (e.g. regime mismatch).

        Returns
        -------
        bool
            True if the agent is active and should be included in the
            signal aggregation pipeline.
        """
        ...
