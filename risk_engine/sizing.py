"""
Sizing orchestrator — thin convenience layer over ``CoachInterface``.

Provides a single function-call API for the most common use-case:
given agent signals and market data, return sized positions.

This module exists so that callers who do not need fine-grained control
over every risk-engine component can get a result in one call:

    from risk_engine.sizing import size_positions
    decision = size_positions(signals, market_data, portfolio_value)

Under the hood it delegates to
:class:`~risk_engine.integration.coach_interface.CoachInterface`.

Reference
---------
Grinold, R.C. & Kahn, R.N. (1999). *Active Portfolio Management*, 2nd ed.
    IR = IC × √BR × TC  (Fundamental Law of Active Management)
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from risk_engine.config.risk_params import RiskParams
from risk_engine.integration.agent_interface import AgentSignal
from risk_engine.integration.coach_interface import CoachDecision, CoachInterface

logger = logging.getLogger(__name__)

# Module-level singleton — lazily initialised on first call.
_coach: CoachInterface | None = None


def _get_coach(params: RiskParams | None = None) -> CoachInterface:
    """Return (and optionally re-create) the module-level CoachInterface."""
    global _coach
    if _coach is None or params is not None:
        _coach = CoachInterface(params=params)
    return _coach


def size_positions(
    signals: list[AgentSignal],
    market_data: pd.DataFrame,
    portfolio_value: float,
    *,
    current_positions: dict[str, float] | None = None,
    params: RiskParams | None = None,
) -> CoachDecision:
    """One-call convenience: signals → sized positions.

    Parameters
    ----------
    signals : list[AgentSignal]
        Agent signals with direction, confidence, and alpha estimates.
    market_data : pd.DataFrame
        Recent OHLCV data (needs a ``"Close"`` column at minimum).
    portfolio_value : float
        Current portfolio NAV in INR.
    current_positions : dict[str, float] | None
        Existing position sizes keyed by instrument.
    params : RiskParams | None
        Override default risk parameters.  Passing a new ``RiskParams``
        re-initialises the internal ``CoachInterface``.

    Returns
    -------
    CoachDecision
        Full decision object including weights, sizes, costs, and
        constraint metadata.
    """
    coach = _get_coach(params)
    return coach.process_signals(
        signals=signals,
        market_data=market_data,
        portfolio_value=portfolio_value,
        current_positions=current_positions,
    )


def update_weights(agent_returns: dict[str, float]) -> None:
    """Update online multiplicative weights with latest agent returns.

    Parameters
    ----------
    agent_returns : dict[str, float]
        Most recent return for each agent (keyed by agent_id).
    """
    coach = _get_coach()
    coach.update_online_weights(agent_returns)


def get_risk_report() -> dict[str, Any]:
    """Return the current risk state from the internal CoachInterface."""
    coach = _get_coach()
    return coach.get_risk_report()


def reset(params: RiskParams | None = None) -> None:
    """Reset the module-level CoachInterface (useful for tests)."""
    global _coach
    _coach = CoachInterface(params=params) if params else None
