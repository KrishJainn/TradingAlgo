"""
Configurable risk parameters — runtime-tunable knobs.

Wraps RISK_DEFAULTS into a validated dataclass so consumers get
attribute access with type safety and boundary checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

from .constants import RISK_DEFAULTS


@dataclass
class RiskParams:
    """Runtime-tunable risk parameters with validation.

    All defaults come from ``RISK_DEFAULTS`` in constants.py.
    Override any subset via constructor kwargs or ``from_dict()``.
    """

    kelly_fraction: float = RISK_DEFAULTS["kelly_fraction"]
    max_single_position_pct: float = RISK_DEFAULTS["max_single_position_pct"]
    max_net_exposure: float = RISK_DEFAULTS["max_net_exposure"]
    max_gross_exposure: float = RISK_DEFAULTS["max_gross_exposure"]
    max_daily_loss_pct: float = RISK_DEFAULTS["max_daily_loss_pct"]
    max_drawdown_pct: float = RISK_DEFAULTS["max_drawdown_pct"]
    ewma_lambda: float = RISK_DEFAULTS["ewma_lambda"]
    vol_lookback: int = RISK_DEFAULTS["vol_lookback"]
    ic_lookback: int = RISK_DEFAULTS["ic_lookback"]
    cost_safety_margin: float = RISK_DEFAULTS["cost_safety_margin"]
    min_agent_sharpe: float = RISK_DEFAULTS["min_agent_sharpe"]
    correlation_lookback: int = RISK_DEFAULTS["correlation_lookback"]
    rebalance_frequency: str = RISK_DEFAULTS["rebalance_frequency"]
    tactical_blend: float = RISK_DEFAULTS["tactical_blend"]
    conformal_coverage: float = RISK_DEFAULTS["conformal_coverage"]
    meta_label_purge: int = RISK_DEFAULTS["meta_label_purge"]

    def __post_init__(self) -> None:
        assert 0.0 < self.kelly_fraction <= 1.0, "kelly_fraction must be in (0, 1]"
        assert 0.0 < self.max_single_position_pct <= 1.0
        assert 0.0 < self.max_net_exposure <= 5.0
        assert 0.0 < self.max_gross_exposure <= 10.0
        assert 0.0 < self.max_daily_loss_pct <= 0.50
        assert 0.0 < self.max_drawdown_pct <= 1.0
        assert 0.5 <= self.ewma_lambda < 1.0
        assert self.vol_lookback >= 10
        assert self.cost_safety_margin >= 1.0
        assert 0.0 < self.conformal_coverage < 1.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RiskParams":
        """Create from a dict, ignoring unknown keys."""
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {k: getattr(self, k) for k in self.__dataclass_fields__}
