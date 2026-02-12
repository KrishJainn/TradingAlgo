"""
risk_engine — Quantitative risk management for the 5-player coach system.

Implements the Fundamental Law of Active Management:

    IR = IC × √BR × TC

where:
    IR  = Information Ratio (risk-adjusted alpha)
    IC  = Information Coefficient (signal quality)
    BR  = Breadth (independent bets per year)
    TC  = Transfer Coefficient (implementation efficiency)

Reference: Grinold & Kahn (1999) *Active Portfolio Management*, 2nd ed.

Modules
-------
config      : Constants, NSE cost tables, tunable risk parameters.
core        : Mathematical building blocks (Kelly, HRP, EWMA, copulas, …).
validation  : Statistical tests (Deflated Sharpe, IC tracker, PBO).
integration : Standard interfaces for agents and the coach.
dashboard   : Real-time metrics for the Streamlit front-end.
"""

from __future__ import annotations

# ── Config ──────────────────────────────────────────────────────────────
from risk_engine.config.constants import (
    NSE_COSTS,
    NSE_MICROSTRUCTURE,
    RISK_DEFAULTS,
    PLAYER_IDS,
    PLAYER_LABELS,
    REGIMES,
    TRADING_DAYS_PER_YEAR,
    ANNUALIZATION_FACTOR,
    RISK_FREE_RATE_ANNUAL,
    RISK_FREE_RATE_DAILY,
)
from risk_engine.config.risk_params import RiskParams

# ── Core ────────────────────────────────────────────────────────────────
from risk_engine.core.alpha_aggregator import AlphaAggregator
from risk_engine.core.risk_estimator import RiskEstimator
from risk_engine.core.position_sizer import PositionSizer, PositionSizeResult
from risk_engine.core.risk_allocator import RiskAllocator, RiskBudget
from risk_engine.core.cost_model import CostModel, CostBreakdown
from risk_engine.core.constraints import Constraints, ConstraintResult
from risk_engine.core.disagreement_signal import DisagreementSignal, DisagreementResult
from risk_engine.core.meta_labeler import MetaLabeler, MetaLabelResult
from risk_engine.core.entropy_pooling import EntropyPooling, EntropyPoolingResult
from risk_engine.core.online_allocator import OnlineAllocator
from risk_engine.core.tail_dependence import TailDependence, TailDependenceResult
from risk_engine.core.conformal_predictor import ConformalPredictor, ConformalResult

# ── Validation ──────────────────────────────────────────────────────────
from risk_engine.validation.deflated_sharpe import DeflatedSharpe, DSRResult
from risk_engine.validation.ic_tracker import ICTracker, ICResult
from risk_engine.validation.pbo import PBO, PBOResult

# ── Integration ─────────────────────────────────────────────────────────
from risk_engine.integration.agent_interface import AgentInterface, AgentSignal
from risk_engine.integration.coach_interface import CoachInterface, CoachDecision

# ── Dashboard ───────────────────────────────────────────────────────────
from risk_engine.dashboard.metrics import RiskDashboardMetrics

__all__ = [
    # Config
    "NSE_COSTS",
    "NSE_MICROSTRUCTURE",
    "RISK_DEFAULTS",
    "PLAYER_IDS",
    "PLAYER_LABELS",
    "REGIMES",
    "TRADING_DAYS_PER_YEAR",
    "ANNUALIZATION_FACTOR",
    "RISK_FREE_RATE_ANNUAL",
    "RISK_FREE_RATE_DAILY",
    "RiskParams",
    # Core
    "AlphaAggregator",
    "RiskEstimator",
    "PositionSizer",
    "PositionSizeResult",
    "RiskAllocator",
    "RiskBudget",
    "CostModel",
    "CostBreakdown",
    "Constraints",
    "ConstraintResult",
    "DisagreementSignal",
    "DisagreementResult",
    "MetaLabeler",
    "MetaLabelResult",
    "EntropyPooling",
    "EntropyPoolingResult",
    "OnlineAllocator",
    "TailDependence",
    "TailDependenceResult",
    "ConformalPredictor",
    "ConformalResult",
    # Validation
    "DeflatedSharpe",
    "DSRResult",
    "ICTracker",
    "ICResult",
    "PBO",
    "PBOResult",
    # Integration
    "AgentInterface",
    "AgentSignal",
    "CoachInterface",
    "CoachDecision",
    # Dashboard
    "RiskDashboardMetrics",
]
