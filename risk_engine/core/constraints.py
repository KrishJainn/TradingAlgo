"""
Position limits, exposure constraints, and kill switches.

Implements a layered constraint system that checks every proposed trade
against configurable risk limits before execution. Constraints are
evaluated in order of severity:

    1. **Kill switches** (daily loss, agent health) — hard stops that
       immediately halt trading for the session or disable an agent.
    2. **Drawdown gates** — soft stops that reduce position sizing.
    3. **Exposure limits** — gross and net exposure caps.
    4. **Position limits** — per-instrument concentration caps.

All thresholds are sourced from ``RiskParams``, which in turn defaults
to the values in ``risk_engine.config.constants.RISK_DEFAULTS``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from risk_engine.config.risk_params import RiskParams

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ConstraintResult:
    """Outcome of applying all constraints to a proposed trade.

    Attributes:
        allowed: Whether the trade is permitted (possibly at reduced size).
        adjusted_fraction: The (possibly reduced) position fraction after
            constraint adjustments. Will be 0.0 if the trade is blocked.
        violations: List of human-readable strings describing each
            constraint that was violated or triggered.
        kill_switch_active: True if a kill switch (daily loss or agent
            health) has been triggered, meaning all trading should cease.
    """

    allowed: bool
    adjusted_fraction: float
    violations: List[str] = field(default_factory=list)
    kill_switch_active: bool = False

    def __repr__(self) -> str:
        status = "ALLOWED" if self.allowed else "BLOCKED"
        kill = " [KILL SWITCH]" if self.kill_switch_active else ""
        n_violations = len(self.violations)
        return (
            f"ConstraintResult({status}{kill}, "
            f"fraction={self.adjusted_fraction:.4f}, "
            f"violations={n_violations})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Constraints engine
# ═══════════════════════════════════════════════════════════════════════════

class Constraints:
    """Position limits, exposure caps, and kill switches.

    Evaluates a proposed trade against the full set of risk constraints
    defined in ``RiskParams``. Each check returns a (passed, reason) tuple,
    and ``apply_all`` aggregates them into a single ``ConstraintResult``.

    Parameters:
        params: A ``RiskParams`` instance containing all threshold values.

    Example::

        params = RiskParams(max_daily_loss_pct=0.02, min_agent_sharpe=-1.0)
        constraints = Constraints(params)
        result = constraints.apply_all(
            proposed_fraction=0.15,
            gross_exposure=1.2,
            net_exposure=0.8,
            daily_loss=0.01,
            drawdown=0.05,
            agent_sharpe=0.3,
        )
        if result.allowed:
            execute_trade(result.adjusted_fraction)
    """

    def __init__(self, params: RiskParams) -> None:
        self.params = params

    # ───────────────────────────────────────────────────────────────────
    # Individual checks
    # ───────────────────────────────────────────────────────────────────

    def check_position_limit(self, proposed_pct: float) -> Tuple[bool, str]:
        """Check whether a single position size exceeds the concentration limit.

        Parameters:
            proposed_pct: Proposed position size as a fraction of portfolio
                (e.g. 0.15 = 15% of NAV).

        Returns:
            (passed, reason): passed is True if within limit; reason is
            an empty string on pass, or a descriptive violation message.
        """
        limit = self.params.max_single_position_pct
        if proposed_pct > limit:
            reason = (
                f"Position limit violated: proposed {proposed_pct:.2%} "
                f"exceeds max {limit:.2%} per instrument."
            )
            logger.warning(reason)
            return False, reason
        return True, ""

    def check_exposure(
        self, gross_pct: float, net_pct: float
    ) -> Tuple[bool, str]:
        """Check gross and net portfolio exposure against limits.

        Parameters:
            gross_pct: Total absolute exposure as fraction of NAV
                (sum of |long| + |short| positions).
            net_pct: Net directional exposure as fraction of NAV
                (long - short).

        Returns:
            (passed, reason): passed is True if both gross and net are
            within limits.
        """
        violations: List[str] = []

        if gross_pct > self.params.max_gross_exposure:
            violations.append(
                f"Gross exposure {gross_pct:.2%} exceeds "
                f"max {self.params.max_gross_exposure:.2%}."
            )

        if abs(net_pct) > self.params.max_net_exposure:
            violations.append(
                f"Net exposure {net_pct:+.2%} exceeds "
                f"max +/-{self.params.max_net_exposure:.2%}."
            )

        if violations:
            reason = " | ".join(violations)
            logger.warning("Exposure check failed: %s", reason)
            return False, reason

        return True, ""

    def check_daily_loss(self, current_loss_pct: float) -> Tuple[bool, str]:
        """Kill switch: halt all trading if daily loss exceeds threshold.

        This is a hard stop. Once triggered, no new positions should be
        opened for the remainder of the trading session. Existing
        positions may be closed at the risk manager's discretion.

        Parameters:
            current_loss_pct: Today's unrealized + realized loss as a
                positive fraction of starting NAV (e.g. 0.02 = 2% loss).

        Returns:
            (passed, reason): passed is False if the kill switch triggers.
        """
        limit = self.params.max_daily_loss_pct
        if current_loss_pct >= limit:
            reason = (
                f"KILL SWITCH: Daily loss {current_loss_pct:.2%} "
                f"has reached limit {limit:.2%}. "
                f"All new trading halted for this session."
            )
            logger.critical(reason)
            return False, reason
        return True, ""

    def check_drawdown(self, current_dd_pct: float) -> Tuple[bool, str]:
        """Drawdown gate: reduce position sizing when drawdown is elevated.

        Unlike the daily loss kill switch, this does not fully halt trading.
        Instead, it signals that position sizes should be reduced (typically
        by 50%) to limit further losses during a drawdown.

        Parameters:
            current_dd_pct: Current drawdown from peak NAV as a positive
                fraction (e.g. 0.08 = 8% drawdown).

        Returns:
            (passed, reason): passed is False if drawdown exceeds
            the threshold, indicating that sizing should be reduced.
        """
        limit = self.params.max_drawdown_pct
        if current_dd_pct >= limit:
            reason = (
                f"Drawdown gate triggered: current drawdown {current_dd_pct:.2%} "
                f"exceeds max {limit:.2%}. "
                f"Position sizing should be reduced by 50%."
            )
            logger.warning(reason)
            return False, reason
        return True, ""

    def check_agent_health(self, rolling_sharpe: float) -> Tuple[bool, str]:
        """Kill switch: disable a trading agent with persistently poor performance.

        If a player/agent's rolling Sharpe ratio falls below the minimum
        threshold, it is considered broken or miscalibrated and should
        be removed from the active ensemble until reviewed.

        Parameters:
            rolling_sharpe: The agent's rolling Sharpe ratio over the
                configured lookback window.

        Returns:
            (passed, reason): passed is False if Sharpe is below the
            minimum, indicating the agent should be disabled.
        """
        limit = self.params.min_agent_sharpe
        if rolling_sharpe < limit:
            reason = (
                f"KILL SWITCH: Agent rolling Sharpe {rolling_sharpe:.2f} "
                f"is below minimum {limit:.2f}. "
                f"Agent should be disabled until reviewed."
            )
            logger.critical(reason)
            return False, reason
        return True, ""

    # ───────────────────────────────────────────────────────────────────
    # Aggregate check
    # ───────────────────────────────────────────────────────────────────

    def apply_all(
        self,
        proposed_fraction: float,
        gross_exposure: float,
        net_exposure: float,
        daily_loss: float,
        drawdown: float,
        agent_sharpe: float,
    ) -> ConstraintResult:
        """Apply all constraints and return an aggregated result.

        Checks are evaluated in order of severity. Kill switches are
        checked first; if any triggers, the trade is fully blocked
        with ``kill_switch_active=True``. Softer constraints (drawdown,
        exposure, position limit) may reduce the position size rather
        than blocking entirely.

        Parameters:
            proposed_fraction: Desired position size as fraction of NAV.
            gross_exposure: Current gross exposure (fraction of NAV).
            net_exposure: Current net exposure (fraction of NAV).
            daily_loss: Today's loss so far (positive fraction of NAV).
            drawdown: Current peak-to-trough drawdown (positive fraction).
            agent_sharpe: Rolling Sharpe ratio of the requesting agent.

        Returns:
            ConstraintResult with the final allowed/blocked decision,
            adjusted position fraction, list of violations, and kill
            switch status.
        """
        violations: List[str] = []
        kill_switch = False
        adjusted = proposed_fraction

        # --- Kill switches (hard stops) ---

        passed, reason = self.check_daily_loss(daily_loss)
        if not passed:
            violations.append(reason)
            kill_switch = True

        passed, reason = self.check_agent_health(agent_sharpe)
        if not passed:
            violations.append(reason)
            kill_switch = True

        if kill_switch:
            logger.critical(
                "Kill switch active: blocking trade. Violations: %d",
                len(violations),
            )
            return ConstraintResult(
                allowed=False,
                adjusted_fraction=0.0,
                violations=violations,
                kill_switch_active=True,
            )

        # --- Drawdown gate (soft stop — reduce sizing by 50%) ---

        passed, reason = self.check_drawdown(drawdown)
        if not passed:
            violations.append(reason)
            adjusted *= 0.5
            logger.info(
                "Drawdown gate: reduced fraction from %.4f to %.4f",
                proposed_fraction,
                adjusted,
            )

        # --- Exposure limits ---

        passed, reason = self.check_exposure(gross_exposure, net_exposure)
        if not passed:
            violations.append(reason)
            # Scale down the proposed fraction to bring exposure within limits
            if gross_exposure > 0:
                gross_headroom = max(
                    0.0,
                    self.params.max_gross_exposure - gross_exposure,
                )
                # Only allow a fraction proportional to remaining headroom
                if gross_headroom < adjusted:
                    adjusted = gross_headroom
                    logger.info(
                        "Exposure cap: reduced fraction to %.4f "
                        "(gross headroom: %.4f)",
                        adjusted,
                        gross_headroom,
                    )

        # --- Position concentration limit ---

        passed, reason = self.check_position_limit(adjusted)
        if not passed:
            violations.append(reason)
            # Clamp to the maximum allowed single position
            adjusted = min(adjusted, self.params.max_single_position_pct)
            logger.info(
                "Position limit: clamped fraction to %.4f",
                adjusted,
            )

        # --- Final determination ---

        allowed = adjusted > 0.0

        if violations:
            logger.info(
                "Constraint check: %d violations, adjusted_fraction=%.4f, "
                "allowed=%s",
                len(violations),
                adjusted,
                allowed,
            )

        return ConstraintResult(
            allowed=allowed,
            adjusted_fraction=adjusted,
            violations=violations,
            kill_switch_active=False,
        )
