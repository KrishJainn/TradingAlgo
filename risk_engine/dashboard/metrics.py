"""Real-time risk metrics computation for the Streamlit dashboard.

Provides portfolio-level, agent-level, risk-utilisation, and cost-attribution
metrics.  All computations are vectorised with numpy and use scipy.stats for
higher moments (skewness, kurtosis).

Mathematical foundations
------------------------

**Sharpe Ratio** (Sharpe 1966):
    SR = (mean(r) - r_f) / std(r) * sqrt(T)

    where r_f is the daily risk-free rate and T is the annualisation factor
    (sqrt(252) for daily returns).

**Sortino Ratio** (Sortino & van der Meer 1991):
    Sortino = (mean(r) - r_f) / downside_deviation * sqrt(T)

    Downside deviation uses only returns below the target (MAR = 0):
        DD = sqrt(mean(min(r - MAR, 0)^2))

**Calmar Ratio** (Young 1991):
    Calmar = annualised_return / max_drawdown

**Value at Risk (VaR)** -- historical percentile method:
    VaR_{alpha} = -percentile(r, 100 * (1 - alpha))

**Conditional VaR (CVaR / Expected Shortfall)**:
    CVaR_{alpha} = -mean(r | r <= -VaR_{alpha})

    CVaR is a coherent risk measure (Artzner et al. 1999) and is preferred
    over VaR for portfolio risk management.

**Herfindahl-Hirschman Index (HHI)**:
    HHI = sum(w_i^2)

    where w_i is the weight of position i.  1/HHI gives the effective
    number of independent positions.

References
----------
Sharpe, W.F. (1966). Mutual Fund Performance. *J. Business*, 39(1), 119-138.
Sortino, F.A. & van der Meer, R. (1991). Downside Risk.
    *J. Portfolio Management*, 17(4), 27-31.
Young, T.W. (1991). Calmar Ratio: A smoother tool.
    *Futures*, 20(1), 40.
Artzner, P. et al. (1999). Coherent Measures of Risk.
    *Mathematical Finance*, 9(3), 203-228.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from risk_engine.config.risk_params import RiskParams
from risk_engine.config.constants import (
    RISK_DEFAULTS,
    ANNUALIZATION_FACTOR,
    TRADING_DAYS_PER_YEAR,
    NSE_COSTS,
)

logger = logging.getLogger(__name__)


class RiskDashboardMetrics:
    """Compute real-time risk and performance metrics for dashboard display.

    This class is designed to be instantiated once and called repeatedly
    with updated data on each bar or at the end of each day.  All methods
    handle edge cases (empty arrays, single elements, all-zero returns)
    gracefully, returning sensible defaults rather than raising exceptions.

    Parameters
    ----------
    params : RiskParams | None
        Risk parameters (used for drawdown thresholds and similar
        configuration).  Defaults to ``RiskParams()`` if not provided.

    Example
    -------
    >>> metrics = RiskDashboardMetrics()
    >>> result = metrics.compute_portfolio_metrics(
    ...     returns=pd.Series(np.random.randn(100) * 0.01),
    ...     positions={"PLAYER_1": 0.15, "PLAYER_2": -0.10},
    ...     portfolio_value=1_000_000,
    ... )
    >>> result["sharpe"]
    0.42
    """

    def __init__(self, params: RiskParams | None = None) -> None:
        self.params: RiskParams = params or RiskParams()

    # ------------------------------------------------------------------
    # Portfolio-level metrics
    # ------------------------------------------------------------------

    def compute_portfolio_metrics(
        self,
        returns: pd.Series,
        positions: dict[str, float],
        portfolio_value: float,
    ) -> dict[str, float]:
        """Compute a comprehensive suite of portfolio performance metrics.

        Parameters
        ----------
        returns : pd.Series
            Time series of portfolio returns (simple returns, not log).
            Index should be a DatetimeIndex or integer index.
        positions : dict[str, float]
            Current positions as agent_id -> fraction of NAV.
        portfolio_value : float
            Current portfolio net asset value in INR.

        Returns
        -------
        dict[str, float]
            Dictionary with keys:
            * ``sharpe`` -- annualised Sharpe ratio.
            * ``sortino`` -- annualised Sortino ratio.
            * ``calmar`` -- Calmar ratio (annualised return / max drawdown).
            * ``max_drawdown`` -- maximum peak-to-trough drawdown.
            * ``current_drawdown`` -- current drawdown from peak.
            * ``var_95`` -- 95% Value at Risk (1-day).
            * ``cvar_95`` -- 95% Conditional VaR (Expected Shortfall).
            * ``vol_annualized`` -- annualised volatility.
            * ``skewness`` -- skewness of the return distribution.
            * ``kurtosis`` -- excess kurtosis of the return distribution.
            * ``hit_rate`` -- fraction of positive-return periods.
            * ``profit_factor`` -- gross profits / gross losses.
        """
        r: np.ndarray = np.asarray(returns.dropna().values, dtype=np.float64)

        if len(r) == 0:
            return self._empty_portfolio_metrics()

        # -- Basic statistics -----------------------------------------------
        mean_r: float = float(np.mean(r))
        std_r: float = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0

        # -- Sharpe Ratio ---------------------------------------------------
        # SR = (mean(r) - r_f) / std(r) * sqrt(T)
        from risk_engine.config.constants import RISK_FREE_RATE_DAILY
        excess_mean: float = mean_r - RISK_FREE_RATE_DAILY
        sharpe: float = (
            (excess_mean / std_r * ANNUALIZATION_FACTOR)
            if std_r > 1e-12
            else 0.0
        )

        # -- Sortino Ratio --------------------------------------------------
        # Downside deviation = sqrt(mean(min(r, 0)^2))
        downside: np.ndarray = np.minimum(r, 0.0)
        downside_dev: float = float(np.sqrt(np.mean(downside ** 2)))
        sortino: float = (
            (excess_mean / downside_dev * ANNUALIZATION_FACTOR)
            if downside_dev > 1e-12
            else 0.0
        )

        # -- Drawdown series ------------------------------------------------
        cumulative: np.ndarray = np.cumprod(1.0 + r)
        running_max: np.ndarray = np.maximum.accumulate(cumulative)
        drawdowns: np.ndarray = (running_max - cumulative) / np.where(
            running_max > 0, running_max, 1.0
        )
        max_drawdown: float = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0
        current_drawdown: float = float(drawdowns[-1]) if len(drawdowns) > 0 else 0.0

        # -- Calmar Ratio ---------------------------------------------------
        # Calmar = annualised_return / max_drawdown
        annualised_return: float = float(
            (cumulative[-1]) ** (TRADING_DAYS_PER_YEAR / max(len(r), 1)) - 1.0
        ) if len(r) > 0 and cumulative[-1] > 0 else 0.0
        calmar: float = (
            annualised_return / max_drawdown
            if max_drawdown > 1e-8
            else 0.0
        )

        # -- VaR and CVaR (Historical) -------------------------------------
        # VaR_95 = -percentile(r, 5)
        var_95: float = float(-np.percentile(r, 5)) if len(r) >= 2 else 0.0

        # CVaR_95 = -mean(r | r <= -VaR_95)
        tail_returns: np.ndarray = r[r <= -var_95]
        cvar_95: float = (
            float(-np.mean(tail_returns))
            if len(tail_returns) > 0
            else var_95
        )

        # -- Annualised Volatility ------------------------------------------
        vol_annualized: float = std_r * ANNUALIZATION_FACTOR

        # -- Higher moments (scipy.stats) -----------------------------------
        skewness: float = float(scipy_stats.skew(r, bias=False)) if len(r) > 2 else 0.0
        kurtosis: float = float(scipy_stats.kurtosis(r, bias=False)) if len(r) > 3 else 0.0

        # -- Hit rate -------------------------------------------------------
        hit_rate: float = float(np.mean(r > 0)) if len(r) > 0 else 0.0

        # -- Profit factor --------------------------------------------------
        # profit_factor = sum(positive returns) / |sum(negative returns)|
        gains: float = float(np.sum(r[r > 0]))
        losses: float = float(np.abs(np.sum(r[r < 0])))
        profit_factor: float = (gains / losses) if losses > 1e-12 else (
            float("inf") if gains > 0 else 0.0
        )

        return {
            "sharpe": round(sharpe, 4),
            "sortino": round(sortino, 4),
            "calmar": round(calmar, 4),
            "max_drawdown": round(max_drawdown, 6),
            "current_drawdown": round(current_drawdown, 6),
            "var_95": round(var_95, 6),
            "cvar_95": round(cvar_95, 6),
            "vol_annualized": round(vol_annualized, 6),
            "skewness": round(skewness, 4),
            "kurtosis": round(kurtosis, 4),
            "hit_rate": round(hit_rate, 4),
            "profit_factor": round(profit_factor, 4),
        }

    # ------------------------------------------------------------------
    # Agent-level metrics
    # ------------------------------------------------------------------

    def compute_agent_metrics(
        self,
        agent_returns: dict[str, pd.Series],
    ) -> dict[str, dict[str, float]]:
        """Compute per-agent performance and contribution metrics.

        Parameters
        ----------
        agent_returns : dict[str, pd.Series]
            Mapping of agent_id -> return series.  Each series should
            cover the same time range for meaningful comparison.

        Returns
        -------
        dict[str, dict[str, float]]
            Mapping of agent_id -> metrics dict with keys:
            * ``sharpe`` -- annualised Sharpe ratio.
            * ``ic`` -- information coefficient (correlation with next-bar
              return; estimated as autocorrelation(1) of the signal returns).
            * ``rolling_pnl`` -- cumulative PnL as a fraction (final
              cumulative return - 1).
            * ``contribution_to_portfolio`` -- proportional contribution
              to the total portfolio return.

        Reference
        ---------
        Grinold, R.C. & Kahn, R.N. (1999). *Active Portfolio Management*,
        Chapter 6: Information Coefficient and the Fundamental Law.
        """
        if not agent_returns:
            return {}

        # Compute total portfolio return for contribution calculation
        all_series: list[pd.Series] = list(agent_returns.values())
        portfolio_return_total: float = 0.0
        for s in all_series:
            clean: np.ndarray = np.asarray(s.dropna().values, dtype=np.float64)
            if len(clean) > 0:
                portfolio_return_total += float(np.sum(clean))

        result: dict[str, dict[str, float]] = {}

        for agent_id, ret_series in agent_returns.items():
            r: np.ndarray = np.asarray(
                ret_series.dropna().values, dtype=np.float64
            )

            if len(r) == 0:
                result[agent_id] = {
                    "sharpe": 0.0,
                    "ic": 0.0,
                    "rolling_pnl": 0.0,
                    "contribution_to_portfolio": 0.0,
                }
                continue

            # -- Sharpe -----------------------------------------------------
            from risk_engine.config.constants import RISK_FREE_RATE_DAILY
            mean_r: float = float(np.mean(r))
            std_r: float = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
            excess: float = mean_r - RISK_FREE_RATE_DAILY
            sharpe: float = (
                (excess / std_r * ANNUALIZATION_FACTOR)
                if std_r > 1e-12
                else 0.0
            )

            # -- Information Coefficient (IC) --------------------------------
            # Estimated as lag-1 autocorrelation of the return series.
            # IC = corr(signal_t, return_{t+1}). When we only have signal
            # returns (not raw signals), autocorrelation is a proxy.
            #
            # Reference: Grinold & Kahn (1999), Chapter 6.
            ic: float = 0.0
            if len(r) > 2:
                ic = float(np.corrcoef(r[:-1], r[1:])[0, 1])
                if np.isnan(ic):
                    ic = 0.0

            # -- Rolling PnL ------------------------------------------------
            cumulative_return: float = float(np.prod(1.0 + r) - 1.0)

            # -- Contribution to portfolio ----------------------------------
            agent_total: float = float(np.sum(r))
            contribution: float = (
                agent_total / portfolio_return_total
                if abs(portfolio_return_total) > 1e-12
                else 0.0
            )

            result[agent_id] = {
                "sharpe": round(sharpe, 4),
                "ic": round(ic, 4),
                "rolling_pnl": round(cumulative_return, 6),
                "contribution_to_portfolio": round(contribution, 4),
            }

        return result

    # ------------------------------------------------------------------
    # Risk utilisation
    # ------------------------------------------------------------------

    def compute_risk_utilization(
        self,
        positions: dict[str, float],
        portfolio_value: float,
        params: RiskParams | None = None,
    ) -> dict[str, float]:
        """Compute risk utilisation metrics relative to configured limits.

        Parameters
        ----------
        positions : dict[str, float]
            Current positions as agent_id -> fraction of NAV.
            Positive = long, negative = short.
        portfolio_value : float
            Current portfolio NAV in INR.
        params : RiskParams | None
            Override risk parameters.  Uses ``self.params`` if None.

        Returns
        -------
        dict[str, float]
            Dictionary with keys:
            * ``net_exposure_pct`` -- net directional exposure as % of NAV.
            * ``gross_exposure_pct`` -- total absolute exposure as % of NAV.
            * ``max_position_pct`` -- largest single position as % of NAV.
            * ``exposure_headroom`` -- remaining gross exposure capacity
              before hitting ``max_gross_exposure``.
            * ``concentration_hhi`` -- Herfindahl-Hirschman Index of
              position weights.  Range [1/N, 1].  Lower = more diversified.

        Reference
        ---------
        Herfindahl, O.C. (1950). *Concentration in the Steel Industry*.
        """
        p: RiskParams = params or self.params

        if not positions:
            return {
                "net_exposure_pct": 0.0,
                "gross_exposure_pct": 0.0,
                "max_position_pct": 0.0,
                "exposure_headroom": p.max_gross_exposure,
                "concentration_hhi": 0.0,
            }

        values: np.ndarray = np.array(
            list(positions.values()), dtype=np.float64
        )

        net_exposure: float = float(np.sum(values))
        gross_exposure: float = float(np.sum(np.abs(values)))
        max_position: float = float(np.max(np.abs(values)))

        # Exposure headroom: how much more gross exposure is available
        headroom: float = max(0.0, p.max_gross_exposure - gross_exposure)

        # HHI: sum of squared weights (using absolute positions normalised
        # by gross exposure)
        hhi: float = 0.0
        if gross_exposure > 1e-12:
            normalised: np.ndarray = np.abs(values) / gross_exposure
            hhi = float(np.sum(normalised ** 2))

        return {
            "net_exposure_pct": round(net_exposure, 6),
            "gross_exposure_pct": round(gross_exposure, 6),
            "max_position_pct": round(max_position, 6),
            "exposure_headroom": round(headroom, 6),
            "concentration_hhi": round(hhi, 6),
        }

    # ------------------------------------------------------------------
    # Cost attribution
    # ------------------------------------------------------------------

    def compute_cost_attribution(
        self,
        trades: list[dict[str, Any]],
        portfolio_value: float,
    ) -> dict[str, Any]:
        """Attribute transaction costs by component.

        Aggregates trade-level cost breakdowns into portfolio-level
        cost attribution.  Each trade dict should contain at minimum
        ``"notional"`` (in INR) and cost component fields.

        Parameters
        ----------
        trades : list[dict[str, Any]]
            List of trade dictionaries, each with keys:
            * ``"notional"`` -- trade notional in INR.
            * ``"brokerage"`` -- brokerage cost in INR.
            * ``"stt"`` -- Securities Transaction Tax in INR.
            * ``"slippage"`` -- estimated slippage in INR.
            * ``"exchange"`` -- exchange transaction charge in INR.
            * ``"gst"`` -- GST in INR.
            * ``"sebi"`` -- SEBI fee in INR.
            * ``"stamp_duty"`` -- stamp duty in INR.
            Missing keys default to 0.0.
        portfolio_value : float
            Current portfolio NAV in INR.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            * ``total_cost_bps`` -- total cost as basis points of NAV.
            * ``cost_by_type`` -- dict mapping cost component name to
              total INR value.
            * ``cost_drag_annualized`` -- estimated annual cost drag
              as a fraction of NAV, extrapolated from the current
              trading frequency.

        Reference
        ---------
        NSE circular on transaction charges; SEBI fee schedule.
        Almgren, R. & Chriss, N. (2001). *J. Risk*, 3(2), 5-39.
        """
        if not trades or portfolio_value <= 0:
            return {
                "total_cost_bps": 0.0,
                "cost_by_type": {
                    "brokerage": 0.0,
                    "stt": 0.0,
                    "slippage": 0.0,
                    "exchange": 0.0,
                    "gst": 0.0,
                    "sebi": 0.0,
                    "stamp_duty": 0.0,
                },
                "cost_drag_annualized": 0.0,
            }

        # Cost component names to aggregate
        cost_components: list[str] = [
            "brokerage", "stt", "slippage", "exchange",
            "gst", "sebi", "stamp_duty",
        ]

        cost_by_type: dict[str, float] = {c: 0.0 for c in cost_components}
        total_cost_inr: float = 0.0

        for trade in trades:
            for component in cost_components:
                val: float = float(trade.get(component, 0.0))
                cost_by_type[component] += val
                total_cost_inr += val

        # Total cost in basis points of portfolio NAV
        total_cost_bps: float = (
            (total_cost_inr / portfolio_value) * 10_000.0
            if portfolio_value > 0
            else 0.0
        )

        # Annualised cost drag:
        # If these trades represent a single day's activity, the annual
        # drag is total_cost_fraction * TRADING_DAYS_PER_YEAR.
        daily_cost_fraction: float = total_cost_inr / portfolio_value if portfolio_value > 0 else 0.0
        cost_drag_annualized: float = daily_cost_fraction * TRADING_DAYS_PER_YEAR

        return {
            "total_cost_bps": round(total_cost_bps, 4),
            "cost_by_type": {k: round(v, 2) for k, v in cost_by_type.items()},
            "cost_drag_annualized": round(cost_drag_annualized, 6),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_portfolio_metrics() -> dict[str, float]:
        """Return a zeroed-out metrics dict when no return data is available.

        Returns
        -------
        dict[str, float]
            All metric values set to 0.0.
        """
        return {
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "var_95": 0.0,
            "cvar_95": 0.0,
            "vol_annualized": 0.0,
            "skewness": 0.0,
            "kurtosis": 0.0,
            "hit_rate": 0.0,
            "profit_factor": 0.0,
        }
