"""
Rolling Information Coefficient (IC) Tracker.

Tracks the predictive power of alpha forecasts over time using the
rank-based Information Coefficient (Spearman's rho) and derived metrics.

Core Concepts
-------------

1. **Information Coefficient (IC)**:

       IC_t = rho_s(forecast_t, realized_t)

   where rho_s is Spearman's rank correlation.  IC measures the monotonic
   association between cross-sectional forecasts and subsequent realised
   returns.  An IC of +1 means perfect ranking skill; 0 means no skill.

2. **Information Coefficient Information Ratio (ICIR)**:

       ICIR = mean(IC) / std(IC)

   Analogous to the Sharpe Ratio of the IC series.  An ICIR above 0.5 is
   generally considered strong; above 1.0 is exceptional.

3. **Fundamental Law of Active Management** (Grinold, 1989):

       IR ~ IC * sqrt(BR)

   where BR (breadth) = number of independent bets per period.  This
   decomposition shows that even modest IC can deliver high IR when
   applied across many assets or high-frequency rebalancing.

4. **Decay-weighted IC**:

   Exponentially weighted mean of the IC series with a configurable
   half-life, giving more weight to recent forecasting accuracy:

       IC_ew = sum(w_i * IC_i) / sum(w_i)

   where  w_i = 2^{-(T - i) / half_life}.

References
----------
Grinold, R. C. (1989). "The Fundamental Law of Active Management."
    *Journal of Portfolio Management*, 15(3), 30--37.

Grinold, R. C. & Kahn, R. N. (1999). *Active Portfolio Management:
    A Quantitative Approach for Producing Superior Returns and Controlling
    Risk*, 2nd ed. McGraw-Hill, Chapter 14.

Qian, E. & Hua, R. (2004). "Active Risk and Information Ratio."
    *Journal of Investment Management*, 2(3), 1--15.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

from risk_engine.config.constants import RISK_DEFAULTS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ICResult:
    """Snapshot of Information Coefficient statistics.

    Attributes
    ----------
    ic : float
        Most recent single-period Information Coefficient
        (Spearman rank correlation between forecasts and realised returns).
    ic_mean : float
        Rolling mean of IC over the lookback window.
    ic_std : float
        Rolling standard deviation of IC over the lookback window.
    icir : float
        Information Coefficient Information Ratio = ic_mean / ic_std.
        Set to 0.0 when ic_std is near zero.
    hit_rate : float
        Fraction of IC observations in the buffer that are positive,
        i.e. ``P(IC > 0)``.  Represents the *consistency* of forecasting
        skill.
    n_obs : int
        Number of IC observations currently in the rolling buffer.
    """

    ic: float
    ic_mean: float
    ic_std: float
    icir: float
    hit_rate: float
    n_obs: int


# ═══════════════════════════════════════════════════════════════════════════
# Main class
# ═══════════════════════════════════════════════════════════════════════════


class ICTracker:
    """Rolling Information Coefficient tracker.

    Maintains a FIFO buffer of per-period IC values and exposes summary
    statistics (mean IC, ICIR, hit rate, decay-weighted IC, breadth-adjusted
    IR).

    Parameters
    ----------
    lookback : int
        Maximum number of IC observations retained in the rolling buffer.
        Defaults to ``RISK_DEFAULTS["ic_lookback"]`` (60).

    Examples
    --------
    >>> import numpy as np
    >>> tracker = ICTracker(lookback=20)
    >>> rng = np.random.default_rng(0)
    >>> for _ in range(25):
    ...     f = rng.standard_normal(30)
    ...     r = 0.05 * f + rng.standard_normal(30) * 0.5
    ...     result = tracker.update(f, r)
    >>> result.n_obs
    20
    >>> result.ic_mean  # doctest: +SKIP
    0.08...
    """

    def __init__(self, lookback: int | None = None) -> None:
        self._lookback: int = lookback if lookback is not None else int(
            RISK_DEFAULTS["ic_lookback"]
        )
        if self._lookback < 1:
            raise ValueError(f"lookback must be >= 1, got {self._lookback}")

        self._buffer: deque[float] = deque(maxlen=self._lookback)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        forecasts: np.ndarray,
        realized: np.ndarray,
    ) -> ICResult:
        """Compute the IC for a single cross-section and update the buffer.

        The Information Coefficient is defined as the Spearman rank
        correlation between the forecast vector and the realised return
        vector:

            IC = rho_s(forecasts, realized)

        where rho_s is computed via ``scipy.stats.spearmanr``.

        Parameters
        ----------
        forecasts : np.ndarray
            1-D array of alpha forecasts for each asset at time *t*.
        realized : np.ndarray
            1-D array of realised returns for each asset over the
            subsequent holding period.

        Returns
        -------
        ICResult
            Updated snapshot of IC statistics.

        Raises
        ------
        ValueError
            If input arrays have different lengths or fewer than 3
            observations (minimum required for meaningful rank correlation).

        Notes
        -----
        Cross-sectional IC requires at least 3 data points for a
        non-degenerate Spearman correlation.  If either array has zero
        variance (all forecasts identical or all returns identical), the
        correlation is undefined and IC is recorded as 0.0.
        """
        forecasts = np.asarray(forecasts, dtype=np.float64).ravel()
        realized = np.asarray(realized, dtype=np.float64).ravel()

        if len(forecasts) != len(realized):
            raise ValueError(
                f"forecasts length ({len(forecasts)}) != "
                f"realized length ({len(realized)})"
            )
        if len(forecasts) < 3:
            raise ValueError(
                f"Need at least 3 cross-sectional observations, "
                f"got {len(forecasts)}"
            )

        # Guard against constant arrays (zero variance -> undefined corr)
        if np.std(forecasts) < 1.0e-15 or np.std(realized) < 1.0e-15:
            ic_value: float = 0.0
        else:
            corr, _pvalue = sp_stats.spearmanr(forecasts, realized)
            ic_value = float(corr) if np.isfinite(corr) else 0.0

        self._buffer.append(ic_value)

        return self._snapshot(ic_value)

    # ------------------------------------------------------------------
    # Rolling IC history
    # ------------------------------------------------------------------

    def rolling_ic(self) -> np.ndarray:
        """Return the full rolling IC history currently in the buffer.

        Returns
        -------
        np.ndarray
            1-D array of IC values, oldest first.  Length is at most
            ``lookback``.
        """
        return np.array(self._buffer, dtype=np.float64)

    # ------------------------------------------------------------------
    # Decay-weighted IC
    # ------------------------------------------------------------------

    def decay_weighted_ic(self, half_life: int = 20) -> float:
        """Exponentially decay-weighted mean of the IC buffer.

        Weights are proportional to:

            w_i = 2^{-(n - 1 - i) / half_life}

        where *i* = 0 is the oldest observation and *n - 1* is the newest.
        This gives the most recent IC observation a weight of 1.0 and
        older observations exponentially less.

        Parameters
        ----------
        half_life : int
            Number of periods for the weight to halve.  Defaults to 20.

        Returns
        -------
        float
            Decay-weighted mean IC.  Returns 0.0 if the buffer is empty.
        """
        if len(self._buffer) == 0:
            return 0.0

        if half_life < 1:
            raise ValueError(f"half_life must be >= 1, got {half_life}")

        ic_arr: np.ndarray = np.array(self._buffer, dtype=np.float64)
        n: int = len(ic_arr)

        # Ages: oldest = n-1, newest = 0  ->  weight = 2^(-age / hl)
        ages: np.ndarray = np.arange(n - 1, -1, -1, dtype=np.float64)
        weights: np.ndarray = np.power(2.0, -ages / half_life)

        weighted_mean: float = float(np.dot(weights, ic_arr) / np.sum(weights))
        return weighted_mean

    # ------------------------------------------------------------------
    # Breadth-adjusted IR
    # ------------------------------------------------------------------

    def breadth_adjusted_ir(self, n_assets: int) -> float:
        """Estimate the strategy Information Ratio via the Fundamental Law.

        Grinold (1989):

            IR = IC * sqrt(BR)

        where **BR** (breadth) is the number of independent bets per
        period.  In the simplest formulation:

            BR = n_assets * rebalance_frequency

        For daily rebalancing (``rebalance_frequency = 252``), and
        ``n_assets = 30``:

            BR = 30 * 252 = 7560
            IR ~ IC * sqrt(7560) ~ IC * 87

        This shows that even a small IC (0.02) can produce a respectable
        IR (~1.7) when breadth is large.

        Parameters
        ----------
        n_assets : int
            Number of assets in the cross-section.

        Returns
        -------
        float
            Breadth-adjusted Information Ratio.  Returns 0.0 if the
            buffer is empty.

        Notes
        -----
        The rebalance frequency is inferred from ``RISK_DEFAULTS``.
        ``"daily"`` maps to 252; ``"weekly"`` to 52; ``"monthly"`` to 12.
        """
        if len(self._buffer) == 0:
            return 0.0

        if n_assets < 1:
            raise ValueError(f"n_assets must be >= 1, got {n_assets}")

        rebal_str: str = str(RISK_DEFAULTS.get("rebalance_frequency", "daily"))
        rebal_map: dict[str, int] = {
            "daily": 252,
            "weekly": 52,
            "monthly": 12,
        }
        rebal_freq: int = rebal_map.get(rebal_str.lower(), 252)

        breadth: float = float(n_assets * rebal_freq)
        ic_mean: float = float(np.mean(self._buffer))

        ir: float = ic_mean * np.sqrt(breadth)
        return float(ir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot(self, latest_ic: float) -> ICResult:
        """Build an ``ICResult`` from the current buffer state.

        Parameters
        ----------
        latest_ic : float
            The IC value that was just appended to the buffer.

        Returns
        -------
        ICResult
        """
        ic_arr: np.ndarray = np.array(self._buffer, dtype=np.float64)
        n: int = len(ic_arr)

        ic_mean: float = float(np.mean(ic_arr))
        ic_std: float = float(np.std(ic_arr, ddof=1)) if n >= 2 else 0.0

        # ICIR = mean / std; protect against near-zero denominator
        icir: float = ic_mean / ic_std if ic_std > 1.0e-10 else 0.0

        hit_rate: float = float(np.mean(ic_arr > 0.0))

        return ICResult(
            ic=round(latest_ic, 6),
            ic_mean=round(ic_mean, 6),
            ic_std=round(ic_std, 6),
            icir=round(icir, 6),
            hit_rate=round(hit_rate, 6),
            n_obs=n,
        )
