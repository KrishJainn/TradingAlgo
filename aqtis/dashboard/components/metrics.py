"""Calculate all backtest performance metrics."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Container for all backtest metrics."""

    # Returns
    total_return: float = 0.0
    cagr: float = 0.0

    # Risk-adjusted
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Drawdown
    max_drawdown: float = 0.0
    avg_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0

    # Win / Loss
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0

    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Trade stats
    total_trades: int = 0
    trades_per_day: float = 0.0
    avg_hold_duration_hours: float = 0.0
    expectancy: float = 0.0

    # Benchmark
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0


def calculate_metrics(
    trades: pd.DataFrame,
    equity_curve: pd.Series,
    benchmark: pd.Series = None,
    risk_free_rate: float = 0.05,
) -> BacktestMetrics:
    """
    Calculate comprehensive backtest metrics.

    Args:
        trades: DataFrame with entry_time, exit_time, pnl, pnl_percent columns.
        equity_curve: Series of portfolio value over time.
        benchmark: Optional benchmark series for comparison.
        risk_free_rate: Annual risk-free rate.

    Returns:
        BacktestMetrics dataclass.
    """
    m = BacktestMetrics()

    if trades is None or trades.empty:
        return m

    pnls = trades["pnl"].dropna()
    pnl_pct = trades.get("pnl_percent", pnls)

    m.total_trades = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    # Win / loss
    m.win_rate = len(wins) / m.total_trades if m.total_trades else 0
    m.avg_win = wins.mean() if len(wins) else 0
    m.avg_loss = losses.mean() if len(losses) else 0
    m.largest_win = wins.max() if len(wins) else 0
    m.largest_loss = losses.min() if len(losses) else 0
    m.profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
    m.expectancy = pnls.mean() if len(pnls) else 0

    # Streaks
    m.max_consecutive_wins = _max_streak(pnls > 0)
    m.max_consecutive_losses = _max_streak(pnls <= 0)

    # Hold duration
    if "entry_time" in trades.columns and "exit_time" in trades.columns:
        durations = (
            pd.to_datetime(trades["exit_time"]) - pd.to_datetime(trades["entry_time"])
        )
        m.avg_hold_duration_hours = durations.dt.total_seconds().mean() / 3600

    # Equity curve metrics
    if equity_curve is not None and len(equity_curve) > 1:
        returns = equity_curve.pct_change().dropna()
        daily_rf = risk_free_rate / 252

        m.total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # CAGR — handle both datetime and integer indices
        try:
            idx_start = equity_curve.index[0]
            idx_end = equity_curve.index[-1]
            if hasattr(idx_start, "days"):
                # Timedelta-like
                n_days = max((idx_end - idx_start).days, 1)
            elif isinstance(idx_start, (pd.Timestamp, np.datetime64)):
                n_days = max((pd.Timestamp(idx_end) - pd.Timestamp(idx_start)).days, 1)
            else:
                # Integer or other index — estimate from length
                n_days = max(len(equity_curve), 1)
        except Exception:
            n_days = max(len(equity_curve), 1)
        m.cagr = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (365 / n_days) - 1

        # Sharpe
        excess = returns - daily_rf
        m.sharpe_ratio = (
            excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0
        )

        # Sortino
        downside = returns[returns < daily_rf] - daily_rf
        downside_std = downside.std() if len(downside) > 0 else 0
        m.sortino_ratio = (
            (returns.mean() - daily_rf) / downside_std * np.sqrt(252)
            if downside_std > 0
            else 0
        )

        # Drawdown
        dd_series = calculate_drawdown_series(equity_curve)
        m.max_drawdown = dd_series["drawdown_pct"].min()
        m.avg_drawdown = dd_series["drawdown_pct"][dd_series["drawdown_pct"] < 0].mean()
        if np.isnan(m.avg_drawdown):
            m.avg_drawdown = 0

        # Max drawdown duration
        m.max_drawdown_duration_days = _max_dd_duration(dd_series["drawdown_pct"])

        # Calmar
        m.calmar_ratio = m.cagr / abs(m.max_drawdown) if m.max_drawdown != 0 else 0

        # Trades per day
        m.trades_per_day = m.total_trades / max(n_days, 1)

        # Benchmark comparison
        if benchmark is not None and len(benchmark) > 1:
            bench_ret = benchmark.pct_change().dropna()
            # Align
            aligned = pd.DataFrame({"port": returns, "bench": bench_ret}).dropna()
            if len(aligned) > 1:
                cov = np.cov(aligned["port"], aligned["bench"])
                m.beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else 0
                m.alpha = aligned["port"].mean() - m.beta * aligned["bench"].mean()
                tracking_err = (aligned["port"] - aligned["bench"]).std()
                m.information_ratio = (
                    (aligned["port"].mean() - aligned["bench"].mean()) / tracking_err * np.sqrt(252)
                    if tracking_err > 0
                    else 0
                )

    return m


def calculate_drawdown_series(equity: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown series.

    Returns DataFrame with: peak, drawdown, drawdown_pct.
    """
    peak = equity.expanding().max()
    drawdown = equity - peak
    drawdown_pct = drawdown / peak
    return pd.DataFrame(
        {"peak": peak, "drawdown": drawdown, "drawdown_pct": drawdown_pct},
        index=equity.index,
    )


def calculate_rolling_metrics(
    trades: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    """Calculate rolling Sharpe, win rate, P&L over a trade window."""
    if trades is None or len(trades) < window:
        return pd.DataFrame()

    pnls = trades["pnl"].values
    results = []
    for i in range(window, len(pnls) + 1):
        chunk = pnls[i - window : i]
        wr = np.sum(chunk > 0) / len(chunk)
        sharpe = np.mean(chunk) / (np.std(chunk) + 1e-10) * np.sqrt(252)
        results.append({"trade_num": i, "win_rate": wr, "sharpe": sharpe, "avg_pnl": np.mean(chunk)})

    return pd.DataFrame(results)


# -----------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------


def _max_streak(mask: pd.Series) -> int:
    """Maximum consecutive True values in a boolean series."""
    if mask.empty:
        return 0
    groups = mask.ne(mask.shift()).cumsum()
    streaks = mask.groupby(groups).sum()
    return int(streaks.max()) if len(streaks) else 0


def _max_dd_duration(dd_pct: pd.Series) -> int:
    """Max duration (in index positions) spent in drawdown."""
    in_dd = dd_pct < 0
    if not in_dd.any():
        return 0
    groups = in_dd.ne(in_dd.shift()).cumsum()
    durations = in_dd.groupby(groups).sum()
    return int(durations.max())
