"""
Walk-Forward Backtesting Framework for the 5-Player Coach Trading System.

Replaces the fixed 50-day window with a rolling train/test walk-forward
methodology to produce realistic out-of-sample performance estimates.

Architecture:
    WalkForwardEngine  — orchestrates the rolling windows
    BacktestResult     — per-window metrics and trades
    AggregateResults   — combined cross-window statistics
    OverfitDetector    — IS vs OOS comparison, Deflated Sharpe Ratio
    Visualizer         — equity curve, heatmaps, rolling Sharpe, regime overlay

Integration points (all optional — degrades gracefully):
    - TransactionCostModel  from transaction_costs.py
    - PortfolioRiskManager  from portfolio_risk_manager.py
    - RegimeDetector        from regime_detector.py

Usage:
    from walk_forward_backtest import WalkForwardEngine

    engine = WalkForwardEngine(
        training_window=120,
        testing_window=20,
        step_size=20,
    )
    results = engine.run(data, strategy_fn)
    results.generate_report()
    results.save_visualizations("output/")
"""

from __future__ import annotations

import logging
import math
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional imports — degrade gracefully
try:
    from transaction_costs import TransactionCostModel, CostAwareBacktester
    _HAS_COSTS = True
except ImportError:
    _HAS_COSTS = False

try:
    from portfolio_risk_manager import PortfolioRiskManager, RiskLimits
    _HAS_RISK = True
except ImportError:
    _HAS_RISK = False

try:
    from regime_detector import RegimeDetector, compute_features, load_nifty_data
    _HAS_REGIME = True
except ImportError:
    _HAS_REGIME = False

try:
    from signal_combiner import (
        RegimeAwareCombiner, PlayerSignal, CombinedSignal,
        BayesianPlayerWeights, TradeOutcome,
        MIN_CONFIDENCE_THRESHOLD, SIGNAL_FLIP_THRESHOLD,
    )
    _HAS_COMBINER = True
except ImportError:
    _HAS_COMBINER = False

try:
    from regime_aware_player import RegimeAwarePlayer, build_registry as build_player_registry
    _HAS_PLAYERS = True
except ImportError:
    _HAS_PLAYERS = False

try:
    from data_cache.indicator_computer import IndicatorComputer
    _HAS_INDICATORS = True
except ImportError:
    _HAS_INDICATORS = False

try:
    from portfolio_optimizer import (
        PortfolioOptimizer, PlayerReturnTracker, VolatilityTargeter,
    )
    _HAS_OPTIMIZER = True
except ImportError:
    _HAS_OPTIMIZER = False


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.065  # India 10y benchmark ~6.5%
PLAYER_IDS = ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]
PLAYER_LABELS = {
    "PLAYER_1": "TrendFollower",
    "PLAYER_2": "MeanReversion",
    "PLAYER_3": "VolumeFlow",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "MultiMomentum",
}


# ═══════════════════════════════════════════════════════════════════════════
# Trade dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """A single completed trade."""
    symbol: str
    direction: str          # 'LONG' or 'SHORT'
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    quantity: int
    gross_pnl: float
    transaction_cost: float = 0.0
    net_pnl: float = 0.0
    player_id: str = ""
    signal_type: str = "neutral"
    regime: str = "Sideways"
    bars_held: int = 0
    exit_reason: str = ""
    risk_event: str = ""    # non-empty if a risk limit was triggered

    def __post_init__(self):
        if self.net_pnl == 0.0:
            self.net_pnl = self.gross_pnl - self.transaction_cost


# ═══════════════════════════════════════════════════════════════════════════
# Strategy protocol
# ═══════════════════════════════════════════════════════════════════════════

class StrategyFn(Protocol):
    """Callable that a strategy must implement.

    Args:
        train_data:  Dict[symbol, DataFrame] for the training window.
        test_data:   Dict[symbol, DataFrame] for the testing window.
        params:      Optional dict of fitted parameters from training.

    Returns:
        (trades, fitted_params)
        trades: list of Trade objects from the test window
        fitted_params: dict of parameters learned during training (passed
                       to the next window as a warm-start hint)
    """
    def __call__(
        self,
        train_data: Dict[str, pd.DataFrame],
        test_data: Dict[str, pd.DataFrame],
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Trade], Dict[str, Any]]: ...


# ═══════════════════════════════════════════════════════════════════════════
# BacktestResult — per-window metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestResult:
    """Metrics for a single train/test window."""

    window_index: int
    window_start: str
    window_end: str
    train_start: str
    train_end: str

    trades: List[Trade] = field(default_factory=list)

    # P&L
    gross_pnl: float = 0.0
    total_costs: float = 0.0
    net_pnl: float = 0.0

    # Risk-adjusted
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown: float = 0.0
    calmar: float = 0.0

    # Regime
    regime_distribution: Dict[str, float] = field(default_factory=dict)

    # Per-player
    per_player_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Risk events
    risk_events: List[str] = field(default_factory=list)

    # Trade stats
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0

    # In-sample Sharpe (for overfit detection)
    in_sample_sharpe: float = 0.0

    # Daily returns for the test window
    daily_returns: List[float] = field(default_factory=list)

    def compute_metrics(self) -> None:
        """Compute all derived metrics from the trades list."""
        if not self.trades:
            return

        self.gross_pnl = sum(t.gross_pnl for t in self.trades)
        self.total_costs = sum(t.transaction_cost for t in self.trades)
        self.net_pnl = sum(t.net_pnl for t in self.trades)

        winners = [t for t in self.trades if t.net_pnl > 0]
        losers = [t for t in self.trades if t.net_pnl <= 0]
        self.winning_trades = len(winners)
        self.losing_trades = len(losers)
        self.avg_win = np.mean([t.net_pnl for t in winners]) if winners else 0.0
        self.avg_loss = np.mean([t.net_pnl for t in losers]) if losers else 0.0
        total = self.winning_trades + self.losing_trades
        self.win_rate = self.winning_trades / total if total > 0 else 0.0

        gross_wins = sum(t.net_pnl for t in winners) if winners else 0.0
        gross_losses = abs(sum(t.net_pnl for t in losers)) if losers else 0.0
        if gross_losses > 0:
            self.profit_factor = gross_wins / gross_losses
        else:
            self.profit_factor = gross_wins if gross_wins > 0 else 0.0

        # Daily returns → Sharpe, Sortino, Drawdown
        if self.daily_returns:
            rets = np.array(self.daily_returns)
            self.sharpe = _annualized_sharpe(rets)
            self.sortino = _annualized_sortino(rets)
            self.max_drawdown = _max_drawdown_from_returns(rets)
            ann_return = np.mean(rets) * TRADING_DAYS_PER_YEAR
            self.calmar = ann_return / abs(self.max_drawdown) if self.max_drawdown != 0 else 0.0

        # Risk events
        self.risk_events = [t.risk_event for t in self.trades if t.risk_event]

        # Per-player breakdown
        self._compute_per_player()

        # Regime distribution
        self._compute_regime_distribution()

    def _compute_per_player(self) -> None:
        player_trades: Dict[str, List[Trade]] = {}
        for t in self.trades:
            pid = t.player_id or "UNKNOWN"
            player_trades.setdefault(pid, []).append(t)

        for pid, trades in player_trades.items():
            pnl = sum(t.net_pnl for t in trades)
            wins = sum(1 for t in trades if t.net_pnl > 0)
            total = len(trades)
            self.per_player_performance[pid] = {
                "net_pnl": round(pnl, 2),
                "num_trades": total,
                "win_rate": round(wins / total, 4) if total > 0 else 0.0,
            }

    def _compute_regime_distribution(self) -> None:
        if not self.trades:
            return
        regime_counts: Dict[str, int] = {}
        for t in self.trades:
            r = t.regime or "Sideways"
            regime_counts[r] = regime_counts.get(r, 0) + 1
        total = sum(regime_counts.values())
        self.regime_distribution = {r: round(c / total, 4) for r, c in regime_counts.items()}


# ═══════════════════════════════════════════════════════════════════════════
# AggregateResults — cross-window statistics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AggregateResults:
    """Combined statistics across all walk-forward windows."""

    windows: List[BacktestResult] = field(default_factory=list)

    # Summary metrics (populated by compute())
    total_net_pnl: float = 0.0
    total_gross_pnl: float = 0.0
    total_costs: float = 0.0
    overall_sharpe: float = 0.0
    overall_sortino: float = 0.0
    overall_calmar: float = 0.0
    overall_max_drawdown: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    cost_drag_pct: float = 0.0

    # Equity and drawdown curves (plot-ready)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    drawdown_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    # Monthly returns
    monthly_returns: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())

    # Performance by year
    yearly_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Performance by regime
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Anti-overfit metrics
    is_vs_oos_ratio: float = 0.0
    deflated_sharpe: float = 0.0
    sharpe_t_stat: float = 0.0
    sharpe_p_value: float = 0.0
    strategy_decay_flag: bool = False
    min_trades_flag: bool = False
    overfit_flag: bool = False

    # Best/worst months
    best_month: Tuple[str, float] = ("", 0.0)
    worst_month: Tuple[str, float] = ("", 0.0)

    # All daily returns concatenated
    all_daily_returns: List[float] = field(default_factory=list)

    def compute(self) -> None:
        """Compute all aggregate metrics from the window list."""
        if not self.windows:
            return

        all_trades: List[Trade] = []
        for w in self.windows:
            all_trades.extend(w.trades)
            self.all_daily_returns.extend(w.daily_returns)

        self.num_trades = len(all_trades)
        self.total_gross_pnl = sum(t.gross_pnl for t in all_trades)
        self.total_costs = sum(t.transaction_cost for t in all_trades)
        self.total_net_pnl = sum(t.net_pnl for t in all_trades)

        winners = [t for t in all_trades if t.net_pnl > 0]
        losers = [t for t in all_trades if t.net_pnl <= 0]
        self.win_rate = len(winners) / len(all_trades) if all_trades else 0.0

        gross_wins = sum(t.net_pnl for t in winners) if winners else 0.0
        gross_losses = abs(sum(t.net_pnl for t in losers)) if losers else 0.0
        if gross_losses > 0:
            self.profit_factor = gross_wins / gross_losses
        else:
            self.profit_factor = gross_wins if gross_wins > 0 else 0.0

        if self.total_gross_pnl > 0:
            self.cost_drag_pct = self.total_costs / self.total_gross_pnl * 100
        else:
            self.cost_drag_pct = 0.0

        # Overall risk metrics from concatenated daily returns
        if self.all_daily_returns:
            rets = np.array(self.all_daily_returns)
            self.overall_sharpe = _annualized_sharpe(rets)
            self.overall_sortino = _annualized_sortino(rets)
            self.overall_max_drawdown = _max_drawdown_from_returns(rets)
            ann_return = np.mean(rets) * TRADING_DAYS_PER_YEAR
            self.overall_calmar = (
                ann_return / abs(self.overall_max_drawdown)
                if self.overall_max_drawdown != 0 else 0.0
            )

        # Equity and drawdown curves
        self._build_equity_curve()

        # Monthly returns
        self._build_monthly_returns()

        # Yearly performance
        self._build_yearly_performance(all_trades)

        # Regime performance
        self._build_regime_performance(all_trades)

        # Anti-overfit checks
        self._run_overfit_checks()

        # Sharpe significance
        self._sharpe_significance()

    def _build_equity_curve(self) -> None:
        """Build cumulative equity curve from daily returns."""
        if not self.all_daily_returns:
            return
        cumulative = np.cumprod(1.0 + np.array(self.all_daily_returns))
        self.equity_curve = pd.Series(cumulative, name="equity")

        # Drawdown curve
        running_max = np.maximum.accumulate(cumulative)
        dd = (cumulative - running_max) / running_max
        self.drawdown_curve = pd.Series(dd, name="drawdown")

    def _build_monthly_returns(self) -> None:
        """Build monthly returns table from per-window trades."""
        monthly: Dict[Tuple[int, int], float] = {}
        for w in self.windows:
            for t in w.trades:
                try:
                    dt = pd.Timestamp(t.exit_date)
                except Exception:
                    continue
                key = (dt.year, dt.month)
                monthly[key] = monthly.get(key, 0.0) + t.net_pnl

        if not monthly:
            return

        rows = []
        for (year, month), pnl in sorted(monthly.items()):
            rows.append({"year": year, "month": month, "pnl": pnl})
        df = pd.DataFrame(rows)
        if df.empty:
            return

        pivot = df.pivot_table(index="year", columns="month", values="pnl", aggfunc="sum")
        pivot = pivot.reindex(columns=range(1, 13), fill_value=0.0)
        pivot.columns = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        self.monthly_returns = pivot

        # Best/worst month
        all_monthly = [(f"{y}-{m:02d}", pnl) for (y, m), pnl in monthly.items()]
        if all_monthly:
            self.best_month = max(all_monthly, key=lambda x: x[1])
            self.worst_month = min(all_monthly, key=lambda x: x[1])

    def _build_yearly_performance(self, trades: List[Trade]) -> None:
        yearly: Dict[int, List[Trade]] = {}
        for t in trades:
            try:
                year = pd.Timestamp(t.exit_date).year
            except Exception:
                continue
            yearly.setdefault(year, []).append(t)

        for year, yr_trades in sorted(yearly.items()):
            pnl = sum(t.net_pnl for t in yr_trades)
            wins = sum(1 for t in yr_trades if t.net_pnl > 0)
            total = len(yr_trades)
            self.yearly_performance[year] = {
                "net_pnl": round(pnl, 2),
                "num_trades": total,
                "win_rate": round(wins / total, 4) if total > 0 else 0.0,
            }

    def _build_regime_performance(self, trades: List[Trade]) -> None:
        regime_trades: Dict[str, List[Trade]] = {}
        for t in trades:
            r = t.regime or "Sideways"
            regime_trades.setdefault(r, []).append(t)

        for regime, r_trades in regime_trades.items():
            pnl = sum(t.net_pnl for t in r_trades)
            wins = sum(1 for t in r_trades if t.net_pnl > 0)
            total = len(r_trades)
            self.regime_performance[regime] = {
                "net_pnl": round(pnl, 2),
                "num_trades": total,
                "win_rate": round(wins / total, 4) if total > 0 else 0.0,
            }

    def _run_overfit_checks(self) -> None:
        """Run anti-overfitting diagnostics."""
        if not self.windows:
            return

        # 1. IS vs OOS Sharpe ratio
        is_sharpes = [w.in_sample_sharpe for w in self.windows if w.in_sample_sharpe != 0]
        oos_sharpes = [w.sharpe for w in self.windows if w.sharpe != 0]

        avg_is = np.mean(is_sharpes) if is_sharpes else 0.0
        avg_oos = np.mean(oos_sharpes) if oos_sharpes else 0.0
        self.is_vs_oos_ratio = avg_oos / avg_is if avg_is != 0 else 0.0

        # Flag if OOS is < 50% of IS
        self.overfit_flag = (
            avg_is > 0 and self.is_vs_oos_ratio < 0.5
        )

        # 2. Minimum trades check
        self.min_trades_flag = self.num_trades < 100

        # 3. Strategy decay: compare first half OOS vs second half OOS
        if len(oos_sharpes) >= 4:
            mid = len(oos_sharpes) // 2
            first_half = np.mean(oos_sharpes[:mid])
            second_half = np.mean(oos_sharpes[mid:])
            self.strategy_decay_flag = (
                first_half > 0 and second_half < first_half * 0.5
            )

        # 4. Deflated Sharpe Ratio (Lopez de Prado)
        if self.overall_sharpe != 0 and self.all_daily_returns:
            n_windows = len(self.windows)
            self.deflated_sharpe = _deflated_sharpe_ratio(
                observed_sharpe=self.overall_sharpe,
                n_trials=max(n_windows, 1),
                n_obs=len(self.all_daily_returns),
                returns=np.array(self.all_daily_returns),
            )

    def _sharpe_significance(self) -> None:
        """t-test: is the Sharpe statistically different from zero?"""
        if not self.all_daily_returns or len(self.all_daily_returns) < 2:
            return
        from scipy import stats as sp_stats
        rets = np.array(self.all_daily_returns)
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        excess = rets - daily_rf
        t_stat, p_value = sp_stats.ttest_1samp(excess, 0.0)
        self.sharpe_t_stat = float(t_stat)
        self.sharpe_p_value = float(p_value)

    # ── Report ──────────────────────────────────────────────────────────

    def generate_report(self) -> str:
        """Generate a formatted text report and print to console."""
        lines: List[str] = []

        lines.append("=" * 72)
        lines.append("  WALK-FORWARD BACKTEST REPORT")
        lines.append("=" * 72)

        lines.append(f"\n  Windows:           {len(self.windows)}")
        lines.append(f"  Total trades:      {self.num_trades}")
        lines.append(f"  Win rate:          {self.win_rate:.1%}")
        lines.append(f"  Profit factor:     {self.profit_factor:.2f}")

        lines.append(f"\n  {'─' * 50}")
        lines.append(f"  P&L Summary")
        lines.append(f"  {'─' * 50}")
        lines.append(f"  Gross P&L:         Rs {self.total_gross_pnl:>14,.2f}")
        lines.append(f"  Transaction costs: Rs {self.total_costs:>14,.2f}")
        lines.append(f"  Net P&L:           Rs {self.total_net_pnl:>14,.2f}")
        lines.append(f"  Cost drag:         {self.cost_drag_pct:>13.2f}%")

        lines.append(f"\n  {'─' * 50}")
        lines.append(f"  Risk-Adjusted Metrics")
        lines.append(f"  {'─' * 50}")
        lines.append(f"  Sharpe ratio:      {self.overall_sharpe:>13.4f}")
        lines.append(f"  Sortino ratio:     {self.overall_sortino:>13.4f}")
        lines.append(f"  Max drawdown:      {self.overall_max_drawdown:>13.2%}")
        lines.append(f"  Calmar ratio:      {self.overall_calmar:>13.4f}")

        lines.append(f"\n  {'─' * 50}")
        lines.append(f"  Statistical Significance")
        lines.append(f"  {'─' * 50}")
        lines.append(f"  Sharpe t-stat:     {self.sharpe_t_stat:>13.4f}")
        lines.append(f"  Sharpe p-value:    {self.sharpe_p_value:>13.6f}")
        sig = "YES" if self.sharpe_t_stat > 2.0 else "NO"
        lines.append(f"  Significant (t>2): {sig:>13s}")

        lines.append(f"\n  {'─' * 50}")
        lines.append(f"  Anti-Overfitting Checks")
        lines.append(f"  {'─' * 50}")
        lines.append(f"  IS vs OOS ratio:   {self.is_vs_oos_ratio:>13.4f}")
        lines.append(f"  Deflated Sharpe:   {self.deflated_sharpe:>13.4f}")
        lines.append(f"  Overfit flag:      {'WARN' if self.overfit_flag else 'OK':>13s}"
                      f"  (OOS < 50% of IS)")
        lines.append(f"  Min trades flag:   {'WARN' if self.min_trades_flag else 'OK':>13s}"
                      f"  (< 100 trades)")
        lines.append(f"  Strategy decay:    {'WARN' if self.strategy_decay_flag else 'OK':>13s}"
                      f"  (recent OOS declining)")

        # Best/worst months
        if self.best_month[0]:
            lines.append(f"\n  {'─' * 50}")
            lines.append(f"  Monthly Extremes")
            lines.append(f"  {'─' * 50}")
            lines.append(f"  Best month:        {self.best_month[0]}  Rs {self.best_month[1]:>10,.2f}")
            lines.append(f"  Worst month:       {self.worst_month[0]}  Rs {self.worst_month[1]:>10,.2f}")

        # Yearly
        if self.yearly_performance:
            lines.append(f"\n  {'─' * 50}")
            lines.append(f"  Performance by Year")
            lines.append(f"  {'─' * 50}")
            for year, stats in sorted(self.yearly_performance.items()):
                lines.append(
                    f"    {year}:  Rs {stats['net_pnl']:>12,.2f}  "
                    f"trades={stats['num_trades']:>4d}  "
                    f"WR={stats['win_rate']:.1%}"
                )

        # Regime
        if self.regime_performance:
            lines.append(f"\n  {'─' * 50}")
            lines.append(f"  Performance by Regime")
            lines.append(f"  {'─' * 50}")
            for regime, stats in sorted(self.regime_performance.items()):
                lines.append(
                    f"    {regime:10s}  Rs {stats['net_pnl']:>12,.2f}  "
                    f"trades={stats['num_trades']:>4d}  "
                    f"WR={stats['win_rate']:.1%}"
                )

        # Per-window summary
        lines.append(f"\n  {'─' * 50}")
        lines.append(f"  Per-Window OOS Results")
        lines.append(f"  {'─' * 50}")
        for w in self.windows:
            lines.append(
                f"    W{w.window_index:>3d}  "
                f"{w.window_start}→{w.window_end}  "
                f"PnL={w.net_pnl:>10,.2f}  "
                f"Sharpe={w.sharpe:>6.2f}  "
                f"DD={w.max_drawdown:>7.2%}  "
                f"trades={len(w.trades):>3d}"
            )

        lines.append(f"\n{'=' * 72}")

        report = "\n".join(lines)
        print(report)
        return report

    # ── Visualization ───────────────────────────────────────────────────

    def save_visualizations(self, output_dir: str = "backtest_output") -> List[str]:
        """Generate and save all visualizations. Returns list of saved file paths."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: List[str] = []

        # 1. Equity curve with drawdown subplot
        if len(self.equity_curve) > 0:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                           sharex=True)
            ax1.plot(self.equity_curve.values, color="steelblue", linewidth=1.0)
            ax1.set_title("Walk-Forward Equity Curve (Out-of-Sample)", fontweight="bold")
            ax1.set_ylabel("Cumulative Return (1.0 = start)")
            ax1.axhline(y=1.0, color="grey", linestyle="--", alpha=0.5)
            ax1.grid(alpha=0.3)

            ax2.fill_between(range(len(self.drawdown_curve)), self.drawdown_curve.values,
                             color="salmon", alpha=0.6)
            ax2.set_ylabel("Drawdown")
            ax2.set_xlabel("Trading Days (OOS)")
            ax2.grid(alpha=0.3)

            fig.tight_layout()
            path = str(out / "equity_curve.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

        # 2. Monthly returns heatmap
        if not self.monthly_returns.empty:
            try:
                import seaborn as sns
                fig, ax = plt.subplots(figsize=(14, max(4, len(self.monthly_returns) * 0.6)))
                sns.heatmap(
                    self.monthly_returns,
                    annot=True, fmt=",.0f", cmap="RdYlGn", center=0,
                    linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Net P&L (Rs)"},
                )
                ax.set_title("Monthly Returns Heatmap", fontweight="bold")
                ax.set_ylabel("Year")
                fig.tight_layout()
                path = str(out / "monthly_returns.png")
                fig.savefig(path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved.append(path)
            except ImportError:
                logger.warning("seaborn not available, skipping heatmap")

        # 3. Rolling Sharpe (30-day window)
        if len(self.all_daily_returns) >= 30:
            rets = np.array(self.all_daily_returns)
            rolling_window = 30
            rolling_sharpes = []
            for i in range(rolling_window, len(rets) + 1):
                window_rets = rets[i - rolling_window:i]
                rolling_sharpes.append(_annualized_sharpe(window_rets))

            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(rolling_sharpes, color="steelblue", linewidth=1.0)
            ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
            ax.axhline(y=2.0, color="green", linestyle="--", alpha=0.3, label="Sharpe = 2.0")
            ax.set_title("Rolling 30-Day Sharpe Ratio (OOS)", fontweight="bold")
            ax.set_xlabel("Trading Days (OOS)")
            ax.set_ylabel("Annualized Sharpe")
            ax.legend()
            ax.grid(alpha=0.3)
            fig.tight_layout()
            path = str(out / "rolling_sharpe.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

        # 4. Per-player P&L contribution over time
        player_cumulative: Dict[str, List[float]] = {}
        for w in self.windows:
            for t in w.trades:
                pid = t.player_id or "UNKNOWN"
                if pid not in player_cumulative:
                    player_cumulative[pid] = [0.0]
                player_cumulative[pid].append(player_cumulative[pid][-1] + t.net_pnl)

        if player_cumulative:
            fig, ax = plt.subplots(figsize=(14, 6))
            for pid, cum_pnl in sorted(player_cumulative.items()):
                label = PLAYER_LABELS.get(pid, pid)
                ax.plot(cum_pnl, label=f"{pid} ({label})", linewidth=1.0)
            ax.set_title("Per-Player Cumulative P&L (OOS)", fontweight="bold")
            ax.set_xlabel("Trade Number")
            ax.set_ylabel("Cumulative Net P&L (Rs)")
            ax.legend(loc="best", fontsize=8)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            path = str(out / "player_contribution.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

        # 5. Regime timeline with equity curve overlay
        if len(self.equity_curve) > 0 and any(w.regime_distribution for w in self.windows):
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(self.equity_curve.values, color="black", linewidth=0.8, label="Equity")

            # Color bands for each window's dominant regime
            colours = {"Bull": "#c8f7c5", "Bear": "#f7c5c5", "Sideways": "#f7f0c5"}
            day_idx = 0
            for w in self.windows:
                n_days = len(w.daily_returns)
                if n_days == 0:
                    continue
                dominant = max(w.regime_distribution, key=w.regime_distribution.get) \
                    if w.regime_distribution else "Sideways"
                ax.axvspan(day_idx, day_idx + n_days,
                           alpha=0.3, color=colours.get(dominant, "#f7f0c5"))
                day_idx += n_days

            from matplotlib.patches import Patch
            legend_patches = [
                Patch(facecolor=colours[r], edgecolor="grey", label=r, alpha=0.5)
                for r in ["Bull", "Bear", "Sideways"]
            ]
            legend_patches.insert(0, plt.Line2D([0], [0], color="black",
                                                linewidth=0.8, label="Equity"))
            ax.legend(handles=legend_patches, loc="upper left", fontsize=9)
            ax.set_title("Equity Curve with Regime Overlay (OOS)", fontweight="bold")
            ax.set_xlabel("Trading Days (OOS)")
            ax.set_ylabel("Cumulative Return")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            path = str(out / "regime_overlay.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved.append(path)

        if saved:
            print(f"\n  Saved {len(saved)} visualizations to {out}/")
            for p in saved:
                print(f"    {p}")

        return saved


# ═══════════════════════════════════════════════════════════════════════════
# WalkForwardEngine
# ═══════════════════════════════════════════════════════════════════════════

class WalkForwardEngine:
    """Rolling walk-forward backtest engine.

    For each step:
        1. Slice training window → fit strategy / regime detector
        2. Slice testing window  → run strategy out-of-sample
        3. Apply transaction costs and risk limits
        4. Record BacktestResult
        5. Roll forward by step_size days
    """

    def __init__(
        self,
        training_window: int = 120,
        testing_window: int = 20,
        step_size: int = 20,
        initial_capital: float = 500_000,
        cost_model: Optional[Any] = None,
        risk_manager: Optional[Any] = None,
        regime_detector: Optional[Any] = None,
        nifty_data: Optional[pd.DataFrame] = None,
    ):
        self.training_window = training_window
        self.testing_window = testing_window
        self.step_size = step_size
        self.initial_capital = initial_capital

        # Optional components
        self.cost_model = cost_model
        if self.cost_model is None and _HAS_COSTS:
            self.cost_model = TransactionCostModel()

        self.risk_manager = risk_manager
        if self.risk_manager is None and _HAS_RISK:
            self.risk_manager = PortfolioRiskManager(total_capital=initial_capital)

        self.regime_detector = regime_detector
        self.nifty_data = nifty_data

    def run(
        self,
        data: Dict[str, pd.DataFrame],
        strategy_fn: Callable,
        verbose: bool = True,
    ) -> AggregateResults:
        """Execute the full walk-forward backtest.

        Args:
            data: Dict of {symbol: DataFrame} with OHLCV columns and
                  DatetimeIndex. All DataFrames should cover the same date range.
            strategy_fn: Callable matching the StrategyFn protocol.
            verbose: Print progress to console.

        Returns:
            AggregateResults with all windows computed.
        """
        # Build a unified date index from all symbols
        all_dates = self._build_date_index(data)
        total_days = len(all_dates)
        min_required = self.training_window + self.testing_window

        if total_days < min_required:
            raise ValueError(
                f"Need at least {min_required} trading days, got {total_days}. "
                f"Provide more data or reduce window sizes."
            )

        # Generate window boundaries
        windows = self._generate_windows(all_dates)

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  Walk-Forward Backtest")
            print(f"{'=' * 60}")
            print(f"  Date range:     {all_dates[0].strftime('%Y-%m-%d')} → "
                  f"{all_dates[-1].strftime('%Y-%m-%d')}")
            print(f"  Total days:     {total_days}")
            print(f"  Train window:   {self.training_window} days")
            print(f"  Test window:    {self.testing_window} days")
            print(f"  Step size:      {self.step_size} days")
            print(f"  Windows:        {len(windows)}")
            print(f"  Symbols:        {len(data)}")
            print(f"  Capital:        Rs {self.initial_capital:,.0f}")
            print(f"{'=' * 60}\n")

        results: List[BacktestResult] = []
        fitted_params: Optional[Dict[str, Any]] = None

        for i, (train_start_idx, train_end_idx, test_start_idx, test_end_idx) in enumerate(windows):
            train_dates = all_dates[train_start_idx:train_end_idx]
            test_dates = all_dates[test_start_idx:test_end_idx]

            if len(train_dates) == 0 or len(test_dates) == 0:
                continue

            # Slice data for train and test
            train_data = self._slice_data(data, train_dates[0], train_dates[-1])
            test_data = self._slice_data(data, test_dates[0], test_dates[-1])

            # Fit regime detector on training data if available
            regime = "Sideways"
            if self.regime_detector is not None and self.nifty_data is not None:
                try:
                    nifty_train = self.nifty_data.loc[
                        (self.nifty_data.index >= train_dates[0]) &
                        (self.nifty_data.index <= train_dates[-1])
                    ]
                    if len(nifty_train) >= 60:
                        self.regime_detector.fit(nifty_train)
                        nifty_test = self.nifty_data.loc[
                            (self.nifty_data.index >= test_dates[0]) &
                            (self.nifty_data.index <= test_dates[-1])
                        ]
                        if len(nifty_test) > 0:
                            nifty_recent = self.nifty_data.loc[
                                self.nifty_data.index <= test_dates[-1]
                            ].tail(60)
                            regime, _, _ = self.regime_detector.predict(nifty_recent)
                except Exception as e:
                    logger.warning(f"Regime detection failed in window {i}: {e}")

            # Pass regime to strategy via params
            strategy_params = dict(fitted_params) if fitted_params else {}
            strategy_params["regime"] = regime

            # Run strategy
            try:
                trades, fitted_params = strategy_fn(train_data, test_data, strategy_params)
            except Exception as e:
                logger.error(f"Strategy failed in window {i}: {e}")
                trades = []

            # Tag trades with regime
            for t in trades:
                if not t.regime:
                    t.regime = regime

            # Apply transaction costs
            if self.cost_model is not None:
                for t in trades:
                    self._apply_costs(t)

            # Apply risk limits
            risk_events: List[str] = []
            if self.risk_manager is not None:
                trades, risk_events = self._apply_risk_limits(trades)

            # Build daily returns for the test window
            daily_returns = self._compute_daily_returns(trades, test_dates)

            # Compute in-sample Sharpe for overfit detection
            is_sharpe = 0.0
            try:
                is_trades, _ = strategy_fn(train_data, train_data, fitted_params)
                is_daily = self._compute_daily_returns(is_trades, train_dates)
                if is_daily:
                    is_sharpe = _annualized_sharpe(np.array(is_daily))
            except Exception:
                pass

            # Build result
            result = BacktestResult(
                window_index=i,
                window_start=test_dates[0].strftime("%Y-%m-%d"),
                window_end=test_dates[-1].strftime("%Y-%m-%d"),
                train_start=train_dates[0].strftime("%Y-%m-%d"),
                train_end=train_dates[-1].strftime("%Y-%m-%d"),
                trades=trades,
                daily_returns=daily_returns,
                in_sample_sharpe=is_sharpe,
            )
            result.compute_metrics()
            results.append(result)

            if verbose:
                print(
                    f"  Window {i:>3d}  "
                    f"train={train_dates[0].strftime('%Y-%m-%d')}→{train_dates[-1].strftime('%Y-%m-%d')}  "
                    f"test={test_dates[0].strftime('%Y-%m-%d')}→{test_dates[-1].strftime('%Y-%m-%d')}  "
                    f"trades={len(trades):>3d}  "
                    f"PnL={result.net_pnl:>10,.2f}  "
                    f"Sharpe={result.sharpe:>6.2f}  "
                    f"regime={regime}"
                )

        # Aggregate
        aggregate = AggregateResults(windows=results)
        aggregate.compute()

        if verbose:
            print(f"\n{'─' * 60}")
            print(f"  Walk-forward complete. {len(results)} windows processed.")
            print(f"{'─' * 60}")

        return aggregate

    # ── Internal helpers ────────────────────────────────────────────────

    def _build_date_index(self, data: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        """Build a sorted union of all trading dates across symbols."""
        all_dates = set()
        for symbol, df in data.items():
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            all_dates.update(df.index)
        return pd.DatetimeIndex(sorted(all_dates))

    def _generate_windows(
        self, dates: pd.DatetimeIndex
    ) -> List[Tuple[int, int, int, int]]:
        """Generate (train_start, train_end, test_start, test_end) index tuples."""
        n = len(dates)
        windows = []
        start = 0
        while True:
            train_start = start
            train_end = train_start + self.training_window
            test_start = train_end
            test_end = test_start + self.testing_window

            if test_end > n:
                # Use remaining days if enough for at least 5 test days
                if test_start < n and (n - test_start) >= 5:
                    test_end = n
                else:
                    break

            windows.append((train_start, train_end, test_start, test_end))
            start += self.step_size

            if test_end >= n:
                break

        return windows

    def _slice_data(
        self,
        data: Dict[str, pd.DataFrame],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Dict[str, pd.DataFrame]:
        """Slice all DataFrames to [start, end] range."""
        sliced = {}
        for symbol, df in data.items():
            mask = (df.index >= start) & (df.index <= end)
            sub = df.loc[mask]
            if len(sub) > 0:
                sliced[symbol] = sub
        return sliced

    def _apply_costs(self, trade: Trade) -> None:
        """Apply TransactionCostModel to a trade."""
        if self.cost_model is None:
            return

        entry_cost = self.cost_model.calculate_cost(
            price=trade.entry_price,
            quantity=trade.quantity,
            side="BUY" if trade.direction == "LONG" else "SELL",
            trade_type="intraday",
            signal_type=trade.signal_type,
        )
        exit_cost = self.cost_model.calculate_cost(
            price=trade.exit_price,
            quantity=trade.quantity,
            side="SELL" if trade.direction == "LONG" else "BUY",
            trade_type="intraday",
            signal_type=trade.signal_type,
        )
        trade.transaction_cost = entry_cost + exit_cost
        trade.net_pnl = trade.gross_pnl - trade.transaction_cost

    def _apply_risk_limits(
        self, trades: List[Trade]
    ) -> Tuple[List[Trade], List[str]]:
        """Apply risk manager filtering (simplified for backtest context)."""
        if self.risk_manager is None:
            return trades, []

        risk_events: List[str] = []
        # In a walk-forward backtest, we simulate risk checks by
        # tracking cumulative exposure and flagging breaches.
        # Full signal-level filtering happens in the strategy_fn.
        capital = self.initial_capital
        cumulative_pnl = 0.0

        for t in trades:
            cumulative_pnl += t.net_pnl
            equity = capital + cumulative_pnl

            # Update risk manager equity for circuit breaker
            self.risk_manager.update_equity(equity)
            cb = self.risk_manager.check_circuit_breaker()
            if cb["triggered"]:
                event = f"Circuit breaker: {cb['reason']}"
                t.risk_event = event
                risk_events.append(event)

        return trades, risk_events

    def _compute_daily_returns(
        self, trades: List[Trade], dates: pd.DatetimeIndex
    ) -> List[float]:
        """Distribute trade P&L across dates to produce daily returns."""
        if not trades or len(dates) == 0:
            return [0.0] * max(len(dates), 1)

        capital = self.initial_capital
        daily_pnl = {d: 0.0 for d in dates}

        for t in trades:
            try:
                exit_dt = pd.Timestamp(t.exit_date)
            except Exception:
                continue

            # Find nearest date in our index
            nearest_idx = dates.get_indexer([exit_dt], method="nearest")[0]
            if 0 <= nearest_idx < len(dates):
                d = dates[nearest_idx]
                daily_pnl[d] += t.net_pnl

        # Convert to returns
        running_capital = capital
        daily_returns = []
        for d in dates:
            pnl = daily_pnl.get(d, 0.0)
            ret = pnl / running_capital if running_capital > 0 else 0.0
            daily_returns.append(ret)
            running_capital += pnl

        return daily_returns


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════

def _annualized_sharpe(returns: np.ndarray) -> float:
    """Compute annualized Sharpe ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    mean_excess = np.mean(excess)
    std = np.std(excess, ddof=1)
    if std < 1e-10:
        return 0.0
    return float(mean_excess / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _annualized_sortino(returns: np.ndarray) -> float:
    """Compute annualized Sortino ratio from daily returns."""
    if len(returns) < 2:
        return 0.0
    daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    excess = returns - daily_rf
    mean_excess = np.mean(excess)
    downside = excess[excess < 0]
    if len(downside) < 1:
        return float(mean_excess * np.sqrt(TRADING_DAYS_PER_YEAR)) if mean_excess > 0 else 0.0
    downside_std = np.std(downside, ddof=1)
    if downside_std < 1e-10:
        return 0.0
    return float(mean_excess / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown_from_returns(returns: np.ndarray) -> float:
    """Compute maximum drawdown from daily returns array."""
    if len(returns) == 0:
        return 0.0
    cumulative = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return float(np.min(drawdowns))


def _deflated_sharpe_ratio(
    observed_sharpe: float,
    n_trials: int,
    n_obs: int,
    returns: np.ndarray,
) -> float:
    """Compute the Deflated Sharpe Ratio (Lopez de Prado, 2014).

    Adjusts the observed Sharpe ratio for:
    - Multiple testing (n_trials)
    - Non-normality (skewness, kurtosis)
    - Sample size

    Returns a DSR value; values > 0.95 suggest the Sharpe is
    unlikely to be due to chance.
    """
    from scipy import stats as sp_stats

    if n_obs < 10 or n_trials < 1:
        return 0.0

    # Moments of the returns
    skew = float(sp_stats.skew(returns))
    kurt = float(sp_stats.kurtosis(returns, fisher=True))  # excess kurtosis

    # Expected maximum Sharpe under null (i.i.d. normal returns)
    # E[max(SR)] ≈ sqrt(2 * ln(n_trials)) - (euler_gamma + ln(ln(n_trials))) / (2 * sqrt(2 * ln(n_trials)))
    if n_trials <= 1:
        e_max_sr = 0.0
    else:
        z = math.sqrt(2.0 * math.log(n_trials))
        euler_gamma = 0.5772156649
        e_max_sr = z - (euler_gamma + math.log(math.log(n_trials))) / (2.0 * z) if z > 0 else 0.0

    # Standard error of the Sharpe ratio (accounting for non-normality)
    # SE(SR) ≈ sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (n-1))
    sr = observed_sharpe / math.sqrt(TRADING_DAYS_PER_YEAR) if TRADING_DAYS_PER_YEAR > 0 else observed_sharpe
    se_sr_sq = (1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr) / max(n_obs - 1, 1)
    se_sr = math.sqrt(max(se_sr_sq, 1e-10))

    # DSR = Prob(SR* < observed_SR) where SR* ~ N(E[max(SR)], SE(SR))
    if se_sr < 1e-10:
        return 0.0
    z_score = (sr - e_max_sr) / se_sr
    dsr = float(sp_stats.norm.cdf(z_score))
    return dsr


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline Strategy: 5-Player Indicator Signals + Regime + Risk Mgmt
# ═══════════════════════════════════════════════════════════════════════════

# Mapping from evolved_player_configs indicator names → column names.
# New indicators are computed directly in indicator_computer.py and are
# already in [-1, 1] range, so they map to themselves (no _norm suffix).
_CONFIG_TO_FALLBACK: Dict[str, str] = {
    # ── Legacy direct matches ─────────────────────────────────────────
    "RSI_7": "RSI_7_norm",
    "RSI_14": "RSI_14_norm",
    "RSI_21": "RSI_21_norm",
    "WILLR_14": "WILLR_14_norm",
    "ATR_14": "ATR_14_norm",
    "PSAR": "PSAR_norm",
    "ADX_14": "ADX_14_norm",
    "CCI_14": "CCI_20_norm",
    "CMF_21": "CMF_20_norm",
    "MACD_8_17_9": "MACD_12_26_9_norm",
    "DONCHIAN_20": "DC_high_norm",
    "STOCH_5_3": "STOCH_k_norm",
    "STOCH_14_3": "STOCH_k_norm",
    "BBANDS_20_2": "BB_pct_norm",
    "BBANDS_10_1.5": "BB_pct_norm",
    "KC_20_2": "KC_mid_norm",
    "CMO_14": "RSI_14_norm",
    "TSI_13_25": "MACD_hist_norm",
    "UO_7_14_28": "MFI_14_norm",
    "KAMA_20": "EMA_21_norm",
    "SUPERTREND_10_2": "ATR_14_norm",
    "NATR_14": "ATR_14_norm",
    # ── New indicators (already in [-1,1] or need _norm) ─────────────
    "PSAR_signal": "PSAR_signal",
    "ICHIMOKU_signal": "ICHIMOKU_signal",
    "DONCHIAN_breakout": "DONCHIAN_breakout",
    "HA_trend": "HA_trend",
    "BB_squeeze_signal": "BB_squeeze_signal",
    "STOCH_divergence": "STOCH_divergence",
    "ELDER_force": "ELDER_force",
    "VWAP_deviation": "VWAP_deviation",
    "VWAP_dev_raw": "VWAP_dev_raw",
    "OBV_slope": "OBV_slope",
    "ATR_ratio": "ATR_ratio_norm",
    "KC_breakout": "KC_breakout",
    "NATR": "NATR_norm",
    "ROC_20": "ROC_20",
    "MACD_hist_slope": "MACD_hist_slope",
    "HIGH_52W_prox": "HIGH_52W_prox",
    "BB_width_raw": "BB_width_raw_norm",
    "MFI_14": "MFI_14_norm",
}


def _compute_player_signal(
    normalized_indicators: pd.DataFrame,
    weights: Dict[str, float],
    idx: int,
) -> Tuple[float, float]:
    """Compute a single player's weighted signal and confidence at bar index `idx`.

    Returns:
        (direction, confidence) where:
        - direction is in [-1, 1] — the raw clamped weighted average
        - confidence is in [0, 1] — fraction of indicators agreeing on direction
    """
    weighted_sum = 0.0
    total_weight = 0.0
    agree_positive = 0
    agree_negative = 0
    total_resolved = 0

    for config_name, weight in weights.items():
        if weight == 0:
            continue

        # Resolve indicator column name
        col = config_name + "_norm"
        if col not in normalized_indicators.columns:
            col = _CONFIG_TO_FALLBACK.get(config_name, "")
        if not col or col not in normalized_indicators.columns:
            continue

        val = normalized_indicators.iloc[idx][col]
        if pd.isna(val):
            continue

        weighted_sum += val * weight
        total_weight += abs(weight)
        total_resolved += 1

        # Track directional agreement for confidence
        if val > 0.05:
            agree_positive += 1
        elif val < -0.05:
            agree_negative += 1

    if total_weight == 0:
        return 0.0, 0.0

    # Bug #4 fix: use clamp instead of tanh(raw * 1.5).
    # tanh(x*1.5) compressed signals so they rarely exceeded 0.3-0.4.
    # Raw weighted average is already bounded roughly by [-1, 1] since
    # inputs are in [-1, 1].  Clamp directly.
    raw = weighted_sum / total_weight
    direction = float(np.clip(raw, -1.0, 1.0))

    # Bug #2 fix: confidence = indicator agreement, NOT abs(signal).
    # Old: confidence = abs(signal) → created s² bias in the voter.
    # New: confidence = fraction of indicators agreeing with the majority direction.
    if total_resolved == 0:
        confidence = 0.0
    else:
        majority = max(agree_positive, agree_negative)
        confidence = majority / total_resolved

    return direction, confidence


def _load_player_configs() -> Dict[str, Dict[str, Any]]:
    """Load evolved player configs from JSON."""
    cfg_path = Path(__file__).parent / "evolved_player_configs.json"
    if cfg_path.exists():
        import json as _json
        raw = _json.loads(cfg_path.read_text(encoding="utf-8"))
        return raw.get("configs", {})

    # Fallback: minimal default configs
    return {
        pid: {
            "label": label,
            "weights": {"RSI_14": 0.5, "BBANDS_20_2": 0.3, "STOCH_14_3": 0.2},
            "entry_threshold": 0.40,
            "exit_threshold": -0.12,
            "min_hold_bars": 3,
        }
        for pid, label in PLAYER_LABELS.items()
    }


def _full_pipeline_strategy(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Trade], Dict[str, Any]]:
    """Full 5-player indicator pipeline with regime-aware signal combination.

    Training phase:
      - Fit regime detector on training window
      - Compute indicators on training data for warm-up
      - Build player return history for HRP allocation

    Testing phase (bar by bar):
      1. Compute indicators for each symbol
      2. 5 RegimeAwarePlayers generate signals from indicator weights
      3. SignalCombiner combines with Bayesian + regime weights
         (confidence threshold 0.6, flip threshold 0.3 enforced)
      4. PortfolioOptimizer scales via vol targeting
      5. Min holding period (3 bars) enforced before exits
      6. Trades recorded with full attribution
    """
    if params is None:
        params = {}

    # Load player configs and build regime-aware players
    player_configs = _load_player_configs()
    regime_players = {}
    if _HAS_PLAYERS:
        regime_players = build_player_registry()

    # Determine regime from training data
    regime = params.get("regime", "Sideways")

    # Adjust configs for regime
    adjusted_configs: Dict[str, Dict[str, Any]] = {}
    for pid in PLAYER_IDS:
        base_cfg = player_configs.get(pid, {})
        if pid in regime_players and base_cfg:
            adjusted_configs[pid] = regime_players[pid].adjust_config(base_cfg, regime)
        else:
            cfg = dict(base_cfg)
            cfg["position_sizing_multiplier"] = 1.0
            adjusted_configs[pid] = cfg

    # Set up signal combiner
    combiner = None
    if _HAS_COMBINER:
        from tempfile import mkdtemp
        combiner = RegimeAwareCombiner(state_dir=Path(mkdtemp()))

    # Set up indicator computer
    indicator_computer = None
    if _HAS_INDICATORS:
        indicator_computer = IndicatorComputer()

    # Set up portfolio optimizer for vol targeting
    vol_targeter = None
    if _HAS_OPTIMIZER:
        vol_targeter = VolatilityTargeter()

    # Regime weights (from regime_detector if available)
    regime_weights = None
    if _HAS_REGIME:
        try:
            from regime_detector import get_regime_weights
            regime_weights = get_regime_weights(regime)
        except (ImportError, Exception):
            pass

    # Track portfolio returns for vol targeting
    portfolio_returns_list: List[float] = []

    # ── Testing phase: bar-by-bar signal generation ──────────────────

    trades: List[Trade] = []

    # Per-symbol state
    # position: None or (entry_date, entry_price, direction, player_id, bars_held, sizing_mult)
    positions: Dict[str, Optional[Tuple]] = {}
    # Pre-compute indicators for all symbols
    symbol_indicators: Dict[str, pd.DataFrame] = {}

    for symbol, test_df in test_data.items():
        # Concatenate train tail for indicator warm-up
        warmup_len = 60
        if symbol in train_data:
            warmup = train_data[symbol].tail(warmup_len)
            combined = pd.concat([warmup, test_df])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = test_df

        if len(combined) < 40:
            continue

        # Compute indicators
        if indicator_computer is not None:
            try:
                indicators = indicator_computer.compute_all_indicators(combined)
                symbol_indicators[symbol] = indicators
            except Exception as e:
                logger.warning(f"Indicator computation failed for {symbol}: {e}")
        positions[symbol] = None

    if not symbol_indicators:
        return [], params

    # Get test date range
    all_test_dates = set()
    for sym, df in test_data.items():
        all_test_dates.update(df.index.tolist())
    test_dates_sorted = sorted(all_test_dates)

    if not test_dates_sorted:
        return [], params

    # Bar-by-bar loop over test dates
    for bar_date in test_dates_sorted:
        daily_pnl = 0.0

        # Increment bars_held for all open positions
        for sym in list(positions.keys()):
            pos = positions[sym]
            if pos is not None:
                entry_date, entry_price, direction, pid, bars_held, sizing = pos
                positions[sym] = (entry_date, entry_price, direction, pid, bars_held + 1, sizing)

        # Generate signals for each symbol
        for symbol in list(symbol_indicators.keys()):
            indicators = symbol_indicators[symbol]

            # Check this date exists in the indicator index
            if bar_date not in indicators.index:
                continue

            bar_idx = indicators.index.get_loc(bar_date)
            if isinstance(bar_idx, slice):
                bar_idx = bar_idx.start

            # Get current price
            test_df = test_data.get(symbol)
            if test_df is None or bar_date not in test_df.index:
                continue
            close_col = "Close" if "Close" in test_df.columns else "close"
            price = float(test_df.loc[bar_date, close_col])

            # Compute each player's signal
            player_signals: List[PlayerSignal] = []
            player_raw_signals: Dict[str, float] = {}

            for pid in PLAYER_IDS:
                cfg = adjusted_configs.get(pid, {})
                weights = cfg.get("weights", {})
                direction, confidence = _compute_player_signal(indicators, weights, bar_idx)
                player_raw_signals[pid] = direction

                if _HAS_COMBINER:
                    player_signals.append(PlayerSignal(
                        player_id=pid,
                        direction=direction,
                        confidence=confidence,
                    ))

            # Combine signals
            should_trade = False
            final_signal = 0.0
            sizing_multiplier = 1.0

            if combiner is not None and player_signals:
                combined_signal = combiner.combine(
                    player_signals, regime, regime_weights
                )
                final_signal = combined_signal.final_signal
                should_trade = combined_signal.should_trade
                sizing_multiplier = combined_signal.position_size_multiplier
            else:
                # Fallback: simple average of player signals
                avg = np.mean(list(player_raw_signals.values()))
                final_signal = float(avg)
                should_trade = abs(final_signal) >= 0.7
                sizing_multiplier = 1.0

            # Vol targeting scale
            vol_scale = 1.0
            if vol_targeter and len(portfolio_returns_list) >= 20:
                scale, _, _ = vol_targeter.get_scale_factor(
                    np.array(portfolio_returns_list), regime
                )
                vol_scale = scale

            # Position management for this symbol
            pos = positions.get(symbol)

            if pos is None:
                # Use per-player entry threshold from config (NOT hardcoded).
                # The SignalCombiner already filters via MIN_CONFIDENCE_THRESHOLD.
                # Here we use the regime-adjusted threshold from the player whose
                # signal contributed most.  Default to 0.35 if not found.
                best_player_for_threshold = None
                best_abs_signal = -1.0
                for _pid in PLAYER_IDS:
                    _abs = abs(player_raw_signals.get(_pid, 0))
                    if _abs > best_abs_signal:
                        best_abs_signal = _abs
                        best_player_for_threshold = _pid
                entry_threshold = adjusted_configs.get(
                    best_player_for_threshold, {}
                ).get("entry_threshold", 0.35)

                # No position — consider entry
                if should_trade and abs(final_signal) >= entry_threshold:
                    direction = "LONG" if final_signal > 0 else "SHORT"

                    # Position size: base Rs 50k, scaled by sizing + vol
                    base_size = 50_000
                    adj_size = base_size * sizing_multiplier * vol_scale

                    # Attribute to player whose signal direction matches AND
                    # has highest weighted contribution (confidence * blended weight)
                    dir_sign = 1.0 if direction == "LONG" else -1.0
                    agreeing = [
                        p for p in PLAYER_IDS
                        if player_raw_signals.get(p, 0) * dir_sign > 0
                    ]
                    if agreeing:
                        best_pid = agreeing[len(trades) % len(agreeing)]
                    else:
                        best_pid = PLAYER_IDS[len(trades) % len(PLAYER_IDS)]
                    # Get that player's sizing multiplier from regime config
                    player_sizing = adjusted_configs.get(best_pid, {}).get(
                        "position_sizing_multiplier", 1.0
                    )
                    adj_size *= player_sizing

                    quantity = max(1, int(adj_size / price))
                    positions[symbol] = (
                        str(bar_date.date()) if hasattr(bar_date, 'date') else str(bar_date),
                        price, direction, best_pid, 0, sizing_multiplier,
                    )
            else:
                # Have position — check exit
                entry_date, entry_price, direction, pid, bars_held, sizing = pos

                # Min holding period check (3 bars)
                if bars_held < 3:
                    continue

                # Exit conditions
                exit_signal = False
                exit_reason = ""

                if direction == "LONG" and final_signal < -0.3:
                    exit_signal = True
                    exit_reason = "signal_reversal"
                elif direction == "SHORT" and final_signal > 0.3:
                    exit_signal = True
                    exit_reason = "signal_reversal"
                elif direction == "LONG" and final_signal < -0.1 and bars_held >= 8:
                    exit_signal = True
                    exit_reason = "weak_signal_timeout"
                elif direction == "SHORT" and final_signal > 0.1 and bars_held >= 8:
                    exit_signal = True
                    exit_reason = "weak_signal_timeout"

                # Window-end exit
                if bar_date == test_dates_sorted[-1] and not exit_signal:
                    exit_signal = True
                    exit_reason = "window_end"

                if exit_signal:
                    quantity = max(1, int(50_000 * sizing / price))
                    if direction == "LONG":
                        gross_pnl = (price - entry_price) * quantity
                    else:
                        gross_pnl = (entry_price - price) * quantity

                    trades.append(Trade(
                        symbol=symbol,
                        direction=direction,
                        entry_date=entry_date,
                        exit_date=str(bar_date.date()) if hasattr(bar_date, 'date') else str(bar_date),
                        entry_price=entry_price,
                        exit_price=price,
                        quantity=quantity,
                        gross_pnl=gross_pnl,
                        net_pnl=gross_pnl,
                        player_id=pid,
                        bars_held=bars_held,
                        signal_type="momentum" if abs(final_signal) > 0.8 else "neutral",
                        regime=regime,
                        exit_reason=exit_reason,
                    ))

                    daily_pnl += gross_pnl
                    positions[symbol] = None

                    # Record outcome for Bayesian updating
                    if combiner is not None:
                        try:
                            combiner.record_trade_outcome(TradeOutcome(
                                player_id=pid,
                                signal_direction=1.0 if direction == "LONG" else -1.0,
                                actual_pnl=gross_pnl,
                                regime=regime,
                            ))
                        except Exception:
                            pass

        # Track portfolio returns
        portfolio_returns_list.append(daily_pnl / 500_000)

    fitted_params = dict(params)
    fitted_params["regime"] = regime
    fitted_params["strategy"] = "full_pipeline"

    return trades, fitted_params


# ═══════════════════════════════════════════════════════════════════════════
# Walk-Forward Config Re-Evolution
# ═══════════════════════════════════════════════════════════════════════════

# All indicator weight keys the system can use
_ALL_INDICATOR_KEYS = [
    "RSI_7", "RSI_14", "STOCH_5_3", "STOCH_14_3", "CCI_14",
    "WILLR_14", "CMO_14", "BBANDS_20_2", "KC_20_2", "ATR_14",
    "MACD_8_17_9", "UO_7_14_28", "PSAR", "TSI_13_25",
    "BBANDS_10_1.5", "CMF_21", "DONCHIAN_20", "NATR_14",
    "KAMA_20", "SUPERTREND_10_2",
]


def _evaluate_config_on_data(
    data: Dict[str, pd.DataFrame],
    weights: Dict[str, float],
    entry_threshold: float,
    exit_threshold: float,
    min_hold_bars: int,
) -> Tuple[float, int, float]:
    """Evaluate a single player config on data. Returns (sharpe, n_trades, total_pnl)."""
    if not _HAS_INDICATORS:
        return 0.0, 0, 0.0

    indicator_computer = IndicatorComputer()
    trades_pnl: List[float] = []
    daily_returns: List[float] = []

    for symbol, df in data.items():
        if len(df) < 40:
            continue

        try:
            indicators = indicator_computer.compute_all_indicators(df)
        except Exception:
            continue

        close_col = "Close" if "Close" in df.columns else "close"
        position = None  # (entry_price, direction, bars_held)

        for idx in range(30, len(indicators)):
            if indicators.index[idx] not in df.index:
                continue
            price = float(df.loc[indicators.index[idx], close_col])
            signal = _compute_player_signal(indicators, weights, idx)

            if position is None:
                if abs(signal) >= entry_threshold:
                    direction = "LONG" if signal > 0 else "SHORT"
                    position = (price, direction, 0)
            else:
                entry_price, direction, bars_held = position
                bars_held += 1
                position = (entry_price, direction, bars_held)

                exit_now = False
                if bars_held >= min_hold_bars:
                    if direction == "LONG" and signal < exit_threshold:
                        exit_now = True
                    elif direction == "SHORT" and signal > -exit_threshold:
                        exit_now = True
                    elif bars_held >= 15:
                        exit_now = True

                if idx == len(indicators) - 1:
                    exit_now = True

                if exit_now:
                    if direction == "LONG":
                        pnl = price - entry_price
                    else:
                        pnl = entry_price - price
                    trades_pnl.append(pnl)
                    daily_returns.append(pnl / entry_price)
                    position = None

    if len(trades_pnl) < 3:
        return 0.0, len(trades_pnl), sum(trades_pnl)

    returns = np.array(daily_returns)
    mean_r = np.mean(returns)
    std_r = np.std(returns, ddof=1)
    sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 1e-10 else 0.0
    return sharpe, len(trades_pnl), sum(trades_pnl)


def _reevolve_configs(
    data: Dict[str, pd.DataFrame],
    n_iterations: int = 30,
    verbose: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Re-evolve player configs using walk-forward validation.

    Splits data into train (first 2 years) and test (last year).
    For each player, randomly perturbs weights and keeps configs
    that improve Sharpe on training AND show positive Sharpe on test.
    """
    import json as _json
    from copy import deepcopy

    # Split data: train = first 2 years, test = last year
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index.tolist())
    sorted_dates = sorted(all_dates)
    total_days = len(sorted_dates)
    split_idx = int(total_days * 2 / 3)  # ~2 years
    split_date = sorted_dates[split_idx]

    train_data: Dict[str, pd.DataFrame] = {}
    test_data: Dict[str, pd.DataFrame] = {}
    for sym, df in data.items():
        train_data[sym] = df[df.index < split_date]
        test_data[sym] = df[df.index >= split_date]

    if verbose:
        print(f"\n  Config re-evolution: train < {split_date.strftime('%Y-%m-%d')}, "
              f"test >= {split_date.strftime('%Y-%m-%d')}")
        print(f"  Train days: ~{split_idx}, Test days: ~{total_days - split_idx}")

    # Load current configs
    configs = _load_player_configs()

    best_configs: Dict[str, Dict[str, Any]] = {}
    rng = np.random.RandomState(42)

    for pid in PLAYER_IDS:
        base_cfg = configs.get(pid, {})
        label = base_cfg.get("label", pid)
        weights = dict(base_cfg.get("weights", {}))
        entry_t = base_cfg.get("entry_threshold", 0.40)
        exit_t = abs(base_cfg.get("exit_threshold", -0.12))
        min_hold = base_cfg.get("min_hold_bars", 3)

        # Evaluate baseline
        base_sharpe, base_trades, base_pnl = _evaluate_config_on_data(
            train_data, weights, entry_t, exit_t, min_hold
        )
        test_sharpe, test_trades, test_pnl = _evaluate_config_on_data(
            test_data, weights, entry_t, exit_t, min_hold
        )

        if verbose:
            print(f"\n  {label} ({pid}) baseline:")
            print(f"    Train: Sharpe={base_sharpe:.3f} trades={base_trades} PnL={base_pnl:.0f}")
            print(f"    Test:  Sharpe={test_sharpe:.3f} trades={test_trades} PnL={test_pnl:.0f}")

        best_train_sharpe = base_sharpe
        best_test_sharpe = test_sharpe
        best_weights = dict(weights)
        best_entry_t = entry_t

        for iteration in range(n_iterations):
            # Perturb weights: randomly scale 3-5 indicators
            trial_weights = dict(best_weights)
            n_perturb = rng.randint(2, 6)
            keys_to_perturb = rng.choice(list(trial_weights.keys()), size=min(n_perturb, len(trial_weights)), replace=False)
            for k in keys_to_perturb:
                factor = rng.uniform(0.3, 1.7)
                trial_weights[k] = float(np.clip(trial_weights[k] * factor, 0.05, 1.0))

            # Occasionally try adding a new indicator
            if rng.random() < 0.3:
                new_key = rng.choice(_ALL_INDICATOR_KEYS)
                if new_key not in trial_weights:
                    trial_weights[new_key] = float(rng.uniform(0.1, 0.6))

            # Also perturb entry threshold slightly
            trial_entry = float(np.clip(best_entry_t + rng.uniform(-0.05, 0.05), 0.25, 0.50))

            # Evaluate on train
            t_sharpe, t_trades, t_pnl = _evaluate_config_on_data(
                train_data, trial_weights, trial_entry, exit_t, min_hold
            )

            # Must beat current best on training
            if t_sharpe <= best_train_sharpe:
                continue

            # Validate on test — must have positive Sharpe
            v_sharpe, v_trades, v_pnl = _evaluate_config_on_data(
                test_data, trial_weights, trial_entry, exit_t, min_hold
            )

            if v_sharpe > 0 and v_trades >= 3:
                best_train_sharpe = t_sharpe
                best_test_sharpe = v_sharpe
                best_weights = dict(trial_weights)
                best_entry_t = trial_entry
                if verbose:
                    print(f"    iter {iteration:2d}: Train Sharpe={t_sharpe:.3f} "
                          f"Test Sharpe={v_sharpe:.3f} trades={v_trades}")

        best_configs[pid] = {
            "label": label,
            "weights": best_weights,
            "entry_threshold": round(best_entry_t, 4),
            "exit_threshold": base_cfg.get("exit_threshold", -0.12),
            "min_hold_bars": min_hold,
        }

        if verbose:
            improved = best_train_sharpe > base_sharpe
            print(f"    Final: Train Sharpe={best_train_sharpe:.3f} "
                  f"Test Sharpe={best_test_sharpe:.3f} "
                  f"{'IMPROVED' if improved else 'unchanged'}")

    # Save to file
    cfg_path = Path(__file__).parent / "evolved_player_configs.json"
    import json as _json
    save_data = {
        "saved_at": datetime.now().isoformat(),
        "configs": best_configs,
        "evolution_method": "walk_forward_validation",
        "train_split": str(split_date.date()) if hasattr(split_date, 'date') else str(split_date),
    }
    cfg_path.write_text(_json.dumps(save_data, indent=2, default=str), encoding="utf-8")
    if verbose:
        print(f"\n  Saved re-evolved configs to {cfg_path}")

    return best_configs


# ═══════════════════════════════════════════════════════════════════════════
# Demo: SMA crossover strategy through walk-forward
# ═══════════════════════════════════════════════════════════════════════════

def _sma_crossover_strategy(
    train_data: Dict[str, pd.DataFrame],
    test_data: Dict[str, pd.DataFrame],
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Trade], Dict[str, Any]]:
    """Simple SMA crossover strategy for demo/sanity check.

    Training: optimize fast/slow SMA periods by testing a grid on training data.
    Testing: apply the best periods to generate trades on test data.
    """
    if params is None:
        params = {}

    # Default SMA periods
    fast = params.get("fast_period", 10)
    slow = params.get("slow_period", 30)

    # Training phase: simple grid search for best SMA combo
    best_pnl = -float("inf")
    best_fast, best_slow = fast, slow

    # Only optimize on the first (largest) symbol for speed
    train_symbol = None
    for sym, df in train_data.items():
        if len(df) >= 40:
            train_symbol = sym
            break

    if train_symbol is not None:
        train_df = train_data[train_symbol]
        close_col = "Close" if "Close" in train_df.columns else "close"

        for f in [5, 8, 10, 15]:
            for s in [20, 25, 30, 40]:
                if f >= s:
                    continue
                pnl = _simulate_sma(train_df[close_col], f, s)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_fast, best_slow = f, s

    fitted_params = {"fast_period": best_fast, "slow_period": best_slow}

    # Testing phase: generate trades on test data
    # Prepend a tail of training data so SMAs can warm up
    trades: List[Trade] = []
    player_cycle = 0

    for symbol, test_df in test_data.items():
        close_col = "Close" if "Close" in test_df.columns else "close"

        # Concatenate warm-up period from training data
        warmup_len = best_slow + 5
        if symbol in train_data:
            warmup = train_data[symbol].tail(warmup_len)
            combined = pd.concat([warmup, test_df])
            combined = combined[~combined.index.duplicated(keep="last")]
        else:
            combined = test_df

        if len(combined) < best_slow + 1:
            continue

        close = combined[close_col].astype(float)
        fast_sma = close.rolling(best_fast).mean()
        slow_sma = close.rolling(best_slow).mean()

        # Only generate trades on dates within the test window
        test_start = test_df.index[0]
        position = None  # None or (entry_date, entry_price, direction)

        for idx in range(best_slow, len(close)):
            date = combined.index[idx]
            price = close.iloc[idx]
            f_val = fast_sma.iloc[idx]
            s_val = slow_sma.iloc[idx]

            if pd.isna(f_val) or pd.isna(s_val):
                continue

            # Before test window: just track SMA state for warm-up
            if date < test_start:
                # Track position state through warm-up
                if position is None:
                    if f_val > s_val:
                        position = (str(date.date()), float(price), "LONG")
                    elif f_val < s_val:
                        position = (str(date.date()), float(price), "SHORT")
                else:
                    _, _, direction = position
                    if direction == "LONG" and f_val < s_val:
                        position = None
                    elif direction == "SHORT" and f_val > s_val:
                        position = None
                continue

            # In test window: generate trades
            if position is None:
                if f_val > s_val:
                    position = (str(date.date()), float(price), "LONG")
                elif f_val < s_val:
                    position = (str(date.date()), float(price), "SHORT")
            else:
                entry_date, entry_price, direction = position
                exit_signal = False
                if direction == "LONG" and f_val < s_val:
                    exit_signal = True
                elif direction == "SHORT" and f_val > s_val:
                    exit_signal = True

                if exit_signal or idx == len(close) - 1:
                    quantity = max(1, int(50000 / entry_price))
                    if direction == "LONG":
                        gross_pnl = (float(price) - entry_price) * quantity
                    else:
                        gross_pnl = (entry_price - float(price)) * quantity

                    pid = PLAYER_IDS[player_cycle % len(PLAYER_IDS)]
                    player_cycle += 1

                    trades.append(Trade(
                        symbol=symbol,
                        direction=direction,
                        entry_date=entry_date,
                        exit_date=str(date.date()),
                        entry_price=entry_price,
                        exit_price=float(price),
                        quantity=quantity,
                        gross_pnl=gross_pnl,
                        net_pnl=gross_pnl,
                        player_id=pid,
                        bars_held=0,
                        exit_reason="crossover" if exit_signal else "window_end",
                    ))
                    position = None

    return trades, fitted_params


def _simulate_sma(close: pd.Series, fast: int, slow: int) -> float:
    """Quick SMA backtest returning total P&L for parameter optimization."""
    fast_sma = close.rolling(fast).mean()
    slow_sma = close.rolling(slow).mean()

    pnl = 0.0
    position = None

    for i in range(slow, len(close)):
        f_val = fast_sma.iloc[i]
        s_val = slow_sma.iloc[i]
        price = close.iloc[i]

        if pd.isna(f_val) or pd.isna(s_val):
            continue

        if position is None:
            if f_val > s_val:
                position = (price, "LONG")
            elif f_val < s_val:
                position = (price, "SHORT")
        else:
            entry_price, direction = position
            if direction == "LONG" and f_val < s_val:
                pnl += price - entry_price
                position = None
            elif direction == "SHORT" and f_val > s_val:
                pnl += entry_price - price
                position = None

    return pnl


def run_demo():
    """Run both SMA baseline and full pipeline, then compare."""
    import yfinance as yf

    print("=" * 72)
    print("  Walk-Forward Backtest — Full Pipeline vs SMA Baseline")
    print("=" * 72)

    # Download 3 years of Nifty 50 index + top 10 stocks
    symbols = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    ]
    nifty_symbol = "^NSEI"

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - pd.DateOffset(years=3)).strftime("%Y-%m-%d")

    print(f"\n  Downloading data: {start_date} → {end_date}")
    print(f"  Symbols: {', '.join(symbols)}")

    # Download stock data
    data: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = yf.download(sym, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            if len(df) > 100:
                data[sym] = df
                print(f"    {sym}: {len(df)} days")
        except Exception as e:
            print(f"    {sym}: FAILED ({e})")

    if not data:
        print("\n  ERROR: No data downloaded. Check network / yfinance.")
        return

    # Download Nifty index for regime detection
    nifty_df = None
    try:
        nifty_raw = yf.download(nifty_symbol, start=start_date, end=end_date, progress=False)
        if isinstance(nifty_raw.columns, pd.MultiIndex):
            nifty_raw.columns = nifty_raw.columns.get_level_values(0)
        nifty_df = nifty_raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        print(f"    {nifty_symbol}: {len(nifty_df)} days")
    except Exception as e:
        print(f"    {nifty_symbol}: FAILED ({e})")

    # Set up regime detector if available
    rd = None
    if _HAS_REGIME and nifty_df is not None and len(nifty_df) >= 100:
        try:
            rd = RegimeDetector(model_path=Path("_demo_regime_model.joblib"))
            rd.fit(nifty_df)
            print(f"\n  Regime detector fitted on {len(nifty_df)} days of Nifty data")
        except Exception as e:
            print(f"\n  Regime detector failed: {e}")
            rd = None

    # ── Run 1: SMA Baseline ──────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  [1/2] Running SMA Crossover Baseline...")
    print("=" * 72)

    engine_sma = WalkForwardEngine(
        training_window=120,
        testing_window=20,
        step_size=20,
        initial_capital=500_000,
        regime_detector=rd,
        nifty_data=nifty_df,
    )
    sma_results = engine_sma.run(data, _sma_crossover_strategy, verbose=True)
    sma_results.generate_report()

    # ── Re-evolve configs with walk-forward validation ──────────────
    print("\n" + "=" * 72)
    print("  Re-evolving player configs (train 2yr / test 1yr)...")
    print("=" * 72)
    _reevolve_configs(data, n_iterations=40, verbose=True)

    # ── Run 2: Full Pipeline ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  [2/2] Running Full 5-Player Pipeline...")
    print("=" * 72)
    print("  Components: Indicators + 5 RegimeAwarePlayers + SignalCombiner")
    print("              + Confidence(0.7) + FlipThreshold(0.3) + MinHold(3)")
    print("              + VolTargeter + BayesianWeights + Bear(0.2x/0.85)")

    engine_full = WalkForwardEngine(
        training_window=120,
        testing_window=20,
        step_size=20,
        initial_capital=500_000,
        regime_detector=rd,
        nifty_data=nifty_df,
    )
    full_results = engine_full.run(data, _full_pipeline_strategy, verbose=True)
    full_results.generate_report()
    full_results.save_visualizations("backtest_output")

    # ── Comparison ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  COMPARISON: Full Pipeline vs SMA Baseline")
    print("=" * 72)

    def _extract_metrics(agg):
        all_trades = []
        for r in agg.windows:
            all_trades.extend(r.trades)
        total_gross = sum(t.gross_pnl for t in all_trades)
        total_net = sum(t.net_pnl for t in all_trades)
        total_costs = sum(t.transaction_cost for t in all_trades)
        n_trades = len(all_trades)
        wins = sum(1 for t in all_trades if t.net_pnl > 0)
        win_rate = wins / n_trades if n_trades > 0 else 0
        # Cost drag: costs as % of gross. If gross is negative, express as ratio to |gross|
        cost_drag = (total_costs / max(abs(total_gross), 1.0) * 100)
        avg_bars = np.mean([t.bars_held for t in all_trades]) if all_trades else 0
        return {
            "trades": n_trades,
            "gross_pnl": total_gross,
            "net_pnl": total_net,
            "costs": total_costs,
            "win_rate": win_rate,
            "cost_drag": cost_drag,
            "sharpe": agg.overall_sharpe,
            "sortino": agg.overall_sortino,
            "max_dd": agg.overall_max_drawdown,
            "avg_bars": avg_bars,
        }

    sma_m = _extract_metrics(sma_results)
    full_m = _extract_metrics(full_results)

    # Per-player breakdown for full pipeline
    player_stats: Dict[str, Dict[str, Any]] = {pid: {"trades": 0, "gross": 0.0, "net": 0.0, "wins": 0} for pid in PLAYER_IDS}
    for r in full_results.windows:
        for t in r.trades:
            if t.player_id in player_stats:
                player_stats[t.player_id]["trades"] += 1
                player_stats[t.player_id]["gross"] += t.gross_pnl
                player_stats[t.player_id]["net"] += t.net_pnl
                if t.net_pnl > 0:
                    player_stats[t.player_id]["wins"] += 1

    fmt = "  {:<25s}  {:>15s}  {:>15s}  {:>10s}"
    print(fmt.format("Metric", "SMA Baseline", "Full Pipeline", "Delta"))
    print("  " + "-" * 70)

    rows = [
        ("Total Trades", f"{sma_m['trades']}", f"{full_m['trades']}", f"{full_m['trades'] - sma_m['trades']:+d}"),
        ("Gross P&L", f"Rs {sma_m['gross_pnl']:,.0f}", f"Rs {full_m['gross_pnl']:,.0f}", f"Rs {full_m['gross_pnl']-sma_m['gross_pnl']:+,.0f}"),
        ("Net P&L", f"Rs {sma_m['net_pnl']:,.0f}", f"Rs {full_m['net_pnl']:,.0f}", f"Rs {full_m['net_pnl']-sma_m['net_pnl']:+,.0f}"),
        ("Transaction Costs", f"Rs {sma_m['costs']:,.0f}", f"Rs {full_m['costs']:,.0f}", f"Rs {full_m['costs']-sma_m['costs']:+,.0f}"),
        ("Cost Drag", f"{sma_m['cost_drag']:.1f}%", f"{full_m['cost_drag']:.1f}%", f"{full_m['cost_drag']-sma_m['cost_drag']:+.1f}%"),
        ("Win Rate", f"{sma_m['win_rate']:.1%}", f"{full_m['win_rate']:.1%}", f"{full_m['win_rate']-sma_m['win_rate']:+.1%}"),
        ("Sharpe Ratio", f"{sma_m['sharpe']:.3f}", f"{full_m['sharpe']:.3f}", f"{full_m['sharpe']-sma_m['sharpe']:+.3f}"),
        ("Sortino Ratio", f"{sma_m['sortino']:.3f}", f"{full_m['sortino']:.3f}", f"{full_m['sortino']-sma_m['sortino']:+.3f}"),
        ("Max Drawdown", f"{sma_m['max_dd']:.1%}", f"{full_m['max_dd']:.1%}", f"{full_m['max_dd']-sma_m['max_dd']:+.1%}"),
        ("Avg Bars Held", f"{sma_m['avg_bars']:.1f}", f"{full_m['avg_bars']:.1f}", f"{full_m['avg_bars']-sma_m['avg_bars']:+.1f}"),
    ]

    for label, sma_val, full_val, delta in rows:
        print(fmt.format(label, sma_val, full_val, delta))

    # Per-player table
    print(f"\n  {'─' * 70}")
    print("  Per-Player Performance (Full Pipeline)")
    print(f"  {'─' * 70}")
    pfmt = "  {:<12s}  {:>8s}  {:>12s}  {:>12s}  {:>10s}"
    print(pfmt.format("Player", "Trades", "Gross P&L", "Net P&L", "Win Rate"))
    print("  " + "-" * 60)
    for pid in PLAYER_IDS:
        ps = player_stats[pid]
        wr = ps["wins"] / ps["trades"] if ps["trades"] > 0 else 0
        label = PLAYER_LABELS.get(pid, pid)
        print(pfmt.format(
            label, str(ps["trades"]),
            f"Rs {ps['gross']:,.0f}", f"Rs {ps['net']:,.0f}",
            f"{wr:.1%}"
        ))

    # Target check
    print(f"\n  {'─' * 70}")
    print("  Target Check (Full Pipeline)")
    print(f"  {'─' * 70}")
    checks = [
        ("Trades < 250", full_m["trades"] < 250, f"{full_m['trades']}"),
        ("Cost Drag < 25%", full_m["cost_drag"] < 25, f"{full_m['cost_drag']:.1f}%"),
        ("Sharpe > 0.5", full_m["sharpe"] > 0.5, f"{full_m['sharpe']:.3f}"),
    ]
    for label, passed, val in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label:20s}  actual={val}")

    # Clean up demo regime model
    demo_model = Path("_demo_regime_model.joblib")
    if demo_model.exists():
        demo_model.unlink()

    print(f"\n  Demo complete.")


if __name__ == "__main__":
    run_demo()
