"""
Comprehensive tests for walk_forward_backtest.py

Tests:
1. Window boundaries — no overlap, no gaps (training=60, testing=20, step=20)
2. Empty trade list — no division by zero in Sharpe/Sortino/profit_factor
3. BacktestResult with all losing trades — metrics compute correctly
4. AggregateResults with single window — no index errors
5. Deflated Sharpe Ratio matches Lopez de Prado formula
6. Equity curve consistency with daily returns (cumprod matches)
7. Transaction cost application — costs correctly applied to trades
"""

import math
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sp_stats

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from walk_forward_backtest import (
    Trade,
    BacktestResult,
    AggregateResults,
    WalkForwardEngine,
    _annualized_sharpe,
    _annualized_sortino,
    _max_drawdown_from_returns,
    _deflated_sharpe_ratio,
    TRADING_DAYS_PER_YEAR,
    RISK_FREE_RATE,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_ohlcv(n_days: int, start: str = "2022-01-03", base_price: float = 100.0,
                seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with n_days trading days."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start, periods=n_days)
    close = base_price + np.cumsum(rng.randn(n_days) * 0.5)
    close = np.maximum(close, 10.0)  # avoid negative prices
    df = pd.DataFrame({
        "Open": close - rng.uniform(0, 1, n_days),
        "High": close + rng.uniform(0, 2, n_days),
        "Low": close - rng.uniform(0, 2, n_days),
        "Close": close,
        "Volume": rng.randint(100_000, 1_000_000, n_days),
    }, index=dates)
    return df


def _make_trade(net_pnl: float, gross_pnl: float = None, cost: float = 0.0,
                symbol: str = "TEST.NS", player_id: str = "PLAYER_1",
                entry_date: str = "2022-06-01", exit_date: str = "2022-06-10",
                entry_price: float = 100.0, exit_price: float = 110.0,
                direction: str = "LONG", regime: str = "Sideways") -> Trade:
    """Create a Trade with sensible defaults."""
    if gross_pnl is None:
        gross_pnl = net_pnl + cost
    return Trade(
        symbol=symbol,
        direction=direction,
        entry_date=entry_date,
        exit_date=exit_date,
        entry_price=entry_price,
        exit_price=exit_price,
        quantity=10,
        gross_pnl=gross_pnl,
        transaction_cost=cost,
        net_pnl=net_pnl,
        player_id=player_id,
        regime=regime,
    )


def _noop_strategy(train_data, test_data, params=None):
    """Strategy that returns zero trades."""
    return [], params or {}


def _fixed_trades_strategy(trades_list):
    """Returns a strategy function that always returns the given trades."""
    def strategy(train_data, test_data, params=None):
        return trades_list[:], params or {}
    return strategy


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: Window boundaries — no overlap, no gaps
# ═══════════════════════════════════════════════════════════════════════════

class TestWindowBoundaries:
    """Verify that WalkForwardEngine with training=60, testing=20, step=20
    creates correct window boundaries with no overlap and no gaps."""

    def test_window_count(self):
        """Correct number of windows for known data length."""
        # 200 business days: first window starts at 0, trains 0..59, tests 60..79
        # step=20: next window trains 20..79, tests 80..99, etc.
        # Last valid window: train_start + 60 + 20 <= 200
        # So train_start can be 0, 20, 40, 60, 80, 100, 120 => test_end = 80,100,...,200
        # That's 7 windows
        data = {"SYM": _make_ohlcv(200)}
        engine = WalkForwardEngine(
            training_window=60, testing_window=20, step_size=20,
            cost_model=None, risk_manager=None,
        )
        dates = engine._build_date_index(data)
        windows = engine._generate_windows(dates)

        # With 200 days, train=60, test=20, step=20:
        # Window test_ends: 80, 100, 120, 140, 160, 180, 200 => 7 windows
        assert len(windows) == 7, f"Expected 7 windows, got {len(windows)}"

    def test_no_gaps_in_test_windows(self):
        """When step == testing_window, the OOS windows should tile the date
        range with no gaps between consecutive test periods."""
        data = {"SYM": _make_ohlcv(200)}
        engine = WalkForwardEngine(
            training_window=60, testing_window=20, step_size=20,
            cost_model=None, risk_manager=None,
        )
        dates = engine._build_date_index(data)
        windows = engine._generate_windows(dates)

        for i in range(len(windows) - 1):
            _, _, _, test_end_i = windows[i]
            _, _, test_start_next, _ = windows[i + 1]
            assert test_end_i == test_start_next, (
                f"Gap between window {i} test_end={test_end_i} and "
                f"window {i+1} test_start={test_start_next}"
            )

    def test_no_overlap_in_test_windows(self):
        """When step == testing_window, consecutive OOS windows must not
        share any date indices."""
        data = {"SYM": _make_ohlcv(200)}
        engine = WalkForwardEngine(
            training_window=60, testing_window=20, step_size=20,
            cost_model=None, risk_manager=None,
        )
        dates = engine._build_date_index(data)
        windows = engine._generate_windows(dates)

        for i in range(len(windows) - 1):
            _, _, test_start_i, test_end_i = windows[i]
            _, _, test_start_next, test_end_next = windows[i + 1]
            test_indices_i = set(range(test_start_i, test_end_i))
            test_indices_next = set(range(test_start_next, test_end_next))
            overlap = test_indices_i & test_indices_next
            assert len(overlap) == 0, (
                f"Window {i} and {i+1} OOS overlap at indices {overlap}"
            )

    def test_train_end_equals_test_start(self):
        """Training end index must equal testing start index (no gap between
        training and testing within a single window)."""
        data = {"SYM": _make_ohlcv(200)}
        engine = WalkForwardEngine(
            training_window=60, testing_window=20, step_size=20,
            cost_model=None, risk_manager=None,
        )
        dates = engine._build_date_index(data)
        windows = engine._generate_windows(dates)

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            assert train_end == test_start, (
                f"Window {i}: train_end={train_end} != test_start={test_start}"
            )

    def test_training_window_length(self):
        """Each training window has exactly `training_window` days."""
        data = {"SYM": _make_ohlcv(200)}
        engine = WalkForwardEngine(
            training_window=60, testing_window=20, step_size=20,
            cost_model=None, risk_manager=None,
        )
        dates = engine._build_date_index(data)
        windows = engine._generate_windows(dates)

        for i, (train_start, train_end, _, _) in enumerate(windows):
            length = train_end - train_start
            assert length == 60, (
                f"Window {i}: training length={length}, expected 60"
            )

    def test_testing_window_length(self):
        """Each testing window has exactly `testing_window` days (except possibly
        the last window which may be shorter)."""
        data = {"SYM": _make_ohlcv(200)}
        engine = WalkForwardEngine(
            training_window=60, testing_window=20, step_size=20,
            cost_model=None, risk_manager=None,
        )
        dates = engine._build_date_index(data)
        windows = engine._generate_windows(dates)

        for i, (_, _, test_start, test_end) in enumerate(windows):
            length = test_end - test_start
            assert 5 <= length <= 20, (
                f"Window {i}: test length={length}, expected 5..20"
            )
            if i < len(windows) - 1:
                assert length == 20, (
                    f"Window {i} (non-last): test length={length}, expected 20"
                )

    def test_step_moves_by_step_size(self):
        """Each successive window's train_start advances by step_size."""
        data = {"SYM": _make_ohlcv(200)}
        engine = WalkForwardEngine(
            training_window=60, testing_window=20, step_size=20,
            cost_model=None, risk_manager=None,
        )
        dates = engine._build_date_index(data)
        windows = engine._generate_windows(dates)

        for i in range(len(windows) - 1):
            diff = windows[i + 1][0] - windows[i][0]
            assert diff == 20, (
                f"Step from window {i} to {i+1}: {diff}, expected 20"
            )


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: Empty trade list — no division by zero
# ═══════════════════════════════════════════════════════════════════════════

class TestEmptyTradeList:
    """Verify that BacktestResult and AggregateResults handle zero trades
    without division by zero errors."""

    def test_backtest_result_empty_trades(self):
        """BacktestResult.compute_metrics() with empty trades list."""
        result = BacktestResult(
            window_index=0,
            window_start="2022-01-03",
            window_end="2022-01-28",
            train_start="2022-01-03",
            train_end="2022-01-28",
            trades=[],
            daily_returns=[0.001, -0.002, 0.0, 0.003, -0.001],
        )
        # Should not raise
        result.compute_metrics()

        assert result.gross_pnl == 0.0
        assert result.net_pnl == 0.0
        assert result.total_costs == 0.0
        assert result.winning_trades == 0
        assert result.losing_trades == 0
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0
        assert result.avg_win == 0.0
        assert result.avg_loss == 0.0

    def test_backtest_result_empty_trades_no_daily_returns(self):
        """BacktestResult with no trades AND no daily returns."""
        result = BacktestResult(
            window_index=0,
            window_start="2022-01-03",
            window_end="2022-01-28",
            train_start="2022-01-03",
            train_end="2022-01-28",
            trades=[],
            daily_returns=[],
        )
        result.compute_metrics()

        assert result.sharpe == 0.0
        assert result.sortino == 0.0
        assert result.max_drawdown == 0.0
        assert result.calmar == 0.0

    def test_aggregate_empty_trades(self):
        """AggregateResults with windows that have zero trades."""
        w1 = BacktestResult(
            window_index=0,
            window_start="2022-01-03", window_end="2022-01-28",
            train_start="2022-01-03", train_end="2022-01-28",
            trades=[], daily_returns=[0.0] * 20,
        )
        w1.compute_metrics()

        w2 = BacktestResult(
            window_index=1,
            window_start="2022-01-31", window_end="2022-02-25",
            train_start="2022-01-03", train_end="2022-01-28",
            trades=[], daily_returns=[0.0] * 20,
        )
        w2.compute_metrics()

        agg = AggregateResults(windows=[w1, w2])
        agg.compute()

        assert agg.num_trades == 0
        assert agg.win_rate == 0.0
        assert agg.profit_factor == 0.0
        assert agg.total_net_pnl == 0.0

    def test_engine_run_empty_strategy(self):
        """WalkForwardEngine.run() with a strategy that returns no trades."""
        data = {"SYM": _make_ohlcv(200)}
        engine = WalkForwardEngine(
            training_window=60, testing_window=20, step_size=20,
            cost_model=None, risk_manager=None,
        )
        # Should not raise
        results = engine.run(data, _noop_strategy, verbose=False)

        assert results.num_trades == 0
        assert results.win_rate == 0.0
        assert results.profit_factor == 0.0
        assert math.isfinite(results.overall_sharpe)
        assert math.isfinite(results.overall_sortino)

    def test_sharpe_helper_empty(self):
        """_annualized_sharpe with empty and single-element arrays."""
        assert _annualized_sharpe(np.array([])) == 0.0
        assert _annualized_sharpe(np.array([0.01])) == 0.0

    def test_sortino_helper_empty(self):
        """_annualized_sortino with empty and single-element arrays."""
        assert _annualized_sortino(np.array([])) == 0.0
        assert _annualized_sortino(np.array([0.01])) == 0.0

    def test_max_drawdown_empty(self):
        """_max_drawdown_from_returns with empty array."""
        assert _max_drawdown_from_returns(np.array([])) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: All losing trades — metrics compute correctly
# ═══════════════════════════════════════════════════════════════════════════

class TestAllLosingTrades:
    """BacktestResult with all losing trades should produce correct metrics."""

    def test_all_losers(self):
        """All trades are losers: win_rate=0, profit_factor=0."""
        trades = [
            _make_trade(net_pnl=-100.0, gross_pnl=-90.0, cost=10.0,
                        exit_date="2022-06-02"),
            _make_trade(net_pnl=-200.0, gross_pnl=-180.0, cost=20.0,
                        exit_date="2022-06-05"),
            _make_trade(net_pnl=-50.0, gross_pnl=-40.0, cost=10.0,
                        exit_date="2022-06-08"),
        ]
        result = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-10",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=trades,
            daily_returns=[-0.002, -0.004, -0.001, -0.003, -0.002],
        )
        result.compute_metrics()

        assert result.winning_trades == 0
        assert result.losing_trades == 3
        assert result.win_rate == 0.0
        assert result.profit_factor == 0.0  # no winners => 0
        assert result.avg_win == 0.0
        assert result.avg_loss < 0  # should be negative (mean of losses)
        assert result.net_pnl == pytest.approx(-350.0)
        assert result.gross_pnl == pytest.approx(-310.0)
        assert result.total_costs == pytest.approx(40.0)

    def test_all_losers_negative_sharpe(self):
        """With all losing returns, Sharpe should be negative."""
        rets = np.array([-0.01, -0.02, -0.005, -0.015, -0.008])
        sharpe = _annualized_sharpe(rets)
        assert sharpe < 0, f"Expected negative Sharpe, got {sharpe}"

    def test_all_losers_aggregate(self):
        """AggregateResults with all-losing windows."""
        trades = [_make_trade(net_pnl=-100.0, exit_date="2022-06-05")]
        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-10",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=trades,
            daily_returns=[-0.002] * 5,
        )
        w.compute_metrics()

        agg = AggregateResults(windows=[w])
        agg.compute()

        assert agg.win_rate == 0.0
        assert agg.profit_factor == 0.0
        assert agg.total_net_pnl < 0

    def test_profit_factor_no_losers(self):
        """All winning trades: profit_factor = gross_wins (not inf)."""
        trades = [
            _make_trade(net_pnl=100.0, exit_date="2022-06-02"),
            _make_trade(net_pnl=200.0, exit_date="2022-06-05"),
        ]
        result = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-10",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=trades,
            daily_returns=[0.002] * 5,
        )
        result.compute_metrics()

        assert result.profit_factor == 300.0  # sum of net_pnl for winners
        assert math.isfinite(result.profit_factor)


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: AggregateResults with single window — no index errors
# ═══════════════════════════════════════════════════════════════════════════

class TestSingleWindowAggregate:
    """AggregateResults with exactly one window should not produce index errors."""

    def test_single_window_with_trades(self):
        """Single window with trades: all aggregate metrics computed."""
        trades = [
            _make_trade(net_pnl=50.0, exit_date="2022-06-02", player_id="PLAYER_1"),
            _make_trade(net_pnl=-30.0, exit_date="2022-06-05", player_id="PLAYER_2"),
            _make_trade(net_pnl=80.0, exit_date="2022-06-08", player_id="PLAYER_3"),
        ]
        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-10",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=trades,
            daily_returns=[0.001, -0.001, 0.002, 0.0, 0.001,
                           -0.002, 0.003, 0.001, -0.001, 0.002],
        )
        w.compute_metrics()

        agg = AggregateResults(windows=[w])
        agg.compute()

        assert agg.num_trades == 3
        assert agg.total_net_pnl == pytest.approx(100.0)
        assert len(agg.equity_curve) == 10
        assert len(agg.drawdown_curve) == 10
        assert math.isfinite(agg.overall_sharpe)
        assert math.isfinite(agg.overall_sortino)
        assert math.isfinite(agg.overall_max_drawdown)
        # Strategy decay: needs >= 4 OOS sharpes, so should be False
        assert agg.strategy_decay_flag is False

    def test_single_window_empty_trades(self):
        """Single window with no trades: no index errors."""
        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-10",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=[],
            daily_returns=[0.0] * 10,
        )
        w.compute_metrics()

        agg = AggregateResults(windows=[w])
        agg.compute()

        assert agg.num_trades == 0
        assert agg.monthly_returns.empty or agg.monthly_returns.sum().sum() == 0
        assert len(agg.equity_curve) == 10

    def test_single_window_report(self):
        """Single window: generate_report() produces valid string."""
        trades = [_make_trade(net_pnl=50.0, exit_date="2022-06-05")]
        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-10",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=trades,
            daily_returns=[0.001] * 10,
        )
        w.compute_metrics()

        agg = AggregateResults(windows=[w])
        agg.compute()

        report = agg.generate_report()
        assert isinstance(report, str)
        assert "WALK-FORWARD BACKTEST REPORT" in report
        assert "Windows:" in report

    def test_single_window_yearly_performance(self):
        """Single window: yearly_performance has exactly one year."""
        trades = [_make_trade(net_pnl=50.0, exit_date="2022-06-05")]
        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-10",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=trades,
            daily_returns=[0.001] * 5,
        )
        w.compute_metrics()

        agg = AggregateResults(windows=[w])
        agg.compute()

        assert 2022 in agg.yearly_performance
        assert agg.yearly_performance[2022]["net_pnl"] == 50.0
        assert agg.yearly_performance[2022]["num_trades"] == 1


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Deflated Sharpe Ratio matches Lopez de Prado formula
# ═══════════════════════════════════════════════════════════════════════════

class TestDeflatedSharpeRatio:
    """Verify the DSR calculation matches the Lopez de Prado (2014) formula."""

    def test_dsr_manual_calculation(self):
        """Compute DSR step-by-step and verify the function matches."""
        np.random.seed(42)
        n_obs = 500
        returns = np.random.normal(0.0003, 0.01, n_obs)

        observed_sharpe = _annualized_sharpe(returns)
        n_trials = 10

        # Manual computation following Lopez de Prado
        skew = float(sp_stats.skew(returns))
        kurt = float(sp_stats.kurtosis(returns, fisher=True))

        # E[max(SR)] under null
        z = math.sqrt(2.0 * math.log(n_trials))
        euler_gamma = 0.5772156649
        e_max_sr = z - (euler_gamma + math.log(math.log(n_trials))) / (2.0 * z)

        # De-annualize observed Sharpe to daily
        sr_daily = observed_sharpe / math.sqrt(TRADING_DAYS_PER_YEAR)

        # SE(SR) with non-normality adjustment
        se_sr_sq = (1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily ** 2) / (n_obs - 1)
        se_sr = math.sqrt(max(se_sr_sq, 1e-10))

        # DSR = Phi((SR_daily - E[max(SR)]) / SE(SR))
        z_score = (sr_daily - e_max_sr) / se_sr
        expected_dsr = float(sp_stats.norm.cdf(z_score))

        # Compare
        actual_dsr = _deflated_sharpe_ratio(observed_sharpe, n_trials, n_obs, returns)
        assert abs(actual_dsr - expected_dsr) < 1e-10, (
            f"DSR mismatch: actual={actual_dsr}, expected={expected_dsr}"
        )

    def test_dsr_single_trial(self):
        """With n_trials=1, E[max(SR)]=0, so DSR depends only on SR and SE."""
        np.random.seed(123)
        returns = np.random.normal(0.001, 0.02, 252)
        observed_sharpe = _annualized_sharpe(returns)

        dsr = _deflated_sharpe_ratio(observed_sharpe, n_trials=1, n_obs=252, returns=returns)

        # With 1 trial, e_max_sr = 0
        # DSR = Phi(sr_daily / se_sr)
        skew = float(sp_stats.skew(returns))
        kurt = float(sp_stats.kurtosis(returns, fisher=True))
        sr_daily = observed_sharpe / math.sqrt(TRADING_DAYS_PER_YEAR)
        se_sr_sq = (1.0 - skew * sr_daily + (kurt - 1.0) / 4.0 * sr_daily ** 2) / 251
        se_sr = math.sqrt(max(se_sr_sq, 1e-10))
        expected = float(sp_stats.norm.cdf(sr_daily / se_sr))
        assert abs(dsr - expected) < 1e-10

    def test_dsr_more_trials_lowers_dsr(self):
        """More trials should generally decrease DSR (higher bar to pass)."""
        np.random.seed(42)
        returns = np.random.normal(0.0005, 0.01, 500)
        observed_sharpe = _annualized_sharpe(returns)

        dsr_5 = _deflated_sharpe_ratio(observed_sharpe, n_trials=5, n_obs=500, returns=returns)
        dsr_50 = _deflated_sharpe_ratio(observed_sharpe, n_trials=50, n_obs=500, returns=returns)
        dsr_500 = _deflated_sharpe_ratio(observed_sharpe, n_trials=500, n_obs=500, returns=returns)

        # More trials => higher E[max(SR)] => lower DSR
        assert dsr_5 >= dsr_50, f"DSR should decrease with more trials: {dsr_5} vs {dsr_50}"
        assert dsr_50 >= dsr_500, f"DSR should decrease with more trials: {dsr_50} vs {dsr_500}"

    def test_dsr_edge_cases(self):
        """DSR with very few observations returns 0."""
        returns = np.array([0.01, 0.02])
        dsr = _deflated_sharpe_ratio(1.0, n_trials=5, n_obs=5, returns=returns)
        # With n_obs=5 < 10, function returns 0.0
        assert dsr == 0.0

    def test_dsr_zero_sharpe(self):
        """With observed_sharpe=0, the aggregate compute() should skip DSR."""
        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-10",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=[],
            daily_returns=[0.0] * 20,
        )
        w.compute_metrics()
        agg = AggregateResults(windows=[w])
        agg.compute()
        # overall_sharpe will be 0.0 (all-zero returns with rf > 0 might not be exactly 0)
        # But DSR should remain at its default 0.0 or be computed safely
        assert math.isfinite(agg.deflated_sharpe)


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Equity curve consistency with daily returns
# ═══════════════════════════════════════════════════════════════════════════

class TestEquityCurveConsistency:
    """Verify equity curve is monotonically consistent with daily returns,
    i.e., equity_curve = cumprod(1 + daily_returns)."""

    def test_equity_matches_cumprod(self):
        """Equity curve must match np.cumprod(1 + returns)."""
        daily_returns = [0.01, -0.005, 0.003, 0.008, -0.002,
                         0.005, -0.001, 0.004, -0.003, 0.002]

        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-14",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=[_make_trade(net_pnl=100.0, exit_date="2022-06-05")],
            daily_returns=daily_returns,
        )
        w.compute_metrics()

        agg = AggregateResults(windows=[w])
        agg.compute()

        expected = np.cumprod(1.0 + np.array(daily_returns))
        np.testing.assert_array_almost_equal(
            agg.equity_curve.values, expected, decimal=10,
            err_msg="Equity curve doesn't match cumprod(1 + returns)"
        )

    def test_equity_multi_window_cumprod(self):
        """Equity curve from multiple windows concatenated daily returns."""
        rets1 = [0.01, -0.005, 0.003, 0.008, -0.002]
        rets2 = [0.005, -0.001, 0.004, -0.003, 0.002]

        w1 = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-07",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=[_make_trade(net_pnl=50.0, exit_date="2022-06-03")],
            daily_returns=rets1,
        )
        w1.compute_metrics()

        w2 = BacktestResult(
            window_index=1,
            window_start="2022-06-08", window_end="2022-06-14",
            train_start="2022-03-21", train_end="2022-06-07",
            trades=[_make_trade(net_pnl=30.0, exit_date="2022-06-10")],
            daily_returns=rets2,
        )
        w2.compute_metrics()

        agg = AggregateResults(windows=[w1, w2])
        agg.compute()

        all_rets = rets1 + rets2
        expected = np.cumprod(1.0 + np.array(all_rets))
        np.testing.assert_array_almost_equal(
            agg.equity_curve.values, expected, decimal=10,
            err_msg="Multi-window equity curve doesn't match concatenated cumprod"
        )

    def test_drawdown_curve_consistency(self):
        """Drawdown curve = (equity - running_max) / running_max."""
        daily_returns = [0.01, 0.02, -0.05, -0.03, 0.04, 0.01]

        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-08",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=[_make_trade(net_pnl=10.0, exit_date="2022-06-05")],
            daily_returns=daily_returns,
        )
        w.compute_metrics()

        agg = AggregateResults(windows=[w])
        agg.compute()

        cumulative = np.cumprod(1.0 + np.array(daily_returns))
        running_max = np.maximum.accumulate(cumulative)
        expected_dd = (cumulative - running_max) / running_max

        np.testing.assert_array_almost_equal(
            agg.drawdown_curve.values, expected_dd, decimal=10,
            err_msg="Drawdown curve doesn't match (equity - running_max) / running_max"
        )

    def test_max_drawdown_matches_curve(self):
        """overall_max_drawdown should equal min of drawdown_curve."""
        daily_returns = [0.01, 0.02, -0.05, -0.03, 0.04, 0.01]

        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-08",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=[_make_trade(net_pnl=10.0, exit_date="2022-06-05")],
            daily_returns=daily_returns,
        )
        w.compute_metrics()

        agg = AggregateResults(windows=[w])
        agg.compute()

        assert agg.overall_max_drawdown == pytest.approx(
            float(agg.drawdown_curve.min()), abs=1e-10
        )

    def test_equity_starts_above_1_if_positive_first_return(self):
        """If first return is positive, first equity point > 1."""
        daily_returns = [0.05, -0.01, 0.02]
        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-03",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=[], daily_returns=daily_returns,
        )
        w.compute_metrics()
        agg = AggregateResults(windows=[w])
        agg.compute()

        assert agg.equity_curve.iloc[0] == pytest.approx(1.05)


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: Transaction cost application
# ═══════════════════════════════════════════════════════════════════════════

class TestTransactionCosts:
    """Verify transaction costs are correctly applied to trades."""

    def test_trade_post_init_calculates_net_pnl(self):
        """Trade.__post_init__ computes net_pnl = gross_pnl - cost when net_pnl=0."""
        t = Trade(
            symbol="TEST.NS", direction="LONG",
            entry_date="2022-06-01", exit_date="2022-06-10",
            entry_price=100.0, exit_price=110.0,
            quantity=10, gross_pnl=100.0,
            transaction_cost=15.0,
            # net_pnl left as default 0.0 => post_init should compute
        )
        assert t.net_pnl == 85.0

    def test_trade_post_init_preserves_explicit_net_pnl(self):
        """Trade.__post_init__ should NOT overwrite if net_pnl is explicitly set
        to a non-zero value."""
        t = Trade(
            symbol="TEST.NS", direction="LONG",
            entry_date="2022-06-01", exit_date="2022-06-10",
            entry_price=100.0, exit_price=110.0,
            quantity=10, gross_pnl=100.0,
            transaction_cost=15.0,
            net_pnl=90.0,  # explicitly set
        )
        assert t.net_pnl == 90.0

    def test_engine_apply_costs(self):
        """WalkForwardEngine._apply_costs modifies trade's transaction_cost and net_pnl."""
        # We test this even if TransactionCostModel isn't available
        # by mocking a simple cost model
        class SimpleCostModel:
            def calculate_cost(self, price, quantity, side, trade_type="intraday",
                               signal_type="neutral", **kwargs):
                # Simple 0.1% of notional
                return price * quantity * 0.001

        engine = WalkForwardEngine(
            cost_model=SimpleCostModel(),
            risk_manager=None,
        )

        t = Trade(
            symbol="TEST.NS", direction="LONG",
            entry_date="2022-06-01", exit_date="2022-06-10",
            entry_price=100.0, exit_price=110.0,
            quantity=10, gross_pnl=100.0,
            transaction_cost=0.0,
            net_pnl=100.0,
        )

        engine._apply_costs(t)

        # Entry cost: 100 * 10 * 0.001 = 1.0
        # Exit cost: 110 * 10 * 0.001 = 1.1
        expected_cost = 1.0 + 1.1
        assert t.transaction_cost == pytest.approx(expected_cost)
        assert t.net_pnl == pytest.approx(100.0 - expected_cost)

    def test_costs_reduce_net_pnl(self):
        """After cost application, net_pnl should be less than gross_pnl."""
        class SimpleCostModel:
            def calculate_cost(self, price, quantity, side, trade_type="intraday",
                               signal_type="neutral", **kwargs):
                return price * quantity * 0.001

        engine = WalkForwardEngine(
            cost_model=SimpleCostModel(),
            risk_manager=None,
        )

        t = _make_trade(net_pnl=500.0, gross_pnl=500.0, cost=0.0)
        engine._apply_costs(t)

        assert t.transaction_cost > 0
        assert t.net_pnl < t.gross_pnl

    def test_aggregate_cost_drag(self):
        """AggregateResults cost_drag_pct = total_costs / total_gross_pnl * 100."""
        trades = [
            _make_trade(net_pnl=90.0, gross_pnl=100.0, cost=10.0, exit_date="2022-06-02"),
            _make_trade(net_pnl=180.0, gross_pnl=200.0, cost=20.0, exit_date="2022-06-05"),
        ]
        w = BacktestResult(
            window_index=0,
            window_start="2022-06-01", window_end="2022-06-10",
            train_start="2022-03-01", train_end="2022-05-31",
            trades=trades,
            daily_returns=[0.001] * 5,
        )
        w.compute_metrics()

        agg = AggregateResults(windows=[w])
        agg.compute()

        assert agg.total_gross_pnl == pytest.approx(300.0)
        assert agg.total_costs == pytest.approx(30.0)
        assert agg.cost_drag_pct == pytest.approx(10.0)  # 30/300 * 100

    def test_no_cost_model_means_zero_costs(self):
        """Without a cost model, _apply_costs is a no-op."""
        engine = WalkForwardEngine(
            cost_model=None, risk_manager=None,
        )
        # WalkForwardEngine auto-creates a cost model if the module is available.
        # Force it to None to test the no-cost-model path.
        engine.cost_model = None

        t = _make_trade(net_pnl=100.0, gross_pnl=100.0, cost=0.0)
        engine._apply_costs(t)
        assert t.transaction_cost == 0.0
        assert t.net_pnl == 100.0

    def test_cost_with_short_direction(self):
        """Cost model called with correct sides for SHORT direction."""
        calls = []

        class TrackingCostModel:
            def calculate_cost(self, price, quantity, side, trade_type="intraday",
                               signal_type="neutral", **kwargs):
                calls.append(side)
                return 1.0

        engine = WalkForwardEngine(
            cost_model=TrackingCostModel(),
            risk_manager=None,
        )
        t = Trade(
            symbol="TEST.NS", direction="SHORT",
            entry_date="2022-06-01", exit_date="2022-06-10",
            entry_price=110.0, exit_price=100.0,
            quantity=10, gross_pnl=100.0,
            transaction_cost=0.0, net_pnl=100.0,
        )
        engine._apply_costs(t)

        # SHORT entry = SELL, SHORT exit = BUY
        assert calls == ["SELL", "BUY"]


# ═══════════════════════════════════════════════════════════════════════════
# Additional edge-case tests
# ═══════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Additional edge case tests."""

    def test_insufficient_data_raises(self):
        """Engine should raise ValueError if data is too short."""
        data = {"SYM": _make_ohlcv(50)}
        engine = WalkForwardEngine(
            training_window=60, testing_window=20, step_size=20,
            cost_model=None, risk_manager=None,
        )
        with pytest.raises(ValueError, match="Need at least"):
            engine.run(data, _noop_strategy, verbose=False)

    def test_sharpe_constant_returns(self):
        """Constant returns should give Sharpe of 0 (zero std)."""
        rets = np.full(100, 0.001)
        # std of excess returns is 0, so sharpe returns 0.0
        assert _annualized_sharpe(rets) == 0.0

    def test_sortino_all_positive_returns(self):
        """All positive excess returns: no downside deviation."""
        rets = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        excess = rets - daily_rf
        # All excess returns are positive
        assert all(e > 0 for e in excess)
        # Sortino should return a positive value (mean * sqrt(252)) since no downside
        sortino = _annualized_sortino(rets)
        assert sortino > 0

    def test_max_drawdown_no_drawdown(self):
        """Monotonically increasing returns have zero drawdown."""
        rets = np.array([0.01, 0.02, 0.01, 0.03, 0.01])
        dd = _max_drawdown_from_returns(rets)
        assert dd == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
