"""
Tests for portfolio_optimizer.py — 18 tests covering HRP, VolTargeter, CVaR, and full pipeline.

Run: python tests/test_portfolio_optimizer.py
"""

from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from portfolio_optimizer import (
    PLAYER_IDS,
    PLAYER_LABELS,
    CVaRRiskManager,
    HRPAllocator,
    OptimizationResult,
    PlayerReturnTracker,
    PortfolioOptimizer,
    VolatilityTargeter,
)

PASS = 0
FAIL = 0


def run_test(name: str, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  [PASS] {name}")
        PASS += 1
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        traceback.print_exc()
        FAIL += 1


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_returns_df(data: np.ndarray) -> pd.DataFrame:
    """Wrap a (n_days, 5) array into a DataFrame with PLAYER_IDS columns."""
    return pd.DataFrame(data, columns=PLAYER_IDS)


def _force_rebalance(hrp: HRPAllocator):
    """Force HRP to actually recompute on next call."""
    hrp._calls_since_rebalance = HRPAllocator.REBALANCE_INTERVAL


# ═══════════════════════════════════════════════════════════════════════════
# HRP Tests (1-5)
# ═══════════════════════════════════════════════════════════════════════════

def test_hrp_identical_streams():
    """5 identical return streams → equal weights (no player should dominate)."""
    np.random.seed(100)
    base = np.random.normal(0.001, 0.01, 60)
    data = np.column_stack([base] * 5)
    # Add tiny noise to avoid singular covariance
    data += np.random.normal(0, 1e-6, data.shape)
    df = _make_returns_df(data)

    hrp = HRPAllocator()
    _force_rebalance(hrp)
    weights = hrp.get_allocation(df)

    # All weights should be close to 20%
    for pid in PLAYER_IDS:
        assert abs(weights[pid] - 0.20) < 0.10, (
            f"{pid} weight {weights[pid]:.3f} deviates too much from 0.20"
        )
    # No single player should have more than 30%
    assert max(weights.values()) < 0.30, (
        f"Max weight {max(weights.values()):.3f} too high for identical streams"
    )


def test_hrp_correlated_pair():
    """1 player perfectly correlated with another → combined ≈ one independent's weight."""
    np.random.seed(101)
    n = 60
    independent = np.random.normal(0.001, 0.01, (n, 3))  # P2, P3, P4
    p1 = np.random.normal(0.001, 0.01, n)
    p5 = p1 + np.random.normal(0, 1e-5, n)  # P5 ≈ P1 (near-perfect correlation)

    data = np.column_stack([p1, independent, p5])
    df = _make_returns_df(data)

    hrp = HRPAllocator()
    _force_rebalance(hrp)
    weights = hrp.get_allocation(df)

    # P1 + P5 combined should be roughly comparable to any single independent player
    combined_correlated = weights["PLAYER_1"] + weights["PLAYER_5"]
    avg_independent = (weights["PLAYER_2"] + weights["PLAYER_3"] + weights["PLAYER_4"]) / 3

    # The correlated pair combined should get less total than 2x an independent player
    assert combined_correlated < 2 * avg_independent + 0.05, (
        f"Correlated pair {combined_correlated:.3f} not penalized vs "
        f"avg independent {avg_independent:.3f}"
    )


def test_hrp_weights_sum_and_bounds():
    """Weights sum to 1.0 and respect min 5% / max 40% constraints."""
    np.random.seed(102)
    data = np.random.normal(0.001, 0.01, (60, 5))
    # Make one player extremely volatile to test max constraint
    data[:, 0] *= 10
    df = _make_returns_df(data)

    hrp = HRPAllocator()
    _force_rebalance(hrp)
    weights = hrp.get_allocation(df)

    # Sum to 1.0
    total = sum(weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    # Min/max bounds
    for pid in PLAYER_IDS:
        w = weights[pid]
        assert w >= HRPAllocator.MIN_WEIGHT - 0.001, (
            f"{pid} weight {w:.4f} below min {HRPAllocator.MIN_WEIGHT}"
        )
        assert w <= HRPAllocator.MAX_WEIGHT + 0.001, (
            f"{pid} weight {w:.4f} above max {HRPAllocator.MAX_WEIGHT}"
        )


def test_hrp_ledoit_wolf_psd():
    """Ledoit-Wolf shrinkage produces a valid positive semi-definite covariance matrix."""
    from sklearn.covariance import LedoitWolf

    np.random.seed(103)
    data = np.random.normal(0, 0.01, (60, 5))
    lw = LedoitWolf().fit(data)
    cov = lw.covariance_

    # Check symmetric
    assert np.allclose(cov, cov.T), "Covariance not symmetric"

    # Check positive semi-definite (all eigenvalues >= 0)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10), (
        f"Negative eigenvalue: {eigenvalues.min()}"
    )


def test_hrp_insufficient_data():
    """< 20 days of data → graceful fallback to equal weights."""
    np.random.seed(104)
    data = np.random.normal(0, 0.01, (10, 5))  # only 10 days
    df = _make_returns_df(data)

    hrp = HRPAllocator()
    _force_rebalance(hrp)
    weights = hrp.get_allocation(df)

    # Should return equal weights
    for pid in PLAYER_IDS:
        assert abs(weights[pid] - 0.20) < 0.001, (
            f"{pid}={weights[pid]:.4f}, expected 0.20 for insufficient data"
        )


# ═══════════════════════════════════════════════════════════════════════════
# VolatilityTargeter Tests (6-9)
# ═══════════════════════════════════════════════════════════════════════════

def test_vol_target_high_realized():
    """Realized vol = 30% with target 15% → scale ≈ 0.5."""
    np.random.seed(200)
    # Daily vol to produce ~30% annualized: 30% / sqrt(252) ≈ 1.89%
    returns = np.random.normal(0, 0.019, 60)

    vt = VolatilityTargeter(target_vol=0.15)
    scale, rvol, tvol = vt.get_scale_factor(returns, regime=None)

    assert tvol == 0.15, f"Target vol should be 0.15, got {tvol}"
    assert scale < 1.0, f"Scale should be < 1.0 for high vol, got {scale:.3f}"
    assert 0.3 < scale < 0.8, f"Scale should be ~0.5, got {scale:.3f}"


def test_vol_target_low_realized():
    """Realized vol = 7.5% with target 15% → scale ≈ 2.0 (clamped to max_leverage)."""
    np.random.seed(201)
    # Daily vol to produce ~7.5% annualized: 7.5% / sqrt(252) ≈ 0.47%
    returns = np.random.normal(0, 0.0047, 60)

    vt = VolatilityTargeter(target_vol=0.15, max_leverage=2.0)
    scale, rvol, tvol = vt.get_scale_factor(returns, regime=None)

    assert scale >= 1.5, f"Scale should be >= 1.5 for low vol, got {scale:.3f}"
    assert scale <= 2.0, f"Scale should be clamped to max_leverage 2.0, got {scale:.3f}"


def test_vol_target_clamping():
    """Scale factor clamped between min_leverage and max_leverage."""
    np.random.seed(202)
    vt = VolatilityTargeter(target_vol=0.15, min_leverage=0.2, max_leverage=2.0)

    # Extremely high vol → should clamp to min_leverage
    extreme_vol = np.random.normal(0, 0.10, 60)
    scale, _, _ = vt.get_scale_factor(extreme_vol)
    assert scale >= 0.2, f"Scale {scale:.3f} below min_leverage 0.2"
    assert scale <= 2.0, f"Scale {scale:.3f} above max_leverage 2.0"

    # Near-zero vol → should clamp to max_leverage
    tiny_vol = np.random.normal(0, 0.0001, 60)
    scale2, _, _ = vt.get_scale_factor(tiny_vol)
    assert scale2 <= 2.0, f"Scale {scale2:.3f} above max_leverage 2.0"
    assert scale2 >= 0.2, f"Scale {scale2:.3f} below min_leverage 0.2"


def test_vol_target_bear_regime():
    """Bear regime should reduce target_vol to 0.10."""
    np.random.seed(203)
    returns = np.random.normal(0, 0.01, 60)

    vt = VolatilityTargeter(target_vol=0.15)
    scale, rvol, tvol = vt.get_scale_factor(returns, regime="Bear")

    assert tvol == 0.10, f"Bear regime target should be 0.10, got {tvol}"


# ═══════════════════════════════════════════════════════════════════════════
# CVaR Tests (10-14)
# ═══════════════════════════════════════════════════════════════════════════

def test_cvar_massive_loss():
    """Portfolio with one massive losing day → CVaR should capture it."""
    np.random.seed(300)
    returns = np.random.normal(0.001, 0.005, 100)
    returns[50] = -0.20  # massive loss day

    cvar_mgr = CVaRRiskManager()
    cvar = cvar_mgr.compute_cvar(returns, confidence=0.95, lookback=120)

    # The massive loss is in the worst 5%, CVaR should be elevated
    assert cvar > 0.01, f"CVaR should capture the massive loss, got {cvar:.4f}"
    # The -20% loss should pull CVaR way up
    assert cvar > 0.03, f"CVaR should be > 3% with a -20% day, got {cvar:.4f}"


def test_cvar_all_positive():
    """All positive returns → CVaR near zero."""
    returns = np.abs(np.random.normal(0.005, 0.003, 100))  # all positive

    cvar_mgr = CVaRRiskManager()
    cvar = cvar_mgr.compute_cvar(returns, confidence=0.95, lookback=120)

    # CVaR for all-positive returns should be 0 (clamped)
    assert cvar == 0.0, f"CVaR for all-positive returns should be 0.0, got {cvar:.6f}"


def test_cvar_marginal_identifies_riskiest():
    """Marginal CVaR identifies the riskiest player correctly."""
    np.random.seed(302)
    n = 120
    # P1-P4: normal returns
    data = np.random.normal(0.001, 0.005, (n, 5))
    # P3 (index 2): inject fat-tail losses every 10 days
    for i in range(0, n, 10):
        data[i, 2] = -0.08

    df = _make_returns_df(data)
    port_returns = df.mean(axis=1).values

    cvar_mgr = CVaRRiskManager()
    marginal = cvar_mgr.compute_marginal_cvar(port_returns, df)

    # P3 (PLAYER_3) should have the highest marginal CVaR
    riskiest = max(marginal, key=marginal.get)
    assert riskiest == "PLAYER_3", (
        f"Expected PLAYER_3 as riskiest, got {riskiest} "
        f"(marginals: {marginal})"
    )


def test_cvar_breach_adjusts_weights():
    """CVaR breach → reduces riskiest player and redistributes."""
    np.random.seed(303)
    n = 120
    data = np.random.normal(0.001, 0.005, (n, 5))
    # Make P1 the riskiest
    for i in range(0, n, 8):
        data[i, 0] = -0.06

    df = _make_returns_df(data)
    port_returns = df.mean(axis=1).values

    cvar_mgr = CVaRRiskManager()
    input_weights = {pid: 0.20 for pid in PLAYER_IDS}

    # Use a very tight limit to force a breach
    adj_w, cvar, breached, marginal, log = cvar_mgr.adjust_weights_for_cvar(
        input_weights, port_returns, df, max_cvar=0.001
    )

    assert breached, f"CVaR should be breached with tight limit, cvar={cvar:.4f}"
    assert len(log) >= 2, f"Adjustment log should have entries, got {log}"

    # Riskiest player should have weight < 0.20
    riskiest = max(marginal, key=marginal.get)
    assert adj_w[riskiest] < 0.20, (
        f"Riskiest player {riskiest} should have reduced weight, "
        f"got {adj_w[riskiest]:.4f}"
    )

    # Weights must still sum to 1.0
    total = sum(adj_w.values())
    assert abs(total - 1.0) < 0.01, f"Adjusted weights sum to {total}, expected 1.0"


def test_cvar_short_data():
    """< 10 data points → no crash, CVaR returns 0.0."""
    returns = np.array([0.01, -0.02, 0.005])  # only 3 points

    cvar_mgr = CVaRRiskManager()
    cvar = cvar_mgr.compute_cvar(returns, lookback=120)
    assert cvar == 0.0, f"CVaR with < 10 points should be 0.0, got {cvar}"

    # marginal CVaR with short data should also not crash
    df = _make_returns_df(np.random.normal(0, 0.01, (3, 5)))
    marginal = cvar_mgr.compute_marginal_cvar(returns, df)
    assert len(marginal) == 5, f"Should return 5 marginal values, got {len(marginal)}"


# ═══════════════════════════════════════════════════════════════════════════
# PortfolioOptimizer Tests (15-18)
# ═══════════════════════════════════════════════════════════════════════════

def test_full_pipeline():
    """Full pipeline: returns in → OptimizationResult out, all steps execute."""
    np.random.seed(400)
    n = 60
    data = np.random.normal(0.001, 0.01, (n, 5))
    df = _make_returns_df(data)
    port_returns = df.mean(axis=1).values

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    tracker = PlayerReturnTracker(path=Path(tmp.name))

    optimizer = PortfolioOptimizer(return_tracker=tracker)
    _force_rebalance(optimizer.hrp)

    regime_w = {pid: 0.20 for pid in PLAYER_IDS}
    bayesian_w = {pid: 0.20 for pid in PLAYER_IDS}

    result = optimizer.optimize(
        portfolio_returns=port_returns,
        player_returns_df=df,
        regime="Sideways",
        regime_weights=regime_w,
        bayesian_weights=bayesian_w,
    )

    # Verify it's an OptimizationResult
    assert isinstance(result, OptimizationResult)
    assert len(result.final_weights) == 5
    assert abs(sum(result.final_weights.values()) - 1.0) < 0.01
    assert result.scale_factor > 0
    assert result.regime == "Sideways"
    assert result.timestamp
    assert len(result.marginal_cvar) == 5
    assert isinstance(result.cvar, float)

    # summary() should not crash
    summary = result.summary()
    assert "PORTFOLIO OPTIMIZATION RESULT" in summary

    Path(tmp.name).unlink(missing_ok=True)


def test_audit_trail():
    """Audit trail captures before/after at each pipeline step."""
    np.random.seed(401)
    n = 120
    data = np.random.normal(0.001, 0.01, (n, 5))
    # Make one player risky to trigger CVaR
    for i in range(0, n, 8):
        data[i, 0] = -0.05
    df = _make_returns_df(data)
    port_returns = df.mean(axis=1).values

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    tracker = PlayerReturnTracker(path=Path(tmp.name))

    optimizer = PortfolioOptimizer(return_tracker=tracker)
    _force_rebalance(optimizer.hrp)

    result = optimizer.optimize(
        portfolio_returns=port_returns,
        player_returns_df=df,
        regime="Bear",
        max_cvar=0.001,  # tight limit to force CVaR breach
    )

    log = result.adjustments_log
    assert len(log) >= 2, f"Audit trail should have >= 2 entries, got {len(log)}"

    # Should contain blended weights step
    assert any("Blended weights" in entry for entry in log), (
        "Audit trail missing 'Blended weights' entry"
    )
    # Should contain vol targeting step
    assert any("Vol targeting" in entry for entry in log), (
        "Audit trail missing 'Vol targeting' entry"
    )
    # Should contain CVaR info (breached with tight limit)
    assert any("CVaR" in entry for entry in log), (
        "Audit trail missing CVaR entry"
    )

    Path(tmp.name).unlink(missing_ok=True)


def test_blended_weights_sum():
    """HRP + Bayesian + regime weights blend to sum to 1.0."""
    np.random.seed(402)
    n = 60
    data = np.random.normal(0.001, 0.01, (n, 5))
    df = _make_returns_df(data)

    optimizer = PortfolioOptimizer()
    _force_rebalance(optimizer.hrp)

    regime_w = {"PLAYER_1": 0.30, "PLAYER_2": 0.10, "PLAYER_3": 0.20,
                "PLAYER_4": 0.15, "PLAYER_5": 0.25}
    bayesian_w = {"PLAYER_1": 0.10, "PLAYER_2": 0.30, "PLAYER_3": 0.20,
                  "PLAYER_4": 0.25, "PLAYER_5": 0.15}

    blended, hrp_w = optimizer.compute_blended_weights(df, regime_w, bayesian_w)

    total = sum(blended.values())
    assert abs(total - 1.0) < 0.01, f"Blended weights sum to {total}, expected 1.0"

    # Each source should also sum to 1.0
    assert abs(sum(hrp_w.values()) - 1.0) < 0.01
    assert abs(sum(regime_w.values()) - 1.0) < 0.01
    assert abs(sum(bayesian_w.values()) - 1.0) < 0.01


def test_zero_signals_no_crash():
    """All players submitting zero signals — no crash, no trades."""
    np.random.seed(403)
    # Zero returns everywhere
    data = np.zeros((60, 5))
    df = _make_returns_df(data)
    port_returns = np.zeros(60)

    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    tracker = PlayerReturnTracker(path=Path(tmp.name))

    optimizer = PortfolioOptimizer(return_tracker=tracker)
    _force_rebalance(optimizer.hrp)

    result = optimizer.optimize(
        portfolio_returns=port_returns,
        player_returns_df=df,
        regime="Sideways",
    )

    assert isinstance(result, OptimizationResult)
    assert len(result.final_weights) == 5
    total = sum(result.final_weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total} with zero signals"
    # Scale factor should be 1.0 (no vol info)
    assert result.scale_factor == 1.0, (
        f"Scale factor should be 1.0 with zero returns, got {result.scale_factor}"
    )
    # CVaR should be 0
    assert result.cvar == 0.0, f"CVaR should be 0.0 with zero returns, got {result.cvar}"

    Path(tmp.name).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("PORTFOLIO OPTIMIZER TESTS")
    print("=" * 60)

    print("\n── HRP Tests ──")
    run_test("1. HRP identical streams → near-equal weights", test_hrp_identical_streams)
    run_test("2. HRP correlated pair → combined weight penalized", test_hrp_correlated_pair)
    run_test("3. HRP weights sum to 1.0 and respect 5%/40% bounds", test_hrp_weights_sum_and_bounds)
    run_test("4. Ledoit-Wolf produces valid PSD covariance", test_hrp_ledoit_wolf_psd)
    run_test("5. HRP < 20 days → equal weights fallback", test_hrp_insufficient_data)

    print("\n── VolatilityTargeter Tests ──")
    run_test("6. High realized vol (30%) → scale ~0.5", test_vol_target_high_realized)
    run_test("7. Low realized vol (7.5%) → scale ~2.0 clamped", test_vol_target_low_realized)
    run_test("8. Scale factor clamped to [min, max] leverage", test_vol_target_clamping)
    run_test("9. Bear regime → target_vol = 0.10", test_vol_target_bear_regime)

    print("\n── CVaR Tests ──")
    run_test("10. Massive losing day captured by CVaR", test_cvar_massive_loss)
    run_test("11. All positive returns → CVaR = 0", test_cvar_all_positive)
    run_test("12. Marginal CVaR identifies riskiest player", test_cvar_marginal_identifies_riskiest)
    run_test("13. CVaR breach → reduce riskiest, redistribute", test_cvar_breach_adjusts_weights)
    run_test("14. Short data (< 10 pts) → no crash", test_cvar_short_data)

    print("\n── PortfolioOptimizer Tests ──")
    run_test("15. Full pipeline → valid OptimizationResult", test_full_pipeline)
    run_test("16. Audit trail captures all steps", test_audit_trail)
    run_test("17. Blended weights sum to 1.0", test_blended_weights_sum)
    run_test("18. Zero signals → no crash, no trades", test_zero_signals_no_crash)

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL}")
    print(f"{'=' * 60}")
