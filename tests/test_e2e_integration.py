"""
End-to-End Integration Test for the 5-Player Coach Trading System.

Tests the FULL pipeline from raw market data through every module:
  1.  Load Nifty 50 data (1 year via yfinance)
  2.  Fit RegimeDetector
  3.  Create 5 RegimeAwarePlayers
  4.  Generate mock signals for 5 stocks
  5.  Run PlayerDebate (with MockLLM)
  6.  Combine via SignalCombiner (Bayesian + regime weights)
  7.  Get HRP weights from PortfolioOptimizer
  8.  Blend all three weight sources (regime + Bayesian + HRP)
  9.  Filter through PortfolioRiskManager (position/sector limits)
  10. Check CVaR constraints via CVaRRiskManager
  11. Apply VolatilityTargeter scaling
  12. Calculate transaction costs via TransactionCostModel
  13. Record results in MetricsStore
  14. Trigger one PlayerReflection cycle
  15. Print data type and shape at each handoff point
  16. Verify no crashes, no None values, no type mismatches
"""

import sys
import tempfile
import traceback
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is on path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from regime_detector import RegimeDetector, load_nifty_data
from regime_aware_player import RegimeAwarePlayer, build_registry, PLAYER_LABELS
from player_debate import (
    DebateSignal, DebateRound, DebateOutcome,
    create_debate_system, PLAYER_LABELS as DEBATE_PLAYER_LABELS,
)
from signal_combiner import (
    PlayerSignal, RegimeAwareCombiner, BayesianPlayerWeights,
    CombinedSignal, AgreementResult,
)
from portfolio_optimizer import (
    PortfolioOptimizer, PlayerReturnTracker, HRPAllocator,
    VolatilityTargeter, CVaRRiskManager, OptimizationResult, PLAYER_IDS,
)
from portfolio_risk_manager import (
    PortfolioRiskManager, PositionRecord, RiskLimits,
)
from transaction_costs import TransactionCostModel, CostAwareBacktester
from metrics_store import MetricsStore, MetricsCalculator
from player_reflection import (
    TradeReflectionEngine, ReflectionScheduler, RuleManager,
    ReflectionStore, ReflectionResult, create_reflection_system,
)
from coach_system.llm.base import MockLLMProvider


# ---------------------------------------------------------------------------
# Helper: describe a handoff value
# ---------------------------------------------------------------------------
def describe(label: str, obj: Any) -> str:
    """Print the type + shape/length at a handoff point."""
    t = type(obj).__name__
    extra = ""
    if isinstance(obj, pd.DataFrame):
        extra = f" shape={obj.shape} cols={list(obj.columns[:6])}"
    elif isinstance(obj, np.ndarray):
        extra = f" shape={obj.shape} dtype={obj.dtype}"
    elif isinstance(obj, dict):
        extra = f" keys({len(obj)})={list(obj.keys())[:6]}"
    elif isinstance(obj, (list, tuple)):
        extra = f" len={len(obj)}"
    elif isinstance(obj, (int, float)):
        extra = f" value={obj}"
    elif isinstance(obj, str):
        extra = f" len={len(obj)} preview={obj[:60]!r}"
    line = f"  [{label:40s}]  type={t:25s}{extra}"
    print(line)
    return line


def assert_not_none(val: Any, label: str) -> None:
    assert val is not None, f"{label} is None — handoff broken"


# ---------------------------------------------------------------------------
# Test constants
# ---------------------------------------------------------------------------
TEST_STOCKS = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]
MOCK_LLM_RESPONSE = '''{
  "reflection_summary": "Test reflection: momentum trades underperformed in sideways regime.",
  "rules": [
    "Avoid momentum entries when ADX < 20",
    "Reduce position size in sideways regime",
    "Tighten stops when RSI > 70 and regime is not Bull"
  ],
  "indicator_adjustments": {"RSI_7": -0.05, "MACD_12_26_9": 0.03},
  "confidence_calibration": "over_confident",
  "key_insight": "Momentum signals give too many false positives in low-ADX environments."
}'''

# Challenge/response for debate mock
MOCK_DEBATE_CHALLENGE = '''{
  "challenger": "PLAYER_2",
  "target": "PLAYER_1",
  "challenge": "Your bullish signal ignores rising VIX. In a sideways market this is risky.",
  "revised_direction": -0.3,
  "revised_confidence": 0.5
}'''

MOCK_DEBATE_RESPONSE = '''{
  "responder": "PLAYER_1",
  "response": "I acknowledge the VIX concern but strong RSI divergence supports my call.",
  "final_direction": 0.5,
  "final_confidence": 0.7
}'''


# ═══════════════════════════════════════════════════════════════════════════
# MAIN TEST
# ═══════════════════════════════════════════════════════════════════════════

def test_full_pipeline():
    """End-to-end integration: Nifty data → every module → final output."""

    errors: List[str] = []
    handoffs: List[str] = []

    def record(label, obj):
        h = describe(label, obj)
        handoffs.append(h)
        assert_not_none(obj, label)
        return obj

    print("\n" + "=" * 80)
    print("  END-TO-END INTEGRATION TEST — 5-Player Coach Trading System")
    print("=" * 80)

    # Use temp dirs so we don't pollute the project
    tmp_dir = Path(tempfile.mkdtemp(prefix="e2e_test_"))
    returns_path = tmp_dir / "player_returns.json"
    metrics_path = tmp_dir / "metrics_snapshots.json"
    reflections_dir = tmp_dir / "reflections"
    debate_dir = tmp_dir / "debates"

    # ─── STEP 1: Load 1 year of Nifty 50 data ────────────────────────
    print("\n▸ STEP 1: Load Nifty 50 data (1 year)")
    end_date = date.today().isoformat()
    start_date = (date.today() - timedelta(days=365)).isoformat()

    nifty_df = load_nifty_data(start_date=start_date, end_date=end_date)
    record("1. Nifty OHLCV DataFrame", nifty_df)

    assert isinstance(nifty_df, pd.DataFrame), "Expected DataFrame"
    assert len(nifty_df) > 100, f"Too few rows: {len(nifty_df)}"
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in nifty_df.columns, f"Missing column: {col}"
    print(f"   ✓ Loaded {len(nifty_df)} days of data")

    # ─── STEP 2: Fit RegimeDetector ──────────────────────────────────
    print("\n▸ STEP 2: Fit RegimeDetector on Nifty data")
    rd = RegimeDetector(n_states=3, n_iter=100)
    rd.fit(nifty_df)
    record("2a. RegimeDetector (fitted)", rd)

    # Predict current regime
    regime, probs, duration = rd.predict(nifty_df.tail(60))
    record("2b. Current regime", regime)
    record("2c. Regime probabilities", probs)
    record("2d. Regime duration", duration)

    assert regime in ("Bull", "Bear", "Sideways"), f"Unexpected regime: {regime}"
    assert isinstance(probs, dict) and len(probs) == 3
    assert all(0 <= v <= 1 for v in probs.values())
    assert isinstance(duration, int) and duration >= 0

    # Get regime weights for players
    regime_player_weights = rd.get_regime_weights(regime)
    record("2e. Regime player weights", regime_player_weights)
    assert len(regime_player_weights) == 5
    assert abs(sum(regime_player_weights.values()) - 1.0) < 0.01

    risk_adj = rd.get_risk_adjustments(regime)
    record("2f. Regime risk adjustments", risk_adj)
    indicator_adj = rd.get_indicator_adjustments(regime)
    record("2g. Regime indicator adjustments", indicator_adj)

    print(f"   ✓ Regime={regime}, probs={probs}, duration={duration}")

    # ─── STEP 3: Create all 5 RegimeAwarePlayers ─────────────────────
    print("\n▸ STEP 3: Create 5 RegimeAwarePlayers")
    registry = build_registry()
    record("3. Player registry", registry)

    assert len(registry) == 5
    for pid, rap in registry.items():
        assert isinstance(rap, RegimeAwarePlayer)
        assert rap.player_id == pid
        assert rap.player_label == PLAYER_LABELS[pid]

    # Adjust configs
    demo_config = {
        "label": "Test",
        "weights": {
            "RSI_7": 0.80, "STOCH_5_3": 0.75, "TSI_13_25": 0.65,
            "CMO_14": 0.60, "SUPERTREND_7_3": 0.55, "ADX_14": 0.50,
            "BBANDS_20_2": 0.50, "MFI_14": 0.45, "DEMA_20": 0.40,
            "NATR_14": 0.35,
        },
        "entry_threshold": 0.25,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    }
    adjusted_configs = {}
    for pid, rap in registry.items():
        adj = rap.adjust_config(demo_config, regime)
        adjusted_configs[pid] = adj
        assert "position_sizing_multiplier" in adj
        assert "_regime" in adj
        assert adj["_regime"] == regime
    record("3b. Adjusted configs", adjusted_configs)
    print(f"   ✓ All 5 players created and configs adjusted for {regime}")

    # ─── STEP 4: Generate mock signals from each player for 5 stocks ─
    print("\n▸ STEP 4: Generate mock signals for 5 stocks")
    np.random.seed(42)

    # Create signals: each player generates for 5 stocks
    # DebateSignal format for debate
    debate_signals_by_stock: Dict[str, Dict[str, DebateSignal]] = {}
    # PlayerSignal format for combiner
    combiner_signals_by_stock: Dict[str, List[PlayerSignal]] = {}
    # Raw format for risk manager
    risk_signals: Dict[str, Dict[str, Tuple[str, float, float]]] = {}

    for stock in TEST_STOCKS:
        debate_signals_by_stock[stock] = {}
        combiner_signals_by_stock[stock] = []

        for pid, rap in registry.items():
            label = rap.player_label
            sizing = rap.get_sizing_multiplier(regime)

            # Generate random signal
            direction = np.random.uniform(-1.0, 1.0)
            confidence = np.random.uniform(0.3, 0.95)
            base_size = 50_000 * sizing

            debate_signals_by_stock[stock][pid] = DebateSignal(
                player_id=pid,
                player_label=label,
                symbol=stock,
                direction=direction,
                confidence=confidence,
                reasoning=f"{label} signal based on regime {regime}",
                indicators={"RSI_7": np.random.uniform(30, 70)},
            )

            combiner_signals_by_stock[stock].append(PlayerSignal(
                player_id=pid,
                direction=direction,
                confidence=confidence,
                reasoning=f"{label} signal for {stock}",
            ))

            trade_dir = "LONG" if direction > 0 else "SHORT"
            risk_signals.setdefault(label, {})[stock] = (
                trade_dir, base_size, confidence,
            )

    record("4a. Debate signals (per stock)", debate_signals_by_stock)
    record("4b. Combiner signals (per stock)", combiner_signals_by_stock)
    record("4c. Risk manager signals", risk_signals)

    assert len(debate_signals_by_stock) == 5  # 5 stocks
    assert all(len(v) == 5 for v in debate_signals_by_stock.values())  # 5 players each
    print(f"   ✓ Generated signals: {len(TEST_STOCKS)} stocks × 5 players = {len(TEST_STOCKS) * 5} signals")

    # ─── STEP 5: Run PlayerDebate ────────────────────────────────────
    print("\n▸ STEP 5: Run PlayerDebate (MockLLM)")
    mock_llm = MockLLMProvider(
        responses={
            "challenge": MOCK_DEBATE_CHALLENGE,
            "respond": MOCK_DEBATE_RESPONSE,
        },
        default_response=MOCK_DEBATE_CHALLENGE,
    )

    debate_system = create_debate_system(
        llm=mock_llm,
        debate_dir=debate_dir,
        max_api_calls=5,
    )
    debate_round = debate_system["debate_round"]
    debate_store = debate_system["store"]
    debate_analytics = debate_system["analytics"]
    record("5a. Debate system", debate_system)

    # Run debate for first stock
    first_stock = TEST_STOCKS[0]
    first_signals = debate_signals_by_stock[first_stock]
    debate_outcome = debate_round.run_debate(first_signals, symbol=first_stock)
    record("5b. Debate outcome", debate_outcome)

    assert isinstance(debate_outcome, DebateOutcome)
    assert debate_outcome.symbol == first_stock
    # Post-signals may equal pre-signals if debate was skipped
    if not debate_outcome.skipped:
        assert len(debate_outcome.post_signals) > 0
        assert debate_outcome.consensus_strength >= 0.0
    print(f"   ✓ Debate {'skipped' if debate_outcome.skipped else 'completed'}: "
          f"consensus={debate_outcome.consensus_strength:.2f}, "
          f"changes={debate_outcome.signals_changed}")

    # ─── STEP 6: Combine via SignalCombiner (Bayesian + regime) ──────
    print("\n▸ STEP 6: SignalCombiner (Bayesian + regime weights)")
    combiner = RegimeAwareCombiner(
        state_dir=tmp_dir / "combiner_state",
        regime_blend=0.5,
    )
    record("6a. RegimeAwareCombiner", combiner)

    # Map regime weights from {label: weight} to {player_id: weight}
    label_to_id = {v: k for k, v in PLAYER_LABELS.items()}
    regime_weights_by_id = {}
    for label, weight in regime_player_weights.items():
        pid = label_to_id.get(label, label)
        regime_weights_by_id[pid] = weight
    record("6b. Regime weights (by ID)", regime_weights_by_id)

    # Combine signals for each stock
    combined_signals: Dict[str, CombinedSignal] = {}
    for stock in TEST_STOCKS:
        signals = combiner_signals_by_stock[stock]
        combined = combiner.combine(
            signals=signals,
            regime=regime,
            regime_weights=regime_weights_by_id,
        )
        combined_signals[stock] = combined
        assert isinstance(combined, CombinedSignal)
        assert -1.0 <= combined.final_signal <= 1.0
        assert combined.position_size_multiplier >= 0.0
        assert isinstance(combined.should_trade, bool)

    record("6c. Combined signals", combined_signals)

    bayesian_weights = combiner.bayesian_weights.get_weights(regime)
    record("6d. Bayesian weights", bayesian_weights)
    assert len(bayesian_weights) == 5
    assert abs(sum(bayesian_weights.values()) - 1.0) < 0.02

    print(f"   ✓ Combined {len(combined_signals)} stock signals")
    for stock, cs in combined_signals.items():
        print(f"     {stock}: signal={cs.final_signal:+.3f} "
              f"size_mult={cs.position_size_multiplier:.2f} "
              f"trade={cs.should_trade}")

    # ─── STEP 7: Get HRP weights from PortfolioOptimizer ────────────
    print("\n▸ STEP 7: PortfolioOptimizer — HRP weights")

    # Generate synthetic player returns for HRP (60 days)
    np.random.seed(42)
    n_days = 60
    mean = [0.001, 0.0005, 0.0007, 0.0008, 0.0009]
    cov_matrix = np.array([
        [0.0004, -0.0001,  0.0001,  0.00005, 0.00028],
        [-0.0001, 0.0001,  0.00003, 0.00002, -0.00005],
        [0.0001,  0.00003, 0.0002,  0.00005, 0.00008],
        [0.00005, 0.00002, 0.00005, 0.0003,  0.00006],
        [0.00028, -0.00005, 0.00008, 0.00006, 0.0004],
    ])
    returns_raw = np.random.multivariate_normal(mean, cov_matrix, size=n_days)
    player_returns_df = pd.DataFrame(returns_raw, columns=PLAYER_IDS)
    record("7a. Player returns DataFrame", player_returns_df)

    portfolio_returns = player_returns_df.mean(axis=1).values
    record("7b. Portfolio returns array", portfolio_returns)

    tracker = PlayerReturnTracker(path=returns_path)
    for i in range(n_days):
        d = (date.today() - timedelta(days=n_days - i)).isoformat()
        pnls = {pid: float(player_returns_df.iloc[i][pid]) for pid in PLAYER_IDS}
        tracker.record_daily_pnl(d, pnls)

    optimizer = PortfolioOptimizer(
        return_tracker=tracker,
        hrp=HRPAllocator(),
        vol_targeter=VolatilityTargeter(),
        cvar_manager=CVaRRiskManager(),
    )
    # Force HRP rebalance
    optimizer.hrp._calls_since_rebalance = HRPAllocator.REBALANCE_INTERVAL

    hrp_weights = optimizer.hrp.get_allocation(player_returns_df)
    record("7c. HRP weights", hrp_weights)
    assert len(hrp_weights) == 5
    assert abs(sum(hrp_weights.values()) - 1.0) < 0.01
    print(f"   ✓ HRP weights: {', '.join(f'{k}={v:.1%}' for k, v in hrp_weights.items())}")

    # ─── STEP 8: Blend all three weight sources ──────────────────────
    print("\n▸ STEP 8: Blend regime + Bayesian + HRP weights")
    blended_weights, hrp_w = optimizer.compute_blended_weights(
        player_returns_df=player_returns_df,
        regime_weights=regime_weights_by_id,
        bayesian_weights=bayesian_weights,
    )
    record("8a. Blended weights", blended_weights)
    record("8b. HRP weights (from blend)", hrp_w)

    assert len(blended_weights) == 5
    assert abs(sum(blended_weights.values()) - 1.0) < 0.01
    print(f"   ✓ Blended weights: {', '.join(f'{k}={v:.1%}' for k, v in blended_weights.items())}")

    # ─── STEP 9: Filter through PortfolioRiskManager ─────────────────
    print("\n▸ STEP 9: PortfolioRiskManager — filter signals")
    risk_mgr = PortfolioRiskManager(total_capital=500_000, limits=RiskLimits())
    record("9a. PortfolioRiskManager", risk_mgr)

    current_positions: List[PositionRecord] = []  # no existing positions
    approved, rejected = risk_mgr.filter_signals(
        player_signals=risk_signals,
        current_positions=current_positions,
        total_capital=500_000,
    )
    record("9b. Approved signals", approved)
    record("9c. Rejected signals", rejected)

    assert isinstance(approved, dict)
    assert isinstance(rejected, dict)
    total_approved = sum(len(v) for v in approved.values())
    total_rejected = sum(len(v) for v in rejected.values())
    print(f"   ✓ Approved: {total_approved} trades, Rejected: {total_rejected} trades")

    # Risk report
    approved_positions: List[PositionRecord] = []
    for player, sigs in approved.items():
        for sym, (direction, size, conf) in sigs.items():
            est_price = 2500.0
            qty = max(1, int(size / est_price))
            approved_positions.append(PositionRecord(
                symbol=sym, player=player, direction=direction,
                quantity=qty, entry_price=est_price, current_price=est_price,
            ))

    risk_report = risk_mgr.risk_report(approved_positions, total_capital=500_000)
    record("9d. Risk report", risk_report)
    assert "breaches" in risk_report
    assert "gross_exposure_pct" in risk_report

    # ─── STEP 10: CVaR constraints ───────────────────────────────────
    print("\n▸ STEP 10: CVaR constraints via CVaRRiskManager")
    cvar_mgr = CVaRRiskManager()

    cvar_value = cvar_mgr.compute_cvar(portfolio_returns)
    record("10a. CVaR (95%)", cvar_value)
    assert isinstance(cvar_value, float)
    assert cvar_value >= 0.0

    marginal_cvar = cvar_mgr.compute_marginal_cvar(portfolio_returns, player_returns_df)
    record("10b. Marginal CVaR", marginal_cvar)
    assert len(marginal_cvar) == 5

    adj_weights, adj_cvar, breached, adj_marginal, cvar_log = \
        cvar_mgr.adjust_weights_for_cvar(
            weights=blended_weights,
            portfolio_returns=portfolio_returns,
            player_returns_df=player_returns_df,
            max_cvar=0.03,
        )
    record("10c. CVaR-adjusted weights", adj_weights)
    record("10d. CVaR value", adj_cvar)
    record("10e. CVaR breached?", breached)
    record("10f. CVaR adjustment log", cvar_log)

    assert isinstance(adj_weights, dict) and len(adj_weights) == 5
    assert isinstance(breached, bool)
    print(f"   ✓ CVaR={adj_cvar:.4f}, breached={breached}")

    # ─── STEP 11: VolatilityTargeter scaling ─────────────────────────
    print("\n▸ STEP 11: VolatilityTargeter scaling")
    vol_targeter = VolatilityTargeter()
    scale_factor, realized_vol, target_vol = vol_targeter.get_scale_factor(
        portfolio_returns, regime=regime,
    )
    record("11a. Scale factor", scale_factor)
    record("11b. Realized vol", realized_vol)
    record("11c. Target vol", target_vol)

    assert isinstance(scale_factor, float) and scale_factor > 0
    assert isinstance(realized_vol, float) and realized_vol >= 0
    assert isinstance(target_vol, float) and target_vol > 0
    print(f"   ✓ Scale={scale_factor:.2f}, realVol={realized_vol:.2%}, targetVol={target_vol:.2%}")

    # Full optimize call
    print("\n   Running full PortfolioOptimizer.optimize() ...")
    # Force HRP rebalance again
    optimizer.hrp._calls_since_rebalance = HRPAllocator.REBALANCE_INTERVAL
    opt_result = optimizer.optimize(
        portfolio_returns=portfolio_returns,
        player_returns_df=player_returns_df,
        regime=regime,
        regime_weights=regime_weights_by_id,
        bayesian_weights=bayesian_weights,
    )
    record("11d. OptimizationResult", opt_result)
    assert isinstance(opt_result, OptimizationResult)
    assert len(opt_result.final_weights) == 5
    assert abs(sum(opt_result.final_weights.values()) - 1.0) < 0.01
    assert opt_result.scale_factor > 0
    assert opt_result.regime == regime
    print(f"   ✓ Optimization complete: {opt_result.regime}, "
          f"scale={opt_result.scale_factor:.2f}, CVaR={opt_result.cvar:.4f}")

    # ─── STEP 12: Transaction costs ──────────────────────────────────
    print("\n▸ STEP 12: Transaction costs via TransactionCostModel")
    cost_model = TransactionCostModel()

    # Calculate cost for a sample trade
    sample_cost = cost_model.calculate_cost(
        price=2500.0,
        quantity=200,
        side="BUY",
        trade_type="intraday",
        avg_daily_volume=5_000_000,
        volatility=0.018,
        signal_type="momentum",
    )
    record("12a. Sample trade cost (INR)", sample_cost)
    assert isinstance(sample_cost, float) and sample_cost > 0

    cost_bps = cost_model.calculate_cost_bps(
        price=2500.0,
        quantity=200,
        side="BUY",
        trade_type="intraday",
        avg_daily_volume=5_000_000,
        volatility=0.018,
        signal_type="momentum",
    )
    record("12b. Cost in bps", cost_bps)
    assert isinstance(cost_bps, float) and cost_bps > 0

    cost_summary = cost_model.summary()
    record("12c. Cost breakdown", cost_summary)
    assert isinstance(cost_summary, dict)

    # CostAwareBacktester
    backtester = CostAwareBacktester(cost_model=cost_model)
    mock_trades = [
        {"entry_price": 2500, "exit_price": 2540, "quantity": 200,
         "signal_type": "momentum", "trade_type": "intraday"},
        {"entry_price": 1600, "exit_price": 1585, "quantity": 125,
         "signal_type": "mean_reversion", "trade_type": "intraday"},
        {"entry_price": 3400, "exit_price": 3450, "quantity": 150,
         "signal_type": "neutral", "trade_type": "delivery"},
    ]
    backtest_result = backtester.apply(mock_trades, initial_capital=100_000)
    record("12d. Backtest cost result", backtest_result)
    assert "total_costs" in backtest_result
    assert "adjusted_pnl" in backtest_result
    print(f"   ✓ Trade cost={sample_cost:.2f} INR ({cost_bps:.1f} bps), "
          f"backtest drag={backtest_result.get('cost_drag_pct', 0):.2f}%")

    # ─── STEP 13: Record results in MetricsStore ─────────────────────
    print("\n▸ STEP 13: MetricsStore — record daily snapshot")
    metrics_store = MetricsStore(path=metrics_path)
    record("13a. MetricsStore", metrics_store)

    # Build a comprehensive snapshot
    daily_snapshot = {
        "date": date.today().isoformat(),
        "regime": regime,
        "regime_probabilities": probs,
        "regime_duration": duration,
        "portfolio": {
            "equity": 510_000,
            "pnl": 10_000,
            "sharpe": MetricsCalculator.compute_sharpe(portfolio_returns),
            "sortino": MetricsCalculator.compute_sortino(portfolio_returns),
            "max_drawdown": MetricsCalculator.compute_max_drawdown(
                np.cumprod(1 + portfolio_returns)
            ),
        },
        "players": {
            pid: {
                "weight": opt_result.final_weights.get(pid, 0.2),
                "pnl": float(np.sum(player_returns_df[pid].values)),
                "win_rate": MetricsCalculator.compute_win_rate(
                    player_returns_df[pid].values
                ),
            }
            for pid in PLAYER_IDS
        },
        "risk": {
            "cvar_95": opt_result.cvar,
            "cvar_breached": opt_result.cvar_breached,
            "scale_factor": opt_result.scale_factor,
            "realized_vol": opt_result.realized_vol,
            "target_vol": opt_result.target_vol,
            "gross_exposure_pct": risk_report.get("gross_exposure_pct", 0),
        },
        "costs": {
            "total_costs": backtest_result.get("total_costs", 0),
            "avg_cost_bps": backtest_result.get("avg_cost_bps", 0),
        },
        "signals": {
            "total_generated": len(TEST_STOCKS) * 5,
            "approved": total_approved,
            "rejected": total_rejected,
        },
    }
    metrics_store.record_daily_snapshot(daily_snapshot)
    record("13b. Daily snapshot", daily_snapshot)

    latest = metrics_store.get_latest()
    record("13c. Latest snapshot retrieved", latest)
    assert latest["date"] == date.today().isoformat()
    assert latest["regime"] == regime
    assert metrics_store.snapshot_count() == 1

    # Test metric extraction
    sharpe_history = metrics_store.get_metric_history("portfolio.sharpe", days=30)
    record("13d. Sharpe history", sharpe_history)
    assert len(sharpe_history) == 1
    print(f"   ✓ Snapshot recorded: Sharpe={daily_snapshot['portfolio']['sharpe']:.2f}, "
          f"equity={daily_snapshot['portfolio']['equity']:,}")

    # ─── STEP 14: Trigger one PlayerReflection cycle ─────────────────
    print("\n▸ STEP 14: PlayerReflection cycle (MockLLM)")
    reflection_llm = MockLLMProvider(
        responses={"trade": MOCK_LLM_RESPONSE},
        default_response=MOCK_LLM_RESPONSE,
    )
    reflection_system = create_reflection_system(
        llm=reflection_llm,
        reflections_dir=reflections_dir,
        trigger_every=5,
    )
    record("14a. Reflection system", reflection_system)

    engine = reflection_system["engine"]
    scheduler = reflection_system["scheduler"]
    rule_mgr = reflection_system["rule_manager"]
    refl_store = reflection_system["store"]

    # Generate mock trades for reflection
    mock_trade_list = []
    for i in range(10):
        pnl = np.random.uniform(-500, 1000)
        mock_trade_list.append({
            "symbol": TEST_STOCKS[i % len(TEST_STOCKS)],
            "direction": "LONG" if np.random.random() > 0.3 else "SHORT",
            "entry_price": 2500.0 + np.random.uniform(-100, 100),
            "exit_price": 2500.0 + np.random.uniform(-50, 150),
            "pnl": pnl,
            "bars_held": np.random.randint(2, 20),
            "exit_reason": "stop_loss" if pnl < 0 else "target",
        })
    record("14b. Mock trades for reflection", mock_trade_list)

    # Force a reflection for PLAYER_1
    refl_result = scheduler.force_reflect(
        player_id="PLAYER_1",
        player_label="Aggressive",
        trades=mock_trade_list,
        regime=regime,
    )
    record("14c. Reflection result", refl_result)
    assert isinstance(refl_result, ReflectionResult)
    assert refl_result.player_id == "PLAYER_1"
    assert refl_result.player_label == "Aggressive"
    assert refl_result.regime == regime
    assert refl_result.trade_count == 10
    assert len(refl_result.rules) > 0
    assert refl_result.confidence_calibration in (
        "over_confident", "under_confident", "well_calibrated", "unknown"
    )

    # Check rules were added
    active_rules = rule_mgr.get_active_rules("PLAYER_1")
    record("14d. Active rules for PLAYER_1", active_rules)
    assert len(active_rules) > 0

    # Check reflection was stored
    stored = refl_store.get_reflections("PLAYER_1", last_n=1)
    record("14e. Stored reflections", stored)
    assert len(stored) == 1
    assert stored[0].reflection_id == refl_result.reflection_id

    # Build memory context
    memory_ctx_factory = reflection_system["memory_context_factory"]
    memory_ctx = memory_ctx_factory(regime_player=registry["PLAYER_1"])
    context_str = memory_ctx.build_context(
        player_id="PLAYER_1",
        player_label="Aggressive",
        regime=regime,
    )
    record("14f. Player memory context", context_str)

    print(f"   ✓ Reflection complete: {len(refl_result.rules)} rules, "
          f"calibration={refl_result.confidence_calibration}, "
          f"insight={refl_result.key_insight[:50]}")

    # ─── STEP 15: Summary of all handoff points ──────────────────────
    print("\n" + "=" * 80)
    print("  HANDOFF SUMMARY — Data types at each module boundary")
    print("=" * 80)
    for h in handoffs:
        print(h)

    # ─── STEP 16: Final validation ───────────────────────────────────
    print("\n" + "=" * 80)
    print("  FINAL VALIDATION")
    print("=" * 80)

    checks = [
        ("No None values at handoff",
         all("is None" not in h for h in handoffs)),
        ("Nifty data loaded",
         len(nifty_df) > 100),
        ("Regime detected",
         regime in ("Bull", "Bear", "Sideways")),
        ("5 players created",
         len(registry) == 5),
        ("Signals generated (25 total)",
         len(TEST_STOCKS) * 5 == 25),
        ("Debate produced outcome",
         isinstance(debate_outcome, DebateOutcome)),
        ("Combiner produced CombinedSignals",
         all(isinstance(v, CombinedSignal) for v in combined_signals.values())),
        ("Bayesian weights valid",
         abs(sum(bayesian_weights.values()) - 1.0) < 0.02),
        ("HRP weights valid",
         abs(sum(hrp_weights.values()) - 1.0) < 0.01),
        ("Blended weights valid",
         abs(sum(blended_weights.values()) - 1.0) < 0.01),
        ("Risk manager filtered signals",
         total_approved + total_rejected > 0),
        ("CVaR computed",
         cvar_value >= 0.0),
        ("Vol targeter scaled",
         scale_factor > 0),
        ("OptimizationResult valid",
         isinstance(opt_result, OptimizationResult)),
        ("Transaction cost positive",
         sample_cost > 0),
        ("Metrics snapshot stored",
         metrics_store.snapshot_count() == 1),
        ("Reflection produced rules",
         len(refl_result.rules) > 0),
    ]

    all_passed = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {name}")

    print()
    if all_passed:
        print("  " + "=" * 60)
        print("  ALL 17 CHECKS PASSED — Full pipeline integration verified!")
        print("  " + "=" * 60)
    else:
        failed = [name for name, passed in checks if not passed]
        print(f"  FAILED: {failed}")
        raise AssertionError(f"Integration test failed: {failed}")

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        test_full_pipeline()
    except Exception as e:
        print(f"\n{'!' * 80}")
        print(f"  INTEGRATION TEST FAILED: {e}")
        print(f"{'!' * 80}")
        traceback.print_exc()
        sys.exit(1)
