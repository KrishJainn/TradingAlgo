"""
AQTIS Unified Paper Trading Engine.

Routes EVERY trading decision through the full agent stack:
- Signal generation (rule-based, 87 indicators)
- Orchestrator lightweight_signal_check (memory-based, 0 tokens)
- Pre-trade workflow for high-conviction signals (LLM-assisted)
- Risk manager position sizing + adaptive exits
- Batch post-mortem learning every 10 days
- Periodic review every 3 days
- Mid-run strategy review at midpoint

Designed for two-phase operation:
  Phase A: Backtesting (60-day historical data) until viable model
  Phase B: Paper trading (forward-looking) once model proves itself

Token budget: ~30 LLM calls per 60-day run
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Add project root
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import the full 87-indicator system from trading_evolution
try:
    from trading_evolution.indicators.calculator import IndicatorCalculator
    from trading_evolution.indicators.normalizer import SignalAggregator
    HAS_FULL_INDICATORS = True
except ImportError:
    HAS_FULL_INDICATORS = False


class PaperTrader:
    """
    Unified paper trading engine that routes every decision through agents.

    5-Phase Daily Loop:
      1. SENSE   — 87 indicators + signal generation (0 tokens)
      2. DECIDE  — orchestrator.lightweight_signal_check filters signals
                    High-conviction signals (>0.25) go through full pre_trade_workflow
      3. EXECUTE — Risk manager sizes positions, sets adaptive stop/profit
      4. LEARN   — Every 10 days: batch post-mortem, weight adjustments
      5. ADAPT   — Every 3 days: periodic review, degradation detection

    Learning Flow:
      Trade → memory.store_trade() → prediction_tracker.record_outcome()
        ↓ (every 10 days)
      post_mortem.analyze_trade() → coach_session + indicator_scores stored
        ↓
      orchestrator.apply_daily_learnings() → weight deltas extracted
        ↓
      _indicator_weights MUTATED → next day uses updated weights
        ↓ (next run)
      _bootstrap_from_memory() → loads best snapshot from prior runs
    """

    def __init__(
        self,
        memory,
        orchestrator,
        config,
        data_provider=None,
        llm_budget: int = 80,
    ):
        self.memory = memory
        self.orchestrator = orchestrator
        self.config = config
        self.llm_budget = llm_budget
        self._llm_calls_used = 0

        # Data provider
        if data_provider:
            self.data_provider = data_provider
        else:
            from aqtis.data.market_data import MarketDataProvider
            cache_dir = getattr(config, "system", {})
            if hasattr(cache_dir, "data_dir"):
                cache_dir = str(cache_dir.data_dir)
            else:
                cache_dir = "data_cache"
            self.data_provider = MarketDataProvider(cache_dir=cache_dir)

        # 87-indicator calculator
        self._indicator_calculator = None
        if HAS_FULL_INDICATORS:
            try:
                self._indicator_calculator = IndicatorCalculator()
            except Exception as e:
                logger.warning(f"Failed to init IndicatorCalculator: {e}")

        # Default weights (evolved from 5-player model best performers)
        self._indicator_weights = {
            # Momentum
            "RSI_14": 0.35, "RSI_7": 0.15, "MACD_12_26_9": 0.30,
            "STOCH_14_3": 0.20, "STOCH_5_3": 0.15, "WILLR_14": 0.20,
            "CCI_20": 0.15, "MOM_10": 0.12, "CMO_14": 0.10,
            "TSI_13_25": 0.18, "UO_7_14_28": 0.12, "AO_5_34": 0.10,
            "ROC_10": 0.12, "KST": 0.10,
            # Trend
            "ADX_14": 0.25, "SUPERTREND_10_2": 0.20, "SUPERTREND_7_3": 0.18,
            "AROON_14": 0.12, "PSAR": 0.15, "ICHIMOKU": 0.10,
            "VORTEX_14": 0.10, "LINREG_SLOPE_14": 0.15,
            # Volatility
            "BBANDS_20_2": 0.20, "BBANDS_10_1.5": 0.12,
            "KC_20_2": 0.10, "ATR_14": 0.08, "NATR_14": 0.10,
            "ZSCORE_20": 0.15,
            # Volume
            "OBV": 0.12, "CMF_20": 0.15, "MFI_14": 0.18,
            "AD": 0.08, "EFI_13": 0.10,
            # Overlap
            "EMA_10": 0.08, "EMA_20": 0.10, "EMA_50": 0.08,
            "SMA_20": 0.06, "SMA_50": 0.08, "HMA_9": 0.10,
            "DEMA_20": 0.08, "TEMA_10": 0.10, "T3_5": 0.08,
        }

        # Entry threshold (can be adjusted by learning)
        self._entry_threshold = 0.15

        # Simulation state
        self._run_id = None
        self._run_number = 0
        self._capital = 100_000.0
        self._positions: Dict[str, Dict] = {}
        self._daily_trades: List[Dict] = []
        self._all_trades: List[Dict] = []
        self._equity_curve: List[Dict] = []
        self._daily_pnl: List[float] = []
        self._agent_log: List[Dict] = []

        # Learning state
        self._trades_since_learning = 0
        self._days_since_review = 0
        self._weight_mutations: List[Dict] = []
        self._confidence_scaling = 1.0  # Adjusted by prediction_tracker LLM

    # ─────────────────────────────────────────────────────────────────
    # MAIN RUN LOOP
    # ─────────────────────────────────────────────────────────────────

    def run(
        self,
        symbols: List[str] = None,
        days: int = 60,
        initial_capital: float = None,
        run_number: int = None,
    ) -> Dict[str, Any]:
        """
        Run a full agent-driven backtest/paper-trade.

        Every decision flows through the agent stack:
        - Signals filtered by orchestrator memory
        - Positions sized by risk manager
        - Exits managed adaptively
        - Learning applied periodically
        """
        # Configuration
        if symbols is None:
            if hasattr(self.config, "market_data"):
                symbols = self.config.market_data.symbols
            else:
                symbols = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

        if initial_capital is not None:
            self._capital = initial_capital
        elif hasattr(self.config, "execution"):
            self._capital = self.config.execution.initial_capital

        starting_capital = self._capital

        # Determine run number
        pnl_trend = self.memory.get_cross_run_pnl_trend()
        if run_number is None:
            self._run_number = len(pnl_trend) + 1
        else:
            self._run_number = run_number

        logger.info(
            f"[PaperTrader] Starting run #{self._run_number}: "
            f"{days} days, {len(symbols)} symbols, capital={self._capital:,.0f}"
        )

        # ── BOOTSTRAP: Load best weights from memory ──
        self._bootstrap_from_memory()

        # ── REGISTER DEFAULT STRATEGY ──
        # The strategy_generator.analyze_signal() requires at least one active
        # strategy in memory.  Without this, it returns should_trade=False for
        # every signal, producing 0 trades.
        self._ensure_default_strategy_registered(symbols)

        # Register run in memory
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_id = self.orchestrator.start_simulation_run(
            run_number=self._run_number,
            session_id=session_id,
            days=days,
            symbols=len(symbols),
            config={
                "symbols": symbols,
                "initial_capital": self._capital,
                "llm_budget": self.llm_budget,
                "engine": "paper_trader",
                "entry_threshold": self._entry_threshold,
            },
        )

        # Fetch data
        logger.info(f"Fetching {days}+30 days of data for {len(symbols)} symbols...")
        market_data = self._fetch_data(symbols, days)

        if not market_data:
            return {"error": "No data fetched for any symbol"}

        fetched_symbols = list(market_data.keys())
        logger.info(f"Data fetched for {len(fetched_symbols)} symbols")

        # Normalize timezone-aware indexes
        for symbol in list(market_data.keys()):
            df = market_data[symbol]
            if df.index.tz is not None:
                market_data[symbol] = df.tz_localize(None)

        # Determine simulation date range
        all_dates = set()
        for df in market_data.values():
            all_dates.update(df.index.normalize().unique())
        sorted_dates = sorted(all_dates)

        warmup = min(20, len(sorted_dates) // 4)
        sim_dates = sorted_dates[warmup: warmup + days]

        if not sim_dates:
            return {"error": "Insufficient data for simulation"}

        logger.info(
            f"Simulating {len(sim_dates)} days: "
            f"{sim_dates[0].strftime('%Y-%m-%d')} to {sim_dates[-1].strftime('%Y-%m-%d')}"
        )

        # Pre-compute indicators (0 tokens)
        indicators = self._compute_indicators(market_data)

        # ════════════════════════════════════════════════════════════════
        # MAIN DAILY LOOP
        # ════════════════════════════════════════════════════════════════
        mid_run_done = False

        for day_idx, sim_date in enumerate(sim_dates):
            day_str = sim_date.strftime("%Y-%m-%d")
            self._daily_trades = []
            self._days_since_review += 1

            # ── Phase 1: SENSE ── (0 tokens)
            day_signals = []
            for symbol in fetched_symbols:
                df = market_data[symbol]
                ind = indicators.get(symbol)
                if ind is None:
                    continue

                day_mask = df.index.normalize() == sim_date
                if not day_mask.any():
                    continue

                bar_idx = df.index.get_indexer(df.index[day_mask])[-1]
                if bar_idx < warmup:
                    continue

                signal = self._generate_signal(symbol, df, ind, bar_idx)
                if signal.get("score", 0) != 0:
                    day_signals.append((symbol, df, ind, bar_idx, signal))

            # ── Phase 2: DECIDE ── (0-2 tokens)
            # Check exits on existing positions FIRST (adaptive exits)
            for symbol in list(self._positions.keys()):
                df = market_data.get(symbol)
                ind = indicators.get(symbol)
                if df is None or ind is None:
                    continue
                day_mask = df.index.normalize() == sim_date
                if not day_mask.any():
                    continue
                bar_idx = df.index.get_indexer(df.index[day_mask])[-1]
                self._check_exit_adaptive(symbol, df, ind, bar_idx, day_str)

            # Filter and route entry signals through agents
            for symbol, df, ind, bar_idx, signal in day_signals:
                if symbol in self._positions:
                    continue  # Already have position

                self._process_signal_through_agents(
                    symbol, df, ind, bar_idx, signal, day_str
                )

            # ── Phase 3: EXECUTE ── (tracked in _process_signal_through_agents)

            # ── End-of-day bookkeeping ──
            day_pnl = self._mark_to_market(market_data, sim_date)
            self._daily_pnl.append(day_pnl)
            self._equity_curve.append({
                "date": day_str,
                "equity": self._capital + self._unrealized_pnl(market_data, sim_date),
                "realized_pnl": sum(self._daily_pnl),
                "positions": len(self._positions),
                "trades_today": len(self._daily_trades),
            })

            # Store trades to memory
            for trade in self._daily_trades:
                self.memory.store_trade(trade)
                self._all_trades.append(trade)
                self._trades_since_learning += 1

            # ── Phase 4: LEARN ── (every 10 days)
            if self._trades_since_learning >= 5 and (day_idx + 1) % 10 == 0:
                self._batch_learning(day_str, day_idx, len(sim_dates))

            # ── Phase 5: ADAPT ── (every 3 days)
            if self._days_since_review >= 3:
                self._periodic_review(day_str, day_idx, len(sim_dates), starting_capital)
                self._days_since_review = 0

            # Mid-run strategy review (once, around midpoint)
            if not mid_run_done and day_idx >= len(sim_dates) // 2:
                self._mid_run_review(starting_capital)
                mid_run_done = True

            # Progress logging
            if (day_idx + 1) % 10 == 0 or day_idx == len(sim_dates) - 1:
                cum_pnl = sum(self._daily_pnl)
                logger.info(
                    f"Day {day_idx + 1}/{len(sim_dates)}: "
                    f"P&L={cum_pnl:+,.0f}, Trades={len(self._all_trades)}, "
                    f"Positions={len(self._positions)}, "
                    f"LLM={self._llm_calls_used}/{self.llm_budget}, "
                    f"Mutations={len(self._weight_mutations)}"
                )

        # ── FINALIZE ──
        return self._finalize_run(market_data, sim_dates, fetched_symbols, starting_capital)

    # ─────────────────────────────────────────────────────────────────
    # BOOTSTRAP FROM MEMORY
    # ─────────────────────────────────────────────────────────────────

    def _bootstrap_from_memory(self):
        """Load best weights from prior runs via orchestrator."""
        best = self.orchestrator.get_best_weights("aqtis_multi_indicator")

        if best and best.get("weights"):
            loaded_weights = best["weights"]
            applied = 0
            for k, v in loaded_weights.items():
                if k in self._indicator_weights:
                    self._indicator_weights[k] = v
                    applied += 1

            if best.get("entry_threshold"):
                self._entry_threshold = best["entry_threshold"]

            self._agent_log.append({
                "day": "bootstrap",
                "agent": "orchestrator",
                "action": "load_best_weights",
                "detail": f"Loaded {applied} weights from run with "
                          f"Sharpe={best.get('sharpe', 0):.2f}, "
                          f"source={best.get('source', 'unknown')}",
            })
            logger.info(
                f"[Bootstrap] Loaded {applied} weights from best snapshot "
                f"(Sharpe={best.get('sharpe', 0):.2f})"
            )
        else:
            logger.info("[Bootstrap] No prior snapshots found, using default weights")

    def _ensure_default_strategy_registered(self, symbols: List[str]):
        """
        Ensure at least one active strategy exists in memory.

        The strategy_generator.analyze_signal() requires active strategies
        to work — it immediately returns should_trade=False when none exist.
        On a fresh DB this means 0 trades forever, preventing the learning
        loop from ever bootstrapping.
        """
        existing = self.memory.get_active_strategies()
        if existing:
            logger.info(f"[Bootstrap] {len(existing)} active strategies already in memory")
            return

        # Register the default multi-indicator strategy
        indicator_names = list(self._indicator_weights.keys())
        self.memory.store_strategy({
            "strategy_id": "aqtis_multi_indicator",
            "strategy_name": "AQTIS Multi-Indicator Ensemble",
            "strategy_type": "ensemble",
            "description": (
                "87-indicator weighted ensemble combining momentum, trend, "
                "volatility, volume and overlap signals. Weights are mutated "
                "by the 5-player coach model across simulation runs."
            ),
            # Pass as dict — store_strategy() calls json.dumps() internally
            "parameters": {
                "indicators": indicator_names,
                "entry_threshold": self._entry_threshold,
                "n_indicators": len(indicator_names),
            },
            "total_trades": 0,
            "win_rate": 0.5,
            "sharpe_ratio": 0.5,  # Neutral starting Sharpe
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "avg_return": 0.0,
            "backtest_sharpe": 0.0,
            "live_sharpe": 0.0,
            "overfitting_score": 0.0,
            # Pass as dict — store_strategy() calls json.dumps() internally
            "performance_by_regime": {},
            "is_active": 1,
            "last_used_at": datetime.now().isoformat(),
        })
        logger.info(
            f"[Bootstrap] Registered default strategy 'aqtis_multi_indicator' "
            f"with {len(indicator_names)} indicators"
        )

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL PROCESSING THROUGH AGENTS
    # ─────────────────────────────────────────────────────────────────

    def _process_signal_through_agents(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame,
        bar_idx: int, signal: Dict, day_str: str,
    ):
        """
        Route a signal through the full agent stack.

        1. orchestrator.lightweight_signal_check() — memory-based filter (0 tokens)
        2. If high conviction (>0.25): orchestrator.pre_trade_workflow (LLM)
        3. risk_manager.dynamic_stop_loss() — regime-aware levels
        4. risk_manager position sizing — Kelly criterion
        5. Execute entry
        """
        raw_score = signal.get("score", 0)

        # Step 1: Memory-based signal filter (0 tokens)
        checked = self.orchestrator.lightweight_signal_check(signal)
        adjusted_score = checked.get("adjusted_score", raw_score)
        adjustments = checked.get("adjustments", [])

        self._agent_log.append({
            "day": day_str,
            "agent": "orchestrator.lightweight_signal_check",
            "action": "filter",
            "symbol": symbol,
            "detail": f"raw={raw_score:.3f} -> adjusted={adjusted_score:.3f} "
                      f"({'|'.join(adjustments[:3]) if adjustments else 'no adj'})",
        })

        # Gate: must exceed entry threshold
        if abs(adjusted_score) < self._entry_threshold:
            return

        # Max positions check
        if len(self._positions) >= 5:
            return

        action = "BUY" if adjusted_score > 0 else "SELL"
        entry_price = signal.get("close", 0)
        if entry_price <= 0:
            return

        confidence = min(abs(adjusted_score) * self._confidence_scaling, 0.95)
        regime = signal.get("regime", "unknown")
        atr = signal.get("atr", entry_price * 0.02)

        # Step 2: High-conviction → full pre-trade workflow (LLM)
        pre_trade_decision = None
        if abs(adjusted_score) > 0.25 and self._can_use_llm():
            try:
                pre_trade_decision = self.orchestrator.pre_trade_workflow({
                    "asset": symbol,
                    "action": action,
                    "price": entry_price,
                    "score": adjusted_score,
                    "regime": regime,
                    "confidence": confidence,
                    "signals": signal.get("signals", {}),
                    "atr": atr,
                })
                self._use_llm_call()

                decision = pre_trade_decision.get("decision", "?")

                self._agent_log.append({
                    "day": day_str,
                    "agent": "orchestrator.pre_trade_workflow",
                    "action": "full_analysis",
                    "symbol": symbol,
                    "detail": f"decision={decision}",
                })

                if decision == "execute":
                    # Use agent-provided position size if available
                    if pre_trade_decision.get("position_size", 0) > 0:
                        position_value = pre_trade_decision["position_size"]
                        shares = int(position_value / entry_price)
                        if shares < 1:
                            pass  # Fall through to rule-based sizing
                        else:
                            self._execute_entry(
                                symbol, action, entry_price, shares, atr, regime,
                                confidence, signal, day_str, pre_trade_decision,
                            )
                            return

                # For skip/reject/error: fall through to rule-based sizing
                # This is essential for bootstrapping — the first run has no
                # memory, so the strategy generator is too conservative.
                # We let rule-based trades through so the learning loop can
                # accumulate data for future, better-informed agent decisions.
                if decision in ("reject", "skip"):
                    # Reduce confidence slightly when agents disagreed
                    confidence *= 0.75
                    self._agent_log.append({
                        "day": day_str,
                        "agent": "paper_trader",
                        "action": "bootstrap_fallback",
                        "symbol": symbol,
                        "detail": f"Agent {decision}, falling through to rule-based "
                                  f"sizing with reduced confidence={confidence:.3f}",
                    })

            except Exception as e:
                logger.warning(f"Pre-trade workflow error for {symbol}: {e}")
                # Fall through to rule-based sizing

        # Step 3: Risk manager dynamic stop/profit levels (0 tokens)
        stop_profit = self.orchestrator.agents["risk_manager"].execute({
            "action": "dynamic_stop_loss",
            "position": {
                "entry_price": entry_price,
                "atr": atr,
                "regime": regime,
                "action": action,
            },
            "current_indicators": signal.get("all_signals", signal.get("signals", {})),
        })

        # Step 4: Position sizing via risk manager Kelly criterion (0 tokens)
        position_result = self.orchestrator.agents["risk_manager"].execute({
            "action": "position_size",
            "prediction": {
                "predicted_confidence": confidence,
                "predicted_return": abs(adjusted_score) * 0.02,
                "asset": symbol,
            },
            "portfolio_value": self._capital,
        })

        position_value = position_result.get("position_size", 0)
        if position_value <= 0:
            # Fallback: simple sizing based on confidence and capital
            # Use 1-5% of capital depending on confidence level
            # This ensures at least 1 share for typical NIFTY 50 stocks
            position_pct = max(0.01, min(confidence * 0.10, 0.05))
            position_value = self._capital * position_pct

        shares = int(position_value / entry_price)
        if shares < 1:
            # Last resort: allocate minimum 1 share if signal passed threshold
            shares = 1

        self._execute_entry(
            symbol, action, entry_price, shares, atr, regime,
            confidence, signal, day_str, None, stop_profit,
        )

    def _execute_entry(
        self, symbol: str, action: str, entry_price: float, shares: int,
        atr: float, regime: str, confidence: float, signal: Dict,
        day_str: str, pre_trade_result: Optional[Dict] = None,
        stop_profit: Optional[Dict] = None,
    ):
        """Execute a trade entry and record it."""
        # Stop/profit from risk manager or defaults
        if stop_profit:
            stop_loss = stop_profit.get("stop_loss", 0)
            take_profit = stop_profit.get("take_profit", 0)
        else:
            if action == "BUY":
                stop_loss = entry_price - 2 * atr
                take_profit = entry_price + 4 * atr
            else:
                stop_loss = entry_price + 2 * atr
                take_profit = entry_price - 4 * atr

        position = {
            "symbol": symbol,
            "action": action,
            "entry_price": entry_price,
            "shares": shares,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "entry_date": day_str,
            "signal_score": signal.get("score", 0),
            "signals": signal.get("signals", {}),
            "regime": regime,
            "atr": atr,
            "prediction_id": (pre_trade_result or {}).get("prediction_id"),
        }

        self._positions[symbol] = position

        trade = {
            "asset": symbol,
            "action": action,
            "strategy_id": "aqtis_multi_indicator",
            "entry_price": entry_price,
            "position_size": shares,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "market_regime": regime,
            "confidence": confidence,
            "timestamp": day_str,
            "signals": signal.get("signals", {}),
            "indicator_signals": signal.get("signals", {}),
            "agent_routed": True,
            "prediction_id": (pre_trade_result or {}).get("prediction_id"),
        }
        self._daily_trades.append(trade)

        self._agent_log.append({
            "day": day_str,
            "agent": "risk_manager",
            "action": "entry",
            "symbol": symbol,
            "detail": f"{action} {shares}sh @{entry_price:.2f}, "
                      f"SL={stop_loss:.2f}, TP={take_profit:.2f}, "
                      f"regime={regime}, conf={confidence:.2f}",
        })

    # ─────────────────────────────────────────────────────────────────
    # ADAPTIVE EXIT (via risk_manager agent)
    # ─────────────────────────────────────────────────────────────────

    def _check_exit_adaptive(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame,
        bar_idx: int, day_str: str,
    ):
        """
        Check exits using risk_manager.should_exit_early() + standard stops.

        Agent-driven adaptive exit that detects:
        - Regime changes
        - Signal reversals
        - Trailing stop violations
        - Standard stop loss / take profit
        - Max hold time (5 days)
        """
        pos = self._positions.get(symbol)
        if not pos:
            return

        current_price = float(df["close"].iloc[bar_idx])
        entry_price = pos["entry_price"]
        action = pos["action"]

        # Standard stop/profit check
        exit_reason = None
        if action == "BUY" and current_price <= pos["stop_loss"]:
            exit_reason = "stop_loss"
        elif action == "SELL" and current_price >= pos["stop_loss"]:
            exit_reason = "stop_loss"
        elif action == "BUY" and current_price >= pos["take_profit"]:
            exit_reason = "take_profit"
        elif action == "SELL" and current_price <= pos["take_profit"]:
            exit_reason = "take_profit"

        # Time-based exit
        entry_date = pd.Timestamp(pos["entry_date"])
        if (pd.Timestamp(day_str) - entry_date).days >= 5:
            exit_reason = "max_hold"

        # Agent-driven adaptive exit (0 tokens)
        if exit_reason is None:
            row = ind.iloc[bar_idx] if bar_idx < len(ind) else {}
            current_indicators = {}
            if hasattr(row, 'to_dict'):
                for col in ind.columns:
                    val = row.get(col, np.nan) if hasattr(row, 'get') else getattr(row, col, np.nan)
                    if pd.notna(val):
                        current_indicators[col] = float(val)

            # Generate current signal score for reversal detection
            current_signal = self._generate_signal(symbol, df, ind, bar_idx)
            current_indicators["_signal_score"] = current_signal.get("score", 0)

            early_exit = self.orchestrator.agents["risk_manager"].execute({
                "action": "should_exit_early",
                "position": pos,
                "current_price": current_price,
                "current_indicators": current_indicators,
            })

            if early_exit.get("should_exit", False):
                exit_reason = f"adaptive:{early_exit.get('reason', 'agent')}"

                self._agent_log.append({
                    "day": day_str,
                    "agent": "risk_manager.should_exit_early",
                    "action": "adaptive_exit",
                    "symbol": symbol,
                    "detail": f"urgency={early_exit.get('urgency', 0):.2f}, "
                              f"reason={early_exit.get('reason', '?')}, "
                              f"regime={early_exit.get('current_regime', '?')}",
                })

        if exit_reason is None:
            return

        self._close_position(symbol, current_price, day_str, exit_reason)

    def _close_position(self, symbol: str, exit_price: float, day_str: str, reason: str):
        """Close a position and record the trade."""
        pos = self._positions.pop(symbol, None)
        if not pos:
            return

        entry_price = pos["entry_price"]
        shares = pos["shares"]
        action = pos["action"]

        if action == "BUY":
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        pnl_pct = pnl / (entry_price * shares) * 100

        # Transaction costs (~0.1% round trip)
        cost = entry_price * shares * 0.001
        pnl -= cost
        self._capital += pnl

        trade_record = {
            "asset": symbol,
            "action": action,
            "strategy_id": "aqtis_multi_indicator",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "position_size": shares,
            "pnl": pnl,
            "pnl_percent": pnl_pct,
            "market_regime": pos.get("regime", "unknown"),
            "exit_reason": reason,
            "timestamp": day_str,
            "entry_date": pos.get("entry_date"),
            "hold_duration_seconds": (
                pd.Timestamp(day_str) - pd.Timestamp(pos["entry_date"])
            ).total_seconds(),
            "signals": pos.get("signals", {}),
            "indicator_signals": pos.get("signals", {}),
            "agent_routed": True,
            "prediction_id": pos.get("prediction_id"),
        }
        self._daily_trades.append(trade_record)

        # Record prediction outcome if we have one
        if pos.get("prediction_id"):
            try:
                self.orchestrator.agents["prediction_tracker"].run({
                    "action": "record_outcome",
                    "prediction_id": pos["prediction_id"],
                    "outcome": {
                        "actual_return": pnl_pct,
                        "actual_hold_seconds": trade_record["hold_duration_seconds"],
                    },
                })
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────
    # BATCH LEARNING (every 10 days)
    # ─────────────────────────────────────────────────────────────────

    def _batch_learning(self, day_str: str, day_idx: int, total_days: int):
        """
        Batch post-mortem + weight adjustment. Uses 1-2 LLM calls.

        Flow:
        1. End-of-day learning (post-mortem agent analyzes recent trades)
        2. Apply daily learnings (orchestrator extracts weight deltas)
        3. Mutate _indicator_weights with the deltas
        """
        logger.info(f"[Learn] Batch learning on day {day_idx + 1}")

        # 1. End-of-day learning via orchestrator (triggers post-mortem)
        if self._can_use_llm():
            try:
                learning_result = self.orchestrator.end_of_day_learning(day_str)
                self._use_llm_call()

                self._agent_log.append({
                    "day": day_str,
                    "agent": "orchestrator.end_of_day_learning",
                    "action": "batch_postmortem",
                    "detail": json.dumps({
                        k: v for k, v in learning_result.items()
                        if k != "timestamp"
                    }, default=str)[:200],
                })
            except Exception as e:
                logger.warning(f"End-of-day learning failed: {e}")

        # 2. Apply daily learnings → weight deltas (0 tokens)
        try:
            learnings = self.orchestrator.apply_daily_learnings()
            weight_adj = learnings.get("weight_adjustments", {})
            threshold_adj = learnings.get("threshold_adjustment", 0)
            reasoning = learnings.get("reasoning", [])

            if weight_adj:
                applied = 0
                for ind_name, delta in weight_adj.items():
                    if ind_name in self._indicator_weights:
                        old = self._indicator_weights[ind_name]
                        self._indicator_weights[ind_name] = max(
                            -0.5, min(1.0, old + delta)
                        )
                        applied += 1

                self._weight_mutations.append({
                    "day": day_str,
                    "day_idx": day_idx,
                    "adjustments": weight_adj,
                    "threshold_adj": threshold_adj,
                    "reasoning": reasoning,
                })

                self._agent_log.append({
                    "day": day_str,
                    "agent": "orchestrator.apply_daily_learnings",
                    "action": "mutate_weights",
                    "detail": f"Adjusted {applied} weights, "
                              f"threshold_adj={threshold_adj:+.3f}, "
                              f"reasons: {'; '.join(reasoning[:2])}",
                })

                logger.info(
                    f"[Learn] Applied {applied} weight mutations, "
                    f"threshold adj={threshold_adj:+.3f}"
                )

            if threshold_adj != 0:
                old_t = self._entry_threshold
                self._entry_threshold = max(0.05, min(0.40, old_t + threshold_adj))
                logger.info(
                    f"[Learn] Entry threshold: {old_t:.3f} -> {self._entry_threshold:.3f}"
                )

        except Exception as e:
            logger.warning(f"Apply daily learnings failed: {e}")

        self._trades_since_learning = 0

    # ─────────────────────────────────────────────────────────────────
    # PERIODIC REVIEW (every 3 days)
    # ─────────────────────────────────────────────────────────────────

    def _periodic_review(
        self, day_str: str, day_idx: int, total_days: int, starting_capital: float,
    ):
        """
        Periodic review every 3 days. Uses 3-4 LLM calls.

        All 6 agents participate:
        - prediction_tracker: degradation detection + LLM calibration analysis
        - backtester: rolling backtest + LLM interpretation
        - risk_manager: LLM portfolio risk assessment
        - post_mortem: weekly review (LLM)
        - strategy_generator + researcher: via orchestrator context
        """
        if not self._can_use_llm():
            return

        try:
            metrics = self._calculate_metrics(starting_capital)

            # ── 1. Prediction Tracker: degradation + LLM calibration ──
            degradation = self.orchestrator.agents["prediction_tracker"].run({
                "action": "detect_degradation",
            })
            alerts = degradation.get("degradation_alerts", [])

            if alerts:
                self._agent_log.append({
                    "day": day_str,
                    "agent": "prediction_tracker",
                    "action": "degradation_check",
                    "detail": f"ALERTS: {', '.join(str(a) for a in alerts[:3])}",
                })
                logger.warning(f"[Review] Degradation alerts: {alerts}")

            # LLM calibration analysis (prediction_tracker uses Gemini)
            if self._can_use_llm():
                cal_result = self.orchestrator.agents["prediction_tracker"].run({
                    "action": "llm_calibration_analysis",
                    "metrics": metrics,
                    "regime_breakdown": metrics.get("regime_breakdown", {}),
                })
                self._use_llm_call()

                # Apply confidence scaling if recommended
                scaling = cal_result.get("confidence_scaling", 1.0)
                if scaling != 1.0 and isinstance(scaling, (int, float)):
                    self._confidence_scaling = max(0.5, min(1.5, scaling))

                self._agent_log.append({
                    "day": day_str,
                    "agent": "prediction_tracker.llm_calibration",
                    "action": "calibration_analysis",
                    "detail": f"scaling={scaling:.2f}, "
                              f"{cal_result.get('reasoning', '')[:120]}",
                })

            # ── 2. Backtester: rolling backtest + LLM interpretation ──
            bt_result = self.orchestrator.agents["backtester"].run({
                "action": "rolling",
                "strategy": {"strategy_id": "aqtis_multi_indicator"},
            })
            best_bt_result = bt_result
            if bt_result.get("degradation_detected"):
                self._agent_log.append({
                    "day": day_str,
                    "agent": "backtester",
                    "action": "rolling_backtest",
                    "detail": f"Degradation detected: "
                              f"sharpe={bt_result.get('recent_sharpe', 0):.2f}",
                })

            # LLM backtest interpretation (backtester uses Gemini)
            if self._can_use_llm() and best_bt_result:
                bt_interp = self.orchestrator.agents["backtester"].run({
                    "action": "llm_interpret",
                    "backtest_results": best_bt_result,
                    "strategy": {"strategy_id": "aqtis_multi_indicator"},
                    "current_weights": dict(self._indicator_weights),
                })
                self._use_llm_call()

                # Apply weight adjustments from backtester
                bt_adj = bt_interp.get("weight_adjustments", {})
                if bt_adj:
                    applied = 0
                    for ind_name, delta in bt_adj.items():
                        if ind_name in self._indicator_weights:
                            old = self._indicator_weights[ind_name]
                            self._indicator_weights[ind_name] = max(
                                -0.5, min(1.0, old + delta)
                            )
                            applied += 1
                    if applied:
                        self._weight_mutations.append({
                            "day": day_str,
                            "source": "backtester_llm",
                            "adjustments": bt_adj,
                            "reasoning": [bt_interp.get("reasoning", "backtester LLM")],
                        })

                self._agent_log.append({
                    "day": day_str,
                    "agent": "backtester.llm_interpret",
                    "action": "backtest_interpretation",
                    "detail": f"adj={len(bt_adj)} weights, "
                              f"{bt_interp.get('reasoning', '')[:120]}",
                })

            # ── 3. Risk Manager: LLM portfolio risk assessment ──
            if self._can_use_llm():
                positions_list = [
                    {**pos, "symbol": sym, "unrealized_pnl": 0}
                    for sym, pos in self._positions.items()
                ]
                risk_result = self.orchestrator.agents["risk_manager"].run({
                    "action": "llm_risk_assessment",
                    "positions": positions_list,
                    "portfolio_value": self._capital,
                    "metrics": metrics,
                    "current_regime": self._get_dominant_regime(),
                })
                self._use_llm_call()

                self._agent_log.append({
                    "day": day_str,
                    "agent": "risk_manager.llm_risk_assessment",
                    "action": "portfolio_risk_review",
                    "detail": f"{risk_result.get('reasoning', '')[:150]}",
                })

            # ── 4. Post-mortem: periodic review (LLM) ──
            if self._can_use_llm():
                review = self.orchestrator.agents["post_mortem"].run({
                    "action": "weekly_review",
                })
                self._use_llm_call()

                self._agent_log.append({
                    "day": day_str,
                    "agent": "post_mortem",
                    "action": "periodic_review",
                    "detail": f"Review completed, day {day_idx + 1}/{total_days}",
                })

        except Exception as e:
            logger.warning(f"Periodic review failed: {e}")

    # ─────────────────────────────────────────────────────────────────
    # MID-RUN STRATEGY REVIEW (1 LLM call)
    # ─────────────────────────────────────────────────────────────────

    def _mid_run_review(self, starting_capital: float):
        """Single LLM call at midpoint to evaluate and adjust strategy."""
        if not self._can_use_llm():
            return

        metrics = self._calculate_metrics(starting_capital)

        try:
            review = self.orchestrator.mid_run_strategy_review(
                metrics=metrics,
                current_weights=dict(self._indicator_weights),
            )
            self._use_llm_call()

            if review.get("weight_adjustments"):
                for k, v in review["weight_adjustments"].items():
                    if k in self._indicator_weights:
                        old = self._indicator_weights[k]
                        self._indicator_weights[k] = max(-0.5, min(1.0, old + v))

                self._weight_mutations.append({
                    "day": "mid_run",
                    "adjustments": review["weight_adjustments"],
                    "reasoning": [review.get("summary", "mid-run review")],
                })

            if review.get("entry_threshold_adjustment", 0) != 0:
                adj = review["entry_threshold_adjustment"]
                self._entry_threshold = max(0.05, min(0.40, self._entry_threshold + adj))

            self._agent_log.append({
                "day": "mid_run",
                "agent": "orchestrator.mid_run_strategy_review",
                "action": "mid_run_review",
                "detail": review.get("summary", "review completed")[:200],
            })

            logger.info(f"[MidRun] {review.get('summary', 'Review completed')[:100]}")

        except Exception as e:
            logger.warning(f"Mid-run review failed: {e}")

    # ─────────────────────────────────────────────────────────────────
    # FINALIZE RUN
    # ─────────────────────────────────────────────────────────────────

    def _finalize_run(
        self, market_data: Dict, sim_dates: List, fetched_symbols: List[str],
        starting_capital: float,
    ) -> Dict:
        """Close positions, calculate metrics, record snapshot, end-of-run review."""
        # Close all remaining positions
        if self._positions and sim_dates:
            self._close_all_positions(market_data, sim_dates[-1])

        # Calculate final metrics
        metrics = self._calculate_metrics(starting_capital)
        total_pnl = sum(self._daily_pnl)
        total_return = (total_pnl / starting_capital) * 100

        # Record run completion
        self.orchestrator.end_simulation_run(
            self._run_id, total_pnl, total_return,
            notes=f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
                  f"WR={metrics.get('win_rate', 0):.1%}, "
                  f"engine=paper_trader, mutations={len(self._weight_mutations)}",
        )

        # Record strategy snapshot with CURRENT (mutated) weights
        self.orchestrator.record_strategy_snapshot(
            strategy_id="aqtis_multi_indicator",
            run_number=self._run_number,
            weights=self._indicator_weights,
            sharpe=metrics.get("sharpe_ratio", 0),
            win_rate=metrics.get("win_rate", 0) * 100,
            net_pnl=total_pnl,
            total_trades=metrics.get("total_trades", 0),
            max_drawdown=metrics.get("max_drawdown", 0),
            profit_factor=metrics.get("profit_factor", 0),
            extra={
                "entry_threshold": self._entry_threshold,
                "weight_mutations": len(self._weight_mutations),
            },
        )

        # End-of-run LLM review
        run_review = None
        if self._can_use_llm() and self._all_trades:
            try:
                cross_run = self.orchestrator.get_cross_run_summary()
                self._use_llm_call()
                run_review = {
                    "cross_run_summary": cross_run,
                    "metrics": metrics,
                }
            except Exception:
                pass

        result = {
            "run_number": self._run_number,
            "run_id": self._run_id,
            "engine": "paper_trader",
            "symbols": fetched_symbols,
            "days_simulated": len(sim_dates),
            "date_range": {
                "start": sim_dates[0].strftime("%Y-%m-%d") if sim_dates else "",
                "end": sim_dates[-1].strftime("%Y-%m-%d") if sim_dates else "",
            },
            "capital": {
                "initial": starting_capital,
                "final": starting_capital + total_pnl,
                "pnl": total_pnl,
                "return_pct": total_return,
            },
            "metrics": metrics,
            "trade_count": len(self._all_trades),
            "trades": self._all_trades,
            "equity_curve": self._equity_curve,
            "llm_usage": {
                "calls_used": self._llm_calls_used,
                "budget": self.llm_budget,
                "remaining": self.llm_budget - self._llm_calls_used,
            },
            "learning": {
                "weight_mutations": len(self._weight_mutations),
                "mutations_log": self._weight_mutations,
                "final_entry_threshold": self._entry_threshold,
                "final_confidence_scaling": self._confidence_scaling,
                "final_weights_sample": {
                    k: round(v, 4) for k, v in
                    sorted(self._indicator_weights.items(), key=lambda x: -abs(x[1]))[:10]
                },
            },
            "agent_log": self._agent_log,
            "run_review": run_review,
        }

        logger.info(
            f"[PaperTrader] Run #{self._run_number} complete: "
            f"P&L={total_pnl:+,.0f} ({total_return:+.2f}%), "
            f"Trades={len(self._all_trades)}, "
            f"Sharpe={metrics.get('sharpe_ratio', 0):.2f}, "
            f"LLM={self._llm_calls_used}/{self.llm_budget}, "
            f"Mutations={len(self._weight_mutations)}"
        )

        return result

    # ─────────────────────────────────────────────────────────────────
    # DATA FETCHING (reused from SimulationRunner)
    # ─────────────────────────────────────────────────────────────────

    def _fetch_data(self, symbols: List[str], days: int) -> Dict[str, pd.DataFrame]:
        """Fetch historical data with buffer for indicator warmup."""
        total_days = days + 30
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=int(total_days * 1.5))).strftime("%Y-%m-%d")

        result = {}
        for symbol in symbols:
            try:
                df = self.data_provider.get_historical(
                    symbol, start_date=start_date, end_date=end_date
                )
                if df is not None and len(df) >= 40:
                    result[symbol] = df
                else:
                    logger.warning(f"Insufficient data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        return result

    # ─────────────────────────────────────────────────────────────────
    # INDICATOR COMPUTATION (reused from SimulationRunner)
    # ─────────────────────────────────────────────────────────────────

    def _compute_indicators(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Compute technical indicators for all symbols. No LLM calls."""
        indicators = {}

        if self._indicator_calculator:
            for symbol, df in market_data.items():
                try:
                    raw = self._indicator_calculator.calculate_all(df, normalize=True)
                    renamed = self._indicator_calculator.rename_to_dna_names(raw)
                    indicators[symbol] = renamed
                except Exception as e:
                    logger.warning(f"Full indicator calc failed for {symbol}: {e}")
                    try:
                        indicators[symbol] = self._calc_indicators(df)
                    except Exception:
                        pass
        else:
            for symbol, df in market_data.items():
                try:
                    indicators[symbol] = self._calc_indicators(df)
                except Exception as e:
                    logger.warning(f"Indicator calc failed for {symbol}: {e}")

        return indicators

    def _calc_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate core technical indicators from OHLCV data (fallback)."""
        ind = pd.DataFrame(index=df.index)
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"] if "volume" in df.columns else pd.Series(0, index=df.index)

        ind["sma_10"] = close.rolling(10).mean()
        ind["sma_20"] = close.rolling(20).mean()
        ind["sma_50"] = close.rolling(50).mean()
        ind["ema_12"] = close.ewm(span=12).mean()
        ind["ema_26"] = close.ewm(span=26).mean()

        ind["macd"] = ind["ema_12"] - ind["ema_26"]
        ind["macd_signal"] = ind["macd"].ewm(span=9).mean()
        ind["macd_hist"] = ind["macd"] - ind["macd_signal"]

        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        ind["rsi"] = 100 - (100 / (1 + rs))

        bb_sma = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        ind["bb_upper"] = bb_sma + 2 * bb_std
        ind["bb_lower"] = bb_sma - 2 * bb_std
        ind["bb_pct"] = (close - ind["bb_lower"]) / (ind["bb_upper"] - ind["bb_lower"] + 1e-10)
        ind["bb_width"] = (ind["bb_upper"] - ind["bb_lower"]) / (bb_sma + 1e-10)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        ind["atr"] = tr.rolling(14).mean()

        ind["volume_sma"] = volume.rolling(20).mean()
        ind["volume_ratio"] = volume / (ind["volume_sma"] + 1e-10)

        ind["returns_1d"] = close.pct_change()
        ind["returns_5d"] = close.pct_change(5)
        ind["returns_20d"] = close.pct_change(20)
        ind["volatility_20d"] = ind["returns_1d"].rolling(20).std() * np.sqrt(252)

        plus_dm = (high.diff()).clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        ind["plus_di"] = 100 * (plus_dm.rolling(14).mean() / (ind["atr"] + 1e-10))
        ind["minus_di"] = 100 * (minus_dm.rolling(14).mean() / (ind["atr"] + 1e-10))
        dx = 100 * (ind["plus_di"] - ind["minus_di"]).abs() / (ind["plus_di"] + ind["minus_di"] + 1e-10)
        ind["adx"] = dx.rolling(14).mean()

        hl2 = (high + low) / 2
        atr_st = tr.rolling(10).mean()
        upper_band = hl2 + 3 * atr_st
        lower_band = hl2 - 3 * atr_st
        supertrend = pd.Series(0.0, index=df.index)
        direction = pd.Series(1, index=df.index)
        for i in range(1, len(df)):
            if close.iloc[i] > upper_band.iloc[i - 1]:
                direction.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
        ind["supertrend"] = supertrend
        ind["supertrend_dir"] = direction

        return ind

    # ─────────────────────────────────────────────────────────────────
    # SIGNAL GENERATION (0 tokens, uses mutable weights)
    # ─────────────────────────────────────────────────────────────────

    def _generate_signal(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame, bar_idx: int,
    ) -> Dict:
        """Generate signal using mutable _indicator_weights."""
        if bar_idx >= len(ind):
            return {"score": 0, "signals": {}}

        row = ind.iloc[bar_idx]

        if self._indicator_calculator and HAS_FULL_INDICATORS:
            return self._generate_signal_full(symbol, df, ind, bar_idx, row)

        return self._generate_signal_basic(symbol, df, ind, bar_idx, row)

    def _generate_signal_full(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame,
        bar_idx: int, row,
    ) -> Dict:
        """Full 87-indicator signal generation using MUTABLE weights."""
        signals = {}
        for col in ind.columns:
            val = row.get(col, np.nan)
            if pd.notna(val) and not np.isnan(val):
                signals[col] = float(val)

        if not signals:
            return {"score": 0, "signals": {}}

        # Use MUTABLE weights (not static defaults)
        row_df = ind.iloc[bar_idx:bar_idx + 1]
        score = SignalAggregator.weighted_at_timestamp(
            row_df, self._indicator_weights, row_df.index[0]
        )

        close_price = float(df["close"].iloc[bar_idx])
        atr = close_price * 0.02

        rsi = 50.0
        for rsi_col in ["RSI_14", "RSI_7"]:
            if rsi_col in signals:
                rsi = (signals[rsi_col] + 1) * 50
                break

        top_signals = sorted(
            [(k, v * self._indicator_weights.get(k, 0))
             for k, v in signals.items()
             if k in self._indicator_weights],
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]

        return {
            "score": float(np.clip(score, -1, 1)),
            "signals": {k: v for k, v in top_signals},
            "all_signals": signals,
            "symbol": symbol,
            "close": close_price,
            "atr": atr,
            "rsi": rsi,
            "regime": self._classify_regime_from_indicators(signals),
            "num_indicators": len(signals),
        }

    def _classify_regime_from_indicators(self, signals: Dict) -> str:
        """Classify regime from normalized indicator values."""
        adx = signals.get("ADX_14", 0)
        st = signals.get("SUPERTREND_10_2", signals.get("SUPERTREND_7_3", 0))
        vol = signals.get("NATR_14", signals.get("ZSCORE_20", 0))

        if abs(vol) > 0.6:
            return "high_vol"
        elif abs(vol) < 0.2 and abs(adx) < 0.2:
            return "low_vol"
        elif adx > 0.3 and st > 0:
            return "trending_up"
        elif adx > 0.3 and st < 0:
            return "trending_down"
        else:
            return "mean_reverting"

    def _generate_signal_basic(
        self, symbol: str, df: pd.DataFrame, ind: pd.DataFrame,
        bar_idx: int, row,
    ) -> Dict:
        """Fallback basic 7-indicator signal generation."""
        signals = {}
        weights = {}

        rsi = row.get("rsi", 50)
        if not np.isnan(rsi):
            if rsi < 30: signals["rsi"] = 1.0
            elif rsi < 40: signals["rsi"] = 0.5
            elif rsi > 70: signals["rsi"] = -1.0
            elif rsi > 60: signals["rsi"] = -0.5
            else: signals["rsi"] = 0.0
            weights["rsi"] = 0.20

        macd_hist = row.get("macd_hist", 0)
        if not np.isnan(macd_hist):
            price = df["close"].iloc[bar_idx]
            signals["macd"] = np.clip(macd_hist / (price * 0.01 + 1e-10), -1, 1)
            weights["macd"] = 0.20

        bb_pct = row.get("bb_pct", 0.5)
        if not np.isnan(bb_pct):
            if bb_pct < 0.05: signals["bollinger"] = 1.0
            elif bb_pct < 0.2: signals["bollinger"] = 0.6
            elif bb_pct > 0.95: signals["bollinger"] = -1.0
            elif bb_pct > 0.8: signals["bollinger"] = -0.6
            else: signals["bollinger"] = 0.0
            weights["bollinger"] = 0.15

        st_dir = row.get("supertrend_dir", 0)
        if not np.isnan(st_dir):
            signals["supertrend"] = float(st_dir)
            weights["supertrend"] = 0.15

        adx = row.get("adx", 0)
        plus_di = row.get("plus_di", 0)
        if not np.isnan(adx) and not np.isnan(plus_di):
            signals["adx_trend"] = (1.0 if plus_di > row.get("minus_di", 0) else -1.0) if adx > 25 else 0.0
            weights["adx_trend"] = 0.10

        vol_ratio = row.get("volume_ratio", 1)
        if not np.isnan(vol_ratio):
            signals["volume"] = min(vol_ratio - 1.0, 0.5) * 0.4 if vol_ratio > 1.5 else 0.0
            weights["volume"] = 0.10

        ret_5d = row.get("returns_5d", 0)
        if not np.isnan(ret_5d):
            signals["momentum_5d"] = np.clip(ret_5d * 10, -1, 1)
            weights["momentum_5d"] = 0.10

        if not signals:
            return {"score": 0, "signals": {}}

        weighted_sum = sum(signals[k] * weights.get(k, 0.1) for k in signals)
        total_weight = sum(weights.get(k, 0.1) for k in signals)
        score = weighted_sum / (total_weight + 1e-10)

        return {
            "score": np.clip(score, -1, 1),
            "signals": signals,
            "symbol": symbol,
            "close": float(df["close"].iloc[bar_idx]),
            "atr": float(row.get("atr", 0)) if not np.isnan(row.get("atr", 0)) else 0,
            "rsi": float(rsi) if not np.isnan(rsi) else 50,
            "regime": self._classify_regime(row),
        }

    def _classify_regime(self, row) -> str:
        """Simple rule-based regime classification."""
        vol = row.get("volatility_20d", 0)
        adx = row.get("adx", 0)
        ret_20d = row.get("returns_20d", 0)

        if np.isnan(vol) or np.isnan(adx):
            return "unknown"

        if vol > 0.4:
            return "high_vol"
        elif vol < 0.15:
            return "low_vol"
        elif adx > 25 and ret_20d > 0.02:
            return "trending_up"
        elif adx > 25 and ret_20d < -0.02:
            return "trending_down"
        else:
            return "mean_reverting"

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _get_dominant_regime(self) -> str:
        """Get the most common regime across current positions."""
        if not self._positions:
            return "unknown"
        regimes = [p.get("regime", "unknown") for p in self._positions.values()]
        from collections import Counter
        most_common = Counter(regimes).most_common(1)
        return most_common[0][0] if most_common else "unknown"

    # ─────────────────────────────────────────────────────────────────
    # PORTFOLIO TRACKING
    # ─────────────────────────────────────────────────────────────────

    def _close_all_positions(self, market_data: Dict, last_date):
        """Close all remaining positions at last available price."""
        day_str = last_date.strftime("%Y-%m-%d")
        for symbol in list(self._positions.keys()):
            df = market_data.get(symbol)
            if df is not None and len(df) > 0:
                last_price = float(df["close"].iloc[-1])
                self._close_position(symbol, last_price, day_str, "end_of_sim")

    def _mark_to_market(self, market_data: Dict, sim_date) -> float:
        """Calculate realized P&L for the day."""
        realized = 0.0
        for trade in self._daily_trades:
            if trade.get("pnl") is not None:
                realized += trade["pnl"]
        return realized

    def _unrealized_pnl(self, market_data: Dict, sim_date) -> float:
        """Calculate unrealized P&L across open positions."""
        unrealized = 0.0
        for symbol, pos in self._positions.items():
            df = market_data.get(symbol)
            if df is None:
                continue
            mask = df.index.normalize() <= sim_date
            if not mask.any():
                continue
            current_price = float(df.loc[mask, "close"].iloc[-1])
            entry_price = pos["entry_price"]
            shares = pos["shares"]
            if pos["action"] == "BUY":
                unrealized += (current_price - entry_price) * shares
            else:
                unrealized += (entry_price - current_price) * shares
        return unrealized

    # ─────────────────────────────────────────────────────────────────
    # LLM BUDGET
    # ─────────────────────────────────────────────────────────────────

    def _can_use_llm(self) -> bool:
        return self._llm_calls_used < self.llm_budget

    def _use_llm_call(self):
        self._llm_calls_used += 1

    # ─────────────────────────────────────────────────────────────────
    # METRICS
    # ─────────────────────────────────────────────────────────────────

    def _calculate_metrics(self, starting_capital: float) -> Dict:
        """Calculate comprehensive performance metrics."""
        completed = [t for t in self._all_trades if t.get("pnl") is not None]

        if not completed:
            return {
                "total_trades": 0, "win_rate": 0, "sharpe_ratio": 0,
                "total_pnl": 0, "max_drawdown": 0,
            }

        pnls = [t.get("pnl", 0) for t in completed]
        wins = sum(1 for p in pnls if p > 0)
        losses = len(pnls) - wins

        daily_returns = np.array(self._daily_pnl) / starting_capital
        sharpe = 0.0
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252))

        equity = np.cumsum([starting_capital] + list(np.array(self._daily_pnl)))
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / (peak + 1e-10)
        max_dd = float(np.min(drawdown))

        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / (gross_loss + 1e-10)

        avg_win = gross_profit / wins if wins > 0 else 0
        avg_loss = gross_loss / losses if losses > 0 else 0

        regime_stats = {}
        for t in completed:
            r = t.get("market_regime", "unknown")
            if r not in regime_stats:
                regime_stats[r] = {"trades": 0, "wins": 0, "pnl": 0}
            regime_stats[r]["trades"] += 1
            if t.get("pnl", 0) > 0:
                regime_stats[r]["wins"] += 1
            regime_stats[r]["pnl"] += t.get("pnl", 0)

        # Exit reason breakdown
        exit_reasons = {}
        for t in completed:
            reason = t.get("exit_reason", "unknown")
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        return {
            "total_trades": len(completed),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(completed) if completed else 0,
            "total_pnl": sum(pnls),
            "avg_pnl": float(np.mean(pnls)),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "regime_breakdown": regime_stats,
            "exit_reasons": exit_reasons,
        }
