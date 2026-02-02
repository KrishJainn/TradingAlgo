"""
AQTIS Backtesting Agent.

Runs backtests and shadow trading simulations to validate strategies.
Wraps trading_evolution's BacktestEngine.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseAgent

logger = logging.getLogger(__name__)

_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class BacktestingAgent(BaseAgent):
    """
    Backtesting agent for strategy validation.

    Capabilities:
    - Instant mini-backtests before trade execution
    - Rolling window backtests
    - Shadow testing of parameter variants
    """

    def __init__(self, memory, llm=None, data_provider=None):
        super().__init__(name="backtester", memory=memory, llm=llm)
        self.data_provider = data_provider
        self._backtest_engine = None

    @property
    def backtest_engine(self):
        if self._backtest_engine is None:
            try:
                from trading_evolution.backtest.engine import BacktestEngine
                self._backtest_engine = BacktestEngine
                logger.info("Using trading_evolution BacktestEngine")
            except ImportError:
                logger.warning("trading_evolution BacktestEngine not available")
        return self._backtest_engine

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute backtesting action."""
        action = context.get("action", "instant")

        if action == "instant":
            return self.instant_backtest(
                context.get("strategy", {}),
                context.get("signal", {}),
                context.get("lookback_days", 30),
            )
        elif action == "rolling":
            return self.rolling_window_backtest(
                context.get("strategy", {}),
                context.get("window_days", 90),
            )
        elif action == "shadow":
            return self.shadow_test_variants(
                context.get("strategy", {}),
                context.get("variants", []),
                context.get("signal", {}),
            )
        else:
            return {"error": f"Unknown action: {action}"}

    # ─────────────────────────────────────────────────────────────────
    # INSTANT BACKTEST
    # ─────────────────────────────────────────────────────────────────

    def instant_backtest(
        self,
        strategy: Dict,
        current_signal: Dict,
        lookback_days: int = 30,
    ) -> Dict:
        """
        Quick backtest using similar historical trades.

        Args:
            strategy: Strategy specification.
            current_signal: Current market signal/setup.
            lookback_days: Days of history to consider.

        Returns:
            Expected performance metrics.
        """
        # Find similar historical setups from memory
        similar_trades = self.memory.get_similar_trades(current_signal, top_k=50)

        if len(similar_trades) < 5:
            return {
                "expected_return": 0.0,
                "win_probability": 0.5,
                "confidence": 0.0,
                "sample_size": len(similar_trades),
                "insufficient_data": True,
            }

        # Calculate statistics from similar trades
        pnls = [t.get("pnl_percent", 0) or 0 for t in similar_trades]
        wins = sum(1 for p in pnls if p > 0)

        win_rate = wins / len(pnls)
        avg_return = float(np.mean(pnls))
        std_return = float(np.std(pnls)) if len(pnls) > 1 else 0.0

        # Confidence based on sample size and consistency
        sample_confidence = min(len(similar_trades) / 50, 1.0)
        consistency = 1 - (std_return / (abs(avg_return) + 1e-6))
        confidence = max(0.0, min(1.0, sample_confidence * max(0, consistency)))

        # Holding period estimate
        hold_durations = [
            t.get("hold_duration_seconds", 0) or 0 for t in similar_trades
        ]
        avg_hold = float(np.mean(hold_durations)) if hold_durations else 0

        return {
            "expected_return": avg_return,
            "win_probability": win_rate,
            "avg_return": avg_return,
            "std_return": std_return,
            "expected_hold_seconds": avg_hold,
            "confidence": confidence,
            "sample_size": len(similar_trades),
            "strategy_id": strategy.get("strategy_id", "unknown"),
        }

    # ─────────────────────────────────────────────────────────────────
    # ROLLING WINDOW BACKTEST
    # ─────────────────────────────────────────────────────────────────

    def rolling_window_backtest(
        self,
        strategy: Dict,
        window_days: int = 90,
    ) -> Dict:
        """
        Rolling window backtest to detect performance degradation.

        Returns performance timeline and degradation detection.
        """
        strategy_id = strategy.get("strategy_id", "")
        all_trades = self.memory.get_trades(strategy_id=strategy_id, limit=500)

        if len(all_trades) < 20:
            return {
                "performance_timeline": [],
                "degradation_detected": False,
                "recommendation": "Insufficient data",
                "total_trades": len(all_trades),
            }

        # Sort by timestamp
        all_trades.sort(key=lambda t: t.get("timestamp", ""))

        # Calculate rolling performance
        window_size = min(window_days, len(all_trades) // 3)
        performance = []

        for i in range(window_size, len(all_trades)):
            window = all_trades[i - window_size: i]
            pnls = [t.get("pnl_percent", 0) or 0 for t in window]
            wins = sum(1 for p in pnls if p > 0)

            returns_arr = np.array(pnls)
            sharpe = 0.0
            if len(returns_arr) > 1 and np.std(returns_arr) > 0:
                sharpe = float(np.mean(returns_arr) / np.std(returns_arr) * np.sqrt(252))

            performance.append({
                "index": i,
                "timestamp": all_trades[i].get("timestamp"),
                "sharpe": sharpe,
                "win_rate": wins / len(pnls) if pnls else 0,
                "avg_return": float(np.mean(pnls)),
            })

        # Check for degradation
        if len(performance) > 10:
            recent = performance[-10:]
            historical = performance[:-10]

            recent_sharpe = np.mean([p["sharpe"] for p in recent])
            hist_sharpe = np.mean([p["sharpe"] for p in historical])

            degradation = hist_sharpe > 0 and recent_sharpe < hist_sharpe * 0.7
        else:
            degradation = False
            recent_sharpe = 0
            hist_sharpe = 0

        return {
            "performance_timeline": performance[-30:],  # Last 30 windows
            "recent_sharpe": float(recent_sharpe) if performance else 0,
            "historical_sharpe": float(hist_sharpe) if performance else 0,
            "degradation_detected": degradation,
            "recommendation": "Pause strategy" if degradation else "Continue",
            "total_trades": len(all_trades),
        }

    # ─────────────────────────────────────────────────────────────────
    # SHADOW TESTING
    # ─────────────────────────────────────────────────────────────────

    def shadow_test_variants(
        self,
        base_strategy: Dict,
        variants: List[Dict],
        trade_signal: Dict,
    ) -> Dict:
        """
        Shadow test parameter variants against the base strategy.

        Simulates what would happen with different parameter settings.
        """
        results = {}

        # Test base strategy
        base_result = self.instant_backtest(base_strategy, trade_signal)
        results["base"] = base_result

        # Test each variant
        for i, variant in enumerate(variants):
            variant_result = self.instant_backtest(variant, trade_signal)
            results[f"variant_{i}"] = variant_result

        # Rank by expected return * confidence
        ranked = sorted(
            results.items(),
            key=lambda x: x[1].get("expected_return", 0) * x[1].get("confidence", 0),
            reverse=True,
        )

        best_name, best_result = ranked[0]
        return {
            "variant_rankings": [(name, res) for name, res in ranked],
            "best_variant": best_name,
            "best_result": best_result,
            "base_result": base_result,
            "improvement_over_base": (
                best_result.get("expected_return", 0) - base_result.get("expected_return", 0)
            ),
        }
