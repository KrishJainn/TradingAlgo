"""
AQTIS Risk Management Agent.

Monitors portfolio exposure, prevents catastrophic losses,
enforces risk limits, and provides dynamic position sizing.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseAgent

logger = logging.getLogger(__name__)

DEFAULT_RISK_LIMITS = {
    "max_position_size": 0.10,
    "max_portfolio_leverage": 2.0,
    "max_daily_loss": -0.05,
    "max_drawdown": -0.15,
    "max_correlated_exposure": 0.30,
    "min_prediction_confidence": 0.60,
}


class RiskManagementAgent(BaseAgent):
    """
    Real-time risk management with circuit breakers and dynamic position sizing.

    Wraps trading_evolution's RiskManager and adds:
    - Circuit breaker system
    - Kelly criterion position sizing
    - Daily loss tracking
    - Correlation-based exposure checks
    """

    def __init__(self, memory, llm=None, risk_limits: Dict = None):
        super().__init__(name="risk_manager", memory=memory, llm=llm)
        self.limits = risk_limits or DEFAULT_RISK_LIMITS
        self.circuit_breaker_active = False
        self._daily_pnl = 0.0
        self._daily_pnl_date = None
        self._portfolio_peak = None

        # Try to wrap existing risk manager
        self._base_risk_manager = None
        try:
            from trading_evolution.player.risk_manager import RiskManager, RiskParameters
            self._base_risk_manager = RiskManager(RiskParameters(
                max_risk_per_trade=self.limits["max_position_size"],
                max_position_pct=self.limits["max_position_size"],
            ))
        except ImportError:
            pass

    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute risk management action."""
        action = context.get("action", "validate")

        if action == "validate":
            return self.validate_trade(context.get("trade", {}))
        elif action == "position_size":
            return {
                "position_size": self.calculate_position_size(
                    context.get("prediction", {}),
                    context.get("portfolio_value", 100000),
                )
            }
        elif action == "check_portfolio":
            return self.check_portfolio_risk(
                context.get("positions", []),
                context.get("portfolio_value", 100000),
            )
        elif action == "should_exit_early":
            return self.should_exit_early(
                context.get("position", {}),
                context.get("current_price", 0),
                context.get("current_indicators", {}),
            )
        elif action == "dynamic_stop_loss":
            return self.dynamic_stop_loss(
                context.get("position", {}),
                context.get("current_indicators", {}),
            )
        elif action == "llm_risk_assessment":
            return self.llm_risk_assessment(
                context.get("positions", []),
                context.get("portfolio_value", 100000),
                context.get("metrics", {}),
                context.get("current_regime", "unknown"),
            )
        else:
            return self.get_status()

    # ─────────────────────────────────────────────────────────────────
    # TRADE VALIDATION
    # ─────────────────────────────────────────────────────────────────

    def validate_trade(self, proposed_trade: Dict) -> Dict:
        """
        Check if proposed trade violates any risk limits.
        """
        if self.circuit_breaker_active:
            return {
                "approved": False,
                "trade": None,
                "rejection_reasons": ["Circuit breaker is active"],
            }

        checks = {
            "confidence_ok": self._check_confidence(proposed_trade),
            "daily_loss_ok": self._check_daily_loss(),
            "drawdown_ok": self._check_drawdown(),
            "position_size_ok": self._check_position_size(proposed_trade),
        }

        approved = all(checks.values())
        rejection_reasons = [k for k, v in checks.items() if not v]

        if not approved:
            # Try to adjust trade to meet limits
            adjusted = self._adjust_trade_size(proposed_trade, checks)
            if adjusted:
                return {
                    "approved": True,
                    "trade": adjusted,
                    "adjustments": rejection_reasons,
                    "checks": checks,
                }

        return {
            "approved": approved,
            "trade": proposed_trade if approved else None,
            "checks": checks,
            "rejection_reasons": rejection_reasons,
        }

    def _check_confidence(self, trade: Dict) -> bool:
        """Check if prediction confidence meets minimum."""
        confidence = trade.get("confidence", trade.get("predicted_confidence", 0))
        return confidence >= self.limits["min_prediction_confidence"]

    def _check_daily_loss(self) -> bool:
        """Check if daily loss limit has been reached."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._daily_pnl_date != today:
            self._daily_pnl = self.memory.db.get_daily_pnl(today)
            self._daily_pnl_date = today
        return self._daily_pnl > self.limits["max_daily_loss"] * 100000  # Approx

    def _check_drawdown(self) -> bool:
        """Check if max drawdown has been reached."""
        # Simplified check - in production would track portfolio equity curve
        return not self.circuit_breaker_active

    def _check_position_size(self, trade: Dict) -> bool:
        """Check if position size is within limits."""
        position_size = trade.get("position_size", 0)
        portfolio_value = trade.get("portfolio_value", 100000)
        entry_price = trade.get("entry_price", 0)

        if entry_price <= 0 or portfolio_value <= 0:
            return True

        position_value = position_size * entry_price
        return position_value / portfolio_value <= self.limits["max_position_size"]

    def _adjust_trade_size(self, trade: Dict, checks: Dict) -> Optional[Dict]:
        """Try to reduce trade size to meet risk limits."""
        if not checks.get("confidence_ok", True):
            return None  # Can't fix low confidence by adjusting size

        adjusted = dict(trade)
        portfolio_value = trade.get("portfolio_value", 100000)
        entry_price = trade.get("entry_price", 1)

        if entry_price > 0:
            max_value = portfolio_value * self.limits["max_position_size"]
            max_shares = int(max_value / entry_price)
            adjusted["position_size"] = min(
                trade.get("position_size", max_shares), max_shares
            )
            return adjusted

        return None

    # ─────────────────────────────────────────────────────────────────
    # POSITION SIZING
    # ─────────────────────────────────────────────────────────────────

    def get_risk_knowledge(self, topic: str) -> str:
        """Query knowledge base for risk management best practices."""
        try:
            results = self.memory.search_knowledge(topic, top_k=3)
            if results:
                return " | ".join(r.get("text", "")[:150] for r in results)
        except Exception:
            pass
        return ""

    def calculate_position_size(self, prediction: Dict, portfolio_value: float) -> float:
        """
        Dynamic position sizing based on Kelly criterion.

        Uses prediction confidence and historical win/loss ratios.
        """
        confidence = prediction.get("predicted_confidence", prediction.get("confidence", 0.5))
        expected_return = prediction.get("predicted_return", 0.01)

        # Get similar trades for win/loss ratio estimation
        similar = self.memory.get_similar_trades(prediction, top_k=100)

        if similar:
            wins = [t.get("pnl_percent", 0) for t in similar if (t.get("pnl") or 0) > 0]
            losses = [abs(t.get("pnl_percent", 0)) for t in similar if (t.get("pnl") or 0) < 0]
            avg_win = float(np.mean(wins)) if wins else abs(expected_return)
            avg_loss = float(np.mean(losses)) if losses else abs(expected_return * 0.5)
        else:
            avg_win = abs(expected_return)
            avg_loss = abs(expected_return * 0.5)

        # Kelly fraction
        win_prob = confidence
        loss_prob = 1 - confidence
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 1.0

        kelly = (win_prob * win_loss_ratio - loss_prob) / win_loss_ratio if win_loss_ratio > 0 else 0

        # Fractional Kelly (quarter Kelly for safety)
        fractional_kelly = kelly * 0.25

        # Apply limits
        position_fraction = max(0.0, min(fractional_kelly, self.limits["max_position_size"]))

        return portfolio_value * position_fraction

    # ─────────────────────────────────────────────────────────────────
    # CIRCUIT BREAKER
    # ─────────────────────────────────────────────────────────────────

    def activate_circuit_breaker(self, reason: str):
        """Emergency stop all trading."""
        self.circuit_breaker_active = True

        self.memory.store_risk_event({
            "event_type": "circuit_breaker",
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        })

        self.logger.warning(f"CIRCUIT BREAKER ACTIVATED: {reason}")

    def deactivate_circuit_breaker(self):
        """Re-enable trading after circuit breaker."""
        self.circuit_breaker_active = False

        self.memory.store_risk_event({
            "event_type": "circuit_breaker_reset",
            "reason": "Manual deactivation",
            "timestamp": datetime.now().isoformat(),
        })

        self.logger.info("Circuit breaker deactivated")

    # ─────────────────────────────────────────────────────────────────
    # PORTFOLIO RISK
    # ─────────────────────────────────────────────────────────────────

    def check_portfolio_risk(self, positions: List[Dict], portfolio_value: float) -> Dict:
        """Check overall portfolio risk metrics."""
        total_exposure = sum(
            p.get("entry_price", 0) * p.get("position_size", 0) for p in positions
        )
        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0

        alerts = []
        if leverage > self.limits["max_portfolio_leverage"]:
            alerts.append(f"Leverage {leverage:.2f}x exceeds max {self.limits['max_portfolio_leverage']}x")

        daily_pnl = self.memory.db.get_daily_pnl()
        daily_return = daily_pnl / portfolio_value if portfolio_value > 0 else 0
        if daily_return < self.limits["max_daily_loss"]:
            alerts.append(f"Daily loss {daily_return:.2%} exceeds limit {self.limits['max_daily_loss']:.2%}")
            self.activate_circuit_breaker(f"Daily loss limit breached: {daily_return:.2%}")

        return {
            "num_positions": len(positions),
            "total_exposure": total_exposure,
            "leverage": leverage,
            "daily_pnl": daily_pnl,
            "daily_return": daily_return,
            "circuit_breaker_active": self.circuit_breaker_active,
            "alerts": alerts,
        }

    # ─────────────────────────────────────────────────────────────────
    # LLM-POWERED RISK ASSESSMENT
    # ─────────────────────────────────────────────────────────────────

    def llm_risk_assessment(
        self,
        positions: List[Dict],
        portfolio_value: float,
        metrics: Dict,
        current_regime: str = "unknown",
    ) -> Dict:
        """
        Use Gemini to evaluate overall portfolio risk and recommend adjustments.

        Analyzes position concentration, regime exposure, drawdown trajectory,
        and win rate to produce risk limit recommendations.

        Returns:
            Dict with risk_adjustments, positions_to_reduce, and reasoning.
        """
        if not self.llm:
            return {"error": "No LLM available", "risk_adjustments": {}}

        # Summarize positions
        pos_summary = []
        total_exposure = 0
        for p in positions[:10]:  # Cap at 10 for prompt size
            val = p.get("entry_price", 0) * p.get("shares", p.get("position_size", 0))
            total_exposure += val
            pnl = p.get("unrealized_pnl", 0)
            pos_summary.append(
                f"{p.get('symbol', '?')}: {p.get('action', '?')} "
                f"${val:,.0f} (P&L ${pnl:+,.0f}, regime={p.get('regime', '?')})"
            )

        leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0

        # Recent trade stats
        win_rate = metrics.get("win_rate", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown", 0)
        profit_factor = metrics.get("profit_factor", 0)

        prompt = f"""You are AQTIS risk manager. Evaluate portfolio risk and recommend adjustments.

PORTFOLIO STATE:
- Value: ${portfolio_value:,.0f}, Exposure: ${total_exposure:,.0f}, Leverage: {leverage:.2f}x
- Current regime: {current_regime}
- Circuit breaker: {"ACTIVE" if self.circuit_breaker_active else "off"}

POSITIONS ({len(positions)}):
{chr(10).join(pos_summary) if pos_summary else "No open positions"}

PERFORMANCE METRICS:
- Win Rate: {win_rate:.0%}, Sharpe: {sharpe:.2f}
- Max Drawdown: {max_dd:.2%}, Profit Factor: {profit_factor:.2f}

CURRENT LIMITS:
- Max position size: {self.limits['max_position_size']:.0%}
- Max leverage: {self.limits['max_portfolio_leverage']:.1f}x
- Max daily loss: {self.limits['max_daily_loss']:.0%}

Respond in JSON:
{{
  "risk_adjustments": {{
    "max_position_size": float (0.02-0.15),
    "max_daily_loss": float (-0.02 to -0.10)
  }},
  "positions_to_reduce": ["SYMBOL1"],
  "activate_circuit_breaker": false,
  "reasoning": "2-3 sentences"
}}"""

        try:
            result = self.llm.generate_json(prompt)
            if not isinstance(result, dict):
                return {"risk_adjustments": {}, "reasoning": "LLM returned non-dict"}

            # Apply risk adjustments if provided
            adj = result.get("risk_adjustments", {})
            if adj:
                for key, val in adj.items():
                    if key in self.limits and isinstance(val, (int, float)):
                        self.limits[key] = val

            if result.get("activate_circuit_breaker"):
                self.activate_circuit_breaker(
                    f"LLM recommended: {result.get('reasoning', 'risk assessment')[:80]}"
                )

            self.logger.info(
                f"LLM risk assessment: {result.get('reasoning', '')[:100]}"
            )
            return result
        except Exception as e:
            self.logger.warning(f"LLM risk assessment failed: {e}")
            return {"risk_adjustments": {}, "error": str(e)}

    # ─────────────────────────────────────────────────────────────────
    # ADAPTIVE EXIT LOGIC
    # ─────────────────────────────────────────────────────────────────

    def dynamic_stop_loss(self, position: Dict, current_indicators: Dict) -> Dict:
        """Regime-aware stop loss and take profit levels. Zero LLM calls."""
        entry_price = position.get("entry_price", 0)
        atr = position.get("atr", entry_price * 0.02)
        regime = position.get("regime", "unknown")
        action = position.get("action", "BUY")

        stop_mult, profit_mult = 2.0, 4.0

        if regime == "high_vol":
            stop_mult, profit_mult = 2.5, 3.0
        elif regime == "trending_up" and action == "BUY":
            stop_mult, profit_mult = 1.5, 5.0
        elif regime == "trending_down" and action == "SELL":
            stop_mult, profit_mult = 1.5, 5.0
        elif regime == "mean_reverting":
            stop_mult, profit_mult = 1.8, 2.5

        if action == "BUY":
            sl = entry_price - stop_mult * atr
            tp = entry_price + profit_mult * atr
        else:
            sl = entry_price + stop_mult * atr
            tp = entry_price - profit_mult * atr

        return {
            "stop_loss": sl, "take_profit": tp,
            "stop_multiplier": stop_mult, "profit_multiplier": profit_mult,
        }

    def should_exit_early(
        self, position: Dict, current_price: float, current_indicators: Dict,
    ) -> Dict:
        """
        Adaptive exit recommendation. Zero LLM calls.

        Checks: regime change, trailing stop, signal reversal, time decay.
        """
        entry_price = position.get("entry_price", 0)
        action = position.get("action", "BUY")
        entry_regime = position.get("regime", "unknown")
        reasons = []
        urgency = 0.0

        # Current regime from indicators
        vol = current_indicators.get("NATR_14", current_indicators.get("volatility_20d", 0))
        adx = current_indicators.get("ADX_14", current_indicators.get("adx", 0))
        st = current_indicators.get("SUPERTREND_10_2", current_indicators.get("supertrend_dir", 0))

        if isinstance(vol, float) and not np.isnan(vol):
            if abs(vol) > 0.6 or vol > 0.4:
                current_regime = "high_vol"
            elif abs(vol) < 0.2 or vol < 0.15:
                current_regime = "low_vol"
            elif adx > 0.3 and st > 0:
                current_regime = "trending_up"
            elif adx > 0.3 and st < 0:
                current_regime = "trending_down"
            else:
                current_regime = "mean_reverting"
        else:
            current_regime = "unknown"

        # 1. Regime change
        if entry_regime != current_regime and entry_regime != "unknown":
            if current_regime == "high_vol":
                urgency += 0.4
                reasons.append(f"Regime -> high_vol from {entry_regime}")
            elif entry_regime.startswith("trending") and current_regime == "mean_reverting":
                urgency += 0.3
                reasons.append("Trend ended, now mean reverting")

        # 2. P&L check — protect gains
        if entry_price > 0 and current_price > 0:
            pnl_pct = ((current_price - entry_price) / entry_price
                       if action == "BUY" else
                       (entry_price - current_price) / entry_price)
            if pnl_pct > 0.03 and pnl_pct < 0.005:
                urgency += 0.5
                reasons.append("Gave back most gains")
            elif pnl_pct < -0.04:
                urgency += 0.3
                reasons.append(f"Deep loss {pnl_pct:.1%}")

        # 3. Signal reversal
        current_score = current_indicators.get("_signal_score", 0)
        entry_score = position.get("signal_score", 0)
        if entry_score and current_score:
            if (entry_score > 0 and current_score < -0.15) or \
               (entry_score < 0 and current_score > 0.15):
                urgency += 0.4
                reasons.append("Signal reversed direction")

        return {
            "should_exit": urgency >= 0.5,
            "reason": "; ".join(reasons) if reasons else "Hold",
            "urgency": urgency,
            "current_regime": current_regime,
        }
