"""
NSE transaction cost model with market microstructure adjustments.

Models the full cost structure for trading on the National Stock Exchange
of India, including statutory charges, brokerage, and a realistic slippage
model that accounts for volatility, participation rate, signal urgency,
and intraday microstructure effects.

Slippage model:
    slippage = base_slippage_bps
               * volatility_multiplier
               * signal_multiplier
               * microstructure_multiplier
               * participation_impact

    where participation_impact = (quantity / ADV) ^ 0.5

    The square-root law for market impact is per:
        Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio
        Transactions." *Journal of Risk*, 3(2), 5-39.

References:
    - Almgren & Chriss (2001) for the square-root market impact law.
    - NSE circular on transaction charges (updated annually).
    - SEBI fee schedule for turnover fees.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from risk_engine.config.constants import NSE_COSTS, NSE_MICROSTRUCTURE

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class CostBreakdown:
    """Itemized breakdown of a single-leg transaction cost.

    All monetary fields are in INR. ``total_bps`` expresses the total
    cost as basis points of the traded notional.

    Attributes:
        brokerage: Brokerage charge (min of flat fee and percentage).
        stt: Securities Transaction Tax.
        exchange: NSE exchange transaction charge.
        gst: Goods & Services Tax on brokerage + exchange charges.
        sebi: SEBI turnover fee.
        stamp_duty: State stamp duty on buy-side transactions.
        slippage: Estimated market impact / slippage cost in INR.
        total: Sum of all cost components in INR.
        total_bps: Total cost expressed in basis points of notional.
    """

    brokerage: float
    stt: float
    exchange: float
    gst: float
    sebi: float
    stamp_duty: float
    slippage: float
    total: float
    total_bps: float

    def __repr__(self) -> str:
        return (
            f"CostBreakdown(total=₹{self.total:,.2f}, "
            f"total_bps={self.total_bps:.2f}, "
            f"brokerage=₹{self.brokerage:.2f}, "
            f"stt=₹{self.stt:.2f}, "
            f"slippage=₹{self.slippage:.2f})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Signal urgency multipliers
# ═══════════════════════════════════════════════════════════════════════════

_SIGNAL_MULTIPLIERS = {
    "urgent": 1.5,       # Must fill immediately — wider market orders
    "normal": 1.0,       # Standard TWAP/VWAP execution
    "passive": 0.6,      # Limit orders with patience — lower impact
    "rebalance": 0.8,    # Scheduled rebalance — moderate urgency
}


# ═══════════════════════════════════════════════════════════════════════════
# Cost model
# ═══════════════════════════════════════════════════════════════════════════

class CostModel:
    """NSE transaction cost model with microstructure-aware slippage.

    Computes the full cost of a trade on NSE, including:
        - Brokerage (minimum of flat fee and percentage of turnover)
        - Securities Transaction Tax (STT)
        - NSE exchange transaction charges
        - SEBI turnover fee
        - State stamp duty
        - GST on brokerage and exchange charges
        - Slippage / market impact (microstructure-adjusted)

    The slippage model follows the square-root law of market impact
    from Almgren & Chriss (2001), adjusted for intraday microstructure
    patterns specific to NSE trading sessions.

    Reference:
        Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio
        Transactions." *Journal of Risk*, 3(2), 5-39.
    """

    def __init__(self) -> None:
        self._costs = NSE_COSTS
        self._micro = NSE_MICROSTRUCTURE

    # ───────────────────────────────────────────────────────────────────
    # Public API
    # ───────────────────────────────────────────────────────────────────

    def one_way_cost(
        self,
        price: float,
        quantity: int,
        side: str,
        trade_type: str = "intraday",
        avg_daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        signal_type: str = "normal",
        hour: float = 12.0,
        is_expiry: bool = False,
    ) -> CostBreakdown:
        """Compute the full one-way transaction cost for a single trade.

        Parameters:
            price: Per-share price in INR.
            quantity: Number of shares to trade.
            side: 'buy' or 'sell'.
            trade_type: 'intraday' or 'delivery'. Determines STT treatment.
            avg_daily_volume: Average daily volume (shares) for the stock.
                Used for participation-rate market impact calculation.
            volatility: Recent daily return volatility (e.g. 0.02 = 2%).
                Scales the slippage estimate.
            signal_type: One of 'urgent', 'normal', 'passive', 'rebalance'.
                Determines execution aggressiveness and hence slippage.
            hour: Hour of day in 24h format (e.g., 9.25 for 9:15 AM).
                Used for microstructure slippage adjustment.
            is_expiry: Whether today is a derivatives expiry day (Thursday).
                Adds additional slippage per NSE_MICROSTRUCTURE.

        Returns:
            CostBreakdown with itemized costs in INR and total in bps.

        Raises:
            ValueError: If side or trade_type is invalid.
        """
        side = side.lower()
        trade_type = trade_type.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        if trade_type not in ("intraday", "delivery"):
            raise ValueError(
                f"trade_type must be 'intraday' or 'delivery', got '{trade_type}'"
            )

        notional = price * quantity

        # 1. Brokerage: min(flat, percentage)
        brokerage_flat = self._costs["brokerage_flat"]
        brokerage_pct = notional * self._costs["brokerage_pct"]
        brokerage = min(brokerage_flat, brokerage_pct)

        # 2. Securities Transaction Tax (STT)
        if trade_type == "intraday":
            # Intraday: STT only on sell side
            stt = notional * self._costs["stt_intraday_sell"] if side == "sell" else 0.0
        else:
            # Delivery: STT on both buy and sell
            stt = notional * self._costs["stt_delivery_both"]

        # 3. Exchange transaction charge
        exchange = notional * self._costs["exchange_txn_nse"]

        # 4. SEBI turnover fee
        sebi = notional * self._costs["sebi_fee"]

        # 5. Stamp duty (only on buy side per most state rules)
        stamp_duty = notional * self._costs["stamp_duty"] if side == "buy" else 0.0

        # 6. GST: 18% on (brokerage + exchange transaction charge)
        gst = (brokerage + exchange) * self._costs["gst_on_brokerage"]

        # 7. Slippage / market impact
        slippage_bps = self._compute_slippage_bps(
            quantity=quantity,
            avg_daily_volume=avg_daily_volume,
            volatility=volatility,
            signal_type=signal_type,
            hour=hour,
            is_expiry=is_expiry,
        )
        slippage = notional * (slippage_bps / 10_000.0)

        # Total
        total = brokerage + stt + exchange + gst + sebi + stamp_duty + slippage
        total_bps = (total / notional * 10_000.0) if notional > 0 else 0.0

        return CostBreakdown(
            brokerage=brokerage,
            stt=stt,
            exchange=exchange,
            gst=gst,
            sebi=sebi,
            stamp_duty=stamp_duty,
            slippage=slippage,
            total=total,
            total_bps=total_bps,
        )

    def round_trip_cost_bps(
        self,
        price: float,
        quantity: int,
        trade_type: str = "intraday",
        avg_daily_volume: float = 1_000_000,
        volatility: float = 0.02,
        signal_type: str = "normal",
        hour: float = 12.0,
        is_expiry: bool = False,
    ) -> float:
        """Compute the round-trip (buy + sell) cost in basis points.

        This is the total cost of opening and closing a position,
        which represents the minimum alpha a trade must generate
        just to break even.

        Parameters:
            price: Per-share price in INR.
            quantity: Number of shares per leg.
            trade_type: 'intraday' or 'delivery'.
            avg_daily_volume: Average daily volume in shares.
            volatility: Daily return volatility.
            signal_type: Execution urgency category.
            hour: Hour of day (24h format).
            is_expiry: Whether today is expiry day.

        Returns:
            Round-trip cost in basis points (float).
        """
        buy_cost = self.one_way_cost(
            price=price,
            quantity=quantity,
            side="buy",
            trade_type=trade_type,
            avg_daily_volume=avg_daily_volume,
            volatility=volatility,
            signal_type=signal_type,
            hour=hour,
            is_expiry=is_expiry,
        )
        sell_cost = self.one_way_cost(
            price=price,
            quantity=quantity,
            side="sell",
            trade_type=trade_type,
            avg_daily_volume=avg_daily_volume,
            volatility=volatility,
            signal_type=signal_type,
            hour=hour,
            is_expiry=is_expiry,
        )
        return buy_cost.total_bps + sell_cost.total_bps

    @staticmethod
    def alpha_threshold(round_trip_bps: float, safety_margin: float = 2.0) -> float:
        """Compute the minimum alpha required to justify a trade.

        A trade is only worth executing if the expected alpha exceeds
        the round-trip cost by a safety margin. This implements the
        cost-aware signal filter:

            alpha_min = round_trip_bps * safety_margin

        Parameters:
            round_trip_bps: Total round-trip cost in basis points.
            safety_margin: Multiplier on cost (default 2.0). A value of
                2.0 means the signal must predict at least 2x the cost
                to be considered tradeable.

        Returns:
            Minimum alpha threshold in basis points.
        """
        if safety_margin < 1.0:
            logger.warning(
                "safety_margin=%.2f is below 1.0; trades may not cover costs.",
                safety_margin,
            )
        return round_trip_bps * safety_margin

    def get_microstructure_multiplier(
        self, hour: float, is_expiry: bool = False
    ) -> float:
        """Look up the intraday microstructure slippage multiplier.

        NSE trading sessions have varying liquidity and spread patterns:
            - 9:15 AM (pre-open): Widest spreads, highest impact.
            - 9:15-10:00 AM: High volatility, still wide spreads.
            - 10:00-11:00 AM: Spreads narrowing, moderate impact.
            - 11:00 AM - 2:00 PM: Most liquid period, lowest impact.
            - 2:00-2:30 PM: Pre-close positioning begins.
            - 2:30-3:30 PM: Closing auction effects, wider spreads.

        On expiry days (Thursdays), an additional multiplier is applied.

        Parameters:
            hour: Hour of day in 24h decimal format (e.g. 9.25 = 9:15 AM).
            is_expiry: Whether today is a derivatives expiry day.

        Returns:
            Slippage multiplier (>= 1.0).
        """
        # Determine base multiplier from time-of-day
        if hour < 9.25:
            # Before market open or during pre-open
            multiplier = self._micro["pre_open_multiplier"]
        elif hour < 10.0:
            multiplier = self._micro["morning_multiplier"]
        elif hour < 11.0:
            multiplier = self._micro["late_morning_mult"]
        elif hour < 14.0:
            # Midday calm zone — best execution window
            multiplier = self._micro["midday_multiplier"]
        elif hour < 14.5:
            multiplier = self._micro["pre_close_multiplier"]
        elif hour <= 15.5:
            multiplier = self._micro["closing_multiplier"]
        else:
            # After hours — use closing multiplier as proxy
            multiplier = self._micro["closing_multiplier"]

        # Expiry day adjustment
        if is_expiry:
            if hour >= 15.0:
                # Last 30 minutes on expiry — extreme impact
                multiplier *= self._micro["expiry_last_30min_mult"]
            else:
                multiplier *= self._micro["expiry_day_multiplier"]

        return multiplier

    # ───────────────────────────────────────────────────────────────────
    # Internal helpers
    # ───────────────────────────────────────────────────────────────────

    def _compute_slippage_bps(
        self,
        quantity: int,
        avg_daily_volume: float,
        volatility: float,
        signal_type: str,
        hour: float,
        is_expiry: bool,
    ) -> float:
        """Estimate slippage in basis points using the Almgren-Chriss model.

        The slippage model combines four multiplicative factors:

        1. **Volatility multiplier**: Higher volatility widens spreads.
           vol_mult = max(1.0, volatility / 0.02) — baseline is 2% daily vol.

        2. **Signal urgency multiplier**: Aggressive signals (market orders)
           incur more impact than passive limit orders.

        3. **Microstructure multiplier**: Intraday patterns on NSE affect
           available liquidity and bid-ask spreads.

        4. **Participation impact**: Square-root law from Almgren & Chriss:
           impact = (quantity / ADV) ^ 0.5
           This captures the nonlinear relationship between order size
           and market impact.

        Reference:
            Almgren, R. & Chriss, N. (2001). "Optimal Execution of
            Portfolio Transactions." *Journal of Risk*, 3(2), 5-39.

        Parameters:
            quantity: Number of shares in the order.
            avg_daily_volume: Average daily volume for the stock.
            volatility: Recent daily return volatility.
            signal_type: Execution urgency category.
            hour: Hour of day (24h format).
            is_expiry: Whether today is derivatives expiry day.

        Returns:
            Estimated slippage in basis points.
        """
        base_slippage_bps = float(self._costs["avg_slippage_liquid_bps"])

        # 1. Volatility multiplier: scale relative to 2% baseline vol
        baseline_vol = 0.02
        vol_multiplier = max(1.0, volatility / baseline_vol)

        # 2. Signal urgency multiplier
        signal_multiplier = _SIGNAL_MULTIPLIERS.get(signal_type, 1.0)
        if signal_type not in _SIGNAL_MULTIPLIERS:
            logger.warning(
                "Unknown signal_type '%s', using multiplier 1.0. "
                "Valid types: %s",
                signal_type,
                list(_SIGNAL_MULTIPLIERS.keys()),
            )

        # 3. Microstructure multiplier (time-of-day + expiry)
        micro_multiplier = self.get_microstructure_multiplier(hour, is_expiry)

        # 4. Participation impact: square-root law (Almgren & Chriss 2001)
        # participation_rate = quantity / ADV
        if avg_daily_volume > 0:
            participation_rate = quantity / avg_daily_volume
        else:
            participation_rate = 1.0  # Assume 100% if ADV unknown
            logger.warning(
                "avg_daily_volume is 0; assuming participation_rate=1.0"
            )
        participation_impact = math.sqrt(participation_rate)

        # Combined slippage
        slippage_bps = (
            base_slippage_bps
            * vol_multiplier
            * signal_multiplier
            * micro_multiplier
            * participation_impact
        )

        return slippage_bps
