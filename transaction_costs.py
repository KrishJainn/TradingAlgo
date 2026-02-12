"""
Transaction Cost Model for Indian Equity Markets.

Advanced cost model with:
- Complete Indian regulatory fee structure (STT, GST, SEBI, stamp duty)
- Quadratic market impact slippage scaled by volume and volatility
- Signal-aware slippage multipliers (momentum vs mean-reversion)
- CostAwareBacktester wrapper for post-hoc cost adjustment

Designed to plug into the 5-player coach backtesting system.

Usage:
    from transaction_costs import TransactionCostModel, CostAwareBacktester

    model = TransactionCostModel()
    cost = model.calculate_cost(
        price=2500, quantity=200, side='BUY', trade_type='intraday',
        avg_daily_volume=5_000_000, volatility=0.018, signal_type='momentum'
    )
    print(model.summary())
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import math
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TransactionCostModel
# ---------------------------------------------------------------------------

@dataclass
class TransactionCostModel:
    """
    Realistic transaction cost model for NSE equity trading.

    Statutory costs follow current SEBI/NSE fee schedules.
    Slippage uses a quadratic impact model calibrated for Nifty large-caps.

    All percentage parameters are expressed as decimals (0.01 = 1%).
    """

    # --- Brokerage ---
    brokerage_pct: float = 0.0003          # 0.03% per trade

    # --- STT (Securities Transaction Tax) ---
    stt_delivery_sell_pct: float = 0.001    # 0.1% on delivery sell
    stt_intraday_sell_pct: float = 0.00025  # 0.025% on intraday sell

    # --- Exchange transaction charges (NSE equity) ---
    exchange_txn_pct: float = 0.0000325     # 0.00325%

    # --- GST on (brokerage + exchange charges) ---
    gst_pct: float = 0.18                   # 18%

    # --- SEBI turnover fee ---
    sebi_fee_pct: float = 0.000001          # 0.0001%

    # --- Stamp duty (buy-side only) ---
    stamp_duty_pct: float = 0.00015         # 0.015%

    # --- Slippage / Market Impact ---
    base_slippage_pct: float = 0.0005       # 0.05% base for Nifty large-caps
    impact_exponent: float = 2.0            # quadratic scaling with participation rate
    baseline_volatility: float = 0.015      # 1.5% daily vol as reference

    # Signal-type slippage multipliers
    momentum_slippage_mult: float = 1.5     # buying into strength -> more adverse fill
    mean_reversion_slippage_mult: float = 0.7  # buying into weakness -> easier fill
    neutral_slippage_mult: float = 1.0

    # Last-calculation state (populated by calculate_cost)
    _last_breakdown: Dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_cost(
        self,
        price: float,
        quantity: int,
        side: str,
        trade_type: str = "intraday",
        avg_daily_volume: float = 5_000_000,
        volatility: float = 0.015,
        signal_type: str = "neutral",
    ) -> float:
        """
        Calculate total transaction cost in rupees.

        Args:
            price:            Per-share price in INR.
            quantity:         Number of shares.
            side:             'BUY' or 'SELL'.
            trade_type:       'intraday' or 'delivery'.
            avg_daily_volume: Average daily traded volume (shares).
            volatility:       Realised daily volatility (decimal, e.g. 0.018).
            signal_type:      'momentum', 'mean_reversion', or 'neutral'.

        Returns:
            Total cost in rupees (always positive).
        """
        side = side.upper()
        trade_type = trade_type.lower()
        trade_value = price * quantity

        # --- 1. Brokerage ---
        brokerage = trade_value * self.brokerage_pct

        # --- 2. STT ---
        stt = 0.0
        if side == "SELL":
            if trade_type == "delivery":
                stt = trade_value * self.stt_delivery_sell_pct
            else:
                stt = trade_value * self.stt_intraday_sell_pct

        # --- 3. Exchange transaction charges ---
        exchange_charges = trade_value * self.exchange_txn_pct

        # --- 4. GST on (brokerage + exchange charges) ---
        gst = (brokerage + exchange_charges) * self.gst_pct

        # --- 5. SEBI turnover fee ---
        sebi_fee = trade_value * self.sebi_fee_pct

        # --- 6. Stamp duty (buy-side only) ---
        stamp_duty = 0.0
        if side == "BUY":
            stamp_duty = trade_value * self.stamp_duty_pct

        # --- 7. Slippage (quadratic impact) ---
        slippage = self._calculate_slippage(
            trade_value, quantity, avg_daily_volume, volatility, signal_type,
        )

        # --- Totals ---
        statutory_total = stt + exchange_charges + gst + sebi_fee + stamp_duty
        total_cost = brokerage + statutory_total + slippage

        # Store breakdown for summary()
        self._last_breakdown = {
            "price": price,
            "quantity": quantity,
            "side": side,
            "trade_type": trade_type,
            "trade_value": round(trade_value, 2),
            "brokerage": round(brokerage, 2),
            "stt": round(stt, 2),
            "exchange_charges": round(exchange_charges, 2),
            "gst": round(gst, 2),
            "sebi_fee": round(sebi_fee, 2),
            "stamp_duty": round(stamp_duty, 2),
            "slippage": round(slippage, 2),
            "statutory_total": round(statutory_total, 2),
            "total_cost": round(total_cost, 2),
            "cost_bps": round(total_cost / trade_value * 10_000, 2) if trade_value else 0,
            "signal_type": signal_type,
            "avg_daily_volume": avg_daily_volume,
            "volatility": volatility,
        }

        return total_cost

    def calculate_cost_bps(self, **kwargs) -> float:
        """
        Calculate total transaction cost in basis points.

        Accepts the same arguments as calculate_cost().

        Returns:
            Cost in basis points (1 bp = 0.01%).
        """
        total = self.calculate_cost(**kwargs)
        trade_value = kwargs["price"] * kwargs["quantity"]
        if trade_value == 0:
            return 0.0
        return total / trade_value * 10_000

    def summary(self) -> Dict[str, Any]:
        """
        Return the full breakdown from the most recent calculate_cost() call.

        Returns:
            Dict with every cost component, trade metadata, and totals.
        """
        return dict(self._last_breakdown)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calculate_slippage(
        self,
        trade_value: float,
        quantity: int,
        avg_daily_volume: float,
        volatility: float,
        signal_type: str,
    ) -> float:
        """
        Quadratic market-impact slippage model.

        Two additive components:

        1. Spread cost (always paid, even for 1 share):
             spread_pct = base_slippage * vol_multiplier * signal_multiplier

        2. Market impact (grows quadratically with order size):
             impact_pct = impact_coeff * (quantity / adv)^exponent * vol_multiplier * signal_mult

        The impact coefficient is scaled so that an order equal to 1% of ADV
        adds roughly base_slippage worth of additional impact on top of the
        spread.  This means a 50k-share order against 5M ADV (1% participation)
        costs meaningfully more than a 100-share order (0.002%).

        total_slippage_pct = spread_pct + impact_pct
        """
        if avg_daily_volume <= 0:
            avg_daily_volume = 1  # safety: avoid div-by-zero

        participation_rate = quantity / avg_daily_volume

        # Volatility scaling (higher vol -> wider spreads -> more slippage)
        vol_multiplier = max(volatility, 0.001) / self.baseline_volatility

        # Signal-type multiplier
        signal_mult = self._signal_multiplier(signal_type)

        # Component 1: spread cost — always present
        spread_pct = self.base_slippage_pct * vol_multiplier * signal_mult

        # Component 2: market impact — quadratic in participation rate
        # impact_coeff is set so that participation == 1% adds ~base_slippage
        # of extra cost:  coeff * 0.01^2 = base_slippage  =>  coeff = base / 0.0001
        impact_coeff = self.base_slippage_pct / math.pow(0.01, self.impact_exponent)
        impact_pct = (
            impact_coeff
            * math.pow(participation_rate, self.impact_exponent)
            * vol_multiplier
            * signal_mult
        )

        total_pct = spread_pct + impact_pct
        return trade_value * total_pct

    def _signal_multiplier(self, signal_type: str) -> float:
        """Map signal type to slippage multiplier."""
        mapping = {
            "momentum": self.momentum_slippage_mult,
            "mean_reversion": self.mean_reversion_slippage_mult,
            "neutral": self.neutral_slippage_mult,
        }
        return mapping.get(signal_type.lower(), self.neutral_slippage_mult)


# ---------------------------------------------------------------------------
# CostAwareBacktester
# ---------------------------------------------------------------------------

class CostAwareBacktester:
    """
    Post-hoc cost adjustment wrapper.

    Takes a list of completed trades (dicts or Trade objects from the
    existing backtest engine) and re-computes P&L with the advanced
    TransactionCostModel applied to every leg.

    Usage:
        cab = CostAwareBacktester(cost_model=TransactionCostModel())
        result = cab.apply(trades, initial_capital=100_000)
        print(result["adjusted_sharpe"])
    """

    def __init__(
        self,
        cost_model: Optional[TransactionCostModel] = None,
        trading_days_per_year: int = 252,
        risk_free_rate: float = 0.065,  # India 10y benchmark ~6.5%
    ):
        self.cost_model = cost_model or TransactionCostModel()
        self.trading_days_per_year = trading_days_per_year
        self.risk_free_rate = risk_free_rate

    def apply(
        self,
        trades: List[Dict[str, Any]],
        initial_capital: float = 100_000,
        avg_daily_volume: float = 5_000_000,
        volatility: float = 0.015,
    ) -> Dict[str, Any]:
        """
        Apply cost model to a list of trades and return adjusted metrics.

        Each trade dict must have at minimum:
            entry_price, exit_price, quantity

        Optional fields used if present:
            trade_type  (default 'intraday')
            signal_type (default 'neutral')
            avg_daily_volume, volatility  (override per-trade)
            gross_pnl   (if absent, calculated from prices)

        Args:
            trades:          List of trade dicts.
            initial_capital: Starting capital for return calculations.
            avg_daily_volume: Default ADV if not per-trade.
            volatility:      Default daily vol if not per-trade.

        Returns:
            Dict with: adjusted_pnl, gross_pnl, total_costs,
                       cost_drag_pct, adjusted_sharpe, per_trade details.
        """
        adjusted_trades: List[Dict[str, Any]] = []
        total_gross = 0.0
        total_costs = 0.0
        daily_returns: List[float] = []

        for t in trades:
            entry_price = t["entry_price"]
            exit_price = t["exit_price"]
            quantity = t["quantity"]
            trade_type = t.get("trade_type", "intraday")
            signal_type = t.get("signal_type", "neutral")
            adv = t.get("avg_daily_volume", avg_daily_volume)
            vol = t.get("volatility", volatility)

            # Gross P&L
            gross_pnl = t.get("gross_pnl", (exit_price - entry_price) * quantity)

            # Entry cost
            entry_cost = self.cost_model.calculate_cost(
                price=entry_price, quantity=quantity, side="BUY",
                trade_type=trade_type, avg_daily_volume=adv,
                volatility=vol, signal_type=signal_type,
            )
            entry_breakdown = self.cost_model.summary()

            # Exit cost
            exit_cost = self.cost_model.calculate_cost(
                price=exit_price, quantity=quantity, side="SELL",
                trade_type=trade_type, avg_daily_volume=adv,
                volatility=vol, signal_type=signal_type,
            )
            exit_breakdown = self.cost_model.summary()

            round_trip_cost = entry_cost + exit_cost
            net_pnl = gross_pnl - round_trip_cost

            total_gross += gross_pnl
            total_costs += round_trip_cost

            position_value = entry_price * quantity
            trade_return = net_pnl / position_value if position_value else 0
            daily_returns.append(trade_return)

            adjusted_trades.append({
                "entry_price": entry_price,
                "exit_price": exit_price,
                "quantity": quantity,
                "gross_pnl": round(gross_pnl, 2),
                "entry_cost": round(entry_cost, 2),
                "exit_cost": round(exit_cost, 2),
                "round_trip_cost": round(round_trip_cost, 2),
                "net_pnl": round(net_pnl, 2),
                "cost_bps": round(
                    round_trip_cost / position_value * 10_000 if position_value else 0, 2
                ),
                "entry_breakdown": entry_breakdown,
                "exit_breakdown": exit_breakdown,
            })

        adjusted_pnl = total_gross - total_costs

        # Cost drag as % of gross returns (guard against zero/negative gross)
        if total_gross > 0:
            cost_drag_pct = total_costs / total_gross * 100
        else:
            cost_drag_pct = 0.0

        # Adjusted Sharpe
        adjusted_sharpe = self._compute_sharpe(daily_returns)

        return {
            "gross_pnl": round(total_gross, 2),
            "total_costs": round(total_costs, 2),
            "adjusted_pnl": round(adjusted_pnl, 2),
            "cost_drag_pct": round(cost_drag_pct, 2),
            "adjusted_sharpe": round(adjusted_sharpe, 4),
            "num_trades": len(trades),
            "avg_cost_per_trade": round(
                total_costs / len(trades) if trades else 0, 2
            ),
            "avg_cost_bps": round(
                sum(t["cost_bps"] for t in adjusted_trades) / len(adjusted_trades)
                if adjusted_trades else 0,
                2,
            ),
            "trades": adjusted_trades,
        }

    def _compute_sharpe(self, returns: List[float]) -> float:
        """Annualised Sharpe from per-trade returns."""
        if len(returns) < 2:
            return 0.0

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std_ret = math.sqrt(variance) if variance > 0 else 1e-10

        daily_rf = self.risk_free_rate / self.trading_days_per_year
        excess = mean_ret - daily_rf

        # Annualise: assume roughly 1 trade per day for scaling
        annualised_sharpe = (excess / std_ret) * math.sqrt(self.trading_days_per_year)
        return annualised_sharpe


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

def _print_breakdown(label: str, model: TransactionCostModel):
    """Pretty-print the last cost breakdown."""
    s = model.summary()
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Side:           {s['side']}")
    print(f"  Trade Type:     {s['trade_type']}")
    print(f"  Trade Value:    Rs {s['trade_value']:>14,.2f}")
    print(f"  Signal Type:    {s['signal_type']}")
    print(f"  ADV:            {s['avg_daily_volume']:>14,.0f} shares")
    print(f"  Volatility:     {s['volatility']:.3%}")
    print(f"  {'─'*40}")
    print(f"  Brokerage:      Rs {s['brokerage']:>14,.2f}")
    print(f"  STT:            Rs {s['stt']:>14,.2f}")
    print(f"  Exchange Chg:   Rs {s['exchange_charges']:>14,.2f}")
    print(f"  GST:            Rs {s['gst']:>14,.2f}")
    print(f"  SEBI Fee:       Rs {s['sebi_fee']:>14,.2f}")
    print(f"  Stamp Duty:     Rs {s['stamp_duty']:>14,.2f}")
    print(f"  Slippage:       Rs {s['slippage']:>14,.2f}")
    print(f"  {'─'*40}")
    print(f"  Statutory:      Rs {s['statutory_total']:>14,.2f}")
    print(f"  TOTAL COST:     Rs {s['total_cost']:>14,.2f}")
    print(f"  Cost (bps):     {s['cost_bps']:>14.2f} bps")


if __name__ == "__main__":
    model = TransactionCostModel()

    # --- Example 1: Rs 50 lakh RELIANCE buy (momentum trade) ---
    reliance_price = 2500.0
    reliance_qty = 2000       # 2000 * 2500 = Rs 50,00,000
    reliance_adv = 8_000_000  # ~8M shares/day typical for RELIANCE

    cost_1 = model.calculate_cost(
        price=reliance_price,
        quantity=reliance_qty,
        side="BUY",
        trade_type="intraday",
        avg_daily_volume=reliance_adv,
        volatility=0.018,      # slightly above-average vol
        signal_type="momentum",
    )
    _print_breakdown(
        f"RELIANCE BUY  |  {reliance_qty} shares @ Rs {reliance_price:,.0f}  =  Rs {reliance_price*reliance_qty:,.0f}",
        model,
    )

    # --- Example 2: Rs 2 lakh HDFCBANK sell (mean-reversion trade) ---
    hdfc_price = 1600.0
    hdfc_qty = 125            # 125 * 1600 = Rs 2,00,000
    hdfc_adv = 6_000_000

    cost_2 = model.calculate_cost(
        price=hdfc_price,
        quantity=hdfc_qty,
        side="SELL",
        trade_type="intraday",
        avg_daily_volume=hdfc_adv,
        volatility=0.012,      # lower vol for HDFC
        signal_type="mean_reversion",
    )
    _print_breakdown(
        f"HDFCBANK SELL  |  {hdfc_qty} shares @ Rs {hdfc_price:,.0f}  =  Rs {hdfc_price*hdfc_qty:,.0f}",
        model,
    )

    # --- CostAwareBacktester demo ---
    print(f"\n\n{'='*60}")
    print("  CostAwareBacktester — applying costs to sample trades")
    print(f"{'='*60}")

    sample_trades = [
        {"entry_price": 2500, "exit_price": 2540, "quantity": 200,
         "signal_type": "momentum", "trade_type": "intraday"},
        {"entry_price": 1600, "exit_price": 1585, "quantity": 125,
         "signal_type": "mean_reversion", "trade_type": "intraday"},
        {"entry_price": 3400, "exit_price": 3450, "quantity": 150,
         "signal_type": "neutral", "trade_type": "delivery"},
    ]

    cab = CostAwareBacktester(cost_model=model)
    result = cab.apply(sample_trades, initial_capital=100_000)

    print(f"\n  Trades:           {result['num_trades']}")
    print(f"  Gross P&L:        Rs {result['gross_pnl']:>12,.2f}")
    print(f"  Total Costs:      Rs {result['total_costs']:>12,.2f}")
    print(f"  Adjusted P&L:     Rs {result['adjusted_pnl']:>12,.2f}")
    print(f"  Cost Drag:        {result['cost_drag_pct']:>11.2f}%")
    print(f"  Avg Cost/Trade:   Rs {result['avg_cost_per_trade']:>12,.2f}")
    print(f"  Avg Cost (bps):   {result['avg_cost_bps']:>11.2f} bps")
    print(f"  Adjusted Sharpe:  {result['adjusted_sharpe']:>11.4f}")

    print(f"\n  Per-trade breakdown:")
    for i, t in enumerate(result["trades"], 1):
        print(f"    Trade {i}: gross={t['gross_pnl']:>8,.2f}  "
              f"cost={t['round_trip_cost']:>7,.2f} ({t['cost_bps']:.1f}bps)  "
              f"net={t['net_pnl']:>8,.2f}")
