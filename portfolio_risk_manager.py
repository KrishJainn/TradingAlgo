"""
Portfolio Risk Manager for the 5-Player Coach Trading System.

Sits BETWEEN player signal generation and trade execution to enforce:
- Single-stock concentration limits
- Sector exposure limits
- Gross / net portfolio exposure caps
- Per-player allocation caps
- Crowding risk (multi-player agreement) haircuts
- Dynamic trailing stops with drawdown-aware tightening
- Intraday circuit breaker

Designed for Nifty 30/50 stocks on NSE.

Usage:
    from portfolio_risk_manager import PortfolioRiskManager

    rm = PortfolioRiskManager(total_capital=500_000)
    approved, rejected = rm.filter_signals(player_signals, current_positions)
    report = rm.risk_report(current_positions)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Nifty stock → sector mapping
# ═══════════════════════════════════════════════════════════════════════════

class Sector(str, Enum):
    """NSE sector classification for Nifty constituents."""
    IT = "IT"
    BANKING = "Banking"
    PHARMA = "Pharma"
    AUTO = "Auto"
    FMCG = "FMCG"
    ENERGY = "Energy"
    METALS = "Metals"
    TELECOM = "Telecom"
    INFRA = "Infra"
    OTHER = "Other"


# Covers all 30 stocks from the system's FIXED_SYMBOLS plus extras from
# data/symbols.py NIFTY_50_SYMBOLS.  Keys are the *bare* ticker (no .NS
# suffix) so look-ups work regardless of whether the caller passes
# "RELIANCE" or "RELIANCE.NS".

NIFTY_SECTOR_MAP: Dict[str, Sector] = {
    # Energy
    "RELIANCE":    Sector.ENERGY,
    "ONGC":        Sector.ENERGY,
    "NTPC":        Sector.ENERGY,
    "POWERGRID":   Sector.ENERGY,
    "BPCL":        Sector.ENERGY,
    "COALINDIA":   Sector.ENERGY,
    "ADANIENT":    Sector.ENERGY,
    "ADANIPORTS":  Sector.INFRA,

    # IT
    "TCS":         Sector.IT,
    "INFY":        Sector.IT,
    "WIPRO":       Sector.IT,
    "HCLTECH":     Sector.IT,
    "TECHM":       Sector.IT,
    "LTI":         Sector.IT,

    # Banking / Financials
    "HDFCBANK":    Sector.BANKING,
    "ICICIBANK":   Sector.BANKING,
    "SBIN":        Sector.BANKING,
    "KOTAKBANK":   Sector.BANKING,
    "AXISBANK":    Sector.BANKING,
    "BAJFINANCE":  Sector.BANKING,
    "BAJAJFINSV":  Sector.BANKING,
    "INDUSINDBK":  Sector.BANKING,
    "HDFCLIFE":    Sector.BANKING,
    "SBILIFE":     Sector.BANKING,
    "SHRIRAMFIN":  Sector.BANKING,

    # FMCG
    "HINDUNILVR":  Sector.FMCG,
    "ITC":         Sector.FMCG,
    "NESTLEIND":   Sector.FMCG,
    "BRITANNIA":   Sector.FMCG,
    "TATACONSUM":  Sector.FMCG,
    "TITAN":       Sector.FMCG,
    "TRENT":       Sector.FMCG,

    # Auto
    "MARUTI":      Sector.AUTO,
    "TATAMOTORS":  Sector.AUTO,
    "M&M":         Sector.AUTO,
    "BAJAJ-AUTO":  Sector.AUTO,
    "EICHERMOT":   Sector.AUTO,
    "HEROMOTOCO":  Sector.AUTO,

    # Pharma
    "SUNPHARMA":   Sector.PHARMA,
    "DRREDDY":     Sector.PHARMA,
    "CIPLA":       Sector.PHARMA,
    "APOLLOHOSP":  Sector.PHARMA,

    # Metals
    "TATASTEEL":   Sector.METALS,
    "JSWSTEEL":    Sector.METALS,
    "HINDALCO":    Sector.METALS,

    # Infra / Conglomerate
    "LT":          Sector.INFRA,
    "ULTRACEMCO":  Sector.INFRA,
    "GRASIM":      Sector.INFRA,

    # Telecom
    "BHARTIARTL":  Sector.TELECOM,

    # Paint
    "ASIANPAINT":  Sector.OTHER,

    # Defence
    "BEL":         Sector.OTHER,
}


def _bare_ticker(symbol: str) -> str:
    """Strip '.NS' / '.BO' suffix if present."""
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def get_sector(symbol: str) -> Sector:
    """Look up sector for a symbol, defaulting to OTHER."""
    return NIFTY_SECTOR_MAP.get(_bare_ticker(symbol), Sector.OTHER)


# ═══════════════════════════════════════════════════════════════════════════
# Risk-limit configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RiskLimits:
    """All risk-limit thresholds — every field is configurable."""

    # Concentration
    max_single_stock_pct: float = 0.25        # 25% of total capital
    max_sector_pct: float = 0.40              # 40%
    max_per_player_pct: float = 0.30          # 30%

    # Portfolio-level exposure
    max_gross_exposure_pct: float = 2.00      # 200% (long + |short|)
    max_net_exposure_pct: float = 1.00        # 100% (long - |short|)

    # Crowding haircuts
    crowding_4_plus_haircut: float = 0.40     # reduce size by 40% if 4+ agree
    crowding_3_haircut: float = 0.20          # reduce size by 20% if 3 agree

    # Dynamic stop-loss
    trailing_stop_largecap_pct: float = 0.02  # 2%
    trailing_stop_midcap_pct: float = 0.03    # 3%
    drawdown_tighten_threshold: float = 0.05  # tighten at 5% portfolio DD
    drawdown_tighten_factor: float = 0.50     # stops halved during DD

    # Minimum holding period
    min_holding_bars: int = 3   # Hold position for at least 3 bars before exit

    # Circuit breaker
    intraday_circuit_breaker_pct: float = 0.05   # 5% intraday drop
    peak_circuit_breaker_pct: float = 0.12       # 12% from peak
    circuit_breaker_flatten_to: float = 0.50     # reduce to 50% of positions


# ═══════════════════════════════════════════════════════════════════════════
# Data structures for positions and signals
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PositionRecord:
    """
    Represents a single open position held by one player.

    This is the minimal record the risk manager needs.  The backtest
    engine's own Position dataclass contains more fields; callers should
    convert to PositionRecord for the risk-manager API or pass dicts
    with the same keys.
    """
    symbol: str
    player: str
    direction: str          # 'LONG' or 'SHORT'
    quantity: int
    entry_price: float
    current_price: float    # latest mark-to-market price
    position_value: float = 0.0  # |current_price * quantity|
    bars_held: int = 0      # how many bars position has been held

    def __post_init__(self):
        self.direction = self.direction.upper()
        if self.position_value == 0.0:
            self.position_value = abs(self.current_price * self.quantity)


@dataclass
class Signal:
    """One proposed trade from a player, used internally after parsing."""
    player: str
    symbol: str
    direction: str      # 'LONG' or 'SHORT'
    size: float         # rupee value of proposed position
    confidence: float   # higher = stronger conviction
    original_size: float = 0.0  # before any haircut

    def __post_init__(self):
        self.direction = self.direction.upper()
        if self.original_size == 0.0:
            self.original_size = self.size


# ═══════════════════════════════════════════════════════════════════════════
# PortfolioRiskManager
# ═══════════════════════════════════════════════════════════════════════════

class PortfolioRiskManager:
    """
    Central risk gate between player signals and trade execution.

    Workflow:
        1. Players generate signals  →  player_signals dict
        2. filter_signals() ranks by confidence, applies crowding
           haircuts, and greedily admits trades that respect all limits.
        3. Approved signals go to execution; rejected signals are logged.
    """

    def __init__(
        self,
        total_capital: float = 500_000,
        limits: Optional[RiskLimits] = None,
    ):
        self.total_capital = total_capital
        self.limits = limits or RiskLimits()

        # Drawdown tracking (updated externally via update_equity)
        self._peak_equity: float = total_capital
        self._current_equity: float = total_capital
        self._day_start_equity: float = total_capital

    # ------------------------------------------------------------------
    # Equity tracking (call from backtest loop)
    # ------------------------------------------------------------------

    def update_equity(self, equity: float, is_new_day: bool = False):
        """
        Feed the latest portfolio equity so the circuit breaker can fire.

        Args:
            equity:     Current mark-to-market portfolio value.
            is_new_day: True on the first bar of a new trading day.
        """
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity
        if is_new_day:
            self._day_start_equity = equity

    # ------------------------------------------------------------------
    # Signal filtering  (core method)
    # ------------------------------------------------------------------

    def filter_signals(
        self,
        player_signals: Dict[str, Dict[str, Tuple[str, float, float]]],
        current_positions: List[PositionRecord],
        total_capital: Optional[float] = None,
    ) -> Tuple[Dict[str, Dict[str, Tuple[str, float, float]]],
               Dict[str, Dict[str, Tuple[str, float, str]]]]:
        """
        Rank all proposed trades by confidence and admit greedily.

        Args:
            player_signals:
                {player_name: {symbol: (direction, size, confidence)}}
                *direction*: 'LONG' or 'SHORT'
                *size*: proposed position value in INR
                *confidence*: 0-1 score (higher = better)

            current_positions:
                List of PositionRecord for every open position across
                all players.

            total_capital:
                Override self.total_capital if provided.

        Returns:
            (approved, rejected)

            approved: same shape as player_signals but only the trades
                      that passed (sizes may be reduced by crowding
                      haircut).

            rejected: {player_name: {symbol: (direction, size, reason)}}
        """
        cap = total_capital if total_capital is not None else self.total_capital
        if cap <= 0:
            raise ValueError(f"total_capital must be positive, got {cap}")

        # --- snapshot current exposure ---
        stock_exposure, sector_exposure, player_exposure, gross, net = (
            self._compute_exposure(current_positions, cap)
        )

        # --- flatten all signals into a list, apply crowding haircut ---
        flat_signals = self._flatten_and_haircut(player_signals)

        # --- sort by confidence descending ---
        flat_signals.sort(key=lambda s: s.confidence, reverse=True)

        approved: Dict[str, Dict[str, Tuple[str, float, float]]] = {}
        rejected: Dict[str, Dict[str, Tuple[str, float, str]]] = {}

        for sig in flat_signals:
            reason = self._check_limits(
                sig, stock_exposure, sector_exposure,
                player_exposure, gross, net, cap,
            )
            if reason is None:
                # Admit the trade and update running totals
                signed = sig.size if sig.direction == "LONG" else -sig.size
                sector = get_sector(sig.symbol)

                stock_exposure[sig.symbol] = (
                    stock_exposure.get(sig.symbol, 0.0) + sig.size
                )
                sector_exposure[sector] = (
                    sector_exposure.get(sector, 0.0) + sig.size
                )
                player_exposure[sig.player] = (
                    player_exposure.get(sig.player, 0.0) + sig.size
                )
                gross += sig.size
                net += signed

                approved.setdefault(sig.player, {})[sig.symbol] = (
                    sig.direction, sig.size, sig.confidence,
                )
            else:
                rejected.setdefault(sig.player, {})[sig.symbol] = (
                    sig.direction, sig.size, reason,
                )
                logger.info(
                    "REJECTED %s %s %s (%.0f): %s",
                    sig.player, sig.direction, sig.symbol, sig.size, reason,
                )

        return approved, rejected

    # ------------------------------------------------------------------
    # Minimum holding period check
    # ------------------------------------------------------------------

    def should_hold_position(self, position: PositionRecord) -> bool:
        """Return True if position must be held (hasn't met min holding period)."""
        return position.bars_held < self.limits.min_holding_bars

    # ------------------------------------------------------------------
    # Dynamic stop-loss manager
    # ------------------------------------------------------------------

    def compute_trailing_stop(
        self,
        position: PositionRecord,
        highest_price_since_entry: float,
        is_large_cap: bool = True,
    ) -> float:
        """
        Return the trailing stop price for a position.

        Stops tighten by ``drawdown_tighten_factor`` when the portfolio
        is in drawdown beyond ``drawdown_tighten_threshold``.

        Args:
            position:                  The open position.
            highest_price_since_entry: The HWM price since entry.
            is_large_cap:              Large-cap gets tighter default stop.

        Returns:
            Stop-loss price.
        """
        base_pct = (
            self.limits.trailing_stop_largecap_pct
            if is_large_cap
            else self.limits.trailing_stop_midcap_pct
        )

        # Tighten if portfolio is in drawdown
        if self._peak_equity > 0:
            dd_from_peak = (
                (self._peak_equity - self._current_equity) / self._peak_equity
            )
            if dd_from_peak >= self.limits.drawdown_tighten_threshold:
                base_pct *= self.limits.drawdown_tighten_factor

        if position.direction == "LONG":
            stop = highest_price_since_entry * (1.0 - base_pct)
        else:
            stop = highest_price_since_entry * (1.0 + base_pct)

        return round(stop, 2)

    def check_circuit_breaker(self) -> Dict[str, Any]:
        """
        Check whether the portfolio-level circuit breaker should fire.

        Returns:
            {
                "triggered": bool,
                "reason": str or None,
                "action": "flatten_to_50" or None,
                "intraday_drawdown_pct": float,
                "peak_drawdown_pct": float,
            }
        """
        intraday_dd = 0.0
        if self._day_start_equity > 0:
            intraday_dd = (
                (self._day_start_equity - self._current_equity)
                / self._day_start_equity
            )

        peak_dd = 0.0
        if self._peak_equity > 0:
            peak_dd = (
                (self._peak_equity - self._current_equity) / self._peak_equity
            )

        triggered = False
        reason = None

        if intraday_dd >= self.limits.intraday_circuit_breaker_pct:
            triggered = True
            reason = (
                f"Intraday drawdown {intraday_dd:.1%} >= "
                f"{self.limits.intraday_circuit_breaker_pct:.0%} limit"
            )
        elif peak_dd >= self.limits.peak_circuit_breaker_pct:
            triggered = True
            reason = (
                f"Peak drawdown {peak_dd:.1%} >= "
                f"{self.limits.peak_circuit_breaker_pct:.0%} limit"
            )

        return {
            "triggered": triggered,
            "reason": reason,
            "action": "flatten_to_50" if triggered else None,
            "intraday_drawdown_pct": round(intraday_dd, 4),
            "peak_drawdown_pct": round(peak_dd, 4),
        }

    # ------------------------------------------------------------------
    # Risk report
    # ------------------------------------------------------------------

    def risk_report(
        self,
        current_positions: List[PositionRecord],
        total_capital: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Full exposure report with limit-breach flags.

        Returns a dict with:
            stock_exposure   – {symbol: value}
            sector_exposure  – {sector: value}
            player_exposure  – {player: value}
            gross_exposure, gross_exposure_pct
            net_exposure, net_exposure_pct
            largest_concentration – (symbol, pct)
            breaches – list of human-readable limit violations
        """
        cap = total_capital if total_capital is not None else self.total_capital
        if cap <= 0:
            raise ValueError(f"total_capital must be positive, got {cap}")
        stock_exp, sector_exp, player_exp, gross, net = (
            self._compute_exposure(current_positions, cap)
        )

        # Percentages
        stock_pct = {s: v / cap for s, v in stock_exp.items()} if cap else {}
        sector_pct = {s: v / cap for s, v in sector_exp.items()} if cap else {}
        player_pct = {p: v / cap for p, v in player_exp.items()} if cap else {}

        gross_pct = gross / cap if cap else 0
        net_pct = net / cap if cap else 0

        largest_sym = max(stock_pct, key=stock_pct.get) if stock_pct else None
        largest_conc = stock_pct.get(largest_sym, 0) if largest_sym else 0

        # Breach detection
        breaches: List[str] = []

        for sym, pct in stock_pct.items():
            if pct > self.limits.max_single_stock_pct:
                breaches.append(
                    f"STOCK {sym}: {pct:.1%} > "
                    f"{self.limits.max_single_stock_pct:.0%} limit"
                )

        for sec, pct in sector_pct.items():
            if pct > self.limits.max_sector_pct:
                breaches.append(
                    f"SECTOR {sec}: {pct:.1%} > "
                    f"{self.limits.max_sector_pct:.0%} limit"
                )

        for pl, pct in player_pct.items():
            if pct > self.limits.max_per_player_pct:
                breaches.append(
                    f"PLAYER {pl}: {pct:.1%} > "
                    f"{self.limits.max_per_player_pct:.0%} limit"
                )

        if gross_pct > self.limits.max_gross_exposure_pct:
            breaches.append(
                f"GROSS EXPOSURE: {gross_pct:.1%} > "
                f"{self.limits.max_gross_exposure_pct:.0%} limit"
            )

        if abs(net_pct) > self.limits.max_net_exposure_pct:
            breaches.append(
                f"NET EXPOSURE: {net_pct:.1%} > "
                f"{self.limits.max_net_exposure_pct:.0%} limit"
            )

        cb = self.check_circuit_breaker()

        return {
            "total_capital": cap,
            "stock_exposure": stock_exp,
            "stock_exposure_pct": stock_pct,
            "sector_exposure": sector_exp,
            "sector_exposure_pct": sector_pct,
            "player_exposure": player_exp,
            "player_exposure_pct": player_pct,
            "gross_exposure": round(gross, 2),
            "gross_exposure_pct": round(gross_pct, 4),
            "net_exposure": round(net, 2),
            "net_exposure_pct": round(net_pct, 4),
            "largest_concentration": (
                largest_sym,
                round(largest_conc, 4),
            ),
            "breaches": breaches,
            "circuit_breaker": cb,
            "num_positions": len(current_positions),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_exposure(
        self,
        positions: List[PositionRecord],
        cap: float,
    ) -> Tuple[
        Dict[str, float],   # stock_exposure
        Dict[str, float],   # sector_exposure
        Dict[str, float],   # player_exposure
        float,               # gross
        float,               # net
    ]:
        """Aggregate exposure from a list of positions."""
        stock_exp: Dict[str, float] = {}
        sector_exp: Dict[str, float] = {}
        player_exp: Dict[str, float] = {}
        gross = 0.0
        net = 0.0

        for pos in positions:
            val = pos.position_value
            signed = val if pos.direction == "LONG" else -val
            sector = get_sector(pos.symbol)

            stock_exp[pos.symbol] = stock_exp.get(pos.symbol, 0.0) + val
            sector_exp[sector] = sector_exp.get(sector, 0.0) + val
            player_exp[pos.player] = player_exp.get(pos.player, 0.0) + val
            gross += val
            net += signed

        return stock_exp, sector_exp, player_exp, gross, net

    def _flatten_and_haircut(
        self,
        player_signals: Dict[str, Dict[str, Tuple[str, float, float]]],
    ) -> List[Signal]:
        """
        Convert nested player_signals to flat list of Signal objects
        and apply crowding haircuts where multiple players agree.
        """
        # First pass: count agreements per (symbol, direction)
        agreement_count: Dict[Tuple[str, str], int] = {}
        for player, sigs in player_signals.items():
            for symbol, (direction, size, conf) in sigs.items():
                key = (symbol, direction.upper())
                agreement_count[key] = agreement_count.get(key, 0) + 1

        # Second pass: build Signal list with haircuts
        flat: List[Signal] = []
        for player, sigs in player_signals.items():
            for symbol, (direction, size, conf) in sigs.items():
                key = (symbol, direction.upper())
                count = agreement_count[key]

                original_size = size
                if count >= 4:
                    size *= (1.0 - self.limits.crowding_4_plus_haircut)
                elif count >= 3:
                    size *= (1.0 - self.limits.crowding_3_haircut)

                flat.append(Signal(
                    player=player,
                    symbol=symbol,
                    direction=direction.upper(),
                    size=size,
                    confidence=conf,
                    original_size=original_size,
                ))

        return flat

    def _check_limits(
        self,
        sig: Signal,
        stock_exp: Dict[str, float],
        sector_exp: Dict[str, float],
        player_exp: Dict[str, float],
        gross: float,
        net: float,
        cap: float,
    ) -> Optional[str]:
        """
        Check whether admitting *sig* would breach any limit.

        Returns None if OK, or a human-readable reason string.
        """
        lim = self.limits

        # Single-stock concentration
        new_stock = stock_exp.get(sig.symbol, 0.0) + sig.size
        if new_stock / cap > lim.max_single_stock_pct:
            return (
                f"Single-stock limit: {sig.symbol} would reach "
                f"{new_stock/cap:.1%} > {lim.max_single_stock_pct:.0%}"
            )

        # Sector concentration
        sector = get_sector(sig.symbol)
        new_sector = sector_exp.get(sector, 0.0) + sig.size
        if new_sector / cap > lim.max_sector_pct:
            return (
                f"Sector limit: {sector.value} would reach "
                f"{new_sector/cap:.1%} > {lim.max_sector_pct:.0%}"
            )

        # Per-player allocation
        new_player = player_exp.get(sig.player, 0.0) + sig.size
        if new_player / cap > lim.max_per_player_pct:
            return (
                f"Player limit: {sig.player} would reach "
                f"{new_player/cap:.1%} > {lim.max_per_player_pct:.0%}"
            )

        # Gross exposure
        new_gross = gross + sig.size
        if new_gross / cap > lim.max_gross_exposure_pct:
            return (
                f"Gross exposure limit: would reach "
                f"{new_gross/cap:.1%} > {lim.max_gross_exposure_pct:.0%}"
            )

        # Net exposure
        signed = sig.size if sig.direction == "LONG" else -sig.size
        new_net = net + signed
        if abs(new_net) / cap > lim.max_net_exposure_pct:
            return (
                f"Net exposure limit: would reach "
                f"{abs(new_net)/cap:.1%} > {lim.max_net_exposure_pct:.0%}"
            )

        return None  # All clear


# ═══════════════════════════════════════════════════════════════════════════
# Demo / self-test
# ═══════════════════════════════════════════════════════════════════════════

def _demo():
    """
    Demonstrate the risk manager blocking concentrated exposure.

    Scenario
    --------
    - Total capital: Rs 5,00,000 across 5 players (Rs 1,00,000 each).
    - All 5 players want to go LONG RELIANCE with high confidence.
    - Two players also want HDFCBANK and one wants INFY.
    - The risk manager should:
        a) Apply 40% crowding haircut to RELIANCE (5 players agree).
        b) Approve the first few RELIANCE signals until the 25%
           single-stock cap is hit, then reject the rest.
        c) Approve HDFCBANK and INFY if limits allow.
    """
    total_capital = 500_000

    print("=" * 70)
    print("  Portfolio Risk Manager — Concentrated Trade Rejection Demo")
    print("=" * 70)

    rm = PortfolioRiskManager(total_capital=total_capital)

    # ── Existing positions: one small INFY long held by PLAYER_2 ──
    current_positions = [
        PositionRecord(
            symbol="INFY.NS", player="PLAYER_2", direction="LONG",
            quantity=50, entry_price=1500, current_price=1520,
        ),
    ]

    # ── Signals: all 5 players pile into RELIANCE ──
    player_signals = {
        "PLAYER_1": {
            "RELIANCE.NS": ("LONG", 100_000, 0.92),   # Aggressive
            "HDFCBANK.NS": ("LONG",  60_000, 0.81),
        },
        "PLAYER_2": {
            "RELIANCE.NS": ("LONG",  80_000, 0.88),   # Conservative
            "INFY.NS":     ("LONG",  50_000, 0.75),
        },
        "PLAYER_3": {
            "RELIANCE.NS": ("LONG",  90_000, 0.85),   # Balanced
        },
        "PLAYER_4": {
            "RELIANCE.NS": ("LONG",  70_000, 0.79),   # VolBreakout
        },
        "PLAYER_5": {
            "RELIANCE.NS": ("LONG",  95_000, 0.90),   # Momentum
            "HDFCBANK.NS": ("LONG",  55_000, 0.77),
        },
    }

    print(f"\n  Total capital:    Rs {total_capital:>12,}")
    print(f"  Single-stock cap: {rm.limits.max_single_stock_pct:.0%}  "
          f"= Rs {total_capital * rm.limits.max_single_stock_pct:>10,.0f}")
    print(f"  Sector cap:       {rm.limits.max_sector_pct:.0%}  "
          f"= Rs {total_capital * rm.limits.max_sector_pct:>10,.0f}")
    print(f"  Per-player cap:   {rm.limits.max_per_player_pct:.0%}  "
          f"= Rs {total_capital * rm.limits.max_per_player_pct:>10,.0f}")

    print(f"\n  Existing position: PLAYER_2 LONG INFY.NS @ Rs 76,000")
    print(f"\n  Incoming signals ({sum(len(s) for s in player_signals.values())} total):")
    for player, sigs in sorted(player_signals.items()):
        for sym, (d, sz, conf) in sigs.items():
            print(f"    {player:10s}  {d:5s}  {sym:14s}  "
                  f"Rs {sz:>9,.0f}  conf={conf:.2f}")

    # ── Run the filter ──
    approved, rejected = rm.filter_signals(
        player_signals, current_positions, total_capital,
    )

    # ── Results ──
    print(f"\n  {'─' * 60}")
    print(f"  APPROVED ({sum(len(s) for s in approved.values())} trades):")
    for player, sigs in sorted(approved.items()):
        for sym, (d, sz, conf) in sigs.items():
            print(f"    {player:10s}  {d:5s}  {sym:14s}  "
                  f"Rs {sz:>9,.0f}  conf={conf:.2f}")

    print(f"\n  REJECTED ({sum(len(s) for s in rejected.values())} trades):")
    for player, sigs in sorted(rejected.items()):
        for sym, (d, sz, reason) in sigs.items():
            print(f"    {player:10s}  {sym:14s}  Rs {sz:>9,.0f}")
            print(f"      reason: {reason}")

    # ── Risk report on the new state ──
    # Build updated positions from approved signals
    all_positions = list(current_positions)
    for player, sigs in approved.items():
        for sym, (d, sz, conf) in sigs.items():
            est_price = 2500 if "RELIANCE" in sym else 1600
            qty = int(sz / est_price)
            all_positions.append(PositionRecord(
                symbol=sym, player=player, direction=d,
                quantity=qty, entry_price=est_price,
                current_price=est_price,
            ))

    report = rm.risk_report(all_positions, total_capital)

    print(f"\n  {'─' * 60}")
    print(f"  POST-FILTER RISK REPORT")
    print(f"  {'─' * 60}")
    print(f"  Gross exposure:        {report['gross_exposure_pct']:.1%}")
    print(f"  Net exposure:          {report['net_exposure_pct']:.1%}")
    print(f"  Largest stock conc:    {report['largest_concentration'][0]}  "
          f"@ {report['largest_concentration'][1]:.1%}")
    print(f"  Positions:             {report['num_positions']}")

    print(f"\n  Sector exposure:")
    for sec, pct in sorted(report["sector_exposure_pct"].items(),
                           key=lambda x: -x[1]):
        bar = "#" * int(pct * 50)
        print(f"    {str(sec):12s}  {pct:>6.1%}  {bar}")

    if report["breaches"]:
        print(f"\n  BREACHES:")
        for b in report["breaches"]:
            print(f"    {b}")
    else:
        print(f"\n  No limit breaches.")

    # ── Trailing stop demo ──
    print(f"\n  {'─' * 60}")
    print(f"  TRAILING STOP DEMO")
    print(f"  {'─' * 60}")

    sample_pos = PositionRecord(
        symbol="RELIANCE.NS", player="PLAYER_1", direction="LONG",
        quantity=24, entry_price=2500, current_price=2560,
    )

    stop_normal = rm.compute_trailing_stop(sample_pos, highest_price_since_entry=2580)
    print(f"  Normal stop (2% trail from HWM 2580): Rs {stop_normal:,.2f}")

    # Simulate a 6% portfolio drawdown
    rm.update_equity(total_capital * 0.94)
    stop_tight = rm.compute_trailing_stop(sample_pos, highest_price_since_entry=2580)
    print(f"  Tight stop  (1% trail, portfolio DD>5%): Rs {stop_tight:,.2f}")

    # ── Circuit breaker demo ──
    rm.update_equity(total_capital * 0.94)  # 6% from peak
    cb = rm.check_circuit_breaker()
    print(f"\n  Circuit breaker (6% intraday DD): "
          f"triggered={cb['triggered']}, action={cb['action']}")

    rm._day_start_equity = total_capital
    rm._peak_equity = total_capital
    rm.update_equity(total_capital * 0.87)  # 13% from peak
    cb = rm.check_circuit_breaker()
    print(f"  Circuit breaker (13% from peak):  "
          f"triggered={cb['triggered']}, action={cb['action']}")
    if cb["reason"]:
        print(f"      reason: {cb['reason']}")


if __name__ == "__main__":
    _demo()
