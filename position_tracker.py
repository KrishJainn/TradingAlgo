"""
Position Tracker for the Paper Trading Pipeline.

Manages open positions with full lifecycle: open, mark-to-market, check stops,
close, and persist to JSON.

Features:
    - 5 independent player accounts, each with ₹1L capital
    - Position CRUD (add, update, close, list)
    - Mark-to-market with latest prices
    - Trailing stop-loss enforcement
    - Per-player P&L tracking with separate capital ledgers
    - JSON persistence (append-only journal + mutable positions file)
    - Portfolio-level summary (equity, exposure, drawdown)

Usage:
    from position_tracker import PositionTracker

    tracker = PositionTracker(capital=1_000_000, num_players=5)
    tracker.open_position("RELIANCE.NS", "PLAYER_1", "LONG", 100, 2500.0)
    closed = tracker.check_stops(current_prices)
    tracker.mark_to_market(current_prices)
    summary = tracker.portfolio_summary()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Position dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    """A single open paper-trading position.

    Extended with SmartExitEngine fields for multi-signal exit evaluation.
    """

    position_id: str                # unique id: "{symbol}_{player}_{date}"
    symbol: str                     # e.g. "RELIANCE.NS"
    player_id: str                  # e.g. "PLAYER_1"
    direction: str                  # "LONG" or "SHORT"
    quantity: int
    entry_price: float
    entry_date: str                 # ISO date
    current_price: float = 0.0
    high_water_mark: float = 0.0   # peak price since entry (for trailing stop)
    trailing_stop_pct: float = 0.02
    unrealised_pnl: float = 0.0
    bars_held: int = 0

    # ── SmartExitEngine fields ─────────────────────────────────────
    entry_signal: float = 0.0           # signal strength at entry
    entry_agreement: float = 0.0        # player agreement at entry (0–1)
    entry_volatility: float = 0.0       # GARCH vol or ATR proxy at entry
    entry_regime: str = "bull"           # regime at entry
    remaining_fraction: float = 1.0     # fraction of original still open
    staged_exits_done: int = 0          # staged profit-taking tranches done (0–2)
    max_favorable_excursion: float = 0.0  # best unrealised P&L %
    max_adverse_excursion: float = 0.0   # worst unrealised P&L % (positive)
    signal_history: List[float] = field(default_factory=list)
    agreement_history: List[float] = field(default_factory=list)
    volume_history: List[float] = field(default_factory=list)

    def __post_init__(self):
        if self.current_price == 0.0:
            self.current_price = self.entry_price
        if self.high_water_mark == 0.0:
            self.high_water_mark = self.entry_price

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Position":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════════════
# PositionTracker — with per-player capital accounts
# ═══════════════════════════════════════════════════════════════════════════

PLAYER_IDS = ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]


class PositionTracker:
    """
    Manages open positions for paper trading.

    Each of the 5 players has its own independent capital account.
    Positions are stored in-memory and persisted to a JSON file.
    """

    def __init__(
        self,
        capital: float = 1_000_000,
        num_players: int = 5,
        capital_per_player: float = 100_000,
        positions_file: Optional[str] = None,
        trailing_stop_pct: float = 0.02,
    ) -> None:
        self.initial_capital = capital
        self.num_players = num_players
        self.capital_per_player_initial = capital_per_player
        self.trailing_stop_pct = trailing_stop_pct
        self.positions: Dict[str, Position] = {}   # position_id -> Position
        self.closed_today: List[Dict[str, Any]] = []  # trades closed today
        self._previous_equity: Optional[float] = None  # for daily return calc

        # Per-player capital accounts (cash available per player)
        self.player_capital: Dict[str, float] = {
            pid: capital_per_player for pid in PLAYER_IDS[:num_players]
        }

        # Legacy: total cash (sum of all player accounts)
        self.current_capital = capital

        self._positions_file = Path(positions_file) if positions_file else None
        if self._positions_file:
            self._positions_file.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # ── Position CRUD ─────────────────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        player_id: str,
        direction: str,
        quantity: int,
        entry_price: float,
        entry_date: Optional[str] = None,
        entry_signal: float = 0.0,
        entry_agreement: float = 0.0,
        entry_volatility: float = 0.0,
        entry_regime: str = "bull",
    ) -> Position:
        """Open a new paper position."""
        d = entry_date or date.today().isoformat()
        pid = f"{symbol}_{player_id}_{d}"

        # Check if already have position in same symbol for this player
        for existing in self.positions.values():
            if existing.symbol == symbol and existing.player_id == player_id:
                logger.warning(
                    f"Already have position in {symbol} for {player_id}, "
                    f"skipping duplicate."
                )
                return existing

        pos = Position(
            position_id=pid,
            symbol=symbol,
            player_id=player_id,
            direction=direction.upper(),
            quantity=quantity,
            entry_price=entry_price,
            entry_date=d,
            current_price=entry_price,
            high_water_mark=entry_price,
            trailing_stop_pct=self.trailing_stop_pct,
            entry_signal=entry_signal,
            entry_agreement=entry_agreement,
            entry_volatility=entry_volatility,
            entry_regime=entry_regime,
        )

        # Capital accounting from player's own account
        # Both LONG and SHORT lock up capital (margin requirement)
        cost = quantity * entry_price
        self.player_capital[player_id] -= cost
        self.current_capital -= cost

        self.positions[pid] = pos
        self._save()

        player_cash = self.player_capital.get(player_id, 0)
        logger.info(
            f"OPEN {direction} {quantity}x {symbol} @ ₹{entry_price:,.2f} "
            f"[{player_id}] cash_left=₹{player_cash:,.0f} id={pid}"
        )
        return pos

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_date: Optional[str] = None,
        reason: str = "signal",
    ) -> Dict[str, Any]:
        """Close a position and return trade summary."""
        pos = self.positions.get(position_id)
        if pos is None:
            logger.warning(f"Position {position_id} not found for closing")
            return {}

        d = exit_date or date.today().isoformat()

        # Calculate P&L
        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        trade = {
            "position_id": position_id,
            "symbol": pos.symbol,
            "player_id": pos.player_id,
            "direction": pos.direction,
            "quantity": pos.quantity,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "entry_date": pos.entry_date,
            "exit_date": d,
            "pnl": round(pnl, 2),
            "return_pct": round(
                pnl / (pos.entry_price * pos.quantity) * 100
                if pos.entry_price * pos.quantity > 0 else 0.0, 4
            ),
            "bars_held": pos.bars_held,
            "close_reason": reason,
            # MFE/MAE for post-trade analysis
            "mfe": round(getattr(pos, "max_favorable_excursion", 0.0), 6),
            "mae": round(getattr(pos, "max_adverse_excursion", 0.0), 6),
            "entry_signal": getattr(pos, "entry_signal", 0.0),
            "entry_agreement": getattr(pos, "entry_agreement", 0.0),
            "entry_volatility": getattr(pos, "entry_volatility", 0.0),
            "entry_regime": getattr(pos, "entry_regime", ""),
            "peak_price": getattr(pos, "high_water_mark", 0.0),
            "staged_exits_done": getattr(pos, "staged_exits_done", 0),
            "remaining_fraction": getattr(pos, "remaining_fraction", 1.0),
            "price_history": list(getattr(pos, "signal_history", [])),
            "volume_history": list(getattr(pos, "volume_history", [])),
        }

        # Return capital to player's own account
        # LONG: sell shares, get back exit_price * qty
        # SHORT: cover position, get back entry margin + profit (or - loss)
        if pos.direction == "LONG":
            self.player_capital[pos.player_id] += pos.quantity * exit_price
            self.current_capital += pos.quantity * exit_price
        else:
            # SHORT close: return original margin + P&L
            # margin locked = entry_price * qty, P&L = (entry - exit) * qty
            returned = pos.quantity * pos.entry_price + pnl
            self.player_capital[pos.player_id] += returned
            self.current_capital += returned

        del self.positions[position_id]
        self.closed_today.append(trade)
        self._save()

        player_cash = self.player_capital.get(pos.player_id, 0)
        logger.info(
            f"CLOSE {pos.direction} {pos.quantity}x {pos.symbol} @ ₹{exit_price:,.2f} "
            f"P&L=₹{pnl:+,.2f} [{pos.player_id}] cash=₹{player_cash:,.0f} reason={reason}"
        )
        return trade

    def partial_close_position(
        self,
        position_id: str,
        shares_to_exit: int,
        exit_price: float,
        exit_date: Optional[str] = None,
        reason: str = "staged_profit",
    ) -> Dict[str, Any]:
        """Partially close a position (staged exit).

        If shares_to_exit >= pos.quantity, delegates to full close_position().
        Otherwise reduces quantity, returns capital, and keeps position open.

        Returns trade summary dict.
        """
        pos = self.positions.get(position_id)
        if pos is None:
            logger.warning(f"Position {position_id} not found for partial close")
            return {}

        # If exiting all or more → full close
        if shares_to_exit >= pos.quantity:
            return self.close_position(position_id, exit_price, exit_date, reason)

        d = exit_date or date.today().isoformat()

        # Calculate partial P&L
        if pos.direction == "LONG":
            gross_pnl = (exit_price - pos.entry_price) * shares_to_exit
        else:
            gross_pnl = (pos.entry_price - exit_price) * shares_to_exit

        net_pnl = gross_pnl  # Paper trading — no transaction costs

        trade = {
            "position_id": position_id,
            "symbol": pos.symbol,
            "player_id": pos.player_id,
            "direction": pos.direction,
            "quantity": shares_to_exit,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "entry_date": pos.entry_date,
            "exit_date": d,
            "pnl": round(net_pnl, 2),
            "return_pct": round(
                net_pnl / (pos.entry_price * shares_to_exit) * 100
                if pos.entry_price * shares_to_exit > 0 else 0.0, 4
            ),
            "bars_held": pos.bars_held,
            "close_reason": reason,
            "partial": True,
            "shares_remaining": pos.quantity - shares_to_exit,
            # MFE/MAE snapshot at partial exit
            "mfe": round(getattr(pos, "max_favorable_excursion", 0.0), 6),
            "mae": round(getattr(pos, "max_adverse_excursion", 0.0), 6),
            "entry_signal": getattr(pos, "entry_signal", 0.0),
            "entry_regime": getattr(pos, "entry_regime", ""),
            "peak_price": getattr(pos, "high_water_mark", 0.0),
            "staged_exits_done": getattr(pos, "staged_exits_done", 0),
        }

        # Return capital for exited shares
        if pos.direction == "LONG":
            returned = shares_to_exit * exit_price
        else:
            # SHORT partial close: return margin fraction + P&L
            returned = shares_to_exit * pos.entry_price + gross_pnl

        self.player_capital[pos.player_id] += returned
        self.current_capital += returned

        # Update position
        original_qty = pos.quantity
        pos.quantity -= shares_to_exit
        pos.remaining_fraction = pos.quantity / original_qty if original_qty > 0 else 0.0

        self.closed_today.append(trade)
        self._save()

        remaining_shares = pos.quantity
        player_cash = self.player_capital.get(pos.player_id, 0)
        logger.info(
            f"📊 PARTIAL EXIT {pos.direction} {pos.symbol}: "
            f"{shares_to_exit} shares @ ₹{exit_price:,.2f} | "
            f"P&L: ₹{net_pnl:+,.2f} | {remaining_shares} shares remaining | "
            f"reason={reason} [{pos.player_id}] cash=₹{player_cash:,.0f}"
        )
        return trade

    def get_position(self, position_id: str) -> Optional[Position]:
        return self.positions.get(position_id)

    def get_positions_for_player(self, player_id: str) -> List[Position]:
        return [p for p in self.positions.values() if p.player_id == player_id]

    def get_positions_for_symbol(self, symbol: str) -> List[Position]:
        return [p for p in self.positions.values() if p.symbol == symbol]

    def capital_deployed_by_player(self, player_id: str) -> float:
        """Return total capital currently deployed by a specific player."""
        total = 0.0
        for p in self.positions.values():
            if p.player_id == player_id:
                total += p.quantity * p.entry_price
        return total

    def player_cash_available(self, player_id: str) -> float:
        """Return cash available for a specific player."""
        return self.player_capital.get(player_id, 0.0)

    @property
    def open_position_count(self) -> int:
        return len(self.positions)

    # ── Mark-to-market ────────────────────────────────────────────────

    def mark_to_market(self, current_prices: Dict[str, float]) -> None:
        """Update all positions with latest prices."""
        for pos in self.positions.values():
            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            pos.current_price = price
            pos.bars_held += 1

            # Update high water mark (for trailing stops)
            if pos.direction == "LONG":
                pos.high_water_mark = max(pos.high_water_mark, price)
            else:
                # For shorts, track lowest price as "high water mark" (inverted)
                pos.high_water_mark = min(pos.high_water_mark, price)

            # Compute unrealised P&L
            if pos.direction == "LONG":
                pos.unrealised_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealised_pnl = (pos.entry_price - price) * pos.quantity

        self._save()

    # ── Stop-loss checking ────────────────────────────────────────────

    def check_stops(
        self, current_prices: Dict[str, float], exit_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Check trailing stops for all positions. Close any that are hit.

        Returns list of closed trade dicts.
        """
        stopped = []
        to_close = []

        for pid, pos in self.positions.items():
            price = current_prices.get(pos.symbol)
            if price is None:
                continue

            hit = False
            if pos.direction == "LONG":
                stop_price = pos.high_water_mark * (1 - pos.trailing_stop_pct)
                if price <= stop_price:
                    hit = True
            else:
                stop_price = pos.high_water_mark * (1 + pos.trailing_stop_pct)
                if price >= stop_price:
                    hit = True

            if hit:
                to_close.append((pid, price))

        for pid, price in to_close:
            trade = self.close_position(
                pid, price, exit_date=exit_date, reason="trailing_stop"
            )
            if trade:
                stopped.append(trade)

        return stopped

    # ── Portfolio summary ─────────────────────────────────────────────

    def portfolio_summary(self) -> Dict[str, Any]:
        """Compute portfolio-level metrics with per-player breakdown."""
        total_unrealised = sum(p.unrealised_pnl for p in self.positions.values())

        long_invested = sum(
            p.entry_price * p.quantity
            for p in self.positions.values() if p.direction == "LONG"
        )
        short_invested = sum(
            p.entry_price * p.quantity
            for p in self.positions.values() if p.direction == "SHORT"
        )
        equity = self.current_capital + long_invested - short_invested + total_unrealised

        # Per-player breakdown
        player_pnl: Dict[str, float] = {}
        player_positions: Dict[str, int] = {}
        player_deployed: Dict[str, float] = {}
        for p in self.positions.values():
            player_pnl[p.player_id] = player_pnl.get(p.player_id, 0) + p.unrealised_pnl
            player_positions[p.player_id] = player_positions.get(p.player_id, 0) + 1
            player_deployed[p.player_id] = player_deployed.get(p.player_id, 0) + (p.quantity * p.entry_price)

        # Per-player account summary
        player_accounts: Dict[str, Dict[str, Any]] = {}
        for pid in self.player_capital:
            cash = self.player_capital[pid]
            deployed = player_deployed.get(pid, 0.0)
            pnl = player_pnl.get(pid, 0.0)
            player_equity = cash + deployed + pnl
            player_accounts[pid] = {
                "cash": round(cash, 2),
                "deployed": round(deployed, 2),
                "unrealised_pnl": round(pnl, 2),
                "equity": round(player_equity, 2),
                "positions": player_positions.get(pid, 0),
            }

        # Exposure
        long_exposure = sum(
            p.current_price * p.quantity
            for p in self.positions.values() if p.direction == "LONG"
        )
        short_exposure = sum(
            p.current_price * p.quantity
            for p in self.positions.values() if p.direction == "SHORT"
        )
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure

        return {
            "equity": round(equity, 2),
            "cash": round(self.current_capital, 2),
            "total_invested": round(long_invested + short_invested, 2),
            "unrealised_pnl": round(total_unrealised, 2),
            "cumulative_return": round(
                (equity - self.initial_capital) / self.initial_capital, 6
            ),
            "daily_return": round(
                (equity - self._previous_equity) / self._previous_equity
                if self._previous_equity and self._previous_equity > 0
                else 0.0, 6
            ),
            "open_positions": self.open_position_count,
            "gross_exposure": round(gross_exposure, 2),
            "net_exposure": round(net_exposure, 2),
            "gross_exposure_pct": round(
                gross_exposure / equity if equity > 0 else 0, 4
            ),
            "net_exposure_pct": round(
                net_exposure / equity if equity > 0 else 0, 4
            ),
            "per_player_pnl": player_pnl,
            "per_player_positions": player_positions,
            "player_accounts": player_accounts,
        }

    def get_positions_as_risk_format(self) -> List[Dict[str, Any]]:
        """Convert positions to PositionRecord-compatible dicts for PortfolioRiskManager."""
        records = []
        for p in self.positions.values():
            records.append({
                "symbol": p.symbol,
                "player": p.player_id,
                "direction": p.direction,
                "quantity": p.quantity,
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "position_value": p.current_price * p.quantity,
                "bars_held": p.bars_held,
            })
        return records

    # ── Today's closed trades ─────────────────────────────────────────

    def get_closed_today(self) -> List[Dict[str, Any]]:
        """Return trades closed during current session."""
        return list(self.closed_today)

    def reset_daily(self) -> None:
        """Reset daily accumulators at start of new trading day."""
        self.closed_today = []
        # Snapshot equity at start of day for daily return calculation
        summary = self.portfolio_summary()
        self._previous_equity = summary["equity"]

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        if self._positions_file is None:
            return
        try:
            data = {
                "current_capital": self.current_capital,
                "initial_capital": self.initial_capital,
                "player_capital": self.player_capital,
                "positions": {
                    pid: pos.to_dict() for pid, pos in self.positions.items()
                },
                "updated_at": datetime.now().isoformat(),
            }
            self._positions_file.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except OSError as e:
            logger.error(f"Failed to save positions: {e}")

    def _load(self) -> None:
        if self._positions_file is None or not self._positions_file.exists():
            return
        try:
            data = json.loads(
                self._positions_file.read_text(encoding="utf-8")
            )
            self.current_capital = data.get("current_capital", self.initial_capital)
            self.initial_capital = data.get("initial_capital", self.initial_capital)

            # Load per-player capital accounts
            saved_player_capital = data.get("player_capital", {})
            if saved_player_capital:
                for pid in self.player_capital:
                    if pid in saved_player_capital:
                        self.player_capital[pid] = saved_player_capital[pid]

            for pid, pdata in data.get("positions", {}).items():
                self.positions[pid] = Position.from_dict(pdata)

            # Log per-player status
            player_summary = ", ".join(
                f"{pid}=₹{cash:,.0f}"
                for pid, cash in self.player_capital.items()
            )
            logger.info(
                f"Loaded {len(self.positions)} positions | {player_summary}"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load positions file: {e}")

    # ── String repr ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PositionTracker(positions={len(self.positions)}, "
            f"capital=₹{self.current_capital:,.0f})"
        )
