"""
Trade Journal for the Paper Trading Pipeline.

Append-only trade logging with filtering, performance summary, and CSV export.

Features:
    - Append-only JSON trade log (one file, growing array)
    - Record trades with full metadata (entry/exit, player, P&L, regime, etc.)
    - Filter by player, symbol, date range, direction
    - Performance summary: total P&L, win rate, avg win/loss ratio, Sharpe
    - Per-player performance breakdown
    - CSV export for analysis in spreadsheets

Usage:
    from trade_journal import TradeJournal

    journal = TradeJournal()
    journal.record_trade({
        "symbol": "RELIANCE.NS", "player_id": "PLAYER_1",
        "direction": "LONG", "entry_price": 2500, "exit_price": 2550,
        "quantity": 100, "pnl": 5000, ...
    })
    summary = journal.performance_summary()
    csv_str = journal.export_csv()
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# TradeJournal
# ═══════════════════════════════════════════════════════════════════════════

class TradeJournal:
    """
    Append-only trade journal with persistence and analytics.

    All trades are stored in a single JSON file as an array of trade dicts.
    """

    def __init__(self, journal_file: Optional[str] = None) -> None:
        self._file = Path(journal_file) if journal_file else None
        self._trades: List[Dict[str, Any]] = []

        if self._file:
            self._file.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    # ── Record trades ─────────────────────────────────────────────────

    def record_trade(self, trade: Dict[str, Any]) -> None:
        """Append a completed trade to the journal.

        Expected keys:
            symbol, player_id, direction, quantity,
            entry_price, exit_price, entry_date, exit_date,
            pnl, return_pct, bars_held, close_reason

        Optional keys:
            regime, signal_confidence, transaction_cost,
            debate_outcome, reflection_triggered
        """
        entry = dict(trade)
        entry.setdefault("recorded_at", datetime.now().isoformat())
        entry.setdefault("trade_id", len(self._trades))

        self._trades.append(entry)
        self._save()

        logger.info(
            f"JOURNAL: {entry.get('direction', '?')} "
            f"{entry.get('quantity', '?')}x {entry.get('symbol', '?')} "
            f"P&L=₹{entry.get('pnl', 0):+,.2f} [{entry.get('player_id', '?')}]"
        )

    def record_trades(self, trades: List[Dict[str, Any]]) -> None:
        """Record multiple trades at once."""
        for t in trades:
            self.record_trade(t)

    # ── Query / filter ────────────────────────────────────────────────

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    def get_all_trades(self) -> List[Dict[str, Any]]:
        """Return all trades (copies)."""
        return [dict(t) for t in self._trades]

    def filter_trades(
        self,
        player_id: Optional[str] = None,
        symbol: Optional[str] = None,
        direction: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Filter trades by various criteria. All filters are AND-combined."""
        results = self._trades

        if player_id:
            results = [t for t in results if t.get("player_id") == player_id]
        if symbol:
            results = [t for t in results if t.get("symbol") == symbol]
        if direction:
            results = [t for t in results if t.get("direction") == direction.upper()]
        if start_date:
            results = [t for t in results if t.get("exit_date", "") >= start_date]
        if end_date:
            results = [t for t in results if t.get("exit_date", "") <= end_date]
        if min_pnl is not None:
            results = [t for t in results if t.get("pnl", 0) >= min_pnl]
        if max_pnl is not None:
            results = [t for t in results if t.get("pnl", 0) <= max_pnl]

        return [dict(t) for t in results]

    def get_player_trades(self, player_id: str) -> List[Dict[str, Any]]:
        """Shortcut for filtering by player."""
        return self.filter_trades(player_id=player_id)

    def get_recent_trades(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the most recent N trades."""
        return [dict(t) for t in self._trades[-n:]]

    # ── Performance summary ───────────────────────────────────────────

    def performance_summary(
        self, trades: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute performance metrics for a set of trades (or all trades).

        Returns dict with:
            total_trades, total_pnl, win_count, loss_count, win_rate,
            avg_win, avg_loss, avg_win_loss_ratio, profit_factor,
            max_win, max_loss, avg_bars_held, sharpe (daily approx)
        """
        data = trades if trades is not None else self._trades
        if not data:
            return {
                "total_trades": 0,
                "total_pnl": 0.0,
                "win_count": 0,
                "loss_count": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "avg_win_loss_ratio": 0.0,
                "profit_factor": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
                "avg_bars_held": 0.0,
                "sharpe": 0.0,
            }

        pnls = [t.get("pnl", 0.0) for t in data]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        bars = [t.get("bars_held", 0) for t in data]

        total_pnl = sum(pnls)
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        avg_loss_abs = abs(avg_loss)

        # Profit factor
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        profit_factor = (
            gross_profit / gross_loss if gross_loss > 1e-9 else
            (10.0 if gross_profit > 0 else 0.0)
        )

        # Approximate Sharpe from trade P&Ls
        if len(pnls) >= 2:
            arr = np.array(pnls, dtype=float)
            std = float(np.std(arr, ddof=1))
            mean_pnl = float(np.mean(arr))
            if std > 1e-9:
                sharpe = float(mean_pnl / std * math.sqrt(252))
            elif mean_pnl < 0:
                sharpe = -10.0  # all-loss, no variance → strongly negative
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        return {
            "total_trades": len(data),
            "total_pnl": round(total_pnl, 2),
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate": round(len(wins) / len(data) * 100, 2) if data else 0.0,
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "avg_win_loss_ratio": round(
                avg_win / avg_loss_abs if avg_loss_abs > 1e-9 else
                (10.0 if avg_win > 0 else 0.0), 2
            ),
            "profit_factor": round(profit_factor, 2),
            "max_win": round(max(pnls), 2) if pnls else 0.0,
            "max_loss": round(min(pnls), 2) if pnls else 0.0,
            "avg_bars_held": round(float(np.mean(bars)), 1) if bars else 0.0,
            "sharpe": round(sharpe, 4),
        }

    def per_player_summary(self) -> Dict[str, Dict[str, Any]]:
        """Performance summary broken down by player."""
        from paper_trading_config import PLAYER_IDS

        result = {}
        for pid in PLAYER_IDS:
            trades = self.filter_trades(player_id=pid)
            result[pid] = self.performance_summary(trades)
        return result

    # ── CSV export ────────────────────────────────────────────────────

    def export_csv(self, trades: Optional[List[Dict[str, Any]]] = None) -> str:
        """Export trades to RFC 4180-compliant CSV string."""
        data = trades if trades is not None else self._trades
        if not data:
            return "trade_id,symbol,player_id,direction,quantity,entry_price,exit_price,pnl\n"

        # Collect all keys
        all_keys: set = set()
        for t in data:
            all_keys.update(t.keys())
        sorted_keys = sorted(all_keys)

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(sorted_keys)
        for t in data:
            row = [str(t.get(k, "")) for k in sorted_keys]
            writer.writerow(row)

        return output.getvalue()

    # ── Persistence ───────────────────────────────────────────────────

    def _save(self) -> None:
        if self._file is None:
            return
        try:
            self._file.write_text(
                json.dumps(self._trades, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to save trade journal: {e}")

    def _load(self) -> None:
        if self._file is None or not self._file.exists():
            return
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._trades = data
            logger.info(f"Loaded {len(self._trades)} trades from journal")
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load trade journal: {e}")

    # ── String repr ───────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"TradeJournal(trades={len(self._trades)})"
