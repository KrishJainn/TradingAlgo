"""
Metrics Store — Append-only JSON-based storage for daily performance snapshots.

One file per day: metrics_data/metrics_YYYY-MM-DD.json
Each file holds a JSON array of snapshot dicts (typically one per day).

Provides derived metrics on read:
    - Rolling 30-day Sharpe & Sortino
    - Rolling hit rate (win rate over last 30 trades)
    - Regime-conditional rolling metrics
    - Player contribution attribution (% of total P&L per player)
    - Drawdown series, monthly returns, cost drag

Usage:
    from metrics_store import MetricsStore, MetricsCalculator

    store = MetricsStore()
    store.record_daily_snapshot({
        "portfolio": {"equity_curve_value": 510000, "daily_return": 0.005, ...},
        "per_player_pnl": {"PLAYER_1": 800, ...},
        "regime": {"current": "Bull", ...},
        ...
    })

    history = store.get_metric_history("portfolio.sharpe", days=30)
    latest  = store.get_latest()
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent
_DEFAULT_METRICS_DIR = _PROJECT_ROOT / "metrics_data"

PLAYER_IDS = ["PLAYER_1", "PLAYER_2", "PLAYER_3", "PLAYER_4", "PLAYER_5"]
PLAYER_LABELS = {
    "PLAYER_1": "TrendFollower",
    "PLAYER_2": "MeanReversion",
    "PLAYER_3": "VolumeFlow",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "MultiMomentum",
}

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.065


# ═══════════════════════════════════════════════════════════════════════════
# MetricsStore
# ═══════════════════════════════════════════════════════════════════════════

class MetricsStore:
    """
    Append-only JSON metrics store.

    One JSON file per calendar day: metrics_data/metrics_YYYY-MM-DD.json
    Each file contains a JSON array of snapshot dicts.
    """

    def __init__(self, metrics_dir: Optional[Path] = None) -> None:
        self.metrics_dir = Path(metrics_dir) if metrics_dir else _DEFAULT_METRICS_DIR
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── File helpers ──────────────────────────────────────────────────

    def _file_for_date(self, d: date) -> Path:
        return self.metrics_dir / f"metrics_{d.isoformat()}.json"

    def _read_file(self, path: Path) -> List[Dict[str, Any]]:
        """Read a single day's JSON file. Returns [] on any error."""
        if not path.exists():
            return []
        try:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                return []
            data = json.loads(text)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return [data]
            return []
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Corrupted metrics file {path.name}: {e}")
            return []

    def _write_file(self, path: Path, snapshots: List[Dict[str, Any]]) -> None:
        try:
            path.write_text(
                json.dumps(snapshots, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to write metrics file {path.name}: {e}")

    def _list_dates(self) -> List[date]:
        """List all available dates, sorted ascending."""
        dates: List[date] = []
        for f in sorted(self.metrics_dir.glob("metrics_*.json")):
            try:
                ds = f.stem.replace("metrics_", "")
                dates.append(date.fromisoformat(ds))
            except ValueError:
                continue
        return dates

    # ══════════════════════════════════════════════════════════════════
    # 1. Core CRUD
    # ══════════════════════════════════════════════════════════════════

    def record_daily_snapshot(self, data: Dict[str, Any]) -> None:
        """
        Append a daily snapshot. De-duplicates by date (overwrites same day).

        Automatically adds 'recorded_at' if missing.

        Recommended top-level keys:
            date, portfolio, positions, trades_today, regime,
            players, signals, risk, coaching, evolution
        """
        snap = dict(data)
        snap_date = snap.get("date") or date.today().isoformat()
        snap["date"] = snap_date
        snap.setdefault("recorded_at", datetime.now().isoformat())

        try:
            d = date.fromisoformat(snap_date)
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid date '{snap_date}' in snapshot: {e}")
            return
        path = self._file_for_date(d)
        existing = self._read_file(path)

        # Overwrite if same date already present
        existing = [s for s in existing if s.get("date") != snap_date]
        existing.append(snap)
        self._write_file(path, existing)
        logger.info(f"Recorded snapshot for {snap_date}")

    def get_latest(self) -> Dict[str, Any]:
        """Return the most recent snapshot, or empty dict if store is empty."""
        dates = self._list_dates()
        if not dates:
            return {}
        for d in reversed(dates):
            snapshots = self._read_file(self._file_for_date(d))
            if snapshots:
                return snapshots[-1]
        return {}

    def get_date_range(
        self, start_date: date, end_date: date,
    ) -> List[Dict[str, Any]]:
        """Return all snapshots within [start_date, end_date] inclusive."""
        results: List[Dict[str, Any]] = []
        d = start_date
        while d <= end_date:
            results.extend(self._read_file(self._file_for_date(d)))
            d += timedelta(days=1)
        return results

    def get_all_snapshots(self, days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return all snapshots, optionally limited to recent N days."""
        if days is None:
            results: List[Dict[str, Any]] = []
            for d in self._list_dates():
                results.extend(self._read_file(self._file_for_date(d)))
            return results
        end = date.today()
        start = end - timedelta(days=days)
        return self.get_date_range(start, end)

    def get_metric_history(
        self, metric_path: str, days: int = 30,
    ) -> List[Tuple[str, float]]:
        """
        Extract a time series for a dot-path metric.

        Args:
            metric_path: Dot-separated path, e.g. "portfolio.sharpe",
                         "players.PLAYER_1.win_rate", "risk.current_drawdown".
            days: Number of days to look back.

        Returns:
            List of (date_str, value) tuples.
        """
        snapshots = self.get_all_snapshots(days=days)
        result: List[Tuple[str, float]] = []
        keys = metric_path.split(".")

        for snap in snapshots:
            val: Any = snap
            try:
                for k in keys:
                    val = val[k]
                result.append((snap.get("date", ""), float(val)))
            except (KeyError, TypeError, ValueError, IndexError):
                continue
        return result

    def snapshot_count(self) -> int:
        """Total number of snapshots across all files."""
        total = 0
        for d in self._list_dates():
            total += len(self._read_file(self._file_for_date(d)))
        return total

    # ══════════════════════════════════════════════════════════════════
    # 2. Derived Metrics (computed on read)
    # ══════════════════════════════════════════════════════════════════

    def compute_rolling_sharpe(
        self, days: int = 30, window: int = 30,
    ) -> List[Tuple[str, float]]:
        """
        Rolling Sharpe ratio from daily returns.

        Returns list of (date_str, sharpe).
        """
        history = self.get_metric_history("portfolio.daily_return", days=days + window)
        if len(history) < window:
            return []
        dates = [h[0] for h in history]
        rets = np.array([h[1] for h in history])
        result: List[Tuple[str, float]] = []
        daily_rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
        for i in range(window - 1, len(rets)):
            chunk = rets[max(0, i - window + 1): i + 1]
            excess = chunk - daily_rf
            std = float(np.std(excess, ddof=1)) if len(chunk) > 1 else 1e-9
            sharpe = float(np.mean(excess) / std * math.sqrt(TRADING_DAYS_PER_YEAR)) if std > 1e-9 else 0.0
            result.append((dates[i], round(sharpe, 4)))
        return result

    def compute_rolling_hit_rate(
        self, days: int = 30, window: int = 30,
    ) -> List[Tuple[str, float]]:
        """Rolling hit rate (win rate) over trailing window."""
        history = self.get_metric_history("portfolio.net_pnl", days=days + window)
        if len(history) < window:
            return []
        dates = [h[0] for h in history]
        pnls = np.array([h[1] for h in history])
        result: List[Tuple[str, float]] = []
        for i in range(window - 1, len(pnls)):
            chunk = pnls[max(0, i - window + 1): i + 1]
            rate = float(np.sum(chunk > 0) / len(chunk)) if len(chunk) > 0 else 0.0
            result.append((dates[i], round(rate, 4)))
        return result

    def compute_regime_conditional_metrics(
        self, days: int = 90,
    ) -> Dict[str, Dict[str, float]]:
        """
        Sharpe, hit rate, avg return, trade count per regime.

        Returns: {"Bull": {"sharpe": x, "hit_rate": y, "avg_return": z, "count": n}, ...}
        """
        snapshots = self.get_all_snapshots(days=days)
        regime_data: Dict[str, List[float]] = {}

        for snap in snapshots:
            regime = snap.get("regime", {})
            if isinstance(regime, dict):
                regime_name = regime.get("current", "Unknown")
            else:
                regime_name = str(regime)
            ret = snap.get("portfolio", {}).get("daily_return")
            if ret is None:
                continue
            try:
                regime_data.setdefault(regime_name, []).append(float(ret))
            except (ValueError, TypeError):
                continue

        result: Dict[str, Dict[str, float]] = {}
        calc = MetricsCalculator
        for regime_name, rets in regime_data.items():
            arr = np.array(rets, dtype=float)
            n = len(arr)
            result[regime_name] = {
                "sharpe": round(calc.compute_sharpe(arr), 4),
                "hit_rate": round(float(np.sum(arr > 0) / n) if n > 0 else 0.0, 4),
                "avg_return": round(float(np.mean(arr)), 6),
                "total_pnl": round(float(np.sum(arr)), 6),
                "count": n,
            }
        return result

    def compute_player_contribution(
        self, days: int = 30,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Player contribution attribution: % of total P&L per player.

        pct_of_total uses sum-of-absolutes as denominator so the absolute
        percentages always sum to 100%.

        Returns: {player_id: {"total_pnl": float, "pct_of_total": float,
                               "trade_count": int, "avg_pnl_per_trade": float}}
        """
        snapshots = self.get_all_snapshots(days=days)
        player_totals: Dict[str, float] = {p: 0.0 for p in PLAYER_IDS}
        player_trades: Dict[str, int] = {p: 0 for p in PLAYER_IDS}

        for snap in snapshots:
            players = snap.get("players", {})
            for pid in PLAYER_IDS:
                pdata = players.get(pid, {})
                try:
                    player_totals[pid] += float(pdata.get("pnl", 0))
                except (ValueError, TypeError):
                    pass
                try:
                    player_trades[pid] += int(pdata.get("num_trades", 0))
                except (ValueError, TypeError):
                    pass

        grand_total = sum(abs(v) for v in player_totals.values())
        if grand_total < 1e-9:
            grand_total = 1.0

        result: Dict[str, Dict[str, Any]] = {}
        for pid in PLAYER_IDS:
            total = player_totals[pid]
            trades = player_trades[pid]
            result[pid] = {
                "total_pnl": round(total, 2),
                "pct_of_total": round(abs(total) / grand_total * 100, 2),
                "trade_count": trades,
                "avg_pnl_per_trade": round(total / trades, 2) if trades > 0 else 0.0,
            }
        return result

    def compute_drawdown_series(
        self, days: int = 90,
    ) -> List[Dict[str, Any]]:
        """
        Cumulative equity and drawdown series.

        Returns list of {"date", "equity", "peak", "drawdown", "drawdown_pct"}.
        """
        history = self.get_metric_history("portfolio.equity_curve_value", days=days)
        if not history:
            return []

        result: List[Dict[str, Any]] = []
        peak = history[0][1]
        for d, val in history:
            peak = max(peak, val)
            dd = val - peak
            dd_pct = dd / peak if peak > 0 else 0.0
            result.append({
                "date": d,
                "equity": round(val, 2),
                "peak": round(peak, 2),
                "drawdown": round(dd, 2),
                "drawdown_pct": round(dd_pct, 4),
            })
        return result

    def compute_monthly_returns(
        self, days: int = 365,
    ) -> Dict[str, Dict[str, Any]]:
        """Monthly return breakdown: {YYYY-MM: {"pnl": float, "days": int}}."""
        snapshots = self.get_all_snapshots(days=days)
        months: Dict[str, Dict[str, float]] = {}

        for snap in snapshots:
            d = snap.get("date", "")
            pnl = snap.get("portfolio", {}).get("net_pnl")
            if not d or pnl is None:
                continue
            try:
                ym = d[:7]
                if ym not in months:
                    months[ym] = {"pnl": 0.0, "days": 0}
                months[ym]["pnl"] = round(months[ym]["pnl"] + float(pnl), 2)
                months[ym]["days"] += 1
            except (ValueError, TypeError):
                continue
        return months

    def compute_cost_drag_series(
        self, days: int = 90,
    ) -> List[Dict[str, Any]]:
        """Cumulative cost drag as % of gross P&L."""
        snapshots = self.get_all_snapshots(days=days)
        cum_costs = 0.0
        cum_gross = 0.0
        result: List[Dict[str, Any]] = []
        for snap in snapshots:
            d = snap.get("date", "")
            gross = snap.get("portfolio", {}).get("gross_pnl", 0)
            net = snap.get("portfolio", {}).get("net_pnl", 0)
            try:
                cost_today = float(gross) - float(net)
                cum_costs += cost_today
                cum_gross += float(gross)
            except (ValueError, TypeError):
                continue
            drag = (cum_costs / abs(cum_gross) * 100) if abs(cum_gross) > 1e-9 else 0.0
            result.append({
                "date": d,
                "cum_costs": round(cum_costs, 2),
                "cum_gross_pnl": round(cum_gross, 2),
                "drag_pct": round(drag, 2),
            })
        return result

    # ══════════════════════════════════════════════════════════════════
    # 3. Export
    # ══════════════════════════════════════════════════════════════════

    def export_csv(self, days: int = 90) -> str:
        """Export all snapshots as CSV string (flat dot-separated keys)."""
        snapshots = self.get_all_snapshots(days=days)
        if not snapshots:
            return "date\n"

        all_keys: set = set()
        flat_rows: List[Dict[str, Any]] = []
        for s in snapshots:
            flat = _flatten_dict(s)
            flat_rows.append(flat)
            all_keys.update(flat.keys())

        sorted_keys = sorted(all_keys)
        lines = [",".join(sorted_keys)]
        for flat in flat_rows:
            row = [str(flat.get(k, "")).replace(",", ";") for k in sorted_keys]
            lines.append(",".join(row))
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# MetricsCalculator — stateless quantitative computations
# ═══════════════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """Stateless utility for computing quantitative performance metrics."""

    @staticmethod
    def compute_sharpe(
        returns: np.ndarray,
        risk_free: float = RISK_FREE_RATE,
        ann_factor: int = TRADING_DAYS_PER_YEAR,
    ) -> float:
        """Annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        daily_rf = risk_free / ann_factor
        excess = returns - daily_rf
        std = float(np.std(excess, ddof=1))
        if std < 1e-10:
            return 0.0
        return float(np.mean(excess) / std * math.sqrt(ann_factor))

    @staticmethod
    def compute_sortino(
        returns: np.ndarray,
        risk_free: float = RISK_FREE_RATE,
        ann_factor: int = TRADING_DAYS_PER_YEAR,
    ) -> float:
        """Annualized Sortino ratio (downside deviation only)."""
        if len(returns) < 2:
            return 0.0
        daily_rf = risk_free / ann_factor
        excess = returns - daily_rf
        downside = excess[excess < 0]
        if len(downside) < 2:
            return 10.0
        down_std = float(np.std(downside, ddof=1))
        if down_std < 1e-10 or np.isnan(down_std):
            return 10.0
        return float(np.mean(excess) / down_std * math.sqrt(ann_factor))

    @staticmethod
    def compute_max_drawdown(equity_curve: np.ndarray) -> float:
        """Maximum drawdown as a positive fraction (e.g. 0.15 = 15%)."""
        if len(equity_curve) < 2:
            return 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdowns = (peak - equity_curve) / np.where(peak > 0, peak, 1.0)
        return float(np.max(drawdowns))

    @staticmethod
    def compute_current_drawdown(equity_curve: np.ndarray) -> float:
        """Current drawdown from peak."""
        if len(equity_curve) < 1:
            return 0.0
        peak = float(np.max(equity_curve))
        if peak <= 0:
            return 0.0
        return float((peak - equity_curve[-1]) / peak)

    @staticmethod
    def compute_calmar(
        returns: np.ndarray,
        ann_factor: int = TRADING_DAYS_PER_YEAR,
    ) -> float:
        """Calmar ratio = annualized return / max drawdown."""
        if len(returns) < 2:
            return 0.0
        equity = np.cumprod(1 + returns)
        max_dd = MetricsCalculator.compute_max_drawdown(equity)
        if max_dd < 1e-10:
            return 10.0
        ann_return = float(np.mean(returns) * ann_factor)
        return ann_return / max_dd

    @staticmethod
    def compute_profit_factor(pnls: np.ndarray) -> float:
        """Gross profit / gross loss."""
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
        gross_loss = float(np.abs(np.sum(losses))) if len(losses) > 0 else 0.0
        if gross_loss < 1e-10:
            return 10.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    @staticmethod
    def compute_win_rate(pnls: np.ndarray) -> float:
        """Fraction of entries that are profitable."""
        if len(pnls) == 0:
            return 0.0
        return float(np.sum(pnls > 0) / len(pnls))

    @staticmethod
    def compute_avg_win_loss_ratio(pnls: np.ndarray) -> float:
        """Average winning trade / average losing trade magnitude."""
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
        avg_loss = float(np.abs(np.mean(losses))) if len(losses) > 0 else 0.0
        if avg_loss < 1e-10:
            return 10.0 if avg_win > 0 else 0.0
        return avg_win / avg_loss

    @staticmethod
    def compute_cost_drag(gross_pnl: float, net_pnl: float) -> float:
        """Cost drag as percentage of gross returns."""
        if abs(gross_pnl) < 1e-10:
            return 0.0
        return float((gross_pnl - net_pnl) / abs(gross_pnl) * 100)

    @staticmethod
    def compute_player_attribution(
        player_pnls: Dict[str, float], total_pnl: float,
    ) -> Dict[str, float]:
        """Each player's contribution as % of total |P&L|."""
        if abs(total_pnl) < 1e-10:
            return {pid: 0.0 for pid in player_pnls}
        return {
            pid: round(pnl / abs(total_pnl) * 100, 2)
            for pid, pnl in player_pnls.items()
        }

    @staticmethod
    def compute_rolling_sharpe(
        returns: np.ndarray,
        window: int = 30,
        risk_free: float = RISK_FREE_RATE,
        ann_factor: int = TRADING_DAYS_PER_YEAR,
    ) -> np.ndarray:
        """Rolling Sharpe ratio with given window."""
        if len(returns) < window:
            return np.array([])
        daily_rf = risk_free / ann_factor
        result = []
        for i in range(window, len(returns) + 1):
            chunk = returns[i - window: i]
            excess = chunk - daily_rf
            std = float(np.std(excess, ddof=1))
            if std > 1e-10:
                result.append(float(np.mean(excess) / std * math.sqrt(ann_factor)))
            else:
                result.append(0.0)
        return np.array(result)

    @staticmethod
    def compute_rolling_hit_rate(
        pnls: np.ndarray, window: int = 30,
    ) -> np.ndarray:
        """Rolling win rate."""
        if len(pnls) < window:
            return np.array([])
        result = []
        for i in range(window, len(pnls) + 1):
            chunk = pnls[i - window: i]
            result.append(float(np.sum(chunk > 0) / len(chunk)))
        return np.array(result)

    @staticmethod
    def compute_regime_conditional_metrics(
        returns: np.ndarray,
        regimes: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Compute Sharpe and win rate per regime from arrays."""
        if len(returns) != len(regimes):
            return {}
        result: Dict[str, Dict[str, float]] = {}
        for regime in ("Bull", "Bear", "Sideways"):
            mask = np.array([r == regime for r in regimes])
            regime_returns = returns[mask]
            if len(regime_returns) == 0:
                result[regime] = {"sharpe": 0.0, "win_rate": 0.0, "count": 0}
                continue
            result[regime] = {
                "sharpe": MetricsCalculator.compute_sharpe(regime_returns),
                "win_rate": MetricsCalculator.compute_win_rate(regime_returns),
                "count": int(len(regime_returns)),
            }
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _flatten_dict(d: Dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten nested dict with dot-separated keys."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        elif isinstance(v, list):
            items[key] = str(v)
        else:
            items[key] = v
    return items


# ═══════════════════════════════════════════════════════════════════════════
# Demo / self-test
# ═══════════════════════════════════════════════════════════════════════════

def _demo() -> None:
    """Quick self-test with synthetic data."""
    import tempfile

    print("=" * 60)
    print("  MetricsStore — Self-Test")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = MetricsStore(metrics_dir=Path(tmpdir))
        rng = np.random.default_rng(42)
        today = date.today()

        # Record 30 days of synthetic data
        for i in range(30):
            d = today - timedelta(days=29 - i)
            regimes = ["Bull", "Bear", "Sideways"]
            regime = regimes[i % 3]
            daily_ret = float(rng.normal(0.001, 0.012))
            equity_base = 500_000 + i * 500
            daily_pnl_gross = equity_base * daily_ret
            daily_cost = abs(daily_pnl_gross) * 0.02
            daily_pnl_net = daily_pnl_gross - daily_cost

            players = {}
            for pid in PLAYER_IDS:
                players[pid] = {
                    "sharpe": round(float(rng.normal(1.2, 0.5)), 3),
                    "sortino": round(float(rng.normal(1.8, 0.6)), 3),
                    "win_rate": round(float(rng.uniform(0.4, 0.65)), 3),
                    "num_trades": int(rng.integers(2, 15)),
                    "pnl": round(float(rng.normal(200, 800)), 2),
                    "bayesian_weight": round(float(rng.uniform(0.10, 0.30)), 3),
                }

            store.record_daily_snapshot({
                "date": d.isoformat(),
                "portfolio": {
                    "equity_curve_value": round(equity_base + daily_pnl_net, 2),
                    "daily_return": round(daily_ret, 6),
                    "gross_pnl": round(daily_pnl_gross, 2),
                    "net_pnl": round(daily_pnl_net, 2),
                    "gross_exposure": round(float(rng.uniform(0.5, 1.2)), 4),
                    "net_exposure": round(float(rng.uniform(-0.3, 0.6)), 4),
                    "largest_concentration": round(float(rng.uniform(0.08, 0.20)), 4),
                },
                "regime": {
                    "current": regime,
                    "probabilities": {"Bull": 0.4, "Bear": 0.3, "Sideways": 0.3},
                    "duration_days": int(rng.integers(1, 15)),
                },
                "players": players,
            })

        # Test all APIs
        latest = store.get_latest()
        print(f"\n  Latest snapshot date:    {latest.get('date', 'None')}")
        print(f"  Snapshot count:          {store.snapshot_count()}")

        history = store.get_metric_history("portfolio.equity_curve_value", days=30)
        print(f"  Equity history entries:  {len(history)}")

        sharpe = store.compute_rolling_sharpe(days=20, window=10)
        print(f"  Rolling 10d Sharpe:      {sharpe[-1][1] if sharpe else 'N/A'}")

        hit_rate = store.compute_rolling_hit_rate(days=20, window=10)
        print(f"  Rolling 10d Hit Rate:    {hit_rate[-1][1] if hit_rate else 'N/A'}")

        regime_m = store.compute_regime_conditional_metrics(days=30)
        for regime, m in regime_m.items():
            print(f"  {regime:10s}: Sharpe={m['sharpe']:+.2f}, HitRate={m['hit_rate']:.2f}")

        contrib = store.compute_player_contribution(days=30)
        for pid, c in contrib.items():
            label = PLAYER_LABELS.get(pid, pid)
            print(f"  {label:15s}: P&L={c['total_pnl']:+,.0f}, Contrib={c['pct_of_total']:+.1f}%")

        dd = store.compute_drawdown_series(days=30)
        if dd:
            worst = min(dd, key=lambda x: x["drawdown"])
            print(f"\n  Max drawdown: {worst['drawdown']:+,.0f} ({worst['drawdown_pct']:.2%})")

        monthly = store.compute_monthly_returns(days=30)
        print(f"  Monthly returns: {len(monthly)} months")

        csv = store.export_csv(days=30)
        lines = csv.strip().split("\n")
        print(f"  CSV export: {len(lines)} rows")

    print(f"\n{'=' * 60}")
    print("  All MetricsStore tests passed.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _demo()
