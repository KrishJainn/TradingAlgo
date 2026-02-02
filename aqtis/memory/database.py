"""
AQTIS Structured Database (SQLite).

Stores trades, predictions, strategies, market state, and risk events.
Adapted from PRD schemas using SQLite syntax.
"""

import json
import sqlite3
import logging
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StructuredDB:
    """SQLite-backed structured storage for AQTIS."""

    def __init__(self, db_path: str = "aqtis.db"):
        self.db_path = Path(db_path)
        self._initialize_schema()

    @contextmanager
    def get_connection(self):
        """Thread-safe connection context manager with WAL mode."""
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create all tables and indexes."""
        with self.get_connection() as conn:
            conn.executescript("""
                -- Trades table
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    position_size REAL,
                    leverage REAL DEFAULT 1.0,

                    -- Market Context
                    market_regime TEXT,
                    vix_level REAL,
                    sector_rotation_score REAL,
                    liquidity_score REAL,

                    -- Performance
                    pnl REAL,
                    pnl_percent REAL,
                    return_attribution TEXT,
                    max_favorable_excursion REAL,
                    max_adverse_excursion REAL,
                    hold_duration_seconds INTEGER,

                    -- Execution Quality
                    expected_slippage REAL,
                    actual_slippage REAL,
                    execution_venue TEXT,

                    -- Metadata
                    prediction_id TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                -- Predictions table
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_id TEXT PRIMARY KEY,
                    trade_id TEXT,
                    timestamp TEXT NOT NULL,
                    strategy_id TEXT NOT NULL,
                    asset TEXT NOT NULL,

                    -- Predictions
                    predicted_return REAL,
                    predicted_confidence REAL,
                    predicted_hold_seconds INTEGER,
                    predicted_max_drawdown REAL,
                    win_probability REAL,

                    -- Model Info
                    model_ensemble_weights TEXT,
                    primary_model TEXT,
                    feature_importance TEXT,

                    -- Context Snapshot
                    market_features TEXT,
                    similar_historical_trades TEXT,

                    -- Actual Outcomes (populated after trade closes)
                    actual_return REAL,
                    actual_hold_seconds INTEGER,
                    actual_max_drawdown REAL,
                    was_profitable INTEGER,

                    -- Accuracy Metrics
                    return_prediction_error REAL,
                    confidence_calibration_error REAL,
                    direction_correct INTEGER,

                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
                );

                -- Strategies table
                CREATE TABLE IF NOT EXISTS strategies (
                    strategy_id TEXT PRIMARY KEY,
                    strategy_name TEXT NOT NULL,
                    strategy_type TEXT,

                    description TEXT,
                    mathematical_formula TEXT,
                    parameters TEXT,

                    -- Performance Metrics
                    total_trades INTEGER DEFAULT 0,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    sortino_ratio REAL,
                    max_drawdown REAL,
                    avg_return REAL,

                    -- Backtest vs Live
                    backtest_sharpe REAL,
                    live_sharpe REAL,
                    overfitting_score REAL,

                    -- Regime Performance
                    performance_by_regime TEXT,

                    is_active INTEGER DEFAULT 1,
                    last_used_at TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                -- Market State table
                CREATE TABLE IF NOT EXISTS market_state (
                    state_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,

                    vix REAL,
                    realized_vol_20d REAL,
                    vol_regime TEXT,

                    spy_trend_strength REAL,
                    sector_rotation TEXT,
                    breadth_indicators TEXT,

                    avg_bid_ask_spread REAL,
                    market_depth_score REAL,

                    asset_correlation_matrix TEXT,
                    correlation_breakdown INTEGER,

                    upcoming_events TEXT,

                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Risk Events table
                CREATE TABLE IF NOT EXISTS risk_events (
                    event_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    reason TEXT,
                    portfolio_state TEXT,
                    details TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Indexes
                CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset);
                CREATE INDEX IF NOT EXISTS idx_trades_regime ON trades(market_regime);

                CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_predictions_strategy ON predictions(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_trade ON predictions(trade_id);

                CREATE INDEX IF NOT EXISTS idx_market_state_timestamp ON market_state(timestamp);
                CREATE INDEX IF NOT EXISTS idx_risk_events_timestamp ON risk_events(timestamp);
            """)

    # ─────────────────────────────────────────────────────────────────
    # TRADE OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_trade(self, trade: Dict) -> str:
        """Store a new trade. Returns trade_id."""
        trade_id = trade.get("trade_id", str(uuid.uuid4()))
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO trades (
                    trade_id, timestamp, asset, strategy_id, action,
                    entry_price, exit_price, position_size, leverage,
                    market_regime, vix_level, sector_rotation_score, liquidity_score,
                    pnl, pnl_percent, return_attribution,
                    max_favorable_excursion, max_adverse_excursion, hold_duration_seconds,
                    expected_slippage, actual_slippage, execution_venue,
                    prediction_id, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                trade.get("timestamp", datetime.now().isoformat()),
                trade["asset"],
                trade["strategy_id"],
                trade["action"],
                trade.get("entry_price"),
                trade.get("exit_price"),
                trade.get("position_size"),
                trade.get("leverage", 1.0),
                trade.get("market_regime"),
                trade.get("vix_level"),
                trade.get("sector_rotation_score"),
                trade.get("liquidity_score"),
                trade.get("pnl"),
                trade.get("pnl_percent"),
                json.dumps(trade.get("return_attribution")) if trade.get("return_attribution") else None,
                trade.get("max_favorable_excursion"),
                trade.get("max_adverse_excursion"),
                trade.get("hold_duration_seconds"),
                trade.get("expected_slippage"),
                trade.get("actual_slippage"),
                trade.get("execution_venue"),
                trade.get("prediction_id"),
                trade.get("notes"),
            ))
        return trade_id

    def update_trade(self, trade_id: str, updates: Dict):
        """Update trade fields after completion."""
        set_clauses = []
        values = []
        for key, value in updates.items():
            if key == "trade_id":
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            values.append(value)

        set_clauses.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(trade_id)

        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE trades SET {', '.join(set_clauses)} WHERE trade_id = ?",
                values,
            )

    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """Get a single trade by ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM trades WHERE trade_id = ?", (trade_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_trades(
        self,
        strategy_id: str = None,
        asset: str = None,
        market_regime: str = None,
        start_date: str = None,
        end_date: str = None,
        outcome: str = None,
        limit: int = None,
    ) -> List[Dict]:
        """Query trades with optional filters."""
        query = "SELECT * FROM trades WHERE 1=1"
        params: List[Any] = []

        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if asset:
            query += " AND asset = ?"
            params.append(asset)
        if market_regime:
            query += " AND market_regime = ?"
            params.append(market_regime)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        if outcome == "win":
            query += " AND pnl > 0"
        elif outcome == "loss":
            query += " AND pnl <= 0"

        query += " ORDER BY timestamp DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────
    # PREDICTION OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_prediction(self, prediction: Dict) -> str:
        """Store a new prediction. Returns prediction_id."""
        prediction_id = prediction.get("prediction_id", str(uuid.uuid4()))
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO predictions (
                    prediction_id, trade_id, timestamp, strategy_id, asset,
                    predicted_return, predicted_confidence, predicted_hold_seconds,
                    predicted_max_drawdown, win_probability,
                    model_ensemble_weights, primary_model, feature_importance,
                    market_features, similar_historical_trades
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction_id,
                prediction.get("trade_id"),
                prediction.get("timestamp", datetime.now().isoformat()),
                prediction["strategy_id"],
                prediction["asset"],
                prediction.get("predicted_return"),
                prediction.get("predicted_confidence"),
                prediction.get("predicted_hold_seconds"),
                prediction.get("predicted_max_drawdown"),
                prediction.get("win_probability"),
                json.dumps(prediction.get("model_ensemble_weights")) if prediction.get("model_ensemble_weights") else None,
                prediction.get("primary_model"),
                json.dumps(prediction.get("feature_importance")) if prediction.get("feature_importance") else None,
                json.dumps(prediction.get("market_features")) if prediction.get("market_features") else None,
                json.dumps(prediction.get("similar_historical_trades")) if prediction.get("similar_historical_trades") else None,
            ))
        return prediction_id

    def update_prediction(self, prediction_id: str, updates: Dict):
        """Update prediction with actual outcome."""
        set_clauses = []
        values = []
        for key, value in updates.items():
            if key == "prediction_id":
                continue
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            set_clauses.append(f"{key} = ?")
            values.append(value)
        values.append(prediction_id)

        with self.get_connection() as conn:
            conn.execute(
                f"UPDATE predictions SET {', '.join(set_clauses)} WHERE prediction_id = ?",
                values,
            )

    def get_prediction(self, prediction_id: str) -> Optional[Dict]:
        """Get a single prediction by ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM predictions WHERE prediction_id = ?", (prediction_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_predictions(
        self,
        strategy_id: str = None,
        asset: str = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = None,
    ) -> List[Dict]:
        """Query predictions with optional filters."""
        query = "SELECT * FROM predictions WHERE 1=1"
        params: List[Any] = []

        if strategy_id:
            query += " AND strategy_id = ?"
            params.append(strategy_id)
        if asset:
            query += " AND asset = ?"
            params.append(asset)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC"
        if limit:
            query += " LIMIT ?"
            params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_prediction_accuracy(self, lookback_days: int = 30) -> Dict:
        """Calculate prediction accuracy over recent period."""
        cutoff = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        with self.get_connection() as conn:
            rows = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(return_prediction_error) as avg_return_error,
                    AVG(confidence_calibration_error) as avg_calibration_error
                FROM predictions
                WHERE timestamp >= ? AND direction_correct IS NOT NULL
            """, (cutoff,)).fetchone()

            if not rows or rows["total"] == 0:
                return {"total": 0, "accuracy": 0.0, "avg_return_error": 0.0, "avg_calibration_error": 0.0}

            return {
                "total": rows["total"],
                "accuracy": rows["correct"] / rows["total"],
                "avg_return_error": rows["avg_return_error"] or 0.0,
                "avg_calibration_error": rows["avg_calibration_error"] or 0.0,
            }

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_strategy(self, strategy: Dict) -> str:
        """Store or update a strategy."""
        strategy_id = strategy["strategy_id"]
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO strategies (
                    strategy_id, strategy_name, strategy_type,
                    description, mathematical_formula, parameters,
                    total_trades, win_rate, sharpe_ratio, sortino_ratio,
                    max_drawdown, avg_return,
                    backtest_sharpe, live_sharpe, overfitting_score,
                    performance_by_regime,
                    is_active, last_used_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                strategy_id,
                strategy.get("strategy_name", strategy_id),
                strategy.get("strategy_type"),
                strategy.get("description"),
                strategy.get("mathematical_formula"),
                json.dumps(strategy.get("parameters")) if strategy.get("parameters") else None,
                strategy.get("total_trades", 0),
                strategy.get("win_rate"),
                strategy.get("sharpe_ratio"),
                strategy.get("sortino_ratio"),
                strategy.get("max_drawdown"),
                strategy.get("avg_return"),
                strategy.get("backtest_sharpe"),
                strategy.get("live_sharpe"),
                strategy.get("overfitting_score"),
                json.dumps(strategy.get("performance_by_regime")) if strategy.get("performance_by_regime") else None,
                1 if strategy.get("is_active", True) else 0,
                strategy.get("last_used_at"),
            ))
        return strategy_id

    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Get a single strategy by ID."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM strategies WHERE strategy_id = ?", (strategy_id,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            if result.get("parameters"):
                result["parameters"] = json.loads(result["parameters"])
            if result.get("performance_by_regime"):
                result["performance_by_regime"] = json.loads(result["performance_by_regime"])
            return result

    def get_active_strategies(self) -> List[Dict]:
        """Get all active strategies."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM strategies WHERE is_active = 1 ORDER BY sharpe_ratio DESC"
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                if d.get("parameters"):
                    d["parameters"] = json.loads(d["parameters"])
                if d.get("performance_by_regime"):
                    d["performance_by_regime"] = json.loads(d["performance_by_regime"])
                results.append(d)
            return results

    # ─────────────────────────────────────────────────────────────────
    # MARKET STATE OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_market_state(self, state: Dict) -> str:
        """Store a market state snapshot."""
        state_id = state.get("state_id", str(uuid.uuid4()))
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO market_state (
                    state_id, timestamp,
                    vix, realized_vol_20d, vol_regime,
                    spy_trend_strength, sector_rotation, breadth_indicators,
                    avg_bid_ask_spread, market_depth_score,
                    asset_correlation_matrix, correlation_breakdown,
                    upcoming_events
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                state_id,
                state.get("timestamp", datetime.now().isoformat()),
                state.get("vix"),
                state.get("realized_vol_20d"),
                state.get("vol_regime"),
                state.get("spy_trend_strength"),
                json.dumps(state.get("sector_rotation")) if state.get("sector_rotation") else None,
                json.dumps(state.get("breadth_indicators")) if state.get("breadth_indicators") else None,
                state.get("avg_bid_ask_spread"),
                state.get("market_depth_score"),
                json.dumps(state.get("asset_correlation_matrix")) if state.get("asset_correlation_matrix") else None,
                1 if state.get("correlation_breakdown") else 0,
                json.dumps(state.get("upcoming_events")) if state.get("upcoming_events") else None,
            ))
        return state_id

    def get_latest_market_state(self) -> Optional[Dict]:
        """Get the most recent market state."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM market_state ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row else None

    def get_market_state_history(self, days: int = 30) -> List[Dict]:
        """Get market state history."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM market_state WHERE timestamp >= ? ORDER BY timestamp DESC",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────
    # RISK EVENT OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_risk_event(self, event: Dict) -> str:
        """Store a risk event."""
        event_id = event.get("event_id", str(uuid.uuid4()))
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO risk_events (
                    event_id, timestamp, event_type, reason,
                    portfolio_state, details
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                event_id,
                event.get("timestamp", datetime.now().isoformat()),
                event["event_type"],
                event.get("reason"),
                json.dumps(event.get("portfolio_state")) if event.get("portfolio_state") else None,
                json.dumps(event.get("details")) if event.get("details") else None,
            ))
        return event_id

    def get_risk_events(self, days: int = 7) -> List[Dict]:
        """Get recent risk events."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM risk_events WHERE timestamp >= ? ORDER BY timestamp DESC",
                (cutoff,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────
    # AGGREGATION QUERIES
    # ─────────────────────────────────────────────────────────────────

    def get_strategy_performance(self, strategy_id: str, regime: str = None) -> Dict:
        """Calculate aggregate performance for a strategy."""
        query = "SELECT * FROM trades WHERE strategy_id = ?"
        params: List[Any] = [strategy_id]
        if regime:
            query += " AND market_regime = ?"
            params.append(regime)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            trades = [dict(r) for r in rows]

        if not trades:
            return {"strategy_id": strategy_id, "total_trades": 0}

        pnls = [t["pnl"] or 0 for t in trades]
        wins = sum(1 for p in pnls if p > 0)

        import numpy as np
        pnl_array = np.array(pnls)
        returns = np.array([t["pnl_percent"] or 0 for t in trades])

        sharpe = 0.0
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(252))

        cumulative = np.cumsum(pnl_array)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - peak
        max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

        return {
            "strategy_id": strategy_id,
            "total_trades": len(trades),
            "win_rate": wins / len(trades),
            "total_pnl": sum(pnls),
            "avg_pnl": float(np.mean(pnl_array)),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "avg_return": float(np.mean(returns)),
            "regime": regime,
        }

    def get_daily_pnl(self, date: str = None) -> float:
        """Get total P&L for a specific day."""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE timestamp LIKE ?",
                (f"{date}%",),
            ).fetchone()
            return row["total"] if row else 0.0

    def get_stats(self) -> Dict:
        """Get overall database statistics."""
        with self.get_connection() as conn:
            trade_count = conn.execute("SELECT COUNT(*) as c FROM trades").fetchone()["c"]
            pred_count = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
            strat_count = conn.execute("SELECT COUNT(*) as c FROM strategies").fetchone()["c"]
            state_count = conn.execute("SELECT COUNT(*) as c FROM market_state").fetchone()["c"]
            event_count = conn.execute("SELECT COUNT(*) as c FROM risk_events").fetchone()["c"]

            return {
                "trades": trade_count,
                "predictions": pred_count,
                "strategies": strat_count,
                "market_states": state_count,
                "risk_events": event_count,
            }
