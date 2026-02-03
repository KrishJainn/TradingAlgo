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

                -- Coach sessions (from 5-player coach model)
                CREATE TABLE IF NOT EXISTS coach_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    strategy_id TEXT,
                    trading_date TEXT,
                    regime TEXT,
                    advice TEXT,
                    weight_changes INTEGER DEFAULT 0,
                    entry_change REAL DEFAULT 0,
                    exit_change REAL DEFAULT 0,
                    mistakes_json TEXT,
                    weight_recs_json TEXT,
                    opportunities_json TEXT,
                    source TEXT DEFAULT 'rule_based',
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Strategy snapshots (from 5-player model)
                CREATE TABLE IF NOT EXISTS strategy_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT,
                    snapshot_type TEXT,
                    label TEXT,
                    weights_json TEXT,
                    entry_threshold REAL,
                    exit_threshold REAL,
                    stop_loss_multiplier REAL,
                    total_trades INTEGER DEFAULT 0,
                    win_rate REAL,
                    net_pnl REAL,
                    sharpe REAL,
                    sortino REAL,
                    max_drawdown REAL,
                    profit_factor REAL,
                    regime_performance_json TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Indicator-regime performance stats (from 5-player model)
                CREATE TABLE IF NOT EXISTS indicator_regime_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    indicator TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    avg_pnl REAL,
                    win_rate REAL,
                    contribution_score REAL,
                    predictive_accuracy REAL,
                    consistency_score REAL,
                    information_ratio REAL,
                    updated_at TEXT
                );

                -- Simulation runs (cross-run learning from 5-player model)
                CREATE TABLE IF NOT EXISTS sim_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_number INTEGER,
                    session_id TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    days INTEGER,
                    symbols INTEGER,
                    team_pnl REAL,
                    team_return REAL,
                    config_json TEXT,
                    notes TEXT
                );

                -- Indicator scores per trade (deep diagnostics)
                CREATE TABLE IF NOT EXISTS indicator_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT,
                    indicator TEXT NOT NULL,
                    weight_used REAL,
                    correlation_with_pnl REAL,
                    predictive_accuracy REAL,
                    signal_quality_score REAL,
                    verdict TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
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

                CREATE INDEX IF NOT EXISTS idx_coach_strategy ON coach_sessions(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_coach_date ON coach_sessions(trading_date);
                CREATE INDEX IF NOT EXISTS idx_strategy_snap ON strategy_snapshots(strategy_id);
                CREATE INDEX IF NOT EXISTS idx_ind_regime ON indicator_regime_stats(indicator, regime);
                CREATE INDEX IF NOT EXISTS idx_sim_runs ON sim_runs(run_number);
                CREATE INDEX IF NOT EXISTS idx_ind_scores_trade ON indicator_scores(trade_id);

                -- Live Paper Trading (5-Player Coach)
                CREATE TABLE IF NOT EXISTS live_player_states (
                    player_id TEXT PRIMARY KEY,
                    label TEXT,
                    weights_json TEXT,
                    entry_threshold REAL,
                    exit_threshold REAL,
                    min_hold_bars INTEGER DEFAULT 4,
                    equity REAL DEFAULT 100000.0,
                    day_pnl REAL DEFAULT 0.0,
                    strategy_version TEXT DEFAULT 'v1.0',
                    positions_json TEXT,
                    bars_held_json TEXT,
                    prev_si_json TEXT,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS live_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    player_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL,
                    exit_price REAL,
                    quantity INTEGER,
                    pnl REAL,
                    si_value REAL,
                    exit_si REAL,
                    exit_reason TEXT,
                    entry_time TEXT,
                    exit_time TEXT,
                    atr REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS live_equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    equity REAL NOT NULL,
                    day_pnl REAL,
                    trades INTEGER,
                    version TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS live_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT,
                    ended_at TEXT,
                    initial_equity REAL,
                    final_equity REAL,
                    total_trades INTEGER,
                    market_regime TEXT,
                    notes TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_live_trades_player ON live_trades(player_id);
                CREATE INDEX IF NOT EXISTS idx_live_trades_time ON live_trades(entry_time);
                CREATE INDEX IF NOT EXISTS idx_live_equity_player ON live_equity_history(player_id);
                CREATE INDEX IF NOT EXISTS idx_live_equity_date ON live_equity_history(date);
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
            coach_count = conn.execute("SELECT COUNT(*) as c FROM coach_sessions").fetchone()["c"]
            snap_count = conn.execute("SELECT COUNT(*) as c FROM strategy_snapshots").fetchone()["c"]
            run_count = conn.execute("SELECT COUNT(*) as c FROM sim_runs").fetchone()["c"]
            ind_stat_count = conn.execute("SELECT COUNT(*) as c FROM indicator_regime_stats").fetchone()["c"]

            return {
                "trades": trade_count,
                "predictions": pred_count,
                "strategies": strat_count,
                "market_states": state_count,
                "risk_events": event_count,
                "coach_sessions": coach_count,
                "strategy_snapshots": snap_count,
                "sim_runs": run_count,
                "indicator_regime_stats": ind_stat_count,
            }

    # ─────────────────────────────────────────────────────────────────
    # COACH SESSION OPERATIONS (from 5-player model)
    # ─────────────────────────────────────────────────────────────────

    def record_coach_session(
        self,
        strategy_id: str,
        trading_date: str,
        regime: str,
        advice: str,
        weight_changes: int = 0,
        entry_change: float = 0,
        exit_change: float = 0,
        mistakes: List = None,
        weight_recs: List = None,
        opportunities: List = None,
        source: str = "rule_based",
        session_id: str = None,
    ) -> int:
        """Record a coach analysis + patch application."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO coach_sessions
                   (session_id, strategy_id, trading_date, regime, advice,
                    weight_changes, entry_change, exit_change,
                    mistakes_json, weight_recs_json, opportunities_json, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id or str(uuid.uuid4()),
                    strategy_id, trading_date, regime, advice,
                    weight_changes, entry_change, exit_change,
                    json.dumps(mistakes or []),
                    json.dumps(weight_recs or []),
                    json.dumps(opportunities or []),
                    source,
                ),
            )
            return cursor.lastrowid

    def get_recent_coach_sessions(
        self, strategy_id: str = None, limit: int = 10
    ) -> List[Dict]:
        """Get recent coach sessions."""
        with self.get_connection() as conn:
            if strategy_id:
                rows = conn.execute(
                    """SELECT * FROM coach_sessions
                       WHERE strategy_id = ?
                       ORDER BY id DESC LIMIT ?""",
                    (strategy_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM coach_sessions ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def get_coach_advice_effectiveness(self, strategy_id: str) -> List[Dict]:
        """Check if coach patches helped or hurt by comparing P&L before/after."""
        with self.get_connection() as conn:
            sessions = conn.execute(
                """SELECT id, trading_date, regime, advice, weight_changes
                   FROM coach_sessions
                   WHERE strategy_id = ?
                   ORDER BY id""",
                (strategy_id,),
            ).fetchall()

            results = []
            for sess in sessions:
                date = sess["trading_date"]
                before = conn.execute(
                    """SELECT COALESCE(SUM(pnl), 0) as pnl
                       FROM trades WHERE strategy_id = ? AND timestamp < ?""",
                    (strategy_id, date),
                ).fetchone()
                after = conn.execute(
                    """SELECT COALESCE(SUM(pnl), 0) as pnl
                       FROM trades WHERE strategy_id = ? AND timestamp >= ?""",
                    (strategy_id, date),
                ).fetchone()
                results.append({
                    "date": date,
                    "regime": sess["regime"],
                    "advice": (sess["advice"] or "")[:80],
                    "pnl_before": round(before["pnl"], 2) if before else 0,
                    "pnl_after": round(after["pnl"], 2) if after else 0,
                    "helped": (after["pnl"] if after else 0) > (before["pnl"] if before else 0),
                })
            return results

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY SNAPSHOT OPERATIONS (from 5-player model)
    # ─────────────────────────────────────────────────────────────────

    def record_strategy_snapshot(
        self,
        strategy_id: str,
        snapshot_type: str,
        label: str = "",
        weights: Dict = None,
        entry_threshold: float = 0,
        exit_threshold: float = 0,
        stop_loss_multiplier: float = 2.0,
        total_trades: int = 0,
        win_rate: float = 0,
        net_pnl: float = 0,
        sharpe: float = 0,
        sortino: float = 0,
        max_drawdown: float = 0,
        profit_factor: float = 0,
        regime_performance: Dict = None,
    ) -> int:
        """Record a strategy configuration snapshot."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO strategy_snapshots
                   (strategy_id, snapshot_type, label, weights_json,
                    entry_threshold, exit_threshold, stop_loss_multiplier,
                    total_trades, win_rate, net_pnl, sharpe, sortino,
                    max_drawdown, profit_factor, regime_performance_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    strategy_id, snapshot_type, label,
                    json.dumps(weights or {}),
                    entry_threshold, exit_threshold, stop_loss_multiplier,
                    total_trades, win_rate, net_pnl, sharpe, sortino,
                    max_drawdown, profit_factor,
                    json.dumps(regime_performance or {}),
                ),
            )
            return cursor.lastrowid

    def get_best_strategy_snapshot(self, strategy_id: str) -> Optional[Dict]:
        """Get the best-performing strategy snapshot."""
        with self.get_connection() as conn:
            row = conn.execute(
                """SELECT * FROM strategy_snapshots
                   WHERE strategy_id = ? AND snapshot_type = 'end'
                     AND total_trades > 0
                   ORDER BY sharpe DESC LIMIT 1""",
                (strategy_id,),
            ).fetchone()
            return dict(row) if row else None

    def get_strategy_snapshot_history(
        self, strategy_id: str, limit: int = 20
    ) -> List[Dict]:
        """Get strategy snapshot history for evolution tracking."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM strategy_snapshots
                   WHERE strategy_id = ?
                   ORDER BY id DESC LIMIT ?""",
                (strategy_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────
    # INDICATOR-REGIME STATS (from 5-player model)
    # ─────────────────────────────────────────────────────────────────

    def update_indicator_regime_stats(self, strategy_id: str = None):
        """Recalculate indicator-regime performance from trade data."""
        with self.get_connection() as conn:
            conn.execute("DELETE FROM indicator_regime_stats")

            where = "WHERE strategy_id = ?" if strategy_id else ""
            params = (strategy_id,) if strategy_id else ()
            rows = conn.execute(
                f"""SELECT pnl, market_regime, return_attribution
                    FROM trades {where}""",
                params,
            ).fetchall()

            stats = {}
            for row in rows:
                regime = row["market_regime"] or "unknown"
                pnl = row["pnl"] or 0
                try:
                    attribution = json.loads(row["return_attribution"] or "{}")
                except (json.JSONDecodeError, TypeError):
                    continue

                for ind_name, ind_val in attribution.items():
                    key = (ind_name, regime)
                    if key not in stats:
                        stats[key] = {"pnls": [], "wins": 0, "total": 0}
                    stats[key]["pnls"].append(pnl)
                    stats[key]["total"] += 1
                    if pnl > 0:
                        stats[key]["wins"] += 1

            now = datetime.now().isoformat()
            insert_rows = []
            for (ind, regime), s in stats.items():
                if s["total"] < 3:
                    continue
                avg_pnl = sum(s["pnls"]) / len(s["pnls"])
                win_rate = s["wins"] / s["total"] if s["total"] > 0 else 0
                contribution = avg_pnl / max(1, abs(avg_pnl)) if avg_pnl != 0 else 0
                insert_rows.append((
                    ind, regime, s["total"], round(avg_pnl, 2),
                    round(win_rate, 4), round(contribution, 4),
                    0, 0, 0, now,
                ))

            conn.executemany(
                """INSERT INTO indicator_regime_stats
                   (indicator, regime, total_trades, avg_pnl, win_rate,
                    contribution_score, predictive_accuracy, consistency_score,
                    information_ratio, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                insert_rows,
            )
            logger.info(f"Updated {len(insert_rows)} indicator-regime stats")

    def get_indicator_regime_stats(self, regime: str = None) -> List[Dict]:
        """Get indicator performance stats, optionally by regime."""
        with self.get_connection() as conn:
            if regime:
                rows = conn.execute(
                    """SELECT * FROM indicator_regime_stats
                       WHERE regime = ? ORDER BY avg_pnl DESC""",
                    (regime,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM indicator_regime_stats ORDER BY avg_pnl DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def get_top_indicators_for_regime(self, regime: str, limit: int = 10) -> List[Dict]:
        """Get the best-performing indicators for a given regime."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM indicator_regime_stats
                   WHERE regime = ? AND total_trades >= 5
                   ORDER BY contribution_score DESC LIMIT ?""",
                (regime, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────
    # SIMULATION RUN OPERATIONS (cross-run learning)
    # ─────────────────────────────────────────────────────────────────

    def start_run(
        self, run_number: int, session_id: str, days: int, symbols: int, config: Dict
    ) -> int:
        """Record the start of a new simulation/backtest run."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO sim_runs
                   (run_number, session_id, started_at, days, symbols, config_json)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (run_number, session_id, datetime.now().isoformat(),
                 days, symbols, json.dumps(config, default=str)),
            )
            return cursor.lastrowid

    def end_run(self, run_id: int, team_pnl: float, team_return: float, notes: str = ""):
        """Record the completion of a simulation run."""
        with self.get_connection() as conn:
            conn.execute(
                """UPDATE sim_runs
                   SET completed_at = ?, team_pnl = ?, team_return = ?, notes = ?
                   WHERE run_id = ?""",
                (datetime.now().isoformat(), team_pnl, team_return, notes, run_id),
            )

    def get_all_runs(self) -> List[Dict]:
        """Get all completed simulation runs."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM sim_runs
                   WHERE completed_at IS NOT NULL
                   ORDER BY run_id""",
            ).fetchall()
            return [dict(r) for r in rows]

    def get_run_count(self) -> int:
        """Get total number of completed runs."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM sim_runs WHERE completed_at IS NOT NULL"
            ).fetchone()
            return row["cnt"] if row else 0

    def get_cross_run_pnl_trend(self) -> List[Dict]:
        """Get P&L trend across all runs."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT run_id, run_number, team_pnl, team_return
                   FROM sim_runs WHERE completed_at IS NOT NULL
                   ORDER BY run_id""",
            ).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────
    # INDICATOR SCORE OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_indicator_scores(self, trade_id: str, scores: List[Dict]):
        """Store per-indicator scores for a trade."""
        with self.get_connection() as conn:
            for s in scores:
                conn.execute(
                    """INSERT INTO indicator_scores
                       (trade_id, indicator, weight_used, correlation_with_pnl,
                        predictive_accuracy, signal_quality_score, verdict)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        trade_id,
                        s.get("indicator", s.get("name", "")),
                        s.get("weight_used", 0),
                        s.get("correlation_with_pnl", 0),
                        s.get("predictive_accuracy", 0),
                        s.get("signal_quality_score", 0),
                        s.get("verdict", "neutral"),
                    ),
                )

    def get_indicator_scores_for_trade(self, trade_id: str) -> List[Dict]:
        """Get indicator scores for a specific trade."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM indicator_scores WHERE trade_id = ? ORDER BY signal_quality_score DESC",
                (trade_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────
    # ADVANCED QUERIES (cross-system analytics)
    # ─────────────────────────────────────────────────────────────────

    def get_strategy_regime_performance(self, strategy_id: str) -> List[Dict]:
        """Get per-regime performance for a strategy across all trades."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT market_regime,
                          COUNT(*) as trades,
                          SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                          ROUND(AVG(pnl), 2) as avg_pnl,
                          ROUND(SUM(pnl), 2) as total_pnl
                   FROM trades
                   WHERE strategy_id = ?
                   GROUP BY market_regime
                   ORDER BY total_pnl DESC""",
                (strategy_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_strategy_asset_performance(self, strategy_id: str) -> List[Dict]:
        """Get per-asset performance for a strategy."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT asset,
                          COUNT(*) as trades,
                          SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                          ROUND(AVG(pnl), 2) as avg_pnl,
                          ROUND(SUM(pnl), 2) as total_pnl
                   FROM trades
                   WHERE strategy_id = ?
                   GROUP BY asset
                   ORDER BY total_pnl DESC""",
                (strategy_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ─────────────────────────────────────────────────────────────────
    # LIVE PAPER TRADING (5-Player Coach)
    # ─────────────────────────────────────────────────────────────────

    def store_live_player_state(self, player_id: str, state: Dict) -> str:
        """Store or update live player state."""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO live_player_states (
                    player_id, label, weights_json, entry_threshold, exit_threshold,
                    min_hold_bars, equity, day_pnl, strategy_version,
                    positions_json, bars_held_json, prev_si_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                player_id,
                state.get("label", ""),
                json.dumps(state.get("weights", {})),
                state.get("entry_threshold", 0.30),
                state.get("exit_threshold", -0.10),
                state.get("min_hold_bars", 4),
                state.get("equity", 100000.0),
                state.get("day_pnl", 0.0),
                state.get("strategy_version", "v1.0"),
                json.dumps(state.get("positions", {})),
                json.dumps(state.get("bars_held", {})),
                json.dumps(state.get("prev_si", {})),
            ))
        return player_id

    def get_live_player_state(self, player_id: str) -> Optional[Dict]:
        """Get live player state."""
        with self.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM live_player_states WHERE player_id = ?",
                (player_id,),
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            for field in ("weights_json", "positions_json", "bars_held_json", "prev_si_json"):
                if result.get(field):
                    result[field.replace("_json", "")] = json.loads(result[field])
            return result

    def get_all_live_player_states(self) -> Dict[str, Dict]:
        """Get all live player states."""
        with self.get_connection() as conn:
            rows = conn.execute("SELECT * FROM live_player_states").fetchall()
            result = {}
            for row in rows:
                data = dict(row)
                for field in ("weights_json", "positions_json", "bars_held_json", "prev_si_json"):
                    if data.get(field):
                        data[field.replace("_json", "")] = json.loads(data[field])
                result[data["player_id"]] = data
            return result

    def get_live_positions(self) -> List[Dict]:
        """Get all open positions from all players."""
        all_states = self.get_all_live_player_states()
        positions = []
        for player_id, state in all_states.items():
            for pos in state.get("positions", {}).values():
                pos["player_id"] = player_id
                positions.append(pos)
        return positions

    def store_live_trade(self, trade: Dict) -> str:
        """Store a live trade."""
        trade_id = trade.get("trade_id", str(uuid.uuid4()))
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO live_trades (
                    trade_id, player_id, symbol, direction,
                    entry_price, exit_price, quantity, pnl,
                    si_value, exit_si, exit_reason,
                    entry_time, exit_time, atr
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_id,
                trade.get("player_id", ""),
                trade.get("symbol", ""),
                trade.get("direction", trade.get("side", "")),
                trade.get("entry_price"),
                trade.get("exit_price"),
                trade.get("quantity"),
                trade.get("pnl"),
                trade.get("si_value"),
                trade.get("exit_si"),
                trade.get("exit_reason"),
                trade.get("entry_time", trade.get("timestamp")),
                trade.get("exit_time"),
                trade.get("atr"),
            ))
        return trade_id

    def get_live_trades(
        self,
        player_id: str = None,
        start_date: str = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get live trades with optional filters."""
        query = "SELECT * FROM live_trades WHERE 1=1"
        params = []

        if player_id:
            query += " AND player_id = ?"
            params.append(player_id)

        if start_date:
            query += " AND entry_time >= ?"
            params.append(start_date)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_player_equity_history(self, player_id: str, limit: int = 60) -> List[Dict]:
        """Get equity history for a player."""
        with self.get_connection() as conn:
            rows = conn.execute(
                """SELECT * FROM live_equity_history
                   WHERE player_id = ?
                   ORDER BY date DESC
                   LIMIT ?""",
                (player_id, limit),
            ).fetchall()
            return [dict(r) for r in reversed(rows)]

    def store_player_equity_snapshot(self, player_id: str, snapshot: Dict) -> int:
        """Store daily equity snapshot for a player."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO live_equity_history (
                    player_id, date, equity, day_pnl, trades, version
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                player_id,
                snapshot.get("date"),
                snapshot.get("equity"),
                snapshot.get("day_pnl"),
                snapshot.get("trades"),
                snapshot.get("version"),
            ))
            return cursor.lastrowid

    def store_live_session(self, session: Dict) -> int:
        """Store a live trading session."""
        with self.get_connection() as conn:
            cursor = conn.execute("""
                INSERT INTO live_sessions (
                    started_at, ended_at, initial_equity, final_equity,
                    total_trades, market_regime, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session.get("started_at"),
                session.get("ended_at"),
                session.get("initial_equity"),
                session.get("final_equity"),
                session.get("total_trades"),
                session.get("market_regime"),
                session.get("notes"),
            ))
            return cursor.lastrowid

    def get_live_sessions(self, limit: int = 30) -> List[Dict]:
        """Get recent live trading sessions."""
        with self.get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM live_sessions ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
