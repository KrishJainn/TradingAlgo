"""
SQLite database module for trade journaling and evolution tracking.

Stores all trades, generations, DNA configurations, and performance metrics.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager


class Database:
    """
    SQLite database manager for the trading evolution system.

    Handles all database operations including:
    - Trade logging
    - Generation tracking
    - DNA configuration storage
    - Indicator performance metrics
    - Hall of Fame management
    """

    def __init__(self, db_path: Path, wal_mode: bool = True):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
            wal_mode: Enable Write-Ahead Logging for better concurrency
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.wal_mode = wal_mode
        self._initialize_schema()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        if self.wal_mode:
            conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialize_schema(self):
        """Create all database tables."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Evolution runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS evolution_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    config_json TEXT NOT NULL,
                    training_start_date TEXT,
                    training_end_date TEXT,
                    validation_start_date TEXT,
                    validation_end_date TEXT,
                    holdout_start_date TEXT,
                    holdout_end_date TEXT,
                    symbols_json TEXT NOT NULL,
                    total_generations INTEGER DEFAULT 0,
                    final_best_dna_id TEXT,
                    final_fitness REAL,
                    status TEXT DEFAULT 'running'
                )
            """)

            # Generations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    generation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    generation_num INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    population_size INTEGER NOT NULL,
                    best_fitness REAL,
                    avg_fitness REAL,
                    std_fitness REAL,
                    best_dna_id TEXT,
                    total_trades INTEGER DEFAULT 0,
                    total_net_pnl REAL DEFAULT 0,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    long_win_rate REAL,
                    short_win_rate REAL,
                    validation_fitness REAL,
                    holdout_fitness REAL,
                    coach_confidence REAL,
                    coach_insights TEXT,
                    status TEXT DEFAULT 'running',
                    FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id)
                )
            """)

            # DNA configurations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dna_configs (
                    dna_id TEXT PRIMARY KEY,
                    run_id INTEGER NOT NULL,
                    generation_num INTEGER NOT NULL,
                    weights_json TEXT NOT NULL,
                    active_indicators_json TEXT NOT NULL,
                    fitness_score REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    net_profit REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    long_trades INTEGER,
                    short_trades INTEGER,
                    parent_1_id TEXT,
                    parent_2_id TEXT,
                    mutation_applied TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id)
                )
            """)

            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    run_id INTEGER NOT NULL,
                    generation_num INTEGER NOT NULL,
                    dna_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL CHECK(direction IN ('LONG', 'SHORT')),
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT NOT NULL,
                    gross_pnl REAL NOT NULL,
                    commission REAL NOT NULL,
                    slippage REAL NOT NULL,
                    net_pnl REAL NOT NULL,
                    net_pnl_pct REAL NOT NULL,
                    holding_period_bars INTEGER,
                    exit_reason TEXT CHECK(exit_reason IN ('signal', 'stop_loss', 'take_profit', 'trailing_stop', 'end_of_data')),
                    super_indicator_entry REAL,
                    super_indicator_exit REAL,
                    stop_loss_price REAL,
                    take_profit_price REAL,
                    atr_at_entry REAL,
                    market_regime TEXT,
                    indicator_snapshot_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id),
                    FOREIGN KEY (dna_id) REFERENCES dna_configs(dna_id)
                )
            """)

            # Indicator performance table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS indicator_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    generation_num INTEGER NOT NULL,
                    indicator_name TEXT NOT NULL,
                    weight_used REAL NOT NULL,
                    correlation_with_returns REAL,
                    predictive_accuracy REAL,
                    signal_quality_score REAL,
                    avg_return_bullish REAL,
                    avg_return_bearish REAL,
                    consistency_score REAL,
                    information_ratio REAL,
                    trades_contributed INTEGER,
                    rank_in_generation INTEGER,
                    long_accuracy REAL,
                    short_accuracy REAL,
                    regime_trending_up REAL,
                    regime_trending_down REAL,
                    regime_ranging REAL,
                    regime_volatile REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id)
                )
            """)

            # Hall of Fame table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hall_of_fame (
                    rank INTEGER PRIMARY KEY,
                    dna_id TEXT NOT NULL UNIQUE,
                    run_id INTEGER NOT NULL,
                    generation_num INTEGER NOT NULL,
                    fitness_score REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    net_profit REAL NOT NULL,
                    win_rate REAL,
                    total_trades INTEGER,
                    validation_fitness REAL,
                    holdout_fitness REAL,
                    weights_json TEXT NOT NULL,
                    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (dna_id) REFERENCES dna_configs(dna_id)
                )
            """)

            # Coach recommendations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS coach_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    generation_num INTEGER NOT NULL,
                    promoted_indicators TEXT,
                    demoted_indicators TEXT,
                    removed_indicators TEXT,
                    weight_adjustments TEXT,
                    regime_insights TEXT,
                    pattern_insights TEXT,
                    confidence_level REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES evolution_runs(run_id)
                )
            """)

            # Market regimes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    regime TEXT NOT NULL CHECK(regime IN ('trending_up', 'trending_down', 'ranging', 'volatile', 'unknown')),
                    atr_percentile REAL,
                    trend_strength REAL,
                    volatility_rank REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_run_gen ON trades(run_id, generation_num)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_dna ON trades(dna_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dna_run_gen ON dna_configs(run_id, generation_num)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_dna_fitness ON dna_configs(fitness_score DESC)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ind_perf_name ON indicator_performance(indicator_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ind_perf_run_gen ON indicator_performance(run_id, generation_num)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_regime_symbol_time ON market_regimes(symbol, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_generations_run ON generations(run_id)")

    # Evolution run operations
    def create_evolution_run(self, config_json: str, symbols: List[str],
                             data_splits: Dict[str, tuple]) -> int:
        """Create a new evolution run record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO evolution_runs (
                    started_at, config_json, symbols_json,
                    training_start_date, training_end_date,
                    validation_start_date, validation_end_date,
                    holdout_start_date, holdout_end_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                config_json,
                json.dumps(symbols),
                data_splits.get('training', (None, None))[0],
                data_splits.get('training', (None, None))[1],
                data_splits.get('validation', (None, None))[0],
                data_splits.get('validation', (None, None))[1],
                data_splits.get('holdout', (None, None))[0],
                data_splits.get('holdout', (None, None))[1],
            ))
            return cursor.lastrowid

    def complete_evolution_run(self, run_id: int, best_dna_id: str,
                               final_fitness: float, total_generations: int):
        """Mark evolution run as complete."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE evolution_runs SET
                    completed_at = ?,
                    final_best_dna_id = ?,
                    final_fitness = ?,
                    total_generations = ?,
                    status = 'completed'
                WHERE run_id = ?
            """, (datetime.now().isoformat(), best_dna_id, final_fitness,
                  total_generations, run_id))

    # Generation operations
    def create_generation(self, run_id: int, generation_num: int,
                          population_size: int) -> int:
        """Create a new generation record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO generations (
                    run_id, generation_num, started_at, population_size
                ) VALUES (?, ?, ?, ?)
            """, (run_id, generation_num, datetime.now().isoformat(), population_size))
            return cursor.lastrowid

    def update_generation(self, generation_id: int, **kwargs):
        """Update generation with computed metrics."""
        valid_fields = {
            'completed_at', 'best_fitness', 'avg_fitness', 'std_fitness',
            'best_dna_id', 'total_trades', 'total_net_pnl', 'sharpe_ratio',
            'max_drawdown', 'win_rate', 'long_win_rate', 'short_win_rate',
            'validation_fitness', 'holdout_fitness', 'coach_confidence',
            'coach_insights', 'status'
        }
        updates = {k: v for k, v in kwargs.items() if k in valid_fields}
        if not updates:
            return

        set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"""
                UPDATE generations SET {set_clause} WHERE generation_id = ?
            """, list(updates.values()) + [generation_id])

    # DNA operations
    def save_dna(self, dna_id: str, run_id: int, generation_num: int,
                 weights: Dict[str, float], active_indicators: List[str],
                 fitness_score: float = None, metrics: Dict = None,
                 parent_ids: tuple = None, mutation_applied: str = None):
        """Save DNA configuration."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO dna_configs (
                    dna_id, run_id, generation_num, weights_json,
                    active_indicators_json, fitness_score, sharpe_ratio,
                    max_drawdown, net_profit, win_rate, total_trades,
                    long_trades, short_trades, parent_1_id, parent_2_id,
                    mutation_applied, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dna_id, run_id, generation_num,
                json.dumps(weights),
                json.dumps(active_indicators),
                fitness_score,
                metrics.get('sharpe_ratio') if metrics else None,
                metrics.get('max_drawdown') if metrics else None,
                metrics.get('net_profit') if metrics else None,
                metrics.get('win_rate') if metrics else None,
                metrics.get('total_trades') if metrics else None,
                metrics.get('long_trades') if metrics else None,
                metrics.get('short_trades') if metrics else None,
                parent_ids[0] if parent_ids else None,
                parent_ids[1] if parent_ids and len(parent_ids) > 1 else None,
                mutation_applied,
                datetime.now().isoformat()
            ))

    def save_dna_config(self, run_id: int, generation_num: int, dna: Any):
        """Save DNA configuration helper."""
        self.save_dna(
            dna_id=dna.dna_id,
            run_id=run_id,
            generation_num=generation_num,
            weights=dna.get_weights(),
            active_indicators=dna.get_active_indicators(),
            fitness_score=dna.fitness_score,
            metrics={
                'sharpe_ratio': dna.sharpe_ratio,
                'max_drawdown': dna.max_drawdown,
                'net_profit': dna.net_profit,
                'win_rate': dna.win_rate,
                'total_trades': dna.total_trades,
                'long_trades': dna.long_trades,
                'short_trades': dna.short_trades
            },
            parent_ids=tuple(dna.parents) if hasattr(dna, 'parents') and dna.parents else None,
            mutation_applied=dna.mutation_history[-1] if hasattr(dna, 'mutation_history') and dna.mutation_history else None
        )

    def get_dna(self, dna_id: str) -> Optional[Dict]:
        """Retrieve DNA configuration."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM dna_configs WHERE dna_id = ?", (dna_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['weights'] = json.loads(result['weights_json'])
                result['active_indicators'] = json.loads(result['active_indicators_json'])
                return result
        return None

    # Trade operations
    def save_trade(self, trade: Dict):
        """Save a trade record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO trades (
                    trade_id, run_id, generation_num, dna_id, symbol, direction,
                    entry_price, exit_price, quantity, entry_time, exit_time,
                    gross_pnl, commission, slippage, net_pnl, net_pnl_pct,
                    holding_period_bars, exit_reason, super_indicator_entry,
                    super_indicator_exit, stop_loss_price, take_profit_price,
                    atr_at_entry, market_regime, indicator_snapshot_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['trade_id'], trade['run_id'], trade['generation_num'],
                trade['dna_id'], trade['symbol'], trade['direction'],
                trade['entry_price'], trade['exit_price'], trade['quantity'],
                trade['entry_time'], trade['exit_time'],
                trade['gross_pnl'], trade['commission'], trade['slippage'],
                trade['net_pnl'], trade['net_pnl_pct'],
                trade.get('holding_period_bars'),
                trade.get('exit_reason'),
                trade.get('super_indicator_entry'),
                trade.get('super_indicator_exit'),
                trade.get('stop_loss_price'),
                trade.get('take_profit_price'),
                trade.get('atr_at_entry'),
                trade.get('market_regime'),
                json.dumps(trade.get('indicator_snapshot', {}))
            ))

    def get_trades_for_generation(self, run_id: int, generation_num: int) -> List[Dict]:
        """Get all trades for a generation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trades
                WHERE run_id = ? AND generation_num = ?
                ORDER BY entry_time
            """, (run_id, generation_num))
            return [dict(row) for row in cursor.fetchall()]

    def get_trades_for_dna(self, dna_id: str) -> List[Dict]:
        """Get all trades for a specific DNA."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM trades WHERE dna_id = ? ORDER BY entry_time
            """, (dna_id,))
            return [dict(row) for row in cursor.fetchall()]

    # Indicator performance operations
    def save_indicator_performance(self, run_id: int, generation_num: int,
                                   performances: List[Dict]):
        """Save indicator performance metrics for a generation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for perf in performances:
                cursor.execute("""
                    INSERT INTO indicator_performance (
                        run_id, generation_num, indicator_name, weight_used,
                        correlation_with_returns, predictive_accuracy,
                        signal_quality_score, avg_return_bullish, avg_return_bearish,
                        consistency_score, information_ratio, trades_contributed,
                        rank_in_generation, long_accuracy, short_accuracy,
                        regime_trending_up, regime_trending_down,
                        regime_ranging, regime_volatile
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    run_id, generation_num,
                    perf['indicator_name'], perf['weight_used'],
                    perf.get('correlation_with_returns'),
                    perf.get('predictive_accuracy'),
                    perf.get('signal_quality_score'),
                    perf.get('avg_return_bullish'),
                    perf.get('avg_return_bearish'),
                    perf.get('consistency_score'),
                    perf.get('information_ratio'),
                    perf.get('trades_contributed'),
                    perf.get('rank_in_generation'),
                    perf.get('long_accuracy'),
                    perf.get('short_accuracy'),
                    perf.get('regime_trending_up'),
                    perf.get('regime_trending_down'),
                    perf.get('regime_ranging'),
                    perf.get('regime_volatile')
                ))

    def get_indicator_history(self, indicator_name: str, run_id: int) -> List[Dict]:
        """Get performance history for an indicator across generations."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM indicator_performance
                WHERE indicator_name = ? AND run_id = ?
                ORDER BY generation_num
            """, (indicator_name, run_id))
            return [dict(row) for row in cursor.fetchall()]

    # Hall of Fame operations
    def update_hall_of_fame(self, dna_id: str, run_id: int, generation_num: int,
                            fitness_score: float, metrics: Dict,
                            weights_json: str, max_size: int = 10):
        """Update Hall of Fame with a new DNA if it qualifies."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Check if this DNA is already in hall of fame
            cursor.execute("SELECT rank FROM hall_of_fame WHERE dna_id = ?", (dna_id,))
            existing = cursor.fetchone()
            if existing:
                return  # Already in hall of fame

            # Get current hall of fame
            cursor.execute("SELECT * FROM hall_of_fame ORDER BY fitness_score DESC")
            hall = cursor.fetchall()

            # Check if this DNA qualifies
            if len(hall) < max_size or fitness_score > hall[-1]['fitness_score']:
                # Find an available temporary rank (use negative to avoid conflicts)
                temp_rank = -1 * (len(hall) + 1)

                # Add to hall of fame with temporary negative rank
                cursor.execute("""
                    INSERT INTO hall_of_fame (
                        rank, dna_id, run_id, generation_num, fitness_score,
                        sharpe_ratio, max_drawdown, net_profit, win_rate,
                        total_trades, validation_fitness, holdout_fitness,
                        weights_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    temp_rank,
                    dna_id, run_id, generation_num, fitness_score,
                    metrics.get('sharpe_ratio', 0),
                    metrics.get('max_drawdown', 0),
                    metrics.get('net_profit', 0),
                    metrics.get('win_rate', 0),
                    metrics.get('total_trades', 0),
                    metrics.get('validation_fitness'),
                    metrics.get('holdout_fitness'),
                    weights_json
                ))

                # Re-rank and trim to max_size
                cursor.execute("""
                    DELETE FROM hall_of_fame
                    WHERE dna_id NOT IN (
                        SELECT dna_id FROM hall_of_fame
                        ORDER BY fitness_score DESC
                        LIMIT ?
                    )
                """, (max_size,))

                # Update ranks - first set all to negative to avoid conflicts
                cursor.execute("SELECT dna_id FROM hall_of_fame ORDER BY fitness_score DESC")
                dna_ids = [row['dna_id'] for row in cursor.fetchall()]

                for i, did in enumerate(dna_ids):
                    cursor.execute(
                        "UPDATE hall_of_fame SET rank = ? WHERE dna_id = ?",
                        (-(i + 100), did)  # Use large negative to avoid conflicts
                    )

                # Now set to actual positive ranks
                for i, did in enumerate(dna_ids, 1):
                    cursor.execute(
                        "UPDATE hall_of_fame SET rank = ? WHERE dna_id = ?",
                        (i, did)
                    )

    def get_hall_of_fame(self) -> List[Dict]:
        """Get all Hall of Fame entries."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM hall_of_fame ORDER BY rank
            """)
            results = []
            for row in cursor.fetchall():
                entry = dict(row)
                entry['weights'] = json.loads(entry['weights_json'])
                results.append(entry)
            return results

    # Coach recommendations
    def save_coach_recommendation(self, run_id: int, generation_num: int,
                                  recommendation: Dict):
        """Save coach recommendation."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO coach_recommendations (
                    run_id, generation_num, promoted_indicators,
                    demoted_indicators, removed_indicators, weight_adjustments,
                    regime_insights, pattern_insights, confidence_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id, generation_num,
                json.dumps(recommendation.get('promoted_indicators', [])),
                json.dumps(recommendation.get('demoted_indicators', [])),
                json.dumps(recommendation.get('removed_indicators', [])),
                json.dumps(recommendation.get('weight_adjustments', {})),
                recommendation.get('regime_insights'),
                recommendation.get('pattern_insights'),
                recommendation.get('confidence_level')
            ))

    # Market regime operations
    def save_market_regime(self, symbol: str, timestamp: str, regime: str,
                           metrics: Dict = None):
        """Save market regime detection."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO market_regimes (
                    symbol, timestamp, regime, atr_percentile,
                    trend_strength, volatility_rank
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                symbol, timestamp, regime,
                metrics.get('atr_percentile') if metrics else None,
                metrics.get('trend_strength') if metrics else None,
                metrics.get('volatility_rank') if metrics else None
            ))

    def get_regime_at(self, symbol: str, timestamp: str) -> Optional[str]:
        """Get market regime at a specific timestamp."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT regime FROM market_regimes
                WHERE symbol = ? AND timestamp <= ?
                ORDER BY timestamp DESC LIMIT 1
            """, (symbol, timestamp))
            row = cursor.fetchone()
            return row['regime'] if row else None

    # Statistics and summaries
    def get_generation_summary(self, run_id: int) -> List[Dict]:
        """Get summary statistics for all generations in a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    g.*,
                    COUNT(t.trade_id) as actual_trades,
                    SUM(CASE WHEN t.net_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN t.direction = 'LONG' THEN 1 ELSE 0 END) as long_trades,
                    SUM(CASE WHEN t.direction = 'SHORT' THEN 1 ELSE 0 END) as short_trades
                FROM generations g
                LEFT JOIN trades t ON g.run_id = t.run_id AND g.generation_num = t.generation_num
                WHERE g.run_id = ?
                GROUP BY g.generation_id
                ORDER BY g.generation_num
            """, (run_id,))
            return [dict(row) for row in cursor.fetchall()]

    def get_best_dna_for_run(self, run_id: int) -> Optional[Dict]:
        """Get the best performing DNA for a run."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM dna_configs
                WHERE run_id = ?
                ORDER BY fitness_score DESC
                LIMIT 1
            """, (run_id,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['weights'] = json.loads(result['weights_json'])
                result['active_indicators'] = json.loads(result['active_indicators_json'])
                return result
        return None
