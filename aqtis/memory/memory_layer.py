"""
AQTIS Memory Layer - Unified interface for all memory operations.

Facade combining StructuredDB (SQLite) and VectorStore (ChromaDB)
to provide a single interface for storing and retrieving trading knowledge.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .database import StructuredDB
from .vector_store import VectorStore
from .context_builder import ContextBuilder

logger = logging.getLogger(__name__)


class MemoryLayer:
    """
    Central interface for all memory operations.

    Combines structured storage (trades, predictions, strategies)
    with semantic search (research papers, trade patterns)
    and LLM context generation (from 5-player coach model).
    """

    def __init__(self, db_path: str = "aqtis.db", vector_path: str = "aqtis_vectors"):
        self.db = StructuredDB(db_path)
        self.vectors = VectorStore(vector_path)
        self.context = ContextBuilder(self.db)

    # ─────────────────────────────────────────────────────────────────
    # TRADE OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_trade(self, trade: Dict, prediction: Dict = None) -> str:
        """Store a new trade and optionally its prediction."""
        trade_id = self.db.store_trade(trade)

        if prediction:
            prediction["trade_id"] = trade_id
            self.db.store_prediction(prediction)

        # Also store as trade pattern for semantic search
        try:
            description = self._trade_to_description(trade)
            self.vectors.add_trade_pattern({
                "trade_id": trade_id,
                "text": description,
                "metadata": {
                    "strategy_id": trade.get("strategy_id", ""),
                    "asset": trade.get("asset", ""),
                    "action": trade.get("action", ""),
                    "market_regime": trade.get("market_regime", ""),
                    "pnl": trade.get("pnl", 0),
                    "pnl_percent": trade.get("pnl_percent", 0),
                    "outcome": "win" if (trade.get("pnl") or 0) > 0 else "loss",
                },
            })
        except Exception:
            pass  # Vector store is optional, don't fail trade storage

        return trade_id

    def update_trade_outcome(self, trade_id: str, outcome: Dict):
        """Update trade with actual outcome and calculate prediction errors."""
        self.db.update_trade(trade_id, outcome)

        trade = self.db.get_trade(trade_id)
        if not trade or not trade.get("prediction_id"):
            return

        prediction = self.db.get_prediction(trade["prediction_id"])
        if not prediction:
            return

        # Calculate prediction errors
        actual_return = outcome.get("pnl_percent", 0)
        predicted_return = prediction.get("predicted_return", 0)

        errors = {
            "actual_return": actual_return,
            "was_profitable": 1 if actual_return > 0 else 0,
            "return_prediction_error": abs(predicted_return - actual_return) if predicted_return else None,
            "direction_correct": 1 if (predicted_return > 0) == (actual_return > 0) else 0,
        }

        if prediction.get("predicted_confidence"):
            errors["confidence_calibration_error"] = abs(
                prediction["predicted_confidence"] - (1.0 if actual_return > 0 else 0.0)
            )

        if outcome.get("hold_duration_seconds"):
            errors["actual_hold_seconds"] = outcome["hold_duration_seconds"]
        if outcome.get("max_drawdown"):
            errors["actual_max_drawdown"] = outcome["max_drawdown"]

        self.db.update_prediction(trade["prediction_id"], errors)

    def get_trade(self, trade_id: str) -> Optional[Dict]:
        """Get a single trade by ID."""
        return self.db.get_trade(trade_id)

    def get_trades(self, **kwargs) -> List[Dict]:
        """Query trades with filters."""
        return self.db.get_trades(**kwargs)

    # ─────────────────────────────────────────────────────────────────
    # PREDICTION OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_prediction(self, prediction: Dict) -> str:
        """Store a new prediction."""
        return self.db.store_prediction(prediction)

    def get_prediction(self, prediction_id: str) -> Optional[Dict]:
        """Get a prediction by ID."""
        return self.db.get_prediction(prediction_id)

    def get_predictions(self, **kwargs) -> List[Dict]:
        """Query predictions with filters."""
        return self.db.get_predictions(**kwargs)

    def get_prediction_accuracy_history(self, lookback_days: int = 30) -> Dict:
        """Calculate prediction accuracy over time."""
        return self.db.get_prediction_accuracy(lookback_days)

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_strategy(self, strategy: Dict) -> str:
        """Store or update a strategy."""
        return self.db.store_strategy(strategy)

    def get_strategy(self, strategy_id: str) -> Optional[Dict]:
        """Get a strategy by ID."""
        return self.db.get_strategy(strategy_id)

    def get_active_strategies(self) -> List[Dict]:
        """Get all active strategies."""
        return self.db.get_active_strategies()

    def get_strategy_performance(self, strategy_id: str, regime: str = None) -> Dict:
        """Get aggregate performance for a strategy."""
        return self.db.get_strategy_performance(strategy_id, regime)

    def get_strategy_asset_performance(self, strategy_id: str) -> List[Dict]:
        """Get per-asset performance breakdown for a strategy."""
        return self.db.get_strategy_asset_performance(strategy_id)

    def get_strategy_regime_performance(self, strategy_id: str) -> List[Dict]:
        """Get per-regime performance breakdown for a strategy."""
        return self.db.get_strategy_regime_performance(strategy_id)

    # ─────────────────────────────────────────────────────────────────
    # SIMILAR TRADES (SEMANTIC SEARCH)
    # ─────────────────────────────────────────────────────────────────

    def get_similar_trades(self, current_setup: Dict, top_k: int = 10) -> List[Dict]:
        """
        Find historically similar trade setups.

        Uses semantic search on trade pattern descriptions.
        """
        description = self._trade_to_description(current_setup)
        similar = self.vectors.find_similar_trades(description, top_k=top_k)

        # Enrich with full trade data from structured DB
        enriched = []
        for match in similar:
            trade_id = match.get("trade_id")
            full_trade = self.db.get_trade(trade_id)
            if full_trade:
                full_trade["similarity_score"] = 1.0 - (match.get("distance", 1.0))
                enriched.append(full_trade)
            else:
                enriched.append(match)

        return enriched

    # ─────────────────────────────────────────────────────────────────
    # RESEARCH OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_research(self, document: Dict) -> str:
        """Store a research paper/insight."""
        return self.vectors.add_research(document)

    def search_research(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search research database."""
        return self.vectors.search_research(query, top_k)

    # ─────────────────────────────────────────────────────────────────
    # KNOWLEDGE BASE OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def search_knowledge(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search the curated knowledge base."""
        return self.vectors.search_knowledge(query, top_k)

    def store_knowledge(self, document: Dict) -> str:
        """Store a document in the knowledge base."""
        return self.vectors.add_knowledge(document)

    # ─────────────────────────────────────────────────────────────────
    # MARKET STATE OPERATIONS
    # ─────────────────────────────────────────────────────────────────

    def store_market_state(self, state: Dict) -> str:
        """Store a market state snapshot."""
        return self.db.store_market_state(state)

    def get_market_regime(self, timestamp: datetime = None) -> Optional[Dict]:
        """Get current or historical market regime."""
        if timestamp is None:
            return self.db.get_latest_market_state()
        # For historical, get closest state
        states = self.db.get_market_state_history(days=365)
        target = timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp)
        for state in states:
            if state["timestamp"] <= target:
                return state
        return None

    # ─────────────────────────────────────────────────────────────────
    # RISK EVENTS
    # ─────────────────────────────────────────────────────────────────

    def store_risk_event(self, event: Dict) -> str:
        """Store a risk event."""
        return self.db.store_risk_event(event)

    def get_risk_events(self, days: int = 7) -> List[Dict]:
        """Get recent risk events."""
        return self.db.get_risk_events(days)

    # ─────────────────────────────────────────────────────────────────
    # STATS
    # ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Get combined statistics from all storage backends."""
        db_stats = self.db.get_stats()
        vector_stats = self.vectors.get_stats()
        return {**db_stats, "vector_collections": vector_stats}

    # ─────────────────────────────────────────────────────────────────
    # COACH OPERATIONS (from 5-player model)
    # ─────────────────────────────────────────────────────────────────

    def record_coach_session(self, **kwargs) -> int:
        """Record a coach analysis + patch application."""
        return self.db.record_coach_session(**kwargs)

    def get_recent_coach_sessions(self, strategy_id: str = None, limit: int = 10) -> List[Dict]:
        """Get recent coach sessions."""
        return self.db.get_recent_coach_sessions(strategy_id, limit)

    def get_coach_advice_effectiveness(self, strategy_id: str) -> List[Dict]:
        """Check if coach patches helped or hurt."""
        return self.db.get_coach_advice_effectiveness(strategy_id)

    # ─────────────────────────────────────────────────────────────────
    # STRATEGY SNAPSHOT OPERATIONS (from 5-player model)
    # ─────────────────────────────────────────────────────────────────

    def record_strategy_snapshot(self, **kwargs) -> int:
        """Record a strategy configuration snapshot."""
        return self.db.record_strategy_snapshot(**kwargs)

    def get_best_strategy_snapshot(self, strategy_id: str) -> Optional[Dict]:
        """Get the best-performing strategy snapshot."""
        return self.db.get_best_strategy_snapshot(strategy_id)

    # ─────────────────────────────────────────────────────────────────
    # INDICATOR-REGIME INTELLIGENCE (from 5-player model)
    # ─────────────────────────────────────────────────────────────────

    def update_indicator_regime_stats(self, strategy_id: str = None):
        """Recalculate indicator-regime performance stats."""
        self.db.update_indicator_regime_stats(strategy_id)

    def get_indicator_regime_stats(self, regime: str = None) -> List[Dict]:
        """Get indicator performance stats by regime."""
        return self.db.get_indicator_regime_stats(regime)

    def get_top_indicators_for_regime(self, regime: str, limit: int = 10) -> List[Dict]:
        """Get the best-performing indicators for a regime."""
        return self.db.get_top_indicators_for_regime(regime, limit)

    # ─────────────────────────────────────────────────────────────────
    # SIMULATION RUN OPERATIONS (cross-run learning)
    # ─────────────────────────────────────────────────────────────────

    def start_run(self, run_number: int, session_id: str, days: int, symbols: int, config: Dict) -> int:
        """Record the start of a new simulation run."""
        return self.db.start_run(run_number, session_id, days, symbols, config)

    def end_run(self, run_id: int, team_pnl: float, team_return: float, notes: str = ""):
        """Record the completion of a simulation run."""
        self.db.end_run(run_id, team_pnl, team_return, notes)

    def get_cross_run_pnl_trend(self) -> List[Dict]:
        """Get P&L trend across all runs."""
        return self.db.get_cross_run_pnl_trend()

    # ─────────────────────────────────────────────────────────────────
    # CONTEXT BUILDING (for LLM prompts)
    # ─────────────────────────────────────────────────────────────────

    def build_strategy_context(self, strategy_id: str) -> str:
        """Build LLM context for strategy optimization."""
        return self.context.build_strategy_context(strategy_id)

    def build_coach_context(self, strategy_id: str, trading_date: str = "") -> str:
        """Build LLM context for daily coach analysis."""
        return self.context.build_coach_context(strategy_id, trading_date)

    def build_pre_trade_context(self, signal: Dict) -> str:
        """Build LLM context for pre-trade decision making."""
        return self.context.build_pre_trade_context(signal)

    def build_full_optimization_context(self, strategy_id: str) -> str:
        """Build comprehensive context for strategy redesign."""
        return self.context.build_full_optimization_context(strategy_id)

    # ─────────────────────────────────────────────────────────────────
    # INDICATOR SCORE STORAGE
    # ─────────────────────────────────────────────────────────────────

    def store_indicator_scores(self, trade_id: str, scores: List[Dict]):
        """Store per-indicator scores for a trade."""
        self.db.store_indicator_scores(trade_id, scores)

    # ─────────────────────────────────────────────────────────────────
    # LIVE PAPER TRADING (5-Player Coach Integration)
    # ─────────────────────────────────────────────────────────────────

    def store_live_player_state(self, player_id: str, state: Dict) -> str:
        """Store or update live player state."""
        return self.db.store_live_player_state(player_id, state)

    def get_live_player_state(self, player_id: str) -> Optional[Dict]:
        """Get live player state."""
        return self.db.get_live_player_state(player_id)

    def get_all_live_player_states(self) -> Dict[str, Dict]:
        """Get all live player states."""
        return self.db.get_all_live_player_states()

    def get_live_positions(self) -> List[Dict]:
        """Get all open live positions across all players."""
        return self.db.get_live_positions()

    def store_live_trade(self, player_id: str, trade: Dict) -> str:
        """Store a live trade for a player."""
        trade["player_id"] = player_id
        return self.db.store_live_trade(trade)

    def get_live_trades(
        self,
        player_id: str = None,
        start_date: str = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get live trades, optionally filtered by player and date."""
        return self.db.get_live_trades(player_id, start_date, limit)

    def get_player_equity_history(
        self, player_id: str, limit: int = 60
    ) -> List[Dict]:
        """Get equity history for a player."""
        return self.db.get_player_equity_history(player_id, limit)

    def store_live_session(self, session: Dict) -> int:
        """Store a live trading session record."""
        return self.db.store_live_session(session)

    def get_live_sessions(self, limit: int = 30) -> List[Dict]:
        """Get recent live trading sessions."""
        return self.db.get_live_sessions(limit)

    # ─────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────

    def _trade_to_description(self, trade: Dict) -> str:
        """Convert trade data to natural language description for embedding."""
        parts = []

        asset = trade.get("asset", "unknown")
        action = trade.get("action", "unknown")
        strategy = trade.get("strategy_id", "unknown")
        regime = trade.get("market_regime", "unknown")

        parts.append(f"{action} {asset} using {strategy} strategy")
        parts.append(f"Market regime: {regime}")

        if trade.get("entry_price"):
            parts.append(f"Entry: {trade['entry_price']}")
        if trade.get("exit_price"):
            parts.append(f"Exit: {trade['exit_price']}")
        if trade.get("pnl_percent"):
            parts.append(f"Return: {trade['pnl_percent']:.2f}%")
        if trade.get("vix_level"):
            parts.append(f"VIX: {trade['vix_level']}")
        if trade.get("notes"):
            parts.append(f"Notes: {trade['notes']}")

        return ". ".join(parts)
