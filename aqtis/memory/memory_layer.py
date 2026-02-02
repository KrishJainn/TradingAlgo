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

logger = logging.getLogger(__name__)


class MemoryLayer:
    """
    Central interface for all memory operations.

    Combines structured storage (trades, predictions, strategies)
    with semantic search (research papers, trade patterns).
    """

    def __init__(self, db_path: str = "aqtis.db", vector_path: str = "aqtis_vectors"):
        self.db = StructuredDB(db_path)
        self.vectors = VectorStore(vector_path)

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
