"""
Tests for AQTIS Memory Layer.

Covers StructuredDB, VectorStore, and MemoryLayer facade.
"""

import os
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aqtis.memory.database import StructuredDB
from aqtis.memory.vector_store import VectorStore
from aqtis.memory.memory_layer import MemoryLayer
from aqtis.config.settings import AQTISConfig, load_config


class TestStructuredDB:
    """Tests for SQLite database."""

    def setup_method(self):
        self.db_path = tempfile.mktemp(suffix=".db")
        self.db = StructuredDB(self.db_path)

    def teardown_method(self):
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_store_and_get_trade(self):
        trade_id = self.db.store_trade({
            "asset": "RELIANCE.NS",
            "strategy_id": "momentum_v1",
            "action": "BUY",
            "entry_price": 2500.0,
            "exit_price": 2550.0,
            "pnl": 50.0,
            "pnl_percent": 2.0,
            "market_regime": "trending_up",
        })

        trade = self.db.get_trade(trade_id)
        assert trade is not None
        assert trade["asset"] == "RELIANCE.NS"
        assert trade["pnl"] == 50.0

    def test_get_trades_with_filters(self):
        for asset in ["RELIANCE.NS", "TCS.NS", "RELIANCE.NS"]:
            self.db.store_trade({
                "asset": asset,
                "strategy_id": "test",
                "action": "BUY",
                "pnl": 10.0 if asset == "RELIANCE.NS" else -5.0,
            })

        reliance_trades = self.db.get_trades(asset="RELIANCE.NS")
        assert len(reliance_trades) == 2

        wins = self.db.get_trades(outcome="win")
        assert len(wins) == 2

    def test_store_and_get_prediction(self):
        pred_id = self.db.store_prediction({
            "strategy_id": "test",
            "asset": "INFY.NS",
            "predicted_return": 0.05,
            "predicted_confidence": 0.75,
        })

        pred = self.db.get_prediction(pred_id)
        assert pred is not None
        assert pred["predicted_return"] == 0.05

    def test_update_prediction_outcome(self):
        pred_id = self.db.store_prediction({
            "strategy_id": "test",
            "asset": "TCS.NS",
            "predicted_return": 0.03,
            "predicted_confidence": 0.7,
        })

        self.db.update_prediction(pred_id, {
            "actual_return": 0.02,
            "direction_correct": 1,
            "was_profitable": 1,
        })

        pred = self.db.get_prediction(pred_id)
        assert pred["actual_return"] == 0.02
        assert pred["direction_correct"] == 1

    def test_store_strategy(self):
        self.db.store_strategy({
            "strategy_id": "mean_rev_v1",
            "strategy_name": "Mean Reversion V1",
            "strategy_type": "mean_reversion",
            "total_trades": 50,
            "win_rate": 0.62,
            "sharpe_ratio": 1.8,
        })

        strategy = self.db.get_strategy("mean_rev_v1")
        assert strategy is not None
        assert strategy["win_rate"] == 0.62

    def test_get_active_strategies(self):
        for i in range(3):
            self.db.store_strategy({
                "strategy_id": f"strat_{i}",
                "strategy_name": f"Strategy {i}",
                "sharpe_ratio": float(i),
            })

        active = self.db.get_active_strategies()
        assert len(active) == 3

    def test_store_market_state(self):
        self.db.store_market_state({
            "vix": 18.5,
            "vol_regime": "medium",
            "spy_trend_strength": 0.6,
        })

        state = self.db.get_latest_market_state()
        assert state is not None
        assert state["vix"] == 18.5

    def test_store_risk_event(self):
        self.db.store_risk_event({
            "event_type": "circuit_breaker",
            "reason": "Daily loss limit",
        })

        events = self.db.get_risk_events()
        assert len(events) == 1

    def test_get_stats(self):
        self.db.store_trade({"asset": "X", "strategy_id": "Y", "action": "BUY"})
        stats = self.db.get_stats()
        assert stats["trades"] == 1


class TestVectorStore:
    """Tests for ChromaDB vector store."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = VectorStore(persist_dir=self.tmpdir)

    def test_add_and_search_research(self):
        self.store.add_research({
            "text": "This paper studies momentum trading strategies in equity markets",
            "metadata": {"title": "Momentum in Equities", "source": "arxiv"},
        })

        results = self.store.search_research("momentum trading", top_k=5)
        assert len(results) >= 1
        assert "momentum" in results[0]["text"].lower()

    def test_add_and_find_trade_pattern(self):
        self.store.add_trade_pattern({
            "trade_id": "t1",
            "text": "BUY RELIANCE in trending market with high momentum",
            "metadata": {"strategy_id": "momentum_v1", "outcome": "win"},
        })

        similar = self.store.find_similar_trades("BUY RELIANCE trending momentum", top_k=5)
        assert len(similar) >= 1

    def test_get_stats(self):
        self.store.add_research({"text": "Test paper"})
        stats = self.store.get_stats()
        assert stats["trading_research"] >= 1


class TestMemoryLayer:
    """Tests for the unified MemoryLayer facade."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test.db")
        self.vector_path = os.path.join(self.tmpdir, "vectors")
        self.memory = MemoryLayer(db_path=self.db_path, vector_path=self.vector_path)

    def test_store_and_retrieve_trade(self):
        trade_id = self.memory.store_trade({
            "asset": "INFY.NS",
            "strategy_id": "test_strat",
            "action": "BUY",
            "entry_price": 1500,
            "pnl": 25,
            "pnl_percent": 1.67,
            "market_regime": "trending_up",
        })

        trade = self.memory.get_trade(trade_id)
        assert trade["asset"] == "INFY.NS"

    def test_store_trade_with_prediction(self):
        trade_id = self.memory.store_trade(
            trade={
                "asset": "TCS.NS",
                "strategy_id": "test",
                "action": "BUY",
            },
            prediction={
                "strategy_id": "test",
                "asset": "TCS.NS",
                "predicted_return": 0.03,
                "predicted_confidence": 0.8,
            },
        )

        trade = self.memory.get_trade(trade_id)
        assert trade is not None

    def test_get_strategy_performance(self):
        for i in range(10):
            self.memory.db.store_trade({
                "asset": "X.NS",
                "strategy_id": "perf_test",
                "action": "BUY",
                "pnl": 10 if i % 2 == 0 else -5,
                "pnl_percent": 1.0 if i % 2 == 0 else -0.5,
            })

        perf = self.memory.get_strategy_performance("perf_test")
        assert perf["total_trades"] == 10
        assert perf["win_rate"] == 0.5

    def test_search_research(self):
        self.memory.store_research({
            "text": "Analysis of high frequency trading patterns in Indian markets",
            "metadata": {"title": "HFT in India"},
        })

        results = self.memory.search_research("high frequency trading")
        assert len(results) >= 1

    def test_get_stats(self):
        stats = self.memory.get_stats()
        assert "trades" in stats
        assert "vector_collections" in stats


class TestConfig:
    """Tests for configuration system."""

    def test_default_config(self):
        config = AQTISConfig()
        assert config.risk.max_position_size == 0.10
        assert config.execution.initial_capital == 100000.0

    def test_config_validation(self):
        config = AQTISConfig()
        issues = config.validate()
        # API key might not be set
        assert isinstance(issues, list)

    def test_load_config_default(self):
        config = load_config("nonexistent.yaml")
        assert isinstance(config, AQTISConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
