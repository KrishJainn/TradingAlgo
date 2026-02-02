"""
Tests for AQTIS Multi-Agent Orchestrator.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aqtis.memory.memory_layer import MemoryLayer
from aqtis.llm.base import MockLLMProvider
from aqtis.orchestrator.orchestrator import MultiAgentOrchestrator


@pytest.fixture
def system():
    tmpdir = tempfile.mkdtemp()
    memory = MemoryLayer(
        db_path=os.path.join(tmpdir, "test.db"),
        vector_path=os.path.join(tmpdir, "vectors"),
    )
    llm = MockLLMProvider(
        responses={
            "analyze": '{"should_trade": true, "confidence": 0.75, "reasoning": "test"}',
        },
        default_response='{"result": "ok"}',
    )
    orchestrator = MultiAgentOrchestrator(
        memory=memory,
        llm=llm,
        config={"portfolio_value": 100000},
    )
    return memory, orchestrator


class TestPreTradeWorkflow:
    def test_skip_when_no_strategies(self, system):
        memory, orchestrator = system
        result = orchestrator.pre_trade_workflow({
            "asset": "RELIANCE.NS",
            "action": "BUY",
            "price": 2500,
        })
        assert result["decision"] in ("skip", "error")

    def test_execute_with_strategy(self, system):
        memory, orchestrator = system

        # Add a strategy
        memory.store_strategy({
            "strategy_id": "test_strat",
            "strategy_name": "Test Strategy",
            "sharpe_ratio": 2.0,
            "win_rate": 0.65,
        })

        result = orchestrator.pre_trade_workflow({
            "asset": "RELIANCE.NS",
            "action": "BUY",
            "price": 2500,
        })
        # Should either execute or reject based on risk
        assert result["decision"] in ("execute", "reject", "skip")


class TestPostTradeWorkflow:
    def test_post_trade_missing(self, system):
        _, orchestrator = system
        result = orchestrator.post_trade_workflow("nonexistent")
        assert "error" in result

    def test_post_trade_success(self, system):
        memory, orchestrator = system

        trade_id = memory.store_trade({
            "asset": "TCS.NS",
            "strategy_id": "test",
            "action": "BUY",
            "pnl": 50,
            "pnl_percent": 2.0,
        })

        result = orchestrator.post_trade_workflow(trade_id)
        assert result["trade_id"] == trade_id
        assert "analysis" in result


class TestDailyRoutine:
    def test_daily_routine(self, system):
        _, orchestrator = system
        result = orchestrator.daily_routine()
        assert "research" in result
        assert "degradation" in result

    def test_agent_statuses(self, system):
        _, orchestrator = system
        statuses = orchestrator.get_agent_statuses()
        assert "strategy_generator" in statuses
        assert "backtester" in statuses

    def test_system_health(self, system):
        _, orchestrator = system
        health = orchestrator.get_system_health()
        assert "agents" in health
        assert "memory" in health
        assert "circuit_breaker" in health


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
