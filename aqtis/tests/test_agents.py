"""
Tests for AQTIS Agents.

Tests each agent with mocked dependencies.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aqtis.memory.memory_layer import MemoryLayer
from aqtis.llm.base import MockLLMProvider
from aqtis.agents.prediction_tracker import PredictionTrackingAgent
from aqtis.agents.risk_manager import RiskManagementAgent
from aqtis.agents.backtester import BacktestingAgent
from aqtis.agents.strategy_generator import StrategyGeneratorAgent
from aqtis.agents.post_mortem import PostMortemAgent
from aqtis.agents.researcher import ResearchAgent


@pytest.fixture
def memory():
    tmpdir = tempfile.mkdtemp()
    return MemoryLayer(
        db_path=os.path.join(tmpdir, "test.db"),
        vector_path=os.path.join(tmpdir, "vectors"),
    )


@pytest.fixture
def mock_llm():
    return MockLLMProvider(
        responses={
            "analyze": '{"should_trade": true, "confidence": 0.7, "reasoning": "Test"}',
            "improve": '{"analysis": "Good", "parameter_changes": {}, "expected_improvement": "5%"}',
            "insights": '{"outcome_summary": "Won", "lessons_learned": ["Lesson1"], "actionable_changes": [], "primary_factors": [], "error_attribution": {}}',
            "weekly": '{"best_strategy": "test", "key_insights": ["insight1"], "focus_next_week": []}',
        },
        default_response='{"result": "ok"}',
    )


class TestPredictionTracker:
    def test_record_prediction(self, memory):
        agent = PredictionTrackingAgent(memory)
        result = agent.run({
            "action": "record_prediction",
            "prediction": {
                "strategy_id": "test",
                "asset": "RELIANCE.NS",
                "predicted_return": 0.02,
                "predicted_confidence": 0.7,
            },
        })
        assert "prediction_id" in result

    def test_record_outcome(self, memory):
        agent = PredictionTrackingAgent(memory)

        # Record prediction first
        pred_result = agent.run({
            "action": "record_prediction",
            "prediction": {
                "strategy_id": "test",
                "asset": "X.NS",
                "predicted_return": 0.05,
                "predicted_confidence": 0.8,
            },
        })

        # Record outcome
        outcome_result = agent.run({
            "action": "record_outcome",
            "prediction_id": pred_result["prediction_id"],
            "outcome": {"actual_return": 0.03},
        })
        assert "errors" in outcome_result

    def test_calibration_report(self, memory):
        agent = PredictionTrackingAgent(memory)
        result = agent.run({"action": "calibrate"})
        assert "bins" in result
        assert "total_predictions" in result

    def test_detect_degradation(self, memory):
        agent = PredictionTrackingAgent(memory)
        result = agent.run({"action": "detect_degradation"})
        assert "degradation_alerts" in result


class TestRiskManager:
    def test_validate_trade_approved(self, memory):
        agent = RiskManagementAgent(memory)
        result = agent.run({
            "action": "validate",
            "trade": {
                "confidence": 0.8,
                "entry_price": 100,
                "position_size": 10,
                "portfolio_value": 100000,
            },
        })
        assert result["approved"] is True

    def test_validate_trade_low_confidence(self, memory):
        agent = RiskManagementAgent(memory)
        result = agent.run({
            "action": "validate",
            "trade": {
                "confidence": 0.3,
                "entry_price": 100,
                "position_size": 10,
            },
        })
        assert result["approved"] is False
        assert "confidence_ok" in result.get("rejection_reasons", [])

    def test_position_sizing(self, memory):
        agent = RiskManagementAgent(memory)
        result = agent.run({
            "action": "position_size",
            "prediction": {
                "predicted_confidence": 0.7,
                "predicted_return": 0.02,
            },
            "portfolio_value": 100000,
        })
        assert "position_size" in result
        assert result["position_size"] >= 0

    def test_circuit_breaker(self, memory):
        agent = RiskManagementAgent(memory)
        agent.activate_circuit_breaker("Test")
        assert agent.circuit_breaker_active

        result = agent.run({"action": "validate", "trade": {"confidence": 0.9}})
        assert result["approved"] is False

        agent.deactivate_circuit_breaker()
        assert not agent.circuit_breaker_active


class TestBacktester:
    def test_instant_backtest_no_data(self, memory):
        agent = BacktestingAgent(memory)
        result = agent.run({
            "action": "instant",
            "strategy": {"strategy_id": "test"},
            "signal": {"asset": "X.NS"},
        })
        assert result.get("insufficient_data") or result.get("confidence", 0) == 0

    def test_rolling_backtest_no_data(self, memory):
        agent = BacktestingAgent(memory)
        result = agent.run({
            "action": "rolling",
            "strategy": {"strategy_id": "nonexistent"},
        })
        assert result["recommendation"] == "Insufficient data"


class TestStrategyGenerator:
    def test_analyze_signal_no_strategies(self, memory, mock_llm):
        agent = StrategyGeneratorAgent(memory, mock_llm)
        result = agent.run({
            "action": "analyze_signal",
            "signal": {"asset": "RELIANCE.NS"},
        })
        assert result["should_trade"] is False

    def test_analyze_signal_with_strategy(self, memory, mock_llm):
        memory.store_strategy({
            "strategy_id": "test_strat",
            "strategy_name": "Test",
            "sharpe_ratio": 1.5,
            "win_rate": 0.6,
        })

        agent = StrategyGeneratorAgent(memory, mock_llm)
        result = agent.run({
            "action": "analyze_signal",
            "signal": {"asset": "RELIANCE.NS"},
        })
        assert "strategy" in result or "should_trade" in result


class TestPostMortem:
    def test_analyze_trade(self, memory, mock_llm):
        trade_id = memory.store_trade({
            "asset": "TCS.NS",
            "strategy_id": "test",
            "action": "BUY",
            "entry_price": 3500,
            "exit_price": 3600,
            "pnl": 100,
            "pnl_percent": 2.86,
        })

        agent = PostMortemAgent(memory, mock_llm)
        result = agent.run({"action": "analyze_trade", "trade_id": trade_id})
        assert result.get("outcome") == "win"

    def test_weekly_review_empty(self, memory, mock_llm):
        agent = PostMortemAgent(memory, mock_llm)
        result = agent.run({"action": "weekly_review"})
        assert "message" in result or "overall" in result

    def test_extract_lessons(self, memory):
        agent = PostMortemAgent(memory)
        result = agent.run({
            "action": "extract_lessons",
            "lookback_days": 30,
        })
        assert "period_days" in result or "message" in result


class TestResearcher:
    def test_search_empty_db(self, memory, mock_llm):
        agent = ResearchAgent(memory, mock_llm)
        result = agent.run({
            "action": "search",
            "query": "momentum trading",
        })
        assert result.get("papers_found", 0) == 0

    def test_add_paper(self, memory, mock_llm):
        agent = ResearchAgent(memory, mock_llm)
        result = agent.run({
            "action": "add_paper",
            "paper": {
                "title": "Test Paper",
                "text": "This is a test paper about momentum trading strategies",
            },
        })
        assert "doc_id" in result

    def test_search_after_add(self, memory, mock_llm):
        agent = ResearchAgent(memory, mock_llm)

        agent.run({
            "action": "add_paper",
            "paper": {
                "title": "Momentum Strategies",
                "text": "Analysis of cross-sectional momentum in equity markets",
            },
        })

        result = agent.run({
            "action": "search",
            "query": "momentum equity",
        })
        assert result.get("papers_found", 0) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
