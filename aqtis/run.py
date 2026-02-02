"""
AQTIS Runner - End-to-end entry point.

Usage:
    python -m aqtis.run                  # Run daily routine
    python -m aqtis.run --analyze RELIANCE.NS
    python -m aqtis.run --dashboard
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aqtis.config.settings import load_config
from aqtis.memory.memory_layer import MemoryLayer
from aqtis.llm.base import MockLLMProvider
from aqtis.orchestrator.orchestrator import MultiAgentOrchestrator


def _get_llm(config):
    """Initialize the best available LLM provider."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            from aqtis.llm.gemini_provider import GeminiProvider
            provider = GeminiProvider(api_key=api_key)
            if provider.is_available():
                return provider
        except Exception:
            pass

    logging.getLogger("aqtis").warning("No LLM API key found, using MockLLMProvider")
    return MockLLMProvider(
        default_response='{"result": "ok", "should_trade": false, "confidence": 0.5}',
    )


def build_system(config_path: str = "aqtis_config.yaml"):
    """Build the full AQTIS system and return (memory, orchestrator)."""
    config = load_config(config_path)

    # Default paths for memory storage
    data_dir = Path("aqtis_data")
    data_dir.mkdir(exist_ok=True)

    memory = MemoryLayer(
        db_path=str(data_dir / "aqtis.db"),
        vector_path=str(data_dir / "vectors"),
    )

    llm = _get_llm(config)

    orchestrator = MultiAgentOrchestrator(
        memory=memory,
        llm=llm,
        config={
            "portfolio_value": config.execution.initial_capital,
            "max_position_size": config.risk.max_position_size,
            "max_daily_loss": config.risk.max_daily_loss,
            "max_drawdown": config.risk.max_drawdown,
        },
    )

    return memory, orchestrator, config


def run_analysis(symbol: str, memory, orchestrator):
    """Run pre-trade analysis on a symbol."""
    print(f"\n--- Pre-Trade Analysis: {symbol} ---\n")

    result = orchestrator.pre_trade_workflow({
        "asset": symbol,
        "action": "BUY",
    })

    print(f"Decision: {result.get('decision', 'unknown')}")
    if result.get("strategy"):
        print(f"Strategy: {result['strategy'].get('strategy_name', 'N/A')}")
    if result.get("risk_check"):
        rc = result["risk_check"]
        print(f"Risk Approved: {rc.get('approved', 'N/A')}")
        if rc.get("position_size"):
            print(f"Position Size: {rc['position_size']:.2f}")
    if result.get("backtest"):
        bt = result["backtest"]
        print(f"Backtest Confidence: {bt.get('confidence', 0):.0%}")
        print(f"Similar Trades Win Rate: {bt.get('win_rate', 0):.0%}")

    return result


def run_daily(memory, orchestrator):
    """Run the full daily routine."""
    print("\n--- AQTIS Daily Routine ---\n")
    result = orchestrator.daily_routine()

    if result.get("research"):
        r = result["research"]
        print(f"Research: {r.get('new_papers', 0)} new papers scanned")

    if result.get("degradation"):
        d = result["degradation"]
        alerts = d.get("degradation_alerts", [])
        if alerts:
            print(f"Degradation Alerts: {len(alerts)}")
            for alert in alerts:
                print(f"  - {alert}")
        else:
            print("Degradation: No alerts")

    if result.get("weekly_review"):
        w = result["weekly_review"]
        if isinstance(w, dict) and "overall" in w:
            print(f"Weekly Review: PnL={w['overall'].get('total_pnl', 0):.2f}")

    print("\nDaily routine complete.")
    return result


def main():
    parser = argparse.ArgumentParser(description="AQTIS - Adaptive Quantitative Trading Intelligence System")
    parser.add_argument("--config", default="aqtis_config.yaml", help="Config file path")
    parser.add_argument("--analyze", metavar="SYMBOL", help="Run pre-trade analysis on a symbol")
    parser.add_argument("--daily", action="store_true", help="Run daily routine")
    parser.add_argument("--dashboard", action="store_true", help="Launch Streamlit dashboard")
    parser.add_argument("--health", action="store_true", help="Check system health")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.dashboard:
        import subprocess
        dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
        return

    memory, orchestrator, config = build_system(args.config)

    if args.health:
        health = orchestrator.get_system_health()
        print("\n--- System Health ---\n")
        print(f"Agents: {len(health.get('agents', {}))} initialized")
        print(f"Memory: {health.get('memory', {}).get('trades', 0)} trades stored")
        cb = health.get("circuit_breaker", False)
        cb_active = cb.get("active") if isinstance(cb, dict) else cb
        print(f"Circuit Breaker: {'ACTIVE' if cb_active else 'OFF'}")
        issues = config.validate()
        if issues:
            print(f"\nConfig Issues ({len(issues)}):")
            for issue in issues:
                print(f"  ! {issue}")
        else:
            print("\nConfig: All checks passed")
        return

    if args.analyze:
        run_analysis(args.analyze, memory, orchestrator)
    elif args.daily:
        run_daily(memory, orchestrator)
    else:
        # Default: show status and run health check
        print("AQTIS v0.1.0")
        print("=" * 40)
        stats = memory.get_stats()
        print(f"Trades: {stats.get('trades', 0)}")
        print(f"Strategies: {stats.get('strategies', 0)}")
        print(f"Predictions: {stats.get('predictions', 0)}")
        print(f"\nUse --analyze SYMBOL, --daily, or --dashboard")


if __name__ == "__main__":
    main()
