"""
Daily Runner — Main Paper Trading Pipeline with 3 run modes.

Orchestrates all modules in the 5-Player Coach Trading System.
Supports 3 intraday modes scheduled at different times:

Modes:
    "intelligence"  — 09:00 IST (pre-market)
        Pull data, detect regime, analyse sentiment, generate
        market heatmap & morning brief.

    "refresh"       — 12:30 IST (midday)
        Refresh news sentiment, compute market breadth,
        flag significant sentiment shifts since morning.

    "full"          — 17:30 IST (post-market)
        Complete 14-step pipeline:
         1. Pull latest OHLCV data (yfinance)
         2. Regime detection (HMM)
         3. News sentiment analysis
         4. Generate per-player signals
         5. Player debate (LLM)
         6. Combine signals (Bayesian + regime-aware)
         7. Risk management filter
         8. Paper execution (open/close positions)
         9. Record metrics snapshot
        10. Player reflection (every N trades)
        11. Coaching cycle (every M trades)
        12. Player evolution (every K days)
        13. Generate daily report
        14. Error handling & logging

Usage:
    from daily_runner import DailyRunner

    runner = DailyRunner()

    # Morning intelligence (09:00 IST)
    report = runner.run(mode="intelligence")

    # Midday refresh (12:30 IST)
    report = runner.run(mode="refresh")

    # Full post-market pipeline (17:30 IST)
    report = runner.run(mode="full")

    # Simulation mode
    report = runner.run(mode="full", sim_date="2025-01-15", sim_prices={...})
"""

from __future__ import annotations

import json
import logging
import math
import os

# Load .env file so GEMINI_API_KEY is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, rely on shell env
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Local imports (with graceful fallbacks) ──────────────────────────────

from paper_trading_config import CONFIG, UNIVERSE, PLAYER_IDS, PLAYER_LABELS, ensure_directories
from position_tracker import PositionTracker
from trade_journal import TradeJournal
from metrics_store import MetricsStore, PLAYER_IDS as MS_PLAYER_IDS

# Optional modules — degrade gracefully if missing
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

try:
    from regime_detector import RegimeDetector, load_nifty_data
    _HAS_REGIME = True
except ImportError:
    _HAS_REGIME = False

try:
    from sentiment_engine import get_sentiment_signals, get_sentiment_context
    _HAS_SENTIMENT = True
except ImportError:
    _HAS_SENTIMENT = False

try:
    from signal_combiner import (
        PlayerSignal, RegimeAwareCombiner, BayesianPlayerWeights, TradeOutcome,
    )
    _HAS_COMBINER = True
except ImportError:
    _HAS_COMBINER = False

try:
    from player_debate import DebateRound, DebateSignal
    _HAS_DEBATE = True
except ImportError:
    _HAS_DEBATE = False

try:
    from portfolio_risk_manager import PortfolioRiskManager, RiskLimits
    _HAS_RISK = True
except ImportError:
    _HAS_RISK = False

try:
    from transaction_costs import TransactionCostModel
    _HAS_COSTS = True
except ImportError:
    _HAS_COSTS = False

try:
    from player_reflection import ReflectionScheduler, TradeReflectionEngine, RuleManager, create_reflection_system
    _HAS_REFLECTION = True
except ImportError:
    _HAS_REFLECTION = False

try:
    from regime_aware_coach import RegimeAwareCoach
    _HAS_COACH = True
except ImportError:
    _HAS_COACH = False

try:
    from coach_system.coaches.ai_coach import AICoach
    _HAS_AI_COACH = True
except ImportError:
    _HAS_AI_COACH = False

try:
    from player_evolution import PlayerEvolutionEngine
    _HAS_EVOLUTION = True
except ImportError:
    _HAS_EVOLUTION = False

try:
    from smart_exit_engine import SmartExitEngine, PositionState, ExitDecision
    _HAS_SMART_EXIT = True
except ImportError:
    _HAS_SMART_EXIT = False

try:
    from claude_exit_reasoning import GeminiExitReasoner
    _HAS_GEMINI_EXIT = True
except ImportError:
    _HAS_GEMINI_EXIT = False

try:
    from event_calendar import EventCalendar
    _HAS_EVENT_CALENDAR = True
except ImportError:
    _HAS_EVENT_CALENDAR = False

try:
    from mfe_mae_analyzer import MFEMAEAnalyzer
    _HAS_MFE_MAE = True
except ImportError:
    _HAS_MFE_MAE = False

try:
    from exit_evolution import ExitEvolutionEngine, save_evolved_params, load_evolved_params
    _HAS_EXIT_EVOLUTION = True
except ImportError:
    _HAS_EXIT_EVOLUTION = False

try:
    from data_cache.indicator_computer import IndicatorComputer
    _HAS_INDICATORS = True
except ImportError:
    _HAS_INDICATORS = False


# ═══════════════════════════════════════════════════════════════════════════
# Evolved player config loader + real indicator signal computation
# ═══════════════════════════════════════════════════════════════════════════

# Fallback map: indicator name → column that exists in indicator data
_INDICATOR_FALLBACK: Dict[str, str] = {
    "RSI_14": "RSI_14_norm", "MFI_14": "MFI_14_norm",
    "ATR_ratio": "ATR_ratio_norm", "NATR": "NATR_norm",
    "BB_width_raw": "BB_width_raw_norm",
    "PSAR_signal": "PSAR_signal", "ICHIMOKU_signal": "ICHIMOKU_signal",
    "DONCHIAN_breakout": "DONCHIAN_breakout", "HA_trend": "HA_trend",
    "BB_squeeze_signal": "BB_squeeze_signal", "STOCH_divergence": "STOCH_divergence",
    "ELDER_force": "ELDER_force", "VWAP_deviation": "VWAP_deviation",
    "VWAP_dev_raw": "VWAP_dev_raw", "OBV_slope": "OBV_slope",
    "KC_breakout": "KC_breakout", "ROC_20": "ROC_20",
    "MACD_hist_slope": "MACD_hist_slope", "HIGH_52W_prox": "HIGH_52W_prox",
    "SUPERTREND_10_2": "ATR_14_norm",
}


def _load_evolved_configs() -> Dict[str, Any]:
    """Load evolved player configs from disk. Falls back to equal-weight defaults."""
    cfg_path = Path(CONFIG.get("evolved_configs_file",
                               str(Path(__file__).parent / "evolved_player_configs.json")))
    try:
        if cfg_path.exists():
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            configs = raw.get("configs", {})
            if configs and all(pid in configs for pid in PLAYER_IDS):
                logger.info(f"[EvolvedConfigs] Loaded {len(configs)} player configs "
                           f"(runs={raw.get('total_runs', 0)})")
                return configs
            logger.warning("[EvolvedConfigs] Config file incomplete, using defaults")
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"[EvolvedConfigs] Failed to load: {e}, using defaults")

    # Default equal-weight fallback
    default_indicators = ["PSAR_signal", "RSI_14", "ROC_20", "HIGH_52W_prox", "VWAP_dev_raw"]
    return {
        pid: {
            "label": PLAYER_LABELS.get(pid, pid),
            "weights": {ind: 0.20 for ind in default_indicators},
            "entry_threshold": 0.30,
            "exit_threshold": -0.12,
            "min_hold_bars": 3,
        }
        for pid in PLAYER_IDS
    }


def _compute_real_signal(
    indicators: pd.DataFrame,
    weights: Dict[str, float],
    idx: int,
) -> Tuple[float, float]:
    """
    Compute a player's weighted signal at bar index `idx` from real indicators.
    Returns (direction, confidence) where:
      direction ∈ [-1.0, 1.0]
      confidence = agreement fraction (how many indicators agree on direction)
    """
    weighted_sum = 0.0
    total_weight = 0.0
    agree_pos = 0
    agree_neg = 0
    total = 0

    for ind_name, weight in weights.items():
        if weight == 0:
            continue
        # Try name_norm first, then fallback
        col = ind_name + "_norm"
        if col not in indicators.columns:
            col = _INDICATOR_FALLBACK.get(ind_name, ind_name)
        if col not in indicators.columns:
            continue

        val = indicators.iloc[idx].get(col)
        if pd.isna(val):
            continue

        weighted_sum += val * weight
        total_weight += abs(weight)
        total += 1

        if val > 0.05:
            agree_pos += 1
        elif val < -0.05:
            agree_neg += 1

    if total_weight == 0:
        return 0.0, 0.0

    direction = float(np.clip(weighted_sum / total_weight, -1.0, 1.0))
    confidence = max(agree_pos, agree_neg) / total if total > 0 else 0.0
    return direction, confidence


def _compute_ensemble_consensus_multiplier(
    player_signals: Dict[str, float],
    current_pid: str,
    evo_config: Dict[str, Any],
) -> float:
    """
    Ensemble consensus multiplier from EVOLVED+ENSEMBLE.
    Counts how many other players agree with current_pid's direction.
    Returns 0.5x (lonely), 1.0x (standard), or 1.2x (strong consensus).
    """
    my_dir = player_signals.get(current_pid, 0.0)
    if abs(my_dir) < 0.05:
        return evo_config.get("ensemble_standard_multiplier", 1.0)

    my_sign = 1.0 if my_dir > 0 else -1.0
    n_agreeing = 0
    for other_pid, other_sig in player_signals.items():
        if other_pid == current_pid:
            continue
        other_sign = 1.0 if other_sig > 0 else -1.0
        if abs(other_sig) > 0.1 and other_sign == my_sign:
            n_agreeing += 1

    if n_agreeing <= 1:
        return evo_config.get("ensemble_lonely_multiplier", 0.5)
    elif n_agreeing >= 3:
        return evo_config.get("ensemble_strong_multiplier", 1.2)
    else:
        return evo_config.get("ensemble_standard_multiplier", 1.0)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Token budget — prevents runaway Gemini API costs
# ═══════════════════════════════════════════════════════════════════════════

_CHARS_PER_TOKEN = 4  # rough estimate for Gemini


class TokenBudget:
    """Daily token budget tracker to prevent runaway API costs."""

    def __init__(self, daily_limit: int = 200_000):
        self.daily_limit = daily_limit
        self.tokens_used_today = 0
        self.calls_today = 0
        self._date = date.today()

    def check_and_log(self, input_tokens: int, output_tokens: int) -> None:
        """Record token usage and raise if daily budget exceeded."""
        if date.today() != self._date:
            logger.info(f"[TokenBudget] New day — resetting. Yesterday: "
                       f"{self.tokens_used_today:,} tokens, {self.calls_today} calls")
            self.tokens_used_today = 0
            self.calls_today = 0
            self._date = date.today()

        self.tokens_used_today += input_tokens + output_tokens
        self.calls_today += 1

        if self.tokens_used_today > self.daily_limit:
            raise RuntimeError(
                f"Daily token budget exceeded: {self.tokens_used_today:,} / "
                f"{self.daily_limit:,}. Pipeline halted. "
                f"Check logs for runaway calls ({self.calls_today} calls today)."
            )

    def summary(self) -> str:
        pct = (self.tokens_used_today / self.daily_limit * 100) if self.daily_limit else 0
        return (f"{self.tokens_used_today:,}/{self.daily_limit:,} tokens "
                f"({pct:.0f}%), {self.calls_today} calls")


TOKEN_BUDGET = TokenBudget(daily_limit=300_000)


# ═══════════════════════════════════════════════════════════════════════════
# LLM wrapper (for debate/reflection modules that expect .generate(prompt))
# ═══════════════════════════════════════════════════════════════════════════

class _GeminiLLM:
    """Minimal LLM wrapper providing `.generate(prompt)` interface.

    All LLM calls in the pipeline flow through this class, which enforces
    the daily token budget to prevent runaway API costs.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self._model = model
        self._client = None

    @property
    def model(self) -> str:
        return self._model

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from google import genai
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if api_key:
                self._client = genai.Client(api_key=api_key)
        except Exception as e:
            logger.warning(f"Gemini client init failed: {e}")

    def generate(self, prompt: str) -> str:
        self._ensure_client()
        if self._client is None:
            return '{"error": "Gemini unavailable"}'
        try:
            # Estimate input tokens and check budget BEFORE calling
            input_tokens = len(prompt) // _CHARS_PER_TOKEN
            remaining = TOKEN_BUDGET.daily_limit - TOKEN_BUDGET.tokens_used_today
            if remaining < input_tokens:
                logger.warning(f"[TokenBudget] Skipping call — only {remaining:,} tokens left, "
                              f"need ~{input_tokens:,}")
                return '{"error": "Token budget exhausted"}'

            from google.genai import types
            config = types.GenerateContentConfig(temperature=0.3)
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )
            result = response.text or ""

            # Log REAL token usage from Gemini response metadata
            usage = getattr(response, "usage_metadata", None)
            if usage:
                real_input = getattr(usage, "prompt_token_count", 0) or 0
                real_output = getattr(usage, "candidates_token_count", 0) or 0
                TOKEN_BUDGET.check_and_log(real_input, real_output)
            else:
                # Fallback: estimate from character count
                output_tokens = len(result) // _CHARS_PER_TOKEN
                TOKEN_BUDGET.check_and_log(input_tokens, output_tokens)

            return result
        except RuntimeError:
            raise  # re-raise budget exceeded
        except Exception as e:
            logger.warning(f"Gemini generate failed: {e}")
            return '{"error": "Gemini call failed"}'


# ═══════════════════════════════════════════════════════════════════════════
# DailyRunner
# ═══════════════════════════════════════════════════════════════════════════

class DailyRunner:
    """
    Orchestrates the full 14-step daily paper trading pipeline.

    Can run in live mode (pulls real data from yfinance, calls Gemini)
    or simulation mode (accepts injected prices and skips external calls).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or CONFIG
        ensure_directories()

        # ── Core components ──────────────────────────────────────────
        self.tracker = PositionTracker(
            capital=self.config["starting_capital"],
            num_players=self.config.get("num_players", 5),
            capital_per_player=self.config.get("capital_per_player", 100_000),
            positions_file=self.config["positions_file"],
            trailing_stop_pct=self.config["trailing_stop_pct"],
        )
        self.journal = TradeJournal(
            journal_file=self.config["journal_file"],
        )
        self.store = MetricsStore(
            metrics_dir=Path(self.config["metrics_dir"]),
        )

        # ── Optional components (initialised lazily) ─────────────────
        self._llm: Optional[_GeminiLLM] = None
        self._regime_detector: Optional[Any] = None
        self._bayesian_weights: Optional[Any] = None
        self._combiner: Optional[Any] = None
        self._risk_manager: Optional[Any] = None
        self._cost_model: Optional[Any] = None
        self._reflection_scheduler: Optional[Any] = None
        self._evolution_engine: Optional[Any] = None

        # ── EVOLVED+ENSEMBLE components ───────────────────────────────
        self._evolved_configs: Dict[str, Any] = _load_evolved_configs()
        self._indicator_computer: Optional[Any] = None
        self._indicator_cache: Dict[str, pd.DataFrame] = {}   # sym → indicator df
        self._ohlcv_cache: Dict[str, pd.DataFrame] = {}       # sym → OHLCV df

        # ── Event Calendar ────────────────────────────────────────────
        self._event_calendar: Optional[Any] = None
        if _HAS_EVENT_CALENDAR:
            try:
                self._event_calendar = EventCalendar()
                logger.info("[EventCalendar] Initialised")
            except Exception as e:
                logger.warning(f"[EventCalendar] Init failed: {e}")

        # ── Pre-load RegimeDetector for HMM model access ──────────────
        # The RegimeDetector auto-loads a saved HMM model from disk.
        # We grab the model + state_map early so SmartExitEngine can use
        # the transition matrix for regime_transition signal.
        _hmm_model = None
        _regime_state_map: Dict[str, int] = {}
        if _HAS_REGIME:
            try:
                if self._regime_detector is None:
                    model_path = Path(__file__).parent / "regime_hmm_model.joblib"
                    self._regime_detector = RegimeDetector(model_path=model_path)
                if self._regime_detector._fitted and self._regime_detector._model is not None:
                    _hmm_model = self._regime_detector._model
                    # Invert _state_map {int→label} to {label→int}
                    _regime_state_map = {
                        label: state_int
                        for state_int, label in self._regime_detector._state_map.items()
                    }
                    logger.info(f"[HMM] Loaded transition matrix for SmartExitEngine "
                               f"(states: {_regime_state_map})")
            except Exception as e:
                logger.warning(f"[HMM] Failed to load for SmartExitEngine: {e}")

        # ── SmartExitEngine ────────────────────────────────────────────
        self._exit_engine: Optional[Any] = None
        if _HAS_SMART_EXIT:
            trade_hist = self.journal.get_all_trades() if hasattr(self.journal, 'get_all_trades') else []
            self._exit_engine = SmartExitEngine(
                trade_history=trade_hist,
                event_calendar=self._event_calendar,
                hmm_model=_hmm_model,
                regime_state_map=_regime_state_map,
            )
            logger.info("[SmartExitEngine] Initialised (regime_transition=%s)" %
                       ("enabled" if _hmm_model else "disabled"))

        # ── Gemini Exit Reasoner (ambiguous zone adjudication) ──────
        self._gemini_exit: Optional[Any] = None
        if _HAS_GEMINI_EXIT:
            try:
                self._gemini_exit = GeminiExitReasoner()
                logger.info("[GeminiExitReasoner] Initialised")
            except Exception as e:
                logger.warning(f"[GeminiExitReasoner] Init failed: {e}")

        # ── MFE/MAE Analyzer ─────────────────────────────────────────
        self._mfe_analyzer: Optional[Any] = None
        self._mfe_prompt_block: Optional[str] = None
        if _HAS_MFE_MAE:
            try:
                self._mfe_analyzer = MFEMAEAnalyzer(min_trades_for_report=10)
                logger.info("[MFEMAEAnalyzer] Initialised")
            except Exception as e:
                logger.warning(f"[MFEMAEAnalyzer] Init failed: {e}")

        # ── Pipeline state ───────────────────────────────────────────
        self._state = self._load_state()

    # ══════════════════════════════════════════════════════════════════
    # Main entry point
    # ══════════════════════════════════════════════════════════════════

    # ── Pipeline modes ───────────────────────────────────────────
    #   "full"         — all 14 steps (post-market, 17:30 IST)
    #   "intelligence"  — steps 1-3 only (data + regime + sentiment → morning brief)
    #   "refresh"       — steps 1, 3 only (news + breadth refresh, midday)

    VALID_MODES = {"full", "intelligence", "refresh", "premarket", "postmarket"}

    def run(
        self,
        mode: str = "full",
        sim_date: Optional[str] = None,
        sim_prices: Optional[Dict[str, float]] = None,
        sim_ohlcv: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Execute the daily pipeline in the specified mode.

        Modes:
            "full"          — Complete 14-step post-market pipeline (signals + trades + metrics).
            "intelligence"  — Morning brief: pull data, regime detection, sentiment,
                              generate market heatmap & morning brief report.
            "refresh"       — Midday update: refresh news sentiment & market breadth only.

        Args:
            mode:       Pipeline mode — "full", "intelligence", or "refresh".
            sim_date:   Override date (YYYY-MM-DD) for simulation.
            sim_prices: Inject {symbol: close_price} instead of pulling from yfinance.
            sim_ohlcv:  Inject OHLCV DataFrame instead of pulling from yfinance.

        Returns:
            Daily report dict with all pipeline outputs.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}")

        today = sim_date or date.today().isoformat()
        mode_labels = {
            "full": "FULL PIPELINE",
            "intelligence": "MORNING INTELLIGENCE",
            "refresh": "MIDDAY REFRESH",
            "premarket": "PRE-MARKET (signals + debates)",
            "postmarket": "POST-MARKET (execute + stops)",
        }
        report: Dict[str, Any] = {
            "date": today,
            "mode": mode,
            "pipeline_start": datetime.now().isoformat(),
            "steps_completed": [],
            "errors": [],
            "warnings": [],
        }

        logger.info(f"{'='*60}")
        logger.info(f"  {mode_labels[mode]} — {today}")
        logger.info(f"{'='*60}")

        if mode in ("full", "premarket", "postmarket"):
            self.tracker.reset_daily()
            self._state["total_runs"] = self._state.get("total_runs", 0) + 1

        try:
            if mode == "intelligence":
                self._run_intelligence(today, sim_prices, sim_ohlcv, report)
            elif mode == "refresh":
                self._run_refresh(today, sim_prices, sim_ohlcv, report)
            elif mode == "premarket":
                self._run_premarket(today, sim_prices, sim_ohlcv, report)
            elif mode == "postmarket":
                self._run_postmarket(today, sim_prices, sim_ohlcv, report)
            else:
                self._run_full(today, sim_prices, sim_ohlcv, report)

        except Exception as e:
            self._step_14_error(e, report)

        report["pipeline_end"] = datetime.now().isoformat()
        report["success"] = len(report["errors"]) == 0

        logger.info(f"\n{'='*60}")
        logger.info(f"  {mode_labels[mode]} complete: {len(report['steps_completed'])} steps, "
                     f"{len(report['errors'])} errors")
        logger.info(f"{'='*60}\n")

        return report

    # ══════════════════════════════════════════════════════════════════
    # Run modes
    # ══════════════════════════════════════════════════════════════════

    def _run_full(
        self,
        today: str,
        sim_prices: Optional[Dict[str, float]],
        sim_ohlcv: Optional[Any],
        report: Dict,
    ) -> None:
        """Full 14-step post-market pipeline (EVOLVED+ENSEMBLE)."""
        prices, ohlcv_data = self._step_1_pull_data(today, sim_prices, sim_ohlcv, report)
        regime, regime_probs, regime_duration = self._step_2_regime(ohlcv_data, report)

        # Store regime in report for evolution trigger detection
        report["regime"] = {
            "current": regime,
            "probabilities": regime_probs,
            "duration_days": regime_duration,
        }

        sentiment_signals = self._step_3_sentiment(report)
        player_signals = self._step_4_signals(prices, regime, sentiment_signals, report)
        debate_outcome = self._step_5_debate(player_signals, report)
        combined = self._step_6_combine(player_signals, regime, report)
        approved_signals, rejected = self._step_7_risk(combined, report)
        trades_today = self._step_8_execute(
            approved_signals, prices, today, regime, report,
            combined_signals=combined,
            player_signals=player_signals,
        )
        self._step_9_metrics(
            today, prices, regime, regime_probs, regime_duration,
            player_signals, trades_today, report,
        )
        self._step_10_reflection(trades_today, regime, report)
        self._step_11_coaching(regime, report)
        self._step_12_evolution(ohlcv_data, report)
        self._step_12b_exit_evolution(report)
        self._step_13_report(today, report)

        # First-run monitoring — logs everything needed to verify the system
        self._log_first_run_monitor(today, report, combined)

    def _run_intelligence(
        self,
        today: str,
        sim_prices: Optional[Dict[str, float]],
        sim_ohlcv: Optional[Any],
        report: Dict,
    ) -> None:
        """Morning intelligence: data + regime + sentiment + heatmap + brief."""
        # Step 1: Pull market data
        prices, ohlcv_data = self._step_1_pull_data(today, sim_prices, sim_ohlcv, report)

        # Step 2: Regime detection
        regime, regime_probs, regime_duration = self._step_2_regime(ohlcv_data, report)

        # Step 3: News sentiment
        sentiment_signals = self._step_3_sentiment(report)

        # Intelligence-only: generate heatmap + morning brief
        self._step_intel_heatmap(prices, sentiment_signals, regime, report)
        self._step_intel_morning_brief(
            today, prices, regime, regime_probs, regime_duration,
            sentiment_signals, report,
        )

    def _run_refresh(
        self,
        today: str,
        sim_prices: Optional[Dict[str, float]],
        sim_ohlcv: Optional[Any],
        report: Dict,
    ) -> None:
        """Midday refresh: updated news + breadth snapshot."""
        # Step 1: Pull latest prices (intraday snapshot)
        prices, ohlcv_data = self._step_1_pull_data(today, sim_prices, sim_ohlcv, report)

        # Step 3: Refresh news sentiment
        sentiment_signals = self._step_3_sentiment(report)

        # Refresh-only: breadth + sentiment delta
        self._step_refresh_breadth(prices, report)
        self._step_refresh_sentiment_delta(today, sentiment_signals, report)

    def _run_premarket(
        self,
        today: str,
        sim_prices: Optional[Dict[str, float]],
        sim_ohlcv: Optional[Any],
        report: Dict,
    ) -> None:
        """Pre-market run (09:00 IST): signals + debates to decide today's trades.

        Steps: data → regime → sentiment → signals → DEBATE → combine → risk → execute.
        No reflection/coaching/evolution (that's post-market).
        """
        # ── One-time P1 reset (short leverage fix, Feb 2026) ──────────
        _reset_flag = Path(__file__).parent / "data" / ".p1_reset_done"
        if not _reset_flag.exists():
            logger.info("[P1 RESET] Closing all PLAYER_1 positions (short leverage fix)")
            p1_positions = self.tracker.get_positions_for_player("PLAYER_1")
            if p1_positions:
                # Need prices to close — pull data first
                _prices, _ = self._step_1_pull_data(today, sim_prices, sim_ohlcv, report)
                for pos in p1_positions:
                    price = _prices.get(pos.symbol, pos.current_price)
                    trade = self.tracker.close_position(
                        pos.position_id, price, exit_date=today,
                        reason="p1_leverage_reset",
                    )
                    if trade:
                        self.journal.record_trade(trade)
                        logger.info(f"  [P1 RESET] Closed {pos.direction} {pos.symbol} @ ₹{price:,.2f}")
            # Reset P1 capital back to ₹1,00,000
            self.tracker.player_capital["PLAYER_1"] = self.tracker.capital_per_player_initial
            self.tracker._save()
            _reset_flag.parent.mkdir(parents=True, exist_ok=True)
            _reset_flag.write_text(f"Reset done on {today}\n")
            logger.info(f"[P1 RESET] Capital restored to ₹{self.tracker.capital_per_player_initial:,.0f}")

        prices, ohlcv_data = self._step_1_pull_data(today, sim_prices, sim_ohlcv, report)
        regime, regime_probs, regime_duration = self._step_2_regime(ohlcv_data, report)
        report["regime"] = {
            "current": regime,
            "probabilities": regime_probs,
            "duration_days": regime_duration,
        }
        sentiment_signals = self._step_3_sentiment(report)
        player_signals = self._step_4_signals(prices, regime, sentiment_signals, report)
        debate_outcome = self._step_5_debate(player_signals, report)
        combined = self._step_6_combine(player_signals, regime, report)
        approved_signals, rejected = self._step_7_risk(combined, report)
        trades_today = self._step_8_execute(
            approved_signals, prices, today, regime, report,
            combined_signals=combined,
            player_signals=player_signals,
        )
        self._step_9_metrics(
            today, prices, regime, regime_probs, regime_duration,
            player_signals, trades_today, report,
        )
        self._step_13_report(today, report)
        self._log_first_run_monitor(today, report, combined)

    def _run_postmarket(
        self,
        today: str,
        sim_prices: Optional[Dict[str, float]],
        sim_ohlcv: Optional[Any],
        report: Dict,
    ) -> None:
        """Post-market run (15:45 IST): update prices, check stops, reflect, coach.

        Steps: data → regime → signals → combine → risk → execute → metrics →
               reflection → coaching → evolution → report.
        NO debates — those ran in pre-market.
        """
        prices, ohlcv_data = self._step_1_pull_data(today, sim_prices, sim_ohlcv, report)
        regime, regime_probs, regime_duration = self._step_2_regime(ohlcv_data, report)
        report["regime"] = {
            "current": regime,
            "probabilities": regime_probs,
            "duration_days": regime_duration,
        }
        sentiment_signals = self._step_3_sentiment(report)
        player_signals = self._step_4_signals(prices, regime, sentiment_signals, report)
        # Skip debate — pre-market already debated
        logger.info("[Step 5] Skipping debate (post-market mode)")
        report["steps_completed"].append("step_5_debate_skipped")
        combined = self._step_6_combine(player_signals, regime, report)
        approved_signals, rejected = self._step_7_risk(combined, report)
        trades_today = self._step_8_execute(
            approved_signals, prices, today, regime, report,
            combined_signals=combined,
            player_signals=player_signals,
        )
        self._step_9_metrics(
            today, prices, regime, regime_probs, regime_duration,
            player_signals, trades_today, report,
        )
        self._step_10_reflection(trades_today, regime, report)
        self._step_11_coaching(regime, report)
        self._step_12_evolution(ohlcv_data, report)
        self._step_12b_exit_evolution(report)
        self._step_13_report(today, report)
        self._log_first_run_monitor(today, report, combined)

    # ══════════════════════════════════════════════════════════════════
    # Intelligence-mode steps
    # ══════════════════════════════════════════════════════════════════

    def _step_intel_heatmap(
        self,
        prices: Dict[str, float],
        sentiment: Dict[str, float],
        regime: str,
        report: Dict,
    ) -> None:
        """Generate sector/stock heatmap data from prices + sentiment."""
        logger.info("[Intel] Generating market heatmap...")

        heatmap: Dict[str, Any] = {}
        for sym in UNIVERSE:
            bare = sym.replace(".NS", "").replace(".BO", "")
            price = prices.get(sym, 0.0)
            sent = sentiment.get(bare, 0.0)

            # Classify sentiment into colour zones
            if sent > 0.3:
                zone = "strong_bullish"
            elif sent > 0.1:
                zone = "bullish"
            elif sent > -0.1:
                zone = "neutral"
            elif sent > -0.3:
                zone = "bearish"
            else:
                zone = "strong_bearish"

            heatmap[bare] = {
                "price": price,
                "sentiment": round(sent, 4),
                "zone": zone,
            }

        report["heatmap"] = heatmap
        report["steps_completed"].append("intel_heatmap")

        # Persist heatmap to file
        try:
            reports_dir = Path(self.config["daily_reports_dir"])
            reports_dir.mkdir(parents=True, exist_ok=True)
            heatmap_file = reports_dir / f"heatmap_{report['date']}.json"
            heatmap_file.write_text(
                json.dumps(heatmap, indent=2, default=str), encoding="utf-8"
            )
            logger.info(f"  Heatmap saved → {heatmap_file.name}")
        except OSError as e:
            report["warnings"].append(f"Failed to save heatmap: {e}")

    def _step_intel_morning_brief(
        self,
        today: str,
        prices: Dict[str, float],
        regime: str,
        regime_probs: Dict[str, float],
        regime_duration: int,
        sentiment: Dict[str, float],
        report: Dict,
    ) -> None:
        """Generate a morning brief summary for the trading day ahead."""
        logger.info("[Intel] Compiling morning brief...")

        # Portfolio snapshot (open positions from previous day)
        summary = self.tracker.portfolio_summary()

        # Top movers by sentiment
        scored = []
        for sym in UNIVERSE:
            bare = sym.replace(".NS", "").replace(".BO", "")
            scored.append((bare, sentiment.get(bare, 0.0)))
        scored.sort(key=lambda x: x[1], reverse=True)

        top_bullish = scored[:3]
        top_bearish = scored[-3:]

        brief = {
            "date": today,
            "regime": regime,
            "regime_confidence": round(max(regime_probs.values()), 4),
            "regime_duration_days": regime_duration,
            "portfolio_equity": summary["equity"],
            "open_positions": summary["open_positions"],
            "unrealised_pnl": summary["unrealised_pnl"],
            "top_bullish": [{"symbol": s, "sentiment": round(v, 4)} for s, v in top_bullish],
            "top_bearish": [{"symbol": s, "sentiment": round(v, 4)} for s, v in top_bearish],
            "stocks_covered": len(prices),
            "sentiment_available": len(sentiment),
        }

        report["morning_brief"] = brief
        report["steps_completed"].append("intel_morning_brief")

        # Persist
        try:
            reports_dir = Path(self.config["daily_reports_dir"])
            brief_file = reports_dir / f"morning_brief_{today}.json"
            brief_file.write_text(
                json.dumps(brief, indent=2, default=str), encoding="utf-8"
            )
            logger.info(f"  Morning brief saved → {brief_file.name}")
        except OSError as e:
            report["warnings"].append(f"Failed to save morning brief: {e}")

        # Log brief to console
        logger.info(f"\n  {'─'*50}")
        logger.info(f"  MORNING BRIEF — {today}")
        logger.info(f"  {'─'*50}")
        logger.info(f"  Regime:      {regime} (conf {brief['regime_confidence']:.0%}, "
                     f"day {regime_duration})")
        logger.info(f"  Equity:      ₹{summary['equity']:,.2f}")
        logger.info(f"  Open Pos:    {summary['open_positions']}")
        logger.info(f"  Unrealised:  ₹{summary['unrealised_pnl']:+,.2f}")
        logger.info(f"  Top Bull:    {', '.join(s for s, _ in top_bullish)}")
        logger.info(f"  Top Bear:    {', '.join(s for s, _ in top_bearish)}")
        logger.info(f"  {'─'*50}")

    # ══════════════════════════════════════════════════════════════════
    # Refresh-mode steps
    # ══════════════════════════════════════════════════════════════════

    def _step_refresh_breadth(
        self, prices: Dict[str, float], report: Dict,
    ) -> None:
        """Compute simple market breadth from universe prices."""
        logger.info("[Refresh] Computing market breadth...")

        if not prices:
            report["breadth"] = {"advancing": 0, "declining": 0, "unchanged": 0, "ratio": 0.0}
            report["steps_completed"].append("refresh_breadth (no data)")
            return

        # Compare current prices to previous session's mark-to-market
        advancing = 0
        declining = 0
        unchanged = 0

        for sym in UNIVERSE:
            price = prices.get(sym)
            if price is None:
                continue

            # Check against last known price in the position tracker
            pos_list = self.tracker.get_positions_for_symbol(sym)
            if pos_list:
                prev_price = pos_list[0].current_price or pos_list[0].entry_price
                if price > prev_price * 1.001:
                    advancing += 1
                elif price < prev_price * 0.999:
                    declining += 1
                else:
                    unchanged += 1
            else:
                # No position → just count as neutral
                unchanged += 1

        total = advancing + declining
        breadth_ratio = round(advancing / total, 4) if total > 0 else 0.5

        breadth = {
            "advancing": advancing,
            "declining": declining,
            "unchanged": unchanged,
            "ratio": breadth_ratio,
            "signal": "bullish" if breadth_ratio > 0.6 else (
                "bearish" if breadth_ratio < 0.4 else "neutral"
            ),
        }

        report["breadth"] = breadth
        report["steps_completed"].append("refresh_breadth")

        logger.info(f"  Breadth: {advancing}↑  {declining}↓  {unchanged}→  "
                     f"(ratio={breadth_ratio:.2f}, {breadth['signal']})")

    def _step_refresh_sentiment_delta(
        self,
        today: str,
        sentiment: Dict[str, float],
        report: Dict,
    ) -> None:
        """Compare current sentiment to morning's sentiment and flag shifts."""
        logger.info("[Refresh] Computing sentiment delta...")

        # Load morning brief if available
        morning_sentiment: Dict[str, float] = {}
        try:
            reports_dir = Path(self.config["daily_reports_dir"])
            brief_file = reports_dir / f"morning_brief_{today}.json"
            if brief_file.exists():
                brief = json.loads(brief_file.read_text(encoding="utf-8"))
                # Extract morning sentiments from top_bullish / top_bearish
                for item in brief.get("top_bullish", []):
                    morning_sentiment[item["symbol"]] = item["sentiment"]
                for item in brief.get("top_bearish", []):
                    morning_sentiment[item["symbol"]] = item["sentiment"]
        except (json.JSONDecodeError, OSError, KeyError):
            pass

        # Compute deltas
        deltas: List[Dict[str, Any]] = []
        for sym in UNIVERSE:
            bare = sym.replace(".NS", "").replace(".BO", "")
            current = sentiment.get(bare, 0.0)
            morning = morning_sentiment.get(bare, 0.0)
            delta = current - morning

            if abs(delta) > 0.15:  # significant shift threshold
                deltas.append({
                    "symbol": bare,
                    "morning": round(morning, 4),
                    "current": round(current, 4),
                    "delta": round(delta, 4),
                    "direction": "improved" if delta > 0 else "deteriorated",
                })

        report["sentiment_deltas"] = deltas
        report["steps_completed"].append("refresh_sentiment_delta")

        if deltas:
            logger.info(f"  {len(deltas)} significant sentiment shifts:")
            for d in deltas[:5]:
                arrow = "↑" if d["direction"] == "improved" else "↓"
                logger.info(f"    {d['symbol']}: {d['morning']:+.2f} → {d['current']:+.2f} "
                           f"({d['delta']:+.2f} {arrow})")
        else:
            logger.info("  No significant sentiment shifts since morning")

        # Persist refresh report
        try:
            reports_dir = Path(self.config["daily_reports_dir"])
            refresh_file = reports_dir / f"midday_refresh_{today}.json"
            refresh_data = {
                "date": today,
                "breadth": report.get("breadth", {}),
                "sentiment_deltas": deltas,
                "timestamp": datetime.now().isoformat(),
            }
            refresh_file.write_text(
                json.dumps(refresh_data, indent=2, default=str), encoding="utf-8"
            )
            logger.info(f"  Midday refresh saved → {refresh_file.name}")
        except OSError as e:
            report["warnings"].append(f"Failed to save midday refresh: {e}")

    # ══════════════════════════════════════════════════════════════════
    # Pipeline steps
    # ══════════════════════════════════════════════════════════════════

    def _step_1_pull_data(
        self,
        today: str,
        sim_prices: Optional[Dict[str, float]],
        sim_ohlcv: Optional[Any],
        report: Dict,
    ) -> Tuple[Dict[str, float], Optional[Any]]:
        """Step 1: Pull latest OHLCV data."""
        logger.info("[Step 1] Pulling market data...")

        if sim_prices is not None:
            report["steps_completed"].append("1_data_pull (simulated)")
            return sim_prices, sim_ohlcv

        if not _HAS_YF:
            report["warnings"].append("yfinance not available, using fallback prices")
            # Fallback: generate synthetic prices
            rng = np.random.default_rng(hash(today) % 2**32)
            prices = {}
            for sym in UNIVERSE:
                prices[sym] = round(float(rng.uniform(500, 5000)), 2)
            report["steps_completed"].append("1_data_pull (fallback)")
            return prices, None

        # Real data pull from yfinance
        try:
            prices = {}
            lookback = self.config["ohlcv_lookback_days"]
            end_dt = datetime.fromisoformat(today)
            start_dt = end_dt - timedelta(days=lookback + 10)

            data = yf.download(
                UNIVERSE,
                start=start_dt.strftime("%Y-%m-%d"),
                end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                group_by="ticker",
                auto_adjust=True,
                progress=False,
            )

            for sym in UNIVERSE:
                try:
                    close = data[sym]["Close"].dropna()
                    if len(close) > 0:
                        prices[sym] = float(close.iloc[-1])
                except (KeyError, IndexError):
                    logger.warning(f"No data for {sym}")

            report["steps_completed"].append("1_data_pull")
            return prices, data

        except Exception as e:
            report["errors"].append(f"Step 1 data pull failed: {e}")
            logger.error(f"Data pull failed: {e}")
            rng = np.random.default_rng(hash(today) % 2**32)
            prices = {sym: round(float(rng.uniform(500, 5000)), 2) for sym in UNIVERSE}
            return prices, None

    def _step_2_regime(
        self,
        ohlcv_data: Optional[Any],
        report: Dict,
    ) -> Tuple[str, Dict[str, float], int]:
        """Step 2: Detect market regime using HMM model."""
        logger.info("[Step 2] Detecting market regime...")

        if not _HAS_REGIME:
            report["warnings"].append("regime_detector not available, defaulting to Sideways")
            report["steps_completed"].append("2_regime (fallback)")
            return "Sideways", {"Bull": 0.33, "Bear": 0.33, "Sideways": 0.34}, 1

        try:
            # Load pre-trained HMM model via RegimeDetector (it auto-loads from disk)
            if self._regime_detector is None:
                model_path = Path(__file__).parent / "regime_hmm_model.joblib"
                self._regime_detector = RegimeDetector(model_path=model_path)

            # Get Nifty OHLCV for regime prediction
            nifty_df = load_nifty_data()
            if nifty_df is not None and len(nifty_df) > 30:
                regime, probs, duration = self._regime_detector.predict(nifty_df)
                logger.info(f"  Regime: {regime} (Bull={probs.get('Bull', 0):.0%}, "
                           f"Bear={probs.get('Bear', 0):.0%}, "
                           f"Sideways={probs.get('Sideways', 0):.0%}, "
                           f"duration={duration}d)")
                report["steps_completed"].append("2_regime")
                return regime, probs, duration
            else:
                report["warnings"].append("Nifty data too short for regime detection")
                report["steps_completed"].append("2_regime (insufficient data)")
                return "Sideways", {"Bull": 0.33, "Bear": 0.33, "Sideways": 0.34}, 1

        except Exception as e:
            report["warnings"].append(f"Regime detection failed: {e}")
            report["steps_completed"].append("2_regime (fallback)")
            logger.warning(f"  Regime detection error: {e}")
            return "Sideways", {"Bull": 0.33, "Bear": 0.33, "Sideways": 0.34}, 1

    def _step_3_sentiment(self, report: Dict) -> Dict[str, float]:
        """Step 3: News sentiment analysis."""
        logger.info("[Step 3] Analyzing news sentiment...")

        if not _HAS_SENTIMENT:
            report["warnings"].append("sentiment_engine not available")
            report["steps_completed"].append("3_sentiment (skipped)")
            return {}

        try:
            signals = get_sentiment_signals(UNIVERSE)
            report["steps_completed"].append("3_sentiment")
            report["sentiment_signals"] = signals
            return signals

        except Exception as e:
            report["warnings"].append(f"Sentiment analysis failed: {e}")
            report["steps_completed"].append("3_sentiment (failed)")
            return {}

    def _step_4_signals(
        self,
        prices: Dict[str, float],
        regime: str,
        sentiment: Dict[str, float],
        report: Dict,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Step 4: Generate per-player signals from REAL technical indicators.

        Uses the EVOLVED+ENSEMBLE approach:
        - Loads evolved player configs (indicator weights from genetic evolution)
        - Computes real indicator signals using IndicatorComputer
        - Applies regime bias and sentiment overlay ON TOP of real signals
        - Each player uses its own evolved weight set for independent signals

        Returns: {symbol: [{player_id, direction, confidence, reasoning}, ...]}
        """
        logger.info("[Step 4] Generating REAL indicator signals (EVOLVED+ENSEMBLE)...")

        signals: Dict[str, List[Dict[str, Any]]] = {}

        # Regime bias overlay (applied additively, clamped)
        regime_bias = {"Bull": 0.05, "Bear": -0.05, "Sideways": 0.0}.get(regime, 0.0)

        # Ensure indicator computer is ready
        if _HAS_INDICATORS and self._indicator_computer is None:
            self._indicator_computer = IndicatorComputer()

        # Reload evolved configs fresh each run (they may have been updated by evolution)
        self._evolved_configs = _load_evolved_configs()

        # Compute indicators for each symbol from cached OHLCV data
        for sym in UNIVERSE:
            if sym not in prices:
                continue

            bare = sym.replace(".NS", "").replace(".BO", "")
            sent_score = sentiment.get(bare, 0.0)

            # Try to compute real indicators from OHLCV data
            indicators = self._get_indicators_for_symbol(sym)

            sym_signals = []
            for pid in PLAYER_IDS:
                cfg = self._evolved_configs.get(pid, {})
                weights = cfg.get("weights", {})
                label = cfg.get("label", PLAYER_LABELS.get(pid, pid))

                if indicators is not None and len(indicators) > 60 and weights:
                    # REAL indicator signal computation
                    idx = len(indicators) - 1  # latest bar
                    direction, confidence = _compute_real_signal(indicators, weights, idx)

                    # Blend in regime bias + sentiment overlay
                    direction = float(np.clip(
                        direction + regime_bias + sent_score * 0.1,
                        -1.0, 1.0
                    ))

                    n_indicators = sum(1 for w in weights.values() if w != 0)
                    reasoning = (
                        f"{label}: {'Bullish' if direction > 0 else 'Bearish'} "
                        f"on {bare} | {n_indicators} indicators | "
                        f"regime={regime} sent={sent_score:+.2f} (REAL)"
                    )
                else:
                    # Fallback: simple regime+sentiment signal (no random!)
                    direction = float(np.clip(regime_bias + sent_score * 0.3, -1.0, 1.0))
                    confidence = 0.3
                    reasoning = (
                        f"{label}: {'Bullish' if direction > 0 else 'Bearish'} "
                        f"on {bare} | regime+sentiment only (FALLBACK)"
                    )

                sym_signals.append({
                    "player_id": pid,
                    "direction": round(direction, 4),
                    "confidence": round(confidence, 4),
                    "reasoning": reasoning,
                })

            signals[sym] = sym_signals

        # Count real vs fallback signals
        real_count = sum(
            1 for sigs in signals.values()
            for s in sigs if "(REAL)" in s.get("reasoning", "")
        )
        total_count = sum(len(v) for v in signals.values())

        report["steps_completed"].append("4_signals (EVOLVED+ENSEMBLE)")
        report["signal_count"] = total_count
        report["real_signal_count"] = real_count
        report["fallback_signal_count"] = total_count - real_count
        logger.info(f"  Generated {total_count} signals ({real_count} real, "
                    f"{total_count - real_count} fallback)")
        return signals

    def _get_indicators_for_symbol(self, sym: str) -> Optional[pd.DataFrame]:
        """Get computed indicators for a symbol, using OHLCV cache."""
        # Return cached indicators if available
        if sym in self._indicator_cache:
            return self._indicator_cache[sym]

        if not _HAS_INDICATORS or self._indicator_computer is None:
            return None

        # Try to get OHLCV data for this symbol
        ohlcv = self._ohlcv_cache.get(sym)
        if ohlcv is None:
            # Download fresh OHLCV if not cached
            if not _HAS_YF:
                return None
            try:
                lookback = self.config["ohlcv_lookback_days"]
                end_dt = datetime.now()
                start_dt = end_dt - timedelta(days=lookback + 30)
                df = yf.download(
                    sym,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
                    progress=False,
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) > 30:
                    ohlcv = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    self._ohlcv_cache[sym] = ohlcv
                else:
                    return None
            except Exception as e:
                logger.warning(f"  Failed to download OHLCV for {sym}: {e}")
                return None

        # Compute indicators
        try:
            indicators = self._indicator_computer.compute_all_indicators(ohlcv)
            self._indicator_cache[sym] = indicators
            return indicators
        except Exception as e:
            logger.warning(f"  Failed to compute indicators for {sym}: {e}")
            return None

    def _step_5_debate(
        self,
        player_signals: Dict[str, List[Dict[str, Any]]],
        report: Dict,
    ) -> Optional[Dict[str, Any]]:
        """Step 5: Player debate (LLM-powered, skipped if no Gemini)."""
        logger.info("[Step 5] Player debate...")

        if not _HAS_DEBATE:
            report["warnings"].append("player_debate not available")
            report["steps_completed"].append("5_debate (skipped)")
            return None

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            report["steps_completed"].append("5_debate (no API key)")
            return None

        try:
            if self._llm is None:
                self._llm = _GeminiLLM(model=self.config["gemini_model"])

            debate_engine = DebateRound(
                llm=self._llm,
                max_api_calls=self.config.get("max_api_calls_per_run", 50),
            )

            # player_signals is {symbol: [list of {player_id, direction, confidence, reasoning}]}
            # We need to find symbols where players DISAGREE and debate those.
            debates_run = 0
            debates_skipped = 0
            total_mind_changes = 0

            for sym, sigs in player_signals.items():
                if len(sigs) < 2:
                    continue

                # Build DebateSignal objects for each player on this symbol
                debate_signals: Dict[str, DebateSignal] = {}
                for sig in sigs:
                    pid = sig["player_id"]
                    debate_signals[pid] = DebateSignal(
                        player_id=pid,
                        player_label=PLAYER_LABELS.get(pid, pid),
                        symbol=sym,
                        direction=sig.get("direction", 0.0),
                        confidence=sig.get("confidence", 0.5),
                        reasoning=sig.get("reasoning", f"Signal strength: {sig.get('confidence', 0):.2f}"),
                    )

                outcome = debate_engine.run_debate(debate_signals, symbol=sym)

                if outcome.skipped:
                    debates_skipped += 1
                else:
                    debates_run += 1
                    if outcome.signals_changed > 0:
                        total_mind_changes += outcome.signals_changed
                        logger.info(f"  [Debate] {sym}: {outcome.signals_changed} signals changed")

                    # Feed revised signals back into player_signals
                    revised_sigs = []
                    for sig in sigs:
                        pid = sig["player_id"]
                        if pid in outcome.post_signals:
                            post = outcome.post_signals[pid]
                            revised_sigs.append({
                                **sig,
                                "direction": round(post.direction, 4),
                                "confidence": round(post.confidence, 4),
                            })
                        else:
                            revised_sigs.append(sig)
                    player_signals[sym] = revised_sigs

            logger.info(
                f"  Debates: {debates_run} ran, {debates_skipped} skipped, "
                f"{total_mind_changes} mind changes"
            )
            report["steps_completed"].append(
                f"5_debate ({debates_run} debates, {total_mind_changes} changes)"
            )
            return {
                "debates_run": debates_run,
                "debates_skipped": debates_skipped,
                "mind_changes": total_mind_changes,
            }

        except Exception as e:
            logger.warning(f"  Debate failed: {e}")
            report["warnings"].append(f"Debate failed: {e}")
            report["steps_completed"].append("5_debate (failed)")
            return None

    def _step_6_combine(
        self,
        player_signals: Dict[str, List[Dict[str, Any]]],
        regime: str,
        report: Dict,
    ) -> Dict[str, Dict[str, Any]]:
        """Step 6: Combine player signals with ENSEMBLE consensus sizing.

        Uses Bayesian voting + ensemble majority filter from EVOLVED+ENSEMBLE:
        - Bayesian confidence-weighted voting across 5 players
        - Ensemble consensus multiplier (0.5x lonely / 1.0x standard / 1.2x strong)

        Returns: {symbol: {"final_signal": float, "should_trade": bool,
                           "position_size_multiplier": float,
                           "consensus_multiplier": float, "best_player": str}}
        """
        logger.info("[Step 6] Combining signals (EVOLVED+ENSEMBLE)...")

        combined: Dict[str, Dict[str, Any]] = {}
        evo_config = self.config.get("evolution", {})

        if _HAS_COMBINER:
            try:
                if self._bayesian_weights is None:
                    state_dir = Path(self.config["signal_combiner_state_dir"])
                    state_dir.mkdir(parents=True, exist_ok=True)
                    self._bayesian_weights = BayesianPlayerWeights(state_dir=state_dir)
                    self._combiner = RegimeAwareCombiner(
                        bayesian_weights=self._bayesian_weights,
                        state_dir=state_dir,
                    )

                for sym, sigs in player_signals.items():
                    ps = [
                        PlayerSignal(
                            player_id=s["player_id"],
                            direction=s["direction"],
                            confidence=s["confidence"],
                            reasoning=s.get("reasoning", ""),
                        )
                        for s in sigs
                    ]

                    regime_weights = self._bayesian_weights.get_weights(regime)
                    result = self._combiner.combine(ps, regime, regime_weights)

                    # ENSEMBLE consensus multiplier
                    per_player_dirs = {s["player_id"]: s["direction"] for s in sigs}
                    best_pid = max(per_player_dirs, key=lambda p: abs(per_player_dirs[p]))
                    consensus_mult = _compute_ensemble_consensus_multiplier(
                        per_player_dirs, best_pid, evo_config,
                    )

                    combined[sym] = {
                        "final_signal": result.final_signal,
                        "should_trade": result.should_trade,
                        "position_size_multiplier": result.position_size_multiplier * consensus_mult,
                        "consensus_multiplier": consensus_mult,
                        "best_player": best_pid,
                    }

                report["steps_completed"].append("6_combine (EVOLVED+ENSEMBLE)")
                return combined

            except Exception as e:
                report["warnings"].append(f"Signal combiner failed: {e}")

        # Fallback: simple average + ensemble consensus
        for sym, sigs in player_signals.items():
            directions = [s["direction"] for s in sigs]
            avg_dir = float(np.mean(directions))
            avg_conf = float(np.mean([s["confidence"] for s in sigs]))
            should_trade = abs(avg_dir) > 0.25 and avg_conf > 0.3

            per_player_dirs = {s["player_id"]: s["direction"] for s in sigs}
            best_pid = max(per_player_dirs, key=lambda p: abs(per_player_dirs[p]))
            consensus_mult = _compute_ensemble_consensus_multiplier(
                per_player_dirs, best_pid, evo_config,
            )

            combined[sym] = {
                "final_signal": round(avg_dir, 4),
                "should_trade": should_trade,
                "position_size_multiplier": min(1.0, avg_conf) * consensus_mult,
                "consensus_multiplier": consensus_mult,
                "best_player": best_pid,
            }

        report["steps_completed"].append("6_combine (fallback+ensemble)")
        return combined

    def _step_7_risk(
        self,
        combined: Dict[str, Dict[str, Any]],
        report: Dict,
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        """Step 7: Risk management filter."""
        logger.info("[Step 7] Risk management filter...")

        # Filter: only trade signals with should_trade=True
        approved = {}
        rejected = []

        max_positions = self.config["max_positions"]
        current_count = self.tracker.open_position_count

        for sym, sig in combined.items():
            if not sig["should_trade"]:
                rejected.append({"symbol": sym, "reason": "should_trade=False"})
                continue

            if current_count + len(approved) >= max_positions:
                # Already have position in this symbol? Allow update
                if not self.tracker.get_positions_for_symbol(sym):
                    rejected.append({"symbol": sym, "reason": "max_positions reached"})
                    continue

            approved[sym] = sig

        report["steps_completed"].append("7_risk")
        report["approved_signals"] = len(approved)
        report["rejected_signals"] = len(rejected)
        return approved, rejected

    def _step_8_execute(
        self,
        approved: Dict[str, Dict[str, Any]],
        prices: Dict[str, float],
        today: str,
        regime: str,
        report: Dict,
        combined_signals: Optional[Dict[str, Dict[str, Any]]] = None,
        player_signals: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 8: Paper execution — SmartExitEngine + open/close positions.

        Flow:
        1. SmartExitEngine evaluates all open positions (replaces old check_stops)
        2. Gemini Exit Reasoning for ambiguous cases (0.30-0.70)
        3. Mark-to-market
        4. Execute new entry signals (with same-day re-entry guard)
        5. Log exit summary + Gemini stats + audit log
        """
        logger.info("[Step 8] Paper execution (SmartExitEngine)...")

        trades_today: List[Dict[str, Any]] = []
        exited_tickers_today: set = set()  # same-day re-entry guard

        # ── Exit engine stats ──────────────────────────────────────────
        exit_stats = {
            "full_exits": 0,
            "partial_exits": 0,
            "exit_reasons": {},
            "urgency_scores": [],
        }

        combined_signals = combined_signals or {}
        player_signals = player_signals or {}

        # Build portfolio context for Gemini
        _portfolio_ctx = None
        try:
            ps = self.tracker.portfolio_summary()
            _portfolio_ctx = {
                "equity": round(ps.get("equity", 0), 0),
                "cash": round(ps.get("cash", 0), 0),
                "n_positions": ps.get("open_positions", 0),
                "gross_exposure_pct": round(ps.get("gross_exposure_pct", 0) * 100, 1),
            }
        except Exception:
            pass

        # ── Pre-compute MFE/MAE prompt block (once per run) ─────────
        if self._mfe_analyzer and _HAS_MFE_MAE:
            try:
                if not getattr(self, '_mfe_prompt_block', None):
                    all_trades = self.journal.get_all_trades() if hasattr(self.journal, 'get_all_trades') else []
                    if len(all_trades) >= self._mfe_analyzer._min_trades:
                        mfe_rpt = self._mfe_analyzer.generate_tuning_report(all_trades)
                        self._mfe_prompt_block = self._mfe_analyzer.format_for_gemini_prompt(mfe_rpt)
                        logger.debug(f"[MFE/MAE] Prompt block ready ({len(all_trades)} trades)")
            except Exception as e:
                logger.debug(f"[MFE/MAE] Prompt block build failed: {e}")

        # ── PHASE 1: SmartExitEngine evaluation ────────────────────────
        if self._exit_engine and _HAS_SMART_EXIT:
            positions_to_evaluate = list(self.tracker.positions.items())
            for pid, pos in positions_to_evaluate:
                sym = pos.symbol
                price = prices.get(sym)
                if price is None:
                    continue

                # Get OHLCV data for this symbol (HIGH, LOW, Volume, ATR)
                ohlcv = self._ohlcv_cache.get(sym)
                if ohlcv is not None and len(ohlcv) > 1:
                    last_bar = ohlcv.iloc[-1]
                    current_low = float(last_bar.get("Low", price * 0.99))
                    current_high = float(last_bar.get("High", price * 1.01))
                    current_volume = float(last_bar.get("Volume", 0))
                else:
                    current_low = price * 0.99
                    current_high = price * 1.01
                    current_volume = 0

                # ATR: try to get from indicators, fallback to 1.5% of price
                indicators = self._indicator_cache.get(sym)
                if indicators is not None and len(indicators) > 0:
                    atr_val = indicators.iloc[-1].get("ATR_14", price * 0.015)
                    atr = float(atr_val) if atr_val and not pd.isna(atr_val) else price * 0.015
                else:
                    atr = price * 0.015

                # Current signal and agreement from combined signals
                sym_sig = combined_signals.get(sym, {})
                current_signal = sym_sig.get("final_signal", 0.0)
                current_agreement = sym_sig.get("consensus_multiplier", 0.5)

                # Vol forecast: use ATR as proxy (annualised)
                current_vol = (atr / price) * math.sqrt(252) if price > 0 else 0.2

                # Build PositionState from Position
                pos_state = PositionState(
                    symbol=sym,
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    entry_date=pos.entry_date,
                    entry_signal=pos.entry_signal,
                    entry_agreement=pos.entry_agreement,
                    entry_volatility=pos.entry_volatility,
                    entry_regime=pos.entry_regime,
                    remaining_fraction=pos.remaining_fraction,
                    staged_exits_done=pos.staged_exits_done,
                    max_favorable_excursion=pos.max_favorable_excursion,
                    max_adverse_excursion=pos.max_adverse_excursion,
                    peak_price=pos.high_water_mark,
                    signal_history=list(pos.signal_history),
                    agreement_history=list(pos.agreement_history),
                    volume_history=list(pos.volume_history),
                    bars_since_entry=pos.bars_held,
                )

                # Parse eval_date for event calendar
                try:
                    from datetime import date as _date_type
                    _eval_date = _date_type.fromisoformat(today) if isinstance(today, str) else today
                except (ValueError, TypeError):
                    _eval_date = None

                decision = self._exit_engine.evaluate(
                    position=pos_state,
                    current_price=price,
                    current_low=current_low,
                    current_volume=current_volume,
                    atr=atr,
                    current_signal=current_signal,
                    current_agreement=current_agreement,
                    current_vol_forecast=current_vol,
                    regime=regime,
                    current_high=current_high,
                    eval_date=_eval_date,
                )

                # ── Gemini Exit Reasoning (ambiguous zone only) ────
                if self._gemini_exit and _HAS_GEMINI_EXIT:
                    try:
                        # Build market_data dict for this symbol
                        _mkt = {
                            "atr": atr,
                            "current_price": price,
                            "current_volume": current_volume,
                            "avg_volume_20d": 0,
                            "current_low": current_low,
                            "current_high": current_high,
                            "garch_vol": current_vol,
                            "entry_vol": pos.entry_volatility,
                            "current_signal": current_signal,
                            "current_agreement": current_agreement,
                        }
                        # Volume 20d average
                        if ohlcv is not None and len(ohlcv) >= 20:
                            _mkt["avg_volume_20d"] = float(ohlcv["Volume"].tail(20).mean())

                        # Per-player signals for this symbol
                        _psigs = []
                        sym_player_sigs = player_signals.get(sym, [])
                        for ps in sym_player_sigs:
                            _psigs.append({
                                "player_id": ps.get("player_id", ""),
                                "direction": ps.get("direction", 0.0),
                                "confidence": ps.get("confidence", 0.0),
                                "player_label": ps.get("player_label", ps.get("player_id", "")),
                            })

                        # Build event context for Gemini prompt
                        _event_ctx = None
                        if self._event_calendar and _eval_date:
                            try:
                                _event_ctx = self._event_calendar.get_event_context_for_prompt(_eval_date)
                            except Exception:
                                pass

                        # MFE/MAE historical context (if available)
                        _mfe_ctx = getattr(self, '_mfe_prompt_block', None)

                        gemini_result = self._gemini_exit.evaluate_with_reasoning(
                            position=pos_state,
                            market_data=_mkt,
                            player_signals=_psigs,
                            regime=regime,
                            rule_decision=decision,
                            portfolio_context=_portfolio_ctx,
                            event_context=_event_ctx,
                            mfe_mae_context=_mfe_ctx,
                        )

                        if gemini_result.get("source") == "gemini":
                            decision = ExitDecision(
                                should_exit=gemini_result["should_exit"],
                                exit_fraction=gemini_result["exit_fraction"],
                                reason=decision.reason,
                                urgency=gemini_result.get("urgency", decision.urgency),
                                details={
                                    **decision.details,
                                    "gemini_override": gemini_result.get("override_rule_based", False),
                                    "gemini_decision": gemini_result.get("reason", ""),
                                    "gemini_reasoning": gemini_result.get("reasoning", ""),
                                    "gemini_confidence": gemini_result.get("confidence", 0.0),
                                    "gemini_ai_consulted": True,
                                },
                            )
                    except Exception as e:
                        logger.warning(f"[GeminiExit] Failed for {sym}: {e}")

                # Sync PositionState back to Position
                pos.high_water_mark = pos_state.peak_price
                pos.max_favorable_excursion = pos_state.max_favorable_excursion
                pos.max_adverse_excursion = pos_state.max_adverse_excursion
                pos.remaining_fraction = pos_state.remaining_fraction
                pos.staged_exits_done = pos_state.staged_exits_done
                pos.signal_history = pos_state.signal_history
                pos.agreement_history = pos_state.agreement_history
                pos.volume_history = pos_state.volume_history

                if decision.should_exit:
                    reason_str = decision.reason.value
                    exit_stats["urgency_scores"].append(decision.urgency)
                    exit_stats["exit_reasons"][reason_str] = exit_stats["exit_reasons"].get(reason_str, 0) + 1

                    if decision.exit_fraction < 1.0:
                        # Partial exit
                        shares_to_exit = max(1, int(pos.quantity * decision.exit_fraction))
                        trade = self.tracker.partial_close_position(
                            pid, shares_to_exit, price,
                            exit_date=today, reason=reason_str,
                        )
                        if trade:
                            trade["regime"] = regime
                            trade["exit_urgency"] = round(decision.urgency, 2)
                            self.journal.record_trade(trade)
                            trades_today.append(trade)
                            exit_stats["partial_exits"] += 1
                            exited_tickers_today.add(sym)
                    else:
                        # Full exit
                        trade = self.tracker.close_position(
                            pid, price, exit_date=today, reason=reason_str,
                        )
                        if trade:
                            trade["regime"] = regime
                            trade["exit_urgency"] = round(decision.urgency, 2)
                            self.journal.record_trade(trade)
                            trades_today.append(trade)
                            exit_stats["full_exits"] += 1
                            exited_tickers_today.add(sym)
        else:
            # Fallback: old trailing stop logic
            stopped = self.tracker.check_stops(prices, exit_date=today)
            for t in stopped:
                t["regime"] = regime
                self.journal.record_trade(t)
                trades_today.append(t)
                exited_tickers_today.add(t["symbol"])
                exit_stats["full_exits"] += 1

        # ── PHASE 2: Mark-to-market ────────────────────────────────────
        self.tracker.mark_to_market(prices)

        # ── PHASE 3: Execute new entry signals ─────────────────────────
        for sym, sig in approved.items():
            price = prices.get(sym)
            if price is None:
                continue

            # Same-day re-entry guard: skip tickers exited today
            if sym in exited_tickers_today:
                logger.debug(f"[Re-entry guard] Skipping {sym} — exited today")
                continue

            direction = "LONG" if sig["final_signal"] > 0 else "SHORT"
            signal_strength = abs(sig["final_signal"])

            # Determine which player takes this trade
            player_id = self._pick_best_player(sig, sym)

            # Position sizing: player uses its own ₹1L cash account
            player_cash = self.tracker.player_cash_available(player_id)

            if player_cash < 500:
                logger.debug(f"[{player_id}] no cash left (₹{player_cash:,.0f}), skipping {sym}")
                continue

            capital_for_trade = player_cash * signal_strength * sig["position_size_multiplier"]
            # Cap to player's available cash
            capital_for_trade = min(capital_for_trade, player_cash)

            if capital_for_trade < 500:
                continue

            quantity = max(1, int(capital_for_trade / price))

            # Check if we already have a position — if opposite direction, close first
            existing = self.tracker.get_positions_for_symbol(sym)
            for pos in existing:
                if pos.direction != direction:
                    trade = self.tracker.close_position(
                        pos.position_id, price, exit_date=today,
                        reason="signal_reversal",
                    )
                    if trade:
                        trade["regime"] = regime
                        self.journal.record_trade(trade)
                        trades_today.append(trade)

            # Open new position (skip if already in same direction)
            existing_same = [p for p in self.tracker.get_positions_for_symbol(sym)
                            if p.direction == direction]
            if not existing_same:
                # Populate SmartExitEngine fields at entry
                entry_signal = signal_strength
                entry_agreement = sig.get("consensus_multiplier", 0.5)
                # Vol forecast from ATR or fallback
                entry_vol = 0.2  # default
                indicators = self._indicator_cache.get(sym)
                if indicators is not None and len(indicators) > 0:
                    atr_val = indicators.iloc[-1].get("ATR_14", price * 0.015)
                    if atr_val and not pd.isna(atr_val) and price > 0:
                        entry_vol = float(atr_val / price) * math.sqrt(252)

                self.tracker.open_position(
                    symbol=sym,
                    player_id=player_id,
                    direction=direction,
                    quantity=quantity,
                    entry_price=price,
                    entry_date=today,
                    entry_signal=entry_signal,
                    entry_agreement=entry_agreement,
                    entry_volatility=entry_vol,
                    entry_regime=regime.lower(),
                )

        # ── PHASE 4: Log exit engine summary ───────────────────────────
        total_exits = exit_stats["full_exits"] + exit_stats["partial_exits"]
        if total_exits > 0:
            avg_urgency = (
                sum(exit_stats["urgency_scores"]) / len(exit_stats["urgency_scores"])
                if exit_stats["urgency_scores"] else 0.0
            )
            reason_breakdown = ", ".join(
                f"{reason}={count}" for reason, count in exit_stats["exit_reasons"].items()
            )
            logger.info(
                f"[SmartExit Summary] {total_exits} exits "
                f"(full={exit_stats['full_exits']}, partial={exit_stats['partial_exits']}) | "
                f"avg_urgency={avg_urgency:.2f} | reasons: {reason_breakdown}"
            )
        else:
            logger.info("[SmartExit Summary] No exits triggered")

        # ── PHASE 4b: Event calendar log ──────────────────────────────
        if self._event_calendar and _HAS_EVENT_CALENDAR:
            try:
                from datetime import date as _date_type
                _today_date = _date_type.fromisoformat(today) if isinstance(today, str) else today
                upcoming = self._event_calendar.get_upcoming_events(_today_date, look_ahead_days=5)
                tightening = self._event_calendar.get_exit_tightening_factor(_today_date)
                if upcoming:
                    event_names = [e["name"] for e in upcoming[:5]]
                    logger.info(
                        f"[EventCalendar] tightening={tightening:.2f} | "
                        f"upcoming: {', '.join(event_names)}"
                    )
                    exit_stats["event_calendar"] = {
                        "tightening_factor": round(tightening, 3),
                        "upcoming_events": upcoming[:5],
                    }
            except Exception as e:
                logger.debug(f"[EventCalendar] Logging failed: {e}")

        # ── PHASE 5: Gemini Exit stats + audit log ─────────────────────
        if self._gemini_exit and _HAS_GEMINI_EXIT:
            try:
                gemini_stats = self._gemini_exit.get_daily_stats()
                exit_stats["gemini_exit"] = gemini_stats
                if gemini_stats.get("ai_calls", 0) > 0:
                    logger.info(
                        f"[GeminiExit Stats] "
                        f"evals={gemini_stats['total_evaluations']} "
                        f"ai_calls={gemini_stats['ai_calls']} "
                        f"rule_only={gemini_stats['rule_only']} "
                        f"overrides={gemini_stats['ai_overrides']} "
                        f"cache_hits={gemini_stats['cache_hits']} "
                        f"cost=${gemini_stats.get('estimated_cost_usd', 0):.4f}"
                    )
                self._gemini_exit.save_decision_log()
            except Exception as e:
                logger.warning(f"[GeminiExit] Stats/log failed: {e}")

        # ── PHASE 6: MFE/MAE post-trade analysis ──────────────────────
        if self._mfe_analyzer and _HAS_MFE_MAE:
            try:
                all_trades = self.journal.get_all_trades() if hasattr(self.journal, 'get_all_trades') else []
                if len(all_trades) >= self._mfe_analyzer._min_trades:
                    mfe_report = self._mfe_analyzer.generate_tuning_report(all_trades)
                    if mfe_report.get("status") == "ok":
                        exit_stats["mfe_mae_analysis"] = {
                            "n_trades": mfe_report.get("n_trades", 0),
                            "avg_capture_ratio": round(mfe_report.get("avg_capture_ratio", 0.0), 4),
                            "avg_edge_ratio": round(mfe_report.get("avg_edge_ratio", 0.0), 4),
                            "avg_mfe_timing": round(mfe_report.get("avg_mfe_timing", 0.0), 4),
                            "avg_giveback_pct": round(mfe_report.get("avg_giveback_pct", 0.0), 4),
                            "pct_poor_exits": round(mfe_report.get("pct_poor_exits", 0.0), 4),
                            "avg_mfe_atr": round(mfe_report.get("avg_mfe_atr", 0.0), 3),
                            "avg_mae_atr": round(mfe_report.get("avg_mae_atr", 0.0), 3),
                        }
                        # Parameter suggestions for SmartExitEngine tuning
                        suggestions = self._mfe_analyzer.generate_parameter_suggestions(mfe_report)
                        if suggestions:
                            exit_stats["mfe_mae_analysis"]["parameter_suggestions"] = suggestions
                        # Log summary
                        logger.info(
                            f"[MFE/MAE] n={mfe_report['n_trades']} | "
                            f"capture={mfe_report['avg_capture_ratio']:.3f} | "
                            f"edge={mfe_report['avg_edge_ratio']:.3f} | "
                            f"timing={mfe_report['avg_mfe_timing']:.3f} | "
                            f"poor_exits={mfe_report['pct_poor_exits']:.1%}"
                        )
                        if mfe_report.get("recommendations_text"):
                            logger.info(f"[MFE/MAE] Recommendations:\n{mfe_report['recommendations_text']}")
                        # Update Gemini prompt block
                        self._mfe_prompt_block = self._mfe_analyzer.format_for_gemini_prompt(mfe_report)
                else:
                    logger.debug(
                        f"[MFE/MAE] Skipped — {len(all_trades)} trades "
                        f"< min {self._mfe_analyzer._min_trades}"
                    )
            except Exception as e:
                logger.debug(f"[MFE/MAE] Analysis failed: {e}")

        report["steps_completed"].append("8_execute")
        report["trades_today"] = len(trades_today)
        report["stops_hit"] = total_exits
        report["smart_exit_stats"] = exit_stats
        report["open_positions"] = self.tracker.open_position_count
        return trades_today

    def _step_9_metrics(
        self,
        today: str,
        prices: Dict[str, float],
        regime: str,
        regime_probs: Dict[str, float],
        regime_duration: int,
        player_signals: Dict[str, List[Dict[str, Any]]],
        trades_today: List[Dict[str, Any]],
        report: Dict,
    ) -> None:
        """Step 9: Record daily metrics snapshot."""
        logger.info("[Step 9] Recording metrics...")

        summary = self.tracker.portfolio_summary()
        per_player_perf = self.journal.per_player_summary()

        # Build per-player section
        players_data = {}
        for pid in PLAYER_IDS:
            perf = per_player_perf.get(pid, {})
            players_data[pid] = {
                "sharpe": perf.get("sharpe", 0.0),
                "sortino": 0.0,
                "win_rate": perf.get("win_rate", 0.0) / 100.0,  # performance_summary returns 0-100 pct
                "num_trades": perf.get("total_trades", 0),
                "pnl": perf.get("total_pnl", 0.0),
                "bayesian_weight": 0.20,
            }

        # Build trades today section
        trades_section = []
        for t in trades_today:
            trades_section.append({
                "symbol": t.get("symbol", ""),
                "player_id": t.get("player_id", ""),
                "direction": t.get("direction", ""),
                "pnl": t.get("pnl", 0.0),
                "close_reason": t.get("close_reason", ""),
            })

        # Build positions section
        positions_section = []
        for pos in self.tracker.positions.values():
            positions_section.append({
                "symbol": pos.symbol,
                "player_id": pos.player_id,
                "direction": pos.direction,
                "quantity": pos.quantity,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "unrealised_pnl": pos.unrealised_pnl,
                "bars_held": pos.bars_held,
            })

        snapshot = {
            "date": today,
            "portfolio": {
                "equity_curve_value": summary["equity"],
                "daily_return": summary["daily_return"],
                "gross_pnl": summary["unrealised_pnl"],
                "net_pnl": summary["unrealised_pnl"] * 0.98,  # rough cost estimate
                "gross_exposure": summary.get("gross_exposure_pct", 0.0),
                "net_exposure": summary.get("net_exposure_pct", 0.0),
                "largest_concentration": 0.0,
                "sharpe": 0.0,
            },
            "positions": positions_section,
            "trades_today": trades_section,
            "regime": {
                "current": regime,
                "probabilities": regime_probs,
                "duration_days": regime_duration,
            },
            "players": players_data,
            "signals": {
                "total_generated": report.get("signal_count", 0),
                "approved": report.get("approved_signals", 0),
                "rejected": report.get("rejected_signals", 0),
            },
            "risk": {
                "current_drawdown": max(0, 1 - summary["equity"] / self.tracker.initial_capital),
                "open_positions": self.tracker.open_position_count,
                "max_concentration": 0.0,
            },
            "coaching": {},
            "evolution": {},
        }

        self.store.record_daily_snapshot(snapshot)
        report["steps_completed"].append("9_metrics")
        report["portfolio_equity"] = summary["equity"]

    def _step_10_reflection(
        self,
        trades_today: List[Dict[str, Any]],
        regime: str,
        report: Dict,
    ) -> None:
        """Step 10: Player reflection (every N trades)."""
        logger.info("[Step 10] Checking reflection triggers...")

        total_trades = self.journal.trade_count
        trigger_every = self.config["reflection_every_n_trades"]
        should_reflect = (
            total_trades > 0 and total_trades % trigger_every == 0
        )

        if not should_reflect:
            report["steps_completed"].append("10_reflection (not triggered)")
            return

        if not _HAS_REFLECTION:
            report["warnings"].append("player_reflection not available")
            report["steps_completed"].append("10_reflection (unavailable)")
            return

        try:
            logger.info(f"  Reflection triggered at trade #{total_trades}")

            # Initialise LLM if needed
            if self._llm is None:
                self._llm = _GeminiLLM(model=self.config["gemini_model"])

            # Build reflection system
            if self._reflection_scheduler is None:
                reflection_system = create_reflection_system(
                    llm=self._llm,
                    trigger_every=trigger_every,
                )
                self._reflection_scheduler = reflection_system["scheduler"]

            # Run reflection for each player that has trades
            all_trades = self.journal.get_all_trades()
            reflections_count = 0
            adjustments_applied = 0

            for pid in PLAYER_IDS:
                label = PLAYER_LABELS.get(pid, pid)
                player_trades = [t for t in all_trades if t.get("player_id") == pid]
                if not player_trades:
                    continue

                result = self._reflection_scheduler.check_and_reflect(
                    player_id=pid,
                    player_label=label,
                    all_trades=player_trades,
                    regime=regime,
                )
                if result and result.rules:
                    reflections_count += 1
                    logger.info(f"  [{pid}] Reflected: {len(result.rules)} new rules, "
                               f"calibration={result.confidence_calibration}")

                    # Apply indicator adjustments from reflection to evolved configs
                    if result.indicator_adjustments and pid in self._evolved_configs:
                        cfg = self._evolved_configs[pid]
                        weights = cfg.get("weights", {})
                        for ind, adj in result.indicator_adjustments.items():
                            if ind in weights:
                                old_w = weights[ind]
                                weights[ind] = max(0.05, min(1.0, old_w + adj))
                                adjustments_applied += 1
                        cfg["weights"] = weights
                        self._evolved_configs[pid] = cfg

            # Save updated configs if any adjustments were applied
            if adjustments_applied > 0:
                cfg_path = Path(self.config.get(
                    "evolved_configs_file",
                    str(Path(__file__).parent / "evolved_player_configs.json")
                ))
                save_data = {
                    "saved_at": datetime.now().isoformat(),
                    "configs": self._evolved_configs,
                    "total_runs": self._state.get("total_runs", 0),
                    "last_update": datetime.now().isoformat(),
                    "design_notes": f"Reflection adjustments ({adjustments_applied} weights), regime={regime}",
                }
                cfg_path.write_text(
                    json.dumps(save_data, indent=2, default=str),
                    encoding="utf-8",
                )
                logger.info(f"  Applied {adjustments_applied} indicator weight adjustments")

            report["steps_completed"].append(
                f"10_reflection ({reflections_count} reflected, {adjustments_applied} adjustments)"
            )
            report["reflection_triggered"] = True
        except Exception as e:
            report["warnings"].append(f"Reflection failed: {e}")
            report["steps_completed"].append("10_reflection (failed)")

    def _step_11_coaching(self, regime: str, report: Dict) -> None:
        """Step 11: AI Coaching cycle — Gemini analyzes performance, queries knowledge base,
        and updates player indicator weights/thresholds.

        This is the core iterative learning loop:
        1. AICoach calls Gemini with trade history + knowledge base context
        2. Gemini decides which indicators to add/remove, what weights to assign
        3. Updated configs are saved to evolved_player_configs.json
        4. Next run uses the improved configs
        """
        logger.info("[Step 11] Checking coaching triggers...")

        total_trades = self.journal.trade_count
        min_trades = self.config["coaching_min_trades"]

        if total_trades < min_trades:
            report["steps_completed"].append(
                f"11_coaching (need {min_trades - total_trades} more trades)"
            )
            return

        if not _HAS_AI_COACH:
            report["warnings"].append("AICoach not available")
            report["steps_completed"].append("11_coaching (unavailable)")
            return

        try:
            logger.info(f"  Coaching triggered at trade #{total_trades}")

            # Initialise LLM wrapper for AICoach
            if self._llm is None:
                self._llm = _GeminiLLM(model=self.config["gemini_model"])

            # Create AICoach with knowledge layer + LLM
            ai_coach = AICoach(
                use_llm=True,
                llm_provider=self._llm,
                use_knowledge=True,  # queries ChromaDB knowledge base
            )

            # Build players_data from journal + evolved configs
            players_data = []
            for pid in PLAYER_IDS:
                cfg = self._evolved_configs.get(pid, {})
                player_trades = self.journal.get_player_trades(pid)
                players_data.append({
                    "player_id": pid,
                    "player_label": PLAYER_LABELS.get(pid, pid),
                    "trades": player_trades,
                    "weights": cfg.get("weights", {}),
                    "config": cfg,
                })

            # Run coaching — Gemini analyzes trades, queries knowledge base,
            # recommends new indicators/weights/thresholds
            batch_results = ai_coach.analyze_players_batch(
                players_data=players_data,
                market_regime=regime,
            )

            # Apply Gemini's recommendations to each player's config
            configs_changed = 0
            for pid in PLAYER_IDS:
                if pid not in batch_results:
                    continue

                analysis = batch_results[pid]
                old_cfg = self._evolved_configs.get(pid, {})
                new_cfg = ai_coach.apply_recommendations(old_cfg, analysis)

                # Check if config actually changed
                old_weights = old_cfg.get("weights", {})
                new_weights = new_cfg.get("weights", {})
                if new_weights != old_weights:
                    configs_changed += 1
                    added = set(new_weights.keys()) - set(old_weights.keys())
                    removed = set(old_weights.keys()) - set(new_weights.keys())
                    if added:
                        logger.info(f"  [{pid}] Added indicators: {', '.join(added)}")
                    if removed:
                        logger.info(f"  [{pid}] Removed indicators: {', '.join(removed)}")
                    logger.info(f"  [{pid}] entry={new_cfg.get('entry_threshold', '?'):.2f} "
                               f"exit={new_cfg.get('exit_threshold', '?'):.2f} "
                               f"hold={new_cfg.get('min_hold_bars', '?')}")

                self._evolved_configs[pid] = new_cfg

            # Save updated configs to disk so next run uses them
            if configs_changed > 0:
                cfg_path = Path(self.config.get(
                    "evolved_configs_file",
                    str(Path(__file__).parent / "evolved_player_configs.json")
                ))
                save_data = {
                    "saved_at": datetime.now().isoformat(),
                    "configs": self._evolved_configs,
                    "total_runs": self._state.get("total_runs", 0),
                    "last_update": datetime.now().isoformat(),
                    "design_notes": f"AI Coach update at trade #{total_trades}, regime={regime}",
                }
                cfg_path.write_text(
                    json.dumps(save_data, indent=2, default=str),
                    encoding="utf-8",
                )
                logger.info(f"  Coaching complete: {configs_changed}/{len(PLAYER_IDS)} players updated")

            report["steps_completed"].append(
                f"11_coaching ({configs_changed} configs updated)"
            )
            report["coaching_configs_changed"] = configs_changed

        except Exception as e:
            report["warnings"].append(f"Coaching failed: {e}")
            report["steps_completed"].append("11_coaching (failed)")
            logger.warning(f"  Coaching error: {e}")
            import traceback
            logger.warning(traceback.format_exc())

    def _step_12_evolution(self, ohlcv_data: Optional[Any], report: Dict) -> None:
        """Step 12: Adaptive player evolution with dual triggers.

        Trigger 1 — REGIME SHIFT: When HMM detects a regime change (e.g. Bull→Bear),
                    immediately evolve (with 10-day cooldown to prevent thrashing).
        Trigger 2 — CALENDAR: If no regime shift, evolve every 25 trading days as baseline.

        Evolution uses paper trading P&L as fitness (NOT backtest performance).
        Requires minimum 15 trading days of data before evolving.
        """
        logger.info("[Step 12] Checking evolution triggers (adaptive)...")

        evo_cfg = self.config.get("evolution", {})
        calendar_days = evo_cfg.get("calendar_evolution_days", 25)
        min_cooldown = evo_cfg.get("min_days_between_evolution", 10)
        min_data_days = evo_cfg.get("min_trading_days_for_evolution", 15)

        days_since = self._state.get("days_since_evolution", 0) + 1
        self._state["days_since_evolution"] = days_since
        current_regime = report.get("regime", {}).get("current", "Sideways") if isinstance(
            report.get("regime"), dict
        ) else "Sideways"

        # Detect regime from step 2 output (may be stored differently)
        if "regime" not in report or not isinstance(report.get("regime"), dict):
            # Try to get from the step_2 output that was stored in the report
            current_regime = self._state.get("last_regime", "Sideways")

        last_regime = self._state.get("last_regime", current_regime)
        self._state["last_regime"] = current_regime

        # Check if enough data for evolution
        total_runs = self._state.get("total_runs", 0)
        if total_runs < min_data_days:
            report["steps_completed"].append(
                f"12_evolution (need {min_data_days - total_runs} more days of data)"
            )
            self._save_state()
            return

        # Determine trigger type
        trigger_type = None
        trigger_reason = ""

        # Trigger 1: Regime shift (with cooldown)
        if current_regime != last_regime and days_since >= min_cooldown:
            trigger_type = "regime_shift"
            trigger_reason = f"Regime shift: {last_regime} → {current_regime}"
            logger.info(f"  REGIME SHIFT detected: {last_regime} → {current_regime}")

        # Trigger 2: Calendar baseline
        elif days_since >= calendar_days:
            trigger_type = "calendar"
            trigger_reason = f"Calendar trigger: {days_since} days since last evolution"

        if trigger_type is None:
            next_cal = calendar_days - days_since
            report["steps_completed"].append(
                f"12_evolution (day {days_since}/{calendar_days}, regime={current_regime})"
            )
            self._save_state()
            return

        # Evolution triggered!
        if not _HAS_EVOLUTION:
            report["warnings"].append("player_evolution not available — evolution skipped")
            report["steps_completed"].append("12_evolution (unavailable)")
            self._save_state()
            return

        try:
            logger.info(f"  EVOLUTION TRIGGERED ({trigger_type}): {trigger_reason}")

            # Get paper trading performance for fitness evaluation
            journal_perf = self.journal.per_player_summary()
            performance_history = {}
            for pid in PLAYER_IDS:
                perf = journal_perf.get(pid, {})
                performance_history[pid] = {
                    "sharpe": perf.get("sharpe", 0.0),
                    "pnl": perf.get("total_pnl", 0.0),
                    "win_rate": perf.get("win_rate", 0.0),
                    "trades": perf.get("total_trades", 0),
                }

            # Initialize evolution engine if needed
            if self._evolution_engine is None:
                self._evolution_engine = PlayerEvolutionEngine(
                    evolution_interval=calendar_days,
                    seed=42,
                )

            # Run evolution on OHLCV data
            if self._ohlcv_cache:
                # Use cached OHLCV data for evolution
                new_configs = self._evolution_engine.evolve(
                    current_configs=self._evolved_configs,
                    performance_history=performance_history,
                    train_data=self._ohlcv_cache,
                )

                if new_configs:
                    # Save evolved configs
                    self._evolved_configs = new_configs
                    cfg_path = Path(self.config.get(
                        "evolved_configs_file",
                        str(Path(__file__).parent / "evolved_player_configs.json")
                    ))
                    save_data = {
                        "saved_at": datetime.now().isoformat(),
                        "configs": new_configs,
                        "player_best_pnl": {
                            pid: performance_history.get(pid, {}).get("pnl", 0)
                            for pid in PLAYER_IDS
                        },
                        "total_runs": total_runs,
                        "last_update": datetime.now().isoformat(),
                        "design_notes": f"Evolved via {trigger_type} trigger",
                    }
                    cfg_path.write_text(
                        json.dumps(save_data, indent=2, default=str),
                        encoding="utf-8",
                    )
                    logger.info(f"  Evolved configs saved to {cfg_path.name}")

            # Reset evolution counter and record trigger info
            self._state["days_since_evolution"] = 0
            self._state["last_evolution_date"] = datetime.now().isoformat()
            self._state["evolution_trigger_type"] = trigger_type
            self._save_state()

            report["steps_completed"].append(f"12_evolution ({trigger_type})")
            report["evolution_triggered"] = True
            report["evolution_trigger_type"] = trigger_type
            report["evolution_reason"] = trigger_reason

        except Exception as e:
            report["warnings"].append(f"Evolution failed: {e}")
            report["steps_completed"].append("12_evolution (failed)")
            logger.error(f"  Evolution error: {e}")
            self._save_state()

    def _step_13_report(self, today: str, report: Dict) -> None:
        """Step 13: Generate and save daily report."""
        logger.info("[Step 13] Generating daily report...")

        summary = self.tracker.portfolio_summary()
        journal_summary = self.journal.performance_summary()

        report["summary"] = {
            "equity": summary["equity"],
            "cash": summary["cash"],
            "unrealised_pnl": summary["unrealised_pnl"],
            "daily_return": summary["daily_return"],
            "open_positions": summary["open_positions"],
            "total_trades": journal_summary["total_trades"],
            "total_pnl": journal_summary["total_pnl"],
            "win_rate": journal_summary["win_rate"],
        }

        # Save report to file
        try:
            reports_dir = Path(self.config["daily_reports_dir"])
            reports_dir.mkdir(parents=True, exist_ok=True)
            report_file = reports_dir / f"report_{today}.json"
            report_file.write_text(
                json.dumps(report, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            report["warnings"].append(f"Failed to save report: {e}")

        report["steps_completed"].append("13_report")

        # Print summary
        logger.info(f"\n  {'─'*50}")
        logger.info(f"  DAILY SUMMARY — {today}")
        logger.info(f"  {'─'*50}")
        logger.info(f"  Equity:       ₹{summary['equity']:,.2f}")
        logger.info(f"  Daily Return: {summary['daily_return']:+.4%}")
        logger.info(f"  Unrealised:   ₹{summary['unrealised_pnl']:+,.2f}")
        logger.info(f"  Open Pos:     {summary['open_positions']}")
        logger.info(f"  Trades Today: {report.get('trades_today', 0)}")
        logger.info(f"  Total Trades: {journal_summary['total_trades']}")
        logger.info(f"  Total P&L:    ₹{journal_summary['total_pnl']:+,.2f}")
        logger.info(f"  Win Rate:     {journal_summary['win_rate']:.1f}%")
        logger.info(f"  Gemini:       {TOKEN_BUDGET.summary()}")
        # MFE/MAE summary line
        mfe_stats = report.get("smart_exit_stats", {}).get("mfe_mae_analysis")
        if mfe_stats:
            logger.info(
                f"  Exit Quality: capture={mfe_stats.get('avg_capture_ratio', 0):.4f} "
                f"edge={mfe_stats.get('avg_edge_ratio', 0):.4f} "
                f"({mfe_stats.get('n_trades', 0)} trades)"
            )
        logger.info(f"  {'─'*50}")

        # Include token budget in report
        report["token_budget"] = {
            "tokens_used": TOKEN_BUDGET.tokens_used_today,
            "daily_limit": TOKEN_BUDGET.daily_limit,
            "calls": TOKEN_BUDGET.calls_today,
        }

    def _step_14_error(self, error: Exception, report: Dict) -> None:
        """Step 14: Error handling."""
        tb = traceback.format_exc()
        report["errors"].append(f"Pipeline error: {error}")
        report["traceback"] = tb
        logger.error(f"PIPELINE ERROR: {error}\n{tb}")

    # ══════════════════════════════════════════════════════════════════
    # First-run monitoring
    # ══════════════════════════════════════════════════════════════════

    def _log_first_run_monitor(
        self,
        today: str,
        report: Dict,
        combined_signals: Dict[str, Dict[str, Any]],
    ) -> None:
        """Log a detailed monitoring entry + print summary to stdout.

        Writes to logs/first_run_monitor.jsonl so each run's state
        can be verified.
        """
        summary = self.tracker.portfolio_summary()
        regime_info = report.get("regime", {})

        # Signal stats from combined
        long_signals = sum(1 for s in combined_signals.values()
                          if s.get("final_signal", 0) > 0 and s.get("should_trade"))
        short_signals = sum(1 for s in combined_signals.values()
                           if s.get("final_signal", 0) < 0 and s.get("should_trade"))
        hold_signals = sum(1 for s in combined_signals.values()
                          if not s.get("should_trade"))

        # Parse debate info from steps
        debate_info = report.get("debate_outcome") or {}
        debates_triggered = 0
        signals_changed = 0
        for step in report.get("steps_completed", []):
            if "5_debate" in str(step):
                import re
                m = re.search(r"(\d+) debates", str(step))
                if m:
                    debates_triggered = int(m.group(1))
                m2 = re.search(r"(\d+) changes", str(step))
                if m2:
                    signals_changed = int(m2.group(1))

        # Reflection / coaching from steps
        reflection_fired = report.get("reflection_triggered", False)
        coaching_configs = report.get("coaching_configs_changed", 0)
        coaching_fired = coaching_configs > 0

        # Evolved configs last modified
        cfg_path = Path(self.config.get(
            "evolved_configs_file",
            str(Path(__file__).parent / "evolved_player_configs.json")
        ))
        cfg_mtime = cfg_path.stat().st_mtime if cfg_path.exists() else 0

        entry = {
            "date": today,
            "run_number": self._state.get("total_runs", 0),

            # Regime verification
            "regime_detected": regime_info.get("current", "Unknown"),
            "regime_confidence": round(
                regime_info.get("probabilities", {}).get(
                    regime_info.get("current", ""), 0
                ) * 100, 1
            ),
            "regime_changed_from_yesterday": (
                regime_info.get("current") != self._state.get("last_regime")
                and self._state.get("last_regime") is not None
            ),

            # Debate verification
            "debates_triggered": debates_triggered,
            "signals_changed_by_debate": signals_changed,
            "gemini_calls_total": TOKEN_BUDGET.calls_today,

            # Reflection verification
            "reflection_triggered": reflection_fired,
            "total_closed_trades": self.journal.trade_count,

            # Coaching verification
            "coaching_triggered": coaching_fired,
            "configs_modified_by_coaching": coaching_configs,

            # Signal sanity
            "total_signals": report.get("signal_count", 0),
            "real_signals": report.get("real_signal_count", 0),
            "long_signals": long_signals,
            "short_signals": short_signals,
            "hold_signals": hold_signals,

            # Portfolio state
            "portfolio_value": round(summary["equity"], 2),
            "cash": round(summary["cash"], 2),
            "open_positions": summary["open_positions"],
            "daily_pnl": round(summary["unrealised_pnl"], 2),

            # Token usage
            "tokens_used_today": TOKEN_BUDGET.tokens_used_today,
            "token_limit": TOKEN_BUDGET.daily_limit,
            "gemini_calls": TOKEN_BUDGET.calls_today,

            # Evolved configs check
            "evolved_configs_last_modified": datetime.fromtimestamp(cfg_mtime).isoformat() if cfg_mtime else None,
        }

        # Write to JSONL log
        try:
            logs_dir = Path(__file__).parent / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            log_file = logs_dir / "first_run_monitor.jsonl"
            with open(log_file, "a") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write first-run monitor: {e}")

        # Print summary to stdout
        regime_str = entry["regime_detected"]
        regime_conf = entry["regime_confidence"]
        refl_str = "FIRED ✅" if reflection_fired else f"Not yet (need {self.config['reflection_every_n_trades']} closed trades)"
        coach_str = f"FIRED ✅ ({coaching_configs} configs)" if coaching_fired else f"Not yet (need {self.config['coaching_min_trades']} closed trades)"

        print(f"\n{'='*60}")
        print(f"  FIRST RUN MONITOR — Day {entry['run_number']}")
        print(f"{'='*60}")
        print(f"  Regime:      {regime_str} ({regime_conf}% confidence)")
        print(f"  Debate:      {debates_triggered} debates, {signals_changed} signals changed")
        print(f"  Reflection:  {refl_str}")
        print(f"  Coaching:    {coach_str}")
        print(f"  Signals:     {long_signals}L / {short_signals}S / {hold_signals}H ({entry['real_signals']} real)")
        print(f"  Portfolio:   ₹{entry['portfolio_value']:,.0f} | PnL: ₹{entry['daily_pnl']:+,.0f}")
        print(f"  Tokens:      {entry['tokens_used_today']:,} / {entry['token_limit']:,} ({entry['gemini_calls']} calls)")
        print(f"{'='*60}")

    # ══════════════════════════════════════════════════════════════════
    # Helper methods
    # ══════════════════════════════════════════════════════════════════

    def _pick_best_player(
        self, signal: Dict[str, Any], symbol: str,
    ) -> str:
        """Pick which player_id to attribute a trade to.
        Uses ensemble best_player from signal combination if available,
        otherwise falls back to round-robin.
        """
        # Use best_player from ensemble consensus if available
        best = signal.get("best_player")
        if best and best in PLAYER_IDS:
            return best

        # Fallback: round-robin based on current position count per player
        counts = {pid: 0 for pid in PLAYER_IDS}
        for pos in self.tracker.positions.values():
            counts[pos.player_id] = counts.get(pos.player_id, 0) + 1

        # Pick the player with fewest open positions
        return min(counts, key=counts.get)

    def _load_state(self) -> Dict[str, Any]:
        """Load pipeline state from disk (with EVOLVED+ENSEMBLE fields)."""
        defaults = {
            "days_since_evolution": 0,
            "total_runs": 0,
            "last_regime": "Sideways",
            "last_evolution_date": None,
            "evolution_trigger_type": None,
        }
        state_file = Path(self.config["state_file"])
        if state_file.exists():
            try:
                loaded = json.loads(state_file.read_text(encoding="utf-8"))
                # Merge with defaults to ensure all fields exist
                defaults.update(loaded)
                return defaults
            except (json.JSONDecodeError, OSError):
                pass
        return defaults

    def _save_state(self) -> None:
        """Save pipeline state to disk."""
        state_file = Path(self.config["state_file"])
        state_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._state["last_run"] = datetime.now().isoformat()
            state_file.write_text(
                json.dumps(self._state, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError as e:
            logger.error(f"Failed to save state: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the daily pipeline from command line.

    Usage:
        python daily_runner.py                       # full pipeline (default)
        python daily_runner.py --mode premarket      # 09:00 IST — signals + debates + execute
        python daily_runner.py --mode postmarket     # 15:45 IST — update prices + stops + reflect/coach
        python daily_runner.py --mode full           # full pipeline (all steps including debate)
        python daily_runner.py --mode intelligence   # morning brief + heatmap (legacy)
        python daily_runner.py --mode refresh        # midday news + breadth (legacy)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse --mode argument
    mode = "full"
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            mode = sys.argv[idx + 1]

    runner = DailyRunner()
    report = runner.run(mode=mode)

    mode_label = {"full": "Full Pipeline", "intelligence": "Morning Intelligence",
                  "refresh": "Midday Refresh", "premarket": "Pre-Market",
                  "postmarket": "Post-Market"}.get(mode, mode)

    if report["success"]:
        print(f"\n✅ {mode_label} completed successfully")
    else:
        print(f"\n❌ {mode_label} completed with {len(report['errors'])} errors")
        for err in report["errors"]:
            print(f"   • {err}")

    # Print mode-specific output
    if mode in ("full", "premarket", "postmarket") and report.get("summary"):
        s = report["summary"]
        print(f"\n  Equity:    ₹{s['equity']:,.2f}")
        print(f"  Win Rate:  {s['win_rate']:.1f}%")
        print(f"  Total P&L: ₹{s['total_pnl']:+,.2f}")

    elif mode == "intelligence" and report.get("morning_brief"):
        b = report["morning_brief"]
        print(f"\n  Regime:    {b['regime']} (conf {b['regime_confidence']:.0%})")
        print(f"  Equity:    ₹{b['portfolio_equity']:,.2f}")
        print(f"  Top Bull:  {', '.join(s['symbol'] for s in b['top_bullish'])}")
        print(f"  Top Bear:  {', '.join(s['symbol'] for s in b['top_bearish'])}")

    elif mode == "refresh":
        breadth = report.get("breadth", {})
        deltas = report.get("sentiment_deltas", [])
        print(f"\n  Breadth:   {breadth.get('advancing', 0)}↑ "
              f"{breadth.get('declining', 0)}↓ ({breadth.get('signal', 'n/a')})")
        print(f"  Shifts:    {len(deltas)} significant sentiment changes")


if __name__ == "__main__":
    main()
