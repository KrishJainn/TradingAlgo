"""
MFE/MAE Analyzer — Maximum Favorable/Adverse Excursion analysis.

Based on John Sweeney's MFE/MAE methodology and Curtis Faith's edge
ratio work.  Quantifies how well the SmartExitEngine captures available
profit and generates actionable tuning recommendations.

Key Metrics
===========
- **Capture Ratio** (actual_pnl / MFE): How much of peak profit we kept.
  Target 0.50-0.60 (institutional).  Below 0.40 = exits too late.
  Above 0.70 = exits too tight (leaving trend continuation on table).

- **Edge Ratio** (MFE/ATR ÷ MAE/ATR): Quality of entry + exit combined.
  Above 1.5 = good.  Above 2.0 = excellent.  Below 1.0 = entries are
  as bad as exits.

- **MFE Timing** (bar_of_peak / bars_held): When peak profit occurs.
  Below 0.35 = profits come early then erode → reduce max_hold_days.
  Above 0.70 = profits develop late → widen trailing stops.

Data Sources
============
The analyzer works with two data sources:

1. **Trades WITH MFE/MAE fields** (new trades from position_tracker):
   Uses the persisted `mfe`, `mae`, `peak_price` fields directly.

2. **Trades WITHOUT MFE/MAE** (legacy trades):
   Falls back to estimating MFE/MAE from `return_pct` and `bars_held`.
   Less accurate but allows analysis of historical trades.

Usage
=====
    from mfe_mae_analyzer import MFEMAEAnalyzer

    analyzer = MFEMAEAnalyzer()
    trades = journal.get_all_trades()

    # Analyze a single trade
    analysis = analyzer.analyze_trade(trades[0])

    # Full tuning report
    report = analyzer.generate_tuning_report(trades)
    print(report["recommendations_text"])

    # SmartExitEngine parameter suggestions
    suggestions = analyzer.generate_parameter_suggestions(report)

    # For Gemini exit prompt
    prompt_block = analyzer.format_for_gemini_prompt(report)
"""

from __future__ import annotations

import logging
import math
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# MFE/MAE Analyzer
# ═══════════════════════════════════════════════════════════════════════════════

class MFEMAEAnalyzer:
    """Post-trade analysis using Maximum Favorable/Adverse Excursion.

    Quantifies exit quality and generates SmartExitEngine tuning
    recommendations based on statistical patterns across completed trades.

    Parameters
    ----------
    min_trades_for_report : int
        Minimum number of analysable trades required to generate a
        tuning report.  Below this, sample size is too small.
    """

    def __init__(self, min_trades_for_report: int = 10) -> None:
        self._min_trades = min_trades_for_report

    # ══════════════════════════════════════════════════════════════════════
    # Single-trade analysis
    # ══════════════════════════════════════════════════════════════════════

    def analyze_trade(self, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyse a single completed trade for MFE/MAE metrics.

        Parameters
        ----------
        trade : dict
            Completed trade dict from trade_journal.  Expected keys:
            entry_price, exit_price, direction, bars_held, return_pct.
            Optional (better accuracy): mfe, mae, peak_price,
            entry_volatility, price_history.

        Returns
        -------
        dict or None
            Analysis dict with MFE, MAE, capture_ratio, edge_ratio,
            mfe_timing, exit_efficiency.  Returns None if trade data
            is insufficient for analysis.
        """
        entry_price = trade.get("entry_price", 0.0)
        exit_price = trade.get("exit_price", 0.0)
        direction = trade.get("direction", "LONG")
        bars_held = trade.get("bars_held", 0)

        if entry_price <= 0 or exit_price <= 0 or bars_held <= 0:
            return None

        # ── Compute actual P&L as fraction ──
        if direction == "LONG":
            actual_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            actual_pnl_pct = (entry_price - exit_price) / entry_price

        # ── Get MFE/MAE — prefer persisted values, fallback to estimate ──
        mfe = trade.get("mfe", 0.0) or 0.0
        mae = trade.get("mae", 0.0) or 0.0
        peak_price = trade.get("peak_price", 0.0) or 0.0

        has_mfe_data = (mfe > 0.0 or mae > 0.0)

        if not has_mfe_data:
            # Fallback: estimate from return_pct and bars_held
            # This is rough — better than nothing for legacy trades
            mfe, mae = self._estimate_mfe_mae(actual_pnl_pct, bars_held)

        # ── Capture ratio: how much of peak profit did we keep? ──
        if mfe > 0.001:
            capture_ratio = actual_pnl_pct / mfe
            # Clamp to [-1.0, 1.5] — can exceed 1.0 if MFE tracking
            # missed an intrabar peak, or go negative if trade lost
            capture_ratio = max(-1.0, min(1.5, capture_ratio))
        elif actual_pnl_pct <= 0:
            # Never profitable, no MFE to capture
            capture_ratio = 0.0
        else:
            # MFE ≈ 0 but actual_pnl > 0 → perfect capture
            capture_ratio = 1.0

        # ── Edge ratio: (MFE/ATR) / (MAE/ATR) = MFE/MAE ──
        # Higher is better — means favorable excursion dominates adverse
        entry_vol = trade.get("entry_volatility", 0.0) or 0.0
        atr_proxy = entry_vol * entry_price / math.sqrt(252) if entry_vol > 0 else entry_price * 0.015

        mfe_atr = (mfe * entry_price) / atr_proxy if atr_proxy > 0 else 0.0
        mae_atr = (mae * entry_price) / atr_proxy if atr_proxy > 0 else 0.0

        if mae_atr > 0.01:
            edge_ratio = mfe_atr / mae_atr
        elif mfe_atr > 0:
            edge_ratio = 10.0  # Capped — near-zero MAE is excellent
        else:
            edge_ratio = 0.0

        # ── MFE timing: when did peak profit occur? ──
        # If we have price_history (signal_history proxy), estimate peak bar
        price_history = trade.get("price_history", [])
        mfe_timing = self._compute_mfe_timing(
            price_history, direction, entry_price, peak_price, bars_held,
        )

        # ── Exit efficiency: how centered is the peak? ──
        # 1.0 when peak is at midpoint, lower when early or late
        exit_efficiency = 1.0 - abs(mfe_timing - 0.5) * 2.0
        exit_efficiency = max(0.0, min(1.0, exit_efficiency))

        # ── Giveback: how much of MFE was given back ──
        if mfe > 0.001:
            giveback_pct = (mfe - actual_pnl_pct) / mfe
            giveback_pct = max(0.0, giveback_pct)
        else:
            giveback_pct = 0.0

        return {
            "symbol": trade.get("symbol", ""),
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "bars_held": bars_held,
            "close_reason": trade.get("close_reason", ""),
            "regime": trade.get("regime", trade.get("entry_regime", "")),
            # Core MFE/MAE metrics
            "actual_pnl_pct": round(actual_pnl_pct, 6),
            "mfe": round(mfe, 6),
            "mae": round(mae, 6),
            "capture_ratio": round(capture_ratio, 4),
            "edge_ratio": round(edge_ratio, 4),
            "giveback_pct": round(giveback_pct, 4),
            # Timing
            "mfe_timing": round(mfe_timing, 4),
            "exit_efficiency": round(exit_efficiency, 4),
            # ATR-normalized
            "mfe_atr": round(mfe_atr, 3),
            "mae_atr": round(mae_atr, 3),
            # Metadata
            "has_mfe_data": has_mfe_data,
            "player_id": trade.get("player_id", ""),
            "partial": trade.get("partial", False),
        }

    # ══════════════════════════════════════════════════════════════════════
    # Aggregate tuning report
    # ══════════════════════════════════════════════════════════════════════

    def generate_tuning_report(
        self,
        trades: List[Dict[str, Any]],
        regime_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate aggregate MFE/MAE report with tuning recommendations.

        Parameters
        ----------
        trades : list of dict
            All completed trades from trade_journal.
        regime_filter : str, optional
            If set, only analyse trades from this regime.

        Returns
        -------
        dict
            Aggregate stats, per-regime breakdown, recommendations text,
            and raw analysed trades list.
        """
        # Filter by regime if requested
        if regime_filter:
            trades = [t for t in trades if t.get("regime", t.get("entry_regime", "")).lower() == regime_filter.lower()]

        # Analyse each trade
        analyses = []
        for t in trades:
            a = self.analyze_trade(t)
            if a is not None:
                analyses.append(a)

        if len(analyses) < self._min_trades:
            return {
                "status": "insufficient_data",
                "n_trades": len(analyses),
                "min_required": self._min_trades,
                "recommendations_text": f"Need at least {self._min_trades} trades for MFE/MAE analysis (have {len(analyses)}).",
            }

        # ── Compute aggregate stats ──
        winners = [a for a in analyses if a["actual_pnl_pct"] > 0]
        losers = [a for a in analyses if a["actual_pnl_pct"] <= 0]

        avg_capture = self._safe_mean([a["capture_ratio"] for a in analyses])
        avg_capture_winners = self._safe_mean([a["capture_ratio"] for a in winners]) if winners else 0.0
        avg_edge = self._safe_mean([a["edge_ratio"] for a in analyses])
        avg_mfe_timing = self._safe_mean([a["mfe_timing"] for a in analyses])
        avg_giveback = self._safe_mean([a["giveback_pct"] for a in analyses])
        avg_mfe = self._safe_mean([a["mfe"] for a in analyses])
        avg_mae = self._safe_mean([a["mae"] for a in analyses])
        avg_mfe_atr = self._safe_mean([a["mfe_atr"] for a in analyses])
        avg_mae_atr = self._safe_mean([a["mae_atr"] for a in analyses])

        # ── Quality percentages ──
        n = len(analyses)
        pct_poor_exits = sum(1 for a in analyses if a["capture_ratio"] < 0.5 and a["mfe"] > 0.01) / n
        pct_early_peak = sum(1 for a in analyses if a["mfe_timing"] < 0.3) / n
        exits_too_late = sum(1 for a in analyses if a["capture_ratio"] < 0.40) / n
        exits_too_early = sum(1 for a in analyses
                              if a["actual_pnl_pct"] > 0 and a["actual_pnl_pct"] < a["mfe"] * 0.30) / n

        # ── Per-regime breakdown ──
        regime_stats = {}
        for regime in ("bull", "bear", "sideways"):
            regime_trades = [a for a in analyses if a.get("regime", "").lower() == regime]
            if len(regime_trades) >= 3:
                regime_stats[regime] = {
                    "n_trades": len(regime_trades),
                    "avg_capture_ratio": round(self._safe_mean([a["capture_ratio"] for a in regime_trades]), 4),
                    "avg_edge_ratio": round(self._safe_mean([a["edge_ratio"] for a in regime_trades]), 4),
                    "avg_mfe_timing": round(self._safe_mean([a["mfe_timing"] for a in regime_trades]), 4),
                    "avg_giveback_pct": round(self._safe_mean([a["giveback_pct"] for a in regime_trades]), 4),
                    "win_rate": round(sum(1 for a in regime_trades if a["actual_pnl_pct"] > 0) / len(regime_trades), 4),
                }

        # ── Per-exit-reason breakdown ──
        reason_stats = {}
        reasons = set(a.get("close_reason", "") for a in analyses)
        for reason in reasons:
            if not reason:
                continue
            reason_trades = [a for a in analyses if a.get("close_reason", "") == reason]
            if len(reason_trades) >= 2:
                reason_stats[reason] = {
                    "n_trades": len(reason_trades),
                    "avg_capture_ratio": round(self._safe_mean([a["capture_ratio"] for a in reason_trades]), 4),
                    "avg_pnl_pct": round(self._safe_mean([a["actual_pnl_pct"] for a in reason_trades]), 6),
                    "avg_bars_held": round(self._safe_mean([a["bars_held"] for a in reason_trades]), 1),
                }

        # ── Generate recommendations ──
        recommendations = self._generate_recommendations(
            avg_capture, avg_capture_winners, avg_edge, avg_mfe_timing,
            avg_giveback, pct_poor_exits, pct_early_peak,
            exits_too_late, exits_too_early, n,
        )

        report = {
            "status": "ok",
            "n_trades": n,
            "n_winners": len(winners),
            "n_losers": len(losers),
            "win_rate": round(len(winners) / n, 4) if n > 0 else 0.0,
            # Core metrics
            "avg_capture_ratio": round(avg_capture, 4),
            "avg_capture_ratio_winners": round(avg_capture_winners, 4),
            "avg_edge_ratio": round(avg_edge, 4),
            "avg_mfe_timing": round(avg_mfe_timing, 4),
            "avg_giveback_pct": round(avg_giveback, 4),
            "avg_mfe": round(avg_mfe, 6),
            "avg_mae": round(avg_mae, 6),
            "avg_mfe_atr": round(avg_mfe_atr, 3),
            "avg_mae_atr": round(avg_mae_atr, 3),
            # Quality buckets
            "pct_poor_exits": round(pct_poor_exits, 4),
            "pct_early_peak": round(pct_early_peak, 4),
            "exits_too_late_pct": round(exits_too_late, 4),
            "exits_too_early_pct": round(exits_too_early, 4),
            # Breakdowns
            "by_regime": regime_stats,
            "by_exit_reason": reason_stats,
            # Recommendations
            "recommendations": recommendations,
            "recommendations_text": "\n".join(recommendations),
            # Raw data (for downstream use)
            "analyses": analyses,
        }

        logger.info(
            f"[MFE/MAE] Report: {n} trades | "
            f"capture={avg_capture:.2f} edge={avg_edge:.2f} "
            f"mfe_timing={avg_mfe_timing:.2f} giveback={avg_giveback:.1%}"
        )

        return report

    # ══════════════════════════════════════════════════════════════════════
    # Parameter suggestion engine
    # ══════════════════════════════════════════════════════════════════════

    def generate_parameter_suggestions(
        self,
        report: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate SmartExitEngine config adjustment suggestions.

        Based on the tuning report metrics, suggests concrete changes
        to REGIME_CONFIGS parameters.

        Parameters
        ----------
        report : dict
            Output from generate_tuning_report().

        Returns
        -------
        dict
            Keyed by config parameter name, each with: current direction
            of adjustment, multiplier, and reasoning.
        """
        if report.get("status") != "ok":
            return {"status": "insufficient_data"}

        suggestions: Dict[str, Any] = {"status": "ok", "adjustments": []}
        adj = suggestions["adjustments"]

        avg_capture = report.get("avg_capture_ratio", 0.5)
        avg_mfe_timing = report.get("avg_mfe_timing", 0.5)
        pct_poor = report.get("pct_poor_exits", 0.0)
        avg_edge = report.get("avg_edge_ratio", 1.0)
        exits_late = report.get("exits_too_late_pct", 0.0)
        exits_early = report.get("exits_too_early_pct", 0.0)

        # ── Capture ratio adjustments ──
        if avg_capture < 0.40:
            adj.append({
                "param": "trailing_stop_pct",
                "action": "tighten",
                "multiplier": 0.85,
                "reason": f"Capture ratio {avg_capture:.2f} < 0.40 — exits too late, tighten trailing stops",
            })
            adj.append({
                "param": "max_hold_days",
                "action": "reduce",
                "delta": -2,
                "reason": f"Capture ratio {avg_capture:.2f} < 0.40 — reduce max holding period",
            })
        elif avg_capture > 0.70:
            adj.append({
                "param": "trailing_stop_pct",
                "action": "widen",
                "multiplier": 1.15,
                "reason": f"Capture ratio {avg_capture:.2f} > 0.70 — exits too tight, let winners run",
            })
            adj.append({
                "param": "max_hold_days",
                "action": "increase",
                "delta": +2,
                "reason": f"Capture ratio {avg_capture:.2f} > 0.70 — allow longer holds",
            })

        # ── MFE timing adjustments ──
        if avg_mfe_timing < 0.35:
            adj.append({
                "param": "partial_exit_1_atr",
                "action": "tighten",
                "multiplier": 0.85,
                "reason": f"MFE timing {avg_mfe_timing:.2f} < 0.35 — profits peak early, take staged exits sooner",
            })
            adj.append({
                "param": "time_decay_start_day",
                "action": "reduce",
                "delta": -1,
                "reason": f"MFE timing {avg_mfe_timing:.2f} — start time decay earlier",
            })
        elif avg_mfe_timing > 0.70:
            adj.append({
                "param": "trailing_stop_pct",
                "action": "widen",
                "multiplier": 1.10,
                "reason": f"MFE timing {avg_mfe_timing:.2f} > 0.70 — profits develop late, widen stops",
            })

        # ── Poor exit rate ──
        if pct_poor > 0.40:
            adj.append({
                "param": "EXIT_THRESHOLD",
                "action": "lower",
                "delta": -0.05,
                "reason": f"Poor exits {pct_poor:.0%} > 40% — lower EXIT_THRESHOLD to trigger exits sooner",
            })

        # ── Edge ratio ──
        if avg_edge < 1.0:
            adj.append({
                "param": "entry_quality",
                "action": "review",
                "reason": f"Edge ratio {avg_edge:.2f} < 1.0 — adverse excursion matches favorable. "
                          f"Entry quality needs work, not just exits.",
            })
        elif avg_edge < 1.5:
            adj.append({
                "param": "sl_atr_mult",
                "action": "tighten",
                "multiplier": 0.90,
                "reason": f"Edge ratio {avg_edge:.2f} < 1.5 — tighten stop-loss multiplier",
            })

        # ── Drawdown protection ──
        if report.get("avg_giveback_pct", 0) > 0.50:
            adj.append({
                "param": "drawdown_from_peak_pct",
                "action": "tighten",
                "multiplier": 0.80,
                "reason": f"Avg giveback {report['avg_giveback_pct']:.0%} > 50% — tighten drawdown protection",
            })

        suggestions["n_adjustments"] = len(adj)
        return suggestions

    # ══════════════════════════════════════════════════════════════════════
    # Gemini prompt formatter
    # ══════════════════════════════════════════════════════════════════════

    def format_for_gemini_prompt(self, report: Dict[str, Any]) -> str:
        """Format the MFE/MAE report as a compact text block for Gemini.

        Designed to be injected into the exit reasoning prompt so Gemini
        can consider historical exit quality when making decisions.

        Parameters
        ----------
        report : dict
            Output from generate_tuning_report().

        Returns
        -------
        str
            Multi-line text block, or empty string if insufficient data.
        """
        if report.get("status") != "ok":
            return ""

        n = report.get("n_trades", 0)
        cap = report.get("avg_capture_ratio", 0)
        edge = report.get("avg_edge_ratio", 0)
        timing = report.get("avg_mfe_timing", 0)
        giveback = report.get("avg_giveback_pct", 0)
        win_rate = report.get("win_rate", 0)
        poor = report.get("pct_poor_exits", 0)

        # Capture quality label
        if cap >= 0.60:
            cap_label = "GOOD"
        elif cap >= 0.45:
            cap_label = "OK"
        elif cap >= 0.35:
            cap_label = "POOR"
        else:
            cap_label = "BAD"

        # Edge quality label
        if edge >= 2.0:
            edge_label = "excellent"
        elif edge >= 1.5:
            edge_label = "good"
        elif edge >= 1.0:
            edge_label = "marginal"
        else:
            edge_label = "poor"

        lines = [
            f"Based on {n} completed trades (win rate {win_rate:.0%}):",
            f"  Capture ratio: {cap:.2f} ({cap_label}) — target 0.50-0.60",
            f"  Edge ratio: {edge:.2f} ({edge_label}) — target >1.5",
            f"  MFE timing: {timing:.2f} — {'profits peak early' if timing < 0.4 else 'profits develop late' if timing > 0.6 else 'well-centered'}",
            f"  Avg giveback: {giveback:.0%} of peak profit",
            f"  Poor exits: {poor:.0%} (capture<0.5 AND MFE>1%)",
        ]

        # Add regime breakdown if available
        by_regime = report.get("by_regime", {})
        if by_regime:
            lines.append("  By regime:")
            for regime, stats in sorted(by_regime.items()):
                lines.append(
                    f"    {regime}: capture={stats['avg_capture_ratio']:.2f} "
                    f"edge={stats['avg_edge_ratio']:.2f} "
                    f"n={stats['n_trades']}"
                )

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════
    # Internal helpers
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        """Mean with empty-list guard."""
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _estimate_mfe_mae(
        actual_pnl_pct: float,
        bars_held: int,
    ) -> Tuple[float, float]:
        """Estimate MFE/MAE for legacy trades without persisted values.

        Heuristic based on:
        - Winners: MFE ≈ actual_pnl × 1.3 (typical 30% giveback)
        - Losers: MFE ≈ |actual_pnl| × 0.3 (briefly profitable)
        - MAE for winners: ≈ actual_pnl × 0.3
        - MAE for losers: ≈ |actual_pnl| × 1.1

        These are rough approximations — real MFE/MAE from position
        tracking is always preferred.
        """
        abs_pnl = abs(actual_pnl_pct)

        if actual_pnl_pct > 0:
            # Winner: was probably more profitable at peak
            mfe = abs_pnl * 1.3
            # Winner still had some adverse excursion
            mae = abs_pnl * 0.3 + 0.002  # floor at 0.2%
        elif actual_pnl_pct < -0.001:
            # Loser: was briefly profitable (or not at all)
            mfe = abs_pnl * 0.3
            mae = abs_pnl * 1.1
        else:
            # Flat trade
            mfe = 0.002
            mae = 0.002

        # Scale slightly by hold duration (longer holds → more excursion)
        duration_scale = min(2.0, 1.0 + bars_held * 0.02)
        mfe *= duration_scale * 0.8  # conservative
        mae *= duration_scale * 0.8

        return (round(mfe, 6), round(mae, 6))

    @staticmethod
    def _compute_mfe_timing(
        price_history: List[float],
        direction: str,
        entry_price: float,
        peak_price: float,
        bars_held: int,
    ) -> float:
        """Compute when MFE occurred as fraction of holding period.

        Returns
        -------
        float
            0.0 = peak at entry, 1.0 = peak at exit, 0.5 = midpoint.
        """
        if bars_held <= 0:
            return 0.5

        # If we have signal_history (used as price proxy), find peak bar
        if price_history and len(price_history) >= 2:
            if direction == "LONG":
                peak_idx = 0
                peak_val = price_history[0]
                for i, v in enumerate(price_history):
                    if v > peak_val:
                        peak_val = v
                        peak_idx = i
            else:
                # SHORT: peak MFE is at minimum signal (most bearish)
                peak_idx = 0
                peak_val = price_history[0]
                for i, v in enumerate(price_history):
                    if v < peak_val:
                        peak_val = v
                        peak_idx = i

            return peak_idx / max(1, len(price_history) - 1)

        # Fallback: if we have peak_price and entry_price, estimate
        # based on how close exit was to peak
        if peak_price > 0 and entry_price > 0:
            # Heuristic: assume peak occurred at ~60% of hold period
            # (institutional average for trend-following systems)
            return 0.6

        # No data — return midpoint as neutral estimate
        return 0.5

    def _generate_recommendations(
        self,
        avg_capture: float,
        avg_capture_winners: float,
        avg_edge: float,
        avg_mfe_timing: float,
        avg_giveback: float,
        pct_poor_exits: float,
        pct_early_peak: float,
        exits_too_late: float,
        exits_too_early: float,
        n_trades: int,
    ) -> List[str]:
        """Generate plain-text tuning recommendations."""
        recs: List[str] = []

        # ── Capture ratio ──
        if avg_capture < 0.35:
            recs.append(
                f"Capture ratio {avg_capture:.2f} — exits far too late. "
                f"Tighten trailing stops (x0.85) AND reduce max_hold_days by 2."
            )
        elif avg_capture < 0.40:
            recs.append(
                f"Capture ratio {avg_capture:.2f} — exits too late. "
                f"Tighten trailing stops or reduce max_hold_days."
            )
        elif avg_capture > 0.75:
            recs.append(
                f"Capture ratio {avg_capture:.2f} — exits too tight. "
                f"Widen trailing stops to let winners run longer."
            )
        elif avg_capture >= 0.50:
            recs.append(
                f"Capture ratio {avg_capture:.2f} — in target range (0.50-0.60). "
                f"Exit calibration is good."
            )

        # ── Winner capture specifically ──
        if avg_capture_winners > 0 and avg_capture_winners < 0.45:
            recs.append(
                f"Winner capture ratio {avg_capture_winners:.2f} — "
                f"winning trades give back too much profit before exit."
            )

        # ── MFE timing ──
        if avg_mfe_timing < 0.28:
            recs.append(
                f"MFE timing {avg_mfe_timing:.2f} — profits peak very early. "
                f"Consider earlier staged exit tranches (partial_exit_1_atr x0.85)."
            )
        elif avg_mfe_timing < 0.35:
            recs.append(
                f"MFE timing {avg_mfe_timing:.2f} — profits peak early. "
                f"Consider earlier staged exit tranches."
            )
        elif avg_mfe_timing > 0.75:
            recs.append(
                f"MFE timing {avg_mfe_timing:.2f} — profits develop late. "
                f"Widen trailing stops, current stops may clip developing trends."
            )

        # ── Edge ratio ──
        if avg_edge < 0.8:
            recs.append(
                f"Edge ratio {avg_edge:.2f} — very poor. "
                f"Adverse excursion exceeds favorable. Entry quality needs review, not just exits."
            )
        elif avg_edge < 1.0:
            recs.append(
                f"Edge ratio {avg_edge:.2f} — poor. "
                f"Entry quality may need work, not just exits."
            )
        elif avg_edge >= 2.0:
            recs.append(
                f"Edge ratio {avg_edge:.2f} — excellent. "
                f"Entries are well-timed relative to adverse moves."
            )

        # ── Giveback ──
        if avg_giveback > 0.60:
            recs.append(
                f"Average giveback {avg_giveback:.0%} of peak profit — "
                f"too much profit erosion. Tighten drawdown_from_peak_pct."
            )

        # ── Poor exit rate ──
        if pct_poor_exits > 0.50:
            recs.append(
                f"Poor exit rate {pct_poor_exits:.0%} — over half of exits have "
                f"capture<0.5 with MFE>1%. Lower EXIT_THRESHOLD by 0.05."
            )
        elif pct_poor_exits > 0.35:
            recs.append(
                f"Poor exit rate {pct_poor_exits:.0%} — consider lowering EXIT_THRESHOLD."
            )

        # ── Early peak rate ──
        if pct_early_peak > 0.50:
            recs.append(
                f"Early peak rate {pct_early_peak:.0%} — over half of trades peak "
                f"before 30% of hold. Positions are held too long after peak."
            )

        # ── Summary verdict ──
        if not recs:
            recs.append("All MFE/MAE metrics within target ranges. No adjustments needed.")

        return recs

    # ══════════════════════════════════════════════════════════════════════
    # Diagnostics
    # ══════════════════════════════════════════════════════════════════════

    def summary(self, report: Dict[str, Any]) -> str:
        """One-line summary for logging."""
        if report.get("status") != "ok":
            return f"MFE/MAE: insufficient data ({report.get('n_trades', 0)} trades)"
        return (
            f"MFE/MAE: {report['n_trades']} trades | "
            f"capture={report['avg_capture_ratio']:.2f} "
            f"edge={report['avg_edge_ratio']:.2f} "
            f"timing={report['avg_mfe_timing']:.2f} "
            f"giveback={report['avg_giveback_pct']:.0%} "
            f"poor_exits={report['pct_poor_exits']:.0%}"
        )
