"""
Dual Memory Context for the Closed-Loop Trading System.

Queries THREE ChromaDB collections and assembles a unified context:
  1. "trading_knowledge" - Books, papers, strategy docs (existing)
  2. "trade_history"     - Past trade records (experience replay)
  3. "run_summaries"     - Past run aggregates (coach memory)

This module COMPOSES the existing KnowledgeContext (does not modify it).
"""

import json
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class DualMemoryContext:
    """
    Queries both the static knowledge base and the experience replay
    collections, then formats a unified context string for player/coach prompts.
    """

    def __init__(self):
        from .context_layer import KnowledgeContext
        from .experience_replay import ExperienceReplayWriter

        self.knowledge = KnowledgeContext()
        self.replay = ExperienceReplayWriter()

    # ------------------------------------------------------------------
    # Player-level context (used in per-player prompts)
    # ------------------------------------------------------------------
    def get_player_context(
        self,
        player_id: str,
        player_label: str,
        regime: str,
        max_tokens: int = 2000,
    ) -> str:
        """
        Build complete context for a player before signal generation.
        Combines book knowledge + own trade history + run summaries.

        Args:
            player_id: e.g. "PLAYER_1"
            player_label: e.g. "Aggressive"
            regime: Current market regime label
            max_tokens: Approximate token budget

        Returns:
            Formatted context string ready for prompt injection.
        """
        sections = []

        # --- Layer 1: Domain knowledge (existing RAG) ---
        try:
            knowledge_ctx = self.knowledge.get_context_for_player(
                player_name=player_label.lower(),
                market_context={"regime": regime},
                query=f"{player_label} trading strategy {regime} market",
                max_tokens=max_tokens // 3,
            )
            if knowledge_ctx and "No specific knowledge" not in knowledge_ctx:
                sections.append("=== DOMAIN KNOWLEDGE ===")
                sections.append(knowledge_ctx)
        except Exception as e:
            logger.warning(f"Knowledge query failed: {e}")

        # --- Layer 2: Own trade history ---
        try:
            trade_results = self.replay.query_trades(
                query_text=f"{player_label} trades in {regime} regime results",
                player_id=player_id,
                regime=regime,
                n_results=15,
            )
            trade_summary = self._summarize_trades(trade_results, player_label, regime)
            if trade_summary:
                sections.append(f"\n=== YOUR PAST TRADES IN {regime.upper()} ===")
                sections.append(trade_summary)
        except Exception as e:
            logger.warning(f"Trade history query failed: {e}")

        # --- Layer 3: Run summaries (coach memory) ---
        try:
            run_results = self.replay.query_runs(
                query_text=f"profitable runs {regime} regime best performance",
                regime=regime,
                n_results=5,
            )
            run_summary = self._summarize_runs(run_results, player_label)
            if run_summary:
                sections.append("\n=== BEST PERFORMING CONFIGURATIONS ===")
                sections.append(run_summary)
        except Exception as e:
            logger.warning(f"Run summary query failed: {e}")

        if not sections:
            return "No prior knowledge or trade history available yet."

        return "\n".join(sections)

    # ------------------------------------------------------------------
    # Coach-level context
    # ------------------------------------------------------------------
    def get_context_for_coach(
        self,
        situation: str,
        regime: str = "unknown",
        max_tokens: int = 2500,
    ) -> str:
        """Get context for coach-level decisions (cross-player)."""
        sections = []

        # Knowledge base
        try:
            kb = self.knowledge.get_context_for_coach(
                situation=situation,
                max_tokens=max_tokens // 2,
            )
            if kb:
                sections.append(kb)
        except Exception:
            pass

        # Cross-player run summaries
        try:
            run_results = self.replay.query_runs(
                query_text=f"team performance {regime} regime",
                regime=regime if regime != "unknown" else None,
                n_results=5,
            )
            if run_results["documents"][0]:
                sections.append("\n=== RECENT RUN HISTORY ===")
                for doc in run_results["documents"][0][:5]:
                    sections.append(f"  {doc[:300]}")
        except Exception:
            pass

        return "\n".join(sections) if sections else ""

    # ------------------------------------------------------------------
    # Proven config retrieval (for LLMAnalyst)
    # ------------------------------------------------------------------
    def get_proven_configs(
        self,
        player_id: str,
        regime: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Retrieve the best-performing configs from run_summaries for
        a given player in a given regime.

        Returns list of dicts with keys: run_id, team_sharpe, config
        """
        try:
            results = self.replay.query_runs(
                query_text=f"best performance {regime} regime high sharpe",
                regime=regime,
                n_results=top_k * 2,
            )

            configs = []
            for meta in results.get("metadatas", [[]])[0]:
                configs_json = meta.get("player_configs_json", "{}")
                try:
                    all_configs = json.loads(configs_json)
                    player_config = all_configs.get(player_id, {})
                    if player_config:
                        configs.append({
                            "run_id": meta.get("run_id", "unknown"),
                            "team_sharpe": meta.get("team_sharpe", 0),
                            "team_pnl": meta.get("team_pnl", 0),
                            "config": player_config,
                        })
                except (json.JSONDecodeError, TypeError):
                    continue

            # Sort by team_sharpe descending
            configs.sort(key=lambda x: x["team_sharpe"], reverse=True)
            return configs[:top_k]

        except Exception as e:
            logger.warning(f"Proven config retrieval failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Compatibility methods (match KnowledgeContext API)
    # ------------------------------------------------------------------
    def get_context_for_player(
        self,
        player_name: str,
        market_context: Dict,
        query: str,
        max_tokens: int = 1500,
    ) -> str:
        """Compatible with KnowledgeContext.get_context_for_player()."""
        regime = market_context.get("regime", "unknown")
        # Map player_name to player_id is not possible without external mapping,
        # so we just use the knowledge layer for this compatibility path
        return self.knowledge.get_context_for_player(
            player_name=player_name,
            market_context=market_context,
            query=query,
            max_tokens=max_tokens,
        )

    def get_risk_management_context(self, position_info: Dict) -> str:
        """Compatible with KnowledgeContext.get_risk_management_context()."""
        return self.knowledge.get_risk_management_context(position_info)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict:
        """Combined stats from knowledge base + experience replay."""
        replay_stats = self.replay.get_stats()
        try:
            kb_status = self.knowledge.get_status()
            kb_chunks = kb_status.get("total_chunks", 0)
        except Exception:
            kb_chunks = 0

        return {
            "knowledge_chunks": kb_chunks,
            "trade_history_count": replay_stats.get("trade_count", 0),
            "run_summary_count": replay_stats.get("run_count", 0),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _summarize_trades(
        self, results: Dict, player_label: str, regime: str
    ) -> str:
        """Summarize trade history results into concise text."""
        metadatas = results.get("metadatas", [[]])[0]
        if not metadatas:
            return f"No prior trade history for {player_label} in {regime} regime yet."

        wins = [m for m in metadatas if m.get("outcome") == "win"]
        losses = [m for m in metadatas if m.get("outcome") == "loss"]

        lines = []

        if wins:
            avg_pnl = np.mean([w.get("pnl", 0) for w in wins])
            avg_bars = np.mean([w.get("bars_held", 0) for w in wins])
            exit_reasons = {}
            for w in wins:
                r = w.get("exit_reason", "unknown")
                exit_reasons[r] = exit_reasons.get(r, 0) + 1
            top_exit = max(exit_reasons, key=exit_reasons.get) if exit_reasons else "N/A"
            lines.append(
                f"Winning trades ({len(wins)}): Avg PnL {avg_pnl:+.0f}, "
                f"Avg hold {avg_bars:.0f} bars, Most common exit: {top_exit}"
            )

        if losses:
            avg_pnl = np.mean([l.get("pnl", 0) for l in losses])
            avg_bars = np.mean([l.get("bars_held", 0) for l in losses])
            exit_reasons = {}
            for l in losses:
                r = l.get("exit_reason", "unknown")
                exit_reasons[r] = exit_reasons.get(r, 0) + 1
            top_exit = max(exit_reasons, key=exit_reasons.get) if exit_reasons else "N/A"
            lines.append(
                f"Losing trades ({len(losses)}): Avg PnL {avg_pnl:+.0f}, "
                f"Avg hold {avg_bars:.0f} bars, Most common exit: {top_exit}"
            )

        if not wins and not losses:
            return f"No prior trade history for {player_label} in {regime} regime yet."

        win_rate = len(wins) / (len(wins) + len(losses)) * 100
        lines.append(f"Historical win rate in {regime}: {win_rate:.0f}%")

        return "\n".join(lines)

    def _summarize_runs(self, results: Dict, player_label: str) -> str:
        """Summarize run history results into concise text."""
        metadatas = results.get("metadatas", [[]])[0]
        if not metadatas:
            return ""

        lines = []
        for meta in metadatas[:5]:
            run_id = meta.get("run_id", "?")
            team_sharpe = meta.get("team_sharpe", 0)
            best = meta.get("best_player", "N/A")
            team_pnl = meta.get("team_pnl", 0)
            lines.append(
                f"Run {run_id}: Team Sharpe {team_sharpe:.2f}, "
                f"PnL {team_pnl:+.0f}, Best: {best}"
            )

        return "\n".join(lines)
