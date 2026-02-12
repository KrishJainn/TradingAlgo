"""
Experience Replay Writer for the Closed-Loop Trading System.

Writes per-trade results and per-run summaries into ChromaDB collections
after each backtest run. This creates the feedback loop: trades from Run N
become context for Run N+1.

Collections:
  - "trade_history": Individual trade records with natural-language descriptions
  - "run_summaries": Per-run aggregate stats with player configs in metadata
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ExperienceReplayWriter:
    """Writes trade results and run summaries back into ChromaDB after each run."""

    def __init__(self):
        from .scripts.query import get_chroma_client, get_embedding_function, load_config

        self._config = load_config()
        self._client = get_chroma_client()
        self._embed_fn = get_embedding_function()

        trade_collection_name = self._config.get("chroma", {}).get(
            "trade_history_collection", "trade_history"
        )
        run_collection_name = self._config.get("chroma", {}).get(
            "run_summaries_collection", "run_summaries"
        )

        self._trades = self._client.get_or_create_collection(
            name=trade_collection_name,
            embedding_function=self._embed_fn,
            metadata={"description": "Per-trade results from backtest runs"},
        )
        self._runs = self._client.get_or_create_collection(
            name=run_collection_name,
            embedding_function=self._embed_fn,
            metadata={"description": "Per-run aggregate statistics"},
        )

    def write_trades(
        self,
        trades: List[Dict],
        player_id: str,
        player_label: str,
        run_id: str,
        regime: str,
    ):
        """
        Batch-write all trades for a player from a single run.

        Args:
            trades: List of trade dicts with keys: symbol, direction, entry_price,
                    exit_price, quantity, pnl, exit_reason, bars_held
            player_id: e.g. "PLAYER_1"
            player_label: e.g. "Aggressive"
            run_id: Unique run identifier
            regime: Market regime label
        """
        if not trades:
            return

        ids = []
        documents = []
        metadatas = []

        for i, trade in enumerate(trades):
            trade_id = f"{run_id}_{player_id}_t{i}"
            pnl = trade.get("pnl", 0)
            outcome = "win" if pnl > 0 else "loss"

            doc = (
                f"{player_label} player {trade.get('direction', 'UNKNOWN')} "
                f"{trade.get('symbol', 'UNKNOWN')} in {regime} regime. "
                f"Entry at {trade.get('entry_price', 0):.2f}, "
                f"exit at {trade.get('exit_price', 0):.2f} "
                f"({trade.get('exit_reason', 'unknown')}). "
                f"Held {trade.get('bars_held', 0)} bars. "
                f"PnL: {pnl:+.2f} ({outcome.upper()})"
            )

            meta = {
                "run_id": run_id,
                "player_id": player_id,
                "player_label": player_label,
                "symbol": trade.get("symbol", "UNKNOWN"),
                "direction": trade.get("direction", "UNKNOWN"),
                "regime": regime,
                "pnl": float(pnl),
                "outcome": outcome,
                "exit_reason": trade.get("exit_reason", "unknown"),
                "bars_held": int(trade.get("bars_held", 0)),
                "entry_price": float(trade.get("entry_price", 0)),
                "exit_price": float(trade.get("exit_price", 0)),
                "timestamp": datetime.now().isoformat(),
            }

            ids.append(trade_id)
            documents.append(doc)
            metadatas.append(meta)

        try:
            self._trades.upsert(ids=ids, documents=documents, metadatas=metadatas)
            logger.info(
                f"[ExperienceReplay] Wrote {len(trades)} trades for {player_id}"
            )
        except Exception as e:
            logger.error(f"[ExperienceReplay] Failed to write trades: {e}")

    def write_run_summary(
        self,
        run_results: Dict,
        run_id: str,
        regime: str,
        player_configs: Dict,
    ):
        """
        Write a run-level summary after a backtest completes.

        Args:
            run_results: The results dict from run_5player_backtest()
            run_id: Unique run identifier
            regime: Market regime label
            player_configs: Dict of player_id -> config dict
        """
        players = run_results.get("players", {})
        team_pnl = run_results.get("team_total_pnl", 0)

        # Build player summary lines
        player_lines = []
        best_player = None
        best_sharpe = float("-inf")
        worst_player = None
        worst_sharpe = float("inf")

        for pid, pdata in players.items():
            label = pdata.get("label", pid)
            pnl = pdata.get("pnl", 0)
            wr = pdata.get("win_rate", 0)
            trades = pdata.get("trades", 0)
            sharpe = pdata.get("sharpe", 0)

            player_lines.append(
                f"{label}: PnL {pnl:+.0f}, WR {wr:.0%}, {trades} trades, Sharpe {sharpe:.2f}"
            )

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_player = label
            if sharpe < worst_sharpe:
                worst_sharpe = sharpe
                worst_player = label

        doc = (
            f"Run {run_id} in {regime} regime. "
            f"Team PnL: {team_pnl:+.0f}. "
            f"Best: {best_player} (Sharpe {best_sharpe:.2f}). "
            f"Worst: {worst_player} (Sharpe {worst_sharpe:.2f}). "
            + " | ".join(player_lines)
        )

        # Compute team sharpe from player sharpes
        sharpes = [p.get("sharpe", 0) for p in players.values()]
        team_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0

        meta = {
            "run_id": run_id,
            "regime": regime,
            "team_pnl": float(team_pnl),
            "team_sharpe": float(team_sharpe),
            "best_player": best_player or "N/A",
            "best_sharpe": float(best_sharpe) if best_sharpe != float("-inf") else 0,
            "worst_player": worst_player or "N/A",
            "worst_sharpe": float(worst_sharpe) if worst_sharpe != float("inf") else 0,
            "total_trades": int(run_results.get("team_total_trades", 0)),
            "timestamp": datetime.now().isoformat(),
            "player_configs_json": json.dumps(
                {pid: _serialize_config(cfg) for pid, cfg in player_configs.items()}
            ),
        }

        # Add per-player PnL as individual metadata fields for filtering
        for pid, pdata in players.items():
            safe_label = pdata.get("label", pid).lower()
            meta[f"pnl_{safe_label}"] = float(pdata.get("pnl", 0))
            meta[f"wr_{safe_label}"] = float(pdata.get("win_rate", 0))

        try:
            self._runs.upsert(
                ids=[f"run_{run_id}"],
                documents=[doc],
                metadatas=[meta],
            )
            logger.info(f"[ExperienceReplay] Wrote run summary for {run_id}")
        except Exception as e:
            logger.error(f"[ExperienceReplay] Failed to write run summary: {e}")

    def query_trades(
        self,
        query_text: str,
        player_id: str = None,
        regime: str = None,
        n_results: int = 10,
    ) -> Dict:
        """Query trade history with optional filters."""
        where_clauses = []
        if player_id:
            where_clauses.append({"player_id": {"$eq": player_id}})
        if regime:
            where_clauses.append({"regime": {"$eq": regime}})

        where = None
        if len(where_clauses) == 1:
            where = where_clauses[0]
        elif len(where_clauses) > 1:
            where = {"$and": where_clauses}

        try:
            return self._trades.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"[ExperienceReplay] Trade query failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def query_runs(
        self,
        query_text: str,
        regime: str = None,
        n_results: int = 5,
    ) -> Dict:
        """Query run summaries with optional regime filter."""
        where = {"regime": {"$eq": regime}} if regime else None

        try:
            return self._runs.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"[ExperienceReplay] Run query failed: {e}")
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_stats(self) -> Dict:
        """Return collection statistics for dashboard display."""
        try:
            return {
                "trade_count": self._trades.count(),
                "run_count": self._runs.count(),
            }
        except Exception:
            return {"trade_count": 0, "run_count": 0}

    @property
    def trades_collection(self):
        return self._trades

    @property
    def runs_collection(self):
        return self._runs


def _serialize_config(config: Dict) -> Dict:
    """Serialize a player config to JSON-safe format."""
    safe = {}
    for k, v in config.items():
        if isinstance(v, dict):
            safe[k] = {str(kk): float(vv) if isinstance(vv, (int, float)) else str(vv) for kk, vv in v.items()}
        elif isinstance(v, (int, float, str, bool)):
            safe[k] = v
        else:
            safe[k] = str(v)
    return safe
