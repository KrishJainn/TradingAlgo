"""
Player Genome - Evolutionary Parameter Tracking for Each Player.

Instead of the LLM regenerating strategies from scratch each run,
each player maintains a parameter vector (genome) that evolves:
  - 70% exploit: use the best known config (small mutation)
  - 20% explore: larger mutation of current config
  - 10% reset: return to default config

The genome tracks all configs tried + their results, enabling
data-driven evolution independent of LLM quality.
"""

import json
import logging
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

GENOMES_DIR = Path(__file__).parent.parent.parent / "player_genomes"


class PlayerGenome:
    """Evolutionary parameter vector for a single player."""

    def __init__(self, player_id: str, player_label: str, initial_config: Dict):
        self.player_id = player_id
        self.player_label = player_label
        self.current_config = deepcopy(initial_config)
        self.best_config: Optional[Dict] = None
        self.best_pnl = float("-inf")
        self.best_sharpe = float("-inf")
        self.generation = 0
        self.history: List[Dict] = []

        self._load()

    def record_result(
        self,
        config: Dict,
        pnl: float,
        win_rate: float,
        sharpe: float,
        regime: str,
    ):
        """Record the result of a run with this config."""
        entry = {
            "generation": self.generation,
            "pnl": pnl,
            "win_rate": win_rate,
            "sharpe": sharpe,
            "regime": regime,
            "timestamp": datetime.now().isoformat(),
            "num_indicators": len(config.get("weights", {})),
            "entry_threshold": config.get("entry_threshold", 0),
            "config_hash": _config_hash(config),
        }
        self.history.append(entry)
        self.generation += 1

        # Update best
        if pnl > self.best_pnl:
            self.best_pnl = pnl
            self.best_sharpe = sharpe
            self.best_config = deepcopy(config)
            logger.info(
                f"[Genome] {self.player_id} new best: PnL {pnl:+.0f}, "
                f"Sharpe {sharpe:.2f} (gen {self.generation})"
            )

        self.current_config = deepcopy(config)
        self._save()

    def get_next_params(self) -> Dict:
        """
        Get the next parameter config using evolutionary strategy:
          70% exploit: small mutation of best config
          20% explore: larger mutation
          10% reset: return to current (possibly LLM-modified) config
        """
        if self.best_config is None:
            return deepcopy(self.current_config)

        r = random.random()
        if r < 0.70:
            # Exploit: small mutation of best known config
            return self._mutate(self.best_config, intensity=0.05)
        elif r < 0.90:
            # Explore: larger mutation
            return self._mutate(self.current_config, intensity=0.15)
        else:
            # Reset: use current config as-is (e.g. LLM-suggested)
            return deepcopy(self.current_config)

    def get_top_configs(self, k: int = 5) -> List[Dict]:
        """Return top k configs by PnL from history."""
        sorted_history = sorted(
            self.history, key=lambda x: x.get("pnl", 0), reverse=True
        )
        return sorted_history[:k]

    def get_regime_best(self, regime: str) -> Optional[Dict]:
        """Return the best config for a specific regime."""
        regime_runs = [h for h in self.history if h.get("regime") == regime]
        if not regime_runs:
            return None
        best = max(regime_runs, key=lambda x: x.get("pnl", 0))
        return best

    def _mutate(self, config: Dict, intensity: float = 0.1) -> Dict:
        """
        Mutate a config's weights and thresholds.

        Args:
            config: Config dict to mutate
            intensity: How much to perturb (0.05 = conservative, 0.15 = aggressive)
        """
        new_config = deepcopy(config)

        # Mutate weights
        weights = new_config.get("weights", {})
        new_weights = {}
        for ind, w in weights.items():
            delta = random.gauss(0, intensity)
            new_w = w * (1 + delta)
            new_weights[ind] = max(0.05, min(1.0, new_w))
        new_config["weights"] = new_weights

        # Possibly add or remove 1 indicator (10% chance each)
        if random.random() < 0.10 and len(new_weights) > 5:
            # Remove weakest
            weakest = min(new_weights, key=new_weights.get)
            del new_config["weights"][weakest]

        if random.random() < 0.10:
            # Add a random indicator from available set
            new_ind = self._random_indicator(exclude=set(new_weights.keys()))
            if new_ind:
                new_config["weights"][new_ind] = random.uniform(0.3, 0.7)

        # Mutate thresholds
        entry = new_config.get("entry_threshold", 0.25)
        entry += random.gauss(0, 0.02 * intensity / 0.1)
        new_config["entry_threshold"] = max(0.15, min(0.40, entry))

        exit_t = new_config.get("exit_threshold", -0.10)
        exit_t += random.gauss(0, 0.01 * intensity / 0.1)
        new_config["exit_threshold"] = max(-0.20, min(-0.05, exit_t))

        hold = new_config.get("min_hold_bars", 4)
        if random.random() < 0.2:
            hold += random.choice([-1, 1])
            new_config["min_hold_bars"] = max(2, min(10, hold))

        return new_config

    def _random_indicator(self, exclude: set) -> Optional[str]:
        """Pick a random indicator not in the exclude set."""
        try:
            from coach_system.coaches.ai_coach import AVAILABLE_INDICATORS

            candidates = [i for i in AVAILABLE_INDICATORS if i not in exclude]
            return random.choice(candidates) if candidates else None
        except ImportError:
            return None

    def _save(self):
        """Persist genome to disk."""
        GENOMES_DIR.mkdir(parents=True, exist_ok=True)
        path = GENOMES_DIR / f"{self.player_id}.json"
        try:
            data = {
                "player_id": self.player_id,
                "player_label": self.player_label,
                "generation": self.generation,
                "best_pnl": self.best_pnl if self.best_pnl != float("-inf") else None,
                "best_sharpe": self.best_sharpe if self.best_sharpe != float("-inf") else None,
                "best_config": self.best_config,
                "history": self.history[-50:],  # Keep last 50 generations
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"[Genome] Failed to save {self.player_id}: {e}")

    def _load(self):
        """Load genome from disk if available."""
        path = GENOMES_DIR / f"{self.player_id}.json"
        if not path.exists():
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
            self.generation = data.get("generation", 0)
            self.best_pnl = data.get("best_pnl") or float("-inf")
            self.best_sharpe = data.get("best_sharpe") or float("-inf")
            self.best_config = data.get("best_config")
            self.history = data.get("history", [])
            logger.info(
                f"[Genome] Loaded {self.player_id}: gen {self.generation}, "
                f"best PnL {self.best_pnl:+.0f}"
            )
        except Exception as e:
            logger.warning(f"[Genome] Failed to load {self.player_id}: {e}")


def _config_hash(config: Dict) -> str:
    """Create a short hash of a config for deduplication."""
    import hashlib

    s = json.dumps(config, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]
