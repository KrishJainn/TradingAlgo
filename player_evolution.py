"""
Player Evolution Engine — Genetic Evolution for 5-Player Trading System.

Pure math. No LLM. No Gemini.

Implements:
  1. INDICATOR POOL: 17 indicators across 4 categories (trend, mean-reversion,
     volume/flow, momentum/volatility)
  2. GENETIC EVOLUTION: genome = (indicator_subset, weights, thresholds, hold_period)
     with mutation, crossover, and random exploration
  3. WALK-FORWARD EVOLUTION: triggers every `evolution_interval` trading days
  4. DIVERSITY ENFORCEMENT: Jaccard similarity < 0.70, min 2 indicator categories
  5. SURVIVAL OF THE FITTEST: rolling 120-day Sharpe tracking, negative Sharpe
     for 3 consecutive periods → randomise; Sharpe > 1.5 for 3 periods → elite
  6. LOGGING: genome change log, indicator survival rates, diversity scores

Integration:
    from player_evolution import PlayerEvolutionEngine

    evo = PlayerEvolutionEngine()
    # At each walk-forward boundary:
    new_configs = evo.evolve(
        current_configs=player_configs,
        performance_history=per_player_sharpe_history,
        train_data=train_data,
    )
"""

from __future__ import annotations

import json
import logging
import math
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# 1. INDICATOR POOL — 17 indicators across 4 categories
# ═══════════════════════════════════════════════════════════════════════════

INDICATOR_CATEGORIES: Dict[str, List[str]] = {
    "trend": [
        "PSAR_signal",        # Parabolic SAR direction
        "ICHIMOKU_signal",    # Ichimoku cloud signal
        "DONCHIAN_breakout",  # Donchian channel breakout
        "HA_trend",           # Heikin-Ashi trend
    ],
    "mean_reversion": [
        "RSI_14",             # Relative Strength Index
        "STOCH_divergence",   # Stochastic divergence
        "VWAP_dev_raw",       # VWAP deviation (continuous)
        "BB_squeeze_signal",  # Bollinger Band squeeze
        "MFI_14",             # Money Flow Index
    ],
    "volume_flow": [
        "ELDER_force",        # Elder Force Index
        "OBV_slope",          # On-Balance Volume slope
    ],
    "momentum_volatility": [
        "ATR_ratio",          # ATR ratio (vol expansion)
        "ROC_20",             # Rate of Change 20-day
        "MACD_hist_slope",    # MACD histogram slope
        "HIGH_52W_prox",      # 52-week high proximity
        "NATR",               # Normalised ATR
        "KC_breakout",        # Keltner Channel breakout
    ],
}

# Flat list of all indicators
ALL_INDICATORS: List[str] = []
INDICATOR_TO_CATEGORY: Dict[str, str] = {}
for cat, inds in INDICATOR_CATEGORIES.items():
    for ind in inds:
        ALL_INDICATORS.append(ind)
        INDICATOR_TO_CATEGORY[ind] = cat

# Player archetype labels
PLAYER_ARCHETYPES = {
    "PLAYER_1": "TrendFollower",
    "PLAYER_2": "MeanReversion",
    "PLAYER_3": "VolumeFlow",
    "PLAYER_4": "VolBreakout",
    "PLAYER_5": "MultiMomentum",
}

# Each archetype's preferred categories (for diversity seeding)
ARCHETYPE_PREFERRED: Dict[str, List[str]] = {
    "TrendFollower":  ["trend", "momentum_volatility"],
    "MeanReversion":  ["mean_reversion", "volume_flow"],
    "VolumeFlow":     ["volume_flow", "mean_reversion"],
    "VolBreakout":    ["momentum_volatility", "mean_reversion"],
    "MultiMomentum":  ["momentum_volatility", "trend"],
}


# ═══════════════════════════════════════════════════════════════════════════
# 2. GENOME — the unit of evolution
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PlayerGenome:
    """
    A player's evolved configuration.

    Genome = {
        indicators: list of 4-7 indicator names from the pool
        weights:    dict of indicator → weight in [0.1, 1.0]
        entry_threshold: float in [0.10, 0.45]
        exit_threshold:  float in [-0.25, -0.05]
        min_hold_bars:   int in [2, 10]
    }
    """
    player_id: str
    label: str
    indicators: List[str]
    weights: Dict[str, float]
    entry_threshold: float = 0.25
    exit_threshold: float = -0.12
    min_hold_bars: int = 4

    def to_config(self) -> Dict[str, Any]:
        """Convert genome to player config dict (compatible with existing system)."""
        return {
            "label": self.label,
            "weights": dict(self.weights),
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "min_hold_bars": self.min_hold_bars,
        }

    @classmethod
    def from_config(cls, player_id: str, config: Dict[str, Any]) -> "PlayerGenome":
        """Create genome from existing player config dict."""
        weights = dict(config.get("weights", {}))
        return cls(
            player_id=player_id,
            label=config.get("label", player_id),
            indicators=list(weights.keys()),
            weights=weights,
            entry_threshold=config.get("entry_threshold", 0.25),
            exit_threshold=config.get("exit_threshold", -0.12),
            min_hold_bars=config.get("min_hold_bars", 4),
        )

    def get_categories(self) -> Set[str]:
        """Return set of indicator categories used by this genome."""
        return {INDICATOR_TO_CATEGORY[ind] for ind in self.indicators
                if ind in INDICATOR_TO_CATEGORY}

    def indicator_set(self) -> Set[str]:
        return set(self.indicators)

    def copy(self) -> "PlayerGenome":
        return PlayerGenome(
            player_id=self.player_id,
            label=self.label,
            indicators=list(self.indicators),
            weights=dict(self.weights),
            entry_threshold=self.entry_threshold,
            exit_threshold=self.exit_threshold,
            min_hold_bars=self.min_hold_bars,
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3. PERFORMANCE TRACKER — rolling Sharpe + survival logic
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceTracker:
    """Track rolling performance for survival-of-the-fittest logic."""
    sharpe_history: List[float] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)
    trades_history: List[int] = field(default_factory=list)
    consecutive_negative: int = 0
    consecutive_elite: int = 0
    is_elite: bool = False
    total_randomizations: int = 0

    def record(self, sharpe: float, pnl: float, n_trades: int) -> None:
        """Record one evolution period's results."""
        self.sharpe_history.append(sharpe)
        self.pnl_history.append(pnl)
        self.trades_history.append(n_trades)

        # Consecutive negative Sharpe tracking
        if sharpe < 0:
            self.consecutive_negative += 1
            self.consecutive_elite = 0
            self.is_elite = False
        else:
            self.consecutive_negative = 0

        # Elite detection: Sharpe > 1.5 for 3 consecutive periods
        if sharpe > 1.5:
            self.consecutive_elite += 1
            if self.consecutive_elite >= 3:
                self.is_elite = True
        else:
            self.consecutive_elite = 0
            self.is_elite = False

    @property
    def should_randomize(self) -> bool:
        """True if player has had negative Sharpe for 3 consecutive periods."""
        return self.consecutive_negative >= 3

    @property
    def rolling_sharpe(self) -> float:
        """Average Sharpe over the last 3 periods (or all if fewer)."""
        if not self.sharpe_history:
            return 0.0
        window = self.sharpe_history[-3:]
        return float(np.mean(window))


# ═══════════════════════════════════════════════════════════════════════════
# 4. EVOLUTION ENGINE — mutation, crossover, diversity, survival
# ═══════════════════════════════════════════════════════════════════════════

class PlayerEvolutionEngine:
    """
    Genetic evolution engine for 5-player trading system.

    Pure math evolution — no LLM, no Gemini.

    Evolution cycle (triggered every `evolution_interval` bars):
      1. Evaluate each player's Sharpe on the training window
      2. Rank players by Sharpe
      3. Apply survival rules:
         - Elite (Sharpe > 1.5 for 3 periods): protect, small mutation only
         - Failing (negative Sharpe for 3 periods): randomize genome
         - Normal: mutate + crossover
      4. Crossover: best player's indicators leak into worst player
      5. Enforce diversity (Jaccard < 0.70, min 2 categories)
      6. Log all changes
    """

    def __init__(
        self,
        evolution_interval: int = 120,      # trading days between evolution (was 60)
        max_indicators: int = 7,            # max indicators per player
        min_indicators: int = 4,            # min indicators per player
        max_jaccard: float = 0.70,          # max Jaccard similarity allowed
        min_categories: int = 2,            # min indicator categories per player
        mutation_rate: float = 0.15,        # probability of mutating each weight (was 0.20)
        weight_mutation_sigma: float = 0.05,  # std dev for weight perturbation (was 0.15)
        crossover_fraction: float = 0.25,   # fraction of indicators to cross (was 0.30)
        crossover_sharpe_threshold: float = -0.5,  # only crossover if worst < this
        do_no_harm_delta: float = 0.3,      # revert if in-sample Sharpe drops > this
        seed: int = 42,
        log_dir: Optional[Path] = None,
    ):
        self.evolution_interval = evolution_interval
        self.max_indicators = max_indicators
        self.min_indicators = min_indicators
        self.max_jaccard = max_jaccard
        self.min_categories = min_categories
        self.mutation_rate = mutation_rate
        self.weight_mutation_sigma = weight_mutation_sigma
        self.crossover_fraction = crossover_fraction
        self.crossover_sharpe_threshold = crossover_sharpe_threshold
        self.do_no_harm_delta = do_no_harm_delta
        self.rng = np.random.RandomState(seed)

        self.log_dir = log_dir or Path(__file__).parent / "evolution_logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self.generation: int = 0
        self.trackers: Dict[str, PerformanceTracker] = {
            pid: PerformanceTracker() for pid in PLAYER_ARCHETYPES
        }
        self.evolution_log: List[Dict[str, Any]] = []
        self.indicator_survival: Dict[str, int] = {ind: 0 for ind in ALL_INDICATORS}

    # ── Main evolution entry point ──────────────────────────────────────

    def evolve(
        self,
        current_configs: Dict[str, Dict[str, Any]],
        train_data: Dict[str, pd.DataFrame],
        per_player_sharpe: Optional[Dict[str, float]] = None,
        per_player_pnl: Optional[Dict[str, float]] = None,
        per_player_trades: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run one evolution cycle.

        Args:
            current_configs: Current player configs (from evolved_player_configs.json)
            train_data: Dict of {symbol: DataFrame} for the training window
            per_player_sharpe: Optional dict of player_id → Sharpe from last period
            per_player_pnl: Optional dict of player_id → PnL from last period
            per_player_trades: Optional dict of player_id → trade count

        Returns:
            New player configs dict (same format as evolved_player_configs.json)
        """
        self.generation += 1

        # Convert to genomes
        genomes: Dict[str, PlayerGenome] = {}
        for pid in PLAYER_ARCHETYPES:
            cfg = current_configs.get(pid, {})
            if cfg:
                genomes[pid] = PlayerGenome.from_config(pid, cfg)
            else:
                genomes[pid] = self._random_genome(pid)

        # If no external Sharpe provided, evaluate on train_data
        if per_player_sharpe is None:
            per_player_sharpe = {}
            per_player_pnl = per_player_pnl or {}
            per_player_trades = per_player_trades or {}
            for pid, genome in genomes.items():
                sharpe, n_trades, pnl = self._evaluate_genome(genome, train_data)
                per_player_sharpe[pid] = sharpe
                per_player_pnl[pid] = pnl
                per_player_trades[pid] = n_trades

        per_player_pnl = per_player_pnl or {pid: 0.0 for pid in PLAYER_ARCHETYPES}
        per_player_trades = per_player_trades or {pid: 0 for pid in PLAYER_ARCHETYPES}

        # Record performance
        for pid in PLAYER_ARCHETYPES:
            self.trackers[pid].record(
                sharpe=per_player_sharpe.get(pid, 0.0),
                pnl=per_player_pnl.get(pid, 0.0),
                n_trades=per_player_trades.get(pid, 0),
            )

        # Rank players by Sharpe
        ranked = sorted(PLAYER_ARCHETYPES.keys(),
                        key=lambda p: per_player_sharpe.get(p, 0.0),
                        reverse=True)

        best_pid = ranked[0]
        worst_pid = ranked[-1]

        log_entry = {
            "generation": self.generation,
            "timestamp": datetime.now().isoformat(),
            "pre_evolution_sharpe": dict(per_player_sharpe),
            "pre_evolution_pnl": dict(per_player_pnl),
            "ranking": ranked,
            "actions": {},
        }

        # ── Step 1: Apply survival rules ──────────────────────────────
        #
        # Tiered protection (FIX 1):
        #   - Sharpe > 0.5:  PROTECT — only tiny weight nudges of ±0.02
        #   - Elite (>1.5 x3): even stricter, weight nudge ±0.01
        #   - Negative Sharpe x3:  RANDOMIZE
        #   - Worst AND Sharpe < -0.5:  CROSSOVER from best + small mutate
        #   - Everyone else:  moderate mutation (intensity 0.05, was 0.10)

        new_genomes: Dict[str, PlayerGenome] = {}

        worst_sharpe = per_player_sharpe.get(worst_pid, 0.0)

        for pid in PLAYER_ARCHETYPES:
            tracker = self.trackers[pid]
            genome = genomes[pid]
            player_sharpe = per_player_sharpe.get(pid, 0.0)

            if tracker.is_elite:
                # ELITE: protect — nudge ±0.01 only
                new_genome = self._mutate(genome, intensity=0.01)
                log_entry["actions"][pid] = "elite_protect (nudge ±0.01)"

            elif player_sharpe > 0.5:
                # PERFORMING WELL: protect — nudge ±0.02 only
                new_genome = self._mutate(genome, intensity=0.02)
                log_entry["actions"][pid] = f"protect (Sharpe {player_sharpe:.2f} > 0.5, nudge ±0.02)"

            elif tracker.should_randomize:
                # FAILING: randomize genome completely
                new_genome = self._random_genome(pid)
                tracker.consecutive_negative = 0
                tracker.total_randomizations += 1
                log_entry["actions"][pid] = f"RANDOMIZE (neg Sharpe x{tracker.total_randomizations})"

            elif (pid == worst_pid and best_pid != worst_pid
                  and worst_sharpe < self.crossover_sharpe_threshold):
                # WORST AND SHARPE < -0.5: crossover from best + small mutation
                new_genome = self._crossover(
                    parent_a=genomes[best_pid],
                    parent_b=genome,
                    child_pid=pid,
                )
                new_genome = self._mutate(new_genome, intensity=0.05)
                log_entry["actions"][pid] = (
                    f"crossover({best_pid}→{pid}) + mutate "
                    f"[Sharpe {worst_sharpe:.2f} < {self.crossover_sharpe_threshold}]"
                )

            else:
                # NORMAL: moderate mutation (was 0.10/0.18, now 0.05/0.08)
                if player_sharpe > 0:
                    intensity = 0.03
                elif player_sharpe > -0.5:
                    intensity = 0.05
                else:
                    intensity = 0.08
                new_genome = self._mutate(genome, intensity=intensity)
                log_entry["actions"][pid] = f"mutate(intensity={intensity:.2f}, Sharpe={player_sharpe:.2f})"

            new_genomes[pid] = new_genome

        # ── Step 1b: Do-no-harm check ────────────────────────────────
        # If mutation drops a player's in-sample Sharpe by > do_no_harm_delta,
        # revert to the original genome (keep what was working).
        harm_reverts: List[str] = []
        for pid in PLAYER_ARCHETYPES:
            old_sharpe = per_player_sharpe.get(pid, 0.0)
            if old_sharpe > 0.3:  # only check for players that were OK
                new_sharpe, _, _ = self._evaluate_genome(new_genomes[pid], train_data)
                drop = old_sharpe - new_sharpe
                if drop > self.do_no_harm_delta:
                    new_genomes[pid] = genomes[pid].copy()  # revert
                    harm_reverts.append(
                        f"{pid}: REVERTED (Sharpe would drop {old_sharpe:.2f}→{new_sharpe:.2f}, "
                        f"Δ={drop:.2f} > {self.do_no_harm_delta})"
                    )
                    log_entry["actions"][pid] += " → REVERTED (do-no-harm)"
        if harm_reverts:
            log_entry["do_no_harm_reverts"] = harm_reverts

        # ── Step 2: Enforce diversity ─────────────────────────────────

        diversity_fixes = self._enforce_diversity(new_genomes)
        if diversity_fixes:
            log_entry["diversity_fixes"] = diversity_fixes

        # ── Step 3: Validate genomes ──────────────────────────────────

        for pid, genome in new_genomes.items():
            self._validate_genome(genome)

        # ── Step 4: Update indicator survival counts ──────────────────

        for pid, genome in new_genomes.items():
            for ind in genome.indicators:
                self.indicator_survival[ind] = self.indicator_survival.get(ind, 0) + 1

        # ── Step 5: Compute diversity score ───────────────────────────

        diversity_score = self._compute_diversity_score(new_genomes)
        log_entry["diversity_score"] = diversity_score
        log_entry["indicator_survival"] = dict(self.indicator_survival)

        # Log
        self.evolution_log.append(log_entry)
        self._save_log()

        # Convert back to configs
        result = {}
        for pid, genome in new_genomes.items():
            result[pid] = genome.to_config()

        return result

    # ── Mutation ──────────────────────────────────────────────────────

    def _mutate(self, genome: PlayerGenome, intensity: float = 0.05) -> PlayerGenome:
        """
        Mutate a genome (conservative — FIX 1).

        intensity controls magnitude:
          0.01 = elite nudge (±0.005 weights, no indicator swap)
          0.02 = protect nudge (±0.01 weights, no indicator swap)
          0.03 = gentle (±0.015 weights, 5% indicator swap chance)
          0.05 = moderate (±0.025 weights, 10% swap chance)
          0.08 = aggressive (±0.04 weights, 15% swap chance)
        """
        new = genome.copy()

        # Weight perturbation: perturb 1 weight at low intensity, 2-3 at high
        n_to_perturb = max(1, int(len(new.weights) * self.mutation_rate * intensity / 0.05))
        n_to_perturb = min(n_to_perturb, len(new.weights))
        keys = list(new.weights.keys())
        perturb_keys = self.rng.choice(keys, size=n_to_perturb, replace=False)

        for k in perturb_keys:
            # sigma scales with intensity: intensity=0.02 → sigma≈0.02, intensity=0.08 → sigma≈0.08
            sigma = self.weight_mutation_sigma * intensity / 0.05
            delta = self.rng.normal(0, sigma)
            new.weights[k] = float(np.clip(new.weights[k] + delta, 0.10, 1.0))

        # Indicator swap — only at intensity >= 0.03, probability scales
        swap_prob = max(0.0, (intensity - 0.02) * 2.0)  # 0 at 0.02, 0.06 at 0.05, 0.12 at 0.08
        if self.rng.random() < swap_prob:
            current_set = set(new.indicators)
            available = [ind for ind in ALL_INDICATORS if ind not in current_set]
            if available and len(new.indicators) >= self.min_indicators:
                weakest = min(new.weights, key=new.weights.get)
                new.indicators.remove(weakest)
                del new.weights[weakest]
                new_ind = self.rng.choice(available)
                new.indicators.append(new_ind)
                new.weights[new_ind] = float(self.rng.uniform(0.3, 0.7))

        # Threshold mutation — 5% chance, small delta
        if self.rng.random() < 0.05 and intensity >= 0.03:
            delta_entry = self.rng.normal(0, 0.015)
            new.entry_threshold = float(np.clip(
                new.entry_threshold + delta_entry, 0.10, 0.45
            ))

        if self.rng.random() < 0.05 and intensity >= 0.03:
            delta_exit = self.rng.normal(0, 0.01)
            new.exit_threshold = float(np.clip(
                new.exit_threshold + delta_exit, -0.25, -0.05
            ))

        # Hold period mutation — 5% chance, only at high intensity
        if self.rng.random() < 0.05 and intensity >= 0.05:
            delta_hold = self.rng.choice([-1, 0, 1])
            new.min_hold_bars = int(np.clip(new.min_hold_bars + delta_hold, 2, 10))

        return new

    # ── Crossover ─────────────────────────────────────────────────────

    def _crossover(
        self,
        parent_a: PlayerGenome,
        parent_b: PlayerGenome,
        child_pid: str,
    ) -> PlayerGenome:
        """
        Crossover: best player (parent_a) donates indicators to worst (parent_b).

        Strategy:
          1. Take all of parent_b's indicators
          2. Replace `crossover_fraction` of them with parent_a's best indicators
          3. Keep child's label and player_id
        """
        child = parent_b.copy()
        child.player_id = child_pid
        child.label = PLAYER_ARCHETYPES.get(child_pid, child.label)

        # Sort parent_a's indicators by weight (highest = best)
        a_sorted = sorted(parent_a.weights.items(), key=lambda x: x[1], reverse=True)

        # How many to donate
        n_donate = max(1, int(len(child.indicators) * self.crossover_fraction))

        # Pick top indicators from parent_a that child doesn't have
        donated = 0
        for ind, weight in a_sorted:
            if donated >= n_donate:
                break
            if ind not in child.indicator_set():
                # Remove child's weakest to make room
                if len(child.indicators) >= self.max_indicators:
                    weakest = min(child.weights, key=child.weights.get)
                    child.indicators.remove(weakest)
                    del child.weights[weakest]

                child.indicators.append(ind)
                child.weights[ind] = float(weight * 0.8)  # slight discount
                donated += 1

        # Blend thresholds (weighted toward parent_a since they're better)
        child.entry_threshold = float(
            0.6 * parent_a.entry_threshold + 0.4 * parent_b.entry_threshold
        )
        child.exit_threshold = float(
            0.6 * parent_a.exit_threshold + 0.4 * parent_b.exit_threshold
        )

        return child

    # ── Random genome generation ──────────────────────────────────────

    def _random_genome(self, player_id: str) -> PlayerGenome:
        """
        Generate a random genome respecting archetype preferences.

        Picks 5-6 indicators with at least 2 from preferred categories
        and at least 2 total categories.
        """
        label = PLAYER_ARCHETYPES.get(player_id, player_id)
        preferred = ARCHETYPE_PREFERRED.get(label, ["trend", "momentum_volatility"])

        indicators: List[str] = []
        weights: Dict[str, float] = {}

        # Pick 2-3 from preferred categories
        preferred_pool = []
        for cat in preferred:
            preferred_pool.extend(INDICATOR_CATEGORIES.get(cat, []))
        preferred_pool = list(set(preferred_pool))
        self.rng.shuffle(preferred_pool)

        n_preferred = min(3, len(preferred_pool))
        for ind in preferred_pool[:n_preferred]:
            indicators.append(ind)
            weights[ind] = float(self.rng.uniform(0.4, 0.9))

        # Fill remaining from other categories
        n_total = self.rng.randint(self.min_indicators, self.max_indicators + 1)
        all_remaining = [ind for ind in ALL_INDICATORS if ind not in indicators]
        self.rng.shuffle(all_remaining)

        for ind in all_remaining:
            if len(indicators) >= n_total:
                break
            indicators.append(ind)
            weights[ind] = float(self.rng.uniform(0.2, 0.7))

        # Random thresholds
        entry = float(self.rng.uniform(0.15, 0.35))
        exit_t = float(self.rng.uniform(-0.20, -0.08))
        hold = int(self.rng.randint(3, 7))

        return PlayerGenome(
            player_id=player_id,
            label=label,
            indicators=indicators,
            weights=weights,
            entry_threshold=round(entry, 3),
            exit_threshold=round(exit_t, 3),
            min_hold_bars=hold,
        )

    # ── Diversity enforcement ─────────────────────────────────────────

    def _enforce_diversity(
        self, genomes: Dict[str, PlayerGenome]
    ) -> List[str]:
        """
        Enforce diversity constraints:
          1. No pair of players may have Jaccard similarity > max_jaccard
          2. Each player must use at least min_categories indicator categories

        Returns list of actions taken.
        """
        fixes: List[str] = []
        pids = list(genomes.keys())

        # Pass 1: Fix category diversity
        for pid, genome in genomes.items():
            cats = genome.get_categories()
            if len(cats) < self.min_categories:
                # Find a category not represented
                missing_cats = [c for c in INDICATOR_CATEGORIES if c not in cats]
                if missing_cats:
                    cat_to_add = self.rng.choice(missing_cats)
                    pool = INDICATOR_CATEGORIES[cat_to_add]
                    available = [ind for ind in pool if ind not in genome.indicator_set()]
                    if available:
                        new_ind = self.rng.choice(available)
                        # If at max, remove one from over-represented category
                        if len(genome.indicators) >= self.max_indicators:
                            # Find category with most indicators
                            cat_counts: Dict[str, int] = {}
                            for ind in genome.indicators:
                                c = INDICATOR_TO_CATEGORY.get(ind, "unknown")
                                cat_counts[c] = cat_counts.get(c, 0) + 1
                            biggest_cat = max(cat_counts, key=cat_counts.get)
                            inds_in_biggest = [
                                ind for ind in genome.indicators
                                if INDICATOR_TO_CATEGORY.get(ind) == biggest_cat
                            ]
                            # Remove weakest from biggest category
                            weakest = min(inds_in_biggest, key=lambda x: genome.weights.get(x, 0))
                            genome.indicators.remove(weakest)
                            del genome.weights[weakest]

                        genome.indicators.append(new_ind)
                        genome.weights[new_ind] = float(self.rng.uniform(0.3, 0.6))
                        fixes.append(f"{pid}: added {new_ind} from {cat_to_add} (category diversity)")

        # Pass 2: Fix Jaccard similarity
        max_attempts = 20
        attempt = 0
        while attempt < max_attempts:
            worst_pair = None
            worst_jaccard = 0.0

            for i in range(len(pids)):
                for j in range(i + 1, len(pids)):
                    jac = self._jaccard(
                        genomes[pids[i]].indicator_set(),
                        genomes[pids[j]].indicator_set(),
                    )
                    if jac > worst_jaccard:
                        worst_jaccard = jac
                        worst_pair = (pids[i], pids[j])

            if worst_jaccard <= self.max_jaccard:
                break  # All pairs are diverse enough

            # Fix: swap an indicator in the worse-performing player of the pair
            pid_a, pid_b = worst_pair
            sharpe_a = self.trackers[pid_a].rolling_sharpe
            sharpe_b = self.trackers[pid_b].rolling_sharpe

            # Modify the worse player
            target_pid = pid_b if sharpe_a >= sharpe_b else pid_a
            target = genomes[target_pid]

            # Find shared indicators
            other_pid = pid_a if target_pid == pid_b else pid_b
            shared = target.indicator_set() & genomes[other_pid].indicator_set()

            if shared:
                # Remove one shared indicator (weakest weight)
                shared_list = list(shared)
                to_remove = min(shared_list, key=lambda x: target.weights.get(x, 0))
                target.indicators.remove(to_remove)
                del target.weights[to_remove]

                # Add a non-shared indicator
                all_used = set()
                for g in genomes.values():
                    all_used |= g.indicator_set()
                candidates = [ind for ind in ALL_INDICATORS
                              if ind not in target.indicator_set()
                              and ind not in genomes[other_pid].indicator_set()]
                if not candidates:
                    candidates = [ind for ind in ALL_INDICATORS
                                  if ind not in target.indicator_set()]

                if candidates:
                    new_ind = self.rng.choice(candidates)
                    target.indicators.append(new_ind)
                    target.weights[new_ind] = float(self.rng.uniform(0.3, 0.7))
                    fixes.append(
                        f"{target_pid}: swapped {to_remove}→{new_ind} "
                        f"(Jaccard {worst_jaccard:.2f} with {other_pid})"
                    )

            attempt += 1

        return fixes

    # ── Evaluation ────────────────────────────────────────────────────

    def _evaluate_genome(
        self,
        genome: PlayerGenome,
        data: Dict[str, pd.DataFrame],
    ) -> Tuple[float, int, float]:
        """
        Evaluate a genome on data. Returns (sharpe, n_trades, total_pnl).

        Uses the same signal computation as _compute_player_signal in
        walk_forward_backtest.py.
        """
        try:
            from data_cache.indicator_computer import IndicatorComputer
        except ImportError:
            return 0.0, 0, 0.0

        ic = IndicatorComputer()
        daily_returns: List[float] = []
        trade_pnls: List[float] = []

        # Column name resolution (matches _CONFIG_TO_FALLBACK in walk_forward_backtest)
        _FALLBACK = {
            "RSI_14": "RSI_14_norm",
            "MFI_14": "MFI_14_norm",
            "ATR_ratio": "ATR_ratio_norm",
            "NATR": "NATR_norm",
            "BB_width_raw": "BB_width_raw_norm",
            # New custom indicators map to themselves
        }

        for symbol, df in data.items():
            if len(df) < 40:
                continue

            try:
                indicators = ic.compute_all_indicators(df)
            except Exception:
                continue

            close_col = "Close" if "Close" in df.columns else "close"
            position = None  # (entry_price, direction, bars_held)

            for idx in range(30, len(indicators)):
                if indicators.index[idx] not in df.index:
                    continue

                price = float(df.loc[indicators.index[idx], close_col])

                # Compute weighted signal (same logic as _compute_player_signal)
                weighted_sum = 0.0
                total_weight = 0.0

                for ind_name, weight in genome.weights.items():
                    if weight == 0:
                        continue

                    # Resolve column
                    col = ind_name + "_norm"
                    if col not in indicators.columns:
                        col = _FALLBACK.get(ind_name, ind_name)
                    if col not in indicators.columns:
                        continue

                    val = indicators.iloc[idx][col]
                    if pd.isna(val):
                        continue

                    weighted_sum += val * weight
                    total_weight += abs(weight)

                if total_weight == 0:
                    signal = 0.0
                else:
                    signal = float(np.clip(weighted_sum / total_weight, -1.0, 1.0))

                # Position management
                if position is None:
                    if abs(signal) >= genome.entry_threshold:
                        direction = "LONG" if signal > 0 else "SHORT"
                        position = (price, direction, 0)
                else:
                    entry_price, direction, bars_held = position
                    bars_held += 1
                    position = (entry_price, direction, bars_held)

                    exit_now = False
                    if bars_held >= genome.min_hold_bars:
                        if direction == "LONG" and signal < genome.exit_threshold:
                            exit_now = True
                        elif direction == "SHORT" and signal > -genome.exit_threshold:
                            exit_now = True
                        elif bars_held >= 15:
                            exit_now = True

                    if idx == len(indicators) - 1:
                        exit_now = True

                    if exit_now:
                        if direction == "LONG":
                            pnl = price - entry_price
                        else:
                            pnl = entry_price - price
                        trade_pnls.append(pnl)
                        daily_returns.append(pnl / entry_price)
                        position = None

        if len(trade_pnls) < 3:
            return 0.0, len(trade_pnls), sum(trade_pnls)

        returns = np.array(daily_returns)
        mean_r = np.mean(returns)
        std_r = np.std(returns, ddof=1)
        sharpe = float(mean_r / std_r * np.sqrt(252)) if std_r > 1e-10 else 0.0

        return sharpe, len(trade_pnls), float(sum(trade_pnls))

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _jaccard(set_a: Set[str], set_b: Set[str]) -> float:
        """Jaccard similarity: |A ∩ B| / |A ∪ B|"""
        if not set_a and not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union > 0 else 0.0

    def _validate_genome(self, genome: PlayerGenome) -> None:
        """Ensure genome is within bounds."""
        # Indicator count bounds
        if len(genome.indicators) > self.max_indicators:
            # Remove weakest excess indicators
            sorted_inds = sorted(genome.weights.items(), key=lambda x: x[1])
            while len(genome.indicators) > self.max_indicators:
                weakest = sorted_inds.pop(0)[0]
                genome.indicators.remove(weakest)
                del genome.weights[weakest]

        if len(genome.indicators) < self.min_indicators:
            # Add random indicators
            available = [ind for ind in ALL_INDICATORS if ind not in genome.indicator_set()]
            while len(genome.indicators) < self.min_indicators and available:
                ind = self.rng.choice(available)
                genome.indicators.append(ind)
                genome.weights[ind] = float(self.rng.uniform(0.3, 0.6))
                available.remove(ind)

        # Weight bounds
        for ind in list(genome.weights.keys()):
            genome.weights[ind] = float(np.clip(genome.weights[ind], 0.10, 1.0))

        # Threshold bounds
        genome.entry_threshold = float(np.clip(genome.entry_threshold, 0.10, 0.45))
        genome.exit_threshold = float(np.clip(genome.exit_threshold, -0.25, -0.05))
        genome.min_hold_bars = int(np.clip(genome.min_hold_bars, 2, 10))

        # Ensure indicators list matches weights keys
        genome.indicators = list(genome.weights.keys())

    def _compute_diversity_score(
        self, genomes: Dict[str, PlayerGenome]
    ) -> Dict[str, Any]:
        """
        Compute diversity metrics across all players.

        Returns:
            {
                "avg_jaccard": float,
                "max_jaccard": float,
                "worst_pair": (str, str),
                "categories_per_player": dict,
                "unique_indicators": int,
            }
        """
        pids = list(genomes.keys())
        jaccards = []
        worst_jac = 0.0
        worst_pair = ("", "")

        for i in range(len(pids)):
            for j in range(i + 1, len(pids)):
                jac = self._jaccard(
                    genomes[pids[i]].indicator_set(),
                    genomes[pids[j]].indicator_set(),
                )
                jaccards.append(jac)
                if jac > worst_jac:
                    worst_jac = jac
                    worst_pair = (pids[i], pids[j])

        all_indicators_used = set()
        cats_per_player = {}
        for pid, genome in genomes.items():
            all_indicators_used |= genome.indicator_set()
            cats_per_player[pid] = len(genome.get_categories())

        return {
            "avg_jaccard": round(float(np.mean(jaccards)) if jaccards else 0.0, 3),
            "max_jaccard": round(worst_jac, 3),
            "worst_pair": worst_pair,
            "categories_per_player": cats_per_player,
            "unique_indicators": len(all_indicators_used),
        }

    def _save_log(self) -> None:
        """Save evolution log to disk."""
        try:
            log_path = self.log_dir / "evolution_history.json"
            with open(log_path, "w") as f:
                json.dump(self.evolution_log, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save evolution log: {e}")

    def get_report(self) -> str:
        """Generate a human-readable evolution report."""
        lines = [
            "=" * 70,
            "  PLAYER EVOLUTION REPORT",
            f"  Generations: {self.generation}",
            "=" * 70,
        ]

        # Per-player summary
        for pid, label in PLAYER_ARCHETYPES.items():
            tracker = self.trackers[pid]
            lines.append(f"\n  {pid} ({label}):")
            if tracker.sharpe_history:
                lines.append(f"    Sharpe history: {[round(s, 2) for s in tracker.sharpe_history[-5:]]}")
                lines.append(f"    Rolling Sharpe: {tracker.rolling_sharpe:.3f}")
                lines.append(f"    Elite: {tracker.is_elite}")
                lines.append(f"    Consecutive negative: {tracker.consecutive_negative}")
                lines.append(f"    Total randomizations: {tracker.total_randomizations}")
            else:
                lines.append(f"    (no history yet)")

        # Indicator survival
        lines.append(f"\n{'─' * 70}")
        lines.append("  INDICATOR SURVIVAL RATES")
        lines.append(f"{'─' * 70}")
        max_count = max(self.indicator_survival.values()) if any(self.indicator_survival.values()) else 1
        sorted_survival = sorted(
            self.indicator_survival.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for ind, count in sorted_survival:
            bar = "█" * int(30 * count / max_count) if max_count > 0 else ""
            pct = count / (self.generation * 5) * 100 if self.generation > 0 else 0
            lines.append(f"    {ind:25s}  {count:>4d} uses  ({pct:5.1f}%)  {bar}")

        # Diversity over time
        if self.evolution_log:
            last = self.evolution_log[-1]
            ds = last.get("diversity_score", {})
            lines.append(f"\n{'─' * 70}")
            lines.append("  CURRENT DIVERSITY")
            lines.append(f"{'─' * 70}")
            lines.append(f"    Avg Jaccard:  {ds.get('avg_jaccard', 'N/A')}")
            lines.append(f"    Max Jaccard:  {ds.get('max_jaccard', 'N/A')}")
            lines.append(f"    Worst pair:   {ds.get('worst_pair', 'N/A')}")
            lines.append(f"    Unique inds:  {ds.get('unique_indicators', 'N/A')}")
            lines.append(f"    Categories:   {ds.get('categories_per_player', {})}")

        lines.append(f"\n{'=' * 70}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 5. WALK-FORWARD EVOLUTION WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

def run_evolution_backtest(
    stock_data: Dict[str, pd.DataFrame],
    training_window: int = 120,
    testing_window: int = 60,
    step_size: int = 60,
    evolve_at_each_window: bool = True,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Run walk-forward backtest WITH genetic evolution at each window boundary.

    Returns:
        (static_results, evolved_results, ensemble_results) — all as dicts.
    """
    try:
        from data_cache.indicator_computer import IndicatorComputer
    except ImportError:
        print("ERROR: Cannot import IndicatorComputer")
        return {}, {}

    # Build date index
    all_dates = set()
    for df in stock_data.values():
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        all_dates.update(df.index)
    sorted_dates = sorted(all_dates)
    total_days = len(sorted_dates)

    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  EVOLUTION WALK-FORWARD BACKTEST")
        print(f"{'=' * 70}")
        print(f"  Date range: {sorted_dates[0].strftime('%Y-%m-%d')} → "
              f"{sorted_dates[-1].strftime('%Y-%m-%d')}")
        print(f"  Total days: {total_days}")
        print(f"  Train window: {training_window}")
        print(f"  Test window: {testing_window}")
        print(f"  Step size: {step_size}")
        print(f"  Evolution: {'ON' if evolve_at_each_window else 'OFF'}")
        print(f"{'=' * 70}\n")

    # Generate windows
    windows = []
    start = 0
    while True:
        train_start = start
        train_end = train_start + training_window
        test_start = train_end
        test_end = test_start + testing_window

        if test_end > total_days:
            if test_start < total_days and (total_days - test_start) >= 10:
                test_end = total_days
            else:
                break

        windows.append((train_start, train_end, test_start, test_end))
        start += step_size

    if verbose:
        print(f"  Generated {len(windows)} windows\n")

    # Load initial static configs
    cfg_path = Path(__file__).parent / "evolved_player_configs.json"
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        static_configs = raw.get("configs", {})
    else:
        static_configs = {}

    # Three parallel runs: static vs evolved vs evolved+ensemble
    evo_engine_1 = PlayerEvolutionEngine(
        evolution_interval=testing_window,
        seed=42,
    )
    evo_engine_2 = PlayerEvolutionEngine(
        evolution_interval=testing_window,
        seed=42,
    )
    evolved_configs_1 = deepcopy(static_configs)
    evolved_configs_2 = deepcopy(static_configs)

    static_results = _run_with_configs(
        "STATIC", static_configs, stock_data, sorted_dates, windows,
        evo_engine=None, ensemble_mode=False, verbose=verbose,
    )

    evolved_results = _run_with_configs(
        "EVOLVED", evolved_configs_1, stock_data, sorted_dates, windows,
        evo_engine=evo_engine_1, ensemble_mode=False, verbose=verbose,
    )

    ensemble_results = _run_with_configs(
        "EVOLVED+ENSEMBLE", evolved_configs_2, stock_data, sorted_dates, windows,
        evo_engine=evo_engine_2, ensemble_mode=True, verbose=verbose,
    )

    # Print evolution report (from the ensemble engine — same evolution)
    if verbose:
        print(evo_engine_2.get_report())

    return static_results, evolved_results, ensemble_results


def _compute_player_signal_standalone(
    indicators: pd.DataFrame,
    weights: Dict[str, float],
    idx: int,
) -> float:
    """
    Compute a single player's weighted signal at bar index `idx`.
    Standalone version that does NOT depend on walk_forward_backtest import.
    Returns signal in [-1, 1].
    """
    _FALLBACK: Dict[str, str] = {
        "RSI_14": "RSI_14_norm", "MFI_14": "MFI_14_norm",
        "ATR_ratio": "ATR_ratio_norm", "NATR": "NATR_norm",
        "BB_width_raw": "BB_width_raw_norm",
    }

    weighted_sum = 0.0
    total_weight = 0.0

    for ind_name, weight in weights.items():
        if weight == 0:
            continue
        col = ind_name + "_norm"
        if col not in indicators.columns:
            col = _FALLBACK.get(ind_name, ind_name)
        if col not in indicators.columns:
            continue
        val = indicators.iloc[idx][col]
        if pd.isna(val):
            continue
        weighted_sum += val * weight
        total_weight += abs(weight)

    if total_weight == 0:
        return 0.0
    return float(np.clip(weighted_sum / total_weight, -1.0, 1.0))


def _run_with_configs(
    label: str,
    configs: Dict[str, Dict[str, Any]],
    stock_data: Dict[str, pd.DataFrame],
    sorted_dates: list,
    windows: List[Tuple[int, int, int, int]],
    evo_engine: Optional[PlayerEvolutionEngine] = None,
    ensemble_mode: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run walk-forward with given configs (optionally evolving).

    IMPORTANT: Each player trades INDEPENDENTLY — every player maintains its
    own positions per symbol, generates its own signals using its own weights,
    and enters/exits based on its own thresholds.

    FIX 2 — Volatility override: when realized vol > 1.5× average,
    reduce ALL player sizing to 0.3× (regardless of regime label).

    FIX 3 — Ensemble majority filter (when ensemble_mode=True):
    At exit, adjust PnL by consensus multiplier:
      - 1 player holds:  0.5× sizing
      - 2 players agree: 1.0× (standard)
      - 3+ agree:        1.2× (conviction boost)

    Position sizing: Rs 10,000 base per player (5 players × Rs 10k = Rs 50k total).
    """
    try:
        from data_cache.indicator_computer import IndicatorComputer
    except ImportError:
        return {}

    ic = IndicatorComputer()
    current_configs = deepcopy(configs)

    all_trades: List[Dict[str, Any]] = []
    all_daily_returns: List[float] = []
    per_player_pnl: Dict[str, float] = {pid: 0.0 for pid in PLAYER_ARCHETYPES}
    per_player_trades: Dict[str, int] = {pid: 0 for pid in PLAYER_ARCHETYPES}
    per_player_wins: Dict[str, int] = {pid: 0 for pid in PLAYER_ARCHETYPES}
    per_regime_pnl: Dict[str, float] = {"Bull": 0.0, "Bear": 0.0, "Sideways": 0.0, "HighVol": 0.0}
    per_regime_trades: Dict[str, int] = {"Bull": 0, "Bear": 0, "Sideways": 0, "HighVol": 0}
    window_results: List[Dict[str, Any]] = []

    BASE_PER_PLAYER = 10_000  # Rs 10k per player, 5 players = Rs 50k total

    if verbose:
        print(f"\n{'─' * 50}")
        print(f"  Running: {label}")
        print(f"{'─' * 50}")

    for w_idx, (train_s, train_e, test_s, test_e) in enumerate(windows):
        train_dates = sorted_dates[train_s:train_e]
        test_dates = sorted_dates[test_s:test_e]

        if not train_dates or not test_dates:
            continue

        # Slice data
        train_data: Dict[str, pd.DataFrame] = {}
        test_data: Dict[str, pd.DataFrame] = {}
        for sym, df in stock_data.items():
            t = df[(df.index >= train_dates[0]) & (df.index <= train_dates[-1])]
            if len(t) > 20:
                train_data[sym] = t
            ts = df[(df.index >= test_dates[0]) & (df.index <= test_dates[-1])]
            if len(ts) > 5:
                test_data[sym] = ts

        # Evolution step (before test window)
        if evo_engine is not None and w_idx > 0:
            prev_trades = window_results[-1].get("trades", []) if window_results else []
            prev_sharpe: Dict[str, float] = {}
            prev_pnl_dict: Dict[str, float] = {}
            prev_trade_count: Dict[str, int] = {}

            for pid in PLAYER_ARCHETYPES:
                pid_trades = [t for t in prev_trades if t.get("player_id") == pid]
                if len(pid_trades) >= 2:
                    rets = np.array([t["pnl"] / max(t["entry_price"], 1) for t in pid_trades])
                    m = np.mean(rets)
                    s = np.std(rets, ddof=1)
                    prev_sharpe[pid] = float(m / s * np.sqrt(252)) if s > 1e-10 else 0.0
                else:
                    prev_sharpe[pid] = 0.0
                prev_pnl_dict[pid] = sum(t["pnl"] for t in pid_trades)
                prev_trade_count[pid] = len(pid_trades)

            current_configs = evo_engine.evolve(
                current_configs=current_configs,
                train_data=train_data,
                per_player_sharpe=prev_sharpe,
                per_player_pnl=prev_pnl_dict,
                per_player_trades=prev_trade_count,
            )

        # Pre-compute indicators for all test symbols
        symbol_indicators: Dict[str, pd.DataFrame] = {}
        for sym, test_df in test_data.items():
            warmup_len = 60
            if sym in train_data:
                warmup = train_data[sym].tail(warmup_len)
                combined = pd.concat([warmup, test_df])
                combined = combined[~combined.index.duplicated(keep="last")]
            else:
                combined = test_df

            if len(combined) < 40:
                continue
            try:
                inds = ic.compute_all_indicators(combined)
                symbol_indicators[sym] = inds
            except Exception:
                continue

        if not symbol_indicators:
            continue

        # ── FIX 2: Compute realized volatility for the training window ──
        # Use average 20-day realized vol across symbols, compare to
        # full-sample average to detect high-vol regimes.
        avg_train_vol = 0.0
        vol_count = 0
        for sym_v, df_v in train_data.items():
            close_col_v = "Close" if "Close" in df_v.columns else "close"
            if close_col_v in df_v.columns and len(df_v) >= 20:
                rets_v = df_v[close_col_v].pct_change().dropna()
                if len(rets_v) >= 20:
                    avg_train_vol += rets_v.tail(20).std() * np.sqrt(252)
                    vol_count += 1
        avg_train_vol = avg_train_vol / max(vol_count, 1)

        # Compare to longer-term vol (full training window)
        long_vol = 0.0
        lv_count = 0
        for sym_v, df_v in train_data.items():
            close_col_v = "Close" if "Close" in df_v.columns else "close"
            if close_col_v in df_v.columns and len(df_v) >= 40:
                rets_v = df_v[close_col_v].pct_change().dropna()
                long_vol += rets_v.std() * np.sqrt(252)
                lv_count += 1
        long_vol = long_vol / max(lv_count, 1)

        # High-vol override: if recent vol > 1.5× long-term, cap sizing at 0.3×
        high_vol = avg_train_vol > 1.5 * long_vol if long_vol > 0 else False
        vol_sizing = 0.3 if high_vol else 1.0
        regime_label = "HighVol" if high_vol else "Normal"

        # ── Per-player independent positions ──
        # positions[pid][sym] = None or (entry_date, entry_price, direction, bars_held)
        positions: Dict[str, Dict[str, Optional[Tuple]]] = {
            pid: {sym: None for sym in symbol_indicators}
            for pid in PLAYER_ARCHETYPES
        }

        window_trades: List[Dict[str, Any]] = []
        window_daily_pnl: List[float] = []

        test_date_list = sorted(set(d for df in test_data.values() for d in df.index))

        for bar_date in test_date_list:
            daily_pnl = 0.0

            # Increment bars held for all open positions
            for pid in PLAYER_ARCHETYPES:
                for sym in list(positions[pid].keys()):
                    pos = positions[pid].get(sym)
                    if pos is not None:
                        entry_date, entry_price, direction, bars_held = pos
                        positions[pid][sym] = (entry_date, entry_price, direction, bars_held + 1)

            for sym in list(symbol_indicators.keys()):
                indicators = symbol_indicators[sym]
                if bar_date not in indicators.index:
                    continue

                bar_idx = indicators.index.get_loc(bar_date)
                if isinstance(bar_idx, slice):
                    bar_idx = bar_idx.start

                test_df = test_data.get(sym)
                if test_df is None or bar_date not in test_df.index:
                    continue
                close_col = "Close" if "Close" in test_df.columns else "close"
                price = float(test_df.loc[bar_date, close_col])

                # ── FIX 3: Compute per-player signals for ensemble filter ──
                player_signals_for_sym: Dict[str, float] = {}

                # Each player trades independently
                for pid in PLAYER_ARCHETYPES:
                    cfg = current_configs.get(pid, {})
                    weights = cfg.get("weights", {})
                    if not weights:
                        continue

                    signal = _compute_player_signal_standalone(indicators, weights, bar_idx)
                    player_signals_for_sym[pid] = signal

                    pos = positions[pid].get(sym)
                    entry_threshold = cfg.get("entry_threshold", 0.25)
                    exit_threshold = cfg.get("exit_threshold", -0.12)
                    min_hold = cfg.get("min_hold_bars", 4)

                    if pos is None:
                        # ENTRY: player's own signal vs player's own threshold
                        if abs(signal) >= entry_threshold:
                            direction = "LONG" if signal > 0 else "SHORT"
                            positions[pid][sym] = (
                                str(bar_date.date()) if hasattr(bar_date, 'date') else str(bar_date),
                                price, direction, 0,
                            )
                    else:
                        # EXIT check
                        entry_date, entry_price, direction, bars_held = pos
                        exit_now = False
                        exit_reason = ""

                        if bars_held >= min_hold:
                            if direction == "LONG" and signal < exit_threshold:
                                exit_now = True
                                exit_reason = "signal_reversal"
                            elif direction == "SHORT" and signal > -exit_threshold:
                                exit_now = True
                                exit_reason = "signal_reversal"
                            elif bars_held >= 15:
                                exit_now = True
                                exit_reason = "timeout"

                        if bar_date == test_date_list[-1] and not exit_now:
                            exit_now = True
                            exit_reason = "window_end"

                        if exit_now:
                            if direction == "LONG":
                                pnl_per_unit = price - entry_price
                            else:
                                pnl_per_unit = entry_price - price

                            # ── FIX 3: Ensemble consensus multiplier ──
                            consensus_mult = 1.0
                            if ensemble_mode:
                                dir_sign = 1.0 if direction == "LONG" else -1.0
                                n_agreeing = 0
                                for other_pid, other_sig in player_signals_for_sym.items():
                                    # Check if other player also has position in same direction
                                    other_pos = positions[other_pid].get(sym)
                                    if other_pos is not None:
                                        _, _, other_dir, _ = other_pos
                                        if other_dir == direction:
                                            n_agreeing += 1
                                    elif other_sig * dir_sign > 0.1:
                                        # Or signal agrees even without position
                                        n_agreeing += 1

                                if n_agreeing <= 1:
                                    consensus_mult = 0.5   # lonely conviction → halve
                                elif n_agreeing >= 3:
                                    consensus_mult = 1.2   # strong consensus → boost
                                # else 2 agreeing → 1.0× standard

                            # Apply vol override sizing (FIX 2)
                            effective_sizing = vol_sizing * consensus_mult
                            quantity = max(1, int(BASE_PER_PLAYER * effective_sizing / entry_price))
                            gross_pnl = pnl_per_unit * quantity

                            trade_rec = {
                                "symbol": sym,
                                "direction": direction,
                                "entry_date": entry_date,
                                "exit_date": str(bar_date.date()) if hasattr(bar_date, 'date') else str(bar_date),
                                "entry_price": entry_price,
                                "exit_price": price,
                                "pnl": gross_pnl,
                                "player_id": pid,
                                "bars_held": bars_held,
                                "exit_reason": exit_reason,
                                "consensus_mult": consensus_mult,
                                "vol_sizing": vol_sizing,
                                "regime": regime_label,
                            }
                            window_trades.append(trade_rec)
                            all_trades.append(trade_rec)

                            per_player_pnl[pid] = per_player_pnl.get(pid, 0.0) + gross_pnl
                            per_player_trades[pid] = per_player_trades.get(pid, 0) + 1
                            if gross_pnl > 0:
                                per_player_wins[pid] = per_player_wins.get(pid, 0) + 1

                            # Track regime performance
                            r_label = regime_label if high_vol else "Normal"
                            per_regime_pnl[r_label] = per_regime_pnl.get(r_label, 0.0) + gross_pnl
                            per_regime_trades[r_label] = per_regime_trades.get(r_label, 0) + 1

                            daily_pnl += gross_pnl
                            positions[pid][sym] = None

            window_daily_pnl.append(daily_pnl / 500_000)
            all_daily_returns.append(daily_pnl / 500_000)

        # Window result
        w_pnl = sum(t["pnl"] for t in window_trades)
        w_rets = np.array(window_daily_pnl) if window_daily_pnl else np.array([0.0])
        w_mean = np.mean(w_rets)
        w_std = np.std(w_rets, ddof=1) if len(w_rets) > 1 else 1e-10
        w_sharpe = float(w_mean / w_std * np.sqrt(252)) if w_std > 1e-10 else 0.0

        w_result = {
            "window": w_idx,
            "train_start": sorted_dates[train_s].strftime("%Y-%m-%d"),
            "test_start": sorted_dates[test_s].strftime("%Y-%m-%d"),
            "test_end": sorted_dates[min(test_e - 1, len(sorted_dates) - 1)].strftime("%Y-%m-%d"),
            "n_trades": len(window_trades),
            "pnl": round(w_pnl, 2),
            "sharpe": round(w_sharpe, 3),
            "trades": window_trades,
        }
        window_results.append(w_result)

        if verbose:
            # Per-player breakdown for this window
            w_player_counts = {}
            for t in window_trades:
                p = t["player_id"]
                w_player_counts[p] = w_player_counts.get(p, 0) + 1
            player_str = "  ".join(
                f"{PLAYER_ARCHETYPES.get(p, p)[:4]}={c}"
                for p, c in sorted(w_player_counts.items())
            )
            print(f"  Window {w_idx:>2d}  "
                  f"test={w_result['test_start']}→{w_result['test_end']}  "
                  f"trades={len(window_trades):>3d}  "
                  f"PnL={w_pnl:>10,.0f}  "
                  f"Sharpe={w_sharpe:>6.3f}  "
                  f"[{player_str}]")

    # Aggregate
    total_pnl = sum(t["pnl"] for t in all_trades)
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t["pnl"] > 0)

    rets = np.array(all_daily_returns) if all_daily_returns else np.array([0.0])
    overall_mean = np.mean(rets)
    overall_std = np.std(rets, ddof=1) if len(rets) > 1 else 1e-10
    overall_sharpe = float(overall_mean / overall_std * np.sqrt(252)) if overall_std > 1e-10 else 0.0

    per_player = {}
    for pid in PLAYER_ARCHETYPES:
        nt = per_player_trades.get(pid, 0)
        nw = per_player_wins.get(pid, 0)
        per_player[pid] = {
            "label": PLAYER_ARCHETYPES[pid],
            "pnl": round(per_player_pnl.get(pid, 0.0), 2),
            "trades": nt,
            "win_rate": round(nw / nt, 3) if nt > 0 else 0.0,
        }

    # Max drawdown
    max_dd = 0.0
    if all_daily_returns:
        cum = np.cumprod(1.0 + np.array(all_daily_returns))
        running_max = np.maximum.accumulate(cum)
        dd = (cum - running_max) / running_max
        max_dd = float(np.min(dd)) * 100  # as percentage

    # Regime performance
    regime_perf = {}
    for r_label in per_regime_pnl:
        rt = per_regime_trades.get(r_label, 0)
        if rt > 0:
            regime_perf[r_label] = {
                "pnl": round(per_regime_pnl[r_label], 2),
                "trades": rt,
            }

    result = {
        "label": label,
        "total_pnl": round(total_pnl, 2),
        "sharpe": round(overall_sharpe, 3),
        "max_drawdown_pct": round(max_dd, 2),
        "n_trades": total_trades,
        "win_rate": round(total_wins / total_trades, 3) if total_trades > 0 else 0.0,
        "per_player": per_player,
        "per_regime": regime_perf,
        "per_window": [{k: v for k, v in w.items() if k != "trades"} for w in window_results],
    }

    if verbose:
        print(f"\n  {label} SUMMARY:")
        print(f"    Total PnL:  Rs {total_pnl:>12,.0f}")
        print(f"    Sharpe:     {overall_sharpe:>8.3f}")
        print(f"    Max DD:     {max_dd:>8.2f}%")
        print(f"    Trades:     {total_trades:>8d}")
        print(f"    Win Rate:   {result['win_rate']:>8.1%}")
        for pid, pp in per_player.items():
            print(f"    {pp['label']:18s}  PnL={pp['pnl']:>10,.0f}  "
                  f"trades={pp['trades']:>3d}  WR={pp['win_rate']:.1%}")
        if regime_perf:
            print(f"    Regime breakdown:")
            for r_label, rp in regime_perf.items():
                print(f"      {r_label:10s}  PnL={rp['pnl']:>10,.0f}  trades={rp['trades']:>4d}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. COMPARISON PRINTER
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison(
    static: Dict[str, Any],
    evolved: Dict[str, Any],
    ensemble: Optional[Dict[str, Any]] = None,
) -> None:
    """Print a 3-way comparison table: Static vs Evolved vs Evolved+Ensemble."""
    has_ens = ensemble is not None and ensemble

    print(f"\n{'=' * 100}")
    print(f"  COMPARISON: STATIC vs EVOLVED vs EVOLVED+ENSEMBLE")
    print(f"{'=' * 100}")

    if has_ens:
        print(f"  {'Metric':<22s}  {'STATIC':>14s}  {'EVOLVED':>14s}  {'EVO+ENSEMBLE':>14s}  {'Best':>6s}")
    else:
        print(f"  {'Metric':<22s}  {'STATIC':>14s}  {'EVOLVED':>14s}")
    print(f"  {'─' * 90}")

    metrics = [
        ("Total PnL (Rs)", "total_pnl", ",.0f"),
        ("Sharpe Ratio", "sharpe", ".3f"),
        ("Max Drawdown (%)", "max_drawdown_pct", ".2f"),
        ("Total Trades", "n_trades", "d"),
        ("Win Rate", "win_rate", ".1%"),
    ]

    for label_m, key, fmt in metrics:
        s_val = static.get(key, 0)
        e_val = evolved.get(key, 0)
        en_val = ensemble.get(key, 0) if has_ens else 0

        s_str = f"{s_val:{fmt}}"
        e_str = f"{e_val:{fmt}}"
        en_str = f"{en_val:{fmt}}" if has_ens else ""

        # Find best (higher is better for PnL, Sharpe, WR; lower for DD, trades)
        best = ""
        if has_ens:
            vals = {"S": s_val, "E": e_val, "E+": en_val}
            if key == "max_drawdown_pct":
                best = max(vals, key=vals.get)  # least negative = best
            elif key == "n_trades":
                best = ""  # no "best" for trade count
            else:
                best = max(vals, key=vals.get)

        if has_ens:
            print(f"  {label_m:<22s}  {s_str:>14s}  {e_str:>14s}  {en_str:>14s}  {best:>6s}")
        else:
            print(f"  {label_m:<22s}  {s_str:>14s}  {e_str:>14s}")

    # Per-player comparison
    print(f"\n  {'─' * 90}")
    print(f"  PER-PLAYER COMPARISON")
    print(f"  {'─' * 90}")
    if has_ens:
        print(f"  {'Player':<18s}  {'Static PnL':>11s}  {'Evo PnL':>11s}  {'Ens PnL':>11s}  "
              f"{'S WR':>6s}  {'E WR':>6s}  {'En WR':>6s}")
    else:
        print(f"  {'Player':<18s}  {'Static PnL':>11s}  {'Evo PnL':>11s}  "
              f"{'S WR':>6s}  {'E WR':>6s}")
    print(f"  {'─' * 90}")

    for pid in PLAYER_ARCHETYPES:
        s_pp = static.get("per_player", {}).get(pid, {})
        e_pp = evolved.get("per_player", {}).get(pid, {})
        en_pp = ensemble.get("per_player", {}).get(pid, {}) if has_ens else {}
        lbl = PLAYER_ARCHETYPES[pid]
        s_pnl = s_pp.get("pnl", 0)
        e_pnl = e_pp.get("pnl", 0)
        en_pnl = en_pp.get("pnl", 0)
        s_wr = s_pp.get("win_rate", 0)
        e_wr = e_pp.get("win_rate", 0)
        en_wr = en_pp.get("win_rate", 0)

        if has_ens:
            print(f"  {lbl:<18s}  {s_pnl:>11,.0f}  {e_pnl:>11,.0f}  {en_pnl:>11,.0f}  "
                  f"{s_wr:>6.1%}  {e_wr:>6.1%}  {en_wr:>6.1%}")
        else:
            print(f"  {lbl:<18s}  {s_pnl:>11,.0f}  {e_pnl:>11,.0f}  "
                  f"{s_wr:>6.1%}  {e_wr:>6.1%}")

    # Per-regime performance for the BEST variant
    best_result = ensemble if has_ens else evolved
    regime_perf = best_result.get("per_regime", {})
    if regime_perf:
        print(f"\n  {'─' * 90}")
        print(f"  PER-REGIME PERFORMANCE (best variant: {best_result.get('label', '?')})")
        print(f"  {'─' * 90}")
        for r_label, rp in sorted(regime_perf.items()):
            print(f"    {r_label:12s}  PnL={rp['pnl']:>12,.0f}  trades={rp['trades']:>4d}")

    # Per-window comparison
    print(f"\n  {'─' * 90}")
    print(f"  PER-WINDOW SHARPE COMPARISON")
    print(f"  {'─' * 90}")
    s_windows = static.get("per_window", [])
    e_windows = evolved.get("per_window", [])
    en_windows = ensemble.get("per_window", []) if has_ens else []

    if has_ens:
        print(f"  {'Win':<4s}  {'Test Period':<24s}  "
              f"{'S PnL':>10s}  {'E PnL':>10s}  {'En PnL':>10s}  "
              f"{'S Shrp':>7s}  {'E Shrp':>7s}  {'En Shrp':>7s}")
    else:
        print(f"  {'Win':<4s}  {'Test Period':<24s}  "
              f"{'S PnL':>10s}  {'E PnL':>10s}  "
              f"{'S Shrp':>7s}  {'E Shrp':>7s}")

    n_wins = max(len(s_windows), len(e_windows), len(en_windows))
    for i in range(n_wins):
        sw = s_windows[i] if i < len(s_windows) else {}
        ew = e_windows[i] if i < len(e_windows) else {}
        enw = en_windows[i] if i < len(en_windows) else {}

        period = f"{sw.get('test_start', '?')}→{sw.get('test_end', '?')}"

        if has_ens:
            print(f"  {i:<4d}  {period:<24s}  "
                  f"{sw.get('pnl', 0):>10,.0f}  {ew.get('pnl', 0):>10,.0f}  {enw.get('pnl', 0):>10,.0f}  "
                  f"{sw.get('sharpe', 0):>7.2f}  {ew.get('sharpe', 0):>7.2f}  {enw.get('sharpe', 0):>7.2f}")
        else:
            print(f"  {i:<4d}  {period:<24s}  "
                  f"{sw.get('pnl', 0):>10,.0f}  {ew.get('pnl', 0):>10,.0f}  "
                  f"{sw.get('sharpe', 0):>7.2f}  {ew.get('sharpe', 0):>7.2f}")

    print(f"\n{'=' * 100}")


# ═══════════════════════════════════════════════════════════════════════════
# 7. STANDALONE ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import warnings
    import sys

    warnings.filterwarnings("ignore")

    PROJECT_ROOT = Path(__file__).parent
    sys.path.insert(0, str(PROJECT_ROOT))

    print("=" * 100)
    print("  PLAYER EVOLUTION ENGINE — 3-YEAR BACKTEST COMPARISON (v2)")
    print("  FIX 1: Conservative evolution (protect Sharpe > 0.5, do-no-harm)")
    print("  FIX 2: Volatility override (high-vol → 0.3× sizing)")
    print("  FIX 3: Ensemble majority filter (consensus multiplier)")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    # Load data
    print("\n[1/5] Loading 3 years of market data...")
    import yfinance as yf

    SYMBOLS = [
        "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS", "LT.NS",
        "AXISBANK.NS", "BAJFINANCE.NS", "SUNPHARMA.NS", "TATASTEEL.NS", "WIPRO.NS",
    ]

    START = "2023-01-01"
    END = "2025-12-31"

    stock_data = {}
    for sym in SYMBOLS:
        try:
            df = yf.download(sym, start=START, end=END, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                stock_data[sym] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception:
            pass

    print(f"  Loaded {len(stock_data)} symbols, "
          f"{sum(len(df) for df in stock_data.values())} total rows")

    # Run comparison
    print("\n[2/5] Running STATIC backtest...")
    print("[3/5] Running EVOLVED backtest (conservative mutation)...")
    print("[4/5] Running EVOLVED+ENSEMBLE backtest (consensus filter)...")

    static_results, evolved_results, ensemble_results = run_evolution_backtest(
        stock_data,
        training_window=120,
        testing_window=60,
        step_size=60,
        evolve_at_each_window=True,
        verbose=True,
    )

    # Print comparison
    print("\n[5/5] Printing 3-way comparison...")
    print_comparison(static_results, evolved_results, ensemble_results)

    # Summary verdict
    print(f"\n{'=' * 100}")
    print("  TARGETS CHECK")
    print(f"{'=' * 100}")
    # Find best variant
    variants = [
        ("STATIC", static_results),
        ("EVOLVED", evolved_results),
        ("EVO+ENSEMBLE", ensemble_results),
    ]
    best_var = max(variants, key=lambda x: x[1].get("sharpe", 0))
    bv_name, bv = best_var

    sharpe_ok = bv.get("sharpe", 0) > 1.3
    dd_ok = abs(bv.get("max_drawdown_pct", -100)) < 10
    all_pos = all(
        bv.get("per_player", {}).get(pid, {}).get("pnl", -1) > 0
        for pid in PLAYER_ARCHETYPES
    )

    print(f"  Best variant: {bv_name}")
    print(f"  Sharpe > 1.3:          {'PASS' if sharpe_ok else 'FAIL'} ({bv.get('sharpe', 0):.3f})")
    print(f"  Max DD < 10%:          {'PASS' if dd_ok else 'FAIL'} ({bv.get('max_drawdown_pct', 0):.2f}%)")
    print(f"  All 5 players net +ve: {'PASS' if all_pos else 'FAIL'}")
    if not all_pos:
        for pid in PLAYER_ARCHETYPES:
            pp = bv.get("per_player", {}).get(pid, {})
            if pp.get("pnl", 0) <= 0:
                print(f"    {PLAYER_ARCHETYPES[pid]}: Rs {pp.get('pnl', 0):,.0f}")
    print(f"{'=' * 100}")
