"""
Exit Parameter Evolution — Genetic Algorithm for SmartExitEngine Tuning.

Extends the 5-Player Coach system's genetic evolution (player_evolution.py
handles entry parameters) to also optimise the SmartExitEngine's exit
parameters.  Creates a co-evolutionary loop where better entries feed
better exits.

Mathematical Foundations
========================

1. **Genome Representation** — 34 floating-point genes encoding all
   tunable parameters of SmartExitEngine across 3 regimes (bull/bear/
   sideways) plus a global exit threshold.  Each gene has biologically-
   inspired bounds that constrain the search space.

2. **Fitness Function** — Multi-objective optimisation disguised as
   scalar:

       fitness = sharpe - dd_penalty + capture_bonus

   where:
       dd_penalty = max(0, |max_drawdown| - 0.10) x 5
       capture_bonus = avg_capture_ratio x 0.5

   Sharpe handles return/risk, capture_bonus handles exit quality
   (locking in profits), dd_penalty handles tail risk.

3. **Tournament Selection (k=3)** — Provides selection pressure while
   maintaining population diversity.  Better than proportional selection
   for non-negative fitness landscapes.

4. **Uniform Crossover** — Each gene independently sampled from either
   parent (50/50).  Appropriate for high-dimensional parameter spaces
   where genes have weak epistasis.

5. **Gaussian Mutation** — Perturbation ~ N(0, strength x range) with
   adaptive rate that increases after fitness stagnation to escape
   local optima.

Usage
=====
    from exit_evolution import ExitEvolutionEngine, ExitParameterGenome

    engine = ExitEvolutionEngine(population_size=20, n_generations=30)
    best = engine.evolve(
        entry_signals=entries,
        data=ohlcv_data,
        indicator_data=indicators,
        signal_map=signals,
        player_configs=configs,
        regime_sequence=regimes,
        total_days=30,
    )
    save_evolved_params(best, Path("data/evolved_exit_params.json"))
"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ExitParameterGenome
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExitParameterGenome:
    """Genome encoding SmartExitEngine's tunable parameters.

    Contains 34 genes: 11 parameters x 3 regimes + 1 global threshold.
    Each gene has defined bounds for safe exploration.

    Attributes:
        genes:              Dict of gene_name -> float value.
        generation:         Which generation this genome was born in.
        fitness_score:      Composite fitness from evaluate_genome().
        sharpe:             Sharpe ratio component of fitness.
        max_drawdown:       Maximum drawdown (decimal, e.g. -0.05 = -5%).
        avg_capture_ratio:  Average MFE capture ratio.
        evaluated:          Whether fitness has been computed.
    """

    genes: Dict[str, float] = field(default_factory=dict)
    generation: int = 0
    fitness_score: float = -999.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    avg_capture_ratio: float = 0.0
    evaluated: bool = False

    # ── Gene bounds ──────────────────────────────────────────────────

    GENE_RANGES: ClassVar[Dict[str, Tuple[float, float]]] = {
        # Bull regime (11 genes)
        "alpha_decay_threshold_bull":      (0.50, 0.75),
        "tp_atr_mult_bull":                (2.0, 4.0),
        "sl_atr_mult_bull":                (1.5, 3.5),
        "trailing_stop_pct_bull":          (0.015, 0.035),
        "max_hold_days_bull":              (10, 20),
        "time_decay_start_day_bull":       (5, 12),
        "partial_exit_1_atr_bull":         (1.5, 2.5),
        "partial_exit_2_atr_bull":         (2.0, 3.0),
        "drawdown_from_peak_pct_bull":     (0.30, 0.50),
        "agreement_decay_threshold_bull":  (0.40, 0.60),
        "vol_expansion_exit_mult_bull":    (2.0, 3.0),

        # Bear regime (11 genes)
        "alpha_decay_threshold_bear":      (0.25, 0.45),
        "tp_atr_mult_bear":                (0.8, 1.8),
        "sl_atr_mult_bear":                (0.7, 1.5),
        "trailing_stop_pct_bear":          (0.008, 0.020),
        "max_hold_days_bear":              (3, 8),
        "time_decay_start_day_bear":       (2, 5),
        "partial_exit_1_atr_bear":         (0.6, 1.2),
        "partial_exit_2_atr_bear":         (0.8, 1.5),
        "drawdown_from_peak_pct_bear":     (0.15, 0.30),
        "agreement_decay_threshold_bear":  (0.25, 0.45),
        "vol_expansion_exit_mult_bear":    (1.5, 2.2),

        # Sideways regime (11 genes)
        "alpha_decay_threshold_sideways":      (0.40, 0.60),
        "tp_atr_mult_sideways":                (1.2, 2.0),
        "sl_atr_mult_sideways":                (1.0, 2.0),
        "trailing_stop_pct_sideways":          (0.012, 0.025),
        "max_hold_days_sideways":              (6, 12),
        "time_decay_start_day_sideways":       (4, 8),
        "partial_exit_1_atr_sideways":         (0.8, 1.5),
        "partial_exit_2_atr_sideways":         (1.0, 1.8),
        "drawdown_from_peak_pct_sideways":     (0.25, 0.40),
        "agreement_decay_threshold_sideways":  (0.35, 0.55),
        "vol_expansion_exit_mult_sideways":    (1.8, 2.5),

        # Global parameter
        "composite_exit_threshold":        (0.45, 0.65),
    }

    # ── Crossover ────────────────────────────────────────────────────

    def crossover(self, other: ExitParameterGenome) -> ExitParameterGenome:
        """Uniform crossover: each gene from either parent (50/50).

        Returns a new genome (neither parent is modified).
        """
        child_genes: Dict[str, float] = {}
        for gene_name in self.GENE_RANGES:
            if np.random.random() < 0.5:
                child_genes[gene_name] = self.genes.get(
                    gene_name, self._default_value(gene_name),
                )
            else:
                child_genes[gene_name] = other.genes.get(
                    gene_name, self._default_value(gene_name),
                )
        return ExitParameterGenome(
            genes=child_genes,
            generation=max(self.generation, other.generation) + 1,
        )

    # ── Mutation ─────────────────────────────────────────────────────

    def mutate(
        self,
        rate: float = 0.1,
        strength: float = 0.2,
    ) -> ExitParameterGenome:
        """Gaussian mutation within gene bounds.

        Each gene is mutated with probability `rate`.  The perturbation
        is drawn from N(0, strength * gene_range) and clamped to bounds.

        Args:
            rate:     Probability of mutating each gene (0-1).
            strength: Mutation magnitude as fraction of gene range.

        Returns:
            New mutated genome (self is not modified).
        """
        mutated_genes = dict(self.genes)

        for gene_name, (lo, hi) in self.GENE_RANGES.items():
            if np.random.random() < rate:
                current = mutated_genes.get(gene_name, (lo + hi) / 2.0)
                gene_range = hi - lo
                delta = np.random.normal(0, strength * gene_range)
                new_value = float(np.clip(current + delta, lo, hi))

                # Round integer-like genes (max_hold_days, time_decay_start_day)
                if "max_hold" in gene_name or "time_decay_start" in gene_name:
                    new_value = round(new_value)

                mutated_genes[gene_name] = new_value

        return ExitParameterGenome(
            genes=mutated_genes,
            generation=self.generation + 1,
        )

    # ── Conversion to/from SmartExitEngine configs ───────────────────

    def to_regime_configs(self) -> Tuple[Dict[str, Dict[str, float]], float]:
        """Convert genome to SmartExitEngine config format.

        Returns:
            (regime_configs, exit_threshold)
            - regime_configs: {"bull": {...}, "bear": {...}, "sideways": {...}}
            - exit_threshold: float
        """
        regime_configs: Dict[str, Dict[str, float]] = {
            "bull": {},
            "bear": {},
            "sideways": {},
        }

        for gene_name, value in self.genes.items():
            if gene_name == "composite_exit_threshold":
                continue

            # Parse: "tp_atr_mult_bull" → param="tp_atr_mult", regime="bull"
            for regime in ("bull", "bear", "sideways"):
                suffix = f"_{regime}"
                if gene_name.endswith(suffix):
                    param_name = gene_name[: -len(suffix)]
                    regime_configs[regime][param_name] = value
                    break

        exit_threshold = self.genes.get("composite_exit_threshold", 0.55)
        return regime_configs, exit_threshold

    @classmethod
    def from_regime_configs(
        cls,
        regime_configs: Dict[str, Dict[str, float]],
        exit_threshold: float,
    ) -> ExitParameterGenome:
        """Create genome from existing SmartExitEngine configs."""
        genes: Dict[str, float] = {}

        for regime, params in regime_configs.items():
            for param_name, value in params.items():
                gene_name = f"{param_name}_{regime}"
                if gene_name in cls.GENE_RANGES:
                    genes[gene_name] = value

        genes["composite_exit_threshold"] = exit_threshold

        # Fill any missing genes with defaults (midpoint of range)
        for gene_name in cls.GENE_RANGES:
            if gene_name not in genes:
                lo, hi = cls.GENE_RANGES[gene_name]
                genes[gene_name] = (lo + hi) / 2.0

        return cls(genes=genes)

    @classmethod
    def random(cls, rng: Optional[np.random.RandomState] = None) -> ExitParameterGenome:
        """Generate a random genome within all bounds."""
        if rng is None:
            rng = np.random.RandomState()

        genes: Dict[str, float] = {}
        for gene_name, (lo, hi) in cls.GENE_RANGES.items():
            val = float(rng.uniform(lo, hi))
            if "max_hold" in gene_name or "time_decay_start" in gene_name:
                val = round(val)
            genes[gene_name] = val

        return cls(genes=genes)

    # ── Helpers ──────────────────────────────────────────────────────

    def _default_value(self, gene_name: str) -> float:
        """Return midpoint of gene range as default."""
        lo, hi = self.GENE_RANGES.get(gene_name, (0.0, 1.0))
        return (lo + hi) / 2.0

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"Gen{self.generation:>3d} "
            f"fitness={self.fitness_score:>7.3f} "
            f"sharpe={self.sharpe:>6.2f} "
            f"DD={self.max_drawdown:>7.2%} "
            f"capture={self.avg_capture_ratio:>.3f}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fitness Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_genome(
    genome: ExitParameterGenome,
    entry_signals: list,
    data: dict,
    indicator_data: dict,
    signal_map: dict,
    player_configs: dict,
    regime_sequence: list,
    total_days: int,
    hmm_model: Optional[Any] = None,
    regime_state_map: Optional[Dict[str, int]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Evaluate genome fitness using a backtest.

    Uses simulate() and compute_metrics() from backtest_exit_comparison.py
    to run the genome's parameters through a 30-day backtest.

    Fitness formula:
        fitness = sharpe - dd_penalty + capture_bonus
    where:
        dd_penalty = max(0, |max_drawdown| - 0.10) x 5
        capture_bonus = avg_capture_ratio x 0.5

    Returns:
        (fitness_score, metrics_dict)
    """
    from backtest_exit_comparison import simulate, compute_metrics

    # Convert genome to SmartExitEngine config
    regime_configs, exit_threshold = genome.to_regime_configs()

    try:
        players = simulate(
            mode="smart_exit",
            entry_events=entry_signals,
            data=data,
            indicator_data=indicator_data,
            signal_map=signal_map,
            player_configs=player_configs,
            regime_sequence=regime_sequence,
            total_days=total_days,
            exit_threshold=exit_threshold,
            hmm_model=hmm_model,
            regime_state_map=regime_state_map,
            regime_config_override=regime_configs,
        )

        metrics = compute_metrics(players, total_days, "evo")

        sharpe = metrics.get("sharpe", 0.0)
        max_dd_pct = metrics.get("max_drawdown_pct", 0.0)  # e.g. -0.66
        max_dd_decimal = abs(max_dd_pct) / 100.0  # e.g. 0.0066
        capture = metrics.get("avg_capture_ratio", 0.0)

        # Fitness components
        dd_penalty = max(0.0, max_dd_decimal - 0.10) * 5.0
        capture_bonus = capture * 0.5

        fitness = sharpe - dd_penalty + capture_bonus

        # Store in genome
        genome.fitness_score = fitness
        genome.sharpe = sharpe
        genome.max_drawdown = -max_dd_decimal  # store as negative decimal
        genome.avg_capture_ratio = capture
        genome.evaluated = True

        return fitness, metrics

    except Exception as e:
        logger.warning(f"[ExitEvolution] Genome evaluation failed: {e}")
        genome.fitness_score = -999.0
        genome.evaluated = True
        return -999.0, {}


# ═══════════════════════════════════════════════════════════════════════════
# ExitEvolutionEngine
# ═══════════════════════════════════════════════════════════════════════════

class ExitEvolutionEngine:
    """Genetic algorithm for evolving SmartExitEngine parameters.

    Population-based search with tournament selection, uniform crossover,
    and adaptive Gaussian mutation.

    Parameters
    ----------
    population_size : int
        Number of genomes in the population (default 20).
    n_generations : int
        Maximum number of generations to evolve (default 30).
    tournament_k : int
        Tournament selection size (default 3).
    mutation_rate : float
        Probability of mutating each gene (default 0.10).
    mutation_strength : float
        Mutation magnitude as fraction of gene range (default 0.20).
    elitism : int
        Number of top genomes kept unchanged each generation (default 2).
    plateau_patience : int
        Stop if best fitness doesn't improve for this many generations
        (default 5).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        population_size: int = 20,
        n_generations: int = 30,
        tournament_k: int = 3,
        mutation_rate: float = 0.10,
        mutation_strength: float = 0.20,
        elitism: int = 2,
        plateau_patience: int = 5,
        seed: int = 42,
    ) -> None:
        self.population_size = population_size
        self.n_generations = n_generations
        self.tournament_k = tournament_k
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elitism = elitism
        self.plateau_patience = plateau_patience
        self.rng = np.random.RandomState(seed)

        self.best_fitness_history: List[float] = []
        self.plateau_counter: int = 0

    def evolve(
        self,
        entry_signals: list,
        data: dict,
        indicator_data: dict,
        signal_map: dict,
        player_configs: dict,
        regime_sequence: list,
        total_days: int,
        hmm_model: Optional[Any] = None,
        regime_state_map: Optional[Dict[str, int]] = None,
        initial_genome: Optional[ExitParameterGenome] = None,
        verbose: bool = True,
    ) -> ExitParameterGenome:
        """Run genetic evolution to find optimal exit parameters.

        Args:
            entry_signals:   Pre-generated entry events (shared across all genomes).
            data:            OHLCV data dict {symbol: DataFrame}.
            indicator_data:  Indicator data dict {symbol: DataFrame}.
            signal_map:      Per-player, per-symbol, per-day signal values.
            player_configs:  Player configuration dicts.
            regime_sequence: List of regime labels per day.
            total_days:      Number of days in the backtest window.
            hmm_model:       Fitted HMM model (optional).
            regime_state_map: Regime label to HMM state index (optional).
            initial_genome:  Previously evolved genome to seed (optional).
            verbose:         Print progress per generation.

        Returns:
            Best genome found.
        """
        t_start = time.time()

        # ── Seed population ──────────────────────────────────────────
        population: List[ExitParameterGenome] = []

        # Genome 0: current SmartExitEngine defaults
        from smart_exit_engine import SmartExitEngine
        default_genome = ExitParameterGenome.from_regime_configs(
            SmartExitEngine.REGIME_CONFIGS,
            SmartExitEngine.EXIT_THRESHOLD,
        )
        default_genome.generation = 0
        population.append(default_genome)

        # Genome 1: previously evolved (if available)
        if initial_genome is not None:
            initial_genome.evaluated = False  # re-evaluate
            population.append(initial_genome)

        # Fill rest with random genomes
        while len(population) < self.population_size:
            population.append(ExitParameterGenome.random(self.rng))

        if verbose:
            print(f"  Population: {len(population)} genomes "
                  f"({len(ExitParameterGenome.GENE_RANGES)} genes each)")

        # ── Evolution loop ───────────────────────────────────────────
        best_ever: Optional[ExitParameterGenome] = None

        for gen in range(self.n_generations):
            t_gen = time.time()

            # Evaluate un-evaluated genomes
            for genome in population:
                if not genome.evaluated:
                    evaluate_genome(
                        genome, entry_signals, data, indicator_data,
                        signal_map, player_configs, regime_sequence,
                        total_days, hmm_model, regime_state_map,
                    )

            # Sort by fitness (descending)
            population.sort(key=lambda g: g.fitness_score, reverse=True)
            best = population[0]

            # Track best ever
            if best_ever is None or best.fitness_score > best_ever.fitness_score:
                best_ever = deepcopy(best)

            # Log progress
            if verbose:
                gen_time = time.time() - t_gen
                print(
                    f"  Gen {gen:>2d}: "
                    f"fitness={best.fitness_score:>7.3f}  "
                    f"sharpe={best.sharpe:>6.2f}  "
                    f"DD={best.max_drawdown:>7.2%}  "
                    f"capture={best.avg_capture_ratio:>.3f}  "
                    f"({gen_time:.1f}s)"
                )

            # ── Plateau detection ────────────────────────────────────
            self.best_fitness_history.append(best.fitness_score)

            if len(self.best_fitness_history) >= self.plateau_patience:
                recent = self.best_fitness_history[-self.plateau_patience:]
                if max(recent) - min(recent) < 0.05:
                    self.plateau_counter += 1
                else:
                    self.plateau_counter = 0

            if self.plateau_counter >= 3:
                if verbose:
                    print(f"  Early stop: fitness plateau for "
                          f"{self.plateau_counter * self.plateau_patience} gens")
                break

            # ── Create next generation ───────────────────────────────

            # Elitism: keep top N unchanged
            next_population = [deepcopy(g) for g in population[:self.elitism]]
            for g in next_population:
                g.evaluated = True  # keep their fitness

            # Adaptive mutation rate
            adaptive_rate = self.mutation_rate
            if self.plateau_counter > 0:
                adaptive_rate = min(0.5, adaptive_rate * (1.2 ** self.plateau_counter))

            # Fill rest via selection → crossover → mutation
            while len(next_population) < self.population_size:
                parent_a = self._tournament_select(population)
                parent_b = self._tournament_select(population)
                child = parent_a.crossover(parent_b)
                child = child.mutate(rate=adaptive_rate, strength=self.mutation_strength)
                next_population.append(child)

            population = next_population

        # ── Final result ─────────────────────────────────────────────
        total_time = time.time() - t_start

        if verbose:
            print(f"\n  Evolution complete in {total_time:.1f}s")
            if best_ever:
                print(f"  Best genome: {best_ever.summary()}")

        return best_ever if best_ever is not None else population[0]

    def _tournament_select(
        self,
        population: List[ExitParameterGenome],
    ) -> ExitParameterGenome:
        """Tournament selection: pick k random, return the fittest."""
        k = min(self.tournament_k, len(population))
        indices = self.rng.choice(len(population), size=k, replace=False)
        contestants = [population[i] for i in indices]
        return max(contestants, key=lambda g: g.fitness_score)


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════

def save_evolved_params(
    genome: ExitParameterGenome,
    save_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save evolved exit parameters to JSON.

    The format is designed for SmartExitEngine._load_evolved_params():
    {
        "saved_at": "...",
        "regime_configs": {"bull": {...}, "bear": {...}, "sideways": {...}},
        "exit_threshold": 0.52,
        "fitness_score": 1.95,
        "sharpe": 1.45,
        "max_drawdown_pct": -6.2,
        "avg_capture_ratio": 0.55,
        "generation": 24,
        "gene_values": {...}
    }
    """
    regime_configs, exit_threshold = genome.to_regime_configs()

    data = {
        "saved_at": datetime.now().isoformat(),
        "regime_configs": regime_configs,
        "exit_threshold": round(exit_threshold, 4),
        "fitness_score": round(genome.fitness_score, 3),
        "sharpe": round(genome.sharpe, 2),
        "max_drawdown_pct": round(genome.max_drawdown * 100, 2),
        "avg_capture_ratio": round(genome.avg_capture_ratio, 3),
        "generation": genome.generation,
        "gene_values": {k: round(v, 6) for k, v in genome.genes.items()},
    }

    if metadata:
        data["metadata"] = metadata

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"[ExitEvolution] Saved evolved params to {save_path}")


def load_evolved_params(
    load_path: Path,
) -> Optional[ExitParameterGenome]:
    """Load evolved exit parameters from JSON.

    Returns None if file doesn't exist or is corrupted.
    """
    if not load_path.exists():
        return None

    try:
        with open(load_path, "r") as f:
            data = json.load(f)

        regime_configs = data.get("regime_configs", {})
        exit_threshold = data.get("exit_threshold", 0.55)

        genome = ExitParameterGenome.from_regime_configs(
            regime_configs, exit_threshold,
        )

        # Restore fitness metadata
        genome.fitness_score = data.get("fitness_score", 0.0)
        genome.sharpe = data.get("sharpe", 0.0)
        genome.max_drawdown = data.get("max_drawdown_pct", 0.0) / 100.0
        genome.avg_capture_ratio = data.get("avg_capture_ratio", 0.0)
        genome.generation = data.get("generation", 0)

        # Prefer stored gene_values if available (more precise)
        if "gene_values" in data:
            for gene_name, value in data["gene_values"].items():
                if gene_name in ExitParameterGenome.GENE_RANGES:
                    genome.genes[gene_name] = value

        logger.info(
            f"[ExitEvolution] Loaded evolved params from {load_path} "
            f"(fitness={genome.fitness_score:.3f})"
        )
        return genome

    except Exception as e:
        logger.warning(f"[ExitEvolution] Failed to load evolved params: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Standalone runner
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Run exit parameter evolution as a standalone script."""
    print("=" * 70)
    print("  EXIT PARAMETER EVOLUTION — SmartExitEngine Genetic Optimiser")
    print("=" * 70)

    # 1. Load data (reuse from backtest_exit_comparison)
    print("\n[1/5] Loading data and indicators...")
    from backtest_exit_comparison import (
        load_data, load_evolved_configs, generate_entry_signals,
        compute_regime_sequence, PLAYERS_CONFIG, COMPARISON_DAYS,
        BARS_PER_DAY,
    )
    from copy import deepcopy

    data, indicator_data = load_data()
    print(f"       {len(data)} symbols loaded")

    if len(data) < 5:
        print("ERROR: Not enough data.")
        return

    # 2. Configs + regime
    print("[2/5] Loading configs and computing regime...")
    configs = load_evolved_configs()
    if not configs:
        configs = deepcopy(PLAYERS_CONFIG)

    sample_df = list(data.values())[0]
    total_days = min(30, len(sample_df) // BARS_PER_DAY)  # 30-day window for evolution
    regime_sequence, hmm_model, regime_state_map = compute_regime_sequence(data, total_days)
    print(f"       {total_days} days, regime: {regime_sequence[0] if regime_sequence else 'unknown'}")

    # 3. Generate entry signals (shared)
    print("[3/5] Generating entry signals...")
    entry_events, signal_map = generate_entry_signals(
        data, indicator_data, configs, regime_sequence, total_days,
    )
    print(f"       {len(entry_events)} entry signals")

    # 4. Load previous best (if exists)
    params_path = Path("data/evolved_exit_params.json")
    prev_best = load_evolved_params(params_path)
    if prev_best:
        print(f"       Previous best: fitness={prev_best.fitness_score:.3f}")

    # 5. Run evolution
    print(f"\n[4/5] Running evolution (pop=20, gen=30)...")
    engine = ExitEvolutionEngine(
        population_size=20,
        n_generations=30,
        plateau_patience=5,
        seed=42,
    )

    best = engine.evolve(
        entry_signals=entry_events,
        data=data,
        indicator_data=indicator_data,
        signal_map=signal_map,
        player_configs=configs,
        regime_sequence=regime_sequence,
        total_days=total_days,
        hmm_model=hmm_model,
        regime_state_map=regime_state_map,
        initial_genome=prev_best,
        verbose=True,
    )

    # 6. Save
    print(f"\n[5/5] Saving evolved parameters...")
    save_evolved_params(best, params_path, metadata={
        "trigger": "manual",
        "evolution_date": datetime.now().isoformat(),
    })

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  EVOLUTION COMPLETE")
    print(f"  Best fitness:   {best.fitness_score:.3f}")
    print(f"  Sharpe:         {best.sharpe:.2f}")
    print(f"  Max drawdown:   {best.max_drawdown:.2%}")
    print(f"  Capture ratio:  {best.avg_capture_ratio:.3f}")
    print(f"  Generation:     {best.generation}")
    print(f"  Saved to:       {params_path}")

    # Print key gene differences from defaults
    from smart_exit_engine import SmartExitEngine
    default = ExitParameterGenome.from_regime_configs(
        SmartExitEngine.REGIME_CONFIGS, SmartExitEngine.EXIT_THRESHOLD,
    )
    print(f"\n  Key parameter changes vs defaults:")
    diffs = []
    for gene, val in best.genes.items():
        default_val = default.genes.get(gene, val)
        if abs(val - default_val) > 0.001:
            pct_change = (val - default_val) / default_val * 100 if default_val != 0 else 0
            diffs.append((gene, default_val, val, pct_change))

    diffs.sort(key=lambda x: abs(x[3]), reverse=True)
    for gene, old, new, pct in diffs[:10]:
        print(f"    {gene:40s}  {old:>8.4f} → {new:>8.4f}  ({pct:+.1f}%)")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
