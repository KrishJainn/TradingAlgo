"""
Genetic Evolution module.

Implements genetic algorithm for evolving Super Indicator DNA.
Uses DEAP-inspired operations without the library dependency.
"""

import random
import numpy as np
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
import copy

from ..super_indicator.dna import SuperIndicatorDNA, generate_dna_id
from ..super_indicator.evolution import (
    MutationOperator, CrossoverOperator, SelectionOperator,
    apply_coach_guidance
)
from .population import PopulationManager
from .fitness import FitnessCalculator, FitnessResult, calculate_metrics_from_trades
from .hall_of_fame import HallOfFame

logger = logging.getLogger(__name__)


@dataclass
class EvolutionConfig:
    """Configuration for genetic evolution."""
    population_size: int = 50
    num_generations: int = 50
    elite_size: int = 5
    tournament_size: int = 3
    crossover_prob: float = 0.7
    mutation_prob: float = 0.2
    mutation_strength: float = 0.15
    validation_frequency: int = 10
    convergence_threshold: float = 0.001
    max_indicators_active: int = 30
    min_indicators_active: int = 10
    rollback_threshold: float = 0.8
    inject_new_indicators_frequency: int = 10


class GeneticEvolution:
    """
    Genetic Algorithm engine for evolving Super Indicator configurations.

    Evolution Process:
    1. Initialize population
    2. Evaluate fitness
    3. Select parents
    4. Apply crossover
    5. Apply mutation
    6. Preserve elite
    7. Get Coach guidance
    8. Validate against holdout
    9. Repeat until convergence
    """

    def __init__(self,
                 config: EvolutionConfig = None,
                 population_manager: PopulationManager = None,
                 fitness_calculator: FitnessCalculator = None,
                 hall_of_fame: HallOfFame = None):
        """
        Initialize genetic evolution engine.

        Args:
            config: Evolution configuration
            population_manager: Population management
            fitness_calculator: Fitness calculation
            hall_of_fame: Best strategy tracking
        """
        self.config = config or EvolutionConfig()
        self.population_manager = population_manager or PopulationManager()
        self.fitness_calculator = fitness_calculator or FitnessCalculator()
        self.hall_of_fame = hall_of_fame or HallOfFame()

        # Operators
        self.mutation = MutationOperator(
            weight_mutation_rate=config.mutation_prob if config else 0.2,
            weight_mutation_strength=config.mutation_strength if config else 0.15
        )
        self.crossover = CrossoverOperator()
        self.selection = SelectionOperator()

        # Tracking
        self.generation_history: List[dict] = []
        self.best_validation_fitness = 0.0
        self.best_checkpoint: Optional[SuperIndicatorDNA] = None
        self.generations_without_improvement = 0

    def evolve(self,
               initial_population: List[SuperIndicatorDNA],
               evaluate_fn: Callable[[SuperIndicatorDNA], dict],
               validate_fn: Callable[[SuperIndicatorDNA], float] = None,
               coach_fn: Callable[[List[dict], dict], dict] = None,
               on_generation_complete: Callable[[dict], None] = None) -> SuperIndicatorDNA:
        """
        Run the complete evolution process.

        Args:
            initial_population: Starting population
            evaluate_fn: Function to evaluate DNA (returns metrics dict)
            validate_fn: Function to validate DNA on holdout (returns fitness)
            coach_fn: Function to get Coach recommendations

        Returns:
            Best evolved DNA configuration
        """
        population = initial_population
        run_id = population[0].run_id if population else 0

        logger.info(f"Starting evolution with {len(population)} individuals")

        # Evaluate initial population
        for dna in population:
            metrics = evaluate_fn(dna)
            fitness_result = self.fitness_calculator.calculate_fitness(metrics)
            dna.fitness_score = fitness_result.fitness_score
            self._update_dna_metrics(dna, metrics)

        # Evolution loop
        for gen in range(self.config.num_generations):
            logger.info(f"Generation {gen + 1}/{self.config.num_generations}")

            # 1. Selection
            offspring = self._select(population)

            # 2. Crossover
            offspring = self._crossover(offspring, gen + 1)

            # 3. Mutation
            offspring = self._mutate(offspring, gen + 1)

            # 4. Evaluate offspring
            for dna in offspring:
                if dna.fitness_score == 0:  # Not yet evaluated
                    metrics = evaluate_fn(dna)
                    fitness_result = self.fitness_calculator.calculate_fitness(metrics)
                    dna.fitness_score = fitness_result.fitness_score
                    self._update_dna_metrics(dna, metrics)

            # 5. Elite preservation
            elite = self.selection.elite_selection(population, self.config.elite_size)

            # 6. Create new population
            population = offspring + elite
            population = sorted(population, key=lambda d: d.fitness_score, reverse=True)
            population = population[:self.config.population_size]

            # 7. Get best of this generation
            best = population[0]
            self.hall_of_fame.update(best)

            # 8. Coach guidance (if provided)
            if coach_fn:
                try:
                    all_metrics = [
                        {'dna_id': d.dna_id, 'fitness': d.fitness_score}
                        for d in population
                    ]
                    coach_rec = coach_fn(all_metrics, best.get_weights())

                    # Apply Coach recommendations to guide mutation
                    if coach_rec:
                        self._apply_coach_guidance(population, coach_rec)
                except Exception as e:
                    logger.warning(f"Coach guidance failed: {e}")

            # 9. Validation check
            if validate_fn and (gen + 1) % self.config.validation_frequency == 0:
                validation_fitness = validate_fn(best)
                best.validation_fitness = validation_fitness

                if validation_fitness > self.best_validation_fitness:
                    self.best_validation_fitness = validation_fitness
                    self.best_checkpoint = best.copy()
                    self.generations_without_improvement = 0
                    logger.info(f"New best validation fitness: {validation_fitness:.4f}")
                else:
                    self.generations_without_improvement += self.config.validation_frequency

                    # Check for rollback
                    if validation_fitness < self.best_validation_fitness * self.config.rollback_threshold:
                        logger.warning("Validation degraded, considering rollback")
                        # Don't actually rollback, just log warning

            # 10. Inject new indicators periodically
            if (gen + 1) % self.config.inject_new_indicators_frequency == 0:
                population = self.population_manager.inject_new_indicators(population)

            # 11. Maintain diversity
            population = self.population_manager.maintain_diversity(population)

            # 12. Log generation stats
            stats = self._log_generation(gen + 1, population, best)

            if on_generation_complete:
                try:
                    on_generation_complete(stats, best)
                except Exception as e:
                    logger.error(f"Failed to run generation callback: {e}")

            # 13. Early stopping check
            if self.generations_without_improvement >= 20:
                logger.info(f"Convergence reached at generation {gen + 1}")
                break

        # Return best
        final_best = self.hall_of_fame.get_best()
        logger.info(f"Evolution complete. Best fitness: {final_best.fitness_score:.4f}")

        return final_best

    def _select(self, population: List[SuperIndicatorDNA]) -> List[SuperIndicatorDNA]:
        """Select parents for next generation."""
        offspring_size = self.config.population_size - self.config.elite_size
        offspring = []

        for _ in range(offspring_size):
            parent = self.selection.tournament_selection(
                population,
                self.config.tournament_size
            )
            offspring.append(parent.copy())

        return offspring

    def _crossover(self, offspring: List[SuperIndicatorDNA],
                   generation: int) -> List[SuperIndicatorDNA]:
        """Apply crossover to offspring pairs."""
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < self.config.crossover_prob:
                child1 = self.crossover.blend_crossover(
                    offspring[i], offspring[i + 1], generation
                )
                child2 = self.crossover.blend_crossover(
                    offspring[i + 1], offspring[i], generation
                )
                offspring[i] = child1
                offspring[i + 1] = child2

        return offspring

    def _mutate(self, offspring: List[SuperIndicatorDNA],
                generation: int) -> List[SuperIndicatorDNA]:
        """Apply mutation to offspring."""
        for i, dna in enumerate(offspring):
            if random.random() < self.config.mutation_prob:
                offspring[i] = self.mutation.mutate(dna, generation)

        return offspring

    def _apply_coach_guidance(self, population: List[SuperIndicatorDNA],
                              recommendation: dict):
        """Apply Coach recommendations to population."""
        promoted = recommendation.get('indicators_to_promote', [])
        demoted = recommendation.get('indicators_to_demote', [])
        removed = recommendation.get('indicators_to_remove', [])
        adjustments = recommendation.get('weight_adjustments', {})

        # Apply to bottom half of population
        mid = len(population) // 2
        for dna in population[mid:]:
            modified = apply_coach_guidance(
                dna, promoted, demoted, removed, adjustments
            )
            # Copy modified genes back
            dna.genes = modified.genes

    def _update_dna_metrics(self, dna: SuperIndicatorDNA, metrics: dict):
        """Update DNA with performance metrics."""
        dna.sharpe_ratio = metrics.get('sharpe_ratio', 0)
        dna.max_drawdown = metrics.get('max_drawdown', 0)
        dna.net_profit = metrics.get('net_profit', 0)
        dna.win_rate = metrics.get('win_rate', 0)
        dna.total_trades = metrics.get('total_trades', 0)
        dna.long_trades = metrics.get('long_trades', 0)
        dna.short_trades = metrics.get('short_trades', 0)
        dna.profit_factor = metrics.get('profit_factor', 0)

    def _log_generation(self, generation: int,
                        population: List[SuperIndicatorDNA],
                        best: SuperIndicatorDNA) -> dict:
        """Log generation statistics."""
        fitnesses = [d.fitness_score for d in population]

        stats = {
            'generation': generation,
            'best_fitness': max(fitnesses),
            'avg_fitness': np.mean(fitnesses),
            'std_fitness': np.std(fitnesses),
            'worst_fitness': min(fitnesses),
            'best_dna_id': best.dna_id,
            'best_sharpe': best.sharpe_ratio,
            'best_profit': best.net_profit,
            'best_win_rate': best.win_rate,
            'diversity': self.population_manager._calculate_diversity(population)
        }

        self.generation_history.append(stats)

        logger.info(
            f"Gen {generation}: Best={stats['best_fitness']:.4f}, "
            f"Avg={stats['avg_fitness']:.4f}, "
            f"Sharpe={best.sharpe_ratio:.2f}, "
            f"Profit=${best.net_profit:.0f}"
        )



        return stats

    def get_evolution_summary(self) -> str:
        """Get summary of evolution run."""
        if not self.generation_history:
            return "No evolution run yet"

        first = self.generation_history[0]
        last = self.generation_history[-1]

        lines = [
            "Evolution Summary",
            "=" * 40,
            f"Generations: {len(self.generation_history)}",
            f"Starting fitness: {first['best_fitness']:.4f}",
            f"Final fitness: {last['best_fitness']:.4f}",
            f"Improvement: {last['best_fitness'] - first['best_fitness']:.4f}",
            f"Final Sharpe: {last['best_sharpe']:.2f}",
            f"Final Profit: ${last['best_profit']:.0f}",
            f"Final Win Rate: {last['best_win_rate']:.1%}"
        ]

        return "\n".join(lines)
