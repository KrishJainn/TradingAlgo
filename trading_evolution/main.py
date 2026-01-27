"""
Main Evolution Loop Orchestrator.

This is the main entry point for running the genetic evolution
to discover optimal Super Indicator configurations.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd

# Configuration
from .config import (
    DataConfig, RiskConfig, PortfolioConfig, IndicatorConfig,
    SignalConfig, EvolutionConfig, FitnessConfig, DatabaseConfig,
    ReportingConfig
)

# Data
from .data.fetcher import DataFetcher
from .data.cache import DataCache
from .data.market_regime import RegimeDetector

# Indicators
from .indicators.universe import IndicatorUniverse
from .indicators.calculator import IndicatorCalculator
from .indicators.normalizer import IndicatorNormalizer
from .indicators.ranking import IndicatorRanker

# Super Indicator
from .super_indicator.dna import SuperIndicatorDNA
from .super_indicator.core import SuperIndicator
from .super_indicator.signals import SignalGenerator, SignalType, PositionState

# Player
from .player.trader import Player
from .player.portfolio import Portfolio
from .player.risk_manager import RiskManager, RiskParameters
from .player.execution import ExecutionEngine

# Coach
from .coach.analyzer import Coach
from .coach.indicator_scorer import IndicatorScorer
from .coach.pattern_detector import PatternDetector
from .coach.regime_analyzer import RegimeAnalyzer
from .coach.recommendations import RecommendationGenerator

# Evolution
from .evolution.genetic import GeneticEvolution, EvolutionConfig as GeneticConfig
from .evolution.fitness import FitnessCalculator
from .evolution.population import PopulationManager
from .evolution.hall_of_fame import HallOfFame

# Journal
from .journal.database import Database
from .journal.trade_logger import TradeLogger
from .journal.generation_logger import GenerationLogger

# Reporting
from .reporting.equity_curve import EquityCurveChart, create_equity_report
from .reporting.indicator_importance import (
    IndicatorImportanceChart, create_indicator_report
)

# Live
from .live.signal_generator import LiveSignalGenerator, generate_daily_signals
from .live.watchlist import Watchlist, create_default_watchlist
from .live.alerts import AlertManager, print_signal_banner

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EvolutionOrchestrator:
    """
    Main orchestrator for the evolution process.

    Coordinates:
    - Data fetching and preprocessing
    - Indicator calculation and normalization
    - Population management
    - Fitness evaluation via Player simulation
    - Coach analysis and guidance
    - Genetic evolution
    - Logging and reporting
    """

    def __init__(self,
                 data_config: DataConfig = None,
                 evolution_config: EvolutionConfig = None,
                 portfolio_config: PortfolioConfig = None,
                 risk_config: RiskConfig = None,
                 database_config: DatabaseConfig = None,
                 reporting_config: ReportingConfig = None):
        """
        Initialize the orchestrator.

        Args:
            data_config: Data fetching configuration
            evolution_config: Genetic evolution configuration
            portfolio_config: Portfolio management configuration
            risk_config: Risk management configuration
            database_config: Database configuration
            reporting_config: Reporting configuration
        """
        # Configs
        self.data_config = data_config or DataConfig()
        self.evolution_config = evolution_config or EvolutionConfig()
        self.portfolio_config = portfolio_config or PortfolioConfig()
        self.risk_config = risk_config or RiskConfig()
        self.database_config = database_config or DatabaseConfig()
        self.reporting_config = reporting_config or ReportingConfig()

        # Initialize database
        self.db = Database(self.database_config.db_path)

        # Initialize components
        self._init_data_components()
        self._init_indicator_components()
        self._init_evolution_components()
        self._init_logging_components()

        # Cached data
        self._market_data: Dict[str, dict] = {}
        self._indicator_data: Dict[str, dict] = {}

        logger.info("EvolutionOrchestrator initialized")

    def _init_data_components(self):
        """Initialize data fetching components."""
        self.data_cache = DataCache(self.data_config.cache_dir)
        self.data_fetcher = DataFetcher(
            cache=self.data_cache,
            cache_dir=self.data_config.cache_dir
        )
        self.regime_detector = RegimeDetector()

    def _init_indicator_components(self):
        """Initialize indicator components."""
        self.indicator_universe = IndicatorUniverse()
        self.indicator_universe.load_all()

        self.indicator_calculator = IndicatorCalculator(
            universe=self.indicator_universe
        )
        self.indicator_normalizer = IndicatorNormalizer()
        self.indicator_ranker = IndicatorRanker()

    def _init_evolution_components(self):
        """Initialize evolution components."""
        # Population manager
        self.population_manager = PopulationManager(
            universe=self.indicator_universe,
            min_indicators=self.evolution_config.min_indicators_active,
            max_indicators=self.evolution_config.max_indicators_active
        )

        # Fitness calculator
        self.fitness_calculator = FitnessCalculator(
            min_trades=30,
            min_win_rate=0.3,
            max_acceptable_drawdown=0.25
        )

        # Hall of Fame
        self.hall_of_fame = HallOfFame(
            database=self.db,
            max_size=10
        )

        # Coach components
        self.indicator_scorer = IndicatorScorer()
        self.pattern_detector = PatternDetector()
        self.regime_analyzer = RegimeAnalyzer()
        self.recommendation_generator = RecommendationGenerator()

        self.coach = Coach(
            database=self.db,
            indicator_scorer=self.indicator_scorer,
            pattern_detector=self.pattern_detector,
            regime_analyzer=self.regime_analyzer,
            recommendation_generator=self.recommendation_generator
        )

        # Genetic evolution engine
        genetic_config = GeneticConfig(
            population_size=self.evolution_config.population_size,
            num_generations=self.evolution_config.num_generations,
            elite_size=self.evolution_config.elite_size,
            tournament_size=self.evolution_config.tournament_size,
            crossover_prob=self.evolution_config.crossover_prob,
            mutation_prob=self.evolution_config.mutation_prob,
            mutation_strength=self.evolution_config.mutation_strength,
            validation_frequency=self.evolution_config.validation_frequency
        )

        self.genetic_evolution = GeneticEvolution(
            config=genetic_config,
            population_manager=self.population_manager,
            fitness_calculator=self.fitness_calculator,
            hall_of_fame=self.hall_of_fame
        )

    def _init_logging_components(self):
        """Initialize logging components."""
        self.trade_logger = TradeLogger(self.db)
        self.generation_logger = GenerationLogger(self.db)

    def run_evolution(self, run_name: str = None) -> SuperIndicatorDNA:
        """
        Run the complete evolution process.

        Args:
            run_name: Optional name for this evolution run

        Returns:
            Best evolved DNA configuration
        """
        # 1. Fetch and prepare data first (needed for date splits)
        logger.info("Fetching market data...")
        self._fetch_all_data()

        # Get date ranges for data splits
        data_splits = {}
        for symbol, data in self._market_data.items():
            if 'train' in data and len(data['train']) > 0:
                data_splits['training'] = (
                    str(data['train'].index[0].date()),
                    str(data['train'].index[-1].date())
                )
            if 'validation' in data and len(data['validation']) > 0:
                data_splits['validation'] = (
                    str(data['validation'].index[0].date()),
                    str(data['validation'].index[-1].date())
                )
            if 'holdout' in data and len(data['holdout']) > 0:
                data_splits['holdout'] = (
                    str(data['holdout'].index[0].date()),
                    str(data['holdout'].index[-1].date())
                )
            break  # Only need dates from first symbol

        # Create run
        config_json = json.dumps({
            'name': run_name or f"Evolution_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'population_size': self.evolution_config.population_size,
            'num_generations': self.evolution_config.num_generations
        })
        run_id = self.db.create_evolution_run(
            config_json=config_json,
            symbols=self.data_config.symbols,
            data_splits=data_splits
        )

        logger.info(f"Starting evolution run {run_id}: {run_name}")

        # 2. Calculate indicators for all data
        logger.info("Calculating indicators...")
        self._calculate_all_indicators()

        # 3. Create initial population
        logger.info("Creating initial population...")
        population = self.population_manager.create_initial_population(
            size=self.evolution_config.population_size,
            run_id=run_id,
            ensure_diversity=True
        )

        # 4. Define evaluation function
        def evaluate_fn(dna: SuperIndicatorDNA) -> dict:
            return self._evaluate_dna(dna, 'train')

        # 5. Define validation function
        def validate_fn(dna: SuperIndicatorDNA) -> float:
            metrics = self._evaluate_dna(dna, 'validation')
            return self.fitness_calculator.calculate_fitness(metrics).fitness_score

        # 6. Define coach function
        def coach_fn(population_metrics: List[dict], weights: dict) -> dict:
            # Get recommendation from most recent Coach analysis
            if self.coach.analysis_history:
                rec = self.coach.analysis_history[-1].get('recommendation')
                if rec:
                    return {
                        'indicators_to_promote': rec.indicators_to_promote,
                        'indicators_to_demote': rec.indicators_to_demote,
                        'indicators_to_remove': rec.indicators_to_remove,
                        'weight_adjustments': rec.weight_adjustments
                    }
            return {}

        # Define callback for incremental saving
        def on_gen_complete(stats: dict, gen_best_dna: SuperIndicatorDNA):
            try:
                # Save generation stats
                gen_id = self.db.create_generation(
                    run_id=run_id, 
                    generation_num=stats['generation'],
                    population_size=self.evolution_config.population_size
                )
                
                self.db.update_generation(
                    generation_id=gen_id,
                    completed_at=pd.Timestamp.now().isoformat(),
                    best_fitness=stats['best_fitness'],
                    avg_fitness=stats['avg_fitness'],
                    std_fitness=stats['std_fitness'],
                    best_dna_id=stats['best_dna_id'],
                    sharpe_ratio=stats['best_sharpe'],
                    total_net_pnl=stats['best_profit'],
                    win_rate=stats['best_win_rate'],
                    status='completed'
                )
                
                # Save best DNA of this generation
                self.db.save_dna_config(run_id, stats['generation'], gen_best_dna)
                
                # Update Hall of Fame in DB
                self.db.update_hall_of_fame(
                    dna_id=gen_best_dna.dna_id,
                    run_id=run_id,
                    generation_num=stats['generation'],
                    fitness_score=gen_best_dna.fitness_score,
                    metrics={
                        'sharpe_ratio': gen_best_dna.sharpe_ratio,
                        'max_drawdown': gen_best_dna.max_drawdown,
                        'net_profit': gen_best_dna.net_profit,
                        'win_rate': gen_best_dna.win_rate,
                        'total_trades': gen_best_dna.total_trades
                    },
                    weights_json=json.dumps(gen_best_dna.get_weights())
                )
                logger.info(f"Saved stats for Run {run_id} Gen {stats['generation']}")
            except Exception as e:
                logger.error(f"Error saving generation stats: {e}")

        # 7. Run evolution
        logger.info("Starting genetic evolution...")
        best_dna = self.genetic_evolution.evolve(
            initial_population=population,
            evaluate_fn=evaluate_fn,
            validate_fn=validate_fn,
            coach_fn=coach_fn,
            on_generation_complete=on_gen_complete
        )

        # 8. Final holdout validation
        logger.info("Running holdout validation...")
        holdout_metrics = self._evaluate_dna(best_dna, 'holdout')
        holdout_fitness = self.fitness_calculator.calculate_fitness(holdout_metrics)
        best_dna.holdout_fitness = holdout_fitness.fitness_score

        # 9. Save final results
        self.db.save_dna_config(run_id, best_dna.generation, best_dna)
        self.hall_of_fame.update(best_dna)

        # 10. Generate reports
        logger.info("Generating reports...")
        self._generate_reports(run_id)

        # 11. Print summary
        self._print_summary(best_dna, holdout_metrics)

        logger.info(f"Evolution complete. Best fitness: {best_dna.fitness_score:.4f}")
        return best_dna

    def _fetch_all_data(self):
        """Fetch data for all symbols."""
        for symbol in self.data_config.symbols:
            logger.info(f"Fetching data for {symbol}...")

            df = self.data_fetcher.fetch(symbol, years=self.data_config.data_years)

            if df is not None and len(df) > 0:
                # Split data inline (time-based split)
                n = len(df)
                train_end = int(n * self.data_config.train_pct)
                val_end = int(n * (self.data_config.train_pct + self.data_config.validation_pct))

                splits = {
                    'train': df.iloc[:train_end].copy(),
                    'validation': df.iloc[train_end:val_end].copy(),
                    'holdout': df.iloc[val_end:].copy()
                }

                self._market_data[symbol] = {
                    'full': df,
                    'train': splits['train'],
                    'validation': splits['validation'],
                    'holdout': splits['holdout']
                }

                logger.info(
                    f"  {symbol}: {len(df)} bars total, "
                    f"{len(splits['train'])} train, "
                    f"{len(splits['validation'])} validation, "
                    f"{len(splits['holdout'])} holdout"
                )

    def _calculate_all_indicators(self):
        """Calculate indicators for all market data."""
        for symbol, data in self._market_data.items():
            logger.info(f"Calculating indicators for {symbol}...")

            for split_name, df in data.items():
                if df is not None and len(df) > 0:
                    indicators = self.indicator_calculator.calculate_all(df)

                    if symbol not in self._indicator_data:
                        self._indicator_data[symbol] = {}

                    self._indicator_data[symbol][split_name] = indicators

    def _evaluate_dna(self, dna: SuperIndicatorDNA, split: str = 'train') -> dict:
        """
        Evaluate DNA by simulating trades.

        Args:
            dna: DNA configuration to evaluate
            split: Data split to use ('train', 'validation', 'holdout')

        Returns:
            Performance metrics dictionary
        """
        # Create Player
        portfolio = Portfolio(
            initial_capital=self.portfolio_config.initial_capital
        )
        risk_params = RiskParameters(
            max_risk_per_trade=self.risk_config.max_risk_per_trade,
            max_position_pct=self.risk_config.max_position_pct
        )
        risk_manager = RiskManager(params=risk_params)
        execution = ExecutionEngine(
            slippage_pct=0.001,
            commission_per_share=0.005
        )

        player = Player(
            portfolio=portfolio,
            risk_manager=risk_manager,
            execution=execution
        )

        # Create Super Indicator
        super_indicator = SuperIndicator(dna, normalizer=self.indicator_normalizer)
        signal_generator = SignalGenerator()

        # Simulate trading for each symbol
        all_trades = []

        for symbol in self.data_config.symbols:
            if symbol not in self._market_data:
                continue

            df = self._market_data[symbol].get(split)
            if df is None or len(df) < 50:
                continue

            # Get indicators for this symbol/split
            indicators = self._indicator_data.get(symbol, {}).get(split)
            if indicators is None or indicators.empty:
                continue

            # Get active indicator list
            active_indicators = dna.get_active_indicators()

            # Ensure we only pick active indicators that exist in the calculated set
            valid_active_indicators = [
                ind for ind in active_indicators 
                if ind in indicators.columns
            ]
            if not valid_active_indicators:
                continue

            # Pre-calculate normalized indicators and SI (Vectorized)
            # Use expanding windows in normalizer to prevent lookahead
            active_indicators_df = indicators[valid_active_indicators]
            normalized_all = self.indicator_normalizer.normalize_all(
                active_indicators_df,
                price_series=df['close']
            )
            
            if normalized_all.empty:
                continue
                
            si_series = super_indicator.calculate(normalized_all)

            # Initialize previous SI value for signal generation
            prev_si_value = 0.0

            # Simulate through each bar
            for i in range(50, len(df)):
                current_bar = df.iloc[i]
                timestamp = current_bar.name

                # Get pre-calculated SI value
                si_value = float(si_series.iloc[i])

                # Determine current position state
                current_position = portfolio.get_position(symbol)
                if current_position is None:
                    pos_state = PositionState.FLAT
                elif current_position.direction == 'LONG':
                    pos_state = PositionState.LONG
                elif current_position.direction == 'SHORT':
                    pos_state = PositionState.SHORT
                else:
                    pos_state = PositionState.FLAT

                # Generate signal
                signal = signal_generator._determine_signal(
                    si=si_value,
                    si_prev=prev_si_value,
                    position=pos_state
                )
                
                # Update previous value for next iteration
                prev_si_value = si_value

                # Get ATR for risk management
                atr = 0.0
                if 'ATR_14' in indicators.columns:
                    atr = float(indicators.iloc[i]['ATR_14'])
                else:
                    # Fallback to TR or 1% of price
                    atr = float(current_bar['high'] - current_bar['low'])
                    if atr <= 0:
                        atr = float(current_bar['close'] * 0.01)

                # Process signal through player
                trade = player.process_signal(
                    symbol=symbol,
                    signal=signal,
                    current_price=current_bar['close'],
                    timestamp=timestamp,
                    high=current_bar['high'],
                    low=current_bar['low'],
                    atr=atr,
                    si_value=si_value
                )

                if trade:
                    all_trades.append(trade)

        # Close all positions at end
        # Prepare closing call
        last_timestamp = None
        final_prices = {}
        
        for s in self.data_config.symbols:
            if s in self._market_data and split in self._market_data[s]:
                d = self._market_data[s][split]
                if not d.empty:
                    final_prices[s] = float(d.iloc[-1]['close'])
                    end_time = d.index[-1]
                    if last_timestamp is None:
                        last_timestamp = end_time
                    elif end_time > last_timestamp:
                        last_timestamp = end_time

        if last_timestamp is None:
            last_timestamp = pd.Timestamp.now()

        final_trades = player.close_all_positions(
            timestamp=last_timestamp,
            prices=final_prices
        )
        all_trades.extend(final_trades)

        # Calculate metrics
        from .evolution.fitness import calculate_metrics_from_trades
        # Convert Trade objects to dicts
        trades_dicts = [t.__dict__ for t in all_trades]
        metrics = calculate_metrics_from_trades(trades_dicts)

        return metrics

    def _generate_reports(self, run_id: int):
        """Generate reports for the evolution run."""
        output_dir = Path(self.reporting_config.output_dir) / f"run_{run_id}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get generation history
        generations = self.genetic_evolution.generation_history

        if not generations:
            return

        # Fitness evolution chart
        chart = IndicatorImportanceChart()
        chart.plot_fitness_evolution(
            generation_stats=generations,
            save_path=str(output_dir / 'fitness_evolution.png'),
            title=f"Fitness Evolution - Run {run_id}"
        )

        # Save generation history as JSON
        with open(output_dir / 'generation_history.json', 'w') as f:
            json.dump(generations, f, indent=2, default=str)

        # Hall of Fame summary
        with open(output_dir / 'hall_of_fame.txt', 'w') as f:
            f.write(self.hall_of_fame.get_summary())

        # Save best DNA
        best_dna = self.hall_of_fame.get_best()
        if best_dna:
            self.hall_of_fame.save_to_file(str(output_dir / 'best_dna.json'))

        logger.info(f"Reports saved to {output_dir}")

    def _print_summary(self, best_dna: SuperIndicatorDNA, holdout_metrics: dict):
        """Print evolution summary."""
        print("\n" + "=" * 70)
        print("EVOLUTION COMPLETE")
        print("=" * 70)
        print(f"\nBest DNA: {best_dna.dna_id}")
        print(f"Generation: {best_dna.generation}")
        print(f"\nTraining Fitness: {best_dna.fitness_score:.4f}")
        val_fit = best_dna.validation_fitness
        print(f"Validation Fitness: {val_fit:.4f}" if val_fit is not None else "Validation Fitness: N/A")
        hold_fit = best_dna.holdout_fitness
        print(f"Holdout Fitness: {hold_fit:.4f}" if hold_fit is not None else "Holdout Fitness: N/A")
        print(f"\nPerformance Metrics:")
        print(f"  Sharpe Ratio: {holdout_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Net Profit: ${holdout_metrics.get('net_profit', 0):.2f}")
        print(f"  Max Drawdown: {holdout_metrics.get('max_drawdown', 0):.1%}")
        print(f"  Win Rate: {holdout_metrics.get('win_rate', 0):.1%}")
        print(f"  Total Trades: {holdout_metrics.get('total_trades', 0)}")
        print(f"\nActive Indicators: {len(best_dna.get_active_indicators())}")
        print("\nTop 10 Indicators by Weight:")

        weights = best_dna.get_weights()
        sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        for name, weight in sorted_weights[:10]:
            print(f"  {name:40s}: {weight:+.4f}")

        print("\n" + "=" * 70)

    def generate_live_signals(self, dna: SuperIndicatorDNA = None) -> str:
        """
        Generate live signals using best DNA.

        Args:
            dna: DNA to use (defaults to best from Hall of Fame)

        Returns:
            Formatted signals report
        """
        if dna is None:
            dna = self.hall_of_fame.get_best()

        if dna is None:
            return "No DNA available. Run evolution first."

        return generate_daily_signals(
            dna=dna,
            symbols=self.data_config.symbols,
            portfolio_value=self.portfolio_config.initial_capital
        )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trading Evolution System')
    parser.add_argument('--run', action='store_true', help='Run evolution')
    parser.add_argument('--signals', action='store_true', help='Generate live signals')
    parser.add_argument('--generations', type=int, default=50, help='Number of generations')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--name', type=str, default=None, help='Run name')

    args = parser.parse_args()

    # Create configs
    evolution_config = EvolutionConfig(
        population_size=args.population,
        num_generations=args.generations
    )

    # Create orchestrator
    orchestrator = EvolutionOrchestrator(
        evolution_config=evolution_config
    )

    if args.run:
        best_dna = orchestrator.run_evolution(run_name=args.name)
        print(f"\nEvolution complete. Best DNA: {best_dna.dna_id}")

    if args.signals:
        signals = orchestrator.generate_live_signals()
        print(signals)

    if not args.run and not args.signals:
        parser.print_help()


if __name__ == '__main__':
    main()
