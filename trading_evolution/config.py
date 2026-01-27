"""
Configuration module for the Paper Trading AI Agent System.

Contains all hyperparameters, settings, and configurable values.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    """Data fetching and storage configuration."""
    symbols: List[str] = field(default_factory=lambda: [
        # Nifty 50 - Top 30 Indian Stocks (NSE)
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS', 'ITC.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'HCLTECH.NS',
        'SUNPHARMA.NS', 'TITAN.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'ULTRACEMCO.NS',
        'NESTLEIND.NS', 'TATAMOTORS.NS', 'POWERGRID.NS', 'NTPC.NS', 'TECHM.NS',
        'M&M.NS', 'TATASTEEL.NS', 'INDUSINDBK.NS', 'ONGC.NS', 'JSWSTEEL.NS'
    ])
    data_years: int = 3
    period: str = "3y"  # yfinance period string
    train_pct: float = 0.67  # ~2 years
    validation_pct: float = 0.165  # ~6 months
    holdout_pct: float = 0.165  # ~6 months
    cache_dir: Path = field(default_factory=lambda: Path("data_cache"))


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_risk_per_trade: float = 0.02  # 2% max risk per trade
    max_concurrent_positions: int = 5
    max_position_pct: float = 0.20  # Max 20% per position
    max_portfolio_risk: float = 0.10  # Max 10% total portfolio risk
    atr_stop_multiplier: float = 2.0  # 2x ATR for stop loss
    min_risk_reward_ratio: float = 1.5  # Minimum 1.5:1 R:R
    slippage_pct: float = 0.001  # 0.1% slippage
    commission_per_trade: float = 0.0  # No commission for simplicity


@dataclass
class PortfolioConfig:
    """Portfolio configuration."""
    initial_capital: float = 100_000.0
    allow_short: bool = True
    allow_position_flip: bool = True  # Can flip from long to short directly


@dataclass
class IndicatorConfig:
    """Indicator calculation configuration."""
    warmup_period: int = 50  # Bars to skip for indicator warmup
    normalization_method: str = "adaptive"  # adaptive, minmax, zscore
    rolling_window: int = 252  # 1 year for rolling stats
    clip_outliers: bool = True
    outlier_std: float = 3.0


@dataclass
class SignalConfig:
    """Signal thresholds for trading decisions."""
    long_entry_threshold: float = 0.7
    long_exit_threshold: float = 0.3
    short_entry_threshold: float = -0.7
    short_exit_threshold: float = -0.3


@dataclass
class EvolutionConfig:
    """Genetic algorithm configuration."""
    population_size: int = 100  # Large population for diversity
    num_generations: int = 100  # Extended generations for deep optimization
    elite_size: int = 8  # Preserve more top performers
    tournament_size: int = 5  # Stronger selection pressure
    crossover_prob: float = 0.75  # Slightly higher crossover
    mutation_prob: float = 0.40  # Higher mutation for exploration
    mutation_strength: float = 0.25  # Stronger mutations
    validation_frequency: int = 15  # Validate every 15 generations
    convergence_threshold: float = 0.0001  # Very low - don't stop early
    max_indicators_active: int = 40  # Allow more complex combinations
    min_indicators_active: int = 15  # Require minimum complexity
    rollback_threshold: float = 0.85  # Stricter validation


@dataclass
class FitnessConfig:
    """Fitness function weights - optimized for high win rate + profit."""
    net_profit_weight: float = 0.30  # Profit matters but not everything
    sharpe_ratio_weight: float = 0.25  # Risk-adjusted returns
    max_drawdown_weight: float = 0.15  # Penalize large drawdowns
    consistency_weight: float = 0.30  # HEAVILY weight win rate consistency
    min_trades_for_validity: int = 50  # Require more trades for statistical significance
    min_win_rate: float = 0.55  # Minimum 55% win rate required
    max_acceptable_drawdown: float = 0.20  # Tighter drawdown limit


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_path: Path = field(default_factory=lambda: Path("trading_evolution.db"))
    wal_mode: bool = True  # Write-Ahead Logging for concurrency


@dataclass
class ReportingConfig:
    """Reporting and visualization configuration."""
    output_dir: Path = field(default_factory=lambda: Path("reports"))
    save_charts: bool = True
    chart_format: str = "png"
    dpi: int = 150


@dataclass
class Config:
    """Main configuration container."""
    name: str = "TradingEvolution"
    data: DataConfig = field(default_factory=DataConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    signals: SignalConfig = field(default_factory=SignalConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)

    def __post_init__(self):
        """Ensure directories exist."""
        self.data.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reporting.output_dir.mkdir(parents=True, exist_ok=True)


def load_config(config_path: str = None) -> Config:
    """
    Load configuration from YAML file or use defaults.

    Args:
        config_path: Path to YAML config file (optional)

    Returns:
        Config object with all settings
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        # Merge with defaults
        config = Config()
        # Update nested configs from YAML
        if 'data' in yaml_config:
            for key, value in yaml_config['data'].items():
                setattr(config.data, key, value)
        if 'risk' in yaml_config:
            for key, value in yaml_config['risk'].items():
                setattr(config.risk, key, value)
        if 'portfolio' in yaml_config:
            for key, value in yaml_config['portfolio'].items():
                setattr(config.portfolio, key, value)
        if 'indicators' in yaml_config:
            for key, value in yaml_config['indicators'].items():
                setattr(config.indicators, key, value)
        if 'signals' in yaml_config:
            for key, value in yaml_config['signals'].items():
                setattr(config.signals, key, value)
        if 'evolution' in yaml_config:
            for key, value in yaml_config['evolution'].items():
                setattr(config.evolution, key, value)
        if 'fitness' in yaml_config:
            for key, value in yaml_config['fitness'].items():
                setattr(config.fitness, key, value)
        if 'database' in yaml_config:
            for key, value in yaml_config['database'].items():
                setattr(config.database, key, value)
        if 'reporting' in yaml_config:
            for key, value in yaml_config['reporting'].items():
                setattr(config.reporting, key, value)
        return config
    return Config()


def save_config(config: Config, config_path: str):
    """Save configuration to YAML file."""
    config_dict = {
        'name': config.name,
        'data': {
            'symbols': config.data.symbols,
            'data_years': config.data.data_years,
            'train_pct': config.data.train_pct,
            'validation_pct': config.data.validation_pct,
            'holdout_pct': config.data.holdout_pct,
        },
        'risk': {
            'max_risk_per_trade': config.risk.max_risk_per_trade,
            'max_concurrent_positions': config.risk.max_concurrent_positions,
            'atr_stop_multiplier': config.risk.atr_stop_multiplier,
            'slippage_pct': config.risk.slippage_pct,
        },
        'portfolio': {
            'initial_capital': config.portfolio.initial_capital,
            'allow_short': config.portfolio.allow_short,
        },
        'evolution': {
            'population_size': config.evolution.population_size,
            'num_generations': config.evolution.num_generations,
            'elite_size': config.evolution.elite_size,
            'crossover_prob': config.evolution.crossover_prob,
            'mutation_prob': config.evolution.mutation_prob,
        },
    }
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)


# Default configuration instance
DEFAULT_CONFIG = Config()
