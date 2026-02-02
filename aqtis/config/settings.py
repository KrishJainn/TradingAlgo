"""
AQTIS Configuration System.

Dataclass-based configuration with YAML loading.
Wraps existing trading_evolution configs where applicable.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import os
import yaml


@dataclass
class SystemConfig:
    """Core system settings."""
    mode: str = "simulation"
    log_level: str = "INFO"
    data_dir: Path = field(default_factory=lambda: Path("data_cache"))
    db_path: Path = field(default_factory=lambda: Path("aqtis.db"))
    vector_db_path: Path = field(default_factory=lambda: Path("aqtis_vectors"))


@dataclass
class RiskLimits:
    """Risk management limits."""
    max_position_size: float = 0.10
    max_portfolio_leverage: float = 2.0
    max_daily_loss: float = -0.05
    max_drawdown: float = -0.15
    max_correlated_exposure: float = 0.30
    min_prediction_confidence: float = 0.60


@dataclass
class BacktestConfig:
    """Backtesting settings."""
    instant_backtest_lookback_days: int = 30
    rolling_window_days: int = 90
    min_similar_trades: int = 20
    max_similar_trades: int = 50


@dataclass
class ModelConfig:
    """ML model ensemble settings."""
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "lstm": 0.25,
        "random_forest": 0.20,
        "linear_regression": 0.15,
        "rules_based": 0.40,
    })
    retraining_frequency: str = "weekly"
    min_training_samples: int = 100
    sequence_length: int = 60


@dataclass
class MemoryConfig:
    """Memory layer settings."""
    vector_db: str = "chromadb"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_similar_trades: int = 50
    research_relevance_threshold: float = 0.6


@dataclass
class LLMConfig:
    """LLM provider settings."""
    provider: str = "gemini"
    model: str = "gemini-2.0-flash"
    temperature: float = 0.3
    max_tokens: int = 4000
    timeout_seconds: int = 30
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    )


@dataclass
class MarketDataConfig:
    """Market data settings."""
    primary_provider: str = "yahoo"
    symbols: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "ITC.NS",
        "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "HCLTECH.NS",
        "SUNPHARMA.NS", "TITAN.NS", "BAJFINANCE.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    ])
    data_years: int = 3
    cache_ttl_hours: int = 24


@dataclass
class ExecutionConfig:
    """Trade execution settings."""
    order_type: str = "market"
    slippage_estimate: float = 0.001
    initial_capital: float = 100_000.0


@dataclass
class RegimeConfig:
    """Market regime detection settings."""
    n_regimes: int = 5
    lookback_days: int = 60
    update_frequency: str = "daily"


@dataclass
class AQTISConfig:
    """Main AQTIS configuration container."""
    system: SystemConfig = field(default_factory=SystemConfig)
    risk: RiskLimits = field(default_factory=RiskLimits)
    backtesting: BacktestConfig = field(default_factory=BacktestConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    market_data: MarketDataConfig = field(default_factory=MarketDataConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)

    def __post_init__(self):
        self.system.data_dir = Path(self.system.data_dir)
        self.system.db_path = Path(self.system.db_path)
        self.system.vector_db_path = Path(self.system.vector_db_path)
        self.system.data_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of issues."""
        issues = []
        if not self.llm.api_key:
            issues.append("LLM API key not set (GEMINI_API_KEY or GOOGLE_API_KEY)")
        if self.risk.max_daily_loss > 0:
            issues.append("max_daily_loss should be negative (e.g., -0.05)")
        if self.risk.max_drawdown > 0:
            issues.append("max_drawdown should be negative (e.g., -0.15)")
        if not self.market_data.symbols:
            issues.append("No symbols configured")
        return issues


def load_config(config_path: str = None) -> AQTISConfig:
    """
    Load configuration from YAML file or use defaults.

    Searches for config in order:
    1. Provided path
    2. ./aqtis_config.yaml
    3. Defaults
    """
    search_paths = []
    if config_path:
        search_paths.append(Path(config_path))
    search_paths.append(Path("aqtis_config.yaml"))

    for path in search_paths:
        if path.exists():
            return _load_from_yaml(path)

    return AQTISConfig()


def _load_from_yaml(path: Path) -> AQTISConfig:
    """Parse YAML file into AQTISConfig."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    config = AQTISConfig()

    _update_dataclass(config.system, raw.get("system", {}))
    _update_dataclass(config.risk, raw.get("risk", {}))
    _update_dataclass(config.backtesting, raw.get("backtesting", {}))
    _update_dataclass(config.models, raw.get("models", {}))
    _update_dataclass(config.memory, raw.get("memory", {}))
    _update_dataclass(config.llm, raw.get("llm", {}))
    _update_dataclass(config.market_data, raw.get("market_data", {}))
    _update_dataclass(config.execution, raw.get("execution", {}))
    _update_dataclass(config.regime, raw.get("regime", {}))

    config.__post_init__()
    return config


def _update_dataclass(obj, data: dict):
    """Update dataclass fields from a dictionary."""
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
