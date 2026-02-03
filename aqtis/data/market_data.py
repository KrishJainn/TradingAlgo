"""
AQTIS Market Data Provider.

Wraps trading_evolution.data.fetcher.DataFetcher for market data access.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Add parent directory to path so we can import trading_evolution
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


class MarketDataProvider:
    """
    Market data provider wrapping trading_evolution's DataFetcher.

    Provides historical and quote data with caching.
    Supports a fast local cache mode via data_cache.CacheManager for
    backtesting without network calls.
    """

    def __init__(
        self,
        cache_dir: str = "data_cache",
        data_years: int = 3,
        use_local_cache: bool = False,
    ):
        self.cache_dir = Path(cache_dir)
        self.data_years = data_years
        self._fetcher = None

        # Fast local cache for backtesting
        self._use_local_cache = use_local_cache
        self._local_provider = None
        if use_local_cache:
            self._init_local_cache()

    def _init_local_cache(self):
        """Set up the local BacktestDataProvider from data_cache module."""
        try:
            from data_cache.cache_manager import CacheManager
            from data_cache.backtest_data_provider import BacktestDataProvider

            cm = CacheManager(str(self.cache_dir))
            self._local_provider = BacktestDataProvider(
                cm, fallback_to_yfinance=True
            )
            logger.info("MarketDataProvider: local cache enabled")
        except Exception as e:
            logger.warning(f"Local cache init failed, falling back to network: {e}")
            self._use_local_cache = False

    @property
    def fetcher(self):
        """Lazy-initialize the underlying DataFetcher."""
        if self._fetcher is None:
            try:
                from trading_evolution.data.fetcher import DataFetcher
                from trading_evolution.data.cache import DataCache
                cache = DataCache(self.cache_dir)
                self._fetcher = DataFetcher(cache=cache)
            except ImportError:
                logger.warning(
                    "trading_evolution not available, using standalone yfinance fetcher"
                )
                self._fetcher = _StandaloneFetcher(self.cache_dir)
        return self._fetcher

    def get_historical(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        years: int = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for a symbol.

        Args:
            symbol: Stock ticker (e.g., 'RELIANCE.NS').
            start_date: Start date (YYYY-MM-DD).
            end_date: End date (YYYY-MM-DD).
            years: Number of years of history (default from config).

        Returns:
            DataFrame with columns: open, high, low, close, volume.
        """
        if self._use_local_cache and self._local_provider:
            df = self._local_provider.fetch(
                symbol, start_date=start_date, end_date=end_date,
                years=years or self.data_years,
            )
            if df is not None:
                return df

        return self.fetcher.fetch(
            symbol,
            start_date=start_date,
            end_date=end_date,
            years=years or self.data_years,
        )

    def get_bulk_data(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        years: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        Returns:
            Dictionary mapping symbol to DataFrame.
        """
        if self._use_local_cache and self._local_provider:
            result = self._local_provider.fetch_multiple(
                symbols, start_date=start_date, end_date=end_date,
                years=years or self.data_years,
            )
            if result:
                return result

        return self.fetcher.fetch_multiple(
            symbols,
            start_date=start_date,
            end_date=end_date,
            years=years or self.data_years,
        )

    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get latest quote for a symbol.

        Returns dict with: price, change, change_pct, volume, timestamp.
        """
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return {
                "symbol": symbol,
                "price": getattr(info, "last_price", None),
                "previous_close": getattr(info, "previous_close", None),
                "market_cap": getattr(info, "market_cap", None),
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.warning(f"Failed to get quote for {symbol}: {e}")
            return None

    def validate_data(self, df: pd.DataFrame) -> tuple:
        """Validate data quality. Returns (is_valid, issues)."""
        if hasattr(self.fetcher, "validate_data"):
            return self.fetcher.validate_data(df)
        issues = []
        if df is None or df.empty:
            return False, ["No data"]
        if len(df) < 60:
            issues.append(f"Only {len(df)} bars")
        return len(issues) == 0, issues


class _StandaloneFetcher:
    """Fallback fetcher when trading_evolution is not available."""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir

    def fetch(self, symbol: str, start_date=None, end_date=None, years=3, use_cache=True):
        import yfinance as yf
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

        try:
            df = yf.Ticker(symbol).history(start=start_date, end=end_date, auto_adjust=False)
            if df is None or df.empty:
                return None
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            required = ["open", "high", "low", "close", "volume"]
            df = df.dropna(subset=[c for c in required if c in df.columns])
            df = df.sort_index()
            return df
        except Exception as e:
            logger.error(f"Fetch error for {symbol}: {e}")
            return None

    def fetch_multiple(self, symbols, **kwargs):
        results = {}
        for s in symbols:
            df = self.fetch(s, **kwargs)
            if df is not None and not df.empty:
                results[s] = df
        return results
