"""
AQTIS Local Data Cache — Pickle + In-Memory.

Ultra-fast local market data cache for backtesting.
Stores OHLCV data and pre-computed indicators in pickle format
with an in-memory dict cache for instant repeated access.

Usage:
    from data.local_cache import get_cache
    cache = get_cache()
    cache.preload_symbols(interval="5m")
    df = cache.get_data("RELIANCE.NS", interval="5m")
"""

import logging
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data.symbols import NIFTY_50_SYMBOLS

logger = logging.getLogger(__name__)

# Singleton instance
_cache_instance: Optional["LocalDataCache"] = None


def get_cache(
    cache_dir: str = "data/cache",
    lookback_days: int = 60,
    auto_refresh_hours: int = 24,
) -> "LocalDataCache":
    """
    Get or create the singleton LocalDataCache instance.

    Args:
        cache_dir: Root directory for pickle files.
        lookback_days: Rolling window of trading days to keep.
        auto_refresh_hours: Re-fetch if cache older than this many hours.

    Returns:
        The shared LocalDataCache instance.
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = LocalDataCache(
            cache_dir=cache_dir,
            lookback_days=lookback_days,
            auto_refresh_hours=auto_refresh_hours,
        )
    return _cache_instance


class LocalDataCache:
    """
    Pickle-based local data cache with in-memory dict for ultra-fast access.

    Directory layout:
        {cache_dir}/5m/{SYMBOL}.pkl
        {cache_dir}/15m/{SYMBOL}.pkl
        {cache_dir}/indicators/5m/{SYMBOL}_indicators.pkl
        {cache_dir}/indicators/15m/{SYMBOL}_indicators.pkl
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        lookback_days: int = 60,
        auto_refresh_hours: int = 24,
    ):
        self.cache_dir = Path(cache_dir)
        self.lookback_days = lookback_days
        self.auto_refresh_hours = auto_refresh_hours

        # In-memory cache: {(symbol, interval): DataFrame}
        self._memory: Dict[Tuple[str, str], pd.DataFrame] = {}

        # Lazy indicator computer
        self._indicator_computer = None

        # Ensure directory structure
        for interval in ("5m", "15m"):
            (self.cache_dir / interval).mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "indicators" / interval).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lazy indicator computer
    # ------------------------------------------------------------------

    @property
    def indicator_computer(self):
        if self._indicator_computer is None:
            from data.indicator_computer import IndicatorComputer
            self._indicator_computer = IndicatorComputer()
        return self._indicator_computer

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def _ohlcv_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / interval / f"{symbol}.pkl"

    def _indicator_path(self, symbol: str, interval: str) -> Path:
        return self.cache_dir / "indicators" / interval / f"{symbol}_indicators.pkl"

    # ------------------------------------------------------------------
    # Pickle I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _write_pickle(data: pd.DataFrame, path: Path) -> None:
        """Write DataFrame to pickle file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _read_pickle(path: Path) -> Optional[pd.DataFrame]:
        """Read DataFrame from pickle file."""
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return None

    # ------------------------------------------------------------------
    # Cache staleness check
    # ------------------------------------------------------------------

    def _is_stale(self, path: Path) -> bool:
        """Check if a cache file is older than auto_refresh_hours."""
        if not path.exists():
            return True
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        age = datetime.now() - mtime
        return age > timedelta(hours=self.auto_refresh_hours)

    # ------------------------------------------------------------------
    # Public API: fetch_and_cache
    # ------------------------------------------------------------------

    def fetch_and_cache(
        self,
        symbol: str,
        interval: str = "5m",
        force_refresh: bool = False,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from yfinance, cache to pickle, compute indicators.

        Args:
            symbol: Ticker (e.g., 'RELIANCE.NS').
            interval: '5m' or '15m'.
            force_refresh: Re-fetch even if cache is fresh.
            max_retries: Number of yfinance retry attempts.

        Returns:
            OHLCV DataFrame (with indicators merged), or empty DataFrame on failure.
        """
        ohlcv_path = self._ohlcv_path(symbol, interval)

        # Check if we need to refresh
        if not force_refresh and not self._is_stale(ohlcv_path):
            # Load from disk, compute full data if needed
            return self._load_full_data(symbol, interval)

        # Fetch from yfinance
        df = self._fetch_from_yfinance(symbol, interval, max_retries)
        if df is None or df.empty:
            # Fall back to existing cache if available
            existing = self._read_pickle(ohlcv_path)
            if existing is not None and not existing.empty:
                logger.warning(f"Fetch failed for {symbol}, using stale cache")
                return self._load_full_data(symbol, interval)
            return pd.DataFrame()

        # Trim to lookback window
        df = self._trim_to_lookback(df)

        # Write OHLCV pickle
        self._write_pickle(df, ohlcv_path)

        # Compute and store indicators
        self._compute_and_store_indicators(symbol, interval, df)

        # Update memory cache with full data
        full = self._merge_ohlcv_indicators(symbol, interval, df)
        self._memory[(symbol, interval)] = full
        return full

    # ------------------------------------------------------------------
    # Public API: load_from_cache
    # ------------------------------------------------------------------

    def load_from_cache(
        self,
        symbol: str,
        interval: str = "5m",
        use_memory_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Load data from cache (memory first, then disk).

        Args:
            symbol: Ticker symbol.
            interval: '5m' or '15m'.
            use_memory_cache: Check in-memory dict first.

        Returns:
            DataFrame with OHLCV + indicators, or None if not cached.
        """
        key = (symbol, interval)

        # Memory cache hit
        if use_memory_cache and key in self._memory:
            return self._memory[key]

        # Disk cache
        full = self._load_full_data(symbol, interval)
        if full is not None and not full.empty:
            if use_memory_cache:
                self._memory[key] = full
            return full

        return None

    # ------------------------------------------------------------------
    # Public API: preload_symbols
    # ------------------------------------------------------------------

    def preload_symbols(
        self,
        symbols: List[str] = None,
        interval: str = "5m",
        force_refresh: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, bool]:
        """
        Preload multiple symbols into memory cache.

        Fetches from yfinance if stale or missing, otherwise loads from disk.

        Args:
            symbols: List of tickers (default: NIFTY 50).
            interval: '5m' or '15m'.
            force_refresh: Force re-fetch from yfinance.
            show_progress: Print progress bar.

        Returns:
            Dict mapping symbol to success status.
        """
        symbols = symbols or NIFTY_50_SYMBOLS
        results: Dict[str, bool] = {}

        total = len(symbols)
        for i, symbol in enumerate(symbols):
            if show_progress:
                pct = (i + 1) / total * 100
                print(f"\r  [{i+1}/{total}] {pct:.0f}% - {symbol}...", end="", flush=True)

            try:
                ohlcv_path = self._ohlcv_path(symbol, interval)
                if force_refresh or self._is_stale(ohlcv_path):
                    df = self.fetch_and_cache(symbol, interval, force_refresh=force_refresh)
                else:
                    df = self.load_from_cache(symbol, interval)

                results[symbol] = df is not None and not df.empty
            except Exception as e:
                logger.warning(f"Preload failed for {symbol}: {e}")
                results[symbol] = False

            # Rate limiting for yfinance
            if force_refresh or self._is_stale(self._ohlcv_path(symbol, interval)):
                time.sleep(0.5)

        if show_progress:
            success = sum(1 for v in results.values() if v)
            print(f"\n  Done: {success}/{total} symbols loaded.")

        return results

    # ------------------------------------------------------------------
    # Public API: get_data (primary access method)
    # ------------------------------------------------------------------

    def get_data(
        self,
        symbol: str,
        interval: str = "5m",
        start: str = None,
        end: str = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get market data for a symbol. Auto-fetches if not cached.

        This is the main method backtests should use.

        Args:
            symbol: Ticker symbol.
            interval: '5m' or '15m'.
            start: Optional start date filter (YYYY-MM-DD).
            end: Optional end date filter (YYYY-MM-DD).

        Returns:
            DataFrame with OHLCV + indicators, or None.
        """
        # Try memory/disk cache first
        df = self.load_from_cache(symbol, interval)

        # Auto-fetch if missing
        if df is None or df.empty:
            df = self.fetch_and_cache(symbol, interval)
            if df is None or df.empty:
                return None

        # Apply date filters
        if start is not None:
            start_ts = pd.Timestamp(start)
            if start_ts.tzinfo is None and df.index.tz is not None:
                start_ts = start_ts.tz_localize(df.index.tz)
            df = df[df.index >= start_ts]

        if end is not None:
            end_ts = pd.Timestamp(end)
            if end_ts.tzinfo is None and df.index.tz is not None:
                end_ts = end_ts.tz_localize(df.index.tz)
            df = df[df.index <= end_ts]

        return df if not df.empty else None

    # ------------------------------------------------------------------
    # Public API: get_multiple
    # ------------------------------------------------------------------

    def get_multiple(
        self,
        symbols: List[str],
        interval: str = "5m",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple symbols at once.

        Args:
            symbols: List of ticker symbols.
            interval: '5m' or '15m'.

        Returns:
            Dict mapping symbol to DataFrame.
        """
        results: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.get_data(symbol, interval=interval)
            if df is not None and not df.empty:
                results[symbol] = df
        return results

    # ------------------------------------------------------------------
    # Public API: cache management
    # ------------------------------------------------------------------

    def clear_memory_cache(self) -> None:
        """Clear in-memory cache (disk files remain)."""
        count = len(self._memory)
        self._memory.clear()
        logger.info(f"Cleared {count} entries from memory cache")

    def clear_all_cache(self) -> None:
        """Delete all pickle files and clear memory cache."""
        self._memory.clear()
        for interval in ("5m", "15m"):
            for p in (self.cache_dir / interval).glob("*.pkl"):
                p.unlink()
            for p in (self.cache_dir / "indicators" / interval).glob("*.pkl"):
                p.unlink()
        logger.info("All cache cleared (memory + disk)")

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        disk_files = {}
        for interval in ("5m", "15m"):
            ohlcv_files = list((self.cache_dir / interval).glob("*.pkl"))
            ind_files = list((self.cache_dir / "indicators" / interval).glob("*.pkl"))
            disk_files[interval] = {
                "ohlcv_files": len(ohlcv_files),
                "indicator_files": len(ind_files),
            }

        # Total disk size
        total_bytes = sum(
            f.stat().st_size
            for f in self.cache_dir.rglob("*.pkl")
            if f.is_file()
        )

        return {
            "memory_entries": len(self._memory),
            "memory_symbols": sorted(set(k[0] for k in self._memory)),
            "disk": disk_files,
            "total_disk_mb": round(total_bytes / (1024 * 1024), 2),
            "lookback_days": self.lookback_days,
            "auto_refresh_hours": self.auto_refresh_hours,
        }

    # ------------------------------------------------------------------
    # Public API: update_daily
    # ------------------------------------------------------------------

    def update_daily(self) -> dict:
        """
        Daily update: refresh all cached symbols.

        Returns summary report.
        """
        report: Dict[str, dict] = {}

        for interval in ("5m", "15m"):
            # Find symbols already cached on disk
            cached_symbols = [
                p.stem for p in (self.cache_dir / interval).glob("*.pkl")
            ]
            if not cached_symbols:
                cached_symbols = NIFTY_50_SYMBOLS

            updated, failed = 0, []
            for symbol in cached_symbols:
                try:
                    df = self.fetch_and_cache(symbol, interval, force_refresh=True)
                    if df is not None and not df.empty:
                        updated += 1
                    else:
                        failed.append(symbol)
                except Exception as e:
                    logger.warning(f"Update failed for {symbol} {interval}: {e}")
                    failed.append(symbol)
                time.sleep(0.3)

            report[interval] = {"updated": updated, "failed": failed}

        return report

    # ------------------------------------------------------------------
    # Compatibility: MarketDataProvider / PaperTrader interface
    # ------------------------------------------------------------------

    def get_historical(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        years: int = None,
    ) -> Optional[pd.DataFrame]:
        """
        Compatible with MarketDataProvider.get_historical().

        Returns DataFrame with lowercase columns: open, high, low, close, volume
        plus any computed indicators. This is the interface PaperTrader expects.
        """
        df = self.get_data(symbol, interval="5m", start=start_date, end=end_date)
        if df is None or df.empty:
            # Also try 15m interval
            df = self.get_data(symbol, interval="15m", start=start_date, end=end_date)
        if df is None or df.empty:
            return None

        # PaperTrader expects lowercase column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        # Ensure minimum required columns exist
        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            return None

        return df

    # ------------------------------------------------------------------
    # Compatibility: DataFetcher-style interface
    # ------------------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        years: int = 3,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Compatible with trading_evolution DataFetcher.fetch().

        Returns daily OHLCV resampled from 5m data.
        """
        df = self.get_data(symbol, interval="5m", start=start_date, end=end_date)
        if df is None:
            return None

        # Resample to daily for callers expecting daily OHLCV
        ohlcv_cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        if not ohlcv_cols:
            return None

        agg = {}
        if "Open" in df.columns:
            agg["Open"] = "first"
        if "High" in df.columns:
            agg["High"] = "max"
        if "Low" in df.columns:
            agg["Low"] = "min"
        if "Close" in df.columns:
            agg["Close"] = "last"
        if "Volume" in df.columns:
            agg["Volume"] = "sum"

        daily = df[list(agg.keys())].resample("1D").agg(agg).dropna()
        daily.columns = [c.lower() for c in daily.columns]
        return daily

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None,
        years: int = 3,
        use_cache: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """Compatible with trading_evolution DataFetcher.fetch_multiple()."""
        results = {}
        for sym in symbols:
            df = self.fetch(sym, start_date=start_date, end_date=end_date, years=years)
            if df is not None and not df.empty:
                results[sym] = df
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_full_data(
        self, symbol: str, interval: str
    ) -> Optional[pd.DataFrame]:
        """Load OHLCV + indicators from disk pickle files."""
        ohlcv = self._read_pickle(self._ohlcv_path(symbol, interval))
        if ohlcv is None or ohlcv.empty:
            return None
        return self._merge_ohlcv_indicators(symbol, interval, ohlcv)

    def _merge_ohlcv_indicators(
        self, symbol: str, interval: str, ohlcv: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge OHLCV with pre-computed indicators."""
        indicators = self._read_pickle(self._indicator_path(symbol, interval))
        if indicators is None or indicators.empty:
            return ohlcv
        return ohlcv.join(indicators, how="left")

    def _compute_and_store_indicators(
        self, symbol: str, interval: str, ohlcv: pd.DataFrame
    ) -> None:
        """Compute indicators and write to pickle."""
        try:
            indicators = self.indicator_computer.compute_all_indicators(ohlcv)
            if indicators is not None and not indicators.empty:
                self._write_pickle(indicators, self._indicator_path(symbol, interval))
        except Exception as e:
            logger.warning(f"Indicator computation failed for {symbol} {interval}: {e}")

    def _fetch_from_yfinance(
        self, symbol: str, interval: str, max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV from yfinance with exponential backoff."""
        import yfinance as yf

        # yfinance limits intraday to ~60 days
        max_days = 55 if interval in ("5m", "15m") else self.lookback_days
        end = datetime.now()
        start = end - timedelta(days=max_days)

        for attempt in range(1, max_retries + 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    interval=interval,
                )

                if df is None or df.empty:
                    logger.warning(f"No data for {symbol} {interval} (attempt {attempt})")
                    time.sleep(2 ** attempt)
                    continue

                df = self._clean_ohlcv(df)
                if df.empty:
                    time.sleep(2 ** attempt)
                    continue

                logger.debug(f"Fetched {symbol} {interval}: {len(df)} bars")
                return df

            except Exception as e:
                logger.warning(f"Fetch error {symbol} {interval} (attempt {attempt}): {e}")
                time.sleep(2 ** attempt)

        logger.error(f"Failed to fetch {symbol} {interval} after {max_retries} retries")
        return None

    @staticmethod
    def _clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and clean OHLCV data."""
        df = df.copy()

        # Standardize to Title case
        col_map = {}
        for c in df.columns:
            lower = c.lower().replace(" ", "_")
            if lower == "open":
                col_map[c] = "Open"
            elif lower == "high":
                col_map[c] = "High"
            elif lower == "low":
                col_map[c] = "Low"
            elif lower == "close":
                col_map[c] = "Close"
            elif lower == "volume":
                col_map[c] = "Volume"
        df = df.rename(columns=col_map)

        # Keep only OHLCV columns
        keep = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        df = df[keep]

        # Drop NaN, dedup, sort
        df = df.dropna(subset=keep)
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()

        # Ensure numeric
        for c in keep:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()

        # Remove non-positive prices
        price_cols = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
        df = df[(df[price_cols] > 0).all(axis=1)]

        # Clip negative volume
        if "Volume" in df.columns:
            df["Volume"] = df["Volume"].clip(lower=0)

        return df

    def _trim_to_lookback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Trim dataframe to keep only the last lookback_days trading days."""
        if df.empty:
            return df
        dates = df.index.normalize().unique()
        if len(dates) > self.lookback_days:
            cutoff = dates[-self.lookback_days]
            df = df[df.index >= cutoff]
        return df
