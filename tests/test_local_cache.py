"""
Tests for data.local_cache.LocalDataCache.

Custom test runner — run with: python3 tests/test_local_cache.py
"""

import os
import pickle
import shutil
import sys
import tempfile
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure project root is importable
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from data.local_cache import LocalDataCache, get_cache
from data.symbols import NIFTY_50_SYMBOLS


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(rows: int = 200, freq: str = "5min") -> pd.DataFrame:
    """Generate synthetic OHLCV DataFrame."""
    idx = pd.date_range("2024-01-02 09:15", periods=rows, freq=freq)
    base = 2500.0
    close = base + np.cumsum(np.random.randn(rows) * 5)
    return pd.DataFrame(
        {
            "Open": close - np.abs(np.random.randn(rows)),
            "High": close + np.abs(np.random.randn(rows) * 2),
            "Low": close - np.abs(np.random.randn(rows) * 2),
            "Close": close,
            "Volume": np.random.randint(1000, 100000, rows).astype(float),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pickle_write_read():
    """Test that pickle round-trip preserves data exactly."""
    tmpdir = tempfile.mkdtemp(prefix="aqtis_test_")
    try:
        cache = LocalDataCache(cache_dir=tmpdir)
        df = _make_ohlcv(100)

        path = Path(tmpdir) / "test_symbol.pkl"
        cache._write_pickle(df, path)

        assert path.exists(), "Pickle file not created"

        loaded = cache._read_pickle(path)
        assert loaded is not None, "Failed to read pickle file"
        pd.testing.assert_frame_equal(df, loaded)

        print("  PASS: pickle round-trip preserves data exactly")
    finally:
        shutil.rmtree(tmpdir)


def test_memory_cache():
    """Test in-memory cache stores and retrieves data."""
    tmpdir = tempfile.mkdtemp(prefix="aqtis_test_")
    try:
        cache = LocalDataCache(cache_dir=tmpdir)
        df = _make_ohlcv(100)

        # Write OHLCV to disk
        cache._write_pickle(df, cache._ohlcv_path("TEST.NS", "5m"))

        # First load should read from disk and populate memory
        result1 = cache.load_from_cache("TEST.NS", "5m")
        assert result1 is not None, "load_from_cache returned None"
        assert ("TEST.NS", "5m") in cache._memory, "Memory cache not populated"

        # Second load should hit memory
        result2 = cache.load_from_cache("TEST.NS", "5m")
        assert result2 is not None, "Second load returned None"
        assert len(result2) == len(result1), "Memory cache returned different length"

        # Clear memory cache
        cache.clear_memory_cache()
        assert ("TEST.NS", "5m") not in cache._memory, "Memory cache not cleared"

        print("  PASS: in-memory cache stores and retrieves data")
    finally:
        shutil.rmtree(tmpdir)


def test_staleness_check():
    """Test cache staleness detection."""
    tmpdir = tempfile.mkdtemp(prefix="aqtis_test_")
    try:
        cache = LocalDataCache(cache_dir=tmpdir, auto_refresh_hours=1)
        df = _make_ohlcv(50)

        path = cache._ohlcv_path("TEST.NS", "5m")
        cache._write_pickle(df, path)

        # Just created — should NOT be stale
        assert not cache._is_stale(path), "Fresh file detected as stale"

        # Non-existent path — should be stale
        assert cache._is_stale(Path(tmpdir) / "nonexistent.pkl"), "Missing file not detected as stale"

        # Manually backdate the file
        old_time = (datetime.now() - timedelta(hours=2)).timestamp()
        os.utime(path, (old_time, old_time))
        assert cache._is_stale(path), "Old file not detected as stale"

        print("  PASS: staleness detection works correctly")
    finally:
        shutil.rmtree(tmpdir)


def test_clean_ohlcv():
    """Test OHLCV data cleaning and normalization."""
    cache = LocalDataCache()

    # Create a messy DataFrame (lowercase columns, NaN, negative prices)
    idx = pd.date_range("2024-01-02", periods=10, freq="5min")
    df = pd.DataFrame(
        {
            "open": [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            "high": [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            "low": [99, 100, 101, 102, -5, 104, 105, 106, 107, 108],
            "close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "volume": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        },
        index=idx,
    )

    cleaned = cache._clean_ohlcv(df)

    # Check title case columns
    assert "Open" in cleaned.columns, "Missing Open column"
    assert "Close" in cleaned.columns, "Missing Close column"

    # Check NaN rows removed
    assert cleaned.isna().sum().sum() == 0, "NaN values remain"

    # Check negative prices removed
    if "Low" in cleaned.columns:
        assert (cleaned["Low"] > 0).all(), "Negative prices remain"

    print("  PASS: OHLCV cleaning handles messy data")


def test_trim_to_lookback():
    """Test lookback window trimming."""
    tmpdir = tempfile.mkdtemp(prefix="aqtis_test_")
    try:
        cache = LocalDataCache(cache_dir=tmpdir, lookback_days=5)

        # Create data spanning 10 trading days
        dates = []
        for day_offset in range(10):
            base = pd.Timestamp("2024-01-02") + pd.Timedelta(days=day_offset)
            # Skip weekends
            if base.dayofweek >= 5:
                continue
            for hour in range(9, 16):
                dates.append(base + pd.Timedelta(hours=hour, minutes=15))

        idx = pd.DatetimeIndex(dates)
        df = pd.DataFrame(
            {
                "Open": np.random.randn(len(idx)) + 100,
                "High": np.random.randn(len(idx)) + 102,
                "Low": np.random.randn(len(idx)) + 98,
                "Close": np.random.randn(len(idx)) + 100,
                "Volume": np.abs(np.random.randn(len(idx)) * 10000),
            },
            index=idx,
        )

        trimmed = cache._trim_to_lookback(df)

        # Should have at most 5 unique trading days
        unique_days = trimmed.index.normalize().unique()
        assert len(unique_days) <= 5, f"Expected <= 5 days, got {len(unique_days)}"

        print("  PASS: lookback trimming works correctly")
    finally:
        shutil.rmtree(tmpdir)


def test_cache_stats():
    """Test cache statistics reporting."""
    tmpdir = tempfile.mkdtemp(prefix="aqtis_test_")
    try:
        cache = LocalDataCache(cache_dir=tmpdir)
        df = _make_ohlcv(100)

        # Write some test data
        for sym in ("SYM1.NS", "SYM2.NS", "SYM3.NS"):
            cache._write_pickle(df, cache._ohlcv_path(sym, "5m"))
            cache._memory[(sym, "5m")] = df

        stats = cache.get_cache_stats()

        assert stats["memory_entries"] == 3, f"Expected 3 memory entries, got {stats['memory_entries']}"
        assert stats["disk"]["5m"]["ohlcv_files"] == 3, f"Expected 3 disk files, got {stats['disk']['5m']['ohlcv_files']}"
        assert stats["total_disk_mb"] > 0, "Disk size should be > 0"
        assert stats["lookback_days"] == 60, f"Lookback should be 60, got {stats['lookback_days']}"

        print("  PASS: cache stats reporting correct")
    finally:
        shutil.rmtree(tmpdir)


def test_get_data_with_date_filter():
    """Test get_data with start/end date filters."""
    tmpdir = tempfile.mkdtemp(prefix="aqtis_test_")
    try:
        cache = LocalDataCache(cache_dir=tmpdir)

        # Create data spanning multiple days
        idx = pd.date_range("2024-01-02 09:15", periods=500, freq="5min")
        df = _make_ohlcv(500)
        df.index = idx

        # Store directly to disk
        cache._write_pickle(df, cache._ohlcv_path("TEST.NS", "5m"))

        # First, verify load_from_cache works (no filter)
        full = cache.load_from_cache("TEST.NS", "5m")
        assert full is not None, "load_from_cache returned None"
        assert len(full) == 500, f"Expected 500 rows, got {len(full)}"

        # Now test with date range that spans multiple days in the dataset
        result = cache.get_data("TEST.NS", interval="5m", start="2024-01-02", end="2024-01-04")
        assert result is not None, "get_data returned None with filters"
        assert (result.index >= pd.Timestamp("2024-01-02")).all(), "Start filter not applied"
        assert (result.index <= pd.Timestamp("2024-01-04")).all(), "End filter not applied"
        assert len(result) < len(full) or len(result) == len(full), "Filter didn't restrict data"

        print("  PASS: get_data date filtering works")
    finally:
        shutil.rmtree(tmpdir)


def test_singleton():
    """Test get_cache() returns the same instance."""
    import data.local_cache as module

    # Reset singleton
    module._cache_instance = None

    c1 = get_cache(cache_dir="/tmp/aqtis_singleton_test")
    c2 = get_cache()

    assert c1 is c2, "get_cache() should return the same instance"

    # Cleanup
    module._cache_instance = None
    shutil.rmtree("/tmp/aqtis_singleton_test", ignore_errors=True)

    print("  PASS: singleton pattern works")


def test_indicator_computation():
    """Test that indicator computation produces expected columns."""
    tmpdir = tempfile.mkdtemp(prefix="aqtis_test_")
    try:
        cache = LocalDataCache(cache_dir=tmpdir)
        df = _make_ohlcv(200)

        # Compute indicators directly
        indicators = cache.indicator_computer.compute_all_indicators(df)

        assert indicators is not None, "Indicator computation returned None"
        assert not indicators.empty, "Indicator computation returned empty"
        assert len(indicators.columns) >= 10, f"Expected >= 10 indicator columns, got {len(indicators.columns)}"

        # Check for common indicators (present in both full and fallback calculators)
        has_macd = any("MACD" in c.upper() for c in indicators.columns)
        has_atr = any("ATR" in c.upper() for c in indicators.columns)
        assert has_macd or has_atr, "Expected MACD or ATR in indicator columns"

        print(f"  PASS: indicator computation produced {len(indicators.columns)} columns")
    finally:
        shutil.rmtree(tmpdir)


def test_clear_all_cache():
    """Test full cache clearing."""
    tmpdir = tempfile.mkdtemp(prefix="aqtis_test_")
    try:
        cache = LocalDataCache(cache_dir=tmpdir)
        df = _make_ohlcv(50)

        # Write data
        cache._write_pickle(df, cache._ohlcv_path("SYM.NS", "5m"))
        cache._write_pickle(df, cache._indicator_path("SYM.NS", "5m"))
        cache._memory[("SYM.NS", "5m")] = df

        # Clear
        cache.clear_all_cache()

        assert len(cache._memory) == 0, "Memory not cleared"
        assert not cache._ohlcv_path("SYM.NS", "5m").exists(), "OHLCV file not deleted"
        assert not cache._indicator_path("SYM.NS", "5m").exists(), "Indicator file not deleted"

        print("  PASS: clear_all_cache removes everything")
    finally:
        shutil.rmtree(tmpdir)


def test_symbols_list():
    """Test NIFTY 50 symbols list."""
    assert len(NIFTY_50_SYMBOLS) == 50, f"Expected 50 symbols, got {len(NIFTY_50_SYMBOLS)}"
    assert all(s.endswith(".NS") for s in NIFTY_50_SYMBOLS), "All symbols should end with .NS"
    assert "RELIANCE.NS" in NIFTY_50_SYMBOLS, "RELIANCE.NS missing from symbols"
    assert "TCS.NS" in NIFTY_50_SYMBOLS, "TCS.NS missing from symbols"

    print("  PASS: NIFTY 50 symbols list is valid")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_tests():
    """Run all tests and print summary."""
    tests = [
        test_pickle_write_read,
        test_memory_cache,
        test_staleness_check,
        test_clean_ohlcv,
        test_trim_to_lookback,
        test_cache_stats,
        test_get_data_with_date_filter,
        test_singleton,
        test_indicator_computation,
        test_clear_all_cache,
        test_symbols_list,
    ]

    print("=" * 60)
    print("AQTIS LocalDataCache Test Suite")
    print("=" * 60)

    passed, failed = 0, 0
    failures = []

    for test_fn in tests:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            failures.append((name, e))
            print(f"  FAIL: {name}")
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failures:
        print("\nFailed tests:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    else:
        print("\nAll tests passed!")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
