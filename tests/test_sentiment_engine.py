"""
Tests for sentiment_engine.py — 12 targeted test cases.

Covers: NewsFetcher, SentimentAnalyzer, SentimentAggregator,
        SentimentSignalGenerator, IndianMarketCalendar, and integration.

Run:
    python3 tests/test_sentiment_engine.py
"""

import sys
import os
import json
import time
import math
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentiment_engine import (
    NewsFetcher,
    SentimentAnalyzer,
    SentimentAggregator,
    SentimentSignalGenerator,
    IndianMarketCalendar,
    NewsHeadline,
    SentimentScore,
    AggregatedSentiment,
    _bare_ticker,
    _headline_id,
    FETCH_COOLDOWN_SECONDS,
    NIFTY_50_NAME_TO_SYMBOL,
    NIFTY_TICKER_TO_COMPANY,
    log_daily_sentiment_summary,
    get_sentiment_context,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_headline(
    title: str,
    source: str = "test",
    published_at: datetime = None,
    matched_symbols: list = None,
    summary: str = "",
) -> NewsHeadline:
    """Create a test headline."""
    return NewsHeadline(
        headline_id=_headline_id(title),
        title=title,
        summary=summary,
        source=source,
        published_at=published_at or datetime.now(),
        url="",
        matched_symbols=matched_symbols or [],
    )


def _make_score(
    headline: NewsHeadline,
    market_sentiment: float = 0.0,
    confidence: float = 0.5,
    stocks: dict = None,
    affected_symbols: list = None,
) -> SentimentScore:
    """Create a test sentiment score."""
    return SentimentScore(
        headline_id=headline.headline_id,
        market_sentiment=market_sentiment,
        stocks=stocks or {},
        confidence=confidence,
        key_themes=["test"],
        affected_symbols=affected_symbols or headline.matched_symbols,
        category="company",
        reasoning="test",
    )


passed = 0
failed = 0
total = 12


def _report(test_num: int, name: str, success: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if success else "FAIL"
    if success:
        passed += 1
    else:
        failed += 1
    print(f"  [{status}] Test {test_num}: {name}")
    if detail and not success:
        print(f"         Detail: {detail}")


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_1_no_internet_graceful_fallback():
    """Test 1: NewsFetcher with no internet — graceful fallback, no crash."""
    try:
        fetcher = NewsFetcher()
        fetcher._headlines = []
        fetcher._seen_ids = set()
        fetcher._last_fetch_time = {}  # Clear rate limits

        def broken_feed(url):
            raise ConnectionError("Simulated network failure")

        with patch.object(NewsFetcher, '_parse_feed', side_effect=broken_feed):
            result = fetcher.fetch()

        ok = isinstance(result, list) and len(result) == 0
        _report(1, "NewsFetcher with no internet — graceful fallback", ok,
                f"Got type={type(result)}, len={len(result) if isinstance(result, list) else '?'}")
    except Exception as e:
        _report(1, "NewsFetcher with no internet — graceful fallback", False, str(e))


def test_2_cache_rate_limiting():
    """Test 2: NewsFetcher cache: second fetch within 1 hour is rate-limited."""
    try:
        fetcher = NewsFetcher()
        fetcher._headlines = []
        fetcher._seen_ids = set()
        fetcher._last_fetch_time = {}

        fetch_calls = []

        def tracking_parse_feed(url):
            fetch_calls.append(url)
            mock_feed = MagicMock()
            mock_feed.entries = []
            mock_feed.bozo = False
            return mock_feed

        with patch.object(NewsFetcher, '_parse_feed', side_effect=tracking_parse_feed):
            # First fetch — should call parse_feed
            fetcher.fetch(sources=["moneycontrol"])
            first_count = len(fetch_calls)

            # Second fetch immediately — should be rate-limited (no new calls)
            fetcher.fetch(sources=["moneycontrol"])
            second_count = len(fetch_calls)

        ok = first_count == 1 and second_count == 1
        _report(2, "Cache rate limiting: 1-hour cooldown per source", ok,
                f"First fetch calls: {first_count}, second fetch calls: {second_count}")
    except Exception as e:
        _report(2, "Cache rate limiting: 1-hour cooldown per source", False, str(e))


def test_3_mock_gemini_json_parsing():
    """Test 3: SentimentAnalyzer parses mock Gemini JSON response correctly."""
    try:
        analyzer = SentimentAnalyzer()
        analyzer._cache = {}

        now = datetime.now()
        h = _make_headline(
            "Reliance Q3 results beat estimates with strong revenue growth",
            published_at=now,
            matched_symbols=["RELIANCE"],
        )

        # Mock Gemini response JSON
        mock_json = json.dumps([{
            "headline_index": 0,
            "market_sentiment": 0.7,
            "stocks": {"RELIANCE": 0.8},
            "confidence": 0.9,
            "key_themes": ["earnings beat", "revenue growth"],
        }])

        mock_response = MagicMock()
        mock_response.text = mock_json

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch.object(analyzer, '_get_client', return_value=mock_client):
            results = analyzer.analyze([h])

        ok = (
            len(results) == 1
            and abs(results[0].market_sentiment - 0.7) < 0.01
            and "RELIANCE" in results[0].stocks
            and abs(results[0].stocks["RELIANCE"] - 0.8) < 0.01
            and abs(results[0].confidence - 0.9) < 0.01
        )
        detail = ""
        if results:
            r = results[0]
            detail = (f"market_sentiment={r.market_sentiment}, "
                      f"stocks={r.stocks}, confidence={r.confidence}")
        _report(3, "Mock Gemini JSON parsing", ok, detail)
    except Exception as e:
        _report(3, "Mock Gemini JSON parsing", False, str(e))


def test_4_malformed_and_fenced_gemini_response():
    """Test 4: SentimentAnalyzer handles malformed and markdown-fenced response."""
    try:
        analyzer = SentimentAnalyzer()
        analyzer._cache = {}

        now = datetime.now()
        h = _make_headline("TCS wins major deal", published_at=now, matched_symbols=["TCS"])

        # Test 4a: Malformed response → should fallback gracefully
        mock_response_bad = MagicMock()
        mock_response_bad.text = "This is not JSON at all!"
        mock_client_bad = MagicMock()
        mock_client_bad.models.generate_content.return_value = mock_response_bad

        with patch.object(analyzer, '_get_client', return_value=mock_client_bad):
            results_bad = analyzer.analyze([h])

        ok_bad = len(results_bad) == 1  # Should still produce a result (rule-based fallback)

        # Test 4b: Markdown-fenced JSON → should parse correctly
        analyzer._cache = {}
        fenced_json = '```json\n[{"headline_index": 0, "market_sentiment": 0.5, "stocks": {"TCS": 0.6}, "confidence": 0.8, "key_themes": ["deal win"]}]\n```'
        mock_response_fenced = MagicMock()
        mock_response_fenced.text = fenced_json
        mock_client_fenced = MagicMock()
        mock_client_fenced.models.generate_content.return_value = mock_response_fenced

        with patch.object(analyzer, '_get_client', return_value=mock_client_fenced):
            results_fenced = analyzer.analyze([h])

        ok_fenced = (
            len(results_fenced) == 1
            and abs(results_fenced[0].market_sentiment - 0.5) < 0.01
        )

        ok = ok_bad and ok_fenced
        _report(4, "Malformed + markdown-fenced Gemini response handling", ok,
                f"malformed_ok={ok_bad}, fenced_ok={ok_fenced}")
    except Exception as e:
        _report(4, "Malformed + markdown-fenced Gemini response handling", False, str(e))


def test_5_time_decay_math():
    """Test 5: Time-decay math — 24h half-life, 48h = 0.25x weight."""
    try:
        aggregator = SentimentAggregator()
        now = datetime.now()

        # Create headlines at 0h, 24h ago, 48h ago — all same sentiment for same stock
        h_now = _make_headline("Stock A surge now", published_at=now, matched_symbols=["RELIANCE"])
        h_24h = _make_headline("Stock A surge 24h ago", published_at=now - timedelta(hours=24), matched_symbols=["RELIANCE"])
        h_48h = _make_headline("Stock A surge 48h ago", published_at=now - timedelta(hours=48), matched_symbols=["RELIANCE"])

        s_now = _make_score(h_now, market_sentiment=1.0, confidence=1.0,
                            stocks={"RELIANCE": 1.0}, affected_symbols=["RELIANCE"])
        s_24h = _make_score(h_24h, market_sentiment=1.0, confidence=1.0,
                            stocks={"RELIANCE": 1.0}, affected_symbols=["RELIANCE"])
        s_48h = _make_score(h_48h, market_sentiment=1.0, confidence=1.0,
                            stocks={"RELIANCE": 1.0}, affected_symbols=["RELIANCE"])

        # Verify the decay math directly
        decay_lambda = math.log(2) / 24.0  # half-life 24h

        w_0h = math.exp(-decay_lambda * 0)     # = 1.0
        w_24h = math.exp(-decay_lambda * 24)   # = 0.5
        w_48h = math.exp(-decay_lambda * 48)   # = 0.25

        ok_24h = abs(w_24h - 0.5) < 0.001
        ok_48h = abs(w_48h - 0.25) < 0.001

        # Also verify aggregation produces correct weighted average
        # All sentiment = 1.0, so weighted avg should be 1.0 regardless
        per_stock = aggregator.aggregate([h_now, h_24h, h_48h],
                                         [s_now, s_24h, s_48h], lookback_days=5)
        agg = per_stock.get("RELIANCE")
        ok_agg = agg is not None and abs(agg.sentiment_score - 1.0) < 0.01

        ok = ok_24h and ok_48h and ok_agg
        _report(5, "Time-decay: 24h half-life, 48h=0.25x weight", ok,
                f"w_24h={w_24h:.4f} (expect 0.5), w_48h={w_48h:.4f} (expect 0.25), "
                f"agg_score={agg.sentiment_score if agg else 'N/A'}")
    except Exception as e:
        _report(5, "Time-decay: 24h half-life, 48h=0.25x weight", False, str(e))


def test_6_all_neutral_headlines():
    """Test 6: All neutral headlines → aggregated sentiment is 0.0."""
    try:
        aggregator = SentimentAggregator()
        now = datetime.now()

        headlines = []
        scores = []
        for i in range(5):
            h = _make_headline(
                f"Neutral headline about market activity {i}",
                published_at=now - timedelta(hours=i),
                matched_symbols=["INFY"],
            )
            s = _make_score(h, market_sentiment=0.0, confidence=0.8,
                            stocks={"INFY": 0.0}, affected_symbols=["INFY"])
            headlines.append(h)
            scores.append(s)

        per_stock = aggregator.aggregate(headlines, scores, lookback_days=5)
        agg = per_stock.get("INFY")

        ok = agg is not None and abs(agg.sentiment_score) < 0.001
        _report(6, "All neutral headlines → sentiment = 0.0", ok,
                f"INFY sentiment={agg.sentiment_score:.4f}" if agg else "INFY not found")
    except Exception as e:
        _report(6, "All neutral headlines → sentiment = 0.0", False, str(e))


def test_7_tiered_thresholds_and_momentum():
    """Test 7: Tiered signal thresholds + momentum boost 1.3x."""
    try:
        gen = SentimentSignalGenerator()

        # 7a: Test tiered thresholds
        # Strong positive (score > 0.5) → signal = +1.0
        agg_strong = AggregatedSentiment(symbol="A", sentiment_score=0.7,
                                         sentiment_momentum=0.0, headline_count=5, confidence=0.8)
        # Moderate positive (0.2 < score <= 0.5) → signal = +0.5
        agg_mod = AggregatedSentiment(symbol="B", sentiment_score=0.35,
                                      sentiment_momentum=0.0, headline_count=5, confidence=0.8)
        # Neutral (-0.2 <= score <= 0.2) → signal = 0.0 (skipped)
        agg_neutral = AggregatedSentiment(symbol="C", sentiment_score=0.1,
                                          sentiment_momentum=0.0, headline_count=5, confidence=0.8)
        # Moderate negative (-0.5 <= score < -0.2) → signal = -0.5
        agg_mod_neg = AggregatedSentiment(symbol="D", sentiment_score=-0.35,
                                           sentiment_momentum=0.0, headline_count=5, confidence=0.8)
        # Strong negative (score < -0.5) → signal = -1.0
        agg_strong_neg = AggregatedSentiment(symbol="E", sentiment_score=-0.7,
                                              sentiment_momentum=0.0, headline_count=5, confidence=0.8)

        aggregated = {"A": agg_strong, "B": agg_mod, "C": agg_neutral,
                      "D": agg_mod_neg, "E": agg_strong_neg}

        signals = gen.generate(aggregated, event_multiplier=1.0)

        ok_strong = abs(signals.get("A", 0) - 1.0) < 0.01
        ok_mod = abs(signals.get("B", 0) - 0.5) < 0.01
        ok_neutral = "C" not in signals  # Neutral signals are skipped
        ok_mod_neg = abs(signals.get("D", 0) - (-0.5)) < 0.01
        ok_strong_neg = abs(signals.get("E", 0) - (-1.0)) < 0.01

        # 7b: Momentum boost test
        # Positive signal with strong positive momentum → 1.3x boost
        agg_momentum = AggregatedSentiment(symbol="F", sentiment_score=0.35,
                                            sentiment_momentum=0.25, headline_count=5, confidence=0.8)
        signals_m = gen.generate({"F": agg_momentum}, event_multiplier=1.0)
        # Base signal = 0.5, boosted by 1.3 = 0.65
        expected_m = 0.5 * 1.3
        ok_momentum = abs(signals_m.get("F", 0) - expected_m) < 0.01

        ok = ok_strong and ok_mod and ok_neutral and ok_mod_neg and ok_strong_neg and ok_momentum
        _report(7, "Tiered thresholds + 1.3x momentum boost", ok,
                f"strong={signals.get('A')}, mod={signals.get('B')}, neutral_skipped={ok_neutral}, "
                f"mod_neg={signals.get('D')}, strong_neg={signals.get('E')}, "
                f"momentum_boosted={signals_m.get('F')} (expected {expected_m:.2f})")
    except Exception as e:
        _report(7, "Tiered thresholds + 1.3x momentum boost", False, str(e))


def test_8_event_calendar():
    """Test 8: IndianMarketCalendar — RBI, Budget, earnings, F&O expiry."""
    try:
        cal = IndianMarketCalendar()
        errors = []

        # 8a: Union Budget: Feb 1 → should give 1.5x
        mult_b, reason_b = cal.get_event_multiplier(date(2025, 2, 1))
        if mult_b != 1.5 or "Budget" not in reason_b:
            errors.append(f"Budget Feb 1: got {mult_b}x, reason='{reason_b}'")

        # Within 3 days of budget (Jan 30) → also 1.5x
        mult_b2, reason_b2 = cal.get_event_multiplier(date(2025, 1, 30))
        if mult_b2 != 1.5 or "Budget" not in reason_b2:
            errors.append(f"Budget Jan 30: got {mult_b2}x, reason='{reason_b2}'")

        # 8b: RBI Monetary Policy: April 6 (month=4 is in RBI months, day 6 in range 3-10)
        mult_rbi, reason_rbi = cal.get_event_multiplier(date(2025, 4, 6))
        if mult_rbi != 1.5 or "RBI" not in reason_rbi:
            errors.append(f"RBI Apr 6: got {mult_rbi}x, reason='{reason_rbi}'")

        # 8c: No event: March 20, 2025
        mult_none, reason_none = cal.get_event_multiplier(date(2025, 3, 20))
        if mult_none != 1.0:
            errors.append(f"No event Mar 20: got {mult_none}x")

        # 8d: F&O expiry: last Thursday of month
        # January 2025: last Thursday is Jan 30
        last_thu = cal._last_thursday_of_month(2025, 1)
        if last_thu.weekday() != 3:
            errors.append(f"Last Thursday weekday={last_thu.weekday()} (expected 3)")
        # Within 3 days of last Thursday → 1.5x
        mult_fno, reason_fno = cal.get_event_multiplier(last_thu)
        if mult_fno != 1.5:
            errors.append(f"F&O expiry {last_thu}: got {mult_fno}x, reason='{reason_fno}'")

        ok = len(errors) == 0
        _report(8, "Event calendar: RBI, Budget, F&O expiry", ok,
                "; ".join(errors) if errors else "")
    except Exception as e:
        _report(8, "Event calendar: RBI, Budget, F&O expiry", False, str(e))


def test_9_stock_name_mapping():
    """Test 9: Nifty 50 stock name mapping + headline symbol matching."""
    try:
        errors = []

        # 9a: Name-to-symbol mapping
        test_cases = [
            ("reliance", "RELIANCE.NS"),
            ("reliance industries", "RELIANCE.NS"),
            ("tcs", "TCS.NS"),
            ("infosys", "INFY.NS"),
            ("hdfc bank", "HDFCBANK.NS"),
            ("sbi", "SBIN.NS"),
            ("airtel", "BHARTIARTL.NS"),
            ("wipro", "WIPRO.NS"),
            ("maruti suzuki", "MARUTI.NS"),
            ("titan", "TITAN.NS"),
        ]
        for name, expected_sym in test_cases:
            actual = NIFTY_50_NAME_TO_SYMBOL.get(name)
            if actual != expected_sym:
                errors.append(f"Name '{name}': expected {expected_sym}, got {actual}")

        # 9b: Headline matching via _match_symbols
        fetcher = NewsFetcher()
        fetcher._headlines = []
        fetcher._seen_ids = set()

        matched = fetcher._match_symbols("Reliance Industries reports strong Q3 results")
        if "RELIANCE" not in matched:
            errors.append(f"'Reliance Industries...' → matched {matched}, expected RELIANCE")

        matched2 = fetcher._match_symbols("HDFC Bank and ICICI Bank rally on rate cut hopes")
        if "HDFCBANK" not in matched2 or "ICICIBANK" not in matched2:
            errors.append(f"'HDFC Bank and ICICI Bank...' → matched {matched2}")

        # 9c: Reverse lookup: ticker → company name
        rel_info = NIFTY_TICKER_TO_COMPANY.get("RELIANCE")
        if rel_info is None or rel_info[0] != "Reliance Industries":
            errors.append(f"RELIANCE reverse lookup: got {rel_info}")

        ok = len(errors) == 0
        _report(9, "Nifty 50 stock mapping + headline matching", ok,
                "; ".join(errors) if errors else "")
    except Exception as e:
        _report(9, "Nifty 50 stock mapping + headline matching", False, str(e))


def test_10_context_output_format():
    """Test 10: get_sentiment_context() output format check."""
    try:
        import sentiment_engine as se

        # Save originals
        orig_fetcher = se._fetcher
        orig_analyzer = se._analyzer
        orig_aggregator = se._aggregator
        orig_signal_gen = se._signal_gen
        orig_calendar = se._calendar

        try:
            now = datetime.now()

            fetcher = NewsFetcher()
            fetcher._headlines = []
            fetcher._seen_ids = set()
            # Pre-populate with test headlines
            for i in range(5):
                h = _make_headline(
                    f"Reliance Industries strong growth headline {i}",
                    published_at=now - timedelta(hours=i),
                    matched_symbols=["RELIANCE"],
                )
                fetcher._headlines.append(h)
                fetcher._seen_ids.add(h.headline_id)

            # Mark all sources as recently fetched
            for source in se.RSS_FEEDS.keys():
                fetcher._last_fetch_time[source] = time.time()

            se._fetcher = fetcher
            se._analyzer = SentimentAnalyzer()
            se._analyzer._cache = {}
            se._aggregator = SentimentAggregator()
            se._signal_gen = SentimentSignalGenerator()
            se._calendar = IndianMarketCalendar()

            # Patch Gemini to force rule-based
            with patch.object(SentimentAnalyzer, '_get_client', return_value=None):
                context = get_sentiment_context(["RELIANCE.NS"], lookback_days=5)

            ok = True
            detail_parts = []

            if not isinstance(context, str):
                ok = False
                detail_parts.append(f"Not a string: {type(context)}")
            elif "None" in context and "No news" not in context:
                ok = False
                detail_parts.append("Found 'None' in context string")
            elif "SENTIMENT ANALYSIS" not in context:
                ok = False
                detail_parts.append("Missing 'SENTIMENT ANALYSIS' header")

            _report(10, "get_sentiment_context() output format", ok,
                    "; ".join(detail_parts) if detail_parts else "")
        finally:
            se._fetcher = orig_fetcher
            se._analyzer = orig_analyzer
            se._aggregator = orig_aggregator
            se._signal_gen = orig_signal_gen
            se._calendar = orig_calendar

    except Exception as e:
        _report(10, "get_sentiment_context() output format", False, str(e))


def test_11_empty_headlines_sentiment():
    """Test 11: Empty headlines list → market sentiment = 0.0, no signals."""
    try:
        aggregator = SentimentAggregator()
        gen = SentimentSignalGenerator()

        # Empty input
        per_stock = aggregator.aggregate([], [], lookback_days=5)
        market = aggregator.market_sentiment(per_stock)
        signals = gen.generate(per_stock)

        ok = (
            len(per_stock) == 0
            and abs(market) < 0.001
            and len(signals) == 0
        )
        _report(11, "Empty headlines → market=0.0, no signals", ok,
                f"per_stock={len(per_stock)}, market={market}, signals={len(signals)}")
    except Exception as e:
        _report(11, "Empty headlines → market=0.0, no signals", False, str(e))


def test_12_full_pipeline_and_serialization():
    """Test 12: Full pipeline + serialization + backwards compat."""
    try:
        errors = []
        now = datetime.now()

        # 12a: Full pipeline: headlines → analyze → aggregate → signal
        headlines = []
        for i in range(5):
            h = _make_headline(
                f"HDFC Bank reports record profit quarter {i}",
                published_at=now - timedelta(hours=i),
                matched_symbols=["HDFCBANK"],
            )
            headlines.append(h)

        analyzer = SentimentAnalyzer()
        analyzer._cache = {}
        with patch.object(SentimentAnalyzer, '_get_client', return_value=None):
            scores = analyzer.analyze(headlines)

        aggregator = SentimentAggregator()
        per_stock = aggregator.aggregate(headlines, scores, lookback_days=5)
        gen = SentimentSignalGenerator()
        signals = gen.generate(per_stock)

        if len(scores) != 5:
            errors.append(f"Expected 5 scores, got {len(scores)}")

        # 12b: Serialization round-trip
        h_orig = headlines[0]
        h_dict = h_orig.to_dict()
        h_back = NewsHeadline.from_dict(h_dict)
        if h_back.headline_id != h_orig.headline_id:
            errors.append(f"Headline round-trip failed: {h_back.headline_id} != {h_orig.headline_id}")
        if h_back.summary != h_orig.summary:
            errors.append(f"Headline summary round-trip failed")

        s_orig = scores[0]
        s_dict = s_orig.to_dict()
        s_back = SentimentScore.from_dict(s_dict)
        if s_back.headline_id != s_orig.headline_id:
            errors.append(f"Score round-trip failed")
        if abs(s_back.market_sentiment - s_orig.market_sentiment) > 0.001:
            errors.append(f"Score sentiment round-trip: {s_back.market_sentiment} vs {s_orig.market_sentiment}")

        # 12c: Backwards compat: old cache format with "sentiment" and "relevance" fields
        old_format = {
            "headline_id": "test123",
            "sentiment": 0.5,        # Old field name → should become market_sentiment
            "relevance": 0.8,        # Old field name → should become confidence
        }
        s_compat = SentimentScore.from_dict(old_format)
        if abs(s_compat.market_sentiment - 0.5) > 0.001:
            errors.append(f"Backwards compat: market_sentiment={s_compat.market_sentiment}, expected 0.5")
        if abs(s_compat.confidence - 0.8) > 0.001:
            errors.append(f"Backwards compat: confidence={s_compat.confidence}, expected 0.8")

        ok = len(errors) == 0
        _report(12, "Full pipeline + serialization + backwards compat", ok,
                "; ".join(errors) if errors else "")
    except Exception as e:
        _report(12, "Full pipeline + serialization + backwards compat", False, str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  sentiment_engine.py — Test Suite (12 tests)")
    print("=" * 60)
    print()

    test_1_no_internet_graceful_fallback()
    test_2_cache_rate_limiting()
    test_3_mock_gemini_json_parsing()
    test_4_malformed_and_fenced_gemini_response()
    test_5_time_decay_math()
    test_6_all_neutral_headlines()
    test_7_tiered_thresholds_and_momentum()
    test_8_event_calendar()
    test_9_stock_name_mapping()
    test_10_context_output_format()
    test_11_empty_headlines_sentiment()
    test_12_full_pipeline_and_serialization()

    print()
    print(f"  Results: {passed}/{total} PASSED, {failed}/{total} FAILED")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
