"""
News Sentiment Signal Layer for the 5-Player Coach Trading System.

Pulls Indian financial news from RSS feeds, analyzes sentiment with Gemini
Flash, aggregates per-stock/sector/market scores, and outputs trading signals
compatible with the player & coach system.

Components:
    NewsFetcher              - RSS feed ingestion with 1-hour caching & dedup
    SentimentAnalyzer        - Gemini batch analysis (10 per call) + rule-based fallback
    SentimentAggregator      - Time-weighted (24h half-life), confidence-weighted aggregation
    SentimentSignalGenerator - Tiered thresholds + momentum overlay
    IndianMarketCalendar     - Event-aware signal boosting (1.5x near key events)

Integration (all optional -- degrades gracefully):
    - Gemini Flash API  (google-genai)
    - feedparser        (RSS parsing)

Usage:
    from sentiment_engine import get_sentiment_context, get_sentiment_signals

    # Inject into AI Coach / Player prompts
    context = get_sentiment_context(["RELIANCE.NS", "TCS.NS"], lookback_days=5)

    # Get trading signals
    signals = get_sentiment_signals(["RELIANCE.NS", "TCS.NS"])
    # Returns: {"RELIANCE": 0.5, "TCS": -0.5}
"""

from __future__ import annotations

import calendar as cal_module
import hashlib
import json
import logging
import math
import os
import ssl
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports -- degrade gracefully
# ---------------------------------------------------------------------------

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

try:
    from google import genai
    from google.genai import types
    _HAS_GENAI = True
except ImportError:
    _HAS_GENAI = False


# ===========================================================================
# Constants
# ===========================================================================

_PROJECT_ROOT = Path(__file__).parent
_NEWS_CACHE_PATH = _PROJECT_ROOT / "data" / "news_cache.json"
_SENTIMENT_CACHE_PATH = _PROJECT_ROOT / "data" / "sentiment_cache.json"

# 1-hour cooldown per source before re-fetching
FETCH_COOLDOWN_SECONDS = 60 * 60

RSS_FEEDS: Dict[str, str] = {
    "moneycontrol":  "https://www.moneycontrol.com/rss/MCtopnews.xml",
    "economictimes": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "livemint":      "https://www.livemint.com/rss/markets",
}

# ---------------------------------------------------------------------------
# Nifty 50 stock mapping (Req 7)
# Common company names / abbreviations -> NSE symbol
# ---------------------------------------------------------------------------

NIFTY_50_NAME_TO_SYMBOL: Dict[str, str] = {
    # Adani group
    "adani enterprises": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",
    "adani port": "ADANIPORTS.NS",
    # Healthcare
    "apollo hospitals": "APOLLOHOSP.NS",
    "apollo hospital": "APOLLOHOSP.NS",
    # Paints
    "asian paints": "ASIANPAINT.NS",
    "asian paint": "ASIANPAINT.NS",
    # Banking
    "axis bank": "AXISBANK.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "bajaj finserv": "BAJAJFINSV.NS",
    "hdfc bank": "HDFCBANK.NS",
    "hdfc life": "HDFCLIFE.NS",
    "icici bank": "ICICIBANK.NS",
    "icici": "ICICIBANK.NS",
    "indusind bank": "INDUSINDBK.NS",
    "indusind": "INDUSINDBK.NS",
    "kotak bank": "KOTAKBANK.NS",
    "kotak mahindra": "KOTAKBANK.NS",
    "state bank of india": "SBIN.NS",
    "sbi": "SBIN.NS",
    "sbi life": "SBILIFE.NS",
    "shriram finance": "SHRIRAMFIN.NS",
    "shriram": "SHRIRAMFIN.NS",
    # Auto
    "bajaj auto": "BAJAJ-AUTO.NS",
    "eicher motors": "EICHERMOT.NS",
    "royal enfield": "EICHERMOT.NS",
    "hero motocorp": "HEROMOTOCO.NS",
    "hero moto": "HEROMOTOCO.NS",
    "mahindra and mahindra": "M&M.NS",
    "mahindra & mahindra": "M&M.NS",
    "maruti suzuki": "MARUTI.NS",
    "maruti": "MARUTI.NS",
    "tata motors": "TATAMOTORS.NS",
    # Defence / PSU
    "bharat electronics": "BEL.NS",
    "bharat petroleum": "BPCL.NS",
    "bpcl": "BPCL.NS",
    "coal india": "COALINDIA.NS",
    "ntpc": "NTPC.NS",
    "ongc": "ONGC.NS",
    "power grid": "POWERGRID.NS",
    "powergrid": "POWERGRID.NS",
    # Telecom
    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",
    # FMCG
    "britannia": "BRITANNIA.NS",
    "hindustan unilever": "HINDUNILVR.NS",
    "hul": "HINDUNILVR.NS",
    "itc": "ITC.NS",
    "itc limited": "ITC.NS",
    "nestle india": "NESTLEIND.NS",
    "nestle": "NESTLEIND.NS",
    "tata consumer": "TATACONSUM.NS",
    "titan": "TITAN.NS",
    "titan company": "TITAN.NS",
    "trent": "TRENT.NS",
    "westside": "TRENT.NS",
    # Pharma
    "cipla": "CIPLA.NS",
    "dr reddy": "DRREDDY.NS",
    "dr. reddy": "DRREDDY.NS",
    "sun pharma": "SUNPHARMA.NS",
    "sun pharmaceutical": "SUNPHARMA.NS",
    # IT
    "hcl tech": "HCLTECH.NS",
    "hcl technologies": "HCLTECH.NS",
    "infosys": "INFY.NS",
    "tata consultancy": "TCS.NS",
    "tcs": "TCS.NS",
    "tech mahindra": "TECHM.NS",
    "wipro": "WIPRO.NS",
    # Metals / Cement / Infra
    "grasim": "GRASIM.NS",
    "hindalco": "HINDALCO.NS",
    "jsw steel": "JSWSTEEL.NS",
    "tata steel": "TATASTEEL.NS",
    "ultratech cement": "ULTRACEMCO.NS",
    "ultratech": "ULTRACEMCO.NS",
    "larsen": "LT.NS",
    "l&t": "LT.NS",
    # Reliance
    "reliance industries": "RELIANCE.NS",
    "reliance": "RELIANCE.NS",
    "ril": "RELIANCE.NS",
}

# Reverse lookup: bare ticker -> (company name, search aliases)
NIFTY_TICKER_TO_COMPANY: Dict[str, Tuple[str, List[str]]] = {
    "ADANIENT":    ("Adani Enterprises", ["adani enterprises"]),
    "ADANIPORTS":  ("Adani Ports", ["adani ports", "adani port"]),
    "APOLLOHOSP":  ("Apollo Hospitals", ["apollo hospitals", "apollo hospital"]),
    "ASIANPAINT":  ("Asian Paints", ["asian paints", "asian paint"]),
    "AXISBANK":    ("Axis Bank", ["axis bank"]),
    "BAJAJ-AUTO":  ("Bajaj Auto", ["bajaj auto"]),
    "BAJFINANCE":  ("Bajaj Finance", ["bajaj finance"]),
    "BAJAJFINSV":  ("Bajaj Finserv", ["bajaj finserv"]),
    "BEL":         ("Bharat Electronics", ["bharat electronics"]),
    "BPCL":        ("Bharat Petroleum", ["bharat petroleum", "bpcl"]),
    "BHARTIARTL":  ("Bharti Airtel", ["bharti airtel", "airtel"]),
    "BRITANNIA":   ("Britannia Industries", ["britannia"]),
    "CIPLA":       ("Cipla", ["cipla"]),
    "COALINDIA":   ("Coal India", ["coal india"]),
    "DRREDDY":     ("Dr Reddys Laboratories", ["dr reddy", "dr. reddy"]),
    "EICHERMOT":   ("Eicher Motors", ["eicher motors", "royal enfield"]),
    "GRASIM":      ("Grasim Industries", ["grasim"]),
    "HCLTECH":     ("HCL Technologies", ["hcl tech", "hcl technologies"]),
    "HDFCBANK":    ("HDFC Bank", ["hdfc bank"]),
    "HDFCLIFE":    ("HDFC Life Insurance", ["hdfc life"]),
    "HEROMOTOCO":  ("Hero MotoCorp", ["hero motocorp", "hero moto"]),
    "HINDALCO":    ("Hindalco Industries", ["hindalco"]),
    "HINDUNILVR":  ("Hindustan Unilever", ["hindustan unilever", "hul"]),
    "ICICIBANK":   ("ICICI Bank", ["icici bank", "icici"]),
    "INDUSINDBK":  ("IndusInd Bank", ["indusind bank", "indusind"]),
    "INFY":        ("Infosys", ["infosys"]),
    "ITC":         ("ITC Limited", ["itc limited", "itc ltd", "itc"]),
    "JSWSTEEL":    ("JSW Steel", ["jsw steel"]),
    "KOTAKBANK":   ("Kotak Mahindra Bank", ["kotak bank", "kotak mahindra"]),
    "LT":          ("Larsen and Toubro", ["larsen", "l&t"]),
    "M&M":         ("Mahindra and Mahindra", ["mahindra & mahindra", "mahindra and mahindra"]),
    "MARUTI":      ("Maruti Suzuki", ["maruti suzuki", "maruti"]),
    "NESTLEIND":   ("Nestle India", ["nestle india", "nestle"]),
    "NTPC":        ("NTPC Limited", ["ntpc"]),
    "ONGC":        ("Oil and Natural Gas Corporation", ["ongc"]),
    "POWERGRID":   ("Power Grid Corporation", ["power grid", "powergrid"]),
    "RELIANCE":    ("Reliance Industries", ["reliance industries", "reliance", "ril"]),
    "SBILIFE":     ("SBI Life Insurance", ["sbi life"]),
    "SHRIRAMFIN":  ("Shriram Finance", ["shriram finance", "shriram"]),
    "SBIN":        ("State Bank of India", ["state bank of india", "sbi"]),
    "SUNPHARMA":   ("Sun Pharmaceutical", ["sun pharma", "sun pharmaceutical"]),
    "TATACONSUM":  ("Tata Consumer Products", ["tata consumer"]),
    "TATAMOTORS":  ("Tata Motors", ["tata motors"]),
    "TATASTEEL":   ("Tata Steel", ["tata steel"]),
    "TCS":         ("Tata Consultancy Services", ["tata consultancy", "tcs"]),
    "TECHM":       ("Tech Mahindra", ["tech mahindra"]),
    "TITAN":       ("Titan Company", ["titan company", "titan"]),
    "TRENT":       ("Trent Limited", ["trent", "westside"]),
    "ULTRACEMCO":  ("UltraTech Cement", ["ultratech cement", "ultratech"]),
    "WIPRO":       ("Wipro", ["wipro"]),
}

# Sector mapping for sector-level sentiment
SECTOR_MAP: Dict[str, str] = {
    "ADANIENT": "Energy", "ADANIPORTS": "Infra", "APOLLOHOSP": "Pharma",
    "ASIANPAINT": "Other", "AXISBANK": "Banking", "BAJAJ-AUTO": "Auto",
    "BAJFINANCE": "Banking", "BAJAJFINSV": "Banking", "BEL": "Other",
    "BPCL": "Energy", "BHARTIARTL": "Telecom", "BRITANNIA": "FMCG",
    "CIPLA": "Pharma", "COALINDIA": "Energy", "DRREDDY": "Pharma",
    "EICHERMOT": "Auto", "GRASIM": "Infra", "HCLTECH": "IT",
    "HDFCBANK": "Banking", "HDFCLIFE": "Banking", "HEROMOTOCO": "Auto",
    "HINDALCO": "Metals", "HINDUNILVR": "FMCG", "ICICIBANK": "Banking",
    "INDUSINDBK": "Banking", "INFY": "IT", "ITC": "FMCG",
    "JSWSTEEL": "Metals", "KOTAKBANK": "Banking", "LT": "Infra",
    "M&M": "Auto", "MARUTI": "Auto", "NESTLEIND": "FMCG",
    "NTPC": "Energy", "ONGC": "Energy", "POWERGRID": "Energy",
    "RELIANCE": "Energy", "SBILIFE": "Banking", "SHRIRAMFIN": "Banking",
    "SBIN": "Banking", "SUNPHARMA": "Pharma", "TATACONSUM": "FMCG",
    "TATAMOTORS": "Auto", "TATASTEEL": "Metals", "TCS": "IT",
    "TECHM": "IT", "TITAN": "FMCG", "TRENT": "FMCG",
    "ULTRACEMCO": "Infra", "WIPRO": "IT",
}


# ===========================================================================
# Utility
# ===========================================================================

def _bare_ticker(symbol: str) -> str:
    """Strip .NS / .BO suffix from a symbol."""
    for suffix in (".NS", ".BO"):
        if symbol.endswith(suffix):
            return symbol[: -len(suffix)]
    return symbol


def _headline_id(title: str) -> str:
    """Generate a stable 12-char hex ID from headline text."""
    return hashlib.md5(title.encode("utf-8", errors="replace")).hexdigest()[:12]


# ===========================================================================
# Data classes
# ===========================================================================

@dataclass
class NewsHeadline:
    """A single news headline with metadata."""
    headline_id: str
    title: str
    summary: str
    source: str
    published_at: datetime
    url: str = ""
    matched_symbols: List[str] = field(default_factory=list)
    fetched_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline_id": self.headline_id,
            "title": self.title,
            "summary": self.summary,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "url": self.url,
            "matched_symbols": self.matched_symbols,
            "fetched_at": self.fetched_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NewsHeadline":
        d = d.copy()
        # Backwards-compat: old cache may lack "summary"
        d.setdefault("summary", "")
        for key in ("published_at", "fetched_at"):
            if isinstance(d.get(key), str):
                d[key] = datetime.fromisoformat(d[key])
        return cls(**d)


@dataclass
class SentimentScore:
    """Sentiment analysis result for one headline (Req 2 output format)."""
    headline_id: str
    market_sentiment: float = 0.0     # -1.0 to +1.0 overall market sentiment
    stocks: Dict[str, float] = field(default_factory=dict)  # {symbol: score}
    confidence: float = 0.0           # 0.0 to 1.0
    key_themes: List[str] = field(default_factory=list)
    # Additional fields for aggregation
    affected_symbols: List[str] = field(default_factory=list)
    category: str = "general"
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "headline_id": self.headline_id,
            "market_sentiment": self.market_sentiment,
            "stocks": self.stocks,
            "confidence": self.confidence,
            "key_themes": self.key_themes,
            "affected_symbols": self.affected_symbols,
            "category": self.category,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SentimentScore":
        d = d.copy()
        # Backwards-compat with old cache format
        if "sentiment" in d and "market_sentiment" not in d:
            d["market_sentiment"] = d.pop("sentiment")
        if "relevance" in d and "confidence" not in d:
            d["confidence"] = d.pop("relevance")
        d.setdefault("stocks", {})
        d.setdefault("key_themes", [])
        d.setdefault("affected_symbols", [])
        d.setdefault("category", "general")
        d.setdefault("reasoning", "")
        # Remove unexpected keys
        valid_keys = {"headline_id", "market_sentiment", "stocks", "confidence",
                      "key_themes", "affected_symbols", "category", "reasoning"}
        d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**d)


@dataclass
class AggregatedSentiment:
    """Per-stock aggregated sentiment (Req 3 output)."""
    symbol: str
    sentiment_score: float = 0.0
    sentiment_momentum: float = 0.0   # improving (+) or deteriorating (-) over 3 days
    headline_count: int = 0
    confidence: float = 0.0
    top_headlines: List[str] = field(default_factory=list)


# ===========================================================================
# 1. NewsFetcher (Req 1)
# ===========================================================================

class NewsFetcher:
    """
    Fetches headlines from Indian financial RSS feeds.

    - Pulls from MoneyControl, Economic Times Markets, LiveMint Markets
    - Parses with feedparser
    - Extracts: title, summary, published_date, source
    - De-duplicates headlines (same title from different sources)
    - Caches results locally to avoid re-fetching within 1 hour
    - Handles connection errors gracefully
    """

    MAX_CACHED_HEADLINES = 500
    MAX_AGE_DAYS = 30

    def __init__(self) -> None:
        self._last_fetch_time: Dict[str, float] = {}
        self._headlines: List[NewsHeadline] = []
        self._seen_ids: set = set()
        self._load_cache()

    # -- Public API --------------------------------------------------------

    def fetch(self, sources: Optional[List[str]] = None) -> List[NewsHeadline]:
        """Fetch from RSS feeds, respecting 1-hour cooldown per source.

        Returns:
            Newly fetched headlines (also merged into internal cache).
        """
        if not _HAS_FEEDPARSER:
            logger.warning("feedparser not installed -- returning cached headlines only")
            return []

        target_sources = sources or list(RSS_FEEDS.keys())
        new_headlines: List[NewsHeadline] = []
        now = time.time()

        for source_name in target_sources:
            url = RSS_FEEDS.get(source_name)
            if not url:
                continue

            # Rate limit: 1-hour cooldown per source
            last = self._last_fetch_time.get(source_name, 0.0)
            if now - last < FETCH_COOLDOWN_SECONDS:
                remaining = int(FETCH_COOLDOWN_SECONDS - (now - last))
                logger.info(f"  [{source_name}] Rate limited -- {remaining}s remaining")
                continue

            try:
                feed = self._parse_feed(url)
                count = 0
                for entry in feed.entries:
                    title = entry.get("title", "").strip()
                    if not title:
                        continue

                    hid = _headline_id(title)
                    if hid in self._seen_ids:
                        continue  # De-duplicate

                    summary = entry.get("summary", "").strip()
                    published = self._parse_published(entry)
                    link = entry.get("link", "")

                    headline = NewsHeadline(
                        headline_id=hid,
                        title=title,
                        summary=summary,
                        source=source_name,
                        published_at=published,
                        url=link,
                        matched_symbols=self._match_symbols(title + " " + summary),
                    )
                    new_headlines.append(headline)
                    self._seen_ids.add(hid)
                    count += 1

                self._last_fetch_time[source_name] = now
                logger.info(f"  [{source_name}] Fetched {count} new headlines")

            except Exception as e:
                # If a feed is down, log warning and continue with others
                logger.warning(f"  [{source_name}] Fetch failed: {e}")

        # Merge into cache
        if new_headlines:
            self._headlines.extend(new_headlines)
            self._headlines.sort(key=lambda h: h.published_at, reverse=True)
            self._trim_cache()
            self._save_cache()
            logger.info(
                f"  Fetched {len(new_headlines)} new headlines, "
                f"{len(self._headlines)} total cached"
            )

        return new_headlines

    def get_headlines(
        self,
        symbols: Optional[List[str]] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[NewsHeadline]:
        """Return cached headlines, optionally filtered by symbols and time."""
        results = self._headlines

        if since is not None:
            results = [h for h in results if h.published_at >= since]

        if symbols is not None:
            bare = {_bare_ticker(s) for s in symbols}
            results = [
                h for h in results
                if not h.matched_symbols or any(s in bare for s in h.matched_symbols)
            ]

        return results[:limit]

    # -- Feed parsing (with SSL fallback for macOS) ------------------------

    @staticmethod
    def _parse_feed(url: str) -> Any:
        """Parse RSS feed with SSL fallback for macOS certificate issues."""
        feed = feedparser.parse(url)
        if feed.entries:
            return feed
        # If empty due to SSL error, fetch with relaxed SSL and parse content
        if feed.bozo:
            try:
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                handler = urllib.request.HTTPSHandler(context=ctx)
                opener = urllib.request.build_opener(handler)
                response = opener.open(url, timeout=15)
                content = response.read()
                feed = feedparser.parse(content)
            except Exception:
                pass
        return feed

    # -- Symbol matching ---------------------------------------------------

    def _match_symbols(self, text: str) -> List[str]:
        """Match headline text against all 50 Nifty tickers."""
        text_lower = text.lower()
        matched: List[str] = []
        for ticker, (_, aliases) in NIFTY_TICKER_TO_COMPANY.items():
            for alias in aliases:
                if len(alias) >= 3 and alias in text_lower:
                    matched.append(ticker)
                    break
        return matched

    # -- Date parsing ------------------------------------------------------

    @staticmethod
    def _parse_published(entry: Any) -> datetime:
        """Extract publication datetime from an RSS entry."""
        if hasattr(entry, "published_parsed") and entry.published_parsed:
            try:
                return datetime(*entry.published_parsed[:6])
            except Exception:
                pass
        if hasattr(entry, "published") and entry.published:
            try:
                return datetime.fromisoformat(
                    entry.published.replace("Z", "+00:00").split("+")[0]
                )
            except Exception:
                pass
        return datetime.now()

    # -- Cache persistence -------------------------------------------------

    def _load_cache(self) -> None:
        if not _NEWS_CACHE_PATH.exists():
            return
        try:
            raw = json.loads(_NEWS_CACHE_PATH.read_text(encoding="utf-8"))
            cutoff = datetime.now() - timedelta(days=self.MAX_AGE_DAYS)
            for item in raw:
                try:
                    h = NewsHeadline.from_dict(item)
                    if h.published_at >= cutoff:
                        self._headlines.append(h)
                        self._seen_ids.add(h.headline_id)
                except Exception:
                    continue
            self._headlines.sort(key=lambda h: h.published_at, reverse=True)
            logger.info(f"Loaded {len(self._headlines)} cached headlines")
        except Exception as e:
            logger.warning(f"Failed to load news cache: {e}")

    def _save_cache(self) -> None:
        try:
            _NEWS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = [h.to_dict() for h in self._headlines]
            _NEWS_CACHE_PATH.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to save news cache: {e}")

    def _trim_cache(self) -> None:
        cutoff = datetime.now() - timedelta(days=self.MAX_AGE_DAYS)
        self._headlines = [h for h in self._headlines if h.published_at >= cutoff]
        self._headlines = self._headlines[: self.MAX_CACHED_HEADLINES]


# ===========================================================================
# 2. SentimentAnalyzer (Req 2)
# ===========================================================================

class SentimentAnalyzer:
    """
    Gemini-powered headline sentiment analysis.

    - Scores each headline for:
        - Overall market sentiment: -1.0 to +1.0
        - Per-stock sentiment: identifies Nifty 50 stocks, scores each
        - Confidence: 0.0 to 1.0
    - Batches headlines (up to 10 per API call) to minimize Gemini calls
    - If Gemini is unavailable, returns neutral (0.0) with confidence 0.0
    """

    BATCH_SIZE = 30   # Larger batches for better coverage in live trading
    MAX_RETRIES = 3

    # Prompt instructs Gemini to return JSON matching Req 2 format
    BATCH_PROMPT = """You are a financial sentiment analyst for the Indian stock market (NSE/Nifty 50).

Analyze each headline below. For EACH headline, return:
- market_sentiment: -1.0 (very bearish) to +1.0 (very bullish) for the overall market
- stocks: a dict mapping NSE ticker symbols (e.g. "RELIANCE", "TCS") to per-stock sentiment scores (-1.0 to +1.0). Only include stocks that are DIRECTLY mentioned or clearly affected.
- confidence: 0.0 to 1.0 (how clearly sentiment-laden is this headline? 0.0 = ambiguous, 1.0 = very clear)
- key_themes: list of 1-3 short theme labels (e.g. "earnings beat", "sector rotation", "rate cut", "FII outflow")

Headlines:
{headlines_block}

Respond ONLY with a JSON array. Each element must have:
{{"headline_index": 0, "market_sentiment": 0.3, "stocks": {{"RELIANCE": 0.5, "TCS": -0.2}}, "confidence": 0.7, "key_themes": ["earnings beat", "sector rotation"]}}

JSON array:"""

    # Keyword lists for rule-based fallback
    POSITIVE_WORDS = frozenset([
        "surge", "gain", "rise", "rally", "beat", "profit", "growth",
        "upgrade", "bullish", "record", "strong", "exceeds", "outperform",
        "dividend", "breakout", "all-time high", "boom", "soar",
    ])
    NEGATIVE_WORDS = frozenset([
        "fall", "drop", "loss", "miss", "decline", "weak", "downgrade",
        "bearish", "concern", "risk", "warning", "crash", "slump",
        "plunge", "sell-off", "selloff", "default", "fraud", "scam",
    ])
    HIGH_IMPACT_WORDS = frozenset([
        "rbi", "rate cut", "rate hike", "budget", "gdp", "earnings",
        "results", "acquisition", "merger", "bankruptcy", "inflation",
        "monetary policy", "fed", "tariff",
    ])

    def __init__(self) -> None:
        self._client = None
        self._cache: Dict[str, SentimentScore] = {}
        self._load_cache()

    # -- Public API --------------------------------------------------------

    def analyze(self, headlines: List[NewsHeadline]) -> List[SentimentScore]:
        """Analyze headlines. Cached results reused, uncached batched to Gemini."""
        results: List[SentimentScore] = []
        uncached: List[NewsHeadline] = []

        for h in headlines:
            if h.headline_id in self._cache:
                results.append(self._cache[h.headline_id])
            else:
                uncached.append(h)

        if uncached:
            # Batch into groups of BATCH_SIZE (10)
            for i in range(0, len(uncached), self.BATCH_SIZE):
                batch = uncached[i : i + self.BATCH_SIZE]
                batch_scores = self._analyze_batch(batch)
                for score in batch_scores:
                    self._cache[score.headline_id] = score
                results.extend(batch_scores)
            self._save_cache()

        # Log all sentiment scores
        for score in results:
            if abs(score.market_sentiment) > 0.3:
                logger.info(
                    f"  Sentiment [{score.headline_id}]: "
                    f"market={score.market_sentiment:+.2f} "
                    f"conf={score.confidence:.2f} "
                    f"stocks={score.stocks} "
                    f"themes={score.key_themes}"
                )

        return results

    # -- Gemini batch analysis ---------------------------------------------

    def _analyze_batch(self, headlines: List[NewsHeadline]) -> List[SentimentScore]:
        """Try Gemini, fall back to rule-based."""
        client = self._get_client()
        if client is not None:
            try:
                return self._analyze_batch_gemini(client, headlines)
            except Exception as e:
                logger.warning(f"Gemini batch analysis failed: {e}")

        # Fallback: return neutral with confidence 0.0 if Gemini unavailable
        return [self._rule_based_fallback(h) for h in headlines]

    def _analyze_batch_gemini(
        self, client: Any, headlines: List[NewsHeadline]
    ) -> List[SentimentScore]:
        """Call Gemini Flash with batch prompt, retry with exponential backoff."""
        # Build prompt
        lines = []
        for idx, h in enumerate(headlines):
            symbols_str = ", ".join(h.matched_symbols) if h.matched_symbols else "market-wide"
            lines.append(f"{idx}. [{h.source}] {h.title}  (detected: {symbols_str})")
        headlines_block = "\n".join(lines)

        prompt = self.BATCH_PROMPT.format(headlines_block=headlines_block)

        # Retry with exponential backoff
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                config = types.GenerateContentConfig(temperature=0.2)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=config,
                )

                response_text = self._extract_response_text(response)
                if response_text:
                    scores = self._parse_batch_response(response_text, headlines)
                    if scores:
                        return scores

            except Exception as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"Gemini call failed (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        logger.warning(f"Gemini failed after {self.MAX_RETRIES} retries: {last_error}")
        return [self._rule_based_fallback(h) for h in headlines]

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        """Extract text from Gemini response (matching gemini_provider.py pattern)."""
        if response is None:
            return ""
        try:
            return response.text
        except Exception:
            try:
                return response.candidates[0].content.parts[0].text
            except Exception:
                return ""

    def _parse_batch_response(
        self, text: str, headlines: List[NewsHeadline]
    ) -> List[SentimentScore]:
        """Parse Gemini's JSON array response into SentimentScore objects."""
        # Strip markdown code fences
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        # Find array boundaries
        start = text.find("[")
        end = text.rfind("]") + 1
        if start < 0 or end <= start:
            logger.warning("No JSON array found in Gemini response")
            return []

        try:
            parsed = json.loads(text[start:end])
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
            return []

        scores: List[SentimentScore] = []
        id_by_index = {i: h.headline_id for i, h in enumerate(headlines)}

        for item in parsed:
            try:
                idx = int(item.get("headline_index", -1))
                hid = id_by_index.get(idx)
                if hid is None:
                    continue

                market_sent = max(-1.0, min(1.0, float(item.get("market_sentiment", 0.0))))
                confidence = max(0.0, min(1.0, float(item.get("confidence", 0.5))))

                # Per-stock sentiment
                raw_stocks = item.get("stocks", {})
                stocks = {}
                if isinstance(raw_stocks, dict):
                    for sym, score in raw_stocks.items():
                        bare = _bare_ticker(str(sym).upper())
                        stocks[bare] = max(-1.0, min(1.0, float(score)))

                key_themes = item.get("key_themes", [])
                if not isinstance(key_themes, list):
                    key_themes = []
                key_themes = [str(t) for t in key_themes[:5]]

                # Determine category
                category = str(item.get("category", "general")).split("|")[0].strip()
                if category not in ("earnings", "macro", "sector", "company",
                                    "geopolitical", "policy", "market", "general"):
                    category = "general"

                affected = list(stocks.keys()) or headlines[idx].matched_symbols

                scores.append(SentimentScore(
                    headline_id=hid,
                    market_sentiment=market_sent,
                    stocks=stocks,
                    confidence=confidence,
                    key_themes=key_themes,
                    affected_symbols=affected,
                    category=category,
                    reasoning="",
                ))
            except Exception:
                continue

        # Fill in any missing headlines with rule-based fallback
        scored_ids = {s.headline_id for s in scores}
        for h in headlines:
            if h.headline_id not in scored_ids:
                scores.append(self._rule_based_fallback(h))

        return scores

    # -- Rule-based fallback -----------------------------------------------

    def _rule_based_fallback(self, headline: NewsHeadline) -> SentimentScore:
        """Keyword-based sentiment scoring when Gemini is unavailable."""
        text_lower = headline.title.lower()

        pos = sum(1 for w in self.POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in self.NEGATIVE_WORDS if w in text_lower)
        high_impact = sum(1 for w in self.HIGH_IMPACT_WORDS if w in text_lower)

        market_sent = max(-1.0, min(1.0, (pos - neg) * 0.25))

        # Per-stock: apply market sentiment to matched symbols
        stocks = {}
        for sym in headline.matched_symbols:
            stocks[sym] = market_sent

        # Confidence: rule-based gets lower confidence
        if high_impact > 0:
            confidence = 0.5
        elif headline.matched_symbols:
            confidence = 0.35
        else:
            confidence = 0.2

        # If Gemini is truly unavailable (not just failed on this batch),
        # return 0.0 with confidence 0.0 per Req 2
        if not _HAS_GENAI and not os.getenv("GEMINI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
            return SentimentScore(
                headline_id=headline.headline_id,
                market_sentiment=0.0,
                stocks={},
                confidence=0.0,
                key_themes=[],
                affected_symbols=headline.matched_symbols,
                category="general",
                reasoning="Gemini unavailable -- neutral fallback",
            )

        # Categorize
        if any(w in text_lower for w in ("earnings", "profit", "revenue", "quarter", "results")):
            category = "earnings"
        elif any(w in text_lower for w in ("rbi", "rate", "gdp", "inflation", "policy", "budget")):
            category = "macro"
        elif any(w in text_lower for w in ("sector", "industry")):
            category = "sector"
        elif headline.matched_symbols:
            category = "company"
        else:
            category = "general"

        # Key themes from keywords
        themes = []
        if pos > 0:
            themes.append("positive keywords")
        if neg > 0:
            themes.append("negative keywords")
        if high_impact > 0:
            themes.append("high-impact event")

        return SentimentScore(
            headline_id=headline.headline_id,
            market_sentiment=market_sent,
            stocks=stocks,
            confidence=confidence,
            key_themes=themes or ["general"],
            affected_symbols=headline.matched_symbols,
            category=category,
            reasoning=f"Rule-based: {pos} positive, {neg} negative keywords",
        )

    # -- Gemini client -----------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not _HAS_GENAI:
            return None
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.info("No GEMINI_API_KEY set -- using rule-based fallback only")
            return None
        try:
            self._client = genai.Client(api_key=api_key)
            return self._client
        except Exception as e:
            logger.warning(f"Failed to initialize Gemini client: {e}")
            return None

    # -- Cache persistence -------------------------------------------------

    def _load_cache(self) -> None:
        if not _SENTIMENT_CACHE_PATH.exists():
            return
        try:
            raw = json.loads(_SENTIMENT_CACHE_PATH.read_text(encoding="utf-8"))
            for item in raw:
                try:
                    s = SentimentScore.from_dict(item)
                    self._cache[s.headline_id] = s
                except Exception:
                    continue
            logger.info(f"Loaded {len(self._cache)} cached sentiment scores")
        except Exception as e:
            logger.warning(f"Failed to load sentiment cache: {e}")

    def _save_cache(self) -> None:
        try:
            _SENTIMENT_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            entries = list(self._cache.values())[-1000:]
            data = [s.to_dict() for s in entries]
            _SENTIMENT_CACHE_PATH.write_text(
                json.dumps(data, indent=2, default=str), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Failed to save sentiment cache: {e}")


# ===========================================================================
# 3. SentimentAggregator (Req 3)
# ===========================================================================

class SentimentAggregator:
    """
    Aggregate per-stock sentiments across multiple headlines.

    - Time-decay weighting: recent headlines weighted more (half-life = 24 hours)
    - Confidence-weighted average (high-confidence headlines count more)
    - Output: dict of {symbol: aggregated_sentiment_score}
    - Tracks sentiment momentum: improving or deteriorating over 3 days
    """

    HALF_LIFE_HOURS = 24.0   # Req 3: 24-hour half-life

    def aggregate(
        self,
        headlines: List[NewsHeadline],
        scores: List[SentimentScore],
        lookback_days: int = 5,
    ) -> Dict[str, AggregatedSentiment]:
        """Per-stock time-weighted, confidence-weighted sentiment aggregation.

        Returns:
            Dict mapping bare ticker to AggregatedSentiment.
        """
        now = datetime.now()
        cutoff = now - timedelta(days=lookback_days)
        decay_lambda = math.log(2) / self.HALF_LIFE_HOURS

        # Build lookups
        headline_map = {h.headline_id: h for h in headlines}
        score_map = {s.headline_id: s for s in scores}

        # Accumulate per-stock
        stock_data: Dict[str, Dict[str, Any]] = {}

        for hid, score in score_map.items():
            headline = headline_map.get(hid)
            if headline is None or headline.published_at < cutoff:
                continue

            # Affected symbols: from per-stock scores first, then from
            # headline-level symbol matching
            affected = list(score.stocks.keys()) if score.stocks else []
            if not affected:
                affected = score.affected_symbols or headline.matched_symbols
            if not affected:
                continue

            hours_ago = max(0.0, (now - headline.published_at).total_seconds() / 3600.0)
            time_weight = math.exp(-decay_lambda * hours_ago)

            for ticker in affected:
                ticker = _bare_ticker(ticker)
                if ticker not in stock_data:
                    stock_data[ticker] = {
                        "weighted_sum": 0.0,
                        "weight_total": 0.0,
                        "headline_count": 0,
                        "conf_sum": 0.0,
                        "headlines": [],
                    }
                sd = stock_data[ticker]

                # Use per-stock score if available, else market sentiment
                stock_sent = score.stocks.get(ticker, score.market_sentiment)
                combined_weight = time_weight * score.confidence
                sd["weighted_sum"] += stock_sent * combined_weight
                sd["weight_total"] += combined_weight
                sd["headline_count"] += 1
                sd["conf_sum"] += score.confidence
                sd["headlines"].append(
                    (abs(stock_sent * score.confidence), headline.title)
                )

        # Build AggregatedSentiment per stock
        result: Dict[str, AggregatedSentiment] = {}
        for ticker, sd in stock_data.items():
            wt = sd["weight_total"]
            count = sd["headline_count"]
            sent = sd["weighted_sum"] / wt if wt > 0 else 0.0

            # Confidence = average confidence scaled by coverage
            avg_conf = sd["conf_sum"] / count if count > 0 else 0.0
            conf = min(1.0, count / 10.0) * avg_conf

            # Top 3 headlines by impact
            sd["headlines"].sort(key=lambda x: x[0], reverse=True)
            top = [title for _, title in sd["headlines"][:3]]

            result[ticker] = AggregatedSentiment(
                symbol=ticker,
                sentiment_score=round(max(-1.0, min(1.0, sent)), 4),
                sentiment_momentum=0.0,
                headline_count=count,
                confidence=round(min(1.0, conf), 4),
                top_headlines=top,
            )

        # Compute 3-day sentiment momentum
        self._compute_momentum(result, headlines, score_map, headline_map, lookback_days)

        return result

    def _compute_momentum(
        self,
        result: Dict[str, AggregatedSentiment],
        headlines: List[NewsHeadline],
        score_map: Dict[str, SentimentScore],
        headline_map: Dict[str, NewsHeadline],
        lookback_days: int,
    ) -> None:
        """Sentiment momentum = avg_sentiment_recent_3d - avg_sentiment_older.

        Positive = improving, negative = deteriorating.
        """
        now = datetime.now()
        recent_cutoff = now - timedelta(days=3)
        full_cutoff = now - timedelta(days=lookback_days)

        for ticker, agg in result.items():
            recent_sum = 0.0
            recent_count = 0
            old_sum = 0.0
            old_count = 0

            for hid, score in score_map.items():
                headline = headline_map.get(hid)
                if headline is None or headline.published_at < full_cutoff:
                    continue

                # Check if this score affects this ticker
                affected_bare = [_bare_ticker(s) for s in
                                 (list(score.stocks.keys()) or score.affected_symbols
                                  or headline.matched_symbols)]
                if ticker not in affected_bare:
                    continue

                stock_sent = score.stocks.get(ticker, score.market_sentiment)

                if headline.published_at >= recent_cutoff:
                    recent_sum += stock_sent
                    recent_count += 1
                else:
                    old_sum += stock_sent
                    old_count += 1

            recent_avg = recent_sum / recent_count if recent_count > 0 else 0.0
            old_avg = old_sum / old_count if old_count > 0 else 0.0
            agg.sentiment_momentum = round(recent_avg - old_avg, 4)

    def market_sentiment(self, per_stock: Dict[str, AggregatedSentiment]) -> float:
        """Equal-weighted average sentiment across all stocks."""
        if not per_stock:
            return 0.0
        scores = [s.sentiment_score for s in per_stock.values()]
        return float(np.mean(scores))

    def sector_sentiment(
        self, per_stock: Dict[str, AggregatedSentiment]
    ) -> Dict[str, float]:
        """Average sentiment grouped by sector."""
        sector_scores: Dict[str, List[float]] = {}
        for ticker, agg in per_stock.items():
            sector = SECTOR_MAP.get(ticker, "Other")
            sector_scores.setdefault(sector, []).append(agg.sentiment_score)

        return {
            sector: round(float(np.mean(scores)), 4)
            for sector, scores in sector_scores.items()
        }


# ===========================================================================
# 4. SentimentSignalGenerator (Req 4)
# ===========================================================================

class SentimentSignalGenerator:
    """
    Convert aggregated sentiment into trading signals.

    Tiered thresholds:
        Strong positive  (>0.5):       bullish signal  (+1.0)
        Moderate positive (0.2 to 0.5): mild bullish   (+0.5)
        Neutral (-0.2 to 0.2):          no signal      (0.0)
        Moderate negative (-0.5 to -0.2): mild bearish  (-0.5)
        Strong negative  (<-0.5):       bearish signal (-1.0)

    Momentum overlay:
        If 3-day trend is strongly positive AND current is positive,
        boost signal by 1.3x.

    Output: dict of {symbol: sentiment_signal}
    """

    MOMENTUM_BOOST = 1.3
    MOMENTUM_THRESHOLD = 0.15   # "strongly positive" 3-day trend

    def generate(
        self,
        aggregated: Dict[str, AggregatedSentiment],
        event_multiplier: float = 1.0,
    ) -> Dict[str, float]:
        """Generate signals: {bare_ticker: signal_value}.

        Signal values are in [-1.0, +1.0] after all overlays.
        """
        signals: Dict[str, float] = {}

        for ticker, agg in aggregated.items():
            score = agg.sentiment_score

            # Tiered threshold mapping (Req 4)
            if score > 0.5:
                signal = 1.0
            elif score > 0.2:
                signal = 0.5
            elif score >= -0.2:
                signal = 0.0
            elif score >= -0.5:
                signal = -0.5
            else:
                signal = -1.0

            # Skip neutral signals
            if signal == 0.0:
                continue

            # Momentum overlay (Req 4):
            # if 3-day trend is strongly positive AND current is positive,
            # boost by 1.3x
            if (agg.sentiment_momentum > self.MOMENTUM_THRESHOLD
                    and signal > 0):
                signal = min(1.0, signal * self.MOMENTUM_BOOST)

            # If 3-day trend is strongly negative AND current is negative,
            # also boost magnitude
            if (agg.sentiment_momentum < -self.MOMENTUM_THRESHOLD
                    and signal < 0):
                signal = max(-1.0, signal * self.MOMENTUM_BOOST)

            # Apply event multiplier to signal weight (Req 6)
            signal = max(-1.0, min(1.0, signal * event_multiplier))

            signals[ticker] = round(signal, 4)

            logger.info(
                f"  Signal [{ticker}]: sent={agg.sentiment_score:+.3f} "
                f"-> signal={signal:+.3f} "
                f"(momentum={agg.sentiment_momentum:+.3f}, "
                f"event_mult={event_multiplier:.1f}x)"
            )

        return signals


# ===========================================================================
# 6. IndianMarketCalendar (Req 6)
# ===========================================================================

class IndianMarketCalendar:
    """
    Event calendar awareness for sentiment signal boosting.

    Key Indian market event types that amplify sentiment importance:
        - RBI Monetary Policy dates (bi-monthly)
        - Union Budget (February 1st)
        - Quarterly earnings seasons (Jan-Feb, Apr-May, Jul-Aug, Oct-Nov)
        - F&O expiry weeks (last Thursday of month)

    When near these events (within 3 days), multiply sentiment signal
    weight by 1.5x.
    """

    # RBI Monetary Policy: bi-monthly, typically first week of even months
    RBI_POLICY_MONTHS = {2, 4, 6, 8, 10, 12}
    RBI_POLICY_DAY_RANGE = (3, 10)

    # Union Budget: Feb 1
    BUDGET_MONTH = 2
    BUDGET_DAY = 1

    # Quarterly earnings seasons
    EARNINGS_SEASONS = [
        (1, 10, 2, 15),    # Q3 results
        (4, 10, 5, 15),    # Q4 results
        (7, 10, 8, 15),    # Q1 results
        (10, 10, 11, 15),  # Q2 results
    ]

    EVENT_MULTIPLIER = 1.5   # Req 6: 1.5x when within 3 days

    def get_event_multiplier(
        self, check_date: Optional[date] = None
    ) -> Tuple[float, str]:
        """Return (multiplier, reason). Highest applicable event wins.

        All events within 3 days get 1.5x multiplier.
        """
        d = check_date or date.today()
        best_mult = 1.0
        best_reason = "No major events"

        # Union Budget -- within 3 days
        try:
            budget_date = date(d.year, self.BUDGET_MONTH, self.BUDGET_DAY)
            if abs((d - budget_date).days) <= 3:
                best_mult = self.EVENT_MULTIPLIER
                best_reason = "Union Budget nearby"
        except ValueError:
            pass

        # RBI Monetary Policy -- within 3 days of policy window
        if d.month in self.RBI_POLICY_MONTHS:
            low, high = self.RBI_POLICY_DAY_RANGE
            if low - 3 <= d.day <= high + 3:
                if self.EVENT_MULTIPLIER > best_mult:
                    best_mult = self.EVENT_MULTIPLIER
                    best_reason = "RBI Monetary Policy nearby"

        # Earnings season
        for m_start, d_start, m_end, d_end in self.EARNINGS_SEASONS:
            try:
                season_start = date(d.year, m_start, d_start)
                season_end = date(d.year, m_end, d_end)
                if season_start <= d <= season_end:
                    if self.EVENT_MULTIPLIER > best_mult:
                        best_mult = self.EVENT_MULTIPLIER
                        best_reason = "Quarterly earnings season"
            except ValueError:
                pass

        # F&O monthly expiry -- within 3 days of last Thursday
        last_thu = self._last_thursday_of_month(d.year, d.month)
        if abs((d - last_thu).days) <= 3:
            if self.EVENT_MULTIPLIER > best_mult:
                best_mult = self.EVENT_MULTIPLIER
                best_reason = "F&O monthly expiry week"

        return best_mult, best_reason

    @staticmethod
    def _last_thursday_of_month(year: int, month: int) -> date:
        """Compute the last Thursday of a given month."""
        last_day = cal_module.monthrange(year, month)[1]
        d = date(year, month, last_day)
        while d.weekday() != 3:  # Thursday = 3
            d -= timedelta(days=1)
        return d


# ===========================================================================
# 8. Daily Sentiment Summary Logger (Req 8)
# ===========================================================================

def log_daily_sentiment_summary(
    per_stock: Dict[str, AggregatedSentiment],
    market_sent: float,
    signals: Dict[str, float],
) -> str:
    """
    Log daily sentiment summary with top 5 most positive and negative stocks.

    Returns the summary string (also logged at INFO level).
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    sorted_stocks = sorted(
        per_stock.values(), key=lambda s: s.sentiment_score, reverse=True
    )
    top_positive = sorted_stocks[:5]
    top_negative = sorted_stocks[-5:][::-1] if len(sorted_stocks) > 5 else []
    # Only include actually negative ones
    top_negative = [s for s in top_negative if s.sentiment_score < 0]

    lines = [
        f"{'=' * 60}",
        f"  DAILY SENTIMENT SUMMARY -- {now}",
        f"{'=' * 60}",
        f"  Market Sentiment:  {market_sent:+.3f}",
        f"  Stocks Covered:    {len(per_stock)}",
        f"  Signals Generated: {len(signals)}",
        f"",
        f"  Top 5 Most Positive:",
    ]

    for s in top_positive:
        lines.append(
            f"    {s.symbol:14s}  {s.sentiment_score:+.3f}  "
            f"({s.headline_count} headlines, momentum={s.sentiment_momentum:+.3f})"
        )

    if top_negative:
        lines.append(f"")
        lines.append(f"  Top 5 Most Negative:")
        for s in top_negative:
            lines.append(
                f"    {s.symbol:14s}  {s.sentiment_score:+.3f}  "
                f"({s.headline_count} headlines, momentum={s.sentiment_momentum:+.3f})"
            )

    if signals:
        lines.append(f"")
        lines.append(f"  Active Signals:")
        for sym, sig in sorted(signals.items(), key=lambda x: -abs(x[1])):
            direction = "BUY " if sig > 0 else "SELL"
            lines.append(f"    {sym:14s}  {direction}  signal={sig:+.3f}")

    lines.append(f"{'=' * 60}")

    summary = "\n".join(lines)
    logger.info(summary)
    return summary


# ===========================================================================
# 5. Integration Interfaces (Req 5)
# ===========================================================================

_fetcher: Optional[NewsFetcher] = None
_analyzer: Optional[SentimentAnalyzer] = None
_aggregator: Optional[SentimentAggregator] = None
_signal_gen: Optional[SentimentSignalGenerator] = None
_calendar: Optional[IndianMarketCalendar] = None


def _ensure_initialized() -> None:
    """Lazy-initialize all components."""
    global _fetcher, _analyzer, _aggregator, _signal_gen, _calendar
    if _fetcher is None:
        _fetcher = NewsFetcher()
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    if _aggregator is None:
        _aggregator = SentimentAggregator()
    if _signal_gen is None:
        _signal_gen = SentimentSignalGenerator()
    if _calendar is None:
        _calendar = IndianMarketCalendar()


def _run_pipeline(
    symbols: List[str], lookback_days: int = 5
) -> Tuple[Dict[str, AggregatedSentiment], float, str]:
    """Run the full fetch -> analyze -> aggregate pipeline.

    Returns:
        (per_stock_sentiment, event_multiplier, event_reason)
    """
    _ensure_initialized()

    # Fetch latest headlines
    _fetcher.fetch()

    # Get headlines for requested symbols
    since = datetime.now() - timedelta(days=lookback_days)
    headlines = _fetcher.get_headlines(symbols=symbols, since=since, limit=200)

    # Also get unfiltered headlines for market context
    all_recent = _fetcher.get_headlines(since=since, limit=200)
    seen = {h.headline_id for h in headlines}
    for h in all_recent:
        if h.headline_id not in seen:
            headlines.append(h)
            seen.add(h.headline_id)

    # Analyze
    scores = _analyzer.analyze(headlines)

    # Aggregate
    per_stock = _aggregator.aggregate(headlines, scores, lookback_days)

    # Event multiplier
    mult, reason = _calendar.get_event_multiplier()

    return per_stock, mult, reason


def get_sentiment_context(
    symbols: List[str], lookback_days: int = 5
) -> str:
    """
    Get formatted sentiment context for injection into player/coach LLM prompts.

    Returns a multi-line string describing market/sector/stock sentiment,
    suitable for appending to any LLM prompt.
    """
    _ensure_initialized()
    per_stock, mult, event_reason = _run_pipeline(symbols, lookback_days)

    if not per_stock:
        return (
            f"SENTIMENT ANALYSIS (last {lookback_days} days):\n"
            "  No news headlines available for analysis.\n"
        )

    market = _aggregator.market_sentiment(per_stock)
    sectors = _aggregator.sector_sentiment(per_stock)
    signals = _signal_gen.generate(per_stock, event_multiplier=mult)

    market_label = "BULLISH" if market > 0.1 else "BEARISH" if market < -0.1 else "NEUTRAL"

    lines = [
        f"SENTIMENT ANALYSIS (last {lookback_days} days):",
        f"  Market Sentiment: {market:+.3f} ({market_label})",
        f"  Event Context:    {event_reason} ({mult}x signal boost)",
        "",
    ]

    # Top bullish / bearish stocks
    sorted_stocks = sorted(per_stock.values(), key=lambda s: s.sentiment_score, reverse=True)
    bullish = [s for s in sorted_stocks if s.sentiment_score > 0.05]
    bearish = [s for s in sorted_stocks if s.sentiment_score < -0.05]

    if bullish:
        top_b = ", ".join(
            f"{s.symbol}({s.sentiment_score:+.2f}, {s.headline_count}h, "
            f"mom={s.sentiment_momentum:+.2f})"
            for s in bullish[:5]
        )
        lines.append(f"  Top Bullish: {top_b}")

    if bearish:
        top_br = ", ".join(
            f"{s.symbol}({s.sentiment_score:+.2f}, {s.headline_count}h, "
            f"mom={s.sentiment_momentum:+.2f})"
            for s in bearish[:5]
        )
        lines.append(f"  Top Bearish: {top_br}")

    # Active sentiment signals
    if signals:
        sig_str = ", ".join(
            f"{sym}={sig:+.1f}" for sym, sig
            in sorted(signals.items(), key=lambda x: -abs(x[1]))[:8]
        )
        lines.append(f"  Signals:     {sig_str}")

    # Sector sentiment
    if sectors:
        sector_str = " | ".join(
            f"{sec}: {score:+.2f}"
            for sec, score in sorted(sectors.items(), key=lambda x: -x[1])
        )
        lines.append(f"  Sectors:     {sector_str}")

    # Key headlines
    all_headlines_flat = []
    for agg in sorted_stocks[:10]:
        for title in agg.top_headlines[:1]:
            all_headlines_flat.append((abs(agg.sentiment_score), title, agg.symbol))
    all_headlines_flat.sort(key=lambda x: x[0], reverse=True)

    if all_headlines_flat:
        lines.append("")
        lines.append("  Key Headlines:")
        for _, title, sym in all_headlines_flat[:5]:
            lines.append(f"    [{sym}] {title[:80]}")

    return "\n".join(lines)


def get_sentiment_signals(symbols: List[str]) -> Dict[str, float]:
    """
    Get sentiment signals per symbol.

    Returns: dict of {symbol: sentiment_signal} where signal is in [-1.0, +1.0]
    """
    _ensure_initialized()
    per_stock, mult, _ = _run_pipeline(symbols)
    return _signal_gen.generate(per_stock, event_multiplier=mult)


def get_market_sentiment() -> float:
    """
    Get overall market sentiment score in [-1.0, 1.0].
    """
    _ensure_initialized()
    _fetcher.fetch()
    since = datetime.now() - timedelta(days=5)
    headlines = _fetcher.get_headlines(since=since, limit=200)
    if not headlines:
        return 0.0
    scores = _analyzer.analyze(headlines)
    per_stock = _aggregator.aggregate(headlines, scores, lookback_days=5)
    return _aggregator.market_sentiment(per_stock)


# ===========================================================================
# Demo / self-test
# ===========================================================================

def _demo() -> None:
    """End-to-end demonstration: fetch, analyze, dashboard, prompt injection."""
    print("=" * 70)
    print("  Sentiment Engine -- News-Driven Sentiment for Nifty 50")
    print("=" * 70)

    # 1. Fetch headlines
    print("\n[1] Fetching headlines from RSS feeds...")
    fetcher = NewsFetcher()
    new_headlines = fetcher.fetch()
    all_headlines = fetcher.get_headlines(limit=50)
    print(f"    Fetched {len(new_headlines)} new  |  {len(all_headlines)} total cached")

    if all_headlines:
        print(f"\n    Recent headlines:")
        for h in all_headlines[:8]:
            syms = ", ".join(h.matched_symbols) or "market"
            print(f"      [{h.source:14s}] {h.title[:70]}  -> {syms}")

    # 2. Analyze sentiment
    print(f"\n[2] Analyzing {len(all_headlines)} headlines...")
    analyzer = SentimentAnalyzer()
    scores = analyzer.analyze(all_headlines)
    gemini_count = sum(1 for s in scores if s.confidence > 0.5)
    rule_count = len(scores) - gemini_count
    print(f"    Gemini: {gemini_count}  |  Rule-based: {rule_count}")

    if scores:
        print(f"\n    Sample scores:")
        for s in scores[:5]:
            print(
                f"      [{s.headline_id}] market={s.market_sentiment:+.2f} "
                f"conf={s.confidence:.2f} stocks={s.stocks} themes={s.key_themes}"
            )

    # 3. Aggregate
    print("\n[3] Aggregating per-stock sentiment...")
    aggregator = SentimentAggregator()
    per_stock = aggregator.aggregate(all_headlines, scores, lookback_days=5)

    # 4. Dashboard
    print(f"\n{'_' * 60}")
    print("  SENTIMENT DASHBOARD")
    print(f"{'_' * 60}")

    market = aggregator.market_sentiment(per_stock)
    label = "BULLISH" if market > 0.1 else "BEARISH" if market < -0.1 else "NEUTRAL"
    print(f"  Market Sentiment:  {market:+.3f}  ({label})")

    # Sector breakdown
    sectors = aggregator.sector_sentiment(per_stock)
    if sectors:
        print(f"\n  Sector Breakdown:")
        for sec, score in sorted(sectors.items(), key=lambda x: -x[1]):
            bar_len = int(abs(score) * 25)
            sign = "+" if score >= 0 else "-"
            bar = "#" * bar_len
            print(f"    {sec:12s}  {score:+.3f}  {sign}{bar}")

    # Top bullish / bearish stocks
    sorted_stocks = sorted(per_stock.values(), key=lambda s: s.sentiment_score, reverse=True)
    bullish = [s for s in sorted_stocks if s.sentiment_score > 0]
    bearish = [s for s in sorted_stocks if s.sentiment_score < 0]

    if bullish:
        print(f"\n  Top Bullish:")
        for s in bullish[:5]:
            print(
                f"    {s.symbol:14s}  {s.sentiment_score:+.3f}  "
                f"({s.headline_count} headlines, conf={s.confidence:.2f}, "
                f"momentum={s.sentiment_momentum:+.3f})"
            )

    if bearish:
        print(f"\n  Top Bearish:")
        for s in bearish[:5]:
            print(
                f"    {s.symbol:14s}  {s.sentiment_score:+.3f}  "
                f"({s.headline_count} headlines, conf={s.confidence:.2f}, "
                f"momentum={s.sentiment_momentum:+.3f})"
            )

    # 5. Event calendar
    calendar = IndianMarketCalendar()
    mult, reason = calendar.get_event_multiplier()
    print(f"\n  Event Calendar:    {reason} (multiplier={mult}x)")

    # 6. Trading signals (tiered thresholds)
    print(f"\n{'_' * 60}")
    print("  TRADING SIGNALS (Tiered Thresholds)")
    print(f"{'_' * 60}")
    signal_gen = SentimentSignalGenerator()
    signals = signal_gen.generate(per_stock, event_multiplier=mult)
    if signals:
        for sym, sig in sorted(signals.items(), key=lambda x: -abs(x[1])):
            dir_label = "BUY " if sig > 0 else "SELL"
            strength = "STRONG" if abs(sig) >= 0.8 else "MODERATE" if abs(sig) >= 0.4 else "MILD"
            print(f"    {sym:14s}  {dir_label}  signal={sig:+.3f}  ({strength})")
    else:
        print("    No signals generated (insufficient data or all neutral)")

    # 7. Prompt injection example
    print(f"\n{'_' * 60}")
    print("  PROMPT INJECTION EXAMPLE (for AI Coach / Player)")
    print(f"{'_' * 60}")
    top_symbols = [
        "RELIANCE.NS", "HDFCBANK.NS", "INFY.NS", "TCS.NS", "ICICIBANK.NS",
        "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "TATASTEEL.NS", "MARUTI.NS",
    ]
    context = get_sentiment_context(top_symbols, lookback_days=5)
    print(context)

    # 8. Daily sentiment summary log
    summary = log_daily_sentiment_summary(per_stock, market, signals)
    print(f"\n{summary}")

    print(f"\n{'=' * 70}")
    print("  Usage:")
    print("    from sentiment_engine import get_sentiment_context, get_sentiment_signals")
    print("    context = get_sentiment_context(['RELIANCE.NS', 'TCS.NS'])")
    print("    signals = get_sentiment_signals(['RELIANCE.NS', 'TCS.NS'])")
    print("    market  = get_market_sentiment()")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
