# 5-Player Trading Coach System
## Product Requirements Document (PRD) v1.0

---

**Organization**: Claflin Investments
**Date**: February 2026
**Status**: MVP Complete
**Repository**: [github.com/KrishJainn/claflin-investments](https://github.com/KrishJainn/claflin-investments)

---

## 1. Executive Summary

The 5-Player Trading Coach System is an AI-powered quantitative trading platform that runs five independent trading "players" — each with a distinct personality and indicator set — coached by a Gemini-powered AI that continuously optimizes their strategies based on performance.

The system trades Indian equities (Nifty 30) on 5-minute intraday bars, with each player evolving independently. A RAG-enhanced knowledge layer provides contextual trading wisdom from ingested books, notes, and strategies.

**Core Philosophy**: Five diverse strategies competing and evolving, guided by AI coaching that learns what works for each personality type.

---

## 2. System Architecture

```
                    +----------------------+
                    |  Streamlit Dashboard |
                    |  (Continuous Backtest|
                    |   + Knowledge Base)  |
                    +----------+-----------+
                               |
                    +----------v-----------+
                    |      AI Coach        |
                    |   (Gemini 3 Flash)   |
                    +----------+-----------+
                               |
         +----------+----------+----------+----------+
         |          |          |          |          |
    +----v----+ +---v----+ +---v----+ +---v----+ +---v----+
    | Player 1| |Player 2| |Player 3| |Player 4| |Player 5|
    |Aggressive| |Conserv.| |Balanced| |VolBreak| |Momentum|
    +---------+ +--------+ +--------+ +--------+ +---------+
         |          |          |          |          |
         +----------+----------+----------+----------+
                               |
                    +----------v-----------+
                    |   Trading Evolution  |
                    |  (80+ Indicators,    |
                    |   Backtest Engine)   |
                    +----------+-----------+
                               |
                    +----------v-----------+
                    |   Knowledge Layer    |
                    |  (ChromaDB + RAG)    |
                    +----------------------+
```

---

## 3. The Five Players

Each player has a distinct trading personality with a curated indicator set:

### Player 1: Aggressive
**Risk Profile**: High
**Style**: Momentum trader seeking quick gains
**Indicators** (12):
- RSI_7, STOCH_5_3, TSI_13_25, CMO_14, WILLR_14
- OBV, MFI_14, ADX_14, DEMA_20, NATR_14, TRUERANGE, NATR_21

### Player 2: Conservative
**Risk Profile**: Low
**Style**: Trend follower with strict risk management
**Indicators** (11):
- ADX_14, SUPERTREND_7_3, EMA_50, AROON_14, CMF_20
- RSI_14, BBANDS_20_2, OBV, VWMA_20, HMA_9, CMO_14

### Player 3: Balanced
**Risk Profile**: Medium
**Style**: Diversified approach across multiple signal types
**Indicators** (12):
- RSI_14, BBANDS_20_2, STOCH_14_3, CMF_20, ZSCORE_20
- MFI_20, DEMA_20, ADX_14, ATR_14, TEMA_20, CCI_14, PSAR

### Player 4: VolBreakout
**Risk Profile**: Medium-High
**Style**: Volatility breakout specialist
**Indicators** (12):
- NATR_14, KC_20_2, ADX_14, BBANDS_20_2, ATR_14
- CCI_14, RSI_7, OBV, CMF_20, NVI, VWAP, ADX_21

### Player 5: Momentum
**Risk Profile**: High
**Style**: Pure momentum rider
**Indicators** (12):
- RSI_7, TSI_13_25, MACD_12_26_9, CMO_14, STOCH_5_3
- COPPOCK, ROC_20, ROC_10, MOM_10, DEMA_20, KAMA_10, T3_10

---

## 4. AI Coach

### Overview
The AI Coach uses **Gemini 3 Flash Preview** to analyze each player's performance and optimize their configuration every 3 trading days.

**Location**: `coach_system/coaches/ai_coach.py`

### Coaching Cycle

```
Performance Data (wins, losses, P&L)
           |
           v
+---------------------+
|   AI Coach (LLM)    |
|   - Analyze trades  |
|   - Query knowledge |
|   - Generate advice |
+---------------------+
           |
           v
+---------------------+
| Optimization Output |
| - Add indicators    |
| - Remove indicators |
| - Adjust weights    |
| - Tune thresholds   |
| - Adjust hold time  |
+---------------------+
           |
           v
    Player Config Updated
```

### What the Coach Optimizes

| Parameter | Range | Description |
|-----------|-------|-------------|
| Indicator Set | From 80+ available | Which indicators to use |
| Indicator Weights | 0.1 - 1.0 | Importance of each indicator |
| Entry Threshold | 0.3 - 0.8 | Signal strength to enter |
| Exit Threshold | 0.2 - 0.7 | Signal strength to exit |
| Min Hold Bars | 1 - 10 | Minimum bars before exit allowed |

### RAG Integration
The coach queries the Knowledge Layer for relevant trading wisdom before making decisions:
- Aggressive/Momentum players → Technical analysis + strategies
- Conservative players → Risk management + fundamental analysis
- Balanced players → Technical analysis + risk management + psychology

### Fallback Mode
When the LLM is unavailable, the coach uses statistical adjustments:
- Increase weights on indicators correlated with winning trades
- Decrease weights on indicators correlated with losing trades
- Tighten thresholds after losses, loosen after wins

---

## 5. Knowledge Layer (RAG)

### Overview
A ChromaDB-powered retrieval system that provides contextual trading wisdom to the AI Coach.

**Location**: `knowledge_layer/`

### Components

```
knowledge_layer/
├── embeddings/           # ChromaDB vector store
├── config/
│   └── settings.yaml     # RAG configuration
├── ingestion/
│   └── document_loader.py  # PDF, EPUB, MD, TXT support
├── context_layer.py      # Query interface
└── retriever.py          # Search + ranking
```

### Supported Document Types
- **PDF** - Trading books, research papers
- **EPUB** - E-books
- **Markdown** - Personal notes, strategies
- **Text** - Plain text documents

### Categories
Documents are tagged with categories for targeted retrieval:
- `trading_books` - Published trading literature
- `personal_notes` - User's own observations
- `strategies` - Documented trading strategies
- `risk_management` - Risk-related content
- `technical_analysis` - TA concepts and patterns
- `market_psychology` - Behavioral finance
- `fundamental_analysis` - Fundamental concepts

### Configuration
```yaml
embedding:
  model: all-MiniLM-L6-v2
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  top_k: 5
  use_mmr: true
  mmr_diversity: 0.3
```

### Key Methods
- `get_context_for_player(player_type)` - Player-specific wisdom
- `get_risk_management_context(query)` - Risk-focused retrieval
- `get_technical_analysis_context(query)` - TA-focused retrieval
- `ingest_document(path, category)` - Add new documents

---

## 6. Trading Evolution Framework

### Indicator Universe (80+)

**Momentum** (14):
RSI, MACD, Stochastic, CCI, CMO, TSI, Williams %R, ROC, MOM, Ultimate Oscillator, Awesome Oscillator, KST, Coppock, PPO

**Trend** (10):
ADX, Aroon, SuperTrend, PSAR, Vortex, Linear Regression, DPO, TRIX, Mass Index, Chande Kroll Stop

**Volatility** (12):
ATR, NATR, Bollinger Bands, Keltner Channels, Donchian Channels, True Range, Ulcer Index, RVI, Historical Volatility, Chaikin Volatility, Standard Deviation, Variance

**Volume** (10):
OBV, Accumulation/Distribution, ADOSC, CMF, MFI, EFI, NVI, PVI, VWAP, Volume SMA

**Overlap** (15+):
EMA, SMA, WMA, DEMA, TEMA, HMA, VWMA, KAMA, T3, ZLEMA, FWMA, SWMA, VIDYA, McGinley Dynamic, Supertrend

### Backtest Engine

**Fixed Parameters**:
| Setting | Value |
|---------|-------|
| Backtest Duration | 50 trading days |
| Symbols | Nifty 30 stocks |
| Capital per Player | ₹100,000 |
| Bar Interval | 5-minute intraday |
| Bars per Day | 26 |
| Max Position Size | 20% of capital |
| Max Concurrent Positions | 5 |
| Stop Loss | 2x ATR |
| Take Profit | 3x ATR |
| Position Types | Long and Short |

### Signal Generation
```
composite_signal = Σ (indicator_value × indicator_weight)
                   ────────────────────────────────────────
                            Σ indicator_weights

if composite_signal > entry_threshold → ENTER
if composite_signal < exit_threshold  → EXIT
```

---

## 7. Persistent Learning

### Best Config Tracking
Each player's best-performing configuration is saved independently in `evolved_player_configs.json`:

```json
{
  "PLAYER_1": {
    "best_pnl": 15420.50,
    "best_config": {
      "weights": {"RSI_7": 0.85, "STOCH_5_3": 0.72, ...},
      "entry_threshold": 0.55,
      "exit_threshold": 0.35,
      "min_hold_bars": 3
    },
    "timestamp": "2026-02-03T15:30:00"
  },
  ...
}
```

**Update Rule**: A player's config is only saved when they beat their **own** previous best P&L. This ensures each player evolves on their own terms rather than copying others.

---

## 8. Dashboard

### Technology
- **Framework**: Streamlit
- **Charts**: Plotly
- **Theme**: Custom dark theme

### Pages

#### 1. Continuous Backtest
**Path**: `coach_system/dashboard/pages/continuous_backtest.py`

**Features**:
- Run multi-day backtests with AI coach optimization
- Configure number of runs and coaching interval
- Real-time progress tracking
- Per-player P&L visualization
- Cumulative equity curves
- Personal best tracking per player
- Run history with expandable details

**Controls**:
- Number of backtest runs (1-100)
- Coaching interval (days between optimizations)
- Start/Stop buttons

#### 2. Knowledge Base
**Path**: `coach_system/dashboard/pages/knowledge_base.py`

**Features**:
- View ingested document statistics
- Query the knowledge base with natural language
- Ingest new documents:
  - Drag & drop upload
  - File path input
  - Directory batch ingestion
- Category tagging
- Document management

#### 3. Paper Trading (Placeholder)
Coming soon - live paper trading with real-time market data.

### Running the Dashboard
```bash
python run_dashboard.py
# or
streamlit run coach_system/dashboard/app.py
```

---

## 9. Data Infrastructure

### Market Data

| Setting | Value |
|---------|-------|
| Market | NSE India |
| Symbols | Nifty 30 stocks |
| Interval | 5-minute bars |
| Source | yfinance |
| Cache | Local pickle files |

### Symbol Universe (Nifty 30)
```
RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, ICICIBANK.NS,
HINDUNILVR.NS, ITC.NS, SBIN.NS, BHARTIARTL.NS, BAJFINANCE.NS,
KOTAKBANK.NS, LT.NS, HCLTECH.NS, AXISBANK.NS, ASIANPAINT.NS,
MARUTI.NS, SUNPHARMA.NS, TITAN.NS, ULTRACEMCO.NS, WIPRO.NS,
ADANIPORTS.NS, NESTLEIND.NS, NTPC.NS, ONGC.NS, POWERGRID.NS,
JSWSTEEL.NS, TATASTEEL.NS, TECHM.NS, BRITANNIA.NS, M&M.NS
```

### Cache Structure
```
data/
├── cache/
│   └── 5m/                    # Raw OHLCV data
│       ├── RELIANCE.NS.pkl
│       ├── TCS.NS.pkl
│       └── ...
└── cache/indicators/
    └── 5m/                    # Pre-computed indicators
        ├── RELIANCE.NS_indicators.pkl
        └── ...
```

### Databases
| Database | Purpose | Size |
|----------|---------|------|
| trading_evolution.db | Trade history, player states | ~2.7 MB |
| trading_memory.db | Context and memory | ~90 KB |

---

## 10. Configuration

### AI Coach Config
```python
# In ai_coach.py
LLM_MODEL = "gemini-3-flash-preview"
COACHING_INTERVAL = 3  # days between coaching sessions
```

### Knowledge Layer Config (`knowledge_layer/config/settings.yaml`)
```yaml
embedding:
  model: all-MiniLM-L6-v2
  device: cpu

chunking:
  chunk_size: 500
  chunk_overlap: 50

retrieval:
  top_k: 5
  use_mmr: true
  mmr_diversity: 0.3

player_categories:
  aggressive: [technical_analysis, strategies]
  conservative: [risk_management, fundamental_analysis]
  balanced: [technical_analysis, risk_management, market_psychology]
  momentum: [technical_analysis, strategies]
  volbreakout: [technical_analysis, volatility]
```

### Environment Variables
| Variable | Required | Description |
|----------|----------|-------------|
| GEMINI_API_KEY | Yes | Google Gemini API key |
| GOOGLE_API_KEY | Alt | Alternative Gemini key |

---

## 11. Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.11+ |
| LLM | Google Gemini 3 Flash Preview |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Market Data | yfinance |
| Indicators | pandas-ta (80+ indicators) |
| Dashboard | Streamlit + Plotly |
| Document Parsing | pypdf, ebooklib, beautifulsoup4 |

---

## 12. File Structure

```
claflin-investments/
├── coach_system/
│   ├── __init__.py
│   ├── coaches/
│   │   ├── __init__.py
│   │   └── ai_coach.py          # Gemini-powered AI Coach
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py               # Streamlit entry point
│   │   ├── theme.py             # Custom dark theme
│   │   └── pages/
│   │       ├── continuous_backtest.py
│   │       └── knowledge_base.py
│   └── llm/
│       ├── __init__.py
│       ├── base.py              # LLM provider abstraction
│       └── gemini_provider.py   # Gemini implementation
│
├── knowledge_layer/
│   ├── __init__.py
│   ├── context_layer.py         # RAG query interface
│   ├── retriever.py             # Search + ranking
│   ├── config/
│   │   └── settings.yaml
│   ├── embeddings/              # ChromaDB storage
│   └── ingestion/
│       └── document_loader.py
│
├── trading_evolution/
│   ├── indicators/
│   │   ├── calculator.py        # 80+ indicator calculations
│   │   └── normalizer.py        # Indicator normalization
│   ├── backtest/
│   │   └── engine.py            # Backtesting engine
│   ├── player/
│   │   ├── trading_player.py    # Player implementation
│   │   └── risk_manager.py      # Position sizing
│   ├── data/
│   │   └── fetcher.py           # yfinance wrapper
│   └── config.py                # Trading parameters
│
├── data/
│   └── cache/                   # Market data + indicator cache
│
├── evolved_player_configs.json  # Best configs per player
├── run_dashboard.py             # Dashboard launcher
├── run_5player_simulation.py    # CLI backtest runner
└── README.md
```

---

## 13. Usage

### Running a Backtest (CLI)
```bash
python run_5player_simulation.py --runs 10 --coaching-interval 3
```

### Running the Dashboard
```bash
python run_dashboard.py
```
Then open http://localhost:8501

### Ingesting Knowledge
```python
from knowledge_layer.context_layer import KnowledgeContext

ctx = KnowledgeContext()
ctx.ingest_document("/path/to/trading_book.pdf", category="trading_books")
```

### Querying Knowledge
```python
results = ctx.get_context_for_player("aggressive", "momentum trading strategies")
```

---

## 14. Performance Tracking

### Metrics Tracked Per Player
- Total P&L (₹)
- Win Rate (%)
- Number of Trades
- Average Trade Return
- Max Drawdown
- Sharpe Ratio (calculated)
- Personal Best P&L

### Coaching Effectiveness
- P&L before/after coaching
- Indicator changes per session
- Weight adjustments over time

---

## 15. Future Roadmap

### Phase 2: Live Trading
- Zerodha Kite API integration
- Real-time market data streaming
- Order execution and management
- Position monitoring

### Phase 3: Advanced Features
- Multi-market support (US, Crypto)
- Options trading strategies
- Portfolio-level optimization
- Cross-player signal aggregation

### Phase 4: Enterprise
- Multi-user support
- Role-based access control
- Audit logging
- Cloud deployment (AWS/GCP)

---

## 16. Key Differentiators

1. **Five Personalities**: Not one strategy — five diverse approaches competing and evolving
2. **Independent Evolution**: Each player optimizes against their own historical best
3. **RAG-Enhanced Coaching**: AI decisions informed by curated trading knowledge
4. **Gemini-Powered**: Uses latest Google LLM for strategy optimization
5. **Indian Market Focus**: Built for NSE Nifty stocks with appropriate market hours
6. **Intraday Focus**: 5-minute bars for active day trading
7. **Visual Dashboard**: Full Streamlit UI for monitoring and control

---

*Built by Claflin Investments. Powered by Gemini AI.*
