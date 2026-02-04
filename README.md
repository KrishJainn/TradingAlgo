# 5-Player Trading System with AI Coach

A Gemini-powered trading system where 5 independent players trade with different strategies, and an AI coach continuously optimizes each player based on their performance.

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    5-PLAYER TRADING SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PLAYER 1        PLAYER 2        PLAYER 3        PLAYER 4      PLAYER 5
│  Aggressive      Conservative    Balanced        VolBreakout   Momentum
│  ───────────     ────────────    ─────────       ───────────   ─────────
│  • RSI_7         • ADX_14        • RSI_14        • NATR_14     • RSI_7
│  • STOCH_5_3     • SUPERTREND    • BBANDS_20     • KC_20_2     • TSI_13
│  • TSI_13_25     • EMA_50        • STOCH_14      • ADX_14      • MACD
│  • CMO_14        • AROON_14      • CMF_20        • BBANDS_20   • CMO_14
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                         AI COACH (Gemini)                       │
│  • Analyzes each player's trades                                │
│  • Adjusts indicator weights                                    │
│  • Adds/removes indicators                                      │
│  • Optimizes entry/exit thresholds                              │
│  • LEARNS and improves over time                                │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

1. **Set up environment:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Gemini API key:**
   ```bash
   echo "GEMINI_API_KEY=your_key_here" > .env
   ```

3. **Run the dashboard:**
   ```bash
   python run_dashboard.py
   # OR
   streamlit run coach_system/dashboard/app.py
   ```

## Project Structure

```
.
├── coach_system/              # Main trading system
│   ├── coaches/
│   │   └── ai_coach.py        # Gemini-powered AI coach
│   ├── llm/
│   │   ├── base.py            # LLM provider abstraction
│   │   └── gemini_provider.py # Gemini implementation
│   └── dashboard/
│       ├── app.py             # Streamlit dashboard
│       └── pages/
│           └── continuous_backtest.py  # 5-player backtest UI
│
├── trading_evolution/         # Core trading framework
│   ├── indicators/            # Technical indicators
│   ├── backtest/              # Backtesting engine
│   └── player/                # Trade execution
│
├── aqtis/
│   └── knowledge/             # Knowledge base ingestion
│
├── data/                      # Market data utilities
├── knowledge_base/            # Trading knowledge docs
└── evolved_player_configs.json # Learned player configs
```

## How It Works

### 1. Five Independent Players
Each player trades with a unique strategy profile:
- **Aggressive**: High risk, momentum-focused, short holds
- **Conservative**: Low risk, trend-following, longer holds
- **Balanced**: Diversified approach, medium risk
- **VolBreakout**: Catches volatility breakouts
- **Momentum**: Pure momentum trading

### 2. AI Coach Optimization
Every N trading days (configurable), the Gemini-powered coach:
- Analyzes each player's recent trades
- Identifies winning/losing patterns
- Adjusts indicator weights
- Adds new indicators / removes underperformers
- Updates entry/exit thresholds

### 3. Continuous Learning
- Configs persist between runs (`evolved_player_configs.json`)
- Coach sees learning history from previous sessions
- Players evolve independently based on their own performance

## Configuration

Edit player configs in the dashboard or directly in `evolved_player_configs.json`.

Key parameters per player:
- `weights`: Dict of indicator name → weight (0.1 to 1.0)
- `entry_threshold`: Signal Index threshold to enter (0.15 to 0.40)
- `exit_threshold`: Signal Index threshold to exit (-0.20 to -0.05)
- `min_hold_bars`: Minimum bars to hold position

## Requirements

- Python 3.11+
- Gemini API key (free tier works)
- See `requirements.txt` for dependencies

## License

MIT License - Claflin Investments
