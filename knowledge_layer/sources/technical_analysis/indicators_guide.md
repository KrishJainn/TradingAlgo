# Technical Indicators Guide

## Momentum Indicators

### RSI (Relative Strength Index)
- **RSI_7**: Fast, more signals, more noise. Best for aggressive traders.
- **RSI_14**: Standard setting, balanced signals.
- **RSI_21**: Slower, fewer but more reliable signals.

Interpretation:
- Above 70: Overbought, potential reversal
- Below 30: Oversold, potential reversal
- Divergence: Price vs RSI can signal reversals

### Stochastic
- **STOCH_5_3**: Very fast, good for scalping
- **STOCH_14_3**: Standard, good balance
- **STOCH_21_5**: Slower, swing trading

When to use:
- Best in ranging markets
- Less effective in strong trends
- Look for crossovers at extremes

### MACD (Moving Average Convergence Divergence)
- **MACD_12_26_9**: Standard setting
- Signal line crossovers indicate momentum shifts
- Histogram shows momentum strength

Best for:
- Trend confirmation
- Divergence analysis
- Entry timing in trending markets

### TSI (True Strength Index)
- **TSI_13_25**: Smoothed momentum indicator
- Less noise than RSI
- Good for identifying trend changes

### CMO (Chande Momentum Oscillator)
- **CMO_14**: Similar to RSI but unsmoothed
- More responsive to recent price changes
- Good for aggressive trading styles

## Trend Indicators

### ADX (Average Directional Index)
- **ADX_14**: Measures trend strength
- **ADX_20**: Smoother, less noise

Interpretation:
- Above 25: Strong trend
- Below 20: Weak trend or ranging
- Rising ADX: Trend strengthening
- Falling ADX: Trend weakening

### Supertrend
- **SUPERTREND_7_3**: Fast, more signals
- **SUPERTREND_10_2**: Balanced
- **SUPERTREND_20_3**: Slow, longer-term trends

Use for:
- Trend direction
- Stop loss placement
- Entry/exit signals

### Aroon
- **AROON_14**: Measures time since high/low
- **AROON_25**: Longer-term perspective

Signals:
- Aroon Up > 70 & Aroon Down < 30: Uptrend
- Aroon Down > 70 & Aroon Up < 30: Downtrend
- Crossovers indicate trend changes

## Volatility Indicators

### ATR (Average True Range)
- **ATR_14**: Standard volatility measure
- **ATR_20**: Smoother volatility reading

Use for:
- Position sizing
- Stop loss placement
- Identifying volatility expansion/contraction

### NATR (Normalized ATR)
- **NATR_14**: ATR as percentage of price
- Better for comparing across different price levels
- Useful for portfolio-wide volatility assessment

### Bollinger Bands
- **BBANDS_20_2**: Price relative to volatility
- Band squeeze = low volatility, breakout imminent
- Band expansion = high volatility period

### Keltner Channels
- **KC_20_2**: ATR-based channels
- Less reactive to outliers than Bollinger
- Good for trend-following strategies

## Volume Indicators

### OBV (On Balance Volume)
- Cumulative volume indicator
- Confirms price trends
- Divergence can signal reversals

### CMF (Chaikin Money Flow)
- **CMF_20**: Volume-weighted buying/selling pressure
- Positive = buying pressure
- Negative = selling pressure

### MFI (Money Flow Index)
- **MFI_14**: Volume-weighted RSI
- **MFI_20**: Smoother version
- Better than RSI for volume confirmation

## Overlap Indicators (Moving Averages)

### EMA (Exponential Moving Average)
- **EMA_10**: Fast, trend following
- **EMA_20**: Medium-term trend
- **EMA_50**: Important support/resistance
- **EMA_200**: Long-term trend

### SMA (Simple Moving Average)
- More stable than EMA
- Less reactive to recent prices
- Good for identifying major trends

### DEMA/TEMA (Double/Triple EMA)
- **DEMA_20**: Less lag than EMA
- **TEMA_20**: Even less lag
- Better for faster trend identification

### HMA (Hull Moving Average)
- **HMA_9**: Very responsive
- Almost no lag
- Good for aggressive traders

## Indicator Combinations

### For Aggressive Trading
1. RSI_7 + STOCH_5_3 (momentum confirmation)
2. CMO_14 + WILLR_14 (oversold/overbought)
3. NATR_14 (volatility-based stops)

### For Conservative Trading
1. ADX_14 + AROON_14 (trend confirmation)
2. SUPERTREND_10_2 (direction)
3. EMA_50 + SMA_50 (support/resistance)

### For Momentum Trading
1. MACD_12_26_9 + TSI_13_25
2. RSI_7 + ROC_10
3. DEMA_20 for trend direction

### For Volatility Breakout
1. BBANDS_20_2 + ATR_14
2. KC_20_2 for channel breakouts
3. ADX_14 for trend strength confirmation
