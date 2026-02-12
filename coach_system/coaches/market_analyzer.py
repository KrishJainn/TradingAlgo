"""
Market Analysis and Forecasting Module.

Analyzes recent market data to:
1. Detect current market regime (trending/volatile/ranging)
2. Identify key support/resistance levels
3. Generate short-term forecasts (15 days)

This information helps the AI Coach make better decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta


class MarketAnalyzer:
    """
    Analyzes market conditions and generates forecasts.
    """

    def __init__(self):
        self.regime_cache = {}
        self.forecast_cache = {}

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to handle both 'Close' and 'close' formats."""
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ['close', 'open', 'high', 'low', 'volume']:
                col_map[col] = lower.capitalize()
        if col_map:
            df = df.rename(columns=col_map)
        return df

    def analyze_market(self, data: Dict[str, pd.DataFrame], lookback_days: int = 14) -> Dict:
        """
        Comprehensive market analysis.

        Args:
            data: Dict of symbol -> OHLCV DataFrame
            lookback_days: Days to look back for analysis

        Returns:
            Dict with market analysis results
        """
        if not data:
            return {"error": "No data provided"}

        # Normalize column names for all dataframes
        data = {sym: self._normalize_columns(df.copy()) for sym, df in data.items()}

        analysis = {
            "timestamp": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "symbols_analyzed": len(data),
            "regime": self._detect_regime(data, lookback_days),
            "volatility": self._analyze_volatility(data, lookback_days),
            "trend": self._analyze_trend(data, lookback_days),
            "momentum": self._analyze_momentum(data, lookback_days),
            "forecast": self._generate_forecast(data, lookback_days),
            "key_levels": self._find_key_levels(data, lookback_days),
        }

        return analysis

    def _detect_regime(self, data: Dict[str, pd.DataFrame], lookback: int) -> Dict:
        """Detect market regime: trending, volatile, or ranging."""
        regimes = []

        for symbol, df in data.items():
            if len(df) < lookback:
                continue

            recent = df.tail(lookback)

            # Calculate metrics
            returns = recent['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized

            # Trend strength (using simple linear regression slope)
            x = np.arange(len(recent))
            y = recent['Close'].values
            slope = np.polyfit(x, y, 1)[0]
            normalized_slope = slope / recent['Close'].mean() * 100  # As percentage

            # ADX-like calculation for trend strength
            high = recent['High'].values
            low = recent['Low'].values
            close = recent['Close'].values

            tr = np.maximum(high[1:] - low[1:],
                          np.maximum(abs(high[1:] - close[:-1]),
                                    abs(low[1:] - close[:-1])))
            atr = np.mean(tr)
            atr_pct = atr / recent['Close'].mean() * 100

            # Classify regime
            if abs(normalized_slope) > 0.5 and volatility < 0.4:
                regime = "TRENDING_UP" if normalized_slope > 0 else "TRENDING_DOWN"
            elif volatility > 0.4 or atr_pct > 2:
                regime = "HIGH_VOLATILITY"
            else:
                regime = "RANGING"

            regimes.append({
                "symbol": symbol,
                "regime": regime,
                "volatility": volatility,
                "trend_slope": normalized_slope,
                "atr_pct": atr_pct,
            })

        # Aggregate regime
        regime_counts = {}
        for r in regimes:
            reg = r["regime"]
            regime_counts[reg] = regime_counts.get(reg, 0) + 1

        dominant_regime = max(regime_counts, key=regime_counts.get) if regime_counts else "UNKNOWN"

        return {
            "dominant": dominant_regime,
            "distribution": regime_counts,
            "avg_volatility": np.mean([r["volatility"] for r in regimes]) if regimes else 0,
            "avg_trend": np.mean([r["trend_slope"] for r in regimes]) if regimes else 0,
        }

    def _analyze_volatility(self, data: Dict[str, pd.DataFrame], lookback: int) -> Dict:
        """Analyze market volatility."""
        volatilities = []

        for symbol, df in data.items():
            if len(df) < lookback:
                continue

            recent = df.tail(lookback)
            returns = recent['Close'].pct_change().dropna()

            vol = returns.std() * np.sqrt(252)

            # Compare to longer-term volatility
            if len(df) >= lookback * 2:
                longer = df.tail(lookback * 2).head(lookback)
                longer_vol = longer['Close'].pct_change().dropna().std() * np.sqrt(252)
                vol_change = (vol - longer_vol) / longer_vol * 100 if longer_vol > 0 else 0
            else:
                vol_change = 0

            volatilities.append({
                "symbol": symbol,
                "current_vol": vol,
                "vol_change_pct": vol_change,
            })

        avg_vol = np.mean([v["current_vol"] for v in volatilities]) if volatilities else 0
        avg_change = np.mean([v["vol_change_pct"] for v in volatilities]) if volatilities else 0

        if avg_vol > 0.4:
            vol_level = "HIGH"
        elif avg_vol > 0.25:
            vol_level = "MODERATE"
        else:
            vol_level = "LOW"

        return {
            "level": vol_level,
            "avg_annualized": avg_vol,
            "change_vs_prior": avg_change,
            "expanding": avg_change > 10,
            "contracting": avg_change < -10,
        }

    def _analyze_trend(self, data: Dict[str, pd.DataFrame], lookback: int) -> Dict:
        """Analyze market trend."""
        trends = []

        for symbol, df in data.items():
            if len(df) < lookback:
                continue

            recent = df.tail(lookback)
            close = recent['Close'].values

            # Short-term trend (last 5 days)
            short_trend = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0

            # Medium-term trend (full lookback)
            med_trend = (close[-1] - close[0]) / close[0] * 100

            # Moving average position
            ma_5 = recent['Close'].rolling(5).mean().iloc[-1]
            ma_10 = recent['Close'].rolling(10).mean().iloc[-1] if len(recent) >= 10 else ma_5
            above_ma = close[-1] > ma_5

            trends.append({
                "symbol": symbol,
                "short_trend_pct": short_trend,
                "med_trend_pct": med_trend,
                "above_ma": above_ma,
            })

        avg_short = np.mean([t["short_trend_pct"] for t in trends]) if trends else 0
        avg_med = np.mean([t["med_trend_pct"] for t in trends]) if trends else 0
        pct_above_ma = sum(1 for t in trends if t["above_ma"]) / len(trends) * 100 if trends else 50

        if avg_med > 2:
            direction = "BULLISH"
        elif avg_med < -2:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "direction": direction,
            "short_term_pct": avg_short,
            "medium_term_pct": avg_med,
            "pct_above_ma": pct_above_ma,
            "strengthening": avg_short > avg_med > 0,
            "weakening": avg_short < avg_med < 0,
        }

    def _analyze_momentum(self, data: Dict[str, pd.DataFrame], lookback: int) -> Dict:
        """Analyze market momentum."""
        momentums = []

        for symbol, df in data.items():
            if len(df) < lookback:
                continue

            recent = df.tail(lookback)
            close = recent['Close'].values

            # RSI calculation (simplified)
            changes = np.diff(close)
            gains = np.where(changes > 0, changes, 0)
            losses = np.where(changes < 0, -changes, 0)

            avg_gain = np.mean(gains) if len(gains) > 0 else 0
            avg_loss = np.mean(losses) if len(losses) > 0 else 0.001

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # Rate of change
            roc = (close[-1] - close[0]) / close[0] * 100

            momentums.append({
                "symbol": symbol,
                "rsi": rsi,
                "roc": roc,
            })

        avg_rsi = np.mean([m["rsi"] for m in momentums]) if momentums else 50
        avg_roc = np.mean([m["roc"] for m in momentums]) if momentums else 0

        if avg_rsi > 70:
            condition = "OVERBOUGHT"
        elif avg_rsi < 30:
            condition = "OVERSOLD"
        else:
            condition = "NEUTRAL"

        return {
            "condition": condition,
            "avg_rsi": avg_rsi,
            "avg_roc": avg_roc,
            "divergence": abs(avg_rsi - 50) > 20 and abs(avg_roc) < 2,  # RSI extreme but price flat
        }

    def _generate_forecast(self, data: Dict[str, pd.DataFrame], lookback: int, forecast_days: int = 15) -> Dict:
        """Generate a probabilistic forecast for next 15 days."""
        forecasts = []

        for symbol, df in data.items():
            if len(df) < lookback * 2:
                continue

            recent = df.tail(lookback)
            close = recent['Close'].values

            # Simple linear extrapolation
            x = np.arange(len(close))
            slope, intercept = np.polyfit(x, close, 1)

            # Forecast values
            future_x = np.arange(len(close), len(close) + forecast_days)
            forecast_values = slope * future_x + intercept

            # Expected return
            expected_return = (forecast_values[-1] - close[-1]) / close[-1] * 100

            # Confidence based on R-squared
            y_pred = slope * x + intercept
            ss_res = np.sum((close - y_pred) ** 2)
            ss_tot = np.sum((close - np.mean(close)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Volatility-adjusted range
            returns = pd.Series(close).pct_change().dropna()
            daily_vol = returns.std()
            expected_range = daily_vol * np.sqrt(forecast_days) * 100 * 2  # 2 std devs

            forecasts.append({
                "symbol": symbol,
                "expected_return_pct": expected_return,
                "confidence": r_squared,
                "expected_range_pct": expected_range,
            })

        if not forecasts:
            return {"error": "Insufficient data for forecast"}

        avg_return = np.mean([f["expected_return_pct"] for f in forecasts])
        avg_confidence = np.mean([f["confidence"] for f in forecasts])
        avg_range = np.mean([f["expected_range_pct"] for f in forecasts])

        # Probability estimates
        if avg_return > 1:
            outlook = "BULLISH"
            prob_up = min(0.65 + avg_confidence * 0.2, 0.85)
        elif avg_return < -1:
            outlook = "BEARISH"
            prob_up = max(0.35 - avg_confidence * 0.2, 0.15)
        else:
            outlook = "NEUTRAL"
            prob_up = 0.5

        return {
            "days_ahead": forecast_days,
            "outlook": outlook,
            "expected_return_pct": avg_return,
            "expected_range_pct": avg_range,
            "probability_up": prob_up,
            "probability_down": 1 - prob_up,
            "confidence": avg_confidence,
            "recommendation": self._get_recommendation(outlook, avg_confidence, avg_return),
        }

    def _get_recommendation(self, outlook: str, confidence: float, expected_return: float) -> str:
        """Generate trading recommendation based on forecast."""
        if confidence < 0.3:
            return "LOW CONFIDENCE - Be cautious, reduce position sizes"

        if outlook == "BULLISH" and confidence > 0.5:
            if expected_return > 3:
                return "STRONG BUY signals - Consider aggressive long positions"
            else:
                return "Moderate bullish - Favor long positions with tight stops"
        elif outlook == "BEARISH" and confidence > 0.5:
            if expected_return < -3:
                return "STRONG SELL signals - Consider defensive positions or shorts"
            else:
                return "Moderate bearish - Reduce exposure, tighten stops"
        else:
            return "NEUTRAL - Focus on range-trading strategies, avoid trend-following"

    def _find_key_levels(self, data: Dict[str, pd.DataFrame], lookback: int) -> Dict:
        """Find key support and resistance levels."""
        levels = []

        for symbol, df in data.items():
            if len(df) < lookback:
                continue

            recent = df.tail(lookback)
            high = recent['High'].max()
            low = recent['Low'].min()
            close = recent['Close'].iloc[-1]

            # Key levels
            resistance = high
            support = low
            pivot = (high + low + close) / 3

            levels.append({
                "symbol": symbol,
                "resistance": resistance,
                "support": support,
                "pivot": pivot,
                "current": close,
                "distance_to_resistance_pct": (resistance - close) / close * 100,
                "distance_to_support_pct": (close - support) / close * 100,
            })

        if not levels:
            return {}

        avg_dist_resistance = np.mean([l["distance_to_resistance_pct"] for l in levels])
        avg_dist_support = np.mean([l["distance_to_support_pct"] for l in levels])

        return {
            "avg_distance_to_resistance_pct": avg_dist_resistance,
            "avg_distance_to_support_pct": avg_dist_support,
            "near_resistance": avg_dist_resistance < 2,
            "near_support": avg_dist_support < 2,
        }

    def get_summary_for_coach(self, data: Dict[str, pd.DataFrame]) -> str:
        """
        Generate a concise summary for the AI Coach prompt.

        Returns a string that can be included in the coach's decision prompt.
        """
        analysis = self.analyze_market(data)

        if "error" in analysis:
            return "Market data unavailable."

        regime = analysis.get("regime", {})
        trend = analysis.get("trend", {})
        volatility = analysis.get("volatility", {})
        momentum = analysis.get("momentum", {})
        forecast = analysis.get("forecast", {})
        levels = analysis.get("key_levels", {})

        summary_parts = [
            f"MARKET ANALYSIS (last {analysis.get('lookback_days', 14)} days):",
            f"Regime: {regime.get('dominant', 'UNKNOWN')} | Volatility: {volatility.get('level', 'UNKNOWN')}",
            f"Trend: {trend.get('direction', 'UNKNOWN')} ({trend.get('medium_term_pct', 0):+.1f}%)",
            f"Momentum: {momentum.get('condition', 'UNKNOWN')} (RSI avg: {momentum.get('avg_rsi', 50):.0f})",
        ]

        if forecast and "outlook" in forecast:
            summary_parts.append(
                f"FORECAST (15 days): {forecast.get('outlook')} "
                f"(Expected: {forecast.get('expected_return_pct', 0):+.1f}%, "
                f"Confidence: {forecast.get('confidence', 0):.0%})"
            )
            summary_parts.append(f"Recommendation: {forecast.get('recommendation', 'N/A')}")

        if levels:
            if levels.get("near_resistance"):
                summary_parts.append("⚠️ NEAR RESISTANCE - Consider taking profits on longs")
            if levels.get("near_support"):
                summary_parts.append("⚠️ NEAR SUPPORT - Consider buying dips")

        return "\n".join(summary_parts)
