#!/usr/bin/env python3
"""
5-Player Backtest with Coach Optimization every 3 days.

Uses 60-day 15m data, simulates day-by-day trading with coach optimization.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass, field
from copy import deepcopy

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from trading_evolution.indicators.universe import IndicatorUniverse
from trading_evolution.indicators.calculator import IndicatorCalculator
from trading_evolution.indicators.normalizer import IndicatorNormalizer

import warnings
warnings.filterwarnings("ignore")

# Player configs
PLAYERS_CONFIG = {
    "PLAYER_1": {
        "label": "Aggressive",
        "weights": {
            "RSI_7": 0.90, "STOCH_5_3": 0.85, "TSI_13_25": 0.75,
            "CMO_14": 0.70, "WILLR_14": 0.65, "OBV": 0.60,
            "MFI_14": 0.55, "ADX_14": 0.50, "EMA_9": 0.45,
            "NATR_14": 0.40,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_2": {
        "label": "Conservative",
        "weights": {
            "ADX_14": 0.90, "SUPERTREND_7_3": 0.85, "EMA_20": 0.80,
            "AROON_14": 0.70, "CMF_20": 0.65, "RSI_14": 0.60,
            "BBANDS_20_2": 0.55, "OBV": 0.50, "VWMA_10": 0.45,
            "HMA_9": 0.40, "ATR_14": 0.35,
        },
        "entry_threshold": 0.35,
        "exit_threshold": -0.15,
        "min_hold_bars": 5,
    },
    "PLAYER_3": {
        "label": "Balanced",
        "weights": {
            "RSI_14": 0.85, "BBANDS_20_2": 0.80, "STOCH_14_3": 0.75,
            "CMF_20": 0.65, "ZSCORE_20": 0.60, "MFI_20": 0.55,
            "EMA_9": 0.50, "ADX_14": 0.45, "ATR_14": 0.40,
            "TEMA_20": 0.35,
        },
        "entry_threshold": 0.30,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
    "PLAYER_4": {
        "label": "VolBreakout",
        "weights": {
            "NATR_14": 0.95, "KELTNER_20_1.5": 0.90, "ADX_14": 0.85,
            "BBANDS_20_2": 0.75, "ATR_14": 0.70, "CCI_20": 0.60,
            "RSI_7": 0.55, "OBV": 0.50, "CMF_20": 0.45,
            "WILLR_14": 0.40,
        },
        "entry_threshold": 0.25,
        "exit_threshold": -0.08,
        "min_hold_bars": 3,
    },
    "PLAYER_5": {
        "label": "Momentum",
        "weights": {
            "RSI_7": 0.95, "TSI_13_25": 0.90, "MACD_12_26_9": 0.85,
            "CMO_14": 0.80, "STOCH_5_3": 0.75, "TRIX_15": 0.65,
            "PPO_12_26_9": 0.60, "ROC_10": 0.55, "MOM_10": 0.50,
            "EMA_9": 0.45,
        },
        "entry_threshold": 0.28,
        "exit_threshold": -0.10,
        "min_hold_bars": 4,
    },
}


def raw_weighted_average(normalized_df: pd.DataFrame, weights: Dict[str, float]) -> pd.Series:
    """Weighted average of normalised indicators."""
    weighted_sum = pd.Series(0.0, index=normalized_df.index)
    total_weight = 0.0
    for name, weight in weights.items():
        if name not in normalized_df.columns or weight == 0:
            continue
        weighted_sum += normalized_df[name].fillna(0) * weight
        total_weight += abs(weight)
    if total_weight == 0:
        return pd.Series(0.0, index=normalized_df.index)
    return (weighted_sum / total_weight).clip(-1.0, 1.0)


@dataclass
class PlayerState:
    player_id: str
    config: Dict
    equity: float = 100000.0
    positions: Dict = field(default_factory=dict)  # sym -> {direction, entry_price, entry_bar}
    bars_held: Dict = field(default_factory=dict)
    prev_si: Dict = field(default_factory=dict)
    trades: List = field(default_factory=list)
    daily_pnl: List = field(default_factory=list)
    coach_history: List = field(default_factory=list)


# Import intelligent coach
try:
    from intelligent_coach import IntelligentCoach, coach_optimize_v2
    INTELLIGENT_COACH = True
except ImportError:
    INTELLIGENT_COACH = False


def coach_optimize(players: Dict[str, PlayerState], day_num: int, market_regime: str = "unknown") -> Dict[str, Dict]:
    """Coach analyzes recent performance and optimizes configs with intelligent adjustments."""

    # Use intelligent coach if available
    if INTELLIGENT_COACH:
        return coach_optimize_v2(players, day_num, market_regime)

    # Fallback to basic optimization
    new_configs = {}

    for pid, state in players.items():
        config = deepcopy(state.config)
        weights = config["weights"]

        recent_trades = state.trades[-50:] if len(state.trades) > 50 else state.trades

        if not recent_trades:
            new_configs[pid] = config
            continue

        wins = sum(1 for t in recent_trades if t["pnl"] > 0)
        win_rate = wins / len(recent_trades) if recent_trades else 0
        total_pnl = sum(t["pnl"] for t in recent_trades)

        if win_rate < 0.40:
            config["entry_threshold"] = min(0.50, config["entry_threshold"] + 0.03)
            config["min_hold_bars"] = min(8, config["min_hold_bars"] + 1)
        elif win_rate > 0.55:
            config["entry_threshold"] = max(0.20, config["entry_threshold"] - 0.02)

        if total_pnl < 0:
            config["exit_threshold"] = max(-0.20, config["exit_threshold"] - 0.02)

        import random
        for k in weights:
            perturbation = random.uniform(-0.05, 0.05)
            weights[k] = max(0.1, min(1.0, weights[k] + perturbation))

        config["weights"] = weights
        new_configs[pid] = config

        print(f"    {pid}: WR={win_rate:.0%}, P&L=${total_pnl:+,.0f} -> thresh={config['entry_threshold']:.2f}")

    return new_configs


def simulate_day(
    day_data: Dict[str, pd.DataFrame],
    indicator_data: Dict[str, pd.DataFrame],
    players: Dict[str, PlayerState],
    normalizer: IndicatorNormalizer,
    day_idx: int,
    bars_per_day: int = 26,
):
    """Simulate one trading day."""

    start_bar = day_idx * bars_per_day
    end_bar = start_bar + bars_per_day

    for pid, state in players.items():
        config = state.config
        weights = config["weights"]
        entry_thresh = config["entry_threshold"]
        exit_thresh = config["exit_threshold"]
        min_hold = config["min_hold_bars"]

        day_pnl = 0.0

        for sym, df in day_data.items():
            if sym not in indicator_data:
                continue

            raw = indicator_data[sym]

            # Get day's bars
            if end_bar > len(df):
                continue

            day_df = df.iloc[start_bar:end_bar]
            day_raw = raw.iloc[start_bar:end_bar]

            if day_df.empty or day_raw.empty:
                continue

            # Get active indicators
            active = [i for i in weights.keys() if i in day_raw.columns]
            if not active:
                continue

            try:
                norm = normalizer.normalize_all(day_raw[active], price_series=day_df["close"])
                if norm.empty:
                    continue
                si_series = raw_weighted_average(norm, weights).fillna(0.0)
            except:
                continue

            # Simulate bar by bar
            for bar_idx in range(len(day_df)):
                si = si_series.iloc[bar_idx]
                price = day_df.iloc[bar_idx]["close"]
                prev_si = state.prev_si.get(sym, 0.0)

                # Check positions
                if sym in state.positions:
                    state.bars_held[sym] = state.bars_held.get(sym, 0) + 1
                    pos = state.positions[sym]
                    held = state.bars_held[sym]

                    # Exit logic
                    should_exit = False
                    if pos["direction"] == "LONG" and held >= min_hold and si < exit_thresh:
                        should_exit = True
                    elif pos["direction"] == "SHORT" and held >= min_hold and si > abs(exit_thresh):
                        should_exit = True

                    # EOD flatten (last bar of day)
                    if bar_idx == len(day_df) - 1:
                        should_exit = True

                    if should_exit:
                        entry_price = pos["entry_price"]
                        if pos["direction"] == "LONG":
                            pnl = (price - entry_price) / entry_price * 1000
                        else:
                            pnl = (entry_price - price) / entry_price * 1000

                        state.equity += pnl
                        day_pnl += pnl

                        state.trades.append({
                            "symbol": sym,
                            "direction": pos["direction"],
                            "entry_price": entry_price,
                            "exit_price": price,
                            "pnl": pnl,
                            "bars_held": held,
                            "day": day_idx,
                        })

                        del state.positions[sym]
                        state.bars_held.pop(sym, None)

                else:
                    # Entry logic (not last 2 bars of day)
                    if bar_idx < len(day_df) - 2:
                        if si > entry_thresh and prev_si > entry_thresh * 0.7:
                            state.positions[sym] = {
                                "direction": "LONG",
                                "entry_price": price,
                            }
                            state.bars_held[sym] = 0
                        elif si < -entry_thresh and prev_si < -entry_thresh * 0.7:
                            state.positions[sym] = {
                                "direction": "SHORT",
                                "entry_price": price,
                            }
                            state.bars_held[sym] = 0

                state.prev_si[sym] = si

        state.daily_pnl.append(day_pnl)


def main():
    print("="*60)
    print("5-PLAYER BACKTEST - 60 DAY 15M DATA")
    print("Coach optimization every 3 days")
    print("="*60)

    # Load 60-day 15m data
    data_dir = Path("backtest_data")
    data = {}

    for f in data_dir.glob("*_60d_15m.csv"):
        sym = f.stem.replace("_60d_15m", "") + ".NS"
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        data[sym] = df

    if not data:
        print("No 60-day data found. Run download first.")
        return

    print(f"Loaded {len(data)} symbols")

    # Calculate indicators for full dataset
    print("Calculating indicators...")
    universe = IndicatorUniverse()
    universe.load_all()
    calculator = IndicatorCalculator(universe=universe)
    normalizer = IndicatorNormalizer()

    indicator_data = {}
    for sym, df in data.items():
        try:
            raw = calculator.calculate_all(df)
            raw = calculator.rename_to_dna_names(raw)
            indicator_data[sym] = raw
        except:
            pass

    print(f"Indicators ready for {len(indicator_data)} symbols")

    # Initialize players
    players = {}
    for pid, config in PLAYERS_CONFIG.items():
        players[pid] = PlayerState(
            player_id=pid,
            config=deepcopy(config),
        )

    # Simulate day by day
    bars_per_day = 26  # ~26 15m bars per trading day
    sample_df = list(data.values())[0]
    total_bars = len(sample_df)
    total_days = total_bars // bars_per_day

    print(f"\nSimulating {total_days} trading days...")
    print("-"*60)

    coach_interval = 3  # Coach every 3 days

    for day in range(total_days):
        simulate_day(data, indicator_data, players, normalizer, day, bars_per_day)

        # Print daily summary
        if (day + 1) % 5 == 0:
            total_equity = sum(p.equity for p in players.values())
            total_trades = sum(len(p.trades) for p in players.values())
            print(f"Day {day+1:3d}: Team Equity=${total_equity:,.0f}, Trades={total_trades}")

        # Coach optimization every 3 days
        if (day + 1) % coach_interval == 0 and day < total_days - 1:
            print(f"\n  COACH (Day {day+1}):")
            new_configs = coach_optimize(players, day)
            for pid, config in new_configs.items():
                players[pid].config = config
            print()

    # Final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)

    total_pnl = 0
    for pid, state in players.items():
        trades = len(state.trades)
        wins = sum(1 for t in state.trades if t["pnl"] > 0)
        win_rate = wins / trades if trades > 0 else 0
        pnl = state.equity - 100000
        total_pnl += pnl

        # Calculate Sharpe
        if state.daily_pnl:
            returns = np.array(state.daily_pnl) / 100000
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        print(f"{pid} ({state.config['label']}):")
        print(f"  Trades: {trades}, Win Rate: {win_rate:.0%}")
        print(f"  P&L: ${pnl:+,.0f}, Sharpe: {sharpe:.2f}")
        print(f"  Final Threshold: {state.config['entry_threshold']:.2f}")

    print(f"\nTOTAL P&L: ${total_pnl:+,.0f}")

    # Save optimized configs
    output = {}
    for pid, state in players.items():
        output[pid] = state.config

    with open("optimized_player_configs.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\nOptimized configs saved to: optimized_player_configs.json")


if __name__ == "__main__":
    main()
