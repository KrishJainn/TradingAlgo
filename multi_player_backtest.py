#!/usr/bin/env python3
"""
5-Player Backtest with Best Strategies from 669 DNA Database.
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from trading_evolution.config import DataConfig
from trading_evolution.data.fetcher import DataFetcher
from trading_evolution.data.cache import DataCache
from trading_evolution.indicators.universe import IndicatorUniverse
from trading_evolution.indicators.calculator import IndicatorCalculator
from trading_evolution.indicators.normalizer import IndicatorNormalizer
from trading_evolution.super_indicator.dna import SuperIndicatorDNA, IndicatorGene
from trading_evolution.super_indicator.core import SuperIndicator
from trading_evolution.super_indicator.signals import SignalType, PositionState
from trading_evolution.player.trader import Player
from trading_evolution.player.portfolio import Portfolio
from trading_evolution.player.risk_manager import RiskManager, RiskParameters
from trading_evolution.player.execution import ExecutionEngine


# 5 BEST DISTINCT STRATEGIES FROM DATABASE
PLAYERS = {
    "PLAYER_1": {
        "dna_id": "178af481",
        "original": "$221k profit, 50.6% WR, Sharpe 6.55",
        "weights": {
            "PSAR": 1.0, "UO_7_14_28": 1.0, "NVI": 0.9591, "AROON_25": 0.8784,
            "NATR_20": 0.7236, "WILLR_14": 0.6319, "AO_5_34": 0.6209, "ZSCORE_20": 0.6194,
            "TSI_13_25": 0.5093, "OBV": 0.4714, "MFI_20": 0.4112, "ZSCORE_50": 0.3574,
            "ADX_20": -0.9140, "COPPOCK": -0.8994, "LINREG_SLOPE_14": -0.8898,
            "T3_10": -0.8648, "SMA_100": -0.7469, "MFI_14": -0.7001, "PIVOTS": -0.6529,
            "ATR_20": -0.5389, "ADX_14": -0.4963, "SUPERTREND_20_3": -0.4921, "CMF_21": -0.4400,
        },
    },
    "PLAYER_2": {
        "dna_id": "3d290598",
        "original": "$55k profit, 82.4% WR, Sharpe 17.36",
        "weights": {
            "TEMA_20": 0.9034, "KST": 0.8221, "CMF_20": 0.7471, "HMA_9": 0.6080,
            "CMO_14": 0.5187, "ZSCORE_20": 0.4706, "BBANDS_20_2": 0.3865, "VORTEX_14": 0.3862,
            "SMA_100": 0.3522, "ZSCORE_50": 0.3520, "WMA_20": 0.3320, "VWMA_10": 0.2957,
            "ADX_20": -1.0, "DONCHIAN_20": -0.6885, "SUPERTREND_7_3": -0.5066, "PVI": -0.4915,
            "TRUERANGE": -0.4556, "ADX_14": -0.4027, "LINREG_SLOPE_14": -0.3821, "OBV": -0.3075,
        },
    },
    "PLAYER_3": {
        "dna_id": "8748f3f8",
        "original": "$89k profit, 78.3% WR, Sharpe 13.62",
        "weights": {
            "TSI_13_25": 0.8883, "NVI": 0.8611, "PVI": 0.7964, "STOCH_5_3": 0.6260,
            "ATR_14": 0.5265, "ZSCORE_20": 0.5002, "AROON_14": 0.4698, "BBANDS_20_2.5": 0.4642,
            "MASS_INDEX": 0.4535, "TEMA_20": 0.4258, "CMF_21": 0.3670, "ATR_20": 0.3443,
            "SUPERTREND_7_3": -0.9663, "AROON_25": -0.9508, "AO_5_34": -0.9286,
            "VWMA_10": -0.9283, "ADX_20": -0.8928, "EFI_13": -0.8716, "MFI_14": -0.8352,
            "VWMA_20": -0.8194, "KST": -0.7956, "WMA_20": -0.7559, "WMA_10": -0.7482,
        },
    },
    "PLAYER_4": {
        "dna_id": "7d8d1a12",
        "original": "$207k profit, 40.6% WR, Sharpe 4.65",
        "weights": {
            "PSAR": 1.0, "NVI": 0.9591, "LINREG_SLOPE_25": 0.8524, "AROON_25": 0.8297,
            "NATR_20": 0.7236, "UO_7_14_28": 0.6906, "WILLR_14": 0.6319, "ZSCORE_20": 0.6194,
            "TSI_13_25": 0.5093, "OBV": 0.4714, "MFI_20": 0.4112, "ZSCORE_50": 0.3574,
            "COPPOCK": -0.8994, "T3_10": -0.8648, "LINREG_SLOPE_14": -0.8423, "SMA_100": -0.7469,
            "MFI_14": -0.7001, "ADX_20": -0.6628, "PIVOTS": -0.6529, "ATR_20": -0.5389,
        },
    },
    "PLAYER_5": {
        "dna_id": "321c0c6f",
        "original": "$178k profit, 54.5% WR, Sharpe 7.06",
        "weights": {
            "NATR_20": 0.9810, "NVI": 0.9591, "PSAR": 0.8580, "VWMA_20": 0.7125,
            "UO_7_14_28": 0.6423, "ZSCORE_20": 0.6375, "WILLR_14": 0.6319, "AROON_25": 0.5984,
            "TSI_13_25": 0.5022, "OBV": 0.4714, "ZSCORE_50": 0.4004, "LINREG_SLOPE_25": 0.3927,
            "LINREG_SLOPE_14": -0.9212, "COPPOCK": -0.8994, "EFI_20": -0.8544, "SMA_100": -0.7469,
            "ADX_20": -0.6926, "PIVOTS": -0.6529, "ATR_20": -0.5389, "ADX_14": -0.4963,
        },
    },
}

# Default params for all
for p in PLAYERS.values():
    p.setdefault("entry_threshold", 0.70)
    p.setdefault("exit_threshold", 0.30)
    p.setdefault("stop_loss_atr", 2.0)
    p.setdefault("take_profit_atr", 0)
    p.setdefault("position_size_pct", 0.10)


@dataclass
class Result:
    player: str
    dna_id: str
    trades: int
    win_rate: float
    profit: float
    sharpe: float
    max_dd_pct: float


def create_dna(weights, dna_id):
    genes = {n: IndicatorGene(name=n, weight=w, active=abs(w) > 0.01, category='unknown') for n, w in weights.items()}
    return SuperIndicatorDNA(dna_id=dna_id, generation=0, run_id=0, genes=genes)


def run_backtest(player_id, config, years=1, use_cache=False):
    print(f"\n{player_id}: DNA {config['dna_id']} ({config['original']})")

    data_config = DataConfig()
    data_config.data_years = years

    # Use local data cache if requested
    if use_cache:
        try:
            from data_cache.backtest_data_provider import BacktestDataProvider
            from data_cache.cache_manager import CacheManager
            cache_mgr = CacheManager(cache_dir="data_cache")
            fetcher = BacktestDataProvider(cache_mgr, fallback_to_yfinance=True)
        except Exception:
            cache = DataCache(data_config.cache_dir)
            fetcher = DataFetcher(cache=cache, cache_dir=data_config.cache_dir)
    else:
        cache = DataCache(data_config.cache_dir)
        fetcher = DataFetcher(cache=cache, cache_dir=data_config.cache_dir)
    universe = IndicatorUniverse()
    universe.load_all()
    calc = IndicatorCalculator(universe=universe)
    norm = IndicatorNormalizer()
    
    dna = create_dna(config['weights'], config['dna_id'])
    si = SuperIndicator(dna, normalizer=norm)
    
    trades = []
    capital = 100000.0
    
    for symbol in data_config.symbols:
        df = fetcher.fetch(symbol, years=years)
        if df is None or len(df) < 50:
            continue
        
        indicators = calc.calculate_all(df)
        if indicators.empty:
            continue
        
        portfolio = Portfolio(initial_capital=capital)
        risk = RiskManager(params=RiskParameters(max_risk_per_trade=0.02, max_position_pct=config['position_size_pct']))
        exec_eng = ExecutionEngine(slippage_pct=0.001, commission_per_share=0.005)
        player = Player(portfolio=portfolio, risk_manager=risk, execution=exec_eng)
        
        active = [i for i in dna.get_active_indicators() if i in indicators.columns]
        if not active:
            continue
        
        normalized = norm.normalize_all(indicators[active], price_series=df['close'])
        if normalized.empty:
            continue
        
        si_series = si.calculate(normalized)
        prev_si = 0.0
        
        for i in range(50, len(df)):
            bar = df.iloc[i]
            si_val = float(si_series.iloc[i])
            
            pos = portfolio.get_position(symbol)
            state = PositionState.FLAT if pos is None else (PositionState.LONG if pos.direction == 'LONG' else PositionState.SHORT)
            
            signal = SignalType.HOLD
            if state == PositionState.FLAT:
                if si_val > config['entry_threshold'] and prev_si <= config['entry_threshold']:
                    signal = SignalType.LONG_ENTRY
                elif si_val < -config['entry_threshold'] and prev_si >= -config['entry_threshold']:
                    signal = SignalType.SHORT_ENTRY
            elif state == PositionState.LONG and si_val < config['exit_threshold']:
                signal = SignalType.LONG_EXIT
            elif state == PositionState.SHORT and si_val > -config['exit_threshold']:
                signal = SignalType.SHORT_EXIT
            
            prev_si = si_val
            atr = float(indicators.iloc[i].get('ATR_14', bar['close'] * 0.02)) or bar['close'] * 0.02
            
            trade = player.process_signal(symbol=symbol, signal=signal, current_price=bar['close'],
                                          timestamp=bar.name, high=bar['high'], low=bar['low'], atr=atr, si_value=si_val)
            if trade:
                trades.append(trade.net_pnl)
        
        if not df.empty:
            for t in player.close_all_positions(timestamp=df.index[-1], prices={symbol: float(df.iloc[-1]['close'])}):
                trades.append(t.net_pnl)
    
    if trades:
        pnls = trades
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        profit = sum(pnls)
        sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10) * np.sqrt(252)
        cum = np.cumsum(pnls)
        dd = np.max(np.maximum.accumulate(cum) - cum) / capital * 100
    else:
        wr, profit, sharpe, dd = 0, 0, 0, 0
    
    print(f"  -> {len(trades)} trades, {wr:.1f}% WR, ${profit:,.0f} profit, {sharpe:.2f} Sharpe")
    return Result(player_id, config['dna_id'], len(trades), wr, profit, sharpe, dd)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', type=int, default=1)
    parser.add_argument('--use-cache', action='store_true', help='Use local data cache')
    args = parser.parse_args()

    print("="*60)
    print("5-PLAYER BACKTEST")
    if args.use_cache:
        print("  Data: LOCAL CACHE")
    print("="*60)

    results = [run_backtest(pid, cfg, args.years, use_cache=args.use_cache) for pid, cfg in PLAYERS.items()]
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"{'Player':<10} {'DNA':<10} {'Trades':<8} {'WR%':<8} {'Profit':<12} {'Sharpe':<8}")
    print("-"*60)
    
    for r in sorted(results, key=lambda x: x.profit, reverse=True):
        print(f"{r.player:<10} {r.dna_id:<10} {r.trades:<8} {r.win_rate:<8.1f} ${r.profit:<11,.0f} {r.sharpe:<8.2f}")
    
    with open('multi_player_backtest_results.json', 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    
    print("\n✅ Done")


if __name__ == '__main__':
    main()
