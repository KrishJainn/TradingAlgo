"""
Script to analyze trades from the best evolved DNA strategy.
"""
import json
import pandas as pd
from datetime import datetime
from trading_evolution.config import DataConfig, PortfolioConfig, RiskConfig
from trading_evolution.data.fetcher import DataFetcher
from trading_evolution.data.cache import DataCache
from trading_evolution.indicators.universe import IndicatorUniverse
from trading_evolution.indicators.calculator import IndicatorCalculator
from trading_evolution.indicators.normalizer import IndicatorNormalizer
from trading_evolution.super_indicator.dna import SuperIndicatorDNA, IndicatorGene
from trading_evolution.super_indicator.core import SuperIndicator
from trading_evolution.super_indicator.signals import SignalGenerator, SignalType, PositionState
from trading_evolution.player.trader import Player
from trading_evolution.player.portfolio import Portfolio
from trading_evolution.player.risk_manager import RiskManager, RiskParameters
from trading_evolution.player.execution import ExecutionEngine

# Best DNA weights from Run 27 (70.6% win rate)
BEST_DNA_WEIGHTS = {"VWMA_20": -0.24619376590689368, "ADOSC_3_10": 0.32201019859590296, "EMA_200": 0.025783571078356788, "WMA_10": -0.20981506031879354, "BBANDS_10_1.5": -0.1282883536939774, "MOM_20": 0.06549793335019323, "COPPOCK": 0.08864446147415761, "RSI_14": -0.03769774008545166, "T3_5": -0.10430948189626436, "MASS_INDEX": 0.18713180214454855, "KST": -0.06018682406377528, "MACD_8_17_9": -0.25050180388186755, "NVI": -0.03503915102375089, "STOCH_5_3": -0.197760876390149, "KC_20_2": -0.691850847149595, "PIVOTS": -0.012165480618031172, "AROON_25": 0.15664528371044284, "DONCHIAN_50": 0.1958626249751722, "EMA_100": 0.4099105860571872, "STOCH_21_5": 0.13022432666539882, "TEMA_20": -0.290516557673259, "TRUERANGE": -0.03884583724968441, "SUPERTREND_10_2": -0.25175599487403677, "MOM_10": 0.028853729239701607, "SMA_10": 0.18923365618319637, "VORTEX_14": -0.25767042144257357, "NATR_20": 0.19454177102611273, "KAMA_20": -0.366932589775956, "CMO_14": -0.03837249373510518, "EFI_13": -0.03844493001555163, "WMA_20": 0.25190829465173803, "ZSCORE_20": 0.03826188733394944, "NATR_14": -0.20415466958022888, "HMA_16": 0.006027223149636576, "KAMA_10": 0.036830016183140024, "CMO_20": -0.14110347032410095, "CMF_20": -0.033812935505722355, "ZSCORE_50": -0.2619463614238301, "DEMA_20": 0.07114418593672157, "EMA_10": 0.05608335096979019, "WILLR_28": 0.13697317917246388, "BBANDS_20_2.5": -0.19399984797607583, "ATR_14": -0.18026122156505198, "PVI": -0.40902404594520175, "MACD_12_26_9": -0.021788412489594178, "HMA_9": 0.02397792168003331, "MACD_5_35_5": -0.04021185245575475, "VWMA_10": 0.0857458033594593, "LINREG_SLOPE_25": 0.03158681351100365, "STOCH_14_3": 0.13380506637515624, "CMF_21": 0.16105349513806835, "AO_5_34": 0.192488034214589, "WILLR_14": -0.06292459752481275, "T3_10": 0.02771821369174613, "ADX_14": 0.019481149291951442, "SUPERTREND_7_3": 0.0007427525049570737, "DEMA_10": 0.1154988917973338, "MFI_20": 0.16999556004183078, "SMA_50": 0.0747164451820983, "PSAR": 0.15443031493262221, "EFI_20": -0.26374396444423387, "ICHIMOKU": -0.12528892604551986, "MFI_14": -0.2871856024429144, "CCI_14": -0.11398926767854414, "ROC_20": -0.017344774970466083, "CCI_20": 0.1558793240693093, "LINREG_SLOPE_14": 0.13022879670402115, "AROON_14": 0.2219110691821039, "BBANDS_20_2": 0.07184185783606015, "OBV": -0.08350977358453282, "UO_7_14_28": -0.010417213378077822, "DONCHIAN_20": -0.03325420862661388, "TSI_13_25": -0.22952688808633026, "SMA_200": -0.003867955311737615, "EMA_50": 0.1854744655360398, "AD": -0.5569938723728218, "RSI_7": -0.11534682305679764, "SUPERTREND_20_3": -0.04115493806757446, "TEMA_10": -0.26641734028796293, "RSI_21": -0.1867599066530693, "ATR_20": -0.38084628350576344, "KC_20_1.5": -0.00557973032521546}

def create_dna_from_weights(weights: dict) -> SuperIndicatorDNA:
    """Create a DNA object from weights dictionary."""
    genes = {}
    for name, weight in weights.items():
        genes[name] = IndicatorGene(
            name=name,
            weight=weight,
            active=abs(weight) > 0.001,
            category='unknown'
        )

    dna = SuperIndicatorDNA(
        dna_id='262df23b',
        generation=20,
        run_id=27,
        genes=genes
    )
    return dna

def main():
    # Config
    data_config = DataConfig()
    data_config.data_years = 1
    portfolio_config = PortfolioConfig()
    risk_config = RiskConfig()

    # Initialize components
    data_cache = DataCache(data_config.cache_dir)
    data_fetcher = DataFetcher(cache=data_cache, cache_dir=data_config.cache_dir)

    indicator_universe = IndicatorUniverse()
    indicator_universe.load_all()
    indicator_calculator = IndicatorCalculator(universe=indicator_universe)
    indicator_normalizer = IndicatorNormalizer()

    # Create DNA
    dna = create_dna_from_weights(BEST_DNA_WEIGHTS)

    # Create Super Indicator
    super_indicator = SuperIndicator(dna, normalizer=indicator_normalizer)
    signal_generator = SignalGenerator()

    # Storage for all trades
    all_trades = []

    # Process each symbol
    for symbol in data_config.symbols:
        print(f"Processing {symbol}...")

        # Fetch data
        df = data_fetcher.fetch(symbol, years=data_config.data_years)
        if df is None or len(df) < 50:
            print(f"  Skipping {symbol} - insufficient data")
            continue

        # Calculate indicators
        indicators = indicator_calculator.calculate_all(df)
        if indicators.empty:
            continue

        # Create Player for this symbol
        portfolio = Portfolio(initial_capital=portfolio_config.initial_capital)
        risk_params = RiskParameters(
            max_risk_per_trade=risk_config.max_risk_per_trade,
            max_position_pct=risk_config.max_position_pct
        )
        risk_manager = RiskManager(params=risk_params)
        execution = ExecutionEngine(slippage_pct=0.001, commission_per_share=0.005)
        player = Player(portfolio=portfolio, risk_manager=risk_manager, execution=execution)

        # Get active indicators
        active_indicators = dna.get_active_indicators()
        valid_active = [ind for ind in active_indicators if ind in indicators.columns]

        if not valid_active:
            continue

        # Calculate normalized indicators and SI
        active_df = indicators[valid_active]
        normalized_all = indicator_normalizer.normalize_all(active_df, price_series=df['close'])

        if normalized_all.empty:
            continue

        si_series = super_indicator.calculate(normalized_all)
        prev_si_value = 0.0

        # Simulate through each bar
        for i in range(50, len(df)):
            current_bar = df.iloc[i]
            timestamp = current_bar.name
            si_value = float(si_series.iloc[i])

            # Determine position state
            current_position = portfolio.get_position(symbol)
            if current_position is None:
                pos_state = PositionState.FLAT
            elif current_position.direction == 'LONG':
                pos_state = PositionState.LONG
            elif current_position.direction == 'SHORT':
                pos_state = PositionState.SHORT
            else:
                pos_state = PositionState.FLAT

            # Generate signal
            signal = signal_generator._determine_signal(
                si=si_value,
                si_prev=prev_si_value,
                position=pos_state
            )
            prev_si_value = si_value

            # Get ATR
            atr = float(indicators.iloc[i].get('ATR_14', current_bar['high'] - current_bar['low']))
            if atr <= 0:
                atr = float(current_bar['close'] * 0.01)

            # Process signal
            trade = player.process_signal(
                symbol=symbol,
                signal=signal,
                current_price=current_bar['close'],
                timestamp=timestamp,
                high=current_bar['high'],
                low=current_bar['low'],
                atr=atr,
                si_value=si_value
            )

            if trade:
                trade_dict = trade.__dict__.copy()
                trade_dict['symbol'] = symbol
                all_trades.append(trade_dict)

        # Close remaining positions
        if not df.empty:
            final_price = float(df.iloc[-1]['close'])
            final_trades = player.close_all_positions(
                timestamp=df.index[-1],
                prices={symbol: final_price}
            )
            for t in final_trades:
                trade_dict = t.__dict__.copy()
                trade_dict['symbol'] = symbol
                all_trades.append(trade_dict)

    # Create DataFrame and display
    if all_trades:
        trades_df = pd.DataFrame(all_trades)

        # Sort by entry time
        if 'entry_time' in trades_df.columns:
            trades_df = trades_df.sort_values('entry_time')

        # Calculate statistics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        total_profit = trades_df['net_pnl'].sum()
        avg_profit = trades_df['net_pnl'].mean()
        avg_winner = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loser = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() if losing_trades > 0 else 0

        print("\n" + "="*100)
        print("TRADING STRATEGY PERFORMANCE SUMMARY")
        print("="*100)
        print(f"\nStrategy: Best DNA from 1-Year Evolution (ID: 262df23b)")
        print(f"Period: Past 1 Year")
        print(f"\n--- OVERALL STATISTICS ---")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total Net Profit: ${total_profit:,.2f}")
        print(f"Average P&L per Trade: ${avg_profit:,.2f}")
        print(f"Average Winner: ${avg_winner:,.2f}")
        print(f"Average Loser: ${avg_loser:,.2f}")

        # By direction
        if 'direction' in trades_df.columns:
            long_trades = trades_df[trades_df['direction'] == 'LONG']
            short_trades = trades_df[trades_df['direction'] == 'SHORT']

            print(f"\n--- BY DIRECTION ---")
            if len(long_trades) > 0:
                long_winners = len(long_trades[long_trades['net_pnl'] > 0])
                print(f"Long Trades: {len(long_trades)} (Win Rate: {long_winners/len(long_trades)*100:.1f}%)")
            if len(short_trades) > 0:
                short_winners = len(short_trades[short_trades['net_pnl'] > 0])
                print(f"Short Trades: {len(short_trades)} (Win Rate: {short_winners/len(short_trades)*100:.1f}%)")

        # Print all trades
        print("\n" + "="*100)
        print("COMPLETE TRADE LOG")
        print("="*100)

        for idx, trade in trades_df.iterrows():
            entry_time = trade.get('entry_time', 'N/A')
            exit_time = trade.get('exit_time', 'N/A')
            symbol = trade.get('symbol', 'N/A')
            direction = trade.get('direction', 'N/A')
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            quantity = trade.get('quantity', 0)
            net_pnl = trade.get('net_pnl', 0)
            pnl_pct = trade.get('net_pnl_pct', 0) * 100 if trade.get('net_pnl_pct') else 0
            exit_reason = trade.get('exit_reason', 'signal')

            result = "WIN" if net_pnl > 0 else "LOSS"

            print(f"\n{'─'*80}")
            print(f"Trade #{trades_df.index.get_loc(idx) + 1}")
            print(f"{'─'*80}")
            print(f"  Symbol:       {symbol}")
            print(f"  Direction:    {direction}")
            print(f"  Entry Date:   {entry_time}")
            print(f"  Exit Date:    {exit_time}")
            print(f"  Entry Price:  ${entry_price:,.2f}")
            print(f"  Exit Price:   ${exit_price:,.2f}")
            print(f"  Quantity:     {quantity:,}")
            print(f"  P&L:          ${net_pnl:,.2f} ({pnl_pct:+.2f}%)")
            print(f"  Result:       {result}")
            print(f"  Exit Reason:  {exit_reason}")

        print("\n" + "="*100)
        print("END OF TRADE LOG")
        print("="*100)

        # Save to CSV
        trades_df.to_csv('trade_log.csv', index=False)
        print(f"\nTrade log saved to: trade_log.csv")
    else:
        print("No trades were generated.")

if __name__ == '__main__':
    main()
