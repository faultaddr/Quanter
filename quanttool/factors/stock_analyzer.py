"""
Stock Analysis Module - Performs comprehensive technical analysis and generates buy/sell signals
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Tuple

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from quanttool.factors.tech_indicators import *
from quanttool.factors.trading_strategies import TradingStrategies
from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials


class StockAnalyzer:
    """
    A comprehensive stock analyzer that calculates technical indicators and evaluates trading strategies
    """

    def __init__(self):
        """Initialize the stock analyzer with data fetcher"""
        self.fetcher = create_data_fetcher_with_credentials()
        self.fetcher.initialize()

    def get_stock_data(self, symbol: str, days: int = 360) -> pd.DataFrame:
        """
        Fetch stock data for the specified symbol and time period
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Normalize symbol format for different data providers
        normalized_symbol = self._normalize_symbol(symbol)
        symbols = [normalized_symbol]

        print(f"Fetching data for {normalized_symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")

        data = self.fetcher.get_bars(symbols, start_date, end_date)

        if normalized_symbol in data and not data[normalized_symbol].empty:
            df = data[normalized_symbol].copy()
            print(f"Retrieved {len(df)} records")
            return df
        else:
            print(f"No data retrieved for {normalized_symbol}")
            return pd.DataFrame()

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize the symbol format for different data providers
        """
        symbol = symbol.upper().strip()

        # Handle various input formats
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            return symbol
        elif len(symbol) == 6:
            if symbol.startswith(('5', '6', '9')):
                return f"{symbol}.SH"  # Shanghai
            else:
                return f"{symbol}.SZ"  # Shenzhen
        else:
            return symbol

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various technical indicators for the given data
        """
        if df.empty or 'close' not in df.columns:
            print("Error: DataFrame is empty or missing 'close' column")
            return df

        df = df.sort_values('timestamp').reset_index(drop=True)

        # Ensure all required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col in ['open', 'high', 'low']:
                    # Use close price for missing OHLC data
                    df[col] = df['close']
                elif col == 'volume':
                    # Create a default volume column if missing
                    df[col] = 1000000  # Default volume

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        # Calculate technical indicators
        print("Calculating technical indicators...")

        # Moving Averages
        df['ma_5'] = MA(close, 5)
        df['ma_10'] = MA(close, 10)
        df['ma_20'] = MA(close, 20)
        df['ma_30'] = MA(close, 30)
        df['ma_50'] = MA(close, 50)
        df['ma_100'] = MA(close, 100)
        df['ma_200'] = MA(close, 200)

        # RSI
        df['rsi_6'] = RSI(close, 6)
        df['rsi_12'] = RSI(close, 12)
        df['rsi_24'] = RSI(close, 24)

        # MACD
        dif, dea, macd = MACD(close)
        df['macd_dif'] = dif
        df['macd_dea'] = dea
        df['macd'] = macd

        # KDJ
        k, d, j = KDJ(close, high, low)
        df['kdj_k'] = k
        df['kdj_d'] = d
        df['kdj_j'] = j

        # Bollinger Bands
        upper, mid, lower = BOLL(close)
        df['boll_upper'] = upper
        df['boll_mid'] = mid
        df['boll_lower'] = lower

        # BIAS
        bias_6, bias_12, bias_24 = BIAS(close)
        df['bias_6'] = bias_6
        df['bias_12'] = bias_12
        df['bias_24'] = bias_24

        # CCI
        df['cci'] = CCI(close, high, low)

        # ATR and True Range
        df['atr_14'] = ATR(close, high, low, 14)

        # DMI/DMI
        pdi, mdi, adx, adxr = DMI(close, high, low)
        df['dmi_pdi'] = pdi
        df['dmi_mdi'] = mdi
        df['dmi_adx'] = adx
        df['dmi_adxr'] = adxr

        # TRIX
        trix, trma = TRIX(close)
        df['trix'] = trix
        df['trix_ma'] = trma

        # VR
        df['vr'] = VR(close, volume)

        # CR
        df['cr'] = CR(close, high, low)

        # Williams %R
        wr, wr_6 = WR(close, high, low)
        df['wr'] = wr
        df['wr_6'] = wr_6

        # BBI
        df['bbi'] = BBI(close)

        # PSY
        psy, psyma = PSY(close)
        df['psy'] = psy
        df['psyma'] = psyma

        # Daily return
        df['daily_return'] = df['close'].pct_change()

        # Price position in the range
        df['price_position'] = (df['close'] - LLV(low, 20)) / (HHV(high, 20) - LLV(low, 20)) * 100

        print("Technical indicators calculated successfully.")
        return df

    def run_trading_strategies(self, df: pd.DataFrame) -> Dict:
        """
        Run various trading strategies on the given data
        """
        if df.empty:
            return {}

        print("Running trading strategies...")

        # Initialize strategy evaluator
        strategies = TradingStrategies()

        results = {}

        # Individual strategies
        results['rsi_strategy'] = strategies.rsi_strategy(df)
        results['macd_strategy'] = strategies.macd_strategy(df)
        results['ma_crossover_strategy'] = strategies.ma_crossover_strategy(df)
        results['bollinger_bands_strategy'] = strategies.bollinger_bands_strategy(df)
        results['combined_strategy'] = strategies.combined_strategy(df)

        # Evaluate current signals
        results['evaluations'] = {}
        results['evaluations']['rsi'] = strategies.evaluate_current_signal(results['rsi_strategy'], 'RSI Strategy')
        results['evaluations']['macd'] = strategies.evaluate_current_signal(results['macd_strategy'], 'MACD Strategy')
        results['evaluations']['ma_crossover'] = strategies.evaluate_current_signal(results['ma_crossover_strategy'], 'MA Crossover Strategy')
        results['evaluations']['bollinger_bands'] = strategies.evaluate_current_signal(results['bollinger_bands_strategy'], 'Bollinger Bands Strategy')
        results['evaluations']['combined'] = strategies.evaluate_current_signal(results['combined_strategy'], 'Combined Strategy')

        print("Trading strategies completed.")
        return results

    def generate_report(self, df: pd.DataFrame, strategies_results: Dict, symbol: str) -> str:
        """
        Generate a comprehensive analysis report
        """
        if df.empty:
            return "No data available for analysis."

        # Get latest data point
        latest_data = df.iloc[-1]

        report = []
        report.append("="*60)
        report.append(f"STOCK ANALYSIS REPORT FOR {symbol}")
        report.append("="*60)
        report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Analysis Period: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        report.append(f"Number of Trading Days: {len(df)}")
        report.append("")

        report.append("CURRENT MARKET DATA:")
        report.append("-"*30)
        report.append(f"Current Price: {latest_data['close']:.2f}")
        report.append(f"Today's Change: {latest_data['daily_return']*100:.2f}%")
        report.append(f"Volume: {latest_data['volume']:,}")
        report.append(f"High: {latest_data['high']:.2f} | Low: {latest_data['low']:.2f}")
        report.append(f"Open: {latest_data['open']:.2f}")
        report.append("")

        report.append("TECHNICAL INDICATORS:")
        report.append("-"*30)
        report.append(f"RSI(24): {latest_data['rsi_24']:.2f}")
        report.append(f"MACD: {latest_data['macd']:.2f}")
        report.append(f"KDJ_K: {latest_data['kdj_k']:.2f} | KDJ_D: {latest_data['kdj_d']:.2f} | KDJ_J: {latest_data['kdj_j']:.2f}")
        report.append(f"MA20: {latest_data['ma_20']:.2f} | MA50: {latest_data['ma_50']:.2f} | MA200: {latest_data['ma_200']:.2f}")
        report.append(f"BOLL_UPPER: {latest_data['boll_upper']:.2f} | BOLL_MID: {latest_data['boll_mid']:.2f} | BOLL_LOWER: {latest_data['boll_lower']:.2f}")
        report.append(f"CCI: {latest_data['cci']:.2f}")
        report.append(f"ATR(14): {latest_data['atr_14']:.2f}")
        report.append(f"DMIPDI: {latest_data['dmi_pdi']:.2f} | DMIMDI: {latest_data['dmi_mdi']:.2f} | ADX: {latest_data['dmi_adx']:.2f}")
        report.append(f"TRIX: {latest_data['trix']:.2f}")
        report.append(f"VR: {latest_data['vr']:.2f}")
        report.append(f"CR: {latest_data['cr']:.2f}")
        report.append(f"WR: {latest_data['wr']:.2f}")
        report.append(f"BBI: {latest_data['bbi']:.2f}")
        report.append("")

        report.append("TRADING STRATEGY EVALUATIONS:")
        report.append("-"*30)

        evaluations = strategies_results.get('evaluations', {})
        for strategy_name, eval_result in evaluations.items():
            if isinstance(eval_result, dict):
                report.append(f"{strategy_name.upper().replace('_', ' ')}:")
                report.append(f"  Current Signal: {eval_result.get('current_signal', 'N/A')}")
                report.append(f"  Action: {eval_result.get('action', 'N/A')}")
                report.append(f"  Confidence: {eval_result.get('confidence', 'N/A')}")
                report.append(f"  Signal Changed: {eval_result.get('signal_changed', 'N/A')}")
                report.append("")

        # Overall recommendation
        report.append("OVERALL RECOMMENDATION:")
        report.append("-"*30)

        # Count different types of signals
        combined_signal = evaluations.get('combined', {}).get('current_signal', 'HOLD')
        rsi_signal = evaluations.get('rsi', {}).get('current_signal', 'HOLD')
        macd_signal = evaluations.get('macd', {}).get('current_signal', 'HOLD')

        buy_signals = sum([
            combined_signal in ['STRONG_BUY', 'WEAK_BUY'],
            rsi_signal in ['BUY'],
            macd_signal in ['BUY']
        ])

        sell_signals = sum([
            combined_signal in ['STRONG_SELL', 'WEAK_SELL'],
            rsi_signal in ['SELL'],
            macd_signal in ['SELL']
        ])

        if buy_signals >= 2:
            report.append("  STRONG BUY: Multiple indicators suggest buying opportunity")
        elif buy_signals == 1:
            report.append("  WEAK BUY: Some indicators suggest potential buying opportunity")
        elif sell_signals >= 2:
            report.append("  STRONG SELL: Multiple indicators suggest selling opportunity")
        elif sell_signals == 1:
            report.append("  WEAK SELL: Some indicators suggest potential selling opportunity")
        else:
            report.append("  HOLD: Mixed signals, no clear direction")

        report.append("")
        report.append("DISCLAIMER: This analysis is for educational purposes only.")
        report.append("Investment decisions should be made based on comprehensive research and personal judgment.")

        return "\n".join(report)

    def analyze_stock(self, symbol: str, days: int = 360) -> str:
        """
        Main method to analyze a stock completely
        """
        print(f"Starting analysis for {symbol}...")

        # Get stock data
        df = self.get_stock_data(symbol, days)
        if df.empty:
            return f"Could not retrieve data for {symbol}"

        # Calculate technical indicators
        df_with_indicators = self.calculate_technical_indicators(df)

        # Run trading strategies
        strategies_results = self.run_trading_strategies(df_with_indicators)

        # Generate comprehensive report
        report = self.generate_report(df_with_indicators, strategies_results, symbol)

        return report


# Example usage
if __name__ == "__main__":
    analyzer = StockAnalyzer()

    # Example: Analyze a stock
    symbol = input("Enter stock symbol (e.g., 601777): ").strip()
    if symbol:
        report = analyzer.analyze_stock(symbol)
        print(report)

        # Optionally save to file
        filename = f"{symbol}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nAnalysis report saved to {filename}")