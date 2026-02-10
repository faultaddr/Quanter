"""
Example usage of the Quantitative Trading Tool for A-Share Market
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from quant_trade_a_share import run_backtest, analyze_performance
from quant_trade_a_share.strategies import MovingAverageCrossoverStrategy, RSIStrategy
from quant_trade_a_share.utils.mock_data_generator import generate_mock_data


def main():
    print("Quantitative Trading Tool for A-Share Market - Example Usage")
    print("="*60)
    
    # Generate mock data for testing
    print("Generating mock stock data...")
    data = generate_mock_data('000001', '2022-01-01', '2022-12-31')
    print(f"Generated {len(data)} days of data from {data.index[0]} to {data.index[-1]}")
    print()
    
    # Example 1: Moving Average Crossover Strategy
    print("Example 1: Moving Average Crossover Strategy")
    print("-" * 40)
    
    ma_strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)
    
    # Run backtest using the backtester directly with mock data
    from quant_trade_a_share.backtest.backtester import Backtester
    backtester = Backtester()
    
    ma_results = backtester.run_with_data(
        ma_strategy,
        data,
        initial_capital=100000,
        symbol='000001'
    )
    
    print(f"Initial Capital: ¥{ma_results['initial_capital']:,.2f}")
    print(f"Final Value: ¥{ma_results['final_value']:,.2f}")
    print(f"Total Return: {ma_results['total_return']:.2%}")
    
    # Analyze performance
    ma_metrics = analyze_performance(ma_results)
    print(f"Sharpe Ratio: {ma_metrics.get('sharpe_ratio', 'N/A'):.2f}")
    print(f"Max Drawdown: {ma_metrics.get('max_drawdown', 'N/A'):.2%}")
    print(f"Win Rate: {ma_metrics.get('win_rate', 'N/A'):.2%}")
    print()
    
    # Example 2: RSI Strategy
    print("Example 2: RSI Strategy")
    print("-" * 40)
    
    rsi_strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)
    
    rsi_results = backtester.run_with_data(
        rsi_strategy,
        data,
        initial_capital=100000,
        symbol='000001'
    )
    
    print(f"Initial Capital: ¥{rsi_results['initial_capital']:,.2f}")
    print(f"Final Value: ¥{rsi_results['final_value']:,.2f}")
    print(f"Total Return: {rsi_results['total_return']:.2%}")
    
    # Analyze performance
    rsi_metrics = analyze_performance(rsi_results)
    print(f"Sharpe Ratio: {rsi_metrics.get('sharpe_ratio', 'N/A'):.2f}")
    print(f"Max Drawdown: {rsi_metrics.get('max_drawdown', 'N/A'):.2%}")
    print(f"Win Rate: {rsi_metrics.get('win_rate', 'N/A'):.2%}")
    print()
    
    # Example 3: Compare strategies
    print("Example 3: Strategy Comparison")
    print("-" * 40)
    
    from quant_trade_a_share.analysis.performance_analyzer import PerformanceAnalyzer
    analyzer = PerformanceAnalyzer()
    
    comparison = analyzer.compare_strategies(
        [ma_results, rsi_results],
        ['Moving Average Crossover', 'RSI Strategy']
    )
    
    print(comparison)
    print()
    
    print("Example completed successfully!")


if __name__ == "__main__":
    main()