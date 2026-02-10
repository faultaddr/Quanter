"""
Main Application for A-Share Market Analysis
Real-time signals using EastMoney, backtesting using Tushare
"""
import sys
import os
from datetime import datetime
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.screeners.stock_screener import StockScreener
from quant_trade_a_share.strategies.strategy_tools import StrategyManager
from quant_trade_a_share.signals.signal_notifier import SignalNotifier, SignalProcessor
from quant_trade_a_share.backtest.backtester_tushare import BacktesterWithTushare


class AShareAnalyzer:
    """
    Main class that integrates all components of the A-Share analysis system
    Real-time signals use EastMoney, backtesting uses Tushare
    """
    def __init__(self, tushare_token=None):
        # Real-time data source
        self.screener = StockScreener()  # Using EastMoney primarily
        self.strategy_manager = StrategyManager()
        self.signal_notifier = SignalNotifier()
        self.signal_processor = SignalProcessor(self.signal_notifier)
        
        # Backtesting data source (only initialize if token provided)
        self.backtester = None
        if tushare_token:
            try:
                self.backtester = BacktesterWithTushare(tushare_token)
            except Exception as e:
                print(f"⚠️  无法初始化Tushare回测模块: {e}")
        
        self.recent_screenings = []
        self.recent_analyses = []
    
    def get_real_time_signals(self, symbols=None):
        """
        Generate real-time signals using EastMoney data
        """
        print("📡 获取实时信号...")
        
        if symbols is None:
            # Use sample symbols for demonstration
            symbols = ['000001.SZ', '600519.SH', '000858.SZ']  # Ping An, Kweichow Moutai, Wuliangye
        
        all_signals = []
        
        for symbol in symbols:
            print(f"📈 分析 {symbol}...")
            
            # Get data using screener (which prioritizes EastMoney)
            data = self.screener.fetch_stock_data(symbol, period='10', freq='D')
            
            if data is None or data.empty:
                print(f"⚠️  无法获取 {symbol} 的数据")
                continue
            
            # Apply strategies to generate signals
            for strategy_name in ['ma_crossover', 'rsi', 'macd']:
                try:
                    strategy = self.strategy_manager.get_strategy(strategy_name)
                    if strategy:
                        signals = strategy.generate_signals(data)
                        
                        # Process and send signals
                        stock_name = symbol  # Would normally fetch from stock list
                        
                        # Get the latest signal
                        if len(signals) > 0:
                            latest_signal = signals.iloc[-1]
                            latest_price = data['close'].iloc[-1] if 'close' in data.columns else None
                            latest_date = data.index[-1] if not data.empty else datetime.now()
                            
                            if latest_signal == 1:  # Buy signal
                                self.signal_notifier.add_signal(
                                    symbol=symbol,
                                    name=stock_name,
                                    signal_type="BUY",
                                    strategy=strategy_name,
                                    price=latest_price,
                                    reason=f"实时分析 {strategy_name}策略产生买入信号",
                                    priority=2
                                )
                                print(f"🟢 {symbol} - {strategy_name}: 买入信号 (¥{latest_price})")
                                
                            elif latest_signal == -1:  # Sell signal
                                self.signal_notifier.add_signal(
                                    symbol=symbol,
                                    name=stock_name,
                                    signal_type="SELL",
                                    strategy=strategy_name,
                                    price=latest_price,
                                    reason=f"实时分析 {strategy_name}策略产生卖出信号",
                                    priority=2
                                )
                                print(f"🔴 {symbol} - {strategy_name}: 卖出信号 (¥{latest_price})")
                            else:
                                print(f"⏸️ {symbol} - {strategy_name}: 持有信号")
                                
                except Exception as e:
                    print(f"⚠️  策略 {strategy_name} 在 {symbol} 上执行失败: {e}")
        
        return all_signals
    
    def run_backtest(self, strategy_name, symbol, start_date, end_date, initial_capital=100000, freq='D'):
        """
        Run backtest using Tushare data
        """
        if not self.backtester:
            print("❌ Tushare回测模块未初始化，请提供有效的token")
            return None
        
        print(f"🔬 运行 {strategy_name} 策略回测...")
        
        # Get the strategy
        strategy = self.strategy_manager.get_strategy(strategy_name)
        if not strategy:
            print(f"❌ 策略 {strategy_name} 不存在")
            return None
        
        # Run the backtest
        results = self.backtester.run_backtest(
            strategy=strategy,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            freq=freq
        )
        
        return results
    
    def get_latest_signals(self, limit=10):
        """
        Get the most recent signals
        """
        return self.signal_notifier.get_recent_signals(limit)


def main():
    """
    Main function demonstrating both real-time signals and backtesting
    """
    print("="*60)
    print("🎯 A股市场分析系统 (EastMoney实时信号 + Tushare回测)")
    print("="*60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Use the provided Tushare token for backtesting
    tushare_token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"

    # Initialize the analyzer
    analyzer = AShareAnalyzer(tushare_token=tushare_token)

    print("✅ 系统初始化完成")
    print()

    print("📡 1. 生成实时信号 (使用EastMoney数据):")
    analyzer.get_real_time_signals()
    print()
    
    print("📋 2. 最新信号摘要:")
    latest_signals = analyzer.get_latest_signals(5)
    for i, signal in enumerate(latest_signals, 1):
        signal_type = "🟢 买入" if signal['signal_type'] == 'BUY' else "🔴 卖出"
        print(f"  {i}. {signal_type} - {signal['symbol']} ({signal['strategy']}): {signal['reason'][:30]}...")
    print()
    
    print("🔬 3. 运行回测示例 (使用Tushare数据):")
    # Run a sample backtest
    backtest_results = analyzer.run_backtest(
        strategy_name='ma_crossover',
        symbol='000001.SZ',  # Ping An Bank
        start_date='20220101',
        end_date='20221231',
        initial_capital=100000,
        freq='D'
    )
    
    if backtest_results:
        print(f"   总收益率: {backtest_results['total_return']:.2%}")
        print(f"   年化收益率: {backtest_results['annualized_return']:.2%}")
        print(f"   最大回撤: {backtest_results['max_drawdown']:.2%}")
        print(f"   最终价值: ¥{backtest_results['final_value']:,.2f}")
    else:
        print("   回测未运行 (可能因权限限制)")
    
    print()
    print("💡 系统特点:")
    print("   • 实时信号: 基于EastMoney数据，稳定可靠")
    print("   • 历史回测: 基于Tushare数据，精确回溯")
    print("   • 多策略支持: 移动平均、RSI、MACD等")
    print("   • 信号通知: 实时推送买卖信号")
    print()
    print("✅ 系统运行完成!")


if __name__ == "__main__":
    main()