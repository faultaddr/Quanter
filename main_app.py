"""
Main Application for A-Share Market Analysis
Integrates stock screening, strategy analysis, and signal generation
"""
import os
import sys
import argparse
from datetime import datetime
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from quant_trade_a_share.screeners.stock_screener import StockScreener
from quant_trade_a_share.strategies.strategy_tools import StrategyManager
from quant_trade_a_share.signals.signal_notifier import SignalNotifier, SignalProcessor
from quant_trade_a_share.viz.dashboard import app


class AShareAnalyzer:
    """
    Main class that integrates all components of the A-Share analysis system
    """
    def __init__(self, tushare_token=None):
        self.screener = StockScreener(tushare_token=tushare_token)
        self.strategy_manager = StrategyManager()
        self.signal_notifier = SignalNotifier()
        self.signal_processor = SignalProcessor(self.signal_notifier)
        self.recent_screenings = []
        self.recent_analyses = []
    
    def screen_stocks(self, filters=None):
        """
        Screen stocks based on provided filters
        """
        print("🔍 开始筛选股票...")
        
        results = self.screener.screen_stocks(filters)
        
        if results is not None and not results.empty:
            print(f"✅ 找到 {len(results)} 只符合条件的股票")
            self.recent_screenings.append({
                'timestamp': datetime.now(),
                'filters': filters,
                'results': results
            })
            return results
        else:
            print("❌ 未找到符合条件的股票")
            return pd.DataFrame()
    
    def analyze_stock(self, symbol, strategy_name):
        """
        Analyze a specific stock with a given strategy
        """
        print(f"📊 正在分析股票 {symbol} 使用 {strategy_name} 策略...")
        
        # Get stock data
        data = self.screener.fetch_stock_data(symbol, period='180')
        if data is None or data.empty:
            print(f"❌ 无法获取股票 {symbol} 的数据")
            return None
        
        # Get stock name from screener data
        stock_info = self.screener.chinese_stocks[self.screener.chinese_stocks['symbol'] == symbol]
        stock_name = stock_info['name'].iloc[0] if not stock_info.empty else symbol
        
        # Run strategy
        strategy = self.strategy_manager.get_strategy(strategy_name)
        if strategy is None:
            print(f"❌ 策略 {strategy_name} 不存在")
            return None
        
        signals = strategy.generate_signals(data)
        
        # Calculate performance metrics
        buy_signals = signals[signals == 1]
        sell_signals = signals[signals == -1]
        
        # Process signals and generate notifications
        self.signal_processor.process_strategy_signals(
            symbol=symbol,
            name=stock_name,
            strategy_name=strategy_name,
            data=data,
            signals=signals
        )
        
        # Create analysis result
        analysis_result = {
            'symbol': symbol,
            'name': stock_name,
            'strategy': strategy_name,
            'data': data,
            'signals': signals,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_signals': len(buy_signals) + len(sell_signals),
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'current_signal': signals.iloc[-1] if len(signals) > 0 else 0,
            'timestamp': datetime.now()
        }
        
        self.recent_analyses.append(analysis_result)
        print(f"✅ 分析完成，共生成 {analysis_result['total_signals']} 个信号，已发送通知")
        
        return analysis_result
    
    def get_top_opportunities(self, n=5):
        """
        Get top opportunities based on screening results
        """
        if not self.recent_screenings:
            print("⚠️  没有可用的筛选结果")
            return pd.DataFrame()
        
        latest_screening = self.recent_screenings[-1]
        results = latest_screening['results']
        
        if len(results) == 0:
            print("⚠️  最近的筛选结果为空")
            return pd.DataFrame()
        
        # Return top N opportunities based on score
        top_opportunities = results.nlargest(n, 'score')
        print(f"🏆 前 {min(n, len(top_opportunities))} 只机会股票:")
        for idx, (_, row) in enumerate(top_opportunities.iterrows(), 1):
            print(f"  {idx}. {row['name']} ({row['code']}) - 潜力分数: {row['score']}")
        
        return top_opportunities
    
    def generate_signals_summary(self):
        """
        Generate a summary of recent signals
        """
        # Get recent signals from the database
        recent_signals = self.signal_notifier.get_recent_signals(10)
        
        if not recent_signals:
            print("⚠️  没有最近的交易信号")
            return
        
        print("\n🔔 最近的交易信号摘要:")
        for signal in recent_signals:
            signal_text = ""
            if signal['signal_type'] == 'BUY':
                signal_text = "📈 买入信号"
            elif signal['signal_type'] == 'SELL':
                signal_text = "📉 卖出信号"
            else:
                signal_text = "⏸️  持有信号"
            
            print(f"  • {signal['timestamp']} - {signal['symbol']} ({signal['name']}): {signal_text}")
            print(f"    策略: {signal['strategy']}, 价格: ¥{signal['price'] or 'N/A'}")
    
    def run_dashboard(self):
        """
        Run the web dashboard
        """
        print("🚀 启动A股市场分析仪表板...")
        print("🌐 请在浏览器中打开 http://127.0.0.1:8050")
        app.run_server(debug=True, host='0.0.0.0', port=8050)


def main():
    parser = argparse.ArgumentParser(description='A-Share Market Analysis Tool')
    parser.add_argument('--mode', choices=['screen', 'analyze', 'dashboard', 'summary'], 
                       default='dashboard', help='运行模式')
    parser.add_argument('--symbol', type=str, help='要分析的股票代码')
    parser.add_argument('--strategy', type=str, help='使用的策略名称')
    parser.add_argument('--top-n', type=int, default=5, help='显示前N只机会股票')
    
    args = parser.parse_args()
    
    analyzer = AShareAnalyzer()
    
    if args.mode == 'screen':
        # Default filters
        filters = {
            'min_price': 10,
            'max_price': 150,
            'min_volume': 5000000,
            'days_back': 60,
            'min_return': 0.02,
            'max_volatility': 0.04
        }
        
        results = analyzer.screen_stocks(filters)
        if not results.empty:
            print("\n筛选结果:")
            print(results.head(10).to_string(index=False))
    
    elif args.mode == 'analyze':
        if not args.symbol or not args.strategy:
            print("❌ 请提供股票代码和策略名称")
            return
        
        result = analyzer.analyze_stock(args.symbol, args.strategy)
        if result:
            print(f"\n分析结果 for {args.symbol} ({args.strategy}):")
            print(f"总信号数: {result['total_signals']}")
            print(f"买入信号: {result['buy_count']}")
            print(f"卖出信号: {result['sell_count']}")
            signal_desc = "买入" if result['current_signal'] == 1 else "卖出" if result['current_signal'] == -1 else "持有"
            print(f"当前信号: {signal_desc}")
    
    elif args.mode == 'summary':
        analyzer.screen_stocks({
            'min_price': 10,
            'max_price': 150,
            'min_volume': 5000000,
            'days_back': 60,
            'min_return': 0.02,
            'max_volatility': 0.04
        })
        
        print("\n" + "="*50)
        top_opps = analyzer.get_top_opportunities(args.top_n)
        
        print("\n" + "="*50)
        analyzer.generate_signals_summary()
    
    elif args.mode == 'dashboard':
        analyzer.run_dashboard()


if __name__ == "__main__":
    main()