"""
Real-time A-Share Market Signal Generator
Using Tushare API for live data and generating 10-minute level buy/sell signals
"""
import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import tushare as ts

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.screeners.stock_screener import StockScreener
from quant_trade_a_share.strategies.strategy_tools import StrategyManager
from quant_trade_a_share.signals.signal_notifier import SignalNotifier, SignalProcessor
from quant_trade_a_share.prediction.predictive_analyzer import PredictiveAnalyzer


class RealTimeSignalGenerator:
    """
    Generates real-time buy/sell signals using 10-minute level data from Tushare
    """
    def __init__(self, tushare_token):
        # Initialize Tushare
        ts.set_token(tushare_token)
        self.pro = ts.pro_api()

        # Initialize components
        self.screener = StockScreener(tushare_token=tushare_token)
        self.strategy_manager = StrategyManager()
        self.signal_notifier = SignalNotifier()
        self.signal_processor = SignalProcessor(self.signal_notifier)
        self.predictive_analyzer = PredictiveAnalyzer()  # Initialize with default settings

        print("✅ 实时信号生成器初始化完成")
    
    def get_top_active_stocks(self, limit=10):
        """
        Get most actively traded stocks for signal generation
        """
        try:
            # Try to get daily market information using query method
            trade_date = datetime.now().strftime('%Y%m%d')
            df = self.pro.query('daily', trade_date=trade_date)
            
            if df is None or df.empty:
                print("⚠️ Tushare未返回当日数据，使用样本股票")
                return [
                    ('000001.SZ', '平安银行', 15.0, 10000000),
                    ('600519.SH', '贵州茅台', 1800.0, 5000000),
                    ('000858.SZ', '五粮液', 220.0, 8000000)
                ]
            
            # Check if required columns exist
            required_cols = ['ts_code', 'name', 'close', 'vol']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"⚠️ 数据缺少必要列: {missing_cols}，使用样本股票")
                return [
                    ('000001.SZ', '平安银行', 15.0, 10000000),
                    ('600519.SH', '贵州茅台', 1800.0, 5000000),
                    ('000858.SZ', '五粮液', 220.0, 8000000)
                ]
            
            # Filter for stocks with high volume and recent activity
            df = df[df['vol'] > 1000000]  # Volume > 1 million
            df = df.sort_values(by='vol', ascending=False)
            
            # Get top stocks by volume
            top_stocks = df.head(limit)[['ts_code', 'name', 'close', 'vol']].values
            print(f"📊 获取到 {len(top_stocks)} 只活跃股票")
            return top_stocks
        except Exception as e:
            print(f"⚠️ 获取活跃股票失败: {e}，使用样本股票")
            # Fallback to sample stocks
            return [
                ('000001.SZ', '平安银行', 15.0, 10000000),
                ('600519.SH', '贵州茅台', 1800.0, 5000000),
                ('000858.SZ', '五粮液', 220.0, 8000000)
            ]
    
    def generate_10min_signals(self, symbols=None):
        """
        Generate buy/sell signals based on 10-minute level data
        """
        if symbols is None:
            # Get top active stocks
            active_stocks = self.get_top_active_stocks()
            symbols = [stock[0] for stock in active_stocks]  # Get stock codes
        
        print(f"🔄 开始为 {len(symbols)} 只股票生成实时信号...")
        
        all_signals = []
        
        for symbol in symbols:
            try:
                print(f"📈 分析 {symbol}...")
                
                # For real-time signals, primarily use EastMoney (which is more reliable for current data)
                # 10-minute data is not typically available through free sources, so we'll use daily data
                # and generate intraday signals based on technical indicators
                df = self.screener.fetch_stock_data(symbol, period='5', freq='D')  # 5 days of daily data
                
                if df is None or df.empty:
                    print(f"⚠️  无法获取 {symbol} 的有效数据")
                    continue
                
                # Apply strategies to generate signals
                for strategy_name in ['ma_crossover', 'rsi', 'macd']:
                    try:
                        strategy = self.strategy_manager.get_strategy(strategy_name)
                        if strategy:
                            signals = strategy.generate_signals(df)
                            
                            # Process and send signals
                            stock_name = symbol  # Would normally fetch from stock list
                            
                            # Count recent signals
                            recent_signals = signals.tail(5)  # Last 5 intervals/periods
                            
                            for date, signal_val in recent_signals.items():
                                if signal_val == 1:  # Buy signal
                                    self.signal_notifier.add_signal(
                                        symbol=symbol,
                                        name=stock_name,
                                        signal_type="BUY",
                                        strategy=strategy_name,
                                        price=df.loc[date, 'close'] if 'close' in df.columns else None,
                                        reason=f"实时分析 {strategy_name}策略产生买入信号",
                                        priority=3
                                    )
                                    print(f"🟢 {symbol} [{date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date}] - {strategy_name}: 买入信号")
                                    
                                elif signal_val == -1:  # Sell signal
                                    self.signal_notifier.add_signal(
                                        symbol=symbol,
                                        name=stock_name,
                                        signal_type="SELL",
                                        strategy=strategy_name,
                                        price=df.loc[date, 'close'] if 'close' in df.columns else None,
                                        reason=f"实时分析 {strategy_name}策略产生卖出信号",
                                        priority=3
                                    )
                                    print(f"🔴 {symbol} [{date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else date}] - {strategy_name}: 卖出信号")
                            
                    except Exception as e:
                        print(f"⚠️  策略 {strategy_name} 在 {symbol} 上执行失败: {e}")
                
            except Exception as e:
                print(f"⚠️  处理 {symbol} 时发生错误: {e}")
        
        return all_signals
    
    def run_predictive_analysis(self, symbols=None, top_n=10):
        """
        Run predictive analysis to identify potentially rising stocks
        """
        print(f"🔍 开始运行预测分析，识别潜在上涨股票...")
        
        if symbols is None:
            # Get top active stocks for analysis
            active_stocks = self.get_top_active_stocks(limit=20)
            symbols = [stock[0] for stock in active_stocks]  # Get stock codes
        
        # Run predictive analysis
        predictions = self.predictive_analyzer.analyze_stocks(symbols=symbols, top_n=top_n)
        
        if not predictions.empty:
            print(f"✅ 预测分析完成，共分析 {len(predictions)} 只股票")
            
            # Print top predictions
            self.predictive_analyzer.print_top_predictions(predictions, top_n=min(top_n, len(predictions)))
            
            # Generate alerts for high-scoring stocks
            high_score_threshold = 3  # Threshold for alert generation
            alert_stocks = predictions[predictions['prediction_score'] >= high_score_threshold]
            
            if not alert_stocks.empty:
                print(f"\n🔔 发现 {len(alert_stocks)} 只高潜力股票 (预测分数 ≥ {high_score_threshold}):")
                for _, row in alert_stocks.iterrows():
                    self.signal_notifier.add_signal(
                        symbol=row['symbol'],
                        name=row['name'] if 'name' in row else row['symbol'],
                        signal_type="STRONG_BUY",
                        strategy="Predictive_Analysis",
                        price=row['current_price'] if 'current_price' in row else None,
                        reason=f"预测分析显示强劲上涨潜力，分数: {row['prediction_score']:.2f}",
                        priority=3
                    )
                    print(f"  🚀 {row['symbol']} - 预测分数: {row['prediction_score']:.2f}")
            else:
                print(f"\n✅ 未发现预测分数 ≥ {high_score_threshold} 的高潜力股票")
        else:
            print("⚠️ 预测分析未返回结果")
        
        return predictions

    def run_predictive_analysis(self, symbols=None, top_n=10):
        """
        Run predictive analysis to identify potentially rising stocks
        """
        print(f"🔍 开始运行预测分析，识别潜在上涨股票...")
        
        if symbols is None:
            # Get top active stocks for analysis
            active_stocks = self.get_top_active_stocks(limit=20)
            symbols = [stock[0] for stock in active_stocks]  # Get stock codes
        
        # Run predictive analysis
        predictions = self.predictive_analyzer.analyze_stocks(symbols=symbols, top_n=top_n)
        
        if not predictions.empty:
            print(f"✅ 预测分析完成，共分析 {len(predictions)} 只股票")
            
            # Print top predictions
            self.predictive_analyzer.print_top_predictions(predictions, top_n=min(top_n, len(predictions)))
            
            # Generate alerts for high-scoring stocks
            high_score_threshold = 3  # Threshold for alert generation
            alert_stocks = predictions[predictions['prediction_score'] >= high_score_threshold]
            
            if not alert_stocks.empty:
                print(f"\n🔔 发现 {len(alert_stocks)} 只高潜力股票 (预测分数 ≥ {high_score_threshold}):")
                for _, row in alert_stocks.iterrows():
                    self.signal_notifier.add_signal(
                        symbol=row['symbol'],
                        name=row['name'] if 'name' in row else row['symbol'],
                        signal_type="STRONG_BUY",
                        strategy="Predictive_Analysis",
                        price=row['current_price'] if 'current_price' in row else None,
                        reason=f"预测分析显示强劲上涨潜力，分数: {row['prediction_score']:.2f}",
                        priority=3
                    )
                    print(f"  🚀 {row['symbol']} - 预测分数: {row['prediction_score']:.2f}")
            else:
                print(f"\n✅ 未发现预测分数 ≥ {high_score_threshold} 的高潜力股票")
        else:
            print("⚠️ 预测分析未返回结果")
        
        return predictions

    def get_latest_signals(self, limit=10):
        """
        Get the most recent signals
        """
        return self.signal_notifier.get_recent_signals(limit)
    
    def monitor_continuously(self, interval_minutes=10):
        """
        Monitor the market continuously and generate signals
        """
        print(f"🔄 开始连续监控，每 {interval_minutes} 分钟更新一次...")
        
        import time
        while True:
            try:
                print(f"\n⏰ [{datetime.now().strftime('%H:%M:%S')}] 更新市场信号...")
                self.generate_10min_signals()
                
                # Get and display latest signals
                latest_signals = self.get_latest_signals(5)
                if latest_signals:
                    print(f"\n🔔 最新信号 (共{len(latest_signals)}条):")
                    for signal in latest_signals:
                        signal_type = "🟢 买入" if signal['signal_type'] == 'BUY' else "🔴 卖出"
                        print(f"  {signal_type} - {signal['symbol']} ({signal['strategy']}): {signal['reason'][:50]}...")
                
                print(f"⏳ 等待 {interval_minutes} 分钟后下次更新...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                print("\n🛑 监控已停止")
                break
            except Exception as e:
                print(f"❌ 监控过程中出错: {e}")
                time.sleep(60)  # Wait 1 minute before retrying


def main():
    """
    Main function to run the real-time signal generator with predictive analysis
    """
    print("="*70)
    print("🔥 A股实时10分钟级别信号生成与预测分析系统")
    print("="*70)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Use the provided tokens
    tushare_token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"
    
    # Your EastMoney cookie for 10-minute data access
    eastmoney_cookie = {
        'ASL': '20494,0000d,8be20aff',
        'ADVC': '3ee81b757962bc',
        'ADVS': '3ee81b757962bc',
        'qgqp_b_id': '5214d909bcc66e93576b49ed3d446e38',
        'st_nvi': 'PZZhsgK0ZsqG3vHBMU-4g0c46',
        'websitepoptg_api_time': '1770665319207',
        'st_si': '44545933999131',
        'nid18': '0cb935b80cd1336d400798228688f23e',
        'nid18_create_time': '1770665319416',
        'gviem': '_krPH3C3Ybs-kJyqdlhK9598d',
        'gviem_create_time': '1770665319416',
        'p_origin': 'https%3A%2F%2Fpassport2.eastmoney.com',
        'mtp': '1',
        'ct': 'j2-rb8gsYEH7Z5hfhA_9WkaiA66JMtMhasWm5IaNF7xSY0Q1QHUR8w2IC_dQlFzfQfbVcNBBm5MdHmEBSXRScIWFyHjzzm0mH1p8lwDeKo--nqL3nTKwKwg08w11_RniWauFoL3tWOwknftIoosjmHsSPjOdn1ZS5PLW_9pHC_4',
        'ut': 'FobyicMgeV6Gl5Ws0rOH5qvs-ZS0k9XvNXWKKa42q-agegqBk6oLosMw8RzR-iuurrDoc1kUl0jT5cRIAUAhTXaafTsuUZo5Ef0TELgIYsuL6W1cH-RjJf-IR6_Qb_7bwQSIRyKP4OqDlhze9fNwQZenBxx4FXFTxBmD9pS_ZoRqb7PVus-sZsyLgYm0tus-oDDyROxO-WE7MVpEDKxbC3s2cYKtYU4TTY8Lot4UXuHn6hUEv_N8tfb3sJyKA9-mxqVVLZYDNDmmRygALO7NNdoNYXTAebWI',
        'pi': '9694097255613200%3Bu9694097255613200%3B%E5%A0%82%E5%A0%82%E6%AD%A3%E6%AD%A3%E7%9A%84%E6%9B%B9%E6%93%8D%3BLSrFBlVclIYPg4pBOrim34v0hS8%2Bw2owuUFcpj0%2BGkIi897wjraBNPTUKgjtxkQI2Z%2BYVW%2F7zHPpH%2Bk7RMVMu8mEpKbMNOVi1ybo6%2FmJTuILjybcZZFcRv7BSbUUyjB4ZLRjN0ID%2FNmlx5RhlDRAyMBeC69O8A96P7KMdBllLB0qcPcL6XlKPyGwxj1OCCxdiivc1%2F4P%3BkKW5qgd%2BimVm5dzQstH7DrvYE%2FIlKvcz5fJwIAUTrjSGhqknW0d9oJwGyNBZY7%2Bbb97NLjZBiQgcOwDVpln%2F7sT7KuzufFPV6TUh0zWDyWjQUy2R6kly72KifsONMTLXsXx3r3ATLwQ4EmVGHXijrfcKBZQNAw%3D%3D',
        'uidal': '9694097255613200%e5%a0%82%e5%a0%82%e6%ad%a3%e6%ad%a3%e7%9a%84%e6%9b%b9%e6%93%8d',
        'sid': '',
        'vtpst': '|',
        'st_pvi': '04127630559918',
        'st_sp': '2026-02-10%2003%3A28%3A39',
        'st_inirUrl': 'https%3A%2F%2Fwww.baidu.com%2Flink',
        'st_sn': '3',
        'st_psi': '20260210032907843-111000300841-7036945406',
        'st_asi': 'delete',
        'fullscreengg': '1',
        'fullscreengg2': '1'
    }

    try:
        # Initialize the signal generator
        signal_gen = RealTimeSignalGenerator(tushare_token)
        
        # Update the predictive analyzer with EastMoney cookie
        signal_gen.predictive_analyzer.eastmoney_cookies = eastmoney_cookie

        print("\n🎯 系统功能:")
        print("  1. 获取活跃股票列表")
        print("  2. 生成10分钟级别买卖信号")
        print("  3. 预测分析 - 识别潜在上涨股票")
        print("  4. 实时监控和通知")
        print()

        # Generate initial signals
        print("🚀 生成初始信号...")
        active_stocks = signal_gen.get_top_active_stocks(limit=5)
        symbols = [stock[0] for stock in active_stocks]
        
        signals = signal_gen.generate_10min_signals(symbols)

        # Run predictive analysis
        print("\n🔍 运行预测分析...")
        predictions = signal_gen.run_predictive_analysis(symbols=symbols, top_n=10)

        # Show recent signals
        recent_signals = signal_gen.get_latest_signals(10)
        print(f"\n📋 最近生成的 {len(recent_signals)} 个信号:")
        for i, signal in enumerate(recent_signals[:5], 1):
            signal_type = "🟢 买入" if signal['signal_type'] == 'BUY' else "🔴 卖出" if signal['signal_type'] == 'SELL' else "⏸️  持有"
            print(f"  {i}. {signal_type} - {signal['symbol']} ({signal['strategy']})")

        print("\n🔄 是否启动连续监控模式? (每10分钟更新一次)")
        choice = input("输入 'y' 开始监控，或任何其他键退出: ").lower()

        if choice == 'y':
            print("\n🔄 连续监控已启动... (按 Ctrl+C 停止)")
            try:
                while True:
                    time.sleep(600)  # Wait 10 minutes
                    print(f"\n🔄 [{datetime.now().strftime('%H:%M:%S')}] 更新市场信号...")
                    
                    # Refresh 10-minute signals
                    new_signals = signal_gen.generate_10min_signals(symbols)
                    
                    # Run predictive analysis every 30 minutes (every 3 cycles)
                    if int(time.time() / 600) % 3 == 0:  # Every 3rd cycle (30 minutes)
                        print(f"\n🔍 [{datetime.now().strftime('%H:%M:%S')}] 运行预测分析...")
                        predictions = signal_gen.run_predictive_analysis(symbols=symbols, top_n=10)
                    
                    # Get and display latest signals
                    latest = signal_gen.get_latest_signals(5)
                    if latest:
                        print(f"🔔 最新信号 (共{len(latest)}条):")
                        for signal in latest:
                            signal_type = "🟢 买入" if signal['signal_type'] == 'BUY' else "🔴 卖出" if signal['signal_type'] == 'SELL' else "⏸️  持有"
                            print(f"  {signal_type} - {signal['symbol']} ({signal['strategy']}): {signal['reason'][:30]}...")
            except KeyboardInterrupt:
                print("\n❌ 连续监控已停止")
        else:
            print("\n✅ 系统运行完成!")

    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查Tushare token是否正确以及网络连接是否正常")


if __name__ == "__main__":
    main()