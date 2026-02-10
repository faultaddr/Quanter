#!/usr/bin/env python3
"""
A-Share Market Analysis Console
Unified interface for all system functions with interactive commands
"""
import sys
import os
from datetime import datetime
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_trade_a_share.screeners.stock_screener import StockScreener
from quant_trade_a_share.strategies.strategy_tools import StrategyManager
from quant_trade_a_share.signals.signal_notifier import SignalNotifier
from quant_trade_a_share.prediction.predictive_analyzer import PredictiveAnalyzer
from quant_trade_a_share.realtime_signals import RealTimeSignalGenerator


class ASConsole:
    """
    Unified console for A-Share market analysis system
    Provides interactive access to all system functions
    """
    def __init__(self, tushare_token, eastmoney_cookie):
        self.tushare_token = tushare_token
        self.eastmoney_cookie = eastmoney_cookie
        
        # Initialize all system components
        self.screener = StockScreener()
        self.strategy_manager = StrategyManager()
        self.signal_notifier = SignalNotifier()
        self.predictive_analyzer = PredictiveAnalyzer()
        self.realtime_generator = RealTimeSignalGenerator(tushare_token)
        
        # Update predictive analyzer with EastMoney cookie
        self.predictive_analyzer.eastmoney_cookies = eastmoney_cookie
        
        print("✅ 系统组件初始化完成")
        
        # Store session data
        self.session_data = {}
        self.current_stocks = []
        
        print("✅ A股市场分析系统控制台初始化完成")
        print("="*60)
        self.show_help()
    
    def show_help(self):
        """
        Display help information with available commands
        """
        print("""
🤖 A股市场分析系统控制台 - 可用命令:
=======================================
📈 市场分析类:
  1. screen_stocks    - 筛选潜在上涨股票 (市值>200亿)
  2. analyze_stock    - 分析单个股票
  3. predict_stocks   - 预测股票上涨概率

📊 策略信号类:
  4. run_strategy     - 运行指定策略
  5. gen_signals      - 生成买卖信号
  6. show_signals     - 显示最新信号

🔍 数据查询类:
  7. get_data         - 获取股票数据
  8. calc_indicators  - 计算技术指标
  9. show_top_stocks  - 显示热门股票

📈 预测分析类:
  10. predictive_analysis - 运行预测分析
  11. top_predictions   - 显示Top预测
  12. analyze_market    - 市场整体分析

⚙️  系统管理类:
  13. show_session     - 显示会话数据
  14. clear_session    - 清空会话数据
  15. help             - 显示帮助信息
  16. quit/exit        - 退出系统

💡 使用方法: 输入命令编号或命令名称
   例如: 输入 '1' 或 'screen_stocks' 开始股票筛选
=======================================
        """)
    
    def run(self):
        """
        Run the interactive console
        """
        print(f"🚀 启动A股市场分析系统控制台 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("输入 'help' 查看可用命令，输入 'quit' 退出系统\n")
        
        while True:
            try:
                user_input = input(">>>(请输入命令): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 感谢使用A股市场分析系统，再见！")
                    break
                elif user_input.lower() in ['help', 'h', '?']:
                    self.show_help()
                elif user_input.isdigit():
                    # Handle numeric command
                    cmd_num = int(user_input)
                    self.handle_numeric_command(cmd_num)
                elif user_input in self.get_command_map():
                    # Handle command by name
                    self.execute_command(user_input)
                else:
                    print(f"❌ 未知命令: {user_input}")
                    self.show_help()
                    
            except KeyboardInterrupt:
                print("\n\n👋 系统被用户中断，再见！")
                break
            except Exception as e:
                print(f"❌ 执行命令时出错: {e}")
    
    def get_command_map(self):
        """
        Get mapping of command names to functions
        """
        return {
            'screen_stocks': self.screen_stocks,
            'analyze_stock': self.analyze_stock,
            'predict_stocks': self.predict_stocks,
            'run_strategy': self.run_strategy,
            'gen_signals': self.gen_signals,
            'show_signals': self.show_signals,
            'get_data': self.get_data,
            'calc_indicators': self.calc_indicators,
            'show_top_stocks': self.show_top_stocks,
            'predictive_analysis': self.predictive_analysis,
            'top_predictions': self.top_predictions,
            'analyze_market': self.analyze_market,
            'show_session': self.show_session,
            'clear_session': self.clear_session,
            'help': self.show_help
        }
    
    def handle_numeric_command(self, cmd_num):
        """
        Handle command by number
        """
        cmd_map = {
            1: self.screen_stocks,
            2: self.analyze_stock, 
            3: self.predict_stocks,
            4: self.run_strategy,
            5: self.gen_signals,
            6: self.show_signals,
            7: self.get_data,
            8: self.calc_indicators,
            9: self.show_top_stocks,
            10: self.predictive_analysis,
            11: self.top_predictions,
            12: self.analyze_market,
            13: self.show_session,
            14: self.clear_session,
            15: self.show_help,
            16: lambda: (print("👋 感谢使用A股市场分析系统，再见！"), exit(0))
        }
        
        if cmd_num in cmd_map:
            try:
                cmd_map[cmd_num]()
            except Exception as e:
                print(f"❌ 执行命令时出错: {e}")
        else:
            print(f"❌ 无效的命令编号: {cmd_num}")
    
    def execute_command(self, cmd_name):
        """
        Execute command by name
        """
        cmd_map = self.get_command_map()
        if cmd_name in cmd_map:
            try:
                cmd_map[cmd_name]()
            except Exception as e:
                print(f"❌ 执行 {cmd_name} 时出错: {e}")
        else:
            print(f"❌ 未知命令: {cmd_name}")
    
    def screen_stocks(self):
        """
        Screen for potentially rising stocks
        """
        print("\n🔍 开始筛选潜在上涨股票 (市值>200亿)...")
        
        # Use default filters
        filters = {
            'min_price': 10,
            'max_price': 150,
            'min_volume': 5000000,
            'days_back': 60,
            'min_return': 0.02,
            'max_volatility': 0.04
        }
        
        print(f"📊 使用筛选条件: {filters}")
        
        try:
            results = self.screener.screen_stocks(filters)
            if not results.empty:
                print(f"\n✅ 筛选完成，找到 {len(results)} 只符合条件的股票:")
                print(results.head(10).to_string(index=False))
                self.session_data['screened_stocks'] = results
            else:
                print("⚠️  未找到符合条件的股票")
        except Exception as e:
            print(f"❌ 筛选过程出错: {e}")
    
    def analyze_stock(self):
        """
        Analyze a specific stock
        """
        symbol = input("请输入股票代码 (例: sh600519): ").strip()
        if not symbol:
            print("❌ 股票代码不能为空")
            return
        
        strategy_name = input("请输入策略名称 (ma_crossover/rsi/macd/bollinger/mean_reversion/breakout，默认: ma_crossover): ").strip() or 'ma_crossover'
        
        print(f"\n📊 分析股票 {symbol} 使用 {strategy_name} 策略...")
        
        try:
            # Get stock data
            data = self.screener.fetch_stock_data(symbol)
            if data is None or data.empty:
                print(f"❌ 无法获取 {symbol} 的数据")
                return
            
            # Get stock name
            stock_info = self.screener.chinese_stocks[self.screener.chinese_stocks['symbol'] == symbol]
            stock_name = stock_info['name'].iloc[0] if not stock_info.empty and 'name' in stock_info.columns else symbol
            
            # Get strategy
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if strategy is None:
                print(f"❌ 策略 {strategy_name} 不存在")
                return
            
            # Generate signals
            signals = strategy.generate_signals(data)
            
            # Calculate recent performance
            recent_performance = ((data['close'].iloc[-1] - data['close'].iloc[0]) / 
                                 data['close'].iloc[0]) * 100
            
            print(f"\n✅ {symbol} ({stock_name}) 分析完成:")
            print(f"   当前价格: {data['close'].iloc[-1]:.2f}")
            print(f"   近期表现: {recent_performance:.2f}%")
            print(f"   生成信号数: {len(signals[signals != 0])}")
            print(f"   最新信号: {signals.iloc[-1]}")
            
            # Store in session
            self.session_data[f'analysis_{symbol}'] = {
                'symbol': symbol,
                'name': stock_name,
                'data': data,
                'signals': signals,
                'recent_performance': recent_performance
            }
            
        except Exception as e:
            print(f"❌ 分析过程出错: {e}")
    
    def predict_stocks(self):
        """
        Predict stock movements
        """
        symbols_input = input("请输入股票代码 (用逗号分隔，留空使用默认): ").strip()
        if symbols_input:
            symbols = [s.strip() for s in symbols_input.split(',')]
        else:
            symbols = ['sh600519', 'sz000858', 'sh600036']  # Default symbols
        
        top_n = input("请输入返回数量 (默认: 10): ").strip()
        top_n = int(top_n) if top_n.isdigit() else 10
        
        print(f"\n🔮 预测 {len(symbols)} 只股票的上涨概率...")
        
        try:
            predictions = self.predictive_analyzer.analyze_stocks(symbols=symbols, top_n=top_n)
            
            if not predictions.empty:
                print(f"\n✅ 预测完成，共分析 {len(predictions)} 只股票:")
                self.predictive_analyzer.print_top_predictions(predictions, top_n=min(top_n, len(predictions)))
                
                # Store predictions
                self.session_data['predictions'] = predictions
            else:
                print("⚠️  预测分析未返回结果")
        except Exception as e:
            print(f"❌ 预测过程出错: {e}")
    
    def run_strategy(self):
        """
        Run a specific strategy
        """
        strategy_name = input("请输入策略名称 (ma_crossover/rsi/macd/bollinger/mean_reversion/breakout): ").strip()
        if not strategy_name:
            print("❌ 策略名称不能为空")
            return
        
        symbols_input = input("请输入股票代码 (用逗号分隔): ").strip()
        if not symbols_input:
            print("❌ 请至少输入一只股票代码")
            return
        
        symbols = [s.strip() for s in symbols_input.split(',')]
        
        print(f"\n🏃 运行 {strategy_name} 策略...")
        
        try:
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if strategy is None:
                print(f"❌ 策略 {strategy_name} 不存在")
                return
            
            all_results = []
            for symbol in symbols:
                print(f"📈 分析 {symbol}...")
                
                data = self.screener.fetch_stock_data(symbol)
                if data is None or data.empty:
                    print(f"⚠️  无法获取 {symbol} 数据，跳过")
                    continue
                
                signals = strategy.generate_signals(data)
                signal_count = len(signals[signals != 0])
                
                result = {
                    'symbol': symbol,
                    'signal_count': signal_count,
                    'latest_signal': signals.iloc[-1] if len(signals) > 0 else 0,
                    'current_price': data['close'].iloc[-1] if 'close' in data.columns else 0
                }
                all_results.append(result)
            
            print(f"\n✅ 策略执行完成:")
            for result in all_results:
                signal_text = "📈 买入" if result['latest_signal'] == 1 else "🔴 卖出" if result['latest_signal'] == -1 else "⏸️  持有"
                print(f"   {result['symbol']}: {signal_text}, 信号数: {result['signal_count']}, 价格: ¥{result['current_price']:.2f}")
            
            # Store results
            self.session_data[f'strategy_{strategy_name}'] = all_results
            
        except Exception as e:
            print(f"❌ 策略执行出错: {e}")
    
    def gen_signals(self):
        """
        Generate buy/sell signals
        """
        symbols_input = input("请输入股票代码 (用逗号分隔): ").strip()
        if not symbols_input:
            print("❌ 请至少输入一只股票代码")
            return
        
        symbols = [s.strip() for s in symbols_input.split(',')]
        
        print(f"\n🔔 为 {len(symbols)} 只股票生成买卖信号...")
        
        try:
            signals = self.realtime_generator.generate_10min_signals(symbols)
            
            if len(signals) > 0:
                print(f"\n✅ 信号生成完成，共生成 {len(signals)} 个信号:")
                for signal in signals:
                    signal_type = "🟢 买入" if signal['signal_type'] == 'BUY' else "🔴 卖出" if signal['signal_type'] == 'SELL' else "⏸️  持有"
                    print(f"   {signal_type} - {signal['symbol']} ({signal['strategy']}): {signal['reason']}")
            else:
                print("⚠️  未生成任何信号")
        except Exception as e:
            print(f"❌ 信号生成出错: {e}")
    
    def show_signals(self):
        """
        Show latest signals
        """
        try:
            latest_signals = self.signal_notifier.get_recent_signals(10)
            
            if latest_signals:
                print(f"\n🔔 最新 {len(latest_signals)} 个信号:")
                for i, signal in enumerate(latest_signals, 1):
                    signal_type = "🟢 买入" if signal['signal_type'] == 'BUY' else "🔴 卖出" if signal['signal_type'] == 'SELL' else "⏸️  持有"
                    print(f"  {i}. {signal_type} - {signal['symbol']} ({signal['strategy']}): {signal['reason'][:30]}...")
            else:
                print("\n✅ 暂无最新信号")
        except Exception as e:
            print(f"❌ 获取信号出错: {e}")
    
    def get_data(self):
        """
        Get stock data
        """
        symbol = input("请输入股票代码 (例: sh600519): ").strip()
        if not symbol:
            print("❌ 股票代码不能为空")
            return
        
        days = input("请输入获取天数 (默认: 30): ").strip()
        days = int(days) if days.isdigit() else 30
        
        print(f"\n📊 获取 {symbol} 最近 {days} 天数据...")
        
        try:
            data = self.screener.fetch_stock_data(symbol, days)
            
            if data is not None and not data.empty:
                print(f"\n✅ 获取到 {len(data)} 条数据:")
                print(data[['open', 'close', 'high', 'low', 'volume']].tail(5).to_string())
                
                # Store in session
                self.session_data[f'data_{symbol}'] = data
            else:
                print("⚠️  无法获取数据")
        except Exception as e:
            print(f"❌ 获取数据出错: {e}")
    
    def calc_indicators(self):
        """
        Calculate technical indicators
        """
        symbol = input("请输入股票代码 (例: sh600519): ").strip()
        if not symbol:
            print("❌ 股票代码不能为空")
            return
        
        print(f"\n🧮 计算 {symbol} 技术指标...")
        
        try:
            data = self.screener.fetch_stock_data(symbol)
            
            if data is None or data.empty:
                print(f"❌ 无法获取 {symbol} 数据")
                return
            
            # Calculate indicators
            data = self.screener.calculate_technical_indicators(data)
            
            print(f"\n✅ 技术指标计算完成:")
            if 'rsi' in data.columns:
                print(f"   RSI: {data['rsi'].iloc[-1]:.2f}")
            if 'macd' in data.columns:
                print(f"   MACD: {data['macd'].iloc[-1]:.2f}")
            if 'ma_5' in data.columns:
                print(f"   MA5: {data['ma_5'].iloc[-1]:.2f}")
            if 'ma_20' in data.columns:
                print(f"   MA20: {data['ma_20'].iloc[-1]:.2f}")
            
            # Store in session
            self.session_data[f'indicators_{symbol}'] = data
            
        except Exception as e:
            print(f"❌ 计算指标出错: {e}")
    
    def show_top_stocks(self):
        """
        Show top active stocks
        """
        print("\n🔝 获取热门股票列表...")
        
        try:
            top_stocks = self.screener.get_top_active_stocks(limit=10)
            
            if top_stocks:
                print(f"\n✅ 获取到 {len(top_stocks)} 只热门股票:")
                for i, stock in enumerate(top_stocks, 1):
                    print(f"  {i}. {stock[1]} ({stock[0]}) - 价格: ¥{stock[2]:.2f}, 成交量: {stock[3]:,}")
                
                # Store in session
                self.session_data['top_stocks'] = top_stocks
            else:
                print("⚠️  无法获取热门股票列表")
        except Exception as e:
            print(f"❌ 获取热门股票出错: {e}")
    
    def predictive_analysis(self):
        """
        Run predictive analysis
        """
        print("\n🔮 运行预测分析...")
        
        try:
            # Get top active stocks for analysis
            top_stocks = self.screener.get_top_active_stocks(limit=20)
            symbols = [stock[0] for stock in top_stocks] if top_stocks else ['sh600519', 'sz000858']
            
            predictions = self.predictive_analyzer.analyze_stocks(symbols=symbols, top_n=10)
            
            if not predictions.empty:
                print(f"\n✅ 预测分析完成，共分析 {len(predictions)} 只股票:")
                self.predictive_analyzer.print_top_predictions(predictions, top_n=10)
                
                # Store in session
                self.session_data['predictions'] = predictions
            else:
                print("⚠️  预测分析未返回结果")
        except Exception as e:
            print(f"❌ 预测分析出错: {e}")
    
    def top_predictions(self):
        """
        Show top predictions from session
        """
        if 'predictions' in self.session_data:
            predictions = self.session_data['predictions']
            if not predictions.empty:
                print(f"\n🏆 Top 预测结果 (共{len(predictions)}只):")
                self.predictive_analyzer.print_top_predictions(predictions, top_n=min(10, len(predictions)))
            else:
                print("\n⚠️  会话中无预测结果")
        else:
            print("\n⚠️  会话中无预测结果，请先运行预测分析")
    
    def analyze_market(self):
        """
        Analyze overall market
        """
        print("\n🏛️  市场整体分析...")
        
        try:
            # Get market overview data
            top_stocks = self.screener.get_top_active_stocks(limit=50)
            
            if top_stocks:
                print(f"\n📊 市场概览 (共{len(top_stocks)}只活跃股票):")
                
                # Calculate market statistics
                total_rising = 0
                total_falling = 0
                total_volume = 0
                
                for stock in top_stocks:
                    symbol = stock[0]
                    try:
                        data = self.screener.fetch_stock_data(symbol, days=5)
                        if data is not None and not data.empty and len(data) >= 2:
                            # Calculate daily change
                            prev_close = data['close'].iloc[-2]
                            curr_close = data['close'].iloc[-1]
                            change_pct = (curr_close - prev_close) / prev_close * 100
                            
                            if change_pct > 0:
                                total_rising += 1
                            elif change_pct < 0:
                                total_falling += 1
                            
                            total_volume += data['volume'].iloc[-1] if 'volume' in data.columns else 0
                    except:
                        continue
                
                if len(top_stocks) > 0:
                    avg_volume = total_volume / len(top_stocks)
                    rising_pct = (total_rising / len(top_stocks)) * 100
                    falling_pct = (total_falling / len(top_stocks)) * 100
                    
                    print(f"   上涨股票: {total_rising} 只 ({rising_pct:.1f}%)")
                    print(f"   下跌股票: {total_falling} 只 ({falling_pct:.1f}%)")
                    print(f"   平均成交量: {avg_volume:,.0f}")
                    print(f"   市场情绪: {'📈 看涨' if rising_pct > falling_pct else '📉 看跌' if falling_pct > rising_pct else '⏸️ 中性'}")
            else:
                print("⚠️  无法获取市场概览数据")
        except Exception as e:
            print(f"❌ 市场分析出错: {e}")
    
    def show_session(self):
        """
        Show session data
        """
        print("\n💾 当前会话数据:")
        
        if self.session_data:
            for key, value in self.session_data.items():
                if isinstance(value, pd.DataFrame):
                    print(f"  📊 {key}: DataFrame with {len(value)} rows")
                elif isinstance(value, list):
                    print(f"  📋 {key}: List with {len(value)} items")
                elif isinstance(value, dict):
                    print(f"  📁 {key}: Dictionary with {len(value)} keys")
                else:
                    print(f"  📝 {key}: {type(value).__name__}")
        else:
            print("  📭 会话中无数据")
    
    def clear_session(self):
        """
        Clear session data
        """
        self.session_data = {}
        print("\n🗑️  会话数据已清空")


def main():
    """
    Main function to run the interactive console
    """
    print("🔍 A股市场分析系统 - 统一交互控制台")
    print("="*50)
    
    # Use your tokens
    tushare_token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"
    
    # Your EastMoney cookie
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
    
    # Initialize console
    console = ASConsole(tushare_token, eastmoney_cookie)
    
    # Run interactive console
    console.run()


if __name__ == "__main__":
    main()