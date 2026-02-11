#!/usr/bin/env python3
"""
Unified CLI Interface for A-Share Market Analysis System
Combines all functionality from multiple entry points into a single interface
"""
import sys
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.screeners.stock_screener import StockScreener
from quant_trade_a_share.strategies import StrategyManager
from quant_trade_a_share.signals.signal_notifier import SignalNotifier
from quant_trade_a_share.prediction.predictive_analyzer import PredictiveAnalyzer
# Note: RealTimeSignalGenerator was removed as part of unification
# Using signal generation from other modules
from quant_trade_a_share.backtest.backtester_tushare import BacktesterWithTushare
from quant_trade_a_share.data.data_fetcher import DataFetcher
from quant_trade_a_share.watchlist.watchlist_manager import WatchlistManager
import sys
import os
# Add the project root directory to the path to import from the root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multi_factor_strategy_template import MultiFactorStrategy
from quant_trade_a_share.integration.qlib_enhancement import enhance_cli_with_qlib
# Try to import DeepQlibIntegration with error handling
try:
    from quant_trade_a_share.integration.deep_qlib_integration import DeepQlibIntegration
    DEEP_QLIB_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"⚠️ 深度 Qlib 集成不可用: {e}")
    print("💡 解决方案: 运行 'brew install libomp' 或 'pip install lightgbm' 或 './install_qlib.sh'")
    DEEP_QLIB_AVAILABLE = False
    # Define a dummy class to avoid further errors
    class DeepQlibIntegration:
        def __init__(self, *args, **kwargs):
            print("❌ 深度 Qlib 集成不可用 (请按提示安装依赖)")

        def __getattr__(self, name):
            return lambda *args, **kwargs: print(f"❌ 功能 '{name}' 不可用 (深度 Qlib 集成未加载)")

# Try to import QlibIntegratedEnhancement with error handling
try:
    from quant_trade_a_share.integration.qlib_integrated_enhancement import QlibIntegratedEnhancement
    QLIB_INTEGRATED_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"⚠️ Qlib集成增强不可用: {e}")
    print("💡 解决方案: 检查因素库依赖是否正确安装")
    QLIB_INTEGRATED_AVAILABLE = False
    # Define a dummy class to avoid further errors
    class QlibIntegratedEnhancement:
        def __init__(self, *args, **kwargs):
            print("❌ Qlib集成增强不可用 (请按提示安装依赖)")

        def __getattr__(self, name):
            return lambda *args, **kwargs: print(f"❌ 功能 '{name}' 不可用 (Qlib集成增强未加载)")


class UnifiedCLIInterface:
    """
    Unified CLI interface combining all system functionality
    """
    def __init__(self, tushare_token, eastmoney_cookie):
        self.tushare_token = tushare_token
        self.eastmoney_cookie = eastmoney_cookie

        # Initialize all system components
        self.screener = StockScreener(tushare_token=tushare_token)
        self.strategy_manager = StrategyManager()
        self.signal_notifier = SignalNotifier()
        self.predictive_analyzer = PredictiveAnalyzer()
        # Note: RealTimeSignalGenerator was removed as part of unification
        # Using signal generation from other modules
        
        # Initialize backtester if token is provided
        self.backtester = None
        if tushare_token:
            try:
                self.backtester = BacktesterWithTushare(tushare_token)
            except Exception as e:
                print(f"⚠️  无法初始化Tushare回测模块: {e}")

        # Initialize data fetcher
        self.data_fetcher = DataFetcher()
        
        # Initialize multi-factor strategy
        self.multi_factor_strategy = MultiFactorStrategy()

        # Store session data
        self.session_data = {}
        self.current_stocks = []
        # Initialize watchlist manager
        self.watchlist_manager = WatchlistManager()

        print("✅ A股市场分析系统统一接口初始化完成")
        print("="*60)

    def show_help(self):
        """
        Display help information with available commands
        """
        print("""
🤖 A股市场分析系统统一接口 - 可用命令:
=======================================
📈 市场分析类:
  1.  screen_stocks    - 筛选潜在上涨股票 (市值>200亿)
  2.  analyze_stock    - 分析单个股票
  3.  predict_stocks   - 预测股票上涨概率

📊 策略信号类:
  4.  run_strategy     - 运行指定策略
  5.  gen_signals      - 生成买卖信号
  6.  show_signals     - 显示最新信号

🔍 数据查询类:
  7.  get_data         - 获取股票数据
  8.  calc_indicators  - 计算技术指标
  9.  show_top_stocks  - 显示热门股票

📈 预测分析类:
  10. predictive_analysis - 运行预测分析
  11. top_predictions   - 显示Top预测
  12. analyze_market    - 市场整体分析

🔬 回测功能类:
  13. run_backtest      - 运行策略回测
  14. compare_strategies - 比较不同策略

🔍 自选股管理类:
  15. batch_analyze_watchlist - 批量分析自选股
  16. manage_watchlist        - 管理自选股列表

📊 多因子分析类:
  17. multi_factor_analysis - 运行100+因子分析
  18. analyze_factors   - 分析因子表现
  19. factor_report     - 生成因子报告

⚙️  系统管理类:
  20. show_session     - 显示会话数据
  21. clear_session    - 清空会话数据
  22. help             - 显示帮助信息
  23. quit/exit        - 退出系统
  24. run_comprehensive_qlib_analysis - 综合性Qlib增强分析

💡 使用方法: 输入命令编号或命令名称
   例如: 输入 '1' 或 'screen_stocks' 开始股票筛选
=======================================
        """)

    def run_interactive(self):
        """
        Run the interactive console
        """
        print(f"🚀 启动A股市场分析系统统一接口 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
            'run_backtest': self.run_backtest,
            'compare_strategies': self.compare_strategies,
            'multi_factor_analysis': self.multi_factor_analysis,
            'analyze_factors': self.analyze_factors,
            'factor_report': self.factor_report,
            'batch_analyze_watchlist': self.batch_analyze_watchlist,
            'manage_watchlist': self.manage_watchlist,
            'show_session': self.show_session,
            'clear_session': self.clear_session,
            'run_comprehensive_qlib_analysis': self.run_comprehensive_qlib_analysis
        }

    def handle_numeric_command(self, cmd_num):
        """
        Handle command by number
        """
        cmd_map = {
            1: 'screen_stocks',
            2: 'analyze_stock',
            3: 'predict_stocks',
            4: 'run_strategy',
            5: 'gen_signals',
            6: 'show_signals',
            7: 'get_data',
            8: 'calc_indicators',
            9: 'show_top_stocks',
            10: 'predictive_analysis',
            11: 'top_predictions',
            12: 'analyze_market',
            13: 'run_backtest',
            14: 'compare_strategies',
            15: 'multi_factor_analysis',
            16: 'analyze_factors',
            17: 'factor_report',
            18: 'batch_analyze_watchlist',
            19: 'manage_watchlist',
            20: 'show_session',
            21: 'clear_session',
            22: 'help',
            23: 'quit',
            24: 'run_comprehensive_qlib_analysis'
        }

        if cmd_num in cmd_map:
            cmd_name = cmd_map[cmd_num]
            if cmd_name == 'quit':
                print("👋 感谢使用A股市场分析系统，再见！")
                exit(0)
            elif cmd_name == 'help':
                self.show_help()
            else:
                self.execute_command(cmd_name)
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
        Screen for potentially rising stocks with detailed analysis
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

                # 提供详细分析选项
                analyze_detail = input("\n是否对筛选出的股票进行详细分析? (y/n, 默认: n): ").strip().lower()
                if analyze_detail == 'y':
                    # 对筛选出的前几只股票进行详细分析
                    num_to_analyze = input(f"请输入要分析的股票数量 (1-{min(10, len(results))}, 默认: 3): ").strip()
                    try:
                        num_to_analyze = int(num_to_analyze) if num_to_analyze else 3
                        num_to_analyze = min(num_to_analyze, len(results), 10)  # 最多分析10只或实际结果数
                    except ValueError:
                        num_to_analyze = 3

                    print(f"\n🚀 开始对前 {num_to_analyze} 只筛选出的股票进行详细分析...")

                    for idx, (_, stock_row) in enumerate(results.head(num_to_analyze).iterrows()):
                        symbol = stock_row.get('symbol', stock_row.get('ts_code', stock_row.name if hasattr(stock_row, 'name') else 'Unknown'))

                        print(f"\n{'='*80}")
                        print(f"📊 第 {idx+1}/{num_to_analyze} 只股票详细分析: {symbol}")
                        print(f"{'='*80}")

                        # 获取股票数据
                        from datetime import datetime, timedelta
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

                        # 优先使用Ashare数据源
                        data = self.data_fetcher.fetch(symbol, start_date, end_date, source='ashare')
                        if data is None or data.empty:
                            print(f"❌ 无法获取 {symbol} 的数据，跳过此股票")
                            continue

                        # 确保技术指标已计算
                        required_indicators = ['rsi6', 'rsi12', 'rsi24', 'macd_dif', 'macd_dea', 'macd_bar',
                                     'kdj_k', 'kdj_d', 'kdj_j', 'wr1', 'wr2', 'ma5', 'ma10', 'ma20',
                                     'ma30', 'ma60', 'boll_upper', 'boll_mid', 'boll_lower', 'cci',
                                     'atr', 'bias6', 'bias12', 'bias24', 'trix', 'trma', 'vr', 'cr',
                                     'obv', 'mfi', 'ema12', 'ema26', 'ema50']

                        missing_indicators = [col for col in required_indicators if col not in data.columns]
                        if missing_indicators:
                            print(f"🔄 检测到缺失指标，正在计算技术指标...")
                            data = self.screener.eastmoney_fetcher.calculate_enhanced_technical_indicators(data)
                            print("✅ 技术指标计算完成")
                        else:
                            print("✅ 数据已包含技术指标")

                        # 获取股票名称
                        if self.screener.chinese_stocks is None:
                            self.screener.get_chinese_stocks_list()

                        stock_info = self.screener.chinese_stocks[self.screener.chinese_stocks['symbol'] == symbol] if self.screener.chinese_stocks is not None else pd.DataFrame()
                        stock_name = symbol  # Default to symbol if name not found
                        if not stock_info.empty and 'name' in stock_info.columns:
                            stock_name = stock_info['name'].iloc[0]

                        # 计算近期表现
                        if not data.empty and len(data) > 0:
                            recent_performance = ((data['close'].iloc[-1] - data['close'].iloc[0]) /
                                                 data['close'].iloc[0]) * 100
                            current_price = data['close'].iloc[-1]
                            # 计算20日和60日表现
                            perf_20d = ((data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]) * 100 if len(data) >= 20 else 0
                            perf_60d = ((data['close'].iloc[-1] - data['close'].iloc[-60]) / data['close'].iloc[-60]) * 100 if len(data) >= 60 else 0
                        else:
                            recent_performance = 0
                            perf_20d = 0
                            perf_60d = 0
                            current_price = 0
                            print(f"⚠️  {symbol} 数据不足，无法计算近期表现")

                        # 运行所有可用策略并收集信号
                        all_strategies = ['ma_crossover', 'rsi', 'macd', 'bollinger', 'mean_reversion', 'breakout']
                        strategy_results = {}

                        print(f"\n🏃 运行所有策略...")
                        for strategy_name in all_strategies:
                            strategy = self.strategy_manager.get_strategy(strategy_name)
                            if strategy is not None:
                                try:
                                    signals = strategy.generate_signals(data)
                                    latest_signal = signals.iloc[-1] if len(signals) > 0 else 0
                                    signal_count = len(signals[signals != 0]) if len(signals) > 0 else 0
                                    strategy_results[strategy_name] = {
                                        'signal': latest_signal,
                                        'signal_count': signal_count,
                                        'signals': signals
                                    }
                                    signal_text = "📈 买入" if latest_signal == 1 else "🔴 卖出" if latest_signal == -1 else "⏸️  持有"
                                    print(f"   {strategy_name}: {signal_text}")
                                except Exception as e:
                                    print(f"   ⚠️  {strategy_name} 策略执行失败: {e}")
                                    strategy_results[strategy_name] = {
                                        'signal': 0,
                                        'signal_count': 0,
                                        'signals': pd.Series(dtype=float)
                                    }
                            else:
                                print(f"   ⚠️  策略 {strategy_name} 不存在")
                                strategy_results[strategy_name] = {
                                    'signal': 0,
                                    'signal_count': 0,
                                    'signals': pd.Series(dtype=float)
                                }

                        # 获取增强技术指标（现在保证已存在）
                        rsi6 = data['rsi6'].iloc[-1] if 'rsi6' in data.columns and not pd.isna(data['rsi6'].iloc[-1]) else 0
                        rsi12 = data['rsi12'].iloc[-1] if 'rsi12' in data.columns and not pd.isna(data['rsi12'].iloc[-1]) else 0
                        rsi24 = data['rsi24'].iloc[-1] if 'rsi24' in data.columns and not pd.isna(data['rsi24'].iloc[-1]) else 0

                        macd = data['macd_dif'].iloc[-1] if 'macd_dif' in data.columns and not pd.isna(data['macd_dif'].iloc[-1]) else 0
                        macd_signal = data['macd_dea'].iloc[-1] if 'macd_dea' in data.columns and not pd.isna(data['macd_dea'].iloc[-1]) else 0
                        macd_histogram = data['macd_bar'].iloc[-1] if 'macd_bar' in data.columns and not pd.isna(data['macd_bar'].iloc[-1]) else 0

                        ma_5 = data['ma5'].iloc[-1] if 'ma5' in data.columns and not pd.isna(data['ma5'].iloc[-1]) else 0
                        ma_10 = data['ma10'].iloc[-1] if 'ma10' in data.columns and not pd.isna(data['ma10'].iloc[-1]) else 0
                        ma_20 = data['ma20'].iloc[-1] if 'ma20' in data.columns and not pd.isna(data['ma20'].iloc[-1]) else 0
                        ma_30 = data['ma30'].iloc[-1] if 'ma30' in data.columns and not pd.isna(data['ma30'].iloc[-1]) else 0
                        ma_60 = data['ma60'].iloc[-1] if 'ma60' in data.columns and not pd.isna(data['ma60'].iloc[-1]) else 0

                        bb_upper = data['boll_upper'].iloc[-1] if 'boll_upper' in data.columns and not pd.isna(data['boll_upper'].iloc[-1]) else 0
                        bb_lower = data['boll_lower'].iloc[-1] if 'boll_lower' in data.columns and not pd.isna(data['boll_lower'].iloc[-1]) else 0
                        bb_middle = data['boll_mid'].iloc[-1] if 'boll_mid' in data.columns and not pd.isna(data['boll_mid'].iloc[-1]) else 0

                        kdj_k = data['kdj_k'].iloc[-1] if 'kdj_k' in data.columns and not pd.isna(data['kdj_k'].iloc[-1]) else 0
                        kdj_d = data['kdj_d'].iloc[-1] if 'kdj_d' in data.columns and not pd.isna(data['kdj_d'].iloc[-1]) else 0
                        kdj_j = data['kdj_j'].iloc[-1] if 'kdj_j' in data.columns and not pd.isna(data['kdj_j'].iloc[-1]) else 0

                        wr1 = data['wr1'].iloc[-1] if 'wr1' in data.columns and not pd.isna(data['wr1'].iloc[-1]) else 0
                        wr2 = data['wr2'].iloc[-1] if 'wr2' in data.columns and not pd.isna(data['wr2'].iloc[-1]) else 0

                        cci = data['cci'].iloc[-1] if 'cci' in data.columns and not pd.isna(data['cci'].iloc[-1]) else 0

                        atr = data['atr'].iloc[-1] if 'atr' in data.columns and not pd.isna(data['atr'].iloc[-1]) else 0

                        volume_ratio = data['volume_ratio'].iloc[-1] if 'volume_ratio' in data.columns and not pd.isna(data['volume_ratio'].iloc[-1]) else 0
                        volatility = data['volatility'].iloc[-1] if 'volatility' in data.columns and not pd.isna(data['volatility'].iloc[-1]) else 0
                        momentum = data['momentum'].iloc[-1] if 'momentum' in data.columns and not pd.isna(data['momentum'].iloc[-1]) else 0
                        roc = data['roc'].iloc[-1] if 'roc' in data.columns and not pd.isna(data['roc'].iloc[-1]) else 0

                        # MyTT指标
                        bias6 = data['bias6'].iloc[-1] if 'bias6' in data.columns and not pd.isna(data['bias6'].iloc[-1]) else 0
                        bias12 = data['bias12'].iloc[-1] if 'bias12' in data.columns and not pd.isna(data['bias12'].iloc[-1]) else 0
                        bias24 = data['bias24'].iloc[-1] if 'bias24' in data.columns and not pd.isna(data['bias24'].iloc[-1]) else 0

                        dmi_pdi = data['dmi_pdi'].iloc[-1] if 'dmi_pdi' in data.columns and not pd.isna(data['dmi_pdi'].iloc[-1]) else 0
                        dmi_mdi = data['dmi_mdi'].iloc[-1] if 'dmi_mdi' in data.columns and not pd.isna(data['dmi_mdi'].iloc[-1]) else 0
                        dmi_adx = data['dmi_adx'].iloc[-1] if 'dmi_adx' in data.columns and not pd.isna(data['dmi_adx'].iloc[-1]) else 0

                        trix = data['trix'].iloc[-1] if 'trix' in data.columns and not pd.isna(data['trix'].iloc[-1]) else 0
                        trma = data['trma'].iloc[-1] if 'trma' in data.columns and not pd.isna(data['trma'].iloc[-1]) else 0

                        vr = data['vr'].iloc[-1] if 'vr' in data.columns and not pd.isna(data['vr'].iloc[-1]) else 0
                        cr = data['cr'].iloc[-1] if 'cr' in data.columns and not pd.isna(data['cr'].iloc[-1]) else 0

                        obv = data['obv'].iloc[-1] if 'obv' in data.columns and not pd.isna(data['obv'].iloc[-1]) else 0
                        mfi = data['mfi'].iloc[-1] if 'mfi' in data.columns and not pd.isna(data['mfi'].iloc[-1]) else 0

                        ema12 = data['ema12'].iloc[-1] if 'ema12' in data.columns and not pd.isna(data['ema12'].iloc[-1]) else 0
                        ema26 = data['ema26'].iloc[-1] if 'ema26' in data.columns and not pd.isna(data['ema26'].iloc[-1]) else 0
                        ema50 = data['ema50'].iloc[-1] if 'ema50' in data.columns and not pd.isna(data['ema50'].iloc[-1]) else 0

                        # 计算额外的分析指标
                        price_to_ma20 = (current_price / ma_20 - 1) * 100 if ma_20 != 0 else 0
                        price_position_bb = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                        volume_change = volume_ratio - 1 if volume_ratio != 0 else 0

                        # 计算趋势
                        trend = "上升" if current_price > ma_20 else "下降"

                        # 生成综合分析报告
                        print(f"\n" + "="*60)
                        print(f"🏆 {symbol} ({stock_name}) 详细分析报告")
                        print("="*60)

                        # 价格与表现部分
                        print(f"💰 价格与表现:")
                        print(f"   当前价格: ¥{current_price:.2f}")
                        print(f"   180日涨幅: {recent_performance:+.2f}%")
                        print(f"   60日涨幅: {perf_60d:+.2f}%")
                        print(f"   20日涨幅: {perf_20d:+.2f}%")
                        print(f"   当前趋势: {trend}")

                        # 技术指标部分
                        print(f"\n🔧 技术指标:")
                        print(f"   RSI (6/12/24): {rsi6:.2f}/{rsi12:.2f}/{rsi24:.2f} ({' oversold' if rsi24 < 30 else ' overbought' if rsi24 > 70 else ' neutral'})")
                        print(f"   MACD: {macd:.4f}, Signal: {macd_signal:.4f}, Histogram: {macd_histogram:.4f}")
                        print(f"   KDJ: K:{kdj_k:.2f}, D:{kdj_d:.2f}, J:{kdj_j:.2f}")
                        print(f"   威廉指标: WR1:{wr1:.2f}, WR2:{wr2:.2f}")
                        print(f"   移动均线: MA5:{ma_5:.2f}, MA10:{ma_10:.2f}, MA20:{ma_20:.2f}, MA30:{ma_30:.2f}, MA60:{ma_60:.2f}")
                        print(f"   指数均线: EMA12:{ema12:.2f}, EMA26:{ema26:.2f}, EMA50:{ema50:.2f}")
                        print(f"   布林带: 上轨{bb_upper:.2f}, 中轨{bb_middle:.2f}, 下轨{bb_lower:.2f}")
                        print(f"   价格在布林带位置: {price_position_bb:.2f} ({'高位' if price_position_bb > 0.8 else '中位' if 0.2 <= price_position_bb <= 0.8 else '低位'})")
                        print(f"   CCI: {cci:.2f}")
                        print(f"   DMI: PDI:{dmi_pdi:.2f}, MDI:{dmi_mdi:.2f}, ADX:{dmi_adx:.2f}")
                        print(f"   BIAS: 6日{bias6:.2f}%, 12日{bias12:.2f}%, 24日{bias24:.2f}%")
                        print(f"   TRIX: {trix:.4f}, TRMA: {trma:.4f}")
                        print(f"   VR: {vr:.2f}, CR: {cr:.2f}")
                        print(f"   OBV: {obv:.2f}, MFI: {mfi:.2f}")
                        print(f"   动量指标: {momentum:.4f}")
                        print(f"   ROC (10日): {roc:.2f}%")
                        print(f"   ATR (14日): {atr:.4f}")

                        # 成交量分析
                        print(f"\n📊 成交量分析:")
                        print(f"   量比: {volume_ratio:.2f} ({'放量' if volume_ratio > 1.5 else '缩量' if volume_ratio < 0.7 else '正常'})")
                        print(f"   成交量变化: {volume_change:+.2f}%")

                        # 风险分析
                        print(f"\n⚠️  风险分析:")
                        print(f"   波动率: {volatility:.4f} ({'高风险' if volatility > 0.04 else '中风险' if volatility > 0.02 else '低风险'})")
                        print(f"   价格距离MA20: {price_to_ma20:+.2f}% ({'远离' if abs(price_to_ma20) > 10 else '合理'})")

                        # 所有策略信号汇总
                        print(f"\n🎯 策略信号汇总:")
                        buy_signals = 0
                        sell_signals = 0
                        hold_signals = 0

                        for strategy_name, result in strategy_results.items():
                            signal = result['signal']
                            signal_count = result['signal_count']
                            signal_text = "📈 买入" if signal == 1 else "🔴 卖出" if signal == -1 else "⏸️  持有"
                            print(f"   {strategy_name.upper()}: {signal_text} (历史信号数: {signal_count})")

                            if signal == 1:
                                buy_signals += 1
                            elif signal == -1:
                                sell_signals += 1
                            else:
                                hold_signals += 1

                        # 共识信号
                        consensus_signal = ""
                        if buy_signals > sell_signals and buy_signals > hold_signals:
                            consensus_signal = "📈 多数策略建议买入"
                        elif sell_signals > buy_signals and sell_signals > hold_signals:
                            consensus_signal = "🔴 多数策略建议卖出"
                        else:
                            consensus_signal = "⏸️  多数策略建议持有/意见分歧"

                        print(f"\n📊 策略共识: {consensus_signal}")
                        print(f"   买入信号: {buy_signals}, 卖出信号: {sell_signals}, 持有信号: {hold_signals}")

                        # 投资建议部分
                        print(f"\n💡 投资建议:")
                        # 使用RSI24和MACD柱状图作为主要建议，因为我们使用了所有策略
                        recommendation = self._generate_investment_recommendation(
                            rsi24, macd_histogram, price_position_bb, volume_ratio,
                            volatility, current_price, ma_20, recent_performance,
                            perf_20d, strategy_results['ma_crossover']['signal'] if 'ma_crossover' in strategy_results else 0,
                            kdj_k, kdj_d, cci, rsi6, rsi12, rsi24
                        )
                        print(f"   {recommendation}")

                        # 未来上涨潜力评估
                        print(f"\n🚀 未来上涨潜力评估:")
                        potential_score = self._assess_future_potential(
                            rsi24, macd_histogram, price_position_bb, volume_ratio,
                            volatility, recent_performance, perf_20d, momentum, roc,
                            cci, kdj_k, kdj_d, bias6, dmi_adx
                        )
                        print(f"   潜力评分: {potential_score}/100")
                        if potential_score >= 80:
                            print(f"   🌟 极具上涨潜力")
                        elif potential_score >= 60:
                            print(f"   📈 有一定上涨潜力")
                        elif potential_score >= 40:
                            print(f"   ⚖️  潜力一般，观望")
                        else:
                            print(f"   📉 上涨潜力有限")

                        # 买卖时机分析
                        print(f"\n⏰ 买卖时机分析:")
                        timing_advice = self._analyze_buy_sell_timing(
                            rsi24, current_price, ma_5, ma_10, ma_20, bb_upper, bb_lower, bb_middle,
                            macd, macd_signal, volume_ratio, roc,
                            kdj_k, kdj_d, cci, atr, bias6
                        )
                        print(f"   {timing_advice}")

                        print("="*60)

                        # 将分析结果存储到会话
                        self.session_data[f'analysis_{symbol}'] = {
                            'symbol': symbol,
                            'name': stock_name,
                            'data': data,
                            'strategy_results': strategy_results,
                            'recent_performance': recent_performance,
                            'technical_indicators': {
                                'rsi': rsi24,
                                'rsi6': rsi6,
                                'rsi12': rsi12,
                                'rsi24': rsi24,
                                'macd': macd,
                                'macd_signal': macd_signal,
                                'macd_histogram': macd_histogram,
                                'kdj_k': kdj_k,
                                'kdj_d': kdj_d,
                                'kdj_j': kdj_j,
                                'wr1': wr1,
                                'wr2': wr2,
                                'ma_5': ma_5,
                                'ma_10': ma_10,
                                'ma_20': ma_20,
                                'ma_30': ma_30,
                                'ma_60': ma_60,
                                'ema12': ema12,
                                'ema26': ema26,
                                'ema50': ema50,
                                'bb_upper': bb_upper,
                                'bb_middle': bb_middle,
                                'bb_lower': bb_lower,
                                'cci': cci,
                                'atr': atr,
                                'volume_ratio': volume_ratio,
                                'volatility': volatility,
                                'momentum': momentum,
                                'roc': roc,
                                'bias6': bias6,
                                'bias12': bias12,
                                'bias24': bias24,
                                'dmi_pdi': dmi_pdi,
                                'dmi_mdi': dmi_mdi,
                                'dmi_adx': dmi_adx,
                                'trix': trix,
                                'trma': trma,
                                'vr': vr,
                                'cr': cr,
                                'obv': obv,
                                'mfi': mfi,
                                'price_to_ma20': price_to_ma20,
                                'price_position_bb': price_position_bb,
                                'volume_change': volume_change
                            },
                            'recommendation': recommendation,
                            'potential_score': potential_score,
                            'timing_advice': timing_advice,
                            'consensus_signal': consensus_signal
                        }
            else:
                print("⚠️  未找到符合条件的股票")
        except Exception as e:
            print(f"❌ 筛选过程出错: {e}")
            import traceback
            traceback.print_exc()

    def analyze_stock(self):
        """
        Analyze a specific stock with detailed fundamental and technical analysis
        Automatically runs all strategies and ensures technical indicators are calculated
        """
        symbol = input("请输入股票代码 (例: sh600519): ").strip()
        if not symbol:
            print("❌ 股票代码不能为空")
            return

        # Ask for data source (always auto to prioritize Ashare)
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: auto (优先使用Ashare)): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock', 'auto'] else 'auto'

        print(f"\n📊 分析股票 {symbol} - 自动运行所有策略...")
        print(f"📈 使用数据源: {source}")

        try:
            # Get stock data using DataFetcher with selected source
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

            data = self.data_fetcher.fetch(symbol, start_date, end_date, source=source)
            if data is None or data.empty:
                print(f"❌ 无法从{source}获取 {symbol} 的数据，尝试使用screener...")
                # Fallback to screener with specified source
                data = self.screener.fetch_stock_data(symbol, period='180', data_source=source)
                if data is None or data.empty:
                    print(f"⚠️  也无法从screener获取 {symbol} 数据，跳过")
                    return

            # Ensure technical indicators are calculated
            required_indicators = ['rsi6', 'rsi12', 'rsi24', 'macd_dif', 'macd_dea', 'macd_bar',
                                 'kdj_k', 'kdj_d', 'kdj_j', 'wr1', 'wr2', 'ma5', 'ma10', 'ma20',
                                 'ma30', 'ma60', 'boll_upper', 'boll_mid', 'boll_lower', 'cci',
                                 'atr', 'bias6', 'bias12', 'bias24', 'trix', 'trma', 'vr', 'cr',
                                 'obv', 'mfi', 'ema12', 'ema26', 'ema50']

            missing_indicators = [col for col in required_indicators if col not in data.columns]
            if missing_indicators:
                print(f"🔄 检测到缺失指标，正在计算技术指标...")
                # Use EastMoneyDataFetcher to calculate enhanced technical indicators
                data = self.screener.eastmoney_fetcher.calculate_enhanced_technical_indicators(data)
                print("✅ 技术指标计算完成")
            else:
                print("✅ 数据已包含技术指标")

            # Get stock name
            if self.screener.chinese_stocks is None:
                self.screener.get_chinese_stocks_list()

            stock_info = self.screener.chinese_stocks[self.screener.chinese_stocks['symbol'] == symbol] if self.screener.chinese_stocks is not None else pd.DataFrame()
            stock_name = symbol  # Default to symbol if name not found
            if not stock_info.empty and 'name' in stock_info.columns:
                stock_name = stock_info['name'].iloc[0]

            # Calculate recent performance if data is sufficient
            if not data.empty and len(data) > 0:
                recent_performance = ((data['close'].iloc[-1] - data['close'].iloc[0]) /
                                     data['close'].iloc[0]) * 100
                current_price = data['close'].iloc[-1]
                # Calculate 20-day and 60-day performances
                perf_20d = ((data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]) * 100 if len(data) >= 20 else 0
                perf_60d = ((data['close'].iloc[-1] - data['close'].iloc[-60]) / data['close'].iloc[-60]) * 100 if len(data) >= 60 else 0
            else:
                recent_performance = 0
                perf_20d = 0
                perf_60d = 0
                current_price = 0
                print(f"⚠️  {symbol} 数据不足，无法计算近期表现")

            # Run all available strategies and collect signals
            all_strategies = ['ma_crossover', 'rsi', 'macd', 'bollinger', 'mean_reversion', 'breakout']
            strategy_results = {}

            print(f"\n🏃 运行所有策略...")
            for strategy_name in all_strategies:
                strategy = self.strategy_manager.get_strategy(strategy_name)
                if strategy is not None:
                    try:
                        signals = strategy.generate_signals(data)
                        latest_signal = signals.iloc[-1] if len(signals) > 0 else 0
                        signal_count = len(signals[signals != 0]) if len(signals) > 0 else 0
                        strategy_results[strategy_name] = {
                            'signal': latest_signal,
                            'signal_count': signal_count,
                            'signals': signals
                        }
                        signal_text = "📈 买入" if latest_signal == 1 else "🔴 卖出" if latest_signal == -1 else "⏸️  持有"
                        print(f"   {strategy_name}: {signal_text}")
                    except Exception as e:
                        print(f"   ⚠️  {strategy_name} 策略执行失败: {e}")
                        strategy_results[strategy_name] = {
                            'signal': 0,
                            'signal_count': 0,
                            'signals': pd.Series(dtype=float)
                        }
                else:
                    print(f"   ⚠️  策略 {strategy_name} 不存在")
                    strategy_results[strategy_name] = {
                        'signal': 0,
                        'signal_count': 0,
                        'signals': pd.Series(dtype=float)
                    }

            # Get enhanced technical indicators (now guaranteed to exist)
            rsi6 = data['rsi6'].iloc[-1] if 'rsi6' in data.columns and not pd.isna(data['rsi6'].iloc[-1]) else 0
            rsi12 = data['rsi12'].iloc[-1] if 'rsi12' in data.columns and not pd.isna(data['rsi12'].iloc[-1]) else 0
            rsi24 = data['rsi24'].iloc[-1] if 'rsi24' in data.columns and not pd.isna(data['rsi24'].iloc[-1]) else 0

            macd = data['macd_dif'].iloc[-1] if 'macd_dif' in data.columns and not pd.isna(data['macd_dif'].iloc[-1]) else 0
            macd_signal = data['macd_dea'].iloc[-1] if 'macd_dea' in data.columns and not pd.isna(data['macd_dea'].iloc[-1]) else 0
            macd_histogram = data['macd_bar'].iloc[-1] if 'macd_bar' in data.columns and not pd.isna(data['macd_bar'].iloc[-1]) else 0

            ma_5 = data['ma5'].iloc[-1] if 'ma5' in data.columns and not pd.isna(data['ma5'].iloc[-1]) else 0
            ma_10 = data['ma10'].iloc[-1] if 'ma10' in data.columns and not pd.isna(data['ma10'].iloc[-1]) else 0
            ma_20 = data['ma20'].iloc[-1] if 'ma20' in data.columns and not pd.isna(data['ma20'].iloc[-1]) else 0
            ma_30 = data['ma30'].iloc[-1] if 'ma30' in data.columns and not pd.isna(data['ma30'].iloc[-1]) else 0
            ma_60 = data['ma60'].iloc[-1] if 'ma60' in data.columns and not pd.isna(data['ma60'].iloc[-1]) else 0

            bb_upper = data['boll_upper'].iloc[-1] if 'boll_upper' in data.columns and not pd.isna(data['boll_upper'].iloc[-1]) else 0
            bb_lower = data['boll_lower'].iloc[-1] if 'boll_lower' in data.columns and not pd.isna(data['boll_lower'].iloc[-1]) else 0
            bb_middle = data['boll_mid'].iloc[-1] if 'boll_mid' in data.columns and not pd.isna(data['boll_mid'].iloc[-1]) else 0

            kdj_k = data['kdj_k'].iloc[-1] if 'kdj_k' in data.columns and not pd.isna(data['kdj_k'].iloc[-1]) else 0
            kdj_d = data['kdj_d'].iloc[-1] if 'kdj_d' in data.columns and not pd.isna(data['kdj_d'].iloc[-1]) else 0
            kdj_j = data['kdj_j'].iloc[-1] if 'kdj_j' in data.columns and not pd.isna(data['kdj_j'].iloc[-1]) else 0

            wr1 = data['wr1'].iloc[-1] if 'wr1' in data.columns and not pd.isna(data['wr1'].iloc[-1]) else 0
            wr2 = data['wr2'].iloc[-1] if 'wr2' in data.columns and not pd.isna(data['wr2'].iloc[-1]) else 0

            cci = data['cci'].iloc[-1] if 'cci' in data.columns and not pd.isna(data['cci'].iloc[-1]) else 0

            atr = data['atr'].iloc[-1] if 'atr' in data.columns and not pd.isna(data['atr'].iloc[-1]) else 0

            volume_ratio = data['volume_ratio'].iloc[-1] if 'volume_ratio' in data.columns and not pd.isna(data['volume_ratio'].iloc[-1]) else 0
            volatility = data['volatility'].iloc[-1] if 'volatility' in data.columns and not pd.isna(data['volatility'].iloc[-1]) else 0
            momentum = data['momentum'].iloc[-1] if 'momentum' in data.columns and not pd.isna(data['momentum'].iloc[-1]) else 0
            roc = data['roc'].iloc[-1] if 'roc' in data.columns and not pd.isna(data['roc'].iloc[-1]) else 0

            # MyTT indicators
            bias6 = data['bias6'].iloc[-1] if 'bias6' in data.columns and not pd.isna(data['bias6'].iloc[-1]) else 0
            bias12 = data['bias12'].iloc[-1] if 'bias12' in data.columns and not pd.isna(data['bias12'].iloc[-1]) else 0
            bias24 = data['bias24'].iloc[-1] if 'bias24' in data.columns and not pd.isna(data['bias24'].iloc[-1]) else 0

            dmi_pdi = data['dmi_pdi'].iloc[-1] if 'dmi_pdi' in data.columns and not pd.isna(data['dmi_pdi'].iloc[-1]) else 0
            dmi_mdi = data['dmi_mdi'].iloc[-1] if 'dmi_mdi' in data.columns and not pd.isna(data['dmi_mdi'].iloc[-1]) else 0
            dmi_adx = data['dmi_adx'].iloc[-1] if 'dmi_adx' in data.columns and not pd.isna(data['dmi_adx'].iloc[-1]) else 0

            trix = data['trix'].iloc[-1] if 'trix' in data.columns and not pd.isna(data['trix'].iloc[-1]) else 0
            trma = data['trma'].iloc[-1] if 'trma' in data.columns and not pd.isna(data['trma'].iloc[-1]) else 0

            vr = data['vr'].iloc[-1] if 'vr' in data.columns and not pd.isna(data['vr'].iloc[-1]) else 0
            cr = data['cr'].iloc[-1] if 'cr' in data.columns and not pd.isna(data['cr'].iloc[-1]) else 0

            obv = data['obv'].iloc[-1] if 'obv' in data.columns and not pd.isna(data['obv'].iloc[-1]) else 0
            mfi = data['mfi'].iloc[-1] if 'mfi' in data.columns and not pd.isna(data['mfi'].iloc[-1]) else 0

            ema12 = data['ema12'].iloc[-1] if 'ema12' in data.columns and not pd.isna(data['ema12'].iloc[-1]) else 0
            ema26 = data['ema26'].iloc[-1] if 'ema26' in data.columns and not pd.isna(data['ema26'].iloc[-1]) else 0
            ema50 = data['ema50'].iloc[-1] if 'ema50' in data.columns and not pd.isna(data['ema50'].iloc[-1]) else 0

            # Calculate additional analysis metrics
            price_to_ma20 = (current_price / ma_20 - 1) * 100 if ma_20 != 0 else 0
            price_position_bb = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            volume_change = volume_ratio - 1 if volume_ratio != 0 else 0

            # Calculate trend
            trend = "上升" if current_price > ma_20 else "下降"

            # Generate comprehensive analysis report
            print(f"\n" + "="*60)
            print(f"🏆 {symbol} ({stock_name}) 详细分析报告")
            print("="*60)

            # Price and Performance Section
            print(f"💰 价格与表现:")
            print(f"   当前价格: ¥{current_price:.2f}")
            print(f"   180日涨幅: {recent_performance:+.2f}%")
            print(f"   60日涨幅: {perf_60d:+.2f}%")
            print(f"   20日涨幅: {perf_20d:+.2f}%")
            print(f"   当前趋势: {trend}")

            # Technical Indicators Section
            print(f"\n🔧 技术指标:")
            print(f"   RSI (6/12/24): {rsi6:.2f}/{rsi12:.2f}/{rsi24:.2f} ({' oversold' if rsi24 < 30 else ' overbought' if rsi24 > 70 else ' neutral'})")
            print(f"   MACD: {macd:.4f}, Signal: {macd_signal:.4f}, Histogram: {macd_histogram:.4f}")
            print(f"   KDJ: K:{kdj_k:.2f}, D:{kdj_d:.2f}, J:{kdj_j:.2f}")
            print(f"   威廉指标: WR1:{wr1:.2f}, WR2:{wr2:.2f}")
            print(f"   移动均线: MA5:{ma_5:.2f}, MA10:{ma_10:.2f}, MA20:{ma_20:.2f}, MA30:{ma_30:.2f}, MA60:{ma_60:.2f}")
            print(f"   指数均线: EMA12:{ema12:.2f}, EMA26:{ema26:.2f}, EMA50:{ema50:.2f}")
            print(f"   布林带: 上轨{bb_upper:.2f}, 中轨{bb_middle:.2f}, 下轨{bb_lower:.2f}")
            print(f"   价格在布林带位置: {price_position_bb:.2f} ({'高位' if price_position_bb > 0.8 else '中位' if 0.2 <= price_position_bb <= 0.8 else '低位'})")
            print(f"   CCI: {cci:.2f}")
            print(f"   DMI: PDI:{dmi_pdi:.2f}, MDI:{dmi_mdi:.2f}, ADX:{dmi_adx:.2f}")
            print(f"   BIAS: 6日{bias6:.2f}%, 12日{bias12:.2f}%, 24日{bias24:.2f}%")
            print(f"   TRIX: {trix:.4f}, TRMA: {trma:.4f}")
            print(f"   VR: {vr:.2f}, CR: {cr:.2f}")
            print(f"   OBV: {obv:.2f}, MFI: {mfi:.2f}")
            print(f"   动量指标: {momentum:.4f}")
            print(f"   ROC (10日): {roc:.2f}%")
            print(f"   ATR (14日): {atr:.4f}")

            # Volume Analysis
            print(f"\n📊 成交量分析:")
            print(f"   量比: {volume_ratio:.2f} ({'放量' if volume_ratio > 1.5 else '缩量' if volume_ratio < 0.7 else '正常'})")
            print(f"   成交量变化: {volume_change:+.2f}%")

            # Risk Analysis
            print(f"\n⚠️  风险分析:")
            print(f"   波动率: {volatility:.4f} ({'高风险' if volatility > 0.04 else '中风险' if volatility > 0.02 else '低风险'})")
            print(f"   价格距离MA20: {price_to_ma20:+.2f}% ({'远离' if abs(price_to_ma20) > 10 else '合理'})")

            # All Strategy Signals Section
            print(f"\n🎯 策略信号汇总:")
            buy_signals = 0
            sell_signals = 0
            hold_signals = 0

            for strategy_name, result in strategy_results.items():
                signal = result['signal']
                signal_count = result['signal_count']
                signal_text = "📈 买入" if signal == 1 else "🔴 卖出" if signal == -1 else "⏸️  持有"
                print(f"   {strategy_name.upper()}: {signal_text} (历史信号数: {signal_count})")

                if signal == 1:
                    buy_signals += 1
                elif signal == -1:
                    sell_signals += 1
                else:
                    hold_signals += 1

            # Consensus signal
            consensus_signal = ""
            if buy_signals > sell_signals and buy_signals > hold_signals:
                consensus_signal = "📈 多数策略建议买入"
            elif sell_signals > buy_signals and sell_signals > hold_signals:
                consensus_signal = "🔴 多数策略建议卖出"
            else:
                consensus_signal = "⏸️  多数策略建议持有/意见分歧"

            print(f"\n📊 策略共识: {consensus_signal}")
            print(f"   买入信号: {buy_signals}, 卖出信号: {sell_signals}, 持有信号: {hold_signals}")

            # Investment Recommendation Section
            print(f"\n💡 投资建议:")
            # Use RSI24 and MACD histogram for the main recommendation since we're using all strategies
            recommendation = self._generate_investment_recommendation(
                rsi24, macd_histogram, price_position_bb, volume_ratio,
                volatility, current_price, ma_20, recent_performance,
                perf_20d, strategy_results['ma_crossover']['signal'] if 'ma_crossover' in strategy_results else 0,
                kdj_k, kdj_d, cci, rsi6, rsi12, rsi24
            )
            print(f"   {recommendation}")

            # Future Potential Assessment
            print(f"\n🚀 未来上涨潜力评估:")
            potential_score = self._assess_future_potential(
                rsi24, macd_histogram, price_position_bb, volume_ratio,
                volatility, recent_performance, perf_20d, momentum, roc,
                cci, kdj_k, kdj_d, bias6, dmi_adx
            )
            print(f"   潜力评分: {potential_score}/100")
            if potential_score >= 80:
                print(f"   🌟 极具上涨潜力")
            elif potential_score >= 60:
                print(f"   📈 有一定上涨潜力")
            elif potential_score >= 40:
                print(f"   ⚖️  潜力一般，观望")
            else:
                print(f"   📉 上涨潜力有限")

            # Buy/Sell Timing
            print(f"\n⏰ 买卖时机分析:")
            timing_advice = self._analyze_buy_sell_timing(
                rsi24, current_price, ma_5, ma_10, ma_20, bb_upper, bb_lower, bb_middle,
                macd, macd_signal, volume_ratio, roc,
                kdj_k, kdj_d, cci, atr, bias6
            )
            print(f"   {timing_advice}")

            print("="*60)

            # Store in session
            self.session_data[f'analysis_{symbol}'] = {
                'symbol': symbol,
                'name': stock_name,
                'data': data,
                'strategy_results': strategy_results,
                'recent_performance': recent_performance,
                'technical_indicators': {
                    'rsi': rsi24,
                    'rsi6': rsi6,
                    'rsi12': rsi12,
                    'rsi24': rsi24,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'macd_histogram': macd_histogram,
                    'kdj_k': kdj_k,
                    'kdj_d': kdj_d,
                    'kdj_j': kdj_j,
                    'wr1': wr1,
                    'wr2': wr2,
                    'ma_5': ma_5,
                    'ma_10': ma_10,
                    'ma_20': ma_20,
                    'ma_30': ma_30,
                    'ma_60': ma_60,
                    'ema12': ema12,
                    'ema26': ema26,
                    'ema50': ema50,
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'cci': cci,
                    'atr': atr,
                    'volume_ratio': volume_ratio,
                    'volatility': volatility,
                    'momentum': momentum,
                    'roc': roc,
                    'bias6': bias6,
                    'bias12': bias12,
                    'bias24': bias24,
                    'dmi_pdi': dmi_pdi,
                    'dmi_mdi': dmi_mdi,
                    'dmi_adx': dmi_adx,
                    'trix': trix,
                    'trma': trma,
                    'vr': vr,
                    'cr': cr,
                    'obv': obv,
                    'mfi': mfi,
                    'price_to_ma20': price_to_ma20,
                    'price_position_bb': price_position_bb,
                    'volume_change': volume_change
                },
                'recommendation': recommendation,
                'potential_score': potential_score,
                'timing_advice': timing_advice,
                'consensus_signal': consensus_signal
            }

        except Exception as e:
            print(f"❌ 分析过程出错: {e}")
            import traceback
            traceback.print_exc()

    def _generate_investment_recommendation(self, rsi, macd_hist, price_pos_bb, vol_ratio,
                                          volatility, current_price, ma_20, perf_long,
                                          perf_short, signal, kdj_k, kdj_d, cci, rsi6, rsi12, rsi24):
        """
        Generate investment recommendation based on technical indicators
        """
        reasons = []

        # RSI Analysis - using the 24-period RSI as primary indicator
        if rsi24 < 30:
            reasons.append("RSI24超卖，可能触底反弹")
        elif rsi24 > 70:
            reasons.append("RSI24超买，短期回调风险")
        else:
            reasons.append("RSI24处于合理区间")

        # MACD Analysis
        if macd_hist > 0:
            reasons.append("MACD柱状图>0，看涨动能")
        else:
            reasons.append("MACD柱状图<0，看跌动能")

        # KDJ Analysis
        if kdj_k < 20 and kdj_d < 20:
            reasons.append("KDJ低位金叉，可能见底")
        elif kdj_k > 80 and kdj_d > 80:
            reasons.append("KDJ高位死叉，可能见顶")
        elif kdj_k > kdj_d:
            reasons.append("KDJ金叉向上，看涨")
        else:
            reasons.append("KDJ死叉向下，看跌")

        # CCI Analysis
        if cci < -100:
            reasons.append("CCI超卖，反转向上的可能性大")
        elif cci > 100:
            reasons.append("CCI超买，回调可能性大")
        else:
            reasons.append("CCI处于正常范围")

        # Price Position in Bollinger Band
        if price_pos_bb < 0.2:
            reasons.append("价格在布林带下轨附近，估值偏低")
        elif price_pos_bb > 0.8:
            reasons.append("价格在布林带上轨附近，估值偏高")
        else:
            reasons.append("价格在布林带中位区域")

        # Volume Analysis
        if vol_ratio > 1.5:
            reasons.append("成交量放大，资金关注")
        elif vol_ratio < 0.7:
            reasons.append("成交量萎缩，缺乏关注")
        else:
            reasons.append("成交量正常")

        # Moving Average Trend
        if current_price > ma_20:
            reasons.append("价格站上20日线，中期趋势向好")
        else:
            reasons.append("价格跌破20日线，中期趋势向下")

        # Performance Analysis
        if perf_short > 0:
            reasons.append("短期表现强劲")
        else:
            reasons.append("短期表现疲弱")

        # Signal Analysis
        if signal == 1:
            reasons.append("策略给出买入信号")
        elif signal == -1:
            reasons.append("策略给出卖出信号")
        else:
            reasons.append("策略建议持有")

        # Generate overall recommendation
        strong_positive = sum(['强势' in r or '看涨' in r or '向上' in r or '买入' in r or '反转向上的可能性大' in r or '估值偏低' in r for r in reasons])
        strong_negative = sum(['弱势' in r or '看跌' in r or '向下' in r or '卖出' in r or '回调可能性大' in r or '估值偏高' in r or '回调风险' in r for r in reasons])

        if strong_positive > strong_negative + 1:
            return f"建议买入: {'; '.join(reasons)}"
        elif strong_negative > strong_positive + 1:
            return f"建议卖出: {'; '.join(reasons)}"
        else:
            return f"建议观望: {'; '.join(reasons)}"

    def _assess_future_potential(self, rsi, macd_hist, price_pos_bb, vol_ratio,
                               volatility, perf_long, perf_short, momentum, roc,
                               cci, kdj_k, kdj_d, bias6, dmi_adx):
        """
        Assess future potential of the stock
        """
        score = 50  # Base score

        # RSI contribution (best between 30-70, especially 40-60)
        if 40 <= rsi <= 60:
            score += 10
        elif 30 <= rsi <= 70:
            score += 5
        elif rsi < 30:  # Oversold, potential rebound
            score += 8
        else:  # Overbought, less favorable
            score -= 5

        # MACD histogram positive
        if macd_hist > 0:
            score += 8
        elif macd_hist < 0:
            score -= 5

        # Price position in Bollinger band (favorable if not too high)
        if 0.2 <= price_pos_bb <= 0.8:
            score += 8
        elif 0.1 <= price_pos_bb <= 0.9:
            score += 4
        else:
            score -= 3

        # Volume ratio (higher is generally better)
        if vol_ratio > 1.5:
            score += 5
        elif vol_ratio > 1.2:
            score += 3
        elif vol_ratio < 0.5:
            score -= 5

        # Performance (positive performance is good)
        if perf_short > 0:
            score += 5
        elif perf_short < -5:  # Strong negative performance reduces score
            score -= 8

        # Momentum (positive momentum is good)
        if momentum > 0:
            score += 3
        elif momentum < -0.1:  # Strong negative momentum reduces score
            score -= 5

        # ROC (positive ROC is good)
        if roc > 0:
            score += 3
        elif roc < -2:  # Strong negative ROC reduces score
            score -= 5

        # CCI contribution (good when between -100 and 100, but also consider extremes)
        if -100 <= cci <= 100:
            score += 5
        elif cci < -100:  # Oversold, potential rebound
            score += 6
        else:  # Overbought
            score += 2

        # KDJ contribution (good when K>D and in middle range)
        if kdj_k > kdj_d and 20 <= kdj_k <= 80:
            score += 6
        elif kdj_k < kdj_d and 20 <= kdj_d <= 80:
            score -= 3

        # Bias contribution (not too far from moving average is good)
        if abs(bias6) < 5:  # Reasonable bias
            score += 5
        elif abs(bias6) > 8:  # Too far from moving average, risky
            score -= 5

        # DMI ADX contribution (higher ADX indicates stronger trend)
        if dmi_adx > 25:
            score += 5  # Strong trend
        elif dmi_adx < 20:
            score -= 3  # Weak trend

        # Limit score between 0 and 100
        score = max(0, min(100, score))

        return score

    def _analyze_buy_sell_timing(self, rsi, current_price, ma_5, ma_10, ma_20,
                               bb_upper, bb_lower, bb_middle, macd, macd_signal, vol_ratio, roc,
                               kdj_k, kdj_d, cci, atr, bias6):
        """
        Analyze current buy/sell timing
        """
        advice_parts = []

        # RSI Timing
        if 30 < rsi < 70:
            advice_parts.append("RSI处于中性区域，适合观察")
        elif rsi < 30:
            advice_parts.append("RSI超卖，可能是较好买点")
        elif rsi > 70:
            advice_parts.append("RSI超买，考虑获利了结")

        # Moving Average Alignment
        if current_price > ma_5 > ma_10 > ma_20:
            advice_parts.append("多头排列，趋势向好")
        elif current_price < ma_5 < ma_10 < ma_20:
            advice_parts.append("空头排列，趋势向淡")
        else:
            advice_parts.append("均线纠缠，方向不明")

        # MACD Timing
        if macd > macd_signal:
            advice_parts.append("MACD金叉向上，看涨信号")
        elif macd < macd_signal:
            advice_parts.append("MACD死叉向下，看跌信号")
        else:
            advice_parts.append("MACD与信号线粘合")

        # Price and Bollinger Bands
        if bb_lower < current_price < bb_middle:
            advice_parts.append("价格在布林带下轨至中轨间，相对安全")
        elif bb_middle < current_price < bb_upper:
            advice_parts.append("价格在布林带中轨至上轨间，注意压力")
        else:
            advice_parts.append("价格偏离布林带，注意回调")

        # KDJ Timing
        if kdj_k > kdj_d and kdj_k < 80:
            advice_parts.append("KDJ金叉向上，看涨信号")
        elif kdj_k < kdj_d and kdj_k > 20:
            advice_parts.append("KDJ死叉向下，看跌信号")
        elif kdj_k > 80 and kdj_d > 80:
            advice_parts.append("KDJ高位钝化，注意回调")
        elif kdj_k < 20 and kdj_d < 20:
            advice_parts.append("KDJ超卖区，关注反弹机会")

        # CCI Timing
        if cci > 100:
            advice_parts.append("CCI超买，短期调整风险")
        elif cci < -100:
            advice_parts.append("CCI超卖，反弹预期")
        elif -100 < cci < 100:
            advice_parts.append("CCI在正常区间")

        # BIAS Timing
        if abs(bias6) > 8:
            advice_parts.append("BIAS偏离过大，注意回归")
        elif abs(bias6) < 3:
            advice_parts.append("BIAS位置合理")

        # ATR and Volatility
        if atr > 0 and roc > 0:
            advice_parts.append("波动率较高，关注趋势持续性")
        elif atr > 0 and roc < 0:
            advice_parts.append("高波动负收益，风险较高")

        # Volume and ROC
        if vol_ratio > 1.2 and roc > 0:
            advice_parts.append("量价配合良好，趋势持续可能性高")
        elif vol_ratio < 0.8 and roc < 0:
            advice_parts.append("量价背离，趋势可持续性存疑")
        else:
            advice_parts.append("量价关系基本正常")

        # Combine advice
        return "综合来看: " + "; ".join(advice_parts)

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

        # Ask for data source
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: auto (优先使用Ashare)): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock', 'auto'] else 'auto'

        symbols = [s.strip() for s in symbols_input.split(',')]

        print(f"\n🏃 运行 {strategy_name} 策略...")
        print(f"📈 使用数据源: {source}")

        try:
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if strategy is None:
                print(f"❌ 策略 {strategy_name} 不存在")
                return

            all_results = []
            for symbol in symbols:
                print(f"📈 分析 {symbol}...")

                # Get stock data using DataFetcher with selected source
                # Use the most recent 180 days of data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

                data = self.data_fetcher.fetch(symbol, start_date, end_date, source=source)
                if data is None or data.empty:
                    print(f"⚠️  无法从{source}获取 {symbol} 数据，尝试使用screener...")
                    # Fallback to screener if DataFetcher fails
                    data = self.screener.fetch_stock_data(symbol, period='180', data_source=source)
                    if data is None or data.empty:
                        print(f"⚠️  也无法从screener获取 {symbol} 数据，跳过")
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

        # Ask for data source
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: auto (优先使用Ashare)): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock', 'auto'] else 'auto'

        symbols = [s.strip() for s in symbols_input.split(',')]

        print(f"\n🔔 为 {len(symbols)} 只股票生成买卖信号...")
        print(f"📈 使用数据源: {source}")

        try:
            all_signals = []

            for symbol in symbols:
                print(f"📈 分析 {symbol}...")

                # Get stock data using DataFetcher with selected source
                # Use the most recent 180 days of data
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

                data = self.data_fetcher.fetch(symbol, start_date, end_date, source=source)
                if data is None or data.empty:
                    print(f"⚠️  无法从{source}获取 {symbol} 数据，尝试使用screener...")
                    # Fallback to screener if DataFetcher fails
                    data = self.screener.fetch_stock_data(symbol, period='180', data_source=source)
                    if data is None or data.empty:
                        print(f"⚠️  也无法从screener获取 {symbol} 数据，跳过")
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
                                        reason=f"策略分析 {strategy_name}策略产生买入信号",
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
                                        reason=f"策略分析 {strategy_name}策略产生卖出信号",
                                        priority=2
                                    )
                                    print(f"🔴 {symbol} - {strategy_name}: 卖出信号 (¥{latest_price})")
                                else:
                                    print(f"⏸️ {symbol} - {strategy_name}: 持有信号")

                    except Exception as e:
                        print(f"⚠️  策略 {strategy_name} 在 {symbol} 上执行失败: {e}")

            # Get the latest signals to display
            latest_signals = self.signal_notifier.get_recent_signals(10)

            if len(latest_signals) > 0:
                print(f"\n✅ 信号生成完成，共生成 {len(latest_signals)} 个信号:")
                for signal in latest_signals:
                    signal_type = "🟢 买入" if signal['signal_type'] == 'BUY' else "🔴 卖出" if signal['signal_type'] == 'SELL' else "⏸️  持有"
                    print(f"   {signal_type} - {signal['symbol']} ({signal['strategy']}): {signal['reason'][:30]}...")
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

        # Ask for data source
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: eastmoney): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock'] else 'eastmoney'

        print(f"\n📊 从 {source} 获取 {symbol} 最近 {days} 天数据...")
        try:
            # Calculate date range
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            data = self.data_fetcher.fetch(symbol, start_date, end_date, source=source)

            if data is not None and not data.empty:
                print(f"\n✅ 从 {source} 获取到 {len(data)} 条数据:")
                print(data[['open', 'close', 'high', 'low', 'volume']].tail(5).to_string())

                # Store in session
                self.session_data[f'data_{symbol}_{source}'] = data
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

        # Ask for data source
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: auto (优先使用Ashare)): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock', 'auto'] else 'auto'

        print(f"\n🧮 计算 {symbol} 技术指标...")
        print(f"📈 使用数据源: {source}")

        try:
            # Get data using the screener with specified source
            data = self.screener.fetch_stock_data(symbol, period='180', data_source=source)

            if data is None or data.empty:
                print(f"❌ 无法从{source}获取 {symbol} 数据")
                return

            # Calculate indicators using the EastMoneyDataFetcher which has enhanced indicators
            data = self.screener.eastmoney_fetcher.calculate_enhanced_technical_indicators(data)

            print(f"\n✅ 技术指标计算完成:")
            if 'rsi' in data.columns:
                print(f"   RSI: {data['rsi'].iloc[-1]:.2f}")
            if 'macd' in data.columns:
                print(f"   MACD: {data['macd'].iloc[-1]:.2f}")
            if 'ma_5' in data.columns:
                print(f"   MA5: {data['ma_5'].iloc[-1]:.2f}")
            if 'ma_20' in data.columns:
                print(f"   MA20: {data['ma_20'].iloc[-1]:.2f}")
            if 'bb_upper' in data.columns:
                print(f"   布林线上轨: {data['bb_upper'].iloc[-1]:.2f}")
                print(f"   布林线下轨: {data['bb_lower'].iloc[-1]:.2f}")

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
            # Use the available method from the screener
            all_stocks = self.screener.get_chinese_stocks_list()
            if all_stocks is not None and not all_stocks.empty:
                # Get first 10 stocks as top stocks
                top_stocks = all_stocks.head(10)
                
                print(f"\n✅ 获取到 {len(top_stocks)} 只股票:")
                for i, (idx, stock) in enumerate(top_stocks.iterrows(), 1):
                    symbol = stock.get('symbol', 'N/A')
                    name = stock.get('name', 'N/A') if 'name' in stock else 'N/A'
                    print(f"  {i}. {name} ({symbol})")
                
                # Store in session
                self.session_data['top_stocks'] = top_stocks
            else:
                print("⚠️  无法获取股票列表")
        except Exception as e:
            print(f"❌ 获取热门股票出错: {e}")
            import traceback
            traceback.print_exc()

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
                avg_volume = 0
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

    def multi_factor_analysis(self):
        """
        Run 100+ factor multi-factor analysis
        """
        print("\n📊 100+因子多因子分析...")

        symbols_input = input("请输入股票代码 (用逗号分隔，如: sh600023,sz000001,sh600519): ").strip()
        if not symbols_input:
            symbols = ['sh600023', 'sh600519', 'sz000001']  # Default stocks
            print("使用默认股票列表")
        else:
            symbols = [s.strip() for s in symbols_input.split(',')]

        start_date = input("请输入开始日期 (YYYY-MM-DD, 默认: 2025-06-01): ").strip() or '2025-06-01'
        end_date = input("请输入结束日期 (YYYY-MM-DD, 默认: 2025-12-31): ").strip() or '2025-12-31'

        print(f"\n🚀 对 {len(symbols)} 只股票进行100+因子分析...")
        print(f"📅 期间: {start_date} 至 {end_date}")

        try:
            # Update the strategy universe
            self.multi_factor_strategy.universe = symbols

            # Run backtest which calculates 100+ factors
            results = self.multi_factor_strategy.run_backtest(start_date=start_date, end_date=end_date)

            if results:
                print(f"\n✅ 多因子分析完成:")
                print(f"📈 共分析 {len(results)} 只股票")

                # Display results
                for stock, result in results.items():
                    print(f"\n   📊 {stock}:")
                    print(f"      策略收益: {result['total_strategy_return']*100:.2f}%")
                    print(f"      基准收益: {result['total_benchmark_return']*100:.2f}%")
                    print(f"      超额收益: {(result['total_strategy_return']-result['total_benchmark_return'])*100:.2f}%")
                    print(f"      信息比率: {result['info_ratio']:.4f}")
                    print(f"      最大回撤: {result['max_drawdown']*100:.2f}%")

                # Store results in session
                self.session_data['multi_factor_results'] = results

                # Show summary
                avg_strategy_ret = np.mean([r['total_strategy_return'] for r in results.values()])
                avg_benchmark_ret = np.mean([r['total_benchmark_return'] for r in results.values()])
                avg_ir = np.mean([r['info_ratio'] for r in results.values()])

                print(f"\n🏆 整体表现:")
                print(f"   平均策略收益: {avg_strategy_ret*100:.2f}%")
                print(f"   平均基准收益: {avg_benchmark_ret*100:.2f}%")
                print(f"   平均超额收益: {(avg_strategy_ret-avg_benchmark_ret)*100:.2f}%")
                print(f"   平均信息比率: {avg_ir:.4f}")
                print(f"   策略有效性: {'✅' if avg_ir > 0.1 else '⚠️ ' if avg_ir > 0 else '❌'}")
            else:
                print("⚠️  多因子分析未返回结果")

        except Exception as e:
            print(f"❌ 多因子分析出错: {e}")
            import traceback
            traceback.print_exc()

    def analyze_factors(self):
        """
        Analyze factor performance
        """
        print("\n🔍 因子表现分析...")

        if 'multi_factor_results' in self.session_data:
            results = self.session_data['multi_factor_results']
            print("\n📊 会话中存在多因子分析结果，显示因子表现:")

            for stock, result in results.items():
                print(f"\n   📊 {stock} 因子表现:")
                print(f"      信息比率 (IR): {result['info_ratio']:.4f}")
                print(f"      夏普比率: {result['sharpe_ratio']:.4f}")
                print(f"      最大回撤: {result['max_drawdown']*100:.2f}%")
                print(f"      波动率: {result['strategy_volatility']*100:.2f}%")
        else:
            print("\n💡 可以先运行 'multi_factor_analysis' 来生成因子分析数据")
            run_now = input("是否现在运行多因子分析? (y/n): ").strip().lower()
            if run_now == 'y':
                self.multi_factor_analysis()

    def factor_report(self):
        """
        Generate factor report
        """
        print("\n📋 生成因子报告...")

        if 'multi_factor_results' in self.session_data:
            results = self.session_data['multi_factor_results']

            print("\n" + "="*60)
            print("📈 100+因子多因子策略报告")
            print("="*60)

            # Create summary table
            summary_data = []
            for stock, result in results.items():
                summary_data.append({
                    '股票': stock,
                    '策略收益': f"{result['total_strategy_return']*100:.2f}%",
                    '基准收益': f"{result['total_benchmark_return']*100:.2f}%",
                    '超额收益': f"{(result['total_strategy_return']-result['total_benchmark_return'])*100:.2f}%",
                    '信息比率': f"{result['info_ratio']:.4f}",
                    '最大回撤': f"{result['max_drawdown']*100:.2f}%",
                    '夏普比率': f"{result['sharpe_ratio']:.4f}"
                })

            import pandas as pd
            summary_df = pd.DataFrame(summary_data)
            print(summary_df.to_string(index=False))

            # Overall metrics
            avg_strategy_ret = np.mean([r['total_strategy_return'] for r in results.values()])
            avg_benchmark_ret = np.mean([r['total_benchmark_return'] for r in results.values()])
            avg_ir = np.mean([r['info_ratio'] for r in results.values()])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in results.values()])
            avg_mdd = np.mean([r['max_drawdown'] for r in results.values()])

            print(f"\n🏆 整体指标:")
            print(f"   平均策略收益: {avg_strategy_ret*100:.2f}%")
            print(f"   平均基准收益: {avg_benchmark_ret*100:.2f}%")
            print(f"   平均超额收益: {(avg_strategy_ret - avg_benchmark_ret)*100:.2f}%")
            print(f"   平均信息比率: {avg_ir:.4f}")
            print(f"   平均夏普比率: {avg_sharpe:.4f}")
            print(f"   平均最大回撤: {avg_mdd*100:.2f}%")

            print(f"\n🎯 策略评价: {'优秀 ⭐⭐⭐' if avg_ir > 0.5 else '良好 ⭐⭐' if avg_ir > 0.2 else '一般 ⭐' if avg_ir > 0 else '待优化 ❌'}")

            print("="*60)
        else:
            print("\n⚠️  会话中无因子分析结果，请先运行 'multi_factor_analysis'")

    def run_comprehensive_qlib_analysis(self):
        """
        Run comprehensive Qlib-enhanced analysis (factor library expansion + model fusion + risk management + auto-tuning)
        """
        print("\n🌟 运行综合性的 Qlib 增强分析...")

        symbols_input = input("请输入股票代码 (用逗号分隔): ").strip()
        if not symbols_input:
            symbols = ['600023', '000001', '600519']  # Default stocks
            print("💡 使用默认股票列表")
        else:
            symbols = [s.strip() for s in symbols_input.split(',')]

        start_date = input("请输入开始日期 (YYYY-MM-DD, 默认: 2024-01-01): ").strip() or '2024-01-01'
        end_date = input("请输入结束日期 (YYYY-MM-DD, 默认: 2024-12-31): ").strip() or '2024-12-31'

        print(f"\n🚀 对 {len(symbols)} 只股票进行综合性分析...")
        print(f"📅 期间: {start_date} 至 {end_date}")

        try:
            # Create integrated enhancement system
            integrated_system = QlibIntegratedEnhancement()

            # Get data
            fetcher = DataFetcher(eastmoney_cookie=self.eastmoney_cookie)

            all_data = pd.DataFrame()
            for symbol in symbols:
                print(f"📊 获取 {symbol} 数据...")
                data = fetcher.fetch_stock_data(symbol, start_date, end_date)
                if not data.empty:
                    data['instrument'] = symbol
                    all_data = pd.concat([all_data, data], ignore_index=True)
                else:
                    print(f"⚠️ 未能获取 {symbol} 的数据")

            if all_data.empty:
                print("❌ 未能获取任何股票数据")
                return

            print(f"📈 开始综合性分析，共 {len(all_data)} 条记录...")

            # Run comprehensive analysis
            results = integrated_system.run_comprehensive_analysis(
                all_data,
                instruments=symbols,
                start_date=start_date,
                end_date=end_date
            )

            # Generate comprehensive report
            report = integrated_system.generate_comprehensive_report(results)
            print(f"\n📋 综合分析报告:")
            print(report)

            # Store results
            self.session_data['comprehensive_qlib_analysis'] = {
                'results': results,
                'report': report,
                'timestamp': datetime.now()
            }

            print(f"\n✅ 综合性 Qlib 增强分析完成!")
            print("💡 分析包含以下四个方面:")
            print("   1. 因子库扩充：Qlib Alpha因子 + MyTT指标")
            print("   2. 模型融合：传统技术指标 + ML模型")
            print("   3. 风险管理：Qlib风险模型 + 投资组合优化")
            print("   4. 自动调参：网格搜索 + 贝叶斯 + 遗传算法")

        except Exception as e:
            print(f"❌ 综合性 Qlib 分析出错: {e}")
            import traceback
            traceback.print_exc()

    def run_backtest(self):
        """
        Run backtesting for a strategy
        """
        if not self.backtester:
            print("❌ Tushare回测模块未初始化，请提供有效的token")
            return

        strategy_name = input("请输入策略名称 (ma_crossover/rsi/macd/bollinger/mean_reversion/breakout): ").strip()
        if not strategy_name:
            print("❌ 策略名称不能为空")
            return

        symbol = input("请输入股票代码 (例: 000001.SZ): ").strip()
        if not symbol:
            print("❌ 股票代码不能为空")
            return

        start_date = input("请输入开始日期 (YYYYMMDD, 默认: 20220101): ").strip() or "20220101"
        end_date = input("请输入结束日期 (YYYYMMDD, 默认: 20221231): ").strip() or "20221231"
        
        initial_capital = input("请输入初始资金 (默认: 100000): ").strip()
        initial_capital = int(initial_capital) if initial_capital.isdigit() else 100000

        print(f"\n🔬 运行 {strategy_name} 策略回测...")

        try:
            # Get the strategy
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if not strategy:
                print(f"❌ 策略 {strategy_name} 不存在")
                return

            # Run the backtest
            results = self.backtester.run_backtest(
                strategy=strategy,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                freq='D'
            )

            if results:
                print(f"\n✅ 回测完成:")
                print(f"   初始资金: ¥{results['initial_capital']:,.2f}")
                print(f"   最终价值: ¥{results['final_value']:,.2f}")
                print(f"   总收益率: {results['total_return']:.2%}")
                print(f"   年化收益率: {results['annualized_return']:.2%}")
                print(f"   最大回撤: {results['max_drawdown']:.2%}")
                print(f"   夏普比率: {results['sharpe_ratio']:.2f}" if 'sharpe_ratio' in results else "")
                
                # Store results
                self.session_data[f'backtest_{strategy_name}_{symbol}'] = results
            else:
                print("⚠️  回测未返回结果")
        except Exception as e:
            print(f"❌ 回测过程出错: {e}")

    def compare_strategies(self):
        """
        Compare multiple strategies
        """
        print("\n📊 策略比较功能...")
        
        symbols_input = input("请输入股票代码 (用逗号分隔): ").strip()
        if not symbols_input:
            print("❌ 请至少输入一只股票代码")
            return

        symbols = [s.strip() for s in symbols_input.split(',')]
        
        strategies_input = input("请输入策略名称 (用逗号分隔, 例: ma_crossover,rsi,macd): ").strip()
        if not strategies_input:
            print("❌ 请至少输入一个策略名称")
            return
            
        strategies_names = [s.strip() for s in strategies_input.split(',')]
        
        start_date = input("请输入开始日期 (YYYYMMDD, 默认: 20220101): ").strip() or "20220101"
        end_date = input("请输入结束日期 (YYYYMMDD, 默认: 20221231): ").strip() or "20221231"
        
        initial_capital = input("请输入初始资金 (默认: 100000): ").strip()
        initial_capital = int(initial_capital) if initial_capital.isdigit() else 100000

        if not self.backtester:
            print("❌ Tushare回测模块未初始化，请提供有效的token")
            return

        print(f"\n🔬 比较 {len(strategies_names)} 个策略在 {len(symbols)} 只股票上的表现...")

        try:
            comparison_results = {}

            for symbol in symbols:
                print(f"\n📈 分析 {symbol}...")
                symbol_results = {}

                for strategy_name in strategies_names:
                    print(f"  运行 {strategy_name} 策略...")

                    # Get the strategy
                    strategy = self.strategy_manager.get_strategy(strategy_name)
                    if not strategy:
                        print(f"    ❌ 策略 {strategy_name} 不存在")
                        continue

                    # Run the backtest
                    results = self.backtester.run_backtest(
                        strategy=strategy,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        initial_capital=initial_capital,
                        freq='D'
                    )

                    if results:
                        symbol_results[strategy_name] = results
                        print(f"    ✅ {strategy_name}: 收益率 {results['total_return']:.2%}")
                    else:
                        print(f"    ⚠️  {strategy_name}: 未返回结果")

                comparison_results[symbol] = symbol_results

            # Print comparison summary
            print(f"\n🏆 策略比较结果:")
            for symbol, results in comparison_results.items():
                print(f"\n  {symbol}:")
                for strategy_name, result in results.items():
                    print(f"    {strategy_name}: {result['total_return']:.2%} (最大回撤: {result['max_drawdown']:.2%})")

            # Store results
            self.session_data['strategy_comparison'] = comparison_results

        except Exception as e:
            print(f"❌ 策略比较过程出错: {e}")

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
                    if 'initial_capital' in value and 'final_value' in value:
                        # This looks like backtest results
                        print(f"  📊 {key}: Backtest results - ROI: {(value['final_value']/value['initial_capital']-1)*100:.2f}%")
                    else:
                        print(f"  📁 {key}: Dictionary with {len(value)} keys")
                else:
                    print(f"  📝 {key}: {type(value).__name__}")
        else:
            print("  📭 会话中无数据")

    def batch_analyze_watchlist(self):
        """
        批量分析自选股列表中的股票
        """
        print("\n🔍 批量分析自选股功能...")

        # 显示现有自选股列表
        watchlist_names = self.watchlist_manager.get_watchlist_names()
        print(f"📋 现有自选股列表: {watchlist_names}")

        # 选择自选股列表
        if len(watchlist_names) > 1:
            selected_watchlist = input(f"请选择自选股列表 (默认: default): ").strip() or "default"
        else:
            selected_watchlist = "default"

        watchlist = self.watchlist_manager.get_watchlist(selected_watchlist)

        if not watchlist:
            print("⚠️  选定的自选股列表为空")
            add_stocks = input("是否手动添加股票到列表? (y/n, 默认: n): ").strip().lower()
            if add_stocks == 'y':
                stocks_input = input("请输入股票代码 (用逗号分隔): ").strip()
                if stocks_input:
                    new_stocks = [s.strip() for s in stocks_input.split(',')]
                    for stock in new_stocks:
                        self.watchlist_manager.add_stock_to_watchlist(stock, selected_watchlist)
                    watchlist = self.watchlist_manager.get_watchlist(selected_watchlist)
                else:
                    print("❌ 未添加任何股票，操作取消")
                    return
            else:
                return

        print(f"📊 正在分析自选股列表 '{selected_watchlist}' 中的 {len(watchlist)} 只股票...")

        # 选择策略
        print("可用策略: ma_crossover, rsi, macd, bollinger, mean_reversion, breakout")
        strategy_name = input("请输入策略名称 (默认: ma_crossover): ").strip() or "ma_crossover"

        # 准备数据获取器
        data_fetcher = DataFetcher()

        # 分析结果存储
        analysis_results = []

        for stock_code in watchlist:
            try:
                print(f"📈 正在分析 {stock_code}...")

                # 获取数据
                data = data_fetcher.fetch_stock_data_ts_code(stock_code, days=60)
                if data is None or data.empty:
                    print(f"⚠️  无法获取 {stock_code} 的数据")
                    continue

                # 运行策略
                signals = self.strategy_manager.run_strategy(strategy_name, data)

                # 获取最新信号
                latest_signal = signals.iloc[-1] if len(signals) > 0 else 0
                signal_text = "买入" if latest_signal == 1 else "卖出" if latest_signal == -1 else "持有"

                # 进行简单回测
                backtest_result = self.strategy_manager.run_backtest(strategy_name, data)

                # 保存结果
                result = {
                    'stock_code': stock_code,
                    'signal': signal_text,
                    'signal_value': latest_signal,
                    'total_return': backtest_result.get('total_return', 0),
                    'sharpe_ratio': backtest_result.get('sharpe_ratio', 0),
                    'max_drawdown': backtest_result.get('max_drawdown', 0),
                    'last_price': data['close'].iloc[-1] if 'close' in data.columns else 0
                }

                analysis_results.append(result)

                print(f"  ✅ {stock_code} - 信号: {signal_text} (收益率: {result['total_return']:.2%})")

            except Exception as e:
                print(f"⚠️  分析 {stock_code} 时出错: {e}")
                continue

        # 显示汇总结果
        if analysis_results:
            df_results = pd.DataFrame(analysis_results)
            print(f"\n📋 批量分析结果 (按收益率排序):")
            print(df_results[['stock_code', 'signal', 'total_return', 'sharpe_ratio', 'max_drawdown', 'last_price']]
                  .sort_values('total_return', ascending=False))

            # 提取买入信号的股票
            buy_signals = df_results[df_results['signal'] == '买入']
            if not buy_signals.empty:
                print(f"\n💡 建议关注 (买入信号):")
                print(buy_signals[['stock_code', 'last_price', 'total_return', 'sharpe_ratio']])

            # 存储结果到会话
            self.session_data[f'batch_analysis_{selected_watchlist}'] = df_results

        else:
            print("❌ 没有成功分析任何股票")

    def manage_watchlist(self):
        """
        管理自选股列表
        """
        print("\n⭐ 自选股管理功能...")

        while True:
            print("\n请选择操作:")
            print("1. 查看自选股列表")
            print("2. 添加股票到自选股")
            print("3. 从自选股移除股票")
            print("4. 创建新的自选股列表")
            print("5. 删除自选股列表")
            print("6. 返回主菜单")

            choice = input("请输入选项 (1-6): ").strip()

            if choice == '1':
                watchlist_names = self.watchlist_manager.get_watchlist_names()
                for name in watchlist_names:
                    stocks = self.watchlist_manager.get_watchlist(name)
                    print(f"📋 {name}: {stocks}")

            elif choice == '2':
                watchlist_names = self.watchlist_manager.get_watchlist_names()
                watchlist_name = input(f"请选择自选股列表 (现有: {', '.join(watchlist_names)}, 默认: default): ").strip() or "default"
                stock_code = input("请输入股票代码: ").strip()
                if stock_code:
                    self.watchlist_manager.add_stock_to_watchlist(stock_code, watchlist_name)
                    print(f"✅ {stock_code} 已添加到 {watchlist_name}")

            elif choice == '3':
                watchlist_names = self.watchlist_manager.get_watchlist_names()
                watchlist_name = input(f"请选择自选股列表 (现有: {', '.join(watchlist_names)}, 默认: default): ").strip() or "default"
                stocks = self.watchlist_manager.get_watchlist(watchlist_name)
                if stocks:
                    print(f"{watchlist_name} 中的股票: {stocks}")
                    stock_code = input("请输入要移除的股票代码: ").strip()
                    if stock_code in stocks:
                        self.watchlist_manager.remove_stock_from_watchlist(stock_code, watchlist_name)
                        print(f"✅ {stock_code} 已从 {watchlist_name} 移除")
                    else:
                        print(f"❌ {stock_code} 不在 {watchlist_name} 中")
                else:
                    print(f"⚠️  {watchlist_name} 列表为空")

            elif choice == '4':
                new_name = input("请输入新的自选股列表名称: ").strip()
                if new_name:
                    self.watchlist_manager.create_watchlist(new_name)
                    print(f"✅ 已创建自选股列表: {new_name}")

            elif choice == '5':
                watchlist_names = self.watchlist_manager.get_watchlist_names()
                if len(watchlist_names) <= 1:
                    print("⚠️  至少保留一个自选股列表")
                else:
                    delete_name = input(f"请输入要删除的自选股列表名称 (现有: {', '.join(watchlist_names)}): ").strip()
                    if delete_name in watchlist_names and delete_name != "default":
                        confirm = input(f"确定删除自选股列表 '{delete_name}'? (y/N): ").strip().lower()
                        if confirm == 'y':
                            self.watchlist_manager.delete_watchlist(delete_name)
                            print(f"✅ 自选股列表 {delete_name} 已删除")
                    else:
                        print("❌ 无效的列表名称或不能删除默认列表")

            elif choice == '6':
                break

            else:
                print("❌ 无效选项")

    def clear_session(self):
        """
        Clear session data
        """
        self.session_data = {}
        self.current_stocks = []
        print("\n🗑️  会话数据已清空")


def main():
    """
    Main function to run the unified CLI interface
    """
    parser = argparse.ArgumentParser(description='A-Share Market Analysis Tool - Unified CLI Interface')
    parser.add_argument('--mode', choices=['interactive', 'screen', 'analyze', 'backtest', 'signals', 'predict'],
                       default='interactive', help='运行模式')
    parser.add_argument('--symbol', type=str, help='要分析的股票代码')
    parser.add_argument('--strategy', type=str, help='使用的策略名称')
    parser.add_argument('--start-date', type=str, help='回测开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, help='回测结束日期 (YYYYMMDD)')

    args = parser.parse_args()

    print("🔍 A股市场分析系统 - 统一CLI接口")
    print("="*50)

    # Use your tokens
    tushare_token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"

    # Your EastMoney cookie
    eastmoney_cookie = {
        'qgqp_b_id': 'b7c0c5065c6db033910b1b3175b7c9bb',
        'st_nvi': 'pr7nepf3axSLFdLauyP5y8deb',
        'websitepoptg_api_time': '1770690681021',
        'st_si': '43191381080720',
        'nid18': '0095a8fdc53e2c9dc00f4d602b3c459e',
        'nid18_create_time': '1770690681336',
        'gviem': '6A44mgyL6Tsg59OPlfAXDd677',
        'gviem_create_time': '1770690681337',
        'p_origin': 'https%3A%2F%2Fpassport2.eastmoney.com',
        'mtp': '1',
        'ct': 'wYdhYQ7SFCReRY7yObWFWJwcS2isXO6R8wHwamkysQRCcR9yEiEaMsskY-1tsHOmajDCrGLWHPVacX0DGd_9HoMFpWjxWtVUZEdR8ibclVermnomP1JWdjUpI3BhaRN2ft3jRsDjazoC6F9O5Jzssk-rkmWM3b3LsGJq5RJDxVM',
        'ut': 'FobyicMgeV5FJnFT189SwEfSo-wAjCKxRGfhgXzug4j9BdKmq4gQdtlHffBaUl7Djr5Ju3CTO3tQqVCOs_Vhp9WUQe_9zHJxPmg__J71QWWtiytGWHR6CUXelUQfxok_geZEOJXcc9bQWieI7LUcRQjQFmB-1bwzaZYU3t525uGbFHwr6SZYdP3PBVz04EfQ796KX06LCuYpITwvNu6laJotFHyE5dflMcANoRBf6d8isLvw34K59yZB985bsVHnckUA0HIycKAoU137ZeAYrEX8rjmONDCZy7QGj-BHcAWyIH9OIF98zmSo71GWwWu_X5FP1R2JqWLg9CMTh9wlVBTitMAXMcc5',
        'pi': '9694097255613200%3Bu9694097255613200%3B%E5%A0%82%E5%A0%82%E6%AD%A3%E6%AD%A3%E7%9A%84%E6%9B%B9%E6%93%8D%3BryhxoVjcWC8PTbi0bFrviFAowUa3asGIsa%2F0auHDuAKp6CJ%2BPVN0UwnSDOaEd7utp5uK4oSJImRgmTF0VD7Nm1Zqq9vnKuG5c1wWVRNZxJmnEN416UgEorQVUQJ5tnsTgIcvWxtVIJHhIll%2F9SIWv6E6wIrLFINK3wF12TZX3gkL7%2FxLaYbHaFQ0YON21YMY%2BZKCiilR%3Bp2dLhWNuZSa0SCigDD%2FOLxaCiti2fW5OSY32vbSSck%2BT1BzvA%2FAQHG2jYCxHc8Httaxt1PRsFPhuwvBF873qXa7Y5muaKZZN0jzerURbzjeerxd31x755Is9mu7LD%2BGWpkI3piLVRUUL5xl2ifRVnekqrax4Yg%3D%3D',
        'uidal': '9694097255613200%e5%a0%82%e5%a0%82%e6%ad%a3%e6%ad%a3%e7%9a%84%e6%9b%b9%e6%93%8d',
        'sid': '',
        'vtpst': '|',
        'st_asi': 'delete',
        'wsc_checkuser_ok': '1',
        'fullscreengg': '1',
        'fullscreengg2': '1',
        'st_pvi': '27562121748759',
        'st_sp': '2025-10-30%2011%3A15%3A42',
        'st_inirUrl': 'https%3A%2F%2Fwww.google.com.hk%2F',
        'st_sn': '5',
        'st_psi': '20260210130257951-111000300841-0487608401'
    }

    # Initialize unified interface
    cli_interface = UnifiedCLIInterface(tushare_token, eastmoney_cookie)

    if args.mode == 'interactive':
        # Run interactive mode
        cli_interface.run_interactive()
    elif args.mode == 'screen':
        # Run stock screening
        cli_interface.screen_stocks()
    elif args.mode == 'analyze':
        # Run stock analysis
        if not args.symbol or not args.strategy:
            print("❌ 请提供股票代码和策略名称")
            return
        cli_interface.analyze_stock()
    elif args.mode == 'backtest':
        # Run backtest
        if not args.symbol or not args.strategy or not args.start_date or not args.end_date:
            print("❌ 请提供股票代码、策略名称、开始日期和结束日期")
            return
        cli_interface.run_backtest()
    elif args.mode == 'signals':
        # Generate signals
        cli_interface.gen_signals()
    elif args.mode == 'predict':
        # Run prediction
        cli_interface.predict_stocks()


if __name__ == "__main__":
    main()