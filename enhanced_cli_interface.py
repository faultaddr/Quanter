#!/usr/bin/env python3
"""
增强版统一 CLI 接口
集成了 Qlib 高级功能的量化交易系统
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.screeners.stock_screener import StockScreener
from quant_trade_a_share.strategies import StrategyManager
from quant_trade_a_share.signals.signal_notifier import SignalNotifier
from quant_trade_a_share.prediction.predictive_analyzer import PredictiveAnalyzer
from quant_trade_a_share.backtest.backtester_tushare import BacktesterWithTushare
from quant_trade_a_share.data.data_fetcher import DataFetcher
from multi_factor_strategy_template import MultiFactorStrategy
from quant_trade_a_share.integration.qlib_enhancement import enhance_cli_with_qlib


class UnifiedCLIInterface:
    """
    统一 CLI 接口，用于 A 股市场分析系统
    """
    def __init__(self, tushare_token, eastmoney_cookie):
        self.tushare_token = tushare_token
        self.eastmoney_cookie = eastmoney_cookie

        # 初始化所有系统组件
        self.screener = StockScreener(tushare_token=tushare_token)
        self.strategy_manager = StrategyManager()
        self.signal_notifier = SignalNotifier()
        self.predictive_analyzer = PredictiveAnalyzer()

        # 初始化回测器（如果有 token）
        self.backtester = None
        if tushare_token:
            try:
                self.backtester = BacktesterWithTushare(tushare_token)
            except Exception as e:
                print(f"⚠️  无法初始化Tushare回测模块: {e}")

        # 初始化数据获取器
        self.data_fetcher = DataFetcher()

        # 初始化多因子策略
        self.multi_factor_strategy = MultiFactorStrategy()

        # 存储会话数据
        self.session_data = {}
        self.current_stocks = []

        print("✅ A股市场分析系统统一接口初始化完成")
        print("="*60)

    def show_help(self):
        """
        显示帮助信息及可用命令
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

📊 多因子分析类:
  15. multi_factor_analysis - 运行100+因子分析
  16. analyze_factors   - 分析因子表现
  17. factor_report     - 生成因子报告

🧪 Qlib 增强功能类:
  22. enhanced_multi_factor_analysis - 运行Qlib增强的多因子分析
  23. enhanced_factor_analysis      - 进行Qlib增强的因子分析
  24. get_qlib_market_status       - 获取Qlib市场状态分析

⚙️  系统管理类:
  18. show_session     - 显示会话数据
  19. clear_session    - 清空会话数据
  20. help             - 显示帮助信息
  21. quit/exit        - 退出系统

💡 使用方法: 输入命令编号或命令名称
   例如: 输入 '1' 或 'screen_stocks' 开始股票筛选
=======================================
        """)

    def run_interactive(self):
        """
        运行交互式控制台
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
                    # 处理数字命令
                    cmd_num = int(user_input)
                    self.handle_numeric_command(cmd_num)
                elif user_input in self.get_command_map():
                    # 处理命令名称
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
        获取命令名称到函数的映射
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
            'show_session': self.show_session,
            'clear_session': self.clear_session
        }

    def handle_numeric_command(self, cmd_num):
        """
        处理数字命令
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
            18: 'show_session',
            19: 'clear_session',
            20: 'help',
            21: 'quit'
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
        执行命令
        """
        cmd_map = self.get_command_map()
        if cmd_name in cmd_map:
            try:
                cmd_map[cmd_name]()
            except Exception as e:
                print(f"❌ 执行 {cmd_name} 时出错: {e}")
        else:
            print(f"❌ 未知命令: {cmd_name}")

    # ... 其他原有方法保持不变 ...

    def screen_stocks(self):
        """
        筛选潜在上涨股票
        """
        print("\n🔍 开始筛选潜在上涨股票 (市值>200亿)...")

        # 使用默认筛选条件
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
        分析单个股票
        """
        symbol = input("请输入股票代码 (例: sh600519): ").strip()
        if not symbol:
            print("❌ 股票代码不能为空")
            return

        strategy_name = input("请输入策略名称 (ma_crossover/rsi/macd/bollinger/mean_reversion/breakout，默认: ma_crossover): ").strip() or 'ma_crossover'

        # 询问数据源
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: auto (优先使用Ashare)): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock', 'auto'] else 'auto'

        print(f"\n📊 分析股票 {symbol} 使用 {strategy_name} 策略...")
        print(f"📈 使用数据源: {source}")

        try:
            # 获取股票数据
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

            data = self.data_fetcher.fetch(symbol, start_date, end_date, source=source)
            if data is None or data.empty:
                print(f"❌ 无法从{source}获取 {symbol} 的数据，尝试使用screener...")
                # 回退到 screener
                data = self.screener.fetch_stock_data(symbol, period='180', data_source=source)
                if data is None or data.empty:
                    print(f"⚠️  也无法从screener获取 {symbol} 数据，跳过")
                    return

            # 获取股票名称
            if self.screener.chinese_stocks is None:
                self.screener.get_chinese_stocks_list()

            stock_info = self.screener.chinese_stocks[self.screener.chinese_stocks['symbol'] == symbol] if self.screener.chinese_stocks is not None else pd.DataFrame()
            stock_name = symbol  # 默认使用代码作为名称
            if not stock_info.empty and 'name' in stock_info.columns:
                stock_name = stock_info['name'].iloc[0]

            # 获取策略
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if strategy is None:
                print(f"❌ 策略 {strategy_name} 不存在")
                return

            # 生成信号
            signals = strategy.generate_signals(data)

            # 计算近期表现（如果有足够数据）
            if not data.empty and len(data) > 0:
                recent_performance = ((data['close'].iloc[-1] - data['close'].iloc[0]) /
                                     data['close'].iloc[0]) * 100
                current_price = data['close'].iloc[-1]
            else:
                recent_performance = 0
                current_price = 0
                print(f"⚠️  {symbol} 数据不足，无法计算近期表现")

            print(f"\n✅ {symbol} ({stock_name}) 分析完成:")
            if current_price > 0:
                print(f"   当前价格: {current_price:.2f}")
            else:
                print(f"   当前价格: N/A")
            print(f"   近期表现: {recent_performance:.2f}%")
            print(f"   生成信号数: {len(signals[signals != 0]) if len(signals) > 0 else 0}")
            print(f"   最新信号: {signals.iloc[-1] if len(signals) > 0 else 0}")

            # 存储到会话
            self.session_data[f'analysis_{symbol}'] = {
                'symbol': symbol,
                'name': stock_name,
                'data': data,
                'signals': signals if len(signals) > 0 else pd.Series(dtype=float),
                'recent_performance': recent_performance
            }

        except Exception as e:
            print(f"❌ 分析过程出错: {e}")

    def predict_stocks(self):
        """
        预测股票走势
        """
        symbols_input = input("请输入股票代码 (用逗号分隔，留空使用默认): ").strip()
        if symbols_input:
            symbols = [s.strip() for s in symbols_input.split(',')]
        else:
            symbols = ['sh600519', 'sz000858', 'sh600036']  # 默认股票

        top_n = input("请输入返回数量 (默认: 10): ").strip()
        top_n = int(top_n) if top_n.isdigit() else 10

        print(f"\n🔮 预测 {len(symbols)} 只股票的上涨概率...")

        try:
            predictions = self.predictive_analyzer.analyze_stocks(symbols=symbols, top_n=top_n)

            if not predictions.empty:
                print(f"\n✅ 预测完成，共分析 {len(predictions)} 只股票:")
                self.predictive_analyzer.print_top_predictions(predictions, top_n=min(top_n, len(predictions)))

                # 存储预测结果
                self.session_data['predictions'] = predictions
            else:
                print("⚠️  预测分析未返回结果")
        except Exception as e:
            print(f"❌ 预测过程出错: {e}")

    def run_strategy(self):
        """
        运行特定策略
        """
        strategy_name = input("请输入策略名称 (ma_crossover/rsi/macd/bollinger/mean_reversion/breakout): ").strip()
        if not strategy_name:
            print("❌ 策略名称不能为空")
            return

        symbols_input = input("请输入股票代码 (用逗号分隔): ").strip()
        if not symbols_input:
            print("❌ 请至少输入一只股票代码")
            return

        # 询问数据源
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

                # 获取股票数据
                # 使用最近180天的数据
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

                data = self.data_fetcher.fetch(symbol, start_date, end_date, source=source)
                if data is None or data.empty:
                    print(f"⚠️  无法从{source}获取 {symbol} 数据，尝试使用screener...")
                    # 如果 DataFetcher 失败，回退到 screener
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

            # 存储结果
            self.session_data[f'strategy_{strategy_name}'] = all_results

        except Exception as e:
            print(f"❌ 策略执行出错: {e}")

    def gen_signals(self):
        """
        生成买卖信号
        """
        symbols_input = input("请输入股票代码 (用逗号分隔): ").strip()
        if not symbols_input:
            print("❌ 请至少输入一只股票代码")
            return

        # 询问数据源
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: auto (优先使用Ashare)): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock', 'auto'] else 'auto'

        symbols = [s.strip() for s in symbols_input.split(',')]

        print(f"\n🔔 为 {len(symbols)} 只股票生成买卖信号...")
        print(f"📈 使用数据源: {source}")

        try:
            all_signals = []

            for symbol in symbols:
                print(f"📈 分析 {symbol}...")

                # 获取股票数据
                # 使用最近180天的数据
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

                data = self.data_fetcher.fetch(symbol, start_date, end_date, source=source)
                if data is None or data.empty:
                    print(f"⚠️  无法从{source}获取 {symbol} 数据，尝试使用screener...")
                    # 如果 DataFetcher 失败，回退到 screener
                    data = self.screener.fetch_stock_data(symbol, period='180', data_source=source)
                    if data is None or data.empty:
                        print(f"⚠️  也无法从screener获取 {symbol} 数据，跳过")
                        continue

                # 应用策略生成信号
                for strategy_name in ['ma_crossover', 'rsi', 'macd']:
                    try:
                        strategy = self.strategy_manager.get_strategy(strategy_name)
                        if strategy:
                            signals = strategy.generate_signals(data)

                            # 处理并发送信号
                            stock_name = symbol  # 通常会从股票列表获取名称

                            # 获取最新信号
                            if len(signals) > 0:
                                latest_signal = signals.iloc[-1]
                                latest_price = data['close'].iloc[-1] if 'close' in data.columns else None
                                latest_date = data.index[-1] if not data.empty else datetime.now()

                                if latest_signal == 1:  # 买入信号
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

                                elif latest_signal == -1:  # 卖出信号
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

            # 获取最新信号以显示
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
        显示最新信号
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
        获取股票数据
        """
        symbol = input("请输入股票代码 (例: sh600519): ").strip()
        if not symbol:
            print("❌ 股票代码不能为空")
            return

        days = input("请输入获取天数 (默认: 30): ").strip()
        days = int(days) if days.isdigit() else 30

        # 询问数据源
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: eastmoney): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock'] else 'eastmoney'

        print(f"\n📊 从 {source} 获取 {symbol} 最近 {days} 天数据...")
        try:
            # 计算日期范围
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

            data = self.data_fetcher.fetch(symbol, start_date, end_date, source=source)

            if data is not None and not data.empty:
                print(f"\n✅ 从 {source} 获取到 {len(data)} 条数据:")
                print(data[['open', 'close', 'high', 'low', 'volume']].tail(5).to_string())

                # 存储到会话
                self.session_data[f'data_{symbol}_{source}'] = data
            else:
                print("⚠️  无法获取数据")
        except Exception as e:
            print(f"❌ 获取数据出错: {e}")

    def calc_indicators(self):
        """
        计算技术指标
        """
        symbol = input("请输入股票代码 (例: sh600519): ").strip()
        if not symbol:
            print("❌ 股票代码不能为空")
            return

        # 询问数据源
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: auto (优先使用Ashare)): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock', 'auto'] else 'auto'

        print(f"\n🧮 计算 {symbol} 技术指标...")
        print(f"📈 使用数据源: {source}")

        try:
            # 使用 screener 获取数据并计算增强技术指标
            data = self.screener.fetch_stock_data(symbol, period='180', data_source=source)

            if data is None or data.empty:
                print(f"❌ 无法从{source}获取 {symbol} 数据")
                return

            # 计算指标
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

            # 存储到会话
            self.session_data[f'indicators_{symbol}'] = data

        except Exception as e:
            print(f"❌ 计算指标出错: {e}")

    def show_top_stocks(self):
        """
        显示热门股票
        """
        print("\n🔝 获取热门股票列表...")

        try:
            # 使用可用的方法获取股票列表
            all_stocks = self.screener.get_chinese_stocks_list()
            if all_stocks is not None and not all_stocks.empty:
                # 获取前10只股票作为热门股票
                top_stocks = all_stocks.head(10)

                print(f"\n✅ 获取到 {len(top_stocks)} 只股票:")
                for i, (idx, stock) in enumerate(top_stocks.iterrows(), 1):
                    symbol = stock.get('symbol', 'N/A')
                    name = stock.get('name', 'N/A') if 'name' in stock else 'N/A'
                    print(f"  {i}. {name} ({symbol})")

                # 存储到会话
                self.session_data['top_stocks'] = top_stocks
            else:
                print("⚠️  无法获取股票列表")
        except Exception as e:
            print(f"❌ 获取热门股票出错: {e}")
            import traceback
            traceback.print_exc()

    def predictive_analysis(self):
        """
        运行预测分析
        """
        print("\n🔮 运行预测分析...")

        try:
            # 获取活跃股票进行分析
            top_stocks = self.screener.get_top_active_stocks(limit=20)
            symbols = [stock[0] for stock in top_stocks] if top_stocks else ['sh600519', 'sz000858']

            predictions = self.predictive_analyzer.analyze_stocks(symbols=symbols, top_n=10)

            if not predictions.empty:
                print(f"\n✅ 预测分析完成，共分析 {len(predictions)} 只股票:")
                self.predictive_analyzer.print_top_predictions(predictions, top_n=10)

                # 存储到会话
                self.session_data['predictions'] = predictions
            else:
                print("⚠️  预测分析未返回结果")
        except Exception as e:
            print(f"❌ 预测分析出错: {e}")

    def top_predictions(self):
        """
        显示会话中的顶级预测
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
        分析整体市场
        """
        print("\n🏛️  市场整体分析...")

        try:
            # 获取市场概览数据
            top_stocks = self.screener.get_top_active_stocks(limit=50)

            if top_stocks:
                print(f"\n📊 市场概览 (共{len(top_stocks)}只活跃股票):")

                # 计算市场统计
                total_rising = 0
                total_falling = 0
                avg_volume = 0
                total_volume = 0

                for stock in top_stocks:
                    symbol = stock[0]
                    try:
                        data = self.screener.fetch_stock_data(symbol, days=5)
                        if data is not None and not data.empty and len(data) >= 2:
                            # 计算每日变化
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
        运行100+因子多因子分析
        """
        print("\n📊 100+因子多因子分析...")

        symbols_input = input("请输入股票代码 (用逗号分隔，如: sh600023,sz000001,sh600519): ").strip()
        if not symbols_input:
            symbols = ['sh600023', 'sh600519', 'sz000001']  # 默认股票
            print("使用默认股票列表")
        else:
            symbols = [s.strip() for s in symbols_input.split(',')]

        start_date = input("请输入开始日期 (YYYY-MM-DD, 默认: 2024-06-01): ").strip() or '2024-06-01'
        end_date = input("请输入结束日期 (YYYY-MM-DD, 默认: 2024-12-31): ").strip() or '2024-12-31'

        print(f"\n🚀 对 {len(symbols)} 只股票进行100+因子分析...")
        print(f"📅 期间: {start_date} 至 {end_date}")

        try:
            # 更新策略股票池
            self.multi_factor_strategy.universe = symbols

            # 运行回测，计算100+因子
            results = self.multi_factor_strategy.run_backtest(start_date=start_date, end_date=end_date)

            if results:
                print(f"\n✅ 多因子分析完成:")
                print(f"📈 共分析 {len(results)} 只股票")

                # 显示结果
                for stock, result in results.items():
                    print(f"\n   📊 {stock}:")
                    print(f"      策略收益: {result['total_strategy_return']*100:.2f}%")
                    print(f"      基准收益: {result['total_benchmark_return']*100:.2f}%")
                    print(f"      超额收益: {(result['total_strategy_return']-result['total_benchmark_return'])*100:.2f}%")
                    print(f"      信息比率: {result['info_ratio']:.4f}")
                    print(f"      最大回撤: {result['max_drawdown']*100:.2f}%")

                # 存储结果到会话
                self.session_data['multi_factor_results'] = results

                # 显示汇总
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
        分析因子表现
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
        生成因子报告
        """
        print("\n📋 生成因子报告...")

        if 'multi_factor_results' in self.session_data:
            results = self.session_data['multi_factor_results']

            print("\n" + "="*60)
            print("📈 100+因子多因子策略报告")
            print("="*60)

            # 创建汇总表
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

            # 整体指标
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

    def run_backtest(self):
        """
        运行策略回测
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
            # 获取策略
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if not strategy:
                print(f"❌ 策略 {strategy_name} 不存在")
                return

            # 运行回测
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

                # 存储结果
                self.session_data[f'backtest_{strategy_name}_{symbol}'] = results
            else:
                print("⚠️  回测未返回结果")
        except Exception as e:
            print(f"❌ 回测过程出错: {e}")

    def compare_strategies(self):
        """
        比较多种策略
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

                    # 获取策略
                    strategy = self.strategy_manager.get_strategy(strategy_name)
                    if not strategy:
                        print(f"    ❌ 策略 {strategy_name} 不存在")
                        continue

                    # 运行回测
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

            # 打印比较汇总
            print(f"\n🏆 策略比较结果:")
            for symbol, results in comparison_results.items():
                print(f"\n  {symbol}:")
                for strategy_name, result in results.items():
                    print(f"    {strategy_name}: {result['total_return']:.2%} (最大回撤: {result['max_drawdown']:.2%})")

            # 存储结果
            self.session_data['strategy_comparison'] = comparison_results

        except Exception as e:
            print(f"❌ 策略比较过程出错: {e}")

    def show_session(self):
        """
        显示会话数据
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
                        # 这看起来像回测结果
                        print(f"  📊 {key}: Backtest results - ROI: {(value['final_value']/value['initial_capital']-1)*100:.2f}%")
                    else:
                        print(f"  📁 {key}: Dictionary with {len(value)} keys")
                else:
                    print(f"  📝 {key}: {type(value).__name__}")
        else:
            print("  📭 会话中无数据")

    def clear_session(self):
        """
        清空会话数据
        """
        self.session_data = {}
        self.current_stocks = []
        print("\n🗑️  会话数据已清空")


def main():
    """
    主函数，运行统一 CLI 接口
    """
    parser = argparse.ArgumentParser(description='A-Share Market Analysis Tool - Unified CLI Interface')
    parser.add_argument('--mode', choices=['interactive', 'screen', 'analyze', 'backtest', 'signals', 'predict'],
                       default='interactive', help='运行模式')
    parser.add_argument('--symbol', type=str, help='要分析的股票代码')
    parser.add_argument('--strategy', type=str, help='使用的策略名称')
    parser.add_argument('--start-date', type=str, help='回测开始日期 (YYYYMMDD)')
    parser.add_argument('--end-date', type=str, help='回测结束日期 (YYYYMMDD)')

    args = parser.parse_args()

    print("🔍 A股市场分析系统 - 统一CLI接口 (增强版)")
    print("="*50)

    # 使用您的令牌
    tushare_token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"

    # 您的 EastMoney cookie
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

    # 初始化统一接口
    basic_cli = UnifiedCLIInterface(tushare_token, eastmoney_cookie)

    # 使用增强功能装饰器
    EnhancedCLIInterface = enhance_cli_with_qlib(UnifiedCLIInterface)
    cli_interface = EnhancedCLIInterface(tushare_token, eastmoney_cookie)

    if args.mode == 'interactive':
        # 运行交互模式
        cli_interface.run_interactive()
    elif args.mode == 'screen':
        # 运行股票筛选
        cli_interface.screen_stocks()
    elif args.mode == 'analyze':
        # 运行股票分析
        if not args.symbol or not args.strategy:
            print("❌ 请提供股票代码和策略名称")
            return
        cli_interface.analyze_stock()
    elif args.mode == 'backtest':
        # 运行回测
        if not args.symbol or not args.strategy or not args.start_date or not args.end_date:
            print("❌ 请提供股票代码、策略名称、开始日期和结束日期")
            return
        cli_interface.run_backtest()
    elif args.mode == 'signals':
        # 生成信号
        cli_interface.gen_signals()
    elif args.mode == 'predict':
        # 运行预测
        cli_interface.predict_stocks()


if __name__ == "__main__":
    main()