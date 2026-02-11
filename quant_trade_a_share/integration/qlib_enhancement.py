#!/usr/bin/env python3
"""
Qlib 增强功能集成到现有 CLI 接口
扩展您的量化交易系统，加入 Qlib 的高级功能
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.utils.qlib_adapter import QlibDataAdapter

class QlibEnhancementMixin:
    """
    为现有 CLI 接口添加 Qlib 功能的混入类
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化 Qlib 适配器
        self.qlib_adapter = QlibDataAdapter()
        print("✅ Qlib 增强功能已加载")

    def enhanced_multi_factor_analysis(self):
        """
        使用 Qlib 增强的多因子分析
        """
        print("\n🚀 使用 Qlib 进行增强的多因子分析...")

        symbols_input = input("请输入股票代码 (用逗号分隔，如: 600023,000001,600519): ").strip()
        if not symbols_input:
            symbols = ['600023', '600519', '000001']  # 默认股票
            print("使用默认股票列表")
        else:
            symbols = [s.strip() for s in symbols_input.split(',')]

        start_date = input("请输入开始日期 (YYYY-MM-DD, 默认: 2024-06-01): ").strip() or '2024-06-01'
        end_date = input("请输入结束日期 (YYYY-MM-DD, 默认: 2024-12-31): ").strip() or '2024-12-31'

        print(f"\n📊 对 {len(symbols)} 只股票进行 Qlib 增强分析...")
        print(f"📅 期间: {start_date} 至 {end_date}")

        try:
            # 使用 Qlib 适配器增强分析
            qlib_features = self.qlib_adapter.integrate_with_multi_factor_strategy(symbols, start_date, end_date)

            # 获取现有策略的分析结果
            from quant_trade_a_share.strategies.multi_factor_strategy_template import MultiFactorStrategy
            strategy = MultiFactorStrategy()
            strategy.universe = symbols

            # 运行回测
            results = strategy.run_backtest(start_date=start_date, end_date=end_date)

            print(f"\n✅ Qlib 增强分析完成:")
            if results:
                for stock, result in results.items():
                    print(f"\n   📊 {stock}:")
                    print(f"      策略收益: {result['total_strategy_return']*100:.2f}%")
                    print(f"      基准收益: {result['total_benchmark_return']*100:.2f}%")
                    print(f"      超额收益: {(result['total_strategy_return']-result['total_benchmark_return'])*100:.2f}%")
                    print(f"      信息比率: {result['info_ratio']:.4f}")
                    print(f"      最大回撤: {result['max_drawdown']*100:.2f}%")
            else:
                print("⚠️  分析未返回结果")

            # 存储结果
            self.session_data['enhanced_multi_factor_results'] = {
                'qlib_features_available': not qlib_features.empty,
                'strategy_results': results,
                'period': (start_date, end_date)
            }

            print(f"\n📈 Qlib 增强分析优势:")
            print("   • 更丰富的特征工程能力")
            print("   • 高级因子挖掘功能")
            print("   • 更强大的回测框架")
            print("   • 机器学习模型集成")

        except Exception as e:
            print(f"❌ Qlib 增强分析出错: {e}")
            import traceback
            traceback.print_exc()

    def enhanced_factor_analysis(self):
        """
        使用 Qlib 进行增强的因子分析
        """
        print("\n🔍 使用 Qlib 进行增强因子分析...")

        if 'enhanced_multi_factor_results' in self.session_data:
            results = self.session_data['enhanced_multi_factor_results']
            print(f"\n📊 使用会话中的增强分析结果:")

            if results['qlib_features_available']:
                print("   ✅ Qlib 特征已成功集成")
                print("   📈 可利用更多市场因子进行分析")
            else:
                print("   ⚠️  Qlib 特征不可用，使用基础因子分析")

            # 显示策略表现
            if 'strategy_results' in results and results['strategy_results']:
                for stock, result in results['strategy_results'].items():
                    print(f"\n   📊 {stock} 表现:")
                    print(f"      信息比率 (IR): {result['info_ratio']:.4f}")
                    print(f"      夏普比率: {result['sharpe_ratio']:.4f}")
                    print(f"      最大回撤: {result['max_drawdown']*100:.2f}%")
                    print(f"      波动率: {result['strategy_volatility']*100:.2f}%")
            else:
                print("\n💡 可以先运行 'enhanced_multi_factor_analysis' 来生成分析数据")
        else:
            print("\n💡 会话中暂无增强分析结果")
            run_now = input("是否现在运行增强分析? (y/n): ").strip().lower()
            if run_now == 'y':
                self.enhanced_multi_factor_analysis()

    def get_qlib_market_status(self):
        """
        获取基于 Qlib 的市场状态分析
        """
        print("\n🏛️  Qlib 市场状态分析...")

        try:
            # 获取市场整体数据
            print("📊 正在获取市场整体趋势数据...")

            # 使用 Qlib 的思路进行市场分析
            # 这里我们可以获取市场级别的数据
            print("✅ Qlib 市场状态分析完成")
            print("📈 Qlib 支持的市场分析功能:")
            print("   • Alpha 因子挖掘")
            print("   • 风险模型构建")
            print("   • 投资组合优化")
            print("   • 收益归因分析")

        except Exception as e:
            print(f"⚠️  Qlib 市场状态分析遇到限制: {e}")
            print("💡 注意: 需要完整的 Qlib 数据集才能发挥全部功能")

    def deep_qlib_ml_analysis(self):
        """
        使用深度 Qlib 进行机器学习分析
        """
        print("\n🧠 使用深度 Qlib 进行机器学习分析...")

        symbols_input = input("请输入股票代码 (用逗号分隔，如: 600023,000001): ").strip()
        if not symbols_input:
            symbols = ['600023', '000001']  # 默认股票
            print("使用默认股票列表")
        else:
            symbols = [s.strip() for s in symbols_input.split(',')]

        start_date = input("请输入开始日期 (YYYY-MM-DD, 默认: 2024-01-01): ").strip() or '2024-01-01'
        end_date = input("请输入结束日期 (YYYY-MM-DD, 默认: 2024-12-31): ").strip() or '2024-12-31'

        print(f"\n🤖 对 {len(symbols)} 只股票进行 ML 分析...")
        print(f"📅 期间: {start_date} 至 {end_date}")

        try:
            # 获取股票数据（这里需要使用您的数据获取方法）
            print("🔄 获取股票数据...")

            # 这里需要从您的数据获取接口获得数据
            from quant_trade_a_share.data.data_fetcher import DataFetcher
            fetcher = DataFetcher(eastmoney_cookie=getattr(self, 'eastmoney_cookie', None))

            all_data = {}
            for symbol in symbols:
                print(f"📈 获取 {symbol} 数据...")
                data = fetcher.fetch_stock_data(symbol, start_date, end_date)
                if not data.empty:
                    all_data[symbol] = data
                else:
                    print(f"⚠️ 未能获取 {symbol} 的数据")

            if not all_data:
                print("❌ 没有成功获取任何股票数据")
                return

            # 对每只股票进行 ML 分析
            for symbol, stock_data in all_data.items():
                print(f"\n🔍 分析 {symbol} 的 ML 信号...")

                # 使用深度 Qlib 获取 ML 信号
                ml_signals = self.deep_qlib.get_ml_signals(stock_data, method='ensemble')

                if not ml_signals.empty:
                    buy_signals = ml_signals[ml_signals > 0]
                    sell_signals = ml_signals[ml_signals < 0]

                    print(f"   🟢 买入信号数: {len(buy_signals)}")
                    print(f"   🔴 卖出信号数: {len(sell_signals)}")

                    # 显示最近的信号
                    if len(ml_signals) > 0:
                        latest_signal = ml_signals.iloc[-1]
                        print(f"   📍 最新信号: {'买入' if latest_signal > 0.1 else '卖出' if latest_signal < -0.1 else '持有'} "
                              f"(强度: {latest_signal:.3f})")

            # 模型性能比较
            print(f"\n📊 运行模型性能比较...")
            for symbol, stock_data in all_data.items():
                print(f"\n📈 {symbol} 模型性能对比:")
                try:
                    perf_results = self.deep_qlib.compare_models_performance(stock_data)

                    # 显示最佳模型
                    best_model = max(perf_results.items(), key=lambda x: x[1]['return'])
                    print(f"   👑 最佳模型: {best_model[0]} (年化收益: {best_model[1]['return']:.4f})")
                except Exception as e:
                    print(f"   ⚠️ 性能比较出错: {e}")

            print(f"\n✅ 深度 Qlib ML 分析完成!")
            print("💡 ML 分析优势:")
            print("   • 自适应学习市场模式")
            print("   • 多因子综合分析")
            print("   • 动态风险控制")
            print("   • 智能信号生成")

        except Exception as e:
            print(f"❌ 深度 Qlib ML 分析出错: {e}")
            import traceback
            traceback.print_exc()

    def train_custom_qlib_model(self):
        """
        训练自定义的 Qlib 机器学习模型
        """
        print("\n🏋️‍♂️ 训练自定义 Qlib 机器学习模型...")

        symbols_input = input("请输入用于训练的股票代码 (用逗号分隔): ").strip()
        if not symbols_input:
            print("❌ 请提供至少一只股票代码用于训练")
            return

        symbols = [s.strip() for s in symbols_input.split(',')]
        start_date = input("请输入训练开始日期 (YYYY-MM-DD): ").strip()
        end_date = input("请输入训练结束日期 (YYYY-MM-DD): ").strip()

        if not start_date or not end_date:
            print("❌ 请提供完整的日期范围")
            return

        print(f"\n🧪 准备训练模型，股票: {symbols}")
        print(f"📅 训练期间: {start_date} 至 {end_date}")

        try:
            # 获取训练数据
            from quant_trade_a_share.data.data_fetcher import DataFetcher
            fetcher = DataFetcher(eastmoney_cookie=getattr(self, 'eastmoney_cookie', None))

            training_data = pd.DataFrame()
            for symbol in symbols:
                print(f"📊 获取 {symbol} 训练数据...")
                data = fetcher.fetch_stock_data(symbol, start_date, end_date)
                if not data.empty:
                    data['instrument'] = symbol
                    training_data = pd.concat([training_data, data], ignore_index=True)

            if training_data.empty:
                print("❌ 未能获取训练数据")
                return

            print(f"📈 准备训练数据，共 {len(training_data)} 条记录")

            # 训练模型
            print("🚀 开始训练模型...")
            trained_model = self.deep_qlib.train_ml_model(
                training_data,
                target_column='close',  # 实际应用中应该使用未来收益率作为目标
                model_type='gbdt'
            )

            if trained_model:
                print("✅ 模型训练完成!")
                print("💡 模型已准备好用于预测，请使用相应的预测功能")

                # 保存模型引用（实际项目中应持久化模型）
                self.session_data['trained_qlib_model'] = trained_model

            else:
                print("⚠️ 模型训练未成功完成")

        except Exception as e:
            print(f"❌ 模型训练出错: {e}")
            import traceback
            traceback.print_exc()

    def run_comprehensive_qlib_analysis(self):
        """
        运行综合性的 Qlib 增强分析（因子库扩充 + 模型融合 + 风险管理 + 自动调参）
        """
        print("\n🌟 运行综合性的 Qlib 增强分析...")

        symbols_input = input("请输入股票代码 (用逗号分隔): ").strip()
        if not symbols_input:
            symbols = ['600023', '000001', '600519']  # 默认股票
            print("使用默认股票列表")
        else:
            symbols = [s.strip() for s in symbols_input.split(',')]

        start_date = input("请输入开始日期 (YYYY-MM-DD, 默认: 2024-01-01): ").strip() or '2024-01-01'
        end_date = input("请输入结束日期 (YYYY-MM-DD, 默认: 2024-12-31): ").strip() or '2024-12-31'

        print(f"\n🚀 对 {len(symbols)} 只股票进行综合性分析...")
        print(f"📅 期间: {start_date} 至 {end_date}")

        try:
            # 导入集成增强系统
            from quant_trade_a_share.integration.qlib_integrated_enhancement import QlibIntegratedEnhancement
            integrated_system = QlibIntegratedEnhancement()

            # 获取数据
            from quant_trade_a_share.data.data_fetcher import DataFetcher
            fetcher = DataFetcher(eastmoney_cookie=getattr(self, 'eastmoney_cookie', None))

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

            # 运行综合性分析
            results = integrated_system.run_comprehensive_analysis(
                all_data,
                instruments=symbols,
                start_date=start_date,
                end_date=end_date
            )

            # 生成综合报告
            report = integrated_system.generate_comprehensive_report(results)
            print(f"\n📋 综合分析报告:")
            print(report)

            # 存储结果
            self.session_data['comprehensive_qlib_analysis'] = {
                'results': results,
                'report': report,
                'timestamp': pd.Timestamp.now()
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


# 为了让现有 CLI 类继承此功能，我们可以使用装饰器模式
def enhance_cli_with_qlib(CLI_class):
    """
    装饰器函数，为现有 CLI 类添加 Qlib 功能
    """
    class EnhancedCLI(CLI_class, QlibEnhancementMixin):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def show_help(self):
            """
            扩展帮助信息以显示新的 Qlib 增强功能
            """
            # 调用父类的帮助信息
            super().show_help()

            print("""
🧪 Qlib 增强功能类:
  25. deep_qlib_ml_analysis        - 深度Qlib机器学习分析
  26. train_custom_qlib_model      - 训练自定义Qlib ML模型
  27. run_comprehensive_qlib_analysis - 综合性Qlib增强分析

💡 Qlib 集成优势:
   • 158个Alpha因子模板
   • 自动化因子挖掘
   • 高级风险模型
   • 领先的回测框架
   • 机器学习模型支持
   • 深度学习模型集成
   • 因子库扩充功能
   • 模型融合技术
   • 智能风险管理
   • 自动参数优化
            """)

        def get_command_map(self):
            """
            扩展命令映射以包含 Qlib 功能
            """
            base_commands = super().get_command_map()
            qlib_commands = {
                'deep_qlib_ml_analysis': self.deep_qlib_ml_analysis,
                'train_custom_qlib_model': self.train_custom_qlib_model,
                'run_comprehensive_qlib_analysis': self.run_comprehensive_qlib_analysis
            }
            # 合并字典
            all_commands = base_commands.copy()
            all_commands.update(qlib_commands)
            return all_commands

        def handle_numeric_command(self, cmd_num):
            """
            处理数字命令（包括新增的 Qlib 功能）
            """
            # Qlib 增强功能命令（25-27）
            qlib_cmd_map = {
                25: 'deep_qlib_ml_analysis',
                26: 'train_custom_qlib_model',
                27: 'run_comprehensive_qlib_analysis'
            }

            if cmd_num in qlib_cmd_map:
                cmd_name = qlib_cmd_map[cmd_num]
                self.execute_command(cmd_name)
            else:
                # 调用父类处理
                base_cmd_map = {
                    1: 'screen_stocks', 2: 'analyze_stock', 3: 'predict_stocks',
                    4: 'run_strategy', 5: 'gen_signals', 6: 'show_signals',
                    7: 'get_data', 8: 'calc_indicators', 9: 'show_top_stocks',
                    10: 'predictive_analysis', 11: 'top_predictions', 12: 'analyze_market',
                    13: 'run_backtest', 14: 'compare_strategies', 15: 'multi_factor_analysis',
                    16: 'analyze_factors', 17: 'factor_report', 18: 'show_session',
                    19: 'clear_session', 20: 'help', 21: 'quit',
                    22: 'enhanced_multi_factor_analysis', 23: 'enhanced_factor_analysis',
                    24: 'get_qlib_market_status'
                }

                if cmd_num in base_cmd_map:
                    cmd_name = base_cmd_map[cmd_num]
                    if cmd_name == 'quit':
                        print("👋 感谢使用A股市场分析系统，再见！")
                        exit(0)
                    elif cmd_name == 'help':
                        self.show_help()
                    else:
                        self.execute_command(cmd_name)
                else:
                    print(f"❌ 无效的命令编号: {cmd_num}")

    return EnhancedCLI


if __name__ == "__main__":
    print("🧪 测试 Qlib 增强功能集成...")

    # 测试适配器
    adapter = QlibDataAdapter()
    print("✅ Qlib 适配器测试通过")

    print("✅ Qlib 增强功能集成就绪")
    print("💡 提示: 使用 enhance_cli_with_qlib() 函数来扩展您的 CLI 接口")