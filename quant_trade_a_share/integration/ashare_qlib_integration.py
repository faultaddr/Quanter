"""
Ashare 与 Qlib 集成模块
专门用于在 Qlib 增强分析中直接使用 Ashare 数据，以当日之前 180 天数据作为输入
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from quant_trade_a_share.data.ashare_data_fetcher import AshareDataFetcher
from quant_trade_a_share.integration.qlib_integrated_enhancement import QlibIntegratedEnhancement
import warnings
warnings.filterwarnings('ignore')


class AshareQlibIntegration:
    """
    Ashare 与 Qlib 集成类
    实现使用 Ashare 数据源进行 Qlib 增强分析的功能
    """

    def __init__(self):
        """初始化集成模块"""
        self.ashare_fetcher = AshareDataFetcher()
        self.qlib_enhancer = QlibIntegratedEnhancement()
        self.data_cache = {}

        print("✅ Ashare-Qlib 集成模块初始化成功")

    def fetch_ashare_data_for_qlib(self, symbol: str, days: int = 180) -> Optional[pd.DataFrame]:
        """
        从 Ashare 获取指定天数的历史数据

        Args:
            symbol: 股票代码
            days: 需要获取的天数，默认 180 天

        Returns:
            pandas.DataFrame: 包含股票数据的 DataFrame
        """
        print(f"📊 从 Ashare 获取 {symbol} 的 {days} 天历史数据...")

        # 使用 Ashare 获取数据
        data = self.ashare_fetcher.fetch_stock_data(symbol, days=days)

        if data is not None and not data.empty:
            # 确保列名符合预期格式
            if 'time' not in data.columns and data.index.name is None:
                data.index.name = 'time'
                data.reset_index(inplace=True)

            # 标准化列名
            column_mapping = {
                'time': 'date',
                'day': 'date',
                'Date': 'date',
                'Datetime': 'date'
            }

            for old_col, new_col in column_mapping.items():
                if old_col in data.columns and new_col not in data.columns:
                    data.rename(columns={old_col: new_col}, inplace=True)

            # 设置日期列为索引
            if 'date' in data.columns:
                data.set_index('date', inplace=True)
            elif 'time' in data.columns:
                data.set_index('time', inplace=True)

            # 检查并标准化列名
            required_columns = ['open', 'close', 'high', 'low', 'volume']
            actual_columns = data.columns.tolist()

            # 映射可能的不同列名
            column_standardization = {
                'Open': 'open',
                'Close': 'close',
                'High': 'high',
                'Low': 'low',
                'Volume': 'volume',
                'open_price': 'open',
                'close_price': 'close',
                'high_price': 'high',
                'low_price': 'low'
            }

            for old_col, new_col in column_standardization.items():
                if old_col in data.columns and new_col not in data.columns:
                    data.rename(columns={old_col: new_col}, inplace=True)

            # 确保数值列是正确的数据类型
            for col in ['open', 'close', 'high', 'low', 'volume']:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')

            # 移除任何 NaN 行
            data.dropna(inplace=True)

            print(f"✅ 成功获取 {len(data)} 条 {symbol} 的数据")
            return data
        else:
            print(f"❌ 无法从 Ashare 获取 {symbol} 的数据")
            return None

    def prepare_qlib_input_data(self, symbol: str, days: int = 180) -> Tuple[pd.DataFrame, str, str, List[str]]:
        """
        准备 Qlib 分析所需的输入数据

        Args:
            symbol: 股票代码
            days: 回溯天数

        Returns:
            tuple: (数据框, 开始日期, 结束日期, 股票列表)
        """
        print(f"🔍 准备 Qlib 分析输入数据...")

        # 获取数据
        data = self.fetch_ashare_data_for_qlib(symbol, days)

        if data is None or data.empty:
            print("❌ 数据获取失败，无法继续分析")
            return pd.DataFrame(), "", "", []

        # 确定日期范围
        end_date = data.index[-1].strftime('%Y-%m-%d') if len(data) > 0 else datetime.now().strftime('%Y-%m-%d')
        start_date = data.index[0].strftime('%Y-%m-%d') if len(data) > 0 else (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        # 创建股票列表
        symbols_list = [symbol]

        print(f"📅 日期范围: {start_date} 至 {end_date}")
        print(f"📈 股票列表: {symbols_list}")

        return data, start_date, end_date, symbols_list

    def run_comprehensive_qlib_analysis_with_ashare(self, symbol: str, days: int = 180) -> Dict[str, Any]:
        """
        使用 Ashare 数据运行综合性 Qlib 分析

        Args:
            symbol: 股票代码
            days: 回溯天数，默认 180 天

        Returns:
            dict: 分析结果
        """
        print(f"🚀 开始使用 Ashare 数据运行综合性 Qlib 分析...")
        print(f"🎯 股票: {symbol}, 回溯天数: {days}")

        # 准备输入数据
        data, start_date, end_date, symbols = self.prepare_qlib_input_data(symbol, days)

        if data.empty:
            print("❌ 输入数据为空，无法执行分析")
            return {}

        print(f"📊 准备分析数据，样本数: {len(data)}")

        # 运行综合性 Qlib 分析
        analysis_results = self.qlib_enhancer.run_comprehensive_analysis(
            data=data,
            instruments=symbols,
            start_date=start_date,
            end_date=end_date
        )

        # 生成报告
        report = self.qlib_enhancer.generate_comprehensive_report(analysis_results)

        print("\n" + "="*50)
        print("📊 综合性分析报告")
        print("="*50)
        print(report)
        print("="*50)

        return analysis_results

    def run_advanced_factor_analysis_with_ashare(self, symbol: str, days: int = 180) -> Dict[str, Any]:
        """
        使用 Ashare 数据运行高级因子分析

        Args:
            symbol: 股票代码
            days: 回溯天数，默认 180 天

        Returns:
            dict: 分析结果
        """
        print(f"🔍 开始使用 Ashare 数据运行高级因子分析...")
        print(f"🎯 股票: {symbol}, 回溯天数: {days}")

        # 准备输入数据
        data, start_date, end_date, symbols = self.prepare_qlib_input_data(symbol, days)

        if data.empty:
            print("❌ 输入数据为空，无法执行分析")
            return {}

        print(f"📊 准备因子分析数据，样本数: {len(data)}")

        # 运行高级因子分析
        factor_results = self.qlib_enhancer.advanced_factor_analysis(
            data=data,
            instruments=symbols,
            start_date=start_date,
            end_date=end_date
        )

        # 打印因子分析摘要
        if 'factors' in factor_results:
            factors_df = factor_results['factors']
            print(f"\n📈 生成因子数量: {len(factors_df.columns)}")
            print(f"📊 样本数量: {len(factors_df)}")
            if not factors_df.empty and len(factors_df.columns) > 0:
                print(f"📊 首个因子统计: 均值={factors_df.iloc[:, 0].mean():.4f}, "
                      f"标准差={factors_df.iloc[:, 0].std():.4f}, "
                      f"最小值={factors_df.iloc[:, 0].min():.4f}, "
                      f"最大值={factors_df.iloc[:, 0].max():.4f}")

        return factor_results

    def run_smart_portfolio_optimization_with_ashare(self, symbols: List[str], days: int = 180) -> Dict[str, Any]:
        """
        使用 Ashare 数据运行智能投资组合优化

        Args:
            symbols: 股票代码列表
            days: 回溯天数，默认 180 天

        Returns:
            dict: 优化结果
        """
        print(f"⚖️  开始使用 Ashare 数据运行智能投资组合优化...")
        print(f"🎯 股票列表: {symbols}, 回溯天数: {days}")

        # 获取所有股票的数据
        all_data = []
        for symbol in symbols:
            data = self.fetch_ashare_data_for_qlib(symbol, days)
            if data is not None and not data.empty:
                # 保留 close 价格用于计算收益率
                data = data[['close']].rename(columns={'close': symbol})
                all_data.append(data)

        if not all_data:
            print("❌ 无法获取任何股票数据，无法执行优化")
            return {}

        # 合并所有数据
        combined_data = pd.concat(all_data, axis=1, join='outer')
        combined_data = combined_data.fillna(method='ffill').fillna(0)

        # 计算收益率矩阵
        returns_data = combined_data.pct_change().dropna()

        if returns_data.empty or returns_data.shape[1] == 0:
            print("❌ 收益率数据为空，无法执行优化")
            return {}

        print(f"📊 收益率数据形状: {returns_data.shape}")

        # 执行投资组合优化
        optimization_results = self.qlib_enhancer.smart_portfolio_optimization(
            returns_data=returns_data
        )

        # 输出优化权重
        weights = optimization_results.get('optimal_weights', {})
        print(f"\n💰 优化权重:")
        for asset, weight in weights.items():
            print(f"  {asset}: {weight:.4f}")

        return optimization_results

    def run_adaptive_signal_generation_with_ashare(self, symbol: str, days: int = 180) -> Dict[str, Any]:
        """
        使用 Ashare 数据运行自适应信号生成

        Args:
            symbol: 股票代码
            days: 回溯天数，默认 180 天

        Returns:
            dict: 信号生成结果
        """
        print(f"🎯 开始使用 Ashare 数据运行自适应信号生成...")
        print(f"🎯 股票: {symbol}, 回溯天数: {days}")

        # 获取数据
        data = self.fetch_ashare_data_for_qlib(symbol, days)

        if data is None or data.empty:
            print("❌ 数据获取失败，无法生成信号")
            return {}

        print(f"📊 准备信号生成数据，样本数: {len(data)}")

        # 生成自适应信号
        signal_results = self.qlib_enhancer.adaptive_signal_generation(
            data=data
        )

        # 统计各种信号的数量
        for signal_type, signals in signal_results.items():
            if hasattr(signals, 'shape') and len(signals) > 0:
                active_signals = len(signals[signals != 0])
                print(f"📊 {signal_type}: 总数={len(signals)}, 活跃信号={active_signals}")

        return signal_results

    def run_all_analysis_with_ashare(self, symbol: str, days: int = 180) -> Dict[str, Any]:
        """
        运行所有类型的分析

        Args:
            symbol: 股票代码
            days: 回溯天数，默认 180 天

        Returns:
            dict: 所有分析结果
        """
        print(f"🌟 开始对 {symbol} 运行完整的 Ashare-Qlib 综合分析...")
        print(f"📊 使用过去 {days} 天的数据进行分析")

        results = {}

        # 1. 综合性分析
        print("\n" + "="*30)
        print("1. 综合性分析")
        print("="*30)
        results['comprehensive'] = self.run_comprehensive_qlib_analysis_with_ashare(symbol, days)

        # 2. 高级因子分析
        print("\n" + "="*30)
        print("2. 高级因子分析")
        print("="*30)
        results['factor_analysis'] = self.run_advanced_factor_analysis_with_ashare(symbol, days)

        # 3. 自适应信号生成
        print("\n" + "="*30)
        print("3. 自适应信号生成")
        print("="*30)
        results['signal_generation'] = self.run_adaptive_signal_generation_with_ashare(symbol, days)

        print(f"\n🎉 {symbol} 的完整 Ashare-Qlib 分析已完成！")

        return results


# 示例使用
if __name__ == "__main__":
    print("🧪 测试 Ashare-Qlib 集成模块...")

    # 创建集成实例
    integration = AshareQlibIntegration()

    # 测试单只股票
    symbol = "sh600023"  # 华能国际
    print(f"\n📈 测试股票: {symbol}")

    # 执行完整分析
    results = integration.run_all_analysis_with_ashare(symbol, days=180)

    print("\n" + "="*50)
    print("Ashare-Qlib 集成模块测试完成")
    print("="*50)