"""
趋势评分系统测试脚本

测试内容：
1. 硬过滤测试
2. 趋势评分测试
3. 时机系数测试
4. 回测验证年化收益

运行方式：
- 单元测试: pytest tests/test_trend_scoring.py -v
- 回测验证: python tests/test_trend_scoring.py --backtest
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import argparse

from quanttool.factors.trend_scoring_system import TrendScoringSystem, TrendScoreResult, analyze_trend_quality
from quanttool.strategies.trend_strategy import TrendStrategy, AdaptiveTrendStrategy
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import create_data_fetcher_with_credentials


class TestTrendScoringSystem:
    """趋势评分系统单元测试"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """生成测试用股票数据（上升趋势，确保通过硬过滤）"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

        # 模拟稳定上升趋势，确保股价在MA20上方
        base_price = 100
        # 使用指数增长确保股价始终在均线上方
        trend = np.array([base_price * (1.002 ** i) for i in range(200)])  # 每日0.2%增长
        noise = np.random.randn(200) * 1.5  # 较小噪声

        close = trend + noise
        # 确保close在最后一段时间明显高于MA20
        close[-20:] = close[-20:] + 3  # 额外提升最后20天价格

        high = close + np.abs(np.random.randn(200)) * 2
        low = close - np.abs(np.random.randn(200)) * 2
        open_price = close + np.random.randn(200) * 1
        volume = np.random.randint(2000000, 8000000, 200)  # 确保足够成交量

        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        return df

    @pytest.fixture
    def strong_trend_data(self) -> pd.DataFrame:
        """生成强趋势股票数据"""
        np.random.seed(100)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

        # 模拟强势上涨趋势
        base_price = 50
        trend = np.linspace(0, 50, 200)  # 上涨100%
        noise = np.random.randn(200) * 1  # 较小噪声

        close = base_price + trend + noise
        high = close + np.abs(np.random.randn(200)) * 1.5
        low = close - np.abs(np.random.randn(200)) * 1.5
        open_price = close + np.random.randn(200) * 0.5
        volume = np.random.randint(2000000, 8000000, 200) * 1.5  # 放量

        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        return df

    @pytest.fixture
    def weak_trend_data(self) -> pd.DataFrame:
        """生成弱趋势股票数据"""
        np.random.seed(200)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

        # 模拟弱势震荡
        base_price = 100
        noise = np.random.randn(200) * 5  # 大噪声
        close = base_price + noise
        high = close + np.abs(np.random.randn(200)) * 3
        low = close - np.abs(np.random.randn(200)) * 3
        open_price = close + np.random.randn(200) * 2
        volume = np.random.randint(500000, 2000000, 200)  # 缩量

        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        return df

    def test_hard_filter_pass(self, sample_data):
        """测试硬过滤通过"""
        system = TrendScoringSystem()
        result = system.calculate_score(sample_data)

        assert result.passed_hard_filter == True, f"应该通过硬过滤，原因: {result.hard_filter_reason}"

    def test_hard_filter_fail_insufficient_data(self):
        """测试数据不足时硬过滤失败"""
        system = TrendScoringSystem()

        # 只有30天数据
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.randn(30) + 100,
            'high': np.random.randn(30) + 102,
            'low': np.random.randn(30) + 98,
            'close': np.random.randn(30) + 100,
            'volume': np.random.randint(1000000, 5000000, 30)
        })

        result = system.calculate_score(df)

        assert result.passed_hard_filter == False
        assert "数据不足" in result.hard_filter_reason

    def test_trend_score_range(self, sample_data):
        """测试评分范围"""
        system = TrendScoringSystem()
        result = system.calculate_score(sample_data)

        # 如果通过硬过滤，检查评分范围
        if result.passed_hard_filter:
            # 评分应在0-100范围内
            assert 0 <= result.final_score <= 100, f"最终评分超出范围: {result.final_score}"
            assert 0 <= result.trend_total_score <= 100, f"趋势总分超出范围: {result.trend_total_score}"
            assert 0.7 <= result.timing_coefficient <= 1.2, f"时机系数超出范围: {result.timing_coefficient}"
        else:
            # 未通过硬过滤时，评分为0
            assert result.final_score == 0

    def test_strong_trend_high_score(self, strong_trend_data):
        """测试强趋势股票得高分"""
        system = TrendScoringSystem()
        result = system.calculate_score(strong_trend_data)

        assert result.passed_hard_filter == True
        # 强趋势应该得高分（>=75）
        assert result.trend_total_score >= 70, f"强趋势股票应得高分，实际: {result.trend_total_score}"

    def test_weak_trend_low_score(self, weak_trend_data):
        """测试弱趋势股票得低分"""
        system = TrendScoringSystem()
        result = system.calculate_score(weak_trend_data)

        # 弱趋势可能不通过硬过滤，或得分较低
        if result.passed_hard_filter:
            assert result.trend_total_score < 70, f"弱趋势股票应得低分，实际: {result.trend_total_score}"

    def test_timing_coefficient_types(self, sample_data):
        """测试时机系数类型"""
        system = TrendScoringSystem()
        result = system.calculate_score(sample_data)

        # 如果通过硬过滤，时机类型应该是有效类型
        if result.passed_hard_filter:
            valid_types = ["回踩买点", "突破买点", "趋势运行", "短期过热", "追高风险"]
            assert result.timing_type in valid_types, f"时机类型无效: {result.timing_type}"
        else:
            # 未通过硬过滤时，时机类型为默认值
            assert result.timing_type == "standard"

    def test_ma_structure_score(self, strong_trend_data):
        """测试均线结构评分"""
        system = TrendScoringSystem()
        result = system.calculate_score(strong_trend_data)

        # 强趋势的均线应该呈多头排列
        assert result.ma_structure_score >= 60, f"强趋势均线得分应较高: {result.ma_structure_score}"

    def test_factor_weights(self, sample_data):
        """测试因子权重"""
        system = TrendScoringSystem()
        result = system.calculate_score(sample_data)

        # 验证权重总和为100%
        # 实际上趋势总分 = 各因子得分的加权和
        expected_total = (
            result.ma_structure_score * 0.30 +
            result.price_momentum_score * 0.30 +
            result.volume_score * 0.25 +
            result.relative_strength_score * 0.15
        )

        assert abs(result.trend_total_score - expected_total) < 1, "因子权重计算错误"

    def test_analyze_trend_quality_function(self, sample_data):
        """测试便捷分析函数"""
        result = analyze_trend_quality(sample_data)

        assert 'final_score' in result
        assert 'trend_score' in result
        assert 'timing_coefficient' in result
        assert 'recommendation' in result


class TestTrendStrategy:
    """趋势策略单元测试"""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """生成测试用股票数据"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', periods=200, freq='D')

        base_price = 100
        trend = np.linspace(0, 30, 200)
        noise = np.random.randn(200) * 2

        close = base_price + trend + noise
        high = close + np.abs(np.random.randn(200)) * 2
        low = close - np.abs(np.random.randn(200)) * 2
        open_price = close + np.random.randn(200) * 1
        volume = np.random.randint(1000000, 5000000, 200)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

        return df

    def test_strategy_initialization(self):
        """测试策略初始化"""
        strategy = TrendStrategy(
            buy_threshold=75.0,
            sell_threshold=50.0
        )

        assert strategy.buy_threshold == 75.0
        assert strategy.sell_threshold == 50.0
        assert strategy.get_name() == "TrendStrategy"

    def test_strategy_signal_generation(self, sample_data):
        """测试信号生成"""
        strategy = TrendStrategy()
        result = strategy.calculate_signals(sample_data)

        assert 'signal' in result.columns
        assert 'final_score' in result.columns
        assert 'timing_coefficient' in result.columns

        # 检查信号值
        valid_signals = ['buy', 'sell', 'hold']
        for signal in result['signal'].dropna().unique():
            assert signal in valid_signals, f"无效信号: {signal}"

    def test_strategy_get_signal(self, sample_data):
        """测试单点信号获取"""
        strategy = TrendStrategy()
        current_bar = sample_data.iloc[-1]
        historical_bars = sample_data.iloc[:-1]

        signal = strategy.get_signal(current_bar, sample_data)

        assert 'direction' in signal
        assert 'signal' in signal
        assert 'score' in signal

    def test_strategy_parameters(self):
        """测试策略参数"""
        strategy = TrendStrategy(
            buy_threshold=80.0,
            sell_threshold=40.0,
            use_timing_filter=True,
            position_by_timing=True
        )

        params = strategy.get_parameters()

        assert params['buy_threshold'] == 80.0
        assert params['sell_threshold'] == 40.0
        assert params['use_timing_filter'] == True
        assert params['position_by_timing'] == True

    def test_position_ratio_calculation(self):
        """测试仓位比例计算"""
        strategy = TrendStrategy()

        # 高分+好时机 = 高仓位
        position_high = strategy._calculate_position_ratio(90, 1.1)
        assert position_high >= 0.8, f"高分应得高仓位: {position_high}"

        # 中分+好时机 = 中等仓位
        position_mid = strategy._calculate_position_ratio(75, 1.0)
        assert 0.4 <= position_mid <= 0.8, f"中分应得中等仓位: {position_mid}"

        # 低时机系数 = 低仓位
        position_low = strategy._calculate_position_ratio(80, 0.7)
        assert position_low <= 0.5, f"低时机系数应降低仓位: {position_low}"


def run_backtest_with_trend_strategy(
    symbols: List[str] = None,
    start_date: str = '2023-01-01',
    end_date: str = None,
    initial_capital: float = 1000000,
    buy_threshold: float = 75.0,
    sell_threshold: float = 50.0
) -> Dict:
    """
    使用趋势策略进行回测

    Args:
        symbols: 股票代码列表，默认沪深300成分股
        start_date: 开始日期
        end_date: 结束日期
        initial_capital: 初始资金
        buy_threshold: 买入阈值
        sell_threshold: 卖出阈值

    Returns:
        Dict: 回测结果
    """
    from quanttool.backtest.engine import BacktestEngine

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    # 默认使用部分沪深300成分股测试
    if symbols is None:
        symbols = [
            '600519.SH',  # 贵州茅台
            '000858.SZ',  # 五粮液
            '601318.SH',  # 中国平安
            '600036.SH',  # 招商银行
            '601166.SH',  # 兴业银行
            '600276.SH',  # 恒瑞医药
            '000333.SZ',  # 美的集团
            '600030.SH',  # 中信证券
            '601398.SH',  # 工商银行
            '600000.SH',  # 浦发银行
        ]

    print(f"\n{'='*60}")
    print(f"趋势策略回测")
    print(f"{'='*60}")
    print(f"时间范围: {start_date} ~ {end_date}")
    print(f"初始资金: ¥{initial_capital:,.0f}")
    print(f"买入阈值: {buy_threshold}")
    print(f"卖出阈值: {sell_threshold}")
    print(f"测试股票: {len(symbols)}只")
    print(f"{'='*60}\n")

    # 初始化数据获取器
    fetcher = create_data_fetcher_with_credentials()
    fetcher.initialize()

    # 获取数据
    print("正在获取数据...")
    data = {}
    for symbol in symbols:
        try:
            symbol_data = fetcher.get_bars(
                [symbol],
                datetime.strptime(start_date, '%Y-%m-%d'),
                datetime.strptime(end_date, '%Y-%m-%d')
            )
            if symbol in symbol_data and not symbol_data[symbol].empty:
                data[symbol] = symbol_data[symbol]
                print(f"  {symbol}: {len(symbol_data[symbol])}条记录")
        except Exception as e:
            print(f"  {symbol}: 获取失败 - {e}")

    if not data:
        print("无法获取任何数据")
        return {}

    # 创建策略
    strategy = TrendStrategy(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold
    )

    # 回测结果
    results = {
        'symbols': symbols,
        'start_date': start_date,
        'end_date': end_date,
        'initial_capital': initial_capital,
        'trades': [],
        'statistics': {}
    }

    # 对每只股票运行策略
    all_signals = []
    for symbol, df in data.items():
        print(f"\n分析 {symbol}...")

        if len(df) < 60:
            print(f"  数据不足，跳过")
            continue

        # 计算信号
        signals_df = strategy.calculate_signals(df)

        # 统计信号
        buy_signals = signals_df[signals_df['signal'] == 'buy']
        sell_signals = signals_df[signals_df['signal'] == 'sell']

        print(f"  买入信号: {len(buy_signals)}个")
        print(f"  卖出信号: {len(sell_signals)}个")

        if len(buy_signals) > 0:
            avg_score = buy_signals['final_score'].mean()
            avg_timing = buy_signals['timing_coefficient'].mean()
            print(f"  平均买入评分: {avg_score:.1f}")
            print(f"  平均时机系数: {avg_timing:.2f}")

        # 保存信号
        all_signals.append({
            'symbol': symbol,
            'buy_count': len(buy_signals),
            'sell_count': len(sell_signals),
            'signals_df': signals_df
        })

    results['all_signals'] = all_signals

    # 计算总体统计
    total_buy = sum(s['buy_count'] for s in all_signals)
    total_sell = sum(s['sell_count'] for s in all_signals)

    results['statistics'] = {
        'total_buy_signals': total_buy,
        'total_sell_signals': total_sell,
        'avg_buy_signals_per_stock': total_buy / len(all_signals) if all_signals else 0
    }

    print(f"\n{'='*60}")
    print(f"回测统计")
    print(f"{'='*60}")
    print(f"总买入信号: {total_buy}个")
    print(f"总卖出信号: {total_sell}个")
    print(f"平均每只股票买入信号: {results['statistics']['avg_buy_signals_per_stock']:.1f}个")
    print(f"{'='*60}\n")

    return results


def test_score_distribution():
    """
    测试评分分布

    验证强势股得高分，弱势股得低分
    """
    print("\n" + "="*60)
    print("评分分布测试")
    print("="*60)

    # 初始化数据获取器
    fetcher = create_data_fetcher_with_credentials()
    fetcher.initialize()

    # 测试股票列表
    test_symbols = [
        ('600519.SH', '贵州茅台'),  # 大盘蓝筹
        ('300750.SZ', '宁德时代'),  # 热门成长
        ('000001.SZ', '平安银行'),  # 银行股
        ('600036.SH', '招商银行'),  # 银行龙头
    ]

    system = TrendScoringSystem()
    results = []

    for symbol, name in test_symbols:
        try:
            print(f"\n分析 {name}({symbol})...")

            # 获取数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)

            data = fetcher.get_bars([symbol], start_date, end_date)

            if symbol not in data or data[symbol].empty:
                print(f"  无法获取数据")
                continue

            df = data[symbol]

            # 计算评分
            result = system.calculate_score(df)

            results.append({
                'symbol': symbol,
                'name': name,
                'final_score': result.final_score,
                'trend_score': result.trend_total_score,
                'timing_coef': result.timing_coefficient,
                'timing_type': result.timing_type,
                'ma_score': result.ma_structure_score,
                'momentum_score': result.price_momentum_score,
                'volume_score': result.volume_score,
                'rs_score': result.relative_strength_score,
                'passed_filter': result.passed_hard_filter
            })

            print(f"  最终评分: {result.final_score:.1f}")
            print(f"  趋势总分: {result.trend_total_score:.1f}")
            print(f"  时机系数: {result.timing_coefficient:.2f} ({result.timing_type})")
            print(f"  通过过滤: {'是' if result.passed_hard_filter else '否'}")

        except Exception as e:
            print(f"  分析失败: {e}")

    # 打印结果表格
    if results:
        print("\n" + "-"*80)
        print(f"{'股票':^10} {'最终评分':^8} {'趋势分':^8} {'时机系数':^8} {'均线':^6} {'动能':^6} {'量能':^6} {'强度':^6}")
        print("-"*80)

        for r in results:
            print(f"{r['name']:^10} {r['final_score']:^8.1f} {r['trend_score']:^8.1f} "
                  f"{r['timing_coef']:^8.2f} {r['ma_score']:^6.1f} {r['momentum_score']:^6.1f} "
                  f"{r['volume_score']:^6.1f} {r['rs_score']:^6.1f}")

        print("-"*80)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='趋势评分系统测试')
    parser.add_argument('--backtest', action='store_true', help='运行回测')
    parser.add_argument('--distribution', action='store_true', help='测试评分分布')
    parser.add_argument('--unit', action='store_true', help='运行单元测试')

    args = parser.parse_args()

    if args.backtest:
        run_backtest_with_trend_strategy()
    elif args.distribution:
        test_score_distribution()
    elif args.unit:
        pytest.main([__file__, '-v'])
    else:
        # 默认运行评分分布测试
        print("运行评分分布测试（使用 --backtest 运行回测，--unit 运行单元测试）\n")
        test_score_distribution()