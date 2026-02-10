#!/usr/bin/env python3
"""
多因子策略使用指南
演示如何使用100+技术指标进行量化分析
"""

import pandas as pd
import numpy as np
from quant_trade_a_share.data.data_fetcher import DataFetcher
import matplotlib.pyplot as plt
import seaborn as sns

def get_available_factors():
    """
    列出所有可用的技术指标/因子
    """
    factors = [
        # 价格相关因子
        'ma_5', 'ma_10', 'ma_20', 'ma_30',  # 移动平均线
        'rsi',  # 相对强弱指标
        'momentum',  # 动量指标
        'log_return',  # 对数收益率

        # MACD相关因子
        'macd', 'signal', 'histogram',

        # 布林带因子
        'bb_middle', 'bb_upper', 'bb_lower',

        # 波动率因子
        'volatility', 'hl_volatility', 'atr',

        # 成交量因子
        'volume_sma', 'volume_ratio', 'vpt',

        # 价格位置因子
        'hl_ratio', 'price_position',

        # 随机指标
        'williams_r', 'stoch_k', 'stoch_d',

        # 趋势因子
        'trend', 'roc',

        # 基础价格因子
        'open', 'high', 'low', 'close', 'volume'
    ]

    print("📊 可用技术指标列表:")
    for i, factor in enumerate(factors, 1):
        print(f"{i:2d}. {factor}")

    print(f"\n总计: {len(factors)} 个技术指标")
    return factors

def build_multi_factor_strategy(data):
    """
    构建多因子策略
    """
    # 检查数据是否包含技术指标
    if data.empty:
        print("❌ 数据为空，无法构建策略")
        return None

    print(f"📈 原始数据形状: {data.shape}")
    print(f"📊 可用列: {list(data.columns)}")

    # 计算额外的高级因子
    df = data.copy()

    # 时间序列因子
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()

    # 多时间周期移动平均
    for period in [5, 10, 20, 30, 50, 100]:
        df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'price_to_ma_{period}'] = df['close'] / df[f'ma_{period}']

    # 多时间周期RSI
    for period in [7, 14, 21]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # 成交量相关因子
    for period in [5, 10, 20]:
        df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']

    # 价量关系因子
    df['price_volume_trend'] = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum()

    # 高级波动率因子
    df['realized_volatility'] = df['log_return'].rolling(10).std() * np.sqrt(252)

    # 形态识别因子
    df['doji'] = np.where(abs(df['close'] - df['open']) / df['open'] < 0.005, 1, 0)  # 十字星
    df['hammer'] = np.where((df['high'] - df['low']) > 3 * abs(df['close'] - df['open']) &
                           (df['close'] == df['high']) | (df['open'] == df['high']), 1, 0)  # 锤头线

    print(f"📈 添加因子后数据形状: {df.shape}")

    # 创建简单的多因子信号
    df['factor_score'] = 0.0

    # RSI信号 (反转信号)
    df['rsi_signal'] = np.where(df['rsi'] < 30, 1,  # 超卖买入
                               np.where(df['rsi'] > 70, -1, 0))  # 超买卖出

    # 均线交叉信号
    df['ma_signal'] = np.where(df['close'] > df['ma_20'], 1, -1)

    # 价格突破信号
    df['breakout_signal'] = np.where(df['close'] > df['bb_upper'], 1,
                                    np.where(df['close'] < df['bb_lower'], -1, 0))

    # 综合信号
    df['composite_signal'] = (df['rsi_signal'] * 0.3 +
                             df['ma_signal'] * 0.4 +
                             df['breakout_signal'] * 0.3)

    # 根据信号计算收益预测
    df['predicted_returns'] = df['composite_signal'].shift(1) * df['returns']

    return df

def analyze_factor_performance(data, factor_name='composite_signal'):
    """
    分析因子表现
    """
    if data is None or factor_name not in data.columns:
        print(f"❌ 因子 {factor_name} 不存在或数据为空")
        return

    # 过滤掉NaN值
    clean_data = data.dropna(subset=[factor_name, 'returns'])

    if clean_data.empty:
        print(f"❌ 没有有效的数据用于分析 {factor_name}")
        return

    # 按因子值分组
    clean_data['factor_decile'] = pd.qcut(clean_data[factor_name], 10, labels=False, duplicates='drop')

    # 计算各分位的平均收益
    factor_returns = clean_data.groupby('factor_decile')['returns'].mean()

    print(f"📊 {factor_name} 因子分位分析:")
    print(factor_returns)

    # 信息比率
    ir = factor_returns.mean() / factor_returns.std() if factor_returns.std() != 0 else 0
    print(f"📈 信息比率 (IR): {ir:.4f}")

    return factor_returns

def run_multi_factor_analysis():
    """
    运行完整的多因子分析
    """
    print("=" * 80)
    print("🎯 量化交易系统 - 多因子策略演示")
    print("=" * 80)

    # 1. 显示可用因子
    available_factors = get_available_factors()

    # 2. 获取数据
    print("\n🔄 获取股票数据...")
    fetcher = DataFetcher()

    try:
        # 获取样本股票数据
        data = fetcher.fetch('sh600023', '2025-01-01', '2026-01-01', source='ashare')

        if data is None or data.empty:
            print("⚠️  获取数据失败，使用模拟数据...")
            # 创建模拟数据
            dates = pd.date_range('2025-01-01', '2026-01-01', freq='D')
            n = len(dates)
            prices = 5 + np.cumsum(np.random.randn(n) * 0.02)  # 随机游走

            data = pd.DataFrame({
                'open': prices * (1 + np.random.randn(n) * 0.005),
                'high': prices * (1 + abs(np.random.randn(n)) * 0.01),
                'low': prices * (1 - abs(np.random.randn(n)) * 0.01),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n)
            }, index=dates)

            # 计算基础技术指标
            data['ma_5'] = data['close'].rolling(5).mean()
            data['ma_10'] = data['close'].rolling(10).mean()
            data['ma_20'] = data['close'].rolling(20).mean()
            data['rsi'] = 50 + np.random.randn(len(data)) * 10  # 模拟RSI
            data['returns'] = data['close'].pct_change()

        # 3. 构建多因子策略
        print("\n🏗️  构建多因子策略...")
        enriched_data = build_multi_factor_strategy(data)

        if enriched_data is not None:
            print(f"✅ 策略构建完成，当前包含 {len(enriched_data.columns)} 个因子")

            # 4. 分析因子性能
            print("\n📊 分析因子表现...")
            analyze_factor_performance(enriched_data, 'composite_signal')

            # 5. 计算策略收益
            if 'predicted_returns' in enriched_data.columns:
                cumulative_return = (1 + enriched_data['predicted_returns']).cumprod()
                buy_hold_return = (1 + enriched_data['returns']).cumprod()

                print(f"\n📈 策略累计收益: {(cumulative_return.iloc[-1] - 1) * 100:.2f}%")
                print(f"📈 买入持有收益: {(buy_hold_return.iloc[-1] - 1) * 100:.2f}%")

                # 计算夏普比率（简化版）
                excess_returns = enriched_data['predicted_returns'] - enriched_data['returns']
                sharpe_ratio = excess_returns.mean() / (excess_returns.std() + 1e-10) * np.sqrt(252) if excess_returns.std() != 0 else 0
                print(f"📊 夏普比率: {sharpe_ratio:.4f}")

    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def create_factor_exposure_matrix():
    """
    创建因子暴露矩阵示例
    """
    print("\n📋 因子暴露矩阵示例:")

    # 模拟几个因子的暴露度
    factors = ['MA_Ratio', 'RSI_Signal', 'Volume_Momentum', 'Volatility_Regime', 'Momentum_Factor']
    stocks = ['SH600000', 'SH600023', 'SZ000001', 'SH600519', 'SZ300770']

    np.random.seed(42)
    exposure_matrix = pd.DataFrame(
        np.random.randn(len(stocks), len(factors)),
        index=stocks,
        columns=factors
    )

    print(exposure_matrix.round(3))

    # 计算因子相关性
    print("\n🔗 因子间相关性:")
    factor_corr = exposure_matrix.corr()
    print(factor_corr.round(3))

    return exposure_matrix

if __name__ == "__main__":
    run_multi_factor_analysis()
    create_factor_exposure_matrix()

    print("\n" + "="*80)
    print("💡 多因子策略使用指南总结:")
    print("="*80)
    print("1. 数据获取: 使用 DataFetcher 获取历史数据")
    print("2. 因子工程: 利用内置技术指标和自定义因子")
    print("3. 策略构建: 结合多个因子生成综合信号")
    print("4. 风险控制: 监控因子暴露和相关性")
    print("5. 绩效评估: 计算收益、风险调整收益等指标")
    print("="*80)