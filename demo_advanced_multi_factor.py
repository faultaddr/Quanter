#!/usr/bin/env python3
"""
多因子策略使用指南
演示如何使用100+技术指标进行量化分析
"""

import pandas as pd
import numpy as np
from quant_trade_a_share.data.data_fetcher import DataFetcher
from quant_trade_a_share.utils.eastmoney_data_fetcher import EastMoneyDataFetcher
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

def calculate_technical_factors(df):
    """
    为DataFrame计算所有技术因子
    """
    if df is None or df.empty:
        return df

    # 复制数据框
    data = df.copy()

    # 基础指标
    data['returns'] = data['close'].pct_change()
    data['log_return'] = np.log(data['close'] / data['close'].shift(1))

    # 1. 移动平均线系列
    for period in [5, 10, 20, 30, 50, 100]:
        data[f'ma_{period}'] = data['close'].rolling(window=period).mean()
        data[f'ema_{period}'] = data['close'].ewm(span=period).mean()
        data[f'price_to_ma_{period}'] = (data['close'] - data[f'ma_{period}']) / data[f'ma_{period}']

    # 2. RSI系列
    for period in [7, 14, 21, 30]:
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        data[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # 3. MACD系列
    for fast, slow, signal_period in [(12, 26, 9), (10, 20, 5), (5, 15, 3)]:
        exp1_fast = data['close'].ewm(span=fast).mean()
        exp1_slow = data['close'].ewm(span=slow).mean()
        data[f'macd_line_{fast}_{slow}'] = exp1_fast - exp1_slow
        data[f'macd_signal_{fast}_{slow}_{signal_period}'] = data[f'macd_line_{fast}_{slow}'].ewm(span=signal_period).mean()
        data[f'macd_histogram_{fast}_{slow}_{signal_period}'] = data[f'macd_line_{fast}_{slow}'] - data[f'macd_signal_{fast}_{slow}_{signal_period}']

    # 4. 布林带系列
    for period in [10, 20, 50]:
        for std_dev in [1, 2, 2.5]:
            bb_middle = data['close'].rolling(window=period).mean()
            bb_std = data['close'].rolling(window=period).std()
            data[f'bb_upper_{period}_{std_dev}'] = bb_middle + (bb_std * std_dev)
            data[f'bb_lower_{period}_{std_dev}'] = bb_middle - (bb_std * std_dev)
            data[f'bb_bandwidth_{period}_{std_dev}'] = (data[f'bb_upper_{period}_{std_dev}'] - data[f'bb_lower_{period}_{std_dev}']) / bb_middle

    # 5. 波动率指标
    for period in [10, 20, 30]:
        data[f'volatility_{period}'] = data['returns'].rolling(window=period).std()
        data[f'high_low_range_{period}'] = (data['high'] - data['low']).rolling(window=period).mean()
        data[f'realized_vol_{period}'] = data['log_return'].rolling(window=period).std() * np.sqrt(252)

    # 6. 成交量指标
    for period in [5, 10, 20, 30]:
        data[f'volume_ma_{period}'] = data['volume'].rolling(window=period).mean()
        data[f'volume_ratio_{period}'] = data['volume'] / data[f'volume_ma_{period}']
        data[f'volume_std_{period}'] = data['volume'].rolling(window=period).std()

    # 7. 价格形态指标
    data['upper_shadow'] = data['high'] - np.maximum(data['open'], data['close'])
    data['lower_shadow'] = np.minimum(data['open'], data['close']) - data['low']
    data['body_size'] = abs(data['close'] - data['open'])
    data['candle_range'] = data['high'] - data['low']
    data['body_to_range'] = data['body_size'] / data['candle_range']
    data['upper_shadow_to_range'] = data['upper_shadow'] / data['candle_range']
    data['lower_shadow_to_range'] = data['lower_shadow'] / data['candle_range']

    # 8. 趋势指标
    data['sma_trend_20'] = np.where(data['close'] > data['ma_20'], 1, -1)
    data['sma_trend_50'] = np.where(data['close'] > data['ma_50'], 1, -1)
    data['long_term_trend'] = np.where(data['ma_20'] > data['ma_50'], 1, -1)

    # 9. 动量指标
    for period in [5, 10, 20, 30]:
        data[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1
        data[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / data['close'].shift(period)) * 100

    # 10. 成交量价格趋势
    for period in [10, 20, 30]:
        data[f'vpt_{period}'] = (data['volume'] * (data['close'] - data['close'].shift(1)) / data['close'].shift(1)).cumsum().rolling(window=period).mean()

    # 11. 威廉指标
    for period in [14, 20, 25]:
        highest_high = data['high'].rolling(window=period).max()
        lowest_low = data['low'].rolling(window=period).min()
        data[f'williams_r_{period}'] = (highest_high - data['close']) / (highest_high - lowest_low) * -100

    # 12. 随机指标
    for period in [14, 20, 25]:
        lowest_low = data['low'].rolling(window=period).min()
        highest_high = data['high'].rolling(window=period).max()
        data[f'stoch_k_{period}'] = (data['close'] - lowest_low) / (highest_high - lowest_low) * 100
        data[f'stoch_d_{period}'] = data[f'stoch_k_{period}'].rolling(window=3).mean()

    # 13. 平均真实波幅
    data['tr1'] = abs(data['high'] - data['low'])
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = pd.concat([data['tr1'], data['tr2'], data['tr3']], axis=1).max(axis=1)
    for period in [14, 20, 25]:
        data[f'atr_{period}'] = data['true_range'].rolling(window=period).mean()

    print(f"📈 已计算 {len(data.columns)} 个技术因子")
    return data

def build_multi_factor_strategy(data):
    """
    构建多因子策略
    """
    if data.empty:
        print("❌ 数据为空，无法构建策略")
        return None

    print(f"📈 原始数据形状: {data.shape}")
    print(f"📊 可用列: {list(data.columns)}")

    # 计算技术因子
    df_with_factors = calculate_technical_factors(data)

    if df_with_factors is not None:
        # 创建简单的多因子信号
        df = df_with_factors.copy()

        # RSI信号 (反转信号)
        df['rsi_signal'] = np.where(df['rsi_14'] < 30, 1,  # 超卖买入
                                   np.where(df['rsi_14'] > 70, -1, 0))  # 超买卖出

        # 均线信号
        df['ma_signal'] = np.where(df['close'] > df['ma_20'], 1, -1)

        # 布林带信号
        df['bb_signal'] = np.where(df['close'] < df['bb_lower_20_2'], 1,  # 突破下轨买入
                                  np.where(df['close'] > df['bb_upper_20_2'], -1, 0))  # 突破上轨卖出

        # MACD信号
        df['macd_signal'] = np.where(df['macd_line_12_26'] > df['macd_signal_12_26_9'], 1, -1)

        # 综合信号 (加权平均)
        weights = {'rsi': 0.25, 'ma': 0.25, 'bb': 0.25, 'macd': 0.25}
        df['composite_signal'] = (weights['rsi'] * df['rsi_signal'] +
                                 weights['ma'] * df['ma_signal'] +
                                 weights['bb'] * df['bb_signal'] +
                                 weights['macd'] * df['macd_signal'])

        # 计算基于信号的预测收益
        df['predicted_returns'] = df['composite_signal'].shift(1) * df['returns']

        print(f"📈 策略构建完成，包含 {len(df.columns)} 个因子")
        return df

    return None

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

    # 计算因子IC (Information Coefficient)
    ic = clean_data[[factor_name, 'returns']].corr().iloc[0, 1]
    print(f"📊 {factor_name} 因子IC: {ic:.4f}")

    # 按因子值分组
    clean_data = clean_data.dropna()
    if len(clean_data) > 10:
        clean_data['factor_decile'] = pd.qcut(clean_data[factor_name], min(10, len(clean_data)//10), labels=False, duplicates='drop')

        # 计算各分位的平均收益
        factor_returns = clean_data.groupby('factor_decile')['returns'].mean()
        print(f"📈 {factor_name} 分位收益分析:")
        print(factor_returns)

        # 信息比率
        ir = factor_returns.mean() / factor_returns.std() if factor_returns.std() != 0 else 0
        print(f"📊 信息比率 (IR): {ir:.4f}")

    return ic

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
            dates = pd.date_range('2025-01-01', '2025-12-31', freq='D')
            n = len(dates)
            prices = 5 + np.cumsum(np.random.randn(n) * 0.02)  # 随机游走

            data = pd.DataFrame({
                'open': prices * (1 + np.random.randn(n) * 0.005),
                'high': prices * (1 + abs(np.random.randn(n)) * 0.01),
                'low': prices * (1 - abs(np.random.randn(n)) * 0.01),
                'close': prices,
                'volume': np.random.randint(1000000, 10000000, n)
            }, index=dates)

        # 3. 构建多因子策略
        print("\n🏗️  构建多因子策略...")
        enriched_data = build_multi_factor_strategy(data)

        if enriched_data is not None:
            print(f"✅ 策略构建完成，当前包含 {len(enriched_data.columns)} 个因子")

            # 4. 分析因子性能
            print("\n📊 分析因子表现...")
            ic_score = analyze_factor_performance(enriched_data, 'composite_signal')

            # 5. 计算策略收益
            if 'predicted_returns' in enriched_data.columns:
                clean_returns = enriched_data[['predicted_returns', 'returns']].dropna()

                if not clean_returns.empty:
                    strategy_cumret = (1 + clean_returns['predicted_returns']).cumprod()
                    benchmark_cumret = (1 + clean_returns['returns']).cumprod()

                    print(f"\n📈 策略累计收益: {(strategy_cumret.iloc[-1] - 1) * 100:.2f}%")
                    print(f"📈 买入持有收益: {(benchmark_cumret.iloc[-1] - 1) * 100:.2f}%")

                    # 计算年化收益和夏普比率
                    total_return = strategy_cumret.iloc[-1] - 1
                    benchmark_return = benchmark_cumret.iloc[-1] - 1
                    n_years = len(clean_returns) / 252  # 假设252个交易日一年

                    annual_return = (strategy_cumret.iloc[-1]) ** (1/n_years) - 1
                    benchmark_annual = (benchmark_cumret.iloc[-1]) ** (1/n_years) - 1

                    excess_return = annual_return - benchmark_annual
                    volatility = clean_returns['predicted_returns'].std() * np.sqrt(252)

                    sharpe_ratio = excess_return / volatility if volatility != 0 else 0

                    print(f"📊 策略年化收益: {annual_return*100:.2f}%")
                    print(f"📊 基准年化收益: {benchmark_annual*100:.2f}%")
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
    print("2. 因子工程: 计算多种技术指标（100+因子）")
    print("3. 策略构建: 结合多个因子生成综合信号")
    print("4. 风险控制: 监控因子暴露和相关性")
    print("5. 绩效评估: 计算收益、信息比率、夏普比率等指标")
    print("="*80)