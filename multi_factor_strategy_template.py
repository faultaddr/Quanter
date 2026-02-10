#!/usr/bin/env python3
"""
A-Share 多因子策略模板
用于构建100+因子的量化投资策略
"""

import pandas as pd
import numpy as np
from quant_trade_a_share.data.data_fetcher import DataFetcher
import warnings
warnings.filterwarnings('ignore')

class MultiFactorStrategy:
    """
    多因子策略类
    支持100+技术指标的因子分析与策略构建
    """

    def __init__(self, universe=['sh600023', 'sh600519', 'sz000001', 'sz300770']):
        """
        初始化多因子策略

        Parameters:
        universe: list of str, 股票池
        """
        self.universe = universe
        self.fetcher = DataFetcher()
        self.factor_weights = {}
        self.signals = {}

    def calculate_all_factors(self, data):
        """
        计算所有技术因子
        总计超过100个技术指标
        """
        df = data.copy()

        # 1. 基础价格因子 (5个)
        df['returns'] = df['close'].pct_change()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']
        df['close_open_pct'] = (df['close'] - df['open']) / df['open']
        df['volume_price'] = df['volume'] * df['close']  # 成交额

        # 2. 移动平均线因子 (12个) - 不同周期的SMA和EMA
        for period in [5, 10, 20, 30, 50, 60, 120]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
            df[f'price_to_ema_{period}'] = df['close'] / df[f'ema_{period}']

        # 3. RSI因子 (4个) - 不同期限
        for period in [7, 14, 21, 30]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # 4. MACD因子 (9个) - 不同参数组合
        for fast, slow, signal in [(12, 26, 9), (10, 20, 5), (5, 15, 3)]:
            exp12 = df['close'].ewm(span=fast).mean()
            exp26 = df['close'].ewm(span=slow).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=signal).mean()
            hist = macd_line - signal_line

            df[f'macd_{fast}_{slow}'] = macd_line
            df[f'macd_signal_{fast}_{slow}_{signal}'] = signal_line
            df[f'macd_hist_{fast}_{slow}_{signal}'] = hist

        # 5. 布林带因子 (12个) - 不同期限和标准差
        for period in [10, 20, 50]:
            for std in [1.5, 2.0, 2.5]:
                bb_mid = df['close'].rolling(window=period).mean()
                bb_std = df['close'].rolling(window=period).std()
                bb_up = bb_mid + (bb_std * std)
                bb_low = bb_mid - (bb_std * std)

                df[f'bb_upper_{period}_{std}'] = bb_up
                df[f'bb_lower_{period}_{std}'] = bb_low
                df[f'bb_middle_{period}'] = bb_mid
                df[f'bb_width_{period}_{std}'] = bb_up - bb_low

        # 6. 成交量因子 (8个)
        for period in [5, 10, 20, 30]:
            df[f'vma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'vma_{period}']
            df[f'volume_std_{period}'] = df['volume'].rolling(window=period).std()
            df[f'volume_zscore_{period}'] = (df['volume'] - df[f'vma_{period}']) / df[f'volume_std_{period}']

        # 7. 波动率因子 (6个)
        for period in [10, 20, 30]:
            df[f'vol_{period}'] = df['returns'].rolling(window=period).std()
            df[f'hl_range_{period}'] = (df['high'] - df['low']).rolling(window=period).mean()
            df[f'realized_vol_{period}'] = df['log_return'].rolling(window=period).std() * np.sqrt(252)

        # 8. 动量因子 (8个)
        for period in [5, 10, 20, 30]:
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
            df[f'cmf_{period}'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])

        # 9. 趋势因子 (6个)
        df['trend_20'] = np.where(df['close'] > df['sma_20'], 1, -1)
        df['trend_50'] = np.where(df['close'] > df['sma_50'], 1, -1)
        df['long_term_trend'] = np.where(df['sma_20'] > df['sma_50'], 1, -1)
        df['trend_strength'] = abs(df['close'] - df['sma_20']) / df['sma_20']
        df['trend_consistency'] = df['trend_20'].rolling(10).sum() / 10
        df['trend_acceleration'] = df['close'].diff().diff()

        # 10. 价格形态因子 (6个)
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['body_size'] = abs(df['close'] - df['open'])
        df['candle_range'] = df['high'] - df['low']
        df['body_to_range'] = df['body_size'] / df['candle_range']
        df['shadow_ratio'] = df['upper_shadow'] / df['lower_shadow'].replace(0, 1e-10)

        # 11. 随机指标 (6个)
        for period in [14, 20, 25]:
            llv = df['low'].rolling(window=period).min()
            hhv = df['high'].rolling(window=period).max()
            df[f'stoch_k_{period}'] = (df['close'] - llv) / (hhv - llv) * 100
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(3).mean()

        # 12. 威廉指标 (3个)
        for period in [14, 20, 25]:
            hhv = df['high'].rolling(window=period).max()
            llv = df['low'].rolling(window=period).min()
            df[f'williams_r_{period}'] = (hhv - df['close']) / (hhv - llv) * -100

        # 13. ATR因子 (3个)
        tr1 = abs(df['high'] - df['low'])
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        for period in [14, 20, 25]:
            df[f'atr_{period}'] = true_range.rolling(window=period).mean()

        # 14. VPT因子 (3个)
        for period in [10, 20, 30]:
            df[f'vpt_{period}'] = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum().rolling(window=period).mean()

        # 15. 相关性因子 (4个)
        df['price_vol_corr_20'] = df['close'].rolling(20).corr(df['volume'])
        df['high_low_corr_20'] = df['high'].rolling(20).corr(df['low'])
        df['open_close_corr_20'] = df['open'].rolling(20).corr(df['close'])
        df['returns_auto_corr_5'] = df['returns'].rolling(5).corr(df['returns'].shift(1))

        print(f"✅ 已计算 {len(df.columns)-5} 个技术因子，总计 {len(df.columns)} 个字段")  # 减去原始5个价格字段
        return df

    def generate_signals(self, stock_data):
        """
        基于技术因子生成交易信号
        """
        data = stock_data.copy()

        # 计算各个信号
        signals = pd.DataFrame(index=data.index)

        # 1. RSI信号 (反转策略)
        signals['rsi_signal'] = np.where(data['rsi_14'] < 30, 1,    # 超卖买入
                                       np.where(data['rsi_14'] > 70, -1, 0))  # 超买卖出

        # 2. 均线信号 (趋势策略)
        signals['ma_signal'] = np.where(data['close'] > data['sma_20'], 1, -1)

        # 3. 布林带信号 (均值回归)
        signals['bb_signal'] = np.where(data['close'] < data['bb_lower_20_2.0'], 1,      # 低于下轨买入
                                     np.where(data['close'] > data['bb_upper_20_2.0'], -1, 0))  # 高于上轨卖出

        # 4. MACD信号 (动能)
        signals['macd_signal'] = np.where(data['macd_12_26'] > data['macd_signal_12_26_9'], 1, -1)

        # 5. 成交量信号 (成交量放大)
        signals['volume_signal'] = np.where(data['volume_ratio_10'] > 1.5, 1,  # 成交量显著放大
                                          np.where(data['volume_ratio_10'] < 0.5, -1, 0))

        # 6. 动量信号 (短期动量)
        signals['momentum_signal'] = np.where(data['momentum_5'] > 0.05, 1,    # 短期强势
                                           np.where(data['momentum_5'] < -0.05, -1, 0))

        # 7. 波动率信号 (波动率突破)
        signals['vol_signal'] = np.where(data['vol_20'] > data['vol_20'].rolling(60).quantile(0.8), -1,  # 高波动卖出
                                       np.where(data['vol_20'] < data['vol_20'].rolling(60).quantile(0.2), 1, 0))  # 低波动买入

        # 8. 趋势强度信号
        signals['trend_strength_signal'] = np.where(data['trend_strength'] > 0.03,
                                                  np.where(data['trend_20'] == 1, 1, -1), 0)

        # 计算综合信号 (加权平均)
        weight_dict = {
            'rsi_signal': 0.15,
            'ma_signal': 0.15,
            'bb_signal': 0.15,
            'macd_signal': 0.15,
            'volume_signal': 0.10,
            'momentum_signal': 0.10,
            'vol_signal': 0.10,
            'trend_strength_signal': 0.10
        }

        # 确保所有信号列都存在
        composite_signal = pd.Series(0, index=data.index)
        for signal_name, weight in weight_dict.items():
            if signal_name in signals.columns:
                composite_signal += signals[signal_name] * weight

        signals['composite_signal'] = composite_signal
        return signals

    def run_backtest(self, start_date='2025-01-01', end_date='2026-01-01'):
        """
        运行回测
        """
        print(f"🚀 开始回测，期间：{start_date} 至 {end_date}")

        all_results = {}

        for stock in self.universe:
            print(f"\n📊 正在分析 {stock}...")

            # 获取数据
            data = self.fetcher.fetch(stock, start_date, end_date, source='ashare')

            if data is None or data.empty:
                print(f"⚠️  无法获取 {stock} 的数据")
                continue

            # 计算技术因子
            factor_data = self.calculate_all_factors(data)

            # 生成信号
            signals = self.generate_signals(factor_data)
            factor_data = pd.concat([factor_data, signals], axis=1)

            # 计算策略收益
            factor_data['position'] = factor_data['composite_signal'].shift(1).fillna(0)  # 前一天信号决定今天仓位
            factor_data['strategy_returns'] = factor_data['position'] * factor_data['returns']
            factor_data['benchmark_returns'] = factor_data['returns']

            # 累计收益
            factor_data['cum_strategy_ret'] = (1 + factor_data['strategy_returns']).cumprod()
            factor_data['cum_benchmark_ret'] = (1 + factor_data['benchmark_returns']).cumprod()

            # 计算绩效指标
            total_strategy_ret = factor_data['cum_strategy_ret'].iloc[-1] - 1
            total_benchmark_ret = factor_data['cum_benchmark_ret'].iloc[-1] - 1
            strategy_annual_ret = (factor_data['cum_strategy_ret'].iloc[-1]) ** (252/len(factor_data)) - 1
            benchmark_annual_ret = (factor_data['cum_benchmark_ret'].iloc[-1]) ** (252/len(factor_data)) - 1

            # 风险指标
            strategy_vol = factor_data['strategy_returns'].std() * np.sqrt(252)
            benchmark_vol = factor_data['benchmark_returns'].std() * np.sqrt(252)
            max_drawdown = (factor_data['cum_strategy_ret'] / factor_data['cum_strategy_ret'].expanding().max() - 1).min()

            # 信息比率
            excess_returns = factor_data['strategy_returns'] - factor_data['benchmark_returns']
            info_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() != 0 else 0

            all_results[stock] = {
                'total_strategy_return': total_strategy_ret,
                'total_benchmark_return': total_benchmark_ret,
                'strategy_annual_return': strategy_annual_ret,
                'benchmark_annual_return': benchmark_annual_ret,
                'strategy_volatility': strategy_vol,
                'benchmark_volatility': benchmark_vol,
                'max_drawdown': max_drawdown,
                'info_ratio': info_ratio,
                'sharpe_ratio': strategy_annual_ret / strategy_vol if strategy_vol != 0 else 0,
                'data': factor_data
            }

            print(f"📈 {stock} 策略总收益: {total_strategy_ret*100:.2f}%")
            print(f"📈 {stock} 基准总收益: {total_benchmark_ret*100:.2f}%")
            print(f"📊 {stock} 信息比率: {info_ratio:.4f}")
            print(f"📊 {stock} 最大回撤: {max_drawdown*100:.2f}%")

        return all_results

    def generate_report(self, results):
        """
        生成回测报告
        """
        print("\n" + "="*80)
        print("📋 多因子策略回测报告")
        print("="*80)

        summary_df = pd.DataFrame({
            stock: {
                '策略收益': f"{result['total_strategy_return']*100:.2f}%",
                '基准收益': f"{result['total_benchmark_return']*100:.2f}%",
                '超额收益': f"{(result['total_strategy_return']-result['total_benchmark_return'])*100:.2f}%",
                '年化收益': f"{result['strategy_annual_return']*100:.2f}%",
                '年化波动率': f"{result['strategy_volatility']*100:.2f}%",
                '夏普比率': f"{result['sharpe_ratio']:.4f}",
                '信息比率': f"{result['info_ratio']:.4f}",
                '最大回撤': f"{result['max_drawdown']*100:.2f}%"
            }
            for stock, result in results.items()
        }).T

        print(summary_df)

        print(f"\n💡 总结:")
        avg_strategy_ret = np.mean([r['strategy_annual_return'] for r in results.values()])
        avg_benchmark_ret = np.mean([r['benchmark_annual_return'] for r in results.values()])
        avg_ir = np.mean([r['info_ratio'] for r in results.values()])

        print(f"• 平均年化超额收益: {(avg_strategy_ret - avg_benchmark_ret)*100:.2f}%")
        print(f"• 平均信息比率: {avg_ir:.4f}")
        print(f"• 策略有效性: {'✅' if avg_ir > 0.1 else '⚠️ ' if avg_ir > 0 else '❌'}")

        print("="*80)

def main():
    """
    主函数 - 演示完整流程
    """
    print("🎯 A-Share 多因子策略系统")
    print("📊 包含100+技术指标的量化分析")

    # 创建策略实例
    strategy = MultiFactorStrategy(universe=['sh600023', 'sh600519', 'sz000001'])

    # 运行回测
    results = strategy.run_backtest(start_date='2025-06-01', end_date='2026-01-01')

    # 生成报告
    strategy.generate_report(results)

    print("\n🎯 系统功能总结:")
    print("1. ✅ 获取A-Share实时数据")
    print("2. ✅ 计算100+技术指标")
    print("3. ✅ 生成多因子交易信号")
    print("4. ✅ 运行策略回测")
    print("5. ✅ 生成详细绩效报告")
    print("\n💡 策略可扩展性:")
    print("  - 可添加自定义因子")
    print("  - 可调整因子权重")
    print("  - 可优化交易信号逻辑")
    print("  - 可扩展至更多股票")

if __name__ == "__main__":
    main()