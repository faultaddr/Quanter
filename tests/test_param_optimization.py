#!/usr/bin/env python
"""
策略参数优化脚本

搜索最佳买卖阈值参数
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from itertools import product

# 导入模块
from quanttool.strategies.score_strategy import ScoreStrategy
from quanttool.factors.scoring_system import ScoringSystem
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher


def fetch_real_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """获取真实股票数据"""
    end_date = datetime.now().strftime('%Y-%m-%d')

    df = AshareFetcher.get_price(
        code=symbol,
        end_date=end_date,
        count=days,
        frequency='1d'
    )

    if df.empty:
        return pd.DataFrame()

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df


def run_single_backtest(df: pd.DataFrame, symbol: str, buy_th: float, sell_th: float) -> dict:
    """运行单个参数组合的回测"""
    strategy = ScoreStrategy(
        buy_threshold=buy_th,
        sell_threshold=sell_th,
        use_dynamic_weights=True,
        use_multi_timeframe=True,
        use_risk_control=True
    )

    engine = BacktestEngine()
    engine.set_initial_cash(100000)

    data = {symbol: df.copy()}

    try:
        result = engine.run_backtest(
            strategy=strategy,
            data=data,
            start_date=df['timestamp'].iloc[0],
            end_date=df['timestamp'].iloc[-1]
        )

        return {
            'buy_threshold': buy_th,
            'sell_threshold': sell_th,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'success': True
        }
    except Exception as e:
        return {
            'buy_threshold': buy_th,
            'sell_threshold': sell_th,
            'error': str(e),
            'success': False
        }


def optimize_parameters(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """参数优化"""
    print(f"\n{'='*60}")
    print(f"参数优化: {symbol}")
    print("="*60)

    # 参数搜索空间
    buy_thresholds = [50, 55, 60, 65, 70]
    sell_thresholds = [20, 25, 30, 35, 40, 45, 50]

    results = []
    total_combinations = len(buy_thresholds) * len(sell_thresholds)
    count = 0

    for buy_th in buy_thresholds:
        for sell_th in sell_thresholds:
            count += 1
            print(f"\r  优化进度: {count}/{total_combinations}  当前: 买入={buy_th}, 卖出={sell_th}", end="")

            if buy_th <= sell_th:
                continue

            result = run_single_backtest(df, symbol, buy_th, sell_th)
            if result.get('success'):
                results.append(result)

    print()  # 换行

    # 转换为 DataFrame
    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("无有效结果")
        return pd.DataFrame()

    # 按总收益排序
    results_df = results_df.sort_values('total_return', ascending=False)

    print(f"\n前 10 名参数组合:")
    print("-" * 80)
    print(f"{'买入阈值':>8} {'卖出阈值':>8} {'总收益':>10} {'夏普比率':>10} {'最大回撤':>10} {'胜率':>8} {'交易数':>6}")
    print("-" * 80)

    for _, row in results_df.head(10).iterrows():
        print(f"{row['buy_threshold']:>8.0f} {row['sell_threshold']:>8.0f} "
              f"{row['total_return']:>10.2%} {row['sharpe_ratio']:>10.2f} "
              f"{row['max_drawdown']:>10.2%} {row['win_rate']:>8.2%} {row['total_trades']:>6.0f}")

    return results_df


def analyze_score_distribution(df: pd.DataFrame) -> dict:
    """分析评分分布"""
    scoring_system = ScoringSystem()

    scores = []
    for i in range(30, len(df)):
        window = df.iloc[:i+1]
        try:
            result = scoring_system.calculate_comprehensive_score(window)
            scores.append(result.get('final_score', 50))
        except Exception:
            scores.append(50)

    scores = np.array(scores)

    # 计算各阈值下的信号数
    thresholds = [40, 45, 50, 55, 60, 65, 70, 75, 80]

    print("\n评分分布与信号统计:")
    print("-" * 50)

    for th in thresholds:
        buy_count = (scores >= th).sum()
        sell_count = (scores <= th).sum()
        print(f"  阈值 {th:2d}: 买入信号={buy_count:4d}, 卖出信号={sell_count:4d}")

    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'median': np.median(scores)
    }


def main():
    """主函数"""
    print("\n" + "="*60)
    print("QuantTool 策略参数优化")
    print("="*60)

    # 测试股票
    test_symbols = [
        ('600519', '贵州茅台'),
        ('000001', '平安银行'),
        ('300750', '宁德时代'),
    ]

    all_results = {}
    best_params_all = {}

    for symbol, name in test_symbols:
        print(f"\n{'='*60}")
        print(f"分析 {name} ({symbol})")
        print("="*60)

        # 获取数据
        df = fetch_real_data(symbol, days=365)

        if df.empty:
            print(f"无法获取 {symbol} 数据")
            continue

        print(f"数据量: {len(df)} 条")

        # 分析评分分布
        score_stats = analyze_score_distribution(df)

        print(f"\n评分统计:")
        print(f"  平均: {score_stats['mean']:.2f}")
        print(f"  标准差: {score_stats['std']:.2f}")
        print(f"  范围: {score_stats['min']:.2f} ~ {score_stats['max']:.2f}")

        # 参数优化
        opt_results = optimize_parameters(symbol, df)

        if not opt_results.empty:
            best = opt_results.iloc[0]
            best_params_all[symbol] = {
                'buy_threshold': best['buy_threshold'],
                'sell_threshold': best['sell_threshold'],
                'total_return': best['total_return'],
                'sharpe_ratio': best['sharpe_ratio']
            }

        all_results[symbol] = opt_results

    # 汇总最佳参数
    print("\n" + "="*60)
    print("各股票最佳参数汇总")
    print("="*60)

    print(f"\n{'股票':>10} {'买入阈值':>8} {'卖出阈值':>8} {'总收益':>10} {'夏普比率':>10}")
    print("-" * 50)

    for symbol, params in best_params_all.items():
        print(f"{symbol:>10} {params['buy_threshold']:>8.0f} {params['sell_threshold']:>8.0f} "
              f"{params['total_return']:>10.2%} {params['sharpe_ratio']:>10.2f}")

    # 推荐参数
    if best_params_all:
        avg_buy = np.mean([p['buy_threshold'] for p in best_params_all.values()])
        avg_sell = np.mean([p['sell_threshold'] for p in best_params_all.values()])

        print(f"\n推荐参数 (各股票平均值):")
        print(f"  买入阈值: {avg_buy:.0f}")
        print(f"  卖出阈值: {avg_sell:.0f}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)