#!/usr/bin/env python
"""
深度参数优化

使用网格搜索找出最优参数组合
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
import itertools
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from quanttool.strategies.qlib_strategy import QlibStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.data_fetcher import AshareFetcher


# 测试股票
TEST_STOCKS = ['600066', '600460', '688271', '600000', '600600', '601857', '600588']

INITIAL_CAPITAL = 50000.0
LOOKBACK_DAYS = 365

# 参数网格 (更细粒度)
PARAM_GRID = {
    'buy_threshold': [0.65, 0.70, 0.75],
    'sell_threshold': [0.45, 0.50, 0.55],
    'stop_loss_pct': [0.08, 0.10, 0.12],
    'take_profit_pct': [0.20, 0.25, 0.30],
}

# 真实成本
COSTS = {
    'commission': 0.0003,
    'slippage': 0.001,
}


def fetch_stock_data(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """获取股票历史数据"""
    end_date = datetime.now().strftime('%Y-%m-%d')
    try:
        df = AshareFetcher.get_price(code=symbol, end_date=end_date, count=days + 100, frequency='1d')
        if df.empty:
            return pd.DataFrame()
        if 'timestamp' not in df.columns:
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)
        return df
    except Exception:
        return pd.DataFrame()


def test_params(params: dict, stock_data: dict) -> dict:
    """测试单组参数"""
    results = []

    for symbol, df in stock_data.items():
        try:
            strategy = QlibStrategy(
                model_type='lgb',
                feature_set='Alpha158',
                buy_threshold=params['buy_threshold'],
                sell_threshold=params['sell_threshold'],
                stop_loss_pct=params['stop_loss_pct'],
                take_profit_pct=params['take_profit_pct'],
                epochs=20,  # 减少训练轮数加快优化
            )

            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
            strategy.train_model(train_data, horizon=10)

            engine = BacktestEngine()
            engine.set_initial_cash(INITIAL_CAPITAL)
            engine.set_commission_rate(COSTS['commission'])
            engine.set_slippage_rate(COSTS['slippage'])
            engine.set_t_plus_1(True)

            result = engine.run_backtest(
                strategy=strategy,
                data={symbol: df.copy()},
                start_date=df['timestamp'].iloc[0],
                end_date=df['timestamp'].iloc[-1]
            )

            # 计算含印花税收益
            estimated_sell_volume = INITIAL_CAPITAL * result.total_trades * 0.5
            stamp_duty_cost = estimated_sell_volume * 0.001
            adjusted_return = result.total_return - stamp_duty_cost / INITIAL_CAPITAL
            adjusted_annual = (1 + adjusted_return) ** (252 / LOOKBACK_DAYS) - 1

            results.append({
                'success': True,
                'adjusted_annual_return': adjusted_annual,
                'sharpe_ratio': result.sharpe_ratio,
                'total_trades': result.total_trades,
            })

        except Exception:
            results.append({'success': False})

    successful = [r for r in results if r.get('success')]
    if not successful:
        return {'success': False, 'score': -999}

    # 计算综合得分 (收益 * 0.6 + 夏普 * 0.3 - 交易频率惩罚 * 0.1)
    avg_return = np.mean([r['adjusted_annual_return'] for r in successful])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in successful])
    avg_trades = np.mean([r['total_trades'] for r in successful])

    # 交易频率惩罚 (每100次交易惩罚0.5%)
    trade_penalty = avg_trades / 100 * 0.005

    score = avg_return * 0.6 + avg_sharpe * 0.03 - trade_penalty * 0.1

    return {
        'success': True,
        'score': score,
        'avg_return': avg_return,
        'avg_sharpe': avg_sharpe,
        'avg_trades': avg_trades,
    }


def main():
    print("=" * 80)
    print("深度参数优化")
    print("=" * 80)

    # 获取数据
    print("\n获取股票数据...")
    stock_data = {}
    for symbol in TEST_STOCKS:
        df = fetch_stock_data(symbol)
        if not df.empty and len(df) >= 120:
            stock_data[symbol] = df
            print(f"  ✅ {symbol}: {len(df)} 条")

    if not stock_data:
        print("❌ 没有有效数据")
        return

    print(f"\n有效股票: {len(stock_data)} 只")

    # 生成参数组合
    keys = PARAM_GRID.keys()
    values = PARAM_GRID.values()
    combinations = list(itertools.product(*values))
    total = len(combinations)

    print(f"\n参数组合数量: {total}")
    print(f"参数范围:")
    for k, v in PARAM_GRID.items():
        print(f"  {k}: {v}")

    # 测试所有参数组合
    print("\n" + "=" * 80)
    print("开始优化...")
    print("=" * 80 + "\n")

    all_results = []
    start_time = time.time()

    for i, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))

        print(f"[{i}/{total}] 买入={params['buy_threshold']}, 卖出={params['sell_threshold']}, "
              f"止损={params['stop_loss_pct']*100:.0f}%, 止盈={params['take_profit_pct']*100:.0f}%...", end=" ")

        result = test_params(params, stock_data)

        if result['success']:
            print(f"✅ 收益={result['avg_return']*100:.2f}%, 夏普={result['avg_sharpe']:.2f}, "
                  f"交易={result['avg_trades']:.0f}, 得分={result['score']:.4f}")
            all_results.append({
                'params': params,
                'score': result['score'],
                'avg_return': result['avg_return'],
                'avg_sharpe': result['avg_sharpe'],
                'avg_trades': result['avg_trades'],
            })
        else:
            print("❌ 失败")

    elapsed = time.time() - start_time

    # 排序结果
    all_results.sort(key=lambda x: x['score'], reverse=True)

    # 打印结果
    print("\n" + "=" * 80)
    print("优化结果")
    print("=" * 80)

    print(f"\n📊 Top 10 参数组合 (按综合得分排序):")
    print("-" * 100)
    print(f"{'排名':<4} {'买入':>6} {'卖出':>6} {'止损':>6} {'止盈':>6} "
          f"{'年化收益':>10} {'夏普':>8} {'交易':>8} {'得分':>10}")
    print("-" * 100)

    for i, r in enumerate(all_results[:10], 1):
        p = r['params']
        print(f"{i:<4} {p['buy_threshold']:>6.2f} {p['sell_threshold']:>6.2f} "
              f"{p['stop_loss_pct']*100:>5.0f}% {p['take_profit_pct']*100:>5.0f}% "
              f"{r['avg_return']*100:>9.2f}% {r['avg_sharpe']:>8.2f} "
              f"{r['avg_trades']:>8.0f} {r['score']:>10.4f}")

    print("-" * 100)

    # 最佳参数
    if all_results:
        best = all_results[0]
        print(f"\n🏆 最佳参数组合:")
        print(f"   买入阈值: {best['params']['buy_threshold']}")
        print(f"   卖出阈值: {best['params']['sell_threshold']}")
        print(f"   止损比例: {best['params']['stop_loss_pct']*100:.0f}%")
        print(f"   止盈比例: {best['params']['take_profit_pct']*100:.0f}%")
        print(f"\n   年化收益: {best['avg_return']*100:.2f}%")
        print(f"   夏普比: {best['avg_sharpe']:.2f}")
        print(f"   平均交易: {best['avg_trades']:.0f} 次/年")
        print(f"   综合得分: {best['score']:.4f}")
        print(f"\n   优化耗时: {elapsed:.1f} 秒")

    return all_results


if __name__ == "__main__":
    results = main()