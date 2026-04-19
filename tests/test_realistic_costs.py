#!/usr/bin/env python
"""
真实成本回测

使用更真实的交易成本设置：
- 手续费：0.03%
- 滑点：0.1% (双向 0.2%)
- 印花税：0.1% (仅卖出，通过调整卖出价格模拟)
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from quanttool.strategies.qlib_strategy import QlibStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher


# 测试股票
TARGET_STOCKS = ['600066', '600460', '688271', '600000', '600600']

INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365

# 真实成本设置
REALISTIC_COSTS = {
    'commission': 0.0003,   # 手续费 0.03%
    'slippage': 0.001,      # 滑点 0.1% (双向 0.2%)
    'stamp_duty': 0.001,    # 印花税 0.1% (仅卖出)
}

# 优化后的策略参数
STRATEGY_PARAMS = {
    'model_type': 'lgb',
    'feature_set': 'Alpha158',
    'buy_threshold': 0.53,
    'sell_threshold': 0.30,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.12,
    'epochs': 50,
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


def get_stock_name(symbol: str) -> str:
    """获取股票名称"""
    names = {
        '600066': '宇通客车',
        '600460': '士兰微',
        '688271': '联影医疗',
        '600000': '浦发银行',
        '600600': '青岛啤酒',
    }
    return names.get(symbol, '未知')


def run_backtest_with_costs(costs: dict, label: str) -> dict:
    """使用指定成本设置运行回测"""
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"成本设置:")
    print(f"  手续费: {costs['commission']*100:.3f}%")
    print(f"  滑点: {costs['slippage']*100:.2f}% (双向 {costs['slippage']*200:.2f}%)")
    print(f"  印花税: {costs['stamp_duty']*100:.2f}% (仅卖出)")

    results = []

    for symbol in TARGET_STOCKS:
        df = fetch_stock_data(symbol)
        if len(df) < 120:
            continue

        try:
            strategy = QlibStrategy(**STRATEGY_PARAMS)

            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
            strategy.train_model(train_data, horizon=10)

            engine = BacktestEngine()
            engine.set_initial_cash(INITIAL_CAPITAL)
            engine.set_commission_rate(costs['commission'])
            engine.set_slippage_rate(costs['slippage'])
            engine.set_t_plus_1(True)

            result = engine.run_backtest(
                strategy=strategy,
                data={symbol: df.copy()},
                start_date=df['timestamp'].iloc[0],
                end_date=df['timestamp'].iloc[-1]
            )

            # 计算印花税影响 (卖出金额的 0.1%)
            # 假设卖出金额约为初始资金 * 交易次数的一半
            estimated_sell_volume = INITIAL_CAPITAL * result.total_trades * 0.5
            stamp_duty_cost = estimated_sell_volume * costs['stamp_duty']
            adjusted_return = result.total_return - stamp_duty_cost / INITIAL_CAPITAL
            adjusted_annual = (1 + adjusted_return) ** (252 / LOOKBACK_DAYS) - 1

            print(f"  {symbol} ({get_stock_name(symbol)}): "
                  f"年化 {result.annual_return*100:.2f}% → {adjusted_annual*100:.2f}% (含印花税), "
                  f"交易 {result.total_trades} 次")

            results.append({
                'symbol': symbol,
                'name': get_stock_name(symbol),
                'success': True,
                'annual_return': result.annual_return,
                'adjusted_annual_return': adjusted_annual,
                'total_return': result.total_return,
                'adjusted_return': adjusted_return,
                'sharpe_ratio': result.sharpe_ratio,
                'total_trades': result.total_trades,
                'stamp_duty_cost': stamp_duty_cost,
            })

        except Exception as e:
            print(f"  {symbol}: ❌ 失败 - {e}")

    if not results:
        return {'success': False}

    # 计算组合收益
    total_capital = INITIAL_CAPITAL * len(results)

    # 原始收益
    total_final_original = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in results)
    total_return_original = (total_final_original - total_capital) / total_capital
    annual_return_original = (1 + total_return_original) ** (252 / LOOKBACK_DAYS) - 1

    # 调整后收益 (含印花税)
    total_final_adjusted = sum(INITIAL_CAPITAL * (1 + r['adjusted_return']) for r in results)
    total_return_adjusted = (total_final_adjusted - total_capital) / total_capital
    annual_return_adjusted = (1 + total_return_adjusted) ** (252 / LOOKBACK_DAYS) - 1

    return {
        'success': True,
        'annual_return_original': annual_return_original,
        'annual_return_adjusted': annual_return_adjusted,
        'avg_sharpe': np.mean([r['sharpe_ratio'] for r in results]),
        'total_trades': sum(r['total_trades'] for r in results),
        'results': results,
    }


def main():
    print("=" * 80)
    print("真实交易成本回测")
    print("=" * 80)

    # 测试当前成本设置 (滑点 0.01%)
    print("\n" + "=" * 60)
    print("场景 1: 当前设置 (滑点 0.02%, 无印花税)")
    print("=" * 60)
    current_costs = {
        'commission': 0.0003,
        'slippage': 0.0001,  # 0.01% 单向, 双向 0.02%
        'stamp_duty': 0,
    }
    current_result = run_backtest_with_costs(current_costs, "当前设置")

    # 测试真实成本设置 (滑点 0.1% + 印花税 0.1%)
    print("\n" + "=" * 60)
    print("场景 2: 真实成本 (滑点 0.2%, 印花税 0.1%)")
    print("=" * 60)
    realistic_costs = {
        'commission': 0.0003,
        'slippage': 0.001,  # 0.1% 单向, 双向 0.2%
        'stamp_duty': 0.001,  # 0.1%
    }
    realistic_result = run_backtest_with_costs(realistic_costs, "真实成本")

    # 对比结果
    print("\n" + "=" * 80)
    print("成本对比结果")
    print("=" * 80)

    print(f"\n{'场景':<20} {'年化收益':>15} {'调整后收益':>15} {'夏普':>10} {'交易次数':>10}")
    print("-" * 80)
    print(f"{'当前设置':<20} {current_result['annual_return_original']*100:>14.2f}% "
          f"{'-':>15} {current_result['avg_sharpe']:>10.2f} {current_result['total_trades']:>10}")
    print(f"{'真实成本':<20} {realistic_result['annual_return_original']*100:>14.2f}% "
          f"{realistic_result['annual_return_adjusted']*100:>14.2f}% "
          f"{realistic_result['avg_sharpe']:>10.2f} {realistic_result['total_trades']:>10}")

    # 计算成本影响
    cost_impact = current_result['annual_return_original'] - realistic_result['annual_return_adjusted']
    print("-" * 80)
    print(f"{'成本影响':<20} {'':<15} {cost_impact*100:>14.2f}%")

    # 各股票对比
    print(f"\n📊 各股票收益对比:")
    print("-" * 80)
    print(f"{'股票':<10} {'当前设置':>12} {'真实成本':>12} {'成本影响':>12}")
    print("-" * 80)

    for curr_r in current_result['results']:
        real_r = next((r for r in realistic_result['results'] if r['symbol'] == curr_r['symbol']), None)
        if real_r:
            impact = (curr_r['annual_return'] - real_r['adjusted_annual_return']) * 100
            print(f"{curr_r['symbol']:<10} {curr_r['annual_return']*100:>11.2f}% "
                  f"{real_r['adjusted_annual_return']*100:>11.2f}% {impact:>11.2f}%")

    print("-" * 80)

    # 结论
    print(f"\n💡 结论:")
    print(f"   真实成本下年化收益: {realistic_result['annual_return_adjusted']*100:.2f}%")
    print(f"   相比当前设置下降: {cost_impact*100:.2f}%")
    print(f"\n   建议: 回测结果应打 {realistic_result['annual_return_adjusted']/current_result['annual_return_original']*100:.0f}% 折扣")


if __name__ == "__main__":
    main()