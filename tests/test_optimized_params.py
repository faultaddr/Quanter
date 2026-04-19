#!/usr/bin/env python
"""
优化参数回测

使用优化后的参数回测用户指定股票
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


# 用户指定股票
TARGET_STOCKS = [
    '688271',  # 联影医疗
    '600460',  # 士兰微
    '600515',  # 海航科技
    '001965',  # 招商公路
    '600066',  # 宇通客车
    '600000',  # 浦发银行
    '600600',  # 青岛啤酒
]

INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365

# 优化后的参数
OPTIMIZED_PARAMS = {
    'model_type': 'lgb',
    'feature_set': 'Alpha158',
    'buy_threshold': 0.53,     # 优化后
    'sell_threshold': 0.30,    # 优化后
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.12,
    'epochs': 50,
}

# 原参数
ORIGINAL_PARAMS = {
    'model_type': 'lgb',
    'feature_set': 'Alpha158',
    'buy_threshold': 0.55,
    'sell_threshold': 0.45,
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
        '688271': '联影医疗',
        '600460': '士兰微',
        '600515': '海航科技',
        '001965': '招商公路',
        '600066': '宇通客车',
        '600000': '浦发银行',
        '600600': '青岛啤酒',
    }
    return names.get(symbol, '未知')


def run_backtest_with_params(params: dict, stock_data: dict, label: str) -> dict:
    """使用指定参数运行回测"""
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"参数: 买入={params['buy_threshold']}, 卖出={params['sell_threshold']}, "
          f"止损={params['stop_loss_pct']*100:.0f}%, 止盈={params['take_profit_pct']*100:.0f}%")

    results = []
    total_capital = INITIAL_CAPITAL * len(stock_data)

    for symbol, df in stock_data.items():
        try:
            strategy = QlibStrategy(
                model_type=params['model_type'],
                feature_set=params['feature_set'],
                buy_threshold=params['buy_threshold'],
                sell_threshold=params['sell_threshold'],
                stop_loss_pct=params['stop_loss_pct'],
                take_profit_pct=params['take_profit_pct'],
                epochs=params['epochs'],
            )

            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
            strategy.train_model(train_data, horizon=10)

            engine = BacktestEngine()
            engine.set_initial_cash(INITIAL_CAPITAL)
            engine.set_commission_rate(0.0003)
            engine.set_t_plus_1(True)

            result = engine.run_backtest(
                strategy=strategy,
                data={symbol: df.copy()},
                start_date=df['timestamp'].iloc[0],
                end_date=df['timestamp'].iloc[-1]
            )

            print(f"  {symbol} ({get_stock_name(symbol)}): 年化 {result.annual_return*100:.2f}%, 夏普 {result.sharpe_ratio:.2f}")

            results.append({
                'symbol': symbol,
                'name': get_stock_name(symbol),
                'success': True,
                'annual_return': result.annual_return,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
            })

        except Exception as e:
            print(f"  {symbol}: ❌ 失败 - {e}")

    successful = [r for r in results if r.get('success')]
    if not successful:
        return {'success': False}

    total_final = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in successful)
    total_return = (total_final - total_capital) / total_capital
    annual_return = (1 + total_return) ** (252 / LOOKBACK_DAYS) - 1

    return {
        'success': True,
        'annual_return': annual_return,
        'total_return': total_return,
        'avg_sharpe': np.mean([r['sharpe_ratio'] for r in successful]),
        'results': successful,
    }


def main():
    print("=" * 80)
    print("优化参数 vs 原参数对比回测")
    print("=" * 80)

    # 获取数据
    print("\n获取股票数据...")
    stock_data = {}
    for symbol in TARGET_STOCKS:
        df = fetch_stock_data(symbol)
        if not df.empty and len(df) >= 120:
            stock_data[symbol] = df
            print(f"  ✅ {symbol} ({get_stock_name(symbol)}): {len(df)} 条")

    if not stock_data:
        print("❌ 没有有效数据")
        return

    print(f"\n有效股票: {len(stock_data)} 只")

    # 测试原参数
    original_result = run_backtest_with_params(ORIGINAL_PARAMS, stock_data, "原参数回测")

    # 测试优化参数
    optimized_result = run_backtest_with_params(OPTIMIZED_PARAMS, stock_data, "优化参数回测")

    # 对比结果
    print("\n" + "=" * 80)
    print("参数对比结果")
    print("=" * 80)

    print(f"\n{'指标':<20} {'原参数':>15} {'优化参数':>15} {'提升':>15}")
    print("-" * 80)
    print(f"{'组合年化收益':<20} {original_result['annual_return']*100:>14.2f}% "
          f"{optimized_result['annual_return']*100:>14.2f}% "
          f"{(optimized_result['annual_return']-original_result['annual_return'])*100:>14.2f}%")
    print(f"{'平均夏普比':<20} {original_result['avg_sharpe']:>15.2f} "
          f"{optimized_result['avg_sharpe']:>15.2f} "
          f"{optimized_result['avg_sharpe']-original_result['avg_sharpe']:>15.2f}")

    print("\n📊 各股票对比:")
    print("-" * 80)
    print(f"{'股票':<10} {'原参数收益':>12} {'优化参数收益':>12} {'提升':>10}")
    print("-" * 80)

    for orig_r in original_result['results']:
        opt_r = next((r for r in optimized_result['results'] if r['symbol'] == orig_r['symbol']), None)
        if opt_r:
            improvement = (opt_r['annual_return'] - orig_r['annual_return']) * 100
            print(f"{orig_r['symbol']:<10} {orig_r['annual_return']*100:>11.2f}% "
                  f"{opt_r['annual_return']*100:>11.2f}% {improvement:>9.2f}%")

    print("-" * 80)

    # 结论
    if optimized_result['annual_return'] > original_result['annual_return']:
        print(f"\n✅ 优化参数表现更好！年化收益提升了 {(optimized_result['annual_return']-original_result['annual_return'])*100:.2f}%")
    else:
        print(f"\n⚠️ 原参数表现更好，建议保持原参数")


if __name__ == "__main__":
    main()