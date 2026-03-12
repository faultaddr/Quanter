#!/usr/bin/env python
"""
低频交易策略优化

目标：降低交易频率，减少交易成本侵蚀
方法：
1. 提高买入/卖出阈值
2. 增加持仓确认周期
3. 添加最小持仓天数
4. 放宽止损/止盈幅度
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
from quanttool.infrastructure.data_providers.data_fetcher import AshareFetcher


# 测试股票
TARGET_STOCKS = ['600066', '600460', '688271', '600000', '600600']

INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365

# 真实成本
COSTS = {
    'commission': 0.0003,
    'slippage': 0.001,
    'stamp_duty': 0.001,
}

# 参数组合测试
PARAM_COMBINATIONS = [
    {
        'name': '高频交易 (基准)',
        'buy_threshold': 0.53,
        'sell_threshold': 0.30,
        'stop_loss_pct': 0.05,
        'take_profit_pct': 0.12,
        'min_hold_days': 0,
    },
    {
        'name': '中频交易',
        'buy_threshold': 0.60,
        'sell_threshold': 0.40,
        'stop_loss_pct': 0.06,
        'take_profit_pct': 0.15,
        'min_hold_days': 3,
    },
    {
        'name': '低频交易',
        'buy_threshold': 0.65,
        'sell_threshold': 0.45,
        'stop_loss_pct': 0.08,
        'take_profit_pct': 0.20,
        'min_hold_days': 5,
    },
    {
        'name': '极低频交易',
        'buy_threshold': 0.70,
        'sell_threshold': 0.50,
        'stop_loss_pct': 0.10,
        'take_profit_pct': 0.25,
        'min_hold_days': 10,
    },
]


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


def run_backtest_with_params(params: dict, stock_data: dict) -> dict:
    """使用指定参数运行回测"""
    results = []
    total_trades_list = []

    for symbol, df in stock_data.items():
        try:
            strategy = QlibStrategy(
                model_type='lgb',
                feature_set='Alpha158',
                buy_threshold=params['buy_threshold'],
                sell_threshold=params['sell_threshold'],
                stop_loss_pct=params['stop_loss_pct'],
                take_profit_pct=params['take_profit_pct'],
                epochs=30,
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

            # 计算印花税影响
            estimated_sell_volume = INITIAL_CAPITAL * result.total_trades * 0.5
            stamp_duty_cost = estimated_sell_volume * COSTS['stamp_duty']
            adjusted_return = result.total_return - stamp_duty_cost / INITIAL_CAPITAL
            adjusted_annual = (1 + adjusted_return) ** (252 / LOOKBACK_DAYS) - 1

            results.append({
                'symbol': symbol,
                'success': True,
                'annual_return': result.annual_return,
                'adjusted_annual_return': adjusted_annual,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'total_trades': result.total_trades,
                'max_drawdown': result.max_drawdown,
            })
            total_trades_list.append(result.total_trades)

        except Exception:
            results.append({'symbol': symbol, 'success': False})

    successful = [r for r in results if r.get('success')]
    if not successful:
        return {'success': False}

    total_capital = INITIAL_CAPITAL * len(successful)
    total_final = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in successful)
    total_return = (total_final - total_capital) / total_capital
    annual_return = (1 + total_return) ** (252 / LOOKBACK_DAYS) - 1

    # 计算调整后收益
    total_final_adj = sum(INITIAL_CAPITAL * (1 + r['adjusted_annual_return'] * LOOKBACK_DAYS / 252) for r in successful)
    total_return_adj = (total_final_adj - total_capital) / total_capital
    annual_return_adj = (1 + total_return_adj) ** (252 / LOOKBACK_DAYS) - 1

    return {
        'success': True,
        'annual_return': annual_return,
        'adjusted_annual_return': annual_return_adj,
        'avg_sharpe': np.mean([r['sharpe_ratio'] for r in successful]),
        'avg_drawdown': np.mean([r['max_drawdown'] for r in successful]),
        'total_trades': sum(total_trades_list),
        'avg_trades_per_stock': np.mean(total_trades_list),
        'results': successful,
    }


def main():
    print("=" * 80)
    print("低频交易策略优化")
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

    # 测试不同参数组合
    print("\n" + "=" * 80)
    print("测试不同交易频率参数...")
    print("=" * 80)

    all_results = []

    for params in PARAM_COMBINATIONS:
        print(f"\n{'='*60}")
        print(f"📊 {params['name']}")
        print(f"{'='*60}")
        print(f"参数: 买入={params['buy_threshold']}, 卖出={params['sell_threshold']}, "
              f"止损={params['stop_loss_pct']*100:.0f}%, 止盈={params['take_profit_pct']*100:.0f}%")

        result = run_backtest_with_params(params, stock_data)

        if result['success']:
            print(f"\n📈 结果:")
            print(f"   年化收益 (含真实成本): {result['adjusted_annual_return']*100:.2f}%")
            print(f"   平均夏普比: {result['avg_sharpe']:.2f}")
            print(f"   平均回撤: {result['avg_drawdown']*100:.2f}%")
            print(f"   总交易次数: {result['total_trades']}")
            print(f"   每只股票平均交易: {result['avg_trades_per_stock']:.1f} 次")

            all_results.append({
                'name': params['name'],
                'params': params,
                'adjusted_annual_return': result['adjusted_annual_return'],
                'avg_sharpe': result['avg_sharpe'],
                'avg_drawdown': result['avg_drawdown'],
                'total_trades': result['total_trades'],
                'avg_trades_per_stock': result['avg_trades_per_stock'],
            })
        else:
            print("   ❌ 回测失败")

    # 对比结果
    print("\n" + "=" * 80)
    print("参数对比结果")
    print("=" * 80)

    print(f"\n{'策略':<15} {'年化收益':>12} {'夏普':>8} {'回撤':>8} {'交易次数':>10} {'每只交易':>10}")
    print("-" * 80)

    for r in all_results:
        print(f"{r['name']:<15} {r['adjusted_annual_return']*100:>11.2f}% "
              f"{r['avg_sharpe']:>8.2f} {r['avg_drawdown']*100:>7.2f}% "
              f"{r['total_trades']:>10} {r['avg_trades_per_stock']:>10.1f}")

    print("-" * 80)

    # 找出最佳参数
    best = max(all_results, key=lambda x: x['adjusted_annual_return'])
    print(f"\n🏆 最佳策略: {best['name']}")
    print(f"   年化收益 (含真实成本): {best['adjusted_annual_return']*100:.2f}%")
    print(f"   总交易次数: {best['total_trades']}")
    print(f"   交易频率: 每只股票 {best['avg_trades_per_stock']:.1f} 次/年")

    # 计算收益/交易效率
    print(f"\n📊 效率分析:")
    print(f"{'策略':<15} {'收益/交易':>12} {'说明':>30}")
    print("-" * 60)
    for r in all_results:
        efficiency = r['adjusted_annual_return'] * 100 / r['total_trades'] if r['total_trades'] > 0 else 0
        note = "高频低效" if efficiency < 0.02 else ("中频中效" if efficiency < 0.03 else "低频高效")
        print(f"{r['name']:<15} {efficiency:>11.3f}% {note:>30}")

    # 推荐
    print(f"\n💡 推荐配置:")
    print(f"   买入阈值: {best['params']['buy_threshold']}")
    print(f"   卖出阈值: {best['params']['sell_threshold']}")
    print(f"   止损比例: {best['params']['stop_loss_pct']*100:.0f}%")
    print(f"   止盈比例: {best['params']['take_profit_pct']*100:.0f}%")


if __name__ == "__main__":
    main()