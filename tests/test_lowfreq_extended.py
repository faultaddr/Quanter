#!/usr/bin/env python
"""
极低频策略扩展回测

使用优化后的极低频参数回测更多股票
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


# 沪深300成分股（扩展列表）
CSI300_STOCKS = [
    # 金融
    '600036', '601318', '601166', '601398', '601288', '601939', '601328', '600000',
    # 消费
    '600519', '600887', '000858', '000568', '600600', '000895', '002714',
    # 科技
    '600031', '000725', '002415', '600588', '002230',
    # 医药
    '600276', '000538', '002007', '300015',
    # 能源
    '601088', '601857', '600028', '601225', '600900',
    # 制造
    '600066', '600460', '600875', '601766', '600893',
    # 其他
    '600309', '600585', '600660', '600009', '600104',
]

INITIAL_CAPITAL = 30000.0
LOOKBACK_DAYS = 365
N_STOCKS = 30  # 回测股票数量

# 极低频策略参数 (优化后)
STRATEGY_PARAMS = {
    'model_type': 'lgb',
    'feature_set': 'Alpha158',
    'buy_threshold': 0.70,
    'sell_threshold': 0.50,
    'stop_loss_pct': 0.10,
    'take_profit_pct': 0.25,
    'epochs': 30,
}

# 真实成本
COSTS = {
    'commission': 0.0003,
    'slippage': 0.001,
    'stamp_duty': 0.001,
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


def main():
    print("=" * 80)
    print("极低频策略扩展回测")
    print("=" * 80)
    print(f"\n策略参数:")
    print(f"  买入阈值: {STRATEGY_PARAMS['buy_threshold']}")
    print(f"  卖出阈值: {STRATEGY_PARAMS['sell_threshold']}")
    print(f"  止损: {STRATEGY_PARAMS['stop_loss_pct']*100:.0f}%")
    print(f"  止盈: {STRATEGY_PARAMS['take_profit_pct']*100:.0f}%")

    # 获取股票列表
    stocks = CSI300_STOCKS[:N_STOCKS]
    print(f"\n回测股票: {len(stocks)} 只")

    # 获取数据
    print("\n获取股票数据...")
    stock_data = {}
    for i, symbol in enumerate(stocks, 1):
        print(f"  [{i}/{len(stocks)}] {symbol}...", end=" ")
        df = fetch_stock_data(symbol)
        if not df.empty and len(df) >= 120:
            stock_data[symbol] = df
            print(f"✅ {len(df)} 条")
        else:
            print("❌ 数据不足")

    if not stock_data:
        print("❌ 没有有效数据")
        return

    print(f"\n有效股票: {len(stock_data)} 只")

    # 回测
    print("\n" + "=" * 80)
    print("开始回测...")
    print("=" * 80 + "\n")

    results = []
    total_capital = INITIAL_CAPITAL * len(stock_data)
    start_time = time.time()

    for symbol, df in stock_data.items():
        try:
            strategy = QlibStrategy(**STRATEGY_PARAMS)

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
                'max_drawdown': result.max_drawdown,
                'total_trades': result.total_trades,
            })

            print(f"  {symbol}: 年化 {adjusted_annual*100:.2f}%, 夏普 {result.sharpe_ratio:.2f}, 交易 {result.total_trades} 次")

        except Exception as e:
            print(f"  {symbol}: ❌ 失败")

    elapsed = time.time() - start_time

    # 汇总结果
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)

    successful = [r for r in results if r.get('success')]
    if not successful:
        print("❌ 所有回测失败")
        return

    successful.sort(key=lambda x: x['adjusted_annual_return'], reverse=True)

    # Top 10
    print(f"\n📊 Top 10 股票:")
    print("-" * 80)
    print(f"{'排名':<4} {'代码':<8} {'年化收益':>10} {'夏普':>8} {'回撤':>8} {'交易':>6}")
    print("-" * 80)

    for i, r in enumerate(successful[:10], 1):
        print(f"{i:<4} {r['symbol']:<8} {r['adjusted_annual_return']*100:>9.2f}% "
              f"{r['sharpe_ratio']:>8.2f} {r['max_drawdown']*100:>7.2f}% {r['total_trades']:>6}")

    print("-" * 80)

    # 组合统计
    total_final = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in successful)
    total_return = (total_final - total_capital) / total_capital
    annual_return = (1 + total_return) ** (252 / LOOKBACK_DAYS) - 1

    # 计算真实成本后收益
    total_final_adj = sum(INITIAL_CAPITAL * (1 + r['adjusted_annual_return'] * LOOKBACK_DAYS / 252) for r in successful)
    total_return_adj = (total_final_adj - total_capital) / total_capital
    annual_return_adj = (1 + total_return_adj) ** (252 / LOOKBACK_DAYS) - 1

    print(f"\n📈 组合统计:")
    print(f"   成功股票: {len(successful)}/{len(results)} 只")
    print(f"   组合年化收益 (含真实成本): {annual_return_adj*100:.2f}%")
    print(f"   平均夏普比: {np.mean([r['sharpe_ratio'] for r in successful]):.2f}")
    print(f"   平均回撤: {np.mean([r['max_drawdown'] for r in successful])*100:.2f}%")
    print(f"   总交易次数: {sum(r['total_trades'] for r in successful)}")
    print(f"   每只股票平均交易: {np.mean([r['total_trades'] for r in successful]):.1f} 次")
    print(f"   总耗时: {elapsed:.1f} 秒")

    # 收益分布
    positive = [r for r in successful if r['adjusted_annual_return'] > 0]
    negative = [r for r in successful if r['adjusted_annual_return'] <= 0]

    print(f"\n📊 收益分布:")
    print(f"   盈利股票: {len(positive)} 只 ({len(positive)/len(successful)*100:.1f}%)")
    print(f"   亏损股票: {len(negative)} 只 ({len(negative)/len(successful)*100:.1f}%)")
    if positive:
        print(f"   平均盈利: {np.mean([r['adjusted_annual_return'] for r in positive])*100:.2f}%")
    if negative:
        print(f"   平均亏损: {np.mean([r['adjusted_annual_return'] for r in negative])*100:.2f}%")

    return results


if __name__ == "__main__":
    results = main()