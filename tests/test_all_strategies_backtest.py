#!/usr/bin/env python
"""
多策略回测对比

对指定股票使用所有策略进行回测，比较各策略表现
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
from typing import Dict, List, Any, Type

from quanttool.domain.interfaces.strategy import IStrategy
from quanttool.strategies.ma_cross import MACrossStrategy
from quanttool.strategies.dual_ma import DualMAStrategy
from quanttool.strategies.macd import MACDStrategy
from quanttool.strategies.breakout import BreakoutStrategy
from quanttool.strategies.score_strategy import ScoreStrategy
from quanttool.strategies.trend_strategy import TrendStrategy
from quanttool.strategies.trend_momentum_strategy import TrendMomentumStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher


# 目标股票列表
TARGET_STOCKS = [
    '000876',  # 新希望
    '600515',  # 海航科技
    '688131',  # 皓元医药
    '600600',  # 青岛啤酒
    '600460',  # 士兰微
    '688271',  # 联影医疗
    '001965',  # 招商公路
]

INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365  # 一年


def fetch_stock_data(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """获取股票历史数据"""
    print(f"  获取 {symbol} 数据...")

    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        df = AshareFetcher.get_price(
            code=symbol,
            end_date=end_date,
            count=days + 50,
            frequency='1d'
        )

        if df.empty:
            print(f"  ❌ {symbol} 数据为空")
            return pd.DataFrame()

        # 标准化列名
        if 'timestamp' not in df.columns:
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)

        print(f"  ✅ {symbol}: {len(df)} 条数据")
        return df

    except Exception as e:
        print(f"  ❌ {symbol} 获取失败: {e}")
        return pd.DataFrame()


def get_all_strategies() -> List[tuple]:
    """获取所有策略及其默认参数"""
    return [
        ("MA_Cross(10/30)", MACrossStrategy()),
        ("Dual_MA(5/20)", DualMAStrategy()),
        ("MACD(12/26/9)", MACDStrategy()),
        ("Breakout(20日)", BreakoutStrategy()),
        ("ScoreStrategy(70/50)", ScoreStrategy(buy_threshold=70, sell_threshold=50)),
        ("TrendStrategy(75/50)", TrendStrategy(buy_threshold=75, sell_threshold=50)),
        ("TrendMomentum(55/40)", TrendMomentumStrategy(buy_threshold=55, sell_threshold=40)),
    ]


def run_strategy_backtest(
    strategy: IStrategy,
    strategy_name: str,
    symbol: str,
    df: pd.DataFrame
) -> dict:
    """运行单个策略的回测"""
    engine = BacktestEngine()
    engine.set_initial_cash(INITIAL_CAPITAL)

    data = {symbol: df.copy()}

    try:
        result = engine.run_backtest(
            strategy=strategy,
            data=data,
            start_date=df['timestamp'].iloc[0],
            end_date=df['timestamp'].iloc[-1]
        )

        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'success': True,
            'annual_return': result.annual_return,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
        }

    except Exception as e:
        return {
            'strategy': strategy_name,
            'symbol': symbol,
            'success': False,
            'error': str(e),
            'annual_return': 0,
            'total_return': 0,
        }


def main():
    """主函数"""
    print("=" * 80)
    print("多策略回测对比")
    print("=" * 80)

    # 获取所有策略
    strategies = get_all_strategies()
    print(f"\n策略列表 ({len(strategies)} 个):")
    for name, _ in strategies:
        print(f"  - {name}")

    print(f"\n目标股票 ({len(TARGET_STOCKS)} 只): {TARGET_STOCKS}")
    print(f"初始资金: ${INITIAL_CAPITAL:,.0f}")
    print(f"回测周期: {LOOKBACK_DAYS} 天")

    # 收集所有数据
    all_data = {}
    for symbol in TARGET_STOCKS:
        df = fetch_stock_data(symbol)
        if not df.empty and len(df) >= 60:
            all_data[symbol] = df

    if not all_data:
        print("\n❌ 没有获取到有效数据")
        return

    # 运行所有策略对所有股票的回测
    all_results = []

    print(f"\n{'='*80}")
    print("开始回测...")
    print("=" * 80)

    for strategy_name, strategy in strategies:
        print(f"\n策略: {strategy_name}")
        strategy_results = []

        for symbol, df in all_data.items():
            result = run_strategy_backtest(strategy, strategy_name, symbol, df)
            strategy_results.append(result)

            if result['success']:
                print(
                    f"  {symbol}: 年化收益 {result['annual_return']*100:+.2f}%, "
                    f"交易 {result['total_trades']}, "
                    f"胜率 {result['win_rate']*100:.1f}%"
                )
            else:
                print(f"  {symbol}: ❌ 失败 - {result.get('error', 'Unknown')}")

        all_results.extend(strategy_results)

    # 汇总结果
    print("\n" + "=" * 80)
    print("策略表现汇总 (按平均年化收益排序)")
    print("=" * 80)

    # 按策略汇总
    strategy_summary = {}
    for result in all_results:
        name = result['strategy']
        if name not in strategy_summary:
            strategy_summary[name] = {
                'returns': [],
                'trades': [],
                'win_rates': [],
                'drawdowns': [],
                'sharpes': [],
            }
        if result['success']:
            strategy_summary[name]['returns'].append(result['annual_return'])
            strategy_summary[name]['trades'].append(result['total_trades'])
            strategy_summary[name]['win_rates'].append(result['win_rate'])
            strategy_summary[name]['drawdowns'].append(result['max_drawdown'])
            strategy_summary[name]['sharpes'].append(result['sharpe_ratio'])

    # 计算平均指标并排序
    summary_list = []
    for name, data in strategy_summary.items():
        if data['returns']:
            avg_return = np.mean(data['returns'])
            avg_trades = np.mean(data['trades'])
            avg_win_rate = np.mean(data['win_rates'])
            avg_drawdown = np.mean(data['drawdowns'])
            avg_sharpe = np.mean(data['sharpes'])

            summary_list.append({
                'strategy': name,
                'avg_return': avg_return,
                'avg_trades': avg_trades,
                'avg_win_rate': avg_win_rate,
                'avg_drawdown': avg_drawdown,
                'avg_sharpe': avg_sharpe,
            })

    # 按收益排序
    summary_list.sort(key=lambda x: x['avg_return'], reverse=True)

    # 打印表格
    print(f"\n{'策略':^25} {'平均年化收益':^12} {'平均交易数':^10} {'平均胜率':^10} {'平均夏普':^10} {'平均回撤':^10}")
    print("-" * 80)

    for s in summary_list:
        print(
            f"{s['strategy']:^25} "
            f"{s['avg_return']*100:^12.2f}% "
            f"{s['avg_trades']:^10.1f} "
            f"{s['avg_win_rate']*100:^10.1f}% "
            f"{s['avg_sharpe']:^10.2f} "
            f"{s['avg_drawdown']*100:^10.2f}%"
        )

    # 最佳策略
    if summary_list:
        best = summary_list[0]
        print("\n" + "=" * 80)
        print(f"🏆 最佳策略: {best['strategy']}")
        print(f"   平均年化收益: {best['avg_return']*100:.2f}%")
        print(f"   平均胜率: {best['avg_win_rate']*100:.1f}%")
        print(f"   平均夏普比率: {best['avg_sharpe']:.2f}")
        print("=" * 80)

    # 每只股票的最佳策略
    print("\n每只股票的最佳策略:")
    print("-" * 60)

    for symbol in all_data.keys():
        stock_results = [r for r in all_results if r['symbol'] == symbol and r['success']]
        if stock_results:
            best_for_stock = max(stock_results, key=lambda x: x['annual_return'])
            print(
                f"  {symbol}: {best_for_stock['strategy']} "
                f"(年化 {best_for_stock['annual_return']*100:+.2f}%)"
            )

    return summary_list


if __name__ == "__main__":
    results = main()