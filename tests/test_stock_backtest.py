#!/usr/bin/env python
"""
趋势动量策略股票回测脚本

使用 TrendMomentumStrategy 对指定股票进行过去一年的回测
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

from quanttool.strategies.trend_momentum_strategy import TrendMomentumStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.data_fetcher import AshareFetcher


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

# 策略参数
BUY_THRESHOLD = 55.0
SELL_THRESHOLD = 40.0
INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365  # 一年


def fetch_stock_data(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    获取股票历史数据

    Args:
        symbol: 股票代码
        days: 回溯天数

    Returns:
        DataFrame with OHLCV data
    """
    print(f"  获取 {symbol} 数据...")

    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        df = AshareFetcher.get_price(
            code=symbol,
            end_date=end_date,
            count=days + 50,  # 多获取一些以确保有足够数据
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

        # 只保留需要的数据量
        if len(df) > days:
            df = df.tail(days).reset_index(drop=True)

        print(f"  ✅ {symbol}: {len(df)} 条数据 ({df['timestamp'].min().date()} ~ {df['timestamp'].max().date()})")
        return df

    except Exception as e:
        print(f"  ❌ {symbol} 获取失败: {e}")
        return pd.DataFrame()


def run_single_backtest(
    symbol: str,
    df: pd.DataFrame,
    buy_threshold: float = BUY_THRESHOLD,
    sell_threshold: float = SELL_THRESHOLD
) -> dict:
    """
    对单只股票运行回测

    Args:
        symbol: 股票代码
        df: 历史数据
        buy_threshold: 买入阈值
        sell_threshold: 卖出阈值

    Returns:
        回测结果字典
    """
    # 创建策略
    strategy = TrendMomentumStrategy(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
    )

    # 初始化回测引擎
    engine = BacktestEngine()
    engine.set_initial_cash(INITIAL_CAPITAL)

    # 准备数据
    data = {symbol: df.copy()}

    # 运行回测
    try:
        result = engine.run_backtest(
            strategy=strategy,
            data=data,
            start_date=df['timestamp'].iloc[0],
            end_date=df['timestamp'].iloc[-1]
        )

        return {
            'symbol': symbol,
            'success': True,
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
        }

    except Exception as e:
        print(f"  ❌ {symbol} 回测失败: {e}")
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }


def print_summary(results: list):
    """打印汇总结果"""
    print("\n" + "=" * 70)
    print("回测结果汇总")
    print("=" * 70)

    # 表头
    print(f"\n{'股票代码':^10} {'年化收益':^12} {'总收益':^10} {'交易次数':^8} {'胜率':^8} {'夏普':^8} {'最大回撤':^10}")
    print("-" * 70)

    successful = [r for r in results if r.get('success')]
    failed = [r for r in results if not r.get('success')]

    for r in successful:
        print(
            f"{r['symbol']:^10} "
            f"{r['annual_return']*100:^12.2f}% "
            f"{r['total_return']*100:^10.2f}% "
            f"{r['total_trades']:^8} "
            f"{r['win_rate']*100:^8.1f}% "
            f"{r['sharpe_ratio']:^8.2f} "
            f"{r['max_drawdown']*100:^10.2f}%"
        )

    if failed:
        print("\n失败的股票:")
        for r in failed:
            print(f"  {r['symbol']}: {r.get('error', 'Unknown error')}")

    # 统计汇总
    if successful:
        avg_return = np.mean([r['annual_return'] for r in successful])
        avg_win_rate = np.mean([r['win_rate'] for r in successful])
        avg_trades = np.mean([r['total_trades'] for r in successful])
        total_trades = sum(r['total_trades'] for r in successful)

        print("\n" + "-" * 70)
        print(f"平均年化收益: {avg_return*100:.2f}%")
        print(f"平均胜率: {avg_win_rate*100:.1f}%")
        print(f"平均交易次数: {avg_trades:.1f}")
        print(f"总交易次数: {total_trades}")
        print(f"成功回测: {len(successful)}/{len(results)}")


def main():
    """主函数"""
    print("=" * 70)
    print("趋势动量策略股票回测")
    print("=" * 70)
    print(f"\n目标股票: {TARGET_STOCKS}")
    print(f"策略参数: 买入阈值={BUY_THRESHOLD}, 卖出阈值={SELL_THRESHOLD}")
    print(f"回测周期: 过去 {LOOKBACK_DAYS} 天")
    print(f"初始资金: ${INITIAL_CAPITAL:,.0f}")

    results = []

    # 逐只股票获取数据并回测
    for symbol in TARGET_STOCKS:
        print(f"\n处理 {symbol}...")

        # 获取数据
        df = fetch_stock_data(symbol)

        if df.empty:
            results.append({
                'symbol': symbol,
                'success': False,
                'error': 'No data available'
            })
            continue

        # 检查数据量是否足够 (策略需要至少60天)
        if len(df) < 60:
            print(f"  ❌ {symbol} 数据不足 ({len(df)} < 60)")
            results.append({
                'symbol': symbol,
                'success': False,
                'error': f'Insufficient data ({len(df)} days)'
            })
            continue

        # 运行回测
        result = run_single_backtest(symbol, df)
        results.append(result)

        if result['success']:
            print(
                f"  ✅ 年化收益: {result['annual_return']*100:.2f}%, "
                f"交易: {result['total_trades']}, "
                f"胜率: {result['win_rate']*100:.1f}%"
            )

    # 打印汇总
    print_summary(results)

    return results


if __name__ == "__main__":
    results = main()