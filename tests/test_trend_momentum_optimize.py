#!/usr/bin/env python
"""
趋势动量策略参数优化

目标: 年化收益 > 15%
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

from quanttool.strategies.trend_momentum_strategy import TrendMomentumStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher


TARGET_STOCKS = ['000876', '600515', '688131', '600600', '600460', '688271', '001965']
INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365
TARGET_RETURN = 0.15  # 15%


def fetch_stock_data(symbol: str) -> pd.DataFrame:
    """获取股票历史数据"""
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        df = AshareFetcher.get_price(
            code=symbol,
            end_date=end_date,
            count=LOOKBACK_DAYS + 50,
            frequency='1d'
        )

        if df.empty:
            return pd.DataFrame()

        if 'timestamp' not in df.columns:
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)

        if len(df) > LOOKBACK_DAYS:
            df = df.tail(LOOKBACK_DAYS).reset_index(drop=True)

        return df

    except Exception as e:
        return pd.DataFrame()


def run_backtest(symbol: str, df: pd.DataFrame, buy_th: float, sell_th: float,
                 stop_loss: float = 0.07, take_profit: float = 0.15) -> dict:
    """运行单次回测"""
    strategy = TrendMomentumStrategy(
        buy_threshold=buy_th,
        sell_threshold=sell_th,
        stop_loss_pct=stop_loss,
        take_profit_pct=take_profit,
    )
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
            'success': True,
            'annual_return': result.annual_return,
            'total_return': result.total_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
        }
    except Exception as e:
        return {'success': False, 'error': str(e), 'annual_return': -1}


def optimize_single_stock(symbol: str, df: pd.DataFrame) -> dict:
    """对单只股票进行参数优化"""
    print(f"\n优化 {symbol}...")

    # 扩展参数搜索范围
    buy_thresholds = [45, 50, 55, 60, 65, 70, 75]
    sell_thresholds = [25, 30, 35, 40, 45, 50]
    stop_losses = [0.05, 0.07, 0.10, 0.12]
    take_profits = [0.10, 0.15, 0.20, 0.25, 0.30]

    best_result = None
    best_return = -float('inf')
    best_params = None
    tested = 0

    # 网格搜索
    for buy_th, sell_th, stop_loss, take_profit in product(
        buy_thresholds, sell_thresholds, stop_losses, take_profits
    ):
        # 买入阈值必须大于卖出阈值
        if buy_th <= sell_th:
            continue

        result = run_backtest(symbol, df, buy_th, sell_th, stop_loss, take_profit)
        tested += 1

        if result['success'] and result['annual_return'] > best_return:
            best_return = result['annual_return']
            best_result = result
            best_params = {
                'buy_threshold': buy_th,
                'sell_threshold': sell_th,
                'stop_loss_pct': stop_loss,
                'take_profit_pct': take_profit,
            }

            # 如果达到目标，提前退出
            if best_return >= TARGET_RETURN:
                print(f"  ✅ 达到目标! 年化收益 {best_return*100:.2f}%")
                break

    print(f"  测试了 {tested} 组参数")
    if best_result:
        print(f"  最佳年化收益: {best_return*100:.2f}%")
        print(f"  最佳参数: 买入={best_params['buy_threshold']}, "
              f"卖出={best_params['sell_threshold']}, "
              f"止损={best_params['stop_loss_pct']*100}%, "
              f"止盈={best_params['take_profit_pct']*100}%")

    return {
        'symbol': symbol,
        'best_return': best_return,
        'best_result': best_result,
        'best_params': best_params,
        'reached_target': best_return >= TARGET_RETURN,
    }


def main():
    print("=" * 70)
    print("趋势动量策略参数优化 - 目标年化收益 > 15%")
    print("=" * 70)

    # 收集数据
    all_data = {}
    for symbol in TARGET_STOCKS:
        df = fetch_stock_data(symbol)
        if not df.empty and len(df) >= 60:
            all_data[symbol] = df
            print(f"  ✅ {symbol}: {len(df)} 条数据")

    if not all_data:
        print("\n❌ 没有有效数据")
        return

    # 优化每只股票
    results = []
    for symbol, df in all_data.items():
        result = optimize_single_stock(symbol, df)
        results.append(result)

    # 汇总结果
    print("\n" + "=" * 70)
    print("📊 优化结果汇总")
    print("=" * 70)

    results.sort(key=lambda x: x['best_return'], reverse=True)

    print(f"\n{'排名':^4} {'股票':^8} {'最佳年化收益':^12} {'是否达标':^8} {'买入阈值':^8} {'卖出阈值':^8}")
    print("-" * 70)

    for i, r in enumerate(results, 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        status = "✅ 达标" if r['reached_target'] else "❌ 未达标"
        params = r['best_params'] or {}

        print(
            f"{medal}{i:^2} {r['symbol']:^8} "
            f"{r['best_return']*100:^12.2f}% "
            f"{status:^8} "
            f"{params.get('buy_threshold', '-'):^8} "
            f"{params.get('sell_threshold', '-'):^8}"
        )

    # 统计
    reached = [r for r in results if r['reached_target']]
    print("\n" + "-" * 70)
    print(f"达标股票数: {len(reached)}/{len(results)}")

    if reached:
        print("\n✅ 达到 15% 年化收益的股票:")
        for r in reached:
            p = r['best_params']
            print(f"  {r['symbol']}: 年化 {r['best_return']*100:.2f}%")
            print(f"    参数: 买入={p['buy_threshold']}, 卖出={p['sell_threshold']}, "
                  f"止损={p['stop_loss_pct']*100}%, 止盈={p['take_profit_pct']*100}%")
    else:
        print("\n⚠️ 没有股票达到 15% 年化收益目标")
        print("建议: 可能需要换股或调整策略逻辑")

    return results


if __name__ == "__main__":
    results = main()