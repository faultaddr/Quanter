#!/usr/bin/env python
"""
趋势策略详细回测报告

对指定股票使用 TrendStrategy 进行回测
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime

from quanttool.strategies.trend_strategy import TrendStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher


TARGET_STOCKS = ['000876', '600515', '688131', '600600', '600460', '688271', '001965']
INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365


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
        print(f"  ❌ {symbol} 获取失败: {e}")
        return pd.DataFrame()


def run_backtest(symbol: str, df: pd.DataFrame, buy_th: float, sell_th: float) -> dict:
    """运行单次回测"""
    strategy = TrendStrategy(buy_threshold=buy_th, sell_threshold=sell_th)
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


def main():
    print("=" * 70)
    print("趋势策略(TrendStrategy)详细回测报告")
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

    # 参数优化范围
    param_grid = [
        (70, 40),
        (75, 45),
        (75, 50),
        (80, 50),
        (80, 55),
        (85, 55),
        (85, 60),
    ]

    # 存储所有结果
    all_results = []

    print(f"\n{'='*70}")
    print("参数优化回测...")
    print("=" * 70)

    for symbol, df in all_data.items():
        print(f"\n{symbol}:")
        best_result = None
        best_return = -float('inf')

        for buy_th, sell_th in param_grid:
            result = run_backtest(symbol, df, buy_th, sell_th)

            if result['success']:
                ret = result['annual_return']
                if ret > best_return:
                    best_return = ret
                    best_result = {**result, 'buy_th': buy_th, 'sell_th': sell_th}

                print(
                    f"  参数({buy_th}/{sell_th}): 年化 {ret*100:+.2f}%, "
                    f"交易 {result['total_trades']}, "
                    f"胜率 {result['win_rate']*100:.1f}%"
                )

        if best_result:
            all_results.append({
                'symbol': symbol,
                **best_result
            })

    # 排序输出
    all_results.sort(key=lambda x: x['annual_return'], reverse=True)

    print("\n" + "=" * 70)
    print("📊 回测结果排名（按年化收益）")
    print("=" * 70)

    print(f"\n{'排名':^4} {'股票':^8} {'年化收益':^10} {'总收益':^10} {'交易次数':^8} {'胜率':^8} {'夏普':^8} {'参数':^10}")
    print("-" * 70)

    for i, r in enumerate(all_results, 1):
        medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else "  "))
        print(
            f"{medal}{i:^2} {r['symbol']:^8} "
            f"{r['annual_return']*100:^+10.2f}% "
            f"{r['total_return']*100:^+10.2f}% "
            f"{r['total_trades']:^8} "
            f"{r['win_rate']*100:^8.1f}% "
            f"{r['sharpe_ratio']:^8.2f} "
            f"({r['buy_th']}/{r['sell_th']})"
        )

    # 最佳股票详情
    if all_results:
        best = all_results[0]
        print("\n" + "=" * 70)
        print(f"🏆 最佳表现股票: {best['symbol']}")
        print("=" * 70)
        print(f"  年化收益率: {best['annual_return']*100:.2f}%")
        print(f"  总收益率:   {best['total_return']*100:.2f}%")
        print(f"  夏普比率:   {best['sharpe_ratio']:.2f}")
        print(f"  最大回撤:   {best['max_drawdown']*100:.2f}%")
        print(f"  交易次数:   {best['total_trades']}")
        print(f"  胜率:       {best['win_rate']*100:.1f}%")
        print(f"  最佳参数:   买入阈值={best['buy_th']}, 卖出阈值={best['sell_th']}")

    return all_results


if __name__ == "__main__":
    results = main()