#!/usr/bin/env python
"""
沪深300 Top20 回测脚本

使用趋势动量评分系统选出的 Top20 股票进行回测
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from quanttool.strategies.trend_momentum_strategy import TrendMomentumStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher


# 沪深300 Top20 股票（2026-03-07 扫描结果）
TOP20_STOCKS = [
    '600900',  # 长江电力 - 评分 92.0
    '601225',  # 陕西煤业 - 评分 86.0
    '600519',  # 贵州茅台 - 评分 84.0
    '601088',  # 中国神华 - 评分 84.0
    '601288',  # 农业银行 - 评分 84.0
    '600482',  # 中国动力 - 评分 84.0
    '601669',  # 中国电建 - 评分 80.0
    '601919',  # 中远海控 - 评分 78.0
    '002384',  # 东山精密 - 评分 77.0
    '600875',  # 东方电气 - 评分 77.0
    '600989',  # 宝丰能源 - 评分 77.0
    '601877',  # 正泰电器 - 评分 77.0
    '600938',  # 中国海油 - 评分 75.0
    '601857',  # 中国石油 - 评分 75.0
    '600026',  # 中远海能 - 评分 73.0
    '601898',  # 中煤能源 - 评分 73.0
    '002714',  # 牧原股份 - 评分 72.0
    '000895',  # 双汇发展 - 评分 70.0
    '000999',  # 华润三九 - 评分 70.0
    '600028',  # 中国石化 - 评分 70.0
]

# 策略参数
INITIAL_CAPITAL = 50000.0  # 每只股票初始资金 5 万
LOOKBACK_DAYS = 365  # 回测周期 1 年

# 趋势动量策略参数（优化版）
STRATEGY_PARAMS = {
    'buy_threshold': 60,      # 买入评分阈值（降低门槛）
    'sell_threshold': 40,     # 卖出评分阈值
    'stop_loss_pct': 0.05,    # 止损 5%（更紧止损）
    'take_profit_pct': 0.12,  # 止盈 12%
    'commission': 0.0003,     # 手续费
}


def fetch_stock_data(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """获取股票历史数据"""
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        df = AshareFetcher.get_price(
            code=symbol,
            end_date=end_date,
            count=days + 100,
            frequency='1d'
        )

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

    except Exception as e:
        print(f"  ❌ {symbol} 获取失败: {e}")
        return pd.DataFrame()


def run_backtest(stock_data: dict, params: dict) -> dict:
    """运行回测"""
    results = []
    total_capital = INITIAL_CAPITAL * len(stock_data)

    for symbol, df in stock_data.items():
        if len(df) < 120:
            print(f"  ⚠️ {symbol} 数据不足 ({len(df)} 条)，跳过")
            continue

        try:
            # 创建趋势动量策略
            strategy = TrendMomentumStrategy(
                buy_threshold=params.get('buy_threshold', 70),
                sell_threshold=params.get('sell_threshold', 50),
                stop_loss_pct=params.get('stop_loss_pct', 0.07),
                take_profit_pct=params.get('take_profit_pct', 0.15),
            )

            # 回测
            engine = BacktestEngine()
            engine.set_initial_cash(INITIAL_CAPITAL)
            engine.set_commission_rate(params.get('commission', 0.0003))
            engine.set_t_plus_1(True)

            data = {symbol: df.copy()}
            result = engine.run_backtest(
                strategy=strategy,
                data=data,
                start_date=df['timestamp'].iloc[0],
                end_date=df['timestamp'].iloc[-1]
            )

            results.append({
                'symbol': symbol,
                'success': True,
                'annual_return': result.annual_return,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
            })

            print(f"  ✅ {symbol}: 年化 {result.annual_return*100:.2f}%, "
                  f"夏普 {result.sharpe_ratio:.2f}, "
                  f"胜率 {result.win_rate*100:.1f}%, "
                  f"交易 {result.total_trades} 次")

        except Exception as e:
            print(f"  ❌ {symbol} 回测失败: {e}")
            results.append({
                'symbol': symbol,
                'success': False,
                'error': str(e),
                'annual_return': 0,
                'total_return': 0,
            })

    # 计算组合收益
    successful = [r for r in results if r.get('success')]
    if not successful:
        return {
            'success': False,
            'error': 'All backtests failed',
            'annual_return': 0,
        }

    total_final = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in successful)
    total_return = (total_final - total_capital) / total_capital
    annual_return = (1 + total_return) ** (252 / LOOKBACK_DAYS) - 1

    return {
        'success': True,
        'annual_return': annual_return,
        'total_return': total_return,
        'avg_sharpe': np.mean([r['sharpe_ratio'] for r in successful]),
        'avg_drawdown': np.mean([r['max_drawdown'] for r in successful]),
        'avg_win_rate': np.mean([r['win_rate'] for r in successful]),
        'total_trades': sum(r['total_trades'] for r in successful),
        'n_stocks': len(successful),
        'results': results,
    }


def main():
    """主函数"""
    print("=" * 80)
    print("沪深300 Top20 回测")
    print("策略: 趋势动量评分")
    print("=" * 80)

    # 获取股票数据
    print(f"\n获取股票数据 (回测周期: {LOOKBACK_DAYS} 天)...")
    stock_data = {}
    for symbol in TOP20_STOCKS:
        print(f"  获取 {symbol}...", end=" ")
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

    # 运行回测
    print("\n" + "=" * 80)
    print("运行回测...")
    print("=" * 80)

    result = run_backtest(stock_data, STRATEGY_PARAMS)

    if not result.get('success'):
        print(f"\n❌ 回测失败: {result.get('error', 'Unknown')}")
        return

    # 打印结果
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)

    successful_results = [r for r in result['results'] if r.get('success')]
    successful_results.sort(key=lambda x: x['annual_return'], reverse=True)

    print("\n📊 股票收益排名:")
    print("-" * 80)
    print(f"{'排名':<4} {'代码':<8} {'年化收益':>10} {'夏普':>8} {'胜率':>8} {'交易次数':>8}")
    print("-" * 80)

    for i, r in enumerate(successful_results, 1):
        print(f"{i:<4} {r['symbol']:<8} {r['annual_return']*100:>9.2f}% "
              f"{r['sharpe_ratio']:>8.2f} {r['win_rate']*100:>7.1f}% "
              f"{r['total_trades']:>8}")

    print("-" * 80)

    # 组合统计
    print(f"\n📈 组合统计:")
    print(f"   股票数量: {result['n_stocks']} 只")
    print(f"   组合年化收益: {result['annual_return']*100:.2f}%")
    print(f"   组合总收益: {result['total_return']*100:.2f}%")
    print(f"   平均夏普比: {result['avg_sharpe']:.2f}")
    print(f"   平均回撤: {result['avg_drawdown']*100:.2f}%")
    print(f"   平均胜率: {result['avg_win_rate']*100:.1f}%")
    print(f"   总交易次数: {result['total_trades']} 次")

    # 最佳和最差
    if successful_results:
        best = successful_results[0]
        worst = successful_results[-1]
        print(f"\n🏆 最佳股票: {best['symbol']} - 年化 {best['annual_return']*100:.2f}%")
        print(f"📉 最差股票: {worst['symbol']} - 年化 {worst['annual_return']*100:.2f}%")

    return result


if __name__ == "__main__":
    results = main()