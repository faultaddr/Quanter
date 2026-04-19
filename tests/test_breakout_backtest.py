#!/usr/bin/env python
"""
低位盘整突破策略回测验证脚本

使用真实股票数据验证低位盘整突破策略的表现
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quanttool.factors.breakout_scoring_system import BreakoutScoringSystem, analyze_breakout_quality
from quanttool.factors.scoring_system import ScoringSystem
from quanttool.strategies.score_strategy import ScoreStrategy
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import create_data_fetcher_with_credentials


def fetch_real_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """获取真实股票数据"""
    print(f"\n获取 {symbol} 最近 {days} 天数据...")

    fetcher = create_data_fetcher_with_credentials()
    fetcher.initialize()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    data = fetcher.get_bars([symbol], start_date, end_date)

    if symbol not in data or data[symbol].empty:
        print(f"❌ 无法获取 {symbol} 数据")
        return pd.DataFrame()

    df = data[symbol].copy()

    if 'timestamp' not in df.columns:
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"✅ 成功获取 {len(df)} 条数据")
    print(f"   日期范围: {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}")
    print(f"   价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

    return df


def run_breakout_backtest(df: pd.DataFrame, initial_capital: float = 100000) -> dict:
    """
    运行低位盘整突破策略回测

    策略逻辑：
    1. 每日计算评分
    2. 当评分>=70且形态完整时买入
    3. 止损/止盈后卖出
    """
    print("\n" + "="*60)
    print("低位盘整突破策略回测")
    print("="*60)

    system = BreakoutScoringSystem()

    # 交易记录
    trades = []
    cash = initial_capital
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    entry_date = None

    # 回测窗口（需要足够的历史数据）
    lookback = 60

    for i in range(lookback, len(df)):
        current_date = df['timestamp'].iloc[i]
        current_price = df['close'].iloc[i]

        # 获取历史数据
        hist_df = df.iloc[:i+1].copy()

        # 计算评分
        result = system.calculate_score(hist_df)

        # 持仓检查
        if position > 0:
            # 检查止损
            if current_price <= stop_loss:
                # 止损卖出
                sell_value = position * current_price
                pnl = (current_price - entry_price) * position
                cash += sell_value

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': position,
                    'pnl': pnl,
                    'return': pnl / (entry_price * position),
                    'exit_reason': 'stop_loss',
                    'score': result.final_score
                })

                position = 0
                entry_price = 0

            # 检查止盈
            elif current_price >= take_profit:
                # 止盈卖出
                sell_value = position * current_price
                pnl = (current_price - entry_price) * position
                cash += sell_value

                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'shares': position,
                    'pnl': pnl,
                    'return': pnl / (entry_price * position),
                    'exit_reason': 'take_profit',
                    'score': result.final_score
                })

                position = 0
                entry_price = 0

        # 无持仓时检查买入信号
        if position == 0 and result.passed_filter and result.final_score >= 70:
            # 买入
            shares = int(cash * 0.95 / current_price / 100) * 100  # 整手
            if shares > 0:
                cost = shares * current_price
                cash -= cost
                position = shares
                entry_price = current_price
                stop_loss = result.stop_loss_price
                take_profit = result.take_profit_price
                entry_date = current_date

    # 最后如果还持仓，按收盘价平仓
    if position > 0:
        final_price = df['close'].iloc[-1]
        pnl = (final_price - entry_price) * position
        cash += position * final_price

        trades.append({
            'entry_date': entry_date,
            'exit_date': df['timestamp'].iloc[-1],
            'entry_price': entry_price,
            'exit_price': final_price,
            'shares': position,
            'pnl': pnl,
            'return': pnl / (entry_price * position),
            'exit_reason': 'end_of_test',
            'score': 0
        })

    # 计算绩效
    total_return = (cash - initial_capital) / initial_capital

    # 计算其他指标
    if trades:
        returns = [t['return'] for t in trades]
        win_trades = [t for t in trades if t['pnl'] > 0]
        lose_trades = [t for t in trades if t['pnl'] <= 0]

        win_rate = len(win_trades) / len(trades) if trades else 0
        avg_return = np.mean(returns) if returns else 0

        # 计算最大回撤
        equity_curve = [initial_capital]
        for t in trades:
            equity_curve.append(equity_curve[-1] + t['pnl'])

        peak = initial_capital
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
    else:
        win_rate = 0
        avg_return = 0
        max_dd = 0

    return {
        'initial_capital': initial_capital,
        'final_capital': cash,
        'total_return': total_return,
        'total_trades': len(trades),
        'win_rate': win_rate,
        'avg_return': avg_return,
        'max_drawdown': max_dd,
        'trades': trades
    }


def run_benchmark_backtest(df: pd.DataFrame, initial_capital: float = 100000) -> dict:
    """
    运行基准策略（买入持有）回测
    """
    if len(df) < 60:
        return {
            'initial_capital': initial_capital,
            'final_capital': initial_capital,
            'total_return': 0
        }

    first_price = df['close'].iloc[60]  # 与策略相同起点
    last_price = df['close'].iloc[-1]

    shares = int(initial_capital * 0.95 / first_price / 100) * 100
    final_value = shares * last_price
    remaining_cash = initial_capital - shares * first_price

    total_return = (final_value + remaining_cash - initial_capital) / initial_capital

    return {
        'initial_capital': initial_capital,
        'final_capital': final_value + remaining_cash,
        'total_return': total_return
    }


def main():
    """主函数"""
    print("="*60)
    print("低位盘整突破策略回测验证")
    print("="*60)

    # 测试股票列表
    test_stocks = [
        ('600519.SH', '贵州茅台'),
        ('000001.SZ', '平安银行'),
        ('000858.SZ', '五粮液'),
        ('601318.SH', '中国平安'),
        ('600036.SH', '招商银行'),
    ]

    results = []

    for symbol, name in test_stocks:
        print(f"\n{'='*60}")
        print(f"测试股票: {symbol} ({name})")
        print('='*60)

        df = fetch_real_data(symbol, days=365)

        if df.empty or len(df) < 100:
            print(f"跳过 {symbol}: 数据不足")
            continue

        # 运行低位盘整突破策略
        breakout_result = run_breakout_backtest(df)

        # 运行基准策略
        benchmark_result = run_benchmark_backtest(df)

        print(f"\n📊 回测结果:")
        print(f"  策略总收益: {breakout_result['total_return']*100:.2f}%")
        print(f"  基准收益: {benchmark_result['total_return']*100:.2f}%")
        print(f"  超额收益: {(breakout_result['total_return'] - benchmark_result['total_return'])*100:.2f}%")
        print(f"  总交易次数: {breakout_result['total_trades']}")
        print(f"  胜率: {breakout_result['win_rate']*100:.1f}%")
        print(f"  平均收益: {breakout_result['avg_return']*100:.2f}%")
        print(f"  最大回撤: {breakout_result['max_drawdown']*100:.2f}%")

        results.append({
            'symbol': symbol,
            'name': name,
            'breakout_return': breakout_result['total_return'],
            'benchmark_return': benchmark_result['total_return'],
            'excess_return': breakout_result['total_return'] - benchmark_result['total_return'],
            'trades': breakout_result['total_trades'],
            'win_rate': breakout_result['win_rate'],
            'max_dd': breakout_result['max_drawdown']
        })

    # 汇总统计
    if results:
        print("\n" + "="*60)
        print("📈 汇总统计")
        print("="*60)

        avg_breakout_return = np.mean([r['breakout_return'] for r in results])
        avg_benchmark_return = np.mean([r['benchmark_return'] for r in results])
        avg_excess_return = np.mean([r['excess_return'] for r in results])
        avg_win_rate = np.mean([r['win_rate'] for r in results])

        print(f"\n平均策略收益: {avg_breakout_return*100:.2f}%")
        print(f"平均基准收益: {avg_benchmark_return*100:.2f}%")
        print(f"平均超额收益: {avg_excess_return*100:.2f}%")
        print(f"平均胜率: {avg_win_rate*100:.1f}%")

        # 统计盈利股票
        profitable = sum(1 for r in results if r['excess_return'] > 0)
        print(f"\n战胜基准股票数: {profitable}/{len(results)}")

        # 生成报告
        print("\n" + "="*60)
        print("📋 详细结果表")
        print("="*60)
        print(f"{'股票':<12} {'名称':<8} {'策略收益':<12} {'基准收益':<12} {'超额':<10} {'交易次数':<8} {'胜率':<8}")
        print("-"*60)
        for r in results:
            print(f"{r['symbol']:<12} {r['name']:<8} {r['breakout_return']*100:>10.2f}% {r['benchmark_return']*100:>10.2f}% {r['excess_return']*100:>8.2f}% {r['trades']:<8} {r['win_rate']*100:>6.1f}%")


if __name__ == '__main__':
    main()