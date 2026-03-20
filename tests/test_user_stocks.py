#!/usr/bin/env python
"""
指定股票回测

使用 LightGBM 策略回测用户指定股票
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

# LightGBM 最佳策略参数
STRATEGY_PARAMS = {
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
    except Exception as e:
        print(f"  ❌ {symbol} 获取失败: {e}")
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


def run_backtest():
    """运行回测"""
    print("=" * 80)
    print("LightGBM 策略回测")
    print("=" * 80)
    print(f"\n回测股票: {TARGET_STOCKS}")
    print(f"回测周期: {LOOKBACK_DAYS} 天")
    print(f"初始资金: ¥{INITIAL_CAPITAL:,.0f} / 只")
    print()

    # 获取数据
    print("获取股票数据...")
    stock_data = {}
    for symbol in TARGET_STOCKS:
        df = fetch_stock_data(symbol)
        if not df.empty and len(df) >= 120:
            stock_data[symbol] = df
            print(f"  ✅ {symbol} ({get_stock_name(symbol)}): {len(df)} 条")
        else:
            print(f"  ❌ {symbol} 数据不足")

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
        print(f"\n{'='*60}")
        print(f"📊 {symbol} ({get_stock_name(symbol)})")
        print(f"{'='*60}")

        try:
            # 创建策略
            strategy = QlibStrategy(
                model_type='lgb',
                feature_set='Alpha158',
                buy_threshold=0.55,
                sell_threshold=0.45,
                stop_loss_pct=0.05,
                take_profit_pct=0.12,
                epochs=50,
            )

            # 训练模型
            print("训练模型中...")
            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
            strategy.train_model(train_data, horizon=10)

            # 回测
            print("运行回测...")
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

            print(f"\n📈 回测结果:")
            print(f"   年化收益: {result.annual_return*100:.2f}%")
            print(f"   总收益: {result.total_return*100:.2f}%")
            print(f"   夏普比率: {result.sharpe_ratio:.2f}")
            print(f"   最大回撤: {result.max_drawdown*100:.2f}%")
            print(f"   胜率: {result.win_rate*100:.1f}%")
            print(f"   交易次数: {result.total_trades}")

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
            print(f"\n❌ 回测失败: {e}")
            results.append({
                'symbol': symbol,
                'name': get_stock_name(symbol),
                'success': False,
                'error': str(e),
            })

    elapsed = time.time() - start_time

    # 汇总结果
    print("\n" + "=" * 80)
    print("回测汇总")
    print("=" * 80)

    successful = [r for r in results if r.get('success')]
    if not successful:
        print("❌ 所有回测失败")
        return

    # 排序
    successful.sort(key=lambda x: x['annual_return'], reverse=True)

    print(f"\n📊 收益排名:")
    print("-" * 80)
    print(f"{'排名':<4} {'代码':<8} {'名称':<10} {'年化收益':>10} {'夏普':>8} {'胜率':>8} {'交易':>6}")
    print("-" * 80)

    for i, r in enumerate(successful, 1):
        print(f"{i:<4} {r['symbol']:<8} {r['name']:<10} {r['annual_return']*100:>9.2f}% "
              f"{r['sharpe_ratio']:>8.2f} {r['win_rate']*100:>7.1f}% {r['total_trades']:>6}")

    print("-" * 80)

    # 组合统计
    total_final = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in successful)
    total_return = (total_final - total_capital) / total_capital
    annual_return = (1 + total_return) ** (252 / LOOKBACK_DAYS) - 1

    print(f"\n📈 组合统计:")
    print(f"   成功股票: {len(successful)}/{len(results)} 只")
    print(f"   组合年化收益: {annual_return*100:.2f}%")
    print(f"   组合总收益: {total_return*100:.2f}%")
    print(f"   平均夏普比: {np.mean([r['sharpe_ratio'] for r in successful]):.2f}")
    print(f"   平均回撤: {np.mean([r['max_drawdown'] for r in successful])*100:.2f}%")
    print(f"   平均胜率: {np.mean([r['win_rate'] for r in successful])*100:.1f}%")
    print(f"   总耗时: {elapsed:.1f} 秒")

    # 最佳股票
    if successful:
        best = successful[0]
        print(f"\n🏆 最佳股票: {best['symbol']} ({best['name']})")
        print(f"   年化收益: {best['annual_return']*100:.2f}%")
        print(f"   夏普比率: {best['sharpe_ratio']:.2f}")

    return results


if __name__ == "__main__":
    results = run_backtest()