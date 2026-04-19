#!/usr/bin/env python
"""
Qlib 策略回测脚本

使用微软 Qlib 的 Alpha158 特征集进行股票回测
目标: 年化收益率 > 15%
"""
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quanttool.strategies.qlib_strategy import QlibStrategy, QlibFeatureEngineer
from quanttool.backtest.engine import BacktestEngine
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

# 默认策略参数（基于文献和经验的固定参数，不做优化）
DEFAULT_PARAMS = {
    'feature_set': 'Alpha158',
    'model_type': 'lgb',
    'buy_threshold': 0.55,
    'sell_threshold': 0.45,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.10,
    'commission': 0.0003,  # 交易成本：万三
}

INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365 * 3  # 3 年数据，支持更稳健的模型训练


def fetch_stock_data(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """获取股票历史数据"""
    # 留出一年做回测，不使用最新数据
    end_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    try:
        df = AshareFetcher.get_price(
            code=symbol,
            end_date=end_date,
            count=days + 100,  # 多获取一些以确保有足够数据
            frequency='1d'
        )

        if df.empty:
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

        return df

    except Exception as e:
        print(f"  ❌ {symbol} 获取失败: {e}")
        return pd.DataFrame()


def run_single_backtest(
    symbol: str,
    df: pd.DataFrame,
    params: dict
) -> dict:
    """对单只股票运行回测"""
    # 创建策略
    strategy = QlibStrategy(
        feature_set=params.get('feature_set', 'Alpha158'),
        model_type=params.get('model_type', 'lgb'),
        buy_threshold=params.get('buy_threshold', 0.55),
        sell_threshold=params.get('sell_threshold', 0.45),
        stop_loss_pct=params.get('stop_loss_pct', 0.05),
        take_profit_pct=params.get('take_profit_pct', 0.10),
    )

    # 训练模型 (使用滚动窗口：前 70% 数据训练，后 30% 测试)
    train_size = int(len(df) * 0.7)
    train_data = df.iloc[:train_size]
    strategy.train_model(train_data, horizon=10)

    # 初始化回测引擎
    engine = BacktestEngine()
    engine.set_initial_cash(INITIAL_CAPITAL)
    # 设置交易成本
    commission = params.get('commission', 0.0003)
    engine.set_commission_rate(commission)
    # 启用 A 股 T+1 规则
    engine.set_t_plus_1(True)

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
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e)
        }


def run_portfolio_backtest(
    stock_data: dict,
    params: dict
) -> dict:
    """组合回测"""
    all_results = []
    total_capital = INITIAL_CAPITAL * len(stock_data)

    for symbol, df in stock_data.items():
        if len(df) < 120:
            continue
        result = run_single_backtest(symbol, df, params)
        all_results.append(result)

    # 计算组合收益
    successful = [r for r in all_results if r.get('success')]
    if not successful:
        return {'success': False, 'error': 'All backtests failed'}

    total_final = sum(r['final_capital'] for r in successful)
    total_return = (total_final - total_capital) / total_capital

    # 年化收益
    trading_days = LOOKBACK_DAYS
    annual_return = (1 + total_return) ** (252 / trading_days) - 1

    # 平均夏普和最大回撤
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in successful])
    avg_drawdown = np.mean([r['max_drawdown'] for r in successful])
    avg_win_rate = np.mean([r['win_rate'] for r in successful])
    total_trades = sum(r['total_trades'] for r in successful)

    return {
        'success': True,
        'total_return': total_return,
        'annual_return': annual_return,
        'avg_sharpe': avg_sharpe,
        'avg_drawdown': avg_drawdown,
        'avg_win_rate': avg_win_rate,
        'total_trades': total_trades,
        'n_stocks': len(successful),
        'individual_results': successful,
    }




def main():
    """主函数 - 使用固定参数进行回测，不做参数优化"""
    print("=" * 70)
    print("Qlib 策略回测 - 使用固定参数（基于文献/经验）")
    print("=" * 70)

    # 获取所有股票数据
    print("\n获取股票数据...")
    stock_data = {}
    for symbol in TARGET_STOCKS:
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

    print(f"\n有效股票: {list(stock_data.keys())}")
    print(f"\n使用固定参数:")
    print(f"  买入阈值: {DEFAULT_PARAMS['buy_threshold']}")
    print(f"  卖出阈值: {DEFAULT_PARAMS['sell_threshold']}")
    print(f"  止损比例: {DEFAULT_PARAMS['stop_loss_pct']*100}%")
    print(f"  止盈比例: {DEFAULT_PARAMS['take_profit_pct']*100}%")
    print(f"  交易成本: {DEFAULT_PARAMS['commission']*10000:.1f}bp")

    # 使用固定参数运行回测（不做参数优化）
    print(f"\n{'='*70}")
    print("运行回测...")
    print(f"{'='*70}")

    result = run_portfolio_backtest(stock_data, DEFAULT_PARAMS)

    if result.get('success'):
        print_result_summary(result, DEFAULT_PARAMS)
    else:
        print(f"回测失败: {result.get('error')}")

    return result


def print_result_summary(result: dict, params: dict):
    """打印结果汇总"""
    print("\n策略参数:")
    print(f"  买入阈值: {params.get('buy_threshold')}")
    print(f"  卖出阈值: {params.get('sell_threshold')}")
    print(f"  止损比例: {params.get('stop_loss_pct')*100}%")
    print(f"  止盈比例: {params.get('take_profit_pct')*100}%")
    print(f"  交易成本: {params.get('commission', 0.0003)*10000:.1f}bp (万{params.get('commission', 0.0003)*10000:.1f})")
    print(f"  T+1规则: 已启用 (A股)")

    print("\n组合表现:")
    print(f"  年化收益: {result['annual_return']*100:.2f}%")
    print(f"  总收益: {result['total_return']*100:.2f}%")
    print(f"  平均夏普: {result['avg_sharpe']:.2f}")
    print(f"  平均回撤: {result['avg_drawdown']*100:.2f}%")
    print(f"  平均胜率: {result['avg_win_rate']*100:.1f}%")
    print(f"  交易次数: {result['total_trades']}")

    print("\n各股票详情:")
    for r in result.get('individual_results', []):
        print(f"  {r['symbol']}: 年化 {r['annual_return']*100:.2f}%, "
              f"交易 {r['total_trades']}, 胜率 {r['win_rate']*100:.1f}%")


if __name__ == "__main__":
    result = main()