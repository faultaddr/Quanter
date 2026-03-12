#!/usr/bin/env python
"""
PyTorch 深度学习模型回测脚本

测试 PyTorch 序列模型和高级模型:
- 序列模型: LSTM, GRU, ALSTM, Transformer, TCN, Localformer
- 高级模型: GATs, SFM, TabNet, ADARNN, ADD, HIST, IGMTF, KRNN, TRA, TCTS, Sandwich
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 关键：先导入 PyTorch 并初始化，避免 segmentation fault
import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from quanttool.strategies.qlib_strategy import QlibStrategy
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

# PyTorch 序列模型 (先只测试核心模型)
PYTORCH_SEQ_MODELS = ['lstm']  # 先只测试 LSTM

# PyTorch 高级模型
PYTORCH_ADV_MODELS = ['gats', 'sfm', 'tabnet', 'adarnn', 'add', 'hist', 'igmtf', 'krnn', 'tra', 'tcts', 'sandwich']

# 策略参数
DEFAULT_PARAMS = {
    'feature_set': 'Alpha158',
    'buy_threshold': 0.55,
    'sell_threshold': 0.45,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.10,
    'commission': 0.0003,
    'device': 'cpu',  # 强制使用 CPU 避免 MPS 崩溃
}

INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365 * 2  # 2年数据
EPOCHS = 10  # PyTorch 训练轮数（减少以加快测试）


def fetch_stock_data(symbol: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """获取股票历史数据"""
    end_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

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


def run_single_model_backtest(
    model_type: str,
    stock_data: dict,
    params: dict,
    epochs: int = EPOCHS,
) -> dict:
    """使用单个模型运行回测"""
    import gc
    import torch

    results = []
    total_capital = INITIAL_CAPITAL * len(stock_data)

    for symbol, df in stock_data.items():
        if len(df) < 120:
            continue

        try:
            print(f"    训练 {symbol}...", end=" ")

            # 强制清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 创建策略
            strategy = QlibStrategy(
                feature_set=params.get('feature_set', 'Alpha158'),
                model_type=model_type,
                buy_threshold=params.get('buy_threshold', 0.55),
                sell_threshold=params.get('sell_threshold', 0.45),
                stop_loss_pct=params.get('stop_loss_pct', 0.05),
                take_profit_pct=params.get('take_profit_pct', 0.10),
                epochs=epochs,
                hidden_size=64,
                num_layers=2,
                device=params.get('device', 'cpu'),  # 强制 CPU 避免 MPS 崩溃
            )

            # 训练模型
            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
            strategy.train_model(train_data, horizon=10)

            print("回测...", end=" ")

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

            print(f"✅ {result.annual_return*100:.1f}%")

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

            # 清理
            del strategy
            del engine
            gc.collect()

        except Exception as e:
            print(f"❌ {e}")
            results.append({
                'symbol': symbol,
                'success': False,
                'error': str(e),
                'annual_return': 0,
                'total_return': 0,
            })
            gc.collect()

    # 计算组合收益
    successful = [r for r in results if r.get('success')]
    if not successful:
        return {
            'model_type': model_type,
            'success': False,
            'error': 'All backtests failed',
            'annual_return': 0,
        }

    total_final = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in successful)
    total_return = (total_final - total_capital) / total_capital
    annual_return = (1 + total_return) ** (252 / LOOKBACK_DAYS) - 1

    return {
        'model_type': model_type,
        'success': True,
        'annual_return': annual_return,
        'total_return': total_return,
        'avg_sharpe': np.mean([r['sharpe_ratio'] for r in successful]),
        'avg_drawdown': np.mean([r['max_drawdown'] for r in successful]),
        'avg_win_rate': np.mean([r['win_rate'] for r in successful]),
        'total_trades': sum(r['total_trades'] for r in successful),
        'n_stocks': len(successful),
    }


def main():
    """主函数"""
    print("=" * 80)
    print("PyTorch 深度学习模型回测")
    print("=" * 80)

    # 获取股票数据
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

    all_results = []

    # 测试 PyTorch 序列模型
    print("\n" + "=" * 80)
    print("测试 PyTorch 序列模型")
    print("=" * 80)

    for model_type in PYTORCH_SEQ_MODELS:
        print(f"\n测试 {model_type.upper()}...", end=" ")
        start_time = time.time()

        try:
            result = run_single_model_backtest(model_type, stock_data, DEFAULT_PARAMS, EPOCHS)
            elapsed = time.time() - start_time

            if result.get('success'):
                print(f"✅ 年化收益: {result['annual_return']*100:.2f}% (耗时: {elapsed:.1f}s)")
                all_results.append(result)
            else:
                print(f"❌ 失败: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"❌ 异常: {e}")

    # 测试部分高级模型（选择几个代表性的）
    print("\n" + "=" * 80)
    print("测试 PyTorch 高级模型 (部分)")
    print("=" * 80)

    # 只测试几个代表性的高级模型
    test_adv_models = ['gats', 'tabnet', 'adarnn', 'hist']

    for model_type in test_adv_models:
        print(f"\n测试 {model_type.upper()}...", end=" ")
        start_time = time.time()

        try:
            result = run_single_model_backtest(model_type, stock_data, DEFAULT_PARAMS, EPOCHS)
            elapsed = time.time() - start_time

            if result.get('success'):
                print(f"✅ 年化收益: {result['annual_return']*100:.2f}% (耗时: {elapsed:.1f}s)")
                all_results.append(result)
            else:
                print(f"❌ 失败: {result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"❌ 异常: {e}")

    # 生成对比报告
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)

    # 排序结果
    successful_results = [r for r in all_results if r.get('success')]
    successful_results.sort(key=lambda x: x['annual_return'], reverse=True)

    # 打印结果表格
    print("\n📊 PyTorch 模型收益排名:")
    print("-" * 80)
    print(f"{'排名':<4} {'模型':<15} {'类型':<12} {'年化收益':>10} {'夏普':>8} {'胜率':>8} {'交易次数':>8}")
    print("-" * 80)

    for i, r in enumerate(successful_results, 1):
        model_type = r['model_type']
        category = '序列模型' if model_type in PYTORCH_SEQ_MODELS else '高级模型'
        print(f"{i:<4} {model_type.upper():<15} {category:<12} "
              f"{r['annual_return']*100:>9.2f}% {r.get('avg_sharpe', 0):>8.2f} "
              f"{r.get('avg_win_rate', 0)*100:>7.1f}% {r.get('total_trades', 0):>8}")

    print("-" * 80)

    # 最佳模型
    if successful_results:
        best = successful_results[0]
        print(f"\n🏆 最佳 PyTorch 模型: {best['model_type'].upper()}")
        print(f"   年化收益率: {best['annual_return']*100:.2f}%")
        print(f"   平均夏普比: {best.get('avg_sharpe', 0):.2f}")
        print(f"   平均胜率: {best.get('avg_win_rate', 0)*100:.1f}%")
        print(f"   平均回撤: {best.get('avg_drawdown', 0)*100:.2f}%")

    # 与 GBDT 最佳对比
    print("\n📈 与 GBDT 最佳模型对比:")
    print("   CatBoost (GBDT): 22.02% 年化收益")
    if successful_results:
        print(f"   {successful_results[0]['model_type'].upper()} (PyTorch): {successful_results[0]['annual_return']*100:.2f}% 年化收益")

    return successful_results


if __name__ == "__main__":
    results = main()