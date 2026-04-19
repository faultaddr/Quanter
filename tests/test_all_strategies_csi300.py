#!/usr/bin/env python
"""
沪深300 Top20 全策略对比回测

使用所有可用策略回测沪深300 Top20，找出最佳策略
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

import pandas as pd
import numpy as np

from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher


# 沪深300 Top20 股票
TOP20_STOCKS = [
    '600900', '601225', '600519', '601088', '601288',
    '600482', '601669', '601919', '002384', '600875',
    '600989', '601877', '600938', '601857', '600026',
    '601898', '002714', '000895', '000999', '600028',
]

INITIAL_CAPITAL = 50000.0
LOOKBACK_DAYS = 365

# 策略配置
STRATEGIES = {
    '趋势动量': {
        'module': 'quanttool.strategies.trend_momentum_strategy',
        'class': 'TrendMomentumStrategy',
        'params': {'buy_threshold': 60, 'sell_threshold': 40, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    '趋势强度': {
        'module': 'quanttool.strategies.trend_strategy',
        'class': 'TrendStrategy',
        'params': {'buy_threshold': 65, 'sell_threshold': 45, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    '多因子评分': {
        'module': 'quanttool.strategies.score_strategy',
        'class': 'ScoreStrategy',
        'params': {'buy_threshold': 70, 'sell_threshold': 50, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    '均线交叉': {
        'module': 'quanttool.strategies.dual_ma',
        'class': 'DualMAStrategy',
        'params': {'short_period': 5, 'long_period': 20, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    'MACD': {
        'module': 'quanttool.strategies.macd',
        'class': 'MACDStrategy',
        'params': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    'RSI': {
        'module': 'quanttool.strategies.rsi',
        'class': 'RSIStrategy',
        'params': {'period': 14, 'oversold': 30, 'overbought': 70, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    '布林带': {
        'module': 'quanttool.strategies.bollinger',
        'class': 'BollingerStrategy',
        'params': {'period': 20, 'std_dev': 2.0, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    'KDJ': {
        'module': 'quanttool.strategies.kdj',
        'class': 'KDJStrategy',
        'params': {'n': 9, 'm1': 3, 'm2': 3, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    '均线排列': {
        'module': 'quanttool.strategies.ma_alignment',
        'class': 'MAAlignmentStrategy',
        'params': {'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    '突破策略': {
        'module': 'quanttool.strategies.breakout',
        'class': 'BreakoutStrategy',
        'params': {'period': 20, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    '海龟交易': {
        'module': 'quanttool.strategies.turtle',
        'class': 'TurtleStrategy',
        'params': {'entry_period': 20, 'exit_period': 10, 'stop_loss_pct': 0.05},
    },
    'Qlib-CatBoost': {
        'module': 'quanttool.strategies.qlib_strategy',
        'class': 'QlibStrategy',
        'params': {'model_type': 'catboost', 'buy_threshold': 0.55, 'sell_threshold': 0.45, 'epochs': 30, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    'Qlib-LightGBM': {
        'module': 'quanttool.strategies.qlib_strategy',
        'class': 'QlibStrategy',
        'params': {'model_type': 'lgb', 'buy_threshold': 0.55, 'sell_threshold': 0.45, 'epochs': 30, 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
    'Qlib-LSTM': {
        'module': 'quanttool.strategies.qlib_strategy',
        'class': 'QlibStrategy',
        'params': {'model_type': 'lstm', 'buy_threshold': 0.55, 'sell_threshold': 0.45, 'epochs': 30, 'hidden_size': 64, 'num_layers': 2, 'device': 'cpu', 'stop_loss_pct': 0.05, 'take_profit_pct': 0.12},
    },
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


def run_strategy_backtest(strategy_name: str, stock_data: dict, config: dict) -> dict:
    """运行单个策略的回测"""
    results = []
    total_capital = INITIAL_CAPITAL * len(stock_data)

    # 动态导入策略
    try:
        module = __import__(config['module'], fromlist=[config['class']])
        StrategyClass = getattr(module, config['class'])
    except Exception as e:
        return {'strategy': strategy_name, 'success': False, 'error': f'导入失败: {e}'}

    for symbol, df in stock_data.items():
        if len(df) < 120:
            continue

        try:
            # 创建策略实例
            strategy = StrategyClass(**config['params'])

            # Qlib策略需要训练
            if 'Qlib' in strategy_name:
                train_size = int(len(df) * 0.7)
                train_data = df.iloc[:train_size]
                if hasattr(strategy, 'train_model'):
                    strategy.train_model(train_data, horizon=10)

            # 回测
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
        except Exception as e:
            results.append({'symbol': symbol, 'success': False, 'error': str(e)})

    # 计算组合收益
    successful = [r for r in results if r.get('success')]
    if not successful:
        return {'strategy': strategy_name, 'success': False, 'error': 'All backtests failed'}

    total_final = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in successful)
    total_return = (total_final - total_capital) / total_capital
    annual_return = (1 + total_return) ** (252 / LOOKBACK_DAYS) - 1

    return {
        'strategy': strategy_name,
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
    print("沪深300 Top20 全策略对比回测")
    print("=" * 80)

    # 获取股票数据
    print(f"\n获取股票数据 (回测周期: {LOOKBACK_DAYS} 天)...")
    stock_data = {}
    for symbol in TOP20_STOCKS:
        df = fetch_stock_data(symbol)
        if not df.empty and len(df) >= 120:
            stock_data[symbol] = df
    print(f"有效股票: {len(stock_data)} 只")

    if not stock_data:
        print("❌ 没有有效数据")
        return

    # 运行所有策略
    print("\n" + "=" * 80)
    print("运行策略回测...")
    print("=" * 80)

    all_results = []
    for strategy_name, config in STRATEGIES.items():
        print(f"\n测试 {strategy_name}...", end=" ")
        start_time = time.time()

        result = run_strategy_backtest(strategy_name, stock_data, config)
        elapsed = time.time() - start_time

        if result.get('success'):
            print(f"✅ 年化收益: {result['annual_return']*100:.2f}% (耗时: {elapsed:.1f}s)")
            all_results.append(result)
        else:
            print(f"❌ 失败: {result.get('error', 'Unknown')}")

    # 排序结果
    all_results.sort(key=lambda x: x['annual_return'], reverse=True)

    # 打印汇总
    print("\n" + "=" * 80)
    print("策略收益排名")
    print("=" * 80)
    print(f"\n{'排名':<4} {'策略':<15} {'年化收益':>10} {'夏普':>8} {'胜率':>8} {'交易次数':>8}")
    print("-" * 80)

    for i, r in enumerate(all_results, 1):
        print(f"{i:<4} {r['strategy']:<15} {r['annual_return']*100:>9.2f}% "
              f"{r['avg_sharpe']:>8.2f} {r['avg_win_rate']*100:>7.1f}% {r['total_trades']:>8}")

    print("-" * 80)

    if all_results:
        best = all_results[0]
        print(f"\n🏆 最佳策略: {best['strategy']}")
        print(f"   年化收益率: {best['annual_return']*100:.2f}%")
        print(f"   平均夏普比: {best['avg_sharpe']:.2f}")
        print(f"   平均胜率: {best['avg_win_rate']*100:.1f}%")
        print(f"   平均回撤: {best['avg_drawdown']*100:.2f}%")

    return all_results


if __name__ == "__main__":
    results = main()