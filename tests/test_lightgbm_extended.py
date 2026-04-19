#!/usr/bin/env python
"""
Qlib-LightGBM 扩展回测

使用最佳策略（LightGBM）回测更多股票
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
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher


# 获取沪深300成分股
def get_csi300_stocks():
    """获取沪深300成分股列表"""
    try:
        # 尝试从数据源获取
        df = AshareFetcher.get_index_constituents('000300')
        if not df.empty:
            return df['code'].tolist()
    except:
        pass

    # 备用：返回已知的部分成分股
    return [
        '600519', '600036', '601318', '600900', '601166',
        '600031', '601888', '600887', '601012', '600276',
        '601398', '601288', '601939', '601328', '600000',
        '601658', '601818', '600030', '601211', '601688',
        '600048', '600016', '600009', '600104', '600585',
        '600893', '601888', '600690', '601888', '601766',
        '600309', '600809', '600066', '600100', '600588',
        '600406', '600660', '600004', '600085', '600196',
        '600438', '600482', '600570', '600588', '600637',
        '600703', '600887', '600900', '601012', '601066',
    ]


INITIAL_CAPITAL = 30000.0
LOOKBACK_DAYS = 365
N_STOCKS = 50  # 回测股票数量

# LightGBM 最佳策略参数
STRATEGY_PARAMS = {
    'model_type': 'lgb',
    'feature_set': 'Alpha158',
    'buy_threshold': 0.55,
    'sell_threshold': 0.45,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.12,
    'epochs': 30,
    'commission': 0.0003,
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


def run_backtest(stock_list: list) -> dict:
    """运行批量回测"""
    results = []
    total_capital = INITIAL_CAPITAL * len(stock_list)

    print(f"\n{'='*80}")
    print(f"开始回测 {len(stock_list)} 只股票...")
    print(f"{'='*80}\n")

    for i, symbol in enumerate(stock_list, 1):
        print(f"[{i}/{len(stock_list)}] {symbol}...", end=" ")

        df = fetch_stock_data(symbol)
        if len(df) < 120:
            print("❌ 数据不足")
            continue

        try:
            # 创建 Qlib-LightGBM 策略
            strategy = QlibStrategy(
                model_type='lgb',
                feature_set='Alpha158',
                buy_threshold=0.55,
                sell_threshold=0.45,
                stop_loss_pct=0.05,
                take_profit_pct=0.12,
                epochs=30,
            )

            # 训练模型
            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
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

            annual_return = result.annual_return
            sharpe = result.sharpe_ratio

            print(f"✅ 年化 {annual_return*100:.2f}%, 夏普 {sharpe:.2f}")

            results.append({
                'symbol': symbol,
                'success': True,
                'annual_return': annual_return,
                'total_return': result.total_return,
                'sharpe_ratio': sharpe,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
            })

        except Exception as e:
            print(f"❌ 失败: {e}")

    # 计算组合收益
    if not results:
        return {'success': False, 'error': 'All backtests failed'}

    successful = [r for r in results if r.get('success')]
    total_final = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in successful)
    total_return = (total_final - INITIAL_CAPITAL * len(successful)) / (INITIAL_CAPITAL * len(successful))
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
    print("=" * 80)
    print("Qlib-LightGBM 扩展回测")
    print("=" * 80)

    # 获取股票列表
    print("\n获取股票列表...")
    all_stocks = get_csi300_stocks()
    stocks = all_stocks[:N_STOCKS]
    print(f"选择 {len(stocks)} 只股票进行回测")

    # 运行回测
    start_time = time.time()
    result = run_backtest(stocks)
    elapsed = time.time() - start_time

    if not result.get('success'):
        print(f"\n❌ 回测失败: {result.get('error', 'Unknown')}")
        return

    # 打印结果
    print("\n" + "=" * 80)
    print("回测结果汇总")
    print("=" * 80)

    successful_results = [r for r in result['results'] if r.get('success')]
    successful_results.sort(key=lambda x: x['annual_return'], reverse=True)

    # Top 10
    print(f"\n📊 Top 10 股票:")
    print("-" * 80)
    print(f"{'排名':<4} {'代码':<8} {'年化收益':>10} {'夏普':>8} {'胜率':>8} {'交易':>6}")
    print("-" * 80)

    for i, r in enumerate(successful_results[:10], 1):
        print(f"{i:<4} {r['symbol']:<8} {r['annual_return']*100:>9.2f}% "
              f"{r['sharpe_ratio']:>8.2f} {r['win_rate']*100:>7.1f}% {r['total_trades']:>6}")

    print("-" * 80)

    # 组合统计
    print(f"\n📈 组合统计:")
    print(f"   回测股票: {result['n_stocks']} 只")
    print(f"   组合年化收益: {result['annual_return']*100:.2f}%")
    print(f"   组合总收益: {result['total_return']*100:.2f}%")
    print(f"   平均夏普比: {result['avg_sharpe']:.2f}")
    print(f"   平均回撤: {result['avg_drawdown']*100:.2f}%")
    print(f"   平均胜率: {result['avg_win_rate']*100:.1f}%")
    print(f"   总交易次数: {result['total_trades']} 次")
    print(f"   总耗时: {elapsed:.1f} 秒")

    # 收益分布
    positive = [r for r in successful_results if r['annual_return'] > 0]
    negative = [r for r in successful_results if r['annual_return'] <= 0]

    print(f"\n📊 收益分布:")
    print(f"   盈利股票: {len(positive)} 只 ({len(positive)/len(successful_results)*100:.1f}%)")
    print(f"   亏损股票: {len(negative)} 只 ({len(negative)/len(successful_results)*100:.1f}%)")

    if positive:
        print(f"   平均盈利: {np.mean([r['annual_return'] for r in positive])*100:.2f}%")
    if negative:
        print(f"   平均亏损: {np.mean([r['annual_return'] for r in negative])*100:.2f}%")

    return result


if __name__ == "__main__":
    results = main()