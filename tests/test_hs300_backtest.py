#!/usr/bin/env python
"""
沪深 300 股票池回测分析

使用最佳模型进行扩展测试，分析：
- 最大回撤
- 风险敞口
- 策略泛化能力
"""
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings('ignore')

# 先导入 PyTorch 避免崩溃
import torch
import torch.nn as nn

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from quanttool.strategies.qlib_strategy import QlibStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher


# 沪深 300 部分成分股（按行业分布选取代表性股票）
HS300_STOCKS = [
    # 金融
    '600036',  # 招商银行
    '601318',  # 中国平安
    '601166',  # 兴业银行
    '600000',  # 浦发银行
    '601398',  # 工商银行
    # 科技
    '000063',  # 中兴通讯
    '002415',  # 海康威视
    '000725',  # 京东方A
    '002475',  # 立讯精密
    '600588',  # 用友网络
    # 消费
    '000858',  # 五粮液
    '000568',  # 泸州老窖
    '600887',  # 伊利股份
    '000333',  # 美的集团
    '000651',  # 格力电器
    # 医药
    '000661',  # 长春高新
    '300760',  # 迈瑞医疗
    '600276',  # 恒瑞医药
    '000538',  # 云南白药
    '300015',  # 爱尔眼科
    # 新能源
    '300750',  # 宁德时代
    '002594',  # 比亚迪
    '600900',  # 长江电力
    '601012',  # 隆基绿能
    '002129',  # 中环股份
    # 基建/地产
    '600048',  # 保利发展
    '000002',  # 万科A
    '601668',  # 中国建筑
    '600585',  # 海螺水泥
    '601888',  # 中国中免
    # 制造业
    '600031',  # 三一重工
    '002050',  # 三花智控
    '600690',  # 海尔智家
    '002352',  # 顺丰控股
    # 周期
    '601899',  # 紫金矿业
    '600028',  # 中国石化
    '601088',  # 中国神华
    '600309',  # 万华化学
    '600346',  # 恒力石化
]

# 测试模型
TEST_MODELS = [
    ('catboost', 'GBDT'),
    ('tabnet', '深度学习'),
]

# 策略参数
DEFAULT_PARAMS = {
    'feature_set': 'Alpha158',
    'buy_threshold': 0.55,
    'sell_threshold': 0.45,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.10,
    'commission': 0.0003,
    'device': 'cpu',
}

INITIAL_CAPITAL = 100000.0
LOOKBACK_DAYS = 365 * 2
EPOCHS = 10


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
        return pd.DataFrame()


def run_backtest(model_type: str, stock_data: dict, params: dict, epochs: int = EPOCHS) -> dict:
    """运行回测并返回详细结果"""
    import gc

    all_results = []

    for symbol, df in stock_data.items():
        if len(df) < 120:
            continue

        try:
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
                device=params.get('device', 'cpu'),
            )

            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
            strategy.train_model(train_data, horizon=10)

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

            all_results.append({
                'symbol': symbol,
                'annual_return': result.annual_return,
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'total_trades': result.total_trades,
            })

            del strategy
            del engine
            gc.collect()

        except Exception as e:
            pass

    if not all_results:
        return None

    returns = [r['annual_return'] for r in all_results]
    drawdowns = [r['max_drawdown'] for r in all_results]
    sharpe_ratios = [r['sharpe_ratio'] for r in all_results]
    win_rates = [r['win_rate'] for r in all_results]
    trades = [r['total_trades'] for r in all_results]

    total_capital = INITIAL_CAPITAL * len(all_results)
    total_final = sum(INITIAL_CAPITAL * (1 + r['total_return']) for r in all_results)
    portfolio_return = (total_final - total_capital) / total_capital
    portfolio_annual = (1 + portfolio_return) ** (252 / LOOKBACK_DAYS) - 1

    return {
        'model_type': model_type,
        'n_stocks': len(all_results),
        'portfolio_annual_return': portfolio_annual,
        'portfolio_total_return': portfolio_return,
        'avg_annual_return': np.mean(returns),
        'median_annual_return': np.median(returns),
        'std_annual_return': np.std(returns),
        'avg_max_drawdown': np.mean(drawdowns),
        'max_drawdown': np.max(drawdowns),
        'avg_sharpe': np.mean(sharpe_ratios),
        'avg_win_rate': np.mean(win_rates),
        'total_trades': sum(trades),
        'positive_ratio': sum(1 for r in returns if r > 0) / len(returns),
        'stock_results': all_results,
    }


def analyze_risk_exposure(results: dict) -> dict:
    """分析风险敞口"""
    stock_results = results['stock_results']
    returns = [r['annual_return'] for r in stock_results]
    drawdowns = [r['max_drawdown'] for r in stock_results]

    return {
        'return_90th': np.percentile(returns, 90),
        'return_75th': np.percentile(returns, 75),
        'return_50th': np.percentile(returns, 50),
        'return_25th': np.percentile(returns, 25),
        'return_10th': np.percentile(returns, 10),
        'drawdown_90th': np.percentile(drawdowns, 90),
        'drawdown_75th': np.percentile(drawdowns, 75),
        'drawdown_50th': np.percentile(drawdowns, 50),
        'drawdown_25th': np.percentile(drawdowns, 25),
        'drawdown_10th': np.percentile(drawdowns, 10),
        'return_per_drawdown': results['avg_annual_return'] / results['avg_max_drawdown'] if results['avg_max_drawdown'] > 0 else 0,
        'max_single_loss': min(returns),
        'max_single_gain': max(returns),
        'worst_drawdown': max(drawdowns),
        'win_rate': results['positive_ratio'],
    }


def main():
    print("=" * 80)
    print("沪深 300 股票池回测分析")
    print("=" * 80)

    print("\n获取沪深 300 成分股数据...")
    stock_data = {}
    failed = []

    for i, symbol in enumerate(HS300_STOCKS):
        print(f"  [{i+1}/{len(HS300_STOCKS)}] 获取 {symbol}...", end=" ")
        df = fetch_stock_data(symbol)
        if not df.empty and len(df) >= 120:
            stock_data[symbol] = df
            print(f"✅ {len(df)} 条")
        else:
            failed.append(symbol)
            print("❌ 数据不足")

    print(f"\n成功获取: {len(stock_data)} 只股票")
    print(f"失败: {len(failed)} 只")

    if len(stock_data) < 10:
        print("❌ 数据不足，无法进行回测")
        return

    all_model_results = []

    for model_type, model_cat in TEST_MODELS:
        print(f"\n{'='*80}")
        print(f"测试 {model_type.upper()} ({model_cat})")
        print("=" * 80)

        start_time = time.time()
        result = run_backtest(model_type, stock_data, DEFAULT_PARAMS, EPOCHS)
        elapsed = time.time() - start_time

        if result:
            print(f"\n✅ 回测完成 (耗时: {elapsed:.1f}s)")
            print(f"   有效股票: {result['n_stocks']} 只")
            print(f"   组合年化收益: {result['portfolio_annual_return']*100:.2f}%")
            print(f"   平均年化收益: {result['avg_annual_return']*100:.2f}%")
            print(f"   平均最大回撤: {result['avg_max_drawdown']*100:.2f}%")
            print(f"   平均夏普比: {result['avg_sharpe']:.2f}")
            print(f"   盈利股票比例: {result['positive_ratio']*100:.1f}%")
            all_model_results.append(result)

    print("\n" + "=" * 80)
    print("投资组合风险分析报告")
    print("=" * 80)

    for result in all_model_results:
        model_type = result['model_type']
        risk = analyze_risk_exposure(result)

        print(f"\n{'─'*80}")
        print(f"模型: {model_type.upper()}")
        print(f"{'─'*80}")

        print("\n【收益分布】")
        print(f"  90分位: {risk['return_90th']*100:>8.2f}%")
        print(f"  75分位: {risk['return_75th']*100:>8.2f}%")
        print(f"  50分位: {risk['return_50th']*100:>8.2f}%")
        print(f"  25分位: {risk['return_25th']*100:>8.2f}%")
        print(f"  10分位: {risk['return_10th']*100:>8.2f}%")

        print("\n【最大回撤分布】")
        print(f"  90分位: {risk['drawdown_90th']*100:>8.2f}%")
        print(f"  75分位: {risk['drawdown_75th']*100:>8.2f}%")
        print(f"  50分位: {risk['drawdown_50th']*100:>8.2f}%")
        print(f"  25分位: {risk['drawdown_25th']*100:>8.2f}%")
        print(f"  10分位: {risk['drawdown_10th']*100:>8.2f}%")

        print("\n【风险指标】")
        print(f"  最大单股亏损: {risk['max_single_loss']*100:.2f}%")
        print(f"  最大单股盈利: {risk['max_single_gain']*100:.2f}%")
        print(f"  最差回撤: {risk['worst_drawdown']*100:.2f}%")
        print(f"  收益/回撤比: {risk['return_per_drawdown']:.2f}")
        print(f"  盈利股票比例: {risk['win_rate']*100:.1f}%")

    if len(all_model_results) >= 2:
        print("\n" + "=" * 80)
        print("模型对比")
        print("=" * 80)

        print(f"\n{'指标':<20} {'CatBoost':>15} {'TABNET':>15} {'差异':>15}")
        print("─" * 65)

        r1, r2 = all_model_results[0], all_model_results[1]

        metrics = [
            ('组合年化收益', 'portfolio_annual_return', '%'),
            ('平均年化收益', 'avg_annual_return', '%'),
            ('平均最大回撤', 'avg_max_drawdown', '%'),
            ('平均夏普比', 'avg_sharpe', ''),
            ('盈利比例', 'positive_ratio', '%'),
            ('总交易次数', 'total_trades', ''),
        ]

        for name, key, unit in metrics:
            v1, v2 = r1[key], r2[key]
            diff = v2 - v1
            if unit == '%':
                print(f"{name:<20} {v1*100:>14.2f}% {v2*100:>14.2f}% {diff*100:>14.2f}%")
            else:
                print(f"{name:<20} {v1:>15.2f} {v2:>15.2f} {diff:>15.2f}")

    print("\n" + "=" * 80)
    print("投资建议")
    print("=" * 80)

    if all_model_results:
        best = max(all_model_results, key=lambda x: x['portfolio_annual_return'])
        risk = analyze_risk_exposure(best)

        print(f"""
1. 模型选择建议:
   - 推荐使用: {best['model_type'].upper()}
   - 组合年化收益: {best['portfolio_annual_return']*100:.2f}%
   - 风险控制: 平均回撤 {best['avg_max_drawdown']*100:.2f}%

2. 风险提示:
   - 单股最大亏损可能达 {abs(risk['max_single_loss'])*100:.2f}%
   - 建议单股仓位控制在 5-10%
   - 总仓位建议控制在 70-80%

3. 策略优化方向:
   - 可考虑加入止损止盈机制
   - 建议结合市场情绪指标过滤信号
   - 考虑行业分散配置降低集中度风险
""")

    return all_model_results


if __name__ == "__main__":
    results = main()
