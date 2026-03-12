#!/usr/bin/env python
"""
Qlib 高级 PyTorch 模型沪深300回测脚本

测试11个高级PyTorch模型在沪深300成分股上的表现：
- GATs, SFM, TabNet, ADARNN, ADD, HIST, IGMTF, KRNN, TRA, TCTS, Sandwich

特性：
- 40只沪深300代表性成分股
- 50轮训练（高精度）
- 2年历史数据
- 进度保存和恢复
- 详细结果汇总
"""
import sys
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 先导入 PyTorch 避免崩溃
import torch
import torch.nn as nn

import pandas as pd
import numpy as np

from quanttool.strategies.qlib_strategy import QlibStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.data_fetcher import AshareFetcher


# ============================================================================
# 配置参数
# ============================================================================

# 11个高级PyTorch模型
ADVANCED_MODELS = {
    'gats': {'name': 'GATs', 'category': 'PyTorch高级'},
    'sfm': {'name': 'SFM', 'category': 'PyTorch高级'},
    'tabnet': {'name': 'TabNet', 'category': 'PyTorch高级'},
    'adarnn': {'name': 'ADARNN', 'category': 'PyTorch高级'},
    'add': {'name': 'ADD', 'category': 'PyTorch高级'},
    'hist': {'name': 'HIST', 'category': 'PyTorch高级'},
    'igmtf': {'name': 'IGMTF', 'category': 'PyTorch高级'},
    'krnn': {'name': 'KRNN', 'category': 'PyTorch高级'},
    'tra': {'name': 'TRA', 'category': 'PyTorch高级'},
    'tcts': {'name': 'TCTS', 'category': 'PyTorch高级'},
    'sandwich': {'name': 'Sandwich', 'category': 'PyTorch高级'},
}

# 沪深300部分成分股（按行业分布选取代表性股票，40只）
HS300_STOCKS = [
    # 金融（5只）
    '600036',  # 招商银行
    '601318',  # 中国平安
    '601166',  # 兴业银行
    '600000',  # 浦发银行
    '601398',  # 工商银行
    # 科技（5只）
    '000063',  # 中兴通讯
    '002415',  # 海康威视
    '000725',  # 京东方A
    '002475',  # 立讯精密
    '600588',  # 用友网络
    # 消费（5只）
    '000858',  # 五粮液
    '000568',  # 泸州老窖
    '600887',  # 伊利股份
    '000333',  # 美的集团
    '000651',  # 格力电器
    # 医药（5只）
    '000661',  # 长春高新
    '300760',  # 迈瑞医疗
    '600276',  # 恒瑞医药
    '000538',  # 云南白药
    '300015',  # 爱尔眼科
    # 新能源（5只）
    '300750',  # 宁德时代
    '002594',  # 比亚迪
    '600900',  # 长江电力
    '601012',  # 隆基绿能
    '002129',  # 中环股份
    # 基建/地产（5只）
    '600048',  # 保利发展
    '000002',  # 万科A
    '601668',  # 中国建筑
    '600585',  # 海螺水泥
    '601888',  # 中国中免
    # 制造业（5只）
    '600031',  # 三一重工
    '002050',  # 三花智控
    '600690',  # 海尔智家
    '002352',  # 顺丰控股
    '601766',  # 中国中车
    # 周期（5只）
    '601899',  # 紫金矿业
    '600028',  # 中国石化
    '601088',  # 中国神华
    '600309',  # 万华化学
    '600346',  # 恒力石化
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
LOOKBACK_DAYS = 365 * 2  # 2年数据
EPOCHS = 50  # 用户选择的训练轮数

# 进度保存文件
PROGRESS_FILE = Path(__file__).parent / '.backtest_progress.json'


# ============================================================================
# 核心函数
# ============================================================================

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


def save_progress(completed_models: list, results: list):
    """保存进度到文件"""
    progress = {
        'timestamp': datetime.now().isoformat(),
        'completed_models': completed_models,
        'results': results,
    }
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2, default=str)
    print(f"  💾 进度已保存 ({len(completed_models)}/{len(ADVANCED_MODELS)} 模型完成)")


def load_progress() -> tuple:
    """从文件加载进度"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, 'r') as f:
                progress = json.load(f)
            print(f"\n📂 发现保存的进度: {progress['timestamp']}")
            print(f"   已完成模型: {progress['completed_models']}")
            return progress['completed_models'], progress['results']
        except Exception as e:
            print(f"  ⚠️ 加载进度失败: {e}")
    return [], []


def clear_progress():
    """清除进度文件"""
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        print("  🗑️ 进度文件已清除")


def run_single_model_backtest(
    model_type: str,
    stock_data: dict,
    params: dict,
    epochs: int = EPOCHS,
) -> dict:
    """使用单个模型运行回测"""
    import gc

    results = []
    total_capital = INITIAL_CAPITAL * len(stock_data)

    for i, (symbol, df) in enumerate(stock_data.items(), 1):
        if len(df) < 120:
            continue

        print(f"    [{i}/{len(stock_data)}] {symbol}...", end=" ", flush=True)

        try:
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
                device=params.get('device', 'cpu'),
            )

            # 训练模型
            train_size = int(len(df) * 0.7)
            train_data = df.iloc[:train_size]
            strategy.train_model(train_data, horizon=10)

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

            print(f"✅ {result.annual_return*100:.2f}%")

            # 清理内存
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
        'max_drawdown': np.max([r['max_drawdown'] for r in successful]),
        'avg_win_rate': np.mean([r['win_rate'] for r in successful]),
        'total_trades': sum(r['total_trades'] for r in successful),
        'n_stocks': len(successful),
        'positive_ratio': sum(1 for r in successful if r['annual_return'] > 0) / len(successful),
        'individual_results': successful,
    }


def print_summary(all_results: list):
    """打印汇总报告"""
    print("\n" + "=" * 80)
    print("📊 回测结果汇总")
    print("=" * 80)

    # 排序结果
    successful_results = [r for r in all_results if r.get('success')]
    successful_results.sort(key=lambda x: x['annual_return'], reverse=True)

    # 打印结果表格
    print("\n┌" + "─" * 78 + "┐")
    print(f"│ {'排名':<4} {'模型':<12} {'年化收益':>10} {'夏普比':>8} {'胜率':>8} {'盈利比例':>8} {'股票数':>6} │")
    print("├" + "─" * 78 + "┤")

    for i, r in enumerate(successful_results, 1):
        model_info = ADVANCED_MODELS.get(r['model_type'], {'name': r['model_type']})
        print(f"│ {i:<4} {model_info['name']:<12} "
              f"{r['annual_return']*100:>9.2f}% "
              f"{r.get('avg_sharpe', 0):>8.2f} "
              f"{r.get('avg_win_rate', 0)*100:>7.1f}% "
              f"{r.get('positive_ratio', 0)*100:>7.1f}% "
              f"{r.get('n_stocks', 0):>6} │")

    print("└" + "─" * 78 + "┘")

    # 最佳模型
    if successful_results:
        print("\n" + "=" * 80)
        print("🏆 最佳模型")
        print("=" * 80)

        best = successful_results[0]
        best_info = ADVANCED_MODELS.get(best['model_type'], {'name': best['model_type']})

        print(f"""
模型: {best_info['name']} ({best['model_type']})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  年化收益率:  {best['annual_return']*100:>10.2f}%
  总收益率:    {best['total_return']*100:>10.2f}%
  平均夏普比:  {best.get('avg_sharpe', 0):>10.2f}
  平均最大回撤: {best.get('avg_drawdown', 0)*100:>10.2f}%
  最差回撤:    {best.get('max_drawdown', 0)*100:>10.2f}%
  平均胜率:    {best.get('avg_win_rate', 0)*100:>10.1f}%
  盈利股票比例: {best.get('positive_ratio', 0)*100:>10.1f}%
  总交易次数:  {best.get('total_trades', 0):>10}
  成功股票数:  {best.get('n_stocks', 0):>10}
""")

    # 失败的模型
    failed = [r for r in all_results if not r.get('success')]
    if failed:
        print("=" * 80)
        print("⚠️ 失败模型")
        print("=" * 80)
        for r in failed:
            print(f"  - {r['model_type']}: {r.get('error', 'Unknown error')}")

    # 投资建议
    if successful_results:
        best = successful_results[0]
        print("=" * 80)
        print("💡 投资建议")
        print("=" * 80)

        print(f"""
1. 模型选择:
   - 推荐使用: {ADVANCED_MODELS.get(best['model_type'], {}).get('name', best['model_type'])}
   - 组合年化收益: {best['annual_return']*100:.2f}%
   - 风险控制: 平均回撤 {best.get('avg_drawdown', 0)*100:.2f}%

2. 风险提示:
   - 最大回撤可能达 {best.get('max_drawdown', 0)*100:.2f}%
   - 建议单股仓位控制在 5-10%
   - 总仓位建议控制在 70-80%

3. 策略优化方向:
   - 可考虑加入止损止盈机制
   - 建议结合市场情绪指标过滤信号
   - 考虑行业分散配置降低集中度风险
""")

    return successful_results


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数 - 测试所有高级PyTorch模型"""
    print("=" * 80)
    print("Qlib 高级 PyTorch 模型沪深300回测")
    print("=" * 80)

    print(f"\n配置:")
    print(f"  模型数量: {len(ADVANCED_MODELS)} 个")
    print(f"  股票池: {len(HS300_STOCKS)} 只沪深300成分股")
    print(f"  训练轮数: {EPOCHS} 轮")
    print(f"  历史数据: {LOOKBACK_DAYS} 天 ({LOOKBACK_DAYS//365} 年)")

    # 尝试加载进度
    completed_models, saved_results = load_progress()

    if completed_models:
        print(f"\n检测到中断的进度，将从中断处继续...")
        print(f"已完成: {completed_models}")
        response = input("是否继续? (y/n): ").strip().lower()
        if response != 'y':
            print("清除进度，重新开始...")
            completed_models = []
            saved_results = []
            clear_progress()

    # 获取股票数据
    print("\n" + "=" * 80)
    print("第一步: 获取股票数据")
    print("=" * 80)

    stock_data = {}
    failed_stocks = []

    for i, symbol in enumerate(HS300_STOCKS, 1):
        print(f"  [{i}/{len(HS300_STOCKS)}] 获取 {symbol}...", end=" ")
        df = fetch_stock_data(symbol)
        if not df.empty and len(df) >= 120:
            stock_data[symbol] = df
            print(f"✅ {len(df)} 条")
        else:
            failed_stocks.append(symbol)
            print("❌ 数据不足")

    print(f"\n成功获取: {len(stock_data)} 只股票")
    print(f"失败: {len(failed_stocks)} 只")

    if len(stock_data) < 10:
        print("❌ 数据不足，无法进行回测")
        return

    # 运行模型回测
    print("\n" + "=" * 80)
    print("第二步: 运行模型回测")
    print("=" * 80)

    all_results = saved_results.copy()

    for model_type, model_info in ADVANCED_MODELS.items():
        # 跳过已完成的模型
        if model_type in completed_models:
            print(f"\n⏭️ 跳过 {model_info['name']} (已完成)")
            continue

        print(f"\n{'─'*80}")
        print(f"测试 {model_info['name']} ({model_type})")
        print(f"{'─'*80}")

        start_time = time.time()

        try:
            result = run_single_model_backtest(
                model_type, stock_data, DEFAULT_PARAMS, EPOCHS
            )
            elapsed = time.time() - start_time

            if result.get('success'):
                print(f"\n✅ 回测完成 (耗时: {elapsed/60:.1f} 分钟)")
                print(f"   有效股票: {result['n_stocks']} 只")
                print(f"   年化收益: {result['annual_return']*100:.2f}%")
                print(f"   平均夏普比: {result.get('avg_sharpe', 0):.2f}")
                print(f"   盈利股票比例: {result.get('positive_ratio', 0)*100:.1f}%")
                all_results.append(result)
            else:
                print(f"\n❌ 回测失败: {result.get('error', 'Unknown')}")
                all_results.append({
                    'model_type': model_type,
                    'success': False,
                    'annual_return': -999,
                    'error': result.get('error'),
                })

        except Exception as e:
            print(f"\n❌ 异常: {e}")
            all_results.append({
                'model_type': model_type,
                'success': False,
                'annual_return': -999,
                'error': str(e),
            })

        # 保存进度
        completed_models.append(model_type)
        save_progress(completed_models, all_results)

    # 打印汇总
    successful_results = print_summary(all_results)

    # 清除进度文件
    clear_progress()

    print("\n" + "=" * 80)
    print("回测完成!")
    print("=" * 80)

    return successful_results


if __name__ == "__main__":
    results = main()
