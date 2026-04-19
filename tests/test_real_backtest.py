#!/usr/bin/env python
"""
真实数据回测验证脚本

使用真实股票数据验证评分策略的表现
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 导入模块
from quanttool.strategies.score_strategy import ScoreStrategy, EnhancedScoreStrategy
from quanttool.factors.scoring_system import ScoringSystem
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher
from quanttool.optimization.weight_optimizer import DynamicWeightOptimizer
from quanttool.risk.risk_controller import RiskController
from quanttool.validation.score_validator import ScoreValidator
from quanttool.reports.signal_backtest_report import SignalBacktestReporter


def fetch_real_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    获取真实股票数据

    Args:
        symbol: 股票代码 (如 600519, 000001)
        days: 获取天数

    Returns:
        DataFrame with OHLCV data
    """
    print(f"\n获取 {symbol} 最近 {days} 天数据...")

    end_date = datetime.now().strftime('%Y-%m-%d')

    df = AshareFetcher.get_price(
        code=symbol,
        end_date=end_date,
        count=days,
        frequency='1d'
    )

    if df.empty:
        print(f"❌ 无法获取 {symbol} 数据")
        return pd.DataFrame()

    # 确保数据格式正确
    if 'timestamp' not in df.columns:
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"✅ 成功获取 {len(df)} 条数据")
    print(f"   日期范围: {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}")
    print(f"   价格范围: {df['close'].min():.2f} ~ {df['close'].max():.2f}")

    return df


def analyze_score_distribution(df: pd.DataFrame, strategy: ScoreStrategy) -> dict:
    """分析评分分布"""
    print("\n" + "="*60)
    print("评分分布分析")
    print("="*60)

    # 计算评分
    signals_df = strategy.calculate_signals(df)

    scores = signals_df['final_score']
    buy_signals = (signals_df['signal'] == 'buy').sum()
    sell_signals = (signals_df['signal'] == 'sell').sum()

    print(f"\n评分统计:")
    print(f"  平均评分: {scores.mean():.2f}")
    print(f"  评分标准差: {scores.std():.2f}")
    print(f"  最低评分: {scores.min():.2f}")
    print(f"  最高评分: {scores.max():.2f}")
    print(f"  中位数: {scores.median():.2f}")

    print(f"\n信号分布:")
    print(f"  买入信号 (>= {strategy.buy_threshold}): {buy_signals}")
    print(f"  卖出信号 (<= {strategy.sell_threshold}): {sell_signals}")
    print(f"  持仓信号: {len(signals_df) - buy_signals - sell_signals}")

    # 评分区间统计
    print(f"\n评分区间分布:")
    bins = [0, 30, 40, 50, 60, 70, 80, 90, 100]
    for i in range(len(bins) - 1):
        count = ((scores >= bins[i]) & (scores < bins[i+1])).sum()
        pct = count / len(scores) * 100
        bar = '█' * int(pct / 2)
        print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {count:4d} ({pct:5.1f}%) {bar}")

    return {
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max(),
        'buy_signals': buy_signals,
        'sell_signals': sell_signals
    }


def run_backtest_with_real_data(
    df: pd.DataFrame,
    symbol: str,
    buy_threshold: float = 70.0,
    sell_threshold: float = 50.0
) -> dict:
    """使用真实数据运行回测"""
    print("\n" + "="*60)
    print(f"回测: {symbol}")
    print("="*60)

    # 创建策略
    strategy = ScoreStrategy(
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        use_dynamic_weights=True,
        use_multi_timeframe=True,
        use_risk_control=True
    )

    print(f"\n策略配置:")
    print(f"  买入阈值: {buy_threshold}")
    print(f"  卖出阈值: {sell_threshold}")
    print(f"  动态权重: 启用")
    print(f"  多周期确认: 启用")
    print(f"  风险控制: 启用")

    # 初始化回测引擎
    engine = BacktestEngine()
    engine.set_initial_cash(100000)

    # 准备数据
    data = {symbol: df.copy()}

    # 运行回测
    print("\n运行回测...")

    try:
        result = engine.run_backtest(
            strategy=strategy,
            data=data,
            start_date=df['timestamp'].iloc[0],
            end_date=df['timestamp'].iloc[-1]
        )

        print(f"\n回测结果:")
        print(f"  期初资金: ${result.initial_capital:,.2f}")
        print(f"  期末资金: ${result.final_capital:,.2f}")
        print(f"  总收益: {result.total_return:.2%}")
        print(f"  年化收益: {result.annual_return:.2%}")
        print(f"  夏普比率: {result.sharpe_ratio:.2f}")
        print(f"  最大回撤: {result.max_drawdown:.2%}")
        print(f"  胜率: {result.win_rate:.2%}")
        print(f"  总交易数: {result.total_trades}")

        return {
            'success': True,
            'initial_capital': result.initial_capital,
            'final_capital': result.final_capital,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'result': result
        }

    except Exception as e:
        print(f"\n回测执行出错: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def validate_scoring_predictiveness(df: pd.DataFrame) -> dict:
    """验证评分对未来收益的预测能力"""
    print("\n" + "="*60)
    print("评分预测能力验证")
    print("="*60)

    scoring_system = ScoringSystem()
    validator = ScoreValidator()

    # 计算评分
    scores = []
    for i in range(30, len(df)):
        window = df.iloc[:i+1]
        try:
            result = scoring_system.calculate_comprehensive_score(window)
            scores.append(result.get('final_score', 50))
        except Exception:
            scores.append(50)

    df_test = df.iloc[30:].copy()
    df_test['final_score'] = scores

    # 计算未来收益
    df_test['future_return_5d'] = df_test['close'].pct_change(5).shift(-5)
    df_test['future_return_10d'] = df_test['close'].pct_change(10).shift(-10)
    df_test = df_test.dropna()

    if len(df_test) < 30:
        print("数据不足，跳过验证")
        return {}

    # 计算IC
    ic_result = validator.validate_score_correlation(
        df_test.set_index('timestamp')['final_score'],
        df_test.set_index('timestamp')['future_return_5d'],
        horizon_days=5
    )

    print(f"\nIC 分析结果:")
    print(f"  IC (信息系数): {ic_result.ic:.4f}")
    print(f"  Rank IC: {ic_result.rank_ic:.4f}")
    print(f"  IC IR: {ic_result.ic_ir:.4f}")
    print(f"  样本数: {ic_result.sample_size}")

    # 分位数分析
    quantile_df = validator.calculate_score_quantile_analysis(
        df_test.set_index('timestamp')['final_score'],
        df_test.set_index('timestamp')['future_return_5d'],
        n_quantiles=5
    )

    if not quantile_df.empty:
        print(f"\n分位数收益分析:")
        print(quantile_df.to_string())

    return {
        'ic': ic_result.ic,
        'rank_ic': ic_result.rank_ic,
        'ic_ir': ic_result.ic_ir,
        'quantile_analysis': quantile_df.to_dict() if not quantile_df.empty else {}
    }


def optimize_parameters(df: pd.DataFrame) -> dict:
    """优化策略参数"""
    print("\n" + "="*60)
    print("策略参数优化")
    print("="*60)

    best_result = None
    best_params = None
    best_return = -float('inf')

    # 参数搜索空间
    buy_thresholds = [60, 65, 70, 75, 80]
    sell_thresholds = [30, 40, 45, 50, 55]

    results = []

    for buy_th in buy_thresholds:
        for sell_th in sell_thresholds:
            if buy_th <= sell_th:
                continue

            result = run_backtest_with_real_data(
                df, "TEST",
                buy_threshold=buy_th,
                sell_threshold=sell_th
            )

            if result.get('success') and result.get('total_return', -float('inf')) > best_return:
                best_return = result['total_return']
                best_result = result
                best_params = {
                    'buy_threshold': buy_th,
                    'sell_threshold': sell_th
                }
                results.append({
                    'buy_threshold': buy_th,
                    'sell_threshold': sell_th,
                    'total_return': result['total_return'],
                    'total_trades': result['total_trades']
                })

    if best_params:
        print(f"\n最佳参数:")
        print(f"  买入阈值: {best_params['buy_threshold']}")
        print(f"  卖出阈值: {best_params['sell_threshold']}")
        print(f"  总收益: {best_return:.2%}")

    return {
        'best_params': best_params,
        'best_return': best_return,
        'all_results': results
    }


def generate_report(
    symbol: str,
    df: pd.DataFrame,
    backtest_result: dict,
    score_analysis: dict,
    validation_result: dict
) -> str:
    """生成综合分析报告"""
    print("\n" + "="*60)
    print("生成分析报告")
    print("="*60)

    report = f"""# 量化评分策略验证报告

## 基本信息

- **股票代码**: {symbol}
- **数据范围**: {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}
- **数据量**: {len(df)} 条

## 评分分布

| 指标 | 数值 |
|------|------|
| 平均评分 | {score_analysis.get('mean', 0):.2f} |
| 评分标准差 | {score_analysis.get('std', 0):.2f} |
| 最低评分 | {score_analysis.get('min', 0):.2f} |
| 最高评分 | {score_analysis.get('max', 0):.2f} |
| 买入信号数 | {score_analysis.get('buy_signals', 0)} |
| 卖出信号数 | {score_analysis.get('sell_signals', 0)} |

## 回测结果

| 指标 | 数值 |
|------|------|
| 期初资金 | ${backtest_result.get('initial_capital', 0):,.2f} |
| 期末资金 | ${backtest_result.get('final_capital', 0):,.2f} |
| 总收益 | {backtest_result.get('total_return', 0):.2%} |
| 年化收益 | {backtest_result.get('annual_return', 0):.2%} |
| 夏普比率 | {backtest_result.get('sharpe_ratio', 0):.2f} |
| 最大回撤 | {backtest_result.get('max_drawdown', 0):.2%} |
| 胜率 | {backtest_result.get('win_rate', 0):.2%} |
| 总交易数 | {backtest_result.get('total_trades', 0)} |

## 评分预测能力

| 指标 | 数值 |
|------|------|
| IC (信息系数) | {validation_result.get('ic', 0):.4f} |
| Rank IC | {validation_result.get('rank_ic', 0):.4f} |
| IC IR | {validation_result.get('ic_ir', 0):.4f} |

## 结论

"""

    if backtest_result.get('total_trades', 0) == 0:
        report += """
⚠️ **交易数为零**: 当前参数下策略未产生交易信号。

**可能原因**:
1. 买入阈值过高，评分很少达到
2. 数据期间评分分布不适合当前阈值

**建议**:
1. 降低买入阈值 (如从 70 降到 60 或 65)
2. 提高卖出阈值 (如从 50 提到 45)
3. 运行参数优化寻找最佳参数组合
"""
    elif backtest_result.get('total_return', 0) > 0:
        report += f"""
✅ **策略有效**: 策略在回测期间实现了 {backtest_result.get('total_return', 0):.2%} 的正收益。

**建议**:
1. 继续监控策略在不同市场环境下的表现
2. 考虑结合风险控制模块优化仓位管理
"""
    else:
        report += f"""
❌ **策略亏损**: 策略在回测期间亏损 {backtest_result.get('total_return', 0):.2%}。

**建议**:
1. 调整策略参数
2. 增加评分因子的预测能力
3. 优化信号过滤逻辑
"""

    report += """
---

*本报告由 QuantTool 评分策略验证系统生成*
"""

    return report


def main():
    """主函数"""
    print("\n" + "="*60)
    print("QuantTool 真实数据回测验证")
    print("="*60)

    # 测试股票列表
    test_symbols = [
        ('600519', '贵州茅台'),  # 白酒龙头
        ('000001', '平安银行'),  # 银行
        ('300750', '宁德时代'),  # 新能源
    ]

    all_results = {}

    for symbol, name in test_symbols:
        print(f"\n{'='*60}")
        print(f"分析 {name} ({symbol})")
        print("="*60)

        # 获取数据
        df = fetch_real_data(symbol, days=365)

        if df.empty:
            continue

        # 分析评分分布
        strategy = ScoreStrategy(buy_threshold=70, sell_threshold=50)
        score_analysis = analyze_score_distribution(df, strategy)

        # 验证评分预测能力
        validation_result = validate_scoring_predictiveness(df)

        # 运行回测
        backtest_result = run_backtest_with_real_data(
            df, symbol,
            buy_threshold=70,
            sell_threshold=50
        )

        # 生成报告
        report = generate_report(
            symbol, df,
            backtest_result,
            score_analysis,
            validation_result
        )

        # 保存报告
        report_dir = Path(__file__).parent.parent / "reports"
        report_dir.mkdir(exist_ok=True)

        report_path = report_dir / f"strategy_validation_{symbol}_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\n报告已保存: {report_path}")

        all_results[symbol] = {
            'name': name,
            'score_analysis': score_analysis,
            'backtest_result': backtest_result,
            'validation_result': validation_result
        }

    # 汇总结果
    print("\n" + "="*60)
    print("汇总结果")
    print("="*60)

    for symbol, result in all_results.items():
        backtest = result.get('backtest_result', {})
        print(f"\n{result['name']} ({symbol}):")
        print(f"  总收益: {backtest.get('total_return', 0):.2%}")
        print(f"  总交易数: {backtest.get('total_trades', 0)}")
        print(f"  IC: {result.get('validation_result', {}).get('ic', 0):.4f}")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)