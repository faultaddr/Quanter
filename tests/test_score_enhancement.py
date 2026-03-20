#!/usr/bin/env python
"""
评分系统增强验证脚本

验证新增模块的功能并运行回测
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

# 导入新模块
from quanttool.validation.score_validator import ScoreValidator, validate_scoring_system
from quanttool.optimization.weight_optimizer import DynamicWeightOptimizer, MarketRegime
from quanttool.risk.risk_controller import RiskController, StopLossType
from quanttool.analysis.multi_timeframe_analyzer import MultiTimeframeAnalyzer, analyze_multi_timeframe
from quanttool.reports.signal_backtest_report import SignalBacktestReporter
from quanttool.strategies.score_strategy import ScoreStrategy, EnhancedScoreStrategy
from quanttool.factors.scoring_system import ScoringSystem
from quanttool.backtest.engine import BacktestEngine

# 数据获取
from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials


def generate_mock_data(days: int = 250) -> pd.DataFrame:
    """生成模拟数据用于测试"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # 生成价格数据
    close = 10 + np.cumsum(np.random.randn(days) * 0.02)
    high = close + np.abs(np.random.randn(days)) * 0.1
    low = close - np.abs(np.random.randn(days)) * 0.1
    open_price = close + np.random.randn(days) * 0.05
    volume = np.random.randint(100000, 1000000, days)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })

    return df


def test_score_validator():
    """测试评分验证模块"""
    print("\n" + "="*60)
    print("测试 1: 评分验证模块")
    print("="*60)

    # 生成模拟数据
    df = generate_mock_data(250)

    # 添加评分
    scoring_system = ScoringSystem()
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

    # 验证评分
    validator = ScoreValidator()
    df_test['future_return_5d'] = df_test['close'].pct_change(5).shift(-5)
    df_test = df_test.dropna()

    if len(df_test) > 30:
        result = validator.validate_score_correlation(
            df_test.set_index('timestamp')['final_score'],
            df_test.set_index('timestamp')['future_return_5d'],
            horizon_days=5
        )

        print(f"\n评分验证结果:")
        print(f"  IC (信息系数): {result.ic:.4f}")
        print(f"  Rank IC (秩相关系数): {result.rank_ic:.4f}")
        print(f"  IC IR (信息比率): {result.ic_ir:.4f}")
        print(f"  样本数: {result.sample_size}")

        # 分位数分析
        quantile_df = validator.calculate_score_quantile_analysis(
            df_test.set_index('timestamp')['final_score'],
            df_test.set_index('timestamp')['future_return_5d'],
            n_quantiles=5
        )
        if not quantile_df.empty:
            print(f"\n分位数收益分析:")
            print(quantile_df.to_string())

    print("\n✅ 评分验证模块测试完成")
    return True


def test_weight_optimizer():
    """测试动态权重优化模块"""
    print("\n" + "="*60)
    print("测试 2: 动态权重优化模块")
    print("="*60)

    df = generate_mock_data(250)

    optimizer = DynamicWeightOptimizer(lookback_period=60)

    # 检测市场状态
    regime = optimizer.detect_market_regime(df)
    print(f"\n市场状态识别: {regime.value}")

    # 获取对应权重
    weights = optimizer.get_current_weights()
    print(f"\n动态权重配置:")
    print(f"  趋势因子: {weights.trend:.2%}")
    print(f"  动能因子: {weights.momentum:.2%}")
    print(f"  资金因子: {weights.money:.2%}")

    # 市场状态统计
    stats = optimizer.get_regime_statistics()
    print(f"\n市场状态统计:")
    print(f"  当前状态: {stats.get('current_regime', 'N/A')}")

    print("\n✅ 动态权重优化模块测试完成")
    return True


def test_risk_controller():
    """测试风险控制模块"""
    print("\n" + "="*60)
    print("测试 3: 风险控制模块")
    print("="*60)

    df = generate_mock_data(250)
    entry_price = df['close'].iloc[-1]

    controller = RiskController(
        default_risk_per_trade=0.02,
        max_position_size=0.1,
        atr_period=14,
        atr_multiplier=2.0
    )

    # 测试止损计算
    stop_result = controller.calculate_dynamic_stop_loss(
        df, entry_price, signal_strength=0.8
    )

    print(f"\n止损计算:")
    print(f"  入场价: {entry_price:.2f}")
    print(f"  止损价: {stop_result.stop_price:.2f}")
    print(f"  止损类型: {stop_result.stop_type.value}")
    print(f"  止损距离: {stop_result.distance_percent:.2%}")

    # 测试仓位计算
    capital = 100000
    position_result = controller.calculate_position_size(
        capital, entry_price, stop_result.stop_price
    )

    print(f"\n仓位计算 (资金: {capital}):")
    print(f"  建议股数: {position_result.shares:.0f}")
    print(f"  仓位金额: {position_result.position_value:.2f}")
    print(f"  风险金额: {position_result.risk_amount:.2f}")

    # 测试回撤预警
    alert = controller.check_drawdown_alert(90000, 100000)
    if alert:
        print(f"\n回撤预警:")
        print(f"  级别: {alert.level.value}")
        print(f"  回撤幅度: {alert.current_drawdown:.2%}")
        print(f"  建议: {alert.action_suggested}")

    print("\n✅ 风险控制模块测试完成")
    return True


def test_multi_timeframe():
    """测试多周期分析模块"""
    print("\n" + "="*60)
    print("测试 4: 多周期分析模块")
    print("="*60)

    df = generate_mock_data(250)
    scoring_system = ScoringSystem()

    analyzer = MultiTimeframeAnalyzer(scoring_system=scoring_system)

    # 重采样
    weekly_data = analyzer.resample_to_weekly(df)
    monthly_data = analyzer.resample_to_monthly(df)

    print(f"\n数据重采样:")
    print(f"  日线: {len(df)} 条")
    print(f"  周线: {len(weekly_data)} 条")
    print(f"  月线: {len(monthly_data)} 条")

    # 多周期分析
    result = analyzer.analyze_timeframe_alignment(df, weekly_data, monthly_data)

    print(f"\n多周期分析结果:")
    print(f"  日线评分: {result.daily.score:.2f}")
    print(f"  日线趋势: {result.daily.trend}")
    print(f"  日线信号: {result.daily.signal}")

    if result.weekly:
        print(f"  周线评分: {result.weekly.score:.2f}")
        print(f"  周线趋势: {result.weekly.trend}")

    print(f"  对齐状态: {result.alignment.value}")
    print(f"  综合评分: {result.combined_score:.2f}")
    print(f"  对齐奖励: {result.alignment_bonus:.2%}")
    print(f"  置信度: {result.confidence:.2%}")

    print("\n✅ 多周期分析模块测试完成")
    return True


def test_signal_backtest_report():
    """测试信号回测报告模块"""
    print("\n" + "="*60)
    print("测试 5: 信号回测报告模块")
    print("="*60)

    df = generate_mock_data(250)

    # 添加评分
    scoring_system = ScoringSystem()
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

    reporter = SignalBacktestReporter(score_threshold_buy=70)
    analysis = reporter.analyze_historical_signals(df_test)

    print(f"\n信号分析结果:")
    overall = analysis.overall
    print(f"  信号次数: {overall.signal_count}")
    print(f"  5日胜率: {overall.win_rate:.2%}")
    print(f"  平均5日收益: {overall.avg_return_5d:.2%}")
    print(f"  平均MFE: {overall.mfe_avg:.2%}")
    print(f"  平均MAE: {overall.mae_avg:.2%}")

    # 生成报告
    report = reporter.generate_report_markdown(analysis, "TEST")
    print(f"\n生成的报告预览:")
    print(report[:500] + "...")

    print("\n✅ 信号回测报告模块测试完成")
    return True


def test_score_strategy():
    """测试评分策略"""
    print("\n" + "="*60)
    print("测试 6: 评分策略模块")
    print("="*60)

    df = generate_mock_data(250)

    strategy = ScoreStrategy(
        buy_threshold=70,
        sell_threshold=50,
        use_dynamic_weights=True,
        use_multi_timeframe=True,
        use_risk_control=True
    )

    print(f"\n策略配置:")
    print(f"  名称: {strategy.get_name()}")
    print(f"  描述: {strategy.get_description()}")

    # 计算信号
    signals_df = strategy.calculate_signals(df)

    buy_signals = (signals_df['signal'] == 'buy').sum()
    sell_signals = (signals_df['signal'] == 'sell').sum()

    print(f"\n信号统计:")
    print(f"  买入信号: {buy_signals}")
    print(f"  卖出信号: {sell_signals}")
    print(f"  持仓信号: {len(signals_df) - buy_signals - sell_signals}")

    # 获取最近信号
    last_bar = df.iloc[-1].to_dict()
    last_bar['timestamp'] = df.index[-1] if isinstance(df.index[-1], datetime) else datetime.now()

    signal = strategy.get_signal(df.iloc[-1], df)

    print(f"\n最新信号:")
    print(f"  方向: {signal.get('direction', 'N/A')}")
    print(f"  评分: {signal.get('score', 0):.2f}")
    print(f"  调整后评分: {signal.get('adjusted_score', 0):.2f}")

    # 信号统计
    stats = strategy.get_signal_statistics()
    print(f"\n策略统计:")
    print(f"  总信号数: {stats.get('total_signals', 0)}")
    print(f"  买入信号: {stats.get('buy_signals', 0)}")

    print("\n✅ 评分策略模块测试完成")
    return True


def run_backtest():
    """运行回测"""
    print("\n" + "="*60)
    print("测试 7: 完整回测")
    print("="*60)

    df = generate_mock_data(365)  # 一年数据

    # 创建策略
    strategy = ScoreStrategy(
        buy_threshold=70,
        sell_threshold=50,
        use_dynamic_weights=True,
        use_risk_control=True
    )

    # 初始化回测引擎
    engine = BacktestEngine()
    engine.set_initial_cash(100000)

    # 准备数据
    data = {'TEST': df.copy()}

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

        print("\n✅ 回测完成")
        return True

    except Exception as e:
        print(f"\n回测执行出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("QuantTool 评分系统增强验证")
    print("="*60)

    results = {}

    # 运行所有测试
    tests = [
        ("评分验证模块", test_score_validator),
        ("动态权重优化", test_weight_optimizer),
        ("风险控制模块", test_risk_controller),
        ("多周期分析模块", test_multi_timeframe),
        ("信号回测报告", test_signal_backtest_report),
        ("评分策略模块", test_score_strategy),
        ("完整回测", run_backtest),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n❌ {name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # 汇总结果
    print("\n" + "="*60)
    print("测试汇总")
    print("="*60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    print(f"\n总计: {passed}/{total} 测试通过")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)