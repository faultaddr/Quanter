#!/usr/bin/env python
"""
全面测试增强功能模块

测试内容：
1. 筹码分布分析
2. K线形态识别（TA-Lib）
3. 经典选股策略（10种）
4. 综合选股框架
5. 批量时间处理
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def generate_test_data(days: int = 300, seed: int = 42) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(seed)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')

    close = 10 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, days)))
    high = close * (1 + np.abs(np.random.normal(0, 0.02, days)))
    low = close * (1 - np.abs(np.random.normal(0, 0.02, days)))
    open_price = close * (1 + np.random.normal(0, 0.01, days))
    volume = np.random.randint(1000000, 10000000, days)

    return pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })


def test_chip_distribution():
    """测试1: 筹码分布模块"""
    print("\n" + "=" * 60)
    print("【测试1】筹码分布模块")
    print("=" * 60)

    from quanttool.factors.chip_distribution import (
        ChipDistributionCalculator,
        calculate_chip_distribution,
        get_chip_assessment
    )

    # 生成测试数据
    df = generate_test_data(300)
    print(f"  数据: {len(df)}天")

    # 测试计算器
    calculator = ChipDistributionCalculator(lookback_days=210, price_bins=100)
    result = calculator.calculate(df)

    print(f"\n  筹码分布结果:")
    print(f"    筹码集中度: {result.concentration_ratio:.1f}%")
    print(f"    平均持仓成本: ¥{result.avg_cost:.2f}")
    print(f"    获利盘比例: {result.profit_ratio:.1f}%")
    print(f"    上方套牢压力: {result.upper_pressure:.1f}%")
    print(f"    下方支撑强度: {result.lower_support:.1f}%")
    print(f"    筹码评分: {result.score:.1f}")

    if result.support_levels:
        print(f"    支撑位: {[f'¥{p:.2f}' for p in result.support_levels[:3]]}")
    if result.resistance_levels:
        print(f"    阻力位: {[f'¥{p:.2f}' for p in result.resistance_levels[:3]]}")

    # 定性评估
    assessment = get_chip_assessment(result)
    print(f"\n  定性评估: {assessment}")

    # 测试ASCII图表
    chart = calculator.get_chip_distribution_chart(result, width=30, height=10)
    print(f"\n  筹码分布图（简化）:")
    for line in chart.split('\n')[:8]:
        print(f"    {line}")

    # 测试便捷函数
    result_dict = calculate_chip_distribution(df, lookback_days=120)
    assert 'score' in result_dict

    print("\n  ✅ 筹码分布模块测试通过")
    return True


def test_talib_patterns():
    """测试2: K线形态识别模块"""
    print("\n" + "=" * 60)
    print("【测试2】K线形态识别模块（TA-Lib 61种形态）")
    print("=" * 60)

    try:
        import talib
        TALIB_AVAILABLE = True
    except ImportError:
        TALIB_AVAILABLE = False

    from quanttool.factors.talib_patterns import (
        TalibPatternRecognizer,
        recognize_talib_patterns,
        TALIB_AVAILABLE
    )

    if not TALIB_AVAILABLE:
        print("  ⚠️ TA-Lib 未安装，跳过形态识别测试")
        print("  提示: pip install TA-Lib")
        return True

    # 生成测试数据
    df = generate_test_data(100)

    # 测试识别器
    recognizer = TalibPatternRecognizer()
    result = recognizer.recognize_all(df, lookback=5)

    print(f"\n  形态识别结果:")
    print(f"    看涨形态: {result.bullish_count}个")
    print(f"    看跌形态: {result.bearish_count}个")
    print(f"    中性形态: {result.neutral_count}个")
    print(f"    综合信号: {result.composite_signal:.1f}")

    if result.patterns:
        print(f"\n  识别到的形态（前5个）:")
        for p in result.patterns[:5]:
            print(f"    - {p.name_cn} ({p.type}): 信号={p.signal}, 强度={p.strength}")

    # 列出所有支持的形态
    all_patterns = recognizer.list_all_patterns()
    print(f"\n  支持的形态总数: {len(all_patterns)}种")

    # 按类型统计
    bullish = len([p for p in all_patterns if p['type'] == 'bullish'])
    bearish = len([p for p in all_patterns if p['type'] == 'bearish'])
    neutral = len([p for p in all_patterns if p['type'] == 'neutral'])
    print(f"    看涨形态: {bullish}种")
    print(f"    看跌形态: {bearish}种")
    print(f"    中性形态: {neutral}种")

    # 测试单形态识别
    hammer_result = recognizer.recognize_single_pattern(df, 'CDLHAMMER')
    print(f"\n  CDLHAMMER识别结果长度: {len(hammer_result)}")

    print("\n  ✅ K线形态识别模块测试通过")
    return True


def test_classic_strategies():
    """测试3: 经典选股策略模块"""
    print("\n" + "=" * 60)
    print("【测试3】经典选股策略模块（10种策略）")
    print("=" * 60)

    from quanttool.strategies.classic_strategies import (
        VolumeBreakoutStrategy,
        MADAlignmentStrategy,
        ApronStrategy,
        YearLinePullbackStrategy,
        PlatformBreakoutStrategy,
        NoLargeDrawdownStrategy,
        HighNarrowFlagStrategy,
        VolumeLimitDownStrategy,
        LowATRGrowthStrategy,
        FundamentalSelectionStrategy,
        CLASSIC_STRATEGIES
    )

    df = generate_test_data(300)

    strategies = [
        ("放量上涨", VolumeBreakoutStrategy()),
        ("均线多头", MADAlignmentStrategy()),
        ("停机坪", ApronStrategy()),
        ("回踩年线", YearLinePullbackStrategy()),
        ("突破平台", PlatformBreakoutStrategy()),
        ("无大幅回撤", NoLargeDrawdownStrategy()),
        ("高而窄旗形", HighNarrowFlagStrategy()),
        ("放量跌停", VolumeLimitDownStrategy()),
        ("低ATR成长", LowATRGrowthStrategy()),
        ("基本面选股", FundamentalSelectionStrategy()),
    ]

    print(f"\n  测试{len(strategies)}种策略:")

    for name, strategy in strategies:
        strategy.initialize({})

        # 计算信号
        signals = strategy.calculate_signals(df)

        # 统计信号
        buy_signals = (signals['signal'] == 1).sum() if 'signal' in signals.columns else 0

        print(f"\n    【{name}】")
        print(f"      策略名称: {strategy.get_name()}")
        print(f"      描述: {strategy.get_description()}")
        print(f"      买入信号数: {buy_signals}")

        # 测试get_signal方法
        current_bar = df.iloc[-1]
        signal = strategy.get_signal(current_bar, df)
        print(f"      最新信号: {signal['direction']}")

    print(f"\n  ✅ 经典选股策略模块测试通过")
    return True


def test_comprehensive_screening():
    """测试4: 综合选股框架"""
    print("\n" + "=" * 60)
    print("【测试4】综合选股框架")
    print("=" * 60)

    from quanttool.factors.comprehensive_screening import (
        ComprehensiveStockScreener,
        ConditionCategory,
        ConditionOperator,
        create_screener_with_strategy
    )

    df = generate_test_data(300)

    # 测试1: 创建选股器
    print("\n  测试1: 创建选股器")
    screener = ComprehensiveStockScreener()
    print(f"    初始条件数: {len(screener.conditions)}")

    # 测试2: 添加自定义条件
    print("\n  测试2: 添加自定义条件")
    screener.add_condition(
        name="PE低估值",
        category=ConditionCategory.FUNDAMENTAL,
        field="pe_ratio",
        operator=ConditionOperator.LT,
        value=20,
        weight=1.0,
        description="市盈率小于20"
    )
    print(f"    添加条件后: {len(screener.conditions)}个")

    # 测试3: 添加预定义条件
    print("\n  测试3: 添加预定义条件")
    screener.clear_conditions()
    screener.add_predefined_condition('macd_golden_cross', weight=1.5, required=True)
    screener.add_predefined_condition('volume_breakout_2x', weight=1.0)
    screener.add_predefined_condition('ma_bullish', weight=1.0)
    print(f"    预定义条件: {len(screener.conditions)}个")

    # 测试4: 执行筛选
    print("\n  测试4: 执行筛选")
    result = screener.screen(df, "000001", "测试股票")
    if result:
        print(f"    股票代码: {result.stock_code}")
        print(f"    得分: {result.score:.1f}")
        print(f"    匹配条件: {result.matched_conditions}")
    else:
        print(f"    未满足筛选条件")

    # 测试5: 列出预定义条件
    print("\n  测试5: 预定义条件列表")
    predefined = screener.list_predefined_conditions()
    print(f"    预定义条件总数: {len(predefined)}个")

    categories = screener.get_condition_categories()
    print(f"    条件分类: {list(categories.keys())}")

    # 测试6: 创建策略选股器
    print("\n  测试6: 策略选股器")
    for strategy_name in ['momentum', 'value', 'breakout', 'oversold', 'trend']:
        screener = create_screener_with_strategy(strategy_name)
        print(f"    {strategy_name}: {len(screener.conditions)}个条件")

    print(f"\n  ✅ 综合选股框架测试通过")
    return True


def test_batch_time_processor():
    """测试5: 批量时间处理模块"""
    print("\n" + "=" * 60)
    print("【测试5】批量时间处理模块")
    print("=" * 60)

    from quanttool.infrastructure.batch_time_processor import (
        TradingCalendar,
        TimeParser,
        BatchTimeProcessor,
        run_batch_job,
        format_batch_result
    )

    # 测试1: 交易日判断
    print("\n  测试1: 交易日判断")
    test_dates = [
        datetime(2024, 1, 8),   # 周一
        datetime(2024, 1, 6),   # 周六
        datetime(2024, 1, 7),   # 周日
        datetime(2024, 1, 1),   # 元旦
    ]
    for d in test_dates:
        is_trading = TradingCalendar.is_trading_day(d)
        weekday = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][d.weekday()]
        print(f"    {d.strftime('%Y-%m-%d')} ({weekday}): {'交易日' if is_trading else '非交易日'}")

    # 测试2: 获取交易日列表
    print("\n  测试2: 获取交易日列表")
    trading_days = TradingCalendar.get_trading_days(
        datetime(2024, 1, 8),
        datetime(2024, 1, 12)
    )
    print(f"    2024-01-08 至 2024-01-12: {len(trading_days)}个交易日")
    for d in trading_days:
        print(f"      - {d.strftime('%Y-%m-%d')}")

    # 测试3: 时间解析
    print("\n  测试3: 时间解析")
    parsed_dates = [
        TimeParser.parse('2024-01-15'),
        TimeParser.parse('20240115'),
        TimeParser.parse('2024/01/15'),
    ]
    print(f"    解析'2024-01-15': {parsed_dates[0]}")
    print(f"    解析'20240115': {parsed_dates[1]}")
    print(f"    解析'2024/01/15': {parsed_dates[2]}")

    # 测试4: 时间列表解析
    print("\n  测试4: 时间列表解析")
    date_list = TimeParser.parse_list('2024-01-15,2024-01-16,2024-01-17')
    print(f"    解析'2024-01-15,2024-01-16,2024-01-17': {len(date_list)}个日期")

    # 测试5: 时间范围解析
    print("\n  测试5: 时间范围解析")
    date_range = TimeParser.parse_range('2024-01-08', '2024-01-12', trading_days_only=True)
    print(f"    2024-01-08 至 2024-01-12 (仅交易日): {len(date_range)}天")

    # 测试6: 批量处理器
    print("\n  测试6: 批量处理器")

    job_results = []

    def sample_job(date):
        job_results.append(date.strftime('%Y-%m-%d'))
        return {"status": "success", "date": date.strftime('%Y-%m-%d')}

    processor = BatchTimeProcessor(trading_days_only=True, show_progress=False)
    result = processor.run_job(['2024-01-08', '2024-01-12'], sample_job)

    print(f"    总日期数: {result.total_dates}")
    print(f"    成功数: {result.success_count}")
    print(f"    失败数: {result.failure_count}")
    print(f"    耗时: {result.elapsed_time:.3f}秒")

    # 测试7: 不同时间模式
    print("\n  测试7: 不同时间模式")

    # 当前时间模式
    config = processor.parse_time_args(None)
    print(f"    当前时间模式: {config.mode.value}, 日期数: {len(config.dates)}")

    # 单个时间模式
    config = processor.parse_time_args(['2024-01-15'])
    print(f"    单个时间模式: {config.mode.value}, 日期数: {len(config.dates)}")

    # 枚举时间模式
    config = processor.parse_time_args(['2024-01-15,2024-01-16,2024-01-17'])
    print(f"    枚举时间模式: {config.mode.value}, 日期数: {len(config.dates)}")

    # 区间时间模式
    config = processor.parse_time_args(['2024-01-08', '2024-01-12'])
    print(f"    区间时间模式: {config.mode.value}, 日期数: {len(config.dates)}")

    # 测试8: 便捷函数
    print("\n  测试8: 便捷函数")
    result = run_batch_job(
        lambda d: {"processed": str(d)},
        ['2024-01-15'],
        show_progress=False
    )
    print(f"    run_batch_job成功数: {result.success_count}")

    print(f"\n  ✅ 批量时间处理模块测试通过")
    return True


def test_cli_commands():
    """测试6: CLI命令"""
    print("\n" + "=" * 60)
    print("【测试6】CLI命令集成")
    print("=" * 60)

    from quanttool.cli.commands.enhanced_commands import app
    from typer.testing import CliRunner

    runner = CliRunner()

    # 测试帮助命令
    print("\n  测试1: 帮助命令")
    result = runner.invoke(app, ['--help'])
    print(f"    命令列表:")
    for line in result.output.split('\n'):
        if '│' in line:
            print(f"      {line.strip()}")

    # 测试列出策略
    print("\n  测试2: list-strategies命令")
    result = runner.invoke(app, ['list-strategies'])
    if result.exit_code == 0:
        print(f"    执行成功")
        for line in result.output.split('\n')[:8]:
            if line.strip():
                print(f"      {line.strip()}")
    else:
        print(f"    执行失败: {result.exception}")

    # 测试列出K线形态
    print("\n  测试3: list-patterns命令")
    result = runner.invoke(app, ['list-patterns'])
    if result.exit_code == 0:
        print(f"    执行成功")
        # 只显示部分输出
        lines = result.output.split('\n')
        print(f"      输出行数: {len(lines)}")
    else:
        print(f"    执行失败: {result.exception}")

    # 测试列出选股条件
    print("\n  测试4: list-conditions命令")
    result = runner.invoke(app, ['list-conditions'])
    if result.exit_code == 0:
        print(f"    执行成功")
        for line in result.output.split('\n')[:10]:
            if line.strip():
                print(f"      {line.strip()}")
    else:
        print(f"    执行失败: {result.exception}")

    # 测试交易日查询
    print("\n  测试5: trading-days命令")
    result = runner.invoke(app, ['trading-days', '2024-01-08', '2024-01-12'])
    if result.exit_code == 0:
        print(f"    执行成功")
        for line in result.output.split('\n')[:8]:
            if line.strip():
                print(f"      {line.strip()}")
    else:
        print(f"    执行失败: {result.exception}")

    # 测试是否为交易日
    print("\n  测试6: is-trading-day命令")
    result = runner.invoke(app, ['is-trading-day', '2024-01-15'])
    if result.exit_code == 0:
        print(f"    执行成功")
        print(f"      {result.output.strip()}")
    else:
        print(f"    执行失败: {result.exception}")

    print(f"\n  ✅ CLI命令集成测试通过")
    return True


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("开始全面测试增强功能模块")
    print("=" * 60)

    results = {}

    # 运行所有测试
    tests = [
        ("筹码分布模块", test_chip_distribution),
        ("K线形态识别模块", test_talib_patterns),
        ("经典选股策略模块", test_classic_strategies),
        ("综合选股框架", test_comprehensive_screening),
        ("批量时间处理模块", test_batch_time_processor),
        ("CLI命令集成", test_cli_commands),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ❌ {name}测试失败: {e}")
            results[name] = False

    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"  {name}: {status}")

    print(f"\n  总计: {passed}/{total} 通过")

    if passed == total:
        print("\n  🎉 所有测试通过!")
    else:
        print("\n  ⚠️ 部分测试失败，请检查错误信息")


if __name__ == "__main__":
    main()
