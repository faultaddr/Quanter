"""
增强功能 CLI 命令

集成以下新功能：
- 筹码分布分析
- TA-Lib K线形态识别
- 经典选股策略
- 综合选股
- 批量时间处理
"""
import typer
import sys
import os
from datetime import datetime
from typing import Optional, List
from enum import Enum

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

from quanttool.factors.chip_distribution import (
    ChipDistributionCalculator,
    calculate_chip_distribution,
    get_chip_assessment
)
from quanttool.factors.talib_patterns import (
    TalibPatternRecognizer,
    recognize_talib_patterns,
    format_patterns_report,
    get_pattern_assessment,
    TALIB_AVAILABLE
)
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
from quanttool.factors.comprehensive_screening import (
    ComprehensiveStockScreener,
    ConditionCategory,
    ConditionOperator,
    create_screener_with_strategy
)
from quanttool.infrastructure.batch_time_processor import (
    BatchTimeProcessor,
    TimeParser,
    TradingCalendar,
    run_batch_job,
    format_batch_result
)
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import create_data_fetcher_with_credentials, EnhancedDataFetcher

app = typer.Typer(help="增强功能：筹码分析、K线形态、经典策略、综合选股、批量处理")


def _get_single_stock_data(symbol: str, days: int = 300) -> pd.DataFrame:
    """获取单只股票数据"""
    try:
        # 使用 Ashare 的 get_price 方法（最快）
        df = EnhancedDataFetcher.get_price(symbol, count=days)
        if df is not None and not df.empty:
            # 重命名列以匹配我们的格式
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})
            return df
    except Exception:
        pass

    # 备用方法：使用 fetcher
    try:
        fetcher = create_data_fetcher_with_credentials()
        fetcher.initialize()
        from datetime import datetime, timedelta
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 50)
        result = fetcher.get_bars([symbol], start_date, end_date)
        if symbol in result:
            return result[symbol]
    except Exception:
        pass

    return pd.DataFrame()


# ==================== 筹码分布命令 ====================

@app.command(name="chip")
def analyze_chip_distribution(
    symbol: str = typer.Argument(..., help="股票代码"),
    days: int = typer.Option(210, "--days", "-d", help="回看天数（默认210日）"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件")
):
    """
    分析股票筹码分布

    计算筹码集中度、获利盘比例、支撑阻力位等

    示例：
    - quanttool enhanced chip 000001
    - quanttool enhanced chip 000001 --days 120
    """
    typer.echo(f"正在分析筹码分布：{symbol}")
    typer.echo(f"回看天数：{days}")

    # 获取数据
    df = _get_single_stock_data(symbol, days + 50)

    if df is None or len(df) < 30:
        typer.echo("错误：数据不足")
        return

    # 计算筹码分布
    calculator = ChipDistributionCalculator(lookback_days=days)
    result = calculator.calculate(df)

    # 输出结果
    typer.echo("\n" + "=" * 50)
    typer.echo("筹码分布分析结果")
    typer.echo("=" * 50)
    typer.echo(f"筹码集中度: {result.concentration_ratio:.1f}%")
    typer.echo(f"平均持仓成本: ¥{result.avg_cost:.2f}")
    typer.echo(f"获利盘比例: {result.profit_ratio:.1f}%")
    typer.echo(f"上方套牢压力: {result.upper_pressure:.1f}%")
    typer.echo(f"下方支撑强度: {result.lower_support:.1f}%")
    typer.echo(f"筹码评分: {result.score:.1f}")

    if result.support_levels:
        typer.echo(f"\n支撑位: {', '.join([f'¥{p:.2f}' for p in result.support_levels])}")
    if result.resistance_levels:
        typer.echo(f"阻力位: {', '.join([f'¥{p:.2f}' for p in result.resistance_levels])}")

    if result.peak_prices:
        typer.echo(f"\n筹码峰价格: {', '.join([f'¥{p:.2f}' for p in result.peak_prices[:3]])}")

    # 定性评估
    assessment = get_chip_assessment(result)
    typer.echo(f"\n评估: {assessment}")

    # 生成ASCII图表
    chart = calculator.get_chip_distribution_chart(result)
    typer.echo("\n" + chart)

    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(chart)
        typer.echo(f"\n结果已保存至：{output}")


# ==================== K线形态识别命令 ====================

@app.command(name="patterns")
def recognize_patterns(
    symbol: str = typer.Argument(..., help="股票代码"),
    lookback: int = typer.Option(5, "--lookback", "-l", help="回顾天数"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件")
):
    """
    识别K线形态（TA-Lib 61种形态）

    示例：
    - quanttool enhanced patterns 000001
    - quanttool enhanced patterns 000001 --lookback 10
    """
    if not TALIB_AVAILABLE:
        typer.echo("错误：TA-Lib 未安装，无法使用此功能")
        typer.echo("请安装 TA-Lib: pip install TA-Lib")
        return

    typer.echo(f"正在识别K线形态：{symbol}")

    # 获取数据
    df = _get_single_stock_data(symbol, 300)

    if df is None or len(df) < 10:
        typer.echo("错误：数据不足")
        return

    # 识别形态
    recognizer = TalibPatternRecognizer()
    result = recognizer.recognize_all(df, lookback)

    # 格式化报告
    report = format_patterns_report(result)
    typer.echo(report)

    # 综合评估
    assessment = get_pattern_assessment(result)
    typer.echo(f"\n综合评估: {assessment}")

    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(report)
        typer.echo(f"\n结果已保存至：{output}")


@app.command(name="list-patterns")
def list_all_patterns():
    """列出所有支持的K线形态"""
    recognizer = TalibPatternRecognizer()
    patterns = recognizer.list_all_patterns()

    typer.echo(f"支持的K线形态（共{len(patterns)}种）：\n")

    # 按类型分组
    bullish = [p for p in patterns if p['type'] == 'bullish']
    bearish = [p for p in patterns if p['type'] == 'bearish']
    neutral = [p for p in patterns if p['type'] == 'neutral']

    typer.echo("【看涨形态】")
    for p in bullish:
        typer.echo(f"  {p['name_cn']} ({p['name']}): {p['description']}")

    typer.echo("\n【看跌形态】")
    for p in bearish:
        typer.echo(f"  {p['name_cn']} ({p['name']}): {p['description']}")

    typer.echo("\n【中性形态】")
    for p in neutral:
        typer.echo(f"  {p['name_cn']} ({p['name']}): {p['description']}")


# ==================== 经典策略命令 ====================

class StrategyChoice(str, Enum):
    """策略选择"""
    VOLUME_BREAKOUT = "volume_breakout"
    MA_ALIGNMENT = "ma_alignment"
    APRON = "apron"
    YEAR_LINE_PULLBACK = "year_line_pullback"
    PLATFORM_BREAKOUT = "platform_breakout"
    NO_LARGE_DRAWDOWN = "no_large_drawdown"
    HIGH_NARROW_FLAG = "high_narrow_flag"
    VOLUME_LIMIT_DOWN = "volume_limit_down"
    LOW_ATR_GROWTH = "low_atr_growth"
    FUNDAMENTAL = "fundamental"


@app.command(name="strategy")
def run_classic_strategy(
    symbol: str = typer.Argument(..., help="股票代码"),
    strategy: StrategyChoice = typer.Option(
        StrategyChoice.VOLUME_BREAKOUT,
        "--strategy", "-s",
        help="策略类型"
    ),
    days: int = typer.Option(250, "--days", "-d", help="分析天数"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件")
):
    """
    运行经典选股策略

    策略类型：
    - volume_breakout: 放量上涨策略
    - ma_alignment: 均线多头策略
    - apron: 停机坪策略
    - year_line_pullback: 回踩年线策略
    - platform_breakout: 突破平台策略
    - no_large_drawdown: 无大幅回撤策略
    - high_narrow_flag: 高而窄的旗形策略
    - volume_limit_down: 放量跌停策略
    - low_atr_growth: 低ATR成长策略
    - fundamental: 基本面选股策略

    示例：
    - quanttool enhanced strategy 000001 --strategy ma_alignment
    """
    typer.echo(f"正在运行策略：{strategy.value}")
    typer.echo(f"股票代码：{symbol}")

    # 获取数据
    df = _get_single_stock_data(symbol, days + 50)

    if df is None or len(df) < 60:
        typer.echo("错误：数据不足")
        return

    # 选择策略
    strategy_map = {
        StrategyChoice.VOLUME_BREAKOUT: VolumeBreakoutStrategy,
        StrategyChoice.MA_ALIGNMENT: MADAlignmentStrategy,
        StrategyChoice.APRON: ApronStrategy,
        StrategyChoice.YEAR_LINE_PULLBACK: YearLinePullbackStrategy,
        StrategyChoice.PLATFORM_BREAKOUT: PlatformBreakoutStrategy,
        StrategyChoice.NO_LARGE_DRAWDOWN: NoLargeDrawdownStrategy,
        StrategyChoice.HIGH_NARROW_FLAG: HighNarrowFlagStrategy,
        StrategyChoice.VOLUME_LIMIT_DOWN: VolumeLimitDownStrategy,
        StrategyChoice.LOW_ATR_GROWTH: LowATRGrowthStrategy,
        StrategyChoice.FUNDAMENTAL: FundamentalSelectionStrategy,
    }

    strategy_cls = strategy_map[strategy]
    strategy_instance = strategy_cls()
    strategy_instance.initialize({})

    # 计算信号
    signals = strategy_instance.calculate_signals(df.tail(days))

    # 输出结果
    typer.echo("\n" + "=" * 50)
    typer.echo(f"策略：{strategy_instance.get_description()}")
    typer.echo("=" * 50)

    # 找到有信号的日期
    signal_dates = signals[signals['signal'] != 0] if 'signal' in signals.columns else pd.DataFrame()

    if len(signal_dates) > 0:
        typer.echo(f"\n发现 {len(signal_dates)} 个信号：")
        for _, row in signal_dates.tail(10).iterrows():
            date = row.get('timestamp', '未知日期')
            signal_type = "买入" if row['signal'] == 1 else "卖出"
            typer.echo(f"  {date}: {signal_type}信号")
    else:
        typer.echo("\n未发现交易信号")

    if output:
        signals.to_csv(output, index=False)
        typer.echo(f"\n信号数据已保存至：{output}")


@app.command(name="list-strategies")
def list_all_strategies():
    """列出所有经典策略"""
    typer.echo("经典选股策略列表：\n")

    for key, cls, desc in CLASSIC_STRATEGIES:
        instance = cls()
        typer.echo(f"【{desc}】")
        typer.echo(f"  键名: {key}")
        typer.echo(f"  描述: {instance.get_description()}")
        typer.echo()


# ==================== 综合选股命令 ====================

class ScreeningStrategy(str, Enum):
    """选股策略"""
    MOMENTUM = "momentum"
    VALUE = "value"
    BREAKOUT = "breakout"
    OVERSOLD = "oversold"
    TREND = "trend"
    CUSTOM = "custom"


@app.command(name="screen")
def comprehensive_screening(
    index: str = typer.Option("hs300", "--index", "-i", help="指数范围 (hs300/csi1000/all)"),
    strategy: ScreeningStrategy = typer.Option(
        ScreeningStrategy.MOMENTUM,
        "--strategy", "-s",
        help="选股策略"
    ),
    top_n: int = typer.Option(20, "--top", "-t", help="返回前N只股票"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件")
):
    """
    综合选股

    从指定指数范围内筛选符合条件的股票

    示例：
    - quanttool enhanced screen --index hs300 --strategy momentum --top 20
    - quanttool enhanced screen --index csi1000 --strategy value
    """
    typer.echo(f"正在进行综合选股...")
    typer.echo(f"指数范围: {index}")
    typer.echo(f"策略: {strategy.value}")

    # 获取股票列表
    try:
        fetcher = create_data_fetcher_with_credentials()
        fetcher.initialize()

        if index == "hs300":
            constituents = fetcher.get_csi300_constituents(include_names=True)
        elif index == "csi1000":
            constituents = fetcher.get_csi1000_constituents(include_names=True)
        else:
            constituents = fetcher.get_all_stocks()

        if not constituents:
            typer.echo("错误：无法获取股票列表")
            return

        typer.echo(f"共获取 {len(constituents)} 只股票")

    except Exception as e:
        typer.echo(f"获取股票列表失败: {e}")
        return

    # 创建选股器
    if strategy == ScreeningStrategy.CUSTOM:
        screener = ComprehensiveStockScreener()
        # 添加默认条件
        screener.add_predefined_condition('macd_golden_cross', weight=1.0, required=True)
        screener.add_predefined_condition('volume_breakout_2x', weight=0.5)
    else:
        screener = create_screener_with_strategy(strategy.value)

    # 批量筛选
    results = []
    total = len(constituents)

    for i, stock in enumerate(constituents):
        code = stock['code'] if isinstance(stock, dict) else stock
        name = stock.get('name', '') if isinstance(stock, dict) else ''

        if (i + 1) % 50 == 0:
            typer.echo(f"处理进度: {i+1}/{total}")

        try:
            df = _get_single_stock_data(code, 300)
            if df is not None and len(df) >= 60:
                result = screener.screen(df, code, name)
                if result:
                    results.append(result)
        except Exception:
            continue

    # 排序
    results.sort(key=lambda x: x.score, reverse=True)
    results = results[:top_n]

    # 输出结果
    typer.echo("\n" + "=" * 60)
    typer.echo(f"综合选股结果（前{len(results)}名）")
    typer.echo("=" * 60)

    if results:
        typer.echo(f"\n{'排名':<6}{'代码':<12}{'名称':<10}{'得分':<10}{'匹配条件'}")
        typer.echo("-" * 60)

        for r in results:
            conditions = ', '.join(r.matched_conditions[:3])
            typer.echo(f"{r.rank:<6}{r.stock_code:<12}{r.stock_name:<10}{r.score:<10.1f}{conditions}")
    else:
        typer.echo("\n未找到符合条件的股票")

    if output:
        # 保存为JSON
        import json
        output_data = [
            {
                'rank': r.rank,
                'code': r.stock_code,
                'name': r.stock_name,
                'score': r.score,
                'conditions': r.matched_conditions
            }
            for r in results
        ]
        with open(output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        typer.echo(f"\n结果已保存至：{output}")


@app.command(name="list-conditions")
def list_screening_conditions():
    """列出所有预定义选股条件"""
    screener = ComprehensiveStockScreener()
    categories = screener.get_condition_categories()

    typer.echo("预定义选股条件：\n")

    for cat_name, conditions in categories.items():
        typer.echo(f"【{cat_name}】")
        for c in conditions:
            typer.echo(f"  {c['key']}: {c['name']} - {c['description']}")
        typer.echo()


# ==================== 批量时间处理命令 ====================

@app.command(name="batch")
def batch_process(
    time_args: List[str] = typer.Argument(
        None,
        help="时间参数（空=当前，单个日期，枚举日期，或开始结束日期）"
    ),
    job_type: str = typer.Option("analysis", "--job", "-j", help="作业类型 (analysis/screen/backtest)"),
    symbol: Optional[str] = typer.Option(None, "--symbol", "-s", help="股票代码（单股票作业）"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出目录")
):
    """
    批量时间处理

    时间参数格式：
    - 空：当前时间作业
    - 2024-01-15：单个时间作业
    - 2024-01-01,2024-01-02,2024-01-03：枚举时间作业
    - 2024-01-01 2024-01-31：区间时间作业

    示例：
    - quanttool enhanced batch
    - quanttool enhanced batch 2024-01-15
    - quanttool enhanced batch 2024-01-01,2024-01-02,2024-01-03
    - quanttool enhanced batch 2024-01-01 2024-01-31
    """
    typer.echo("批量时间处理")
    typer.echo(f"时间参数: {time_args if time_args else '当前时间'}")
    typer.echo(f"作业类型: {job_type}")

    # 定义作业函数
    def job_func(date: datetime):
        typer.echo(f"  处理日期: {date.strftime('%Y-%m-%d')}")
        # 这里可以调用具体的分析或选股函数
        return {"date": date.strftime('%Y-%m-%d'), "status": "success"}

    # 运行批量作业
    processor = BatchTimeProcessor(trading_days_only=True, show_progress=True)
    time_config = processor.parse_time_args(time_args)

    typer.echo(f"\n待处理日期数: {len(time_config.dates)}")
    typer.echo(f"时间模式: {time_config.mode.value}")

    result = processor.run_job(time_args, job_func)

    # 输出结果
    typer.echo("\n" + format_batch_result(result))


@app.command(name="trading-days")
def get_trading_days(
    start: str = typer.Argument(..., help="开始日期"),
    end: str = typer.Argument(..., help="结束日期")
):
    """
    获取交易日列表

    示例：
    - quanttool enhanced trading-days 2024-01-01 2024-01-31
    """
    start_date = TimeParser.parse(start)
    end_date = TimeParser.parse(end)

    trading_days = TradingCalendar.get_trading_days(start_date, end_date)

    typer.echo(f"\n{start} 至 {end} 共 {len(trading_days)} 个交易日：\n")

    for i, day in enumerate(trading_days):
        typer.echo(f"{i+1:3}. {day.strftime('%Y-%m-%d')} ({['周一','周二','周三','周四','周五'][day.weekday()]})")


@app.command(name="is-trading-day")
def check_trading_day(
    date: str = typer.Argument(..., help="日期")
):
    """
    检查是否为交易日

    示例：
    - quanttool enhanced is-trading-day 2024-01-15
    """
    check_date = TimeParser.parse(date)
    is_trading = TradingCalendar.is_trading_day(check_date)

    weekday = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][check_date.weekday()]

    typer.echo(f"\n{date} ({weekday})")
    if is_trading:
        typer.echo("✅ 是交易日")
    else:
        typer.echo("❌ 不是交易日")


# ==================== GBM 沪深300 荐股命令 ====================

@app.command(name="gbm-pick")
def gbm_csi300_pick(
    top_n: int = typer.Option(10, "--top", "-n", help="返回前 N 只股票"),
    force_train: bool = typer.Option(False, "--train", "-t", help="强制重新训练模型"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件路径"),
    json_output: bool = typer.Option(False, "--json", "-j", help="JSON 格式输出"),
):
    """
    GBM 沪深300 每日荐股

    使用 LightGBM 模型对沪深300成分股进行预测，返回 Top N 推荐股票

    示例：
    - quanttool enhanced gbm-pick
    - quanttool enhanced gbm-pick -n 20
    - quanttool enhanced gbm-pick --train
    - quanttool enhanced gbm-pick -n 10 -o report.md
    """
    typer.echo(f"\n{'='*60}")
    typer.echo("GBM 沪深300 每日荐股")
    typer.echo(f"{'='*60}\n")

    if force_train:
        typer.echo("⚠️  强制重新训练模型...")

    try:
        from quanttool.application.gbm_picker_service import (
            GBMCsi300Picker,
            format_pick_report
        )

        # 创建荐股器
        picker = GBMCsi300Picker(top_n=top_n)

        # 获取推荐
        result = picker.get_daily_picks(force_train=force_train)

        if json_output:
            import json
            output_data = {
                "date": result.date,
                "total_stocks": result.total_stocks,
                "valid_stocks": result.valid_stocks,
                "top_stocks": [
                    {
                        "code": r.code,
                        "pred_return": r.pred_return,
                        "probability": r.probability,
                        "percentile": r.percentile,
                        "confidence": r.confidence,
                        "signal": r.signal,
                        "stop_loss": r.stop_loss,
                        "take_profit": r.take_profit,
                        "close": r.close,
                    }
                    for r in result.top_stocks
                ],
                "model_info": result.model_info,
            }
            report = json.dumps(output_data, ensure_ascii=False, indent=2)
        else:
            report = format_pick_report(result)

        # 输出
        typer.echo(report)

        # 保存文件
        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(report)
            typer.echo(f"\n结果已保存至：{output}")

    except Exception as e:
        typer.echo(f"❌ 荐股失败: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
