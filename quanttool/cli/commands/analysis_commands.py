"""Commands for stock analysis."""
import typer
import sys
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from quanttool.factors.stock_analyzer import StockAnalyzer
from quanttool.factors.scoring_system import ScoringSystem
from quanttool.factors.trend_scoring_system import TrendScoringSystem, analyze_trend_quality
from quanttool.factors.breakout_scoring_system import BreakoutScoringSystem, analyze_breakout_quality
from quanttool.factors.trend_momentum_scoring import TrendMomentumScoring
from quanttool.factors.analysis_context import ScoringSystemType
from quanttool.strategies.score_strategy import ScoreStrategy
from quanttool.strategies.trend_strategy import TrendStrategy
from quanttool.strategies.adaptive_threshold import (
    AdaptiveThresholdManager,
    get_adaptive_thresholds,
    IndexMarketDetector,
    DualMarketState,
    MarketRegime,
    CombinedSignal,
)
from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials
from quanttool.infrastructure.stores.meta_db import MetaDB
import pandas as pd
import json

app = typer.Typer()


class SystemChoice(str, Enum):
    """评分系统选择"""
    AUTO = "auto"         # 自动选择（根据市场状态）
    CLASSIC = "classic"   # 经典多因子评分
    TREND = "trend"       # 趋势强度评分
    BREAKOUT = "breakout" # 低位盘整突破评分
    MOMENTUM = "momentum" # 趋势动量评分


@app.command(name="single")
def analyze_single(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze (e.g., 601777, 000001.SZ)"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    system: SystemChoice = typer.Option(
        SystemChoice.AUTO,
        "--system", "-s",
        help="Primary scoring system: auto (default), classic, trend, breakout"
    ),
    use_context: bool = typer.Option(
        False,
        "--unified", "-u",
        help="Use unified analysis context (recommended for consistent reports)"
    ),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file to save the analysis report")
):
    """
    Analyze a single stock with technical indicators and trading strategies.

    评分系统选项：
    - auto: 根据市场状态自动选择主评分系统（推荐）
    - classic: 经典多因子评分（趋势+动能+资金因子）
    - trend: 趋势强度评分（纯趋势分析）
    - breakout: 低位盘整突破评分（寻找低位突破机会）

    示例：
    - quanttool analysis single 000001
    - quanttool analysis single 000001 --system trend
    - quanttool analysis single 000001 --unified
    """
    # 映射到 ScoringSystemType
    system_map = {
        SystemChoice.AUTO: ScoringSystemType.AUTO,
        SystemChoice.CLASSIC: ScoringSystemType.CLASSIC,
        SystemChoice.TREND: ScoringSystemType.TREND,
        SystemChoice.BREAKOUT: ScoringSystemType.BREAKOUT,
    }
    primary_system = system_map[system]

    _run_analysis(symbol, days, primary_system, use_context, output)


@app.command(name="enhanced")
def analyze_enhanced(
    symbol: str = typer.Argument(..., help="股票代码"),
    days: int = typer.Option(360, "--days", "-d", help="分析天数"),
    chip: bool = typer.Option(True, "--chip/--no-chip", help="是否包含筹码分布分析"),
    patterns: bool = typer.Option(True, "--patterns/--no-patterns", help="是否包含K线形态分析"),
    strategies: bool = typer.Option(True, "--strategies/--no-strategies", help="是否包含策略信号"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="输出文件")
):
    """
    增强版股票分析 - 整合筹码分布、K线形态、策略信号

    整合功能：
    - 筹码分布分析（集中度、获利盘、支撑阻力位）
    - K线形态识别（TA-Lib 61种形态）
    - 经典策略信号（10种策略共振分析）

    示例：
    - quanttool analysis enhanced 000001
    - quanttool analysis enhanced 000001 --days 250
    - quanttool analysis enhanced 000001 --no-chip  # 不包含筹码分析
    """
    typer.echo(f"正在执行增强分析：{symbol}")
    typer.echo(f"分析周期：{days} 天")
    typer.echo(f"筹码分布：{'是' if chip else '否'}")
    typer.echo(f"K线形态：{'是' if patterns else '否'}")
    typer.echo(f"策略信号：{'是' if strategies else '否'}")
    typer.echo("-" * 50)

    analyzer = StockAnalyzer()

    try:
        report = analyzer.analyze_stock_enhanced(
            symbol,
            days,
            include_chip=chip,
            include_talib_patterns=patterns,
            include_strategies=strategies
        )

        typer.echo(report)

        if output:
            with open(output, 'w', encoding='utf-8') as f:
                f.write(report)
            typer.echo(f"\n分析报告已保存至：{output}")

    except Exception as e:
        typer.echo(f"分析出错: {e}")


def _run_analysis(
    symbol: str,
    days: int,
    primary_system: ScoringSystemType,
    use_context: bool,
    output: Optional[str]
):
    """Internal function to run the analysis."""
    typer.echo(f"正在分析股票：{symbol}")
    typer.echo(f"分析周期：{days} 天")
    typer.echo(f"主评分系统：{primary_system.value}")
    typer.echo("-" * 50)

    # Create analyzer instance
    analyzer = StockAnalyzer()

    # Run analysis
    if use_context:
        # 使用统一分析上下文（推荐）
        context, report = analyzer.analyze_stock_with_context(symbol, days, primary_system)

        # 打印评分摘要
        typer.echo(f"\n=== 三系统评分摘要 ===")
        typer.echo(f"经典评分: {context.classic_score.score:.1f}分")
        if context.trend_score.passed_hard_filter:
            typer.echo(f"趋势评分: {context.trend_score.final_score:.1f}分 (时机: {context.trend_score.timing_type})")
        else:
            typer.echo(f"趋势评分: 未通过过滤 ({context.trend_score.hard_filter_reason})")
        if context.breakout_score.passed_filter:
            typer.echo(f"突破评分: {context.breakout_score.final_score:.1f}分")
        else:
            typer.echo(f"突破评分: 未通过筛选 ({context.breakout_score.filter_reason})")
        typer.echo(f"\n最终推荐: {context.final_recommendation.get_action_display()}")
        typer.echo("-" * 50)
    else:
        # 使用传统分析方法
        report = analyzer.analyze_stock(symbol, days)

    # Print report
    typer.echo(report)

    # Save to file if requested
    if output:
        with open(output, 'w', encoding='utf-8') as f:
            f.write(report)
        typer.echo(f"\n分析报告已保存至：{output}")


def get_csi300_constituents() -> List[Dict[str, str]]:
    """Get CSI 300 (沪深300) index constituents with names using DataFetcher."""
    try:
        fetcher = create_data_fetcher_with_credentials()
        fetcher.initialize()

        # Use include_names=True to get both code and name
        constituents = fetcher.get_csi300_constituents(include_names=True)

        if constituents:
            return constituents

        # If both failed, log warning and return empty list
        typer.echo("警告：无法从任何数据源获取沪深300成分股")
        return []

    except Exception as e:
        typer.echo(f"获取沪深300成分股失败: {e}")
        return []


def get_csi1000_constituents() -> List[Dict[str, str]]:
    """Get CSI 1000 (中证1000) index constituents with names using DataFetcher."""
    try:
        fetcher = create_data_fetcher_with_credentials()
        fetcher.initialize()

        # Use include_names=True to get both code and name
        constituents = fetcher.get_csi1000_constituents(include_names=True)

        if constituents:
            return constituents

        # If both failed, log warning and return empty list
        typer.echo("警告：无法从任何数据源获取中证1000成分股")
        return []

    except Exception as e:
        typer.echo(f"获取中证1000成分股失败: {e}")
        return []


def _get_reason_type(skip_reason: Optional[str]) -> str:
    """Extract reason type from skip reason for grouping.

    Examples:
        "数据获取失败" -> "数据获取失败"
        "新股/数据不足 (15条/需20条, 最早2025-01-15)" -> "新股/数据不足"
        "乖离率过滤 (BIAS6=5.5% > 5%)" -> "乖离率过滤"
        "评分错误: xxx" -> "评分错误"
        "分析异常: xxx" -> "分析异常"
    """
    if not skip_reason:
        return "未知原因"
    if skip_reason.startswith("数据获取失败"):
        return "数据获取失败"
    if skip_reason.startswith("新股/数据不足"):
        return "新股/数据不足"
    if skip_reason.startswith("乖离率过滤"):
        return "乖离率过滤"
    if skip_reason.startswith("评分错误"):
        return "评分错误"
    if skip_reason.startswith("分析异常"):
        return "分析异常"
    return skip_reason


def analyze_stock_trend_score(
        stock_info: Dict[str, str],
        days: int,
        analyzer: StockAnalyzer,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """使用趋势评分系统分析单只股票。

    Args:
        stock_info: 股票信息字典，包含 'code' 和 'name'
        days: 分析天数（当 start_date/end_date 未提供时使用）
        analyzer: StockAnalyzer 实例
        start_date: 可选的开始日期
        end_date: 可选的结束日期

    Returns:
        Tuple of (result_dict, skip_reason). 成功时 result_dict 包含分析数据，
        skip_reason 为 None。失败时 result_dict 为 None，skip_reason 包含跳过原因。
    """
    try:
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol

        # Get stock data with optional date range
        df = analyzer.get_stock_data(symbol, days, start_date, end_date)
        if df.empty:
            return None, "数据获取失败"
        if len(df) < 60:
            return None, f"数据不足60天 ({len(df)}条)"

        # 使用趋势评分系统
        trend_system = TrendScoringSystem()
        result = trend_system.calculate_score(df)

        if not result.passed_hard_filter:
            return None, f"趋势过滤: {result.hard_filter_reason}"

        # 获取最新价格信息
        latest = df.iloc[-1]
        close = latest.get('close', 0)

        return {
            "symbol": symbol,
            "name": name,
            "close": close,
            "score": result.final_score,
            "trend_score": result.trend_total_score,
            "timing_coefficient": result.timing_coefficient,
            "timing_type": result.timing_type,
            "ma_score": result.ma_structure_score,
            "momentum_score": result.price_momentum_score,
            "volume_score": result.volume_score,
            "rs_score": result.relative_strength_score,
            "score_grade": _get_trend_score_grade(result.final_score),
            "trigger_type": "trend",
            "trigger_detail": result.timing_type,
            "details": result.details,
        }, None

    except Exception as e:
        return None, f"分析异常: {str(e)}"


def _get_trend_score_grade(score: float) -> str:
    """获取趋势评分等级"""
    if score >= 90:
        return "极强"
    elif score >= 75:
        return "强势"
    elif score >= 60:
        return "一般"
    elif score >= 45:
        return "弱势"
    else:
        return "极弱"


def analyze_stock_breakout_score(
        stock_info: Dict[str, str],
        days: int,
        analyzer: StockAnalyzer,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """使用低位盘整突破评分系统分析单只股票。

    Args:
        stock_info: 股票信息字典，包含 'code' 和 'name'
        days: 分析天数（当 start_date/end_date 未提供时使用）
        analyzer: StockAnalyzer 实例
        start_date: 可选的开始日期
        end_date: 可选的结束日期

    Returns:
        Tuple of (result_dict, skip_reason). 成功时 result_dict 包含分析数据，
        skip_reason 为 None。失败时 result_dict 为 None，skip_reason 包含跳过原因。
    """
    try:
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol

        # Get stock data with optional date range
        df = analyzer.get_stock_data(symbol, days, start_date, end_date)
        if df.empty:
            return None, "数据获取失败"
        if len(df) < 60:
            return None, f"数据不足60天 ({len(df)}条)"

        # 使用低位盘整突破评分系统
        breakout_system = BreakoutScoringSystem()
        result = breakout_system.calculate_score(df)

        # 获取最新价格信息
        latest = df.iloc[-1]
        close = latest.get('close', 0)

        # 如果形态不满足，返回跳过原因
        if not result.passed_filter:
            return None, f"形态筛选: {result.filter_reason}"

        return {
            "symbol": symbol,
            "name": name,
            "close": close,
            "score": result.final_score,
            "is_low_position": result.is_low_position,
            "is_consolidating": result.is_consolidating,
            "has_breakout": result.has_breakout,
            "quality_score": result.quality_score,
            "growth_score": result.growth_score,
            "value_score": result.value_score,
            "momentum_score": result.momentum_score,
            "flow_score": result.flow_score,
            "risk_score": result.risk_score,
            "consolidation_days": result.consolidation_days,
            "price_range": result.price_range,
            "volume_ratio": result.volume_ratio,
            "breakout_strength": result.breakout_strength,
            "stop_loss": result.stop_loss_price,
            "take_profit": result.take_profit_price,
            "score_grade": _get_breakout_score_grade(result.final_score),
            "trigger_type": "breakout" if result.has_breakout else "consolidating",
            "trigger_detail": f"低位盘整{result.consolidation_days}天后{'突破' if result.has_breakout else '蓄势'}",
            "details": result.details,
        }, None

    except Exception as e:
        return None, f"分析异常: {str(e)}"


def _get_breakout_score_grade(score: float) -> str:
    """获取低位盘整突破评分等级"""
    if score >= 80:
        return "优秀"
    elif score >= 70:
        return "良好"
    elif score >= 60:
        return "一般"
    elif score >= 50:
        return "较弱"
    else:
        return "较差"


def analyze_stock_momentum_score(
        stock_info: Dict[str, str],
        days: int,
        analyzer: StockAnalyzer,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """使用趋势动量评分系统分析单只股票。

    Args:
        stock_info: 股票信息字典，包含 'code' 和 'name'
        days: 分析天数（当 start_date/end_date 未提供时使用）
        analyzer: StockAnalyzer 实例
        start_date: 可选的开始日期
        end_date: 可选的结束日期

    Returns:
        Tuple of (result_dict, skip_reason). 成功时 result_dict 包含分析数据，
        skip_reason 为 None。失败时 result_dict 为 None，skip_reason 包含跳过原因。
    """
    try:
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol

        # Get stock data with optional date range
        df = analyzer.get_stock_data(symbol, days, start_date, end_date)
        if df.empty:
            return None, "数据获取失败"
        if len(df) < 60:
            return None, f"数据不足60天 ({len(df)}条)"

        # 使用趋势动量评分系统
        momentum_system = TrendMomentumScoring()
        result = momentum_system.calculate_score(df)

        # 获取最新价格信息
        latest = df.iloc[-1]
        close = latest.get('close', 0)

        return {
            "symbol": symbol,
            "name": name,
            "close": close,
            "score": result.final_score,
            "signal": result.signal,
            "momentum_score": result.momentum_score,
            "ma_score": result.ma_score,
            "volume_score": result.volume_score,
            "position_score": result.position_score,
            "breakout_score": result.breakout_score,
            "stop_loss": result.stop_loss,
            "take_profit": result.take_profit,
            "signals": result.signals,
            "score_grade": _get_momentum_score_grade(result.final_score),
            "trigger_type": "momentum",
            "trigger_detail": "趋势动量" if result.signal else "观望",
            "details": result.details,
        }, None

    except Exception as e:
        return None, f"分析异常: {str(e)}"


def _get_momentum_score_grade(score: float) -> str:
    """获取趋势动量评分等级"""
    if score >= 80:
        return "极强"
    elif score >= 65:
        return "强势"
    elif score >= 55:
        return "一般"
    elif score >= 40:
        return "弱势"
    else:
        return "极弱"


def analyze_stock_score(
        stock_info: Dict[str, str],
        days: int,
        analyzer: StockAnalyzer,
        bias_min: Optional[float] = None,
        bias_max: Optional[float] = None,
        use_strategy: bool = True,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Analyze a single stock and return its score data with BIAS filtering.

    Args:
        stock_info: Stock information dict with 'code' and 'name'
        days: Number of days to analyze (used when start_date/end_date not provided)
        analyzer: StockAnalyzer instance
        bias_min: Minimum BIAS(6) filter
        bias_max: Maximum BIAS(6) filter
        use_strategy: Whether to use strategy layer for signal generation
        start_date: Optional start date for data range
        end_date: Optional end date for data range

    Returns:
        Tuple of (result_dict, skip_reason). If analysis succeeds, result_dict contains
        the analysis data and skip_reason is None. If analysis fails, result_dict is None
        and skip_reason contains the reason for skipping.
    """
    try:
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol

        # Get stock data with optional date range
        df = analyzer.get_stock_data(symbol, days, start_date, end_date)
        if df.empty:
            return None, "数据获取失败"
        if len(df) < 20:
            # Check if it's a new stock by looking at the date range
            if 'timestamp' in df.columns and len(df) > 0:
                earliest_date = pd.to_datetime(df['timestamp'].min())
                trading_days = len(df)
                days_since_listing = (datetime.now() - earliest_date).days
                return None, f"新股/数据不足 ({trading_days}条/需20条, 最早{earliest_date.strftime('%Y-%m-%d')})"
            else:
                return None, f"数据条数不足 ({len(df)}条/需20条)"

        # Calculate technical indicators
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # Get latest data for BIAS check
        latest = df_with_indicators.iloc[-1]

        # BIAS filtering (乖离率过滤)
        bias_6 = latest.get('bias_6', 0)
        bias_12 = latest.get('bias_12', 0)
        bias_24 = latest.get('bias_24', 0)

        # Calculate BIAS(20) for display
        close = latest.get('close', 0)
        ma20 = latest.get('ma_20', 0)
        bias_20 = (close / ma20 - 1) * 100 if ma20 > 0 else 0

        # Apply BIAS filter if specified
        if bias_min is not None and bias_6 < bias_min:
            return None, f"乖离率过滤 (BIAS6={bias_6:.2f}% < {bias_min}%)"
        if bias_max is not None and bias_6 > bias_max:
            return None, f"乖离率过滤 (BIAS6={bias_6:.2f}% > {bias_max}%)"

        # Run scoring system
        scoring = ScoringSystem()
        score_result = scoring.calculate_all_scores(df_with_indicators, stock_code=symbol)

        if "error" in score_result:
            return None, f"评分错误: {score_result.get('error', '未知错误')}"

        # 新增：策略层信号生成
        strategy_signal = None
        adaptive_thresholds = None
        dual_market_state = None

        if use_strategy:
            try:
                # 获取双重市场状态（大盘 + 个股）
                market_detector = IndexMarketDetector(default_index='hs300')
                dual_state = market_detector.get_dual_market_state(df_with_indicators)
                dual_market_state = dual_state.to_dict()

                # 获取自适应阈值
                threshold_manager = AdaptiveThresholdManager()
                adaptive_config = threshold_manager.get_adaptive_thresholds(df_with_indicators)

                # 根据综合信号调整阈值
                # 优化后的默认阈值：买入50，卖出25
                base_buy = 50.0
                base_sell = 25.0

                # 根据综合信号微调阈值
                combined_signal = dual_state.combined_signal
                if combined_signal == CombinedSignal.STRONG_BUY:
                    # 强买入信号，降低买入门槛
                    base_buy = 45.0
                elif combined_signal == CombinedSignal.CASH:
                    # 空仓信号，大幅提高买入门槛
                    base_buy = 80.0
                elif combined_signal == CombinedSignal.AVOID:
                    # 回避信号，提高买入门槛
                    base_buy = 70.0
                elif combined_signal == CombinedSignal.LIGHT_POSITION:
                    # 轻仓信号，略微提高买入门槛
                    base_buy = 55.0

                adaptive_thresholds = {
                    'buy_threshold': adaptive_config.buy_threshold,
                    'sell_threshold': adaptive_config.sell_threshold,
                    'regime': adaptive_config.market_regime.value,
                    'volatility': adaptive_config.volatility_level.value,
                    'adjusted_buy_threshold': base_buy,
                    'adjusted_sell_threshold': base_sell,
                }

                # 创建策略实例（使用优化后的阈值）
                strategy = ScoreStrategy(
                    buy_threshold=base_buy,
                    sell_threshold=base_sell,
                    use_dynamic_weights=True,
                    use_multi_timeframe=True,
                    use_risk_control=True
                )

                # 获取信号
                signal = strategy.get_signal(latest, df_with_indicators)
                strategy_signal = {
                    'direction': signal.get('direction'),
                    'signal': signal.get('signal'),
                    'adjusted_score': signal.get('adjusted_score'),
                    'mtf_bonus': signal.get('mtf_bonus'),
                    'stop_loss': signal.get('stop_loss'),
                }
            except Exception as e:
                # 策略层失败不影响基础评分
                strategy_signal = {'error': str(e)}

        return {
            "symbol": symbol,
            "name": name,
            "close": latest['close'],
            "daily_return": latest.get('daily_return', 0),
            "score": score_result['score'],
            "score_grade": score_result['score_grade'],
            "trigger_type": score_result['trigger_type'],
            "trigger_detail": score_result['trigger_detail'],
            "factors_raw": score_result['factors_raw'],
            "factors_score": score_result['factors_score'],
            "execution": score_result['execution'],
            "warnings": score_result['warnings'],
            # 因子组得分 - 从 score_result 顶层提取
            "trend_score": score_result.get('trend_score', 50),
            "momentum_score": score_result.get('momentum_score', 50),
            "money_score": score_result.get('money_score', 50),
            "position_modifier": score_result.get('position_modifier', 1.0),
            # BIAS data
            "bias_6": bias_6,
            "bias_12": bias_12,
            "bias_24": bias_24,
            "bias_20": bias_20,
            # 新增：策略信号
            "strategy_signal": strategy_signal,
            "adaptive_thresholds": adaptive_thresholds,
            "dual_market_state": dual_market_state,
        }, None
    except Exception as e:
        return None, f"分析异常: {str(e)}"


@app.command()
def scan(
    market: str = typer.Option("csi300", "--market", "-m", help="Market to scan: csi300, csi1000, sh, sz, or all"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top stocks to return (default: 10)"),
    # 评分系统选择
    use_trend_score: bool = typer.Option(True, "--trend/--classic", help="Use trend scoring system (default) or classic scoring system"),
    use_breakout_score: bool = typer.Option(False, "--breakout", help="Use breakout scoring system (low position + consolidation + breakout)"),
    use_momentum_score: bool = typer.Option(False, "--momentum", help="Use trend momentum scoring system (momentum + MA + volume)"),
    # BIAS filter options (乖离率过滤)
    bias_min: Optional[float] = typer.Option(None, "--bias-min", help="Minimum BIAS(6) value to include stock (e.g., -5.0) - Note: Hard filter uses BIAS(20) > +8%"),
    bias_max: Optional[float] = typer.Option(None, "--bias-max", help="Maximum BIAS(6) value to include stock (e.g., 5.0) - Note: Hard filter uses BIAS(20) > +8%"),
    # Record saving options
    save_record: bool = typer.Option(True, "--save/--no-save", help="Save scan record to database"),
    db_path: str = typer.Option("./quanttool.db", "--db-path", help="Path to the database file"),
    # Report generation options
    generate_reports: bool = typer.Option(True, "--reports/--no-reports", help="Generate detailed markdown reports for top N stocks"),
    output_dir: str = typer.Option("./reports", "--output-dir", "-o", help="Directory to save detailed reports"),
):
    """Scan the market for potential opportunities based on technical indicators.

    评分系统说明：
    - 趋势评分系统（默认）：纯趋势强度评分，强势股得高分，不会被过度惩罚
    - 经典评分系统：包含位置惩罚系数，超买股会被大幅扣分
    - 低位盘整突破评分系统：寻找低位盘整后放量突破的股票
    - 趋势动量评分系统：抓住趋势启动点，动量+均线+量能综合评分

    Examples:
        # 使用趋势评分系统扫描沪深300（默认）
        quanttool analysis scan

        # 使用经典评分系统扫描
        quanttool analysis scan --classic

        # 使用低位盘整突破评分系统扫描
        quanttool analysis scan --breakout

        # 使用趋势动量评分系统扫描
        quanttool analysis scan --momentum

        # Scan with BIAS filter (乖离率过滤)
        quanttool analysis scan --bias-min -5.0 --bias-max 5.0

        # Scan without saving record and reports
        quanttool analysis scan --no-save --no-reports

        # Scan with custom output directory
        quanttool analysis scan --output-dir ./my_reports
    """

    # Get stock list based on market parameter
    if market.lower() == "csi300":
        typer.echo("正在获取沪深300成分股...")
        stock_list = get_csi300_constituents()
    elif market.lower() == "csi1000":
        typer.echo("正在获取中证1000成分股...")
        stock_list = get_csi1000_constituents()
    else:
        typer.echo(f"不支持的扫描市场: {market}")
        typer.echo("当前支持的市场: csi300 (沪深300), csi1000 (中证1000)")
        return

    if not stock_list:
        typer.echo("无法获取股票列表，请检查数据接口配置。")
        return

    typer.echo(f"共获取 {len(stock_list)} 只股票，开始分析...")
    typer.echo(f"分析周期：{days} 天")

    # Display BIAS filter info
    if bias_min is not None or bias_max is not None:
        typer.echo(f"乖离率过滤：BIAS(6) ", nl=False)
        if bias_min is not None:
            typer.echo(f">= {bias_min}% ", nl=False)
        if bias_max is not None:
            typer.echo(f"<= {bias_max}%", nl=False)
        typer.echo()

    # 显示评分系统选择
    if use_momentum_score:
        typer.echo("📊 评分系统：趋势动量评分系统（动量+均线+量能）")
    elif use_breakout_score:
        typer.echo("📊 评分系统：低位盘整突破评分系统（低位+盘整+突破）")
    elif use_trend_score:
        typer.echo("📊 评分系统：趋势评分系统（纯趋势强度评分）")
    else:
        typer.echo("📊 评分系统：经典评分系统（含位置惩罚系数）")
        typer.echo("注意：系统已启用 BIAS(20) > +8% 硬过滤（自动剔除追高风险股票）")
    typer.echo("-" * 60)

    # Create analyzer instance (uses local file cache, not memory cache)
    analyzer = StockAnalyzer()

    # Calculate unified time range ONCE at the start
    # This ensures cache keys are consistent between warmup and analysis
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    # Warm up caches (both local file cache and in-memory cache)
    # This saves to disk for persistence AND to memory for fast access
    typer.echo(f"📦 正在预热缓存...")
    import time
    start_time = time.time()

    symbols = [stock['code'] if isinstance(stock, dict) else stock for stock in stock_list]

    # Fetch in batches to avoid overwhelming the system
    batch_size = 30
    cached_count = 0
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        typer.echo(f"  获取第 {batch_num}/{total_batches} 批 ({len(batch)} 只)...")

        # Use get_bars_cached - checks local cache first, fetches missing data in parallel
        data = analyzer.fetcher.get_bars_cached(batch, start_date, end_date)
        cached_count += len(data)

        # Store in memory cache for fast access during analysis
        for symbol, df in data.items():
            if not df.empty:
                analyzer._batch_data_cache[symbol] = df

    elapsed = time.time() - start_time
    typer.echo(f"✅ 缓存预热完成: {cached_count}/{len(stock_list)} 只股票 (耗时 {elapsed:.1f} 秒)")
    typer.echo("-" * 60)

    # Analyze stocks in parallel
    results = []
    skipped_stocks = []
    total = len(stock_list)
    completed_count = 0
    progress_lock = threading.Lock()

    def analyze_single_stock(stock_info: Dict) -> Tuple[Optional[Dict], Optional[Dict]]:
        """分析单只股票（用于并行执行）"""
        nonlocal completed_count
        symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info
        name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol

        if use_momentum_score:
            result, skip_reason = analyze_stock_momentum_score(
                stock_info, days, analyzer, start_date, end_date
            )
        elif use_breakout_score:
            result, skip_reason = analyze_stock_breakout_score(
                stock_info, days, analyzer, start_date, end_date
            )
        elif use_trend_score:
            result, skip_reason = analyze_stock_trend_score(
                stock_info, days, analyzer, start_date, end_date
            )
        else:
            result, skip_reason = analyze_stock_score(
                stock_info, days, analyzer, bias_min, bias_max, True, start_date, end_date
            )

        with progress_lock:
            completed_count += 1
            if result:
                if use_momentum_score:
                    signal_str = "✓信号" if result.get('signal') else "观望"
                    typer.echo(f"[{completed_count}/{total}] {symbol} 评分: {result['score']:.1f} | 动量: {result['momentum_score']:.0f} | {signal_str}")
                elif use_breakout_score:
                    typer.echo(f"[{completed_count}/{total}] {symbol} 评分: {result['score']:.1f} | 盘整: {result['consolidation_days']}天")
                elif use_trend_score:
                    timing_type = result.get('timing_type', '标准')
                    typer.echo(f"[{completed_count}/{total}] {symbol} 评分: {result['score']:.1f} | 时机: {timing_type}")
                else:
                    typer.echo(f"[{completed_count}/{total}] {symbol} 评分: {result['score']:.1f}")
            else:
                typer.echo(f"[{completed_count}/{total}] {symbol} 跳过 ({skip_reason})")

        if result:
            return result, None
        else:
            return None, {
                'symbol': symbol,
                'name': name,
                'reason': skip_reason or "未知原因",
                'reason_type': _get_reason_type(skip_reason)
            }

    # 并行分析
    analyze_start = time.time()
    max_workers = min(10, len(stock_list))

    typer.echo(f"🚀 开始并行分析 ({max_workers} 线程)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(analyze_single_stock, stock): stock for stock in stock_list}

        for future in as_completed(futures):
            try:
                result, skip_info = future.result()
                if result:
                    results.append(result)
                elif skip_info:
                    skipped_stocks.append(skip_info)
            except Exception as e:
                stock = futures[future]
                symbol = stock['code'] if isinstance(stock, dict) else stock
                typer.echo(f"⚠️ {symbol} 分析异常: {e}")

    analyze_elapsed = time.time() - analyze_start
    typer.echo(f"✅ 分析完成: {len(results)} 只股票成功, {len(skipped_stocks)} 只跳过 (耗时 {analyze_elapsed:.1f} 秒)")

    if not results:
        typer.echo("\n没有成功分析任何股票，请检查数据接口配置。")
        return

    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Display filter summary
    if not use_trend_score and (bias_min is not None or bias_max is not None):
        typer.echo(f"\n📊 乖离率过滤：{filtered_count} 只股票被过滤")

    # Display top N results
    typer.echo("\n" + "=" * 110)
    if use_momentum_score:
        typer.echo(f"📊 沪深300成分股趋势动量评分排名 - Top {top_n}")
    elif use_breakout_score:
        typer.echo(f"📊 沪深300成分股低位盘整突破评分排名 - Top {top_n}")
    elif use_trend_score:
        typer.echo(f"📊 沪深300成分股趋势评分排名 - Top {top_n}")
    else:
        typer.echo(f"📊 沪深300成分股评分排名 - Top {top_n}")
    typer.echo("=" * 110)

    if use_momentum_score:
        # 趋势动量评分表格格式
        typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'评分':<10} {'动量':<8} {'均线':<8} {'量能':<8} {'等级':<8}")
        typer.echo("-" * 110)

        for i, r in enumerate(results[:top_n], 1):
            symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
            name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
            typer.echo(f"{i:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {r['score']:<10.1f} {r['momentum_score']:<8.0f} {r['ma_score']:<8.0f} {r['volume_score']:<8.0f} {r['score_grade']:<8}")
    elif use_breakout_score:
        # 低位盘整突破评分表格格式
        typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'评分':<10} {'盘整天数':<10} {'量比':<8} {'等级':<8}")
        typer.echo("-" * 110)

        for i, r in enumerate(results[:top_n], 1):
            symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
            name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
            typer.echo(f"{i:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {r['score']:<10.1f} {r['consolidation_days']:<10} {r['volume_ratio']:<8.1f} {r['score_grade']:<8}")
    elif use_trend_score:
        # 趋势评分表格格式
        typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'评分':<10} {'时机系数':<10} {'时机类型':<12} {'等级':<8}")
        typer.echo("-" * 110)

        for i, r in enumerate(results[:top_n], 1):
            symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
            name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
            typer.echo(f"{i:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {r['score']:<10.1f} {r['timing_coefficient']:<10.2f} {r['timing_type']:<12} {r['score_grade']:<8}")
    else:
        # 经典评分表格格式
        typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'评分':<10} {'BIAS(20)':<10} {'等级':<8} {'触发类型':<12}")
        typer.echo("-" * 110)

        for i, r in enumerate(results[:top_n], 1):
            symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
            name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
            bias_str = f"{r.get('bias_20', 0):+.2f}%"
            typer.echo(f"{i:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {r['score']:<10.1f} {bias_str:<10} {r['score_grade']:<8} {r['trigger_type']:<12}")

    # Display detailed breakdown for top N
    typer.echo("\n" + "=" * 80)
    typer.echo(f"📈 Top {top_n} 详细评分维度")
    typer.echo("=" * 80)

    for i, r in enumerate(results[:top_n], 1):
        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')

        if use_momentum_score:
            # 趋势动量评分详细显示格式
            signal_str = "✓买入信号" if r.get('signal') else "观望"
            typer.echo(f"\n【#{i}】{symbol_short} {r['name']} - 评分: {r['score']:.1f}/100 | 等级: {r['score_grade']} | {signal_str}")
            typer.echo(f"    收盘价: ¥{r['close']:.2f}")
            typer.echo(f"    评分: 动量={r['momentum_score']:.0f} | 均线={r['ma_score']:.0f} | 量能={r['volume_score']:.0f} | 位置={r['position_score']:.0f} | 突破={r['breakout_score']:.0f}")
            typer.echo(f"    交易: 止损=¥{r['stop_loss']:.2f} | 止盈=¥{r['take_profit']:.2f}")
            if r.get('signals'):
                typer.echo(f"    信号: {', '.join(r['signals'][:5])}")
        elif use_breakout_score:
            # 低位盘整突破评分详细显示格式
            typer.echo(f"\n【#{i}】{symbol_short} {r['name']} - 评分: {r['score']:.1f}/100 | 等级: {r['score_grade']}")
            typer.echo(f"    收盘价: ¥{r['close']:.2f}")
            typer.echo(f"    形态: 低位={r['is_low_position']} | 盘整={r['is_consolidating']} | 突破={r['has_breakout']}")
            typer.echo(f"    盘整: {r['consolidation_days']}天 | 振幅: {r['price_range']*100:.1f}% | 量比: {r['volume_ratio']:.1f}")
            typer.echo(f"    因子: 质量={r['quality_score']:.0f} | 成长={r['growth_score']:.0f} | 估值={r['value_score']:.0f}")
            typer.echo(f"          动量={r['momentum_score']:.0f} | 资金={r['flow_score']:.0f} | 风险={r['risk_score']:.0f}")
            typer.echo(f"    交易: 止损=¥{r['stop_loss']:.2f} | 止盈=¥{r['take_profit']:.2f}")
        elif use_trend_score:
            # 趋势评分详细显示格式
            typer.echo(f"\n【#{i}】{symbol_short} {r['name']} - 评分: {r['score']:.1f}/100 | 等级: {r['score_grade']} | 时机: {r['timing_type']} ({r['timing_coefficient']:.2f})")
            typer.echo(f"    收盘价: ¥{r['close']:.2f}")
            typer.echo(f"    趋势评分: {r.get('trend_score', 0):.1f} | MA结构: {r.get('ma_score', 0):.1f} | 动量: {r.get('momentum_score', 0):.1f}")
            typer.echo(f"    成交量: {r.get('volume_score', 0):.1f} | 相对强度: {r.get('rs_score', 0):.1f}")
        else:
            # 经典评分详细显示格式
            typer.echo(f"\n【#{i}】{symbol_short} {r['name']} - 评分: {r['score']:.1f}/100 | 等级: {r['score_grade']} | 触发: {r['trigger_type']}")
            typer.echo(f"    收盘价: ¥{r['close']:.2f}")
            typer.echo(f"    乖离率: BIAS(20)={r.get('bias_20', 0):+.2f}% | BIAS(6)={r.get('bias_6', 0):+.2f}% | BIAS(12)={r.get('bias_12', 0):+.2f}%")

        # 新增：显示策略信号
        strategy_signal = r.get('strategy_signal')
        if strategy_signal and 'error' not in strategy_signal:
            direction = strategy_signal.get('direction', 'hold')
            adjusted_score = strategy_signal.get('adjusted_score', 0)
            mtf_bonus = strategy_signal.get('mtf_bonus', 0)
            stop_loss = strategy_signal.get('stop_loss')

            # 信号方向emoji
            signal_emoji = {'buy': '✅', 'sell': '❌', 'hold': '➖'}.get(direction, '➖')
            signal_cn = {'buy': '买入', 'sell': '卖出', 'hold': '观望'}.get(direction, '观望')

            typer.echo(f"    策略信号: {signal_cn} {signal_emoji}")
            if adjusted_score:
                typer.echo(f"      • 调整后评分: {adjusted_score:.1f}")
            if mtf_bonus:
                typer.echo(f"      • 多周期确认加成: {mtf_bonus*100:.1f}%")
            if stop_loss:
                typer.echo(f"      • 建议止损位: ¥{stop_loss:.2f}")

        # 新增：显示双重市场状态
        dual_market_state = r.get('dual_market_state')
        if dual_market_state:
            index_regime = dual_market_state.get('index_regime', 'sideway')
            stock_regime = dual_market_state.get('stock_regime', 'sideway')
            combined_signal = dual_market_state.get('combined_signal', '观望')
            index_name = dual_market_state.get('index_name', '沪深300')
            confidence = dual_market_state.get('confidence', 0.5)

            regime_cn = {'bull': '牛市📈', 'bear': '熊市📉', 'sideway': '震荡↔️', 'volatile': '剧烈波动⚡'}
            index_regime_cn = regime_cn.get(index_regime, index_regime)
            stock_regime_cn = regime_cn.get(stock_regime, stock_regime)

            # 综合信号emoji
            signal_emoji_map = {
                '强买入': '🚀', '关注': '👀', '回避': '⚠️',
                '轻仓': '💰', '观望': '➖', '空仓': '🛑'
            }
            signal_emoji = signal_emoji_map.get(combined_signal, '➖')

            typer.echo(f"    双重市场状态: {signal_emoji} {combined_signal}")
            typer.echo(f"      • 大盘({index_name}): {index_regime_cn} | 个股: {stock_regime_cn}")
            typer.echo(f"      • 置信度: {confidence*100:.0f}%")

        # 新增：显示自适应阈值
        adaptive_thresholds = r.get('adaptive_thresholds')
        if adaptive_thresholds:
            regime = adaptive_thresholds.get('regime', 'sideway')
            volatility = adaptive_thresholds.get('volatility', 'normal')
            buy_th = adaptive_thresholds.get('adjusted_buy_threshold', adaptive_thresholds.get('buy_threshold', 50))
            sell_th = adaptive_thresholds.get('adjusted_sell_threshold', adaptive_thresholds.get('sell_threshold', 25))

            regime_cn = {'bull': '牛市📈', 'bear': '熊市📉', 'sideway': '震荡↔️', 'volatile': '剧烈波动⚡'}.get(regime, regime)
            vol_cn = {'low': '低波动', 'normal': '正常', 'high': '高波动', 'extreme': '极端波动'}.get(volatility, volatility)

            typer.echo(f"    自适应阈值: 买入 {buy_th:.1f} / 卖出 {sell_th:.1f}")
            typer.echo(f"    个股状态: {regime_cn} | 波动率: {vol_cn}")

        # 显示触发详情（仅经典评分）
        if not use_trend_score and r.get('trigger_detail'):
            typer.echo(f"    触发详情: {r['trigger_detail']}")

        # 显示因子得分
        factors_score = r.get('factors_score', {})
        if factors_score:
            typer.echo(f"    因子得分:")
            factors_raw = r.get('factors_raw', {})

            # 解析嵌套结构
            trend_factors = factors_raw.get('trend_factors', {})
            momentum_factors = factors_raw.get('momentum_factors', {})
            money_factors = factors_raw.get('money_factors', {})
            aux_factors = factors_raw.get('aux_factors', {})

            # 定义因子到嵌套位置的映射
            factor_location = {
                'trend_strength': ('aux_factors', 'bias20', 'percent'),  # 趋势强度用 bias20
                'ma_slope': ('trend_factors', 'ma_alignment', 'score'),
                'macd_momentum': ('trend_factors', 'macd_momentum', 'float'),
                'money_flow': ('money_factors', 'obv_flow', 'score'),
                'volume_ratio': ('money_factors', 'volume_ratio', 'ratio'),
            }

            for factor_name, score_val in factors_score.items():
                if factor_name in factor_location:
                    group_key, raw_key, fmt = factor_location[factor_name]
                    if group_key == 'trend_factors':
                        raw_val = trend_factors.get(raw_key, 'N/A')
                    elif group_key == 'momentum_factors':
                        raw_val = momentum_factors.get(raw_key, 'N/A')
                    elif group_key == 'money_factors':
                        raw_val = money_factors.get(raw_key, 'N/A')
                    elif group_key == 'aux_factors':
                        raw_val = aux_factors.get(raw_key, 'N/A')
                    else:
                        raw_val = 'N/A'

                    # 格式化原始值
                    if raw_val == 'N/A':
                        raw_str = 'N/A'
                    elif fmt == 'percent':
                        raw_str = f"{raw_val*100:.2f}%"
                    elif fmt == 'score':
                        raw_str = f"{raw_val:.0f}分"
                    elif fmt == 'ratio':
                        raw_str = f"{raw_val:.2f}"
                    else:
                        raw_str = f"{raw_val:.4f}"
                else:
                    raw_str = 'N/A'

                typer.echo(f"      • {factor_name}: {score_val:.2f} (原始值: {raw_str})")

        # 显示交易执行计划
        execution = r.get('execution', {})
        if execution:
            typer.echo(f"    交易计划:")
            if 'position_suggest' in execution:
                typer.echo(f"      • 建议仓位: {execution['position_suggest']}")
            if 'buy_price' in execution:
                typer.echo(f"      • 买入价位: ¥{execution['buy_price']:.2f}")
            if 'stop_price' in execution:
                stop_desc = execution.get('stop_loss_desc', f"{execution.get('stop_loss_pct', 0.05)*100:.0f}%")
                typer.echo(f"      • 止损价位: ¥{execution['stop_price']:.2f} ({stop_desc})")
            if 'action_guide' in execution:
                typer.echo(f"      • 操作指引: {execution['action_guide']}")

        if r.get('warnings'):
            typer.echo(f"    ⚠️ 风险提示:")
            for warning in r['warnings']:
                typer.echo(f"      - {warning}")

    typer.echo("\n" + "=" * 80)
    typer.echo("📋 评分说明:")
    typer.echo("  • 评分范围: 0-100 分")
    typer.echo("  • >= 85 分: 优秀，强烈看多")
    typer.echo("  • 70-84 分: 良好，偏多")
    typer.echo("  • 50-69 分: 一般，观望")
    typer.echo("  • 35-49 分: 较差，偏空")
    typer.echo("  • < 35 分: 差，看空")
    typer.echo("=" * 80)

    # Display skipped stocks report
    if skipped_stocks:
        typer.echo("\n" + "=" * 80)
        typer.echo(f"⚠️  未纳入计算的股票报告 (共 {len(skipped_stocks)} 只)")
        typer.echo("=" * 80)

        # Group skipped stocks by reason_type
        from collections import defaultdict
        skipped_by_type = defaultdict(list)
        for stock in skipped_stocks:
            skipped_by_type[stock['reason_type']].append(stock)

        # Display each group
        for reason_type, stocks in sorted(skipped_by_type.items(), key=lambda x: -len(x[1])):
            # Show an example reason for this type
            example_reason = stocks[0]['reason']
            typer.echo(f"\n【{reason_type}】({len(stocks)} 只)")
            if len(stocks) <= 3:
                # For small groups, show detailed reasons for each
                for s in stocks:
                    symbol_short = s['symbol'].replace('.SH', '').replace('.SZ', '')
                    typer.echo(f"  {symbol_short}({s['name']}) - {s['reason']}")
            else:
                # For larger groups, show example and list stocks
                if '(' in example_reason and ')' in example_reason:
                    typer.echo(f"  示例: {example_reason}")
                # Show stocks in rows of 5
                stock_list_str = []
                for s in stocks:
                    symbol_short = s['symbol'].replace('.SH', '').replace('.SZ', '')
                    stock_list_str.append(f"{symbol_short}({s['name']})")

                # Print in chunks of 5
                for i in range(0, len(stock_list_str), 5):
                    chunk = stock_list_str[i:i+5]
                    typer.echo("  " + ", ".join(chunk))

        typer.echo("\n" + "=" * 80)
        typer.echo(f"统计: 共扫描 {total} 只股票，成功分析 {len(results)} 只，跳过 {len(skipped_stocks)} 只")
        typer.echo("=" * 80)

    # Save scan record to database
    if save_record:
        try:
            meta_db = MetaDB(db_path)
            scan_date = datetime.now().isoformat()

            scan_data = {
                "scan_date": scan_date,
                "market": market,
                "days_analyzed": days,
                "total_stocks": len(stock_list),
                "bias_filter_min": bias_min,
                "bias_filter_max": bias_max,
                "results": results[:top_n],  # Only save top N results
            }

            scan_id = meta_db.save_scan_record(scan_data)
            typer.echo(f"\n💾 扫描记录已保存 (ID: {scan_id})")
            typer.echo(f"   数据库: {db_path}")
        except Exception as e:
            typer.echo(f"\n⚠️ 保存扫描记录失败: {e}")

    # Generate detailed reports for top N stocks
    if generate_reports and results:
        try:
            import os

            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            scan_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(output_dir, f"scan_top{top_n}_{scan_timestamp}.md")

            typer.echo(f"\n📝 正在为 Top {top_n} 股票生成详细分析报告...")

            with open(report_file, 'w', encoding='utf-8') as f:
                # Write report header
                f.write(f"# 股票扫描分析报告\n\n")
                f.write(f"**扫描时间：** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"**扫描市场：** {market.upper()}\n\n")
                f.write(f"**分析周期：** {days} 天\n\n")
                f.write(f"**股票总数：** {len(stock_list)} 只\n\n")
                f.write(f"**成功分析：** {len(results)} 只\n\n")
                if bias_min is not None or bias_max is not None:
                    f.write(f"**乖离率过滤：** BIAS(6) ")
                    if bias_min is not None:
                        f.write(f">= {bias_min}% ")
                    if bias_max is not None:
                        f.write(f"<= {bias_max}%")
                    f.write("\n\n")
                f.write("---\n\n")

                # Write summary table - 增强版：包含因子组得分明细
                f.write("## 📊 Top {} 股票概览\n\n".format(top_n))

                # 根据评分系统类型选择不同的表头
                if use_trend_score:
                    # 趋势评分系统
                    f.write("| 排名 | 代码 | 名称 | 收盘价 | 评分 | 等级 | 均线分 | 动能分 | 量能分 | 强度分 | 时机系数 | 时机类型 |\n")
                    f.write("|------|------|------|--------|------|------|--------|--------|--------|--------|----------|----------|\n")

                    for i, r in enumerate(results[:top_n], 1):
                        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
                        name = r.get('name', symbol_short)
                        f.write(f"| {i} | {symbol_short} | {name} | ¥{r['close']:.2f} | {r['score']:.1f} | {r['score_grade']} | {r.get('ma_score', 0):.0f} | {r.get('momentum_score', 0):.0f} | {r.get('volume_score', 0):.0f} | {r.get('rs_score', 0):.0f} | {r.get('timing_coefficient', 1.0):.2f} | {r.get('timing_type', '-')} |\n")

                elif use_breakout_score:
                    # 低位盘整突破评分系统
                    f.write("| 排名 | 代码 | 名称 | 收盘价 | 评分 | 等级 | 质量分 | 成长分 | 价值分 | 动量分 | 资金分 | 盘整天数 | 量比 |\n")
                    f.write("|------|------|------|--------|------|------|--------|--------|--------|--------|--------|----------|------|\n")

                    for i, r in enumerate(results[:top_n], 1):
                        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
                        name = r.get('name', symbol_short)
                        f.write(f"| {i} | {symbol_short} | {name} | ¥{r['close']:.2f} | {r['score']:.1f} | {r['score_grade']} | {r.get('quality_score', 0):.0f} | {r.get('growth_score', 0):.0f} | {r.get('value_score', 0):.0f} | {r.get('momentum_score', 0):.0f} | {r.get('flow_score', 0):.0f} | {r.get('consolidation_days', 0)} | {r.get('volume_ratio', 0):.1f} |\n")

                elif use_momentum_score:
                    # 趋势动量评分系统
                    f.write("| 排名 | 代码 | 名称 | 收盘价 | 评分 | 等级 | 动量分 | 均线分 | 量能分 | 位置分 | 突破分 | 信号 |\n")
                    f.write("|------|------|------|--------|------|------|--------|--------|--------|--------|--------|------|\n")

                    for i, r in enumerate(results[:top_n], 1):
                        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
                        name = r.get('name', symbol_short)
                        signal_str = "✓买入" if r.get('signal') else "观望"
                        f.write(f"| {i} | {symbol_short} | {name} | ¥{r['close']:.2f} | {r['score']:.1f} | {r['score_grade']} | {r.get('momentum_score', 0):.0f} | {r.get('ma_score', 0):.0f} | {r.get('volume_score', 0):.0f} | {r.get('position_score', 0):.0f} | {r.get('breakout_score', 0):.0f} | {signal_str} |\n")

                else:
                    # 经典评分系统 - 包含因子组得分明细和技术指标
                    f.write("| 排名 | 代码 | 名称 | 收盘价 | 评分 | 等级 | 趋势分 | 动能分 | 资金分 | 位置系数 | BIAS(20) | RSI | MACD |\n")
                    f.write("|------|------|------|--------|------|------|--------|--------|--------|----------|----------|-----|------|\n")

                    for i, r in enumerate(results[:top_n], 1):
                        symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
                        name = r.get('name', symbol_short)
                        bias_str = f"{r.get('bias_20', 0):+.2f}%"

                        # 提取技术指标原始值
                        factors_raw = r.get('factors_raw', {})
                        momentum_factors = factors_raw.get('momentum_factors', {})
                        trend_factors = factors_raw.get('trend_factors', {})

                        rsi_val = momentum_factors.get('rsi', 50)
                        macd_hist = trend_factors.get('macd_hist', 0)
                        macd_str = f"{'+' if macd_hist > 0 else ''}{macd_hist:.2f}"

                        # 因子组得分 - 注意：trend_score/momentum_score/money_score 在顶层返回，不在 factors_score 中
                        trend_score = r.get('trend_score', 50)
                        momentum_score = r.get('momentum_score', 50)
                        money_score = r.get('money_score', 50)
                        position_mod = r.get('position_modifier', 1.0)

                        f.write(f"| {i} | {symbol_short} | {name} | ¥{r['close']:.2f} | {r['score']:.1f} | {r['score_grade']} | {trend_score:.0f} | {momentum_score:.0f} | {money_score:.0f} | {position_mod:.2f} | {bias_str} | {rsi_val:.0f} | {macd_str} |\n")

                f.write("\n---\n\n")

                # 新增：技术指标原始数值表
                f.write("## 📈 技术指标原始数值\n\n")
                f.write("| 代码 | 名称 | MA5 | MA10 | MA20 | K | D | J | RSI | MACD-DIF | MACD-DEA | 量比 |\n")
                f.write("|------|------|-----|------|------|---|---|---|-----|----------|----------|------|\n")

                for i, r in enumerate(results[:top_n], 1):
                    symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
                    name = r.get('name', symbol_short)

                    factors_raw = r.get('factors_raw', {})
                    trend_factors = factors_raw.get('trend_factors', {})
                    momentum_factors = factors_raw.get('momentum_factors', {})
                    money_factors = factors_raw.get('money_factors', {})
                    aux_factors = factors_raw.get('aux_factors', {})

                    # 均线值（在 aux_factors 中）
                    ma5 = aux_factors.get('ma5', 0)
                    ma10 = aux_factors.get('ma10', 0)
                    ma20 = aux_factors.get('ma20', 0)

                    # KDJ
                    k_val = momentum_factors.get('k', 50)
                    d_val = momentum_factors.get('d', 50)
                    j_val = momentum_factors.get('j', 50)

                    # RSI
                    rsi_val = momentum_factors.get('rsi', 50)

                    # MACD
                    macd_dif = trend_factors.get('macd_dif', 0)
                    macd_dea = trend_factors.get('macd_dea', 0)

                    # 量比
                    vol_ratio = money_factors.get('volume_ratio', 1.0)

                    f.write(f"| {symbol_short} | {name} | {ma5:.2f} | {ma10:.2f} | {ma20:.2f} | {k_val:.1f} | {d_val:.1f} | {j_val:.1f} | {rsi_val:.0f} | {macd_dif:.3f} | {macd_dea:.3f} | {vol_ratio:.2f} |\n")

                f.write("\n---\n\n")

                # 新增：评分逻辑解释
                f.write("## 💡 评分逻辑说明\n\n")
                f.write("### 三大类因子组权重\n\n")
                f.write("| 因子组 | 权重 | 说明 |\n")
                f.write("|--------|------|------|\n")
                f.write("| 趋势因子 | 35% | 确认趋势方向：均线排列、DMI强度、MACD方向 |\n")
                f.write("| 动能因子 | 40% | 确认动能强度：KDJ位置、RSI强度、MTM动量、ROC变化率 |\n")
                f.write("| 资金因子 | 25% | 确认资金真实性：OBV资金流、MFI强度、量价关系 |\n")
                f.write("\n")

                f.write("### 位置修正系数\n\n")
                f.write("| 区域 | BIAS(20) | 修正系数 | 风险等级 |\n")
                f.write("|------|----------|----------|----------|\n")
                f.write("| 安全区 | -8% ~ +2% | 1.0 | 低风险，可积极建仓 |\n")
                f.write("| 适中区 | +2% ~ +5% | 0.85 | 中等风险，适度参与 |\n")
                f.write("| 警戒区 | +5% ~ +8% | 0.6 | 较高风险，谨慎参与 |\n")
                f.write("| 危险区 | > +8% | 0.35 | 高风险，建议观望 |\n")
                f.write("\n")

                f.write("### 触发类型说明\n\n")
                f.write("| 类型 | 含义 | 建议 |\n")
                f.write("|------|------|------|\n")
                f.write("| breakout | 突破信号，价格放量突破平台 | 积极买入 |\n")
                f.write("| pullback | 回踩信号，价格回踩均线企稳 | 逢低买入 |\n")
                f.write("| none | 普通信号，无特殊触发 | 按评分决定 |\n")
                f.write("\n---\n\n")

                # 新增：Top 3 个股评分解析
                f.write("## 🔍 Top 3 个股评分解析\n\n")

                for i, r in enumerate(results[:3], 1):
                    symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
                    name = r.get('name', symbol_short)
                    score = r['score']
                    score_grade = r['score_grade']

                    f.write(f"### #{i} {symbol_short} {name} ({score:.1f}分 - {score_grade})\n\n")

                    # 因子得分分析
                    factors_score = r.get('factors_score', {})
                    factors_raw = r.get('factors_raw', {})

                    trend_factors = factors_raw.get('trend_factors', {})
                    momentum_factors = factors_raw.get('momentum_factors', {})
                    money_factors = factors_raw.get('money_factors', {})

                    f.write("**趋势分析：**\n")
                    ma_alignment = trend_factors.get('ma_alignment', 50)
                    dmi_strength = trend_factors.get('dmi_strength', 50)
                    macd_direction = trend_factors.get('macd_direction', 50)
                    adx = trend_factors.get('adx', 0)
                    f.write(f"- 均线排列得分: {ma_alignment:.0f}分\n")
                    f.write(f"- DMI趋势强度: {dmi_strength:.0f}分 (ADX={adx:.1f})\n")
                    f.write(f"- MACD方向得分: {macd_direction:.0f}分\n\n")

                    f.write("**动能分析：**\n")
                    kdj_pos = momentum_factors.get('kdj_position', 50)
                    rsi_strength = momentum_factors.get('rsi_strength', 50)
                    k_val = momentum_factors.get('k', 50)
                    rsi_val = momentum_factors.get('rsi', 50)
                    f.write(f"- KDJ位置得分: {kdj_pos:.0f}分 (K={k_val:.1f})\n")
                    f.write(f"- RSI强度得分: {rsi_strength:.0f}分 (RSI={rsi_val:.1f})\n\n")

                    f.write("**资金分析：**\n")
                    obv_flow = money_factors.get('obv_flow', 50)
                    mfi_strength = money_factors.get('mfi_strength', 50)
                    vol_ratio = money_factors.get('volume_ratio', 1.0)
                    f.write(f"- OBV资金流得分: {obv_flow:.0f}分\n")
                    f.write(f"- MFI强度得分: {mfi_strength:.0f}分\n")
                    f.write(f"- 量比: {vol_ratio:.2f}\n\n")

                    # 综合评价
                    f.write("**综合评价：**\n")
                    if score >= 80:
                        f.write(f"✅ 强势股票，多项指标共振向上，趋势明确，可重点关注。\n\n")
                    elif score >= 65:
                        f.write(f"⚠️ 中等强度，部分指标表现良好，建议结合其他因素判断。\n\n")
                    else:
                        f.write(f"ℹ️ 表现一般，建议观望或等待更好的入场时机。\n\n")

                    f.write("---\n\n")

                # Generate detailed report for each top stock (with three-system scoring)
                for i, r in enumerate(results[:top_n], 1):
                    symbol = r['symbol']
                    name = r.get('name', symbol)

                    typer.echo(f"  [{i}/{top_n}] 生成 {symbol} ({name}) 的详细报告...")

                    try:
                        # Use analyze_stock_with_context for three-system scoring
                        context, report = analyzer.analyze_stock_with_context(symbol, days)

                        # Write to file
                        f.write(f"## #{i} {symbol} - {name}\n\n")
                        f.write(report)
                        f.write("\n\n---\n\n")

                    except Exception as e:
                        f.write(f"## #{i} {symbol} - {name}\n\n")
                        f.write(f"生成报告时出错: {e}\n\n")
                        f.write("---\n\n")

            typer.echo(f"\n✅ 详细分析报告已保存: {report_file}")

        except Exception as e:
            typer.echo(f"\n⚠️ 生成详细报告失败: {e}")


@app.command(name="history")
def scan_history(
    market: str = typer.Option(None, "--market", "-m", help="Filter by market"),
    limit: int = typer.Option(10, "--limit", "-n", help="Number of records to show"),
    db_path: str = typer.Option("./quanttool.db", "--db-path", help="Path to the database file"),
):
    """查看扫描历史记录。"""
    try:
        meta_db = MetaDB(db_path)
        history = meta_db.get_scan_history(market=market, limit=limit)

        if not history:
            typer.echo("暂无扫描历史记录")
            return

        typer.echo("\n" + "=" * 100)
        typer.echo("📊 扫描历史记录")
        typer.echo("=" * 100)
        typer.echo(f"{'ID':<36} {'日期':<20} {'市场':<10} {'股票数':<8} {'乖离率过滤':<20}")
        typer.echo("-" * 100)

        for record in history:
            scan_id_short = record['id'][:8] + "..."
            scan_date = record['scan_date'][:19] if record['scan_date'] else "N/A"
            market_str = record['market']
            total_stocks = record['total_stocks']

            bias_filter_str = "无"
            if record['bias_filter_min'] is not None or record['bias_filter_max'] is not None:
                bias_filter_str = ""
                if record['bias_filter_min'] is not None:
                    bias_filter_str += f">={record['bias_filter_min']}%"
                if record['bias_filter_max'] is not None:
                    if bias_filter_str:
                        bias_filter_str += ","
                    bias_filter_str += f"<={record['bias_filter_max']}%"

            typer.echo(f"{scan_id_short:<36} {scan_date:<20} {market_str:<10} {total_stocks:<8} {bias_filter_str:<20}")

        typer.echo("=" * 100)
        typer.echo(f"\n共 {len(history)} 条记录")
        typer.echo(f"使用 `quanttool analysis view <scan_id>` 查看详细结果")
        typer.echo(f"使用 `quanttool analysis compare <scan_id1> <scan_id2>` 对比两次扫描")

    except Exception as e:
        typer.echo(f"查询扫描历史失败: {e}")


@app.command(name="view")
def view_scan(
    scan_id: str = typer.Argument(..., help="Scan record ID (or first 8 characters)"),
    top_n: int = typer.Option(10, "--top", "-n", help="Number of top stocks to show"),
    db_path: str = typer.Option("./quanttool.db", "--db-path", help="Path to the database file"),
):
    """查看指定扫描记录的详细结果。"""
    try:
        meta_db = MetaDB(db_path)

        # Try to find scan by full ID or partial ID
        if len(scan_id) < 36:
            # Search for partial ID match
            history = meta_db.get_scan_history(limit=100)
            matching_scans = [h for h in history if h['id'].startswith(scan_id)]
            if not matching_scans:
                typer.echo(f"未找到匹配的扫描记录: {scan_id}")
                return
            if len(matching_scans) > 1:
                typer.echo(f"找到多个匹配的扫描记录，请提供更完整的ID:")
                for scan in matching_scans:
                    typer.echo(f"  {scan['id']}")
                return
            scan_id = matching_scans[0]['id']

        scan_record = meta_db.get_scan_record(scan_id)

        if not scan_record:
            typer.echo(f"未找到扫描记录: {scan_id}")
            return

        typer.echo("\n" + "=" * 110)
        typer.echo(f"📊 扫描详情 - {scan_record['scan_date'][:19]}")
        typer.echo("=" * 110)
        typer.echo(f"扫描ID: {scan_record['id']}")
        typer.echo(f"市场: {scan_record['market']}")
        typer.echo(f"分析天数: {scan_record['days_analyzed']}")
        typer.echo(f"扫描股票数: {scan_record['total_stocks']}")

        if scan_record['bias_filter_min'] is not None or scan_record['bias_filter_max'] is not None:
            typer.echo(f"乖离率过滤: ", nl=False)
            if scan_record['bias_filter_min'] is not None:
                typer.echo(f"BIAS >= {scan_record['bias_filter_min']}% ", nl=False)
            if scan_record['bias_filter_max'] is not None:
                typer.echo(f"BIAS <= {scan_record['bias_filter_max']}%", nl=False)
            typer.echo()

        results = scan_record.get('results', [])

        typer.echo("\n" + "-" * 110)
        typer.echo(f"{'排名':<4} {'代码':<10} {'股票名称':<10} {'收盘价':<10} {'评分':<10} {'BIAS(20)':<10} {'等级':<8} {'触发类型':<12}")
        typer.echo("-" * 110)

        for i, r in enumerate(results[:top_n], 1):
            symbol_short = r['symbol'].replace('.SH', '').replace('.SZ', '')
            name_short = r['name'][:8] if len(r['name']) > 8 else r['name']
            bias_str = f"{r.get('bias_20', r.get('bias_6', 0)):+.2f}%"
            score = r.get('score', 0)
            score_grade = r.get('score_grade', 'N/A')
            trigger_type = r.get('trigger_type', 'N/A')
            typer.echo(f"{i:<4} {symbol_short:<10} {name_short:<10} ¥{r['close']:<9.2f} {score:<10.1f} {bias_str:<10} {score_grade:<8} {trigger_type:<12}")

        typer.echo("=" * 110)

    except Exception as e:
        typer.echo(f"查看扫描记录失败: {e}")


@app.command(name="compare")
def compare_scans(
    scan_id_1: str = typer.Argument(..., help="First scan record ID"),
    scan_id_2: str = typer.Argument(..., help="Second scan record ID"),
    db_path: str = typer.Option("./quanttool.db", "--db-path", help="Path to the database file"),
):
    """对比两次扫描的结果。"""
    try:
        meta_db = MetaDB(db_path)

        # Resolve partial IDs
        def resolve_scan_id(partial_id: str) -> str:
            if len(partial_id) >= 36:
                return partial_id
            history = meta_db.get_scan_history(limit=100)
            matching = [h for h in history if h['id'].startswith(partial_id)]
            if not matching:
                raise ValueError(f"未找到匹配的扫描记录: {partial_id}")
            if len(matching) > 1:
                raise ValueError(f"ID '{partial_id}' 匹配到多个记录，请提供更完整的ID")
            return matching[0]['id']

        scan_id_1 = resolve_scan_id(scan_id_1)
        scan_id_2 = resolve_scan_id(scan_id_2)

        comparison = meta_db.compare_scans(scan_id_1, scan_id_2)

        typer.echo("\n" + "=" * 100)
        typer.echo("📊 扫描结果对比")
        typer.echo("=" * 100)
        typer.echo(f"扫描 1: {comparison['scan_id_1'][:8]}...")
        typer.echo(f"扫描 2: {comparison['scan_id_2'][:8]}...")
        typer.echo()

        # Common stocks
        typer.echo(f"📈 共同关注的股票: {comparison['common_count']} 只")
        if comparison['common_stocks']:
            typer.echo(f"{'股票':<10} {'扫描1排名':<10} {'扫描2排名':<10} {'排名变化':<10} {'扫描1评分':<10} {'扫描2评分':<10} {'评分变化':<10}")
            typer.echo("-" * 80)
            for stock in comparison['common_stocks'][:10]:  # Show top 10
                rank_change = f"+{stock['rank_change']}" if stock['rank_change'] > 0 else str(stock['rank_change'])
                score_change = f"+{stock['score_change']}" if stock['score_change'] > 0 else str(stock['score_change'])
                symbol_short = stock['symbol'].replace('.SH', '').replace('.SZ', '')
                typer.echo(f"{symbol_short:<10} {stock['scan_1_rank']:<10} {stock['scan_2_rank']:<10} {rank_change:<10} {stock['scan_1_score']:<10} {stock['scan_2_score']:<10} {score_change:<10}")

        typer.echo()

        # Only in scan 1
        if comparison['only_in_scan_1']:
            typer.echo(f"📉 仅在扫描1中出现: {comparison['only_in_scan_1_count']} 只")
            typer.echo(f"   {', '.join([s.replace('.SH', '').replace('.SZ', '') for s in comparison['only_in_scan_1'][:10]])}")
            if len(comparison['only_in_scan_1']) > 10:
                typer.echo(f"   ... 还有 {len(comparison['only_in_scan_1']) - 10} 只")
            typer.echo()

        # Only in scan 2
        if comparison['only_in_scan_2']:
            typer.echo(f"📈 仅在扫描2中出现: {comparison['only_in_scan_2_count']} 只")
            typer.echo(f"   {', '.join([s.replace('.SH', '').replace('.SZ', '') for s in comparison['only_in_scan_2'][:10]])}")
            if len(comparison['only_in_scan_2']) > 10:
                typer.echo(f"   ... 还有 {len(comparison['only_in_scan_2']) - 10} 只")

        typer.echo("=" * 100)

    except Exception as e:
        typer.echo(f"对比扫描记录失败: {e}")


@app.command(name="trend")
def analyze_trend(
    symbol: str = typer.Argument(..., help="Stock symbol to analyze (e.g., 601777, 000001.SZ)"),
    days: int = typer.Option(360, "--days", "-d", help="Number of days to analyze (default: 360)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file to save the analysis report")
):
    """
    使用趋势评分系统分析股票。

    趋势评分系统特点：
    - 纯趋势强度评分，不再使用位置惩罚系数
    - 强势股得高分，不会被过度惩罚
    - 时机系数用于风险控制

    Examples:
        # 分析单只股票
        quanttool analysis trend 600519

        # 指定分析天数
        quanttool analysis trend 600519 --days 500

        # 保存报告到文件
        quanttool analysis trend 600519 -o report.md
    """
    typer.echo(f"正在使用趋势评分系统分析股票：{symbol}")
    typer.echo(f"分析周期：{days} 天")
    typer.echo("-" * 60)

    try:
        # 使用 StockAnalyzer 获取数据（自动使用增量数据管理器）
        analyzer = StockAnalyzer()
        df = analyzer.get_stock_data(symbol, days=days)

        if df.empty:
            typer.echo(f"无法获取 {symbol} 的数据")
            return

        # 计算趋势评分
        trend_system = TrendScoringSystem()
        result = trend_system.calculate_score(df)

        # 显示结果
        typer.echo("\n" + "=" * 60)
        typer.echo(f"📊 趋势评分结果")
        typer.echo("=" * 60)

        if not result.passed_hard_filter:
            typer.echo(f"\n❌ 未通过硬过滤")
            typer.echo(f"   原因: {result.hard_filter_reason}")
            typer.echo("\n硬过滤标准：")
            typer.echo("  • 20日均成交额 > 1亿")
            typer.echo("  • MA20斜率 > 0（趋势向上）")
            typer.echo("  • 股价 > MA20（趋势存在）")
        else:
            # 评分等级
            if result.final_score >= 90:
                grade = "极强趋势 🚀"
            elif result.final_score >= 75:
                grade = "强趋势 📈"
            elif result.final_score >= 60:
                grade = "趋势一般 ➖"
            else:
                grade = "趋势弱 📉"

            typer.echo(f"\n✅ 通过硬过滤")
            typer.echo(f"\n**最终评分**: {result.final_score:.1f}分 ({grade})")
            typer.echo(f"   趋势总分: {result.trend_total_score:.1f}分")
            typer.echo(f"   时机系数: {result.timing_coefficient:.2f} ({result.timing_type})")

            # 各因子得分
            typer.echo(f"\n**因子得分明细**:")
            typer.echo(f"   均线结构: {result.ma_structure_score:.1f}分 (权重30%)")
            typer.echo(f"   价格动能: {result.price_momentum_score:.1f}分 (权重30%)")
            typer.echo(f"   量能配合: {result.volume_score:.1f}分 (权重25%)")
            typer.echo(f"   相对强度: {result.relative_strength_score:.1f}分 (权重15%)")

            # 时机分析详情
            details = result.details
            timing_details = details.get('timing', {})
            if timing_details:
                typer.echo(f"\n**时机分析**:")
                typer.echo(f"   时机类型: {result.timing_type}")
                if 'timing_reason' in timing_details:
                    typer.echo(f"   原因: {timing_details['timing_reason']}")
                if 'return_5d' in timing_details:
                    typer.echo(f"   5日涨幅: {timing_details['return_5d']:.2f}%")
                if 'return_10d' in timing_details:
                    typer.echo(f"   10日涨幅: {timing_details['return_10d']:.2f}%")
                if 'dist_to_ma20' in timing_details:
                    typer.echo(f"   距MA20: {timing_details['dist_to_ma20']:.2f}%")

            # 操作建议
            typer.echo(f"\n**操作建议**:")
            if result.final_score >= 75:
                typer.echo("   ✅ 趋势强劲，可考虑入场或加仓")
                if result.timing_coefficient >= 1.1:
                    typer.echo("   🎯 时机较好，建议积极参与")
                elif result.timing_coefficient <= 0.8:
                    typer.echo("   ⚠️ 时机一般，建议控制仓位")
            elif result.final_score >= 60:
                typer.echo("   ➖ 趋势一般，建议观望或轻仓")
            else:
                typer.echo("   ❌ 趋势较弱，不建议入场")

        typer.echo("\n" + "=" * 60)

        # 保存报告
        if output:
            report_lines = []
            report_lines.append(f"# 趋势评分分析报告 - {symbol}")
            report_lines.append(f"\n**分析日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            report_lines.append(f"**分析周期**: {days}天")
            report_lines.append(f"\n## 评分结果\n")
            report_lines.append(f"- **最终评分**: {result.final_score:.1f}分")
            report_lines.append(f"- **趋势总分**: {result.trend_total_score:.1f}分")
            report_lines.append(f"- **时机系数**: {result.timing_coefficient:.2f}")
            report_lines.append(f"- **时机类型**: {result.timing_type}")
            report_lines.append(f"- **通过硬过滤**: {'是' if result.passed_hard_filter else '否'}")
            if not result.passed_hard_filter:
                report_lines.append(f"- **过滤原因**: {result.hard_filter_reason}")
            report_lines.append(f"\n## 因子得分\n")
            report_lines.append(f"| 因子 | 得分 | 权重 |")
            report_lines.append(f"|------|------|------|")
            report_lines.append(f"| 均线结构 | {result.ma_structure_score:.1f} | 30% |")
            report_lines.append(f"| 价格动能 | {result.price_momentum_score:.1f} | 30% |")
            report_lines.append(f"| 量能配合 | {result.volume_score:.1f} | 25% |")
            report_lines.append(f"| 相对强度 | {result.relative_strength_score:.1f} | 15% |")

            with open(output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report_lines))
            typer.echo(f"\n报告已保存至: {output}")

    except Exception as e:
        typer.echo(f"分析失败: {e}")


if __name__ == "__main__":
    app()