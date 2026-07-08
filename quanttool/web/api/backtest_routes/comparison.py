"""Backtest strategy comparison API route."""

from datetime import datetime
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter

from ....core.logging import get_logger
from ...schemas.backtest import BacktestRequest
from ..utils import to_python_types


logger = get_logger(__name__)
router = APIRouter()

@router.post("/backtest/run-all")
async def run_all_strategies_backtest(request: BacktestRequest) -> Dict[str, Any]:
    """
    运行所有策略的回测

    返回所有策略的回测结果对比
    """
    # 所有可用策略（移除 gbm，需要额外模型加载）
    all_strategies = [
        "score", "ma_cross", "dual_ma",
        "breakout", "macd", "rsi", "kdj", "bollinger", "turtle"
    ]

    results = []

    # 获取请求的日期范围，如果日期过旧则使用最近的数据
    requested_start = datetime.fromisoformat(request.get_start_date())
    requested_end = datetime.fromisoformat(request.get_end_date())

    # 检查数据是否在有效范围内
    from quanttool.factors.stock_analyzer import StockAnalyzer
    analyzer = StockAnalyzer(use_realtime_price=True)

    # 检查第一只股票的数据范围
    check_symbol = request.symbols[0] if request.symbols else '300750'
    check_df = analyzer.get_stock_data(check_symbol, 60)

    if not check_df.empty:
        # 使用数据中最新的日期
        df_dates = check_df['timestamp'] if 'timestamp' in check_df.columns else check_df.index
        latest_date = df_dates.max()
        earliest_date = df_dates.min()

        # 如果请求的结束日期比最新数据还新，使用最新数据日期
        if requested_end > latest_date:
            end_date = latest_date
            # 从最新日期往前推与请求相同的天数
            days_requested = (requested_end - requested_start).days
            start_date = latest_date - pd.Timedelta(days=days_requested)
        else:
            start_date = requested_start
            end_date = requested_end
    else:
        # 数据不可用，使用默认值
        start_date = requested_start
        end_date = requested_end

    from quanttool.application.backtest_service import BacktestService

    # 获取沪深300基准收益
    benchmark_return = 0
    benchmark_annual_return = 0
    benchmark_curve = []  # 基准收益曲线
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer
        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)

        # 尝试获取沪深300 ETF (510300) 作为基准
        # qlib 中没有 ETF 数据，会自动回退到备用数据源
        benchmark_df = analyzer.get_stock_data('510300.SH', 365)
        if benchmark_df.empty:
            # 如果 ETF 数据不可用，使用浦发银行（SH600000）作为沪深300参考
            logger.info("ETF 数据不可用，使用 SH600000 作为市场参考")
            benchmark_df = analyzer.get_stock_data('SH600000', 365)

        if not benchmark_df.empty:
            # 处理不同的日期列名
            if 'trade_date' in benchmark_df.columns:
                benchmark_df = benchmark_df.set_index('trade_date')
            elif 'timestamp' in benchmark_df.columns:
                benchmark_df = benchmark_df.set_index('timestamp')
            elif 'date' in benchmark_df.columns:
                benchmark_df = benchmark_df.set_index('date')

            # 确保索引是 datetime 类型
            benchmark_df.index = pd.to_datetime(benchmark_df.index)

            # 过滤日期范围 - 统一转换为 datetime 比较
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            start_mask = benchmark_df.index >= start_dt
            end_mask = benchmark_df.index <= end_dt

            benchmark_period = benchmark_df[start_mask & end_mask]
            if len(benchmark_period) >= 2:
                start_price = benchmark_period.iloc[0]['close']
                end_price = benchmark_period.iloc[-1]['close']
                benchmark_return = (end_price - start_price) / start_price

                # 计算基准收益曲线
                initial_cash = request.initial_cash
                for idx, (date, row) in enumerate(benchmark_period.iterrows()):
                    cumulative_return = (row['close'] - start_price) / start_price
                    benchmark_curve.append({
                        'timestamp': date.strftime('%Y-%m-%d'),
                        'value': initial_cash * (1 + cumulative_return)
                    })

                # 计算年化收益
                days = (end_date - start_date).days
                if days > 0:
                    benchmark_annual_return = (1 + benchmark_return) ** (365 / days) - 1
    except Exception as e:
        logger.warning(f"获取基准收益失败: {e}")

    # 复用同一个 BacktestService 实例，避免重复初始化
    backtest_service = None

    for strategy_name in all_strategies:
        try:
            # 首次创建服务，后续复用
            if backtest_service is None:
                backtest_service = BacktestService(use_qlib=True, use_realtime_price=True)

            result = backtest_service.run_backtest(
                strategy_name=strategy_name,
                strategy_params={},
                symbols=request.symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1d",
                initial_cash=request.initial_cash,
                commission_rate=request.commission_rate,
            )

            # 计算相对于基准的超额收益
            strategy_return = result.annual_return or 0
            excess_vs_benchmark = strategy_return - benchmark_annual_return

            # 收集结果
            # 转换 equity_curve 为可序列化格式
            equity_curve_serializable = []
            if hasattr(result, 'equity_curve') and result.equity_curve:
                for point in result.equity_curve:
                    ts = point.get('timestamp')
                    if hasattr(ts, 'timestamp'):
                        # pandas Timestamp 或 datetime
                        ts_str = ts.strftime('%Y-%m-%d') if hasattr(ts, 'strftime') else str(ts)
                    else:
                        ts_str = str(ts) if ts else ''
                    equity_curve_serializable.append({
                        "timestamp": ts_str,
                        "portfolio_value": float(point.get('portfolio_value', 0) or 0)
                    })

            # 转换 trades 为可序列化格式
            trades_serializable = []
            if result.trades:
                for t in result.trades:
                    ts = t.timestamp if hasattr(t, 'timestamp') else None
                    if hasattr(ts, 'strftime'):
                        ts_str = ts.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        ts_str = str(ts) if ts else ''
                    # 获取 side/action
                    side = getattr(t, 'side', None)
                    if side is None:
                        side = getattr(t, 'action', 'sell')
                    action = side.value if hasattr(side, 'value') else str(side)

                    trades_serializable.append({
                        "timestamp": ts_str,
                        "action": action,
                        "type": action,
                        "price": float(t.price) if t.price else 0,
                        "shares": float(t.quantity) if t.quantity else 0,
                        "pnl": float(t.pnl) if hasattr(t, 'pnl') and t.pnl else None
                    })

            result_dict = {
                "strategy": strategy_name,
                "strategy_display": {
                    "score": "评分策略",
                    "ma_cross": "均线交叉",
                    "dual_ma": "双均线策略",
                    "breakout": "突破策略",
                    "macd": "MACD策略",
                    "rsi": "RSI策略",
                    "kdj": "KDJ策略",
                    "bollinger": "布林带策略",
                    "turtle": "海龟交易",
                    "gbm": "GBM机器学习",
                }.get(strategy_name, strategy_name),
                "total_return": float(result.total_return or 0),
                "annual_return": float(result.annual_return or 0),
                "excess_return": float(excess_vs_benchmark),  # 相对沪深300的超额收益
                "benchmark_return": float(benchmark_return),  # 基准收益
                "max_drawdown": float(getattr(result, 'max_drawdown', 0) or 0),
                "sharpe_ratio": float(getattr(result, 'sharpe_ratio', 0) or 0),
                "win_rate": float(result.win_rate or 0),
                "total_trades": int(result.total_trades or 0),
                "profit_factor": float(getattr(result, 'profit_factor', 0) or 0),
                "final_capital": float(result.final_capital or request.initial_cash),
                "trades_count": len(result.trades) if hasattr(result, 'trades') and result.trades else 0,
                "equity_curve": equity_curve_serializable,
                "trades": trades_serializable,
            }
            results.append(result_dict)

        except Exception as e:
            logger.warning(f"策略 {strategy_name} 回测失败: {e}")
            results.append({
                "strategy": strategy_name,
                "strategy_display": {
                    "score": "评分策略",
                    "ma_cross": "均线交叉",
                    "dual_ma": "双均线策略",
                    "breakout": "突破策略",
                    "macd": "MACD策略",
                    "rsi": "RSI策略",
                    "kdj": "KDJ策略",
                    "bollinger": "布林带策略",
                    "turtle": "海龟交易",
                    "gbm": "GBM机器学习",
                }.get(strategy_name, strategy_name),
                "error": str(e),
                "total_return": 0,
                "annual_return": 0,
            })

    # 按年化收益排序
    results.sort(key=lambda x: x.get('annual_return', 0), reverse=True)

    return {
        "symbols": request.symbols,
        "start_date": request.get_start_date(),
        "end_date": request.get_end_date(),
        "initial_cash": request.initial_cash,
        "benchmark_return": float(benchmark_return) if benchmark_return else 0,
        "benchmark_annual_return": float(benchmark_annual_return) if benchmark_annual_return else 0,
        "benchmark_curve": benchmark_curve,
        "results": to_python_types(results),
        "total_strategies": len(all_strategies),
        "successful_strategies": len([r for r in results if not r.get('error')]),
    }
