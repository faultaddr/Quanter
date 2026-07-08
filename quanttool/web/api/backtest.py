"""Backtest API routes."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
import os
import queue
import threading
import time
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...core.logging import get_logger
from .utils import get_cached_analysis, set_cached_analysis, to_python_types


logger = get_logger(__name__)
router = APIRouter()

from ..schemas.backtest import BacktestRequest


@router.get("/backtest/strategies")
async def list_backtest_strategies() -> List[Dict[str, Any]]:
    """列出可用的回测策略"""
    return [
        {
            "name": "ma_cross",
            "display_name": "均线交叉策略",
            "description": "短期均线上穿长期均线买入，下穿卖出",
            "category": "traditional",
            "params": {
                "short_window": {"type": "int", "default": 10, "description": "短期均线周期"},
                "long_window": {"type": "int", "default": 30, "description": "长期均线周期"}
            }
        },
        {
            "name": "breakout",
            "display_name": "突破策略",
            "description": "价格突破N日高点买入，跌破N日低点卖出",
            "category": "traditional",
            "params": {
                "period": {"type": "int", "default": 20, "description": "突破周期"}
            }
        },
        {
            "name": "score",
            "display_name": "评分策略",
            "description": "首次突破策略：评分首次突破阈值时买入/卖出。买入=80,卖出=60为最优参数",
            "category": "traditional",
            "params": {
                "buy_threshold": {"type": "float", "default": 80.0, "description": "买入评分阈值（首次突破触发）"},
                "sell_threshold": {"type": "float", "default": 60.0, "description": "卖出评分阈值（首次跌破触发）"}
            }
        },
        {
            "name": "enhanced_score",
            "display_name": "增强评分策略",
            "description": "首次突破+动态权重+风险控制。评分首次突破80买入，首次跌破60卖出",
            "category": "traditional",
            "params": {
                "buy_threshold": {"type": "float", "default": 80.0, "description": "买入评分阈值"},
                "sell_threshold": {"type": "float", "default": 60.0, "description": "卖出评分阈值"},
                "use_dynamic_weights": {"type": "bool", "default": True, "description": "使用动态权重"},
                "use_risk_control": {"type": "bool", "default": True, "description": "使用风险控制"}
            }
        },
        {
            "name": "dual_ma",
            "display_name": "双均线策略",
            "description": "经典双均线交叉策略，支持多周期组合",
            "category": "traditional",
            "params": {
                "fast_period": {"type": "int", "default": 5, "description": "快线周期"},
                "slow_period": {"type": "int", "default": 20, "description": "慢线周期"}
            }
        },
        {
            "name": "macd",
            "display_name": "MACD策略",
            "description": "基于MACD指标的金叉死叉信号",
            "category": "traditional",
            "params": {
                "fast_period": {"type": "int", "default": 12, "description": "快线周期"},
                "slow_period": {"type": "int", "default": 26, "description": "慢线周期"},
                "signal_period": {"type": "int", "default": 9, "description": "信号线周期"}
            }
        },
        {
            "name": "rsi",
            "display_name": "RSI策略",
            "description": "基于RSI超买超卖信号",
            "category": "traditional",
            "params": {
                "period": {"type": "int", "default": 14, "description": "RSI周期"},
                "oversold": {"type": "int", "default": 30, "description": "超卖阈值"},
                "overbought": {"type": "int", "default": 70, "description": "超买阈值"}
            }
        },
        {
            "name": "kdj",
            "display_name": "KDJ策略",
            "description": "基于KDJ指标的金叉死叉信号",
            "category": "traditional",
            "params": {
                "n": {"type": "int", "default": 9, "description": "KDJ周期"},
                "m1": {"type": "int", "default": 3, "description": "K平滑周期"},
                "m2": {"type": "int", "default": 3, "description": "D平滑周期"}
            }
        },
        {
            "name": "bollinger",
            "display_name": "布林带策略",
            "description": "基于布林带上下轨突破信号",
            "category": "traditional",
            "params": {
                "period": {"type": "int", "default": 20, "description": "布林带周期"},
                "std_dev": {"type": "float", "default": 2.0, "description": "标准差倍数"}
            }
        },
        {
            "name": "turtle",
            "display_name": "海龟交易策略",
            "description": "经典海龟交易系统，基于通道突破",
            "category": "traditional",
            "params": {
                "entry_period": {"type": "int", "default": 20, "description": "入场周期"},
                "exit_period": {"type": "int", "default": 10, "description": "出场周期"}
            }
        },
        {
            "name": "gbm",
            "display_name": "GBM机器学习策略",
            "description": "基于LightGBM的机器学习策略，使用Alpha158特征和百分位排名信号",
            "category": "ml",
            "params": {
                "buy_threshold": {"type": "float", "default": 0.35, "description": "买入百分位阈值（前65%触发买入）"},
                "sell_threshold": {"type": "float", "default": 0.35, "description": "卖出百分位阈值（后35%触发卖出）"},
                "stop_loss_pct": {"type": "float", "default": 0.05, "description": "止损比例"},
                "take_profit_pct": {"type": "float", "default": 0.10, "description": "止盈比例"}
            }
        }
    ]


@router.get("/backtest/history")


@router.post("/backtest/run")
async def run_backtest(request: BacktestRequest) -> Dict[str, Any]:
    """
    运行回测

    支持的策略：
    - ma_cross: 均线交叉
    - breakout: 突破策略
    - trend_momentum: 趋势动量
    - adaptive_threshold: 自适应阈值

    数据提供者：
    - enhanced_data_fetcher: 直接从网络获取
    - incremental_data_fetcher: 优先使用缓存，增量拉取（推荐）
    """
    try:
        from quanttool.application.backtest_service import BacktestService

        # 解析日期（使用动态默认值：最近一年）
        start_date = datetime.fromisoformat(request.get_start_date())
        end_date = datetime.fromisoformat(request.get_end_date())

        # 初始化回测服务（使用实时价格数据，避免复权价格显示异常）
        backtest_service = BacktestService(use_qlib=True, use_realtime_price=True)

        # 运行回测
        result = backtest_service.run_backtest(
            strategy_name=request.strategy_name,
            strategy_params=request.strategy_params,
            symbols=request.symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
            initial_cash=request.initial_cash,
            commission_rate=request.commission_rate,
        )

        # 转换结果
        result_dict = {
            "strategy": request.strategy_name,
            "symbols": request.symbols,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "excess_return": result.annual_return - 0.05 if result.annual_return else 0,  # 假设基准5%
            "max_drawdown": getattr(result, 'max_drawdown', 0),
            "sharpe_ratio": getattr(result, 'sharpe_ratio', 0),
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "profit_factor": getattr(result, 'profit_factor', 0),
            "trades": []
        }

        # 添加交易记录（如果有）
        if hasattr(result, 'trades') and result.trades:
            for trade in result.trades[:50]:  # 限制返回数量
                # Trade 对象使用 side 字段 (OrderSide 枚举)
                side = getattr(trade, 'side', None)
                if side is None:
                    side = getattr(trade, 'action', 'sell')
                action = side.value if hasattr(side, 'value') else str(side)

                result_dict["trades"].append({
                    "strategy": request.strategy_name,
                    "symbol": getattr(trade, 'symbol', ''),
                    "action": action,
                    "type": action,
                    "price": getattr(trade, 'price', 0),
                    "shares": getattr(trade, 'quantity', getattr(trade, 'shares', 0)),
                    "timestamp": str(getattr(trade, 'timestamp', '')),
                    "profit": getattr(trade, 'pnl', getattr(trade, 'profit', None)),
                })

        return result_dict

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"回测失败: {str(e)}")


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


# ==================== 流式回测 API ====================

@router.post("/backtest/run-all-stream")
async def run_all_strategies_backtest_stream(request: BacktestRequest):
    """
    流式运行所有策略的回测

    每个策略完成后立即返回，使用 SSE (Server-Sent Events)
    """
    from fastapi.responses import StreamingResponse
    import json
    import math

    # 自定义 JSON 编码器，处理 Infinity
    class SafeJSONEncoder(json.JSONEncoder):
        def encode(self, o):
            return super().encode(self._clean(o))

        def _clean(self, obj):
            if isinstance(obj, dict):
                return {k: self._clean(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._clean(v) for v in obj]
            elif isinstance(obj, float):
                if math.isinf(obj):
                    return None  # 将 Infinity 转为 null
                return obj
            elif isinstance(obj, float) and math.isnan(obj):
                return None
            return obj

    def safe_json_dumps(obj):
        return json.dumps(obj, cls=SafeJSONEncoder, ensure_ascii=False)

    # 所有可用策略
    all_strategies = [
        "score", "ma_cross", "dual_ma",
        "breakout", "macd", "rsi", "kdj", "bollinger", "turtle"
    ]

    # 策略显示名称映射
    strategy_display_map = {
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
    }

    async def event_generator():
        import asyncio

        # 发送开始消息
        yield "data: " + safe_json_dumps({'type': 'start', 'total': len(all_strategies)}) + "\n\n"
        await asyncio.sleep(0)  # 强制立即发送

        # 获取请求的日期范围
        requested_start = datetime.fromisoformat(request.get_start_date())
        requested_end = datetime.fromisoformat(request.get_end_date())

        # 检查数据有效性
        from quanttool.factors.stock_analyzer import StockAnalyzer
        analyzer = StockAnalyzer(use_realtime_price=True)

        check_symbol = request.symbols[0] if request.symbols else '300750'
        check_df = analyzer.get_stock_data(check_symbol, 60)

        if not check_df.empty:
            df_dates = check_df['timestamp'] if 'timestamp' in check_df.columns else check_df.index
            latest_date = df_dates.max()

            if requested_end > latest_date:
                end_date = latest_date
                days_requested = (requested_end - requested_start).days
                start_date = latest_date - pd.Timedelta(days=days_requested)
            else:
                start_date = requested_start
                end_date = requested_end
        else:
            start_date = requested_start
            end_date = requested_end

        # 获取基准收益
        benchmark_return = 0
        benchmark_annual_return = 0
        try:
            benchmark_df = analyzer.get_stock_data('510300.SH', 365)
            if benchmark_df.empty:
                benchmark_df = analyzer.get_stock_data('SH600000', 365)

            if not benchmark_df.empty:
                if 'trade_date' in benchmark_df.columns:
                    benchmark_df = benchmark_df.set_index('trade_date')
                elif 'timestamp' in benchmark_df.columns:
                    benchmark_df = benchmark_df.set_index('timestamp')
                benchmark_df.index = pd.to_datetime(benchmark_df.index)

                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                benchmark_period = benchmark_df[(benchmark_df.index >= start_dt) & (benchmark_df.index <= end_dt)]

                if len(benchmark_period) >= 2:
                    start_price = benchmark_period.iloc[0]['close']
                    end_price = benchmark_period.iloc[-1]['close']
                    benchmark_return = (end_price - start_price) / start_price

                    days = (end_date - start_date).days
                    if days > 0:
                        benchmark_annual_return = (1 + benchmark_return) ** (365 / days) - 1
        except Exception as e:
            logger.warning(f"获取基准收益失败: {e}")

        # 发送基准收益
        yield "data: " + safe_json_dumps({'type': 'benchmark', 'return': float(benchmark_return), 'annual': float(benchmark_annual_return)}) + "\n\n"
        await asyncio.sleep(0)

        # 创建回测服务
        from quanttool.application.backtest_service import BacktestService
        backtest_service = BacktestService(use_qlib=True, use_realtime_price=True)

        completed = 0
        results = []

        for strategy_name in all_strategies:
            try:
                # 使用 to_thread 将同步调用放到线程池，避免阻塞事件循环
                result = await asyncio.to_thread(backtest_service.run_backtest,
                    strategy_name=strategy_name,
                    strategy_params={},
                    symbols=request.symbols,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe="1d",
                    initial_cash=request.initial_cash,
                    commission_rate=request.commission_rate,
                )

                strategy_return = result.annual_return or 0
                excess_vs_benchmark = strategy_return - benchmark_annual_return

                result_dict = {
                    "strategy": strategy_name,
                    "strategy_display": strategy_display_map.get(strategy_name, strategy_name),
                    "total_return": float(result.total_return or 0),
                    "annual_return": float(result.annual_return or 0),
                    "excess_return": float(excess_vs_benchmark),
                    "benchmark_return": float(benchmark_return),
                    "max_drawdown": float(getattr(result, 'max_drawdown', 0) or 0),
                    "sharpe_ratio": float(getattr(result, 'sharpe_ratio', 0) or 0),
                    "win_rate": float(result.win_rate or 0),
                    "total_trades": int(result.total_trades or 0),
                    "profit_factor": float(getattr(result, 'profit_factor', 0) or 0),
                    "final_capital": float(result.final_capital or request.initial_cash),
                }
                results.append(result_dict)
                error = None
            except Exception as e:
                logger.warning(f"策略 {strategy_name} 回测失败: {e}")
                result_dict = {
                    "strategy": strategy_name,
                    "strategy_display": strategy_display_map.get(strategy_name, strategy_name),
                    "error": str(e),
                    "total_return": 0,
                    "annual_return": 0,
                }
                results.append(result_dict)
                error = str(e)

            completed += 1

            # 发送策略完成消息
            yield "data: " + safe_json_dumps({
                'type': 'strategy_complete',
                'strategy': strategy_name,
                'result': result_dict,
                'completed': completed,
                'total': len(all_strategies)
            }) + "\n\n"
            await asyncio.sleep(0)

        # 按年化收益排序
        results.sort(key=lambda x: x.get('annual_return', 0), reverse=True)

        # 发送完成消息
        yield "data: " + safe_json_dumps({'type': 'done', 'results': results}) + "\n\n"
        await asyncio.sleep(0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ==================== 原有 API ====================

@router.get("/experiments")
async def list_experiments(
    run_type: str = None, status: str = None
) -> List[Dict[str, Any]]:
    """List experiment runs with optional filtering."""
    from ...infrastructure.stores.meta_db_async import get_async_meta_db

    db = get_async_meta_db()
    runs = await db.get_experiment_runs(run_type=run_type, status=status)

    return runs


@router.get("/backtest/runs/{run_id}")
async def get_backtest_result(run_id: str) -> Dict[str, Any]:
    """Get results for a specific backtest run."""
    from ...infrastructure.stores.meta_db_async import get_async_meta_db

    db = get_async_meta_db()
    run = await db.get_experiment_run(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")

    return run
