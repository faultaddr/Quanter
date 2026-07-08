"""Streaming backtest comparison API route."""

from datetime import datetime
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ....core.logging import get_logger
from ...schemas.backtest import BacktestRequest


logger = get_logger(__name__)
router = APIRouter()

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
