"""API routes for QuantTool web application."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime, timedelta
from pydantic import BaseModel
import pandas as pd
import json
import threading
import queue
from ..schemas.experiment import ExperimentRunSchema
from quanttool.application.backtest_service import BacktestService
from quanttool.application.factor_service import FactorService
from quanttool.application.data_service import DataService
from ...core.logging import get_logger

logger = get_logger(__name__)


router = APIRouter()


# ==================== Pydantic Models ====================

# ==================== 任务管理 Models ====================

class TaskCreateRequest(BaseModel):
    """任务创建请求"""
    name: str  # qlib_train, qlib_predict, stock_analyze, market_scan
    params: Dict[str, Any] = {}


# ==================== 原 Models ====================

class AnalyzeRequest(BaseModel):
    """股票分析请求"""
    symbol: str
    days: int = 360


class ScanRequest(BaseModel):
    """扫描筛选请求"""
    market: str = "csi300"
    days: int = 360
    top_n: int = 10
    use_trend_score: bool = True
    use_breakout_score: bool = False
    use_momentum_score: bool = False


class EnhancedAnalyzeRequest(BaseModel):
    """增强分析请求"""
    symbol: str
    days: int = 360
    include_chip: bool = True
    include_patterns: bool = True
    include_strategies: bool = True


class BacktestRequest(BaseModel):
    """回测请求

    默认回测时间为最近一年（从今天往前推一年）
    """
    strategy_name: str = "ma_cross"
    symbols: List[str] = []
    start_date: Optional[str] = None  # 默认一年前
    end_date: Optional[str] = None    # 默认今天
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    strategy_params: Dict[str, Any] = {}
    data_provider: str = "incremental_data_fetcher"  # 增量数据提供者（优先使用缓存）

    def get_start_date(self) -> str:
        """获取开始日期，默认为一年前"""
        if self.start_date:
            return self.start_date
        return (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    def get_end_date(self) -> str:
        """获取结束日期，默认为今天"""
        if self.end_date:
            return self.end_date
        return datetime.now().strftime('%Y-%m-%d')


# ==================== 任务管理 API ====================

@router.post("/tasks/create")
async def create_task(request: TaskCreateRequest) -> Dict[str, Any]:
    """
    创建异步任务

    支持的任务类型:
    - qlib_train: Qlib 模型训练
    - qlib_predict: Qlib 模型预测
    - stock_analyze: 股票分析
    - market_scan: 市场扫描

    返回任务 ID，客户端可通过 /tasks/{task_id}/status 查询进度
    """
    try:
        from ..task_handlers import create_task

        task_id = create_task(request.name, request.params)

        return {
            "task_id": task_id,
            "name": request.name,
            "status": "pending",
            "message": f"任务已创建: {task_id}",
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建任务失败: {str(e)}")


@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """获取任务状态"""
    from ..task_manager import get_task_manager

    manager = get_task_manager()
    status = manager.get_task_status(task_id)

    if status is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return status


@router.get("/tasks/{task_id}/result")
async def get_task_result(task_id: str) -> Dict[str, Any]:
    """获取任务结果"""
    from ..task_manager import get_task_manager

    manager = get_task_manager()
    task = manager.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    if task.status.value != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"任务尚未完成，当前状态: {task.status.value}"
        )

    return {
        "task_id": task_id,
        "status": task.status.value,
        "result": task.result,
    }


@router.get("/tasks/{task_id}/logs")
async def get_task_logs(task_id: str) -> Dict[str, Any]:
    """获取任务日志"""
    from ..task_manager import get_task_manager

    manager = get_task_manager()
    task = manager.get_task(task_id)

    if task is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")

    return {
        "task_id": task_id,
        "logs": task.logs,
    }


@router.get("/tasks/{task_id}/stream")
async def stream_task_progress(task_id: str):
    """
    SSE 流式获取任务进度

    客户端可通过 EventSource 连接此端点，实时获取进度更新
    """
    from ..task_manager import get_task_manager, TaskStatus

    def event_generator():
        manager = get_task_manager()
        last_progress = -1

        while True:
            task = manager.get_task(task_id)

            if task is None:
                yield f"event: error\ndata: {{\"error\": \"任务不存在\"}}\n\n"
                break

            # 发送进度更新
            if task.progress.percent != last_progress:
                data = {
                    "status": task.status.value,
                    "progress": task.progress.percent,
                    "message": task.progress.message,
                    "stage": task.progress.stage,
                }
                yield f"event: progress\ndata: {json.dumps(data)}\n\n"
                last_progress = task.progress.percent

            # 任务完成或失败
            if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                final_data = {
                    "status": task.status.value,
                    "result": task.result,
                    "error": task.error,
                }
                yield f"event: complete\ndata: {json.dumps(final_data)}\n\n"
                break

            # 发送最新日志
            if task.logs:
                log_data = {"logs": task.logs[-5:]}  # 最近5条日志
                yield f"event: logs\ndata: {json.dumps(log_data)}\n\n"

            import time
            time.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get("/tasks")
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    列出任务

    Args:
        status: 过滤状态 (pending, running, completed, failed, cancelled)
        limit: 返回数量限制
    """
    from ..task_manager import get_task_manager, TaskStatus

    manager = get_task_manager()
    status_filter = TaskStatus(status) if status else None

    return manager.list_tasks(status=status_filter, limit=limit)


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """取消/删除任务"""
    from ..task_manager import get_task_manager

    manager = get_task_manager()

    if manager.delete_task(task_id):
        return {"task_id": task_id, "message": "任务已删除"}
    else:
        raise HTTPException(status_code=404, detail=f"任务不存在: {task_id}")


# ==================== 便捷任务创建端点 ====================

@router.post("/qlib/train/async")
async def train_qlib_model_async(request: "QlibTrainRequest") -> Dict[str, Any]:
    """
    异步训练 Qlib 模型

    立即返回任务 ID，训练在后台执行
    """
    from ..task_handlers import create_task

    params = request.model_dump()
    task_id = create_task("qlib_train", params)

    return {
        "task_id": task_id,
        "name": "qlib_train",
        "status": "pending",
        "message": f"训练任务已创建: {task_id}，请通过 /tasks/{task_id}/status 查询进度",
    }


@router.post("/qlib/predict/async")
async def predict_qlib_model_async(request: "QlibPredictRequest") -> Dict[str, Any]:
    """
    异步预测

    立即返回任务 ID，预测在后台执行
    """
    from ..task_handlers import create_task

    params = request.model_dump()
    task_id = create_task("qlib_predict", params)

    return {
        "task_id": task_id,
        "name": "qlib_predict",
        "status": "pending",
        "message": f"预测任务已创建: {task_id}，请通过 /tasks/{task_id}/status 查询进度",
    }


@router.post("/analyze/async")
async def analyze_stock_async(request: AnalyzeRequest) -> Dict[str, Any]:
    """异步股票分析"""
    from ..task_handlers import create_task

    params = request.model_dump()
    task_id = create_task("stock_analyze", params)

    return {
        "task_id": task_id,
        "name": "stock_analyze",
        "status": "pending",
        "message": f"分析任务已创建: {task_id}",
    }


@router.post("/scan/async")
async def scan_market_async(request: ScanRequest) -> Dict[str, Any]:
    """异步市场扫描"""
    from ..task_handlers import create_task

    params = request.model_dump()
    task_id = create_task("market_scan", params)

    return {
        "task_id": task_id,
        "name": "market_scan",
        "status": "pending",
        "message": f"扫描任务已创建: {task_id}",
    }


# ==================== CLI 功能映射 ====================

@router.post("/analyze")
async def analyze_stock(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    股票分析 - 对应 CLI: quanttool analysis analyze <symbol>

    返回完整的技术分析报告，包括评分、因子、交易计划等
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        report = analyzer.analyze_stock(request.symbol, request.days)

        return {
            "symbol": request.symbol,
            "days": request.days,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@router.post("/analyze/enhanced")
async def analyze_stock_enhanced(request: EnhancedAnalyzeRequest) -> Dict[str, Any]:
    """
    增强版股票分析 - 对应 CLI: quanttool analysis enhanced <symbol>

    整合筹码分布、K线形态、策略信号
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        report = analyzer.analyze_stock_enhanced(
            request.symbol,
            request.days,
            include_chip=request.include_chip,
            include_talib_patterns=request.include_patterns,
            include_strategies=request.include_strategies
        )

        return {
            "symbol": request.symbol,
            "days": request.days,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"增强分析失败: {str(e)}")


@router.post("/scan")
async def scan_stocks(request: ScanRequest) -> Dict[str, Any]:
    """
    股票扫描筛选 - 对应 CLI: quanttool analysis scan

    扫描市场寻找潜在机会
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.cli.commands.analysis_commands import (
            get_csi300_constituents,
            get_csi1000_constituents,
            analyze_stock_trend_score,
            analyze_stock_breakout_score,
            analyze_stock_momentum_score,
            analyze_stock_score
        )

        # 获取股票列表
        if request.market.lower() == "csi300":
            stock_list = get_csi300_constituents()
        elif request.market.lower() == "csi1000":
            stock_list = get_csi1000_constituents()
        else:
            raise HTTPException(status_code=400, detail=f"不支持的市场: {request.market}")

        analyzer = StockAnalyzer()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)

        results = []
        for stock_info in stock_list:
            symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info

            if request.use_momentum_score:
                result, _ = analyze_stock_momentum_score(
                    stock_info, request.days, analyzer, start_date, end_date
                )
            elif request.use_breakout_score:
                result, _ = analyze_stock_breakout_score(
                    stock_info, request.days, analyzer, start_date, end_date
                )
            elif request.use_trend_score:
                result, _ = analyze_stock_trend_score(
                    stock_info, request.days, analyzer, start_date, end_date
                )
            else:
                result, _ = analyze_stock_score(
                    stock_info, request.days, analyzer, None, None, True, start_date, end_date
                )

            if result:
                results.append(result)

        # 排序
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:request.top_n]

        return {
            "market": request.market,
            "total_stocks": len(stock_list),
            "analyzed_stocks": len(results),
            "top_n": request.top_n,
            "results": top_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"扫描失败: {str(e)}")


@router.get("/scan/markets")
async def list_scan_markets() -> List[Dict[str, str]]:
    """列出可扫描的市场"""
    return [
        {"code": "csi300", "name": "沪深300", "description": "沪深300成分股"},
        {"code": "csi1000", "name": "中证1000", "description": "中证1000成分股"},
    ]


@router.get("/stock/{symbol}/info")
async def get_stock_info(symbol: str) -> Dict[str, Any]:
    """获取股票基本信息和最新数据"""
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        df = analyzer.get_stock_data(symbol, 30)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {symbol}")

        latest = df.iloc[-1]

        return {
            "symbol": symbol,
            "latest_price": float(latest.get('close', 0)),
            "volume": int(latest.get('volume', 0)),
            "high": float(latest.get('high', 0)),
            "low": float(latest.get('low', 0)),
            "date": str(latest.get('trade_date', latest.get('timestamp', ''))),
            "data_days": len(df)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票信息失败: {str(e)}")


@router.get("/stock/{symbol}/kline")
async def get_stock_kline(symbol: str, days: int = 60) -> Dict[str, Any]:
    """
    获取股票 K 线数据（用于图表展示）

    Args:
        symbol: 股票代码
        days: 获取天数，默认60天

    Returns:
        K线数据和指标数据
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        df = analyzer.get_stock_data(symbol, days)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {symbol} 的数据")

        # 计算技术指标
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # 转换为前端可用的格式
        kline_data = []
        for _, row in df_with_indicators.iterrows():
            timestamp = row.get('trade_date', row.get('timestamp', None))
            if timestamp is None:
                continue

            # 处理时间戳格式
            if hasattr(timestamp, 'timestamp'):
                ts = int(timestamp.timestamp())
            elif isinstance(timestamp, str):
                ts = int(datetime.fromisoformat(timestamp.replace('Z', '')).timestamp())
            else:
                continue

            kline_data.append({
                "time": ts,
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
                "volume": int(row.get('volume', 0)),
            })

        # 提取均线数据
        ma_data = {
            "ma5": [],
            "ma10": [],
            "ma20": [],
            "ma60": []
        }

        # 列名映射（数据框中使用 ma_5 格式）
        ma_column_map = {
            "ma5": "ma_5",
            "ma10": "ma_10",
            "ma20": "ma_20",
            "ma60": "ma_60"
        }

        for _, row in df_with_indicators.iterrows():
            timestamp = row.get('trade_date', row.get('timestamp', None))
            if timestamp is None:
                continue

            if hasattr(timestamp, 'timestamp'):
                ts = int(timestamp.timestamp())
            elif isinstance(timestamp, str):
                ts = int(datetime.fromisoformat(timestamp.replace('Z', '')).timestamp())
            else:
                continue

            for ma_key, col_name in ma_column_map.items():
                val = row.get(col_name)
                if val is not None and not pd.isna(val):
                    ma_data[ma_key].append({"time": ts, "value": float(val)})

        # 获取最新价格信息
        latest = df_with_indicators.iloc[-1] if len(df_with_indicators) > 0 else {}
        prev_close = df_with_indicators.iloc[-2]['close'] if len(df_with_indicators) > 1 else latest.get('close', 0)
        current_price = float(latest.get('close', 0))
        change = current_price - float(prev_close) if prev_close else 0
        change_pct = (change / float(prev_close) * 100) if prev_close else 0

        return {
            "symbol": symbol,
            "days": days,
            "kline": kline_data,
            "ma": ma_data,
            "count": len(kline_data),
            # 实时价格信息
            "quote": {
                "price": current_price,
                "open": float(latest.get('open', 0)),
                "high": float(latest.get('high', 0)),
                "low": float(latest.get('low', 0)),
                "volume": int(latest.get('volume', 0)),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "prev_close": round(float(prev_close), 2) if prev_close else 0
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取K线数据失败: {str(e)}")


@router.get("/stock/{symbol}/chip")
async def get_chip_distribution(symbol: str, days: int = 210) -> Dict[str, Any]:
    """
    获取筹码分布数据

    Args:
        symbol: 股票代码
        days: 计算周期，默认210天

    Returns:
        筹码分布数据，用于绘制类似东方财富的筹码图
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.factors.chip_distribution import ChipDistributionCalculator
        import numpy as np

        analyzer = StockAnalyzer()
        # 获取足够多的数据用于筹码计算
        df = analyzer.get_stock_data(symbol, max(days, 365))

        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"数据不足，无法计算筹码分布")

        # 计算筹码分布
        calculator = ChipDistributionCalculator(lookback_days=days)
        result = calculator.calculate(df)

        # 转换为前端可用的格式
        chip_data = []
        for i, (price, chip) in enumerate(zip(result.price_levels, result.chip_distribution)):
            if chip > 0.01:  # 只返回有效数据
                chip_data.append({
                    "price": round(float(price), 2),
                    "chip": round(float(chip), 4)
                })

        # 获取当前价格
        current_price = float(df.iloc[-1]['close'])

        # 计算90%成本区间
        sorted_indices = np.argsort(result.chip_distribution)[::-1]
        cumulative = 0
        cost_low = result.price_levels[0]
        cost_high = result.price_levels[-1]

        for idx in sorted_indices:
            cumulative += result.chip_distribution[idx]
            if cumulative >= 90:
                # 找到90%筹码的价格区间
                involved_indices = sorted_indices[:list(sorted_indices).index(idx)+1]
                cost_low = float(result.price_levels[min(involved_indices)])
                cost_high = float(result.price_levels[max(involved_indices)])
                break

        return {
            "symbol": symbol,
            "days": days,
            "chip_data": chip_data,
            "current_price": current_price,
            "avg_cost": round(result.avg_cost, 2),
            "cost_90_low": round(cost_low, 2),
            "cost_90_high": round(cost_high, 2),
            "metrics": {
                "concentration_ratio": round(result.concentration_ratio, 2),
                "avg_cost": round(result.avg_cost, 2),
                "profit_ratio": round(result.profit_ratio, 2),
                "upper_pressure": round(result.upper_pressure, 2),
                "lower_support": round(result.lower_support, 2),
                "score": round(result.score, 2)
            },
            "levels": {
                "support": [round(x, 2) for x in result.support_levels[:3]],
                "resistance": [round(x, 2) for x in result.resistance_levels[:3]],
                "peaks": [round(x, 2) for x in result.peak_prices[:3]]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"计算筹码分布失败: {str(e)}")


@router.get("/stock/{symbol}/signals")
async def get_technical_signals(symbol: str, days: int = 60) -> Dict[str, Any]:
    """
    获取技术指标异动信号

    包括：MACD金叉/死叉、RSI金叉/超卖、KDJ金叉/超卖、均线排列等

    Args:
        symbol: 股票代码
        days: 分析周期

    Returns:
        技术指标异动信号列表，包含时间戳用于图表标记
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        df = analyzer.get_stock_data(symbol, days)

        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"数据不足")

        # 计算技术指标
        df = analyzer.calculate_technical_indicators(df)

        # 获取最新数据的时间戳
        def get_timestamp(row):
            ts = row.get('trade_date', row.get('timestamp', None))
            if ts is None:
                return None
            if hasattr(ts, 'timestamp'):
                return int(ts.timestamp())
            elif isinstance(ts, str):
                return int(datetime.fromisoformat(ts.replace('Z', '')).timestamp())
            return None

        signals = []
        markers = []  # 用于K线图标记
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        latest_ts = get_timestamp(latest)

        # ============ MACD 信号 ============
        # 列名：macd_dif (DIF), macd_dea (DEA/信号线), macd (柱状图)
        macd_dif = latest.get('macd_dif', latest.get('macd', 0))
        macd_dea = latest.get('macd_dea', latest.get('macd_signal', 0))
        macd_hist = latest.get('macd', latest.get('macd_hist', 0))

        prev_dif = prev.get('macd_dif', prev.get('macd', 0))
        prev_dea = prev.get('macd_dea', prev.get('macd_signal', 0))

        if prev_dif is not None and prev_dea is not None:
            # MACD 金叉：DIF从下向上穿越DEA
            if prev_dif <= prev_dea and macd_dif > macd_dea:
                signals.append({
                    "name": "MACD金叉",
                    "type": "bullish",
                    "description": "DIF上穿DEA，短期趋势转强",
                    "value": f"DIF:{macd_dif:.3f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "belowBar",
                    "color": "#26a69a",
                    "shape": "arrowUp",
                    "text": "MACD金叉"
                })
            # MACD 死叉：DIF从上向下穿越DEA
            elif prev_dif >= prev_dea and macd_dif < macd_dea:
                signals.append({
                    "name": "MACD死叉",
                    "type": "bearish",
                    "description": "DIF下穿DEA，短期趋势转弱",
                    "value": f"DIF:{macd_dif:.3f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "aboveBar",
                    "color": "#ef5350",
                    "shape": "arrowDown",
                    "text": "MACD死叉"
                })

        # MACD 柱状图方向
        if macd_hist is not None:
            prev_hist = prev.get('macd', prev.get('macd_hist', 0))
            if prev_hist is not None:
                if macd_hist > 0 and prev_hist <= 0:
                    signals.append({
                        "name": "MACD转多",
                        "type": "bullish",
                        "description": "MACD柱状图由负转正",
                        "value": f"柱:{macd_hist:.3f}",
                        "time": latest_ts
                    })
                elif macd_hist < 0 and prev_hist >= 0:
                    signals.append({
                        "name": "MACD转空",
                        "type": "bearish",
                        "description": "MACD柱状图由正转负",
                        "value": f"柱:{macd_hist:.3f}",
                        "time": latest_ts
                    })

        # ============ RSI 信号 ============
        rsi = latest.get('rsi_24', latest.get('rsi_14', 50))
        prev_rsi = prev.get('rsi_24', prev.get('rsi_14', 50))

        if rsi is not None:
            # RSI 超卖
            if rsi < 30:
                signals.append({
                    "name": "RSI超卖",
                    "type": "bullish",
                    "description": f"RSI={rsi:.1f}，超卖区可能反弹",
                    "value": f"RSI:{rsi:.1f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "belowBar",
                    "color": "#2196F3",
                    "shape": "circle",
                    "text": "RSI超卖"
                })
            # RSI 超买
            elif rsi > 70:
                signals.append({
                    "name": "RSI超买",
                    "type": "bearish",
                    "description": f"RSI={rsi:.1f}，超买区注意回调",
                    "value": f"RSI:{rsi:.1f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "aboveBar",
                    "color": "#FF9800",
                    "shape": "circle",
                    "text": "RSI超买"
                })

            # RSI 金叉（从超卖区回升）
            if prev_rsi is not None and prev_rsi < 30 and rsi >= 30:
                signals.append({
                    "name": "RSI回升",
                    "type": "bullish",
                    "description": "RSI脱离超卖区",
                    "value": f"{prev_rsi:.1f}→{rsi:.1f}",
                    "time": latest_ts
                })

        # ============ KDJ 信号 ============
        kdj_k = latest.get('kdj_k', 0)
        kdj_d = latest.get('kdj_d', 0)
        kdj_j = latest.get('kdj_j', 0)

        prev_k = prev.get('kdj_k', 0)
        prev_d = prev.get('kdj_d', 0)

        if kdj_k is not None and kdj_d is not None:
            # KDJ 金叉
            if prev_k is not None and prev_d is not None:
                if prev_k <= prev_d and kdj_k > kdj_d:
                    signals.append({
                        "name": "KDJ金叉",
                        "type": "bullish",
                        "description": f"K({kdj_k:.1f})上穿D({kdj_d:.1f})",
                        "value": f"J:{kdj_j:.1f}",
                        "time": latest_ts
                    })
                    markers.append({
                        "time": latest_ts,
                        "position": "belowBar",
                        "color": "#9C27B0",
                        "shape": "arrowUp",
                        "text": "KDJ金叉"
                    })
                elif prev_k >= prev_d and kdj_k < kdj_d:
                    signals.append({
                        "name": "KDJ死叉",
                        "type": "bearish",
                        "description": f"K({kdj_k:.1f})下穿D({kdj_d:.1f})",
                        "value": f"J:{kdj_j:.1f}",
                        "time": latest_ts
                    })
                    markers.append({
                        "time": latest_ts,
                        "position": "aboveBar",
                        "color": "#E91E63",
                        "shape": "arrowDown",
                        "text": "KDJ死叉"
                    })

            # KDJ 超卖
            if kdj_j is not None and kdj_j < 20:
                signals.append({
                    "name": "KDJ超卖",
                    "type": "bullish",
                    "description": f"J值={kdj_j:.1f}，严重超卖",
                    "value": f"J:{kdj_j:.1f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "belowBar",
                    "color": "#9C27B0",
                    "shape": "circle",
                    "text": "KDJ超卖"
                })
            elif kdj_j is not None and kdj_j > 100:
                signals.append({
                    "name": "KDJ超买",
                    "type": "bearish",
                    "description": f"J值={kdj_j:.1f}，严重超买",
                    "value": f"J:{kdj_j:.1f}",
                    "time": latest_ts
                })

        # ============ 均线排列 ============
        ma5 = latest.get('ma_5', 0)
        ma10 = latest.get('ma_10', 0)
        ma20 = latest.get('ma_20', 0)

        if ma5 and ma10 and ma20:
            # 多头排列：MA5 > MA10 > MA20
            if ma5 > ma10 > ma20:
                signals.append({
                    "name": "均线多头",
                    "type": "bullish",
                    "description": "MA5>MA10>MA20，多头排列",
                    "value": f"5:{ma5:.2f}",
                    "time": latest_ts
                })
            # 空头排列：MA5 < MA10 < MA20
            elif ma5 < ma10 < ma20:
                signals.append({
                    "name": "均线空头",
                    "type": "bearish",
                    "description": "MA5<MA10<MA20，空头排列",
                    "value": f"5:{ma5:.2f}",
                    "time": latest_ts
                })

        # ============ 布林带信号 ============
        close = latest.get('close', 0)
        boll_upper = latest.get('boll_upper', 0)
        boll_lower = latest.get('boll_lower', 0)

        if close and boll_upper and boll_lower:
            if close >= boll_upper:
                signals.append({
                    "name": "突破布林上轨",
                    "type": "neutral",
                    "description": "股价突破布林上轨，强势或超买",
                    "value": f"{close:.2f}",
                    "time": latest_ts
                })
            elif close <= boll_lower:
                signals.append({
                    "name": "跌破布林下轨",
                    "type": "bullish",
                    "description": "股价跌破布林下轨，可能反弹",
                    "value": f"{close:.2f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "belowBar",
                    "color": "#4CAF50",
                    "shape": "circle",
                    "text": "布林下轨"
                })

        # ============ DMI 趋势信号 ============
        pdi = latest.get('dmi_pdi', latest.get('pdi', 0))
        mdi = latest.get('dmi_mdi', latest.get('mdi', 0))
        adx = latest.get('dmi_adx', latest.get('adx', 0))

        if pdi is not None and mdi is not None:
            if pdi > mdi and adx and adx > 25:
                signals.append({
                    "name": "DMI多头趋势",
                    "type": "bullish",
                    "description": f"PDI>{mdi:.1f}且ADX={adx:.1f}，多头趋势强",
                    "value": f"PDI:{pdi:.1f}",
                    "time": latest_ts
                })
            elif mdi > pdi and adx and adx > 25:
                signals.append({
                    "name": "DMI空头趋势",
                    "type": "bearish",
                    "description": f"MDI>{pdi:.1f}且ADX={adx:.1f}，空头趋势强",
                    "value": f"MDI:{mdi:.1f}",
                    "time": latest_ts
                })

        return {
            "symbol": symbol,
            "signals": signals,
            "markers": markers,  # K线图标记
            "signal_count": len(signals),
            "bullish_count": len([s for s in signals if s["type"] == "bullish"]),
            "bearish_count": len([s for s in signals if s["type"] == "bearish"]),
            "latest_price": float(latest.get('close', 0)),
            "latest_change": float(latest.get('close', 0) - prev.get('close', 0)) if prev.get('close') else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取信号失败: {str(e)}")


# ==================== 回测 API ====================

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
        }
    ]


# ==================== Qlib ML 模型 API ====================

@router.get("/qlib/models")
async def list_qlib_models() -> List[Dict[str, Any]]:
    """
    列出可用的 Qlib ML 模型

    包括 21 种 Qlib 原生模型：
    - GBDT 系列: LightGBM, XGBoost, CatBoost, DoubleEnsemble
    - PyTorch 序列: LSTM, GRU, ALSTM, Transformer, TCN, Localformer
    - PyTorch 高级: GATs, SFM, TabNet, ADARNN, ADD, HIST, IGMTF, KRNN, TRA, TCTS, Sandwich
    """
    try:
        from quanttool.strategies.qlib import list_available_models
        df = list_available_models()
        models = df.to_dict('records')

        # 添加参数说明
        model_params = {
            'gbdt': {
                'n_estimators': {'type': 'int', 'default': 200, 'description': '树的数量'},
                'max_depth': {'type': 'int', 'default': 6, 'description': '最大深度'},
                'learning_rate': {'type': 'float', 'default': 0.01, 'description': '学习率'},
            },
            'pytorch_sequence': {
                'hidden_size': {'type': 'int', 'default': 64, 'description': '隐藏层大小'},
                'num_layers': {'type': 'int', 'default': 2, 'description': '层数'},
                'dropout': {'type': 'float', 'default': 0.1, 'description': 'Dropout率'},
                'epochs': {'type': 'int', 'default': 100, 'description': '训练轮数'},
                'batch_size': {'type': 'int', 'default': 256, 'description': '批大小'},
            },
            'pytorch_advanced': {
                'hidden_size': {'type': 'int', 'default': 64, 'description': '隐藏层大小'},
                'num_layers': {'type': 'int', 'default': 2, 'description': '层数'},
                'dropout': {'type': 'float', 'default': 0.1, 'description': 'Dropout率'},
                'epochs': {'type': 'int', 'default': 100, 'description': '训练轮数'},
            }
        }

        for model in models:
            category = model.get('category', 'unknown')
            if category in model_params:
                model['params'] = model_params[category]
            else:
                model['params'] = {}

        return models
    except ImportError:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.get("/qlib/saved-models")
async def list_saved_models() -> List[Dict[str, Any]]:
    """
    列出已保存的模型文件

    返回模型文件列表，按修改时间降序排列。
    包含模型元数据（如特征数量、训练参数等）
    """
    import os
    import joblib
    from pathlib import Path

    model_dir = Path("models/qlib")
    if not model_dir.exists():
        return []

    models = []
    for model_file in model_dir.glob("*.pkl"):
        try:
            stat = model_file.stat()
            # 解析模型名称：{model_type}_{id}.pkl
            name_parts = model_file.stem.split('_')
            model_type = name_parts[0] if name_parts else "unknown"
            model_id = name_parts[1] if len(name_parts) > 1 else ""

            # 尝试加载模型元数据
            feature_count = None
            feature_set = None
            train_stocks = None

            try:
                saved_data = joblib.load(model_file)
                if isinstance(saved_data, dict):
                    model = saved_data.get('model')
                    feature_names = saved_data.get('feature_names', [])
                    feature_count = len(feature_names) if feature_names else None
                    # 尝试获取更多信息
                    if hasattr(model, 'feature_names_'):
                        feature_count = len(model.feature_names_)
            except Exception:
                pass  # 无法加载元数据，使用默认值

            # 模型类型显示名称
            display_names = {
                'lgb': 'LightGBM',
                'xgboost': 'XGBoost',
                'catboost': 'CatBoost',
                'lstm': 'LSTM',
                'gru': 'GRU',
                'transformer': 'Transformer',
            }

            models.append({
                "id": model_id,
                "name": model_file.name,
                "path": str(model_file),
                "model_type": model_type,
                "display_name": display_names.get(model_type, model_type.upper()),
                "feature_count": feature_count,
                "size_kb": round(stat.st_size / 1024, 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created_date": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d"),
            })
        except Exception as e:
            logger.warning(f"Failed to read model file {model_file}: {e}")

    # 按修改时间降序排列
    models.sort(key=lambda x: x["modified_at"], reverse=True)
    return models


@router.get("/qlib/saved-models/{model_id}")
async def get_saved_model_detail(model_id: str) -> Dict[str, Any]:
    """
    获取已保存模型的详细信息

    Args:
        model_id: 模型ID或模型文件名
    """
    import joblib
    from pathlib import Path

    model_dir = Path("models/qlib")
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="模型目录不存在")

    # 查找模型文件
    model_file = None
    for f in model_dir.glob("*.pkl"):
        if model_id in f.name:
            model_file = f
            break

    if model_file is None:
        raise HTTPException(status_code=404, detail=f"未找到模型: {model_id}")

    try:
        saved_data = joblib.load(model_file)

        model = saved_data.get('model')
        feature_names = saved_data.get('feature_names', [])

        # 获取模型详情
        detail = {
            "path": str(model_file),
            "name": model_file.name,
            "feature_count": len(feature_names),
            "feature_names": feature_names[:20] if feature_names else [],  # 前20个特征
            "has_model": model is not None,
        }

        # 获取模型参数
        if hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                # 过滤掉复杂的参数
                simple_params = {}
                for k, v in params.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        simple_params[k] = v
                detail["params"] = simple_params
            except Exception:
                pass

        # 文件信息
        stat = model_file.stat()
        detail["size_kb"] = round(stat.st_size / 1024, 2)
        detail["modified_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        return detail

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


class QlibTrainRequest(BaseModel):
    """Qlib 模型训练请求"""
    model_type: str = "lgb"
    # 用户输入的股票代码仅用于预测，训练使用沪深300成分股
    symbols: List[str] = []
    # 数据集划分（按年份）
    # 训练集: 2019-2022 (4年)
    train_start: str = "2019-01-01"
    train_end: str = "2022-12-31"
    # 验证集: 2023-2024 (2年)
    valid_start: str = "2023-01-01"
    valid_end: str = "2024-12-31"
    # 测试/预测集: 2025至今
    test_start: str = "2025-01-01"
    test_end: str = "2026-03-17"  # 当前日期，由前端动态更新
    # 特征配置
    features: List[str] = []  # 空列表表示使用全部可用特征
    use_rich_features: bool = True  # 是否使用 Alpha158 特征工程 (150+ 特征)
    feature_set: str = "Alpha158"  # Alpha158 或 Alpha360
    label: str = "return_5d"  # 预测目标
    # 模型参数
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.01
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 100
    batch_size: int = 256
    # 训练配置
    max_train_stocks: int = 0  # 0表示使用全部沪深300股票
    # 回测参数
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003  # 手续费率 0.03%
    slippage_rate: float = 0.0001   # 滑点率 0.01%


class QlibPredictRequest(BaseModel):
    """Qlib 模型预测请求

    预测日期默认为最近一年
    """
    model_type: str = "lgb"
    model_path: str = ""  # 已训练模型的路径，为空则自动使用最新模型
    symbols: List[str] = []  # 预测的股票代码
    features: List[str] = []  # 空列表表示使用与训练相同的丰富特征
    use_rich_features: bool = True  # 是否使用 Alpha158 特征工程
    feature_set: str = "Alpha158"  # Alpha158 或 Alpha360
    # 预测/回测参数（默认最近一年）
    predict_start_date: Optional[str] = None  # 默认一年前
    predict_end_date: Optional[str] = None    # 默认今天
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001

    def get_predict_start_date(self) -> str:
        """获取预测开始日期，默认为一年前"""
        if self.predict_start_date:
            return self.predict_start_date
        return (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

    def get_predict_end_date(self) -> str:
        """获取预测结束日期，默认为今天"""
        if self.predict_end_date:
            return self.predict_end_date
        return datetime.now().strftime('%Y-%m-%d')


@router.post("/qlib/train")
async def train_qlib_model(request: QlibTrainRequest) -> Dict[str, Any]:
    """
    训练 Qlib ML 模型

    使用沪深300成分股作为训练数据，按年份划分训练/验证/测试集：
    - 训练集: 2020-2023年
    - 验证集: 2024-2025年
    - 测试集: 2026年

    用户输入的股票代码仅用于预测，不参与训练
    """
    try:
        from quanttool.strategies.qlib import create_model
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.cli.commands.analysis_commands import get_csi300_constituents
        import numpy as np
        import os
        import uuid

        # 获取沪深300成分股作为训练数据
        csi300_stocks = get_csi300_constituents()
        train_symbols = [s['code'] if isinstance(s, dict) else s for s in csi300_stocks]

        # 限制训练股票数量
        if request.max_train_stocks > 0:
            train_symbols = train_symbols[:request.max_train_stocks]

        logger.info(f"Training with {len(train_symbols)} CSI300 stocks")

        # 获取训练数据
        analyzer = StockAnalyzer()
        train_data = []
        valid_data = []
        test_data = []

        # 解析日期
        train_start_dt = datetime.fromisoformat(request.train_start)
        train_end_dt = datetime.fromisoformat(request.train_end)
        valid_start_dt = datetime.fromisoformat(request.valid_start)
        valid_end_dt = datetime.fromisoformat(request.valid_end)
        test_start_dt = datetime.fromisoformat(request.test_start)
        test_end_dt = datetime.fromisoformat(request.test_end)

        logger.info(f"Date ranges - Train: {request.train_start} to {request.train_end}, "
                   f"Valid: {request.valid_start} to {request.valid_end}, "
                   f"Test: {request.test_start} to {request.test_end}")

        success_count = 0
        first_symbol_features = None  # 记录第一个成功股票的特征列名，确保所有股票使用相同特征

        # 初始化 Alpha158 特征工程器
        from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
        feature_engineer = QlibFeatureEngineer(feature_set=request.feature_set)

        for symbol in train_symbols:
            try:
                # 获取足够的历史数据（约7年，覆盖2019-2026训练+验证+预测期间）
                df = analyzer.get_stock_data(symbol, 2500)
                if df.empty or len(df) < 120:  # Alpha158 需要至少 120 条数据
                    logger.warning(f"Insufficient data for {symbol}: {len(df) if not df.empty else 0} rows")
                    continue

                # 确定日期列
                date_column = None
                if 'trade_date' in df.columns:
                    date_column = 'trade_date'
                elif 'timestamp' in df.columns:
                    date_column = 'timestamp'

                if not date_column:
                    logger.warning(f"No date column found for {symbol}")
                    continue

                df['_date'] = pd.to_datetime(df[date_column])

                if request.use_rich_features:
                    # 使用 Alpha158 特征工程 (150+ 特征)
                    try:
                        feature_df = feature_engineer.generate_features(df)
                        available_features = list(feature_df.columns)
                        df = pd.concat([df, feature_df], axis=1)
                    except Exception as e:
                        logger.warning(f"Feature engineering failed for {symbol}: {e}")
                        continue
                else:
                    # 计算技术指标
                    df = analyzer.calculate_technical_indicators(df)

                    if request.features:
                        # 使用用户指定的特征
                        available_features = [f for f in request.features if f in df.columns]
                    else:
                        # 使用基本特征
                        available_features = ['close', 'volume', 'ma_5', 'ma_10', 'ma_20']

                if not available_features:
                    logger.warning(f"No available features for {symbol}")
                    continue

                # 确保所有股票使用相同的特征列
                if first_symbol_features is None:
                    first_symbol_features = available_features
                else:
                    # 使用第一个股票的特征列，确保一致性
                    available_features = [f for f in first_symbol_features if f in df.columns]
                    if len(available_features) != len(first_symbol_features):
                        logger.warning(f"Feature mismatch for {symbol}, expected {len(first_symbol_features)}, got {len(available_features)}")
                        continue

                logger.info(f"Using {len(available_features)} features for {symbol}")

                # 计算标签（未来5日收益率）
                df['return_5d'] = df['close'].pct_change(5).shift(-5)

                # 调试：输出数据的日期范围
                data_min_date = df['_date'].min()
                data_max_date = df['_date'].max()
                logger.info(f"{symbol}: data range {data_min_date} to {data_max_date}, {len(df)} rows")

                # 按日期划分数据
                row_count = 0
                for idx, row in df.iterrows():
                    date_val = row['_date']
                    if pd.isna(date_val):
                        continue

                    feature_vals = [row.get(f) for f in available_features]
                    label_val = row.get('return_5d')

                    # 过滤无效值
                    if any(pd.isna(v) for v in feature_vals) or pd.isna(label_val):
                        continue

                    row_data = {
                        'features': feature_vals,
                        'label': label_val,
                        'symbol': symbol,
                        'date': date_val
                    }

                    # 划分数据集
                    if train_start_dt <= date_val <= train_end_dt:
                        train_data.append(row_data)
                        row_count += 1
                    elif valid_start_dt <= date_val <= valid_end_dt:
                        valid_data.append(row_data)
                        row_count += 1
                    elif test_start_dt <= date_val <= test_end_dt:
                        test_data.append(row_data)
                        row_count += 1

                if row_count > 0:
                    success_count += 1

            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
                continue

        logger.info(f"Data collection complete: {success_count} stocks succeeded, "
                   f"train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")

        if not train_data:
            raise HTTPException(
                status_code=400,
                detail=f"无法获取足够的训练数据。收集了 {success_count} 只股票，训练集 {len(train_data)} 条，"
                       f"验证集 {len(valid_data)} 条，测试集 {len(test_data)} 条。请检查日期范围是否在数据覆盖范围内。"
            )

        # 准备训练数据
        feature_cols = available_features
        X_train = np.array([d['features'] for d in train_data])
        y_train = np.array([d['label'] for d in train_data])

        # 创建模型
        config_kwargs = {
            'n_estimators': request.n_estimators,
            'max_depth': request.max_depth,
            'learning_rate': request.learning_rate,
            'hidden_size': request.hidden_size,
            'num_layers': request.num_layers,
            'dropout': request.dropout,
            'epochs': request.epochs,
            'batch_size': request.batch_size,
        }

        model = create_model(request.model_type, **config_kwargs)

        # 训练
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        y_train_series = pd.Series(y_train)
        model.fit(X_train_df, y_train_series)
        model.feature_names_ = feature_cols

        # 保存模型
        model_dir = "models/qlib"
        os.makedirs(model_dir, exist_ok=True)
        model_id = str(uuid.uuid4())[:8]
        model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"
        model.save(model_path)

        # 评估训练集
        train_pred = model.predict(X_train_df)
        train_mse = np.mean((train_pred - y_train) ** 2)
        train_mae = np.mean(np.abs(train_pred - y_train))
        train_ic = np.corrcoef(train_pred, y_train)[0, 1] if len(train_pred) > 1 else 0

        # 评估验证集
        valid_metrics = {}
        if valid_data:
            X_valid = np.array([d['features'] for d in valid_data])
            y_valid = np.array([d['label'] for d in valid_data])
            X_valid_df = pd.DataFrame(X_valid, columns=feature_cols)
            valid_pred = model.predict(X_valid_df)
            valid_metrics = {
                "samples": len(valid_data),
                "mse": round(float(np.mean((valid_pred - y_valid) ** 2)), 6),
                "mae": round(float(np.mean(np.abs(valid_pred - y_valid))), 6),
                "ic": round(float(np.corrcoef(valid_pred, y_valid)[0, 1]), 4) if len(valid_pred) > 1 else 0,
            }

        # 评估测试集
        test_metrics = {}
        if test_data:
            X_test = np.array([d['features'] for d in test_data])
            y_test = np.array([d['label'] for d in test_data])
            X_test_df = pd.DataFrame(X_test, columns=feature_cols)
            test_pred = model.predict(X_test_df)
            test_metrics = {
                "samples": len(test_data),
                "mse": round(float(np.mean((test_pred - y_test) ** 2)), 6),
                "mae": round(float(np.mean(np.abs(test_pred - y_test))), 6),
                "ic": round(float(np.corrcoef(test_pred, y_test)[0, 1]), 4) if len(test_pred) > 1 else 0,
            }

        return {
            "model_id": model_id,
            "model_type": request.model_type,
            "model_path": model_path,
            "train_symbols_count": len(train_symbols),
            "predict_symbols": request.symbols,  # 用户输入的股票代码（仅用于预测）
            "train_samples": len(train_data),
            "features": feature_cols,
            "feature_count": len(feature_cols),
            "use_rich_features": request.use_rich_features,
            "data_split": {
                "train": {
                    "period": f"{request.train_start} ~ {request.train_end}",
                    "samples": len(train_data),
                },
                "valid": {
                    "period": f"{request.valid_start} ~ {request.valid_end}",
                    "samples": len(valid_data),
                },
                "test": {
                    "period": f"{request.test_start} ~ {request.test_end}",
                    "samples": len(test_data),
                },
            },
            "metrics": {
                "train": {
                    "mse": round(float(train_mse), 6),
                    "mae": round(float(train_mae), 6),
                    "ic": round(float(train_ic), 4) if not np.isnan(train_ic) else 0,
                },
                "valid": valid_metrics,
                "test": test_metrics,
            },
            "backtest_params": {
                "initial_cash": request.initial_cash,
                "commission_rate": request.commission_rate,
                "slippage_rate": request.slippage_rate,
                "t_plus_1": True,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")


@router.post("/qlib/train/stream")
async def train_qlib_model_stream(request: QlibTrainRequest):
    """
    使用 SSE 流式推送训练进度

    事件类型:
    - progress: 进度更新
    - log: 日志消息
    - complete: 训练完成
    - error: 错误
    """
    import asyncio

    # 使用同步队列（线程安全）
    message_queue = queue.Queue()

    def send_event(event_type: str, data: Dict[str, Any]):
        """发送SSE事件到队列"""
        message_queue.put({"event": event_type, "data": data})

    def training_worker():
        """后台训练线程"""
        try:
            from quanttool.strategies.qlib import create_model
            from quanttool.factors.stock_analyzer import StockAnalyzer
            from quanttool.cli.commands.analysis_commands import get_csi300_constituents
            import numpy as np
            import os
            import uuid

            # 阶段1: 初始化
            send_event("progress", {
                "stage": "init",
                "progress": 0,
                "message": "初始化训练环境..."
            })

            # 获取沪深300成分股
            csi300_stocks = get_csi300_constituents()
            train_symbols = [s['code'] if isinstance(s, dict) else s for s in csi300_stocks]

            if request.max_train_stocks > 0:
                train_symbols = train_symbols[:request.max_train_stocks]

            total_stocks = len(train_symbols)
            send_event("progress", {
                "stage": "init",
                "progress": 5,
                "message": f"准备获取 {total_stocks} 只沪深300成分股数据"
            })

            # 阶段2: 数据获取
            analyzer = StockAnalyzer()
            train_data = []
            valid_data = []
            test_data = []

            train_start_dt = datetime.fromisoformat(request.train_start)
            train_end_dt = datetime.fromisoformat(request.train_end)
            valid_start_dt = datetime.fromisoformat(request.valid_start)
            valid_end_dt = datetime.fromisoformat(request.valid_end)
            test_start_dt = datetime.fromisoformat(request.test_start)
            test_end_dt = datetime.fromisoformat(request.test_end)

            success_count = 0
            cache_hits = 0
            first_symbol_features = None  # 记录第一个成功股票的特征列名

            # 初始化 Alpha158 特征工程器
            from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
            feature_engineer = QlibFeatureEngineer(feature_set=request.feature_set)

            for i, symbol in enumerate(train_symbols):
                try:
                    send_event("progress", {
                        "stage": "data_collection",
                        "progress": 5 + int((i / total_stocks) * 55),
                        "current": symbol,
                        "processed": i + 1,
                        "total": total_stocks,
                        "cache_hits": cache_hits,
                        "message": f"正在获取数据: {symbol} ({i + 1}/{total_stocks}) [缓存命中: {cache_hits}]"
                    })

                    # 获取数据：优先使用缓存
                    # 计算实际需要的日期范围
                    train_end_date = datetime.fromisoformat(request.train_end)
                    start_date = train_end_date - timedelta(days=2500)  # 约 7 年

                    df = analyzer.get_stock_data(
                        symbol,
                        start_date=start_date,
                        end_date=datetime.now(),
                        force_refresh=False  # 优先使用缓存
                    )

                    # 检测是否命中缓存（数据量大且获取快）
                    if len(df) >= 500:  # 约2年数据，通常来自缓存
                        cache_hits += 1

                    if df.empty or len(df) < 120:  # Alpha158 需要至少 120 条数据
                        continue

                    # 确定日期列
                    date_column = None
                    if 'trade_date' in df.columns:
                        date_column = 'trade_date'
                    elif 'timestamp' in df.columns:
                        date_column = 'timestamp'

                    if not date_column:
                        send_event("log", {"message": f"警告: {symbol} 无日期列"})
                        continue

                    df['_date'] = pd.to_datetime(df[date_column])

                    if request.use_rich_features:
                        # 使用 Alpha158 特征工程 (150+ 特征)
                        try:
                            feature_df = feature_engineer.generate_features(df)
                            available_features = list(feature_df.columns)
                            df = pd.concat([df, feature_df], axis=1)
                        except Exception as e:
                            send_event("log", {"message": f"警告: {symbol} 特征工程失败: {str(e)}"})
                            continue
                    else:
                        df = analyzer.calculate_technical_indicators(df)
                        if request.features:
                            available_features = [f for f in request.features if f in df.columns]
                        else:
                            available_features = ['close', 'volume', 'ma_5', 'ma_10', 'ma_20']

                    if not available_features:
                        send_event("log", {"message": f"警告: {symbol} 无可用特征"})
                        continue

                    # 确保所有股票使用相同的特征列
                    if first_symbol_features is None:
                        first_symbol_features = available_features
                    else:
                        available_features = [f for f in first_symbol_features if f in df.columns]
                        if len(available_features) != len(first_symbol_features):
                            continue

                    # 计算标签：未来5天收益率（连续值，用于回归任务）
                    # Alpha158 特征用于预测连续收益率
                    df['return_5d'] = df['close'].pct_change(5).shift(-5)
                    df['label'] = df['return_5d']  # 连续收益率标签

                    for idx, row in df.iterrows():
                        date_val = row['_date']
                        if pd.isna(date_val):
                            continue

                        feature_vals = [row.get(f) for f in available_features]
                        label_val = row.get('label')  # 连续收益率标签

                        if any(pd.isna(v) for v in feature_vals) or pd.isna(label_val):
                            continue

                        row_data = {
                            'features': feature_vals,
                            'label': label_val,
                            'symbol': symbol,
                            'date': date_val,
                            'return_5d': row.get('return_5d', 0)  # 保留实际收益率用于评估
                        }

                        if train_start_dt <= date_val <= train_end_dt:
                            train_data.append(row_data)
                        elif valid_start_dt <= date_val <= valid_end_dt:
                            valid_data.append(row_data)
                        elif test_start_dt <= date_val <= test_end_dt:
                            test_data.append(row_data)

                    success_count += 1

                except Exception as e:
                    send_event("log", {"message": f"警告: {symbol} 数据获取失败: {str(e)}"})
                    continue

            # 阶段3: 数据准备
            send_event("progress", {
                "stage": "preparation",
                "progress": 65,
                "message": f"数据获取完成，成功 {success_count} 只股票 (缓存命中: {cache_hits})，准备训练数据..."
            })

            if not train_data:
                send_event("error", {
                    "message": f"无法获取足够的训练数据。成功 {success_count} 只股票，训练集 {len(train_data)} 条"
                })
                return

            feature_cols = available_features
            X_train = np.array([d['features'] for d in train_data])
            y_train = np.array([d['label'] for d in train_data])

            send_event("progress", {
                "stage": "preparation",
                "progress": 70,
                "message": f"训练样本: {len(train_data)}, 验证样本: {len(valid_data)}, 测试样本: {len(test_data)}, 特征数: {len(feature_cols)}"
            })

            # 阶段4: 模型训练
            send_event("progress", {
                "stage": "training",
                "progress": 75,
                "message": f"创建 {request.model_type.upper()} 模型..."
            })

            config_kwargs = {
                'n_estimators': request.n_estimators,
                'max_depth': request.max_depth,
                'learning_rate': request.learning_rate,
                'hidden_size': request.hidden_size,
                'num_layers': request.num_layers,
                'dropout': request.dropout,
                'epochs': request.epochs,
                'batch_size': request.batch_size,
            }

            model = create_model(request.model_type, **config_kwargs)

            send_event("progress", {
                "stage": "training",
                "progress": 80,
                "message": "开始模型训练..."
            })

            X_train_df = pd.DataFrame(X_train, columns=feature_cols)
            y_train_series = pd.Series(y_train)

            # 检查数据有效性
            send_event("log", {"message": f"数据形状: X={X_train_df.shape}, y={y_train_series.shape}"})
            if X_train_df.empty or len(X_train_df) == 0:
                send_event("error", {
                    "message": f"训练数据为空，请检查数据获取和特征工程"
                })
                return

            # 处理 NaN 和 Inf 值
            X_train_df = X_train_df.fillna(0).replace([np.inf, -np.inf], 0)
            y_train_series = y_train_series.fillna(0).replace([np.inf, -np.inf], 0)

            # 直接使用 LightGBM sklearn 接口训练
            try:
                import lightgbm as lgb
                lgb_model = lgb.LGBMRegressor(
                    n_estimators=request.n_estimators or 100,
                    max_depth=request.max_depth or 5,
                    learning_rate=request.learning_rate or 0.1,
                    random_state=42,
                    verbose=-1
                )
                lgb_model.fit(X_train_df, y_train_series)
                model = lgb_model
                model.feature_names_ = feature_cols
                send_event("log", {"message": "LightGBM 模型训练成功"})
            except Exception as e:
                send_event("error", {"message": f"模型训练失败: {str(e)}"})
                return

            # 阶段5: 模型评估
            send_event("progress", {
                "stage": "evaluation",
                "progress": 90,
                "message": "评估模型性能..."
            })

            # 保存模型
            import joblib
            model_dir = "models/qlib"
            os.makedirs(model_dir, exist_ok=True)
            model_id = str(uuid.uuid4())[:8]
            model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"
            joblib.dump(model, model_path)

            # 评估
            train_pred = model.predict(X_train_df)
            # 回归指标：MSE, MAE
            train_mse = np.mean((train_pred - y_train) ** 2)
            train_mae = np.mean(np.abs(train_pred - y_train))

            # 正确的 IC 计算：按日期分组计算横截面 IC (Rank IC)
            def calculate_ic(predictions, data_list):
                from scipy.stats import spearmanr
                date_data = {}
                for i, d in enumerate(data_list):
                    date_val = d['date']
                    if date_val not in date_data:
                        date_data[date_val] = {'pred': [], 'return': []}
                    date_data[date_val]['pred'].append(predictions[i])
                    date_data[date_val]['return'].append(d.get('return_5d', 0))

                ics = []
                for date_val, data in date_data.items():
                    preds = np.array(data['pred'])
                    returns = np.array(data['return'])
                    if len(preds) >= 5:
                        if np.std(preds) > 1e-10 and np.std(returns) > 1e-10:
                            try:
                                ic, _ = spearmanr(preds, returns)
                                if not np.isnan(ic):
                                    ics.append(ic)
                            except:
                                pass
                return np.mean(ics) if ics else 0.0

            train_ic = calculate_ic(train_pred, train_data)

            valid_metrics = {}
            if valid_data:
                X_valid = np.array([d['features'] for d in valid_data])
                y_valid = np.array([d['label'] for d in valid_data])
                X_valid_df = pd.DataFrame(X_valid, columns=feature_cols)
                valid_pred = model.predict(X_valid_df)
                valid_mse = np.mean((valid_pred - y_valid) ** 2)
                valid_mae = np.mean(np.abs(valid_pred - y_valid))
                valid_metrics = {
                    "samples": len(valid_data),
                    "mse": round(float(valid_mse), 6),
                    "mae": round(float(valid_mae), 6),
                    "ic": round(float(calculate_ic(valid_pred, valid_data)), 4),
                }

            test_metrics = {}
            if test_data:
                X_test = np.array([d['features'] for d in test_data])
                y_test = np.array([d['label'] for d in test_data])
                X_test_df = pd.DataFrame(X_test, columns=feature_cols)
                test_pred = model.predict(X_test_df)
                test_mse = np.mean((test_pred - y_test) ** 2)
                test_mae = np.mean(np.abs(test_pred - y_test))
                test_metrics = {
                    "samples": len(test_data),
                    "mse": round(float(test_mse), 6),
                    "mae": round(float(test_mae), 6),
                    "ic": round(float(calculate_ic(test_pred, test_data)), 4),
                }

            # 阶段6: 完成
            send_event("progress", {
                "stage": "complete",
                "progress": 100,
                "message": "训练完成！"
            })

            result = {
                "model_id": model_id,
                "model_type": request.model_type,
                "model_path": model_path,
                "train_symbols_count": len(train_symbols),
                "predict_symbols": request.symbols,
                "train_samples": len(train_data),
                "features": feature_cols,
                "data_split": {
                    "train": {"period": f"{request.train_start} ~ {request.train_end}", "samples": len(train_data)},
                    "valid": {"period": f"{request.valid_start} ~ {request.valid_end}", "samples": len(valid_data)},
                    "test": {"period": f"{request.test_start} ~ {request.test_end}", "samples": len(test_data)},
                },
                "metrics": {
                    "train": {
                        "samples": len(train_data),
                        "mse": round(float(train_mse), 6),
                        "mae": round(float(train_mae), 6),
                        "ic": round(float(train_ic), 4) if not np.isnan(train_ic) else 0,
                    },
                    "valid": valid_metrics,
                    "test": test_metrics,
                },
                "backtest_params": {
                    "initial_cash": request.initial_cash,
                    "commission_rate": request.commission_rate,
                    "slippage_rate": request.slippage_rate,
                    "t_plus_1": True,
                },
            }

            send_event("complete", {"result": result})

        except Exception as e:
            import traceback
            traceback.print_exc()
            send_event("error", {"message": f"训练失败: {str(e)}"})

    async def event_stream():
        """生成SSE事件流"""
        loop = asyncio.get_event_loop()

        # 在线程池中运行训练工作器
        thread = threading.Thread(target=training_worker)
        thread.start()

        # 从同步队列读取事件并发送
        while True:
            try:
                # 使用 run_in_executor 非阻塞地获取消息
                msg = await loop.run_in_executor(None, lambda: message_queue.get(timeout=0.1))

                event_type = msg.get("event", "message")
                data = msg.get("data", {})

                # 格式化SSE
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                # 完成或错误时结束
                if event_type in ("complete", "error"):
                    break
            except queue.Empty:
                # 检查线程是否结束
                if not thread.is_alive():
                    break
                # 发送心跳保持连接
                yield "event: heartbeat\ndata: {}\n\n"

        # 等待线程结束
        thread.join()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/qlib/predict")
async def predict_with_qlib_model(request: QlibPredictRequest) -> Dict[str, Any]:
    """
    使用 Qlib ML 模型进行预测并回测

    返回预测结果、信号和回测收益
    """
    try:
        import joblib
        from quanttool.factors.stock_analyzer import StockAnalyzer
        import numpy as np
        from datetime import datetime, timedelta
        from pathlib import Path

        # 查找模型文件
        model_path = request.model_path
        if not model_path:
            # 自动查找对应 model_type 的最新模型
            model_dir = Path("models/qlib")
            if model_dir.exists():
                # 查找匹配的模型文件
                pattern = f"{request.model_type}_*.pkl"
                model_files = list(model_dir.glob(pattern))

                if not model_files:
                    # 尝试其他命名方式
                    all_models = list(model_dir.glob("*.pkl"))
                    if all_models:
                        # 使用最新的模型
                        model_files = sorted(all_models, key=lambda x: x.stat().st_mtime, reverse=True)[:1]
                        logger.info(f"No {request.model_type} model found, using latest: {model_files[0].name}")
                else:
                    # 按修改时间排序，取最新的
                    model_files = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[:1]

                if model_files:
                    model_path = str(model_files[0])
                    logger.info(f"Auto-selected model: {model_path}")

        if not model_path:
            raise HTTPException(
                status_code=400,
                detail=f"未找到已保存的模型。请先训练模型，或检查 models/qlib/ 目录下是否有 {request.model_type}_*.pkl 文件"
            )

        logger.info(f"Loading model from: {model_path}")
        saved_data = joblib.load(model_path)

        # 兼容两种保存格式：直接保存模型 或 保存为字典
        if isinstance(saved_data, dict):
            model = saved_data.get('model')
            feature_names = saved_data.get('feature_names', request.features)
        else:
            # 直接保存的模型对象
            model = saved_data
            feature_names = getattr(model, 'feature_names_', request.features)

        if model is None:
            raise HTTPException(status_code=400, detail="模型文件无效")

        # 获取内部模型进行预测
        inner_model = None
        if hasattr(model, 'model'):
            inner_model = model.model
        elif hasattr(model, 'booster'):
            inner_model = model.booster
        else:
            inner_model = model

        # 获取预测数据
        analyzer = StockAnalyzer()
        predictions = {}

        # 解析回测日期（使用动态默认值：最近一年）
        predict_start = datetime.fromisoformat(request.get_predict_start_date())
        predict_end = datetime.fromisoformat(request.get_predict_end_date())

        # 回测参数
        initial_cash = request.initial_cash
        commission_rate = request.commission_rate
        slippage_rate = request.slippage_rate

        # 回测结果
        backtest_results = {
            "initial_cash": initial_cash,
            "final_capital": initial_cash,
            "total_return": 0.0,
            "annual_return": 0.0,
            "total_trades": 0,
            "win_trades": 0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "trades": [],
        }

        # 初始化 Alpha158 特征工程器
        from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
        feature_engineer = QlibFeatureEngineer(feature_set=request.feature_set)

        for symbol in request.symbols:
            df = analyzer.get_stock_data(symbol, 500)  # 获取更多数据用于回测
            if df.empty or len(df) < 120:
                continue

            # 确定日期列
            date_column = None
            if 'trade_date' in df.columns:
                date_column = 'trade_date'
            elif 'timestamp' in df.columns:
                date_column = 'timestamp'

            if not date_column:
                continue

            df['_date'] = pd.to_datetime(df[date_column])

            # 使用 Alpha158 特征工程
            if request.use_rich_features:
                try:
                    feature_df = feature_engineer.generate_features(df)
                    df = pd.concat([df, feature_df], axis=1)
                except Exception as e:
                    logger.warning(f"Feature engineering failed for {symbol}: {e}")
                    continue
            else:
                df = analyzer.calculate_technical_indicators(df)

            # 使用模型期望的特征
            available_features = [f for f in feature_names if f in df.columns]
            if not available_features:
                continue

            # 记录数据日期范围
            data_start = str(df['_date'].min())[:10]
            data_end = str(df['_date'].max())[:10]

            # ====== 回测逻辑 ======
            cash = initial_cash
            position = 0  # 持仓数量
            trades = []
            total_commission = 0.0
            total_slippage = 0.0

            # T+1 交易：记录买入日期，卖出时检查是否满足 T+1
            buy_date = None
            buy_price = 0.0

            for i in range(len(df) - 5):  # 留出预测窗口
                row = df.iloc[i]

                # 获取交易日期
                trade_date = None
                if date_column and date_column in row:
                    trade_date = row[date_column]
                elif df.index.name:
                    trade_date = df.index[i]

                if trade_date is None:
                    continue

                # 检查是否在回测日期范围内
                try:
                    if hasattr(trade_date, 'to_pydatetime'):
                        trade_dt = trade_date.to_pydatetime()
                    elif isinstance(trade_date, str):
                        trade_dt = datetime.fromisoformat(trade_date[:10])
                    elif hasattr(trade_date, 'strftime'):
                        # pandas Timestamp
                        trade_dt = trade_date.to_pydatetime()
                    else:
                        continue
                except:
                    continue

                if trade_dt < predict_start or trade_dt > predict_end:
                    continue

                # 获取当日特征
                X = df[available_features].iloc[i:i+1].values

                try:
                    pred = inner_model.predict(X)[0]
                except:
                    try:
                        pred = float(inner_model.predict(X.reshape(1, -1))[0])
                    except:
                        continue

                if isinstance(pred, (int, float)):
                    pred_value = float(pred)
                elif hasattr(pred, '__len__'):
                    pred_value = float(pred[0])
                else:
                    pred_value = float(pred)

                # 生成信号 (阈值可调整)
                signal = "hold"
                if pred_value > 0.55:
                    signal = "buy"
                elif pred_value < 0.45:
                    signal = "sell"

                # 获取价格
                close_price = float(row['close'])

                # 执行交易 (考虑 T+1)
                if signal == "buy" and position == 0 and cash > 0:
                    # 买入
                    slippage = close_price * slippage_rate
                    buy_price_actual = close_price + slippage
                    shares = int(cash / buy_price_actual / 100) * 100  # A股一手100股

                    if shares > 0:
                        commission = max(shares * buy_price_actual * commission_rate, 5)  # 最低5元
                        total_cost = shares * buy_price_actual + commission

                        if total_cost <= cash:
                            position = shares
                            cash -= total_cost
                            buy_date = trade_dt
                            buy_price = buy_price_actual
                            total_commission += commission
                            total_slippage += shares * slippage
                            trades.append({
                                "type": "buy",
                                "date": str(trade_date)[:10],
                                "price": round(buy_price_actual, 2),
                                "shares": shares,
                                "commission": round(commission, 2),
                                "slippage": round(shares * slippage, 2)
                            })

                elif signal == "sell" and position > 0:
                    # T+1 检查：卖出日期必须比买入日期晚至少1天
                    if buy_date is None or trade_dt <= buy_date:
                        continue

                    # 卖出
                    slippage = close_price * slippage_rate
                    sell_price_actual = close_price - slippage
                    sell_amount = position * sell_price_actual
                    commission = max(sell_amount * commission_rate, 5)

                    profit = position * (sell_price_actual - buy_price) - commission
                    cash += sell_amount - commission
                    total_commission += commission
                    total_slippage += position * slippage

                    trades.append({
                        "type": "sell",
                        "date": str(trade_date)[:10],
                        "price": round(sell_price_actual, 2),
                        "shares": position,
                        "commission": round(commission, 2),
                        "slippage": round(position * slippage, 2),
                        "profit": round(profit, 2)
                    })

                    if profit > 0:
                        backtest_results["win_trades"] += 1

                    position = 0
                    buy_date = None

            # 计算最终市值
            if len(df) > 0:
                final_price = float(df['close'].iloc[-1])
                final_capital = cash + position * final_price
            else:
                final_capital = cash

            total_return = (final_capital - initial_cash) / initial_cash

            # 计算年化收益
            days = (predict_end - predict_start).days
            annual_return = total_return * 252 / max(days, 1) if days > 0 else 0

            # ====== 计算最大回撤 ======
            max_drawdown = 0.0
            if trades:
                # 重建市值曲线
                equity_curve = [initial_cash]
                peak_equity = initial_cash

                # 简化：根据交易记录估算市值变化
                running_cash = initial_cash
                running_position = 0
                running_buy_price = 0.0

                for trade in trades:
                    if trade['type'] == 'buy':
                        running_cash -= trade['shares'] * trade['price'] + trade['commission'] + trade.get('slippage', 0)
                        running_position = trade['shares']
                        running_buy_price = trade['price']
                    elif trade['type'] == 'sell':
                        running_cash += trade['shares'] * trade['price'] - trade['commission']
                        running_position = 0

                    # 假设当日市值为现金（简化计算）
                    equity = running_cash
                    equity_curve.append(equity)

                    # 计算回撤
                    if equity > peak_equity:
                        peak_equity = equity
                    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

            # ====== 计算夏普比率 ======
            sharpe_ratio = 0.0
            if days > 0 and total_return != 0:
                # 简化：使用年化收益率和假设的波动率
                # 实际应该用每日收益率计算
                # 这里用估算：假设年化波动率约 20%
                assumed_volatility = 0.20
                risk_free_rate = 0.02  # 无风险利率 2%
                if assumed_volatility > 0:
                    sharpe_ratio = (annual_return - risk_free_rate) / assumed_volatility

            # 获取最新预测
            X_latest = df[available_features].iloc[-1:].values
            try:
                pred_latest = inner_model.predict(X_latest)[0]
            except:
                pred_latest = 0.5

            predictions[symbol] = {
                "prediction": round(float(pred_latest), 4),
                "signal": "buy" if float(pred_latest) > 0.55 else ("sell" if float(pred_latest) < 0.45 else "hold"),
                "latest_price": round(float(df['close'].iloc[-1]), 2),
                "data_period": {
                    "start_date": data_start,
                    "end_date": data_end,
                },
                "backtest": {
                    "initial_cash": initial_cash,
                    "final_capital": round(final_capital, 2),
                    "total_return": round(total_return * 100, 2),
                    "annual_return": round(annual_return * 100, 2),
                    "max_drawdown": round(max_drawdown * 100, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "total_trades": len(trades),
                    "win_rate": round(backtest_results["win_trades"] / len(trades) * 100, 1) if trades else 0,
                    "total_commission": round(total_commission, 2),
                    "total_slippage": round(total_slippage, 2),
                    "trades": trades[-10:],  # 最近10笔交易
                }
            }

            backtest_results["total_trades"] += len(trades)

        # 计算汇总统计
        total_final_capital = sum(
            p["backtest"]["final_capital"] for p in predictions.values()
        )
        total_win_trades = sum(
            1 for p in predictions.values()
            for t in p["backtest"].get("trades", [])
            if t.get("type") == "sell" and t.get("profit", 0) > 0
        )
        total_sell_trades = sum(
            1 for p in predictions.values()
            for t in p["backtest"].get("trades", [])
            if t.get("type") == "sell"
        )

        # 汇总回测结果
        summary = {
            "total_return_pct": round((total_final_capital - initial_cash * len(predictions)) / (initial_cash * len(predictions)) * 100, 2) if predictions else 0,
            "total_trades": backtest_results["total_trades"],
            "win_rate": round(total_win_trades / total_sell_trades * 100, 1) if total_sell_trades > 0 else 0,
            "predicted_stocks": len(predictions),
        }

        return {
            "success": True,
            "model_type": request.model_type,
            "model_path": model_path,
            "model_name": Path(model_path).name if model_path else None,
            "feature_count": len(feature_names),
            "predict_period": {
                "start_date": request.predict_start_date,
                "end_date": request.predict_end_date,
                "days": (predict_end - predict_start).days,
            },
            "backtest_params": {
                "initial_cash": f"¥{initial_cash:,.0f}",
                "initial_cash_raw": initial_cash,
                "commission_rate": f"{commission_rate * 100:.4f}%",
                "slippage_rate": f"{slippage_rate * 100:.4f}%",
                "total_cost_rate": f"{(commission_rate + slippage_rate) * 100:.4f}%",
                "t_plus_1": True,
            },
            "summary": summary,
            "predictions": predictions,
            "total_stocks": len(request.symbols),
            "predicted_stocks": len(predictions)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.get("/qlib/models/categories")
async def get_qlib_model_categories() -> List[Dict[str, Any]]:
    """获取 Qlib 模型分类"""
    return [
        {
            "category": "gbdt",
            "display_name": "GBDT 系列",
            "description": "梯度提升决策树，适合表格数据，训练快",
            "models": ["lgb", "lightgbm", "xgboost", "xgb", "catboost", "double_ensemble"],
            "recommended": "lgb"
        },
        {
            "category": "pytorch_sequence",
            "display_name": "PyTorch 序列模型",
            "description": "深度学习序列模型，适合时间序列预测",
            "models": ["lstm", "gru", "alstm", "transformer", "tcn", "localformer"],
            "recommended": "lstm"
        },
        {
            "category": "pytorch_advanced",
            "display_name": "PyTorch 高级模型",
            "description": "前沿深度学习模型，适合复杂市场模式",
            "models": ["gats", "sfm", "tabnet", "adarnn", "add", "hist", "igmtf", "krnn", "tra", "tcts", "sandwich"],
            "recommended": "gats"
        }
    ]


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

        # 初始化回测服务
        backtest_service = BacktestService()

        # 选择数据提供者（优先使用增量数据提供者）
        data_provider = getattr(request, 'data_provider', 'incremental_data_fetcher')

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
            data_provider=data_provider,
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
            "max_drawdown": getattr(result, 'max_drawdown', 0),
            "sharpe_ratio": getattr(result, 'sharpe_ratio', 0),
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "trades": []
        }

        # 添加交易记录（如果有）
        if hasattr(result, 'trades') and result.trades:
            for trade in result.trades[:50]:  # 限制返回数量
                result_dict["trades"].append({
                    "symbol": getattr(trade, 'symbol', ''),
                    "action": getattr(trade, 'action', ''),
                    "price": getattr(trade, 'price', 0),
                    "shares": getattr(trade, 'shares', 0),
                    "timestamp": str(getattr(trade, 'timestamp', ''))
                })

        return result_dict

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"回测失败: {str(e)}")


# ==================== 原有 API ====================

@router.get("/experiments")
async def list_experiments(
    run_type: str = None, status: str = None
) -> List[Dict[str, Any]]:
    """List experiment runs with optional filtering."""
    from ..infrastructure.stores.meta_db import MetaDB

    db = MetaDB()
    runs = db.get_experiment_runs(run_type=run_type, status=status)

    return runs


@router.get("/backtest/runs/{run_id}")
async def get_backtest_result(run_id: str) -> Dict[str, Any]:
    """Get results for a specific backtest run."""
    from ..infrastructure.stores.meta_db import MetaDB

    db = MetaDB()
    run = db.get_experiment_run(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")

    return run


@router.post("/factors/mine")
async def mine_factors(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mine factors across a universe of stocks."""
    try:
        # Extract parameters
        factor_name = request_data.get("factor_name", "momentum")
        symbols = request_data.get("symbols", [])
        start_date_str = request_data.get("start_date", "2023-01-01")
        end_date_str = request_data.get("end_date", "2023-12-31")

        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)

        # Initialize factor service
        factor_service = FactorService()

        # Run factor mining
        results = factor_service.mine_factor(
            factor_name=factor_name,
            factor_params=request_data.get("factor_params", {}),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_provider=request_data.get("data_provider", "tushare"),
        )

        # Convert results to serializable format
        serialized_results = {}
        for symbol, result in results.items():
            serialized_results[symbol] = {
                "factor_name": result.factor_name,
                "ic": result.ic,
                "rank_ic": result.rank_ic,
                "win_rate": result.win_rate,
                "avg_return": result.avg_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
            }

        # Save to metadata DB
        from ..infrastructure.stores.meta_db import MetaDB
        import uuid

        db = MetaDB()
        run_id = str(uuid.uuid4())
        db.save_experiment_run(
            {
                "id": run_id,
                "type": "factor_mining",
                "parameters": request_data,
                "git_commit": "unknown",
                "data_version": "v1.0",
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "status": "completed",
                "results": serialized_results,
                "artifacts": [],
            }
        )

        return {"run_id": run_id, "results": serialized_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error mining factors: {str(e)}")


@router.get("/data/providers")
async def list_data_providers() -> List[str]:
    """List available data providers."""
    from ..core.registry import registry, ComponentType

    providers = registry.list_available(ComponentType.DATA_PROVIDER)
    return providers


@router.get("/strategies")
async def list_strategies() -> List[str]:
    """List available strategies."""
    from ..core.registry import registry, ComponentType

    strategies = registry.list_available(ComponentType.STRATEGY)
    return strategies


@router.get("/factors")
async def list_factors() -> List[str]:
    """List available factors."""
    from ..core.registry import registry, ComponentType

    factors = registry.list_available(ComponentType.FACTOR)
    return factors
