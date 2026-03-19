"""API routes for QuantTool web application."""

import os
# 解决 OpenMP 库版本冲突问题 (必须在所有 import 之前)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional, Generator
from datetime import datetime, timedelta
from pydantic import BaseModel, field_serializer, field_validator
import pandas as pd
import numpy as np
import json
import threading
import queue
import time
from functools import lru_cache
from ..schemas.experiment import ExperimentRunSchema
from quanttool.application.backtest_service import BacktestService
from quanttool.application.factor_service import FactorService
from quanttool.application.data_service import DataService
from ...core.logging import get_logger

# 导入策略模块以触发注册
import quanttool.strategies

logger = get_logger(__name__)


router = APIRouter()

# ==================== 缓存配置 ====================
# 简单的内存缓存，用于减少重复请求的延迟
_analysis_cache: Dict[str, tuple] = {}  # {cache_key: (data, timestamp)}
_analysis_cache_ttl = 60  # 缓存60秒

def _get_cached_analysis(cache_key: str) -> Optional[Dict]:
    """从缓存获取分析结果"""
    if cache_key in _analysis_cache:
        data, timestamp = _analysis_cache[cache_key]
        if time.time() - timestamp < _analysis_cache_ttl:
            return data
    return None

def _set_cached_analysis(cache_key: str, data: Dict) -> None:
    """设置分析结果缓存"""
    _analysis_cache[cache_key] = (data, time.time())
    # 清理过期缓存
    current_time = time.time()
    expired_keys = [k for k, (_, t) in _analysis_cache.items()
                    if current_time - t > _analysis_cache_ttl * 2]
    for k in expired_keys:
        del _analysis_cache[k]


def to_python_types(obj: Any) -> Any:
    """将 numpy 类型转换为 Python 原生类型"""
    if obj is None:
        return None
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_python_types(item) for item in obj]
    return obj


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
    数据来源：使用 qlib 数据（完整数据集）
    """
    strategy_name: str = "ma_cross"
    symbols: List[str] = []
    start_date: Optional[str] = None  # 默认一年前
    end_date: Optional[str] = None    # 默认今天
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    strategy_params: Dict[str, Any] = {}

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

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
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
    返回完整的数据结构供前端展示
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.factors.chip_distribution import ChipDistributionCalculator

        # 使用实时数据源（优先备用数据源获取实时股价）
        analyzer = StockAnalyzer(use_realtime_price=True)

        # 标准化股票代码并显示
        normalized_symbol = analyzer._normalize_symbol(request.symbol)
        logger.info(f"分析股票: {request.symbol} -> {normalized_symbol}")

        # 获取股票数据（实时股价）
        df = analyzer.get_stock_data(request.symbol, request.days)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"无法获取 {request.symbol} 的数据")

        # 计算技术指标
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # 1. 生成 K 线数据和技术指标
        kline_data = []
        macd_data = []
        kdj_data = []
        rsi_data = []
        volume_data = []
        amount_data = []  # 成交额数据
        ma_data = {"ma5": [], "ma10": [], "ma20": [], "ma60": []}

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

            # K线数据
            kline_data.append({
                "time": ts,
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
            })

            # 成交量
            volume = int(row.get('volume', 0))
            volume_data.append({
                "time": ts,
                "value": volume,
                "color": "#ef4444" if row.get('close', 0) >= row.get('open', 0) else "#10b981"
            })

            # 成交额 = 成交量 * 成交均价 (vwap 或 close)
            vwap = row.get('vwap', row.get('close', 0))
            amount = volume * float(vwap) if vwap else 0
            amount_data.append({
                "time": ts,
                "value": amount,
                "color": "#3b82f6"  # 蓝色
            })

            # 均线数据
            for ma_key, col_name in [("ma5", "ma_5"), ("ma10", "ma_10"), ("ma20", "ma_20"), ("ma60", "ma_60")]:
                val = row.get(col_name)
                if val is not None and not pd.isna(val):
                    ma_data[ma_key].append({"time": ts, "value": float(val)})

            # MACD 数据
            macd_dif = row.get('macd_dif', row.get('macd', None))
            macd_dea = row.get('macd_dea', row.get('macd_signal', None))
            if macd_dif is not None and macd_dea is not None:
                macd_hist = float(macd_dif) - float(macd_dea)
                macd_data.append({
                    "time": ts,
                    "dif": float(macd_dif) if macd_dif else 0,
                    "dea": float(macd_dea) if macd_dea else 0,
                    "hist": macd_hist,
                })

            # KDJ 数据
            k_val = row.get('kdj_k', row.get('k_value', None))
            d_val = row.get('kdj_d', row.get('d_value', None))
            j_val = row.get('kdj_j', row.get('j_value', None))
            if k_val is not None and d_val is not None:
                kdj_data.append({
                    "time": ts,
                    "k": float(k_val) if k_val else 0,
                    "d": float(d_val) if d_val else 0,
                    "j": float(j_val) if j_val else 0,
                })

            # RSI 数据
            rsi_val = row.get('rsi_14', row.get('rsi_12', row.get('rsi_6', None)))
            if rsi_val is not None:
                rsi_data.append({
                    "time": ts,
                    "value": float(rsi_val) if rsi_val else 50,
                })

        # 2. 获取行情数据
        latest = df_with_indicators.iloc[-1] if len(df_with_indicators) > 0 else {}
        prev_close = df_with_indicators.iloc[-2]['close'] if len(df_with_indicators) > 1 else latest.get('close', 0)
        current_price = float(latest.get('close', 0))
        change = current_price - float(prev_close) if prev_close else 0
        change_pct = (change / float(prev_close) * 100) if prev_close else 0

        quote = {
            "symbol": request.symbol,
            "close": current_price,
            "open": float(latest.get('open', 0)),
            "high": float(latest.get('high', 0)),
            "low": float(latest.get('low', 0)),
            "volume": int(latest.get('volume', 0)),
            "change": change,
            "change_pct": change_pct,
            "date": str(latest.get('trade_date', latest.get('timestamp', ''))),
        }

        # 3. 生成信号
        signals = {"bullish": [], "bearish": []}

        # 均线信号
        ma_5 = latest.get('ma_5', 0)
        ma_10 = latest.get('ma_10', 0)
        ma_20 = latest.get('ma_20', 0)
        if ma_5 and ma_10 and ma_20:
            if ma_5 > ma_10 > ma_20:
                signals["bullish"].append({"name": "均线多头排列", "description": f"MA5({ma_5:.2f}) > MA10({ma_10:.2f}) > MA20({ma_20:.2f})"})
            elif ma_5 < ma_10 < ma_20:
                signals["bearish"].append({"name": "均线空头排列", "description": f"MA5({ma_5:.2f}) < MA10({ma_10:.2f}) < MA20({ma_20:.2f})"})
            # 金叉/死叉信号
            prev_ma_5 = df_with_indicators.iloc[-2].get('ma_5', 0)
            prev_ma_10 = df_with_indicators.iloc[-2].get('ma_10', 0)
            if prev_ma_5 and prev_ma_10:
                if prev_ma_5 <= prev_ma_10 and ma_5 > ma_10:
                    signals["bullish"].append({"name": "MA5金叉MA10", "description": f"MA5({ma_5:.2f}) 上穿 MA10({ma_10:.2f})"})
                elif prev_ma_5 >= prev_ma_10 and ma_5 < ma_10:
                    signals["bearish"].append({"name": "MA5死叉MA10", "description": f"MA5({ma_5:.2f}) 下穿 MA10({ma_10:.2f})"})

        # MACD 信号
        macd_dif = latest.get('macd_dif', latest.get('macd', 0))
        macd_dea = latest.get('macd_dea', latest.get('macd_signal', 0))
        macd_hist = macd_dif - macd_dea if macd_dif and macd_dea else 0

        if macd_hist > 0:
            signals["bullish"].append({"name": "MACD多头", "description": f"MACD柱状图为正 ({macd_hist:.4f})"})
        elif macd_hist < 0:
            signals["bearish"].append({"name": "MACD空头", "description": f"MACD柱状图为负 ({macd_hist:.4f})"})

        # RSI 信号
        rsi = latest.get('rsi_14', latest.get('rsi_12', latest.get('rsi_6', 50)))
        if rsi:
            if rsi > 70:
                signals["bearish"].append({"name": "RSI超买", "description": f"RSI = {rsi:.1f}，超买区域"})
            elif rsi < 30:
                signals["bullish"].append({"name": "RSI超卖", "description": f"RSI = {rsi:.1f}，超卖区域"})
            elif rsi > 50:
                signals["bullish"].append({"name": "RSI偏强", "description": f"RSI = {rsi:.1f}，位于强势区域"})

        # 布林带信号
        boll_upper = latest.get('boll_upper', 0)
        boll_lower = latest.get('boll_lower', 0)
        boll_mid = latest.get('boll_mid', latest.get('ma_20', 0))
        if boll_upper and boll_lower:
            if current_price > boll_upper:
                signals["bearish"].append({"name": "突破布林上轨", "description": f"价格 {current_price:.2f} > 上轨 {boll_upper:.2f}"})
            elif current_price < boll_lower:
                signals["bullish"].append({"name": "跌破布林下轨", "description": f"价格 {current_price:.2f} < 下轨 {boll_lower:.2f}"})
            elif current_price > boll_mid:
                signals["bullish"].append({"name": "价格在中轨上方", "description": f"价格 {current_price:.2f} > 中轨 {boll_mid:.2f}"})

        # 4. 计算筹码分布
        chip_data = None
        chip_metrics = None
        if request.include_chip and len(df) >= 60:
            try:
                chip_calc = ChipDistributionCalculator()
                chip_result = chip_calc.calculate(df)
                if chip_result:
                    chip_data = []
                    for price, chip in zip(chip_result.price_levels, chip_result.chip_distribution):
                        if chip > 0.01:
                            chip_data.append({
                                "price": round(float(price), 2),
                                "chip": round(float(chip), 4)
                            })

                    # 计算90%成本区间
                    import numpy as np
                    sorted_indices = np.argsort(chip_result.chip_distribution)[::-1]
                    cumulative = 0
                    cost_90_low = float(chip_result.price_levels[0])
                    cost_90_high = float(chip_result.price_levels[-1])

                    for idx in sorted_indices:
                        cumulative += chip_result.chip_distribution[idx]
                        if cumulative >= 90:
                            involved_indices = sorted_indices[:list(sorted_indices).index(idx)+1]
                            cost_90_low = float(chip_result.price_levels[min(involved_indices)])
                            cost_90_high = float(chip_result.price_levels[max(involved_indices)])
                            break

                    chip_metrics = {
                        "profit_ratio": float(chip_result.profit_ratio) / 100 if chip_result.profit_ratio > 1 else float(chip_result.profit_ratio),
                        "avg_cost": float(chip_result.avg_cost) if hasattr(chip_result, 'avg_cost') else current_price,
                        "concentration": float(chip_result.concentration_ratio) / 100 if hasattr(chip_result, 'concentration_ratio') else 0.5,
                        "score": float(chip_result.score) if hasattr(chip_result, 'score') else 50,
                        "cost_90_low": round(cost_90_low, 2),
                        "cost_90_high": round(cost_90_high, 2),
                        "upper_pressure": float(chip_result.upper_pressure) if hasattr(chip_result, 'upper_pressure') else 0,
                        "lower_support": float(chip_result.lower_support) if hasattr(chip_result, 'lower_support') else 0,
                    }
            except Exception as e:
                logger.warning(f"筹码分布计算失败: {e}")

        # 5. 生成分析报告
        report = analyzer.analyze_stock_enhanced(
            request.symbol,
            request.days,
            include_chip=request.include_chip,
            include_talib_patterns=request.include_patterns,
            include_strategies=request.include_strategies
        )

        return {
            "symbol": request.symbol,
            "normalized_symbol": normalized_symbol,
            "name": "",  # 可以从数据中获取
            "days": request.days,
            "kline": kline_data,
            "volume": volume_data,
            "amount": amount_data,
            "ma": ma_data,
            "macd": macd_data,
            "kdj": kdj_data,
            "rsi": rsi_data,
            "quote": quote,
            "signals": signals,
            "chip": chip_data,
            "chip_metrics": chip_metrics,
            "report": report
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"增强分析失败: {str(e)}")


@router.post("/scan")
@router.post("/scan/market")
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

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)

        # 先并发预加载所有股票数据（显著提升性能）
        print(f"正在预加载 {len(stock_list)} 只股票数据...")
        loaded_count = analyzer.preload_data_for_scan(stock_list, request.days)
        print(f"成功预加载 {loaded_count} 只股票数据")

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
            "results": to_python_types(top_results)
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

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
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
    # 检查缓存
    cache_key = f"kline_{symbol}_{days}"
    cached = _get_cached_analysis(cache_key)
    if cached:
        return cached

    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
        df = analyzer.get_stock_data(symbol, days)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {symbol} 的数据")

        # 计算技术指标
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # 转换为前端可用的格式
        kline_data = []
        volume_data = []
        prev_close = None
        for idx, row in df_with_indicators.iterrows():
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

            close_price = float(row.get('close', 0))
            open_price = float(row.get('open', 0))
            volume = int(row.get('volume', 0))

            kline_data.append({
                "time": ts,
                "open": open_price,
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": close_price,
            })

            # 成交量数据（单独数组，根据涨跌着色）
            if prev_close is None:
                color = '#ef4444' if close_price >= open_price else '#10b981'
            else:
                color = '#ef4444' if close_price >= prev_close else '#10b981'
            volume_data.append({
                "time": ts,
                "value": volume,
                "color": color
            })
            prev_close = close_price

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

        result = {
            "symbol": symbol,
            "days": days,
            "kline": kline_data,
            "volume": volume_data,
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

        # 缓存结果
        _set_cached_analysis(cache_key, result)
        return result
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

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
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
    # 检查缓存
    cache_key = f"signals_{symbol}_{days}"
    cached = _get_cached_analysis(cache_key)
    if cached:
        return cached

    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
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

        result = {
            "symbol": symbol,
            "signals": signals,
            "markers": markers,  # K线图标记
            "signal_count": len(signals),
            "bullish_count": len([s for s in signals if s["type"] == "bullish"]),
            "bearish_count": len([s for s in signals if s["type"] == "bearish"]),
            "latest_price": float(latest.get('close', 0)),
            "latest_change": float(latest.get('close', 0) - prev.get('close', 0)) if prev.get('close') else 0
        }

        # 缓存结果
        _set_cached_analysis(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取信号失败: {str(e)}")


@router.get("/stock/{symbol}/analysis")
async def get_stock_analysis(symbol: str, days: int = 120) -> Dict[str, Any]:
    """
    获取股票完整分析数据（组合接口）

    一次性返回 K线、筹码、信号等所有分析数据
    """
    # 检查缓存
    cache_key = f"analysis_{symbol}_{days}"
    cached = _get_cached_analysis(cache_key)
    if cached:
        return cached

    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        # 使用与 kline 接口相同的数据获取方式
        analyzer = StockAnalyzer(use_realtime_price=True)
        df = analyzer.get_stock_data(symbol, days)

        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"数据不足")

        # 计算技术指标
        df = analyzer.calculate_technical_indicators(df)

        # K线数据
        kline = []
        for _, row in df.iterrows():
            ts = row.get('trade_date', row.get('timestamp', None))
            if ts is not None:
                if hasattr(ts, 'strftime'):
                    date_str = ts.strftime('%Y-%m-%d')
                else:
                    date_str = str(ts)[:10]
            else:
                date_str = ""

            kline.append({
                "date": date_str,
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
                "volume": float(row.get('volume', 0)),
                "amount": float(row.get('amount', 0)),
            })

        # 获取股票名称
        name = symbol
        try:
            info = analyzer.get_stock_info(symbol)
            if info:
                name = info.get('name', symbol)
        except Exception:
            pass

        # 筹码分布（简化版本 - 基于价格区间）
        chip = []
        try:
            import numpy as np
            prices = df['close'].dropna().values
            if len(prices) > 0:
                min_price = prices.min()
                max_price = prices.max()
                price_range = max_price - min_price
                if price_range > 0:
                    # 分成20个价格区间
                    bins = np.linspace(min_price, max_price, 21)
                    hist, _ = np.histogram(prices, bins=bins)
                    total = hist.sum()
                    if total > 0:
                        for i, count in enumerate(hist):
                            chip.append({
                                "price": float((bins[i] + bins[i+1]) / 2),
                                "percent": float(count / total),
                            })
        except Exception as e:
            logger.warning(f"计算筹码分布失败: {e}")

        # 技术指标
        indicators = {
            "macd": {
                "dif": df['macd'].dropna().tolist()[-60:] if 'macd' in df.columns else [],
                "dea": df['macd_signal'].dropna().tolist()[-60:] if 'macd_signal' in df.columns else [],
                "macd": df['macd_hist'].dropna().tolist()[-60:] if 'macd_hist' in df.columns else [],
            },
            "kdj": {
                "k": df['kdj_k'].dropna().tolist()[-60:] if 'kdj_k' in df.columns else [],
                "d": df['kdj_d'].dropna().tolist()[-60:] if 'kdj_d' in df.columns else [],
                "j": df['kdj_j'].dropna().tolist()[-60:] if 'kdj_j' in df.columns else [],
            },
            "rsi": {
                "rsi6": df['rsi_6'].dropna().tolist()[-60:] if 'rsi_6' in df.columns else [],
                "rsi12": df['rsi_12'].dropna().tolist()[-60:] if 'rsi_12' in df.columns else [],
                "rsi24": df['rsi_24'].dropna().tolist()[-60:] if 'rsi_24' in df.columns else [],
            },
            "ma": {
                "ma5": df['ma_5'].dropna().tolist()[-60:] if 'ma_5' in df.columns else [],
                "ma10": df['ma_10'].dropna().tolist()[-60:] if 'ma_10' in df.columns else [],
                "ma20": df['ma_20'].dropna().tolist()[-60:] if 'ma_20' in df.columns else [],
                "ma60": df['ma_60'].dropna().tolist()[-60:] if 'ma_60' in df.columns else [],
            },
        }

        # 简单信号检测
        signals = []
        latest = df.iloc[-1]
        macd = latest.get('macd', 0)
        macd_signal_val = latest.get('macd_signal', 0)
        if macd is not None and macd_signal_val is not None:
            if macd > macd_signal_val and df.iloc[-2].get('macd', 0) <= df.iloc[-2].get('macd_signal', 0):
                signals.append({"name": "MACD金叉", "type": "buy", "description": "MACD金叉买入信号"})
            elif macd < macd_signal_val and df.iloc[-2].get('macd', 0) >= df.iloc[-2].get('macd_signal', 0):
                signals.append({"name": "MACD死叉", "type": "sell", "description": "MACD死叉卖出信号"})

        rsi = latest.get('rsi_14', latest.get('rsi_6', 0))
        if rsi is not None:
            if rsi < 30:
                signals.append({"name": "RSI超卖", "type": "buy", "description": f"RSI={rsi:.1f}，超卖区间"})
            elif rsi > 70:
                signals.append({"name": "RSI超买", "type": "sell", "description": f"RSI={rsi:.1f}，超买区间"})

        result = {
            "symbol": symbol,
            "name": name,
            "kline": kline,
            "chip": chip,
            "signals": signals,
            "indicators": indicators,
            "latest_price": float(latest.get('close', 0)),
            "latest_change_pct": float(latest.get('pct_chg', 0)),
        }

        # 缓存结果
        _set_cached_analysis(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取分析数据失败: {str(e)}")


@router.get("/index/{index_code}/data")
async def get_index_data(index_code: str, days: int = 120) -> List[Dict[str, Any]]:
    """
    获取指数历史数据

    Args:
        index_code: 指数代码 (如 000001=上证指数, 399001=深证成指)
        days: 获取天数
    """
    try:
        from quanttool.infrastructure.data_providers.data_fetcher import DataFetcher

        fetcher = DataFetcher()
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime('%Y%m%d')

        df = fetcher.fetch_index_daily(index_code, start_date=start_date, end_date=end_date)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"无法获取指数 {index_code} 数据")

        # 只取最近 days 天
        df = df.tail(days)

        result = []
        for _, row in df.iterrows():
            ts = row.get('trade_date', row.index if hasattr(row, 'index') else None)
            if ts is not None:
                if hasattr(ts, 'strftime'):
                    date_str = ts.strftime('%Y-%m-%d')
                else:
                    date_str = str(ts)[:10]
            else:
                date_str = ""

            result.append({
                "date": date_str,
                "value": float(row.get('close', 0)),
            })

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get index data: {e}")
        raise HTTPException(status_code=500, detail=f"获取指数数据失败: {str(e)}")


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
async def get_backtest_history(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    获取历史回测记录

    Args:
        symbol: 可选，筛选特定股票的记录
    """
    # 这里返回空列表，实际项目中可以从数据库读取
    # 或者从文件系统读取保存的回测结果
    return []


# ==================== Qlib ML 模型 API ====================

@router.get("/qlib/models")
async def list_qlib_models() -> List[Dict[str, Any]]:
    """
    列出可用的 Qlib ML 模型

    支持 GBDT 系列模型：
    - LightGBM, XGBoost, CatBoost, DoubleEnsemble
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
                'lightgbm': 'LightGBM',
                'xgboost': 'XGBoost',
                'xgb': 'XGBoost',
                'catboost': 'CatBoost',
                'double_ensemble': 'DoubleEnsemble',
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
                "source": "trained",
            })
        except Exception as e:
            logger.warning(f"Failed to read model file {model_file}: {e}")

    # 按修改时间降序排列
    models.sort(key=lambda x: x["modified_at"], reverse=True)
    return models


@router.get("/qlib/pretrained-models")
async def list_pretrained_models() -> List[Dict[str, Any]]:
    """
    列出本地预训练模型

    扫描 qlib_data/cn_data/ 目录下的预训练模型文件 (model_*.pkl)
    返回模型列表，包含模型类型、大小、修改时间等信息
    """
    import joblib
    from pathlib import Path

    models = []

    # 预训练模型目录
    pretrained_dirs = [
        Path("qlib_data/cn_data"),
        Path("models/qlib"),
    ]

    # 模型类型显示名称
    display_names = {
        'lgb': 'LightGBM',
        'lightgbm': 'LightGBM',
        'xgboost': 'XGBoost',
        'xgb': 'XGBoost',
        'catboost': 'CatBoost',
        'lstm': 'LSTM',
        'gru': 'GRU',
        'transformer': 'Transformer',
        'mlp': 'MLP',
        'gbdt': 'GBDT',
    }

    for model_dir in pretrained_dirs:
        if not model_dir.exists():
            continue

        for model_file in model_dir.glob("model_*.pkl"):
            try:
                stat = model_file.stat()

                # 解析模型类型: model_lgb.pkl -> lgb
                model_type = model_file.stem.replace("model_", "").lower()

                # 尝试加载模型元数据
                feature_count = None
                ic_score = None
                train_info = {}

                try:
                    saved_data = joblib.load(model_file)
                    if isinstance(saved_data, dict):
                        model = saved_data.get('model')
                        feature_names = saved_data.get('feature_names', [])
                        feature_count = len(feature_names) if feature_names else None

                        # 尝试获取训练信息
                        if 'config' in saved_data:
                            train_info['config'] = saved_data['config']
                        if 'metrics' in saved_data:
                            train_info['metrics'] = saved_data['metrics']
                            ic_score = saved_data['metrics'].get('ic')

                        # 从模型对象获取特征数量
                        if hasattr(model, 'feature_names_') and feature_count is None:
                            feature_count = len(model.feature_names_)
                except Exception as e:
                    logger.debug(f"Could not load model metadata: {e}")

                models.append({
                    "name": model_file.name,
                    "path": str(model_file),
                    "model_type": model_type,
                    "display_name": display_names.get(model_type, model_type.upper()),
                    "feature_count": feature_count,
                    "ic_score": ic_score,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created_date": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d"),
                    "source": "pretrained" if "qlib_data" in str(model_dir) else "trained",
                    "train_info": train_info,
                })
            except Exception as e:
                logger.warning(f"Failed to read pretrained model {model_file}: {e}")

    # 按修改时间降序排列
    models.sort(key=lambda x: x["modified_at"], reverse=True)
    return models


@router.get("/qlib/all-models")
async def list_all_models() -> Dict[str, Any]:
    """
    列出所有模型（预训练 + 训练后的模型）

    返回分组展示的模型列表
    """
    import joblib
    from pathlib import Path

    # 模型类型显示名称
    display_names = {
        'lgb': 'LightGBM',
        'lightgbm': 'LightGBM',
        'xgboost': 'XGBoost',
        'xgb': 'XGBoost',
        'catboost': 'CatBoost',
        'lstm': 'LSTM',
        'gru': 'GRU',
        'transformer': 'Transformer',
        'mlp': 'MLP',
        'gbdt': 'GBDT',
    }

    def scan_model_dir(model_dir: Path, source: str) -> List[Dict[str, Any]]:
        """扫描模型目录"""
        models = []
        if not model_dir.exists():
            return models

        for model_file in model_dir.glob("*.pkl"):
            try:
                stat = model_file.stat()

                # 解析模型类型
                stem = model_file.stem
                if stem.startswith("model_"):
                    model_type = stem.replace("model_", "").lower()
                else:
                    # 格式: {model_type}_{id}
                    name_parts = stem.split('_')
                    model_type = name_parts[0] if name_parts else "unknown"

                # 尝试加载模型元数据
                feature_count = None
                ic_score = None

                try:
                    saved_data = joblib.load(model_file)
                    if isinstance(saved_data, dict):
                        feature_names = saved_data.get('feature_names', [])
                        feature_count = len(feature_names) if feature_names else None

                        if 'metrics' in saved_data:
                            ic_score = saved_data['metrics'].get('ic')

                        model = saved_data.get('model')
                        if hasattr(model, 'feature_names_') and feature_count is None:
                            feature_count = len(model.feature_names_)
                except Exception:
                    pass

                models.append({
                    "name": model_file.name,
                    "path": str(model_file),
                    "model_type": model_type,
                    "display_name": display_names.get(model_type, model_type.upper()),
                    "feature_count": feature_count,
                    "ic_score": ic_score,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created_date": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d"),
                    "source": source,
                })
            except Exception as e:
                logger.warning(f"Failed to read model file {model_file}: {e}")

        return models

    # 扫描预训练模型目录
    pretrained_models = scan_model_dir(Path("qlib_data/cn_data"), "pretrained")

    # 扫描训练后的模型目录
    trained_models = scan_model_dir(Path("models/qlib"), "trained")

    return {
        "pretrained": sorted(pretrained_models, key=lambda x: x["modified_at"], reverse=True),
        "trained": sorted(trained_models, key=lambda x: x["modified_at"], reverse=True),
        "total_count": len(pretrained_models) + len(trained_models),
    }


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
    early_stopping_rounds: int = 20
    n_head: int = 4  # Transformer attention heads
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


class GBMTrainRequest(BaseModel):
    """GBM 策略训练请求（使用固定优化参数）"""
    # 股票代码（使用沪深300成分股训练，此参数仅用于预测）
    symbols: List[str] = []
    # 训练配置
    max_train_stocks: int = 0  # 0表示使用全部沪深300股票


# 预制的优化参数（基于超参数搜索结果）
GBM_OPTIMAL_PARAMS = {
    "feature_type": "alpha158",
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.2,
    "num_leaves": 210,
    "subsample": 0.8789,
    "colsample_bytree": 0.8879,
    "reg_alpha": 205.6999,  # lambda_l1
    "reg_lambda": 580.9768,  # lambda_l2
    "n_jobs": 20,
    "label_horizon": 10,
    "buy_threshold": 0.50,  # 降低买入阈值，增加交易机会
    "sell_threshold": 0.50,  # 卖出阈值
}


@router.post("/gbm/train")
async def train_gbm_model(request: GBMTrainRequest) -> Dict[str, Any]:
    """
    训练 GBM 策略

    使用 LightGBM (sklearn 接口) 和 Alpha158 特征
    使用预制的优化参数，忽略用户输入

    数据划分（按年份固定）:
    - 训练集: 2017-01-01 ~ 2022-12-31
    - 验证集: 2023-01-01 ~ 2024-06-30
    - 测试集: 2024-07-01 ~ 当前
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        from quanttool.cli.commands.analysis_commands import get_csi300_constituents
        import uuid

        # 获取沪深300成分股作为训练数据
        csi300_stocks = get_csi300_constituents()
        train_symbols = [s['code'] if isinstance(s, dict) else s for s in csi300_stocks]

        # 限制训练股票数量
        if request.max_train_stocks > 0:
            train_symbols = train_symbols[:request.max_train_stocks]

        logger.info(f"GBM 训练: 使用 {len(train_symbols)} 只沪深300成分股")

        # 使用固定优化参数（忽略用户输入）
        config = GBMConfig(**GBM_OPTIMAL_PARAMS)

        # 创建策略
        strategy = GBMStrategy(config)

        # 训练模型
        result = strategy.train(
            instruments=train_symbols,
            start_date="2017-01-01",
            end_date="2026-12-31",
        )

        # 保存模型
        model_dir = "models/gbm"
        os.makedirs(model_dir, exist_ok=True)
        model_id = str(uuid.uuid4())[:8]
        model_path = f"{model_dir}/lgbm_{model_id}.pkl"
        strategy.save_model(model_path)

        return {
            "success": True,
            "model_id": model_id,
            "model_path": model_path,
            "train_samples": to_python_types(result.get("train_samples", 0)),
            "valid_samples": to_python_types(result.get("valid_samples", 0)),
            "test_samples": to_python_types(result.get("test_samples", 0)),
            "feature_count": to_python_types(result.get("feature_count", 0)),
            "train_ic": to_python_types(result.get("train_ic", 0)),
            "valid_ic": to_python_types(result.get("valid_ic", 0)),
            "test_ic": to_python_types(result.get("test_ic", 0)),
            "best_iteration": to_python_types(result.get("best_iteration", 0)),
        }

    except Exception as e:
        logger.error(f"GBM 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")


class GBMPredictRequest(BaseModel):
    """GBM 策略预测请求"""
    model_path: str = ""  # 模型路径，为空则自动使用最新模型
    symbols: List[str] = []  # 预测的股票代码


@router.post("/gbm/predict")
async def predict_gbm_model(request: GBMPredictRequest) -> Dict[str, Any]:
    """
    使用 GBM 策略预测

    返回每只股票的预测收益率和交易信号
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        import glob

        # 查找模型
        model_path = request.model_path
        if not model_path:
            model_files = glob.glob("models/gbm/lgbm_*.pkl")
            if not model_files:
                raise HTTPException(status_code=404, detail="未找到训练好的模型")
            model_path = max(model_files, key=os.path.getmtime)
            logger.info(f"使用最新模型: {model_path}")

        # 加载模型
        config = GBMConfig()
        strategy = GBMStrategy(config)
        strategy.load_model(model_path)

        # 预测
        predictions = []
        for symbol in request.symbols:
            try:
                pred = strategy.predict(symbol)
                predictions.append(to_python_types(pred))
            except Exception as e:
                logger.warning(f"预测失败 [{symbol}]: {e}")
                predictions.append({
                    "instrument": symbol,
                    "error": str(e),
                })

        return {
            "success": True,
            "model_path": model_path,
            "predictions": predictions,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GBM 预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.get("/gbm/models")
async def list_gbm_models() -> List[Dict[str, Any]]:
    """列出所有 GBM 模型"""
    import glob

    model_files = glob.glob("models/gbm/lgbm_*.pkl")

    result = []
    for path in model_files:
        stat = os.stat(path)
        result.append({
            "path": path,
            "filename": os.path.basename(path),
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        })

    return sorted(result, key=lambda x: x["modified"], reverse=True)


@router.delete("/gbm/models/{model_id}")
async def delete_gbm_model(model_id: str) -> Dict[str, Any]:
    """删除指定的 GBM 模型"""
    import glob

    # 查找匹配的模型文件
    model_files = glob.glob(f"models/gbm/*{model_id}*.pkl")

    if not model_files:
        # 也可能是 qrun 模型
        import shutil
        qrun_dirs = glob.glob(f"mlruns/0/*{model_id}*")
        if qrun_dirs:
            for dir_path in qrun_dirs:
                shutil.rmtree(dir_path)
            return {"success": True, "message": f"已删除模型目录: {model_id}"}
        raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")

    deleted = []
    for path in model_files:
        try:
            os.remove(path)
            deleted.append(path)
        except Exception as e:
            logger.warning(f"删除模型文件失败 {path}: {e}")

    return {"success": True, "deleted": deleted}


@router.get("/gbm/train/{task_id}/progress")
async def get_training_progress(task_id: str) -> Dict[str, Any]:
    """获取训练任务进度"""
    # 检查任务状态（从全局任务存储）
    if task_id in _training_tasks:
        task_info = _training_tasks[task_id]
        return {
            "status": task_info.get("status", "unknown"),
            "progress": task_info.get("progress", 0),
            "message": task_info.get("message", ""),
        }

    # 检查 mlruns 是否有对应的结果
    import glob
    result_dirs = glob.glob(f"mlruns/0/{task_id}")
    if result_dirs:
        return {
            "status": "completed",
            "progress": 100,
            "message": "训练已完成",
        }

    return {
        "status": "not_found",
        "progress": 0,
        "message": f"任务 {task_id} 不存在",
    }


# 训练任务存储
_training_tasks: Dict[str, Dict[str, Any]] = {}


@router.get("/gbm/qrun-models")
async def list_qrun_models() -> List[Dict[str, Any]]:
    """
    列出所有 qrun 训练的模型

    返回 mlruns 目录中所有可用的模型信息
    """
    try:
        from quanttool.application.gbm_picker_service import list_all_qrun_models

        models = list_all_qrun_models()

        # 移除不需要返回的字段
        for model in models:
            model.pop('modified_timestamp', None)

        return models

    except Exception as e:
        logger.error(f"获取 qrun 模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


class GBMPicksRequest(BaseModel):
    """GBM 荐股请求"""
    top_n: int = 10  # 返回前 N 只推荐股票
    force_train: bool = False  # 是否强制重新训练
    model_path: Optional[str] = None  # 指定模型路径（qrun 模型）


@router.post("/gbm/picks")
async def get_gbm_picks(request: GBMPicksRequest) -> Dict[str, Any]:
    """
    GBM 模型智能荐股

    使用已训练的 GBM 模型对沪深300成分股进行预测，返回 top N 推荐股票。
    如果没有可用模型，会自动训练一个新模型。

    支持使用 qrun 训练的模型，通过 model_path 参数指定。
    """
    try:
        from quanttool.application.gbm_picker_service import GBMCsi300Picker

        # 创建荐股器
        picker = GBMCsi300Picker(
            top_n=request.top_n,
            model_path=request.model_path
        )

        # 获取推荐
        result = picker.get_daily_picks(force_train=request.force_train)

        # 转换为响应格式 (确保所有数值都是 Python 原生类型)
        top_stocks = []
        for rec in result.top_stocks:
            top_stocks.append({
                "code": rec.code,
                "name": rec.name,
                "pred_return": float(round(rec.pred_return, 4)) if rec.pred_return else 0.0,
                "percentile": float(round(rec.percentile, 4)) if rec.percentile else 0.0,
                "confidence": float(round(rec.confidence, 4)) if rec.confidence else 0.0,
                "probability": float(round(rec.probability, 4)) if rec.probability else 0.0,
                "signal": rec.signal,
                "close": float(round(rec.close, 2)) if rec.close else None,
                "stop_loss": float(round(rec.stop_loss, 2)) if rec.stop_loss else None,
                "take_profit": float(round(rec.take_profit, 2)) if rec.take_profit else None,
            })

        return {
            "success": True,
            "date": result.date,
            "total_stocks": int(result.total_stocks),
            "valid_stocks": int(result.valid_stocks),
            "top_stocks": top_stocks,
            "model_info": result.model_info,
        }

    except Exception as e:
        logger.error(f"GBM 荐股失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"荐股失败: {str(e)}")


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

            # 计算实际需要的日期范围
            train_end_date = datetime.fromisoformat(request.train_end)
            start_date = train_end_date - timedelta(days=2500)  # 约 7 年

            # 先并发预加载所有股票数据（显著提升性能）
            send_event("progress", {
                "stage": "data_preload",
                "progress": 5,
                "message": f"并发预加载 {total_stocks} 只股票数据..."
            })

            loaded_count = analyzer.preload_data_for_scan(train_symbols, days=2500)
            send_event("progress", {
                "stage": "data_preload",
                "progress": 10,
                "message": f"预加载完成，成功获取 {loaded_count} 只股票数据"
            })

            # 使用 qlib 原生训练流程
            send_event("progress", {
                "stage": "qlib_setup",
                "progress": 15,
                "message": "初始化 Qlib 训练环境..."
            })

            try:
                from quanttool.infrastructure.data_providers.qlib_data_converter import (
                    QlibDataConverter,
                    QlibTrainingPipeline,
                    QlibDataConfig
                )

                # 配置 Qlib 数据转换器
                qlib_config = QlibDataConfig(
                    cache_dir=".cache/incremental_data",
                    output_dir="qlib_data/cn_data",
                    feature_type="alpha158" if request.use_rich_features else "alpha360",
                    start_date=request.train_start,
                    end_date=request.train_end,
                )

                converter = QlibDataConverter(qlib_config)
                pipeline = QlibTrainingPipeline(converter)

                send_event("log", {"message": f"使用 Qlib 原生训练流程 (特征: {qlib_config.feature_type})"})

                # 阶段3: 转换数据为 qlib 格式
                send_event("progress", {
                    "stage": "data_conversion",
                    "progress": 20,
                    "message": "转换数据为 Qlib 原生格式..."
                })

                # 转换股票代码格式 (000001.SZ -> 000001_SZ)
                qlib_symbols = [s.replace('.', '_') for s in train_symbols]

                # 创建 qlib DatasetH
                dataset = converter.create_qlib_dataset(
                    symbols=qlib_symbols,
                    start_date=request.train_start,
                    end_date=request.test_end,
                    feature_type=qlib_config.feature_type,
                    label_type="return_10"
                )

                send_event("log", {"message": f"Qlib DatasetH 创建成功"})

            except Exception as e:
                send_event("log", {"message": f"Qlib 原生流程失败，回退到 sklearn: {e}"})
                import traceback
                traceback.print_exc()

                # 回退到传统流程
                for i, symbol in enumerate(train_symbols):
                    try:
                        send_event("progress", {
                            "stage": "data_collection",
                            "progress": 10 + int((i / total_stocks) * 50),
                            "current": symbol,
                            "processed": i + 1,
                            "total": total_stocks,
                            "cache_hits": cache_hits,
                            "message": f"正在处理数据: {symbol} ({i + 1}/{total_stocks})"
                        })

                        df = analyzer.get_stock_data(
                            symbol,
                            start_date=start_date,
                            end_date=datetime.now(),
                            force_refresh=False
                        )

                        if len(df) >= 500:
                            cache_hits += 1

                        if df.empty or len(df) < 120:
                            continue

                        date_column = None
                        if 'trade_date' in df.columns:
                            date_column = 'trade_date'
                        elif 'timestamp' in df.columns:
                            date_column = 'timestamp'

                        if not date_column:
                            continue

                        df['_date'] = pd.to_datetime(df[date_column])

                        if request.use_rich_features:
                            try:
                                feature_df = feature_engineer.generate_features(df)
                                available_features = list(feature_df.columns)
                                df = pd.concat([df, feature_df], axis=1)
                            except Exception as e:
                                continue
                        else:
                            df = analyzer.calculate_technical_indicators(df)
                            if request.features:
                                available_features = [f for f in request.features if f in df.columns]
                            else:
                                available_features = ['close', 'volume', 'ma_5', 'ma_10', 'ma_20']

                        if not available_features:
                            continue

                        if first_symbol_features is None:
                            first_symbol_features = available_features
                        else:
                            available_features = [f for f in first_symbol_features if f in df.columns]
                            if len(available_features) != len(first_symbol_features):
                                continue

                        df['return_5d'] = df['close'].pct_change(5).shift(-5)
                        df['label'] = df['return_5d']

                        for idx, row in df.iterrows():
                            date_val = row['_date']
                            if pd.isna(date_val):
                                continue

                            feature_vals = [row.get(f) for f in available_features]
                            label_val = row.get('label')

                            if any(pd.isna(v) for v in feature_vals) or pd.isna(label_val):
                                continue

                            row_data = {
                                'features': feature_vals,
                                'label': label_val,
                                'symbol': symbol,
                                'date': date_val,
                            }

                            if train_start_dt <= date_val <= train_end_dt:
                                train_data.append(row_data)
                            elif valid_start_dt <= date_val <= valid_end_dt:
                                valid_data.append(row_data)
                            elif test_start_dt <= date_val <= test_end_dt:
                                test_data.append(row_data)

                        success_count += 1

                    except Exception as e:
                        continue

                # 传统流程：准备数据
                if train_data:
                    feature_cols = available_features
                    X_train = np.array([d['features'] for d in train_data])
                    y_train = np.array([d['label'] for d in train_data])
                    dataset = None  # 标记使用传统流程

            # 阶段4: 模型训练
            n_estimators = request.n_estimators or 100

            if dataset is not None:
                # 使用 Qlib 原生训练流程
                send_event("progress", {
                    "stage": "training",
                    "progress": 75,
                    "message": f"使用 Qlib 原生 {request.model_type.upper()} 模型训练..."
                })

                try:
                    # 初始化 Qlib
                    import qlib
                    if not hasattr(qlib, '_initialized') or not qlib._initialized:
                        qlib.init(provider_uri="qlib_data/cn_data")
                        qlib._initialized = True

                    # Qlib 内置模型映射表: (模块名, 类名, 模型类型)
                    QLIB_MODELS = {
                        # GBDT 系列
                        'lgb': ('gbdt', 'LGBModel', 'gbdt'),
                        'lightgbm': ('gbdt', 'LGBModel', 'gbdt'),
                        'xgboost': ('xgboost', 'XGBModel', 'gbdt'),
                        'xgb': ('xgboost', 'XGBModel', 'gbdt'),
                        'catboost': ('catboost_model', 'CatBoostModel', 'gbdt'),
                        'double_ensemble': ('double_ensemble', 'DEnsembleModel', 'gbdt'),
                    }

                    model_type_lower = request.model_type.lower()

                    if model_type_lower not in QLIB_MODELS:
                        supported = ', '.join(sorted(QLIB_MODELS.keys()))
                        raise ValueError(f"不支持的模型类型: {request.model_type}。支持: {supported}")

                    module_name, class_name, model_category = QLIB_MODELS[model_type_lower]
                    ModelClass = getattr(__import__(f'qlib.contrib.model.{module_name}', fromlist=[class_name]), class_name)

                    # 创建 GBDT 模型
                    model = ModelClass(
                        loss='mse',
                        n_estimators=n_estimators,
                        max_depth=request.max_depth or 6,
                        learning_rate=request.learning_rate or 0.01,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                    )

                    send_event("log", {"message": f"创建 Qlib 原生 {request.model_type.upper()} 模型成功"})

                    import time
                    train_start_time = time.time()

                    train_msg = f"Qlib {request.model_type.upper()} 训练中 ({n_estimators} 棵树)..."

                    send_event("progress", {
                        "stage": "training",
                        "progress": 80,
                        "message": train_msg
                    })

                    model.fit(dataset)

                    train_elapsed = time.time() - train_start_time
                    send_event("log", {"message": f"Qlib 原生训练完成 (耗时 {train_elapsed:.1f}s)"})

                    # 获取特征数
                    feature_cols = []
                    try:
                        df_sample = dataset.prepare("train", col_set=["feature"])
                        if isinstance(df_sample, dict):
                            feature_cols = list(df_sample["feature"].columns)
                        else:
                            feature_cols = list(df_sample.xs('feature', axis=1, level=0).columns)
                    except Exception:
                        feature_cols = ["alpha158_features"]

                    # 保存模型到指定目录
                    model_dir = "models/qlib"
                    os.makedirs(model_dir, exist_ok=True)
                    model_id = str(uuid.uuid4())[:8]
                    model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"

                    # 使用 Qlib 的 to_pickle 方法保存模型
                    model.to_pickle(model_path)
                    send_event("log", {"message": f"模型已保存: {model_path}"})

                except Exception as e:
                    send_event("error", {"message": f"Qlib 原生训练失败: {str(e)}"})
                    import traceback
                    traceback.print_exc()
                    return

            else:
                # 传统 sklearn 流程
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

                send_event("progress", {
                    "stage": "training",
                    "progress": 80,
                    "message": f"开始训练 {request.model_type.upper()} 模型 ({n_estimators} 棵树)..."
                })

                try:
                    import time
                    train_start_time = time.time()
                    model.fit(X_train_df, y_train_series)
                    train_elapsed = time.time() - train_start_time

                    model.feature_names_ = feature_cols
                    send_event("log", {"message": f"模型训练完成 (耗时 {train_elapsed:.1f}s)"})
                except Exception as e:
                    send_event("error", {"message": f"模型训练失败: {str(e)}"})
                    return

            # 阶段5: 模型评估
            send_event("progress", {
                "stage": "evaluation",
                "progress": 90,
                "message": "评估模型性能..."
            })

            # 保存模型（如果还没有保存）
            if dataset is None:
                # 传统 sklearn 流程需要在这里保存
                import joblib
                model_dir = "models/qlib"
                os.makedirs(model_dir, exist_ok=True)
                model_id = str(uuid.uuid4())[:8]
                model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"
                joblib.dump(model, model_path)
                send_event("log", {"message": f"模型已保存: {model_path}"})

            # 评估
            if dataset is not None:
                # Qlib 原生模型评估
                try:
                    send_event("log", {"message": f"开始评估，数据集类型: {type(dataset).__name__}"})

                    # 从数据集获取训练数据进行评估
                    send_event("log", {"message": "准备获取训练集数据..."})
                    train_df = dataset.prepare("train", col_set=["feature", "label"])
                    send_event("log", {"message": f"训练集数据获取完成，类型: {type(train_df).__name__}"})
                    if isinstance(train_df, dict):
                        send_event("log", {"message": f"训练集是 dict，keys: {list(train_df.keys())}"})
                        X_train_eval = train_df["feature"]
                        y_train_eval = train_df["label"].values.ravel()
                    else:
                        send_event("log", {"message": f"训练集是 DataFrame，shape: {train_df.shape}"})
                        X_train_eval = train_df.xs('feature', axis=1, level=0)
                        y_train_eval = train_df.xs('label', axis=1, level=0).values.ravel()

                    send_event("log", {"message": f"训练集特征形状: {X_train_eval.shape}, 标签形状: {y_train_eval.shape}"})

                    # 使用 Qlib 模型预测
                    send_event("log", {"message": "开始训练集预测..."})
                    if hasattr(model, 'model') and model.model is not None:
                        train_pred = model.model.predict(X_train_eval.values)
                    else:
                        train_pred = model.predict(X_train_eval)
                    send_event("log", {"message": f"训练集预测完成，预测形状: {train_pred.shape}"} )

                    train_pred = train_pred.ravel() if len(train_pred.shape) > 1 else train_pred

                    train_mse = np.mean((train_pred - y_train_eval) ** 2)
                    train_mae = np.mean(np.abs(train_pred - y_train_eval))
                    train_ic = np.corrcoef(train_pred, y_train_eval)[0, 1] if len(train_pred) > 1 else 0

                    send_event("log", {"message": f"训练集评估: MSE={train_mse:.6f}, MAE={train_mae:.6f}, IC={train_ic:.4f}"})

                    valid_metrics = {}
                    test_metrics = {}

                    # 验证集评估
                    send_event("log", {"message": "开始验证集评估..."})
                    try:
                        valid_df = dataset.prepare("valid", col_set=["feature", "label"])
                        send_event("log", {"message": f"验证集数据获取完成，类型: {type(valid_df).__name__}"})
                        if isinstance(valid_df, dict):
                            X_valid_eval = valid_df["feature"]
                            y_valid_eval = valid_df["label"].values.ravel()
                        else:
                            X_valid_eval = valid_df.xs('feature', axis=1, level=0)
                            y_valid_eval = valid_df.xs('label', axis=1, level=0).values.ravel()

                        if hasattr(model, 'model') and model.model is not None:
                            valid_pred = model.model.predict(X_valid_eval.values)
                        else:
                            valid_pred = model.predict(X_valid_eval)

                        valid_pred = valid_pred.ravel() if len(valid_pred.shape) > 1 else valid_pred
                        valid_mse = np.mean((valid_pred - y_valid_eval) ** 2)
                        valid_mae = np.mean(np.abs(valid_pred - y_valid_eval))
                        valid_ic = np.corrcoef(valid_pred, y_valid_eval)[0, 1] if len(valid_pred) > 1 else 0

                        valid_metrics = {
                            "samples": len(y_valid_eval),
                            "mse": round(float(valid_mse), 6),
                            "mae": round(float(valid_mae), 6),
                            "ic": round(float(valid_ic), 4),
                        }
                    except Exception as ve:
                        send_event("log", {"message": f"验证集评估失败: {ve}"})
                        pass

                    # 测试集评估
                    send_event("log", {"message": "开始测试集评估..."})
                    try:
                        test_df = dataset.prepare("test", col_set=["feature", "label"])
                        send_event("log", {"message": f"测试集数据获取完成，类型: {type(test_df).__name__}"})
                        if isinstance(test_df, dict):
                            X_test_eval = test_df["feature"]
                            y_test_eval = test_df["label"].values.ravel()
                        else:
                            X_test_eval = test_df.xs('feature', axis=1, level=0)
                            y_test_eval = test_df.xs('label', axis=1, level=0).values.ravel()

                        if hasattr(model, 'model') and model.model is not None:
                            test_pred = model.model.predict(X_test_eval.values)
                        else:
                            test_pred = model.predict(X_test_eval)

                        test_pred = test_pred.ravel() if len(test_pred.shape) > 1 else test_pred
                        test_mse = np.mean((test_pred - y_test_eval) ** 2)
                        test_mae = np.mean(np.abs(test_pred - y_test_eval))
                        test_ic = np.corrcoef(test_pred, y_test_eval)[0, 1] if len(test_pred) > 1 else 0

                        test_metrics = {
                            "samples": len(y_test_eval),
                            "mse": round(float(test_mse), 6),
                            "mae": round(float(test_mae), 6),
                            "ic": round(float(test_ic), 4),
                        }
                    except Exception as te:
                        send_event("log", {"message": f"测试集评估失败: {te}"})
                        pass

                    send_event("log", {"message": "模型评估完成"})

                except Exception as e:
                    send_event("log", {"message": f"评估警告: {e}"})
                    train_mse = 0
                    train_mae = 0
                    train_ic = 0
                    valid_metrics = {}
                    test_metrics = {}

            else:
                # 传统 sklearn 流程评估
                train_pred = model.predict(X_train_df)
                train_mse = np.mean((train_pred - y_train) ** 2)
                train_mae = np.mean(np.abs(train_pred - y_train))

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

            # 统计样本数
            train_samples = len(train_data) if train_data else 0
            valid_samples = len(valid_data) if valid_data else 0
            test_samples = len(test_data) if test_data else 0

            # 如果使用 Qlib 原生流程，从 dataset 获取样本数
            if dataset is not None:
                try:
                    train_df = dataset.prepare("train", col_set=["feature"])
                    train_samples = len(train_df) if hasattr(train_df, '__len__') else 0
                    valid_df = dataset.prepare("valid", col_set=["feature"])
                    valid_samples = len(valid_df) if hasattr(valid_df, '__len__') else 0
                    test_df = dataset.prepare("test", col_set=["feature"])
                    test_samples = len(test_df) if hasattr(test_df, '__len__') else 0
                except Exception:
                    pass

            result = {
                "model_id": model_id,
                "model_type": request.model_type,
                "model_path": model_path,
                "train_symbols_count": len(train_symbols),
                "predict_symbols": request.symbols,
                "train_samples": train_samples,
                "features": list(feature_cols) if feature_cols else [],
                "data_split": {
                    "train": {"period": f"{request.train_start} ~ {request.train_end}", "samples": train_samples},
                    "valid": {"period": f"{request.valid_start} ~ {request.valid_end}", "samples": valid_samples},
                    "test": {"period": f"{request.test_start} ~ {request.test_end}", "samples": test_samples},
                },
                "metrics": {
                    "train": {
                        "samples": train_samples,
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

        # 获取预测数据（使用实时价格数据，避免 qlib 复权价格显示异常）
        analyzer = StockAnalyzer(use_realtime_price=True)
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

                # 生成信号 (回归模型预测收益率，阈值需要适配)
                # 回归值范围通常在 -0.1 到 0.1 之间
                signal = "hold"
                if pred_value > 0.005:  # 预测上涨 > 0.5%
                    signal = "buy"
                elif pred_value < -0.005:  # 预测下跌 > 0.5%
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
    # 所有可用策略
    all_strategies = [
        "score", "ma_cross", "dual_ma",
        "breakout", "macd", "rsi", "kdj", "bollinger", "turtle", "gbm"
    ]

    results = []
    start_date = datetime.fromisoformat(request.get_start_date())
    end_date = datetime.fromisoformat(request.get_end_date())

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

    for strategy_name in all_strategies:
        try:
            # 使用实时价格数据，避免复权价格显示异常
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
        from ...infrastructure.stores.meta_db_async import get_async_meta_db
        import uuid

        db = get_async_meta_db()
        run_id = str(uuid.uuid4())
        await db.save_experiment_run(
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
    from ...core.registry import registry, ComponentType

    providers = registry.list_available(ComponentType.DATA_PROVIDER)
    return providers


@router.get("/strategies")
async def list_strategies() -> List[str]:
    """List available strategies."""
    from ...core.registry import registry, ComponentType

    strategies = registry.list_available(ComponentType.STRATEGY)
    return strategies


@router.get("/factors")
async def list_factors() -> List[str]:
    """List available factors."""
    from ...core.registry import registry, ComponentType

    factors = registry.list_available(ComponentType.FACTOR)
    return factors


# ==================== 实时数据 API ====================

# 监控服务管理器（全局状态）
_monitor_services: Dict[str, Any] = {}

# 熔断器状态：记录失败的服务和失败时间
_circuit_breaker: Dict[str, float] = {}
_CIRCUIT_BREAKER_TIMEOUT = 300  # 5 分钟熔断时间


def _is_circuit_open(service_name: str) -> bool:
    """检查熔断器是否打开（服务是否应该被跳过）"""
    if service_name in _circuit_breaker:
        failure_time = _circuit_breaker[service_name]
        if time.time() - failure_time < _CIRCUIT_BREAKER_TIMEOUT:
            return True
        else:
            # 熔断时间已过，重置
            del _circuit_breaker[service_name]
    return False


def _record_failure(service_name: str):
    """记录服务失败"""
    _circuit_breaker[service_name] = time.time()
    logger.warning(f"Circuit breaker opened for {service_name}")


def get_minute_provider():
    """获取 AkShare 分钟数据提供者（延迟初始化）- 保留用于向后兼容"""
    from ...infrastructure.data_providers.akshare_minute_provider import AkShareMinuteProvider

    # 检查熔断器
    if _is_circuit_open("akshare_minute"):
        raise RuntimeError("AkShare minute provider is circuit-broken")

    if not hasattr(get_minute_provider, "_instance"):
        try:
            get_minute_provider._instance = AkShareMinuteProvider()
            get_minute_provider._instance.initialize()
        except Exception as e:
            _record_failure("akshare_minute")
            raise
    return get_minute_provider._instance


def get_realtime_provider():
    """获取统一实时数据提供者（延迟初始化）- 推荐"""
    from ...infrastructure.data_providers.realtime_data_provider import get_realtime_provider as _get_provider
    return _get_provider()


def get_incremental_minute_provider():
    """获取增量分钟数据提供者（延迟初始化）- 推荐"""
    from ...infrastructure.data_providers.incremental_minute_provider import get_incremental_minute_provider as _get_provider
    return _get_provider()


class RealtimeQuoteResponse(BaseModel):
    """实时行情响应"""
    symbol: str
    name: str = ""
    price: float = 0
    open: float = 0
    high: float = 0
    low: float = 0
    volume: float = 0
    amount: float = 0
    pct_change: float = 0
    change: float = 0
    turnover: float = 0
    timestamp: str = ""

    @field_validator('price', 'open', 'high', 'low', 'volume', 'amount', 'pct_change', 'change', 'turnover', mode='before')
    @classmethod
    def convert_numpy_types(cls, v):
        """将 numpy 类型转换为 Python 原生类型"""
        if isinstance(v, (np.integer, np.floating)):
            return float(v)
        return v


class MonitorStartRequest(BaseModel):
    """启动监控请求"""
    symbols: List[str]
    strategy: str = "breakout"
    interval_minutes: int = 5
    buy_threshold: int = 50
    sell_threshold: int = 40
    history_days: int = 120


class MonitorStatusResponse(BaseModel):
    """监控状态响应"""
    running: bool
    symbols: List[str] = []
    strategy: str = ""
    interval_minutes: int = 5
    check_count: int = 0
    signal_count: int = 0
    last_check: Optional[str] = None


@router.get("/realtime/quote/{symbol}")
async def get_realtime_quote(symbol: str) -> Dict[str, Any]:
    """获取实时行情（使用新的实时数据通路）"""
    try:
        # 使用新的统一实时数据提供者
        provider = get_realtime_provider()
        quote = provider.get_realtime_quote(symbol)

        if not quote:
            raise HTTPException(status_code=404, detail=f"无法获取 {symbol} 的实时行情")

        # 转换为字典
        quote_dict = quote.to_dict()

        # 处理 timestamp
        ts = quote_dict.get("timestamp")
        if ts:
            if isinstance(ts, datetime):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)
        else:
            ts_str = ""

        return {
            "symbol": quote_dict.get("symbol", symbol),
            "name": quote_dict.get("name", ""),
            "price": float(quote_dict.get("price", 0) or 0),
            "open": float(quote_dict.get("open", 0) or 0),
            "high": float(quote_dict.get("high", 0) or 0),
            "low": float(quote_dict.get("low", 0) or 0),
            "volume": float(quote_dict.get("volume", 0) or 0),
            "amount": float(quote_dict.get("amount", 0) or 0),
            "pct_change": float(quote_dict.get("change_pct", 0) or 0),
            "change": float(quote_dict.get("change_amount", 0) or 0),
            "turnover": float(quote_dict.get("turnover_rate", 0) or 0),
            "timestamp": ts_str,
            "source": quote_dict.get("source", ""),
            "bid_prices": quote_dict.get("bid_prices", []),
            "ask_prices": quote_dict.get("ask_prices", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get realtime quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"获取实时行情失败: {str(e)}")


@router.post("/realtime/batch")
async def get_realtime_quotes_batch(request: Request) -> List[Dict[str, Any]]:
    """批量获取实时行情"""
    try:
        body = await request.json()
        symbols = body.get("symbols", [])

        if not symbols:
            return []

        provider = get_realtime_provider()
        results = []

        for symbol in symbols:
            try:
                quote = provider.get_realtime_quote(symbol)
                if quote:
                    quote_dict = quote.to_dict()
                    ts = quote_dict.get("timestamp")
                    if ts:
                        if isinstance(ts, datetime):
                            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            ts_str = str(ts)
                    else:
                        ts_str = ""

                    # 计算涨跌幅（如果没有提供）
                    price = float(quote_dict.get("price", 0) or 0)
                    pre_close = float(quote_dict.get("pre_close", 0) or 0)
                    change_pct = float(quote_dict.get("change_pct", 0) or 0)
                    change_amount = float(quote_dict.get("change_amount", 0) or 0)

                    # 如果涨跌幅为0但有昨收价，则计算
                    if change_pct == 0 and pre_close > 0 and price > 0:
                        change_amount = price - pre_close
                        change_pct = change_amount / pre_close

                    results.append({
                        "symbol": quote_dict.get("symbol", symbol),
                        "name": quote_dict.get("name", ""),
                        "price": price,
                        "open": float(quote_dict.get("open", 0) or 0),
                        "high": float(quote_dict.get("high", 0) or 0),
                        "low": float(quote_dict.get("low", 0) or 0),
                        "volume": float(quote_dict.get("volume", 0) or 0),
                        "amount": float(quote_dict.get("amount", 0) or 0),
                        "pct_change": change_pct,
                        "change": change_amount,
                        "turnover": float(quote_dict.get("turnover_rate", 0) or 0),
                        "timestamp": ts_str,
                        "source": quote_dict.get("source", ""),
                    })
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
                # 添加一个空记录表示失败
                results.append({
                    "symbol": symbol,
                    "name": "",
                    "price": 0,
                    "open": 0,
                    "high": 0,
                    "low": 0,
                    "volume": 0,
                    "amount": 0,
                    "pct_change": 0,
                    "change": 0,
                    "turnover": 0,
                    "timestamp": "",
                    "source": "",
                    "error": str(e),
                })

        return results
    except Exception as e:
        logger.error(f"Failed to get batch realtime quotes: {e}")
        raise HTTPException(status_code=500, detail=f"批量获取行情失败: {str(e)}")


@router.get("/realtime/kline/{symbol}")
async def get_realtime_kline(
    symbol: str,
    timeframe: str = "5m",
    count: int = 60
) -> Dict[str, Any]:
    """获取分钟K线数据（使用增量分钟数据通路）"""
    try:
        # 使用新的增量分钟数据提供者
        provider = get_incremental_minute_provider()
        df = provider.get_minute_bars(symbol, timeframe, count=count)

        if df.empty:
            # 回退到旧的 AkShare 提供者
            old_provider = get_minute_provider()
            df = old_provider.get_latest_bars(symbol, count, timeframe)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"无法获取 {symbol} 的K线数据")

        # 转换为前端友好的格式
        kline_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": []
        }

        for _, row in df.iterrows():
            bar = {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row.get("timestamp")) else "",
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
                "amount": float(row.get("amount", 0))
            }
            kline_data["bars"].append(bar)

        return kline_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get kline for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"获取K线数据失败: {str(e)}")


@router.get("/realtime/search")
async def search_stocks(query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """搜索股票"""
    if not query:
        return []

    # 检查缓存
    cache_key = f"search_{query}_{limit}"
    cached = _get_cached_analysis(cache_key)
    if cached:
        logger.info(f"Search cache hit for {cache_key}")
        return cached

    logger.info(f"Search cache miss for {cache_key}")

    try:
        provider = get_minute_provider()
        results = provider.search_symbols(query)

        # 格式化结果
        formatted = []
        for item in results[:limit]:
            formatted.append({
                "symbol": item.get("symbol", ""),
                "name": item.get("name", ""),
                "price": float(item.get("price", 0))
            })

        # 缓存结果
        _set_cached_analysis(cache_key, formatted)
        return formatted
    except Exception as e:
        logger.warning(f"AkShare search failed: {e}, using fallback")

        # 降级策略：使用本地静态数据
        static_stocks = [
            {"symbol": "600519", "name": "贵州茅台"},
            {"symbol": "000001", "name": "平安银行"},
            {"symbol": "000002", "name": "万科A"},
            {"symbol": "000333", "name": "美的集团"},
            {"symbol": "000651", "name": "格力电器"},
            {"symbol": "000858", "name": "五粮液"},
            {"symbol": "002415", "name": "海康威视"},
            {"symbol": "002594", "name": "比亚迪"},
            {"symbol": "300750", "name": "宁德时代"},
            {"symbol": "601318", "name": "中国平安"},
            {"symbol": "601398", "name": "工商银行"},
            {"symbol": "601939", "name": "建设银行"},
            {"symbol": "600036", "name": "招商银行"},
            {"symbol": "600276", "name": "恒瑞医药"},
            {"symbol": "600887", "name": "伊利股份"},
            {"symbol": "603259", "name": "药明康德"},
            {"symbol": "600309", "name": "万华化学"},
            {"symbol": "002304", "name": "洋河股份"},
            {"symbol": "000568", "name": "泸州老窖"},
            {"symbol": "002352", "name": "顺丰控股"},
        ]

        # 简单匹配
        query_lower = query.lower()
        results = []
        for stock in static_stocks:
            if query_lower in stock["symbol"].lower() or query_lower in stock["name"].lower():
                results.append({
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "price": 0
                })

        # 缓存结果
        _set_cached_analysis(cache_key, results[:limit])
        return results[:limit]


@router.post("/monitor/start")
async def start_monitor(request: MonitorStartRequest) -> Dict[str, Any]:
    """启动监控服务"""
    import uuid
    from ...application.realtime_monitor_service import RealtimeMonitorService, MonitorConfig

    monitor_id = str(uuid.uuid4())[:8]

    try:
        # 创建监控配置
        config = MonitorConfig(
            symbols=request.symbols,
            strategy=request.strategy,
            interval_minutes=request.interval_minutes,
            buy_threshold=request.buy_threshold,
            sell_threshold=request.sell_threshold,
            history_days=request.history_days
        )

        # 创建监控服务
        provider = get_minute_provider()
        service = RealtimeMonitorService(
            config=config,
            data_provider=provider
        )

        # 保存到全局状态
        _monitor_services[monitor_id] = {
            "service": service,
            "config": config,
            "task": None
        }

        # 在后台启动监控
        import asyncio

        async def run_monitor():
            try:
                await service.start()
            except Exception as e:
                logger.error(f"Monitor {monitor_id} error: {e}")

        task = asyncio.create_task(run_monitor())
        _monitor_services[monitor_id]["task"] = task

        logger.info(f"Started monitor {monitor_id} for {request.symbols}")

        return {
            "monitor_id": monitor_id,
            "status": "started",
            "symbols": request.symbols,
            "strategy": request.strategy
        }

    except Exception as e:
        logger.error(f"Failed to start monitor: {e}")
        raise HTTPException(status_code=500, detail=f"启动监控失败: {str(e)}")


@router.post("/monitor/stop/{monitor_id}")
async def stop_monitor(monitor_id: str) -> Dict[str, Any]:
    """停止监控服务"""
    if monitor_id not in _monitor_services:
        raise HTTPException(status_code=404, detail=f"监控 {monitor_id} 不存在")

    try:
        monitor = _monitor_services[monitor_id]
        service = monitor["service"]

        await service.stop()

        # 取消任务
        if monitor["task"]:
            monitor["task"].cancel()

        del _monitor_services[monitor_id]

        logger.info(f"Stopped monitor {monitor_id}")

        return {"monitor_id": monitor_id, "status": "stopped"}

    except Exception as e:
        logger.error(f"Failed to stop monitor {monitor_id}: {e}")
        raise HTTPException(status_code=500, detail=f"停止监控失败: {str(e)}")


@router.get("/monitor/status/{monitor_id}")
async def get_monitor_status(monitor_id: str) -> Dict[str, Any]:
    """获取监控状态"""
    if monitor_id not in _monitor_services:
        raise HTTPException(status_code=404, detail=f"监控 {monitor_id} 不存在")

    try:
        monitor = _monitor_services[monitor_id]
        service = monitor["service"]
        status = service.get_status()

        # 转换 numpy 类型
        status = to_python_types(status)

        return {
            "running": status.get("running", False),
            "symbols": status.get("symbols", []),
            "strategy": status.get("strategy", ""),
            "interval_minutes": status.get("interval_minutes", 5),
            "check_count": status.get("check_count", 0),
            "signal_count": status.get("signal_count", 0),
            "last_check": status.get("last_check")
        }

    except Exception as e:
        logger.error(f"Failed to get monitor status: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控状态失败: {str(e)}")


@router.get("/monitor/list")
async def list_monitors() -> List[Dict[str, Any]]:
    """列出所有监控"""
    result = []
    for monitor_id, monitor in _monitor_services.items():
        service = monitor["service"]
        status = service.get_status()
        result.append({
            "monitor_id": monitor_id,
            "symbols": status.get("symbols", []),
            "strategy": status.get("strategy", ""),
            "running": status.get("running", False),
            "check_count": status.get("check_count", 0),
            "signal_count": status.get("signal_count", 0)
        })
    return result


@router.get("/monitor/{monitor_id}/signals")
async def get_monitor_signals(monitor_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """获取监控信号"""
    if monitor_id not in _monitor_services:
        raise HTTPException(status_code=404, detail=f"监控 {monitor_id} 不存在")

    try:
        monitor = _monitor_services[monitor_id]
        service = monitor["service"]
        signals = service.get_recent_signals(limit)

        result = []
        for s in signals:
            signal_data = {
                "score": s.score,
                "passed_filter": s.passed_filter,
                "filter_reason": s.filter_reason,
                "signal": None
            }

            if s.signal:
                signal_data["signal"] = {
                    "symbol": s.signal.symbol,
                    "direction": "buy" if s.signal.direction.value == "buy" else "sell",
                    "timestamp": s.signal.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "strength": s.signal.strength,
                    "reason": s.signal.reason,
                    "confidence": s.signal.confidence
                }

            result.append(signal_data)

        return result

    except Exception as e:
        logger.error(f"Failed to get monitor signals: {e}")
        raise HTTPException(status_code=500, detail=f"获取信号失败: {str(e)}")


# ==================== ML 模型策略 API ====================

class MLBacktestRequest(BaseModel):
    """ML 模型回测请求"""
    model_path: str = ""  # 模型路径，为空则自动使用最新模型
    symbols: List[str] = []  # 回测股票列表
    start_date: str = ""  # 开始日期，默认一年前
    end_date: str = ""  # 结束日期，默认今天
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    # 使用与训练一致的阈值
    buy_threshold: float = 0.50
    sell_threshold: float = 0.50


@router.post("/ml/backtest")
async def run_ml_backtest(request: MLBacktestRequest) -> Dict[str, Any]:
    """
    使用 ML 模型进行回测

    使用训练好的 GBM 模型对指定股票进行回测
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        from quanttool.infrastructure.data_providers.qlib_data_loader import QlibDataLoader
        import glob

        # 查找模型
        model_path = request.model_path
        if not model_path:
            model_files = glob.glob("models/gbm/lgbm_*.pkl")
            if not model_files:
                raise HTTPException(status_code=404, detail="未找到训练好的模型，请先训练模型")
            model_path = max(model_files, key=os.path.getmtime)
            logger.info(f"使用最新模型: {model_path}")

        # 解析日期
        end_date = datetime.now() if not request.end_date else datetime.fromisoformat(request.end_date)
        start_date = end_date - timedelta(days=365) if not request.start_date else datetime.fromisoformat(request.start_date)

        # 加载模型
        config = GBMConfig(
            buy_threshold=request.buy_threshold,
            sell_threshold=request.sell_threshold,
        )
        strategy = GBMStrategy(config)
        strategy.load_model(model_path)

        # 初始化数据加载器
        data_loader = QlibDataLoader()
        if not data_loader.init_qlib():
            raise HTTPException(status_code=500, detail="Qlib 初始化失败")

        # 回测逻辑
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')

        # 加载所有股票数据
        # 注意：使用 StockAnalyzer 获取真实价格数据，而非 qlib 数据
        from quanttool.factors.stock_analyzer import StockAnalyzer
        stock_analyzer = StockAnalyzer(use_realtime_price=True)

        all_data = {}
        for symbol in request.symbols:
            df = stock_analyzer.get_stock_data(symbol, days=365)
            if df.empty:
                # 回退到 qlib
                df = data_loader.load_stock_data(symbol, start_str, end_str, use_adjclose=False)
            if not df.empty:
                df = df.reset_index()
                if 'date' in df.columns:
                    df = df.rename(columns={'date': 'timestamp'})
                all_data[symbol] = df

        if not all_data:
            raise HTTPException(status_code=400, detail="没有获取到任何数据")

        # 模拟回测
        cash = request.initial_cash
        position = {}  # 持仓 {symbol: shares}
        trades = []
        portfolio_values = []

        # 获取所有交易日
        all_dates = set()
        for symbol, df in all_data.items():
            for t in df['timestamp']:
                all_dates.add(t)
        sorted_dates = sorted(all_dates)

        for current_date in sorted_dates:
            # 计算当前组合价值
            position_value = 0
            for symbol, shares in position.items():
                if symbol in all_data:
                    df = all_data[symbol]
                    row = df[df['timestamp'] == current_date]
                    if not row.empty:
                        position_value += row['close'].values[0] * shares

            portfolio_value = cash + position_value
            portfolio_values.append({
                'date': current_date,
                'value': portfolio_value
            })

            # 对每只股票生成信号
            for symbol in request.symbols:
                if symbol not in all_data:
                    continue

                df = all_data[symbol]
                historical = df[df['timestamp'] <= current_date]

                if len(historical) < 120:  # 需要足够的历史数据
                    continue

                current_bar = historical.iloc[-1]

                try:
                    signal = strategy.get_signal(current_bar, historical)
                except Exception as e:
                    continue

                # 执行交易
                close = current_bar['close']
                signal_type = signal.get('signal', 'hold')

                if signal_type == 'buy' and symbol not in position:
                    # 买入
                    shares = int(cash * 0.2 / close)  # 每次20%仓位
                    if shares > 0:
                        cost = shares * close * (1 + request.commission_rate)
                        if cost <= cash:
                            cash -= cost
                            position[symbol] = shares
                            trades.append({
                                'symbol': symbol,
                                'action': 'buy',
                                'price': close,
                                'shares': shares,
                                'timestamp': current_date,
                                'reason': f"ML预测概率: {signal.get('probability', 0):.2%}"
                            })

                elif signal_type == 'sell' and symbol in position:
                    # 卖出
                    shares = position[symbol]
                    revenue = shares * close * (1 - request.commission_rate)
                    cash += revenue
                    del position[symbol]
                    trades.append({
                        'symbol': symbol,
                        'action': 'sell',
                        'price': close,
                        'shares': shares,
                        'timestamp': current_date,
                        'reason': f"ML预测概率: {signal.get('probability', 0):.2%}"
                    })

        # 最终价值
        final_position_value = 0
        for symbol, shares in position.items():
            if symbol in all_data:
                df = all_data[symbol]
                if not df.empty:
                    final_position_value += df['close'].iloc[-1] * shares

        final_value = cash + final_position_value
        total_return = (final_value - request.initial_cash) / request.initial_cash

        # 计算最大回撤
        values = [p['value'] for p in portfolio_values]
        max_drawdown = 0
        peak = values[0] if values else 0
        for v in values:
            if v > peak:
                peak = v
            drawdown = (peak - v) / peak if peak > 0 else 0
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 计算年化收益
        days = (end_date - start_date).days if isinstance(end_date, datetime) else 365
        annual_return = total_return * (365 / max(days, 1)) if total_return else 0

        # 计算胜率：盈利的卖出次数 / 总卖出次数
        sell_trades = [t for t in trades if t['action'] == 'sell']
        # 计算每笔卖出的盈亏
        buy_prices = {}
        for t in trades:
            if t['action'] == 'buy':
                buy_prices[t['symbol']] = t['price']
            elif t['action'] == 'sell' and t['symbol'] in buy_prices:
                t['profit'] = (t['price'] - buy_prices[t['symbol']]) * t['shares']

        win_count = sum(1 for t in sell_trades if t.get('profit', 0) > 0)
        win_rate = win_count / max(len(sell_trades), 1)

        return to_python_types({
            "success": True,
            "strategy": "ML-GBM",
            "model_path": model_path,
            "symbols": request.symbols,
            "start_date": start_str,
            "end_date": end_str,
            "initial_capital": request.initial_cash,
            "final_capital": final_value,
            "total_return": total_return,
            "annual_return": annual_return,
            "excess_return": annual_return - 0.05,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": annual_return / max(0.15, max_drawdown) if max_drawdown > 0 else 0,
            "total_trades": len(trades),
            "win_rate": win_rate,
            "trades": trades[-50:],  # 最近50笔交易
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML 回测失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"回测失败: {str(e)}")


class MLScanRequest(BaseModel):
    """ML 模型选股请求"""
    model_path: str = ""
    symbols: List[str] = []  # 候选股票列表
    top_n: int = 20  # 返回前N只
    min_probability: float = 0.50  # 使用与训练一致的买入阈值


@router.post("/ml/scan")
async def scan_with_ml_model(request: MLScanRequest) -> Dict[str, Any]:
    """
    使用 ML 模型进行智能选股

    对候选股票进行预测，返回得分最高的股票
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        from quanttool.infrastructure.data_providers.qlib_data_loader import QlibDataLoader
        from quanttool.cli.commands.analysis_commands import get_csi300_constituents
        import glob

        # 查找模型
        model_path = request.model_path
        if not model_path:
            model_files = glob.glob("models/gbm/lgbm_*.pkl")
            if not model_files:
                raise HTTPException(status_code=404, detail="未找到训练好的模型，请先训练模型")
            model_path = max(model_files, key=os.path.getmtime)
            logger.info(f"使用最新模型: {model_path}")

        # 获取候选股票
        symbols = request.symbols
        if not symbols:
            # 默认使用沪深300成分股
            csi300 = get_csi300_constituents()
            symbols = [s['code'] if isinstance(s, dict) else s for s in csi300]

        # 加载模型
        config = GBMConfig()
        strategy = GBMStrategy(config)
        strategy.load_model(model_path)

        # 初始化数据加载器
        data_loader = QlibDataLoader()
        if not data_loader.init_qlib():
            raise HTTPException(status_code=500, detail="Qlib 初始化失败")

        # 预测所有股票
        results = []
        for symbol in symbols:
            try:
                pred = strategy.predict(symbol)
                if pred.get('probability', 0) >= request.min_probability:
                    results.append({
                        'symbol': symbol,
                        'probability': pred['probability'],
                        'pred_return': pred.get('return_pred', 0),
                        'signal': pred.get('signal', 'hold'),
                        'close': pred.get('close', 0),
                    })
            except Exception as e:
                logger.debug(f"预测失败 {symbol}: {e}")
                continue

        # 按概率排序
        results.sort(key=lambda x: x['probability'], reverse=True)
        top_results = results[:request.top_n]

        return {
            "success": True,
            "model_path": model_path,
            "total_scanned": len(symbols),
            "qualified_count": len(results),
            "min_probability": request.min_probability,
            "results": to_python_types(top_results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML 选股失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"选股失败: {str(e)}")


class MLMonitorRequest(BaseModel):
    """ML 模型监控请求"""
    model_path: str = ""
    symbols: List[str] = []
    interval_seconds: int = 60


@router.post("/ml/monitor/start")
async def start_ml_monitor(request: MLMonitorRequest) -> Dict[str, Any]:
    """
    启动 ML 模型实时监控

    定时对指定股票进行预测并生成信号
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        import glob

        # 查找模型
        model_path = request.model_path
        if not model_path:
            model_files = glob.glob("models/gbm/lgbm_*.pkl")
            if not model_files:
                raise HTTPException(status_code=404, detail="未找到训练好的模型")
            model_path = max(model_files, key=os.path.getmtime)

        # 加载模型
        config = GBMConfig()
        strategy = GBMStrategy(config)
        strategy.load_model(model_path)

        monitor_id = str(uuid.uuid4())[:8]

        # 存储监控信息
        _monitor_services[monitor_id] = {
            "service": strategy,
            "model_path": model_path,
            "symbols": request.symbols,
            "signals": [],
            "started_at": datetime.now(),
            "task": None,
        }

        async def run_ml_monitor():
            while True:
                try:
                    for symbol in request.symbols:
                        try:
                            pred = strategy.predict(symbol)
                            signal = {
                                "symbol": symbol,
                                "probability": pred.get('probability', 0),
                                "signal": pred.get('signal', 'hold'),
                                "timestamp": datetime.now().isoformat(),
                            }
                            _monitor_services[monitor_id]["signals"].insert(0, signal)
                            # 保留最近100条信号
                            _monitor_services[monitor_id]["signals"] = _monitor_services[monitor_id]["signals"][:100]
                        except Exception as e:
                            logger.debug(f"监控预测失败 {symbol}: {e}")

                    await asyncio.sleep(request.interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"ML 监控错误: {e}")
                    await asyncio.sleep(5)

        task = asyncio.create_task(run_ml_monitor())
        _monitor_services[monitor_id]["task"] = task

        return {
            "monitor_id": monitor_id,
            "model_path": model_path,
            "symbols": request.symbols,
            "interval_seconds": request.interval_seconds,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动 ML 监控失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动失败: {str(e)}")


@router.get("/ml/monitor/{monitor_id}/signals")
async def get_ml_monitor_signals(monitor_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """获取 ML 监控信号"""
    if monitor_id not in _monitor_services:
        raise HTTPException(status_code=404, detail=f"监控 {monitor_id} 不存在")

    monitor = _monitor_services[monitor_id]
    signals = monitor.get("signals", [])[:limit]
    return to_python_types(signals)
