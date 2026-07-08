"""Task management API routes."""

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

from ..schemas.model import QlibPredictRequest, QlibTrainRequest
from ..schemas.scan import ScanRequest
from ..schemas.stock import AnalyzeRequest
from ..schemas.tasks import TaskCreateRequest


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
