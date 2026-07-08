"""Realtime monitor API routes."""

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

from .dependencies import get_minute_provider
from ..schemas.monitor import MonitorStartRequest, MonitorStatusResponse


_monitor_services: Dict[str, Any] = {}


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
