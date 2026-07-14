"""ML monitor API routes."""

from datetime import datetime
import asyncio
import os
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ....core.logging import get_logger
from ...schemas.ml import MLMonitorRequest
from ..utils import to_python_types


logger = get_logger(__name__)
router = APIRouter()

_monitor_services: Dict[str, Any] = {}


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
