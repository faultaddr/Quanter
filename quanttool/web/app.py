"""Main FastAPI web application for QuantTool."""

import os
import json
from typing import Any

# 解决 OpenMP 库版本冲突问题 (PyTorch, scikit-learn, LightGBM 等都自带 libomp)
# 必须在任何导入 numpy/torch/sklearn 之前设置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
from fastapi import FastAPI, Response, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from .api.routes import router as api_router
from .ws import signal_websocket_endpoint
from ..core.logging import get_logger


logger = get_logger(__name__)


class NumpyJSONResponse(JSONResponse):
    """自定义 JSON 响应类，支持 numpy 类型"""

    def render(self, content: Any) -> bytes:
        """渲染响应内容，处理 numpy 类型"""
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=True,
            indent=None,
            separators=(",", ":"),
            default=self._numpy_encoder,
        ).encode("utf-8")

    @staticmethod
    def _numpy_encoder(obj):
        """将 numpy 类型转换为 Python 原生类型"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif obj is None or isinstance(obj, (str, int, float, bool, list, dict)):
            return obj
        # 尝试转换为字典（处理 Pydantic 模型等）
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


app = FastAPI(
    title="QuantTool API",
    description="A comprehensive quantitative trading platform for A-share stocks",
    version="0.1.0",
    default_response_class=NumpyJSONResponse,
)

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


# WebSocket endpoint for real-time monitoring
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time quotes and signals."""
    await signal_websocket_endpoint(websocket)


@app.get("/")
async def root():
    """Serve the main web interface."""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to QuantTool API"}


@app.on_event("startup")
async def startup_event():
    """Startup event handler - ensure all strategies and data providers are registered."""
    logger.info("QuantTool API starting up...")

    # Import all strategies to register them
    try:
        from .. import strategies
        logger.info(f"Strategies loaded: {strategies.__all__}")
    except Exception as e:
        logger.warning(f"Failed to load strategies: {e}")

    # Import all data providers to register them
    try:
        from ..infrastructure.data_providers import data_fetcher
        from ..infrastructure.data_providers import ashare_provider
        from ..infrastructure.data_providers import tushare_provider
        logger.info("Data providers loaded")
    except Exception as e:
        logger.warning(f"Failed to load data providers: {e}")

    # Log available components
    from ..core.registry import registry, ComponentType
    strategies_available = registry.list_available(ComponentType.STRATEGY)
    providers_available = registry.list_available(ComponentType.DATA_PROVIDER)
    logger.info(f"Registered strategies: {strategies_available}")
    logger.info(f"Registered data providers: {providers_available}")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("QuantTool API shutting down...")
