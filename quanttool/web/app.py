"""Main FastAPI web application for QuantTool."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .api.routes import router as api_router
from ..core.logging import get_logger
import os


logger = get_logger(__name__)

app = FastAPI(
    title="QuantTool API",
    description="A comprehensive quantitative trading platform for A-share stocks",
    version="0.1.0",
)

# Serve static files
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


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
