"""Main FastAPI web application for QuantTool."""

from fastapi import FastAPI
from .api.routes import router as api_router
from ..core.logging import get_logger


logger = get_logger(__name__)

app = FastAPI(
    title="QuantTool API",
    description="A comprehensive quantitative trading platform for A-share stocks",
    version="0.1.0",
)


# Include API routes
app.include_router(api_router, prefix="/api", tags=["api"])


@app.get("/")
async def root():
    """Root endpoint for the API."""
    return {"message": "Welcome to QuantTool API"}


@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("QuantTool API starting up...")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("QuantTool API shutting down...")
