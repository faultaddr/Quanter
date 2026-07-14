"""Backtest API route modules."""

from fastapi import APIRouter

from .catalog import router as catalog_router
from .execution import router as execution_router
from .comparison import router as comparison_router
from .stream import router as stream_router
from .experiments import router as experiments_router


router = APIRouter()
router.include_router(catalog_router)
router.include_router(execution_router)
router.include_router(comparison_router)
router.include_router(stream_router)
router.include_router(experiments_router)
