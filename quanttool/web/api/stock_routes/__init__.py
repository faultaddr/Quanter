"""Stock API route modules."""

from fastapi import APIRouter

from .analysis import router as analysis_router
from .market_data import router as market_data_router
from .chip_signals import router as chip_signals_router
from .insights import router as insights_router


router = APIRouter()
router.include_router(analysis_router)
router.include_router(market_data_router)
router.include_router(chip_signals_router)
router.include_router(insights_router)
