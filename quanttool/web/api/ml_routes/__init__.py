"""ML strategy API route modules."""

from fastapi import APIRouter

from .backtest import router as backtest_router
from .scan import router as scan_router
from .monitor import router as monitor_router


router = APIRouter()
router.include_router(backtest_router)
router.include_router(scan_router)
router.include_router(monitor_router)
