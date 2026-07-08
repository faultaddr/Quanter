"""Backtest API route aggregate."""

from fastapi import APIRouter

from .backtest_routes import router as backtest_routes_router


router = APIRouter()
router.include_router(backtest_routes_router)
