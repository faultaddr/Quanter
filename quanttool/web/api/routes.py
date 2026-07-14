"""Aggregate API router for QuantTool web application."""

from fastapi import APIRouter

from . import (
    backtest,
    factors,
    ml,
    models,
    monitor,
    realtime,
    research,
    registry,
    risk,
    scan,
    stock,
    tasks,
)


router = APIRouter()

router.include_router(tasks.router)
router.include_router(stock.router)
router.include_router(scan.router)
router.include_router(backtest.router)
router.include_router(models.router)
router.include_router(factors.router)
router.include_router(registry.router)
router.include_router(risk.router)
router.include_router(realtime.router)
router.include_router(research.router)
router.include_router(monitor.router)
router.include_router(ml.router)
