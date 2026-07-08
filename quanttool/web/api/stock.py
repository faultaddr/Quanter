"""Stock analysis API route aggregate."""

from fastapi import APIRouter

from .stock_routes import router as stock_routes_router


router = APIRouter()
router.include_router(stock_routes_router)
