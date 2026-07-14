"""ML strategy API route aggregate."""

from fastapi import APIRouter

from .ml_routes import router as ml_routes_router


router = APIRouter()
router.include_router(ml_routes_router)
