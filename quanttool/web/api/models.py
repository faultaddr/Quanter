"""GBM and Qlib model API route aggregate."""

from fastapi import APIRouter

from .model_routes import router as model_routes_router


router = APIRouter()
router.include_router(model_routes_router)
