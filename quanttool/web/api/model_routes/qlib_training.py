"""Qlib model training API route aggregate."""

from fastapi import APIRouter

from .qlib_training_routes import router as qlib_training_routes_router


router = APIRouter()
router.include_router(qlib_training_routes_router)
