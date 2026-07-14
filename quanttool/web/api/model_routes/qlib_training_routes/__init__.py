"""Qlib training API route modules."""

from fastapi import APIRouter

from .batch import router as batch_router
from .stream import router as stream_router


router = APIRouter()
router.include_router(batch_router)
router.include_router(stream_router)
