"""GBM and Qlib model API route modules."""

from fastapi import APIRouter

from .discovery import router as discovery_router
from .gbm import router as gbm_router
from .qlib_training import router as qlib_training_router
from .qlib_prediction import router as qlib_prediction_router


router = APIRouter()
router.include_router(discovery_router)
router.include_router(gbm_router)
router.include_router(qlib_training_router)
router.include_router(qlib_prediction_router)
