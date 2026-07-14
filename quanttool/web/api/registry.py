"""Registry listing API routes."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
import os
import queue
import threading
import time
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...core.logging import get_logger
from .utils import get_cached_analysis, set_cached_analysis, to_python_types


logger = get_logger(__name__)
router = APIRouter()

import quanttool.strategies


@router.get("/data/providers")
async def list_data_providers() -> List[str]:
    """List available data providers."""
    from ...core.registry import registry, ComponentType

    providers = registry.list_available(ComponentType.DATA_PROVIDER)
    return providers


@router.get("/strategies")
async def list_strategies() -> List[str]:
    """List available strategies."""
    from ...core.registry import registry, ComponentType

    strategies = registry.list_available(ComponentType.STRATEGY)
    return strategies


@router.get("/factors")
async def list_factors() -> List[str]:
    """List available factors."""
    from ...core.registry import registry, ComponentType

    factors = registry.list_available(ComponentType.FACTOR)
    return factors


# ==================== 因子有效性检验 API ====================
