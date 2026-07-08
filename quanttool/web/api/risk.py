"""Risk API routes."""

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

from ..schemas.risk import PortfolioCheckRequest


@router.post("/risk/portfolio/check")
async def check_portfolio_risk(request: PortfolioCheckRequest) -> dict:
    """组合风险检查

    Args:
        request: 包含positions, industry_map, portfolio_value, peak_value

    Returns:
        风险检查报告
    """
    from quanttool.risk.risk_controller import PortfolioRiskManager

    manager = PortfolioRiskManager()
    report = manager.check_risk(
        positions=request.positions,
        industry_map=request.industry_map,
        portfolio_value=request.portfolio_value,
        peak_value=request.peak_value,
    )

    return {
        "risk_score": report.overall_risk_score,
        "industry_violations": [
            {"industry": v.industry, "exposure": v.exposure, "limit": v.limit}
            for v in report.industry_violations
        ],
        "blacklist_violations": report.blacklist_violations,
        "position_shrink_factor": report.position_shrink_factor,
        "recommendations": report.recommendations,
    }


# ==================== 实时数据 API ====================
