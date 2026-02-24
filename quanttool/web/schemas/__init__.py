"""Pydantic schemas for web API."""

from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any, List, Optional


class ExperimentRunSchema(BaseModel):
    id: str
    type: str
    parameters: Dict[str, Any]
    git_commit: Optional[str] = None
    data_version: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str
    results: Optional[Dict[str, Any]] = None
    artifacts: List[str] = []


class BacktestRequest(BaseModel):
    strategy_name: str
    symbols: List[str]
    start_date: str
    end_date: str
    timeframe: str = "10m"
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    data_provider: str = "tushare"
    strategy_params: Dict[str, Any] = {}


class FactorMiningRequest(BaseModel):
    factor_name: str
    symbols: List[str]
    start_date: str
    end_date: str
    data_provider: str = "tushare"
    factor_params: Dict[str, Any] = {}
