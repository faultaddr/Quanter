"""Factor API schemas."""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional


class FactorMineRequest(BaseModel):
    """Request schema for factor mining."""

    factor_name: str = Field(..., description="Name of the factor to mine")
    symbols: List[str] = Field(..., description="List of symbols to analyze")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    data_provider: str = Field(default="tushare", description="Data provider to use")
    factor_params: Dict[str, Any] = Field(default_factory=dict, description="Factor parameters")


class FactorResultSchema(BaseModel):
    """Schema for individual factor evaluation result."""

    factor_name: str
    ic: float
    rank_ic: float
    ic_ir: float
    win_rate: float
    avg_return: float
    volatility: float
    sharpe_ratio: float
    turnover: float
    max_exposure: float


class FactorResponse(BaseModel):
    """Response schema for factor mining results."""

    run_id: str
    factor_name: str
    results: Dict[str, FactorResultSchema]
