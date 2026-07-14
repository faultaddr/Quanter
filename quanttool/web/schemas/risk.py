"""Portfolio risk API schemas."""

from typing import Dict

from pydantic import BaseModel


class PortfolioCheckRequest(BaseModel):
    """Portfolio risk check request."""

    positions: Dict[str, dict]
    industry_map: Dict[str, str]
    portfolio_value: float
    peak_value: float
