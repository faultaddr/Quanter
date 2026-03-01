"""Data API schemas."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional


class DataPullRequest(BaseModel):
    """Request schema for pulling data."""

    provider: str = Field(default="tushare", description="Data provider to use")
    symbols: List[str] = Field(..., description="List of symbols to pull")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    timeframe: str = Field(default="1d", description="Timeframe (1m, 5m, 10m, 1d)")
    save_to_store: bool = Field(default=True, description="Whether to save to store")


class DataSearchRequest(BaseModel):
    """Request schema for searching symbols."""

    provider: str = Field(default="tushare", description="Data provider to use")
    query: str = Field(..., description="Search query")


class SymbolInfoSchema(BaseModel):
    """Schema for symbol information."""

    symbol: str
    name: str
    area: Optional[str] = None
    industry: Optional[str] = None
    list_date: Optional[str] = None


class DataPullResponse(BaseModel):
    """Response schema for data pull."""

    provider: str
    symbols: List[str]
    timeframe: str
    start_date: datetime
    end_date: datetime
    num_symbols_retrieved: int
