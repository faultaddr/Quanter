"""Stock analysis API schemas."""

from pydantic import BaseModel


class AnalyzeRequest(BaseModel):
    """Stock analysis request."""

    symbol: str
    days: int = 360


class EnhancedAnalyzeRequest(BaseModel):
    """Enhanced stock analysis request."""

    symbol: str
    days: int = 360
    include_chip: bool = True
    include_patterns: bool = True
    include_strategies: bool = True
