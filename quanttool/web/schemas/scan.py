"""Market scan API schemas."""

from pydantic import BaseModel


class ScanRequest(BaseModel):
    """Market scan request."""

    market: str = "csi300"
    days: int = 360
    top_n: int = 10
    use_unified_score: bool = False
    use_trend_score: bool = True
    use_breakout_score: bool = False
    use_momentum_score: bool = False
    include_fundamentals: bool = False
    include_market_state: bool = False
