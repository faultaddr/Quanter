"""Realtime quote API schemas."""

import numpy as np
from pydantic import BaseModel, field_validator


class RealtimeQuoteResponse(BaseModel):
    """Realtime quote response."""

    symbol: str
    name: str = ""
    price: float = 0
    open: float = 0
    high: float = 0
    low: float = 0
    volume: float = 0
    amount: float = 0
    pct_change: float = 0
    change: float = 0
    turnover: float = 0
    timestamp: str = ""

    @field_validator(
        "price",
        "open",
        "high",
        "low",
        "volume",
        "amount",
        "pct_change",
        "change",
        "turnover",
        mode="before",
    )
    @classmethod
    def convert_numpy_types(cls, value):
        """Convert numpy values into native Python floats."""
        if isinstance(value, (np.integer, np.floating)):
            return float(value)
        return value
