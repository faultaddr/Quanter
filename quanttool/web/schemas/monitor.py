"""Realtime monitor API schemas."""

from typing import List, Optional

from pydantic import BaseModel


class MonitorStartRequest(BaseModel):
    """Start monitor request."""

    symbols: List[str]
    strategy: str = "breakout"
    interval_minutes: int = 5
    buy_threshold: int = 50
    sell_threshold: int = 40
    history_days: int = 120


class MonitorStatusResponse(BaseModel):
    """Monitor status response."""

    running: bool
    symbols: List[str] = []
    strategy: str = ""
    interval_minutes: int = 5
    check_count: int = 0
    signal_count: int = 0
    last_check: Optional[str] = None
