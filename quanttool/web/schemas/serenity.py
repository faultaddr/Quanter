"""HTTP response schema for Serenity research endpoints."""

from datetime import datetime
from typing import Optional, Union

from pydantic import BaseModel

from quanttool.domain.models.serenity import SerenityScorecard, SerenityScoreResult


class SerenityResponse(BaseModel):
    """Isolated HTTP envelope for Serenity research responses."""

    success: bool
    data: Optional[Union[SerenityScoreResult, SerenityScorecard]] = None
    error: Optional[str] = None
    timestamp: datetime


__all__ = ["SerenityResponse"]
