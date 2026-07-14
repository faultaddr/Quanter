"""Common API schemas."""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class ApiResponse(BaseModel):
    """Generic API response envelope."""

    success: bool = True
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
