"""Task API schemas."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    """Task creation request."""

    name: str = Field(..., description="Task name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Task params")
