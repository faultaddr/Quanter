"""Experiment API schemas."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, List, Optional


class ExperimentRunSchema(BaseModel):
    """Schema for experiment runs."""

    id: str = Field(..., description="Experiment run ID")
    type: str = Field(..., description="Type of experiment (backtest, factor_mining, etc.)")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Experiment parameters")
    git_commit: Optional[str] = Field(None, description="Git commit hash")
    data_version: str = Field(..., description="Data version")
    start_time: datetime = Field(..., description="Start time")
    end_time: Optional[datetime] = Field(None, description="End time")
    status: str = Field(..., description="Status (pending, running, completed, failed)")
    results: Optional[Dict[str, Any]] = Field(None, description="Experiment results")
    artifacts: List[str] = Field(default_factory=list, description="Paths to output files")


class ExperimentListResponse(BaseModel):
    """Response schema for listing experiments."""

    total: int
    experiments: List[ExperimentRunSchema]
