"""Validated domain contracts for Serenity research prioritization."""

from datetime import date
from enum import Enum
from typing import Dict, List, Optional

from pydantic import VERSION, BaseModel, Field


if VERSION.startswith("1."):

    class _SerenityInputModel(BaseModel):
        class Config:
            extra = "forbid"
            validate_assignment = True

else:
    from pydantic import ConfigDict

    class _SerenityInputModel(BaseModel):
        model_config = ConfigDict(extra="forbid", validate_assignment=True)


class EvidenceStrength(str, Enum):
    """Strength assigned to a piece of research evidence."""

    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"
    UNVERIFIED = "unverified"


class ResearchVerdict(str, Enum):
    """Research priority band derived from the Serenity score."""

    TOP_PRIORITY = "top_priority"
    HIGH_PRIORITY = "high_priority"
    WORTH_TRACKING = "worth_tracking"
    EARLY_LEAD = "early_lead"


class ResearchTimingQuadrant(str, Enum):
    """Relationship between research priority and independent timing score."""

    PRIORITY_NOW = "priority_now"
    RESEARCH_WAIT = "research_wait"
    TIMING_ONLY = "timing_only"
    LOW_PRIORITY = "low_priority"


class SerenityFactors(_SerenityInputModel):
    """Positive Serenity research factors, each rated from zero to five."""

    demand_inflection: float = Field(0.0, ge=0.0, le=5.0)
    architecture_coupling: float = Field(0.0, ge=0.0, le=5.0)
    chokepoint_severity: float = Field(0.0, ge=0.0, le=5.0)
    supplier_concentration: float = Field(0.0, ge=0.0, le=5.0)
    expansion_difficulty: float = Field(0.0, ge=0.0, le=5.0)
    evidence_quality: float = Field(0.0, ge=0.0, le=5.0)
    valuation_disconnect: float = Field(0.0, ge=0.0, le=5.0)
    catalyst_timing: float = Field(0.0, ge=0.0, le=5.0)


class SerenityPenalties(_SerenityInputModel):
    """Research risks that reduce priority, each rated from zero to five."""

    dilution_financing: float = Field(0.0, ge=0.0, le=5.0)
    governance: float = Field(0.0, ge=0.0, le=5.0)
    geopolitics: float = Field(0.0, ge=0.0, le=5.0)
    liquidity: float = Field(0.0, ge=0.0, le=5.0)
    hype_risk: float = Field(0.0, ge=0.0, le=5.0)
    accounting_quality: float = Field(0.0, ge=0.0, le=5.0)
    cyclicality: float = Field(0.0, ge=0.0, le=5.0)
    alternative_design_risk: float = Field(0.0, ge=0.0, le=5.0)


class SerenityEvidence(_SerenityInputModel):
    """A claim and source retained as part of the research result."""

    claim: str
    source: str
    strength: EvidenceStrength
    published_at: Optional[date] = None


class SerenityScorecard(_SerenityInputModel):
    """Research candidate and the ratings supplied by a researcher."""

    ticker: str = ""
    company: str = ""
    market: str = "A-share"
    theme: str = ""
    layer: str = ""
    role: str = ""
    factors: SerenityFactors = Field(default_factory=SerenityFactors)
    penalties: SerenityPenalties = Field(default_factory=SerenityPenalties)
    evidence: List[SerenityEvidence] = Field(default_factory=list)
    what_could_weaken_view: List[str] = Field(default_factory=list)
    timing_score: Optional[float] = Field(None, ge=0.0, le=100.0)


class SerenityScoreDetail(BaseModel):
    """Contribution made by one factor or penalty rating."""

    rating: float
    weight: float
    points: float


class SerenityEvidenceSummary(BaseModel):
    """Evidence counts grouped by declared strength."""

    total: int = 0
    strong: int = 0
    medium: int = 0
    weak: int = 0
    unverified: int = 0


class SerenityScoreResult(BaseModel):
    """Complete, presentation-neutral Serenity scoring result."""

    ticker: str
    company: str
    market: str
    theme: str
    layer: str
    role: str
    research_priority_score: float
    raw_factor_points: float
    penalty_points: float
    verdict: ResearchVerdict
    timing_score: Optional[float]
    quadrant: Optional[ResearchTimingQuadrant]
    factor_details: Dict[str, SerenityScoreDetail]
    penalty_details: Dict[str, SerenityScoreDetail]
    evidence_summary: SerenityEvidenceSummary
    evidence: List[SerenityEvidence]
    what_could_weaken_view: List[str]


__all__ = [
    "EvidenceStrength",
    "ResearchTimingQuadrant",
    "ResearchVerdict",
    "SerenityEvidence",
    "SerenityEvidenceSummary",
    "SerenityFactors",
    "SerenityPenalties",
    "SerenityScoreDetail",
    "SerenityScoreResult",
    "SerenityScorecard",
]
