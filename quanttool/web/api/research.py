"""Serenity research scorecard API routes."""

from datetime import datetime, timezone

from fastapi import APIRouter

from quanttool.application.serenity_service import SerenityService
from quanttool.domain.models.serenity import SerenityScorecard
from quanttool.web.schemas.serenity import SerenityResponse


router = APIRouter(prefix="/research/serenity", tags=["research"])


def _utc_now() -> datetime:
    """Return an aware UTC timestamp for a Serenity HTTP response."""

    return datetime.now(timezone.utc)


def _failure_response(error: Exception) -> SerenityResponse:
    """Keep unexpected service failures inside the Serenity response contract."""

    return SerenityResponse(
        success=False,
        data=None,
        error=str(error),
        timestamp=_utc_now(),
    )


@router.get("/template", response_model=SerenityResponse)
def get_serenity_template() -> SerenityResponse:
    """Return a valid blank Serenity scorecard for clients to populate."""

    try:
        return SerenityResponse(
            success=True,
            data=SerenityService().template(),
            error=None,
            timestamp=_utc_now(),
        )
    except Exception as error:
        return _failure_response(error)


@router.post("/scorecard", response_model=SerenityResponse)
def score_serenity_scorecard(scorecard: SerenityScorecard) -> SerenityResponse:
    """Score a validated Serenity research candidate without mixing timing axes."""

    try:
        return SerenityResponse(
            success=True,
            data=SerenityService().score(scorecard),
            error=None,
            timestamp=_utc_now(),
        )
    except Exception as error:
        return _failure_response(error)
