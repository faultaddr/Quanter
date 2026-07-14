"""Pure scoring and rendering service for Serenity research scorecards."""

from copy import deepcopy
from typing import Dict, List, Optional

from quanttool.domain.models.serenity import (
    EvidenceStrength,
    ResearchTimingQuadrant,
    ResearchVerdict,
    SerenityEvidenceSummary,
    SerenityScoreDetail,
    SerenityScoreResult,
    SerenityScorecard,
)


FACTOR_WEIGHTS = {
    "demand_inflection": 15.0,
    "architecture_coupling": 10.0,
    "chokepoint_severity": 15.0,
    "supplier_concentration": 12.0,
    "expansion_difficulty": 12.0,
    "evidence_quality": 15.0,
    "valuation_disconnect": 11.0,
    "catalyst_timing": 10.0,
}

PENALTY_WEIGHTS = {
    "dilution_financing": 2.0,
    "governance": 2.0,
    "geopolitics": 2.0,
    "liquidity": 2.0,
    "hype_risk": 2.0,
    "accounting_quality": 2.0,
    "cyclicality": 2.0,
    "alternative_design_risk": 2.0,
}


def classify_quadrant(
    research_score: float,
    timing_score: Optional[float],
    threshold: float = 70.0,
) -> Optional[ResearchTimingQuadrant]:
    """Classify separate research and timing scores without combining them."""

    if timing_score is None:
        return None

    research_is_high = research_score >= threshold
    timing_is_high = timing_score >= threshold
    if research_is_high and timing_is_high:
        return ResearchTimingQuadrant.PRIORITY_NOW
    if research_is_high:
        return ResearchTimingQuadrant.RESEARCH_WAIT
    if timing_is_high:
        return ResearchTimingQuadrant.TIMING_ONLY
    return ResearchTimingQuadrant.LOW_PRIORITY


def _classify_verdict(score: float) -> ResearchVerdict:
    if score >= 85.0:
        return ResearchVerdict.TOP_PRIORITY
    if score >= 70.0:
        return ResearchVerdict.HIGH_PRIORITY
    if score >= 55.0:
        return ResearchVerdict.WORTH_TRACKING
    return ResearchVerdict.EARLY_LEAD


class SerenityService:
    """Score validated research inputs without external data dependencies."""

    def score(self, scorecard: SerenityScorecard) -> SerenityScoreResult:
        """Calculate research priority and preserve its supporting context."""

        factor_details = self._score_fields(scorecard.factors, FACTOR_WEIGHTS)
        penalty_details = self._score_fields(
            scorecard.penalties,
            PENALTY_WEIGHTS,
            normalize_rating=False,
        )
        raw_factor_points = sum(detail.points for detail in factor_details.values())
        penalty_points = sum(detail.points for detail in penalty_details.values())
        research_score = min(max(raw_factor_points - penalty_points, 0.0), 100.0)

        evidence_counts = {strength.value: 0 for strength in EvidenceStrength}
        for evidence in scorecard.evidence:
            evidence_counts[evidence.strength.value] += 1

        return SerenityScoreResult(
            ticker=scorecard.ticker,
            company=scorecard.company,
            market=scorecard.market,
            theme=scorecard.theme,
            layer=scorecard.layer,
            role=scorecard.role,
            research_priority_score=research_score,
            raw_factor_points=raw_factor_points,
            penalty_points=penalty_points,
            verdict=_classify_verdict(research_score),
            timing_score=scorecard.timing_score,
            quadrant=classify_quadrant(research_score, scorecard.timing_score),
            factor_details=factor_details,
            penalty_details=penalty_details,
            evidence_summary=SerenityEvidenceSummary(
                total=len(scorecard.evidence),
                strong=evidence_counts[EvidenceStrength.STRONG.value],
                medium=evidence_counts[EvidenceStrength.MEDIUM.value],
                weak=evidence_counts[EvidenceStrength.WEAK.value],
                unverified=evidence_counts[EvidenceStrength.UNVERIFIED.value],
            ),
            evidence=deepcopy(scorecard.evidence),
            what_could_weaken_view=list(scorecard.what_could_weaken_view),
        )

    def template(self) -> SerenityScorecard:
        """Return a fresh, valid scorecard template."""

        return SerenityScorecard()

    def to_markdown(self, result: SerenityScoreResult) -> str:
        """Render a complete human-readable research scorecard."""

        identity = result.ticker or "Untitled candidate"
        if result.company:
            identity = "{} ({})".format(identity, result.company)

        lines = [
            "# Serenity research scorecard: {}".format(identity),
            "",
            "## Candidate",
            "",
            "| Field | Value |",
            "| --- | --- |",
            "| Ticker | {} |".format(result.ticker or "Not provided"),
            "| Company | {} |".format(result.company or "Not provided"),
            "| Market | {} |".format(result.market or "Not provided"),
            "| Theme | {} |".format(result.theme or "Not provided"),
            "| Layer | {} |".format(result.layer or "Not provided"),
            "| Role | {} |".format(result.role or "Not provided"),
            "",
            "## Research priority",
            "",
            "Research priority score: **{:.1f} / 100**".format(
                result.research_priority_score
            ),
            "",
            "Verdict: **{}**".format(result.verdict.value),
            "",
        ]

        if result.timing_score is None:
            lines.append("Timing score: **Not provided**")
            lines.append("")
            lines.append("Quadrant: **Not available without a timing score**")
        else:
            lines.append("Timing score: **{:.1f} / 100**".format(result.timing_score))
            lines.append("")
            lines.append("Quadrant: **{}**".format(result.quadrant.value))

        lines.extend(
            [
                "",
                "## Factor scores",
                "",
                "| Factor | Rating | Weight | Points |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        lines.extend(self._detail_rows(result.factor_details))

        lines.extend(
            [
                "",
                "## Penalties",
                "",
                "| Penalty | Rating | Weight | Points |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        lines.extend(self._detail_rows(result.penalty_details))

        summary = result.evidence_summary
        lines.extend(
            [
                "",
                "## Evidence",
                "",
                "Total: {} | Strong: {} | Medium: {} | Weak: {} | Unverified: {}".format(
                    summary.total,
                    summary.strong,
                    summary.medium,
                    summary.weak,
                    summary.unverified,
                ),
                "",
            ]
        )
        if result.evidence:
            for evidence in result.evidence:
                published = ""
                if evidence.published_at is not None:
                    published = ", published {}".format(evidence.published_at.isoformat())
                lines.append(
                    "- **{}** {} (Source: {}{})".format(
                        evidence.strength.value,
                        evidence.claim,
                        evidence.source,
                        published,
                    )
                )
        else:
            lines.append("- None provided.")

        lines.extend(["", "## What could weaken the view", ""])
        if result.what_could_weaken_view:
            lines.extend(
                "- {}".format(condition)
                for condition in result.what_could_weaken_view
            )
        else:
            lines.append("- None provided.")

        lines.extend(
            [
                "",
                "## Research boundary",
                "",
                "Research priority only. This is not a trading instruction.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _score_fields(
        values: object,
        weights: Dict[str, float],
        normalize_rating: bool = True,
    ) -> Dict[str, SerenityScoreDetail]:
        details = {}
        for field_name, weight in weights.items():
            rating = float(getattr(values, field_name))
            points = rating * weight
            if normalize_rating:
                points /= 5.0
            details[field_name] = SerenityScoreDetail(
                rating=rating,
                weight=weight,
                points=points,
            )
        return details

    @staticmethod
    def _detail_rows(details: Dict[str, SerenityScoreDetail]) -> List[str]:
        return [
            "| {} | {:.2f} | {:.2f} | {:.2f} |".format(
                field_name,
                detail.rating,
                detail.weight,
                detail.points,
            )
            for field_name, detail in details.items()
        ]


__all__ = [
    "FACTOR_WEIGHTS",
    "PENALTY_WEIGHTS",
    "SerenityService",
    "classify_quadrant",
]
