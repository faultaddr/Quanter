import unittest

from pydantic import ValidationError

from quanttool.application.serenity_service import SerenityService, classify_quadrant
from quanttool.domain.models.serenity import (
    EvidenceStrength,
    ResearchTimingQuadrant,
    ResearchVerdict,
    SerenityEvidence,
    SerenityFactors,
    SerenityPenalties,
    SerenityScorecard,
)


class SerenityServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = SerenityService()

    def make_scorecard(self, factor_overrides=None, penalties=None, timing_score=82.0):
        factor_values = {
            "demand_inflection": 5.0,
            "architecture_coupling": 5.0,
            "chokepoint_severity": 5.0,
            "supplier_concentration": 5.0,
            "expansion_difficulty": 5.0,
            "evidence_quality": 5.0,
            "valuation_disconnect": 5.0,
            "catalyst_timing": 5.0,
        }
        factor_values.update(factor_overrides or {})
        return SerenityScorecard(
            ticker="688001.SH",
            company="Example Semiconductor",
            market="A-share",
            theme="AI semiconductors",
            layer="equipment",
            role="critical supplier",
            factors=SerenityFactors(**factor_values),
            penalties=penalties or SerenityPenalties(dilution_financing=5.0),
            evidence=[
                SerenityEvidence(
                    claim="Capacity remains constrained.",
                    source="Company annual report",
                    strength=EvidenceStrength.STRONG,
                ),
                SerenityEvidence(
                    claim="Demand could accelerate.",
                    source="Unverified discussion",
                    strength=EvidenceStrength.UNVERIFIED,
                ),
            ],
            what_could_weaken_view=["A substitute design gains adoption."],
            timing_score=timing_score,
        )

    def test_score_uses_exact_weights_and_penalties(self):
        result = self.service.score(self.make_scorecard())

        expected_factor_weights = {
            "demand_inflection": 15.0,
            "architecture_coupling": 10.0,
            "chokepoint_severity": 15.0,
            "supplier_concentration": 12.0,
            "expansion_difficulty": 12.0,
            "evidence_quality": 15.0,
            "valuation_disconnect": 11.0,
            "catalyst_timing": 10.0,
        }

        self.assertEqual(result.raw_factor_points, 100.0)
        self.assertEqual(result.penalty_points, 10.0)
        self.assertEqual(result.research_priority_score, 90.0)
        self.assertEqual(
            {
                name: detail.weight for name, detail in result.factor_details.items()
            },
            expected_factor_weights,
        )
        self.assertEqual(result.factor_details["demand_inflection"].points, 15.0)
        self.assertEqual(result.penalty_details["dilution_financing"].points, 10.0)

    def test_score_clamps_negative_final_score_to_zero(self):
        result = self.service.score(
            SerenityScorecard(
                factors=SerenityFactors(),
                penalties=SerenityPenalties(dilution_financing=5.0),
            )
        )

        self.assertEqual(result.raw_factor_points, 0.0)
        self.assertEqual(result.penalty_points, 10.0)
        self.assertEqual(result.research_priority_score, 0.0)

    def test_score_assigns_each_verdict_threshold(self):
        cases = (
            ({"evidence_quality": 0.0}, ResearchVerdict.TOP_PRIORITY),
            (
                {"demand_inflection": 0.0, "evidence_quality": 0.0},
                ResearchVerdict.HIGH_PRIORITY,
            ),
            (
                {
                    "demand_inflection": 0.0,
                    "evidence_quality": 0.0,
                    "chokepoint_severity": 0.0,
                },
                ResearchVerdict.WORTH_TRACKING,
            ),
            (
                {
                    "demand_inflection": 0.0,
                    "evidence_quality": 0.0,
                    "chokepoint_severity": 0.0,
                    "catalyst_timing": 4.995,
                },
                ResearchVerdict.EARLY_LEAD,
            ),
        )

        for factor_overrides, expected_verdict in cases:
            with self.subTest(expected_verdict=expected_verdict):
                result = self.service.score(
                    self.make_scorecard(
                        factor_overrides=factor_overrides,
                        penalties=SerenityPenalties(),
                    )
                )
                self.assertEqual(result.verdict, expected_verdict)

    def test_models_reject_out_of_range_ratings_and_timing_score(self):
        with self.assertRaises(ValidationError):
            SerenityFactors(demand_inflection=5.01)
        with self.assertRaises(ValidationError):
            SerenityPenalties(governance=-0.01)
        with self.assertRaises(ValidationError):
            SerenityScorecard(factors=SerenityFactors(), timing_score=100.01)

    def test_models_reject_out_of_range_assignments(self):
        factors = SerenityFactors()
        penalties = SerenityPenalties()
        scorecard = SerenityScorecard()

        with self.assertRaises(ValidationError):
            factors.demand_inflection = 5.01
        with self.assertRaises(ValidationError):
            penalties.governance = -0.01
        with self.assertRaises(ValidationError):
            scorecard.timing_score = 100.01

    def test_models_reject_unknown_factor_and_penalty_fields(self):
        with self.assertRaises(ValidationError):
            SerenityFactors(demand_inflecton=4.0)
        with self.assertRaises(ValidationError):
            SerenityPenalties(governace=4.0)

    def test_penalty_keys_match_serenity_scorecard_contract(self):
        penalties = SerenityPenalties(hype_risk=2.0, alternative_design_risk=3.0)

        result = self.service.score(
            SerenityScorecard(
                factors=SerenityFactors(),
                penalties=penalties,
            )
        )

        self.assertEqual(result.penalty_details["hype_risk"].points, 4.0)
        self.assertEqual(
            result.penalty_details["alternative_design_risk"].points,
            6.0,
        )

    def test_score_counts_evidence_by_strength_and_preserves_weakening_conditions(self):
        result = self.service.score(self.make_scorecard())

        self.assertEqual(result.evidence_summary.total, 2)
        self.assertEqual(result.evidence_summary.strong, 1)
        self.assertEqual(result.evidence_summary.medium, 0)
        self.assertEqual(result.evidence_summary.weak, 0)
        self.assertEqual(result.evidence_summary.unverified, 1)
        self.assertEqual(
            result.what_could_weaken_view,
            ["A substitute design gains adoption."],
        )

    def test_classify_quadrant_covers_all_combinations_and_missing_timing(self):
        self.assertEqual(
            classify_quadrant(70.0, 70.0),
            ResearchTimingQuadrant.PRIORITY_NOW,
        )
        self.assertEqual(
            classify_quadrant(70.0, 69.99),
            ResearchTimingQuadrant.RESEARCH_WAIT,
        )
        self.assertEqual(
            classify_quadrant(69.99, 70.0),
            ResearchTimingQuadrant.TIMING_ONLY,
        )
        self.assertEqual(
            classify_quadrant(69.99, 69.99),
            ResearchTimingQuadrant.LOW_PRIORITY,
        )
        self.assertIsNone(classify_quadrant(90.0, None))

    def test_template_returns_independent_valid_scorecards(self):
        first_template = self.service.template()
        second_template = self.service.template()

        first_template.what_could_weaken_view.append("Changed only on the first template.")

        self.assertEqual(first_template.market, "A-share")
        self.assertEqual(first_template.factors.demand_inflection, 0.0)
        self.assertEqual(second_template.what_could_weaken_view, [])

    def test_markdown_contains_research_sections_and_boundary(self):
        markdown = self.service.to_markdown(self.service.score(self.make_scorecard()))

        for expected_section in (
            "# Serenity research scorecard: 688001.SH (Example Semiconductor)",
            "## Candidate",
            "## Research priority",
            "Timing score: **82.0 / 100**",
            "## Factor scores",
            "## Penalties",
            "## Evidence",
            "## What could weaken the view",
            "## Research boundary",
            "Research priority only. This is not a trading instruction.",
        ):
            with self.subTest(expected_section=expected_section):
                self.assertIn(expected_section, markdown)


if __name__ == "__main__":
    unittest.main()
