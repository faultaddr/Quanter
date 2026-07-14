"""FastAPI coverage for Serenity research scorecards."""

from datetime import timezone
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from quanttool.domain.models.serenity import SerenityScorecard


def response_dict(response):
    """Serialize Pydantic responses across supported major versions."""

    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


class SerenityApiTests(unittest.TestCase):
    """Verify the research API is a thin, stable adapter over SerenityService."""

    @staticmethod
    def scorecard_payload():
        return {
            "ticker": "688012.SH",
            "company": "Example Semiconductor",
            "factors": {
                "demand_inflection": 5.0,
                "architecture_coupling": 4.0,
                "chokepoint_severity": 4.0,
                "evidence_quality": 4.0,
            },
            "penalties": {"hype_risk": 1.0},
            "timing_score": 82.0,
        }

    def test_research_router_registers_template_and_scorecard_routes(self):
        from quanttool.web.api.research import router

        routes = {
            (method, route.path)
            for route in router.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method not in {"HEAD", "OPTIONS"}
        }

        self.assertEqual(
            routes,
            {
                ("GET", "/research/serenity/template"),
                ("POST", "/research/serenity/scorecard"),
            },
        )

    def test_template_endpoint_returns_exact_utc_success_envelope(self):
        from quanttool.web.api.research import get_serenity_template

        response = get_serenity_template()
        body = response_dict(response)

        self.assertEqual(set(body), {"success", "data", "error", "timestamp"})
        self.assertTrue(body["success"])
        self.assertIsNone(body["error"])
        self.assertEqual(body["data"]["market"], "A-share")
        self.assertEqual(body["data"]["factors"]["demand_inflection"], 0.0)
        self.assertEqual(body["timestamp"].tzinfo, timezone.utc)

    def test_scorecard_endpoint_returns_domain_result_in_exact_utc_envelope(self):
        from quanttool.web.api.research import score_serenity_scorecard

        response = score_serenity_scorecard(SerenityScorecard(**self.scorecard_payload()))
        body = response_dict(response)

        self.assertEqual(set(body), {"success", "data", "error", "timestamp"})
        self.assertTrue(body["success"])
        self.assertIsNone(body["error"])
        self.assertEqual(body["data"]["ticker"], "688012.SH")
        self.assertIn("research_priority_score", body["data"])
        self.assertEqual(body["timestamp"].tzinfo, timezone.utc)

    def test_scorecard_request_rejects_out_of_range_factor_with_fastapi_422(self):
        from quanttool.web.app import app

        payload = self.scorecard_payload()
        payload["factors"]["demand_inflection"] = 5.01

        response = TestClient(app).post(
            "/api/research/serenity/scorecard",
            json=payload,
        )

        self.assertEqual(response.status_code, 422)

    def test_scorecard_endpoint_wraps_unexpected_service_error(self):
        from quanttool.web.api.research import score_serenity_scorecard

        with patch(
            "quanttool.web.api.research.SerenityService.score",
            side_effect=RuntimeError("scoring unavailable"),
        ):
            with patch("quanttool.web.api.research.logger.exception") as log_mock:
                response = score_serenity_scorecard(SerenityScorecard())

        body = response_dict(response)
        self.assertEqual(set(body), {"success", "data", "error", "timestamp"})
        self.assertFalse(body["success"])
        self.assertIsNone(body["data"])
        self.assertEqual(body["error"], "Serenity research service unavailable")
        self.assertNotIn("scoring unavailable", body["error"])
        self.assertEqual(body["timestamp"].tzinfo, timezone.utc)
        log_mock.assert_called_once_with("Serenity research request failed")


if __name__ == "__main__":
    unittest.main()
