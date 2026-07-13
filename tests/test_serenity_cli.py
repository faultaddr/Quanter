"""CLI coverage for Serenity research scorecards."""

import json
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from quanttool.cli.main import app


class SerenityCliTests(unittest.TestCase):
    """Verify the research CLI stays a thin, dependable adapter."""

    def setUp(self):
        self.runner = CliRunner()

    def scorecard_payload(self):
        return {
            "ticker": "688012.SH",
            "company": "中微公司",
            "market": "A-share",
            "theme": "AI semiconductors",
            "layer": "equipment",
            "role": "critical supplier",
            "factors": {
                "demand_inflection": 5.0,
                "architecture_coupling": 4.0,
                "chokepoint_severity": 4.0,
                "supplier_concentration": 3.0,
                "expansion_difficulty": 4.0,
                "evidence_quality": 4.0,
                "valuation_disconnect": 3.0,
                "catalyst_timing": 4.0,
            },
            "penalties": {"hype_risk": 1.0},
            "evidence": [
                {
                    "claim": "Demand remains constrained.",
                    "source": "Company annual report",
                    "strength": "strong",
                    "published_at": "2026-07-01",
                }
            ],
            "what_could_weaken_view": ["A substitute design gains adoption."],
            "timing_score": 82.0,
        }

    def assert_click_error(self, result):
        self.assertNotEqual(result.exit_code, 0, result.output)
        self.assertIn("Error", result.output)
        self.assertNotIn("Traceback", result.output)

    def test_template_prints_valid_json(self):
        result = self.runner.invoke(app, ["research", "template"])

        self.assertEqual(result.exit_code, 0, result.output)
        template = json.loads(result.output)
        self.assertEqual(template["market"], "A-share")
        self.assertEqual(template["factors"]["demand_inflection"], 0.0)

    def test_root_help_preserves_existing_command_names(self):
        result = self.runner.invoke(app, ["--help"])

        self.assertEqual(result.exit_code, 0, result.output)
        for command_name in (
            "data",
            "backtest",
            "analysis",
            "scheduler",
            "portfolio",
            "report",
            "monitor",
            "qlib",
            "enhanced",
            "research",
        ):
            with self.subTest(command_name=command_name):
                self.assertIn(command_name, result.output)

    def test_scorecard_reads_utf8_json_file_and_prints_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "scorecard.json"
            input_path.write_text(
                json.dumps(self.scorecard_payload(), ensure_ascii=False),
                encoding="utf-8",
            )

            result = self.runner.invoke(app, ["research", "scorecard", str(input_path)])

        self.assertEqual(result.exit_code, 0, result.output)
        score = json.loads(result.output)
        self.assertEqual(score["ticker"], "688012.SH")
        self.assertEqual(score["company"], "中微公司")
        self.assertIn("中微公司", result.output)
        self.assertEqual(score["evidence"][0]["published_at"], "2026-07-01")

    def test_scorecard_reads_standard_input_and_prints_markdown(self):
        result = self.runner.invoke(
            app,
            ["research", "scorecard", "-", "--format", "md"],
            input=json.dumps(self.scorecard_payload(), ensure_ascii=False),
        )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("# Serenity research scorecard: 688012.SH (中微公司)", result.output)
        self.assertIn("Research priority only. This is not a trading instruction.", result.output)

    def test_scorecard_both_format_separates_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = Path(tmp_dir) / "scorecard.json"
            input_path.write_text(json.dumps(self.scorecard_payload()), encoding="utf-8")

            result = self.runner.invoke(
                app,
                ["research", "scorecard", str(input_path), "--format", "both"],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        json_output, markdown = result.output.split("\n\n---\n\n", maxsplit=1)
        self.assertEqual(json.loads(json_output)["ticker"], "688012.SH")
        self.assertIn("## Research boundary", markdown)

    def test_scorecard_reports_malformed_json_without_traceback(self):
        result = self.runner.invoke(
            app,
            ["research", "scorecard", "-"],
            input="{not valid json",
        )

        self.assert_click_error(result)
        self.assertIn("Invalid JSON", result.output)

    def test_scorecard_reports_missing_file_without_traceback(self):
        result = self.runner.invoke(
            app,
            ["research", "scorecard", "does-not-exist.json"],
        )

        self.assert_click_error(result)
        self.assertIn("Could not read input", result.output)

    def test_scorecard_reports_model_validation_without_traceback(self):
        payload = self.scorecard_payload()
        payload["factors"]["demand_inflection"] = 5.01

        result = self.runner.invoke(
            app,
            ["research", "scorecard", "-"],
            input=json.dumps(payload),
        )

        self.assert_click_error(result)
        self.assertIn("Invalid scorecard", result.output)


if __name__ == "__main__":
    unittest.main()
