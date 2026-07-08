"""Regression tests for frontend and CLI optimization work."""

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import click
from typer.testing import CliRunner


REPO_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_ROOT = REPO_ROOT / "quanttool" / "web" / "frontend"


class CliOptimizationTests(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop("quanttool.cli.main", None)
        sys.modules.pop("quanttool.factors.stock_analyzer", None)

    def test_cli_main_import_does_not_import_stock_analyzer(self):
        sys.modules.pop("quanttool.cli.main", None)
        sys.modules.pop("quanttool.factors.stock_analyzer", None)

        importlib.import_module("quanttool.cli.main")

        self.assertFalse(
            "quanttool.factors.stock_analyzer" in sys.modules,
            "stock_analyzer should not be imported at CLI import time",
        )

    def test_quick_analyze_uses_unified_context_and_writes_output(self):
        calls = []

        class FakeScore:
            score = 66.0
            final_score = 72.0
            passed_hard_filter = True
            timing_type = "趋势运行"
            passed_filter = False
            filter_reason = "无突破信号"

        class FakeRecommendation:
            def get_action_display(self):
                return "买入"

        class FakeContext:
            classic_score = FakeScore()
            trend_score = FakeScore()
            breakout_score = FakeScore()
            final_recommendation = FakeRecommendation()

        class FakeAnalyzer:
            def analyze_stock_with_context(self, symbol, days):
                calls.append((symbol, days))
                return FakeContext(), "REPORT BODY"

        fake_module = types.ModuleType("quanttool.factors.stock_analyzer")
        fake_module.StockAnalyzer = FakeAnalyzer

        with patch.dict(sys.modules, {"quanttool.factors.stock_analyzer": fake_module}):
            sys.modules.pop("quanttool.cli.main", None)
            cli_main = importlib.import_module("quanttool.cli.main")
            runner = CliRunner()
            with tempfile.TemporaryDirectory() as tmp_dir:
                output = Path(tmp_dir) / "report.md"
                result = runner.invoke(
                    cli_main.app,
                    ["analyze", "000001", "--days", "120", "--output", str(output)],
                )
                saved_report = output.read_text(encoding="utf-8")

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(calls, [("000001", 120)])
        self.assertIn("=== 三系统评分摘要 ===", result.output)
        self.assertIn("最终推荐: 买入", result.output)
        self.assertIn("REPORT BODY", result.output)
        self.assertEqual(saved_report, "REPORT BODY")

    def test_quick_analyze_converts_failures_to_click_exception(self):
        class FakeAnalyzer:
            def analyze_stock_with_context(self, symbol, days):
                raise RuntimeError("boom")

        fake_module = types.ModuleType("quanttool.factors.stock_analyzer")
        fake_module.StockAnalyzer = FakeAnalyzer

        with patch.dict(sys.modules, {"quanttool.factors.stock_analyzer": fake_module}):
            sys.modules.pop("quanttool.cli.main", None)
            cli_main = importlib.import_module("quanttool.cli.main")
            with self.assertRaises(click.ClickException) as ctx:
                cli_main.analyze("000001", days=120, output=None)

        self.assertIn("boom", str(ctx.exception))


class FrontendOptimizationSourceTests(unittest.TestCase):
    def test_api_client_uses_configurable_base_url(self):
        source = (FRONTEND_ROOT / "lib" / "api" / "index.ts").read_text(encoding="utf-8")

        self.assertIn("export function getApiBaseUrl", source)
        self.assertIn("NEXT_PUBLIC_API_BASE_URL", source)
        self.assertIn("http://localhost:8000/api", source)
        self.assertIn("baseURL: getApiBaseUrl()", source)

    def test_no_hardcoded_localhost_api_host_outside_config(self):
        allowed = {
            FRONTEND_ROOT / "lib" / "api" / "index.ts",
            FRONTEND_ROOT / "lib" / "api.ts",
            FRONTEND_ROOT / "next.config.js",
        }
        offenders = []
        for path in FRONTEND_ROOT.rglob("*"):
            if path.suffix not in {".ts", ".tsx", ".js"}:
                continue
            if "node_modules" in path.parts or ".next" in path.parts:
                continue
            if path in allowed:
                continue
            if "http://localhost:8000" in path.read_text(encoding="utf-8"):
                offenders.append(str(path.relative_to(REPO_ROOT)))

        self.assertEqual(offenders, [])

    def test_overview_action_color_classes_are_static(self):
        source = (FRONTEND_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")

        self.assertIn("ACTION_COLOR_CLASSES", source)
        self.assertNotIn("bg-${action.color}", source)
        self.assertNotIn("text-${action.color}", source)
        self.assertIn("marketError", source)

    def test_navigation_active_state_is_path_derived(self):
        navigation = (FRONTEND_ROOT / "lib" / "navigation.ts").read_text(encoding="utf-8")
        header = (FRONTEND_ROOT / "components" / "layout" / "AppHeader.tsx").read_text(encoding="utf-8")
        sidebar = (FRONTEND_ROOT / "components" / "layout" / "AppSidebar.tsx").read_text(encoding="utf-8")

        self.assertIn("export function getPageKeyFromPath", navigation)
        self.assertIn("usePathname", header)
        self.assertIn("getPageKeyFromPath", header)
        self.assertIn("usePathname", sidebar)
        self.assertIn("getPageKeyFromPath", sidebar)


if __name__ == "__main__":
    unittest.main()
