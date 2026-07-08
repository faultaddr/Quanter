"""Smoke tests for QuantTool project structure."""

from pathlib import Path
import unittest


class ImportSmokeTests(unittest.TestCase):
    def test_fastapi_app_imports(self):
        from quanttool.web.app import app

        self.assertEqual(app.title, "QuantTool API")

    def test_cli_app_imports(self):
        from quanttool.cli.main import app

        self.assertIsNotNone(app)

    def test_analysis_service_imports(self):
        from quanttool.application.analysis_service import AnalysisService

        self.assertIsNotNone(AnalysisService)


class ApiRouteContractTests(unittest.TestCase):
    def test_core_api_routes_are_registered(self):
        from quanttool.web.app import app

        routes = {
            (method, route.path)
            for route in app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method not in {"HEAD", "OPTIONS"}
        }

        expected = {
            ("POST", "/api/backtest/run"),
            ("POST", "/api/gbm/train"),
            ("GET", "/api/realtime/search"),
            ("GET", "/api/stock/{symbol}/analysis"),
        }

        self.assertTrue(expected.issubset(routes))

    def test_api_route_paths_remain_unique_per_method(self):
        from quanttool.web.app import app

        seen = set()
        duplicates = []
        for route in app.routes:
            if not hasattr(route, "methods"):
                continue
            for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
                key = (method, route.path)
                if key in seen:
                    duplicates.append(key)
                seen.add(key)

        self.assertEqual(duplicates, [])


class PackagingSmokeTests(unittest.TestCase):
    def test_pyproject_discovers_quanttool_subpackages(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("[tool.setuptools.packages.find]", text)
        self.assertIn('include = ["quanttool*"]', text)
        self.assertNotIn('packages = ["quanttool"]', text)


if __name__ == "__main__":
    unittest.main()
