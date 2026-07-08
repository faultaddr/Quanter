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


class ApiStructureTests(unittest.TestCase):
    def test_model_api_router_is_split_into_focused_modules(self):
        api_dir = Path("quanttool/web/api")
        aggregate = api_dir / "models.py"
        route_dir = api_dir / "model_routes"

        self.assertLessEqual(
            len(aggregate.read_text(encoding="utf-8").splitlines()),
            120,
        )
        for module_name in [
            "__init__.py",
            "discovery.py",
            "gbm.py",
            "qlib_training.py",
            "qlib_prediction.py",
        ]:
            self.assertTrue((route_dir / module_name).is_file(), module_name)

    def test_model_api_routes_remain_registered(self):
        from quanttool.web.app import app

        routes = {
            (method, route.path)
            for route in app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method not in {"HEAD", "OPTIONS"}
        }

        expected = {
            ("GET", "/api/qlib/models"),
            ("GET", "/api/qlib/saved-models"),
            ("GET", "/api/qlib/pretrained-models"),
            ("GET", "/api/qlib/all-models"),
            ("GET", "/api/qlib/saved-models/{model_id}"),
            ("GET", "/api/qlib/models/categories"),
            ("POST", "/api/qlib/train"),
            ("POST", "/api/qlib/train/stream"),
            ("POST", "/api/qlib/predict"),
            ("POST", "/api/gbm/train"),
            ("POST", "/api/gbm/predict"),
            ("GET", "/api/gbm/models"),
            ("DELETE", "/api/gbm/models/{model_id}"),
            ("GET", "/api/gbm/train/{task_id}/progress"),
            ("GET", "/api/gbm/qrun-models"),
            ("POST", "/api/gbm/picks"),
        }

        self.assertTrue(expected.issubset(routes))

    def test_stock_api_router_is_split_into_focused_modules(self):
        api_dir = Path("quanttool/web/api")
        aggregate = api_dir / "stock.py"
        route_dir = api_dir / "stock_routes"

        self.assertLessEqual(
            len(aggregate.read_text(encoding="utf-8").splitlines()),
            120,
        )
        for module_name in [
            "__init__.py",
            "analysis.py",
            "market_data.py",
            "chip_signals.py",
            "insights.py",
        ]:
            self.assertTrue((route_dir / module_name).is_file(), module_name)

    def test_stock_api_routes_remain_registered(self):
        from quanttool.web.app import app

        routes = {
            (method, route.path)
            for route in app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method not in {"HEAD", "OPTIONS"}
        }

        expected = {
            ("POST", "/api/analyze"),
            ("POST", "/api/analyze/enhanced"),
            ("GET", "/api/stock/{symbol}/info"),
            ("GET", "/api/stock/{symbol}/kline"),
            ("GET", "/api/stock/{symbol}/chip"),
            ("GET", "/api/stock/{symbol}/signals"),
            ("GET", "/api/stock/{symbol}/analysis"),
            ("GET", "/api/stock/{symbol}/flow"),
            ("GET", "/api/stock/{symbol}/risk"),
            ("GET", "/api/stock/{symbol}/factors"),
            ("GET", "/api/stock/{symbol}/feasibility"),
            ("GET", "/api/stock/{symbol}/backtest-compare"),
            ("GET", "/api/index/{index_code}/data"),
        }

        self.assertTrue(expected.issubset(routes))

    def test_qlib_training_api_router_is_split_into_focused_modules(self):
        api_dir = Path("quanttool/web/api/model_routes")
        aggregate = api_dir / "qlib_training.py"
        route_dir = api_dir / "qlib_training_routes"

        self.assertLessEqual(
            len(aggregate.read_text(encoding="utf-8").splitlines()),
            120,
        )
        for module_name in [
            "__init__.py",
            "batch.py",
            "stream.py",
        ]:
            self.assertTrue((route_dir / module_name).is_file(), module_name)

    def test_qlib_training_api_routes_remain_registered(self):
        from quanttool.web.app import app

        routes = {
            (method, route.path)
            for route in app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method not in {"HEAD", "OPTIONS"}
        }

        expected = {
            ("POST", "/api/qlib/train"),
            ("POST", "/api/qlib/train/stream"),
        }

        self.assertTrue(expected.issubset(routes))

    def test_backtest_api_router_is_split_into_focused_modules(self):
        api_dir = Path("quanttool/web/api")
        aggregate = api_dir / "backtest.py"
        route_dir = api_dir / "backtest_routes"

        self.assertLessEqual(
            len(aggregate.read_text(encoding="utf-8").splitlines()),
            120,
        )
        for module_name in [
            "__init__.py",
            "catalog.py",
            "execution.py",
            "comparison.py",
            "stream.py",
            "experiments.py",
        ]:
            self.assertTrue((route_dir / module_name).is_file(), module_name)

    def test_ml_api_router_is_split_into_focused_modules(self):
        api_dir = Path("quanttool/web/api")
        aggregate = api_dir / "ml.py"
        route_dir = api_dir / "ml_routes"

        self.assertLessEqual(
            len(aggregate.read_text(encoding="utf-8").splitlines()),
            120,
        )
        for module_name in [
            "__init__.py",
            "backtest.py",
            "scan.py",
            "monitor.py",
        ]:
            self.assertTrue((route_dir / module_name).is_file(), module_name)

    def test_backtest_and_ml_api_routes_remain_registered(self):
        from quanttool.web.app import app

        routes = {
            (method, route.path)
            for route in app.routes
            if hasattr(route, "methods")
            for method in route.methods
            if method not in {"HEAD", "OPTIONS"}
        }

        expected = {
            ("GET", "/api/backtest/strategies"),
            ("GET", "/api/backtest/history"),
            ("POST", "/api/backtest/run"),
            ("POST", "/api/backtest/run-all"),
            ("POST", "/api/backtest/run-all-stream"),
            ("GET", "/api/experiments"),
            ("GET", "/api/backtest/runs/{run_id}"),
            ("POST", "/api/ml/backtest"),
            ("POST", "/api/ml/scan"),
            ("POST", "/api/ml/monitor/start"),
            ("GET", "/api/ml/monitor/{monitor_id}/signals"),
        }

        self.assertTrue(expected.issubset(routes))


class PackagingSmokeTests(unittest.TestCase):
    def test_pyproject_discovers_quanttool_subpackages(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")

        self.assertIn("[tool.setuptools.packages.find]", text)
        self.assertIn('include = ["quanttool*"]', text)
        self.assertNotIn('packages = ["quanttool"]', text)


if __name__ == "__main__":
    unittest.main()
