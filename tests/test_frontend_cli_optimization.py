"""Regression tests for frontend and CLI optimization work."""

import importlib
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import click
from fastapi import HTTPException
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
    def test_browser_api_clients_default_to_same_origin_proxy(self):
        api_client_paths = (
            FRONTEND_ROOT / "lib" / "api" / "index.ts",
            FRONTEND_ROOT / "lib" / "api.ts",
        )

        for path in api_client_paths:
            source = path.read_text(encoding="utf-8")

            self.assertIn("export function getApiBaseUrl", source)
            self.assertIn("NEXT_PUBLIC_API_BASE_URL", source)
            self.assertIn("|| '/api'", source)
            self.assertNotIn("http://localhost:8000/api", source)
            self.assertIn("baseURL: getApiBaseUrl()", source)

    def test_next_rewrites_use_server_only_api_proxy_base_url(self):
        source = (FRONTEND_ROOT / "next.config.js").read_text(encoding="utf-8")

        self.assertIn("API_PROXY_BASE_URL", source)
        self.assertNotIn("NEXT_PUBLIC_API_BASE_URL", source)
        self.assertIn("http://localhost:8000/api", source)
        self.assertIn("destination: `${apiProxyBaseUrl}/:path*`", source)

    def test_standalone_start_prepares_next_static_assets(self):
        package_source = (FRONTEND_ROOT / "package.json").read_text(encoding="utf-8")
        prepare_script = FRONTEND_ROOT / "scripts" / "prepare-standalone.js"

        self.assertTrue(prepare_script.exists(), "standalone asset preparation script should exist")
        self.assertIn('"prepare:standalone": "node scripts/prepare-standalone.js"', package_source)
        self.assertIn('"start": "npm run prepare:standalone && node .next/standalone/server.js"', package_source)
        script_source = prepare_script.read_text(encoding="utf-8")
        self.assertIn(".next/static", script_source)
        self.assertIn(".next/standalone/.next/static", script_source)

    def test_frontend_api_base_urls_trim_trailing_slashes(self):
        index_source = (FRONTEND_ROOT / "lib" / "api" / "index.ts").read_text(encoding="utf-8")
        api_source = (FRONTEND_ROOT / "lib" / "api.ts").read_text(encoding="utf-8")
        next_config_source = (FRONTEND_ROOT / "next.config.js").read_text(encoding="utf-8")

        for source in (index_source, api_source, next_config_source):
            self.assertIn("normalizeApiBaseUrl(", source)
            self.assertIn(r"replace(/\/+$/, '')", source)

    def test_no_hardcoded_localhost_api_host_outside_config(self):
        allowed = {
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

        self.assertIn("quickActions", source)
        self.assertIn("candidatePrompts", source)
        self.assertIn("PageHeader", source)
        self.assertIn("MetricTile", source)
        self.assertIn("Section", source)
        self.assertNotIn("bg-${action.color}", source)
        self.assertNotIn("text-${action.color}", source)
        self.assertIn("marketError", source)
        self.assertIn("刷新指数", source)
        self.assertIn("border-warning/30", source)
        self.assertIn("bg-warning/10", source)
        self.assertIn("重新加载", source)

    def test_navigation_active_state_is_path_derived(self):
        navigation = (FRONTEND_ROOT / "lib" / "navigation.ts").read_text(encoding="utf-8")
        header = (FRONTEND_ROOT / "components" / "layout" / "AppHeader.tsx").read_text(encoding="utf-8")
        sidebar = (FRONTEND_ROOT / "components" / "layout" / "AppSidebar.tsx").read_text(encoding="utf-8")

        self.assertIn("export function getPageKeyFromPath", navigation)
        self.assertIn("StockSearch", header)
        self.assertIn("toggleSidebar", header)
        self.assertIn("本地模式", header)
        self.assertNotIn("const navItems", header)
        self.assertNotIn("secondaryNavItems", header)
        self.assertIn("navigationGroups", sidebar)
        self.assertIn("usePathname", sidebar)
        self.assertIn("getPageKeyFromPath", sidebar)

    def test_backtest_stream_source_handles_missing_body_and_reader_failures(self):
        source = (FRONTEND_ROOT / "lib" / "api" / "backtest.ts").read_text(encoding="utf-8")

        self.assertIn("if (!response.body)", source)
        self.assertIn("onError('后端未返回流式响应')", source)
        self.assertIn("return read();", source)
        self.assertIn("await reader.read()", source)
        self.assertIn("catch (err)", source)
        self.assertIn("流式回调失败", source)

    def test_backtest_page_does_not_double_prefix_signed_percentages(self):
        source = (FRONTEND_ROOT / "app" / "backtest" / "page.tsx").read_text(encoding="utf-8")

        self.assertNotIn("{result.excess_return > 0 ? '+' : ''}{formatPercent(result.excess_return)}", source)
        self.assertIn("{formatPercent(result.excess_return)}", source)

    def test_scan_api_unwraps_backend_response_results(self):
        source = (FRONTEND_ROOT / "lib" / "api" / "monitor.ts").read_text(encoding="utf-8")

        self.assertIn("interface ScanApiResponse", source)
        self.assertIn("normalizeScanResult", source)
        self.assertIn("result.close", source)
        self.assertIn("buildFallbackScanSignals(result)", source)
        self.assertIn("use_unified_score?: boolean", source)
        self.assertIn("include_fundamentals?: boolean", source)
        self.assertIn("include_market_state?: boolean", source)
        self.assertIn("api.post<any, ScanApiResponse>('/scan', params)", source)
        self.assertIn("return (response.results || []).map(normalizeScanResult)", source)

    def test_scan_page_uses_supported_markets_and_empty_state(self):
        source = (FRONTEND_ROOT / "app" / "scan" / "page.tsx").read_text(encoding="utf-8")

        self.assertIn("useState('csi300')", source)
        self.assertIn("value: 'csi300'", source)
        self.assertIn("value: 'csi1000'", source)
        self.assertNotIn("value: 'sh'", source)
        self.assertNotIn("value: 'sz'", source)
        self.assertNotIn("value: 'bj'", source)
        self.assertIn("hasScanned", source)
        self.assertIn("暂无符合条件股票", source)

    def test_scan_page_exposes_scoring_mode_selection(self):
        source = (FRONTEND_ROOT / "app" / "scan" / "page.tsx").read_text(encoding="utf-8")

        self.assertIn("type ScoringMode = 'unified' | 'trend' | 'classic' | 'breakout' | 'momentum'", source)
        self.assertIn("useState<ScoringMode>('unified')", source)
        self.assertIn("use_unified_score: scoringMode === 'unified'", source)
        self.assertIn("use_trend_score: scoringMode === 'trend'", source)
        self.assertIn("use_breakout_score: scoringMode === 'breakout'", source)
        self.assertIn("use_momentum_score: scoringMode === 'momentum'", source)
        self.assertIn("const [includeFundamentals, setIncludeFundamentals] = useState(false)", source)
        self.assertIn("include_fundamentals: includeFundamentals", source)
        self.assertIn("include_market_state: includeFundamentals", source)
        self.assertIn("htmlFor=\"includeFundamentals\"", source)

    def test_qrun_models_use_path_as_selectable_model_id(self):
        source = (FRONTEND_ROOT / "lib" / "api" / "model.ts").read_text(encoding="utf-8")

        self.assertIn("function transformQrunModelData", source)
        self.assertIn("id: apiModel.path", source)
        self.assertIn("apiModel.run_name || apiModel.run_id || apiModel.path", source)
        self.assertIn("map(transformQrunModelData)", source)

    def test_picks_api_sends_backend_fields_and_maps_top_stocks(self):
        source = (FRONTEND_ROOT / "lib" / "api" / "monitor.ts").read_text(encoding="utf-8")

        self.assertIn("interface PicksApiResponse", source)
        self.assertIn("normalizePickResult", source)
        self.assertIn("top_n: topK", source)
        self.assertIn("model_path: modelId || undefined", source)
        self.assertIn("return (response.top_stocks || []).map(normalizePickResult)", source)

    def test_model_api_matches_sync_train_and_model_path_predict_contracts(self):
        source = (FRONTEND_ROOT / "lib" / "api" / "model.ts").read_text(encoding="utf-8")

        self.assertIn("interface TrainResponse", source)
        self.assertIn("model_id?: string", source)
        self.assertIn("model_path?: string", source)
        self.assertIn("api.post<any, TrainResponse>('/gbm/train', params)", source)
        self.assertIn("interface PredictApiResponse", source)
        self.assertIn("model_path: modelId", source)
        self.assertIn("return (response.predictions || []).map(normalizePredictionResult)", source)

    def test_model_page_handles_sync_train_completion_and_predicts_with_model_path(self):
        page_source = (FRONTEND_ROOT / "app" / "model" / "page.tsx").read_text(encoding="utf-8")
        card_source = (FRONTEND_ROOT / "components" / "model" / "ModelCard.tsx").read_text(encoding="utf-8")

        self.assertIn("result.model_id", page_source)
        self.assertIn("setTrainingProgress({ status: 'completed', progress: 100", page_source)
        self.assertIn("loadModels();", page_source)
        self.assertIn("onPredict(model.path || model.id)", card_source)

    def test_factors_page_maps_flat_backend_scores_to_table_rows(self):
        source = (FRONTEND_ROOT / "app" / "factors" / "page.tsx").read_text(encoding="utf-8")

        self.assertIn("interface BackendFactorResponse", source)
        self.assertIn("const FACTOR_SCORE_LABELS", source)
        self.assertIn("buildFactorScores(data)", source)
        self.assertIn("data.momentum", source)
        self.assertIn("data.value", source)
        self.assertIn("data.quality", source)
        self.assertIn("data.growth", source)
        self.assertIn("factor_scores: buildFactorScores(data)", source)

    def test_risk_page_uses_real_portfolio_check_api(self):
        source = (FRONTEND_ROOT / "app" / "risk" / "page.tsx").read_text(encoding="utf-8")

        self.assertNotIn("mockRiskCheck", source)
        self.assertIn("interface BackendRiskResponse", source)
        self.assertIn("buildRiskRequest(symbols, positions)", source)
        self.assertIn("fetch('/api/risk/portfolio/check'", source)
        self.assertIn("mapRiskResponse(data, requestBody)", source)
        self.assertIn("positions: Object.fromEntries", source)

    def test_research_api_unwraps_success_envelope_and_rejects_invalid_responses(self):
        path = FRONTEND_ROOT / "lib" / "api" / "research.ts"

        self.assertTrue(path.exists(), "research API client should exist")
        source = path.read_text(encoding="utf-8")
        self.assertIn("'/research/serenity/scorecard'", source)
        self.assertIn("response.data", source)
        self.assertIn("!response.success", source)
        self.assertIn("!response.data", source)
        self.assertIn("throw new Error", source)
        self.assertIn("isAxiosError", source)
        self.assertIn("response?.data?.detail", source)

    def test_research_route_exposes_all_serenity_factors_and_research_only_copy(self):
        path = FRONTEND_ROOT / "app" / "research" / "page.tsx"

        self.assertTrue(path.exists(), "research workbench route should exist")
        source = path.read_text(encoding="utf-8")
        factor_keys = (
            "demand_inflection",
            "architecture_coupling",
            "chokepoint_severity",
            "supplier_concentration",
            "expansion_difficulty",
            "evidence_quality",
            "valuation_disconnect",
            "catalyst_timing",
        )
        for factor_key in factor_keys:
            self.assertIn(factor_key, source)

        penalty_keys = (
            "dilution_financing",
            "governance",
            "geopolitics",
            "liquidity",
            "hype_risk",
            "accounting_quality",
            "cyclicality",
            "alternative_design_risk",
        )
        for penalty_key in penalty_keys:
            self.assertIn(penalty_key, source)

        self.assertIn("from '@/lib/api/research'", source)
        self.assertIn("研究优先级不是交易建议", source)
        self.assertIn("研究优先级", source)
        self.assertIn("交易时机", source)
        self.assertIn("不参与研究优先级计算", source)

    def test_research_retry_clears_stale_result_and_dynamic_rows_use_stable_keys(self):
        source = (FRONTEND_ROOT / "app" / "research" / "page.tsx").read_text(encoding="utf-8")

        submit_start = source.index("const handleSubmit")
        request_start = source.index("researchApi.scorecard", submit_start)
        self.assertIn("setResult(null)", source[submit_start:request_start])
        self.assertNotIn("key={index}", source)
        self.assertIn("evidenceRowIds.current[index]", source)
        self.assertIn("weakeningRowIds.current[index]", source)

    def test_research_result_details_are_keyed_by_the_exact_score_contract(self):
        source = (FRONTEND_ROOT / "types" / "research.ts").read_text(encoding="utf-8")

        self.assertIn("Record<keyof SerenityFactors, SerenityScoreDetail>", source)
        self.assertIn("Record<keyof SerenityPenalties, SerenityScoreDetail>", source)

    def test_shared_input_associates_visible_labels_and_errors_with_the_control(self):
        source = (FRONTEND_ROOT / "components" / "ui" / "Input.tsx").read_text(encoding="utf-8")

        self.assertIn("useId", source)
        self.assertIn("htmlFor={inputId}", source)
        self.assertIn("id={inputId}", source)
        self.assertIn("aria-invalid={Boolean(error)}", source)
        self.assertIn("id={errorId}", source)

    def test_research_success_result_is_announced_without_moving_form_state(self):
        source = (FRONTEND_ROOT / "app" / "research" / "page.tsx").read_text(encoding="utf-8")

        self.assertIn('aria-live="polite"', source)
        self.assertNotIn("setScorecard(createInitialScorecard())", source)

    def test_research_score_controls_use_stable_half_point_scale(self):
        path = FRONTEND_ROOT / "components" / "research" / "ScoreField.tsx"

        self.assertTrue(path.exists(), "research score control should exist")
        source = path.read_text(encoding="utf-8")
        self.assertIn("min={0}", source)
        self.assertIn("max={5}", source)
        self.assertIn("step={0.5}", source)
        self.assertIn("value.toFixed(1)", source)

    def test_research_navigation_is_path_derived_and_visible_in_sidebar(self):
        navigation = (FRONTEND_ROOT / "lib" / "navigation.ts").read_text(encoding="utf-8")
        sidebar = (FRONTEND_ROOT / "components" / "layout" / "AppSidebar.tsx").read_text(encoding="utf-8")

        self.assertIn("['/research', 'research']", navigation)
        self.assertIn("key: 'research'", sidebar)
        self.assertIn("label: '产业链研究'", sidebar)
        self.assertIn("href: '/research'", sidebar)

    def test_mobile_navigation_uses_an_overlay_drawer_instead_of_shrinking_content(self):
        sidebar = (FRONTEND_ROOT / "components" / "layout" / "AppSidebar.tsx").read_text(encoding="utf-8")
        header = (FRONTEND_ROOT / "components" / "layout" / "AppHeader.tsx").read_text(encoding="utf-8")
        store = (FRONTEND_ROOT / "stores" / "useAppStore.ts").read_text(encoding="utf-8")

        self.assertIn("mobileSidebarOpen", store)
        self.assertIn("toggleMobileSidebar", header)
        self.assertIn("window.matchMedia('(max-width: 767px)')", header)
        self.assertIn("fixed bottom-0 left-0 top-14", sidebar)
        self.assertIn("-translate-x-full", sidebar)
        self.assertIn("md:static", sidebar)
        self.assertIn('aria-label="关闭导航"', sidebar)
        self.assertIn("closeMobileSidebar", sidebar)
        self.assertIn("aria-controls=\"app-navigation\"", header)
        self.assertIn("aria-expanded={mobileSidebarOpen}", header)
        self.assertIn("mobileDrawerHidden", sidebar)
        self.assertIn("aria-hidden={mobileDrawerHidden}", sidebar)
        self.assertIn("tabIndex={mobileDrawerHidden ? -1 : undefined}", sidebar)
        self.assertIn("event.key === 'Escape'", sidebar)
        self.assertIn("showLabels", sidebar)


class WebScanApiOptimizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unsupported_scan_market_preserves_400_response(self):
        from quanttool.web.api.scan import scan_stocks
        from quanttool.web.schemas.scan import ScanRequest

        with self.assertRaises(HTTPException) as ctx:
            await scan_stocks(ScanRequest(market="sh"))

        self.assertEqual(ctx.exception.status_code, 400)

    async def test_empty_scan_returns_before_initializing_analyzer(self):
        from quanttool.cli.commands import analysis_commands
        from quanttool.web.api.scan import scan_stocks
        from quanttool.web.schemas.scan import ScanRequest

        class UnexpectedAnalyzer:
            def __init__(self, *args, **kwargs):
                raise AssertionError("StockAnalyzer should not be initialized for an empty stock list")

        fake_module = types.ModuleType("quanttool.factors.stock_analyzer")
        fake_module.StockAnalyzer = UnexpectedAnalyzer

        with patch.dict(sys.modules, {"quanttool.factors.stock_analyzer": fake_module}):
            with patch.object(analysis_commands, "get_csi300_constituents", return_value=[]):
                response = await scan_stocks(ScanRequest(market="csi300", top_n=3))

        self.assertEqual(response["total_stocks"], 0)
        self.assertEqual(response["analyzed_stocks"], 0)
        self.assertEqual(response["results"], [])

    async def test_scan_scores_stocks_concurrently_after_preload(self):
        from quanttool.cli.commands import analysis_commands
        from quanttool.web.api.scan import scan_stocks
        from quanttool.web.schemas.scan import ScanRequest

        stock_list = [
            {"code": "000001", "name": "平安银行"},
            {"code": "000002", "name": "万科A"},
            {"code": "600000", "name": "浦发银行"},
            {"code": "600519", "name": "贵州茅台"},
        ]
        active_count = 0
        max_active_count = 0
        counter_lock = threading.Lock()
        preloaded_symbols = []
        realtime_symbols = []

        class FakeAnalyzer:
            def __init__(self, *args, **kwargs):
                pass

            def preload_data_for_scan(self, stocks, days):
                preloaded_symbols.extend(stock["code"] for stock in stocks)
                return len(stocks)

            def preload_realtime_prices(self, symbols):
                realtime_symbols.extend(symbols)
                return len(symbols)

        def fake_trend_score(stock_info, days, analyzer, start_date, end_date):
            nonlocal active_count, max_active_count
            with counter_lock:
                active_count += 1
                max_active_count = max(max_active_count, active_count)
            time.sleep(0.05)
            with counter_lock:
                active_count -= 1
            return {
                "symbol": stock_info["code"],
                "name": stock_info["name"],
                "score": float(len(stock_info["code"])),
            }, None

        fake_module = types.ModuleType("quanttool.factors.stock_analyzer")
        fake_module.StockAnalyzer = FakeAnalyzer

        with patch.dict(sys.modules, {"quanttool.factors.stock_analyzer": fake_module}):
            with patch.object(analysis_commands, "get_csi300_constituents", return_value=stock_list):
                with patch.object(analysis_commands, "analyze_stock_trend_score", side_effect=fake_trend_score):
                    response = await scan_stocks(ScanRequest(market="csi300", top_n=2))

        self.assertEqual(preloaded_symbols, ["000001", "000002", "600000", "600519"])
        self.assertEqual(realtime_symbols, ["000001", "000002", "600000", "600519"])
        self.assertGreater(max_active_count, 1)
        self.assertEqual(response["total_stocks"], 4)
        self.assertEqual(response["analyzed_stocks"], 4)
        self.assertEqual(len(response["results"]), 2)

    async def test_scan_can_route_to_unified_scoring(self):
        from quanttool.cli.commands import analysis_commands
        from quanttool.web.api.scan import scan_stocks
        from quanttool.web.schemas.scan import ScanRequest

        stock_list = [
            {"code": "000001", "name": "平安银行"},
            {"code": "000002", "name": "万科A"},
        ]
        routed_symbols = []

        class FakeAnalyzer:
            def __init__(self, *args, **kwargs):
                pass

            def preload_data_for_scan(self, stocks, days):
                return len(stocks)

            def preload_realtime_prices(self, symbols):
                return len(symbols)

        def fake_unified_score(stock_info, days, analyzer, start_date, end_date):
            routed_symbols.append(stock_info["code"])
            return {
                "symbol": stock_info["code"],
                "name": stock_info["name"],
                "close": 10.0,
                "score": 80.0 if stock_info["code"] == "000001" else 70.0,
                "trigger_type": "unified",
            }, None

        fake_module = types.ModuleType("quanttool.factors.stock_analyzer")
        fake_module.StockAnalyzer = FakeAnalyzer

        with patch.dict(sys.modules, {"quanttool.factors.stock_analyzer": fake_module}):
            with patch.object(analysis_commands, "get_csi300_constituents", return_value=stock_list):
                with patch.object(analysis_commands, "analyze_stock_unified_score", side_effect=fake_unified_score) as unified_mock:
                    with patch.object(analysis_commands, "analyze_stock_trend_score") as trend_mock:
                        response = await scan_stocks(
                            ScanRequest(
                                market="csi300",
                                top_n=2,
                                use_unified_score=True,
                            )
                        )

        self.assertCountEqual(routed_symbols, ["000001", "000002"])
        self.assertEqual(unified_mock.call_count, 2)
        trend_mock.assert_not_called()
        self.assertEqual(response["scoring_mode"], "unified")
        self.assertEqual([item["trigger_type"] for item in response["results"]], ["unified", "unified"])

    async def test_unified_scan_skips_deep_fundamentals_by_default(self):
        from quanttool.cli.commands import analysis_commands
        from quanttool.factors.analysis_context import FundamentalData, UnifiedMarketState
        from quanttool.web.api.scan import scan_stocks
        from quanttool.web.schemas.scan import ScanRequest

        stock_list = [{"code": "000001", "name": "平安银行"}]
        observed_provider_results = []
        observed_market_states = []
        observed_fast_mode = []

        class FakeAnalyzer:
            def __init__(self, *args, **kwargs):
                self.analysis_orchestrator = types.SimpleNamespace(
                    fundamental_provider=None,
                    market_state_builder=None,
                )

            def preload_data_for_scan(self, stocks, days):
                return len(stocks)

            def preload_realtime_prices(self, symbols):
                return len(symbols)

        def fake_unified_score(stock_info, days, analyzer, start_date, end_date):
            provider = analyzer.analysis_orchestrator.fundamental_provider
            observed_provider_results.append(provider(stock_info["code"]) if provider else None)
            market_state_builder = analyzer.analysis_orchestrator.market_state_builder
            observed_market_states.append(market_state_builder(None) if market_state_builder else None)
            observed_fast_mode.append(getattr(analyzer, "_scan_fast_mode", False))
            return {
                "symbol": stock_info["code"],
                "name": stock_info["name"],
                "close": 10.0,
                "score": 80.0,
                "trigger_type": "unified",
            }, None

        fake_module = types.ModuleType("quanttool.factors.stock_analyzer")
        fake_module.StockAnalyzer = FakeAnalyzer

        with patch.dict(sys.modules, {"quanttool.factors.stock_analyzer": fake_module}):
            with patch.object(analysis_commands, "get_csi300_constituents", return_value=stock_list):
                with patch.object(analysis_commands, "analyze_stock_unified_score", side_effect=fake_unified_score):
                    response = await scan_stocks(
                        ScanRequest(market="csi300", top_n=1, use_unified_score=True)
                    )

        self.assertEqual(response["analyzed_stocks"], 1)
        self.assertEqual(len(observed_provider_results), 1)
        self.assertIsInstance(observed_provider_results[0], FundamentalData)
        self.assertEqual(observed_provider_results[0].data_source, "")
        self.assertIsInstance(observed_market_states[0], UnifiedMarketState)
        self.assertEqual(observed_market_states[0].index_code, "000300.SH")
        self.assertEqual(observed_fast_mode, [True])


class RealtimeProviderPerformanceTests(unittest.TestCase):
    def tearDown(self):
        from quanttool.infrastructure.data_providers.realtime.types import get_realtime_circuit_breaker

        get_realtime_circuit_breaker().reset()

    def test_pytdx_limits_failed_connection_probe_and_cools_down(self):
        from quanttool.infrastructure.data_providers.realtime.pytdx_source import PytdxRealtimeProvider
        from quanttool.infrastructure.data_providers.realtime.types import get_realtime_circuit_breaker

        get_realtime_circuit_breaker().reset()
        connect_calls = []

        class FakeApi:
            def connect(self, host, port, time_out=5):
                connect_calls.append((host, port, time_out))
                return False

            def disconnect(self):
                pass

        provider = PytdxRealtimeProvider(
            hosts=[("h1", 7709), ("h2", 7709), ("h3", 7709)],
            connect_timeout=0.25,
            max_hosts_per_attempt=2,
            failure_cooldown_seconds=60,
        )

        with patch.object(provider, "_get_pytdx_api", return_value=FakeApi):
            self.assertEqual(provider.get_realtime_quotes(["000001", "000002"]), {})
            self.assertEqual(provider.get_realtime_quotes(["000001", "000002"]), {})

        self.assertEqual(connect_calls, [("h1", 7709, 0.25), ("h2", 7709, 0.25)])


if __name__ == "__main__":
    unittest.main()
