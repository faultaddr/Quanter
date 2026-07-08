# Frontend CLI Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve frontend reliability and CLI usability without changing algorithm behavior or expanding product scope.

**Architecture:** Add Python smoke tests for CLI behavior and frontend source invariants, then make small entry-point changes. The CLI quick analyze command becomes a thin unified-context facade with lazy imports; the frontend gets one configurable API client path, static overview color classes, route-derived nav state, and visible market-index errors.

**Tech Stack:** Python 3.8+, unittest, Typer, Next.js 14 App Router, TypeScript, Tailwind CSS, Axios.

## Global Constraints

- No page redesign, landing page, or new frontend routes.
- No chart rewrites, WebSocket protocol changes, or API endpoint changes.
- No changes to scoring thresholds, factor weights, recommendation rules, or report Markdown text.
- No backtest, Qlib, GBM, or ML feature-engineering changes.
- No new runtime dependencies.
- No cleanup of unrelated legacy CLI modules in this pass.
- Frontend must not add a JS test framework in this pass.
- Keep unrelated user files such as `AGENTS.md` out of commits.

---

## File Structure

- Create: `tests/test_frontend_cli_optimization.py`
  - Python regression tests for CLI lazy import, CLI quick analyze behavior, and static frontend source invariants.
- Modify: `quanttool/cli/main.py`
  - Lazy-load `StockAnalyzer`, use unified context workflow, print summary, and normalize quick-analysis errors.
- Modify: `quanttool/cli/commands/analysis_commands.py`
  - Move the eager `StockAnalyzer` import into functions that need it so importing `quanttool.cli.main` does not import `quanttool.factors.stock_analyzer`.
- Modify: `quanttool/web/frontend/lib/api/index.ts`
  - Add `getApiBaseUrl()` and `getApiUrl()` and use the helper in the Axios client.
- Modify: `quanttool/web/frontend/lib/api.ts`
  - Keep compatibility with any future `@/lib/api` imports by mirroring the configurable API client.
- Modify: `quanttool/web/frontend/lib/api/backtest.ts`
  - Replace hard-coded stream URL with `getApiUrl("/backtest/run-all-stream")`.
- Modify: `quanttool/web/frontend/next.config.js`
  - Make the API rewrite destination use `NEXT_PUBLIC_API_BASE_URL` with the current local fallback.
- Create: `quanttool/web/frontend/lib/navigation.ts`
  - Export `getPageKeyFromPath(pathname: string): string`.
- Modify: `quanttool/web/frontend/components/layout/AppHeader.tsx`
  - Use `usePathname()` plus `getPageKeyFromPath()` for active nav state.
- Modify: `quanttool/web/frontend/components/layout/AppSidebar.tsx`
  - Use `usePathname()` plus `getPageKeyFromPath()` for active nav state.
- Modify: `quanttool/web/frontend/app/page.tsx`
  - Replace dynamic Tailwind class interpolation with static class maps; add market fetch error state and manual refresh.

---

### Task 1: Add Frontend and CLI Regression Tests

**Files:**
- Create: `tests/test_frontend_cli_optimization.py`

**Interfaces:**
- Produces tests that later tasks must satisfy.

- [ ] **Step 1: Create the failing tests**

Create `tests/test_frontend_cli_optimization.py`:

```python
"""Regression tests for frontend and CLI optimization work."""

import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import typer
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

        self.assertNotIn("quanttool.factors.stock_analyzer", sys.modules)

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

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(calls, [("000001", 120)])
        self.assertIn("=== 三系统评分摘要 ===", result.output)
        self.assertIn("最终推荐: 买入", result.output)
        self.assertIn("REPORT BODY", result.output)
        self.assertEqual(output.read_text(encoding="utf-8"), "REPORT BODY")

    def test_quick_analyze_converts_failures_to_click_exception(self):
        class FakeAnalyzer:
            def analyze_stock_with_context(self, symbol, days):
                raise RuntimeError("boom")

        fake_module = types.ModuleType("quanttool.factors.stock_analyzer")
        fake_module.StockAnalyzer = FakeAnalyzer

        with patch.dict(sys.modules, {"quanttool.factors.stock_analyzer": fake_module}):
            sys.modules.pop("quanttool.cli.main", None)
            cli_main = importlib.import_module("quanttool.cli.main")
            with self.assertRaises(typer.ClickException) as ctx:
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
```

- [ ] **Step 2: Run the tests to verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_frontend_cli_optimization -v
```

Expected: FAIL. The expected failures include:

- `quanttool.factors.stock_analyzer` is imported during `quanttool.cli.main` import.
- quick `analyze` still calls `analyze_stock`, so the fake analyzer lacks the expected method path.
- frontend files do not yet expose `getApiBaseUrl`, `ACTION_COLOR_CLASSES`, or `navigation.ts`.

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add tests/test_frontend_cli_optimization.py
git commit -m "test: add frontend cli optimization guards"
```

---

### Task 2: Optimize CLI Quick Analysis Entry Point

**Files:**
- Modify: `quanttool/cli/main.py`
- Modify: `quanttool/cli/commands/analysis_commands.py`
- Test: `tests/test_frontend_cli_optimization.py`

**Interfaces:**
- Consumes: tests from Task 1.
- Produces: `quanttool.cli.main.analyze(symbol: str, days: int = 360, output: Optional[str] = None)` using unified context workflow.

- [ ] **Step 1: Run CLI tests to confirm current red state**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_frontend_cli_optimization.CliOptimizationTests -v
```

Expected: FAIL on lazy import and unified quick analyze behavior.

- [ ] **Step 2: Remove eager StockAnalyzer import from `main.py`**

In `quanttool/cli/main.py`, delete:

```python
from quanttool.factors.stock_analyzer import StockAnalyzer
```

Add this helper near the `analyze` command:

```python
def _echo_context_summary(context) -> None:
    """Print a concise unified-context score summary."""
    typer.echo("\n=== 三系统评分摘要 ===")
    typer.echo(f"经典评分: {context.classic_score.score:.1f}分")
    if context.trend_score.passed_hard_filter:
        typer.echo(
            f"趋势评分: {context.trend_score.final_score:.1f}分 "
            f"(时机: {context.trend_score.timing_type})"
        )
    else:
        typer.echo(
            f"趋势评分: 未通过过滤 ({context.trend_score.hard_filter_reason})"
        )
    if context.breakout_score.passed_filter:
        typer.echo(f"突破评分: {context.breakout_score.final_score:.1f}分")
    else:
        typer.echo(
            f"突破评分: 未通过筛选 ({context.breakout_score.filter_reason})"
        )
    typer.echo(f"\n最终推荐: {context.final_recommendation.get_action_display()}")
    typer.echo("-" * 50)
```

- [ ] **Step 3: Replace quick `analyze` body**

Replace the command body in `quanttool/cli/main.py` with:

```python
    """Analyze a stock with the unified analysis context."""
    typer.echo(f"正在分析股票：{symbol}")
    typer.echo(f"分析周期：{days} 天")
    typer.echo("-" * 50)

    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        context, report = analyzer.analyze_stock_with_context(symbol, days)
    except Exception as exc:
        raise typer.ClickException(str(exc)) from exc

    _echo_context_summary(context)
    typer.echo(report)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(report)
        typer.echo(f"\n分析报告已保存至：{output}")
```

- [ ] **Step 4: Move `StockAnalyzer` lazy import in analysis subcommands**

In `quanttool/cli/commands/analysis_commands.py`, delete the top-level line:

```python
from quanttool.factors.stock_analyzer import StockAnalyzer
```

Add local imports in functions that instantiate or type-use `StockAnalyzer`.

At the top of `analyze_enhanced(...)`, before `analyzer = StockAnalyzer()`:

```python
    from quanttool.factors.stock_analyzer import StockAnalyzer
```

At the top of `_run_analysis(...)`, before `analyzer = StockAnalyzer()`:

```python
    from quanttool.factors.stock_analyzer import StockAnalyzer
```

For function annotations that currently require `StockAnalyzer` at import time, add this near the imports:

```python
from __future__ import annotations
```

- [ ] **Step 5: Verify CLI tests green**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_frontend_cli_optimization.CliOptimizationTests -v
```

Expected: PASS, 3 tests OK.

- [ ] **Step 6: Run existing smoke tests**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_smoke.ImportSmokeTests.test_cli_app_imports -v
```

Expected: PASS.

- [ ] **Step 7: Commit CLI changes**

Run:

```bash
git add quanttool/cli/main.py quanttool/cli/commands/analysis_commands.py
git commit -m "refactor: optimize cli quick analysis entry"
```

---

### Task 3: Stabilize Frontend API, Navigation, and Overview Feedback

**Files:**
- Modify: `quanttool/web/frontend/lib/api/index.ts`
- Modify: `quanttool/web/frontend/lib/api.ts`
- Modify: `quanttool/web/frontend/lib/api/backtest.ts`
- Modify: `quanttool/web/frontend/next.config.js`
- Create: `quanttool/web/frontend/lib/navigation.ts`
- Modify: `quanttool/web/frontend/components/layout/AppHeader.tsx`
- Modify: `quanttool/web/frontend/components/layout/AppSidebar.tsx`
- Modify: `quanttool/web/frontend/app/page.tsx`
- Test: `tests/test_frontend_cli_optimization.py`

**Interfaces:**
- Produces: `getApiBaseUrl(): string`
- Produces: `getApiUrl(path: string): string`
- Produces: `getPageKeyFromPath(pathname: string): string`

- [ ] **Step 1: Run frontend source tests to confirm red state**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_frontend_cli_optimization.FrontendOptimizationSourceTests -v
```

Expected: FAIL because helpers and source invariants are not implemented yet.

- [ ] **Step 2: Update `lib/api/index.ts`**

Replace the API client setup with:

```ts
import axios from 'axios';

export function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api';
}

export function getApiUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalizedPath}`;
}

const api = axios.create({
  baseURL: getApiBaseUrl(),
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message || '请求失败';
    console.error('API Error:', message);
    return Promise.reject(error);
  }
);

export { api };
export { stockApi } from './stock';
export { backtestApi } from './backtest';
export { modelApi } from './model';
export { monitorApi } from './monitor';
```

- [ ] **Step 3: Update compatibility API file**

Replace `quanttool/web/frontend/lib/api.ts` with:

```ts
import axios from 'axios';

export function getApiBaseUrl(): string {
  return process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api';
}

export function getApiUrl(path: string): string {
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${getApiBaseUrl()}${normalizedPath}`;
}

const api = axios.create({
  baseURL: getApiBaseUrl(),
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.message || '请求失败';
    console.error('API Error:', message);
    return Promise.reject(error);
  }
);

export default api;
export { api };
```

- [ ] **Step 4: Update streaming backtest URL**

In `quanttool/web/frontend/lib/api/backtest.ts`, change the import and fetch:

```ts
import { api, getApiUrl } from './index';
```

Replace:

```ts
fetch('http://localhost:8000/api/backtest/run-all-stream', {
```

with:

```ts
fetch(getApiUrl('/backtest/run-all-stream'), {
```

- [ ] **Step 5: Update Next rewrite config**

Replace the top of `quanttool/web/frontend/next.config.js` with:

```js
/** @type {import('next').NextConfig} */
const apiBaseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api';

const nextConfig = {
  output: 'standalone',
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${apiBaseUrl}/:path*`,
      },
    ];
  },
```

Keep the existing `headers()` block unchanged.

- [ ] **Step 6: Add path-to-page helper**

Create `quanttool/web/frontend/lib/navigation.ts`:

```ts
const PATH_PAGE_KEYS: Array<[string, string]> = [
  ['/analyze', 'analyze'],
  ['/backtest', 'backtest'],
  ['/model', 'model'],
  ['/monitor', 'monitor'],
  ['/scan', 'scan'],
  ['/picks', 'picks'],
  ['/factors', 'factors'],
  ['/risk', 'risk'],
];

export function getPageKeyFromPath(pathname: string): string {
  if (pathname === '/') {
    return 'overview';
  }

  const match = PATH_PAGE_KEYS.find(([prefix]) => pathname.startsWith(prefix));
  return match ? match[1] : 'overview';
}
```

- [ ] **Step 7: Update header active state**

In `quanttool/web/frontend/components/layout/AppHeader.tsx`, add imports:

```ts
import { usePathname } from 'next/navigation';
import { getPageKeyFromPath } from '@/lib/navigation';
```

Inside `AppHeader()`, replace the store-derived active page line with:

```ts
  const pathname = usePathname();
  const activePage = getPageKeyFromPath(pathname);
```

Keep `setActivePage` for click compatibility.

- [ ] **Step 8: Update sidebar active state**

In `quanttool/web/frontend/components/layout/AppSidebar.tsx`, add imports:

```ts
import { usePathname } from 'next/navigation';
import { getPageKeyFromPath } from '@/lib/navigation';
```

Inside `AppSidebar()`, replace the store-derived active page line with:

```ts
  const pathname = usePathname();
  const activePage = getPageKeyFromPath(pathname);
```

Keep `setActivePage` for click compatibility.

- [ ] **Step 9: Add static overview action color classes**

In `quanttool/web/frontend/app/page.tsx`, add this map after the index constants:

```ts
const ACTION_COLOR_CLASSES: Record<string, string> = {
  primary: 'bg-primary/20 text-primary',
  success: 'bg-success/20 text-success',
  danger: 'bg-danger/20 text-danger',
  warning: 'bg-warning/20 text-warning',
  info: 'bg-cyan-500/20 text-cyan-400',
  secondary: 'bg-slate-500/20 text-slate-300',
};
```

Replace both occurrences of:

```tsx
<div className={`p-3 rounded-lg bg-${action.color}/20 text-${action.color}`}>
```

with:

```tsx
<div className={`p-3 rounded-lg ${ACTION_COLOR_CLASSES[action.color]}`}>
```

- [ ] **Step 10: Add market-index error and manual refresh**

In `HomePage()`, add state:

```ts
  const [marketError, setMarketError] = useState<string | null>(null);
```

At the start of `fetchMarketIndices`, add:

```ts
    setMarketError(null);
```

In the `catch` block, add:

```ts
      setMarketError('市场指数加载失败，请稍后重试');
```

Change the market index card action to:

```tsx
<Card
  title="市场指数"
  action={
    <div className="flex items-center gap-2">
      <Badge variant={marketError ? 'warning' : 'success'}>
        {marketError ? '异常' : '实时'}
      </Badge>
      <Button size="sm" variant="ghost" onClick={fetchMarketIndices} loading={loading}>
        刷新
      </Button>
    </div>
  }
>
```

Before the empty state branch, add:

```tsx
          ) : marketError ? (
            <div className="text-center py-8">
              <div className="text-warning">{marketError}</div>
              <Button size="sm" variant="ghost" className="mt-3" onClick={fetchMarketIndices}>
                重新加载
              </Button>
            </div>
```

- [ ] **Step 11: Verify frontend source tests green**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_frontend_cli_optimization.FrontendOptimizationSourceTests -v
```

Expected: PASS, 4 tests OK.

- [ ] **Step 12: Run frontend build**

Run:

```bash
npm run build
```

from:

```bash
cd quanttool/web/frontend
```

Expected: PASS. If it fails on unrelated pre-existing TypeScript errors, record the failure and fix only errors caused by this task.

- [ ] **Step 13: Commit frontend changes**

Run:

```bash
git add \
  quanttool/web/frontend/lib/api/index.ts \
  quanttool/web/frontend/lib/api.ts \
  quanttool/web/frontend/lib/api/backtest.ts \
  quanttool/web/frontend/next.config.js \
  quanttool/web/frontend/lib/navigation.ts \
  quanttool/web/frontend/components/layout/AppHeader.tsx \
  quanttool/web/frontend/components/layout/AppSidebar.tsx \
  quanttool/web/frontend/app/page.tsx
git commit -m "refactor: stabilize frontend shell entrypoints"
```

---

### Task 4: Final Verification

**Files:**
- No source changes expected unless verification finds a regression.

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: verified frontend and CLI optimization pass.

- [ ] **Step 1: Run all Python tests**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
```

Expected: PASS.

- [ ] **Step 2: Run frontend build**

Run:

```bash
npm run build
```

from:

```bash
cd quanttool/web/frontend
```

Expected: PASS or documented unrelated pre-existing failure.

- [ ] **Step 3: Run targeted CLI help smoke**

Run:

```bash
.venv-mcp/bin/python -m quanttool.cli.main --help
```

Expected: exits successfully and does not emit stock analyzer initialization warnings.

- [ ] **Step 4: Check changed files**

Run:

```bash
git status --short
git diff --check HEAD
git diff --name-status 30385b24c..HEAD
```

Expected:

- only intended frontend, CLI, and test files changed since the design commit;
- no `AGENTS.md` staged;
- no whitespace errors.

- [ ] **Step 5: Commit any verification-only fix**

If verification required a fix, commit it:

```bash
git add <fixed-files>
git commit -m "fix: complete frontend cli optimization verification"
```

If no fix was required, do not create an empty commit.
