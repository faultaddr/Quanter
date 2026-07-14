# QuantTool API Final Wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the current API-layer refactor by splitting the remaining multi-endpoint backtest and ML routers into focused route modules while preserving public API behavior.

**Architecture:** Keep `quanttool/web/api/backtest.py` and `quanttool/web/api/ml.py` as compatibility aggregates exporting `router`. Move endpoint implementations into `backtest_routes/` and `ml_routes/` modules grouped by responsibility, and protect the final boundary with structural smoke tests.

**Tech Stack:** Python 3.9+, FastAPI `APIRouter`, FastAPI `StreamingResponse`, unittest-compatible tests, existing QuantTool schemas in `quanttool.web.schemas.backtest` and `quanttool.web.schemas.ml`.

## Global Constraints

- Preserve existing `/api/backtest/*`, `/api/experiments`, and `/api/ml/*` paths and HTTP methods.
- Do not alter backtest execution, strategy comparison, stream events, experiment lookup, ML backtest, ML scan, ML monitor behavior, or response fields.
- Do not introduce new runtime dependencies.
- Use `apply_patch` for manual edits; a mechanical script may be used only for exact code movement.
- Keep unrelated user files such as `AGENTS.md` out of commits.

---

## File Structure

- Modify: `tests/test_smoke.py` to add final backtest and ML router structure checks.
- Modify: `quanttool/web/api/backtest.py` into a thin aggregate router.
- Create: `quanttool/web/api/backtest_routes/__init__.py`.
- Create: `quanttool/web/api/backtest_routes/catalog.py`.
- Create: `quanttool/web/api/backtest_routes/execution.py`.
- Create: `quanttool/web/api/backtest_routes/comparison.py`.
- Create: `quanttool/web/api/backtest_routes/stream.py`.
- Create: `quanttool/web/api/backtest_routes/experiments.py`.
- Modify: `quanttool/web/api/ml.py` into a thin aggregate router.
- Create: `quanttool/web/api/ml_routes/__init__.py`.
- Create: `quanttool/web/api/ml_routes/backtest.py`.
- Create: `quanttool/web/api/ml_routes/scan.py`.
- Create: `quanttool/web/api/ml_routes/monitor.py`.

---

### Task 1: Add Final Structural Tests

**Files:**
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Consumes: FastAPI app route registry.
- Produces: tests that fail while `backtest.py` and `ml.py` remain monolithic.

- [ ] **Step 1: Add failing backtest structure test**

```python
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
```

- [ ] **Step 2: Add failing ML structure test**

```python
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
```

- [ ] **Step 3: Add final route contract coverage**

```python
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
```

- [ ] **Step 4: Verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_smoke.ApiStructureTests.test_backtest_api_router_is_split_into_focused_modules tests.test_smoke.ApiStructureTests.test_ml_api_router_is_split_into_focused_modules -v
```

Expected: FAIL because both aggregate files have more than 120 lines and route directories do not exist yet.

### Task 2: Split Backtest Routes

**Files:**
- Modify: `quanttool/web/api/backtest.py`
- Create: `quanttool/web/api/backtest_routes/__init__.py`
- Create: `quanttool/web/api/backtest_routes/catalog.py`
- Create: `quanttool/web/api/backtest_routes/execution.py`
- Create: `quanttool/web/api/backtest_routes/comparison.py`
- Create: `quanttool/web/api/backtest_routes/stream.py`
- Create: `quanttool/web/api/backtest_routes/experiments.py`

**Interfaces:**
- Consumes: existing `BacktestRequest` from `quanttool.web.schemas.backtest`.
- Produces: `quanttool.web.api.backtest.router` that includes all backtest route modules.

- [ ] **Step 1: Move endpoint groups mechanically**

- `catalog.py`: `list_backtest_strategies`.
- `execution.py`: stacked `/backtest/history` and `/backtest/run` decorators with `run_backtest`.
- `comparison.py`: `run_all_strategies_backtest`.
- `stream.py`: `run_all_strategies_backtest_stream`.
- `experiments.py`: `list_experiments`, `get_backtest_result`.

- [ ] **Step 2: Replace `backtest.py` with aggregate router**

```python
"""Backtest API route aggregate."""

from fastapi import APIRouter

from .backtest_routes import router as backtest_routes_router


router = APIRouter()
router.include_router(backtest_routes_router)
```

### Task 3: Split ML Routes

**Files:**
- Modify: `quanttool/web/api/ml.py`
- Create: `quanttool/web/api/ml_routes/__init__.py`
- Create: `quanttool/web/api/ml_routes/backtest.py`
- Create: `quanttool/web/api/ml_routes/scan.py`
- Create: `quanttool/web/api/ml_routes/monitor.py`

**Interfaces:**
- Consumes: existing ML request schemas from `quanttool.web.schemas.ml`.
- Produces: `quanttool.web.api.ml.router` that includes all ML route modules.

- [ ] **Step 1: Move endpoint groups mechanically**

- `backtest.py`: `run_ml_backtest`.
- `scan.py`: `scan_with_ml_model`.
- `monitor.py`: `_monitor_services`, `start_ml_monitor`, `get_ml_monitor_signals`.

- [ ] **Step 2: Replace `ml.py` with aggregate router**

```python
"""ML strategy API route aggregate."""

from fastapi import APIRouter

from .ml_routes import router as ml_routes_router


router = APIRouter()
router.include_router(ml_routes_router)
```

### Task 4: Final Verification and Commit

**Files:**
- All files touched in Tasks 1 through 3.

**Interfaces:**
- Consumes: complete test suite and lint command.
- Produces: committed final API cleanup.

- [ ] **Step 1: Run full verification**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
cd quanttool/web/frontend && npm run lint
```

Expected: tests pass, compileall exits 0, frontend lint reports no warnings or errors.

- [ ] **Step 2: Confirm route snapshot**

Run:

```bash
.venv-mcp/bin/python - <<'PY' | rg '^(GET|POST|DELETE|PUT|PATCH) /api' > /tmp/quanttool_api_routes_after_final_wave.txt
from quanttool.web.app import app
for method, path in sorted(
    (method, route.path)
    for route in app.routes
    if hasattr(route, "methods")
    for method in route.methods - {"HEAD", "OPTIONS"}
):
    print(f"{method} {path}")
PY
diff -u /tmp/quanttool_routes_before.txt /tmp/quanttool_api_routes_after_final_wave.txt
```

Expected: no diff.

- [ ] **Step 3: Commit**

Run:

```bash
git add docs/superpowers/specs/2026-07-09-api-final-wave-design.md docs/superpowers/plans/2026-07-09-api-final-wave.md tests/test_smoke.py quanttool/web/api/backtest.py quanttool/web/api/backtest_routes quanttool/web/api/ml.py quanttool/web/api/ml_routes
git commit -m "refactor: complete api route consolidation"
```
