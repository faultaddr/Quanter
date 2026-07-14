# QuantTool Stock API Routes Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the large stock API router into focused stock route modules without changing public API behavior.

**Architecture:** Keep `quanttool/web/api/stock.py` as a compatibility aggregate exporting `router`. Move endpoint implementations into `quanttool/web/api/stock_routes/` modules grouped by stock API responsibility, and protect the boundary with structural smoke tests.

**Tech Stack:** Python 3.9+, FastAPI `APIRouter`, unittest-compatible tests, existing QuantTool stock schemas in `quanttool.web.schemas.stock`.

## Global Constraints

- Preserve existing `/api/analyze*`, `/api/stock/{symbol}/*`, and `/api/index/{index_code}/data` paths and HTTP methods.
- Do not alter stock analysis, enhanced analysis, chip distribution, signal, flow, risk, factor, feasibility, index, or backtest-compare behavior.
- Do not introduce new runtime dependencies.
- Use `apply_patch` for manual edits; a mechanical script may be used only for exact code movement.
- Keep unrelated user files such as `AGENTS.md` out of commits.

---

## File Structure

- Modify: `tests/test_smoke.py` to add stock-router structure and contract checks.
- Modify: `quanttool/web/api/stock.py` into a thin aggregate router.
- Create: `quanttool/web/api/stock_routes/__init__.py` for stock-route aggregation.
- Create: `quanttool/web/api/stock_routes/analysis.py` for `/analyze`, `/analyze/enhanced`, and `/stock/{symbol}/analysis`.
- Create: `quanttool/web/api/stock_routes/market_data.py` for stock info, K line, and index data endpoints.
- Create: `quanttool/web/api/stock_routes/chip_signals.py` for chip and technical signal endpoints.
- Create: `quanttool/web/api/stock_routes/insights.py` for flow, risk, factors, feasibility, and backtest comparison endpoints.

---

### Task 1: Add Structural Test

**Files:**
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Consumes: FastAPI app route registry.
- Produces: tests that fail while `quanttool/web/api/stock.py` remains monolithic.

- [ ] **Step 1: Add failing structure test**

Add a test method that checks `quanttool/web/api/stock.py` is a thin aggregate and required stock route modules exist:

```python
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
```

- [ ] **Step 2: Add stock endpoint contract coverage**

Add a route check for stock endpoints:

```python
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
```

- [ ] **Step 3: Verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_smoke.ApiStructureTests.test_stock_api_router_is_split_into_focused_modules -v
```

Expected: FAIL because `quanttool/web/api/stock.py` has more than 120 lines and `stock_routes/` does not exist yet.

### Task 2: Split Stock Routes

**Files:**
- Modify: `quanttool/web/api/stock.py`
- Create: `quanttool/web/api/stock_routes/__init__.py`
- Create: `quanttool/web/api/stock_routes/analysis.py`
- Create: `quanttool/web/api/stock_routes/market_data.py`
- Create: `quanttool/web/api/stock_routes/chip_signals.py`
- Create: `quanttool/web/api/stock_routes/insights.py`

**Interfaces:**
- Consumes: existing request models from `quanttool.web.schemas.stock`.
- Produces: `quanttool.web.api.stock.router` that includes all stock-route modules.

- [ ] **Step 1: Move endpoint groups mechanically**

Use the existing function boundaries:

- `analysis.py`: `analyze_stock`, `analyze_stock_enhanced`, `get_stock_analysis`.
- `market_data.py`: `get_stock_info`, `get_stock_kline`, `get_index_data`.
- `chip_signals.py`: `get_chip_distribution`, `get_technical_signals`.
- `insights.py`: `get_stock_flow`, `get_stock_risk`, `get_stock_factors`, `get_stock_feasibility`, `get_stock_backtest_compare`.

- [ ] **Step 2: Replace `stock.py` with aggregate router**

Use this shape:

```python
"""Stock analysis API route aggregate."""

from fastapi import APIRouter

from .stock_routes import router as stock_routes_router


router = APIRouter()
router.include_router(stock_routes_router)
```

- [ ] **Step 3: Verify green**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
```

Expected: all tests pass and compileall exits 0.

### Task 3: Final Verification and Commit

**Files:**
- All files touched in Tasks 1 and 2.

**Interfaces:**
- Consumes: complete test suite and lint command.
- Produces: committed phase-3 stock API cleanup.

- [ ] **Step 1: Run full verification**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
cd quanttool/web/frontend && npm run lint
```

Expected: tests pass, compileall exits 0, frontend lint reports no warnings or errors.

- [ ] **Step 2: Commit**

Run:

```bash
git add docs/superpowers/specs/2026-07-09-stock-api-routes-phase3-design.md docs/superpowers/plans/2026-07-09-stock-api-routes-phase3.md tests/test_smoke.py quanttool/web/api/stock.py quanttool/web/api/stock_routes
git commit -m "refactor: split stock api routes"
```
