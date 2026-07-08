# QuantTool Qlib Training Routes Phase 4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the large Qlib training API router into focused batch and streaming route modules without changing public API behavior.

**Architecture:** Keep `quanttool/web/api/model_routes/qlib_training.py` as a compatibility aggregate exporting `router`. Move endpoint implementations into `quanttool/web/api/model_routes/qlib_training_routes/` modules grouped by training mode, and protect the boundary with structural smoke tests.

**Tech Stack:** Python 3.9+, FastAPI `APIRouter`, FastAPI `StreamingResponse`, unittest-compatible tests, existing QuantTool model schemas in `quanttool.web.schemas.model`.

## Global Constraints

- Preserve existing `/api/qlib/train` and `/api/qlib/train/stream` paths and HTTP methods.
- Do not alter Qlib training, streaming events, fallback behavior, metrics, model persistence, or response fields.
- Do not introduce new runtime dependencies.
- Use `apply_patch` for manual edits; a mechanical script may be used only for exact code movement.
- Keep unrelated user files such as `AGENTS.md` out of commits.

---

## File Structure

- Modify: `tests/test_smoke.py` to add Qlib training-router structure checks.
- Modify: `quanttool/web/api/model_routes/qlib_training.py` into a thin aggregate router.
- Create: `quanttool/web/api/model_routes/qlib_training_routes/__init__.py` for Qlib training route aggregation.
- Create: `quanttool/web/api/model_routes/qlib_training_routes/batch.py` for `/qlib/train`.
- Create: `quanttool/web/api/model_routes/qlib_training_routes/stream.py` for `/qlib/train/stream`.

---

### Task 1: Add Structural Test

**Files:**
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Consumes: FastAPI app route registry.
- Produces: tests that fail while `quanttool/web/api/model_routes/qlib_training.py` remains monolithic.

- [ ] **Step 1: Add failing structure test**

Add a test method that checks `qlib_training.py` is a thin aggregate and required route modules exist:

```python
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
```

- [ ] **Step 2: Add Qlib training endpoint contract coverage**

Add a route check for Qlib training endpoints:

```python
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
```

- [ ] **Step 3: Verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_smoke.ApiStructureTests.test_qlib_training_api_router_is_split_into_focused_modules -v
```

Expected: FAIL because `quanttool/web/api/model_routes/qlib_training.py` has more than 120 lines and `qlib_training_routes/` does not exist yet.

### Task 2: Split Qlib Training Routes

**Files:**
- Modify: `quanttool/web/api/model_routes/qlib_training.py`
- Create: `quanttool/web/api/model_routes/qlib_training_routes/__init__.py`
- Create: `quanttool/web/api/model_routes/qlib_training_routes/batch.py`
- Create: `quanttool/web/api/model_routes/qlib_training_routes/stream.py`

**Interfaces:**
- Consumes: existing `QlibTrainRequest` from `quanttool.web.schemas.model`.
- Produces: `quanttool.web.api.model_routes.qlib_training.router` that includes batch and stream training routers.

- [ ] **Step 1: Move endpoint groups mechanically**

Use existing function boundaries:

- `batch.py`: `train_qlib_model`.
- `stream.py`: `train_qlib_model_stream`.

- [ ] **Step 2: Replace `qlib_training.py` with aggregate router**

Use this shape:

```python
"""Qlib model training API route aggregate."""

from fastapi import APIRouter

from .qlib_training_routes import router as qlib_training_routes_router


router = APIRouter()
router.include_router(qlib_training_routes_router)
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
- Produces: committed phase-4 Qlib training route cleanup.

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
git add docs/superpowers/specs/2026-07-09-qlib-training-routes-phase4-design.md docs/superpowers/plans/2026-07-09-qlib-training-routes-phase4.md tests/test_smoke.py quanttool/web/api/model_routes/qlib_training.py quanttool/web/api/model_routes/qlib_training_routes
git commit -m "refactor: split qlib training routes"
```
