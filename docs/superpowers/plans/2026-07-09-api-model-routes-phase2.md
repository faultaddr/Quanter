# QuantTool API Model Routes Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the remaining large model API router into focused GBM and Qlib router modules without changing public API behavior.

**Architecture:** Keep `quanttool/web/api/models.py` as a compatibility aggregate exporting `router`. Move endpoint implementations into `quanttool/web/api/model_routes/` modules grouped by responsibility, and protect the boundary with structural smoke tests.

**Tech Stack:** Python 3.9+, FastAPI `APIRouter`, unittest-compatible tests, existing QuantTool schemas in `quanttool.web.schemas.model`.

## Global Constraints

- Preserve existing `/api/gbm/*` and `/api/qlib/*` paths and HTTP methods.
- Do not alter training, prediction, feature engineering, backtest, or model persistence logic.
- Do not introduce new runtime dependencies.
- Use `apply_patch` for manual edits; a mechanical script may be used only for exact code movement.
- Keep unrelated user files such as `AGENTS.md` out of commits.

---

## File Structure

- Modify: `tests/test_smoke.py` to add model-router structure and contract checks.
- Modify: `quanttool/web/api/models.py` into a thin aggregate router.
- Create: `quanttool/web/api/model_routes/__init__.py` for model-route aggregation.
- Create: `quanttool/web/api/model_routes/discovery.py` for Qlib model listing and detail endpoints.
- Create: `quanttool/web/api/model_routes/gbm.py` for GBM endpoints and `_training_tasks`.
- Create: `quanttool/web/api/model_routes/qlib_training.py` for Qlib train and streaming train endpoints.
- Create: `quanttool/web/api/model_routes/qlib_prediction.py` for Qlib prediction/backtest endpoint.

---

### Task 1: Add Structural Test

**Files:**
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Consumes: FastAPI app route registry.
- Produces: tests that fail while `quanttool/web/api/models.py` remains monolithic.

- [ ] **Step 1: Add failing structure test**

Add a test class that checks `quanttool/web/api/models.py` is a thin aggregate and required model route modules exist:

```python
class ApiStructureTests(unittest.TestCase):
    def test_model_api_router_is_split_into_focused_modules(self):
        api_dir = Path("quanttool/web/api")
        aggregate = api_dir / "models.py"
        route_dir = api_dir / "model_routes"

        self.assertLessEqual(len(aggregate.read_text(encoding="utf-8").splitlines()), 120)
        for module_name in [
            "__init__.py",
            "discovery.py",
            "gbm.py",
            "qlib_training.py",
            "qlib_prediction.py",
        ]:
            self.assertTrue((route_dir / module_name).is_file(), module_name)
```

- [ ] **Step 2: Add model endpoint contract coverage**

Add a route check for the key model endpoints:

```python
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
```

- [ ] **Step 3: Verify red**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_smoke.ApiStructureTests.test_model_api_router_is_split_into_focused_modules -v
```

Expected: FAIL because `quanttool/web/api/models.py` has more than 120 lines and `model_routes/` does not exist yet.

### Task 2: Split Model Routes

**Files:**
- Modify: `quanttool/web/api/models.py`
- Create: `quanttool/web/api/model_routes/__init__.py`
- Create: `quanttool/web/api/model_routes/discovery.py`
- Create: `quanttool/web/api/model_routes/gbm.py`
- Create: `quanttool/web/api/model_routes/qlib_training.py`
- Create: `quanttool/web/api/model_routes/qlib_prediction.py`

**Interfaces:**
- Consumes: existing request models from `quanttool.web.schemas.model`.
- Produces: `quanttool.web.api.models.router` that includes all model-route modules.

- [ ] **Step 1: Move endpoint groups mechanically**

Use the existing function boundaries:

- `discovery.py`: `list_qlib_models`, `list_saved_models`, `list_pretrained_models`, `list_all_models`, `get_saved_model_detail`, `get_qlib_model_categories`.
- `gbm.py`: `GBM_OPTIMAL_PARAMS`, `train_gbm_model`, `predict_gbm_model`, `list_gbm_models`, `delete_gbm_model`, `_training_tasks`, `get_training_progress`, `list_qrun_models`, `get_gbm_picks`.
- `qlib_training.py`: `train_qlib_model`, `train_qlib_model_stream`.
- `qlib_prediction.py`: `predict_with_qlib_model`.

- [ ] **Step 2: Replace `models.py` with aggregate router**

Use this shape:

```python
"""GBM and Qlib model API route aggregate."""

from fastapi import APIRouter

from .model_routes import router as model_routes_router


router = APIRouter()
router.include_router(model_routes_router)
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
- Produces: committed phase-2 cleanup.

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
git add docs/superpowers/specs/2026-07-09-api-model-routes-phase2-design.md docs/superpowers/plans/2026-07-09-api-model-routes-phase2.md tests/test_smoke.py quanttool/web/api/models.py quanttool/web/api/model_routes
git commit -m "refactor: split model api routes"
```
