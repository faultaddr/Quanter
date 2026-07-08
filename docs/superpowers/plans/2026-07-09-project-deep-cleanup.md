# QuantTool Project Deep Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize QuantTool's engineering boundaries by splitting the monolithic API layer, centralizing schemas, fixing packaging/dependency drift, cleaning tracked generated files, and adding smoke-test coverage.

**Architecture:** Keep the public FastAPI contract stable while moving internal endpoint code into domain-specific routers. Keep `quanttool.web.app` including one aggregate router, and make `quanttool/web/api/routes.py` a thin router registry. Move Pydantic request models into `quanttool/web/schemas/` and shared API helpers into `quanttool/web/api/utils.py` and `quanttool/web/api/dependencies.py`.

**Tech Stack:** Python 3.9+, FastAPI, Pydantic v2-compatible model APIs, Typer CLI, setuptools, unittest/pytest-compatible tests, Next.js frontend with npm lint.

## Global Constraints

- Preserve existing HTTP paths, methods, query parameters, and broad response shapes.
- Do not split internal algorithm-heavy files in this pass: `quanttool/factors/stock_analyzer.py`, `quanttool/infrastructure/data_providers/historical/enhanced_fetcher.py`, `quanttool/factors/scoring_system.py`, `quanttool/strategies/qlib_strategy.py`.
- Do not delete legacy public classes that are still imported by code.
- Delete only confirmed generated or miscopied tracked files: `.DS_Store`, root `__init__.py`, `quanttool/web/frontend/tsconfig.tsbuildinfo`, `quanttool/cli/quanttool/config/default.yaml`, `quanttool/web/frontend/quanttool/config/default.yaml`.
- Prefer ASCII for edited code and docs unless existing text requires Chinese.
- Use `apply_patch` for manual edits; mechanical route splitting may use a script because it is a bulk rewrite.
- Keep unrelated user changes (`AGENTS.md`, `reports/`, pre-existing `.DS_Store` modification) out of commits unless explicitly part of this cleanup.

---

## File Structure

- Create: `tests/test_smoke.py` for import and router-contract smoke tests.
- Modify: `quanttool/application/analysis_service.py` to use the new incremental manager import path.
- Modify: `pyproject.toml` to discover all `quanttool*` subpackages.
- Modify: `requirements.txt` to align runtime dependencies with backend startup requirements.
- Modify: `.gitignore` to ignore reports, TypeScript build info, and generated local artifacts.
- Delete: `.DS_Store`, root `__init__.py`, `quanttool/web/frontend/tsconfig.tsbuildinfo`, `quanttool/cli/quanttool/config/default.yaml`, `quanttool/web/frontend/quanttool/config/default.yaml`.
- Create: `quanttool/web/api/utils.py` for `to_python_types` and short-lived analysis cache helpers.
- Create: `quanttool/web/api/dependencies.py` for realtime/minute provider factories and circuit-breaker state.
- Create: `quanttool/web/api/tasks.py`, `stock.py`, `scan.py`, `backtest.py`, `models.py`, `realtime.py`, `monitor.py`, `ml.py`, `factors.py`, `risk.py`, `registry.py`.
- Modify: `quanttool/web/api/routes.py` into a thin aggregate router.
- Create: `quanttool/web/schemas/tasks.py`, `stock.py`, `scan.py`, `model.py`, `realtime.py`, `monitor.py`, `ml.py`, `risk.py`, `common.py`.
- Modify: `quanttool/web/schemas/backtest.py`, `factor.py`, `__init__.py`.
- Modify: `README.md` to reflect the actual layout and verification commands.

---

### Task 1: Smoke Test Baseline

**Files:**
- Create: `tests/test_smoke.py`

**Interfaces:**
- Consumes: existing `quanttool.web.app.app`, `quanttool.cli.main.app`, `quanttool.application.analysis_service.AnalysisService`, and `pyproject.toml`.
- Produces: unittest-compatible smoke coverage runnable by `python -m unittest discover -s tests`.

- [ ] **Step 1: Write the failing import and route-contract tests**

Create `tests/test_smoke.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify expected failures**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
```

Expected before fixes:

```text
FAIL: test_pyproject_discovers_quanttool_subpackages
ERROR: test_analysis_service_imports
```

- [ ] **Step 3: Commit the failing tests**

Run:

```bash
git add tests/test_smoke.py
git commit -m "test: add project smoke checks"
```

---

### Task 2: Import and Packaging Fixes

**Files:**
- Modify: `quanttool/application/analysis_service.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`

**Interfaces:**
- Consumes: `quanttool.infrastructure.data_providers.incremental.manager.IncrementalDataManager`.
- Produces: importable `AnalysisService`, package discovery for all `quanttool*` subpackages, documented runtime dependency parity.

- [ ] **Step 1: Fix the stale incremental manager import**

In `quanttool/application/analysis_service.py`, replace:

```python
from ..infrastructure.data_providers.incremental_data_manager import IncrementalDataManager, DataType
```

with:

```python
from ..infrastructure.data_providers.incremental.manager import IncrementalDataManager, DataType
```

- [ ] **Step 2: Fix setuptools package discovery**

In `pyproject.toml`, replace:

```toml
[tool.setuptools]
packages = ["quanttool"]
```

with:

```toml
[tool.setuptools.packages.find]
include = ["quanttool*"]
```

- [ ] **Step 3: Align runtime dependency declarations**

Ensure `pyproject.toml` main dependencies include runtime imports required by current backend startup:

```toml
    "uvicorn[standard]>=0.15.0",
    "sqlalchemy>=1.4.0",
    "asyncpg>=0.28.0",
    "websockets>=10.0",
```

Ensure `requirements.txt` keeps these same runtime dependencies:

```text
uvicorn[standard]>=0.15.0
sqlalchemy>=1.4.0
asyncpg>=0.28.0
websockets>=10.0
```

- [ ] **Step 4: Run smoke tests**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

- [ ] **Step 5: Commit import and packaging fixes**

Run:

```bash
git add quanttool/application/analysis_service.py pyproject.toml requirements.txt
git commit -m "fix: repair imports and package discovery"
```

---

### Task 3: Generated File and Misplaced Config Cleanup

**Files:**
- Modify: `.gitignore`
- Delete: `.DS_Store`
- Delete: `__init__.py`
- Delete: `quanttool/web/frontend/tsconfig.tsbuildinfo`
- Delete: `quanttool/cli/quanttool/config/default.yaml`
- Delete: `quanttool/web/frontend/quanttool/config/default.yaml`

**Interfaces:**
- Consumes: current config lookup in `quanttool/config/settings.py`, which only checks `config/default.yaml`, `quanttool/config/default.yaml`, and user home config.
- Produces: cleaner tracked tree with generated files ignored and one default config source.

- [ ] **Step 1: Update `.gitignore`**

Add these entries if they are not already present:

```gitignore
# Generated reports and runtime outputs
reports/

# TypeScript incremental build state
*.tsbuildinfo
```

- [ ] **Step 2: Confirm misplaced configs are unused**

Run:

```bash
rg -n "cli/quanttool|frontend/quanttool|quanttool/config/default|default.yaml" quanttool docs README.md CLAUDE.md AGENTS.md -g '!quanttool/web/frontend/tsconfig.tsbuildinfo'
```

Expected relevant code references:

```text
quanttool/config/settings.py:17:            "config/default.yaml",
quanttool/config/settings.py:18:            "quanttool/config/default.yaml",
```

- [ ] **Step 3: Remove tracked generated and misplaced files**

Run:

```bash
git rm .DS_Store __init__.py quanttool/web/frontend/tsconfig.tsbuildinfo quanttool/cli/quanttool/config/default.yaml quanttool/web/frontend/quanttool/config/default.yaml
```

- [ ] **Step 4: Run smoke tests and frontend lint**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
cd quanttool/web/frontend && npm run lint
```

Expected:

```text
OK
✔ No ESLint warnings or errors
```

- [ ] **Step 5: Commit cleanup**

Run:

```bash
git add .gitignore
git commit -m "chore: clean generated project artifacts"
```

---

### Task 4: API Shared Helpers and Schemas

**Files:**
- Create: `quanttool/web/api/utils.py`
- Create: `quanttool/web/api/dependencies.py`
- Create: `quanttool/web/schemas/tasks.py`
- Create: `quanttool/web/schemas/stock.py`
- Create: `quanttool/web/schemas/scan.py`
- Create: `quanttool/web/schemas/model.py`
- Create: `quanttool/web/schemas/realtime.py`
- Create: `quanttool/web/schemas/monitor.py`
- Create: `quanttool/web/schemas/ml.py`
- Create: `quanttool/web/schemas/risk.py`
- Create: `quanttool/web/schemas/common.py`
- Modify: `quanttool/web/schemas/backtest.py`
- Modify: `quanttool/web/schemas/factor.py`
- Modify: `quanttool/web/schemas/__init__.py`

**Interfaces:**
- Consumes: Pydantic model definitions currently in `quanttool/web/api/routes.py`.
- Produces: importable schema modules used by the split routers.

- [ ] **Step 1: Create `quanttool/web/api/utils.py`**

```python
"""Shared helpers for QuantTool API routers."""

from typing import Any, Dict, Optional
import time

import numpy as np


_analysis_cache: Dict[str, tuple] = {}
_analysis_cache_ttl = 60


def get_cached_analysis(cache_key: str) -> Optional[Dict]:
    """Return a cached analysis payload if it is still fresh."""
    if cache_key in _analysis_cache:
        data, timestamp = _analysis_cache[cache_key]
        if time.time() - timestamp < _analysis_cache_ttl:
            return data
    return None


def set_cached_analysis(cache_key: str, data: Dict) -> None:
    """Cache an analysis payload and evict stale entries."""
    _analysis_cache[cache_key] = (data, time.time())
    current_time = time.time()
    expired_keys = [
        key
        for key, (_, timestamp) in _analysis_cache.items()
        if current_time - timestamp > _analysis_cache_ttl * 2
    ]
    for key in expired_keys:
        del _analysis_cache[key]


def to_python_types(obj: Any) -> Any:
    """Convert numpy values into JSON-friendly Python values."""
    if obj is None:
        return None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: to_python_types(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python_types(item) for item in obj]
    return obj
```

- [ ] **Step 2: Create `quanttool/web/api/dependencies.py`**

Move these exact functions and module-level state from `routes.py` into `dependencies.py`:

```text
_is_circuit_open
_record_failure
get_minute_provider
get_realtime_provider
get_incremental_minute_provider
```

Keep their existing implementation and imports, and add:

```python
"""Lazy factories shared by API routers."""
```

at the top.

- [ ] **Step 3: Create schema modules**

Move the `BaseModel` classes from `routes.py` into these modules:

```text
TaskCreateRequest -> quanttool/web/schemas/tasks.py
AnalyzeRequest, EnhancedAnalyzeRequest -> quanttool/web/schemas/stock.py
ScanRequest -> quanttool/web/schemas/scan.py
QlibTrainRequest, QlibPredictRequest, GBMTrainRequest, GBMPredictRequest, GBMPicksRequest -> quanttool/web/schemas/model.py
PortfolioCheckRequest -> quanttool/web/schemas/risk.py
RealtimeQuoteResponse -> quanttool/web/schemas/realtime.py
MonitorStartRequest, MonitorStatusResponse -> quanttool/web/schemas/monitor.py
MLBacktestRequest, MLScanRequest, MLMonitorRequest -> quanttool/web/schemas/ml.py
```

For `BacktestRequest`, keep the existing public class name in `quanttool/web/schemas/backtest.py`, but replace its fields with the current `routes.py` implementation that includes `get_start_date()` and `get_end_date()`.

- [ ] **Step 4: Export schemas from `quanttool/web/schemas/__init__.py`**

Use explicit imports:

```python
"""Web API schemas."""

from .backtest import BacktestRequest
from .model import (
    GBMPicksRequest,
    GBMPredictRequest,
    GBMTrainRequest,
    QlibPredictRequest,
    QlibTrainRequest,
)
from .monitor import MonitorStartRequest, MonitorStatusResponse
from .ml import MLBacktestRequest, MLMonitorRequest, MLScanRequest
from .realtime import RealtimeQuoteResponse
from .risk import PortfolioCheckRequest
from .scan import ScanRequest
from .stock import AnalyzeRequest, EnhancedAnalyzeRequest
from .tasks import TaskCreateRequest

__all__ = [
    "AnalyzeRequest",
    "BacktestRequest",
    "EnhancedAnalyzeRequest",
    "GBMPicksRequest",
    "GBMPredictRequest",
    "GBMTrainRequest",
    "MLBacktestRequest",
    "MLMonitorRequest",
    "MLScanRequest",
    "MonitorStartRequest",
    "MonitorStatusResponse",
    "PortfolioCheckRequest",
    "QlibPredictRequest",
    "QlibTrainRequest",
    "RealtimeQuoteResponse",
    "ScanRequest",
    "TaskCreateRequest",
]
```

- [ ] **Step 5: Run compile and smoke tests**

Run:

```bash
.venv-mcp/bin/python -m compileall -q quanttool
.venv-mcp/bin/python -m unittest discover -s tests -v
```

Expected:

```text
OK
```

- [ ] **Step 6: Commit shared helpers and schemas**

Run:

```bash
git add quanttool/web/api/utils.py quanttool/web/api/dependencies.py quanttool/web/schemas
git commit -m "refactor: extract api helpers and schemas"
```

---

### Task 5: Split API Routers

**Files:**
- Create: `quanttool/web/api/tasks.py`
- Create: `quanttool/web/api/stock.py`
- Create: `quanttool/web/api/scan.py`
- Create: `quanttool/web/api/backtest.py`
- Create: `quanttool/web/api/models.py`
- Create: `quanttool/web/api/realtime.py`
- Create: `quanttool/web/api/monitor.py`
- Create: `quanttool/web/api/ml.py`
- Create: `quanttool/web/api/factors.py`
- Create: `quanttool/web/api/risk.py`
- Create: `quanttool/web/api/registry.py`
- Modify: `quanttool/web/api/routes.py`

**Interfaces:**
- Consumes: endpoint function blocks currently in `routes.py`.
- Produces: one router module per domain and an aggregate `routes.router` with the same public routes.

- [ ] **Step 1: Record the existing API contract**

Run:

```bash
.venv-mcp/bin/python - <<'PY' > /tmp/quanttool_routes_before.txt
from quanttool.web.app import app

items = []
for route in app.routes:
    if not hasattr(route, "methods"):
        continue
    for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
        if route.path.startswith("/api/"):
            items.append((method, route.path))
for method, path in sorted(items):
    print(f"{method} {path}")
PY
```

- [ ] **Step 2: Move endpoint blocks by exact source ranges**

Move these endpoint blocks from `routes.py` into target modules. Preserve function bodies and decorators exactly, except for imports and helper/schema references.

```text
tasks.py:     lines 142-394
stock.py:     lines 396-683 and 769-1932
scan.py:      lines 684-768
backtest.py:  lines 1933-2056 and 4210-4755
models.py:    lines 2071-4209
factors.py:   lines 4756-4821 and 4851-4934
registry.py:  lines 4822-4849
risk.py:      lines 4935-4983
realtime.py:  lines 5077-5324
monitor.py:   lines 5325-5498
ml.py:        lines 5512-5896
```

Each new module starts with:

```python
"""Domain-specific API routes."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json
import queue
import threading
import time

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...core.logging import get_logger
from .utils import get_cached_analysis, set_cached_analysis, to_python_types


logger = get_logger(__name__)
router = APIRouter()
```

Then remove unused imports after `compileall` identifies them only if doing so is obvious and local.

- [ ] **Step 3: Replace old helper names in moved code**

In moved route modules:

```text
_get_cached_analysis -> get_cached_analysis
_set_cached_analysis -> set_cached_analysis
```

In realtime and monitor modules, import provider helpers from `dependencies.py`:

```python
from .dependencies import (
    get_incremental_minute_provider,
    get_minute_provider,
    get_realtime_provider,
)
```

If route code still calls `_is_circuit_open` or `_record_failure`, import those exact names from `dependencies.py`.

- [ ] **Step 4: Replace schema references in moved code**

Use these imports:

```python
from ..schemas.backtest import BacktestRequest
from ..schemas.model import (
    GBMPicksRequest,
    GBMPredictRequest,
    GBMTrainRequest,
    QlibPredictRequest,
    QlibTrainRequest,
)
from ..schemas.monitor import MonitorStartRequest, MonitorStatusResponse
from ..schemas.ml import MLBacktestRequest, MLMonitorRequest, MLScanRequest
from ..schemas.realtime import RealtimeQuoteResponse
from ..schemas.risk import PortfolioCheckRequest
from ..schemas.scan import ScanRequest
from ..schemas.stock import AnalyzeRequest, EnhancedAnalyzeRequest
from ..schemas.tasks import TaskCreateRequest
```

- [ ] **Step 5: Replace `routes.py` with an aggregate router**

Replace `quanttool/web/api/routes.py` with:

```python
"""Aggregate API router for QuantTool web application."""

from fastapi import APIRouter

from . import (
    backtest,
    factors,
    ml,
    models,
    monitor,
    realtime,
    registry,
    risk,
    scan,
    stock,
    tasks,
)


router = APIRouter()

router.include_router(tasks.router)
router.include_router(stock.router)
router.include_router(scan.router)
router.include_router(backtest.router)
router.include_router(models.router)
router.include_router(factors.router)
router.include_router(registry.router)
router.include_router(risk.router)
router.include_router(realtime.router)
router.include_router(monitor.router)
router.include_router(ml.router)
```

- [ ] **Step 6: Compare route contract**

Run:

```bash
.venv-mcp/bin/python - <<'PY' > /tmp/quanttool_routes_after.txt
from quanttool.web.app import app

items = []
for route in app.routes:
    if not hasattr(route, "methods"):
        continue
    for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
        if route.path.startswith("/api/"):
            items.append((method, route.path))
for method, path in sorted(items):
    print(f"{method} {path}")
PY
diff -u /tmp/quanttool_routes_before.txt /tmp/quanttool_routes_after.txt
```

Expected:

```text
```

The `diff` command must print no output.

- [ ] **Step 7: Run compile, smoke tests, and frontend lint**

Run:

```bash
.venv-mcp/bin/python -m compileall -q quanttool
.venv-mcp/bin/python -m unittest discover -s tests -v
cd quanttool/web/frontend && npm run lint
```

Expected:

```text
OK
✔ No ESLint warnings or errors
```

- [ ] **Step 8: Commit router split**

Run:

```bash
git add quanttool/web/api quanttool/web/schemas tests/test_smoke.py
git commit -m "refactor: split web api routers"
```

---

### Task 6: Documentation Refresh

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: final project structure and validation commands.
- Produces: accurate setup, layout, and verification guidance.

- [ ] **Step 1: Update README structure section**

Replace the stale tree with a structure that includes:

```text
quanttool/
├── application/
├── backtest/
├── config/
├── core/
├── domain/
├── factors/
│   ├── scoring/
│   └── technical/
├── infrastructure/
│   ├── cache/
│   ├── data_providers/
│   │   ├── historical/
│   │   ├── incremental/
│   │   └── realtime/
│   ├── database/
│   ├── notifiers/
│   └── stores/
├── reports/
│   └── generators/
├── strategies/
├── validation/
├── web/
│   ├── api/
│   ├── frontend/
│   └── schemas/
└── cli/
tests/
```

- [ ] **Step 2: Update verification instructions**

Add:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
.venv-mcp/bin/python -m quanttool --help
cd quanttool/web/frontend && npm run lint
```

- [ ] **Step 3: Document generated artifacts**

Add a short note that `reports/`, `.cache/`, `quanttool.log`, `.next/`, `node_modules/`, and `*.tsbuildinfo` are local runtime/build artifacts and should not be committed.

- [ ] **Step 4: Run final verification**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
.venv-mcp/bin/python -m quanttool --help
cd quanttool/web/frontend && npm run lint
```

Expected:

```text
OK
✔ No ESLint warnings or errors
```

- [ ] **Step 5: Commit docs**

Run:

```bash
git add README.md
git commit -m "docs: refresh project structure guide"
```

---

### Task 7: Final Verification and Status

**Files:**
- No code files expected unless verification reveals a local issue.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: final evidence that cleanup is complete.

- [ ] **Step 1: Run full verification commands**

Run:

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
.venv-mcp/bin/python -m compileall -q quanttool
.venv-mcp/bin/python -m quanttool --help
cd quanttool/web/frontend && npm run lint
```

Expected:

```text
OK
✔ No ESLint warnings or errors
```

- [ ] **Step 2: Check route file size**

Run:

```bash
wc -l quanttool/web/api/routes.py
```

Expected:

```text
under 100 lines
```

- [ ] **Step 3: Check worktree status**

Run:

```bash
git status --short
```

Expected:

```text
only pre-existing user-local files remain untracked or modified
```

- [ ] **Step 4: Summarize cleanup**

Report:

```text
- API router split completed with route contract preserved.
- Schemas moved under quanttool/web/schemas.
- Stale import fixed.
- Package discovery fixed.
- Generated/miscopied tracked files removed.
- Smoke tests and frontend lint passed.
```
