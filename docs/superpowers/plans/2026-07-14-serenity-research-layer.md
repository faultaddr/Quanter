# QuantTool Serenity Research Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Serenity-style research-priority layer that shares one scorecard contract across Python, CLI, API, and a frontend research workbench without blending it into QuantTool trading scores.

**Architecture:** Pydantic domain models own validation, while a pure `SerenityService` owns scoring, evidence summaries, quadrant classification, and Markdown rendering. CLI and FastAPI are thin adapters. The frontend calls the scorecard endpoint and presents research priority separately from optional quantitative timing.

**Tech Stack:** Python >=3.8, Pydantic, Typer, FastAPI, unittest, Next.js 14 App Router, React 18, TypeScript, Tailwind CSS.

## Global Constraints

- Keep `research_priority_score` and `timing_score` as separate axes; never add, average, multiply, or otherwise merge them into one score.
- Use factor weights `15, 10, 15, 12, 12, 15, 11, 10` in the documented factor order and a penalty multiplier of `2.0`.
- Accept factor and penalty ratings only in the inclusive range `0..5`; accept optional timing score only in `0..100`.
- Use the fixed research threshold `70` for quadrant classification.
- Preserve evidence and `what_could_weaken_view` in every output format.
- Output research support only; do not emit buy/sell commands, return promises, position sizing, or automatic trading actions.
- Do not modify existing classic, trend, breakout, recommendation, scan, or backtest weights and thresholds.
- Do not add Python runtime dependencies or require network, database, qlib, or realtime data in tests.
- Preserve all unrelated dirty worktree changes.

---

### Task 1: Build the Serenity Domain Contract and Pure Service

**Files:**
- Create: `quanttool/domain/models/serenity.py`
- Modify: `quanttool/domain/models/__init__.py`
- Create: `quanttool/application/serenity_service.py`
- Modify: `quanttool/application/__init__.py`
- Create: `tests/test_serenity_service.py`

**Interfaces:**
- Produces: `SerenityScorecard`, `SerenityScoreResult`, `SerenityFactors`, `SerenityPenalties`, `SerenityEvidence`.
- Produces: `SerenityService.score(scorecard)`, `SerenityService.template()`, `SerenityService.to_markdown(result)`.
- Produces: `classify_quadrant(research_score: float, timing_score: Optional[float], threshold: float = 70.0)`.

- [ ] **Step 1: Write failing service tests**

Cover exact weighted score, penalty clamping, all four verdict thresholds, model validation, evidence counts, all four quadrants, no-quadrant behavior, and Markdown sections. Use a scorecard with all factor ratings at `5`, one penalty at `5`, one strong and one unverified evidence item, and one weakening condition.

- [ ] **Step 2: Run the focused test and confirm RED**

Run:

```bash
/Users/missy/.venvs/quanttool/bin/python -m unittest tests.test_serenity_service -v
```

Expected: import failure for `quanttool.domain.models.serenity` or `quanttool.application.serenity_service`.

- [ ] **Step 3: Implement validated domain models**

Use string enums for `EvidenceStrength`, `ResearchVerdict`, and `ResearchTimingQuadrant`. Define each factor and penalty as `float = Field(0.0, ge=0.0, le=5.0)`, use `Field(default_factory=...)` for nested objects and lists, and define `timing_score` as `Optional[float] = Field(None, ge=0.0, le=100.0)`.

- [ ] **Step 4: Implement the pure scoring service**

Keep weights and supported penalties in module-level constants. Return per-field rating, weight, and points; count evidence by strength; clamp only the final score; classify the verdict and optional quadrant; render Markdown with candidate identity, score, verdict, separate timing score, factor table, penalty table, evidence, weakening conditions, and the research-only boundary.

- [ ] **Step 5: Run focused and regression tests**

Run:

```bash
/Users/missy/.venvs/quanttool/bin/python -m unittest tests.test_serenity_service tests.test_scoring_contracts -v
```

Expected: all tests pass and existing scoring contract tests remain unchanged.

### Task 2: Add the `quant research` CLI

**Files:**
- Create: `quanttool/cli/commands/research_commands.py`
- Modify: `quanttool/cli/commands/__init__.py`
- Modify: `quanttool/cli/main.py`
- Create: `tests/test_serenity_cli.py`

**Interfaces:**
- Produces: `quant research template`.
- Produces: `quant research scorecard INPUT --format json|md|both`.
- Consumes: `SerenityService` and `SerenityScorecard` from Task 1.

- [ ] **Step 1: Write failing `CliRunner` tests**

Verify template JSON, file input JSON output, standard-input Markdown output, `both` separator, malformed JSON, missing file, and out-of-range model validation. Assert errors have non-zero exit codes and no traceback text.

- [ ] **Step 2: Run the focused CLI test and confirm RED**

Run:

```bash
/Users/missy/.venvs/quanttool/bin/python -m unittest tests.test_serenity_cli -v
```

Expected: `research` command is absent.

- [ ] **Step 3: Implement the thin Typer adapter**

Load UTF-8 JSON from a path or `stdin`, parse with `SerenityScorecard`, call the service, and use `json.dumps(..., ensure_ascii=False, indent=2)`. Convert file, JSON, and validation failures to `click.ClickException`. Register the command group lazily with the existing CLI without importing stock analysis at module import time.

- [ ] **Step 4: Run CLI tests and a real command**

Run:

```bash
/Users/missy/.venvs/quanttool/bin/python -m unittest tests.test_serenity_cli tests.test_frontend_cli_optimization.CliOptimizationTests -v
/Users/missy/.venvs/quanttool/bin/quant research template
```

Expected: tests pass and the command prints valid UTF-8 JSON.

### Task 3: Expose the Scorecard Through FastAPI

**Files:**
- Create: `quanttool/web/schemas/serenity.py`
- Modify: `quanttool/web/schemas/__init__.py`
- Create: `quanttool/web/api/research.py`
- Modify: `quanttool/web/api/registry.py`
- Create: `tests/test_serenity_api.py`
- Modify: `tests/test_smoke.py`

**Interfaces:**
- Produces: `GET /api/research/serenity/template`.
- Produces: `POST /api/research/serenity/scorecard`.
- Response: `{success, data, error, timestamp}`.
- Consumes: `SerenityService` from Task 1.

- [ ] **Step 1: Write failing API tests**

Inspect routes directly for registration, call endpoint functions directly for deterministic response checks, and use FastAPI request validation for one out-of-range field. Assert the exact top-level response keys and that `data.research_priority_score` is present.

- [ ] **Step 2: Run the focused API test and confirm RED**

Run:

```bash
/Users/missy/.venvs/quanttool/bin/python -m unittest tests.test_serenity_api -v
```

Expected: research API module or routes are absent.

- [ ] **Step 3: Implement schemas and router**

Keep HTTP-only response wrappers in `quanttool/web/schemas/serenity.py`. Use a router prefix `/research/serenity` and tags `research`. Build timestamps with the project's UTC helper or timezone-aware UTC datetime. Return validation errors through FastAPI and unexpected service errors through the standard failure envelope.

- [ ] **Step 4: Register the router and lock route uniqueness**

Add the router to `ROUTER_SPECS` with `/api` prefix. Extend smoke expectations with both routes; do not duplicate method/path pairs.

- [ ] **Step 5: Run API and full Python tests**

Run:

```bash
/Users/missy/.venvs/quanttool/bin/python -m unittest tests.test_serenity_api tests.test_smoke -v
/Users/missy/.venvs/quanttool/bin/python -m unittest discover -s tests -v
```

Expected: all old and new tests pass.

### Task 4: Add the Frontend Research Workbench

**Files:**
- Create: `quanttool/web/frontend/types/research.ts`
- Create: `quanttool/web/frontend/lib/api/research.ts`
- Modify: `quanttool/web/frontend/lib/api/index.ts`
- Create: `quanttool/web/frontend/app/research/page.tsx`
- Create: `quanttool/web/frontend/components/research/ScoreField.tsx`
- Create: `quanttool/web/frontend/components/research/ResearchResult.tsx`
- Create: `quanttool/web/frontend/components/research/index.ts`
- Modify: `quanttool/web/frontend/components/layout/AppSidebar.tsx`
- Modify: `quanttool/web/frontend/lib/navigation.ts`
- Modify: `tests/test_frontend_cli_optimization.py`

**Interfaces:**
- Consumes: `POST /api/research/serenity/scorecard`.
- Produces: `/research` route and `research` navigation key.
- Produces: `researchApi.scorecard(input): Promise<SerenityScoreResult>`.

- [ ] **Step 1: Write failing frontend source-contract tests**

Assert the route exists, the API client uses `/research/serenity/scorecard`, navigation derives the `research` key from `/research`, sidebar exposes `产业链研究`, the page includes all eight factor keys, and visible copy separates research priority from trading advice.

- [ ] **Step 2: Run the focused source test and confirm RED**

Run:

```bash
/Users/missy/.venvs/quanttool/bin/python -m unittest tests.test_frontend_cli_optimization.FrontendOptimizationSourceTests -v
```

Expected: new research route and client assertions fail.

- [ ] **Step 3: Implement typed API client and focused components**

Define input/result types matching the backend. `ScoreField` renders a label, short research-oriented description, stable numeric input with min `0`, max `5`, step `0.5`, and the current value. `ResearchResult` renders separate research and timing metrics, quadrant text, evidence counts, factor/penalty details, and weakening conditions without nested cards.

- [ ] **Step 4: Implement the `/research` workbench**

Use existing `PageContainer`, `PageHeader`, `Section`, `MetricTile`, `StatusBadge`, and `Button`. Keep identity, factors, penalties, evidence, weakening conditions, and results in full-width sections. Preserve form state on request failure. Use responsive one/two-column grids with stable control dimensions and no hero or marketing section.

- [ ] **Step 5: Register navigation and run frontend verification**

Run:

```bash
cd quanttool/web/frontend && npm run build
```

Expected: Next.js production build succeeds and `/research` appears in the route list.

- [ ] **Step 6: Run browser checks**

Start backend and frontend with the documented commands. At desktop and mobile widths, verify `/research` loads, all labels fit, a valid scorecard submits, results remain separate from timing, request errors do not clear inputs, and no controls overlap.

### Task 5: Final Integration Review

**Files:**
- Verify all files from Tasks 1-4.
- Update: `.superpowers/sdd/progress.md` during execution only.

**Interfaces:**
- Confirms the design completion boundary end to end.

- [ ] **Step 1: Run full verification**

Run:

```bash
/Users/missy/.venvs/quanttool/bin/python -m unittest discover -s tests -v
/Users/missy/.venvs/quanttool/bin/python -m compileall -q quanttool
cd quanttool/web/frontend && npm run build
git diff --check HEAD
```

- [ ] **Step 2: Audit requirements**

Confirm every design completion item with current files, test output, route registration, CLI output, build output, and browser evidence. Confirm no arithmetic combines research and timing scores and no buy/sell copy was introduced.

- [ ] **Step 3: Request whole-change code review**

Create a review package from the pre-task base to current HEAD and dispatch a fresh high-capability reviewer. Fix all Critical and Important findings, re-run covering tests, and repeat review until clean.
