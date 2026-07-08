# QuantTool Frontend and CLI Optimization Design

## Goal

Improve the day-to-day usability and reliability of the existing frontend and CLI without expanding product scope or changing algorithm-layer behavior.

This is a first optimization pass after the algorithm-core refactor. It should make the app easier to operate, reduce surprising runtime behavior, and add focused regression coverage around the changed entry points.

## Scope

### Frontend

1. Stabilize production styling on the overview page.
   - Replace dynamic Tailwind color class construction in `app/page.tsx` with explicit class maps.
   - Keep the existing page composition and information density.
   - Do not redesign the product shell or introduce a new visual system.

2. Centralize API base URL handling.
   - Use one frontend API client configuration path under `quanttool/web/frontend/lib/api`.
   - Support `NEXT_PUBLIC_API_BASE_URL`, falling back to `http://localhost:8000/api` for local development.
   - Remove hard-coded API hosts from feature modules where practical in this pass.

3. Improve overview market-index feedback.
   - Show a user-facing error state when market index fetch fails.
   - Add a manual refresh affordance.
   - Preserve the 30-second auto-refresh behavior.

4. Make navigation active state path-driven.
   - Add a small route-to-page helper so header/sidebar active state can follow the current URL.
   - Keep the existing Zustand store for persisted theme, sidebar state, history, and toast state.

### CLI

1. Reduce import-time side effects in the top-level CLI.
   - Remove eager `StockAnalyzer` import from `quanttool/cli/main.py`.
   - Import analysis-heavy modules inside commands that actually need them.

2. Make the quick `analyze` command use the unified context workflow.
   - Preserve command signature: `quanttool analyze SYMBOL --days/-d --output/-o`.
   - Use `StockAnalyzer.analyze_stock_with_context(...)` by default for consistent reports.
   - Print a concise score summary before the report, matching the existing `analysis single --unified` behavior.

3. Normalize CLI error handling for quick analysis.
   - Convert unexpected analysis errors to a clear `typer.ClickException`.
   - Keep successful output and optional file write behavior.

## Non-Goals

- No page redesign, landing page, or new frontend routes.
- No chart rewrites, WebSocket protocol changes, or API endpoint changes.
- No changes to scoring thresholds, factor weights, recommendation rules, or report Markdown text.
- No backtest, Qlib, GBM, or ML feature-engineering changes.
- No new runtime dependencies.
- No cleanup of unrelated legacy CLI modules in this pass.

## Architecture

### Frontend API Client

Create or update a single API client source of truth:

- `quanttool/web/frontend/lib/api/index.ts` exports `api`.
- `getApiBaseUrl()` resolves `process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api"`.
- Feature modules import from `./index` as they already do.
- Direct `fetch("http://localhost:8000/...")` should be replaced with a helper-derived base URL where touched.

This preserves current local behavior while making frontend deployment configurable.

### Frontend Navigation State

Add a helper such as:

```ts
export function getPageKeyFromPath(pathname: string): string
```

Use it from layout navigation components with `usePathname()` so active state follows URL changes, refreshes, direct links, and browser back/forward. Existing `setActivePage` remains for compatibility but is no longer the sole source of truth for shell navigation highlighting.

### Overview Page Feedback

The overview page keeps its dashboard-style structure. It gains:

- explicit action color classes,
- `marketError` state,
- a refresh button in the market-index card action,
- clear loading/error/empty states.

### CLI Quick Analysis

`quanttool/cli/main.py` should stay a light command registry. The quick `analyze` command should import `StockAnalyzer` only inside the function and delegate to unified context analysis:

```python
context, report = analyzer.analyze_stock_with_context(symbol, days)
```

It should print the score summary and then the report. On failure, raise `typer.ClickException(str(exc))`.

## Testing

### Python

Add smoke/behavior tests for CLI behavior:

- importing `quanttool.cli.main` does not import `quanttool.factors.stock_analyzer`;
- quick `analyze` calls `analyze_stock_with_context`;
- quick `analyze --output file` writes the returned report;
- quick `analyze` converts analyzer failures to `typer.ClickException`.

Use monkeypatching with `unittest.mock` or direct `sys.modules` stubs. Tests must not instantiate real `StockAnalyzer`, hit network providers, or touch databases.

### Frontend

Use available project tooling:

- `npm run build` for TypeScript/Next production validation;
- if no frontend unit-test runner exists, add no new JS test framework in this pass.

Python smoke tests remain the required automated guard for CLI changes.

## Risks and Mitigations

- **Risk:** changing API base URL breaks local browser calls.
  - Mitigation: keep default `http://localhost:8000/api`, matching current `lib/api/index.ts`.

- **Risk:** CLI import tests become brittle if command modules import heavy dependencies.
  - Mitigation: scope the import-time assertion to `quanttool.cli.main` and `StockAnalyzer`; do not require every subcommand to be side-effect free.

- **Risk:** overview active state conflicts with existing store.
  - Mitigation: path-derived active key only drives nav highlighting; existing store remains for pages that explicitly set history or active page.

- **Risk:** frontend build may expose unrelated TypeScript issues.
  - Mitigation: fix only issues directly caused by this pass; document unrelated pre-existing build failures if they appear.

## Acceptance Criteria

- `quanttool analyze` uses the unified context workflow and keeps output/file behavior.
- Importing `quanttool.cli.main` no longer imports `StockAnalyzer`.
- Frontend overview action cards use static Tailwind classes.
- Frontend API base URL is configurable via `NEXT_PUBLIC_API_BASE_URL`.
- Header/sidebar active state follows direct navigation URLs.
- Market index fetch failures are visible to users.
- `python -m unittest discover -s tests -v` passes.
- Frontend `npm run build` passes or any pre-existing unrelated blockers are documented.
