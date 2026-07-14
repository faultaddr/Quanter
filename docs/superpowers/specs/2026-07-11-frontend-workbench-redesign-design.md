# QuantTool Frontend Workbench Redesign Design

## Goal

Turn the frontend from a feature launcher into a practical A-share analysis workbench.

The first outcome should be a reliable local frontend, a clearer navigation model, and a more useful first screen for the daily workflow: check market state, scan candidates, open a stock analysis workbench, and continue into backtest or risk review.

## Current Problems

1. Runtime styling is unstable in local dev.
   - The page can render with a missing Next.js CSS asset, causing Tailwind classes to be ignored and SVG icons to expand to the viewport.
   - This blocks trustworthy visual review and must be fixed before design polish.

2. Navigation is duplicated.
   - `AppHeader` and `AppSidebar` both act as primary navigation.
   - The user has to choose between two competing route maps instead of following a single product path.

3. The overview page is a feature grid.
   - It lists modules, but it does not answer the user's likely first questions: what is the market doing, what should I inspect, and where do I continue?

4. Core pages are too large and mixed.
   - `app/analyze/page.tsx`, `app/risk/page.tsx`, `app/backtest/page.tsx`, `app/factors/page.tsx`, and `app/scan/page.tsx` exceed the preferred 300-line component limit.
   - Data fetching, form state, visual layout, and task-specific content are combined in single page files.

5. Visual language is too uniform.
   - The UI is heavily dark slate/blue and card-driven.
   - It should feel like a dense professional trading tool: restrained, readable, fast to scan, and action-oriented.

## Product Workflow

The frontend should guide this primary path:

1. `盘面概览`: read market state, data freshness, and today's candidate entry points.
2. `智能选股`: choose a scan preset, run fast or deep scanning, inspect scored candidates with reasons.
3. `股票分析`: open one candidate, review quote, K-line, indicators, chips, funds, risk, and signals.
4. `策略回测` / `组合风控`: validate or monitor selected ideas.

Secondary paths remain available for model management, factors, monitoring, and picks, but they should not compete visually with the primary scan-to-analysis flow.

## Architecture

### App Shell

Use one primary navigation surface:

- Sidebar owns route navigation.
- Header becomes a context toolbar with:
  - QuantTool brand and current workspace label.
  - Global stock search.
  - Data/API status.
  - Theme and API docs actions.

The sidebar groups routes by workflow:

- `市场`: overview, monitor.
- `研究`: scan, analyze, factors, picks.
- `验证`: backtest, risk.
- `模型`: model.

This removes duplicate header nav while keeping every current route.

### Shared UI Primitives

Add focused UI primitives before rewriting pages:

- `PageHeader`: title, description, actions, compact metadata.
- `Section`: unframed section shell with optional action slot.
- `MetricTile`: dense metric display for index, quote, risk, and score stats.
- `StatusBadge`: consistent state, score, signal, and data freshness badges.
- `SegmentedControl`: tabs and mode switching without repeated ad hoc button groups.
- `Toolbar`: compact horizontal control strip for filters/actions.

Cards remain for repeated records, tables, and bounded tool surfaces. Page sections should not become nested cards.

### Styling Direction

Keep dark mode as the default, but reduce one-note blue/slate dominance:

- Base: near-black charcoal and neutral slate.
- Positive/negative: market green/red with enough contrast.
- Accent: limited cyan/blue for primary actions and active route.
- Warning: amber for delayed/stale data and partial scan modes.
- Radius: 8px or less for cards and controls.
- Typography: smaller, denser headings inside work surfaces; no hero-scale text.

### Overview Page

Replace the feature-grid first impression with an operational cockpit:

- Market strip: major indices, breadth, freshness, and refresh state.
- Quick actions: `运行沪深300快扫`, `打开单股分析`, `查看最近结果`.
- Candidate queue: top scan candidates or empty state.
- Recent work: last analyzed symbols and recent scans.
- System state: API/data-source status.

### Scan Page

Reframe scan as a guided workflow:

- Preset row: `沪深300快扫`, `全市场趋势`, `低风险观察`, `深度基本面`.
- Primary action remains obvious and single.
- Advanced filters are available but collapsed by default.
- Results show score, timing, pass/fail reason, risk tags, and direct actions:
  - `分析`
  - `回测`
  - `加入观察`

### Analyze Page

Turn the analysis page into a stock workbench:

- Sticky top context: symbol search, current quote, final recommendation, data freshness.
- Left/main area: K-line and indicators.
- Right/secondary area: score breakdown, signals, risk, chips/funds summary.
- Tabs or segmented controls stay task-based: `走势`, `信号`, `筹码资金`, `风险`, `回测`.
- Empty state should offer common symbols and a direct search path.

## First Implementation Wave

The first wave should stay narrow:

1. Fix frontend dev/build runtime stability.
2. Refactor App Shell navigation and header toolbar.
3. Add shared UI primitives.
4. Rebuild overview as the operational cockpit.
5. Lightly reshape scan/analyze entry states without rewriting all charts.

Large page decomposition for `analyze`, `risk`, `backtest`, and `factors` can continue in later waves once the shell and design primitives are stable.

## Testing

Required validation:

- `npm run build`
- Browser sanity check for `/`, `/scan`, `/analyze`.
- CSS asset check: the loaded stylesheet must return `200` and contain Tailwind utilities.
- Python regression suite remains green because frontend API and scan changes are coupled to backend tests.

Suggested source-level tests:

- Frontend source invariant for no duplicate header primary nav.
- Frontend source invariant for shared primitives usage on overview.
- Existing `tests/test_frontend_cli_optimization.py` should be updated only if affected by intentional shell changes.

## Non-Goals

- No new product routes.
- No rewrite of chart engines.
- No algorithm scoring changes.
- No backend API schema changes beyond fields already introduced by scan optimization.
- No landing or marketing page.

## Acceptance Criteria

- Local frontend no longer shows oversized SVG/CSS failure in the browser.
- Header and sidebar no longer duplicate primary navigation.
- Overview makes the scan-to-analysis workflow visible above the fold.
- Scan page exposes fast/deep intent clearly.
- Analyze page empty state and header make starting analysis obvious.
- New UI primitives reduce repeated layout/control code.
- Touched frontend component files stay near or under the 300-line guideline where feasible in this wave.
