# Frontend Workbench Redesign Implementation Plan

**Goal:** Deliver the first wave of the QuantTool frontend workbench redesign: stable styling/runtime, one primary app shell, shared UI primitives, and a clearer overview-to-scan-to-analysis flow.

**Architecture:** Stabilize Next.js dev/build assets first, then introduce a small UI primitive layer. Refactor shell navigation around a sidebar-led workflow and turn the overview page into an operational cockpit. Keep backend contracts and algorithm behavior unchanged.

**Tech Stack:** Next.js 14 App Router, React 18, TypeScript, Tailwind CSS, Axios, existing Zustand stores.

## Global Constraints

- Preserve unrelated dirty worktree changes.
- No algorithm scoring changes.
- No chart engine rewrite in this wave.
- No new routes.
- No marketing/landing page.
- Cards and controls use 8px radius or less.
- Header and sidebar must not duplicate primary navigation.
- Prefer existing project helpers before adding dependencies.

---

## File Structure

- Create or modify: `docs/superpowers/specs/2026-07-11-frontend-workbench-redesign-design.md`
- Create or modify: `docs/superpowers/plans/2026-07-11-frontend-workbench-redesign.md`
- Modify: `quanttool/web/frontend/package.json`
- Modify: `quanttool/web/frontend/next.config.js`
- Modify: `quanttool/web/frontend/app/globals.css`
- Modify: `quanttool/web/frontend/app/layout.tsx`
- Modify: `quanttool/web/frontend/app/page.tsx`
- Modify: `quanttool/web/frontend/app/scan/page.tsx`
- Modify: `quanttool/web/frontend/app/analyze/page.tsx`
- Modify: `quanttool/web/frontend/components/layout/AppHeader.tsx`
- Modify: `quanttool/web/frontend/components/layout/AppSidebar.tsx`
- Modify: `quanttool/web/frontend/components/layout/PageContainer.tsx`
- Create: `quanttool/web/frontend/components/ui/PageHeader.tsx`
- Create: `quanttool/web/frontend/components/ui/Section.tsx`
- Create: `quanttool/web/frontend/components/ui/MetricTile.tsx`
- Create: `quanttool/web/frontend/components/ui/StatusBadge.tsx`
- Create: `quanttool/web/frontend/components/ui/SegmentedControl.tsx`
- Update: `quanttool/web/frontend/components/ui/index.ts`

---

## Task 1: Stabilize Frontend Runtime Styling

- [ ] Inspect current dev server and `.next` artifact mismatch.
- [ ] Add a reliable clean-dev script so stale `.next` manifests do not survive between builds.
- [ ] Keep standalone production output available, but document the correct local commands.
- [ ] Verify loaded CSS on `localhost` returns `200` and includes Tailwind utilities.

## Task 2: Add Shared UI Primitives

- [ ] Add `PageHeader` with title, description, actions, and metadata slots.
- [ ] Add `Section` for unframed page sections and bounded tool surfaces.
- [ ] Add `MetricTile` for market, quote, score, and risk metrics.
- [ ] Add `StatusBadge` for state and signal labels.
- [ ] Add `SegmentedControl` for mode and tab controls.
- [ ] Export primitives from `components/ui/index.ts`.

## Task 3: Refactor App Shell

- [ ] Remove primary route navigation from `AppHeader`.
- [ ] Add a global stock-search affordance to the header.
- [ ] Keep theme and API-doc actions in the header.
- [ ] Reorganize sidebar groups into market/research/validation/model workflow groups.
- [ ] Keep collapsed sidebar behavior and active state from `getPageKeyFromPath`.
- [ ] Remove unused imports and manual duplicated route arrays where possible.

## Task 4: Rebuild Overview as Cockpit

- [ ] Replace feature-card grid with market strip, quick actions, candidate queue, recent work, and system status.
- [ ] Preserve market index fetching and refresh behavior.
- [ ] Make scan and analyze next steps visible above the fold.
- [ ] Use shared primitives and avoid nested cards.

## Task 5: Improve Scan and Analyze Entry Flow

- [ ] On scan, add preset/mode controls for fast versus deep scan intent.
- [ ] Keep current scan API payload and newly added `include_deep_data` control.
- [ ] Improve results action labels toward `分析`, `回测`, and candidate review.
- [ ] On analyze, improve empty state with direct next actions and common symbols.
- [ ] Keep existing chart/data sections intact unless touched for shell integration.

## Task 6: Verify

- [ ] Run frontend build.
- [ ] Run Python regression suite if frontend API/source invariants are touched.
- [ ] Start local frontend and browser-check `/`, `/scan`, `/analyze`.
- [ ] Confirm CSS asset status and SVG dimensions in browser.
- [ ] Run `git diff --check HEAD`.

## Acceptance Criteria

- Frontend route screenshots no longer show CSS failure or oversized SVGs.
- Header is a toolbar, not duplicate primary navigation.
- Sidebar is the primary route map.
- Overview first viewport supports market glance -> scan -> analyze.
- Scan exposes fast/deep scanning clearly.
- Analyze empty state gives an obvious start path.
- Build and relevant tests pass, or any unrelated blockers are documented with exact output.
