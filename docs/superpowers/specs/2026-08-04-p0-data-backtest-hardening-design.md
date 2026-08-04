# P0 Data and Backtest Hardening Design

**Date:** 2026-08-04
**Status:** Approved for implementation planning
**Scope:** First production-readiness remediation batch for QuantTool

## Objective

Make the existing daily A-share selection and backtest path fail closed instead of producing unverifiable results. This batch removes committed credentials and process-wide network side effects, prevents simulated market data from entering production, and makes backtest execution obey dated A-share rules and costs without look-ahead.

The first production target remains an internal, post-close stock-selection and paper-portfolio system. This batch does not authorize public recommendations or automated brokerage execution.

## Chosen Approach

Use a thin-boundary hardening approach. Preserve the current provider, service, API, and frontend contracts where possible, but place explicit runtime, provenance, validation, symbol, rule, and fee boundaries in front of the existing implementations.

The alternatives were rejected for this batch:

- Patching only the existing large files would be faster initially but would leave security, provider selection, and trading rules coupled to unrelated code.
- Building a parallel replacement quant core would provide stronger isolation but would make the first remediation too large to verify safely.

## In Scope

- Remove hard-coded data-source credentials from the current tree.
- Stop data-provider imports from mutating global proxy environment variables.
- Distinguish `test`, `development`, and `production` runtime modes through `QUANTTOOL_ENV`; the default is `development`.
- Allow simulated providers only in `test` mode and through explicit test construction.
- Consolidate the registered Ashare provider onto the existing real Sina/Tencent fetch path and remove its random-data behavior.
- Validate daily OHLCV frames and attach provider provenance before strategy use.
- Correct backtest event ordering, previous-close handling, symbol normalization, board classification, lot sizing, T+1 handling, dated transaction fees, and structured order rejection.
- Support dated A-share rules from 2017-01-01 through the current rule set.
- Add offline regression and golden-case tests for every changed behavior.

## Out of Scope

- API authentication and role-based access control.
- Durable task queues, scheduler replacement, and WebSocket repair.
- ML leakage, model calibration, model registry, and factor promotion gates.
- Frontend interaction changes.
- Automated trading or brokerage connectivity.
- Destructive Git-history rewriting. Credential revocation and history cleanup remain explicit operator actions.

## Architecture

### Runtime and Provider Policy

Add a small runtime policy module with an enum for `test`, `development`, and `production`. Provider registration consumes this policy rather than inspecting ad hoc environment variables throughout the codebase.

Production registration contains real providers only. A simulated provider is a test utility and cannot be imported into the production registration path. When no real source is usable, the application raises a data-unavailable error; it never manufactures a DataFrame.

The existing Ashare public provider name remains stable. Its implementation delegates to the real Sina-first, Tencent-fallback adapter already present in the repository. This avoids a broad consumer migration while eliminating the separate random-data implementation.

### Data Validation and Provenance

Every provider result passes through one validator before it reaches analysis or strategy code. For daily bars the validator requires:

- non-empty requested results unless the provider reports a structured no-data outcome;
- timestamps within the requested interval, strictly increasing, and unique;
- numeric open, high, low, close, volume, and amount fields;
- `high >= max(open, close)`, `low <= min(open, close)`, and `high >= low`;
- non-negative volume and amount.

Validated frames carry `quanttool_provenance` in `DataFrame.attrs`, containing provider name, retrieval timestamp, frequency, adjustment mode, and `simulated=false`. This is an intentionally narrow compatibility bridge; a later data-platform batch can replace it with an immutable domain object.

Fallback preserves the source that actually supplied the data. For example, a failed Sina request followed by a successful Tencent request records Tencent rather than Ashare or realtime as the concrete source.

### Network and Credential Handling

Data-provider modules do not delete or overwrite `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, or `NO_PROXY` at import time. A provider that needs special proxy behavior configures its own HTTP session without mutating the process environment.

TuShare tokens and EastMoney cookies are read only from runtime configuration. The repository contains variable names and documentation but no example secret values. A missing optional credential disables that provider only; it does not disable the token-free Sina/Tencent path.

### Backtest Rule Boundary

Keep the existing backtest engine entry points and place three focused responsibilities behind them:

1. A symbol normalizer converts forms such as `600000`, `SH600000`, and `600000.SH` into a canonical exchange and six-digit code.
2. A dated rule resolver returns board, price-limit, lot-size, listing-stage, and sellability rules for a security and trade date.
3. A dated fee schedule returns commission, stamp-tax, and transfer-fee behavior for a trade date and side.

Unknown symbols, unsupported dates, or missing rules are errors. The engine must not apply a generic ten-percent rule as a fallback.

### Backtest Event Flow

For each symbol, bars are sorted by timestamp before the run starts. On date `T`, previous close comes only from the preceding valid bar for that symbol. No initialization reads from the tail of the complete dataset.

Strategies observe the completed `T` bar and may create a pending order. The earliest execution point is the next available tradable bar in the supplied dataset. T+1 sellability also follows actual bar timestamps instead of weekend arithmetic.

Before execution, the engine resolves the dated rule and checks market identity, listing stage, suspension, price limit, T+1, lot size, and available cash or position. Accepted quantities are rounded down to the permitted lot. Rejected orders produce a structured reason code and do not mutate cash or positions.

All accepted fills pass through one transaction-cost function. Gross value, commission, stamp tax, transfer fee, slippage, and net cash impact are recorded separately. Portfolio returns and trade profit use the net values.

## Error Handling

- No usable real provider: fail the scan with a data-unavailable error.
- A provider returns invalid bars: reject that result and try the next configured real provider; retain validation details in logs.
- More symbols are missing than the configured batch-completeness threshold: fail the batch and return the missing-symbol list. `production` defaults to zero tolerated missing symbols; a non-zero threshold must be set explicitly for a named job and is included in that job's audit record.
- Missing optional token or cookie: disable only the corresponding optional provider.
- Unknown market, unsupported rule date, or absent fee schedule: stop the affected backtest with a configuration error.
- Suspension, price limit, T+1, lot, cash, or position constraint: reject the order with a stable reason code and continue the backtest.

Errors exposed to callers must be structured and must not include credential values, cookies, or raw response bodies.

## Testing Strategy

Implementation follows red-green-refactor. Unit tests are offline and use deterministic fixtures.

### Data and Security Tests

- Importing provider modules leaves all proxy environment variables unchanged.
- A current-tree secret scan finds no embedded Token or Cookie value.
- Production policy cannot register or construct a simulated provider.
- Exhausted real providers raise a data-unavailable error rather than returning random rows.
- Sina failure followed by Tencent success records Tencent provenance.
- Duplicate timestamps, descending timestamps, invalid OHLC relationships, and negative volume each fail validation.

### Backtest Golden Cases

- Appending future bars does not change signals, fills, or positions before a fixed cutoff.
- A signal generated from bar `T` cannot fill on bar `T`.
- Price-limit checks use the preceding bar close.
- `600000`, `SH600000`, and `600000.SH` resolve identically; `002` resolves to the Shenzhen main board, while `300` and `688` resolve to their correct boards.
- Main-board quantities use 100-share lots; STAR-market quantities obey its minimum declaration rule; insufficient cash rounds quantity down.
- A Friday or pre-holiday purchase becomes sellable on the next supplied trading bar, not the next weekday.
- Fee calculations are checked immediately before and after the 2022 transfer-fee and 2023 stamp-tax effective dates.
- Unknown market, unsupported date, suspension, price limit, T+1, and insufficient cash generate stable rejection codes.
- Win rate uses closed trades only, and every transaction fee reduces net return.

## Acceptance Criteria

- Every changed production behavior has a regression test that was observed failing before the change and passing after it.
- The existing 92-test `unittest` suite continues to pass.
- Python compilation succeeds for the package and tests.
- A current-tree scan reports no hard-coded credential value.
- Offline tests require no public network access.
- One read-only live smoke test obtains a small Sina/Tencent daily sample and verifies dates, OHLC relationships, and concrete provenance. Network failure is reported separately and does not invalidate offline correctness tests.
- Frontend build is required only if an API contract or frontend dependency changes.
- The final handoff lists credential revocation and Git-history cleanup as unresolved launch blockers until the operator confirms them.

## Rollout and Compatibility

The existing provider name, backtest service entry points, and frontend contracts remain stable. Production behavior becomes stricter: calls that previously returned simulated, mislabeled, or rule-defaulted output now fail explicitly. Development mode also uses real providers by default; deterministic simulation is confined to tests.

The implementation is divided into independently verifiable commits: runtime and credential hardening, provider validation and provenance, backtest rules and fees, then event-flow integration and end-to-end regression coverage.
