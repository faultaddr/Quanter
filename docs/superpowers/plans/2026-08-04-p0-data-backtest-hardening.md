# P0 Data and Backtest Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make QuantTool's first production stock-selection path use real, attributable market data and produce A-share backtests without look-ahead, invalid board rules, fractional lots, or omitted transaction costs.

**Architecture:** Preserve the registered `ashare` name and existing backtest service entry points, but add small runtime, data-integrity, symbol-rule, and fee-schedule boundaries. The provider path fails closed, while the backtest engine queues signals for the next supplied trading bar and resolves all execution constraints through dated A-share rules.

**Tech Stack:** Python 3.11, pandas, Pydantic models, standard-library `unittest`, existing QuantTool registry and backtest engine.

## Global Constraints

- Follow PEP 8, use type annotations, and use Google-style docstrings.
- Keep every new Python file below 800 lines.
- Preserve existing provider names, backtest service entry points, and frontend request contracts.
- Use `QUANTTOOL_ENV` with exactly `test`, `development`, or `production`; default to `development`.
- Permit deterministic simulated data only through direct construction in `test` mode.
- Support dated A-share rules from 2017-01-01 onward; reject earlier dates.
- Do not rewrite Git history or claim that removing current-tree credentials revokes leaked credentials.
- Write a failing `unittest`, observe the expected failure, then write production code.
- Do not require public network access for automated tests.

---

## File Structure

- `quanttool/core/runtime.py`: Parse and enforce the process runtime mode.
- `quanttool/infrastructure/data_providers/validation.py`: Validate OHLCV frames, attach immutable provenance metadata, and enforce batch completeness.
- `quanttool/infrastructure/data_providers/historical/ashare_provider.py`: Keep the public provider contract while delegating bars to the real Sina/Tencent fetcher.
- `quanttool/infrastructure/data_providers/historical/enhanced_fetcher.py`: Preserve proxy state, expose concrete Sina/Tencent provenance, and load optional credentials only from runtime configuration.
- `quanttool/infrastructure/data_providers/historical/csv_provider.py`: Remain a direct, test-only fixture provider and leave the production registry.
- `quanttool/backtest/a_share_rules.py`: Normalize symbols and resolve dated board, limit, and lot rules.
- `quanttool/backtest/fee_schedule.py`: Resolve dated commission, stamp-tax, and transfer-fee costs.
- `quanttool/backtest/ashare_constraints.py`: Adapt the existing constraint API to the dated rule and fee modules.
- `quanttool/domain/models/__init__.py`: Add backward-compatible fill-cost and rejection fields to `Trade` and `Order`.
- `quanttool/backtest/engine.py`: Queue orders, execute them on the next supplied bar, enforce lots/T+1, and account for net costs.
- `tests/test_runtime_provider_policy.py`: Runtime isolation, proxy preservation, and current-tree credential tests.
- `tests/test_market_data_integrity.py`: Deterministic validation, provenance, fallback, and production-provider tests.
- `tests/test_a_share_rules.py`: Dated market-rule, lot, and fee golden cases.
- `tests/test_backtest_engine_integrity.py`: Event ordering, anti-look-ahead, cost, rejection, and metric regressions.
- `README.md`: Document runtime mode and unresolved credential-rotation requirements.

---

### Task 1: Runtime, Credential, and Network Side-Effect Hardening

**Files:**
- Create: `quanttool/core/runtime.py`
- Modify: `quanttool/infrastructure/data_providers/historical/enhanced_fetcher.py:1-50,382-487,2300-2330`
- Modify: `quanttool/infrastructure/data_providers/historical/csv_provider.py:1-25`
- Test: `tests/test_runtime_provider_policy.py`

**Interfaces:**
- Produces: `RuntimeMode`, `get_runtime_mode(env: Optional[Mapping[str, str]] = None) -> RuntimeMode`, and `require_test_mode(feature: str, env: Optional[Mapping[str, str]] = None) -> None`.
- Preserves: `create_data_fetcher_with_credentials() -> EnhancedDataFetcher`, but makes it environment-only for backward compatibility.
- Consumed by: Task 2 provider construction and every test-only simulated provider.

- [ ] **Step 1: Write runtime parsing and test-only-provider failure tests**

Create `tests/test_runtime_provider_policy.py` with these cases:

```python
import os
import subprocess
import sys
import unittest
from unittest.mock import patch

from quanttool.core.errors import ConfigurationError


class RuntimePolicyTests(unittest.TestCase):
    def test_runtime_defaults_to_development(self):
        from quanttool.core.runtime import RuntimeMode, get_runtime_mode

        self.assertEqual(get_runtime_mode({}), RuntimeMode.DEVELOPMENT)

    def test_runtime_rejects_unknown_value(self):
        from quanttool.core.runtime import get_runtime_mode

        with self.assertRaises(ConfigurationError):
            get_runtime_mode({"QUANTTOOL_ENV": "staging"})

    def test_csv_provider_requires_test_mode(self):
        from quanttool.infrastructure.data_providers.historical.csv_provider import CSVProvider

        with patch.dict(os.environ, {"QUANTTOOL_ENV": "production"}, clear=False):
            with self.assertRaises(ConfigurationError):
                CSVProvider()

    def test_enhanced_fetcher_import_preserves_proxy_environment(self):
        code = """
import json, os
before = {key: os.environ.get(key) for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY')}
import quanttool.infrastructure.data_providers.historical.enhanced_fetcher
after = {key: os.environ.get(key) for key in before}
print(json.dumps({'before': before, 'after': after}, sort_keys=True))
"""
        env = os.environ.copy()
        env.update({
            "HTTP_PROXY": "http://127.0.0.1:18080",
            "HTTPS_PROXY": "http://127.0.0.1:18443",
            "ALL_PROXY": "socks5://127.0.0.1:1080",
            "NO_PROXY": "localhost",
        })
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        payload = __import__("json").loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(payload["before"], payload["after"])

    def test_fetcher_factory_reads_optional_credentials_from_environment(self):
        from quanttool.infrastructure.data_providers.historical import enhanced_fetcher

        with patch.dict(
            os.environ,
            {
                "TUSHARE_TOKEN": "token-from-runtime",
                "EASTMONEY_COOKIE": "cookie-from-runtime",
            },
            clear=False,
        ), patch.object(enhanced_fetcher, "EnhancedDataFetcher") as fetcher_type:
            enhanced_fetcher.create_data_fetcher_with_credentials()

        fetcher_type.assert_called_once_with(
            tushare_token="token-from-runtime",
            eastmoney_cookie="cookie-from-runtime",
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the new tests and observe the expected red state**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_runtime_provider_policy -v
```

Expected: errors because `quanttool.core.runtime` does not exist, plus failures showing the CSV provider is constructible in production, proxy variables change on import, and the compatibility factory does not read credentials exclusively from runtime environment variables.

- [ ] **Step 3: Implement the runtime policy**

Create `quanttool/core/runtime.py`:

```python
"""Runtime-mode policy for production-sensitive QuantTool components."""

from enum import Enum
import os
from typing import Mapping, Optional

from .errors import ConfigurationError


class RuntimeMode(str, Enum):
    TEST = "test"
    DEVELOPMENT = "development"
    PRODUCTION = "production"


def get_runtime_mode(
    env: Optional[Mapping[str, str]] = None,
) -> RuntimeMode:
    values = os.environ if env is None else env
    raw = values.get("QUANTTOOL_ENV", RuntimeMode.DEVELOPMENT.value).strip().lower()
    try:
        return RuntimeMode(raw)
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in RuntimeMode)
        raise ConfigurationError(
            f"Invalid QUANTTOOL_ENV={raw!r}; expected one of: {allowed}"
        ) from exc


def require_test_mode(
    feature: str,
    env: Optional[Mapping[str, str]] = None,
) -> None:
    mode = get_runtime_mode(env)
    if mode is not RuntimeMode.TEST:
        raise ConfigurationError(
            f"{feature} is test-only and cannot run in {mode.value} mode"
        )
```

- [ ] **Step 4: Remove proxy mutation and embedded credentials**

In `enhanced_fetcher.py`:

- Delete the module-level loop that removes proxy variables and the assignment to `NO_PROXY`.
- Delete every literal token and Cookie value.
- Keep `create_data_fetcher_with_credentials`, but reduce it to:

```python
def create_data_fetcher_with_credentials() -> EnhancedDataFetcher:
    """Create a fetcher using optional credentials from the environment."""
    return EnhancedDataFetcher(
        tushare_token=os.getenv("TUSHARE_TOKEN"),
        eastmoney_cookie=os.getenv("EASTMONEY_COOKIE"),
    )
```

- Remove the constructor guard that requires TuShare or AkShare. The built-in Sina/Tencent Ashare path is always a usable configured source; optional providers remain disabled when their dependency or credential is absent.

- [ ] **Step 5: Make the CSV fixture provider test-only and unregistered**

In `csv_provider.py`:

- Remove `@registry.register(ComponentType.DATA_PROVIDER, "csv_mock")` and the now-unused registry imports.
- Call `require_test_mode("CSVProvider")` at the start of `CSVProvider.__init__`.
- Keep direct construction available to tests after setting `QUANTTOOL_ENV=test`.

- [ ] **Step 6: Run the focused tests and the import smoke test**

Run:

```bash
.venv-mcp/bin/python -m unittest tests.test_runtime_provider_policy tests.test_smoke.ImportSmokeTests -v
rg -n 'tushare_token\s*=\s*["\x27][A-Za-z0-9]{20,}|qgqp_b_id=' quanttool --glob '*.py'
```

Expected: all focused runtime tests and import smoke tests pass; the release-time secret scan prints no matches and no secret values.

- [ ] **Step 7: Commit the runtime hardening**

```bash
git add quanttool/core/runtime.py quanttool/infrastructure/data_providers/historical/enhanced_fetcher.py quanttool/infrastructure/data_providers/historical/csv_provider.py tests/test_runtime_provider_policy.py
git commit -m "fix: harden provider runtime configuration"
```

---

### Task 2: Real Ashare Provider, Validation, and Provenance

**Files:**
- Create: `quanttool/infrastructure/data_providers/validation.py`
- Replace implementation: `quanttool/infrastructure/data_providers/historical/ashare_provider.py`
- Modify: `quanttool/infrastructure/data_providers/historical/enhanced_fetcher.py:235-371`
- Test: `tests/test_market_data_integrity.py`

**Interfaces:**
- Produces: `DataProvenance`, `validate_market_data(...) -> pd.DataFrame`, and `validate_batch_completeness(...) -> None`.
- Produces: `AShareProvider(fetcher: type[AshareFetcher] = AshareFetcher, max_missing_ratio: Optional[float] = None)` implementing `IDataProvider` without synthetic output.
- Provenance location: `frame.attrs["quanttool_provenance"]` as a plain dictionary.
- Consumes: `RuntimeMode` from Task 1 and existing `DataNotAvailableError` / `ValidationError`.

- [ ] **Step 1: Write validator and provenance failure tests**

Create `tests/test_market_data_integrity.py` with a deterministic builder and these core assertions:

```python
from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import pandas as pd

from quanttool.core.errors import DataNotAvailableError, ValidationError


def make_daily_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-08-03", "2026-08-04"]),
        "open": [10.0, 10.2],
        "high": [10.5, 10.6],
        "low": [9.9, 10.1],
        "close": [10.3, 10.4],
        "volume": [1000.0, 1200.0],
        "amount": [10300.0, 12480.0],
    })


class MarketDataValidationTests(unittest.TestCase):
    def test_validation_attaches_concrete_provenance(self):
        from quanttool.infrastructure.data_providers.validation import (
            DataProvenance,
            validate_market_data,
        )

        result = validate_market_data(
            make_daily_frame(),
            start_date=datetime(2026, 8, 3),
            end_date=datetime(2026, 8, 4),
            provenance=DataProvenance(
                provider="tencent",
                retrieved_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
                frequency="1d",
                adjustment="qfq",
                simulated=False,
            ),
        )
        self.assertEqual(result.attrs["quanttool_provenance"]["provider"], "tencent")
        self.assertFalse(result.attrs["quanttool_provenance"]["simulated"])

    def test_validation_rejects_duplicate_timestamp(self):
        from quanttool.infrastructure.data_providers.validation import (
            DataProvenance,
            validate_market_data,
        )

        frame = make_daily_frame()
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
        with self.assertRaises(ValidationError):
            validate_market_data(
                frame,
                datetime(2026, 8, 3),
                datetime(2026, 8, 4),
                DataProvenance("sina", datetime.now(timezone.utc), "1d", "qfq", False),
            )

    def test_validation_rejects_bad_ohlc_and_negative_volume(self):
        from quanttool.infrastructure.data_providers.validation import (
            DataProvenance,
            validate_market_data,
        )

        provenance = DataProvenance(
            "sina", datetime.now(timezone.utc), "1d", "qfq", False
        )
        bad_ohlc = make_daily_frame()
        bad_ohlc.loc[0, "high"] = 9.0
        with self.assertRaises(ValidationError):
            validate_market_data(
                bad_ohlc, datetime(2026, 8, 3), datetime(2026, 8, 4), provenance
            )
        negative_volume = make_daily_frame()
        negative_volume.loc[0, "volume"] = -1
        with self.assertRaises(ValidationError):
            validate_market_data(
                negative_volume,
                datetime(2026, 8, 3),
                datetime(2026, 8, 4),
                provenance,
            )

    def test_validation_rejects_descending_out_of_range_and_non_numeric_data(self):
        from quanttool.infrastructure.data_providers.validation import (
            DataProvenance,
            validate_market_data,
        )

        provenance = DataProvenance(
            "sina", datetime.now(timezone.utc), "1d", "qfq", False
        )
        cases = []
        descending = make_daily_frame().iloc[::-1].reset_index(drop=True)
        cases.append(descending)
        out_of_range = make_daily_frame()
        out_of_range.loc[0, "timestamp"] = pd.Timestamp("2026-08-02")
        cases.append(out_of_range)
        non_numeric = make_daily_frame()
        non_numeric.loc[0, "close"] = "not-a-price"
        cases.append(non_numeric)
        negative_amount = make_daily_frame()
        negative_amount.loc[0, "amount"] = -1
        cases.append(negative_amount)
        for frame in cases:
            with self.subTest(frame=frame):
                with self.assertRaises(ValidationError):
                    validate_market_data(
                        frame,
                        datetime(2026, 8, 3),
                        datetime(2026, 8, 4),
                        provenance,
                    )

    def test_ashare_fallback_records_tencent(self):
        from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher

        with patch.object(AshareFetcher, "_get_price_sina", return_value=pd.DataFrame()), patch.object(
            AshareFetcher, "_get_price_day_tx", return_value=make_daily_frame()
        ):
            result = AshareFetcher.get_price("600000.SH", count=2, frequency="1d")
        self.assertEqual(result.attrs["concrete_source"], "tencent")

    def test_ashare_provider_never_manufactures_missing_symbol(self):
        from quanttool.infrastructure.data_providers.historical.ashare_provider import AShareProvider

        class EmptyFetcher:
            @classmethod
            def get_price(cls, *args, **kwargs):
                return pd.DataFrame()

        provider = AShareProvider(fetcher=EmptyFetcher, max_missing_ratio=0.0)
        with self.assertRaises(DataNotAvailableError):
            provider.get_bars(
                ["600000.SH"], datetime(2026, 8, 3), datetime(2026, 8, 4), "1d"
            )
```

- [ ] **Step 2: Run the focused tests and observe the expected red state**

```bash
.venv-mcp/bin/python -m unittest tests.test_market_data_integrity -v
```

Expected: import errors for the validation module, missing `fetcher` constructor support, and missing concrete-source provenance.

- [ ] **Step 3: Implement deterministic validation and batch completeness**

Create `validation.py` with:

```python
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Mapping, Sequence

import pandas as pd

from ...core.errors import DataNotAvailableError, ValidationError


@dataclass(frozen=True)
class DataProvenance:
    provider: str
    retrieved_at: datetime
    frequency: str
    adjustment: str
    simulated: bool = False


def validate_market_data(
    frame: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    provenance: DataProvenance,
) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(frame.columns)
    if frame.empty or missing:
        raise ValidationError(f"Invalid market data: empty={frame.empty}, missing={sorted(missing)}")
    result = frame.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    if not result["timestamp"].is_monotonic_increasing or result["timestamp"].duplicated().any():
        raise ValidationError("Market-data timestamps must be strictly increasing and unique")
    if result["timestamp"].min() < pd.Timestamp(start_date) or result["timestamp"].max() > pd.Timestamp(end_date):
        raise ValidationError("Market-data timestamp is outside the requested interval")
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if result[numeric].isna().any().any():
        raise ValidationError("Market data contains non-numeric values")
    if not ((result["high"] >= result[["open", "close"]].max(axis=1)) &
            (result["low"] <= result[["open", "close"]].min(axis=1)) &
            (result["high"] >= result["low"])).all():
        raise ValidationError("Market data violates OHLC invariants")
    if (result[["volume", "amount"]] < 0).any().any():
        raise ValidationError("Market data contains negative volume or amount")
    result.attrs["quanttool_provenance"] = asdict(provenance)
    return result


def validate_batch_completeness(
    requested: Sequence[str],
    results: Mapping[str, pd.DataFrame],
    max_missing_ratio: float,
) -> None:
    missing = sorted(set(requested) - set(results))
    ratio = len(missing) / len(requested) if requested else 0.0
    if ratio > max_missing_ratio:
        raise DataNotAvailableError(
            f"Market-data batch incomplete: missing={missing}, ratio={ratio:.4f}"
        )
```

- [ ] **Step 4: Record the concrete Ashare source**

In `AshareFetcher.get_price`, set `df.attrs["concrete_source"] = "sina"` immediately before returning a non-empty Sina frame, and set it to `"tencent"` before returning a non-empty Tencent frame. Do not relabel fallback output as realtime.

- [ ] **Step 5: Replace the synthetic AShareProvider with the real adapter**

Implement the constructor and daily-bar path as:

```python
def __init__(self, fetcher=AshareFetcher, max_missing_ratio=None):
    self.fetcher = fetcher
    if max_missing_ratio is None:
        self.max_missing_ratio = 0.0 if get_runtime_mode() is RuntimeMode.PRODUCTION else 0.05
    else:
        self.max_missing_ratio = max_missing_ratio
    self._initialized = False
```

For every requested symbol, call `fetcher.get_price` with the requested end date and enough calendar days to cover the interval, filter the returned frame to the exact interval, attach `timeframe` and `symbol`, then call `validate_market_data` with the concrete source. After the loop, call `validate_batch_completeness`.

Implement `get_latest_bar` through the same real path with `count=1`. Raise `UnsupportedOperationError` from `get_supported_symbols`, `search_symbols`, and `get_calendar` until an attributable real endpoint is implemented; never return hard-coded symbols or a weekday calendar.

- [ ] **Step 6: Run the focused provider tests**

```bash
.venv-mcp/bin/python -m unittest tests.test_market_data_integrity tests.test_runtime_provider_policy -v
```

Expected: every test passes without a network request.

- [ ] **Step 7: Commit the real-provider boundary**

```bash
git add quanttool/infrastructure/data_providers/validation.py quanttool/infrastructure/data_providers/historical/ashare_provider.py quanttool/infrastructure/data_providers/historical/enhanced_fetcher.py tests/test_market_data_integrity.py
git commit -m "fix: enforce real attributed market data"
```

---

### Task 3: Dated A-Share Symbol, Trading-Rule, and Fee Boundaries

**Files:**
- Create: `quanttool/backtest/a_share_rules.py`
- Create: `quanttool/backtest/fee_schedule.py`
- Modify: `quanttool/backtest/ashare_constraints.py`
- Test: `tests/test_a_share_rules.py`

**Interfaces:**
- Produces: `NormalizedSymbol`, `TradingRule`, `normalize_symbol(symbol: str) -> NormalizedSymbol`, `resolve_trading_rule(symbol: str, trade_date: date, stock_name: Optional[str] = None, listing_session: Optional[int] = None) -> TradingRule`, and `round_buy_quantity(desired: float, rule: TradingRule) -> int`.
- Produces: `FeeRates`, `TransactionCostBreakdown`, `resolve_fee_rates(trade_date: date, commission_rate: float = 0.0003, min_commission: float = 5.0) -> FeeRates`, and `calculate_transaction_cost(price: float, quantity: int, side: str, trade_date: date, commission_rate: float = 0.0003, min_commission: float = 5.0) -> TransactionCostBreakdown`.
- Consumed by: `ASShareConstraints` and Task 4's backtest engine.

- [ ] **Step 1: Write symbol, dated-limit, lot, and fee golden tests**

Create `tests/test_a_share_rules.py`:

```python
from datetime import date
import unittest

from quanttool.core.errors import BacktestError


class AShareRuleTests(unittest.TestCase):
    def test_symbol_forms_normalize_identically(self):
        from quanttool.backtest.a_share_rules import normalize_symbol

        expected = normalize_symbol("600000")
        self.assertEqual(normalize_symbol("SH600000"), expected)
        self.assertEqual(normalize_symbol("600000.SH"), expected)

    def test_board_classification_and_lots(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        main = resolve_trading_rule("002415.SZ", date(2026, 8, 4))
        chinext = resolve_trading_rule("300750.SZ", date(2026, 8, 4))
        star = resolve_trading_rule("688981.SH", date(2026, 8, 4))
        bse = resolve_trading_rule("920001.BJ", date(2026, 8, 4))
        self.assertEqual((main.board, main.price_limit, main.min_buy_quantity, main.buy_increment), ("main", 0.10, 100, 100))
        self.assertEqual((chinext.board, chinext.price_limit), ("chinext", 0.20))
        self.assertEqual((star.board, star.price_limit, star.min_buy_quantity, star.buy_increment), ("star", 0.20, 200, 1))
        self.assertEqual((bse.board, bse.price_limit, bse.min_buy_quantity, bse.buy_increment), ("bse", 0.30, 100, 1))

    def test_chinext_and_main_st_limits_are_dated(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        self.assertEqual(resolve_trading_rule("300750.SZ", date(2020, 8, 23)).price_limit, 0.10)
        self.assertEqual(resolve_trading_rule("300750.SZ", date(2020, 8, 24)).price_limit, 0.20)
        self.assertEqual(resolve_trading_rule("600000.SH", date(2026, 7, 5), stock_name="ST浦发").price_limit, 0.05)
        self.assertEqual(resolve_trading_rule("600000.SH", date(2026, 7, 6), stock_name="ST浦发").price_limit, 0.10)

    def test_first_five_registration_sessions_have_no_limit(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        self.assertIsNone(resolve_trading_rule("688981.SH", date(2026, 8, 4), listing_session=5).price_limit)
        self.assertEqual(resolve_trading_rule("688981.SH", date(2026, 8, 4), listing_session=6).price_limit, 0.20)

    def test_buy_quantities_obey_board_rules(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule, round_buy_quantity

        main = resolve_trading_rule("600000.SH", date(2026, 8, 4))
        star = resolve_trading_rule("688981.SH", date(2026, 8, 4))
        self.assertEqual(round_buy_quantity(299.9, main), 200)
        self.assertEqual(round_buy_quantity(199.9, star), 0)
        self.assertEqual(round_buy_quantity(245.9, star), 245)

    def test_unknown_symbol_and_pre_2017_date_fail_closed(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        with self.assertRaises(BacktestError):
            resolve_trading_rule("123456", date(2026, 8, 4))
        with self.assertRaises(BacktestError):
            resolve_trading_rule("600000.SH", date(2016, 12, 31))

    def test_board_rules_do_not_apply_before_board_launch(self):
        from quanttool.backtest.a_share_rules import resolve_trading_rule

        with self.assertRaises(BacktestError):
            resolve_trading_rule("688981.SH", date(2019, 7, 21))
        with self.assertRaises(BacktestError):
            resolve_trading_rule("920001.BJ", date(2021, 11, 14))

    def test_fee_schedule_changes_on_official_dates(self):
        from quanttool.backtest.fee_schedule import resolve_fee_rates

        self.assertEqual(resolve_fee_rates(date(2022, 4, 28)).transfer_fee_rate, 0.00002)
        self.assertEqual(resolve_fee_rates(date(2022, 4, 29)).transfer_fee_rate, 0.00001)
        self.assertEqual(resolve_fee_rates(date(2023, 8, 27)).stamp_tax_rate, 0.001)
        self.assertEqual(resolve_fee_rates(date(2023, 8, 28)).stamp_tax_rate, 0.0005)

    def test_stamp_tax_applies_to_sell_only(self):
        from quanttool.backtest.fee_schedule import calculate_transaction_cost

        buy = calculate_transaction_cost(10.0, 1000, "buy", date(2026, 8, 4))
        sell = calculate_transaction_cost(10.0, 1000, "sell", date(2026, 8, 4))
        self.assertEqual(buy.stamp_tax, 0.0)
        self.assertEqual(sell.stamp_tax, 5.0)
        self.assertGreater(sell.total_fee, buy.total_fee)
```

- [ ] **Step 2: Run the golden tests and observe missing-module failures**

```bash
.venv-mcp/bin/python -m unittest tests.test_a_share_rules -v
```

Expected: import failures for `a_share_rules` and `fee_schedule`.

- [ ] **Step 3: Implement symbol normalization and dated rules**

Create immutable `NormalizedSymbol` and `TradingRule` dataclasses. Normalize explicit `.SH`, `.SZ`, `.BJ`, `SH`/`SZ`/`BJ` prefixes, and `.XSHG`/`.XSHE`; infer bare codes only for recognized prefixes. Resolve:

- Shanghai/Shenzhen main board: 10%, minimum/increment 100.
- ChiNext: 10% through 2020-08-23 and 20% from 2020-08-24, minimum/increment 100.
- STAR: 20%, minimum 200 and increment 1 from the board launch on 2019-07-22; reject earlier dates.
- Beijing: 30%, minimum 100 and increment 1 from the exchange launch on 2021-11-15; recognize explicit `.BJ` and bare `4`, `8`, or `92` prefixes and reject earlier dates.
- Main-board risk-warning stocks: 5% through 2026-07-05 and 10% from 2026-07-06.
- Registration-based IPO sessions 1 through 5: `price_limit=None` when `listing_session` is supplied; session 6 uses the board rule.
- Dates before 2017-01-01 and unrecognized symbols: `BacktestError`.

Use `Decimal(str(prev_close)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)` when calculating limit prices; do not use Python's banker-style `round`.

- [ ] **Step 4: Implement the dated fee schedule**

Create `FeeRates` and `TransactionCostBreakdown` dataclasses. Encode:

```python
stamp_tax_rate = 0.001 if trade_date < date(2023, 8, 28) else 0.0005
transfer_fee_rate = 0.00002 if trade_date < date(2022, 4, 29) else 0.00001
```

Commission and transfer fee apply to both sides; stamp tax applies to sells only. Reject non-positive prices, non-positive quantities, unknown sides, and dates before 2017-01-01 with `BacktestError`.

- [ ] **Step 5: Delegate ASShareConstraints to the new boundaries**

Replace `get_market_type`, limit-rate selection, limit-price calculation, and `apply_transaction_costs` internals with calls to `resolve_trading_rule` and `calculate_transaction_cost`. Keep the existing public method names so callers continue to work. Add an optional `trade_date` argument to transaction-cost calculation, defaulting only to the caller-provided execution timestamp inside the engine; tests must not depend on wall-clock time.

- [ ] **Step 6: Run rule and constraint tests**

```bash
.venv-mcp/bin/python -m unittest tests.test_a_share_rules -v
```

Expected: all dated rule, lot, and fee tests pass.

- [ ] **Step 7: Commit the dated rule boundary**

```bash
git add quanttool/backtest/a_share_rules.py quanttool/backtest/fee_schedule.py quanttool/backtest/ashare_constraints.py tests/test_a_share_rules.py
git commit -m "fix: version A-share trading rules and fees"
```

Rule sources to retain in module comments:

- Shanghai 2026 rules and risk-warning change: `https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml`
- Beijing trading rules: `https://www.bse.cn/jygl_list/200028217.html`
- 2023 stamp-tax change: `https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html`
- 2022 transfer-fee change: `https://www.chinaclear.cn/zdjs/gszb/202204/f89e788c65a241e88e7f0d0348de586f.shtml`

---

### Task 4: Event-Ordered Backtest Execution and Net Transaction Costs

**Files:**
- Modify: `quanttool/domain/models/__init__.py:54-81`
- Modify: `quanttool/backtest/engine.py:1-540`
- Test: `tests/test_backtest_engine_integrity.py`

**Interfaces:**
- Adds backward-compatible optional fields to `Trade`: `gross_amount`, `commission`, `stamp_tax`, `transfer_fee`, and `slippage_cost`, all defaulting to `0.0`.
- Adds backward-compatible optional fields to `Order`: `rejection_code` and `rejection_reason`, both defaulting to `None`.
- Adds internal immutable `PendingSignal(symbol: str, signal: Dict[str, Any], signal_time: datetime, execution_time: datetime)`.
- Adds `BacktestEngine.rejected_orders` through `orders` entries whose status is `rejected`.
- Consumes Task 3's rule, lot, limit-price, and transaction-cost functions.

- [ ] **Step 1: Write deterministic strategy and bar fixtures**

Start `tests/test_backtest_engine_integrity.py` with:

```python
from datetime import datetime
import unittest

import pandas as pd

from quanttool.backtest.engine import BacktestEngine
from quanttool.domain.interfaces.strategy import IStrategy


class SequenceStrategy(IStrategy):
    def __init__(self, signals):
        self.signals = signals

    def initialize(self, parameters):
        return None

    def calculate_signals(self, bars):
        return bars.copy()

    def get_signal(self, current_bar, historical_bars):
        return self.signals.get(len(historical_bars), {"direction": "hold"})

    def get_name(self):
        return "sequence"

    def get_parameters(self):
        return {}

    def get_description(self):
        return "Deterministic test strategy"


def make_bars(rows=5):
    timestamps = pd.bdate_range("2026-07-27", periods=rows)
    closes = [10.0, 10.2, 10.4, 10.3, 10.5][:rows]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": closes,
        "high": [value + 0.2 for value in closes],
        "low": [value - 0.2 for value in closes],
        "close": closes,
        "volume": [1_000_000.0] * rows,
        "amount": [value * 1_000_000 for value in closes],
    })
```

- [ ] **Step 2: Write next-bar, lot, and cost tests**

Add:

```python
class BacktestExecutionIntegrityTests(unittest.TestCase):
    def test_signal_fills_at_next_bar_open(self):
        bars = make_bars(5)
        strategy = SequenceStrategy({1: {"direction": "buy"}, 3: {"direction": "sell"}})
        engine = BacktestEngine(initial_cash=100_000)
        result = engine.run_backtest(
            strategy,
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        self.assertEqual(result.trades[0].timestamp, bars.timestamp.iloc[1].to_pydatetime())
        self.assertEqual(result.trades[0].price, bars.open.iloc[1])
        self.assertEqual(result.trades[1].timestamp, bars.timestamp.iloc[3].to_pydatetime())

    def test_main_board_buy_is_integer_hundred_share_lot(self):
        bars = make_bars(3)
        result = BacktestEngine(initial_cash=100_000).run_backtest(
            SequenceStrategy({1: {"direction": "buy"}}),
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        quantity = result.trades[0].quantity
        self.assertIsInstance(quantity, int)
        self.assertEqual(quantity % 100, 0)

    def test_trade_fee_contains_all_transaction_costs(self):
        bars = make_bars(5)
        result = BacktestEngine(initial_cash=100_000).run_backtest(
            SequenceStrategy({1: {"direction": "buy"}, 3: {"direction": "sell"}}),
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        buy, sell = result.trades
        self.assertEqual(buy.fee, buy.commission + buy.transfer_fee + buy.stamp_tax)
        self.assertEqual(sell.fee, sell.commission + sell.transfer_fee + sell.stamp_tax)
        self.assertEqual(buy.stamp_tax, 0.0)
        self.assertGreater(sell.stamp_tax, 0.0)
```

- [ ] **Step 3: Write anti-look-ahead, price-limit, and T+1 tests**

Add:

```python
    def test_appending_future_bars_does_not_change_prior_fills(self):
        short = make_bars(4)
        long = make_bars(5)
        strategy = SequenceStrategy({1: {"direction": "buy"}, 3: {"direction": "sell"}})
        short_result = BacktestEngine(initial_cash=100_000).run_backtest(
            strategy, {"600000.SH": short}, short.timestamp.iloc[0].to_pydatetime(), short.timestamp.iloc[-1].to_pydatetime()
        )
        long_result = BacktestEngine(initial_cash=100_000).run_backtest(
            strategy, {"600000.SH": long}, long.timestamp.iloc[0].to_pydatetime(), long.timestamp.iloc[-1].to_pydatetime()
        )
        short_fills = [(t.side, t.quantity, t.price, t.timestamp) for t in short_result.trades]
        comparable = [(t.side, t.quantity, t.price, t.timestamp) for t in long_result.trades if t.timestamp <= short.timestamp.iloc[-1]]
        self.assertEqual(short_fills, comparable)

    def test_limit_up_rejection_uses_preceding_close(self):
        bars = make_bars(3)
        bars.loc[1, ["open", "high", "low", "close"]] = [11.0, 11.0, 11.0, 11.0]
        engine = BacktestEngine(initial_cash=100_000)
        result = engine.run_backtest(
            SequenceStrategy({1: {"direction": "buy"}}),
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        self.assertEqual(result.trades, [])
        rejected = [order for order in result.orders if order.status == "rejected"]
        self.assertEqual(rejected[0].rejection_code, "limit_up")

    def test_t_plus_one_uses_next_supplied_bar(self):
        bars = make_bars(5)
        bars.loc[2, "timestamp"] = pd.Timestamp("2026-08-03")
        bars.loc[3, "timestamp"] = pd.Timestamp("2026-08-04")
        bars.loc[4, "timestamp"] = pd.Timestamp("2026-08-05")
        result = BacktestEngine(initial_cash=100_000).run_backtest(
            SequenceStrategy({1: {"direction": "buy"}, 2: {"direction": "sell"}}),
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        sell_trades = [trade for trade in result.trades if trade.side == "sell"]
        self.assertEqual(sell_trades[0].timestamp, pd.Timestamp("2026-08-03").to_pydatetime())
```

- [ ] **Step 4: Run the new engine tests and observe the expected failures**

```bash
.venv-mcp/bin/python -m unittest tests.test_backtest_engine_integrity -v
```

Expected: failures showing immediate future mutation, close-price execution, fractional shares, omitted tax/transfer fees, broken price-limit checks, and missing rejection fields.

- [ ] **Step 5: Add backward-compatible cost and rejection fields**

Extend `Trade` and `Order` with defaults:

```python
class Trade(BaseModel):
    # existing fields remain unchanged
    gross_amount: float = 0.0
    commission: float = 0.0
    stamp_tax: float = 0.0
    transfer_fee: float = 0.0
    slippage_cost: float = 0.0


class Order(BaseModel):
    # existing fields remain unchanged
    rejection_code: Optional[str] = None
    rejection_reason: Optional[str] = None
```

- [ ] **Step 6: Replace future immediate execution with a pending queue**

Add `PendingSignal` and a `pending_by_timestamp` mapping. At each timestamp:

1. Execute pending signals for that timestamp at the current bar's `open`, using the cached previous valid close.
2. Update latest prices and evaluate intrabar stop logic.
3. Generate the strategy signal from history ending at the current bar.
4. Queue that signal for the next greater timestamp in that symbol's sorted frame.
5. Update the cached close only after execution and signal generation for the current bar.

Delete the initialization from `df.iloc[-2]` and delete `_process_strategy_signal` logic that looks forward and immediately mutates the portfolio.

- [ ] **Step 7: Enforce lots, real bar dates, and structured rejection**

Resolve the rule from the execution timestamp, call `round_buy_quantity`, and calculate the next sellable timestamp from the next supplied bar after the execution date. A rejected constraint appends an `Order(status="rejected", rejection_code=..., rejection_reason=...)` and leaves cash and positions unchanged.

For sells, permit a full odd-lot exit but round a partial main-board sell down to the board increment. Unknown rules stop the backtest with `BacktestError`; ordinary order constraints reject only that order.

- [ ] **Step 8: Apply one dated transaction-cost path**

Use `calculate_transaction_cost` for buys and sells. Store total fees in the existing `Trade.fee` field and the components in the new fields. Set the position average cost from buy net amount divided by quantity so later PnL includes buy fees. Sell PnL is sell net proceeds minus the fee-inclusive position cost basis.

- [ ] **Step 9: Run focused engine and rule tests**

```bash
.venv-mcp/bin/python -m unittest tests.test_backtest_engine_integrity tests.test_a_share_rules -v
```

Expected: all execution-order, lot, T+1, rejection, and net-cost tests pass.

- [ ] **Step 10: Commit event-ordered execution**

```bash
git add quanttool/domain/models/__init__.py quanttool/backtest/engine.py tests/test_backtest_engine_integrity.py
git commit -m "fix: make backtests event ordered and cost aware"
```

---

### Task 5: Metrics, Documentation, and Full Acceptance Verification

**Files:**
- Modify: `quanttool/backtest/engine.py:670-860`
- Modify: `tests/test_backtest_engine_integrity.py`
- Modify: `README.md`

**Interfaces:**
- Preserves: `BacktestResult.win_rate`, `winning_trades`, `losing_trades`, and `total_trades`.
- Changes semantics: win/loss counts and win rate use closed sell trades with non-`None` PnL; total trade count remains the number of fills.

- [ ] **Step 1: Write the closed-trade metric regression test**

Add to `tests/test_backtest_engine_integrity.py`:

```python
    def test_win_rate_uses_closed_trades_only(self):
        bars = make_bars(5)
        result = BacktestEngine(initial_cash=100_000).run_backtest(
            SequenceStrategy({1: {"direction": "buy"}, 3: {"direction": "sell"}}),
            {"600000.SH": bars},
            bars.timestamp.iloc[0].to_pydatetime(),
            bars.timestamp.iloc[-1].to_pydatetime(),
        )
        self.assertEqual(result.total_trades, 2)
        self.assertEqual(result.winning_trades + result.losing_trades, 1)
        self.assertIn(result.win_rate, {0.0, 1.0})
```

- [ ] **Step 2: Run the metric test and observe the wrong denominator**

```bash
.venv-mcp/bin/python -m unittest tests.test_backtest_engine_integrity.BacktestExecutionIntegrityTests.test_win_rate_uses_closed_trades_only -v
```

Expected: failure because the current result counts buy fills in the win-rate denominator.

- [ ] **Step 3: Correct metric semantics**

Use:

```python
closed_trades = [
    trade for trade in self.trades
    if trade.side == OrderSide.SELL and trade.pnl is not None
]
winning_trades = [trade for trade in closed_trades if trade.pnl > 0]
losing_trades = [trade for trade in closed_trades if trade.pnl < 0]
win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0.0
```

Apply the same closed-trade definition in both `_generate_backtest_result` and `calculate_metrics`.

- [ ] **Step 4: Document runtime and operator blockers**

Add a concise README production-hardening section documenting:

```text
QUANTTOOL_ENV=production
TUSHARE_TOKEN=<optional runtime secret>
EASTMONEY_COOKIE=<optional runtime secret>
```

State that `csv_mock` is test-only, Sina/Tencent needs no token, production fails when real data is unavailable, and previously committed credentials must be revoked and removed from Git history before launch.

- [ ] **Step 5: Run the complete backend suite**

```bash
.venv-mcp/bin/python -m unittest discover -s tests -v
```

Expected: the original 92 tests plus all new tests pass with zero failures and zero errors.

- [ ] **Step 6: Compile all production and test modules**

```bash
.venv-mcp/bin/python -m compileall -q quanttool tests
```

Expected: exit code 0 and no syntax errors.

- [ ] **Step 7: Run current-tree credential and whitespace checks**

Run the credential scan without printing any matching secret value:

```bash
.venv-mcp/bin/python -c "from pathlib import Path; import re; paths=list(Path('quanttool').rglob('*.py')); bad=[str(p) for p in paths if re.search(r'(?:tushare_token\s*=\s*[\"\'][A-Za-z0-9]{20,}|qgqp_b_id=)', p.read_text(encoding='utf-8', errors='ignore'))]; print({'credential_files': bad}); raise SystemExit(bool(bad))"
```

Expected: `{'credential_files': []}` and exit code 0.

Then run:

```bash
git diff --check
```

Expected: exit code 0 with no whitespace errors.

- [ ] **Step 8: Run one read-only live Ashare smoke test**

```bash
QUANTTOOL_ENV=development .venv-mcp/bin/python -c "from datetime import datetime, timedelta; from quanttool.infrastructure.data_providers.historical.ashare_provider import AShareProvider; end=datetime.now(); start=end-timedelta(days=14); frame=AShareProvider(max_missing_ratio=0.0).get_bars(['600519.SH'], start, end, '1d')['600519.SH']; print({'rows': len(frame), 'start': str(frame.timestamp.min()), 'end': str(frame.timestamp.max()), 'source': frame.attrs['quanttool_provenance']['provider'], 'ohlc_valid': bool(((frame.high >= frame[['open','close']].max(axis=1)) & (frame.low <= frame[['open','close']].min(axis=1))).all())})"
```

Expected when the public endpoint is reachable: at least one row, concrete source `sina` or `tencent`, and `ohlc_valid=True`. If the endpoint is unavailable, record the network failure separately; do not weaken or skip offline tests.

- [ ] **Step 9: Review final scope and repository state**

```bash
git status --short --branch
git diff c7e53d8b3 --stat
```

Expected: only the planned runtime, provider, rule, engine, model, test, and README paths differ from the design baseline. Do not stage unrelated files.

- [ ] **Step 10: Commit metrics and operator documentation**

```bash
git add quanttool/backtest/engine.py tests/test_backtest_engine_integrity.py README.md
git commit -m "fix: verify P0 data and backtest integrity"
```

---

## Final Acceptance Checklist

- [ ] Runtime-mode tests prove simulated providers cannot run in production.
- [ ] Import tests prove provider imports preserve proxy variables.
- [ ] Current-tree scanning proves hard-coded market-data credentials are absent.
- [ ] Real-provider tests prove no random rows are generated and concrete fallback provenance is retained.
- [ ] Validation tests cover duplicate timestamps, ordering, OHLC, numeric fields, interval bounds, and negative volume/amount.
- [ ] Rule tests cover normalized symbols, main/ChiNext/STAR/Beijing boards, dated ST/ChiNext changes, IPO sessions, lots, and dates before 2017.
- [ ] Fee tests cover both official transition dates and sell-only stamp tax.
- [ ] Engine tests prove next-bar execution, anti-look-ahead invariance, previous-close price limits, real-bar T+1, integer lots, structured rejection, and full net costs.
- [ ] Metrics tests prove win rate uses closed trades only.
- [ ] Full `unittest` discovery and `compileall` finish with exit code 0.
- [ ] Live read-only smoke evidence is recorded without becoming an offline-test dependency.
- [ ] Final handoff states that leaked credential revocation and destructive Git-history cleanup remain operator launch blockers.
