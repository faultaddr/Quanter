"""Versioned A-share transaction-fee schedule."""

from dataclasses import dataclass
from datetime import date, datetime
import math

from ..core.errors import BacktestError


# Rule references:
# 2023 stamp-tax reduction:
# https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html
# 2022 transfer-fee reduction:
# https://www.chinaclear.cn/zdjs/gszb/202204/f89e788c65a241e88e7f0d0348de586f.shtml

_EARLIEST_SUPPORTED_DATE = date(2017, 1, 1)
_TRANSFER_FEE_CHANGE_DATE = date(2022, 4, 29)
_STAMP_TAX_CHANGE_DATE = date(2023, 8, 28)


@dataclass(frozen=True)
class FeeRates:
    """Rates that apply to one execution date."""

    commission_rate: float
    min_commission: float
    stamp_tax_rate: float
    transfer_fee_rate: float


@dataclass(frozen=True)
class TransactionCostBreakdown:
    """Gross and net cash impact of one A-share fill."""

    gross_amount: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    total_fee: float
    net_amount: float


def _as_date(value: date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, date):
        raise BacktestError("trade_date must be a date")
    return value


def resolve_fee_rates(
    trade_date: date,
    commission_rate: float = 0.0003,
    min_commission: float = 5.0,
) -> FeeRates:
    """Resolve transaction rates from official effective dates."""
    resolved_date = _as_date(trade_date)
    if resolved_date < _EARLIEST_SUPPORTED_DATE:
        raise BacktestError(
            "A-share fee schedules are unsupported before 2017-01-01"
        )
    if commission_rate < 0 or min_commission < 0:
        raise BacktestError("Commission configuration must be non-negative")

    stamp_tax_rate = (
        0.001
        if resolved_date < _STAMP_TAX_CHANGE_DATE
        else 0.0005
    )
    transfer_fee_rate = (
        0.00002
        if resolved_date < _TRANSFER_FEE_CHANGE_DATE
        else 0.00001
    )
    return FeeRates(
        commission_rate=commission_rate,
        min_commission=min_commission,
        stamp_tax_rate=stamp_tax_rate,
        transfer_fee_rate=transfer_fee_rate,
    )


def calculate_transaction_cost(
    price: float,
    quantity: int,
    side: str,
    trade_date: date,
    commission_rate: float = 0.0003,
    min_commission: float = 5.0,
) -> TransactionCostBreakdown:
    """Calculate all fees and net cash impact for one accepted fill."""
    if not math.isfinite(price) or price <= 0:
        raise BacktestError("Trade price must be positive")
    if quantity <= 0:
        raise BacktestError("Trade quantity must be positive")
    normalized_side = side.lower()
    if normalized_side not in {"buy", "sell"}:
        raise BacktestError(f"Unsupported trade side: {side}")

    rates = resolve_fee_rates(
        trade_date,
        commission_rate=commission_rate,
        min_commission=min_commission,
    )
    gross_amount = float(price * quantity)
    commission = max(
        rates.min_commission,
        gross_amount * rates.commission_rate,
    )
    stamp_tax = (
        gross_amount * rates.stamp_tax_rate
        if normalized_side == "sell"
        else 0.0
    )
    transfer_fee = gross_amount * rates.transfer_fee_rate
    total_fee = commission + stamp_tax + transfer_fee
    net_amount = (
        gross_amount + total_fee
        if normalized_side == "buy"
        else gross_amount - total_fee
    )
    return TransactionCostBreakdown(
        gross_amount=gross_amount,
        commission=commission,
        stamp_tax=stamp_tax,
        transfer_fee=transfer_fee,
        total_fee=total_fee,
        net_amount=net_amount,
    )
