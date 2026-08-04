"""Dated A-share symbol, price-limit and order-quantity rules."""

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
import math
from typing import Optional, Tuple

from ..core.errors import BacktestError


# Rule references:
# Shanghai 2026 rules and risk-warning change:
# https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml
# Beijing Stock Exchange trading rules:
# https://www.bse.cn/jygl_list/200028217.html

_EARLIEST_SUPPORTED_DATE = date(2017, 1, 1)
_STAR_LAUNCH_DATE = date(2019, 7, 22)
_CHINEXT_REFORM_DATE = date(2020, 8, 24)
_BSE_LAUNCH_DATE = date(2021, 11, 15)
_MAIN_ST_LIMIT_CHANGE_DATE = date(2026, 7, 6)


@dataclass(frozen=True)
class NormalizedSymbol:
    """Canonical exchange identity and board classification."""

    code: str
    exchange: str
    board: str

    @property
    def canonical(self) -> str:
        """Return the conventional six-digit code and exchange suffix."""
        return f"{self.code}.{self.exchange}"


@dataclass(frozen=True)
class TradingRule:
    """Trading rules resolved for a security on one trade date."""

    symbol: NormalizedSymbol
    board: str
    price_limit: Optional[float]
    min_buy_quantity: int
    buy_increment: int
    t_plus_one: bool = True


def _classify(code: str, exchange: Optional[str]) -> Tuple[str, str]:
    if code.startswith("688"):
        inferred_exchange, board = "SH", "star"
    elif code.startswith(("600", "601", "603", "605")):
        inferred_exchange, board = "SH", "main"
    elif code.startswith(("300", "301")):
        inferred_exchange, board = "SZ", "chinext"
    elif code.startswith(("000", "001", "002", "003")):
        inferred_exchange, board = "SZ", "main"
    elif code.startswith(("4", "8", "92")):
        inferred_exchange, board = "BJ", "bse"
    else:
        raise BacktestError(f"Unsupported A-share symbol: {code}")

    if exchange is not None and exchange != inferred_exchange:
        raise BacktestError(
            "Symbol exchange conflicts with code: "
            f"code={code}, exchange={exchange}"
        )
    return inferred_exchange, board


def normalize_symbol(symbol: str) -> NormalizedSymbol:
    """Normalize supported A-share symbol forms and reject ambiguity."""
    if not isinstance(symbol, str) or not symbol.strip():
        raise BacktestError("A-share symbol must be a non-empty string")

    raw = symbol.strip().upper()
    exchange: Optional[str] = None
    suffixes = {
        ".XSHG": "SH",
        ".XSHE": "SZ",
        ".SH": "SH",
        ".SZ": "SZ",
        ".BJ": "BJ",
    }
    for suffix, suffix_exchange in suffixes.items():
        if raw.endswith(suffix):
            exchange = suffix_exchange
            raw = raw[: -len(suffix)]
            break

    for prefix in ("SH", "SZ", "BJ"):
        if raw.startswith(prefix):
            if exchange is not None and exchange != prefix:
                raise BacktestError(f"Conflicting symbol exchanges: {symbol}")
            exchange = prefix
            raw = raw[len(prefix):]
            break

    if len(raw) != 6 or not raw.isdigit():
        raise BacktestError(f"Unsupported A-share symbol: {symbol}")

    exchange, board = _classify(raw, exchange)
    return NormalizedSymbol(code=raw, exchange=exchange, board=board)


def _as_date(value: date) -> date:
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, date):
        raise BacktestError("trade_date must be a date")
    return value


def resolve_trading_rule(
    symbol: str,
    trade_date: date,
    stock_name: Optional[str] = None,
    listing_session: Optional[int] = None,
) -> TradingRule:
    """Resolve versioned board, limit and declaration-size rules."""
    resolved_date = _as_date(trade_date)
    if resolved_date < _EARLIEST_SUPPORTED_DATE:
        raise BacktestError(
            "A-share trading rules are unsupported before 2017-01-01"
        )
    normalized = normalize_symbol(symbol)

    if normalized.board == "star":
        if resolved_date < _STAR_LAUNCH_DATE:
            raise BacktestError("STAR Market rule requested before board launch")
        price_limit = 0.20
        minimum, increment = 200, 1
    elif normalized.board == "bse":
        if resolved_date < _BSE_LAUNCH_DATE:
            raise BacktestError("BSE rule requested before exchange launch")
        price_limit = 0.30
        minimum, increment = 100, 1
    elif normalized.board == "chinext":
        price_limit = 0.20 if resolved_date >= _CHINEXT_REFORM_DATE else 0.10
        minimum, increment = 100, 100
    elif normalized.board == "main":
        is_risk_warning = bool(stock_name and "ST" in stock_name.upper())
        if is_risk_warning and resolved_date < _MAIN_ST_LIMIT_CHANGE_DATE:
            price_limit = 0.05
        else:
            price_limit = 0.10
        minimum, increment = 100, 100
    else:  # Defensive even though normalization is fail-closed.
        raise BacktestError(f"Unsupported A-share board: {normalized.board}")

    if listing_session is not None:
        if listing_session <= 0:
            raise BacktestError("listing_session must be positive")
        if listing_session <= 5:
            price_limit = None

    return TradingRule(
        symbol=normalized,
        board=normalized.board,
        price_limit=price_limit,
        min_buy_quantity=minimum,
        buy_increment=increment,
    )


def round_buy_quantity(desired: float, rule: TradingRule) -> int:
    """Round a desired quantity down to the board's valid buy declaration."""
    if not math.isfinite(desired) or desired <= 0:
        return 0
    whole_quantity = math.floor(desired)
    if whole_quantity < rule.min_buy_quantity:
        return 0
    return (whole_quantity // rule.buy_increment) * rule.buy_increment


def calculate_limit_prices(
    prev_close: float,
    rule: TradingRule,
) -> Tuple[Optional[float], Optional[float]]:
    """Calculate exchange-style cent prices using half-up rounding."""
    if not math.isfinite(prev_close) or prev_close <= 0:
        raise BacktestError("Previous close must be positive")
    if rule.price_limit is None:
        return None, None

    close = Decimal(str(prev_close))
    rate = Decimal(str(rule.price_limit))
    cent = Decimal("0.01")
    limit_up = (close * (Decimal("1") + rate)).quantize(
        cent,
        rounding=ROUND_HALF_UP,
    )
    limit_down = (close * (Decimal("1") - rate)).quantize(
        cent,
        rounding=ROUND_HALF_UP,
    )
    return float(limit_up), float(limit_down)
