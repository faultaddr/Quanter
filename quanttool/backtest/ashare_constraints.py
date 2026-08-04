"""
A股市场约束处理模块

实现A股特有的交易约束：
- 涨跌停检查
- 停牌检查
- ST股票检查
- 动态滑点计算
- 真实交易成本计算
"""

from dataclasses import dataclass
from datetime import date as Date, datetime
from enum import Enum
from typing import Dict, Optional, Tuple

from ..core.errors import BacktestError
from .a_share_rules import (
    calculate_limit_prices,
    normalize_symbol,
    resolve_trading_rule,
)
from .fee_schedule import calculate_transaction_cost


class LimitStatus(str, Enum):
    """涨跌停状态"""
    NORMAL = "normal"           # 正常交易
    LIMIT_UP = "limit_up"       # 涨停
    LIMIT_DOWN = "limit_down"   # 跌停
    SUSPENDED = "suspended"      # 停牌


class StockStatus(str, Enum):
    """股票状态"""
    NORMAL = "normal"           # 正常
    ST = "st"                   # ST股
    STAR_ST = "st"              # *ST股
    DOUBLE_ST = "st"            # **ST股
    PT = "pt"                   # PT股（退市整理）


@dataclass
class TradeConstraint:
    """交易约束结果"""
    can_trade: bool
    limit_status: LimitStatus
    stock_status: StockStatus
    reason: str
    slippage_rate: float
    commission_rate: float
    min_commission: float


@dataclass
class TransactionCost:
    """交易成本详情"""
    gross_amount: float        # 总金额
    commission: float          # 手续费
    stamp_tax: float           # 印花税（仅卖出）
    transfer_fee: float        # 过户费
    net_amount: float         # 净金额


class ASShareConstraints:
    """
    A股交易约束处理器

    处理A股特有的交易规则和约束：
    - 涨跌停板限制（主板10%，创业板/科创板20%）
    - 停牌股票不能交易
    - ST股票交易限制
    - 动态滑点（根据流动性、波动率调整）
    - 真实交易成本（佣金、印花税、过户费）
    """

    # A股涨跌幅限制
    LIMIT_UP_RATES = {
        "main": 0.10,      # 主板10%
        "chinext": 0.20,   # 创业板20%
        "star": 0.20,      # 科创板20%
        "default": 0.10,  # 默认10%
    }

    LIMIT_DOWN_RATES = {
        "main": 0.10,
        "chinext": 0.20,
        "star": 0.20,
        "default": 0.10,
    }

    # 默认交易成本
    DEFAULT_COMMISSION_RATE = 0.0003  # 万三
    DEFAULT_STAMP_TAX_RATE = 0.001     # 千一（仅卖出）
    DEFAULT_TRANSFER_FEE_RATE = 0.00002  # 过户费万分之0.2
    DEFAULT_MIN_COMMISSION = 5.0       # 最低佣金5元

    # ST股票前缀
    ST_PREFIXES = ["ST", "*ST", "**ST", "PT"]

    def __init__(
        self,
        commission_rate: float = DEFAULT_COMMISSION_RATE,
        stamp_tax_rate: float = DEFAULT_STAMP_TAX_RATE,
        transfer_fee_rate: float = DEFAULT_TRANSFER_FEE_RATE,
        min_commission: float = DEFAULT_MIN_COMMISSION,
        enable_st_restriction: bool = True,
        enable_limit_check: bool = True,
    ):
        """
        初始化A股约束处理器

        Args:
            commission_rate: 佣金费率（默认万三）
            stamp_tax_rate: 印花税率（默认千分之一，卖出时收取）
            transfer_fee_rate: 过户费率（默认万分之0.2）
            min_commission: 最低佣金（默认5元）
            enable_st_restriction: 是否启用ST股票限制
            enable_limit_check: 是否启用涨跌停检查
        """
        self.commission_rate = commission_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.transfer_fee_rate = transfer_fee_rate
        self.min_commission = min_commission
        self.enable_st_restriction = enable_st_restriction
        self.enable_limit_check = enable_limit_check

        # 股票信息缓存
        self._stock_info_cache: Dict[str, Dict] = {}
        # 涨跌停价格缓存
        self._limit_price_cache: Dict[str, Dict[Tuple[str, datetime], Tuple[float, float]]] = {}

    def get_market_type(self, symbol: str) -> str:
        """
        判断股票所属市场

        Args:
            symbol: 股票代码

        Returns:
            市场类型: main, chinext, star, default
        """
        return normalize_symbol(symbol).board

    def get_limit_rates(
        self,
        symbol: str,
        trade_date: Optional[Date] = None,
        stock_name: Optional[str] = None,
        listing_session: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        获取涨跌停限制比例

        Args:
            symbol: 股票代码

        Returns:
            (涨跌幅上限, 跌涨幅下限)
        """
        resolved_date = self._require_trade_date(trade_date)
        rule = resolve_trading_rule(
            symbol,
            resolved_date,
            stock_name=stock_name,
            listing_session=listing_session,
        )
        if rule.price_limit is None:
            return 0.0, 0.0
        return rule.price_limit, rule.price_limit

    def calculate_limit_price(
        self,
        symbol: str,
        prev_close: float,
        date: Optional[datetime] = None,
        stock_name: Optional[str] = None,
        listing_session: Optional[int] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        计算涨跌停价格

        Args:
            symbol: 股票代码
            prev_close: 前收盘价
            date: 日期（可选）

        Returns:
            (涨停价, 跌停价)
        """
        resolved_date = self._require_trade_date(date)
        rule = resolve_trading_rule(
            symbol,
            resolved_date,
            stock_name=stock_name,
            listing_session=listing_session,
        )
        return calculate_limit_prices(prev_close, rule)

    def check_limit_status(
        self,
        symbol: str,
        current_price: float,
        prev_close: float,
        date: Optional[datetime] = None,
        stock_name: Optional[str] = None,
        listing_session: Optional[int] = None,
    ) -> LimitStatus:
        """
        检查涨跌停状态

        Args:
            symbol: 股票代码
            current_price: 当前价格
            prev_close: 前收盘价
            date: 日期（可选）

        Returns:
            LimitStatus: 涨跌停状态
        """
        if not self.enable_limit_check:
            return LimitStatus.NORMAL

        limit_up, limit_down = self.calculate_limit_price(
            symbol,
            prev_close,
            date,
            stock_name=stock_name,
            listing_session=listing_session,
        )
        if limit_up is None or limit_down is None:
            return LimitStatus.NORMAL

        # 使用精确比较，考虑价格精度
        if abs(current_price - limit_up) < 0.001:
            return LimitStatus.LIMIT_UP
        elif abs(current_price - limit_down) < 0.001:
            return LimitStatus.LIMIT_DOWN
        else:
            return LimitStatus.NORMAL

    def check_stock_status(self, stock_name: Optional[str]) -> StockStatus:
        """
        检查股票状态（是否ST）

        Args:
            stock_name: 股票名称（可选）

        Returns:
            StockStatus: 股票状态
        """
        if not stock_name:
            return StockStatus.NORMAL

        for prefix in self.ST_PREFIXES:
            if prefix in stock_name:
                return StockStatus.ST

        return StockStatus.NORMAL

    def can_buy(
        self,
        symbol: str,
        current_price: float,
        prev_close: float,
        is_suspended: bool = False,
        stock_name: Optional[str] = None,
        trade_date: Optional[datetime] = None,
    ) -> TradeConstraint:
        """
        检查是否可以买入

        Args:
            symbol: 股票代码
            current_price: 当前价格
            prev_close: 前收盘价
            is_suspended: 是否停牌
            stock_name: 股票名称（用于ST检查）

        Returns:
            TradeConstraint: 交易约束结果
        """
        # 检查停牌
        if is_suspended:
            return TradeConstraint(
                can_trade=False,
                limit_status=LimitStatus.SUSPENDED,
                stock_status=StockStatus.NORMAL,
                reason=f"股票 {symbol} 已停牌",
                slippage_rate=0.0,
                commission_rate=self.commission_rate,
                min_commission=self.min_commission,
            )

        # 检查ST
        stock_status = self.check_stock_status(stock_name)
        if self.enable_st_restriction and stock_status == StockStatus.ST:
            return TradeConstraint(
                can_trade=False,
                limit_status=LimitStatus.NORMAL,
                stock_status=StockStatus.ST,
                reason=f"股票 {symbol} 为ST股票，禁止买入",
                slippage_rate=0.0,
                commission_rate=self.commission_rate,
                min_commission=self.min_commission,
            )

        # 检查涨停
        if self.enable_limit_check:
            limit_status = self.check_limit_status(
                symbol,
                current_price,
                prev_close,
                date=trade_date,
                stock_name=stock_name,
            )
            if limit_status == LimitStatus.LIMIT_UP:
                return TradeConstraint(
                    can_trade=False,
                    limit_status=LimitStatus.LIMIT_UP,
                    stock_status=stock_status,
                    reason=f"股票 {symbol} 已涨停，无法买入",
                    slippage_rate=0.0,
                    commission_rate=self.commission_rate,
                    min_commission=self.min_commission,
                )

        # 可以买入，计算动态滑点
        slippage = self._calculate_slippage(symbol, current_price, prev_close)

        return TradeConstraint(
            can_trade=True,
            limit_status=LimitStatus.NORMAL,
            stock_status=stock_status,
            reason="",
            slippage_rate=slippage,
            commission_rate=self.commission_rate,
            min_commission=self.min_commission,
        )

    def can_sell(
        self,
        symbol: str,
        current_price: float,
        prev_close: float,
        is_suspended: bool = False,
        stock_name: Optional[str] = None,
        trade_date: Optional[datetime] = None,
    ) -> TradeConstraint:
        """
        检查是否可以卖出

        Args:
            symbol: 股票代码
            current_price: 当前价格
            prev_close: 前收盘价
            is_suspended: 是否停牌
            stock_name: 股票名称（用于ST检查）

        Returns:
            TradeConstraint: 交易约束结果
        """
        # 检查停牌
        if is_suspended:
            return TradeConstraint(
                can_trade=False,
                limit_status=LimitStatus.SUSPENDED,
                stock_status=StockStatus.NORMAL,
                reason=f"股票 {symbol} 已停牌",
                slippage_rate=0.0,
                commission_rate=self.commission_rate,
                min_commission=self.min_commission,
            )

        # ST股票可以卖出（不限制卖出）
        stock_status = self.check_stock_status(stock_name)

        # 检查跌停
        if self.enable_limit_check:
            limit_status = self.check_limit_status(
                symbol,
                current_price,
                prev_close,
                date=trade_date,
                stock_name=stock_name,
            )
            if limit_status == LimitStatus.LIMIT_DOWN:
                return TradeConstraint(
                    can_trade=False,
                    limit_status=LimitStatus.LIMIT_DOWN,
                    stock_status=stock_status,
                    reason=f"股票 {symbol} 已跌停，无法卖出",
                    slippage_rate=0.0,
                    commission_rate=self.commission_rate,
                    min_commission=self.min_commission,
                )

        # 可以卖出，计算动态滑点
        slippage = self._calculate_slippage(symbol, current_price, prev_close)

        return TradeConstraint(
            can_trade=True,
            limit_status=LimitStatus.NORMAL,
            stock_status=stock_status,
            reason="",
            slippage_rate=slippage,
            commission_rate=self.commission_rate,
            min_commission=self.min_commission,
        )

    def _calculate_slippage(
        self,
        symbol: str,
        current_price: float,
        prev_close: float
    ) -> float:
        """
        计算动态滑点

        基于以下因素调整滑点：
        - 价格水平（低价股滑点更大）
        - 波动率（日内波动越大滑点越大）
        - 流动性（价格水平反映流动性）

        Args:
            symbol: 股票代码
            current_price: 当前价格
            prev_close: 前收盘价

        Returns:
            滑点费率
        """
        # 基础滑点
        base_slippage = 0.0001  # 万分之一

        # 价格水平因子：低价股（<5元）滑点更大
        if current_price < 5:
            price_factor = 2.0
        elif current_price < 10:
            price_factor = 1.5
        elif current_price < 20:
            price_factor = 1.0
        else:
            price_factor = 0.8

        # 波动因子：基于当日涨跌幅
        daily_change = abs(current_price - prev_close) / prev_close
        if daily_change > 0.05:  # 大涨大跌
            volatility_factor = 1.5
        elif daily_change > 0.03:
            volatility_factor = 1.2
        elif daily_change > 0.01:
            volatility_factor = 1.0
        else:
            volatility_factor = 0.8

        # 创业板/科创板滑点更大
        market_type = self.get_market_type(symbol)
        if market_type in ["chinext", "star"]:
            market_factor = 1.3
        else:
            market_factor = 1.0

        dynamic_slippage = base_slippage * price_factor * volatility_factor * market_factor

        return min(dynamic_slippage, 0.001)  # 最大千分之一

    def apply_transaction_costs(
        self,
        price: float,
        quantity: float,
        side: str,
        trade_date: Optional[Date] = None,
    ) -> TransactionCost:
        """
        计算真实交易成本

        A股交易成本组成：
        1. 佣金（买卖都要收取）
        2. 印花税（仅卖出收取，千分之一）
        3. 过户费（买卖都要收取，万分之0.2）

        Args:
            price: 成交价格
            quantity: 成交数量
            side: 买卖方向 ("buy" 或 "sell")

        Returns:
            TransactionCost: 交易成本详情
        """
        resolved_date = self._require_trade_date(trade_date)
        breakdown = calculate_transaction_cost(
            price=price,
            quantity=int(quantity),
            side=side,
            trade_date=resolved_date,
            commission_rate=self.commission_rate,
            min_commission=self.min_commission,
        )

        return TransactionCost(
            gross_amount=breakdown.gross_amount,
            commission=breakdown.commission,
            stamp_tax=breakdown.stamp_tax,
            transfer_fee=breakdown.transfer_fee,
            net_amount=breakdown.net_amount,
        )

    def get_effective_price(
        self,
        symbol: str,
        price: float,
        prev_close: float,
        side: str,
        order_type: str = "market",
        trade_date: Optional[datetime] = None,
    ) -> float:
        """
        获取有效成交价格（考虑滑点）

        Args:
            symbol: 股票代码
            price: 基准价格
            prev_close: 前收盘价
            side: 买卖方向
            order_type: 订单类型

        Returns:
            考虑滑点后的价格
        """
        if order_type != "market":
            return price

        constraint = (
            self.can_buy(
                symbol,
                price,
                prev_close,
                trade_date=trade_date,
            )
            if side.lower() == "buy"
            else self.can_sell(
                symbol,
                price,
                prev_close,
                trade_date=trade_date,
            )
        )

        if side.lower() == "buy":
            return price * (1 + constraint.slippage_rate)
        else:
            return price * (1 - constraint.slippage_rate)

    @staticmethod
    def _require_trade_date(trade_date: Optional[Date]) -> Date:
        """Require an explicit event date for versioned exchange rules."""
        if trade_date is None:
            raise BacktestError("An explicit trade_date is required")
        return trade_date.date() if isinstance(trade_date, datetime) else trade_date


def create_constraints(
    commission_rate: Optional[float] = None,
    enable_st: bool = True,
    enable_limit: bool = True
) -> ASShareConstraints:
    """
    便捷函数：创建A股约束处理器

    Args:
        commission_rate: 自定义佣金费率
        enable_st: 是否启用ST限制
        enable_limit: 是否启用涨跌停检查

    Returns:
        ASShareConstraints: 约束处理器实例
    """
    return ASShareConstraints(
        commission_rate=commission_rate or ASShareConstraints.DEFAULT_COMMISSION_RATE,
        enable_st_restriction=enable_st,
        enable_limit_check=enable_limit,
    )
