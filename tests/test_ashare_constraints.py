"""Tests for A-share constraints module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quanttool.backtest.ashare_constraints import (
    ASShareConstraints,
    create_constraints,
    LimitStatus,
    StockStatus,
    TradeConstraint,
    TransactionCost,
)


class TestASShareConstraints:
    """Test cases for ASShareConstraints."""

    @pytest.fixture
    def constraints(self):
        """Create constraints instance."""
        return ASShareConstraints()

    @pytest.fixture
    def constraints_with_limits(self):
        """Create constraints with limit checking enabled."""
        return ASShareConstraints(enable_limit_check=True, enable_st_restriction=True)

    # ========== 市场类型测试 ==========

    def test_get_market_type_main(self, constraints):
        """Test market type detection for main board."""
        assert constraints.get_market_type("600000") == "main"
        assert constraints.get_market_type("000001") == "main"

    def test_get_market_type_chinext(self, constraints):
        """Test market type detection for ChiNext."""
        assert constraints.get_market_type("300001") == "chinext"
        assert constraints.get_market_type("002001") == "chinext"

    def test_get_market_type_star(self, constraints):
        """Test market type detection for STAR market."""
        assert constraints.get_market_type("688001") == "star"

    def test_get_limit_rates_main(self, constraints):
        """Test limit rates for main board (10%)."""
        limit_up, limit_down = constraints.get_limit_rates("600000")
        assert limit_up == 0.10
        assert limit_down == 0.10

    def test_get_limit_rates_chinext(self, constraints):
        """Test limit rates for ChiNext (20%)."""
        limit_up, limit_down = constraints.get_limit_rates("300001")
        assert limit_up == 0.20
        assert limit_down == 0.20

    # ========== 涨跌停价格计算测试 ==========

    def test_calculate_limit_price(self, constraints):
        """Test limit price calculation."""
        limit_up, limit_down = constraints.calculate_limit_price("600000", 10.0)
        assert limit_up == 11.0
        assert limit_down == 9.0

    def test_calculate_limit_price_rounding(self, constraints):
        """Test limit price rounding to cents."""
        limit_up, limit_down = constraints.calculate_limit_price("600000", 10.55)
        assert limit_up == 11.61  # 10.55 * 1.1 = 11.605 -> 11.61
        assert limit_down == 9.50  # 10.55 * 0.9 = 9.495 -> 9.50

    # ========== 涨跌停状态检查测试 ==========

    def test_check_limit_status_normal(self, constraints):
        """Test normal price status."""
        status = constraints.check_limit_status("600000", 10.5, 10.0)
        assert status == LimitStatus.NORMAL

    def test_check_limit_status_limit_up(self, constraints):
        """Test limit up status."""
        status = constraints.check_limit_status("600000", 11.0, 10.0)
        assert status == LimitStatus.LIMIT_UP

    def test_check_limit_status_limit_down(self, constraints):
        """Test limit down status."""
        status = constraints.check_limit_status("600000", 9.0, 10.0)
        assert status == LimitStatus.LIMIT_DOWN

    # ========== 股票状态检查测试 ==========

    def test_check_stock_status_normal(self, constraints):
        """Test normal stock status."""
        status = constraints.check_stock_status("平安银行")
        assert status == StockStatus.NORMAL

    def test_check_stock_status_st(self, constraints):
        """Test ST stock status."""
        status = constraints.check_stock_status("ST某某")
        assert status == StockStatus.ST

    def test_check_stock_status_star_st(self, constraints):
        """Test *ST stock status."""
        status = constraints.check_stock_status("*ST某某")
        assert status == StockStatus.ST

    # ========== 买入检查测试 ==========

    def test_can_buy_normal(self, constraints_with_limits):
        """Test normal buy permission."""
        constraint = constraints_with_limits.can_buy("600000", 10.5, 10.0)
        assert constraint.can_trade is True
        assert constraint.limit_status == LimitStatus.NORMAL

    def test_can_buy_limit_up(self, constraints_with_limits):
        """Test cannot buy when limit up."""
        constraint = constraints_with_limits.can_buy("600000", 11.0, 10.0)
        assert constraint.can_trade is False
        assert constraint.limit_status == LimitStatus.LIMIT_UP
        assert "涨停" in constraint.reason

    def test_can_buy_st_stock(self, constraints_with_limits):
        """Test cannot buy ST stock."""
        constraint = constraints_with_limits.can_buy("600000", 10.5, 10.0, stock_name="ST某某")
        assert constraint.can_trade is False
        assert "ST" in constraint.reason

    def test_can_buy_suspended(self, constraints_with_limits):
        """Test cannot buy suspended stock."""
        constraint = constraints_with_limits.can_buy("600000", 10.5, 10.0, is_suspended=True)
        assert constraint.can_trade is False
        assert "停牌" in constraint.reason

    # ========== 卖出检查测试 ==========

    def test_can_sell_normal(self, constraints_with_limits):
        """Test normal sell permission."""
        constraint = constraints_with_limits.can_sell("600000", 10.5, 10.0)
        assert constraint.can_trade is True

    def test_can_sell_limit_down(self, constraints_with_limits):
        """Test cannot sell when limit down."""
        constraint = constraints_with_limits.can_sell("600000", 9.0, 10.0)
        assert constraint.can_trade is False
        assert constraint.limit_status == LimitStatus.LIMIT_DOWN

    # ========== 动态滑点测试 ==========

    def test_slippage_low_price(self, constraints):
        """Test higher slippage for low price stocks."""
        # 低价股滑点应该更大
        slippage_low = constraints._calculate_slippage("600000", 3.0, 3.0)
        slippage_high = constraints._calculate_slippage("600000", 50.0, 50.0)
        assert slippage_low > slippage_high

    def test_slippage_high_volatility(self, constraints):
        """Test higher slippage for high volatility."""
        slippage_calm = constraints._calculate_slippage("600000", 10.0, 10.0)
        slippage_volatile = constraints._calculate_slippage("600000", 10.5, 10.0)  # 5% change
        assert slippage_volatile > slippage_calm

    # ========== 交易成本测试 ==========

    def test_transaction_cost_buy(self, constraints):
        """Test transaction cost for buy."""
        cost = constraints.apply_transaction_costs(10.0, 1000, "buy")
        assert isinstance(cost, TransactionCost)
        assert cost.gross_amount == 10000.0
        assert cost.stamp_tax == 0.0  # 买入不收印花税
        assert cost.commission > 0

    def test_transaction_cost_sell(self, constraints):
        """Test transaction cost for sell."""
        cost = constraints.apply_transaction_costs(10.0, 1000, "sell")
        assert cost.stamp_tax > 0  # 卖出收取印花税

    def test_min_commission(self, constraints):
        """Test minimum commission enforcement."""
        # 小额交易应该收取最低佣金
        cost = constraints.apply_transaction_costs(10.0, 10, "buy")  # 100元
        assert cost.commission >= constraints.min_commission

    # ========== 便捷函数测试 ==========

    def test_create_constraints(self):
        """Test create_constraints helper function."""
        constraints = create_constraints(commission_rate=0.001)
        assert constraints.commission_rate == 0.001

    def test_create_constraints_defaults(self):
        """Test create_constraints with default values."""
        constraints = create_constraints()
        assert constraints.enable_limit_check is True
        assert constraints.enable_st_restriction is True


class TestConstraintsIntegration:
    """Integration tests for constraints."""

    def test_multiple_stocks(self):
        """Test constraints with multiple stocks."""
        constraints = ASShareConstraints(enable_limit_check=True)

        # 测试多只股票
        assert constraints.can_buy("600000", 10.0, 10.0).can_trade is True
        assert constraints.can_buy("300001", 10.0, 10.0).can_trade is True
        assert constraints.can_buy("688001", 10.0, 10.0).can_trade is True

    def test_limit_disabled(self):
        """Test with limit checking disabled."""
        constraints = ASShareConstraints(enable_limit_check=False)
        constraint = constraints.can_buy("600000", 11.0, 10.0)  # 涨停价
        assert constraint.can_trade is True  # 应该允许交易

    def test_st_disabled(self):
        """Test with ST restriction disabled."""
        constraints = ASShareConstraints(enable_st_restriction=False)
        constraint = constraints.can_buy("600000", 10.0, 10.0, stock_name="ST某某")
        assert constraint.can_trade is True  # 应该允许交易
