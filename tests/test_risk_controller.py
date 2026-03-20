"""Tests for enhanced risk controller module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quanttool.risk.risk_controller import (
    RiskController,
    PortfolioRiskManager,
    StopLossType,
    DrawdownLevel,
)


class TestPortfolioRiskManager:
    """Test cases for PortfolioRiskManager."""

    @pytest.fixture
    def risk_manager(self):
        """Create risk manager instance."""
        return PortfolioRiskManager(
            industry_limits={"银行": 0.20, "地产": 0.15},
            style_limits={"size": 0.5, "value": 0.5},
            max_single_stock_exposure=0.10,
            blacklist=["666666", "999999"],
        )

    @pytest.fixture
    def sample_positions(self):
        """Create sample positions."""
        return {
            "600000": {"industry": "银行", "value": 150000},
            "600036": {"industry": "银行", "value": 100000},
            "000001": {"industry": "地产", "value": 80000},
            "300001": {"industry": "科技", "value": 70000},
        }

    @pytest.fixture
    def industry_map(self):
        """Create industry mapping."""
        return {
            "600000": "银行",
            "600036": "银行",
            "000001": "地产",
            "300001": "科技",
        }

    # ========== 黑名单测试 ==========

    def test_add_to_blacklist(self, risk_manager):
        """Test adding to blacklist."""
        risk_manager.add_to_blacklist("123456")
        assert risk_manager.is_blacklisted("123456")

    def test_remove_from_blacklist(self, risk_manager):
        """Test removing from blacklist."""
        risk_manager.remove_from_blacklist("666666")
        assert not risk_manager.is_blacklisted("666666")

    def test_initial_blacklist(self, risk_manager):
        """Test initial blacklist."""
        assert risk_manager.is_blacklisted("666666")
        assert risk_manager.is_blacklisted("999999")

    # ========== 行业暴露测试 ==========

    def test_check_industry_exposure_normal(self, risk_manager, sample_positions, industry_map):
        """Test normal industry exposure."""
        portfolio_value = 400000

        violations, total = risk_manager.check_industry_exposure(
            sample_positions, industry_map, portfolio_value
        )

        # 银行暴露 = 250000/400000 = 62.5% 超过20%限制
        assert len(violations) > 0
        assert total > 0.5

    def test_check_industry_exposure_no_violation(self, risk_manager, industry_map):
        """Test industry exposure without violation."""
        positions = {
            "600000": {"industry": "银行", "value": 10000},
            "000001": {"industry": "地产", "value": 10000},
        }
        portfolio_value = 100000

        violations, total = risk_manager.check_industry_exposure(
            positions, industry_map, portfolio_value
        )

        # 各行业暴露都在限制内
        assert len(violations) == 0

    # ========== 风格暴露测试 ==========

    def test_check_style_exposure(self, risk_manager, sample_positions):
        """Test style exposure check."""
        style_factors = {
            "600000": {"size": 0.3, "value": 0.2},
            "600036": {"size": 0.4, "value": 0.3},
        }
        portfolio_value = 400000

        violations, total = risk_manager.check_style_exposure(
            sample_positions, style_factors, portfolio_value
        )

        assert isinstance(violations, list)
        assert isinstance(total, float)

    def test_check_style_exposure_empty(self, risk_manager, sample_positions):
        """Test with empty style factors."""
        violations, total = risk_manager.check_style_exposure(
            sample_positions, None, 400000
        )

        assert violations == []
        assert total == 0.0

    # ========== 黑名单检查测试 ==========

    def test_check_blacklist_violations(self, risk_manager, sample_positions):
        """Test blacklist violations check."""
        # 添加黑名单股票到持仓
        positions = sample_positions.copy()
        positions["666666"] = {"industry": "银行", "value": 50000}

        violations = risk_manager.check_blacklist_violations(positions)

        assert "666666" in violations

    # ========== 仓位收缩测试 ==========

    def test_calculate_position_shrink_no_drawdown(self, risk_manager):
        """Test position shrink with no drawdown."""
        shrink = risk_manager.calculate_position_shrink_factor(
            portfolio_value=100000,
            peak_value=100000,
            current_drawdown=0.0,
        )

        assert shrink == 1.0

    def test_calculate_position_shrink_small_drawdown(self, risk_manager):
        """Test position shrink with small drawdown."""
        shrink = risk_manager.calculate_position_shrink_factor(
            portfolio_value=95000,
            peak_value=100000,
            current_drawdown=0.05,
        )

        assert shrink == 0.9

    def test_calculate_position_shrink_large_drawdown(self, risk_manager):
        """Test position shrink with large drawdown."""
        shrink = risk_manager.calculate_position_shrink_factor(
            portfolio_value=75000,
            peak_value=100000,
            current_drawdown=0.25,
        )

        assert shrink <= 0.5

    # ========== 完整风险检查测试 ==========

    def test_check_risk_complete(self, risk_manager, sample_positions, industry_map):
        """Test complete risk check."""
        style_factors = {
            "600000": {"size": 0.3, "value": 0.2},
        }

        report = risk_manager.check_risk(
            positions=sample_positions,
            industry_map=industry_map,
            style_factors=style_factors,
            portfolio_value=400000,
            peak_value=500000,
        )

        assert report.overall_risk_score < 100  # 应该有风险
        assert len(report.recommendations) > 0

    def test_check_risk_with_blacklist(self, risk_manager, industry_map):
        """Test risk check with blacklist violations."""
        positions = {
            "666666": {"industry": "银行", "value": 50000},
        }

        report = risk_manager.check_risk(
            positions=positions,
            industry_map=industry_map,
            portfolio_value=50000,
            peak_value=50000,
        )

        assert len(report.blacklist_violations) > 0
        assert "666666" in report.blacklist_violations

    # ========== 风险摘要测试 ==========

    def test_get_risk_summary(self, risk_manager, sample_positions, industry_map):
        """Test getting risk summary."""
        # 先执行一次风险检查
        risk_manager.check_risk(
            positions=sample_positions,
            industry_map=industry_map,
            portfolio_value=400000,
            peak_value=500000,
        )

        summary = risk_manager.get_risk_summary()

        assert "risk_score" in summary
        assert summary["risk_score"] is not None


class TestRiskControllerBasics:
    """Test basic risk controller functions."""

    @pytest.fixture
    def risk_controller(self):
        """Create risk controller."""
        return RiskController(
            default_risk_per_trade=0.02,
            max_position_size=0.1,
        )

    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data."""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame({
            "open": 100 + np.random.randn(n),
            "high": 105 + np.random.randn(n),
            "low": 95 + np.random.randn(n),
            "close": 100 + np.random.randn(n),
            "volume": np.random.randint(1000000, 5000000, n),
        }, index=pd.date_range("2023-01-01", periods=n))

        return data

    # 测试原有的止损功能
    def test_calculate_dynamic_stop_loss(self, risk_controller, sample_price_data):
        """Test dynamic stop loss calculation."""
        result = risk_controller.calculate_dynamic_stop_loss(
            df=sample_price_data,
            entry_price=100.0,
            signal_strength=0.8,
        )

        assert result.stop_price < 100.0  # 止损价应该低于入场价

    def test_calculate_position_size(self, risk_controller):
        """Test position size calculation."""
        result = risk_controller.calculate_position_size(
            capital=100000,
            entry_price=10.0,
            stop_price=9.0,
        )

        assert result.shares > 0
        assert result.position_value > 0

    def test_check_drawdown_alert(self, risk_controller):
        """Test drawdown alert - 20% drawdown triggers LEVEL_4."""
        alert = risk_controller.check_drawdown_alert(
            portfolio_value=80000,
            peak_value=100000,
        )

        assert alert is not None
        # 20% drawdown matches LEVEL_4 threshold
        assert alert.level == DrawdownLevel.LEVEL_4

    def test_record_trade(self, risk_controller):
        """Test recording trade."""
        risk_controller.record_trade(
            entry_price=100.0,
            exit_price=105.0,
            max_adverse_excursion=3.0,
            max_favorable_excursion=8.0,
        )

        stats = risk_controller.get_mae_statistics()
        assert stats["count"] == 1
