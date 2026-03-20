"""Tests for enhanced weight optimizer module."""

import pytest
import pandas as pd
import numpy as np

from quanttool.optimization.weight_optimizer import (
    ICIRWeightOptimizer,
    OptimizerType,
    WeightConfig,
    MarketRegime,
    DynamicWeightOptimizer,
)


class TestICIRWeightOptimizer:
    """Test cases for ICIRWeightOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance."""
        return ICIRWeightOptimizer(
            min_weight=0.05,
            max_weight=0.50,
        )

    @pytest.fixture
    def factor_names(self):
        """Sample factor names."""
        return ["factor1", "factor2", "factor3"]

    @pytest.fixture
    def sample_ic_data(self, optimizer):
        """Setup sample IC data."""
        for name in ["factor1", "factor2", "factor3"]:
            ic_series = pd.Series(np.random.randn(100) * 0.05 + 0.03)
            optimizer.update_factor_metrics(name, ic_series)

    # ========== IC优化测试 ==========

    def test_optimize_by_ic(self, optimizer, factor_names):
        """Test IC-weighted optimization."""
        # 添加IC数据
        optimizer.ic_history = {
            "factor1": [0.08, 0.06, 0.07],
            "factor2": [0.03, 0.02, 0.04],
            "factor3": [0.05, 0.04, 0.05],
        }

        weights = optimizer.optimize_by_ic(factor_names)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01  # 权重和为1

    def test_optimize_by_ic_no_data(self, optimizer, factor_names):
        """Test IC optimization with no data."""
        weights = optimizer.optimize_by_ic(factor_names)

        # 应该返回等权
        for w in weights.values():
            assert abs(w - 1.0 / 3) < 0.01

    # ========== IR优化测试 ==========

    def test_optimize_by_ir(self, optimizer, factor_names):
        """Test IR-weighted optimization."""
        optimizer.ir_history = {
            "factor1": [1.5, 1.2, 1.3],
            "factor2": [0.5, 0.4, 0.6],
            "factor3": [1.0, 0.8, 0.9],
        }

        weights = optimizer.optimize_by_ir(factor_names)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    # ========== 风险平价测试 ==========

    def test_optimize_risk_parity_no_returns(self, optimizer, factor_names):
        """Test risk parity without returns data."""
        weights = optimizer.optimize_risk_parity(factor_names, None)

        # 没有数据应该返回等权
        for w in weights.values():
            assert abs(w - 1.0 / 3) < 0.01

    def test_optimize_risk_parity_with_returns(self, optimizer, factor_names):
        """Test risk parity with returns."""
        np.random.seed(42)
        returns = pd.DataFrame({
            "factor1": np.random.randn(100),
            "factor2": np.random.randn(100),
            "factor3": np.random.randn(100),
        })

        weights = optimizer.optimize_risk_parity(factor_names, returns)

        assert len(weights) == 3
        assert abs(sum(weights.values()) - 1.0) < 0.01

    # ========== 均值方差测试 ==========

    def test_optimize_mean_variance(self, optimizer, factor_names):
        """Test mean-variance optimization."""
        np.random.seed(42)
        returns = pd.DataFrame({
            "factor1": np.random.randn(100) * 0.01 + 0.001,
            "factor2": np.random.randn(100) * 0.02 + 0.002,
            "factor3": np.random.randn(100) * 0.015 + 0.0015,
        })

        weights = optimizer.optimize_mean_variance(factor_names, returns)

        assert len(weights) == 3
        assert all(0 <= w <= 1 for w in weights.values())

    def test_optimize_mean_variance_no_returns(self, optimizer, factor_names):
        """Test mean-variance without returns - should return equal weights."""
        weights = optimizer.optimize_mean_variance(factor_names, None)
        # 没有数据应该返回等权
        for w in weights.values():
            assert abs(w - 1.0 / 3) < 0.01

    # ========== 权重约束测试 ==========

    def test_weight_constraints(self, optimizer):
        """Test weight constraints."""
        weights = {"f1": 0.02, "f2": 0.98}  # 违反约束

        result = optimizer._apply_weight_constraints(weights)

        # 应该应用约束 - 最小权重不低于0.05
        assert result["f1"] >= 0.05
        # f2应该被约束到合理范围
        assert result["f2"] <= 1.0

    # ========== 完整优化测试 ==========

    def test_optimize_equal(self, optimizer, factor_names):
        """Test equal weight optimization."""
        weights = optimizer.optimize(factor_names, OptimizerType.EQUAL)

        for w in weights.values():
            assert abs(w - 1.0 / 3) < 0.01

    def test_optimize_ic_weighted(self, optimizer, factor_names):
        """Test IC weighted optimization."""
        optimizer.ic_history = {
            "factor1": [0.1],
            "factor2": [0.05],
            "factor3": [0.08],
        }

        weights = optimizer.optimize(factor_names, OptimizerType.IC_WEIGHTED)

        assert len(weights) == 3


class TestDynamicWeightOptimizer:
    """Test cases for DynamicWeightOptimizer."""

    @pytest.fixture
    def dyn_optimizer(self):
        """Create dynamic optimizer."""
        return DynamicWeightOptimizer(lookback_period=60)

    @pytest.fixture
    def price_data(self):
        """Create sample price data."""
        np.random.seed(42)
        n = 100

        # 创建牛市数据
        returns = np.random.normal(0.001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n),
        }, index=pd.date_range("2023-01-01", periods=n))

        return df

    def test_detect_market_regime_bull(self, dyn_optimizer, price_data):
        """Test market regime detection - bull."""
        # 创建上涨趋势
        price_data["close"] = np.linspace(100, 150, 100)

        regime = dyn_optimizer.detect_market_regime(price_data)

        assert regime in [MarketRegime.BULL, MarketRegime.VOLATILE]

    def test_detect_market_regime_bear(self, dyn_optimizer):
        """Test market regime detection - bear."""
        # 创建下跌趋势
        price_data = pd.DataFrame({
            "close": np.linspace(150, 100, 100),
        })

        regime = dyn_optimizer.detect_market_regime(price_data)

        assert regime in [MarketRegime.BEAR, MarketRegime.VOLATILE]

    def test_detect_market_regime_sideway(self, dyn_optimizer):
        """Test market regime detection - sideway."""
        # 创建震荡市场
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)

        price_data = pd.DataFrame({"close": close})

        regime = dyn_optimizer.detect_market_regime(price_data)

        assert regime in [MarketRegime.SIDEWAY, MarketRegime.BULL, MarketRegime.BEAR]

    def test_get_current_weights(self, dyn_optimizer):
        """Test getting current weights."""
        weights = dyn_optimizer.get_current_weights()

        assert hasattr(weights, "trend")
        assert hasattr(weights, "momentum")
        assert hasattr(weights, "money")
