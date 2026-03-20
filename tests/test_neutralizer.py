"""Tests for neutralizer module."""

import pytest
import pandas as pd
import numpy as np

from quanttool.factors.neutralizer import (
    FactorNeutralizer,
    neutralize_factor,
)


class TestFactorNeutralizer:
    """Test cases for FactorNeutralizer."""

    @pytest.fixture
    def neutralizer(self):
        """Create neutralizer instance."""
        return FactorNeutralizer()

    @pytest.fixture
    def sample_data(self):
        """Create sample data for neutralization."""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame({
            'factor': np.random.randn(n) * 10 + 50,
            'market_cap': np.random.exponential(1000, n),
            'industry': np.random.choice(['银行', '地产', '科技', '消费'], n),
        })

        return data['factor'], data['market_cap'], data['industry']

    # ========== 市值中性化测试 ==========

    def test_neutralize_by_market_cap(self, neutralizer, sample_data):
        """Test market cap neutralization."""
        factor, market_cap, _ = sample_data

        result = neutralizer.neutralize_by_market_cap(factor, market_cap)

        # 中性化后，因子应该与市值无关
        # 检查因子与市值的相关性是否降低
        from scipy.stats import spearmanr
        corr_before, _ = spearmanr(factor, market_cap)
        corr_after, _ = spearmanr(result, market_cap)

        assert abs(corr_after) < abs(corr_before) or np.isnan(corr_before)

    def test_neutralize_by_market_cap_log(self, neutralizer, sample_data):
        """Test market cap neutralization with log transform."""
        factor, market_cap, _ = sample_data

        result = neutralizer.neutralize_by_market_cap(factor, market_cap, log_transform=True)

        # 结果不应该有NaN
        assert result.isna().sum() == 0

    def test_neutralize_by_market_cap_insufficient_data(self, neutralizer):
        """Test neutralization with insufficient data."""
        factor = pd.Series([1, 2, 3])
        market_cap = pd.Series([100, 200, 300])

        result = neutralizer.neutralize_by_market_cap(factor, market_cap)
        # 数据不足应该返回原始因子
        assert result.equals(factor)

    # ========== 行业中性化测试 ==========

    def test_neutralize_by_industry(self, neutralizer, sample_data):
        """Test industry neutralization."""
        factor, _, industry = sample_data

        result = neutralizer.neutralize_by_industry(factor, industry)

        # 中性化后，各行业的均值应该接近
        neutralized_by_industry = pd.DataFrame({
            'factor': result,
            'industry': industry
        })

        industry_means = neutralized_by_industry.groupby('industry')['factor'].mean()
        assert industry_means.std() < 0.1  # 行业间差异应该很小

    def test_neutralize_by_industry_all_same(self, neutralizer):
        """Test industry neutralization with same industry."""
        factor = pd.Series([1, 2, 3, 4, 5])
        industry = pd.Series(['银行'] * 5)

        result = neutralizer.neutralize_by_industry(factor, industry)
        # 同一行业，中性化后应该全为0
        assert result.sum() < 0.01

    # ========== 行业+市值联合中性化测试 ==========

    def test_neutralize_industry_and_market_cap(self, neutralizer, sample_data):
        """Test combined industry and market cap neutralization."""
        factor, market_cap, industry = sample_data

        result = neutralizer.neutralize_industry_and_market_cap(
            factor, industry, market_cap, order="industry_first"
        )

        # 结果不应该有NaN
        assert result.isna().sum() == 0

    def test_neutralize_market_cap_first(self, neutralizer, sample_data):
        """Test neutralization with market cap first."""
        factor, market_cap, industry = sample_data

        result = neutralizer.neutralize_industry_and_market_cap(
            factor, industry, market_cap, order="market_cap_first"
        )

        assert result.isna().sum() == 0

    # ========== 多因子中性化测试 ==========

    def test_neutralize_multi_factor(self, neutralizer):
        """Test multi-factor neutralization."""
        np.random.seed(42)
        n = 100

        factors = pd.DataFrame({
            'factor1': np.random.randn(n) * 10,
            'factor2': np.random.randn(n) * 20,
            'factor3': np.random.randn(n) * 5,
        })

        control = pd.DataFrame({
            'factor3': factors['factor3'],  # 控制变量
            'size': np.random.randn(n),
        })

        result = neutralizer.neutralize_multi_factor(factors, control)

        # factor1和factor2应该被中性化
        assert 'factor1' in result.columns
        assert 'factor2' in result.columns

    # ========== 风格因子中性化测试 ==========

    def test_neutralize_with_style_factors(self, neutralizer):
        """Test neutralization with style factors."""
        np.random.seed(42)
        n = 100

        factor = pd.Series(np.random.randn(n) * 10 + 50)
        style_factors = pd.DataFrame({
            'size': np.random.randn(n),
            'value': np.random.randn(n),
            'momentum': np.random.randn(n),
        })

        result = neutralizer.neutralize_with_style_factors(factor, style_factors)

        assert result.isna().sum() == 0


class TestNeutralizeFactorHelper:
    """Test helper function."""

    def test_neutralize_factor_industry_only(self):
        """Test neutralize_factor with industry only."""
        factor = pd.Series(np.random.randn(100))
        industry = pd.Series(np.random.choice(['A', 'B', 'C'], 100))

        result = neutralize_factor(factor, industry=industry)

        assert result is not None

    def test_neutralize_factor_market_cap_only(self):
        """Test neutralize_factor with market cap only."""
        factor = pd.Series(np.random.randn(100))
        market_cap = pd.Series(np.random.exponential(1000, 100))

        result = neutralize_factor(factor, market_cap=market_cap)

        assert result is not None

    def test_neutralize_factor_no_neutralization(self):
        """Test neutralize_factor with no neutralization."""
        factor = pd.Series(np.random.randn(100))

        result = neutralize_factor(factor)

        # 没有提供任何中性化参数，应该返回原始因子
        assert result.equals(factor)
