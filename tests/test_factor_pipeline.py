"""Tests for factor pipeline module."""

import pytest
import pandas as pd
import numpy as np

from quanttool.factors.factor_pipeline import (
    FactorPipeline,
    PipelineConfig,
    create_pipeline,
    process_factors,
    NeutralizationType,
    StandardizationMethod,
)


class TestFactorPipeline:
    """Test cases for FactorPipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create sample factor data."""
        np.random.seed(42)
        n = 100

        # 创建包含极端值的数据
        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n),
            'factor1': np.random.randn(n) * 10 + 50,
            'factor2': np.random.randn(n) * 20 + 100,
            'factor3': np.random.randn(n) * 5 + 25,
            'industry': np.random.choice(['银行', '地产', '科技', '消费'], n),
            'market_cap': np.random.exponential(1000, n),
        })

        # 添加一些极端值
        data.loc[0, 'factor1'] = 1000  # 极端高值
        data.loc[1, 'factor1'] = -500   # 极端低值

        # 添加缺失值
        data.loc[10, 'factor1'] = np.nan
        data.loc[20, 'factor2'] = np.nan

        return data.set_index('date')

    @pytest.fixture
    def config(self):
        """Create pipeline config."""
        return PipelineConfig(
            winsorize_lower=0.01,
            winsorize_upper=0.99,
            standardization=StandardizationMethod.ZSCORE,
            neutralization=NeutralizationType.NONE,
            fill_method='median',
        )

    # ========== 初始化测试 ==========

    def test_pipeline_init_default(self):
        """Test pipeline initialization with defaults."""
        pipeline = FactorPipeline()
        assert pipeline.config.winsorize_lower == 0.01
        assert pipeline.config.winsorize_upper == 0.99

    def test_pipeline_init_custom(self, config):
        """Test pipeline initialization with custom config."""
        pipeline = FactorPipeline(config)
        assert pipeline.config == config

    # ========== 缺失值处理测试 ==========

    def test_handle_missing_values_median(self, sample_data, config):
        """Test missing value handling with median."""
        config.fill_method = 'median'
        pipeline = FactorPipeline(config)

        result = pipeline._handle_missing_values(sample_data, ['factor1', 'factor2'])
        assert result['factor1'].isna().sum() == 0
        assert result['factor2'].isna().sum() == 0

    def test_handle_missing_values_zero(self, sample_data, config):
        """Test missing value handling with zero."""
        config.fill_method = 'zero'
        pipeline = FactorPipeline(config)

        result = pipeline._handle_missing_values(sample_data, ['factor1'])
        assert result['factor1'].isna().sum() == 0

    # ========== 去极值测试 ==========

    def test_winsorize(self, sample_data, config):
        """Test winsorization."""
        pipeline = FactorPipeline(config)
        result = pipeline._winsorize(sample_data, ['factor1'])

        # 极值应该被截断
        max_val = result['factor1'].quantile(0.99)
        original_max = sample_data['factor1'].max()
        assert max_val < original_max

    # ========== 标准化测试 ==========

    def test_zscore_standardization(self, sample_data, config):
        """Test Z-score standardization."""
        config.standardization = StandardizationMethod.ZSCORE
        pipeline = FactorPipeline(config)

        # 先处理缺失值和去极值
        data = sample_data.copy()
        data = pipeline._handle_missing_values(data, ['factor1'])
        data = pipeline._winsorize(data, ['factor1'])

        result, stats = pipeline._standardize(data, ['factor1'])

        # Z-score 后均值接近 0，标准差接近 1
        assert abs(result['factor1'].mean()) < 0.1
        assert abs(result['factor1'].std() - 1.0) < 0.1

    def test_rank_standardization(self, sample_data, config):
        """Test rank standardization."""
        config.standardization = StandardizationMethod.RANK
        pipeline = FactorPipeline(config)

        data = sample_data.copy()
        data = pipeline._handle_missing_values(data, ['factor1'])
        data = pipeline._winsorize(data, ['factor1'])

        result, _ = pipeline._standardize(data, ['factor1'])

        # 排序标准化后应该在 [0, 1] 范围内
        assert result['factor1'].min() >= 0
        assert result['factor1'].max() <= 1

    def test_minmax_standardization(self, sample_data, config):
        """Test Min-Max standardization."""
        config.standardization = StandardizationMethod.MINMAX
        pipeline = FactorPipeline(config)

        data = sample_data.copy()
        data = pipeline._handle_missing_values(data, ['factor1'])
        data = pipeline._winsorize(data, ['factor1'])

        result, _ = pipeline._standardize(data, ['factor1'])

        # Min-Max 后应该在 [0, 1] 范围内
        assert result['factor1'].min() >= 0
        assert result['factor1'].max() <= 1

    # ========== 中性化测试 ==========

    def test_industry_neutralization(self, sample_data, config):
        """Test industry neutralization."""
        config.neutralization = NeutralizationType.INDUSTRY
        pipeline = FactorPipeline(config)

        result_data = sample_data.copy()
        result_data = pipeline._handle_missing_values(result_data, ['factor1'])
        result_data = pipeline._winsorize(result_data, ['factor1'])

        result = pipeline._neutralize(
            result_data,
            ['factor1'],
            market_cap_column='market_cap',
            industry_column='industry',
        )

        # 行业中性化后，各行业的因子均值应该接近
        industry_means = result.groupby('industry')['factor1'].mean()
        assert industry_means.std() < 0.1  # 行业间差异应该很小

    # ========== 完整流水线测试 ==========

    def test_full_pipeline(self, sample_data, config):
        """Test complete pipeline processing."""
        pipeline = FactorPipeline(config)
        result = pipeline.process(
            sample_data,
            factor_columns=['factor1', 'factor2'],
            market_cap_column='market_cap',
            industry_column='industry',
        )

        assert result.data is not None
        assert 'factor1' in result.data.columns
        assert 'factor2' in result.data.columns

    def test_pipeline_statistics(self, sample_data, config):
        """Test pipeline statistics output."""
        pipeline = FactorPipeline(config)
        result = pipeline.process(
            sample_data,
            factor_columns=['factor1'],
        )

        assert 'factor1' in result.statistics
        assert 'mean' in result.statistics['factor1']
        assert 'std' in result.statistics['factor1']


class TestPipelineHelperFunctions:
    """Test helper functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample factor data."""
        np.random.seed(42)
        n = 100

        data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=n),
            'factor1': np.random.randn(n) * 10 + 50,
            'factor2': np.random.randn(n) * 20 + 100,
            'factor3': np.random.randn(n) * 5 + 25,
            'industry': np.random.choice(['银行', '地产', '科技', '消费'], n),
            'market_cap': np.random.exponential(1000, n),
        })
        return data.set_index('date')

    def test_create_pipeline(self):
        """Test create_pipeline helper."""
        pipeline = create_pipeline(
            winsorize=(0.02, 0.98),
            standardization='rank',
            neutralization='industry',
        )
        assert pipeline.config.winsorize_lower == 0.02
        assert pipeline.config.winsorize_upper == 0.98
        assert pipeline.config.standardization == StandardizationMethod.RANK

    def test_process_factors(self, sample_data):
        """Test process_factors helper."""
        result = process_factors(
            sample_data,
            factor_columns=['factor1', 'factor2'],
            industry_column='industry',
        )
        assert 'factor1' in result.columns
        assert 'factor2' in result.columns
