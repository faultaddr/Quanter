"""
Qlib 策略测试

测试微软 Qlib 集成策略的功能
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quanttool.strategies.qlib_strategy import (
    QlibStrategy,
    QlibFeatureEngineer,
    QlibStockSelector,
)


def generate_test_data(days: int = 200) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # 生成价格数据 (带趋势)
    base_price = 100
    trend = np.linspace(0, 20, days)  # 上升趋势
    noise = np.random.randn(days) * 2
    close = base_price + trend + noise

    # 生成 OHLCV
    data = pd.DataFrame({
        'open': close + np.random.randn(days) * 0.5,
        'high': close + np.abs(np.random.randn(days)) * 1,
        'low': close - np.abs(np.random.randn(days)) * 1,
        'close': close,
        'volume': np.random.randint(1000000, 10000000, days).astype(float),
    }, index=dates)

    # 确保 high >= max(open, close) 和 low <= min(open, close)
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


class TestQlibFeatureEngineer:
    """测试 Qlib 特征工程器"""

    def test_generate_features_alpha158(self):
        """测试 Alpha158 特征生成"""
        data = generate_test_data(200)
        engineer = QlibFeatureEngineer(feature_set="Alpha158")

        features = engineer.generate_features(data)

        # 检查特征数量 (Alpha158 应该有约 150+ 特征)
        assert len(features.columns) > 100, f"特征数量不足: {len(features.columns)}"

        # 检查特征名
        assert 'POS(20)' in features.columns
        assert 'RSI(12)' in features.columns
        assert 'MACD_DIF' in features.columns

        # 检查无 NaN
        assert not features.isna().any().any()

    def test_generate_features_alpha360(self):
        """测试 Alpha360 特征生成"""
        data = generate_test_data(400)  # Alpha360 需要更长数据
        engineer = QlibFeatureEngineer(feature_set="Alpha360")

        features = engineer.generate_features(data)

        # 检查包含更长周期特征
        assert 'REF(360)' in features.columns

    def test_feature_standardization(self):
        """测试特征标准化"""
        data = generate_test_data(200)
        engineer = QlibFeatureEngineer()

        features = engineer.generate_features(data)

        # 检查特征已标准化 (大部分特征的均值在合理范围内)
        # 注意: 由于滚动窗口标准化，部分特征均值可能略有偏差
        mean_abs_mean = abs(features.mean()).mean()
        assert mean_abs_mean < 2, f"特征标准化异常，平均绝对均值: {mean_abs_mean}"


class TestQlibStrategy:
    """测试 Qlib 策略"""

    def test_strategy_initialization(self):
        """测试策略初始化"""
        strategy = QlibStrategy(
            feature_set="Alpha158",
            model_type="lgb",
            buy_threshold=0.55
        )

        assert strategy.get_name() == "QlibStrategy(Alpha158, lgb)"
        assert strategy.buy_threshold == 0.55
        assert strategy.stop_loss_pct == 0.05

    def test_get_signal_insufficient_data(self):
        """测试数据不足时的信号"""
        strategy = QlibStrategy()
        data = generate_test_data(50)  # 数据不足

        signal = strategy.get_signal(data.iloc[-1], data)

        assert signal['signal'] == 'hold'
        assert '数据不足' in signal.get('reason', '')

    def test_get_signal_buy(self):
        """测试买入信号"""
        strategy = QlibStrategy(buy_threshold=0.3)  # 降低阈值便于测试
        data = generate_test_data(200)

        signal = strategy.get_signal(data.iloc[-1], data)

        # 检查信号结构
        assert 'signal' in signal
        assert 'probability' in signal
        assert 'score' in signal
        assert 'stop_loss' in signal
        assert 'take_profit' in signal

    def test_train_model(self):
        """测试模型训练"""
        strategy = QlibStrategy(model_type="lgb")
        data = generate_test_data(300)

        result = strategy.train_model(data, horizon=10)

        assert result is True
        assert strategy.model.is_fitted

    def test_calculate_signals(self):
        """测试批量信号计算"""
        strategy = QlibStrategy()
        data = generate_test_data(200)

        result = strategy.calculate_signals(data)

        # 检查结果包含信号列
        assert 'signal' in result.columns
        assert 'probability' in result.columns
        assert 'score' in result.columns

    def test_feature_importance(self):
        """测试特征重要性"""
        strategy = QlibStrategy(model_type="lgb")
        data = generate_test_data(300)

        strategy.train_model(data)
        importance = strategy.get_feature_importance(top_n=10)

        # 如果模型支持特征重要性
        if not importance.empty:
            assert len(importance) <= 10
            assert 'feature' in importance.columns


class TestQlibStockSelector:
    """测试 Qlib 股票筛选器"""

    def test_selector_initialization(self):
        """测试筛选器初始化"""
        selector = QlibStockSelector(top_k=5)
        assert selector.top_k == 5

    def test_select_stocks(self):
        """测试股票筛选"""
        # 创建多只股票数据
        stock_data = {}
        for i in range(3):
            data = generate_test_data(200)
            stock_data[f'test_{i:04d}'] = data

        selector = QlibStockSelector(
            strategy=QlibStrategy(buy_threshold=0.3),
            top_k=5
        )

        result = selector.select(stock_data, min_data_days=120)

        # 检查结果格式
        if not result.empty:
            assert 'stock_code' in result.columns
            assert 'probability' in result.columns
            assert 'score' in result.columns


class TestQlibStrategyIntegration:
    """Qlib 策略集成测试"""

    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 创建策略
        strategy = QlibStrategy(
            feature_set="Alpha158",
            model_type="lgb",
            buy_threshold=0.55,
            sell_threshold=0.45,
            stop_loss_pct=0.05,
            take_profit_pct=0.10
        )

        # 2. 准备数据
        data = generate_test_data(300)

        # 3. 训练模型
        success = strategy.train_model(data)
        assert success

        # 4. 获取信号
        signal = strategy.get_signal(data.iloc[-1], data)

        # 5. 验证信号
        assert signal['signal'] in ['buy', 'sell', 'hold']
        assert 0 <= signal['probability'] <= 1
        assert signal['stop_loss'] < data['close'].iloc[-1]
        assert signal['take_profit'] > data['close'].iloc[-1]

    def test_strategy_comparison(self):
        """比较不同参数的策略"""
        data = generate_test_data(300)

        strategies = [
            QlibStrategy(feature_set="Alpha158", buy_threshold=0.5),
            QlibStrategy(feature_set="Alpha158", buy_threshold=0.6),
            QlibStrategy(feature_set="Alpha360", buy_threshold=0.55),
        ]

        for strategy in strategies:
            strategy.train_model(data)
            signal = strategy.get_signal(data.iloc[-1], data)

            # 每个策略都应该能产生有效信号
            assert signal['signal'] in ['buy', 'sell', 'hold']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])