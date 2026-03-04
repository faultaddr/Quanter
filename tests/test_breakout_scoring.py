"""
低位盘整突破评分系统测试

测试形态检测和因子评分功能
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from quanttool.factors.breakout_scoring_system import (
    BreakoutScoringSystem,
    BreakoutScoreResult,
    analyze_breakout_quality,
)


def generate_test_data(
    days: int = 300,
    trend: str = 'sideway',
    start_price: float = 10.0,
    volatility: float = 0.02,
    with_breakout: bool = False,
) -> pd.DataFrame:
    """生成测试数据"""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    if trend == 'up':
        trend_return = 0.0005
    elif trend == 'down':
        trend_return = -0.0005
    else:
        trend_return = 0

    returns = np.random.normal(trend_return, volatility, days)
    prices = start_price * np.cumprod(1 + returns)

    high = prices * (1 + np.abs(np.random.normal(0, volatility/2, days)))
    low = prices * (1 - np.abs(np.random.normal(0, volatility/2, days)))
    open_price = prices * (1 + np.random.normal(0, volatility/3, days))

    base_volume = 1000000
    volume = base_volume * (1 + np.random.uniform(-0.3, 0.3, days))

    if with_breakout:
        prices[-1] = max(prices[-20:]) * 1.05
        high[-1] = prices[-1] * 1.01
        low[-1] = prices[-1] * 0.99
        open_price[-1] = prices[-1] * 0.98
        volume[-1] = base_volume * 2.5

    return pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': prices,
        'volume': volume,
    })


def generate_consolidation_data(
    days: int = 300,
    consolidation_days: int = 25,
    start_price: float = 10.0,
    with_breakout: bool = True,
) -> pd.DataFrame:
    """生成盘整形态数据"""
    np.random.seed(42)

    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    phase1_days = days - consolidation_days - 20
    phase1_prices = start_price * np.exp(np.linspace(0, -0.35, phase1_days))
    phase1_vol = np.random.normal(0, 0.015, phase1_days)
    phase1_prices = phase1_prices * (1 + phase1_vol)

    consolidation_base = phase1_prices[-1]
    consolidation_prices = consolidation_base * (1 + np.random.uniform(
        -0.08, 0.08, consolidation_days
    ))

    if with_breakout:
        breakout_price = max(consolidation_prices) * 1.05
        breakout_prices = np.linspace(consolidation_prices[-1], breakout_price, 20)
        breakout_prices = breakout_prices * (1 + np.random.normal(0, 0.01, 20))
    else:
        breakout_prices = consolidation_prices[-1] * (1 + np.random.normal(0, 0.015, 20))

    all_prices = np.concatenate([phase1_prices, consolidation_prices, breakout_prices])

    if len(all_prices) > days:
        all_prices = all_prices[-days:]
    elif len(all_prices) < days:
        padding = all_prices[0] * np.ones(days - len(all_prices))
        all_prices = np.concatenate([padding, all_prices])

    high = all_prices * 1.02
    low = all_prices * 0.98
    open_price = all_prices * (1 + np.random.uniform(-0.01, 0.01, days))

    base_volume = 1000000
    volume = base_volume * np.ones(days)
    if with_breakout:
        volume[-20:] = base_volume * 1.8

    return pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': all_prices,
        'volume': volume,
    })


class TestBreakoutScoringSystem:
    """测试低位盘整突破评分系统"""

    def test_init(self):
        """测试初始化"""
        system = BreakoutScoringSystem()
        assert system.min_amount == 5e7
        assert system.min_list_days == 120

    def test_create_default_result(self):
        """测试默认结果创建"""
        system = BreakoutScoringSystem()
        result = system._create_default_result()

        assert result.final_score == 0
        assert result.is_low_position == False
        assert result.is_consolidating == False
        assert result.has_breakout == False
        assert result.passed_filter == False

    def test_detect_low_position_low(self):
        """测试低位检测 - 低位股票"""
        system = BreakoutScoringSystem()
        df = generate_consolidation_data(days=300, with_breakout=False)
        is_low, details = system._detect_low_position(df)
        assert details['drawdown'] >= 0.30 or details['percentile'] <= 0.35

    def test_detect_low_position_high(self):
        """测试低位检测 - 非低位股票"""
        system = BreakoutScoringSystem()
        df = generate_test_data(days=300, trend='up', start_price=10.0)
        is_low, details = system._detect_low_position(df)
        assert is_low == False

    def test_detect_breakout(self):
        """测试突破检测"""
        system = BreakoutScoringSystem()
        df = generate_test_data(days=300, with_breakout=True)
        has_breakout, details = system._detect_breakout(df)
        assert 'price_breakout' in details

    def test_calculate_factor_scores(self):
        """测试因子得分计算"""
        system = BreakoutScoringSystem()
        df = generate_test_data(days=300)
        scores = system._calculate_factor_scores(df)

        assert 'quality' in scores
        assert 'growth' in scores
        assert 'value' in scores
        assert 'momentum' in scores
        assert 'flow' in scores
        assert 'risk' in scores

        for score in scores.values():
            assert 0 <= score <= 100

    def test_calculate_score_full_pattern(self):
        """测试完整形态评分"""
        system = BreakoutScoringSystem()
        df = generate_consolidation_data(days=300, with_breakout=True)
        result = system.calculate_score(df)

        assert isinstance(result, BreakoutScoreResult)
        assert result.final_score >= 0
        assert result.consolidation_days >= 0
        assert result.volume_ratio >= 0

    def test_calculate_score_insufficient_data(self):
        """测试数据不足情况"""
        system = BreakoutScoringSystem()
        df = generate_test_data(days=50)
        result = system.calculate_score(df)

        assert result.passed_filter == False

    def test_analyze_breakout_quality(self):
        """测试便捷函数"""
        df = generate_consolidation_data(days=300, with_breakout=True)
        result = analyze_breakout_quality(df)

        assert 'final_score' in result
        assert 'is_low_position' in result
        assert 'is_consolidating' in result
        assert 'has_breakout' in result
        assert 'recommendation' in result


class TestTechnicalIndicators:
    """测试技术指标计算"""

    def test_ma(self):
        """测试移动平均"""
        system = BreakoutScoringSystem()
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ma = system._ma(data, 5)
        assert len(ma) == len(data)
        assert ma[4] == 3.0

    def test_obv(self):
        """测试OBV指标"""
        system = BreakoutScoringSystem()
        close = np.array([10, 11, 10, 12, 13])
        volume = np.array([100, 150, 120, 200, 180])
        obv = system._obv(close, volume)
        assert len(obv) == len(close)
        assert obv[0] == 0

    def test_atr(self):
        """测试ATR指标"""
        system = BreakoutScoringSystem()
        df = pd.DataFrame({
            'high': [12, 13, 11, 14, 12],
            'low': [10, 11, 9, 12, 10],
            'close': [11, 12, 10, 13, 11],
        })
        atr = system._atr(df, period=3)
        assert atr > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])