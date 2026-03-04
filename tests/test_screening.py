"""
筛选层模块测试

测试 K线形态筛选器的核心逻辑：
- 低位 + 看涨形态 = PASS
- 高位 + 看涨形态 = WARNING
- 高位 + 看跌形态 = FILTER
- 低位 + 看跌形态 = WARNING
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quanttool.factors.screening import (
    ScreenResult,
    ScreeningOutcome,
    CandlestickPatternScreener,
    StockScreener,
)


class TestCandlestickPatternScreener:
    """K线形态筛选器测试"""

    @pytest.fixture
    def screener(self):
        """创建筛选器实例"""
        return CandlestickPatternScreener()

    def _create_df_with_pattern(
        self,
        pattern_type: str = "bullish",
        num_days: int = 10
    ) -> pd.DataFrame:
        """
        创建模拟K线数据

        Args:
            pattern_type: 形态类型 'bullish' 或 'bearish'
            num_days: 天数

        Note:
            K线形态识别器使用 c1, c2, c3 = df['close'].iloc[-3:].values
            所以 c1, o1 是倒数第三天，c2, o2 是倒数第二天，c3, o3 是最后一天
            看涨吞没检查: c1 < o1 (前一天阴线) and c2 > o2 (后一天阳线)
        """
        dates = pd.date_range(start='2024-01-01', periods=num_days, freq='D')

        if pattern_type == "bullish":
            # 创建看涨吞没形态
            # c1, o1 = 倒数第三天（被吞没的前一天）
            # c2, o2 = 倒数第二天（吞没的那一天）
            base_price = 100
            opens = []
            closes = []
            highs = []
            lows = []

            # 前7天：震荡下跌趋势
            for i in range(num_days - 3):
                o = base_price - i * 2
                c = o - 1
                opens.append(o)
                closes.append(c)
                highs.append(o + 0.5)
                lows.append(c - 0.5)

            # 倒数第三天 (c1, o1)：阴线，作为被吞没的前一天
            opens.append(85)    # o1
            closes.append(80)   # c1 < o1 = 阴线
            highs.append(86)
            lows.append(79)

            # 倒数第二天 (c2, o2)：大阳线吞没前一天
            opens.append(79)    # o2 < c1 (开盘低于前一天收盘)
            closes.append(88)   # c2 > o2 = 阳线, c2 > o1 = 吞没
            highs.append(90)
            lows.append(78)

            # 最后一天：小阳线延续
            opens.append(87)
            closes.append(89)
            highs.append(91)
            lows.append(86)

        else:  # bearish
            # 创建看跌吞没形态
            base_price = 100
            opens = []
            closes = []
            highs = []
            lows = []

            # 前7天：震荡上涨趋势
            for i in range(num_days - 3):
                o = base_price + i * 2
                c = o + 1
                opens.append(o)
                closes.append(c)
                highs.append(c + 0.5)
                lows.append(o - 0.5)

            # 倒数第三天 (c1, o1)：阳线，作为被吞没的前一天
            opens.append(115)   # o1
            closes.append(120)  # c1 > o1 = 阳线
            highs.append(122)
            lows.append(114)

            # 倒数第二天 (c2, o2)：大阴线吞没前一天
            opens.append(121)   # o2 > c1 (开盘高于前一天收盘)
            closes.append(112)  # c2 < o2 = 阴线, c2 < o1 = 吞没
            highs.append(123)
            lows.append(111)

            # 最后一天：小阴线延续
            opens.append(113)
            closes.append(111)
            highs.append(115)
            lows.append(110)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
        })

        return df

    def test_low_position_bullish_should_pass(self, screener):
        """
        测试：低位 + 看涨形态 = PASS

        底部信号确认，应该通过筛选
        """
        df = self._create_df_with_pattern("bullish")

        # 模拟低位：position_ratio < 0.35
        result = screener.screen(
            df,
            position_ratio=0.2,  # 低位
            bias20=-0.08,        # 负乖离率
            boll_pctb=0.1        # 接近布林带下轨
        )

        assert result.result == ScreenResult.PASS
        assert "底部信号" in "".join(result.reasons) or "PASS" in "".join(result.reasons)

    def test_high_position_bullish_should_warning(self, screener):
        """
        测试：高位 + 看涨形态 = WARNING

        警惕诱多/力竭，应该警示
        """
        df = self._create_df_with_pattern("bullish")

        # 模拟高位：position_ratio > 0.70
        result = screener.screen(
            df,
            position_ratio=0.85,  # 高位
            bias20=0.06,          # 正乖离率
            boll_pctb=0.9         # 接近布林带上轨
        )

        assert result.result == ScreenResult.WARNING
        assert result.score_modifier < 1.0  # 评分修正系数应该降低

    def test_high_position_bearish_should_filter(self, screener):
        """
        测试：高位 + 看跌形态 = FILTER

        顶部信号，建议回避，应该过滤
        """
        df = self._create_df_with_pattern("bearish")

        # 模拟高位
        result = screener.screen(
            df,
            position_ratio=0.85,  # 高位
            bias20=0.06,
            boll_pctb=0.9
        )

        assert result.result == ScreenResult.FILTER
        assert "顶部信号" in "".join(result.reasons) or "回避" in "".join(result.reasons)

    def test_low_position_bearish_should_warning(self, screener):
        """
        测试：低位 + 看跌形态 = WARNING

        可能是最后洗盘，应该警示
        """
        df = self._create_df_with_pattern("bearish")

        # 模拟低位
        result = screener.screen(
            df,
            position_ratio=0.2,  # 低位
            bias20=-0.08,
            boll_pctb=0.1
        )

        assert result.result == ScreenResult.WARNING
        assert "洗盘" in "".join(result.reasons) or "WARNING" in "".join(result.reasons)

    def test_mid_position_neutral_should_pass(self, screener):
        """
        测试：中位 + 无显著形态 = PASS

        默认通过
        """
        # 创建无明显形态的数据
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100.5] * 10,
        })

        # 模拟中位
        result = screener.screen(
            df,
            position_ratio=0.5,
            bias20=0.0,
            boll_pctb=0.5
        )

        # 无显著形态应该通过
        assert result.result == ScreenResult.PASS


class TestStockScreener:
    """综合筛选器测试"""

    @pytest.fixture
    def screener(self):
        """创建综合筛选器实例"""
        return StockScreener()

    def _create_mock_data(self) -> pd.DataFrame:
        """创建模拟股票数据"""
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        np.random.seed(42)

        # 创建有波动的价格序列
        close_prices = 100 + np.cumsum(np.random.randn(60) * 0.5)

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close_prices + np.random.rand(60) * 0.5,
            'high': close_prices + np.random.rand(60) * 1,
            'low': close_prices - np.random.rand(60) * 1,
            'close': close_prices,
        })

        return df

    def test_screen_with_high_bias_should_filter(self, screener):
        """
        测试：乖离率过高(>+8%)应该过滤
        """
        df = self._create_mock_data()

        # 模拟高乖离率评分结果
        score_result = {
            'factors_raw': {
                'position_ratio': 0.9,
                'bias20': 0.10,  # 乖离率 > 8%
                'boll_pctb': 0.95,
            }
        }

        result = screener.screen(df, score_result)

        assert result.result == ScreenResult.FILTER
        assert "乖离率过高" in "".join(result.reasons)

    def test_screen_normal_should_pass(self, screener):
        """
        测试：正常情况应该通过
        """
        df = self._create_mock_data()

        # 模拟正常评分结果
        score_result = {
            'factors_raw': {
                'position_ratio': 0.5,
                'bias20': 0.02,
                'boll_pctb': 0.5,
            }
        }

        result = screener.screen(df, score_result)

        # 正常情况应该通过（取决于K线形态）
        assert result.result in [ScreenResult.PASS, ScreenResult.WARNING]


class TestScreeningOutcome:
    """筛选结果数据类测试"""

    def test_default_values(self):
        """测试默认值"""
        outcome = ScreeningOutcome(result=ScreenResult.PASS)

        assert outcome.result == ScreenResult.PASS
        assert outcome.score_modifier == 1.0
        assert outcome.reasons == []
        assert outcome.details == {}

    def test_with_values(self):
        """测试带值初始化"""
        outcome = ScreeningOutcome(
            result=ScreenResult.WARNING,
            score_modifier=0.9,
            reasons=["测试原因"],
            details={"key": "value"}
        )

        assert outcome.result == ScreenResult.WARNING
        assert outcome.score_modifier == 0.9
        assert "测试原因" in outcome.reasons
        assert outcome.details["key"] == "value"


class TestScreenResult:
    """筛选结果枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert ScreenResult.PASS.value == "pass"
        assert ScreenResult.FILTER.value == "filter"
        assert ScreenResult.WARNING.value == "warning"