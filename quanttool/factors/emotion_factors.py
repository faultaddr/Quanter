"""
情绪资金因子模块

实现市场情绪和资金流向相关的因子：
- 北向资金净流入
- 融资余额变化
- 大单净买入
- 市场情绪指标
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


@dataclass
class EmotionFactorResult:
    """情绪因子计算结果"""
    north_money_flow: float      # 北向资金净流入
    margin_balance_change: float # 融资余额变化率
    block_trade_net: float       # 大单净买入
    sentiment_index: float       # 市场情绪指数
    composite_score: float       # 综合情绪评分
    confidence: float            # 置信度


class EmotionFactorCalculator:
    """
    情绪资金因子计算器

    计算与市场情绪和资金流向相关的因子
    """

    # 情绪因子权重
    EMOTION_FACTOR_WEIGHTS = {
        'north_money_flow': 0.30,    # 北向资金权重
        'margin_balance': 0.25,      # 融资余额权重
        'block_trade': 0.25,         # 大单交易权重
        'sentiment': 0.20,           # 市场情绪权重
    }

    def __init__(
        self,
        lookback_period: int = 20,
        north_money_threshold: float = 0.02,
        margin_change_threshold: float = 0.03
    ):
        """
        初始化情绪因子计算器

        Args:
            lookback_period: 回看周期
            north_money_threshold: 北向资金变化阈值
            margin_change_threshold: 融资余额变化阈值
        """
        self.lookback_period = lookback_period
        self.north_money_threshold = north_money_threshold
        self.margin_change_threshold = margin_change_threshold

    def calculate_all_factors(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> EmotionFactorResult:
        """
        计算所有情绪因子

        Args:
            df: 股票数据DataFrame
            external_data: 外部数据字典（北向资金、融资数据等）

        Returns:
            EmotionFactorResult: 因子计算结果
        """
        if df.empty or len(df) < self.lookback_period:
            return self._create_empty_result()

        # 计算各因子
        north_money_score = self._calculate_north_money_score(df, external_data)
        margin_score = self._calculate_margin_score(df, external_data)
        block_trade_score = self._calculate_block_trade_score(df)
        sentiment_score = self._calculate_sentiment_score(df)

        # 计算综合评分
        composite_score = (
            north_money_score * self.EMOTION_FACTOR_WEIGHTS['north_money_flow'] +
            margin_score * self.EMOTION_FACTOR_WEIGHTS['margin_balance'] +
            block_trade_score * self.EMOTION_FACTOR_WEIGHTS['block_trade'] +
            sentiment_score * self.EMOTION_FACTOR_WEIGHTS['sentiment']
        )

        # 计算置信度
        confidence = self._calculate_confidence(df, external_data)

        return EmotionFactorResult(
            north_money_flow=north_money_score,
            margin_balance_change=margin_score,
            block_trade_net=block_trade_score,
            sentiment_index=sentiment_score,
            composite_score=composite_score,
            confidence=confidence
        )

    def calculate_north_money_flow(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> float:
        """
        计算北向资金净流入因子

        北向资金是A股市场重要的增量资金来源

        Args:
            df: 股票数据DataFrame
            external_data: 外部数据（包含北向资金数据）

        Returns:
            float: 北向资金评分 (0-100)
        """
        return self._calculate_north_money_score(df, external_data)

    def _calculate_north_money_score(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> float:
        """
        计算北向资金评分

        评分逻辑：
        1. 有外部数据时，使用实际北向资金数据
        2. 无外部数据时，使用量价关系估算
        """
        if external_data and 'north_money_flow' in external_data:
            # 使用实际北向资金数据
            north_flow = external_data['north_money_flow']
            if isinstance(north_flow, (list, np.ndarray)):
                # 使用近N日净流入
                recent_flow = sum(north_flow[-5:]) if len(north_flow) >= 5 else sum(north_flow)
            else:
                recent_flow = north_flow

            # 计算评分：正值加分，负值减分
            if recent_flow > 0:
                return min(100, 50 + recent_flow * 10)  # 正流入加分
            else:
                return max(0, 50 + recent_flow * 10)    # 负流入减分

        # 无外部数据时，使用量价关系估算
        # 如果股价上涨且放量，假设有资金流入
        if len(df) < 5:
            return 50.0

        recent = df.tail(5)
        price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
        volume_change = recent['volume'].iloc[-1] / recent['volume'].iloc[0] if recent['volume'].iloc[0] > 0 else 1.0

        # 价涨量增 = 资金流入
        if price_change > 0 and volume_change > 1.0:
            return min(100, 50 + price_change * 500 + (volume_change - 1) * 20)
        # 价跌量增 = 资金流出（可能是主力出货）
        elif price_change < 0 and volume_change > 1.0:
            return max(0, 50 + price_change * 500 - (volume_change - 1) * 10)
        # 价涨量缩 = 观望
        elif price_change > 0 and volume_change < 1.0:
            return 50 + price_change * 200
        # 价跌量缩 = 资金流出
        else:
            return max(0, 50 + price_change * 300)

    def calculate_margin_balance(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> float:
        """
        计算融资余额变化因子

        融资余额增加表示市场看好，减少表示看淡

        Args:
            df: 股票数据DataFrame
            external_data: 外部数据（包含融资数据）

        Returns:
            float: 融资余额评分 (0-100)
        """
        return self._calculate_margin_score(df, external_data)

    def _calculate_margin_score(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> float:
        """
        计算融资余额评分
        """
        if external_data and 'margin_balance' in external_data:
            margin_data = external_data['margin_balance']
            if isinstance(margin_data, dict):
                current = margin_data.get('current', 0)
                previous = margin_data.get('previous', current)
            elif isinstance(margin_data, (list, np.ndarray)):
                current = margin_data[-1] if len(margin_data) > 0 else 0
                previous = margin_data[-2] if len(margin_data) > 1 else current
            else:
                current = margin_data
                previous = current

            if previous > 0:
                change_rate = (current - previous) / previous
                # 融资余额增加 = 看多
                if change_rate > 0:
                    return min(100, 50 + change_rate * 1000)
                else:
                    return max(0, 50 + change_rate * 1000)

        # 无外部数据时，使用价格趋势估算
        if len(df) < 10:
            return 50.0

        # 使用均线斜率估算融资意愿
        ma5 = df['close'].rolling(5).mean()
        ma10 = df['close'].rolling(10).mean()

        if len(ma5) > 1 and len(ma10) > 1:
            slope = (ma5.iloc[-1] - ma5.iloc[-5]) / ma5.iloc[-5] if ma5.iloc[-5] > 0 else 0
            return min(100, max(0, 50 + slope * 500))

        return 50.0

    def calculate_block_trade(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> float:
        """
        计算大单净买入因子

        大单买入表示主力资金看好

        Args:
            df: 股票数据DataFrame
            external_data: 外部数据（包含大单数据）

        Returns:
            float: 大单净买入评分 (0-100)
        """
        return self._calculate_block_trade_score(df, external_data)

    def _calculate_block_trade_score(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> float:
        """
        计算大单净买入评分
        """
        if external_data and 'block_trade' in external_data:
            block_data = external_data['block_trade']
            if isinstance(block_data, dict):
                buy = block_data.get('buy', 0)
                sell = block_data.get('sell', 0)
                net = buy - sell
                total = buy + sell if (buy + sell) > 0 else 1
                net_ratio = net / total
                return min(100, max(0, 50 + net_ratio * 100))
            elif isinstance(block_data, (int, float)):
                return min(100, max(0, 50 + block_data))

        # 无外部数据时，使用成交量和价格波动估算
        if len(df) < 5:
            return 50.0

        # 大单通常伴随较大的价格波动
        recent = df.tail(5)
        price_volatility = recent['close'].pct_change().std()
        volume_mean = recent['volume'].mean()
        volume_std = recent['volume'].std()

        # 成交量波动大 + 价格波动大 = 可能有主力资金
        if volume_std > 0 and volume_mean > 0:
            vol_cv = volume_std / volume_mean  # 成交量变异系数
            if vol_cv > 0.5:  # 成交量波动较大
                # 判断方向
                price_change = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
                if price_change > 0:
                    return min(100, 60 + vol_cv * 30)
                else:
                    return max(0, 40 - vol_cv * 30)

        return 50.0

    def calculate_sentiment_index(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> float:
        """
        计算市场情绪指标

        综合多个维度评估市场情绪

        Args:
            df: 股票数据DataFrame
            external_data: 外部数据

        Returns:
            float: 情绪指数 (0-100)
        """
        return self._calculate_sentiment_score(df, external_data)

    def _calculate_sentiment_score(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> float:
        """
        计算市场情绪评分

        基于技术指标组合评估市场情绪
        """
        if len(df) < 20:
            return 50.0

        scores = []

        # 1. 价格位置（相对20日高低）
        high_20 = df['high'].tail(20).max()
        low_20 = df['low'].tail(20).min()
        close = df['close'].iloc[-1]

        if high_20 > low_20:
            position = (close - low_20) / (high_20 - low_20)
            # 中间位置情绪稳定，极端位置情绪极端
            position_score = 50 + (position - 0.5) * 40
            scores.append(position_score)

        # 2. 涨跌天数比例
        up_days = (df['close'].tail(10).diff() > 0).sum()
        up_ratio = up_days / 10
        up_score = up_ratio * 100
        scores.append(up_score)

        # 3. 成交量趋势
        vol_ma5 = df['volume'].tail(5).mean()
        vol_ma20 = df['volume'].tail(20).mean()
        if vol_ma20 > 0:
            vol_ratio = vol_ma5 / vol_ma20
            vol_score = min(100, max(0, 50 + (vol_ratio - 1) * 100))
            scores.append(vol_score)

        # 4. 使用外部情绪数据（如果有）
        if external_data and 'market_sentiment' in external_data:
            market_sentiment = external_data['market_sentiment']
            if isinstance(market_sentiment, (int, float)):
                scores.append(market_sentiment)

        return np.mean(scores) if scores else 50.0

    def _calculate_confidence(
        self,
        df: pd.DataFrame,
        external_data: Optional[Dict] = None
    ) -> float:
        """
        计算因子置信度

        数据越完整，置信度越高
        """
        confidence = 0.5  # 基础置信度

        # 有足够历史数据
        if len(df) >= 60:
            confidence += 0.1
        elif len(df) >= 20:
            confidence += 0.05

        # 有外部数据
        if external_data:
            if 'north_money_flow' in external_data:
                confidence += 0.1
            if 'margin_balance' in external_data:
                confidence += 0.1
            if 'block_trade' in external_data:
                confidence += 0.1
            if 'market_sentiment' in external_data:
                confidence += 0.05

        return min(1.0, confidence)

    def _create_empty_result(self) -> EmotionFactorResult:
        """创建空结果"""
        return EmotionFactorResult(
            north_money_flow=50.0,
            margin_balance_change=50.0,
            block_trade_net=50.0,
            sentiment_index=50.0,
            composite_score=50.0,
            confidence=0.0
        )


def calculate_emotion_factors(
    df: pd.DataFrame,
    external_data: Optional[Dict] = None
) -> Dict:
    """
    便捷函数：计算情绪因子

    Args:
        df: 股票数据DataFrame
        external_data: 外部数据字典

    Returns:
        Dict: 情绪因子结果
    """
    calculator = EmotionFactorCalculator()
    result = calculator.calculate_all_factors(df, external_data)

    return {
        'north_money_flow': result.north_money_flow,
        'margin_balance_change': result.margin_balance_change,
        'block_trade_net': result.block_trade_net,
        'sentiment_index': result.sentiment_index,
        'composite_score': result.composite_score,
        'confidence': result.confidence
    }