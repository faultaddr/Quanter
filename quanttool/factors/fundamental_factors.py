"""
基本面因子模块

实现估值和财务质量相关的因子：
- PE分位数
- PB分位数
- ROE趋势
- 利润增长率
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FundamentalFactorResult:
    """基本面因子计算结果"""
    pe_percentile: float          # PE分位数
    pb_percentile: float          # PB分位数
    roe_trend: float              # ROE趋势
    profit_growth: float          # 利润增长率
    valuation_score: float        # 估值评分
    quality_score: float          # 质量评分
    composite_score: float        # 综合评分
    confidence: float             # 置信度


class FundamentalFactorCalculator:
    """
    基本面因子计算器

    计算估值和财务质量相关的因子
    """

    # 基本面因子权重
    FUNDAMENTAL_FACTOR_WEIGHTS = {
        'pe_percentile': 0.25,     # PE分位数权重
        'pb_percentile': 0.20,     # PB分位数权重
        'roe_trend': 0.30,         # ROE趋势权重
        'profit_growth': 0.25,     # 利润增长权重
    }

    # 估值区间阈值
    VALUATION_THRESHOLDS = {
        'pe_low': 15,              # PE低估阈值
        'pe_high': 40,             # PE高估阈值
        'pb_low': 1.0,             # PB低估阈值
        'pb_high': 4.0,            # PB高估阈值
    }

    def __init__(
        self,
        lookback_years: int = 3,
        use_industry_adjusted: bool = True
    ):
        """
        初始化基本面因子计算器

        Args:
            lookback_years: 历史数据回看年数
            use_industry_adjusted: 是否使用行业调整
        """
        self.lookback_years = lookback_years
        self.use_industry_adjusted = use_industry_adjusted

    def calculate_all_factors(
        self,
        df: pd.DataFrame,
        fundamental_data: Optional[Dict] = None
    ) -> FundamentalFactorResult:
        """
        计算所有基本面因子

        Args:
            df: 股票数据DataFrame
            fundamental_data: 基本面数据字典

        Returns:
            FundamentalFactorResult: 因子计算结果
        """
        # 计算各因子
        pe_score = self.calculate_pe_percentile(df, fundamental_data)
        pb_score = self.calculate_pb_percentile(df, fundamental_data)
        roe_score = self.calculate_roe_trend(df, fundamental_data)
        profit_score = self.calculate_profit_growth(df, fundamental_data)

        # 计算估值评分（低估值高分）
        valuation_score = (
            pe_score * self.FUNDAMENTAL_FACTOR_WEIGHTS['pe_percentile'] +
            pb_score * self.FUNDAMENTAL_FACTOR_WEIGHTS['pb_percentile']
        ) / (self.FUNDAMENTAL_FACTOR_WEIGHTS['pe_percentile'] +
             self.FUNDAMENTAL_FACTOR_WEIGHTS['pb_percentile'])

        # 计算质量评分（高质量高分）
        quality_score = (
            roe_score * self.FUNDAMENTAL_FACTOR_WEIGHTS['roe_trend'] +
            profit_score * self.FUNDAMENTAL_FACTOR_WEIGHTS['profit_growth']
        ) / (self.FUNDAMENTAL_FACTOR_WEIGHTS['roe_trend'] +
             self.FUNDAMENTAL_FACTOR_WEIGHTS['profit_growth'])

        # 计算综合评分
        composite_score = (
            pe_score * self.FUNDAMENTAL_FACTOR_WEIGHTS['pe_percentile'] +
            pb_score * self.FUNDAMENTAL_FACTOR_WEIGHTS['pb_percentile'] +
            roe_score * self.FUNDAMENTAL_FACTOR_WEIGHTS['roe_trend'] +
            profit_score * self.FUNDAMENTAL_FACTOR_WEIGHTS['profit_growth']
        )

        # 计算置信度
        confidence = self._calculate_confidence(fundamental_data)

        return FundamentalFactorResult(
            pe_percentile=pe_score,
            pb_percentile=pb_score,
            roe_trend=roe_score,
            profit_growth=profit_score,
            valuation_score=valuation_score,
            quality_score=quality_score,
            composite_score=composite_score,
            confidence=confidence
        )

    def calculate_pe_percentile(
        self,
        df: pd.DataFrame,
        fundamental_data: Optional[Dict] = None
    ) -> float:
        """
        计算PE分位数因子

        PE分位数低表示当前估值相对历史较低，具有安全边际

        Args:
            df: 股票数据DataFrame
            fundamental_data: 基本面数据

        Returns:
            float: PE评分 (0-100)
        """
        if fundamental_data and 'pe_history' in fundamental_data:
            pe_history = fundamental_data['pe_history']
            current_pe = fundamental_data.get('pe_current', 0)

            if isinstance(pe_history, (list, np.ndarray)) and len(pe_history) > 0:
                pe_array = np.array(pe_history)
                # 计算当前PE在历史中的分位数
                percentile = np.mean(pe_array <= current_pe)
                # 分位数越低越好（估值越便宜）
                # 转换为评分：低分位数 = 高分
                score = (1 - percentile) * 100
                return score

        # 无外部数据时，使用价格位置估算
        if len(df) < 60:
            return 50.0

        # 使用价格相对位置估算估值水平
        close = df['close'].iloc[-1]
        ma60 = df['close'].rolling(60).mean().iloc[-1]
        ma120 = df['close'].rolling(120).mean().iloc[-1] if len(df) >= 120 else ma60

        # 价格低于长期均线 = 可能低估
        if ma120 > 0:
            price_position = (close - ma120) / ma120
            if price_position < -0.2:
                return 80.0  # 可能低估
            elif price_position < -0.1:
                return 65.0
            elif price_position < 0:
                return 55.0
            elif price_position < 0.1:
                return 45.0
            elif price_position < 0.2:
                return 35.0
            else:
                return 25.0  # 可能高估

        return 50.0

    def calculate_pb_percentile(
        self,
        df: pd.DataFrame,
        fundamental_data: Optional[Dict] = None
    ) -> float:
        """
        计算PB分位数因子

        PB分位数低表示当前估值相对历史较低

        Args:
            df: 股票数据DataFrame
            fundamental_data: 基本面数据

        Returns:
            float: PB评分 (0-100)
        """
        if fundamental_data and 'pb_history' in fundamental_data:
            pb_history = fundamental_data['pb_history']
            current_pb = fundamental_data.get('pb_current', 0)

            if isinstance(pb_history, (list, np.ndarray)) and len(pb_history) > 0:
                pb_array = np.array(pb_history)
                percentile = np.mean(pb_array <= current_pb)
                score = (1 - percentile) * 100
                return score

        # 无外部数据时，使用市净率替代指标
        # 用账面价值增长率估算
        if fundamental_data and 'book_value_growth' in fundamental_data:
            bv_growth = fundamental_data['book_value_growth']
            if bv_growth > 0.1:
                return 70.0
            elif bv_growth > 0.05:
                return 55.0
            elif bv_growth > 0:
                return 45.0
            else:
                return 30.0

        return 50.0

    def calculate_roe_trend(
        self,
        df: pd.DataFrame,
        fundamental_data: Optional[Dict] = None
    ) -> float:
        """
        计算ROE趋势因子

        ROE上升表示公司盈利能力增强

        Args:
            df: 股票数据DataFrame
            fundamental_data: 基本面数据

        Returns:
            float: ROE趋势评分 (0-100)
        """
        if fundamental_data and 'roe_history' in fundamental_data:
            roe_history = fundamental_data['roe_history']

            if isinstance(roe_history, (list, np.ndarray)) and len(roe_history) >= 4:
                roe_array = np.array(roe_history[-4:])  # 最近4个季度

                # 计算ROE趋势
                if len(roe_array) >= 2:
                    recent_roe = np.mean(roe_array[-2:])
                    previous_roe = np.mean(roe_array[:-2]) if len(roe_array) > 2 else roe_array[0]

                    # ROE趋势向上
                    if recent_roe > previous_roe:
                        improvement = (recent_roe - previous_roe) / previous_roe if previous_roe > 0 else 0
                        return min(100, 60 + improvement * 200)
                    elif recent_roe == previous_roe:
                        return 50.0
                    else:
                        decline = (previous_roe - recent_roe) / previous_roe if previous_roe > 0 else 0
                        return max(0, 50 - decline * 200)

        # 无外部数据时，使用净利润率变化估算
        if fundamental_data and 'net_margin_history' in fundamental_data:
            margin_history = fundamental_data['net_margin_history']
            if isinstance(margin_history, (list, np.ndarray)) and len(margin_history) >= 2:
                current_margin = margin_history[-1]
                previous_margin = margin_history[-2]
                if previous_margin > 0:
                    margin_change = (current_margin - previous_margin) / previous_margin
                    return min(100, max(0, 50 + margin_change * 300))

        return 50.0

    def calculate_profit_growth(
        self,
        df: pd.DataFrame,
        fundamental_data: Optional[Dict] = None
    ) -> float:
        """
        计算利润增长率因子

        利润增长是股价上涨的核心驱动

        Args:
            df: 股票数据DataFrame
            fundamental_data: 基本面数据

        Returns:
            float: 利润增长评分 (0-100)
        """
        if fundamental_data and 'profit_growth' in fundamental_data:
            growth_rate = fundamental_data['profit_growth']

            # 将增长率转换为评分
            if growth_rate > 0.3:  # 30%以上增长
                return 90.0
            elif growth_rate > 0.2:
                return 80.0
            elif growth_rate > 0.1:
                return 70.0
            elif growth_rate > 0.05:
                return 60.0
            elif growth_rate > 0:
                return 55.0
            elif growth_rate > -0.1:
                return 40.0
            elif growth_rate > -0.2:
                return 30.0
            else:
                return 20.0

        # 无外部数据时，使用营收增长估算
        if fundamental_data and 'revenue_growth' in fundamental_data:
            rev_growth = fundamental_data['revenue_growth']
            # 营收增长通常与利润增长正相关
            return min(100, max(0, 50 + rev_growth * 150))

        return 50.0

    def _calculate_confidence(
        self,
        fundamental_data: Optional[Dict] = None
    ) -> float:
        """
        计算因子置信度
        """
        if not fundamental_data:
            return 0.3

        confidence = 0.3

        required_fields = [
            'pe_history', 'pb_history', 'roe_history',
            'profit_growth', 'revenue_growth'
        ]

        for field in required_fields:
            if field in fundamental_data:
                confidence += 0.14

        return min(1.0, confidence)

    def get_valuation_assessment(
        self,
        pe_score: float,
        pb_score: float
    ) -> str:
        """
        获取估值评估描述

        Args:
            pe_score: PE评分
            pb_score: PB评分

        Returns:
            str: 估值评估描述
        """
        avg_score = (pe_score + pb_score) / 2

        if avg_score >= 70:
            return "低估：具有较高安全边际，适合左侧布局"
        elif avg_score >= 55:
            return "合理偏低：估值较为合理，可适当配置"
        elif avg_score >= 45:
            return "合理：估值处于正常区间"
        elif avg_score >= 30:
            return "合理偏高：估值偏高，注意回调风险"
        else:
            return "高估：估值偏高，建议谨慎"

    def get_quality_assessment(
        self,
        roe_score: float,
        profit_score: float
    ) -> str:
        """
        获取质量评估描述

        Args:
            roe_score: ROE评分
            profit_score: 利润增长评分

        Returns:
            str: 质量评估描述
        """
        avg_score = (roe_score + profit_score) / 2

        if avg_score >= 70:
            return "优质：盈利能力强且持续增长"
        elif avg_score >= 55:
            return "良好：盈利能力稳定"
        elif avg_score >= 45:
            return "一般：盈利能力一般"
        elif avg_score >= 30:
            return "偏弱：盈利能力有所下降"
        else:
            return "较弱：盈利能力下滑明显"


def calculate_fundamental_factors(
    df: pd.DataFrame,
    fundamental_data: Optional[Dict] = None
) -> Dict:
    """
    便捷函数：计算基本面因子

    Args:
        df: 股票数据DataFrame
        fundamental_data: 基本面数据字典

    Returns:
        Dict: 基本面因子结果
    """
    calculator = FundamentalFactorCalculator()
    result = calculator.calculate_all_factors(df, fundamental_data)

    return {
        'pe_percentile': result.pe_percentile,
        'pb_percentile': result.pb_percentile,
        'roe_trend': result.roe_trend,
        'profit_growth': result.profit_growth,
        'valuation_score': result.valuation_score,
        'quality_score': result.quality_score,
        'composite_score': result.composite_score,
        'confidence': result.confidence
    }