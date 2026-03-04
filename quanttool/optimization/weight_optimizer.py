"""
动态因子权重优化模块

基于市场环境动态优化因子权重，实现：
- 市场状态识别（BULL/BEAR/SIDEWAY/VOLATILE）
- 滚动窗口优化权重
- 状态特定权重配置
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class MarketRegime(str, Enum):
    """市场状态"""
    BULL = "bull"       # 牛市：价格上涨，波动低
    BEAR = "bear"       # 熊市：价格下跌，波动高
    SIDEWAY = "sideway" # 震荡：价格横盘，波动低
    VOLATILE = "volatile" # 剧烈波动：大幅震荡


@dataclass
class WeightConfig:
    """权重配置"""
    trend: float      # 趋势因子权重
    momentum: float   # 动能因子权重
    money: float      # 资金因子权重

    def to_dict(self) -> Dict[str, float]:
        return {
            'trend': self.trend,
            'momentum': self.momentum,
            'money': self.money
        }

    def validate(self) -> bool:
        """验证权重和为1"""
        return abs(self.trend + self.momentum + self.money - 1.0) < 0.001


# 不同市场环境下的默认权重配置
DEFAULT_REGIME_WEIGHTS = {
    MarketRegime.BULL: WeightConfig(trend=0.50, momentum=0.30, money=0.20),
    MarketRegime.BEAR: WeightConfig(trend=0.30, momentum=0.25, money=0.45),
    MarketRegime.SIDEWAY: WeightConfig(trend=0.25, momentum=0.45, money=0.30),
    MarketRegime.VOLATILE: WeightConfig(trend=0.35, momentum=0.30, money=0.35),
}


class DynamicWeightOptimizer:
    """
    动态权重优化器

    基于市场环境动态优化因子权重
    """

    def __init__(
        self,
        lookback_period: int = 60,
        regime_weights: Optional[Dict[MarketRegime, WeightConfig]] = None
    ):
        """
        初始化权重优化器

        Args:
            lookback_period: 市场状态识别的回看周期
            regime_weights: 各市场状态的权重配置
        """
        self.lookback_period = lookback_period
        self.regime_weights = regime_weights or DEFAULT_REGIME_WEIGHTS
        self.current_regime: Optional[MarketRegime] = None
        self.regime_history: List[Tuple[datetime, MarketRegime]] = []

    def detect_market_regime(
        self,
        df: pd.DataFrame,
        price_column: str = 'close'
    ) -> MarketRegime:
        """
        识别市场状态

        Args:
            df: 价格数据DataFrame
            price_column: 价格列名

        Returns:
            MarketRegime: 识别出的市场状态
        """
        if len(df) < self.lookback_period:
            return MarketRegime.SIDEWAY

        # 取最近的回看周期数据
        recent_data = df.tail(self.lookback_period)
        prices = recent_data[price_column].values

        # 计算收益率
        returns = np.diff(prices) / prices[:-1]

        # 计算特征
        # 1. 价格趋势（收益率均值）
        trend = np.mean(returns)

        # 2. 波动率（收益率标准差）
        volatility = np.std(returns)

        # 3. 价格位置（相对于区间高低点的位置）
        price_position = (prices[-1] - np.min(prices)) / (np.max(prices) - np.min(prices))

        # 4. 方向一致性（上涨天数占比）
        up_days = np.sum(returns > 0) / len(returns)

        # 市场状态判定逻辑
        # 牛市：趋势向上、波动率低、价格位置高
        # 熊市：趋势向下、波动率高
        # 震荡：趋势接近零、波动率低
        # 剧烈波动：波动率高

        # 年化波动率阈值（日波动率 * sqrt(252)）
        annual_vol = volatility * np.sqrt(252)
        high_vol_threshold = 0.25  # 25%年化波动率

        # 趋势阈值（年化收益率）
        annual_trend = trend * 252
        bull_trend_threshold = 0.10  # 10%年化收益
        bear_trend_threshold = -0.10

        if annual_vol > high_vol_threshold:
            # 高波动市场
            if annual_trend > bull_trend_threshold:
                regime = MarketRegime.BULL  # 波动但上涨
            elif annual_trend < bear_trend_threshold:
                regime = MarketRegime.BEAR  # 波动且下跌
            else:
                regime = MarketRegime.VOLATILE  # 剧烈震荡
        else:
            # 低波动市场
            if annual_trend > bull_trend_threshold:
                regime = MarketRegime.BULL
            elif annual_trend < bear_trend_threshold:
                regime = MarketRegime.BEAR
            else:
                regime = MarketRegime.SIDEWAY

        self.current_regime = regime
        self.regime_history.append((datetime.now(), regime))

        return regime

    def optimize_weights(
        self,
        factor_returns: pd.DataFrame,
        method: str = 'rolling_ic'
    ) -> WeightConfig:
        """
        优化因子权重

        Args:
            factor_returns: 因子收益DataFrame，列为因子名，行为日期
            method: 优化方法 ('rolling_ic', 'equal', 'regime_based')

        Returns:
            WeightConfig: 优化后的权重配置
        """
        if method == 'equal':
            return WeightConfig(trend=1/3, momentum=1/3, money=1/3)

        if method == 'regime_based':
            if self.current_regime is None:
                raise ValueError("请先调用 detect_market_regime 识别市场状态")
            return self.regime_weights[self.current_regime]

        if method == 'rolling_ic':
            return self._optimize_by_rolling_ic(factor_returns)

        raise ValueError(f"未知的优化方法: {method}")

    def _optimize_by_rolling_ic(self, factor_returns: pd.DataFrame) -> WeightConfig:
        """基于滚动IC优化权重"""
        if len(factor_returns) < 20:
            return WeightConfig(trend=1/3, momentum=1/3, money=1/3)

        # 计算各因子的IC
        ic_values = {}
        for col in ['trend_return', 'momentum_return', 'money_return']:
            if col in factor_returns.columns:
                # 计算因子收益与总收益的相关性
                total_return = factor_returns.sum(axis=1)
                ic = factor_returns[col].rolling(20).corr(total_return)
                ic_values[col] = ic.iloc[-1] if not ic.empty else 0

        # 基于IC确定权重（IC绝对值越大，权重越高）
        total_ic = sum(abs(v) for v in ic_values.values())
        if total_ic == 0:
            return WeightConfig(trend=1/3, momentum=1/3, money=1/3)

        return WeightConfig(
            trend=abs(ic_values.get('trend_return', 0)) / total_ic,
            momentum=abs(ic_values.get('momentum_return', 0)) / total_ic,
            money=abs(ic_values.get('money_return', 0)) / total_ic
        )

    def get_current_weights(self) -> WeightConfig:
        """获取当前权重配置"""
        if self.current_regime is None:
            return WeightConfig(trend=0.40, momentum=0.35, money=0.25)
        return self.regime_weights[self.current_regime]

    def get_regime_statistics(self) -> Dict:
        """获取市场状态统计"""
        if not self.regime_history:
            return {}

        regime_counts = {}
        for _, regime in self.regime_history:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1

        total = len(self.regime_history)
        return {
            'total_observations': total,
            'regime_distribution': {k: v / total for k, v in regime_counts.items()},
            'current_regime': self.current_regime.value if self.current_regime else None
        }


def calculate_factor_returns(
    df: pd.DataFrame,
    score_breakdown: Dict[str, float],
    return_horizon: int = 5
) -> Dict[str, float]:
    """
    计算因子收益贡献

    Args:
        df: 价格数据
        score_breakdown: 各因子评分
        return_horizon: 收益计算周期

    Returns:
        Dict: 各因子的收益贡献
    """
    if len(df) < return_horizon:
        return {}

    future_return = df['close'].iloc[-1] / df['close'].iloc[-return_horizon-1] - 1

    total_score = sum(score_breakdown.values())
    if total_score == 0:
        return {}

    return {
        factor: (score / total_score) * future_return
        for factor, score in score_breakdown.items()
    }