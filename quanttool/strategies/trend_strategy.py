"""
趋势策略模块

基于趋势评分系统生成交易信号

核心逻辑：
- 评分 >= buy_threshold → buy
- 评分 <= sell_threshold → sell
- 同时返回时机系数用于仓位控制
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..domain.interfaces.strategy import IStrategy
from ..factors.trend_scoring_system import TrendScoringSystem, TrendScoreResult
from ..core.logging import get_logger


logger = get_logger(__name__)


class TrendStrategy(IStrategy):
    """
    趋势策略：基于趋势评分生成信号

    特点：
    1. 纯趋势强度评分，不再使用位置惩罚
    2. 时机系数用于风险控制和仓位管理
    3. 强势股得高分，不会被过度惩罚
    """

    def __init__(
        self,
        buy_threshold: float = 75.0,
        sell_threshold: float = 50.0,
        use_timing_filter: bool = True,
        position_by_timing: bool = True
    ):
        """
        初始化趋势策略

        Args:
            buy_threshold: 买入评分阈值（默认75）
            sell_threshold: 卖出评分阈值（默认50）
            use_timing_filter: 是否使用时机系数过滤
            position_by_timing: 是否根据时机系数调整仓位
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_timing_filter = use_timing_filter
        self.position_by_timing = position_by_timing

        # 初始化评分系统
        self.scoring_system = TrendScoringSystem()

        # 策略参数
        self.parameters = {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'use_timing_filter': use_timing_filter,
            'position_by_timing': position_by_timing
        }

        # 信号历史
        self.signals_history: List[Dict] = []

        # 最后一次评分结果
        self.last_score_result: Optional[TrendScoreResult] = None

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """初始化策略参数"""
        self.parameters.update(parameters)
        self.buy_threshold = self.parameters.get('buy_threshold', 75.0)
        self.sell_threshold = self.parameters.get('sell_threshold', 50.0)
        self.use_timing_filter = self.parameters.get('use_timing_filter', True)
        self.position_by_timing = self.parameters.get('position_by_timing', True)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        计算信号

        Args:
            bars: K线数据

        Returns:
            DataFrame: 包含信号列的数据
        """
        result = bars.copy()

        # 需要至少60天数据
        min_bars = 60

        # 初始化列
        result['final_score'] = np.nan
        result['trend_score'] = np.nan
        result['timing_coefficient'] = np.nan
        result['ma_score'] = np.nan
        result['momentum_score'] = np.nan
        result['volume_score'] = np.nan
        result['rs_score'] = np.nan
        result['signal'] = 'hold'
        result['position_ratio'] = 0.0

        # 逐日计算评分
        for i in range(min_bars, len(bars)):
            window_data = bars.iloc[:i+1]
            try:
                score_result = self.scoring_system.calculate_score(window_data)

                result.loc[result.index[i], 'final_score'] = score_result.final_score
                result.loc[result.index[i], 'trend_score'] = score_result.trend_total_score
                result.loc[result.index[i], 'timing_coefficient'] = score_result.timing_coefficient
                result.loc[result.index[i], 'ma_score'] = score_result.ma_structure_score
                result.loc[result.index[i], 'momentum_score'] = score_result.price_momentum_score
                result.loc[result.index[i], 'volume_score'] = score_result.volume_score
                result.loc[result.index[i], 'rs_score'] = score_result.relative_strength_score

                # 生成信号
                if score_result.passed_hard_filter:
                    final_score = score_result.final_score

                    if final_score >= self.buy_threshold:
                        result.loc[result.index[i], 'signal'] = 'buy'
                        # 根据时机系数调整仓位
                        if self.position_by_timing:
                            position = self._calculate_position_ratio(
                                final_score,
                                score_result.timing_coefficient
                            )
                            result.loc[result.index[i], 'position_ratio'] = position
                        else:
                            result.loc[result.index[i], 'position_ratio'] = 1.0

                    elif final_score <= self.sell_threshold:
                        result.loc[result.index[i], 'signal'] = 'sell'
                        result.loc[result.index[i], 'position_ratio'] = 0.0
                    else:
                        result.loc[result.index[i], 'position_ratio'] = 0.5  # 持仓观望

            except Exception as e:
                logger.warning(f"评分计算失败: {e}")
                continue

        return result

    def get_signal(
        self,
        current_bar: pd.Series,
        historical_bars: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        获取交易信号

        Args:
            current_bar: 当前K线
            historical_bars: 历史K线

        Returns:
            Dict: 信号字典
        """
        if len(historical_bars) < 60:
            return {
                'direction': None,
                'signal': 'hold',
                'reason': '数据不足'
            }

        try:
            # 计算趋势评分
            score_result = self.scoring_system.calculate_score(historical_bars)
            self.last_score_result = score_result

            # 未通过硬过滤
            if not score_result.passed_hard_filter:
                signal = {
                    'direction': None,
                    'signal': 'hold',
                    'score': 0,
                    'reason': score_result.hard_filter_reason,
                    'strategy_name': 'TrendStrategy',
                    'timestamp': current_bar.get('timestamp', datetime.now())
                }
                self.signals_history.append(signal)
                return signal

            final_score = score_result.final_score

            # 使用时机系数过滤
            if self.use_timing_filter:
                timing_coef = score_result.timing_coefficient
                # 时机系数过低时，降低评分
                if timing_coef < 0.8:
                    final_score = final_score * timing_coef

            # 确定方向
            direction = None
            signal_type = 'hold'
            position_ratio = 0.0

            if final_score >= self.buy_threshold:
                direction = 'buy'
                signal_type = 'buy'
                position_ratio = self._calculate_position_ratio(
                    final_score,
                    score_result.timing_coefficient
                ) if self.position_by_timing else 1.0

            elif final_score <= self.sell_threshold:
                direction = 'sell'
                signal_type = 'sell'

            signal = {
                'direction': direction,
                'signal': signal_type,
                'score': score_result.final_score,
                'trend_score': score_result.trend_total_score,
                'timing_coefficient': score_result.timing_coefficient,
                'timing_type': score_result.timing_type,
                'position_ratio': position_ratio,
                'ma_score': score_result.ma_structure_score,
                'momentum_score': score_result.price_momentum_score,
                'volume_score': score_result.volume_score,
                'rs_score': score_result.relative_strength_score,
                'details': score_result.details,
                'strategy_name': 'TrendStrategy',
                'timestamp': current_bar.get('timestamp', datetime.now())
            }

            self.signals_history.append(signal)
            return signal

        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return {
                'direction': None,
                'signal': 'hold',
                'error': str(e)
            }

    def _calculate_position_ratio(self, score: float, timing_coef: float) -> float:
        """
        计算仓位比例

        基于评分和时机系数计算建议仓位

        Args:
            score: 最终评分
            timing_coef: 时机系数

        Returns:
            float: 仓位比例 (0-1)
        """
        # 基础仓位基于评分
        if score >= 90:
            base_position = 1.0
        elif score >= 80:
            base_position = 0.8
        elif score >= 75:
            base_position = 0.6
        else:
            base_position = 0.4

        # 根据时机系数调整
        # 时机系数 1.2 -> 满仓
        # 时机系数 1.0 -> 标准仓位
        # 时机系数 0.8 -> 减半仓位
        # 时机系数 0.7 -> 1/4仓位
        timing_multiplier = (timing_coef - 0.6) / 0.6  # 归一化到 0-1
        timing_multiplier = max(0.25, min(1.0, timing_multiplier))

        return round(base_position * timing_multiplier, 2)

    def get_name(self) -> str:
        """获取策略名称"""
        return "TrendStrategy"

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.parameters.copy()

    def get_description(self) -> str:
        """获取策略描述"""
        return (
            f"趋势策略: 买入阈值={self.buy_threshold}, "
            f"卖出阈值={self.sell_threshold}, "
            f"时机过滤={'启用' if self.use_timing_filter else '禁用'}, "
            f"仓位调整={'启用' if self.position_by_timing else '禁用'}"
        )

    def get_signal_statistics(self) -> Dict:
        """获取信号统计"""
        if not self.signals_history:
            return {}

        buy_signals = [s for s in self.signals_history if s.get('direction') == 'buy']
        sell_signals = [s for s in self.signals_history if s.get('direction') == 'sell']

        return {
            'total_signals': len(self.signals_history),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            'avg_buy_score': np.mean([s.get('score', 0) for s in buy_signals]) if buy_signals else 0,
            'avg_sell_score': np.mean([s.get('score', 0) for s in sell_signals]) if sell_signals else 0,
            'avg_timing_coefficient': np.mean([s.get('timing_coefficient', 1) for s in buy_signals]) if buy_signals else 1
        }

    def get_last_score_breakdown(self) -> Dict:
        """获取最后一次评分明细"""
        if self.last_score_result is None:
            return {}

        return {
            'final_score': self.last_score_result.final_score,
            'trend_score': self.last_score_result.trend_total_score,
            'timing_coefficient': self.last_score_result.timing_coefficient,
            'ma_structure': self.last_score_result.ma_structure_score,
            'price_momentum': self.last_score_result.price_momentum_score,
            'volume': self.last_score_result.volume_score,
            'relative_strength': self.last_score_result.relative_strength_score,
            'timing_type': self.last_score_result.timing_type,
            'passed_filter': self.last_score_result.passed_hard_filter,
            'details': self.last_score_result.details
        }


class AdaptiveTrendStrategy(TrendStrategy):
    """
    自适应趋势策略

    根据市场环境动态调整参数
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 市场环境参数
        self.market_regime = 'normal'
        self.volatility_level = 'medium'

        # 动态阈值
        self.dynamic_buy_threshold = self.buy_threshold
        self.dynamic_sell_threshold = self.sell_threshold

    def update_market_regime(self, index_data: pd.DataFrame) -> str:
        """
        根据指数数据更新市场环境

        Args:
            index_data: 指数数据

        Returns:
            str: 市场环境类型
        """
        if len(index_data) < 20:
            return self.market_regime

        close = index_data['close'].values

        # 计算指数MA趋势
        ma20 = np.mean(close[-20:])
        ma50 = np.mean(close[-50:]) if len(close) >= 50 else ma20

        # 计算波动率
        returns = np.diff(close[-20:]) / close[-21:-1]
        volatility = np.std(returns) * np.sqrt(252)

        # 判断市场环境
        if ma20 > ma50 * 1.02:
            self.market_regime = 'bull'
            self.dynamic_buy_threshold = self.buy_threshold - 5  # 牛市降低买入门槛
            self.dynamic_sell_threshold = self.sell_threshold - 5
        elif ma20 < ma50 * 0.98:
            self.market_regime = 'bear'
            self.dynamic_buy_threshold = self.buy_threshold + 10  # 熊市提高买入门槛
            self.dynamic_sell_threshold = self.sell_threshold + 5
        else:
            self.market_regime = 'normal'
            self.dynamic_buy_threshold = self.buy_threshold
            self.dynamic_sell_threshold = self.sell_threshold

        # 判断波动率水平
        if volatility > 0.25:
            self.volatility_level = 'high'
            self.dynamic_buy_threshold += 5  # 高波动时提高门槛
        elif volatility < 0.15:
            self.volatility_level = 'low'
        else:
            self.volatility_level = 'medium'

        return self.market_regime

    def get_signal(
        self,
        current_bar: pd.Series,
        historical_bars: pd.DataFrame
    ) -> Dict[str, Any]:
        """获取信号（使用动态阈值）"""
        # 临时替换阈值
        original_buy = self.buy_threshold
        original_sell = self.sell_threshold

        self.buy_threshold = self.dynamic_buy_threshold
        self.sell_threshold = self.dynamic_sell_threshold

        signal = super().get_signal(current_bar, historical_bars)

        # 恢复阈值
        self.buy_threshold = original_buy
        self.sell_threshold = original_sell

        # 添加市场环境信息
        signal['market_regime'] = self.market_regime
        signal['volatility_level'] = self.volatility_level
        signal['dynamic_buy_threshold'] = self.dynamic_buy_threshold

        return signal

    def get_description(self) -> str:
        """获取策略描述"""
        return (
            f"自适应趋势策略: 买入阈值={self.dynamic_buy_threshold}, "
            f"卖出阈值={self.dynamic_sell_threshold}, "
            f"市场环境={self.market_regime}, "
            f"波动率={self.volatility_level}"
        )