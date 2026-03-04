"""
趋势动量策略模块

基于趋势动量评分系统生成交易信号

核心逻辑：抓住趋势启动点，而非等待形态确认

评分构成:
- 动量突破信号 (40分): 5/10/20日动量
- 均线系统 (30分): MA5/10/20/60排列
- 量能确认 (20分): 量比、价涨量增
- 位置判断 (10分): 60日位置
- 突破确认 (加分): 突破新高
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..domain.interfaces.strategy import IStrategy
from ..factors.trend_momentum_scoring import TrendMomentumScoring, TrendMomentumResult
from ..core.logging import get_logger


logger = get_logger(__name__)


class TrendMomentumStrategy(IStrategy):
    """
    趋势动量策略：抓住趋势启动点

    特点：
    1. 评分达到阈值生成买入信号
    2. 内置止损止盈逻辑
    3. 量价配合确认趋势
    """

    def __init__(
        self,
        buy_threshold: float = 55.0,
        sell_threshold: float = 40.0,
        stop_loss_pct: float = 0.07,
        take_profit_pct: float = 0.15,
    ):
        """
        初始化趋势动量策略

        Args:
            buy_threshold: 买入阈值 (默认55)
            sell_threshold: 卖出阈值 (默认40)
            stop_loss_pct: 止损比例 (默认7%)
            take_profit_pct: 止盈比例 (默认15%)
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        # 初始化评分系统
        self.scoring_system = TrendMomentumScoring(
            buy_threshold=buy_threshold,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
        )

        # 策略参数
        self.parameters = {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct
        }

        # 信号历史
        self.signals_history: List[Dict] = []

        # 最后一次评分结果
        self.last_score_result: Optional[TrendMomentumResult] = None

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """初始化策略参数"""
        self.parameters.update(parameters)
        self.buy_threshold = self.parameters.get('buy_threshold', 55.0)
        self.sell_threshold = self.parameters.get('sell_threshold', 40.0)
        self.stop_loss_pct = self.parameters.get('stop_loss_pct', 0.07)
        self.take_profit_pct = self.parameters.get('take_profit_pct', 0.15)

        # 更新评分系统参数
        self.scoring_system = TrendMomentumScoring(
            buy_threshold=self.buy_threshold,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
        )

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
        result['momentum_score'] = np.nan
        result['ma_score'] = np.nan
        result['volume_score'] = np.nan
        result['position_score'] = np.nan
        result['breakout_score'] = np.nan
        result['signal'] = 'hold'
        result['stop_loss'] = np.nan
        result['take_profit'] = np.nan

        # 逐日计算评分
        for i in range(min_bars, len(bars)):
            window_data = bars.iloc[:i+1]
            try:
                score_result = self.scoring_system.calculate_score(window_data)

                result.loc[result.index[i], 'final_score'] = score_result.final_score
                result.loc[result.index[i], 'momentum_score'] = score_result.momentum_score
                result.loc[result.index[i], 'ma_score'] = score_result.ma_score
                result.loc[result.index[i], 'volume_score'] = score_result.volume_score
                result.loc[result.index[i], 'position_score'] = score_result.position_score
                result.loc[result.index[i], 'breakout_score'] = score_result.breakout_score
                result.loc[result.index[i], 'stop_loss'] = score_result.stop_loss
                result.loc[result.index[i], 'take_profit'] = score_result.take_profit

                # 生成信号
                final_score = score_result.final_score

                if final_score >= self.buy_threshold:
                    result.loc[result.index[i], 'signal'] = 'buy'
                elif final_score <= self.sell_threshold:
                    result.loc[result.index[i], 'signal'] = 'sell'

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
                'reason': '数据不足(需60日)'
            }

        try:
            # 计算趋势动量评分
            score_result = self.scoring_system.calculate_score(historical_bars)
            self.last_score_result = score_result

            final_score = score_result.final_score

            # 确定方向
            direction = None
            signal_type = 'hold'

            if final_score >= self.buy_threshold:
                direction = 'buy'
                signal_type = 'buy'
            elif final_score <= self.sell_threshold:
                direction = 'sell'
                signal_type = 'sell'

            signal = {
                'direction': direction,
                'signal': signal_type,
                'score': final_score,
                'momentum_score': score_result.momentum_score,
                'ma_score': score_result.ma_score,
                'volume_score': score_result.volume_score,
                'position_score': score_result.position_score,
                'breakout_score': score_result.breakout_score,
                'stop_loss': score_result.stop_loss,
                'take_profit': score_result.take_profit,
                'signals': score_result.signals,
                'details': score_result.details,
                'strategy_name': 'TrendMomentumStrategy',
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

    def get_name(self) -> str:
        """获取策略名称"""
        return "TrendMomentumStrategy"

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.parameters.copy()

    def get_description(self) -> str:
        """获取策略描述"""
        return (
            f"趋势动量策略: 买入阈值={self.buy_threshold}, "
            f"卖出阈值={self.sell_threshold}, "
            f"止损={self.stop_loss_pct*100}%, "
            f"止盈={self.take_profit_pct*100}%"
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
        }

    def get_last_score_breakdown(self) -> Dict:
        """获取最后一次评分明细"""
        if self.last_score_result is None:
            return {}

        return {
            'final_score': self.last_score_result.final_score,
            'momentum_score': self.last_score_result.momentum_score,
            'ma_score': self.last_score_result.ma_score,
            'volume_score': self.last_score_result.volume_score,
            'position_score': self.last_score_result.position_score,
            'breakout_score': self.last_score_result.breakout_score,
            'stop_loss': self.last_score_result.stop_loss,
            'take_profit': self.last_score_result.take_profit,
            'signals': self.last_score_result.signals,
            'details': self.last_score_result.details
        }