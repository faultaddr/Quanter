"""
评分策略模块

基于评分系统生成交易信号：
- 评分>70 买入
- 评分<50 卖出
- 集成动态权重和风险控制
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..domain.interfaces.strategy import IStrategy
from ..factors.scoring_system import ScoringSystem
from ..optimization.weight_optimizer import DynamicWeightOptimizer, MarketRegime
from ..risk.risk_controller import RiskController, StopLossType
from ..analysis.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from ..core.logging import get_logger


logger = get_logger(__name__)


class ScoreStrategy(IStrategy):
    """
    基于评分的交易策略

    将评分系统转化为可回测的交易策略
    """

    def __init__(
        self,
        buy_threshold: float = 70.0,
        sell_threshold: float = 50.0,
        use_dynamic_weights: bool = True,
        use_multi_timeframe: bool = True,
        use_risk_control: bool = True
    ):
        """
        初始化评分策略

        Args:
            buy_threshold: 买入评分阈值
            sell_threshold: 卖出评分阈值
            use_dynamic_weights: 是否使用动态权重
            use_multi_timeframe: 是否使用多周期确认
            use_risk_control: 是否使用风险控制
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.use_dynamic_weights = use_dynamic_weights
        self.use_multi_timeframe = use_multi_timeframe
        self.use_risk_control = use_risk_control

        # 初始化组件
        self.scoring_system = ScoringSystem()
        self.weight_optimizer = DynamicWeightOptimizer() if use_dynamic_weights else None
        self.risk_controller = RiskController() if use_risk_control else None
        self.multi_timeframe_analyzer = MultiTimeframeAnalyzer(
            scoring_system=self.scoring_system
        ) if use_multi_timeframe else None

        # 策略状态
        self.parameters = {
            'buy_threshold': buy_threshold,
            'sell_threshold': sell_threshold,
            'use_dynamic_weights': use_dynamic_weights,
            'use_multi_timeframe': use_multi_timeframe,
            'use_risk_control': use_risk_control
        }

        # 交易记录
        self.signals_history: List[Dict] = []

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """初始化策略参数"""
        self.parameters.update(parameters)
        self.buy_threshold = self.parameters.get('buy_threshold', 70.0)
        self.sell_threshold = self.parameters.get('sell_threshold', 50.0)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """计算信号"""
        result = bars.copy()

        # 计算评分
        scores = []
        for i in range(len(bars)):
            if i < 30:  # 需要足够的历史数据
                scores.append({'final_score': 50, 'trend_score': 50, 'momentum_score': 50, 'money_score': 50})
            else:
                window_data = bars.iloc[:i+1]
                try:
                    score_result = self.scoring_system.calculate_comprehensive_score(window_data)
                    scores.append(score_result)
                except Exception as e:
                    logger.warning(f"评分计算失败: {e}")
                    scores.append({'final_score': 50, 'trend_score': 50, 'momentum_score': 50, 'money_score': 50})

        # 添加评分列
        result['final_score'] = [s['final_score'] for s in scores]
        result['trend_score'] = [s.get('trend_score', 50) for s in scores]
        result['momentum_score'] = [s.get('momentum_score', 50) for s in scores]
        result['money_score'] = [s.get('money_score', 50) for s in scores]

        # 生成信号
        result['signal'] = 'hold'
        result.loc[result['final_score'] >= self.buy_threshold, 'signal'] = 'buy'
        result.loc[result['final_score'] <= self.sell_threshold, 'signal'] = 'sell'

        # 信号强度
        result['signal_strength'] = (result['final_score'] - 50) / 50

        return result

    def get_signal(
        self,
        current_bar: pd.Series,
        historical_bars: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        获取交易信号

        Args:
            current_bar: 当前K线数据
            historical_bars: 历史K线数据

        Returns:
            Dict: 信号字典
        """
        if len(historical_bars) < 30:
            return {'direction': None, 'signal': 'hold'}

        try:
            # 计算评分
            score_result = self.scoring_system.calculate_comprehensive_score(historical_bars)
            final_score = score_result.get('final_score', 50)

            # 多周期确认
            mtf_bonus = 0.0
            if self.use_multi_timeframe and self.multi_timeframe_analyzer is not None:
                try:
                    mtf_result = self.multi_timeframe_analyzer.analyze_timeframe_alignment(historical_bars)
                    mtf_bonus = mtf_result.alignment_bonus
                except Exception:
                    pass

            # 调整后的评分
            adjusted_score = final_score * (1 + mtf_bonus)

            # 确定方向
            direction = None
            signal_type = 'hold'

            if adjusted_score >= self.buy_threshold:
                direction = 'buy'
                signal_type = 'buy'
            elif adjusted_score <= self.sell_threshold:
                direction = 'sell'
                signal_type = 'sell'

            # 计算止损位
            stop_loss = None
            if direction == 'buy' and self.use_risk_control and self.risk_controller is not None:
                try:
                    stop_result = self.risk_controller.calculate_dynamic_stop_loss(
                        historical_bars,
                        current_bar['close'],
                        signal_strength=adjusted_score / 100
                    )
                    stop_loss = stop_result.stop_price
                except Exception:
                    pass

            signal = {
                'direction': direction,
                'signal': signal_type,
                'score': final_score,
                'adjusted_score': adjusted_score,
                'mtf_bonus': mtf_bonus,
                'stop_loss': stop_loss,
                'strategy_name': 'ScoreStrategy',
                'timestamp': current_bar.get('timestamp', datetime.now())
            }

            self.signals_history.append(signal)

            return signal

        except Exception as e:
            logger.error(f"信号生成失败: {e}")
            return {'direction': None, 'signal': 'hold', 'error': str(e)}

    def get_name(self) -> str:
        """获取策略名称"""
        return "ScoreStrategy"

    def get_parameters(self) -> Dict[str, Any]:
        """获取策略参数"""
        return self.parameters.copy()

    def get_description(self) -> str:
        """获取策略描述"""
        return (
            f"评分策略: 买入阈值={self.buy_threshold}, "
            f"卖出阈值={self.sell_threshold}, "
            f"动态权重={'启用' if self.use_dynamic_weights else '禁用'}, "
            f"多周期确认={'启用' if self.use_multi_timeframe else '禁用'}, "
            f"风险控制={'启用' if self.use_risk_control else '禁用'}"
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
            'avg_sell_score': np.mean([s.get('score', 0) for s in sell_signals]) if sell_signals else 0
        }


class EnhancedScoreStrategy(ScoreStrategy):
    """
    增强版评分策略

    集成完整的验证、优化、风险控制功能
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 验证历史
        self.validation_results: List[Dict] = []

    def validate_signals(
        self,
        historical_data: pd.DataFrame,
        lookback: int = 250
    ) -> Dict:
        """
        验证历史信号表现

        Args:
            historical_data: 历史数据
            lookback: 回看期

        Returns:
            Dict: 验证结果
        """
        if len(historical_data) < lookback:
            return {'status': 'insufficient_data'}

        # 计算历史评分
        signals_df = self.calculate_signals(historical_data.tail(lookback))

        # 计算未来收益
        signals_df['future_return_5d'] = signals_df['close'].pct_change(5).shift(-5)
        signals_df['future_return_10d'] = signals_df['close'].pct_change(10).shift(-10)

        # 分析买入信号表现
        buy_signals = signals_df[signals_df['signal'] == 'buy']

        if len(buy_signals) == 0:
            return {'status': 'no_buy_signals'}

        results = {
            'status': 'success',
            'total_buy_signals': len(buy_signals),
            'win_rate_5d': (buy_signals['future_return_5d'] > 0).mean(),
            'avg_return_5d': buy_signals['future_return_5d'].mean(),
            'avg_return_10d': buy_signals['future_return_10d'].mean(),
            'avg_score': buy_signals['final_score'].mean()
        }

        self.validation_results.append(results)
        return results

    def optimize_parameters(
        self,
        historical_data: pd.DataFrame,
        metric: str = 'win_rate_5d'
    ) -> Dict:
        """
        优化策略参数

        Args:
            historical_data: 历史数据
            metric: 优化目标指标

        Returns:
            Dict: 优化结果
        """
        best_result = None
        best_params = None
        best_metric = -float('inf')

        # 参数搜索空间
        buy_thresholds = [60, 65, 70, 75, 80]
        sell_thresholds = [40, 45, 50, 55, 60]

        for buy_th in buy_thresholds:
            for sell_th in sell_thresholds:
                if buy_th <= sell_th:
                    continue

                # 临时设置参数
                original_buy = self.buy_threshold
                original_sell = self.sell_threshold

                self.buy_threshold = buy_th
                self.sell_threshold = sell_th

                # 验证
                result = self.validate_signals(historical_data, lookback=250)

                # 恢复参数
                self.buy_threshold = original_buy
                self.sell_threshold = original_sell

                if result.get('status') != 'success':
                    continue

                current_metric = result.get(metric, 0)

                if current_metric > best_metric:
                    best_metric = current_metric
                    best_result = result
                    best_params = {
                        'buy_threshold': buy_th,
                        'sell_threshold': sell_th
                    }

        return {
            'best_params': best_params,
            'best_metric': best_metric,
            'best_result': best_result
        }