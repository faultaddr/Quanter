"""
信号归因分析模块

分析盈利和亏损信号的关键因子：
- 盈利信号归因
- 亏损信号归因
- 因子贡献分析
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SignalAttribution:
    """信号归因结果"""
    signal_id: str
    direction: str
    entry_date: datetime
    exit_date: Optional[datetime]
    return_pct: float
    winning: bool
    factor_contributions: Dict[str, float]
    top_positive_factors: List[str]
    top_negative_factors: List[str]


@dataclass
class AttributionReport:
    """归因报告"""
    total_signals: int
    winning_signals: int
    losing_signals: int
    win_rate: float
    avg_winning_return: float
    avg_losing_return: float
    profitable_factor_ranking: List[Tuple[str, float]]
    losing_factor_ranking: List[Tuple[str, float]]
    factor_effectiveness: Dict[str, float]
    recommendations: List[str]


class SignalAttributor:
    """
    信号归因分析器

    分析交易信号盈亏的关键因子
    """

    def __init__(
        self,
        min_signals: int = 10,
        factor_names: Optional[List[str]] = None
    ):
        """
        初始化信号归因器

        Args:
            min_signals: 最小信号数
            factor_names: 因子名称列表
        """
        self.min_signals = min_signals
        self.factor_names = factor_names or [
            'trend_strength', 'ma_slope', 'macd_momentum',
            'money_flow', 'volume_ratio', 'kdj_position',
            'rsi_strength', 'mtm_momentum', 'roc_rate',
            'obv_flow', 'mfi_strength', 'volume_price'
        ]

        # 信号记录
        self.signals: List[Dict] = []
        self.attributions: List[SignalAttribution] = []

    def record_signal(
        self,
        signal_id: str,
        direction: str,
        entry_date: datetime,
        factor_scores: Dict[str, float],
        entry_price: float
    ):
        """
        记录信号

        Args:
            signal_id: 信号ID
            direction: 方向
            entry_date: 入场日期
            factor_scores: 因子评分
            entry_price: 入场价格
        """
        self.signals.append({
            'signal_id': signal_id,
            'direction': direction,
            'entry_date': entry_date,
            'factor_scores': factor_scores,
            'entry_price': entry_price,
            'exit_date': None,
            'exit_price': None,
            'return_pct': None
        })

    def update_signal_result(
        self,
        signal_id: str,
        exit_date: datetime,
        exit_price: float
    ):
        """
        更新信号结果

        Args:
            signal_id: 信号ID
            exit_date: 出场日期
            exit_price: 出场价格
        """
        for signal in self.signals:
            if signal['signal_id'] == signal_id:
                signal['exit_date'] = exit_date
                signal['exit_price'] = exit_price
                signal['return_pct'] = (exit_price - signal['entry_price']) / signal['entry_price']
                break

    def analyze_profitable_signals(
        self,
        trade_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        分析盈利信号的关键因子

        Args:
            trade_history: 交易历史（可选）

        Returns:
            Dict: 盈利信号分析结果
        """
        signals = trade_history or self.signals
        profitable = [s for s in signals if s.get('return_pct', 0) > 0]

        if len(profitable) < self.min_signals:
            return {'status': 'insufficient_data', 'count': len(profitable)}

        # 统计各因子在盈利信号中的平均得分
        factor_scores = defaultdict(list)

        for signal in profitable:
            for factor, score in signal.get('factor_scores', {}).items():
                factor_scores[factor].append(score)

        # 计算平均得分
        avg_scores = {
            factor: np.mean(scores)
            for factor, scores in factor_scores.items()
        }

        # 排序
        sorted_factors = sorted(
            avg_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'status': 'success',
            'count': len(profitable),
            'avg_return': np.mean([s['return_pct'] for s in profitable]),
            'factor_avg_scores': avg_scores,
            'top_factors': sorted_factors[:5],
            'factor_ranking': sorted_factors
        }

    def analyze_losing_signals(
        self,
        trade_history: Optional[List[Dict]] = None
    ) -> Dict:
        """
        分析亏损信号的问题因子

        Args:
            trade_history: 交易历史（可选）

        Returns:
            Dict: 亏损信号分析结果
        """
        signals = trade_history or self.signals
        losing = [s for s in signals if s.get('return_pct', 0) <= 0]

        if len(losing) < self.min_signals:
            return {'status': 'insufficient_data', 'count': len(losing)}

        # 统计各因子在亏损信号中的平均得分
        factor_scores = defaultdict(list)

        for signal in losing:
            for factor, score in signal.get('factor_scores', {}).items():
                factor_scores[factor].append(score)

        # 计算平均得分
        avg_scores = {
            factor: np.mean(scores)
            for factor, scores in factor_scores.items()
        }

        # 找出问题因子（得分低但信号仍买入）
        problem_factors = [
            (factor, score)
            for factor, score in avg_scores.items()
            if score < 50
        ]

        sorted_problem = sorted(
            problem_factors,
            key=lambda x: x[1]
        )

        return {
            'status': 'success',
            'count': len(losing),
            'avg_return': np.mean([s['return_pct'] for s in losing]),
            'factor_avg_scores': avg_scores,
            'problem_factors': sorted_problem,
            'recommendations': self._generate_losing_recommendations(sorted_problem)
        }

    def _generate_losing_recommendations(
        self,
        problem_factors: List[Tuple[str, float]]
    ) -> List[str]:
        """生成亏损信号改进建议"""
        recommendations = []

        for factor, score in problem_factors[:3]:
            if factor in ['rsi_strength', 'kdj_position']:
                recommendations.append(
                    f"关注{factor}：当前平均得分{score:.1f}，"
                    "避免在超买区域买入"
                )
            elif factor in ['macd_momentum', 'mtm_momentum']:
                recommendations.append(
                    f"关注{factor}：当前平均得分{score:.1f}，"
                    "等待动量确认后再入场"
                )
            elif factor in ['money_flow', 'obv_flow']:
                recommendations.append(
                    f"关注{factor}：当前平均得分{score:.1f}，"
                    "确保资金流向支持后再入场"
                )
            else:
                recommendations.append(
                    f"因子{factor}平均得分{score:.1f}，建议提高该因子权重或增加过滤条件"
                )

        return recommendations

    def generate_attribution_report(
        self,
        trade_history: Optional[List[Dict]] = None
    ) -> AttributionReport:
        """
        生成归因报告

        Args:
            trade_history: 交易历史

        Returns:
            AttributionReport: 归因报告
        """
        signals = trade_history or self.signals

        if len(signals) < self.min_signals:
            return AttributionReport(
                total_signals=len(signals),
                winning_signals=0,
                losing_signals=0,
                win_rate=0,
                avg_winning_return=0,
                avg_losing_return=0,
                profitable_factor_ranking=[],
                losing_factor_ranking=[],
                factor_effectiveness={},
                recommendations=['数据不足，无法生成归因报告']
            )

        # 分类信号
        winning = [s for s in signals if s.get('return_pct', 0) > 0]
        losing = [s for s in signals if s.get('return_pct', 0) <= 0]

        # 分析盈利信号
        profitable_analysis = self.analyze_profitable_signals(signals)

        # 分析亏损信号
        losing_analysis = self.analyze_losing_signals(signals)

        # 计算因子有效性
        factor_effectiveness = self._calculate_factor_effectiveness(
            profitable_analysis, losing_analysis
        )

        # 生成建议
        recommendations = self._generate_recommendations(
            profitable_analysis, losing_analysis, factor_effectiveness
        )

        return AttributionReport(
            total_signals=len(signals),
            winning_signals=len(winning),
            losing_signals=len(losing),
            win_rate=len(winning) / len(signals) if signals else 0,
            avg_winning_return=np.mean([s['return_pct'] for s in winning]) if winning else 0,
            avg_losing_return=np.mean([s['return_pct'] for s in losing]) if losing else 0,
            profitable_factor_ranking=profitable_analysis.get('factor_ranking', []),
            losing_factor_ranking=losing_analysis.get('problem_factors', []),
            factor_effectiveness=factor_effectiveness,
            recommendations=recommendations
        )

    def _calculate_factor_effectiveness(
        self,
        profitable_analysis: Dict,
        losing_analysis: Dict
    ) -> Dict[str, float]:
        """
        计算因子有效性

        比较因子在盈利和亏损信号中的得分差异
        """
        effectiveness = {}

        profit_scores = profitable_analysis.get('factor_avg_scores', {})
        losing_scores = losing_analysis.get('factor_avg_scores', {})

        for factor in self.factor_names:
            p_score = profit_scores.get(factor, 50)
            l_score = losing_scores.get(factor, 50)

            # 差异越大，因子区分度越高
            diff = p_score - l_score
            effectiveness[factor] = diff

        return effectiveness

    def _generate_recommendations(
        self,
        profitable_analysis: Dict,
        losing_analysis: Dict,
        factor_effectiveness: Dict[str, float]
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []

        # 基于因子有效性
        sorted_effectiveness = sorted(
            factor_effectiveness.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 高有效性因子
        high_effectiveness = [f for f, e in sorted_effectiveness if e > 10]
        if high_effectiveness:
            recommendations.append(
                f"高区分度因子: {', '.join(high_effectiveness[:3])}，建议增加权重"
            )

        # 低有效性因子
        low_effectiveness = [f for f, e in sorted_effectiveness if e < 0]
        if low_effectiveness:
            recommendations.append(
                f"低区分度因子: {', '.join(low_effectiveness[:3])}，建议降低权重或剔除"
            )

        # 添加亏损分析建议
        recommendations.extend(losing_analysis.get('recommendations', []))

        return recommendations[:5]  # 最多5条建议


def analyze_signal_attribution(
    trade_history: List[Dict]
) -> Dict:
    """
    便捷函数：分析信号归因

    Args:
        trade_history: 交易历史

    Returns:
        Dict: 归因分析结果
    """
    attributor = SignalAttributor()
    report = attributor.generate_attribution_report(trade_history)

    return {
        'total_signals': report.total_signals,
        'win_rate': report.win_rate,
        'avg_winning_return': report.avg_winning_return,
        'avg_losing_return': report.avg_losing_return,
        'factor_effectiveness': report.factor_effectiveness,
        'recommendations': report.recommendations
    }