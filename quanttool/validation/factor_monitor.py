"""
因子有效性监控模块

监控因子表现，检测因子衰减：
- 滚动IC计算
- 因子衰减检测
- 权重调整建议
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FactorEffectivenessResult:
    """因子有效性结果"""
    factor_name: str
    current_ic: float
    rolling_ic_mean: float
    rolling_ic_std: float
    ic_trend: float  # IC趋势斜率
    decay_rate: float  # 衰减率
    effectiveness_score: float  # 有效性评分
    status: str  # 状态描述
    suggested_weight_adjustment: float  # 建议权重调整


@dataclass
class FactorMonitorReport:
    """因子监控报告"""
    timestamp: datetime
    factor_results: List[FactorEffectivenessResult]
    overall_effectiveness: float
    high_priority_factors: List[str]
    warning_factors: List[str]
    recommended_adjustments: Dict[str, float]


class FactorEffectivenessMonitor:
    """
    因子有效性监控器

    监控因子IC变化，检测衰减，提供调整建议
    """

    # IC阈值
    IC_THRESHOLDS = {
        'excellent': 0.10,   # 优秀
        'good': 0.05,        # 良好
        'normal': 0.03,      # 正常
        'weak': 0.01,        # 微弱
        'negative': 0.0,     # 无效/反向
    }

    # 衰减阈值
    DECAY_THRESHOLDS = {
        'no_decay': 0.0,      # 无衰减
        'mild': 0.1,          # 轻微衰减
        'moderate': 0.2,      # 中度衰减
        'severe': 0.3,        # 严重衰减
    }

    def __init__(
        self,
        rolling_window: int = 60,
        decay_window: int = 20,
        min_samples: int = 30,
        ic_significance_level: float = 0.05
    ):
        """
        初始化因子监控器

        Args:
            rolling_window: 滚动IC计算窗口
            decay_window: 衰减检测窗口
            min_samples: 最小样本数
            ic_significance_level: IC显著性水平
        """
        self.rolling_window = rolling_window
        self.decay_window = decay_window
        self.min_samples = min_samples
        self.ic_significance_level = ic_significance_level

        # 历史记录
        self.ic_history: Dict[str, List[float]] = {}
        self.monitoring_history: List[FactorMonitorReport] = []

    def calculate_rolling_ic(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        计算滚动IC

        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            window: 滚动窗口

        Returns:
            pd.Series: 滚动IC序列
        """
        window = window or self.rolling_window

        # 对齐数据
        common_index = factor_values.dropna().index.intersection(returns.dropna().index)
        if len(common_index) < self.min_samples:
            return pd.Series(dtype=float)

        factor_aligned = factor_values.loc[common_index]
        returns_aligned = returns.loc[common_index]

        # 计算滚动IC
        ic_values = []
        dates = []

        for i in range(window, len(factor_aligned)):
            window_factor = factor_aligned.iloc[i-window:i]
            window_returns = returns_aligned.iloc[i-window:i]

            if len(window_factor) >= self.min_samples:
                ic, _ = stats.spearmanr(window_factor.values, window_returns.values)
                ic_values.append(ic)
                dates.append(factor_aligned.index[i])

        return pd.Series(ic_values, index=dates)

    def detect_factor_decay(
        self,
        ic_series: pd.Series
    ) -> Dict:
        """
        检测因子衰减

        分析IC趋势，判断因子是否衰减

        Args:
            ic_series: IC序列

        Returns:
            Dict: 衰减检测结果
        """
        if len(ic_series) < self.decay_window:
            return {
                'has_decay': False,
                'decay_rate': 0.0,
                'trend': 0.0,
                'status': 'insufficient_data'
            }

        # 计算IC趋势（线性回归斜率）
        x = np.arange(len(ic_series))
        y = ic_series.values

        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            trend = slope
        except Exception:
            trend = 0.0

        # 计算衰减率（近期IC vs 历史IC）
        recent_ic = ic_series.tail(self.decay_window).mean()
        historical_ic = ic_series.head(len(ic_series) - self.decay_window).mean()

        if abs(historical_ic) > 1e-6:
            decay_rate = (historical_ic - recent_ic) / abs(historical_ic)
        else:
            decay_rate = 0.0

        # 判断状态
        has_decay = decay_rate > self.DECAY_THRESHOLDS['moderate'] or trend < -0.001

        if decay_rate > self.DECAY_THRESHOLDS['severe']:
            status = 'severe_decay'
        elif decay_rate > self.DECAY_THRESHOLDS['moderate']:
            status = 'moderate_decay'
        elif decay_rate > self.DECAY_THRESHOLDS['mild']:
            status = 'mild_decay'
        else:
            status = 'healthy'

        return {
            'has_decay': has_decay,
            'decay_rate': decay_rate,
            'trend': trend,
            'recent_ic': recent_ic,
            'historical_ic': historical_ic,
            'status': status
        }

    def suggest_weight_adjustment(
        self,
        current_weight: float,
        ic_history: pd.Series,
        decay_info: Dict
    ) -> float:
        """
        建议权重调整

        根据IC历史和衰减情况建议权重调整

        Args:
            current_weight: 当前权重
            ic_history: IC历史
            decay_info: 衰减信息

        Returns:
            float: 建议调整量（正=增加，负=减少）
        """
        # 基础调整量
        adjustment = 0.0

        # 根据IC水平调整
        recent_ic = decay_info.get('recent_ic', 0)

        if recent_ic > self.IC_THRESHOLDS['excellent']:
            adjustment += 0.1  # IC优秀，增加权重
        elif recent_ic > self.IC_THRESHOLDS['good']:
            adjustment += 0.05
        elif recent_ic > self.IC_THRESHOLDS['normal']:
            adjustment += 0.0
        elif recent_ic > self.IC_THRESHOLDS['weak']:
            adjustment -= 0.05
        else:
            adjustment -= 0.1  # IC为负，减少权重

        # 根据衰减调整
        if decay_info['has_decay']:
            decay_rate = decay_info['decay_rate']
            if decay_rate > self.DECAY_THRESHOLDS['severe']:
                adjustment -= 0.2  # 严重衰减，大幅减少
            elif decay_rate > self.DECAY_THRESHOLDS['moderate']:
                adjustment -= 0.1
            else:
                adjustment -= 0.05

        # 根据趋势调整
        trend = decay_info.get('trend', 0)
        if trend < -0.001:  # IC下降趋势
            adjustment -= 0.05
        elif trend > 0.001:  # IC上升趋势
            adjustment += 0.03

        return adjustment

    def evaluate_factor(
        self,
        factor_name: str,
        factor_values: pd.Series,
        returns: pd.Series,
        current_weight: float
    ) -> FactorEffectivenessResult:
        """
        评估单个因子有效性

        Args:
            factor_name: 因子名称
            factor_values: 因子值
            returns: 收益率
            current_weight: 当前权重

        Returns:
            FactorEffectivenessResult: 评估结果
        """
        # 计算滚动IC
        rolling_ic = self.calculate_rolling_ic(factor_values, returns)

        if rolling_ic.empty:
            return FactorEffectivenessResult(
                factor_name=factor_name,
                current_ic=0.0,
                rolling_ic_mean=0.0,
                rolling_ic_std=0.0,
                ic_trend=0.0,
                decay_rate=0.0,
                effectiveness_score=50.0,
                status='insufficient_data',
                suggested_weight_adjustment=0.0
            )

        # 计算IC统计
        current_ic = rolling_ic.iloc[-1]
        ic_mean = rolling_ic.mean()
        ic_std = rolling_ic.std()

        # 检测衰减
        decay_info = self.detect_factor_decay(rolling_ic)

        # 计算有效性评分
        effectiveness_score = self._calculate_effectiveness_score(
            ic_mean, ic_std, decay_info
        )

        # 建议权重调整
        suggested_adjustment = self.suggest_weight_adjustment(
            current_weight, rolling_ic, decay_info
        )

        # 记录IC历史
        if factor_name not in self.ic_history:
            self.ic_history[factor_name] = []
        self.ic_history[factor_name].extend(rolling_ic.tolist())

        return FactorEffectivenessResult(
            factor_name=factor_name,
            current_ic=current_ic,
            rolling_ic_mean=ic_mean,
            rolling_ic_std=ic_std,
            ic_trend=decay_info['trend'],
            decay_rate=decay_info['decay_rate'],
            effectiveness_score=effectiveness_score,
            status=decay_info['status'],
            suggested_weight_adjustment=suggested_adjustment
        )

    def _calculate_effectiveness_score(
        self,
        ic_mean: float,
        ic_std: float,
        decay_info: Dict
    ) -> float:
        """
        计算因子有效性评分

        评分 = IC均值贡献 + IC稳定性贡献 - 衰减惩罚
        """
        # IC均值贡献（-50到50）
        ic_score = ic_mean * 500

        # IC稳定性贡献（标准差越小越好）
        stability_score = max(0, 20 - ic_std * 100)

        # 衰减惩罚
        decay_penalty = decay_info['decay_rate'] * 50

        # 综合评分
        score = 50 + ic_score + stability_score - decay_penalty

        return max(0, min(100, score))

    def generate_monitor_report(
        self,
        factor_data: Dict[str, pd.Series],
        returns: pd.Series,
        current_weights: Dict[str, float]
    ) -> FactorMonitorReport:
        """
        生成因子监控报告

        Args:
            factor_data: 因子数据字典
            returns: 收益率序列
            current_weights: 当前权重字典

        Returns:
            FactorMonitorReport: 监控报告
        """
        factor_results = []
        high_priority_factors = []
        warning_factors = []
        recommended_adjustments = {}

        for factor_name, factor_values in factor_data.items():
            current_weight = current_weights.get(factor_name, 1.0 / len(factor_data))

            # 评估因子
            result = self.evaluate_factor(
                factor_name, factor_values, returns, current_weight
            )
            factor_results.append(result)

            # 识别优先因子
            if result.effectiveness_score >= 70:
                high_priority_factors.append(factor_name)

            # 识别警告因子
            if result.status in ['severe_decay', 'moderate_decay'] or result.current_ic < 0:
                warning_factors.append(factor_name)

            # 记录建议调整
            if result.suggested_weight_adjustment != 0:
                recommended_adjustments[factor_name] = result.suggested_weight_adjustment

        # 计算整体有效性
        overall_effectiveness = np.mean([r.effectiveness_score for r in factor_results])

        report = FactorMonitorReport(
            timestamp=datetime.now(),
            factor_results=factor_results,
            overall_effectiveness=overall_effectiveness,
            high_priority_factors=high_priority_factors,
            warning_factors=warning_factors,
            recommended_adjustments=recommended_adjustments
        )

        self.monitoring_history.append(report)

        return report

    def get_factor_ranking(self) -> List[Tuple[str, float]]:
        """
        获取因子排名（按有效性评分）
        """
        if not self.ic_history:
            return []

        rankings = []
        for factor_name, ic_values in self.ic_history.items():
            if ic_values:
                recent_ic = np.mean(ic_values[-20:]) if len(ic_values) >= 20 else np.mean(ic_values)
                rankings.append((factor_name, recent_ic))

        return sorted(rankings, key=lambda x: x[1], reverse=True)

    def get_decay_summary(self) -> Dict:
        """
        获取衰减摘要
        """
        if not self.monitoring_history:
            return {}

        latest_report = self.monitoring_history[-1]

        return {
            'warning_factors': latest_report.warning_factors,
            'high_priority_factors': latest_report.high_priority_factors,
            'overall_effectiveness': latest_report.overall_effectiveness,
            'recommended_adjustments': latest_report.recommended_adjustments
        }


def monitor_factor_effectiveness(
    factor_data: Dict[str, pd.Series],
    returns: pd.Series,
    current_weights: Dict[str, float]
) -> Dict:
    """
    便捷函数：监控因子有效性

    Args:
        factor_data: 因子数据
        returns: 收益率
        current_weights: 当前权重

    Returns:
        Dict: 监控结果
    """
    monitor = FactorEffectivenessMonitor()
    report = monitor.generate_monitor_report(factor_data, returns, current_weights)

    return {
        'overall_effectiveness': report.overall_effectiveness,
        'warning_factors': report.warning_factors,
        'high_priority_factors': report.high_priority_factors,
        'recommended_adjustments': report.recommended_adjustments,
        'factor_details': [
            {
                'name': r.factor_name,
                'ic': r.current_ic,
                'effectiveness': r.effectiveness_score,
                'status': r.status,
                'weight_adjustment': r.suggested_weight_adjustment
            }
            for r in report.factor_results
        ]
    }