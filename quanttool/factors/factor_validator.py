"""
因子有效性检验模块

实现因子有效性分析：
- IC (Information Coefficient) 分析
- IR (Information Ratio) 计算
- IC衰减分析
- 分层回测
- 因子收益归因
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import spearmanr, pearsonr


@dataclass
class ICResult:
    """IC分析结果"""
    factor_name: str
    ic_series: pd.Series              # IC时间序列
    mean_ic: float                    # 平均IC
    std_ic: float                     # IC标准差
    ir: float                         # 信息比率 (IC均值/IC标准差)
    ic_tstat: float                   # IC的t统计量
    ic_pvalue: float                  # IC的p值
    positive_ic_ratio: float          # IC为正的比率
    rank_ic_mean: float               # 平均Rank IC
    rank_ic_ir: float                 # Rank IC的IR


@dataclass
class DecayResult:
    """IC衰减分析结果"""
    factor_name: str
    decay_ic: Dict[int, float]        # 不同持有期的IC
    optimal_horizon: int              # 最优持有期
    half_life: int                    # 半衰期（IC衰减到一半的持有期）


@dataclass
class QuantileResult:
    """Quantile backtest result"""
    factor_name: str
    num_groups: int
    group_returns: Dict[int, float]
    group_stds: Dict[int, float]
    long_short_return: float
    long_short_sharpe: float
    top_minus_bottom: float
    hit_rate: float
    ic_correlation: float


@dataclass
class FactorValidationReport:
    """Factor validation report"""
    factor_name: str
    timestamp: datetime
    ic_result: Optional[ICResult] = None
    decay_result: Optional[DecayResult] = None
    quantile_result: Optional[QuantileResult] = None
    overall_score: float = 0.0
    is_effective: bool = False
    recommendations: List[str] = field(default_factory=list)


class FactorValidator:
    """
    因子有效性检验器

    提供完整的因子有效性分析功能
    """

    # IC有效性阈值
    IC_THRESHOLDS = {
        "excellent": 0.08,   # 优秀
        "good": 0.05,        # 良好
        "acceptable": 0.03,  # 可接受
        "weak": 0.01,        # 弱
    }

    # IR有效性阈值
    IR_THRESHOLDS = {
        "excellent": 0.8,
        "good": 0.5,
        "acceptable": 0.3,
        "weak": 0.2,
    }

    def __init__(
        self,
        min_periods: int = 20,
        ic_rolling_window: int = 60,
    ):
        """
        初始化因子有效性检验器

        Args:
            min_periods: 最小分析周期数
            ic_rolling_window: 滚动IC窗口
        """
        self.min_periods = min_periods
        self.ic_rolling_window = ic_rolling_window
        self._ic_cache: Dict[str, pd.Series] = {}

    def calculate_ic(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        method: str = "spearman"
    ) -> pd.Series:
        """
        计算滚动IC

        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            method: IC计算方法 (spearman, pearson)

        Returns:
            滚动IC序列
        """
        # 对齐数据
        common_idx = factor_values.dropna().index.intersection(returns.dropna().index)

        if len(common_idx) < self.min_periods:
            return pd.Series(dtype=float)

        factor_aligned = factor_values.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        ic_values = []
        dates = []

        for i in range(self.ic_rolling_window, len(factor_aligned)):
            window_factor = factor_aligned.iloc[i-self.ic_rolling_window:i]
            window_returns = returns_aligned.iloc[i-self.ic_rolling_window:i]

            # 去除nan
            valid_mask = window_factor.notna() & window_returns.notna()

            if valid_mask.sum() < 10:
                continue

            if method == "spearman":
                ic, _ = spearmanr(
                    window_factor[valid_mask].values,
                    window_returns[valid_mask].values
                )
            else:
                ic, _ = pearsonr(
                    window_factor[valid_mask].values,
                    window_returns[valid_mask].values
                )

            if not np.isnan(ic):
                ic_values.append(ic)
                dates.append(factor_aligned.index[i])

        return pd.Series(ic_values, index=dates)

    def analyze_ic(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        factor_name: str = "factor"
    ) -> ICResult:
        """
        分析IC指标

        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称

        Returns:
            ICResult: IC分析结果
        """
        ic_series = self.calculate_ic(factor_values, returns)

        if len(ic_series) < self.min_periods:
            return ICResult(
                factor_name=factor_name,
                ic_series=pd.Series(),
                mean_ic=0.0,
                std_ic=0.0,
                ir=0.0,
                ic_tstat=0.0,
                ic_pvalue=1.0,
                positive_ic_ratio=0.0,
                rank_ic_mean=0.0,
                rank_ic_ir=0.0,
            )

        # 计算IC统计
        mean_ic = ic_series.mean()
        std_ic = ic_series.std()
        ir = mean_ic / std_ic if std_ic > 0 else 0.0

        # t统计量
        ic_tstat = mean_ic / (std_ic / np.sqrt(len(ic_series))) if std_ic > 0 else 0.0
        ic_pvalue = 2 * (1 - stats.t.cdf(abs(ic_tstat), len(ic_series) - 1))

        # IC为正的比率
        positive_ratio = (ic_series > 0).sum() / len(ic_series)

        # Rank IC (使用Spearman就是Rank IC)
        rank_ic_series = self.calculate_ic(factor_values, returns, method="spearman")
        rank_ic_mean = rank_ic_series.mean() if len(rank_ic_series) > 0 else 0.0
        rank_ic_std = rank_ic_series.std() if len(rank_ic_series) > 0 else 1.0
        rank_ic_ir = rank_ic_mean / rank_ic_std if rank_ic_std > 0 else 0.0

        return ICResult(
            factor_name=factor_name,
            ic_series=ic_series,
            mean_ic=mean_ic,
            std_ic=std_ic,
            ir=ir,
            ic_tstat=ic_tstat,
            ic_pvalue=ic_pvalue,
            positive_ic_ratio=positive_ratio,
            rank_ic_mean=rank_ic_mean,
            rank_ic_ir=rank_ic_ir,
        )

    def analyze_decay(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        factor_name: str = "factor",
        max_horizon: int = 20
    ) -> DecayResult:
        """
        分析IC衰减

        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            max_horizon: 最大持有期

        Returns:
            DecayResult: 衰减分析结果
        """
        # 对齐数据
        common_idx = factor_values.dropna().index.intersection(returns.dropna().index)
        factor_aligned = factor_values.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        decay_ic = {}

        # 计算不同持有期的IC
        for horizon in range(1, max_horizon + 1):
            # 因子t期，收益t+horizon期
            factor_at_t = factor_aligned.iloc[:-horizon]
            returns_at_t_plus_horizon = returns_aligned.iloc[horizon:]

            # 对齐
            valid_idx = factor_at_t.index.intersection(returns_at_t_plus_horizon.index)

            if len(valid_idx) < self.min_periods:
                continue

            f = factor_at_t.loc[valid_idx]
            r = returns_at_t_plus_horizon.loc[valid_idx]

            valid_mask = f.notna() & r.notna()

            if valid_mask.sum() >= 10:
                ic, _ = spearmanr(f[valid_mask], r[valid_mask])
                if not np.isnan(ic):
                    decay_ic[horizon] = ic

        if not decay_ic:
            return DecayResult(
                factor_name=factor_name,
                decay_ic={},
                optimal_horizon=1,
                half_life=max_horizon,
            )

        # 找到最优持有期（IC最大的持有期）
        optimal_horizon = max(decay_ic.keys(), key=lambda k: abs(decay_ic[k]))

        # 计算半衰期（IC衰减到一半的持有期）
        initial_ic = decay_ic.get(1, 0)
        half_ic = abs(initial_ic) / 2

        half_life = max_horizon
        for h, ic in decay_ic.items():
            if abs(ic) <= half_ic:
                half_life = h
                break

        return DecayResult(
            factor_name=factor_name,
            decay_ic=decay_ic,
            optimal_horizon=optimal_horizon,
            half_life=half_life,
        )

    def run分层_backtest(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        factor_name: str = "factor",
        num_groups: int = 5,
        benchmark_return: Optional[pd.Series] = None
    ) -> QuantileResult:
        """
        运行分层回测

        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            num_groups: 分组数量
            benchmark_return: 基准收益率序列

        Returns:
            QuantileResult: 分层回测结果
        """
        # 对齐数据
        common_idx = factor_values.dropna().index.intersection(returns.dropna().index)
        factor_aligned = factor_values.loc[common_idx]
        returns_aligned = returns.loc[common_idx]

        if len(factor_aligned) < self.min_periods:
            return QuantileResult(
                factor_name=factor_name,
                num_groups=num_groups,
                group_returns={},
                group_stds={},
                long_short_return=0.0,
                long_short_sharpe=0.0,
                top_minus_bottom=0.0,
                hit_rate=0.0,
                ic_correlation=0.0,
            )

        # 分组
        factor_aligned_copy = factor_aligned.copy()
        factor_aligned_copy = factor_aligned_copy[factor_aligned_copy != 0]

        # 使用分位数分组
        factor_aligned_copy["group"] = pd.qcut(
            factor_aligned_copy,
            q=num_groups,
            labels=False,
            duplicates="drop"
        )

        group_returns = {}
        group_stds = {}

        for g in range(num_groups):
            group_mask = factor_aligned_copy["group"] == g
            if group_mask.sum() > 0:
                group_ret = returns_aligned[group_mask]
                group_returns[g] = group_ret.mean() * 252  # 年化收益
                group_stds[g] = group_ret.std() * np.sqrt(252)  # 年化波动

        if len(group_returns) < 2:
            return QuantileResult(
                factor_name=factor_name,
                num_groups=num_groups,
                group_returns=group_returns,
                group_stds=group_stds,
                long_short_return=0.0,
                long_short_sharpe=0.0,
                top_minus_bottom=0.0,
                hit_rate=0.0,
                ic_correlation=0.0,
            )

        # 多空组合（做多高因子组，做空低因子组）
        top_group = max(group_returns.keys(), key=lambda k: group_returns[k])
        bottom_group = min(group_returns.keys(), key=lambda k: group_returns[k])

        long_short_ret = group_returns.get(top_group, 0) - group_returns.get(bottom_group, 0)

        # 多空夏普比率
        combined_std = np.sqrt(group_stds.get(top_group, 0)**2 + group_stds.get(bottom_group, 0)**2)
        long_short_sharpe = long_short_ret / combined_std if combined_std > 0 else 0.0

        # 第一组减最后一组
        sorted_groups = sorted(group_returns.keys())
        top_minus_bottom = group_returns.get(sorted_groups[-1], 0) - group_returns.get(sorted_groups[0], 0)

        # 命中率（top组跑赢benchmark的比例）
        hit_rate = 0.5
        if benchmark_return is not None:
            top_returns = returns_aligned[factor_aligned_copy["group"] == top_group]
            hits = (top_returns > benchmark_return.reindex(top_returns.index)).sum()
            hit_rate = hits / len(top_returns) if len(top_returns) > 0 else 0.5

        # 与IC的相关性（因子值与收益的相关）
        ic_corr, _ = spearmanr(factor_aligned, returns_aligned)

        return QuantileResult(
            factor_name=factor_name,
            num_groups=num_groups,
            group_returns=group_returns,
            group_stds=group_stds,
            long_short_return=long_short_ret,
            long_short_sharpe=long_short_sharpe,
            top_minus_bottom=top_minus_bottom,
            hit_rate=hit_rate,
            ic_correlation=ic_corr,
        )

    def validate(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        factor_name: str = "factor",
        benchmark_return: Optional[pd.Series] = None
    ) -> FactorValidationReport:
        """
        完整的因子有效性检验

        Args:
            factor_values: 因子值序列
            returns: 收益率序列
            factor_name: 因子名称
            benchmark_return: 基准收益率序列

        Returns:
            FactorValidationReport: 检验报告
        """
        recommendations = []

        # IC分析
        ic_result = self.analyze_ic(factor_values, returns, factor_name)

        # IC衰减分析
        decay_result = self.analyze_decay(factor_values, returns, factor_name)

        # 分层回测
        quantile_result = self.run分层_backtest(
            factor_values, returns, factor_name, num_groups=5,
            benchmark_return=benchmark_return
        )

        # 计算综合评分
        score = 0.0

        # IC评分 (0-30分)
        if abs(ic_result.mean_ic) >= self.IC_THRESHOLDS["excellent"]:
            score += 30
            recommendations.append("IC优秀，考虑增加权重")
        elif abs(ic_result.mean_ic) >= self.IC_THRESHOLDS["good"]:
            score += 20
        elif abs(ic_result.mean_ic) >= self.IC_THRESHOLDS["acceptable"]:
            score += 10
        else:
            recommendations.append("IC较弱，考虑替换或与其他因子组合")

        # IR评分 (0-30分)
        if abs(ic_result.ir) >= self.IR_THRESHOLDS["excellent"]:
            score += 30
        elif abs(ic_result.ir) >= self.IR_THRESHOLDS["good"]:
            score += 20
        elif abs(ic_result.ir) >= self.IR_THRESHOLDS["acceptable"]:
            score += 10
            recommendations.append("IR可接受，但稳定性有待提升")
        else:
            recommendations.append("IR较低，因子稳定性不足")

        # 分层回测评分 (0-40分)
        if quantile_result.long_short_return > 0.2:
            score += 40
        elif quantile_result.long_short_return > 0.1:
            score += 30
        elif quantile_result.long_short_return > 0.05:
            score += 20
        elif quantile_result.long_short_return > 0:
            score += 10
        else:
            recommendations.append("分层回测收益为负，考虑反向因子")

        # 判断是否有效
        is_effective = score >= 40 and ic_result.mean_ic > 0

        # 添加更多建议
        if decay_result.half_life < 5:
            recommendations.append(f"IC半衰期较短({decay_result.half_life}期)，建议短期持有")
        elif decay_result.half_life > 15:
            recommendations.append(f"IC半衰期较长({decay_result.half_life}期)，适合中长期持有")

        return FactorValidationReport(
            factor_name=factor_name,
            timestamp=datetime.now(),
            ic_result=ic_result,
            decay_result=decay_result,
            quantile_result=quantile_result,
            overall_score=score,
            is_effective=is_effective,
            recommendations=recommendations,
        )


def validate_factor(
    factor_values: pd.Series,
    returns: pd.Series,
    factor_name: str = "factor",
    **kwargs
) -> FactorValidationReport:
    """
    便捷函数：快速检验因子有效性

    Args:
        factor_values: 因子值序列
        returns: 收益率序列
        factor_name: 因子名称
        **kwargs: 其他参数

    Returns:
        FactorValidationReport: 检验报告
    """
    validator = FactorValidator(**kwargs)
    return validator.validate(factor_values, returns, factor_name)
