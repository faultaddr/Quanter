"""
评分-收益验证系统

验证评分系统对未来收益的预测能力，计算关键指标：
- IC (Information Coefficient): 评分与收益的相关系数
- Rank IC: 秩相关系数
- IC IR: IC均值/IC标准差
- 分位数收益差: Top quintile - Bottom quintile
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ScoreValidationResult:
    """评分验证结果"""
    # IC 相关指标
    ic: float  # 信息系数（Pearson相关系数）
    rank_ic: float  # 秩相关系数（Spearman）
    ic_ir: float  # IC信息比率
    ic_mean: float  # IC均值
    ic_std: float  # IC标准差

    # 分位数分析
    quantile_returns: Dict[int, float]  # 各分位数平均收益
    top_bottom_spread: float  # Top-Bottom收益差

    # 胜率分析
    win_rate_by_score: Dict[str, float]  # 不同评分区间的胜率

    # 统计显著性
    t_statistic: float
    p_value: float

    # 元数据
    sample_size: int
    horizon_days: int
    timestamp: datetime


class ScoreValidator:
    """
    评分验证器

    验证评分系统对收益的预测能力
    """

    def __init__(self, min_samples: int = 30):
        """
        初始化验证器

        Args:
            min_samples: 最小样本数，低于此数量不计算统计量
        """
        self.min_samples = min_samples
        self.validation_history: List[ScoreValidationResult] = []

    def validate_score_correlation(
        self,
        scores: pd.Series,
        returns: pd.Series,
        horizon_days: int = 5
    ) -> ScoreValidationResult:
        """
        计算评分与收益的相关性

        Args:
            scores: 评分序列（索引为日期）
            returns: 收益序列（索引为日期，未来收益）
            horizon_days: 收益计算周期（天）

        Returns:
            ScoreValidationResult: 验证结果
        """
        # 对齐数据
        common_index = scores.index.intersection(returns.index)
        if len(common_index) < self.min_samples:
            return self._create_empty_result(len(common_index), horizon_days)

        scores_aligned = scores.loc[common_index]
        returns_aligned = returns.loc[common_index]

        # 计算 IC（Pearson相关系数）
        ic, ic_pvalue = stats.pearsonr(scores_aligned.values, returns_aligned.values)

        # 计算 Rank IC（Spearman秩相关系数）
        rank_ic, rank_pvalue = stats.spearmanr(scores_aligned.values, returns_aligned.values)

        # 计算 IC IR（需要滚动计算）
        ic_series = self._calculate_rolling_ic(scores_aligned, returns_aligned, window=20)
        ic_mean = ic_series.mean() if len(ic_series) > 0 else 0.0
        ic_std = ic_series.std() if len(ic_series) > 1 else 1.0
        ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

        # 分位数分析
        quantile_returns = self._calculate_quantile_returns(scores_aligned, returns_aligned, n_quantiles=5)
        top_bottom_spread = quantile_returns.get(4, 0) - quantile_returns.get(0, 0)

        # 按评分区间计算胜率
        win_rate_by_score = self._calculate_win_rate_by_score_range(scores_aligned, returns_aligned)

        # T检验
        t_statistic, p_value = stats.ttest_ind(
            returns_aligned[scores_aligned >= scores_aligned.median()],
            returns_aligned[scores_aligned < scores_aligned.median()]
        )

        result = ScoreValidationResult(
            ic=ic,
            rank_ic=rank_ic,
            ic_ir=ic_ir,
            ic_mean=ic_mean,
            ic_std=ic_std,
            quantile_returns=quantile_returns,
            top_bottom_spread=top_bottom_spread,
            win_rate_by_score=win_rate_by_score,
            t_statistic=t_statistic,
            p_value=p_value,
            sample_size=len(common_index),
            horizon_days=horizon_days,
            timestamp=datetime.now()
        )

        self.validation_history.append(result)
        return result

    def calculate_score_quantile_analysis(
        self,
        scores: pd.Series,
        returns: pd.Series,
        n_quantiles: int = 5
    ) -> pd.DataFrame:
        """
        按评分分位数分析收益分布

        Args:
            scores: 评分序列
            returns: 收益序列
            n_quantiles: 分位数数量

        Returns:
            DataFrame: 各分位数的统计分析
        """
        common_index = scores.index.intersection(returns.index)
        if len(common_index) < self.min_samples:
            return pd.DataFrame()

        scores_aligned = scores.loc[common_index]
        returns_aligned = returns.loc[common_index]

        # 计算分位数边界
        quantile_bounds = pd.qcut(scores_aligned, n_quantiles, labels=False, duplicates='drop')

        results = []
        for q in range(n_quantiles):
            mask = quantile_bounds == q
            if mask.sum() == 0:
                continue

            q_returns = returns_aligned[mask]

            results.append({
                'quantile': q + 1,
                'count': len(q_returns),
                'mean_return': q_returns.mean(),
                'std_return': q_returns.std(),
                'median_return': q_returns.median(),
                'win_rate': (q_returns > 0).mean(),
                'max_return': q_returns.max(),
                'min_return': q_returns.min(),
                'sharpe': q_returns.mean() / q_returns.std() if q_returns.std() > 0 else 0
            })

        return pd.DataFrame(results)

    def calculate_ic_decay(
        self,
        scores: pd.Series,
        returns_by_horizon: Dict[int, pd.Series],
        horizons: List[int] = [1, 3, 5, 10, 20]
    ) -> pd.DataFrame:
        """
        计算IC随持有周期的衰减

        Args:
            scores: 评分序列
            returns_by_horizon: 各周期的收益序列字典
            horizons: 周期列表

        Returns:
            DataFrame: 各周期的IC值
        """
        results = []
        for horizon in horizons:
            if horizon not in returns_by_horizon:
                continue

            returns = returns_by_horizon[horizon]
            result = self.validate_score_correlation(scores, returns, horizon)
            results.append({
                'horizon': horizon,
                'ic': result.ic,
                'rank_ic': result.rank_ic,
                'ic_ir': result.ic_ir,
                'sample_size': result.sample_size
            })

        return pd.DataFrame(results)

    def _calculate_rolling_ic(
        self,
        scores: pd.Series,
        returns: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """计算滚动IC"""
        if len(scores) < window:
            return pd.Series(dtype=float)

        ic_values = []
        dates = []

        for i in range(window, len(scores)):
            window_scores = scores.iloc[i-window:i]
            window_returns = returns.iloc[i-window:i]

            if len(window_scores) >= self.min_samples:
                ic, _ = stats.spearmanr(window_scores.values, window_returns.values)
                ic_values.append(ic)
                dates.append(scores.index[i])

        return pd.Series(ic_values, index=dates)

    def _calculate_quantile_returns(
        self,
        scores: pd.Series,
        returns: pd.Series,
        n_quantiles: int = 5
    ) -> Dict[int, float]:
        """计算各分位数平均收益"""
        try:
            quantile_bounds = pd.qcut(scores, n_quantiles, labels=False, duplicates='drop')
            quantile_returns = {}
            for q in range(n_quantiles):
                mask = quantile_bounds == q
                if mask.sum() > 0:
                    quantile_returns[q] = returns[mask].mean()
            return quantile_returns
        except Exception:
            return {}

    def _calculate_win_rate_by_score_range(
        self,
        scores: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """按评分区间计算胜率"""
        ranges = {
            '0-20': (0, 20),
            '20-40': (20, 40),
            '40-60': (40, 60),
            '60-80': (60, 80),
            '80-100': (80, 100)
        }

        win_rates = {}
        for name, (low, high) in ranges.items():
            mask = (scores >= low) & (scores < high)
            if mask.sum() > 0:
                win_rates[name] = (returns[mask] > 0).mean()

        return win_rates

    def _create_empty_result(self, sample_size: int, horizon_days: int) -> ScoreValidationResult:
        """创建空结果"""
        return ScoreValidationResult(
            ic=0.0,
            rank_ic=0.0,
            ic_ir=0.0,
            ic_mean=0.0,
            ic_std=0.0,
            quantile_returns={},
            top_bottom_spread=0.0,
            win_rate_by_score={},
            t_statistic=0.0,
            p_value=1.0,
            sample_size=sample_size,
            horizon_days=horizon_days,
            timestamp=datetime.now()
        )

    def get_validation_summary(self) -> Dict:
        """获取验证历史摘要"""
        if not self.validation_history:
            return {}

        ic_values = [r.ic for r in self.validation_history]
        rank_ic_values = [r.rank_ic for r in self.validation_history]

        return {
            'total_validations': len(self.validation_history),
            'ic_mean': np.mean(ic_values),
            'ic_std': np.std(ic_values),
            'rank_ic_mean': np.mean(rank_ic_values),
            'rank_ic_std': np.std(rank_ic_values),
            'positive_ic_rate': sum(1 for ic in ic_values if ic > 0) / len(ic_values),
            'last_validation': self.validation_history[-1].timestamp
        }


def validate_scoring_system(
    df: pd.DataFrame,
    score_column: str = 'final_score',
    return_horizon: int = 5
) -> ScoreValidationResult:
    """
    便捷函数：验证DataFrame中的评分系统

    Args:
        df: 包含评分和价格数据的DataFrame
        score_column: 评分列名
        return_horizon: 收益计算周期

    Returns:
        ScoreValidationResult: 验证结果
    """
    validator = ScoreValidator()

    # 计算未来收益
    df = df.copy()
    df['future_return'] = df['close'].pct_change(return_horizon).shift(-return_horizon)

    # 移除NaN
    df = df.dropna(subset=[score_column, 'future_return'])

    if len(df) < validator.min_samples:
        print(f"警告: 样本数不足 ({len(df)} < {validator.min_samples})")
        return validator._create_empty_result(len(df), return_horizon)

    scores = df.set_index('timestamp')[score_column] if 'timestamp' in df.columns else pd.Series(df[score_column].values)
    returns = df.set_index('timestamp')['future_return'] if 'timestamp' in df.columns else pd.Series(df['future_return'].values)

    return validator.validate_score_correlation(scores, returns, return_horizon)