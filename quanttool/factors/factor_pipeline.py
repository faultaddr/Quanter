"""
因子预处理流水线模块

实现因子预处理的标准流程：
- 去极值（Winsorization）
- 标准化（Z-Score）
- 中性化处理（行业、市值）
- 缺失值处理
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
import pandas as pd
import numpy as np
from scipy import stats


class NeutralizationType(str, Enum):
    """中性化类型"""
    NONE = "none"
    INDUSTRY = "industry"         # 行业中性化
    MARKET_CAP = "market_cap"     # 市值中性化
    INDUSTRY_MARKET_CAP = "industry_market_cap"  # 行业+市值中性化


class StandardizationMethod(str, Enum):
    """标准化方法"""
    ZSCORE = "zscore"            # Z-Score
    RANK = "rank"                # 排序标准化
    MINMAX = "minmax"            # Min-Max标准化


@dataclass
class PipelineConfig:
    """流水线配置"""
    # 去极值配置
    winsorize_lower: float = 0.01   # 下界百分位
    winsorize_upper: float = 0.99   # 上界百分位

    # 标准化配置
    standardization: StandardizationMethod = StandardizationMethod.ZSCORE

    # 中性化配置
    neutralization: NeutralizationType = NeutralizationType.NONE

    # 缺失值处理
    fill_method: str = "median"     # 填充方法: median, mean, forward, zero

    # 其他配置
    group_column: Optional[str] = None  # 分组列（如行业）


@dataclass
class FactorPipelineResult:
    """流水线处理结果"""
    data: pd.DataFrame
    factor_names: List[str]
    config: PipelineConfig
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class FactorPipeline:
    """
    因子预处理流水线

    标准化因子处理流程，确保因子数据质量
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        初始化因子流水线

        Args:
            config: 流水线配置
        """
        self.config = config or PipelineConfig()

    def process(
        self,
        data: pd.DataFrame,
        factor_columns: List[str],
        market_cap_column: Optional[str] = None,
        industry_column: Optional[str] = None,
    ) -> FactorPipelineResult:
        """
        处理因子数据

        Args:
            data: 输入数据（包含因子列）
            factor_columns: 因子列名列表
            market_cap_column: 市值列名（用于中性化）
            industry_column: 行业列名（用于中性化）

        Returns:
            FactorPipelineResult: 处理结果
        """
        result_data = data.copy()

        # 步骤1: 缺失值处理
        result_data = self._handle_missing_values(result_data, factor_columns)

        # 步骤2: 去极值
        result_data = self._winsorize(result_data, factor_columns)

        # 步骤3: 标准化
        result_data, stats_dict = self._standardize(result_data, factor_columns)

        # 步骤4: 中性化
        if self.config.neutralization != NeutralizationType.NONE:
            result_data = self._neutralize(
                result_data,
                factor_columns,
                market_cap_column,
                industry_column,
            )

        return FactorPipelineResult(
            data=result_data,
            factor_names=factor_columns,
            config=self.config,
            statistics=stats_dict,
        )

    def _handle_missing_values(
        self,
        data: pd.DataFrame,
        factor_columns: List[str]
    ) -> pd.DataFrame:
        """处理缺失值"""
        result = data.copy()

        for col in factor_columns:
            if col not in result.columns:
                continue

            if self.config.fill_method == "median":
                fill_value = result[col].median()
            elif self.config.fill_method == "mean":
                fill_value = result[col].mean()
            elif self.config.fill_method == "zero":
                fill_value = 0.0
            elif self.config.fill_method == "forward":
                # 前向填充
                result[col] = result[col].fillna(method="ffill").fillna(method="bfill")
                continue
            else:
                fill_value = result[col].median()

            result[col] = result[col].fillna(fill_value)

        return result

    def _winsorize(
        self,
        data: pd.DataFrame,
        factor_columns: List[str]
    ) -> pd.DataFrame:
        """去极值处理"""
        result = data.copy()

        for col in factor_columns:
            if col not in result.columns:
                continue

            # 使用百分位数去极值
            lower = result[col].quantile(self.config.winsorize_lower)
            upper = result[col].quantile(self.config.winsorize_upper)

            result[col] = result[col].clip(lower=lower, upper=upper)

        return result

    def _standardize(
        self,
        data: pd.DataFrame,
        factor_columns: List[str]
    ) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
        """标准化处理"""
        result = data.copy()
        stats_dict = {}

        for col in factor_columns:
            if col not in result.columns:
                continue

            col_stats = {
                "mean": result[col].mean(),
                "std": result[col].std(),
                "median": result[col].median(),
                "min": result[col].min(),
                "max": result[col].max(),
            }
            stats_dict[col] = col_stats

            if self.config.standardization == StandardizationMethod.ZSCORE:
                # Z-Score 标准化
                if col_stats["std"] > 1e-8:
                    result[col] = (result[col] - col_stats["mean"]) / col_stats["std"]
                else:
                    result[col] = 0.0

            elif self.config.standardization == StandardizationMethod.RANK:
                # 排序标准化到 [0, 1]
                result[col] = result[col].rank(pct=True)

            elif self.config.standardization == StandardizationMethod.MINMAX:
                # Min-Max 标准化
                col_min = result[col].min()
                col_max = result[col].max()
                if col_max - col_min > 1e-8:
                    result[col] = (result[col] - col_min) / (col_max - col_min)
                else:
                    result[col] = 0.5

        return result, stats_dict

    def _neutralize(
        self,
        data: pd.DataFrame,
        factor_columns: List[str],
        market_cap_column: Optional[str],
        industry_column: Optional[str],
    ) -> pd.DataFrame:
        """中性化处理"""
        result = data.copy()

        if self.config.neutralization == NeutralizationType.INDUSTRY:
            if industry_column and industry_column in result.columns:
                result = self._neutralize_by_group(result, factor_columns, industry_column)

        elif self.config.neutralization == NeutralizationType.MARKET_CAP:
            if market_cap_column and market_cap_column in result.columns:
                result = self._neutralize_by_market_cap(result, factor_columns, market_cap_column)

        elif self.config.neutralization == NeutralizationType.INDUSTRY_MARKET_CAP:
            # 先做行业中性化，再做市值中性化
            if industry_column and industry_column in result.columns:
                result = self._neutralize_by_group(result, factor_columns, industry_column)
            if market_cap_column and market_cap_column in result.columns:
                result = self._neutralize_by_market_cap(result, factor_columns, market_cap_column)

        return result

    def _neutralize_by_group(
        self,
        data: pd.DataFrame,
        factor_columns: List[str],
        group_column: str
    ) -> pd.DataFrame:
        """按分组进行中性化（行业中性化）"""
        result = data.copy()

        for col in factor_columns:
            if col not in result.columns:
                continue

            # 计算组内均值
            group_means = result.groupby(group_column)[col].transform("mean")
            result[col] = result[col] - group_means

        return result

    def _neutralize_by_market_cap(
        self,
        data: pd.DataFrame,
        factor_columns: List[str],
        market_cap_column: str
    ) -> pd.DataFrame:
        """市值中性化（对市值取对数后回归）"""
        result = data.copy()

        # 对市值取对数
        log_market_cap = np.log(result[market_cap_column].replace(0, np.nan))

        for col in factor_columns:
            if col not in result.columns:
                continue

            # 去除缺失值
            valid_mask = log_market_cap.notna() & result[col].notna()

            if valid_mask.sum() < 10:
                continue

            # 线性回归取残差
            X = log_market_cap[valid_mask].values.reshape(-1, 1)
            y = result[col][valid_mask].values

            try:
                from sklearn.linear_model import LinearRegression

                reg = LinearRegression()
                reg.fit(X, y)

                # 预测值
                y_pred = reg.predict(X)

                # 残差 = 原始值 - 预测值
                residuals = y - y_pred

                # 更新结果
                result.loc[valid_mask, col] = residuals
            except Exception:
                # 如果回归失败，做简单的市值排序中立
                result[col] = result[col] - result[col].rank(pct=True) * result[market_cap_column] / result[market_cap_column].max()

        return result


def create_pipeline(
    winsorize: Tuple[float, float] = (0.01, 0.99),
    standardization: str = "zscore",
    neutralization: str = "none",
    fill_method: str = "median",
) -> FactorPipeline:
    """
    便捷函数：创建因子流水线

    Args:
        winsorize: 去极值百分位 (lower, upper)
        standardization: 标准化方法 (zscore, rank, minmax)
        neutralization: 中性化方法 (none, industry, market_cap, industry_market_cap)
        fill_method: 缺失值填充方法 (median, mean, zero, forward)

    Returns:
        FactorPipeline: 因子流水线实例
    """
    config = PipelineConfig(
        winsorize_lower=winsorize[0],
        winsorize_upper=winsorize[1],
        standardization=StandardizationMethod(standardization),
        neutralization=NeutralizationType(neutralization),
        fill_method=fill_method,
    )
    return FactorPipeline(config)


def process_factors(
    data: pd.DataFrame,
    factor_columns: List[str],
    market_cap_column: Optional[str] = None,
    industry_column: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    便捷函数：快速处理因子

    Args:
        data: 输入数据
        factor_columns: 因子列名
        market_cap_column: 市值列名
        industry_column: 行业列名
        **kwargs: 传递给 PipelineConfig 的参数

    Returns:
        处理后的数据
    """
    config = PipelineConfig(**kwargs)
    pipeline = FactorPipeline(config)
    result = pipeline.process(data, factor_columns, market_cap_column, industry_column)
    return result.data
