"""
因子中性化处理模块

实现行业和市值中性化：
- 线性回归中性化
- 分组均值中性化
- 多因子中性化
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class NeutralizationMethod(str, Enum):
    """中性化方法"""
    REGRESSION = "regression"     # 线性回归中性化
    GROUP_MEAN = "group_mean"      # 分组均值中性化
    RESIDUAL = "residual"          # 残差法中性化


@dataclass
class NeutralizationResult:
    """中性化结果"""
    original_factor: pd.Series
    neutralized_factor: pd.Series
    r_squared: float
    coefficients: Dict[str, float]
    method: NeutralizationMethod


class FactorNeutralizer:
    """
    因子中性化处理器

    提供多种中性化方法：
    - 线性回归中性化
    - 分组均值中性化
    - 残差法中性化
    """

    def __init__(self):
        """初始化中性化处理器"""
        self._cache: Dict[str, NeutralizationResult] = {}

    def neutralize_by_market_cap(
        self,
        factor: pd.Series,
        market_cap: pd.Series,
        log_transform: bool = True,
    ) -> pd.Series:
        """
        市值中性化

        对因子进行市值回归，取残差作为中性化后的因子

        Args:
            factor: 因子值
            market_cap: 市值
            log_transform: 是否取对数

        Returns:
            中性化后的因子
        """
        # 对齐数据
        valid_idx = factor.notna() & market_cap.notna() & (market_cap > 0)
        if valid_idx.sum() < 10:
            return factor

        f = factor[valid_idx]
        mc = market_cap[valid_idx]

        if log_transform:
            mc = np.log(mc)

        # 线性回归
        X = np.column_stack([np.ones(len(mc)), mc.values])
        y = f.values

        try:
            # 使用最小二乘法
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta

            # 残差 = 原始值 - 预测值
            residual = y - y_pred

            # 保持原始索引
            result = factor.copy()
            result[valid_idx] = residual

            return result

        except Exception:
            # 如果回归失败，返回原始因子
            return factor

    def neutralize_by_industry(
        self,
        factor: pd.Series,
        industry: pd.Series,
    ) -> pd.Series:
        """
        行业中性化

        用行业均值进行中性化

        Args:
            factor: 因子值
            industry: 行业分类

        Returns:
            中性化后的因子
        """
        result = factor.copy()

        # 计算各行业的均值
        industry_means = factor.groupby(industry).transform("mean")

        # 中性化 = 原始值 - 行业均值
        result = factor - industry_means

        return result

    def neutralize_industry_and_market_cap(
        self,
        factor: pd.Series,
        industry: pd.Series,
        market_cap: pd.Series,
        order: str = "industry_first",
    ) -> pd.Series:
        """
        行业 + 市值中性化

        Args:
            factor: 因子值
            industry: 行业分类
            market_cap: 市值
            order: 中性化顺序 ("industry_first" 或 "market_cap_first")

        Returns:
            中性化后的因子
        """
        if order == "industry_first":
            # 先行业中性化
            neutralized = self.neutralize_by_industry(factor, industry)
            # 再市值中性化
            neutralized = self.neutralize_by_market_cap(neutralized, market_cap)
        else:
            # 先市值中性化
            neutralized = self.neutralize_by_market_cap(factor, market_cap)
            # 再行业中性化
            neutralized = self.neutralize_by_industry(neutralized, industry)

        return neutralized

    def neutralize_multi_factor(
        self,
        factors: pd.DataFrame,
        control_factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        多因子中性化

        对多个因子进行控制变量的回归中性化

        Args:
            factors: 待中性化的因子DataFrame（多列）
            control_factors: 控制变量DataFrame

        Returns:
            中性化后的因子DataFrame
        """
        result = factors.copy()

        for col in factors.columns:
            if col in control_factors.columns:
                continue

            control_cols = [c for c in control_factors.columns if c in factors.columns and c != col]

            if not control_cols:
                continue

            # 对齐数据
            valid_mask = factors[col].notna()
            for c in control_cols:
                valid_mask = valid_mask & control_factors[c].notna()

            if valid_mask.sum() < 20:
                continue

            y = factors.loc[valid_mask, col].values
            X = control_factors.loc[valid_mask, control_cols].values

            # 添加常数项
            X = np.column_stack([np.ones(len(y)), X])

            try:
                # 回归
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                y_pred = X @ beta
                residual = y - y_pred

                result.loc[valid_mask, col] = residual
            except Exception:
                continue

        return result

    def neutralize_with_style_factors(
        self,
        factor: pd.Series,
        style_factors: pd.DataFrame,
    ) -> pd.Series:
        """
        风格因子中性化

        对因子进行风格因子（如市值、估值、动量等）的回归中性化

        Args:
            factor: 因子值
            style_factors: 风格因子DataFrame

        Returns:
            中性化后的因子
        """
        # 对齐数据
        valid_mask = factor.notna()
        for col in style_factors.columns:
            valid_mask = valid_mask & style_factors[col].notna()

        if valid_mask.sum() < 20:
            return factor

        y = factor[valid_mask].values
        X = style_factors.loc[valid_mask].values

        # 添加常数项
        X = np.column_stack([np.ones(len(y)), X])

        try:
            # 回归
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residual = y - y_pred

            result = factor.copy()
            result[valid_mask] = residual
            return result

        except Exception:
            return factor


def neutralize_factor(
    factor: pd.Series,
    industry: Optional[pd.Series] = None,
    market_cap: Optional[pd.Series] = None,
    style_factors: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    便捷函数：快速中性化因子

    Args:
        factor: 因子值
        industry: 行业分类
        market_cap: 市值
        style_factors: 风格因子

    Returns:
        中性化后的因子
    """
    neutralizer = FactorNeutralizer()

    if industry is not None and market_cap is not None:
        return neutralizer.neutralize_industry_and_market_cap(
            factor, industry, market_cap
        )
    elif industry is not None:
        return neutralizer.neutralize_by_industry(factor, industry)
    elif market_cap is not None:
        return neutralizer.neutralize_by_market_cap(factor, market_cap)
    elif style_factors is not None:
        return neutralizer.neutralize_with_style_factors(factor, style_factors)
    else:
        return factor
