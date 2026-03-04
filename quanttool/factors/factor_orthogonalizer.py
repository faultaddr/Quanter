"""
因子正交化处理模块

使用施密特正交化去除因子共线性：
- 计算因子相关性矩阵
- 施密特正交化处理
- 因子有效性检验
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


@dataclass
class OrthogonalizationResult:
    """正交化结果"""
    orthogonal_factors: pd.DataFrame    # 正交化后的因子
    correlation_matrix: pd.DataFrame    # 原始相关性矩阵
    explained_variance: pd.Series       # 各因子解释方差
    transformation_matrix: np.ndarray   # 变换矩阵
    condition_number: float             # 条件数


class FactorOrthogonalizer:
    """
    因子正交化处理器

    使用施密特正交化去除因子共线性
    """

    def __init__(
        self,
        method: str = 'schmidt',
        normalize: bool = True,
        min_variance_explained: float = 0.01
    ):
        """
        初始化正交化处理器

        Args:
            method: 正交化方法 ('schmidt', 'pca', 'cholesky')
            normalize: 是否标准化因子
            min_variance_explained: 最小解释方差阈值
        """
        self.method = method
        self.normalize = normalize
        self.min_variance_explained = min_variance_explained

    def orthogonalize_factors(
        self,
        factor_df: pd.DataFrame,
        factor_order: Optional[List[str]] = None
    ) -> OrthogonalizationResult:
        """
        对因子进行正交化处理

        Args:
            factor_df: 因子DataFrame，每列是一个因子
            factor_order: 因子优先级顺序（用于施密特正交化）

        Returns:
            OrthogonalizationResult: 正交化结果
        """
        if factor_df.empty:
            raise ValueError("因子数据为空")

        # 处理缺失值
        factor_df = factor_df.dropna()

        if len(factor_df) < 10:
            raise ValueError("样本数量不足")

        # 标准化
        if self.normalize:
            factor_normalized = self._normalize_factors(factor_df)
        else:
            factor_normalized = factor_df.copy()

        # 计算相关性矩阵
        correlation_matrix = factor_normalized.corr()

        # 根据方法选择正交化
        if self.method == 'schmidt':
            orthogonal_factors, transform_matrix = self._schmidt_orthogonalization(
                factor_normalized, factor_order
            )
        elif self.method == 'pca':
            orthogonal_factors, transform_matrix = self._pca_orthogonalization(
                factor_normalized
            )
        elif self.method == 'cholesky':
            orthogonal_factors, transform_matrix = self._cholesky_orthogonalization(
                factor_normalized
            )
        else:
            raise ValueError(f"未知的正交化方法: {self.method}")

        # 计算解释方差
        explained_variance = orthogonal_factors.var()

        # 计算条件数
        condition_number = np.linalg.cond(correlation_matrix.values)

        return OrthogonalizationResult(
            orthogonal_factors=orthogonal_factors,
            correlation_matrix=correlation_matrix,
            explained_variance=explained_variance,
            transformation_matrix=transform_matrix,
            condition_number=condition_number
        )

    def _normalize_factors(self, factor_df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化因子（Z-score）
        """
        return (factor_df - factor_df.mean()) / (factor_df.std() + 1e-10)

    def _schmidt_orthogonalization(
        self,
        factor_df: pd.DataFrame,
        factor_order: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        施密特正交化

        按照指定顺序，依次将因子投影到已正交化因子的正交补空间

        Args:
            factor_df: 标准化后的因子
            factor_order: 因子优先级顺序

        Returns:
            Tuple: (正交化因子, 变换矩阵)
        """
        if factor_order is None:
            # 默认按方差排序
            factor_order = factor_df.var().sort_values(ascending=False).index.tolist()

        # 确保所有因子都在列表中
        remaining = [f for f in factor_df.columns if f not in factor_order]
        factor_order = factor_order + remaining

        n_samples = len(factor_df)
        n_factors = len(factor_order)

        # 初始化正交化因子矩阵
        orthogonal_matrix = np.zeros((n_samples, n_factors))
        transform_matrix = np.eye(n_factors)

        # 按顺序进行正交化
        for i, factor_name in enumerate(factor_order):
            if factor_name not in factor_df.columns:
                continue

            current_vector = factor_df[factor_name].values.copy()

            # 减去与之前所有正交因子的投影
            for j in range(i):
                if j < len(factor_order) and factor_order[j] in factor_df.columns:
                    prev_orthogonal = orthogonal_matrix[:, j]
                    # 计算投影系数
                    projection = np.dot(current_vector, prev_orthogonal) / (
                        np.dot(prev_orthogonal, prev_orthogonal) + 1e-10
                    )
                    current_vector = current_vector - projection * prev_orthogonal

                    # 记录变换矩阵
                    transform_matrix[j, i] = -projection

            orthogonal_matrix[:, i] = current_vector

        # 创建DataFrame
        orthogonal_df = pd.DataFrame(
            orthogonal_matrix,
            index=factor_df.index,
            columns=factor_order
        )

        return orthogonal_df, transform_matrix

    def _pca_orthogonalization(
        self,
        factor_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        PCA正交化

        使用主成分分析获取正交因子
        """
        from sklearn.decomposition import PCA

        n_components = len(factor_df.columns)
        pca = PCA(n_components=n_components)

        # 拟合PCA
        orthogonal_matrix = pca.fit_transform(factor_df.values)

        # 创建DataFrame
        component_names = [f'PC{i+1}' for i in range(n_components)]
        orthogonal_df = pd.DataFrame(
            orthogonal_matrix,
            index=factor_df.index,
            columns=component_names
        )

        return orthogonal_df, pca.components_

    def _cholesky_orthogonalization(
        self,
        factor_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Cholesky分解正交化

        使用Cholesky分解获取正交因子
        """
        # 计算协方差矩阵
        cov_matrix = factor_df.cov().values

        try:
            # Cholesky分解
            L = np.linalg.cholesky(cov_matrix)

            # 变换因子
            L_inv = np.linalg.inv(L)
            orthogonal_matrix = factor_df.values @ L_inv.T

            # 创建DataFrame
            orthogonal_df = pd.DataFrame(
                orthogonal_matrix,
                index=factor_df.index,
                columns=factor_df.columns
            )

            return orthogonal_df, L_inv

        except np.linalg.LinAlgError:
            # 如果Cholesky分解失败，回退到施密特正交化
            return self._schmidt_orthogonalization(factor_df)

    def calculate_factor_correlation(
        self,
        factor_df: pd.DataFrame,
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        计算因子相关性矩阵

        Args:
            factor_df: 因子DataFrame
            method: 相关性计算方法 ('pearson', 'spearman', 'kendall')

        Returns:
            pd.DataFrame: 相关性矩阵
        """
        if method == 'pearson':
            return factor_df.corr()
        elif method == 'spearman':
            return factor_df.corr(method='spearman')
        elif method == 'kendall':
            return factor_df.corr(method='kendall')
        else:
            raise ValueError(f"未知的相关性计算方法: {method}")

    def detect_multicollinearity(
        self,
        factor_df: pd.DataFrame,
        threshold: float = 0.7
    ) -> Dict:
        """
        检测多重共线性

        Args:
            factor_df: 因子DataFrame
            threshold: 相关性阈值

        Returns:
            Dict: 多重共线性检测结果
        """
        correlation_matrix = self.calculate_factor_correlation(factor_df)

        # 找出高相关因子对
        high_corr_pairs = []
        n = len(correlation_matrix)
        for i in range(n):
            for j in range(i+1, n):
                corr = correlation_matrix.iloc[i, j]
                if abs(corr) > threshold:
                    high_corr_pairs.append({
                        'factor1': correlation_matrix.index[i],
                        'factor2': correlation_matrix.columns[j],
                        'correlation': corr
                    })

        # 计算VIF（方差膨胀因子）
        vif_dict = self._calculate_vif(factor_df)

        # 计算条件数
        condition_number = np.linalg.cond(correlation_matrix.values)

        return {
            'high_correlation_pairs': high_corr_pairs,
            'vif': vif_dict,
            'condition_number': condition_number,
            'has_multicollinearity': len(high_corr_pairs) > 0 or condition_number > 30
        }

    def _calculate_vif(self, factor_df: pd.DataFrame) -> Dict[str, float]:
        """
        计算方差膨胀因子（VIF）

        VIF > 10 表示存在严重多重共线性
        """
        vif_dict = {}
        columns = factor_df.columns.tolist()

        for i, col in enumerate(columns):
            # 使用其他因子预测当前因子
            other_cols = [c for c in columns if c != col]
            if len(other_cols) == 0:
                vif_dict[col] = 1.0
                continue

            X = factor_df[other_cols].values
            y = factor_df[col].values

            try:
                # 计算R²
                X_with_intercept = np.column_stack([np.ones(len(X)), X])
                beta = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
                y_pred = X_with_intercept @ beta
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                # 计算VIF
                vif = 1 / (1 - r_squared) if r_squared < 1 else float('inf')
                vif_dict[col] = vif

            except Exception:
                vif_dict[col] = float('inf')

        return vif_dict

    def get_factor_importance(
        self,
        factor_df: pd.DataFrame,
        returns: pd.Series
    ) -> pd.Series:
        """
        计算因子重要性（基于IC）

        Args:
            factor_df: 因子DataFrame
            returns: 收益率序列

        Returns:
            pd.Series: 各因子的IC绝对值
        """
        ic_values = {}

        for col in factor_df.columns:
            factor_values = factor_df[col]

            # 对齐数据
            common_idx = factor_values.dropna().index.intersection(returns.dropna().index)
            if len(common_idx) < 10:
                ic_values[col] = 0
                continue

            factor_aligned = factor_values.loc[common_idx]
            returns_aligned = returns.loc[common_idx]

            # 计算IC
            ic, _ = stats.spearmanr(factor_aligned.values, returns_aligned.values)
            ic_values[col] = abs(ic) if not np.isnan(ic) else 0

        return pd.Series(ic_values).sort_values(ascending=False)

    def suggest_factor_selection(
        self,
        factor_df: pd.DataFrame,
        returns: pd.Series,
        max_factors: int = 10,
        corr_threshold: float = 0.6
    ) -> List[str]:
        """
        建议因子选择

        基于IC和共线性选择最优因子组合

        Args:
            factor_df: 因子DataFrame
            returns: 收益率序列
            max_factors: 最大因子数量
            corr_threshold: 相关性阈值

        Returns:
            List[str]: 建议选择的因子列表
        """
        # 计算因子重要性
        importance = self.get_factor_importance(factor_df, returns)

        # 检测多重共线性
        multicollinearity = self.detect_multicollinearity(factor_df, corr_threshold)
        high_corr_pairs = multicollinearity['high_correlation_pairs']

        # 贪心选择
        selected = []
        for factor in importance.index:
            if len(selected) >= max_factors:
                break

            # 检查与已选因子的相关性
            can_add = True
            for selected_factor in selected:
                for pair in high_corr_pairs:
                    if ((pair['factor1'] == factor and pair['factor2'] == selected_factor) or
                        (pair['factor2'] == factor and pair['factor1'] == selected_factor)):
                        if abs(pair['correlation']) > corr_threshold:
                            can_add = False
                            break
                if not can_add:
                    break

            if can_add:
                selected.append(factor)

        return selected


def orthogonalize_factors(
    factor_df: pd.DataFrame,
    method: str = 'schmidt'
) -> pd.DataFrame:
    """
    便捷函数：因子正交化

    Args:
        factor_df: 因子DataFrame
        method: 正交化方法

    Returns:
        pd.DataFrame: 正交化后的因子
    """
    orthogonalizer = FactorOrthogonalizer(method=method)
    result = orthogonalizer.orthogonalize_factors(factor_df)
    return result.orthogonal_factors