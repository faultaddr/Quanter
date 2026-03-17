"""
Qlib 数据适配器

将普通 DataFrame 数据转换为 Qlib 原生模型所需的 DatasetH 格式
"""

import warnings
from typing import Dict, Tuple, Optional, List, Any, Union
from datetime import datetime
import pandas as pd
import numpy as np

from ...core.logging import get_logger

logger = get_logger(__name__)

# Qlib 可用性检查
try:
    import qlib
    from qlib.data.dataset import DatasetH, TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    DatasetH = None
    TSDatasetH = None
    DataHandlerLP = None
    logger.warning("pyqlib 未安装，QlibDataAdapter 将不可用")


def create_qlib_dataset_from_dataframe(
    features: pd.DataFrame,
    labels: pd.Series,
    instrument: str = "stock",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> Any:
    """
    从 DataFrame 创建 Qlib 原生 DatasetH

    Args:
        features: 特征 DataFrame (索引为 datetime 或 MultiIndex(datetime, instrument))
        labels: 标签 Series (索引为 datetime 或 MultiIndex(datetime, instrument))
        instrument: 股票代码
        train_ratio: 训练集比例
        valid_ratio: 验证集比例

    Returns:
        Qlib DatasetH 实例
    """
    if not QLIB_AVAILABLE:
        raise ImportError("pyqlib 未安装，无法创建 Qlib DatasetH")

    # 检查是否已经是 MultiIndex
    is_multiindex = isinstance(features.index, pd.MultiIndex)

    if is_multiindex:
        # 已经是 MultiIndex，直接使用
        index = features.index
        # 提取日期用于计算 segments
        dates = features.index.get_level_values('datetime')
    else:
        # 确保索引是 DatetimeIndex
        if not isinstance(features.index, pd.DatetimeIndex):
            features.index = pd.to_datetime(features.index)
        if not isinstance(labels.index, pd.DatetimeIndex):
            labels.index = pd.to_datetime(labels.index)

        # 创建 MultiIndex (datetime, instrument)
        index = pd.MultiIndex.from_arrays(
            [features.index, [instrument] * len(features)],
            names=['datetime', 'instrument']
        )
        dates = features.index

    # 创建 Qlib 格式的 DataFrame
    # 特征列: MultiIndex(('feature', col_name), ...)
    feature_cols = pd.MultiIndex.from_product([['feature'], features.columns])
    feature_df = pd.DataFrame(features.values, index=index, columns=feature_cols)

    # 标签列: MultiIndex(('label', 'label'),)
    label_df = pd.DataFrame(
        labels.values.astype(float),
        index=index,
        columns=pd.MultiIndex.from_product([['label'], ['label']])
    )

    # 合并
    qlib_df = pd.concat([feature_df, label_df], axis=1)

    # 创建 DataHandlerLP
    handler = DataHandlerLP.from_df(qlib_df)

    # 计算数据划分
    total_len = len(features)
    train_end = int(total_len * train_ratio)
    valid_end = int(total_len * (train_ratio + valid_ratio))

    segments = {
        'train': (dates[0].strftime('%Y-%m-%d'), dates[train_end - 1].strftime('%Y-%m-%d')),
        'valid': (dates[train_end].strftime('%Y-%m-%d'), dates[valid_end - 1].strftime('%Y-%m-%d')),
        'test': (dates[valid_end].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')),
    }

    # 创建 DatasetH
    dataset = DatasetH(handler=handler, segments=segments)

    logger.info(f"创建 Qlib DatasetH 成功: train={segments['train']}, valid={segments['valid']}")

    return dataset


def create_ts_dataset_from_dataframe(
    features: pd.DataFrame,
    labels: pd.Series,
    step_len: int = 30,
    instrument: str = "stock",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> Any:
    """
    从 DataFrame 创建 Qlib 时间序列数据集 TSDatasetH

    Args:
        features: 特征 DataFrame (索引为 datetime)
        labels: 标签 Series (索引为 datetime)
        step_len: 时间窗口长度
        instrument: 股票代码
        train_ratio: 训练集比例
        valid_ratio: 验证集比例

    Returns:
        Qlib TSDatasetH 实例
    """
    if not QLIB_AVAILABLE:
        raise ImportError("pyqlib 未安装，无法创建 Qlib TSDatasetH")

    # 确保索引是 DatetimeIndex
    if not isinstance(features.index, pd.DatetimeIndex):
        features.index = pd.to_datetime(features.index)
    if not isinstance(labels.index, pd.DatetimeIndex):
        labels.index = pd.to_datetime(labels.index)

    # 创建 MultiIndex (datetime, instrument)
    index = pd.MultiIndex.from_arrays(
        [features.index, [instrument] * len(features)],
        names=['datetime', 'instrument']
    )

    # 创建 Qlib 格式的 DataFrame
    feature_cols = pd.MultiIndex.from_product([['feature'], features.columns])
    feature_df = pd.DataFrame(features.values, index=index, columns=feature_cols)

    label_df = pd.DataFrame(
        labels.values.astype(float),
        index=index,
        columns=pd.MultiIndex.from_product([['label'], ['label']])
    )

    qlib_df = pd.concat([feature_df, label_df], axis=1)

    # 创建 DataHandlerLP
    handler = DataHandlerLP.from_df(qlib_df)

    # 计算数据划分
    total_len = len(features)
    train_end = int(total_len * train_ratio)
    valid_end = int(total_len * (train_ratio + valid_ratio))

    dates = features.index
    segments = {
        'train': (dates[0].strftime('%Y-%m-%d'), dates[train_end - 1].strftime('%Y-%m-%d')),
        'valid': (dates[train_end].strftime('%Y-%m-%d'), dates[valid_end - 1].strftime('%Y-%m-%d')),
        'test': (dates[valid_end].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')),
    }

    # 创建 TSDatasetH（用于时间序列模型）
    dataset = TSDatasetH(handler=handler, segments=segments, step_len=step_len)

    logger.info(f"创建 Qlib TSDatasetH 成功: step_len={step_len}, train={segments['train']}")

    return dataset


class SimpleDatasetH:
    """
    简化版 DatasetH (不依赖 Qlib 原生 DatasetH)

    当 Qlib 不可用时作为后备方案
    """

    def __init__(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        segments: Dict[str, Tuple[str, str]],
        instrument: str = "stock",
    ):
        """
        初始化简化版 DatasetH

        Args:
            features: 特征 DataFrame
            labels: 标签 Series
            segments: 数据划分 {"train": (start, end), "valid": (start, end), "test": (start, end)}
            instrument: 股票代码
        """
        self.features = features.copy()
        self.labels = labels.copy()
        self.segments = segments
        self.instrument = instrument

        # 检查索引类型并提取日期
        self._is_multiindex = isinstance(self.features.index, pd.MultiIndex)
        if self._is_multiindex:
            # MultiIndex 格式：(datetime, instrument)
            self._datetime_index = self.features.index.get_level_values('datetime')
        else:
            # 普通索引，尝试转换为日期时间
            if not isinstance(self.features.index, pd.DatetimeIndex):
                self.features.index = pd.to_datetime(self.features.index)
            if not isinstance(self.labels.index, pd.DatetimeIndex):
                self.labels.index = pd.to_datetime(self.labels.index)
            self._datetime_index = self.features.index

        # 存储原始数据（用于回退模型训练）
        self._raw_feature_data = self.features.values
        self._raw_label_data = self.labels.values.astype(float)
        self._feature_columns = list(self.features.columns)

        # 准备各段数据的索引
        self._segment_indices = {}
        for key, (start, end) in self.segments.items():
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)
            mask = (self._datetime_index >= start_dt) & (self._datetime_index <= end_dt)
            self._segment_indices[key] = mask

    def prepare(
        self,
        segment_key: Union[str, List[str]],
        col_set: Optional[List[str]] = None,
        data_key: str = None,
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        准备数据（兼容 Qlib DatasetH 接口）

        Args:
            segment_key: 数据段名称 ("train", "valid", "test") 或列表 ["train", "valid"]
            col_set: 列集合 (["feature", "label"])
            data_key: 数据键 (兼容参数)

        Returns:
            包含 MultiIndex 列的 DataFrame
        """
        if col_set is None:
            col_set = ["feature", "label"]

        # 支持 segment_key 为列表的情况（Qlib 原生模型会传入 ["train", "valid"]）
        if isinstance(segment_key, list):
            results = [self._prepare_single(key, col_set, data_key) for key in segment_key]
            if len(results) == 2:
                return results[0], results[1]
            return results

        return self._prepare_single(segment_key, col_set, data_key)

    def _prepare_single(
        self,
        segment_key: str,
        col_set: List[str],
        data_key: str = None,
    ) -> pd.DataFrame:
        """准备单个数据段"""
        mask = self._segment_indices[segment_key]

        # 创建 MultiIndex (datetime, instrument)
        index = pd.MultiIndex.from_arrays(
            [self.features.index[mask], [self.instrument] * mask.sum()],
            names=['datetime', 'instrument']
        )

        result_parts = []

        if "feature" in col_set:
            feat = pd.DataFrame(
                self.features.loc[mask].values,
                index=index,
                columns=pd.MultiIndex.from_product([['feature'], self.features.columns])
            )
            result_parts.append(feat)

        if "label" in col_set:
            lab = self.labels.loc[mask].values.astype(float)
            lab_df = pd.DataFrame(
                lab,
                index=index,
                columns=pd.MultiIndex.from_product([['label'], ['label']])
            )
            result_parts.append(lab_df)

        if len(result_parts) == 0:
            return pd.DataFrame()

        result = pd.concat(result_parts, axis=1)
        return result

    def __len__(self) -> int:
        return len(self.features)


def create_qlib_compatible_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    use_native: bool = True,
) -> Union[Any, SimpleDatasetH]:
    """
    创建与 Qlib 兼容的数据集

    优先使用 Qlib 原生 DatasetH，如果失败则回退到 SimpleDatasetH

    Args:
        features: 特征 DataFrame
        labels: 标签 Series
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        use_native: 是否尝试使用 Qlib 原生 DatasetH

    Returns:
        DatasetH 或 SimpleDatasetH 实例
    """
    total_len = len(features)
    train_end = int(total_len * train_ratio)
    valid_end = int(total_len * (train_ratio + valid_ratio))

    dates = features.index

    # 检查索引是否为有效的日期索引
    # 首先检查是否是 MultiIndex，如果是则提取 datetime 层级
    if isinstance(dates, pd.MultiIndex):
        # 尝试找到 datetime 层级
        if 'datetime' in dates.names:
            dates = dates.get_level_values('datetime')
        elif dates.nlevels >= 1:
            # 使用第一层作为日期
            dates = dates.get_level_values(0)

    has_valid_dates = isinstance(dates, pd.DatetimeIndex)
    if not has_valid_dates:
        # 尝试转换，但检查结果是否合理
        try:
            converted_dates = pd.to_datetime(dates)
            # 如果所有日期都是 1970-01-01，说明索引不是日期
            if converted_dates[0].year == 1970 and converted_dates[-1].year == 1970:
                has_valid_dates = False
                # 创建虚拟日期索引（按行号生成日期）
                dates = pd.date_range(start='2020-01-01', periods=total_len, freq='D')
                logger.info("输入无有效日期索引，使用虚拟日期范围")
            else:
                dates = converted_dates
                has_valid_dates = True
        except Exception:
            # 无法转换为日期，使用虚拟日期
            dates = pd.date_range(start='2020-01-01', periods=total_len, freq='D')
            logger.info("无法解析日期索引，使用虚拟日期范围")

    segments = {
        'train': (dates[0].strftime('%Y-%m-%d'), dates[train_end - 1].strftime('%Y-%m-%d')),
        'valid': (dates[train_end].strftime('%Y-%m-%d'), dates[valid_end - 1].strftime('%Y-%m-%d')),
        'test': (dates[valid_end].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')),
    }

    # 尝试使用 Qlib 原生 DatasetH
    if use_native and QLIB_AVAILABLE and has_valid_dates:
        try:
            return create_qlib_dataset_from_dataframe(
                features, labels, "stock", train_ratio, valid_ratio
            )
        except Exception as e:
            logger.warning(f"创建 Qlib 原生 DatasetH 失败: {e}，使用 SimpleDatasetH")

    # 回退到 SimpleDatasetH
    return SimpleDatasetH(features, labels, segments)


def create_ts_compatible_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    step_len: int = 30,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    use_native: bool = True,
) -> Union[Any, SimpleDatasetH]:
    """
    创建时间序列数据集

    优先使用 Qlib 原生 TSDatasetH，如果失败则回退到 SimpleDatasetH

    Args:
        features: 特征 DataFrame
        labels: 标签 Series
        step_len: 时间窗口长度
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        use_native: 是否尝试使用 Qlib 原生 TSDatasetH

    Returns:
        TSDatasetH 或 SimpleDatasetH 实例
    """
    total_len = len(features)
    train_end = int(total_len * train_ratio)
    valid_end = int(total_len * (train_ratio + valid_ratio))

    dates = features.index
    if not isinstance(dates, pd.DatetimeIndex):
        dates = pd.to_datetime(dates)

    segments = {
        'train': (dates[0].strftime('%Y-%m-%d'), dates[train_end - 1].strftime('%Y-%m-%d')),
        'valid': (dates[train_end].strftime('%Y-%m-%d'), dates[valid_end - 1].strftime('%Y-%m-%d')),
        'test': (dates[valid_end].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')),
    }

    # 尝试使用 Qlib 原生 TSDatasetH
    if use_native and QLIB_AVAILABLE:
        try:
            return create_ts_dataset_from_dataframe(
                features, labels, step_len, "stock", train_ratio, valid_ratio
            )
        except Exception as e:
            logger.warning(f"创建 Qlib 原生 TSDatasetH 失败: {e}，使用 SimpleDatasetH")

    # 回退到 SimpleDatasetH
    return SimpleDatasetH(features, labels, segments)