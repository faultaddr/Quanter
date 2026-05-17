"""
Qlib 数据加载器

直接使用 qlib 标准数据加载器，支持：
1. 从 ~/.qlib/qlib_data/cn_data 加载数据
2. 批量获取股票特征数据
3. Alpha158 特征工程
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from ...core.logging import get_logger

logger = get_logger(__name__)

# Qlib 可用性检查
try:
    import qlib
    from qlib.data import D
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("pyqlib 未安装，QlibDataLoader 将不可用")


class QlibDataLoader:
    """
    Qlib 数据加载器

    直接使用 qlib 标准数据（~/.qlib/qlib_data/cn_data）
    """

    # 默认 qlib 数据路径
    DEFAULT_QLIB_DATA_PATH = "~/.qlib/qlib_data/cn_data"

    # 数据划分日期（按年份固定划分）
    TRAIN_END_DATE = "2022-12-31"
    VALID_END_DATE = "2024-06-30"

    # 股票代码前缀映射
    # 沪市: 600, 601, 603, 688, 900 (B股)
    # 深市: 000, 001, 002, 003, 200 (B股), 300, 301
    # 北交所: 8, 4
    SH_PREFIXES = ('60', '68', '90')  # 沪市
    SZ_PREFIXES = ('00', '20', '30')  # 深市

    def __init__(self, provider_uri: Optional[str] = None):
        """
        初始化数据加载器

        Args:
            provider_uri: qlib 数据路径，默认 ~/.qlib/qlib_data/cn_data
        """
        self.provider_uri = os.path.expanduser(
            provider_uri or self.DEFAULT_QLIB_DATA_PATH
        )
        self._initialized = False
        self._all_instruments = None

    @staticmethod
    def normalize_instrument(instrument: str) -> str:
        """
        标准化股票代码为 qlib 格式

        支持的输入格式:
        - "600640" -> "SH600640"
        - "600640.SH" -> "SH600640"
        - "SH600640" -> "SH600640"
        - "000001" -> "SZ000001"
        - "000001.SZ" -> "SZ000001"
        - "BJ430017" -> "BJ430017"

        Args:
            instrument: 原始股票代码

        Returns:
            qlib 标准格式的股票代码
        """
        instrument = instrument.strip().upper()

        # 已经是 qlib 格式 (SH600640, SZ000001, BJ430017)
        if instrument.startswith(('SH', 'SZ', 'BJ')):
            return instrument

        # 移除后缀 (.SH, .SZ)
        if '.' in instrument:
            code, suffix = instrument.rsplit('.', 1)
            suffix = suffix.upper()
            if suffix == 'SH':
                return f'SH{code}'
            elif suffix == 'SZ':
                return f'SZ{code}'
            elif suffix == 'BJ':
                return f'BJ{code}'
            instrument = code

        # 根据代码前缀判断市场
        if instrument.startswith(QlibDataLoader.SH_PREFIXES):
            return f'SH{instrument}'
        elif instrument.startswith(QlibDataLoader.SZ_PREFIXES):
            return f'SZ{instrument}'
        elif instrument.startswith(('4', '8')):
            # 北交所
            return f'BJ{instrument}'
        else:
            # 默认深市
            return f'SZ{instrument}'

    def init_qlib(self) -> bool:
        """
        初始化 Qlib

        Returns:
            是否初始化成功
        """
        if not QLIB_AVAILABLE:
            logger.error("pyqlib 未安装，请运行: pip install pyqlib")
            return False

        if self._initialized:
            return True

        try:
            # 检查数据路径是否存在
            if not Path(self.provider_uri).exists():
                logger.error(f"Qlib 数据路径不存在: {self.provider_uri}")
                return False

            # 初始化 qlib
            qlib.init(provider_uri=self.provider_uri)
            self._initialized = True
            logger.info(f"Qlib 初始化成功: {self.provider_uri}")
            return True

        except Exception as e:
            logger.error(f"Qlib 初始化失败: {e}")
            return False

    def get_all_instruments(self) -> List[str]:
        """
        获取所有股票代码

        Returns:
            股票代码列表（qlib 格式，如 SH600000, SZ000001）
        """
        if not self._initialized:
            if not self.init_qlib():
                return []

        if self._all_instruments is not None:
            return self._all_instruments

        try:
            # 从 instruments/all.txt 读取股票列表
            instruments_file = Path(self.provider_uri) / 'instruments' / 'all.txt'
            if instruments_file.exists():
                instruments = []
                with open(instruments_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if parts:
                            instruments.append(parts[0])
                self._all_instruments = sorted(instruments)
                logger.info(f"加载 {len(instruments)} 只股票")
                return self._all_instruments
            else:
                logger.warning(f"股票列表文件不存在: {instruments_file}")
                return []
        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return []

    def get_trade_dates(
        self,
        start_date: str = "2017-01-01",
        end_date: str = "2026-12-31"
    ) -> List[str]:
        """
        获取交易日历

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表
        """
        if not self._initialized:
            if not self.init_qlib():
                return []

        try:
            dates = D.calendar(freq='day')
            # 过滤日期范围
            dates = dates[(dates >= start_date) & (dates <= end_date)]
            return [d.strftime('%Y-%m-%d') for d in dates]
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return []

    def load_stock_data(
        self,
        instrument: str,
        start_date: str = "2017-01-01",
        end_date: str = "2026-12-31",
        fields: Optional[List[str]] = None,
        use_adjclose: bool = False
    ) -> pd.DataFrame:
        """
        加载单只股票的数据

        Args:
            instrument: 股票代码（支持多种格式：600640, 600640.SH, SH600640）
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要加载的字段，默认 ['open', 'close', 'high', 'low', 'volume']
            use_adjclose: 是否使用前复权价格（与行情软件一致），默认 False
                - False: 使用后复权价格（$close），适合技术分析和回测
                - True: 使用前复权价格，与行情软件显示一致
                  计算方式：前复权 = $close / $factor[latest]

        Returns:
            DataFrame，索引为日期

        价格说明:
            qlib 数据的复权关系：
            - $close: 后复权价格，保持历史价格连续性
            - $factor: 复权因子，raw_price = $close / $factor
            - $adjclose: qlib 内置字段，数据有误（数值膨胀），不可直接使用
            - 正确前复权 = $close / $factor[latest]，与行情软件一致
        """
        if not self._initialized:
            if not self.init_qlib():
                return pd.DataFrame()

        # 标准化股票代码
        normalized_instrument = self.normalize_instrument(instrument)

        # qlib 字段格式：需要使用 $ 前缀
        if fields is None:
            # 始终获取 close 和 adjclose，用于计算前复权价格
            fields = ['$open', '$close', '$adjclose', '$high', '$low', '$volume', '$vwap', '$factor']

        try:
            # 使用 qlib D.features 加载数据
            df = D.features(
                [normalized_instrument],
                fields,
                start_date,
                end_date,
                freq='day'
            )

            if df.empty:
                return pd.DataFrame()

            # 重置索引
            df = df.reset_index()
            df = df.rename(columns={'datetime': 'date'})

            # 重命名列：移除 $ 前缀
            df.columns = [col.lstrip('$') if col.startswith('$') else col for col in df.columns]

            # 处理前复权价格：使用 factor 计算正确的前复权价格
            # qlib $adjclose 数据有误（累积了所有复权因子，数值膨胀数十倍）
            # 正确公式：前复权 = $close / $factor[latest_day]
            if use_adjclose and 'factor' in df.columns and 'close' in df.columns:
                latest_factor = df['factor'].iloc[-1]
                if latest_factor > 0:
                    adj_ratio = 1.0 / latest_factor
                    df['open'] = df['open'] * adj_ratio
                    df['high'] = df['high'] * adj_ratio
                    df['low'] = df['low'] * adj_ratio
                    df['close'] = df['close'] * adj_ratio

            # 设置日期索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

            return df

        except Exception as e:
            logger.error(f"加载股票数据失败 [{instrument}]: {e}")
            return pd.DataFrame()

    def load_stocks_batch(
        self,
        instruments: List[str],
        start_date: str = "2017-01-01",
        end_date: str = "2026-12-31",
        fields: Optional[List[str]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载多只股票的数据

        Args:
            instruments: 股票代码列表（支持多种格式）
            start_date: 开始日期
            end_date: 结束日期
            fields: 需要加载的字段

        Returns:
            字典 {股票代码: DataFrame}
        """
        if not self._initialized:
            if not self.init_qlib():
                return {}

        # 标准化所有股票代码
        original_to_normalized = {inst: self.normalize_instrument(inst) for inst in instruments}
        normalized_instruments = list(original_to_normalized.values())

        # qlib 字段格式：需要使用 $ 前缀
        # 默认字段：open, close, high, low, volume, vwap 等
        if fields is None:
            fields = ['$open', '$close', '$high', '$low', '$volume', '$vwap']

        try:
            # 使用 qlib D.features 批量加载
            df = D.features(
                normalized_instruments,
                fields,
                start_date,
                end_date,
                freq='day'
            )

            if df.empty:
                return {}

            result = {}

            # 按 instrument 分组
            df = df.reset_index()

            # 重命名列：移除 $ 前缀
            df.columns = [col.lstrip('$') if col.startswith('$') else col for col in df.columns]

            for original, normalized in original_to_normalized.items():
                stock_df = df[df['instrument'] == normalized].copy()
                if stock_df.empty:
                    continue

                # 重命名并设置索引
                stock_df = stock_df.rename(columns={'datetime': 'date'})
                if 'date' in stock_df.columns:
                    stock_df['date'] = pd.to_datetime(stock_df['date'])
                    stock_df = stock_df.set_index('date')

                stock_df = stock_df.drop(columns=['instrument'], errors='ignore')
                result[original] = stock_df

            logger.info(f"批量加载完成: {len(result)}/{len(instruments)} 只股票")
            return result

        except Exception as e:
            logger.error(f"批量加载数据失败: {e}")
            return {}

    def create_dataset(
        self,
        instruments: List[str],
        start_date: str = "2017-01-01",
        end_date: str = "2026-12-31",
        feature_type: str = "alpha158",
        label_horizon: int = 10
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        创建训练数据集

        按年份固定划分：
        - 训练集: 2017-01-01 ~ 2022-12-31
        - 验证集: 2023-01-01 ~ 2024-06-30
        - 测试集: 2024-07-01 ~ end_date

        Args:
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            feature_type: 特征类型 (alpha158, alpha360)
            label_horizon: 标签预测周期

        Returns:
            (特征 DataFrame, 标签 Series)
        """
        if not self._initialized:
            if not self.init_qlib():
                return pd.DataFrame(), pd.Series()

        # 导入特征工程
        from .qlib_data_converter import Alpha158Features, Alpha360Features

        all_features = []
        all_labels = []
        all_indices = []

        # 加载所有股票数据
        stock_data = self.load_stocks_batch(instruments, start_date, end_date)

        for instrument, df in stock_data.items():
            if len(df) < 120:
                logger.debug(f"[{instrument}] 数据不足，跳过")
                continue

            try:
                # 生成特征
                if feature_type == "alpha158":
                    features = Alpha158Features.generate(df)
                elif feature_type == "alpha360":
                    features = Alpha360Features.generate(df)
                else:
                    features = Alpha158Features.generate(df)

                # 生成标签：未来 N 天收益率
                labels = df['close'].shift(-label_horizon) / df['close'] - 1

                # 对齐数据
                valid_idx = features.dropna(how='all').index.intersection(labels.dropna().index)
                features = features.loc[valid_idx]
                labels = labels.loc[valid_idx]

                # 构建 MultiIndex
                for idx in features.index:
                    all_features.append(features.loc[idx].values)
                    all_labels.append(labels.loc[idx])
                    all_indices.append((idx, instrument))

            except Exception as e:
                logger.warning(f"[{instrument}] 特征生成失败: {e}")
                continue

        if not all_features:
            logger.error("没有有效的训练数据")
            return pd.DataFrame(), pd.Series()

        # 构建 DataFrame
        feature_names = features.columns.tolist() if all_features else []

        features_df = pd.DataFrame(
            all_features,
            columns=feature_names,
            index=pd.MultiIndex.from_tuples(all_indices, names=['datetime', 'instrument'])
        )

        labels_series = pd.Series(
            all_labels,
            index=pd.MultiIndex.from_tuples(all_indices, names=['datetime', 'instrument']),
            name='label'
        )

        logger.info(
            f"数据集创建完成: {len(features_df)} 条记录, "
            f"{len(instruments)} 只股票, {len(feature_names)} 个特征"
        )

        return features_df, labels_series

    def get_data_segments(self) -> Dict[str, Tuple[str, str]]:
        """
        获取数据划分

        Returns:
            字典 {'train': (start, end), 'valid': (start, end), 'test': (start, end)}
        """
        return {
            'train': ('2017-01-01', self.TRAIN_END_DATE),
            'valid': ('2023-01-01', self.VALID_END_DATE),
            'test': ('2024-07-01', '2026-12-31'),
        }


# 便捷函数
def get_qlib_loader(provider_uri: Optional[str] = None) -> QlibDataLoader:
    """获取 Qlib 数据加载器实例"""
    return QlibDataLoader(provider_uri)


def load_qlib_data(
    instruments: List[str],
    start_date: str = "2017-01-01",
    end_date: str = "2026-12-31"
) -> Dict[str, pd.DataFrame]:
    """便捷函数：批量加载 Qlib 数据"""
    loader = QlibDataLoader()
    return loader.load_stocks_batch(instruments, start_date, end_date)
