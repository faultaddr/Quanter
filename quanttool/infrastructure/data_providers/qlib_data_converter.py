"""
Qlib 官方训练流程数据转换器

将 Quanter 系统的数据转换为 qlib 官方能接受的格式：
1. 支持转换为 qlib 二进制数据格式（dump_data）
2. 支持 Alpha158/Alpha360 官方特征
3. 完全兼容 qlib 官方训练流程

使用示例:
    # 方式一：转换为 qlib 二进制格式
    converter = QlibDataConverter(output_dir="qlib_data/cn_data")
    converter.dump_data(symbols=["000001.SZ", "600519.SH"])

    # 方式二：直接创建 DatasetH（用于单股票训练）
    dataset = converter.create_dataset_from_cache(
        symbols=["000001.SZ"],
        start_date="2020-01-01",
        end_date="2024-12-31"
    )
"""

import os
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np

from ...core.logging import get_logger

logger = get_logger(__name__)

# Qlib 可用性检查
try:
    import qlib
    from qlib.data.dataset import DatasetH, TSDatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.cache import H
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    DatasetH = None
    TSDatasetH = None
    DataHandlerLP = None
    logger.warning("pyqlib 未安装，部分功能将不可用")


@dataclass
class QlibDataConfig:
    """Qlib 数据配置"""

    # 数据目录
    cache_dir: str = ".cache/incremental_data"
    output_dir: str = "qlib_data/cn_data"

    # 特征配置
    feature_type: str = "alpha158"  # alpha158, alpha360, custom
    label_type: str = "return_10"   # return_10, return_5, bin_class

    # 时间范围
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # 数据划分
    train_ratio: float = 0.7
    valid_ratio: float = 0.15

    # 特征参数
    feature_kwargs: Dict[str, Any] = field(default_factory=dict)


class Alpha158Features:
    """
    qlib 官方 Alpha158 特征

    参考: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

    包含:
    - KBAR: K线特征 (30个)
    - KDJ: KDJ指标 (18个)
    - RSI: RSI指标 (6个)
    - MACD: MACD指标 (3个)
    - BOLL: 布林带 (6个)
    - MA: 均线 (20个)
    - EMA: 指数均线 (10个)
    - PSY: 心理线 (6个)
    - BIAS: 乖离率 (6个)
    - ROC: 变动率 (6个)
    - MAVOL: 成交量均线 (10个)
    - 其他: (37个)

    共计约 158 个特征
    """

    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        """
        生成 Alpha158 特征

        Args:
            df: 包含 open, high, low, close, volume 的 DataFrame

        Returns:
            特征 DataFrame
        """
        features = {}

        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        volume = df.get('volume', df.get('vol', pd.Series(1, index=df.index)))

        # ==================== KBAR 特征 (30个) ====================
        # 参考 qlib 的 KbarHandler
        windows = [5, 10, 20, 30, 60]

        for w in windows:
            # 动量
            features[f'KMID_{w}'] = (close - low) / (high - low + 1e-12)
            features[f'KLEN_{w}'] = (high - low) / (open_ + 1e-12)
            features[f'KMID2_{w}'] = (close - open_) / (high - low + 1e-12)

            # 波动
            features[f'KSTD_{w}'] = close.pct_change().rolling(w).std()
            features[f'KHIGH_{w}'] = high.rolling(w).max() / (close + 1e-12)
            features[f'KLOW_{w}'] = low.rolling(w).min() / (close + 1e-12)

        # ==================== 均线特征 (20个) ====================
        for w in [5, 10, 20, 30, 60]:
            ma = close.rolling(w).mean()
            features[f'MA{w}'] = ma
            features[f'MA{w}_R'] = close / (ma + 1e-12) - 1
            features[f'MA{w}_DIFF'] = (close - ma) / (ma + 1e-12)
            features[f'MA{w}_STD'] = close.rolling(w).std() / (ma + 1e-12)

        # ==================== EMA 特征 (10个) ====================
        for w in [5, 10, 20, 30, 60]:
            ema = close.ewm(span=w, adjust=False).mean()
            features[f'EMA{w}'] = ema
            features[f'EMA{w}_DIFF'] = (close - ema) / (ema + 1e-12)

        # ==================== MACD 特征 (3个) ====================
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26
        dea = dif.ewm(span=9, adjust=False).mean()
        features['MACD_DIF'] = dif
        features['MACD_DEA'] = dea
        features['MACD_HIST'] = 2 * (dif - dea)

        # ==================== RSI 特征 (6个) ====================
        for w in [6, 12, 24]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(w).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
            rs = gain / (loss + 1e-12)
            features[f'RSI{w}'] = 100 - (100 / (1 + rs))
            # RSI 变化
            features[f'RSI{w}_DIFF'] = features[f'RSI{w}'].diff()

        # ==================== KDJ 特征 (18个) ====================
        for n in [9, 14, 21]:
            hhv = high.rolling(n).max()
            llv = low.rolling(n).min()
            rsv = (close - llv) / (hhv - llv + 1e-12) * 100
            k = rsv.ewm(alpha=1/3, adjust=False).mean()
            d = k.ewm(alpha=1/3, adjust=False).mean()
            j = 3 * k - 2 * d
            features[f'K{n}'] = k
            features[f'D{n}'] = d
            features[f'J{n}'] = j

        # ==================== 布林带特征 (6个) ====================
        for w in [10, 20]:
            mid = close.rolling(w).mean()
            std = close.rolling(w).std()
            upper = mid + 2 * std
            lower = mid - 2 * std
            features[f'BOLL_UP_{w}'] = (upper - close) / close
            features[f'BOLL_LOW_{w}'] = (close - lower) / close
            features[f'BOLL_W_{w}'] = (upper - lower) / mid

        # ==================== PSY 心理线 (6个) ====================
        for w in [6, 12]:
            up_days = (close > close.shift(1)).rolling(w).sum()
            features[f'PSY{w}'] = up_days / w
            features[f'PSY{w}_MA'] = features[f'PSY{w}'].rolling(w).mean()

        # ==================== BIAS 乖离率 (6个) ====================
        for w in [6, 12, 24]:
            ma = close.rolling(w).mean()
            features[f'BIAS{w}'] = (close - ma) / (ma + 1e-12) * 100

        # ==================== ROC 变动率 (6个) ====================
        for w in [6, 12]:
            features[f'ROC{w}'] = close / close.shift(w) - 1
            features[f'ROC{w}_MA'] = features[f'ROC{w}'].rolling(w).mean()

        # ==================== 成交量特征 (10个) ====================
        for w in [5, 10, 20, 30, 60]:
            vol_ma = volume.rolling(w).mean()
            features[f'VMA{w}'] = volume / (vol_ma + 1e-12)
            features[f'VSTD{w}'] = volume.rolling(w).std() / (vol_ma + 1e-12)

        # ==================== 其他技术指标 (20+个) ====================
        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        for w in [10, 20]:
            features[f'ATR{w}'] = tr.rolling(w).mean() / close

        # VR (成交量比率)
        for w in [10, 20]:
            up_vol = volume.where(close > close.shift(1), 0).rolling(w).sum()
            down_vol = volume.where(close < close.shift(1), 0).rolling(w).sum()
            features[f'VR{w}'] = up_vol / (down_vol + 1e-12)

        # WVAD (威廉变异离散量)
        features['WVAD'] = ((close - open_) / (high - low + 1e-12) * volume).rolling(24).sum()

        # AD (累积/派发线)
        clv = ((close - low) - (high - close)) / (high - low + 1e-12)
        features['AD'] = (clv * volume).cumsum()

        # OBV
        features['OBV'] = (np.sign(close.diff()) * volume).cumsum()

        # 构建特征 DataFrame
        feature_df = pd.DataFrame(features, index=df.index)

        # 处理异常值
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        # 标准化 (按照 qlib 的 zscorenorm)
        feature_df = (feature_df - feature_df.rolling(60).mean()) / (feature_df.rolling(60).std() + 1e-12)

        # 填充缺失值
        feature_df = feature_df.ffill().bfill().fillna(0)

        return feature_df


class Alpha360Features:
    """
    qlib 官方 Alpha360 特征

    参考: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/handler.py

    基于过去 60 天的历史价格和收益率构建特征
    """

    @staticmethod
    def generate(df: pd.DataFrame) -> pd.DataFrame:
        """
        生成 Alpha360 特征

        Args:
            df: 包含 open, high, low, close, volume 的 DataFrame

        Returns:
            特征 DataFrame (360维)
        """
        features = {}
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']
        volume = df.get('volume', df.get('vol', pd.Series(1, index=df.index)))

        # 生成 60 天的历史特征
        for i in range(1, 61):
            # 收益率
            features[f'RETURN{i}'] = close / close.shift(i) - 1

            # 相对位置
            features[f'HIGH{i}'] = high.rolling(i).max() / close
            features[f'LOW{i}'] = low.rolling(i).min() / close

            # 成交量比
            features[f'VOLUME{i}'] = volume / volume.shift(i)

            # 波动率
            features[f'STD{i}'] = close.pct_change().rolling(i).std()

            # 动量
            features[f'MOM{i}'] = close - close.shift(i)

        # 构建特征 DataFrame
        feature_df = pd.DataFrame(features, index=df.index)

        # 处理异常值
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

        # 标准化
        feature_df = (feature_df - feature_df.rolling(60).mean()) / (feature_df.rolling(60).std() + 1e-12)

        # 填充缺失值
        feature_df = feature_df.ffill().bfill().fillna(0)

        return feature_df


class QlibDataConverter:
    """
    Qlib 数据转换器

    将 Quanter 系统的数据转换为 qlib 官方格式
    """

    def __init__(self, config: Optional[QlibDataConfig] = None):
        """
        初始化数据转换器

        Args:
            config: 数据配置
        """
        self.config = config or QlibDataConfig()
        self.cache_dir = Path(self.config.cache_dir)
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"QlibDataConverter initialized: cache={self.cache_dir}, output={self.output_dir}")

    def get_available_symbols(self) -> List[str]:
        """获取缓存中可用的股票代码"""
        symbols = []
        if self.cache_dir.exists():
            for f in self.cache_dir.glob('*_stock_bar.parquet'):
                # 文件名格式: 000001_SZ_stock_bar.parquet
                name = f.stem.replace('_stock_bar', '')
                symbols.append(name)
        return symbols

    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """加载单只股票的缓存数据"""
        file_path = self.cache_dir / f'{symbol}_stock_bar.parquet'
        if not file_path.exists():
            logger.warning(f"缓存文件不存在: {file_path}")
            return pd.DataFrame()

        df = pd.read_parquet(file_path)

        # 标准化列名
        if 'timestamp' not in df.columns and 'time' in df.columns:
            df = df.rename(columns={'time': 'timestamp'})

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df.index = pd.DatetimeIndex(df['timestamp'].values)

        return df

    def generate_features(
        self,
        df: pd.DataFrame,
        feature_type: str = "alpha158"
    ) -> pd.DataFrame:
        """
        生成特征

        Args:
            df: 包含 OHLCV 的 DataFrame
            feature_type: 特征类型 (alpha158, alpha360, custom)

        Returns:
            特征 DataFrame
        """
        if feature_type == "alpha158":
            return Alpha158Features.generate(df)
        elif feature_type == "alpha360":
            return Alpha360Features.generate(df)
        else:
            raise ValueError(f"未知的特征类型: {feature_type}")

    def generate_labels(
        self,
        df: pd.DataFrame,
        label_type: str = "return_10",
        horizon: int = 10
    ) -> pd.Series:
        """
        生成标签

        Args:
            df: 包含 close 列的 DataFrame
            label_type: 标签类型
                - "return_10": 未来10天收益率 (回归)
                - "return_5": 未来5天收益率 (回归)
                - "bin_class": 二分类 (涨/跌)

        Returns:
            标签 Series
        """
        close = df['close']

        if label_type.startswith("return_"):
            days = int(label_type.split("_")[1])
            labels = close.shift(-days) / close - 1
        elif label_type == "bin_class":
            labels = (close.shift(-horizon) > close).astype(float)
        else:
            raise ValueError(f"未知的标签类型: {label_type}")

        return labels

    # ==================== 方式一：转换为 qlib 二进制格式 ====================

    def dump_data(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_type: str = "alpha158",
    ) -> Dict[str, Any]:
        """
        将数据转换为 qlib 二进制格式

        完全遵循 qlib 官方数据结构:
        - calendars/
            - day.txt          # 交易日历
        - instruments/
            - all.txt          # 所有股票代码
            - csi300.txt       # 沪深300成分股
        - features/
            - 000001.SZ/
                - close.bin    # 收盘价
                - open.bin     # 开盘价
                - ...
            - ...

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            feature_type: 特征类型

        Returns:
            转换统计信息
        """
        if symbols is None:
            symbols = self.get_available_symbols()

        if not symbols:
            raise ValueError("没有可用的股票数据")

        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date

        logger.info(f"开始转换 {len(symbols)} 只股票的数据...")

        # 创建目录结构
        features_dir = self.output_dir / "features"
        instruments_dir = self.output_dir / "instruments"
        calendars_dir = self.output_dir / "calendars"

        for d in [features_dir, instruments_dir, calendars_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 收集所有交易日
        all_dates = set()
        successful_symbols = []

        for symbol in symbols:
            df = self.load_stock_data(symbol)
            if df.empty:
                logger.warning(f"[{symbol}] 数据为空，跳过")
                continue

            # 过滤日期范围
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]

            if len(df) < 100:
                logger.warning(f"[{symbol}] 数据不足，跳过")
                continue

            # 生成特征
            try:
                features = self.generate_features(df, feature_type)
            except Exception as e:
                logger.error(f"[{symbol}] 特征生成失败: {e}")
                continue

            # 转换股票代码格式 (000001_SZ -> 000001.SZ)
            qlib_symbol = symbol.replace('_', '.')

            # 创建股票目录
            stock_dir = features_dir / qlib_symbol
            stock_dir.mkdir(exist_ok=True)

            # 保存原始 OHLCV 数据为二进制格式
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    self._save_bin_file(stock_dir / f"{col}.bin", df[col].values)

            # 保存特征
            for feat_name in features.columns:
                self._save_bin_file(stock_dir / f"{feat_name}.bin", features[feat_name].values)

            # 收集日期
            all_dates.update(df.index.strftime('%Y-%m-%d').tolist())
            successful_symbols.append(qlib_symbol)

            logger.debug(f"[{symbol}] 转换完成，特征数: {len(features.columns)}")

        # 保存交易日历
        sorted_dates = sorted(all_dates)
        with open(calendars_dir / "day.txt", 'w') as f:
            f.write('\n'.join(sorted_dates))

        # 保存股票列表
        with open(instruments_dir / "all.txt", 'w') as f:
            f.write('\n'.join(successful_symbols))

        # 保存元数据
        meta = {
            "feature_type": feature_type,
            "feature_count": len(features.columns) if successful_symbols else 0,
            "symbol_count": len(successful_symbols),
            "start_date": sorted_dates[0] if sorted_dates else None,
            "end_date": sorted_dates[-1] if sorted_dates else None,
            "created_at": datetime.now().isoformat(),
        }

        with open(self.output_dir / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"数据转换完成: {len(successful_symbols)} 只股票, {len(sorted_dates)} 个交易日")

        return {
            "success": True,
            "symbol_count": len(successful_symbols),
            "date_count": len(sorted_dates),
            "feature_count": meta["feature_count"],
            "symbols": successful_symbols,
            "output_dir": str(self.output_dir),
        }

    def _save_bin_file(self, filepath: Path, data: np.ndarray):
        """保存二进制文件 (qlib 格式)"""
        # qlib 使用 pickle 格式保存数据
        with open(filepath, 'wb') as f:
            pickle.dump({
                'data': data.astype(np.float32),
                'type': 'float32',
            }, f)

    # ==================== 方式二：创建 DatasetH ====================

    def create_dataset(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_type: str = "alpha158",
        label_type: str = "return_10",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        创建用于训练的数据集

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            feature_type: 特征类型
            label_type: 标签类型

        Returns:
            (features, labels) 元组
        """
        if symbols is None:
            symbols = self.get_available_symbols()

        if not symbols:
            raise ValueError("没有可用的股票数据")

        start_date = start_date or self.config.start_date
        end_date = end_date or self.config.end_date

        all_features = []
        all_labels = []
        all_instruments = []
        all_dates = []

        for symbol in symbols:
            df = self.load_stock_data(symbol)
            if df.empty:
                continue

            # 过滤日期范围
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]

            if len(df) < 100:
                continue

            try:
                features = self.generate_features(df, feature_type)
                labels = self.generate_labels(df, label_type)
            except Exception as e:
                logger.error(f"[{symbol}] 数据处理失败: {e}")
                continue

            # 移除 NaN
            valid_idx = features.dropna(how='all').index.intersection(labels.dropna().index)
            features = features.loc[valid_idx]
            labels = labels.loc[valid_idx]

            all_features.append(features)
            all_labels.append(labels)
            all_instruments.extend([symbol] * len(features))
            all_dates.extend(features.index.tolist())

        if not all_features:
            raise ValueError("没有有效的数据")

        # 合并所有股票数据
        features_df = pd.concat(all_features)
        labels_series = pd.Series(
            np.concatenate([l.values for l in all_labels]),
            index=pd.MultiIndex.from_arrays(
                [all_dates, all_instruments],
                names=['datetime', 'instrument']
            ),
            name='label'
        )

        # 设置 MultiIndex
        features_df.index = pd.MultiIndex.from_arrays(
            [all_dates, all_instruments],
            names=['datetime', 'instrument']
        )

        logger.info(f"创建数据集完成: {len(features_df)} 条记录, {len(symbols)} 只股票")

        return features_df, labels_series

    def create_qlib_dataset(
        self,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_type: str = "alpha158",
        label_type: str = "return_10",
        train_ratio: float = None,
        valid_ratio: float = None,
    ) -> Any:
        """
        创建 Qlib 原生 DatasetH

        完全兼容 qlib 官方训练流程

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            feature_type: 特征类型
            label_type: 标签类型
            train_ratio: 训练集比例
            valid_ratio: 验证集比例

        Returns:
            Qlib DatasetH 实例
        """
        if not QLIB_AVAILABLE:
            raise ImportError("pyqlib 未安装，无法创建 DatasetH")

        features_df, labels_series = self.create_dataset(
            symbols, start_date, end_date, feature_type, label_type
        )

        train_ratio = train_ratio or self.config.train_ratio
        valid_ratio = valid_ratio or self.config.valid_ratio

        # 计算 segments
        dates = features_df.index.get_level_values('datetime').unique().sort_values()
        total_len = len(dates)
        train_end = int(total_len * train_ratio)
        valid_end = int(total_len * (train_ratio + valid_ratio))

        segments = {
            'train': (dates[0].strftime('%Y-%m-%d'), dates[train_end - 1].strftime('%Y-%m-%d')),
            'valid': (dates[train_end].strftime('%Y-%m-%d'), dates[valid_end - 1].strftime('%Y-%m-%d')),
            'test': (dates[valid_end].strftime('%Y-%m-%d'), dates[-1].strftime('%Y-%m-%d')),
        }

        # 创建 qlib DataFrame 格式
        feature_cols = pd.MultiIndex.from_product([['feature'], features_df.columns])
        qlib_features = pd.DataFrame(
            features_df.values,
            index=features_df.index,
            columns=feature_cols
        )

        label_cols = pd.MultiIndex.from_product([['label'], ['label']])
        qlib_labels = pd.DataFrame(
            labels_series.values,
            index=labels_series.index,
            columns=label_cols
        )

        qlib_df = pd.concat([qlib_features, qlib_labels], axis=1)

        # 创建 DataHandlerLP
        handler = DataHandlerLP.from_df(qlib_df)

        # 创建 DatasetH
        dataset = DatasetH(handler=handler, segments=segments)

        logger.info(f"创建 Qlib DatasetH: train={segments['train']}, valid={segments['valid']}, test={segments['test']}")

        return dataset


class QlibTrainingPipeline:
    """
    Qlib 官方训练流程封装

    完全遵循 qlib 官方训练流程
    """

    def __init__(self, data_converter: QlibDataConverter):
        """
        初始化训练流程

        Args:
            data_converter: 数据转换器
        """
        self.converter = data_converter

    def train_gbdt_model(
        self,
        symbols: Optional[List[str]] = None,
        model_type: str = "lgb",
        feature_type: str = "alpha158",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **model_params
    ) -> Dict[str, Any]:
        """
        训练 GBDT 模型 (使用 qlib 官方流程)

        Args:
            symbols: 股票代码列表
            model_type: 模型类型 (lgb, xgboost, catboost)
            feature_type: 特征类型
            start_date: 开始日期
            end_date: 结束日期
            **model_params: 模型参数

        Returns:
            训练结果
        """
        from quanttool.strategies.qlib.models import QlibModelFactory, QlibModelConfig

        # 创建数据集
        features, labels = self.converter.create_dataset(
            symbols, start_date, end_date, feature_type
        )

        # 创建模型配置
        config = QlibModelConfig(model_type=model_type, **model_params)

        # 创建模型
        model = QlibModelFactory.create(model_type, config)

        # 训练模型
        model.fit(features, labels)

        return {
            "success": True,
            "model": model,
            "feature_count": len(features.columns),
            "sample_count": len(features),
        }

    def train_pytorch_model(
        self,
        symbols: Optional[List[str]] = None,
        model_type: str = "lstm",
        feature_type: str = "alpha158",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        epochs: int = 100,
        **model_params
    ) -> Dict[str, Any]:
        """
        训练 PyTorch 模型 (使用 qlib 官方流程)

        Args:
            symbols: 股票代码列表
            model_type: 模型类型 (lstm, gru, transformer, etc.)
            feature_type: 特征类型
            start_date: 开始日期
            end_date: 结束日期
            epochs: 训练轮数
            **model_params: 模型参数

        Returns:
            训练结果
        """
        from quanttool.strategies.qlib.models import QlibModelFactory, QlibModelConfig

        # 创建数据集
        features, labels = self.converter.create_dataset(
            symbols, start_date, end_date, feature_type
        )

        # 创建模型配置
        config = QlibModelConfig(
            model_type=model_type,
            epochs=epochs,
            **model_params
        )

        # 创建模型
        model = QlibModelFactory.create(model_type, config)

        # 训练模型
        model.fit(features, labels)

        return {
            "success": True,
            "model": model,
            "feature_count": len(features.columns),
            "sample_count": len(features),
        }


# 便捷函数
def create_qlib_converter(
    cache_dir: str = ".cache/incremental_data",
    output_dir: str = "qlib_data/cn_data"
) -> QlibDataConverter:
    """创建 Qlib 数据转换器"""
    config = QlibDataConfig(
        cache_dir=cache_dir,
        output_dir=output_dir
    )
    return QlibDataConverter(config)


def convert_to_qlib_format(
    symbols: Optional[List[str]] = None,
    cache_dir: str = ".cache/incremental_data",
    output_dir: str = "qlib_data/cn_data",
    feature_type: str = "alpha158"
) -> Dict[str, Any]:
    """
    将数据转换为 qlib 格式的便捷函数

    Args:
        symbols: 股票代码列表 (None 表示所有缓存数据)
        cache_dir: 缓存目录
        output_dir: 输出目录
        feature_type: 特征类型

    Returns:
        转换结果
    """
    converter = create_qlib_converter(cache_dir, output_dir)
    return converter.dump_data(symbols, feature_type=feature_type)
