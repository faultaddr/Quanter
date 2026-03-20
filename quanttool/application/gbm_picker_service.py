"""
GBM 沪深300 每日荐股服务

使用 LightGBM 模型对沪深300成分股进行预测，返回 top N 推荐股票

支持两种模型类型:
1. 自定义 sklearn 接口模型 (GBMStrategy)
2. qrun 训练的 qlib LGBModel
"""

import os
import pickle
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from ..core.logging import get_logger
from ..strategies.gbm_strategy import GBMStrategy, GBMConfig, GBMModel

logger = get_logger(__name__)

# 默认模型保存路径
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / "models" / "gbm"
# qrun 模型路径
MLRUNS_DIR = Path(__file__).parent.parent.parent / "mlruns"


class QlibModelAdapter:
    """
    qlib LGBModel 适配器

    将 qrun 训练的 qlib.contrib.model.gbdt.LGBModel 适配为统一接口
    """

    def __init__(self, model_path: str):
        """
        初始化适配器

        Args:
            model_path: 模型文件路径 (params.pkl)
        """
        with open(model_path, 'rb') as f:
            self._qlib_model = pickle.load(f)

        # 获取内部 Booster
        self._booster = self._qlib_model.model
        self._feature_count = len(self._booster.feature_name())
        logger.info(f"加载 qlib LGBModel: {model_path}, 特征数: {self._feature_count}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        预测

        Args:
            X: 特征 DataFrame

        Returns:
            预测值数组
        """
        # qlib 模型特征名是 Column_0, Column_1, ...
        # 需要重命名特征列
        if X.shape[1] == self._feature_count:
            X_renamed = X.copy()
            X_renamed.columns = [f"Column_{i}" for i in range(self._feature_count)]
            return self._booster.predict(X_renamed)
        else:
            raise ValueError(f"特征数不匹配: 输入 {X.shape[1]}, 模型需要 {self._feature_count}")

    @property
    def feature_count(self) -> int:
        """返回特征数量"""
        return self._feature_count


def detect_model_type(model_path: str) -> str:
    """
    检测模型类型

    Args:
        model_path: 模型文件路径

    Returns:
        'sklearn' 或 'qlib'
    """
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)

        # 检查是否是 qlib LGBModel
        if hasattr(data, 'predict') and hasattr(data, 'model'):
            # qlib LGBModel 有 model 属性
            inner_model = getattr(data, 'model', None)
            if inner_model is not None and hasattr(inner_model, 'feature_name'):
                return 'qlib'

        # 检查是否是自定义 sklearn 模型格式
        if isinstance(data, dict) and 'model' in data:
            return 'sklearn'

        # 旧格式: 直接是 LGBMRegressor
        if hasattr(data, 'feature_importances_'):
            return 'sklearn'

        logger.warning(f"无法识别模型类型: {model_path}")
        return 'unknown'

    except Exception as e:
        logger.error(f"检测模型类型失败: {e}")
        return 'unknown'


def find_latest_qrun_model() -> Optional[str]:
    """
    查找最新的 qrun 模型

    Returns:
        模型路径或 None
    """
    models = list_all_qrun_models()
    if not models:
        return None
    # 返回最新的模型路径
    latest = max(models, key=lambda x: x['modified'])
    logger.info(f"找到 qrun 模型: {latest['path']}")
    return latest['path']


def list_all_qrun_models() -> List[Dict[str, Any]]:
    """
    列出所有 qrun 训练的模型

    Returns:
        模型信息列表，按修改时间降序排列
    """
    if not MLRUNS_DIR.exists():
        return []

    models = []

    # 查找所有 params.pkl 文件
    # 结构: mlruns/{experiment_id}/{run_id}/artifacts/params.pkl
    for params_file in MLRUNS_DIR.glob("*/*/artifacts/params.pkl"):
        try:
            stat = params_file.stat()
            run_dir = params_file.parent.parent  # run 目录
            experiment_id = run_dir.parent.name  # experiment_id
            run_id = run_dir.name  # run_id

            # 读取 meta.yaml 获取运行名称和时间
            meta_file = run_dir / "meta.yaml"
            run_name = ""
            start_time = None

            if meta_file.exists():
                try:
                    import yaml
                    with open(meta_file, 'r') as f:
                        meta = yaml.safe_load(f)
                        run_name = meta.get('run_name', '')
                        start_time = meta.get('start_time', None)
                except Exception:
                    pass

            # 检测模型类型
            model_type = detect_model_type(str(params_file))

            # 获取模型配置信息
            config_info = _get_model_config(str(params_file))

            models.append({
                'path': str(params_file),
                'experiment_id': experiment_id,
                'run_id': run_id,
                'run_name': run_name,
                'model_type': model_type,
                'size_mb': round(stat.st_size / 1024 / 1024, 2),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'modified_timestamp': stat.st_mtime,
                'config': config_info,
            })

        except Exception as e:
            logger.debug(f"解析模型信息失败 [{params_file}]: {e}")
            continue

    # 按修改时间降序排列
    models.sort(key=lambda x: x['modified_timestamp'], reverse=True)

    return models


def _get_model_config(model_path: str) -> Dict[str, Any]:
    """
    获取模型配置信息

    Args:
        model_path: 模型路径

    Returns:
        配置信息字典
    """
    config = {}
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # qlib LGBModel
        if hasattr(model, 'params'):
            params = model.params
            config['n_estimators'] = params.get('n_estimators', 'N/A')
            config['max_depth'] = params.get('max_depth', 'N/A')
            config['learning_rate'] = params.get('learning_rate', 'N/A')
            config['num_leaves'] = params.get('num_leaves', 'N/A')

        # 获取特征数量
        if hasattr(model, 'model') and hasattr(model.model, 'feature_name'):
            config['feature_count'] = len(model.model.feature_name())

    except Exception as e:
        logger.debug(f"获取模型配置失败: {e}")

    return config


@dataclass
class StockRecommendation:
    """股票推荐结果"""
    code: str                    # 股票代码
    name: str = ""               # 股票名称
    pred_return: float = 0.0     # 预测收益率
    probability: float = 0.0     # 上涨概率
    percentile: float = 0.0      # 百分位排名
    confidence: float = 0.0      # 置信度
    signal: str = 'hold'         # 信号类型
    stop_loss: Optional[float] = None   # 止损价
    take_profit: Optional[float] = None # 止盈价
    close: Optional[float] = None       # 最新收盘价
    trend_warning: List[str] = field(default_factory=list)  # 趋势警告


@dataclass
class DailyPickResult:
    """每日荐股结果"""
    date: str                           # 日期
    total_stocks: int = 0               # 总股票数
    valid_stocks: int = 0               # 有效预测数
    top_stocks: List[StockRecommendation] = field(default_factory=list)
    model_info: Dict[str, Any] = field(default_factory=dict)


class GBMCsi300Picker:
    """
    GBM 沪深300 每日荐股器

    功能:
    1. 从 qlib 数据获取沪深300成分股
    2. 加载或训练 GBM 模型（支持 qrun 模型）
    3. 对每只股票进行预测
    4. 返回 top N 推荐股票
    """

    # qlib 数据路径
    QLIB_DATA_PATH = Path.home() / ".qlib" / "qlib_data" / "cn_data"

    def __init__(
        self,
        model_dir: Optional[Path] = None,
        use_cache: bool = True,
        top_n: int = 10,
        model_path: Optional[str] = None
    ):
        """
        初始化荐股器

        Args:
            model_dir: 模型保存目录
            use_cache: 是否使用缓存
            top_n: 返回前 N 只推荐股票
            model_path: 指定模型路径（支持 qrun 模型）
        """
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.use_cache = use_cache
        self.top_n = top_n
        self.model_path = model_path

        self._strategy: Optional[GBMStrategy] = None
        self._qlib_adapter: Optional[QlibModelAdapter] = None
        self._model_type: str = 'sklearn'  # 'sklearn' 或 'qlib'
        self._data_loader = None
        self._model_loaded = False
        self._use_ashare = False  # 是否使用 Ashare 数据源

    def _init_data_loader(self):
        """初始化数据加载器"""
        if self._data_loader is None:
            # 优先尝试 qlib
            try:
                from ..infrastructure.data_providers.qlib_data_loader import QlibDataLoader
                self._data_loader = QlibDataLoader()
                if self._data_loader.init_qlib():
                    logger.info("使用 Qlib 数据源")
                    return
            except Exception as e:
                logger.debug(f"Qlib 不可用: {e}")

            # 回退到 Ashare
            logger.info("回退到 Ashare 数据源")
            self._use_ashare = True

    def get_csi300_stocks(self, active_only: bool = True) -> List[str]:
        """
        获取沪深300成分股

        Args:
            active_only: 是否只返回当前仍在指数中的股票

        Returns:
            股票代码列表 (qlib 格式，如 SH600000)
        """
        instruments_file = self.QLIB_DATA_PATH / "instruments" / "csi300.txt"

        if not instruments_file.exists():
            logger.error(f"沪深300成分股文件不存在: {instruments_file}")
            return []

        # 使用字典存储每只股票的最新记录
        # key: 股票代码, value: (end_date, 完整行)
        stock_records: Dict[str, Tuple[str, str]] = {}

        with open(instruments_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    code = parts[0]
                    end_date = parts[2]

                    # 只保留最新的记录
                    if code not in stock_records or end_date > stock_records[code][0]:
                        stock_records[code] = (end_date, line.strip())

        if not active_only:
            # 返回所有股票
            stocks = list(stock_records.keys())
        else:
            # 获取最晚的结束日期
            max_end_date = max(r[0] for r in stock_records.values())
            logger.info(f"数据中最晚日期: {max_end_date}")

            # 返回结束日期等于最晚日期的股票（即最新成分股）
            stocks = [code for code, (end_date, _) in stock_records.items()
                     if end_date == max_end_date]

        logger.info(f"获取沪深300成分股: {len(stocks)} 只")
        return sorted(stocks)

    def load_or_train_model(
        self,
        force_train: bool = False,
        start_date: str = "2017-01-01",
        end_date: str = "2026-12-31"
    ) -> bool:
        """
        加载或训练模型

        支持两种模型类型:
        1. qrun 训练的 qlib LGBModel (优先)
        2. 自定义 sklearn 接口模型

        Args:
            force_train: 是否强制重新训练
            start_date: 训练数据开始日期
            end_date: 训练数据结束日期

        Returns:
            是否成功
        """
        if force_train:
            # 强制训练，跳过加载
            return self._train_new_model(start_date, end_date)

        # 1. 优先使用指定模型路径
        if self.model_path:
            return self._load_model_by_path(self.model_path)

        # 2. 尝试加载 qrun 模型
        qrun_model = find_latest_qrun_model()
        if qrun_model:
            if self._load_model_by_path(qrun_model):
                return True

        # 3. 尝试加载自定义 sklearn 模型
        model_files = list(self.model_dir.glob("lgbm_*.pkl"))
        if model_files:
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            if self._load_model_by_path(str(latest_model)):
                return True

        # 4. 没有可用模型，训练新模型
        return self._train_new_model(start_date, end_date)

    def _load_model_by_path(self, model_path: str) -> bool:
        """
        根据路径加载模型

        Args:
            model_path: 模型文件路径

        Returns:
            是否成功
        """
        model_type = detect_model_type(model_path)

        if model_type == 'qlib':
            try:
                self._qlib_adapter = QlibModelAdapter(model_path)
                self._model_type = 'qlib'
                self._model_loaded = True
                logger.info(f"加载 qlib 模型成功: {model_path}")
                return True
            except Exception as e:
                logger.warning(f"加载 qlib 模型失败: {e}")
                return False

        elif model_type == 'sklearn':
            try:
                self._strategy = GBMStrategy()
                self._strategy.load_model(model_path)
                self._model_type = 'sklearn'
                self._model_loaded = True
                logger.info(f"加载 sklearn 模型成功: {model_path}")
                return True
            except Exception as e:
                logger.warning(f"加载 sklearn 模型失败: {e}")
                return False

        else:
            logger.error(f"无法识别模型类型: {model_path}")
            return False

    def _train_new_model(
        self,
        start_date: str = "2017-01-01",
        end_date: str = "2026-12-31"
    ) -> bool:
        """
        训练新模型

        Args:
            start_date: 训练数据开始日期
            end_date: 训练数据结束日期

        Returns:
            是否成功
        """
        logger.info("开始训练 GBM 模型...")
        self._init_data_loader()

        # 获取沪深300成分股
        stocks = self.get_csi300_stocks(active_only=False)
        if not stocks:
            logger.error("无法获取沪深300成分股")
            return False

        # 训练模型
        self._strategy = GBMStrategy(GBMConfig())

        try:
            result = self._strategy.train(
                instruments=stocks,
                start_date=start_date,
                end_date=end_date
            )

            if result.get('success'):
                # 保存模型
                self.model_dir.mkdir(parents=True, exist_ok=True)
                model_path = self.model_dir / f"lgbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                self._strategy.save_model(str(model_path))
                self._model_type = 'sklearn'
                self._model_loaded = True
                logger.info(f"模型训练成功，已保存: {model_path}")
                return True
            else:
                logger.error("模型训练失败")
                return False

        except Exception as e:
            logger.error(f"模型训练异常: {e}")
            return False

    def predict_stocks(
        self,
        stocks: Optional[List[str]] = None,
        min_data_days: int = 120
    ) -> List[StockRecommendation]:
        """
        预测股票

        Args:
            stocks: 股票列表，默认使用沪深300成分股
            min_data_days: 最小数据天数

        Returns:
            预测结果列表
        """
        if not self._model_loaded:
            if not self.load_or_train_model():
                raise RuntimeError("模型未加载")

        self._init_data_loader()

        # 获取股票列表
        if stocks is None:
            stocks = self.get_csi300_stocks(active_only=True)

        results = []

        # 根据模型类型选择预测方式
        if self._model_type == 'qlib':
            results = self._predict_with_qlib_model(stocks, min_data_days)
        else:
            results = self._predict_with_sklearn_model(stocks, min_data_days)

        # 按百分位排序
        results.sort(key=lambda x: x.percentile, reverse=True)

        return results

    def _predict_with_qlib_model(
        self,
        stocks: List[str],
        min_data_days: int = 120,
        realtime_days: int = 360  # 使用最近 360 天实时数据
    ) -> List[StockRecommendation]:
        """
        使用 qlib 模型预测

        使用实时价格数据（Ashare）而非 qlib 历史数据进行预测

        Args:
            stocks: 股票列表
            min_data_days: 最小数据天数
            realtime_days: 实时数据天数，默认 360 天

        Returns:
            预测结果列表
        """
        results = []
        all_predictions = []  # 收集所有预测值用于计算百分位

        logger.info(f"使用 Ashare 实时数据（最近 {realtime_days} 天）进行预测")

        # 第一轮：收集所有预测值
        for i, stock in enumerate(stocks):
            if (i + 1) % 50 == 0:
                logger.info(f"预测进度: {i + 1}/{len(stocks)}")

            try:
                # 强制使用 Ashare 实时数据，而非 qlib 数据
                df = self._load_stock_data_ashare(stock, days=realtime_days)
                if df is None or len(df) < min_data_days:
                    logger.debug(f"数据不足 [{stock}]: {len(df) if df is not None else 0} 天")
                    continue

                # 生成 Alpha158 特征
                features = self._generate_alpha158_features(df)
                if features is None or features.empty:
                    logger.debug(f"特征生成失败 [{stock}]")
                    continue

                # 使用 qlib 模型预测
                pred_return = self._qlib_adapter.predict(features.iloc[[-1]])[0]

                all_predictions.append({
                    'stock': stock,
                    'pred_return': pred_return,
                    'close': df['close'].iloc[-1],
                    'features': features,
                    'df': df
                })

            except Exception as e:
                logger.debug(f"预测失败 [{stock}]: {e}")
                continue

        if not all_predictions:
            return results

        # 计算百分位
        pred_returns = [p['pred_return'] for p in all_predictions]
        for pred in all_predictions:
            pred_value = pred['pred_return']
            # 计算百分位：比当前值小的比例
            percentile = sum(1 for r in pred_returns if r <= pred_value) / len(pred_returns)

            # 计算置信度（基于预测值的绝对值）
            confidence = min(1.0, abs(pred_value) * 10)

            # 生成信号
            signal = 'hold'
            if percentile >= 0.75:
                signal = 'buy'
            elif percentile <= 0.25:
                signal = 'sell'

            # 计算止损止盈
            close = pred['close']
            stop_loss = close * 0.95 if close else None
            take_profit = close * 1.10 if close else None

            rec = StockRecommendation(
                code=pred['stock'],
                pred_return=float(pred_value),
                probability=1 / (1 + np.exp(-10 * pred_value)),  # sigmoid 转换
                percentile=percentile,
                confidence=confidence,
                signal=signal,
                stop_loss=stop_loss,
                take_profit=take_profit,
                close=close
            )

            # 只保留买入信号或高分位的股票
            if rec.percentile >= 0.5 or rec.signal == 'buy':
                results.append(rec)

        return results

    def _predict_with_sklearn_model(
        self,
        stocks: List[str],
        min_data_days: int = 120
    ) -> List[StockRecommendation]:
        """
        使用 sklearn 模型预测

        Args:
            stocks: 股票列表
            min_data_days: 最小数据天数

        Returns:
            预测结果列表
        """
        results = []

        for i, stock in enumerate(stocks):
            if (i + 1) % 50 == 0:
                logger.info(f"预测进度: {i + 1}/{len(stocks)}")

            try:
                df = self._get_stock_data(stock)
                if df is None or len(df) < min_data_days:
                    continue

                # 确保有必要的列
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                if not all(col in df.columns for col in required_cols):
                    continue

                # 生成信号
                current_bar = df.iloc[-1].to_dict()
                current_bar['timestamp'] = df.index[-1]

                signal = self._strategy.get_signal(
                    current_bar=current_bar,
                    historical_bars=df
                )

                if signal.get('error'):
                    continue

                # 创建推荐结果
                rec = StockRecommendation(
                    code=stock,
                    pred_return=signal.get('pred_return', 0),
                    probability=signal.get('probability', 0),
                    percentile=signal.get('percentile', 0),
                    confidence=signal.get('confidence', 0),
                    signal=signal.get('signal', 'hold'),
                    stop_loss=signal.get('stop_loss'),
                    take_profit=signal.get('take_profit'),
                    close=df['close'].iloc[-1]
                )

                # 只保留买入信号或高分位的股票
                if rec.percentile >= 0.5 or rec.signal == 'buy':
                    results.append(rec)

            except Exception as e:
                logger.debug(f"预测失败 [{stock}]: {e}")
                continue

        return results

    def _get_stock_data(self, stock: str, start_date: str = "2024-01-01") -> Optional[pd.DataFrame]:
        """
        获取股票数据

        Args:
            stock: 股票代码
            start_date: 开始日期

        Returns:
            DataFrame 或 None
        """
        if self._use_ashare:
            return self._load_stock_data_ashare(stock)
        else:
            return self._data_loader.load_stock_data(
                instrument=stock,
                start_date=start_date,
                use_adjclose=True
            )

    def _generate_alpha158_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        生成 Alpha158 特征

        Args:
            df: 股票 OHLCV 数据

        Returns:
            特征 DataFrame 或 None
        """
        try:
            from ..infrastructure.data_providers.qlib_data_converter import Alpha158Features
            generator = Alpha158Features()
            features = generator.generate(df)
            return features
        except Exception as e:
            logger.debug(f"生成 Alpha158 特征失败: {e}")
            return None

    def _load_stock_data_ashare(self, stock: str, days: int = 500) -> Optional[pd.DataFrame]:
        """
        使用 Ashare 加载股票数据

        Args:
            stock: 股票代码 (qlib 格式，如 SH600000)
            days: 加载天数

        Returns:
            DataFrame 或 None
        """
        try:
            from ..infrastructure.data_providers.data_fetcher import AshareFetcher

            # 转换股票代码格式: SH600000 -> sh600000
            code = stock.lower()
            if code.startswith('sh'):
                code = code[2:] + '.SH'
            elif code.startswith('sz'):
                code = code[2:] + '.SZ'
            else:
                code = code[2:] + '.SH' if stock.startswith('6') else code[2:] + '.SZ'

            # 获取数据
            df = AshareFetcher.get_price(code=code, count=days + 50, frequency='1d')

            if df.empty:
                return None

            # 重命名列
            if 'time' in df.columns:
                df = df.rename(columns={'time': 'timestamp'})

            # 设置日期索引
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')

            df = df.sort_index()

            # 只保留需要的列
            cols = ['open', 'high', 'low', 'close', 'volume']
            df = df[[c for c in cols if c in df.columns]]

            # 取最近 days 天
            if len(df) > days:
                df = df.tail(days)

            return df

        except Exception as e:
            logger.debug(f"Ashare 加载数据失败 [{stock}]: {e}")
            return None

    def get_daily_picks(
        self,
        top_n: Optional[int] = None,
        force_train: bool = False
    ) -> DailyPickResult:
        """
        获取每日推荐股票

        Args:
            top_n: 返回前 N 只，默认使用初始化时的设置
            force_train: 是否强制重新训练模型

        Returns:
            每日荐股结果
        """
        top_n = top_n or self.top_n

        # 确保模型已加载
        if not self._model_loaded:
            if not self.load_or_train_model(force_train=force_train):
                raise RuntimeError("模型加载失败")

        # 获取沪深300成分股
        stocks = self.get_csi300_stocks(active_only=True)
        total_stocks = len(stocks)

        # 预测
        predictions = self.predict_stocks(stocks)

        # 获取 top N
        top_picks = predictions[:top_n]

        return DailyPickResult(
            date=datetime.now().strftime("%Y-%m-%d"),
            total_stocks=total_stocks,
            valid_stocks=len(predictions),
            top_stocks=top_picks,
            model_info={
                'feature_type': self._strategy.config.feature_type if self._strategy else 'unknown',
                'model_type': 'LightGBM',
            }
        )


def format_pick_report(result: DailyPickResult) -> str:
    """
    格式化荐股报告

    Args:
        result: 荐股结果

    Returns:
        格式化的报告字符串
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"GBM 沪深300 每日荐股 ({result.date})")
    lines.append("=" * 60)
    lines.append(f"扫描股票: {result.total_stocks} 只")
    lines.append(f"有效预测: {result.valid_stocks} 只")
    lines.append("")

    if not result.top_stocks:
        lines.append("暂无推荐股票")
        return "\n".join(lines)

    lines.append(f"Top {len(result.top_stocks)} 推荐股票:")
    lines.append("-" * 60)

    for i, rec in enumerate(result.top_stocks, 1):
        lines.append(f"\n{i}. {rec.code}")
        lines.append(f"   预测收益: {rec.pred_return:.2%}")
        lines.append(f"   百分位:   {rec.percentile:.1%}")
        lines.append(f"   置信度:   {rec.confidence:.2%}")

        if rec.close:
            lines.append(f"   最新价:   {rec.close:.2f}")
        if rec.stop_loss:
            lines.append(f"   止损价:   {rec.stop_loss:.2f}")
        if rec.take_profit:
            lines.append(f"   止盈价:   {rec.take_profit:.2f}")

        if rec.trend_warning:
            lines.append(f"   ⚠️ 趋势警告: {', '.join(rec.trend_warning)}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("提示: 以上仅供参考，不构成投资建议")

    return "\n".join(lines)


# 便捷函数
def get_csi300_daily_picks(top_n: int = 10, force_train: bool = False) -> DailyPickResult:
    """
    获取沪深300每日荐股

    Args:
        top_n: 返回前 N 只
        force_train: 是否强制重新训练

    Returns:
        荐股结果
    """
    picker = GBMCsi300Picker(top_n=top_n)
    return picker.get_daily_picks(force_train=force_train)


if __name__ == "__main__":
    # 测试
    result = get_csi300_daily_picks(top_n=10)
    print(format_pick_report(result))
