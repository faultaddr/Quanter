"""
GBDT 系列模型

支持:
- LightGBM (LGBModel)
- XGBoost (XGBModel)
- CatBoost (CatBoostModel)
- DoubleEnsemble (DEnsembleModel)

优先使用 Qlib 原生模型，失败时回退到 sklearn 接口
"""

import warnings
from typing import Optional, Union
import numpy as np
import pandas as pd

from .models import QlibModelBase, QlibModelConfig, QLIB_AVAILABLE
from .data_adapter import create_qlib_compatible_dataset, SimpleDatasetH, QLIB_AVAILABLE as DATA_QLIB_AVAILABLE
from ...core.logging import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore')


def _ensure_qlib_initialized():
    """确保 qlib 已初始化"""
    if QLIB_AVAILABLE:
        try:
            import qlib
            # 检查是否已初始化
            if not hasattr(qlib, '_initialized') or not qlib._initialized:
                qlib.init()
        except Exception:
            pass


class CatBoostModelWrapper(QlibModelBase):
    """CatBoost 模型封装 - 优先使用 Qlib 原生模型"""

    def __init__(self, config: Optional[QlibModelConfig] = None):
        super().__init__(config)
        self.config.model_type = 'catboost'
        self._use_qlib = False
        self._qlib_model = None
        self._init_model()

    def _init_model(self):
        """初始化模型，优先使用 Qlib 原生模型"""
        if QLIB_AVAILABLE:
            try:
                from qlib.contrib.model.catboost_model import CatBoostModel
                self._qlib_model = CatBoostModel(
                    loss='Logloss',
                    iterations=self.config.n_estimators,
                    depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                )
                self._use_qlib = True
                logger.info("使用 Qlib 原生 CatBoostModel")
                return
            except Exception as e:
                logger.warning(f"Qlib CatBoostModel 初始化失败: {e}")

        self._init_sklearn_fallback()

    def _init_sklearn_fallback(self):
        """初始化 sklearn 接口作为后备"""
        try:
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor(
                iterations=self.config.n_estimators,
                depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                verbose=self.config.verbose,
            )
            self._use_qlib = False
            logger.info("使用 catboost 库 CatBoostRegressor")
        except ImportError:
            raise ImportError("请安装 catboost: pip install catboost")

    def fit(self, X: Union[pd.DataFrame, SimpleDatasetH], y: Optional[pd.Series] = None, **kwargs) -> 'CatBoostModelWrapper':
        self.feature_names_ = []

        # 如果是 DataFrame，创建 Qlib 兼容的数据集
        if isinstance(X, pd.DataFrame) and y is not None:
            self.feature_names_ = list(X.columns)
            dataset = create_qlib_compatible_dataset(X, y, use_native=self._use_qlib)
        elif isinstance(X, SimpleDatasetH):
            self.feature_names_ = list(X.features.columns)
            dataset = X
        else:
            raise ValueError("X 必须是 DataFrame 或 SimpleDatasetH")

        if self._use_qlib:
            try:
                # 确保 qlib 已初始化
                _ensure_qlib_initialized()
                # Qlib 原生模型训练
                self._qlib_model.fit(
                    dataset,
                    num_boost_round=self.config.n_estimators,
                    early_stopping_rounds=50,
                    verbose_eval=False,
                )
                self.model = self._qlib_model
                logger.info(f"Qlib 原生 CatBoost 模型训练完成，特征数: {len(self.feature_names_)}")
            except Exception as e:
                logger.warning(f"Qlib 原生模型训练失败: {e}，使用 sklearn 接口")
                self._use_qlib = False
                self._init_sklearn_fallback()
                # 从 dataset 获取数据
                df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
                X_train = df_train["feature"]
                y_train = df_train["label"].values.ravel()
                self.model.fit(X_train, y_train)
                logger.info(f"CatBoost 模型训练完成 (sklearn)，特征数: {len(self.feature_names_)}")
        else:
            # sklearn 接口训练
            df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
            X_train = df_train["feature"]
            y_train = df_train["label"].values.ravel()
            self.model.fit(X_train, y_train)
            logger.info(f"CatBoost 模型训练完成，特征数: {len(self.feature_names_)}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        if self._use_qlib and hasattr(self._qlib_model, 'model') and self._qlib_model.model is not None:
            pred = self._qlib_model.model.predict(X.values)
            return pred.ravel() if len(pred.shape) > 1 else pred

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测上涨概率（基于收益率预测值）"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        # 回归器预测收益率，转为上涨概率
        pred = self.predict(X)
        # 使用 sigmoid 将收益率映射到概率
        prob = 1 / (1 + np.exp(-10 * pred))
        return np.clip(prob, 0, 1)


class LGBModelWrapper(QlibModelBase):
    """LightGBM 模型封装 - 优先使用 Qlib 原生模型"""

    def __init__(self, config: Optional[QlibModelConfig] = None):
        super().__init__(config)
        self.config.model_type = 'lgb'
        self._use_qlib = False
        self._qlib_model = None
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        if QLIB_AVAILABLE:
            try:
                from qlib.contrib.model.gbdt import LGBModel
                self._qlib_model = LGBModel(
                    loss='binary',
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    num_leaves=self.config.num_leaves,
                    min_child_samples=self.config.min_child_samples,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                )
                self._use_qlib = True
                logger.info("使用 Qlib 原生 LGBModel")
                return
            except Exception as e:
                logger.warning(f"Qlib LGBModel 初始化失败: {e}")

        self._init_sklearn_fallback()

    def _init_sklearn_fallback(self):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                min_child_samples=self.config.min_child_samples,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbose=self.config.verbose,
            )
            self._use_qlib = False
            logger.info("使用 lightgbm 库 LGBMRegressor")
        except ImportError:
            raise ImportError("请安装 lightgbm: pip install lightgbm")

    def fit(self, X: Union[pd.DataFrame, SimpleDatasetH], y: Optional[pd.Series] = None, **kwargs) -> 'LGBModelWrapper':
        self.feature_names_ = []

        if isinstance(X, pd.DataFrame) and y is not None:
            self.feature_names_ = list(X.columns)
            dataset = create_qlib_compatible_dataset(X, y, use_native=self._use_qlib)
        elif isinstance(X, SimpleDatasetH):
            self.feature_names_ = list(X.features.columns)
            dataset = X
        else:
            raise ValueError("X 必须是 DataFrame 或 SimpleDatasetH")

        if self._use_qlib:
            try:
                # 确保 qlib 已初始化
                _ensure_qlib_initialized()
                self._qlib_model.fit(dataset)
                self.model = self._qlib_model
                logger.info(f"Qlib 原生 LGBModel 训练完成，特征数: {len(self.feature_names_)}")
            except Exception as e:
                logger.warning(f"Qlib 原生模型训练失败: {e}，使用 sklearn 接口")
                self._use_qlib = False
                self._init_sklearn_fallback()
                df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
                X_train = df_train["feature"]
                y_train = df_train["label"].values.ravel()
                self.model.fit(X_train, y_train)
        else:
            df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
            X_train = df_train["feature"]
            y_train = df_train["label"].values.ravel()
            self.model.fit(X_train, y_train)
            logger.info(f"LightGBM 模型训练完成，特征数: {len(self.feature_names_)}")

        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        if self._use_qlib and hasattr(self._qlib_model, 'model') and self._qlib_model.model is not None:
            pred = self._qlib_model.model.predict(X.values)
            return pred.ravel() if len(pred.shape) > 1 else pred

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测上涨概率（基于收益率预测值）"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        # 回归器预测收益率，转为上涨概率
        pred = self.predict(X)
        # 使用 sigmoid 将收益率映射到概率
        # 假设收益率在 [-0.2, 0.2] 范围内，使用 scaled sigmoid
        prob = 1 / (1 + np.exp(-10 * pred))  # 10x scale for better sensitivity
        return np.clip(prob, 0, 1)


class XGBModelWrapper(QlibModelBase):
    """XGBoost 模型封装 - 使用 Qlib 原生模型"""

    def __init__(self, config: Optional[QlibModelConfig] = None):
        super().__init__(config)
        self.config.model_type = 'xgboost'
        self._use_qlib = False
        self._qlib_model = None
        self._dataset = None  # 保存训练时的 dataset 用于预测
        self._init_model()

    def _init_model(self):
        if QLIB_AVAILABLE:
            try:
                from qlib.contrib.model.xgboost import XGBModel
                self._qlib_model = XGBModel(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    subsample=self.config.subsample,
                    colsample_bytree=self.config.colsample_bytree,
                    random_state=self.config.random_state,
                    n_jobs=self.config.n_jobs,
                )
                self._use_qlib = True
                logger.info("使用 Qlib 原生 XGBModel")
                return
            except Exception as e:
                logger.warning(f"Qlib XGBModel 初始化失败: {e}")

        self._init_sklearn_fallback()

    def _init_sklearn_fallback(self):
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                subsample=self.config.subsample,
                colsample_bytree=self.config.colsample_bytree,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
                verbosity=self.config.verbose,
            )
            self._use_qlib = False
            logger.info("使用 xgboost 库 XGBRegressor")
        except ImportError:
            raise ImportError("请安装 xgboost: pip install xgboost")

    def fit(self, X: Union[pd.DataFrame, SimpleDatasetH], y: Optional[pd.Series] = None, **kwargs) -> 'XGBModelWrapper':
        self.feature_names_ = []

        if isinstance(X, pd.DataFrame) and y is not None:
            self.feature_names_ = list(X.columns)
            dataset = create_qlib_compatible_dataset(X, y, use_native=self._use_qlib)
        elif isinstance(X, SimpleDatasetH):
            self.feature_names_ = list(X.features.columns)
            dataset = X
        else:
            raise ValueError("X 必须是 DataFrame 或 SimpleDatasetH")

        if self._use_qlib:
            try:
                # 确保 qlib 已初始化
                _ensure_qlib_initialized()
                self._qlib_model.fit(dataset)
                self.model = self._qlib_model
                self._dataset = dataset  # 保存用于预测
                logger.info(f"Qlib 原生 XGBModel 训练完成，特征数: {len(self.feature_names_)}")
            except Exception as e:
                logger.warning(f"Qlib 原生模型训练失败: {e}，使用 sklearn 接口")
                self._use_qlib = False
                self._init_sklearn_fallback()
                df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
                X_train = df_train["feature"]
                y_train = df_train["label"].values.ravel()
                self.model.fit(X_train, y_train)
        else:
            df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
            X_train = df_train["feature"]
            y_train = df_train["label"].values.ravel()
            self.model.fit(X_train, y_train)
            logger.info(f"XGBoost 模型训练完成，特征数: {len(self.feature_names_)}")

        self.is_fitted = True
        return self

    def _create_predict_dataset(self, X: pd.DataFrame):
        """为预测创建 DatasetH 格式的数据"""
        from .data_adapter import SimpleDatasetH
        import pandas as pd

        # 创建假的标签（预测时不需要真实标签）
        dummy_labels = pd.Series([0] * len(X), index=X.index)

        # 创建临时 dataset
        dataset = SimpleDatasetH(
            features=X,
            labels=dummy_labels,
            segments={'test': (X.index[0].strftime('%Y-%m-%d'), X.index[-1].strftime('%Y-%m-%d'))},
            instrument='stock'
        )
        return dataset

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        if self._use_qlib:
            # 使用 Qlib 原生预测（需要 DatasetH 格式）
            try:
                pred_dataset = self._create_predict_dataset(X)
                predictions = self._qlib_model.predict(pred_dataset, 'test')
                return predictions.values.ravel()
            except Exception as e:
                logger.warning(f"Qlib 原生预测失败: {e}，尝试直接预测")
                # 回退：直接使用内部模型
                if hasattr(self._qlib_model, 'model') and self._qlib_model.model is not None:
                    import xgboost as xgb
                    dmatrix = xgb.DMatrix(X.values)
                    pred = self._qlib_model.model.predict(dmatrix)
                    return pred.ravel()

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测上涨概率（基于收益率预测值）"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        # 回归器预测收益率，转为上涨概率
        pred = self.predict(X)
        # 使用 sigmoid 将收益率映射到概率
        prob = 1 / (1 + np.exp(-10 * pred))
        return np.clip(prob, 0, 1)


class DoubleEnsembleModelWrapper(QlibModelBase):
    """Double Ensemble 模型封装"""

    def __init__(self, config: Optional[QlibModelConfig] = None):
        super().__init__(config)
        self.config.model_type = 'double_ensemble'
        self._use_qlib = False
        self._qlib_model = None
        self._init_model()

    def _init_model(self):
        if QLIB_AVAILABLE:
            try:
                from qlib.contrib.model.double_ensemble import DEnsembleModel
                # 关键：必须设置 decay 参数，否则会报 pow() 错误
                self._qlib_model = DEnsembleModel(
                    n_estimators=self.config.n_estimators,
                    max_depth=self.config.max_depth,
                    learning_rate=self.config.learning_rate,
                    random_state=self.config.random_state,
                    decay=1.0,  # 必须设置，默认 None 会导致 pow 错误
                    epochs=self.config.epochs,
                )
                self._use_qlib = True
                logger.info("使用 Qlib 原生 DEnsembleModel")
                return
            except Exception as e:
                logger.warning(f"Qlib DEnsembleModel 初始化失败: {e}")

        self._init_sklearn_fallback()

    def _init_sklearn_fallback(self):
        try:
            import lightgbm as lgb
            self.model = lgb.LGBMRegressor(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs,
            )
            self._use_qlib = False
            logger.info("使用 lightgbm 库 LGBMRegressor (DoubleEnsemble 后端)")
        except ImportError:
            raise ImportError("请安装 lightgbm: pip install lightgbm")

    def fit(self, X: Union[pd.DataFrame, SimpleDatasetH], y: Optional[pd.Series] = None, **kwargs) -> 'DoubleEnsembleModelWrapper':
        self.feature_names_ = []

        if isinstance(X, pd.DataFrame) and y is not None:
            self.feature_names_ = list(X.columns)
            dataset = create_qlib_compatible_dataset(X, y, use_native=self._use_qlib)
        elif isinstance(X, SimpleDatasetH):
            self.feature_names_ = list(X.features.columns)
            dataset = X
        else:
            raise ValueError("X 必须是 DataFrame 或 SimpleDatasetH")

        if self._use_qlib:
            try:
                # 确保 qlib 已初始化
                _ensure_qlib_initialized()
                self._qlib_model.fit(dataset)
                self.model = self._qlib_model
                logger.info(f"Qlib 原生 DEnsembleModel 训练完成，特征数: {len(self.feature_names_)}")
            except Exception as e:
                logger.warning(f"Qlib 原生模型训练失败: {e}，使用 sklearn 接口")
                self._use_qlib = False
                self._init_sklearn_fallback()
                df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
                X_train = df_train["feature"]
                y_train = df_train["label"].values.ravel()
                self.model.fit(X_train, y_train)
        else:
            df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"])
            X_train = df_train["feature"]
            y_train = df_train["label"].values.ravel()
            self.model.fit(X_train, y_train)
            logger.info(f"DoubleEnsemble 模型训练完成，特征数: {len(self.feature_names_)}")

        self.is_fitted = True
        return self

    def _create_predict_dataset(self, X: pd.DataFrame):
        """为预测创建 DatasetH 格式的数据"""
        from .data_adapter import SimpleDatasetH
        import pandas as pd

        dummy_labels = pd.Series([0] * len(X), index=X.index)
        dataset = SimpleDatasetH(
            features=X,
            labels=dummy_labels,
            segments={'test': (X.index[0].strftime('%Y-%m-%d'), X.index[-1].strftime('%Y-%m-%d'))},
            instrument='stock'
        )
        return dataset

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        if self._use_qlib:
            try:
                pred_dataset = self._create_predict_dataset(X)
                predictions = self._qlib_model.predict(pred_dataset, 'test')
                return predictions.values.ravel()
            except Exception as e:
                logger.warning(f"Qlib 原生预测失败: {e}，使用内部模型预测")
                # 回退：直接使用内部集成模型
                if hasattr(self._qlib_model, 'ensemble') and self._qlib_model.ensemble is not None:
                    pred = np.zeros(len(X))
                    for i_sub, submodel in enumerate(self._qlib_model.ensemble):
                        feat_sub = self._qlib_model.sub_features[i_sub]
                        sub_pred = submodel.predict(X[feat_sub].values)
                        pred += sub_pred * self._qlib_model.sub_weights[i_sub]
                    pred = pred / np.sum(self._qlib_model.sub_weights)
                    return pred.ravel()

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测上涨概率（基于收益率预测值）"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        # 回归器预测收益率，转为上涨概率
        pred = self.predict(X)
        # 使用 sigmoid 将收益率映射到概率
        prob = 1 / (1 + np.exp(-10 * pred))
        return np.clip(prob, 0, 1)


def create_gbdt_model(config: QlibModelConfig) -> QlibModelBase:
    """创建 GBDT 模型"""
    model_type = config.model_type.lower()

    if model_type in ['lgb', 'lightgbm']:
        return LGBModelWrapper(config)
    elif model_type in ['xgboost', 'xgb']:
        return XGBModelWrapper(config)
    elif model_type == 'catboost':
        return CatBoostModelWrapper(config)
    elif model_type == 'double_ensemble':
        return DoubleEnsembleModelWrapper(config)
    else:
        raise ValueError(f"未知的 GBDT 模型类型: {model_type}")