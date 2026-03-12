"""
Qlib 模型工厂和基类

统一管理 21 种 Qlib 原生模型:
- GBDT 系列: LightGBM, XGBoost, CatBoost, DoubleEnsemble
- PyTorch 序列: LSTM, GRU, ALSTM, Transformer, TCN, Localformer
- PyTorch 高级: GATs, SFM, TabNet, ADARNN, ADD, HIST, IGMTF, KRNN, TRA, TCTS, Sandwich
"""

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Type
import numpy as np
import pandas as pd

from ...core.logging import get_logger

logger = get_logger(__name__)

# Qlib 可用性检查
try:
    import qlib
    from qlib.data.dataset import DatasetH
    from qlib.contrib.model.gbdt import LGBModel
    QLIB_AVAILABLE = True
except ImportError:
    QLIB_AVAILABLE = False
    logger.warning("pyqlib 未安装，部分功能将不可用")


@dataclass
class QlibModelConfig:
    """Qlib 模型配置"""
    # 模型类型
    model_type: str = "lgb"

    # GBDT 参数
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.01
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8

    # PyTorch 参数
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    batch_size: int = 256
    epochs: int = 100
    early_stopping_rounds: int = 20
    lr_scheduler: str = "cosine"  # cosine, step, exponential

    # Transformer 特定参数
    n_head: int = 4
    d_model: int = 64

    # 设备配置
    device: str = "auto"  # auto, cpu, cuda, mps

    # 其他
    random_state: int = 42
    n_jobs: int = -1
    verbose: int = 0

    def get_device(self) -> str:
        """自动检测并返回设备"""
        if self.device != "auto":
            return self.device

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'model_type': self.model_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'min_child_samples': self.min_child_samples,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_rounds': self.early_stopping_rounds,
            'n_head': self.n_head,
            'd_model': self.d_model,
            'device': self.device,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'verbose': self.verbose,
        }


class QlibModelBase(ABC):
    """
    Qlib 模型基类

    定义统一接口:
    - fit(X, y): 训练模型
    - predict(X): 预测
    - predict_proba(X): 预测概率
    - save(path): 保存模型
    - load(path): 加载模型
    """

    def __init__(self, config: Optional[QlibModelConfig] = None):
        self.config = config or QlibModelConfig()
        self.model = None
        self.is_fitted = False
        self.feature_names_: List[str] = []

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> 'QlibModelBase':
        """训练模型"""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """预测"""
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """预测概率 (分类模型)"""
        # 默认实现：返回预测值作为概率
        return self.predict(X)

    def save(self, filepath: str):
        """保存模型"""
        import joblib
        joblib.dump({
            'model': self.model,
            'config': self.config,
            'feature_names': self.feature_names_,
        }, filepath)
        logger.info(f"模型已保存: {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        import joblib
        data = joblib.load(filepath)
        self.model = data['model']
        self.config = data.get('config', self.config)
        self.feature_names_ = data.get('feature_names', [])
        self.is_fitted = True
        logger.info(f"模型已加载: {filepath}")

    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_names_,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return pd.DataFrame()

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': self.config.model_type,
            'is_fitted': self.is_fitted,
            'n_features': len(self.feature_names_),
            'device': self.config.get_device(),
        }


# 模型注册表
MODEL_REGISTRY: Dict[str, Tuple[str, str]] = {
    # GBDT 系列
    'lgb': ('gbdt', 'LGBModel', 'LightGBM 梯度提升模型'),
    'lightgbm': ('gbdt', 'LGBModel', 'LightGBM 梯度提升模型'),
    'xgboost': ('xgboost', 'XGBModel', 'XGBoost 梯度提升模型'),
    'xgb': ('xgboost', 'XGBModel', 'XGBoost 梯度提升模型'),
    'catboost': ('catboost_model', 'CatBoostModel', 'CatBoost 梯度提升模型'),
    'double_ensemble': ('double_ensemble', 'DEnsembleModel', 'Double Ensemble 模型'),

    # PyTorch 序列模型
    'lstm': ('pytorch_lstm', 'LSTM', 'LSTM 长短期记忆网络'),
    'gru': ('pytorch_gru', 'GRU', 'GRU 门控循环单元'),
    'alstm': ('pytorch_alstm', 'ALSTM', 'Attention LSTM'),
    'transformer': ('pytorch_transformer', 'Transformer', 'Transformer 模型'),
    'tcn': ('pytorch_tcn', 'TCN', '时间卷积网络'),
    'localformer': ('pytorch_localformer', 'LocalformerModel', 'Local Transformer'),

    # PyTorch 高级模型
    'gats': ('pytorch_gats', 'GATs', '图注意力网络'),
    'sfm': ('pytorch_sfm', 'SFM', '状态频域模型'),
    'tabnet': ('pytorch_tabnet', 'TabNet', 'TabNet 表格网络'),
    'adarnn': ('pytorch_adarnn', 'ADARNN', '自适应 RNN'),
    'add': ('pytorch_add', 'ADD', 'ADD 模型'),
    'hist': ('pytorch_hist', 'HIST', 'HIST 历史感知模型'),
    'igmtf': ('pytorch_igmtf', 'IGMTF', 'IGMTF 模型'),
    'krnn': ('pytorch_krnn', 'KRNN', 'KNN-RNN 混合模型'),
    'tra': ('pytorch_tra', 'TRA', 'TRA 模型'),
    'tcts': ('pytorch_tcts', 'TCTS', 'TCTS 模型'),
    'sandwich': ('pytorch_sandwich', 'Sandwich', 'Sandwich 模型'),
}

# 模型类型分类
MODEL_CATEGORIES = {
    'gbdt': ['lgb', 'lightgbm', 'xgboost', 'xgb', 'catboost', 'double_ensemble'],
    'pytorch_sequence': ['lstm', 'gru', 'alstm', 'transformer', 'tcn', 'localformer'],
    'pytorch_advanced': ['gats', 'sfm', 'tabnet', 'adarnn', 'add', 'hist', 'igmtf', 'krnn', 'tra', 'tcts', 'sandwich'],
}


def get_model_category(model_type: str) -> str:
    """获取模型类别"""
    for category, models in MODEL_CATEGORIES.items():
        if model_type.lower() in models:
            return category
    return 'unknown'


def list_available_models() -> pd.DataFrame:
    """列出所有可用模型"""
    models = []
    for model_type, (module, class_name, description) in MODEL_REGISTRY.items():
        models.append({
            'model_type': model_type,
            'category': get_model_category(model_type),
            'class': class_name,
            'description': description,
            'qlib_native': QLIB_AVAILABLE if model_type in ['lgb', 'lightgbm'] else '需要 qlib',
        })
    return pd.DataFrame(models)


def get_model_info(model_type: str) -> Dict[str, Any]:
    """获取模型详细信息"""
    if model_type.lower() not in MODEL_REGISTRY:
        return {'error': f'未知模型类型: {model_type}'}

    module, class_name, description = MODEL_REGISTRY[model_type.lower()]
    return {
        'model_type': model_type,
        'module': module,
        'class': class_name,
        'description': description,
        'category': get_model_category(model_type),
        'qlib_available': QLIB_AVAILABLE,
    }


class QlibModelFactory:
    """
    Qlib 模型工厂

    统一创建和管理 Qlib 原生模型
    """

    @classmethod
    def create(cls, model_type: str, config: Optional[QlibModelConfig] = None, **kwargs) -> QlibModelBase:
        """
        创建模型实例

        Args:
            model_type: 模型类型
            config: 模型配置
            **kwargs: 额外参数

        Returns:
            模型实例
        """
        model_type = model_type.lower()

        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"未知模型类型: {model_type}. 可用模型: {list(MODEL_REGISTRY.keys())}")

        config = config or QlibModelConfig(model_type=model_type)
        config.model_type = model_type

        # 更新配置
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        category = get_model_category(model_type)

        # 创建对应类别的模型
        if category == 'gbdt':
            from .gbdt_models import create_gbdt_model
            return create_gbdt_model(config)
        elif category == 'pytorch_sequence':
            from .pytorch_models import create_pytorch_sequence_model
            return create_pytorch_sequence_model(config)
        elif category == 'pytorch_advanced':
            from .advanced_models import create_advanced_model
            return create_advanced_model(config)
        else:
            raise ValueError(f"未知模型类别: {category}")


def create_model(model_type: str, **kwargs) -> QlibModelBase:
    """
    创建模型的便捷函数

    Args:
        model_type: 模型类型
        **kwargs: 模型参数

    Returns:
        模型实例
    """
    return QlibModelFactory.create(model_type, **kwargs)