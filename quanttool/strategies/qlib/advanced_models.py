"""
PyTorch 高级模型

支持 11 种 Qlib 原生模型:
- GATs (图注意力网络)
- SFM (状态频域模型)
- TabNet
- ADARNN
- ADD
- HIST
- IGMTF
- KRNN
- TRA
- TCTS
- Sandwich

使用 Qlib 原生的 fit(dataset) 和 predict(dataset) 接口进行训练和推理
"""

import warnings
from typing import Optional, Union
import numpy as np
import pandas as pd

from .models import QlibModelBase, QlibModelConfig, QLIB_AVAILABLE
from .data_adapter import create_ts_compatible_dataset, SimpleDatasetH, create_qlib_dataset_from_dataframe
from ...core.logging import get_logger

logger = get_logger(__name__)

warnings.filterwarnings('ignore')


def _ensure_qlib_initialized():
    """确保 qlib 已初始化"""
    if QLIB_AVAILABLE:
        try:
            import qlib
            if not hasattr(qlib, '_initialized') or not qlib._initialized:
                qlib.init()
        except Exception:
            pass


def _get_gpu_id(device: str) -> int:
    """将设备字符串转换为 GPU ID"""
    if device == 'cuda':
        return 0
    elif device == 'mps':
        return 0
    return None  # CPU


class QlibNativeModelBase(QlibModelBase):
    """
    Qlib 原生模型基类

    使用 Qlib 原生的 fit(dataset) 和 predict(dataset) 接口
    """

    # 子类需要定义的属性
    MODEL_NAME: str = "base"
    QLIB_MODULE: str = ""
    QLIB_CLASS: str = ""
    INTERNAL_ATTR: str = ""
    DEFAULT_PARAMS: dict = {}

    def __init__(self, config: Optional[QlibModelConfig] = None):
        super().__init__(config)
        self.config.model_type = self.MODEL_NAME
        self._use_qlib = False
        self._qlib_model = None
        self._internal_model = None
        self._dataset = None
        self.device = config.get_device() if config else 'cpu'
        self._init_model()

    def _get_qlib_model_params(self) -> dict:
        """获取 Qlib 模型参数"""
        gpu = _get_gpu_id(self.config.get_device())
        params = {
            'n_epochs': self.config.epochs,
            'lr': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'early_stop': self.config.early_stopping_rounds,
            'GPU': gpu,
            'seed': self.config.random_state,
        }
        params.update(self.DEFAULT_PARAMS)
        return params

    def _init_model(self):
        """初始化 Qlib 原生模型"""
        if QLIB_AVAILABLE:
            try:
                import importlib
                module = importlib.import_module(self.QLIB_MODULE)
                model_class = getattr(module, self.QLIB_CLASS)
                params = self._get_qlib_model_params()
                self._qlib_model = model_class(**params)
                self._use_qlib = True
                logger.info(f"使用 Qlib 原生 {self.MODEL_NAME} 模型")
                return
            except Exception as e:
                logger.warning(f"Qlib {self.MODEL_NAME} 初始化失败: {e}")

        self._init_fallback()

    def _init_fallback(self, input_size: int = None):
        """初始化回退模型 - 子类需要实现"""
        raise NotImplementedError("子类需要实现 _init_fallback 方法")

    def fit(self, X: Union[pd.DataFrame, SimpleDatasetH], y: Optional[pd.Series] = None, **kwargs):
        """使用 Qlib 原生 fit(dataset) 方法训练"""
        self.feature_names_ = []

        if isinstance(X, pd.DataFrame) and y is not None:
            self.feature_names_ = list(X.columns)
            # 保存原始数据用于回退模型
            self._raw_X = X.values
            self._raw_y = y.values.ravel()

            if self._use_qlib:
                try:
                    self._dataset = create_qlib_dataset_from_dataframe(X, y)
                except Exception as e:
                    logger.warning(f"创建 Qlib 原生数据集失败: {e}")
                    self._dataset = create_ts_compatible_dataset(X, y, step_len=30, use_native=False)
            else:
                self._dataset = create_ts_compatible_dataset(X, y, step_len=30, use_native=False)
        elif isinstance(X, SimpleDatasetH):
            self.feature_names_ = list(X.features.columns)
            self._raw_X = X.features.values
            self._raw_y = X.labels.values.ravel()
            self._dataset = X
        else:
            raise ValueError("X 必须是 DataFrame 或 SimpleDatasetH")

        if self._use_qlib and self._qlib_model is not None:
            try:
                _ensure_qlib_initialized()
                logger.info(f"使用 Qlib 原生 fit(dataset) 方法训练 {self.MODEL_NAME}...")
                self._qlib_model.fit(self._dataset)
                if self.INTERNAL_ATTR:
                    self._internal_model = getattr(self._qlib_model, self.INTERNAL_ATTR, None)
                logger.info(f"Qlib 原生 {self.MODEL_NAME} 模型训练完成，特征数: {len(self.feature_names_)}")
            except Exception as e:
                logger.warning(f"Qlib 原生 fit() 失败: {e}，使用回退模型")
                self._use_qlib = False
                self._init_fallback(input_size=len(self.feature_names_))
                self._train_fallback_with_dataset()
        else:
            self._init_fallback(input_size=len(self.feature_names_))
            self._train_fallback_with_dataset()

        self.is_fitted = True
        return self

    def _train_fallback_with_dataset(self):
        """使用数据集训练回退模型"""
        import torch
        import torch.nn as nn

        X_train, y_train = self._extract_data_from_dataset()

        if X_train is None or y_train is None or len(X_train) == 0:
            raise ValueError("训练数据为空")

        if len(X_train.shape) == 2:
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.FloatTensor(y_train).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        for epoch in range(self.config.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        logger.info(f"{self.MODEL_NAME} 回退模型训练完成")

    def _extract_data_from_dataset(self) -> tuple:
        """从数据集提取训练数据"""
        # 优先使用保存的原始数据
        if hasattr(self, '_raw_X') and self._raw_X is not None:
            return self._raw_X, self._raw_y
        # 从 SimpleDatasetH 提取
        elif hasattr(self._dataset, '_raw_feature_data'):
            return self._dataset._raw_feature_data, self._dataset._raw_label_data.ravel()
        elif hasattr(self._dataset, 'features') and hasattr(self._dataset, 'labels'):
            return self._dataset.features.values, self._dataset.labels.values.ravel()
        raise ValueError("无法从数据集提取训练数据")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """使用 Qlib 原生 predict 方法或直接推理"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        if self._use_qlib and self._qlib_model is not None:
            try:
                # 使用 Qlib 原生 predict 方法
                pred_dataset = create_qlib_dataset_from_dataframe(
                    X, pd.Series(np.zeros(len(X)), index=X.index)
                )
                predictions = self._qlib_model.predict(pred_dataset)
                if isinstance(predictions, pd.DataFrame):
                    return predictions.values.ravel()
                elif isinstance(predictions, np.ndarray):
                    return predictions.ravel()
                return np.array(predictions).ravel()
            except Exception as e:
                logger.warning(f"Qlib 原生 predict() 失败: {e}，尝试直接推理")
                try:
                    return self._direct_predict(X)
                except Exception as e2:
                    logger.warning(f"直接推理也失败: {e2}")

        return self._fallback_predict(X)

    def _direct_predict(self, X: pd.DataFrame) -> np.ndarray:
        """直接使用内部 PyTorch 模型推理"""
        import torch

        if self._internal_model is None and self.INTERNAL_ATTR:
            self._internal_model = getattr(self._qlib_model, self.INTERNAL_ATTR, None)

        if self._internal_model is None:
            raise RuntimeError("内部模型不可用")

        X_tensor = torch.FloatTensor(X.values).to(self.device)
        if len(X.values.shape) == 2:
            X_tensor = X_tensor.unsqueeze(1)

        self._internal_model.eval()
        with torch.no_grad():
            output = self._internal_model(X_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().numpy().ravel()

    def _fallback_predict(self, X: pd.DataFrame) -> np.ndarray:
        """回退模型预测"""
        import torch
        X_arr = X.values.reshape(X.shape[0], 1, X.shape[1])
        X_tensor = torch.FloatTensor(X_arr).to(self.device)
        self.model.eval()
        with torch.no_grad():
            return self.model(X_tensor).cpu().numpy()

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.predict(X)


# ============================================================================
# 具体模型实现
# ============================================================================

class GATsModelWrapper(QlibNativeModelBase):
    """GATs 图注意力网络模型"""

    MODEL_NAME = "gats"
    QLIB_MODULE = "qlib.contrib.model.pytorch_gats"
    QLIB_CLASS = "GATs"
    INTERNAL_ATTR = "GAT_model"
    DEFAULT_PARAMS = {'d_feat': 6, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['hidden_size'] = self.config.hidden_size
        params['num_layers'] = self.config.num_layers
        params['dropout'] = self.config.dropout
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleGATsFallback(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,
                                   dropout=dropout if num_layers > 1 else 0)
                self.attn = nn.Linear(hidden_size, 1)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                attn_weights = torch.softmax(self.attn(out), dim=1)
                context = torch.sum(attn_weights * out, dim=1)
                return self.fc(context).squeeze(-1)

        self.model = SimpleGATsFallback(input_size, self.config.hidden_size,
                                         self.config.num_layers, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 GATs 回退模型, input_size={input_size}")


class SFMModelWrapper(QlibNativeModelBase):
    """SFM 状态频域模型"""

    MODEL_NAME = "sfm"
    QLIB_MODULE = "qlib.contrib.model.pytorch_sfm"
    QLIB_CLASS = "SFM"
    INTERNAL_ATTR = "sfm_model"
    DEFAULT_PARAMS = {'d_feat': 6, 'hidden_size': 64, 'dropout': 0.1}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['hidden_size'] = self.config.hidden_size
        params['dropout'] = self.config.dropout
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleSFMFallback(nn.Module):
            def __init__(self, input_size, hidden_size, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.dropout(out[:, -1, :])
                return self.fc(out).squeeze(-1)

        self.model = SimpleSFMFallback(input_size, self.config.hidden_size, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 SFM 回退模型, input_size={input_size}")


class TabNetModelWrapper(QlibNativeModelBase):
    """TabNet 表格网络模型"""

    MODEL_NAME = "tabnet"
    QLIB_MODULE = "qlib.contrib.model.pytorch_tabnet"
    QLIB_CLASS = "TabNetModel"
    INTERNAL_ATTR = ""
    DEFAULT_PARAMS = {'n_d': 8, 'n_a': 8, 'n_steps': 3}

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleTabNetFallback(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.bn = nn.BatchNorm1d(hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
                self.fc3 = nn.Linear(hidden_size // 2, 1)
                self.relu = nn.ReLU()

            def forward(self, x):
                if x.dim() == 3:
                    x = x.squeeze(1)
                x = self.relu(self.bn(self.fc1(x)))
                x = self.relu(self.fc2(x))
                return self.fc3(x).squeeze(-1)

        self.model = SimpleTabNetFallback(input_size, self.config.hidden_size)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 TabNet 回退模型, input_size={input_size}")


class ADARNNModelWrapper(QlibNativeModelBase):
    """ADARNN 自适应 RNN 模型"""

    MODEL_NAME = "adarnn"
    QLIB_MODULE = "qlib.contrib.model.pytorch_adarnn"
    QLIB_CLASS = "ADARNN"
    INTERNAL_ATTR = "model"
    DEFAULT_PARAMS = {'d_feat': 6, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['hidden_size'] = self.config.hidden_size
        params['num_layers'] = self.config.num_layers
        params['dropout'] = self.config.dropout
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleADARNNFallback(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                                     dropout=dropout if num_layers > 1 else 0)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.model = SimpleADARNNFallback(input_size, self.config.hidden_size,
                                            self.config.num_layers, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 ADARNN 回退模型, input_size={input_size}")


class ADDModelWrapper(QlibNativeModelBase):
    """ADD 模型"""

    MODEL_NAME = "add"
    QLIB_MODULE = "qlib.contrib.model.pytorch_add"
    QLIB_CLASS = "ADD"
    INTERNAL_ATTR = "ADD_model"
    DEFAULT_PARAMS = {'d_feat': 6, 'hidden_size': 64}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['hidden_size'] = self.config.hidden_size
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleADDFallback(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.model = SimpleADDFallback(input_size, self.config.hidden_size)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 ADD 回退模型, input_size={input_size}")


class HISTModelWrapper(QlibNativeModelBase):
    """HIST 历史感知模型"""

    MODEL_NAME = "hist"
    QLIB_MODULE = "qlib.contrib.model.pytorch_hist"
    QLIB_CLASS = "HIST"
    INTERNAL_ATTR = "HIST_model"
    DEFAULT_PARAMS = {'d_feat': 6, 'hidden_size': 64, 'num_layers': 2}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['hidden_size'] = self.config.hidden_size
        params['num_layers'] = self.config.num_layers
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleHISTFallback(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.attn = nn.Linear(hidden_size, 1)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                attn_w = torch.softmax(self.attn(out), dim=1)
                context = torch.sum(attn_w * out, dim=1)
                return self.fc(context).squeeze(-1)

        self.model = SimpleHISTFallback(input_size, self.config.hidden_size, self.config.num_layers)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 HIST 回退模型, input_size={input_size}")


class IGMTFModelWrapper(QlibNativeModelBase):
    """IGMTF 模型"""

    MODEL_NAME = "igmtf"
    QLIB_MODULE = "qlib.contrib.model.pytorch_igmtf"
    QLIB_CLASS = "IGMTF"
    INTERNAL_ATTR = "igmtf_model"
    DEFAULT_PARAMS = {'d_feat': 6, 'hidden_size': 64}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['hidden_size'] = self.config.hidden_size
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleIGMTFFallback(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.model = SimpleIGMTFFallback(input_size, self.config.hidden_size)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 IGMTF 回退模型, input_size={input_size}")


class KRNNModelWrapper(QlibNativeModelBase):
    """KRNN KNN-RNN 混合模型"""

    MODEL_NAME = "krnn"
    QLIB_MODULE = "qlib.contrib.model.pytorch_krnn"
    QLIB_CLASS = "KRNN"
    INTERNAL_ATTR = "krnn_model"
    DEFAULT_PARAMS = {'d_feat': 6, 'hidden_size': 64}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['hidden_size'] = self.config.hidden_size
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleKRNNFallback(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.rnn(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.model = SimpleKRNNFallback(input_size, self.config.hidden_size)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 KRNN 回退模型, input_size={input_size}")


class TRAModelWrapper(QlibNativeModelBase):
    """TRA 模型"""

    MODEL_NAME = "tra"
    QLIB_MODULE = "qlib.contrib.model.pytorch_tra"
    QLIB_CLASS = "TRA"
    INTERNAL_ATTR = ""
    DEFAULT_PARAMS = {'d_feat': 6, 'hidden_size': 64}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['hidden_size'] = self.config.hidden_size
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleTRAFallback(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.model = SimpleTRAFallback(input_size, self.config.hidden_size)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 TRA 回退模型, input_size={input_size}")


class TCTSModelWrapper(QlibNativeModelBase):
    """TCTS 模型"""

    MODEL_NAME = "tcts"
    QLIB_MODULE = "qlib.contrib.model.pytorch_tcts"
    QLIB_CLASS = "TCTS"
    INTERNAL_ATTR = "fore_model"
    DEFAULT_PARAMS = {'d_feat': 6, 'hidden_size': 64, 'num_layers': 2, 'dropout': 0.1}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['hidden_size'] = self.config.hidden_size
        params['num_layers'] = self.config.num_layers
        params['dropout'] = self.config.dropout
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleTCTSFallback(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True,
                                     dropout=dropout if num_layers > 1 else 0)
                self.fc2 = nn.Linear(hidden_size, 1)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                out, _ = self.lstm(x)
                return self.fc2(out[:, -1, :]).squeeze(-1)

        self.model = SimpleTCTSFallback(input_size, self.config.hidden_size,
                                          self.config.num_layers, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 TCTS 回退模型, input_size={input_size}")


class SandwichModelWrapper(QlibNativeModelBase):
    """Sandwich 模型"""

    MODEL_NAME = "sandwich"
    QLIB_MODULE = "qlib.contrib.model.pytorch_sandwich"
    QLIB_CLASS = "Sandwich"
    INTERNAL_ATTR = "sandwich_model"
    DEFAULT_PARAMS = {'fea_dim': 6, 'cnn_dim_1': 64, 'cnn_dim_2': 32, 'rnn_dim_1': 16, 'rnn_dim_2': 8}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        hidden = self.config.hidden_size
        params['cnn_dim_1'] = hidden
        params['cnn_dim_2'] = hidden // 2
        params['rnn_dim_1'] = hidden // 4
        params['rnn_dim_2'] = hidden // 8
        params['dropout'] = self.config.dropout
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class SimpleSandwichFallback(nn.Module):
            def __init__(self, input_size, hidden_size, dropout):
                super().__init__()
                self.cnn = nn.Sequential(
                    nn.Conv1d(1, hidden_size, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                self.rnn = nn.GRU(hidden_size, hidden_size // 2, batch_first=True)
                self.fc = nn.Linear(hidden_size // 2, 1)

            def forward(self, x):
                if x.dim() == 3 and x.shape[1] == 1:
                    x = x.squeeze(1)
                x = x.unsqueeze(1)
                cnn_out = self.cnn(x)
                cnn_out = cnn_out.permute(0, 2, 1)
                rnn_out, _ = self.rnn(cnn_out)
                return self.fc(rnn_out[:, -1, :]).squeeze(-1)

        self.model = SimpleSandwichFallback(input_size, self.config.hidden_size, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 Sandwich 回退模型, input_size={input_size}")


def create_advanced_model(config: QlibModelConfig) -> QlibModelBase:
    """创建高级模型"""
    model_map = {
        'gats': GATsModelWrapper,
        'sfm': SFMModelWrapper,
        'tabnet': TabNetModelWrapper,
        'adarnn': ADARNNModelWrapper,
        'add': ADDModelWrapper,
        'hist': HISTModelWrapper,
        'igmtf': IGMTFModelWrapper,
        'krnn': KRNNModelWrapper,
        'tra': TRAModelWrapper,
        'tcts': TCTSModelWrapper,
        'sandwich': SandwichModelWrapper,
    }
    model_type = config.model_type.lower()
    if model_type in model_map:
        return model_map[model_type](config)
    raise ValueError(f"未知的高级模型类型: {model_type}")
