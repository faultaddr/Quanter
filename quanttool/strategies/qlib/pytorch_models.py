"""
PyTorch 序列模型

支持:
- LSTM
- GRU
- ALSTM (Attention LSTM)
- Transformer
- TCN (时间卷积网络)
- Localformer

使用 Qlib 原生的 fit(dataset) 和 predict(dataset) 接口
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


def _get_gpu_id(device: str) -> int:
    """将设备字符串转换为 GPU ID"""
    if device == 'cuda':
        return 0
    elif device == 'mps':
        return 0
    return None


class QlibSequenceModelBase(QlibModelBase):
    """
    Qlib 序列模型基类

    使用 Qlib 原生的 fit(dataset) 和 predict(dataset) 接口
    """

    MODEL_NAME: str = "base"
    QLIB_MODULE: str = ""
    QLIB_CLASS: str = ""
    DEFAULT_PARAMS: dict = {}

    def __init__(self, config: Optional[QlibModelConfig] = None):
        super().__init__(config)
        self.config.model_type = self.MODEL_NAME
        self._use_qlib = False
        self._qlib_model = None
        self._dataset = None
        self.device = config.get_device() if config else 'cpu'
        self._init_model()

    def _get_qlib_model_params(self) -> dict:
        """获取 Qlib 模型参数"""
        gpu = _get_gpu_id(self.config.get_device())
        params = {
            'd_feat': 6,
            'hidden_size': self.config.hidden_size,
            'num_layers': self.config.num_layers,
            'dropout': self.config.dropout,
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
        """初始化回退模型 - 子类实现"""
        raise NotImplementedError("子类需要实现 _init_fallback")

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
                    logger.warning(f"创建 Qlib 数据集失败: {e}")
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
                import qlib
                if not hasattr(qlib, '_initialized') or not qlib._initialized:
                    qlib.init()
                logger.info(f"使用 Qlib 原生 fit(dataset) 方法训练 {self.MODEL_NAME}...")
                self._qlib_model.fit(self._dataset)
                logger.info(f"Qlib 原生 {self.MODEL_NAME} 训练完成")
            except Exception as e:
                logger.warning(f"Qlib fit() 失败: {e}，使用回退模型")
                self._use_qlib = False
                self._init_fallback(input_size=len(self.feature_names_))
                self._train_fallback()
        else:
            self._init_fallback(input_size=len(self.feature_names_))
            self._train_fallback()

        self.is_fitted = True
        return self

    def _train_fallback(self):
        """训练回退模型"""
        import torch
        import torch.nn as nn

        # 优先使用保存的原始数据
        if hasattr(self, '_raw_X') and self._raw_X is not None:
            X_train = self._raw_X
            y_train = self._raw_y
        # 从 SimpleDatasetH 提取数据
        elif hasattr(self._dataset, '_raw_feature_data'):
            X_train = self._dataset._raw_feature_data
            y_train = self._dataset._raw_label_data.ravel()
        elif hasattr(self._dataset, 'features') and hasattr(self._dataset, 'labels'):
            X_train = self._dataset.features.values
            y_train = self._dataset.labels.values.ravel()
        else:
            raise ValueError("无法从数据集提取训练数据")

        # 确保数据有效
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

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """使用 Qlib 原生 predict 方法"""
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        if self._use_qlib and self._qlib_model is not None:
            try:
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
                logger.warning(f"Qlib predict() 失败: {e}")

        # 回退预测
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

class LSTMModelWrapper(QlibSequenceModelBase):
    """LSTM 模型"""

    MODEL_NAME = "lstm"
    QLIB_MODULE = "qlib.contrib.model.pytorch_lstm"
    QLIB_CLASS = "LSTM"
    DEFAULT_PARAMS = {'rnn_type': 'lstm'}

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class CustomLSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                     dropout=dropout if num_layers > 1 else 0, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.model = CustomLSTM(input_size, self.config.hidden_size,
                                 self.config.num_layers, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 LSTM 回退模型, input_size={input_size}")


class GRUModelWrapper(QlibSequenceModelBase):
    """GRU 模型"""

    MODEL_NAME = "gru"
    QLIB_MODULE = "qlib.contrib.model.pytorch_gru"
    QLIB_CLASS = "GRU"
    DEFAULT_PARAMS = {'rnn_type': 'gru'}

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class CustomGRU(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.gru = nn.GRU(input_size, hidden_size, num_layers,
                                   dropout=dropout if num_layers > 1 else 0, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.model = CustomGRU(input_size, self.config.hidden_size,
                                 self.config.num_layers, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 GRU 回退模型, input_size={input_size}")


class ALSTMModelWrapper(QlibSequenceModelBase):
    """Attention LSTM 模型"""

    MODEL_NAME = "alstm"
    QLIB_MODULE = "qlib.contrib.model.pytorch_alstm"
    QLIB_CLASS = "ALSTM"
    DEFAULT_PARAMS = {'rnn_type': 'lstm'}

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class AttentionLayer(nn.Module):
            def __init__(self, hidden_size):
                super().__init__()
                self.attention = nn.Linear(hidden_size, 1)

            def forward(self, x):
                weights = torch.softmax(self.attention(x), dim=1)
                return (x * weights).sum(dim=1)

        class CustomALSTM(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                     dropout=dropout if num_layers > 1 else 0, batch_first=True)
                self.attention = AttentionLayer(hidden_size)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                out = self.attention(out)
                return self.fc(out).squeeze(-1)

        self.model = CustomALSTM(input_size, self.config.hidden_size,
                                   self.config.num_layers, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 ALSTM 回退模型, input_size={input_size}")


class TransformerModelWrapper(QlibSequenceModelBase):
    """Transformer 模型"""

    MODEL_NAME = "transformer"
    QLIB_MODULE = "qlib.contrib.model.pytorch_transformer"
    QLIB_CLASS = "Transformer"
    DEFAULT_PARAMS = {'n_head': 4, 'd_model': 64}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['n_head'] = self.config.n_head
        params['d_model'] = self.config.d_model
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=500):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        class CustomTransformer(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, dropout):
                super().__init__()
                self.input_proj = nn.Linear(input_size, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(d_model, 1)

            def forward(self, x):
                x = self.input_proj(x)
                x = self.pos_encoder(x)
                out = self.transformer(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.model = CustomTransformer(input_size, self.config.d_model,
                                        self.config.n_head, self.config.num_layers, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 Transformer 回退模型, input_size={input_size}")


class TCNModelWrapper(QlibSequenceModelBase):
    """TCN 时间卷积网络模型"""

    MODEL_NAME = "tcn"
    QLIB_MODULE = "qlib.contrib.model.pytorch_tcn"
    QLIB_CLASS = "TCN"
    DEFAULT_PARAMS = {}

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        class TemporalBlock(nn.Module):
            def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout):
                super().__init__()
                self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                        padding=(kernel_size-1)*dilation, dilation=dilation)
                self.relu1 = nn.ReLU()
                self.dropout1 = nn.Dropout(dropout)
                self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                        padding=(kernel_size-1)*dilation*2, dilation=dilation*2)
                self.relu2 = nn.ReLU()
                self.dropout2 = nn.Dropout(dropout)
                self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

            def forward(self, x):
                out = self.dropout1(self.relu1(self.conv1(x)))
                out = self.dropout2(self.relu2(self.conv2(out)))
                res = x if self.downsample is None else self.downsample(x)
                return out + res

        class CustomTCN(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                layers = [TemporalBlock(input_size if i == 0 else hidden_size, hidden_size, 3, 2**i, dropout)
                          for i in range(num_layers)]
                self.tcn = nn.Sequential(*layers)
                self.fc = nn.Linear(hidden_size, 1)

            def forward(self, x):
                x = x.transpose(1, 2)
                out = self.tcn(x)[:, :, -1]
                return self.fc(out).squeeze(-1)

        self.model = CustomTCN(input_size, self.config.hidden_size,
                                 self.config.num_layers, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 TCN 回退模型, input_size={input_size}")


class LocalformerModelWrapper(QlibSequenceModelBase):
    """Localformer 模型"""

    MODEL_NAME = "localformer"
    QLIB_MODULE = "qlib.contrib.model.pytorch_localformer"
    QLIB_CLASS = "LocalformerModel"
    DEFAULT_PARAMS = {'n_head': 4, 'd_model': 64}

    def _get_qlib_model_params(self) -> dict:
        params = super()._get_qlib_model_params()
        params['n_head'] = self.config.n_head
        params['d_model'] = self.config.d_model
        return params

    def _init_fallback(self, input_size: int = None):
        import torch
        import torch.nn as nn

        if input_size is None:
            input_size = len(self.feature_names_) if self.feature_names_ else 6

        # 使用 Transformer 实现作为回退
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=500):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer('pe', pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, :x.size(1)]

        class CustomLocalformer(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers, dropout):
                super().__init__()
                self.input_proj = nn.Linear(input_size, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(d_model, 1)

            def forward(self, x):
                x = self.input_proj(x)
                x = self.pos_encoder(x)
                return self.fc(self.transformer(x)[:, -1, :]).squeeze(-1)

        self.model = CustomLocalformer(input_size, self.config.d_model,
                                        self.config.n_head, self.config.num_layers, self.config.dropout)
        self.device = self.config.get_device()
        self.model.to(self.device)
        logger.info(f"使用 Localformer 回退模型, input_size={input_size}")


def create_pytorch_sequence_model(config: QlibModelConfig) -> QlibSequenceModelBase:
    """创建 PyTorch 序列模型"""
    model_map = {
        'lstm': LSTMModelWrapper,
        'gru': GRUModelWrapper,
        'alstm': ALSTMModelWrapper,
        'transformer': TransformerModelWrapper,
        'tcn': TCNModelWrapper,
        'localformer': LocalformerModelWrapper,
    }
    model_type = config.model_type.lower()
    if model_type in model_map:
        return model_map[model_type](config)
    raise ValueError(f"未知的 PyTorch 序列模型类型: {model_type}")
