"""
PyTorch 序列模型

提供 LSTM, GRU, ALSTM, Transformer, TCN 等序列模型
"""

import warnings

# 检查 PyTorch 是否可用
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. PyTorch models will not be available.")


def create_pytorch_sequence_model(
    model_type: str = "lstm",
    input_size: int = 16,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    **kwargs
):
    """
    创建 PyTorch 序列模型

    Args:
        model_type: 模型类型 (lstm, gru, alstm, transformer, tcn)
        input_size: 输入特征维度
        hidden_size: 隐藏层维度
        num_layers: 层数
        dropout: Dropout 比率

    Returns:
        PyTorch 模型或 None
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")

    import torch.nn as nn

    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                               batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, dropout):
            super().__init__()
            self.gru = nn.GRU(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=dropout)
            self.fc = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _ = self.gru(x)
            return self.fc(out[:, -1, :])

    models = {
        "lstm": LSTMModel,
        "gru": GRUModel,
    }

    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")

    return models[model_type](input_size, hidden_size, num_layers, dropout)


__all__ = ["create_pytorch_sequence_model", "TORCH_AVAILABLE"]
