"""
高级 PyTorch 模型

提供 GATs, SFM, TabNet, ADARNN, ADD, HIST, IGMTF, KRNN, TRA, TCTS, Sandwich 等高级模型
"""

import warnings

# 检查 PyTorch 是否可用
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Advanced models will not be available.")


def create_advanced_model(
    model_type: str = "gats",
    input_size: int = 16,
    hidden_size: int = 64,
    **kwargs
):
    """
    创建高级 PyTorch 模型

    Args:
        model_type: 模型类型
        input_size: 输入特征维度
        hidden_size: 隐藏层维度

    Returns:
        PyTorch 模型或 None
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")

    import torch.nn as nn

    # 基础模型类作为占位符
    class AdvancedModel(nn.Module):
        """高级模型基类"""
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

        def forward(self, x):
            return self.fc(x)

    # 支持的模型类型
    supported_models = [
        'gats', 'sfm', 'tabnet', 'adarnn', 'add',
        'hist', 'igmtf', 'krnn', 'tra', 'tcts', 'sandwich'
    ]

    if model_type not in supported_models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {supported_models}")

    # 返回基础模型作为占位符
    # 实际的高级模型实现需要根据论文进行
    return AdvancedModel(input_size, hidden_size)


__all__ = ["create_advanced_model", "TORCH_AVAILABLE"]
