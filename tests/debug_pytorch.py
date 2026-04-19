#!/usr/bin/env python
"""PyTorch 诊断脚本"""
import sys
import warnings
warnings.filterwarnings('ignore')

print("Step 1: 检查 PyTorch 环境...")
try:
    import torch
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        print(f"  MPS 可用: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"  ❌ PyTorch 未安装: {e}")
    sys.exit(1)

print("\nStep 2: 测试基本张量操作...")
try:
    x = torch.randn(100, 10)
    y = torch.randn(100)
    print(f"  ✅ 创建张量成功: {x.shape}")
except Exception as e:
    print(f"  ❌ 张量操作失败: {e}")
    sys.exit(1)

print("\nStep 3: 测试 LSTM 模型...")
try:
    import torch.nn as nn

    class SimpleLSTM(nn.Module):
        def __init__(self, input_size, hidden_size):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.sigmoid(self.fc(out[:, -1, :])).squeeze(-1)

    model = SimpleLSTM(10, 64)
    x = torch.randn(32, 1, 10)
    out = model(x)
    print(f"  ✅ LSTM 前向传播成功: {out.shape}")

    # 测试训练
    y = torch.rand(32)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    optimizer.zero_grad()
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    print(f"  ✅ LSTM 训练步骤成功: loss={loss.item():.4f}")

except Exception as e:
    print(f"  ❌ LSTM 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 4: 测试数据加载和特征工程...")
try:
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher
    from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta

    # 获取数据
    end_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    df = AshareFetcher.get_price(code='000876', end_date=end_date, count=200, frequency='1d')
    print(f"  数据条数: {len(df)}")

    # 生成特征
    engineer = QlibFeatureEngineer('Alpha158')
    features = engineer.generate_features(df)
    print(f"  特征数: {len(features.columns)}")
    print(f"  特征 shape: {features.shape}")

    # 检查 NaN/Inf
    nan_count = features.isna().sum().sum()
    inf_count = np.isinf(features.values).sum()
    print(f"  NaN 数量: {nan_count}")
    print(f"  Inf 数量: {inf_count}")

    if nan_count > 0 or inf_count > 0:
        print("  ⚠️ 数据中有 NaN/Inf，正在清理...")
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.ffill().bfill().fillna(0)

    print("  ✅ 特征工程成功")

except Exception as e:
    print(f"  ❌ 数据测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: 测试 PyTorch 模型训练...")
try:
    from quanttool.strategies.qlib.pytorch_models import LSTMModelWrapper, QlibModelConfig

    config = QlibModelConfig(
        model_type='lstm',
        hidden_size=32,
        num_layers=1,
        epochs=2,
        batch_size=32,
        device='cpu'
    )

    model = LSTMModelWrapper(config)

    # 准备训练数据
    close = df['close']
    labels = (close.shift(-10) / close - 1 > 0).astype(int)
    valid_idx = features.dropna().index.intersection(labels.dropna().index)
    X = features.loc[valid_idx].tail(100)
    y = labels.loc[valid_idx].tail(100)

    print(f"  训练数据: X={X.shape}, y={y.shape}")
    print(f"  标签分布: {y.value_counts().to_dict()}")

    # 训练
    model.fit(X, y)
    print("  ✅ 模型训练成功")

    # 预测
    prob = model.predict_proba(X.tail(5))
    print(f"  预测结果: {prob}")

except Exception as e:
    print(f"  ❌ PyTorch 模型训练失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("✅ 所有诊断测试通过！")
print("="*50)