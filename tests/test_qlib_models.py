"""
Qlib 模型工厂测试
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_model_registry():
    """测试模型注册表"""
    from quanttool.strategies.qlib.models import MODEL_REGISTRY, list_available_models

    # 检查所有模型都在注册表中
    assert 'lgb' in MODEL_REGISTRY
    assert 'xgboost' in MODEL_REGISTRY
    assert 'lstm' in MODEL_REGISTRY
    assert 'transformer' in MODEL_REGISTRY
    assert 'gats' in MODEL_REGISTRY

    # 列出可用模型
    models_df = list_available_models()
    assert len(models_df) >= 20  # 至少 20 种模型
    print(f"\n可用模型数量: {len(models_df)}")
    print(models_df[['model_type', 'category', 'description']])


def test_model_config():
    """测试模型配置"""
    from quanttool.strategies.qlib.models import QlibModelConfig

    config = QlibModelConfig(
        model_type='transformer',
        hidden_size=128,
        num_layers=4,
        n_head=8,
    )

    assert config.model_type == 'transformer'
    assert config.hidden_size == 128
    assert config.num_layers == 4
    assert config.n_head == 8

    # 测试设备检测
    device = config.get_device()
    assert device in ['cpu', 'cuda', 'mps']
    print(f"\n检测到设备: {device}")


def test_create_lgb_model():
    """测试创建 LightGBM 模型"""
    from quanttool.strategies.qlib.models import create_model, QlibModelConfig

    config = QlibModelConfig(model_type='lgb', n_estimators=100)
    model = create_model('lgb', config=config)

    assert model is not None
    assert model.config.model_type == 'lgb'
    print(f"\nLightGBM 模型创建成功: {model.get_model_info()}")


def test_create_pytorch_model():
    """测试创建 PyTorch 模型"""
    pytest.importorskip("torch")

    from quanttool.strategies.qlib.models import create_model, QlibModelConfig

    # 测试 LSTM
    config = QlibModelConfig(
        model_type='lstm',
        hidden_size=32,
        num_layers=2,
        epochs=5,
    )
    model = create_model('lstm', config=config)

    assert model is not None
    print(f"\nLSTM 模型创建成功: {model.get_model_info()}")

    # 测试 Transformer
    model = create_model('transformer', hidden_size=32, num_layers=2, epochs=5)
    assert model is not None
    print(f"\nTransformer 模型创建成功: {model.get_model_info()}")


def test_model_fit_predict():
    """测试模型训练和预测"""
    pytest.importorskip("lightgbm")

    from quanttool.strategies.qlib.models import create_model

    # 生成测试数据
    np.random.seed(42)
    n_samples = 200
    n_features = 50

    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 2, n_samples), name='label')

    # 训练 LightGBM
    model = create_model('lgb', n_estimators=50)
    model.fit(X, y)

    # 预测
    predictions = model.predict(X.iloc[:10])
    assert len(predictions) == 10

    proba = model.predict_proba(X.iloc[:10])
    assert len(proba) == 10
    assert all(0 <= p <= 1 for p in proba)

    print(f"\nLightGBM 训练预测成功")
    print(f"预测概率: {proba[:5]}")


def test_strategy_model_types():
    """测试策略支持的模型类型"""
    from quanttool.strategies.qlib_strategy import QlibStrategy

    # 获取支持的模型
    supported = QlibStrategy.list_supported_models()
    print(f"\n支持的模型类型:")
    for category, models in supported.items():
        print(f"  {category}: {models}")

    assert len(supported['GBDT系列']) >= 4
    assert len(supported['PyTorch序列']) >= 6
    assert len(supported['PyTorch高级']) >= 11


def test_strategy_with_different_models():
    """测试策略使用不同模型"""
    from quanttool.strategies.qlib_strategy import QlibStrategy

    # 测试不同模型类型
    model_types = ['lgb', 'gru', 'transformer']

    for model_type in model_types:
        try:
            strategy = QlibStrategy(
                model_type=model_type,
                hidden_size=32,
                num_layers=2,
                epochs=5,
            )
            print(f"\n策略创建成功: {strategy.get_description()}")
        except ImportError as e:
            print(f"\n跳过 {model_type}: {e}")


if __name__ == '__main__':
    test_model_registry()
    test_model_config()
    test_create_lgb_model()
    test_create_pytorch_model()
    test_model_fit_predict()
    test_strategy_model_types()
    test_strategy_with_different_models()
    print("\n✅ 所有测试通过!")