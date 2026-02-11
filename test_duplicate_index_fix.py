#!/usr/bin/env python3
"""
测试重复索引修复的脚本
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_duplicate_index_fix():
    """测试修复后的代码是否能处理重复索引"""

    print("🧪 开始测试重复索引修复...")

    # 创建包含重复索引的数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    # 创建有重复索引的数据
    sample_data = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.02),
        'open': prices + np.random.randn(100) * 0.1,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    # 添加一些重复的索引（模拟真实世界中可能出现的情况）
    # 重复索引可能是由于数据加载错误、合并数据时产生的
    duplicate_rows = sample_data.iloc[[10, 20, 30]]
    duplicate_rows.index = [sample_data.index[10]] * 3  # 人为制造重复索引

    sample_data_with_duplicates = pd.concat([sample_data, duplicate_rows])

    print(f"原始数据形状: {sample_data.shape}")
    print(f"带重复索引的数据形状: {sample_data_with_duplicates.shape}")
    print(f"重复索引的数量: {sample_data_with_duplicates.index.duplicated().sum()}")

    try:
        # 测试 ModelFusion 类
        from quant_trade_a_share.models.model_fusion import ModelFusion

        print("\n✅ 创建 ModelFusion 实例...")
        fusion = ModelFusion()

        print("\n🔄 测试计算技术指标信号...")
        technical_signals = fusion.calculate_technical_signals(sample_data_with_duplicates)
        print(f"📊 生成的技术指标信号数量: {len(technical_signals)}")

        print("\n🤖 测试计算机器学习信号...")
        ml_signals = fusion.calculate_ml_signals(sample_data_with_duplicates)
        print(f"🤖 生成的 ML 信号数量: {len(ml_signals[ml_signals != 0])}")

        print("\n🔗 测试计算集成信号...")
        ensemble_signal = fusion.calculate_ensemble_signal(technical_signals, ml_signals)
        print(f"🔗 生成的集成信号数量: {len(ensemble_signal[ensemble_signal != 0])}")

        print("\n✅ 所有测试通过！重复索引问题已修复")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_deep_qlib_integration():
    """测试深度 Qlib 集成模块"""

    print("\n🧪 开始测试 DeepQlibIntegration 修复...")

    # 创建测试数据
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    prices = 100 + np.cumsum(np.random.randn(50) * 0.3)

    # 创建有重复索引的数据
    test_data = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.randn(50)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(50)) * 0.01),
        'open': prices + np.random.randn(50) * 0.05,
        'close': prices,
        'volume': np.random.randint(100000, 500000, 50)
    }, index=dates)

    # 添加重复索引
    duplicate_rows = test_data.iloc[[5, 15]]
    duplicate_rows.index = [test_data.index[5]] * 2
    test_data_with_duplicates = pd.concat([test_data, duplicate_rows])

    try:
        from quant_trade_a_share.integration.deep_qlib_integration import DeepQlibIntegration

        print("✅ 创建 DeepQlibIntegration 实例...")
        integration = DeepQlibIntegration()

        print("\n🔍 测试特征准备...")
        features = integration.prepare_ml_features(test_data_with_duplicates)
        print(f"🔍 生成的特征数量: {features.shape[1] if not features.empty else 0}")

        print("\n🤖 测试 ML 信号生成...")
        if not features.empty:
            ml_signals = integration.ml_signal_generation(features)
            print(f"🤖 生成的 ML 信号数量: {len(ml_signals[ml_signals != 0]) if not ml_signals.empty else 0}")

        print("\n📊 测试传统信号生成...")
        trad_signals = integration.get_traditional_signals(test_data_with_duplicates)
        print(f"📊 生成的传统信号数量: {len(trad_signals[trad_signals != 0]) if not trad_signals.empty else 0}")

        print("\n✅ DeepQlibIntegration 测试通过！")
        return True

    except Exception as e:
        print(f"❌ DeepQlibIntegration 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 修复重复索引错误 - 综合测试")
    print("="*50)

    success1 = test_duplicate_index_fix()
    success2 = test_deep_qlib_integration()

    print("\n" + "="*50)
    if success1 and success2:
        print("🎉 所有测试通过！重复索引问题已完全修复。")
    else:
        print("❌ 部分测试失败，请检查错误信息。")