#!/usr/bin/env python3
"""
测试原始问题是否解决的简单脚本
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_original_problem_fixed():
    """测试原始错误是否已修复"""

    print("🧪 测试原始问题是否已修复...")

    # 模拟原始问题 - 包含重复索引的数据
    dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
    prices = 100 + np.cumsum(np.random.randn(20) * 0.5)

    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(20) * 0.01),
        'high': prices * (1 + abs(np.random.randn(20)) * 0.02),
        'low': prices * (1 - abs(np.random.randn(20)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 20)
    }, index=dates)

    # 创建重复索引（模拟数据处理中可能遇到的问题）
    duplicate_row = data.iloc[5:6]  # 选取一行
    duplicate_row.index = [data.index[5]]  # 设置相同的索引
    problematic_data = pd.concat([data, duplicate_row])  # 这样会产生重复索引

    print(f"数据形状: {problematic_data.shape}")
    print(f"重复索引数量: {problematic_data.index.duplicated().sum()}")

    try:
        # 尝试使用原来的处理方式，看是否还会报错
        from quant_trade_a_share.models.model_fusion import ModelFusion

        fusion = ModelFusion()

        print("🔄 计算技术指标信号...")
        tech_signals = fusion.calculate_technical_signals(problematic_data)

        print("🤖 计算机器学习信号...")
        ml_signals = fusion.calculate_ml_signals(problematic_data)

        print("🔗 计算集成信号...")
        ensemble_signal = fusion.calculate_ensemble_signal(tech_signals, ml_signals)

        print(f"✅ 成功生成 {len(ensemble_signal[ensemble_signal != 0])} 个交易信号")
        print("🎉 原始问题已解决！")
        return True

    except Exception as e:
        print(f"❌ 问题仍然存在: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 验证原始问题是否解决")
    print("="*40)
    success = test_original_problem_fixed()
    print("="*40)
    if success:
        print("✅ 原始 'cannot reindex on an axis with duplicate labels' 错误已修复！")
    else:
        print("❌ 问题仍然存在")