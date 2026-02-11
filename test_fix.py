#!/usr/bin/env python
"""
Test script to verify the fix for the 'numpy.ndarray' object has no attribute 'shift' error
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 导入我们的模块
from quant_trade_a_share.optimization.automated_parameter_tuning import AutomatedParameterTuning

def test_technical_strategy_evaluation():
    """测试技术策略评估功能"""
    print("🧪 测试技术策略评估修复...")

    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)

    sample_data = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.randn(200)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(200)) * 0.02),
        'open': prices + np.random.randn(200) * 0.1,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)

    # 创建调参器实例
    tuner = AutomatedParameterTuning()

    # 测试参数
    params = {
        'ma_short': 5,
        'ma_long': 20,
        'rsi_period': 14,
        'rsi_buy_threshold': 30,
        'rsi_sell_threshold': 70
    }

    try:
        # 尝试评估技术策略 - 这是之前出错的地方
        score = tuner._evaluate_technical_strategy(sample_data, params)
        print(f"✅ 技术策略评估成功! 得分: {score}")
        return True
    except AttributeError as e:
        if "'numpy.ndarray' object has no attribute 'shift'" in str(e):
            print(f"❌ 错误仍然存在: {e}")
            return False
        else:
            print(f"❌ 其他AttributeError: {e}")
            return False
    except Exception as e:
        print(f"❌ 发生其他错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_grid_search():
    """测试网格搜索功能"""
    print("\n🧪 测试网格搜索功能...")

    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')  # 较小的数据集以加快测试
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)

    sample_data = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.randn(50)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(50)) * 0.02),
        'open': prices + np.random.randn(50) * 0.1,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 50)
    }, index=dates)

    # 创建调参器实例
    tuner = AutomatedParameterTuning()

    # 定义一个小的参数网格进行测试
    param_grid = {
        'ma_short': [5, 10],
        'ma_long': [15, 20],
        'rsi_period': [10, 14]
    }

    try:
        best_params, best_score = tuner.grid_search_optimization(sample_data, param_grid)
        print(f"✅ 网格搜索成功! 最佳参数: {best_params}, 得分: {best_score}")
        return True
    except AttributeError as e:
        if "'numpy.ndarray' object has no attribute 'shift'" in str(e):
            print(f"❌ 网格搜索中仍存在错误: {e}")
            return False
        else:
            print(f"❌ 网格搜索中发生其他AttributeError: {e}")
            return False
    except Exception as e:
        print(f"❌ 网格搜索发生其他错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 测试修复后的自动化参数调优模块")
    print("="*50)

    success1 = test_technical_strategy_evaluation()
    success2 = test_grid_search()

    print("\n" + "="*50)
    if success1 and success2:
        print("🎉 所有测试通过！错误已修复。")
    else:
        print("❌ 测试失败，请检查代码。")