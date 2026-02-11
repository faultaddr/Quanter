#!/usr/bin/env python3
"""
Corrected step-by-step execution test for the quant trading system to verify the fix for
'numpy.ndarray' object has no attribute 'shift' error
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.append('/root/Quanter')

# Import modules for testing
from quant_trade_a_share.factors.factor_library_expansion import FactorLibraryExpansion
from quant_trade_a_share.models.model_fusion import ModelFusion
from quant_trade_a_share.risk.portfolio_risk_management import PortfolioRiskManagement
from quant_trade_a_share.optimization.automated_parameter_tuning import AutomatedParameterTuning

def test_step_1_factor_expansion():
    """第一步：因子库扩充测试"""
    print("📊 第一步：因子库扩充")

    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    sample_data = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.02),
        'open': prices + np.random.randn(100) * 0.1,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    try:
        print("🔄 正在生成综合因子...")

        # 创建因子库扩展实例
        factor_expansion = FactorLibraryExpansion()

        # 获取综合因子 - 使用正确的API
        instruments = ['SH000300']  # 示例股票代码
        start_date = '2024-01-01'
        end_date = '2024-04-10'

        # 尝试获取综合因子
        combined_factors = factor_expansion.get_comprehensive_factors(
            data=sample_data,
            instruments=instruments,
            start_date=start_date,
            end_date=end_date
        )

        print(f"📊 综合因子生成完成，共 {len(combined_factors.columns) if not combined_factors.empty else 0} 个因子")
        return combined_factors

    except Exception as e:
        print(f"❌ 因子库扩充步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def test_step_2_model_fusion():
    """第二步：模型融合测试"""
    print("\n🤖 第二步：模型融合")

    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    sample_data = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.02),
        'open': prices + np.random.randn(100) * 0.1,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    try:
        print("🔄 开始运行融合策略...")

        # 创建模型融合实例
        model_fusion = ModelFusion()

        # 计算技术指标信号
        print("📊 计算技术指标信号...")
        tech_signals = model_fusion.calculate_technical_signals(sample_data)
        print(f"✅ 计算完成 {len(tech_signals) if tech_signals else 0} 种技术指标信号")

        # 计算机器学习信号
        print("🤖 计算机器学习信号...")
        ml_signals = model_fusion.calculate_ml_signals(sample_data)
        print(f"✅ 计算机器学习信号，长度: {len(ml_signals)}")

        # 简单融合策略
        print("🔗 计算集成信号...")
        if tech_signals and not ml_signals.empty:
            # 取平均技术信号
            avg_tech_signal = pd.Series(0.0, index=sample_data.index)
            for _, sig in tech_signals.items():
                avg_tech_signal += sig
            avg_tech_signal /= len(tech_signals)

            # 与ML信号融合 (等权重)
            integrated_signals = (avg_tech_signal + ml_signals.reindex(sample_data.index, fill_value=0)) / 2
            print(f"✅ 生成集成信号，包含 {len(integrated_signals)} 个交易信号")
        else:
            integrated_signals = pd.Series(0.0, index=sample_data.index)
            print("⚠️ 使用默认零信号")

        # 评估策略表现
        print("📈 评估策略表现...")
        initial_capital = 100000
        returns = sample_data['close'].pct_change().fillna(0)

        # 计算使用融合信号的收益
        if len(integrated_signals) == len(returns):
            strategy_returns = integrated_signals.shift(1).fillna(0) * returns

            # 计算最终价值
            cumulative_returns = (1 + strategy_returns).cumprod()
            final_value = initial_capital * cumulative_returns.iloc[-1]

            # 计算年化收益
            years = len(returns) / 252  # 假设252个交易日
            annual_return = (final_value / initial_capital) ** (1/years) - 1 if years > 0 else 0

            # 计算夏普比率
            sharpe_ratio = strategy_returns.mean() / (strategy_returns.std() + 1e-10) * np.sqrt(252)

            # 计算最大回撤
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(drawdowns.min())

            print(f"✅ 融合策略运行完成")
            print(f"💰 初始资金: {initial_capital:,}")
            print(f"💰 最终价值: {final_value:,.2f}")
            print(f"📊 年化收益: {annual_return:.4f}")
            print(f"📊 夏普比率: {sharpe_ratio:.4f}")
            print(f"📊 最大回撤: {max_drawdown:.4f}")
            print("✅ 融合策略执行完成")

            return True

        else:
            print("⚠️ 数据长度不匹配，无法评估策略表现")
            return False

    except Exception as e:
        print(f"❌ 模型融合步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_3_risk_management():
    """第三步：风险管理测试"""
    print("\n🛡️  第三步：风险管理")

    try:
        # 创建风险管理实例
        risk_manager = PortfolioRiskManagement()

        # 创建示例资产数据
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        returns_data = pd.DataFrame({
            'asset1': np.random.normal(0.001, 0.02, 100),
            'asset2': np.random.normal(0.0015, 0.015, 100),
            'asset3': np.random.normal(0.0008, 0.018, 100)
        }, index=dates)

        # 执行基本风险评估
        print("📊 计算基本风险指标...")
        risk_metrics = risk_manager.calculate_basic_risk_metrics(returns_data['asset1'])

        print("✅ 风险评估完成")
        print(f"📊 风险指标: {list(risk_metrics.keys()) if risk_metrics else 'None'}")
        return True

    except Exception as e:
        print(f"❌ 风险管理步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_step_4_auto_tuning():
    """第四步：自动调参测试"""
    print("\n⚙️  第四步：自动调参")

    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(50) * 0.5)

    sample_data = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.randn(50)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(50)) * 0.02),
        'open': prices + np.random.randn(50) * 0.1,
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 50)
    }, index=dates)

    try:
        print("🚀 开始综合参数优化...")

        # 创建自动调参实例
        tuner = AutomatedParameterTuning()

        # 定义较小的参数网格用于测试
        param_grid = {
            'ma_short': [5, 10],
            'ma_long': [15, 20],
            'rsi_period': [10, 14]
        }

        print("🔄 执行 grid_search 优化...")
        print("🔍 开始网格搜索参数优化...")
        print(f"📊 将测试 {len(param_grid['ma_short']) * len(param_grid['ma_long']) * len(param_grid['rsi_period'])} 种参数组合")

        # 执行网格搜索 - 这是之前出错的地方，现在应该已修复
        best_params, best_score = tuner.grid_search_optimization(sample_data, param_grid)

        print(f"✅ 网格搜索完成，最佳参数: {best_params}, 得分: {best_score:.4f}")
        print("✅ 自动调参完成")

        return True

    except Exception as e:
        print(f"❌ 自动调参步骤失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数，按步骤执行量化交易系统的测试"""
    print("🔍 A股量化交易系统 - 逐步执行测试")
    print("="*50)

    # Step 1: 因子库扩充
    factors = test_step_1_factor_expansion()

    # Step 2: 模型融合
    model_success = test_step_2_model_fusion()

    # Step 3: 风险管理
    risk_success = test_step_3_risk_management()

    # Step 4: 自动调参
    tuning_success = test_step_4_auto_tuning()

    print("\n" + "="*50)
    print("📋 执行结果总结:")
    print(f"📊 因子库扩充: {'✅ 成功' if not factors.empty else '⚠️ 部分成功' if len(factors.columns) > 0 else '❌ 失败'}")
    print(f"🤖 模型融合: {'✅ 成功' if model_success else '❌ 失败'}")
    print(f"🛡️ 风险管理: {'✅ 成功' if risk_success else '❌ 失败'}")
    print(f"⚙️ 自动调参: {'✅ 成功' if tuning_success else '❌ 失败'}")

    if tuning_success:
        print("\n🎉 自动调参模块已成功修复！错误 'numpy.ndarray' object has no attribute 'shift' 已解决。")
        return True
    else:
        print("\n⚠️ 关键的自动调参模块仍有问题。")
        return False

if __name__ == "__main__":
    main()