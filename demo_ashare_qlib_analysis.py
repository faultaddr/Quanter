#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ashare-Qlib 综合分析示例
展示如何使用当日之前 180 天的 Ashare 数据进行 Qlib 增强分析
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_trade_a_share.integration.ashare_qlib_integration import AshareQlibIntegration


def demo_ashare_qlib_analysis():
    """演示 Ashare-Qlib 综合分析功能"""

    print("🌟 欢迎使用 Ashare-Qlib 综合分析系统")
    print("="*60)
    print("💡 本系统将使用 Ashare 数据源提供的当日之前 180 天数据")
    print("💡 进行 Qlib 增强分析")
    print("="*60)

    # 创建集成实例
    integration = AshareQlibIntegration()

    # 使用示例股票代码
    sample_symbols = ["sh600023", "sz000001", "sh600519"]

    print(f"\n📊 可用示例股票: {sample_symbols}")

    # 测试第一只股票 - 华能国际
    symbol = sample_symbols[0]
    print(f"\n🎯 开始分析股票: {symbol}")

    # 运行完整分析
    results = integration.run_all_analysis_with_ashare(symbol, days=180)

    print("\n" + "="*60)
    print("📋 分析概要:")

    # 显示关键结果
    if 'comprehensive' in results:
        comp_results = results['comprehensive']
        if 'factors' in comp_results:
            print(f"  • 生成因子数量: {len(comp_results['factors'].columns) if not comp_results['factors'].empty else 0}")

        if 'fusion' in comp_results:
            fusion_perf = comp_results['fusion'].get('performance', {})
            if fusion_perf:
                print(f"  • 策略年化收益率: {fusion_perf.get('annual_return', 0):.2%}")
                print(f"  • 夏普比率: {fusion_perf.get('sharpe_ratio', 0):.3f}")

    if 'factor_analysis' in results:
        factor_results = results['factor_analysis']
        if 'factors' in factor_results:
            print(f"  • 高级因子分析因子数: {len(factor_results['factors'].columns) if not factor_results['factors'].empty else 0}")

    if 'signal_generation' in results:
        signal_results = results['signal_generation']
        if 'adaptive_signal' in signal_results:
            adaptive_signal = signal_results['adaptive_signal']
            if hasattr(adaptive_signal, 'shape'):
                active_signals = len(adaptive_signal[adaptive_signal != 0])
                print(f"  • 生成自适应信号数: {active_signals}")

    print("="*60)
    print("✅ 分析完成!")

    # 展示特定分析功能
    print(f"\n🔍 现在运行单一功能分析...")

    # 高级因子分析
    print(f"\n📈 对 {sample_symbols[2]} (贵州茅台) 进行高级因子分析...")
    factor_results = integration.run_advanced_factor_analysis_with_ashare(sample_symbols[2], days=180)

    if 'factors' in factor_results:
        factors_df = factor_results['factors']
        print(f"   生成 {len(factors_df.columns)} 个因子，{len(factors_df)} 条数据")


def explain_implementation():
    """解释实现的技术细节"""

    print("\n" + "="*60)
    print("🔧 技术实现说明:")
    print("="*60)
    print("""
    1. 数据获取:
       • 使用 AshareDataFetcher 从腾讯/新浪获取实时 A 股数据
       • 自动适配多种数据源和错误恢复机制

    2. 数据处理:
       • 自动获取当日之前 180 天的历史数据
       • 标准化数据格式，适配 Qlib 分析要求
       • 数据清洗和异常值处理

    3. Qlib 集成:
       • 因子库扩充: 结合 Qlib Alpha 因子和 MyTT 技术指标
       • 模型融合: 传统技术指标 + 机器学习模型
       • 风险管理: Qlib 风险模型 + 投资组合优化
       • 自动调参: 网格搜索 + 贝叶斯 + 遗传算法

    4. 分析功能:
       • 综合性分析: 四大功能一体化分析
       • 高级因子分析: 因子统计、相关性、IC 分析
       • 智能投资组合优化: 风险平价等策略
       • 自适应信号生成: 融合多模型输出
    """)


if __name__ == "__main__":
    demo_ashare_qlib_analysis()
    explain_implementation()
    print("\n🎉 Ashare-Qlib 综合分析演示完成！")