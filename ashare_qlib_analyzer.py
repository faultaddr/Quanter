#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ashare-Qlib 综合分析主入口
直接使用 Ashare 数据进行 Qlib 增强分析，以当日之前 180 天数据作为输入
"""

import argparse
import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_trade_a_share.integration.ashare_qlib_integration import AshareQlibIntegration


def main():
    parser = argparse.ArgumentParser(description='Ashare-Qlib 综合分析工具')
    parser.add_argument('--symbol', type=str, required=True, help='股票代码 (例如: sh600023)')
    parser.add_argument('--days', type=int, default=180, help='回溯天数 (默认: 180)')
    parser.add_argument('--analysis-type', type=str, choices=[
        'comprehensive', 'factor', 'signal', 'portfolio', 'all'
    ], default='all', help='分析类型 (默认: all)')

    args = parser.parse_args()

    print("🚀 启动 Ashare-Qlib 综合分析...")
    print(f"🎯 股票代码: {args.symbol}")
    print(f"📊 回溯天数: {args.days}")
    print(f"🔍 分析类型: {args.analysis_type}")
    print("-" * 50)

    # 创建集成实例
    integration = AshareQlibIntegration()

    # 根据分析类型执行相应功能
    if args.analysis_type == 'comprehensive':
        print("\n📊 执行综合性分析...")
        results = integration.run_comprehensive_qlib_analysis_with_ashare(
            symbol=args.symbol,
            days=args.days
        )

    elif args.analysis_type == 'factor':
        print("\n📈 执行高级因子分析...")
        results = integration.run_advanced_factor_analysis_with_ashare(
            symbol=args.symbol,
            days=args.days
        )

    elif args.analysis_type == 'signal':
        print("\n🎯 执行自适应信号生成...")
        results = integration.run_adaptive_signal_generation_with_ashare(
            symbol=args.symbol,
            days=args.days
        )

    elif args.analysis_type == 'portfolio':
        print("\n⚖️  执行智能投资组合优化...")
        results = integration.run_smart_portfolio_optimization_with_ashare(
            symbols=[args.symbol],
            days=args.days
        )

    elif args.analysis_type == 'all':
        print("\n🌟 执行完整分析...")
        results = integration.run_all_analysis_with_ashare(
            symbol=args.symbol,
            days=args.days
        )

    print("\n✅ 分析完成!")


if __name__ == "__main__":
    main()