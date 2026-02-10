#!/usr/bin/env python3
"""
Enhanced A-Share Stock Prediction System
With 100+ advanced factors and improved error handling
"""

import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.prediction.predictive_analyzer import PredictiveAnalyzer

def main():
    """
    Main function to run the enhanced predictive analysis
    """
    print("🔮 A股增强版预测分析系统")
    print("="*60)
    print("📈 集成100+高级因子和多重策略")
    print("🔧 改进连接错误处理和重试机制")
    print("="*60)

    # Initialize analyzer
    analyzer = PredictiveAnalyzer()

    try:
        # Analyze stocks (using sample symbols for demonstration)
        sample_symbols = ['sh600519', 'sz000858', 'sh600036', 'sz000001', 'sh601398',
                         'sh601318', 'sz000002', 'sh600030', 'sh600276', 'sh000001']

        print("🔍 开始分析股票...")
        print("📊 使用增强版预测模型...")

        predictions = analyzer.analyze_stocks(symbols=None, top_n=10)

        if not predictions.empty:
            # Print predictions
            print("\n" + "="*100)
            print("🏆 增强版预测结果 (基于100+因子组合筛选)")
            print("="*100)

            # Enhanced output with factor information if available
            analyzer.print_top_predictions(predictions)

            print("\n💡 预测说明:")
            print("   • 预测分数基于100+技术因子综合计算")
            print("   • 结合了趋势、动量、均值回归、波动率等多种策略")
            print("   • 考虑了市场情绪、资金流向等高级指标")
            print("   • 推荐关注预测分数较高且基本面良好的股票")

            # Check if we can access advanced strategy info
            if analyzer.advanced_strategy_manager:
                print(f"\n⚙️  已激活 {len(analyzer.advanced_strategy_manager.get_strategy_names())} 种高级策略")
                print("📊 策略包括: 均值回归、动量追踪、成交量分析、振荡器策略等")

                # Show factor exposure for the first analyzed stock if possible
                if not predictions.empty:
                    first_symbol = predictions.iloc[0]['symbol']
                    print(f"\n🔍 {first_symbol} 因子暴露度示例:")
                    try:
                        sample_data = analyzer.fetch_stock_data(first_symbol, days=60)
                        if sample_data is not None:
                            factors = analyzer.advanced_strategy_manager.get_factor_exposure(sample_data)
                            print(f"   总计 {factors.shape[1]} 个因子已计算")
                            print(f"   样例因子: {list(factors.columns[:5])}...")
                    except:
                        print("   (因子数据获取中)")

        else:
            print("❌ 未能获取有效的预测结果")
            print("可能原因: 网络连接问题或数据源不可用")
            print("建议检查网络连接或稍后重试")

    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("✅ 分析完成!")
    print("⚠️  注意: 预测结果仅供参考，投资需谨慎")
    print("💡 本系统集成了多种技术分析方法和机器学习策略")


if __name__ == "__main__":
    main()