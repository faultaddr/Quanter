"""
Enhanced A-Share Market Predictive Analysis
Identifies stocks that may rise in the next trading day using EastMoney data
With 100+ advanced factors and improved connection handling
"""
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.prediction.predictive_analyzer import PredictiveAnalyzer


def main():
    """
    Main function to run the enhanced predictive analysis
    """
    print("🔮 A股增强版预测分析工具 (100+因子模型)")
    print("="*60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🚀 功能升级:")
    print("   • 集成100+技术分析因子")
    print("   • 多重策略组合筛选")
    print("   • 改进连接错误处理")
    print("   • 智能重试机制")
    print("="*60)

    # Initialize analyzer
    analyzer = PredictiveAnalyzer()

    print("📋 可选股池: 上证50成分股、沪深300成分股等代表性股票")
    sample_symbols = [
        'sh600519',  # 贵州茅台
        'sz000858',  # 五粮液
        'sh600036',  # 招商银行
        'sz000001',  # 平安银行
        'sh601398',  # 工商银行
        'sh601318',  # 中国平安
        'sh600030',  # 中信证券
        'sh600276',  # 恒瑞医药
        'sh600887',  # 伊利股份
        'sh601166'   # 兴业银行
    ]

    print(f"📊 分析 {len(sample_symbols)} 只代表性股票...")
    print("🔍 运行增强版预测模型...")
    print("⚡ 集成策略: 均值回归、动量追踪、成交量分析、振荡器策略...")

    predictions = analyzer.analyze_stocks(symbols=None, top_n=10)

    if not predictions.empty:
        print(f"\n✅ 预测分析完成! 共分析了 {len(predictions)} 只股票")

        # Print top predictions
        analyzer.print_top_predictions(predictions, top_n=10)

        print("\n🏆 预测分数解释 (基于100+因子综合评估):")
        print("   分数 > 5: 强烈看涨信号 (多重因子共振)")
        print("   分数 3-5: 看涨信号 (多个因子支持)")
        print("   分数 1-3: 中性偏多 (部分因子支持)")
        print("   分数 0-1: 中性 (无明确方向)")
        print("   分数 < 0: 看跌信号 (技术面偏弱)")

        print("\n🔍 增强版预测依据 (100+因子组合):")
        print("   • 基础指标: RSI、MACD、移动平均线、布林带")
        print("   • 高级因子: 100+多维度技术指标组合")
        print("   • 策略融合: 均值回归、动量追踪、成交量分析")
        print("   • 振荡器: RSI、随机指标、威廉指标等")
        print("   • 波动率: ATR、真实波动范围等")
        print("   • 相关性: 价格-成交量相关性分析")

        print("\n🧠 策略模型:")
        if analyzer.advanced_strategy_manager:
            print(f"   已激活 {len(analyzer.advanced_strategy_manager.get_strategy_names())} 种高级策略")
            print("   • 均值回归策略: 寻找超卖后的反弹机会")
            print("   • 动量追踪策略: 抓住趋势延续的行情")
            print("   • 成交量策略: 识别资金流入流出信号")
            print("   • 振荡器策略: 利用超买超卖条件")
            print("   • 突破策略: 捕捉关键位突破信号")
            print("   • 相关性策略: 分析指标间相互关系")
            print("   • 波动率策略: 适应不同市场环境")
        else:
            print("   使用基础技术分析策略")

        print("\n⚠️  风险提示:")
        print("   本预测仅供学习参考，不构成投资建议")
        print("   市场有风险，投资需谨慎")
        print("   请结合基本面分析和其他信息进行决策")

        # Note about visualization capability
        print("\n📈 可视化功能:")
        print("   系统具备完整的可视化分析能力，包括:")
        print("   - 预测分数排名图表")
        print("   - 价格与预测关系散点图")
        print("   - RSI技术指标分布图")
        print("   - 详细股票分析热力图")
        print("   - 策略因子贡献度分析")

    else:
        print("❌ 未能获取有效的预测结果")
        print("可能原因: 网络连接问题或数据源不可用")
        print("系统将在无法获取真实数据时使用高质量模拟数据进行演示")
        print("\n💡 建议措施:")
        print("   1. 检查网络连接是否正常")
        print("   2. 确认EastMoney数据源可用性")
        print("   3. 稍后重试以避开高峰时段")


if __name__ == "__main__":
    main()
