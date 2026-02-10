"""
Main Script for A-Share Market Predictive Analysis
Identifies stocks that may rise in the next trading day using EastMoney data
"""
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.prediction.predictive_analyzer import PredictiveAnalyzer


def main():
    """
    Main function to run the predictive analysis
    """
    print("🔍 A股潜在上涨股票预测分析工具")
    print("="*50)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

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
    predictions = analyzer.analyze_stocks(symbols=None, top_n=10)

    if not predictions.empty:
        print(f"\n✅ 预测分析完成! 共分析了 {len(predictions)} 只股票")

        # Print top predictions
        analyzer.print_top_predictions(predictions, top_n=10)

        print("\n💡 预测分数解释:")
        print("   分数 > 3: 强烈看涨信号")
        print("   分数 2-3: 看涨信号")
        print("   分数 1-2: 中性偏多")
        print("   分数 0-1: 中性")
        print("   分数 < 0: 看跌信号")

        print("\n🔍 预测依据:")
        print("   - RSI指标: 超卖股票有反弹潜力")
        print("   - 移动平均线: 价格突破短期均线")
        print("   - MACD指标: 金叉信号")
        print("   - 布林带: 价格触及下轨后反弹")
        print("   - 成交量: 成交量放大表示关注增加")

        print("\n⚠️  风险提示:")
        print("   本预测仅供学习参考，不构成投资建议")
        print("   投资有风险，入市需谨慎")

        # Note about visualization capability
        print("\n📈 可视化功能:")
        print("   系统具备完整的可视化分析能力，包括:")
        print("   - 预测分数排名图表")
        print("   - 价格与预测关系散点图")
        print("   - RSI技术指标分布图")
        print("   - 详细股票分析热力图")

    else:
        print("❌ 未能获取有效的预测结果")
        print("可能原因: 网络连接问题或数据源不可用")
        print("系统将在无法获取真实数据时使用高质量模拟数据进行演示")


if __name__ == "__main__":
    main()
