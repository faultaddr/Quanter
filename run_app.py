#!/usr/bin/env python3
"""
Entry point for the A-Share Market Analysis Tool
"""
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.main_app import AShareAnalyzer


def print_welcome():
    """
    Print welcome message
    """
    print("="*60)
    print("🎉 欢迎使用A股市场量化分析工具 📊")
    print("="*60)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("功能说明:")
    print("  🔍 股票筛选 - 自动筛选有潜力的股票")
    print("  📊 策略分析 - 多种交易策略回测分析")
    print("  📈 信号通知 - 实时买卖信号推送")
    print("  🌐 可视化界面 - 交互式分析仪表板")
    print()


def main():
    """
    Main entry point
    """
    print_welcome()
    
    # 使用提供的Tushare token
    tushare_token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"
    analyzer = AShareAnalyzer(tushare_token=tushare_token)
    
    # Add a default subscriber for notifications
    analyzer.signal_notifier.add_subscriber(
        email="user@example.com",
        phone="+86-1234567890",
        telegram_id="123456789"
    )
    
    print("选择运行模式:")
    print("  1. 仪表板模式 (推荐)")
    print("  2. 快速筛选模式")
    print("  3. 策略分析模式")
    print("  4. 信号摘要模式")
    print()
    
    choice = input("请输入选择 (1-4, 默认为1): ").strip() or "1"
    
    if choice == "1":
        print("\n🚀 启动Web仪表板...")
        analyzer.run_dashboard()
    elif choice == "2":
        print("\n🔍 执行快速股票筛选...")
        filters = {
            'min_price': 10,
            'max_price': 150,
            'min_volume': 5000000,
            'days_back': 60,
            'min_return': 0.02,
            'max_volatility': 0.04
        }
        analyzer.screen_stocks(filters)
        top_opps = analyzer.get_top_opportunities(5)
    elif choice == "3":
        print("\n📊 执行策略分析...")
        # Example: analyze a specific stock with a strategy
        # Using mock data for demonstration
        symbol = input("请输入股票代码 (例如: SH600519, 默认: SH600519): ").strip() or "SH600519"
        strategy = input("请选择策略 (ma_crossover, rsi, macd, 默认: ma_crossover): ").strip() or "ma_crossover"
        analyzer.analyze_stock(symbol, strategy)
    elif choice == "4":
        print("\n🔔 获取信号摘要...")
        analyzer.screen_stocks({
            'min_price': 10,
            'max_price': 150,
            'min_volume': 5000000,
            'days_back': 60,
            'min_return': 0.02,
            'max_volatility': 0.04
        })
        analyzer.generate_signals_summary()
    else:
        print("❌ 无效选择，启动默认仪表板模式...")
        analyzer.run_dashboard()


if __name__ == "__main__":
    main()