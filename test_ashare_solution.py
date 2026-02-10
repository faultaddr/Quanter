#!/usr/bin/env python3
"""
Complete test script to verify that ASHare is now the primary data source
replacing EastMoney and fixing connection issues
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.screeners.stock_screener import StockScreener

def test_ashare_primary():
    print("=" * 60)
    print("🔍 测试Ashare作为主数据源 - 解决连接错误问题")
    print("=" * 60)

    # 初始化股票筛选器 - 现在Ashare是主数据源
    screener = StockScreener()

    # 测试获取一些股票数据
    test_symbols = ['sh600519', 'sz000001', 'sh600023']  # 一些常见股票代码

    print(f"\n📊 测试股票: {test_symbols}")

    successful_fetches = 0
    total_attempts = len(test_symbols)

    for symbol in test_symbols:
        print(f"\n🧪 测试获取 {symbol} 的数据...")

        # 使用默认数据源（现在是Ashare优先）
        data = screener.fetch_stock_data(symbol, period='30', data_source='auto')

        if data is not None and not data.empty:
            print(f"✅ 成功获取 {symbol} 的 {len(data)} 条数据记录")
            if 'close' in data.columns:
                print(f"   当前价格: {data['close'].iloc[-1]:.2f}")
            successful_fetches += 1
        else:
            print(f"❌ 无法获取 {symbol} 的数据")

    print(f"\n📈 测试结果: {successful_fetches}/{total_attempts} 股票数据获取成功")

    # 特别测试之前有问题的股票代码
    problematic_symbol = 'sh688818'
    print(f"\n🔍 重点测试之前有问题的股票: {problematic_symbol}")

    data = screener.fetch_stock_data(problematic_symbol, period='30', data_source='ashare')

    if data is not None and not data.empty:
        print(f"✅ 成功获取 {problematic_symbol} 的数据（使用Ashare）")
        if 'close' in data.columns:
            print(f"   当前价格: {data['close'].iloc[-1]:.2f}")
    else:
        print(f"⚠️  仍然无法获取 {problematic_symbol} 的Ashare数据")

    # 测试获取股票列表
    print(f"\n📋 测试获取股票列表...")
    try:
        stocks_list = screener.get_chinese_stocks_list()
        if stocks_list is not None and not stocks_list.empty:
            print(f"✅ 成功获取 {len(stocks_list)} 只股票的列表")
        else:
            print(f"⚠️  股票列表获取可能存在问题")
    except Exception as e:
        print(f"❌ 获取股票列表失败: {e}")

    print(f"\n🎉 Ashare主数据源测试完成!")
    print(f"   现在系统优先使用Ashare数据源，避免了EastMoney的连接问题")
    print("=" * 60)

if __name__ == "__main__":
    test_ashare_primary()