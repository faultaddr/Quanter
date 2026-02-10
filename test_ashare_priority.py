#!/usr/bin/env python3
"""
Test script to verify ASHare data source priority
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.screeners.stock_screener import StockScreener

def test_ashare_priority():
    print("🔍 测试Ashare数据源优先级...")

    # 初始化股票筛选器
    screener = StockScreener()

    # 测试获取一些股票数据，默认应该使用Ashare
    test_symbols = ['sh600023', 'sz000001', 'sh600519']  # 选择一些常见的A股代码

    for symbol in test_symbols:
        print(f"\n🧪 测试股票: {symbol}")

        # 使用默认数据源（现在应该是Ashare优先）
        data = screener.fetch_stock_data(symbol, period='30', data_source='auto')

        if data is not None and not data.empty:
            print(f"✅ 成功获取 {symbol} 的 {len(data)} 条数据记录")
            if 'close' in data.columns:
                print(f"   当前价格: {data['close'].iloc[-1]:.2f}")
        else:
            print(f"❌ 无法获取 {symbol} 的数据")

    # 现在测试明确指定Ashare数据源
    print(f"\n🎯 明确指定使用Ashare数据源:")
    for symbol in test_symbols:
        print(f"\n🧪 测试股票: {symbol} (明确指定Ashare)")

        data = screener.fetch_stock_data(symbol, period='30', data_source='ashare')

        if data is not None and not data.empty:
            print(f"✅ 成功获取 {symbol} 的 {len(data)} 条数据记录")
            if 'close' in data.columns:
                print(f"   当前价格: {data['close'].iloc[-1]:.2f}")
        else:
            print(f"❌ 无法获取 {symbol} 的Ashare数据")

    print(f"\n🎉 Ashare数据源优先级测试完成!")

if __name__ == "__main__":
    test_ashare_priority()