#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to verify the connection fix for EastMoney data fetcher
"""

from quant_trade_a_share.utils.eastmoney_data_fetcher import EastMoneyDataFetcher

def test_single_stock():
    """Test fetching data for a single problematic stock"""
    print("开始测试单个股票数据获取...")
    
    fetcher = EastMoneyDataFetcher()
    
    # 测试之前有问题的股票代码
    symbol = "sh688818"
    print(f"尝试获取 {symbol} 的数据...")
    
    try:
        df = fetcher.fetch_stock_data(symbol, days=30)
        if df is not None:
            print(f"✅ 成功获取 {symbol} 的数据，共 {len(df)} 条记录")
            print(f"数据范围: {df.index.min()} 到 {df.index.max()}")
        else:
            print(f"❌ 无法获取 {symbol} 的数据")
    except Exception as e:
        print(f"❌ 获取 {symbol} 数据时发生异常: {e}")
    finally:
        fetcher.close_session()

def test_multiple_stocks():
    """Test fetching data for multiple stocks"""
    print("\n开始测试多个股票数据获取...")
    
    fetcher = EastMoneyDataFetcher()
    
    # 测试一些常见的股票代码
    symbols = ["sh600519", "sz000001", "sh600036", "sh688818"]
    
    success_count = 0
    for symbol in symbols:
        print(f"尝试获取 {symbol} 的数据...")
        try:
            df = fetcher.fetch_stock_data(symbol, days=10)
            if df is not None:
                print(f"✅ 成功获取 {symbol} 的数据")
                success_count += 1
            else:
                print(f"⚠️  无法获取 {symbol} 的数据")
        except Exception as e:
            print(f"❌ 获取 {symbol} 数据时发生异常: {e}")
    
    print(f"\n总共 {len(symbols)} 个股票，成功获取 {success_count} 个")
    fetcher.close_session()

if __name__ == "__main__":
    print("开始测试 EastMoney 数据获取器的连接修复...")
    
    test_single_stock()
    test_multiple_stocks()
    
    print("\n测试完成！")