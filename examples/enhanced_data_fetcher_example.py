#!/usr/bin/env python3
"""
Example script demonstrating the usage of EnhancedDataFetcher with Tushare and EastMoney integration.
"""

import os
import sys
from datetime import datetime, timedelta

# 添加项目根目录到Python路径
sys.path.insert(0, '/root/CuferPan/quanttool')

from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials, EnhancedDataFetcher


def main():
    print("Initializing Enhanced DataFetcher with integrated Tushare and EastMoney support...")

    # 创建集成的数据获取器实例
    fetcher = create_data_fetcher_with_credentials()

    # 初始化数据提供器
    fetcher.initialize()

    print("DataFetcher initialized successfully!")
    print()

    # 示例1: 获取指定股票的历史数据
    print("Example 1: Fetching historical data for 000001.SZ (Ping An Bank)")
    start_date = datetime.now() - timedelta(days=30)  # 过去30天
    end_date = datetime.now()

    symbols = ["000001.SZ"]
    data = fetcher.get_bars(symbols, start_date, end_date)

    if "000001.SZ" in data and not data["000001.SZ"].empty:
        df = data["000001.SZ"]
        print(f"Retrieved {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")
        print("Sample data:")
        print(df.head())
        print()
    else:
        print("No data retrieved for 000001.SZ")
        print()

    # 示例2: 获取最新交易数据
    print("Example 2: Fetching latest bar for 000001.SZ")
    latest_bar = fetcher.get_latest_bar("000001.SZ")
    if latest_bar is not None and not latest_bar.empty:
        print("Latest bar data:")
        print(latest_bar)
        print()
    else:
        print("No latest bar data found for 000001.SZ")
        print()

    # 示例3: 搜索股票
    print("Example 3: Searching for stocks with '平安' (Ping An)")
    search_results = fetcher.search_symbols("平安")
    if search_results:
        print(f"Found {len(search_results)} matching stocks:")
        for result in search_results[:5]:  # 显示前5个结果
            print(f"  - {result['symbol']}: {result['name']} (Industry: {result['industry']})")
        print()
    else:
        print("No stocks found matching '平安'")
        print()


if __name__ == "__main__":
    main()