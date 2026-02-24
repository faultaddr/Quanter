#!/usr/bin/env python3
"""
Check the actual name for stock 601777.
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, '/root/CuferPan/quanttool')

from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials


def check_stock_name():
    """Check the actual name for stock 601777."""

    print("Checking the actual name for stock 601777...")

    # Create the data fetcher with credentials
    print("\nInitializing Enhanced DataFetcher...")
    fetcher = create_data_fetcher_with_credentials()
    fetcher.initialize()

    print("Searching for stock 601777...")

    # Search for the stock
    search_results = fetcher.search_symbols("601777")

    if search_results:
        print(f"Found {len(search_results)} matching stocks:")
        for result in search_results:
            print(f"  - {result['symbol']}: {result['name']} (Area: {result['area']}, Industry: {result['industry']})")
    else:
        print("No results found for 601777, trying variations...")

        # Try searching with different formats
        for query in ["国投", "千里", "601777"]:
            print(f"\nSearching for '{query}':")
            search_results = fetcher.search_symbols(query)

            if search_results:
                print(f"Found {len(search_results)} matching stocks:")
                for result in search_results[:5]:  # Show first 5 matches
                    print(f"  - {result['symbol']}: {result['name']} (Area: {result['area']}, Industry: {result['industry']})")
                break
            else:
                print(f"No results found for '{query}'")


if __name__ == "__main__":
    check_stock_name()