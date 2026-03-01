"""
CLI module for stock analysis
Allows users to run technical analysis on stocks directly from command line
"""
import argparse
import sys
import os
from datetime import datetime

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from quanttool.factors.stock_analyzer import StockAnalyzer


def main():
    parser = argparse.ArgumentParser(description='Analyze stock with technical indicators and trading strategies.')
    parser.add_argument('symbol', help='Stock symbol to analyze (e.g., 601777, 000001.SZ)')
    parser.add_argument('--days', type=int, default=360, help='Number of days to analyze (default: 360)')
    parser.add_argument('--output', '-o', help='Output file to save the analysis report')

    args = parser.parse_args()

    print(f"正在分析股票：{args.symbol}")
    print(f"分析周期：{args.days} 天")
    print("-" * 50)

    # Create analyzer instance
    analyzer = StockAnalyzer()

    # Run analysis
    report = analyzer.analyze_stock(args.symbol, args.days)

    # Print report
    print(report)

    # Save to file if requested
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n分析报告已保存至：{args.output}")


if __name__ == "__main__":
    main()