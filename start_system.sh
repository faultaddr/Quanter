#!/bin/bash
# Quick Start Script for A-Share Quantitative Trading System

echo "🚀 A-Share Quantitative Trading System - Quick Start"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking required packages..."
python3 -c "import pandas, numpy, requests, qlib, tushare, baostock" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Some required packages are missing. Installing..."
    pip install -r requirements.txt
fi

echo "✅ All required packages are available"

echo ""
echo "💡 Available Commands:"
echo "   ./start_system.sh interactive    - Start interactive CLI"
echo "   ./start_system.sh screen         - Screen for stocks"
echo "   ./start_system.sh analyze        - Analyze a stock"
echo "   ./start_system.sh backtest       - Run backtesting"
echo "   ./start_system.sh demo           - Run system demo"
echo ""

case "$1" in
    "interactive")
        echo "🎮 Starting Interactive Mode..."
        python3 cli_interface.py --mode interactive
        ;;
    "screen")
        echo "🔍 Screening for stocks..."
        python3 cli_interface.py --mode screen
        ;;
    "analyze")
        echo "📊 Analyzing stocks..."
        python3 cli_interface.py --mode analyze
        ;;
    "backtest")
        echo "📈 Running backtesting..."
        python3 cli_interface.py --mode backtest
        ;;
    "demo")
        echo "🎬 Running system demo..."
        python3 demo_system.py
        ;;
    *)
        echo "🤖 Starting Interactive Mode (default)..."
        echo "   Type 'help' to see all available commands"
        python3 cli_interface.py --mode interactive
        ;;
esac