#!/usr/bin/env python3
"""
Main Entry Point for A-Share Market Analysis System
Uses the unified CLI interface for all functionality
"""
import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cli_interface import main

if __name__ == "__main__":
    print("🔍 A股市场分析系统 - 主入口点")
    print("="*50)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Call the unified CLI interface main function
    main()