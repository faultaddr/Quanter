#!/usr/bin/env python3
"""
A-Share Market Analysis System - Simplified Runner
This script provides a simplified way to run the system and avoid dependency issues
"""

import sys
import os
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🔍 A股市场分析系统 - 简化启动器")
    print("="*50)
    print(f"启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    try:
        # Attempt to import and run the main system
        from cli_interface import main as cli_main
        print("✅ 主系统导入成功")
        print()

        # Run the CLI interface
        cli_main()

    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        print()
        print("可能的原因及解决方案：")
        print("1. 缺少依赖包 - 请运行: pip install -r requirements.txt")
        print("2. 模块路径问题 - 请检查模块是否存在")
        print("3. 版本兼容性问题 - 请检查Python版本和库版本")
        print()
        print("当前Python版本:", sys.version)

    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()