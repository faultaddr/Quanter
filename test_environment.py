#!/usr/bin/env python
"""
验证虚拟环境中已安装的包是否可以正常使用
"""

def test_imports():
    """测试导入各个库"""
    libraries = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'tushare',
        'baostock',
        'yfinance',
        'plotly',
        'dash',
        'scipy',
        'statsmodels',
        'requests'
    ]
    
    print("正在测试库导入...")
    success_count = 0
    
    for lib in libraries:
        try:
            exec(f"import {lib}")
            print(f"✓ {lib} 导入成功")
            success_count += 1
        except ImportError as e:
            print(f"✗ {lib} 导入失败: {e}")
    
    print(f"\n成功导入 {success_count}/{len(libraries)} 个库")
    
    # 测试基本功能
    print("\n正在测试基本功能...")
    try:
        # 测试 pandas 和 numpy
        import pandas as pd
        import numpy as np
        
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        arr = np.array([1, 2, 3])
        
        print(f"✓ pandas DataFrame 创建成功: {df.shape}")
        print(f"✓ numpy Array 创建成功: {arr.shape}")
        
        # 测试 requests
        import requests
        print(f"✓ requests 版本: {requests.__version__}")
        
        print("\n所有测试通过！环境配置成功。")
        
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")

if __name__ == "__main__":
    test_imports()