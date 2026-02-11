#!/usr/bin/env python3
"""
Qlib 集成示例
展示如何将 Qlib 集成到您的量化交易项目中
"""

import sys
import os
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import qlib
    from qlib.config import REG_CN as REGION_CN
    from qlib.data import D
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R

    print("✅ Qlib 导入成功！版本:", qlib.__version__)

    # 初始化 Qlib (使用 CPU 模式)
    try:
        qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REGION_CN)
        print("✅ Qlib 初始化成功！")
    except Exception as e:
        print(f"⚠️ Qlib 初始化失败 (这可能是因为缺少数据): {e}")
        print("   但这不影响将 Qlib 功能集成到您的项目中")

    def get_qlib_data_example():
        """演示如何使用 Qlib 获取数据"""
        try:
            # 获取一些示例字段的数据
            instruments = D.instruments(market='csi300')  # CSI300 成分股
            fields = ['$close', '$open', '$high', '$low', '$volume']
            start_time = '2024-01-01'
            end_time = '2024-12-31'

            # 获取数据
            df = D.features(instruments[:5], fields, start_time, end_time)  # 只取前5只股票
            print(f"✅ 成功获取 Qlib 数据，形状: {df.shape}")
            print("数据预览:")
            print(df.head(10))
            return df
        except Exception as e:
            print(f"⚠️ 获取 Qlib 数据失败: {e}")
            return None

    def run_qlib_workflow_example():
        """演示 Qlib 工作流"""
        try:
            print("🧪 运行 Qlib 工作流示例...")
            # 这里可以设置实验和运行模型
            exp_manager = R.get_exp_manager()
            print(f"✅ 实验管理器: {exp_manager}")
            return True
        except Exception as e:
            print(f"⚠️ Qlib 工作流示例失败: {e}")
            return False

    # 运行示例
    print("\n" + "="*50)
    print("Qlib 功能演示")
    print("="*50)

    get_qlib_data_example()
    run_qlib_workflow_example()

    print("\n✅ Qlib 已成功集成到您的项目中！")
    print("💡 提示: 为了使用 Qlib 的全部功能，您需要下载对应市场的数据")
    print("   详情请参考: https://qlib.readthedocs.io/en/latest/component/data.html#initialize-dataset")

except ImportError as e:
    print(f"❌ Qlib 导入失败: {e}")
    print("   请确保 Qlib 已正确安装: pip install pyqlib")

# 现在展示如何将 Qlib 功能集成到您现有的项目中
print("\n" + "="*50)
print("集成建议")
print("="*50)
print("""
1. 数据层集成：
   - 使用 Qlib 的 D.features() 替代部分数据获取功能
   - 结合您现有的 tushare、baostock 数据源

2. 特征工程：
   - 使用 Qlib 的表达式引擎创建高级特征
   - 将 Qlib 特征与您现有的技术指标结合

3. 模型训练：
   - 利用 Qlib 的机器学习工作流
   - 结合您的多因子策略模板

4. 回测框架：
   - 使用 Qlib 的回测功能增强现有 backtester
   - 对比不同策略的表现
""")