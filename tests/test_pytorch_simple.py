#!/usr/bin/env python
"""
简化版 PyTorch 模型回测测试
逐步测试找出 segmentation fault 的原因
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import gc
warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

print("=" * 60)
print("Step 1: 导入模块")
print("=" * 60)

from quanttool.strategies.qlib_strategy import QlibStrategy
from quanttool.backtest.engine import BacktestEngine
from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher

print("✅ 模块导入成功")

print("\n" + "=" * 60)
print("Step 2: 获取股票数据")
print("=" * 60)

# 只测试一只股票
TEST_STOCK = '000876'
end_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')

df = AshareFetcher.get_price(
    code=TEST_STOCK,
    end_date=end_date,
    count=365 * 2,
    frequency='1d'
)

if 'timestamp' not in df.columns:
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"✅ 获取 {TEST_STOCK} 数据: {len(df)} 条")
print(f"   日期范围: {df['timestamp'].iloc[0]} ~ {df['timestamp'].iloc[-1]}")

print("\n" + "=" * 60)
print("Step 3: 测试 LSTM 模型训练")
print("=" * 60)

try:
    strategy = QlibStrategy(
        feature_set='Alpha158',
        model_type='lstm',
        buy_threshold=0.55,
        sell_threshold=0.45,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
        epochs=10,
        hidden_size=64,
        num_layers=2,
        device='cpu',  # 强制 CPU
    )

    # 训练
    train_size = int(len(df) * 0.3)  # 减少训练数据量
    train_data = df.iloc[:train_size]
    print(f"   训练数据: {len(train_data)} 条")

    strategy.train_model(train_data, horizon=10)
    print("✅ LSTM 训练成功")

except Exception as e:
    print(f"❌ 训练失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 4: 测试回测引擎")
print("=" * 60)

try:
    # 清理内存
    gc.collect()

    engine = BacktestEngine()
    engine.set_initial_cash(100000.0)
    engine.set_commission_rate(0.0003)
    engine.set_t_plus_1(True)

    data = {TEST_STOCK: df.copy()}
    print(f"   回测数据: {len(df)} 条")

    result = engine.run_backtest(
        strategy=strategy,
        data=data,
        start_date=df['timestamp'].iloc[0],
        end_date=df['timestamp'].iloc[-1]
    )

    print("✅ 回测完成")
    print(f"   年化收益: {result.annual_return*100:.2f}%")
    print(f"   总收益: {result.total_return*100:.2f}%")
    print(f"   夏普比: {result.sharpe_ratio:.2f}")
    print(f"   最大回撤: {result.max_drawdown*100:.2f}%")
    print(f"   胜率: {result.win_rate*100:.1f}%")
    print(f"   交易次数: {result.total_trades}")

except Exception as e:
    print(f"❌ 回测失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 清理
del strategy
del engine
gc.collect()

print("\n" + "=" * 60)
print("✅ 所有测试通过！LSTM 模型可以正常工作")
print("=" * 60)