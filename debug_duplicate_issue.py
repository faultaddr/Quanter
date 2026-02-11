#!/usr/bin/env python3
"""
详细诊断重复索引错误的脚本
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def diagnose_duplicate_index_issue():
    """诊断重复索引错误的根本原因"""

    print("🔍 诊断重复索引问题...")

    # 创建有重复索引的数据
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    prices = 100 + np.cumsum(np.random.randn(50) * 0.3)

    test_data = pd.DataFrame({
        'high': prices * (1 + np.abs(np.random.randn(50)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(50)) * 0.01),
        'open': prices + np.random.randn(50) * 0.05,
        'close': prices,
        'volume': np.random.randint(100000, 500000, 50)
    }, index=dates)

    # 人为制造重复索引
    duplicate_rows = test_data.iloc[[5, 15, 25]]
    duplicate_rows.index = [test_data.index[5]] * 3  # 制造3个重复索引
    test_data_with_duplicates = pd.concat([test_data, duplicate_rows])

    print(f"原始数据形状: {test_data.shape}")
    print(f"带重复索引的数据形状: {test_data_with_duplicates.shape}")
    print(f"重复索引的数量: {test_data_with_duplicates.index.duplicated().sum()}")

    # 检查索引情况
    duplicated_indices = test_data_with_duplicates.index[test_data_with_duplicates.index.duplicated(keep=False)]
    print(f"重复的索引值: {duplicated_indices.unique()}")

    # 检查是否所有重复索引都在同一位置
    print("重复索引详细信息:")
    for idx in duplicated_indices.unique():
        pos = test_data_with_duplicates.index.get_loc(idx)
        if isinstance(pos, slice) or hasattr(pos, '__len__'):
            if hasattr(pos, '__len__') and len(pos) > 1:
                print(f"  索引 '{idx}' 出现在位置: {pos}")
            elif isinstance(pos, slice):
                print(f"  索引 '{idx}' 是切片: {pos}")

    # 检查我们的修复方法
    print("\n🔧 测试修复方法...")

    # 方法1: 使用drop_duplicates
    fixed_data_v1 = test_data_with_duplicates.copy()
    print(f"修复前数据形状: {fixed_data_v1.shape}")
    fixed_data_v1 = fixed_data_v1[~fixed_data_v1.index.duplicated(keep='first')]
    print(f"使用 drop_duplicates 后数据形状: {fixed_data_v1.shape}")
    print(f"修复后重复索引数量: {fixed_data_v1.index.duplicated().sum()}")

    # 方法2: 重置索引然后用整数替代
    fixed_data_v2 = test_data_with_duplicates.copy()
    print(f"\n使用 reset_index 后数据形状: {fixed_data_v2.shape}")
    fixed_data_v2_clean = fixed_data_v2.reset_index().drop_duplicates(subset=['index'], keep='first').set_index('index')
    print(f"reset_index 修复后数据形状: {fixed_data_v2_clean.shape}")
    print(f"修复后重复索引数量: {fixed_data_v2_clean.index.duplicated().sum()}")

    # 方法3: 创建新索引
    fixed_data_v3 = test_data_with_duplicates.copy()
    print(f"\n使用 RangeIndex 修复前数据形状: {fixed_data_v3.shape}")
    fixed_data_v3.index = pd.RangeIndex(len(fixed_data_v3))
    print(f"RangeIndex 修复后数据形状: {fixed_data_v3.shape}")
    print(f"修复后重复索引数量: {fixed_data_v3.index.duplicated().sum()}")

    return fixed_data_v1

def fix_model_fusion_for_duplicates():
    """更新 model_fusion.py 中的修复逻辑"""

    # 读取当前文件
    with open('/root/Quanter/quant_trade_a_share/models/model_fusion.py', 'r') as f:
        content = f.read()

    # 替换 _basic_ml_signals 方法以使用更好的去重策略
    new_method = '''
    def _basic_ml_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        基础机器学习信号计算
        """
        # 确保数据索引唯一：删除重复索引保留第一次出现
        if data.index.duplicated().any():
            print("🔍 检测到重复索引，正在处理...")
            original_length = len(data)
            data = data[~data.index.duplicated(keep='first')]
            print(f"✅ 从 {original_length} 行清理到 {len(data)} 行")

        signals = pd.Series(0.0, index=data.index)

        try:
            # 创建基础特征
            features = pd.DataFrame(index=data.index)
            features['close_lag1'] = data['close'].shift(1)
            features['pct_chg'] = data['close'].pct_change()
            features['volume_lag1'] = data['volume'].shift(1)
            features['volume_pct_chg'] = data['volume'].pct_change()

            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

            # 防止除零和重复索引
            rs = gain / (loss + 1e-10)  # 添加小值防止除零
            rsi = 100 - (100 / (1 + rs))

            # 确保rsi与data索引一致（这里可能是关键问题所在）
            if not rsi.index.equals(data.index):
                rsi = rsi.reindex(data.index)
            features['rsi'] = rsi

            # MACD
            exp12 = data['close'].ewm(span=12).mean()
            exp26 = data['close'].ewm(span=26).mean()
            macd = exp12 - exp26
            signal_line = macd.ewm(span=9).mean()

            # 确保MACD相关指标与data索引一致
            if not macd.index.equals(data.index):
                macd = macd.reindex(data.index)
            if not signal_line.index.equals(data.index):
                signal_line = signal_line.reindex(data.index)
            features['macd'] = macd
            features['macd_signal'] = signal_line

            # 删除含 NaN 的行
            features = features.dropna()

            if len(features) > 10:  # 需要有足够的数据点
                # 基于 RSI 的 ML 信号
                rsi_values = features['rsi']
                rsi_sig = pd.Series(0.0, index=features.index)

                # 确保索引一致后再进行shift操作
                rsi_shifted = rsi_values.shift(1)
                if not rsi_shifted.index.equals(rsi_values.index):
                    rsi_shifted = rsi_shifted.reindex(rsi_values.index)

                rsi_sig[(rsi_values < 30) & (rsi_shifted >= 30)] = 0.8  # 超卖
                rsi_sig[(rsi_values > 70) & (rsi_shifted <= 70)] = -0.8  # 超买
                signals.loc[features.index] += rsi_sig

                # 基于 MACD 的 ML 信号
                macd_values = features['macd']
                macd_signal_values = features['macd_signal']
                macd_sig = pd.Series(0.0, index=features.index)

                # 确保索引一致后再进行shift操作
                macd_shifted = macd_values.shift(1)
                macd_signal_shifted = macd_signal_values.shift(1)

                if not macd_shifted.index.equals(macd_values.index):
                    macd_shifted = macd_shifted.reindex(macd_values.index)
                if not macd_signal_shifted.index.equals(macd_signal_values.index):
                    macd_signal_shifted = macd_signal_shifted.reindex(macd_signal_values.index)

                macd_sig[(macd_values > macd_signal_values) &
                        (macd_shifted <= macd_signal_shifted)] = 0.6  # 金叉
                macd_sig[(macd_values < macd_signal_values) &
                        (macd_shifted >= macd_signal_shifted)] = -0.6  # 死叉
                signals.loc[features.index] += macd_sig

                # 趋势信号
                trend_sig = pd.Series(0.0, index=features.index)
                trend_sig[features['pct_chg'] > 0.02] = 0.4  # 上涨趋势
                trend_sig[features['pct_chg'] < -0.02] = -0.4  # 下跌趋势
                signals.loc[features.index] += trend_sig

        except Exception as e:
            print(f"⚠️ 基础 ML 信号计算失败: {e}")
            import traceback
            traceback.print_exc()

        return signals
'''

    # 替换方法
    import re
    pattern = r'def _basic_ml_signals\(self, data: pd\.DataFrame\) -> pd\.Series:[\s\S]*?(?=^\s*def|\Z)'
    updated_content = re.sub(pattern, new_method.strip(), content, count=1, flags=re.MULTILINE)

    # 写回文件
    with open('/root/Quanter/quant_trade_a_share/models/model_fusion.py', 'w') as f:
        f.write(updated_content)

    print("✅ 已更新 _basic_ml_signals 方法")


if __name__ == "__main__":
    print("🔧 详细诊断重复索引错误")
    print("="*50)

    diagnose_duplicate_index_issue()
    fix_model_fusion_for_duplicates()

    print("\n✅ 修复完成，接下来可以重新测试")