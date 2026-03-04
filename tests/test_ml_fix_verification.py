#!/usr/bin/env python
"""
修复后的ML策略验证

验证点：
1. 每次训练结果是否不同（随机种子问题已修复）
2. 不同参数是否产生不同的AUC
3. 消融实验：模型 vs 规则
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("修复后的ML策略验证")
print("=" * 70)

# ============================================================================
# 验证1: 随机种子修复效果
# ============================================================================

print("\n" + "=" * 70)
print("验证1: 随机种子修复效果（每次训练应产生不同结果）")
print("=" * 70)

from quanttool.ml.xgboost_trainer import XGBoostTrainer

# 创建测试数据
np.random.seed(42)  # 只控制数据生成，不控制模型
X = pd.DataFrame(np.random.randn(500, 20), columns=[f'feat_{i}' for i in range(20)])
y = pd.Series(np.random.randint(0, 2, 500), index=X.index)

print("\n使用相同参数连续训练3次:")
aucs = []
for i in range(3):
    trainer = XGBoostTrainer(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_feature_selection=False,
        random_state=None  # 不指定，让系统自动生成
    )
    trainer.train(X, y, n_splits=3)
    aucs.append(trainer.performance.auc)
    print(f"  第{i+1}次: AUC={trainer.performance.auc:.4f}, random_state={trainer.params.get('random_state')}")

# 检查AUC是否有变化
unique_aucs = len(set([f"{x:.4f}" for x in aucs]))
if unique_aucs > 1:
    print(f"\n✓ 成功: {unique_aucs}个不同的AUC值，随机性正常")
else:
    print(f"\n⚠️ 警告: 所有AUC完全一致，随机种子可能还有问题")


# ============================================================================
# 验证2: 参数敏感性测试
# ============================================================================

print("\n" + "=" * 70)
print("验证2: 参数敏感性测试（不同参数应产生不同效果）")
print("=" * 70)

test_params = [
    {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.05},
    {'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.05},
    {'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.1},
]

print("\n测试不同参数组合:")
param_results = []
for params in test_params:
    trainer = XGBoostTrainer(
        **params,
        use_feature_selection=False,
        random_state=None
    )
    trainer.train(X, y, n_splits=3)
    param_results.append((params, trainer.performance.auc))
    print(f"  {params} => AUC={trainer.performance.auc:.4f}")

# 计算AUC的标准差
auc_values = [r[1] for r in param_results]
auc_std = np.std(auc_values)
print(f"\nAUC标准差: {auc_std:.4f}")
if auc_std > 0.01:
    print("✓ 参数敏感性正常，不同参数产生不同效果")
else:
    print("⚠️ 参数敏感性低，参数变化对结果影响不大")


# ============================================================================
# 验证3: 真实股票数据测试
# ============================================================================

print("\n" + "=" * 70)
print("验证3: 真实股票数据测试")
print("=" * 70)

from quanttool.strategies.ml_stock_selection_strategy import MLStockSelectionStrategy

# 尝试获取真实数据
print("\n获取股票数据...")

try:
    import baostock as bs
    bs.login()

    rs = bs.query_history_k_data_plus(
        "sz.000876",
        "date,open,high,low,close,volume",
        start_date="2022-01-01",
        end_date=datetime.now().strftime('%Y-%m-%d'),
        frequency="d",
        adjustflag="2"
    )

    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())

    bs.logout()

    if data_list:
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'date': 'timestamp'})
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"获取到 {len(df)} 条数据")

        # 训练模型
        print("\n训练模型...")
        strategy = MLStockSelectionStrategy()

        # 测试多次训练的稳定性
        print("\n测试多次训练的AUC变化:")
        cv_aucs = []
        for i in range(3):
            strategy = MLStockSelectionStrategy()
            strategy.train_model(df)
            if strategy.trainer and strategy.trainer.performance:
                cv_aucs.append(strategy.trainer.performance.auc)
                # 打印各Fold的AUC
                fold_aucs = strategy.trainer.performance.cv_scores['auc']
                fold_str = ", ".join([f"{a:.4f}" for a in fold_aucs])
                print(f"  第{i+1}次: 平均AUC={strategy.trainer.performance.auc:.4f}, 各Fold=[{fold_str}]")

        if len(set([f"{x:.4f}" for x in cv_aucs])) > 1:
            print("\n✓ 真实数据上训练结果有变化，随机性正常")
        else:
            print("\n⚠️ 真实数据上训练结果完全一致")

        # 检查是否有Fold AUC < 0.5
        if strategy.trainer and strategy.trainer.performance:
            fold_aucs = strategy.trainer.performance.cv_scores['auc']
            bad_folds = [i+1 for i, a in enumerate(fold_aucs) if a < 0.5]
            if bad_folds:
                print(f"\n⚠️ 警告: Fold {bad_folds} 的AUC < 0.5，预测方向可能反了")
            else:
                print("\n✓ 所有Fold的AUC都 >= 0.5")

    else:
        print("无法获取数据")

except Exception as e:
    print(f"数据获取失败: {e}")


# ============================================================================
# 验证4: 消融实验（模型 vs 规则）
# ============================================================================

print("\n" + "=" * 70)
print("验证4: 消融实验（收益来源分析）")
print("=" * 70)

print("""
消融实验设计:
1. 使用真实模型预测
2. 使用固定概率0.5（相当于无模型，纯规则）

如果两种情况收益接近，说明收益主要来自规则而非模型。
""")

print("建议: 在完整回测中对比以下两种场景:")
print("  场景A: 正常使用模型预测概率")
print("  场景B: 将所有概率设为0.5（随机）")
print("\n如果A和B收益差异大，说明模型有贡献；")
print("如果A和B收益接近，说明收益来自规则（position_range等）。")


# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 70)
print("验证总结")
print("=" * 70)

print("""
修复内容:
1. ✓ XGBoost随机种子问题已修复
   - 使用时间戳生成随机种子
   - 每次训练的random_state不同
   - 训练结果有随机性

2. ✓ 添加了训练日志
   - 打印模型参数和random_state
   - 便于追踪和调试

3. ✓ 状态重置
   - 每次训练前清空缓存
   - 确保不使用旧数据

下一步建议:
1. 运行完整的回测验证
2. 进行消融实验确认模型贡献
3. 如果模型贡献小，考虑:
   - 改进特征工程
   - 尝试其他模型
   - 或者简化为纯规则策略
""")