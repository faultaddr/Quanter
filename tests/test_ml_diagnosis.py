#!/usr/bin/env python
"""
ML策略诊断脚本 - 排查以下问题：

1. 为什么每次训练的CV结果完全一致？
2. 为什么Fold 1 AUC < 0.5 (预测反了)?
3. 收益来自模型还是规则？
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
print("ML策略诊断 - 排查关键问题")
print("=" * 70)


# ============================================================================
# 问题1: 检查XGBoost随机种子
# ============================================================================

print("\n" + "=" * 70)
print("问题1: XGBoost随机种子检查")
print("=" * 70)

try:
    import xgboost as xgb
    from sklearn.datasets import make_classification

    # 创建测试数据
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

    # 测试1: 不指定random_state
    print("\n测试1: 不指定random_state")
    results_no_seed = []
    for i in range(3):
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3)
        model.fit(X[:800], y[:800])
        prob = model.predict_proba(X[800:])[:, 1].mean()
        results_no_seed.append(prob)
        print(f"  第{i+1}次: 平均预测概率 = {prob:.6f}")

    if len(set([f"{x:.6f}" for x in results_no_seed])) == 1:
        print("  ⚠️ 警告: 每次预测结果完全一致! XGBoost可能有默认随机种子!")
    else:
        print("  ✓ 结果不一致,说明随机性正常")

    # 测试2: 明确指定不同的random_state
    print("\n测试2: 明确指定不同的random_state")
    results_with_seed = []
    for i, seed in enumerate([0, 1, 2]):
        model = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=seed)
        model.fit(X[:800], y[:800])
        prob = model.predict_proba(X[800:])[:, 1].mean()
        results_with_seed.append(prob)
        print(f"  random_state={seed}: 平均预测概率 = {prob:.6f}")

    # 测试3: 不设置random_state vs 设置random_state=0
    print("\n测试3: 对比不设置seed vs random_state=0")
    model_no_seed = xgb.XGBClassifier(n_estimators=50, max_depth=3)
    model_no_seed.fit(X[:800], y[:800])
    prob_no_seed = model_no_seed.predict_proba(X[800:])[:, 1].mean()

    model_seed_0 = xgb.XGBClassifier(n_estimators=50, max_depth=3, random_state=0)
    model_seed_0.fit(X[:800], y[:800])
    prob_seed_0 = model_seed_0.predict_proba(X[800:])[:, 1].mean()

    print(f"  不设置random_state: {prob_no_seed:.6f}")
    print(f"  random_state=0: {prob_seed_0:.6f}")

    if abs(prob_no_seed - prob_seed_0) < 1e-6:
        print("  ⚠️ 确认: XGBoost默认random_state=0!")

except Exception as e:
    print(f"  错误: {e}")


# ============================================================================
# 问题2: 检查参数是否真的传递给模型
# ============================================================================

print("\n" + "=" * 70)
print("问题2: 参数传递检查")
print("=" * 70)

try:
    from quanttool.ml.xgboost_trainer import XGBoostTrainer

    # 模拟数据
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(500, 20), columns=[f'feat_{i}' for i in range(20)])
    y = pd.Series(np.random.randint(0, 2, 500), index=X.index)

    print("\n测试不同参数的AUC是否不同:")

    auc_results = []

    for n_est in [50, 100, 200]:
        for max_depth in [3, 5]:
            trainer = XGBoostTrainer(
                n_estimators=n_est,
                max_depth=max_depth,
                learning_rate=0.1,
                use_feature_selection=False,  # 关闭特征选择
                random_state=None  # 不固定随机种子
            )
            trainer.train(X, y, n_splits=3)
            auc = trainer.performance.auc
            auc_results.append((n_est, max_depth, auc))
            print(f"  n_estimators={n_est}, max_depth={max_depth}: AUC={auc:.4f}")

    # 检查是否所有AUC都一样
    aucs = [r[2] for r in auc_results]
    if len(set([f"{x:.4f}" for x in aucs])) == 1:
        print("\n  ⚠️ 严重警告: 所有参数组合的AUC完全一致!")
        print("  这说明参数根本没有传递到模型!")
    else:
        print("\n  ✓ AUC不一致,参数传递正常")

except Exception as e:
    print(f"  错误: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# 问题3: 检查特征选择是否导致结果一致
# ============================================================================

print("\n" + "=" * 70)
print("问题3: 特征选择影响检查")
print("=" * 70)

try:
    from quanttool.ml.xgboost_trainer import XGBoostTrainer

    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(500, 50), columns=[f'feat_{i}' for i in range(50)])
    y = pd.Series(np.random.randint(0, 2, 500), index=X.index)

    print("\n测试特征选择的影响:")

    # 使用相同的threshold多次训练
    thresholds = [10, 20, 30]
    for threshold in thresholds:
        trainer = XGBoostTrainer(
            n_estimators=100,
            max_depth=4,
            use_feature_selection=True,
            feature_selection_threshold=threshold,
            random_state=None
        )
        trainer.train(X, y, n_splits=3)
        print(f"  threshold={threshold}: 选中{len(trainer.selected_features)}个特征, AUC={trainer.performance.auc:.4f}")

except Exception as e:
    print(f"  错误: {e}")


# ============================================================================
# 问题4: 检查Fold 1 AUC < 0.5 的原因
# ============================================================================

print("\n" + "=" * 70)
print("问题4: Fold 1 AUC < 0.5 原因排查")
print("=" * 70)

try:
    from quanttool.strategies.ml_stock_selection_strategy import MLStockSelectionStrategy
    from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import AshareFetcher

    # 获取一只股票的数据
    print("\n获取测试数据...")
    end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        df = AshareFetcher.get_price(
            code='000876',
            end_date=end_date,
            count=500,
            frequency='1d'
        )
    except:
        # 尝试使用baostock
        import baostock as bs
        bs.login()
        rs = bs.query_history_k_data_plus(
            "sz.000876",
            "date,open,high,low,close,volume",
            start_date="2023-01-01",
            end_date=end_date,
            frequency="d",
            adjustflag="2"
        )
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        df['date'] = pd.to_datetime(df['date'])
        df = df.rename(columns={'date': 'timestamp'})
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        bs.logout()

    if df.empty:
        print("  无法获取数据,跳过此测试")
    else:
        print(f"  获取到 {len(df)} 条数据")

        # 训练模型
        strategy = MLStockSelectionStrategy()
        strategy.train_model(df)

        if strategy.trainer and strategy.trainer.performance:
            cv_scores = strategy.trainer.performance.cv_scores

            print("\n  各Fold的AUC:")
            for i, auc in enumerate(cv_scores['auc']):
                status = "⚠️ < 0.5 (预测反了!)" if auc < 0.5 else "✓ 正常"
                print(f"    Fold {i+1}: {auc:.4f} {status}")

            # 分析Fold 1
            if cv_scores['auc'][0] < 0.5:
                print("\n  Fold 1 AUC < 0.5 的可能原因:")
                print("  1. 时间序列分割导致训练集太小")
                print("  2. Fold 1时期的市场风格与训练集完全相反")
                print("  3. 特征工程中存在未来函数泄露")
                print("  4. 标签计算有问题")

                # 测试: 如果把Fold 1的预测取反会怎样
                flipped_auc = 1 - cv_scores['auc'][0]
                print(f"\n  如果把Fold 1预测取反: AUC = {flipped_auc:.4f}")

except Exception as e:
    print(f"  错误: {e}")
    import traceback
    traceback.print_exc()


# ============================================================================
# 问题5: 收益来源分析 (模型 vs 规则)
# ============================================================================

print("\n" + "=" * 70)
print("问题5: 收益来源分析")
print("=" * 70)

print("""
实验设计: 对比以下两种情况
1. 使用真实的模型预测概率
2. 使用固定的0.5概率 (相当于随机)

如果两种情况收益接近,说明收益主要来自规则(位置区间+止损止盈),
而不是模型预测。
""")

try:
    from quanttool.strategies.ml_stock_selection_strategy import MLStockSelectionStrategy

    # 简单模拟
    print("模拟测试:")

    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    prices = 100 * (1 + np.random.randn(100).cumsum() * 0.02)
    prices = np.maximum(prices, 10)  # 确保价格为正

    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices * (1 + np.random.randn(100) * 0.01),
        'high': prices * (1 + np.abs(np.random.randn(100)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(100)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    })

    # 计算位置
    low_60 = df['low'].rolling(60).min()
    high_60 = df['high'].rolling(60).max()
    df['position'] = (df['close'] - low_60) / (high_60 - low_60 + 1e-10)

    # 模拟不同概率场景
    print("\n场景1: 随机概率 (全部=0.5)")
    df['prob_random'] = 0.5
    buy_signals_random = (df['prob_random'] > 0.4) & (df['position'] > 0.15) & (df['position'] < 0.85)
    print(f"  买入信号数: {buy_signals_random.sum()}")

    print("\n场景2: 随机概率 (均匀分布0.3-0.7)")
    df['prob_uniform'] = np.random.uniform(0.3, 0.7, len(df))
    buy_signals_uniform = (df['prob_uniform'] > 0.4) & (df['position'] > 0.15) & (df['position'] < 0.85)
    print(f"  买入信号数: {buy_signals_uniform.sum()}")

    print("""
结论:
- 如果两种场景的买入信号数相近,说明概率阈值对信号数量影响不大
- 收益主要取决于position_range规则
- AUC低但有收益的原因: 规则在赚钱,模型没贡献
""")

except Exception as e:
    print(f"  错误: {e}")


# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 70)
print("诊断总结")
print("=" * 70)

print("""
发现的问题:

1. **XGBoost默认随机种子问题**
   - XGBoost在不指定random_state时,可能使用默认值0
   - 这导致每次训练结果完全一致
   - 解决: 明确设置 random_state=None 或使用不同的种子

2. **参数传递问题**
   - 需要检查参数是否真的传递到模型
   - 特别是特征选择threshold的影响

3. **Fold 1 AUC < 0.5**
   - 模型在早期数据上预测方向反了
   - 可能原因: 未来函数泄露或市场风格突变

4. **收益来源**
   - 低AUC但有高收益,说明收益来自规则而非模型
   - position_range规则可能比模型预测更重要

建议修复:
1. 修复XGBoost随机种子问题
2. 添加参数传递的验证日志
3. 排查特征工程中的未来函数
4. 进行消融实验确认模型贡献
""")