#!/usr/bin/env python
"""
Qlib 官方 Benchmark 训练脚本

使用官方 Qlib 流程进行模型训练和回测：
1. 使用 qlib.init() 初始化，加载本地 qlib 格式数据
2. 使用 Alpha158 特征处理器
3. 使用官方模型：LightGBM, XGBoost, LSTM, GRU, Transformer 等
4. 使用官方回测框架

参考: https://github.com/microsoft/qlib/tree/main/examples/benchmarks
"""

import sys
import os
import warnings
import argparse
from pathlib import Path
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

# Qlib 相关导入
try:
    import qlib
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.backtest import backtest_executor, position
    from qlib.rl.interpreter import ActionInterpreter
    QLIB_AVAILABLE = True
except ImportError as e:
    print(f"qlib 导入失败: {e}")
    QLIB_AVAILABLE = False


# ============================================================================
# 配置
# ============================================================================

# 数据路径
QLIB_DATA_PATH = str(project_root / "qlib_data" / "cn_data")

# 时间配置
TRAIN_START = "2018-01-01"
TRAIN_END = "2022-12-31"
VALID_START = "2023-01-01"
VALID_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2024-12-31"

# 模型配置
MODEL_CONFIGS = {
    'lgb': {
        'name': 'LightGBM',
        'class': 'LGBModel',
        'module': 'qlib.contrib.model.gbdt',
        'params': {
            'loss': 'mse',
            'colsample_bytree': 0.85,
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 500,
            'num_leaves': 127,
            'subsample': 0.85,
            'random_state': 42,
            'n_jobs': -1,
        }
    },
    'xgboost': {
        'name': 'XGBoost',
        'class': 'XGBModel',
        'module': 'qlib.contrib.model.xgboost',
        'params': {
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'random_state': 42,
            'n_jobs': -1,
        }
    },
    'lstm': {
        'name': 'LSTM',
        'class': 'LSTM',
        'module': 'qlib.contrib.model.pytorch_lstm',
        'params': {
            'd_feat': 6,  # 特征维度，需要能被总特征数整除
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'n_epochs': 50,
            'lr': 0.001,
            'batch_size': 256,
            'early_stop': 10,
            'GPU': 0,
        }
    },
    'gru': {
        'name': 'GRU',
        'class': 'GRU',
        'module': 'qlib.contrib.model.pytorch_gru',
        'params': {
            'd_feat': 6,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'n_epochs': 50,
            'lr': 0.001,
            'batch_size': 256,
            'early_stop': 10,
            'GPU': 0,
        }
    },
    'transformer': {
        'name': 'Transformer',
        'class': 'Transformer',
        'module': 'qlib.contrib.model.pytorch_transformer',
        'params': {
            'd_feat': 6,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dropout': 0.1,
        }
    },
    'gats': {
        'name': 'GATs',
        'class': 'GATs',
        'module': 'qlib.contrib.model.pytorch_gats',
        'params': {
            'd_feat': 6,  # GATs 要求 d_feat=6
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.1,
            'n_epochs': 50,
            'lr': 0.001,
            'early_stop': 10,
            'GPU': 0,
        }
    },
}


# ============================================================================
# Qlib 初始化
# ============================================================================

def init_qlib():
    """初始化 Qlib，使用本地数据"""
    print("=" * 60)
    print("初始化 Qlib")
    print("=" * 60)

    # 检查数据目录
    if not Path(QLIB_DATA_PATH).exists():
        raise FileNotFoundError(f"Qlib 数据目录不存在: {QLIB_DATA_PATH}")

    # 初始化 qlib
    try:
        qlib.init(
            provider_uri=QLIB_DATA_PATH,
            region="cn",
            expression_cache=None,
            dataset_cache=None,
        )
        print(f"✅ Qlib 初始化成功")
        print(f"   数据路径: {QLIB_DATA_PATH}")
    except Exception as e:
        print(f"❌ Qlib 初始化失败: {e}")
        raise

    return True


# ============================================================================
# 数据集创建
# ============================================================================

def create_dataset(
    instruments: str = "all",
    start_time: str = TRAIN_START,
    end_time: str = TEST_END,
    fit_start_time: str = TRAIN_START,
    fit_end_time: str = TRAIN_END,
    eval_start_time: str = VALID_START,
    eval_end_time: str = VALID_END,
):
    """
    创建 Qlib 数据集

    使用 Qlib 0.9.x API
    """
    print(f"\n创建数据集...")
    print(f"  训练集: {fit_start_time} ~ {fit_end_time}")
    print(f"  验证集: {eval_start_time} ~ {eval_end_time}")
    print(f"  测试集: {TEST_START} ~ {TEST_END}")

    try:
        from qlib.data.dataset import DatasetH
        from qlib.data.dataset.handler import DataHandlerLP

        # 创建 DataHandler
        handler = DataHandlerLP(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            label=["Ref($close, -2) / Ref($close, -1) - 1"],  # 未来收益标签
        )

        # 创建数据集
        dataset = DatasetH(
            handler=handler,
            segments={
                "train": (fit_start_time, fit_end_time),
                "valid": (eval_start_time, eval_end_time),
                "test": (TEST_START, TEST_END),
            },
        )

        print(f"✅ 数据集创建成功")
        return dataset

    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_simple_dataset():
    """
    创建简化版数据集（不使用 Alpha158 Handler）

    直接使用已有的特征数据
    """
    print(f"\n创建简化数据集...")

    # 使用简单的 DataHandlerLP
    handler_config = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "start_time": TRAIN_START,
            "end_time": TEST_END,
            "fit_start_time": TRAIN_START,
            "fit_end_time": TRAIN_END,
            "instruments": "all",
            "infer_processors": [
                {"class": "ProcessInf", "module_path": "qlib.data.dataset.processor"},
                {"class": "ZScoreNorm", "module_path": "qlib.data.dataset.processor"},
                {"class": "Fillna", "module_path": "qlib.data.dataset.processor"},
            ],
            "learn_processors": [
                {"class": "DropnaLabel", "module_path": "qlib.data.dataset.processor"},
            ],
            "label": ["Ref($close, -2) / Ref($close, -1) - 1"],  # 未来2日收益率
        },
    }

    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": handler_config,
            "segments": {
                "train": (TRAIN_START, TRAIN_END),
                "valid": (VALID_START, VALID_END),
                "test": (TEST_START, TEST_END),
            },
        },
    }

    try:
        from qlib.utils import init_instance_by_config
        dataset = init_instance_by_config(dataset_config)
        print(f"✅ 数据集创建成功")
        return dataset
    except Exception as e:
        print(f"❌ 数据集创建失败: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# 模型训练
# ============================================================================

def train_model(model_type: str, dataset, save_path: str = None):
    """
    训练模型

    Args:
        model_type: 模型类型 (lgb, xgboost, lstm, gru, transformer, gats)
        dataset: Qlib 数据集
        save_path: 模型保存路径
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型类型: {model_type}. 支持: {list(MODEL_CONFIGS.keys())}")

    config = MODEL_CONFIGS[model_type]
    print(f"\n{'=' * 60}")
    print(f"训练模型: {config['name']}")
    print(f"{'=' * 60}")

    # 导入模型类
    try:
        module = __import__(config['module'], fromlist=[config['class']])
        ModelClass = getattr(module, config['class'])
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        raise

    # 创建模型实例
    model_params = config['params'].copy()

    # 特殊处理 PyTorch 模型的 GPU 参数
    if model_type in ['lstm', 'gru', 'transformer', 'gats']:
        import torch
        if not torch.cuda.is_available():
            model_params['GPU'] = None

    print(f"模型参数: {model_params}")

    try:
        model = ModelClass(**model_params)
        print(f"✅ 模型创建成功")
    except Exception as e:
        print(f"❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 训练模型
    print(f"\n开始训练...")
    start_time = time.time()

    try:
        model.fit(dataset)
        elapsed = time.time() - start_time
        print(f"✅ 训练完成 (耗时: {elapsed:.1f}s)")
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 保存模型
    if save_path:
        try:
            model.to_pickle(save_path)
            print(f"✅ 模型已保存: {save_path}")
        except Exception as e:
            print(f"⚠️ 模型保存失败: {e}")

    return model


# ============================================================================
# 预测和回测
# ============================================================================

def predict_and_backtest(model, dataset, model_name: str = "Model"):
    """
    预测并回测

    Args:
        model: 训练好的模型
        dataset: Qlib 数据集
        model_name: 模型名称
    """
    print(f"\n{'=' * 60}")
    print(f"预测和回测: {model_name}")
    print(f"{'=' * 60}")

    # 预测
    print(f"\n生成预测...")
    try:
        predictions = model.predict(dataset, segment="test")
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.iloc[:, 0]
        print(f"✅ 预测完成，共 {len(predictions)} 条")
        print(f"   预测值范围: [{predictions.min():.4f}, {predictions.max():.4f}]")
    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # 简单回测（使用 Qlib 官方回测框架）
    print(f"\n运行回测...")
    try:
        from qlib.backtest import backtest
        from qlib.contrib.strategy import TopkDropoutStrategy

        # 策略配置
        strategy_config = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
                "signal": predictions,
                "topk": 30,  # 持仓前30只
                "n_drop": 5,  # 每次换仓5只
            },
        }

        # 执行器配置
        executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        }

        # 回测
        backtest_config = {
            "strategy": strategy_config,
            "executor": executor_config,
            "start_time": TEST_START,
            "end_time": TEST_END,
        }

        # 尝试运行回测
        try:
            from qlib.utils import init_instance_by_config
            portfolio_metrics = backtest(**backtest_config)

            print(f"✅ 回测完成")
            if isinstance(portfolio_metrics, dict):
                print(f"   年化收益: {portfolio_metrics.get('annual_return', 'N/A')}")
                print(f"   夏普比率: {portfolio_metrics.get('sharpe', 'N/A')}")
                print(f"   最大回撤: {portfolio_metrics.get('max_drawdown', 'N/A')}")

            return portfolio_metrics

        except Exception as e:
            print(f"⚠️ 官方回测失败: {e}")
            print("   使用简化回测...")

            # 简化回测：计算预测准确率
            return simple_backtest(predictions, dataset)

    except Exception as e:
        print(f"❌ 回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def simple_backtest(predictions: pd.Series, dataset):
    """
    简化回测：计算预测的 IC 和 Rank IC
    """
    print("  计算预测指标...")

    try:
        # 获取测试集标签
        df_test = dataset.prepare("test", col_set=["label"])

        if isinstance(df_test, dict):
            labels = df_test["label"]
        else:
            labels = df_test.xs("label", axis=1, level=0)

        # 对齐预测和标签
        common_index = predictions.index.intersection(labels.index)
        if len(common_index) == 0:
            print("  ⚠️ 预测和标签无共同索引")
            return None

        pred_aligned = predictions.loc[common_index]
        label_aligned = labels.loc[common_index]

        # 计算 IC (Information Coefficient)
        ic = pred_aligned.corr(label_aligned.iloc[:, 0])
        rank_ic = pred_aligned.rank().corr(label_aligned.iloc[:, 0].rank())

        print(f"  IC: {ic:.4f}")
        print(f"  Rank IC: {rank_ic:.4f}")

        return {
            'ic': ic,
            'rank_ic': rank_ic,
            'n_predictions': len(predictions),
        }

    except Exception as e:
        print(f"  ❌ 简化回测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


# ============================================================================
# 主程序
# ============================================================================

def run_benchmark(
    model_types: list = None,
    use_alpha158: bool = True,
    save_models: bool = True,
):
    """
    运行 Benchmark

    Args:
        model_types: 要测试的模型列表，默认测试所有模型
        use_alpha158: 是否使用 Alpha158 特征
        save_models: 是否保存模型
    """
    print("=" * 60)
    print("Qlib 官方 Benchmark 训练")
    print("=" * 60)
    print(f"数据路径: {QLIB_DATA_PATH}")
    print(f"训练期: {TRAIN_START} ~ {TRAIN_END}")
    print(f"验证期: {VALID_START} ~ {VALID_END}")
    print(f"测试期: {TEST_START} ~ {TEST_END}")

    # 初始化 Qlib
    init_qlib()

    # 创建数据集
    if use_alpha158:
        dataset = create_dataset()
    else:
        dataset = create_simple_dataset()

    # 测试模型
    if model_types is None:
        model_types = ['lgb', 'xgboost']  # 默认只测试 GBDT 模型

    results = {}
    model_dir = project_root / "qlib_data" / "cn_data"

    for model_type in model_types:
        config = MODEL_CONFIGS.get(model_type)
        if config is None:
            print(f"⚠️ 跳过未知模型: {model_type}")
            continue

        # 保存路径
        save_path = None
        if save_models:
            save_path = str(model_dir / f"model_{model_type}.pkl")

        try:
            # 训练模型
            model = train_model(model_type, dataset, save_path)

            # 预测和回测
            metrics = predict_and_backtest(model, dataset, config['name'])

            results[model_type] = {
                'name': config['name'],
                'metrics': metrics,
                'success': True,
            }

        except Exception as e:
            print(f"❌ {config['name']} 测试失败: {e}")
            results[model_type] = {
                'name': config['name'],
                'error': str(e),
                'success': False,
            }

    # 结果汇总
    print("\n" + "=" * 60)
    print("Benchmark 结果汇总")
    print("=" * 60)

    for model_type, result in results.items():
        status = "✅" if result['success'] else "❌"
        print(f"\n{status} {result['name']} ({model_type})")

        if result['success'] and result.get('metrics'):
            metrics = result['metrics']
            if isinstance(metrics, dict):
                for k, v in metrics.items():
                    if isinstance(v, float):
                        print(f"   {k}: {v:.4f}")
                    else:
                        print(f"   {k}: {v}")
        elif not result['success']:
            print(f"   错误: {result.get('error', 'Unknown')}")

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="Qlib 官方 Benchmark 训练")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=['lgb', 'xgboost'],
        choices=list(MODEL_CONFIGS.keys()),
        help="要测试的模型类型",
    )
    parser.add_argument(
        "--no-alpha158",
        action="store_true",
        help="不使用 Alpha158 特征",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="不保存模型",
    )

    args = parser.parse_args()

    results = run_benchmark(
        model_types=args.models,
        use_alpha158=not args.no_alpha158,
        save_models=not args.no_save,
    )

    return results


if __name__ == "__main__":
    results = main()
