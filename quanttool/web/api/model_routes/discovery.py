"""Qlib model discovery API routes."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ....core.logging import get_logger


logger = get_logger(__name__)
router = APIRouter()

@router.get("/qlib/models")
async def list_qlib_models() -> List[Dict[str, Any]]:
    """
    列出可用的 Qlib ML 模型

    支持 GBDT 系列模型：
    - LightGBM, XGBoost, CatBoost, DoubleEnsemble
    """
    try:
        from quanttool.strategies.qlib import list_available_models
        df = list_available_models()
        models = df.to_dict('records')

        # 添加参数说明
        model_params = {
            'gbdt': {
                'n_estimators': {'type': 'int', 'default': 200, 'description': '树的数量'},
                'max_depth': {'type': 'int', 'default': 6, 'description': '最大深度'},
                'learning_rate': {'type': 'float', 'default': 0.01, 'description': '学习率'},
            },
        }

        for model in models:
            category = model.get('category', 'unknown')
            if category in model_params:
                model['params'] = model_params[category]
            else:
                model['params'] = {}

        return models
    except ImportError:
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.get("/qlib/saved-models")
async def list_saved_models() -> List[Dict[str, Any]]:
    """
    列出已保存的模型文件

    返回模型文件列表，按修改时间降序排列。
    包含模型元数据（如特征数量、训练参数等）
    """
    import os
    import joblib
    from pathlib import Path

    model_dir = Path("models/qlib")
    if not model_dir.exists():
        return []

    models = []
    for model_file in model_dir.glob("*.pkl"):
        try:
            stat = model_file.stat()
            # 解析模型名称：{model_type}_{id}.pkl
            name_parts = model_file.stem.split('_')
            model_type = name_parts[0] if name_parts else "unknown"
            model_id = name_parts[1] if len(name_parts) > 1 else ""

            # 尝试加载模型元数据
            feature_count = None
            feature_set = None
            train_stocks = None

            try:
                saved_data = joblib.load(model_file)
                if isinstance(saved_data, dict):
                    model = saved_data.get('model')
                    feature_names = saved_data.get('feature_names', [])
                    feature_count = len(feature_names) if feature_names else None
                    # 尝试获取更多信息
                    if hasattr(model, 'feature_names_'):
                        feature_count = len(model.feature_names_)
            except Exception:
                pass  # 无法加载元数据，使用默认值

            # 模型类型显示名称
            display_names = {
                'lgb': 'LightGBM',
                'lightgbm': 'LightGBM',
                'xgboost': 'XGBoost',
                'xgb': 'XGBoost',
                'catboost': 'CatBoost',
                'double_ensemble': 'DoubleEnsemble',
            }

            models.append({
                "id": model_id,
                "name": model_file.name,
                "path": str(model_file),
                "model_type": model_type,
                "display_name": display_names.get(model_type, model_type.upper()),
                "feature_count": feature_count,
                "size_kb": round(stat.st_size / 1024, 2),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "created_date": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d"),
                "source": "trained",
            })
        except Exception as e:
            logger.warning(f"Failed to read model file {model_file}: {e}")

    # 按修改时间降序排列
    models.sort(key=lambda x: x["modified_at"], reverse=True)
    return models


@router.get("/qlib/pretrained-models")
async def list_pretrained_models() -> List[Dict[str, Any]]:
    """
    列出本地预训练模型

    扫描 qlib_data/cn_data/ 目录下的预训练模型文件 (model_*.pkl)
    返回模型列表，包含模型类型、大小、修改时间等信息
    """
    import joblib
    from pathlib import Path

    models = []

    # 预训练模型目录
    pretrained_dirs = [
        Path("qlib_data/cn_data"),
        Path("models/qlib"),
    ]

    # 模型类型显示名称
    display_names = {
        'lgb': 'LightGBM',
        'lightgbm': 'LightGBM',
        'xgboost': 'XGBoost',
        'xgb': 'XGBoost',
        'catboost': 'CatBoost',
        'lstm': 'LSTM',
        'gru': 'GRU',
        'transformer': 'Transformer',
        'mlp': 'MLP',
        'gbdt': 'GBDT',
    }

    for model_dir in pretrained_dirs:
        if not model_dir.exists():
            continue

        for model_file in model_dir.glob("model_*.pkl"):
            try:
                stat = model_file.stat()

                # 解析模型类型: model_lgb.pkl -> lgb
                model_type = model_file.stem.replace("model_", "").lower()

                # 尝试加载模型元数据
                feature_count = None
                ic_score = None
                train_info = {}

                try:
                    saved_data = joblib.load(model_file)
                    if isinstance(saved_data, dict):
                        model = saved_data.get('model')
                        feature_names = saved_data.get('feature_names', [])
                        feature_count = len(feature_names) if feature_names else None

                        # 尝试获取训练信息
                        if 'config' in saved_data:
                            train_info['config'] = saved_data['config']
                        if 'metrics' in saved_data:
                            train_info['metrics'] = saved_data['metrics']
                            ic_score = saved_data['metrics'].get('ic')

                        # 从模型对象获取特征数量
                        if hasattr(model, 'feature_names_') and feature_count is None:
                            feature_count = len(model.feature_names_)
                except Exception as e:
                    logger.debug(f"Could not load model metadata: {e}")

                models.append({
                    "name": model_file.name,
                    "path": str(model_file),
                    "model_type": model_type,
                    "display_name": display_names.get(model_type, model_type.upper()),
                    "feature_count": feature_count,
                    "ic_score": ic_score,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created_date": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d"),
                    "source": "pretrained" if "qlib_data" in str(model_dir) else "trained",
                    "train_info": train_info,
                })
            except Exception as e:
                logger.warning(f"Failed to read pretrained model {model_file}: {e}")

    # 按修改时间降序排列
    models.sort(key=lambda x: x["modified_at"], reverse=True)
    return models


@router.get("/qlib/all-models")
async def list_all_models() -> Dict[str, Any]:
    """
    列出所有模型（预训练 + 训练后的模型）

    返回分组展示的模型列表
    """
    import joblib
    from pathlib import Path

    # 模型类型显示名称
    display_names = {
        'lgb': 'LightGBM',
        'lightgbm': 'LightGBM',
        'xgboost': 'XGBoost',
        'xgb': 'XGBoost',
        'catboost': 'CatBoost',
        'lstm': 'LSTM',
        'gru': 'GRU',
        'transformer': 'Transformer',
        'mlp': 'MLP',
        'gbdt': 'GBDT',
    }

    def scan_model_dir(model_dir: Path, source: str) -> List[Dict[str, Any]]:
        """扫描模型目录"""
        models = []
        if not model_dir.exists():
            return models

        for model_file in model_dir.glob("*.pkl"):
            try:
                stat = model_file.stat()

                # 解析模型类型
                stem = model_file.stem
                if stem.startswith("model_"):
                    model_type = stem.replace("model_", "").lower()
                else:
                    # 格式: {model_type}_{id}
                    name_parts = stem.split('_')
                    model_type = name_parts[0] if name_parts else "unknown"

                # 尝试加载模型元数据
                feature_count = None
                ic_score = None

                try:
                    saved_data = joblib.load(model_file)
                    if isinstance(saved_data, dict):
                        feature_names = saved_data.get('feature_names', [])
                        feature_count = len(feature_names) if feature_names else None

                        if 'metrics' in saved_data:
                            ic_score = saved_data['metrics'].get('ic')

                        model = saved_data.get('model')
                        if hasattr(model, 'feature_names_') and feature_count is None:
                            feature_count = len(model.feature_names_)
                except Exception:
                    pass

                models.append({
                    "name": model_file.name,
                    "path": str(model_file),
                    "model_type": model_type,
                    "display_name": display_names.get(model_type, model_type.upper()),
                    "feature_count": feature_count,
                    "ic_score": ic_score,
                    "size_kb": round(stat.st_size / 1024, 2),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "created_date": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d"),
                    "source": source,
                })
            except Exception as e:
                logger.warning(f"Failed to read model file {model_file}: {e}")

        return models

    # 扫描预训练模型目录
    pretrained_models = scan_model_dir(Path("qlib_data/cn_data"), "pretrained")

    # 扫描训练后的模型目录
    trained_models = scan_model_dir(Path("models/qlib"), "trained")

    return {
        "pretrained": sorted(pretrained_models, key=lambda x: x["modified_at"], reverse=True),
        "trained": sorted(trained_models, key=lambda x: x["modified_at"], reverse=True),
        "total_count": len(pretrained_models) + len(trained_models),
    }


@router.get("/qlib/saved-models/{model_id}")
async def get_saved_model_detail(model_id: str) -> Dict[str, Any]:
    """
    获取已保存模型的详细信息

    Args:
        model_id: 模型ID或模型文件名
    """
    import joblib
    from pathlib import Path

    model_dir = Path("models/qlib")
    if not model_dir.exists():
        raise HTTPException(status_code=404, detail="模型目录不存在")

    # 查找模型文件
    model_file = None
    for f in model_dir.glob("*.pkl"):
        if model_id in f.name:
            model_file = f
            break

    if model_file is None:
        raise HTTPException(status_code=404, detail=f"未找到模型: {model_id}")

    try:
        saved_data = joblib.load(model_file)

        model = saved_data.get('model')
        feature_names = saved_data.get('feature_names', [])

        # 获取模型详情
        detail = {
            "path": str(model_file),
            "name": model_file.name,
            "feature_count": len(feature_names),
            "feature_names": feature_names[:20] if feature_names else [],  # 前20个特征
            "has_model": model is not None,
        }

        # 获取模型参数
        if hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                # 过滤掉复杂的参数
                simple_params = {}
                for k, v in params.items():
                    if isinstance(v, (str, int, float, bool, type(None))):
                        simple_params[k] = v
                detail["params"] = simple_params
            except Exception:
                pass

        # 文件信息
        stat = model_file.stat()
        detail["size_kb"] = round(stat.st_size / 1024, 2)
        detail["modified_at"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

        return detail

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")


@router.get("/qlib/models/categories")
async def get_qlib_model_categories() -> List[Dict[str, Any]]:
    """获取 Qlib 模型分类"""
    return [
        {
            "category": "gbdt",
            "display_name": "GBDT 系列",
            "description": "梯度提升决策树，适合表格数据，训练快",
            "models": ["lgb", "lightgbm", "xgboost", "xgb", "catboost", "double_ensemble"],
            "recommended": "lgb"
        }
    ]
