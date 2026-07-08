"""GBM and Qlib model API routes."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
import os
import queue
import threading
import time
import uuid

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ...core.logging import get_logger
from .utils import get_cached_analysis, set_cached_analysis, to_python_types


logger = get_logger(__name__)
router = APIRouter()

from pathlib import Path

from ..schemas.model import (
    GBMPicksRequest,
    GBMPredictRequest,
    GBMTrainRequest,
    QlibPredictRequest,
    QlibTrainRequest,
)


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


# 预制的优化参数（基于超参数搜索结果）
GBM_OPTIMAL_PARAMS = {
    "feature_type": "alpha158",
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.2,
    "num_leaves": 210,
    "subsample": 0.8789,
    "colsample_bytree": 0.8879,
    "reg_alpha": 205.6999,  # lambda_l1
    "reg_lambda": 580.9768,  # lambda_l2
    "n_jobs": 20,
    "label_horizon": 10,
    "buy_threshold": 0.50,  # 降低买入阈值，增加交易机会
    "sell_threshold": 0.50,  # 卖出阈值
}


@router.post("/gbm/train")
async def train_gbm_model(request: GBMTrainRequest) -> Dict[str, Any]:
    """
    训练 GBM 策略

    使用 LightGBM (sklearn 接口) 和 Alpha158 特征
    使用预制的优化参数，忽略用户输入

    数据划分（按年份固定）:
    - 训练集: 2017-01-01 ~ 2022-12-31
    - 验证集: 2023-01-01 ~ 2024-06-30
    - 测试集: 2024-07-01 ~ 当前
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        from quanttool.cli.commands.analysis_commands import get_csi300_constituents
        import uuid

        # 获取沪深300成分股作为训练数据
        csi300_stocks = get_csi300_constituents()
        train_symbols = [s['code'] if isinstance(s, dict) else s for s in csi300_stocks]

        # 限制训练股票数量
        if request.max_train_stocks > 0:
            train_symbols = train_symbols[:request.max_train_stocks]

        logger.info(f"GBM 训练: 使用 {len(train_symbols)} 只沪深300成分股")

        # 使用固定优化参数（忽略用户输入）
        config = GBMConfig(**GBM_OPTIMAL_PARAMS)

        # 创建策略
        strategy = GBMStrategy(config)

        # 训练模型
        result = strategy.train(
            instruments=train_symbols,
            start_date="2017-01-01",
            end_date="2026-12-31",
        )

        # 保存模型
        model_dir = "models/gbm"
        os.makedirs(model_dir, exist_ok=True)
        model_id = str(uuid.uuid4())[:8]
        model_path = f"{model_dir}/lgbm_{model_id}.pkl"
        strategy.save_model(model_path)

        return {
            "success": True,
            "model_id": model_id,
            "model_path": model_path,
            "train_samples": to_python_types(result.get("train_samples", 0)),
            "valid_samples": to_python_types(result.get("valid_samples", 0)),
            "test_samples": to_python_types(result.get("test_samples", 0)),
            "feature_count": to_python_types(result.get("feature_count", 0)),
            "train_ic": to_python_types(result.get("train_ic", 0)),
            "valid_ic": to_python_types(result.get("valid_ic", 0)),
            "test_ic": to_python_types(result.get("test_ic", 0)),
            "best_iteration": to_python_types(result.get("best_iteration", 0)),
        }

    except Exception as e:
        logger.error(f"GBM 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")


@router.post("/gbm/predict")
async def predict_gbm_model(request: GBMPredictRequest) -> Dict[str, Any]:
    """
    使用 GBM 策略预测

    返回每只股票的预测收益率和交易信号
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        import glob

        # 查找模型
        model_path = request.model_path
        if not model_path:
            model_files = glob.glob("models/gbm/lgbm_*.pkl")
            if not model_files:
                raise HTTPException(status_code=404, detail="未找到训练好的模型")
            model_path = max(model_files, key=os.path.getmtime)
            logger.info(f"使用最新模型: {model_path}")

        # 加载模型
        config = GBMConfig()
        strategy = GBMStrategy(config)
        strategy.load_model(model_path)

        # 预测
        predictions = []
        for symbol in request.symbols:
            try:
                pred = strategy.predict(symbol)
                predictions.append(to_python_types(pred))
            except Exception as e:
                logger.warning(f"预测失败 [{symbol}]: {e}")
                predictions.append({
                    "instrument": symbol,
                    "error": str(e),
                })

        return {
            "success": True,
            "model_path": model_path,
            "predictions": predictions,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"GBM 预测失败: {e}")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.get("/gbm/models")
async def list_gbm_models() -> List[Dict[str, Any]]:
    """列出所有 GBM 模型"""
    import glob

    model_files = glob.glob("models/gbm/lgbm_*.pkl")

    result = []
    for path in model_files:
        stat = os.stat(path)
        result.append({
            "path": path,
            "filename": os.path.basename(path),
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
        })

    return sorted(result, key=lambda x: x["modified"], reverse=True)


@router.delete("/gbm/models/{model_id}")
async def delete_gbm_model(model_id: str) -> Dict[str, Any]:
    """删除指定的 GBM 模型"""
    import glob

    # 查找匹配的模型文件
    model_files = glob.glob(f"models/gbm/*{model_id}*.pkl")

    if not model_files:
        # 也可能是 qrun 模型
        import shutil
        qrun_dirs = glob.glob(f"mlruns/0/*{model_id}*")
        if qrun_dirs:
            for dir_path in qrun_dirs:
                shutil.rmtree(dir_path)
            return {"success": True, "message": f"已删除模型目录: {model_id}"}
        raise HTTPException(status_code=404, detail=f"模型 {model_id} 不存在")

    deleted = []
    for path in model_files:
        try:
            os.remove(path)
            deleted.append(path)
        except Exception as e:
            logger.warning(f"删除模型文件失败 {path}: {e}")

    return {"success": True, "deleted": deleted}


@router.get("/gbm/train/{task_id}/progress")
async def get_training_progress(task_id: str) -> Dict[str, Any]:
    """获取训练任务进度"""
    # 检查任务状态（从全局任务存储）
    if task_id in _training_tasks:
        task_info = _training_tasks[task_id]
        return {
            "status": task_info.get("status", "unknown"),
            "progress": task_info.get("progress", 0),
            "message": task_info.get("message", ""),
        }

    # 检查 mlruns 是否有对应的结果
    import glob
    result_dirs = glob.glob(f"mlruns/0/{task_id}")
    if result_dirs:
        return {
            "status": "completed",
            "progress": 100,
            "message": "训练已完成",
        }

    return {
        "status": "not_found",
        "progress": 0,
        "message": f"任务 {task_id} 不存在",
    }


# 训练任务存储
_training_tasks: Dict[str, Dict[str, Any]] = {}


@router.get("/gbm/qrun-models")
async def list_qrun_models() -> List[Dict[str, Any]]:
    """
    列出所有 qrun 训练的模型

    返回 mlruns 目录中所有可用的模型信息
    """
    try:
        from quanttool.application.gbm_picker_service import list_all_qrun_models

        models = list_all_qrun_models()

        # 移除不需要返回的字段
        for model in models:
            model.pop('modified_timestamp', None)

        return models

    except Exception as e:
        logger.error(f"获取 qrun 模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")


@router.post("/gbm/picks")
async def get_gbm_picks(request: GBMPicksRequest) -> Dict[str, Any]:
    """
    GBM 模型智能荐股

    使用已训练的 GBM 模型对沪深300成分股进行预测，返回 top N 推荐股票。
    如果没有可用模型，会自动训练一个新模型。

    支持使用 qrun 训练的模型，通过 model_path 参数指定。
    """
    try:
        from quanttool.application.gbm_picker_service import GBMCsi300Picker

        # 创建荐股器
        picker = GBMCsi300Picker(
            top_n=request.top_n,
            model_path=request.model_path
        )

        # 获取推荐
        result = picker.get_daily_picks(force_train=request.force_train)

        # 转换为响应格式 (确保所有数值都是 Python 原生类型)
        top_stocks = []
        for rec in result.top_stocks:
            top_stocks.append({
                "code": rec.code,
                "name": rec.name,
                "pred_return": float(round(rec.pred_return, 4)) if rec.pred_return else 0.0,
                "percentile": float(round(rec.percentile, 4)) if rec.percentile else 0.0,
                "confidence": float(round(rec.confidence, 4)) if rec.confidence else 0.0,
                "probability": float(round(rec.probability, 4)) if rec.probability else 0.0,
                "signal": rec.signal,
                "close": float(round(rec.close, 2)) if rec.close else None,
                "stop_loss": float(round(rec.stop_loss, 2)) if rec.stop_loss else None,
                "take_profit": float(round(rec.take_profit, 2)) if rec.take_profit else None,
            })

        return {
            "success": True,
            "date": result.date,
            "total_stocks": int(result.total_stocks),
            "valid_stocks": int(result.valid_stocks),
            "top_stocks": top_stocks,
            "model_info": result.model_info,
        }

    except Exception as e:
        logger.error(f"GBM 荐股失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"荐股失败: {str(e)}")


@router.post("/qlib/train")
async def train_qlib_model(request: QlibTrainRequest) -> Dict[str, Any]:
    """
    训练 Qlib ML 模型

    使用沪深300成分股作为训练数据，按年份划分训练/验证/测试集：
    - 训练集: 2020-2023年
    - 验证集: 2024-2025年
    - 测试集: 2026年

    用户输入的股票代码仅用于预测，不参与训练
    """
    try:
        from quanttool.strategies.qlib import create_model
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.cli.commands.analysis_commands import get_csi300_constituents
        import numpy as np
        import os
        import uuid

        # 获取沪深300成分股作为训练数据
        csi300_stocks = get_csi300_constituents()
        train_symbols = [s['code'] if isinstance(s, dict) else s for s in csi300_stocks]

        # 限制训练股票数量
        if request.max_train_stocks > 0:
            train_symbols = train_symbols[:request.max_train_stocks]

        logger.info(f"Training with {len(train_symbols)} CSI300 stocks")

        # 获取训练数据
        analyzer = StockAnalyzer()
        train_data = []
        valid_data = []
        test_data = []

        # 解析日期
        train_start_dt = datetime.fromisoformat(request.train_start)
        train_end_dt = datetime.fromisoformat(request.train_end)
        valid_start_dt = datetime.fromisoformat(request.valid_start)
        valid_end_dt = datetime.fromisoformat(request.valid_end)
        test_start_dt = datetime.fromisoformat(request.test_start)
        test_end_dt = datetime.fromisoformat(request.test_end)

        logger.info(f"Date ranges - Train: {request.train_start} to {request.train_end}, "
                   f"Valid: {request.valid_start} to {request.valid_end}, "
                   f"Test: {request.test_start} to {request.test_end}")

        success_count = 0
        first_symbol_features = None  # 记录第一个成功股票的特征列名，确保所有股票使用相同特征

        # 初始化 Alpha158 特征工程器
        from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
        feature_engineer = QlibFeatureEngineer(feature_set=request.feature_set)

        for symbol in train_symbols:
            try:
                # 获取足够的历史数据（约7年，覆盖2019-2026训练+验证+预测期间）
                df = analyzer.get_stock_data(symbol, 2500)
                if df.empty or len(df) < 120:  # Alpha158 需要至少 120 条数据
                    logger.warning(f"Insufficient data for {symbol}: {len(df) if not df.empty else 0} rows")
                    continue

                # 确定日期列
                date_column = None
                if 'trade_date' in df.columns:
                    date_column = 'trade_date'
                elif 'timestamp' in df.columns:
                    date_column = 'timestamp'

                if not date_column:
                    logger.warning(f"No date column found for {symbol}")
                    continue

                df['_date'] = pd.to_datetime(df[date_column])

                if request.use_rich_features:
                    # 使用 Alpha158 特征工程 (150+ 特征)
                    try:
                        feature_df = feature_engineer.generate_features(df)
                        available_features = list(feature_df.columns)
                        df = pd.concat([df, feature_df], axis=1)
                    except Exception as e:
                        logger.warning(f"Feature engineering failed for {symbol}: {e}")
                        continue
                else:
                    # 计算技术指标
                    df = analyzer.calculate_technical_indicators(df)

                    if request.features:
                        # 使用用户指定的特征
                        available_features = [f for f in request.features if f in df.columns]
                    else:
                        # 使用基本特征
                        available_features = ['close', 'volume', 'ma_5', 'ma_10', 'ma_20']

                if not available_features:
                    logger.warning(f"No available features for {symbol}")
                    continue

                # 确保所有股票使用相同的特征列
                if first_symbol_features is None:
                    first_symbol_features = available_features
                else:
                    # 使用第一个股票的特征列，确保一致性
                    available_features = [f for f in first_symbol_features if f in df.columns]
                    if len(available_features) != len(first_symbol_features):
                        logger.warning(f"Feature mismatch for {symbol}, expected {len(first_symbol_features)}, got {len(available_features)}")
                        continue

                logger.info(f"Using {len(available_features)} features for {symbol}")

                # 计算标签（未来5日收益率）
                df['return_5d'] = df['close'].pct_change(5).shift(-5)

                # 调试：输出数据的日期范围
                data_min_date = df['_date'].min()
                data_max_date = df['_date'].max()
                logger.info(f"{symbol}: data range {data_min_date} to {data_max_date}, {len(df)} rows")

                # 按日期划分数据
                row_count = 0
                for idx, row in df.iterrows():
                    date_val = row['_date']
                    if pd.isna(date_val):
                        continue

                    feature_vals = [row.get(f) for f in available_features]
                    label_val = row.get('return_5d')

                    # 过滤无效值
                    if any(pd.isna(v) for v in feature_vals) or pd.isna(label_val):
                        continue

                    row_data = {
                        'features': feature_vals,
                        'label': label_val,
                        'symbol': symbol,
                        'date': date_val
                    }

                    # 划分数据集
                    if train_start_dt <= date_val <= train_end_dt:
                        train_data.append(row_data)
                        row_count += 1
                    elif valid_start_dt <= date_val <= valid_end_dt:
                        valid_data.append(row_data)
                        row_count += 1
                    elif test_start_dt <= date_val <= test_end_dt:
                        test_data.append(row_data)
                        row_count += 1

                if row_count > 0:
                    success_count += 1

            except Exception as e:
                logger.warning(f"Failed to get data for {symbol}: {e}")
                continue

        logger.info(f"Data collection complete: {success_count} stocks succeeded, "
                   f"train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")

        if not train_data:
            raise HTTPException(
                status_code=400,
                detail=f"无法获取足够的训练数据。收集了 {success_count} 只股票，训练集 {len(train_data)} 条，"
                       f"验证集 {len(valid_data)} 条，测试集 {len(test_data)} 条。请检查日期范围是否在数据覆盖范围内。"
            )

        # 准备训练数据
        feature_cols = available_features
        X_train = np.array([d['features'] for d in train_data])
        y_train = np.array([d['label'] for d in train_data])

        # 创建模型
        config_kwargs = {
            'n_estimators': request.n_estimators,
            'max_depth': request.max_depth,
            'learning_rate': request.learning_rate,
            'hidden_size': request.hidden_size,
            'num_layers': request.num_layers,
            'dropout': request.dropout,
            'epochs': request.epochs,
            'batch_size': request.batch_size,
        }

        model = create_model(request.model_type, **config_kwargs)

        # 训练
        X_train_df = pd.DataFrame(X_train, columns=feature_cols)
        y_train_series = pd.Series(y_train)
        model.fit(X_train_df, y_train_series)
        model.feature_names_ = feature_cols

        # 保存模型
        model_dir = "models/qlib"
        os.makedirs(model_dir, exist_ok=True)
        model_id = str(uuid.uuid4())[:8]
        model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"
        model.save(model_path)

        # 评估训练集
        train_pred = model.predict(X_train_df)
        train_mse = np.mean((train_pred - y_train) ** 2)
        train_mae = np.mean(np.abs(train_pred - y_train))
        train_ic = np.corrcoef(train_pred, y_train)[0, 1] if len(train_pred) > 1 else 0

        # 评估验证集
        valid_metrics = {}
        if valid_data:
            X_valid = np.array([d['features'] for d in valid_data])
            y_valid = np.array([d['label'] for d in valid_data])
            X_valid_df = pd.DataFrame(X_valid, columns=feature_cols)
            valid_pred = model.predict(X_valid_df)
            valid_metrics = {
                "samples": len(valid_data),
                "mse": round(float(np.mean((valid_pred - y_valid) ** 2)), 6),
                "mae": round(float(np.mean(np.abs(valid_pred - y_valid))), 6),
                "ic": round(float(np.corrcoef(valid_pred, y_valid)[0, 1]), 4) if len(valid_pred) > 1 else 0,
            }

        # 评估测试集
        test_metrics = {}
        if test_data:
            X_test = np.array([d['features'] for d in test_data])
            y_test = np.array([d['label'] for d in test_data])
            X_test_df = pd.DataFrame(X_test, columns=feature_cols)
            test_pred = model.predict(X_test_df)
            test_metrics = {
                "samples": len(test_data),
                "mse": round(float(np.mean((test_pred - y_test) ** 2)), 6),
                "mae": round(float(np.mean(np.abs(test_pred - y_test))), 6),
                "ic": round(float(np.corrcoef(test_pred, y_test)[0, 1]), 4) if len(test_pred) > 1 else 0,
            }

        return {
            "model_id": model_id,
            "model_type": request.model_type,
            "model_path": model_path,
            "train_symbols_count": len(train_symbols),
            "predict_symbols": request.symbols,  # 用户输入的股票代码（仅用于预测）
            "train_samples": len(train_data),
            "features": feature_cols,
            "feature_count": len(feature_cols),
            "use_rich_features": request.use_rich_features,
            "data_split": {
                "train": {
                    "period": f"{request.train_start} ~ {request.train_end}",
                    "samples": len(train_data),
                },
                "valid": {
                    "period": f"{request.valid_start} ~ {request.valid_end}",
                    "samples": len(valid_data),
                },
                "test": {
                    "period": f"{request.test_start} ~ {request.test_end}",
                    "samples": len(test_data),
                },
            },
            "metrics": {
                "train": {
                    "mse": round(float(train_mse), 6),
                    "mae": round(float(train_mae), 6),
                    "ic": round(float(train_ic), 4) if not np.isnan(train_ic) else 0,
                },
                "valid": valid_metrics,
                "test": test_metrics,
            },
            "backtest_params": {
                "initial_cash": request.initial_cash,
                "commission_rate": request.commission_rate,
                "slippage_rate": request.slippage_rate,
                "t_plus_1": True,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")


@router.post("/qlib/train/stream")
async def train_qlib_model_stream(request: QlibTrainRequest):
    """
    使用 SSE 流式推送训练进度

    事件类型:
    - progress: 进度更新
    - log: 日志消息
    - complete: 训练完成
    - error: 错误
    """
    import asyncio

    # 使用同步队列（线程安全）
    message_queue = queue.Queue()

    def send_event(event_type: str, data: Dict[str, Any]):
        """发送SSE事件到队列"""
        message_queue.put({"event": event_type, "data": data})

    def training_worker():
        """后台训练线程"""
        try:
            from quanttool.strategies.qlib import create_model
            from quanttool.factors.stock_analyzer import StockAnalyzer
            from quanttool.cli.commands.analysis_commands import get_csi300_constituents
            import numpy as np
            import os
            import uuid

            # 阶段1: 初始化
            send_event("progress", {
                "stage": "init",
                "progress": 0,
                "message": "初始化训练环境..."
            })

            # 获取沪深300成分股
            csi300_stocks = get_csi300_constituents()
            train_symbols = [s['code'] if isinstance(s, dict) else s for s in csi300_stocks]

            if request.max_train_stocks > 0:
                train_symbols = train_symbols[:request.max_train_stocks]

            total_stocks = len(train_symbols)
            send_event("progress", {
                "stage": "init",
                "progress": 5,
                "message": f"准备获取 {total_stocks} 只沪深300成分股数据"
            })

            # 阶段2: 数据获取
            analyzer = StockAnalyzer()
            train_data = []
            valid_data = []
            test_data = []

            train_start_dt = datetime.fromisoformat(request.train_start)
            train_end_dt = datetime.fromisoformat(request.train_end)
            valid_start_dt = datetime.fromisoformat(request.valid_start)
            valid_end_dt = datetime.fromisoformat(request.valid_end)
            test_start_dt = datetime.fromisoformat(request.test_start)
            test_end_dt = datetime.fromisoformat(request.test_end)

            success_count = 0
            cache_hits = 0
            first_symbol_features = None  # 记录第一个成功股票的特征列名

            # 初始化 Alpha158 特征工程器
            from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
            feature_engineer = QlibFeatureEngineer(feature_set=request.feature_set)

            # 计算实际需要的日期范围
            train_end_date = datetime.fromisoformat(request.train_end)
            start_date = train_end_date - timedelta(days=2500)  # 约 7 年

            # 先并发预加载所有股票数据（显著提升性能）
            send_event("progress", {
                "stage": "data_preload",
                "progress": 5,
                "message": f"并发预加载 {total_stocks} 只股票数据..."
            })

            loaded_count = analyzer.preload_data_for_scan(train_symbols, days=2500)
            send_event("progress", {
                "stage": "data_preload",
                "progress": 10,
                "message": f"预加载完成，成功获取 {loaded_count} 只股票数据"
            })

            # 使用 qlib 原生训练流程
            send_event("progress", {
                "stage": "qlib_setup",
                "progress": 15,
                "message": "初始化 Qlib 训练环境..."
            })

            try:
                from quanttool.infrastructure.data_providers.qlib_data_converter import (
                    QlibDataConverter,
                    QlibTrainingPipeline,
                    QlibDataConfig
                )

                # 配置 Qlib 数据转换器
                qlib_config = QlibDataConfig(
                    cache_dir=".cache/incremental_data",
                    output_dir="qlib_data/cn_data",
                    feature_type="alpha158" if request.use_rich_features else "alpha360",
                    start_date=request.train_start,
                    end_date=request.train_end,
                )

                converter = QlibDataConverter(qlib_config)
                pipeline = QlibTrainingPipeline(converter)

                send_event("log", {"message": f"使用 Qlib 原生训练流程 (特征: {qlib_config.feature_type})"})

                # 阶段3: 转换数据为 qlib 格式
                send_event("progress", {
                    "stage": "data_conversion",
                    "progress": 20,
                    "message": "转换数据为 Qlib 原生格式..."
                })

                # 转换股票代码格式 (000001.SZ -> 000001_SZ)
                qlib_symbols = [s.replace('.', '_') for s in train_symbols]

                # 创建 qlib DatasetH
                dataset = converter.create_qlib_dataset(
                    symbols=qlib_symbols,
                    start_date=request.train_start,
                    end_date=request.test_end,
                    feature_type=qlib_config.feature_type,
                    label_type="return_10"
                )

                send_event("log", {"message": f"Qlib DatasetH 创建成功"})

            except Exception as e:
                send_event("log", {"message": f"Qlib 原生流程失败，回退到 sklearn: {e}"})
                import traceback
                traceback.print_exc()

                # 回退到传统流程
                for i, symbol in enumerate(train_symbols):
                    try:
                        send_event("progress", {
                            "stage": "data_collection",
                            "progress": 10 + int((i / total_stocks) * 50),
                            "current": symbol,
                            "processed": i + 1,
                            "total": total_stocks,
                            "cache_hits": cache_hits,
                            "message": f"正在处理数据: {symbol} ({i + 1}/{total_stocks})"
                        })

                        df = analyzer.get_stock_data(
                            symbol,
                            start_date=start_date,
                            end_date=datetime.now(),
                            force_refresh=False
                        )

                        if len(df) >= 500:
                            cache_hits += 1

                        if df.empty or len(df) < 120:
                            continue

                        date_column = None
                        if 'trade_date' in df.columns:
                            date_column = 'trade_date'
                        elif 'timestamp' in df.columns:
                            date_column = 'timestamp'

                        if not date_column:
                            continue

                        df['_date'] = pd.to_datetime(df[date_column])

                        if request.use_rich_features:
                            try:
                                feature_df = feature_engineer.generate_features(df)
                                available_features = list(feature_df.columns)
                                df = pd.concat([df, feature_df], axis=1)
                            except Exception as e:
                                continue
                        else:
                            df = analyzer.calculate_technical_indicators(df)
                            if request.features:
                                available_features = [f for f in request.features if f in df.columns]
                            else:
                                available_features = ['close', 'volume', 'ma_5', 'ma_10', 'ma_20']

                        if not available_features:
                            continue

                        if first_symbol_features is None:
                            first_symbol_features = available_features
                        else:
                            available_features = [f for f in first_symbol_features if f in df.columns]
                            if len(available_features) != len(first_symbol_features):
                                continue

                        df['return_5d'] = df['close'].pct_change(5).shift(-5)
                        df['label'] = df['return_5d']

                        for idx, row in df.iterrows():
                            date_val = row['_date']
                            if pd.isna(date_val):
                                continue

                            feature_vals = [row.get(f) for f in available_features]
                            label_val = row.get('label')

                            if any(pd.isna(v) for v in feature_vals) or pd.isna(label_val):
                                continue

                            row_data = {
                                'features': feature_vals,
                                'label': label_val,
                                'symbol': symbol,
                                'date': date_val,
                            }

                            if train_start_dt <= date_val <= train_end_dt:
                                train_data.append(row_data)
                            elif valid_start_dt <= date_val <= valid_end_dt:
                                valid_data.append(row_data)
                            elif test_start_dt <= date_val <= test_end_dt:
                                test_data.append(row_data)

                        success_count += 1

                    except Exception as e:
                        continue

                # 传统流程：准备数据
                if train_data:
                    feature_cols = available_features
                    X_train = np.array([d['features'] for d in train_data])
                    y_train = np.array([d['label'] for d in train_data])
                    dataset = None  # 标记使用传统流程

            # 阶段4: 模型训练
            n_estimators = request.n_estimators or 100

            if dataset is not None:
                # 使用 Qlib 原生训练流程
                send_event("progress", {
                    "stage": "training",
                    "progress": 75,
                    "message": f"使用 Qlib 原生 {request.model_type.upper()} 模型训练..."
                })

                try:
                    # 初始化 Qlib
                    import qlib
                    if not hasattr(qlib, '_initialized') or not qlib._initialized:
                        qlib.init(provider_uri="qlib_data/cn_data")
                        qlib._initialized = True

                    # Qlib 内置模型映射表: (模块名, 类名, 模型类型)
                    QLIB_MODELS = {
                        # GBDT 系列
                        'lgb': ('gbdt', 'LGBModel', 'gbdt'),
                        'lightgbm': ('gbdt', 'LGBModel', 'gbdt'),
                        'xgboost': ('xgboost', 'XGBModel', 'gbdt'),
                        'xgb': ('xgboost', 'XGBModel', 'gbdt'),
                        'catboost': ('catboost_model', 'CatBoostModel', 'gbdt'),
                        'double_ensemble': ('double_ensemble', 'DEnsembleModel', 'gbdt'),
                    }

                    model_type_lower = request.model_type.lower()

                    if model_type_lower not in QLIB_MODELS:
                        supported = ', '.join(sorted(QLIB_MODELS.keys()))
                        raise ValueError(f"不支持的模型类型: {request.model_type}。支持: {supported}")

                    module_name, class_name, model_category = QLIB_MODELS[model_type_lower]
                    ModelClass = getattr(__import__(f'qlib.contrib.model.{module_name}', fromlist=[class_name]), class_name)

                    # 创建 GBDT 模型
                    model = ModelClass(
                        loss='mse',
                        n_estimators=n_estimators,
                        max_depth=request.max_depth or 6,
                        learning_rate=request.learning_rate or 0.01,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42,
                        n_jobs=-1,
                    )

                    send_event("log", {"message": f"创建 Qlib 原生 {request.model_type.upper()} 模型成功"})

                    import time
                    train_start_time = time.time()

                    train_msg = f"Qlib {request.model_type.upper()} 训练中 ({n_estimators} 棵树)..."

                    send_event("progress", {
                        "stage": "training",
                        "progress": 80,
                        "message": train_msg
                    })

                    model.fit(dataset)

                    train_elapsed = time.time() - train_start_time
                    send_event("log", {"message": f"Qlib 原生训练完成 (耗时 {train_elapsed:.1f}s)"})

                    # 获取特征数
                    feature_cols = []
                    try:
                        df_sample = dataset.prepare("train", col_set=["feature"])
                        if isinstance(df_sample, dict):
                            feature_cols = list(df_sample["feature"].columns)
                        else:
                            feature_cols = list(df_sample.xs('feature', axis=1, level=0).columns)
                    except Exception:
                        feature_cols = ["alpha158_features"]

                    # 保存模型到指定目录
                    model_dir = "models/qlib"
                    os.makedirs(model_dir, exist_ok=True)
                    model_id = str(uuid.uuid4())[:8]
                    model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"

                    # 使用 Qlib 的 to_pickle 方法保存模型
                    model.to_pickle(model_path)
                    send_event("log", {"message": f"模型已保存: {model_path}"})

                except Exception as e:
                    send_event("error", {"message": f"Qlib 原生训练失败: {str(e)}"})
                    import traceback
                    traceback.print_exc()
                    return

            else:
                # 传统 sklearn 流程
                config_kwargs = {
                    'n_estimators': request.n_estimators,
                    'max_depth': request.max_depth,
                    'learning_rate': request.learning_rate,
                    'hidden_size': request.hidden_size,
                    'num_layers': request.num_layers,
                    'dropout': request.dropout,
                    'epochs': request.epochs,
                    'batch_size': request.batch_size,
                }

                model = create_model(request.model_type, **config_kwargs)

                X_train_df = pd.DataFrame(X_train, columns=feature_cols)
                y_train_series = pd.Series(y_train)

                # 检查数据有效性
                send_event("log", {"message": f"数据形状: X={X_train_df.shape}, y={y_train_series.shape}"})
                if X_train_df.empty or len(X_train_df) == 0:
                    send_event("error", {
                        "message": f"训练数据为空，请检查数据获取和特征工程"
                    })
                    return

                # 处理 NaN 和 Inf 值
                X_train_df = X_train_df.fillna(0).replace([np.inf, -np.inf], 0)
                y_train_series = y_train_series.fillna(0).replace([np.inf, -np.inf], 0)

                send_event("progress", {
                    "stage": "training",
                    "progress": 80,
                    "message": f"开始训练 {request.model_type.upper()} 模型 ({n_estimators} 棵树)..."
                })

                try:
                    import time
                    train_start_time = time.time()
                    model.fit(X_train_df, y_train_series)
                    train_elapsed = time.time() - train_start_time

                    model.feature_names_ = feature_cols
                    send_event("log", {"message": f"模型训练完成 (耗时 {train_elapsed:.1f}s)"})
                except Exception as e:
                    send_event("error", {"message": f"模型训练失败: {str(e)}"})
                    return

            # 阶段5: 模型评估
            send_event("progress", {
                "stage": "evaluation",
                "progress": 90,
                "message": "评估模型性能..."
            })

            # 保存模型（如果还没有保存）
            if dataset is None:
                # 传统 sklearn 流程需要在这里保存
                import joblib
                model_dir = "models/qlib"
                os.makedirs(model_dir, exist_ok=True)
                model_id = str(uuid.uuid4())[:8]
                model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"
                joblib.dump(model, model_path)
                send_event("log", {"message": f"模型已保存: {model_path}"})

            # 评估
            if dataset is not None:
                # Qlib 原生模型评估
                try:
                    send_event("log", {"message": f"开始评估，数据集类型: {type(dataset).__name__}"})

                    # 从数据集获取训练数据进行评估
                    send_event("log", {"message": "准备获取训练集数据..."})
                    train_df = dataset.prepare("train", col_set=["feature", "label"])
                    send_event("log", {"message": f"训练集数据获取完成，类型: {type(train_df).__name__}"})
                    if isinstance(train_df, dict):
                        send_event("log", {"message": f"训练集是 dict，keys: {list(train_df.keys())}"})
                        X_train_eval = train_df["feature"]
                        y_train_eval = train_df["label"].values.ravel()
                    else:
                        send_event("log", {"message": f"训练集是 DataFrame，shape: {train_df.shape}"})
                        X_train_eval = train_df.xs('feature', axis=1, level=0)
                        y_train_eval = train_df.xs('label', axis=1, level=0).values.ravel()

                    send_event("log", {"message": f"训练集特征形状: {X_train_eval.shape}, 标签形状: {y_train_eval.shape}"})

                    # 使用 Qlib 模型预测
                    send_event("log", {"message": "开始训练集预测..."})
                    if hasattr(model, 'model') and model.model is not None:
                        train_pred = model.model.predict(X_train_eval.values)
                    else:
                        train_pred = model.predict(X_train_eval)
                    send_event("log", {"message": f"训练集预测完成，预测形状: {train_pred.shape}"} )

                    train_pred = train_pred.ravel() if len(train_pred.shape) > 1 else train_pred

                    train_mse = np.mean((train_pred - y_train_eval) ** 2)
                    train_mae = np.mean(np.abs(train_pred - y_train_eval))
                    train_ic = np.corrcoef(train_pred, y_train_eval)[0, 1] if len(train_pred) > 1 else 0

                    send_event("log", {"message": f"训练集评估: MSE={train_mse:.6f}, MAE={train_mae:.6f}, IC={train_ic:.4f}"})

                    valid_metrics = {}
                    test_metrics = {}

                    # 验证集评估
                    send_event("log", {"message": "开始验证集评估..."})
                    try:
                        valid_df = dataset.prepare("valid", col_set=["feature", "label"])
                        send_event("log", {"message": f"验证集数据获取完成，类型: {type(valid_df).__name__}"})
                        if isinstance(valid_df, dict):
                            X_valid_eval = valid_df["feature"]
                            y_valid_eval = valid_df["label"].values.ravel()
                        else:
                            X_valid_eval = valid_df.xs('feature', axis=1, level=0)
                            y_valid_eval = valid_df.xs('label', axis=1, level=0).values.ravel()

                        if hasattr(model, 'model') and model.model is not None:
                            valid_pred = model.model.predict(X_valid_eval.values)
                        else:
                            valid_pred = model.predict(X_valid_eval)

                        valid_pred = valid_pred.ravel() if len(valid_pred.shape) > 1 else valid_pred
                        valid_mse = np.mean((valid_pred - y_valid_eval) ** 2)
                        valid_mae = np.mean(np.abs(valid_pred - y_valid_eval))
                        valid_ic = np.corrcoef(valid_pred, y_valid_eval)[0, 1] if len(valid_pred) > 1 else 0

                        valid_metrics = {
                            "samples": len(y_valid_eval),
                            "mse": round(float(valid_mse), 6),
                            "mae": round(float(valid_mae), 6),
                            "ic": round(float(valid_ic), 4),
                        }
                    except Exception as ve:
                        send_event("log", {"message": f"验证集评估失败: {ve}"})
                        pass

                    # 测试集评估
                    send_event("log", {"message": "开始测试集评估..."})
                    try:
                        test_df = dataset.prepare("test", col_set=["feature", "label"])
                        send_event("log", {"message": f"测试集数据获取完成，类型: {type(test_df).__name__}"})
                        if isinstance(test_df, dict):
                            X_test_eval = test_df["feature"]
                            y_test_eval = test_df["label"].values.ravel()
                        else:
                            X_test_eval = test_df.xs('feature', axis=1, level=0)
                            y_test_eval = test_df.xs('label', axis=1, level=0).values.ravel()

                        if hasattr(model, 'model') and model.model is not None:
                            test_pred = model.model.predict(X_test_eval.values)
                        else:
                            test_pred = model.predict(X_test_eval)

                        test_pred = test_pred.ravel() if len(test_pred.shape) > 1 else test_pred
                        test_mse = np.mean((test_pred - y_test_eval) ** 2)
                        test_mae = np.mean(np.abs(test_pred - y_test_eval))
                        test_ic = np.corrcoef(test_pred, y_test_eval)[0, 1] if len(test_pred) > 1 else 0

                        test_metrics = {
                            "samples": len(y_test_eval),
                            "mse": round(float(test_mse), 6),
                            "mae": round(float(test_mae), 6),
                            "ic": round(float(test_ic), 4),
                        }
                    except Exception as te:
                        send_event("log", {"message": f"测试集评估失败: {te}"})
                        pass

                    send_event("log", {"message": "模型评估完成"})

                except Exception as e:
                    send_event("log", {"message": f"评估警告: {e}"})
                    train_mse = 0
                    train_mae = 0
                    train_ic = 0
                    valid_metrics = {}
                    test_metrics = {}

            else:
                # 传统 sklearn 流程评估
                train_pred = model.predict(X_train_df)
                train_mse = np.mean((train_pred - y_train) ** 2)
                train_mae = np.mean(np.abs(train_pred - y_train))

                def calculate_ic(predictions, data_list):
                    from scipy.stats import spearmanr
                    date_data = {}
                    for i, d in enumerate(data_list):
                        date_val = d['date']
                        if date_val not in date_data:
                            date_data[date_val] = {'pred': [], 'return': []}
                        date_data[date_val]['pred'].append(predictions[i])
                        date_data[date_val]['return'].append(d.get('return_5d', 0))

                    ics = []
                    for date_val, data in date_data.items():
                        preds = np.array(data['pred'])
                        returns = np.array(data['return'])
                        if len(preds) >= 5:
                            if np.std(preds) > 1e-10 and np.std(returns) > 1e-10:
                                try:
                                    ic, _ = spearmanr(preds, returns)
                                    if not np.isnan(ic):
                                        ics.append(ic)
                                except:
                                    pass
                    return np.mean(ics) if ics else 0.0

                train_ic = calculate_ic(train_pred, train_data)

                valid_metrics = {}
                if valid_data:
                    X_valid = np.array([d['features'] for d in valid_data])
                    y_valid = np.array([d['label'] for d in valid_data])
                    X_valid_df = pd.DataFrame(X_valid, columns=feature_cols)
                    valid_pred = model.predict(X_valid_df)
                    valid_mse = np.mean((valid_pred - y_valid) ** 2)
                    valid_mae = np.mean(np.abs(valid_pred - y_valid))
                    valid_metrics = {
                        "samples": len(valid_data),
                        "mse": round(float(valid_mse), 6),
                        "mae": round(float(valid_mae), 6),
                        "ic": round(float(calculate_ic(valid_pred, valid_data)), 4),
                    }

                test_metrics = {}
                if test_data:
                    X_test = np.array([d['features'] for d in test_data])
                    y_test = np.array([d['label'] for d in test_data])
                    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
                    test_pred = model.predict(X_test_df)
                    test_mse = np.mean((test_pred - y_test) ** 2)
                    test_mae = np.mean(np.abs(test_pred - y_test))
                    test_metrics = {
                        "samples": len(test_data),
                        "mse": round(float(test_mse), 6),
                        "mae": round(float(test_mae), 6),
                        "ic": round(float(calculate_ic(test_pred, test_data)), 4),
                    }

            # 阶段6: 完成
            send_event("progress", {
                "stage": "complete",
                "progress": 100,
                "message": "训练完成！"
            })

            # 统计样本数
            train_samples = len(train_data) if train_data else 0
            valid_samples = len(valid_data) if valid_data else 0
            test_samples = len(test_data) if test_data else 0

            # 如果使用 Qlib 原生流程，从 dataset 获取样本数
            if dataset is not None:
                try:
                    train_df = dataset.prepare("train", col_set=["feature"])
                    train_samples = len(train_df) if hasattr(train_df, '__len__') else 0
                    valid_df = dataset.prepare("valid", col_set=["feature"])
                    valid_samples = len(valid_df) if hasattr(valid_df, '__len__') else 0
                    test_df = dataset.prepare("test", col_set=["feature"])
                    test_samples = len(test_df) if hasattr(test_df, '__len__') else 0
                except Exception:
                    pass

            result = {
                "model_id": model_id,
                "model_type": request.model_type,
                "model_path": model_path,
                "train_symbols_count": len(train_symbols),
                "predict_symbols": request.symbols,
                "train_samples": train_samples,
                "features": list(feature_cols) if feature_cols else [],
                "data_split": {
                    "train": {"period": f"{request.train_start} ~ {request.train_end}", "samples": train_samples},
                    "valid": {"period": f"{request.valid_start} ~ {request.valid_end}", "samples": valid_samples},
                    "test": {"period": f"{request.test_start} ~ {request.test_end}", "samples": test_samples},
                },
                "metrics": {
                    "train": {
                        "samples": train_samples,
                        "mse": round(float(train_mse), 6),
                        "mae": round(float(train_mae), 6),
                        "ic": round(float(train_ic), 4) if not np.isnan(train_ic) else 0,
                    },
                    "valid": valid_metrics,
                    "test": test_metrics,
                },
                "backtest_params": {
                    "initial_cash": request.initial_cash,
                    "commission_rate": request.commission_rate,
                    "slippage_rate": request.slippage_rate,
                    "t_plus_1": True,
                },
            }

            send_event("complete", {"result": result})

        except Exception as e:
            import traceback
            traceback.print_exc()
            send_event("error", {"message": f"训练失败: {str(e)}"})

    async def event_stream():
        """生成SSE事件流"""
        loop = asyncio.get_event_loop()

        # 在线程池中运行训练工作器
        thread = threading.Thread(target=training_worker)
        thread.start()

        # 从同步队列读取事件并发送
        while True:
            try:
                # 使用 run_in_executor 非阻塞地获取消息
                msg = await loop.run_in_executor(None, lambda: message_queue.get(timeout=0.1))

                event_type = msg.get("event", "message")
                data = msg.get("data", {})

                # 格式化SSE
                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                # 完成或错误时结束
                if event_type in ("complete", "error"):
                    break
            except queue.Empty:
                # 检查线程是否结束
                if not thread.is_alive():
                    break
                # 发送心跳保持连接
                yield "event: heartbeat\ndata: {}\n\n"

        # 等待线程结束
        thread.join()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/qlib/predict")
async def predict_with_qlib_model(request: QlibPredictRequest) -> Dict[str, Any]:
    """
    使用 Qlib ML 模型进行预测并回测

    返回预测结果、信号和回测收益
    """
    try:
        import joblib
        from quanttool.factors.stock_analyzer import StockAnalyzer
        import numpy as np
        from datetime import datetime, timedelta
        from pathlib import Path

        # 查找模型文件
        model_path = request.model_path
        if not model_path:
            # 自动查找对应 model_type 的最新模型
            model_dir = Path("models/qlib")
            if model_dir.exists():
                # 查找匹配的模型文件
                pattern = f"{request.model_type}_*.pkl"
                model_files = list(model_dir.glob(pattern))

                if not model_files:
                    # 尝试其他命名方式
                    all_models = list(model_dir.glob("*.pkl"))
                    if all_models:
                        # 使用最新的模型
                        model_files = sorted(all_models, key=lambda x: x.stat().st_mtime, reverse=True)[:1]
                        logger.info(f"No {request.model_type} model found, using latest: {model_files[0].name}")
                else:
                    # 按修改时间排序，取最新的
                    model_files = sorted(model_files, key=lambda x: x.stat().st_mtime, reverse=True)[:1]

                if model_files:
                    model_path = str(model_files[0])
                    logger.info(f"Auto-selected model: {model_path}")

        if not model_path:
            raise HTTPException(
                status_code=400,
                detail=f"未找到已保存的模型。请先训练模型，或检查 models/qlib/ 目录下是否有 {request.model_type}_*.pkl 文件"
            )

        logger.info(f"Loading model from: {model_path}")
        saved_data = joblib.load(model_path)

        # 兼容两种保存格式：直接保存模型 或 保存为字典
        if isinstance(saved_data, dict):
            model = saved_data.get('model')
            feature_names = saved_data.get('feature_names', request.features)
        else:
            # 直接保存的模型对象
            model = saved_data
            feature_names = getattr(model, 'feature_names_', request.features)

        if model is None:
            raise HTTPException(status_code=400, detail="模型文件无效")

        # 获取内部模型进行预测
        inner_model = None
        if hasattr(model, 'model'):
            inner_model = model.model
        elif hasattr(model, 'booster'):
            inner_model = model.booster
        else:
            inner_model = model

        # 获取预测数据（使用实时价格数据，避免 qlib 复权价格显示异常）
        analyzer = StockAnalyzer(use_realtime_price=True)
        predictions = {}

        # 解析回测日期（使用动态默认值：最近一年）
        predict_start = datetime.fromisoformat(request.get_predict_start_date())
        predict_end = datetime.fromisoformat(request.get_predict_end_date())

        # 回测参数
        initial_cash = request.initial_cash
        commission_rate = request.commission_rate
        slippage_rate = request.slippage_rate

        # 回测结果
        backtest_results = {
            "initial_cash": initial_cash,
            "final_capital": initial_cash,
            "total_return": 0.0,
            "annual_return": 0.0,
            "total_trades": 0,
            "win_trades": 0,
            "total_commission": 0.0,
            "total_slippage": 0.0,
            "trades": [],
        }

        # 初始化 Alpha158 特征工程器
        from quanttool.strategies.qlib_strategy import QlibFeatureEngineer
        feature_engineer = QlibFeatureEngineer(feature_set=request.feature_set)

        for symbol in request.symbols:
            df = analyzer.get_stock_data(symbol, 500)  # 获取更多数据用于回测
            if df.empty or len(df) < 120:
                continue

            # 确定日期列
            date_column = None
            if 'trade_date' in df.columns:
                date_column = 'trade_date'
            elif 'timestamp' in df.columns:
                date_column = 'timestamp'

            if not date_column:
                continue

            df['_date'] = pd.to_datetime(df[date_column])

            # 使用 Alpha158 特征工程
            if request.use_rich_features:
                try:
                    feature_df = feature_engineer.generate_features(df)
                    df = pd.concat([df, feature_df], axis=1)
                except Exception as e:
                    logger.warning(f"Feature engineering failed for {symbol}: {e}")
                    continue
            else:
                df = analyzer.calculate_technical_indicators(df)

            # 使用模型期望的特征
            available_features = [f for f in feature_names if f in df.columns]
            if not available_features:
                continue

            # 记录数据日期范围
            data_start = str(df['_date'].min())[:10]
            data_end = str(df['_date'].max())[:10]

            # ====== 回测逻辑 ======
            cash = initial_cash
            position = 0  # 持仓数量
            trades = []
            total_commission = 0.0
            total_slippage = 0.0

            # T+1 交易：记录买入日期，卖出时检查是否满足 T+1
            buy_date = None
            buy_price = 0.0

            for i in range(len(df) - 5):  # 留出预测窗口
                row = df.iloc[i]

                # 获取交易日期
                trade_date = None
                if date_column and date_column in row:
                    trade_date = row[date_column]
                elif df.index.name:
                    trade_date = df.index[i]

                if trade_date is None:
                    continue

                # 检查是否在回测日期范围内
                try:
                    if hasattr(trade_date, 'to_pydatetime'):
                        trade_dt = trade_date.to_pydatetime()
                    elif isinstance(trade_date, str):
                        trade_dt = datetime.fromisoformat(trade_date[:10])
                    elif hasattr(trade_date, 'strftime'):
                        # pandas Timestamp
                        trade_dt = trade_date.to_pydatetime()
                    else:
                        continue
                except:
                    continue

                if trade_dt < predict_start or trade_dt > predict_end:
                    continue

                # 获取当日特征
                X = df[available_features].iloc[i:i+1].values

                try:
                    pred = inner_model.predict(X)[0]
                except:
                    try:
                        pred = float(inner_model.predict(X.reshape(1, -1))[0])
                    except:
                        continue

                if isinstance(pred, (int, float)):
                    pred_value = float(pred)
                elif hasattr(pred, '__len__'):
                    pred_value = float(pred[0])
                else:
                    pred_value = float(pred)

                # 生成信号 (回归模型预测收益率，阈值需要适配)
                # 回归值范围通常在 -0.1 到 0.1 之间
                signal = "hold"
                if pred_value > 0.005:  # 预测上涨 > 0.5%
                    signal = "buy"
                elif pred_value < -0.005:  # 预测下跌 > 0.5%
                    signal = "sell"

                # 获取价格
                close_price = float(row['close'])

                # 执行交易 (考虑 T+1)
                if signal == "buy" and position == 0 and cash > 0:
                    # 买入
                    slippage = close_price * slippage_rate
                    buy_price_actual = close_price + slippage
                    shares = int(cash / buy_price_actual / 100) * 100  # A股一手100股

                    if shares > 0:
                        commission = max(shares * buy_price_actual * commission_rate, 5)  # 最低5元
                        total_cost = shares * buy_price_actual + commission

                        if total_cost <= cash:
                            position = shares
                            cash -= total_cost
                            buy_date = trade_dt
                            buy_price = buy_price_actual
                            total_commission += commission
                            total_slippage += shares * slippage
                            trades.append({
                                "type": "buy",
                                "date": str(trade_date)[:10],
                                "price": round(buy_price_actual, 2),
                                "shares": shares,
                                "commission": round(commission, 2),
                                "slippage": round(shares * slippage, 2)
                            })

                elif signal == "sell" and position > 0:
                    # T+1 检查：卖出日期必须比买入日期晚至少1天
                    if buy_date is None or trade_dt <= buy_date:
                        continue

                    # 卖出
                    slippage = close_price * slippage_rate
                    sell_price_actual = close_price - slippage
                    sell_amount = position * sell_price_actual
                    commission = max(sell_amount * commission_rate, 5)

                    profit = position * (sell_price_actual - buy_price) - commission
                    cash += sell_amount - commission
                    total_commission += commission
                    total_slippage += position * slippage

                    trades.append({
                        "type": "sell",
                        "date": str(trade_date)[:10],
                        "price": round(sell_price_actual, 2),
                        "shares": position,
                        "commission": round(commission, 2),
                        "slippage": round(position * slippage, 2),
                        "profit": round(profit, 2)
                    })

                    if profit > 0:
                        backtest_results["win_trades"] += 1

                    position = 0
                    buy_date = None

            # 计算最终市值
            if len(df) > 0:
                final_price = float(df['close'].iloc[-1])
                final_capital = cash + position * final_price
            else:
                final_capital = cash

            total_return = (final_capital - initial_cash) / initial_cash

            # 计算年化收益
            days = (predict_end - predict_start).days
            annual_return = total_return * 252 / max(days, 1) if days > 0 else 0

            # ====== 计算最大回撤 ======
            max_drawdown = 0.0
            if trades:
                # 重建市值曲线
                equity_curve = [initial_cash]
                peak_equity = initial_cash

                # 简化：根据交易记录估算市值变化
                running_cash = initial_cash
                running_position = 0
                running_buy_price = 0.0

                for trade in trades:
                    if trade['type'] == 'buy':
                        running_cash -= trade['shares'] * trade['price'] + trade['commission'] + trade.get('slippage', 0)
                        running_position = trade['shares']
                        running_buy_price = trade['price']
                    elif trade['type'] == 'sell':
                        running_cash += trade['shares'] * trade['price'] - trade['commission']
                        running_position = 0

                    # 假设当日市值为现金（简化计算）
                    equity = running_cash
                    equity_curve.append(equity)

                    # 计算回撤
                    if equity > peak_equity:
                        peak_equity = equity
                    drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

            # ====== 计算夏普比率 ======
            sharpe_ratio = 0.0
            if days > 0 and total_return != 0:
                # 简化：使用年化收益率和假设的波动率
                # 实际应该用每日收益率计算
                # 这里用估算：假设年化波动率约 20%
                assumed_volatility = 0.20
                risk_free_rate = 0.02  # 无风险利率 2%
                if assumed_volatility > 0:
                    sharpe_ratio = (annual_return - risk_free_rate) / assumed_volatility

            # 获取最新预测
            X_latest = df[available_features].iloc[-1:].values
            try:
                pred_latest = inner_model.predict(X_latest)[0]
            except:
                pred_latest = 0.5

            predictions[symbol] = {
                "prediction": round(float(pred_latest), 4),
                "signal": "buy" if float(pred_latest) > 0.55 else ("sell" if float(pred_latest) < 0.45 else "hold"),
                "latest_price": round(float(df['close'].iloc[-1]), 2),
                "data_period": {
                    "start_date": data_start,
                    "end_date": data_end,
                },
                "backtest": {
                    "initial_cash": initial_cash,
                    "final_capital": round(final_capital, 2),
                    "total_return": round(total_return * 100, 2),
                    "annual_return": round(annual_return * 100, 2),
                    "max_drawdown": round(max_drawdown * 100, 2),
                    "sharpe_ratio": round(sharpe_ratio, 2),
                    "total_trades": len(trades),
                    "win_rate": round(backtest_results["win_trades"] / len(trades) * 100, 1) if trades else 0,
                    "total_commission": round(total_commission, 2),
                    "total_slippage": round(total_slippage, 2),
                    "trades": trades[-10:],  # 最近10笔交易
                }
            }

            backtest_results["total_trades"] += len(trades)

        # 计算汇总统计
        total_final_capital = sum(
            p["backtest"]["final_capital"] for p in predictions.values()
        )
        total_win_trades = sum(
            1 for p in predictions.values()
            for t in p["backtest"].get("trades", [])
            if t.get("type") == "sell" and t.get("profit", 0) > 0
        )
        total_sell_trades = sum(
            1 for p in predictions.values()
            for t in p["backtest"].get("trades", [])
            if t.get("type") == "sell"
        )

        # 汇总回测结果
        summary = {
            "total_return_pct": round((total_final_capital - initial_cash * len(predictions)) / (initial_cash * len(predictions)) * 100, 2) if predictions else 0,
            "total_trades": backtest_results["total_trades"],
            "win_rate": round(total_win_trades / total_sell_trades * 100, 1) if total_sell_trades > 0 else 0,
            "predicted_stocks": len(predictions),
        }

        return {
            "success": True,
            "model_type": request.model_type,
            "model_path": model_path,
            "model_name": Path(model_path).name if model_path else None,
            "feature_count": len(feature_names),
            "predict_period": {
                "start_date": request.predict_start_date,
                "end_date": request.predict_end_date,
                "days": (predict_end - predict_start).days,
            },
            "backtest_params": {
                "initial_cash": f"¥{initial_cash:,.0f}",
                "initial_cash_raw": initial_cash,
                "commission_rate": f"{commission_rate * 100:.4f}%",
                "slippage_rate": f"{slippage_rate * 100:.4f}%",
                "total_cost_rate": f"{(commission_rate + slippage_rate) * 100:.4f}%",
                "t_plus_1": True,
            },
            "summary": summary,
            "predictions": predictions,
            "total_stocks": len(request.symbols),
            "predicted_stocks": len(predictions)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


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
