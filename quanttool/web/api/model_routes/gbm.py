"""GBM model API routes."""

from datetime import datetime
import os
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from ....core.logging import get_logger
from ...schemas.model import GBMPicksRequest, GBMPredictRequest, GBMTrainRequest
from ..utils import to_python_types


logger = get_logger(__name__)
router = APIRouter()

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
