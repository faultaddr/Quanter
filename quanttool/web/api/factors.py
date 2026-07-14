"""Factor API routes."""

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

from quanttool.application.factor_service import FactorService


@router.post("/factors/mine")
async def mine_factors(request_data: Dict[str, Any]) -> Dict[str, Any]:
    """Mine factors across a universe of stocks."""
    try:
        # Extract parameters
        factor_name = request_data.get("factor_name", "momentum")
        symbols = request_data.get("symbols", [])
        start_date_str = request_data.get("start_date", "2023-01-01")
        end_date_str = request_data.get("end_date", "2023-12-31")

        start_date = datetime.fromisoformat(start_date_str)
        end_date = datetime.fromisoformat(end_date_str)

        # Initialize factor service
        factor_service = FactorService()

        # Run factor mining
        results = factor_service.mine_factor(
            factor_name=factor_name,
            factor_params=request_data.get("factor_params", {}),
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            data_provider=request_data.get("data_provider", "tushare"),
        )

        # Convert results to serializable format
        serialized_results = {}
        for symbol, result in results.items():
            serialized_results[symbol] = {
                "factor_name": result.factor_name,
                "ic": result.ic,
                "rank_ic": result.rank_ic,
                "win_rate": result.win_rate,
                "avg_return": result.avg_return,
                "volatility": result.volatility,
                "sharpe_ratio": result.sharpe_ratio,
            }

        # Save to metadata DB
        from ...infrastructure.stores.meta_db_async import get_async_meta_db
        import uuid

        db = get_async_meta_db()
        run_id = str(uuid.uuid4())
        await db.save_experiment_run(
            {
                "id": run_id,
                "type": "factor_mining",
                "parameters": request_data,
                "git_commit": "unknown",
                "data_version": "v1.0",
                "start_time": datetime.now(),
                "end_time": datetime.now(),
                "status": "completed",
                "results": serialized_results,
                "artifacts": [],
            }
        )

        return {"run_id": run_id, "results": serialized_results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error mining factors: {str(e)}")


@router.post("/factors/validate")
async def validate_factor(
    factor_values: List[float],
    returns: List[float],
    factor_name: str = "factor",
) -> dict:
    """因子有效性检验 - IC/IR分析

    Args:
        factor_values: 因子值序列
        returns: 收益率序列
        factor_name: 因子名称

    Returns:
        因子有效性检验结果
    """
    import pandas as pd
    from quanttool.factors.factor_validator import FactorValidator

    # 转换为Series
    factor_series = pd.Series(factor_values)
    returns_series = pd.Series(returns)

    # 验证因子
    validator = FactorValidator()
    report = validator.validate(factor_series, returns_series, factor_name)

    return {
        "factor_name": report.factor_name,
        "ic_mean": report.ic_result.mean_ic if report.ic_result else 0,
        "ic_std": report.ic_result.std_ic if report.ic_result else 0,
        "ir": report.ic_result.ir if report.ic_result else 0,
        "long_short_return": report.quantile_result.long_short_return if report.quantile_result else 0,
        "overall_score": report.overall_score,
        "is_effective": report.is_effective,
        "recommendations": report.recommendations,
    }


# ==================== 因子优化 API ====================

@router.post("/factors/optimize")
async def optimize_factor_weights(
    factor_names: List[str],
    ic_history: Dict[str, List[float]],
    method: str = "ir_weighted",
) -> dict:
    """因子权重优化

    Args:
        factor_names: 因子名称列表
        ic_history: 各因子IC历史 {factor_name: [ic_values]}
        method: 优化方法 (equal, ic_weighted, ir_weighted, risk_parity)

    Returns:
        优化后的权重配置
    """
    from quanttool.optimization.weight_optimizer import ICIRWeightOptimizer, OptimizerType

    optimizer = ICIRWeightOptimizer()

    # 更新因子IC数据
    for name, ic_values in ic_history.items():
        import pandas as pd
        optimizer.update_factor_metrics(name, pd.Series(ic_values))

    # 选择优化方法
    opt_type = OptimizerType.IR_WEIGHTED
    if method == "equal":
        opt_type = OptimizerType.EQUAL
    elif method == "ic_weighted":
        opt_type = OptimizerType.IC_WEIGHTED
    elif method == "risk_parity":
        opt_type = OptimizerType.RISK_PARITY

    # 优化权重
    weights = optimizer.optimize(factor_names, opt_type)

    return {"weights": weights, "method": method}


# ==================== 组合风险管理 API ====================

# 定义请求模型
