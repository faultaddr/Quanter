"""ML stock scan API route."""

import os
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ....core.logging import get_logger
from ...schemas.ml import MLScanRequest
from ..utils import to_python_types


logger = get_logger(__name__)
router = APIRouter()

@router.post("/ml/scan")
async def scan_with_ml_model(request: MLScanRequest) -> Dict[str, Any]:
    """
    使用 ML 模型进行智能选股

    对候选股票进行预测，返回得分最高的股票
    """
    try:
        from quanttool.strategies.gbm_strategy import GBMStrategy, GBMConfig
        from quanttool.infrastructure.data_providers.qlib_data_loader import QlibDataLoader
        from quanttool.cli.commands.analysis_commands import get_csi300_constituents
        import glob

        # 查找模型
        model_path = request.model_path
        if not model_path:
            model_files = glob.glob("models/gbm/lgbm_*.pkl")
            if not model_files:
                raise HTTPException(status_code=404, detail="未找到训练好的模型，请先训练模型")
            model_path = max(model_files, key=os.path.getmtime)
            logger.info(f"使用最新模型: {model_path}")

        # 获取候选股票
        symbols = request.symbols
        if not symbols:
            # 默认使用沪深300成分股
            csi300 = get_csi300_constituents()
            symbols = [s['code'] if isinstance(s, dict) else s for s in csi300]

        # 加载模型
        config = GBMConfig()
        strategy = GBMStrategy(config)
        strategy.load_model(model_path)

        # 初始化数据加载器
        data_loader = QlibDataLoader()
        if not data_loader.init_qlib():
            raise HTTPException(status_code=500, detail="Qlib 初始化失败")

        # 预测所有股票
        results = []
        for symbol in symbols:
            try:
                pred = strategy.predict(symbol)
                if pred.get('probability', 0) >= request.min_probability:
                    results.append({
                        'symbol': symbol,
                        'probability': pred['probability'],
                        'pred_return': pred.get('return_pred', 0),
                        'signal': pred.get('signal', 'hold'),
                        'close': pred.get('close', 0),
                    })
            except Exception as e:
                logger.debug(f"预测失败 {symbol}: {e}")
                continue

        # 按概率排序
        results.sort(key=lambda x: x['probability'], reverse=True)
        top_results = results[:request.top_n]

        return {
            "success": True,
            "model_path": model_path,
            "total_scanned": len(symbols),
            "qualified_count": len(results),
            "min_probability": request.min_probability,
            "results": to_python_types(top_results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ML 选股失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"选股失败: {str(e)}")
