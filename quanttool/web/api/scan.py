"""Market scan API routes."""

from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from ..schemas.scan import ScanRequest


def _configure_scan_context(
    analyzer: Any,
    include_fundamentals: bool,
    include_market_state: bool,
) -> Dict[str, str]:
    """Keep broad market scans responsive unless deep context is requested."""
    orchestrator = getattr(analyzer, "analysis_orchestrator", None)
    if orchestrator is None:
        return {"fundamental_mode": "unavailable", "market_state_mode": "unavailable"}

    setattr(analyzer, "_scan_fast_mode", not include_market_state)

    if include_fundamentals:
        fundamental_mode = "full"
    else:
        from quanttool.factors.analysis_context import FundamentalData

        orchestrator.fundamental_provider = lambda symbol: FundamentalData()
        fundamental_mode = "skipped"

    if include_market_state:
        market_state_mode = "full"
    else:
        from quanttool.factors.analysis_context import UnifiedMarketState

        orchestrator.market_state_builder = lambda df: UnifiedMarketState(
            index_code="000300.SH",
            index_name="沪深300",
        )
        market_state_mode = "skipped"

    return {
        "fundamental_mode": fundamental_mode,
        "market_state_mode": market_state_mode,
    }


@router.post("/scan")
@router.post("/scan/market")
async def scan_stocks(request: ScanRequest) -> Dict[str, Any]:
    """
    股票扫描筛选 - 对应 CLI: quanttool analysis scan

    扫描市场寻找潜在机会
    """
    try:
        from quanttool.cli.commands.analysis_commands import (
            get_csi300_constituents,
            get_csi1000_constituents,
        )

        # 获取股票列表
        if request.market.lower() == "csi300":
            stock_list = get_csi300_constituents()
        elif request.market.lower() == "csi1000":
            stock_list = get_csi1000_constituents()
        else:
            raise HTTPException(status_code=400, detail=f"不支持的市场: {request.market}")

        if not stock_list:
            return {
                "market": request.market,
                "total_stocks": 0,
                "analyzed_stocks": 0,
                "top_n": request.top_n,
                "results": [],
            }

        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.cli.commands.analysis_commands import (
            analyze_stock_unified_score,
            analyze_stock_trend_score,
            analyze_stock_breakout_score,
            analyze_stock_momentum_score,
            analyze_stock_score
        )

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
        scan_context_modes = _configure_scan_context(
            analyzer,
            request.include_fundamentals,
            request.include_market_state,
        )
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)

        # 先并发预加载所有股票数据（显著提升性能）
        print(f"正在预加载 {len(stock_list)} 只股票数据...")
        loaded_count = analyzer.preload_data_for_scan(stock_list, request.days)
        print(f"成功预加载 {loaded_count} 只股票数据")

        symbols = [stock['code'] if isinstance(stock, dict) else stock for stock in stock_list]
        print("正在批量获取实时价格...")
        realtime_count = analyzer.preload_realtime_prices(symbols)
        print(f"成功获取 {realtime_count} 只股票实时价格")

        results = []

        def analyze_single_stock(stock_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            if request.use_unified_score:
                result, _ = analyze_stock_unified_score(
                    stock_info, request.days, analyzer, start_date, end_date
                )
            elif request.use_momentum_score:
                result, _ = analyze_stock_momentum_score(
                    stock_info, request.days, analyzer, start_date, end_date
                )
            elif request.use_breakout_score:
                result, _ = analyze_stock_breakout_score(
                    stock_info, request.days, analyzer, start_date, end_date
                )
            elif request.use_trend_score:
                result, _ = analyze_stock_trend_score(
                    stock_info, request.days, analyzer, start_date, end_date
                )
            else:
                result, _ = analyze_stock_score(
                    stock_info, request.days, analyzer, None, None, True, start_date, end_date
                )
            return result

        max_workers = min(10, len(stock_list))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(analyze_single_stock, stock_info) for stock_info in stock_list]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        # 排序
        results.sort(key=lambda x: x['score'], reverse=True)
        top_results = results[:request.top_n]

        return {
            "market": request.market,
            "total_stocks": len(stock_list),
            "analyzed_stocks": len(results),
            "top_n": request.top_n,
            "scoring_mode": (
                "unified" if request.use_unified_score else
                "momentum" if request.use_momentum_score else
                "breakout" if request.use_breakout_score else
                "trend" if request.use_trend_score else
                "classic"
            ),
            **scan_context_modes,
            "results": to_python_types(top_results)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"扫描失败: {str(e)}")


@router.get("/scan/markets")
async def list_scan_markets() -> List[Dict[str, str]]:
    """列出可扫描的市场"""
    return [
        {"code": "csi300", "name": "沪深300", "description": "沪深300成分股"},
        {"code": "csi1000", "name": "中证1000", "description": "中证1000成分股"},
    ]
