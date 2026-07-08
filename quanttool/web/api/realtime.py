"""Realtime quote API routes."""

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

from .dependencies import (
    get_incremental_minute_provider,
    get_minute_provider,
    get_realtime_provider,
)
from ..schemas.realtime import RealtimeQuoteResponse


@router.get("/realtime/quote/{symbol}")
async def get_realtime_quote(symbol: str) -> Dict[str, Any]:
    """获取实时行情（使用新的实时数据通路）"""
    try:
        # 使用新的统一实时数据提供者
        provider = get_realtime_provider()
        quote = provider.get_realtime_quote(symbol)

        if not quote:
            raise HTTPException(status_code=404, detail=f"无法获取 {symbol} 的实时行情")

        # 转换为字典
        quote_dict = quote.to_dict()

        # 处理 timestamp
        ts = quote_dict.get("timestamp")
        if ts:
            if isinstance(ts, datetime):
                ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
            else:
                ts_str = str(ts)
        else:
            ts_str = ""

        return {
            "symbol": quote_dict.get("symbol", symbol),
            "name": quote_dict.get("name", ""),
            "price": float(quote_dict.get("price", 0) or 0),
            "open": float(quote_dict.get("open", 0) or 0),
            "high": float(quote_dict.get("high", 0) or 0),
            "low": float(quote_dict.get("low", 0) or 0),
            "volume": float(quote_dict.get("volume", 0) or 0),
            "amount": float(quote_dict.get("amount", 0) or 0),
            "change_pct": float(quote_dict.get("change_pct", 0) or 0),
            "change": float(quote_dict.get("change_amount", 0) or 0),
            "turnover": float(quote_dict.get("turnover_rate", 0) or 0),
            "timestamp": ts_str,
            "source": quote_dict.get("source", ""),
            "bid_prices": quote_dict.get("bid_prices", []),
            "ask_prices": quote_dict.get("ask_prices", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get realtime quote for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"获取实时行情失败: {str(e)}")


@router.post("/realtime/batch")
async def get_realtime_quotes_batch(request: Request) -> List[Dict[str, Any]]:
    """批量获取实时行情"""
    try:
        body = await request.json()
        symbols = body.get("symbols", [])

        if not symbols:
            return []

        provider = get_realtime_provider()
        results = []

        for symbol in symbols:
            try:
                quote = provider.get_realtime_quote(symbol)
                if quote:
                    quote_dict = quote.to_dict()
                    ts = quote_dict.get("timestamp")
                    if ts:
                        if isinstance(ts, datetime):
                            ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
                        else:
                            ts_str = str(ts)
                    else:
                        ts_str = ""

                    # 计算涨跌幅（如果没有提供）
                    price = float(quote_dict.get("price", 0) or 0)
                    pre_close = float(quote_dict.get("pre_close", 0) or 0)
                    change_pct = float(quote_dict.get("change_pct", 0) or 0)
                    change_amount = float(quote_dict.get("change_amount", 0) or 0)

                    # 如果涨跌幅为0但有昨收价，则计算
                    if change_pct == 0 and pre_close > 0 and price > 0:
                        change_amount = price - pre_close
                        change_pct = change_amount / pre_close

                    results.append({
                        "symbol": quote_dict.get("symbol", symbol),
                        "name": quote_dict.get("name", ""),
                        "price": price,
                        "open": float(quote_dict.get("open", 0) or 0),
                        "high": float(quote_dict.get("high", 0) or 0),
                        "low": float(quote_dict.get("low", 0) or 0),
                        "volume": float(quote_dict.get("volume", 0) or 0),
                        "amount": float(quote_dict.get("amount", 0) or 0),
                        "change_pct": change_pct,
                        "change": change_amount,
                        "turnover": float(quote_dict.get("turnover_rate", 0) or 0),
                        "timestamp": ts_str,
                        "source": quote_dict.get("source", ""),
                    })
            except Exception as e:
                logger.warning(f"Failed to get quote for {symbol}: {e}")
                # 添加一个空记录表示失败
                results.append({
                    "symbol": symbol,
                    "name": "",
                    "price": 0,
                    "open": 0,
                    "high": 0,
                    "low": 0,
                    "volume": 0,
                    "amount": 0,
                    "change_pct": 0,
                    "change": 0,
                    "turnover": 0,
                    "timestamp": "",
                    "source": "",
                    "error": str(e),
                })

        return results
    except Exception as e:
        logger.error(f"Failed to get batch realtime quotes: {e}")
        raise HTTPException(status_code=500, detail=f"批量获取行情失败: {str(e)}")


@router.get("/realtime/kline/{symbol}")
async def get_realtime_kline(
    symbol: str,
    timeframe: str = "5m",
    count: int = 60
) -> Dict[str, Any]:
    """获取分钟K线数据（使用增量分钟数据通路）"""
    try:
        # 使用新的增量分钟数据提供者
        provider = get_incremental_minute_provider()
        df = provider.get_minute_bars(symbol, timeframe, count=count)

        if df.empty:
            # 回退到旧的 AkShare 提供者
            old_provider = get_minute_provider()
            df = old_provider.get_latest_bars(symbol, count, timeframe)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"无法获取 {symbol} 的K线数据")

        # 转换为前端友好的格式
        kline_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": []
        }

        for _, row in df.iterrows():
            bar = {
                "timestamp": row["timestamp"].strftime("%Y-%m-%d %H:%M:%S") if pd.notna(row.get("timestamp")) else "",
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "close": float(row.get("close", 0)),
                "volume": float(row.get("volume", 0)),
                "amount": float(row.get("amount", 0))
            }
            kline_data["bars"].append(bar)

        return kline_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get kline for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"获取K线数据失败: {str(e)}")


@router.get("/realtime/search")
async def search_stocks(query: str = "", limit: int = 20) -> List[Dict[str, Any]]:
    """搜索股票"""
    if not query:
        return []

    # 检查缓存
    cache_key = f"search_{query}_{limit}"
    cached = get_cached_analysis(cache_key)
    if cached:
        logger.info(f"Search cache hit for {cache_key}")
        return cached

    logger.info(f"Search cache miss for {cache_key}")

    try:
        provider = get_minute_provider()
        results = provider.search_symbols(query)

        # 格式化结果
        formatted = []
        for item in results[:limit]:
            formatted.append({
                "symbol": item.get("symbol", ""),
                "name": item.get("name", ""),
                "price": float(item.get("price", 0))
            })

        # 缓存结果
        set_cached_analysis(cache_key, formatted)
        return formatted
    except Exception as e:
        logger.warning(f"AkShare search failed: {e}, using fallback")

        # 降级策略：使用本地静态数据
        static_stocks = [
            {"symbol": "600519", "name": "贵州茅台"},
            {"symbol": "000001", "name": "平安银行"},
            {"symbol": "000002", "name": "万科A"},
            {"symbol": "000333", "name": "美的集团"},
            {"symbol": "000651", "name": "格力电器"},
            {"symbol": "000858", "name": "五粮液"},
            {"symbol": "002415", "name": "海康威视"},
            {"symbol": "002594", "name": "比亚迪"},
            {"symbol": "300750", "name": "宁德时代"},
            {"symbol": "601318", "name": "中国平安"},
            {"symbol": "601398", "name": "工商银行"},
            {"symbol": "601939", "name": "建设银行"},
            {"symbol": "600036", "name": "招商银行"},
            {"symbol": "600276", "name": "恒瑞医药"},
            {"symbol": "600887", "name": "伊利股份"},
            {"symbol": "603259", "name": "药明康德"},
            {"symbol": "600309", "name": "万华化学"},
            {"symbol": "002304", "name": "洋河股份"},
            {"symbol": "000568", "name": "泸州老窖"},
            {"symbol": "002352", "name": "顺丰控股"},
        ]

        # 简单匹配
        query_lower = query.lower()
        results = []
        for stock in static_stocks:
            if query_lower in stock["symbol"].lower() or query_lower in stock["name"].lower():
                results.append({
                    "symbol": stock["symbol"],
                    "name": stock["name"],
                    "price": 0
                })

        # 缓存结果
        set_cached_analysis(cache_key, results[:limit])
        return results[:limit]
