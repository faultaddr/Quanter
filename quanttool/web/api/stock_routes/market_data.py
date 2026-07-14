"""Stock market data API routes."""

from datetime import datetime, timedelta
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException

from ....core.logging import get_logger
from ..utils import get_cached_analysis, set_cached_analysis


logger = get_logger(__name__)
router = APIRouter()

@router.get("/stock/{symbol}/info")
async def get_stock_info(symbol: str) -> Dict[str, Any]:
    """获取股票基本信息和最新数据"""
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
        df = analyzer.get_stock_data(symbol, 30)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {symbol}")

        latest = df.iloc[-1]

        return {
            "symbol": symbol,
            "latest_price": float(latest.get('close', 0)),
            "volume": int(latest.get('volume', 0)),
            "high": float(latest.get('high', 0)),
            "low": float(latest.get('low', 0)),
            "date": str(latest.get('trade_date', latest.get('timestamp', ''))),
            "data_days": len(df)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取股票信息失败: {str(e)}")


@router.get("/stock/{symbol}/kline")
async def get_stock_kline(symbol: str, days: int = 60) -> Dict[str, Any]:
    """
    获取股票 K 线数据（用于图表展示）

    Args:
        symbol: 股票代码
        days: 获取天数，默认60天

    Returns:
        K线数据和指标数据
    """
    # 检查缓存
    cache_key = f"kline_{symbol}_{days}"
    cached = get_cached_analysis(cache_key)
    if cached:
        return cached

    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
        df = analyzer.get_stock_data(symbol, days)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {symbol} 的数据")

        # 计算技术指标
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # 转换为前端可用的格式
        kline_data = []
        volume_data = []
        prev_close = None
        for idx, row in df_with_indicators.iterrows():
            timestamp = row.get('trade_date', row.get('timestamp', None))
            if timestamp is None:
                continue

            # 处理时间戳格式
            if hasattr(timestamp, 'timestamp'):
                ts = int(timestamp.timestamp())
            elif isinstance(timestamp, str):
                ts = int(datetime.fromisoformat(timestamp.replace('Z', '')).timestamp())
            else:
                continue

            close_price = float(row.get('close', 0))
            open_price = float(row.get('open', 0))
            volume = int(row.get('volume', 0))

            kline_data.append({
                "time": ts,
                "open": open_price,
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": close_price,
            })

            # 成交量数据（单独数组，根据涨跌着色）
            if prev_close is None:
                color = '#ef4444' if close_price >= open_price else '#10b981'
            else:
                color = '#ef4444' if close_price >= prev_close else '#10b981'
            volume_data.append({
                "time": ts,
                "value": volume,
                "color": color
            })
            prev_close = close_price

        # 提取均线数据
        ma_data = {
            "ma5": [],
            "ma10": [],
            "ma20": [],
            "ma60": []
        }

        # 列名映射（数据框中使用 ma_5 格式）
        ma_column_map = {
            "ma5": "ma_5",
            "ma10": "ma_10",
            "ma20": "ma_20",
            "ma60": "ma_60"
        }

        for _, row in df_with_indicators.iterrows():
            timestamp = row.get('trade_date', row.get('timestamp', None))
            if timestamp is None:
                continue

            if hasattr(timestamp, 'timestamp'):
                ts = int(timestamp.timestamp())
            elif isinstance(timestamp, str):
                ts = int(datetime.fromisoformat(timestamp.replace('Z', '')).timestamp())
            else:
                continue

            for ma_key, col_name in ma_column_map.items():
                val = row.get(col_name)
                if val is not None and not pd.isna(val):
                    ma_data[ma_key].append({"time": ts, "value": float(val)})

        # 获取最新价格信息
        latest = df_with_indicators.iloc[-1] if len(df_with_indicators) > 0 else {}
        prev_close = df_with_indicators.iloc[-2]['close'] if len(df_with_indicators) > 1 else latest.get('close', 0)
        current_price = float(latest.get('close', 0))
        change = current_price - float(prev_close) if prev_close else 0
        change_pct = (change / float(prev_close) * 100) if prev_close else 0

        result = {
            "symbol": symbol,
            "days": days,
            "kline": kline_data,
            "volume": volume_data,
            "ma": ma_data,
            "count": len(kline_data),
            # 实时价格信息
            "quote": {
                "price": current_price,
                "open": float(latest.get('open', 0)),
                "high": float(latest.get('high', 0)),
                "low": float(latest.get('low', 0)),
                "volume": int(latest.get('volume', 0)),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "prev_close": round(float(prev_close), 2) if prev_close else 0
            }
        }

        # 缓存结果
        set_cached_analysis(cache_key, result)
        return result
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取K线数据失败: {str(e)}")



@router.get("/index/{index_code}/data")
async def get_index_data(index_code: str, days: int = 120) -> List[Dict[str, Any]]:
    """
    获取指数历史数据

    Args:
        index_code: 指数代码 (如 000001=上证指数, 399001=深证成指)
        days: 获取天数
    """
    try:
        import akshare as ak

        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=days * 2)).strftime('%Y%m%d')

        # 使用 AkShare 获取指数数据
        df = ak.index_zh_a_hist(symbol=index_code, period="daily", start_date=start_date, end_date=end_date)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"无法获取指数 {index_code} 数据")

        # 只取最近 days 天
        df = df.tail(days)

        result = []
        for _, row in df.iterrows():
            date_val = row.get('日期', row.get('date', ''))
            if hasattr(date_val, 'strftime'):
                date_str = date_val.strftime('%Y-%m-%d')
            else:
                date_str = str(date_val)[:10]

            result.append({
                "date": date_str,
                "value": float(row.get('收盘', row.get('close', 0))),
            })

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get index data: {e}")
        raise HTTPException(status_code=500, detail=f"获取指数数据失败: {str(e)}")
