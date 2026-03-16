"""API routes for QuantTool web application."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import pandas as pd
from ..schemas.experiment import ExperimentRunSchema
from quanttool.application.backtest_service import BacktestService
from quanttool.application.factor_service import FactorService
from quanttool.application.data_service import DataService


router = APIRouter()


# ==================== Pydantic Models ====================

class AnalyzeRequest(BaseModel):
    """股票分析请求"""
    symbol: str
    days: int = 360


class ScanRequest(BaseModel):
    """扫描筛选请求"""
    market: str = "csi300"
    days: int = 360
    top_n: int = 10
    use_trend_score: bool = True
    use_breakout_score: bool = False
    use_momentum_score: bool = False


class EnhancedAnalyzeRequest(BaseModel):
    """增强分析请求"""
    symbol: str
    days: int = 360
    include_chip: bool = True
    include_patterns: bool = True
    include_strategies: bool = True


class BacktestRequest(BaseModel):
    """回测请求"""
    strategy_name: str = "ma_cross"
    symbols: List[str] = []
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    strategy_params: Dict[str, Any] = {}


# ==================== CLI 功能映射 ====================

@router.post("/analyze")
async def analyze_stock(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    股票分析 - 对应 CLI: quanttool analysis analyze <symbol>

    返回完整的技术分析报告，包括评分、因子、交易计划等
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        report = analyzer.analyze_stock(request.symbol, request.days)

        return {
            "symbol": request.symbol,
            "days": request.days,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")


@router.post("/analyze/enhanced")
async def analyze_stock_enhanced(request: EnhancedAnalyzeRequest) -> Dict[str, Any]:
    """
    增强版股票分析 - 对应 CLI: quanttool analysis enhanced <symbol>

    整合筹码分布、K线形态、策略信号
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        report = analyzer.analyze_stock_enhanced(
            request.symbol,
            request.days,
            include_chip=request.include_chip,
            include_talib_patterns=request.include_patterns,
            include_strategies=request.include_strategies
        )

        return {
            "symbol": request.symbol,
            "days": request.days,
            "report": report
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"增强分析失败: {str(e)}")


@router.post("/scan")
async def scan_stocks(request: ScanRequest) -> Dict[str, Any]:
    """
    股票扫描筛选 - 对应 CLI: quanttool analysis scan

    扫描市场寻找潜在机会
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.cli.commands.analysis_commands import (
            get_csi300_constituents,
            get_csi1000_constituents,
            analyze_stock_trend_score,
            analyze_stock_breakout_score,
            analyze_stock_momentum_score,
            analyze_stock_score
        )

        # 获取股票列表
        if request.market.lower() == "csi300":
            stock_list = get_csi300_constituents()
        elif request.market.lower() == "csi1000":
            stock_list = get_csi1000_constituents()
        else:
            raise HTTPException(status_code=400, detail=f"不支持的市场: {request.market}")

        analyzer = StockAnalyzer()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=request.days)

        results = []
        for stock_info in stock_list:
            symbol = stock_info['code'] if isinstance(stock_info, dict) else stock_info

            if request.use_momentum_score:
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
            "results": top_results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"扫描失败: {str(e)}")


@router.get("/scan/markets")
async def list_scan_markets() -> List[Dict[str, str]]:
    """列出可扫描的市场"""
    return [
        {"code": "csi300", "name": "沪深300", "description": "沪深300成分股"},
        {"code": "csi1000", "name": "中证1000", "description": "中证1000成分股"},
    ]


@router.get("/stock/{symbol}/info")
async def get_stock_info(symbol: str) -> Dict[str, Any]:
    """获取股票基本信息和最新数据"""
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
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
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        df = analyzer.get_stock_data(symbol, days)

        if df.empty:
            raise HTTPException(status_code=404, detail=f"未找到股票 {symbol} 的数据")

        # 计算技术指标
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # 转换为前端可用的格式
        kline_data = []
        for _, row in df_with_indicators.iterrows():
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

            kline_data.append({
                "time": ts,
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
                "volume": int(row.get('volume', 0)),
            })

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

        return {
            "symbol": symbol,
            "days": days,
            "kline": kline_data,
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
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取K线数据失败: {str(e)}")


@router.get("/stock/{symbol}/chip")
async def get_chip_distribution(symbol: str, days: int = 210) -> Dict[str, Any]:
    """
    获取筹码分布数据

    Args:
        symbol: 股票代码
        days: 计算周期，默认210天

    Returns:
        筹码分布数据，用于绘制类似东方财富的筹码图
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.factors.chip_distribution import ChipDistributionCalculator
        import numpy as np

        analyzer = StockAnalyzer()
        # 获取足够多的数据用于筹码计算
        df = analyzer.get_stock_data(symbol, max(days, 365))

        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"数据不足，无法计算筹码分布")

        # 计算筹码分布
        calculator = ChipDistributionCalculator(lookback_days=days)
        result = calculator.calculate(df)

        # 转换为前端可用的格式
        chip_data = []
        for i, (price, chip) in enumerate(zip(result.price_levels, result.chip_distribution)):
            if chip > 0.01:  # 只返回有效数据
                chip_data.append({
                    "price": round(float(price), 2),
                    "chip": round(float(chip), 4)
                })

        # 获取当前价格
        current_price = float(df.iloc[-1]['close'])

        # 计算90%成本区间
        sorted_indices = np.argsort(result.chip_distribution)[::-1]
        cumulative = 0
        cost_low = result.price_levels[0]
        cost_high = result.price_levels[-1]

        for idx in sorted_indices:
            cumulative += result.chip_distribution[idx]
            if cumulative >= 90:
                # 找到90%筹码的价格区间
                involved_indices = sorted_indices[:list(sorted_indices).index(idx)+1]
                cost_low = float(result.price_levels[min(involved_indices)])
                cost_high = float(result.price_levels[max(involved_indices)])
                break

        return {
            "symbol": symbol,
            "days": days,
            "chip_data": chip_data,
            "current_price": current_price,
            "avg_cost": round(result.avg_cost, 2),
            "cost_90_low": round(cost_low, 2),
            "cost_90_high": round(cost_high, 2),
            "metrics": {
                "concentration_ratio": round(result.concentration_ratio, 2),
                "avg_cost": round(result.avg_cost, 2),
                "profit_ratio": round(result.profit_ratio, 2),
                "upper_pressure": round(result.upper_pressure, 2),
                "lower_support": round(result.lower_support, 2),
                "score": round(result.score, 2)
            },
            "levels": {
                "support": [round(x, 2) for x in result.support_levels[:3]],
                "resistance": [round(x, 2) for x in result.resistance_levels[:3]],
                "peaks": [round(x, 2) for x in result.peak_prices[:3]]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"计算筹码分布失败: {str(e)}")


@router.get("/stock/{symbol}/signals")
async def get_technical_signals(symbol: str, days: int = 60) -> Dict[str, Any]:
    """
    获取技术指标异动信号

    包括：MACD金叉/死叉、RSI金叉/超卖、KDJ金叉/超卖、均线排列等

    Args:
        symbol: 股票代码
        days: 分析周期

    Returns:
        技术指标异动信号列表，包含时间戳用于图表标记
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        analyzer = StockAnalyzer()
        df = analyzer.get_stock_data(symbol, days)

        if df.empty or len(df) < 20:
            raise HTTPException(status_code=404, detail=f"数据不足")

        # 计算技术指标
        df = analyzer.calculate_technical_indicators(df)

        # 获取最新数据的时间戳
        def get_timestamp(row):
            ts = row.get('trade_date', row.get('timestamp', None))
            if ts is None:
                return None
            if hasattr(ts, 'timestamp'):
                return int(ts.timestamp())
            elif isinstance(ts, str):
                return int(datetime.fromisoformat(ts.replace('Z', '')).timestamp())
            return None

        signals = []
        markers = []  # 用于K线图标记
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        latest_ts = get_timestamp(latest)

        # ============ MACD 信号 ============
        # 列名：macd_dif (DIF), macd_dea (DEA/信号线), macd (柱状图)
        macd_dif = latest.get('macd_dif', latest.get('macd', 0))
        macd_dea = latest.get('macd_dea', latest.get('macd_signal', 0))
        macd_hist = latest.get('macd', latest.get('macd_hist', 0))

        prev_dif = prev.get('macd_dif', prev.get('macd', 0))
        prev_dea = prev.get('macd_dea', prev.get('macd_signal', 0))

        if prev_dif is not None and prev_dea is not None:
            # MACD 金叉：DIF从下向上穿越DEA
            if prev_dif <= prev_dea and macd_dif > macd_dea:
                signals.append({
                    "name": "MACD金叉",
                    "type": "bullish",
                    "description": "DIF上穿DEA，短期趋势转强",
                    "value": f"DIF:{macd_dif:.3f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "belowBar",
                    "color": "#26a69a",
                    "shape": "arrowUp",
                    "text": "MACD金叉"
                })
            # MACD 死叉：DIF从上向下穿越DEA
            elif prev_dif >= prev_dea and macd_dif < macd_dea:
                signals.append({
                    "name": "MACD死叉",
                    "type": "bearish",
                    "description": "DIF下穿DEA，短期趋势转弱",
                    "value": f"DIF:{macd_dif:.3f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "aboveBar",
                    "color": "#ef5350",
                    "shape": "arrowDown",
                    "text": "MACD死叉"
                })

        # MACD 柱状图方向
        if macd_hist is not None:
            prev_hist = prev.get('macd', prev.get('macd_hist', 0))
            if prev_hist is not None:
                if macd_hist > 0 and prev_hist <= 0:
                    signals.append({
                        "name": "MACD转多",
                        "type": "bullish",
                        "description": "MACD柱状图由负转正",
                        "value": f"柱:{macd_hist:.3f}",
                        "time": latest_ts
                    })
                elif macd_hist < 0 and prev_hist >= 0:
                    signals.append({
                        "name": "MACD转空",
                        "type": "bearish",
                        "description": "MACD柱状图由正转负",
                        "value": f"柱:{macd_hist:.3f}",
                        "time": latest_ts
                    })

        # ============ RSI 信号 ============
        rsi = latest.get('rsi_24', latest.get('rsi_14', 50))
        prev_rsi = prev.get('rsi_24', prev.get('rsi_14', 50))

        if rsi is not None:
            # RSI 超卖
            if rsi < 30:
                signals.append({
                    "name": "RSI超卖",
                    "type": "bullish",
                    "description": f"RSI={rsi:.1f}，超卖区可能反弹",
                    "value": f"RSI:{rsi:.1f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "belowBar",
                    "color": "#2196F3",
                    "shape": "circle",
                    "text": "RSI超卖"
                })
            # RSI 超买
            elif rsi > 70:
                signals.append({
                    "name": "RSI超买",
                    "type": "bearish",
                    "description": f"RSI={rsi:.1f}，超买区注意回调",
                    "value": f"RSI:{rsi:.1f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "aboveBar",
                    "color": "#FF9800",
                    "shape": "circle",
                    "text": "RSI超买"
                })

            # RSI 金叉（从超卖区回升）
            if prev_rsi is not None and prev_rsi < 30 and rsi >= 30:
                signals.append({
                    "name": "RSI回升",
                    "type": "bullish",
                    "description": "RSI脱离超卖区",
                    "value": f"{prev_rsi:.1f}→{rsi:.1f}",
                    "time": latest_ts
                })

        # ============ KDJ 信号 ============
        kdj_k = latest.get('kdj_k', 0)
        kdj_d = latest.get('kdj_d', 0)
        kdj_j = latest.get('kdj_j', 0)

        prev_k = prev.get('kdj_k', 0)
        prev_d = prev.get('kdj_d', 0)

        if kdj_k is not None and kdj_d is not None:
            # KDJ 金叉
            if prev_k is not None and prev_d is not None:
                if prev_k <= prev_d and kdj_k > kdj_d:
                    signals.append({
                        "name": "KDJ金叉",
                        "type": "bullish",
                        "description": f"K({kdj_k:.1f})上穿D({kdj_d:.1f})",
                        "value": f"J:{kdj_j:.1f}",
                        "time": latest_ts
                    })
                    markers.append({
                        "time": latest_ts,
                        "position": "belowBar",
                        "color": "#9C27B0",
                        "shape": "arrowUp",
                        "text": "KDJ金叉"
                    })
                elif prev_k >= prev_d and kdj_k < kdj_d:
                    signals.append({
                        "name": "KDJ死叉",
                        "type": "bearish",
                        "description": f"K({kdj_k:.1f})下穿D({kdj_d:.1f})",
                        "value": f"J:{kdj_j:.1f}",
                        "time": latest_ts
                    })
                    markers.append({
                        "time": latest_ts,
                        "position": "aboveBar",
                        "color": "#E91E63",
                        "shape": "arrowDown",
                        "text": "KDJ死叉"
                    })

            # KDJ 超卖
            if kdj_j is not None and kdj_j < 20:
                signals.append({
                    "name": "KDJ超卖",
                    "type": "bullish",
                    "description": f"J值={kdj_j:.1f}，严重超卖",
                    "value": f"J:{kdj_j:.1f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "belowBar",
                    "color": "#9C27B0",
                    "shape": "circle",
                    "text": "KDJ超卖"
                })
            elif kdj_j is not None and kdj_j > 100:
                signals.append({
                    "name": "KDJ超买",
                    "type": "bearish",
                    "description": f"J值={kdj_j:.1f}，严重超买",
                    "value": f"J:{kdj_j:.1f}",
                    "time": latest_ts
                })

        # ============ 均线排列 ============
        ma5 = latest.get('ma_5', 0)
        ma10 = latest.get('ma_10', 0)
        ma20 = latest.get('ma_20', 0)

        if ma5 and ma10 and ma20:
            # 多头排列：MA5 > MA10 > MA20
            if ma5 > ma10 > ma20:
                signals.append({
                    "name": "均线多头",
                    "type": "bullish",
                    "description": "MA5>MA10>MA20，多头排列",
                    "value": f"5:{ma5:.2f}",
                    "time": latest_ts
                })
            # 空头排列：MA5 < MA10 < MA20
            elif ma5 < ma10 < ma20:
                signals.append({
                    "name": "均线空头",
                    "type": "bearish",
                    "description": "MA5<MA10<MA20，空头排列",
                    "value": f"5:{ma5:.2f}",
                    "time": latest_ts
                })

        # ============ 布林带信号 ============
        close = latest.get('close', 0)
        boll_upper = latest.get('boll_upper', 0)
        boll_lower = latest.get('boll_lower', 0)

        if close and boll_upper and boll_lower:
            if close >= boll_upper:
                signals.append({
                    "name": "突破布林上轨",
                    "type": "neutral",
                    "description": "股价突破布林上轨，强势或超买",
                    "value": f"{close:.2f}",
                    "time": latest_ts
                })
            elif close <= boll_lower:
                signals.append({
                    "name": "跌破布林下轨",
                    "type": "bullish",
                    "description": "股价跌破布林下轨，可能反弹",
                    "value": f"{close:.2f}",
                    "time": latest_ts
                })
                markers.append({
                    "time": latest_ts,
                    "position": "belowBar",
                    "color": "#4CAF50",
                    "shape": "circle",
                    "text": "布林下轨"
                })

        # ============ DMI 趋势信号 ============
        pdi = latest.get('dmi_pdi', latest.get('pdi', 0))
        mdi = latest.get('dmi_mdi', latest.get('mdi', 0))
        adx = latest.get('dmi_adx', latest.get('adx', 0))

        if pdi is not None and mdi is not None:
            if pdi > mdi and adx and adx > 25:
                signals.append({
                    "name": "DMI多头趋势",
                    "type": "bullish",
                    "description": f"PDI>{mdi:.1f}且ADX={adx:.1f}，多头趋势强",
                    "value": f"PDI:{pdi:.1f}",
                    "time": latest_ts
                })
            elif mdi > pdi and adx and adx > 25:
                signals.append({
                    "name": "DMI空头趋势",
                    "type": "bearish",
                    "description": f"MDI>{pdi:.1f}且ADX={adx:.1f}，空头趋势强",
                    "value": f"MDI:{mdi:.1f}",
                    "time": latest_ts
                })

        return {
            "symbol": symbol,
            "signals": signals,
            "markers": markers,  # K线图标记
            "signal_count": len(signals),
            "bullish_count": len([s for s in signals if s["type"] == "bullish"]),
            "bearish_count": len([s for s in signals if s["type"] == "bearish"]),
            "latest_price": float(latest.get('close', 0)),
            "latest_change": float(latest.get('close', 0) - prev.get('close', 0)) if prev.get('close') else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取信号失败: {str(e)}")


# ==================== 回测 API ====================

@router.get("/backtest/strategies")
async def list_backtest_strategies() -> List[Dict[str, Any]]:
    """列出可用的回测策略"""
    return [
        {
            "name": "ma_cross",
            "display_name": "均线交叉策略",
            "description": "短期均线上穿长期均线买入，下穿卖出",
            "category": "traditional",
            "params": {
                "short_window": {"type": "int", "default": 10, "description": "短期均线周期"},
                "long_window": {"type": "int", "default": 30, "description": "长期均线周期"}
            }
        },
        {
            "name": "breakout",
            "display_name": "突破策略",
            "description": "价格突破N日高点买入，跌破N日低点卖出",
            "category": "traditional",
            "params": {
                "period": {"type": "int", "default": 20, "description": "突破周期"}
            }
        },
        {
            "name": "score",
            "display_name": "评分策略",
            "description": "首次突破策略：评分首次突破阈值时买入/卖出。买入=80,卖出=60为最优参数",
            "category": "traditional",
            "params": {
                "buy_threshold": {"type": "float", "default": 80.0, "description": "买入评分阈值（首次突破触发）"},
                "sell_threshold": {"type": "float", "default": 60.0, "description": "卖出评分阈值（首次跌破触发）"}
            }
        },
        {
            "name": "enhanced_score",
            "display_name": "增强评分策略",
            "description": "首次突破+动态权重+风险控制。评分首次突破80买入，首次跌破60卖出",
            "category": "traditional",
            "params": {
                "buy_threshold": {"type": "float", "default": 80.0, "description": "买入评分阈值"},
                "sell_threshold": {"type": "float", "default": 60.0, "description": "卖出评分阈值"},
                "use_dynamic_weights": {"type": "bool", "default": True, "description": "使用动态权重"},
                "use_risk_control": {"type": "bool", "default": True, "description": "使用风险控制"}
            }
        },
        {
            "name": "dual_ma",
            "display_name": "双均线策略",
            "description": "经典双均线交叉策略，支持多周期组合",
            "category": "traditional",
            "params": {
                "fast_period": {"type": "int", "default": 5, "description": "快线周期"},
                "slow_period": {"type": "int", "default": 20, "description": "慢线周期"}
            }
        },
        {
            "name": "macd",
            "display_name": "MACD策略",
            "description": "基于MACD指标的金叉死叉信号",
            "category": "traditional",
            "params": {
                "fast_period": {"type": "int", "default": 12, "description": "快线周期"},
                "slow_period": {"type": "int", "default": 26, "description": "慢线周期"},
                "signal_period": {"type": "int", "default": 9, "description": "信号线周期"}
            }
        },
        {
            "name": "rsi",
            "display_name": "RSI策略",
            "description": "基于RSI超买超卖信号",
            "category": "traditional",
            "params": {
                "period": {"type": "int", "default": 14, "description": "RSI周期"},
                "oversold": {"type": "int", "default": 30, "description": "超卖阈值"},
                "overbought": {"type": "int", "default": 70, "description": "超买阈值"}
            }
        },
        {
            "name": "kdj",
            "display_name": "KDJ策略",
            "description": "基于KDJ指标的金叉死叉信号",
            "category": "traditional",
            "params": {
                "n": {"type": "int", "default": 9, "description": "KDJ周期"},
                "m1": {"type": "int", "default": 3, "description": "K平滑周期"},
                "m2": {"type": "int", "default": 3, "description": "D平滑周期"}
            }
        },
        {
            "name": "bollinger",
            "display_name": "布林带策略",
            "description": "基于布林带上下轨突破信号",
            "category": "traditional",
            "params": {
                "period": {"type": "int", "default": 20, "description": "布林带周期"},
                "std_dev": {"type": "float", "default": 2.0, "description": "标准差倍数"}
            }
        },
        {
            "name": "turtle",
            "display_name": "海龟交易策略",
            "description": "经典海龟交易系统，基于通道突破",
            "category": "traditional",
            "params": {
                "entry_period": {"type": "int", "default": 20, "description": "入场周期"},
                "exit_period": {"type": "int", "default": 10, "description": "出场周期"}
            }
        }
    ]


# ==================== Qlib ML 模型 API ====================

@router.get("/qlib/models")
async def list_qlib_models() -> List[Dict[str, Any]]:
    """
    列出可用的 Qlib ML 模型

    包括 21 种 Qlib 原生模型：
    - GBDT 系列: LightGBM, XGBoost, CatBoost, DoubleEnsemble
    - PyTorch 序列: LSTM, GRU, ALSTM, Transformer, TCN, Localformer
    - PyTorch 高级: GATs, SFM, TabNet, ADARNN, ADD, HIST, IGMTF, KRNN, TRA, TCTS, Sandwich
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
            'pytorch_sequence': {
                'hidden_size': {'type': 'int', 'default': 64, 'description': '隐藏层大小'},
                'num_layers': {'type': 'int', 'default': 2, 'description': '层数'},
                'dropout': {'type': 'float', 'default': 0.1, 'description': 'Dropout率'},
                'epochs': {'type': 'int', 'default': 100, 'description': '训练轮数'},
                'batch_size': {'type': 'int', 'default': 256, 'description': '批大小'},
            },
            'pytorch_advanced': {
                'hidden_size': {'type': 'int', 'default': 64, 'description': '隐藏层大小'},
                'num_layers': {'type': 'int', 'default': 2, 'description': '层数'},
                'dropout': {'type': 'float', 'default': 0.1, 'description': 'Dropout率'},
                'epochs': {'type': 'int', 'default': 100, 'description': '训练轮数'},
            }
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


class QlibTrainRequest(BaseModel):
    """Qlib 模型训练请求"""
    model_type: str = "lgb"
    symbols: List[str] = []
    start_date: str = "2023-01-01"
    end_date: str = "2024-12-31"
    features: List[str] = ["close", "volume", "ma_5", "ma_10", "ma_20"]
    label: str = "return_5d"  # 预测目标
    # 模型参数
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.01
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    epochs: int = 100
    batch_size: int = 256
    # 回测参数
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003  # 手续费率 0.03%
    slippage_rate: float = 0.0001   # 滑点率 0.01%


class QlibPredictRequest(BaseModel):
    """Qlib 模型预测请求"""
    model_type: str = "lgb"
    model_path: str = ""  # 已训练模型的路径
    symbols: List[str] = []
    features: List[str] = ["close", "volume", "ma_5", "ma_10", "ma_20"]
    # 回测参数
    predict_start_date: str = "2024-01-01"
    predict_end_date: str = "2024-12-31"
    initial_cash: float = 100000.0
    commission_rate: float = 0.0003
    slippage_rate: float = 0.0001


@router.post("/qlib/train")
async def train_qlib_model(request: QlibTrainRequest) -> Dict[str, Any]:
    """
    训练 Qlib ML 模型

    使用历史数据训练模型，返回模型路径和训练指标
    """
    try:
        from quanttool.strategies.qlib import create_model
        from quanttool.factors.stock_analyzer import StockAnalyzer
        import numpy as np
        import os
        import uuid

        # 获取训练数据
        analyzer = StockAnalyzer()
        all_data = []
        data_dates = {'start': None, 'end': None}

        for symbol in request.symbols:
            df = analyzer.get_stock_data(symbol, 500)
            if df.empty:
                continue

            # 记录数据日期范围
            if 'trade_date' in df.columns:
                dates = df['trade_date']
                if data_dates['start'] is None or dates.min() < data_dates['start']:
                    data_dates['start'] = dates.min()
                if data_dates['end'] is None or dates.max() > data_dates['end']:
                    data_dates['end'] = dates.max()

            # 计算技术指标
            df = analyzer.calculate_technical_indicators(df)

            # 计算标签（未来5日收益率）
            df['return_5d'] = df['close'].pct_change(5).shift(-5)

            # 选择特征
            available_features = [f for f in request.features if f in df.columns]
            if not available_features:
                continue

            df['symbol'] = symbol
            all_data.append(df[available_features + ['return_5d', 'symbol']].dropna())

        if not all_data:
            raise HTTPException(status_code=400, detail="无法获取足够的训练数据")

        # 合并数据
        train_df = pd.concat(all_data, ignore_index=True)

        # 准备特征和标签
        feature_cols = request.features
        X = train_df[feature_cols].values
        y = train_df['return_5d'].values

        # 过滤无效值
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]

        if len(X) < 100:
            raise HTTPException(status_code=400, detail=f"训练数据不足: {len(X)} 条")

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
        X_df = pd.DataFrame(X, columns=feature_cols)
        y_series = pd.Series(y)
        model.fit(X_df, y_series)
        model.feature_names_ = feature_cols

        # 保存模型
        model_dir = "models/qlib"
        os.makedirs(model_dir, exist_ok=True)
        model_id = str(uuid.uuid4())[:8]
        model_path = f"{model_dir}/{request.model_type}_{model_id}.pkl"
        model.save(model_path)

        # 评估
        predictions = model.predict(X_df)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        ic = np.corrcoef(predictions, y)[0, 1] if len(predictions) > 1 else 0

        # 格式化日期
        start_date_str = str(data_dates['start'])[:10] if data_dates['start'] else request.start_date
        end_date_str = str(data_dates['end'])[:10] if data_dates['end'] else request.end_date

        return {
            "model_id": model_id,
            "model_type": request.model_type,
            "model_path": model_path,
            "symbols": request.symbols,
            "train_samples": len(X),
            "features": feature_cols,
            "data_period": {
                "start_date": start_date_str,
                "end_date": end_date_str,
                "requested_start": request.start_date,
                "requested_end": request.end_date,
            },
            "metrics": {
                "mse": round(float(mse), 6),
                "mae": round(float(mae), 6),
                "ic": round(float(ic), 4) if not np.isnan(ic) else 0,
            },
            "backtest_params": {
                "initial_cash": request.initial_cash,
                "commission_rate": request.commission_rate,
                "slippage_rate": request.slippage_rate,
                "t_plus_1": True,  # A股 T+1 交易
            },
            "feature_importance": model.get_feature_importance().to_dict('records') if hasattr(model, 'get_feature_importance') else []
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"模型训练失败: {str(e)}")


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

        # 直接加载保存的模型文件
        if not request.model_path:
            raise HTTPException(status_code=400, detail="请提供已训练模型的路径")

        saved_data = joblib.load(request.model_path)
        model = saved_data.get('model')
        feature_names = saved_data.get('feature_names', request.features)

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

        # 获取预测数据
        analyzer = StockAnalyzer()
        predictions = {}

        # 解析回测日期
        predict_start = datetime.fromisoformat(request.predict_start_date)
        predict_end = datetime.fromisoformat(request.predict_end_date)

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

        for symbol in request.symbols:
            df = analyzer.get_stock_data(symbol, 500)  # 获取更多数据用于回测
            if df.empty:
                continue

            df = analyzer.calculate_technical_indicators(df)

            # 使用模型期望的特征
            available_features = [f for f in feature_names if f in df.columns]
            if not available_features:
                continue

            # 确定日期列 (trade_date 或 timestamp)
            date_column = None
            if 'trade_date' in df.columns:
                date_column = 'trade_date'
            elif 'timestamp' in df.columns:
                date_column = 'timestamp'

            # 记录数据日期范围
            data_start = None
            data_end = None
            if date_column:
                dates = pd.to_datetime(df[date_column])
                data_start = str(dates.min())[:10]
                data_end = str(dates.max())[:10]

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

                # 生成信号 (阈值可调整)
                signal = "hold"
                if pred_value > 0.55:
                    signal = "buy"
                elif pred_value < 0.45:
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
                    "total_trades": len(trades),
                    "win_rate": round(backtest_results["win_trades"] / len(trades) * 100, 1) if trades else 0,
                    "total_commission": round(total_commission, 2),
                    "total_slippage": round(total_slippage, 2),
                    "trades": trades[-10:],  # 最近10笔交易
                }
            }

            backtest_results["total_trades"] += len(trades)

        return {
            "model_type": request.model_type,
            "model_path": request.model_path,
            "predict_period": {
                "start_date": request.predict_start_date,
                "end_date": request.predict_end_date,
            },
            "backtest_params": {
                "initial_cash": initial_cash,
                "commission_rate": commission_rate,
                "slippage_rate": slippage_rate,
                "t_plus_1": True,
            },
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
        },
        {
            "category": "pytorch_sequence",
            "display_name": "PyTorch 序列模型",
            "description": "深度学习序列模型，适合时间序列预测",
            "models": ["lstm", "gru", "alstm", "transformer", "tcn", "localformer"],
            "recommended": "lstm"
        },
        {
            "category": "pytorch_advanced",
            "display_name": "PyTorch 高级模型",
            "description": "前沿深度学习模型，适合复杂市场模式",
            "models": ["gats", "sfm", "tabnet", "adarnn", "add", "hist", "igmtf", "krnn", "tra", "tcts", "sandwich"],
            "recommended": "gats"
        }
    ]


@router.post("/backtest/run")
async def run_backtest(request: BacktestRequest) -> Dict[str, Any]:
    """
    运行回测

    支持的策略：
    - ma_cross: 均线交叉
    - breakout: 突破策略
    - trend_momentum: 趋势动量
    - adaptive_threshold: 自适应阈值
    """
    try:
        from quanttool.application.backtest_service import BacktestService

        # 解析日期
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)

        # 初始化回测服务
        backtest_service = BacktestService()

        # 运行回测
        result = backtest_service.run_backtest(
            strategy_name=request.strategy_name,
            strategy_params=request.strategy_params,
            symbols=request.symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe="1d",
            initial_cash=request.initial_cash,
            commission_rate=request.commission_rate,
            data_provider="enhanced_data_fetcher",
        )

        # 转换结果
        result_dict = {
            "strategy": request.strategy_name,
            "symbols": request.symbols,
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "initial_capital": result.initial_capital,
            "final_capital": result.final_capital,
            "total_return": result.total_return,
            "annual_return": result.annual_return,
            "max_drawdown": getattr(result, 'max_drawdown', 0),
            "sharpe_ratio": getattr(result, 'sharpe_ratio', 0),
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "trades": []
        }

        # 添加交易记录（如果有）
        if hasattr(result, 'trades') and result.trades:
            for trade in result.trades[:50]:  # 限制返回数量
                result_dict["trades"].append({
                    "symbol": getattr(trade, 'symbol', ''),
                    "action": getattr(trade, 'action', ''),
                    "price": getattr(trade, 'price', 0),
                    "shares": getattr(trade, 'shares', 0),
                    "timestamp": str(getattr(trade, 'timestamp', ''))
                })

        return result_dict

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"回测失败: {str(e)}")


# ==================== 原有 API ====================

@router.get("/experiments")
async def list_experiments(
    run_type: str = None, status: str = None
) -> List[Dict[str, Any]]:
    """List experiment runs with optional filtering."""
    from ..infrastructure.stores.meta_db import MetaDB

    db = MetaDB()
    runs = db.get_experiment_runs(run_type=run_type, status=status)

    return runs


@router.get("/backtest/runs/{run_id}")
async def get_backtest_result(run_id: str) -> Dict[str, Any]:
    """Get results for a specific backtest run."""
    from ..infrastructure.stores.meta_db import MetaDB

    db = MetaDB()
    run = db.get_experiment_run(run_id)

    if not run:
        raise HTTPException(status_code=404, detail=f"Backtest run {run_id} not found")

    return run


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
        from ..infrastructure.stores.meta_db import MetaDB
        import uuid

        db = MetaDB()
        run_id = str(uuid.uuid4())
        db.save_experiment_run(
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


@router.get("/data/providers")
async def list_data_providers() -> List[str]:
    """List available data providers."""
    from ..core.registry import registry, ComponentType

    providers = registry.list_available(ComponentType.DATA_PROVIDER)
    return providers


@router.get("/strategies")
async def list_strategies() -> List[str]:
    """List available strategies."""
    from ..core.registry import registry, ComponentType

    strategies = registry.list_available(ComponentType.STRATEGY)
    return strategies


@router.get("/factors")
async def list_factors() -> List[str]:
    """List available factors."""
    from ..core.registry import registry, ComponentType

    factors = registry.list_available(ComponentType.FACTOR)
    return factors
