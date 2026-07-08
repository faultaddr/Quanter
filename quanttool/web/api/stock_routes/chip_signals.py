"""Stock chip distribution and technical signal API routes."""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from ....core.logging import get_logger
from ..utils import get_cached_analysis, set_cached_analysis


logger = get_logger(__name__)
router = APIRouter()

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

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
        # 获取足够多的数据用于筹码计算
        fetch_days = max(days, 365)
        df = analyzer.get_stock_data(symbol, fetch_days)

        if df.empty or len(df) < 10:
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
    # 检查缓存
    cache_key = f"signals_{symbol}_{days}"
    cached = get_cached_analysis(cache_key)
    if cached:
        return cached

    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
        # 增加获取天数以确保有足够数据
        fetch_days = max(days, 60)
        df = analyzer.get_stock_data(symbol, fetch_days)

        if df.empty or len(df) < 10:
            raise HTTPException(status_code=404, detail=f"数据不足，请稍后重试")

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

        result = {
            "symbol": symbol,
            "signals": signals,
            "markers": markers,  # K线图标记
            "signal_count": len(signals),
            "bullish_count": len([s for s in signals if s["type"] == "bullish"]),
            "bearish_count": len([s for s in signals if s["type"] == "bearish"]),
            "latest_price": float(latest.get('close', 0)),
            "latest_change": float(latest.get('close', 0) - prev.get('close', 0)) if prev.get('close') else 0
        }

        # 缓存结果
        set_cached_analysis(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取信号失败: {str(e)}")

