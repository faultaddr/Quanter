"""Stock analysis API routes."""

from datetime import datetime
from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, HTTPException

from ....core.logging import get_logger
from ...schemas.stock import AnalyzeRequest, EnhancedAnalyzeRequest
from ..utils import get_cached_analysis, set_cached_analysis


logger = get_logger(__name__)
router = APIRouter()

@router.post("/analyze")
async def analyze_stock(request: AnalyzeRequest) -> Dict[str, Any]:
    """
    股票分析 - 对应 CLI: quanttool analysis analyze <symbol>

    返回完整的技术分析报告，包括评分、因子、交易计划等
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        # 使用实时价格数据，避免 qlib 复权价格显示异常
        analyzer = StockAnalyzer(use_realtime_price=True)
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
    返回完整的数据结构供前端展示
    """
    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer
        from quanttool.factors.chip_distribution import ChipDistributionCalculator

        # 使用实时数据源（优先备用数据源获取实时股价）
        analyzer = StockAnalyzer(use_realtime_price=True)

        # 标准化股票代码并显示
        normalized_symbol = analyzer._normalize_symbol(request.symbol)
        logger.info(f"分析股票: {request.symbol} -> {normalized_symbol}")

        # 获取股票数据（实时股价）
        df = analyzer.get_stock_data(request.symbol, request.days)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"无法获取 {request.symbol} 的数据")

        # 计算技术指标
        df_with_indicators = analyzer.calculate_technical_indicators(df)

        # 1. 生成 K 线数据和技术指标
        kline_data = []
        macd_data = []
        kdj_data = []
        rsi_data = []
        volume_data = []
        amount_data = []  # 成交额数据
        ma_data = {"ma5": [], "ma10": [], "ma20": [], "ma60": []}

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

            # K线数据
            kline_data.append({
                "time": ts,
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
            })

            # 成交量
            volume = int(row.get('volume', 0))
            volume_data.append({
                "time": ts,
                "value": volume,
                "color": "#ef4444" if row.get('close', 0) >= row.get('open', 0) else "#10b981"
            })

            # 成交额 = 成交量 * 成交均价 (vwap 或 close)
            vwap = row.get('vwap', row.get('close', 0))
            amount = volume * float(vwap) if vwap else 0
            amount_data.append({
                "time": ts,
                "value": amount,
                "color": "#3b82f6"  # 蓝色
            })

            # 均线数据
            for ma_key, col_name in [("ma5", "ma_5"), ("ma10", "ma_10"), ("ma20", "ma_20"), ("ma60", "ma_60")]:
                val = row.get(col_name)
                if val is not None and not pd.isna(val):
                    ma_data[ma_key].append({"time": ts, "value": float(val)})

            # MACD 数据
            macd_dif = row.get('macd_dif', row.get('macd', None))
            macd_dea = row.get('macd_dea', row.get('macd_signal', None))
            if macd_dif is not None and macd_dea is not None:
                macd_hist = float(macd_dif) - float(macd_dea)
                macd_data.append({
                    "time": ts,
                    "dif": float(macd_dif) if macd_dif else 0,
                    "dea": float(macd_dea) if macd_dea else 0,
                    "hist": macd_hist,
                })

            # KDJ 数据
            k_val = row.get('kdj_k', row.get('k_value', None))
            d_val = row.get('kdj_d', row.get('d_value', None))
            j_val = row.get('kdj_j', row.get('j_value', None))
            if k_val is not None and d_val is not None:
                kdj_data.append({
                    "time": ts,
                    "k": float(k_val) if k_val else 0,
                    "d": float(d_val) if d_val else 0,
                    "j": float(j_val) if j_val else 0,
                })

            # RSI 数据
            rsi_val = row.get('rsi_14', row.get('rsi_12', row.get('rsi_6', None)))
            if rsi_val is not None:
                rsi_data.append({
                    "time": ts,
                    "value": float(rsi_val) if rsi_val else 50,
                })

        # 2. 获取行情数据
        latest = df_with_indicators.iloc[-1] if len(df_with_indicators) > 0 else {}
        prev_close = df_with_indicators.iloc[-2]['close'] if len(df_with_indicators) > 1 else latest.get('close', 0)
        current_price = float(latest.get('close', 0))
        change = current_price - float(prev_close) if prev_close else 0
        change_pct = (change / float(prev_close) * 100) if prev_close else 0

        quote = {
            "symbol": request.symbol,
            "close": current_price,
            "open": float(latest.get('open', 0)),
            "high": float(latest.get('high', 0)),
            "low": float(latest.get('low', 0)),
            "volume": int(latest.get('volume', 0)),
            "change": change,
            "change_pct": change_pct,
            "date": str(latest.get('trade_date', latest.get('timestamp', ''))),
        }

        # 3. 生成信号
        signals = {"bullish": [], "bearish": []}

        # 均线信号
        ma_5 = latest.get('ma_5', 0)
        ma_10 = latest.get('ma_10', 0)
        ma_20 = latest.get('ma_20', 0)
        if ma_5 and ma_10 and ma_20:
            if ma_5 > ma_10 > ma_20:
                signals["bullish"].append({"name": "均线多头排列", "description": f"MA5({ma_5:.2f}) > MA10({ma_10:.2f}) > MA20({ma_20:.2f})"})
            elif ma_5 < ma_10 < ma_20:
                signals["bearish"].append({"name": "均线空头排列", "description": f"MA5({ma_5:.2f}) < MA10({ma_10:.2f}) < MA20({ma_20:.2f})"})
            # 金叉/死叉信号
            prev_ma_5 = df_with_indicators.iloc[-2].get('ma_5', 0)
            prev_ma_10 = df_with_indicators.iloc[-2].get('ma_10', 0)
            if prev_ma_5 and prev_ma_10:
                if prev_ma_5 <= prev_ma_10 and ma_5 > ma_10:
                    signals["bullish"].append({"name": "MA5金叉MA10", "description": f"MA5({ma_5:.2f}) 上穿 MA10({ma_10:.2f})"})
                elif prev_ma_5 >= prev_ma_10 and ma_5 < ma_10:
                    signals["bearish"].append({"name": "MA5死叉MA10", "description": f"MA5({ma_5:.2f}) 下穿 MA10({ma_10:.2f})"})

        # MACD 信号
        macd_dif = latest.get('macd_dif', latest.get('macd', 0))
        macd_dea = latest.get('macd_dea', latest.get('macd_signal', 0))
        macd_hist = macd_dif - macd_dea if macd_dif and macd_dea else 0

        if macd_hist > 0:
            signals["bullish"].append({"name": "MACD多头", "description": f"MACD柱状图为正 ({macd_hist:.4f})"})
        elif macd_hist < 0:
            signals["bearish"].append({"name": "MACD空头", "description": f"MACD柱状图为负 ({macd_hist:.4f})"})

        # RSI 信号
        rsi = latest.get('rsi_14', latest.get('rsi_12', latest.get('rsi_6', 50)))
        if rsi:
            if rsi > 70:
                signals["bearish"].append({"name": "RSI超买", "description": f"RSI = {rsi:.1f}，超买区域"})
            elif rsi < 30:
                signals["bullish"].append({"name": "RSI超卖", "description": f"RSI = {rsi:.1f}，超卖区域"})
            elif rsi > 50:
                signals["bullish"].append({"name": "RSI偏强", "description": f"RSI = {rsi:.1f}，位于强势区域"})

        # 布林带信号
        boll_upper = latest.get('boll_upper', 0)
        boll_lower = latest.get('boll_lower', 0)
        boll_mid = latest.get('boll_mid', latest.get('ma_20', 0))
        if boll_upper and boll_lower:
            if current_price > boll_upper:
                signals["bearish"].append({"name": "突破布林上轨", "description": f"价格 {current_price:.2f} > 上轨 {boll_upper:.2f}"})
            elif current_price < boll_lower:
                signals["bullish"].append({"name": "跌破布林下轨", "description": f"价格 {current_price:.2f} < 下轨 {boll_lower:.2f}"})
            elif current_price > boll_mid:
                signals["bullish"].append({"name": "价格在中轨上方", "description": f"价格 {current_price:.2f} > 中轨 {boll_mid:.2f}"})

        # 4. 计算筹码分布
        chip_data = None
        chip_metrics = None
        if request.include_chip and len(df) >= 60:
            try:
                chip_calc = ChipDistributionCalculator()
                chip_result = chip_calc.calculate(df)
                if chip_result:
                    chip_data = []
                    for price, chip in zip(chip_result.price_levels, chip_result.chip_distribution):
                        if chip > 0.01:
                            chip_data.append({
                                "price": round(float(price), 2),
                                "chip": round(float(chip), 4)
                            })

                    # 计算90%成本区间
                    import numpy as np
                    sorted_indices = np.argsort(chip_result.chip_distribution)[::-1]
                    cumulative = 0
                    cost_90_low = float(chip_result.price_levels[0])
                    cost_90_high = float(chip_result.price_levels[-1])

                    for idx in sorted_indices:
                        cumulative += chip_result.chip_distribution[idx]
                        if cumulative >= 90:
                            involved_indices = sorted_indices[:list(sorted_indices).index(idx)+1]
                            cost_90_low = float(chip_result.price_levels[min(involved_indices)])
                            cost_90_high = float(chip_result.price_levels[max(involved_indices)])
                            break

                    chip_metrics = {
                        "profit_ratio": float(chip_result.profit_ratio) / 100 if chip_result.profit_ratio > 1 else float(chip_result.profit_ratio),
                        "avg_cost": float(chip_result.avg_cost) if hasattr(chip_result, 'avg_cost') else current_price,
                        "concentration": float(chip_result.concentration_ratio) / 100 if hasattr(chip_result, 'concentration_ratio') else 0.5,
                        "score": float(chip_result.score) if hasattr(chip_result, 'score') else 50,
                        "cost_90_low": round(cost_90_low, 2),
                        "cost_90_high": round(cost_90_high, 2),
                        "upper_pressure": float(chip_result.upper_pressure) if hasattr(chip_result, 'upper_pressure') else 0,
                        "lower_support": float(chip_result.lower_support) if hasattr(chip_result, 'lower_support') else 0,
                    }
            except Exception as e:
                logger.warning(f"筹码分布计算失败: {e}")

        # 5. 生成分析报告（传递预计算数据避免重复获取）
        report = analyzer.analyze_stock_enhanced(
            request.symbol,
            request.days,
            include_chip=request.include_chip,
            include_talib_patterns=request.include_patterns,
            include_strategies=request.include_strategies,
            precomputed_data={'df': df, 'df_with_indicators': df_with_indicators},
            fast_mode=True  # 快速模式，跳过耗时的市场检测和趋势评分
        )

        return {
            "symbol": request.symbol,
            "normalized_symbol": normalized_symbol,
            "name": "",  # 可以从数据中获取
            "days": request.days,
            "kline": kline_data,
            "volume": volume_data,
            "amount": amount_data,
            "ma": ma_data,
            "macd": macd_data,
            "kdj": kdj_data,
            "rsi": rsi_data,
            "quote": quote,
            "signals": signals,
            "chip": chip_data,
            "chip_metrics": chip_metrics,
            "report": report
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"增强分析失败: {str(e)}")



@router.get("/stock/{symbol}/analysis")
async def get_stock_analysis(symbol: str, days: int = 120) -> Dict[str, Any]:
    """
    获取股票完整分析数据（组合接口）

    一次性返回 K线、筹码、信号等所有分析数据
    """
    # 检查缓存
    cache_key = f"analysis_{symbol}_{days}"
    cached = get_cached_analysis(cache_key)
    if cached:
        return cached

    try:
        from quanttool.factors.stock_analyzer import StockAnalyzer

        # 使用与 kline 接口相同的数据获取方式
        analyzer = StockAnalyzer(use_realtime_price=True)
        # 增加获取天数以确保有足够数据
        fetch_days = max(days, 60)
        df = analyzer.get_stock_data(symbol, fetch_days)

        if df.empty or len(df) < 10:
            raise HTTPException(status_code=404, detail=f"数据不足，请稍后重试")

        # 计算技术指标
        df = analyzer.calculate_technical_indicators(df)

        # K线数据
        kline = []
        for _, row in df.iterrows():
            ts = row.get('trade_date', row.get('timestamp', None))
            if ts is not None:
                if hasattr(ts, 'strftime'):
                    date_str = ts.strftime('%Y-%m-%d')
                else:
                    date_str = str(ts)[:10]
            else:
                date_str = ""

            kline.append({
                "date": date_str,
                "open": float(row.get('open', 0)),
                "high": float(row.get('high', 0)),
                "low": float(row.get('low', 0)),
                "close": float(row.get('close', 0)),
                "volume": float(row.get('volume', 0)),
                "amount": float(row.get('amount', 0)),
            })

        # 获取股票名称
        name = symbol
        try:
            info = analyzer.get_stock_info(symbol)
            if info:
                name = info.get('name', symbol)
        except Exception:
            pass

        # 筹码分布（简化版本 - 基于价格区间）
        chip = []
        try:
            import numpy as np
            prices = df['close'].dropna().values
            if len(prices) > 0:
                min_price = prices.min()
                max_price = prices.max()
                price_range = max_price - min_price
                if price_range > 0:
                    # 分成20个价格区间
                    bins = np.linspace(min_price, max_price, 21)
                    hist, _ = np.histogram(prices, bins=bins)
                    total = hist.sum()
                    if total > 0:
                        for i, count in enumerate(hist):
                            chip.append({
                                "price": float((bins[i] + bins[i+1]) / 2),
                                "percent": float(count / total),
                            })
        except Exception as e:
            logger.warning(f"计算筹码分布失败: {e}")

        # 技术指标
        indicators = {
            "macd": {
                "dif": df['macd'].dropna().tolist()[-60:] if 'macd' in df.columns else [],
                "dea": df['macd_signal'].dropna().tolist()[-60:] if 'macd_signal' in df.columns else [],
                "macd": df['macd_hist'].dropna().tolist()[-60:] if 'macd_hist' in df.columns else [],
            },
            "kdj": {
                "k": df['kdj_k'].dropna().tolist()[-60:] if 'kdj_k' in df.columns else [],
                "d": df['kdj_d'].dropna().tolist()[-60:] if 'kdj_d' in df.columns else [],
                "j": df['kdj_j'].dropna().tolist()[-60:] if 'kdj_j' in df.columns else [],
            },
            "rsi": {
                "rsi6": df['rsi_6'].dropna().tolist()[-60:] if 'rsi_6' in df.columns else [],
                "rsi12": df['rsi_12'].dropna().tolist()[-60:] if 'rsi_12' in df.columns else [],
                "rsi24": df['rsi_24'].dropna().tolist()[-60:] if 'rsi_24' in df.columns else [],
            },
            "ma": {
                "ma5": df['ma_5'].dropna().tolist()[-60:] if 'ma_5' in df.columns else [],
                "ma10": df['ma_10'].dropna().tolist()[-60:] if 'ma_10' in df.columns else [],
                "ma20": df['ma_20'].dropna().tolist()[-60:] if 'ma_20' in df.columns else [],
                "ma60": df['ma_60'].dropna().tolist()[-60:] if 'ma_60' in df.columns else [],
            },
        }

        # 简单信号检测
        signals = []
        latest = df.iloc[-1]
        macd = latest.get('macd', 0)
        macd_signal_val = latest.get('macd_signal', 0)
        if macd is not None and macd_signal_val is not None:
            if macd > macd_signal_val and df.iloc[-2].get('macd', 0) <= df.iloc[-2].get('macd_signal', 0):
                signals.append({"name": "MACD金叉", "type": "buy", "description": "MACD金叉买入信号"})
            elif macd < macd_signal_val and df.iloc[-2].get('macd', 0) >= df.iloc[-2].get('macd_signal', 0):
                signals.append({"name": "MACD死叉", "type": "sell", "description": "MACD死叉卖出信号"})

        rsi = latest.get('rsi_14', latest.get('rsi_6', 0))
        if rsi is not None:
            if rsi < 30:
                signals.append({"name": "RSI超卖", "type": "buy", "description": f"RSI={rsi:.1f}，超卖区间"})
            elif rsi > 70:
                signals.append({"name": "RSI超买", "type": "sell", "description": f"RSI={rsi:.1f}，超买区间"})

        result = {
            "symbol": symbol,
            "name": name,
            "kline": kline,
            "chip": chip,
            "signals": signals,
            "indicators": indicators,
            "latest_price": float(latest.get('close', 0)),
            "latest_change_pct": float(latest.get('pct_chg', 0)),
        }

        # 缓存结果
        set_cached_analysis(cache_key, result)
        return result

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"获取分析数据失败: {str(e)}")

