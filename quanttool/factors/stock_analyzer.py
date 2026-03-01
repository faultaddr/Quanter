"""
Stock Analysis Module - Performs comprehensive technical analysis and generates buy/sell signals
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from typing import Dict, List, Tuple

# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from quanttool.factors.tech_indicators import *
from quanttool.factors.trading_strategies import TradingStrategies
from quanttool.factors.scoring_system import ScoringSystem
from quanttool.factors.candlestick_patterns import (
    CandlestickPatternRecognizer, get_pattern_assessment, analyze_candlestick_patterns,
    draw_candlestick_chart, draw_pattern_illustration
)
from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials


class StockAnalyzer:
    """
    A comprehensive stock analyzer that calculates technical indicators and evaluates trading strategies
    """

    def __init__(self):
        """Initialize the stock analyzer with data fetcher"""
        self.fetcher = create_data_fetcher_with_credentials()
        self.fetcher.initialize()

    def get_stock_data(self, symbol: str, days: int = 360) -> pd.DataFrame:
        """
        Fetch stock data for the specified symbol and time period
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        # Normalize symbol format for different data providers
        normalized_symbol = self._normalize_symbol(symbol)
        symbols = [normalized_symbol]

        print(f"正在获取 {normalized_symbol} 的数据，时间范围：{start_date.strftime('%Y-%m-%d')} 至 {end_date.strftime('%Y-%m-%d')}...")

        data = self.fetcher.get_bars(symbols, start_date, end_date)

        if normalized_symbol in data and not data[normalized_symbol].empty:
            df = data[normalized_symbol].copy()
            print(f"成功获取 {len(df)} 条记录")
            return df
        else:
            print(f"未能获取 {normalized_symbol} 的数据")
            return pd.DataFrame()

    def _normalize_symbol(self, symbol: str) -> str:
        """
        Normalize the symbol format for different data providers
        """
        symbol = symbol.upper().strip()

        # Handle various input formats
        if symbol.endswith('.SH') or symbol.endswith('.SZ'):
            return symbol
        elif len(symbol) == 6:
            if symbol.startswith(('5', '6', '9')):
                return f"{symbol}.SH"  # Shanghai
            else:
                return f"{symbol}.SZ"  # Shenzhen
        else:
            return symbol

    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various technical indicators for the given data
        """
        if df.empty or 'close' not in df.columns:
            print("Error: DataFrame is empty or missing 'close' column")
            return df

        df = df.sort_values('timestamp').reset_index(drop=True)

        # Ensure all required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                if col in ['open', 'high', 'low']:
                    # Use close price for missing OHLC data
                    df[col] = df['close']
                elif col == 'volume':
                    # Create a default volume column if missing
                    df[col] = 1000000  # Default volume

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        open_price = df['open'].values

        # Calculate technical indicators
        print("正在计算技术指标...")

        # Moving Averages
        df['ma_5'] = MA(close, 5)
        df['ma_6'] = MA(close, 6)  # 用于BIAS(6)计算
        df['ma_10'] = MA(close, 10)
        df['ma_20'] = MA(close, 20)
        df['ma_30'] = MA(close, 30)
        df['ma_50'] = MA(close, 50)
        df['ma_100'] = MA(close, 100)
        df['ma_200'] = MA(close, 200)

        # RSI
        df['rsi_6'] = RSI(close, 6)
        df['rsi_12'] = RSI(close, 12)
        df['rsi_24'] = RSI(close, 24)

        # MACD
        dif, dea, macd = MACD(close)
        df['macd_dif'] = dif
        df['macd_dea'] = dea
        df['macd'] = macd

        # KDJ
        k, d, j = KDJ(close, high, low)
        df['kdj_k'] = k
        df['kdj_d'] = d
        df['kdj_j'] = j

        # Bollinger Bands
        upper, mid, lower = BOLL(close)
        df['boll_upper'] = upper
        df['boll_mid'] = mid
        df['boll_lower'] = lower

        # BIAS
        bias_6, bias_12, bias_24 = BIAS(close)
        df['bias_6'] = bias_6
        df['bias_12'] = bias_12
        df['bias_24'] = bias_24

        # CCI
        df['cci'] = CCI(close, high, low)

        # ATR and True Range
        df['atr_14'] = ATR(close, high, low, 14)

        # DMI/DMI
        pdi, mdi, adx, adxr = DMI(close, high, low)
        df['dmi_pdi'] = pdi
        df['dmi_mdi'] = mdi
        df['dmi_adx'] = adx
        df['dmi_adxr'] = adxr

        # TRIX
        trix, trma = TRIX(close)
        df['trix'] = trix
        df['trix_ma'] = trma

        # VR
        df['vr'] = VR(close, volume)

        # CR
        df['cr'] = CR(close, high, low)

        # Williams %R
        wr, wr_6 = WR(close, high, low)
        df['wr'] = wr
        df['wr_6'] = wr_6

        # BBI
        df['bbi'] = BBI(close)

        # PSY
        psy, psyma = PSY(close)
        df['psy'] = psy
        df['psyma'] = psyma

        # OBV 能量潮
        df['obv'] = OBV(close, volume)

        # MFI 资金流量指标
        df['mfi'] = MFI(close, high, low, volume)

        # Daily return
        df['daily_return'] = df['close'].pct_change()

        # Price position in the range
        df['price_position'] = (df['close'] - LLV(low, 20)) / (HHV(high, 20) - LLV(low, 20)) * 100

        print("技术指标计算完成。")
        return df

    def run_trading_strategies(self, df: pd.DataFrame, symbol: str = "") -> Dict:
        """
        Run various trading strategies and scoring system on the given data
        """
        if df.empty:
            return {}

        print("正在运行交易策略...")

        # Initialize strategy evaluator (传统策略，保留兼容性)
        strategies = TradingStrategies()

        results = {}

        # Individual strategies
        results['rsi_strategy'] = strategies.rsi_strategy(df)
        results['macd_strategy'] = strategies.macd_strategy(df)
        results['ma_crossover_strategy'] = strategies.ma_crossover_strategy(df)
        results['bollinger_bands_strategy'] = strategies.bollinger_bands_strategy(df)
        results['combined_strategy'] = strategies.combined_strategy(df)

        # Evaluate current signals
        results['evaluations'] = {}
        results['evaluations']['rsi'] = strategies.evaluate_current_signal(results['rsi_strategy'], 'RSI Strategy')
        results['evaluations']['macd'] = strategies.evaluate_current_signal(results['macd_strategy'], 'MACD Strategy')
        results['evaluations']['ma_crossover'] = strategies.evaluate_current_signal(results['ma_crossover_strategy'], 'MA Crossover Strategy')
        results['evaluations']['bollinger_bands'] = strategies.evaluate_current_signal(results['bollinger_bands_strategy'], 'Bollinger Bands Strategy')
        results['evaluations']['combined'] = strategies.evaluate_current_signal(results['combined_strategy'], 'Combined Strategy')

        # Run NEW scoring system (百分制)
        print("正在计算多维度评分（百分制）...")
        scoring = ScoringSystem(stop_loss_pct=0.05)

        # 获取日期信息
        latest_date = df['timestamp'].iloc[-1].strftime('%Y-%m-%d') if 'timestamp' in df.columns else ''

        # K线形态分析（已集成到评分系统，此处仅用于报告展示）
        candlestick_result = analyze_candlestick_patterns(df, lookback=5)

        # 计算股价位置（用于形态定性判断）
        latest = df.iloc[-1]
        close = latest.get('close', 0)
        ma20 = latest.get('ma_20', close)
        bias20 = (close / ma20 - 1) if ma20 > 0 else 0

        # 计算60日高低点位置
        position_ratio = 0.5
        if len(df) >= 60:
            high_60 = df['high'].iloc[-60:].max()
            low_60 = df['low'].iloc[-60:].min()
            if high_60 > low_60:
                position_ratio = (close - low_60) / (high_60 - low_60)

        # 计算布林带百分比位置
        boll_pctb = 0.5
        boll_upper = latest.get('boll_upper', close)
        boll_lower = latest.get('boll_lower', close)
        if boll_upper != boll_lower:
            boll_pctb = (close - boll_lower) / (boll_upper - boll_lower)

        # 评分计算（K线形态已集成到评分系统，参与趋势评分）
        results['scoring'] = scoring.calculate_all_scores(
            df=df,
            stock_code=symbol,
            trade_date_T=latest_date,
            trade_date_T1=None,
            open_T1=None
        )

        # 保存形态和位置信息供报告使用（评分详情在 factors_raw.candlestick_detail 中）
        results['candlestick_patterns'] = candlestick_result
        results['position_info'] = {
            'position_ratio': position_ratio,
            'bias20': bias20,
            'boll_pctb': boll_pctb,
            'is_low': position_ratio < 0.35 or bias20 < -0.05 or boll_pctb < 0.2,
            'is_high': position_ratio > 0.70 or bias20 > 0.05 or boll_pctb > 0.8,
        }

        print(f"评分计算完成。综合评分: {results['scoring'].get('score', 0):.1f}分")
        print("交易策略运行完成。")
        return results

    def _analyze_signal_conflicts(self, latest_data: pd.Series, strategies_results: Dict,
                                   df: pd.DataFrame = None,
                                   candlestick_result: Dict = None) -> List[str]:
        """
        分析指标信号冲突情况（改进版：加入动量因子和K线形态）

        检测MACD、TRIX、RSI、DMI等指标之间的方向一致性，
        同时考虑MACD动量变化和成交量配合，
        当存在矛盾信号时给出解释。

        修复：
        1. DMI指标当PDI≈MDI时显示"多空平衡"而非强行归类
        2. 增加信号明细表格，列出每个指标的判断结果
        3. K线形态纳入信号统计（强晨星等计入看多信号）
        """
        conflicts = []

        # DMI多空平衡阈值
        DMI_BALANCE_THRESHOLD = 0.5

        # 获取各指标当前值
        macd = latest_data.get('macd', 0)
        macd_dif = latest_data.get('macd_dif', 0)
        macd_dea = latest_data.get('macd_dea', 0)
        trix = latest_data.get('trix', 0)
        rsi = latest_data.get('rsi_24', 50)
        pdi = latest_data.get('dmi_pdi', 0)
        mdi = latest_data.get('dmi_mdi', 0)
        adx = latest_data.get('dmi_adx', 0)
        cci = latest_data.get('cci', 0)

        # 判断各指标方向（多头/空头/中性）
        macd_bull = macd > 0 and macd_dif > macd_dea  # MACD柱状图为正且DIF在DEA上方
        trix_bull = trix > 0  # TRIX为正表示向上
        rsi_bull = rsi > 50  # RSI大于50偏多
        rsi_neutral = 45 <= rsi <= 55  # RSI接近50视为中性

        # 修复：DMI增加多空平衡判断
        dmi_diff = abs(pdi - mdi)
        dmi_bull = pdi > mdi and dmi_diff >= DMI_BALANCE_THRESHOLD
        dmi_bear = mdi > pdi and dmi_diff >= DMI_BALANCE_THRESHOLD
        dmi_neutral = dmi_diff < DMI_BALANCE_THRESHOLD  # PDI≈MDI时视为平衡

        cci_bull = cci > 0  # CCI为正表示强势

        # === 新增：动量衰减和量能分析 ===
        macd_declining = False
        volume_weak = False

        if df is not None and len(df) >= 2:
            # MACD动量衰减检测：当前MACD柱状图 < 前一日
            macd_hist_current = df['macd'].iloc[-1] if 'macd' in df.columns else macd
            macd_hist_prev = df['macd'].iloc[-2] if 'macd' in df.columns else macd_hist_current
            macd_declining = macd_hist_current < macd_hist_prev

            # 量能不足检测：当前成交量 < 5日均量
            vol_current = latest_data.get('volume', 0)
            vol_ma5 = df['volume'].tail(5).mean() if 'volume' in df.columns and len(df) >= 5 else vol_current
            volume_weak = vol_current < vol_ma5

        # === 新增：K线形态信号统计 ===
        candlestick_bull = False
        candlestick_bear = False
        candlestick_neutral = False
        pattern_names = []

        if candlestick_result and 'patterns' in candlestick_result:
            patterns = candlestick_result['patterns']
            strong_bullish = ["晨星", "看涨吞没", "白色三兵"]
            medium_bullish = ["锤子线", "倒锤子", "穿刺线", "大阳线"]
            strong_bearish = ["暮星", "看跌吞没", "黑色三鸦"]
            medium_bearish = ["流星线", "吊颈线", "乌云盖顶", "大阴线"]

            for p in patterns:
                name = p.get('name', '')
                strength = p.get('strength', '')
                ptype = p.get('type', '')
                pattern_names.append(f"{name}({strength})")

                # 强形态计入信号统计
                if name in strong_bullish or (name in medium_bullish and strength == '强'):
                    candlestick_bull = True
                elif name in strong_bearish or (name in medium_bearish and strength == '强'):
                    candlestick_bear = True
                elif ptype == 'neutral':
                    candlestick_neutral = True

        # 统计原始信号数量（5个技术指标 + K线形态）
        technical_bull = sum([macd_bull, trix_bull, rsi_bull, dmi_bull, cci_bull])
        technical_bear = sum([not macd_bull, not trix_bull, not rsi_bull, dmi_bear, not cci_bull])

        # 总信号统计（包含K线形态）
        bull_signals = technical_bull + (1 if candlestick_bull else 0)
        bear_signals = technical_bear + (1 if candlestick_bear else 0)
        neutral_signals = (1 if dmi_neutral else 0) + (1 if rsi_neutral else 0) + (1 if candlestick_neutral else 0)

        # === 新增：信号明细表格 ===
        conflicts.append("📊 **指标信号明细：**")
        conflicts.append("")
        conflicts.append("| 指标 | 当前值 | 判断 | 说明 |")
        conflicts.append("|------|--------|------|------|")

        # MACD
        macd_status = "看多✅" if macd_bull else "看空❌"
        macd_note = f"柱状图{macd:.3f}"
        conflicts.append(f"| MACD | DIF:{macd_dif:.2f} DEA:{macd_dea:.2f} | {macd_status} | {macd_note} |")

        # TRIX
        trix_status = "看多✅" if trix_bull else "看空❌"
        conflicts.append(f"| TRIX | {trix:.3f} | {trix_status} | {'向上' if trix_bull else '向下'} |")

        # RSI
        if rsi_neutral:
            rsi_status = "中性➖"
        else:
            rsi_status = "看多✅" if rsi_bull else "看空❌"
        conflicts.append(f"| RSI(24) | {rsi:.1f} | {rsi_status} | {'偏多' if rsi > 60 else '偏空' if rsi < 40 else '中性'} |")

        # DMI（修复：显示多空平衡）
        if dmi_neutral:
            dmi_status = "平衡⚖️"
            dmi_note = f"PDI≈MDI({dmi_diff:.2f}差值)，多空力量均衡"
        elif dmi_bull:
            dmi_status = "看多✅"
            dmi_note = f"PDI({pdi:.1f}) > MDI({mdi:.1f})"
        else:
            dmi_status = "看空❌"
            dmi_note = f"MDI({mdi:.1f}) > PDI({pdi:.1f})"
        conflicts.append(f"| DMI | PDI:{pdi:.1f} MDI:{mdi:.1f} ADX:{adx:.1f} | {dmi_status} | {dmi_note} |")

        # CCI
        cci_status = "看多✅" if cci_bull else "看空❌"
        cci_note = "强势" if cci > 100 else "弱势" if cci < -100 else "正常"
        conflicts.append(f"| CCI | {cci:.1f} | {cci_status} | {cci_note} |")

        # K线形态
        if pattern_names:
            pattern_str = ", ".join(pattern_names[:3])  # 最多显示3个
            if candlestick_bull:
                pattern_status = "看多✅"
            elif candlestick_bear:
                pattern_status = "看空❌"
            else:
                pattern_status = "中性➖"
            conflicts.append(f"| K线形态 | - | {pattern_status} | {pattern_str} |")

        conflicts.append("")

        # 统计原始信号数量
        bull_signals = sum([macd_bull, trix_bull, rsi_bull, dmi_bull, cci_bull])
        bear_signals = 5 - bull_signals

        # === 新增：信号强度调整 ===
        # 如果MACD看多但动能衰减且量能不足，视为"弱势偏多"或"中性"
        signal_strength = "normal"
        if macd_bull and macd_declining and volume_weak:
            signal_strength = "weak_bullish"
            bull_signals = max(0, bull_signals - 1)  # 降低多头信号计数

        # 检测主要冲突
        if macd_bull and not trix_bull:
            conflicts.append("⚠️ **MACD与TRIX矛盾**：")
            conflicts.append(f"   - MACD为正（{macd:.2f}），显示短期动量偏多")
            conflicts.append(f"   - TRIX为负（{trix:.2f}），显示中长期趋势仍向下")
            conflicts.append("   - **解读**：MACD反弹可能是短期技术性修复，需观察TRIX是否拐头向上确认趋势反转")
            conflicts.append("")

        if not macd_bull and trix_bull:
            conflicts.append("⚠️ **MACD与TRIX矛盾**：")
            conflicts.append(f"   - MACD为负（{macd:.2f}），显示短期动量偏弱")
            conflicts.append(f"   - TRIX为正（{trix:.2f}），显示中长期趋势向上")
            conflicts.append("   - **解读**：短期回调不改中长期上升趋势，可关注支撑位低吸机会")
            conflicts.append("")

        # 检测RSI与价格趋势背离
        if rsi > 70 and not macd_bull:
            conflicts.append("⚠️ **RSI超买与MACD背离**：")
            conflicts.append(f"   - RSI处于超买区（{rsi:.1f}）")
            conflicts.append(f"   - 但MACD未同步走强（{macd:.2f}）")
            conflicts.append("   - **解读**：上涨动能可能衰竭，警惕冲高回落风险")
            conflicts.append("")

        # === 新增：检测MACD动量衰减 ===
        if macd > 0 and macd_declining and volume_weak:
            conflicts.append("⚠️ **信号混合：趋势虽在均线上方，但动能衰减，量能不足**：")
            conflicts.append(f"   - MACD为正（{macd:.2f}），显示短期趋势偏多")
            conflicts.append(f"   - 但MACD柱状图下降（{macd_hist_current:.2f} < {macd_hist_prev:.2f}），动能减弱")
            conflicts.append(f"   - 成交量萎缩（{vol_current:,.0f} < {vol_ma5:,.0f}），市场参与度下降")
            conflicts.append("   - **解读**：属于弱势整理，建议观望等待量能配合")
            conflicts.append("")

        # 检测DMI与MACD一致性
        if dmi_bull and not macd_bull:
            conflicts.append("⚠️ **DMI与MACD矛盾**：")
            conflicts.append(f"   - DMI显示多头占优（PDI{pdi:.1f} > MDI{mdi:.1f}）")
            conflicts.append(f"   - 但MACD显示空头（{macd:.2f}）")
            conflicts.append("   - **解读**：多方力量占优但动能不足，可能进入盘整阶段")
            conflicts.append("")

        # === 新增：动量衰减与量能不足检测 ===
        if signal_strength == "weak_bullish":
            conflicts.append("⚠️ **信号混合：趋势与动能背离**：")
            conflicts.append(f"   - MACD为正（{macd:.2f}），显示趋势在均线上方")
            conflicts.append(f"   - 但MACD柱状图下降（{macd_hist_current:.3f} < {macd_hist_prev:.3f}），动能衰减")
            conflicts.append(f"   - 成交量不足（{vol_current:,.0f} < 均量{vol_ma5:,.0f}），资金参与度低")
            conflicts.append("   - **解读**：属于弱势整理状态，不宜盲目追多")
            conflicts.append("")

        # 信号一致性总结（改进版：包含K线形态和DMI平衡状态）
        total_signals = 6 if candlestick_result and pattern_names else 5  # 5技术指标 + 可选K线形态

        if dmi_neutral:
            dmi_balance_note = f"（注：DMI显示多空平衡，PDI≈MDI差值仅{dmi_diff:.2f}）"
        else:
            dmi_balance_note = ""

        if signal_strength == "weak_bullish":
            conflicts.append(f"📊 **信号一致性：中性偏弱**（技术看多{technical_bull} vs 看空{technical_bear}，K线{'看多' if candlestick_bull else '看空' if candlestick_bear else '中性'}）")
            if dmi_balance_note:
                conflicts.append(f"   {dmi_balance_note}")
            conflicts.append("   趋势虽在均线上方，但动能衰减，量能不足，建议观望")
            conflicts.append("")
        elif bull_signals >= 4:
            conflicts.append(f"✅ **信号一致性：偏多**（{bull_signals}/{total_signals}个信号看多，含K线形态）")
            if pattern_names and candlestick_bull:
                conflicts.append(f"   K线形态强力支持：{', '.join(pattern_names[:2])}")
            conflicts.append("   多数指标方向一致，趋势较为明确")
            conflicts.append("")
        elif bear_signals >= 4:
            conflicts.append(f"⚠️ **信号一致性：偏空**（{bear_signals}/{total_signals}个信号看空）")
            if pattern_names and candlestick_bear:
                conflicts.append(f"   K线形态看跌：{', '.join(pattern_names[:2])}")
            conflicts.append("   多数指标方向一致，需警惕下行风险")
            conflicts.append("")
        elif dmi_neutral and neutral_signals >= 2:
            conflicts.append(f"📊 **信号一致性：中性震荡**（看多{bull_signals} vs 看空{bear_signals}，中性信号{neutral_signals}个）")
            conflicts.append(f"   {dmi_balance_note}")
            conflicts.append("   市场处于震荡整理阶段，建议观望等待方向明确")
            conflicts.append("")
        else:
            conflicts.append(f"📊 **信号一致性：混合**（看多{bull_signals} vs 看空{bear_signals}）")
            if dmi_balance_note:
                conflicts.append(f"   {dmi_balance_note}")
            conflicts.append("   指标信号分歧较大，建议观望等待方向明确")
            conflicts.append("")

        return conflicts

    def generate_report(self, df: pd.DataFrame, strategies_results: Dict, symbol: str) -> str:
        """
        Generate comprehensive analysis report using four-section architecture
        """
        if df.empty:
            return "No data available for report generation"

        latest_data = df.iloc[-1]
        report = []

        # 基本信息
        report.append(f"# 股票技术分析报告：{symbol}")
        report.append("")
        report.append(f"**分析日期：** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append("")

        # 获取评分数据
        scoring = strategies_results.get('scoring', {})
        if not scoring or 'error' in scoring:
            report.append("*评分系统未返回有效结果*")
            return "\n".join(report)

        score = scoring.get('score', 50)

        # 使用增强的位置判断方法（综合均线位置 + 技术超买超卖指标）
        # 优先使用 scoring 中的位置修正系数，避免双重判断
        position_modifier = scoring.get('position_modifier', 1.0)
        position_info = self._determine_position_status(latest_data, position_modifier)

        # 保留原有的 price_ratio 转换逻辑（兼容后续代码）
        position_info['price_ratio_display'] = position_info['price_ratio'] * 50

        # 获取K线形态评估（定性）
        candlestick_recognizer = CandlestickPatternRecognizer()
        candlestick_result = candlestick_recognizer.recognize_all_patterns(df, lookback=5)

        # 将形态结果转换为评估字典
        pattern_assessment = self._convert_pattern_to_assessment(candlestick_result, df)

        # ============ 四部分报告架构 ============

        # 第一部分：核心结论区
        report.extend(self._generate_core_conclusion(scoring, position_info, pattern_assessment))

        # 第二部分：多维信号共振分析
        report.extend(self._generate_signal_resonance(latest_data, position_info, pattern_assessment, scoring, df))

        # 第三部分：量化评分与因子拆解
        report.extend(self._generate_score_breakdown(scoring))

        # 第四部分：交易执行计划
        report.extend(self._generate_trading_plan(scoring, position_info, pattern_assessment, latest_data))

        # 原始信号明细（保留在附录）
        report.append("---")
        report.append("")
        report.append("## 附录：原始技术指标")
        report.append("")
        report.append(f"| 指标 | 数值 | 状态 |")
        report.append(f"|------|------|------|")

        # RSI
        rsi_val = latest_data.get('rsi_24', 50)
        if rsi_val > 70:
            rsi_desc = "超买区"
        elif rsi_val < 30:
            rsi_desc = "超卖区"
        else:
            rsi_desc = "中性"
        report.append(f"| RSI(24) | {rsi_val:.2f} | {rsi_desc} |")

        # MACD
        macd_val = latest_data['macd']
        macd_desc = "多头" if macd_val > 0 else "空头"
        report.append(f"| MACD | {macd_val:.2f} | {macd_desc} |")

        # KDJ
        k, d, j = latest_data['kdj_k'], latest_data['kdj_d'], latest_data['kdj_j']
        kdj_desc = "J值偏高" if j > 80 else "J值偏低" if j < 20 else "正常"
        report.append(f"| KDJ | K: {k:.2f} / D: {d:.2f} / J: {j:.2f} | {kdj_desc} |")

        # MA - 从 position_info 获取值
        ma20 = position_info.get('ma20', 0)
        ma50 = position_info.get('ma50', 0)
        close = position_info.get('close', 0)
        ma200 = latest_data.get('ma_200', np.nan)
        ma200_str = f"{ma200:.2f}" if not pd.isna(ma200) else "无数据"
        report.append(f"| 移动平均线 | MA20: ¥{ma20:.2f} / MA50: ¥{ma50:.2f} / MA200: {ma200_str} | 趋势参考 |")

        # BOLL
        bu, bm, bl = latest_data['boll_upper'], latest_data['boll_mid'], latest_data['boll_lower']
        boll_desc = "触及上轨" if close >= bu else "触及下轨" if close <= bl else "中轨附近"
        report.append(f"| 布林带 | 上轨: ¥{bu:.2f} / 中轨: ¥{bm:.2f} / 下轨: ¥{bl:.2f} | {boll_desc} |")

        # CCI
        cci_val = latest_data['cci']
        if cci_val > 200:
            cci_desc = "严重超买区"
        elif cci_val > 100:
            cci_desc = "超买区"
        elif cci_val < -200:
            cci_desc = "严重超卖区"
        elif cci_val < -100:
            cci_desc = "超卖区"
        else:
            cci_desc = "正常区间"
        report.append(f"| CCI | {cci_val:.2f} | {cci_desc} |")

        # DMI
        pdi, mdi, adx = latest_data['dmi_pdi'], latest_data['dmi_mdi'], latest_data['dmi_adx']
        dmi_diff = abs(pdi - mdi)
        DMI_BALANCE_THRESHOLD = 0.5
        if dmi_diff < DMI_BALANCE_THRESHOLD:
            dmi_desc = f"多空平衡（差值{dmi_diff:.2f}）"
        elif pdi > mdi:
            dmi_desc = "多头占优"
        else:
            dmi_desc = "空头占优"
        report.append(f"| DMI | PDI: {pdi:.2f} / MDI: {mdi:.2f} / ADX: {adx:.2f} | {dmi_desc} |")

        # WR
        wr_val = latest_data['wr']
        if wr_val < 10:
            wr_desc = "严重超买"
        elif wr_val > 90:
            wr_desc = "严重超卖"
        else:
            wr_desc = "正常区间"
        report.append(f"| WR | {wr_val:.2f} | {wr_desc} |")

        # BIAS
        bias6 = latest_data['bias_6']
        bias12 = latest_data['bias_12']
        bias24 = latest_data['bias_24']
        bias_avg = (bias6 + bias12 + bias24) / 3
        if bias_avg > 5:
            bias_desc = "超买偏离"
        elif bias_avg < -5:
            bias_desc = "超卖偏离"
        else:
            bias_desc = "正常区间"
        report.append(f"| BIAS | BIAS6: {bias6:.2f}% / BIAS12: {bias12:.2f}% / BIAS24: {bias24:.2f}% | {bias_desc} |")

        report.append("")

        # 添加K线形态报告（如果有）
        if candlestick_result and candlestick_result.get('patterns'):
            report.append("### K线形态识别")
            report.append("")
            candlestick_report = candlestick_recognizer.format_patterns_report(candlestick_result)
            report.append(candlestick_report)

        report.append("")
        report.append("> **免责声明：** 本分析仅供学习参考，不构成投资建议。投资决策应基于全面研究和独立判断。")
        report.append("")

        return "\n".join(report)

    def _determine_position_status(self, latest_data: pd.Series, position_modifier: float = 1.0) -> Dict:
        """
        综合判断位置状态

        优化：使用 scoring_system 计算的 position_modifier 作为主要判断依据，避免双重标准

        返回: {
            'position': 'high'/'low'/'middle',
            'price_ratio': float,
            'is_overbought': bool,
            'is_oversold': bool,
            'is_extreme_overbought': bool,
            'is_extreme_oversold': bool,
            'reason': str,
            'close': float,
            'ma20': float,
            'ma50': float
        }
        """
        close = latest_data.get('close', 0)
        ma20 = latest_data.get('ma_20', close)
        ma50 = latest_data.get('ma_50', close)

        # 1. 均线位置判断（辅助判断）
        avg_ma = (ma20 + ma50) / 2 if ma20 > 0 and ma50 > 0 else close
        price_ratio = close / avg_ma if avg_ma > 0 else 1.0

        # 2. 技术位置判断（基于超买超卖指标）
        wr = latest_data.get('wr_14', latest_data.get('wr', 50))
        cci = latest_data.get('cci', 0)
        rsi = latest_data.get('rsi_24', 50)
        boll_upper = latest_data.get('boll_upper', 0)
        boll_lower = latest_data.get('boll_lower', 0)
        boll_pctb = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5

        # 极端超买判断
        is_extreme_overbought = (
            wr < 10 or
            cci > 200 or
            rsi > 80 or
            boll_pctb > 0.95
        )

        # 普通超买判断
        is_overbought = (
            wr < 20 or
            cci > 100 or
            rsi > 70 or
            boll_pctb > 0.85
        ) and not is_extreme_overbought

        # 极端超卖判断
        is_extreme_oversold = (
            wr > 90 or
            cci < -200 or
            rsi < 20 or
            boll_pctb < 0.05
        )

        # 普通超卖判断
        is_oversold = (
            wr > 80 or
            cci < -100 or
            rsi < 30 or
            boll_pctb < 0.15
        ) and not is_extreme_oversold

        # 3. 综合判断：优先使用 position_modifier（来自 scoring_system）
        # position_modifier 范围: 0.3 ~ 1.0
        if position_modifier < 0.5:
            position = 'high'
            reason = f"位置危险（修正系数={position_modifier:.2f}，严重超买）"
        elif position_modifier < 0.7:
            position = 'high'
            reason = f"位置偏高（修正系数={position_modifier:.2f}，技术指标超买）"
        elif position_modifier >= 0.95:
            position = 'low'
            reason = f"位置安全（修正系数={position_modifier:.2f}，入场位置良好）"
        elif position_modifier >= 0.8:
            position = 'middle'
            reason = f"位置适中（修正系数={position_modifier:.2f}）"
        else:
            # 回退到技术指标判断
            if is_extreme_overbought:
                position = 'high'
                reason = f"极端超买（WR={wr:.1f}, CCI={cci:.1f}, RSI={rsi:.1f}）"
            elif is_overbought:
                position = 'high'
                reason = f"技术指标超买（WR={wr:.1f}, RSI={rsi:.1f}）"
            elif is_extreme_oversold:
                position = 'low'
                reason = f"极端超卖（WR={wr:.1f}, CCI={cci:.1f}, RSI={rsi:.1f}）"
            elif is_oversold:
                position = 'low'
                reason = f"技术指标超卖（WR={wr:.1f}, RSI={rsi:.1f}）"
            else:
                # 最终回退到均线位置判断
                if price_ratio > 1.15:
                    position = 'high'
                elif price_ratio < 0.85:
                    position = 'low'
                else:
                    position = 'middle'
                reason = f"均线位置判断（价格/均价={price_ratio:.2f}）"

        return {
            'position': position,
            'price_ratio': price_ratio,
            'is_overbought': is_overbought or is_extreme_overbought,
            'is_oversold': is_oversold or is_extreme_oversold,
            'is_extreme_overbought': is_extreme_overbought,
            'is_extreme_oversold': is_extreme_oversold,
            'reason': reason,
            'close': close,
            'ma20': ma20,
            'ma50': ma50,
            'wr': wr,
            'cci': cci,
            'rsi': rsi,
            'boll_pctb': boll_pctb
        }

    def _convert_pattern_to_assessment(self, candlestick_result: Dict, df: pd.DataFrame) -> Dict:
        """
        将K线形态识别结果转换为形态评估字典
        """
        patterns = candlestick_result.get('patterns', [])

        if not patterns:
            return {
                'type': 'none',
                'strength': 0,
                'patterns': [],
                'stop_loss': None
            }

        # 中文强度值映射为数字
        strength_map = {'强': 4, '中': 2, '弱': 1}

        # 按强度排序（使用映射后的数字值）
        def get_strength_value(p):
            s = p.get('strength', 1)
            return strength_map.get(s, 1) if isinstance(s, str) else int(s)

        sorted_patterns = sorted(patterns, key=get_strength_value, reverse=True)
        strongest = sorted_patterns[0]

        pattern_name = strongest.get('name', '')
        pattern_type = strongest.get('type', '')

        # 中文强度值映射为数字
        strength_map = {'强': 4, '中': 2, '弱': 1}
        strength_raw = strongest.get('strength', 1)
        strength = strength_map.get(strength_raw, 1) if isinstance(strength_raw, str) else int(strength_raw)

        # 计算止损位（基于形态最低点）
        stop_loss = None
        if len(df) >= 3:
            recent_low = df['low'].iloc[-3:].min()
            stop_loss = recent_low * 0.98  # 形态最低点下方2%

        # 映射类型
        if pattern_type == 'bullish':
            assessment_type = 'bullish_reversal'
        elif pattern_type == 'bearish':
            assessment_type = 'bearish_reversal'
        else:
            assessment_type = 'neutral'

        return {
            'type': assessment_type,
            'strength': strength,
            'patterns': [p.get('name', '') for p in sorted_patterns[:3]],
            'stop_loss': stop_loss,
            'description': strongest.get('description', '')
        }

    def analyze_stock(self, symbol: str, days: int = 360) -> str:
        """
        Main method to analyze a stock completely
        """
        print(f"开始分析 {symbol}...")

        # Get stock data
        df = self.get_stock_data(symbol, days)
        if df.empty:
            return f"无法获取 {symbol} 的数据"

        # Calculate technical indicators
        df_with_indicators = self.calculate_technical_indicators(df)

        # Run trading strategies
        strategies_results = self.run_trading_strategies(df_with_indicators, symbol)

        # Generate comprehensive report using four-section architecture
        report = self.generate_report(df_with_indicators, strategies_results, symbol)

        return report

    def _add_detailed_reasoning(self, report: List[str], score: float,
                                factors_raw: Dict, factors_score: Dict,
                                latest_data: pd.Series, trigger_type: str) -> None:
        """
        添加详细的结论理由说明

        Args:
            report: 报告内容列表
            score: 综合评分
            factors_raw: 因子原始值
            factors_score: 因子得分
            latest_data: 最新数据
            trigger_type: 触发类型
        """
        report.append("")
        report.append("**📋 理由说明：**")
        report.append("")

        # 1. 分析主要支撑和拖累因子
        factor_scores_list = [
            ('trend1', '趋势偏离', factors_score.get('trend1', 0)),
            ('trend2', '均线斜率', factors_score.get('trend2', 0)),
            ('mom1', 'MACD动量', factors_score.get('mom1', 0)),
            ('mom2', 'RSI位置', factors_score.get('mom2', 0)),
            ('flow1', '成交量比', factors_score.get('flow1', 0)),
            ('flow2', 'OBV资金流', factors_score.get('flow2', 0)),
            ('pos1', '布林带位置', factors_score.get('pos1', 0)),
            ('pos2', '波动率位置', factors_score.get('pos2', 0)),
            ('bias20', 'MA20乖离率', factors_score.get('bias20', 0)),
        ]

        # 排序找出最强和最弱因子
        factor_scores_list.sort(key=lambda x: x[2], reverse=True)

        strong_factors = [f for f in factor_scores_list if f[2] >= 0.7][:2]
        weak_factors = [f for f in factor_scores_list if f[2] <= 0.4][:2]

        if strong_factors:
            report.append("✅ **支撑因素：**")
            for key, name, val in strong_factors:
                raw_val = factors_raw.get(key, 0)
                report.append(f"   - {name}表现良好（得分{val*100:.0f}分）")
                # 添加具体说明
                if key == 'trend1' and raw_val > 0:
                    report.append(f"     股价在MA20上方{raw_val*100:.1f}%，短期趋势偏多")
                elif key == 'mom1' and raw_val > 0:
                    report.append(f"     MACD动量增强，柱状图3日变化+{raw_val:.2f}")
                elif key == 'flow2' and raw_val > 0:
                    report.append(f"     OBV资金流入，5日变化率+{raw_val:.1f}%")
                elif key == 'bias20' and abs(raw_val) < 0.05:
                    report.append(f"     乖离率适中（{raw_val*100:.2f}%），无超买超卖风险")
            report.append("")

        if weak_factors:
            report.append("⚠️ **拖累因素：**")
            for key, name, val in weak_factors:
                raw_val = factors_raw.get(key, 0)
                report.append(f"   - {name}表现较弱（得分{val*100:.0f}分）")
                # 添加具体说明
                if key == 'trend2' and raw_val < 0:
                    report.append(f"     短期均线斜率为负，上升趋势放缓")
                elif key == 'mom1' and raw_val < 0:
                    report.append(f"     MACD动量减弱，柱状图3日变化{raw_val:.2f}")
                elif key == 'mom2':
                    rsi_val = latest_data.get('rsi_24', 50)
                    if rsi_val > 70:
                        report.append(f"     RSI处于超买区（{rsi_val:.1f}），短期回调风险")
                    elif rsi_val < 40:
                        report.append(f"     RSI偏弱（{rsi_val:.1f}），动能不足")
                elif key == 'flow1' and raw_val < 0:
                    report.append(f"     成交量萎缩，市场参与度下降")
                elif key == 'bias20' and raw_val > 0.05:
                    report.append(f"     乖离率偏高（{raw_val*100:.2f}%），有回归均线需求")
            report.append("")

        # 2. 触发信号说明
        if trigger_type != 'none':
            report.append(f"📊 **触发信号：** 检测到{trigger_type}型信号，")
            if trigger_type == 'breakout':
                report.append("   股价放量突破近期高点，显示多方力量增强")
            elif trigger_type == 'pullback':
                report.append("   缩量回踩均线后收阳，显示支撑有效")
            report.append("")

        # 3. 关键价位提示
        close = latest_data.get('close', 0)
        ma20 = latest_data.get('ma_20', 0)
        boll_upper = latest_data.get('boll_upper', 0)
        boll_lower = latest_data.get('boll_lower', 0)

        report.append("📍 **关键价位：**")
        if ma20 > 0:
            report.append(f"   - MA20支撑位：¥{ma20:.2f}")
        if boll_upper > 0 and boll_lower > 0:
            report.append(f"   - 布林带上轨压力位：¥{boll_upper:.2f}")
            report.append(f"   - 布林带下轨支撑位：¥{boll_lower:.2f}")
        report.append("")

        # 4. 综合判断
        report.append("🔍 **综合判断：**")
        if score >= 75:
            report.append(f"   综合评分{score:.1f}分，各因子整体表现良好，")
            report.append("   主要指标方向一致，建议积极参与。")
        elif score <= 40:
            report.append(f"   综合评分{score:.1f}分，存在较多不利因素，")
            report.append("   建议控制仓位，等待信号改善。")
        else:
            report.append(f"   综合评分{score:.1f}分，多空因素交织，")
            report.append("   建议观望等待更明确的方向信号。")
        report.append("")

    def _generate_core_conclusion(self, scoring: Dict, position_info: Dict,
                                   pattern_assessment: Dict) -> List[str]:
        """
        生成第一部分：核心结论区（重构版）

        新架构：趋势得分 × 位置修正系数
        修复：优先使用评分系统的熔断机制结果
        """
        report = []
        report.append("## 第一部分：核心结论")
        report.append("")

        score = scoring.get('score', 50)
        score_grade = scoring.get('score_grade', 'N/A')
        trend_score = scoring.get('trend_score', score)
        position_modifier = scoring.get('position_modifier', 1.0)

        # 获取评分系统的操作指引（已包含熔断机制）
        execution = scoring.get('execution', {})
        action_guide = execution.get('action_guide', '')

        # 获取形态和位置信息
        pattern_type = pattern_assessment.get('type', 'none')
        pattern_strength = int(pattern_assessment.get('strength', 0))
        position = position_info.get('position', 'middle')
        is_overbought = position_info.get('is_overbought', False)
        is_oversold = position_info.get('is_oversold', False)

        # ========== 关键修复：优先使用熔断机制的结果 ==========
        action_override = None
        override_reason = None

        # 获取K线形态详情（区分长短期位置）- 提前定义
        factors_raw = scoring.get('factors_raw', {})
        candlestick_detail = factors_raw.get('candlestick_detail', {})
        long_term_pos = candlestick_detail.get('long_term_position', 'mid')
        short_term_pos = candlestick_detail.get('short_term_position', 'mid')

        # 如果 action_guide 包含熔断关键词，直接使用
        if '熔断' in action_guide or '回避' in action_guide:
            if '熔断-回避' in action_guide:
                action_emoji = "🔴"
                action_text = "回避/卖出"
                action_override = 'fuse'
                override_reason = action_guide
            elif '熔断-诱多' in action_guide:
                action_emoji = "🟡"
                action_text = "观望（警惕诱多）"
                action_override = 'fuse'
                override_reason = action_guide
            else:
                action_emoji = "🟡"
                action_text = "观望"
                action_override = 'fuse'
                override_reason = action_guide
        elif '风险警告' in action_guide:
            action_emoji = "🟡"
            action_text = "谨慎观望"
            action_override = 'warning'
            override_reason = action_guide
        else:
            # 形态覆盖规则（修正：区分长短期位置）
            if long_term_pos == 'low' and short_term_pos == 'low' and pattern_type in ['bullish_reversal', 'bullish_continuation'] and pattern_strength >= 3:
                action_override = 'buy'
                override_reason = f"长短期双低位+强底部形态({pattern_assessment.get('patterns', ['未知'])[0]})"
            elif short_term_pos == 'high' and pattern_type in ['bearish_reversal', 'bearish_continuation'] and pattern_strength >= 3:
                action_override = 'sell'
                override_reason = f"短期高位+强顶部形态({pattern_assessment.get('patterns', ['未知'])[0]})"
            elif short_term_pos == 'high' and pattern_type in ['bullish_reversal', 'bullish_continuation']:
                action_override = 'wait'
                override_reason = f"短期高位看涨形态，可能是诱多/力竭"

        # 根据位置修正系数调整操作建议（细化逻辑）
        if action_override in ['fuse', 'warning']:
            # 熔断机制或风险警告已确定操作，不做调整
            pass
        elif action_override == 'buy':
            action_emoji = "🟢"
            action_text = "试探性买入"
        elif action_override == 'sell':
            action_emoji = "🔴"
            action_text = "卖出/减仓"
        elif action_override == 'wait':
            action_emoji = "🟡"
            action_text = "观望（警惕诱多）"
        elif position_modifier < 0.5:
            # 位置危险，建议卖出或观望
            action_emoji = "🔴"
            action_text = "不宜追高"
        elif position_modifier < 0.7:
            # 位置警戒
            action_emoji = "🟡"
            action_text = "谨慎观望"
        elif score >= 65:
            # 趋势好+位置安全
            action_emoji = "🟢"
            action_text = "买入"
        elif score >= 50:
            action_emoji = "🟡"
            action_text = "观望"
        else:
            action_emoji = "🔴"
            action_text = "不建议"

        report.append(f"### {action_emoji} 操作指令：{action_text}")
        report.append("")

        # 评分展示（新架构）
        if override_reason:
            report.append(f"**技术评分：{score:.1f}分（{score_grade}）→ {action_text}**")
            report.append(f"**覆盖原因：{override_reason}**")
        else:
            report.append(f"**技术评分：{score:.1f}分（{score_grade}）**")
        report.append("")

        # 评分构成
        report.append(f"**评分构成：** 趋势分 {trend_score:.1f} × 位置系数 {position_modifier:.2f} = {score:.1f}")
        report.append("")

        # 置信度计算
        factors_score = scoring.get('factors_score', {})
        bullish_count = sum(1 for v in factors_score.values() if isinstance(v, (int, float)) and v > 60)
        bearish_count = sum(1 for v in factors_score.values() if isinstance(v, (int, float)) and v < 40)

        if bullish_count >= 3 or bearish_count >= 3:
            if pattern_strength >= 3:
                confidence = "高"
                confidence_reason = f"多数因子同向+形态支持"
            else:
                confidence = "中高"
                confidence_reason = f"多数因子同向"
        elif bullish_count >= 2 or bearish_count >= 2:
            confidence = "中"
            confidence_reason = f"因子方向较一致"
        else:
            confidence = "中低"
            confidence_reason = "因子分歧，需综合判断"

        report.append(f"**置信度：{confidence}**（{confidence_reason}）")
        report.append("")

        # 关键理由（区分长短期位置）
        report.append("### 💡 关键理由")
        report.append("")

        reason_parts = []

        # 长短期位置分析
        if long_term_pos == 'low' and short_term_pos == 'high':
            reason_parts.append("⚠️ 长期低位但短期超买，建议等待回踩后介入")
        elif long_term_pos == 'low' and short_term_pos == 'low':
            reason_parts.append("✅ 长期低位+短期超卖，黄金坑机会")
        elif long_term_pos == 'high' and short_term_pos == 'high':
            reason_parts.append("⚠️ 长短期双高位，风险较大")
        elif long_term_pos == 'high' and short_term_pos == 'low':
            reason_parts.append("⚠️ 长期高位短期超卖，可能是反弹但趋势未改")
        elif short_term_pos == 'high':
            reason_parts.append("⚠️ 短期超买，不宜追高")
        elif short_term_pos == 'low':
            reason_parts.append("✅ 短期超卖，可关注反弹机会")

        # 基于位置修正系数
        if position_modifier < 0.5:
            reason_parts.append("入场位置危险（严重超买）")
        elif position_modifier < 0.7:
            reason_parts.append("入场位置偏高（技术指标超买）")
        elif position_modifier >= 0.95:
            reason_parts.append("入场位置安全")

        # MA200压力判断（新增）
        close = position_info.get('close', 0)
        ma200 = position_info.get('ma200', 0)
        if ma200 > 0 and close > 0:
            ma200_distance = (ma200 - close) / close * 100
            if ma200 > close and ma200_distance > 3:
                reason_parts.append(f"⚠️ MA200(¥{ma200:.2f})在上方{ma200_distance:.1f}%处形成压力，需突破确认")

        # 基于趋势得分
        if trend_score >= 70:
            reason_parts.append("趋势强势确立")
        elif trend_score >= 55:
            reason_parts.append("趋势偏强")
        elif trend_score < 40:
            # 检测"接飞刀"行情：趋势极弱但位置看起来安全
            if position_modifier >= 0.95:
                reason_parts.append("⚠️【接飞刀风险】位置虽低但趋势极弱，切勿盲目抄底")
            else:
                reason_parts.append("趋势极弱")
        elif trend_score < 45:
            reason_parts.append("趋势偏弱")

        # 基于形态（结合位置判断）
        if pattern_strength >= 3:
            patterns = pattern_assessment.get('patterns', [])
            if patterns:
                # 区分形态在当前位置的含义
                if short_term_pos == 'high' and pattern_type in ['bullish_reversal', 'bullish_continuation']:
                    reason_parts.append(f"⚠️ 高位看涨形态「{patterns[0]}」可能是诱多/力竭")
                elif short_term_pos == 'low' and pattern_type in ['bullish_reversal', 'bullish_continuation']:
                    reason_parts.append(f"✅ 低位看涨形态「{patterns[0]}」支持反弹")
                else:
                    reason_parts.append(f"K线形态：{patterns[0]}")

        # 添加警告
        warnings = scoring.get('warnings', [])
        if warnings:
            reason_parts.append(f"注意：{warnings[0]}")

        report.append("，".join(reason_parts) if reason_parts else "综合分析，建议谨慎操作")
        report.append("")
        return report

    def _generate_signal_resonance(self, latest_data: pd.Series, position_info: Dict,
                                    pattern_assessment: Dict, scoring: Dict,
                                    df: pd.DataFrame = None) -> List[str]:
        """
        生成第二部分：多维信号共振分析
        """
        report = []
        report.append("## 第二部分：多维信号共振分析")
        report.append("")

        # 1. 趋势状态（综合考虑DMI和均线排列）
        report.append("### 📊 趋势状态")
        report.append("")

        pdi = latest_data.get('dmi_pdi', 50)
        mdi = latest_data.get('dmi_mdi', 20)
        adx = latest_data.get('dmi_adx', 20)
        ma20 = latest_data.get('ma_20', 0)
        ma50 = latest_data.get('ma_50', 0)
        close = latest_data.get('close', 0)

        dmi_diff = abs(pdi - mdi)
        DMI_BALANCE_THRESHOLD = 0.5

        # 判断均线排列和股价位置
        ma_bullish = False
        ma_bearish = False
        price_above_ma = False  # 股价在均线上方
        price_below_ma = False  # 股价在均线下方
        if ma20 > 0 and ma50 > 0 and close > 0:
            if close > ma20 > ma50:
                ma_bullish = True  # 完美多头排列
            elif close < ma20 < ma50:
                ma_bearish = True  # 完美空头排列
            elif close > ma20 and close > ma50:
                # 股价在MA20和MA50上方，即使MA20<MA50也是偏多
                price_above_ma = True
            elif close < ma20 and close < ma50:
                # 股价在MA20和MA50下方
                price_below_ma = True

        # 综合判断趋势状态（均线优先，DMI辅助）
        if ma_bearish or price_below_ma:
            # 均线空头排列或股价在均线下方
            if price_below_ma and not ma_bearish:
                trend_state = "下跌趋势（弱）"
                trend_desc = f"⚠️ 股价在MA20({ma20:.2f})和MA50({ma50:.2f})下方，均线压制"
            elif adx > 25:
                trend_state = "下跌趋势（强）"
                trend_desc = f"均线空头排列（股价<MA20<MA50），ADX={adx:.1f}"
            else:
                trend_state = "下跌趋势（弱）"
                trend_desc = f"均线空头排列（股价<MA20<MA50），PDI>MDI仅为短期反弹"
        elif ma_bullish:
            # 完美多头排列（股价>MA20>MA50）
            # 关键修复：必须结合DMI判断，避免"均线多头但DMI空头"的误判
            if mdi > pdi and dmi_diff >= DMI_BALANCE_THRESHOLD:
                # DMI空头占优 - 均线多头可能是诱多/假突破
                trend_state = "震荡偏空（警惕诱多）"
                trend_desc = f"⚠️ 均线虽多头排列，但MDI({mdi:.1f})>PDI({pdi:.1f})，空头力量更强，警惕回调"
            elif pdi > mdi and dmi_diff >= DMI_BALANCE_THRESHOLD and adx > 25:
                trend_state = "上升趋势（强）"
                trend_desc = f"✅ 均线多头排列+DMI多头占优，ADX={adx:.1f}"
            elif pdi > mdi and dmi_diff >= DMI_BALANCE_THRESHOLD:
                trend_state = "上升趋势（弱）"
                trend_desc = f"✅ 均线多头排列，PDI({pdi:.1f})>MDI({mdi:.1f})，但ADX={adx:.1f}偏低"
            else:
                # DMI平衡，均线多头但无明确趋势
                trend_state = "震荡偏多"
                trend_desc = f"均线多头排列，但PDI≈MDI（差值{dmi_diff:.2f}），趋势不明确"
        elif price_above_ma:
            # 股价在均线上方，但均线未多头排列 - 需要结合DMI判断
            # 关键修复：不能简单判定为上升，需看DMI多空力量
            if pdi > mdi and dmi_diff >= DMI_BALANCE_THRESHOLD:
                # DMI多头占优
                if adx > 25:
                    trend_state = "上升趋势（强）"
                else:
                    trend_state = "上升趋势（弱）"
                trend_desc = f"股价在MA20({ma20:.2f})和MA50({ma50:.2f})上方，PDI>MDI，偏多"
            elif mdi > pdi and dmi_diff >= DMI_BALANCE_THRESHOLD:
                # DMI空头占优 - 尽管股价在均线上方，但空方力量更强
                trend_state = "震荡偏空"
                trend_desc = f"⚠️ 股价虽在均线上方，但MDI({mdi:.1f})>PDI({pdi:.1f})，空头占优，警惕回调"
            else:
                # DMI平衡
                trend_state = "震荡/无趋势"
                trend_desc = f"股价在MA20({ma20:.2f})和MA50({ma50:.2f})上方，但PDI≈MDI（差值{dmi_diff:.2f}），多空力量均衡"
        elif dmi_diff < DMI_BALANCE_THRESHOLD:
            trend_state = "震荡/无趋势"
            trend_desc = f"PDI≈MDI（差值{dmi_diff:.2f}），多空力量均衡"
        elif pdi > mdi:
            if adx > 25:
                trend_state = "上升趋势（强）"
            else:
                trend_state = "上升趋势（弱）"
            trend_desc = f"PDI>MDI，多头占优，ADX={adx:.1f}（均线交织，需确认）"
        else:
            if adx > 25:
                trend_state = "下跌趋势（强）"
            else:
                trend_state = "下跌趋势（弱）"
            trend_desc = f"PDI<MDI，空头占优，ADX={adx:.1f}"

        report.append(f"- **状态**：{trend_state}")
        report.append(f"- **说明**：{trend_desc}")
        report.append("")

        # 2. 动能状态
        report.append("### ⚡ 动能状态")
        report.append("")

        macd = latest_data.get('macd', 0)
        rsi = latest_data.get('rsi_24', 50)

        # 从 scoring 中获取 MACD 动量原始值
        factors_raw = scoring.get('factors_raw', {})
        mom1 = factors_raw.get('macd_momentum', 0)

        if macd > 0 and mom1 > 0:
            momentum_state = "多头动能增强"
            momentum_desc = "MACD红柱放大，动能向上"
        elif macd > 0 and mom1 < 0:
            momentum_state = "多头动能减弱（警惕）"
            momentum_desc = "MACD红柱缩短，存在顶背离可能"
        elif macd < 0 and mom1 < 0:
            momentum_state = "空头动能增强"
            momentum_desc = "MACD绿柱放大，动能向下"
        elif macd < 0 and mom1 > 0:
            momentum_state = "空头动能减弱（关注）"
            momentum_desc = "MACD绿柱缩短，存在底背离可能"
        else:
            momentum_state = "动能中性"
            momentum_desc = "MACD动能变化不明显"

        report.append(f"- **状态**：{momentum_state}")
        report.append(f"- **说明**：{momentum_desc}")

        # RSI斜率判断
        if rsi > 70:
            rsi_state = "超买区"
        elif rsi < 30:
            rsi_state = "超卖区"
        elif rsi > 60:
            rsi_state = "偏强"
        elif rsi < 40:
            rsi_state = "偏弱"
        else:
            rsi_state = "中性"
        report.append(f"- **RSI(24)**：{rsi:.1f}（{rsi_state}）")
        report.append("")

        # 3. 位置状态
        report.append("### 📍 位置状态")
        report.append("")

        close = latest_data.get('close', 0)
        boll_upper = latest_data.get('boll_upper', 0)
        boll_lower = latest_data.get('boll_lower', 0)
        cci = latest_data.get('cci', 0)
        wr = latest_data.get('wr', 50)

        # 布林带位置
        if boll_upper > boll_lower:
            boll_pct = (close - boll_lower) / (boll_upper - boll_lower) * 100
        else:
            boll_pct = 50

        if boll_pct > 90:
            boll_state = "极度超买（触及上轨）"
        elif boll_pct > 70:
            boll_state = "偏高"
        elif boll_pct < 10:
            boll_state = "极度超卖（触及下轨）"
        elif boll_pct < 30:
            boll_state = "偏低"
        else:
            boll_state = "中部"

        report.append(f"- **布林带**：{boll_state}（{boll_pct:.0f}%位置）")

        # CCI状态
        if cci > 200:
            cci_state = "严重超买"
        elif cci > 100:
            cci_state = "超买区"
        elif cci < -200:
            cci_state = "严重超卖"
        elif cci < -100:
            cci_state = "超卖区"
        else:
            cci_state = "正常区间"
        report.append(f"- **CCI(14)**：{cci:.1f}（{cci_state}）")

        # WR状态
        if wr < 10:
            wr_state = "严重超买"
        elif wr < 20:
            wr_state = "接近超买"
        elif wr > 90:
            wr_state = "严重超卖"
        elif wr > 80:
            wr_state = "接近超卖"
        else:
            wr_state = "正常区间"
        report.append(f"- **WR(14)**：{wr:.1f}（{wr_state}）")
        report.append("")

        # 4. 形态特权区
        report.append("### 🕯️ 形态特权区（定性分析）")
        report.append("")

        pattern_type = pattern_assessment.get('type', 'none')
        pattern_strength = pattern_assessment.get('strength', 0)
        patterns = pattern_assessment.get('patterns', [])
        position = position_info.get('position', 'middle')

        # 关键修复：降低显示阈值，强度>=1（弱形态）也显示
        # 之前的阈值>=3过高，导致"中"强度形态（如长上影线）被忽略
        if pattern_strength >= 1 and patterns:
            pattern_name = patterns[0] if patterns else '形态'

            # 强度描述
            strength_desc = "强" if pattern_strength >= 4 else "中" if pattern_strength >= 2 else "弱"

            if position == 'low' and pattern_type == 'bullish_reversal':
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：低位（{position_info.get('price_ratio', 50):.0f}%分位）")
                if pattern_strength >= 3:
                    report.append(f"- **定性影响**：🔥 强力支撑信号 - 底部形态在低位形成，可靠性高")
                else:
                    report.append(f"- **定性影响**：✅ 偏多信号 - 低位出现看涨形态，值得关注")
            elif position == 'high' and pattern_type == 'bearish_reversal':
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：高位（{position_info.get('price_ratio', 50):.0f}%分位）")
                if pattern_strength >= 3:
                    report.append(f"- **定性影响**：⚠️ 顶部警示信号 - 顶部形态在高位形成，需警惕回调")
                else:
                    report.append(f"- **定性影响**：⚠️ 偏空信号 - 高位出现看跌形态，保持谨慎")
            elif position == 'high' and pattern_type == 'bullish_reversal':
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：高位（{position_info.get('price_ratio', 50):.0f}%分位）")
                report.append(f"- **定性影响**：⚠️ 中性偏警惕 - 高位出现看涨形态可能是诱多/力竭")
            elif position == 'low' and pattern_type == 'bearish_reversal':
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：低位（{position_info.get('price_ratio', 50):.0f}%分位）")
                report.append(f"- **定性影响**：➖ 中性 - 低位看跌形态可能是洗盘/诱空")
            elif pattern_type == 'neutral':
                # 中性形态（如长上影线、长下影线等）
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：{position}")
                report.append(f"- **定性影响**：➖ 中性信号 - {pattern_assessment.get('description', '需结合其他信号判断')}")
            else:
                report.append(f"- **形态**：{pattern_name}（强度：{strength_desc}）")
                report.append(f"- **位置权重**：{position}")
                report.append(f"- **定性影响**：形态存在但位置权重一般")
        else:
            report.append("- **形态**：无明显形态")
            report.append("- **定性影响**：当前无显著K线形态信号")

        # 新增：绘制近期K线图
        if df is not None and len(df) >= 5:
            report.append("")
            report.append("#### 📊 近期K线图")
            report.append("")
            report.append("```")
            report.append(draw_candlestick_chart(df, num_candles=10, width_per_candle=6, height=13))
            report.append("```")
            report.append("")
            report.append("> 图例：🟢 绿色 = 阳线（涨） | 🔴 红色 = 阴线（跌） | │ = 影线")

        # 新增：如果有识别到形态，显示形态示意图
        if pattern_strength >= 1 and patterns:
            pattern_name = patterns[0]
            illustration = draw_pattern_illustration(pattern_name)
            if illustration and len(illustration.strip()) > 10:
                report.append("")
                report.append(f"#### 📖 「{pattern_name}」形态说明")
                report.append("")
                report.append("```")
                report.append(illustration)
                report.append("```")

        report.append("")
        return report

    def _generate_score_breakdown(self, scoring: Dict) -> List[str]:
        """
        生成第三部分：量化评分与因子拆解（重构版）

        新架构：最终评分 = 趋势得分 × 位置修正系数
        """
        report = []
        report.append("## 第三部分：量化评分与因子拆解")
        report.append("")

        score = scoring.get('score', 0)
        score_grade = scoring.get('score_grade', 'N/A')
        trend_score = scoring.get('trend_score', score)
        position_modifier = scoring.get('position_modifier', 1.0)
        factors_raw = scoring.get('factors_raw', {})
        factors_score = scoring.get('factors_score', {})

        # 综合评分（展示计算公式）
        report.append(f"### 📊 综合评分：{score:.1f} 分（{score_grade}）")
        report.append("")
        report.append(f"> **计算公式**：最终评分 = 趋势得分 × 位置修正系数")
        report.append(f"> `{trend_score:.1f} × {position_modifier:.2f} = {score:.1f}`")
        report.append("")

        # 趋势得分分解
        report.append("### 趋势因子得分")
        report.append("")
        report.append("| 因子 | 权重 | 得分 | 说明 |")
        report.append("|------|------|------|------|")

        # 新的因子定义（与 scoring_system.py TREND_FACTOR_WEIGHTS 一致）
        factor_info = {
            'trend_strength': ('趋势强度', 0.20, '股价相对MA20位置，3-8%最佳'),
            'ma_slope': ('均线斜率', 0.20, '斜率向上且陡峭为佳'),
            'macd_momentum': ('MACD动量', 0.20, '柱状图扩大为佳'),
            'money_flow': ('资金流向', 0.20, 'OBV持续流入为佳'),
            'volume_ratio': ('成交量', 0.10, '温和放量1.2-2倍为佳'),
            'candlestick_pattern': ('K线形态', 0.10, '位置敏感评分，低位看涨加分'),
        }

        for key, (name, weight, desc) in factor_info.items():
            score_val = factors_score.get(key, 0)
            # factors_score 已经是 0-100 的分数
            report.append(f"| {name} | {weight*100:.0f}% | {score_val:.1f} | {desc} |")

        report.append("")

        # 位置修正系数说明
        report.append("### 📐 位置修正系数")
        report.append("")

        # 修正系数分解展示
        warnings = scoring.get('warnings', [])
        if warnings and position_modifier < 1.0:
            report.append("**系数分解：**")
            report.append("")
            report.append(f"- 基础系数：1.00")
            for warning in warnings[:4]:  # 最多显示4个惩罚项
                report.append(f"- {warning}")
            report.append(f"- **最终系数：{position_modifier:.2f}**")
            report.append("")

        report.append(f"**当前修正系数：{position_modifier:.2f}**")
        report.append("")

        # 修正系数解读
        if position_modifier >= 0.95:
            risk_level = "🟢 安全区"
            risk_desc = "入场位置安全，可正常操作"
        elif position_modifier >= 0.75:
            risk_level = "🟡 适中区"
            risk_desc = "位置适中，需谨慎"
        elif position_modifier >= 0.5:
            risk_level = "🟠 警戒区"
            risk_desc = "技术指标超买，风险增加"
        else:
            risk_level = "🔴 危险区"
            risk_desc = "严重超买，不建议追高"

        report.append(f"| 风险等级 | 修正系数 | 说明 |")
        report.append(f"|----------|----------|------|")
        report.append(f"| {risk_level} | {position_modifier:.2f} | {risk_desc} |")
        report.append("")

        # 修正系数参考表
        report.append("| 位置状态 | 修正系数范围 | 典型触发条件 |")
        report.append("|----------|--------------|--------------|")
        report.append("| 🟢 安全区 | 0.95~1.00 | 布林中下轨，RSI 30-50 |")
        report.append("| 🟡 适中区 | 0.75~0.95 | 布林中上轨，RSI 50-65 |")
        report.append("| 🟠 警戒区 | 0.50~0.75 | 布林上轨附近，RSI 65-75 |")
        report.append("| 🔴 危险区 | <0.50 | 布林上轨外，RSI>75，WR<10 |")
        report.append("")

        # 红黑榜（基于新因子，包含K线形态）
        report.append("### 🔴🟢 红黑榜")
        report.append("")

        factor_scores_list = [
            ('trend_strength', '趋势强度', factors_score.get('trend_strength', 50)),
            ('ma_slope', '均线斜率', factors_score.get('ma_slope', 50)),
            ('macd_momentum', 'MACD动量', factors_score.get('macd_momentum', 50)),
            ('money_flow', '资金流向', factors_score.get('money_flow', 50)),
            ('volume_ratio', '成交量', factors_score.get('volume_ratio', 50)),
            ('candlestick_pattern', 'K线形态', factors_score.get('candlestick_pattern', 50)),
        ]

        # 确保分数是百分制
        factor_scores_list = [(k, n, v*100 if v <= 1.0 else v) for k, n, v in factor_scores_list]
        factor_scores_list.sort(key=lambda x: x[2], reverse=True)

        # 红榜（得分高的）
        strong_factors = [f for f in factor_scores_list if f[2] >= 70]
        if strong_factors:
            report.append("**🟢 核心支撑：**")
            for key, name, val in strong_factors[:3]:
                report.append(f"- {name}（得分{val:.0f}分）")
            report.append("")

        # 黑榜（得分低的）
        weak_factors = [f for f in factor_scores_list if f[2] <= 40]
        if weak_factors:
            report.append("**🔴 主要拖累：**")
            for key, name, val in weak_factors[:3]:
                report.append(f"- {name}（得分{val:.0f}分）")
            report.append("")

        # K线形态详情展示（位置敏感评估）
        candlestick_detail = factors_raw.get('candlestick_detail', {})
        if candlestick_detail and candlestick_detail.get('patterns'):
            report.append("### 🕯️ K线形态分析（位置敏感）")
            report.append("")

            position_zone = candlestick_detail.get('position_zone', 'mid_position')
            position_desc = candlestick_detail.get('position_desc', '中位区域')
            long_term_pos = candlestick_detail.get('long_term_position', 'mid')
            short_term_pos = candlestick_detail.get('short_term_position', 'mid')
            assessment = candlestick_detail.get('assessment', '')

            # 长短期位置说明（关键修复：区分长短期）
            long_emoji = {'low': '🟢', 'mid': '🟡', 'high': '🔴'}.get(long_term_pos, '⚪')
            short_emoji = {'low': '🟢', 'mid': '🟡', 'high': '🔴'}.get(short_term_pos, '⚪')
            long_desc = {'low': '低位', 'mid': '中位', 'high': '高位'}.get(long_term_pos, '中位')
            short_desc = {'low': '超卖', 'mid': '正常', 'high': '超买'}.get(short_term_pos, '正常')

            report.append(f"**位置分析：**")
            report.append(f"- 长期位置：{long_emoji} {long_desc}（基于60日分位）")
            report.append(f"- 短期位置：{short_emoji} {short_desc}（基于乖离率/布林带）")
            report.append(f"- 综合判定：{position_desc}")
            report.append("")

            # 形态详情
            patterns = candlestick_detail.get('patterns', [])
            if patterns:
                report.append("| 形态名称 | 类型 | 强度 | 基础权重 | 位置修正 | 最终得分 |")
                report.append("|----------|------|------|----------|----------|----------|")
                for p in patterns[:5]:  # 最多显示5个形态
                    name = p.get('name', '未知')
                    p_type = '看涨' if p.get('type') == 'bullish' else '看跌' if p.get('type') == 'bearish' else '中性'
                    strength = p.get('strength', '中')
                    base_weight = p.get('base_weight', 0)
                    modifier = p.get('modifier', 1.0)
                    final_score = p.get('final_score', 0)
                    report.append(f"| {name} | {p_type} | {strength} | {base_weight:+.1f} | ×{modifier:.1f} | {final_score:+.1f} |")
                report.append("")

            # 评估结论
            if assessment:
                report.append(f"**评估结论：** {assessment}")
                report.append("")

            # 位置敏感逻辑说明
            report.append("> **位置敏感逻辑：**")
            report.append("> - 长期低位+短期低位+看涨 = 黄金坑机会")
            report.append("> - 长期低位+短期高位+看涨 = 警惕诱多，等待回踩")
            report.append("> - 高位+看跌 = 强力减分")
            report.append("> - 低位+看跌 = 可能洗盘")
            report.append("")

        return report

    def _generate_trading_plan(self, scoring: Dict, position_info: Dict,
                                pattern_assessment: Dict, latest_data: pd.Series) -> List[str]:
        """
        生成第四部分：交易执行计划
        """
        report = []
        report.append("## 第四部分：交易执行计划")
        report.append("")

        score = scoring.get('score', 50)
        trigger_type = scoring.get('trigger_type', 'none')
        execution = scoring.get('execution', {})
        confidence = scoring.get('confidence', '中')

        close = latest_data.get('close', 0)
        ma20 = latest_data.get('ma_20', 0)
        ma50 = latest_data.get('ma_50', 0)
        ma200 = latest_data.get('ma_200', 0)
        boll_upper = latest_data.get('boll_upper', 0)
        boll_lower = latest_data.get('boll_lower', 0)

        position = position_info.get('position', 'middle')
        pattern_strength = pattern_assessment.get('strength', 0)

        # 1. 策略类型判定
        report.append("### 📈 策略类型")
        report.append("")

        # 获取趋势方向和评分
        trend_score = scoring.get('trend_score', score)
        trend_direction = scoring.get('trend_direction', 'sideways')

        # 策略类型判断逻辑（重构版 - 与操作指令联动）
        # 左侧交易：主动预测拐点，在趋势反转前介入（高风险）
        # 右侧交易：跟随已确认的趋势，在趋势确立后介入（较安全）

        # 获取熔断状态 - 优先检查
        action_guide = execution.get('action_guide', '')
        is_fuse_triggered = '熔断' in action_guide

        # 关键修复：策略类型必须与第一部分的操作指令(action_text)保持一致
        # 需要重新计算 action_text（与 _generate_core_conclusion 保持一致）
        position_modifier = scoring.get('position_modifier', 1.0)
        factors_raw = scoring.get('factors_raw', {})
        candlestick_detail = factors_raw.get('candlestick_detail', {})
        long_term_pos = candlestick_detail.get('long_term_position', 'mid')
        short_term_pos = candlestick_detail.get('short_term_position', 'mid')
        pattern_type = pattern_assessment.get('type', 'none')
        pattern_strength_val = int(pattern_assessment.get('strength', 0))

        # 计算 action_text（与第一部分逻辑一致）
        if is_fuse_triggered:
            if '熔断-回避' in action_guide:
                action_text = "回避/卖出"
            elif '熔断-诱多' in action_guide:
                action_text = "观望（警惕诱多）"
            else:
                action_text = "观望"
        elif '风险警告' in action_guide:
            action_text = "谨慎观望"
        elif long_term_pos == 'low' and short_term_pos == 'low' and pattern_type in ['bullish_reversal', 'bullish_continuation'] and pattern_strength_val >= 3:
            action_text = "试探性买入"
        elif short_term_pos == 'high' and pattern_type in ['bearish_reversal', 'bearish_continuation'] and pattern_strength_val >= 3:
            action_text = "卖出/减仓"
        elif short_term_pos == 'high' and pattern_type in ['bullish_reversal', 'bullish_continuation']:
            action_text = "观望（警惕诱多）"
        elif position_modifier < 0.5:
            action_text = "不宜追高"
        elif position_modifier < 0.7:
            action_text = "谨慎观望"
        elif score >= 65:
            action_text = "买入"
        elif score >= 50:
            action_text = "观望"
        else:
            action_text = "不建议"

        if is_fuse_triggered:
            # 熔断触发 - 风控/持币观望
            if '熔断-回避' in action_guide:
                strategy_type = "🚫 风控（回避）"
                strategy_desc = "触发熔断机制，高位顶部信号，建议观望或卖出"
            elif '熔断-诱多' in action_guide:
                strategy_type = "⚠️ 风控（警惕诱多）"
                strategy_desc = "触发熔断机制，高位看涨形态可能是诱多陷阱"
            else:
                strategy_type = "🛡️ 风控（持币观望）"
                strategy_desc = "触发熔断机制，建议持币观望"
        elif action_text in ["买入", "试探性买入"]:
            # 操作指令是买入 - 必须显示买入相关策略
            if trigger_type == 'breakout':
                strategy_type = "✅ 右侧交易（突破买入）"
                strategy_desc = "价格突破关键位置，趋势启动信号明确"
            elif trigger_type == 'pullback':
                strategy_type = "✅ 右侧交易（回调买入）"
                strategy_desc = "上涨趋势中的回踩，逢低布局机会"
            elif position == 'low' and pattern_strength >= 3:
                strategy_type = "✅ 左侧交易（底部博弈）"
                strategy_desc = "低位出现底部形态，博弈趋势反转"
            elif trend_direction == 'up':
                strategy_type = "✅ 右侧交易（趋势跟随）"
                strategy_desc = "上涨趋势确立，跟随趋势买入"
            else:
                strategy_type = "✅ 信号驱动（逢低布局）"
                strategy_desc = "综合评分良好，当前位置适合布局"
        elif action_text in ["谨慎观望", "观望（警惕诱多）"]:
            strategy_type = "⚠️ 谨慎观望"
            strategy_desc = "存在风险信号，建议等待更明确的机会"
        elif action_text == "观望":
            strategy_type = "🔍 观望型"
            strategy_desc = "信号不明确，等待更好的入场机会"
        elif action_text in ["不宜追高", "卖出/减仓", "不建议"]:
            strategy_type = "🛡️ 防御型"
            strategy_desc = "当前风险较高，建议规避或减仓"
        elif position == 'high':
            # 高位区域 - 防御为主
            strategy_type = "🛡️ 防御型"
            strategy_desc = "当前位置偏高，以风险控制为主，不宜追高"
        elif trend_direction == 'up' and trend_score >= 55:
            # 上涨趋势确认 - 右侧交易
            if position == 'low':
                strategy_type = "✅ 右侧交易（回调买入）"
                strategy_desc = "上涨趋势中的回调，趋势确立后的低吸机会"
            else:
                strategy_type = "✅ 右侧交易（趋势跟随）"
                strategy_desc = "上涨趋势确立，跟随趋势操作"
        elif trend_direction == 'down' and trend_score < 45:
            # 下跌趋势 - 谨慎
            if position == 'low' and pattern_strength >= 3:
                strategy_type = "⚠️ 左侧交易（博反弹）"
                strategy_desc = "下跌趋势中的底部形态，尝试抄底博反弹（风险较高）"
            else:
                strategy_type = "🚫 观望型"
                strategy_desc = "下跌趋势未结束，等待趋势企稳信号"
        elif trigger_type == 'breakout':
            # 突破信号 - 右侧交易
            strategy_type = "✅ 右侧交易（突破买入）"
            strategy_desc = "价格突破关键位置，趋势可能启动"
        elif position == 'low' and pattern_strength >= 3:
            # 低位+形态 - 可能是左侧交易
            if trend_direction == 'sideways':
                strategy_type = "⚠️ 左侧交易（形态博弈）"
                strategy_desc = "震荡底部的形态信号，博弈趋势反转"
            else:
                strategy_type = "✅ 右侧交易（底部确认）"
                strategy_desc = "底部形态确认，趋势可能反转向上"
        else:
            # 默认观望
            strategy_type = "🔍 观望型"
            strategy_desc = "信号不明确，等待更好的入场机会"

        report.append(f"- **类型**：{strategy_type}")
        report.append(f"- **说明**：{strategy_desc}")
        report.append("")

        # 2. 具体点位（根据风险状态调整）
        report.append("### 📍 具体点位")
        report.append("")

        # 检查是否极端超买（高风险状态）
        # 注意：action_guide 在前面已定义，用于策略类型判断
        position_modifier = scoring.get('position_modifier', 1.0)
        warnings_list = scoring.get('warnings', [])
        is_extreme_risk = (
            position_modifier < 0.5 or
            any('极端' in w or '熔断' in w for w in warnings_list) or
            '熔断' in action_guide
        )

        if is_extreme_risk:
            # 极端风险：不显示入场区间，显示风险提示
            report.append("| ⚠️ 风险提示 | **当前不宜入场** | 指标严重超买，追高风险极高 |")
            report.append("")
            report.append(f"| 建议操作 | 观望/止盈 | 等待回调至安全区间 |")
            if ma20 > 0 and ma20 < close:
                report.append(f"| 回调买点 | ¥{ma20:.2f}附近 | 等待回踩MA20 |")
            if boll_lower > 0:
                report.append(f"| 安全买点 | ¥{boll_lower:.2f}附近 | 等待布林下轨 |")

            # 止损位（如果持有仓位）
            stop_price = execution.get('stop_price', close * 0.95)
            report.append(f"| 止盈/止损 | ¥{stop_price:.2f} | 若持有，设止盈保护利润 |")
        else:
            # 正常情况：显示入场区间
            report.append("| 项目 | 建议数值 | 说明 |")
            report.append("|------|----------|------|")

            # 入场区间（使用ATR计算合理区间）
            atr = latest_data.get('atr', close * 0.02)  # 默认2%波动
            if atr <= 0 or pd.isna(atr):
                atr = close * 0.02

            # 根据趋势方向调整入场区间
            trend_direction = scoring.get('trend_direction', 'sideways')
            if trend_direction == 'up':
                # 上涨趋势：入场区间在现价下方
                entry_low = max(close - atr, close * 0.97)
                entry_high = close
                entry_desc = "回调买入区间"
            elif trend_direction == 'down':
                # 下跌趋势：谨慎，入场区间更宽
                entry_low = close - atr * 1.5
                entry_high = close + atr * 0.5
                entry_desc = "分批试探区间（谨慎）"
            else:
                # 横盘：围绕现价
                entry_low = close - atr * 0.5
                entry_high = close + atr * 0.5
                entry_desc = "震荡区间内分批建仓"

            report.append(f"| 入场区间 | ¥{entry_low:.2f} ~ ¥{entry_high:.2f} | {entry_desc} |")

            # 触发加仓
            if boll_upper > 0:
                report.append(f"| 触发加仓 | 突破¥{boll_upper:.2f} | 突破布林上轨且放量 |")
            else:
                report.append(f"| 触发加仓 | 突破¥{close*1.05:.2f} | 突破近期高点5% |")

            # 止损位
            stop_price = execution.get('stop_price', close * 0.95)
            if pattern_strength >= 3 and pattern_assessment.get('stop_loss'):
                stop_price = pattern_assessment.get('stop_loss', stop_price)
                stop_desc = "基于形态最低点"
            else:
                stop_desc = "固定5%止损"
            report.append(f"| 止损位 | ¥{stop_price:.2f} | {stop_desc} |")

            # 目标位（区分上涨趋势和下跌趋势）
            targets = []

            # 根据趋势方向决定显示逻辑
            if trend_direction == 'down':
                # 下跌趋势：显示需要突破的阻力位
                report.append("| 操作提示 | 下跌趋势中 | 需突破阻力确认反转 |")
                if boll_upper > 0 and boll_upper > close:
                    report.append(f"| 首要阻力 | ¥{boll_upper:.2f} | 突破布林上轨 |")
                if ma20 > 0 and ma20 > close:
                    report.append(f"| 次要阻力 | ¥{ma20:.2f} | 突破MA20确认 |")
                if ma50 > 0 and ma50 > close:
                    report.append(f"| 关键阻力 | ¥{ma50:.2f} | 突破MA50转强 |")
                # 支撑位
                if boll_lower > 0 and boll_lower < close:
                    report.append(f"| 下档支撑 | ¥{boll_lower:.2f} | 布林下轨支撑 |")
            else:
                # 上涨趋势或横盘：显示目标价
                if boll_upper > 0 and boll_upper > close:
                    targets.append(f"| 短期目标 | ¥{boll_upper:.2f} | 布林带上轨 |")
                if ma50 > 0 and ma50 > close:
                    targets.append(f"| 中期目标 | ¥{ma50:.2f} | MA50压力位 |")
                if ma200 > 0 and not pd.isna(ma200) and ma200 > close:
                    targets.append(f"| 长期目标 | ¥{ma200:.2f} | MA200趋势线 |")

                if targets:
                    for target in targets:
                        report.append(target)
                else:
                    # 股价已突破主要阻力
                    report.append("| 当前状态 | 股价强势 | 突破主要阻力 |")
                    # 显示支撑位
                    if ma20 > 0 and ma20 < close:
                        report.append(f"| 回调支撑 | ¥{ma20:.2f} | MA20支撑 |")
                    if ma50 > 0 and ma50 < close:
                        report.append(f"| 重要支撑 | ¥{ma50:.2f} | MA50支撑 |")

        report.append("")

        # 3. 仓位建议
        report.append("### 💰 仓位建议")
        report.append("")

        if confidence == "高":
            position_pct = "50-70%"
            position_desc = "高置信度信号，可积极建仓"
        elif confidence == "中高":
            position_pct = "30-50%"
            position_desc = "中高置信度，适度参与"
        elif confidence == "中":
            position_pct = "20-30%"
            position_desc = "中等置信度，轻仓试探"
        elif confidence == "中低":
            position_pct = "10-20%"
            position_desc = "置信度一般，极小仓位或观望"
        else:
            position_pct = "0-10%"
            position_desc = "低置信度，建议观望"

        report.append(f"- **建议仓位**：{position_pct}")
        report.append(f"- **说明**：{position_desc}")
        report.append("")

        # 4. 风险提示
        report.append("### ⚠️ 风险提示")
        report.append("")

        trend_score = scoring.get('trend_score', score)
        position_modifier = scoring.get('position_modifier', 1.0)
        factors_score = scoring.get('factors_score', {})

        # 风险分类收集
        high_risks = []  # 高风险
        medium_risks = []  # 中风险
        low_risks = []  # 低风险

        # ========== 高风险检测 ==========
        # 风险1：接飞刀风险（趋势极弱 + 位置安全）
        if trend_score < 40 and position_modifier >= 0.95:
            high_risks.append({
                'title': '接飞刀行情',
                'desc': '当前处于强下跌趋势，虽然位置极低，但属于"接飞刀"行情。资金持续流出，反弹难度极大。',
                'advice': '放弃或极小仓位试探，切勿重仓抄底'
            })

        # 风险2：追高风险（位置危险）
        if position_modifier < 0.5:
            high_risks.append({
                'title': '追高被套',
                'desc': '当前位置严重超买，技术指标处于极端高位。短期回调概率大，追高风险极高。',
                'advice': '观望等待回调，或极小仓位试探'
            })

        # ========== 中风险检测 ==========
        # 风险3：高位诱多风险（高位 + 看涨形态）
        if position_modifier < 0.6 and pattern_strength >= 3:
            pattern_type = pattern_assessment.get('type', '')
            if pattern_type in ['bullish_reversal', 'bullish_continuation']:
                medium_risks.append({
                    'title': '高位诱多',
                    'desc': '高位出现看涨形态，需警惕主力诱多出货。如果成交量异常放大，可能是最后的拉升。',
                    'advice': '谨慎参与，设置紧止损'
                })

        # 风险4：假突破风险（突破但量能不足）
        volume_score = factors_score.get('volume_ratio', 50)
        if trigger_type == 'breakout' and volume_score < 50:
            medium_risks.append({
                'title': '假突破',
                'desc': '价格突破但成交量未能有效配合。假突破概率较大，可能快速回落。',
                'advice': '等待放量确认后再入场'
            })

        # 风险5：顶部背离风险（MACD背离）
        macd_momentum = factors_score.get('macd_momentum', 50)
        if position_modifier < 0.7 and macd_momentum < 40:
            medium_risks.append({
                'title': '顶部背离',
                'desc': '价格处于高位但MACD动能减弱。存在顶背离可能，趋势可能反转。',
                'advice': '减仓或设置止盈保护利润'
            })

        # 风险6：流动性风险（成交量过低）
        if volume_score < 35:
            medium_risks.append({
                'title': '流动性不足',
                'desc': '成交量严重萎缩，流动性差。买卖价差大，滑点成本高。',
                'advice': '小仓位或等待放量后再参与'
            })

        # ========== 低风险检测 ==========
        # 风险7：均线压制风险（重要均线在上方，仅当位置低时）
        if close < ma20 and close < ma50 and ma20 > 0 and ma50 > 0 and position_modifier >= 0.7:
            low_risks.append({
                'title': '均线压制',
                'desc': f'MA20(¥{ma20:.2f})和MA50(¥{ma50:.2f})在上方形成压力。突破需要放量配合，否则可能受阻回落。',
                'advice': '在均线附近观察量能变化'
            })

        # 风险8：弱势企稳（合并原"弱势反弹"和"下跌中继"）
        if 40 <= trend_score < 50 and position_modifier >= 0.85:
            low_risks.append({
                'title': '弱势企稳',
                'desc': '趋势偏弱但位置较低，可能是下跌中继或弱势反弹。真正的趋势反转尚未确认。',
                'advice': '轻仓试探或观望，严格止损'
            })

        # ========== 输出风险提示（按优先级）==========
        # 高风险：最多显示1个
        if high_risks:
            risk = high_risks[0]
            report.append(f"- **【高风险：{risk['title']}】**")
            report.append(f"  - {risk['desc']}")
            report.append(f"  - **建议**：{risk['advice']}")

        # 中风险：最多显示2个
        for risk in medium_risks[:2]:
            report.append(f"- **【中风险：{risk['title']}】**")
            report.append(f"  - {risk['desc']}")
            report.append(f"  - **建议**：{risk['advice']}")

        # 低风险：最多显示2个（仅在没有高风险时显示）
        if not high_risks:
            for risk in low_risks[:2]:
                report.append(f"- **【低风险：{risk['title']}】**")
                report.append(f"  - {risk['desc']}")
                report.append(f"  - **建议**：{risk['advice']}")

        # 添加系统级警告（来自scoring_system）
        warnings = scoring.get('warnings', [])
        if warnings:
            for warning in warnings[:2]:  # 最多显示2个系统警告
                report.append(f"- {warning}")

        # 如果没有任何风险提示
        if not high_risks and not medium_risks and not low_risks and not warnings:
            report.append("- 当前风险可控，建议按计划执行")
            if position_modifier < 0.7:
                report.append("- 注意：位置略高，建议设置好止损保护")

        report.append("")
        return report


# Example usage
if __name__ == "__main__":
    analyzer = StockAnalyzer()

    # Example: Analyze a stock
    symbol = input("Enter stock symbol (e.g., 601777): ").strip()
    if symbol:
        report = analyzer.analyze_stock(symbol)
        print(report)

        # Optionally save to file
        filename = f"{symbol}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\nAnalysis report saved to {filename}")