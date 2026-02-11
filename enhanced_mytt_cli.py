#!/usr/bin/env python3
"""
Enhanced CLI Interface with MyTT Technical Indicators
Extends the unified CLI interface with advanced MyTT-based analysis
"""

import sys
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cli_interface import UnifiedCLIInterface
from quant_trade_a_share.utils.mytt_indicators import calculate_mytt_indicators


class EnhancedMyTTCLIInterface(UnifiedCLIInterface):
    """
    Enhanced CLI interface with MyTT technical indicators integration
    """

    def __init__(self, tushare_token, eastmoney_cookie):
        super().__init__(tushare_token, eastmoney_cookie)
        print("🎯 已启用MyTT高级技术指标分析")

    def enhanced_analyze_stock(self):
        """
        Enhanced stock analysis with MyTT indicators
        """
        symbol = input("请输入股票代码 (例: sh600519): ").strip()
        if not symbol:
            print("❌ 股票代码不能为空")
            return

        strategy_name = input("请输入策略名称 (ma_crossover/rsi/macd/bollinger/mean_reversion/breakout，默认: ma_crossover): ").strip() or 'ma_crossover'

        # Ask for data source
        source = input("请选择数据源 (eastmoney/ashare/tushare/baostock, 默认: auto (优先使用Ashare)): ").strip()
        source = source if source in ['eastmoney', 'ashare', 'tushare', 'baostock', 'auto'] else 'auto'

        print(f"\n📊 使用MyTT指标分析股票 {symbol} 使用 {strategy_name} 策略...")
        print(f"📈 使用数据源: {source}")

        try:
            # Get stock data using DataFetcher with selected source
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

            data = self.data_fetcher.fetch(symbol, start_date, end_date, source=source)
            if data is None or data.empty:
                print(f"❌ 无法从{source}获取 {symbol} 的数据，尝试使用screener...")
                # Fallback to screener with specified source
                data = self.screener.fetch_stock_data(symbol, period='180', data_source=source)
                if data is None or data.empty:
                    print(f"⚠️  也无法从screener获取 {symbol} 数据，跳过")
                    return

            # Get stock name
            if self.screener.chinese_stocks is None:
                self.screener.get_chinese_stocks_list()

            stock_info = self.screener.chinese_stocks[self.screener.chinese_stocks['symbol'] == symbol] if self.screener.chinese_stocks is not None else pd.DataFrame()
            stock_name = symbol  # Default to symbol if name not found
            if not stock_info.empty and 'name' in stock_info.columns:
                stock_name = stock_info['name'].iloc[0]

            # Calculate MyTT indicators
            print("📈 计算MyTT技术指标...")
            data = calculate_mytt_indicators(data)

            # Get strategy
            strategy = self.strategy_manager.get_strategy(strategy_name)
            if strategy is None:
                print(f"❌ 策略 {strategy_name} 不存在")
                return

            # Generate signals
            signals = strategy.generate_signals(data)

            # Calculate recent performance if data is sufficient
            if not data.empty and len(data) > 0:
                recent_performance = ((data['close'].iloc[-1] - data['close'].iloc[0]) /
                                     data['close'].iloc[0]) * 100
                current_price = data['close'].iloc[-1]
                # Calculate 20-day and 60-day performances
                perf_20d = ((data['close'].iloc[-1] - data['close'].iloc[-20]) / data['close'].iloc[-20]) * 100 if len(data) >= 20 else 0
                perf_60d = ((data['close'].iloc[-1] - data['close'].iloc[-60]) / data['close'].iloc[-60]) * 100 if len(data) >= 60 else 0
            else:
                recent_performance = 0
                perf_20d = 0
                perf_60d = 0
                current_price = 0
                print(f"⚠️  {symbol} 数据不足，无法计算近期表现")

            # Extract MyTT indicators
            latest = data.iloc[-1]

            # Extract indicators from the enhanced dataset
            rsi6 = latest['rsi6'] if 'rsi6' in data.columns else 0
            rsi12 = latest['rsi12'] if 'rsi12' in data.columns else 0
            rsi24 = latest['rsi24'] if 'rsi24' in data.columns else 0

            macd = latest['macd_dif'] if 'macd_dif' in data.columns else 0
            macd_signal = latest['macd_dea'] if 'macd_dea' in data.columns else 0
            macd_histogram = latest['macd_bar'] if 'macd_bar' in data.columns else 0

            kdj_k = latest['kdj_k'] if 'kdj_k' in data.columns else 0
            kdj_d = latest['kdj_d'] if 'kdj_d' in data.columns else 0
            kdj_j = latest['kdj_j'] if 'kdj_j' in data.columns else 0

            wr1 = latest['wr1'] if 'wr1' in data.columns else 0
            wr2 = latest['wr2'] if 'wr2' in data.columns else 0

            cci = latest['cci'] if 'cci' in data.columns else 0

            ma_5 = latest['ma5'] if 'ma5' in data.columns else 0
            ma_10 = latest['ma10'] if 'ma10' in data.columns else 0
            ma_20 = latest['ma20'] if 'ma20' in data.columns else 0
            ma_30 = latest['ma30'] if 'ma30' in data.columns else 0
            ma_60 = latest['ma60'] if 'ma60' in data.columns else 0

            bb_upper = latest['boll_upper'] if 'boll_upper' in data.columns else 0
            bb_lower = latest['boll_lower'] if 'boll_lower' in data.columns else 0
            bb_middle = latest['boll_mid'] if 'boll_mid' in data.columns else 0

            atr = latest['atr'] if 'atr' in data.columns else 0

            volume_ratio = latest['volume_ratio'] if 'volume_ratio' in data.columns else 0
            volatility = latest['volatility'] if 'volatility' in data.columns else 0
            momentum = latest['momentum'] if 'momentum' in data.columns else 0
            roc = latest['roc'] if 'roc' in data.columns else 0

            # MyTT-specific indicators
            bias6 = latest['bias6'] if 'bias6' in data.columns else 0
            bias12 = latest['bias12'] if 'bias12' in data.columns else 0
            bias24 = latest['bias24'] if 'bias24' in data.columns else 0

            dmi_pdi = latest['dmi_pdi'] if 'dmi_pdi' in data.columns else 0
            dmi_mdi = latest['dmi_mdi'] if 'dmi_mdi' in data.columns else 0
            dmi_adx = latest['dmi_adx'] if 'dmi_adx' in data.columns else 0

            trix = latest['trix'] if 'trix' in data.columns else 0
            trma = latest['trma'] if 'trma' in data.columns else 0

            vr = latest['vr'] if 'vr' in data.columns else 0
            cr = latest['cr'] if 'cr' in data.columns else 0

            obv = latest['obv'] if 'obv' in data.columns else 0
            mfi = latest['mfi'] if 'mfi' in data.columns else 0

            ema12 = latest['ema12'] if 'ema12' in data.columns else 0
            ema26 = latest['ema26'] if 'ema26' in data.columns else 0
            ema50 = latest['ema50'] if 'ema50' in data.columns else 0

            # Calculate additional analysis metrics
            price_to_ma20 = (current_price / ma_20 - 1) * 100 if ma_20 != 0 else 0
            price_position_bb = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
            volume_change = volume_ratio - 1 if volume_ratio != 0 else 0

            # Calculate trend
            trend = "上升" if current_price > ma_20 else "下降"

            # Analyze signal
            signal_text = "📈 买入" if signals.iloc[-1] == 1 else "🔴 卖出" if signals.iloc[-1] == -1 else "⏸️  持有"

            # Generate comprehensive analysis report with MyTT indicators
            print(f"\n" + "="*60)
            print(f"🏆 {symbol} ({stock_name}) MyTT增强分析报告")
            print("="*60)

            # Price and Performance Section
            print(f"💰 价格与表现:")
            print(f"   当前价格: ¥{current_price:.2f}")
            print(f"   180日涨幅: {recent_performance:+.2f}%")
            print(f"   60日涨幅: {perf_60d:+.2f}%")
            print(f"   20日涨幅: {perf_20d:+.2f}%")
            print(f"   当前趋势: {trend}")

            # Technical Indicators Section
            print(f"\n🔧 MyTT技术指标:")
            print(f"   RSI (6/12/24): {rsi6:.2f}/{rsi12:.2f}/{rsi24:.2f} ({' oversold' if rsi24 < 30 else ' overbought' if rsi24 > 70 else ' neutral'})")
            print(f"   MACD: {macd:.4f}, Signal: {macd_signal:.4f}, Histogram: {macd_histogram:.4f}")
            print(f"   KDJ: K:{kdj_k:.2f}, D:{kdj_d:.2f}, J:{kdj_j:.2f}")
            print(f"   威廉指标: WR1:{wr1:.2f}, WR2:{wr2:.2f}")
            print(f"   移动均线: MA5:{ma_5:.2f}, MA10:{ma_10:.2f}, MA20:{ma_20:.2f}, MA30:{ma_30:.2f}, MA60:{ma_60:.2f}")
            print(f"   指数均线: EMA12:{ema12:.2f}, EMA26:{ema26:.2f}, EMA50:{ema50:.2f}")
            print(f"   布林带: 上轨{bb_upper:.2f}, 中轨{bb_middle:.2f}, 下轨{bb_lower:.2f}")
            print(f"   价格在布林带位置: {price_position_bb:.2f} ({'高位' if price_position_bb > 0.8 else '中位' if 0.2 <= price_position_bb <= 0.8 else '低位'})")
            print(f"   CCI: {cci:.2f}")
            print(f"   DMI: PDI:{dmi_pdi:.2f}, MDI:{dmi_mdi:.2f}, ADX:{dmi_adx:.2f}")
            print(f"   BIAS: 6日{bias6:.2f}%, 12日{bias12:.2f}%, 24日{bias24:.2f}%")
            print(f"   TRIX: {trix:.4f}, TRMA: {trma:.4f}")
            print(f"   VR: {vr:.2f}, CR: {cr:.2f}")
            print(f"   OBV: {obv:.2f}, MFI: {mfi:.2f}")
            print(f"   动量指标: {momentum:.4f}")
            print(f"   ROC (10日): {roc:.2f}%")
            print(f"   ATR (14日): {atr:.4f}")

            # Volume Analysis
            print(f"\n📊 成交量分析:")
            print(f"   量比: {volume_ratio:.2f} ({'放量' if volume_ratio > 1.5 else '缩量' if volume_ratio < 0.7 else '正常'})")
            print(f"   成交量变化: {volume_change:+.2f}%")

            # Risk Analysis
            print(f"\n⚠️  风险分析:")
            print(f"   波动率: {volatility:.4f} ({'高风险' if volatility > 0.04 else '中风险' if volatility > 0.02 else '低风险'})")
            print(f"   价格距离MA20: {price_to_ma20:+.2f}% ({'远离' if abs(price_to_ma20) > 10 else '合理'})")

            # Strategy Signal
            print(f"\n🎯 策略信号:")
            print(f"   {strategy_name.upper()} 策略信号: {signal_text}")
            print(f"   信号强度: {signals.iloc[-1] if len(signals) > 0 else 0}")
            print(f"   最近信号数: {len(signals[signals != 0]) if len(signals) > 0 else 0}")

            # MyTT-based Investment Recommendation Section
            print(f"\n💡 MyTT增强投资建议:")
            recommendation = self._enhanced_mytt_recommendation(
                rsi24, macd_histogram, price_position_bb, volume_ratio,
                volatility, current_price, ma_20, recent_performance,
                perf_20d, signals.iloc[-1] if len(signals) > 0 else 0,
                kdj_k, kdj_d, cci, rsi6, rsi12, bias6, dmi_adx, trix, obv, mfi
            )
            print(f"   {recommendation}")

            # MyTT-based Future Potential Assessment
            print(f"\n🚀 MyTT增强未来上涨潜力评估:")
            potential_score = self._enhanced_mytt_potential(
                rsi24, macd_histogram, price_position_bb, volume_ratio,
                volatility, recent_performance, perf_20d, momentum, roc,
                cci, kdj_k, kdj_d, bias6, dmi_adx, trix, vr, mfi
            )
            print(f"   潜力评分: {potential_score}/100")
            if potential_score >= 80:
                print(f"   🌟 极具上涨潜力")
            elif potential_score >= 60:
                print(f"   📈 有一定上涨潜力")
            elif potential_score >= 40:
                print(f"   ⚖️  潜力一般，观望")
            else:
                print(f"   📉 上涨潜力有限")

            # MyTT-based Buy/Sell Timing
            print(f"\n⏰ MyTT增强买卖时机分析:")
            timing_advice = self._enhanced_mytt_timing(
                rsi24, current_price, ma_5, ma_10, ma_20, bb_upper, bb_lower, bb_middle,
                macd, macd_signal, volume_ratio, roc,
                kdj_k, kdj_d, cci, atr, bias6, dmi_pdi, dmi_mdi, trix, cr, obv
            )
            print(f"   {timing_advice}")

            print("="*60)

            # Store in session
            self.session_data[f'analysis_{symbol}'] = {
                'symbol': symbol,
                'name': stock_name,
                'data': data,
                'signals': signals if len(signals) > 0 else pd.Series(dtype=float),
                'recent_performance': recent_performance,
                'technical_indicators': {
                    'rsi': rsi24,
                    'rsi6': rsi6,
                    'rsi12': rsi12,
                    'rsi24': rsi24,
                    'macd': macd,
                    'macd_signal': macd_signal,
                    'macd_histogram': macd_histogram,
                    'kdj_k': kdj_k,
                    'kdj_d': kdj_d,
                    'kdj_j': kdj_j,
                    'wr1': wr1,
                    'wr2': wr2,
                    'ma_5': ma_5,
                    'ma_10': ma_10,
                    'ma_20': ma_20,
                    'ma_30': ma_30,
                    'ma_60': ma_60,
                    'ema12': ema12,
                    'ema26': ema26,
                    'ema50': ema50,
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'cci': cci,
                    'atr': atr,
                    'volume_ratio': volume_ratio,
                    'volatility': volatility,
                    'momentum': momentum,
                    'roc': roc,
                    'bias6': bias6,
                    'bias12': bias12,
                    'bias24': bias24,
                    'dmi_pdi': dmi_pdi,
                    'dmi_mdi': dmi_mdi,
                    'dmi_adx': dmi_adx,
                    'trix': trix,
                    'trma': trma,
                    'vr': vr,
                    'cr': cr,
                    'obv': obv,
                    'mfi': mfi,
                    'price_to_ma20': price_to_ma20,
                    'price_position_bb': price_position_bb,
                    'volume_change': volume_change
                },
                'recommendation': recommendation,
                'potential_score': potential_score,
                'timing_advice': timing_advice
            }

        except Exception as e:
            print(f"❌ MyTT增强分析过程出错: {e}")
            import traceback
            traceback.print_exc()

    def _enhanced_mytt_recommendation(self, rsi, macd_hist, price_pos_bb, vol_ratio,
                                      volatility, current_price, ma_20, perf_long,
                                      perf_short, signal, kdj_k, kdj_d, cci, rsi6, rsi12,
                                      bias6, dmi_adx, trix, obv, mfi):
        """
        Enhanced investment recommendation using MyTT indicators
        """
        reasons = []

        # Multi-timeframe RSI Analysis
        rsi_avg = (rsi6 + rsi12 + rsi) / 3
        if rsi_avg < 30:
            reasons.append("多周期RSI平均值超卖，可能触底反弹")
        elif rsi_avg > 70:
            reasons.append("多周期RSI平均值超买，短期回调风险")
        else:
            reasons.append("RSI处于合理区间")

        # MACD Analysis
        if macd_hist > 0:
            reasons.append("MACD柱状图>0，看涨动能")
        else:
            reasons.append("MACD柱状图<0，看跌动能")

        # KDJ Analysis
        if kdj_k < 20 and kdj_d < 20:
            reasons.append("KDJ低位金叉，可能见底")
        elif kdj_k > 80 and kdj_d > 80:
            reasons.append("KDJ高位死叉，可能见顶")
        elif kdj_k > kdj_d:
            reasons.append("KDJ金叉向上，看涨")
        else:
            reasons.append("KDJ死叉向下，看跌")

        # CCI Analysis
        if cci < -100:
            reasons.append("CCI超卖，反转向上的可能性大")
        elif cci > 100:
            reasons.append("CCI超买，回调可能性大")
        else:
            reasons.append("CCI处于正常范围")

        # BIAS Analysis
        if abs(bias6) > 8:
            reasons.append("BIAS偏离较大，注意均值回归")
        elif abs(bias6) < 3:
            reasons.append("BIAS位置合理，处于正常波动范围")

        # DMI Analysis
        if dmi_adx > 25:
            reasons.append("DMI趋势强度较强")
        elif dmi_adx < 20:
            reasons.append("DMI趋势强度较弱，可能震荡")

        # TRIX Analysis
        if trix > 0:
            reasons.append("TRIX大于0，中长期趋势向上")
        else:
            reasons.append("TRIX小于0，中长期趋势向下")

        # Price Position in Bollinger Band
        if price_pos_bb < 0.2:
            reasons.append("价格在布林带下轨附近，估值偏低")
        elif price_pos_bb > 0.8:
            reasons.append("价格在布林带上轨附近，估值偏高")
        else:
            reasons.append("价格在布林带中位区域")

        # Volume Analysis
        if vol_ratio > 1.5:
            reasons.append("成交量放大，资金关注")
        elif vol_ratio < 0.7:
            reasons.append("成交量萎缩，缺乏关注")
        else:
            reasons.append("成交量正常")

        # Moving Average Trend
        if current_price > ma_20:
            reasons.append("价格站上20日线，中期趋势向好")
        else:
            reasons.append("价格跌破20日线，中期趋势向下")

        # Performance Analysis
        if perf_short > 0:
            reasons.append("短期表现强劲")
        else:
            reasons.append("短期表现疲弱")

        # Signal Analysis
        if signal == 1:
            reasons.append("策略给出买入信号")
        elif signal == -1:
            reasons.append("策略给出卖出信号")
        else:
            reasons.append("策略建议持有")

        # Generate overall recommendation
        strong_positive = sum(['强势' in r or '看涨' in r or '向上' in r or '买入' in r or
                               '反转向上的可能性大' in r or '估值偏低' in r or '趋势强度较强' in r for r in reasons])
        strong_negative = sum(['弱势' in r or '看跌' in r or '向下' in r or '卖出' in r or
                               '回调可能性大' in r or '估值偏高' in r or '回调风险' in r or
                               '趋势强度较弱' in r or '偏离较大' in r for r in reasons])

        if strong_positive > strong_negative + 1:
            return f"建议买入: {'; '.join(reasons)}"
        elif strong_negative > strong_positive + 1:
            return f"建议卖出: {'; '.join(reasons)}"
        else:
            return f"建议观望: {'; '.join(reasons)}"

    def _enhanced_mytt_potential(self, rsi, macd_hist, price_pos_bb, vol_ratio,
                               volatility, perf_long, perf_short, momentum, roc,
                               cci, kdj_k, kdj_d, bias6, dmi_adx, trix, vr, mfi):
        """
        Enhanced potential assessment using MyTT indicators
        """
        score = 50  # Base score

        # Multi-timeframe RSI contribution
        if 40 <= rsi <= 60:
            score += 10
        elif 30 <= rsi <= 70:
            score += 5
        elif rsi < 30:  # Oversold, potential rebound
            score += 8
        else:  # Overbought, less favorable
            score -= 5

        # MACD histogram positive
        if macd_hist > 0:
            score += 8
        elif macd_hist < 0:
            score -= 5

        # Price position in Bollinger band (favorable if not too high)
        if 0.2 <= price_pos_bb <= 0.8:
            score += 8
        elif 0.1 <= price_pos_bb <= 0.9:
            score += 4
        else:
            score -= 3

        # Volume ratio (higher is generally better)
        if vol_ratio > 1.5:
            score += 5
        elif vol_ratio > 1.2:
            score += 3
        elif vol_ratio < 0.5:
            score -= 5

        # Performance (positive performance is good)
        if perf_short > 0:
            score += 5
        elif perf_short < -5:  # Strong negative performance reduces score
            score -= 8

        # Momentum (positive momentum is good)
        if momentum > 0:
            score += 3
        elif momentum < -0.1:  # Strong negative momentum reduces score
            score -= 5

        # ROC (positive ROC is good)
        if roc > 0:
            score += 3
        elif roc < -2:  # Strong negative ROC reduces score
            score -= 5

        # CCI contribution (good when between -100 and 100, but also consider extremes)
        if -100 <= cci <= 100:
            score += 5
        elif cci < -100:  # Oversold, potential rebound
            score += 6
        else:  # Overbought
            score += 2

        # KDJ contribution (good when K>D and in middle range)
        if kdj_k > kdj_d and 20 <= kdj_k <= 80:
            score += 6
        elif kdj_k < kdj_d and 20 <= kdj_d <= 80:
            score -= 3

        # Bias contribution (not too far from moving average is good)
        if abs(bias6) < 5:  # Reasonable bias
            score += 5
        elif abs(bias6) > 8:  # Too far from moving average, risky
            score -= 5

        # DMI ADX contribution (higher ADX indicates stronger trend)
        if dmi_adx > 25:
            score += 5  # Strong trend
        elif dmi_adx < 20:
            score -= 3  # Weak trend

        # TRIX contribution (trend following indicator)
        if trix > 0.1:
            score += 5  # Positive trend
        elif trix < -0.1:
            score -= 3  # Negative trend

        # VR contribution (volume confirming price movement)
        if 80 <= vr <= 300:
            score += 3  # Healthy volume relationship
        elif vr > 500:
            score -= 3  # Excessive volume might indicate distribution

        # Limit score between 0 and 100
        score = max(0, min(100, score))

        return score

    def _enhanced_mytt_timing(self, rsi, current_price, ma_5, ma_10, ma_20,
                               bb_upper, bb_lower, bb_middle, macd, macd_signal, vol_ratio, roc,
                               kdj_k, kdj_d, cci, atr, bias6, dmi_pdi, dmi_mdi, trix, cr, obv):
        """
        Enhanced buy/sell timing analysis using MyTT indicators
        """
        advice_parts = []

        # RSI Timing
        if 30 < rsi < 70:
            advice_parts.append("RSI处于中性区域，适合观察")
        elif rsi < 30:
            advice_parts.append("RSI超卖，可能是较好买点")
        elif rsi > 70:
            advice_parts.append("RSI超买，考虑获利了结")

        # Moving Average Alignment
        if current_price > ma_5 > ma_10 > ma_20:
            advice_parts.append("多头排列，趋势向好")
        elif current_price < ma_5 < ma_10 < ma_20:
            advice_parts.append("空头排列，趋势向淡")
        else:
            advice_parts.append("均线纠缠，方向不明")

        # MACD Timing
        if macd > macd_signal:
            advice_parts.append("MACD金叉向上，看涨信号")
        elif macd < macd_signal:
            advice_parts.append("MACD死叉向下，看跌信号")
        else:
            advice_parts.append("MACD与信号线粘合")

        # KDJ Timing
        if kdj_k > kdj_d and kdj_k < 80:
            advice_parts.append("KDJ金叉向上，看涨信号")
        elif kdj_k < kdj_d and kdj_k > 20:
            advice_parts.append("KDJ死叉向下，看跌信号")
        elif kdj_k > 80 and kdj_d > 80:
            advice_parts.append("KDJ高位钝化，注意回调")
        elif kdj_k < 20 and kdj_d < 20:
            advice_parts.append("KDJ超卖区，关注反弹机会")

        # CCI Timing
        if cci > 100:
            advice_parts.append("CCI超买，短期调整风险")
        elif cci < -100:
            advice_parts.append("CCI超卖，反弹预期")
        elif -100 < cci < 100:
            advice_parts.append("CCI在正常区间")

        # Price and Bollinger Bands
        if bb_lower < current_price < bb_middle:
            advice_parts.append("价格在布林带下轨至中轨间，相对安全")
        elif bb_middle < current_price < bb_upper:
            advice_parts.append("价格在布林带中轨至上轨间，注意压力")
        else:
            advice_parts.append("价格偏离布林带，注意回调")

        # BIAS Timing
        if abs(bias6) > 8:
            advice_parts.append("BIAS偏离过大，注意回归")
        elif abs(bias6) < 3:
            advice_parts.append("BIAS位置合理")

        # DMI Analysis
        if dmi_pdi > dmi_mdi:
            advice_parts.append("DMI多头排列，趋势向上")
        else:
            advice_parts.append("DMI空头排列，趋势向下")

        # TRIX Analysis
        if trix > 0:
            advice_parts.append("TRIX趋势向上")
        else:
            advice_parts.append("TRIX趋势向下")

        # CR Analysis (Psychological Line)
        if cr > 150:
            advice_parts.append("CR高位，警惕回调")
        elif cr < 50:
            advice_parts.append("CR低位，关注反弹")
        else:
            advice_parts.append("CR处于合理区域")

        # Volume and ROC
        if vol_ratio > 1.2 and roc > 0:
            advice_parts.append("量价配合良好，趋势持续可能性高")
        elif vol_ratio < 0.8 and roc < 0:
            advice_parts.append("量价背离，趋势可持续性存疑")
        else:
            advice_parts.append("量价关系基本正常")

        # Combine advice
        return "MyTT综合分析: " + "; ".join(advice_parts)

    def show_mytt_help(self):
        """
        Display enhanced help with MyTT analysis options
        """
        print("""
🤖 A股市场分析系统(MyTT增强版) - 可用命令:
=======================================
📈 市场分析类:
  1.  screen_stocks    - 筛选潜在上涨股票 (市值>200亿)
  2.  analyze_stock    - 分析单个股票(传统指标)
  3.  enhanced_analyze - MyTT增强版股票分析
  4.  predict_stocks   - 预测股票上涨概率

📊 策略信号类:
  5.  run_strategy     - 运行指定策略
  6.  gen_signals      - 生成买卖信号
  7.  show_signals     - 显示最新信号

🔍 数据查询类:
  8.  get_data         - 获取股票数据
  9.  calc_indicators  - 计算技术指标
  10. show_top_stocks  - 显示热门股票

📈 预测分析类:
  11. predictive_analysis - 运行预测分析
  12. top_predictions   - 显示Top预测
  13. analyze_market    - 市场整体分析

🔬 回测功能类:
  14. run_backtest      - 运行策略回测
  15. compare_strategies - 比较不同策略

📊 多因子分析类:
  16. multi_factor_analysis - 运行100+因子分析
  17. analyze_factors   - 分析因子表现
  18. factor_report     - 生成因子报告

⚙️  系统管理类:
  19. show_session     - 显示会话数据
  20. clear_session    - 清空会话数据
  21. help             - 显示基础帮助信息
  22. mytt_help        - 显示MyTT增强版帮助
  23. quit/exit        - 退出系统

💡 使用方法: 输入命令编号或命令名称
   例如: 输入 '2' 或 'analyze_stock' 开始股票分析
        输入 '3' 或 'enhanced_analyze' 开始MyTT增强分析
=======================================
        """)

    def run_interactive(self):
        """
        Run the enhanced interactive console
        """
        print(f"🚀 启动A股市场分析系统(MyTT增强版) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("输入 'mytt_help' 查看增强版命令，输入 'quit' 退出系统\n")

        while True:
            try:
                user_input = input(">>>(请输入命令): ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 感谢使用A股市场分析系统，再见！")
                    break
                elif user_input.lower() in ['mytt_help', 'enhanced_help']:
                    self.show_mytt_help()
                elif user_input in self.get_enhanced_command_map():
                    # Handle enhanced command by name
                    self.execute_enhanced_command(user_input)
                elif user_input.isdigit():
                    # Handle numeric command
                    cmd_num = int(user_input)
                    self.handle_numeric_command(cmd_num)
                elif user_input in self.get_command_map():
                    # Handle base command by name
                    self.execute_command(user_input)
                else:
                    print(f"❌ 未知命令: {user_input}")
                    self.show_mytt_help()

            except KeyboardInterrupt:
                print("\n\n👋 系统被用户中断，再见！")
                break
            except Exception as e:
                print(f"❌ 执行命令时出错: {e}")

    def get_enhanced_command_map(self):
        """
        Get mapping of enhanced command names to functions
        """
        base_map = self.get_command_map()
        enhanced_map = {
            'enhanced_analyze': self.enhanced_analyze_stock,
            'mytt_help': self.show_mytt_help,
            'enhanced_analyze_stock': self.enhanced_analyze_stock
        }
        # Combine base and enhanced commands
        combined_map = base_map.copy()
        combined_map.update(enhanced_map)
        return combined_map

    def execute_enhanced_command(self, cmd_name):
        """
        Execute enhanced command by name
        """
        cmd_map = self.get_enhanced_command_map()
        if cmd_name in cmd_map:
            try:
                cmd_map[cmd_name]()
            except Exception as e:
                print(f"❌ 执行 {cmd_name} 时出错: {e}")
        else:
            print(f"❌ 未知命令: {cmd_name}")


def main():
    """
    Main function to run the enhanced MyTT CLI interface
    """
    print("🔍 A股市场分析系统(MyTT增强版)")
    print("="*50)

    # Use your tokens
    tushare_token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"

    # Your EastMoney cookie
    eastmoney_cookie = {
        'qgqp_b_id': 'b7c0c5065c6db033910b1b3175b7c9bb',
        'st_nvi': 'pr7nepf3axSLFdLauyP5y8deb',
        'websitepoptg_api_time': '1770690681021',
        'st_si': '43191381080720',
        'nid18': '0095a8fdc53e2c9dc00f4d602b3c459e',
        'nid18_create_time': '1770690681336',
        'gviem': '6A44mgyL6Tsg59OPlfAXDd677',
        'gviem_create_time': '1770690681337',
        'p_origin': 'https%3A%2F%2Fpassport2.eastmoney.com',
        'mtp': '1',
        'ct': 'wYdhYQ7SFCReRY7yObWFWJwcS2isXO6R8wHwamkysQRCcR9yEiEaMsskY-1tsHOmajDCrGLWHPVacX0DGd_9HoMFpWjxWtVUZEdR8ibclVermnomP1JWdjUpI3BhaRN2ft3jRsDjazoC6F9O5Jzssk-rkmWM3b3LsGJq5RJDxVM',
        'ut': 'FobyicMgeV5FJnFT189SwEfSo-wAjCKxRGfhgXzug4j9BdKmq4gQdtlHffBaUl7Djr5Ju3CTO3tQqVCOs_Vhp9WUQe_9zHJxPmg__J71QWWtiytGWHR6CUXelUQfxok_geZEOJXcc9bQWieI7LUcRQjQFmB-1bwzaZYU3t525uGbFHwr6SZYdP3PBVz04EfQ796KX06LCuYpITwvNu6laJotFHyE5dflMcANoRBf6d8isLvw34K59yZB985bsVHnckUA0HIycKAoU137ZeAYrEX8rjmONDCZy7QGj-BHcAWyIH9OIF98zmSo71GWwWu_X5FP1R2JqWLg9CMTh9wlVBTitMAXMcc5',
        'pi': '9694097255613200%3Bu9694097255613200%3B%E5%A0%82%E5%A0%82%E6%AD%A3%E6%AD%A3%E7%9A%84%E6%9B%B9%E6%93%8D%3BryhxoVjcWC8PTbi0bFrviFAowUa3asGIsa%2F0auHDuAKp6CJ%2BPVN0UwnSDOaEd7utp5uK4oSJImRgmTF0VD7Nm1Zqq9vnKuG5c1wWVRNZxJmnEN416UgEorQVUQJ5tnsTgIcvWxtVIJHhIll%2F9SIWv6E6wIrLFINK3wF12TZX3gkL7%2FxLaYbHaFQ0YON21YMY%2BZKCiilR%3Bp2dLhWNuZSa0SCigDD%2FOLxaCiti2fW5OSY32vbSSck%2BT1BzvA%2FAQHG2jYCxHc8Httaxt1PRsFPhuwvBF873qXa7Y5muaKZZN0jzerURbzjeerxd31x755Is9mu7LD%2BGWpkI3piLVRUUL5xl2ifRVnekqrax4Yg%3D%3D',
        'uidal': '9694097255613200%e5%a0%82%e5%a0%82%e6%ad%a3%e6%ad%a3%e7%9a%84%e6%9b%b9%e6%93%8d',
        'sid': '',
        'vtpst': '|',
        'st_asi': 'delete',
        'wsc_checkuser_ok': '1',
        'fullscreengg': '1',
        'fullscreengg2': '1',
        'st_pvi': '27562121748759',
        'st_sp': '2025-10-30%2011%3A15%3A42',
        'st_inirUrl': 'https%3A%2F%2Fwww.google.com.hk%2F',
        'st_sn': '5',
        'st_psi': '20260210130257951-111000300841-0487608401'
    }

    # Initialize enhanced interface
    cli_interface = EnhancedMyTTCLIInterface(tushare_token, eastmoney_cookie)

    # Run interactive mode
    cli_interface.run_interactive()


if __name__ == "__main__":
    main()