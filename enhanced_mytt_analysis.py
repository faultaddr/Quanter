#!/usr/bin/env python3
"""
Enhanced Analysis using MyTT Technical Indicators
This script demonstrates comprehensive analysis using MyTT indicators integrated with the quant trading system
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.utils.mytt_indicators import calculate_mytt_indicators
from quant_trade_a_share.utils.eastmoney_data_fetcher import EastMoneyDataFetcher


class MyTTAnalyzer:
    """
    Analyzer class that uses MyTT technical indicators for comprehensive market analysis
    """

    def __init__(self, cookies=None):
        """
        Initialize analyzer with EastMoney data fetcher
        """
        self.fetcher = EastMoneyDataFetcher(cookies)

    def analyze_stock(self, symbol, days=180):
        """
        Perform comprehensive analysis of a stock using MyTT indicators
        """
        print(f"🔍 开始分析股票 {symbol} (最近{days}天)")
        print("="*60)

        # Fetch data
        data = self.fetcher.fetch_stock_data(symbol, days=days)
        if data is None or data.empty:
            print(f"❌ 无法获取 {symbol} 的数据")
            return None

        print(f"✅ 获取到 {len(data)} 条数据记录")

        # Calculate MyTT indicators
        print("📈 计算MyTT技术指标...")
        data = calculate_mytt_indicators(data)

        # Get the latest values
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else data.iloc[-1]

        current_price = latest['close']

        print(f"\n💰 基本信息:")
        print(f"   当前价格: ¥{current_price:.2f}")
        print(f"   最高价: ¥{latest['high']:.2f}")
        print(f"   最低价: ¥{latest['low']:.2f}")
        print(f"   开盘价: ¥{latest['open']:.2f}")
        print(f"   成交量: {latest['volume']:,}")

        # Technical indicators analysis
        self._analyze_technical_indicators(latest, prev, current_price)

        # Trading signals analysis
        self._analyze_trading_signals(latest, prev)

        # Comprehensive evaluation
        self._comprehensive_evaluation(data, latest)

        return data

    def _analyze_technical_indicators(self, latest, prev, current_price):
        """
        Analyze various technical indicators from MyTT
        """
        print(f"\n📊 技术指标分析:")
        print(f"   - RSI6/12/24: {latest['rsi6']:.2f}/{latest['rsi12']:.2f}/{latest['rsi24']:.2f}")
        print(f"   - MACD: {latest['macd_dif']:.4f}, Signal: {latest['macd_dea']:.4f}, Bar: {latest['macd_bar']:.4f}")
        print(f"   - KDJ: K:{latest['kdj_k']:.2f}, D:{latest['kdj_d']:.2f}, J:{latest['kdj_j']:.2f}")
        print(f"   - CCI: {latest['cci']:.2f}")
        print(f"   - BOLL: 上轨{latest['boll_upper']:.2f}, 中轨{latest['boll_mid']:.2f}, 下轨{latest['boll_lower']:.2f}")
        print(f"   - 均线: MA5:{latest['ma5']:.2f}, MA10:{latest['ma10']:.2f}, MA20:{latest['ma20']:.2f}")
        print(f"   - DMI: PDI:{latest['dmi_pdi']:.2f}, MDI:{latest['dmi_mdi']:.2f}, ADX:{latest['dmi_adx']:.2f}")
        print(f"   - BIAS6/12/24: {latest['bias6']:.2f}/{latest['bias12']:.2f}/{latest['bias24']:.2f}")
        print(f"   - TRIX: {latest['trix']:.4f}, TRMA: {latest['trma']:.4f}")
        print(f"   - VR: {latest['vr']:.2f}, CR: {latest['cr']:.2f}")
        print(f"   - ATR: {latest['atr']:.4f}")

        # Analysis based on indicators
        analysis_notes = []

        # RSI Analysis
        if latest['rsi24'] > 70:
            analysis_notes.append("RSI24 > 70，可能处于超买区域")
        elif latest['rsi24'] < 30:
            analysis_notes.append("RSI24 < 30，可能处于超卖区域")

        # MACD Analysis
        if latest['macd_dif'] > latest['macd_dea']:
            analysis_notes.append("DIF > DEA，MACD呈多头排列")
        else:
            analysis_notes.append("DIF < DEA，MACD呈空头排列")

        # KDJ Analysis
        if latest['kdj_k'] > latest['kdj_d']:
            analysis_notes.append("K > D，KDJ呈多头排列")
        if latest['kdj_k'] < 20 and latest['kdj_d'] < 20:
            analysis_notes.append("KDJ进入超卖区，可能存在反弹机会")
        elif latest['kdj_k'] > 80 and latest['kdj_d'] > 80:
            analysis_notes.append("KDJ进入超买区，可能存在回调风险")

        # CCI Analysis
        if latest['cci'] > 100:
            analysis_notes.append("CCI > 100，可能进入超买状态")
        elif latest['cci'] < -100:
            analysis_notes.append("CCI < -100，可能进入超卖状态")
        elif -100 <= latest['cci'] <= 100:
            analysis_notes.append("CCI在正常波动区间")

        # Price position relative to BOLL
        if current_price > latest['boll_upper']:
            analysis_notes.append("价格突破布林带上轨，短期可能回调")
        elif current_price < latest['boll_lower']:
            analysis_notes.append("价格跌破布林带下轨，短期可能反弹")
        elif current_price > latest['boll_mid']:
            analysis_notes.append("价格在布林带中轨上方运行")
        else:
            analysis_notes.append("价格在布林带中轨下方运行")

        # Moving Average Analysis
        if current_price > latest['ma20']:
            analysis_notes.append("价格站稳20日均线，中期趋势偏多")
        else:
            analysis_notes.append("价格跌破20日均线，中期趋势偏空")

        if latest['ma5'] > latest['ma10'] > latest['ma20']:
            analysis_notes.append("5/10/20日均线多头排列，短期趋势向好")
        elif latest['ma5'] < latest['ma10'] < latest['ma20']:
            analysis_notes.append("5/10/20日均线空头排列，短期趋势向淡")

        # BIAS Analysis
        if abs(latest['bias6']) > 8:
            analysis_notes.append("BIAS6偏离较大，注意均值回归风险")
        elif abs(latest['bias6']) < 3:
            analysis_notes.append("BIAS6位置合理，处于正常波动范围")

        # DMI Analysis
        if latest['dmi_adx'] > 25:
            analysis_notes.append("ADX > 25，趋势强度较强")
        elif latest['dmi_adx'] < 20:
            analysis_notes.append("ADX < 20，趋势强度较弱，可能震荡")

        print(f"\n📝 技术分析要点:")
        for note in analysis_notes:
            print(f"   • {note}")

    def _analyze_trading_signals(self, latest, prev):
        """
        Analyze trading signals from various indicators
        """
        print(f"\n🎯 交易信号分析:")

        signals = []

        # MACD Signal
        if latest['macd_dif'] > latest['macd_dea'] and prev['macd_dif'] <= prev['macd_dea']:
            signals.append("🟢 MACD金叉 - 买入信号")
        elif latest['macd_dif'] < latest['macd_dea'] and prev['macd_dif'] >= prev['macd_dea']:
            signals.append("🔴 MACD死叉 - 卖出信号")

        # KDJ Signal
        if latest['kdj_k'] > latest['kdj_d'] and prev['kdj_k'] <= prev['kdj_d']:
            signals.append("🟢 KDJ金叉 - 买入信号")
        elif latest['kdj_k'] < latest['kdj_d'] and prev['kdj_k'] >= prev['kdj_d']:
            signals.append("🔴 KDJ死叉 - 卖出信号")

        # Price and Moving Average Signals
        if latest['close'] > latest['ma5'] and prev['close'] <= prev['ma5']:
            signals.append("🟢 突破5日均线 - 买入信号")
        elif latest['close'] < latest['ma5'] and prev['close'] >= prev['ma5']:
            signals.append("🔴 跌破5日均线 - 卖出信号")

        # CCI Signals
        if latest['cci'] > -100 and prev['cci'] <= -100:
            signals.append("🟢 CCI从超卖回升 - 买入信号")
        elif latest['cci'] < 100 and prev['cci'] >= 100:
            signals.append("🔴 CCI从超买回落 - 卖出信号")

        # TRIX Signals
        if latest['trix'] > latest['trma'] and prev['trix'] <= prev['trma']:
            signals.append("🟢 TRIX金叉TRMA - 买入信号")
        elif latest['trix'] < latest['trma'] and prev['trix'] >= prev['trma']:
            signals.append("🔴 TRIX死叉TRMA - 卖出信号")

        if signals:
            for signal in signals:
                print(f"   {signal}")
        else:
            print("   无明显交易信号")

    def _comprehensive_evaluation(self, data, latest):
        """
        Comprehensive evaluation based on all indicators
        """
        print(f"\n🏆 综合评估:")

        # Calculate scores for different aspects
        trend_score = self._calculate_trend_score(latest, data)
        momentum_score = self._calculate_momentum_score(latest)
        volatility_score = self._calculate_volatility_score(latest)
        risk_score = self._calculate_risk_score(latest, data)

        overall_score = (trend_score + momentum_score + volatility_score + risk_score) / 4

        print(f"   趋势得分: {trend_score:.1f}/100 ({self._score_to_desc(trend_score)})")
        print(f"   动量得分: {momentum_score:.1f}/100 ({self._score_to_desc(momentum_score)})")
        print(f"   波动得分: {volatility_score:.1f}/100 ({self._score_to_desc(volatility_score)})")
        print(f"   风险得分: {risk_score:.1f}/100 ({self._score_to_desc(risk_score)})")
        print(f"   综合评分: {overall_score:.1f}/100 ({self._score_to_desc(overall_score)})")

        # Investment recommendation based on scores
        recommendation = self._investment_recommendation(overall_score, trend_score, risk_score)
        print(f"\n💡 投资建议: {recommendation}")

    def _calculate_trend_score(self, latest, data):
        """
        Calculate trend score based on multiple indicators
        """
        score = 50  # Base score

        # Moving averages trend
        if latest['close'] > latest['ma20']:
            score += 15
        else:
            score -= 15

        if latest['ma5'] > latest['ma10'] > latest['ma20']:
            score += 10
        elif latest['ma5'] < latest['ma10'] < latest['ma20']:
            score -= 10

        # DMI trend strength
        if latest['dmi_adx'] > 30:
            score += 10
        elif latest['dmi_adx'] < 20:
            score -= 10

        # Price vs BOLL position
        if latest['boll_lower'] < latest['close'] < latest['boll_upper']:
            score += 5
        else:
            score -= 5

        return max(0, min(100, score))

    def _calculate_momentum_score(self, latest):
        """
        Calculate momentum score based on momentum indicators
        """
        score = 50  # Base score

        # RSI momentum
        if 30 <= latest['rsi24'] <= 70:
            score += 10
        elif latest['rsi24'] < 30:
            score += 8  # Oversold bounce potential
        elif latest['rsi24'] > 70:
            score -= 5  # Overbought correction risk

        # MACD momentum
        if latest['macd_bar'] > 0:
            score += 8
        else:
            score -= 5

        # KDJ momentum
        if latest['kdj_j'] > latest['kdj_k'] > latest['kdj_d']:
            score += 8
        elif latest['kdj_j'] < latest['kdj_k'] < latest['kdj_d']:
            score -= 8

        # CCI momentum
        if -100 <= latest['cci'] <= 100:
            score += 5
        elif latest['cci'] < -100:
            score += 6  # Oversold potential
        elif latest['cci'] > 100:
            score -= 3  # Overbought risk

        return max(0, min(100, score))

    def _calculate_volatility_score(self, latest):
        """
        Calculate volatility score
        """
        score = 50  # Base score

        # BOLL width (volatility indicator)
        boll_width = (latest['boll_upper'] - latest['boll_lower']) / latest['boll_mid']
        if 0.05 <= boll_width <= 0.15:
            score += 5  # Moderate volatility is good
        elif boll_width > 0.15:
            score -= 5  # Too volatile
        else:
            score += 3  # Low volatility

        return max(0, min(100, score))

    def _calculate_risk_score(self, latest, data):
        """
        Calculate risk score based on various risk factors
        """
        score = 50  # Base score

        # Risk factors - lower score for higher risk
        if latest['rsi24'] > 80 or latest['rsi24'] < 20:
            score -= 10  # Extreme RSI

        if abs(latest['bias6']) > 8:
            score -= 8  # High BIAS deviation

        if latest['dmi_adx'] < 15:
            score += 5  # Weak trend = less risk of big moves
        elif latest['dmi_adx'] > 40:
            score -= 8  # Very strong trend = higher risk of reversal

        # Volume risk
        if latest['volume_ratio'] > 3:
            score -= 5  # Unusually high volume may indicate manipulation

        return max(0, min(100, score))

    def _score_to_desc(self, score):
        """
        Convert score to descriptive text
        """
        if score >= 80:
            return "优秀"
        elif score >= 60:
            return "良好"
        elif score >= 40:
            return "一般"
        else:
            return "较差"

    def _investment_recommendation(self, overall_score, trend_score, risk_score):
        """
        Generate investment recommendation based on scores
        """
        if overall_score >= 75 and trend_score >= 70 and risk_score >= 60:
            return "📈 强烈推荐关注 - 趋势强劲，风险可控"
        elif overall_score >= 65 and trend_score >= 60:
            return "📊 推荐关注 - 技术面较为健康"
        elif overall_score >= 50:
            return "⚖️  谨慎关注 - 存在一定机会"
        elif overall_score >= 40:
            return "⏳ 观望等待 - 技术面偏弱"
        else:
            return "❌ 暂不推荐 - 技术面较差，风险较高"


def main():
    """
    Main function to demonstrate MyTT-based analysis
    """
    print("🚀 基于MyTT指标的增强版股票分析系统")
    print("="*60)

    # Initialize analyzer
    analyzer = MyTTAnalyzer()

    # Get user input
    symbol = input("请输入股票代码 (例: sh600519): ").strip()
    if not symbol:
        symbol = "sh600519"  # Default to Kweichow Moutai

    try:
        # Perform analysis
        data = analyzer.analyze_stock(symbol)

        if data is not None:
            print(f"\n✅ {symbol} 分析完成！")

            # Ask if user wants to save the data
            save_option = input("\n是否保存分析数据到CSV文件? (y/n): ").strip().lower()
            if save_option == 'y':
                filename = f"{symbol}_mytt_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                data.to_csv(filename, encoding='utf-8-sig')
                print(f"💾 数据已保存到 {filename}")

    except KeyboardInterrupt:
        print("\n\n👋 用户中断操作")
    except Exception as e:
        print(f"\n❌ 分析过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()