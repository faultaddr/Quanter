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

    def run_trading_strategies(self, df: pd.DataFrame) -> Dict:
        """
        Run various trading strategies and scoring system on the given data
        """
        if df.empty:
            return {}

        print("正在运行交易策略...")

        # Initialize strategy evaluator
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

        # Run scoring system
        print("正在计算多维度评分...")
        scoring = ScoringSystem()
        results['scoring'] = scoring.calculate_all_scores(df)
        print("评分计算完成。")

        print("交易策略运行完成。")
        return results

    def generate_report(self, df: pd.DataFrame, strategies_results: Dict, symbol: str) -> str:
        """
        生成综合分析报告（中文）
        """
        if df.empty:
            return "无可用数据进行分析。"

        # 获取最新数据点
        latest_data = df.iloc[-1]

        # 信号中文映射
        signal_map = {
            'STRONG_BUY': '强烈买入',
            'WEAK_BUY': '弱势买入',
            'BUY': '买入',
            'STRONG_SELL': '强烈卖出',
            'WEAK_SELL': '弱势卖出',
            'SELL': '卖出',
            'HOLD': '持有',
            'N/A': '不适用'
        }

        confidence_map = {
            'High': '高',
            'Medium': '中',
            'Low': '低',
            'N/A': '不适用'
        }

        def get_signal_cn(signal):
            return signal_map.get(signal, signal)

        def get_confidence_cn(conf):
            return confidence_map.get(conf, conf)

        report = []
        report.append("")
        report.append(f"## 股票分析报告：{symbol}")
        report.append("")
        report.append(f"**分析日期：** {datetime.now().strftime('%Y年%m月%d日 %H:%M')}")
        report.append(f"**分析周期：** {df['timestamp'].min().date()} 至 {df['timestamp'].max().date()}（共{len(df)}个交易日）")
        report.append("")
        report.append("---")
        report.append("")

        # 当前市场数据
        report.append("### 当前市场数据")
        report.append("")
        report.append(f"| 项目 | 数值 |")
        report.append(f"|------|------|")
        report.append(f"| 当前股价 | ¥{latest_data['close']:.2f} |")
        report.append(f"| 今日涨跌 | {latest_data['daily_return']*100:+.2f}% |")
        report.append(f"| 成交量 | {latest_data['volume']:,.0f}手 |")
        report.append(f"| 今日最高/最低 | ¥{latest_data['high']:.2f} / ¥{latest_data['low']:.2f} |")
        report.append(f"| 今日开盘 | ¥{latest_data['open']:.2f} |")
        report.append("")

        # 技术指标
        report.append("### 技术指标一览")
        report.append("")
        report.append(f"| 指标 | 数值 | 说明 |")
        report.append(f"|------|------|------|")

        # RSI
        rsi_val = latest_data['rsi_24']
        rsi_desc = "超买" if rsi_val > 70 else "超卖" if rsi_val < 30 else "中性"
        report.append(f"| RSI(24) | {rsi_val:.2f} | {rsi_desc} |")

        # MACD
        macd_val = latest_data['macd']
        macd_desc = "多头" if macd_val > 0 else "空头"
        report.append(f"| MACD | {macd_val:.2f} | {macd_desc} |")

        # KDJ
        k, d, j = latest_data['kdj_k'], latest_data['kdj_d'], latest_data['kdj_j']
        kdj_desc = "J值偏高" if j > 80 else "J值偏低" if j < 20 else "正常"
        report.append(f"| KDJ | K: {k:.2f} / D: {d:.2f} / J: {j:.2f} | {kdj_desc} |")

        # MA
        ma20, ma50, ma200 = latest_data['ma_20'], latest_data['ma_50'], latest_data['ma_200']
        ma200_str = f"{ma200:.2f}" if not pd.isna(ma200) else "无数据"
        report.append(f"| 移动平均线 | MA20: ¥{ma20:.2f} / MA50: ¥{ma50:.2f} / MA200: {ma200_str} | 趋势参考 |")

        # BOLL
        bu, bm, bl = latest_data['boll_upper'], latest_data['boll_mid'], latest_data['boll_lower']
        close_val = latest_data['close']
        boll_desc = "触及上轨" if close_val >= bu else "触及下轨" if close_val <= bl else "中轨附近"
        report.append(f"| 布林带 | 上轨: ¥{bu:.2f} / 中轨: ¥{bm:.2f} / 下轨: ¥{bl:.2f} | {boll_desc} |")

        # CCI
        cci_val = latest_data['cci']
        cci_desc = "超买区" if cci_val > 200 else "超卖区" if cci_val < -200 else "正常"
        report.append(f"| CCI | {cci_val:.2f} | {cci_desc} |")

        # ATR
        report.append(f"| ATR(14) | {latest_data['atr_14']:.2f} | 波动率 |")

        # DMI
        pdi, mdi, adx = latest_data['dmi_pdi'], latest_data['dmi_mdi'], latest_data['dmi_adx']
        dmi_desc = "多头占优" if pdi > mdi else "空头占优"
        report.append(f"| DMI | PDI: {pdi:.2f} / MDI: {mdi:.2f} / ADX: {adx:.2f} | {dmi_desc} |")

        # TRIX
        trix_val = latest_data['trix']
        trix_desc = "向上" if trix_val > 0 else "向下"
        report.append(f"| TRIX | {trix_val:.2f} | {trix_desc} |")

        # VR
        vr_val = latest_data['vr']
        vr_desc = "量能活跃" if vr_val > 150 else "量能萎缩" if vr_val < 80 else "正常"
        report.append(f"| VR（成交量比率） | {vr_val:.2f} | {vr_desc} |")

        # CR
        report.append(f"| CR | {latest_data['cr']:.2f} | 人气指标 |")

        # WR
        wr_val = latest_data['wr']
        wr_desc = "严重超买" if wr_val < 10 else "严重超卖" if wr_val > 90 else "正常"
        report.append(f"| WR（威廉指标） | {wr_val:.2f} | {wr_desc} |")

        # BBI
        bbi_val = latest_data['bbi']
        bbi_desc = "上方" if close_val > bbi_val else "下方"
        report.append(f"| BBI（多空指数） | ¥{bbi_val:.2f} | 股价在{bbi_desc} |")
        report.append("")

        # 交易策略信号
        report.append("### 交易策略信号")
        report.append("")
        report.append(f"| 策略 | 当前信号 | 建议操作 | 置信度 | 信号变化 |")
        report.append(f"|------|---------|---------|--------|---------|")

        evaluations = strategies_results.get('evaluations', {})
        strategy_name_map = {
            'rsi': 'RSI策略',
            'macd': 'MACD策略',
            'ma_crossover': '均线交叉',
            'bollinger_bands': '布林带策略',
            'combined': '综合策略'
        }

        for strategy_key, eval_result in evaluations.items():
            if isinstance(eval_result, dict):
                strategy_cn = strategy_name_map.get(strategy_key, strategy_key.upper())
                signal = get_signal_cn(eval_result.get('current_signal', 'N/A'))
                action = eval_result.get('action', 'N/A')
                confidence = get_confidence_cn(eval_result.get('confidence', 'N/A'))
                changed = "是" if eval_result.get('signal_changed', False) else "否"
                report.append(f"| {strategy_cn} | {signal} | {action} | {confidence} | {changed} |")

        report.append("")

        # 多维度打分系统
        report.append("### 多维度量化评分")
        report.append("")

        scoring = strategies_results.get('scoring', {})
        if scoring:
            report.append(f"| 维度 | 检查项 | 得分 | 说明 |")
            report.append(f"|------|--------|------|------|")

            dimensions = scoring.get('dimensions', {})
            for dim_name, dim_data in dimensions.items():
                report.append(f"| {dim_data.get('name', dim_name)} | {dim_data.get('check', '-')} | **{dim_data.get('score', 0):+d}** | {dim_data.get('desc', '-')} |")

            report.append("")
            report.append(f"**总分：{scoring.get('total_score', 0):+d} / 10**")
            report.append("")

            # 风险提示
            warnings = scoring.get('warnings', [])
            if warnings:
                report.append("⚠️ **风险提示：**")
                for warning in warnings:
                    report.append(f"- {warning}")
                report.append("")

        # 综合建议（基于打分系统）
        report.append("### 综合建议")
        report.append("")

        if scoring:
            total_score = scoring.get('total_score', 0)
            rating = scoring.get('rating', '观望')
            action = scoring.get('action', '继续观望')

            if total_score > 3:
                report.append(f"## 📈 买入信号（总分：+{total_score}）")
                report.append("")
                report.append(f"**评级：{rating}**")
                report.append("")
                report.append(f"**建议操作：** {action}")
            elif total_score < -3:
                report.append(f"## 📉 卖出信号（总分：{total_score}）")
                report.append("")
                report.append(f"**评级：{rating}**")
                report.append("")
                report.append(f"**建议操作：** {action}")
            else:
                report.append(f"## ➖ 观望（总分：{total_score:+d}）")
                report.append("")
                report.append(f"**评级：{rating}**")
                report.append("")
                report.append(f"**建议操作：** {action}")

            report.append("")
            report.append("**评分逻辑：**")
            report.append(f"- 趋势维度：{dimensions.get('trend', {}).get('score', 0):+d} 分")
            report.append(f"- 动量维度：{dimensions.get('momentum', {}).get('score', 0):+d} 分")
            report.append(f"- 波动维度：{dimensions.get('volatility', {}).get('score', 0):+d} 分")
            report.append(f"- 资金维度：{dimensions.get('capital', {}).get('score', 0):+d} 分")
            report.append(f"- 结构维度：{dimensions.get('structure', {}).get('score', 0):+d} 分")
        else:
            # Fallback to old logic if scoring not available
            combined_signal = evaluations.get('combined', {}).get('current_signal', 'HOLD')
            rsi_signal = evaluations.get('rsi', {}).get('current_signal', 'HOLD')
            macd_signal = evaluations.get('macd', {}).get('current_signal', 'HOLD')

            buy_signals = sum([
                combined_signal in ['STRONG_BUY', 'WEAK_BUY'],
                rsi_signal in ['BUY'],
                macd_signal in ['BUY']
            ])

            sell_signals = sum([
                combined_signal in ['STRONG_SELL', 'WEAK_SELL'],
                rsi_signal in ['SELL'],
                macd_signal in ['SELL']
            ])

            if buy_signals >= 2:
                report.append("## 强烈买入（STRONG BUY）")
                report.append("")
                report.append("多个指标显示买入机会，建议积极建仓。")
            elif buy_signals == 1:
                report.append("## 弱势买入（WEAK BUY）")
                report.append("")
                report.append("部分指标显示潜在买入机会，可小仓位试探。")
            elif sell_signals >= 2:
                report.append("## 强烈卖出（STRONG SELL）")
                report.append("")
                report.append("多个指标显示卖出信号，建议减仓规避风险。")
            elif sell_signals == 1:
                report.append("## 弱势卖出（WEAK SELL）")
                report.append("")
                report.append("部分指标显示潜在卖出机会，可考虑小幅减仓。")
            else:
                report.append("## 持有观望（HOLD）")
                report.append("")
                report.append("信号混合，暂无明确方向，建议继续持有观望。")

        report.append("")
        report.append("---")
        report.append("")
        report.append("> **免责声明：** 本分析仅供学习参考，不构成投资建议。投资决策应基于全面研究和独立判断。")
        report.append("")

        return "\n".join(report)

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
        strategies_results = self.run_trading_strategies(df_with_indicators)

        # Generate comprehensive report
        report = self.generate_report(df_with_indicators, strategies_results, symbol)

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