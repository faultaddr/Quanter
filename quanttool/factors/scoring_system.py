"""
股票多维度打分系统
基于趋势、动量、波动、资金、结构五个维度进行综合评分
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List


class ScoringSystem:
    """
    股票多维度打分系统
    满分10分，根据五个维度（趋势、动量、波动、资金、结构）进行综合评估
    """

    def __init__(self):
        self.score_breakdown = {}

    def calculate_all_scores(self, df: pd.DataFrame) -> Dict:
        """
        计算所有维度的得分

        Returns:
            Dict: 包含各维度得分、总分、评级和操作推荐的字典
        """
        if df.empty or len(df) < 20:
            return {"error": "数据不足，无法打分"}

        latest = df.iloc[-1]
        prev = df.iloc[-5] if len(df) >= 5 else df.iloc[-1]  # 5周期前数据

        # 各维度打分
        trend_score, trend_detail = self._score_trend(df, latest)
        momentum_score, momentum_detail = self._score_momentum(df, latest)
        volatility_score, volatility_detail = self._score_volatility(df, latest)
        capital_score, capital_detail = self._score_capital(df, latest, prev)
        structure_score, structure_detail = self._score_structure(df, latest)
        bias_score, bias_detail = self._score_bias(latest)  # 新增乖离率维度

        # 计算总分 (现在6个维度，满分12分)
        total_score = trend_score + momentum_score + volatility_score + capital_score + structure_score + bias_score

        # 评级和操作推荐
        rating, action, risk_level = self._get_rating_and_action(total_score, capital_score)

        # 收集警告
        warnings = []
        if capital_score <= -3:
            warnings.append("量价背离：价格上涨但量能不足或OBV下降，存在回调风险")
        if momentum_score == -1 and "背离" in momentum_detail:
            warnings.append("MACD背离：价格与动量指标出现背离")
        if volatility_score == -2:
            warnings.append("超买警告：价格接近布林带上轨，短期回调概率增加")
        if bias_score <= -1:
            warnings.append(f"乖离率偏高：BIAS(6)={latest.get('bias_6', 0):.2f}%，股价偏离均线较远，注意回调风险")

        return {
            "dimensions": {
                "trend": {"name": "趋势维度", "check": "MA均线", "score": trend_score, "desc": trend_detail},
                "momentum": {"name": "动量维度", "check": "MACD+RSI", "score": momentum_score, "desc": momentum_detail},
                "volatility": {"name": "波动维度", "check": "布林带", "score": volatility_score, "desc": volatility_detail},
                "capital": {"name": "资金维度", "check": "OBV+VR", "score": capital_score, "desc": capital_detail},
                "structure": {"name": "结构维度", "check": "DMI+位置", "score": structure_score, "desc": structure_detail},
                "bias": {"name": "乖离率维度", "check": "BIAS", "score": bias_score, "desc": bias_detail},
            },
            "total_score": total_score,
            "rating": rating,
            "action": action,
            "risk_level": risk_level,
            "warnings": warnings
        }

    def _score_trend(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[int, str]:
        """
        趋势维度：基于MA均线系统评分
        多头排列 +2，空头排列 -2，均线之下/之上但排列完好 ±1，纠缠 0
        """
        ma20 = latest.get('ma_20', np.nan)
        ma50 = latest.get('ma_50', np.nan)
        ma200 = latest.get('ma_200', np.nan)
        close = latest.get('close', np.nan)

        # 检查数据有效性
        if pd.isna(ma20) or pd.isna(ma50):
            return 0, "均线数据不足"

        # 判断多头排列：短期 > 中期 > 长期（如果有长期数据）
        if not pd.isna(ma200):
            if close > ma20 > ma50 > ma200:
                return 2, f"多头排列（收盘价¥{close:.2f} > MA20¥{ma20:.2f} > MA50¥{ma50:.2f} > MA200¥{ma200:.2f}）"
            elif close < ma20 < ma50 < ma200:
                return -2, f"空头排列（收盘价¥{close:.2f} < MA20¥{ma20:.2f} < MA50¥{ma50:.2f} < MA200¥{ma200:.2f}）"
            elif ma20 > ma50 > ma200 and close < ma20:
                # 均线多头排列但股价跌破MA20，短期转弱
                return -1, f"多头趋势转弱（收盘价¥{close:.2f} < MA20¥{ma20:.2f}，但MA20>MA50>MA200）"
            elif ma20 < ma50 < ma200 and close > ma20:
                # 均线空头排列但股价站上MA20，短期转强
                return 1, f"空头趋势转强（收盘价¥{close:.2f} > MA20¥{ma20:.2f}，但MA20<MA50<MA200）"
            else:
                return 0, f"均线纠缠（MA20¥{ma20:.2f} / MA50¥{ma50:.2f} / MA200¥{ma200:.2f}）"
        else:
            # 只有MA20和MA50
            if close > ma20 > ma50:
                return 2, f"短期多头（收盘价¥{close:.2f} > MA20¥{ma20:.2f} > MA50¥{ma50:.2f}）"
            elif close < ma20 < ma50:
                return -2, f"短期空头（收盘价¥{close:.2f} < MA20¥{ma20:.2f} < MA50¥{ma50:.2f}）"
            elif close < ma20 and close < ma50 and ma20 > ma50:
                # 股价跌破两条均线，且MA20>MA50（死叉后下跌趋势确认）
                return -2, f"空头确认（收盘价¥{close:.2f} < MA20¥{ma20:.2f}且<MA50¥{ma50:.2f}，MA20>MA50）"
            elif close > ma20 and close > ma50 and ma20 < ma50:
                # 股价站上两条均线，且MA20<MA50（金叉后上涨趋势确认）
                return 2, f"多头确认（收盘价¥{close:.2f} > MA20¥{ma20:.2f}且>MA50¥{ma50:.2f}，MA20<MA50）"
            elif ma20 > ma50 and close < ma20:
                # MA20>MA50但股价跌破MA20，短期回调
                return -1, f"多头回调（收盘价¥{close:.2f} < MA20¥{ma20:.2f}，但MA20>MA50¥{ma50:.2f}）"
            elif ma20 < ma50 and close > ma20:
                # MA20<MA50但股价站上MA20，短期反弹
                return 1, f"空头反弹（收盘价¥{close:.2f} > MA20¥{ma20:.2f}，但MA20<MA50¥{ma50:.2f}）"
            else:
                return 0, f"趋势不明（MA20¥{ma20:.2f} vs MA50¥{ma50:.2f}）"

    def _score_momentum(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[int, str]:
        """
        动量维度：基于MACD和RSI评分
        双多 +2，双空 -2，顶背离 -1~0，底背离 +1~0，中性 0
        """
        macd = latest.get('macd', np.nan)
        rsi = latest.get('rsi_24', np.nan)

        if pd.isna(macd) or pd.isna(rsi):
            return 0, "动量数据不足"

        macd_bull = macd > 0
        rsi_bull = 30 < rsi < 70  # RSI在合理区间视为中性偏多
        rsi_overbought = rsi > 70
        rsi_oversold = rsi < 30

        # 检查MACD背离（区分顶背离和底背离）
        divergence_type = None  # None, 'top', 'bottom'
        if len(df) >= 10:
            # 使用更长的窗口检测背离（10日 vs 5日）
            price_recent = df['close'].iloc[-5:].mean()
            price_prev = df['close'].iloc[-10:-5].mean()
            macd_recent = df['macd'].iloc[-5:].mean()
            macd_prev = df['macd'].iloc[-10:-5].mean()

            price_up = price_recent > price_prev
            macd_up = macd_recent > macd_prev

            if price_up and not macd_up:
                # 价格新高但MACD未新高 → 顶背离（看跌）
                divergence_type = 'top'
            elif not price_up and macd_up:
                # 价格新低但MACD未新低 → 底背离（看涨）
                divergence_type = 'bottom'

        if macd_bull and not rsi_overbought and not rsi_oversold:
            if divergence_type == 'top':
                return -1, f"MACD偏多但出现顶背离，警惕回调（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
            return 2, f"动量健康向上（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
        elif not macd_bull and (rsi_overbought or rsi_oversold):
            if rsi_overbought:
                return -2, f"动量偏空，RSI超买回调（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
            else:
                if divergence_type == 'bottom':
                    return 1, f"RSI超卖+底背离，或有反弹机会（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
                return -2, f"动量偏空，RSI超卖（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
        elif macd_bull and rsi_overbought:
            if divergence_type == 'top':
                return -2, f"MACD向上但RSI超买+顶背离，回调风险高（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
            return -1, f"MACD向上但RSI超买警告（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
        elif not macd_bull and rsi_oversold:
            if divergence_type == 'bottom':
                return 1, f"MACD底背离+RSI超卖，反弹概率增加（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
            return 0, f"MACD负值但RSI超卖，观望（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
        elif divergence_type == 'top':
            return -1, f"MACD顶背离，注意回调风险（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
        elif divergence_type == 'bottom':
            return 1, f"MACD底背离，或有反弹机会（MACD:{macd:.2f}, RSI:{rsi:.2f}）"
        else:
            return 0, f"动量中性（MACD:{macd:.2f}, RSI:{rsi:.2f}）"

    def _score_volatility(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[int, str]:
        """
        波动维度：基于布林带位置评分
        下轨附近 +2，上轨附近 -2，中间区 ±1
        """
        close = latest.get('close', np.nan)
        boll_upper = latest.get('boll_upper', np.nan)
        boll_mid = latest.get('boll_mid', np.nan)
        boll_lower = latest.get('boll_lower', np.nan)

        if pd.isna(boll_upper) or pd.isna(boll_lower) or pd.isna(boll_mid):
            return 0, "布林带数据不足"

        # 计算在布林带中的位置（0-100%）
        band_range = boll_upper - boll_lower
        if band_range == 0:
            return 0, "布林带宽度为0"

        position_pct = (close - boll_lower) / band_range * 100

        if position_pct <= 10:  # 下轨10%以内
            return 2, f"接近下轨超卖区（位置:{position_pct:.1f}%, 下轨¥{boll_lower:.2f}）"
        elif position_pct >= 90:  # 上轨10%以内
            return -2, f"接近上轨超买区（位置:{position_pct:.1f}%, 上轨¥{boll_upper:.2f}）"
        elif position_pct <= 30:  # 下轨附近
            return 1, f"偏下轨偏多（位置:{position_pct:.1f}%）"
        elif position_pct >= 70:  # 上轨附近
            return -1, f"偏上轨偏空（位置:{position_pct:.1f}%）"
        else:
            return 0, f"布林带中轨附近（位置:{position_pct:.1f}%, 中轨¥{boll_mid:.2f}）"

    def _score_capital(self, df: pd.DataFrame, latest: pd.Series, prev: pd.Series) -> int:
        """
        资金维度：基于OBV和VR评分
        量价齐升 +2，量价背离 -3，缩量/中性 0
        返回 Tuple[int, str]
        """
        close = latest.get('close', np.nan)
        volume = latest.get('volume', np.nan)
        vr = latest.get('vr', np.nan)

        # 计算OBV（如果还没计算）
        if 'obv' in df.columns:
            obv_current = latest.get('obv', np.nan)
            obv_prev = prev.get('obv', np.nan)
        else:
            # 手动计算简单OBV趋势
            if len(df) >= 10:
                recent_df = df.tail(10)
                obv_trend = self._calculate_obv_trend(recent_df)
                obv_current = obv_trend
                obv_prev = 0
            else:
                obv_current = np.nan
                obv_prev = np.nan

        prev_close = prev.get('close', close)
        price_change = (close - prev_close) / prev_close if prev_close != 0 else 0

        # 判断量价关系
        if pd.isna(vr):
            return 0, "成交量数据不足"

        # 量价齐升
        if price_change > 0.02 and vr > 120:  # 涨2%以上且成交量活跃
            if not pd.isna(obv_current) and not np.isnan(obv_prev) and obv_current > obv_prev:
                return 2, f"量价齐升（涨{price_change*100:.1f}%, VR:{vr:.1f}, OBV上升）"
            return 1, f"价格上涨量能配合（涨{price_change*100:.1f}%, VR:{vr:.1f}）"

        # 量价背离（价格涨但量能不足，或OBV下降）
        if price_change > 0.02:
            if vr < 80:  # 价格上涨但成交量萎缩
                return -3, f"⚠️ 量价背离预警（涨{price_change*100:.1f}%, 但VR:{vr:.1f}缩量）"
            if not pd.isna(obv_current) and not np.isnan(obv_prev) and obv_current < obv_prev:
                return -3, f"⚠️ OBV顶背离（价格涨{price_change*100:.1f}%, OBV却下降）"

        # 价格下跌放量
        if price_change < -0.02 and vr > 150:
            return -2, f"放量下跌（跌{price_change*100:.1f}%, VR:{vr:.1f}）"

        # 缩量调整
        if vr < 80:
            return 0, f"成交量萎缩（VR:{vr:.1f}）"

        return 0, f"量能中性（VR:{vr:.1f}）"

    def _calculate_obv_trend(self, df: pd.DataFrame) -> float:
        """计算OBV趋势值（简化版）"""
        obv = 0
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv += df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv -= df['volume'].iloc[i]
        return obv

    def _score_structure(self, df: pd.DataFrame, latest: pd.Series) -> Tuple[int, str]:
        """
        结构维度：基于DMI趋势强度和关键位置评分
        趋势强且顺势 +2，趋势衰竭 -2，震荡 0
        """
        adx = latest.get('dmi_adx', np.nan)
        pdi = latest.get('dmi_pdi', np.nan)
        mdi = latest.get('dmi_mdi', np.nan)
        close = latest.get('close', np.nan)

        if pd.isna(adx) or pd.isna(pdi) or pd.isna(mdi):
            return 0, "DMI数据不足"

        # ADX > 25 表示趋势明显
        strong_trend = adx > 25
        bullish = pdi > mdi

        # 计算价格位置（相对于20日高低点）
        if len(df) >= 20:
            high_20 = df['high'].tail(20).max()
            low_20 = df['low'].tail(20).min()
            if high_20 != low_20:
                position = (close - low_20) / (high_20 - low_20) * 100
            else:
                position = 50
        else:
            position = 50

        if strong_trend:
            if bullish:
                if position > 70:
                    return 2, f"强趋势多头且接近新高（ADX:{adx:.1f}, PDI>{mdi:.1f}, 位置:{position:.0f}%）"
                else:
                    return 1, f"强趋势多头（ADX:{adx:.1f}, PDI>{mdi:.1f}）"
            else:
                if position < 30:
                    return -2, f"强趋势空头且接近新低（ADX:{adx:.1f}, MDI>{pdi:.1f}, 位置:{position:.0f}%）"
                else:
                    return -1, f"强趋势空头（ADX:{adx:.1f}, MDI>{pdi:.1f}）"
        else:
            # ADX < 25 震荡市
            if 40 < position < 60:
                return 0, f"震荡市中间位置（ADX:{adx:.1f}, 位置:{position:.0f}%）"
            elif position >= 60:
                return -1, f"震荡市偏高位（ADX:{adx:.1f}, 位置:{position:.0f}%）"
            else:
                return 1, f"震荡市偏低位（ADX:{adx:.1f}, 位置:{position:.0f}%）"

    def _score_bias(self, latest: pd.Series) -> Tuple[int, str]:
        """
        乖离率维度：基于BIAS指标评分
        BIAS反映股价与均线的偏离程度
        过度负乖离（超跌）+2，适度负乖离 +1，正常范围 0，过度正乖离（超涨）-2
        """
        bias_6 = latest.get('bias_6', np.nan)
        bias_12 = latest.get('bias_12', np.nan)
        bias_24 = latest.get('bias_24', np.nan)

        if pd.isna(bias_6):
            return 0, "乖离率数据不足"

        # BIAS(6) 评分标准（6日乖离率最敏感）
        # <-5%: 严重超跌，强烈看多信号
        # -5% ~ -3%: 轻度超跌，偏多信号
        # -3% ~ +3%: 正常范围，中性
        # +3% ~ +5%: 轻度超涨，偏空信号
        # >+5%: 严重超涨，强烈看空信号

        if bias_6 <= -5.0:
            return 2, f"严重负乖离，超跌反弹概率高（BIAS6:{bias_6:.2f}%, BIAS12:{bias_12:.2f}%, BIAS24:{bias_24:.2f}%）"
        elif bias_6 <= -3.0:
            return 1, f"负乖离，股价低于均线（BIAS6:{bias_6:.2f}%, BIAS12:{bias_12:.2f}%）"
        elif bias_6 >= 5.0:
            return -2, f"严重正乖离，超买回调风险高（BIAS6:{bias_6:.2f}%, BIAS12:{bias_12:.2f}%, BIAS24:{bias_24:.2f}%）"
        elif bias_6 >= 3.0:
            return -1, f"正乖离，股价高于均线（BIAS6:{bias_6:.2f}%, BIAS12:{bias_12:.2f}%）"
        else:
            return 0, f"乖离率正常，股价贴近均线（BIAS6:{bias_6:.2f}%, BIAS12:{bias_12:.2f}%）"

    def _get_rating_and_action(self, total_score: float, capital_score: int) -> Tuple[str, str, str]:
        """
        根据总分和操作阈值确定评级和建议

        Returns:
            (评级, 操作建议, 风险等级)
        """
        # 检查是否有严重背离
        has_divergence = capital_score <= -3

        # 调整阈值以适应6个维度（满分12分）
        if total_score > 4:  # 原来是3，现在调整为4（约33%）
            if has_divergence:
                return "谨慎偏多", "有买入信号但存在背离，建议小仓位试探", "中高风险"
            return "强烈看多", "多维度共振，可考虑加仓", "中低风险"
        elif total_score > 0:
            return "偏多观望", "偏正面但信号不强，持仓观望", "低风险"
        elif total_score >= -4:  # 原来是-3，现在调整为-4
            if has_divergence:
                return "偏空观望", "存在背离信号，建议减仓或停止开新仓", "中风险"
            return "中性观望", "信号混合，暂无明确方向，观望为主", "低风险"
        else:
            if has_divergence:
                return "强烈看空", "多重负面信号叠加背离，建议减仓离场", "高风险"
            return "看空", "负面信号占优，考虑减仓", "中高风险"

    def format_score_report(self, score_result: Dict) -> str:
        """
        格式化打分报告为 Markdown 表格
        """
        if "error" in score_result:
            return f"\n**打分失败：** {score_result['error']}\n"

        lines = []
        lines.append("\n### 多维度量化打分")
        lines.append("")

        # 总分和评级
        total = score_result['总分']
        rating = score_result['评级']
        action = score_result['操作建议']
        risk = score_result['风险等级']

        lines.append(f"**总分：{total:+d} 分** | **评级：{rating}** | **风险：{risk}**")
        lines.append("")

        # 打分表
        lines.append("| 维度 | 得分 | 分值范围 | 详细说明 |")
        lines.append("|------|------|----------|----------|")

        dimensions = score_result['各维度得分']
        for dim_name, dim_data in dimensions.items():
            score = dim_data['score']
            max_score = dim_data.get('max', 2)
            min_score = dim_data.get('min', -2)
            detail = dim_data['detail']

            # 得分颜色标记
            if score > 0:
                score_str = f"**+{score}**"
            elif score < 0:
                score_str = f"**{score}**"
            else:
                score_str = f"{score}"

            lines.append(f"| {dim_name} | {score_str} | {min_score} ~ +{max_score} | {detail} |")

        lines.append("")

        # 操作建议
        lines.append(f"#### 📊 操作建议：{action}")
        lines.append("")

        # 阈值说明
        lines.append("> **打分阈值：** >+3分买入，<-3分卖出，-3~+3分观望")
        lines.append("")

        return "\n".join(lines)
