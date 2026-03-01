"""
K线形态识别模块
实现各种经典K线形态的识别，包括单根形态和多根组合形态
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional


class CandlestickPatternRecognizer:
    """
    K线形态识别器

    支持识别的形态：
    单根形态：锤子线、倒锤子、十字星、流星线、光头光脚等
    多根组合：吞没形态、晨星、暮星、穿刺线、乌云盖顶等
    """

    def __init__(self):
        """初始化形态识别器"""
        self.patterns_detected = []

    def recognize_all_patterns(self, df: pd.DataFrame, lookback: int = 5) -> Dict:
        """
        识别近期所有K线形态

        Args:
            df: 股票数据DataFrame，需包含open, high, low, close列
            lookback: 回顾多少个交易日，默认5日

        Returns:
            Dict: 包含所有识别到的形态信息
        """
        if df.empty or len(df) < 3:
            return {"error": "数据不足，至少需要3个交易日数据"}

        self.patterns_detected = []

        # 获取最近lookback天的数据
        recent_df = df.tail(lookback).reset_index(drop=True)

        # 识别单根K线形态（最近3天）
        for i in range(max(0, len(recent_df) - 3), len(recent_df)):
            patterns = self._recognize_single_candlestick(recent_df, i)
            if patterns:
                for p in patterns:
                    p['date'] = recent_df.iloc[i].get('timestamp', None)
                    p['index'] = i
                self.patterns_detected.extend(patterns)

        # 识别多根K线组合（最近5天）
        combo_patterns = self._recognize_combination_patterns(recent_df)
        self.patterns_detected.extend(combo_patterns)

        # 分析形态统计
        bullish_count = sum(1 for p in self.patterns_detected if p.get('type') == 'bullish')
        bearish_count = sum(1 for p in self.patterns_detected if p.get('type') == 'bearish')
        neutral_count = sum(1 for p in self.patterns_detected if p.get('type') == 'neutral')

        return {
            "patterns": self.patterns_detected,
            "summary": {
                "total": len(self.patterns_detected),
                "bullish": bullish_count,
                "bearish": bearish_count,
                "neutral": neutral_count
            },
            "lookback_days": lookback
        }

    def _recognize_single_candlestick(self, df: pd.DataFrame, index: int) -> List[Dict]:
        """
        识别单根K线形态

        Args:
            df: DataFrame
            index: 要分析的行的索引

        Returns:
            List[Dict]: 识别到的形态列表
        """
        if index < 0 or index >= len(df):
            return []

        row = df.iloc[index]
        open_p = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        patterns = []

        # 计算K线基本属性
        body = abs(close - open_p)  # 实体大小
        upper_shadow = high - max(open_p, close)  # 上影线
        lower_shadow = min(open_p, close) - low  # 下影线
        total_range = high - low if high != low else 0.001  # 总波动范围（避免除以0）

        body_pct = body / total_range  # 实体占比
        upper_pct = upper_shadow / total_range  # 上影线占比
        lower_pct = lower_shadow / total_range  # 下影线占比

        # 计算相对于昨日收盘价的真实涨跌
        prev_close = df.iloc[index - 1]['close'] if index > 0 else open_p
        real_change_pct = (close - prev_close) / prev_close if prev_close > 0 else 0
        body_vs_close_pct = body / close if close > 0 else 0

        is_bullish = close > open_p  # 阳线（假阴真阳也算阳线）
        is_bearish = close < open_p  # 阴线
        is_real_bullish = real_change_pct > 0  # 真实上涨（相对昨日收盘）
        is_real_bearish = real_change_pct < 0  # 真实下跌

        # 判断是否有长影线
        has_long_upper_shadow = upper_pct > 0.3 and upper_shadow > body * 0.5
        has_long_lower_shadow = lower_pct > 0.3 and lower_shadow > body * 0.5

        # 1. 十字星（实体很小）
        if body_pct < 0.1:
            if upper_pct > 0.4 and lower_pct > 0.4:
                patterns.append({
                    "name": "长脚十字星",
                    "type": "neutral",
                    "signal": "多空平衡，趋势可能反转",
                    "strength": "中",
                    "description": f"上下影线较长，显示多空激烈争夺"
                })
            elif upper_pct > 0.3:
                patterns.append({
                    "name": "T字星",
                    "type": "neutral",
                    "signal": "下方支撑存在",
                    "strength": "弱",
                    "description": f"下影线较长，收盘价接近最高价"
                })
            elif lower_pct > 0.3:
                patterns.append({
                    "name": "倒T字星",
                    "type": "neutral",
                    "signal": "上方阻力存在",
                    "strength": "弱",
                    "description": f"上影线较长，收盘价接近最低价"
                })
            else:
                patterns.append({
                    "name": "十字星",
                    "type": "neutral",
                    "signal": "趋势可能反转",
                    "strength": "中",
                    "description": f"开盘价与收盘价接近，显示多空力量均衡"
                })

        # 2. 锤子线/吊颈线（下影线长，实体小）- 用位置区分，而非K线颜色
        # 关键修复：锤子线和吊颈线几何形状相同，区别在于出现的位置
        # - 低位出现 = 锤子线（看涨）
        # - 高位出现 = 吊颈线（看跌）
        # 判断当前位置：通过最近5日的高低点判断
        lookback_period = min(10, len(df)) if hasattr(df, '__len__') else 5
        if index >= 0 and hasattr(df, 'iloc'):
            recent_highs = [df.iloc[max(0, index - i)]['high'] for i in range(lookback_period) if max(0, index - i) < len(df)]
            recent_lows = [df.iloc[max(0, index - i)]['low'] for i in range(lookback_period) if max(0, index - i) < len(df)]
            recent_high = max(recent_highs) if recent_highs else high
            recent_low = min(recent_lows) if recent_lows else low
            # 计算当前K线在近期范围内的位置比例
            price_range = recent_high - recent_low if recent_high != recent_low else 1
            position_in_range = (close - recent_low) / price_range if price_range > 0 else 0.5
            is_at_high = position_in_range > 0.7  # 高位（接近近期高点）
            is_at_low = position_in_range < 0.3   # 低位（接近近期低点）
        else:
            is_at_high = False
            is_at_low = False

        if lower_pct > 0.6 and body_pct < 0.3:
            # 长下影线 + 小实体 - 锤子线或吊颈线
            if is_at_low:
                # 低位出现 = 锤子线（看涨）
                patterns.append({
                    "name": "锤子线",
                    "type": "bullish",
                    "signal": "可能的底部反转",
                    "strength": "中",
                    "description": f"下影线较长({lower_pct:.1%})，出现在低位，显示下方支撑强劲"
                })
            elif is_at_high:
                # 高位出现 = 吊颈线（看跌）
                patterns.append({
                    "name": "吊颈线",
                    "type": "bearish",
                    "signal": "可能的顶部反转",
                    "strength": "中",
                    "description": f"下影线较长({lower_pct:.1%})，出现在高位，警惕多头力竭"
                })
            # 中间位置 - 视为中性偏多（支撑信号）

        # 3. 倒锤子/流星线（上影线长，实体小）- 同样用位置区分
        if upper_pct > 0.6 and body_pct < 0.3:
            # 长上影线 + 小实体 - 倒锤子或流星线
            if is_at_low:
                # 低位出现 = 倒锤子（看涨）
                patterns.append({
                    "name": "倒锤子",
                    "type": "bullish",
                    "signal": "可能的底部反转",
                    "strength": "中",
                    "description": f"上影线较长({upper_pct:.1%})，出现在低位，显示多方尝试反攻"
                })
            elif is_at_high:
                # 高位出现 = 流星线（看跌）
                patterns.append({
                    "name": "流星线",
                    "type": "bearish",
                    "signal": "可能的顶部反转",
                    "strength": "中",
                    "description": f"上影线较长({upper_pct:.1%})，出现在高位，显示上方阻力强劲"
                })
            # 中间位置 - 视为中性偏空（阻力信号）

        # 6. 大阳线 - 严格定义
        # 条件：
        # 1. 实体占比 > 90%（几乎光头光脚）
        # 2. 实体相对收盘价 > 2%（确保实体足够大）
        # 3. 收盘价 > 开盘价（阳线）
        # 4. 相对昨日收盘真实上涨（避免假阴真阳误判）
        if body_pct > 0.9 and is_bullish:
            # 检查是否真实上涨（相对于昨日收盘）
            real_bullish = is_real_bullish  # 使用之前计算的 real_change_pct

            # 只有实体足够大且真实上涨才算大阳线
            if body_vs_close_pct > 0.02 and real_bullish:
                patterns.append({
                    "name": "大阳线",
                    "type": "bullish",
                    "signal": "强势上涨",
                    "strength": "强",
                    "description": f"几乎光头光脚，实体占{body_pct:.1%}，多方力量强劲"
                })

        # 6.1 长上影线警示（抛压）
        # 当上影线占比 > 30% 且实体较小时，显示上方抛压
        if upper_pct > 0.3 and body_pct < 0.5 and is_bullish:
            # 长上影线说明冲高回落，上方有阻力
            patterns.append({
                "name": "长上影线",
                "type": "neutral",
                "signal": "上方抛压",
                "strength": "中",
                "description": f"上影线占{upper_pct:.1%}，冲高回落显示上方阻力"
            })

        # 6.2 假阴真阳识别
        # 收盘价 > 开盘价，但相对昨日下跌（低开高走但仍收跌）
        if is_bullish and is_real_bearish:  # 阳线但真实下跌
            patterns.append({
                "name": "假阴真阳",
                "type": "neutral",
                "signal": "弱势反弹",
                "strength": "弱",
                "description": f"收盘价高于开盘价但低于昨日收盘，实体仅{body_vs_close_pct:.2%}，弱势整理"
            })

        # 7. 光头光脚大阴线
        if body_pct > 0.9 and is_bearish:
            patterns.append({
                "name": "大阴线",
                "type": "bearish",
                "signal": "强势下跌",
                "strength": "强",
                "description": f"几乎光头光脚，空方力量强劲"
            })

        # 8. 纺锤线（小实体，影线较短）- 趋势中继或反转信号
        if body_pct < 0.3 and upper_pct < 0.4 and lower_pct < 0.4:
            patterns.append({
                "name": "纺锤线",
                "type": "neutral",
                "signal": "动能减弱",
                "strength": "弱",
                "description": f"实体较小，市场犹豫不决"
            })

        return patterns

    def _recognize_combination_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        识别多根K线组合形态

        Args:
            df: DataFrame（最近5天数据）

        Returns:
            List[Dict]: 识别到的组合形态列表
        """
        if len(df) < 3:
            return []

        patterns = []
        n = len(df)

        # 获取最近3天的数据用于识别
        c1, c2, c3 = df['close'].iloc[-3:].values
        o1, o2, o3 = df['open'].iloc[-3:].values
        h1, h2, h3 = df['high'].iloc[-3:].values
        l1, l2, l3 = df['low'].iloc[-3:].values

        # 1. 吞没形态（需要2根K线）
        if n >= 2:
            # 看涨吞没
            if c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1:
                patterns.append({
                    "name": "看涨吞没",
                    "type": "bullish",
                    "signal": "底部反转信号",
                    "strength": "强",
                    "description": f"阳线完全包住前一日阴线，多方力量强劲",
                    "date": df.iloc[-1].get('timestamp', None)
                })

            # 看跌吞没
            if c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1:
                patterns.append({
                    "name": "看跌吞没",
                    "type": "bearish",
                    "signal": "顶部反转信号",
                    "strength": "强",
                    "description": f"阴线完全包住前一日阳线，空方力量强劲",
                    "date": df.iloc[-1].get('timestamp', None)
                })

        # 2. 穿刺线（需要2根K线，看涨）
        if n >= 2:
            body1 = abs(c1 - o1)
            if c1 < o1 and c2 > o2 and o2 < l1 and c2 > (o1 + c1) / 2 and c2 < o1:
                patterns.append({
                    "name": "穿刺线",
                    "type": "bullish",
                    "signal": "可能的底部反转",
                    "strength": "中",
                    "description": f"阳线收盘深入前一日阴线实体中部",
                    "date": df.iloc[-1].get('timestamp', None)
                })

        # 3. 乌云盖顶（需要2根K线，看跌）
        if n >= 2:
            if c1 > o1 and c2 < o2 and o2 > h1 and c2 < (o1 + c1) / 2 and c2 > o1:
                patterns.append({
                    "name": "乌云盖顶",
                    "type": "bearish",
                    "signal": "可能的顶部反转",
                    "strength": "中",
                    "description": f"阴线收盘深入前一日阳线实体中部",
                    "date": df.iloc[-1].get('timestamp', None)
                })

        # 4. 晨星（需要3根K线，看涨）
        if n >= 3:
            body1 = abs(c1 - o1)
            body2 = abs(c2 - o2)
            body3 = abs(c3 - o3)

            if c1 < o1 and body2 < body1 * 0.3 and c3 > o3 and c3 > (o1 + c1) / 2:
                # 第一天阴线，第二天小实体，第三天阳线且收盘深入第一天实体
                patterns.append({
                    "name": "晨星",
                    "type": "bullish",
                    "signal": "强烈的底部反转",
                    "strength": "强",
                    "description": f"三日K线组合，显示空头力量衰竭，多头反击",
                    "date": df.iloc[-1].get('timestamp', None)
                })

        # 5. 暮星（需要3根K线，看跌）
        if n >= 3:
            if c1 > o1 and body2 < body1 * 0.3 and c3 < o3 and c3 < (o1 + c1) / 2:
                # 第一天阳线，第二天小实体，第三天阴线且收盘深入第一天实体
                patterns.append({
                    "name": "暮星",
                    "type": "bearish",
                    "signal": "强烈的顶部反转",
                    "strength": "强",
                    "description": f"三日K线组合，显示多头力量衰竭，空头反击",
                    "date": df.iloc[-1].get('timestamp', None)
                })

        # 6. 白色三兵（连续3根阳线，看涨）
        if n >= 3:
            if c1 > o1 and c2 > o2 and c3 > o3 and c2 > c1 and c3 > c2:
                patterns.append({
                    "name": "白色三兵",
                    "type": "bullish",
                    "signal": "强势上涨延续",
                    "strength": "强",
                    "description": f"连续三根阳线，且收盘价逐步抬高",
                    "date": df.iloc[-1].get('timestamp', None)
                })

        # 7. 黑色三鸦（连续3根阴线，看跌）
        if n >= 3:
            if c1 < o1 and c2 < o2 and c3 < o3 and c2 < c1 and c3 < c2:
                patterns.append({
                    "name": "黑色三鸦",
                    "type": "bearish",
                    "signal": "强势下跌延续",
                    "strength": "强",
                    "description": f"连续三根阴线，且收盘价逐步降低",
                    "date": df.iloc[-1].get('timestamp', None)
                })

        return patterns

    def format_patterns_report(self, patterns_result: Dict) -> str:
        """
        将形态识别结果格式化为Markdown报告

        Args:
            patterns_result: recognize_all_patterns的返回结果

        Returns:
            str: Markdown格式的报告
        """
        if "error" in patterns_result:
            return f"\n**K线形态分析：** {patterns_result['error']}\n"

        lines = []
        lines.append("\n### K线形态分析")
        lines.append("")

        summary = patterns_result.get("summary", {})
        patterns = patterns_result.get("patterns", [])

        if not patterns:
            lines.append("最近5个交易日未识别到明显K线形态。")
            lines.append("")
            return "\n".join(lines)

        # 形态统计
        lines.append("**形态统计：**")
        lines.append(f"- 看涨形态：{summary.get('bullish', 0)}个")
        lines.append(f"- 看跌形态：{summary.get('bearish', 0)}个")
        lines.append(f"- 中性形态：{summary.get('neutral', 0)}个")
        lines.append("")

        # 按类型分组显示
        bullish_patterns = [p for p in patterns if p.get('type') == 'bullish']
        bearish_patterns = [p for p in patterns if p.get('type') == 'bearish']
        neutral_patterns = [p for p in patterns if p.get('type') == 'neutral']

        # 显示看涨形态
        if bullish_patterns:
            lines.append("**📈 看涨形态：**")
            lines.append("")
            for p in bullish_patterns[-3:]:  # 最多显示最近3个
                lines.append(f"- **{p['name']}**（强度：{p['strength']}）")
                lines.append(f"  - 信号：{p['signal']}")
                lines.append(f"  - {p['description']}")
            lines.append("")

        # 显示看跌形态
        if bearish_patterns:
            lines.append("**📉 看跌形态：**")
            lines.append("")
            for p in bearish_patterns[-3:]:
                lines.append(f"- **{p['name']}**（强度：{p['strength']}）")
                lines.append(f"  - 信号：{p['signal']}")
                lines.append(f"  - {p['description']}")
            lines.append("")

        # 显示中性形态
        if neutral_patterns:
            lines.append("**➖ 中性形态：**")
            lines.append("")
            for p in neutral_patterns[-2:]:
                lines.append(f"- **{p['name']}**（强度：{p['strength']}）")
                lines.append(f"  - 信号：{p['signal']}")
                lines.append(f"  - {p['description']}")
            lines.append("")

        return "\n".join(lines)


def analyze_candlestick_patterns(df: pd.DataFrame, lookback: int = 5) -> Dict:
    """
    便捷的K线形态分析函数

    Args:
        df: 股票数据DataFrame
        lookback: 回顾天数

    Returns:
        Dict: 形态分析结果
    """
    recognizer = CandlestickPatternRecognizer()
    return recognizer.recognize_all_patterns(df, lookback)


def format_candlestick_report(patterns_result: Dict) -> str:
    """
    便捷的报告格式化函数

    Args:
        patterns_result: 形态分析结果

    Returns:
        str: Markdown格式的报告
    """
    recognizer = CandlestickPatternRecognizer()
    return recognizer.format_patterns_report(patterns_result)


def get_pattern_assessment(patterns_result: Dict,
                           position_ratio: float = 0.5,
                           bias20: float = 0.0,
                           boll_pctb: float = 0.5) -> str:
    """
    基于"位置+形态"逻辑对K线形态进行定性评估（不调整分数）

    核心逻辑：位置决定形态意义
    - 低位 + 看涨形态 = 强力底部信号
    - 高位 + 看涨形态 = 警惕诱多/力竭
    - 高位 + 看跌形态 = 强力顶部信号
    - 低位 + 看跌形态 = 可能是最后洗盘

    Args:
        patterns_result: recognize_all_patterns的返回结果
        position_ratio: 股价相对60日高低点的位置 (0-1)
        bias20: MA20乖离率 (用于辅助判断)
        boll_pctb: 布林带百分比位置 (0-1, 0=下轨, 1=上轨)

    Returns:
        str: 定性评估描述
    """
    if "error" in patterns_result or not patterns_result.get("patterns"):
        return ""

    patterns = patterns_result.get("patterns", [])

    # 判断位置
    is_low_position = position_ratio < 0.35 or bias20 < -0.05 or boll_pctb < 0.2
    is_high_position = position_ratio > 0.70 or bias20 > 0.05 or boll_pctb > 0.8

    if is_low_position:
        position_desc = "低位"
    elif is_high_position:
        position_desc = "高位"
    else:
        position_desc = "中位"

    # 形态分类
    strong_bullish = ["晨星", "看涨吞没", "白色三兵"]
    medium_bullish = ["锤子线", "倒锤子", "穿刺线"]
    strong_bearish = ["暮星", "看跌吞没", "黑色三鸦"]
    medium_bearish = ["流星线", "吊颈线", "乌云盖顶"]

    assessments = []

    for p in patterns:
        name = p.get("name", "")
        strength = p.get("strength", "中")

        # 低位 + 看涨形态 = 强力底部信号
        if is_low_position and name in (strong_bullish + medium_bullish):
            if name in strong_bullish:
                assessments.append(f"【强力底部信号】{position_desc}出现强{name}，底部反转概率高")
            else:
                assessments.append(f"【底部信号】{position_desc}出现{name}，关注反弹机会")

        # 高位 + 看跌形态 = 强力顶部信号
        elif is_high_position and name in (strong_bearish + medium_bearish):
            if name in strong_bearish:
                assessments.append(f"【强力顶部信号】{position_desc}出现强{name}，顶部确认")
            else:
                assessments.append(f"【顶部信号】{position_desc}出现{name}，警惕回调")

        # 高位 + 看涨形态 = 警惕诱多
        elif is_high_position and name in (strong_bullish + medium_bullish):
            if name in strong_bullish:
                assessments.append(f"【警惕】{position_desc}出现{name}，可能是诱多/力竭")
            else:
                assessments.append(f"【中性】{position_desc}出现{name}，需量能确认")

        # 低位 + 看跌形态 = 可能是最后洗盘
        elif is_low_position and name in (strong_bearish + medium_bearish):
            assessments.append(f"【洗盘信号】{position_desc}出现{name}，可能是最后恐慌洗盘")

        # 中位区域
        elif not is_low_position and not is_high_position:
            if name in strong_bullish:
                assessments.append(f"【延续信号】{position_desc}出现{name}，趋势可能延续")
            elif name in strong_bearish:
                assessments.append(f"【调整信号】{position_desc}出现{name}，关注调整深度")

    return "; ".join(assessments) if assessments else ""


def draw_candlestick_ascii(open_p: float, high: float, low: float, close: float,
                           width: int = 11, height: int = 15) -> str:
    """
    绘制单根K线的ASCII艺术图

    Args:
        open_p: 开盘价
        high: 最高价
        low: 最低价
        close: 收盘价
        width: K线宽度（字符数）
        height: K线高度（行数）

    Returns:
        str: ASCII艺术K线图
    """
    # 判断阴阳线
    is_bullish = close >= open_p

    # 计算价格范围
    price_range = high - low
    if price_range == 0:
        price_range = 0.01  # 避免除零

    # 创建画布
    canvas = [[' ' for _ in range(width)] for _ in range(height)]

    # 计算各部分的行位置（从上到下）
    # 行0是最高价，行height-1是最低价
    high_row = 0
    low_row = height - 1

    # 计算实体位置
    body_top = max(open_p, close)
    body_bottom = min(open_p, close)

    body_top_row = int((high - body_top) / price_range * (height - 1))
    body_bottom_row = int((high - body_bottom) / price_range * (height - 1))

    # 确保至少有一行实体
    if body_top_row == body_bottom_row:
        body_bottom_row = min(body_top_row + 1, height - 1)

    # 选择字符
    if is_bullish:
        body_char = '█'  # 阳线实体
        shadow_char = '│'  # 影线
    else:
        body_char = '▓'  # 阴线实体
        shadow_char = '│'  # 影线

    mid_col = width // 2

    # 绘制上影线
    for row in range(high_row, body_top_row):
        canvas[row][mid_col] = shadow_char

    # 绘制实体
    for row in range(body_top_row, body_bottom_row + 1):
        for col in range(max(0, mid_col - 2), min(width, mid_col + 3)):
            canvas[row][col] = body_char

    # 绘制下影线
    for row in range(body_bottom_row + 1, low_row + 1):
        canvas[row][mid_col] = shadow_char

    # 转换为字符串
    lines = []
    for row in canvas:
        lines.append(''.join(row))

    return '\n'.join(lines)


def draw_candlestick_chart(df: pd.DataFrame, num_candles: int = 10,
                           width_per_candle: int = 7, height: int = 15) -> str:
    """
    绘制K线图的ASCII艺术图

    Args:
        df: 股票数据DataFrame，需包含open, high, low, close列
        num_candles: 显示的K线数量
        width_per_candle: 每根K线的宽度
        height: 图表高度

    Returns:
        str: ASCII艺术K线图
    """
    if df.empty or len(df) < 1:
        return "无数据"

    # 取最近N根K线
    recent_df = df.tail(num_candles).reset_index(drop=True)

    # 计算全局价格范围
    all_high = recent_df['high'].max()
    all_low = recent_df['low'].min()
    price_range = all_high - all_low
    if price_range == 0:
        price_range = 0.01

    # 创建画布
    total_width = num_candles * width_per_candle
    canvas = [[' ' for _ in range(total_width)] for _ in range(height)]

    # 为每根K线绘制
    for i, row in recent_df.iterrows():
        open_p = row['open']
        high = row['high']
        low = row['low']
        close = row['close']

        is_bullish = close >= open_p

        # 计算各部分位置
        high_row = int((all_high - high) / price_range * (height - 1))
        low_row = int((all_high - low) / price_range * (height - 1))

        body_top = max(open_p, close)
        body_bottom = min(open_p, close)
        body_top_row = int((all_high - body_top) / price_range * (height - 1))
        body_bottom_row = int((all_high - body_bottom) / price_range * (height - 1))

        # 确保实体至少有一行
        if body_top_row >= body_bottom_row:
            body_bottom_row = max(body_top_row + 1, min(body_top_row + 1, height - 1))

        # 选择字符（带颜色）
        # ANSI颜色代码：\033[92m 绿色（阳线），\033[91m 红色（阴线），\033[0m 重置
        # 注意：在Markdown代码块中颜色可能不显示，但终端中有效
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'

        if is_bullish:
            body_char = GREEN + '█' + RESET  # 绿色阳线
        else:
            body_char = RED + '█' + RESET  # 红色阴线
        shadow_char = '│'

        # 计算这根K线的列范围
        col_start = i * width_per_candle + width_per_candle // 2
        col_start = min(col_start, total_width - 2)

        # 绘制上影线
        for r in range(high_row, body_top_row):
            if 0 <= r < height and 0 <= col_start < total_width:
                canvas[r][col_start] = shadow_char

        # 绘制实体
        for r in range(body_top_row, min(body_bottom_row + 1, height)):
            for c in range(max(0, col_start - 1), min(total_width, col_start + 2)):
                if 0 <= r < height:
                    canvas[r][c] = body_char

        # 绘制下影线
        for r in range(min(body_bottom_row + 1, height), min(low_row + 1, height)):
            if 0 <= col_start < total_width:
                canvas[r][col_start] = shadow_char

    # 添加价格标签
    result_lines = []
    result_lines.append(f"最高: ¥{all_high:.2f}")
    result_lines.append("")

    for row in canvas:
        result_lines.append(''.join(row))

    result_lines.append("")
    result_lines.append(f"最低: ¥{all_low:.2f}")

    # 添加日期范围
    if 'trade_date' in recent_df.columns:
        start_date = recent_df['trade_date'].iloc[0]
        end_date = recent_df['trade_date'].iloc[-1]
        result_lines.append(f"日期: {start_date} ~ {end_date}")
    elif 'timestamp' in recent_df.columns:
        start_date = recent_df['timestamp'].iloc[0]
        end_date = recent_df['timestamp'].iloc[-1]
        result_lines.append(f"日期: {start_date} ~ {end_date}")

    return '\n'.join(result_lines)


def draw_pattern_illustration(pattern_name: str) -> str:
    """
    绘制K线形态的示意图

    Args:
        pattern_name: 形态名称

    Returns:
        str: ASCII艺术示意图
    """
    illustrations = {
        "锤子线": """
    ┌─────────────────┐
    │      │          │  上影线短或无
    │    ┌───┐        │
    │    │   │ 小实体 │  实体在上方
    │    └───┘        │
    │       │         │
    │       │         │  长下影线
    │       │         │  (>=实体2倍)
    │       │         │
    └─────────────────┘
    出现在低位 = 看涨反转信号
""",
        "吊颈线": """
    ┌─────────────────┐
    │      │          │  上影线短或无
    │    ┌───┐        │
    │    │   │ 小实体 │  实体在上方
    │    └───┘        │
    │       │         │
    │       │         │  长下影线
    │       │         │  (>=实体2倍)
    │       │         │
    └─────────────────┘
    出现在高位 = 看跌反转信号
    ⚠️ 形状与锤子线相同，区别在位置！
""",
        "流星线": """
    ┌─────────────────┐
    │       │         │
    │       │         │  长上影线
    │       │         │  (>=实体2倍)
    │       │         │
    │    ┌───┐        │
    │    │   │ 小实体 │  实体在下方
    │    └───┘        │
    │      │          │  下影线短或无
    └─────────────────┘
    出现在高位 = 看跌反转信号
""",
        "倒锤子": """
    ┌─────────────────┐
    │       │         │
    │       │         │  长上影线
    │       │         │  (>=实体2倍)
    │       │         │
    │    ┌───┐        │
    │    │   │ 小实体 │  实体在下方
    │    └───┘        │
    │      │          │  下影线短或无
    └─────────────────┘
    出现在低位 = 看涨反转信号
""",
        "大阳线": """
    ┌─────────────────┐
    │    ┌───┐        │
    │    │   │        │
    │    │   │        │
    │    │   │ 大实体 │  几乎光头光脚
    │    │   │        │  实体占比>90%
    │    │   │        │
    │    │   │        │
    │    └───┘        │
    └─────────────────┘
    强势看涨信号
""",
        "大阴线": """
    ┌─────────────────┐
    │    ┌───┐        │
    │    │▓▓▓│        │
    │    │▓▓▓│        │
    │    │▓▓▓│ 大实体 │  几乎光头光脚
    │    │▓▓▓│        │  实体占比>90%
    │    │▓▓▓│        │
    │    │▓▓▓│        │
    │    └───┘        │
    └─────────────────┘
    强势看跌信号
""",
        "十字星": """
    ┌─────────────────┐
    │       │         │
    │       │         │  上影线
    │    ─────         │  极小实体/十字
    │       │         │  下影线
    │       │         │
    └─────────────────┘
    多空平衡，趋势可能反转
""",
        "看涨吞没": """
    ┌─────────────────┐
    │   ┌─────────┐   │
    │   │  ┌───┐  │   │  小阴线被大阳线吞没
    │   │  │▓▓▓│  │   │
    │   │  └───┘  │   │
    │   │    ████ │   │
    │   │    ████ │   │  大阳线完全包含前一根
    │   └─────────┘   │
    └─────────────────┘
    出现在下跌趋势末端 = 强烈看涨
""",
        "看跌吞没": """
    ┌─────────────────┐
    │   ┌─────────┐   │
    │   │  ┌───┐  │   │  小阳线被大阴线吞没
    │   │  │███│  │   │
    │   │  └───┘  │   │
    │   │    ▓▓▓▓ │   │
    │   │    ▓▓▓▓ │   │  大阴线完全包含前一根
    │   └─────────┘   │
    └─────────────────┘
    出现在上涨趋势末端 = 强烈看跌
""",
        "长上影线": """
    ┌─────────────────┐
    │       │         │
    │       │         │  长上影线
    │       │         │  占据主要部分
    │       │         │  (>=70%)
    │    ┌───┐        │
    │    │   │ 小实体 │  实体在下方
    │    └───┘        │
    │                 │  下影线短或无
    └─────────────────┘
    含义：冲高回落，上方抛压重
    高位出现 = 看跌信号
    低位出现 = 试盘信号
""",
        "长下影线": """
    ┌─────────────────┐
    │      │          │  上影线短或无
    │    ┌───┐        │
    │    │   │ 小实体 │  实体在上方
    │    └───┘        │
    │       │         │
    │       │         │  长下影线
    │       │         │  占据主要部分
    │       │         │  (>=70%)
    └─────────────────┘
    含义：探底回升，下方有支撑
    低位出现 = 看涨信号
    高位出现 = 可能是吊颈线
""",
        "纺锤线": """
    ┌─────────────────┐
    │      │          │  影线较短
    │    ┌───┐        │
    │    │   │ 小实体 │  实体很小
    │    └───┘        │
    │      │          │  影线较短
    └─────────────────┘
    含义：多空平衡，市场犹豫
    趋势中继或反转信号
""",
    }

    # 默认形态
    default_illustration = """
    ┌─────────────────┐
    │    ┌───┐        │
    │    │   │        │
    │    │   │ 实体   │
    │    │   │        │
    │    └───┘        │
    │       │         │
    │       │         │  影线
    └─────────────────┘
"""

    return illustrations.get(pattern_name, default_illustration)
