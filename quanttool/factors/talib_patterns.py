"""
TA-Lib K线形态识别模块

使用 TA-Lib 实现61种经典K线形态识别，与通达信/同花顺结果一致。

支持的形态列表：
1. 两只乌鸦 (CDL2CROWS)
2. 三只乌鸦 (CDL3BLACKCROWS)
3. 三内部上涨/下跌 (CDL3INSIDE)
4. 三线打击 (CDL3LINESTRIKE)
5. 三外部上涨/下跌 (CDL3OUTSIDE)
6. 南方三星 (CDL3STARSINSOUTH)
7. 三个白兵 (CDL3WHITESOLDIERS)
8. 弃婴 (CDLABANDONEDBABY)
9. 大敌当前 (CDLADVANCEBLOCK)
10. 捉腰带线 (CDLBELTHOLD)
... 共61种

结果含义：
- 负值：看跌信号（出现卖出信号）
- 0：没有出现该形态
- 正值：看涨信号（出现买入信号）
- 绝对值越大，信号越强
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# 尝试导入talib
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    warnings.warn("TA-Lib not installed. K-line pattern recognition will be limited.")


@dataclass
class PatternResult:
    """单个形态识别结果"""
    name: str           # 形态名称
    name_cn: str        # 中文名称
    signal: int         # 信号值 (负=看跌, 0=无, 正=看涨)
    strength: str       # 信号强度 (弱/中/强)
    type: str           # 类型 (bullish/bearish/neutral)
    description: str    # 描述
    date: Optional[str] = None  # 出现日期


@dataclass
class AllPatternsResult:
    """所有形态识别结果"""
    patterns: List[PatternResult]
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_patterns: int
    composite_signal: float  # 综合信号 (-100 到 100)


# TA-Lib 形态配置
# 格式: (函数名, 中文名称, 类型描述, 描述, 看涨中文名, 看跌中文名)
# 对于 neutral 类型的形态，需要提供看涨和看跌两个方向的名称
#
# 重要分类说明：
# - bullish: 明确看涨形态（如晨星、锤头等）
# - bearish: 明确看跌形态（如暮星、上吊线等）
# - neutral: 中性形态，需要根据信号方向判断多空，但统计时归为中性
# - weak_neutral: 弱中性形态（如短蜡烛、纺锤等），统计时不计入多空
TALIB_PATTERNS = [
    # 明确看涨形态
    ('CDL3LINESTRIKE', '三线打击', 'bullish', '下跌后的强势反转', None, None),
    ('CDL3STARSINSOUTH', '南方三星', 'bullish', '底部反转信号', None, None),
    ('CDL3WHITESOLDIERS', '三个白兵', 'bullish', '连续三根阳线，看涨', None, None),
    ('CDLABANDONEDBABY', '弃婴', 'bullish', '底部反转形态', None, None),
    ('CDLBREAKAWAY', '脱离', 'bullish', '脱离形态', None, None),
    ('CDLCONCEALBABYSWALL', '藏婴吞没', 'bullish', '底部反转', None, None),
    ('CDLDRAGONFLYDOJI', '蜻蜓十字', 'bullish', '底部反转信号', None, None),
    ('CDLHAMMER', '锤头', 'bullish', '底部反转信号', None, None),
    ('CDLHOMINGPIGEON', '家鸽', 'bullish', '底部反转信号', None, None),
    ('CDLINVERTEDHAMMER', '倒锤头', 'bullish', '底部反转信号', None, None),
    ('CDLKICKING', '反冲形态', 'bullish', '强势反转', None, None),
    ('CDLKICKINGBYLENGTH', '长影反冲', 'bullish', '强势反转', None, None),
    ('CDLLADDERBOTTOM', '梯底', 'bullish', '底部反转', None, None),
    ('CDLMATCHINGLOW', '相同低价', 'bullish', '支撑确认', None, None),
    ('CDLMATHOLD', '铺垫', 'bullish', '趋势延续', None, None),
    ('CDLMORNINGDOJISTAR', '十字晨星', 'bullish', '底部反转信号', None, None),
    ('CDLMORNINGSTAR', '晨星', 'bullish', '底部反转信号', None, None),
    ('CDLPIERCING', '刺透形态', 'bullish', '底部反转信号', None, None),
    ('CDLSTICKSANDWICH', '条形三明治', 'bullish', '底部反转', None, None),
    ('CDLTAKURI', '探水竿', 'bullish', '底部支撑', None, None),
    ('CDLUNIQUE3RIVER', '奇特三河床', 'bullish', '底部反转', None, None),

    # 明确看跌形态
    ('CDL2CROWS', '两只乌鸦', 'bearish', '三根K线看跌形态', None, None),
    ('CDL3BLACKCROWS', '三只乌鸦', 'bearish', '连续三根阴线，看跌', None, None),
    ('CDLADVANCEBLOCK', '大敌当前', 'bearish', '上涨受阻信号', None, None),
    ('CDLDARKCLOUDCOVER', '乌云盖顶', 'bearish', '顶部反转信号', None, None),
    ('CDLEVENINGDOJISTAR', '十字暮星', 'bearish', '顶部反转信号', None, None),
    ('CDLEVENINGSTAR', '暮星', 'bearish', '顶部反转信号', None, None),
    ('CDLGRAVESTONEDOJI', '墓碑十字', 'bearish', '顶部反转信号', None, None),
    ('CDLHANGINGMAN', '上吊线', 'bearish', '顶部反转信号', None, None),
    ('CDLIDENTICAL3CROWS', '三胞胎乌鸦', 'bearish', '看跌形态', None, None),
    ('CDLINNECK', '颈内线', 'bearish', '看跌延续', None, None),
    ('CDLONNECK', '颈上线', 'bearish', '看跌延续', None, None),
    ('CDLSHOOTINGSTAR', '射击之星', 'bearish', '顶部反转信号', None, None),
    ('CDLSTALLEDPATTERN', '停顿形态', 'bearish', '上涨停顿', None, None),
    ('CDLTHRUSTING', '插入', 'bearish', '看跌延续', None, None),
    ('CDLUPSIDEGAP2CROWS', '跳空双鸦', 'bearish', '看跌形态', None, None),

    # 方向性中性形态（根据信号方向判断多空，统计时计入多空）
    ('CDL3INSIDE', '孕线', 'directional', '孕线形态', '三内部上涨', '三内部下跌'),
    ('CDL3OUTSIDE', '外包线', 'directional', '外包线形态', '三外部上涨', '三外部下跌'),
    ('CDLBELTHOLD', '捉腰带线', 'directional', '单日反转信号', '看涨捉腰带', '看跌捉腰带'),
    ('CDLCLOSINGMARUBOZU', '收盘缺影线', 'directional', '强势K线', '收盘阳线', '收盘阴线'),
    ('CDLCOUNTERATTACK', '反击线', 'directional', '反转信号', '看涨反击线', '看跌反击线'),
    ('CDLENGULFING', '吞没形态', 'directional', '强势反转信号', '看涨吞没', '看跌吞没'),
    ('CDLHIKKAKE', '陷阱', 'directional', '趋势陷阱', '看涨陷阱', '看跌陷阱'),
    ('CDLHIKKAKEMOD', '修正陷阱', 'directional', '趋势陷阱', '看涨修正陷阱', '看跌修正陷阱'),
    ('CDLLONGLINE', '长蜡烛', 'directional', '强势K线', '长阳线', '长阴线'),
    ('CDLMARUBOZU', '光头光脚', 'directional', '强势K线', '光头光脚阳', '光头光脚阴'),
    ('CDLRISEFALL3METHODS', '三法', 'directional', '趋势延续', '上升三法', '下降三法'),
    ('CDLSEPARATINGLINES', '分离线', 'directional', '趋势反转', '看涨分离线', '看跌分离线'),
    ('CDLTRISTAR', '三星', 'directional', '趋势反转', '看涨三星', '看跌三星'),
    ('CDLXSIDEGAP3METHODS', '跳空三法', 'directional', '趋势延续', '上升跳空三法', '下降跳空三法'),

    # 弱中性形态（统计时不计入多空，仅作为市场状态参考）
    ('CDLDOJI', '十字星', 'weak_neutral', '多空平衡', None, None),
    ('CDLDOJISTAR', '十字星', 'weak_neutral', '趋势反转信号', None, None),
    ('CDLGAPSIDESIDEWHITE', '跳空并列阳线', 'weak_neutral', '趋势延续', None, None),
    ('CDLHARAMI', '母子线', 'weak_neutral', '趋势停顿信号', None, None),
    ('CDLHARAMICROSS', '十字孕线', 'weak_neutral', '趋势停顿信号', None, None),
    ('CDLHIGHWAVE', '风高浪大线', 'weak_neutral', '市场犹豫', None, None),
    ('CDLLONGLEGGEDDOJI', '长脚十字', 'weak_neutral', '市场犹豫', None, None),
    ('CDLRICKSHAWMAN', '黄包车夫', 'weak_neutral', '市场犹豫', None, None),
    ('CDLSHORTLINE', '短蜡烛', 'weak_neutral', '市场犹豫', None, None),
    ('CDLSPINNINGTOP', '纺锤', 'weak_neutral', '市场犹豫', None, None),
    ('CDLTASUKIGAP', '跳空并列阴阳线', 'weak_neutral', '趋势延续', None, None),
]

# 方向性形态名称映射
DIRECTIONAL_PATTERN_NAMES = {
    'CDL3INSIDE': ('三内部上涨', '三内部下跌'),
    'CDL3OUTSIDE': ('三外部上涨', '三外部下跌'),
    'CDLBELTHOLD': ('看涨捉腰带', '看跌捉腰带'),
    'CDLCLOSINGMARUBOZU': ('收盘阳线', '收盘阴线'),
    'CDLCOUNTERATTACK': ('看涨反击线', '看跌反击线'),
    'CDLENGULFING': ('看涨吞没', '看跌吞没'),
    'CDLHIKKAKE': ('看涨陷阱', '看跌陷阱'),
    'CDLHIKKAKEMOD': ('看涨修正陷阱', '看跌修正陷阱'),
    'CDLLONGLINE': ('长阳线', '长阴线'),
    'CDLMARUBOZU': ('光头光脚阳', '光头光脚阴'),
    'CDLRISEFALL3METHODS': ('上升三法', '下降三法'),
    'CDLSEPARATINGLINES': ('看涨分离线', '看跌分离线'),
    'CDLTRISTAR': ('看涨三星', '看跌三星'),
    'CDLXSIDEGAP3METHODS': ('上升跳空三法', '下降跳空三法'),
}


class TalibPatternRecognizer:
    """
    TA-Lib K线形态识别器

    支持61种经典K线形态识别
    """

    def __init__(self):
        """初始化形态识别器"""
        self.patterns_config = {p[0]: p for p in TALIB_PATTERNS}

    def recognize_all(
        self,
        df: pd.DataFrame,
        lookback: int = 5
    ) -> AllPatternsResult:
        """
        识别所有K线形态

        Args:
            df: 股票数据DataFrame，需包含 open, high, low, close 列
            lookback: 回顾天数，默认5日

        Returns:
            AllPatternsResult: 所有形态识别结果
        """
        if not TALIB_AVAILABLE:
            return self._empty_result("TA-Lib 未安装")

        if df is None or len(df) < 5:
            return self._empty_result("数据不足")

        # 提取数据
        open_price = df['open'].values
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values

        patterns_found = []
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        total_signal = 0.0

        # 获取最近的日期
        recent_dates = df['timestamp'].tail(lookback).values if 'timestamp' in df.columns else None

        # 遍历所有形态（新格式：6个元素）
        for pattern_info in TALIB_PATTERNS:
            pattern_name = pattern_info[0]
            name_cn = pattern_info[1]
            pattern_type = pattern_info[2]  # bullish, bearish, directional, weak_neutral
            description = pattern_info[3]
            # 可选的方向性名称
            bullish_name = pattern_info[4] if len(pattern_info) > 4 else None
            bearish_name = pattern_info[5] if len(pattern_info) > 5 else None

            try:
                # 获取TA-Lib函数
                func = getattr(talib, pattern_name)
                result = func(open_price, high_price, low_price, close_price)

                # 检查最近lookback天是否有形态出现
                for i in range(-lookback, 0):
                    signal = int(result.iloc[i] if hasattr(result, 'iloc') else result[i])

                    if signal != 0:
                        # 确定形态类型和统计方式
                        actual_type = pattern_type
                        is_neutral_for_count = False  # 是否在统计时归为中性

                        if pattern_type == 'bullish':
                            # 明确看涨形态，信号应为正
                            if signal < 0:
                                continue  # 忽略反向信号
                        elif pattern_type == 'bearish':
                            # 明确看跌形态，信号应为负
                            if signal > 0:
                                continue  # 忽略反向信号
                        elif pattern_type == 'directional':
                            # 方向性中性形态，根据信号方向确定
                            actual_type = 'bullish' if signal > 0 else 'bearish'
                        elif pattern_type == 'weak_neutral':
                            # 弱中性形态，统计时归为中性
                            is_neutral_for_count = True
                            actual_type = 'neutral'

                        # 根据信号方向选择名称
                        if signal > 0 and bullish_name:
                            actual_name_cn = bullish_name
                        elif signal < 0 and bearish_name:
                            actual_name_cn = bearish_name
                        else:
                            actual_name_cn = name_cn

                        # 确定强度
                        strength = self._get_strength(abs(signal))

                        # 获取日期
                        date = None
                        if recent_dates is not None:
                            idx = i + lookback  # 修正日期索引
                            if 0 <= idx < len(recent_dates):
                                date = str(recent_dates[idx])[:10]  # 只取日期部分

                        pattern_result = PatternResult(
                            name=pattern_name,
                            name_cn=actual_name_cn,
                            signal=signal,
                            strength=strength,
                            type=actual_type,
                            description=description,
                            date=date
                        )

                        patterns_found.append(pattern_result)

                        # 统计（弱中性形态不计入多空）
                        if is_neutral_for_count:
                            neutral_count += 1
                        elif actual_type == 'bullish':
                            bullish_count += 1
                            total_signal += abs(signal)
                        elif actual_type == 'bearish':
                            bearish_count += 1
                            total_signal -= abs(signal)
                        else:
                            neutral_count += 1

            except Exception as e:
                # 忽略无法识别的形态
                continue

        # 计算综合信号（只统计明确多空形态）
        if bullish_count + bearish_count > 0:
            composite_signal = total_signal / (bullish_count + bearish_count) * 10
            composite_signal = max(-100, min(100, composite_signal))
        else:
            composite_signal = 0.0

        return AllPatternsResult(
            patterns=patterns_found,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            total_patterns=len(patterns_found),
            composite_signal=composite_signal
        )

    def recognize_single_pattern(
        self,
        df: pd.DataFrame,
        pattern_name: str
    ) -> np.ndarray:
        """
        识别单个形态

        Args:
            df: 股票数据DataFrame
            pattern_name: 形态名称 (如 'CDLHAMMER')

        Returns:
            形态信号数组
        """
        if not TALIB_AVAILABLE:
            return np.zeros(len(df))

        if pattern_name not in self.patterns_config:
            raise ValueError(f"未知的形态: {pattern_name}")

        open_price = df['open'].values
        high_price = df['high'].values
        low_price = df['low'].values
        close_price = df['close'].values

        func = getattr(talib, pattern_name)
        result = func(open_price, high_price, low_price, close_price)

        return result

    def get_pattern_description(self, pattern_name: str) -> Dict:
        """
        获取形态描述

        Args:
            pattern_name: 形态名称

        Returns:
            形态描述字典
        """
        if pattern_name in self.patterns_config:
            pattern_info = self.patterns_config[pattern_name]
            _, name_cn, pattern_type, description = pattern_info[:4]
            return {
                'name': pattern_name,
                'name_cn': name_cn,
                'type': pattern_type,
                'description': description
            }
        return {}

    def list_all_patterns(self) -> List[Dict]:
        """
        列出所有支持的形态

        Returns:
            形态列表
        """
        return [
            {
                'name': p[0],
                'name_cn': p[1],
                'type': p[2],
                'description': p[3]
            }
            for p in TALIB_PATTERNS
        ]

    def _get_strength(self, signal_abs: int) -> str:
        """根据信号绝对值确定强度"""
        if signal_abs >= 200:
            return '极强'
        elif signal_abs >= 100:
            return '强'
        elif signal_abs >= 50:
            return '中'
        else:
            return '弱'

    def _empty_result(self, reason: str = "无数据") -> AllPatternsResult:
        """返回空结果"""
        return AllPatternsResult(
            patterns=[],
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            total_patterns=0,
            composite_signal=0.0
        )


def recognize_talib_patterns(
    df: pd.DataFrame,
    lookback: int = 5
) -> Dict:
    """
    便捷函数：识别TA-Lib K线形态

    Args:
        df: 股票数据DataFrame
        lookback: 回顾天数

    Returns:
        Dict: 形态识别结果
    """
    recognizer = TalibPatternRecognizer()
    result = recognizer.recognize_all(df, lookback)

    return {
        'patterns': [
            {
                'name': p.name,
                'name_cn': p.name_cn,
                'signal': p.signal,
                'strength': p.strength,
                'type': p.type,
                'description': p.description,
                'date': p.date
            }
            for p in result.patterns
        ],
        'bullish_count': result.bullish_count,
        'bearish_count': result.bearish_count,
        'neutral_count': result.neutral_count,
        'total_patterns': result.total_patterns,
        'composite_signal': result.composite_signal
    }


def format_patterns_report(result: AllPatternsResult) -> str:
    """
    格式化形态识别报告（改进版：区分最新形态和历史形态）

    Args:
        result: 形态识别结果

    Returns:
        Markdown格式报告
    """
    lines = []
    lines.append("\n### K线形态分析 (TA-Lib 61种形态)")
    lines.append("")

    if result.total_patterns == 0:
        lines.append("最近5个交易日未识别到明显K线形态。")
        return '\n'.join(lines)

    # 按类型分组
    bullish = [p for p in result.patterns if p.type == 'bullish']
    bearish = [p for p in result.patterns if p.type == 'bearish']
    neutral = [p for p in result.patterns if p.type == 'neutral']

    # 区分最新形态和历史形态
    all_dates = sorted(set(p.date for p in result.patterns if p.date), reverse=True)
    latest_date = all_dates[0] if all_dates else None

    latest_patterns = [p for p in result.patterns if p.date == latest_date] if latest_date else []
    historical_patterns = [p for p in result.patterns if p.date != latest_date] if latest_date else result.patterns

    # 统计信息（改进：明确说明统计范围）
    lines.append("**形态统计（最近5日扫描）：**")
    lines.append(f"- 看涨形态：{result.bullish_count}个（方向明确）")
    lines.append(f"- 看跌形态：{result.bearish_count}个（方向明确）")
    lines.append(f"- 中性形态：{result.neutral_count}个（如十字星、纺锤等）")
    lines.append(f"- 综合信号：{result.composite_signal:.1f}（仅统计方向明确形态）")
    lines.append("")

    # 最新形态（如果有）
    if latest_patterns:
        lines.append(f"**🔥 最新形态（{latest_date}）：**")
        lines.append("")

        latest_bullish = [p for p in latest_patterns if p.type == 'bullish']
        latest_bearish = [p for p in latest_patterns if p.type == 'bearish']
        latest_neutral = [p for p in latest_patterns if p.type == 'neutral']

        if latest_bullish:
            lines.append("📈 看涨：")
            for p in latest_bullish:
                lines.append(f"  - **{p.name_cn}**（{p.strength}）")
        if latest_bearish:
            lines.append("📉 看跌：")
            for p in latest_bearish:
                lines.append(f"  - **{p.name_cn}**（{p.strength}）")
        if latest_neutral:
            lines.append("➖ 中性：")
            for p in latest_neutral[:3]:
                lines.append(f"  - **{p.name_cn}**（{p.strength}）")
        lines.append("")

    # 历史形态（带日期）
    if historical_patterns:
        lines.append("**📅 历史命中形态：**")
        lines.append("")

        hist_bullish = [p for p in historical_patterns if p.type == 'bullish']
        hist_bearish = [p for p in historical_patterns if p.type == 'bearish']

        if hist_bullish:
            lines.append("📈 看涨：")
            for p in hist_bullish[:5]:
                date_str = p.date if p.date else "未知"
                lines.append(f"  - **{p.name_cn}**（{date_str}）")
        if hist_bearish:
            lines.append("📉 看跌：")
            for p in hist_bearish[:5]:
                date_str = p.date if p.date else "未知"
                lines.append(f"  - **{p.name_cn}**（{date_str}）")
        lines.append("")

    # 综合评估
    assessment = get_pattern_assessment(result)
    lines.append(f"**综合评估：** {assessment}")

    # 说明
    if bullish and bearish:
        lines.append("")
        lines.append("**📌 说明：** 最新形态对当日交易参考价值更高，历史形态反映近期市场状态。")
        lines.append("若多空形态在不同日期出现，说明近期市场反复震荡。")

    lines.append("")

    return '\n'.join(lines)


def get_pattern_assessment(result: AllPatternsResult) -> str:
    """
    获取形态综合评估

    Args:
        result: 形态识别结果

    Returns:
        评估描述
    """
    if result.total_patterns == 0:
        return "无明显K线形态"

    assessments = []

    # 看涨/看跌比例（改进：更准确的描述）
    total = result.bullish_count + result.bearish_count
    if total > 0:
        bullish_ratio = result.bullish_count / total
        # 关键改进：区分"多空均衡"和"略多"
        if bullish_ratio >= 0.8:
            assessments.append("看涨形态占优，多头信号明显")
        elif bullish_ratio >= 0.6:
            assessments.append("看涨形态略多，偏多格局")
        elif bullish_ratio > 0.5:
            # 新增：略微偏多但不够明显
            assessments.append("看涨形态稍多，但优势不明显")
        elif bullish_ratio == 0.5:
            # 新增：完全均衡
            assessments.append("多空形态均衡，方向不明确")
        elif bullish_ratio >= 0.4:
            # 新增：略微偏空但不够明显
            assessments.append("看跌形态稍多，但优势不明显")
        elif bullish_ratio >= 0.2:
            assessments.append("看跌形态略多，偏空格局")
        else:
            assessments.append("看跌形态占优，空头信号明显")

    # 综合信号
    if result.composite_signal >= 30:
        assessments.append("【综合信号偏多】")
    elif result.composite_signal <= -30:
        assessments.append("【综合信号偏空】")
    else:
        assessments.append("【综合信号中性】")

    return "；".join(assessments)


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
        "锤头": """
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
        "上吊线": """
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
""",
        "射击之星": """
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
        "倒锤头": """
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
        "晨星": """
    ┌─────────────────┐
    │  ┌───┐          │
    │  │▓▓▓│ 第一根   │  阴线，下跌趋势
    │  └───┘          │
    │    ───          │  小实体，转折信号
    │       ┌───┐     │
    │       │███│     │  阳线，深入第一根实体
    │       └───┘     │
    └─────────────────┘
    出现在低位 = 强烈看涨反转信号
""",
        "暮星": """
    ┌─────────────────┐
    │       ┌───┐     │
    │       │███│     │  阳线，上涨趋势
    │       └───┘     │
    │    ───          │  小实体，转折信号
    │  ┌───┐          │
    │  │▓▓▓│ 第三根   │  阴线，深入第一根实体
    │  └───┘          │
    └─────────────────┘
    出现在高位 = 强烈看跌反转信号
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
        "三个白兵": """
    ┌─────────────────┐
    │       ┌───┐     │
    │       │███│     │  第三根阳线
    │    ┌───┤███│    │  第二根阳线
    │    │███└───┤    │
    │ ┌──┤███│        │  第一根阳线
    │ │██└───┤        │
    │ └──────┘        │
    └─────────────────┘
    连续三根阳线 = 强势看涨
""",
        "三只乌鸦": """
    ┌─────────────────┐
    │ ┌───┐           │  第一根阴线
    │ │▓▓▓├───┐       │
    │ └───┤▓▓▓│       │  第二根阴线
    │     │▓▓▓├───┐   │
    │     └───┤▓▓▓│   │  第三根阴线
    │         └───┘   │
    └─────────────────┘
    连续三根阴线 = 强势看跌
""",
    }

    return illustrations.get(pattern_name, "")
