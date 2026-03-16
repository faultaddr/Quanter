"""
筛选层模块

独立于评分系统的筛选逻辑，用于在因子评分之后进行二次筛选：
- K线形态筛选（位置敏感）
- 乖离率过滤
- 多维度交叉验证

核心设计：
1. 评分系统只负责纯因子打分
2. 筛选层负责信号确认/过滤/警示
3. 两者解耦，便于独立调整
"""
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .talib_patterns import recognize_talib_patterns


class ScreenResult(Enum):
    """筛选结果枚举"""
    PASS = "pass"           # 通过筛选，可以买入
    FILTER = "filter"       # 过滤掉，不建议买入
    WARNING = "warning"     # 警示，需谨慎考虑


@dataclass
class ScreeningOutcome:
    """筛选结果"""
    result: ScreenResult                    # 筛选结果
    score_modifier: float = 1.0             # 风险修正系数（用于调整评分）
    reasons: List[str] = field(default_factory=list)  # 筛选原因
    details: Dict[str, Any] = field(default_factory=dict)  # 详细信息


class CandlestickPatternScreener:
    """
    K线形态筛选器

    核心逻辑：位置决定形态意义
    - 低位 + 看涨形态 = PASS（底部信号确认）
    - 高位 + 看涨形态 = WARNING（诱多警示）
    - 高位 + 看跌形态 = FILTER（顶部信号，建议回避）
    - 低位 + 看跌形态 = WARNING（可能洗盘）
    """

    # 强看涨形态（TA-Lib形态名称）
    STRONG_BULLISH = ['晨星', '十字晨星', '看涨吞没', '三个白兵', '弃婴', '南方三星']
    # 中等看涨形态
    MEDIUM_BULLISH = ['锤头', '倒锤头', '刺透形态', '光头光脚阳', '蜻蜓十字']
    # 强看跌形态
    STRONG_BEARISH = ['暮星', '十字暮星', '看跌吞没', '三只乌鸦', '墓碑十字']
    # 中等看跌形态
    MEDIUM_BEARISH = ['射击之星', '上吊线', '乌云盖顶', '光头光脚阴']

    def __init__(self):
        """初始化K线形态筛选器"""
        pass

    def screen(
        self,
        df: Any,
        position_ratio: float = 0.5,
        bias20: float = 0.0,
        boll_pctb: float = 0.5
    ) -> ScreeningOutcome:
        """
        执行K线形态筛选

        Args:
            df: 股票数据DataFrame
            position_ratio: 股价相对60日高低点位置 (0-1)
            bias20: MA20乖离率
            boll_pctb: 布林带百分比位置 (0-1)

        Returns:
            ScreeningOutcome: 筛选结果
        """
        # 调用形态识别
        patterns_result = recognize_talib_patterns(df, lookback=5)

        if not patterns_result or not patterns_result.get("patterns"):
            # 无形态识别结果，默认通过
            return ScreeningOutcome(
                result=ScreenResult.PASS,
                score_modifier=1.0,
                reasons=["无显著K线形态"],
                details={"patterns": [], "position_zone": "unknown"}
            )

        # 判断位置区域
        position_zone, is_low, is_high = self._determine_position(
            position_ratio, bias20, boll_pctb
        )

        # 获取形态列表
        patterns = patterns_result.get("patterns", [])

        # 分析形态与位置组合
        result = self._analyze_patterns_with_position(
            patterns, position_zone, is_low, is_high
        )

        return result

    def _determine_position(
        self,
        position_ratio: float,
        bias20: float,
        boll_pctb: float
    ) -> tuple:
        """
        判断股价位置

        Returns:
            tuple: (位置区域名称, 是否低位, 是否高位)
        """
        is_low = position_ratio < 0.35 or bias20 < -0.05 or boll_pctb < 0.2
        is_high = position_ratio > 0.70 or bias20 > 0.05 or boll_pctb > 0.8

        if is_low:
            return "low_position", True, False
        elif is_high:
            return "high_position", False, True
        else:
            return "mid_position", False, False

    def _analyze_patterns_with_position(
        self,
        patterns: List[Dict],
        position_zone: str,
        is_low: bool,
        is_high: bool
    ) -> ScreeningOutcome:
        """
        分析形态与位置的组合效果

        Returns:
            ScreeningOutcome: 筛选结果
        """
        reasons = []
        score_modifier = 1.0
        has_strong_bullish = False
        has_strong_bearish = False
        has_medium_bullish = False
        has_medium_bearish = False

        pattern_details = []

        for p in patterns:
            name = p.get("name", "")
            p_type = p.get("type", "neutral")
            strength = p.get("strength", "中")

            pattern_details.append({
                "name": name,
                "type": p_type,
                "strength": strength
            })

            if name in self.STRONG_BULLISH:
                has_strong_bullish = True
            elif name in self.MEDIUM_BULLISH:
                has_medium_bullish = True
            elif name in self.STRONG_BEARISH:
                has_strong_bearish = True
            elif name in self.MEDIUM_BEARISH:
                has_medium_bearish = True

        # 根据位置和形态组合决定筛选结果
        # 规则表：
        # | 位置 | 形态 | 结果 | 说明 |
        # |------|------|------|------|
        # | 低位 | 强看涨 | PASS | 底部信号确认 |
        # | 高位 | 强看涨 | WARNING | 诱多警示 |
        # | 高位 | 强看跌 | FILTER | 顶部信号，建议回避 |
        # | 低位 | 看跌 | WARNING | 可能洗盘 |

        if is_low:
            # 低位区域
            if has_strong_bullish:
                reasons.append("【强力底部信号】低位出现强看涨形态，底部反转概率高")
                score_modifier = 1.1  # 略微加分
                return ScreeningOutcome(
                    result=ScreenResult.PASS,
                    score_modifier=score_modifier,
                    reasons=reasons,
                    details={"patterns": pattern_details, "position_zone": position_zone}
                )
            elif has_medium_bullish:
                reasons.append("【底部信号】低位出现看涨形态，关注反弹机会")
                return ScreeningOutcome(
                    result=ScreenResult.PASS,
                    score_modifier=1.0,
                    reasons=reasons,
                    details={"patterns": pattern_details, "position_zone": position_zone}
                )
            elif has_strong_bearish or has_medium_bearish:
                reasons.append("【洗盘信号】低位出现看跌形态，可能是最后恐慌洗盘")
                score_modifier = 0.95  # 略微减分
                return ScreeningOutcome(
                    result=ScreenResult.WARNING,
                    score_modifier=score_modifier,
                    reasons=reasons,
                    details={"patterns": pattern_details, "position_zone": position_zone}
                )

        elif is_high:
            # 高位区域
            if has_strong_bearish:
                reasons.append("【强力顶部信号】高位出现强看跌形态，建议回避")
                return ScreeningOutcome(
                    result=ScreenResult.FILTER,
                    score_modifier=0.8,
                    reasons=reasons,
                    details={"patterns": pattern_details, "position_zone": position_zone}
                )
            elif has_medium_bearish:
                reasons.append("【顶部信号】高位出现看跌形态，警惕回调")
                score_modifier = 0.9
                return ScreeningOutcome(
                    result=ScreenResult.WARNING,
                    score_modifier=score_modifier,
                    reasons=reasons,
                    details={"patterns": pattern_details, "position_zone": position_zone}
                )
            elif has_strong_bullish:
                reasons.append("【警惕】高位出现强看涨形态，可能是诱多/力竭")
                score_modifier = 0.85
                return ScreeningOutcome(
                    result=ScreenResult.WARNING,
                    score_modifier=score_modifier,
                    reasons=reasons,
                    details={"patterns": pattern_details, "position_zone": position_zone}
                )
            elif has_medium_bullish:
                reasons.append("【中性】高位出现看涨形态，需量能确认")
                return ScreeningOutcome(
                    result=ScreenResult.WARNING,
                    score_modifier=0.95,
                    reasons=reasons,
                    details={"patterns": pattern_details, "position_zone": position_zone}
                )

        else:
            # 中位区域
            if has_strong_bullish:
                reasons.append("【延续信号】中位出现强看涨形态，趋势可能延续")
                return ScreeningOutcome(
                    result=ScreenResult.PASS,
                    score_modifier=1.0,
                    reasons=reasons,
                    details={"patterns": pattern_details, "position_zone": position_zone}
                )
            elif has_strong_bearish:
                reasons.append("【调整信号】中位出现强看跌形态，关注调整深度")
                return ScreeningOutcome(
                    result=ScreenResult.WARNING,
                    score_modifier=0.95,
                    reasons=reasons,
                    details={"patterns": pattern_details, "position_zone": position_zone}
                )

        # 默认通过
        if not reasons:
            reasons.append("无明显形态信号")

        return ScreeningOutcome(
            result=ScreenResult.PASS,
            score_modifier=score_modifier,
            reasons=reasons,
            details={"patterns": pattern_details, "position_zone": position_zone}
        )


class StockScreener:
    """
    综合筛选器

    整合多个筛选条件：
    1. K线形态筛选
    2. 乖离率过滤（可选）
    3. 其他筛选条件（可扩展）
    """

    def __init__(self):
        """初始化综合筛选器"""
        self.candlestick_screener = CandlestickPatternScreener()

    def screen(
        self,
        df: Any,
        score_result: Dict[str, Any]
    ) -> ScreeningOutcome:
        """
        执行综合筛选

        Args:
            df: 股票数据DataFrame
            score_result: 评分结果字典

        Returns:
            ScreeningOutcome: 综合筛选结果
        """
        # 提取位置信息
        factors_raw = score_result.get('factors_raw', {})
        position_ratio = factors_raw.get('position_ratio', 0.5)
        bias20 = factors_raw.get('bias20', 0.0)
        boll_pctb = factors_raw.get('boll_pctb', 0.5)

        # 如果评分结果中已有位置信息，优先使用
        position_info = score_result.get('position_info', {})
        if position_info:
            position_ratio = position_info.get('position_ratio', position_ratio)
            bias20 = position_info.get('bias20', bias20)
            boll_pctb = position_info.get('boll_pctb', boll_pctb)

        # K线形态筛选
        cs_outcome = self.candlestick_screener.screen(
            df, position_ratio, bias20, boll_pctb
        )

        # 乖离率过滤（可选，超过+8%时过滤）
        if bias20 > 0.08:
            return ScreeningOutcome(
                result=ScreenResult.FILTER,
                score_modifier=0.7,
                reasons=["乖离率过高(>+8%)，短期风险大"],
                details={"bias20": bias20}
            )

        # 返回K线形态筛选结果
        return cs_outcome

    def get_screening_summary(self, outcome: ScreeningOutcome) -> str:
        """
        生成筛选结果摘要

        Args:
            outcome: 筛选结果

        Returns:
            str: 摘要文本
        """
        result_map = {
            ScreenResult.PASS: "通过",
            ScreenResult.FILTER: "过滤",
            ScreenResult.WARNING: "警示"
        }

        lines = [
            f"**筛选结果**: {result_map.get(outcome.result, '未知')}",
            f"**评分修正系数**: {outcome.score_modifier:.2f}",
            f"**筛选原因**: {'; '.join(outcome.reasons)}"
        ]

        return "\n".join(lines)