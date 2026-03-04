"""
统一推荐引擎 - 单点决策系统

核心理念：
- 所有决策逻辑集中在一个引擎中
- 生成唯一的 FinalRecommendation 对象
- 报告各部分使用同一个 recommendation，消除矛盾

决策流程：
1. 检查硬过滤条件（熔断机制）
2. 检查市场状态确定主评分系统
3. 综合三套评分系统结果
4. 生成统一操作/入场/止损/目标
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

from .analysis_context import (
    AnalysisContext,
    ActionType,
    FinalRecommendation,
    MarketState,
    PositionAssessment,
    ScoringSystemType,
    StopLossConfig,
    StopLossType,
    UnifiedMarketState,
    ClassicScore,
    TrendScore,
    BreakoutScore,
)


class RecommendationEngine:
    """
    统一推荐引擎

    确保所有报告部分使用相同决策结果
    """

    # 决策矩阵阈值
    CLASSIC_BUY_THRESHOLD = 65      # 经典评分买入阈值
    CLASSIC_WAIT_THRESHOLD = 50     # 经典评分观望阈值
    TREND_BUY_THRESHOLD = 75        # 趋势评分买入阈值
    TREND_WAIT_THRESHOLD = 60       # 趋势评分观望阈值
    BREAKOUT_BUY_THRESHOLD = 70     # 突破评分买入阈值
    BREAKOUT_WAIT_THRESHOLD = 60    # 突破评分观望阈值

    # 位置系数阈值
    POSITION_DANGER_THRESHOLD = 0.5     # 危险位置
    POSITION_WARNING_THRESHOLD = 0.7    # 警戒位置
    POSITION_SAFE_THRESHOLD = 0.95      # 安全位置

    # 仓位配置
    POSITION_SIZES = {
        ActionType.STRONG_BUY: "80-100%",
        ActionType.BUY: "50-80%",
        ActionType.LIGHT_POSITION: "20-30%",
        ActionType.WAIT: "0%",
        ActionType.AVOID: "0%",
        ActionType.SELL: "清仓",
    }

    def generate_recommendation(self, context: AnalysisContext) -> FinalRecommendation:
        """
        生成最终推荐

        决策流程：
        1. 检查硬过滤条件
        2. 检查熔断条件
        3. 根据市场状态确定主评分系统
        4. 生成统一操作/入场/止损/目标
        """
        recommendation = FinalRecommendation()

        # 1. 获取市场状态和位置评估
        market_state = context.market_state
        position = context.position_assessment
        classic = context.classic_score
        trend = context.trend_score
        breakout = context.breakout_score

        # 2. 确定主评分系统
        primary_system = self._determine_primary_system(
            market_state, classic, trend, breakout
        )
        recommendation.primary_system = primary_system

        # 3. 检查硬过滤/熔断条件
        fuse_reason = self._check_fuse_conditions(context)
        if fuse_reason:
            recommendation.fuse_reason = fuse_reason
            recommendation.action = ActionType.AVOID
            recommendation.warnings.append(fuse_reason)
            recommendation.position_size = "0%"
            recommendation.final_score = 0
            recommendation.score_grade = "回避"
            return recommendation

        # 4. 检查形态覆盖条件
        pattern_override = self._check_pattern_override(context)
        if pattern_override:
            recommendation.pattern_override = pattern_override

        # 5. 根据主评分系统生成决策
        if primary_system == ScoringSystemType.TREND:
            self._apply_trend_decision(context, recommendation)
        elif primary_system == ScoringSystemType.BREAKOUT:
            self._apply_breakout_decision(context, recommendation)
        else:  # CLASSIC
            self._apply_classic_decision(context, recommendation)

        # 6. 应用位置修正
        self._apply_position_adjustment(context, recommendation)

        # 7. 计算入场区间和止损
        self._calculate_entry_and_stop(context, recommendation)

        # 8. 生成理由和警告
        self._generate_reasons_and_warnings(context, recommendation)

        return recommendation

    def _determine_primary_system(
        self,
        market_state: UnifiedMarketState,
        classic: ClassicScore,
        trend: TrendScore,
        breakout: BreakoutScore
    ) -> ScoringSystemType:
        """
        根据市场状态和评分结果确定主评分系统

        决策逻辑：
        | 市场状态 | 经典评分 | 趋势评分 | 突破评分 | 主系统 |
        |---------|---------|---------|---------|-------|
        | BULL    | >= 65   | >= 75   | -       | Trend |
        | BULL    | >= 65   | < 75    | -       | Classic |
        | BEAR    | -       | -       | 通过    | Breakout |
        | SIDEWAY | >= 50   | -       | 通过    | Classic |
        | Any     | < 40    | < 60    | 失败    | None   |
        """
        combined = market_state.combined_regime

        # 牛市：优先使用趋势系统
        if combined == MarketState.BULL:
            if trend.passed_hard_filter and trend.final_score >= self.TREND_BUY_THRESHOLD:
                return ScoringSystemType.TREND
            elif classic.score >= self.CLASSIC_BUY_THRESHOLD:
                return ScoringSystemType.CLASSIC
            elif trend.passed_hard_filter:
                return ScoringSystemType.TREND
            else:
                return ScoringSystemType.CLASSIC

        # 熊市：使用突破系统寻找低位机会
        elif combined == MarketState.BEAR:
            if breakout.passed_filter and breakout.has_breakout:
                return ScoringSystemType.BREAKOUT
            elif classic.score >= self.CLASSIC_WAIT_THRESHOLD:
                return ScoringSystemType.CLASSIC
            else:
                return ScoringSystemType.BREAKOUT

        # 震荡市：结合经典和突破
        else:  # SIDEWAY or VOLATILE
            if breakout.passed_filter and breakout.has_breakout:
                return ScoringSystemType.BREAKOUT
            elif classic.score >= self.CLASSIC_WAIT_THRESHOLD:
                return ScoringSystemType.CLASSIC
            elif trend.passed_hard_filter and trend.final_score >= self.TREND_WAIT_THRESHOLD:
                return ScoringSystemType.TREND
            else:
                return ScoringSystemType.CLASSIC

    def _check_fuse_conditions(self, context: AnalysisContext) -> Optional[str]:
        """
        检查熔断条件

        返回熔断原因，如果无熔断返回 None
        """
        position = context.position_assessment
        classic = context.classic_score
        trend = context.trend_score

        # 1. 极端超买熔断
        if position.is_extreme_overbought:
            return "熔断-极端超买: WR<10 或 CCI>200 或 RSI>80"

        # 2. 高位下跌趋势熔断
        if position.short_term_position == 'high':
            if classic.trend_score < 40:
                return "熔断-高位下跌: 短期高位且趋势极弱"

        # 3. 评分系统熔断
        execution = classic.execution
        action_guide = execution.get('action_guide', '')
        if '熔断' in action_guide:
            return action_guide

        # 4. 筛选系统熔断
        screening = context.screening_result
        if screening.get('result') == 'filter':
            return f"筛选过滤: {screening.get('reason', '未知原因')}"

        return None

    def _check_pattern_override(self, context: AnalysisContext) -> Optional[str]:
        """
        检查形态覆盖条件

        返回覆盖原因，如果无覆盖返回 None
        """
        position = context.position_assessment
        patterns = context.candlestick_patterns

        if not patterns:
            return None

        # 获取最强形态
        strongest = patterns[0] if patterns else {}
        pattern_type = strongest.get('type', '')
        strength = strongest.get('strength', '弱')
        pattern_name = strongest.get('name', '')

        # 强度映射
        strength_map = {'强': 4, '中': 2, '弱': 1}
        strength_val = strength_map.get(strength, 1)

        # 长短期双低位 + 强底部形态
        if (position.long_term_position == 'low' and
            position.short_term_position == 'low' and
            pattern_type == 'bullish' and
            strength_val >= 3):
            return f"长短期双低位+强底部形态({pattern_name})"

        # 短期高位 + 强顶部形态
        if (position.short_term_position == 'high' and
            pattern_type == 'bearish' and
            strength_val >= 3):
            return f"短期高位+强顶部形态({pattern_name})"

        return None

    def _apply_trend_decision(
        self,
        context: AnalysisContext,
        recommendation: FinalRecommendation
    ):
        """应用趋势评分系统决策"""
        trend = context.trend_score
        position = context.position_assessment

        if not trend.passed_hard_filter:
            recommendation.action = ActionType.AVOID
            recommendation.warnings.append(f"趋势系统未通过过滤: {trend.hard_filter_reason}")
            recommendation.final_score = 0
            recommendation.score_grade = "回避"
            return

        score = trend.final_score
        recommendation.final_score = score
        recommendation.score_grade = self._get_score_grade(score)

        # 根据评分和时机系数决策
        timing = trend.timing_coefficient
        timing_type = trend.timing_type

        if score >= 90 and timing >= 1.0:
            recommendation.action = ActionType.STRONG_BUY
        elif score >= 75 and timing >= 0.9:
            recommendation.action = ActionType.BUY
        elif score >= 60 and timing >= 0.8:
            recommendation.action = ActionType.LIGHT_POSITION
        elif score >= 45:
            recommendation.action = ActionType.WAIT
        else:
            recommendation.action = ActionType.AVOID

        # 时机系数调整
        if timing_type == "追高风险":
            if recommendation.action in [ActionType.STRONG_BUY, ActionType.BUY]:
                recommendation.action = ActionType.WAIT
                recommendation.warnings.append("追高风险: 时机系数低，建议观望")
        elif timing_type == "短期过热":
            if recommendation.action == ActionType.STRONG_BUY:
                recommendation.action = ActionType.LIGHT_POSITION
                recommendation.warnings.append("短期过热: 建议轻仓")

        # 位置修正
        if position.position_modifier < self.POSITION_DANGER_THRESHOLD:
            recommendation.action = ActionType.AVOID
            recommendation.warnings.append("位置危险: 不宜追高")

    def _apply_breakout_decision(
        self,
        context: AnalysisContext,
        recommendation: FinalRecommendation
    ):
        """应用突破评分系统决策"""
        breakout = context.breakout_score
        position = context.position_assessment

        if not breakout.passed_filter:
            recommendation.action = ActionType.AVOID
            recommendation.warnings.append(f"突破系统未通过过滤: {breakout.filter_reason}")
            recommendation.final_score = 0
            recommendation.score_grade = "回避"
            return

        score = breakout.final_score
        recommendation.final_score = score
        recommendation.score_grade = self._get_score_grade(score)

        # 检查形态完整性
        if breakout.has_breakout:
            # 完整突破形态
            if score >= 80:
                recommendation.action = ActionType.BUY
            elif score >= 70:
                recommendation.action = ActionType.LIGHT_POSITION
            elif score >= 60:
                recommendation.action = ActionType.LIGHT_POSITION
                recommendation.warnings.append("形态尚可，建议轻仓观察")
            else:
                recommendation.action = ActionType.WAIT
                recommendation.warnings.append("因子质量较差，建议观望")
        elif breakout.is_consolidating:
            # 盘整中，尚未突破
            recommendation.action = ActionType.WAIT
            recommendation.warnings.append("盘整蓄势中，等待突破信号")
        else:
            recommendation.action = ActionType.AVOID

        # 熊市环境下更保守
        if context.market_state.combined_regime == MarketState.BEAR:
            if recommendation.action == ActionType.BUY:
                recommendation.action = ActionType.LIGHT_POSITION
                recommendation.warnings.append("熊市环境，建议轻仓试探")

    def _apply_classic_decision(
        self,
        context: AnalysisContext,
        recommendation: FinalRecommendation
    ):
        """应用经典评分系统决策"""
        classic = context.classic_score
        position = context.position_assessment

        score = classic.score
        recommendation.final_score = score
        recommendation.score_grade = classic.score_grade

        # 形态覆盖优先
        if recommendation.pattern_override:
            if "底部形态" in recommendation.pattern_override:
                recommendation.action = ActionType.LIGHT_POSITION
            elif "顶部形态" in recommendation.pattern_override:
                recommendation.action = ActionType.SELL
            return

        # 根据评分决策
        if score >= 75:
            recommendation.action = ActionType.BUY
        elif score >= 65:
            recommendation.action = ActionType.LIGHT_POSITION
        elif score >= 50:
            recommendation.action = ActionType.WAIT
        else:
            recommendation.action = ActionType.AVOID

        # 位置修正
        if position.position_modifier < self.POSITION_DANGER_THRESHOLD:
            if recommendation.action in [ActionType.BUY, ActionType.LIGHT_POSITION]:
                recommendation.action = ActionType.WAIT
                recommendation.warnings.append("位置危险，建议观望")
        elif position.position_modifier < self.POSITION_WARNING_THRESHOLD:
            if recommendation.action == ActionType.BUY:
                recommendation.action = ActionType.LIGHT_POSITION
                recommendation.warnings.append("位置偏高，建议轻仓")

        # 趋势风险检查
        if classic.trend_score < 40:
            if position.position_modifier >= self.POSITION_SAFE_THRESHOLD:
                recommendation.warnings.append("【接飞刀风险】位置虽低但趋势极弱，切勿盲目抄底")
            if recommendation.action in [ActionType.BUY, ActionType.LIGHT_POSITION]:
                recommendation.action = ActionType.WAIT

    def _apply_position_adjustment(
        self,
        context: AnalysisContext,
        recommendation: FinalRecommendation
    ):
        """应用位置修正"""
        position = context.position_assessment

        # 设置仓位
        if recommendation.action not in [ActionType.AVOID, ActionType.SELL]:
            recommendation.position_size = self.POSITION_SIZES.get(
                recommendation.action, "0%"
            )

        # 高位追高警告
        if position.short_term_position == 'high':
            if recommendation.action == ActionType.STRONG_BUY:
                recommendation.action = ActionType.LIGHT_POSITION
                recommendation.warnings.append("短期高位，不宜重仓追高")

        # 长短期双低位机会
        if (position.long_term_position == 'low' and
            position.short_term_position == 'low'):
            if recommendation.action == ActionType.WAIT:
                recommendation.action = ActionType.LIGHT_POSITION
                recommendation.reasons.append("长短期双低位，存在反弹机会")

    def _calculate_entry_and_stop(
        self,
        context: AnalysisContext,
        recommendation: FinalRecommendation
    ):
        """计算入场区间和止损"""
        current_price = context.current_price
        stop_config = context.stop_loss_config

        # 入场区间计算
        if recommendation.is_actionable():
            # 入场区间：当前价格附近 ±2%
            recommendation.entry_low = round(current_price * 0.98, 2)
            recommendation.entry_high = round(current_price * 1.02, 2)
        else:
            recommendation.entry_low = 0
            recommendation.entry_high = 0

        # 设置止损
        recommendation.stop_loss = stop_config

    def _generate_reasons_and_warnings(
        self,
        context: AnalysisContext,
        recommendation: FinalRecommendation
    ):
        """生成理由和警告"""
        position = context.position_assessment
        market = context.market_state

        # 理由
        reasons = []

        # 主评分系统说明
        system_names = {
            ScoringSystemType.CLASSIC: "经典评分",
            ScoringSystemType.TREND: "趋势评分",
            ScoringSystemType.BREAKOUT: "突破评分",
        }
        reasons.append(f"主评分系统: {system_names.get(recommendation.primary_system, '未知')}")

        # 评分说明
        reasons.append(f"综合评分: {recommendation.final_score:.1f}分 ({recommendation.score_grade})")

        # 市场状态说明
        regime_names = {
            MarketState.BULL: "牛市",
            MarketState.BEAR: "熊市",
            MarketState.SIDEWAY: "震荡市",
            MarketState.VOLATILE: "剧烈波动",
        }
        reasons.append(f"市场状态: {regime_names.get(market.combined_regime, '未知')}")

        # 位置说明
        if position.position == 'low':
            reasons.append("入场位置安全")
        elif position.position == 'high':
            reasons.append("入场位置偏高")
            recommendation.warnings.append("位置偏高，注意回调风险")

        # 长短期位置说明
        if position.long_term_position == 'low' and position.short_term_position == 'low':
            reasons.append("长短期双低位")
        elif position.long_term_position == 'high' and position.short_term_position == 'high':
            reasons.append("长短期双高位")
            recommendation.warnings.append("双高位风险，建议谨慎")

        recommendation.reasons = reasons

    def _get_score_grade(self, score: float) -> str:
        """获取评分等级"""
        if score >= 90:
            return "优秀"
        elif score >= 75:
            return "良好"
        elif score >= 60:
            return "一般"
        elif score >= 45:
            return "较弱"
        else:
            return "较差"


def create_recommendation_engine() -> RecommendationEngine:
    """创建推荐引擎实例"""
    return RecommendationEngine()