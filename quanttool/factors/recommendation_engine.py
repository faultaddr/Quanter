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


# 低位超卖保护仓位配置
PROTECTION_POSITION_SIZES = {
    "strong": "20-30%",    # 强保护触发：极端低位，反弹概率高
    "standard": "10-20%",  # 标准保护触发：低位超卖，轻仓试探
    "weak": "5-10%",       # 弱保护触发：观察为主
}

# 流动性和置信度约束阈值
LIQUIDITY_MIN_AMT_MA20 = 100000   # 20日日均成交额（千元），即1亿元
LOW_CONFIDENCE_THRESHOLD = 0.4    # 低置信度阈值


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

        score = trend.final_score * context.classic_score.position_modifier
        recommendation.final_score = round(score, 1)
        recommendation.score_grade = self._get_score_grade(recommendation.final_score)

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

        score = breakout.final_score * context.classic_score.position_modifier
        recommendation.final_score = round(score, 1)
        recommendation.score_grade = self._get_score_grade(recommendation.final_score)

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

        # 用 position_modifier 修正分数（高位惩罚）
        score = classic.score * classic.position_modifier
        recommendation.final_score = round(score, 1)
        recommendation.score_grade = self._get_score_grade(recommendation.final_score)

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

        # 趋势风险检查（考虑低位超卖保护）
        if classic.trend_score < 40:
            # 检查是否有低位超卖保护
            protection_level = self._get_protection_level(context)

            if protection_level:
                # 有保护：不强制转为观望，添加警告
                recommendation.warnings.append("【趋势偏弱】但低位超卖保护生效，可轻仓观察")
            else:
                # 无保护：正常趋势风险检查
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
        classic = context.classic_score
        trend = context.trend_score
        breakout = context.breakout_score

        # 获取低位超卖保护级别
        protection_level = self._get_protection_level(context)

        # 判断是否有任何评分系统给出正面信号
        has_positive_signal = (
            classic.score * classic.position_modifier >= 50 or
            (trend.passed_hard_filter and trend.final_score >= 60) or
            (breakout.passed_filter and breakout.has_breakout)
        )

        # 低位超卖保护覆盖回避信号（需有至少一个正面信号支撑）
        if protection_level and has_positive_signal:
            if recommendation.action == ActionType.AVOID:
                if protection_level == "strong":
                    recommendation.action = ActionType.LIGHT_POSITION
                    recommendation.position_size = PROTECTION_POSITION_SIZES["strong"]
                    recommendation.reasons.append("【强保护】极度低位超卖+有评分支撑，建议轻仓试探")
                elif protection_level == "standard":
                    recommendation.action = ActionType.WAIT
                    recommendation.position_size = PROTECTION_POSITION_SIZES["weak"]
                    recommendation.reasons.append("【标准保护】低位超卖，但评分不足，建议观察为主")
                else:  # weak
                    recommendation.action = ActionType.WAIT
                    recommendation.position_size = PROTECTION_POSITION_SIZES["weak"]
                    recommendation.reasons.append("【弱保护】偏低位超卖，建议观察")
        elif protection_level and not has_positive_signal:
            # 有保护但无正面信号，不覆盖回避，只降级到观望
            if recommendation.action == ActionType.AVOID:
                recommendation.action = ActionType.WAIT
                recommendation.position_size = PROTECTION_POSITION_SIZES["weak"]
                recommendation.warnings.append("虽有低位超卖保护，但三系统均无正面信号，不宜入场")

        # 设置仓位（如果未被保护逻辑覆盖）
        if recommendation.action not in [ActionType.AVOID, ActionType.SELL]:
            if not protection_level:  # 无保护时使用默认仓位
                recommendation.position_size = self.POSITION_SIZES.get(
                    recommendation.action, "0%"
                )

        # 高位追高警告
        if position.short_term_position == 'high':
            if recommendation.action == ActionType.STRONG_BUY:
                recommendation.action = ActionType.LIGHT_POSITION
                recommendation.warnings.append("短期高位，不宜重仓追高")

        # 长短期双低位机会（需有正面信号支撑）
        if (position.long_term_position == 'low' and
            position.short_term_position == 'low'):
            if recommendation.action == ActionType.WAIT:
                if has_positive_signal:
                    recommendation.action = ActionType.LIGHT_POSITION
                    recommendation.reasons.append("长短期双低位+有评分支撑，存在反弹机会")
                elif protection_level:
                    recommendation.reasons.append("长短期双低位+超卖保护，但缺乏评分支撑，继续观察")

        # 流动性和置信度约束：降低仓位
        self._apply_liquidity_and_confidence_adjustment(context, recommendation, has_positive_signal)

    def _apply_liquidity_and_confidence_adjustment(
        self,
        context: AnalysisContext,
        recommendation: FinalRecommendation,
        has_positive_signal: bool
    ):
        """根据流动性和置信度调整仓位"""
        low_liquidity = False
        low_confidence = False

        # 检查流动性：20日日均成交额 < 1亿元
        df_row = context.df_last_row or {}
        amt_ma20 = df_row.get('amt_ma20', 0)
        if not amt_ma20:
            # 从 amount 字段估算（单日成交额，千元单位）
            amt_today = df_row.get('amount', 0) or df_row.get('amt', 0)
            if amt_today and amt_today < LIQUIDITY_MIN_AMT_MA20:
                low_liquidity = True
        elif amt_ma20 < LIQUIDITY_MIN_AMT_MA20:
            low_liquidity = True

        # 检查置信度
        market_confidence = context.market_state.confidence
        if market_confidence < LOW_CONFIDENCE_THRESHOLD:
            low_confidence = True

        # 无正面信号时进一步降低仓位
        no_signal = not has_positive_signal

        if low_liquidity or low_confidence or no_signal:
            reasons = []
            if low_liquidity:
                reasons.append("流动性不足")
            if low_confidence:
                reasons.append(f"置信度低({market_confidence:.0%})")
            if no_signal:
                reasons.append("三系统无正面信号")

            # 仓位上限：低流动性/低置信度/无信号 → 5-10%
            if recommendation.action in [ActionType.STRONG_BUY, ActionType.BUY]:
                recommendation.action = ActionType.LIGHT_POSITION
                recommendation.position_size = "5-10%"
            elif recommendation.action == ActionType.LIGHT_POSITION:
                recommendation.position_size = "5-10%"
            elif recommendation.action == ActionType.WAIT:
                recommendation.position_size = "0%"

            # 多重风险叠加时进一步降级
            risk_count = sum([low_liquidity, low_confidence, no_signal])
            if risk_count >= 2:
                if recommendation.action == ActionType.LIGHT_POSITION:
                    recommendation.action = ActionType.WAIT
                    recommendation.position_size = "0-5%"
                    reasons.append("多重风险叠加")

            recommendation.warnings.append(f"⚠ 仓位约束: {', '.join(reasons)}，建议极轻仓或观望")

        # 清理矛盾警告：最终为观望/回避时删除"可轻仓观察"等与结论矛盾的措辞
        if recommendation.action in [ActionType.WAIT, ActionType.AVOID]:
            recommendation.warnings = [
                w for w in recommendation.warnings
                if "可轻仓" not in w and "轻仓试探" not in w
            ]

        # 去重：如果多条警告都提到"三系统无正面信号"，只保留最详细的一条
        no_signal_warnings = [w for w in recommendation.warnings if "三系统" in w and "无正面信号" in w]
        if len(no_signal_warnings) > 1:
            # 保留最长的一条（通常最详细），删除其余
            longest = max(no_signal_warnings, key=len)
            recommendation.warnings = [
                w for w in recommendation.warnings
                if not ("三系统" in w and "无正面信号" in w) or w == longest
            ]

    def _get_protection_level(self, context: AnalysisContext) -> Optional[str]:
        """
        从经典评分中获取保护级别

        从 classic_score.warnings 中提取保护级别信息

        Returns:
            "strong" | "standard" | "weak" | None
        """
        # 从经典评分的警告中提取
        warnings = context.classic_score.warnings or []

        for warning in warnings:
            if "强保护" in warning:
                return "strong"
            elif "标准保护" in warning:
                return "standard"
            elif "弱保护" in warning:
                return "weak"

        return None

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
        classic = context.classic_score
        trend = context.trend_score
        breakout = context.breakout_score

        is_negative = recommendation.action in [ActionType.AVOID, ActionType.WAIT]

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

        # 位置说明 — 根据结论方向调整措辞
        if position.position == 'low':
            if is_negative:
                pass  # 观望原因统一在下方详细列出
            else:
                reasons.append("入场位置安全")
        elif position.position == 'high':
            reasons.append("入场位置偏高")
            recommendation.warnings.append("位置偏高，注意回调风险")

        # 长短期位置说明
        if position.long_term_position == 'low' and position.short_term_position == 'low':
            if not is_negative:
                reasons.append("长短期双低位")
        elif position.long_term_position == 'high' and position.short_term_position == 'high':
            reasons.append("长短期双高位")
            recommendation.warnings.append("双高位风险，建议谨慎")

        # 观望/回避时补充具体原因（含位置因素）
        if is_negative:
            detail_reasons = []
            if classic.trend_score < 40:
                detail_reasons.append("趋势极弱")
            if not trend.passed_hard_filter:
                detail_reasons.append("趋势系统未通过过滤")
            if not breakout.passed_filter:
                detail_reasons.append("突破系统未通过过滤")
            if classic.score * classic.position_modifier < 50:
                detail_reasons.append("经典评分不足")
            if market.combined_regime == MarketState.BEAR:
                detail_reasons.append("熊市环境")

            # ADX 极端趋势警告
            adx = context.df_last_row.get('dmi_adx', 0) if context.df_last_row else 0
            mdi = context.df_last_row.get('dmi_mdi', 0) if context.df_last_row else 0
            pdi = context.df_last_row.get('dmi_pdi', 0) if context.df_last_row else 0
            if adx > 60 and mdi > pdi:
                recommendation.warnings.append(f"⚠ 极强下降趋势: ADX={adx:.1f}, MDI>PDI，逆势风险极高")
            elif adx > 40 and mdi > pdi:
                detail_reasons.append(f"下降趋势较强(ADX={adx:.0f})")
            if detail_reasons:
                prefix = ""
                if position.long_term_position == 'low' and position.short_term_position == 'low':
                    prefix = "双低位但"
                elif position.position == 'low':
                    prefix = "位置虽低但"
                reasons.append(f"观望原因: {prefix}{', '.join(detail_reasons)}")

        # 补充正面策略信号（即使观望也应提示）
        if is_negative:
            strategy_signals = context.strategy_signals or {}
            buy_signals = []
            buy_keywords = ['BUY', 'STRONG_BUY', 'WEAK_BUY']
            for name, sig in strategy_signals.items():
                if isinstance(sig, dict):
                    current = str(sig.get('current_signal', ''))
                    if any(kw in current for kw in buy_keywords):
                        buy_signals.append(f"{name.upper()}")
            if buy_signals:
                recommendation.warnings.append(f"潜在正面信号: {', '.join(buy_signals)}，可关注后续确认")

        # 基本面因素
        fd = context.fundamental_data
        if fd and fd.data_source:
            if fd.pe_ttm and fd.pe_ttm > 50:
                recommendation.warnings.append(f"估值偏高: PE(TTM)={fd.pe_ttm:.1f}x")
            if fd.revenue_yoy is not None and fd.revenue_yoy < -10:
                recommendation.warnings.append(f"营收增长停滞: 同比{fd.revenue_yoy:.1f}%")
            if fd.debt_ratio and fd.debt_ratio > 70:
                recommendation.warnings.append(f"高负债风险: 负债率{fd.debt_ratio:.1f}%")
            if fd.annual_revenue and fd.annual_revenue < 5:
                recommendation.warnings.append(f"小盘股: 年营收仅{fd.annual_revenue:.1f}亿，流动性风险")

        # 根据市场状态置信度设置推荐置信度
        market_conf = context.market_state.confidence
        if market_conf >= 0.7:
            recommendation.confidence = "高"
        elif market_conf >= 0.5:
            recommendation.confidence = "中"
        else:
            recommendation.confidence = "低"

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