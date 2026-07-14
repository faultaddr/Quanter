"""Markdown stock report generation from AnalysisContext."""

from datetime import datetime
from typing import List

import pandas as pd

from quanttool.factors.analysis_context import (
    ActionType,
    AnalysisContext,
    MarketState,
    ScoringSystemType,
    StopLossType as UnifiedStopLossType,
)
from quanttool.factors.fundamental_rating import FundamentalRating


class StockReportGenerator:
    """Generate Markdown reports from a prepared AnalysisContext."""

    def generate(
        self,
        df: pd.DataFrame,
        context: AnalysisContext,
        symbol: str,
    ) -> str:
        """Generate Markdown report from a prepared analysis context."""
        if df.empty:
            return "No data available for report generation"

        report = []
        rec = context.final_recommendation

        # 基本信息
        report.append(f"# 股票技术分析报告：{symbol}")
        report.append("")
        report.append(f"**分析日期：** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        report.append(f"**主评分系统：** {rec.primary_system.value}")
        report.append("")

        # ========== 第一部分：核心结论 ==========
        report.extend(self._generate_core_conclusion_v2(context))

        # ========== 第二部分：三系统评分对比 ==========
        report.extend(self._generate_three_system_analysis(context))

        # ========== 第三部分：市场状态与风险 ==========
        report.extend(self._generate_market_risk_section(context))

        # ========== 第四部分：交易执行计划 ==========
        report.extend(self._generate_trading_plan_v2(context))

        # 附录：原始技术指标
        report.append("---")
        report.append("")
        report.append("## 附录：原始技术指标")
        report.append("")

        latest = df.iloc[-1]
        report.extend(self._generate_technical_indicators_table(latest, context))

        report.append("")
        report.append("> **免责声明：** 本分析仅供学习参考，不构成投资建议。投资决策应基于全面研究和独立判断。")
        report.append("")

        return "\n".join(report)

    def _generate_core_conclusion_v2(self, context: AnalysisContext) -> List[str]:
        """生成核心结论（使用统一推荐）"""
        report = []
        rec = context.final_recommendation

        report.append("## 第一部分：核心结论")
        report.append("")

        # 操作指令
        report.append(f"### {rec.get_action_emoji()} 操作指令：{rec.get_action_display()}")
        report.append("")

        # 评分展示
        report.append(f"**技术评分：{rec.final_score:.1f}分（{rec.score_grade}）**")
        report.append("")

        # 主评分系统说明
        system_names = {
            ScoringSystemType.CLASSIC: "经典多因子评分",
            ScoringSystemType.TREND: "趋势强度评分",
            ScoringSystemType.BREAKOUT: "低位盘整突破评分",
        }
        report.append(f"**主评分系统：{system_names.get(rec.primary_system, '未知')}**")
        report.append("")

        # 熔断原因（如果有）
        if rec.fuse_reason:
            report.append(f"**⚠️ 熔断原因：{rec.fuse_reason}**")
            report.append("")

        # 形态覆盖原因（如果有）
        if rec.pattern_override:
            report.append(f"**🔍 形态覆盖：{rec.pattern_override}**")
            report.append("")

        # 置信度
        report.append(f"**置信度：{rec.confidence}**")
        report.append("")

        # 关键理由
        report.append("### 💡 关键理由")
        report.append("")
        for reason in rec.reasons:
            report.append(f"- {reason}")

        # 警告
        if rec.warnings:
            report.append("")
            report.append("### ⚠️ 风险警告")
            report.append("")
            for warning in rec.warnings:
                report.append(f"- {warning}")

        report.append("")
        return report

    def _generate_three_system_analysis(self, context: AnalysisContext) -> List[str]:
        """生成三系统评分对比"""
        report = []

        report.append("## 第二部分：三系统评分对比")
        report.append("")

        report.append("| 评分系统 | 最终评分 | 状态 | 说明 |")
        report.append("|----------|----------|------|------|")

        # 经典评分（显示修正后分数）
        classic = context.classic_score
        classic_adjusted = classic.score * classic.position_modifier
        classic_status = "✅ 通过" if classic_adjusted >= 50 else "❌ 未通过"
        if classic.position_modifier < 1.0:
            classic_note = f"原始{classic.score:.1f} × 位置系数{classic.position_modifier:.2f}(越高越安全)"
        else:
            classic_note = f"趋势分{classic.trend_score:.1f}"
        report.append(f"| 经典评分 | {classic_adjusted:.1f}分 | {classic_status} | {classic_note} |")

        # 趋势评分
        trend = context.trend_score
        if trend.passed_hard_filter:
            trend_status = "✅ 通过"
            trend_note = f"时机系数{trend.timing_coefficient:.2f}({trend.timing_type})"
        else:
            trend_status = "❌ 未通过"
            trend_note = trend.hard_filter_reason
        report.append(f"| 趋势评分 | {trend.final_score:.1f}分 | {trend_status} | {trend_note} |")

        # 突破评分
        breakout = context.breakout_score
        if breakout.passed_filter:
            breakout_status = "✅ 通过"
            if breakout.has_breakout:
                breakout_note = f"盘整{breakout.consolidation_days}天后突破"
            elif breakout.is_consolidating:
                breakout_note = "盘整蓄势中"
            else:
                breakout_note = "形态不完整"
        else:
            breakout_status = "❌ 未通过"
            breakout_note = breakout.filter_reason
        report.append(f"| 突破评分 | {breakout.final_score:.1f}分 | {breakout_status} | {breakout_note} |")

        report.append("")

        # 主系统选择说明
        report.append(f"**主系统选择原因：** {context.final_recommendation.primary_system.value}")
        market = context.market_state
        regime_names = {
            MarketState.BULL: "牛市",
            MarketState.BEAR: "熊市",
            MarketState.SIDEWAY: "震荡市",
            MarketState.VOLATILE: "剧烈波动",
        }
        report.append(f"- 市场状态：{regime_names.get(market.combined_regime, '未知')}")
        report.append(f"- 综合信号：{market.combined_signal}")
        report.append("")

        return report

    def _generate_market_risk_section(self, context: AnalysisContext) -> List[str]:
        """生成市场状态与风险部分"""
        report = []
        market = context.market_state
        position = context.position_assessment
        stop_loss = context.stop_loss_config

        report.append("## 第三部分：市场状态与风险控制")
        report.append("")

        # 市场状态
        report.append("### 🌡️ 市场状态")
        report.append("")

        regime_emoji = {
            MarketState.BULL: '📈',
            MarketState.BEAR: '📉',
            MarketState.SIDEWAY: '↔️',
            MarketState.VOLATILE: '⚡'
        }
        regime_cn = {
            MarketState.BULL: '牛市',
            MarketState.BEAR: '熊市',
            MarketState.SIDEWAY: '震荡',
            MarketState.VOLATILE: '剧烈波动'
        }

        report.append(f"- **综合状态**: {regime_cn.get(market.combined_regime, '未知')} {regime_emoji.get(market.combined_regime, '➖')}")
        report.append(f"- **综合信号**: {market.combined_signal}")
        report.append(f"- **置信度**: {market.confidence*100:.0f}%")
        report.append("")

        # 位置评估
        report.append("### 📍 位置评估")
        report.append("")

        position_emoji = {'high': '🔴', 'middle': '🟡', 'low': '🟢'}
        report.append(f"- **价格位置**: {position_emoji.get(position.position, '⚪')} {position.position}")
        report.append(f"- **长期位置**: {position_emoji.get(position.long_term_position, '⚪')} {position.long_term_position}")
        report.append(f"- **短期位置**: {position_emoji.get(position.short_term_position, '⚪')} {position.short_term_position}")
        report.append(f"- **位置修正系数**: {position.position_modifier:.2f} (1.0=安全, 越低风险越大, 用于修正评分)")
        report.append(f"- **原因**: {position.reason}")
        report.append("")

        # 风险控制
        report.append("### 🛡️ 风险控制")
        report.append("")

        stop_type_cn = {
            UnifiedStopLossType.PATTERN: '形态止损',
            UnifiedStopLossType.SUPPORT: '支撑位止损',
            UnifiedStopLossType.ATR: 'ATR止损',
            UnifiedStopLossType.MA: '均线止损',
            UnifiedStopLossType.PERCENTAGE: '固定比例止损',
        }

        report.append(f"- **建议止损位**: ¥{stop_loss.stop_price:.2f} ({stop_type_cn.get(stop_loss.stop_type, '止损')})")
        report.append(f"- **止损幅度**: {stop_loss.distance_percent*100:.1f}%")
        if stop_loss.take_profit_price > 0:
            report.append(f"- **建议止盈位**: ¥{stop_loss.take_profit_price:.2f}")
            report.append(f"- **盈亏比**: {stop_loss.risk_reward_ratio:.1f}:1")
        report.append("")

        # 基本面评估
        fd = context.fundamental_data
        if fd.data_source:  # 有基本面数据时才展示
            report.extend(self._generate_fundamental_section(fd))

        return report

    def _generate_fundamental_section(self, fd) -> List[str]:
        """生成基本面评估报告"""
        report = []

        report.append("### 📊 基本面评估")
        report.append("")

        # 估值指标
        report.append("#### 估值指标")
        report.append("")
        pe_str = f"{fd.pe_ttm:.1f}x" if fd.pe_ttm else "N/A"
        pb_str = f"{fd.pb:.2f}" if fd.pb else "N/A"
        cap_str = f"{fd.total_market_cap:.1f}亿" if fd.total_market_cap else "N/A"
        fcap_str = f"{fd.float_market_cap:.1f}亿" if fd.float_market_cap else "N/A"
        report.append(f"| PE(TTM) | PB | 总市值 | 流通市值 |")
        report.append(f"|---------|-----|--------|---------|")
        report.append(f"| {pe_str} | {pb_str} | {cap_str} | {fcap_str} |")
        report.append("")

        # 盈利能力
        report.append("#### 盈利能力")
        report.append("")
        roe_str = f"{fd.roe:.1f}%" if fd.roe else "N/A"
        gm_str = f"{fd.gross_margin:.1f}%" if fd.gross_margin else "N/A"
        pm_str = f"{fd.profit_margin:.1f}%" if fd.profit_margin else "N/A"
        eps_str = f"{fd.eps:.2f}" if fd.eps else "N/A"
        deps_str = f"{fd.deduct_eps:.2f}" if fd.deduct_eps else "N/A"
        report.append(f"| ROE | 毛利率 | 净利率 | EPS | 扣非EPS |")
        report.append(f"|-----|--------|--------|-----|---------|")
        report.append(f"| {roe_str} | {gm_str} | {pm_str} | {eps_str} | {deps_str} |")
        report.append("")

        # 成长性
        report.append("#### 成长性")
        report.append("")
        rev_str = f"{fd.annual_revenue:.1f}" if fd.annual_revenue else "N/A"
        rev_yoy_str = f"{fd.revenue_yoy:+.1f}%" if fd.revenue_yoy is not None else "N/A"
        prof_str = f"{fd.annual_profit:.1f}" if fd.annual_profit else "N/A"
        prof_yoy_str = f"{fd.profit_yoy:+.1f}%" if fd.profit_yoy is not None else "N/A"
        report.append(f"| 年营收(亿) | 营收同比 | 年净利(亿) | 净利同比 |")
        report.append(f"|-----------|---------|-----------|---------|")
        report.append(f"| {rev_str} | {rev_yoy_str} | {prof_str} | {prof_yoy_str} |")
        report.append("")

        # 近5年财务趋势
        if fd.annual_history:
            report.append("#### 近5年财务趋势")
            report.append("")
            report.append("| 年度 | 营收(亿) | 净利(亿) | EPS | ROE | 扣非EPS |")
            report.append("|------|---------|---------|-----|-----|---------|")
            for h in fd.annual_history:
                yr = h.get('year', '')
                rv = h.get('revenue', 'N/A')
                pf = h.get('profit', 'N/A')
                ep = h.get('eps', 'N/A')
                re = h.get('roe', 'N/A')
                de = h.get('deduct_eps', 'N/A')
                rv_s = f"{rv:.1f}" if isinstance(rv, (int, float)) else rv
                pf_s = f"{pf:.1f}" if isinstance(pf, (int, float)) else pf
                ep_s = f"{ep:.2f}" if isinstance(ep, (int, float)) else ep
                re_s = f"{re:.1f}%" if isinstance(re, (int, float)) else re
                de_s = f"{de:.2f}" if isinstance(de, (int, float)) else de
                report.append(f"| {yr} | {rv_s} | {pf_s} | {ep_s} | {re_s} | {de_s} |")
            report.append("")

        # 基本面评级
        rating = FundamentalRating().rate(fd.to_dict())
        stars = lambda s: "★" * s + "☆" * (5 - s)
        report.append("#### 基本面评级")
        report.append("")
        report.append(f"**综合评分: {rating.total_score:.0f}/100 — {rating.total_label}**")
        report.append("")
        report.append(f"- 盈利能力: {stars(rating.profitability.stars)} {rating.profitability.score:.0f}/25 — {rating.profitability.label}")
        report.append(f"- 成长性: {stars(rating.growth.stars)} {rating.growth.score:.0f}/25 — {rating.growth.label}")
        report.append(f"- 估值: {stars(rating.valuation.stars)} {rating.valuation.score:.0f}/25 — {rating.valuation.label}")
        report.append(f"- 财务安全: {stars(rating.safety.stars)} {rating.safety.score:.0f}/25 — {rating.safety.label}")
        report.append("")

        return report

    def _generate_trading_plan_v2(self, context: AnalysisContext) -> List[str]:
        """生成交易执行计划（使用统一推荐）"""
        report = []
        rec = context.final_recommendation
        stop_loss = context.stop_loss_config

        report.append("## 第四部分：交易执行计划")
        report.append("")

        # 策略类型
        report.append("### 📈 策略类型")
        report.append("")

        if rec.is_actionable():
            if rec.primary_system == ScoringSystemType.TREND:
                strategy_type = "✅ 右侧交易（趋势跟随）"
                strategy_desc = "趋势确立，跟随趋势操作"
            elif rec.primary_system == ScoringSystemType.BREAKOUT:
                strategy_type = "✅ 突破交易（形态驱动）"
                strategy_desc = "低位盘整突破，形态确认"
            else:
                strategy_type = "✅ 信号驱动（逢低布局）"
                strategy_desc = "综合评分良好，当前位置适合布局"
        else:
            if rec.action == ActionType.SELL:
                strategy_type = "🛡️ 防御型（减仓/清仓）"
                strategy_desc = "风险信号明确，建议规避"
            else:
                strategy_type = "🔍 观望型（等待机会）"
                strategy_desc = "信号不明确，等待更好的入场时机"

        report.append(f"- **类型**：{strategy_type}")
        report.append(f"- **说明**：{strategy_desc}")
        report.append("")

        # 具体点位
        report.append("### 📍 具体点位")
        report.append("")

        if rec.is_actionable():
            report.append("| 项目 | 建议数值 | 说明 |")
            report.append("|------|----------|------|")
            report.append(f"| 入场区间 | ¥{rec.entry_low:.2f} ~ ¥{rec.entry_high:.2f} | 当前价格附近 |")
            report.append(f"| 止损位 | ¥{stop_loss.stop_price:.2f} | {stop_loss.stop_type.value} |")
            if stop_loss.take_profit_price > 0:
                report.append(f"| 目标位 | ¥{stop_loss.take_profit_price:.2f} | 盈亏比{stop_loss.risk_reward_ratio:.1f}:1 |")
        else:
            report.append(f"| ⚠️ 操作建议 | **{rec.get_action_display()}** | {rec.fuse_reason or '等待更明确信号'} |")

        report.append("")

        # 仓位建议
        report.append("### 💰 仓位建议")
        report.append("")
        report.append(f"- **建议仓位**：{rec.position_size}")
        report.append("")

        # 风险提示
        if rec.warnings:
            report.append("### ⚠️ 风险提示")
            report.append("")
            for warning in rec.warnings:
                report.append(f"- {warning}")
            report.append("")

        # 策略信号共振
        strategy_signals = context.strategy_signals
        if strategy_signals and 'error' not in strategy_signals:
            report.append("### 📊 策略信号共振")
            report.append("")
            report.append("| 策略 | 当前信号 | 置信度 | 操作建议 |")
            report.append("|------|----------|--------|----------|")
            for key in ['rsi', 'macd', 'ma', 'boll']:
                sig = strategy_signals.get(key, {})
                if sig and 'error' not in sig:
                    name = sig.get('strategy', key)
                    signal = sig.get('current_signal', '-')
                    confidence = sig.get('confidence', '-')
                    action = sig.get('action', '-')
                    report.append(f"| {name} | {signal} | {confidence} | {action} |")
            report.append("")

            # 筛选结果
            screening = context.screening_result
            if screening and screening.get('result') != 'pass':
                result_map = {'pass': '通过', 'filter': '过滤', 'warning': '警示'}
                report.append(f"- **筛选结果**: {result_map.get(screening['result'], screening['result'])}")
                if screening.get('reasons'):
                    report.append(f"- **筛选原因**: {'; '.join(screening['reasons'])}")
                report.append("")

        return report

    def _generate_technical_indicators_table(
        self,
        latest: pd.Series,
        context: AnalysisContext
    ) -> List[str]:
        """生成技术指标表格"""
        report = []

        report.append("| 指标 | 数值 | 状态 |")
        report.append("|------|------|------|")

        close = latest.get('close', 0)

        # RSI
        rsi_val = latest.get('rsi_24', 50)
        if rsi_val > 70:
            rsi_desc = "超买区"
        elif rsi_val > 60:
            rsi_desc = "偏强"
        elif rsi_val < 30:
            rsi_desc = "超卖区"
        elif rsi_val < 40:
            rsi_desc = "偏弱"
        else:
            rsi_desc = "中性"
        report.append(f"| RSI(24) | {rsi_val:.2f} | {rsi_desc} |")

        # MACD
        macd_val = latest.get('macd', 0)
        if abs(macd_val) < 0.02:
            macd_desc = "零轴附近"
        elif macd_val > 0:
            macd_desc = "多头"
        else:
            macd_desc = "空头"
        report.append(f"| MACD | {macd_val:.2f} | {macd_desc} |")
        k = latest.get('kdj_k', 0)
        d = latest.get('kdj_d', 0)
        j = latest.get('kdj_j', 0)
        kdj_desc = "J值偏高" if j > 80 else "J值偏低" if j < 20 else "正常"
        report.append(f"| KDJ | K: {k:.2f} / D: {d:.2f} / J: {j:.2f} | {kdj_desc} |")

        # 均线
        ma20 = latest.get('ma_20', 0)
        ma50 = latest.get('ma_50', 0)
        ma200 = latest.get('ma_200', 0)
        ma200_str = f"¥{ma200:.2f}" if not pd.isna(ma200) else "无数据"
        report.append(f"| 移动平均线 | MA20: ¥{ma20:.2f} / MA50: ¥{ma50:.2f} / MA200: {ma200_str} | 趋势参考 |")
        boll_upper = latest.get('boll_upper', close)
        boll_mid = latest.get('boll_mid', close)
        boll_lower = latest.get('boll_lower', close)
        boll_pctb = (close - boll_lower) / (boll_upper - boll_lower) if boll_upper > boll_lower else 0.5
        boll_desc = "触及上轨" if close >= boll_upper else "触及下轨" if close <= boll_lower else "中轨附近"
        report.append(f"| 布林带 | 上轨: ¥{boll_upper:.2f} / 中轨: ¥{boll_mid:.2f} / 下轨: ¥{boll_lower:.2f} | {boll_desc} |")

        # BIAS 乖离率
        bias_6 = latest.get('bias_6', 0)
        bias_12 = latest.get('bias_12', 0)
        bias_24 = latest.get('bias_24', 0)
        # 分指标评估：任一指标超阈值即标注，避免平均后掩盖
        max_abs_bias = max(abs(bias_6), abs(bias_12), abs(bias_24))
        if max_abs_bias > 8:
            bias_desc = "极端偏离"
        elif abs(bias_6) > 5 or abs(bias_12) > 5 or abs(bias_24) > 5:
            bias_desc = "偏离较大"
        elif abs(bias_6) > 3 or abs(bias_12) > 3 or abs(bias_24) > 3:
            bias_desc = "轻微偏离"
        else:
            bias_desc = "正常区间"
        report.append(f"| BIAS(乖离率) | BIAS6: {bias_6:.2f}% / BIAS12: {bias_12:.2f}% / BIAS24: {bias_24:.2f}% | {bias_desc} |")

        # DMI 动向指标
        pdi = latest.get('dmi_pdi', 0)
        mdi = latest.get('dmi_mdi', 0)
        adx = latest.get('dmi_adx', 0)
        if pdi > mdi and adx > 25:
            dmi_desc = "多头趋势强"
        elif mdi > pdi and adx > 25:
            dmi_desc = "空头趋势强"
        elif adx < 20:
            dmi_desc = "趋势不明"
        else:
            dmi_desc = "多空平衡"
        report.append(f"| DMI(动向指标) | PDI: {pdi:.2f} / MDI: {mdi:.2f} / ADX: {adx:.2f} | {dmi_desc} |")

        # ATR 真实波幅
        atr = latest.get('atr_14', 0)
        atr_pct = (atr / close * 100) if close > 0 else 0
        atr_desc = "高波动" if atr_pct > 3 else "低波动" if atr_pct < 1 else "正常波动"
        report.append(f"| ATR(真实波幅) | {atr:.2f} ({atr_pct:.2f}%) | {atr_desc} |")

        # CCI 顺势指标
        cci = latest.get('cci', 0)
        if cci > 200:
            cci_desc = "极度超买"
        elif cci > 100:
            cci_desc = "超买区"
        elif cci < -200:
            cci_desc = "极度超卖"
        elif cci < -100:
            cci_desc = "超卖区"
        else:
            cci_desc = "正常区间"
        report.append(f"| CCI(顺势指标) | {cci:.2f} | {cci_desc} |")

        # WR 威廉指标（0-100范围：>80超卖，<20超买）
        wr = latest.get('wr', 0)
        wr_6 = latest.get('wr_6', 0)
        if wr > 80:
            wr_desc = "超卖区"
        elif wr < 20:
            wr_desc = "超买区"
        else:
            wr_desc = "正常区间"
        report.append(f"| WR(威廉指标) | WR14: {wr:.2f} / WR6: {wr_6:.2f} | {wr_desc} |")

        # TRIX 三重平滑移动平均
        trix = latest.get('trix', 0)
        trix_ma = latest.get('trix_ma', 0)
        if trix > trix_ma:
            if trix > 0:
                trix_desc = "多头"
            else:
                trix_desc = "弱多(零轴下方)"
        else:
            if trix < 0:
                trix_desc = "空头"
            else:
                trix_desc = "弱空(零轴上方)"
        report.append(f"| TRIX | TRIX: {trix:.4f} / TRMA: {trix_ma:.4f} | {trix_desc} |")

        # OBV 能量潮（累积指标，正值不代表当前流入）
        obv = latest.get('obv', 0)
        obv_ma = latest.get('obv_ma', 0)
        if obv_ma and obv_ma > 0:
            obv_desc = "OBV>均线(偏多)" if obv > obv_ma else "OBV<均线(偏空)"
        else:
            obv_desc = "累积正值" if obv > 0 else "累积负值"
        report.append(f"| OBV(能量潮) | {obv:,.0f} | {obv_desc} |")

        # MFI 资金流量指标
        mfi = latest.get('mfi', 50)
        if mfi > 80:
            mfi_desc = "资金超买"
        elif mfi < 20:
            mfi_desc = "资金超卖"
        else:
            mfi_desc = "正常"
        report.append(f"| MFI(资金流量) | {mfi:.2f} | {mfi_desc} |")

        # VR 容量比率
        vr = latest.get('vr', 100)
        if vr > 350:
            vr_desc = "超买区"
        elif vr < 70:
            vr_desc = "超卖区"
        else:
            vr_desc = "正常区间"
        report.append(f"| VR(容量比率) | {vr:.2f} | {vr_desc} |")

        # PSY 心理线
        psy = latest.get('psy', 50)
        if psy > 75:
            psy_desc = "超买区"
        elif psy < 25:
            psy_desc = "超卖区"
        else:
            psy_desc = "正常区间"
        report.append(f"| PSY(心理线) | {psy:.2f}% | {psy_desc} |")

        # BBI 多空指标
        bbi = latest.get('bbi', close)
        bbi_desc = "多头排列" if close > bbi else "空头排列"
        report.append(f"| BBI(多空指标) | ¥{bbi:.2f} | {bbi_desc} |")

        report.append("")
        return report
