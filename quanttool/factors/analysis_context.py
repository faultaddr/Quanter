"""
统一分析上下文 - 单只股票分析系统的核心数据结构

核心理念：
- 单一数据源：所有分析结果存储在一个上下文对象中
- 单点决策：RecommendationEngine 生成唯一的 FinalRecommendation
- 消除矛盾：报告各部分使用同一个 recommendation 对象

数据流：
[数据获取] → DataFrame
      ↓
[三套评分系统并行计算] → classic_score, trend_score, breakout_score
      ↓
[市场状态检测] → UnifiedMarketState
      ↓
[位置评估] → PositionAssessment
      ↓
[止损计算] → StopLossConfig
      ↓
[推荐引擎] → FinalRecommendation（单点决策）
      ↓
[报告生成] → 所有部分使用同一 recommendation
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np


class MarketState(Enum):
    """市场状态枚举"""
    BULL = "bull"           # 牛市
    BEAR = "bear"           # 熊市
    SIDEWAY = "sideway"     # 震荡市
    VOLATILE = "volatile"   # 剧烈波动


class ActionType(Enum):
    """操作类型枚举"""
    STRONG_BUY = "strong_buy"   # 强烈买入
    BUY = "buy"                 # 买入
    LIGHT_POSITION = "light"    # 轻仓试探
    WAIT = "wait"               # 观望
    AVOID = "avoid"             # 回避
    SELL = "sell"               # 卖出


class StopLossType(Enum):
    """止损类型枚举"""
    PATTERN = "pattern"     # 形态止损（突破系统检测到有效形态）
    SUPPORT = "support"     # 支撑位止损（有明确支撑）
    ATR = "atr"             # ATR止损（默认）
    MA = "ma"               # 均线止损
    PERCENTAGE = "percentage"  # 固定比例止损


class ScoringSystemType(Enum):
    """评分系统类型枚举"""
    CLASSIC = "classic"     # 经典多因子评分
    TREND = "trend"         # 趋势强度评分
    BREAKOUT = "breakout"   # 低位盘整突破评分
    AUTO = "auto"           # 自动选择（根据市场状态）


@dataclass
class UnifiedMarketState:
    """
    统一市场状态

    整合双重市场状态（指数+个股）和自适应阈值检测结果
    """
    # 市场状态
    index_regime: MarketState = MarketState.SIDEWAY  # 指数/大盘状态
    stock_regime: MarketState = MarketState.SIDEWAY  # 个股状态
    combined_regime: MarketState = MarketState.SIDEWAY  # 综合状态

    # 置信度
    confidence: float = 0.5

    # 波动率水平
    volatility_level: str = "normal"  # low, normal, high, extreme

    # 综合信号（从 DualMarketState.combined_signal 映射）
    combined_signal: str = "观望"  # 强买入/关注/轻仓/观望/回避/空仓

    # 自适应阈值
    buy_threshold: float = 50.0
    sell_threshold: float = 25.0

    # 元数据
    index_code: str = ""
    index_name: str = "沪深300"

    def get_primary_system(self) -> ScoringSystemType:
        """
        根据市场状态确定主评分系统

        决策逻辑：
        - BULL + 经典评分高 → Classic
        - BULL + 趋势评分高 → Trend
        - BEAR → Breakout（寻找低位机会）
        - SIDEWAY → Classic + Breakout 结合
        """
        if self.combined_regime == MarketState.BULL:
            return ScoringSystemType.TREND
        elif self.combined_regime == MarketState.BEAR:
            return ScoringSystemType.BREAKOUT
        else:  # SIDEWAY or VOLATILE
            return ScoringSystemType.CLASSIC

    def to_dict(self) -> Dict:
        return {
            'index_regime': self.index_regime.value,
            'stock_regime': self.stock_regime.value,
            'combined_regime': self.combined_regime.value,
            'confidence': self.confidence,
            'volatility_level': self.volatility_level,
            'combined_signal': self.combined_signal,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold,
            'index_code': self.index_code,
            'index_name': self.index_name,
        }


@dataclass
class PositionAssessment:
    """
    统一位置评估

    区分"价格位置风险"和"趋势风险"
    """
    # 价格位置
    position: str = "middle"  # high, middle, low

    # 长期位置（基于250日/半年）
    long_term_position: str = "mid"  # high, mid, low

    # 短期位置（基于60日/月度）
    short_term_position: str = "mid"

    # 技术指标状态
    is_overbought: bool = False
    is_oversold: bool = False
    is_extreme_overbought: bool = False
    is_extreme_oversold: bool = False

    # 位置修正系数（仅反映价格位置风险，不含趋势风险）
    position_modifier: float = 1.0

    # 详细信息
    price_ratio: float = 1.0  # 价格/均价
    boll_pctb: float = 0.5    # 布林带位置
    bias20: float = 0.0       # MA20乖离率

    # 关键价位
    close: float = 0.0
    ma20: float = 0.0
    ma50: float = 0.0
    ma200: float = 0.0

    # 描述
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            'position': self.position,
            'long_term_position': self.long_term_position,
            'short_term_position': self.short_term_position,
            'is_overbought': self.is_overbought,
            'is_oversold': self.is_oversold,
            'is_extreme_overbought': self.is_extreme_overbought,
            'is_extreme_oversold': self.is_extreme_oversold,
            'position_modifier': self.position_modifier,
            'price_ratio': self.price_ratio,
            'boll_pctb': self.boll_pctb,
            'bias20': self.bias20,
            'close': self.close,
            'ma20': self.ma20,
            'ma50': self.ma50,
            'ma200': self.ma200,
            'reason': self.reason,
        }


@dataclass
class StopLossConfig:
    """
    统一止损配置

    优先级：形态止损 > 支撑位止损 > ATR止损 > 均线止损
    """
    # 止损价格
    stop_price: float = 0.0

    # 止损类型
    stop_type: StopLossType = StopLossType.ATR

    # 止损幅度（百分比）
    distance_percent: float = 0.05

    # 置信度
    confidence: float = 0.5

    # 各类型止损价格（供选择）
    pattern_stop: float = 0.0      # 形态止损价
    support_stop: float = 0.0       # 支撑位止损价
    atr_stop: float = 0.0           # ATR止损价
    ma_stop: float = 0.0            # 均线止损价

    # ATR值
    atr_value: float = 0.0

    # 止盈价格
    take_profit_price: float = 0.0

    # 盈亏比
    risk_reward_ratio: float = 2.0

    def to_dict(self) -> Dict:
        return {
            'stop_price': self.stop_price,
            'stop_type': self.stop_type.value,
            'distance_percent': self.distance_percent,
            'confidence': self.confidence,
            'pattern_stop': self.pattern_stop,
            'support_stop': self.support_stop,
            'atr_stop': self.atr_stop,
            'ma_stop': self.ma_stop,
            'atr_value': self.atr_value,
            'take_profit_price': self.take_profit_price,
            'risk_reward_ratio': self.risk_reward_ratio,
        }


@dataclass
class FinalRecommendation:
    """
    最终推荐（单点决策）

    这是整个分析系统的核心输出，报告各部分使用同一个对象
    """
    # 操作建议
    action: ActionType = ActionType.WAIT

    # 主评分系统
    primary_system: ScoringSystemType = ScoringSystemType.CLASSIC

    # 最终评分（综合三套系统）
    final_score: float = 50.0

    # 评分等级
    score_grade: str = "一般"

    # 入场区间
    entry_low: float = 0.0
    entry_high: float = 0.0

    # 止损配置
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)

    # 仓位建议
    position_size: str = "0%"  # 建议仓位百分比

    # 操作理由
    reasons: List[str] = field(default_factory=list)

    # 风险警告
    warnings: List[str] = field(default_factory=list)

    # 置信度
    confidence: str = "中"

    # 熔断原因（如果有）
    fuse_reason: str = ""

    # 形态覆盖原因（如果有）
    pattern_override: str = ""

    def is_actionable(self) -> bool:
        """是否可操作（买入或轻仓）"""
        return self.action in [ActionType.STRONG_BUY, ActionType.BUY, ActionType.LIGHT_POSITION]

    def get_action_display(self) -> str:
        """获取操作显示文本"""
        action_map = {
            ActionType.STRONG_BUY: "强烈买入",
            ActionType.BUY: "买入",
            ActionType.LIGHT_POSITION: "轻仓试探",
            ActionType.WAIT: "观望",
            ActionType.AVOID: "回避",
            ActionType.SELL: "卖出",
        }
        return action_map.get(self.action, "未知")

    def get_action_emoji(self) -> str:
        """获取操作emoji"""
        emoji_map = {
            ActionType.STRONG_BUY: "🚀",
            ActionType.BUY: "🟢",
            ActionType.LIGHT_POSITION: "💰",
            ActionType.WAIT: "➖",
            ActionType.AVOID: "🔴",
            ActionType.SELL: "🔻",
        }
        return emoji_map.get(self.action, "➖")

    def to_dict(self) -> Dict:
        return {
            'action': self.action.value,
            'action_display': self.get_action_display(),
            'action_emoji': self.get_action_emoji(),
            'primary_system': self.primary_system.value,
            'final_score': self.final_score,
            'score_grade': self.score_grade,
            'entry_low': self.entry_low,
            'entry_high': self.entry_high,
            'stop_loss': self.stop_loss.to_dict(),
            'position_size': self.position_size,
            'reasons': self.reasons,
            'warnings': self.warnings,
            'confidence': self.confidence,
            'fuse_reason': self.fuse_reason,
            'pattern_override': self.pattern_override,
            'is_actionable': self.is_actionable(),
        }


@dataclass
class ClassicScore:
    """经典评分系统结果"""
    score: float = 50.0
    trend_score: float = 50.0
    position_modifier: float = 1.0
    score_grade: str = "一般"
    factors_score: Dict = field(default_factory=dict)
    factors_raw: Dict = field(default_factory=dict)
    execution: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            'score': self.score,
            'trend_score': self.trend_score,
            'position_modifier': self.position_modifier,
            'score_grade': self.score_grade,
            'factors_score': self.factors_score,
            'factors_raw': self.factors_raw,
            'execution': self.execution,
            'warnings': self.warnings,
        }


@dataclass
class TrendScore:
    """趋势评分系统结果"""
    final_score: float = 0.0
    trend_total_score: float = 0.0
    timing_coefficient: float = 1.0
    timing_type: str = "标准"
    passed_hard_filter: bool = False
    hard_filter_reason: str = ""
    ma_structure_score: float = 0.0
    price_momentum_score: float = 0.0
    volume_score: float = 0.0
    relative_strength_score: float = 0.0
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'final_score': self.final_score,
            'trend_total_score': self.trend_total_score,
            'timing_coefficient': self.timing_coefficient,
            'timing_type': self.timing_type,
            'passed_hard_filter': self.passed_hard_filter,
            'hard_filter_reason': self.hard_filter_reason,
            'ma_structure_score': self.ma_structure_score,
            'price_momentum_score': self.price_momentum_score,
            'volume_score': self.volume_score,
            'relative_strength_score': self.relative_strength_score,
            'details': self.details,
        }


@dataclass
class BreakoutScore:
    """低位盘整突破评分系统结果"""
    final_score: float = 0.0
    is_low_position: bool = False
    is_consolidating: bool = False
    has_breakout: bool = False
    passed_filter: bool = False
    filter_reason: str = ""
    quality_score: float = 50.0
    growth_score: float = 50.0
    value_score: float = 50.0
    momentum_score: float = 50.0
    flow_score: float = 50.0
    risk_score: float = 50.0
    consolidation_days: int = 0
    price_range: float = 0.0
    volume_ratio: float = 1.0
    breakout_strength: float = 0.0
    stop_loss_price: float = 0.0
    take_profit_price: float = 0.0
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'final_score': self.final_score,
            'is_low_position': self.is_low_position,
            'is_consolidating': self.is_consolidating,
            'has_breakout': self.has_breakout,
            'passed_filter': self.passed_filter,
            'filter_reason': self.filter_reason,
            'quality_score': self.quality_score,
            'growth_score': self.growth_score,
            'value_score': self.value_score,
            'momentum_score': self.momentum_score,
            'flow_score': self.flow_score,
            'risk_score': self.risk_score,
            'consolidation_days': self.consolidation_days,
            'price_range': self.price_range,
            'volume_ratio': self.volume_ratio,
            'breakout_strength': self.breakout_strength,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'details': self.details,
        }


@dataclass
class AnalysisContext:
    """
    统一分析上下文 - 单一数据源

    这是整个分析系统的核心数据结构，包含所有分析结果
    """
    # 基本信息
    symbol: str
    current_price: float
    analysis_date: datetime

    # 三套评分结果
    classic_score: ClassicScore = field(default_factory=ClassicScore)
    trend_score: TrendScore = field(default_factory=TrendScore)
    breakout_score: BreakoutScore = field(default_factory=BreakoutScore)

    # 统一市场状态
    market_state: UnifiedMarketState = field(default_factory=UnifiedMarketState)

    # 统一位置评估
    position_assessment: PositionAssessment = field(default_factory=PositionAssessment)

    # 统一止损配置
    stop_loss_config: StopLossConfig = field(default_factory=StopLossConfig)

    # 最终推荐（单点决策）
    final_recommendation: FinalRecommendation = field(default_factory=FinalRecommendation)

    # 原始数据（DataFrame 转为 dict）
    df_last_row: Dict = field(default_factory=dict)

    # K线形态
    candlestick_patterns: List[Dict] = field(default_factory=list)

    # 筛选结果
    screening_result: Dict = field(default_factory=dict)

    # 额外信息
    extra_info: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'current_price': self.current_price,
            'analysis_date': self.analysis_date.isoformat(),
            'classic_score': self.classic_score.to_dict(),
            'trend_score': self.trend_score.to_dict(),
            'breakout_score': self.breakout_score.to_dict(),
            'market_state': self.market_state.to_dict(),
            'position_assessment': self.position_assessment.to_dict(),
            'stop_loss_config': self.stop_loss_config.to_dict(),
            'final_recommendation': self.final_recommendation.to_dict(),
            'df_last_row': self.df_last_row,
            'candlestick_patterns': self.candlestick_patterns,
            'screening_result': self.screening_result,
            'extra_info': self.extra_info,
        }

    def get_primary_score(self) -> float:
        """获取主评分系统的分数"""
        system = self.final_recommendation.primary_system
        if system == ScoringSystemType.CLASSIC:
            return self.classic_score.score
        elif system == ScoringSystemType.TREND:
            return self.trend_score.final_score
        elif system == ScoringSystemType.BREAKOUT:
            return self.breakout_score.final_score
        else:
            # AUTO: 返回最高分
            return max(
                self.classic_score.score,
                self.trend_score.final_score if self.trend_score.passed_hard_filter else 0,
                self.breakout_score.final_score if self.breakout_score.passed_filter else 0
            )

    def get_all_scores_summary(self) -> Dict[str, float]:
        """获取所有评分系统分数摘要"""
        return {
            'classic': self.classic_score.score,
            'trend': self.trend_score.final_score if self.trend_score.passed_hard_filter else None,
            'breakout': self.breakout_score.final_score if self.breakout_score.passed_filter else None,
            'primary': self.get_primary_score(),
        }