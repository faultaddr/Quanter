"""
风险控制模块

实现增强的风险控制与仓位管理：
- 动态止损计算（ATR止损/支撑位止损/历史MAE止损）
- 风险预算仓位管理
- 回撤预警系统
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class StopLossType(str, Enum):
    """止损类型"""
    ATR = "atr"               # ATR止损
    SUPPORT = "support"       # 支撑位止损
    MAE = "mae"               # 历史MAE止损
    PERCENTAGE = "percentage" # 固定比例止损


class DrawdownLevel(str, Enum):
    """回撤预警级别"""
    LEVEL_1 = "level_1"  # 5%回撤
    LEVEL_2 = "level_2"  # 10%回撤
    LEVEL_3 = "level_3"  # 15%回撤
    LEVEL_4 = "level_4"  # 20%回撤


@dataclass
class StopLossResult:
    """止损计算结果"""
    stop_price: float         # 止损价格
    stop_type: StopLossType   # 止损类型
    risk_percent: float       # 风险比例
    distance_percent: float   # 止损距离比例
    confidence: float         # 置信度 (0-1)


@dataclass
class PositionSizeResult:
    """仓位计算结果"""
    shares: float            # 股数
    position_value: float    # 仓位金额
    risk_amount: float       # 风险金额
    risk_percent: float      # 风险比例


@dataclass
class DrawdownAlert:
    """回撤预警"""
    level: DrawdownLevel
    current_drawdown: float
    threshold: float
    message: str
    action_suggested: str
    timestamp: datetime


class RiskController:
    """
    风险控制器

    实现增强的风险控制与仓位管理
    """

    # 回撤预警阈值
    DRAWDOWN_THRESHOLDS = {
        DrawdownLevel.LEVEL_1: 0.05,
        DrawdownLevel.LEVEL_2: 0.10,
        DrawdownLevel.LEVEL_3: 0.15,
        DrawdownLevel.LEVEL_4: 0.20,
    }

    # 回撤预警建议行动
    DRAWDOWN_ACTIONS = {
        DrawdownLevel.LEVEL_1: "注意观察，保持警惕",
        DrawdownLevel.LEVEL_2: "考虑减少新开仓，审视持仓",
        DrawdownLevel.LEVEL_3: "建议减仓，收紧止损",
        DrawdownLevel.LEVEL_4: "强烈建议大幅减仓或清仓",
    }

    def __init__(
        self,
        default_risk_per_trade: float = 0.02,
        max_position_size: float = 0.1,
        atr_period: int = 14,
        atr_multiplier: float = 2.0
    ):
        """
        初始化风险控制器

        Args:
            default_risk_per_trade: 默认单笔交易风险比例
            max_position_size: 最大仓位比例
            atr_period: ATR计算周期
            atr_multiplier: ATR止损乘数
        """
        self.default_risk_per_trade = default_risk_per_trade
        self.max_position_size = max_position_size
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier

        # 历史统计
        self.trade_history: List[Dict] = []
        self.historical_mae: List[float] = []

    def calculate_dynamic_stop_loss(
        self,
        df: pd.DataFrame,
        entry_price: float,
        signal_strength: float = 1.0,
        historical_mae: Optional[float] = None,
        stop_type: StopLossType = StopLossType.ATR
    ) -> StopLossResult:
        """
        动态止损计算

        Args:
            df: 价格数据DataFrame
            entry_price: 入场价格
            signal_strength: 信号强度 (0-1)
            historical_mae: 历史最大不利偏移
            stop_type: 止损类型

        Returns:
            StopLossResult: 止损计算结果
        """
        if len(df) < self.atr_period:
            # 数据不足，使用默认比例止损
            default_stop = entry_price * 0.95
            return StopLossResult(
                stop_price=default_stop,
                stop_type=StopLossType.PERCENTAGE,
                risk_percent=0.05,
                distance_percent=0.05,
                confidence=0.5
            )

        # 根据止损类型计算
        if stop_type == StopLossType.ATR:
            return self._calculate_atr_stop(df, entry_price, signal_strength)
        elif stop_type == StopLossType.SUPPORT:
            return self._calculate_support_stop(df, entry_price, signal_strength)
        elif stop_type == StopLossType.MAE:
            return self._calculate_mae_stop(entry_price, historical_mae)
        else:
            return self._calculate_atr_stop(df, entry_price, signal_strength)

    def _calculate_atr_stop(
        self,
        df: pd.DataFrame,
        entry_price: float,
        signal_strength: float
    ) -> StopLossResult:
        """ATR止损计算"""
        # 计算ATR
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(
            high[-self.atr_period:] - low[-self.atr_period:],
            np.maximum(
                np.abs(high[-self.atr_period:] - close[-self.atr_period-1:-1]),
                np.abs(low[-self.atr_period:] - close[-self.atr_period-1:-1])
            )
        )
        atr = np.mean(tr)

        # 根据信号强度调整ATR乘数
        adjusted_multiplier = self.atr_multiplier * (1.5 - 0.5 * signal_strength)
        stop_distance = atr * adjusted_multiplier
        stop_price = entry_price - stop_distance

        # 确保止损价格合理
        stop_price = max(stop_price, entry_price * 0.80)  # 最大止损20%

        distance_percent = (entry_price - stop_price) / entry_price

        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.ATR,
            risk_percent=distance_percent,
            distance_percent=distance_percent,
            confidence=min(1.0, signal_strength * 1.2)
        )

    def _calculate_support_stop(
        self,
        df: pd.DataFrame,
        entry_price: float,
        signal_strength: float
    ) -> StopLossResult:
        """支撑位止损计算"""
        lookback = min(60, len(df))

        # 寻找近期低点作为支撑
        recent_lows = df['low'].tail(lookback).values
        support_level = np.min(recent_lows[-20:])  # 近20日最低

        # 使用近期低点作为止损位
        stop_price = support_level * 0.98  # 留2%缓冲

        # 如果支撑位太近，使用ATR止损
        if stop_price > entry_price * 0.97:
            return self._calculate_atr_stop(df, entry_price, signal_strength)

        distance_percent = (entry_price - stop_price) / entry_price

        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.SUPPORT,
            risk_percent=distance_percent,
            distance_percent=distance_percent,
            confidence=0.8
        )

    def _calculate_mae_stop(
        self,
        entry_price: float,
        historical_mae: Optional[float]
    ) -> StopLossResult:
        """历史MAE止损计算"""
        if historical_mae is None or historical_mae <= 0:
            # 无历史数据，使用默认
            historical_mae = 0.08  # 默认8% MAE

        # 留一定余量
        stop_distance = historical_mae * 1.2
        stop_price = entry_price * (1 - stop_distance)

        return StopLossResult(
            stop_price=stop_price,
            stop_type=StopLossType.MAE,
            risk_percent=stop_distance,
            distance_percent=stop_distance,
            confidence=0.7
        )

    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_price: float,
        risk_per_trade: Optional[float] = None
    ) -> PositionSizeResult:
        """
        风险预算仓位管理

        Position = Capital × Risk% / (Entry - Stop)

        Args:
            capital: 可用资金
            entry_price: 入场价格
            stop_price: 止损价格
            risk_per_trade: 单笔风险比例

        Returns:
            PositionSizeResult: 仓位计算结果
        """
        if risk_per_trade is None:
            risk_per_trade = self.default_risk_per_trade

        # 计算每股风险
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            # 止损价格高于入场价，使用默认值
            risk_per_share = entry_price * 0.05

        # 计算风险金额
        risk_amount = capital * risk_per_trade

        # 计算股数
        shares = risk_amount / risk_per_share

        # 限制最大仓位
        max_position_value = capital * self.max_position_size
        position_value = shares * entry_price

        if position_value > max_position_value:
            position_value = max_position_value
            shares = position_value / entry_price
            risk_amount = shares * risk_per_share

        # 计算实际风险比例
        actual_risk_percent = risk_amount / capital

        return PositionSizeResult(
            shares=shares,
            position_value=position_value,
            risk_amount=risk_amount,
            risk_percent=actual_risk_percent
        )

    def check_drawdown_alert(
        self,
        portfolio_value: float,
        peak_value: float
    ) -> Optional[DrawdownAlert]:
        """
        回撤预警检查

        Args:
            portfolio_value: 当前组合价值
            peak_value: 历史最高价值

        Returns:
            Optional[DrawdownAlert]: 回撤预警（如果有）
        """
        if peak_value <= 0:
            return None

        current_drawdown = (peak_value - portfolio_value) / peak_value

        # 确定预警级别
        alert_level = None
        for level in [DrawdownLevel.LEVEL_4, DrawdownLevel.LEVEL_3,
                      DrawdownLevel.LEVEL_2, DrawdownLevel.LEVEL_1]:
            if current_drawdown >= self.DRAWDOWN_THRESHOLDS[level]:
                alert_level = level
                break

        if alert_level is None:
            return None

        return DrawdownAlert(
            level=alert_level,
            current_drawdown=current_drawdown,
            threshold=self.DRAWDOWN_THRESHOLDS[alert_level],
            message=f"当前回撤 {current_drawdown:.2%}，触发 {alert_level.value} 预警",
            action_suggested=self.DRAWDOWN_ACTIONS[alert_level],
            timestamp=datetime.now()
        )

    def record_trade(
        self,
        entry_price: float,
        exit_price: float,
        max_adverse_excursion: float,
        max_favorable_excursion: float
    ):
        """
        记录交易用于历史统计

        Args:
            entry_price: 入场价
            exit_price: 出场价
            max_adverse_excursion: 最大不利偏移（MAE）
            max_favorable_excursion: 最大有利偏移（MFE）
        """
        mae_percent = max_adverse_excursion / entry_price
        self.historical_mae.append(mae_percent)

        self.trade_history.append({
            'entry_price': entry_price,
            'exit_price': exit_price,
            'mae': max_adverse_excursion,
            'mfe': max_favorable_excursion,
            'mae_percent': mae_percent,
            'timestamp': datetime.now()
        })

    def get_mae_statistics(self) -> Dict:
        """获取MAE统计"""
        if not self.historical_mae:
            return {}

        mae_array = np.array(self.historical_mae)
        return {
            'count': len(mae_array),
            'mean': np.mean(mae_array),
            'median': np.median(mae_array),
            'std': np.std(mae_array),
            'percentile_75': np.percentile(mae_array, 75),
            'percentile_90': np.percentile(mae_array, 90),
            'max': np.max(mae_array),
        }

    def get_risk_summary(self) -> Dict:
        """获取风险摘要"""
        return {
            'total_trades': len(self.trade_history),
            'mae_statistics': self.get_mae_statistics(),
            'default_risk_per_trade': self.default_risk_per_trade,
            'max_position_size': self.max_position_size,
        }


def calculate_mfe_mae(
    df: pd.DataFrame,
    entry_idx: int,
    exit_idx: int,
    direction: str = 'long'
) -> Tuple[float, float]:
    """
    计算最大有利偏移（MFE）和最大不利偏移（MAE）

    Args:
        df: 价格数据
        entry_idx: 入场索引
        exit_idx: 出场索引
        direction: 方向 ('long' or 'short')

    Returns:
        Tuple[float, float]: (MFE, MAE)
    """
    if entry_idx >= exit_idx or entry_idx < 0 or exit_idx > len(df):
        return 0.0, 0.0

    trade_data = df.iloc[entry_idx:exit_idx+1]
    entry_price = df.iloc[entry_idx]['close']

    if direction == 'long':
        highest = trade_data['high'].max()
        lowest = trade_data['low'].min()
        mfe = highest - entry_price  # 最大有利
        mae = entry_price - lowest   # 最大不利
    else:
        highest = trade_data['high'].max()
        lowest = trade_data['low'].min()
        mfe = entry_price - lowest   # 做空时最大有利
        mae = highest - entry_price  # 做空时最大不利

    return mfe, mae


# ========== 组合层面风控 ==========

@dataclass
class IndustryExposure:
    """行业暴露"""
    industry: str
    exposure: float          # 暴露比例
    limit: float            # 限制比例
    is_violated: bool        # 是否超限


@dataclass
class StyleExposure:
    """风格因子暴露"""
    style: str               # 风格因子名称
    exposure: float          # 暴露值
    limit: float             # 限制值
    is_violated: bool        # 是否超限


@dataclass
class PortfolioRiskReport:
    """组合风险报告"""
    timestamp: datetime
    total_industry_exposure: float
    industry_violations: List[IndustryExposure]
    total_style_exposure: float
    style_violations: List[StyleExposure]
    blacklist_violations: List[str]
    position_shrink_factor: float
    overall_risk_score: float
    recommendations: List[str]


class PortfolioRiskManager:
    """
    组合层面风险管理器

    实现组合层面的风控：
    - 行业暴露限制
    - 风格因子限制
    - 黑名单机制
    - 动态仓位收缩
    """

    def __init__(
        self,
        industry_limits: Optional[Dict[str, float]] = None,
        style_limits: Optional[Dict[str, float]] = None,
        max_total_industry_exposure: float = 0.30,
        max_single_stock_exposure: float = 0.05,
        blacklist: Optional[List[str]] = None,
    ):
        """
        初始化组合风控管理器

        Args:
            industry_limits: 各行业暴露限制
            style_limits: 各风格因子暴露限制
            max_total_industry_exposure: 最大行业总暴露
            max_single_stock_exposure: 单只股票最大暴露
            blacklist: 黑名单股票列表
        """
        # 行业暴露限制（默认30%）
        self.industry_limits = industry_limits or {}
        self.max_total_industry_exposure = max_total_industry_exposure

        # 风格暴露限制
        self.style_limits = style_limits or {
            "size": 0.5,
            "value": 0.5,
            "momentum": 0.5,
            "quality": 0.5,
            "volatility": 0.5,
            "liquidity": 0.5,
        }

        # 单票限制
        self.max_single_stock_exposure = max_single_stock_exposure

        # 黑名单
        self.blacklist = set(blacklist or [])

        # 风险历史
        self.risk_history: List[PortfolioRiskReport] = []

    def add_to_blacklist(self, symbol: str, reason: str = ""):
        """添加黑名单"""
        self.blacklist.add(symbol)

    def remove_from_blacklist(self, symbol: str):
        """移除黑名单"""
        self.blacklist.discard(symbol)

    def is_blacklisted(self, symbol: str) -> bool:
        """检查是否在黑名单"""
        return symbol in self.blacklist

    def check_industry_exposure(
        self,
        positions: Dict[str, Dict],
        industry_map: Dict[str, str],
        portfolio_value: float,
    ) -> Tuple[List[IndustryExposure], float]:
        """
        检查行业暴露

        Args:
            positions: 持仓dict {symbol: {industry, value}}
            industry_map: 股票行业映射 {symbol: industry}
            portfolio_value: 组合价值

        Returns:
            (行业暴露列表, 总暴露)
        """
        industry_values: Dict[str, float] = {}

        # 计算各行业市值
        for symbol, pos in positions.items():
            industry = industry_map.get(symbol, "unknown")
            value = pos.get("value", 0)
            industry_values[industry] = industry_values.get(industry, 0) + value

        # 计算暴露比例
        violations = []
        total_exposure = 0.0

        for industry, value in industry_values.items():
            exposure = value / portfolio_value if portfolio_value > 0 else 0
            limit = self.industry_limits.get(industry, self.max_total_industry_exposure)

            is_violated = exposure > limit
            if is_violated:
                violations.append(IndustryExposure(
                    industry=industry,
                    exposure=exposure,
                    limit=limit,
                    is_violated=True,
                ))

            total_exposure += exposure

        return violations, total_exposure

    def check_style_exposure(
        self,
        positions: Dict[str, Dict],
        style_factors: Dict[str, Dict[str, float]],
        portfolio_value: float,
    ) -> Tuple[List[StyleExposure], float]:
        """
        检查风格因子暴露

        Args:
            positions: 持仓dict
            style_factors: 风格因子暴露 {symbol: {factor: value}}
            portfolio_value: 组合价值

        Returns:
            (风格暴露列表, 总暴露)
        """
        if not style_factors:
            return [], 0.0

        # 计算各风格因子的加权暴露
        style_exposures: Dict[str, float] = {}

        for symbol, pos in positions.items():
            if symbol not in style_factors:
                continue

            weight = pos.get("value", 0) / portfolio_value if portfolio_value > 0 else 0

            for factor, value in style_factors[symbol].items():
                style_exposures[factor] = style_exposures.get(factor, 0) + weight * value

        violations = []
        total_exposure = 0.0

        for style, exposure in style_exposures.items():
            abs_exposure = abs(exposure)
            limit = self.style_limits.get(style, 0.5)

            if abs_exposure > limit:
                violations.append(StyleExposure(
                    style=style,
                    exposure=exposure,
                    limit=limit,
                    is_violated=True,
                ))

            total_exposure += abs_exposure

        return violations, total_exposure

    def check_blacklist_violations(
        self,
        positions: Dict[str, Dict],
    ) -> List[str]:
        """
        检查黑名单违规

        Args:
            positions: 持仓dict

        Returns:
            违规股票列表
        """
        return [symbol for symbol in positions.keys() if self.is_blacklisted(symbol)]

    def calculate_position_shrink_factor(
        self,
        portfolio_value: float,
        peak_value: float,
        current_drawdown: float,
    ) -> float:
        """
        计算动态仓位收缩系数

        根据当前回撤动态调整仓位

        Args:
            portfolio_value: 当前组合价值
            peak_value: 历史最高价值
            current_drawdown: 当前回撤

        Returns:
            仓位收缩系数 (0-1)
        """
        if peak_value <= 0:
            return 1.0

        # 基础收缩系数
        base_shrink = 1.0

        # 根据回撤调整
        if current_drawdown < 0.05:
            shrink = 1.0
        elif current_drawdown < 0.10:
            shrink = 0.9
        elif current_drawdown < 0.15:
            shrink = 0.8
        elif current_drawdown < 0.20:
            shrink = 0.7
        else:
            shrink = max(0.5, 1.0 - current_drawdown * 2)

        return shrink

    def check_risk(
        self,
        positions: Dict[str, Dict],
        industry_map: Dict[str, str],
        style_factors: Optional[Dict[str, Dict[str, float]]] = None,
        portfolio_value: float = 0.0,
        peak_value: float = 0.0,
    ) -> PortfolioRiskReport:
        """
        全面风险检查

        Args:
            positions: 持仓dict
            industry_map: 股票行业映射
            style_factors: 风格因子暴露
            portfolio_value: 组合价值
            peak_value: 历史最高价值

        Returns:
            PortfolioRiskReport: 风险报告
        """
        # 计算当前回撤
        current_drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0

        # 检查各项风险
        industry_violations, total_industry = self.check_industry_exposure(
            positions, industry_map, portfolio_value
        )

        style_violations = []
        total_style = 0.0
        if style_factors:
            style_violations, total_style = self.check_style_exposure(
                positions, style_factors, portfolio_value
            )

        blacklist_violations = self.check_blacklist_violations(positions)

        # 计算仓位收缩系数
        shrink_factor = self.calculate_position_shrink_factor(
            portfolio_value, peak_value, current_drawdown
        )

        # 计算风险评分
        risk_score = self._calculate_risk_score(
            industry_violations,
            style_violations,
            blacklist_violations,
            current_drawdown,
        )

        # 生成建议
        recommendations = self._generate_recommendations(
            industry_violations,
            style_violations,
            blacklist_violations,
            current_drawdown,
            shrink_factor,
        )

        report = PortfolioRiskReport(
            timestamp=datetime.now(),
            total_industry_exposure=total_industry,
            industry_violations=industry_violations,
            total_style_exposure=total_style,
            style_violations=style_violations,
            blacklist_violations=blacklist_violations,
            position_shrink_factor=shrink_factor,
            overall_risk_score=risk_score,
            recommendations=recommendations,
        )

        self.risk_history.append(report)

        return report

    def _calculate_risk_score(
        self,
        industry_violations: List[IndustryExposure],
        style_violations: List[StyleExposure],
        blacklist_violations: List[str],
        drawdown: float,
    ) -> float:
        """计算综合风险评分 (0-100)"""
        score = 100.0

        # 行业违规扣分
        score -= len(industry_violations) * 10

        # 风格违规扣分
        score -= len(style_violations) * 8

        # 黑名单违规扣分
        score -= len(blacklist_violations) * 20

        # 回撤扣分
        score -= drawdown * 100

        return max(0, min(100, score))

    def _generate_recommendations(
        self,
        industry_violations: List[IndustryExposure],
        style_violations: List[StyleExposure],
        blacklist_violations: List[str],
        drawdown: float,
        shrink_factor: float,
    ) -> List[str]:
        """生成风险建议"""
        recommendations = []

        # 行业建议
        for violation in industry_violations:
            recommendations.append(
                f"行业 [{violation.industry}] 暴露 {violation.exposure:.1%} 超过限制 {violation.limit:.1%}，建议减仓"
            )

        # 风格建议
        for violation in style_violations:
            recommendations.append(
                f"风格因子 [{violation.style}] 暴露 {violation.exposure:.2f} 超过限制 {violation.limit:.2f}"
            )

        # 黑名单建议
        for symbol in blacklist_violations:
            recommendations.append(f"持仓股票 [{symbol}] 在黑名单中，建议卖出")

        # 回撤建议
        if drawdown > 0.15:
            recommendations.append(f"当前回撤 {drawdown:.1%}，建议减仓收紧风险敞口")
        elif drawdown > 0.10:
            recommendations.append(f"当前回撤 {drawdown:.1%}，建议密切关注")

        # 仓位收缩建议
        if shrink_factor < 1.0:
            recommendations.append(f"建议仓位收缩至 {shrink_factor:.0%}")

        return recommendations

    def get_risk_summary(self) -> Dict:
        """获取风险摘要"""
        if not self.risk_history:
            return {}

        latest = self.risk_history[-1]

        return {
            "timestamp": latest.timestamp.isoformat(),
            "risk_score": latest.overall_risk_score,
            "industry_violations": len(latest.industry_violations),
            "style_violations": len(latest.style_violations),
            "blacklist_violations": len(latest.blacklist_violations),
            "position_shrink": latest.position_shrink_factor,
            "recommendations": latest.recommendations,
        }