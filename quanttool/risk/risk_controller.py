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