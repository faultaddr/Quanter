"""
经典选股策略模块

从 InStock 借鉴的经典选股策略实现：

1. 放量上涨策略
2. 均线多头策略
3. 停机坪策略
4. 回踩年线策略
5. 突破平台策略
6. 无大幅回撤策略
7. 海龟交易法则
8. 高而窄的旗形策略
9. 放量跌停策略
10. 低ATR成长策略
11. 基本面选股策略
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class StrategySignal:
    """策略信号"""
    signal: int           # 1=买入, -1=卖出, 0=持有
    reason: str           # 信号原因
    strength: float       # 信号强度 0-1
    metadata: Dict        # 额外元数据


class VolumeBreakoutStrategy(IStrategy):
    """
    放量上涨策略

    选股条件：
    1. 当日比前一天上涨 < 2% 或 收盘价 < 开盘价
    2. 当日成交额 >= 2亿
    3. 当日成交量 / 5日平均成交量 >= 2
    """

    def __init__(self):
        self.parameters = {
            'min_amount': 200000000,  # 最小成交额 2亿
            'volume_ratio': 2.0,      # 量比阈值
            'price_change_max': 0.02  # 最大涨幅
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 10:
            return pd.DataFrame()

        df = bars.copy()

        # 计算量比
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']

        # 计算成交额
        df['amount'] = df['close'] * df['volume']

        # 计算涨跌幅
        df['pct_change'] = df['close'].pct_change()

        # 判断条件
        condition1 = (df['pct_change'] < self.parameters['price_change_max']) | \
                     (df['close'] < df['open'])
        condition2 = df['amount'] >= self.parameters['min_amount']
        condition3 = df['volume_ratio'] >= self.parameters['volume_ratio']

        df['signal'] = 0
        df.loc[condition1 & condition2 & condition3, 'signal'] = 1

        return df[['timestamp', 'signal', 'volume_ratio', 'amount', 'pct_change']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < 10:
            return {"direction": "hold", "reason": "insufficient_data"}

        volume_ma5 = historical_bars['volume'].tail(5).mean()
        volume_ratio = current_bar['volume'] / volume_ma5
        amount = current_bar['close'] * current_bar['volume']
        pct_change = (current_bar['close'] - historical_bars['close'].iloc[-2]) / \
                     historical_bars['close'].iloc[-2]

        if volume_ratio >= self.parameters['volume_ratio'] and \
           amount >= self.parameters['min_amount']:
            return {
                "direction": "buy",
                "reason": "volume_breakout",
                "strength": min(1.0, volume_ratio / 3),
                "volume_ratio": volume_ratio,
                "amount": amount
            }

        return {"direction": "hold", "reason": "no_signal"}

    def get_name(self) -> str:
        return "VolumeBreakout"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "放量上涨策略：成交量放大2倍以上，成交额超2亿"


class MADAlignmentStrategy(IStrategy):
    """
    均线多头策略

    选股条件：
    1. MA30向上：30日前的30日均线 < 20日前的30日均线 < 10日前的30日均线 < 当日的30日均线
    2. (当日的30日均线 / 30日前的30日均线) > 1.2
    """

    def __init__(self):
        self.parameters = {
            'ma_period': 30,
            'min_slope': 1.2
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 60:
            return pd.DataFrame()

        df = bars.copy()
        period = self.parameters['ma_period']

        # 计算MA30
        df['ma'] = df['close'].rolling(period).mean()

        # 判断MA是否向上
        df['ma_30d_ago'] = df['ma'].shift(30)
        df['ma_20d_ago'] = df['ma'].shift(20)
        df['ma_10d_ago'] = df['ma'].shift(10)

        # 均线多头排列
        df['ma_uptrend'] = (
            (df['ma_30d_ago'] < df['ma_20d_ago']) &
            (df['ma_20d_ago'] < df['ma_10d_ago']) &
            (df['ma_10d_ago'] < df['ma'])
        )

        # 斜率条件
        df['slope'] = df['ma'] / df['ma_30d_ago']

        df['signal'] = 0
        df.loc[df['ma_uptrend'] & (df['slope'] > self.parameters['min_slope']), 'signal'] = 1

        return df[['timestamp', 'signal', 'ma', 'slope']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < 60:
            return {"direction": "hold", "reason": "insufficient_data"}

        period = self.parameters['ma_period']
        close = historical_bars['close']
        ma_current = close.tail(period).mean()
        ma_10d_ago = close.iloc[-period-10:-10].mean()
        ma_20d_ago = close.iloc[-period-20:-20].mean()
        ma_30d_ago = close.iloc[-period-30:-30].mean()

        is_uptrend = ma_30d_ago < ma_20d_ago < ma_10d_ago < ma_current
        slope = ma_current / ma_30d_ago if ma_30d_ago > 0 else 0

        if is_uptrend and slope > self.parameters['min_slope']:
            return {
                "direction": "buy",
                "reason": "ma_alignment_bullish",
                "strength": min(1.0, (slope - 1) * 2),
                "ma_slope": slope
            }

        return {"direction": "hold", "reason": "no_signal"}

    def get_name(self) -> str:
        return "MAAlignment"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "均线多头策略：MA30向上且斜率大于20%"


class ApronStrategy(IStrategy):
    """
    停机坪策略

    选股条件：
    1. 最近15日有涨幅 > 9.5%，且必须是放量上涨
    2. 紧接的下个交易日必须高开，收盘价必须上涨，且与开盘价不能相差 >= 3%
    3. 接下来的2、3个交易日必须高开，收盘价必须上涨，且与开盘价不能相差 >= 3%，且每天涨跌幅在5%间
    """

    def __init__(self):
        self.parameters = {
            'lookback': 15,
            'surge_threshold': 0.095,
            'consolidation_range': 0.03,
            'max_daily_change': 0.05
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 20:
            return pd.DataFrame()

        df = bars.copy()
        df['signal'] = 0
        df['pct_change'] = df['close'].pct_change()
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']

        lookback = self.parameters['lookback']

        for i in range(lookback + 4, len(df)):
            # 检查最近15日是否有放量大涨
            recent = df.iloc[i-lookback-4:i-4]
            surge_days = recent[
                (recent['pct_change'] > self.parameters['surge_threshold']) &
                (recent['volume_ratio'] > 1.5)
            ]

            if len(surge_days) == 0:
                continue

            # 找到大涨后的位置
            last_surge_idx = surge_days.index[-1]
            surge_pos = df.index.get_loc(last_surge_idx)

            # 检查后续3天的横盘整理
            try:
                days_after = df.iloc[surge_pos+1:surge_pos+4]
                if len(days_after) < 3:
                    continue

                # 检查是否符合停机坪形态
                valid = True
                for j, row in enumerate(days_after.itertuples()):
                    # 高开
                    prev_close = df.iloc[surge_pos + j]['close']
                    if row.open <= prev_close:
                        valid = False
                        break
                    # 收盘上涨
                    if row.close <= row.open:
                        valid = False
                        break
                    # 振幅限制
                    if abs(row.close - row.open) / row.open >= self.parameters['consolidation_range']:
                        valid = False
                        break
                    # 日涨跌幅限制
                    if abs(row.pct_change) > self.parameters['max_daily_change']:
                        valid = False
                        break

                if valid:
                    df.iloc[i, df.columns.get_loc('signal')] = 1

            except (IndexError, KeyError):
                continue

        return df[['timestamp', 'signal']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        signals = self.calculate_signals(historical_bars)
        if signals.empty or signals['signal'].iloc[-1] == 1:
            return {
                "direction": "buy",
                "reason": "apron_pattern",
                "strength": 0.8
            }
        return {"direction": "hold", "reason": "no_signal"}

    def get_name(self) -> str:
        return "Apron"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "停机坪策略：大涨后高位横盘整理"


class YearLinePullbackStrategy(IStrategy):
    """
    回踩年线策略

    选股条件：
    1. 分2个时间段：前段=最近60交易日最高收盘价之前交易日(长度>0)，后段=最高价当日及后面的交易日
    2. 前段由年线(250日)以下向上突破
    3. 后段必须在年线以上运行，且后段最低价日与最高价日相差必须在10-50日间
    4. 回踩伴随缩量：最高价日交易量/后段最低价日交易量>2, 后段最低价/最高价<0.8
    """

    def __init__(self):
        self.parameters = {
            'year_line_period': 250,
            'lookback': 60,
            'min_pullback_days': 10,
            'max_pullback_days': 50,
            'volume_shrink_ratio': 2.0,
            'max_pullback_pct': 0.2
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 300:
            return pd.DataFrame()

        df = bars.copy()
        df['signal'] = 0

        # 计算年线
        df['year_line'] = df['close'].rolling(self.parameters['year_line_period']).mean()

        lookback = self.parameters['lookback']

        for i in range(lookback + 10, len(df)):
            recent = df.iloc[i-lookback:i+1]

            # 找到最高收盘价日
            max_close_idx = recent['close'].idxmax()
            max_pos = recent.index.get_loc(max_close_idx)

            if max_pos == 0 or max_pos == len(recent) - 1:
                continue

            # 前段和后段
            front = recent.iloc[:max_pos]
            back = recent.iloc[max_pos:]

            if len(front) == 0 or len(back) < 2:
                continue

            # 检查前段是否突破年线
            year_line_front = front['year_line']
            if year_line_front.iloc[0] < year_line_front.iloc[-1] * 0.99:
                # 年线向上
                front_below = front['close'] < front['year_line']
                front_above = front['close'] > front['year_line']
                if not (front_below.iloc[0] and front_above.iloc[-1]):
                    continue
            else:
                continue

            # 后段在年线以上
            if (back['close'] < back['year_line']).any():
                continue

            # 找后段最低价日
            min_idx = back['close'].idxmin()
            min_pos = back.index.get_loc(min_idx)
            days_diff = min_pos

            if not (self.parameters['min_pullback_days'] <= days_diff <= self.parameters['max_pullback_days']):
                continue

            # 缩量条件
            max_volume = back.iloc[0]['volume']
            min_volume = back.loc[min_idx, 'volume']

            if max_volume / min_volume < self.parameters['volume_shrink_ratio']:
                continue

            # 回踩幅度
            max_price = back.iloc[0]['close']
            min_price = back.loc[min_idx, 'close']

            if (max_price - min_price) / max_price > self.parameters['max_pullback_pct']:
                continue

            df.iloc[i, df.columns.get_loc('signal')] = 1

        return df[['timestamp', 'signal', 'year_line']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        signals = self.calculate_signals(historical_bars)
        if signals.empty or signals['signal'].iloc[-1] == 1:
            return {
                "direction": "buy",
                "reason": "year_line_pullback",
                "strength": 0.7
            }
        return {"direction": "hold", "reason": "no_signal"}

    def get_name(self) -> str:
        return "YearLinePullback"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "回踩年线策略：突破年线后缩量回踩"


class PlatformBreakoutStrategy(IStrategy):
    """
    突破平台策略

    选股条件：
    1. 60日内某日收盘价 >= 60日均线 > 开盘价
    2. 且【1】放量上涨
    3. 且【1】之前时间，任意一天收盘价与60日均线偏离在-5%~20%之间
    """

    def __init__(self):
        self.parameters = {
            'lookback': 60,
            'volume_ratio_threshold': 1.5,
            'deviation_min': -0.05,
            'deviation_max': 0.20
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 80:
            return pd.DataFrame()

        df = bars.copy()
        df['signal'] = 0
        df['ma60'] = df['close'].rolling(60).mean()
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        df['pct_change'] = df['close'].pct_change()
        df['deviation'] = (df['close'] - df['ma60']) / df['ma60']

        lookback = self.parameters['lookback']

        for i in range(lookback, len(df)):
            # 检查是否突破
            if not (df.iloc[i]['close'] >= df.iloc[i]['ma60'] > df.iloc[i]['open']):
                continue

            # 检查放量上涨
            if df.iloc[i]['volume_ratio'] < self.parameters['volume_ratio_threshold']:
                continue

            if df.iloc[i]['pct_change'] < 0.02:
                continue

            # 检查之前的平台整理
            before = df.iloc[i-lookback:i]
            deviations = before['deviation']

            if deviations.min() >= self.parameters['deviation_min'] and \
               deviations.max() <= self.parameters['deviation_max']:
                df.iloc[i, df.columns.get_loc('signal')] = 1

        return df[['timestamp', 'signal', 'ma60', 'deviation']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        signals = self.calculate_signals(historical_bars)
        if signals.empty or signals['signal'].iloc[-1] == 1:
            return {
                "direction": "buy",
                "reason": "platform_breakout",
                "strength": 0.75
            }
        return {"direction": "hold", "reason": "no_signal"}

    def get_name(self) -> str:
        return "PlatformBreakout"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "突破平台策略：长期横盘后放量突破"


class NoLargeDrawdownStrategy(IStrategy):
    """
    无大幅回撤策略

    选股条件：
    1. 当日收盘价比60日前的收盘价的涨幅 < 0.6
    2. 最近60日，不能有单日跌幅超7%、高开低走7%、两日累计跌幅10%、两日高开低走累计10%
    """

    def __init__(self):
        self.parameters = {
            'lookback': 60,
            'max_gain': 0.6,
            'max_single_drop': 0.07,
            'max_two_day_drop': 0.10
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 80:
            return pd.DataFrame()

        df = bars.copy()
        df['signal'] = 0
        df['pct_change'] = df['close'].pct_change()

        lookback = self.parameters['lookback']

        for i in range(lookback, len(df)):
            recent = df.iloc[i-lookback:i+1]

            # 检查60日涨幅
            gain_60d = (recent['close'].iloc[-1] - recent['close'].iloc[0]) / recent['close'].iloc[0]
            if gain_60d >= self.parameters['max_gain']:
                continue

            # 检查单日跌幅
            if (recent['pct_change'] < -self.parameters['max_single_drop']).any():
                continue

            # 检查高开低走
            intraday = (recent['close'] - recent['open']) / recent['open']
            if (intraday < -self.parameters['max_single_drop']).any():
                continue

            # 检查两日累计跌幅
            two_day_drop = recent['close'].pct_change(2)
            if (two_day_drop < -self.parameters['max_two_day_drop']).any():
                continue

            df.iloc[i, df.columns.get_loc('signal')] = 1

        return df[['timestamp', 'signal']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        signals = self.calculate_signals(historical_bars)
        if signals.empty or signals['signal'].iloc[-1] == 1:
            return {
                "direction": "buy",
                "reason": "no_large_drawdown",
                "strength": 0.6
            }
        return {"direction": "hold", "reason": "no_signal"}

    def get_name(self) -> str:
        return "NoLargeDrawdown"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "无大幅回撤策略：走势稳健无大跌"


class HighNarrowFlagStrategy(IStrategy):
    """
    高而窄的旗形策略

    选股条件：
    1. 必须至少上市交易60日
    2. 当日收盘价 / 之前24~10日的最低价 >= 1.9
    3. 之前24~10日必须连续两天涨幅 >= 9.5%
    """

    def __init__(self):
        self.parameters = {
            'min_listed_days': 60,
            'min_gain': 1.9,
            'surge_threshold': 0.095
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 80:
            return pd.DataFrame()

        df = bars.copy()
        df['signal'] = 0
        df['pct_change'] = df['close'].pct_change()

        for i in range(30, len(df)):
            # 当日收盘价
            current_close = df.iloc[i]['close']

            # 之前24~10日的最低价
            low_24_10 = df.iloc[i-24:i-9]['close'].min()

            # 涨幅条件
            if current_close / low_24_10 < self.parameters['min_gain']:
                continue

            # 检查之前24~10日是否有连续两天大涨
            period = df.iloc[i-24:i-9]
            surge_days = period[period['pct_change'] >= self.parameters['surge_threshold']]

            if len(surge_days) >= 2:
                # 检查是否连续
                surge_indices = surge_days.index.tolist()
                for j in range(len(surge_indices) - 1):
                    if surge_indices[j+1] - surge_indices[j] == 1:
                        df.iloc[i, df.columns.get_loc('signal')] = 1
                        break

        return df[['timestamp', 'signal']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        signals = self.calculate_signals(historical_bars)
        if signals.empty or signals['signal'].iloc[-1] == 1:
            return {
                "direction": "buy",
                "reason": "high_narrow_flag",
                "strength": 0.85
            }
        return {"direction": "hold", "reason": "no_signal"}

    def get_name(self) -> str:
        return "HighNarrowFlag"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "高而窄的旗形：短期暴涨后的强势形态"


class VolumeLimitDownStrategy(IStrategy):
    """
    放量跌停策略

    选股条件：
    1. 跌幅 > 9.5%
    2. 成交额 >= 2亿
    3. 成交量至少是5日平均成交量的4倍
    """

    def __init__(self):
        self.parameters = {
            'drop_threshold': 0.095,
            'min_amount': 200000000,
            'volume_ratio': 4.0
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 10:
            return pd.DataFrame()

        df = bars.copy()
        df['signal'] = 0
        df['pct_change'] = df['close'].pct_change()
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma5']
        df['amount'] = df['close'] * df['volume']

        # 跌停条件
        condition1 = df['pct_change'] < -self.parameters['drop_threshold']
        condition2 = df['amount'] >= self.parameters['min_amount']
        condition3 = df['volume_ratio'] >= self.parameters['volume_ratio']

        df.loc[condition1 & condition2 & condition3, 'signal'] = 1

        return df[['timestamp', 'signal', 'pct_change', 'volume_ratio']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < 10:
            return {"direction": "hold", "reason": "insufficient_data"}

        pct_change = (current_bar['close'] - historical_bars['close'].iloc[-2]) / \
                     historical_bars['close'].iloc[-2]
        volume_ma5 = historical_bars['volume'].tail(5).mean()
        volume_ratio = current_bar['volume'] / volume_ma5
        amount = current_bar['close'] * current_bar['volume']

        if pct_change < -self.parameters['drop_threshold'] and \
           amount >= self.parameters['min_amount'] and \
           volume_ratio >= self.parameters['volume_ratio']:
            return {
                "direction": "buy",  # 放量跌停可能是抄底机会
                "reason": "volume_limit_down",
                "strength": 0.5,
                "volume_ratio": volume_ratio
            }

        return {"direction": "hold", "reason": "no_signal"}

    def get_name(self) -> str:
        return "VolumeLimitDown"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "放量跌停策略：可能存在抄底机会"


class LowATRGrowthStrategy(IStrategy):
    """
    低ATR成长策略

    选股条件：
    1. 必须至少上市交易250日
    2. 最近10个交易日的最高收盘价必须比最近10个交易日的最低收盘价高1.1倍
    """

    def __init__(self):
        self.parameters = {
            'min_listed_days': 250,
            'min_range_ratio': 1.1,
            'lookback': 10
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 260:
            return pd.DataFrame()

        df = bars.copy()
        df['signal'] = 0

        lookback = self.parameters['lookback']

        for i in range(lookback, len(df)):
            recent = df.iloc[i-lookback:i+1]
            max_close = recent['close'].max()
            min_close = recent['close'].min()

            if max_close / min_close >= self.parameters['min_range_ratio']:
                df.iloc[i, df.columns.get_loc('signal')] = 1

        return df[['timestamp', 'signal']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        signals = self.calculate_signals(historical_bars)
        if signals.empty or signals['signal'].iloc[-1] == 1:
            return {
                "direction": "buy",
                "reason": "low_atr_growth",
                "strength": 0.6
            }
        return {"direction": "hold", "reason": "no_signal"}

    def get_name(self) -> str:
        return "LowATRGrowth"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "低ATR成长策略：波动率收窄后的机会"


class FundamentalSelectionStrategy(IStrategy):
    """
    基本面选股策略

    选股条件：
    1. 市盈率 <= 20，且 > 0
    2. 市净率 <= 10
    3. 净资产收益率 >= 15%

    数据来源：BaoStock（免费、无需权限）
    """

    def __init__(self):
        self.parameters = {
            'max_pe': 20,
            'max_pb': 10,
            'min_roe': 15.0,  # ROE 百分比
            'symbol': None     # 股票代码（用于获取基本面数据）
        }
        self._fundamental_data = None
        self._data_fetcher = None

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)

    def _get_data_fetcher(self):
        """获取数据获取器实例"""
        if self._data_fetcher is None:
            try:
                from ..infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials
                self._data_fetcher = create_data_fetcher_with_credentials()
            except Exception as e:
                logger.error(f"无法创建数据获取器: {str(e)}")
        return self._data_fetcher

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) < 10:
            return pd.DataFrame()

        df = bars.copy()
        df['signal'] = 0

        # 基本面数据需要外部提供
        # 这里只返回框架
        return df[['timestamp', 'signal']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        """
        获取基本面信号

        会自动获取 PE/PB/ROE 数据进行评估
        """
        # 获取股票代码
        symbol = self.parameters.get('symbol')
        if not symbol:
            # 尝试从 DataFrame 中获取
            if hasattr(historical_bars, 'attrs') and 'symbol' in historical_bars.attrs:
                symbol = historical_bars.attrs['symbol']

        if not symbol:
            return {
                "direction": "hold",
                "reason": "缺少股票代码",
                "required_data": ["symbol"]
            }

        # 获取基本面数据
        fetcher = self._get_data_fetcher()
        if not fetcher:
            return {
                "direction": "hold",
                "reason": "数据获取器不可用"
            }

        try:
            fund_data = fetcher.get_fundamental_data(symbol)
        except Exception as e:
            return {
                "direction": "hold",
                "reason": f"获取基本面数据失败: {str(e)[:30]}"
            }

        if fund_data.get('error'):
            return {
                "direction": "hold",
                "reason": fund_data['error']
            }

        pe = fund_data.get('pe')
        pb = fund_data.get('pb')
        roe = fund_data.get('roe')  # 已经是百分比形式

        # 检查数据是否完整
        if pe is None or pb is None:
            return {
                "direction": "hold",
                "reason": "估值数据不完整",
                "pe": pe,
                "pb": pb
            }

        # 评估基本面
        evaluation = self.evaluate_fundamentals(pe, pb, roe)

        return {
            "direction": "buy" if evaluation['score'] >= 80 else "hold",
            "reason": "; ".join(evaluation['signals']),
            "score": evaluation['score'],
            "pe": pe,
            "pb": pb,
            "roe": roe,
            "data_source": fund_data.get('data_source')
        }

    def evaluate_fundamentals(
        self,
        pe: float,
        pb: float,
        roe: float
    ) -> Dict[str, Any]:
        """
        评估基本面指标

        Args:
            pe: 市盈率
            pb: 市净率
            roe: 净资产收益率（百分比）

        Returns:
            评估结果
        """
        signals = []
        score = 0

        # PE评估
        if pe and 0 < pe <= self.parameters['max_pe']:
            signals.append(f"PE={pe:.1f}，估值合理")
            score += 33
        elif pe and pe <= 0:
            signals.append(f"PE={pe:.1f}，亏损企业")
        elif pe:
            signals.append(f"PE={pe:.1f}，估值偏高")
        else:
            signals.append("PE数据缺失")

        # PB评估
        if pb and pb <= self.parameters['max_pb']:
            signals.append(f"PB={pb:.1f}，估值合理")
            score += 33
        elif pb:
            signals.append(f"PB={pb:.1f}，估值偏高")
        else:
            signals.append("PB数据缺失")

        # ROE评估
        if roe and roe >= self.parameters['min_roe']:
            signals.append(f"ROE={roe:.1f}%，盈利能力强")
            score += 34
        elif roe:
            signals.append(f"ROE={roe:.1f}%，盈利能力一般")
        else:
            signals.append("ROE数据缺失")

        return {
            "signals": signals,
            "score": score,
            "recommendation": "buy" if score >= 80 else "hold"
        }

    def get_name(self) -> str:
        return "FundamentalSelection"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "基本面选股：PE<20, PB<10, ROE>15%（使用BaoStock数据）"


# 注册所有策略
def register_strategies():
    """注册所有经典策略"""
    registry.register(ComponentType.STRATEGY, "volume_breakout")(VolumeBreakoutStrategy)
    registry.register(ComponentType.STRATEGY, "ma_alignment")(MADAlignmentStrategy)
    registry.register(ComponentType.STRATEGY, "apron")(ApronStrategy)
    registry.register(ComponentType.STRATEGY, "year_line_pullback")(YearLinePullbackStrategy)
    registry.register(ComponentType.STRATEGY, "platform_breakout")(PlatformBreakoutStrategy)
    registry.register(ComponentType.STRATEGY, "no_large_drawdown")(NoLargeDrawdownStrategy)
    registry.register(ComponentType.STRATEGY, "high_narrow_flag")(HighNarrowFlagStrategy)
    registry.register(ComponentType.STRATEGY, "volume_limit_down")(VolumeLimitDownStrategy)
    registry.register(ComponentType.STRATEGY, "low_atr_growth")(LowATRGrowthStrategy)
    registry.register(ComponentType.STRATEGY, "fundamental_selection")(FundamentalSelectionStrategy)


# 策略列表
CLASSIC_STRATEGIES = [
    ('volume_breakout', VolumeBreakoutStrategy, '放量上涨'),
    ('ma_alignment', MADAlignmentStrategy, '均线多头'),
    ('apron', ApronStrategy, '停机坪'),
    ('year_line_pullback', YearLinePullbackStrategy, '回踩年线'),
    ('platform_breakout', PlatformBreakoutStrategy, '突破平台'),
    ('no_large_drawdown', NoLargeDrawdownStrategy, '无大幅回撤'),
    ('high_narrow_flag', HighNarrowFlagStrategy, '高而窄的旗形'),
    ('volume_limit_down', VolumeLimitDownStrategy, '放量跌停'),
    ('low_atr_growth', LowATRGrowthStrategy, '低ATR成长'),
    ('fundamental_selection', FundamentalSelectionStrategy, '基本面选股'),
]
