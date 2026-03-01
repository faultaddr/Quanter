"""
Trading strategies module for evaluating buy/sell signals based on technical indicators
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .tech_indicators import *


class TradingStrategies:
    """
    A collection of trading strategies based on technical indicators
    """

    @staticmethod
    def rsi_strategy(df: pd.DataFrame, rsi_period: int = 14, rsi_buy_threshold: float = 30, rsi_sell_threshold: float = 70) -> pd.Series:
        """
        RSI-based strategy: Buy when RSI is below threshold, Sell when above threshold
        """
        signals = pd.Series(index=df.index, dtype='object')
        signals[:] = 'HOLD'

        # Use the RSI column based on period - for now using the 24-period which was calculated
        rsi_col = 'rsi_24'  # Default to 24-period RSI
        if rsi_period == 6:
            rsi_col = 'rsi_6'
        elif rsi_period == 12:
            rsi_col = 'rsi_12'

        if rsi_col in df.columns:
            signals[df.index[df[rsi_col] < rsi_buy_threshold]] = 'BUY'
            signals[df.index[df[rsi_col] > rsi_sell_threshold]] = 'SELL'

        return signals

    @staticmethod
    def macd_strategy(df: pd.DataFrame) -> pd.Series:
        """
        MACD-based strategy: Buy when MACD line crosses above signal line, Sell when below
        """
        signals = pd.Series(index=df.index, dtype='object')
        signals[:] = 'HOLD'

        # Find MACD crossover points
        if 'macd' in df.columns and 'macd_dea' in df.columns:
            macd_line = df['macd']
            signal_line = df['macd_dea']

            # Buy when MACD crosses above signal line
            buy_signals = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
            sell_signals = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))

            signals[buy_signals] = 'BUY'
            signals[sell_signals] = 'SELL'

        return signals

    @staticmethod
    def ma_crossover_strategy(df: pd.DataFrame, short_ma: int = 5, long_ma: int = 20) -> pd.Series:
        """
        Moving average crossover strategy: Buy when short MA crosses above long MA, Sell when below
        """
        signals = pd.Series(index=df.index, dtype='object')
        signals[:] = 'HOLD'

        short_ma_col = f'ma_{short_ma}'
        long_ma_col = f'ma_{long_ma}'

        if short_ma_col in df.columns and long_ma_col in df.columns:
            ma_short = df[short_ma_col]
            ma_long = df[long_ma_col]

            # Buy when short MA crosses above long MA
            buy_signals = (ma_short > ma_long) & (ma_short.shift(1) <= ma_long.shift(1))
            sell_signals = (ma_short < ma_long) & (ma_short.shift(1) >= ma_long.shift(1))

            signals[buy_signals] = 'BUY'
            signals[sell_signals] = 'SELL'

        return signals

    @staticmethod
    def bollinger_bands_strategy(df: pd.DataFrame, bb_period: int = 20, bb_std: int = 2) -> pd.Series:
        """
        Bollinger Bands strategy: Buy when price touches lower band, Sell when touches upper band
        """
        signals = pd.Series(index=df.index, dtype='object')
        signals[:] = 'HOLD'

        if 'close' in df.columns and 'boll_lower' in df.columns and 'boll_upper' in df.columns:
            prices = df['close']
            bb_lower = df['boll_lower']
            bb_upper = df['boll_upper']

            # Buy when price touches or goes below lower band
            buy_signals = prices <= bb_lower
            # Sell when price touches or goes above upper band
            sell_signals = prices >= bb_upper

            signals[buy_signals] = 'BUY'
            signals[sell_signals] = 'SELL'

        return signals

    @staticmethod
    def combined_strategy(df: pd.DataFrame) -> pd.Series:
        """
        Combined strategy: Uses multiple indicators for stronger signals
        """
        # Calculate individual signals
        rsi_sig = TradingStrategies.rsi_strategy(df)
        macd_sig = TradingStrategies.macd_strategy(df)
        ma_sig = TradingStrategies.ma_crossover_strategy(df)
        bb_sig = TradingStrategies.bollinger_bands_strategy(df)

        # Count buy/sell signals
        buy_count = (rsi_sig == 'BUY').astype(int) + \
                    (macd_sig == 'BUY').astype(int) + \
                    (ma_sig == 'BUY').astype(int) + \
                    (bb_sig == 'BUY').astype(int)

        sell_count = (rsi_sig == 'SELL').astype(int) + \
                     (macd_sig == 'SELL').astype(int) + \
                     (ma_sig == 'SELL').astype(int) + \
                     (bb_sig == 'SELL').astype(int)

        signals = pd.Series(index=df.index, dtype='object')
        signals[:] = 'HOLD'

        # Strong buy signal: 3 or more indicators suggest buying
        strong_buy = (buy_count >= 2) & (sell_count == 0)
        # Strong sell signal: 3 or more indicators suggest selling
        strong_sell = (sell_count >= 2) & (buy_count == 0)
        # Weak signals: 1 indicator suggests action
        weak_buy = (buy_count >= 1) & (sell_count == 0) & ~strong_buy
        weak_sell = (sell_count >= 1) & (buy_count == 0) & ~strong_sell

        signals[strong_buy] = 'STRONG_BUY'
        signals[weak_buy] = 'WEAK_BUY'
        signals[strong_sell] = 'STRONG_SELL'
        signals[weak_sell] = 'WEAK_SELL'

        return signals

    @staticmethod
    def evaluate_current_signal(signals: pd.Series, strategy_name: str = "Combined Strategy") -> Dict:
        """
        评估当前信号（中文版本）
        """
        if len(signals) == 0:
            return {"error": "无可用信号"}

        latest_signal = signals.iloc[-1]
        prev_signal = signals.iloc[-2] if len(signals) > 1 else None

        # Get the index of the last signal change
        changes = signals != signals.shift(1)
        last_change_idx = changes[::-1].idxmax() if changes.any() else signals.index[-1]

        evaluation = {
            "strategy": strategy_name,
            "current_signal": latest_signal,
            "previous_signal": prev_signal,
            "signal_changed": latest_signal != prev_signal if prev_signal else False,
            "last_change_date": last_change_idx,
            "strength": "normal"
        }

        if latest_signal == 'STRONG_BUY':
            evaluation["action"] = "强烈买入 - 建议积极加仓"
            evaluation["confidence"] = "High"
        elif latest_signal == 'WEAK_BUY':
            evaluation["action"] = "弱势买入 - 可小仓位试探"
            evaluation["confidence"] = "Medium"
        elif latest_signal == 'STRONG_SELL':
            evaluation["action"] = "强烈卖出 - 建议减仓规避风险"
            evaluation["confidence"] = "High"
        elif latest_signal == 'WEAK_SELL':
            evaluation["action"] = "弱势卖出 - 可考虑小幅减仓"
            evaluation["confidence"] = "Medium"
        elif latest_signal == 'BUY':
            evaluation["action"] = "买入信号"
            evaluation["confidence"] = "Medium"
        elif latest_signal == 'SELL':
            evaluation["action"] = "卖出信号"
            evaluation["confidence"] = "Medium"
        else:
            evaluation["action"] = "维持现状"
            evaluation["confidence"] = "Low"

        return evaluation