"""
Trading strategies for A-Share market
"""
import pandas as pd
import numpy as np
from ..backtest.backtester import BaseStrategy


class MovingAverageCrossoverStrategy(BaseStrategy):
    """
    Moving Average Crossover Strategy
    Generates buy/sell signals based on short-term and long-term moving average crossovers
    """
    
    def __init__(self, short_window=10, long_window=30, **kwargs):
        super().__init__(**kwargs)
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossover
        
        Args:
            data (pd.DataFrame): Historical price data
        
        Returns:
            pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.Series(0, index=data.index)
        
        # Calculate moving averages
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # Generate signals
        # Buy signal: short MA crosses above long MA
        buy_signals = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
        
        # Sell signal: short MA crosses below long MA
        sell_signals = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals


class RSIStrategy(BaseStrategy):
    """
    RSI (Relative Strength Index) Strategy
    Generates buy/sell signals based on RSI values
    """
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70, **kwargs):
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def generate_signals(self, data):
        """
        Generate trading signals based on RSI
        
        Args:
            data (pd.DataFrame): Historical price data
        
        Returns:
            pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.Series(0, index=data.index)
        
        # Calculate RSI
        rsi = self.calculate_rsi(data['close'], self.rsi_period)
        
        # Generate signals
        # Buy when RSI crosses above oversold level (indicating potential reversal from oversold condition)
        buy_signals = (rsi > self.oversold) & (rsi.shift(1) <= self.oversold)
        
        # Sell when RSI crosses below overbought level (indicating potential reversal from overbought condition)
        sell_signals = (rsi < self.overbought) & (rsi.shift(1) >= self.overbought)
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals


class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy
    Buys when price is significantly below moving average, sells when significantly above
    """
    
    def __init__(self, window=20, threshold=1.5, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.threshold = threshold  # Number of standard deviations
    
    def generate_signals(self, data):
        """
        Generate trading signals based on mean reversion
        
        Args:
            data (pd.DataFrame): Historical price data
        
        Returns:
            pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.Series(0, index=data.index)
        
        # Calculate moving average and standard deviation
        ma = data['close'].rolling(window=self.window).mean()
        std = data['close'].rolling(window=self.window).std()
        
        # Calculate z-score
        z_score = (data['close'] - ma) / std
        
        # Generate signals
        # Buy when price is significantly below mean (z_score < -threshold)
        buy_signals = z_score < -self.threshold
        
        # Sell when price is significantly above mean (z_score > threshold)
        sell_signals = z_score > self.threshold
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Strategy
    Buys when price touches lower band, sells when touches upper band
    """
    
    def __init__(self, window=20, num_std=2, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data):
        """
        Generate trading signals based on Bollinger Bands
        
        Args:
            data (pd.DataFrame): Historical price data
        
        Returns:
            pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.Series(0, index=data.index)
        
        # Calculate Bollinger Bands
        ma = data['close'].rolling(window=self.window).mean()
        std = data['close'].rolling(window=self.window).std()
        
        upper_band = ma + (std * self.num_std)
        lower_band = ma - (std * self.num_std)
        
        # Generate signals
        # Buy when price touches lower band
        buy_signals = (data['close'] <= lower_band) & (data['close'].shift(1) > lower_band.shift(1))
        
        # Sell when price touches upper band
        sell_signals = (data['close'] >= upper_band) & (data['close'].shift(1) < upper_band.shift(1))
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals


class MACDStrategy(BaseStrategy):
    """
    MACD (Moving Average Convergence Divergence) Strategy
    Generates signals based on MACD line and signal line crossovers
    """
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9, **kwargs):
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data):
        """
        Generate trading signals based on MACD
        
        Args:
            data (pd.DataFrame): Historical price data
        
        Returns:
            pd.Series: Trading signals (1 for buy, -1 for sell, 0 for hold)
        """
        signals = pd.Series(0, index=data.index)
        
        # Calculate MACD
        exp1 = data['close'].ewm(span=self.fast_period).mean()
        exp2 = data['close'].ewm(span=self.slow_period).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=self.signal_period).mean()
        
        # Generate signals
        # Buy when MACD line crosses above signal line
        buy_signals = (macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))
        
        # Sell when MACD line crosses below signal line
        sell_signals = (macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))
        
        signals[buy_signals] = 1
        signals[sell_signals] = -1
        
        return signals