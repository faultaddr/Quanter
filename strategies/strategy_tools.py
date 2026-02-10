"""
Strategy Tools Set for A-Share Market
Multiple trading strategies with buy/sell signal generation
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class Strategy(ABC):
    """
    交易策略抽象基类
    """
    def __init__(self, name):
        self.name = name
    
    @abstractmethod
    def generate_signals(self, data):
        """
        生成买卖信号
        :param data: 包含OHLCV等技术指标的数据
        :return: 信号序列 (1=买入, -1=卖出, 0=持有)
        """
        pass


class MovingAverageCrossoverStrategy(Strategy):
    """
    移动平均线交叉策略
    当短期均线上穿长期均线时买入，下穿时卖出
    """
    def __init__(self, short_window=10, long_window=30):
        super().__init__("MA Crossover")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)
        
        # 计算移动平均线
        short_ma = data['close'].rolling(window=self.short_window).mean()
        long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # 生成信号
        for i in range(1, len(data)):
            # 金叉：短期均线上穿长期均线
            if short_ma.iloc[i] > long_ma.iloc[i] and short_ma.iloc[i-1] <= long_ma.iloc[i-1]:
                signals.iloc[i] = 1
            # 死叉：短期均线下穿长期均线
            elif short_ma.iloc[i] < long_ma.iloc[i] and short_ma.iloc[i-1] >= long_ma.iloc[i-1]:
                signals.iloc[i] = -1
        
        return signals


class RSIStrategy(Strategy):
    """
    RSI超买超卖策略
    RSI低于30时买入，高于70时卖出
    """
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        super().__init__("RSI")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)
        
        # 如果数据中没有RSI列，则计算RSI
        if 'rsi' not in data.columns:
            # 计算RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
        else:
            rsi = data['rsi']
        
        # 生成信号
        for i in range(1, len(data)):
            # 超卖买入
            if rsi.iloc[i] <= self.oversold and rsi.iloc[i-1] > self.oversold:
                signals.iloc[i] = 1
            # 超买卖出
            elif rsi.iloc[i] >= self.overbought and rsi.iloc[i-1] < self.overbought:
                signals.iloc[i] = -1
        
        return signals


class MACDStrategy(Strategy):
    """
    MACD策略
    MACD线上穿信号线时买入，下穿时卖出
    """
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        super().__init__("MACD")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)
        
        # 如果数据中没有MACD相关列，则计算MACD
        if 'macd' not in data.columns or 'signal' not in data.columns:
            exp12 = data['close'].ewm(span=self.fast_period).mean()
            exp26 = data['close'].ewm(span=self.slow_period).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=self.signal_period).mean()
        else:
            macd = data['macd']
            signal = data['signal']
        
        # 生成信号
        for i in range(1, len(data)):
            # MACD上穿信号线（金叉）买入
            if macd.iloc[i] > signal.iloc[i] and macd.iloc[i-1] <= signal.iloc[i-1]:
                signals.iloc[i] = 1
            # MACD下穿信号线（死叉）卖出
            elif macd.iloc[i] < signal.iloc[i] and macd.iloc[i-1] >= signal.iloc[i-1]:
                signals.iloc[i] = -1
        
        return signals


class BollingerBandsStrategy(Strategy):
    """
    布林带策略
    价格触及下轨买入，触及上轨卖出
    """
    def __init__(self, window=20, num_std=2):
        super().__init__("Bollinger Bands")
        self.window = window
        self.num_std = num_std
    
    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)
        
        # 如果数据中没有布林带相关列，则计算布林带
        if 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
            rolling_mean = data['close'].rolling(window=self.window).mean()
            rolling_std = data['close'].rolling(window=self.window).std()
            bb_upper = rolling_mean + (rolling_std * self.num_std)
            bb_lower = rolling_mean - (rolling_std * self.num_std)
        else:
            bb_upper = data['bb_upper']
            bb_lower = data['bb_lower']
        
        # 生成信号
        for i in range(1, len(data)):
            # 触及下轨买入
            if data['close'].iloc[i] <= bb_lower.iloc[i] and data['close'].iloc[i-1] > bb_lower.iloc[i-1]:
                signals.iloc[i] = 1
            # 触及上轨卖出
            elif data['close'].iloc[i] >= bb_upper.iloc[i] and data['close'].iloc[i-1] < bb_upper.iloc[i-1]:
                signals.iloc[i] = -1
        
        return signals


class MeanReversionStrategy(Strategy):
    """
    均值回归策略
    当价格偏离移动平均线超过阈值时反向操作
    """
    def __init__(self, window=20, threshold=1.5):
        super().__init__("Mean Reversion")
        self.window = window
        self.threshold = threshold
    
    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)
        
        # 计算移动平均线和标准差
        ma = data['close'].rolling(window=self.window).mean()
        std = data['close'].rolling(window=self.window).std()
        
        # 生成信号
        for i in range(self.window, len(data)):
            # 当前价格与移动平均线的偏差
            deviation = (data['close'].iloc[i] - ma.iloc[i]) / std.iloc[i]
            
            # 显著高于均值，卖出
            if deviation > self.threshold:
                signals.iloc[i] = -1
            # 显著低于均值，买入
            elif deviation < -self.threshold:
                signals.iloc[i] = 1
        
        return signals


class BreakoutStrategy(Strategy):
    """
    突破策略
    价格突破近期高点买入，跌破近期低点卖出
    """
    def __init__(self, window=20):
        super().__init__("Breakout")
        self.window = window
    
    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)
        
        # 计算滚动窗口的高低点
        high_roll = data['high'].rolling(window=self.window).max()
        low_roll = data['low'].rolling(window=self.window).min()
        
        # 生成信号
        for i in range(self.window, len(data)):
            # 突破前期高点买入
            if data['close'].iloc[i] > high_roll.iloc[i-1]:
                signals.iloc[i] = 1
            # 跌破前期低点卖出
            elif data['close'].iloc[i] < low_roll.iloc[i-1]:
                signals.iloc[i] = -1
        
        return signals


class StrategyManager:
    """
    策略管理器
    管理多种策略并提供统一接口
    """
    def __init__(self):
        self.strategies = {}
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """
        注册默认策略
        """
        self.register_strategy('ma_crossover', MovingAverageCrossoverStrategy())
        self.register_strategy('rsi', RSIStrategy())
        self.register_strategy('macd', MACDStrategy())
        self.register_strategy('bollinger', BollingerBandsStrategy())
        self.register_strategy('mean_reversion', MeanReversionStrategy())
        self.register_strategy('breakout', BreakoutStrategy())
    
    def register_strategy(self, name, strategy):
        """
        注册新策略
        """
        self.strategies[name] = strategy
    
    def get_strategy(self, name):
        """
        获取策略
        """
        return self.strategies.get(name)
    
    def run_strategy(self, strategy_name, data):
        """
        运行指定策略
        """
        strategy = self.get_strategy(strategy_name)
        if strategy is None:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        return strategy.generate_signals(data)
    
    def run_all_strategies(self, data):
        """
        运行所有策略并返回信号矩阵
        """
        signals_matrix = pd.DataFrame(index=data.index)
        
        for name, strategy in self.strategies.items():
            signals_matrix[name] = strategy.generate_signals(data)
        
        return signals_matrix
    
    def get_strategy_names(self):
        """
        获取所有策略名称
        """
        return list(self.strategies.keys())


if __name__ == "__main__":
    # 示例使用
    import pandas as pd
    import numpy as np
    
    # 创建模拟数据
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(200) * 0.01),
        'high': prices * (1 + abs(np.random.randn(200)) * 0.02),
        'low': prices * (1 - abs(np.random.randn(200)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)
    
    # 计算一些基本的技术指标
    data['ma_10'] = data['close'].rolling(10).mean()
    data['ma_30'] = data['close'].rolling(30).mean()
    
    # 计算RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # 计算MACD
    exp12 = data['close'].ewm(span=12).mean()
    exp26 = data['close'].ewm(span=26).mean()
    data['macd'] = exp12 - exp26
    data['signal'] = data['macd'].ewm(span=9).mean()
    
    # 计算布林带
    data['bb_middle'] = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    
    # 测试策略
    manager = StrategyManager()
    
    print("测试各策略信号生成:")
    for strategy_name in manager.get_strategy_names()[:3]:  # 只测试前3个策略
        signals = manager.run_strategy(strategy_name, data)
        print(f"\n{strategy_name} 策略信号 (最近10天):")
        print(signals.tail(10))