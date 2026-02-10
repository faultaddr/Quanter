"""
Strategy Tools Set for A-Share Market
Multiple trading strategies with buy/sell signal generation
Uses Qlib-based strategies for advanced analysis
"""
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from .qlib_strategies import QlibStrategyManager

# Import the new advanced strategies
try:
    from .advanced_strategies import AdvancedStrategyManager
except ImportError:
    print("⚠️  Advanced strategies module not found, using basic strategies")
    AdvancedStrategyManager = None


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


class QlibStrategyAdapter(Strategy):
    """
    Adapter to wrap Qlib strategies in the original Strategy interface
    """
    def __init__(self, qlib_strategy):
        super().__init__(qlib_strategy.name)
        self.qlib_strategy = qlib_strategy

    def generate_signals(self, data):
        """
        Generate signals using the Qlib strategy
        """
        return self.qlib_strategy.generate_signals(data)


class StrategyManager:
    """
    策略管理器
    管理多种策略并提供统一接口
    Now uses Qlib-based strategies for advanced analysis
    Plus 100+ factor analysis from advanced strategies
    """
    def __init__(self):
        # Initialize Qlib-based strategy manager
        self.qlib_manager = QlibStrategyManager()

        # Initialize advanced strategy manager if available (before registering strategies)
        self.advanced_strategy_manager = None
        if AdvancedStrategyManager:
            try:
                self.advanced_strategy_manager = AdvancedStrategyManager()
                print(f"✅ Advanced strategy manager initialized with {len(self.advanced_strategy_manager.get_strategy_names())} strategies and 100+ factors")
            except Exception as e:
                print(f"⚠️  Could not initialize advanced strategy manager: {e}")
                self.advanced_strategy_manager = None

        self.strategies = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """
        注册默认策略 (now using Qlib-based strategies)
        """
        # Register Qlib-based strategies
        for strategy_name in self.qlib_manager.get_strategy_names():
            qlib_strategy = self.qlib_manager.get_strategy(strategy_name)
            if qlib_strategy:
                # Wrap the Qlib strategy in an adapter
                self.register_strategy(strategy_name, QlibStrategyAdapter(qlib_strategy))

        # Also register some traditional strategies for compatibility
        self.register_strategy('ma_crossover', TraditionalMovingAverageCrossoverStrategy())
        self.register_strategy('rsi', TraditionalRSIStrategy())
        self.register_strategy('macd', TraditionalMACDStrategy())

        # Register advanced strategies if available
        if self.advanced_strategy_manager:
            for strategy_name in self.advanced_strategy_manager.get_strategy_names():
                self.register_strategy(f'adv_{strategy_name}', self.advanced_strategy_manager.get_strategy(strategy_name))

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

    def get_advanced_strategy_manager(self):
        """
        获取高级策略管理器
        """
        return self.advanced_strategy_manager

    def get_strategy_names(self):
        """
        获取所有策略名称
        """
        return list(self.strategies.keys())

    def run_backtest(self, strategy_name, data, start_date=None, end_date=None):
        """
        Run backtest for a specific strategy using Qlib
        """
        if strategy_name in self.qlib_manager.get_strategy_names():
            return self.qlib_manager.run_backtest(strategy_name, data, start_date, end_date)
        else:
            # For traditional strategies, return a simple backtest result
            signals = self.run_strategy(strategy_name, data)
            return self._simple_backtest(signals, data)

    def _simple_backtest(self, signals, data):
        """
        Simple backtest for traditional strategies
        """
        if 'close' in data.columns:
            # Simple backtest logic
            positions = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
            daily_returns = data['close'].pct_change()
            strategy_returns = daily_returns * positions.shift(1)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
            
            # Calculate metrics
            total_return = cumulative_returns.iloc[-1] - 1 if len(cumulative_returns) > 0 else 0
            annualized_return = (cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1 if len(cumulative_returns) > 0 else 0
            
            # Calculate max drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min() if not drawdowns.empty else 0
            
            # Calculate Sharpe ratio (assuming risk-free rate of 0)
            excess_returns = strategy_returns.dropna()
            sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() != 0 else 0
            
            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'cumulative_returns': cumulative_returns,
                'signals': signals,
                'positions': positions
            }
        else:
            return {
                'total_return': 0,
                'annualized_return': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'cumulative_returns': pd.Series(dtype=float),
                'signals': signals,
                'positions': pd.Series(dtype=float)
            }


# Traditional strategies for backward compatibility
class TraditionalMovingAverageCrossoverStrategy(Strategy):
    """
    移动平均线交叉策略 (Traditional implementation kept for compatibility)
    当短期均线上穿长期均线时买入，下穿时卖出
    """
    def __init__(self):
        super().__init__("MA Crossover")

    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)

        if 'close' in data.columns:
            # 计算移动平均线 - Use shorter periods for more sensitivity
            short_ma = data['close'].rolling(window=5).mean()   # Changed from 10 to 5
            long_ma = data['close'].rolling(window=20).mean()   # Changed from 30 to 20

            # Calculate the difference between short and long MA
            ma_diff = short_ma - long_ma
            
            # Generate signals with additional sensitivity
            buy_signals = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1)) & \
                          (ma_diff > ma_diff.rolling(window=5).mean() * 0.5)  # Additional condition for confirmation
            sell_signals = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1)) & \
                           (ma_diff < ma_diff.rolling(window=5).mean() * 0.5)  # Additional condition for confirmation

            signals[buy_signals] = 1
            signals[sell_signals] = -1

        return signals


class TraditionalRSIStrategy(Strategy):
    """
    RSI超买超卖策略 (Traditional implementation kept for compatibility)
    RSI低于30时买入，高于70时卖出
    """
    def __init__(self):
        super().__init__("RSI")

    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)

        if 'close' in data.columns:
            # 计算RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            # 生成信号 - 使用更敏感的阈值
            buy_signals = (rsi < 35) & (rsi.shift(1) >= 35)  # Lowered from 30 to 35
            sell_signals = (rsi > 65) & (rsi.shift(1) <= 65)  # Lowered from 70 to 65

            signals[buy_signals] = 1
            signals[sell_signals] = -1

        return signals


class TraditionalMACDStrategy(Strategy):
    """
    MACD策略 (Traditional implementation kept for compatibility)
    MACD线上穿信号线时买入，下穿时卖出
    """
    def __init__(self):
        super().__init__("MACD")

    def generate_signals(self, data):
        signals = pd.Series(0, index=data.index)

        if 'close' in data.columns:
            # 计算MACD
            exp12 = data['close'].ewm(span=12).mean()
            exp26 = data['close'].ewm(span=26).mean()
            macd = exp12 - exp26
            signal = macd.ewm(span=9).mean()
            
            # Calculate histogram for additional sensitivity
            histogram = macd - signal

            # 生成信号 - Add additional conditions for more sensitivity
            buy_signals = ((macd > signal) & (macd.shift(1) <= signal.shift(1))) | \
                          ((histogram > 0) & (histogram.shift(1) <= 0))  # Histogram crossing zero line upward
            sell_signals = ((macd < signal) & (macd.shift(1) >= signal.shift(1))) | \
                           ((histogram < 0) & (histogram.shift(1) >= 0))  # Histogram crossing zero line downward

            signals[buy_signals] = 1
            signals[sell_signals] = -1

        return signals


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

    # 测试策略
    manager = StrategyManager()

    print("Available strategies:", manager.get_strategy_names())
    
    # Test Qlib-based strategies
    for strategy_name in manager.get_strategy_names()[:3]:  # Test first 3 strategies
        try:
            signals = manager.run_strategy(strategy_name, data)
            print(f"\n{strategy_name} 策略信号 (非零信号数量: {len(signals[signals != 0])})")
            
            # Run backtest
            backtest_result = manager.run_backtest(strategy_name, data)
            print(f"  总收益率: {backtest_result['total_return']:.4f}")
            print(f"  夏普比率: {backtest_result['sharpe_ratio']:.4f}")
            print(f"  最大回撤: {backtest_result['max_drawdown']:.4f}")
        except Exception as e:
            print(f"Error testing {strategy_name}: {e}")