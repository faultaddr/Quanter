"""
Qlib-inspired Strategy Manager for A-Share Market
Uses Qlib-like structure and concepts for strategy development and analysis
"""
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


class QlibStrategyManager:
    """
    Strategy manager using Qlib's advanced algorithms
    """
    def __init__(self):
        # Initialize Qlib
        self._init_qlib()
        self.strategies = {}
        self._register_qlib_strategies()

    def _init_qlib(self):
        """
        Initialize Qlib with A-Share configuration
        """
        try:
            import qlib
            from qlib.constant import REG_CN
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
            print("✅ Qlib initialized successfully")
        except ImportError:
            print("⚠️  Qlib not installed, proceeding with limited functionality")
        except Exception as e:
            print(f"⚠️  Failed to initialize Qlib with local data: {e}")
            # Try initializing with online data
            try:
                import qlib
                from qlib.constant import REG_CN
                qlib.init(provider_uri="https://github.com/microsoft/qlib/tree/main/examples/data", region=REG_CN)
                print("✅ Qlib initialized with online data")
            except Exception as e2:
                print(f"⚠️  Failed to initialize Qlib: {e2}")
                print("⚠️  Proceeding with limited functionality")

    def _register_qlib_strategies(self):
        """
        Register Qlib-based strategies
        """
        # Register Qlib-style strategies
        self.register_strategy('alphas101', QlibAlpha101Strategy())
        self.register_strategy('ml_gbdt', QlibGBDTStrategy())
        self.register_strategy('technical', QlibTechnicalStrategy())
        self.register_strategy('ensemble', QlibEnsembleStrategy())

    def register_strategy(self, name, strategy):
        """
        Register a new strategy
        """
        self.strategies[name] = strategy

    def get_strategy(self, name):
        """
        Get a strategy by name
        """
        return self.strategies.get(name)

    def run_strategy(self, strategy_name, data):
        """
        Run a specific strategy on the provided data
        """
        strategy = self.get_strategy(strategy_name)
        if strategy is None:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        return strategy.generate_signals(data)

    def run_all_strategies(self, data):
        """
        Run all registered strategies and return signals matrix
        """
        signals_matrix = pd.DataFrame(index=data.index)

        for name, strategy in self.strategies.items():
            try:
                signals_matrix[name] = strategy.generate_signals(data)
            except Exception as e:
                print(f"⚠️  Error running strategy {name}: {e}")
                signals_matrix[name] = pd.Series(0, index=data.index)

        return signals_matrix

    def get_strategy_names(self):
        """
        Get all registered strategy names
        """
        return list(self.strategies.keys())

    def run_backtest(self, strategy_name, data, start_date=None, end_date=None):
        """
        Run backtest for a specific strategy
        """
        strategy = self.get_strategy(strategy_name)
        if strategy is None:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        return strategy.backtest(data, start_date, end_date)


class BaseQlibStrategy:
    """
    Base class for Qlib-based strategies
    """
    def __init__(self, name):
        self.name = name

    def generate_signals(self, data):
        """
        Generate trading signals based on input data
        """
        raise NotImplementedError("Subclasses must implement generate_signals method")

    def backtest(self, data, start_date=None, end_date=None):
        """
        Run backtest on the strategy
        """
        raise NotImplementedError("Subclasses must implement backtest method")


class QlibAlpha101Strategy(BaseQlibStrategy):
    """
    Qlib Alpha101-based strategy
    Uses Qlib's alpha101 features for signal generation
    """
    def __init__(self):
        super().__init__("Alpha101")
        self.alpha_features = [
            '$close/$open-1',  # Return
            'Rank($volume)/Rank($close)',  # Volume-price rank ratio
            'Ts_Sum($high-$low, 10)/Ts_Sum(Ts_Sum($high-$low, 2), 5)'  # Volatility measure
        ]

    def generate_signals(self, data):
        """
        Generate signals using Alpha101-inspired features
        """
        signals = pd.Series(0.0, index=data.index)

        try:
            # Calculate alpha-inspired features
            if 'close' in data.columns and 'open' in data.columns:
                returns = (data['close'] / data['open'] - 1) if (data['open'] != 0).all() else pd.Series(0, index=data.index)
                
                # Normalize returns to range [-1, 1] for signal
                returns_normalized = returns / (returns.abs().max() + 1e-10)  # Avoid division by zero
                
                # Generate signals based on normalized returns
                signals = returns_normalized.clip(-1, 1)

                # Convert to discrete signals: 1 for buy, -1 for sell, 0 for hold
                # Made thresholds more sensitive to catch smaller movements
                signals = signals.apply(lambda x: 1 if x > 0.02 else (-1 if x < -0.02 else 0))
        except Exception as e:
            print(f"⚠️  Error in Alpha101 strategy: {e}")
            signals = pd.Series(0, index=data.index)

        return signals

    def backtest(self, data, start_date=None, end_date=None):
        """
        Run backtest for Alpha101 strategy
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Simple backtest logic
        positions = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate returns
        if 'close' in data.columns:
            daily_returns = data['close'].pct_change()
            strategy_returns = daily_returns * positions.shift(1)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
            
            # Calculate metrics
            total_return = cumulative_returns.iloc[-1] - 1
            annualized_return = (cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1 if len(cumulative_returns) > 0 else 0
            
            # Calculate max drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
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


class QlibGBDTStrategy(BaseQlibStrategy):
    """
    Gradient Boosting Decision Tree strategy using Qlib
    """
    def __init__(self):
        super().__init__("GBDT")
        # Define features commonly used in ML models
        self.features = ['open', 'high', 'low', 'close', 'volume']

    def generate_signals(self, data):
        """
        Generate signals using a simple ML approach
        """
        signals = pd.Series(0.0, index=data.index)

        try:
            # Create technical features
            if 'close' in data.columns:
                # Moving averages
                data['ma_5'] = data['close'].rolling(5).mean()
                data['ma_10'] = data['close'].rolling(10).mean()
                data['ma_20'] = data['close'].rolling(20).mean()
                
                # RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
                
                # Price position relative to moving averages
                if 'ma_5' in data.columns:
                    data['price_pos_ma5'] = (data['close'] - data['ma_5']) / data['ma_5']
                
                # Generate signals based on technical indicators
                conditions_buy = []
                conditions_sell = []

                # RSI conditions - Made more sensitive
                if 'rsi' in data.columns:
                    conditions_buy.append((data['rsi'] < 35) & (data['rsi'].shift(1) >= 35))  # Lowered from 30 to 35
                    conditions_sell.append((data['rsi'] > 65) & (data['rsi'].shift(1) <= 65))  # Lowered from 70 to 65

                # Moving average conditions
                if 'ma_5' in data.columns and 'ma_20' in data.columns:
                    conditions_buy.append((data['close'] > data['ma_5']) & (data['ma_5'] > data['ma_20']))  # Bullish cross
                    conditions_sell.append((data['close'] < data['ma_5']) & (data['ma_5'] < data['ma_20']))  # Bearish cross

                # Additional conditions for more sensitivity
                # Price position relative to moving averages
                if 'price_pos_ma5' in data.columns:
                    conditions_buy.append((data['price_pos_ma5'] > 0.02) & (data['price_pos_ma5'].shift(1) <= 0.02))  # Positive movement
                    conditions_sell.append((data['price_pos_ma5'] < -0.02) & (data['price_pos_ma5'].shift(1) >= -0.02))  # Negative movement

                # Combine conditions
                buy_condition = pd.Series(False, index=data.index)
                sell_condition = pd.Series(False, index=data.index)

                for cond in conditions_buy:
                    buy_condition |= cond

                for cond in conditions_sell:
                    sell_condition |= cond

                signals[buy_condition] = 1
                signals[sell_condition] = -1
                
        except Exception as e:
            print(f"⚠️  Error in GBDT strategy: {e}")
            signals = pd.Series(0, index=data.index)

        return signals

    def backtest(self, data, start_date=None, end_date=None):
        """
        Run backtest for GBDT strategy
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Simple backtest logic
        positions = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate returns
        if 'close' in data.columns:
            daily_returns = data['close'].pct_change()
            strategy_returns = daily_returns * positions.shift(1)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
            
            # Calculate metrics
            total_return = cumulative_returns.iloc[-1] - 1
            annualized_return = (cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1 if len(cumulative_returns) > 0 else 0
            
            # Calculate max drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
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


class QlibTechnicalStrategy(BaseQlibStrategy):
    """
    Advanced technical analysis strategy using Qlib concepts
    """
    def __init__(self):
        super().__init__("Technical")
        self.lookback_period = 20

    def generate_signals(self, data):
        """
        Generate signals using advanced technical indicators
        """
        signals = pd.Series(0.0, index=data.index)

        try:
            # Calculate technical indicators
            if 'close' in data.columns:
                # Bollinger Bands
                data['bb_mid'] = data['close'].rolling(self.lookback_period).mean()
                bb_std = data['close'].rolling(self.lookback_period).std()
                data['bb_upper'] = data['bb_mid'] + (bb_std * 2)
                data['bb_lower'] = data['bb_mid'] - (bb_std * 2)
                
                # MACD
                exp12 = data['close'].ewm(span=12).mean()
                exp26 = data['close'].ewm(span=26).mean()
                macd_line = exp12 - exp26
                signal_line = macd_line.ewm(span=9).mean()
                data['macd_hist'] = macd_line - signal_line
                
                # RSI
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                data['rsi'] = 100 - (100 / (1 + rs))
                
                # Generate signals based on technical combinations
                # Buy signals - Made more sensitive
                bb_buy = (data['close'] <= data['bb_lower'] * 1.02) & (data['close'].shift(1) > data['bb_lower'] * 1.02)  # Slightly above lower band
                macd_buy = (data['macd_hist'] > data['macd_hist'].shift(1)) & (data['macd_hist'] <= 0)  # Allow zero or slightly negative
                rsi_buy = (data['rsi'] < 35) & (data['rsi'].shift(1) >= 35)  # Lowered from 30 to 35

                # Sell signals - Made more sensitive
                bb_sell = (data['close'] >= data['bb_upper'] * 0.98) & (data['close'].shift(1) < data['bb_upper'] * 0.98)  # Slightly below upper band
                macd_sell = (data['macd_hist'] < data['macd_hist'].shift(1)) & (data['macd_hist'] >= 0)  # Allow zero or slightly positive
                rsi_sell = (data['rsi'] > 65) & (data['rsi'].shift(1) <= 65)  # Lowered from 70 to 65

                # Additional signals for more sensitivity
                # Momentum-based signals
                momentum_buy = (data['momentum'] > 0.02) & (data['momentum'].shift(1) <= 0.02)  # Positive momentum
                momentum_sell = (data['momentum'] < -0.02) & (data['momentum'].shift(1) >= -0.02)  # Negative momentum

                # Combine signals
                combined_buy = bb_buy | (macd_buy & rsi_buy) | (bb_buy & macd_buy) | momentum_buy
                combined_sell = bb_sell | (macd_sell & rsi_sell) | (bb_sell & macd_sell) | momentum_sell
                
                signals[combined_buy] = 1
                signals[combined_sell] = -1
                
        except Exception as e:
            print(f"⚠️  Error in Technical strategy: {e}")
            signals = pd.Series(0, index=data.index)

        return signals

    def backtest(self, data, start_date=None, end_date=None):
        """
        Run backtest for Technical strategy
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Simple backtest logic
        positions = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate returns
        if 'close' in data.columns:
            daily_returns = data['close'].pct_change()
            strategy_returns = daily_returns * positions.shift(1)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
            
            # Calculate metrics
            total_return = cumulative_returns.iloc[-1] - 1
            annualized_return = (cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1 if len(cumulative_returns) > 0 else 0
            
            # Calculate max drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
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


class QlibEnsembleStrategy(BaseQlibStrategy):
    """
    Ensemble strategy combining multiple Qlib approaches
    """
    def __init__(self):
        super().__init__("Ensemble")
        self.alpha_strategy = QlibAlpha101Strategy()
        self.gbdt_strategy = QlibGBDTStrategy()
        self.tech_strategy = QlibTechnicalStrategy()

    def generate_signals(self, data):
        """
        Generate ensemble signals by combining multiple strategies
        """
        # Get signals from individual strategies
        alpha_signals = self.alpha_strategy.generate_signals(data)
        gbdt_signals = self.gbdt_strategy.generate_signals(data)
        tech_signals = self.tech_strategy.generate_signals(data)
        
        # Combine signals (simple voting approach)
        combined_signals = pd.DataFrame({
            'alpha': alpha_signals,
            'gbdt': gbdt_signals,
            'tech': tech_signals
        })
        
        # Calculate ensemble signal as average of all strategies
        ensemble_signal = combined_signals.mean(axis=1)
        
        # Convert to discrete signals: 1 for buy, -1 for sell, 0 for hold
        signals = ensemble_signal.apply(lambda x: 1 if x > 0.3 else (-1 if x < -0.3 else 0))
        
        return signals

    def backtest(self, data, start_date=None, end_date=None):
        """
        Run backtest for Ensemble strategy
        """
        # Generate signals
        signals = self.generate_signals(data)
        
        # Simple backtest logic
        positions = signals.replace(0, np.nan).fillna(method='ffill').fillna(0)
        
        # Calculate returns
        if 'close' in data.columns:
            daily_returns = data['close'].pct_change()
            strategy_returns = daily_returns * positions.shift(1)
            
            # Calculate cumulative returns
            cumulative_returns = (1 + strategy_returns.fillna(0)).cumprod()
            
            # Calculate metrics
            total_return = cumulative_returns.iloc[-1] - 1
            annualized_return = (cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1 if len(cumulative_returns) > 0 else 0
            
            # Calculate max drawdown
            rolling_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
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


def get_qlib_strategy_manager():
    """
    Factory function to get a Qlib-based strategy manager
    """
    return QlibStrategyManager()


if __name__ == "__main__":
    # Example usage
    print("Initializing Qlib Strategy Manager...")
    manager = QlibStrategyManager()
    
    print(f"Available strategies: {manager.get_strategy_names()}")
    
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    
    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(100) * 0.01),
        'high': prices * (1 + abs(np.random.randn(100)) * 0.02),
        'low': prices * (1 - abs(np.random.randn(100)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)
    
    print("\nTesting Qlib strategies on sample data...")
    for strategy_name in manager.get_strategy_names():
        try:
            signals = manager.run_strategy(strategy_name, sample_data)
            print(f"{strategy_name}: Generated {len(signals[signals != 0])} non-zero signals")
            
            # Run backtest
            backtest_result = manager.run_backtest(strategy_name, sample_data)
            print(f"  Total Return: {backtest_result['total_return']:.4f}")
            print(f"  Sharpe Ratio: {backtest_result['sharpe_ratio']:.4f}")
            print(f"  Max Drawdown: {backtest_result['max_drawdown']:.4f}")
        except Exception as e:
            print(f"Error testing {strategy_name}: {e}")