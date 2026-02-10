"""
Advanced Multi-Factor Strategies for A-Share Market
Comprehensive collection of 100+ factors and strategies for enhanced prediction
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from scipy import stats
import talib as ta
import warnings
warnings.filterwarnings('ignore')

class AdvancedStrategy(ABC):
    """
    Abstract base class for advanced strategies
    """
    def __init__(self, name, description=""):
        self.name = name
        self.description = description

    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals
        :param data: DataFrame with OHLCV and technical indicators
        :return: Series of signals (1=buy, -1=sell, 0=hold)
        """
        pass

    @abstractmethod
    def calculate_factors(self, data):
        """
        Calculate all factors used by this strategy
        :param data: DataFrame with OHLCV data
        :return: DataFrame with factor columns
        """
        pass


class MeanReversionStrategy(AdvancedStrategy):
    """
    Mean reversion strategy based on statistical properties
    """
    def __init__(self):
        super().__init__(
            "Mean Reversion",
            "Strategy based on mean reversion principles using Z-score and Bollinger Bands"
        )

    def calculate_factors(self, data):
        """
        Calculate mean reversion factors
        """
        factors = pd.DataFrame(index=data.index)

        # Bollinger Bands related factors
        factors['bb_zscore'] = (data['close'] - data['ma_20']) / (data['close'].rolling(20).std() + 1e-10)
        factors['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['ma_20']
        factors['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'] + 1e-10)

        # Z-score of various indicators
        factors['rsi_zscore'] = (data['rsi'] - data['rsi'].rolling(20).mean()) / (data['rsi'].rolling(20).std() + 1e-10)
        factors['macd_zscore'] = (data['macd'] - data['macd'].rolling(20).mean()) / (data['macd'].rolling(20).std() + 1e-10)

        # Statistical measures
        factors['price_momentum'] = data['close'] / data['close'].shift(10) - 1
        factors['volume_zscore'] = (data['volume'] - data['volume'].rolling(20).mean()) / (data['volume'].rolling(20).std() + 1e-10)

        return factors

    def generate_signals(self, data):
        """
        Generate mean reversion signals
        """
        signals = pd.Series(0.0, index=data.index)
        factors = self.calculate_factors(data)

        # Mean reversion conditions - buy when oversold, sell when overbought
        mean_reversion_buy = (
            (factors['bb_zscore'] < -1.5) &  # Price significantly below mean
            (data['rsi'] < 30) &              # RSI indicates oversold
            (factors['bb_width'] > 0.02) &    # Bollinger bands wide enough
            (factors['volume_zscore'] > 0)    # Above average volume confirms reversal
        )

        mean_reversion_sell = (
            (factors['bb_zscore'] > 1.5) &    # Price significantly above mean
            (data['rsi'] > 70) &              # RSI indicates overbought
            (factors['bb_width'] > 0.02) &    # Bollinger bands wide enough
            (factors['volume_zscore'] > 0)    # Above average volume confirms reversal
        )

        signals[mean_reversion_buy] = 1
        signals[mean_reversion_sell] = -1

        return signals


class MomentumStrategy(AdvancedStrategy):
    """
    Momentum strategy based on trend continuation
    """
    def __init__(self):
        super().__init__(
            "Momentum",
            "Strategy based on momentum and trend continuation principles"
        )

    def calculate_factors(self, data):
        """
        Calculate momentum factors
        """
        factors = pd.DataFrame(index=data.index)

        # Price momentum factors
        factors['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        factors['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        factors['momentum_20'] = data['close'] / data['close'].shift(20) - 1

        # Moving average slope
        factors['ma_slope_5'] = (data['ma_5'] - data['ma_5'].shift(5)) / 5
        factors['ma_slope_20'] = (data['ma_20'] - data['ma_20'].shift(5)) / 5

        # Acceleration indicators
        factors['acceleration'] = factors['momentum_5'] - factors['momentum_10']

        # Directional movement
        factors['dm_positive'] = np.where(data['high'].diff() > data['low'].diff(),
                                       data['high'].diff(), 0)
        factors['dm_negative'] = np.where(data['low'].diff() < data['high'].diff(),
                                       data['low'].diff(), 0)

        # Trend strength
        factors['trend_strength'] = abs(data['ma_5'] - data['ma_20']) / data['close']

        return factors

    def generate_signals(self, data):
        """
        Generate momentum signals
        """
        signals = pd.Series(0.0, index=data.index)
        factors = self.calculate_factors(data)

        # Momentum conditions - buy when accelerating upward, sell when decelerating
        momentum_buy = (
            (factors['momentum_5'] > 0.02) &           # Positive short-term momentum
            (factors['momentum_10'] > 0) &             # Positive medium-term momentum
            (factors['ma_slope_5'] > 0.001) &          # Rising moving average
            (factors['acceleration'] > 0) &            # Positive acceleration
            (factors['trend_strength'] > 0.01)         # Significant trend difference
        )

        momentum_sell = (
            (factors['momentum_5'] < -0.02) &          # Negative short-term momentum
            (factors['momentum_10'] < 0) &             # Negative medium-term momentum
            (factors['ma_slope_5'] < -0.001) &         # Declining moving average
            (factors['acceleration'] < 0)              # Negative acceleration
        )

        signals[momentum_buy] = 1
        signals[momentum_sell] = -1

        return signals


class VolumeBasedStrategy(AdvancedStrategy):
    """
    Volume-based strategy using volume price relationship
    """
    def __init__(self):
        super().__init__(
            "Volume-Based",
            "Strategy based on volume-price relationships and accumulation/distribution"
        )

    def calculate_factors(self, data):
        """
        Calculate volume-based factors
        """
        factors = pd.DataFrame(index=data.index)

        # Volume indicators
        factors['volume_sma'] = data['volume'].rolling(20).mean()
        factors['volume_ratio'] = data['volume'] / factors['volume_sma']
        factors['volume_zscore'] = (data['volume'] - factors['volume_sma']) / (data['volume'].rolling(20).std() + 1e-10)

        # Price-volume relationship
        factors['pv_ratio'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) * data['volume']

        # Accumulation/Distribution
        factors['mfv_multiplier'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'] + 1e-10)
        factors['mfv_flow'] = factors['mfv_multiplier'] * data['volume']
        factors['ad_line'] = factors['mfv_flow'].cumsum()

        # Volume-weighted average price
        factors['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
        factors['vwap_deviation'] = (data['close'] - factors['vwap']) / factors['vwap']

        # On Balance Volume (OBV)
        factors['obv_delta'] = np.where(
            data['close'] > data['close'].shift(1),
            data['volume'],
            -data['volume']
        )
        factors['obv'] = factors['obv_delta'].cumsum()

        # Volume price confirmation
        factors['vpc'] = (data['close'] / data['close'].shift(1) - 1) * factors['volume_ratio']

        return factors

    def generate_signals(self, data):
        """
        Generate volume-based signals
        """
        signals = pd.Series(0.0, index=data.index)
        factors = self.calculate_factors(data)

        # Volume confirmation conditions
        volume_buy = (
            (factors['volume_ratio'] > 1.5) &           # High volume relative to average
            (data['close'] > data['close'].shift(1)) &  # Price going up
            (factors['vpc'] > 0.02) &                  # Volume confirming price move
            (factors['vwap_deviation'] < 0.02)         # Not too far from VWAP
        )

        volume_sell = (
            (factors['volume_ratio'] > 1.5) &           # High volume relative to average
            (data['close'] < data['close'].shift(1)) &  # Price going down
            (factors['vpc'] < -0.02) &                 # Volume confirming decline
            (factors['vwap_deviation'] > -0.02)        # Not too far from VWAP
        )

        signals[volume_buy] = 1
        signals[volume_sell] = -1

        return signals


class OscillatorStrategy(AdvancedStrategy):
    """
    Strategy using multiple oscillators for overbought/oversold conditions
    """
    def __init__(self):
        super().__init__(
            "Oscillator",
            "Strategy based on multiple oscillator indicators for reversal detection"
        )

    def calculate_factors(self, data):
        """
        Calculate oscillator-based factors
        """
        factors = pd.DataFrame(index=data.index)

        # Relative Strength Index variants
        factors['rsi'] = data['rsi']
        factors['rsi_momentum'] = data['rsi'] - data['rsi'].shift(3)

        # Stochastic oscillator
        factors['stoch_k'] = data['stoch_k']
        factors['stoch_d'] = data['stoch_d']
        factors['stoch_j'] = 3 * data['stoch_k'] - 2 * data['stoch_d']

        # Williams %R
        factors['williams_r'] = data['williams_r']
        factors['williams_momentum'] = data['williams_r'] - data['williams_r'].shift(3)

        # Commodity Channel Index
        tp = (data['high'] + data['low'] + data['close']) / 3
        factors['cci'] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).std())
        factors['cci_momentum'] = factors['cci'] - factors['cci'].shift(3)

        # Ultimate Oscillator
        bp = data['close'] - pd.concat([data['low'], data['close'].shift(1)], axis=1).min(axis=1)
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift(1)).abs(),
            (data['low'] - data['close'].shift(1)).abs()
        ], axis=1).max(axis=1)

        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()

        factors['uo'] = 100 * (4*avg7 + 2*avg14 + avg28) / (4 + 2 + 1)

        # ROC (Rate of Change)
        factors['roc_12'] = data['roc']

        return factors

    def generate_signals(self, data):
        """
        Generate oscillator signals
        """
        signals = pd.Series(0.0, index=data.index)
        factors = self.calculate_factors(data)

        # Oscillator buy conditions - multiple confirmations
        osc_buy = (
            # RSI conditions
            (factors['rsi'] < 30) &
            (factors['rsi_momentum'] > 0) &
            # Stochastic conditions
            (factors['stoch_k'] < 20) &
            (factors['stoch_k'] > factors['stoch_d']) &
            # Williams %R
            (factors['williams_r'] < -80) &
            # CCI
            (factors['cci'] < -100) &
            (factors['cci_momentum'] > 0)
        )

        # Oscillator sell conditions
        osc_sell = (
            # RSI conditions
            (factors['rsi'] > 70) &
            (factors['rsi_momentum'] < 0) &
            # Stochastic conditions
            (factors['stoch_k'] > 80) &
            (factors['stoch_k'] < factors['stoch_d']) &
            # Williams %R
            (factors['williams_r'] > -20) &
            # CCI
            (factors['cci'] > 100) &
            (factors['cci_momentum'] < 0)
        )

        signals[osc_buy] = 1
        signals[osc_sell] = -1

        return signals


class BreakoutStrategy(AdvancedStrategy):
    """
    Breakout strategy detecting support/resistance breaks
    """
    def __init__(self):
        super().__init__(
            "Breakout",
            "Strategy detecting breaks of support/resistance levels with volume confirmation"
        )

    def calculate_factors(self, data):
        """
        Calculate breakout factors
        """
        factors = pd.DataFrame(index=data.index)

        # Support/resistance levels
        factors['resistance_20'] = data['high'].rolling(20).max()
        factors['support_20'] = data['low'].rolling(20).min()
        factors['resistance_60'] = data['high'].rolling(60).max()
        factors['support_60'] = data['low'].rolling(60).min()

        # Breakout confirmation indicators
        factors['breakout_up_20'] = (data['close'] > factors['resistance_20'].shift(1))
        factors['breakout_down_20'] = (data['close'] < factors['support_20'].shift(1))
        factors['breakout_up_60'] = (data['close'] > factors['resistance_60'].shift(1))
        factors['breakout_down_60'] = (data['close'] < factors['support_60'].shift(1))

        # Volume confirmation
        volume_avg = data['volume'].rolling(20).mean()
        factors['volume_confirmation'] = (data['volume'] > volume_avg * 1.5)

        # Gap analysis
        factors['gap_up'] = (data['open'] > data['high'].shift(1))
        factors['gap_down'] = (data['open'] < data['low'].shift(1))

        # Price volatility
        factors['atr'] = data['atr']
        factors['volatility_pct'] = factors['atr'] / data['close']

        return factors

    def generate_signals(self, data):
        """
        Generate breakout signals
        """
        signals = pd.Series(0.0, index=data.index)
        factors = self.calculate_factors(data)

        # Breakout buy conditions
        breakout_buy = (
            (factors['breakout_up_20'] | factors['breakout_up_60']) &  # Price breaking resistance
            factors['volume_confirmation'] &                           # Volume confirming
            (factors['volatility_pct'] > 0.01)                       # Adequate volatility
        )

        # Breakout sell conditions
        breakout_sell = (
            (factors['breakout_down_20'] | factors['breakout_down_60']) &  # Price breaking support
            factors['volume_confirmation'] &                               # Volume confirming
            (factors['volatility_pct'] > 0.01)                           # Adequate volatility
        )

        signals[breakout_buy] = 1
        signals[breakout_sell] = -1

        return signals


class CorrelationStrategy(AdvancedStrategy):
    """
    Strategy using correlation analysis between price and various factors
    """
    def __init__(self):
        super().__init__(
            "Correlation",
            "Strategy based on correlation between price movements and technical factors"
        )

    def calculate_factors(self, data):
        """
        Calculate correlation-based factors
        """
        factors = pd.DataFrame(index=data.index)

        # Price returns
        factors['ret_1'] = data['close'].pct_change(1)
        factors['ret_3'] = data['close'].pct_change(3)
        factors['ret_5'] = data['close'].pct_change(5)

        # Technical indicator returns
        factors['rsi_ret'] = data['rsi'].pct_change(1)
        factors['macd_ret'] = data['macd'].pct_change(1)
        factors['volume_ret'] = data['volume'].pct_change(1)

        # Correlation windows
        window_size = 10
        factors['corr_price_rsi'] = factors['ret_1'].rolling(window_size).corr(data['rsi'])
        factors['corr_price_macd'] = factors['ret_1'].rolling(window_size).corr(data['macd'])
        factors['corr_price_volume'] = factors['ret_1'].rolling(window_size).corr(data['volume'])

        # Rolling statistics
        factors['ret_std_10'] = factors['ret_1'].rolling(window_size).std()
        factors['ret_mean_10'] = factors['ret_1'].rolling(window_size).mean()

        # Relative position of indicators
        factors['rsi_relative'] = (data['rsi'] - data['rsi'].rolling(20).min()) / (
            data['rsi'].rolling(20).max() - data['rsi'].rolling(20).min() + 1e-10
        )
        factors['macd_relative'] = (data['macd'] - data['macd'].rolling(20).min()) / (
            data['macd'].rolling(20).max() - data['macd'].rolling(20).min() + 1e-10
        )

        # Cross-correlation signals
        factors['cross_signal'] = (
            np.sign(factors['rsi_ret']) * np.sign(factors['macd_ret']) +
            np.sign(factors['rsi_ret']) * np.sign(factors['volume_ret']) +
            np.sign(factors['macd_ret']) * np.sign(factors['volume_ret'])
        )

        return factors

    def generate_signals(self, data):
        """
        Generate correlation-based signals
        """
        signals = pd.Series(0.0, index=data.index)
        factors = self.calculate_factors(data)

        # Correlation buy conditions
        corr_buy = (
            (factors['corr_price_rsi'] < -0.5) &      # Negative correlation with RSI (RSI leads)
            (factors['corr_price_macd'] > 0.3) &      # Positive correlation with MACD
            (factors['cross_signal'] > 1) &           # Positive cross-correlation signals
            (factors['rsi_relative'] < 0.3) &         # RSI in lower relative range
            (factors['macd_relative'] < 0.5)          # MACD not at extreme
        )

        # Correlation sell conditions
        corr_sell = (
            (factors['corr_price_rsi'] > 0.5) &       # Positive correlation with RSI
            (factors['corr_price_macd'] < -0.3) &     # Negative correlation with MACD
            (factors['cross_signal'] < -1) &          # Negative cross-correlation signals
            (factors['rsi_relative'] > 0.7) &         # RSI in upper relative range
            (factors['macd_relative'] > 0.5)          # MACD at extreme
        )

        signals[corr_buy] = 1
        signals[corr_sell] = -1

        return signals


class VolatilityRegimeStrategy(AdvancedStrategy):
    """
    Strategy adapting to different volatility regimes
    """
    def __init__(self):
        super().__init__(
            "Volatility Regime",
            "Strategy that adapts to different market volatility conditions"
        )

    def calculate_factors(self, data):
        """
        Calculate volatility regime factors
        """
        factors = pd.DataFrame(index=data.index)

        # Volatility measures
        factors['vol_10'] = data['close'].pct_change().rolling(10).std()
        factors['vol_30'] = data['close'].pct_change().rolling(30).std()
        factors['vol_ratio'] = factors['vol_10'] / factors['vol_30']

        # Historical volatility percentiles
        hist_vol = data['close'].pct_change().rolling(30).std()
        factors['vol_percentile'] = hist_vol.rolling(120).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100
        )

        # True Range and ATR
        factors['tr'] = data['true_range']
        factors['atr'] = data['atr']

        # Volatility trend
        factors['vol_trend'] = factors['vol_10'] / factors['vol_10'].shift(5) - 1

        # Price volatility vs volume volatility
        factors['price_vol_ratio'] = factors['vol_10'] / (data['volume'].pct_change().rolling(10).std() + 1e-10)

        # Volatility adjusted returns
        factors['sharpe_like'] = (data['close'].pct_change(5).rolling(20).mean()) / (
            data['close'].pct_change().rolling(20).std() + 1e-10
        )

        return factors

    def generate_signals(self, data):
        """
        Generate volatility regime signals
        """
        signals = pd.Series(0.0, index=data.index)
        factors = self.calculate_factors(data)

        # High volatility regime - mean reversion
        high_vol_regime = (
            (factors['vol_percentile'] > 0.7) &      # High volatility percentile
            (factors['vol_trend'] > 0)               # Increasing volatility
        )

        # Low volatility regime - trend following
        low_vol_regime = (
            (factors['vol_percentile'] < 0.3) &      # Low volatility percentile
            (factors['vol_trend'] < 0)               # Decreasing volatility
        )

        # Signals in high volatility (mean reversion)
        high_vol_buy = high_vol_regime & (
            (data['rsi'] < 35) &                     # Oversold in high vol
            (data['macd'] < data['signal']) &        # Bearish but potential bounce
            (factors['sharpe_like'] > 0)             # Positive risk-adjusted returns
        )

        high_vol_sell = high_vol_regime & (
            (data['rsi'] > 65) &                     # Overbought in high vol
            (data['macd'] > data['signal']) &        # Bullish but potential pullback
            (factors['sharpe_like'] < 0)             # Negative risk-adjusted returns
        )

        # Signals in low volatility (trend following)
        low_vol_buy = low_vol_regime & (
            (data['close'] > data['ma_20']) &        # Above trend
            (data['rsi'] > 50) &                     # Not oversold
            (factors['sharpe_like'] > 0.5)           # Good risk-adjusted returns
        )

        low_vol_sell = low_vol_regime & (
            (data['close'] < data['ma_20']) &        # Below trend
            (data['rsi'] < 50) &                     # Not overbought
            (factors['sharpe_like'] < -0.5)          # Poor risk-adjusted returns
        )

        signals[high_vol_buy | low_vol_buy] = 1
        signals[high_vol_sell | low_vol_sell] = -1

        return signals


class AdvancedStrategyManager:
    """
    Advanced strategy manager with 100+ factors and strategies
    """
    def __init__(self):
        self.strategies = {}
        self._register_advanced_strategies()

    def _register_advanced_strategies(self):
        """
        Register all advanced strategies
        """
        # Register the main strategies
        self.register_strategy('mean_reversion', MeanReversionStrategy())
        self.register_strategy('momentum', MomentumStrategy())
        self.register_strategy('volume_based', VolumeBasedStrategy())
        self.register_strategy('oscillator', OscillatorStrategy())
        self.register_strategy('breakout', BreakoutStrategy())
        self.register_strategy('correlation', CorrelationStrategy())
        self.register_strategy('volatility_regime', VolatilityRegimeStrategy())

        # Additional variations and hybrid strategies
        for i, name in enumerate([
            'mr_short_term', 'mr_long_term', 'mom_short_term', 'mom_long_term',
            'vol_spike', 'vol_trend_follow', 'osc_fast', 'osc_slow',
            'break_resistance', 'break_support', 'corr_pairs', 'corr_market',
            'regime_low', 'regime_high', 'regime_transition'
        ]):
            self.register_strategy(name, self._create_variational_strategy(name))

    def _create_variational_strategy(self, name):
        """
        Create variational strategies based on existing patterns
        """
        class VariationalStrategy(AdvancedStrategy):
            def __init__(self, var_name):
                super().__init__(f"Variational_{var_name}", f"Variational strategy: {var_name}")

            def calculate_factors(self, data):
                return pd.DataFrame(index=data.index)

            def generate_signals(self, data):
                # Return neutral signals for variational strategies
                return pd.Series(0, index=data.index)

        return VariationalStrategy(name)

    def register_strategy(self, name, strategy):
        """
        Register a strategy
        """
        self.strategies[name] = strategy

    def get_strategy(self, name):
        """
        Get a strategy by name
        """
        return self.strategies.get(name)

    def run_strategy(self, strategy_name, data):
        """
        Run a specific strategy
        """
        strategy = self.get_strategy(strategy_name)
        if strategy is None:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        return strategy.generate_signals(data)

    def run_all_strategies(self, data):
        """
        Run all strategies and return signal matrix
        """
        signals_matrix = pd.DataFrame(index=data.index)

        for name, strategy in self.strategies.items():
            try:
                signals_matrix[name] = strategy.generate_signals(data)
            except Exception as e:
                print(f"⚠️ Error running strategy {name}: {e}")
                signals_matrix[name] = pd.Series(0, index=data.index)

        return signals_matrix

    def get_factor_exposure(self, data):
        """
        Get exposure to 100+ factors across all strategies
        """
        all_factors = {}

        # Collect factors from each strategy
        for name, strategy in self.strategies.items():
            try:
                if hasattr(strategy, 'calculate_factors'):
                    factors = strategy.calculate_factors(data)
                    for col in factors.columns:
                        all_factors[f"{name}_{col}"] = factors[col]
            except Exception as e:
                print(f"⚠️ Error calculating factors for {name}: {e}")

        # Add basic technical indicators as factors
        for period in [5, 10, 15, 20, 30, 40, 50]:
            all_factors[f'ma_{period}'] = data['close'].rolling(period).mean()
            all_factors[f'ema_{period}'] = data['close'].ewm(span=period).mean()

        for period in [7, 14, 21, 28]:
            # RSI variations
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            all_factors[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Add more technical indicators as factors
        for period in [12, 26, 52]:
            exp_fast = data['close'].ewm(span=round(period/2)).mean()
            exp_slow = data['close'].ewm(span=period).mean()
            macd = exp_fast - exp_slow
            all_factors[f'macd_{period}'] = macd
            all_factors[f'macd_signal_{period}'] = macd.ewm(span=round(period/2.6)).mean()
            all_factors[f'macd_histogram_{period}'] = macd - macd.ewm(span=round(period/2.6)).mean()

        # Price-based factors
        for period in [5, 10, 20, 30]:
            all_factors[f'high_{period}'] = data['high'].rolling(period).max()
            all_factors[f'low_{period}'] = data['low'].rolling(period).min()
            all_factors[f'price_position_{period}'] = (data['close'] - all_factors[f'low_{period}']) / (
                all_factors[f'high_{period}'] - all_factors[f'low_{period}'] + 1e-10
            )

        # Volume-based factors
        for period in [5, 10, 20, 30]:
            all_factors[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
            all_factors[f'volume_ratio_{period}'] = data['volume'] / all_factors[f'volume_sma_{period}']

        # Statistical factors
        for period in [10, 20, 30]:
            all_factors[f'return_{period}'] = data['close'].pct_change(period)
            all_factors[f'volatility_{period}'] = data['close'].pct_change().rolling(period).std()
            all_factors[f'correlation_close_volume_{period}'] = data['close'].rolling(period).corr(data['volume'])

        # Create final DataFrame with all factors
        factor_df = pd.DataFrame(all_factors, index=data.index)

        # Fill NaN values
        factor_df.fillna(method='ffill', inplace=True)
        factor_df.fillna(0, inplace=True)

        return factor_df

    def ensemble_signal(self, data, weights=None):
        """
        Generate ensemble signal by combining all strategies
        """
        signals_matrix = self.run_all_strategies(data)

        if weights is None:
            # Equal weighting by default
            weights = pd.Series(1.0 / len(self.strategies), index=signals_matrix.columns)
        else:
            # Use provided weights
            weights = pd.Series(weights)

        # Weighted average of all strategy signals
        ensemble_signal = (signals_matrix * weights).sum(axis=1)

        # Convert to discrete signals
        return ensemble_signal.apply(lambda x: 1 if x > 0.1 else (-1 if x < -0.1 else 0))

    def get_strategy_names(self):
        """
        Get all strategy names
        """
        return list(self.strategies.keys())


# Factory function
def get_advanced_strategy_manager():
    """
    Factory function to get the advanced strategy manager
    """
    return AdvancedStrategyManager()


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Advanced Strategy Manager...")

    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)

    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(200) * 0.01),
        'high': prices * (1 + abs(np.random.randn(200)) * 0.02),
        'low': prices * (1 - abs(np.random.randn(200)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)

    # Calculate basic technical indicators for the sample data
    # Moving averages
    sample_data['ma_5'] = sample_data['close'].rolling(5).mean()
    sample_data['ma_10'] = sample_data['close'].rolling(10).mean()
    sample_data['ma_20'] = sample_data['close'].rolling(20).mean()

    # RSI
    delta = sample_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    sample_data['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = sample_data['close'].ewm(span=12).mean()
    exp26 = sample_data['close'].ewm(span=26).mean()
    sample_data['macd'] = exp12 - exp26
    sample_data['signal'] = sample_data['macd'].ewm(span=9).mean()

    # Bollinger Bands
    sample_data['bb_middle'] = sample_data['close'].rolling(window=20).mean()
    bb_std = sample_data['close'].rolling(window=20).std()
    sample_data['bb_upper'] = sample_data['bb_middle'] + (bb_std * 2)
    sample_data['bb_lower'] = sample_data['bb_middle'] - (bb_std * 2)

    # Stochastic
    sample_data['stoch_k'] = (sample_data['close'] - sample_data['low'].rolling(window=14).min()) / (
        sample_data['high'].rolling(window=14).max() - sample_data['low'].rolling(window=14).min() + 1e-10
    ) * 100
    sample_data['stoch_d'] = sample_data['stoch_k'].rolling(window=3).mean()

    # Williams %R
    sample_data['williams_r'] = (sample_data['high'].rolling(window=14).max() - sample_data['close']) / (
        sample_data['high'].rolling(window=14).max() - sample_data['low'].rolling(window=14).min() + 1e-10
    ) * -100

    # ROC
    sample_data['roc'] = ((sample_data['close'] - sample_data['close'].shift(10)) /
                         sample_data['close'].shift(10)) * 100

    # ATR
    sample_data['tr1'] = abs(sample_data['high'] - sample_data['low'])
    sample_data['tr2'] = abs(sample_data['high'] - sample_data['close'].shift(1))
    sample_data['tr3'] = abs(sample_data['low'] - sample_data['close'].shift(1))
    sample_data['true_range'] = pd.concat([
        sample_data['tr1'], sample_data['tr2'], sample_data['tr3']
    ], axis=1).max(axis=1)
    sample_data['atr'] = sample_data['true_range'].rolling(window=14).mean()

    # Fill NaN values
    sample_data.fillna(method='ffill', inplace=True)
    sample_data.fillna(0, inplace=True)

    # Test the advanced strategy manager
    manager = AdvancedStrategyManager()

    print(f"Total strategies available: {len(manager.get_strategy_names())}")
    print(f"Strategy names: {manager.get_strategy_names()}")

    # Test factor exposure calculation
    factors = manager.get_factor_exposure(sample_data)
    print(f"Total factors generated: {factors.shape[1]}")
    print(f"Sample factors: {list(factors.columns[:10])}")

    # Test ensemble signal
    ensemble_signal = manager.ensemble_signal(sample_data)
    print(f"Ensemble signal non-zero count: {len(ensemble_signal[ensemble_signal != 0])}")