"""Feature engineering for ML models."""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from ..core.logging import get_logger


logger = get_logger(__name__)


class FeatureEngineer:
    """Engineer features from price data for ML models."""

    def __init__(self, target_horizon: int = 5):
        """Initialize the feature engineer.

        Args:
            target_horizon: Number of periods ahead to predict
        """
        self.target_horizon = target_horizon
        self.feature_columns = []

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features from price data.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with features
        """
        data = df.copy()

        # Price-based features
        data = self._add_price_features(data)

        # Technical indicators
        data = self._add_momentum_features(data)
        data = self._add_volatility_features(data)
        data = self._add_volume_features(data)
        data = self._add_trend_features(data)

        # Target variable
        data = self._add_target(data)

        self.feature_columns = [c for c in data.columns if c not in ['timestamp', 'symbol', 'timeframe', 'target']]

        return data

    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_1d'] = df['returns']
        df['returns_3d'] = df['close'].pct_change(3)
        df['returns_5d'] = df['close'].pct_change(5)
        df['returns_10d'] = df['close'].pct_change(10)

        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price position within daily range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Body size (for candlestick patterns)
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Gap features
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicator features."""
        close = df['close']

        # RSI
        for period in [6, 12, 24]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD
        ema_12 = close.ewm(span=12).mean()
        ema_26 = close.ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Rate of Change
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period)

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = close - close.shift(period)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicator features."""
        returns = df['close'].pct_change()

        # Standard deviation of returns
        for period in [5, 10, 20]:
            df[f'volatility_{period}d'] = returns.rolling(period).std()

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()
        df['atr_ratio'] = df['atr_14'] / df['close']

        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / ((df['bb_upper'] - df['bb_lower']) + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        volume = df['volume']

        # Volume moving averages
        df['volume_sma_5'] = volume.rolling(5).mean()
        df['volume_sma_20'] = volume.rolling(20).mean()
        df['volume_ratio'] = volume / df['volume_sma_20']

        # Volume change
        df['volume_change'] = volume.pct_change()

        # Price-volume relationship
        df['price_volume_trend'] = (df['close'].pct_change() * volume).cumsum()

        # OBV (On Balance Volume)
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv

        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * volume
        positive_flow = pd.Series(np.where(typical_price > typical_price.shift(1), money_flow, 0))
        negative_flow = pd.Series(np.where(typical_price < typical_price.shift(1), money_flow, 0))
        positive_mf = positive_flow.rolling(14).sum()
        negative_mf = negative_flow.rolling(14).sum()
        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        df['mfi'] = mfi

        return df

    def _add_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicator features."""
        close = df['close']

        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            df[f'ma_{period}'] = close.rolling(period).mean()
            df[f'ma_ratio_{period}'] = close / df[f'ma_{period}']

        # Moving average crossovers
        df['ma_10_20_cross'] = (df['ma_10'] > df['ma_20']).astype(int)
        df['ma_20_50_cross'] = (df['ma_20'] > df['ma_50']).astype(int)

        # EMAs
        for period in [12, 26]:
            df[f'ema_{period}'] = close.ewm(span=period).mean()

        # ADX (Average Directional Index)
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)) * 100
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add target variable for prediction."""
        # Future returns
        future_return = df['close'].shift(-self.target_horizon) / df['close'] - 1

        # Binary target: 1 if positive return, 0 otherwise
        df['target'] = (future_return > 0).astype(int)

        # Also store the actual return for reference
        df['future_return'] = future_return

        return df

    def prepare_train_test_split(
        self,
        df: pd.DataFrame,
        test_size: float = 0.2,
        drop_na: bool = True
    ) -> tuple:
        """Prepare train/test split from feature DataFrame.

        Args:
            df: DataFrame with features
            test_size: Fraction for test set
            drop_na: Whether to drop NaN values

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if drop_na:
            df = df.dropna()

        if not self.feature_columns:
            self.feature_columns = [c for c in df.columns
                                    if c not in ['timestamp', 'symbol', 'timeframe', 'target', 'future_return']]

        X = df[self.feature_columns]
        y = df['target']

        # Chronological split
        split_idx = int(len(X) * (1 - test_size))

        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        return X_train, X_test, y_train, y_test

    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_columns
