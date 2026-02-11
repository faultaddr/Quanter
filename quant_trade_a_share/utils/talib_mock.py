"""
Mock TA-Lib module to provide compatibility when TA-Lib is not available
"""
import numpy as np
import pandas as pd


def _validate_arrays(*arrays):
    """Validate that all arrays have the same length and are not empty."""
    if not arrays:
        return True
    lengths = [len(arr) for arr in arrays if arr is not None]
    if not lengths:
        return True
    if len(set(lengths)) > 1:
        raise ValueError("All arrays must have the same length")
    return lengths[0] > 0


def SMA(real, timeperiod=30):
    """Simple Moving Average"""
    if not _validate_arrays(real):
        return np.array([])
    return pd.Series(real).rolling(window=timeperiod).mean().values


def EMA(real, timeperiod=30):
    """Exponential Moving Average"""
    if not _validate_arrays(real):
        return np.array([])
    return pd.Series(real).ewm(span=timeperiod).mean().values


def RSI(real, timeperiod=14):
    """Relative Strength Index"""
    if not _validate_arrays(real):
        return np.array([])
    series = pd.Series(real)
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=timeperiod).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=timeperiod).mean()
    rs = gain / loss
    return (100 - 100 / (1 + rs)).values


def MACD(real, fastperiod=12, slowperiod=26, signalperiod=9):
    """Moving Average Convergence Divergence"""
    if not _validate_arrays(real):
        return np.array([]), np.array([]), np.array([])

    series = pd.Series(real)
    exp1 = series.ewm(span=fastperiod).mean()
    exp2 = series.ewm(span=slowperiod).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signalperiod).mean()
    hist = macd - signal

    return macd.values, signal.values, hist.values


def BBANDS(real, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    """Bollinger Bands"""
    if not _validate_arrays(real):
        return np.array([]), np.array([]), np.array([])

    series = pd.Series(real)
    middle = series.rolling(window=timeperiod).mean()
    std = series.rolling(window=timeperiod).std()
    upper = middle + (std * nbdevup)
    lower = middle - (std * nbdevdn)

    return upper.values, middle.values, lower.values


def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator"""
    if not _validate_arrays(high, low, close):
        return np.array([]), np.array([])

    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)

    lowest_low = low_series.rolling(window=fastk_period).min()
    highest_high = high_series.rolling(window=fastk_period).max()

    stoch_k = 100 * (close_series - lowest_low) / (highest_high - lowest_low + 1e-10)
    stoch_d = stoch_k.rolling(window=slowk_period).mean()

    return stoch_k.values, stoch_d.values


def ATR(high, low, close, timeperiod=14):
    """Average True Range"""
    if not _validate_arrays(high, low, close):
        return np.array([])

    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)

    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=timeperiod).mean()

    return atr.values


def ADX(high, low, close, timeperiod=14):
    """Average Directional Movement Index"""
    if not _validate_arrays(high, low, close):
        return np.array([])

    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)

    # Calculate True Range
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calculate directional movements
    plus_dm = high_series.diff()
    minus_dm = -low_series.diff()

    # Apply filters for DM
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # Smooth the values
    tr_smooth = tr.rolling(window=timeperiod).sum()
    plus_di = 100 * (plus_dm.rolling(window=timeperiod).sum() / tr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=timeperiod).sum() / tr_smooth)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(window=timeperiod).mean()

    return adx.values


def WILLR(high, low, close, timeperiod=14):
    """Williams' %R"""
    if not _validate_arrays(high, low, close):
        return np.array([])

    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)

    highest_high = high_series.rolling(window=timeperiod).max()
    lowest_low = low_series.rolling(window=timeperiod).min()

    willr = 100 * (highest_high - close_series) / (highest_high - lowest_low + 1e-10)

    return willr.values


def ROC(real, timeperiod=10):
    """Rate of Change"""
    if not _validate_arrays(real):
        return np.array([])

    series = pd.Series(real)
    roc = 100 * (series / series.shift(timeperiod) - 1)

    return roc.values


def CCI(high, low, close, timeperiod=14):
    """Commodity Channel Index"""
    if not _validate_arrays(high, low, close):
        return np.array([])

    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)

    typical_price = (high_series + low_series + close_series) / 3
    ma_tp = typical_price.rolling(window=timeperiod).mean()
    mean_dev = typical_price.rolling(window=timeperiod).apply(lambda x: abs(x - x.mean()).mean())

    cci = (typical_price - ma_tp) / (0.015 * mean_dev)

    return cci.values


# Constants for function lookbacks and other TA-Lib-like constants
class Func:
    pass


def CDLHAMMER(open, high, low, close):
    """Hammer candle pattern"""
    if not _validate_arrays(open, high, low, close):
        return np.array([])

    # Simplified hammer detection: long lower shadow, small body, little or no upper shadow
    body = abs(close - open)
    upper_shadow = high - np.maximum(close, open)
    lower_shadow = np.minimum(close, open) - low

    hammer = np.where(
        (lower_shadow > 2 * body) &
        (upper_shadow < body) &
        (body > 0),
        100,  # TA-Lib convention for bullish patterns
        0
    )

    return hammer


def CDLENGULFING(open, high, low, close):
    """Engulfing candle pattern"""
    if not _validate_arrays(open, high, low, close):
        return np.array([])

    prev_open = np.roll(open, 1)
    prev_close = np.roll(close, 1)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)

    # Current candle engulfs previous candle's body
    bullish_engulf = (close > open) & (prev_close < prev_open) & (close > prev_open) & (open < prev_close)
    bearish_engulf = (close < open) & (prev_close > prev_open) & (close < prev_open) & (open > prev_close)

    engulfing = np.where(bullish_engulf, 100, np.where(bearish_engulf, -100, 0))
    engulfing[0] = 0  # First value is undefined due to roll

    return engulfing


def MOM(real, timeperiod=10):
    """Momentum indicator"""
    if not _validate_arrays(real):
        return np.array([])

    series = pd.Series(real)
    mom = series - series.shift(timeperiod)
    return mom.values


def STDDEV(real, timeperiod=5, nbdev=1):
    """Standard deviation"""
    if not _validate_arrays(real):
        return np.array([])

    series = pd.Series(real)
    std = series.rolling(window=timeperiod).std()
    return std.values


def VAR(real, timeperiod=5, nbdev=1):
    """Variance"""
    if not _validate_arrays(real):
        return np.array([])

    series = pd.Series(real)
    var = series.rolling(window=timeperiod).var()
    return var.values


def MIN(real, timeperiod=30):
    """Lowest value over a specified period"""
    if not _validate_arrays(real):
        return np.array([])

    series = pd.Series(real)
    return series.rolling(window=timeperiod).min().values


def MAX(real, timeperiod=30):
    """Highest value over a specified period"""
    if not _validate_arrays(real):
        return np.array([])

    series = pd.Series(real)
    return series.rolling(window=timeperiod).max().values


def MINMAX(real, timeperiod=30):
    """Lowest and highest values over a specified period"""
    if not _validate_arrays(real):
        return np.array([]), np.array([])

    series = pd.Series(real)
    min_vals = series.rolling(window=timeperiod).min().values
    max_vals = series.rolling(window=timeperiod).max().values
    return min_vals, max_vals


# Mock abstract functions
def abstract(*args, **kwargs):
    """Placeholder for abstract API"""
    pass


def get_function_groups():
    """Return function groups like TA-Lib"""
    return {
        'overlap': ['SMA', 'EMA', 'BBANDS'],
        'momentum': ['RSI', 'MACD', 'STOCH', 'WILLR', 'ROC', 'CCI'],
        'volume': [],
        'price': [],
        'volatility': ['ATR'],
        'pattern': ['CDLHAMMER', 'CDLENGULFING'],
        'math': [],
        'operators': []
    }


def get_function_names(group=None):
    """Return function names, optionally filtered by group"""
    if group:
        groups = get_function_groups()
        return groups.get(group, [])
    else:
        # Return all function names that are implemented
        return [
            'SMA', 'EMA', 'RSI', 'MACD', 'BBANDS', 'STOCH', 'ATR',
            'ADX', 'WILLR', 'ROC', 'CCI', 'CDLHAMMER', 'CDLENGULFING',
            'MOM', 'STDDEV', 'VAR', 'MIN', 'MAX', 'MINMAX'
        ]


# Constants that TA-Lib defines
TA_FUNC_UNST_NONE = 0
TA_FUNC_UNST_ALL = 0xFFFFFFFF