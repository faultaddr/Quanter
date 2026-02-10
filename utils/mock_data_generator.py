"""
Mock data generator for testing the quantitative trading tool
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_mock_data(symbol='000001', start_date='2022-01-01', end_date='2022-12-31', seed=42):
    """
    Generate mock stock data for testing purposes
    
    Args:
        symbol (str): Stock symbol
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        seed (int): Random seed for reproducibility
    
    Returns:
        pandas.DataFrame: Mock stock data with columns [date, open, high, low, close, volume]
    """
    np.random.seed(seed)
    
    # Parse dates
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Generate date range (skip weekends)
    dates = pd.date_range(start=start, end=end, freq='D')
    # Keep only business days
    dates = dates[dates.weekday < 5]
    
    n = len(dates)
    
    # Generate mock data
    initial_price = 10.0  # Starting price
    prices = [initial_price]
    
    # Simulate price movement with some trend and volatility
    for i in range(1, n):
        # Random walk with slight upward bias
        change_percent = np.random.normal(0.001, 0.02)  # Mean 0.1%, std 2%
        new_price = prices[-1] * (1 + change_percent)
        prices.append(max(new_price, 0.1))  # Prevent negative prices
    
    # Convert to numpy array for vectorized operations
    prices = np.array(prices)
    
    # Add some random volatility to create OHLC data
    # Open is previous day's close (with some variation)
    opens = [prices[0]] + list(prices[:-1] * np.random.uniform(0.99, 1.01, n-1))
    
    # High, Low, Close with realistic relationships
    high_mult = np.random.uniform(1.0, 1.03, n)
    low_mult = np.random.uniform(0.97, 1.0, n)
    
    highs = prices * high_mult
    lows = np.maximum(prices * low_mult, 0.1)  # Ensure positive values
    closes = prices
    
    # Adjust opens to be within high-low range
    opens = np.clip(opens, lows, highs)
    
    # Generate volume (random but with some correlation to price movement)
    price_changes = np.abs(np.diff(closes, prepend=closes[0]))
    base_volume = 1000000  # Base volume
    volumes = base_volume + (price_changes * 100000)
    volumes = volumes.astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Set date as index
    df.set_index('date', inplace=True)
    
    return df


if __name__ == "__main__":
    # Generate and display sample data
    mock_data = generate_mock_data()
    print("Generated mock stock data:")
    print(mock_data.head(10))
    print(f"\nData shape: {mock_data.shape}")
    print(f"Date range: {mock_data.index[0]} to {mock_data.index[-1]}")
    print(f"Price range: {mock_data['close'].min():.2f} to {mock_data['close'].max():.2f}")