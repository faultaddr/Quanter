"""
Data fetching module for A-Share market
Supports multiple data sources: EastMoney (cookie-based), tushare, baostock, yahoo finance
"""
import pandas as pd
import numpy as np
import tushare as ts
import baostock as bs
import yfinance as yf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import the new EastMoney data fetcher
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.eastmoney_data_fetcher import EastMoneyDataFetcher

class DataFetcher:
    def __init__(self):
        # Initialize data sources
        self.tushare_token = None  # Will need to be set by user
        self.bs_logged_in = False
        self.eastmoney_fetcher = EastMoneyDataFetcher()
        
    def fetch(self, symbol, start_date, end_date, source="eastmoney"):
        """
        Fetch historical data for a given symbol from specified source

        Args:
            symbol (str): Stock symbol (e.g., '000001' for Ping An, 'sh000001' for SSE Index)
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            source (str): Data source ('eastmoney', 'tushare', 'baostock', 'yahoo')

        Returns:
            pandas.DataFrame: Historical price data with columns [date, open, high, low, close, volume]
        """
        if source == "eastmoney":
            return self._fetch_eastmoney(symbol, start_date, end_date)
        elif source == "tushare":
            return self._fetch_tushare(symbol, start_date, end_date)
        elif source == "baostock":
            return self._fetch_baostock(symbol, start_date, end_date)
        elif source == "yahoo":
            return self._fetch_yahoo(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {source}")
    
    def _fetch_eastmoney(self, symbol, start_date, end_date):
        """Fetch data using EastMoney API with cookies"""
        try:
            # Convert symbol if needed
            # For A-shares, we typically use codes like '000001'
            if '.' not in symbol and len(symbol) == 6:
                # Determine if it's Shanghai or Shenzhen based on first digit
                if symbol.startswith(('5', '6')):
                    symbol = f'sh{symbol}'  # Shanghai
                else:
                    symbol = f'sz{symbol}'  # Shenzhen

            # Calculate number of days between start and end date
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end_dt - start_dt).days

            # Use EastMoney fetcher to get stock data
            df = self.eastmoney_fetcher.fetch_stock_data(symbol, days=max(days, 30))

            if df is not None and not df.empty:
                # Filter data to be within the requested date range
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                
                # Ensure required columns exist
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in required_cols:
                    if col not in df.columns:
                        print(f"Warning: Column '{col}' not found in EastMoney data")
                        return pd.DataFrame()

                return df
            else:
                print(f"Error: EastMoney fetcher returned no data for {symbol}")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching data from EastMoney: {e}")
            return pd.DataFrame()
    
    def _fetch_tushare(self, symbol, start_date, end_date):
        """Fetch data using Tushare"""
        if not self.tushare_token:
            print("Tushare token not set. Please set token using set_tushare_token().")
            return pd.DataFrame()
        
        try:
            ts.set_token(self.tushare_token)
            pro = ts.pro_api()
            
            # Format dates for tushare
            start = start_date.replace('-', '')
            end = end_date.replace('-', '')
            
            # Fetch daily data
            df = pro.daily(ts_code=symbol, start_date=start, end_date=end)
            
            # Rename columns to standard format
            df.rename(columns={
                'trade_date': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'vol': 'volume'
            }, inplace=True)
            
            # Convert date column
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Sort by date
            df.sort_index(inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching data from Tushare: {e}")
            return pd.DataFrame()
    
    def _fetch_baostock(self, symbol, start_date, end_date):
        """Fetch data using Baostock"""
        try:
            if not self.bs_logged_in:
                bs.login()
                self.bs_logged_in = True
            
            # Convert symbol if needed
            if '.' not in symbol:
                # Assume it's an A-share code
                if symbol.startswith(('5', '6')):
                    symbol = f'sh.{symbol}'  # Shanghai
                else:
                    symbol = f'sz.{symbol}'  # Shenzhen
            
            rs = bs.query_history_k_data_plus(
                symbol,
                "date,open,high,low,close,volume",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"
            )
            
            df = pd.DataFrame()
            while (rs.error_code == '0') & rs.next():
                df = pd.concat([df, rs.get_row_data()], axis=1)
            
            if not df.empty:
                df = df.T  # Transpose
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                
                # Convert data types
                df['date'] = pd.to_datetime(df['date'])
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                
                df.set_index('date', inplace=True)
            
            return df
        except Exception as e:
            print(f"Error fetching data from Baostock: {e}")
            return pd.DataFrame()
    
    def _fetch_yahoo(self, symbol, start_date, end_date):
        """Fetch data using Yahoo Finance"""
        try:
            # Convert A-share symbol to Yahoo format if needed
            # For Chinese stocks, Yahoo uses suffixes: SS for Shanghai, SZ for Shenzhen
            if '.' not in symbol:
                if symbol.startswith(('5', '6')):
                    symbol = f"{symbol}.SS"  # Shanghai
                else:
                    symbol = f"{symbol}.SZ"  # Shenzhen
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            # Rename columns to standard format if needed
            if 'Open' in df.columns:
                df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                }, inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
        except Exception as e:
            print(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    def set_tushare_token(self, token):
        """Set Tushare token for API access"""
        self.tushare_token = token
        print("Tushare token set successfully.")
    
    def close_connections(self):
        """Close connections to data sources"""
        if self.bs_logged_in:
            bs.logout()
            self.bs_logged_in = False