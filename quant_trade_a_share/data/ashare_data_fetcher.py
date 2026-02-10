#-*- coding:utf-8 -*-
"""
Ashare Data Fetcher Module
Integrates the Ashare stock data source into the A-Share market analysis system
Original Ashare source: https://github.com/mpquant/Ashare
"""

import json
import requests
import datetime
import pandas as pd
from typing import Optional, Union
from datetime import datetime as dt
import time


class AshareDataFetcher:
    """
    Data fetcher class that implements the Ashare stock data source
    Supports both Tencent and Sina data sources with fallback mechanisms
    """

    def __init__(self):
        """
        Initialize the Ashare data fetcher
        """
        print("✅ Ashare数据源加载成功")

    def fetch_stock_data(self, symbol: str, days: int = 60, frequency: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch stock data using Ashare implementation

        Args:
            symbol: Stock symbol (e.g., 'sh600023', '000001.XSHG')
            days: Number of days of historical data to fetch
            frequency: Data frequency ('1d' for daily, '1m', '5m', '15m', '30m', '60m' for minute data)

        Returns:
            pandas.DataFrame: Historical price data with columns [time, open, close, high, low, volume]
        """
        try:
            # Call the main get_price function from the Ashare implementation
            df = self.get_price(symbol, count=days, frequency=frequency)

            if df is not None and not df.empty:
                # Ensure proper column names and data types
                required_columns = ['open', 'close', 'high', 'low', 'volume']

                # Standardize column names if they differ
                if 'time' not in df.columns and df.index.name is None:
                    df.index.name = 'time'
                    df.reset_index(inplace=True)

                # Make sure all required columns exist
                for col in required_columns:
                    if col not in df.columns:
                        print(f"Warning: Column '{col}' not found in Ashare data")

                # Convert data types to numeric, handling any non-numeric values
                # Only convert the numeric columns, not the time/index column
                numeric_columns = ['open', 'close', 'high', 'low', 'volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # Remove any rows with NaN values after conversion
                df.dropna(inplace=True)

                # Ensure the dataframe has the standard column names expected by the system
                # Standardize the column names if needed
                column_mapping = {
                    'time': 'date',  # if time is used as column name instead of index
                    'day': 'date'    # if day is used as column name
                }

                for old_col, new_col in column_mapping.items():
                    if old_col in df.columns and new_col not in df.columns:
                        df.rename(columns={old_col: new_col}, inplace=True)

                # Set the date column as index if it exists
                if 'date' in df.columns:
                    df.set_index('date', inplace=True)
                elif 'time' in df.columns:
                    df.set_index('time', inplace=True)

                print(f"✅ 成功使用Ashare获取 {symbol} 数据，共 {len(df)} 条记录")
                return df
            else:
                print(f"❌ Ashare未能获取 {symbol} 的数据")
                return None

        except Exception as e:
            print(f"❌ 获取Ashare数据时出错 {symbol}: {e}")
            return None

    def get_price(self, code: str, end_date: str = '', count: int = 10, frequency: str = '1d', fields: list = []) -> pd.DataFrame:
        """
        Main function to get price data - mirrors the original Ashare implementation
        """
        xcode = code.replace('.XSHG', '').replace('.XSHE', '')  # Security code encoding compatibility processing
        xcode = 'sh' + xcode if ('XSHG' in code) else 'sz' + xcode if ('XSHE' in code) else code

        if frequency in ['1d', '1w', '1M']:  # 1d daily, 1w weekly, 1M monthly
            return self.get_price_sina(xcode, end_date=end_date, count=count, frequency=frequency)  # Main

        if frequency in ['1m', '5m', '15m', '30m', '60m']:  # Minute data, 1m only Tencent, 5m 5m, 60m 60m
            if frequency in '1m':
                return self.get_price_min_tx(xcode, end_date=end_date, count=count, frequency=frequency)
            return self.get_price_sina(xcode, end_date=end_date, count=count, frequency=frequency)

    # --- Tencent Daily ---
    def get_price_day_tx(self, code: str, end_date: str = '', count: int = 10, frequency: str = '1d') -> pd.DataFrame:
        """Daily acquisition with improved error handling and retry mechanism"""
        max_retries = 3
        retry_delay = 2  # seconds

        unit = 'week' if frequency in '1w' else 'month' if frequency in '1M' else 'day'  # Judge day, week, month
        if end_date:
            end_date = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.date) else end_date.split(' ')[0]
        end_date = '' if end_date == datetime.datetime.now().strftime('%Y-%m-%d') else end_date  # If today becomes empty
        URL = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},{unit},,{end_date},{count},qfq'

        for attempt in range(max_retries):
            try:
                response = requests.get(URL, timeout=30)

                if response.status_code != 200:
                    print(f"Warning: Received status code {response.status_code} for {code}")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                # Try to parse JSON response
                try:
                    st = json.loads(response.content)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for {code}: {e}")
                    print(f"Response content: {response.content[:500]}...")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                # Check if response has expected structure
                if 'data' not in st or code not in st['data']:
                    print(f"No data found for {code} in API response")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                ms = 'qfq' + unit
                stk = st['data'][code]
                buf = stk[ms] if ms in stk else stk[unit] if unit in stk else None  # Index return is not qfqday, it is day

                if not buf:
                    print(f"No kline data found for {code}")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                df = pd.DataFrame(buf, columns=['time', 'open', 'close', 'high', 'low', 'volume'])  # Remove dtype='float' to avoid string-to-float conversion error
                df.time = pd.to_datetime(df.time)
                df.set_index(['time'], inplace=True)
                df.index.name = ''  # Process index
                return df

            except requests.exceptions.ConnectionError as e:
                print(f"连接错误 {code}: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure
            except requests.exceptions.Timeout as e:
                print(f"请求超时 {code}: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure
            except requests.exceptions.RequestException as e:
                print(f"请求异常 {code}: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure
            except Exception as e:
                print(f"处理 {code} 数据时发生错误: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure

    # Tencent Minute Line
    def get_price_min_tx(self, code: str, end_date: Union[str, None] = None, count: int = 10, frequency: str = '1d') -> pd.DataFrame:
        """Minute line acquisition with improved error handling and retry mechanism"""
        max_retries = 3
        retry_delay = 2  # seconds

        ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1  # Parse K-line cycle number
        if end_date:
            end_date = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime.date) else end_date.split(' ')[0]
        URL = f'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param={code},m{ts},,{count}'

        for attempt in range(max_retries):
            try:
                response = requests.get(URL, timeout=30)

                if response.status_code != 200:
                    print(f"Warning: Received status code {response.status_code} for {code}")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                # Try to parse JSON response
                try:
                    st = json.loads(response.content)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for {code}: {e}")
                    print(f"Response content: {response.content[:500]}...")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                # Check if response has expected structure
                if 'data' not in st or code not in st['data'] or f'm{ts}' not in st['data'][code]:
                    print(f"No data found for {code} in API response")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                buf = st['data'][code]['m' + str(ts)]
                df = pd.DataFrame(buf, columns=['time', 'open', 'close', 'high', 'low', 'volume', 'n1', 'n2'])
                df = df[['time', 'open', 'close', 'high', 'low', 'volume']]
                df[['open', 'close', 'high', 'low', 'volume']] = df[['open', 'close', 'high', 'low', 'volume']].astype('float')
                df.time = pd.to_datetime(df.time)
                df.set_index(['time'], inplace=True)
                df.index.name = ''  # Process index
                df['close'][-1] = float(st['data'][code]['qt'][code][3])  # Latest fund data is 3 bits
                return df

            except requests.exceptions.ConnectionError as e:
                print(f"连接错误 {code}: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure
            except requests.exceptions.Timeout as e:
                print(f"请求超时 {code}: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure
            except requests.exceptions.RequestException as e:
                print(f"请求异常 {code}: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure
            except Exception as e:
                print(f"处理 {code} 数据时发生错误: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure

    # sina sina all cycle acquisition function, minute 5m,15m,30m,60m day 1d=240m week 1w=1200m 1month=7200m
    def get_price_sina(self, code: str, end_date: str = '', count: int = 10, frequency: str = '60m') -> pd.DataFrame:
        """Sina all cycle acquisition function with improved error handling and retry mechanism"""
        import time

        max_retries = 3
        retry_delay = 2  # seconds

        frequency = frequency.replace('1d', '240m').replace('1w', '1200m').replace('1M', '7200m')
        mcount = count
        ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1  # Parse K-line cycle number
        if (end_date != '') & (frequency in ['240m', '1200m', '7200m']):
            end_date = pd.to_datetime(end_date) if not isinstance(end_date, datetime.date) else end_date  # Convert to datetime
            unit = 4 if frequency == '1200m' else 29 if frequency == '7200m' else 1  # 4,29 a few more data don't affect speed
            count = count + (datetime.datetime.now() - end_date).days // unit  # How many natural days from end time to today (>trading day)
            # print(code,end_date,count)
        URL = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={ts}&ma=5&datalen={count}'

        for attempt in range(max_retries):
            try:
                response = requests.get(URL, timeout=30)

                if response.status_code != 200:
                    print(f"Warning: Received status code {response.status_code} for {code}")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                # Try to parse JSON response
                try:
                    dstr = json.loads(response.content)
                except json.JSONDecodeError as e:
                    print(f"JSON decode error for {code}: {e}")
                    print(f"Response content: {response.content[:500]}...")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                # Check if response is a list with data
                if not isinstance(dstr, list) or len(dstr) == 0:
                    print(f"No data found for {code} in Sina API response")
                    if attempt < max_retries - 1:
                        print(f"等待 {retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                        continue
                    else:
                        return pd.DataFrame()  # Return empty DataFrame on failure

                df = pd.DataFrame(dstr, columns=['day', 'open', 'high', 'low', 'close', 'volume'])  # Removed dtype='float' to prevent string-to-float conversion errors
                df['open'] = df['open'].astype(float)
                df['high'] = df['high'].astype(float)  # Convert data type
                df['low'] = df['low'].astype(float)
                df['close'] = df['close'].astype(float)
                df['volume'] = df['volume'].astype(float)
                df.day = pd.to_datetime(df.day)
                df.set_index(['day'], inplace=True)
                df.index.name = ''  # Process index
                if (end_date != '') & (frequency in ['240m', '1200m', '7200m']):
                    return df[df.index <= end_date][-mcount:]  # Day with end time first return
                return df

            except requests.exceptions.ConnectionError as e:
                print(f"连接错误 {code}: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure
            except requests.exceptions.Timeout as e:
                print(f"请求超时 {code}: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure
            except requests.exceptions.RequestException as e:
                print(f"请求异常 {code}: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure
            except Exception as e:
                print(f"处理 {code} 数据时发生错误: {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return pd.DataFrame()  # Return empty DataFrame on failure

    def convert_symbol_format(self, symbol: str) -> str:
        """
        Convert various symbol formats to Ashare-compatible format
        """
        # Remove exchange suffixes if present
        clean_symbol = symbol.replace('.XSHG', '').replace('.XSHE', '').replace('.SS', '').replace('.SZ', '')

        # If already has sh/sz prefix, return as is
        if clean_symbol.startswith(('sh', 'sz', 'bj')):
            return clean_symbol

        # Add appropriate prefix based on stock code
        if len(clean_symbol) == 6:
            if clean_symbol.startswith(('5', '6')):  # Shanghai stocks
                return f'sh{clean_symbol}'
            elif clean_symbol.startswith(('0', '2', '3')):  # Shenzhen stocks
                return f'sz{clean_symbol}'
            elif clean_symbol.startswith(('4', '8')):  # Beijing stocks
                return f'bj{clean_symbol}'

        # If none of the above, default to sz for 0-series, sh for 6-series
        if clean_symbol.startswith('0') or clean_symbol.startswith('3'):
            return f'sz{clean_symbol}'
        else:
            return f'sh{clean_symbol}'


# Example usage and testing
if __name__ == '__main__':
    fetcher = AshareDataFetcher()

    # Test with some common A-share symbols
    symbols = ['sh600023', 'sz000001', 'sh600519']

    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        data = fetcher.fetch_stock_data(symbol, days=30)
        if data is not None and not data.empty:
            print(f"Retrieved {len(data)} records for {symbol}")
            print(data.tail())
        else:
            print(f"Failed to retrieve data for {symbol}")