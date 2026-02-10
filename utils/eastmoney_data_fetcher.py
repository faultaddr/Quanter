"""
Reusable EastMoney Data Fetcher Module
Provides a centralized way to fetch A-Share market data using EastMoney cookies
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import json


class EastMoneyDataFetcher:
    """
    Reusable class to fetch A-Share market data using EastMoney cookies
    """
    
    def __init__(self, cookies=None):
        """
        Initialize with EastMoney cookies
        If no cookies provided, uses default cookies from the prediction module
        """
        if cookies is None:
            # Default cookies from the predictive_analyzer.py
            self.cookies = {
                'ASL': '20494,0000d,8be20aff',
                'ADVC': '3ee81b757962bc',
                'ADVS': '3ee81b757962bc',
                'qgqp_b_id': '5214d909bcc66e93576b49ed3d446e38',
                'st_nvi': 'PZZhsgK0ZsqG3vHBMU-4g0c46',
                'websitepoptg_api_time': '1770665319207',
                'st_si': '44545933999131',
                'nid18': '0cb935b80cd1336d400798228688f23e',
                'nid18_create_time': '1770665319416',
                'gviem': '_krPH3C3Ybs-kJyqdlhK9598d',
                'gviem_create_time': '1770665319416',
                'p_origin': 'https%3A%2F%2Fpassport2.eastmoney.com',
                'mtp': '1',
                'ct': 'j2-rb8gsYEH7Z5hfhA_9WkaiA66JMtMhasWm5IaNF7xSY0Q1QHUR8w2IC_dQlFzfQfbVcNBBm5MdHmEBSXRScIWFyHjzzm0mH1p8lwDeKo--nqL3nTKwKwg08w11_RniWauFoL3tWOwknftIoosjmHsSPjOdn1ZS5PLW_9pHC_4',
                'ut': 'FobyicMgeV6Gl5Ws0rOH5qvs-ZS0k9XvNXWKKa42q-agegqBk6oLosMw8RzR-iuurrDoc1kUl0jT5cRIAUAhTXaafTsuUZo5Ef0TELgIYsuL6W1cH-RjJf-IR6_Qb_7bwQSIRyKP4OqDlhze9fNwQZenBxx4FXFTxBmD9pS_ZoRqb7PVus-sZsyLgYm0tus-oDDyROxO-WE7MVpEDKxbC3s2cYKtYU4TTY8Lot4UXuHn6hUEv_N8tfb3sJyKA9-mxqVVLZYDNDmmRygALO7NNdoNYXTAebWI',
                'pi': '9694097255613200%3Bu9694097255613200%3B%E5%A0%82%E5%A0%82%E6%AD%A3%E6%AD%A3%E7%9A%84%E6%9B%B9%E6%93%8D%3BLSrFBlVclIYPg4pBOrim34v0hS8%2Bw2owuUFcpj0%2BGkIi897wjraBNPTUKgjtxkQI2Z%2BYVW%2F7zHPpH%2Bk7RMVMu8mEpKbMNOVi1ybo6%2FmJTuILjybcZZFcRv7BSbUUyjB4ZLRjN0ID%2FNmlx5RhlDRAyMBeC69O8A96P7KMdBllLB0qcPcL6XlKPyGwxj1OCCxdiivc1%2F4P%3BkKW5qgd%2BimVm5dzQstH7DrvYE%2FIlKvcz5fJwIAUTrjSGhqknW0d9oJwGyNBZY7%2Bbb97NLjZBiQgcOwDVpln%2F7sT7KuzufFPV6TUh0zWDyWjQUy2R6kly72KifsONMTLXsXx3r3ATLwQ4EmVGHXijrfcKBZQNAw%3D%3D',
                'uidal': '9694097255613200%e5%a0%82%e5%a0%82%e6%ad%a3%e6%ad%a3%e7%9a%84%e6%9b%b9%e6%93%8d',
                'sid': '',
                'vtpst': '|',
                'st_pvi': '04127630559918',
                'st_sp': '2026-02-10%2003%3A28%3A39',
                'st_inirUrl': 'https%3A%2F%2Fwww.baidu.com%2Flink',
                'st_sn': '3',
                'st_psi': '20260210032907843-111000300841-7036945406',
                'st_asi': 'delete',
                'fullscreengg': '1',
                'fullscreengg2': '1'
            }
        else:
            self.cookies = cookies
            
        self.max_retries = 3
        self.retry_delay = 2  # seconds

    def fetch_stock_data(self, symbol, days=60):
        """
        Fetch stock data using EastMoney API with cookies
        """
        for attempt in range(self.max_retries):
            try:
                # Calculate date range
                end_date = datetime.now().strftime('%Y%m%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

                # Use requests to fetch data with cookies to bypass anti-bot measures
                url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
                params = {
                    'fields1': 'f1,f2,f3,f4,f5,f6',
                    'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61,f116',
                    'ut': '7eea3edcaed734bea9cbfc24409ed989',
                    'klt': '101',  # Daily data
                    'fqt': '0',
                    'secid': f"{1 if symbol.startswith('sh') else 0}.{symbol[2:]}",  # Format as "1.600519" or "0.000001"
                    'beg': start_date,
                    'end': end_date
                }

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': '*/*',
                    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'Referer': 'https://quote.eastmoney.com/',
                    'Cookie': '; '.join([f"{k}={v}" for k, v in self.cookies.items()])
                }

                response = requests.get(url, params=params, headers=headers)

                if response.status_code == 200:
                    data = response.json()

                    if 'data' in data and data['data'] and 'klines' in data['data']:
                        klines = data['data']['klines']

                        if klines:
                            # Parse kline data
                            dates = []
                            opens = []
                            closes = []
                            highs = []
                            lows = []
                            volumes = []

                            for kline in klines:
                                parts = kline.split(',')
                                if len(parts) >= 7:
                                    dates.append(parts[0])  # date
                                    opens.append(float(parts[1]))  # open
                                    closes.append(float(parts[2]))  # close
                                    highs.append(float(parts[3]))  # high
                                    lows.append(float(parts[4]))  # low
                                    volumes.append(int(float(parts[6])))  # volume

                            # Create DataFrame
                            df = pd.DataFrame({
                                'date': pd.to_datetime(dates),
                                'open': opens,
                                'close': closes,
                                'high': highs,
                                'low': lows,
                                'volume': volumes
                            })

                            df.set_index('date', inplace=True)

                            # Calculate enhanced technical indicators using Qlib
                            df = self.calculate_enhanced_technical_indicators(df)

                            print(f"✅ 成功使用EastMoney Cookie获取 {symbol} 数据")
                            return df
                        else:
                            print(f"EastMoney API未返回 {symbol} 的数据")
                    else:
                        print(f"EastMoney API响应结构异常: {data}")
                else:
                    print(f"EastMoney API请求失败，状态码: {response.status_code}")

            except Exception as e:
                print(f"获取 {symbol} 数据失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    print(f"等待 {self.retry_delay} 秒后重试...")
                    time.sleep(self.retry_delay)

        # If all attempts fail, return None
        print(f"❌ 所有尝试均失败，无法获取 {symbol} 的数据")
        return None

    def calculate_enhanced_technical_indicators(self, df):
        """
        Calculate enhanced technical indicators using Qlib methodology
        """
        # Basic indicators that were already calculated
        # Moving averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_10'] = df['close'].rolling(window=10).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_30'] = df['close'].rolling(window=30).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd'] = exp12 - exp26
        df['signal'] = df['macd'].ewm(span=9).mean()
        df['histogram'] = df['macd'] - df['signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Enhanced indicators inspired by Qlib
        # Price momentum
        df['momentum'] = df['close'].pct_change(periods=5)
        df['volatility'] = df['close'].pct_change().rolling(window=10).std()

        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=10).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Qlib-inspired indicators
        # High-Low spread
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        
        # Price position in the day's range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-12)  # Adding small value to avoid division by zero
        
        # Log returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # High-low volatility
        df['hl_volatility'] = df['log_return'].rolling(window=5).std()
        
        # Volume-price trend
        df['vpt'] = (df['volume'] * (df['close'] - df['close'].shift(1)) / df['close'].shift(1)).cumsum()
        
        # Williams %R
        df['williams_r'] = (df['high'].rolling(window=14).max() - df['close']) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) * -100
        
        # Stochastic oscillator
        df['stoch_k'] = (df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        # Rate of Change
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
        
        # Average True Range (ATR)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['true_range'] = pd.concat([df['tr1'], df['tr2'], df['tr3']], axis=1).max(axis=1)
        df['atr'] = df['true_range'].rolling(window=14).mean()
        
        # Trend indicators
        df['trend'] = np.where(df['close'] > df['ma_20'], 1, 0)  # 1 for uptrend, 0 for downtrend

        # Fill NaN values with 0 for clean data
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        return df

    def get_all_stocks(self):
        """
        Get list of all A-share stocks using EastMoney API
        """
        try:
            print("获取股票列表...")
            
            # Get Shanghai stocks
            sh_url = "https://23.push2.eastmoney.com/api/qt/clist/get"
            sh_params = {
                'pn': '1',
                'pz': '5000',  # Large number to get all stocks
                'po': '1',
                'np': '1',
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': '2',
                'invt': '2',
                'fid': 'f3',
                'fs': 'm:1+t:2,m:1+t:23',
                'fields': 'f12,f14'
            }
            
            # Get Shenzhen stocks
            sz_url = "https://23.push2.eastmoney.com/api/qt/clist/get"
            sz_params = {
                'pn': '1',
                'pz': '5000',  # Large number to get all stocks
                'po': '1',
                'np': '1',
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': '2',
                'invt': '2',
                'fid': 'f3',
                'fs': 'm:0+t:6,m:0+t:80',
                'fields': 'f12,f14'
            }
            
            # Get Beijing stocks
            bj_url = "https://23.push2.eastmoney.com/api/qt/clist/get"
            bj_params = {
                'pn': '1',
                'pz': '5000',  # Large number to get all stocks
                'po': '1',
                'np': '1',
                'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
                'fltt': '2',
                'invt': '2',
                'fid': 'f3',
                'fs': 'm:0+t:81+s:2048',
                'fields': 'f12,f14'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': '*/*',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://quote.eastmoney.com/',
                'Cookie': '; '.join([f"{k}={v}" for k, v in self.cookies.items()])
            }
            
            # Fetch all stocks
            all_stocks_data = []
            
            # Get Shanghai stocks
            sh_response = requests.get(sh_url, params=sh_params, headers=headers)
            if sh_response.status_code == 200:
                sh_data = sh_response.json()
                if 'data' in sh_data and 'diff' in sh_data['data']:
                    for item in sh_data['data']['diff']:
                        all_stocks_data.append({
                            'symbol': f"sh{item['f12']}",
                            'code': item['f12'],
                            'name': item['f14']
                        })
            
            # Get Shenzhen stocks
            sz_response = requests.get(sz_url, params=sz_params, headers=headers)
            if sz_response.status_code == 200:
                sz_data = sz_response.json()
                if 'data' in sz_data and 'diff' in sz_data['data']:
                    for item in sz_data['data']['diff']:
                        all_stocks_data.append({
                            'symbol': f"sz{item['f12']}",
                            'code': item['f12'],
                            'name': item['f14']
                        })
            
            # Get Beijing stocks
            bj_response = requests.get(bj_url, params=bj_params, headers=headers)
            if bj_response.status_code == 200:
                bj_data = bj_response.json()
                if 'data' in bj_data and 'diff' in bj_data['data']:
                    for item in bj_data['data']['diff']:
                        all_stocks_data.append({
                            'symbol': f"bj{item['f12']}",
                            'code': item['f12'],
                            'name': item['f14']
                        })
            
            if all_stocks_data:
                all_stocks = pd.DataFrame(all_stocks_data)
                print(f"成功获取 {len(all_stocks)} 只股票")
                return all_stocks
            else:
                print("无法通过EastMoney API获取股票列表")
                # Return sample data for testing
                sample_data = {
                    'symbol': ['sh600000', 'sz000001', 'sh600036', 'sz000858', 'sh600519'],
                    'code': ['600000', '000001', '600036', '000858', '600519'],
                    'name': ['浦发银行', '平安银行', '招商银行', '五粮液', '贵州茅台']
                }
                return pd.DataFrame(sample_data)
                
        except Exception as e:
            print(f"获取股票列表失败: {e}")
            # Return sample data for testing
            sample_data = {
                'symbol': ['sh600000', 'sz000001', 'sh600036', 'sz000858', 'sh600519'],
                'code': ['600000', '000001', '600036', '000858', '600519'],
                'name': ['浦发银行', '平安银行', '招商银行', '五粮液', '贵州茅台']
            }
            return pd.DataFrame(sample_data)