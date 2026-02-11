"""
Reusable EastMoney Data Fetcher Module
Provides a centralized way to fetch A-Share market data using EastMoney cookies
Enhanced with better error handling and retry mechanisms
Integrating MyTT technical indicators for comprehensive analysis
"""

import pandas as pd
import numpy as np
import requests
from requests.exceptions import ConnectionError, Timeout, ChunkedEncodingError, RequestException
from urllib3.exceptions import ProtocolError, NewConnectionError, MaxRetryError
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
                'qgqp_b_id': 'b7c0c5065c6db033910b1b3175b7c9bb',
                'st_nvi': 'pr7nepf3axSLFdLauyP5y8deb',
                'websitepoptg_api_time': '1770690681021',
                'st_si': '43191381080720',
                'nid18': '0095a8fdc53e2c9dc00f4d602b3c459e',
                'nid18_create_time': '1770690681336',
                'gviem': '6A44mgyL6Tsg59OPlfAXDd677',
                'gviem_create_time': '1770690681337',
                'p_origin': 'https%3A%2F%2Fpassport2.eastmoney.com',
                'mtp': '1',
                'ct': 'wYdhYQ7SFCReRY7yObWFWJwcS2isXO6R8wHwamkysQRCcR9yEiEaMsskY-1tsHOmajDCrGLWHPVacX0DGd_9HoMFpWjxWtVUZEdR8ibclVermnomP1JWdjUpI3BhaRN2ft3jRsDjazoC6F9O5Jzssk-rkmWM3b3LsGJq5RJDxVM',
                'ut': 'FobyicMgeV5FJnFT189SwEfSo-wAjCKxRGfhgXzug4j9BdKmq4gQdtlHffBaUl7Djr5Ju3CTO3tQqVCOs_Vhp9WUQe_9zHJxPmg__J71QWWtiytGWHR6CUXelUQfxok_geZEOJXcc9bQWieI7LUcRQjQFmB-1bwzaZYU3t525uGbFHwr6SZYdP3PBVz04EfQ796KX06LCuYpITwvNu6laJotFHyE5dflMcANoRBf6d8isLvw34K59yZB985bsVHnckUA0HIycKAoU137ZeAYrEX8rjmONDCZy7QGj-BHcAWyIH9OIF98zmSo71GWwWu_X5FP1R2JqWLg9CMTh9wlVBTitMAXMcc5',
                'pi': '9694097255613200%3Bu9694097255613200%3B%E5%A0%82%E5%A0%82%E6%AD%A3%E6%AD%A3%E7%9A%84%E6%9B%B9%E6%93%8D%3BryhxoVjcWC8PTbi0bFrviFAowUa3asGIsa%2F0auHDuAKp6CJ%2BPVN0UwnSDOaEd7utp5uK4oSJImRgmTF0VD7Nm1Zqq9vnKuG5c1wWVRNZxJmnEN416UgEorQVUQJ5tnsTgIcvWxtVIJHhIll%2F9SIWv6E6wIrLFINK3wF12TZX3gkL7%2FxLaYbHaFQ0YON21YMY%2BZKCiilR%3Bp2dLhWNuZSa0SCigDD%2FOLxaCiti2fW5OSY32vbSSck%2BT1BzvA%2FAQHG2jYCxHc8Httaxt1PRsFPhuwvBF873qXa7Y5muaKZZN0jzerURbzjeerxd31x755Is9mu7LD%2BGWpkI3piLVRUUL5xl2ifRVnekqrax4Yg%3D%3D',
                'uidal': '9694097255613200%e5%a0%82%e5%a0%82%e6%ad%a3%e6%ad%a3%e7%9a%84%e6%9b%b9%e6%93%8d',
                'sid': '',
                'vtpst': '|',
                'st_asi': 'delete',
                'wsc_checkuser_ok': '1',
                'fullscreengg': '1',
                'fullscreengg2': '1',
                'st_pvi': '27562121748759',
                'st_sp': '2025-10-30%2011%3A15%3A42',
                'st_inirUrl': 'https%3A%2F%2Fwww.google.com.hk%2F',
                'st_sn': '5',
                'st_psi': '20260210130257951-111000300841-0487608401'
            }
        else:
            self.cookies = cookies

        self.session = requests.Session()

        # 配置适配器，不设置重试策略
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter()
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def fetch_stock_data(self, symbol, days=60):
        """
        Fetch stock data using EastMoney API with cookies
        """
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
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'application/json, text/javascript, */*; q=0.01',
                'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://quote.eastmoney.com/',
                'X-Requested-With': 'XMLHttpRequest',
                'Connection': 'keep-alive',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-site',
                'Pragma': 'no-cache',
                'Cache-Control': 'no-cache',
                'Cookie': '; '.join([f"{k}={v}" for k, v in self.cookies.items()])
            }

            response = self.session.get(url, params=params, headers=headers, timeout=30)

            # Handle compressed content (gzip, deflate, or brotli)
            content_encoding = response.headers.get('Content-Encoding', '').lower()
            if content_encoding == 'br':
                # Handle Brotli compression
                try:
                    import brotli
                    response._content = brotli.decompress(response.content)
                    # Update encoding to ensure proper text interpretation
                    response.encoding = 'utf-8'
                except ImportError:
                    print("Brotli library not available, response may be corrupted")
                except Exception as e:
                    print(f"Brotli decompression failed: {e}")
                    # Don't raise here, let the JSON parsing handle any issues
            elif content_encoding == 'gzip':
                # Handle gzip compression, but verify it's actually gzipped
                try:
                    import gzip
                    # Check if content actually looks like gzip data (starts with 1f8b)
                    if len(response.content) >= 2 and response.content[0] == 0x1f and response.content[1] == 0x8b:
                        response._content = gzip.decompress(response.content)
                        response.encoding = 'utf-8'
                    else:
                        print("Content marked as gzip but doesn't appear to be gzipped, leaving as-is")
                except Exception as e:
                    print(f"Gzip decompression failed: {e}")
            elif content_encoding == 'deflate':
                # Handle deflate compression
                try:
                    import zlib
                    # Try standard deflate decompression
                    response._content = zlib.decompress(response.content)
                    response.encoding = 'utf-8'
                except zlib.error:
                    # Some servers use raw deflate without zlib headers
                    try:
                        response._content = zlib.decompress(response.content, -zlib.MAX_WBITS)
                        response.encoding = 'utf-8'
                    except Exception as e:
                        print(f"Deflate decompression failed: {e}")
                except Exception as e:
                    print(f"Deflate decompression failed: {e}")

            if response.status_code == 200:
                # Check if response is valid JSON
                try:
                    data = response.json()

                    # Verify that the response contains expected structure
                    if isinstance(data, dict) and 'data' in data and data['data'] and 'klines' in data['data']:
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
                            return None
                    else:
                        print(f"EastMoney API响应结构异常或数据为空: {type(data)}")
                        # Print first 200 chars of response text for debugging
                        print(f"Response preview: {response.text[:200]}...")
                        return None
                except ValueError as e:
                    # Response is not valid JSON - probably an error page or blocked request
                    print(f"EastMoney API返回非JSON格式内容: {e}")
                    print(f"Response status: {response.status_code}")
                    print(f"Response headers: {dict(response.headers)}")
                    print(f"Response preview: {response.text[:500]}...")

                    # Check if we should return based on response
                    if "访问过于频繁" in response.text or "请求过于频繁" in response.text or response.status_code == 403:
                        return None
                    return None
                except requests.RequestException:
                    # Re-raise RequestExceptions to be caught by outer exception handler
                    raise
                except Exception as e:
                    # Catch any other exceptions during parsing
                    print(f"解析EastMoney API响应时出错: {e}")
                    print(f"Response status: {response.status_code}")
                    print(f"Response preview: {response.text[:500]}...")
                    return None
            else:
                print(f"EastMoney API请求失败，状态码: {response.status_code}")
                print(f"Response preview: {response.text[:200]}...")

                # If it's a server error, return None
                return None

        except requests.exceptions.ConnectionError as e:
            print(f"连接错误 {symbol}: {e}")
            print(f"❌ 无法获取 {symbol} 的数据")
            return None
        except requests.exceptions.Timeout as e:
            print(f"请求超时 {symbol}: {e}")
            print(f"❌ 无法获取 {symbol} 的数据")
            return None
        except requests.exceptions.ChunkedEncodingError as e:
            print(f"分块编码错误 {symbol}: {e}")
            print(f"❌ 无法获取 {symbol} 的数据")
            return None
        except requests.exceptions.RequestException as e:
            print(f"请求异常 {symbol}: {e}")
            print(f"❌ 无法获取 {symbol} 的数据")
            return None
        except ProtocolError as e:
            print(f"协议错误 {symbol}: {e}")
            print(f"❌ 无法获取 {symbol} 的数据")
            return None
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
            # If error occurs, return None
            print(f"❌ 无法获取 {symbol} 的数据")
            return None

        # If all attempts failed
        print(f"❌ 无法获取 {symbol} 的数据")
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

        # Import MyTT indicators
        from quant_trade_a_share.utils.mytt_indicators import calculate_mytt_indicators

        # Calculate all MyTT indicators
        df = calculate_mytt_indicators(df)

        # Additional indicators beyond MyTT
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

        # Rate of Change
        df['roc'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100

        # Trend indicators
        df['trend'] = np.where(df['close'] > df['ma20'], 1, 0)  # 1 for uptrend, 0 for downtrend

        # Fill NaN values with 0 for clean data
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)

        return df

    def close_session(self):
        """Close the session to free up resources"""
        if hasattr(self, 'session'):
            self.session.close()

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
            sh_response = self.session.get(sh_url, params=sh_params, headers=headers, timeout=30)
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
            sz_response = self.session.get(sz_url, params=sz_params, headers=headers, timeout=30)
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
            bj_response = self.session.get(bj_url, params=bj_params, headers=headers, timeout=30)
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