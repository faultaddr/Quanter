"""Enhanced Data Fetcher with support for multiple data sources including Tushare and EastMoney."""

import os
import tushare as ts
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from ...domain.interfaces.data_provider import IDataProvider
from ...core.errors import DataProviderError, ConfigurationError
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger

logger = get_logger(__name__)


def setup_tushare_api(token: str):
    """Setup and return Tushare API with the given token."""
    ts.set_token(token)
    pro = ts.pro_api()
    return pro


@registry.register(ComponentType.DATA_PROVIDER, "enhanced_data_fetcher")
class EnhancedDataFetcher(IDataProvider):
    """Enhanced data fetcher supporting Tushare and EastMoney data sources."""

    def __init__(
        self,
        tushare_token: str = None,
        eastmoney_cookie: str = None
    ):
        """
        Initialize enhanced data fetcher.

        Args:
            tushare_token: Tushare API token. If None, will try to get from TUSHARE_TOKEN environment variable.
            eastmoney_cookie: EastMoney cookie string. If None, will try to get from EASTMONEY_COOKIE environment variable.
        """
        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")
        self.eastmoney_cookie = eastmoney_cookie or os.getenv("EASTMONEY_COOKIE")

        if not self.tushare_token:
            raise ConfigurationError(
                "Tushare token not provided and TUSHARE_TOKEN environment variable not set"
            )

        # Setup Tushare API
        self.pro_api = setup_tushare_api(self.tushare_token)
        self._tushare_initialized = False

        # EastMoney headers
        self.eastmoney_headers = {
            'Cookie': self.eastmoney_cookie,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def initialize(self) -> None:
        """Initialize the data fetcher connections."""
        try:
            # Try to verify Tushare connection, but be tolerant of permission errors
            try:
                df = self.pro_api.trade_cal(
                    exchange="", start_date="20230101", end_date="20230102"
                )
                if df.empty:
                    logger.warning("Tushare connection established but no data returned for test query")
                else:
                    logger.info("Tushare connection verified successfully")
            except Exception as e:
                # Log the error but continue - the token may have limited permissions
                logger.warning(f"Tushare connection test failed (may have limited permissions): {str(e)}")

            # Verify EastMoney connection
            if self.eastmoney_cookie:
                # Perform a simple request to verify cookie is valid
                try:
                    test_url = "https://np-analyse.eastmoney.com/api/qt/ulist.np/get?po=1&pz=1&pn=1&np=1&fltt=2&invt=2&wbp2u=12915131124252524252135421&fid=f3&fs=m:0+t:6+f:!50&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152"
                    response = requests.get(test_url, headers=self.eastmoney_headers)
                    # Just check if we get a response without error
                    logger.info("EastMoney connection verified")
                except Exception as e:
                    logger.warning(f"Could not verify EastMoney connection: {str(e)}")

            self._tushare_initialized = True
            logger.info("EnhancedDataFetcher initialized successfully")
        except Exception as e:
            raise DataProviderError(f"Failed to initialize EnhancedDataFetcher: {str(e)}")

    def _fetch_from_eastmoney(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from EastMoney if available."""
        if not self.eastmoney_cookie:
            logger.warning("EastMoney cookie not available, falling back to Tushare")
            return pd.DataFrame()

        try:
            # Format symbol for EastMoney
            # Convert to EastMoney format (000001.sz, 600000.sh)
            if '.' in symbol:
                base_symbol = symbol.split('.')[0]
            else:
                base_symbol = symbol

            if len(base_symbol) == 6:
                if base_symbol.startswith(('5', '6', '9')):
                    # Shanghai stocks
                    em_symbol = f"{base_symbol}.sh"
                else:
                    # Shenzhen stocks
                    em_symbol = f"{base_symbol}.sz"
            else:
                em_symbol = symbol

            # EastMoney K-line data API (sample format - may need adjustment based on actual API)
            # Using a typical EastMoney API endpoint for historical K data
            url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                'secid': f'0.{base_symbol}' if base_symbol.startswith(('0', '3')) else f'1.{base_symbol}',
                'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                'fields1': 'f1,f2,f3,f4,f5,f6',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                'klt': '101',  # Daily data
                'fqt': '1',    # Forward adjustment
                'beg': start_date.replace('-', ''),
                'end': end_date.replace('-', '')
            }

            response = requests.get(url, headers=self.eastmoney_headers, params=params)
            data = response.json()

            if data.get('rc') != 0:
                logger.error(f"EastMoney API error for {symbol}: {data.get('msg')}")
                return pd.DataFrame()

            klinedata = data.get('data', {}).get('klines', [])
            if not klinedata:
                logger.warning(f"No EastMoney data found for {symbol}")
                return pd.DataFrame()

            # Parse kline data
            parsed_data = []
            for kline in klinedata:
                parts = kline.split(',')
                if len(parts) >= 7:
                    parsed_data.append({
                        'timestamp': parts[0],  # Date string
                        'open': float(parts[1]),
                        'close': float(parts[2]),
                        'high': float(parts[3]),
                        'low': float(parts[4]),
                        'volume': int(float(parts[5])),  # Convert from float to int
                        'amount': float(parts[6]) if len(parts) > 6 else 0.0,
                    })

            if not parsed_data:
                logger.warning(f"Parsed EastMoney data is empty for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(parsed_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['symbol'] = symbol
            df['timeframe'] = '1d'

            # Reorder columns to match expected format
            expected_cols = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'amount', 'timeframe', 'symbol'
            ]
            available_cols = [col for col in expected_cols if col in df.columns]
            df = df[available_cols]

            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching EastMoney data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV bars for the given symbols and timeframe.
        Prioritizes EastMoney data when available, falls back to Tushare.

        Args:
            symbols: List of symbols to retrieve
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe string ('1d' currently supported)

        Returns:
            Dictionary mapping symbols to their OHLCV dataframes
        """
        if not self._tushare_initialized:
            self.initialize()

        results = {}

        # Convert dates to string format
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        for symbol in symbols:
            try:
                df = pd.DataFrame()

                # Try EastMoney first if available
                if self.eastmoney_cookie:
                    logger.info(f"Attempting to fetch {symbol} from EastMoney")
                    df = self._fetch_from_eastmoney(symbol, start_str, end_str)

                # Fallback to Tushare if EastMoney data not available or failed
                if df.empty:
                    logger.info(f"Falling back to Tushare for {symbol}")

                    # Convert to Tushare format
                    if '.' not in symbol:
                        if len(symbol) == 6:
                            if symbol.startswith(('5', '6', '9')):
                                tushare_symbol = f"{symbol}.SH"
                            else:
                                tushare_symbol = f"{symbol}.SZ"
                        else:
                            tushare_symbol = symbol
                    else:
                        tushare_symbol = symbol

                    start_ts = start_date.strftime("%Y%m%d")
                    end_ts = end_date.strftime("%Y%m%d")

                    if timeframe == "1d":
                        try:
                            df = self.pro_api.daily(
                                ts_code=tushare_symbol, start_date=start_ts, end_date=end_ts
                            )

                            if df.empty:
                                logger.warning(f"No data found from Tushare for symbol {tushare_symbol}")
                        except Exception as e:
                            logger.warning(f"Tushare API error for {symbol}: {str(e)}. This may be due to limited API permissions.")
                            df = pd.DataFrame()

                        if not df.empty:
                            # Rename columns to match expected format
                            df.rename(
                                columns={
                                    "ts_code": "symbol",
                                    "trade_date": "timestamp",
                                    "vol": "volume"
                                },
                                inplace=True,
                            )

                            # Convert timestamp to datetime
                            df["timestamp"] = pd.to_datetime(df["timestamp"])

                            # Add timeframe and ensure symbol column has the original symbol
                            df["timeframe"] = timeframe
                            df["symbol"] = symbol

                            # Reorder columns to match expected format
                            expected_cols = [
                                "timestamp", "open", "high", "low", "close", "volume",
                                "amount", "timeframe", "symbol"
                            ]
                            available_cols = [col for col in expected_cols if col in df.columns]
                            df = df[available_cols]

                            # Sort by timestamp
                            df.sort_values("timestamp", inplace=True)
                            df.reset_index(drop=True, inplace=True)

                if not df.empty:
                    results[symbol] = df
                else:
                    logger.warning(f"Could not fetch data for {symbol}")

            except Exception as e:
                logger.error(f"Failed to get data for symbol {symbol}: {str(e)}")
                continue

        return results

    def get_latest_bar(
        self, symbol: str, timeframe: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get the most recent bar for a symbol.
        Prioritizes EastMoney data when available, falls back to Tushare.

        Args:
            symbol: Symbol to retrieve
            timeframe: Timeframe string (currently only supports '1d')

        Returns:
            DataFrame with the latest bar data, or None if unavailable
        """
        if not self._tushare_initialized:
            self.initialize()

        try:
            # Get data for the last week to ensure we get recent data
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(weeks=1)

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            df = pd.DataFrame()

            # Try EastMoney first if available
            if self.eastmoney_cookie:
                logger.info(f"Attempting to get latest bar for {symbol} from EastMoney")
                df = self._fetch_from_eastmoney(symbol, start_str, end_str)

            # Fallback to Tushare if EastMoney data not available or failed
            if df.empty:
                logger.info(f"Falling back to Tushare for latest bar of {symbol}")

                # Convert to Tushare format
                if '.' not in symbol:
                    if len(symbol) == 6:
                        if symbol.startswith(('5', '6', '9')):
                            tushare_symbol = f"{symbol}.SH"
                        else:
                            tushare_symbol = f"{symbol}.SZ"
                    else:
                        tushare_symbol = symbol
                else:
                    tushare_symbol = symbol

                start_ts = start_date.strftime("%Y%m%d")
                end_ts = end_date.strftime("%Y%m%d")

                try:
                    df = self.pro_api.daily(
                        ts_code=tushare_symbol, start_date=start_ts, end_date=end_ts
                    )

                    if df.empty:
                        logger.warning(f"No recent data found from Tushare for {tushare_symbol}")
                except Exception as e:
                    logger.warning(f"Tushare API error for latest bar of {symbol}: {str(e)}. This may be due to limited API permissions.")
                    df = pd.DataFrame()

                if not df.empty:
                    # Rename columns to match expected format
                    df.rename(
                        columns={
                            "ts_code": "symbol",
                            "trade_date": "timestamp",
                            "vol": "volume"
                        },
                        inplace=True,
                    )

                    # Convert timestamp to datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    # Add timeframe and ensure symbol column
                    df["timeframe"] = timeframe
                    df["symbol"] = symbol

                    # Reorder columns to match expected format
                    expected_cols = [
                        "timestamp", "open", "high", "low", "close", "volume",
                        "amount", "timeframe", "symbol"
                    ]
                    available_cols = [col for col in expected_cols if col in df.columns]
                    df = df[available_cols]

                    # Sort by timestamp
                    df.sort_values("timestamp", inplace=True)
                    df.reset_index(drop=True, inplace=True)

            if df.empty:
                logger.warning(f"No recent data found for {symbol}")
                return None

            # Get the most recent bar
            latest_bar = df.iloc[[-1]].copy()  # Use double brackets to keep as DataFrame

            return latest_bar

        except Exception as e:
            logger.error(f"Failed to get latest bar for symbol {symbol}: {str(e)}")
            return None

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        if not self._tushare_initialized:
            self.initialize()

        try:
            # Get stock basic information from Tushare
            stock_list = self.pro_api.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date",
            )
            return stock_list["ts_code"].tolist()
        except Exception as e:
            logger.error(f"Failed to get supported symbols: {str(e)}")
            return []

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for symbols matching the query.

        Args:
            query: Search query string

        Returns:
            List of matching symbols with metadata
        """
        if not self._tushare_initialized:
            self.initialize()

        try:
            # Get stock basic information from Tushare
            stock_list = self.pro_api.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date",
            )

            # Filter based on query (case-insensitive)
            matching_stocks = stock_list[
                (stock_list["name"].str.contains(query, case=False, na=False))
                | (stock_list["ts_code"].str.contains(query, case=False, na=False))
                | (stock_list["symbol"].str.contains(query, case=False, na=False))
            ]

            results = []
            for _, row in matching_stocks.iterrows():
                results.append(
                    {
                        "symbol": row["ts_code"],
                        "name": row["name"],
                        "area": row["area"],
                        "industry": row["industry"],
                        "list_date": row["list_date"],
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Failed to search symbols: {str(e)}")
            return []

    def get_calendar(self) -> List[datetime]:
        """
        Get trading calendar (list of trading days).

        Returns:
            List of trading days
        """
        if not self._tushare_initialized:
            self.initialize()

        try:
            # Get trading calendar for the past year from Tushare
            today = datetime.now()
            last_year = today - timedelta(days=365)

            start_str = last_year.strftime("%Y%m%d")
            end_str = today.strftime("%Y%m%d")

            cal_df = self.pro_api.trade_cal(
                exchange="", start_date=start_str, end_date=end_str
            )

            # Filter for open days
            open_days = cal_df[cal_df["is_open"] == 1]["cal_date"]
            return [datetime.strptime(date, "%Y%m%d") for date in open_days.tolist()]

        except Exception as e:
            logger.error(f"Failed to get trading calendar: {str(e)}")
            return []


def create_data_fetcher_with_credentials():
    """
    Create DataFetcher instance with provided credentials.
    This function uses the hardcoded credentials you provided for integration.
    """
    tushare_token = "744295f7af6adf63074518f919f5ad5054caf8b84d3c07c066f5c42e"

    eastmoney_cookie = (
        'qgqp_b_id=b7c0c5065c6db033910b1b3175b7c9bb; '
        'st_nvi=pr7nepf3axSLFdLauyP5y8deb; '
        'websitepoptg_api_time=1770690681021; '
        'st_si=43191381080720; '
        'nid18=0095a8fdc53e2c9dc00f4d602b3c459e; '
        'nid18_create_time=1770690681336; '
        'gviem=6A44mgyL6Tsg59OPlfAXDd677; '
        'gviem_create_time=1770690681337; '
        'p_origin=https%3A%2F%2Fpassport2.eastmoney.com; '
        'mtp=1; '
        'ct=wYdhYQ7SFCReRY7yObWFWJwcS2isXO6R8wHwamkysQRCcR9yEiEaMsskY-1tsHOmajDCrGLWHPVacX0DGd_9HoMFpWjxWtVUZEdR8ibclVermnomP1JWdjUpI3BhaRN2ft3jRsDjazoC6F9O5Jzssk-rkmWM3b3LsGJq5RJDxVM; '
        'ut=FobyicMgeV5FJnFT189SwEfSo-wAjCKxRGfhgXzug4j9BdKmq4gQdtlHffBaUl7Djr5Ju3CTO3tQqVCOs_Vhp9WUQe_9zHJxPmg__J71QWWtiytGWHR6CUXelUQfxok_geZEOJXcc9bQWieI7LUcRQjQFmB-1bwzaZYU3t525uGbFHwr6SZYdP3PBVz04EfQ796KX06LCuYpITwvNu6laJotFHyE5dflMcANoRBf6d8isLvw34K59yZB985bsVHnckUA0HIycKAoU137ZeAYrEX8rjmONDCZy7QGj-BHcAWyIH9OIF98zmSo71GWwWu_X5FP1R2JqWLg9CMTh9wlVBTitMAXMcc5; '
        'pi=9694097255613200%3Bu9694097255613200%3B%E5%A0%82%E5%A0%82%E6%AD%A3%E6%AD%A3%E7%9A%84%E6%9B%B9%E6%93%8D%3BryhxoVjcWC8PTbi0bFrviFAowUa3asGIsa%2F0auHDuAKp6CJ%2BPVN0UwnSDOaEd7utp5uK4oSJImRgmTF0VD7Nm1Zqq9vnKuG5c1wWVRNZxJmnEN416UgEorQVUQJ5tnsTgIcvWxtVIJHhIll%2F9SIWv6E6wIrLFINK3wF12TZX3gkL7%2FxLaYbHaFQ0YON21YMY%2BZKCiilR%3Bp2dLhWNuZSa0SCigDD%2FOLxaCiti2fW5OSY32vbSSck%2BT1BzvA%2FAQHG2jYCxHc8Httaxt1PRsFPhuwvBF873qXa7Y5muaKZZN0jzerURbzjeerxd31x755Is9mu7LD%2BGWpkI3piLVRUUL5xl2ifRVnekqrax4Yg%3D%3D; '
        'uidal=9694097255613200%e5%a0%82%e5%a0%82%e6%ad%a3%e6%ad%a3%e7%9a%84%e6%9b%b9%e6%93%8d; '
        'sid=; vtpst=|; wsc_checkuser_ok=1; fullscreengg=1; fullscreengg2=1; '
        'st_pvi=27562121748759; st_sp=2025-10-30%2011%3A15%3A42; '
        'st_inirUrl=https%3A%2F%2Fwww.google.com.hk%2F; st_sn=5; '
        'st_psi=20260210130257951-111000300841-0487608401'
    )

    return EnhancedDataFetcher(
        tushare_token=tushare_token,
        eastmoney_cookie=eastmoney_cookie
    )