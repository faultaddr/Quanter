"""Real data provider implementation for QuantTool using multiple data sources."""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from ...domain.interfaces.data_provider import IDataProvider
from ...core.errors import DataProviderError, ConfigurationError
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger

# Import required libraries conditionally
try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False
    print("Warning: tushare not available, Tushare data source will be disabled")

try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False
    print("Warning: baostock not available, Baostock data source will be disabled")

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance not available, Yahoo Finance data source will be disabled")
except TypeError:
    YFINANCE_AVAILABLE = False
    print("Warning: yfinance compatibility issue detected, Yahoo Finance data source will be disabled")


logger = get_logger(__name__)


def convert_symbol_format(symbol: str, source: str = "tushare") -> str:
    """
    Convert symbol format between different data sources.

    Args:
        symbol: Original symbol format
        source: Target data source ('tushare', 'baostock', 'yahoo', 'eastmoney')

    Returns:
        Converted symbol format
    """
    # If symbol already has exchange prefix (e.g., 000001.SZ), handle appropriately
    if '.' in symbol:
        base_symbol = symbol.split('.')[0]
    else:
        base_symbol = symbol

    # Check if it's a 6-digit code (A-share format)
    if len(base_symbol) == 6:
        if source == "tushare" or source == "eastmoney":
            # Tushare and EastMoney typically use 000001.SZ format
            if base_symbol.startswith(('5', '6', '9')):
                # Shanghai: 5xxx, 6xxx, 9xxx
                return f"{base_symbol}.SH"
            else:
                # Shenzhen: 0xxx, 2xxx, 3xxx, etc.
                return f"{base_symbol}.SZ"
        elif source == "baostock":
            # Baostock uses sh.000001, sz.000001 format
            if base_symbol.startswith(('5', '6', '9')):
                return f"sh.{base_symbol}"
            else:
                return f"sz.{base_symbol}"
        elif source == "yahoo":
            # Yahoo Finance uses 000001.SS (Shanghai), 000001.SZ (Shenzhen)
            if base_symbol.startswith(('5', '6', '9')):
                return f"{base_symbol}.SS"
            else:
                return f"{base_symbol}.SZ"
    return symbol


@registry.register(ComponentType.DATA_PROVIDER, "real_a_share")
class RealAShareDataProvider(IDataProvider):
    """Real A-share data provider implementation supporting multiple data sources."""

    def __init__(
        self,
        primary_source: str = "tushare",
        tushare_token: str = None,
        use_fallback: bool = True
    ):
        """
        Initialize real A-share data provider.

        Args:
            primary_source: Primary data source ('tushare', 'baostock', 'yahoo', 'eastmoney')
            tushare_token: Tushare API token (if using tushare)
            use_fallback: Whether to use fallback sources when primary fails
        """
        self.primary_source = primary_source
        self.use_fallback = use_fallback
        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")
        self.bs_logged_in = False
        self._initialized = False

        # Validate primary source
        valid_sources = ["tushare", "baostock", "yahoo"]
        if primary_source not in valid_sources:
            raise ValueError(f"Invalid primary source: {primary_source}. Valid options: {valid_sources}")

        if primary_source == "tushare" and not TUSHARE_AVAILABLE:
            raise RuntimeError("Tushare is not available. Please install it: pip install tushare")
        if primary_source == "baostock" and not BAOSTOCK_AVAILABLE:
            raise RuntimeError("Baostock is not available. Please install it: pip install baostock")
        if primary_source == "yahoo" and not YFINANCE_AVAILABLE:
            raise RuntimeError("Yahoo Finance is not available. Please install it: pip install yfinance")

    def initialize(self) -> None:
        """Initialize the data provider."""
        try:
            if self.primary_source == "tushare":
                if not self.tushare_token:
                    raise ConfigurationError(
                        "Tushare token not provided and TUSHARE_TOKEN environment variable not set"
                    )
                ts.set_token(self.tushare_token)
                self.pro_api = ts.pro_api()

                # Verify connection by fetching basic info
                df = self.pro_api.trade_cal(
                    exchange="", start_date="20230101", end_date="20230102"
                )
                if df.empty:
                    raise DataProviderError(
                        "Failed to connect to Tushare API - no data returned"
                    )

            elif self.primary_source == "baostock":
                if not self.bs_logged_in:
                    bs.login()
                    self.bs_logged_in = True

            self._initialized = True
            logger.info(f"RealAShareDataProvider initialized with primary source: {self.primary_source}")
        except Exception as e:
            raise DataProviderError(f"Failed to initialize RealAShareDataProvider: {str(e)}")

    def _fetch_tushare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Tushare."""
        if not self._initialized:
            self.initialize()

        try:
            # Convert symbol to Tushare format (000001.SZ, 600000.SH)
            tushare_symbol = convert_symbol_format(symbol, "tushare")

            # Format dates for Tushare
            start_str = start_date.replace("-", "")
            end_str = end_date.replace("-", "")

            # Fetch daily data
            df = self.pro_api.daily(
                ts_code=tushare_symbol, start_date=start_str, end_date=end_str
            )

            if df.empty:
                logger.warning(f"No data found from Tushare for symbol {tushare_symbol}")
                return pd.DataFrame()

            # Rename columns to match expected format
            df.rename(
                columns={
                    "ts_code": "symbol",
                    "trade_date": "timestamp",
                    "vol": "volume",
                    "amount": "amount"
                },
                inplace=True,
            )

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Add timeframe column
            df["timeframe"] = "1d"

            # Ensure symbol column has the original symbol format
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

            return df

        except Exception as e:
            logger.error(f"Error fetching data from Tushare for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _fetch_baostock(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Baostock."""
        if not self._initialized:
            self.initialize()

        try:
            # Convert symbol to Baostock format (sz.000001, sh.600000)
            bst_symbol = convert_symbol_format(symbol, "baostock")

            # Query historical data
            rs = bs.query_history_k_data_plus(
                bst_symbol,
                "date,code,open,high,low,close,volume,amount",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="3"  # No adjustment
            )

            # Collect data row by row
            data_list = []
            while (rs.error_code == '0') & rs.next():
                row_data = rs.get_row_data()
                data_list.append(row_data)

            if not data_list:
                logger.warning(f"No data found from Baostock for symbol {bst_symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data_list, columns=["date", "code", "open", "high", "low", "close", "volume", "amount"])

            if df.empty:
                logger.warning(f"No data returned from Baostock for symbol {bst_symbol}")
                return pd.DataFrame()

            # Convert data types
            numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Convert date column to datetime
            df["timestamp"] = pd.to_datetime(df["date"])

            # Add required columns
            df["timeframe"] = "1d"
            df["symbol"] = symbol  # Use original symbol format

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

            return df

        except Exception as e:
            logger.error(f"Error fetching data from Baostock for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _fetch_yahoo(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        if not YFINANCE_AVAILABLE:
            logger.error("Yahoo Finance is not available")
            return pd.DataFrame()

        try:
            # Convert symbol to Yahoo format (000001.SS, 000001.SZ)
            yahoo_symbol = convert_symbol_format(symbol, "yahoo")

            # Fetch data using yfinance
            ticker = yf.Ticker(yahoo_symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning(f"No data found from Yahoo Finance for symbol {yahoo_symbol}")
                return pd.DataFrame()

            # Rename columns to match expected format
            df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Dividends": "dividends",
                    "Stock Splits": "stock_splits"
                },
                inplace=True,
            )

            # Add timestamp column (index is already datetime)
            df["timestamp"] = df.index
            df["timeframe"] = "1d"
            df["symbol"] = symbol  # Use original symbol format

            # Add amount if not present (calculate as close * volume / 1000)
            if "amount" not in df.columns and "close" in df.columns and "volume" in df.columns:
                df["amount"] = df["close"] * df["volume"] / 1000

            # Reorder columns to match expected format
            expected_cols = [
                "timestamp", "open", "high", "low", "close", "volume",
                "amount", "timeframe", "symbol"
            ]
            available_cols = [col for col in expected_cols if col in df.columns]
            df = df[available_cols]

            # Reset index to remove date index
            df.reset_index(drop=True, inplace=True)

            # Sort by timestamp
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _try_fallback_sources(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Try alternative data sources when primary fails."""
        sources_to_try = []

        # Define priority order for fallback
        if self.primary_source != "tushare" and TUSHARE_AVAILABLE:
            sources_to_try.append(("tushare", self._fetch_tushare))
        if self.primary_source != "baostock" and BAOSTOCK_AVAILABLE:
            sources_to_try.append(("baostock", self._fetch_baostock))
        if self.primary_source != "yahoo" and YFINANCE_AVAILABLE:
            sources_to_try.append(("yahoo", self._fetch_yahoo))

        for source_name, fetch_func in sources_to_try:
            logger.info(f"Trying fallback source: {source_name}")
            try:
                df = fetch_func(symbol, start_date, end_date)
                if not df.empty:
                    logger.info(f"Successfully fetched data from {source_name}")
                    return df
            except Exception as e:
                logger.error(f"Fallback to {source_name} failed: {str(e)}")
                continue

        return pd.DataFrame()

    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV bars for the given symbols and timeframe from real data sources.

        Args:
            symbols: List of symbols to retrieve
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe string (currently only supports '1d' for real data)

        Returns:
            Dictionary mapping symbols to their OHLCV dataframes
        """
        if not self._initialized:
            self.initialize()

        results = {}

        # Convert dates to string format
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        for symbol in symbols:
            try:
                # Try primary source first
                logger.info(f"Fetching data for {symbol} from {self.primary_source}")

                if self.primary_source == "tushare":
                    df = self._fetch_tushare(symbol, start_str, end_str)
                elif self.primary_source == "baostock":
                    df = self._fetch_baostock(symbol, start_str, end_str)
                elif self.primary_source == "yahoo":
                    df = self._fetch_yahoo(symbol, start_str, end_str)
                else:
                    df = pd.DataFrame()

                # If primary source fails and fallback is enabled, try alternatives
                if df.empty and self.use_fallback:
                    logger.warning(f"Primary source failed for {symbol}, trying fallback sources")
                    df = self._try_fallback_sources(symbol, start_str, end_str)

                if df.empty:
                    logger.warning(f"Could not fetch data for {symbol}")
                    continue

                # Filter for requested timeframe if needed
                if timeframe != "1d":
                    logger.warning(f"Timeframe {timeframe} not supported, using 1d data")

                # Set the correct timeframe
                df["timeframe"] = timeframe

                results[symbol] = df

            except Exception as e:
                logger.error(f"Failed to get data for symbol {symbol}: {str(e)}")
                continue

        return results

    def get_latest_bar(
        self, symbol: str, timeframe: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get the most recent bar for a symbol from real data source.

        Args:
            symbol: Symbol to retrieve
            timeframe: Timeframe string (currently only supports '1d')

        Returns:
            DataFrame with the latest bar data, or None if unavailable
        """
        if not self._initialized:
            self.initialize()

        try:
            # Get data for the last week to ensure we get recent data
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(weeks=1)

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            if self.primary_source == "tushare":
                df = self._fetch_tushare(symbol, start_str, end_str)
            elif self.primary_source == "baostock":
                df = self._fetch_baostock(symbol, start_str, end_str)
            elif self.primary_source == "yahoo":
                df = self._fetch_yahoo(symbol, start_str, end_str)
            else:
                df = pd.DataFrame()

            # If primary source fails and fallback is enabled, try alternatives
            if df.empty and self.use_fallback:
                logger.warning(f"Primary source failed for {symbol}, trying fallback sources")
                df = self._try_fallback_sources(symbol, start_str, end_str)

            if df.empty or df.shape[0] == 0:
                logger.warning(f"No recent data found for {symbol}")
                return None

            # Get the most recent bar
            latest_bar = df.iloc[[-1]].copy()  # Use double brackets to keep as DataFrame

            return latest_bar

        except Exception as e:
            logger.error(f"Failed to get latest bar for symbol {symbol}: {str(e)}")
            return None

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols (currently returns empty as real-time query is expensive)."""
        if not self._initialized:
            self.initialize()

        # This is expensive for real data sources, so we return an empty list
        # In a real implementation, you'd want to cache this or provide a different mechanism
        logger.info("Returning empty symbol list - querying all symbols is expensive for real data sources")
        return []

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for symbols matching the query in real data source.

        Args:
            query: Search query string

        Returns:
            List of matching symbols with metadata
        """
        if not self._initialized:
            self.initialize()

        try:
            # This is complex for real data sources - for now return empty
            # In a real implementation, you'd query the data source's symbol list
            logger.info("Symbol search not implemented for real data sources - too expensive")
            return []

        except Exception as e:
            logger.error(f"Failed to search symbols: {str(e)}")
            return []

    def get_calendar(self) -> List[datetime]:
        """
        Get trading calendar (list of trading days) from real data source.

        Returns:
            List of trading days
        """
        if not self._initialized:
            self.initialize()

        try:
            # For now, return a recent range of trading days
            # In a real implementation, you'd fetch from the actual calendar
            if self.primary_source == "tushare":
                # Get trading calendar from Tushare
                today = datetime.now()
                last_year = today - pd.DateOffset(years=1)

                start_str = last_year.strftime("%Y%m%d")
                end_str = today.strftime("%Y%m%d")

                cal_df = self.pro_api.trade_cal(
                    exchange="", start_date=start_str, end_date=end_str
                )

                # Filter for open days
                open_days = cal_df[cal_df["is_open"] == 1]["cal_date"]
                return [datetime.strptime(date, "%Y%m%d") for date in open_days.tolist()]
            else:
                # For other sources, return recent trading days
                # This is a simplified approach - in real implementation, fetch actual calendar
                trading_days = pd.date_range(
                    start=datetime.now() - pd.DateOffset(months=6),
                    end=datetime.now(),
                    freq='B'  # Business days (Mon-Fri, excluding holidays)
                )
                return [day.to_pydatetime() for day in trading_days]

        except Exception as e:
            logger.error(f"Failed to get trading calendar: {str(e)}")
            return []