"""TuShare data provider implementation."""

import tushare as ts
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import os
from ...domain.interfaces.data_provider import IDataProvider
from ...core.errors import DataProviderError, ConfigurationError
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger


logger = get_logger(__name__)


@registry.register(ComponentType.DATA_PROVIDER, "tushare")
class TuShareProvider(IDataProvider):
    """TuShare data provider implementation."""

    def __init__(self, token: str = None):
        """
        Initialize TuShare provider.

        Args:
            token: TuShare API token. If None, will try to get from TUSHARE_TOKEN environment variable.
        """
        token = token or os.getenv("TUSHARE_TOKEN")
        if not token:
            raise ConfigurationError(
                "TuShare token not provided and TUSHARE_TOKEN environment variable not set"
            )

        self.token = token
        self.pro_api = None

    def initialize(self) -> None:
        """Initialize the TuShare API connection."""
        try:
            ts.set_token(self.token)
            self.pro_api = ts.pro_api()

            # Verify connection by fetching basic info
            df = self.pro_api.trade_cal(
                exchange="", start_date="20230101", end_date="20230102"
            )
            if df.empty:
                raise DataProviderError(
                    "Failed to connect to TuShare API - no data returned"
                )

            logger.info("Successfully connected to TuShare API")
        except Exception as e:
            raise DataProviderError(f"Failed to initialize TuShare provider: {str(e)}")

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        if not self.pro_api:
            self.initialize()

        try:
            # Get stock basic information
            stock_list = self.pro_api.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date",
            )
            return stock_list["ts_code"].tolist()
        except Exception as e:
            raise DataProviderError(f"Failed to get supported symbols: {str(e)}")

    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV bars for the given symbols and timeframe.

        Args:
            symbols: List of symbols to retrieve
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe string ('1m', '5m', '10m', '15m', '30m', '1h', '1d')

        Returns:
            Dictionary mapping symbols to their OHLCV dataframes
        """
        if not self.pro_api:
            self.initialize()

        results = {}

        # Convert dates to string format for TuShare
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        for symbol in symbols:
            try:
                if timeframe == "1d":
                    # Use daily data
                    df = self.pro_api.daily(
                        ts_code=symbol, start_date=start_str, end_date=end_str
                    )

                    # Rename columns to match expected format
                    df.rename(
                        columns={"trade_date": "timestamp", "vol": "volume"},
                        inplace=True,
                    )

                    # Convert timestamp to datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df["timeframe"] = timeframe
                    df["symbol"] = symbol

                    # Reorder columns to match expected format
                    df = df[
                        [
                            "timestamp",
                            "open",
                            "high",
                            "low",
                            "close",
                            "volume",
                            "amount",
                            "timeframe",
                            "symbol",
                        ]
                    ]

                    # Sort by timestamp
                    df.sort_values("timestamp", inplace=True)
                    df.reset_index(drop=True, inplace=True)

                    results[symbol] = df
                elif timeframe in ["1m", "5m", "10m", "15m", "30m", "1h"]:
                    # For Chinese A-shares, TuShare doesn't typically provide intraday data
                    # We'll return an empty dataframe and log a warning
                    logger.warning(f"Minute-level data not available for Chinese A-shares through TuShare: {timeframe}")
                    # Create an empty DataFrame with the expected structure
                    df = pd.DataFrame(columns=[
                        "timestamp", "open", "high", "low", "close", "volume", "amount", "timeframe", "symbol"
                    ])
                    df["timeframe"] = timeframe
                    df["symbol"] = symbol
                    results[symbol] = df
                else:
                    raise NotImplementedError(
                        f"Timeframe {timeframe} not implemented for TuShare provider"
                    )

            except Exception as e:
                logger.error(f"Failed to get data for symbol {symbol}: {str(e)}")
                continue

        return results

    def get_latest_bar(
        self, symbol: str, timeframe: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get the most recent bar for a symbol.

        Args:
            symbol: Symbol to retrieve
            timeframe: Timeframe string (currently only supports '1d')

        Returns:
            DataFrame with the latest bar data, or None if unavailable
        """
        if not self.pro_api:
            self.initialize()

        try:
            # Get today and yesterday's date to ensure we get recent data
            today = datetime.now()
            week_ago = today - timedelta(days=7)

            start_str = week_ago.strftime("%Y%m%d")
            end_str = today.strftime("%Y%m%d")

            if timeframe == "1d":
                df = self.pro_api.daily(
                    ts_code=symbol, start_date=start_str, end_date=end_str
                )

                if df.empty:
                    return None

                # Get the most recent bar
                latest_bar = df.iloc[[0]]  # Get as dataframe, not series

                # Rename columns to match expected format
                latest_bar.rename(
                    columns={"trade_date": "timestamp", "vol": "volume"}, inplace=True
                )

                # Convert timestamp to datetime
                latest_bar["timestamp"] = pd.to_datetime(latest_bar["timestamp"])
                latest_bar["timeframe"] = timeframe
                latest_bar["symbol"] = symbol

                # Reorder columns to match expected format
                latest_bar = latest_bar[
                    [
                        "timestamp",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "amount",
                        "timeframe",
                        "symbol",
                    ]
                ]

                return latest_bar
            else:
                raise NotImplementedError(
                    f"Timeframe {timeframe} not implemented for TuShare provider"
                )

        except Exception as e:
            logger.error(f"Failed to get latest bar for symbol {symbol}: {str(e)}")
            return None

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for symbols matching the query.

        Args:
            query: Search query string

        Returns:
            List of matching symbols with metadata
        """
        if not self.pro_api:
            self.initialize()

        try:
            # Get stock basic information
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
        if not self.pro_api:
            self.initialize()

        try:
            # Get trading calendar for the past year
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
