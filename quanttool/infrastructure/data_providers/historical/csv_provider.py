"""CSV fixture provider for explicit test-mode use."""

import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path
from ....domain.interfaces.data_provider import IDataProvider
from ....core.errors import DataProviderError
from ....core.logging import get_logger
from ....core.runtime import require_test_mode


logger = get_logger(__name__)


class CSVProvider(IDataProvider):
    """CSV mock data provider implementation for testing purposes."""

    def __init__(self, data_dir: str = "./mock_data"):
        """
        Initialize CSV provider.

        Args:
            data_dir: Directory containing CSV files with mock data
        """
        require_test_mode("CSVProvider")
        self.data_dir = Path(data_dir)
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the CSV provider by loading mock data."""
        try:
            if not self.data_dir.exists():
                # Create sample data for demonstration
                self._create_sample_data()

            logger.info(
                f"CSV provider initialized with data directory: {self.data_dir}"
            )
            self._initialized = True
        except Exception as e:
            raise DataProviderError(f"Failed to initialize CSV provider: {str(e)}")

    def _create_sample_data(self):
        """Create sample CSV data files for testing."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Create sample data for some common stocks
        symbols = ["000001.SZ", "000002.SZ", "600000.SH", "600036.SH"]

        for symbol in symbols:
            # Create sample OHLCV data
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
            # Filter out weekends to simulate trading days
            dates = dates[dates.weekday < 5]

            # Generate mock price data
            n_days = len(dates)
            # Start with a base price
            base_price = 100 + (
                hash(symbol) % 100
            )  # Different base price for each symbol

            # Create price movements
            daily_returns = [0]  # Starting return
            for i in range(1, n_days):
                # Random walk with slight upward bias
                change_percent = (
                    hash(f"{symbol}_{i}") % 1000 - 500
                ) / 10000  # -5% to +5%
                daily_returns.append(change_percent)

            close_prices = [base_price]
            for ret in daily_returns[1:]:
                close_prices.append(close_prices[-1] * (1 + ret))

            open_prices = [
                close_prices[0] * (0.99 + 0.02 * (hash(f"o_{symbol}_0") % 100) / 100)
            ]
            high_prices = []
            low_prices = []

            for i in range(1, n_days):
                yesterday_close = close_prices[i - 1]
                today_close = close_prices[i]

                # Calculate open price (slightly different from yesterday's close)
                today_open = yesterday_close * (
                    0.995 + 0.01 * (hash(f"o_{symbol}_{i}") % 100) / 100
                )
                open_prices.append(today_open)

                # High and low prices around open and close
                price_max = max(today_open, today_close)
                price_min = min(today_open, today_close)

                # Add some variation for high and low
                high_variation = 0.02 * (hash(f"h_{symbol}_{i}") % 100) / 100
                low_variation = 0.02 * (hash(f"l_{symbol}_{i}") % 100) / 100

                high_prices.append(price_max * (1 + high_variation))
                low_prices.append(price_min * (1 - low_variation))

            # Add the first high and low values
            first_high = max(open_prices[0], close_prices[0]) * (
                1 + 0.02 * (hash(f"h_{symbol}_0") % 100) / 100
            )
            first_low = min(open_prices[0], close_prices[0]) * (
                1 - 0.02 * (hash(f"l_{symbol}_0") % 100) / 100
            )
            high_prices.insert(0, first_high)
            low_prices.insert(0, first_low)

            # Generate volume data
            volumes = [
                1000000 + 500000 * (hash(f"v_{symbol}_{i}") % 100)
                for i in range(n_days)
            ]

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "timestamp": dates,
                    "open": open_prices,
                    "high": high_prices,
                    "low": low_prices,
                    "close": close_prices,
                    "volume": volumes,
                    "amount": [
                        c * v / 1000000 for c, v in zip(close_prices, volumes)
                    ],  # Simplified amount
                }
            )

            # Save to CSV
            csv_path = self.data_dir / f"{symbol}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Created mock data file: {csv_path}")

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols from available CSV files."""
        if not self._initialized:
            self.initialize()

        symbols = []
        for file_path in self.data_dir.glob("*.csv"):
            symbol = file_path.stem
            symbols.append(symbol)

        return symbols

    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV bars for the given symbols and timeframe from CSV files.

        Args:
            symbols: List of symbols to retrieve
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe string (currently only supports '1d' for CSV provider)

        Returns:
            Dictionary mapping symbols to their OHLCV dataframes
        """
        if not self._initialized:
            self.initialize()

        results = {}

        for symbol in symbols:
            try:
                csv_path = self.data_dir / f"{symbol}.csv"

                if not csv_path.exists():
                    logger.warning(f"No CSV file found for symbol {symbol}")
                    continue

                # Read the CSV file
                df = pd.read_csv(csv_path)

                # Convert timestamp to datetime
                df["timestamp"] = pd.to_datetime(df["timestamp"])

                # Filter by date range
                mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
                df = df.loc[mask].copy()

                if df.empty:
                    logger.info(
                        f"No data found for symbol {symbol} in date range {start_date} to {end_date}"
                    )
                    continue

                # Add timeframe and symbol columns
                df["timeframe"] = timeframe
                df["symbol"] = symbol

                # Reorder columns to match expected format
                cols_order = [
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
                df = df[cols_order]

                # Sort by timestamp
                df.sort_values("timestamp", inplace=True)
                df.reset_index(drop=True, inplace=True)

                results[symbol] = df

            except Exception as e:
                logger.error(f"Failed to get data for symbol {symbol}: {str(e)}")
                continue

        return results

    def get_latest_bar(
        self, symbol: str, timeframe: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get the most recent bar for a symbol from CSV file.

        Args:
            symbol: Symbol to retrieve
            timeframe: Timeframe string (currently only supports '1d')

        Returns:
            DataFrame with the latest bar data, or None if unavailable
        """
        if not self._initialized:
            self.initialize()

        try:
            csv_path = self.data_dir / f"{symbol}.csv"

            if not csv_path.exists():
                logger.warning(f"No CSV file found for symbol {symbol}")
                return None

            # Read the CSV file
            df = pd.read_csv(csv_path)

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Get the most recent bar
            latest_bar = df.iloc[
                [-1]
            ].copy()  # Use iloc with double brackets to keep as DataFrame

            # Add timeframe and symbol columns
            latest_bar["timeframe"] = timeframe
            latest_bar["symbol"] = symbol

            # Reorder columns to match expected format
            cols_order = [
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
            latest_bar = latest_bar[cols_order]

            return latest_bar

        except Exception as e:
            logger.error(f"Failed to get latest bar for symbol {symbol}: {str(e)}")
            return None

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for symbols matching the query in CSV files.

        Args:
            query: Search query string

        Returns:
            List of matching symbols with metadata
        """
        if not self._initialized:
            self.initialize()

        all_symbols = self.get_supported_symbols()

        results = []
        for symbol in all_symbols:
            if query.lower() in symbol.lower():
                # In a real implementation, we'd have more metadata
                results.append(
                    {
                        "symbol": symbol,
                        "name": f"Mock {symbol}",  # Placeholder name
                        "area": "Mock Area",
                        "industry": "Mock Industry",
                    }
                )

        return results

    def get_calendar(self) -> List[datetime]:
        """
        Get trading calendar (list of trading days) from available CSV data.

        Returns:
            List of trading days
        """
        if not self._initialized:
            self.initialize()

        all_dates = set()

        # Collect all dates from all CSV files
        for file_path in self.data_dir.glob("*.csv"):
            df = pd.read_csv(file_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            dates = df["timestamp"].dt.date.unique()
            all_dates.update(dates)

        # Convert to datetime objects and sort
        trading_days = [
            datetime.combine(date, datetime.min.time()) for date in sorted(all_dates)
        ]

        return trading_days
