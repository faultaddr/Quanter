"""Async data fetcher for high-performance concurrent data retrieval."""

import asyncio
import aiohttp
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional, Any
from pathlib import Path
import json

from ...core.logging import get_logger
from ..cache import LocalDataCache

logger = get_logger(__name__)


class AsyncDataFetcher:
    """
    Asynchronous data fetcher using aiohttp for concurrent requests.

    Features:
    - Concurrent HTTP requests with configurable limit
    - Automatic retry on failures
    - Connection pooling for efficiency
    - Cache integration

    Usage:
        async with AsyncDataFetcher(max_concurrent=20) as fetcher:
            data = await fetcher.fetch_all(symbols, start_date, end_date)
    """

    # Data source configurations
    SOURCES = {
        "sina": {
            "url_template": "http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale=240&ma=5&datalen={count}",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "http://finance.sina.com.cn"
            }
        },
        "tencent": {
            "url_template": "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,{end_date},{count},qfq",
            "headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "http://gu.qq.com"
            }
        }
    }

    def __init__(
        self,
        max_concurrent: int = 20,
        timeout: int = 30,
        retry_count: int = 3,
        retry_delay: float = 0.5,
        cache_dir: Optional[str] = ".cache/stock_data",
        cache_ttl: int = 86400
    ):
        """
        Initialize async data fetcher.

        Args:
            max_concurrent: Maximum concurrent requests (default: 20)
            timeout: Request timeout in seconds (default: 30)
            retry_count: Number of retries on failure (default: 3)
            retry_delay: Delay between retries in seconds (default: 0.5)
            cache_dir: Cache directory path (default: .cache/stock_data)
            cache_ttl: Cache TTL in seconds (default: 86400)
        """
        self.max_concurrent = max_concurrent
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.semaphore: Optional[asyncio.Semaphore] = None
        self.session: Optional[aiohttp.ClientSession] = None

        # Cache
        if cache_dir:
            self._cache = LocalDataCache(cache_dir=cache_dir, default_ttl=cache_ttl)
        else:
            self._cache = None

    async def __aenter__(self):
        """Initialize async context."""
        self.semaphore = asyncio.Semaphore(self.max_concurrent)
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=5,
            enable_cleanup_closed=True
        )
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            connector=connector
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup async context."""
        if self.session:
            await self.session.close()
        if self._cache:
            self._cache.close()

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize stock code to sina/tencent format."""
        code = code.replace('.XSHG', '').replace('.XSHE', '')
        code = code.replace('.SH', '').replace('.SZ', '')

        if code.startswith(('sh', 'sz', 'SH', 'SZ')):
            return code.lower()

        if code.startswith(('5', '6', '9')):
            return f'sh{code}'
        else:
            return f'sz{code}'

    @staticmethod
    def _parse_sina_data(data: str) -> pd.DataFrame:
        """Parse Sina API response to DataFrame."""
        try:
            parsed = json.loads(data)
            if not parsed:
                return pd.DataFrame()

            df = pd.DataFrame(parsed, columns=['day', 'open', 'high', 'low', 'close', 'volume'])
            df['day'] = pd.to_datetime(df['day'])
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['amount'] = df['close'] * df['volume'] * 100
            df = df.rename(columns={'day': 'timestamp'})

            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def _parse_tencent_data(data: str, symbol: str) -> pd.DataFrame:
        """Parse Tencent API response to DataFrame."""
        try:
            parsed = json.loads(data)
            stock_data = parsed.get('data', {}).get(symbol, {})

            kline_data = stock_data.get('qfqday') or stock_data.get('day', [])
            if not kline_data:
                return pd.DataFrame()

            df = pd.DataFrame(kline_data, columns=['timestamp', 'open', 'close', 'high', 'low', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['open'] = df['open'].astype(float)
            df['close'] = df['close'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['amount'] = df['close'] * df['volume'] * 100

            return df
        except Exception:
            return pd.DataFrame()

    async def _fetch_with_retry(
        self,
        url: str,
        headers: Dict[str, str]
    ) -> Optional[str]:
        """Fetch URL with retry logic."""
        for attempt in range(self.retry_count):
            try:
                async with self.semaphore:
                    async with self.session.get(url, headers=headers) as response:
                        if response.status == 200:
                            return await response.text()
                        elif response.status == 429:
                            # Rate limited - wait longer
                            await asyncio.sleep(self.retry_delay * (attempt + 2))
                        else:
                            logger.warning(f"HTTP {response.status} for {url}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout for {url}, attempt {attempt + 1}")
            except aiohttp.ClientError as e:
                logger.warning(f"Client error for {url}: {e}")

            if attempt < self.retry_count - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        return None

    async def fetch_single(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        count: int = 500
    ) -> pd.DataFrame:
        """
        Fetch data for a single symbol asynchronously.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            count: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data
        """
        # Check cache first
        if self._cache:
            cached = self._cache.get(symbol, start_date, end_date, "1d")
            if cached is not None and not cached.empty:
                return cached

        code = self._normalize_code(symbol)

        # Try Sina first
        source = self.SOURCES["sina"]
        url = source["url_template"].format(code=code, count=count)
        data = await self._fetch_with_retry(url, source["headers"])

        if data:
            df = self._parse_sina_data(data)
            if not df.empty:
                # Filter by date range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
                df['symbol'] = symbol
                df['timeframe'] = '1d'

                # Cache result
                if self._cache and not df.empty:
                    self._cache.set(symbol, start_date, end_date, df)

                return df

        # Fallback to Tencent
        source = self.SOURCES["tencent"]
        end_fmt = end_date.replace('-', '')
        url = source["url_template"].format(code=code, end_date=end_fmt, count=count)
        data = await self._fetch_with_retry(url, source["headers"])

        if data:
            df = self._parse_tencent_data(data, code)
            if not df.empty:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
                df['symbol'] = symbol
                df['timeframe'] = '1d'

                if self._cache and not df.empty:
                    self._cache.set(symbol, start_date, end_date, df)

                return df

        logger.warning(f"No data fetched for {symbol}")
        return pd.DataFrame()

    async def fetch_all(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        count: int = 500,
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols concurrently.

        Args:
            symbols: List of stock symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            count: Number of bars to fetch per symbol
            show_progress: Whether to log progress

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        tasks = [
            self.fetch_single(symbol, start_date, end_date, count)
            for symbol in symbols
        ]

        results = {}
        total = len(symbols)
        completed = 0

        for coro in asyncio.as_completed(tasks):
            try:
                df = await coro
                if not df.empty and 'symbol' in df.columns:
                    symbol = df['symbol'].iloc[0]
                    results[symbol] = df

                completed += 1
                if show_progress and completed % 20 == 0:
                    logger.info(f"Async fetch progress: {completed}/{total}")

            except Exception as e:
                logger.error(f"Error in async fetch: {e}")
                completed += 1

        logger.info(f"Async fetch completed: {len(results)}/{total} symbols")
        return results


async def fetch_symbols_async(
    symbols: List[str],
    start_date: str,
    end_date: str,
    max_concurrent: int = 20
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch multiple symbols asynchronously.

    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_concurrent: Maximum concurrent requests

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    async with AsyncDataFetcher(max_concurrent=max_concurrent) as fetcher:
        return await fetcher.fetch_all(symbols, start_date, end_date)


def fetch_symbols(
    symbols: List[str],
    start_date: str,
    end_date: str,
    max_concurrent: int = 20
) -> Dict[str, pd.DataFrame]:
    """
    Synchronous wrapper for async fetching.

    Args:
        symbols: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        max_concurrent: Maximum concurrent requests

    Returns:
        Dictionary mapping symbols to DataFrames
    """
    return asyncio.run(
        fetch_symbols_async(symbols, start_date, end_date, max_concurrent)
    )