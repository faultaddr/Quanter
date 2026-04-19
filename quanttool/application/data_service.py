"""Data service for QuantTool."""

from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from ..domain.interfaces.data_provider import IDataProvider
from ..domain.interfaces.store import IStore
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger
from ..infrastructure.data_providers.incremental.manager import IncrementalDataManager, DataType


logger = get_logger(__name__)


class DataService:
    """Service class for managing data operations."""

    def __init__(self, use_incremental: bool = True):
        """Initialize data service.

        Args:
            use_incremental: 是否使用增量数据获取
        """
        self.data_providers: Dict[str, IDataProvider] = {}
        self.store: Optional[IStore] = None
        self.use_incremental = use_incremental
        self._incremental_manager: Optional[IncrementalDataManager] = None
        if use_incremental:
            try:
                self._incremental_manager = IncrementalDataManager()
                logger.info("增量数据管理器初始化成功")
            except Exception as e:
                logger.warning(f"增量数据管理器初始化失败: {e}")
                self._incremental_manager = None

    def _get_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        provider,
        timeframe: str = "1d",
    ) -> pd.DataFrame:
        """获取股票数据（优先使用增量数据管理器）"""
        # 优先使用增量数据管理器
        if self._incremental_manager:
            try:
                df = self._incremental_manager.get_data(
                    symbol,
                    start_date,
                    end_date,
                    provider,
                    data_type=DataType.STOCK_BAR,
                )
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"增量获取失败 {symbol}: {e}，回退到直接获取")

        # 回退到直接获取
        data = provider.get_bars([symbol], start_date, end_date, timeframe)
        return data.get(symbol, pd.DataFrame())

    def register_data_provider(self, name: str, provider: IDataProvider):
        """
        Register a data provider.

        Args:
            name: Name of the provider
            provider: Instance of the data provider
        """
        self.data_providers[name] = provider
        logger.info(f"Registered data provider: {name}")

    def get_data_provider(self, name: str) -> IDataProvider:
        """
        Get a registered data provider.

        Args:
            name: Name of the provider

        Returns:
            Instance of the data provider
        """
        if name not in self.data_providers:
            # Try to instantiate from registry
            provider_class = registry.get(ComponentType.DATA_PROVIDER, name)
            provider = provider_class()
            if hasattr(provider, "initialize"):
                provider.initialize()
            self.data_providers[name] = provider

        return self.data_providers[name]

    def set_store(self, store: IStore):
        """
        Set the data store.

        Args:
            store: Instance of the data store
        """
        self.store = store

    def pull_data(
        self,
        provider_name: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
        save_to_store: bool = True,
    ) -> Dict[str, Any]:
        """
        Pull data from a provider and optionally save to store.

        Args:
            provider_name: Name of the data provider to use
            symbols: List of symbols to retrieve
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe for the data
            save_to_store: Whether to save data to the configured store

        Returns:
            Dictionary with the retrieved data and metadata
        """
        provider = self.get_data_provider(provider_name)

        logger.info(
            f"Pulling data for {len(symbols)} symbols from {provider_name} "
            f"for timeframe {timeframe} from {start_date} to {end_date}"
        )

        # Get the data (使用增量数据管理器)
        data = {}
        for symbol in symbols:
            df = self._get_data(symbol, start_date, end_date, provider, timeframe)
            if not df.empty:
                data[symbol] = df

        # Optionally save to store
        if save_to_store and self.store:
            for symbol, df in data.items():
                key = f"data/{timeframe}/{symbol}/{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"

                # Create metadata
                metadata = {
                    "provider": provider_name,
                    "timeframe": timeframe,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "symbol": symbol,
                    "num_rows": len(df),
                    "fetched_at": datetime.now().isoformat(),
                }

                # Save to store
                success = self.store.save_data(key, df, metadata)
                if success:
                    logger.info(f"Saved data for {symbol} to store with key: {key}")
                else:
                    logger.warning(f"Failed to save data for {symbol} to store")

        return {
            "data": data,
            "provider": provider_name,
            "symbols": symbols,
            "timeframe": timeframe,
            "start_date": start_date,
            "end_date": end_date,
            "retrieved_at": datetime.now(),
        }

    def get_calendar(self, provider_name: str) -> List[datetime]:
        """
        Get the trading calendar from a provider.

        Args:
            provider_name: Name of the data provider

        Returns:
            List of trading days
        """
        provider = self.get_data_provider(provider_name)
        return provider.get_calendar()

    def search_symbols(self, provider_name: str, query: str) -> List[Dict[str, Any]]:
        """
        Search for symbols using a provider.

        Args:
            provider_name: Name of the data provider
            query: Search query

        Returns:
            List of matching symbols with metadata
        """
        provider = self.get_data_provider(provider_name)
        return provider.search_symbols(query)

    def get_latest_bar(
        self, provider_name: str, symbol: str, timeframe: str = "1d"
    ) -> Optional[Any]:
        """
        Get the latest bar for a symbol from a provider.

        Args:
            provider_name: Name of the data provider
            symbol: Symbol to retrieve
            timeframe: Timeframe for the data

        Returns:
            Latest bar data or None if not available
        """
        provider = self.get_data_provider(provider_name)
        return provider.get_latest_bar(symbol, timeframe)
