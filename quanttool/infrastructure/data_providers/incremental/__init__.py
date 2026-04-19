"""Incremental data providers for QuantTool."""

from quanttool.infrastructure.data_providers.incremental.minute_provider import IncrementalMinuteProvider
from quanttool.infrastructure.data_providers.incremental.incremental_provider import IncrementalDataProvider
from quanttool.infrastructure.data_providers.incremental.async_fetcher import AsyncDataFetcher

# 别名，用于未来统一接口
MinuteProvider = IncrementalMinuteProvider

__all__ = [
    'MinuteProvider',
    'IncrementalMinuteProvider',
    'IncrementalDataProvider',
    'AsyncDataFetcher',
]
