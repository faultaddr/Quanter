"""Data providers package for QuantTool.

重构后的模块结构：
- historical/   : 历史数据提供者
- realtime/     : 实时数据提供者
- incremental/  : 增量/分钟数据提供者
"""

import warnings

# ============================================================================
# 新的模块结构导入
# ============================================================================

# 历史数据提供者
from .historical.ashare_provider import AShareProvider
from .historical.tushare_provider import TuShareProvider
from .historical.csv_provider import CSVProvider
from .historical.enhanced_fetcher import EnhancedDataFetcher, AshareFetcher

# 实时数据提供者
from .realtime.types import (
    RealtimeQuote,
    RealtimeSource,
    MinuteBar,
    CircuitBreaker,
    get_realtime_circuit_breaker,
    safe_float,
    safe_int,
    normalize_symbol,
    is_etf_code,
    is_index_code,
)
from .realtime.realtime_provider import (
    RealtimeDataProvider,
    RealtimeDataProviderConfig,
    get_realtime_provider,
    get_realtime_quote,
    get_realtime_quotes,
)
from .realtime.sina_source import (
    SinaRealtimeProvider,
    get_sina_provider,
)
from .realtime.pytdx_source import (
    PytdxRealtimeProvider,
    get_pytdx_provider,
)

# 增量数据提供者
from .incremental import MinuteProvider, IncrementalMinuteProvider
from .incremental.minute_provider import get_incremental_minute_provider
from .incremental.incremental_provider import IncrementalDataProvider
from .incremental.async_fetcher import AsyncDataFetcher, fetch_symbols, fetch_symbols_async

# Qlib 数据处理（保留在原位置）
from .qlib_data_converter import (
    QlibDataConverter,
    QlibDataConfig,
    QlibTrainingPipeline,
    Alpha158Features,
    Alpha360Features,
    create_qlib_converter,
    convert_to_qlib_format,
)
from .qlib_data_loader import (
    QlibDataLoader,
    get_qlib_loader,
    load_qlib_data,
)

# ============================================================================
# 向后兼容 - 模块别名（带 deprecation warning）
# ============================================================================

def __getattr__(name):
    """向后兼容的模块访问，带 deprecation warning"""
    if name == 'tushare_provider':
        warnings.warn(
            "直接访问 tushare_provider 模块已弃用，请使用 from quanttool.infrastructure.data_providers.historical import TuShareProvider",
            DeprecationWarning,
            stacklevel=2
        )
        return __import__('quanttool.infrastructure.data_providers.historical.tushare_provider', fromlist=[name])
    if name == 'ashare_provider':
        warnings.warn(
            "直接访问 ashare_provider 模块已弃用，请使用 from quanttool.infrastructure.data_providers.historical import AShareProvider",
            DeprecationWarning,
            stacklevel=2
        )
        return __import__('quanttool.infrastructure.data_providers.historical.ashare_provider', fromlist=[name])
    if name == 'csv_provider':
        warnings.warn(
            "直接访问 csv_provider 模块已弃用，请使用 from quanttool.infrastructure.data_providers.historical import CSVProvider",
            DeprecationWarning,
            stacklevel=2
        )
        return __import__('quanttool.infrastructure.data_providers.historical.csv_provider', fromlist=[name])
    if name == 'data_fetcher':
        warnings.warn(
            "直接访问 data_fetcher 模块已弃用，请使用 from quanttool.infrastructure.data_providers.historical import EnhancedDataFetcher",
            DeprecationWarning,
            stacklevel=2
        )
        return __import__('quanttool.infrastructure.data_providers.historical.enhanced_fetcher', fromlist=[name])
    if name == 'incremental_data_provider':
        warnings.warn(
            "直接访问 incremental_data_provider 模块已弃用，请使用 from quanttool.infrastructure.data_providers.incremental import IncrementalDataProvider",
            DeprecationWarning,
            stacklevel=2
        )
        return __import__('quanttool.infrastructure.data_providers.incremental.incremental_provider', fromlist=[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # 历史数据提供者
    'AShareProvider',
    'TuShareProvider',
    'CSVProvider',
    'EnhancedDataFetcher',
    'AshareFetcher',
    # 实时数据类型
    'RealtimeQuote',
    'RealtimeSource',
    'MinuteBar',
    'CircuitBreaker',
    'get_realtime_circuit_breaker',
    'safe_float',
    'safe_int',
    'normalize_symbol',
    'is_etf_code',
    'is_index_code',
    # 实时数据提供者
    'RealtimeDataProvider',
    'RealtimeDataProviderConfig',
    'get_realtime_provider',
    'get_realtime_quote',
    'get_realtime_quotes',
    'SinaRealtimeProvider',
    'get_sina_provider',
    'PytdxRealtimeProvider',
    'get_pytdx_provider',
    # 增量数据提供者
    'MinuteProvider',
    'IncrementalMinuteProvider',
    'get_incremental_minute_provider',
    'IncrementalDataProvider',
    'AsyncDataFetcher',
    'fetch_symbols',
    'fetch_symbols_async',
    # Qlib 数据处理
    'QlibDataConverter',
    'QlibDataConfig',
    'QlibTrainingPipeline',
    'Alpha158Features',
    'Alpha360Features',
    'create_qlib_converter',
    'convert_to_qlib_format',
    'QlibDataLoader',
    'get_qlib_loader',
    'load_qlib_data',
    # 向后兼容（已弃用）
    'tushare_provider',
    'ashare_provider',
    'csv_provider',
    'data_fetcher',
    'incremental_data_provider',
]
