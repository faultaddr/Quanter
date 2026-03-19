"""Data providers package for QuantTool."""

# Import data providers to register them
from . import tushare_provider, ashare_provider, csv_provider, real_data_provider, data_fetcher, incremental_data_provider
from .async_data_fetcher import AsyncDataFetcher, fetch_symbols, fetch_symbols_async
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

# 实时行情数据类型和接口
from .realtime_types import (
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

# 实时数据提供者
from .realtime_data_provider import (
    RealtimeDataProvider,
    RealtimeDataProviderConfig,
    get_realtime_provider,
    get_realtime_quote,
    get_realtime_quotes,
)

# Pytdx 实时行情
from .pytdx_realtime_provider import (
    PytdxRealtimeProvider,
    get_pytdx_provider,
)

# Sina 实时行情
from .sina_realtime_provider import (
    SinaRealtimeProvider,
    get_sina_provider,
)

# 增量分钟数据
from .incremental_minute_provider import (
    IncrementalMinuteProvider,
    get_incremental_minute_provider,
)

__all__ = [
    'tushare_provider',
    'ashare_provider',
    'csv_provider',
    'real_data_provider',
    'data_fetcher',
    'incremental_data_provider',
    'AsyncDataFetcher',
    'fetch_symbols',
    'fetch_symbols_async',
    # Qlib data converter
    'QlibDataConverter',
    'QlibDataConfig',
    'QlibTrainingPipeline',
    'Alpha158Features',
    'Alpha360Features',
    'create_qlib_converter',
    'convert_to_qlib_format',
    # Qlib data loader
    'QlibDataLoader',
    'get_qlib_loader',
    'load_qlib_data',
    # 实时行情类型
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
    # Pytdx 实时行情
    'PytdxRealtimeProvider',
    'get_pytdx_provider',
    # Sina 实时行情
    'SinaRealtimeProvider',
    'get_sina_provider',
    # 增量分钟数据
    'IncrementalMinuteProvider',
    'get_incremental_minute_provider',
]