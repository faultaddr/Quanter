"""
数据提供者配置

集中管理数据获取的性能参数，包括：
- 超时设置
- 重试策略
- 数据源优先级
- 缓存策略
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
from enum import Enum


class DataSourcePriority(Enum):
    """数据源优先级"""
    ASHARE = 1        # 免费、无需Token、双核心（新浪+腾讯）
    EASTMONEY = 2     # 免费、数据丰富
    AKSHARE = 3       # 免费、接口丰富
    TUSHARE = 4       # 需要Token、有频率限制
    BAOSTOCK = 5      # 免费、稳定、作为最后备选


@dataclass
class RetryConfig:
    """重试配置"""
    max_retries: int = 1           # 最大重试次数（最小化延迟）
    base_delay: float = 0.5        # 基础延迟（秒）
    max_delay: float = 10.0        # 最大延迟（秒）
    exponential_base: float = 2.0  # 指数退避基数


@dataclass
class TimeoutConfig:
    """超时配置"""
    connect_timeout: float = 5.0   # 连接超时（秒）
    read_timeout: float = 10.0     # 读取超时（秒）
    total_timeout: float = 15.0    # 总超时（秒）


@dataclass
class CacheConfig:
    """缓存配置"""
    enabled: bool = True           # 是否启用缓存
    ttl_seconds: int = 60          # 缓存过期时间（秒）
    max_entries: int = 1000        # 最大缓存条目数


@dataclass
class PerformanceConfig:
    """
    性能配置

    性能目标：
    - 缓存命中：P50 < 10ms, P95 < 50ms
    - 数据获取：P50 < 500ms, P95 < 2s
    - 完整分析：P50 < 2s, P95 < 5s
    """
    # 重试配置
    retry: RetryConfig = field(default_factory=RetryConfig)

    # 超时配置
    timeout: TimeoutConfig = field(default_factory=TimeoutConfig)

    # 缓存配置
    cache: CacheConfig = field(default_factory=CacheConfig)

    # 数据源优先级
    priority_order: List[str] = field(default_factory=lambda: [
        "ashare",       # 优先使用 Ashare
        "eastmoney",    # 备选 EastMoney
        "akshare",      # 备选 AkShare
        "tushare",      # 备选 TuShare
        "baostock",     # 最后备选 BaoStock
    ])

    # 并发配置
    max_concurrent_requests: int = 5    # 最大并发请求数
    request_delay_ms: float = 100       # 请求间隔（毫秒）

    # 降级策略
    fallback_on_timeout: bool = True    # 超时时是否降级到下一个数据源
    fallback_on_error: bool = True      # 错误时是否降级到下一个数据源

    def get_timeout_tuple(self) -> tuple:
        """获取 requests 库使用的超时元组"""
        return (self.timeout.connect_timeout, self.timeout.read_timeout)


# 全局默认配置
DEFAULT_CONFIG = PerformanceConfig()


def get_config() -> PerformanceConfig:
    """获取全局配置"""
    return DEFAULT_CONFIG


def configure(
    max_retries: int = None,
    timeout: float = None,
    cache_ttl: int = None,
    priority_order: List[str] = None,
) -> PerformanceConfig:
    """
    配置数据提供者性能参数

    Args:
        max_retries: 最大重试次数
        timeout: 请求超时（秒）
        cache_ttl: 缓存过期时间（秒）
        priority_order: 数据源优先级列表

    Returns:
        更新后的配置对象

    Example:
        from quanttool.infrastructure.data_providers import configure

        # 最小化延迟配置
        configure(
            max_retries=1,
            timeout=10,
            cache_ttl=60,
            priority_order=["ashare", "akshare", "tushare"]
        )
    """
    global DEFAULT_CONFIG

    if max_retries is not None:
        DEFAULT_CONFIG.retry.max_retries = max_retries

    if timeout is not None:
        DEFAULT_CONFIG.timeout.read_timeout = timeout
        DEFAULT_CONFIG.timeout.total_timeout = timeout

    if cache_ttl is not None:
        DEFAULT_CONFIG.cache.ttl_seconds = cache_ttl

    if priority_order is not None:
        DEFAULT_CONFIG.priority_order = priority_order

    return DEFAULT_CONFIG


# 预设配置
class PresetConfigs:
    """预设配置"""

    @staticmethod
    def minimal_latency() -> PerformanceConfig:
        """最小化延迟配置（适合实时场景）"""
        return PerformanceConfig(
            retry=RetryConfig(max_retries=1, base_delay=0.3, max_delay=5.0),
            timeout=TimeoutConfig(connect_timeout=3.0, read_timeout=8.0, total_timeout=10.0),
            cache=CacheConfig(enabled=True, ttl_seconds=30, max_entries=500),
            max_concurrent_requests=10,
            request_delay_ms=50,
        )

    @staticmethod
    def balanced() -> PerformanceConfig:
        """平衡配置（默认）"""
        return PerformanceConfig(
            retry=RetryConfig(max_retries=1, base_delay=0.5, max_delay=10.0),
            timeout=TimeoutConfig(connect_timeout=5.0, read_timeout=10.0, total_timeout=15.0),
            cache=CacheConfig(enabled=True, ttl_seconds=60, max_entries=1000),
            max_concurrent_requests=5,
            request_delay_ms=100,
        )

    @staticmethod
    def reliable() -> PerformanceConfig:
        """高可靠性配置（适合批量数据获取）"""
        return PerformanceConfig(
            retry=RetryConfig(max_retries=3, base_delay=1.0, max_delay=30.0),
            timeout=TimeoutConfig(connect_timeout=10.0, read_timeout=30.0, total_timeout=45.0),
            cache=CacheConfig(enabled=True, ttl_seconds=300, max_entries=2000),
            max_concurrent_requests=3,
            request_delay_ms=200,
        )
