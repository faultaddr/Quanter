# -*- coding: utf-8 -*-
"""
===================================
RealtimeDataProvider - 统一实时数据接口
===================================

整合多个实时数据源，按优先级自动切换：
1. Pytdx（通达信直连）- 最快，秒级
2. Sina（新浪财经）- 备用，支持批量

设计目标：
- 统一接口，屏蔽底层差异
- 自动故障切换
- 缓存优化
- 异步支持
"""

import logging
import time
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

import pandas as pd

from .types import (
    RealtimeQuote,
    RealtimeSource,
    MinuteBar,
    CircuitBreaker,
    get_realtime_circuit_breaker,
    safe_float,
    safe_int,
)

logger = logging.getLogger(__name__)


@dataclass
class RealtimeDataProviderConfig:
    """实时数据提供者配置"""
    cache_ttl: int = 3                    # 缓存过期时间(秒)
    enable_pytdx: bool = True             # 启用 Pytdx
    enable_sina: bool = True              # 启用 Sina
    enable_akshare: bool = True           # 启用 AkShare（分钟数据）
    max_concurrent: int = 20              # 并发数


class RealtimeDataProvider:
    """
    统一实时数据提供者

    数据源优先级：
    1. Pytdx（通达信直连）- 秒级，支持五档
    2. Sina（新浪财经）- 批量查询，备用

    使用示例：
        provider = RealtimeDataProvider()

        # 单只股票
        quote = provider.get_realtime_quote("600519")

        # 批量获取
        quotes = provider.get_realtime_quotes(["600519", "000001"])

        # 分钟K线
        df = provider.get_minute_bars("600519", period="5m", count=100)
    """

    def __init__(self, config: Optional[RealtimeDataProviderConfig] = None):
        """
        初始化统一实时数据提供者

        Args:
            config: 配置对象
        """
        self._config = config or RealtimeDataProviderConfig()

        # 数据源实例（延迟加载）
        self._pytdx_provider = None
        self._sina_provider = None
        self._akshare_provider = None

        # 全局缓存
        self._quote_cache: Dict[str, tuple] = {}
        self._minute_cache: Dict[str, tuple] = {}

        # 熔断器
        self._circuit_breaker = get_realtime_circuit_breaker()

    def _get_pytdx_provider(self):
        """获取 Pytdx 提供者（延迟加载）"""
        if self._pytdx_provider is None and self._config.enable_pytdx:
            try:
                from .pytdx_realtime_provider import PytdxRealtimeProvider
                self._pytdx_provider = PytdxRealtimeProvider(
                    cache_ttl=self._config.cache_ttl
                )
                logger.info("Pytdx 实时行情提供者已加载")
            except ImportError:
                logger.warning("Pytdx 不可用，请安装: pip install pytdx")
        return self._pytdx_provider

    def _get_sina_provider(self):
        """获取 Sina 提供者（延迟加载）"""
        if self._sina_provider is None and self._config.enable_sina:
            from .sina_realtime_provider import SinaRealtimeProvider
            self._sina_provider = SinaRealtimeProvider(
                cache_ttl=self._config.cache_ttl
            )
            logger.info("Sina 实时行情提供者已加载")
        return self._sina_provider

    def _get_akshare_provider(self):
        """获取 AkShare 提供者（延迟加载）"""
        if self._akshare_provider is None and self._config.enable_akshare:
            try:
                from .akshare_minute_provider import AkShareMinuteProvider
                self._akshare_provider = AkShareMinuteProvider()
                logger.info("AkShare 分钟数据提供者已加载")
            except ImportError:
                logger.warning("AkShare 不可用，请安装: pip install akshare")
        return self._akshare_provider

    # ==========================================
    # 实时行情接口
    # ==========================================

    def get_realtime_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """
        获取单只股票实时行情

        按优先级尝试数据源：
        1. Pytdx（最快）
        2. Sina（备用）

        Args:
            symbol: 股票代码

        Returns:
            RealtimeQuote 或 None
        """
        # 检查缓存
        cache_key = f"quote_{symbol}"
        if cache_key in self._quote_cache:
            data, timestamp = self._quote_cache[cache_key]
            if time.time() - timestamp < self._config.cache_ttl:
                return data

        # 尝试 Pytdx
        pytdx = self._get_pytdx_provider()
        if pytdx and pytdx.is_available():
            quote = pytdx.get_realtime_quote(symbol)
            if quote and quote.has_basic_data():
                self._quote_cache[cache_key] = (quote, time.time())
                return quote

        # 尝试 Sina
        sina = self._get_sina_provider()
        if sina and sina.is_available():
            quote = sina.get_realtime_quote(symbol)
            if quote and quote.has_basic_data():
                self._quote_cache[cache_key] = (quote, time.time())
                return quote

        logger.warning(f"所有数据源均无法获取 {symbol} 实时行情")
        return None

    def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, RealtimeQuote]:
        """
        批量获取实时行情（推荐使用）

        自动选择最优数据源进行批量查询

        Args:
            symbols: 股票代码列表

        Returns:
            {symbol: RealtimeQuote} 字典
        """
        if not symbols:
            return {}

        # 检查缓存，收集未缓存的股票
        cached_results = {}
        uncached_symbols = []
        current_time = time.time()

        for symbol in symbols:
            cache_key = f"quote_{symbol}"
            if cache_key in self._quote_cache:
                data, timestamp = self._quote_cache[cache_key]
                if current_time - timestamp < self._config.cache_ttl:
                    cached_results[symbol] = data
                    continue
            uncached_symbols.append(symbol)

        if not uncached_symbols:
            return cached_results

        # 批量获取未缓存的股票
        results = dict(cached_results)

        # 优先使用 Pytdx（速度最快）
        pytdx = self._get_pytdx_provider()
        if pytdx and pytdx.is_available():
            pytdx_results = pytdx.get_realtime_quotes(uncached_symbols)
            if pytdx_results:
                results.update(pytdx_results)
                # 更新缓存
                for symbol, quote in pytdx_results.items():
                    self._quote_cache[f"quote_{symbol}"] = (quote, current_time)

                # 检查是否有遗漏
                missing = [s for s in uncached_symbols if s not in results]
                if not missing:
                    return results
                uncached_symbols = missing

        # 使用 Sina 补充
        sina = self._get_sina_provider()
        if sina and sina.is_available():
            sina_results = sina.get_realtime_quotes(uncached_symbols)
            if sina_results:
                results.update(sina_results)
                for symbol, quote in sina_results.items():
                    self._quote_cache[f"quote_{symbol}"] = (quote, current_time)

        return results

    async def get_realtime_quotes_async(
        self,
        symbols: List[str]
    ) -> Dict[str, RealtimeQuote]:
        """
        异步批量获取实时行情

        Args:
            symbols: 股票代码列表

        Returns:
            {symbol: RealtimeQuote} 字典
        """
        # 优先使用 Sina 异步接口
        sina = self._get_sina_provider()
        if sina and sina.is_available():
            return await sina.get_realtime_quotes_async(symbols)

        # 降级到同步
        return self.get_realtime_quotes(symbols)

    # ==========================================
    # 分钟K线接口
    # ==========================================

    def get_minute_bars(
        self,
        symbol: str,
        period: str = '5m',
        count: int = 100,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> pd.DataFrame:
        """
        获取分钟K线数据

        Args:
            symbol: 股票代码
            period: 周期 (1m, 5m, 15m, 30m, 60m)
            count: 获取数量
            start_time: 开始时间（可选）
            end_time: 结束时间（可选）

        Returns:
            DataFrame，包含 open/high/low/close/volume/amount 列
        """
        # 检查缓存
        cache_key = f"minute_{symbol}_{period}"
        if cache_key in self._minute_cache:
            df, timestamp = self._minute_cache[cache_key]
            # 分钟数据缓存 60 秒
            if time.time() - timestamp < 60:
                if count > 0:
                    return df.tail(count).reset_index(drop=True)
                return df.copy()

        # 优先使用 Pytdx
        pytdx = self._get_pytdx_provider()
        if pytdx and pytdx.is_available():
            bars = pytdx.get_minute_bars(symbol, period, count)
            if bars:
                df = pd.DataFrame(bars)
                self._minute_cache[cache_key] = (df, time.time())
                return df

        # 使用 AkShare
        akshare = self._get_akshare_provider()
        if akshare:
            try:
                df = akshare.get_latest_bars(symbol, count, period)
                if not df.empty:
                    self._minute_cache[cache_key] = (df, time.time())
                    return df
            except Exception as e:
                logger.warning(f"AkShare 获取分钟数据失败 {symbol}: {e}")

        return pd.DataFrame()

    def get_latest_bars(
        self,
        symbol: str,
        period: str = '5m',
        count: int = 60
    ) -> pd.DataFrame:
        """
        获取最近N根K线

        Args:
            symbol: 股票代码
            period: 周期
            count: 数量

        Returns:
            DataFrame
        """
        return self.get_minute_bars(symbol, period, count)

    # ==========================================
    # 便捷方法
    # ==========================================

    def get_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        quote = self.get_realtime_quote(symbol)
        return quote.price if quote else None

    def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """批量获取最新价格"""
        quotes = self.get_realtime_quotes(symbols)
        return {
            symbol: quote.price
            for symbol, quote in quotes.items()
            if quote and quote.price is not None
        }

    def get_change_pct(self, symbol: str) -> Optional[float]:
        """获取涨跌幅"""
        quote = self.get_realtime_quote(symbol)
        return quote.change_pct if quote else None

    def get_volume(self, symbol: str) -> Optional[int]:
        """获取成交量"""
        quote = self.get_realtime_quote(symbol)
        return quote.volume if quote else None

    # ==========================================
    # 状态管理
    # ==========================================

    def get_status(self) -> Dict[str, Any]:
        """获取提供者状态"""
        status = {
            'circuit_breaker': self._circuit_breaker.get_status(),
            'cache_size': len(self._quote_cache),
            'providers': {}
        }

        pytdx = self._get_pytdx_provider()
        if pytdx:
            status['providers']['pytdx'] = pytdx.get_status()

        sina = self._get_sina_provider()
        if sina:
            status['providers']['sina'] = sina.get_status()

        return status

    def clear_cache(self) -> None:
        """清空缓存"""
        self._quote_cache.clear()
        self._minute_cache.clear()
        logger.info("实时数据缓存已清空")

    def reset_circuit_breaker(self, source: str = None) -> None:
        """重置熔断器"""
        self._circuit_breaker.reset(source)


# 全局实例
_realtime_provider_instance: Optional[RealtimeDataProvider] = None


def get_realtime_provider() -> RealtimeDataProvider:
    """获取全局 RealtimeDataProvider 实例"""
    global _realtime_provider_instance
    if _realtime_provider_instance is None:
        _realtime_provider_instance = RealtimeDataProvider()
    return _realtime_provider_instance


def get_realtime_quote(symbol: str) -> Optional[RealtimeQuote]:
    """便捷函数：获取单只股票实时行情"""
    return get_realtime_provider().get_realtime_quote(symbol)


def get_realtime_quotes(symbols: List[str]) -> Dict[str, RealtimeQuote]:
    """便捷函数：批量获取实时行情"""
    return get_realtime_provider().get_realtime_quotes(symbols)
