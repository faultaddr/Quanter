# -*- coding: utf-8 -*-
"""
===================================
IncrementalMinuteProvider - 增量分钟数据获取
===================================

设计目标：
1. 避免重复获取全量分钟数据
2. 增量更新机制
3. 多级缓存策略
4. 支持 Pytdx 和 AkShare 数据源

缓存策略：
- 内存缓存：最近访问的股票数据（LRU）
- 本地文件缓存：历史分钟数据
- TTL：交易日盘中 60 秒，盘后永久
"""

import logging
import time
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from pathlib import Path
from collections import OrderedDict
import threading

import pandas as pd
import numpy as np

from .realtime_types import safe_float, safe_int, normalize_symbol

logger = logging.getLogger(__name__)


class LRUCache:
    """线程安全的 LRU 缓存"""

    def __init__(self, max_size: int = 100):
        self._max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                # 移到末尾（最近使用）
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            # 超出容量时删除最旧的
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


class IncrementalMinuteProvider:
    """
    增量分钟数据提供者

    特点：
    - 增量获取，避免重复请求
    - 多级缓存（内存 + 文件）
    - 自动合并数据
    - 支持 Pytdx 和 AkShare

    使用示例：
        provider = IncrementalMinuteProvider()

        # 第一次获取（全量）
        df1 = provider.get_minute_bars("600519", period="5m", count=100)

        # 第二次获取（增量）
        df2 = provider.get_minute_bars("600519", period="5m", count=100)
    """

    # 支持的周期
    SUPPORTED_PERIODS = ['1m', '5m', '15m', '30m', '60m']

    def __init__(
        self,
        cache_dir: str = ".cache/minute_data",
        memory_cache_size: int = 100,
        cache_ttl_seconds: int = 60,
        enable_file_cache: bool = True,
    ):
        """
        初始化增量分钟数据提供者

        Args:
            cache_dir: 文件缓存目录
            memory_cache_size: 内存缓存大小（股票数量）
            cache_ttl_seconds: 缓存 TTL（秒）
            enable_file_cache: 是否启用文件缓存
        """
        self._cache_dir = Path(cache_dir)
        self._cache_ttl = cache_ttl_seconds
        self._enable_file_cache = enable_file_cache

        # 内存缓存
        self._memory_cache = LRUCache(max_size=memory_cache_size)
        # 元数据缓存 {cache_key: (last_time, timestamp)}
        self._metadata_cache: Dict[str, tuple] = {}

        # 数据源
        self._pytdx_provider = None
        self._akshare_provider = None

        # 创建缓存目录
        if self._enable_file_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_pytdx_provider(self):
        """获取 Pytdx 提供者"""
        if self._pytdx_provider is None:
            try:
                from .pytdx_realtime_provider import PytdxRealtimeProvider
                self._pytdx_provider = PytdxRealtimeProvider()
            except ImportError:
                logger.debug("Pytdx 不可用")
        return self._pytdx_provider

    def _get_akshare_provider(self):
        """获取 AkShare 提供者"""
        if self._akshare_provider is None:
            try:
                from .akshare_minute_provider import AkShareMinuteProvider
                self._akshare_provider = AkShareMinuteProvider()
            except ImportError:
                logger.debug("AkShare 不可用")
        return self._akshare_provider

    def _get_cache_key(self, symbol: str, period: str) -> str:
        """生成缓存键"""
        code, _ = normalize_symbol(symbol)
        return f"{code}_{period}"

    def _get_file_path(self, symbol: str, period: str) -> Path:
        """获取文件缓存路径"""
        code, market = normalize_symbol(symbol)
        market_str = 'sh' if market == 1 else 'sz'
        return self._cache_dir / f"{market_str}{code}_{period}.parquet"

    def _is_trading_time(self) -> bool:
        """判断是否在交易时间"""
        now = datetime.now()
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()

        # 周末
        if weekday >= 5:
            return False

        # 上午 9:30 - 11:30
        if 9 <= hour < 12:
            if hour == 9 and minute < 30:
                return False
            if hour == 11 and minute > 30:
                return False
            return True

        # 下午 13:00 - 15:00
        if 13 <= hour < 15:
            return True

        return False

    def _get_cache_ttl(self) -> int:
        """获取当前缓存 TTL"""
        if self._is_trading_time():
            return self._cache_ttl  # 盘中 60 秒
        return 3600  # 盘后 1 小时

    def get_minute_bars(
        self,
        symbol: str,
        period: str = '5m',
        start_time: datetime = None,
        end_time: datetime = None,
        count: int = 0
    ) -> pd.DataFrame:
        """
        获取分钟K线数据（支持增量）

        Args:
            symbol: 股票代码
            period: 周期 (1m, 5m, 15m, 30m, 60m)
            start_time: 开始时间
            end_time: 结束时间
            count: 获取数量（0 表示全部）

        Returns:
            DataFrame，包含 timestamp/open/high/low/close/volume/amount 列
        """
        if period not in self.SUPPORTED_PERIODS:
            logger.warning(f"不支持的周期: {period}")
            return pd.DataFrame()

        cache_key = self._get_cache_key(symbol, period)
        current_time = time.time()
        cache_ttl = self._get_cache_ttl()

        # 1. 检查内存缓存
        cached_df = self._memory_cache.get(cache_key)
        if cached_df is not None and not cached_df.empty:
            # 检查缓存是否过期
            if cache_key in self._metadata_cache:
                _, cache_time = self._metadata_cache[cache_key]
                if current_time - cache_time < cache_ttl:
                    logger.debug(f"内存缓存命中: {symbol} {period}")
                    return self._filter_df(cached_df, start_time, end_time, count)

        # 2. 检查文件缓存
        if self._enable_file_cache:
            file_path = self._get_file_path(symbol, period)
            if file_path.exists():
                try:
                    cached_df = pd.read_parquet(file_path)
                    if not cached_df.empty:
                        cached_df['timestamp'] = pd.to_datetime(cached_df['timestamp'])
                        # 更新内存缓存
                        self._memory_cache.set(cache_key, cached_df)
                        self._metadata_cache[cache_key] = (
                            cached_df['timestamp'].max().timestamp() if len(cached_df) > 0 else 0,
                            current_time
                        )
                        logger.debug(f"文件缓存命中: {symbol} {period}")
                        return self._filter_df(cached_df, start_time, end_time, count)
                except Exception as e:
                    logger.warning(f"读取文件缓存失败: {e}")

        # 3. 获取新数据
        df = self._fetch_minute_bars(symbol, period, count)

        if df is not None and not df.empty:
            # 更新缓存
            self._memory_cache.set(cache_key, df)
            self._metadata_cache[cache_key] = (
                df['timestamp'].max().timestamp() if len(df) > 0 else 0,
                current_time
            )

            # 写入文件缓存
            if self._enable_file_cache:
                self._save_to_file(symbol, period, df)

        return self._filter_df(df, start_time, end_time, count) if df is not None else pd.DataFrame()

    def _fetch_minute_bars(
        self,
        symbol: str,
        period: str,
        count: int
    ) -> Optional[pd.DataFrame]:
        """从数据源获取分钟数据"""
        # 优先使用 Pytdx
        pytdx = self._get_pytdx_provider()
        if pytdx and pytdx.is_available():
            try:
                bars = pytdx.get_minute_bars(symbol, period, max(count, 500))
                if bars:
                    df = pd.DataFrame(bars)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    logger.debug(f"Pytdx 获取分钟数据: {symbol} {period} {len(df)} 条")
                    return df
            except Exception as e:
                logger.debug(f"Pytdx 获取分钟数据失败: {e}")

        # 使用 AkShare
        akshare = self._get_akshare_provider()
        if akshare:
            try:
                df = akshare.get_latest_bars(symbol, max(count, 500), period)
                if df is not None and not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    logger.debug(f"AkShare 获取分钟数据: {symbol} {period} {len(df)} 条")
                    return df
            except Exception as e:
                logger.debug(f"AkShare 获取分钟数据失败: {e}")

        logger.warning(f"所有数据源均无法获取 {symbol} 分钟数据")
        return None

    def _filter_df(
        self,
        df: pd.DataFrame,
        start_time: datetime,
        end_time: datetime,
        count: int
    ) -> pd.DataFrame:
        """过滤 DataFrame"""
        if df is None or df.empty:
            return df

        result = df.copy()

        if start_time is not None:
            result = result[result['timestamp'] >= start_time]

        if end_time is not None:
            result = result[result['timestamp'] <= end_time]

        if count > 0 and len(result) > count:
            result = result.tail(count)

        return result.reset_index(drop=True)

    def _save_to_file(self, symbol: str, period: str, df: pd.DataFrame) -> None:
        """保存到文件缓存"""
        try:
            file_path = self._get_file_path(symbol, period)
            df.to_parquet(file_path, index=False)
            logger.debug(f"保存文件缓存: {file_path}")
        except Exception as e:
            logger.warning(f"保存文件缓存失败: {e}")

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
        return self.get_minute_bars(symbol, period, count=count)

    def get_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取实时行情

        通过获取最新一根分钟 K 线实现
        """
        df = self.get_latest_bars(symbol, '1m', count=1)
        if df is not None and not df.empty:
            row = df.iloc[-1]
            return {
                'symbol': symbol,
                'price': float(row.get('close', 0)),
                'open': float(row.get('open', 0)),
                'high': float(row.get('high', 0)),
                'low': float(row.get('low', 0)),
                'volume': int(row.get('volume', 0)),
                'amount': float(row.get('amount', 0)),
                'timestamp': row.get('timestamp'),
            }
        return None

    def update_cache(self, symbol: str, period: str = '5m') -> bool:
        """
        强制更新缓存

        Args:
            symbol: 股票代码
            period: 周期

        Returns:
            是否成功
        """
        cache_key = self._get_cache_key(symbol, period)

        # 获取缓存中的最新时间
        cached_df = self._memory_cache.get(cache_key)
        last_time = None
        if cached_df is not None and not cached_df.empty:
            last_time = cached_df['timestamp'].max()

        # 获取新数据
        df = self._fetch_minute_bars(symbol, period, 1000)
        if df is not None and not df.empty:
            # 合并数据
            if cached_df is not None and not cached_df.empty:
                # 去重合并
                df = pd.concat([cached_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=['timestamp'], keep='last')
                df = df.sort_values('timestamp').reset_index(drop=True)

            # 更新缓存
            self._memory_cache.set(cache_key, df)
            self._metadata_cache[cache_key] = (
                df['timestamp'].max().timestamp() if len(df) > 0 else 0,
                time.time()
            )

            # 保存文件
            if self._enable_file_cache:
                self._save_to_file(symbol, period, df)

            return True

        return False

    def clear_cache(self, symbol: str = None, period: str = None) -> None:
        """
        清空缓存

        Args:
            symbol: 股票代码（None 表示清空全部）
            period: 周期
        """
        if symbol is None:
            self._memory_cache.clear()
            self._metadata_cache.clear()
            logger.info("已清空所有分钟数据缓存")
        else:
            cache_key = self._get_cache_key(symbol, period or '5m')
            # 清除内存缓存
            self._memory_cache.set(cache_key, None)
            if cache_key in self._metadata_cache:
                del self._metadata_cache[cache_key]
            # 清除文件缓存
            if self._enable_file_cache and period:
                file_path = self._get_file_path(symbol, period)
                if file_path.exists():
                    file_path.unlink()
            logger.info(f"已清空 {symbol} {period} 缓存")

    def get_cache_status(self) -> Dict[str, Any]:
        """获取缓存状态"""
        return {
            'memory_cache_size': len(self._memory_cache._cache),
            'metadata_cache_size': len(self._metadata_cache),
            'file_cache_dir': str(self._cache_dir),
            'cache_ttl': self._get_cache_ttl(),
            'is_trading_time': self._is_trading_time(),
        }


# 全局实例
_incremental_minute_provider: Optional[IncrementalMinuteProvider] = None


def get_incremental_minute_provider() -> IncrementalMinuteProvider:
    """获取全局 IncrementalMinuteProvider 实例"""
    global _incremental_minute_provider
    if _incremental_minute_provider is None:
        _incremental_minute_provider = IncrementalMinuteProvider()
    return _incremental_minute_provider
