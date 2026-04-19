# -*- coding: utf-8 -*-
"""
===================================
PytdxRealtimeProvider - 通达信实时行情 (核心秒级数据源)
===================================

数据来源：通达信行情服务器（pytdx 库）
特点：免费、无需 Token、直连行情服务器、秒级更新

关键策略：
1. 多服务器自动切换
2. 连接池复用
3. 批量获取优化
4. 本地缓存避免重复请求

参考 ZhuLinsen/daily_stock_analysis 的 pytdx_fetcher.py 设计
"""

import logging
import time
import os
from contextlib import contextmanager
from typing import Optional, Dict, Any, List, Tuple, Generator
from datetime import datetime
from dataclasses import dataclass

from .types import (
    RealtimeQuote,
    RealtimeSource,
    CircuitBreaker,
    get_realtime_circuit_breaker,
    safe_float,
    safe_int,
    normalize_symbol,
    is_etf_code,
    is_index_code,
)

logger = logging.getLogger(__name__)


# 默认通达信行情服务器列表
DEFAULT_PYTDX_HOSTS = [
    ("119.147.212.81", 7709),   # 深圳
    ("112.74.214.43", 7727),    # 深圳
    ("221.231.141.60", 7709),   # 上海
    ("101.227.73.20", 7709),    # 上海
    ("101.227.77.254", 7709),   # 上海
    ("14.215.128.18", 7709),    # 广州
    ("59.173.18.140", 7709),    # 武汉
    ("180.153.39.51", 7709),    # 杭州
]


def _parse_hosts_from_env() -> Optional[List[Tuple[str, int]]]:
    """从环境变量解析通达信服务器列表"""
    servers = os.getenv("PYTDX_SERVERS", "").strip()
    if servers:
        result = []
        for part in servers.split(","):
            part = part.strip()
            if ":" in part:
                host, port_str = part.rsplit(":", 1)
                host, port_str = host.strip(), port_str.strip()
                if host and port_str:
                    try:
                        result.append((host, int(port_str)))
                    except ValueError:
                        logger.warning(f"Invalid PYTDX_SERVERS entry: {part}")
        if result:
            return result

    host = os.getenv("PYTDX_HOST", "").strip()
    port_str = os.getenv("PYTDX_PORT", "").strip()
    if host and port_str:
        try:
            return [(host, int(port_str))]
        except ValueError:
            logger.warning(f"Invalid PYTDX_HOST/PYTDX_PORT: {host}:{port_str}")

    return None


class PytdxRealtimeProvider:
    """
    通达信实时行情提供者

    特点：
    - 秒级实时数据
    - 支持批量获取（一次请求多只股票）
    - 支持五档买卖盘
    - 多服务器自动切换
    - 本地缓存避免重复请求

    使用示例：
        provider = PytdxRealtimeProvider()
        quote = provider.get_realtime_quote("600519")
        quotes = provider.get_realtime_quotes(["600519", "000001", "000002"])
    """

    SOURCE_NAME = "pytdx"

    def __init__(
        self,
        hosts: Optional[List[Tuple[str, int]]] = None,
        cache_ttl: int = 3,
        connect_timeout: int = 5,
    ):
        """
        初始化通达信实时行情提供者

        Args:
            hosts: 服务器列表 [(host, port), ...]
            cache_ttl: 缓存过期时间(秒)，默认3秒
            connect_timeout: 连接超时(秒)
        """
        self._hosts = hosts or _parse_hosts_from_env() or DEFAULT_PYTDX_HOSTS
        self._cache_ttl = cache_ttl
        self._connect_timeout = connect_timeout
        self._current_host_idx = 0

        # 缓存 {cache_key: (data, timestamp)}
        self._quote_cache: Dict[str, Tuple[RealtimeQuote, float]] = {}
        self._batch_cache: Dict[str, Tuple[Dict[str, RealtimeQuote], float]] = {}

        # 熔断器
        self._circuit_breaker = get_realtime_circuit_breaker()

        # pytdx 模块（延迟加载）
        self._pytdx_api = None
        self._initialized = False

    def _get_pytdx_api(self):
        """延迟加载 pytdx 模块"""
        if self._pytdx_api is None:
            try:
                from pytdx.hq import TdxHq_API
                self._pytdx_api = TdxHq_API
                logger.info("Pytdx 模块加载成功")
            except ImportError:
                logger.warning("pytdx 未安装，请运行: pip install pytdx")
                return None
        return self._pytdx_api

    @contextmanager
    def _pytdx_session(self) -> Generator:
        """
        Pytdx 连接上下文管理器

        自动管理连接和断开
        """
        TdxHq_API = self._get_pytdx_api()
        if TdxHq_API is None:
            raise RuntimeError("pytdx 库未安装")

        api = TdxHq_API()
        connected = False

        try:
            # 尝试连接服务器
            for i in range(len(self._hosts)):
                host_idx = (self._current_host_idx + i) % len(self._hosts)
                host, port = self._hosts[host_idx]

                try:
                    if api.connect(host, port, time_out=self._connect_timeout):
                        connected = True
                        self._current_host_idx = host_idx
                        logger.debug(f"Pytdx 连接成功: {host}:{port}")
                        break
                except Exception as e:
                    logger.debug(f"Pytdx 连接 {host}:{port} 失败: {e}")
                    continue

            if not connected:
                raise RuntimeError("Pytdx 无法连接任何服务器")

            yield api

        finally:
            try:
                api.disconnect()
            except Exception:
                pass

    def _get_market_code(self, symbol: str) -> Tuple[int, str]:
        """
        根据股票代码判断市场

        Returns:
            (市场代码, 纯代码)
            市场代码: 0=深圳, 1=上海
        """
        code, market = normalize_symbol(symbol)
        return market, code

    def _check_cache(self, key: str) -> Optional[Any]:
        """检查缓存"""
        if key in self._quote_cache:
            data, timestamp = self._quote_cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """设置缓存"""
        self._quote_cache[key] = (data, time.time())

    def get_realtime_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """
        获取单只股票实时行情

        Args:
            symbol: 股票代码

        Returns:
            RealtimeQuote 或 None
        """
        # 检查缓存
        cache_key = f"quote_{symbol}"
        cached = self._check_cache(cache_key)
        if cached is not None:
            return cached

        # 检查熔断器
        if not self._circuit_breaker.is_available(self.SOURCE_NAME):
            logger.warning(f"[熔断] {self.SOURCE_NAME} 处于熔断状态")
            return None

        try:
            market, code = self._get_market_code(symbol)

            with self._pytdx_session() as api:
                data = api.get_security_quotes([(market, code)])

                if data and len(data) > 0:
                    quote = self._parse_quote(symbol, data[0])
                    self._set_cache(cache_key, quote)
                    self._circuit_breaker.record_success(self.SOURCE_NAME)
                    return quote

        except Exception as e:
            logger.warning(f"Pytdx 获取实时行情失败 {symbol}: {e}")
            self._circuit_breaker.record_failure(self.SOURCE_NAME, str(e))

        return None

    def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, RealtimeQuote]:
        """
        批量获取实时行情（推荐使用）

        一次请求获取多只股票，效率更高

        Args:
            symbols: 股票代码列表

        Returns:
            {symbol: RealtimeQuote} 字典
        """
        if not symbols:
            return {}

        # 检查缓存
        cache_key = f"batch_{','.join(sorted(symbols[:10]))}"  # 取前10个作为缓存键
        if cache_key in self._batch_cache:
            data, timestamp = self._batch_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return data

        # 检查熔断器
        if not self._circuit_breaker.is_available(self.SOURCE_NAME):
            logger.warning(f"[熔断] {self.SOURCE_NAME} 处于熔断状态")
            return {}

        results = {}
        try:
            # 构建请求参数
            params = []
            for symbol in symbols:
                market, code = self._get_market_code(symbol)
                params.append((market, code))

            with self._pytdx_session() as api:
                # pytdx 支持一次请求多只股票
                data = api.get_security_quotes(params)

                if data:
                    for i, item in enumerate(data):
                        if i < len(symbols):
                            symbol = symbols[i]
                            quote = self._parse_quote(symbol, item)
                            results[symbol] = quote

                    self._circuit_breaker.record_success(self.SOURCE_NAME)
                    self._batch_cache[cache_key] = (results, time.time())

        except Exception as e:
            logger.warning(f"Pytdx 批量获取实时行情失败: {e}")
            self._circuit_breaker.record_failure(self.SOURCE_NAME, str(e))

        return results

    def _parse_quote(self, symbol: str, data: Dict[str, Any]) -> RealtimeQuote:
        """解析 pytdx 返回的数据"""
        return RealtimeQuote(
            symbol=symbol,
            name=str(data.get('name', '')),
            source=RealtimeSource.PYTDX,
            price=safe_float(data.get('price')),
            open=safe_float(data.get('open')),
            high=safe_float(data.get('high')),
            low=safe_float(data.get('low')),
            pre_close=safe_float(data.get('last_close')),
            volume=safe_int(data.get('vol')),
            amount=safe_float(data.get('amount')),
            # 五档数据
            bid_prices=[
                safe_float(data.get(f'bid{i}', 0)) or 0.0
                for i in range(1, 6)
            ],
            bid_volumes=[
                safe_int(data.get(f'bid_vol{i}', 0)) or 0
                for i in range(1, 6)
            ],
            ask_prices=[
                safe_float(data.get(f'ask{i}', 0)) or 0.0
                for i in range(1, 6)
            ],
            ask_volumes=[
                safe_int(data.get(f'ask_vol{i}', 0)) or 0
                for i in range(1, 6)
            ],
            timestamp=datetime.now(),
        )

    def get_minute_bars(
        self,
        symbol: str,
        period: str = '5m',
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取分钟K线数据

        Args:
            symbol: 股票代码
            period: 周期 (1m, 5m, 15m, 30m, 60m)
            count: 获取数量

        Returns:
            K线数据列表
        """
        # 周期映射
        period_map = {
            '1m': 8,
            '5m': 0,
            '15m': 1,
            '30m': 2,
            '60m': 3,
        }
        category = period_map.get(period, 0)

        try:
            market, code = self._get_market_code(symbol)

            with self._pytdx_session() as api:
                data = api.get_security_bars(category, market, code, 0, count)

                if data:
                    bars = []
                    for item in data:
                        bars.append({
                            'symbol': symbol,
                            'timestamp': datetime.strptime(
                                item.get('datetime', ''), '%Y-%m-%d %H:%M'
                            ),
                            'open': safe_float(item.get('open')),
                            'high': safe_float(item.get('high')),
                            'low': safe_float(item.get('low')),
                            'close': safe_float(item.get('close')),
                            'volume': safe_int(item.get('vol')),
                            'amount': safe_float(item.get('amount')),
                            'period': period,
                        })
                    return bars

        except Exception as e:
            logger.warning(f"Pytdx 获取分钟K线失败 {symbol}: {e}")

        return []

    def get_stock_list(self) -> List[Dict[str, str]]:
        """
        获取股票列表

        Returns:
            [{'code': '600519', 'name': '贵州茅台'}, ...]
        """
        results = []
        try:
            with self._pytdx_session() as api:
                for market in (0, 1):
                    start = 0
                    while True:
                        stocks = api.get_security_list(market, start)
                        if not stocks:
                            break

                        for stock in stocks:
                            code = stock.get('code', '')
                            name = stock.get('name', '')
                            if code and name:
                                results.append({'code': code, 'name': name})

                        if len(stocks) < 1000:
                            break
                        start += 1000

        except Exception as e:
            logger.warning(f"Pytdx 获取股票列表失败: {e}")

        return results

    def is_available(self) -> bool:
        """检查数据源是否可用"""
        return self._circuit_breaker.is_available(self.SOURCE_NAME)

    def get_status(self) -> Dict[str, Any]:
        """获取提供者状态"""
        return {
            'source': self.SOURCE_NAME,
            'available': self.is_available(),
            'circuit_breaker': self._circuit_breaker.get_status(),
            'current_host': self._hosts[self._current_host_idx] if self._hosts else None,
        }


# 单例实例
_pytdx_provider_instance: Optional[PytdxRealtimeProvider] = None


def get_pytdx_provider() -> PytdxRealtimeProvider:
    """获取全局 PytdxRealtimeProvider 实例"""
    global _pytdx_provider_instance
    if _pytdx_provider_instance is None:
        _pytdx_provider_instance = PytdxRealtimeProvider()
    return _pytdx_provider_instance
