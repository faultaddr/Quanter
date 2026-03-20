# -*- coding: utf-8 -*-
"""
===================================
SinaRealtimeProvider - 新浪实时行情 (备用数据源)
===================================

数据来源：新浪财经行情接口
特点：免费、无需 Token、支持批量查询、秒级更新

关键策略：
1. 批量查询优化（一次请求多只股票）
2. 本地缓存避免重复请求
3. 异步并发支持
4. 熔断保护
"""

import logging
import time
import re
import asyncio
import aiohttp
from typing import Optional, Dict, Any, List
from datetime import datetime
from urllib.parse import quote

import requests

from .realtime_types import (
    RealtimeQuote,
    RealtimeSource,
    CircuitBreaker,
    get_realtime_circuit_breaker,
    safe_float,
    safe_int,
    normalize_symbol,
)

logger = logging.getLogger(__name__)


class SinaRealtimeProvider:
    """
    新浪实时行情提供者

    特点：
    - 支持批量查询（推荐）
    - 秒级实时数据
    - 免费无需注册
    - 本地缓存

    使用示例：
        provider = SinaRealtimeProvider()
        quote = provider.get_realtime_quote("600519")
        quotes = provider.get_realtime_quotes(["600519", "000001", "000002"])
    """

    SOURCE_NAME = "sina"

    # 新浪行情接口
    QUOTE_URL = "http://hq.sinajs.cn/list={}"

    # 默认请求头
    DEFAULT_HEADERS = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://finance.sina.com.cn',
    }

    def __init__(
        self,
        cache_ttl: int = 3,
        timeout: int = 5,
        batch_size: int = 50,
    ):
        """
        初始化新浪实时行情提供者

        Args:
            cache_ttl: 缓存过期时间(秒)，默认3秒
            timeout: 请求超时(秒)
            batch_size: 批量查询每批数量
        """
        self._cache_ttl = cache_ttl
        self._timeout = timeout
        self._batch_size = batch_size

        # 缓存
        self._quote_cache: Dict[str, tuple] = {}
        self._batch_cache: Dict[str, tuple] = {}

        # 熔断器
        self._circuit_breaker = get_realtime_circuit_breaker()

    def _convert_symbol_to_sina(self, symbol: str) -> str:
        """
        将股票代码转换为新浪格式

        Examples:
            600519 -> sh600519
            000001 -> sz000001
            600519.SH -> sh600519
        """
        code, market = normalize_symbol(symbol)
        prefix = 'sh' if market == 1 else 'sz'
        return f"{prefix}{code}"

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

        sina_symbol = self._convert_symbol_to_sina(symbol)
        url = self.QUOTE_URL.format(sina_symbol)

        try:
            response = requests.get(
                url,
                headers=self.DEFAULT_HEADERS,
                timeout=self._timeout
            )
            response.encoding = 'gbk'

            if response.status_code == 200:
                quote = self._parse_response(symbol, response.text)
                if quote:
                    self._set_cache(cache_key, quote)
                    self._circuit_breaker.record_success(self.SOURCE_NAME)
                    return quote

        except Exception as e:
            logger.warning(f"新浪获取实时行情失败 {symbol}: {e}")
            self._circuit_breaker.record_failure(self.SOURCE_NAME, str(e))

        return None

    def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, RealtimeQuote]:
        """
        批量获取实时行情（推荐使用）

        新浪支持一次请求多只股票，格式如：
        http://hq.sinajs.cn/list=sh600519,sz000001,sz000002

        Args:
            symbols: 股票代码列表

        Returns:
            {symbol: RealtimeQuote} 字典
        """
        if not symbols:
            return {}

        # 检查缓存
        cache_key = f"batch_{','.join(sorted(symbols[:10]))}"
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
            # 分批处理
            for i in range(0, len(symbols), self._batch_size):
                batch = symbols[i:i + self._batch_size]
                batch_results = self._fetch_batch(batch)
                results.update(batch_results)

            self._circuit_breaker.record_success(self.SOURCE_NAME)
            self._batch_cache[cache_key] = (results, time.time())

        except Exception as e:
            logger.warning(f"新浪批量获取实时行情失败: {e}")
            self._circuit_breaker.record_failure(self.SOURCE_NAME, str(e))

        return results

    def _fetch_batch(self, symbols: List[str]) -> Dict[str, RealtimeQuote]:
        """获取一批股票的行情"""
        results = {}

        # 构建 URL
        sina_symbols = [self._convert_symbol_to_sina(s) for s in symbols]
        url = self.QUOTE_URL.format(','.join(sina_symbols))

        try:
            response = requests.get(
                url,
                headers=self.DEFAULT_HEADERS,
                timeout=self._timeout
            )
            response.encoding = 'gbk'

            if response.status_code == 200:
                results = self._parse_batch_response(symbols, response.text)

        except Exception as e:
            logger.warning(f"新浪批量请求失败: {e}")

        return results

    def _parse_response(self, symbol: str, text: str) -> Optional[RealtimeQuote]:
        """解析新浪返回的数据"""
        # 格式：var hq_str_sh600519="贵州茅台,1850.00,1845.00,..."
        match = re.search(r'="([^"]*)"', text)
        if not match:
            return None

        data = match.group(1).split(',')
        if len(data) < 6:
            return None

        try:
            # 指数数据格式：名称,开盘,昨收,最新,最高,最低,0,0,成交量,成交额,...,日期,时间
            # 股票数据格式：名称,开盘,昨收,最新,最高,最低,...,五档数据,...(32+字段)
            # 判断依据：五档数据是否为0（指数的五档全是0）
            is_index = False
            if len(data) > 10:
                # 检查五档区域是否全是0
                bid1 = safe_float(data[10]) or 0
                bid_vol1 = safe_int(data[11]) or 0
                if bid1 == 0 and bid_vol1 == 0:
                    is_index = True

            # 解析基本价格
            open_price = safe_float(data[1]) if len(data) > 1 else None
            pre_close = safe_float(data[2]) if len(data) > 2 else None
            price = safe_float(data[3]) if len(data) > 3 else None
            high = safe_float(data[4]) if len(data) > 4 else None
            low = safe_float(data[5]) if len(data) > 5 else None
            volume = safe_int(data[8]) if len(data) > 8 else None
            amount = safe_float(data[9]) if len(data) > 9 else None

            # 计算涨跌幅
            change_pct = None
            change_amount = None
            if price is not None and pre_close is not None and pre_close > 0:
                change_amount = price - pre_close
                change_pct = change_amount / pre_close

            # 构建基本报价
            quote = RealtimeQuote(
                symbol=symbol,
                name=data[0],
                source=RealtimeSource.SINA,
                open=open_price,
                pre_close=pre_close,
                price=price,
                high=high,
                low=low,
                volume=volume,
                amount=amount,
                change_pct=change_pct,
                change_amount=change_amount,
                timestamp=datetime.now(),
            )

            # 只有股票才有五档数据
            if not is_index and len(data) >= 30:
                quote.bid_prices = [
                    safe_float(data[10]) or 0.0,
                    safe_float(data[12]) or 0.0,
                    safe_float(data[14]) or 0.0,
                    safe_float(data[16]) or 0.0,
                    safe_float(data[18]) or 0.0,
                ]
                quote.bid_volumes = [
                    safe_int(data[11]) or 0,
                    safe_int(data[13]) or 0,
                    safe_int(data[15]) or 0,
                    safe_int(data[17]) or 0,
                    safe_int(data[19]) or 0,
                ]
                quote.ask_prices = [
                    safe_float(data[20]) or 0.0,
                    safe_float(data[22]) or 0.0,
                    safe_float(data[24]) or 0.0,
                    safe_float(data[26]) or 0.0,
                    safe_float(data[28]) or 0.0,
                ]
                quote.ask_volumes = [
                    safe_int(data[21]) or 0,
                    safe_int(data[23]) or 0,
                    safe_int(data[25]) or 0,
                    safe_int(data[27]) or 0,
                    safe_int(data[29]) or 0,
                ]

            return quote
        except (IndexError, ValueError) as e:
            logger.warning(f"解析新浪数据失败 {symbol}: {e}")
            return None

    def _parse_batch_response(
        self,
        symbols: List[str],
        text: str
    ) -> Dict[str, RealtimeQuote]:
        """解析批量返回数据"""
        results = {}

        # 每行一个股票
        lines = text.strip().split('\n')
        for line in lines:
            # 提取股票代码和内容
            match = re.match(r'var hq_str_(\w+)="([^"]*)"', line)
            if not match:
                continue

            sina_symbol = match.group(1)
            content = match.group(2)

            # 找到对应的 symbol
            code = sina_symbol[2:]  # 去掉 sh/sz 前缀
            symbol = None
            for s in symbols:
                pure_code, _ = normalize_symbol(s)
                if pure_code == code:
                    symbol = s
                    break

            if not symbol or not content:
                continue

            data = content.split(',')
            if len(data) < 6:
                continue

            try:
                # 判断依据：五档数据是否为0（指数的五档全是0）
                is_index = False
                if len(data) > 10:
                    bid1 = safe_float(data[10]) or 0
                    bid_vol1 = safe_int(data[11]) or 0
                    if bid1 == 0 and bid_vol1 == 0:
                        is_index = True

                # 解析基本价格
                open_price = safe_float(data[1]) if len(data) > 1 else None
                pre_close = safe_float(data[2]) if len(data) > 2 else None
                price = safe_float(data[3]) if len(data) > 3 else None
                high = safe_float(data[4]) if len(data) > 4 else None
                low = safe_float(data[5]) if len(data) > 5 else None
                volume = safe_int(data[8]) if len(data) > 8 else None
                amount = safe_float(data[9]) if len(data) > 9 else None

                # 计算涨跌幅
                change_pct = None
                change_amount = None
                if price is not None and pre_close is not None and pre_close > 0:
                    change_amount = price - pre_close
                    change_pct = change_amount / pre_close

                # 构建基本报价
                quote = RealtimeQuote(
                    symbol=symbol,
                    name=data[0],
                    source=RealtimeSource.SINA,
                    open=open_price,
                    pre_close=pre_close,
                    price=price,
                    high=high,
                    low=low,
                    volume=volume,
                    amount=amount,
                    change_pct=change_pct,
                    change_amount=change_amount,
                    timestamp=datetime.now(),
                )

                # 只有股票才有五档数据
                if not is_index and len(data) >= 30:
                    quote.bid_prices = [
                        safe_float(data[10]) or 0.0,
                        safe_float(data[12]) or 0.0,
                        safe_float(data[14]) or 0.0,
                        safe_float(data[16]) or 0.0,
                        safe_float(data[18]) or 0.0,
                    ]
                    quote.bid_volumes = [
                        safe_int(data[11]) or 0,
                        safe_int(data[13]) or 0,
                        safe_int(data[15]) or 0,
                        safe_int(data[17]) or 0,
                        safe_int(data[19]) or 0,
                    ]
                    quote.ask_prices = [
                        safe_float(data[20]) or 0.0,
                        safe_float(data[22]) or 0.0,
                        safe_float(data[24]) or 0.0,
                        safe_float(data[26]) or 0.0,
                        safe_float(data[28]) or 0.0,
                    ]
                    quote.ask_volumes = [
                        safe_int(data[21]) or 0,
                        safe_int(data[23]) or 0,
                        safe_int(data[25]) or 0,
                        safe_int(data[27]) or 0,
                        safe_int(data[29]) or 0,
                    ]

                results[symbol] = quote
            except (IndexError, ValueError) as e:
                logger.debug(f"解析新浪数据失败 {symbol}: {e}")

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
        if not symbols:
            return {}

        # 检查缓存
        cache_key = f"batch_{','.join(sorted(symbols[:10]))}"
        if cache_key in self._batch_cache:
            data, timestamp = self._batch_cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return data

        # 检查熔断器
        if not self._circuit_breaker.is_available(self.SOURCE_NAME):
            return {}

        results = {}

        try:
            async with aiohttp.ClientSession() as session:
                # 分批并发请求
                tasks = []
                for i in range(0, len(symbols), self._batch_size):
                    batch = symbols[i:i + self._batch_size]
                    tasks.append(self._fetch_batch_async(session, batch))

                batch_results = await asyncio.gather(*tasks)
                for batch_result in batch_results:
                    results.update(batch_result)

            self._circuit_breaker.record_success(self.SOURCE_NAME)
            self._batch_cache[cache_key] = (results, time.time())

        except Exception as e:
            logger.warning(f"新浪异步批量获取失败: {e}")
            self._circuit_breaker.record_failure(self.SOURCE_NAME, str(e))

        return results

    async def _fetch_batch_async(
        self,
        session: aiohttp.ClientSession,
        symbols: List[str]
    ) -> Dict[str, RealtimeQuote]:
        """异步获取一批股票行情"""
        sina_symbols = [self._convert_symbol_to_sina(s) for s in symbols]
        url = self.QUOTE_URL.format(','.join(sina_symbols))

        try:
            async with session.get(
                url,
                headers=self.DEFAULT_HEADERS,
                timeout=aiohttp.ClientTimeout(total=self._timeout)
            ) as response:
                if response.status == 200:
                    text = await response.text('gbk')
                    return self._parse_batch_response(symbols, text)
        except Exception as e:
            logger.debug(f"新浪异步请求失败: {e}")

        return {}

    def is_available(self) -> bool:
        """检查数据源是否可用"""
        return self._circuit_breaker.is_available(self.SOURCE_NAME)

    def get_status(self) -> Dict[str, Any]:
        """获取提供者状态"""
        return {
            'source': self.SOURCE_NAME,
            'available': self.is_available(),
            'circuit_breaker': self._circuit_breaker.get_status(),
        }


# 单例实例
_sina_provider_instance: Optional[SinaRealtimeProvider] = None


def get_sina_provider() -> SinaRealtimeProvider:
    """获取全局 SinaRealtimeProvider 实例"""
    global _sina_provider_instance
    if _sina_provider_instance is None:
        _sina_provider_instance = SinaRealtimeProvider()
    return _sina_provider_instance
