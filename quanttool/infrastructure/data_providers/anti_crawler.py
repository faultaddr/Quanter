"""
反爬虫防护模块 v2.0

核心功能：
- User-Agent管理
- 请求头生成
- 延迟控制
- 代理池管理（支持免费代理和付费代理API）
- 指数退避重试
- Cookie会话管理

用于规避AShare爬虫被封的风险
"""

import random
import time
import threading
import json
import os
from typing import List, Dict, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import functools
import logging
from urllib.parse import urlparse

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================= User-Agent管理器 =============================

class UserAgentManager:
    """
    管理真实浏览器的User-Agent，支持随机选择和按类型选择
    """

    # 真实浏览器User-Agent列表（2025年最新）
    USER_AGENTS = [
        # Chrome浏览器 (Windows)
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36',

        # Chrome浏览器 (Mac)
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',

        # Firefox浏览器 (Windows)
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:120.0) Gecko/20100101 Firefox/120.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:119.0) Gecko/20100101 Firefox/119.0',

        # Firefox浏览器 (Mac)
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0',

        # Safari浏览器 (Mac)
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',

        # Safari浏览器 (iOS)
        'Mozilla/5.0 (iPhone; CPU iPhone OS 17_2_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Mobile/15E148 Safari/604.1',

        # Edge浏览器 (Windows)
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',

        # Edge浏览器 (Mac)
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',

        # Chrome浏览器 (Linux)
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    ]

    @classmethod
    def get_random_ua(cls) -> str:
        """获取随机的User-Agent"""
        return random.choice(cls.USER_AGENTS)

    @classmethod
    def get_all_uas(cls) -> List[str]:
        """获取所有User-Agent列表"""
        return cls.USER_AGENTS

    @classmethod
    def get_chrome_ua(cls) -> str:
        """获取Chrome浏览器的User-Agent"""
        chrome_uas = [ua for ua in cls.USER_AGENTS if 'Chrome' in ua and 'Edg' not in ua]
        return random.choice(chrome_uas)

    @classmethod
    def get_firefox_ua(cls) -> str:
        """获取Firefox浏览器的User-Agent"""
        firefox_uas = [ua for ua in cls.USER_AGENTS if 'Firefox' in ua]
        return random.choice(firefox_uas)


# ============================= 请求头生成器 =============================

class HeaderGenerator:
    """
    生成完整的HTTP请求头，伪装成真实浏览器
    """

    # 标准的Accept值
    ACCEPT_VALUES = [
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    ]

    # 标准的Accept-Language值
    ACCEPT_LANGUAGE_VALUES = [
        'zh-CN,zh;q=0.9',
        'zh-CN,zh;q=0.9,en;q=0.8',
        'zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7',
        'zh-CN,zh-Hans;q=0.9,en;q=0.8,zh-Hant;q=0.7',
    ]

    # 标准的Accept-Encoding值
    ACCEPT_ENCODING_VALUES = [
        'gzip, deflate, br',
        'gzip, deflate',
    ]

    @staticmethod
    def generate_headers(
        user_agent: Optional[str] = None,
        referer: Optional[str] = None,
        api_mode: bool = False
    ) -> Dict[str, str]:
        """
        生成完整的HTTP请求头

        Args:
            user_agent: 指定的User-Agent（如果为None则随机生成）
            referer: 指定的Referer（如果为None则自动生成）
            api_mode: 是否为API请求优化模式

        Returns:
            字典形式的请求头
        """
        if user_agent is None:
            user_agent = UserAgentManager.get_random_ua()

        # 基础请求头
        headers = {
            'User-Agent': user_agent,
            'Accept': random.choice(HeaderGenerator.ACCEPT_VALUES),
            'Accept-Language': random.choice(HeaderGenerator.ACCEPT_LANGUAGE_VALUES),
            'Accept-Encoding': random.choice(HeaderGenerator.ACCEPT_ENCODING_VALUES),
            'DNT': '1',
            'Connection': 'keep-alive',
            'Cache-Control': 'max-age=0',
        }

        # 添加Referer
        if referer:
            headers['Referer'] = referer
        else:
            # 默认Referer来自主流网站
            default_referrers = [
                'https://www.baidu.com/',
                'https://www.google.com/',
                'https://www.bing.com/',
                'https://www.qq.com/',
                'https://www.sina.com.cn/',
            ]
            headers['Referer'] = random.choice(default_referrers)

        # API模式特定的优化
        if api_mode:
            headers['X-Requested-With'] = 'XMLHttpRequest'

        return headers

    @staticmethod
    def generate_sina_headers() -> Dict[str, str]:
        """生成针对新浪财经的请求头"""
        return HeaderGenerator.generate_headers(
            referer='https://finance.sina.com.cn/',
            api_mode=True
        )

    @staticmethod
    def generate_tencent_headers() -> Dict[str, str]:
        """生成针对腾讯财经的请求头"""
        headers = HeaderGenerator.generate_headers(
            referer='http://gu.qq.com/',
            api_mode=True
        )
        headers['Origin'] = 'http://web.ifzq.gtimg.cn'
        return headers

    @staticmethod
    def generate_eastmoney_headers() -> Dict[str, str]:
        """生成针对东方财富的请求头"""
        return HeaderGenerator.generate_headers(
            referer='https://www.eastmoney.com/',
            api_mode=True
        )


# ============================= 延迟控制系统 =============================

class DelayController:
    """
    控制请求之间的延迟，模拟真实用户行为
    """

    # 全局单例
    _instance: Optional['DelayController'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """
        初始化延迟控制器

        Args:
            min_delay: 最小延迟（秒）
            max_delay: 最大延迟（秒）
        """
        if self._initialized:
            return

        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0.0
        self.request_count = 0
        self._initialized = True

    def wait(self) -> float:
        """
        等待指定的延迟时间

        Returns:
            实际等待的时间（秒）
        """
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)
        self.last_request_time = time.time()
        self.request_count += 1
        logger.debug(f'Delayed {delay:.2f}s (Total requests: {self.request_count})')
        return delay

    def wait_if_needed(self, min_interval: float = 1.0) -> bool:
        """
        如果距离上次请求时间过短，则等待

        Args:
            min_interval: 最小请求间隔（秒）

        Returns:
            是否执行了等待
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < min_interval:
            wait_time = min_interval - elapsed + random.uniform(0, 0.1)
            time.sleep(wait_time)
            self.last_request_time = time.time()
            return True
        return False

    def set_strict_mode(self) -> None:
        """启用严格模式（延迟2-8秒）"""
        self.min_delay = 2.0
        self.max_delay = 8.0
        logger.info('Strict delay mode enabled (2-8s delay)')

    def set_normal_mode(self) -> None:
        """启用正常模式（延迟1-3秒）"""
        self.min_delay = 1.0
        self.max_delay = 3.0
        logger.info('Normal delay mode enabled (1-3s delay)')

    def set_fast_mode(self) -> None:
        """启用快速模式（延迟0.3-1秒）"""
        self.min_delay = 0.3
        self.max_delay = 1.0
        logger.info('Fast delay mode enabled (0.3-1s delay)')

    @classmethod
    def get_instance(cls) -> 'DelayController':
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ============================= 重试装饰器 =============================

def retry_on_failure(
    max_retries: int = 1,  # Reduced from 3 to minimize latency
    base_delay: float = 0.5,  # Reduced from 1.0 for faster retry
    max_delay: float = 10.0,  # Reduced from 60.0
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    失败时指数退避重试装饰器

    Args:
        max_retries: 最大重试次数（默认1次以降低延迟）
        base_delay: 基础延迟（秒）
        max_delay: 最大延迟（秒）
        exceptions: 需要重试的异常类型

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # 指数退避
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        # 添加随机抖动
                        delay = delay + random.uniform(0, delay * 0.1)
                        logger.warning(
                            f'Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. '
                            f'Retrying in {delay:.2f}s...'
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f'All {max_retries + 1} attempts failed')

            raise last_exception

        return wrapper
    return decorator


# ============================= 安全请求函数 =============================

def safe_request(
    url: str,
    method: str = 'GET',
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 15,
    use_delay: bool = True,
    **kwargs
) -> Any:
    """
    安全的HTTP请求函数，集成反爬虫防护

    Args:
        url: 请求URL
        method: HTTP方法
        headers: 自定义请求头（如果为None则自动生成）
        timeout: 超时时间（秒）
        use_delay: 是否使用延迟控制
        **kwargs: 传递给requests的其他参数

    Returns:
        Response对象
    """
    import requests

    # 延迟控制
    if use_delay:
        delay_controller = DelayController.get_instance()
        delay_controller.wait()

    # 生成请求头
    if headers is None:
        headers = HeaderGenerator.generate_headers()

    # 显式禁用系统代理，避免 macOS 代理设置干扰数据源请求
    no_proxy = {"http": None, "https": None}
    kwargs.setdefault("proxies", no_proxy)

    # 执行请求
    if method.upper() == 'GET':
        response = requests.get(url, headers=headers, timeout=timeout, **kwargs)
    elif method.upper() == 'POST':
        response = requests.post(url, headers=headers, timeout=timeout, **kwargs)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

    return response


# ============================= 便捷函数 =============================

def get_random_headers() -> Dict[str, str]:
    """获取随机请求头（便捷函数）"""
    return HeaderGenerator.generate_headers()


def smart_delay() -> float:
    """智能延迟（便捷函数）"""
    return DelayController.get_instance().wait()


def get_sina_headers() -> Dict[str, str]:
    """获取新浪财经请求头"""
    return HeaderGenerator.generate_sina_headers()


def get_tencent_headers() -> Dict[str, str]:
    """获取腾讯财经请求头"""
    return HeaderGenerator.generate_tencent_headers()


def get_eastmoney_headers() -> Dict[str, str]:
    """获取东方财富请求头"""
    return HeaderGenerator.generate_eastmoney_headers()


# ============================= 代理池管理器 =============================

class ProxyStatus(Enum):
    """代理状态枚举"""
    AVAILABLE = "available"
    IN_USE = "in_use"
    FAILED = "failed"
    COOLDOWN = "cooldown"


@dataclass
class ProxyInfo:
    """代理信息"""
    host: str
    port: int
    protocol: str = "http"  # http, https, socks5
    username: Optional[str] = None
    password: Optional[str] = None
    status: ProxyStatus = ProxyStatus.AVAILABLE
    fail_count: int = 0
    success_count: int = 0
    last_used: Optional[datetime] = None
    last_failed: Optional[datetime] = None
    response_time: float = 0.0  # 平均响应时间(秒)

    @property
    def url(self) -> str:
        """获取代理URL"""
        if self.username and self.password:
            return f"{self.protocol}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.protocol}://{self.host}:{self.port}"

    @property
    def address(self) -> str:
        """获取代理地址"""
        return f"{self.host}:{self.port}"

    @property
    def success_rate(self) -> float:
        """成功率"""
        total = self.fail_count + self.success_count
        if total == 0:
            return 1.0
        return self.success_count / total


class ProxyPool:
    """
    代理池管理器

    功能：
    - 支持多种代理来源：文件、API、手动添加
    - 自动健康检查
    - 智能轮换策略
    - 失败自动切换
    - 支持付费代理API（快代理、芝麻代理等）
    """

    _instance: Optional['ProxyPool'] = None
    _lock = threading.Lock()

    # 默认配置
    DEFAULT_CONFIG = {
        'max_fail_count': 3,          # 最大失败次数
        'cooldown_minutes': 10,       # 冷却时间(分钟)
        'check_timeout': 10,          # 健康检查超时(秒)
        'check_url': 'http://httpbin.org/ip',  # 健康检查URL
        'min_success_rate': 0.5,      # 最低成功率阈值
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        proxy_file: Optional[str] = None,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        初始化代理池

        Args:
            proxy_file: 代理列表文件路径(每行一个代理，格式: host:port 或 protocol://user:pass@host:port)
            api_url: 代理API URL（付费代理服务商提供）
            api_key: 代理API密钥
            config: 配置字典
        """
        if self._initialized:
            return

        self._proxies: List[ProxyInfo] = []
        self._current_index = 0
        self._config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._api_url = api_url
        self._api_key = api_key
        self._last_fetch_time: Optional[datetime] = None
        self._fetch_interval = timedelta(minutes=5)  # API获取间隔

        # 加载代理
        if proxy_file:
            self.load_from_file(proxy_file)

        self._initialized = True
        logger.info(f"ProxyPool initialized with {len(self._proxies)} proxies")

    # ==================== 代理加载方法 ====================

    def load_from_file(self, filepath: str) -> int:
        """
        从文件加载代理列表

        Args:
            filepath: 文件路径

        Returns:
            加载的代理数量
        """
        if not os.path.exists(filepath):
            logger.warning(f"Proxy file not found: {filepath}")
            return 0

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                proxy = self._parse_proxy_line(line)
                if proxy:
                    self._proxies.append(proxy)
                    count += 1

        logger.info(f"Loaded {count} proxies from {filepath}")
        return count

    def load_from_list(self, proxy_list: List[str]) -> int:
        """
        从列表加载代理

        Args:
            proxy_list: 代理列表，格式如 ["http://1.2.3.4:8080", "socks5://user:pass@5.6.7.8:1080"]

        Returns:
            加载的代理数量
        """
        count = 0
        for line in proxy_list:
            proxy = self._parse_proxy_line(line)
            if proxy:
                self._proxies.append(proxy)
                count += 1
        logger.info(f"Loaded {count} proxies from list")
        return count

    def add_proxy(
        self,
        host: str,
        port: int,
        protocol: str = "http",
        username: Optional[str] = None,
        password: Optional[str] = None
    ) -> None:
        """手动添加单个代理"""
        proxy = ProxyInfo(
            host=host,
            port=port,
            protocol=protocol,
            username=username,
            password=password
        )
        self._proxies.append(proxy)
        logger.info(f"Added proxy: {proxy.address}")

    def _parse_proxy_line(self, line: str) -> Optional[ProxyInfo]:
        """解析代理行"""
        try:
            # 格式1: protocol://user:pass@host:port
            if '://' in line:
                parsed = urlparse(line)
                protocol = parsed.scheme
                host = parsed.hostname
                port = parsed.port
                username = parsed.username
                password = parsed.password
            # 格式2: host:port
            elif ':' in line:
                parts = line.split(':')
                host = parts[0]
                port = int(parts[1])
                protocol = "http"
                username = None
                password = None
            else:
                return None

            if not host or not port:
                return None

            return ProxyInfo(
                host=host,
                port=port,
                protocol=protocol,
                username=username,
                password=password
            )
        except Exception as e:
            logger.debug(f"Failed to parse proxy line '{line}': {e}")
            return None

    # ==================== 代理获取方法 ====================

    def get_proxy(self) -> Optional[ProxyInfo]:
        """
        获取一个可用代理

        Returns:
            ProxyInfo对象，如果没有可用代理返回None
        """
        with self._lock:
            if not self._proxies:
                # 尝试从API获取
                if self._api_url:
                    self._fetch_from_api()

                if not self._proxies:
                    logger.warning("No proxies available")
                    return None

            # 过滤可用代理
            available = [
                p for p in self._proxies
                if p.status == ProxyStatus.AVAILABLE
                and p.fail_count < self._config['max_fail_count']
            ]

            # 如果没有可用的，尝试恢复冷却中的代理
            if not available:
                available = self._recover_cooldown_proxies()

            if not available:
                logger.warning("All proxies are unavailable or in cooldown")
                return None

            # 按成功率排序，选择成功率最高的
            available.sort(key=lambda p: p.success_rate, reverse=True)

            # 使用加权随机选择（成功率高的更容易被选中）
            proxy = random.choices(
                available,
                weights=[p.success_rate + 0.1 for p in available],  # +0.1避免权重为0
                k=1
            )[0]

            proxy.status = ProxyStatus.IN_USE
            proxy.last_used = datetime.now()

            logger.debug(f"Selected proxy: {proxy.address} (success rate: {proxy.success_rate:.2%})")
            return proxy

    def get_proxy_dict(self) -> Optional[Dict[str, str]]:
        """
        获取代理字典格式（requests库使用）

        Returns:
            {'http': 'http://...', 'https': 'http://...'} 或 None
        """
        proxy = self.get_proxy()
        if not proxy:
            return None

        proxy_url = proxy.url
        return {
            'http': proxy_url,
            'https': proxy_url
        }

    def release_proxy(self, proxy: ProxyInfo, success: bool = True) -> None:
        """
        释放代理，更新状态

        Args:
            proxy: 代理对象
            success: 是否成功
        """
        with self._lock:
            if success:
                proxy.success_count += 1
                proxy.status = ProxyStatus.AVAILABLE
            else:
                proxy.fail_count += 1
                proxy.last_failed = datetime.now()

                if proxy.fail_count >= self._config['max_fail_count']:
                    proxy.status = ProxyStatus.COOLDOWN
                    logger.warning(f"Proxy {proxy.address} moved to cooldown (fail count: {proxy.fail_count})")
                else:
                    proxy.status = ProxyStatus.AVAILABLE

            logger.debug(f"Released proxy {proxy.address}: success={success}, fail_count={proxy.fail_count}")

    # ==================== 代理池管理 ====================

    def _fetch_from_api(self) -> None:
        """从代理API获取新代理"""
        if not self._api_url:
            return

        # 检查获取间隔
        if self._last_fetch_time and datetime.now() - self._last_fetch_time < self._fetch_interval:
            return

        try:
            logger.info("Fetching proxies from API...")

            headers = {}
            if self._api_key:
                headers['Authorization'] = f'Bearer {self._api_key}'

            response = requests.get(
                self._api_url,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()

            # 解析不同API格式
            proxies = self._parse_api_response(data)

            # 添加到代理池
            for proxy_info in proxies:
                # 避免重复
                if not any(p.address == proxy_info.address for p in self._proxies):
                    self._proxies.append(proxy_info)

            self._last_fetch_time = datetime.now()
            logger.info(f"Fetched {len(proxies)} proxies from API")

        except Exception as e:
            logger.error(f"Failed to fetch proxies from API: {e}")

    def _parse_api_response(self, data: Any) -> List[ProxyInfo]:
        """
        解析代理API响应（支持多种格式）

        常见格式：
        1. 快代理: {"data": {"proxy_list": ["ip:port", ...]}}
        2. 芝麻代理: {"code": 0, "data": [{"ip": "x.x.x.x", "port": 8080}, ...]}
        3. 自定义: [{"host": "...", "port": ...}, ...]
        """
        proxies = []

        try:
            # 格式1: 快代理格式
            if isinstance(data, dict) and 'data' in data:
                inner = data['data']
                if isinstance(inner, dict) and 'proxy_list' in inner:
                    for item in inner['proxy_list']:
                        proxy = self._parse_proxy_line(item)
                        if proxy:
                            proxies.append(proxy)

                # 格式2: 芝麻代理格式
                elif isinstance(inner, list):
                    for item in inner:
                        if isinstance(item, dict):
                            host = item.get('ip') or item.get('host')
                            port = item.get('port')
                            if host and port:
                                proxies.append(ProxyInfo(
                                    host=host,
                                    port=int(port),
                                    protocol=item.get('protocol', 'http')
                                ))

            # 格式3: 自定义列表格式
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        proxy = self._parse_proxy_line(item)
                        if proxy:
                            proxies.append(proxy)
                    elif isinstance(item, dict):
                        host = item.get('host') or item.get('ip')
                        port = item.get('port')
                        if host and port:
                            proxies.append(ProxyInfo(
                                host=host,
                                port=int(port),
                                protocol=item.get('protocol', 'http'),
                                username=item.get('username'),
                                password=item.get('password')
                            ))

        except Exception as e:
            logger.error(f"Failed to parse API response: {e}")

        return proxies

    def _recover_cooldown_proxies(self) -> List[ProxyInfo]:
        """恢复冷却时间已过的代理"""
        now = datetime.now()
        cooldown_expired = now - timedelta(minutes=self._config['cooldown_minutes'])

        recovered = []
        for proxy in self._proxies:
            if proxy.status == ProxyStatus.COOLDOWN:
                if proxy.last_failed and proxy.last_failed < cooldown_expired:
                    proxy.status = ProxyStatus.AVAILABLE
                    proxy.fail_count = 0  # 重置失败计数
                    recovered.append(proxy)
                    logger.info(f"Recovered proxy from cooldown: {proxy.address}")

        return recovered

    def health_check(self) -> Dict[str, int]:
        """
        对所有代理进行健康检查

        Returns:
            检查结果统计
        """
        if not REQUESTS_AVAILABLE:
            logger.warning("requests not available, skipping health check")
            return {'checked': 0, 'alive': 0, 'dead': 0}

        results = {'checked': 0, 'alive': 0, 'dead': 0}

        logger.info(f"Starting health check for {len(self._proxies)} proxies...")

        for proxy in self._proxies[:]:  # 使用切片创建副本
            try:
                start_time = time.time()
                response = requests.get(
                    self._config['check_url'],
                    proxies={'http': proxy.url, 'https': proxy.url},
                    timeout=self._config['check_timeout']
                )
                elapsed = time.time() - start_time

                if response.status_code == 200:
                    proxy.status = ProxyStatus.AVAILABLE
                    proxy.response_time = elapsed
                    results['alive'] += 1
                    logger.debug(f"Proxy {proxy.address} is alive ({elapsed:.2f}s)")
                else:
                    proxy.status = ProxyStatus.FAILED
                    results['dead'] += 1
                    logger.debug(f"Proxy {proxy.address} returned status {response.status_code}")

            except Exception as e:
                proxy.status = ProxyStatus.FAILED
                results['dead'] += 1
                logger.debug(f"Proxy {proxy.address} failed: {e}")

            results['checked'] += 1

        logger.info(f"Health check complete: {results['alive']}/{results['checked']} proxies alive")
        return results

    def remove_dead_proxies(self) -> int:
        """移除失效的代理"""
        with self._lock:
            initial_count = len(self._proxies)
            self._proxies = [
                p for p in self._proxies
                if p.status != ProxyStatus.FAILED
                or p.fail_count < self._config['max_fail_count']
            ]
            removed = initial_count - len(self._proxies)
            if removed > 0:
                logger.info(f"Removed {removed} dead proxies")
            return removed

    # ==================== 统计信息 ====================

    def get_stats(self) -> Dict[str, Any]:
        """获取代理池统计信息"""
        status_counts = {}
        for status in ProxyStatus:
            count = sum(1 for p in self._proxies if p.status == status)
            status_counts[status.value] = count

        return {
            'total': len(self._proxies),
            'by_status': status_counts,
            'avg_success_rate': sum(p.success_rate for p in self._proxies) / len(self._proxies) if self._proxies else 0
        }

    @classmethod
    def get_instance(cls) -> 'ProxyPool':
        """获取全局单例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置代理池（测试用）"""
        with cls._lock:
            if cls._instance:
                cls._instance._proxies = []
                cls._instance._initialized = False
            cls._instance = None


# ============================= 带代理的安全请求函数 =============================

def safe_request_with_proxy(
    url: str,
    method: str = 'GET',
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 15,
    use_delay: bool = True,
    use_proxy: bool = True,
    max_proxy_retries: int = 3,
    **kwargs
) -> Any:
    """
    安全的HTTP请求函数，集成反爬虫防护和代理池

    Args:
        url: 请求URL
        method: HTTP方法
        headers: 自定义请求头
        timeout: 超时时间(秒)
        use_delay: 是否使用延迟控制
        use_proxy: 是否使用代理
        max_proxy_retries: 代理失败时的最大重试次数
        **kwargs: 传递给requests的其他参数

    Returns:
        Response对象
    """
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests library is required")

    # 延迟控制
    if use_delay:
        delay_controller = DelayController.get_instance()
        delay_controller.wait()

    # 生成请求头
    if headers is None:
        headers = HeaderGenerator.generate_headers()

    proxy_pool = ProxyPool.get_instance() if use_proxy else None
    current_proxy = None
    last_exception = None

    # 代理重试循环
    for attempt in range(max_proxy_retries + 1):
        try:
            proxies = {"http": None, "https": None}  # 默认禁用系统代理
            if use_proxy and proxy_pool:
                current_proxy = proxy_pool.get_proxy()
                if current_proxy:
                    proxies = {
                        'http': current_proxy.url,
                        'https': current_proxy.url
                    }
                    logger.debug(f"Using proxy: {current_proxy.address}")

            # 执行请求
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout, proxies=proxies, **kwargs)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, timeout=timeout, proxies=proxies, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            # 检查是否被反爬虫拦截
            if _is_blocked(response):
                raise Exception("Blocked by anti-crawler system")

            # 成功，释放代理
            if current_proxy:
                proxy_pool.release_proxy(current_proxy, success=True)

            return response

        except Exception as e:
            last_exception = e

            # 释放失败的代理
            if current_proxy:
                proxy_pool.release_proxy(current_proxy, success=False)
                logger.warning(f"Proxy {current_proxy.address} failed: {e}")

            # 如果不是代理问题或已达到最大重试，抛出异常
            if attempt >= max_proxy_retries or not use_proxy:
                logger.error(f"Request failed after {attempt + 1} attempts: {e}")
                raise last_exception

            # 等待后重试
            time.sleep(random.uniform(1, 3))

    raise last_exception


def _is_blocked(response: Any) -> bool:
    """
    检查响应是否被反爬虫系统拦截

    Args:
        response: requests Response对象

    Returns:
        是否被拦截
    """
    # 检查状态码
    if response.status_code in [403, 429, 503]:
        # 检查响应内容
        content = response.text.lower()
        block_indicators = [
            'blocked', 'captcha', 'verify', 'forbidden',
            'access denied', 'too many requests', 'rate limit',
            'cloudflare', 'ddos protection'
        ]
        if any(indicator in content for indicator in block_indicators):
            return True

    return False


# ============================= 便捷函数 =============================

def get_proxy() -> Optional[ProxyInfo]:
    """获取一个代理（便捷函数）"""
    return ProxyPool.get_instance().get_proxy()


def get_proxy_dict() -> Optional[Dict[str, str]]:
    """获取代理字典（便捷函数）"""
    return ProxyPool.get_instance().get_proxy_dict()


def setup_proxy_pool(
    proxy_file: Optional[str] = None,
    proxy_list: Optional[List[str]] = None,
    api_url: Optional[str] = None,
    api_key: Optional[str] = None
) -> ProxyPool:
    """
    设置代理池（便捷函数）

    Args:
        proxy_file: 代理文件路径
        proxy_list: 代理列表
        api_url: 代理API URL
        api_key: API密钥

    Returns:
        ProxyPool实例
    """
    pool = ProxyPool.get_instance()

    if proxy_file:
        pool.load_from_file(proxy_file)

    if proxy_list:
        pool.load_from_list(proxy_list)

    if api_url:
        pool._api_url = api_url
        pool._api_key = api_key
        pool._fetch_from_api()

    return pool


# 免费代理源（示例，实际使用时需要更新）
FREE_PROXY_SOURCES = {
    'kuaidaili': 'https://www.kuaidaili.com/api/getproxy/?orderid=xxx&num=100&format=json',
    'zhima': 'http://webapi.http.zhimacangku.com/getip?num=100&type=2&pro=&city=0&yys=0&port=1&time=1&ts=1&ys=1&cs=1&lb=1&sb=0&pb=4&mr=1&regions=',
    'ip3366': 'http://www.ip3366.net/api/?key=xxx&num=100&format=json',
}