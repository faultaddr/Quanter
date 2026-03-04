"""
反爬虫防护模块 v1.0

核心功能：
- User-Agent管理
- 请求头生成
- 延迟控制
- 代理管理
- 指数退避重试

用于规避AShare爬虫被封的风险
"""

import random
import time
from typing import List, Dict, Optional, Callable, Any
import functools
import logging

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
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple = (Exception,)
) -> Callable:
    """
    失败时指数退避重试装饰器

    Args:
        max_retries: 最大重试次数
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