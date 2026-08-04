"""Enhanced Data Fetcher with support for multiple data sources including Ashare, EastMoney, Tushare, and AkShare."""

import os
import time
import json
import requests
import tushare as ts
import pandas as pd
import threading
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from tqdm import tqdm
from ....domain.interfaces.data_provider import IDataProvider
from ....core.errors import DataProviderError
from ....core.registry import registry, ComponentType
from ....core.logging import get_logger
from ...cache import LocalDataCache

# 导入反爬虫防护模块
from ..anti_crawler import (
    UserAgentManager,
    HeaderGenerator,
    DelayController,
    retry_on_failure,
    safe_request,
    safe_request_with_proxy,
    get_sina_headers,
    get_tencent_headers,
    get_eastmoney_headers,
    ProxyPool,
    ProxyInfo,
    setup_proxy_pool,
    get_proxy,
    get_proxy_dict,
)

# Try to import AkShare, but make it optional
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    ak = None

# Try to import BaoStock, but make it optional
try:
    import baostock as bs
    BAOSTOCK_AVAILABLE = True
except ImportError:
    BAOSTOCK_AVAILABLE = False
    bs = None

logger = get_logger(__name__)


def _is_index_code(symbol: str) -> bool:
    """判断是否为指数代码"""
    code = symbol.replace('.SH', '').replace('.SZ', '').strip()
    # 上证指数: 000001-000999, 880xxx, 999xxx
    # 深证指数: 399001-399999
    if code.startswith(('000', '880', '999')) and len(code) == 6:
        return True
    if code.startswith('399') and len(code) == 6:
        return True
    return False


def _safe_json_loads(content: bytes) -> Any:
    """安全解析 JSON 响应，支持多种编码

    中国金融数据源（新浪、腾讯、BaoStock）可能返回 GBK/GB2312 编码的数据，
    而不是 UTF-8。此函数按优先级尝试多种编码。

    Args:
        content: HTTP 响应的原始字节内容

    Returns:
        解析后的 JSON 对象

    Raises:
        json.JSONDecodeError: 所有编码尝试都失败时抛出
    """
    # 尝试 UTF-8（优先）
    try:
        return json.loads(content.decode('utf-8'))
    except UnicodeDecodeError:
        pass

    # 尝试 GBK（中国金融数据源常用编码）
    try:
        return json.loads(content.decode('gbk'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    # 尝试 GB2312
    try:
        return json.loads(content.decode('gb2312'))
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    # 最后尝试忽略错误解码
    try:
        decoded = content.decode('utf-8', errors='ignore')
        return json.loads(decoded)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response after trying multiple encodings: {e}")
        raise


# ==================== BaoStock 线程安全登录管理 ====================
class BaoStockSessionManager:
    """
    BaoStock 线程安全的会话管理器

    BaoStock 的 login/logout 不是线程安全的，多线程并发调用会导致
    [Errno 9] Bad file descriptor 错误。
    此管理器确保全局只有一个活跃的 BaoStock 会话。
    """
    _lock = threading.Lock()
    _logged_in = False
    _ref_count = 0

    @classmethod
    def login(cls) -> bool:
        """线程安全的 BaoStock 登录"""
        if not BAOSTOCK_AVAILABLE:
            return False

        with cls._lock:
            if not cls._logged_in:
                try:
                    result = bs.login()
                    if result.error_code == '0':
                        cls._logged_in = True
                        logger.debug("BaoStock logged in successfully")
                    else:
                        logger.warning(f"BaoStock login failed: {result.error_msg}")
                        return False
                except Exception as e:
                    logger.error(f"BaoStock login exception: {str(e)}")
                    return False
            cls._ref_count += 1
            return True

    @classmethod
    def logout(cls):
        """线程安全的 BaoStock 登出（引用计数归零时才真正登出）"""
        if not BAOSTOCK_AVAILABLE:
            return

        with cls._lock:
            cls._ref_count = max(0, cls._ref_count - 1)
            # 保持登录状态，不登出（避免频繁登录/登出）
            # 如果需要强制登出，可以调用 force_logout()

    @classmethod
    def force_logout(cls):
        """强制登出 BaoStock"""
        if not BAOSTOCK_AVAILABLE:
            return

        with cls._lock:
            if cls._logged_in:
                try:
                    bs.logout()
                    cls._logged_in = False
                    cls._ref_count = 0
                    logger.debug("BaoStock logged out successfully")
                except Exception as e:
                    logger.error(f"BaoStock logout exception: {str(e)}")


# ==================== Ashare 数据源（最高优先级）====================
# 基于 https://github.com/mpquant/Ashare 的双核心架构
# 新浪 + 腾讯数据源，自动故障切换，免费无需Token

class AshareFetcher:
    """
    Ashare 数据获取器（集成反爬虫防护）

    特点：
    - 双核心：新浪(主力) + 腾讯(备用)，自动故障切换
    - 免费：无需注册和API Token
    - 实时：支持日线、周线、月线、分钟线
    - 防护：随机User-Agent、延迟控制、指数退避重试
    """

    # 全局延迟控制器
    _delay_controller: Optional[DelayController] = None

    @classmethod
    def _get_delay_controller(cls) -> DelayController:
        """获取延迟控制器实例"""
        if cls._delay_controller is None:
            cls._delay_controller = DelayController(min_delay=0.1, max_delay=1.0)
        return cls._delay_controller

    @staticmethod
    def _normalize_code(code: str) -> str:
        """标准化股票代码为新浪/腾讯格式"""
        # 保存原始后缀信息（用于指数代码的正确市场判断）
        is_sh = '.SH' in code.upper() or '.XSHG' in code.upper()
        is_sz = '.SZ' in code.upper() or '.XSHE' in code.upper()

        # 处理聚宽格式 000001.XSHG -> sh000001
        code = code.replace('.XSHG', '').replace('.XSHE', '')
        code = code.replace('.SH', '').replace('.SZ', '')

        if code.startswith(('sh', 'sz', 'SH', 'SZ')):
            return code.lower()

        # 如果原始代码明确标记了市场，优先使用
        # 例如 000300.SH -> sh000300（沪深300指数，上证市场）
        if is_sh:
            return f'sh{code}'
        if is_sz:
            return f'sz{code}'

        # 根据代码判断市场
        if code.startswith(('5', '6', '9')):
            return f'sh{code}'
        else:
            return f'sz{code}'

    @staticmethod
    def _get_price_day_tx(code: str, end_date: str = '', count: int = 100, frequency: str = '1d') -> pd.DataFrame:
        """腾讯日线数据获取（带反爬虫防护）"""
        unit = 'week' if frequency == '1w' else 'month' if frequency == '1M' else 'day'

        if end_date:
            if isinstance(end_date, datetime):
                end_date = end_date.strftime('%Y-%m-%d')
            end_date = end_date.split(' ')[0]
            if end_date == datetime.now().strftime('%Y-%m-%d'):
                end_date = ''

        url = f'http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},{unit},,{end_date},{count},qfq'

        try:
            # 使用反爬虫请求头
            headers = get_tencent_headers()
            response = safe_request(url, headers=headers, timeout=10, use_delay=True)
            st = _safe_json_loads(response.content)
            ms = 'qfq' + unit
            stk = st['data'][code]
            buf = stk[ms] if ms in stk else stk[unit]

            # 腾讯API返回的列数可能变化，需要动态处理
            # 标准列: ['timestamp', 'open', 'close', 'high', 'low', 'volume']
            # 有时可能多返回一列
            if not buf:
                return pd.DataFrame()

            # 直接从数据创建DataFrame，不指定列名
            df = pd.DataFrame(buf)

            # 根据列数确定列名
            num_cols = len(df.columns)
            if num_cols >= 6:
                # 标准格式: 日期, 开, 收, 高, 低, 量 [, 额]
                df.columns = ['timestamp', 'open', 'close', 'high', 'low', 'volume'] + \
                             (['amount'] if num_cols == 7 else [])
            else:
                # 列数不够，返回空
                return pd.DataFrame()

            # 转换数据类型
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            for col in ['open', 'close', 'high', 'low', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 如果没有 amount 列，用 close * volume 估算
            if 'amount' not in df.columns:
                df['amount'] = df['close'] * df['volume'] * 100  # 手转换为股

            return df
        except Exception as e:
            logger.warning(f"腾讯数据获取失败: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def _get_price_sina(code: str, end_date: str = '', count: int = 100, frequency: str = '1d') -> pd.DataFrame:
        """新浪全周期数据获取（带反爬虫防护）"""
        freq_map = {'1d': '240m', '1w': '1200m', '1M': '7200m'}
        frequency = freq_map.get(frequency, frequency)

        ts = int(frequency[:-1]) if frequency[:-1].isdigit() else 1
        mcount = count

        if end_date and frequency in ['240m', '1200m', '7200m']:
            end_dt = pd.to_datetime(end_date) if not isinstance(end_date, datetime) else end_date
            unit = 4 if frequency == '1200m' else 29 if frequency == '7200m' else 1
            count = count + (datetime.now() - end_dt).days // unit

        url = f'http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale={ts}&ma=5&datalen={count}'

        try:
            # 使用反爬虫请求头
            headers = get_sina_headers()
            response = safe_request(url, headers=headers, timeout=10, use_delay=True)
            dstr = _safe_json_loads(response.content)

            df = pd.DataFrame(dstr, columns=['day', 'open', 'high', 'low', 'close', 'volume'])
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)

            df['day'] = pd.to_datetime(df['day'])
            df = df.rename(columns={'day': 'timestamp'})

            # 添加 amount 列
            df['amount'] = df['close'] * df['volume'] * 100

            if end_date and frequency in ['240m', '1200m', '7200m']:
                end_dt = pd.to_datetime(end_date) if not isinstance(end_date, datetime) else end_date
                df = df[df['timestamp'] <= end_dt][-mcount:]

            return df
        except Exception as e:
            logger.debug(f"新浪数据获取失败: {str(e)}")
            return pd.DataFrame()

    @classmethod
    def get_price(cls, code: str, end_date: str = '', count: int = 100, frequency: str = '1d') -> pd.DataFrame:
        """
        获取股票行情数据

        Args:
            code: 股票代码，支持多种格式 (sh600519, 600519.XSHG, 600519)
            end_date: 结束日期 (YYYY-MM-DD)
            count: 获取数量
            frequency: 周期 ('1d', '1w', '1M', '1m', '5m', '15m', '30m', '60m')

        Returns:
            DataFrame with columns: time, open, high, low, close, volume, amount
        """
        xcode = cls._normalize_code(code)

        # 日线、周线、月线
        if frequency in ['1d', '1w', '1M']:
            # 主力：新浪，备用：腾讯
            try:
                df = cls._get_price_sina(xcode, end_date=end_date, count=count, frequency=frequency)
                if not df.empty:
                    df.attrs["concrete_source"] = "sina"
                    return df
            except Exception:
                pass

            try:
                df = cls._get_price_day_tx(xcode, end_date=end_date, count=count, frequency=frequency)
                if not df.empty:
                    df.attrs["concrete_source"] = "tencent"
                    return df
            except Exception:
                pass

        # 分钟线（暂不支持，返回空）
        if frequency in ['1m', '5m', '15m', '30m', '60m']:
            logger.warning(f"Ashare 暂不支持分钟线数据: {frequency}")

        return pd.DataFrame()


def setup_tushare_api(token: str):
    """Setup and return Tushare API with the given token."""
    ts.set_token(token)
    pro = ts.pro_api()
    return pro


@registry.register(ComponentType.DATA_PROVIDER, "enhanced_data_fetcher")
class EnhancedDataFetcher(IDataProvider):
    """Enhanced data fetcher supporting Tushare, EastMoney, and AkShare data sources."""

    def __init__(
        self,
        tushare_token: str = None,
        eastmoney_cookie: str = None,
        use_akshare: bool = True,
        max_workers: int = 10,
        cache_dir: str = ".cache/stock_data",
        cache_ttl: int = 86400,
        use_cache: bool = True,
        proxy_file: str = None,
        proxy_list: List[str] = None,
        proxy_api_url: str = None,
        proxy_api_key: str = None,
        use_proxy: bool = False
    ):
        """
        Initialize enhanced data fetcher.

        Args:
            tushare_token: Tushare API token. If None, will try to get from TUSHARE_TOKEN environment variable.
            eastmoney_cookie: EastMoney cookie string. If None, will try to get from EASTMONEY_COOKIE environment variable.
            use_akshare: Whether to use AkShare as a fallback data source (default: True).
            max_workers: Maximum number of parallel workers for concurrent fetching (default: 10).
            cache_dir: Directory for local data cache (default: .cache/stock_data).
            cache_ttl: Cache time-to-live in seconds (default: 86400 = 1 day).
            use_cache: Whether to use local cache (default: True).
            proxy_file: 代理列表文件路径(每行一个代理，格式: host:port)
            proxy_list: 代理列表 ['http://1.2.3.4:8080', ...]
            proxy_api_url: 代理API URL（付费代理服务商提供）
            proxy_api_key: 代理API密钥
            use_proxy: 是否启用代理池（默认 False）
        """
        self.tushare_token = tushare_token or os.getenv("TUSHARE_TOKEN")
        self.eastmoney_cookie = eastmoney_cookie or os.getenv("EASTMONEY_COOKIE")
        self.use_akshare = use_akshare and AKSHARE_AVAILABLE
        self.max_workers = max_workers
        self.use_cache = use_cache
        self.use_proxy = use_proxy

        # Setup Tushare API if token available
        self.pro_api = None
        self._tushare_initialized = False
        if self.tushare_token:
            try:
                self.pro_api = setup_tushare_api(self.tushare_token)
            except Exception as e:
                logger.warning(f"Failed to initialize Tushare: {e}")

        # TuShare rate limiting (50 stocks per minute)
        self._tushare_request_count = 0
        self._tushare_last_reset = time.time()

        # EastMoney headers - 使用反爬虫模块生成
        self.eastmoney_headers = get_eastmoney_headers()
        if self.eastmoney_cookie:
            self.eastmoney_headers['Cookie'] = self.eastmoney_cookie

        # 延迟控制器 - 使用更保守的延迟设置
        self._delay_controller = DelayController.get_instance()
        if use_proxy:
            # 使用代理时可以用较短延迟
            self._delay_controller.min_delay = 0.5
            self._delay_controller.max_delay = 2.0
        else:
            # 不使用代理时使用更长延迟避免被封
            self._delay_controller.min_delay = 2.0
            self._delay_controller.max_delay = 5.0

        # 代理池设置
        self._proxy_pool = None
        if use_proxy or proxy_file or proxy_list or proxy_api_url:
            self._proxy_pool = setup_proxy_pool(
                proxy_file=proxy_file,
                proxy_list=proxy_list,
                api_url=proxy_api_url,
                api_key=proxy_api_key
            )
            logger.debug(f"Proxy pool initialized with {len(self._proxy_pool._proxies)} proxies")

        # Thread pool for parallel fetching
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

        # Local cache
        if self.use_cache:
            self._cache = LocalDataCache(
                cache_dir=cache_dir,
                default_ttl=cache_ttl
            )
            logger.debug(f"Local cache enabled at {cache_dir}")
        else:
            self._cache = None

        # AkShare availability
        if self.use_akshare:
            logger.debug("AkShare is available and will be used as fallback")
        elif not AKSHARE_AVAILABLE:
            logger.warning("AkShare is not installed. Install it with: pip install akshare")

    def _make_request(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: int = 15,
        use_delay: bool = True,
        method: str = 'GET',
        **kwargs
    ) -> Any:
        """
        统一的HTTP请求方法，自动处理代理和反爬虫防护

        Args:
            url: 请求URL
            headers: 请求头
            timeout: 超时时间
            use_delay: 是否使用延迟
            method: HTTP方法
            **kwargs: 其他参数

        Returns:
            Response对象
        """
        if self.use_proxy and self._proxy_pool:
            return safe_request_with_proxy(
                url,
                method=method,
                headers=headers,
                timeout=timeout,
                use_delay=use_delay,
                use_proxy=True,
                **kwargs
            )
        else:
            return safe_request(
                url,
                method=method,
                headers=headers,
                timeout=timeout,
                use_delay=use_delay,
                **kwargs
            )

    def initialize(self) -> None:
        """Initialize the data fetcher connections."""
        try:
            # Skip Tushare connection test - it's unreliable
            # Just mark as initialized and fail later if needed
            logger.debug("Skipping Tushare connection test (unreliable)")

            # Verify EastMoney connection
            if self.eastmoney_cookie:
                # Perform a simple request to verify cookie is valid
                try:
                    test_url = "https://np-analyse.eastmoney.com/api/qt/ulist.np/get?po=1&pz=1&pn=1&np=1&fltt=2&invt=2&wbp2u=12915131124252524252135421&fid=f3&fs=m:0+t:6+f:!50&fields=f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f26,f22,f33,f11,f62,f128,f136,f115,f152"
                    response = self._make_request(
                        test_url,
                        headers=self.eastmoney_headers,
                        timeout=10,
                        use_delay=False  # 初始化时不需要延迟
                    )
                    # Just check if we get a response without error
                    logger.debug("EastMoney connection verified")
                except Exception as e:
                    logger.warning(f"Could not verify EastMoney connection: {str(e)}")

            self._tushare_initialized = True
            logger.debug("EnhancedDataFetcher initialized successfully (Ashare as primary)")
        except Exception as e:
            raise DataProviderError(f"Failed to initialize EnhancedDataFetcher: {str(e)}")

    def _fetch_from_ashare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Ashare 获取数据（最高优先级）

        Ashare 特点：
        - 双核心：新浪(主力) + 腾讯(备用)，自动故障切换
        - 免费：无需注册和API Token
        - 实时：支持日线、周线、月线
        """
        try:
            # 计算需要获取的数据条数
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            days = (end_dt - start_dt).days + 1
            count = min(days + 30, 1000)  # 多取一些数据确保覆盖

            logger.debug(f"Fetching {symbol} from Ashare (primary source)")

            df = AshareFetcher.get_price(
                code=symbol,
                end_date=end_date,
                count=count,
                frequency='1d'
            )

            if df.empty:
                logger.warning(f"Ashare returned no data for {symbol}")
                return pd.DataFrame()

            # AshareFetcher 已经返回正确格式的 DataFrame
            # 确保 timestamp 是 datetime 类型
            if 'timestamp' not in df.columns:
                logger.warning(f"Ashare data missing timestamp column, columns: {list(df.columns)}")
                return pd.DataFrame()

            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # 过滤日期范围
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]

            # 添加必要的列
            df['symbol'] = symbol
            df['timeframe'] = '1d'

            # 确保列顺序正确
            expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount', 'timeframe', 'symbol']
            available_cols = [col for col in expected_cols if col in df.columns]
            df = df[available_cols]

            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.debug(f"Fetched {len(df)} bars from Ashare for {symbol}")
            return df

        except Exception as e:
            logger.warning(f"Ashare fetch failed for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _fetch_from_eastmoney(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from EastMoney if available."""
        if not self.eastmoney_cookie:
            logger.warning("EastMoney cookie not available, falling back to Tushare")
            return pd.DataFrame()

        try:
            # Format symbol for EastMoney
            # Convert to EastMoney format (000001.sz, 600000.sh)
            if '.' in symbol:
                base_symbol = symbol.split('.')[0]
            else:
                base_symbol = symbol

            if len(base_symbol) == 6:
                if base_symbol.startswith(('5', '6', '9')):
                    # Shanghai stocks
                    em_symbol = f"{base_symbol}.sh"
                else:
                    # Shenzhen stocks
                    em_symbol = f"{base_symbol}.sz"
            else:
                em_symbol = symbol

            # EastMoney K-line data API (sample format - may need adjustment based on actual API)
            # Using a typical EastMoney API endpoint for historical K data
            url = f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
            params = {
                'secid': f'0.{base_symbol}' if base_symbol.startswith(('0', '3')) else f'1.{base_symbol}',
                'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
                'fields1': 'f1,f2,f3,f4,f5,f6',
                'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
                'klt': '101',  # Daily data
                'fqt': '1',    # Forward adjustment
                'beg': start_date.replace('-', ''),
                'end': end_date.replace('-', '')
            }

            # 使用反爬虫防护和代理
            response = self._make_request(
                url,
                headers=self.eastmoney_headers,
                timeout=15,
                use_delay=True,
                params=params
            )
            data = response.json()

            if data.get('rc') != 0:
                logger.error(f"EastMoney API error for {symbol}: {data.get('msg')}")
                return pd.DataFrame()

            klinedata = data.get('data', {}).get('klines', [])
            if not klinedata:
                logger.warning(f"No EastMoney data found for {symbol}")
                return pd.DataFrame()

            # Parse kline data
            parsed_data = []
            for kline in klinedata:
                parts = kline.split(',')
                if len(parts) >= 7:
                    parsed_data.append({
                        'timestamp': parts[0],  # Date string
                        'open': float(parts[1]),
                        'close': float(parts[2]),
                        'high': float(parts[3]),
                        'low': float(parts[4]),
                        'volume': int(float(parts[5])),  # Convert from float to int
                        'amount': float(parts[6]) if len(parts) > 6 else 0.0,
                    })

            if not parsed_data:
                logger.warning(f"Parsed EastMoney data is empty for {symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(parsed_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['symbol'] = symbol
            df['timeframe'] = '1d'

            # Reorder columns to match expected format
            expected_cols = [
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'amount', 'timeframe', 'symbol'
            ]
            available_cols = [col for col in expected_cols if col in df.columns]
            df = df[available_cols]

            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            return df

        except Exception as e:
            logger.error(f"Error fetching EastMoney data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _fetch_from_akshare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from AkShare as fallback."""
        if not self.use_akshare or not AKSHARE_AVAILABLE:
            logger.warning("AkShare not available")
            return pd.DataFrame()

        try:
            # Format symbol for AkShare (remove .SH/.SZ suffix)
            if '.' in symbol:
                base_symbol = symbol.split('.')[0]
            else:
                base_symbol = symbol

            # Convert dates from YYYY-MM-DD to YYYYMMDD format
            start_formatted = start_date.replace('-', '')
            end_formatted = end_date.replace('-', '')

            logger.debug(f"Fetching {symbol} from AkShare using base symbol {base_symbol}")

            # Use AkShare's stock_zh_a_hist interface with minimal retry for speed
            max_retries = 1  # Reduced from 3 to minimize latency
            df = pd.DataFrame()
            last_error = None

            for attempt in range(max_retries):
                try:
                    df = ak.stock_zh_a_hist(
                        symbol=base_symbol,
                        period="daily",
                        start_date=start_formatted,
                        end_date=end_formatted,
                        adjust="qfq"  # Forward adjusted
                    )
                    if not df.empty:
                        break
                except Exception as e:
                    last_error = e
                    logger.warning(f"AkShare attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)  # Wait before retry

            if df.empty:
                error_msg = str(last_error) if last_error else "No data returned"
                logger.warning(f"No AkShare data found for {symbol}: {error_msg}")
                return pd.DataFrame()

            # Rename columns to match expected format
            # AkShare columns: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 振幅, 涨跌幅, 涨跌额, 换手率
            column_mapping = {
                "日期": "timestamp",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount"
            }

            df = df.rename(columns=column_mapping)

            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Add timeframe and symbol columns
            df["timeframe"] = "1d"
            df["symbol"] = symbol

            # Select and reorder columns to match expected format
            expected_cols = [
                "timestamp", "open", "high", "low", "close", "volume",
                "amount", "timeframe", "symbol"
            ]
            available_cols = [col for col in expected_cols if col in df.columns]
            df = df[available_cols]

            # Sort by timestamp
            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.debug(f"Fetched {len(df)} bars from AkShare for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching AkShare data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _fetch_index_from_akshare(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """从 AkShare 获取指数数据（使用 index_zh_a_hist 接口）"""
        if not self.use_akshare or not AKSHARE_AVAILABLE:
            return pd.DataFrame()

        try:
            # 去掉后缀，AkShare 指数接口只需要纯数字代码
            if '.' in symbol:
                base_symbol = symbol.split('.')[0]
            else:
                base_symbol = symbol

            start_formatted = start_date.replace('-', '')
            end_formatted = end_date.replace('-', '')

            logger.debug(f"Fetching index {symbol} from AkShare using index_zh_a_hist")

            max_retries = 1
            df = pd.DataFrame()
            last_error = None

            for attempt in range(max_retries):
                try:
                    df = ak.index_zh_a_hist(
                        symbol=base_symbol,
                        period="daily",
                        start_date=start_formatted,
                        end_date=end_formatted,
                    )
                    if not df.empty:
                        break
                except Exception as e:
                    last_error = e
                    logger.warning(f"AkShare index attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(1)

            if df.empty:
                error_msg = str(last_error) if last_error else "No data returned"
                logger.warning(f"No AkShare index data for {symbol}: {error_msg}")
                return pd.DataFrame()

            # 列名映射：AkShare 指数接口与个股接口相同
            column_mapping = {
                "日期": "timestamp",
                "开盘": "open",
                "收盘": "close",
                "最高": "high",
                "最低": "low",
                "成交量": "volume",
                "成交额": "amount"
            }

            df = df.rename(columns=column_mapping)

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timeframe"] = "1d"
            df["symbol"] = symbol

            expected_cols = [
                "timestamp", "open", "high", "low", "close", "volume",
                "amount", "timeframe", "symbol"
            ]
            available_cols = [col for col in expected_cols if col in df.columns]
            df = df[available_cols]

            df.sort_values("timestamp", inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.debug(f"Fetched {len(df)} index bars from AkShare for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching AkShare index data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _fetch_index_from_tushare(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """从 TuShare 获取指数数据（使用 index_daily 接口）"""
        if not self._tushare_initialized:
            self.initialize()

        try:
            start_ts = start_date.strftime("%Y%m%d")
            end_ts = end_date.strftime("%Y%m%d")

            # TuShare 指数代码格式：000300.SH
            tushare_symbol = symbol
            if '.' not in symbol:
                if len(symbol) == 6:
                    if symbol.startswith(('5', '6', '9', '000')):
                        tushare_symbol = f"{symbol}.SH"
                    else:
                        tushare_symbol = f"{symbol}.SZ"

            if self._tushare_request_count >= 50:
                elapsed = time.time() - self._tushare_last_reset
                if elapsed < 60:
                    sleep_time = 60 - elapsed
                    logger.debug(f"TuShare rate limit, waiting {sleep_time:.1f}s")
                    time.sleep(sleep_time)
                self._tushare_request_count = 0
                self._tushare_last_reset = time.time()

            df = self.pro_api.index_daily(
                ts_code=tushare_symbol, start_date=start_ts, end_date=end_ts
            )

            if df.empty:
                logger.warning(f"No TuShare index data for {tushare_symbol}")
                return pd.DataFrame()

            self._tushare_request_count += 1

            df.rename(
                columns={
                    "ts_code": "symbol",
                    "trade_date": "timestamp",
                    "vol": "volume"
                },
                inplace=True,
            )

            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["timeframe"] = "1d"
            df["symbol"] = symbol

            expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'amount', 'timeframe', 'symbol']
            available_cols = [col for col in expected_cols if col in df.columns]
            df = df[available_cols]

            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)

            logger.debug(f"Fetched {len(df)} index bars from TuShare for {symbol}")
            return df

        except Exception as e:
            logger.warning(f"TuShare index fetch failed for {symbol}: {str(e)}")
            return pd.DataFrame()

    def _fetch_from_baostock(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from BaoStock as fallback (thread-safe)."""
        if not BAOSTOCK_AVAILABLE:
            logger.warning("BaoStock not available")
            return pd.DataFrame()

        try:
            # Format symbol for BaoStock (remove .SH/.SZ suffix, BaoStock uses sh.600000 / sz.000001 format)
            if '.' in symbol:
                base_symbol = symbol.split('.')[0]
                exchange = symbol.split('.')[1].upper()
                if exchange == 'SH':
                    bs_symbol = f"sh.{base_symbol}"
                else:
                    bs_symbol = f"sz.{base_symbol}"
            else:
                base_symbol = symbol
                if base_symbol.startswith(('5', '6', '9')):
                    bs_symbol = f"sh.{base_symbol}"
                else:
                    bs_symbol = f"sz.{base_symbol}"

            logger.debug(f"Fetching {symbol} from BaoStock using {bs_symbol}")

            # Use thread-safe session manager instead of direct bs.login()
            if not BaoStockSessionManager.login():
                logger.warning(f"BaoStock login failed, skipping {symbol}")
                return pd.DataFrame()

            try:
                # Query history k data
                # BaoStock date format: YYYY-MM-DD
                rs = bs.query_history_k_data_plus(
                    bs_symbol,
                    "date,open,high,low,close,volume,amount",
                    start_date=start_date,
                    end_date=end_date,
                    frequency="d",
                    adjustflag="3"  # 3 = post-adjusted (复权)
                )

                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())

                if not data_list:
                    logger.warning(f"No BaoStock data found for {symbol}")
                    return pd.DataFrame()

                # Create DataFrame
                df = pd.DataFrame(data_list, columns=rs.fields)

                # Convert columns to appropriate types
                df['date'] = pd.to_datetime(df['date'])
                df['open'] = pd.to_numeric(df['open'], errors='coerce')
                df['high'] = pd.to_numeric(df['high'], errors='coerce')
                df['low'] = pd.to_numeric(df['low'], errors='coerce')
                df['close'] = pd.to_numeric(df['close'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

                # Rename columns to match expected format
                df = df.rename(columns={
                    'date': 'timestamp',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'amount': 'amount'
                })

                # Add timeframe and symbol columns
                df['timeframe'] = '1d'
                df['symbol'] = symbol

                # Select and reorder columns
                expected_cols = [
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'amount', 'timeframe', 'symbol'
                ]
                available_cols = [col for col in expected_cols if col in df.columns]
                df = df[available_cols]

                # Drop rows with NaN values
                df = df.dropna(subset=['open', 'high', 'low', 'close'])

                logger.debug(f"Fetched {len(df)} bars from BaoStock for {symbol}")
                return df

            finally:
                # Decrement ref count but don't actually logout
                BaoStockSessionManager.logout()

        except Exception as e:
            error_msg = str(e)
            if 'utf-8' in error_msg.lower() or 'decode' in error_msg.lower() or 'codec' in error_msg.lower():
                logger.warning(f"BaoStock encoding error for {symbol}, this may be a temporary issue: {error_msg}")
            else:
                logger.error(f"Error fetching BaoStock data for {symbol}: {error_msg}")
            return pd.DataFrame()

    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV bars for the given symbols and timeframe.

        数据源优先级：
        1. Ashare（免费、无需Token、双核心：新浪+腾讯）
        2. EastMoney（需要cookie）
        3. AkShare（免费）
        4. TuShare（需要Token、有频率限制）
        5. BaoStock（免费）

        Args:
            symbols: List of symbols to retrieve
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe string ('1d' currently supported)

        Returns:
            Dictionary mapping symbols to their OHLCV dataframes
        """
        if not self._tushare_initialized:
            self.initialize()

        results = {}

        # Convert dates to string format
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        for symbol in symbols:
            try:
                df = pd.DataFrame()

                # 指数专用路径
                if _is_index_code(symbol):
                    # 1. AkShare index_zh_a_hist（最佳指数数据源）
                    if self.use_akshare and AKSHARE_AVAILABLE:
                        logger.debug(f"Fetching index {symbol} from AkShare index API")
                        df = self._fetch_index_from_akshare(symbol, start_str, end_str)
                        if not df.empty:
                            results[symbol] = df
                            continue

                    # 2. Ashare（修复后的 _normalize_code 支持指数）
                    logger.debug(f"Fetching index {symbol} from Ashare")
                    df = self._fetch_from_ashare(symbol, start_str, end_str)
                    if not df.empty:
                        results[symbol] = df
                        continue

                    # 3. TuShare index_daily
                    logger.debug(f"Falling back to TuShare index API for {symbol}")
                    df = self._fetch_index_from_tushare(symbol, start_date, end_date)
                    if not df.empty:
                        results[symbol] = df
                        continue

                    logger.warning(f"All index data sources failed for {symbol}")
                    continue

                # 个股路径
                # 1. 最高优先级：Ashare（免费、无需Token、双核心）
                logger.debug(f"Fetching {symbol} from Ashare (primary source)")
                df = self._fetch_from_ashare(symbol, start_str, end_str)
                if not df.empty:
                    results[symbol] = df
                    continue

                # 2. 备用：EastMoney
                if self.eastmoney_cookie:
                    logger.debug(f"Falling back to EastMoney for {symbol}")
                    df = self._fetch_from_eastmoney(symbol, start_str, end_str)
                    if not df.empty:
                        results[symbol] = df
                        continue

                # 3. 备用：AkShare
                if self.use_akshare:
                    logger.debug(f"Falling back to AkShare for {symbol}")
                    df = self._fetch_from_akshare(symbol, start_str, end_str)
                    if not df.empty:
                        results[symbol] = df
                        continue

                # 4. 备用：TuShare（有频率限制，可能超时）
                logger.debug(f"Falling back to TuShare for {symbol}")

                # Convert to Tushare format
                if '.' not in symbol:
                    if len(symbol) == 6:
                        if symbol.startswith(('5', '6', '9')):
                            tushare_symbol = f"{symbol}.SH"
                        else:
                            tushare_symbol = f"{symbol}.SZ"
                    else:
                        tushare_symbol = symbol
                else:
                    tushare_symbol = symbol

                start_ts = start_date.strftime("%Y%m%d")
                end_ts = end_date.strftime("%Y%m%d")

                # TuShare rate limiting: wait after every 50 requests
                if self._tushare_request_count >= 50:
                    elapsed = time.time() - self._tushare_last_reset
                    if elapsed < 60:
                        sleep_time = 60 - elapsed
                        logger.debug(f"TuShare rate limit reached ({self._tushare_request_count} requests), waiting {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                    self._tushare_request_count = 0
                    self._tushare_last_reset = time.time()

                if timeframe == "1d":
                    try:
                        df = self.pro_api.daily(
                            ts_code=tushare_symbol, start_date=start_ts, end_date=end_ts
                        )

                        if df.empty:
                            logger.warning(f"No data found from Tushare for symbol {tushare_symbol}")
                        else:
                            # Increment request counter on successful call
                            self._tushare_request_count += 1
                            # Rename columns to match expected format
                            df.rename(
                                columns={
                                    "ts_code": "symbol",
                                    "trade_date": "timestamp",
                                    "vol": "volume"
                                },
                                inplace=True,
                            )

                            # Convert timestamp to datetime
                            df["timestamp"] = pd.to_datetime(df["timestamp"])

                            # Add timeframe and ensure symbol column has the original symbol
                            df["timeframe"] = timeframe
                            df["symbol"] = symbol

                            # Reorder columns to match expected format
                            expected_cols = [
                                "timestamp", "open", "high", "low", "close", "volume",
                                "amount", "timeframe", "symbol"
                            ]
                            available_cols = [col for col in expected_cols if col in df.columns]
                            df = df[available_cols]

                            # Sort by timestamp
                            df.sort_values("timestamp", inplace=True)
                            df.reset_index(drop=True, inplace=True)
                    except Exception as e:
                        logger.warning(f"Tushare API error for {symbol}: {str(e)}. This may be due to limited API permissions.")
                        df = pd.DataFrame()

                if not df.empty:
                    results[symbol] = df
                    continue

                # 5. 最后备用：BaoStock
                if BAOSTOCK_AVAILABLE:
                    logger.debug(f"Last resort: falling back to BaoStock for {symbol}")
                    df = self._fetch_from_baostock(symbol, start_str, end_str)
                    if not df.empty:
                        results[symbol] = df
                        continue

            except Exception as e:
                logger.error(f"Failed to get data for symbol {symbol}: {str(e)}")
                continue

        return results

    def _fetch_single_symbol(
        self,
        symbol: str,
        start_str: str,
        end_str: str,
        timeframe: str = "1d"
    ) -> tuple:
        """
        Fetch data for a single symbol (used for parallel execution).

        Args:
            symbol: Stock symbol
            start_str: Start date string (YYYY-MM-DD)
            end_str: End date string (YYYY-MM-DD)
            timeframe: Data timeframe

        Returns:
            Tuple of (symbol, DataFrame or None)
        """
        try:
            # Check cache first
            if self._cache:
                cached = self._cache.get(symbol, start_str, end_str, timeframe)
                if cached is not None and not cached.empty:
                    return (symbol, cached)

            df = pd.DataFrame()

            # 1. 最高优先级：Ashare
            df = self._fetch_from_ashare(symbol, start_str, end_str)

            # 2. 备用：EastMoney
            if df.empty and self.eastmoney_cookie:
                df = self._fetch_from_eastmoney(symbol, start_str, end_str)

            # 3. 备用：AkShare
            if df.empty and self.use_akshare:
                df = self._fetch_from_akshare(symbol, start_str, end_str)

            # 4. 备用：TuShare（有频率限制）
            if df.empty:
                logger.debug(f"Falling back to TuShare for {symbol}")
                # Convert to Tushare format
                if '.' not in symbol:
                    if len(symbol) == 6:
                        if symbol.startswith(('5', '6', '9')):
                            tushare_symbol = f"{symbol}.SH"
                        else:
                            tushare_symbol = f"{symbol}.SZ"
                    else:
                        tushare_symbol = symbol
                else:
                    tushare_symbol = symbol

                start_ts = start_str.replace("-", "")
                end_ts = end_str.replace("-", "")

                # TuShare rate limiting: wait after every 50 requests
                if self._tushare_request_count >= 50:
                    elapsed = time.time() - self._tushare_last_reset
                    if elapsed < 60:
                        sleep_time = 60 - elapsed
                        logger.debug(f"TuShare rate limit reached ({self._tushare_request_count} requests), waiting {sleep_time:.1f} seconds...")
                        time.sleep(sleep_time)
                    self._tushare_request_count = 0
                    self._tushare_last_reset = time.time()

                if timeframe == "1d":
                    try:
                        tushare_df = self.pro_api.daily(
                            ts_code=tushare_symbol, start_date=start_ts, end_date=end_ts
                        )
                        if not tushare_df.empty:
                            # Increment request counter on successful call
                            self._tushare_request_count += 1
                            # Rename columns to match expected format
                            tushare_df.rename(
                                columns={
                                    "ts_code": "symbol",
                                    "trade_date": "timestamp",
                                    "vol": "volume"
                                },
                                inplace=True,
                            )
                            # Convert timestamp to datetime
                            tushare_df["timestamp"] = pd.to_datetime(tushare_df["timestamp"])
                            # Add timeframe and ensure symbol column has the original symbol
                            tushare_df["timeframe"] = timeframe
                            tushare_df["symbol"] = symbol
                            # Reorder columns to match expected format
                            expected_cols = [
                                "timestamp", "open", "high", "low", "close", "volume",
                                "amount", "timeframe", "symbol"
                            ]
                            available_cols = [col for col in expected_cols if col in tushare_df.columns]
                            df = tushare_df[available_cols]
                            # Sort by timestamp
                            df.sort_values("timestamp", inplace=True)
                            df.reset_index(drop=True, inplace=True)
                    except Exception as e:
                        logger.warning(f"Tushare API error for {symbol}: {str(e)}. This may be due to limited API permissions.")

            # 5. 最后备用：BaoStock
            if df.empty and BAOSTOCK_AVAILABLE:
                df = self._fetch_from_baostock(symbol, start_str, end_str)

            # Cache the result
            if not df.empty and self._cache:
                self._cache.set(symbol, start_str, end_str, df, timeframe)

            return (symbol, df if not df.empty else None)

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {str(e)}")
            return (symbol, None)

    def get_bars_parallel(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
        show_progress: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV bars for multiple symbols using parallel fetching.

        Uses ThreadPoolExecutor for concurrent data fetching, significantly
        faster than sequential fetching for multiple symbols.

        Args:
            symbols: List of symbols to retrieve
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe string ('1d' currently supported)
            show_progress: Whether to show progress logs

        Returns:
            Dictionary mapping symbols to their OHLCV dataframes
        """
        if not self._tushare_initialized:
            self.initialize()

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        results = {}
        futures = {}

        # Submit all tasks
        for symbol in symbols:
            future = self._executor.submit(
                self._fetch_single_symbol,
                symbol, start_str, end_str, timeframe
            )
            futures[future] = symbol

        # Collect results with progress tracking
        total = len(symbols)
        completed = 0
        pending_futures = set(futures.keys())

        try:
            # 使用 tqdm 进度条
            pbar = tqdm(total=total, desc="获取数据", unit="只", disable=not show_progress)

            for future in as_completed(futures, timeout=300):
                symbol = futures[future]
                pending_futures.discard(future)
                try:
                    result_symbol, df = future.result(timeout=60)
                    if df is not None and not df.empty:
                        results[result_symbol] = df
                    completed += 1
                    pbar.update(1)

                except Exception as e:
                    logger.debug(f"Failed to get result for {symbol}: {str(e)}")
                    completed += 1
                    pbar.update(1)

            pbar.close()

        except FuturesTimeoutError:
            # Handle timeout gracefully - log which symbols didn't complete
            unfinished_count = len(pending_futures)
            unfinished_symbols = [futures[f] for f in pending_futures]
            logger.warning(
                f"Timeout: {unfinished_count}/{total} symbols unfinished. "
                f"Symbols: {unfinished_symbols[:10]}{'...' if unfinished_count > 10 else ''}"
            )
            # Cancel pending futures
            for future in pending_futures:
                future.cancel()

        if show_progress:
            print(f"✅ 获取完成: {len(results)}/{total} 只股票")
        return results

    def get_bars_cached(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV bars with cache-first strategy.

        First checks cache, then fetches missing data in parallel,
        finally caches the results.

        Args:
            symbols: List of symbols to retrieve
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe string

        Returns:
            Dictionary mapping symbols to their OHLCV dataframes
        """
        if not self._tushare_initialized:
            self.initialize()

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        results = {}
        symbols_to_fetch = []

        # Check cache first
        if self._cache:
            for symbol in symbols:
                cached = self._cache.get(symbol, start_str, end_str, timeframe)
                if cached is not None and not cached.empty:
                    results[symbol] = cached
                else:
                    symbols_to_fetch.append(symbol)
            if results:
                print(f"📦 缓存命中: {len(results)}/{len(symbols)} 只股票")
        else:
            symbols_to_fetch = symbols

        # Fetch remaining symbols in parallel
        if symbols_to_fetch:
            fetched = self.get_bars_parallel(
                symbols_to_fetch, start_date, end_date, timeframe
            )
            results.update(fetched)

        return results

    def get_latest_bar(
        self, symbol: str, timeframe: str = "1d"
    ) -> Optional[pd.DataFrame]:
        """
        Get the most recent bar for a symbol.
        Prioritizes EastMoney data when available, falls back to Tushare, then AkShare.

        Args:
            symbol: Symbol to retrieve
            timeframe: Timeframe string (currently only supports '1d')

        Returns:
            DataFrame with the latest bar data, or None if unavailable
        """
        if not self._tushare_initialized:
            self.initialize()

        try:
            # Get data for the last week to ensure we get recent data
            end_date = datetime.now()
            start_date = end_date - pd.DateOffset(weeks=1)

            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            df = pd.DataFrame()

            # Try EastMoney first if available
            if self.eastmoney_cookie:
                logger.debug(f"Attempting to get latest bar for {symbol} from EastMoney")
                df = self._fetch_from_eastmoney(symbol, start_str, end_str)

            # Fallback to Tushare if EastMoney data not available or failed
            if df.empty:
                logger.debug(f"Falling back to Tushare for latest bar of {symbol}")

                # Convert to Tushare format
                if '.' not in symbol:
                    if len(symbol) == 6:
                        if symbol.startswith(('5', '6', '9')):
                            tushare_symbol = f"{symbol}.SH"
                        else:
                            tushare_symbol = f"{symbol}.SZ"
                    else:
                        tushare_symbol = symbol
                else:
                    tushare_symbol = symbol

                start_ts = start_date.strftime("%Y%m%d")
                end_ts = end_date.strftime("%Y%m%d")

                try:
                    df = self.pro_api.daily(
                        ts_code=tushare_symbol, start_date=start_ts, end_date=end_ts
                    )

                    if df.empty:
                        logger.warning(f"No recent data found from Tushare for {tushare_symbol}")
                except Exception as e:
                    logger.warning(f"Tushare API error for latest bar of {symbol}: {str(e)}. This may be due to limited API permissions.")
                    df = pd.DataFrame()

                if not df.empty:
                    # Rename columns to match expected format
                    df.rename(
                        columns={
                            "ts_code": "symbol",
                            "trade_date": "timestamp",
                            "vol": "volume"
                        },
                        inplace=True,
                    )

                    # Convert timestamp to datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"])

                    # Add timeframe and ensure symbol column
                    df["timeframe"] = timeframe
                    df["symbol"] = symbol

                    # Reorder columns to match expected format
                    expected_cols = [
                        "timestamp", "open", "high", "low", "close", "volume",
                        "amount", "timeframe", "symbol"
                    ]
                    available_cols = [col for col in expected_cols if col in df.columns]
                    df = df[available_cols]

                    # Sort by timestamp
                    df.sort_values("timestamp", inplace=True)
                    df.reset_index(drop=True, inplace=True)

            # Fallback to AkShare if Tushare failed
            if df.empty and self.use_akshare:
                logger.debug(f"Falling back to AkShare for latest bar of {symbol}")
                df = self._fetch_from_akshare(symbol, start_str, end_str)

            # Fallback to BaoStock if AkShare failed
            if df.empty and BAOSTOCK_AVAILABLE:
                logger.debug(f"Falling back to BaoStock for latest bar of {symbol}")
                df = self._fetch_from_baostock(symbol, start_str, end_str)

            if df.empty:
                logger.warning(f"No recent data found for {symbol}")
                return None

            # Get the most recent bar
            latest_bar = df.iloc[[-1]].copy()  # Use double brackets to keep as DataFrame

            return latest_bar

        except Exception as e:
            logger.error(f"Failed to get latest bar for symbol {symbol}: {str(e)}")
            return None

    def get_csi300_constituents(self, include_names: bool = False) -> List:
        """
        Get CSI 300 (沪深300) index constituents.
        Tries Tushare first, falls back to AkShare if available.

        Args:
            include_names: If True, returns list of dicts with 'code' and 'name' keys.
                          If False, returns list of stock codes only.

        Returns:
            List of stock codes in Tushare format (e.g., '000001.SZ', '600000.SH')
            or list of dicts with 'code' and 'name' if include_names=True
        """
        if not self._tushare_initialized:
            self.initialize()

        # Try Tushare first
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")

            df = self.pro_api.index_weight(
                index_code='000300.SH',
                start_date=start_str,
                end_date=end_str
            )

            if not df.empty:
                latest_date = df['trade_date'].max()
                latest_df = df[df['trade_date'] == latest_date]
                constituents = latest_df['con_code'].tolist()

                if include_names:
                    # Fetch stock names from Tushare stock_basic
                    try:
                        stock_basic = self.pro_api.stock_basic(
                            exchange="",
                            list_status="L",
                            fields="ts_code,name"
                        )
                        # Create a mapping from code to name
                        name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
                        constituents = [
                            {"code": code, "name": name_map.get(code, code)}
                            for code in constituents
                        ]
                    except Exception as e:
                        logger.warning(f"Failed to fetch stock names from Tushare: {str(e)}")
                        constituents = [{"code": code, "name": code} for code in constituents]

                logger.debug(f"Got from Tushare")
                return constituents
            else:
                logger.warning("Tushare returned empty CSI 300 constituents list")
        except Exception as e:
            logger.warning(f"Failed to get CSI 300 constituents from Tushare: {str(e)}")

        # Fallback to AkShare
        if self.use_akshare and AKSHARE_AVAILABLE:
            try:
                logger.debug("Trying to get CSI 300 constituents from AkShare...")
                # ak.index_stock_cons_weight_csindex returns df with columns like: 成分券代码, 成分券名称, etc.
                df = ak.index_stock_cons_weight_csindex(symbol="000300")
                if not df.empty:
                    # Convert to Tushare format (add .SH or .SZ suffix)
                    constituents = []
                    for _, row in df.iterrows():
                        code_str = str(row['成分券代码']).zfill(6)
                        name = row.get('成分券名称', code_str)
                        if code_str.startswith(('5', '6', '9')):
                            code_with_suffix = f"{code_str}.SH"
                        else:
                            code_with_suffix = f"{code_str}.SZ"

                        if include_names:
                            constituents.append({"code": code_with_suffix, "name": name})
                        else:
                            constituents.append(code_with_suffix)

                    logger.debug(f"Got from AkShare")
                    return constituents
                else:
                    logger.warning("AkShare returned empty CSI 300 constituents list")
            except Exception as e:
                logger.error(f"Failed to get CSI 300 constituents from AkShare: {str(e)}")
        else:
            logger.warning("AkShare not available, cannot fallback for CSI 300 constituents")

        return []

    def get_csi1000_constituents(self, include_names: bool = False) -> List:
        """
        Get CSI 1000 (中证1000) index constituents.
        Tries Tushare first, falls back to AkShare if available.

        Args:
            include_names: If True, returns list of dicts with 'code' and 'name' keys.
                          If False, returns list of stock codes only.

        Returns:
            List of stock codes in Tushare format (e.g., '000001.SZ', '600000.SH')
            or list of dicts with 'code' and 'name' if include_names=True
        """
        if not self._tushare_initialized:
            self.initialize()

        # Try Tushare first
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            start_str = start_date.strftime("%Y%m%d")
            end_str = end_date.strftime("%Y%m%d")

            df = self.pro_api.index_weight(
                index_code='000852.SH',
                start_date=start_str,
                end_date=end_str
            )

            if not df.empty:
                latest_date = df['trade_date'].max()
                latest_df = df[df['trade_date'] == latest_date]
                constituents = latest_df['con_code'].tolist()

                if include_names:
                    # Fetch stock names from Tushare stock_basic
                    try:
                        stock_basic = self.pro_api.stock_basic(
                            exchange="",
                            list_status="L",
                            fields="ts_code,name"
                        )
                        # Create a mapping from code to name
                        name_map = dict(zip(stock_basic['ts_code'], stock_basic['name']))
                        constituents = [
                            {"code": code, "name": name_map.get(code, code)}
                            for code in constituents
                        ]
                    except Exception as e:
                        logger.warning(f"Failed to fetch stock names from Tushare: {str(e)}")
                        constituents = [{"code": code, "name": code} for code in constituents]

                logger.debug(f"Got from Tushare")
                return constituents
            else:
                logger.warning("Tushare returned empty CSI 1000 constituents list")
        except Exception as e:
            logger.warning(f"Failed to get CSI 1000 constituents from Tushare: {str(e)}")

        # Fallback to AkShare
        if self.use_akshare and AKSHARE_AVAILABLE:
            try:
                logger.debug("Trying to get CSI 1000 constituents from AkShare...")
                # ak.index_stock_cons_weight_csindex returns df with columns like: 成分券代码, 成分券名称, etc.
                df = ak.index_stock_cons_weight_csindex(symbol="000852")
                if not df.empty:
                    # Convert to Tushare format (add .SH or .SZ suffix)
                    constituents = []
                    for _, row in df.iterrows():
                        code_str = str(row['成分券代码']).zfill(6)
                        name = row.get('成分券名称', code_str)
                        if code_str.startswith(('5', '6', '9')):
                            code_with_suffix = f"{code_str}.SH"
                        else:
                            code_with_suffix = f"{code_str}.SZ"

                        if include_names:
                            constituents.append({"code": code_with_suffix, "name": name})
                        else:
                            constituents.append(code_with_suffix)

                    logger.debug(f"Got from AkShare")
                    return constituents
                else:
                    logger.warning("AkShare returned empty CSI 1000 constituents list")
            except Exception as e:
                logger.error(f"Failed to get CSI 1000 constituents from AkShare: {str(e)}")
        else:
            logger.warning("AkShare not available, cannot fallback for CSI 1000 constituents")

        return []

    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        if not self._tushare_initialized:
            self.initialize()

        try:
            # Get stock basic information from Tushare
            stock_list = self.pro_api.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date",
            )
            return stock_list["ts_code"].tolist()
        except Exception as e:
            logger.error(f"Failed to get supported symbols: {str(e)}")
            return []

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for symbols matching the query.

        Args:
            query: Search query string

        Returns:
            List of matching symbols with metadata
        """
        if not self._tushare_initialized:
            self.initialize()

        try:
            # Get stock basic information from Tushare
            stock_list = self.pro_api.stock_basic(
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,list_date",
            )

            # Filter based on query (case-insensitive)
            matching_stocks = stock_list[
                (stock_list["name"].str.contains(query, case=False, na=False))
                | (stock_list["ts_code"].str.contains(query, case=False, na=False))
                | (stock_list["symbol"].str.contains(query, case=False, na=False))
            ]

            results = []
            for _, row in matching_stocks.iterrows():
                results.append(
                    {
                        "symbol": row["ts_code"],
                        "name": row["name"],
                        "area": row["area"],
                        "industry": row["industry"],
                        "list_date": row["list_date"],
                    }
                )

            return results

        except Exception as e:
            logger.error(f"Failed to search symbols: {str(e)}")
            return []

    def get_calendar(self) -> List[datetime]:
        """
        Get trading calendar (list of trading days).

        Returns:
            List of trading days
        """
        if not self._tushare_initialized:
            self.initialize()

        try:
            # Get trading calendar for the past year from Tushare
            today = datetime.now()
            last_year = today - timedelta(days=365)

            start_str = last_year.strftime("%Y%m%d")
            end_str = today.strftime("%Y%m%d")

            cal_df = self.pro_api.trade_cal(
                exchange="", start_date=start_str, end_date=end_str
            )

            # Filter for open days
            open_days = cal_df[cal_df["is_open"] == 1]["cal_date"]
            return [datetime.strptime(date, "%Y%m%d") for date in open_days.tolist()]

        except Exception as e:
            logger.error(f"Failed to get trading calendar: {str(e)}")
            return []

    # ==================== 基本面数据获取 ====================

    def get_fundamental_data(
        self,
        symbol: str,
        include_valuation: bool = True,
        include_profit: bool = True,
        include_growth: bool = True
    ) -> Dict[str, Any]:
        """
        获取股票基本面数据（PE/PB/ROE等）

        使用 BaoStock 作为数据源（免费、无需权限）

        Args:
            symbol: 股票代码，如 '000001.SZ' 或 '000001'
            include_valuation: 是否包含估值指标（PE/PB）
            include_profit: 是否包含盈利能力指标（ROE/净利润率）
            include_growth: 是否包含成长能力指标

        Returns:
            Dict: 基本面数据字典
        """
        result = {
            'symbol': symbol,
            'pe': None,
            'pb': None,
            'roe': None,
            'profit_margin': None,
            'eps': None,
            'yoy_profit': None,
            'data_source': None,
            'error': None
        }

        # 标准化股票代码格式 (BaoStock 使用 sz.000001 或 sh.600000 格式)
        bs_code = self._normalize_symbol_for_baostock(symbol)
        if not bs_code:
            result['error'] = '无效的股票代码格式'
            return result

        # 尝试从 BaoStock 获取数据
        if BAOSTOCK_AVAILABLE:
            try:
                data = self._fetch_fundamental_from_baostock(
                    bs_code,
                    include_valuation,
                    include_profit,
                    include_growth
                )
                if data:
                    result.update(data)
                    result['data_source'] = 'baostock'
                    return result
            except Exception as e:
                logger.warning(f"BaoStock 获取基本面数据失败: {str(e)}")

        # 尝试从 AkShare 获取数据
        if AKSHARE_AVAILABLE:
            try:
                data = self._fetch_fundamental_from_akshare(symbol)
                if data:
                    result.update(data)
                    result['data_source'] = 'akshare'
                    return result
            except Exception as e:
                logger.warning(f"AkShare 获取基本面数据失败: {str(e)}")

        result['error'] = '无法获取基本面数据'
        return result

    def _normalize_symbol_for_baostock(self, symbol: str) -> Optional[str]:
        """
        将股票代码转换为 BaoStock 格式

        Args:
            symbol: 股票代码，如 '000001.SZ', 'sz000001', '000001'

        Returns:
            BaoStock 格式的股票代码，如 'sz.000001'
        """
        if not symbol:
            return None

        # 移除空格
        symbol = symbol.strip().upper()

        # 提取纯数字代码
        import re
        match = re.search(r'(\d{6})', symbol)
        if not match:
            return None

        pure_code = match.group(1)

        # 判断市场
        if pure_code.startswith('6'):
            return f'sh.{pure_code}'
        elif pure_code.startswith(('0', '3')):
            return f'sz.{pure_code}'
        else:
            return f'sz.{pure_code}'  # 默认深圳

    def _fetch_fundamental_from_baostock(
        self,
        bs_code: str,
        include_valuation: bool,
        include_profit: bool,
        include_growth: bool
    ) -> Optional[Dict[str, Any]]:
        """
        从 BaoStock 获取基本面数据

        Args:
            bs_code: BaoStock 格式的股票代码
            include_valuation: 是否包含估值指标
            include_profit: 是否包含盈利能力指标
            include_growth: 是否包含成长能力指标

        Returns:
            基本面数据字典
        """
        import baostock as bs

        result = {}

        # 登录 BaoStock
        lg = bs.login()
        if lg.error_code != '0':
            logger.error(f"BaoStock 登录失败: {lg.error_msg}")
            return None

        try:
            # 1. 获取估值指标（PE/PB）- 从K线数据中获取
            if include_valuation:
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')

                rs = bs.query_history_k_data_plus(
                    bs_code,
                    'date,close,peTTM,pbMRQ,psTTM',
                    start_date=start_date,
                    end_date=end_date,
                    frequency='d',
                    adjustflag='3'
                )

                valuation_data = []
                while (rs.error_code == '0') & rs.next():
                    valuation_data.append(rs.get_row_data())

                if valuation_data:
                    # 取最新一条
                    latest = valuation_data[-1]
                    if len(latest) >= 5:
                        result['close'] = float(latest[1]) if latest[1] else None
                        result['pe'] = float(latest[2]) if latest[2] else None
                        result['pb'] = float(latest[3]) if latest[3] else None
                        result['ps'] = float(latest[4]) if latest[4] else None

            # 2. 获取盈利能力指标（ROE/净利润率）
            if include_profit:
                current_year = datetime.now().year
                current_quarter = (datetime.now().month - 1) // 3 + 1

                # 尝试获取最近报告期数据
                for quarter in [current_quarter, current_quarter - 1, 4, 3, 2, 1]:
                    if quarter <= 0:
                        continue
                    year = current_year if quarter <= current_quarter else current_year - 1
                    if quarter <= 0:
                        year -= 1
                        quarter += 4

                    rs = bs.query_profit_data(
                        code=bs_code,
                        year=year,
                        quarter=quarter
                    )

                    profit_data = []
                    while (rs.error_code == '0') & rs.next():
                        profit_data.append(rs.get_row_data())

                    if profit_data:
                        latest = profit_data[0]
                        # 字段: code, pubDate, statDate, roeAvg, npMargin, gpMargin, netProfit, epsTTM, MBRevenue, totalShare, liqaShare
                        if len(latest) >= 8:
                            result['roe'] = float(latest[3]) * 100 if latest[3] else None  # ROE 转为百分比
                            result['profit_margin'] = float(latest[4]) * 100 if latest[4] else None  # 净利润率
                            result['eps'] = float(latest[7]) if latest[7] else None  # 每股收益TTM
                            result['report_date'] = latest[2]  # 报告期
                        break

            # 3. 获取成长能力指标
            if include_growth:
                current_year = datetime.now().year
                current_quarter = (datetime.now().month - 1) // 3 + 1

                for quarter in [current_quarter, current_quarter - 1, 4, 3, 2, 1]:
                    if quarter <= 0:
                        continue
                    year = current_year if quarter <= current_quarter else current_year - 1
                    if quarter <= 0:
                        year -= 1
                        quarter += 4

                    rs = bs.query_growth_data(
                        code=bs_code,
                        year=year,
                        quarter=quarter
                    )

                    growth_data = []
                    while (rs.error_code == '0') & rs.next():
                        growth_data.append(rs.get_row_data())

                    if growth_data:
                        latest = growth_data[0]
                        # 字段: code, pubDate, statDate, YOYEquity, YOYAsset, YOYNI, YOYEPSBasic, YOYPNI
                        if len(latest) >= 6:
                            result['yoy_profit'] = float(latest[5]) * 100 if latest[5] else None  # 净利润同比增长率
                        break

            return result if result else None

        finally:
            bs.logout()

    def _fetch_fundamental_from_akshare(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        从 AkShare 获取基本面数据（备用方案）

        Args:
            symbol: 股票代码

        Returns:
            基本面数据字典
        """
        import akshare as ak

        result = {}

        # 提取纯数字代码
        import re
        match = re.search(r'(\d{6})', symbol)
        if not match:
            return None
        pure_code = match.group(1)

        try:
            # 获取个股信息
            info_df = ak.stock_individual_info_em(symbol=pure_code)
            if info_df is not None and not info_df.empty:
                info_dict = dict(zip(info_df['item'], info_df['value']))
                result['total_mv'] = info_dict.get('总市值')
                result['float_mv'] = info_dict.get('流通市值')
                result['industry'] = info_dict.get('行业')
        except Exception:
            pass

        return result if result else None

    def get_stock_basic_info(self, symbol: str) -> Dict[str, Any]:
        """
        获取股票基本信息（行业、上市日期等）

        Args:
            symbol: 股票代码

        Returns:
            基本信息字典
        """
        result = {
            'symbol': symbol,
            'name': None,
            'industry': None,
            'list_date': None,
            'area': None
        }

        # 标准化代码
        bs_code = self._normalize_symbol_for_baostock(symbol)
        if not bs_code:
            return result

        if BAOSTOCK_AVAILABLE:
            import baostock as bs
            lg = bs.login()
            if lg.error_code == '0':
                try:
                    # 查询股票基本信息
                    rs = bs.query_stock_basic()

                    while (rs.error_code == '0') & rs.next():
                        row = rs.get_row_data()
                        if row[0] == bs_code:
                            result['name'] = row[1]
                            result['list_date'] = row[2]
                            result['status'] = row[5]
                            break
                finally:
                    bs.logout()

        return result

    # ==========================================
    # 实时行情接口（秒级更新）
    # ==========================================

    @property
    def realtime_provider(self):
        """获取实时数据提供者（延迟初始化）"""
        if not hasattr(self, '_realtime_provider') or self._realtime_provider is None:
            from ..realtime.realtime_provider import get_realtime_provider
            self._realtime_provider = get_realtime_provider()
        return self._realtime_provider

    @property
    def minute_provider(self):
        """获取分钟数据提供者（延迟初始化）"""
        if not hasattr(self, '_minute_provider') or self._minute_provider is None:
            from .incremental_minute_provider import get_incremental_minute_provider
            self._minute_provider = get_incremental_minute_provider()
        return self._minute_provider

    def get_realtime_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        获取单只股票实时行情

        使用 Pytdx（通达信）或 Sina（新浪）数据源，
        支持秒级更新。

        Args:
            symbol: 股票代码

        Returns:
            实时行情字典，包含 price, open, high, low, volume 等
        """
        try:
            quote = self.realtime_provider.get_realtime_quote(symbol)
            if quote:
                return quote.to_dict()
        except Exception as e:
            logger.warning(f"获取实时行情失败 {symbol}: {e}")
        return None

    def get_realtime_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        批量获取实时行情（推荐使用）

        一次请求获取多只股票，效率更高。

        Args:
            symbols: 股票代码列表

        Returns:
            {symbol: quote_dict} 字典
        """
        try:
            quotes = self.realtime_provider.get_realtime_quotes(symbols)
            return {symbol: quote.to_dict() for symbol, quote in quotes.items()}
        except Exception as e:
            logger.warning(f"批量获取实时行情失败: {e}")
            return {}

    def get_minute_bars(
        self,
        symbol: str,
        period: str = '5m',
        count: int = 100,
        start_time: datetime = None,
        end_time: datetime = None
    ) -> pd.DataFrame:
        """
        获取分钟K线数据（支持增量更新）

        Args:
            symbol: 股票代码
            period: 周期 (1m, 5m, 15m, 30m, 60m)
            count: 获取数量
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            DataFrame，包含 open/high/low/close/volume/amount 列
        """
        try:
            return self.minute_provider.get_minute_bars(
                symbol, period, start_time, end_time, count
            )
        except Exception as e:
            logger.warning(f"获取分钟数据失败 {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_bars(
        self,
        symbol: str,
        period: str = '5m',
        count: int = 60
    ) -> pd.DataFrame:
        """
        获取最近N根分钟K线

        Args:
            symbol: 股票代码
            period: 周期
            count: 数量

        Returns:
            DataFrame
        """
        return self.get_minute_bars(symbol, period, count=count)

    def get_realtime_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        quote = self.get_realtime_quote(symbol)
        return quote.get('price') if quote else None

    def get_realtime_prices(self, symbols: List[str]) -> Dict[str, float]:
        """批量获取最新价格"""
        quotes = self.get_realtime_quotes(symbols)
        return {
            symbol: quote.get('price')
            for symbol, quote in quotes.items()
            if quote and quote.get('price') is not None
        }


def create_data_fetcher_with_credentials() -> EnhancedDataFetcher:
    """Create a fetcher using optional credentials from the environment."""
    return EnhancedDataFetcher(
        tushare_token=os.getenv("TUSHARE_TOKEN"),
        eastmoney_cookie=os.getenv("EASTMONEY_COOKIE"),
    )
