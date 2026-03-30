"""
AkShare 分钟数据提供者

提供免费的A股分钟级实时数据获取能力
- 支持分钟K线数据
- 支持实时行情
- 内置缓存机制
- 集成反爬虫防护
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import time

from ...domain.interfaces.data_provider import IDataProvider
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger

# 导入反爬虫防护模块
from .anti_crawler import DelayController, get_eastmoney_headers

logger = get_logger(__name__)

# Try to import AkShare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.warning("AkShare not installed. Install with: pip install akshare")


def convert_symbol_to_akshare(symbol: str) -> str:
    """
    将股票代码转换为AkShare格式

    Args:
        symbol: 原始股票代码，如 "600519.SH" 或 "600519"

    Returns:
        AkShare格式的股票代码，如 "600519"
    """
    if '.' in symbol:
        return symbol.split('.')[0]
    return symbol


@registry.register(ComponentType.DATA_PROVIDER, "akshare_minute")
class AkShareMinuteProvider(IDataProvider):
    """
    AkShare分钟数据提供者（集成反爬虫防护）

    特点:
    - 免费: 无需API Token
    - 实时: 支持分钟级数据
    - 缓存: 内置60秒缓存避免频繁请求
    - 防护: 延迟控制防止被封
    """

    # 支持的时间框架
    SUPPORTED_TIMEFRAMES = ['1m', '5m', '15m', '30m', '60m']

    # AkShare period 映射
    PERIOD_MAP = {
        '1m': '1',
        '5m': '5',
        '15m': '15',
        '30m': '30',
        '60m': '60'
    }

    def __init__(self, cache_ttl_seconds: int = 60, min_delay: float = 0.1, max_delay: float = 0.2):
        """
        初始化AkShare分钟数据提供者

        Args:
            cache_ttl_seconds: 缓存过期时间(秒)
            min_delay: 最小请求延迟(秒)
            max_delay: 最大请求延迟(秒)
        """
        if not AKSHARE_AVAILABLE:
            raise RuntimeError("AkShare is not installed. Install with: pip install akshare")

        self._initialized = False
        self._cache_ttl = cache_ttl_seconds
        self._cache: Dict[str, tuple] = {}  # {cache_key: (data, timestamp)}

        # 延迟控制器
        self._delay_controller = DelayController(min_delay=min_delay, max_delay=max_delay)

        # 股票列表缓存 (用于快速搜索)
        # 缓存 1 小时，因为股票列表变化不频繁
        self._stock_list_cache: Optional[pd.DataFrame] = None
        self._stock_list_timestamp: float = 0
        self._stock_list_ttl: int = 3600  # 1 小时

    def initialize(self) -> None:
        """初始化数据提供者"""
        if self._initialized:
            return

        try:
            # 测试AkShare是否可用
            # 尝试获取一只股票的最新数据来验证连接
            test_df = ak.stock_zh_a_spot_em()
            if test_df is not None and len(test_df) > 0:
                self._initialized = True
                logger.info("AkShareMinuteProvider initialized successfully")
            else:
                raise RuntimeError("AkShare test query returned empty data")
        except Exception as e:
            logger.warning(f"AkShare initialization failed: {e}, using fallback mode")
            # 即使 AkShare 不可用，也允许初始化（使用本地股票列表）
            self._initialized = True
            logger.info("AkShareMinuteProvider initialized in fallback mode")

    def _get_cached(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return data
        return None

    def _convert_to_native(self, obj: Any) -> Any:
        """将 numpy 类型转换为 Python 原生类型"""
        if obj is None:
            return None
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_native(item) for item in obj]
        return obj

    def _set_cache(self, key: str, data: Any) -> None:
        """设置缓存（自动转换 numpy 类型）"""
        converted_data = self._convert_to_native(data)
        self._cache[key] = (converted_data, time.time())

    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        获取历史K线数据

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间框架 ('1m', '5m', '15m', '30m', '60m', '1d')

        Returns:
            {symbol: DataFrame} 字典
        """
        if not self._initialized:
            self.initialize()

        results = {}

        for symbol in symbols:
            try:
                df = self._fetch_bars(symbol, start_date, end_date, timeframe)
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"Failed to get bars for {symbol}: {e}")

        return results

    def _fetch_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """从AkShare获取K线数据"""
        base_symbol = convert_symbol_to_akshare(symbol)

        if timeframe == '1d':
            # 日线数据
            return self._fetch_daily_bars(base_symbol, start_date, end_date)
        elif timeframe in self.SUPPORTED_TIMEFRAMES:
            # 分钟数据
            return self._fetch_minute_bars(base_symbol, timeframe)
        else:
            logger.warning(f"Unsupported timeframe: {timeframe}")
            return pd.DataFrame()

    def _fetch_daily_bars(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """获取日线数据（带反爬虫防护）"""
        cache_key = f"daily_{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # 延迟控制
            self._delay_controller.wait()

            start_str = start_date.strftime('%Y%m%d')
            end_str = end_date.strftime('%Y%m%d')

            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period="daily",
                start_date=start_str,
                end_date=end_str,
                adjust="qfq"  # 前复权
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 重命名列
            df = df.rename(columns={
                '日期': 'timestamp',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount',
                '振幅': 'amplitude',
                '涨跌幅': 'pct_change',
                '涨跌额': 'change',
                '换手率': 'turnover'
            })

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['symbol'] = symbol
            df['timeframe'] = '1d'

            # 排序
            df = df.sort_values('timestamp').reset_index(drop=True)

            self._set_cache(cache_key, df)
            return df

        except Exception as e:
            logger.error(f"Error fetching daily bars for {symbol}: {e}")
            return pd.DataFrame()

    def _fetch_minute_bars(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """获取分钟K线数据（带反爬虫防护）"""
        cache_key = f"minute_{symbol}_{timeframe}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # 延迟控制
            self._delay_controller.wait()

            period = self.PERIOD_MAP.get(timeframe, '5')

            # 使用AkShare获取分钟数据
            df = ak.stock_zh_a_hist_min_em(
                symbol=symbol,
                period=period,
                adjust="qfq"
            )

            if df is None or df.empty:
                return pd.DataFrame()

            # 重命名列
            df = df.rename(columns={
                '时间': 'timestamp',
                '开盘': 'open',
                '最高': 'high',
                '最低': 'low',
                '收盘': 'close',
                '成交量': 'volume',
                '成交额': 'amount'
            })

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['symbol'] = symbol
            df['timeframe'] = timeframe

            # 排序
            df = df.sort_values('timestamp').reset_index(drop=True)

            self._set_cache(cache_key, df)
            return df

        except Exception as e:
            logger.error(f"Error fetching minute bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_bar(
        self, symbol: str, timeframe: str = "5m"
    ) -> Optional[pd.DataFrame]:
        """
        获取最新一根K线

        Args:
            symbol: 股票代码
            timeframe: 时间框架

        Returns:
            包含最新K线的DataFrame
        """
        if not self._initialized:
            self.initialize()

        try:
            base_symbol = convert_symbol_to_akshare(symbol)

            if timeframe == '1d':
                # 日线：获取最近数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)
                df = self._fetch_daily_bars(base_symbol, start_date, end_date)
            else:
                # 分钟线
                df = self._fetch_minute_bars(base_symbol, timeframe)

            if df.empty:
                return None

            # 返回最后一行
            return df.iloc[[-1]].copy()

        except Exception as e:
            logger.error(f"Failed to get latest bar for {symbol}: {e}")
            return None

    def get_latest_bars(
        self,
        symbol: str,
        count: int = 60,
        timeframe: str = "5m"
    ) -> pd.DataFrame:
        """
        获取最近N根K线

        Args:
            symbol: 股票代码
            count: K线数量
            timeframe: 时间框架

        Returns:
            DataFrame
        """
        if not self._initialized:
            self.initialize()

        try:
            base_symbol = convert_symbol_to_akshare(symbol)

            if timeframe == '1d':
                # 日线：获取足够的历史数据
                end_date = datetime.now()
                start_date = end_date - timedelta(days=count * 2)  # 多获取一些
                df = self._fetch_daily_bars(base_symbol, start_date, end_date)
            else:
                # 分钟线
                df = self._fetch_minute_bars(base_symbol, timeframe)

            if df.empty:
                return pd.DataFrame()

            # 返回最后N根
            return df.tail(count).reset_index(drop=True)

        except Exception as e:
            logger.error(f"Failed to get latest bars for {symbol}: {e}")
            return pd.DataFrame()

    def get_realtime_quote(self, symbol: str) -> Dict[str, Any]:
        """
        获取实时行情（带反爬虫防护）

        Args:
            symbol: 股票代码

        Returns:
            实时行情字典
        """
        if not self._initialized:
            self.initialize()

        cache_key = f"quote_{symbol}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            # 延迟控制
            self._delay_controller.wait()

            base_symbol = convert_symbol_to_akshare(symbol)

            # 获取实时行情
            df = ak.stock_zh_a_spot_em()

            if df is None or df.empty:
                return {}

            # 查找目标股票
            mask = df['代码'] == base_symbol
            if not mask.any():
                logger.warning(f"Symbol {symbol} not found in realtime data")
                return {}

            row = df[mask].iloc[0]

            quote = {
                'symbol': symbol,
                'name': str(row.get('名称', '')),
                'price': float(row.get('最新价', 0) or 0),
                'open': float(row.get('今开', 0) or 0),
                'high': float(row.get('最高', 0) or 0),
                'low': float(row.get('最低', 0) or 0),
                'volume': float(row.get('成交量', 0) or 0),
                'amount': float(row.get('成交额', 0) or 0),
                'pct_change': float(row.get('涨跌幅', 0) or 0),
                'change': float(row.get('涨跌额', 0) or 0),
                'turnover': float(row.get('换手率', 0) or 0),
                'timestamp': datetime.now()
            }

            self._set_cache(cache_key, quote)
            return quote

        except Exception as e:
            logger.error(f"Failed to get realtime quote for {symbol}: {e}")
            return {}

    def get_supported_symbols(self) -> List[str]:
        """获取支持的股票列表（带反爬虫防护）"""
        if not self._initialized:
            self.initialize()

        try:
            # 延迟控制
            self._delay_controller.wait()

            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                return df['代码'].tolist()
        except Exception as e:
            logger.error(f"Failed to get supported symbols: {e}")

        return []

    def _get_stock_list(self) -> Optional[pd.DataFrame]:
        """获取股票列表（带缓存和本地回退）"""
        now = time.time()

        # 如果缓存有效，直接返回
        if self._stock_list_cache is not None and (now - self._stock_list_timestamp) < self._stock_list_ttl:
            logger.debug("Using cached stock list for search")
            return self._stock_list_cache

        # 缓存过期或不存在，重新获取
        try:
            # 延迟控制
            self._delay_controller.wait()

            logger.info("Fetching fresh stock list from AkShare...")
            df = ak.stock_zh_a_spot_em()
            if df is not None and not df.empty:
                self._stock_list_cache = df
                self._stock_list_timestamp = now
                logger.info(f"Cached {len(df)} stocks for search")
                return df
        except Exception as e:
            logger.warning(f"Failed to fetch stock list: {e}")

        # 如果有旧缓存，即使过期也继续使用
        if self._stock_list_cache is not None:
            logger.warning("Using stale stock list cache due to fetch error")
            return self._stock_list_cache

        # 最后回退：使用本地预设股票列表
        logger.warning("Using fallback local stock list")
        return self._get_fallback_stock_list()

    def _get_fallback_stock_list(self) -> pd.DataFrame:
        """获取本地预设股票列表（当网络不可用时）"""
        # 常用股票列表 - 包含沪深主板、创业板、科创板
        stocks = [
            # 沪深主板
            ('600519', '贵州茅台', 1800.0),
            ('000858', '五粮液', 130.0),
            ('000001', '平安银行', 10.0),
            ('000002', '万科A', 7.0),
            ('600036', '招商银行', 30.0),
            ('601318', '中国平安', 45.0),
            ('600000', '浦发银行', 8.0),
            ('601166', '兴业银行', 15.0),
            ('600030', '中信证券', 18.0),
            ('600276', '恒瑞医药', 40.0),
            ('000333', '美的集团', 55.0),
            ('000651', '格力电器', 35.0),
            ('002415', '海康威视', 30.0),
            ('300750', '宁德时代', 180.0),
            ('601012', '隆基绿能', 25.0),
            ('002594', '比亚迪', 250.0),
            ('600900', '长江电力', 25.0),
            ('601899', '紫金矿业', 15.0),
            ('600887', '伊利股份', 28.0),
            ('000568', '泸州老窖', 180.0),
            ('002304', '洋河股份', 100.0),
            ('600309', '万华化学', 90.0),
            ('601398', '工商银行', 5.0),
            ('601288', '农业银行', 3.5),
            ('600016', '民生银行', 4.0),
            ('601988', '中国银行', 4.5),
            ('601328', '交通银行', 5.5),
            ('600048', '保利发展', 12.0),
            ('001979', '招商蛇口', 15.0),
            ('002352', '顺丰控股', 40.0),
            # 创业板
            ('300059', '东方财富', 25.0),
            ('300015', '爱尔眼科', 30.0),
            ('300033', '同花顺', 150.0),
            ('300122', '智飞生物', 80.0),
            ('300142', '沃森生物', 40.0),
            ('300454', '网宿科技', 15.0),
            ('300498', '温氏股份', 40.0),
            ('300666', '江丰电子', 60.0),
            ('300676', '华大基因', 80.0),
            ('300750', '宁德时代', 180.0),
            # 科创板
            ('688276', '联影医疗', 150.0),
            ('688041', '华润微', 50.0),
            ('688126', '沪硅产业', 25.0),
            ('688169', '石头科技', 200.0),
            ('688185', '康希诺', 100.0),
            ('688317', '科华生物', 20.0),
            ('688356', '江苏北人', 25.0),
            ('688369', '航发动力', 50.0),
            ('688408', '华兴源创', 40.0),
            ('688466', '金城医药', 35.0),
            ('688521', '芯原股份', 60.0),
            ('688536', '思特威', 80.0),
            ('688561', '昱能科技', 100.0),
            ('688599', '天合光能', 50.0),
            ('688636', '纳芯微', 120.0),
            ('688981', '中芯国际', 45.0),
            ('688126', '沪硅产业', 25.0),
            ('688202', '当虹科技', 60.0),
            ('688223', '晶科能源', 35.0),
            ('688235', '百济神州', 150.0),
            ('688256', '寒武纪', 200.0),
            ('688317', '科华生物', 20.0),
            ('688339', '亿华通', 80.0),
            ('688345', '博腾股份', 40.0),
            ('688356', '江苏北人', 25.0),
            ('688369', '航发动力', 50.0),
            ('688393', '安芯电子', 60.0),
            ('688408', '华兴源创', 40.0),
            ('688439', '振华风光', 50.0),
            ('688466', '金城医药', 35.0),
            ('688499', '利元亨', 40.0),
            ('688521', '芯原股份', 60.0),
            ('688536', '思特威', 80.0),
            ('688561', '昱能科技', 100.0),
            ('688569', '铁岭新城', 10.0),
            ('688578', '艾为电子', 70.0),
            ('688599', '天合光能', 50.0),
            ('688618', '英威腾', 15.0),
            ('688636', '纳芯微', 120.0),
            ('688639', '华润微', 50.0),
            ('688650', '安凯微', 25.0),
            ('688659', '元道通信', 30.0),
            ('688661', '博菲电气', 35.0),
            ('688676', '金盘科技', 40.0),
            ('688700', '格科微', 30.0),
            ('688728', '格科微', 30.0),
            ('688767', '博腾股份', 40.0),
            ('688777', '海康威视', 30.0),
            ('688981', '中芯国际', 45.0),
            ('688993', '华润微', 50.0),
        ]
        return pd.DataFrame(stocks, columns=['代码', '名称', '最新价'])

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """搜索股票（使用缓存的股票列表）"""
        if not self._initialized:
            self.initialize()

        try:
            # 使用缓存的股票列表
            df = self._get_stock_list()
            if df is None or df.empty:
                return []

            # 搜索代码或名称
            mask = df['代码'].str.contains(query, na=False, case=False) | \
                   df['名称'].str.contains(query, na=False, case=False)

            results = []
            for _, row in df[mask].head(20).iterrows():
                results.append({
                    'symbol': str(row['代码']),
                    'name': str(row['名称']),
                    'price': float(row.get('最新价', 0) or 0)
                })

            return results

        except Exception as e:
            logger.error(f"Failed to search symbols: {e}")
            return []

    def get_calendar(self) -> List[datetime]:
        """获取交易日历（带反爬虫防护）"""
        if not self._initialized:
            self.initialize()

        try:
            # 延迟控制
            self._delay_controller.wait()

            # 使用AkShare获取交易日历
            df = ak.tool_trade_date_hist_sina()
            if df is not None and not df.empty:
                return [datetime.strptime(d, '%Y-%m-%d') for d in df['trade_date'].tolist()]
        except Exception as e:
            logger.error(f"Failed to get calendar: {e}")

        # 返回最近一年的工作日作为后备
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=365),
            end=datetime.now(),
            freq='B'  # 工作日
        )
        return [d.to_pydatetime() for d in dates]