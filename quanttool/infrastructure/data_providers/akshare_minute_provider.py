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
            logger.error(f"Failed to initialize AkShareMinuteProvider: {e}")
            raise

    def _get_cached(self, key: str) -> Optional[Any]:
        """从缓存获取数据"""
        if key in self._cache:
            data, timestamp = self._cache[key]
            if time.time() - timestamp < self._cache_ttl:
                return data
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """设置缓存"""
        self._cache[key] = (data, time.time())

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
                'name': row.get('名称', ''),
                'price': float(row.get('最新价', 0)),
                'open': float(row.get('今开', 0)),
                'high': float(row.get('最高', 0)),
                'low': float(row.get('最低', 0)),
                'volume': float(row.get('成交量', 0)),
                'amount': float(row.get('成交额', 0)),
                'pct_change': float(row.get('涨跌幅', 0)),
                'change': float(row.get('涨跌额', 0)),
                'turnover': float(row.get('换手率', 0)),
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

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """搜索股票（带反爬虫防护）"""
        if not self._initialized:
            self.initialize()

        try:
            # 延迟控制
            self._delay_controller.wait()

            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return []

            # 搜索代码或名称
            mask = df['代码'].str.contains(query, na=False) | \
                   df['名称'].str.contains(query, na=False)

            results = []
            for _, row in df[mask].head(20).iterrows():
                results.append({
                    'symbol': row['代码'],
                    'name': row['名称'],
                    'price': row.get('最新价', 0)
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