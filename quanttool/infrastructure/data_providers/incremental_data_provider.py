"""
增量数据提供者

集成了增量数据管理器的数据提供者，用于回测系统。
优先使用缓存数据，减少网络请求。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd

from ...domain.interfaces.data_provider import IDataProvider
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger
from .incremental_data_manager import IncrementalDataManager, DataType
from .data_fetcher import EnhancedDataFetcher

logger = get_logger(__name__)


@registry.register(ComponentType.DATA_PROVIDER, "incremental_data_fetcher")
class IncrementalDataProvider(IDataProvider):
    """
    增量数据提供者

    特点：
    - 优先使用本地缓存数据
    - 只拉取缺失的日期范围
    - 自动合并新旧数据
    - 减少网络请求次数

    使用场景：
    - 回测系统
    - 需要频繁访问相同数据的应用
    """

    def __init__(
        self,
        cache_dir: str = ".cache/incremental_data",
        fallback_provider: str = "enhanced_data_fetcher",
        use_cache_only: bool = False,
        default_ttl_days: int = 1
    ):
        """
        初始化增量数据提供者

        Args:
            cache_dir: 缓存目录
            fallback_provider: 回退数据提供者名称
            use_cache_only: 是否只使用缓存（不拉取新数据）
            default_ttl_days: 数据过期天数
        """
        self.cache_dir = cache_dir
        self.fallback_provider = fallback_provider
        self.use_cache_only = use_cache_only
        self.default_ttl_days = default_ttl_days

        self._incremental_manager: Optional[IncrementalDataManager] = None
        self._fallback_fetcher: Optional[IDataProvider] = None
        self._initialized = False

    def initialize(self) -> None:
        """初始化数据提供者"""
        if self._initialized:
            return

        # 初始化增量数据管理器
        self._incremental_manager = IncrementalDataManager(
            cache_dir=self.cache_dir,
            default_ttl_days=self.default_ttl_days
        )

        # 初始化回退数据提供者
        if not self.use_cache_only:
            try:
                provider_class = registry.get(
                    ComponentType.DATA_PROVIDER,
                    self.fallback_provider
                )
                self._fallback_fetcher = provider_class()
                if hasattr(self._fallback_fetcher, 'initialize'):
                    self._fallback_fetcher.initialize()
            except Exception as e:
                logger.warning(f"无法初始化回退数据提供者: {e}")
                self._fallback_fetcher = None

        self._initialized = True
        logger.info(f"IncrementalDataProvider initialized, cache_only={self.use_cache_only}")

    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        获取K线数据（优先使用缓存）

        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            timeframe: 时间周期

        Returns:
            字典: {symbol: DataFrame}
        """
        if not self._initialized:
            self.initialize()

        results = {}

        for symbol in symbols:
            try:
                df = self._get_single_symbol_data(
                    symbol, start_date, end_date, timeframe
                )
                if not df.empty:
                    results[symbol] = df
            except Exception as e:
                logger.error(f"获取 {symbol} 数据失败: {e}")

        return results

    def _get_single_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> pd.DataFrame:
        """获取单只股票数据"""

        # 标准化股票代码格式
        cache_symbol = self._normalize_symbol(symbol)

        # 确定数据类型
        data_type = DataType.INDEX_BAR if self._is_index(symbol) else DataType.STOCK_BAR

        # 尝试从缓存读取
        cached_df = self._read_from_cache(cache_symbol, data_type)

        if cached_df is not None and not cached_df.empty:
            # 检查缓存数据是否覆盖请求范围
            if self._cache_covers_range(cached_df, start_date, end_date):
                logger.debug(f"缓存命中: {symbol}")
                return self._filter_by_date(cached_df, start_date, end_date)

            # 缓存部分覆盖，尝试增量拉取
            if not self.use_cache_only and self._fallback_fetcher:
                return self._incremental_fetch(
                    symbol, start_date, end_date, timeframe, data_type, cached_df
                )

            # 只使用缓存模式，返回已有数据
            return self._filter_by_date(cached_df, start_date, end_date)

        # 无缓存，从网络拉取
        if not self.use_cache_only and self._fallback_fetcher:
            return self._fetch_and_cache(
                symbol, start_date, end_date, timeframe, data_type
            )

        logger.warning(f"无缓存数据且网络获取禁用: {symbol}")
        return pd.DataFrame()

    def _read_from_cache(self, symbol: str, data_type: str) -> Optional[pd.DataFrame]:
        """从缓存读取数据"""
        try:
            return self._incremental_manager._load_data(symbol, data_type)
        except Exception as e:
            logger.debug(f"读取缓存失败 {symbol}: {e}")
            return None

    def _cache_covers_range(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> bool:
        """检查缓存数据是否覆盖请求范围"""
        if df.empty:
            return False

        # 获取日期列
        date_col = None
        for col in ['timestamp', 'trade_date', 'date']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return False

        # 转换日期
        dates = pd.to_datetime(df[date_col])
        cache_start = dates.min()
        cache_end = dates.max()

        # 只比较日期部分
        start_day = start_date.date() if hasattr(start_date, 'date') else start_date
        end_day = end_date.date() if hasattr(end_date, 'date') else end_date
        cache_start_day = cache_start.date() if hasattr(cache_start, 'date') else cache_start
        cache_end_day = cache_end.date() if hasattr(cache_end, 'date') else cache_end

        return cache_start_day <= start_day and cache_end_day >= end_day

    def _filter_by_date(
        self,
        df: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """按日期范围过滤数据"""
        if df.empty:
            return df

        date_col = None
        for col in ['timestamp', 'trade_date', 'date']:
            if col in df.columns:
                date_col = col
                break

        if date_col is None:
            return df

        df[date_col] = pd.to_datetime(df[date_col])
        return df[
            (df[date_col] >= start_date) &
            (df[date_col] <= end_date)
        ].reset_index(drop=True)

    def _incremental_fetch(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        data_type: str,
        cached_df: pd.DataFrame
    ) -> pd.DataFrame:
        """增量拉取数据"""

        # 使用增量数据管理器
        try:
            df = self._incremental_manager.get_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                fetcher=self._fallback_fetcher,
                data_type=data_type,
                force_refresh=False
            )
            return df
        except Exception as e:
            logger.warning(f"增量拉取失败 {symbol}: {e}")
            # 返回已有缓存数据
            return self._filter_by_date(cached_df, start_date, end_date)

    def _fetch_and_cache(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
        data_type: str
    ) -> pd.DataFrame:
        """从网络获取数据并缓存"""
        try:
            results = self._fallback_fetcher.get_bars(
                [symbol], start_date, end_date, timeframe
            )

            if symbol in results and not results[symbol].empty:
                df = results[symbol]
                # 保存到缓存
                self._incremental_manager._save_data(symbol, df, data_type)
                return df

        except Exception as e:
            logger.error(f"网络获取失败 {symbol}: {e}")

        return pd.DataFrame()

    def _is_index(self, symbol: str) -> bool:
        """判断是否为指数代码"""
        # 上证指数: 000001-000999, 9xxxxx
        # 深证指数: 399001-399999
        if symbol.startswith('000') or symbol.startswith('399') or symbol.startswith('9'):
            return True
        return False

    def _normalize_symbol(self, symbol: str) -> str:
        """
        标准化股票代码格式

        将多种输入格式转换为数据库格式: 600519 -> 600519.SH, 000001 -> 000001.SZ
        """
        # 已经是标准格式
        if '.' in symbol:
            return symbol

        # 清理代码
        code = symbol.replace('.XSHG', '').replace('.XSHE', '')
        code = code.replace('_SH', '').replace('_SZ', '')

        # 根据代码判断市场
        if code.startswith(('6', '5', '9')):
            return f"{code}.SH"
        else:
            return f"{code}.SZ"

    def update_cache(
        self,
        symbols: List[str],
        end_date: Optional[datetime] = None
    ) -> Dict[str, bool]:
        """
        更新缓存数据

        Args:
            symbols: 要更新的股票列表
            end_date: 结束日期（默认今天）

        Returns:
            更新结果: {symbol: success}
        """
        if not self._initialized:
            self.initialize()

        if end_date is None:
            end_date = datetime.now()

        results = {}
        for symbol in symbols:
            try:
                data_type = DataType.INDEX_BAR if self._is_index(symbol) else DataType.STOCK_BAR

                # 获取缓存的最新日期
                cached_range = self._incremental_manager._get_data_range(symbol, data_type)

                if cached_range and cached_range.latest_date:
                    # 从缓存的最新日期开始拉取
                    start_date = cached_range.latest_date
                else:
                    # 无缓存，拉取近5年数据
                    from datetime import timedelta
                    start_date = end_date - timedelta(days=5*365)

                # 获取数据
                df = self._incremental_manager.get_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    fetcher=self._fallback_fetcher,
                    data_type=data_type,
                    force_refresh=False
                )
                results[symbol] = not df.empty

            except Exception as e:
                logger.error(f"更新缓存失败 {symbol}: {e}")
                results[symbol] = False

        return results

    def get_cache_status(self) -> Dict:
        """获取缓存状态"""
        if not self._initialized:
            self.initialize()

        return self._incremental_manager.get_cache_stats()

    def get_supported_symbols(self) -> List[str]:
        """获取支持的股票代码列表（从缓存获取）"""
        if not self._initialized:
            self.initialize()

        # 从缓存目录获取已有的股票代码
        from pathlib import Path
        cache_dir = Path(self.cache_dir)
        symbols = set()

        for f in cache_dir.glob('*_stock_bar.parquet'):
            name = f.stem.replace('_stock_bar', '')
            symbols.add(name)

        for f in cache_dir.glob('*_index_bar.parquet'):
            name = f.stem.replace('_index_bar', '')
            symbols.add(name)

        return sorted(list(symbols))

    def get_latest_bar(self, symbol: str, timeframe: str = "1d") -> Optional[pd.DataFrame]:
        """获取最新的K线数据"""
        if not self._initialized:
            self.initialize()

        data_type = DataType.INDEX_BAR if self._is_index(symbol) else DataType.STOCK_BAR

        # 从缓存读取最新数据
        df = self._read_from_cache(symbol, data_type)
        if df is not None and not df.empty:
            # 返回最后一条记录
            return df.tail(1)

        return None

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """搜索股票代码"""
        # 简单实现：返回匹配的缓存中的股票
        supported = self.get_supported_symbols()
        query_upper = query.upper()

        results = []
        for symbol in supported:
            if query_upper in symbol.upper():
                results.append({
                    'symbol': symbol,
                    'name': symbol,
                    'source': 'cache'
                })

        return results[:20]  # 限制返回数量

    def get_calendar(self) -> List[datetime]:
        """获取交易日历"""
        # 简单实现：返回近两年的交易日
        from datetime import timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)

        # 从缓存数据中提取交易日
        dates = set()
        for symbol in self.get_supported_symbols()[:10]:  # 只检查前10个股票
            df = self._read_from_cache(symbol, DataType.STOCK_BAR)
            if df is not None and not df.empty:
                date_col = None
                for col in ['timestamp', 'trade_date', 'date']:
                    if col in df.columns:
                        date_col = col
                        break

                if date_col:
                    for d in pd.to_datetime(df[date_col]):
                        dates.add(d.date())

        return sorted([datetime.combine(d, datetime.min.time()) for d in dates])
