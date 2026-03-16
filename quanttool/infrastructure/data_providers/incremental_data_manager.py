"""
增量数据管理器

核心设计：
1. 按股票维度存储数据（而不是按时间范围）
2. 记录每只股票的数据范围（最早日期、最新日期）
3. 增量拉取：只拉取缺失的日期范围
4. 自动合并：将新数据与缓存数据合并

使用场景：
- 分析股票时，优先使用缓存数据
- 只拉取缺失的日期范围
- 每日更新时，只拉取最新数据
"""

import os
import sqlite3
import pandas as pd
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from ...core.logging import get_logger

logger = get_logger(__name__)


class DataType:
    """数据类型常量"""
    STOCK_BAR = "stock_bar"      # 股票K线
    INDEX_BAR = "index_bar"      # 指数K线
    MONEY_FLOW = "money_flow"    # 资金流向
    FINANCE = "finance"          # 财务数据


@dataclass
class DataRange:
    """数据范围信息"""
    symbol: str
    data_type: str = DataType.STOCK_BAR
    earliest_date: datetime = None
    latest_date: datetime = None
    row_count: int = 0
    last_updated: datetime = None


class IncrementalDataManager:
    """
    增量数据管理器

    核心功能：
    1. 按股票维度存储数据
    2. 智能判断需要拉取的日期范围
    3. 自动合并新旧数据
    4. 支持数据过期策略

    使用示例：
    >>> manager = IncrementalDataManager()
    >>> # 获取数据（自动增量拉取）
    >>> df = manager.get_data("000001.SZ", "2024-01-01", "2024-12-31", fetcher)
    >>> # 每日更新
    >>> manager.update_latest(fetcher)
    """

    def __init__(
        self,
        cache_dir: str = ".cache/incremental_data",
        default_ttl_days: int = 1,
        max_cache_size_mb: int = 2048
    ):
        """
        初始化增量数据管理器

        Args:
            cache_dir: 缓存目录
            default_ttl_days: 数据过期天数（默认1天，即每日更新）
            max_cache_size_mb: 最大缓存大小（MB）
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_days = default_ttl_days
        self.max_cache_bytes = max_cache_size_mb * 1024 * 1024
        self._lock = threading.RLock()

        self._init_db()
        logger.info(f"IncrementalDataManager initialized at {self.cache_dir}")

    def _init_db(self) -> None:
        """初始化元数据库"""
        self.db_path = self.cache_dir / "data_meta.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)

        # 检查是否需要迁移旧表结构
        self._migrate_old_schema()

        # 数据范围元数据表（支持多数据类型）
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS data_ranges (
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL DEFAULT 'stock_bar',
                earliest_date TEXT NOT NULL,
                latest_date TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                PRIMARY KEY (symbol, data_type)
            )
        """)

        # 数据更新日志表（用于审计和回滚）
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL DEFAULT 'stock_bar',
                update_type TEXT NOT NULL,
                old_range_start TEXT,
                old_range_end TEXT,
                new_range_start TEXT,
                new_range_end TEXT,
                rows_added INTEGER,
                timestamp TEXT NOT NULL
            )
        """)

        # 创建索引
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON data_ranges(expires_at)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_symbol ON update_log(symbol)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_data_type ON data_ranges(data_type)
        """)

        self.conn.commit()

    def _migrate_old_schema(self) -> None:
        """迁移旧表结构"""
        # 检查旧表是否存在
        cursor = self.conn.execute("""
            SELECT name FROM sqlite_master WHERE type='table' AND name='data_ranges'
        """)
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            return

        # 检查是否有 data_type 列
        cursor = self.conn.execute("PRAGMA table_info(data_ranges)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'data_type' not in columns:
            # 需要迁移：重建表
            logger.info("迁移旧表结构，添加 data_type 支持...")

            # 备份旧数据
            self.conn.execute("""
                CREATE TABLE data_ranges_backup AS
                SELECT *, 'stock_bar' as data_type FROM data_ranges
            """)

            # 删除旧表
            self.conn.execute("DROP TABLE data_ranges")

            # 创建新表
            self.conn.execute("""
                CREATE TABLE data_ranges (
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL DEFAULT 'stock_bar',
                    earliest_date TEXT NOT NULL,
                    latest_date TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    PRIMARY KEY (symbol, data_type)
                )
            """)

            # 迁移数据
            self.conn.execute("""
                INSERT INTO data_ranges
                (symbol, data_type, earliest_date, latest_date, row_count, file_path, last_updated, expires_at, size_bytes)
                SELECT symbol, data_type, earliest_date, latest_date, row_count, file_path, last_updated, expires_at, size_bytes
                FROM data_ranges_backup
            """)

            # 删除备份表
            self.conn.execute("DROP TABLE data_ranges_backup")

            self.conn.commit()
            logger.info("data_ranges 表结构迁移完成")

        # 同样处理 update_log 表
        cursor = self.conn.execute("PRAGMA table_info(update_log)")
        log_columns = [col[1] for col in cursor.fetchall()]

        if 'data_type' not in log_columns:
            logger.info("迁移 update_log 表结构...")

            # 备份旧数据
            self.conn.execute("""
                CREATE TABLE update_log_backup AS
                SELECT *, 'stock_bar' as data_type FROM update_log
            """)

            # 删除旧表
            self.conn.execute("DROP TABLE update_log")

            # 创建新表
            self.conn.execute("""
                CREATE TABLE update_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    data_type TEXT NOT NULL DEFAULT 'stock_bar',
                    update_type TEXT NOT NULL,
                    old_range_start TEXT,
                    old_range_end TEXT,
                    new_range_start TEXT,
                    new_range_end TEXT,
                    rows_added INTEGER,
                    timestamp TEXT NOT NULL
                )
            """)

            # 迁移数据
            self.conn.execute("""
                INSERT INTO update_log
                (symbol, data_type, update_type, old_range_start, old_range_end, new_range_start, new_range_end, rows_added, timestamp)
                SELECT symbol, data_type, update_type, old_range_start, old_range_end, new_range_start, new_range_end, rows_added, timestamp
                FROM update_log_backup
            """)

            # 删除备份表
            self.conn.execute("DROP TABLE update_log_backup")

            self.conn.commit()
            logger.info("update_log 表结构迁移完成")

    def get_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        fetcher,
        data_type: str = DataType.STOCK_BAR,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取数据（自动增量拉取）

        核心逻辑：
        1. 检查缓存中是否有该股票的数据
        2. 计算需要拉取的日期范围
        3. 只拉取缺失的部分
        4. 合并数据并更新缓存

        Args:
            symbol: 股票/指数代码
            start_date: 开始日期
            end_date: 结束日期
            fetcher: 数据获取器（需要有 get_bars 方法）
            data_type: 数据类型 (stock_bar, index_bar, money_flow, finance)
            force_refresh: 是否强制刷新

        Returns:
            DataFrame: 数据
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        with self._lock:
            # 1. 获取缓存信息
            cached_range = self._get_data_range(symbol, data_type)

            if force_refresh:
                logger.debug(f"[{symbol}][{data_type}] 强制刷新，删除旧缓存")
                self._delete_data(symbol, data_type)
                cached_range = None

            # 2. 计算需要拉取的范围
            fetch_ranges = self._calculate_fetch_ranges(
                symbol, start_date, end_date, cached_range
            )

            # 3. 拉取缺失数据
            all_new_data = []
            for fetch_start, fetch_end in fetch_ranges:
                logger.debug(f"[{symbol}][{data_type}] 拉取数据: {fetch_start} ~ {fetch_end}")
                try:
                    new_data = fetcher.get_bars(
                        [symbol], fetch_start, fetch_end, "1d"
                    )
                    if symbol in new_data and not new_data[symbol].empty:
                        all_new_data.append(new_data[symbol])
                except Exception as e:
                    logger.error(f"[{symbol}][{data_type}] 拉取失败: {e}")

            # 4. 合并数据
            if cached_range and not force_refresh:
                # 读取缓存数据
                cached_data = self._load_data(symbol, data_type)
                if cached_data is not None and not cached_data.empty:
                    all_new_data.insert(0, cached_data)

            # 5. 合并并去重
            if all_new_data:
                final_data = pd.concat(all_new_data, ignore_index=True)

                # 确保有 timestamp 或 trade_date 列
                date_col = None
                for col in ['timestamp', 'trade_date', 'date']:
                    if col in final_data.columns:
                        date_col = col
                        break

                if date_col:
                    # 转换为 datetime
                    final_data[date_col] = pd.to_datetime(final_data[date_col])
                    # 去重（保留最新的）
                    final_data = final_data.drop_duplicates(
                        subset=[date_col], keep='last'
                    )
                    # 排序
                    final_data = final_data.sort_values(date_col).reset_index(drop=True)

                # 6. 保存到缓存
                if not final_data.empty:
                    self._save_data(symbol, final_data, data_type)

                # 7. 过滤到请求的范围
                if date_col:
                    final_data = final_data[
                        (final_data[date_col] >= start_date) &
                        (final_data[date_col] <= end_date)
                    ]

                return final_data

            return pd.DataFrame()

    def _calculate_fetch_ranges(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        cached_range: Optional[DataRange]
    ) -> List[Tuple[datetime, datetime]]:
        """
        计算需要拉取的日期范围

        情况分析：
        1. 无缓存：拉取完整范围
        2. 缓存完全覆盖：无需拉取
        3. 缓存部分覆盖：拉取缺失的前段和/或后段
        4. 缓存过期：只拉取增量部分

        注意：
        - 只比较日期部分，忽略时间
        - 如果缓存的最新日期是今天或昨天，不拉取后段（今天数据可能还没有）
        - 先判断是否包含交易日，不包含则跳过拉取

        Returns:
            List of (start_date, end_date) tuples
        """
        fetch_ranges = []

        # 标准化日期（只比较日期部分）
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        start_date_day = start_date.date() if hasattr(start_date, 'date') else start_date
        end_date_day = end_date.date() if hasattr(end_date, 'date') else end_date

        # 情况1：无缓存
        if cached_range is None:
            logger.debug(f"[{symbol}] 无缓存，拉取完整范围: {start_date_day} ~ {end_date_day}")
            return [(start_date, end_date)]

        # 获取缓存的日期范围（只取日期部分）
        earliest = cached_range.earliest_date.date() if hasattr(cached_range.earliest_date, 'date') else cached_range.earliest_date
        latest = cached_range.latest_date.date() if hasattr(cached_range.latest_date, 'date') else cached_range.latest_date

        # 需要前段数据？
        if start_date_day < earliest:
            fetch_start = start_date
            fetch_end = datetime.combine(earliest - timedelta(days=1), datetime.max.time())
            # 确保不超过 end_date
            fetch_end = min(fetch_end, end_date)

            # 检查是否包含交易日
            if self._has_trading_days(fetch_start.date(), fetch_end.date()):
                logger.debug(f"[{symbol}] 拉取前段: {fetch_start.date()} ~ {fetch_end.date()}")
                fetch_ranges.append((fetch_start, fetch_end))
            else:
                logger.debug(f"[{symbol}] 前段范围 {fetch_start.date()} ~ {fetch_end.date()} 无交易日，跳过")

        # 需要后段数据？
        # 关键逻辑：
        # 1. 如果缓存的最新日期 >= 昨天，说明数据是最新的，不需要拉取后段
        # 2. 如果缓存的最新日期 < 昨天，需要拉取从 latest+1 到 min(end_date, 昨天) 的数据
        if latest < yesterday:
            # 计算有效结束日期（不超过昨天，因为今天的数据可能还没有）
            effective_end = min(end_date_day, yesterday)
            if effective_end > latest:
                fetch_start = datetime.combine(latest + timedelta(days=1), datetime.min.time())
                fetch_start = max(fetch_start, start_date)
                fetch_end = datetime.combine(effective_end, datetime.max.time())
                fetch_end = min(fetch_end, end_date)

                # 检查是否包含交易日
                if self._has_trading_days(fetch_start.date(), fetch_end.date()):
                    logger.debug(f"[{symbol}] 拉取后段: {fetch_start.date()} ~ {fetch_end.date()}")
                    fetch_ranges.append((fetch_start, fetch_end))
                else:
                    logger.debug(f"[{symbol}] 后段范围 {fetch_start.date()} ~ {fetch_end.date()} 无交易日，跳过")
        elif latest >= yesterday:
            # 数据已经是最新的，但如果请求的范围更大，需要记录
            logger.debug(f"[{symbol}] 缓存最新日期 {latest} >= 昨天 {yesterday}，数据已是最新")

        if not fetch_ranges:
            logger.debug(f"[{symbol}] 缓存完全覆盖 ({earliest} ~ {latest})，无需拉取")

        return fetch_ranges

    def _has_trading_days(self, start_date, end_date) -> bool:
        """
        检查日期范围内是否可能包含交易日

        简单判断：
        - 排除纯周末（周六、周日）
        - 中国A股交易日：周一至周五（节假日除外，但节假日判断需要外部数据）

        Args:
            start_date: 开始日期 (date 对象)
            end_date: 结束日期 (date 对象)

        Returns:
            bool: 是否可能包含交易日
        """
        from datetime import date

        # 确保 start_date <= end_date
        if start_date > end_date:
            return False

        # 检查范围内是否包含周一至周五
        current = start_date
        while current <= end_date:
            # weekday(): 0=周一, 6=周日
            if current.weekday() < 5:  # 周一至周五
                return True
            current += timedelta(days=1)

        return False

    def _get_data_range(self, symbol: str, data_type: str = DataType.STOCK_BAR) -> Optional[DataRange]:
        """获取数据范围信息"""
        cursor = self.conn.execute(
            """SELECT symbol, data_type, earliest_date, latest_date, row_count, last_updated
               FROM data_ranges WHERE symbol = ? AND data_type = ?""",
            (symbol, data_type)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return DataRange(
            symbol=row[0],
            data_type=row[1],
            earliest_date=datetime.fromisoformat(row[2]),
            latest_date=datetime.fromisoformat(row[3]),
            row_count=row[4],
            last_updated=datetime.fromisoformat(row[5])
        )

    def _load_data(self, symbol: str, data_type: str = DataType.STOCK_BAR) -> Optional[pd.DataFrame]:
        """加载缓存数据"""
        cursor = self.conn.execute(
            "SELECT file_path FROM data_ranges WHERE symbol = ? AND data_type = ?",
            (symbol, data_type)
        )
        row = cursor.fetchone()

        if row is None:
            return None

        file_path = self.cache_dir / row[0]
        if not file_path.exists():
            logger.warning(f"[{symbol}][{data_type}] 缓存文件不存在: {file_path}")
            return None

        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"[{symbol}][{data_type}] 读取缓存失败: {e}")
            return None

    def _save_data(self, symbol: str, data: pd.DataFrame, data_type: str = DataType.STOCK_BAR) -> bool:
        """保存数据到缓存"""
        if data.empty:
            return False

        # 确定日期列
        date_col = None
        for col in ['timestamp', 'trade_date', 'date']:
            if col in data.columns:
                date_col = col
                break

        if date_col is None:
            logger.error(f"[{symbol}][{data_type}] 未找到日期列")
            return False

        # 确保日期格式正确
        data[date_col] = pd.to_datetime(data[date_col])

        # 计算数据范围
        earliest = data[date_col].min()
        latest = data[date_col].max()

        # 保存文件（包含数据类型以区分不同数据）
        safe_symbol = symbol.replace('.', '_')
        file_path = f"{safe_symbol}_{data_type}.parquet"
        full_path = self.cache_dir / file_path

        try:
            data.to_parquet(full_path, compression='snappy', index=False)
            size_bytes = full_path.stat().st_size

            # 更新元数据
            now = datetime.now()
            expires_at = now + timedelta(days=self.default_ttl_days)

            self.conn.execute("""
                INSERT OR REPLACE INTO data_ranges
                (symbol, data_type, earliest_date, latest_date, row_count, file_path, last_updated, expires_at, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                data_type,
                earliest.strftime("%Y-%m-%d"),
                latest.strftime("%Y-%m-%d"),
                len(data),
                file_path,
                now.isoformat(),
                expires_at.isoformat(),
                size_bytes
            ))

            # 记录更新日志
            self.conn.execute("""
                INSERT INTO update_log
                (symbol, data_type, update_type, new_range_start, new_range_end, rows_added, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                data_type,
                'update',
                earliest.strftime("%Y-%m-%d"),
                latest.strftime("%Y-%m-%d"),
                len(data),
                now.isoformat()
            ))

            self.conn.commit()

            logger.debug(f"[{symbol}][{data_type}] 保存缓存: {earliest.date()} ~ {latest.date()}, {len(data)} 行")
            return True

        except Exception as e:
            logger.error(f"[{symbol}][{data_type}] 保存缓存失败: {e}")
            return False

    def _delete_data(self, symbol: str, data_type: str = DataType.STOCK_BAR) -> bool:
        """删除缓存数据"""
        try:
            cursor = self.conn.execute(
                "SELECT file_path FROM data_ranges WHERE symbol = ? AND data_type = ?",
                (symbol, data_type)
            )
            row = cursor.fetchone()

            if row:
                file_path = self.cache_dir / row[0]
                if file_path.exists():
                    file_path.unlink()

                self.conn.execute(
                    "DELETE FROM data_ranges WHERE symbol = ? AND data_type = ?",
                    (symbol, data_type)
                )
                self.conn.commit()

            return True
        except Exception as e:
            logger.error(f"[{symbol}][{data_type}] 删除缓存失败: {e}")
            return False

    def update_latest(
        self,
        symbols: List[str],
        fetcher,
        days_back: int = 30,
        data_type: str = DataType.STOCK_BAR
    ) -> Dict[str, int]:
        """
        批量更新最新数据

        用于每日定时任务，只拉取最新数据

        Args:
            symbols: 股票/指数列表
            fetcher: 数据获取器
            days_back: 回溯天数（防止停牌股票漏数据）
            data_type: 数据类型

        Returns:
            Dict[symbol, rows_added]: 每只股票新增的行数
        """
        results = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        for symbol in symbols:
            cached_range = self._get_data_range(symbol, data_type)

            if cached_range:
                # 只拉取最新部分
                fetch_start = cached_range.latest_date + timedelta(days=1)
                if fetch_start > end_date:
                    results[symbol] = 0
                    continue

                try:
                    new_data = fetcher.get_bars(
                        [symbol], fetch_start, end_date, "1d"
                    )
                    if symbol in new_data and not new_data[symbol].empty:
                        # 合并
                        old_data = self._load_data(symbol, data_type)
                        if old_data is not None:
                            combined = pd.concat([old_data, new_data[symbol]], ignore_index=True)
                            # 去重
                            date_col = 'timestamp' if 'timestamp' in combined.columns else 'trade_date'
                            combined = combined.drop_duplicates(subset=[date_col], keep='last')
                            combined = combined.sort_values(date_col).reset_index(drop=True)
                            self._save_data(symbol, combined, data_type)
                            results[symbol] = len(new_data[symbol])
                        else:
                            self._save_data(symbol, new_data[symbol], data_type)
                            results[symbol] = len(new_data[symbol])
                    else:
                        results[symbol] = 0
                except Exception as e:
                    logger.error(f"[{symbol}][{data_type}] 更新失败: {e}")
                    results[symbol] = 0
            else:
                # 无缓存，拉取完整范围
                try:
                    data = fetcher.get_bars(
                        [symbol], start_date, end_date, "1d"
                    )
                    if symbol in data and not data[symbol].empty:
                        self._save_data(symbol, data[symbol], data_type)
                        results[symbol] = len(data[symbol])
                    else:
                        results[symbol] = 0
                except Exception as e:
                    logger.error(f"[{symbol}][{data_type}] 初始拉取失败: {e}")
                    results[symbol] = 0

        return results

    def get_cache_stats(self, data_type: str = None) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            if data_type:
                cursor = self.conn.execute("""
                    SELECT
                        COUNT(*) as symbol_count,
                        COALESCE(SUM(row_count), 0) as total_rows,
                        COALESCE(SUM(size_bytes), 0) as total_size
                    FROM data_ranges
                    WHERE data_type = ?
                """, (data_type,))
            else:
                cursor = self.conn.execute("""
                    SELECT
                        COUNT(*) as symbol_count,
                        COALESCE(SUM(row_count), 0) as total_rows,
                        COALESCE(SUM(size_bytes), 0) as total_size
                    FROM data_ranges
                """)
            row = cursor.fetchone()

            # 按数据类型分组统计
            cursor_by_type = self.conn.execute("""
                SELECT data_type, COUNT(*) as count, SUM(row_count) as rows
                FROM data_ranges
                GROUP BY data_type
            """)
            by_type = {
                r[0]: {"count": r[1], "rows": r[2]}
                for r in cursor_by_type.fetchall()
            }

            return {
                "symbol_count": row[0],
                "total_rows": row[1],
                "total_size_mb": round(row[2] / (1024 * 1024), 2),
                "cache_dir": str(self.cache_dir),
                "by_type": by_type
            }

    def list_symbols(self, data_type: str = None) -> List[Dict[str, Any]]:
        """列出所有已缓存的数据"""
        with self._lock:
            if data_type:
                cursor = self.conn.execute("""
                    SELECT symbol, data_type, earliest_date, latest_date, row_count, last_updated, expires_at
                    FROM data_ranges
                    WHERE data_type = ?
                    ORDER BY last_updated DESC
                """, (data_type,))
            else:
                cursor = self.conn.execute("""
                    SELECT symbol, data_type, earliest_date, latest_date, row_count, last_updated, expires_at
                    FROM data_ranges
                    ORDER BY last_updated DESC
                """)

            return [
                {
                    "symbol": row[0],
                    "data_type": row[1],
                    "earliest_date": row[2],
                    "latest_date": row[3],
                    "row_count": row[4],
                    "last_updated": row[5],
                    "expires_at": row[6]
                }
                for row in cursor.fetchall()
            ]

    def clear_expired(self) -> int:
        """清理过期数据"""
        with self._lock:
            now = datetime.now().isoformat()
            cursor = self.conn.execute(
                "SELECT symbol, data_type, file_path FROM data_ranges WHERE expires_at < ?",
                (now,)
            )
            expired = cursor.fetchall()

            count = 0
            for symbol, data_type, file_path in expired:
                full_path = self.cache_dir / file_path
                if full_path.exists():
                    full_path.unlink()

                self.conn.execute(
                    "DELETE FROM data_ranges WHERE symbol = ? AND data_type = ?",
                    (symbol, data_type)
                )
                count += 1

            self.conn.commit()

            if count > 0:
                logger.debug(f"清理过期数据: {count} 条记录")

            return count

    def close(self) -> None:
        """关闭数据库连接"""
        if self.conn:
            self.conn.close()
            logger.debug("IncrementalDataManager connection closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ==================== 便捷方法 ====================

    def get_stock_data(
        self,
        symbol: str,
        days: int = 120,
        fetcher=None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取股票K线数据（便捷方法）

        Args:
            symbol: 股票代码
            days: 历史天数
            fetcher: 数据获取器
            force_refresh: 是否强制刷新

        Returns:
            DataFrame: 股票K线数据
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.get_data(
            symbol, start_date, end_date, fetcher,
            data_type=DataType.STOCK_BAR,
            force_refresh=force_refresh
        )

    def get_index_data(
        self,
        index_code: str,
        days: int = 120,
        fetcher=None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取指数K线数据（便捷方法）

        Args:
            index_code: 指数代码（如 000300.SH）
            days: 历史天数
            fetcher: 数据获取器
            force_refresh: 是否强制刷新

        Returns:
            DataFrame: 指数K线数据
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.get_data(
            index_code, start_date, end_date, fetcher,
            data_type=DataType.INDEX_BAR,
            force_refresh=force_refresh
        )

    def get_money_flow(
        self,
        symbol: str,
        days: int = 60,
        fetcher=None,
        force_refresh: bool = False
    ) -> pd.DataFrame:
        """
        获取资金流向数据（便捷方法）

        Args:
            symbol: 股票代码
            days: 历史天数
            fetcher: 数据获取器
            force_refresh: 是否强制刷新

        Returns:
            DataFrame: 资金流向数据
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return self.get_data(
            symbol, start_date, end_date, fetcher,
            data_type=DataType.MONEY_FLOW,
            force_refresh=force_refresh
        )


# 单例实例
_instance: Optional[IncrementalDataManager] = None


def get_incremental_manager() -> IncrementalDataManager:
    """获取增量数据管理器单例"""
    global _instance
    if _instance is None:
        _instance = IncrementalDataManager()
    return _instance
