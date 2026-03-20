"""
批量时间处理模块

支持灵活的时间处理方式：
- 当前时间作业
- 单个时间作业
- 枚举时间作业
- 区间时间作业
- 智能交易日识别
- 历史数据回填

参考 InStock 的批量作业模式
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Dict, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class TimeMode(Enum):
    """时间模式"""
    CURRENT = "current"      # 当前时间
    SINGLE = "single"        # 单个时间
    ENUMERATE = "enumerate"  # 枚举时间
    RANGE = "range"          # 区间时间


@dataclass
class TimeConfig:
    """时间配置"""
    mode: TimeMode
    dates: List[datetime]    # 要处理的日期列表
    is_trading_days: List[bool]  # 是否为交易日


@dataclass
class BatchJobResult:
    """批量作业结果"""
    total_dates: int
    processed_dates: int
    success_count: int
    failure_count: int
    results: Dict[str, Any]
    errors: List[Dict[str, Any]]
    elapsed_time: float


class TradingCalendar:
    """
    交易日历

    智能识别A股交易日
    """

    # 简化的交易日判断（不考虑节假日调休）
    # 真实场景应使用交易所交易日历
    WEEKEND_DAYS = {5, 6}  # 周六、周日

    # 2024年主要节假日（示例）
    HOLIDAYS_2024 = [
        # 元旦
        datetime(2024, 1, 1),
        # 春节
        datetime(2024, 2, 9), datetime(2024, 2, 10), datetime(2024, 2, 11),
        datetime(2024, 2, 12), datetime(2024, 2, 13), datetime(2024, 2, 14),
        datetime(2024, 2, 15), datetime(2024, 2, 16), datetime(2024, 2, 17),
        # 清明
        datetime(2024, 4, 4), datetime(2024, 4, 5), datetime(2024, 4, 6),
        # 劳动节
        datetime(2024, 5, 1), datetime(2024, 5, 2), datetime(2024, 5, 3),
        datetime(2024, 5, 4), datetime(2024, 5, 5),
        # 端午
        datetime(2024, 6, 8), datetime(2024, 6, 9), datetime(2024, 6, 10),
        # 中秋
        datetime(2024, 9, 15), datetime(2024, 9, 16), datetime(2024, 9, 17),
        # 国庆
        datetime(2024, 10, 1), datetime(2024, 10, 2), datetime(2024, 10, 3),
        datetime(2024, 10, 4), datetime(2024, 10, 5), datetime(2024, 10, 6),
        datetime(2024, 10, 7),
    ]

    # 2025年主要节假日（示例）
    HOLIDAYS_2025 = [
        # 元旦
        datetime(2025, 1, 1),
        # 春节
        datetime(2025, 1, 28), datetime(2025, 1, 29), datetime(2025, 1, 30),
        datetime(2025, 1, 31), datetime(2025, 2, 1), datetime(2025, 2, 2),
        datetime(2025, 2, 3), datetime(2025, 2, 4),
        # 清明
        datetime(2025, 4, 4), datetime(2025, 4, 5), datetime(2025, 4, 6),
        # 劳动节
        datetime(2025, 5, 1), datetime(2025, 5, 2), datetime(2025, 5, 3),
        datetime(2025, 5, 4), datetime(2025, 5, 5),
        # 端午
        datetime(2025, 5, 31), datetime(2025, 6, 1), datetime(2025, 6, 2),
        # 中秋+国庆
        datetime(2025, 10, 1), datetime(2025, 10, 2), datetime(2025, 10, 3),
        datetime(2025, 10, 4), datetime(2025, 10, 5), datetime(2025, 10, 6),
        datetime(2025, 10, 7), datetime(2025, 10, 8),
    ]

    # 2026年主要节假日
    HOLIDAYS_2026 = [
        # 元旦
        datetime(2026, 1, 1), datetime(2026, 1, 2), datetime(2026, 1, 3),
        # 春节
        datetime(2026, 2, 16), datetime(2026, 2, 17), datetime(2026, 2, 18),
        datetime(2026, 2, 19), datetime(2026, 2, 20), datetime(2026, 2, 21),
        datetime(2026, 2, 22), datetime(2026, 2, 23),
    ]

    ALL_HOLIDAYS = HOLIDAYS_2024 + HOLIDAYS_2025 + HOLIDAYS_2026

    @classmethod
    def is_trading_day(cls, date: Union[datetime, str]) -> bool:
        """
        判断是否为交易日

        Args:
            date: 日期

        Returns:
            是否为交易日
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        # 周末不是交易日
        if date.weekday() in cls.WEEKEND_DAYS:
            return False

        # 节假日不是交易日
        date_only = datetime(date.year, date.month, date.day)
        if date_only in cls.ALL_HOLIDAYS:
            return False

        return True

    @classmethod
    def get_trading_days(
        cls,
        start_date: Union[datetime, str],
        end_date: Union[datetime, str]
    ) -> List[datetime]:
        """
        获取两个日期之间的所有交易日

        Args:
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            交易日列表
        """
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')

        trading_days = []
        current = start_date

        while current <= end_date:
            if cls.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)

        return trading_days

    @classmethod
    def get_previous_trading_day(cls, date: Union[datetime, str]) -> datetime:
        """
        获取前一个交易日

        Args:
            date: 当前日期

        Returns:
            前一个交易日
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        current = date - timedelta(days=1)
        while not cls.is_trading_day(current):
            current -= timedelta(days=1)

        return current

    @classmethod
    def get_next_trading_day(cls, date: Union[datetime, str]) -> datetime:
        """
        获取下一个交易日

        Args:
            date: 当前日期

        Returns:
            下一个交易日
        """
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d')

        current = date + timedelta(days=1)
        while not cls.is_trading_day(current):
            current += timedelta(days=1)

        return current

    @classmethod
    def get_latest_trading_day(cls, date: Optional[datetime] = None) -> datetime:
        """
        获取最近的交易日

        Args:
            date: 参考日期，默认今天

        Returns:
            最近的交易日
        """
        if date is None:
            date = datetime.now()

        if cls.is_trading_day(date):
            # 如果当前是交易日，检查时间
            if date.hour >= 15:
                return date
            else:
                return cls.get_previous_trading_day(date)
        else:
            return cls.get_previous_trading_day(date)


class TimeParser:
    """
    时间解析器

    支持多种时间格式解析
    """

    @staticmethod
    def parse(date_str: str) -> datetime:
        """
        解析日期字符串

        支持格式：
        - 2024-01-15
        - 20240115
        - 2024/01/15

        Args:
            date_str: 日期字符串

        Returns:
            datetime对象
        """
        date_str = date_str.strip()

        # 尝试不同格式
        formats = [
            '%Y-%m-%d',
            '%Y%m%d',
            '%Y/%m/%d',
            '%Y.%m.%d'
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        raise ValueError(f"无法解析日期: {date_str}")

    @staticmethod
    def parse_list(date_str: str) -> List[datetime]:
        """
        解析日期列表字符串

        支持格式：
        - 2024-01-01,2024-01-02,2024-01-03

        Args:
            date_str: 日期列表字符串

        Returns:
            日期列表
        """
        dates = []
        for d in date_str.split(','):
            d = d.strip()
            if d:
                dates.append(TimeParser.parse(d))
        return dates

    @staticmethod
    def parse_range(
        start_str: str,
        end_str: str,
        trading_days_only: bool = True
    ) -> List[datetime]:
        """
        解析日期范围

        Args:
            start_str: 开始日期
            end_str: 结束日期
            trading_days_only: 是否只返回交易日

        Returns:
            日期列表
        """
        start = TimeParser.parse(start_str)
        end = TimeParser.parse(end_str)

        if trading_days_only:
            return TradingCalendar.get_trading_days(start, end)
        else:
            dates = []
            current = start
            while current <= end:
                dates.append(current)
                current += timedelta(days=1)
            return dates


class BatchTimeProcessor:
    """
    批量时间处理器

    支持多种时间模式的批量处理
    """

    def __init__(
        self,
        trading_days_only: bool = True,
        show_progress: bool = True
    ):
        """
        初始化处理器

        Args:
            trading_days_only: 是否只处理交易日
            show_progress: 是否显示进度
        """
        self.trading_days_only = trading_days_only
        self.show_progress = show_progress

    def parse_time_args(
        self,
        time_args: Optional[List[str]] = None
    ) -> TimeConfig:
        """
        解析时间参数

        Args:
            time_args: 时间参数列表
                - None: 当前时间
                - ['2024-01-15']: 单个时间
                - ['2024-01-01,2024-01-02,2024-01-03']: 枚举时间
                - ['2024-01-01', '2024-01-31']: 区间时间

        Returns:
            TimeConfig: 时间配置
        """
        if time_args is None or len(time_args) == 0:
            # 当前时间模式
            now = datetime.now()
            if self.trading_days_only:
                dates = [TradingCalendar.get_latest_trading_day(now)]
            else:
                dates = [now]

            return TimeConfig(
                mode=TimeMode.CURRENT,
                dates=dates,
                is_trading_days=[TradingCalendar.is_trading_day(d) for d in dates]
            )

        elif len(time_args) == 1:
            arg = time_args[0]

            if ',' in arg:
                # 枚举时间模式
                dates = TimeParser.parse_list(arg)
            else:
                # 单个时间模式
                dates = [TimeParser.parse(arg)]

            if self.trading_days_only:
                dates = [d for d in dates if TradingCalendar.is_trading_day(d)]

            return TimeConfig(
                mode=TimeMode.ENUMERATE if ',' in arg else TimeMode.SINGLE,
                dates=dates,
                is_trading_days=[TradingCalendar.is_trading_day(d) for d in dates]
            )

        elif len(time_args) >= 2:
            # 区间时间模式
            start = time_args[0]
            end = time_args[1]

            dates = TimeParser.parse_range(start, end, self.trading_days_only)

            return TimeConfig(
                mode=TimeMode.RANGE,
                dates=dates,
                is_trading_days=[True] * len(dates) if self.trading_days_only else
                              [TradingCalendar.is_trading_day(d) for d in dates]
            )

        return TimeConfig(mode=TimeMode.CURRENT, dates=[], is_trading_days=[])

    def process(
        self,
        time_config: TimeConfig,
        job_func: Callable[[datetime], Any],
        on_success: Optional[Callable[[datetime, Any], None]] = None,
        on_failure: Optional[Callable[[datetime, Exception], None]] = None
    ) -> BatchJobResult:
        """
        执行批量作业

        Args:
            time_config: 时间配置
            job_func: 作业函数，接收datetime参数
            on_success: 成功回调
            on_failure: 失败回调

        Returns:
            BatchJobResult: 批量作业结果
        """
        import time
        start_time = time.time()

        results = {}
        errors = []
        success_count = 0
        failure_count = 0

        total = len(time_config.dates)

        for i, date in enumerate(time_config.dates):
            try:
                if self.show_progress:
                    print(f"处理 [{i+1}/{total}]: {date.strftime('%Y-%m-%d')}")

                result = job_func(date)
                results[date.strftime('%Y-%m-%d')] = result
                success_count += 1

                if on_success:
                    on_success(date, result)

            except Exception as e:
                failure_count += 1
                error_info = {
                    'date': date.strftime('%Y-%m-%d'),
                    'error': str(e)
                }
                errors.append(error_info)
                results[date.strftime('%Y-%m-%d')] = None

                if on_failure:
                    on_failure(date, e)

                if self.show_progress:
                    print(f"  错误: {e}")

        elapsed_time = time.time() - start_time

        return BatchJobResult(
            total_dates=total,
            processed_dates=success_count + failure_count,
            success_count=success_count,
            failure_count=failure_count,
            results=results,
            errors=errors,
            elapsed_time=elapsed_time
        )

    def run_job(
        self,
        time_args: Optional[List[str]] = None,
        job_func: Callable[[datetime], Any] = None,
        on_success: Optional[Callable[[datetime, Any], None]] = None,
        on_failure: Optional[Callable[[datetime, Exception], None]] = None
    ) -> BatchJobResult:
        """
        便捷方法：解析时间参数并执行作业

        Args:
            time_args: 时间参数
            job_func: 作业函数
            on_success: 成功回调
            on_failure: 失败回调

        Returns:
            BatchJobResult: 批量作业结果
        """
        time_config = self.parse_time_args(time_args)
        return self.process(time_config, job_func, on_success, on_failure)


def run_batch_job(
    job_func: Callable[[datetime], Any],
    time_args: Optional[List[str]] = None,
    trading_days_only: bool = True,
    show_progress: bool = True
) -> BatchJobResult:
    """
    便捷函数：运行批量作业

    Args:
        job_func: 作业函数
        time_args: 时间参数
        trading_days_only: 是否只处理交易日
        show_progress: 是否显示进度

    Returns:
        BatchJobResult: 批量作业结果

    示例:
        # 当前时间作业
        run_batch_job(my_job)

        # 单个时间作业
        run_batch_job(my_job, ['2024-01-15'])

        # 枚举时间作业
        run_batch_job(my_job, ['2024-01-01,2024-01-02,2024-01-03'])

        # 区间时间作业
        run_batch_job(my_job, ['2024-01-01', '2024-01-31'])
    """
    processor = BatchTimeProcessor(
        trading_days_only=trading_days_only,
        show_progress=show_progress
    )
    return processor.run_job(time_args, job_func)


def format_batch_result(result: BatchJobResult) -> str:
    """
    格式化批量作业结果

    Args:
        result: 批量作业结果

    Returns:
        格式化字符串
    """
    lines = []
    lines.append("=" * 50)
    lines.append("批量作业执行结果")
    lines.append("=" * 50)
    lines.append(f"总日期数: {result.total_dates}")
    lines.append(f"处理数: {result.processed_dates}")
    lines.append(f"成功: {result.success_count}")
    lines.append(f"失败: {result.failure_count}")
    lines.append(f"耗时: {result.elapsed_time:.2f}秒")
    lines.append("")

    if result.errors:
        lines.append("错误详情:")
        for err in result.errors:
            lines.append(f"  - {err['date']}: {err['error']}")

    return '\n'.join(lines)
