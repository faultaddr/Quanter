"""A-Share calendar implementation."""

from datetime import datetime, timedelta
from typing import List
import pandas as pd
from zoneinfo import ZoneInfo
from ...domain.interfaces.calendar import ICalendar
from ...core.registry import registry, ComponentType
from ...core.timeutils import get_next_trading_time, generate_trading_times


@registry.register(ComponentType.CALENDAR, "a_share")
class AShareCalendar(ICalendar):
    """A-Share market calendar implementation."""

    def __init__(
        self, holidays: List[datetime] = None, early_closes: List[datetime] = None
    ):
        """
        Initialize A-Share calendar.

        Args:
            holidays: List of holiday dates
            early_closes: List of early close dates
        """
        self.holidays = holidays or []
        self.early_closes = early_closes or []
        self.timezone = ZoneInfo("Asia/Shanghai")

    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a date is a trading day.

        Args:
            date: Date to check

        Returns:
            True if trading day, False otherwise
        """
        # Convert to Shanghai timezone
        date = date.astimezone(self.timezone)

        # Check if it's a weekend
        if date.weekday() >= 5:  # Saturday and Sunday
            return False

        # Check if it's a holiday
        date_only = date.date()
        for holiday in self.holidays:
            if holiday.date() == date_only:
                return False

        return True

    def is_trading_time(self, dt: datetime) -> bool:
        """
        Check if a datetime is within trading hours.

        Args:
            dt: Datetime to check

        Returns:
            True if within trading hours, False otherwise
        """
        dt = dt.astimezone(self.timezone)

        # Check if it's a trading day
        if not self.is_trading_day(dt):
            return False

        # Check if it's a holiday or early close day
        date_only = dt.date()
        is_early_close = any(ec.date() == date_only for ec in self.early_closes)

        # Get trading hours for the date
        trading_sessions = self.get_trading_hours(dt)

        # Check if time is within any of the trading sessions
        for start, end in trading_sessions:
            if start <= dt <= end:
                return True

        return False

    def get_next_trading_day(self, date: datetime) -> datetime:
        """
        Get the next trading day after the given date.

        Args:
            date: Starting date

        Returns:
            Next trading day
        """
        candidate = date.astimezone(self.timezone) + timedelta(days=1)

        while not self.is_trading_day(candidate):
            candidate += timedelta(days=1)

        return candidate

    def get_prev_trading_day(self, date: datetime) -> datetime:
        """
        Get the previous trading day before the given date.

        Args:
            date: Starting date

        Returns:
            Previous trading day
        """
        candidate = date.astimezone(self.timezone) - timedelta(days=1)

        while not self.is_trading_day(candidate):
            candidate -= timedelta(days=1)

        return candidate

    def get_trading_days(
        self, start_date: datetime, end_date: datetime
    ) -> List[datetime]:
        """
        Get all trading days in the given range.

        Args:
            start_date: Start date (inclusive)
            end_date: End date (inclusive)

        Returns:
            List of trading days in the range
        """
        start_date = start_date.astimezone(self.timezone)
        end_date = end_date.astimezone(self.timezone)

        trading_days = []
        current_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)

        while current_date <= end_date:
            if self.is_trading_day(current_date):
                trading_days.append(current_date)
            current_date += timedelta(days=1)

        return trading_days

    def get_trading_hours(self, date: datetime) -> List[tuple]:
        """
        Get trading hours for a given date.

        Args:
            date: Date to get trading hours for

        Returns:
            List of tuples containing (start_time, end_time) for each session
        """
        date = date.astimezone(self.timezone)
        date_only = date.date()

        # Check if it's an early close day
        is_early_close = any(ec.date() == date_only for ec in self.early_closes)

        # A-Share trading sessions
        morning_start = datetime.combine(
            date_only, pd.Timestamp("09:30").time()
        ).replace(tzinfo=self.timezone)
        morning_end = datetime.combine(date_only, pd.Timestamp("11:30").time()).replace(
            tzinfo=self.timezone
        )

        afternoon_start = datetime.combine(
            date_only, pd.Timestamp("13:00").time()
        ).replace(tzinfo=self.timezone)

        # On early close days, afternoon session ends at 14:00
        if is_early_close:
            afternoon_end = datetime.combine(
                date_only, pd.Timestamp("14:00").time()
            ).replace(tzinfo=self.timezone)
        else:
            afternoon_end = datetime.combine(
                date_only, pd.Timestamp("15:00").time()
            ).replace(tzinfo=self.timezone)

        return [(morning_start, morning_end), (afternoon_start, afternoon_end)]

    def get_next_trading_time(
        self, dt: datetime, timeframe_minutes: int = 1
    ) -> datetime:
        """
        Get the next trading time aligned to the specified timeframe.

        Args:
            dt: Current datetime
            timeframe_minutes: Timeframe in minutes

        Returns:
            Next aligned trading time
        """
        return get_next_trading_time(dt, timeframe_minutes)

    def get_aligned_trading_times(
        self, start: datetime, end: datetime, timeframe_minutes: int = 10
    ) -> List[datetime]:
        """
        Generate all trading times between start and end, aligned to the specified timeframe.

        Args:
            start: Start datetime
            end: End datetime
            timeframe_minutes: Timeframe in minutes

        Returns:
            List of aligned trading times
        """
        return generate_trading_times(start, end, timeframe_minutes)
