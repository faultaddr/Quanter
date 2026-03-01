"""Calendar interface for QuantTool."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List


class ICalendar(ABC):
    """Abstract interface for calendar systems."""

    @abstractmethod
    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a date is a trading day.

        Args:
            date: Date to check

        Returns:
            True if trading day, False otherwise
        """
        pass

    @abstractmethod
    def is_trading_time(self, dt: datetime) -> bool:
        """
        Check if a datetime is within trading hours.

        Args:
            dt: Datetime to check

        Returns:
            True if within trading hours, False otherwise
        """
        pass

    @abstractmethod
    def get_next_trading_day(self, date: datetime) -> datetime:
        """
        Get the next trading day after the given date.

        Args:
            date: Starting date

        Returns:
            Next trading day
        """
        pass

    @abstractmethod
    def get_prev_trading_day(self, date: datetime) -> datetime:
        """
        Get the previous trading day before the given date.

        Args:
            date: Starting date

        Returns:
            Previous trading day
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_trading_hours(self, date: datetime) -> List[tuple]:
        """
        Get trading hours for a given date.

        Args:
            date: Date to get trading hours for

        Returns:
            List of tuples containing (start_time, end_time) for each session
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
