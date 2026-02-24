"""Time utilities for QuantTool."""

from datetime import datetime, timedelta, time
from typing import Union, List
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # For Python < 3.9, use backport
    from backports.zoneinfo import ZoneInfo

import pytz  # Alternative for older Python versions


def get_a_share_market_hours(date: datetime) -> List[tuple]:
    """
    Get A-share market hours for a given date.

    Args:
        date: Date to get market hours for

    Returns:
        List of tuples containing (start_time, end_time) for each session
    """
    shanghai_tz = ZoneInfo("Asia/Shanghai")
    date = date.astimezone(shanghai_tz).date()

    morning_start = datetime.combine(date, time(9, 30)).replace(tzinfo=shanghai_tz)
    morning_end = datetime.combine(date, time(11, 30)).replace(tzinfo=shanghai_tz)
    afternoon_start = datetime.combine(date, time(13, 0)).replace(tzinfo=shanghai_tz)
    afternoon_end = datetime.combine(date, time(15, 0)).replace(tzinfo=shanghai_tz)

    return [(morning_start, morning_end), (afternoon_start, afternoon_end)]


def is_trading_time(dt: datetime) -> bool:
    """
    Check if a given datetime is within A-share trading hours.

    Args:
        dt: Datetime to check

    Returns:
        True if within trading hours, False otherwise
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    shanghai_tz = ZoneInfo("Asia/Shanghai")

    # Handle timezone-naive datetimes by assuming they are in the target timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=shanghai_tz)
    else:
        dt = dt.astimezone(shanghai_tz)

    date = dt.date()
    market_sessions = get_a_share_market_hours(datetime.combine(date, time(0, 0)))

    for start, end in market_sessions:
        if start <= dt <= end:
            return True

    return False


def get_next_trading_time(dt: datetime, timeframe_minutes: int = 10) -> datetime:
    """
    Get the next trading time aligned to the specified timeframe.

    Args:
        dt: Current datetime
        timeframe_minutes: Timeframe in minutes (default 10)

    Returns:
        Next aligned trading time
    """
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    shanghai_tz = ZoneInfo("Asia/Shanghai")

    # Handle timezone-naive datetimes by assuming they are in the target timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=shanghai_tz)
    else:
        dt = dt.astimezone(shanghai_tz)

    # Align to the next timeframe interval
    minutes = dt.minute
    seconds = dt.second
    microseconds = dt.microsecond

    # Calculate next aligned time
    aligned_minute = ((minutes // timeframe_minutes) + 1) * timeframe_minutes

    if aligned_minute >= 60:
        # Move to next hour
        aligned_hour = dt.hour + 1
        aligned_minute = 0

        # Adjust date if we move to next day
        if aligned_hour >= 24:
            dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                days=1
            )
        else:
            dt = dt.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
    else:
        dt = dt.replace(minute=aligned_minute, second=0, microsecond=0)

    # Check if this time is within trading hours
    while not is_trading_time(dt):
        # If not in trading hours, move to next aligned time
        aligned_minute = ((dt.minute // timeframe_minutes) + 1) * timeframe_minutes

        if aligned_minute >= 60:
            # Move to next hour
            aligned_hour = dt.hour + 1
            aligned_minute = 0

            # Adjust date if we move to next day
            if aligned_hour >= 24:
                dt = dt.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
                    days=1
                )
            else:
                dt = dt.replace(
                    hour=aligned_hour, minute=aligned_minute, second=0, microsecond=0
                )
        else:
            dt = dt.replace(minute=aligned_minute, second=0, microsecond=0)

    return dt


def generate_trading_times(
    start: datetime, end: datetime, timeframe_minutes: int = 10
) -> List[datetime]:
    """
    Generate all trading times between start and end, aligned to the specified timeframe.
    Each timestamp represents the END of the bar period.

    Args:
        start: Start datetime
        end: End datetime
        timeframe_minutes: Timeframe in minutes (default 10)

    Returns:
        List of aligned trading times representing end of each bar
    """
    shanghai_tz = ZoneInfo("Asia/Shanghai")
    start = start.astimezone(shanghai_tz)
    end = end.astimezone(shanghai_tz)

    times = []
    current_time = start

    # Find the first aligned trading time
    while current_time < start:
        current_time = get_next_trading_time(current_time, timeframe_minutes)

    # Generate all aligned trading times until end
    while current_time <= end:
        if is_trading_time(current_time):
            times.append(current_time)
        current_time = get_next_trading_time(current_time, timeframe_minutes)

    return times


def align_to_trading_bar(dt: datetime, timeframe_minutes: int = 10) -> datetime:
    """
    Align a datetime to the nearest previous trading bar end time.
    This represents the end time of the bar that contains this moment.

    Args:
        dt: Datetime to align
        timeframe_minutes: Timeframe in minutes (default 10)

    Returns:
        Aligned trading bar end time
    """
    shanghai_tz = ZoneInfo("Asia/Shanghai")
    dt = dt.astimezone(shanghai_tz)

    # Get all trading times for the day
    start_of_day = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    trading_times = generate_trading_times(start_of_day, end_of_day, timeframe_minutes)

    # Find the last trading time that is <= dt
    aligned_time = None
    for time_slot in reversed(trading_times):
        if time_slot <= dt:
            aligned_time = time_slot
            break

    if aligned_time is None:
        # If we didn't find a matching time in this day, look at previous day
        prev_day_start = start_of_day - timedelta(days=1)
        prev_day_end = end_of_day - timedelta(days=1)
        prev_day_times = generate_trading_times(
            prev_day_start, prev_day_end, timeframe_minutes
        )

        for time_slot in reversed(prev_day_times):
            if time_slot <= dt:
                aligned_time = time_slot
                break

    return aligned_time if aligned_time else dt


def get_standard_timeframe_multiplier(timeframe_str: str) -> int:
    """
    Get the multiplier in minutes for standard timeframe strings.

    Args:
        timeframe_str: Standard timeframe string (e.g., '1m', '5m', '10m', '1h', '1d')

    Returns:
        Multiplier in minutes
    """
    mapping = {
        "1m": 1,
        "5m": 5,
        "10m": 10,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "4h": 240,
        "1d": 24 * 60,
        "1w": 7 * 24 * 60,
        "1mo": 30 * 24 * 60,
    }

    if timeframe_str.lower() in mapping:
        return mapping[timeframe_str.lower()]
    else:
        raise ValueError(
            f"Unknown timeframe: {timeframe_str}. Supported: {list(mapping.keys())}"
        )


def get_next_trading_bar_timestamp(
    reference_time: Union[datetime, pd.Timestamp], timeframe_minutes: int = 10
) -> Union[datetime, pd.Timestamp]:
    """
    Get the timestamp for the NEXT trading bar after the reference time.
    This is particularly useful for determining when a signal generated at reference_time
    will be executed (at the close of the NEXT bar).

    Args:
        reference_time: Time when signal was generated (can be datetime or pandas Timestamp)
        timeframe_minutes: Timeframe in minutes (default 10)

    Returns:
        Timestamp of the next trading bar's close (same type as input)
    """
    # Get all potential bar end times for the day
    try:
        from zoneinfo import ZoneInfo
    except ImportError:
        from backports.zoneinfo import ZoneInfo

    shanghai_tz = ZoneInfo("Asia/Shanghai")

    # Remember the original type and timezone info
    is_pandas_ts = isinstance(reference_time, pd.Timestamp)
    was_timezone_naive = (
        (isinstance(reference_time, pd.Timestamp) and reference_time.tz is None) or
        (isinstance(reference_time, datetime) and reference_time.tzinfo is None)
    )

    # Convert pandas timestamp to datetime if needed and handle timezone
    if isinstance(reference_time, pd.Timestamp):
        # Check if it's timezone-naive
        if reference_time.tz is None:
            # Convert to datetime and assume it's in Shanghai timezone
            ref_dt = datetime(
                reference_time.year, reference_time.month, reference_time.day,
                reference_time.hour, reference_time.minute, reference_time.second,
                reference_time.microsecond
            )
            ref_dt = ref_dt.replace(tzinfo=shanghai_tz)
        else:
            # It has timezone info, convert to Shanghai timezone
            ref_dt = reference_time.to_pydatetime().astimezone(shanghai_tz)
    else:  # It's a regular datetime object
        if reference_time.tzinfo is None:
            ref_dt = reference_time.replace(tzinfo=shanghai_tz)
        else:
            ref_dt = reference_time.astimezone(shanghai_tz)

    # Generate all trading times starting from the next potential bar
    start_time = ref_dt
    current_time = get_next_trading_time(start_time, timeframe_minutes)

    # Convert back to the same format as the input
    if was_timezone_naive:
        # Remove timezone info to return timezone-naive datetime
        result_naive = current_time.replace(tzinfo=None)
        if is_pandas_ts:
            return pd.Timestamp(result_naive)
        else:
            return result_naive
    else:
        # Return with timezone info
        if is_pandas_ts:
            return pd.Timestamp(current_time)
        else:
            return current_time
