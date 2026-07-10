"""
Date and time utilities for stock trading.
"""

from datetime import date, datetime, timedelta
from typing import List, Optional, Union

import pandas as pd

from src.core.exceptions import ValidationError

# Type aliases
DateInput = Union[str, datetime, date, pd.Timestamp]
DateOutput = datetime


def parse_date(
    date_input: DateInput,
    format: Optional[str] = None,
) -> DateOutput:
    """
    Parse date from various input types.

    Args:
        date_input: Date string, datetime, date, or Timestamp
        format: Optional format string for parsing

    Returns:
        Parsed datetime object

    Raises:
        ValidationError: If parsing fails
    """
    if date_input is None:
        raise ValidationError("Date input is None")

    # Already a datetime
    if isinstance(date_input, datetime):
        return date_input

    # Date object
    if isinstance(date_input, date):
        return datetime.combine(date_input, datetime.min.time())

    # Pandas Timestamp
    if isinstance(date_input, pd.Timestamp):
        return date_input.to_pydatetime()

    # String input
    if isinstance(date_input, str):
        date_input = date_input.strip()

        # Try common formats
        formats_to_try = [
            "%Y-%m-%d",
            "%Y%m%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
        ]

        if format:
            formats_to_try.insert(0, format)

        for fmt in formats_to_try:
            try:
                return datetime.strptime(date_input, fmt)
            except ValueError:
                continue

        raise ValidationError(f"Unable to parse date: {date_input}")

    raise ValidationError(f"Unsupported date type: {type(date_input)}")


def format_date(
    date_input: DateInput,
    format: str = "%Y-%m-%d",
) -> str:
    """
    Format date to string.

    Args:
        date_input: Date to format
        format: Output format string

    Returns:
        Formatted date string
    """
    dt = parse_date(date_input)
    return dt.strftime(format)


def get_yesterday() -> datetime:
    """Get yesterday's date."""
    return datetime.now() - timedelta(days=1)


def get_today() -> datetime:
    """Get today's date."""
    return datetime.now()


def get_trading_days(
    start_date: Optional[DateInput] = None,
    end_date: Optional[DateInput] = None,
    n_days: Optional[int] = None,
) -> List[datetime]:
    """
    Get list of trading days (weekdays only).

    Args:
        start_date: Start date (default: today)
        end_date: End date (default: start + n_days)
        n_days: Number of days to return

    Returns:
        List of trading day datetimes
    """
    if start_date is None:
        start_date = get_today()
    else:
        start_date = parse_date(start_date)

    if n_days:
        end_date = start_date + timedelta(days=n_days * 2)  # Buffer for weekends
    elif end_date:
        end_date = parse_date(end_date)
    else:
        end_date = start_date + timedelta(days=30)

    # Generate date range and filter weekdays
    trading_days = []
    current = start_date
    while current <= end_date:
        # Monday=0, Friday=4
        if current.weekday() < 5:
            trading_days.append(current)
            if n_days and len(trading_days) >= n_days:
                break
        current += timedelta(days=1)

    return trading_days


def is_trading_day(date_input: DateInput) -> bool:
    """
    Check if a date is a trading day (weekday).

    Args:
        date_input: Date to check

    Returns:
        True if weekday, False if weekend
    """
    dt = parse_date(date_input)
    return dt.weekday() < 5  # Monday=0, Friday=4


def add_trading_days(
    date_input: DateInput,
    n_days: int,
) -> datetime:
    """
    Add N trading days to a date (skipping weekends).

    Args:
        date_input: Starting date
        n_days: Number of trading days to add (can be negative)

    Returns:
        Result date
    """
    start = parse_date(date_input)
    result = start
    days_added = 0
    direction = 1 if n_days >= 0 else -1

    while days_added < abs(n_days):
        result += timedelta(days=direction)
        if is_trading_day(result):
            days_added += 1

    return result


def get_date_range(
    start_date: DateInput,
    end_date: DateInput,
    freq: str = "D",
) -> List[datetime]:
    """
    Get list of dates in range.

    Args:
        start_date: Start date
        end_date: End date
        freq: Frequency (D=daily, W=weekly, M=monthly)

    Returns:
        List of datetimes
    """
    start = parse_date(start_date)
    end = parse_date(end_date)

    if freq == "D":
        current = start
        dates = []
        while current <= end:
            dates.append(current)
            current += timedelta(days=1)
        return dates

    # Use pandas for other frequencies
    date_range = pd.date_range(start=start, end=end, freq=freq)
    return [dt.to_pydatetime() for dt in date_range]


def get_quarter(date_input: DateInput) -> int:
    """
    Get quarter (1-4) for a date.

    Args:
        date_input: Input date

    Returns:
        Quarter number (1-4)
    """
    dt = parse_date(date_input)
    return (dt.month - 1) // 3 + 1


def get_quarter_start(date_input: DateInput) -> datetime:
    """
    Get start of quarter for a date.

    Args:
        date_input: Input date

    Returns:
        First day of quarter
    """
    dt = parse_date(date_input)
    quarter = get_quarter(dt)
    month = (quarter - 1) * 3 + 1
    return datetime(dt.year, month, 1)


def get_quarter_end(date_input: DateInput) -> datetime:
    """
    Get end of quarter for a date.

    Args:
        date_input: Input date

    Returns:
        Last day of quarter
    """
    dt = parse_date(date_input)
    quarter = get_quarter(dt)
    month = quarter * 3
    if month == 12:
        return datetime(dt.year, 12, 31)
    else:
        return datetime(dt.year, month + 1, 1) - timedelta(days=1)


def age_days(date_input: DateInput) -> int:
    """
    Get age in days from a past date to now.

    Args:
        date_input: Past date

    Returns:
        Number of days
    """
    past = parse_date(date_input)
    now = datetime.now()
    return (now - past).days


def format_timedelta_days(days: int) -> str:
    """
    Format days into human-readable string.

    Args:
        days: Number of days

    Returns:
        Formatted string (e.g., "30 days", "3 months")
    """
    if days < 7:
        return f"{days} days"
    elif days < 30:
        weeks = days // 7
        return f"{weeks} week{'s' if weeks > 1 else ''}"
    elif days < 365:
        months = days // 30
        return f"{months} month{'s' if months > 1 else ''}"
    else:
        years = days // 365
        return f"{years} year{'s' if years > 1 else ''}"
