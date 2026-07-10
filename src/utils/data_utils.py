"""
Data processing utilities.
"""

from typing import List, Optional, Union

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.core.exceptions import ValidationError


def clean_dataframe(
    df: pd.DataFrame,
    drop_na: bool = True,
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Clean and prepare a dataframe.

    Args:
        df: Input dataframe
        drop_na: Whether to drop rows with NaN values
        sort_by: Column name to sort by
        ascending: Sort ascending if True

    Returns:
        Cleaned dataframe

    Raises:
        ValidationError: If dataframe is invalid
    """
    if df is None or df.empty:
        raise ValidationError("Dataframe is empty or None")

    # Make a copy to avoid modifying original
    result = df.copy()

    # Drop NaN rows
    if drop_na:
        result = result.dropna()

    if result.empty:
        raise ValidationError("Dataframe is empty after dropping NaN rows")

    # Sort if requested
    if sort_by and sort_by in result.columns:
        result = result.sort_values(by=sort_by, ascending=ascending)
        result = result.reset_index(drop=True)

    return result


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 1,
) -> bool:
    """
    Validate dataframe has required structure.

    Args:
        df: Dataframe to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required

    Returns:
        True if valid

    Raises:
        ValidationError: If validation fails
    """
    if df is None or df.empty:
        raise ValidationError("Dataframe is None or empty")

    if len(df) < min_rows:
        raise ValidationError(
            f"Dataframe has {len(df)} rows, required at least {min_rows}"
        )

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValidationError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    return True


def merge_dataframes(
    dfs: List[pd.DataFrame],
    on: Union[str, List[str]],
    how: str = "inner",
) -> pd.DataFrame:
    """
    Merge multiple dataframes.

    Args:
        dfs: List of dataframes to merge
        on: Column(s) to merge on
        how: Merge type (inner, outer, left, right)

    Returns:
        Merged dataframe

    Raises:
        ValidationError: If merge fails
    """
    if not dfs or len(dfs) < 2:
        raise ValidationError("Need at least 2 dataframes to merge")

    try:
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on=on, how=how)
        return result
    except Exception as e:
        raise ValidationError(f"Failed to merge dataframes: {e}")


def resample_data(
    df: pd.DataFrame,
    date_column: str,
    freq: str = "D",
    agg_dict: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Resample time series data.

    Args:
        df: Input dataframe
        date_column: Name of date column
        freq: Resampling frequency (D, W, M, etc.)
        agg_dict: Aggregation dict for different columns

    Returns:
        Resampled dataframe
    """
    # Ensure date column is datetime
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])

    # Set date as index
    df = df.set_index(date_column)

    # Resample
    if agg_dict:
        resampled = df.resample(freq).agg(agg_dict)
    else:
        resampled = df.resample(freq).mean()

    # Reset index
    resampled = resampled.reset_index()

    return resampled


def normalize_column(
    df: pd.DataFrame,
    column: str,
    method: str = "minmax",
) -> pd.DataFrame:
    """
    Normalize a column in the dataframe.

    Args:
        df: Input dataframe
        column: Column to normalize
        method: Normalization method (minmax, zscore)

    Returns:
        Dataframe with normalized column
    """
    result = df.copy()

    if method == "minmax":
        min_val = result[column].min()
        max_val = result[column].max()
        if max_val - min_val > 0:
            result[f"{column}_norm"] = (result[column] - min_val) / (max_val - min_val)
        else:
            result[f"{column}_norm"] = 0.0

    elif method == "zscore":
        mean = result[column].mean()
        std = result[column].std()
        if std > 0:
            result[f"{column}_norm"] = (result[column] - mean) / std
        else:
            result[f"{column}_norm"] = 0.0

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return result


def clip_outliers(
    df: pd.DataFrame,
    column: str,
    method: str = "iqr",
    factor: float = 1.5,
) -> pd.DataFrame:
    """
    Clip outliers in a column.

    Args:
        df: Input dataframe
        column: Column to process
        method: Method (iqr, zscore)
        factor: Factor for outlier detection

    Returns:
        Dataframe with clipped outliers
    """
    result = df.copy()

    if method == "iqr":
        Q1 = result[column].quantile(0.25)
        Q3 = result[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        result[column] = result[column].clip(lower, upper)

    elif method == "zscore":
        mean = result[column].mean()
        std = result[column].std()
        zscore = (result[column] - mean) / std
        result.loc[abs(zscore) > factor, column] = np.nan

    return result


def calculate_returns(
    df: pd.DataFrame,
    price_column: str = "close",
    periods: int = 1,
) -> pd.Series:
    """
    Calculate returns from price column.

    Args:
        df: Input dataframe
        price_column: Name of price column
        periods: Number of periods for return calculation

    Returns:
        Series of returns
    """
    return df[price_column].pct_change(periods)


def calculate_volatility(
    df: pd.DataFrame,
    return_column: Optional[str] = None,
    price_column: str = "close",
    window: int = 20,
) -> pd.Series:
    """
    Calculate rolling volatility.

    Args:
        df: Input dataframe
        return_column: Name of return column (calculated if None)
        price_column: Price column for return calculation
        window: Rolling window size

    Returns:
        Series of volatility
    """
    if return_column is None:
        returns = calculate_returns(df, price_column)
    else:
        returns = df[return_column]

    return returns.rolling(window).std()
