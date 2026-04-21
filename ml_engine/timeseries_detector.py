"""
AutoML Problem Solver - Time Series Detector
Auto-detects time series data and extracts temporal metadata.
"""

import pandas as pd
import numpy as np
import re


def detect_timeseries(df, problem_statement=""):
    """
    Analyze the dataframe and determine if it's time series data.
    
    Returns:
        dict: {
            'is_timeseries': bool,
            'datetime_column': str or None,
            'frequency': str,
            'seasonality': dict,
            'trend': str,
            'metadata': dict
        }
    """
    result = {
        'is_timeseries': False,
        'datetime_column': None,
        'frequency': None,
        'seasonality': None,
        'trend': None,
        'metadata': {},
    }
    
    # Check problem statement for time series keywords
    ts_keywords = [
        'forecast', 'time series', 'timeseries', 'temporal', 'predict future',
        'trend', 'seasonality', 'date', 'monthly', 'daily', 'weekly',
        'yearly', 'quarterly', 'hourly', 'stock price', 'sales forecast',
        'demand', 'time-series',
    ]
    
    problem_lower = problem_statement.lower()
    keyword_match = any(kw in problem_lower for kw in ts_keywords)
    
    # Strategy 1: Find datetime columns
    datetime_col = _find_datetime_column(df)
    
    if datetime_col is None and not keyword_match:
        return result
    
    if datetime_col:
        result['datetime_column'] = datetime_col
        result['is_timeseries'] = True
        
        # Parse datetime
        dt_series = pd.to_datetime(df[datetime_col], errors='coerce')
        valid = dt_series.dropna()
        
        if len(valid) > 1:
            # Detect frequency
            result['frequency'] = _detect_frequency(valid)
            
            # Check for trend
            result['trend'] = _detect_trend(df, datetime_col)
            
            # Metadata
            result['metadata'] = {
                'start_date': str(valid.min()),
                'end_date': str(valid.max()),
                'n_periods': len(valid),
                'date_range_days': (valid.max() - valid.min()).days,
            }
    elif keyword_match:
        # Problem statement suggests time series but no datetime column found
        # Check for sequential numeric index
        if _is_sequential_index(df):
            result['is_timeseries'] = True
            result['metadata'] = {'note': 'No datetime column found, using index as time'}
    
    return result


def extract_temporal_features(df, datetime_col):
    """
    Extract temporal features from a datetime column.
    
    Returns:
        DataFrame with added temporal feature columns
    """
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Basic temporal features
    df['_year'] = dt.dt.year
    df['_month'] = dt.dt.month
    df['_day'] = dt.dt.day
    df['_dayofweek'] = dt.dt.dayofweek
    df['_dayofyear'] = dt.dt.dayofyear
    df['_quarter'] = dt.dt.quarter
    df['_weekofyear'] = dt.dt.isocalendar().week.astype(int)
    df['_hour'] = dt.dt.hour
    
    # Cyclical encoding (sin/cos for periodic features)
    df['_month_sin'] = np.sin(2 * np.pi * df['_month'] / 12)
    df['_month_cos'] = np.cos(2 * np.pi * df['_month'] / 12)
    df['_dow_sin'] = np.sin(2 * np.pi * df['_dayofweek'] / 7)
    df['_dow_cos'] = np.cos(2 * np.pi * df['_dayofweek'] / 7)
    
    # Is weekend
    df['_is_weekend'] = (df['_dayofweek'] >= 5).astype(int)
    
    # Remove zero-variance temporal columns
    temporal_cols = [c for c in df.columns if c.startswith('_')]
    for col in temporal_cols:
        if df[col].nunique() <= 1:
            df.drop(columns=[col], inplace=True)
    
    return df


def create_lag_features(df, target_col, lags=None, rolling_windows=None):
    """
    Create lag and rolling window features for time series.
    
    Returns:
        DataFrame with lag/rolling features added
    """
    df = df.copy()
    
    if lags is None:
        lags = [1, 2, 3, 7, 14, 30]
    if rolling_windows is None:
        rolling_windows = [3, 7, 14, 30]
    
    # Only use lags that make sense for data size
    max_lag = min(max(lags), len(df) // 4)
    lags = [l for l in lags if l <= max_lag and l > 0]
    
    # Lag features
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    # Rolling statistics
    max_window = min(max(rolling_windows), len(df) // 3)
    rolling_windows = [w for w in rolling_windows if w <= max_window and w > 1]
    
    for window in rolling_windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
    
    # Diff features
    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    if len(df) > 7:
        df[f'{target_col}_diff_7'] = df[target_col].diff(7)
    
    # Drop rows with NaN from lag features
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df


def _find_datetime_column(df):
    """Find the most likely datetime column."""
    # Check existing datetime columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
    
    # Check column names
    date_patterns = [
        r'date', r'time', r'timestamp', r'datetime', r'dt',
        r'created', r'updated', r'period', r'year.*month',
    ]
    
    for col in df.columns:
        col_lower = col.lower().strip()
        for pattern in date_patterns:
            if re.search(pattern, col_lower):
                # Verify it can be parsed as datetime
                try:
                    parsed = pd.to_datetime(df[col], errors='coerce')
                    valid_ratio = parsed.notna().sum() / len(df)
                    if valid_ratio > 0.7:
                        return col
                except Exception:
                    pass
    
    # Try parsing object columns
    for col in df.select_dtypes(include=['object']).columns:
        try:
            parsed = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
            valid_ratio = parsed.notna().sum() / len(df)
            if valid_ratio > 0.8:
                return col
        except Exception:
            pass
    
    return None


def _detect_frequency(dt_series):
    """Detect the frequency of a datetime series."""
    if len(dt_series) < 2:
        return 'unknown'
    
    sorted_dt = dt_series.sort_values()
    diffs = sorted_dt.diff().dropna()
    median_diff = diffs.median()
    
    days = median_diff.days
    seconds = median_diff.total_seconds()
    
    if seconds < 3600:
        return 'minutely'
    elif seconds < 86400:
        return 'hourly'
    elif days <= 1:
        return 'daily'
    elif 5 <= days <= 9:
        return 'weekly'
    elif 25 <= days <= 35:
        return 'monthly'
    elif 85 <= days <= 95:
        return 'quarterly'
    elif 350 <= days <= 380:
        return 'yearly'
    else:
        return f'{days}-day intervals'


def _detect_trend(df, datetime_col):
    """Simple trend detection."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return 'unknown'
    
    # Use the last numeric column as proxy target
    target = df[numeric_cols[-1]].values
    n = len(target)
    if n < 10:
        return 'unknown'
    
    # Compare first and last quarters
    q1_mean = np.mean(target[:n // 4])
    q4_mean = np.mean(target[-n // 4:])
    
    change = (q4_mean - q1_mean) / max(abs(q1_mean), 1e-10)
    
    if change > 0.1:
        return 'upward'
    elif change < -0.1:
        return 'downward'
    else:
        return 'stable'


def _is_sequential_index(df):
    """Check if data has a sequential numeric index suggesting time ordering."""
    if df.index.is_monotonic_increasing and len(df) > 50:
        return True
    
    # Check first numeric column for monotonic sequence
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if df[col].is_monotonic_increasing and df[col].nunique() == len(df):
                return True
            break  # Only check first numeric
    
    return False
