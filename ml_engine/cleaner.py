"""
AutoML Problem Solver - Data Cleaner
Handles missing values, duplicates, outliers, and data type corrections.
"""

import pandas as pd
import numpy as np


def clean_dataset(df, profile):
    """
    Automatically clean the dataset based on the profile.
    
    Returns:
        tuple: (cleaned_df, cleaning_report)
    """
    report = {
        'steps': [],
        'summary': {},
    }
    original_shape = df.shape
    
    # Step 1: Remove duplicates
    df, step = _remove_duplicates(df)
    report['steps'].append(step)
    
    # Step 2: Fix data types
    df, step = _fix_data_types(df, profile)
    report['steps'].append(step)
    
    # Step 3: Drop columns with too many missing values
    df, step = _drop_high_missing_columns(df, threshold=0.7)
    report['steps'].append(step)
    
    # Step 4: Handle missing values
    df, step = _handle_missing_values(df, profile)
    report['steps'].append(step)
    
    # Step 5: Handle outliers
    target_col = profile.get('target_column')
    df, step = _handle_outliers(df, target_col)
    report['steps'].append(step)
    
    # Summary
    report['summary'] = {
        'original_rows': original_shape[0],
        'original_cols': original_shape[1],
        'cleaned_rows': df.shape[0],
        'cleaned_cols': df.shape[1],
        'rows_removed': original_shape[0] - df.shape[0],
        'cols_removed': original_shape[1] - df.shape[1],
    }
    
    return df, report


def _remove_duplicates(df):
    """Remove duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    
    step = {
        'name': 'Remove Duplicates',
        'icon': '🔄',
        'description': f'Removed {n_removed} duplicate rows' if n_removed > 0 else 'No duplicates found',
        'count': n_removed,
        'applied': n_removed > 0,
    }
    return df, step


def _fix_data_types(df, profile):
    """Fix columns that have incorrect data types."""
    fixed_columns = []
    
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try converting to numeric
            try:
                converted = pd.to_numeric(df[col].str.strip(), errors='coerce')
                # If more than 70% can be converted, it's likely numeric
                valid_ratio = converted.notna().sum() / max(len(converted), 1)
                if valid_ratio > 0.7:
                    df[col] = converted
                    fixed_columns.append(col)
            except (AttributeError, TypeError):
                pass
    
    step = {
        'name': 'Fix Data Types',
        'icon': '🔧',
        'description': f'Corrected {len(fixed_columns)} columns: {", ".join(fixed_columns)}' if fixed_columns else 'All data types correct',
        'count': len(fixed_columns),
        'columns': fixed_columns,
        'applied': len(fixed_columns) > 0,
    }
    return df, step


def _drop_high_missing_columns(df, threshold=0.7):
    """Drop columns with more than threshold % missing values."""
    missing_pct = df.isnull().sum() / len(df)
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
    
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    
    step = {
        'name': 'Drop High-Missing Columns',
        'icon': '🗑️',
        'description': f'Dropped {len(cols_to_drop)} columns (>70% missing): {", ".join(cols_to_drop)}' if cols_to_drop else 'No columns dropped',
        'count': len(cols_to_drop),
        'columns': cols_to_drop,
        'applied': len(cols_to_drop) > 0,
    }
    return df, step


def _handle_missing_values(df, profile):
    """Fill missing values using appropriate strategies."""
    filled_cols = []
    strategies = {}
    
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                # Numeric: use median (robust to outliers)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                strategies[col] = f'median ({round(median_val, 2)})'
            else:
                # Categorical: use mode
                if not df[col].mode().empty:
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    strategies[col] = f'mode ({mode_val})'
                else:
                    df[col] = df[col].fillna('Unknown')
                    strategies[col] = 'filled with Unknown'
            filled_cols.append(col)
    
    step = {
        'name': 'Handle Missing Values',
        'icon': '🩹',
        'description': f'Filled missing values in {len(filled_cols)} columns' if filled_cols else 'No missing values found',
        'count': len(filled_cols),
        'strategies': strategies,
        'applied': len(filled_cols) > 0,
    }
    return df, step


def _handle_outliers(df, target_col=None):
    """Detect and handle outliers using smart strategies per column.
    
    Improvements over naive IQR:
    - Skips ID-like columns (monotonic, high cardinality)
    - Skips columns with <2% outliers (they're fine)
    - Uses Winsorization (1st/99th percentile) for skewed distributions
    - Uses IQR capping for normally distributed features
    """
    capped_cols = []
    outlier_counts = {}
    strategies_used = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Don't handle outliers in the target column
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    n_rows = len(df)
    
    for col in numeric_cols:
        # Skip ID-like columns (monotonically increasing, very high cardinality)
        if df[col].nunique() > 0.9 * n_rows and n_rows > 50:
            continue
        # Skip if column is monotonically increasing/decreasing (likely an index)
        if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
            continue
        # Skip binary/low-cardinality columns
        if df[col].nunique() <= 5:
            continue
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue
        
        lower_iqr = Q1 - 1.5 * IQR
        upper_iqr = Q3 + 1.5 * IQR
        
        n_outliers = int(((df[col] < lower_iqr) | (df[col] > upper_iqr)).sum())
        outlier_pct = n_outliers / max(n_rows, 1)
        
        # Skip columns with very few outliers (<2%)
        if outlier_pct < 0.02:
            continue
        
        # Choose strategy based on distribution skewness
        skewness = abs(df[col].skew())
        
        if skewness > 2:
            # Highly skewed: use Winsorization (1st/99th percentile) — gentler
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower, upper=upper)
            n_capped = n_outliers  # approximate
            strategies_used[col] = 'winsorize_1_99'
        else:
            # Normal-ish: use IQR capping
            df[col] = df[col].clip(lower=lower_iqr, upper=upper_iqr)
            n_capped = n_outliers
            strategies_used[col] = 'iqr_cap'
        
        capped_cols.append(col)
        outlier_counts[col] = n_capped
    
    total_outliers = sum(outlier_counts.values())
    
    # Build description
    if capped_cols:
        winsorized = [c for c, s in strategies_used.items() if s == 'winsorize_1_99']
        iqr_capped = [c for c, s in strategies_used.items() if s == 'iqr_cap']
        desc_parts = []
        if iqr_capped:
            desc_parts.append(f'IQR-capped {len(iqr_capped)} columns')
        if winsorized:
            desc_parts.append(f'Winsorized {len(winsorized)} skewed columns')
        description = f'Handled {total_outliers} outliers: {", ".join(desc_parts)}'
    else:
        description = 'No significant outliers found (skipped ID-like and low-outlier columns)'
    
    step = {
        'name': 'Handle Outliers',
        'icon': '📊',
        'description': description,
        'count': total_outliers,
        'outlier_counts': outlier_counts,
        'strategies': strategies_used,
        'applied': len(capped_cols) > 0,
    }
    return df, step

