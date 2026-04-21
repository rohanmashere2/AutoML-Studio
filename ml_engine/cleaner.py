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
    """Detect and cap outliers using IQR method for numeric columns."""
    capped_cols = []
    outlier_counts = {}
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Don't cap outliers in the target column
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        if IQR == 0:
            continue
            
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        
        if n_outliers > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            capped_cols.append(col)
            outlier_counts[col] = n_outliers
    
    total_outliers = sum(outlier_counts.values())
    
    step = {
        'name': 'Handle Outliers',
        'icon': '📊',
        'description': f'Capped {total_outliers} outliers in {len(capped_cols)} columns (IQR method)' if capped_cols else 'No significant outliers found',
        'count': total_outliers,
        'outlier_counts': outlier_counts,
        'applied': len(capped_cols) > 0,
    }
    return df, step
