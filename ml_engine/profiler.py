"""
AutoML Problem Solver - Dataset Profiler
Handles Step 1: Upload & auto-detect target column, problem type, and dataset analysis.
Supports CSV, Excel, JSON, Parquet, TSV formats.
Detects time series and text/NLP columns.
"""

import pandas as pd
import numpy as np
import re
import os

from ml_engine.timeseries_detector import detect_timeseries
from ml_engine.text_processor import detect_text_columns


# Common target column name patterns
TARGET_PATTERNS = [
    r'^target$', r'^label$', r'^class$', r'^y$', r'^output$',
    r'^survived$', r'^survival$', r'^outcome$', r'^result$',
    r'^price$', r'^salary$', r'^income$', r'^revenue$',
    r'^churn$', r'^fraud$', r'^default$', r'^diagnosis$',
    r'^species$', r'^category$', r'^type$', r'^grade$',
    r'^status$', r'^risk$', r'^prediction$', r'^response$',
    r'^sale', r'^cost', r'^amount', r'^value$',
    r'^rating$', r'^score$', r'^quality$',
]

# Keywords in problem statements that hint at classification
CLASSIFICATION_KEYWORDS = [
    'classify', 'classification', 'predict class', 'predict category',
    'detect', 'whether', 'yes or no', 'true or false',
    'spam', 'fraud', 'churn', 'survive', 'diagnose', 'disease',
    'positive or negative', 'pass or fail', 'win or lose',
    'binary', 'multiclass', 'multi-class', 'categorize',
    'is it', 'will it', 'does it', 'can it',
]

# Keywords hinting at regression
REGRESSION_KEYWORDS = [
    'regression', 'predict price', 'predict value', 'predict amount',
    'how much', 'how many', 'forecast', 'estimate',
    'continuous', 'numeric prediction', 'predict salary',
    'predict cost', 'predict revenue', 'predict score',
    'predict rating', 'quantity', 'predict number',
]


def read_dataset(filepath):
    """
    Read a dataset file, supporting multiple formats.
    
    Supports: CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet
    
    Returns:
        pd.DataFrame
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.csv':
        # Try to detect delimiter
        try:
            df = pd.read_csv(filepath)
        except Exception:
            df = pd.read_csv(filepath, sep=None, engine='python')
    elif ext == '.tsv':
        df = pd.read_csv(filepath, sep='\t')
    elif ext in ('.xlsx', '.xls'):
        df = pd.read_excel(filepath)
    elif ext == '.json':
        df = pd.read_json(filepath)
    elif ext == '.parquet':
        df = pd.read_parquet(filepath)
    else:
        # Default: try CSV
        df = pd.read_csv(filepath)
    
    return df


def profile_dataset(filepath, problem_statement=""):
    """
    Analyze the uploaded dataset file and return a comprehensive profile.
    
    Returns:
        dict: Dataset profile including detected target, problem type, stats,
              time series detection, text column detection
    """
    # Read the dataset (multi-format support)
    df = read_dataset(filepath)
    
    # Basic stats
    n_rows, n_cols = df.shape
    
    # Column analysis
    columns_info = []
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': round(df[col].isnull().sum() / n_rows * 100, 2),
            'unique': int(df[col].nunique()),
            'unique_pct': round(df[col].nunique() / n_rows * 100, 2),
        }
        
        if df[col].dtype in ['int64', 'float64']:
            col_info['mean'] = round(float(df[col].mean()), 2) if not df[col].isnull().all() else None
            col_info['std'] = round(float(df[col].std()), 2) if not df[col].isnull().all() else None
            col_info['min'] = round(float(df[col].min()), 2) if not df[col].isnull().all() else None
            col_info['max'] = round(float(df[col].max()), 2) if not df[col].isnull().all() else None
            col_info['is_numeric'] = True
        else:
            col_info['top_values'] = df[col].value_counts().head(5).to_dict()
            col_info['is_numeric'] = False
            
        columns_info.append(col_info)
    
    # Auto-detect target column
    target_col = _detect_target_column(df, problem_statement)
    
    # Time Series Detection
    ts_info = detect_timeseries(df, problem_statement)
    
    # Auto-detect problem type (with time series awareness)
    if ts_info.get('is_timeseries'):
        problem_type = 'forecasting'
    else:
        problem_type = _detect_problem_type(df, target_col, problem_statement)
    
    # Text Column Detection
    text_columns = detect_text_columns(df)
    
    # Class distribution (for classification)
    class_distribution = None
    if problem_type == 'classification' and target_col:
        class_distribution = df[target_col].value_counts().to_dict()
        class_distribution = {str(k): int(v) for k, v in class_distribution.items()}
    
    # Overall dataset health
    total_missing = int(df.isnull().sum().sum())
    total_cells = n_rows * n_cols
    duplicates = int(df.duplicated().sum())
    
    # Preview data (first 10 rows)
    preview = df.head(10).fillna('NaN').to_dict(orient='records')
    
    # File format
    file_ext = os.path.splitext(filepath)[1].lower()
    
    profile = {
        'n_rows': n_rows,
        'n_cols': n_cols,
        'columns': columns_info,
        'target_column': target_col,
        'problem_type': problem_type,
        'class_distribution': class_distribution,
        'total_missing': total_missing,
        'total_missing_pct': round(total_missing / total_cells * 100, 2) if total_cells > 0 else 0,
        'duplicates': duplicates,
        'duplicates_pct': round(duplicates / n_rows * 100, 2) if n_rows > 0 else 0,
        'preview': preview,
        'column_names': list(df.columns),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(exclude=[np.number]).columns),
        'file_format': file_ext,
        # New: Time Series info
        'timeseries_info': ts_info,
        'is_timeseries': ts_info.get('is_timeseries', False),
        'datetime_column': ts_info.get('datetime_column'),
        # New: Text columns
        'text_columns': text_columns,
        'has_text_columns': len(text_columns) > 0,
    }
    
    return profile


def _detect_target_column(df, problem_statement):
    """Auto-detect the most likely target column."""
    columns = list(df.columns)
    problem_lower = problem_statement.lower()
    
    # Strategy 1: Check if any column name is mentioned in the problem statement
    for col in columns:
        col_lower = col.lower().strip()
        if len(col_lower) > 1 and col_lower in problem_lower:
            return col
    
    # Strategy 2: Match against known target patterns
    for col in columns:
        col_lower = col.lower().strip()
        for pattern in TARGET_PATTERNS:
            if re.match(pattern, col_lower):
                return col
    
    # Strategy 3: Last column is often the target
    return columns[-1]


def _detect_problem_type(df, target_col, problem_statement):
    """Auto-detect whether this is a classification, regression, or multi-label problem."""
    problem_lower = problem_statement.lower()
    
    # Check for explicit multi-label keywords
    multilabel_keywords = ['multi-label', 'multilabel', 'multi label', 'tagging', 'tags', 'multiple labels']
    if any(kw in problem_lower for kw in multilabel_keywords):
        return 'multilabel'
    
    # Strategy 1: Check problem statement keywords
    clf_score = sum(1 for kw in CLASSIFICATION_KEYWORDS if kw in problem_lower)
    reg_score = sum(1 for kw in REGRESSION_KEYWORDS if kw in problem_lower)
    
    if clf_score > reg_score:
        return 'classification'
    if reg_score > clf_score:
        return 'regression'
    
    # Strategy 2: Analyze the target column
    if target_col and target_col in df.columns:
        target = df[target_col]
        n_unique = target.nunique()
        
        # Multi-label detection: check for pipe/comma-separated values
        if target.dtype == 'object':
            sample = target.dropna().head(100).astype(str)
            delimited = sample.str.contains(r'[|,;]', regex=True).mean()
            if delimited > 0.3:
                return 'multilabel'
        
        # If target is object/string type -> classification
        if target.dtype == 'object':
            return 'classification'
        
        # If target has very few unique values -> classification
        if n_unique <= 20 and n_unique / len(df) < 0.05:
            return 'classification'
        
        # If target is float with many unique values -> regression
        if target.dtype == 'float64' and n_unique > 20:
            return 'regression'
        
        # If target is int with many unique values -> regression
        if n_unique > 20:
            return 'regression'
        
        return 'classification'
    
    return 'classification'  # Default

