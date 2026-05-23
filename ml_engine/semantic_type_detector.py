"""
AutoML Studio — Semantic Feature Type Detector (Feature #22)
Detects the SEMANTIC meaning of each column beyond Python dtype.
Identifies IDs, ordinals, cyclicals, lat/lon, currency, booleans, etc.
"""

import re
import numpy as np
import pandas as pd


def detect_semantic_types(df, target_col=None):
    """
    Detect semantic type for every column.

    Returns:
        dict with per-column semantic type and recommended encoding
    """
    results = []

    for col in df.columns:
        series = df[col]
        dtype = str(series.dtype)
        n_unique = series.nunique()
        n_rows = len(series)
        non_null = series.dropna()

        semantic_type = 'unknown'
        confidence = 0.0
        encoding = 'keep_as_is'
        action = None
        col_lower = col.lower().strip().replace(' ', '_')

        # ── ID column (unique, sequential)
        if _is_id_column(col_lower, series, n_unique, n_rows):
            semantic_type = 'id'
            confidence = 0.9
            encoding = 'drop'
            action = 'Auto-drop before training (no predictive value)'

        # ── Boolean
        elif _is_boolean(series, n_unique):
            semantic_type = 'boolean'
            confidence = 0.95
            encoding = 'binary_encode'
            action = 'Map to 0/1'

        # ── Ordinal
        elif _is_ordinal(col_lower, series, n_unique):
            semantic_type = 'ordinal'
            confidence = 0.8
            encoding = 'ordinal_encode'
            action = 'Ordinal encode (preserve order)'

        # ── Cyclical (month, day, hour)
        elif _is_cyclical(col_lower, series, n_unique):
            semantic_type = 'cyclical'
            confidence = 0.85
            encoding = 'sin_cos_encode'
            action = 'Apply sin/cos encoding to preserve cyclical nature'

        # ── Date string
        elif _is_date_string(series):
            semantic_type = 'date_string'
            confidence = 0.9
            encoding = 'parse_datetime'
            action = 'Parse to datetime → extract year, month, day, dayofweek features'

        # ── Currency
        elif _is_currency(series):
            semantic_type = 'currency'
            confidence = 0.85
            encoding = 'strip_to_float'
            action = 'Strip currency symbols ($, €, £) and convert to float'

        # ── Percentage
        elif _is_percentage(series):
            semantic_type = 'percentage'
            confidence = 0.85
            encoding = 'strip_to_decimal'
            action = 'Strip % symbol and divide by 100'

        # ── Latitude/Longitude
        elif _is_geo(col_lower, series):
            semantic_type = 'geo_coordinate'
            confidence = 0.8
            encoding = 'keep_numeric'
            action = 'Use as-is or create distance features'

        # ── Free text vs categorical
        elif series.dtype == 'object':
            avg_len = non_null.astype(str).str.len().mean() if len(non_null) > 0 else 0
            avg_words = non_null.astype(str).str.split().str.len().mean() if len(non_null) > 0 else 0
            if avg_words > 5:
                semantic_type = 'free_text'
                confidence = 0.8
                encoding = 'nlp_tfidf'
                action = 'Apply TF-IDF or sentence embeddings'
            elif n_unique <= 20:
                semantic_type = 'low_cardinality_categorical'
                confidence = 0.9
                encoding = 'one_hot_encode'
                action = 'One-hot encode'
            else:
                semantic_type = 'high_cardinality_categorical'
                confidence = 0.8
                encoding = 'frequency_encode'
                action = 'Frequency or target encode'

        # ── Numeric defaults
        elif pd.api.types.is_numeric_dtype(series):
            if n_unique == 2:
                semantic_type = 'binary_numeric'
                confidence = 0.9
                encoding = 'keep_as_is'
                action = 'Already binary — use as-is'
            elif n_unique <= 10:
                semantic_type = 'discrete_numeric'
                confidence = 0.7
                encoding = 'consider_categorical'
                action = 'Consider treating as categorical (few unique values)'
            else:
                semantic_type = 'continuous_numeric'
                confidence = 0.9
                encoding = 'scale'
                action = 'Standard scaling'

        is_target = col == target_col
        results.append({
            'column': col,
            'python_dtype': dtype,
            'semantic_type': semantic_type,
            'confidence': round(confidence, 2),
            'recommended_encoding': encoding,
            'action': action,
            'n_unique': n_unique,
            'is_target': is_target,
            'icon': _get_icon(semantic_type),
        })

    # Summary
    type_counts = {}
    for r in results:
        t = r['semantic_type']
        type_counts[t] = type_counts.get(t, 0) + 1

    auto_actions = [r for r in results if r['action'] and r['encoding'] != 'keep_as_is'
                    and not r['is_target']]

    return {
        'columns': results,
        'type_counts': type_counts,
        'actionable_count': len(auto_actions),
        'auto_drop': [r['column'] for r in results if r['encoding'] == 'drop'],
        'recommendation': (
            f'{len(auto_actions)} columns need special encoding. '
            f'{len([r for r in results if r["encoding"] == "drop"])} ID columns should be dropped.'
        ),
    }


def _is_id_column(col_lower, series, n_unique, n_rows):
    id_hints = ['id', 'index', 'row_id', 'record_id', 'serial', 'uuid', 'key']
    if any(col_lower == h or col_lower.endswith('_id') for h in id_hints):
        return True
    if n_unique == n_rows and pd.api.types.is_numeric_dtype(series):
        diffs = series.dropna().diff().dropna()
        if len(diffs) > 0 and (diffs == 1).mean() > 0.9:
            return True
    return n_unique == n_rows and n_unique > 100


def _is_boolean(series, n_unique):
    if n_unique != 2:
        return False
    vals = set(series.dropna().unique())
    bool_sets = [
        {0, 1}, {'yes', 'no'}, {'true', 'false'}, {'y', 'n'},
        {'t', 'f'}, {'male', 'female'}, {'m', 'f'},
    ]
    vals_lower = {str(v).lower().strip() for v in vals}
    return any(vals_lower == bs for bs in bool_sets)


def _is_ordinal(col_lower, series, n_unique):
    ordinal_hints = ['rating', 'rank', 'level', 'grade', 'priority', 'severity',
                     'satisfaction', 'experience', 'quality']
    if any(h in col_lower for h in ordinal_hints):
        return True
    if series.dtype == 'object' and n_unique <= 10:
        vals = {str(v).lower().strip() for v in series.dropna().unique()}
        ordinal_vals = {'low', 'medium', 'high', 'very high', 'very low',
                        'poor', 'fair', 'good', 'excellent'}
        return len(vals & ordinal_vals) >= 2
    return False


def _is_cyclical(col_lower, series, n_unique):
    cyclic_hints = ['month', 'day_of_week', 'dayofweek', 'hour', 'minute',
                    'weekday', 'quarter', 'season']
    if any(h in col_lower for h in cyclic_hints):
        return True
    if pd.api.types.is_numeric_dtype(series) and n_unique <= 31:
        vals = series.dropna().unique()
        if set(vals).issubset(set(range(0, 32))):
            return True
    return False


def _is_date_string(series):
    if series.dtype != 'object':
        return False
    sample = series.dropna().head(20).astype(str)
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}',
    ]
    matches = sum(1 for v in sample if any(re.match(p, v.strip()) for p in date_patterns))
    return matches / max(len(sample), 1) > 0.5


def _is_currency(series):
    if series.dtype != 'object':
        return False
    sample = series.dropna().head(20).astype(str)
    currency_pattern = r'^[\$€£₹¥]\s?[\d,]+\.?\d*$'
    matches = sum(1 for v in sample if re.match(currency_pattern, v.strip()))
    return matches / max(len(sample), 1) > 0.5


def _is_percentage(series):
    if series.dtype != 'object':
        return False
    sample = series.dropna().head(20).astype(str)
    pct_pattern = r'^[\d.]+\s?%$'
    matches = sum(1 for v in sample if re.match(pct_pattern, v.strip()))
    return matches / max(len(sample), 1) > 0.5


def _is_geo(col_lower, series):
    geo_hints = ['latitude', 'longitude', 'lat', 'lon', 'lng', 'long']
    if any(h == col_lower for h in geo_hints):
        return pd.api.types.is_numeric_dtype(series)
    return False


def _get_icon(semantic_type):
    icons = {
        'id': '🔑', 'boolean': '✅', 'ordinal': '📶', 'cyclical': '🔄',
        'date_string': '📅', 'currency': '💰', 'percentage': '📊',
        'geo_coordinate': '🌍', 'free_text': '📝',
        'low_cardinality_categorical': '🏷️', 'high_cardinality_categorical': '🔤',
        'binary_numeric': '⚡', 'discrete_numeric': '🔢',
        'continuous_numeric': '📈',
    }
    return icons.get(semantic_type, '❓')
