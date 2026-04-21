"""
AutoML Studio — Problem Reframer
Suggests alternative problem framings based on data characteristics.
"""

import numpy as np
import pandas as pd


def suggest_reframings(df, target_column, current_problem_type):
    """
    Analyze data and suggest alternative problem framings.

    Returns:
        list of suggestion dicts with type, title, rationale, estimated_benefit
    """
    suggestions = []

    if target_column not in df.columns:
        return suggestions

    target = df[target_column]

    # 1. Binary classification → Time-series forecasting
    if current_problem_type == 'classification':
        if _has_temporal_pattern(df):
            suggestions.append({
                'type': 'temporal_reframing',
                'icon': '📅',
                'title': 'Consider Time-Series Forecasting',
                'rationale': (
                    'We detected temporal features (dates, timestamps, or monotonically increasing columns) '
                    'in your dataset. If the order of records matters, framing this as a forecasting problem '
                    'may capture trends and seasonality that static models miss.'
                ),
                'estimated_benefit': 'May capture temporal trends that static classification models miss.',
                'difficulty': 'medium',
            })

    # 2. Multiclass classification → Ordinal regression
    if current_problem_type == 'classification' and target.nunique() > 3:
        if _is_ordered(target):
            suggestions.append({
                'type': 'ordinal_regression',
                'icon': '📊',
                'title': 'Try Ordinal Regression',
                'rationale': (
                    f'Your target has {target.nunique()} classes that appear to have a natural ordering. '
                    f'Ordinal regression respects this ordering and often outperforms standard classification '
                    f'because it learns that class 3 is closer to class 2 than to class 0.'
                ),
                'estimated_benefit': 'Better handling of ordered categories, fewer misclassification errors between adjacent classes.',
                'difficulty': 'low',
            })

    # 3. Regression → Classification (binning)
    if current_problem_type == 'regression':
        n_unique = target.nunique()
        if n_unique < 10:
            suggestions.append({
                'type': 'treat_as_classification',
                'icon': '🎯',
                'title': 'Treat as Classification',
                'rationale': (
                    f'Your target has only {n_unique} unique values. Even though they\'re numeric, '
                    f'treating this as a classification problem may yield better results since '
                    f'the model can learn distinct decision boundaries for each value.'
                ),
                'estimated_benefit': 'Better accuracy when target has very few distinct values.',
                'difficulty': 'low',
            })
        elif _is_heavily_skewed(target):
            suggestions.append({
                'type': 'binned_classification',
                'icon': '📦',
                'title': 'Bin Target into Categories',
                'rationale': (
                    f'Your regression target is heavily skewed (skew={target.skew():.2f}). '
                    f'Consider binning it into categories (e.g., low/medium/high) and treating '
                    f'it as a classification problem. This avoids the impact of extreme values '
                    f'on model training.'
                ),
                'suggested_bins': _suggest_bins(target),
                'estimated_benefit': 'Reduced impact of outliers and extreme values.',
                'difficulty': 'low',
            })

    # 4. Binary classification → Anomaly detection
    if current_problem_type == 'classification' and target.nunique() == 2:
        vc = target.value_counts(normalize=True)
        minority_pct = vc.min()
        if minority_pct < 0.05:
            suggestions.append({
                'type': 'anomaly_detection',
                'icon': '🔍',
                'title': 'Try Anomaly Detection',
                'rationale': (
                    f'Your positive class represents only {minority_pct*100:.1f}% of the data. '
                    f'With such extreme imbalance, anomaly detection methods (Isolation Forest, '
                    f'One-Class SVM) may perform better than standard classification because they '
                    f'learn what "normal" looks like rather than trying to classify both classes.'
                ),
                'estimated_benefit': 'Better recall for rare events, no need for resampling.',
                'difficulty': 'medium',
            })

    # 5. Clustering suggestion (no target or weak target)
    if current_problem_type in ('classification', 'regression'):
        top_corr = _get_target_correlation_strength(df, target_column)
        if top_corr < 0.1:
            suggestions.append({
                'type': 'unsupervised_first',
                'icon': '🧩',
                'title': 'Run Clustering First',
                'rationale': (
                    'The features show very weak correlation with the target variable '
                    f'(max |r| = {top_corr:.3f}). Consider running clustering first to discover '
                    f'natural segments in your data, then build separate models per cluster.'
                ),
                'estimated_benefit': 'Cluster-specific models can capture segment-level patterns.',
                'difficulty': 'high',
            })

    # 6. Multi-label suggestion
    if current_problem_type == 'classification':
        if _might_be_multilabel(df, target_column):
            suggestions.append({
                'type': 'multi_label',
                'icon': '🏷️',
                'title': 'Consider Multi-Label Classification',
                'rationale': (
                    'Your target column contains values that might represent multiple labels '
                    '(e.g., comma-separated categories). If records can belong to multiple classes '
                    'simultaneously, multi-label classification would be more appropriate.'
                ),
                'estimated_benefit': 'Captures overlapping class memberships.',
                'difficulty': 'medium',
            })

    return suggestions


# ── Helper Functions ──────────────────────────────────────────

def _has_temporal_pattern(df):
    """Check if the dataset contains temporal features."""
    for col in df.columns:
        # Check datetime columns
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return True

        # Check string columns that look like dates
        if df[col].dtype == 'object':
            try:
                sample = df[col].dropna().head(10)
                parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                if parsed.notna().sum() > 7:
                    return True
            except Exception:
                pass

        # Check for monotonically increasing numeric (like timestamps)
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].is_monotonic_increasing and df[col].nunique() > len(df) * 0.9:
                return True

    return False


def _is_ordered(target):
    """Check if target values have a natural ordering."""
    if pd.api.types.is_numeric_dtype(target):
        return True

    # Check if string values have ordering indicators
    vals = target.dropna().unique()
    order_indicators = ['low', 'medium', 'high', 'very', 'poor', 'fair', 'good', 'excellent',
                        'small', 'large', 'mild', 'moderate', 'severe',
                        '1', '2', '3', '4', '5', 'a', 'b', 'c', 'd']
    matches = sum(1 for v in vals if any(ind in str(v).lower() for ind in order_indicators))
    return matches >= len(vals) * 0.5


def _is_heavily_skewed(target):
    """Check if target is heavily skewed."""
    try:
        return abs(float(target.skew())) > 2
    except Exception:
        return False


def _suggest_bins(target):
    """Suggest bin boundaries for a continuous target."""
    try:
        q33 = float(target.quantile(0.33))
        q66 = float(target.quantile(0.66))
        return {
            'low': f'≤ {q33:.2f}',
            'medium': f'{q33:.2f} – {q66:.2f}',
            'high': f'> {q66:.2f}',
        }
    except Exception:
        return {}


def _get_target_correlation_strength(df, target_column):
    """Get the maximum absolute correlation with target."""
    try:
        numeric = df.select_dtypes(include='number')
        if target_column in numeric.columns and numeric.shape[1] > 1:
            corrs = numeric.corr()[target_column].drop(target_column).abs()
            return float(corrs.max()) if len(corrs) > 0 else 0
    except Exception:
        pass
    return 0


def _might_be_multilabel(df, target_column):
    """Check if the target might be multi-label (comma-separated, etc.)."""
    if df[target_column].dtype != 'object':
        return False

    sample = df[target_column].dropna().head(100)
    contains_separator = sample.str.contains(r'[,;|]', regex=True).mean()
    return contains_separator > 0.1
