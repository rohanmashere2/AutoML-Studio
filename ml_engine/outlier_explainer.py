"""
AutoML Studio — Outlier Explanation Engine (Feature #20)
For each detected outlier, explains WHY it's an outlier and whether
it should be kept (legitimate extreme) or removed (data error).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from scipy import stats as sp_stats


def explain_outliers(df, target_col=None, max_outliers=50):
    """
    Detect and explain outliers using multiple methods.

    Returns:
        dict with per-outlier explanations and keep/remove recommendations
    """
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if not numeric_cols:
        return {'error': 'No numeric features for outlier analysis'}

    X = df[numeric_cols].fillna(df[numeric_cols].median())

    # Method 1: IQR
    iqr_outliers = _detect_iqr(X)
    # Method 2: Z-score
    z_outliers = _detect_zscore(X)
    # Method 3: Isolation Forest
    iso_outliers = _detect_isolation_forest(X)

    # Combine: an outlier is flagged by at least 2 methods
    all_indices = set()
    for idx_set in [iqr_outliers, z_outliers, iso_outliers]:
        all_indices.update(idx_set)

    outlier_details = []
    for idx in sorted(all_indices):
        if len(outlier_details) >= max_outliers:
            break
        methods_flagged = []
        if idx in iqr_outliers:
            methods_flagged.append('IQR')
        if idx in z_outliers:
            methods_flagged.append('Z-score')
        if idx in iso_outliers:
            methods_flagged.append('Isolation Forest')

        severity = len(methods_flagged)
        # Which features make it an outlier?
        row = X.iloc[idx]
        outlier_features = []
        for col in numeric_cols:
            val = float(row[col])
            col_mean = float(X[col].mean())
            col_std = float(X[col].std())
            if col_std < 1e-8:
                continue
            z = abs(val - col_mean) / col_std
            if z > 2.5:
                outlier_features.append({
                    'feature': col,
                    'value': round(val, 4),
                    'mean': round(col_mean, 4),
                    'z_score': round(z, 2),
                    'direction': 'above' if val > col_mean else 'below',
                })

        outlier_features.sort(key=lambda x: x['z_score'], reverse=True)

        # Check if there are similar samples (not isolated)
        if len(outlier_features) > 0:
            main_feat = outlier_features[0]['feature']
            main_val = outlier_features[0]['value']
            similar_count = int(((X[main_feat] - main_val).abs() / max(X[main_feat].std(), 1e-8) < 0.5).sum())
        else:
            similar_count = 0

        # Classification
        if severity >= 3:
            classification = 'likely_error'
            recommendation = 'REMOVE — flagged by all 3 methods. Likely a data entry error.'
            icon = '🔴'
        elif severity == 2 and similar_count <= 2:
            classification = 'suspicious'
            recommendation = 'INVESTIGATE — flagged by 2 methods and isolated. Could be error or rare event.'
            icon = '🟡'
        elif similar_count >= 5:
            classification = 'legitimate_extreme'
            recommendation = f'KEEP — {similar_count} similar samples exist. Likely a real extreme value.'
            icon = '🟢'
        else:
            classification = 'borderline'
            recommendation = 'REVIEW — borderline case. Check domain context.'
            icon = '🟠'

        outlier_details.append({
            'index': int(idx),
            'methods_flagged': methods_flagged,
            'severity': severity,
            'outlier_features': outlier_features[:5],
            'similar_count': similar_count,
            'classification': classification,
            'recommendation': recommendation,
            'icon': icon,
        })

    # Summary
    n_error = sum(1 for o in outlier_details if o['classification'] == 'likely_error')
    n_legit = sum(1 for o in outlier_details if o['classification'] == 'legitimate_extreme')
    n_suspicious = sum(1 for o in outlier_details if o['classification'] == 'suspicious')

    return {
        'outliers': outlier_details,
        'total_outliers': len(outlier_details),
        'summary': {
            'likely_errors': n_error,
            'legitimate_extremes': n_legit,
            'suspicious': n_suspicious,
            'borderline': len(outlier_details) - n_error - n_legit - n_suspicious,
        },
        'recommendation': (
            f'Found {len(outlier_details)} outliers: {n_error} likely errors (remove), '
            f'{n_legit} legitimate extremes (keep), {n_suspicious} need investigation.'
        ),
    }


def _detect_iqr(X):
    indices = set()
    for col in X.columns:
        Q1, Q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        mask = (X[col] < Q1 - 2.0 * IQR) | (X[col] > Q3 + 2.0 * IQR)
        indices.update(X.index[mask].tolist())
    return indices


def _detect_zscore(X):
    indices = set()
    for col in X.columns:
        z = np.abs(sp_stats.zscore(X[col], nan_policy='omit'))
        mask = z > 3.0
        indices.update(X.index[mask].tolist())
    return indices


def _detect_isolation_forest(X):
    try:
        iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
        labels = iso.fit_predict(X)
        return set(X.index[labels == -1].tolist())
    except Exception:
        return set()
