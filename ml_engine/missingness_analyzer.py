"""
AutoML Studio — Missingness Pattern Analyzer
Runs Little's MCAR test, classifies columns as MCAR / MAR / MNAR,
produces a missingness correlation heatmap, and auto-creates binary
_is_missing indicator features for MNAR columns.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ── Little's MCAR Test (simplified) ──────────────────────────

def _littles_mcar_test(df):
    """
    Simplified Little's MCAR test.
    Tests whether missingness pattern is completely at random.

    Returns:
        dict with chi2 statistic, df, p_value
    """
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] < 2:
        return {'chi2': 0, 'df': 0, 'p_value': 1.0, 'is_mcar': True}

    # Build missingness pattern matrix
    missing_mask = numeric.isnull().astype(int)
    n_missing = missing_mask.sum().sum()

    if n_missing == 0:
        return {'chi2': 0, 'df': 0, 'p_value': 1.0, 'is_mcar': True}

    # Group rows by missingness pattern
    patterns = missing_mask.apply(lambda row: tuple(row), axis=1)
    unique_patterns = patterns.unique()

    if len(unique_patterns) <= 1:
        return {'chi2': 0, 'df': 0, 'p_value': 1.0, 'is_mcar': True}

    # Compute overall means and covariance
    overall_means = numeric.mean()
    n_vars = numeric.shape[1]
    chi2_total = 0.0
    df_total = 0

    for pattern in unique_patterns:
        mask = patterns == pattern
        group = numeric[mask]
        n_group = len(group)

        if n_group < 2:
            continue

        # Which columns are observed in this pattern?
        pattern_arr = np.array(pattern)
        observed_idx = np.where(pattern_arr == 0)[0]

        if len(observed_idx) == 0:
            continue

        observed_cols = numeric.columns[observed_idx]
        group_observed = group[observed_cols].dropna()

        if len(group_observed) < 2:
            continue

        # Mean difference for observed variables
        group_means = group_observed.mean()
        mean_diff = (group_means - overall_means[observed_cols]).values

        # Covariance of observed variables
        try:
            cov = group_observed.cov().values
            if cov.shape[0] == 0:
                continue
            cov_inv = np.linalg.pinv(cov / n_group)
            chi2_contrib = n_group * mean_diff @ cov_inv @ mean_diff
            chi2_total += float(chi2_contrib)
            df_total += len(observed_idx)
        except (np.linalg.LinAlgError, ValueError):
            continue

    df_total = max(df_total - n_vars, 1)

    try:
        p_value = float(1 - sp_stats.chi2.cdf(chi2_total, df_total))
    except Exception:
        p_value = 1.0

    return {
        'chi2': round(chi2_total, 4),
        'df': df_total,
        'p_value': round(p_value, 6),
        'is_mcar': p_value > 0.05,
    }


# ── Per-Column Missingness Pattern ───────────────────────────

def _classify_column_missingness(df, col):
    """
    Classify a single column's missingness as MCAR, MAR, or MNAR.

    Strategy:
    - MCAR: missingness is independent of all other columns
    - MAR: missingness can be predicted by other observed columns
    - MNAR: missingness depends on the column's own (unobserved) value
    """
    if df[col].isnull().sum() == 0:
        return {
            'pattern': 'complete',
            'mcar_p_value': 1.0,
            'mar_score': 0.0,
            'mar_predictors': [],
            'recommendation': 'No missing values',
        }

    missing_indicator = df[col].isnull().astype(int)
    other_cols = [c for c in df.columns if c != col]

    # ── Test for MCAR: correlation between missingness and other columns
    mcar_p_values = []
    for other in other_cols:
        if df[other].isnull().sum() == len(df):
            continue
        try:
            if pd.api.types.is_numeric_dtype(df[other]):
                # Point-biserial correlation
                observed = df[other].dropna()
                if len(observed) < 10:
                    continue
                # Align
                common_idx = df[[col, other]].dropna(subset=[other]).index
                if len(common_idx) < 10:
                    continue
                corr, p_val = sp_stats.pointbiserialr(
                    missing_indicator.loc[common_idx],
                    df[other].loc[common_idx]
                )
                mcar_p_values.append(p_val)
            else:
                # Chi-square test for categorical
                observed = df[other].dropna()
                if len(observed) < 10:
                    continue
                common_idx = df[[col, other]].dropna(subset=[other]).index
                if len(common_idx) < 10:
                    continue
                contingency = pd.crosstab(
                    missing_indicator.loc[common_idx],
                    df[other].loc[common_idx].astype(str)
                )
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    continue
                chi2, p_val, _, _ = sp_stats.chi2_contingency(contingency)
                mcar_p_values.append(p_val)
        except Exception:
            continue

    # MCAR if no significant correlations
    if not mcar_p_values:
        mcar_p_value = 1.0
    else:
        # Use Bonferroni correction
        min_p = min(mcar_p_values)
        mcar_p_value = min(min_p * len(mcar_p_values), 1.0)

    # ── Test for MAR: can other columns predict missingness?
    mar_score = 0.0
    mar_predictors = []

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        # Use numeric columns to predict missingness
        numeric_others = df[other_cols].select_dtypes(include='number')
        numeric_others = numeric_others.dropna(axis=1, how='all')

        if numeric_others.shape[1] > 0 and len(df) > 20:
            X = numeric_others.fillna(numeric_others.median())
            y = missing_indicator

            # Quick logistic regression
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            lr = LogisticRegression(max_iter=500, random_state=42)
            lr.fit(X_scaled, y)

            # AUC as MAR strength
            from sklearn.metrics import roc_auc_score
            y_prob = lr.predict_proba(X_scaled)[:, 1]
            mar_score = float(roc_auc_score(y, y_prob))

            # Top predictors (by absolute coefficient)
            coef_abs = np.abs(lr.coef_[0])
            top_idx = np.argsort(coef_abs)[::-1][:3]
            mar_predictors = [numeric_others.columns[i] for i in top_idx
                              if coef_abs[i] > 0.1]

    except Exception:
        mar_score = 0.5  # Can't determine

    # ── Classify
    if mcar_p_value > 0.05 and mar_score < 0.6:
        pattern = 'MCAR'
        recommendation = 'Safe to impute with mean/median/mode — missingness is random.'
    elif mar_score >= 0.6:
        pattern = 'MAR'
        predictors_str = ', '.join(mar_predictors[:3]) if mar_predictors else 'other columns'
        recommendation = (
            f'Missingness predicted by {predictors_str}. '
            f'Use KNN or MICE imputation for best results.'
        )
    else:
        pattern = 'MNAR'
        recommendation = (
            'Missingness likely depends on the missing value itself (informative). '
            'Create a binary _is_missing indicator to preserve this signal before imputing.'
        )

    return {
        'pattern': pattern,
        'mcar_p_value': round(mcar_p_value, 6),
        'mar_score': round(mar_score, 4),
        'mar_predictors': mar_predictors,
        'recommendation': recommendation,
    }


# ── Public API ───────────────────────────────────────────────

def analyze_missingness(df):
    """
    Full missingness analysis for the dataset.

    Returns:
        dict with per-column patterns, overall MCAR test, heatmap data
    """
    # Overall MCAR test
    mcar_test = _littles_mcar_test(df)

    # Per-column analysis (only columns with missing values)
    columns_with_missing = [c for c in df.columns if df[c].isnull().sum() > 0]
    column_results = {}

    for col in columns_with_missing:
        missing_pct = round(df[col].isnull().mean() * 100, 2)
        result = _classify_column_missingness(df, col)
        result['missing_count'] = int(df[col].isnull().sum())
        result['missing_pct'] = missing_pct
        column_results[col] = result

    # Missingness correlation heatmap
    heatmap = compute_missingness_heatmap(df)

    # Summary
    patterns = [r['pattern'] for r in column_results.values()]
    mcar_count = patterns.count('MCAR')
    mar_count = patterns.count('MAR')
    mnar_count = patterns.count('MNAR')

    overall = 'MCAR' if mcar_test['is_mcar'] else 'mixed'
    if mnar_count > 0:
        overall = 'contains_MNAR'

    return {
        'mcar_test': mcar_test,
        'columns': column_results,
        'heatmap': heatmap,
        'summary': {
            'total_columns_with_missing': len(columns_with_missing),
            'mcar_columns': mcar_count,
            'mar_columns': mar_count,
            'mnar_columns': mnar_count,
            'overall_pattern': overall,
        },
    }


def compute_missingness_heatmap(df):
    """
    Compute correlation matrix of missingness indicators.
    Shows which columns tend to be missing together.

    Returns:
        dict with column names and correlation matrix
    """
    missing_cols = [c for c in df.columns if df[c].isnull().sum() > 0]

    if len(missing_cols) < 2:
        return {'columns': missing_cols, 'matrix': []}

    missing_matrix = df[missing_cols].isnull().astype(int)
    corr = missing_matrix.corr()

    return {
        'columns': missing_cols,
        'matrix': [[round(float(corr.iloc[i, j]), 4)
                     for j in range(len(missing_cols))]
                    for i in range(len(missing_cols))],
    }


def create_missing_indicators(df, mnar_columns):
    """
    Create binary _is_missing indicator features for MNAR columns.
    Must be called BEFORE imputation to preserve the signal.

    Args:
        df: DataFrame
        mnar_columns: list of column names classified as MNAR

    Returns:
        (modified_df, list of indicator column names created)
    """
    df = df.copy()
    indicators_created = []

    for col in mnar_columns:
        if col in df.columns and df[col].isnull().sum() > 0:
            indicator_name = f'{col}_is_missing'
            df[indicator_name] = df[col].isnull().astype(int)
            indicators_created.append(indicator_name)

    return df, indicators_created
