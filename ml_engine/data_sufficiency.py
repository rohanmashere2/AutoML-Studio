"""
AutoML Studio — Data Sufficiency Calculator (Feature #25)
Before training, estimates whether the dataset has enough samples for
reliable modelling.  Produces a verdicts report with specific guidance.
"""

import numpy as np
import pandas as pd


def analyze_sufficiency(df, target_col=None, problem_type='classification'):
    """
    Assess whether the dataset is large enough for robust ML.

    Returns:
        dict with checks list, overall verdict, and recommendations
    """
    n_rows, n_cols = df.shape
    n_features = n_cols - 1 if target_col else n_cols

    checks = []
    recommendations = []

    # ── 1. Rows-to-features ratio ────────────────────────────
    ratio = n_rows / max(n_features, 1)
    if ratio >= 20:
        status, severity = 'pass', 'good'
        msg = f'Rows/features ratio is {ratio:.1f} (≥20). Sufficient data density.'
    elif ratio >= 10:
        status, severity = 'warning', 'marginal'
        msg = f'Rows/features ratio is {ratio:.1f} (10-20). Marginal — regularisation recommended.'
        recommendations.append('Use strong regularisation (Ridge/Lasso) or reduce features via PCA.')
    else:
        status, severity = 'fail', 'insufficient'
        msg = f'Rows/features ratio is {ratio:.1f} (<10). HIGH RISK of overfitting.'
        recommendations.append(f'Collect at least {int(n_features * 20 - n_rows)} more samples, or reduce features to ≤{n_rows // 20}.')
    checks.append({'name': 'Rows / Features Ratio', 'value': round(ratio, 1),
                    'threshold': '≥10', 'status': status, 'severity': severity,
                    'message': msg, 'icon': '📊'})

    # ── 2. Class-specific checks (classification) ────────────
    if problem_type == 'classification' and target_col and target_col in df.columns:
        class_counts = df[target_col].value_counts()
        n_classes = len(class_counts)
        min_class_name = str(class_counts.idxmin())
        min_class_count = int(class_counts.min())
        avg_class_count = int(class_counts.mean())

        # Minimum samples per class
        if min_class_count >= 50:
            status, severity = 'pass', 'good'
            msg = f'Smallest class "{min_class_name}" has {min_class_count} samples (≥50).'
        elif min_class_count >= 20:
            status, severity = 'warning', 'marginal'
            msg = f'Smallest class "{min_class_name}" has only {min_class_count} samples (20-50).'
            recommendations.append(f'Collect {50 - min_class_count}+ more samples of class "{min_class_name}".')
        else:
            status, severity = 'fail', 'insufficient'
            msg = f'Smallest class "{min_class_name}" has only {min_class_count} samples (<20). Model will struggle.'
            recommendations.append(f'Collect {50 - min_class_count}+ more samples of "{min_class_name}", or merge rare classes, or use SMOTE.')
        checks.append({'name': 'Min Samples per Class', 'value': min_class_count,
                        'threshold': '≥50', 'status': status, 'severity': severity,
                        'message': msg, 'icon': '🏷️'})

        # Average samples per class
        if avg_class_count >= 100:
            status, severity = 'pass', 'good'
        elif avg_class_count >= 30:
            status, severity = 'warning', 'marginal'
        else:
            status, severity = 'fail', 'insufficient'
        checks.append({'name': 'Avg Samples per Class', 'value': avg_class_count,
                        'threshold': '≥100', 'status': status, 'severity': severity,
                        'message': f'{avg_class_count} samples per class on average across {n_classes} classes.',
                        'icon': '📦'})

        # Class balance
        imbalance_ratio = round(int(class_counts.max()) / max(min_class_count, 1), 1)
        if imbalance_ratio <= 2:
            status, severity = 'pass', 'good'
            msg = f'Class balance ratio {imbalance_ratio}:1. Well balanced.'
        elif imbalance_ratio <= 5:
            status, severity = 'warning', 'marginal'
            msg = f'Class balance ratio {imbalance_ratio}:1. Moderate imbalance.'
            recommendations.append('Use class_weight="balanced" or SMOTE resampling.')
        else:
            status, severity = 'fail', 'insufficient'
            msg = f'Class balance ratio {imbalance_ratio}:1. Severe imbalance.'
            recommendations.append('Apply SMOTE/ADASYN and use F1 instead of accuracy as primary metric.')
        checks.append({'name': 'Class Balance', 'value': f'{imbalance_ratio}:1',
                        'threshold': '≤3:1', 'status': status, 'severity': severity,
                        'message': msg, 'icon': '⚖️'})

    # ── 3. Regression-specific checks ────────────────────────
    if problem_type == 'regression' and target_col and target_col in df.columns:
        target = df[target_col].dropna()
        # Minimum samples for regression
        min_needed = max(50, n_features * 10)
        if n_rows >= min_needed:
            status, severity = 'pass', 'good'
            msg = f'{n_rows} samples (need ≥{min_needed} for {n_features} features).'
        else:
            status, severity = 'fail', 'insufficient'
            msg = f'Only {n_rows} samples (need ≥{min_needed} for {n_features} features).'
            recommendations.append(f'Collect {min_needed - n_rows}+ more samples for reliable regression.')
        checks.append({'name': 'Regression Sample Size', 'value': n_rows,
                        'threshold': f'≥{min_needed}', 'status': status,
                        'severity': severity, 'message': msg, 'icon': '📈'})

    # ── 4. Missing value density ─────────────────────────────
    missing_pct = round(df.isnull().sum().sum() / max(n_rows * n_cols, 1) * 100, 2)
    if missing_pct <= 5:
        status, severity = 'pass', 'good'
        msg = f'Missing values: {missing_pct}%. Minimal impact on effective sample size.'
    elif missing_pct <= 20:
        status, severity = 'warning', 'marginal'
        msg = f'Missing values: {missing_pct}%. Effective sample size is reduced.'
        recommendations.append('Use iterative imputation (MICE) to preserve information.')
    else:
        status, severity = 'fail', 'insufficient'
        msg = f'Missing values: {missing_pct}%. Severe data loss — effective rows much lower than {n_rows}.'
        recommendations.append('Many rows have missing data. Consider dropping columns with >70% missing, or collecting cleaner data.')
    checks.append({'name': 'Missing Data Density', 'value': f'{missing_pct}%',
                    'threshold': '≤5%', 'status': status, 'severity': severity,
                    'message': msg, 'icon': '🕳️'})

    # ── 5. Feature diversity ─────────────────────────────────
    n_numeric = len(df.select_dtypes(include='number').columns)
    n_categorical = len(df.select_dtypes(include=['object', 'category']).columns)
    if target_col and target_col in df.select_dtypes(include='number').columns:
        n_numeric -= 1

    if n_numeric >= 3 and n_features >= 5:
        status, severity = 'pass', 'good'
        msg = f'{n_numeric} numeric + {n_categorical} categorical features. Good diversity.'
    elif n_features >= 2:
        status, severity = 'warning', 'marginal'
        msg = f'Only {n_features} features. Consider adding more predictive columns.'
    else:
        status, severity = 'fail', 'insufficient'
        msg = 'Very few features. Model will have limited predictive power.'
        recommendations.append('Collect additional features or engineer new ones (polynomial, interactions).')
    checks.append({'name': 'Feature Diversity', 'value': f'{n_numeric}N + {n_categorical}C',
                    'threshold': '≥5 total', 'status': status, 'severity': severity,
                    'message': msg, 'icon': '🎨'})

    # ── 6. Duplicate ratio ───────────────────────────────────
    dup_count = int(df.duplicated().sum())
    dup_pct = round(dup_count / max(n_rows, 1) * 100, 2)
    effective_rows = n_rows - dup_count
    if dup_pct <= 1:
        status, severity = 'pass', 'good'
        msg = f'{dup_pct}% duplicates. Negligible impact.'
    elif dup_pct <= 10:
        status, severity = 'warning', 'marginal'
        msg = f'{dup_pct}% duplicates ({dup_count} rows). Effective unique rows: {effective_rows}.'
    else:
        status, severity = 'fail', 'insufficient'
        msg = f'{dup_pct}% duplicates ({dup_count} rows). Effective dataset is only {effective_rows} rows.'
        recommendations.append('High duplicate rate. Verify data collection process.')
    checks.append({'name': 'Duplicate Impact', 'value': f'{dup_pct}%',
                    'threshold': '≤1%', 'status': status, 'severity': severity,
                    'message': msg, 'icon': '🔄'})

    # ── Overall verdict ──────────────────────────────────────
    fail_count = sum(1 for c in checks if c['status'] == 'fail')
    warn_count = sum(1 for c in checks if c['status'] == 'warning')
    pass_count = sum(1 for c in checks if c['status'] == 'pass')

    if fail_count >= 2:
        verdict = 'insufficient'
        verdict_msg = f'Dataset is INSUFFICIENT for reliable modelling. {fail_count} critical issues found.'
        verdict_icon = '🔴'
    elif fail_count == 1:
        verdict = 'marginal'
        verdict_msg = f'Dataset is MARGINAL. 1 critical issue and {warn_count} warnings.'
        verdict_icon = '🟡'
    elif warn_count >= 2:
        verdict = 'marginal'
        verdict_msg = f'Dataset is MARGINAL. {warn_count} warnings — model may have unstable performance.'
        verdict_icon = '🟡'
    else:
        verdict = 'sufficient'
        verdict_msg = f'Dataset is SUFFICIENT for reliable modelling. {pass_count}/{len(checks)} checks passed.'
        verdict_icon = '🟢'

    return {
        'checks': checks,
        'verdict': verdict,
        'verdict_message': verdict_msg,
        'verdict_icon': verdict_icon,
        'pass_count': pass_count,
        'warn_count': warn_count,
        'fail_count': fail_count,
        'total_checks': len(checks),
        'recommendations': recommendations,
        'dataset_stats': {
            'n_rows': n_rows,
            'n_features': n_features,
            'n_numeric': n_numeric,
            'n_categorical': n_categorical,
            'missing_pct': missing_pct,
            'duplicate_pct': dup_pct,
        },
    }
