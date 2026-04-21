"""
AutoML Studio - Data Drift Monitor
Detects data drift for both numerical (PSI + KS-test) and categorical (Chi-square) features.
Generates comparison tables and automatic retraining recommendations.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def compute_drift(training_data, new_data, feature_names=None):
    """
    Compute data drift metrics between training and new data.
    Handles both numerical and categorical features.

    Returns:
        dict: Per-feature drift scores, overall drift status, retraining recommendation
    """
    if feature_names is None:
        feature_names = list(training_data.columns) if isinstance(training_data, pd.DataFrame) else [f'feature_{i}' for i in range(training_data.shape[1])]

    # Work with DataFrames when possible for dtype detection
    if isinstance(training_data, pd.DataFrame):
        train_df = training_data
    else:
        train_df = pd.DataFrame(training_data, columns=feature_names[:training_data.shape[1]])

    if isinstance(new_data, pd.DataFrame):
        new_df = new_data
    else:
        new_df = pd.DataFrame(new_data, columns=feature_names[:new_data.shape[1]])

    feature_drift = []
    total_psi = 0
    drift_count = 0
    n_features_analyzed = 0

    for fname in feature_names:
        if fname not in train_df.columns or fname not in new_df.columns:
            continue

        train_col = train_df[fname]
        new_col = new_df[fname]

        # Determine if numerical or categorical
        is_numeric = pd.api.types.is_numeric_dtype(train_col) and pd.api.types.is_numeric_dtype(new_col)

        if is_numeric:
            drift_info = _compute_numerical_drift(train_col, new_col, fname)
        else:
            drift_info = _compute_categorical_drift(train_col, new_col, fname)

        if drift_info is None:
            continue

        n_features_analyzed += 1
        total_psi += drift_info.get('psi', drift_info.get('drift_score', 0))

        if drift_info['status'] == 'high_drift':
            drift_count += 1
        elif drift_info['status'] == 'moderate_drift':
            drift_count += 0.5

        feature_drift.append(drift_info)

    # Sort by drift score (most drifted first)
    feature_drift.sort(key=lambda x: x.get('psi', x.get('drift_score', 0)), reverse=True)

    # Overall status
    avg_psi = total_psi / max(n_features_analyzed, 1)
    drifted_features = sum(1 for f in feature_drift if f['status'] == 'high_drift')
    moderate_features = sum(1 for f in feature_drift if f['status'] == 'moderate_drift')

    if avg_psi > 0.25 or drifted_features > n_features_analyzed * 0.3:
        overall_status = 'critical'
        recommendation = 'Model retraining is strongly recommended. Significant data drift detected.'
    elif avg_psi > 0.1 or drifted_features > 0:
        overall_status = 'warning'
        recommendation = 'Some features show drift. Monitor closely and consider retraining soon.'
    else:
        overall_status = 'healthy'
        recommendation = 'No significant data drift detected. Model predictions should remain reliable.'

    result = {
        'feature_drift': feature_drift,
        'overall_status': overall_status,
        'recommendation': recommendation,
        'summary': {
            'avg_psi': round(avg_psi, 4),
            'total_features': n_features_analyzed,
            'high_drift_features': drifted_features,
            'moderate_drift_features': moderate_features,
            'healthy_features': n_features_analyzed - drifted_features - moderate_features,
        },
    }

    # Automatic retraining recommendation
    if overall_status in ('critical', 'warning'):
        result['retraining_recommendation'] = {
            'urgency': 'high' if overall_status == 'critical' else 'medium',
            'reason': f'{drifted_features} features show significant drift',
            'drifted_features': [f['feature'] for f in feature_drift if f['status'] == 'high_drift'],
            'suggested_action': 'retrain',
            'estimated_impact': 'Model accuracy may have degraded by 5-15%' if overall_status == 'critical' else 'Model accuracy may have degraded by 2-5%',
            'auto_retrain_available': True,
        }

    return result


def _compute_numerical_drift(train_col, new_col, fname):
    """Compute drift metrics for a numerical feature using PSI + KS-test."""
    train_vals = train_col.dropna().astype(float).values
    new_vals = new_col.dropna().astype(float).values

    if len(train_vals) < 10 or len(new_vals) < 5:
        return None

    # PSI (Population Stability Index)
    psi = _compute_psi(train_vals, new_vals)

    # KS Test
    ks_stat, ks_pvalue = sp_stats.ks_2samp(train_vals, new_vals)

    # Distribution statistics
    train_mean = float(np.mean(train_vals))
    new_mean = float(np.mean(new_vals))
    train_std = float(np.std(train_vals))
    new_std = float(np.std(new_vals))

    # Mean shift (in std-dev units)
    mean_shift = abs(new_mean - train_mean) / max(train_std, 1e-10)

    # Determine drift status
    if psi > 0.25 or ks_pvalue < 0.001:
        status = 'high_drift'
    elif psi > 0.1 or ks_pvalue < 0.05:
        status = 'moderate_drift'
    else:
        status = 'no_drift'

    return {
        'feature': fname,
        'type': 'numerical',
        'psi': round(psi, 4),
        'drift_score': round(psi, 4),
        'ks_statistic': round(float(ks_stat), 4),
        'ks_pvalue': round(float(ks_pvalue), 6),
        'mean_shift': round(mean_shift, 4),
        'train_mean': round(train_mean, 4),
        'new_mean': round(new_mean, 4),
        'train_std': round(train_std, 4),
        'new_std': round(new_std, 4),
        'status': status,
    }


def _compute_categorical_drift(train_col, new_col, fname):
    """Compute drift metrics for a categorical feature using Chi-square test."""
    train_vals = train_col.dropna().astype(str)
    new_vals = new_col.dropna().astype(str)

    if len(train_vals) < 10 or len(new_vals) < 5:
        return None

    # Compute value distributions
    train_counts = train_vals.value_counts(normalize=True)
    new_counts = new_vals.value_counts(normalize=True)

    # Align categories
    all_categories = sorted(set(train_counts.index) | set(new_counts.index))
    train_dist = {cat: train_counts.get(cat, 0) for cat in all_categories}
    new_dist = {cat: new_counts.get(cat, 0) for cat in all_categories}

    # New / missing categories
    new_categories = [c for c in new_counts.index if c not in train_counts.index]
    missing_categories = [c for c in train_counts.index if c not in new_counts.index]

    # Chi-square test
    try:
        train_freq = np.array([train_dist[c] for c in all_categories]) * len(train_vals)
        new_freq = np.array([new_dist[c] for c in all_categories]) * len(new_vals)

        # Add small constant to avoid zero-division
        train_freq = np.maximum(train_freq, 0.1)
        new_freq = np.maximum(new_freq, 0.1)

        chi2, chi2_pvalue = sp_stats.chisquare(new_freq, f_exp=train_freq * len(new_vals) / len(train_vals))
        chi2 = float(chi2)
        chi2_pvalue = float(chi2_pvalue)
    except Exception:
        chi2 = 0
        chi2_pvalue = 1.0

    # Cramér's V (effect size for categorical drift)
    try:
        n = len(new_vals)
        k = len(all_categories)
        cramers_v = float(np.sqrt(chi2 / (n * max(k - 1, 1)))) if n > 0 else 0
    except Exception:
        cramers_v = 0

    # PSI-like metric for categorical
    cat_psi = 0
    for cat in all_categories:
        p = max(train_dist[cat], 0.0001)
        q = max(new_dist[cat], 0.0001)
        cat_psi += (q - p) * np.log(q / p)

    # Determine drift status
    if cat_psi > 0.25 or chi2_pvalue < 0.001 or cramers_v > 0.3:
        status = 'high_drift'
    elif cat_psi > 0.1 or chi2_pvalue < 0.05 or cramers_v > 0.15:
        status = 'moderate_drift'
    else:
        status = 'no_drift'

    return {
        'feature': fname,
        'type': 'categorical',
        'psi': round(abs(cat_psi), 4),
        'drift_score': round(abs(cat_psi), 4),
        'chi2_statistic': round(chi2, 4),
        'chi2_pvalue': round(chi2_pvalue, 6),
        'cramers_v': round(cramers_v, 4),
        'reference_distribution': {k: round(v, 4) for k, v in list(train_dist.items())[:20]},
        'current_distribution': {k: round(v, 4) for k, v in list(new_dist.items())[:20]},
        'new_categories': new_categories[:10],
        'missing_categories': missing_categories[:10],
        'status': status,
    }


def generate_comparison_table(drift_result):
    """Generate an Evidently-style reference-vs-current comparison table."""
    comparison = []
    for feature in drift_result.get('feature_drift', []):
        entry = {
            'feature': feature['feature'],
            'type': feature.get('type', 'numerical'),
            'drift_score': feature.get('drift_score', feature.get('psi', 0)),
            'status': feature['status'],
        }

        if feature.get('type') == 'numerical':
            entry['reference'] = {
                'mean': feature.get('train_mean'),
                'std': feature.get('train_std'),
            }
            entry['current'] = {
                'mean': feature.get('new_mean'),
                'std': feature.get('new_std'),
            }
            entry['test'] = 'KS-test'
            entry['p_value'] = feature.get('ks_pvalue')
        else:
            entry['reference'] = feature.get('reference_distribution', {})
            entry['current'] = feature.get('current_distribution', {})
            entry['test'] = 'Chi-square'
            entry['p_value'] = feature.get('chi2_pvalue')
            entry['new_categories'] = feature.get('new_categories', [])
            entry['missing_categories'] = feature.get('missing_categories', [])

        comparison.append(entry)

    return comparison


def compute_prediction_drift(original_predictions, new_predictions):
    """
    Compare prediction distributions between training and new data.

    Returns:
        dict: Prediction drift analysis
    """
    orig = np.array(original_predictions, dtype=float)
    new = np.array(new_predictions, dtype=float)

    # KS test
    ks_stat, ks_pvalue = sp_stats.ks_2samp(orig, new)

    # PSI
    psi = _compute_psi(orig, new)

    # Distribution stats
    result = {
        'psi': round(psi, 4),
        'ks_statistic': round(float(ks_stat), 4),
        'ks_pvalue': round(float(ks_pvalue), 6),
        'original_stats': {
            'mean': round(float(np.mean(orig)), 4),
            'std': round(float(np.std(orig)), 4),
            'median': round(float(np.median(orig)), 4),
        },
        'new_stats': {
            'mean': round(float(np.mean(new)), 4),
            'std': round(float(np.std(new)), 4),
            'median': round(float(np.median(new)), 4),
        },
    }

    if psi > 0.25:
        result['status'] = 'critical'
        result['message'] = 'Prediction distribution has shifted significantly.'
    elif psi > 0.1:
        result['status'] = 'warning'
        result['message'] = 'Prediction distribution shows moderate shift.'
    else:
        result['status'] = 'healthy'
        result['message'] = 'Prediction distribution is stable.'

    return result


def _compute_psi(expected, actual, n_bins=10):
    """Compute Population Stability Index."""
    try:
        # Create bins from expected distribution
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 2:
            return 0.0

        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        # Normalize
        expected_pct = expected_counts / max(len(expected), 1)
        actual_pct = actual_counts / max(len(actual), 1)

        # Replace zeros
        expected_pct = np.maximum(expected_pct, 0.0001)
        actual_pct = np.maximum(actual_pct, 0.0001)

        # PSI formula
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return float(psi)
    except Exception:
        return 0.0


def generate_drift_report(drift_result):
    """Generate a human-readable drift report with retraining recommendation."""
    summary = drift_result['summary']
    status = drift_result['overall_status']

    status_emoji = {'critical': '🔴', 'warning': '🟡', 'healthy': '🟢'}

    report = {
        'title': f'{status_emoji.get(status, "⚪")} Data Drift Report',
        'status': status,
        'status_text': drift_result['recommendation'],
        'stats': {
            'total_features': summary['total_features'],
            'high_drift': summary['high_drift_features'],
            'moderate_drift': summary['moderate_drift_features'],
            'healthy': summary['healthy_features'],
            'avg_psi': summary['avg_psi'],
        },
        'top_drifted': drift_result['feature_drift'][:5],
        'should_retrain': status == 'critical',
        'comparison_table': generate_comparison_table(drift_result),
    }

    # Include retraining recommendation if present
    if 'retraining_recommendation' in drift_result:
        report['retraining_recommendation'] = drift_result['retraining_recommendation']

    return report


def detect_model_drift(model, X_new, y_new, original_score, problem_type='classification'):
    """
    Detect model performance drift by comparing predictions on new labeled data
    against the original training performance.

    Args:
        model: trained model
        X_new: new feature data
        y_new: new actual labels
        original_score: original best score from training
        problem_type: 'classification' or 'regression'

    Returns:
        dict with drift analysis
    """
    from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error
    import numpy as np

    try:
        y_pred = model.predict(X_new)
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

    if problem_type == 'classification':
        new_score = float(accuracy_score(y_new, y_pred))
        f1 = float(f1_score(y_new, y_pred, average='weighted', zero_division=0))
        metric_name = 'accuracy'
    else:
        new_score = float(r2_score(y_new, y_pred))
        mae = float(mean_absolute_error(y_new, y_pred))
        metric_name = 'r2'

    degradation = original_score - new_score
    degradation_pct = round(degradation * 100, 2)

    if degradation > 0.15:
        status = 'critical'
        message = f'⚠️ Model performance has decreased significantly by {degradation_pct}%.'
        action = 'Immediate retraining recommended.'
    elif degradation > 0.05:
        status = 'warning'
        message = f'Model performance decreased by {degradation_pct}%. Monitor closely.'
        action = 'Consider retraining with recent data.'
    elif degradation > 0:
        status = 'minor'
        message = f'Slight performance decrease ({degradation_pct}%). Still acceptable.'
        action = 'No immediate action needed.'
    else:
        status = 'healthy'
        message = 'Model performance is stable or improved on new data.'
        action = 'No action needed.'

    result = {
        'status': status,
        'message': message,
        'action': action,
        'original_score': round(original_score, 4),
        'new_score': round(new_score, 4),
        'degradation': round(degradation, 4),
        'degradation_pct': degradation_pct,
        'metric': metric_name,
        'n_samples': len(y_new),
    }

    if problem_type == 'classification':
        result['f1_score'] = round(f1, 4)
    else:
        result['mae'] = round(mae, 4)

    return result
