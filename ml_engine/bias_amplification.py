"""
AutoML Studio — Bias Amplification Detector (Feature #23)
Checks whether the model AMPLIFIES existing data biases rather than
just reflecting them.  Reports amplification factor per sensitive attribute.
"""

import numpy as np
import pandas as pd


def detect_bias_amplification(model, X_test, y_test, sensitive_columns,
                               feature_names=None):
    """
    Compare disparity in ground-truth labels vs model predictions to detect
    whether the model amplifies, maintains, or reduces existing biases.

    Args:
        model: trained model
        X_test: feature DataFrame/array
        y_test: true labels
        sensitive_columns: list of column names to audit
        feature_names: optional list of feature names for arrays

    Returns:
        dict with per-attribute amplification analysis
    """
    if not sensitive_columns:
        return {'error': 'No sensitive columns specified'}

    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

    y_true = np.array(y_test)
    y_pred = np.array(y_pred)

    if hasattr(X_test, 'columns'):
        df = X_test.copy()
    else:
        cols = feature_names or [f'f_{i}' for i in range(X_test.shape[1])]
        df = pd.DataFrame(X_test, columns=cols)

    results = []
    overall_risk = 'low'

    for col in sensitive_columns:
        if col not in df.columns:
            continue

        groups = df[col].unique()
        if len(groups) < 2 or len(groups) > 20:
            continue

        group_stats = []
        for g in groups:
            mask = df[col] == g
            n = int(mask.sum())
            if n < 5:
                continue

            # Positive rate in ground truth
            is_classification = len(np.unique(y_true)) <= 30
            if is_classification:
                # Use the most common positive class (highest value or last class)
                positive_class = max(np.unique(y_true))
                data_positive_rate = float((y_true[mask] == positive_class).mean())
                pred_positive_rate = float((y_pred[mask] == positive_class).mean())
            else:
                data_positive_rate = float(y_true[mask].mean())
                pred_positive_rate = float(y_pred[mask].mean())

            group_stats.append({
                'group': str(g),
                'n_samples': n,
                'data_rate': round(data_positive_rate, 4),
                'pred_rate': round(pred_positive_rate, 4),
            })

        if len(group_stats) < 2:
            continue

        # Compute disparity: max_rate - min_rate
        data_rates = [g['data_rate'] for g in group_stats]
        pred_rates = [g['pred_rate'] for g in group_stats]

        data_disparity = max(data_rates) - min(data_rates)
        pred_disparity = max(pred_rates) - min(pred_rates)

        # Amplification factor
        if data_disparity > 0.01:
            amplification = pred_disparity / data_disparity
        elif pred_disparity > 0.01:
            amplification = float('inf')
        else:
            amplification = 1.0

        # Classify
        if amplification > 1.5:
            severity = 'critical'
            message = f'Model AMPLIFIES bias {amplification:.1f}x. Disparity grew from {data_disparity:.1%} to {pred_disparity:.1%}.'
            icon = '🔴'
            overall_risk = 'critical'
        elif amplification > 1.1:
            severity = 'warning'
            message = f'Model slightly amplifies bias ({amplification:.1f}x). Disparity: {data_disparity:.1%} → {pred_disparity:.1%}.'
            icon = '🟡'
            if overall_risk != 'critical':
                overall_risk = 'warning'
        elif amplification >= 0.9:
            severity = 'neutral'
            message = f'Model maintains existing bias level ({amplification:.1f}x). Disparity unchanged at ~{data_disparity:.1%}.'
            icon = '🟠'
        else:
            severity = 'good'
            message = f'Model REDUCES bias ({amplification:.1f}x). Disparity shrank from {data_disparity:.1%} to {pred_disparity:.1%}.'
            icon = '🟢'

        # Find most affected groups
        most_advantaged = max(group_stats, key=lambda g: g['pred_rate'] - g['data_rate'])
        most_disadvantaged = min(group_stats, key=lambda g: g['pred_rate'] - g['data_rate'])

        results.append({
            'attribute': col,
            'groups': group_stats,
            'data_disparity': round(data_disparity, 4),
            'prediction_disparity': round(pred_disparity, 4),
            'amplification_factor': round(amplification, 4) if amplification != float('inf') else 999.0,
            'severity': severity,
            'message': message,
            'icon': icon,
            'most_advantaged_group': most_advantaged['group'],
            'most_disadvantaged_group': most_disadvantaged['group'],
            'recommendation': _get_recommendation(severity, col),
        })

    # Overall summary
    if not results:
        return {
            'results': [],
            'overall_risk': 'unknown',
            'summary': 'No valid sensitive columns found for analysis.',
        }

    amplified = [r for r in results if r['severity'] in ('critical', 'warning')]
    reduced = [r for r in results if r['severity'] == 'good']

    summary_parts = []
    if amplified:
        summary_parts.append(f'{len(amplified)} attribute(s) show bias amplification')
    if reduced:
        summary_parts.append(f'{len(reduced)} attribute(s) show bias reduction')

    return {
        'results': results,
        'overall_risk': overall_risk,
        'summary': '. '.join(summary_parts) if summary_parts else 'Bias levels are maintained.',
        'total_attributes_checked': len(results),
        'amplified_count': len(amplified),
        'reduced_count': len(reduced),
    }


def _get_recommendation(severity, col):
    """Generate remediation recommendation."""
    if severity == 'critical':
        return (
            f'CRITICAL: Retrain with sample_weight or class_weight adjustments '
            f'for "{col}". Consider adversarial debiasing or removing '
            f'"{col}" from features if it is not predictive.'
        )
    elif severity == 'warning':
        return (
            f'Monitor "{col}" bias in production. Consider threshold '
            f'calibration per group or post-processing equalised odds.'
        )
    elif severity == 'neutral':
        return f'Bias level for "{col}" is unchanged. Acceptable for most use cases.'
    else:
        return f'Model reduces bias for "{col}". Good outcome — no action needed.'
