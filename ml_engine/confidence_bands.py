"""
AutoML Studio — Prediction Confidence Bands (Feature #24)
Uses multi-model ensemble to generate confidence bands for every prediction.
If models agree → high confidence. If they disagree → low confidence.
"""

import numpy as np


def compute_confidence_bands(trained_models, X_new, problem_type='classification'):
    """
    Generate confidence bands by aggregating predictions from all trained models.

    Returns:
        dict with per-prediction confidence bands and reliability flags
    """
    if not trained_models or len(trained_models) < 2:
        return {'error': 'Need at least 2 trained models'}

    names = list(trained_models.keys())
    preds = {}
    probas = {}
    for name, model in trained_models.items():
        try:
            preds[name] = np.array(model.predict(X_new))
            if hasattr(model, 'predict_proba'):
                probas[name] = model.predict_proba(X_new)
        except Exception:
            continue

    if len(preds) < 2:
        return {'error': 'Fewer than 2 models produced predictions'}

    pred_names = list(preds.keys())
    pred_matrix = np.array([preds[n] for n in pred_names])
    n_models = len(pred_names)
    n_samples = pred_matrix.shape[1]
    is_clf = problem_type == 'classification'

    results = []
    reliability_counts = {'high': 0, 'medium': 0, 'low': 0}

    for i in range(min(n_samples, 500)):
        sample = {'index': i}

        if is_clf:
            col_preds = pred_matrix[:, i]
            vals, counts = np.unique(col_preds, return_counts=True)
            majority = vals[np.argmax(counts)]
            agreement = float(counts.max() / n_models)

            sample['point_prediction'] = _s(majority)
            sample['agreement_rate'] = round(agreement, 4)
            sample['models_agree'] = int(counts.max())
            sample['models_total'] = n_models

            # Average probability from models that support predict_proba
            if probas:
                avg_proba = np.mean([probas[n][i] for n in probas], axis=0)
                sample['avg_confidence'] = round(float(np.max(avg_proba)), 4)
                sample['class_probabilities'] = [round(float(p), 4) for p in avg_proba]
        else:
            col_preds = pred_matrix[:, i].astype(float)
            mean_pred = float(np.mean(col_preds))
            std_pred = float(np.std(col_preds))

            sample['point_prediction'] = round(mean_pred, 4)
            sample['lower_bound'] = round(mean_pred - 2 * std_pred, 4)
            sample['upper_bound'] = round(mean_pred + 2 * std_pred, 4)
            sample['band_width'] = round(4 * std_pred, 4)
            sample['model_std'] = round(std_pred, 4)

            # Agreement as normalised inverse of std
            y_range = float(np.ptp(pred_matrix)) if np.ptp(pred_matrix) > 0 else 1.0
            agreement = max(0, 1 - (std_pred / y_range * 3))

        # Reliability classification
        if is_clf:
            threshold_high, threshold_low = 0.8, 0.5
        else:
            threshold_high, threshold_low = 0.7, 0.4
            sample['agreement_rate'] = round(agreement, 4)

        ag = sample.get('agreement_rate', agreement)
        if ag >= threshold_high:
            sample['reliability'] = 'HIGH'
            sample['reliability_icon'] = '🟢'
            reliability_counts['high'] += 1
        elif ag >= threshold_low:
            sample['reliability'] = 'MEDIUM'
            sample['reliability_icon'] = '🟡'
            reliability_counts['medium'] += 1
        else:
            sample['reliability'] = 'LOW'
            sample['reliability_icon'] = '🔴'
            reliability_counts['low'] += 1

        results.append(sample)

    total = sum(reliability_counts.values())
    return {
        'predictions': results,
        'reliability_summary': reliability_counts,
        'reliability_pcts': {
            'high': round(reliability_counts['high'] / max(total, 1) * 100, 1),
            'medium': round(reliability_counts['medium'] / max(total, 1) * 100, 1),
            'low': round(reliability_counts['low'] / max(total, 1) * 100, 1),
        },
        'n_models': n_models,
        'models_used': pred_names,
        'n_predictions': len(results),
        'recommendation': _rec(reliability_counts, total),
    }


def _rec(counts, total):
    low_pct = counts['low'] / max(total, 1) * 100
    if low_pct > 20:
        return (f'{low_pct:.0f}% of predictions have LOW reliability. '
                f'Use an ensemble or retrain with more data.')
    elif low_pct > 5:
        return (f'{low_pct:.0f}% low-reliability predictions. '
                f'Flag these for manual review in production.')
    return 'Most predictions have HIGH reliability. Safe for production deployment.'


def _s(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 4)
    return v
