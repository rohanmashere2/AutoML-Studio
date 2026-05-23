"""
AutoML Studio — Model Disagreement Analyzer (Feature #14)
Finds exactly where different trained models disagree. Disagreement zones
are high-risk regions for production deployment.
"""

import numpy as np
import pandas as pd


def analyze_disagreement(trained_models, X_test, y_test, feature_names=None):
    """
    Analyze where models disagree with each other.

    Returns:
        dict with pairwise disagreement, consensus map, danger zones
    """
    if len(trained_models) < 2:
        return {'error': 'Need at least 2 models'}

    names = list(trained_models.keys())
    preds = {}
    for name, model in trained_models.items():
        try:
            preds[name] = np.array(model.predict(X_test))
        except Exception:
            continue

    if len(preds) < 2:
        return {'error': 'Fewer than 2 models produced valid predictions'}

    pred_names = list(preds.keys())
    pred_matrix = np.array([preds[n] for n in pred_names])
    n_models, n_samples = pred_matrix.shape
    y_arr = np.array(y_test)
    is_clf = len(np.unique(y_arr)) <= 30

    # Pairwise disagreement rate
    pairwise = []
    for i in range(len(pred_names)):
        for j in range(i + 1, len(pred_names)):
            if is_clf:
                disagree_rate = float(np.mean(pred_matrix[i] != pred_matrix[j]))
            else:
                diff = np.abs(pred_matrix[i] - pred_matrix[j])
                y_range = max(y_arr.max() - y_arr.min(), 1e-8)
                disagree_rate = float(np.mean(diff / y_range > 0.1))

            pairwise.append({
                'model_a': pred_names[i],
                'model_b': pred_names[j],
                'disagreement_rate': round(disagree_rate * 100, 1),
            })

    pairwise.sort(key=lambda x: x['disagreement_rate'], reverse=True)

    # Per-sample consensus
    if is_clf:
        consensus_scores = np.zeros(n_samples)
        for i in range(n_samples):
            col_preds = pred_matrix[:, i]
            vals, counts = np.unique(col_preds, return_counts=True)
            consensus_scores[i] = counts.max() / n_models
    else:
        pred_std = pred_matrix.std(axis=0)
        y_range = max(y_arr.max() - y_arr.min(), 1e-8)
        consensus_scores = 1.0 - np.clip(pred_std / y_range, 0, 1)

    # Classify zones
    high_trust = int(np.sum(consensus_scores >= 0.8))
    medium_trust = int(np.sum((consensus_scores >= 0.5) & (consensus_scores < 0.8)))
    danger_zone = int(np.sum(consensus_scores < 0.5))

    # Find features that characterise danger zones
    danger_patterns = []
    if hasattr(X_test, 'columns') and danger_zone >= 5:
        danger_mask = consensus_scores < 0.5
        safe_mask = consensus_scores >= 0.8
        cols = feature_names or list(X_test.columns)
        for col in cols[:15]:
            if col not in X_test.columns or not pd.api.types.is_numeric_dtype(X_test[col]):
                continue
            d_mean = float(X_test.loc[danger_mask, col].mean())
            s_mean = float(X_test.loc[safe_mask, col].mean()) if safe_mask.sum() > 0 else float(X_test[col].mean())
            std = float(X_test[col].std())
            if std < 1e-8:
                continue
            effect = abs(d_mean - s_mean) / std
            if effect > 0.3:
                danger_patterns.append({
                    'feature': col, 'effect_size': round(effect, 3),
                    'danger_zone_mean': round(d_mean, 4),
                    'safe_zone_mean': round(s_mean, 4),
                })
        danger_patterns.sort(key=lambda x: x['effect_size'], reverse=True)

    # Who's the outlier? Which model disagrees most?
    model_disagree_rates = {}
    for i, name in enumerate(pred_names):
        others = [pred_matrix[j] for j in range(n_models) if j != i]
        if is_clf:
            rates = [float(np.mean(pred_matrix[i] != o)) for o in others]
        else:
            rates = [float(np.mean(np.abs(pred_matrix[i] - o) / y_range > 0.1)) for o in others]
        model_disagree_rates[name] = round(float(np.mean(rates)) * 100, 1)

    most_disagreeable = max(model_disagree_rates, key=model_disagree_rates.get)

    return {
        'pairwise_disagreement': pairwise,
        'trust_zones': {
            'high_trust': high_trust,
            'medium_trust': medium_trust,
            'danger_zone': danger_zone,
            'high_trust_pct': round(high_trust / max(n_samples, 1) * 100, 1),
            'danger_zone_pct': round(danger_zone / max(n_samples, 1) * 100, 1),
        },
        'danger_zone_patterns': danger_patterns[:5],
        'model_disagreement_rates': model_disagree_rates,
        'most_disagreeable_model': most_disagreeable,
        'n_models': n_models,
        'n_samples': n_samples,
        'recommendation': _recommend(danger_zone, n_samples, most_disagreeable),
    }


def _recommend(danger, total, outlier_model):
    if danger / max(total, 1) > 0.2:
        return (f'HIGH RISK: {danger} samples ({danger/total:.0%}) are in the danger zone. '
                f'Consider using an ensemble to reduce disagreement, or collect more training data.')
    elif danger > 0:
        return (f'{danger} samples in danger zone. Model "{outlier_model}" disagrees most — '
                f'review its predictions carefully before deployment.')
    return 'All models show strong consensus. Safe for deployment.'
