"""
AutoML Studio — Sample Difficulty Scorer (Feature #12)
Rates every sample from "easy" (all models agree) to "hard" (all models fail).
Identifies mislabelled candidates, boundary cases, and common hard-sample patterns.
"""

import numpy as np
import pandas as pd


def score_sample_difficulty(trained_models, X_test, y_test, feature_names=None):
    """
    Score each test sample by difficulty based on multi-model agreement.

    Args:
        trained_models: dict {model_name: fitted_model}
        X_test: feature matrix
        y_test: true labels / values
        feature_names: list of feature column names

    Returns:
        dict with per-sample scores, distribution summary, hard-sample analysis
    """
    if not trained_models:
        return {'error': 'No trained models provided'}

    n_samples = len(y_test)
    model_names = list(trained_models.keys())
    n_models = len(model_names)

    # Collect predictions from all models
    all_predictions = {}
    for name, model in trained_models.items():
        try:
            preds = model.predict(X_test)
            all_predictions[name] = preds
        except Exception:
            continue

    if len(all_predictions) < 2:
        return {'error': 'Need at least 2 working models for difficulty scoring'}

    pred_matrix = np.array(list(all_predictions.values()))  # (n_models, n_samples)
    y_arr = np.array(y_test)
    working_models = list(all_predictions.keys())
    n_working = len(working_models)

    # Determine if classification or regression
    is_classification = _is_classification(y_arr)

    if is_classification:
        scores, details = _score_classification(pred_matrix, y_arr, n_working)
    else:
        scores, details = _score_regression(pred_matrix, y_arr, n_working)

    # Build per-sample results (limit to avoid huge response)
    per_sample = []
    for i in range(min(n_samples, 500)):
        sample = {
            'index': int(i),
            'difficulty': round(float(scores[i]), 4),
            'true_value': _safe_val(y_arr[i]),
            'category': details['categories'][i],
        }
        if is_classification:
            sample['correct_count'] = int(details['correct_counts'][i])
            sample['total_models'] = n_working
            sample['majority_prediction'] = _safe_val(details['majority_preds'][i])
        per_sample.append(sample)

    # Difficulty distribution
    easy = int(np.sum(scores >= 0.8))
    medium = int(np.sum((scores >= 0.4) & (scores < 0.8)))
    hard = int(np.sum((scores >= 0.1) & (scores < 0.4)))
    very_hard = int(np.sum(scores < 0.1))

    hardest_samples = []
    for item in sorted(per_sample, key=lambda x: x['difficulty'])[:50]:
        reason = ''
        if is_classification:
            correct = int(item.get('correct_count', 0))
            total = int(item.get('total_models', n_working))
            if correct == 0:
                reason = 'All models disagreed with the true label'
            else:
                reason = f'{correct}/{total} models agreed with the true label'
        else:
            reason = 'High prediction variance across models'

        hardest_samples.append({
            'index': item['index'],
            'difficulty': item['difficulty'],
            'category': item['category'],
            'reason': reason,
        })

    # Suspected mislabels (all models unanimously disagree with ground truth)
    mislabel_candidates = []
    if is_classification:
        for i in range(n_samples):
            if details['correct_counts'][i] == 0 and n_working >= 3:
                mislabel_candidates.append({
                    'index': int(i),
                    'true_label': _safe_val(y_arr[i]),
                    'all_models_predict': _safe_val(details['majority_preds'][i]),
                    'confidence': 'HIGH — all models unanimously disagree',
                })
            if len(mislabel_candidates) >= 20:
                break

    # Hard sample pattern analysis
    hard_patterns = _analyze_hard_patterns(X_test, scores, feature_names, threshold=0.3)

    return {
        'scores': [round(float(s), 4) for s in scores],
        'per_sample': per_sample,
        'distribution': {
            'easy': easy,
            'medium': medium,
            'hard': hard,
            'very_hard': very_hard,
            'easy_pct': round(easy / max(n_samples, 1) * 100, 1),
            'hard_pct': round((hard + very_hard) / max(n_samples, 1) * 100, 1),
        },
        'easy_count': easy,
        'medium_count': medium,
        'hard_count': hard + very_hard,
        'mislabel_count': len(mislabel_candidates),
        'hardest_samples': hardest_samples,
        'hard_samples': hardest_samples,
        'summary': {
            'mean_difficulty': round(float(np.mean(scores)), 4),
            'median_difficulty': round(float(np.median(scores)), 4),
            'n_samples': n_samples,
            'n_models_used': n_working,
            'models_used': working_models,
            'is_classification': is_classification,
        },
        'mislabel_candidates': mislabel_candidates,
        'hard_patterns': hard_patterns,
    }


def _score_classification(pred_matrix, y_true, n_models):
    """Score classification samples by agreement rate."""
    n_samples = len(y_true)
    correct_counts = np.zeros(n_samples)
    majority_preds = []

    for i in range(n_samples):
        preds_i = pred_matrix[:, i]
        correct_counts[i] = np.sum(preds_i == y_true[i])
        # Majority vote
        values, counts = np.unique(preds_i, return_counts=True)
        majority_preds.append(values[np.argmax(counts)])

    # Score = fraction of models that got it right
    scores = correct_counts / max(n_models, 1)

    categories = []
    for s in scores:
        if s >= 0.8:
            categories.append('easy')
        elif s >= 0.5:
            categories.append('medium')
        elif s >= 0.2:
            categories.append('hard')
        else:
            categories.append('very_hard')

    return scores, {
        'correct_counts': correct_counts,
        'majority_preds': majority_preds,
        'categories': categories,
    }


def _score_regression(pred_matrix, y_true, n_models):
    """Score regression samples by prediction variance / error consistency."""
    n_samples = len(y_true)
    # Mean absolute error per sample across models
    errors = np.abs(pred_matrix - y_true[np.newaxis, :])
    mean_error = errors.mean(axis=0)
    pred_std = pred_matrix.std(axis=0)

    # Normalize errors to 0-1 range (1 = easy)
    max_err = np.percentile(mean_error, 95) if len(mean_error) > 0 else 1.0
    max_err = max(max_err, 1e-6)
    scores = 1.0 - np.clip(mean_error / max_err, 0, 1)

    categories = []
    for s in scores:
        if s >= 0.8:
            categories.append('easy')
        elif s >= 0.5:
            categories.append('medium')
        elif s >= 0.2:
            categories.append('hard')
        else:
            categories.append('very_hard')

    return scores, {'categories': categories, 'correct_counts': scores,
                    'majority_preds': pred_matrix.mean(axis=0)}


def _analyze_hard_patterns(X, scores, feature_names, threshold=0.3):
    """Find common patterns among hard samples."""
    patterns = []
    if not hasattr(X, 'iloc'):
        return patterns

    hard_mask = scores < threshold
    n_hard = int(hard_mask.sum())
    if n_hard < 5:
        return patterns

    try:
        hard_X = X[hard_mask]
        easy_X = X[~hard_mask]

        cols = feature_names if feature_names else list(X.columns) if hasattr(X, 'columns') else []
        for col in cols[:20]:
            if col not in X.columns:
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            hard_mean = float(hard_X[col].mean())
            easy_mean = float(easy_X[col].mean())
            overall_std = float(X[col].std())
            if overall_std < 1e-8:
                continue
            diff = abs(hard_mean - easy_mean) / overall_std
            if diff > 0.5:
                direction = 'higher' if hard_mean > easy_mean else 'lower'
                patterns.append({
                    'feature': col,
                    'hard_mean': round(hard_mean, 4),
                    'easy_mean': round(easy_mean, 4),
                    'effect_size': round(diff, 4),
                    'direction': direction,
                    'insight': f'Hard samples have {direction} "{col}" (effect size {diff:.2f}σ)',
                })
        patterns.sort(key=lambda x: x['effect_size'], reverse=True)
    except Exception:
        pass

    return patterns[:10]


def _is_classification(y):
    """Detect if target is classification."""
    if hasattr(y, 'dtype') and y.dtype == 'object':
        return True
    n_unique = len(np.unique(y))
    return n_unique <= 30 and n_unique / max(len(y), 1) < 0.05


def _safe_val(v):
    """Convert numpy types to JSON-safe Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v
