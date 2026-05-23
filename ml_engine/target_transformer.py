"""
AutoML Studio — Target Distribution Auto-Transformer (Feature #16)
For regression, detects if the target needs log/sqrt/Box-Cox transformation
and auto-applies the best one to improve model performance.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


def analyze_target_distribution(y, target_name='target'):
    """Analyze target distribution and recommend transformations."""
    y = np.array(y, dtype=float)
    y_clean = y[~np.isnan(y)]
    if len(y_clean) < 20:
        return {'error': 'Not enough non-null target values'}

    # Normality test
    sample = y_clean[:5000] if len(y_clean) > 5000 else y_clean
    try:
        stat, p_value = sp_stats.shapiro(sample[:500])
    except Exception:
        stat, p_value = 0, 0

    skewness = float(sp_stats.skew(y_clean))
    kurtosis = float(sp_stats.kurtosis(y_clean))
    is_normal = p_value > 0.05 and abs(skewness) < 0.5

    return {
        'target_name': target_name,
        'n_samples': len(y_clean),
        'mean': round(float(y_clean.mean()), 4),
        'std': round(float(y_clean.std()), 4),
        'min': round(float(y_clean.min()), 4),
        'max': round(float(y_clean.max()), 4),
        'skewness': round(skewness, 4),
        'kurtosis': round(kurtosis, 4),
        'shapiro_p_value': round(float(p_value), 6),
        'is_normal': is_normal,
        'has_negatives': bool(y_clean.min() < 0),
        'has_zeros': bool(np.any(y_clean == 0)),
        'recommendation': _recommend_transform(skewness, is_normal, y_clean.min()),
    }


def auto_transform_target(X, y, problem_type='regression', cv=3):
    """
    Try multiple target transforms and pick the best one via quick CV.

    Returns:
        dict with best transform, before/after scores, transform params
    """
    if problem_type != 'regression':
        return {'error': 'Target transformation only applies to regression'}

    y = np.array(y, dtype=float)
    X_arr = np.array(X) if not isinstance(X, np.ndarray) else X

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
    scoring = 'r2'

    transforms = {'none': {'y': y, 'inverse': lambda x: x}}

    # Log transform (only if all positive)
    if y.min() > 0:
        transforms['log'] = {'y': np.log(y), 'inverse': np.exp}
        transforms['log1p'] = {'y': np.log1p(y), 'inverse': np.expm1}

    # Square root (only if non-negative)
    if y.min() >= 0:
        transforms['sqrt'] = {'y': np.sqrt(y), 'inverse': lambda x: x ** 2}

    # Box-Cox (only if all positive)
    if y.min() > 0:
        try:
            y_bc, lmbda = sp_stats.boxcox(y)
            transforms['boxcox'] = {
                'y': y_bc,
                'inverse': lambda x, l=lmbda: sp_stats.inv_boxcox(x, l),
                'lambda': lmbda,
            }
        except Exception:
            pass

    # Yeo-Johnson (works with negatives too)
    try:
        y_yj, lmbda = sp_stats.yeojohnson(y)
        transforms['yeojohnson'] = {
            'y': y_yj,
            'inverse': lambda x, l=lmbda: _inv_yeojohnson(x, l),
            'lambda': lmbda,
        }
    except Exception:
        pass

    # Evaluate each
    results = {}
    for name, t in transforms.items():
        try:
            scores = cross_val_score(model, X_arr, t['y'], cv=cv, scoring=scoring, n_jobs=-1)
            results[name] = {
                'mean_score': round(float(scores.mean()), 4),
                'std_score': round(float(scores.std()), 4),
                'skewness_after': round(float(sp_stats.skew(t['y'])), 4),
            }
            if 'lambda' in t:
                results[name]['lambda'] = round(float(t['lambda']), 4)
        except Exception:
            results[name] = {'mean_score': -999, 'std_score': 0, 'skewness_after': 0}

    # Pick best
    best_name = max(results, key=lambda k: results[k]['mean_score'])
    best_score = results[best_name]['mean_score']
    none_score = results.get('none', {}).get('mean_score', -999)
    improvement = round(best_score - none_score, 4)

    return {
        'best_transform': best_name,
        'improvement': improvement,
        'worth_it': improvement > 0.01,
        'original_score': none_score,
        'best_score': best_score,
        'all_results': results,
        'transformed_y': transforms[best_name]['y'].tolist() if best_name != 'none' else None,
        'recommendation': (
            f'Apply "{best_name}" transform for +{improvement:.1%} R² improvement.'
            if improvement > 0.01 else
            'No transformation improves performance. Keep raw target values.'
        ),
    }


def _recommend_transform(skewness, is_normal, min_val):
    if is_normal:
        return 'Target is approximately normal. No transformation needed.'
    if abs(skewness) > 2 and min_val > 0:
        return 'Highly right-skewed. Log or Box-Cox transformation strongly recommended.'
    if abs(skewness) > 1:
        return 'Moderately skewed. Try sqrt or Yeo-Johnson transformation.'
    return 'Slight skew. Transformation may give marginal improvement.'


def _inv_yeojohnson(y, lmbda):
    """Inverse Yeo-Johnson transform."""
    y = np.array(y, dtype=float)
    result = np.zeros_like(y)
    pos = y >= 0
    neg = ~pos
    if lmbda != 0:
        result[pos] = np.power(y[pos] * lmbda + 1, 1 / lmbda) - 1
    else:
        result[pos] = np.exp(y[pos]) - 1
    if lmbda != 2:
        result[neg] = 1 - np.power(-(2 - lmbda) * y[neg] + 1, 1 / (2 - lmbda))
    else:
        result[neg] = 1 - np.exp(-y[neg])
    return result
