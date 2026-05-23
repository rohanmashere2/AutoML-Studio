"""
AutoML Studio — Learning Curve Predictor (Feature #13)
Trains on increasing data fractions, fits a power-law curve, and
extrapolates performance at 2x, 5x, 10x current data size.
"""

import numpy as np
from sklearn.model_selection import learning_curve


def predict_learning_curve(model, X, y, problem_type='classification', cv=5):
    """
    Generate learning curve and extrapolate future performance.

    Returns:
        dict with actual curve, extrapolation, plateau detection
    """
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    n_samples = len(y)
    fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    fractions = [f for f in fractions if int(f * n_samples) >= 10]

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, train_sizes=fractions, cv=min(cv, 5),
            scoring=scoring, n_jobs=-1, random_state=42
        )
    except Exception as e:
        return {'error': f'Learning curve computation failed: {str(e)}'}

    train_means = train_scores.mean(axis=1)
    val_means = val_scores.mean(axis=1)
    val_stds = val_scores.std(axis=1)

    actual_curve = [
        {'n_samples': int(train_sizes[i]),
         'fraction': round(float(train_sizes[i] / n_samples), 2),
         'train_score': round(float(train_means[i]), 4),
         'val_score': round(float(val_means[i]), 4),
         'val_std': round(float(val_stds[i]), 4)}
        for i in range(len(train_sizes))
    ]

    # Fit power-law: score = a - b * n^(-c)
    extrapolation = _extrapolate(train_sizes, val_means, n_samples)

    # Plateau detection
    if len(val_means) >= 3:
        recent_improvement = val_means[-1] - val_means[-3]
        if abs(recent_improvement) < 0.005:
            plateau = True
            plateau_msg = (
                'Model has PLATEAUED — more data is unlikely to help significantly. '
                'Focus on feature engineering or model architecture changes.'
            )
        else:
            plateau = False
            plateau_msg = (
                'Model is still improving with more data. '
                'Collecting more samples will likely boost performance.'
            )
    else:
        plateau = False
        plateau_msg = 'Not enough data points to detect plateau.'

    # Overfitting gap
    gap = float(train_means[-1] - val_means[-1])
    if gap > 0.1:
        gap_msg = f'Overfitting gap: {gap:.1%}. Regularisation or more data recommended.'
    elif gap > 0.05:
        gap_msg = f'Moderate gap: {gap:.1%}. Acceptable but monitor on new data.'
    else:
        gap_msg = f'Minimal gap: {gap:.1%}. Good generalisation.'

    return {
        'actual_curve': actual_curve,
        'extrapolation': extrapolation,
        'plateau_detected': plateau,
        'plateau_message': plateau_msg,
        'current_score': round(float(val_means[-1]), 4),
        'overfitting_gap': round(gap, 4),
        'overfitting_message': gap_msg,
        'n_current_samples': n_samples,
        'metric': scoring,
    }


def _extrapolate(sizes, scores, current_n):
    """Fit power law and predict at larger data sizes."""
    from scipy.optimize import curve_fit

    def power_law(n, a, b, c):
        return a - b * np.power(n, -c)

    try:
        popt, _ = curve_fit(power_law, sizes.astype(float), scores,
                            p0=[scores[-1] + 0.05, 1.0, 0.5],
                            maxfev=5000, bounds=([0, 0, 0.01], [1.5, 100, 5.0]))
        a, b, c = popt

        predictions = []
        for multiplier in [2, 5, 10, 20]:
            future_n = current_n * multiplier
            pred_score = float(power_law(future_n, a, b, c))
            pred_score = min(pred_score, 1.0)
            improvement = pred_score - float(scores[-1])
            predictions.append({
                'multiplier': f'{multiplier}x',
                'n_samples': future_n,
                'predicted_score': round(pred_score, 4),
                'estimated_improvement': round(improvement, 4),
                'worth_it': improvement > 0.01,
            })

        asymptote = round(float(a), 4)
        return {
            'predictions': predictions,
            'asymptote': min(asymptote, 1.0),
            'asymptote_message': f'Theoretical maximum with infinite data: {min(asymptote, 1.0):.1%}',
            'model_fit': 'power_law',
        }
    except Exception:
        return {
            'predictions': [],
            'asymptote': None,
            'asymptote_message': 'Could not fit extrapolation model.',
            'model_fit': 'failed',
        }
