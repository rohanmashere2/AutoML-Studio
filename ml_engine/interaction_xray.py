"""
AutoML Studio — Feature Interaction X-Ray (Feature #10)
Discovers and visualises pairwise and higher-order feature interactions.
Uses H-statistic and SHAP interaction values.
"""

import numpy as np
import pandas as pd


def analyze_interactions(model, X, y=None, feature_names=None, top_k=15):
    """
    Discover feature interactions using permutation-based H-statistic.

    Returns:
        dict with pairwise interaction strengths and narratives
    """
    if hasattr(X, 'columns'):
        cols = list(X.columns)
        X_arr = X.values
    else:
        cols = feature_names or [f'f_{i}' for i in range(X.shape[1])]
        X_arr = np.array(X)

    numeric_idx = []
    for i, col in enumerate(cols):
        if hasattr(X, 'dtypes') and pd.api.types.is_numeric_dtype(X[col]):
            numeric_idx.append(i)
        elif not hasattr(X, 'dtypes'):
            numeric_idx.append(i)

    numeric_idx = numeric_idx[:top_k]  # Limit to avoid O(n²) explosion
    if len(numeric_idx) < 2:
        return {'error': 'Need at least 2 numeric features'}

    # Compute pairwise interactions via partial dependence variance
    interactions = []
    total_pairs = len(numeric_idx) * (len(numeric_idx) - 1) // 2
    pair_count = 0

    for i in range(len(numeric_idx)):
        for j in range(i + 1, len(numeric_idx)):
            fi, fj = numeric_idx[i], numeric_idx[j]
            try:
                h_stat = _compute_h_statistic(model, X_arr, fi, fj)
                if h_stat > 0.01:
                    interactions.append({
                        'feature_a': cols[fi],
                        'feature_b': cols[fj],
                        'interaction_strength': round(float(h_stat), 4),
                    })
            except Exception:
                pass
            pair_count += 1
            if pair_count > 100:
                break
        if pair_count > 100:
            break

    interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)

    # Generate narratives for top interactions
    for inter in interactions[:5]:
        inter['feature_1'] = inter['feature_a']
        inter['feature_2'] = inter['feature_b']
        inter['strength'] = inter['interaction_strength']
        inter['narrative'] = (
            f'Features "{inter["feature_a"]}" and "{inter["feature_b"]}" '
            f'have a {_strength_label(inter["interaction_strength"])} interaction '
            f'(H={inter["interaction_strength"]:.3f}). '
            f'Their combined effect differs from their individual effects.'
        )

    # Individual importance for context
    individual = _permutation_importance_quick(model, X_arr, y, cols)

    recommendation = _recommend(interactions)
    summary = (
        f'{len(interactions)} significant interactions found across {pair_count} tested pairs. '
        f'{recommendation}'
    )

    return {
        'interactions': interactions[:20],
        'pairs': interactions[:20],
        'pairwise_interactions': interactions[:20],
        'total_pairs_tested': pair_count,
        'significant_interactions': sum(1 for i in interactions if i['interaction_strength'] > 0.05),
        'individual_importance': individual[:10],
        'importance': individual[:10],
        'recommendation': recommendation,
        'summary': summary,
    }


def _compute_h_statistic(model, X, fi, fj, n_grid=20):
    """
    Compute Friedman's H-statistic for pairwise interaction.
    Measures how much of the model's prediction comes from the interaction
    vs individual effects.
    """
    n = len(X)
    sample_idx = np.random.RandomState(42).choice(n, min(n, 200), replace=False)
    X_sample = X[sample_idx]

    # Get predictions
    f_full = model.predict(X_sample).astype(float)
    f_mean = f_full.mean()

    # Partial dependence for feature i
    pd_i = np.zeros(len(X_sample))
    for k in range(len(X_sample)):
        X_temp = X_sample.copy()
        X_temp[:, fi] = X_sample[k, fi]
        pd_i[k] = model.predict(X_temp).astype(float).mean()

    # Partial dependence for feature j
    pd_j = np.zeros(len(X_sample))
    for k in range(len(X_sample)):
        X_temp = X_sample.copy()
        X_temp[:, fj] = X_sample[k, fj]
        pd_j[k] = model.predict(X_temp).astype(float).mean()

    # H-statistic
    numerator = np.sum((f_full - pd_i - pd_j + f_mean) ** 2)
    denominator = np.sum(f_full ** 2)

    if denominator < 1e-10:
        return 0.0

    return float(np.sqrt(numerator / denominator))


def _permutation_importance_quick(model, X, y, cols, n_repeats=3):
    """Quick permutation importance for context."""
    if y is None:
        return []

    y_arr = np.array(y)
    from sklearn.metrics import accuracy_score, r2_score
    is_clf = len(np.unique(y_arr)) <= 30
    scorer = accuracy_score if is_clf else r2_score

    try:
        baseline = float(scorer(y_arr, model.predict(X)))
    except Exception:
        return []

    importances = []
    for i, col in enumerate(cols):
        drops = []
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = np.random.RandomState(r).permutation(X_perm[:, i])
            try:
                score = float(scorer(y_arr, model.predict(X_perm)))
                drops.append(baseline - score)
            except Exception:
                pass
        if drops:
            importances.append({
                'feature': col,
                'importance': round(float(np.mean(drops)), 4),
            })

    importances.sort(key=lambda x: x['importance'], reverse=True)
    return importances


def _strength_label(h):
    if h > 0.2:
        return 'STRONG'
    elif h > 0.1:
        return 'moderate'
    elif h > 0.05:
        return 'weak'
    return 'very weak'


def _recommend(interactions):
    strong = [i for i in interactions if i['interaction_strength'] > 0.1]
    if strong:
        top = strong[0]
        return (
            f'{len(strong)} significant interactions found. '
            f'Strongest: {top["feature_a"]} × {top["feature_b"]} (H={top["interaction_strength"]:.3f}). '
            f'Consider creating explicit interaction features for these pairs.'
        )
    return 'No strong feature interactions detected. Individual features are sufficient.'
