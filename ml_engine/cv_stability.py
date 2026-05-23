"""
AutoML Studio — Cross-Validation Stability Report (Feature #19)
Tracks per-sample prediction stability across CV folds.
Identifies unstable samples that flip-flop between correct/incorrect.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def analyze_cv_stability(model, X, y, problem_type='classification', n_splits=5):
    """
    Run K-fold CV tracking per-sample predictions across all folds.

    Returns:
        dict with stability scores, unstable sample analysis, patterns
    """
    n_samples = len(y)
    y_arr = np.array(y)
    is_clf = problem_type == 'classification'

    if is_clf:
        try:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kf.split(X, y))
        except Exception:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            splits = list(kf.split(X))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kf.split(X))

    # Track per-sample: how many folds predicted correctly
    sample_correct = np.zeros(n_samples)
    sample_tested = np.zeros(n_samples)
    sample_preds = [[] for _ in range(n_samples)]

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y_arr[train_idx]
        y_val_fold = y_arr[val_idx]

        try:
            model_clone = _clone_model(model)
            model_clone.fit(X_train_fold, y_train_fold)
            preds = model_clone.predict(X_val_fold)

            for i, idx in enumerate(val_idx):
                sample_tested[idx] += 1
                if is_clf:
                    sample_correct[idx] += int(preds[i] == y_val_fold[i])
                else:
                    err = abs(preds[i] - y_val_fold[i])
                    y_range = max(y_arr.max() - y_arr.min(), 1e-8)
                    sample_correct[idx] += int(err / y_range < 0.1)
                sample_preds[idx].append(_safe(preds[i]))
        except Exception:
            continue

    # Stability scores
    tested_mask = sample_tested > 0
    stability = np.zeros(n_samples)
    stability[tested_mask] = sample_correct[tested_mask] / sample_tested[tested_mask]

    # Categorize
    categories = []
    for s in stability:
        if s >= 0.8:
            categories.append('stable_correct')
        elif s >= 0.5:
            categories.append('unstable')
        elif s > 0:
            categories.append('mostly_wrong')
        else:
            categories.append('stable_incorrect')

    n_stable = sum(1 for c in categories if c == 'stable_correct')
    n_unstable = sum(1 for c in categories if c == 'unstable')
    n_mostly_wrong = sum(1 for c in categories if c == 'mostly_wrong')
    n_stable_wrong = sum(1 for c in categories if c == 'stable_incorrect')

    # Unstable sample details (limit output)
    unstable_samples = []
    for i in range(n_samples):
        if categories[i] == 'unstable' and len(unstable_samples) < 50:
            unstable_samples.append({
                'index': int(i),
                'true_value': _safe(y_arr[i]),
                'stability_score': round(float(stability[i]), 4),
                'correct_folds': int(sample_correct[i]),
                'total_folds': int(sample_tested[i]),
                'predictions_across_folds': sample_preds[i],
            })

    return {
        'distribution': {
            'stable_correct': n_stable,
            'unstable': n_unstable,
            'mostly_wrong': n_mostly_wrong,
            'stable_incorrect': n_stable_wrong,
            'stable_pct': round(n_stable / max(n_samples, 1) * 100, 1),
            'unstable_pct': round(n_unstable / max(n_samples, 1) * 100, 1),
        },
        'mean_stability': round(float(stability[tested_mask].mean()) if tested_mask.any() else 0, 4),
        'unstable_samples': unstable_samples,
        'n_splits': n_splits,
        'n_samples': n_samples,
        'recommendation': _recommend(n_unstable, n_stable_wrong, n_samples),
    }


def _clone_model(model):
    """Clone a model preserving its parameters."""
    from sklearn.base import clone
    try:
        return clone(model)
    except Exception:
        return model.__class__(**model.get_params())


def _recommend(unstable, wrong, total):
    unstable_pct = unstable / max(total, 1) * 100
    if unstable_pct > 20:
        return (f'{unstable_pct:.0f}% of samples are unstable across folds. '
                f'Model performance is unreliable. Consider: more data, regularisation, '
                f'or ensemble methods.')
    elif unstable_pct > 10:
        return (f'{unstable_pct:.0f}% unstable samples. Monitor these in production. '
                f'An ensemble may stabilise predictions.')
    return f'Only {unstable_pct:.0f}% unstable. Model predictions are stable and reliable.'


def _safe(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 4)
    return v
