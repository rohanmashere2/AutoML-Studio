"""
AutoML Studio — Automated Error Slice Analysis (Feature #18)
Automatically slices the test set by feature values and finds subgroups
where the model performs terribly (or exceptionally well).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score, f1_score


def analyze_error_slices(model, X_test, y_test, problem_type='classification',
                         feature_names=None, max_slices=50):
    """
    Find subgroups where the model fails or excels.

    Returns:
        dict with worst slices, best slices, and actionable insights
    """
    y_pred = model.predict(X_test)
    y_true = np.array(y_test)
    n_samples = len(y_true)
    is_clf = problem_type == 'classification'

    if hasattr(X_test, 'columns'):
        df = X_test.copy()
    else:
        cols = feature_names or [f'f_{i}' for i in range(X_test.shape[1])]
        df = pd.DataFrame(X_test, columns=cols)

    # Overall score
    if is_clf:
        overall = float(accuracy_score(y_true, y_pred))
        metric_name = 'accuracy'
    else:
        overall = float(r2_score(y_true, y_pred))
        metric_name = 'r2'

    slices = []

    for col in df.columns[:30]:
        if pd.api.types.is_numeric_dtype(df[col]):
            # Bin numeric into quartiles
            try:
                bins = pd.qcut(df[col], q=4, duplicates='drop')
                for label in bins.unique():
                    mask = bins == label
                    n = int(mask.sum())
                    if n < 10:
                        continue
                    score = _compute_score(y_true[mask], y_pred[mask], is_clf)
                    slices.append({
                        'feature': col, 'condition': str(label),
                        'n_samples': n, 'score': score,
                        'gap': round(score - overall, 4),
                    })
            except Exception:
                pass
        else:
            # Categorical: each unique value
            for val in df[col].unique()[:10]:
                mask = df[col] == val
                n = int(mask.sum())
                if n < 10:
                    continue
                score = _compute_score(y_true[mask], y_pred[mask], is_clf)
                slices.append({
                    'feature': col, 'condition': f'{col} = {val}',
                    'n_samples': n, 'score': score,
                    'gap': round(score - overall, 4),
                })

    # Sort worst first
    slices.sort(key=lambda x: x['score'])
    worst = slices[:10]
    best = sorted(slices, key=lambda x: x['score'], reverse=True)[:10]

    # Insights
    insights = []
    for s in worst[:3]:
        if s['gap'] < -0.1:
            insights.append(
                f'⚠️ When {s["condition"]}: {metric_name}={s["score"]:.1%} '
                f'(vs {overall:.1%} overall). {abs(s["gap"]):.1%} worse. '
                f'n={s["n_samples"]} samples.'
            )

    return {
        'overall_score': round(overall, 4),
        'metric': metric_name,
        'worst_slices': worst,
        'best_slices': best,
        'total_slices_analyzed': len(slices),
        'insights': insights,
        'n_underperforming': sum(1 for s in slices if s['gap'] < -0.05),
        'recommendation': _recommend_slices(worst, overall, metric_name),
    }


def _compute_score(y_true, y_pred, is_clf):
    try:
        if is_clf:
            return round(float(accuracy_score(y_true, y_pred)), 4)
        else:
            return round(float(r2_score(y_true, y_pred)), 4)
    except Exception:
        return 0.0


def _recommend_slices(worst, overall, metric):
    if not worst:
        return 'No significant underperforming slices found.'
    gap = worst[0]['score'] - overall
    if gap < -0.2:
        return (f'CRITICAL: Slice "{worst[0]["condition"]}" has {metric}='
                f'{worst[0]["score"]:.1%} — {abs(gap):.1%} below overall. '
                f'Collect more data for this subgroup or engineer targeted features.')
    elif gap < -0.1:
        return (f'Moderate weakness in "{worst[0]["condition"]}". '
                f'Consider stratified sampling or subgroup-specific tuning.')
    return 'No major subgroup weaknesses. Model performs consistently.'
