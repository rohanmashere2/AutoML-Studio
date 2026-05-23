"""
AutoML Studio — Model Shelf-Life Predictor (Feature #4)
Predicts WHEN a model will become stale and need retraining.
Uses feature stability analysis and bootstrap sensitivity.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def predict_shelf_life(model, X_train, y_train, X_test, y_test,
                        problem_type='classification'):
    """
    Estimate how long the model will maintain acceptable performance.

    Returns:
        dict with estimated shelf life, feature velocities, retraining schedule
    """
    is_clf = problem_type == 'classification'
    from sklearn.metrics import accuracy_score, r2_score
    scorer = accuracy_score if is_clf else r2_score

    baseline = float(scorer(y_test, model.predict(X_test)))

    # 1. Bootstrap stability: how sensitive is the model to small data changes?
    stability = _bootstrap_stability(model, X_test, y_test, scorer, n_iter=20)

    # 2. Feature volatility: which features are most variable?
    volatility = _compute_feature_volatility(X_train, X_test)

    # 3. Simulated drift: progressively perturb test data
    drift_curve = _simulate_drift_curve(model, X_test, y_test, scorer, baseline)

    # 4. Estimate shelf life
    shelf_life = _estimate_shelf_life(stability, volatility, drift_curve, baseline)

    # 5. Feature risk ranking
    feature_risks = []
    for feat, vol in sorted(volatility.items(), key=lambda x: x[1], reverse=True)[:10]:
        feature_risks.append({
            'feature': feat,
            'volatility': round(vol, 4),
            'risk': 'HIGH' if vol > 0.3 else 'MEDIUM' if vol > 0.15 else 'LOW',
        })

    return {
        'baseline_score': round(baseline, 4),
        'stability': stability,
        'feature_risks': feature_risks,
        'drift_curve': drift_curve,
        'shelf_life': shelf_life,
        'metric': 'accuracy' if is_clf else 'r2',
    }


def _bootstrap_stability(model, X_test, y_test, scorer, n_iter=20):
    """Measure score variance under bootstrap resampling."""
    n = len(y_test)
    scores = []
    for i in range(n_iter):
        idx = np.random.RandomState(i).choice(n, n, replace=True)
        X_b = X_test.iloc[idx] if hasattr(X_test, 'iloc') else X_test[idx]
        y_b = np.array(y_test)[idx]
        try:
            score = float(scorer(y_b, model.predict(X_b)))
            scores.append(score)
        except Exception:
            pass

    if not scores:
        return {'mean': 0, 'std': 0, 'stability_grade': 'unknown'}

    mean_s = float(np.mean(scores))
    std_s = float(np.std(scores))

    if std_s < 0.01:
        grade = 'very_stable'
    elif std_s < 0.03:
        grade = 'stable'
    elif std_s < 0.06:
        grade = 'moderate'
    else:
        grade = 'unstable'

    return {
        'mean': round(mean_s, 4),
        'std': round(std_s, 4),
        'stability_grade': grade,
        'scores': [round(s, 4) for s in scores],
    }


def _compute_feature_volatility(X_train, X_test):
    """Measure distribution difference between train and test per feature."""
    volatility = {}
    cols = X_train.columns if hasattr(X_train, 'columns') else []

    for col in cols:
        if not pd.api.types.is_numeric_dtype(X_train[col]):
            continue
        try:
            train_vals = X_train[col].dropna().values
            test_vals = X_test[col].dropna().values
            if len(train_vals) < 10 or len(test_vals) < 10:
                continue
            # KS statistic: measures distribution difference
            ks_stat, _ = sp_stats.ks_2samp(train_vals, test_vals)
            volatility[col] = float(ks_stat)
        except Exception:
            pass

    return volatility


def _simulate_drift_curve(model, X_test, y_test, scorer, baseline, steps=5):
    """Progressively add noise to simulate drift over time."""
    curve = [{'drift_level': 0, 'score': round(baseline, 4)}]

    for step in range(1, steps + 1):
        noise_scale = step * 0.05  # 5%, 10%, 15%, 20%, 25%
        X_drifted = X_test.copy() if hasattr(X_test, 'copy') else pd.DataFrame(X_test)
        for col in X_drifted.select_dtypes(include='number').columns:
            std = X_drifted[col].std()
            if std > 1e-8:
                noise = np.random.RandomState(step).normal(0, std * noise_scale, len(X_drifted))
                X_drifted[col] = X_drifted[col] + noise
        try:
            score = float(scorer(np.array(y_test), model.predict(X_drifted)))
        except Exception:
            score = 0
        curve.append({
            'drift_level': step * 5,
            'score': round(score, 4),
            'drop': round(baseline - score, 4),
        })

    return curve


def _estimate_shelf_life(stability, volatility, drift_curve, baseline):
    """Estimate days until model needs retraining."""
    # Heuristic based on stability and drift sensitivity
    stability_factor = 1.0
    grade = stability.get('stability_grade', 'moderate')
    if grade == 'very_stable':
        stability_factor = 2.0
    elif grade == 'stable':
        stability_factor = 1.5
    elif grade == 'unstable':
        stability_factor = 0.5

    # How quickly does score degrade with drift?
    if len(drift_curve) >= 3:
        drop_at_10pct = drift_curve[2].get('drop', 0.05)
    else:
        drop_at_10pct = 0.05

    if drop_at_10pct < 0.02:
        drift_sensitivity = 'low'
        base_days = 180
    elif drop_at_10pct < 0.05:
        drift_sensitivity = 'moderate'
        base_days = 90
    elif drop_at_10pct < 0.1:
        drift_sensitivity = 'high'
        base_days = 45
    else:
        drift_sensitivity = 'critical'
        base_days = 14

    estimated_days = int(base_days * stability_factor)

    # Retraining schedule
    if estimated_days > 120:
        schedule = 'quarterly'
    elif estimated_days > 60:
        schedule = 'monthly'
    elif estimated_days > 21:
        schedule = 'bi-weekly'
    else:
        schedule = 'weekly'

    return {
        'estimated_days': estimated_days,
        'drift_sensitivity': drift_sensitivity,
        'recommended_schedule': schedule,
        'message': (
            f'Model estimated to maintain >{baseline - 0.05:.1%} performance for '
            f'~{estimated_days} days. Recommended retraining: {schedule}.'
        ),
        'confidence': 'estimate based on bootstrap stability and simulated drift',
    }
