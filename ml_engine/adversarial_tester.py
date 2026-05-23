"""
AutoML Studio — Adversarial Stress Test Suite (Feature #3)
Automatically attacks your model to find weaknesses before deployment.
Tests: feature perturbation, boundary probing, distribution shift, missing data attack.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score


def run_stress_test(model, X_test, y_test, problem_type='classification',
                     feature_names=None):
    """
    Run full adversarial stress test suite.

    Returns:
        dict with per-test results, robustness score, vulnerability report
    """
    y_true = np.array(y_test)
    is_clf = problem_type == 'classification'

    if hasattr(X_test, 'columns'):
        df = X_test.copy()
        cols = list(df.columns)
    else:
        cols = feature_names or [f'f_{i}' for i in range(X_test.shape[1])]
        df = pd.DataFrame(X_test, columns=cols)

    baseline = _baseline_score(model, df, y_true, is_clf)
    tests = []

    # Test 1: Feature perturbation attack
    tests.append(_perturbation_attack(model, df, y_true, is_clf, baseline))

    # Test 2: Missing data attack
    tests.append(_missing_data_attack(model, df, y_true, is_clf, baseline))

    # Test 3: Distribution shift simulation
    tests.append(_distribution_shift(model, df, y_true, is_clf, baseline))

    # Test 4: Boundary probing (classification only)
    if is_clf and hasattr(model, 'predict_proba'):
        tests.append(_boundary_probe(model, df, y_true, baseline))

    # Test 5: Feature knockout
    tests.append(_feature_knockout(model, df, y_true, is_clf, baseline))

    # Overall robustness score (A-F)
    drops = [t.get('score_drop', 0) for t in tests]
    avg_drop = np.mean(drops) if drops else 0
    grade, grade_msg = _compute_grade(avg_drop)

    return {
        'baseline_score': round(baseline, 4),
        'tests': tests,
        'robustness_grade': grade,
        'robustness_message': grade_msg,
        'average_score_drop': round(avg_drop, 4),
        'metric': 'accuracy' if is_clf else 'r2',
        'vulnerabilities': [t for t in tests if t.get('score_drop', 0) > 0.05],
    }


def _baseline_score(model, X, y, is_clf):
    try:
        preds = model.predict(X)
        return float(accuracy_score(y, preds) if is_clf else r2_score(y, preds))
    except Exception:
        return 0.0


def _perturbation_attack(model, X, y, is_clf, baseline):
    """Add small noise to features and measure degradation."""
    results = []
    for noise_level in [0.01, 0.05, 0.1, 0.2]:
        X_noisy = X.copy()
        for col in X.select_dtypes(include='number').columns:
            std = X[col].std()
            if std > 1e-8:
                noise = np.random.RandomState(42).normal(0, std * noise_level, len(X))
                X_noisy[col] = X[col] + noise
        try:
            preds = model.predict(X_noisy)
            score = float(accuracy_score(y, preds) if is_clf else r2_score(y, preds))
        except Exception:
            score = 0
        results.append({'noise_level': f'{noise_level:.0%}', 'score': round(score, 4)})

    worst_drop = baseline - results[-1]['score']
    return {
        'name': 'Feature Perturbation Attack',
        'icon': '🎯',
        'description': f'Added random noise at 1%-20% levels. Score dropped by {worst_drop:.1%} at max noise.',
        'score_drop': round(worst_drop, 4),
        'details': results,
        'severity': 'high' if worst_drop > 0.1 else 'medium' if worst_drop > 0.05 else 'low',
    }


def _missing_data_attack(model, X, y, is_clf, baseline):
    """Randomly remove features and measure degradation."""
    results = []
    for miss_rate in [0.05, 0.1, 0.2, 0.3]:
        X_miss = X.copy()
        mask = np.random.RandomState(42).random(X_miss.shape) < miss_rate
        numeric_cols = X_miss.select_dtypes(include='number').columns
        for col in numeric_cols:
            col_mask = mask[:, list(X_miss.columns).index(col)] if col in X_miss.columns else np.zeros(len(X_miss), dtype=bool)
            X_miss.loc[col_mask, col] = X_miss[col].median()
        try:
            preds = model.predict(X_miss)
            score = float(accuracy_score(y, preds) if is_clf else r2_score(y, preds))
        except Exception:
            score = 0
        results.append({'missing_rate': f'{miss_rate:.0%}', 'score': round(score, 4)})

    worst_drop = baseline - results[-1]['score']
    return {
        'name': 'Missing Data Attack',
        'icon': '🕳️',
        'description': f'Randomly replaced 5%-30% of values with medians. Score dropped by {worst_drop:.1%}.',
        'score_drop': round(worst_drop, 4),
        'details': results,
        'severity': 'high' if worst_drop > 0.1 else 'medium' if worst_drop > 0.05 else 'low',
    }


def _distribution_shift(model, X, y, is_clf, baseline):
    """Simulate distribution shift by scaling features."""
    results = []
    for shift in [0.9, 0.8, 0.7]:
        X_shifted = X.copy()
        for col in X.select_dtypes(include='number').columns:
            X_shifted[col] = X[col] * shift
        try:
            preds = model.predict(X_shifted)
            score = float(accuracy_score(y, preds) if is_clf else r2_score(y, preds))
        except Exception:
            score = 0
        pct = int((1 - shift) * 100)
        results.append({'shift': f'-{pct}%', 'score': round(score, 4)})

    worst_drop = baseline - results[-1]['score']
    return {
        'name': 'Distribution Shift Simulation',
        'icon': '📉',
        'description': f'Simulated 10%-30% feature decrease. Score dropped by {worst_drop:.1%}.',
        'score_drop': round(worst_drop, 4),
        'details': results,
        'severity': 'high' if worst_drop > 0.15 else 'medium' if worst_drop > 0.08 else 'low',
    }


def _boundary_probe(model, X, y, baseline):
    """Find samples near decision boundary."""
    proba = model.predict_proba(X)
    max_proba = np.max(proba, axis=1)
    boundary_mask = max_proba < 0.6
    n_boundary = int(boundary_mask.sum())
    boundary_pct = n_boundary / max(len(y), 1) * 100

    if n_boundary > 0:
        boundary_acc = float(accuracy_score(y[boundary_mask], model.predict(X[boundary_mask]
                             if not hasattr(X, 'iloc') else X.iloc[boundary_mask])))
    else:
        boundary_acc = baseline

    return {
        'name': 'Boundary Probing',
        'icon': '🔍',
        'description': f'{n_boundary} samples ({boundary_pct:.1f}%) near decision boundary (<60% confidence). '
                       f'Boundary accuracy: {boundary_acc:.1%} vs overall {baseline:.1%}.',
        'score_drop': round(baseline - boundary_acc, 4),
        'details': {'boundary_count': n_boundary, 'boundary_pct': round(boundary_pct, 1),
                    'boundary_accuracy': round(boundary_acc, 4)},
        'severity': 'high' if boundary_pct > 20 else 'medium' if boundary_pct > 10 else 'low',
    }


def _feature_knockout(model, X, y, is_clf, baseline):
    """Remove one feature at a time and measure impact."""
    knockouts = []
    numeric_cols = X.select_dtypes(include='number').columns.tolist()

    for col in numeric_cols[:15]:
        X_ko = X.copy()
        X_ko[col] = X_ko[col].median()
        try:
            preds = model.predict(X_ko)
            score = float(accuracy_score(y, preds) if is_clf else r2_score(y, preds))
            drop = baseline - score
            knockouts.append({'feature': col, 'score_without': round(score, 4),
                              'drop': round(drop, 4)})
        except Exception:
            pass

    knockouts.sort(key=lambda x: x['drop'], reverse=True)
    worst = knockouts[0] if knockouts else {'feature': 'none', 'drop': 0}

    return {
        'name': 'Feature Knockout Test',
        'icon': '🥊',
        'description': f'Most critical feature: "{worst["feature"]}" — removing it drops score by {worst["drop"]:.1%}.',
        'score_drop': round(worst['drop'], 4),
        'details': knockouts[:10],
        'severity': 'high' if worst['drop'] > 0.1 else 'medium' if worst['drop'] > 0.05 else 'low',
    }


def _compute_grade(avg_drop):
    if avg_drop < 0.02:
        return 'A', 'Excellent robustness. Model handles adversarial conditions well.'
    elif avg_drop < 0.05:
        return 'B', 'Good robustness. Minor vulnerabilities exist.'
    elif avg_drop < 0.1:
        return 'C', 'Fair robustness. Several vulnerabilities need attention.'
    elif avg_drop < 0.2:
        return 'D', 'Poor robustness. Model is fragile — significant risk in production.'
    else:
        return 'F', 'Critical vulnerability. Model fails under stress. Do NOT deploy without fixes.'
