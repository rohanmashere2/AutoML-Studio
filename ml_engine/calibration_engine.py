"""
AutoML Studio — Prediction Confidence Calibration
Measures and fixes prediction confidence calibration using
Platt Scaling or Isotonic Regression.
"""

import numpy as np
import pandas as pd


def compute_calibration(model, X_test, y_test, n_bins=10):
    """
    Compute calibration curve and Expected Calibration Error (ECE).

    Returns:
        dict: Calibration curve data, ECE, reliability score badge
    """
    if not hasattr(model, 'predict_proba'):
        return {'error': 'Model does not support predict_proba — calibration requires probability outputs.'}

    try:
        from sklearn.calibration import calibration_curve

        probs = model.predict_proba(X_test)

        # Handle binary vs multiclass
        if probs.shape[1] == 2:
            prob_positive = probs[:, 1]
            y_binary = y_test
        else:
            # For multiclass, compute per-class calibration for predicted class
            y_pred = model.predict(X_test)
            prob_positive = np.max(probs, axis=1)
            y_binary = (y_pred == y_test).astype(int)

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_binary, prob_positive, n_bins=n_bins, strategy='uniform'
        )

        # Expected Calibration Error
        ece = _compute_ece(prob_positive, y_binary, n_bins)

        # Brier Score
        brier = float(np.mean((prob_positive - y_binary.astype(float)) ** 2))

        # Reliability score (0-100, higher is better)
        reliability_score = max(0, round(100 - ece * 1000))

        # Calibration quality badge
        if ece < 0.03:
            badge = 'excellent'
            badge_color = 'green'
        elif ece < 0.07:
            badge = 'good'
            badge_color = 'blue'
        elif ece < 0.15:
            badge = 'fair'
            badge_color = 'yellow'
        else:
            badge = 'poor'
            badge_color = 'red'

        # Confidence distribution histogram
        conf_hist, conf_edges = np.histogram(prob_positive, bins=20, range=(0, 1))

        return {
            'calibration_curve': {
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
            },
            'ece': round(ece, 4),
            'brier_score': round(brier, 4),
            'is_well_calibrated': ece < 0.05,
            'reliability_score': reliability_score,
            'badge': badge,
            'badge_color': badge_color,
            'confidence_distribution': {
                'counts': conf_hist.tolist(),
                'edges': conf_edges.tolist(),
            },
            'n_samples': len(y_test),
            'recommendation': _calibration_recommendation(ece, badge),
        }
    except Exception as e:
        return {'error': f'Calibration computation failed: {str(e)}'}


def auto_calibrate(model, X_train, y_train, X_test, y_test, method='auto'):
    """
    Apply Platt Scaling or Isotonic Regression to fix miscalibration.

    Args:
        model: trained model
        method: 'platt', 'isotonic', or 'auto' (try both, pick best)

    Returns:
        dict with calibrated model, before/after comparison
    """
    from sklearn.calibration import CalibratedClassifierCV

    if not hasattr(model, 'predict_proba'):
        return {'error': 'Model does not support predict_proba'}

    try:
        # Before calibration
        before = compute_calibration(model, X_test, y_test)
        before_ece = before.get('ece', 1.0)

        results = {}

        methods_to_try = []
        if method == 'auto':
            methods_to_try = ['sigmoid', 'isotonic']
        elif method == 'platt':
            methods_to_try = ['sigmoid']
        else:
            methods_to_try = ['isotonic']

        best_method = None
        best_ece = before_ece
        best_model = None

        for cal_method in methods_to_try:
            try:
                calibrated = CalibratedClassifierCV(model, method=cal_method, cv=3)
                calibrated.fit(X_train, y_train)

                after = compute_calibration(calibrated, X_test, y_test)
                cal_ece = after.get('ece', 1.0)

                results[cal_method] = {
                    'ece': cal_ece,
                    'reliability_score': after.get('reliability_score', 0),
                    'badge': after.get('badge', 'unknown'),
                    'improvement': round(before_ece - cal_ece, 4),
                }

                if cal_ece < best_ece:
                    best_ece = cal_ece
                    best_method = cal_method
                    best_model = calibrated
            except Exception:
                continue

        if best_model is None:
            return {
                'success': False,
                'message': 'Calibration did not improve the model.',
                'before_ece': before_ece,
            }

        after_metrics = compute_calibration(best_model, X_test, y_test)

        return {
            'success': True,
            'calibrated_model': best_model,
            'method_used': best_method,
            'before': {
                'ece': before_ece,
                'reliability_score': before.get('reliability_score', 0),
                'badge': before.get('badge', 'unknown'),
            },
            'after': {
                'ece': best_ece,
                'reliability_score': after_metrics.get('reliability_score', 0),
                'badge': after_metrics.get('badge', 'unknown'),
            },
            'improvement': round(before_ece - best_ece, 4),
            'improvement_pct': f'{(before_ece - best_ece) / max(before_ece, 0.001) * 100:.1f}%',
            'methods_tested': results,
        }
    except Exception as e:
        return {'error': f'Auto-calibration failed: {str(e)}'}


def _compute_ece(probs, labels, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(probs)

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = (probs >= bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            continue

        avg_confidence = probs[mask].mean()
        avg_accuracy = labels.astype(float).values[mask].mean() if hasattr(labels, 'values') else labels[mask].astype(float).mean()

        ece += (n_in_bin / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def _calibration_recommendation(ece, badge):
    """Generate recommendation based on calibration quality."""
    if badge == 'excellent':
        return 'Calibration is excellent — no action needed. Confidence scores are reliable.'
    elif badge == 'good':
        return 'Calibration is good. Minor deviations exist but are unlikely to affect decisions.'
    elif badge == 'fair':
        return 'Calibration needs improvement. Consider applying Platt Scaling or Isotonic Regression.'
    else:
        return 'Calibration is poor — confidence scores are unreliable. Run auto-calibration to fix this.'
