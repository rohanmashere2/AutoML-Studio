"""
AutoML Studio — Conformal Prediction Engine (Feature #11)
Guaranteed prediction intervals (regression) and prediction sets (classification).
"""

import numpy as np


def build_conformal_predictor(model, X_cal, y_cal, problem_type='classification',
                               confidence=0.95):
    """Build conformal predictor from calibration set."""
    alpha = 1 - confidence
    y_cal = np.array(y_cal)
    n = len(y_cal)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)

    if problem_type == 'classification':
        if not hasattr(model, 'predict_proba'):
            return {'error': 'Model needs predict_proba'}
        proba = model.predict_proba(X_cal)
        scores = np.array([1 - proba[i, int(y_cal[i])] if int(y_cal[i]) < proba.shape[1] else 1.0
                           for i in range(n)])
        threshold = float(np.quantile(scores, q_level))
        return {'type': 'classification', 'threshold': threshold, 'alpha': alpha,
                'confidence': confidence, 'n_calibration': n, 'n_classes': proba.shape[1]}
    else:
        preds = model.predict(X_cal)
        residuals = np.abs(y_cal - preds)
        margin = float(np.quantile(residuals, q_level))
        return {'type': 'regression', 'margin': margin, 'alpha': alpha,
                'confidence': confidence, 'n_calibration': n,
                'residual_mean': round(float(residuals.mean()), 4)}


def conformal_predict(model, X_new, cal_data, problem_type='classification'):
    """Generate conformal predictions."""
    n = len(X_new) if hasattr(X_new, '__len__') else X_new.shape[0]

    if problem_type == 'classification':
        threshold = cal_data['threshold']
        proba = model.predict_proba(X_new)
        point_preds = model.predict(X_new)
        results = []
        set_sizes = []
        for i in range(min(n, 300)):
            included = [int(c) for c in range(proba.shape[1]) if (1 - proba[i, c]) <= threshold]
            if not included:
                included = [int(np.argmax(proba[i]))]
            set_sizes.append(len(included))
            results.append({'index': i, 'point_prediction': _s(point_preds[i]),
                            'prediction_set': included, 'set_size': len(included),
                            'max_probability': round(float(np.max(proba[i])), 4)})
        avg_sz = float(np.mean(set_sizes)) if set_sizes else 1.0
        return {'predictions': results, 'confidence': cal_data['confidence'],
                'average_set_size': round(avg_sz, 2),
                'singleton_rate': round(float(np.mean(np.array(set_sizes) == 1)) * 100, 1)}
    else:
        margin = cal_data['margin']
        preds = model.predict(X_new)
        results = []
        for i in range(min(n, 300)):
            results.append({'index': i, 'point_prediction': round(float(preds[i]), 4),
                            'lower_bound': round(float(preds[i] - margin), 4),
                            'upper_bound': round(float(preds[i] + margin), 4)})
        return {'predictions': results, 'confidence': cal_data['confidence'],
                'margin': round(margin, 4),
                'average_width': round(margin * 2, 4)}


def _s(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 4)
    return v
