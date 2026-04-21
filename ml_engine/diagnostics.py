"""
AutoML Problem Solver - Advanced Model Diagnostics
ROC curves, Precision-Recall curves, residual analysis, learning curves, calibration.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import learning_curve, cross_val_predict
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.calibration import calibration_curve


def generate_diagnostics(models, X_train, X_test, y_train, y_test, problem_type='classification'):
    """
    Generate comprehensive diagnostics for all trained models.
    
    Returns:
        dict: Per-model diagnostic data
    """
    diagnostics = {}
    
    for name, model in models.items():
        try:
            diag = _diagnose_single_model(model, name, X_train, X_test, y_train, y_test, problem_type)
            diagnostics[name] = diag
        except Exception as e:
            diagnostics[name] = {'error': str(e)}
    
    return diagnostics


def _diagnose_single_model(model, name, X_train, X_test, y_train, y_test, problem_type):
    """Generate diagnostics for a single model."""
    diag = {'model_name': name}
    
    y_test_pred = model.predict(X_test)
    
    if problem_type == 'classification':
        diag.update(_classification_diagnostics(model, X_train, X_test, y_train, y_test, y_test_pred))
    else:
        diag.update(_regression_diagnostics(model, X_train, X_test, y_train, y_test, y_test_pred))
    
    # Learning curves (both)
    diag['learning_curve'] = _compute_learning_curve(model, X_train, y_train, problem_type)
    
    return diag


def _classification_diagnostics(model, X_train, X_test, y_train, y_test, y_test_pred):
    """Classification-specific diagnostics."""
    diag = {}
    classes = np.unique(np.concatenate([y_train, y_test]))
    is_binary = len(classes) == 2
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    diag['confusion_matrix'] = {
        'matrix': cm.tolist(),
        'labels': [str(c) for c in classes],
    }
    
    # Per-class metrics
    report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
    per_class = []
    for cls in classes:
        cls_str = str(cls)
        if cls_str in report:
            per_class.append({
                'class': cls_str,
                'precision': round(report[cls_str]['precision'], 4),
                'recall': round(report[cls_str]['recall'], 4),
                'f1': round(report[cls_str]['f1-score'], 4),
                'support': int(report[cls_str]['support']),
            })
    diag['per_class_metrics'] = per_class
    
    # ROC curves
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)
        
        if is_binary:
            fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            diag['roc_curve'] = {
                'fpr': _downsample(fpr.tolist(), 100),
                'tpr': _downsample(tpr.tolist(), 100),
                'auc': round(float(roc_auc), 4),
            }
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
            ap = average_precision_score(y_test, y_proba[:, 1])
            diag['pr_curve'] = {
                'precision': _downsample(precision.tolist(), 100),
                'recall': _downsample(recall.tolist(), 100),
                'avg_precision': round(float(ap), 4),
            }
            
            # Calibration curve
            try:
                prob_true, prob_pred = calibration_curve(y_test, y_proba[:, 1], n_bins=10, strategy='uniform')
                diag['calibration_curve'] = {
                    'prob_true': prob_true.tolist(),
                    'prob_pred': prob_pred.tolist(),
                }
            except Exception:
                pass
        else:
            # Multi-class ROC (one-vs-rest)
            roc_per_class = []
            for i, cls in enumerate(classes):
                try:
                    y_binary = (y_test == cls).astype(int)
                    fpr, tpr, _ = roc_curve(y_binary, y_proba[:, i])
                    roc_auc = auc(fpr, tpr)
                    roc_per_class.append({
                        'class': str(cls),
                        'fpr': _downsample(fpr.tolist(), 50),
                        'tpr': _downsample(tpr.tolist(), 50),
                        'auc': round(float(roc_auc), 4),
                    })
                except Exception:
                    pass
            diag['roc_curves_multiclass'] = roc_per_class
    
    # Misclassified examples summary
    misclassified_mask = y_test_pred != y_test
    n_misclassified = int(misclassified_mask.sum())
    diag['misclassified'] = {
        'count': n_misclassified,
        'pct': round(n_misclassified / len(y_test) * 100, 2),
    }
    
    return diag


def _regression_diagnostics(model, X_train, X_test, y_train, y_test, y_test_pred):
    """Regression-specific diagnostics."""
    diag = {}
    y_train_pred = model.predict(X_train)
    
    # Residuals
    residuals = y_test - y_test_pred
    
    diag['residual_analysis'] = {
        'residuals': _downsample(residuals.tolist(), 200),
        'predicted': _downsample(y_test_pred.tolist(), 200),
        'actual': _downsample(y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test), 200),
        'mean_residual': round(float(np.mean(residuals)), 4),
        'std_residual': round(float(np.std(residuals)), 4),
        'residual_skew': round(float(pd.Series(residuals).skew()), 4),
    }
    
    # QQ plot data (sorted residuals vs theoretical quantiles)
    sorted_residuals = np.sort(residuals)
    n = len(sorted_residuals)
    theoretical_quantiles = np.array([(i - 0.5) / n for i in range(1, n + 1)])
    from scipy import stats as sp_stats
    try:
        theoretical_quantiles = sp_stats.norm.ppf(theoretical_quantiles)
        diag['qq_plot'] = {
            'theoretical': _downsample(theoretical_quantiles.tolist(), 100),
            'observed': _downsample(sorted_residuals.tolist(), 100),
        }
    except Exception:
        pass
    
    # Residual distribution (histogram data)
    try:
        hist_counts, hist_edges = np.histogram(residuals, bins=30)
        diag['residual_distribution'] = {
            'counts': hist_counts.tolist(),
            'edges': hist_edges.tolist(),
        }
    except Exception:
        pass
    
    # Actual vs Predicted scatter
    diag['actual_vs_predicted'] = {
        'actual': _downsample(y_test.tolist() if hasattr(y_test, 'tolist') else list(y_test), 200),
        'predicted': _downsample(y_test_pred.tolist(), 200),
        'r2': round(float(r2_score(y_test, y_test_pred)), 4),
    }
    
    # Error distribution by predicted value ranges
    diag['error_metrics'] = {
        'mae': round(float(mean_absolute_error(y_test, y_test_pred)), 4),
        'mse': round(float(mean_squared_error(y_test, y_test_pred)), 4),
        'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 4),
        'r2': round(float(r2_score(y_test, y_test_pred)), 4),
        'train_r2': round(float(r2_score(y_train, y_train_pred)), 4),
        'mape': round(float(_mape(y_test, y_test_pred)), 4),
    }
    
    return diag


def _compute_learning_curve(model, X_train, y_train, problem_type, cv=5):
    """Compute learning curve data."""
    try:
        scoring = 'accuracy' if problem_type == 'classification' else 'r2'
        
        n_samples = X_train.shape[0]
        if n_samples < 50:
            return None
        
        train_sizes = np.linspace(0.1, 1.0, 8)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train,
            train_sizes=train_sizes,
            cv=min(cv, 5),
            scoring=scoring,
            n_jobs=-1,
            random_state=42,
        )
        
        return {
            'train_sizes': train_sizes_abs.tolist(),
            'train_mean': np.mean(train_scores, axis=1).round(4).tolist(),
            'train_std': np.std(train_scores, axis=1).round(4).tolist(),
            'val_mean': np.mean(val_scores, axis=1).round(4).tolist(),
            'val_std': np.std(val_scores, axis=1).round(4).tolist(),
        }
    except Exception:
        return None


def _mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def _downsample(data, max_points):
    """Downsample a list to max_points for JSON efficiency."""
    if len(data) <= max_points:
        return [round(float(x), 6) if isinstance(x, (float, np.floating)) else x for x in data]
    
    indices = np.linspace(0, len(data) - 1, max_points, dtype=int)
    return [round(float(data[i]), 6) if isinstance(data[i], (float, np.floating)) else data[i] for i in indices]
