"""
AutoML Studio — Data Prescription Engine (Feature #2)
After training, tells the user EXACTLY what data to collect to improve
the model — specific subgroups, estimated impact, and collection priority.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score


def prescribe_data(model, X_train, y_train, X_test, y_test,
                    problem_type='classification', feature_names=None):
    """
    Analyze model failures and prescribe specific data collection actions.

    Returns:
        dict with ranked prescriptions, each with subgroup, count, estimated impact
    """
    y_pred = model.predict(X_test)
    y_true = np.array(y_test)
    y_train_arr = np.array(y_train)
    is_clf = problem_type == 'classification'

    if hasattr(X_test, 'columns'):
        df_test = X_test.copy()
        df_train = X_train.copy() if hasattr(X_train, 'columns') else pd.DataFrame(X_train, columns=X_test.columns)
    else:
        cols = feature_names or [f'f_{i}' for i in range(X_test.shape[1])]
        df_test = pd.DataFrame(X_test, columns=cols)
        df_train = pd.DataFrame(X_train, columns=cols)

    if is_clf:
        overall_score = float(accuracy_score(y_true, y_pred))
    else:
        overall_score = float(r2_score(y_true, y_pred))

    prescriptions = []

    # 1. Subgroup error analysis → prescribe more data for failing subgroups
    prescriptions += _prescribe_subgroups(model, df_test, df_train, y_true, y_pred,
                                          y_train_arr, is_clf, overall_score)

    # 2. Class-specific prescriptions (classification)
    if is_clf:
        prescriptions += _prescribe_classes(y_train_arr, y_true, y_pred)

    # 3. Feature coverage gaps
    prescriptions += _prescribe_coverage(df_train, df_test, y_true, y_pred, is_clf)

    # 4. Learning curve extrapolation hint
    prescriptions += _prescribe_volume(len(y_train_arr), overall_score, is_clf)

    # Rank by estimated impact
    prescriptions.sort(key=lambda x: x.get('estimated_impact', 0), reverse=True)

    return {
        'prescriptions': prescriptions[:15],
        'overall_score': round(overall_score, 4),
        'metric': 'accuracy' if is_clf else 'r2',
        'total_prescriptions': len(prescriptions),
        'top_priority': prescriptions[0] if prescriptions else None,
    }


def _prescribe_subgroups(model, df_test, df_train, y_true, y_pred,
                          y_train, is_clf, overall_score):
    """Find subgroups with high error and prescribe more data."""
    prescriptions = []

    for col in df_test.columns[:20]:
        if not pd.api.types.is_numeric_dtype(df_test[col]):
            # Categorical slicing
            for val in df_test[col].unique()[:8]:
                mask = df_test[col] == val
                n_test = int(mask.sum())
                if n_test < 10:
                    continue
                score = _score(y_true[mask], y_pred[mask], is_clf)
                gap = overall_score - score
                if gap > 0.08:
                    train_count = int((df_train[col] == val).sum())
                    need = max(50, train_count)
                    prescriptions.append({
                        'type': 'subgroup',
                        'description': f'Collect {need}+ more samples where {col} = "{val}"',
                        'condition': f'{col} = {val}',
                        'current_train_samples': train_count,
                        'recommended_additional': need,
                        'subgroup_score': round(score, 4),
                        'overall_score': round(overall_score, 4),
                        'gap': round(gap, 4),
                        'estimated_impact': round(gap * n_test / max(len(y_true), 1), 4),
                        'priority': 'HIGH' if gap > 0.15 else 'MEDIUM',
                        'icon': '🎯',
                    })
        else:
            # Numeric: quartile slicing
            try:
                bins = pd.qcut(df_test[col], q=4, duplicates='drop')
                for label in bins.unique():
                    mask = bins == label
                    n_test = int(mask.sum())
                    if n_test < 10:
                        continue
                    score = _score(y_true[mask], y_pred[mask], is_clf)
                    gap = overall_score - score
                    if gap > 0.08:
                        prescriptions.append({
                            'type': 'subgroup',
                            'description': f'Collect more samples in range {col} ∈ {label}',
                            'condition': f'{col} in {label}',
                            'subgroup_score': round(score, 4),
                            'gap': round(gap, 4),
                            'estimated_impact': round(gap * n_test / max(len(y_true), 1), 4),
                            'priority': 'HIGH' if gap > 0.15 else 'MEDIUM',
                            'icon': '📊',
                        })
            except Exception:
                pass

    return prescriptions


def _prescribe_classes(y_train, y_true, y_pred):
    """Prescribe more data for underrepresented or poorly predicted classes."""
    prescriptions = []
    classes = np.unique(y_train)

    for cls in classes:
        train_count = int((y_train == cls).sum())
        test_mask = y_true == cls
        n_test = int(test_mask.sum())
        if n_test < 3:
            continue
        cls_acc = float((y_pred[test_mask] == cls).mean())
        if cls_acc < 0.7:
            need = max(100, train_count * 2) - train_count
            prescriptions.append({
                'type': 'class_balance',
                'description': f'Collect {need}+ more samples of class "{cls}" (only {cls_acc:.0%} recall)',
                'class': str(cls),
                'current_count': train_count,
                'recommended_additional': max(need, 0),
                'current_recall': round(cls_acc, 4),
                'estimated_impact': round((0.8 - cls_acc) * n_test / max(len(y_true), 1), 4),
                'priority': 'HIGH',
                'icon': '🏷️',
            })

    return prescriptions


def _prescribe_coverage(df_train, df_test, y_true, y_pred, is_clf):
    """Find feature-space regions with sparse training data."""
    prescriptions = []

    for col in df_train.select_dtypes(include='number').columns[:10]:
        train_range = (float(df_train[col].min()), float(df_train[col].max()))
        test_outside = ((df_test[col] < train_range[0]) | (df_test[col] > train_range[1]))
        n_outside = int(test_outside.sum())
        if n_outside >= 5:
            score = _score(y_true[test_outside], y_pred[test_outside], is_clf)
            prescriptions.append({
                'type': 'coverage_gap',
                'description': f'{n_outside} test samples have {col} outside training range [{train_range[0]:.2f}, {train_range[1]:.2f}]',
                'feature': col,
                'n_out_of_range': n_outside,
                'oor_score': round(score, 4),
                'estimated_impact': round(0.05 * n_outside / max(len(y_true), 1), 4),
                'priority': 'MEDIUM',
                'icon': '🔍',
            })

    return prescriptions


def _prescribe_volume(n_train, score, is_clf):
    """Simple volume-based prescription."""
    threshold = 0.85 if is_clf else 0.7
    if score < threshold and n_train < 5000:
        return [{
            'type': 'volume',
            'description': f'General: increasing dataset from {n_train} to {n_train * 3} samples could improve performance significantly.',
            'current_samples': n_train,
            'recommended_total': n_train * 3,
            'estimated_impact': 0.02,
            'priority': 'LOW',
            'icon': '📦',
        }]
    return []


def _score(y_true, y_pred, is_clf):
    try:
        if is_clf:
            return float(accuracy_score(y_true, y_pred))
        else:
            return float(r2_score(y_true, y_pred))
    except Exception:
        return 0.0
