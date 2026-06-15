"""
Automated Feature Selection — Post-training feature pruning using permutation importance.

Identifies and removes features with negligible contribution to model performance,
reducing overfitting and improving generalization, especially after feature engineering
creates many new features.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score

logger = logging.getLogger(__name__)


def select_features(model, X_train, y_train, X_test, y_test,
                    problem_type='classification', threshold=0.001,
                    n_repeats=10, retrain=True):
    """
    Remove features with negligible permutation importance.
    
    Args:
        model: Trained model to evaluate feature importance for.
        X_train: Training feature DataFrame.
        y_train: Training target Series.
        X_test: Test feature DataFrame.
        y_test: Test target Series.
        problem_type: 'classification' or 'regression'.
        threshold: Minimum mean permutation importance to keep a feature.
        n_repeats: Number of permutation repeats for stability.
        retrain: If True, retrain model on selected features and report score change.
    
    Returns:
        dict: {
            'selected_features': list of kept feature names,
            'removed_features': list of dropped feature names,
            'n_selected': int,
            'n_removed': int,
            'importances': list of dicts with feature, importance, std,
            'score_before': float,
            'score_after': float (if retrain=True),
            'score_change': float (if retrain=True),
        }
    """
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    
    # Compute permutation importance on test set
    try:
        result = permutation_importance(
            model, X_test, y_test,
            n_repeats=n_repeats,
            random_state=42,
            scoring=scoring,
            n_jobs=-1,
        )
    except Exception as e:
        logger.warning(f"Permutation importance failed: {e}")
        return {
            'selected_features': list(X_train.columns),
            'removed_features': [],
            'n_selected': X_train.shape[1],
            'n_removed': 0,
            'error': str(e),
        }
    
    importances_mean = result.importances_mean
    importances_std = result.importances_std
    feature_names = list(X_train.columns) if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Build importance ranking
    importance_records = []
    for i, name in enumerate(feature_names):
        importance_records.append({
            'feature': name,
            'importance': round(float(importances_mean[i]), 6),
            'std': round(float(importances_std[i]), 6),
        })
    importance_records.sort(key=lambda x: x['importance'], reverse=True)
    
    # Select features above threshold
    selected = [r['feature'] for r in importance_records if r['importance'] > threshold]
    removed = [r['feature'] for r in importance_records if r['importance'] <= threshold]
    
    # Ensure we keep at least 3 features
    if len(selected) < 3 and len(feature_names) >= 3:
        selected = [r['feature'] for r in importance_records[:max(3, len(feature_names) // 2)]]
        removed = [f for f in feature_names if f not in selected]
    
    # Score before selection
    y_pred_before = model.predict(X_test)
    if problem_type == 'classification':
        score_before = round(float(accuracy_score(y_test, y_pred_before)), 4)
    else:
        score_before = round(float(r2_score(y_test, y_pred_before)), 4)
    
    report = {
        'selected_features': selected,
        'removed_features': removed,
        'n_selected': len(selected),
        'n_removed': len(removed),
        'importances': importance_records,
        'score_before': score_before,
        'threshold': threshold,
    }
    
    # Optionally retrain with selected features
    if retrain and removed:
        try:
            X_train_sel = X_train[selected] if isinstance(X_train, pd.DataFrame) else X_train[:, [feature_names.index(f) for f in selected]]
            X_test_sel = X_test[selected] if isinstance(X_test, pd.DataFrame) else X_test[:, [feature_names.index(f) for f in selected]]
            
            m = clone(model)
            m.fit(X_train_sel, y_train)
            y_pred_after = m.predict(X_test_sel)
            
            if problem_type == 'classification':
                score_after = round(float(accuracy_score(y_test, y_pred_after)), 4)
            else:
                score_after = round(float(r2_score(y_test, y_pred_after)), 4)
            
            report['score_after'] = score_after
            report['score_change'] = round(score_after - score_before, 4)
            report['retrained_model'] = m
        except Exception as e:
            logger.warning(f"Retrain with selected features failed: {e}")
            report['retrain_error'] = str(e)
    
    logger.info(
        "Feature selection: kept %d/%d features (removed %d with importance <= %.4f)",
        len(selected), len(feature_names), len(removed), threshold,
    )
    
    return report
