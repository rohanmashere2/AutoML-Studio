"""
Label Error Detection — Data-centric AI module using confident learning.

Identifies likely mislabeled training samples using cross-validated predicted
probabilities. When cleanlab is available, uses its robust implementation;
otherwise falls back to a manual confident learning approach.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.base import clone

logger = logging.getLogger(__name__)


def detect_label_errors(model, X_train, y_train, n_folds=5, threshold=0.5):
    """
    Detect likely mislabeled samples using confident learning.
    
    Uses cross-validated predicted probabilities to find samples where the
    model is confident the true label is different from the given label.
    
    Args:
        model: Sklearn-compatible classifier with predict_proba().
        X_train: Training features (DataFrame or ndarray).
        y_train: Training labels (Series or ndarray).
        n_folds: Number of cross-validation folds.
        threshold: Confidence threshold for flagging label errors.
    
    Returns:
        dict: {
            'n_issues': int — number of detected label errors,
            'issue_indices': list — indices of likely mislabeled samples,
            'pct_issues': float — percentage of training data with errors,
            'details': list of dicts with index, given_label, predicted_label, confidence,
            'method': str — 'cleanlab' or 'manual_confident_learning',
        }
    """
    y = np.array(y_train)
    
    # Try cleanlab first (more robust implementation)
    try:
        return _detect_with_cleanlab(model, X_train, y)
    except ImportError:
        logger.info("cleanlab not installed, using manual confident learning")
    except Exception as e:
        logger.warning(f"cleanlab failed: {e}, falling back to manual method")
    
    # Fallback: manual confident learning
    return _detect_manual(model, X_train, y, n_folds, threshold)


def _detect_with_cleanlab(model, X_train, y):
    """Use cleanlab's robust confident learning implementation."""
    from cleanlab.classification import CleanLearning
    
    cl = CleanLearning(clone(model))
    label_issues = cl.find_label_issues(X_train, y)
    
    issue_mask = label_issues['is_label_issue']
    issue_indices = label_issues[issue_mask].index.tolist()
    
    details = []
    for idx in issue_indices[:50]:  # Cap at 50 for report readability
        details.append({
            'index': int(idx),
            'given_label': int(y[idx]) if hasattr(y[idx], 'item') else y[idx],
            'predicted_label': int(label_issues.loc[idx, 'predicted_label']) if 'predicted_label' in label_issues else None,
            'confidence': round(float(label_issues.loc[idx, 'label_quality']), 4) if 'label_quality' in label_issues else None,
        })
    
    return {
        'n_issues': int(issue_mask.sum()),
        'issue_indices': issue_indices,
        'pct_issues': round(float(issue_mask.mean() * 100), 2),
        'details': details,
        'method': 'cleanlab',
    }


def _detect_manual(model, X_train, y, n_folds=5, threshold=0.5):
    """Manual confident learning using cross-validated probabilities."""
    try:
        # Get cross-validated predicted probabilities
        pred_proba = cross_val_predict(
            clone(model), X_train, y,
            cv=min(n_folds, len(np.unique(y))),
            method='predict_proba',
            n_jobs=-1,
        )
    except Exception as e:
        logger.error(f"Cross-validation failed for label error detection: {e}")
        return {
            'n_issues': 0,
            'issue_indices': [],
            'pct_issues': 0.0,
            'details': [],
            'method': 'manual_confident_learning',
            'error': str(e),
        }
    
    classes = np.unique(y)
    n_samples = len(y)
    
    # For each sample, check if the model is confident the label is wrong
    issue_indices = []
    details = []
    
    for i in range(n_samples):
        given_label = y[i]
        given_label_idx = np.where(classes == given_label)[0][0]
        given_label_prob = pred_proba[i, given_label_idx]
        predicted_label_idx = pred_proba[i].argmax()
        predicted_label = classes[predicted_label_idx]
        max_prob = pred_proba[i, predicted_label_idx]
        
        # Flag as error if:
        # 1. Predicted label differs from given label, AND
        # 2. Model is confident in the predicted label (prob > threshold), AND
        # 3. Model assigns low probability to the given label
        if predicted_label != given_label and max_prob > threshold and given_label_prob < (1 - threshold):
            issue_indices.append(i)
            if len(details) < 50:  # Cap at 50
                details.append({
                    'index': int(i),
                    'given_label': int(given_label) if hasattr(given_label, 'item') else given_label,
                    'predicted_label': int(predicted_label) if hasattr(predicted_label, 'item') else predicted_label,
                    'confidence': round(float(max_prob), 4),
                    'given_label_prob': round(float(given_label_prob), 4),
                })
    
    # Sort by confidence (most likely errors first)
    details.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        'n_issues': len(issue_indices),
        'issue_indices': issue_indices,
        'pct_issues': round(len(issue_indices) / max(n_samples, 1) * 100, 2),
        'details': details,
        'method': 'manual_confident_learning',
    }


def clean_labels(X_train, y_train, issue_indices, strategy='remove'):
    """
    Clean detected label errors from the training data.
    
    Args:
        X_train: Training features.
        y_train: Training labels.
        issue_indices: Indices of detected label errors.
        strategy: 'remove' to drop, or 'relabel' to use model's prediction (requires pred_labels).
    
    Returns:
        tuple: (cleaned_X, cleaned_y, n_removed)
    """
    if not issue_indices:
        return X_train, y_train, 0
    
    mask = np.ones(len(y_train), dtype=bool)
    mask[issue_indices] = False
    
    if isinstance(X_train, pd.DataFrame):
        X_clean = X_train.iloc[mask].reset_index(drop=True)
    else:
        X_clean = X_train[mask]
    
    if isinstance(y_train, pd.Series):
        y_clean = y_train.iloc[mask].reset_index(drop=True)
    else:
        y_clean = y_train[mask]
    
    return X_clean, y_clean, len(issue_indices)
