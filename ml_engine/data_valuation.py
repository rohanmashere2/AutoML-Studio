"""
Data Valuation Engine — Shapley-based data point importance scoring.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


def valuate_data(model, X_train, y_train, X_test, y_test, problem_type='classification', n_iterations=100):
    """Estimate the value of each training data point using KNN-proxy Shapley."""
    n_train = len(X_train)
    n_test = len(X_test)
    
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    # Use KNN-proxy for efficient Shapley estimation
    k = min(10, n_train - 1)
    
    if problem_type == 'classification':
        knn = KNeighborsClassifier(n_neighbors=k)
    else:
        knn = KNeighborsRegressor(n_neighbors=k)
    
    knn.fit(X_train, y_train)
    
    # Get nearest neighbors for each test point
    distances, indices = knn.kneighbors(X_test)
    
    # Monte Carlo Shapley estimation
    values = np.zeros(n_train)
    
    for _ in range(n_iterations):
        perm = np.random.permutation(n_train)
        
        for test_idx in range(min(n_test, 50)):  # Sample test points
            nn_indices = indices[test_idx]
            y_true = y_test[test_idx]
            
            # Marginal contribution of each neighbor
            prev_score = 0
            for rank, train_idx in enumerate(nn_indices):
                neighbors_so_far = nn_indices[:rank + 1]
                if problem_type == 'classification':
                    pred = _majority_vote(y_train[neighbors_so_far])
                    curr_score = 1.0 if pred == y_true else 0.0
                else:
                    pred = y_train[neighbors_so_far].mean()
                    curr_score = -abs(pred - y_true)
                
                marginal = curr_score - prev_score
                values[train_idx] += marginal
                prev_score = curr_score
    
    # Normalize
    values = values / max(n_iterations * min(n_test, 50), 1)
    
    # Rank data points
    ranked = np.argsort(values)
    
    # Identify high/low value points
    high_value = []
    low_value = []
    
    for idx in ranked[-20:][::-1]:
        high_value.append({
            'index': int(idx),
            'value': round(float(values[idx]), 6),
            'label': _safe_label(y_train[idx]),
        })
    
    for idx in ranked[:20]:
        low_value.append({
            'index': int(idx),
            'value': round(float(values[idx]), 6),
            'label': _safe_label(y_train[idx]),
            'possibly_mislabeled': float(values[idx]) < np.percentile(values, 5),
        })
    
    # Distribution
    score_dist = {
        'mean': round(float(values.mean()), 6),
        'std': round(float(values.std()), 6),
        'min': round(float(values.min()), 6),
        'max': round(float(values.max()), 6),
        'positive_pct': round(float((values > 0).mean() * 100), 1),
        'negative_pct': round(float((values < 0).mean() * 100), 1),
    }
    
    # Histogram
    counts, edges = np.histogram(values, bins=30)
    histogram = {
        'counts': counts.tolist(),
        'edges': [round(float(e), 6) for e in edges],
    }
    
    n_suspicious = int((values < np.percentile(values, 5)).sum())
    
    return {
        'high_value_points': high_value,
        'low_value_points': low_value,
        'distribution': score_dist,
        'histogram': histogram,
        'n_suspicious': n_suspicious,
        'n_train': n_train,
        'all_values': values.tolist(),
    }


def _majority_vote(labels):
    """Return most common label."""
    unique, counts = np.unique(labels, return_counts=True)
    return unique[np.argmax(counts)]


def _safe_label(val):
    """Convert label to JSON-safe format."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return round(float(val), 4)
    return str(val)
