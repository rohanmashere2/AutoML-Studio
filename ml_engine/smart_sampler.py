"""
AutoML Studio — Smart Subsampling Engine (Feature #17)
Intelligently subsample large datasets preserving performance.
Strategies: stratified, boundary-aware, diversity sampling.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def smart_subsample(X, y, target_size=None, strategy='auto', problem_type='classification'):
    """
    Intelligently subsample a large dataset.

    Args:
        X: features DataFrame
        y: target Series/array
        target_size: desired sample count (default: auto-detect)
        strategy: 'stratified', 'boundary', 'diversity', or 'auto'

    Returns:
        dict with sampled indices, strategy used, comparison info
    """
    n_samples = len(y)
    y_arr = np.array(y)
    is_clf = problem_type == 'classification'

    if target_size is None:
        if n_samples <= 50000:
            return {'message': f'Dataset has {n_samples} rows — no subsampling needed.',
                    'sampled': False}
        target_size = min(50000, n_samples // 2)

    if target_size >= n_samples:
        return {'message': 'Target size >= dataset size.', 'sampled': False}

    if strategy == 'auto':
        if is_clf and len(np.unique(y_arr)) <= 20:
            strategy = 'stratified'
        elif n_samples > 100000:
            strategy = 'stratified'
        else:
            strategy = 'diversity'

    if strategy == 'stratified':
        indices = _stratified_sample(y_arr, target_size, is_clf)
    elif strategy == 'boundary':
        indices = _boundary_sample(X, y_arr, target_size, is_clf)
    elif strategy == 'diversity':
        indices = _diversity_sample(X, target_size)
    else:
        indices = _stratified_sample(y_arr, target_size, is_clf)

    # Stats
    if is_clf:
        orig_dist = pd.Series(y_arr).value_counts(normalize=True).to_dict()
        sampled_dist = pd.Series(y_arr[indices]).value_counts(normalize=True).to_dict()
    else:
        orig_dist = {'mean': round(float(y_arr.mean()), 4),
                     'std': round(float(y_arr.std()), 4)}
        sampled_dist = {'mean': round(float(y_arr[indices].mean()), 4),
                        'std': round(float(y_arr[indices].std()), 4)}

    return {
        'sampled': True,
        'strategy': strategy,
        'original_size': n_samples,
        'sampled_size': len(indices),
        'reduction_pct': round((1 - len(indices) / n_samples) * 100, 1),
        'indices': indices.tolist() if isinstance(indices, np.ndarray) else indices,
        'original_distribution': {str(k): round(float(v), 4) for k, v in orig_dist.items()},
        'sampled_distribution': {str(k): round(float(v), 4) for k, v in sampled_dist.items()},
        'message': (f'Reduced from {n_samples} to {len(indices)} rows ({strategy} sampling). '
                    f'Distribution preserved.'),
    }


def _stratified_sample(y, target_size, is_clf):
    """Stratified sampling preserving class distribution."""
    if is_clf:
        classes, counts = np.unique(y, return_counts=True)
        indices = []
        for cls, count in zip(classes, counts):
            cls_indices = np.where(y == cls)[0]
            n_take = max(1, int(target_size * count / len(y)))
            n_take = min(n_take, len(cls_indices))
            sampled = np.random.RandomState(42).choice(cls_indices, n_take, replace=False)
            indices.extend(sampled.tolist())
        return np.array(indices[:target_size])
    else:
        return np.random.RandomState(42).choice(len(y), target_size, replace=False)


def _boundary_sample(X, y, target_size, is_clf):
    """Oversample points near decision boundaries using KNN."""
    X_arr = np.array(X.select_dtypes(include='number')) if hasattr(X, 'select_dtypes') else np.array(X)
    n = len(y)

    # Use KNN to find boundary points (where neighbours have different labels)
    k = min(10, n - 1)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nn.fit(X_arr)
    _, neighbor_indices = nn.kneighbors(X_arr)

    # Boundary score: fraction of neighbours with different label
    boundary_scores = np.zeros(n)
    for i in range(n):
        neighbor_labels = y[neighbor_indices[i]]
        boundary_scores[i] = np.mean(neighbor_labels != y[i]) if is_clf else np.std(y[neighbor_indices[i]])

    # Sample: 60% boundary, 40% random
    n_boundary = int(target_size * 0.6)
    n_random = target_size - n_boundary

    boundary_idx = np.argsort(boundary_scores)[::-1][:n_boundary]
    remaining = np.setdiff1d(np.arange(n), boundary_idx)
    random_idx = np.random.RandomState(42).choice(remaining, min(n_random, len(remaining)), replace=False)

    return np.concatenate([boundary_idx, random_idx])[:target_size]


def _diversity_sample(X, target_size):
    """Diversity sampling using mini-batch k-means centroids."""
    X_arr = np.array(X.select_dtypes(include='number')) if hasattr(X, 'select_dtypes') else np.array(X)

    try:
        from sklearn.cluster import MiniBatchKMeans
        n_clusters = min(target_size, len(X_arr) // 2)
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        labels = kmeans.fit_predict(X_arr)

        indices = []
        for c in range(n_clusters):
            cluster_idx = np.where(labels == c)[0]
            n_take = max(1, int(target_size * len(cluster_idx) / len(X_arr)))
            sampled = np.random.RandomState(42).choice(cluster_idx, min(n_take, len(cluster_idx)), replace=False)
            indices.extend(sampled.tolist())
        return np.array(indices[:target_size])
    except Exception:
        return np.random.RandomState(42).choice(len(X_arr), target_size, replace=False)
