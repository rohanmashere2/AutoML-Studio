"""
Clustering Engine — Auto-clustering with 5 algorithms and optimal k selection.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def auto_cluster(X, max_k=10):
    """Run multiple clustering algorithms and pick the best one."""
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_arr = X.values
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        X_arr = X
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)
    
    n_samples = X_scaled.shape[0]
    max_k = min(max_k, n_samples // 3, 15)
    if max_k < 2:
        max_k = 2
    
    results = {}
    
    # 1. Find optimal k via silhouette
    k_scores = []
    inertias = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        k_scores.append({'k': k, 'silhouette': round(sil, 4)})
        inertias.append({'k': k, 'inertia': round(float(km.inertia_), 2)})
    
    optimal_k = max(k_scores, key=lambda x: x['silhouette'])['k'] if k_scores else 3
    
    # 2. Run algorithms with optimal k
    algorithms = {}
    
    # K-Means
    try:
        km = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        algorithms['K-Means'] = {
            'labels': labels.tolist(), 'n_clusters': optimal_k,
            'silhouette': round(sil, 4),
            'calinski_harabasz': round(calinski_harabasz_score(X_scaled, labels), 2) if len(set(labels)) > 1 else 0,
            'centers': km.cluster_centers_.tolist()
        }
    except Exception:
        pass
    
    # DBSCAN
    try:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=min(5, n_samples - 1))
        nn.fit(X_scaled)
        distances, _ = nn.kneighbors(X_scaled)
        eps = float(np.percentile(distances[:, -1], 90))
        
        db = DBSCAN(eps=eps, min_samples=max(3, n_samples // 100))
        labels = db.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            mask = labels >= 0
            sil = silhouette_score(X_scaled[mask], labels[mask]) if mask.sum() > 10 else -1
            algorithms['DBSCAN'] = {
                'labels': labels.tolist(), 'n_clusters': n_clusters,
                'silhouette': round(sil, 4), 'noise_points': int((labels == -1).sum()),
                'eps': round(eps, 4)
            }
    except Exception:
        pass
    
    # Hierarchical
    try:
        hc = AgglomerativeClustering(n_clusters=optimal_k)
        labels = hc.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        algorithms['Hierarchical'] = {
            'labels': labels.tolist(), 'n_clusters': optimal_k,
            'silhouette': round(sil, 4)
        }
    except Exception:
        pass
    
    # GMM
    try:
        gmm = GaussianMixture(n_components=optimal_k, random_state=42, max_iter=200)
        labels = gmm.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else -1
        algorithms['Gaussian Mixture'] = {
            'labels': labels.tolist(), 'n_clusters': optimal_k,
            'silhouette': round(sil, 4), 'bic': round(float(gmm.bic(X_scaled)), 2)
        }
    except Exception:
        pass
    
    # HDBSCAN (optional)
    try:
        import hdbscan
        hdb = hdbscan.HDBSCAN(min_cluster_size=max(5, n_samples // 50))
        labels = hdb.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            mask = labels >= 0
            sil = silhouette_score(X_scaled[mask], labels[mask]) if mask.sum() > 10 else -1
            algorithms['HDBSCAN'] = {
                'labels': labels.tolist(), 'n_clusters': n_clusters,
                'silhouette': round(sil, 4), 'noise_points': int((labels == -1).sum())
            }
    except ImportError:
        pass
    
    # Pick best algorithm
    best_name = max(algorithms, key=lambda k: algorithms[k]['silhouette']) if algorithms else 'K-Means'
    best = algorithms.get(best_name, {})
    best_labels = best.get('labels', [0] * n_samples)
    
    # Generate cluster profiles
    profiles = _generate_cluster_profiles(X_arr, best_labels, feature_names)
    
    return {
        'best_algorithm': best_name,
        'optimal_k': optimal_k,
        'k_analysis': k_scores,
        'elbow_data': inertias,
        'algorithms': {k: {kk: vv for kk, vv in v.items() if kk != 'labels'} for k, v in algorithms.items()},
        'best_labels': best_labels,
        'best_silhouette': best.get('silhouette', -1),
        'cluster_profiles': profiles,
        'n_samples': n_samples,
        'n_features': X_scaled.shape[1],
    }


def _generate_cluster_profiles(X, labels, feature_names):
    """Generate descriptive profiles for each cluster."""
    df = pd.DataFrame(X, columns=feature_names)
    df['_cluster'] = labels
    profiles = []
    
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        subset = df[df['_cluster'] == cluster_id]
        profile = {
            'cluster_id': int(cluster_id),
            'size': len(subset),
            'percentage': round(len(subset) / len(df) * 100, 1),
            'features': {}
        }
        for feat in feature_names[:15]:
            if pd.api.types.is_numeric_dtype(df[feat]):
                profile['features'][feat] = {
                    'mean': round(float(subset[feat].mean()), 3),
                    'median': round(float(subset[feat].median()), 3),
                    'std': round(float(subset[feat].std()), 3),
                    'overall_mean': round(float(df[feat].mean()), 3)
                }
        profiles.append(profile)
    
    return profiles
