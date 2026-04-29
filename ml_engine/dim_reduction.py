"""
Dimensionality Reduction Engine — PCA, t-SNE, UMAP for visualization.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def reduce_dimensions(X, labels=None, method='all', n_components=2):
    """Reduce dimensions for visualization. Returns 2D/3D coordinates."""
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_arr = X.values
    else:
        feature_names = [f'f_{i}' for i in range(X.shape[1])]
        X_arr = X
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X_arr, nan=0.0))
    
    results = {}
    
    # PCA
    try:
        n_comp = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
        pca = PCA(n_components=n_comp, random_state=42)
        coords = pca.fit_transform(X_scaled)
        
        # Full PCA for explained variance
        pca_full = PCA(n_components=min(X_scaled.shape[1], X_scaled.shape[0], 20), random_state=42)
        pca_full.fit(X_scaled)
        
        results['PCA'] = {
            'coordinates': coords.tolist(),
            'explained_variance': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': np.cumsum(pca_full.explained_variance_ratio_).tolist(),
            'all_variances': pca_full.explained_variance_ratio_.tolist(),
            'n_components_95': int(np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1),
            'loadings': pca.components_.tolist(),
            'top_features_pc1': _top_features(pca.components_[0], feature_names),
            'top_features_pc2': _top_features(pca.components_[1], feature_names) if n_comp > 1 else [],
        }
    except Exception as e:
        results['PCA'] = {'error': str(e)}
    
    # t-SNE
    try:
        from sklearn.manifold import TSNE
        n_samples = X_scaled.shape[0]
        perplexity = min(30, max(5, n_samples // 5))
        tsne = TSNE(n_components=min(n_components, 3), perplexity=perplexity, random_state=42, max_iter=1000)
        coords = tsne.fit_transform(X_scaled[:min(n_samples, 5000)])
        results['t-SNE'] = {
            'coordinates': coords.tolist(),
            'kl_divergence': round(float(tsne.kl_divergence_), 4),
            'perplexity': perplexity,
        }
    except Exception as e:
        results['t-SNE'] = {'error': str(e)}
    
    # UMAP (2D)
    try:
        import umap
        n_neighbors_umap = min(15, X_scaled.shape[0] - 1)
        reducer = umap.UMAP(n_components=min(n_components, 2), random_state=42,
                            n_neighbors=n_neighbors_umap)
        coords = reducer.fit_transform(X_scaled)
        
        # Compute trustworthiness
        try:
            from sklearn.manifold import trustworthiness as tw_func
            trust = round(float(tw_func(X_scaled, coords, n_neighbors=min(5, X_scaled.shape[0] - 2))), 4)
        except Exception:
            trust = None
        
        results['UMAP'] = {
            'coordinates': coords.tolist(),
            'trustworthiness': trust,
        }
    except ImportError:
        results['UMAP'] = {'error': 'umap-learn not installed'}
    except Exception as e:
        results['UMAP'] = {'error': str(e)}
    
    # 3D Projections (UMAP and t-SNE)
    if n_components >= 3 or n_components == 3:
        # UMAP 3D
        try:
            import umap
            reducer_3d = umap.UMAP(n_components=3, random_state=42,
                                    n_neighbors=min(15, X_scaled.shape[0] - 1))
            coords_3d = reducer_3d.fit_transform(X_scaled)
            results['UMAP_3D'] = {
                'coordinates': coords_3d.tolist(),
            }
        except Exception:
            pass
        
        # t-SNE 3D
        try:
            from sklearn.manifold import TSNE
            n_samples = X_scaled.shape[0]
            perplexity = min(30, max(5, n_samples // 5))
            tsne_3d = TSNE(n_components=3, perplexity=perplexity, random_state=42, max_iter=1000)
            coords_3d = tsne_3d.fit_transform(X_scaled[:min(n_samples, 3000)])
            results['t-SNE_3D'] = {
                'coordinates': coords_3d.tolist(),
                'kl_divergence': round(float(tsne_3d.kl_divergence_), 4),
            }
        except Exception:
            pass
    
    # Add labels if provided
    if labels is not None:
        for method_name in results:
            if 'coordinates' in results[method_name]:
                results[method_name]['labels'] = [int(l) for l in labels]
    
    return results


def _top_features(component, feature_names, top_n=5):
    """Get top contributing features for a PCA component."""
    abs_comp = np.abs(component)
    top_idx = abs_comp.argsort()[-top_n:][::-1]
    return [{'feature': feature_names[i], 'loading': round(float(component[i]), 4)} for i in top_idx]
