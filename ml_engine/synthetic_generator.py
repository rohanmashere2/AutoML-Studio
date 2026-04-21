"""
Synthetic Data Generator — Generate realistic tabular data using statistical methods.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from scipy import stats


def generate_synthetic_data(df, n_samples=None, class_column=None, class_ratios=None):
    """Generate synthetic data that preserves distributions and correlations."""
    if n_samples is None:
        n_samples = len(df)
    
    n_samples = min(n_samples, 50000)
    
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    
    synthetic = pd.DataFrame()
    
    # Generate numeric columns using Gaussian Copula approach
    if numeric_cols:
        synthetic_numeric = _generate_numeric_copula(df[numeric_cols], n_samples)
        synthetic = pd.concat([synthetic, synthetic_numeric], axis=1)
    
    # Generate categorical columns preserving distributions
    if cat_cols:
        synthetic_cat = _generate_categorical(df[cat_cols], n_samples)
        synthetic = pd.concat([synthetic, synthetic_cat], axis=1)
    
    # Enforce class ratios if specified
    if class_column and class_ratios and class_column in synthetic.columns:
        synthetic = _enforce_class_ratios(synthetic, class_column, class_ratios, n_samples)
    
    # Validate quality
    quality = _validate_quality(df, synthetic, numeric_cols, cat_cols)
    
    return {
        'synthetic_data': synthetic,
        'n_generated': len(synthetic),
        'quality_metrics': quality,
        'columns': synthetic.columns.tolist(),
    }


def _generate_numeric_copula(df_numeric, n_samples):
    """Generate numeric data using Gaussian Copula to preserve correlations."""
    df_clean = df_numeric.fillna(df_numeric.median())
    
    # Store original distributions
    marginals = {}
    uniform = pd.DataFrame()
    
    for col in df_clean.columns:
        vals = df_clean[col].values
        # Rank transform to uniform
        ranks = stats.rankdata(vals) / (len(vals) + 1)
        uniform[col] = ranks
        marginals[col] = {
            'min': float(vals.min()),
            'max': float(vals.max()),
            'mean': float(vals.mean()),
            'std': float(vals.std()),
            'sorted_vals': np.sort(vals)
        }
    
    # Transform to normal
    normal = uniform.apply(stats.norm.ppf)
    normal = normal.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Get correlation structure
    corr_matrix = normal.corr().values
    # Ensure positive semi-definite
    eigvals, eigvecs = np.linalg.eigh(corr_matrix)
    eigvals = np.maximum(eigvals, 1e-8)
    corr_matrix = eigvecs @ np.diag(eigvals) @ eigvecs.T
    np.fill_diagonal(corr_matrix, 1.0)
    
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        L = np.eye(corr_matrix.shape[0])
    
    # Generate correlated normal samples
    z = np.random.randn(n_samples, len(df_clean.columns))
    correlated_normal = z @ L.T
    
    # Transform back to original distributions using inverse CDF
    synthetic = pd.DataFrame()
    for i, col in enumerate(df_clean.columns):
        uniform_samples = stats.norm.cdf(correlated_normal[:, i])
        uniform_samples = np.clip(uniform_samples, 0.001, 0.999)
        
        # Inverse transform using original sorted values
        sorted_vals = marginals[col]['sorted_vals']
        indices = (uniform_samples * len(sorted_vals)).astype(int)
        indices = np.clip(indices, 0, len(sorted_vals) - 1)
        synthetic[col] = sorted_vals[indices]
        
        # Add small noise
        noise_scale = marginals[col]['std'] * 0.01
        synthetic[col] += np.random.normal(0, max(noise_scale, 1e-6), n_samples)
    
    # Preserve integer types
    for col in df_clean.columns:
        if df_clean[col].dtype in ['int64', 'int32']:
            synthetic[col] = synthetic[col].round().astype(int)
    
    return synthetic


def _generate_categorical(df_cat, n_samples):
    """Generate categorical columns preserving frequency distributions."""
    synthetic = pd.DataFrame()
    
    for col in df_cat.columns:
        value_counts = df_cat[col].value_counts(normalize=True, dropna=False)
        values = value_counts.index.tolist()
        probs = value_counts.values
        
        # Sample with replacement according to original distribution
        synthetic[col] = np.random.choice(values, size=n_samples, p=probs)
    
    return synthetic


def _enforce_class_ratios(synthetic, class_column, class_ratios, n_samples):
    """Enforce specific class distribution in the synthetic data."""
    total = sum(class_ratios.values())
    normalized = {k: v / total for k, v in class_ratios.items()}
    
    frames = []
    for cls, ratio in normalized.items():
        n_cls = int(n_samples * ratio)
        cls_data = synthetic[synthetic[class_column] == cls]
        if len(cls_data) == 0:
            continue
        if len(cls_data) < n_cls:
            cls_data = cls_data.sample(n=n_cls, replace=True, random_state=42)
        else:
            cls_data = cls_data.sample(n=n_cls, random_state=42)
        frames.append(cls_data)
    
    if frames:
        return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return synthetic


def _validate_quality(real_df, synthetic_df, numeric_cols, cat_cols):
    """Validate synthetic data quality against real data."""
    metrics = {
        'column_shape_similarity': [],
        'overall_quality': 0,
    }
    
    similarities = []
    
    # Numeric columns: KS test
    for col in numeric_cols:
        if col in synthetic_df.columns and col in real_df.columns:
            real_vals = real_df[col].dropna().values
            syn_vals = synthetic_df[col].dropna().values
            if len(real_vals) > 0 and len(syn_vals) > 0:
                ks_stat, _ = stats.ks_2samp(real_vals, syn_vals)
                similarity = 1 - ks_stat
                similarities.append(similarity)
                metrics['column_shape_similarity'].append({
                    'column': col, 'type': 'numeric',
                    'ks_statistic': round(ks_stat, 4),
                    'similarity': round(similarity, 4)
                })
    
    # Categorical columns: frequency comparison
    for col in cat_cols:
        if col in synthetic_df.columns and col in real_df.columns:
            real_freq = real_df[col].value_counts(normalize=True).sort_index()
            syn_freq = synthetic_df[col].value_counts(normalize=True).sort_index()
            common = real_freq.index.intersection(syn_freq.index)
            if len(common) > 0:
                diff = abs(real_freq[common] - syn_freq.reindex(common, fill_value=0)).mean()
                similarity = 1 - diff
                similarities.append(similarity)
                metrics['column_shape_similarity'].append({
                    'column': col, 'type': 'categorical',
                    'freq_diff': round(diff, 4),
                    'similarity': round(similarity, 4)
                })
    
    metrics['overall_quality'] = round(np.mean(similarities) * 100, 1) if similarities else 0
    metrics['quality_grade'] = 'A' if metrics['overall_quality'] >= 90 else 'B' if metrics['overall_quality'] >= 75 else 'C' if metrics['overall_quality'] >= 60 else 'D'
    
    return metrics
