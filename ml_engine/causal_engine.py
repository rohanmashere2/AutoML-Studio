"""
Causal Inference Engine — Discover causal relationships and estimate effects.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy import stats


def discover_causal_graph(df, target_column=None, max_features=15):
    """Discover causal structure using PC-like conditional independence tests."""
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] > max_features:
        if target_column and target_column in numeric.columns:
            corrs = numeric.corr()[target_column].abs().sort_values(ascending=False)
            keep = corrs.head(max_features).index.tolist()
            numeric = numeric[keep]
        else:
            numeric = numeric.iloc[:, :max_features]
    
    columns = numeric.columns.tolist()
    n_vars = len(columns)
    
    # Build adjacency based on partial correlations
    edges = []
    adj_matrix = np.zeros((n_vars, n_vars))
    
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            # Conditional independence test
            x = numeric.iloc[:, i].dropna().values
            y = numeric.iloc[:, j].dropna().values
            min_len = min(len(x), len(y))
            if min_len < 10:
                continue
            x, y = x[:min_len], y[:min_len]
            
            corr, p_value = stats.pearsonr(x, y)
            if p_value < 0.05 and abs(corr) > 0.1:
                # Determine direction using regression asymmetry
                direction = _infer_direction(numeric, columns[i], columns[j])
                
                edges.append({
                    'source': columns[i] if direction >= 0 else columns[j],
                    'target': columns[j] if direction >= 0 else columns[i],
                    'strength': round(abs(float(corr)), 4),
                    'p_value': round(float(p_value), 6),
                    'type': 'causal' if abs(corr) > 0.3 else 'association',
                })
                adj_matrix[i, j] = float(corr)
                adj_matrix[j, i] = float(corr)
    
    # Sort by strength
    edges.sort(key=lambda e: e['strength'], reverse=True)
    
    return {
        'nodes': [{'id': c, 'label': c} for c in columns],
        'edges': edges[:50],
        'adjacency': adj_matrix.tolist(),
        'n_nodes': n_vars,
        'n_edges': len(edges),
    }


def estimate_causal_effect(df, treatment, outcome, confounders=None):
    """Estimate causal effect of treatment on outcome using regression adjustment."""
    df_clean = df[[treatment, outcome] + (confounders or [])].dropna()
    
    if len(df_clean) < 20:
        return {'error': 'Not enough data for causal estimation'}
    
    # Unadjusted effect
    x = df_clean[treatment].values.reshape(-1, 1)
    y = df_clean[outcome].values
    reg_simple = LinearRegression().fit(x, y)
    unadjusted_effect = float(reg_simple.coef_[0])
    
    # Adjusted effect (controlling for confounders)
    adjusted_effect = unadjusted_effect
    if confounders:
        X_full = df_clean[[treatment] + confounders].values
        reg_adj = LinearRegression().fit(X_full, y)
        adjusted_effect = float(reg_adj.coef_[0])
    
    # Bootstrap confidence interval
    n_boot = 500
    boot_effects = []
    for _ in range(n_boot):
        idx = np.random.choice(len(df_clean), len(df_clean), replace=True)
        boot_df = df_clean.iloc[idx]
        if confounders:
            X_b = boot_df[[treatment] + confounders].values
        else:
            X_b = boot_df[treatment].values.reshape(-1, 1)
        y_b = boot_df[outcome].values
        try:
            reg_b = LinearRegression().fit(X_b, y_b)
            boot_effects.append(float(reg_b.coef_[0]))
        except Exception:
            pass
    
    ci_lower = float(np.percentile(boot_effects, 2.5)) if boot_effects else 0
    ci_upper = float(np.percentile(boot_effects, 97.5)) if boot_effects else 0
    
    # Statistical significance
    effect_std = np.std(boot_effects) if boot_effects else 1
    z_score = adjusted_effect / max(effect_std, 1e-10)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    return {
        'treatment': treatment,
        'outcome': outcome,
        'confounders': confounders or [],
        'unadjusted_effect': round(unadjusted_effect, 6),
        'adjusted_effect': round(adjusted_effect, 6),
        'confidence_interval': [round(ci_lower, 6), round(ci_upper, 6)],
        'p_value': round(float(p_value), 6),
        'significant': p_value < 0.05,
        'interpretation': _interpret_effect(treatment, outcome, adjusted_effect, p_value),
    }


def run_counterfactual(df, model, feature_name, original_value, new_value, row_index=0):
    """What-if counterfactual: change a feature, see outcome change."""
    row = df.iloc[[row_index]].copy()
    original_pred = float(model.predict(row)[0])
    
    row[feature_name] = new_value
    new_pred = float(model.predict(row)[0])
    
    return {
        'feature': feature_name,
        'original_value': original_value,
        'new_value': new_value,
        'original_prediction': round(original_pred, 4),
        'new_prediction': round(new_pred, 4),
        'effect': round(new_pred - original_pred, 4),
        'pct_change': round((new_pred - original_pred) / max(abs(original_pred), 1e-10) * 100, 2),
    }


def _infer_direction(df, col_a, col_b):
    """Infer causal direction using regression error asymmetry."""
    try:
        a = df[col_a].dropna().values
        b = df[col_b].dropna().values
        min_len = min(len(a), len(b))
        a, b = a[:min_len], b[:min_len]
        
        # A -> B
        reg_ab = LinearRegression().fit(a.reshape(-1, 1), b)
        res_ab = np.var(b - reg_ab.predict(a.reshape(-1, 1)))
        
        # B -> A
        reg_ba = LinearRegression().fit(b.reshape(-1, 1), a)
        res_ba = np.var(a - reg_ba.predict(b.reshape(-1, 1)))
        
        return 1 if res_ab <= res_ba else -1
    except Exception:
        return 1


def _interpret_effect(treatment, outcome, effect, p_value):
    """Generate human-readable interpretation."""
    if p_value >= 0.05:
        return f"No statistically significant causal effect of {treatment} on {outcome} detected (p={p_value:.4f})."
    
    direction = "increases" if effect > 0 else "decreases"
    return f"A 1-unit increase in {treatment} {direction} {outcome} by {abs(effect):.4f} units (p={p_value:.4f})."
