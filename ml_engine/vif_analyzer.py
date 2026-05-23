"""
AutoML Studio — VIF Multicollinearity Analyzer (Feature #15)
Computes Variance Inflation Factor for each feature. Catches multicollinearity
that pairwise correlation misses.
"""

import numpy as np
import pandas as pd


def compute_vif(X, threshold=10.0):
    """
    Compute VIF for each numeric feature.

    Args:
        X: DataFrame of features (no target)
        threshold: VIF above this is severe multicollinearity

    Returns:
        dict with per-feature VIF, flagged features, recommendations
    """
    numeric = X.select_dtypes(include='number')
    if numeric.shape[1] < 2:
        return {'error': 'Need at least 2 numeric features for VIF analysis'}

    # Drop constant columns
    numeric = numeric.loc[:, numeric.std() > 1e-10]
    if numeric.shape[1] < 2:
        return {'error': 'Not enough variable features for VIF analysis'}

    vif_data = []
    cols = list(numeric.columns)

    for i, col in enumerate(cols):
        try:
            y = numeric[col].values
            X_other = numeric.drop(columns=[col]).values
            # Add intercept
            X_other = np.column_stack([np.ones(len(y)), X_other])
            # OLS: R² = 1 - SS_res / SS_tot
            beta = np.linalg.lstsq(X_other, y, rcond=None)[0]
            y_pred = X_other @ beta
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r_squared = 1 - (ss_res / max(ss_tot, 1e-10))
            vif = 1.0 / max(1 - r_squared, 1e-10)
        except Exception:
            vif = 0.0

        if vif > 1000:
            severity = 'critical'
            icon = '🔴'
        elif vif > threshold:
            severity = 'high'
            icon = '🟠'
        elif vif > 5:
            severity = 'moderate'
            icon = '🟡'
        else:
            severity = 'low'
            icon = '🟢'

        vif_data.append({
            'feature': col,
            'vif': round(float(vif), 2),
            'severity': severity,
            'icon': icon,
        })

    vif_data.sort(key=lambda x: x['vif'], reverse=True)

    flagged = [v for v in vif_data if v['vif'] > threshold]
    high_vif = [v for v in vif_data if v['vif'] > 5]

    recommendations = []
    if flagged:
        worst = flagged[0]
        recommendations.append(
            f'Feature "{worst["feature"]}" has VIF={worst["vif"]:.1f} — severe multicollinearity. '
            f'Consider dropping it or using PCA.'
        )
    if len(flagged) > 3:
        recommendations.append(
            f'{len(flagged)} features have VIF>{threshold}. Apply PCA or Ridge regression '
            f'to handle multicollinearity automatically.'
        )
    if not flagged:
        recommendations.append('No severe multicollinearity detected. Features are independent enough.')

    return {
        'vif_scores': vif_data,
        'flagged_features': [f['feature'] for f in flagged],
        'flagged_count': len(flagged),
        'max_vif': vif_data[0]['vif'] if vif_data else 0,
        'threshold': threshold,
        'total_features': len(vif_data),
        'recommendations': recommendations,
        'summary': (
            f'{len(flagged)} of {len(vif_data)} features exceed VIF threshold of {threshold}.'
            if flagged else 'All features have acceptable VIF levels.'
        ),
    }


def suggest_vif_fix(vif_result, X):
    """Simulate dropping the worst VIF feature and recompute."""
    flagged = vif_result.get('flagged_features', [])
    if not flagged:
        return {'message': 'No features need fixing.'}

    worst = flagged[0]
    X_reduced = X.drop(columns=[worst], errors='ignore')
    new_result = compute_vif(X_reduced, vif_result.get('threshold', 10.0))

    return {
        'dropped_feature': worst,
        'original_max_vif': vif_result.get('max_vif', 0),
        'new_max_vif': new_result.get('max_vif', 0),
        'improvement': round(vif_result.get('max_vif', 0) - new_result.get('max_vif', 0), 2),
        'new_flagged_count': new_result.get('flagged_count', 0),
    }
