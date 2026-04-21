"""
Autonomous ML Agent — Self-improving pipeline that iterates toward better performance.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif, f_regression


def run_autonomous_agent(model, X_train, y_train, X_test, y_test, problem_type='classification',
                          max_iterations=5, time_budget_seconds=300):
    """Run autonomous improvement loop."""
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    
    # Baseline
    base_model = clone(model)
    base_model.fit(X_train, y_train)
    base_score = float(cross_val_score(base_model, X_train, y_train, cv=3, scoring=scoring).mean())
    
    iterations = []
    best_score = base_score
    best_X_train = X_train.copy() if isinstance(X_train, pd.DataFrame) else X_train.copy()
    best_X_test = X_test.copy() if isinstance(X_test, pd.DataFrame) else X_test.copy()
    best_model = base_model
    no_improve_count = 0
    
    strategies = [
        ('Feature Selection', _try_feature_selection),
        ('Polynomial Features', _try_polynomial_features),
        ('Outlier Removal', _try_outlier_removal),
        ('Binning Features', _try_binning),
        ('Interaction Features', _try_interactions),
    ]
    
    for iteration in range(max_iterations):
        strategy_name, strategy_fn = strategies[iteration % len(strategies)]
        
        try:
            result = strategy_fn(
                clone(model), best_X_train, y_train, best_X_test, y_test,
                problem_type, scoring
            )
        except Exception as e:
            result = {'success': False, 'error': str(e)}
        
        iter_record = {
            'iteration': iteration + 1,
            'strategy': strategy_name,
            'previous_score': round(best_score, 6),
        }
        
        if result.get('success') and result.get('score', 0) > best_score:
            improvement = result['score'] - best_score
            best_score = result['score']
            best_X_train = result.get('X_train', best_X_train)
            best_X_test = result.get('X_test', best_X_test)
            best_model = result.get('model', best_model)
            no_improve_count = 0
            
            iter_record.update({
                'new_score': round(best_score, 6),
                'improvement': round(improvement, 6),
                'applied': True,
                'description': result.get('description', ''),
            })
        else:
            no_improve_count += 1
            iter_record.update({
                'new_score': round(best_score, 6),
                'improvement': 0,
                'applied': False,
                'reason': result.get('error', 'No improvement'),
            })
        
        iterations.append(iter_record)
        
        # Early stopping
        if no_improve_count >= 3:
            break
    
    total_improvement = best_score - base_score
    
    return {
        'baseline_score': round(base_score, 6),
        'final_score': round(best_score, 6),
        'total_improvement': round(total_improvement, 6),
        'improvement_pct': round(total_improvement / max(abs(base_score), 1e-10) * 100, 2),
        'iterations': iterations,
        'n_iterations': len(iterations),
        'strategies_tried': len(iterations),
        'strategies_applied': sum(1 for i in iterations if i.get('applied')),
        'converged': no_improve_count >= 3,
    }


def _try_feature_selection(model, X_train, y_train, X_test, y_test, problem_type, scoring):
    """Try selecting top-k features."""
    if isinstance(X_train, pd.DataFrame):
        X_tr = X_train.values
        X_te = X_test.values
    else:
        X_tr, X_te = X_train, X_test
    
    k = max(3, X_tr.shape[1] // 2)
    score_fn = f_classif if problem_type == 'classification' else f_regression
    selector = SelectKBest(score_fn, k=k)
    X_tr_sel = selector.fit_transform(X_tr, y_train)
    X_te_sel = selector.transform(X_te)
    
    m = clone(model)
    m.fit(X_tr_sel, y_train)
    score = float(cross_val_score(m, X_tr_sel, y_train, cv=3, scoring=scoring).mean())
    
    return {
        'success': True, 'score': score, 'model': m,
        'X_train': X_tr_sel, 'X_test': X_te_sel,
        'description': f'Selected top {k} features from {X_tr.shape[1]}'
    }


def _try_polynomial_features(model, X_train, y_train, X_test, y_test, problem_type, scoring):
    """Try adding polynomial/interaction features."""
    if isinstance(X_train, pd.DataFrame):
        X_tr = X_train.values
        X_te = X_test.values
    else:
        X_tr, X_te = X_train, X_test
    
    if X_tr.shape[1] > 10:
        # Only use top 10 features
        X_tr = X_tr[:, :10]
        X_te = X_te[:, :10]
    
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_tr_poly = poly.fit_transform(X_tr)
    X_te_poly = poly.transform(X_te)
    
    m = clone(model)
    m.fit(X_tr_poly, y_train)
    score = float(cross_val_score(m, X_tr_poly, y_train, cv=3, scoring=scoring).mean())
    
    return {
        'success': True, 'score': score, 'model': m,
        'X_train': X_tr_poly, 'X_test': X_te_poly,
        'description': f'Added polynomial interaction features ({X_tr.shape[1]} → {X_tr_poly.shape[1]})'
    }


def _try_outlier_removal(model, X_train, y_train, X_test, y_test, problem_type, scoring):
    """Try removing outlier training samples."""
    if isinstance(X_train, pd.DataFrame):
        X_tr = X_train.values
    else:
        X_tr = X_train
    
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(contamination=0.05, random_state=42)
    mask = iso.fit_predict(X_tr) == 1
    
    if mask.sum() < len(y_train) * 0.8:
        return {'success': False, 'error': 'Too many points removed'}
    
    X_clean = X_tr[mask]
    y_clean = np.array(y_train)[mask]
    
    m = clone(model)
    m.fit(X_clean, y_clean)
    score = float(cross_val_score(m, X_clean, y_clean, cv=3, scoring=scoring).mean())
    
    return {
        'success': True, 'score': score, 'model': m,
        'X_train': X_clean, 'X_test': X_test if isinstance(X_test, np.ndarray) else X_test.values,
        'description': f'Removed {(~mask).sum()} outlier samples ({(~mask).mean()*100:.1f}%)'
    }


def _try_binning(model, X_train, y_train, X_test, y_test, problem_type, scoring):
    """Try binning numeric features."""
    if isinstance(X_train, pd.DataFrame):
        X_tr = X_train.values.copy()
        X_te = X_test.values.copy()
    else:
        X_tr = X_train.copy()
        X_te = X_test.copy()
    
    from sklearn.preprocessing import KBinsDiscretizer
    n_bins = 10
    kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X_tr_binned = kbd.fit_transform(X_tr)
    X_te_binned = kbd.transform(X_te)
    
    X_tr_combined = np.hstack([X_tr, X_tr_binned])
    X_te_combined = np.hstack([X_te, X_te_binned])
    
    m = clone(model)
    m.fit(X_tr_combined, y_train)
    score = float(cross_val_score(m, X_tr_combined, y_train, cv=3, scoring=scoring).mean())
    
    return {
        'success': True, 'score': score, 'model': m,
        'X_train': X_tr_combined, 'X_test': X_te_combined,
        'description': f'Added {n_bins}-bin discretized features'
    }


def _try_interactions(model, X_train, y_train, X_test, y_test, problem_type, scoring):
    """Try adding ratio/difference features between top pairs."""
    if isinstance(X_train, pd.DataFrame):
        X_tr = X_train.values.copy()
        X_te = X_test.values.copy()
    else:
        X_tr = X_train.copy()
        X_te = X_test.copy()
    
    n_features = X_tr.shape[1]
    if n_features < 2:
        return {'success': False, 'error': 'Need at least 2 features'}
    
    new_feats_train = []
    new_feats_test = []
    
    for i in range(min(5, n_features)):
        for j in range(i + 1, min(5, n_features)):
            # Ratio
            denom_tr = np.abs(X_tr[:, j]) + 1e-8
            denom_te = np.abs(X_te[:, j]) + 1e-8
            new_feats_train.append(X_tr[:, i] / denom_tr)
            new_feats_test.append(X_te[:, i] / denom_te)
            # Difference
            new_feats_train.append(X_tr[:, i] - X_tr[:, j])
            new_feats_test.append(X_te[:, i] - X_te[:, j])
    
    X_tr_aug = np.column_stack([X_tr] + new_feats_train)
    X_te_aug = np.column_stack([X_te] + new_feats_test)
    
    m = clone(model)
    m.fit(X_tr_aug, y_train)
    score = float(cross_val_score(m, X_tr_aug, y_train, cv=3, scoring=scoring).mean())
    
    return {
        'success': True, 'score': score, 'model': m,
        'X_train': X_tr_aug, 'X_test': X_te_aug,
        'description': f'Added {len(new_feats_train)} ratio/difference interaction features'
    }
