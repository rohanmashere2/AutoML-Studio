"""
AutoML Problem Solver - Retrainer
Applies recommendations and retrains models with optimized settings.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)

from ml_engine.trainer import get_models, CLASSIFICATION_PARAMS, REGRESSION_PARAMS


def retrain_with_recommendations(df, profile, transform_metadata, recommendations, 
                                  original_results, output_dir, progress_callback=None):
    """
    Apply recommendations and retrain models.
    
    Returns:
        dict: Retrain results with before/after comparison
    """
    target_col = transform_metadata.get('target_column') or profile.get('target_column')
    problem_type = transform_metadata.get('problem_type') or profile.get('problem_type', 'classification')
    
    # Collect all actions from recommendations
    actions = {}
    applied_recommendations = []
    applied_rec_details = []  # Full details for report
    
    for rec in recommendations:
        action = rec.get('action', {})
        if action:
            actions.update(action)
            applied_recommendations.append(rec['title'])
            applied_rec_details.append({
                'title': rec['title'],
                'description': rec['description'],
                'impact': rec.get('impact', 'medium'),
                'icon': rec.get('icon', '💡'),
                'category': rec.get('category', 'general'),
                'action_taken': [],  # Will be populated below
                'status': 'applied',
            })
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.select_dtypes(include=[np.number])
    
    if X.empty:
        return {'error': 'No numeric features available for retraining.'}
    
    if progress_callback:
        progress_callback('Applying recommendations...', 10)
    
    # Apply Feature Engineering recommendations
    feature_changes = []
    
    # Polynomial features
    if actions.get('polynomial_features'):
        degree = actions.get('degree', 2)
        if X.shape[1] <= 8:  # Only if not too many features
            try:
                poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
                X_poly = poly.fit_transform(X)
                feature_names = poly.get_feature_names_out(X.columns)
                X = pd.DataFrame(X_poly, columns=feature_names, index=X.index)
                change_msg = f'Added polynomial interaction features (degree={degree}): {X.shape[1]} total features'
                feature_changes.append(change_msg)
                # Tag which recs caused this
                for rd in applied_rec_details:
                    if rd['category'] in ('features', 'fitting', 'performance'):
                        rd['action_taken'].append(change_msg)
            except Exception:
                pass
    
    # PCA
    if actions.get('apply_pca'):
        n_components = min(X.shape[1], max(5, int(X.shape[1] * 0.8)))
        try:
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X)
            explained = sum(pca.explained_variance_ratio_)
            X = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)], index=X.index)
            change_msg = f'Applied PCA: {n_components} components ({explained:.1%} variance explained)'
            feature_changes.append(change_msg)
            for rd in applied_rec_details:
                if rd['category'] == 'features':
                    rd['action_taken'].append(change_msg)
        except Exception:
            pass
    
    if progress_callback:
        progress_callback('Retraining models...', 30)
    
    # Train/test split
    stratify = y if problem_type == 'classification' else None
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=stratify
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    
    # Get models with optimized settings
    use_balanced = actions.get('class_weight') == 'balanced'
    models = get_models(problem_type, class_weight_balanced=use_balanced)
    
    # If regularization recommended, add regularized versions
    if actions.get('regularization') or actions.get('reduce_complexity'):
        if problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            models['Random Forest (Regularized)'] = RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_split=10,
                min_samples_leaf=4, random_state=42,
                class_weight='balanced' if use_balanced else None
            )
            models['Gradient Boosting (Regularized)'] = GradientBoostingClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
        else:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            models['Random Forest (Regularized)'] = RandomForestRegressor(
                n_estimators=200, max_depth=5, min_samples_split=10, random_state=42
            )
            models['Gradient Boosting (Regularized)'] = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42
            )
    
    # If increase complexity recommended
    if actions.get('increase_complexity'):
        if problem_type == 'classification':
            from sklearn.ensemble import RandomForestClassifier
            models['Random Forest (Complex)'] = RandomForestClassifier(
                n_estimators=500, max_depth=None, random_state=42,
                class_weight='balanced' if use_balanced else None
            )
        else:
            from sklearn.ensemble import RandomForestRegressor
            models['Random Forest (Complex)'] = RandomForestRegressor(
                n_estimators=500, max_depth=None, random_state=42
            )
    
    # Determine scoring metric
    if actions.get('optimize_metric') == 'f1':
        scoring = 'f1_weighted' if problem_type == 'classification' else 'r2'
    else:
        scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    
    # Train all models
    leaderboard = []
    trained_models = {}
    total_models = len(models)
    
    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(f'Retraining {name}...', 30 + int((idx / total_models) * 50))
        
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            
            if problem_type == 'classification':
                is_binary = len(np.unique(y)) == 2
                avg = 'binary' if is_binary else 'weighted'
                
                metrics = {
                    'accuracy': round(float(accuracy_score(y_test, y_test_pred)), 4),
                    'precision': round(float(precision_score(y_test, y_test_pred, average=avg, zero_division=0)), 4),
                    'recall': round(float(recall_score(y_test, y_test_pred, average=avg, zero_division=0)), 4),
                    'f1': round(float(f1_score(y_test, y_test_pred, average=avg, zero_division=0)), 4),
                    'train_accuracy': round(float(accuracy_score(y_train, y_train_pred)), 4),
                    'cv_mean': round(float(cv_scores.mean()), 4),
                    'cv_std': round(float(cv_scores.std()), 4),
                }
                cm = confusion_matrix(y_test, y_test_pred)
                metrics['confusion_matrix'] = cm.tolist()
                primary_metric = metrics['accuracy']
            else:
                metrics = {
                    'mae': round(float(mean_absolute_error(y_test, y_test_pred)), 4),
                    'mse': round(float(mean_squared_error(y_test, y_test_pred)), 4),
                    'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 4),
                    'r2': round(float(r2_score(y_test, y_test_pred)), 4),
                    'train_r2': round(float(r2_score(y_train, y_train_pred)), 4),
                    'cv_mean': round(float(cv_scores.mean()), 4),
                    'cv_std': round(float(cv_scores.std()), 4),
                }
                primary_metric = metrics['r2']
            
            leaderboard.append({
                'rank': 0,
                'model': name,
                'primary_metric': round(float(primary_metric), 4),
                'metrics': metrics,
            })
        except Exception as e:
            leaderboard.append({
                'rank': 0,
                'model': name,
                'primary_metric': -999,
                'metrics': {'error': str(e)},
            })
    
    # Hyperparameter tuning for top 3
    if progress_callback:
        progress_callback('Tuning retrained models...', 85)
    
    leaderboard.sort(key=lambda x: x['primary_metric'], reverse=True)
    
    param_grids = CLASSIFICATION_PARAMS if problem_type == 'classification' else REGRESSION_PARAMS
    
    for entry in leaderboard[:3]:
        name = entry['model']
        # Strip "(Regularized)" etc. for param lookup
        base_name = name.split(' (')[0]
        if base_name in param_grids and name in trained_models:
            try:
                model_cls = get_models(problem_type, use_balanced)
                if base_name in model_cls:
                    model = model_cls[base_name]
                    search = RandomizedSearchCV(
                        model, param_grids[base_name],
                        n_iter=15, cv=3, scoring=scoring,
                        random_state=42, n_jobs=-1
                    )
                    search.fit(X_train, y_train)
                    
                    y_test_pred = search.best_estimator_.predict(X_test)
                    
                    if problem_type == 'classification':
                        tuned_score = round(float(accuracy_score(y_test, y_test_pred)), 4)
                    else:
                        tuned_score = round(float(r2_score(y_test, y_test_pred)), 4)
                    
                    if tuned_score >= entry['primary_metric']:
                        trained_models[name] = search.best_estimator_
                        entry['primary_metric'] = tuned_score
                        y_train_pred_tuned = search.best_estimator_.predict(X_train)
                        if problem_type == 'classification':
                            entry['metrics']['accuracy'] = tuned_score
                            entry['metrics']['train_accuracy'] = round(float(accuracy_score(y_train, y_train_pred_tuned)), 4)
                        else:
                            entry['metrics']['r2'] = tuned_score
                            entry['metrics']['train_r2'] = round(float(r2_score(y_train, y_train_pred_tuned)), 4)
            except Exception:
                pass
    
    # Final sort with ranks
    leaderboard.sort(key=lambda x: x['primary_metric'], reverse=True)
    for i, entry in enumerate(leaderboard):
        entry['rank'] = i + 1
    
    best_model_name = leaderboard[0]['model'] if leaderboard else None
    
    # Feature importance
    feature_importance = []
    if best_model_name and best_model_name in trained_models:
        best_model = trained_models[best_model_name]
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            total = importances.sum()
            if total > 0:
                importances = importances / total
            fi = [{'feature': name, 'importance': round(float(imp), 4)}
                  for name, imp in zip(X.columns, importances)]
            fi.sort(key=lambda x: x['importance'], reverse=True)
            feature_importance = fi[:20]
    
    # Save improved model
    best_model_path = None
    if best_model_name and best_model_name in trained_models:
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, 'improved_model.pkl')
        joblib.dump(trained_models[best_model_name], best_model_path)
    
    if progress_callback:
        progress_callback('Retraining complete!', 100)
    
    # Before vs After comparison
    original_best = original_results.get('best_score', 0)
    new_best = leaderboard[0]['primary_metric'] if leaderboard else 0
    improvement = round(new_best - original_best, 4)
    
    # Build per-model before/after comparison
    orig_leaderboard = original_results.get('leaderboard', [])
    model_comparison = []
    for entry in leaderboard:
        orig_entry = next((o for o in orig_leaderboard if o['model'] == entry['model']), None)
        if orig_entry and orig_entry['primary_metric'] > -999 and entry['primary_metric'] > -999:
            delta = round(entry['primary_metric'] - orig_entry['primary_metric'], 4)
            model_comparison.append({
                'model': entry['model'],
                'before': orig_entry['primary_metric'],
                'after': entry['primary_metric'],
                'delta': delta,
                'improved': delta > 0,
            })
    
    # Tag action_taken for class_weight / regularization recs
    if actions.get('class_weight') == 'balanced':
        for rd in applied_rec_details:
            if rd['category'] == 'imbalance':
                rd['action_taken'].append('Applied class_weight="balanced" to all compatible models')
    if actions.get('regularization') or actions.get('reduce_complexity'):
        for rd in applied_rec_details:
            if rd['category'] == 'fitting':
                rd['action_taken'].append('Added regularized model variants (lower max_depth, learning_rate=0.05)')
    if actions.get('increase_complexity'):
        for rd in applied_rec_details:
            if rd['category'] == 'fitting':
                rd['action_taken'].append('Added complex model variants (higher n_estimators, unlimited depth)')
    if actions.get('optimize_metric') == 'f1':
        for rd in applied_rec_details:
            if rd['category'] == 'imbalance':
                rd['action_taken'].append('Optimized scoring metric changed to F1 (weighted)')
    
    # Mark recs with no concrete action
    for rd in applied_rec_details:
        if not rd['action_taken']:
            rd['action_taken'].append('Recommendation noted — applied general model tuning')
    
    # Overall verdict
    n_improved = sum(1 for m in model_comparison if m['improved'])
    n_total = len(model_comparison)
    
    if improvement > 0.02:
        verdict = 'excellent'
        verdict_text = f'🎉 Excellent! Retrain improved best score by +{improvement*100:.2f}%. {n_improved}/{n_total} models improved.'
    elif improvement > 0:
        verdict = 'good'
        verdict_text = f'✅ Good improvement! Best score improved by +{improvement*100:.2f}%. {n_improved}/{n_total} models improved.'
    elif improvement == 0:
        verdict = 'neutral'
        verdict_text = f'⚖️ No change in best score. Recommendations did not significantly affect this dataset. Consider collecting more data or engineering domain-specific features.'
    else:
        verdict = 'mixed'
        verdict_text = f'⚠️ Best score slightly decreased by {improvement*100:.2f}%. This can happen when regularization reduces overfitting — check if CV scores improved.'
    
    # Retrain report
    retrain_report = {
        'applied_rec_details': applied_rec_details,
        'model_comparison': model_comparison,
        'n_models_improved': n_improved,
        'n_models_total': n_total,
        'verdict': verdict,
        'verdict_text': verdict_text,
    }
    
    results = {
        'leaderboard': leaderboard,
        'best_model': best_model_name,
        'best_score': new_best,
        'original_best_score': original_best,
        'improvement': improvement,
        'improvement_pct': round(improvement * 100, 2),
        'primary_metric_name': 'accuracy' if problem_type == 'classification' else 'r2',
        'problem_type': problem_type,
        'feature_importance': feature_importance,
        'applied_recommendations': applied_recommendations,
        'feature_changes': feature_changes,
        'best_model_path': best_model_path,
        'retrain_report': retrain_report,
    }
    
    return results
