"""
AutoML Problem Solver - Model Trainer
Trains multiple ML models, cross-validates, tunes hyperparameters, and returns a leaderboard.
"""

import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


# Hyperparameter grids for tuning
CLASSIFICATION_PARAMS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [15, 31, 63],
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [500, 1000],
    },
}

REGRESSION_PARAMS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, -1],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    },
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100],
    },
    'Lasso': {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
    },
}


def get_models(problem_type, class_weight_balanced=False):
    """Get a dict of model_name -> model_instance based on problem type."""
    if problem_type == 'classification':
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(probability=True, random_state=42, max_iter=5000),
            'KNN': KNeighborsClassifier(),
        }
        if class_weight_balanced:
            models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
            models['SVM'] = SVC(probability=True, random_state=42, max_iter=5000, class_weight='balanced')
        
        if HAS_XGB:
            models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)
        if HAS_LGBM:
            models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
    else:
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
        }
        if HAS_XGB:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
        if HAS_LGBM:
            models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
    
    return models


def train_models(df, profile, transform_metadata, output_dir, progress_callback=None):
    """
    Train all models, cross-validate, tune top models, and return results.
    
    Returns:
        dict: Training results including leaderboard, best model info, metrics, feature importance
    """
    target_col = transform_metadata.get('target_column') or profile.get('target_column')
    problem_type = transform_metadata.get('problem_type') or profile.get('problem_type', 'classification')
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Ensure all features are numeric
    X = X.select_dtypes(include=[np.number])
    
    if X.empty:
        return {'error': 'No numeric features available for training.'}
    
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
    
    # Get models
    models = get_models(problem_type)
    
    # Train and evaluate all models
    leaderboard = []
    trained_models = {}
    cv_results = {}
    
    total_models = len(models)
    
    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(f'Training {name}...', int((idx / total_models) * 100))
        
        try:
            # Train
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Cross-validation
            scoring = 'accuracy' if problem_type == 'classification' else 'r2'
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
            cv_results[name] = {
                'mean': round(float(cv_scores.mean()), 4),
                'std': round(float(cv_scores.std()), 4),
                'scores': [round(float(s), 4) for s in cv_scores],
            }
            
            # Compute metrics
            if problem_type == 'classification':
                is_binary = len(np.unique(y)) == 2
                avg = 'binary' if is_binary else 'weighted'
                
                metrics = {
                    'accuracy': round(float(accuracy_score(y_test, y_test_pred)), 4),
                    'precision': round(float(precision_score(y_test, y_test_pred, average=avg, zero_division=0)), 4),
                    'recall': round(float(recall_score(y_test, y_test_pred, average=avg, zero_division=0)), 4),
                    'f1': round(float(f1_score(y_test, y_test_pred, average=avg, zero_division=0)), 4),
                    'train_accuracy': round(float(accuracy_score(y_train, y_train_pred)), 4),
                    'cv_mean': cv_results[name]['mean'],
                    'cv_std': cv_results[name]['std'],
                }
                
                # ROC-AUC
                try:
                    if is_binary and hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics['roc_auc'] = round(float(roc_auc_score(y_test, y_proba)), 4)
                    else:
                        metrics['roc_auc'] = None
                except Exception:
                    metrics['roc_auc'] = None
                
                primary_metric = metrics['accuracy']
                
                # Confusion matrix
                cm = confusion_matrix(y_test, y_test_pred)
                metrics['confusion_matrix'] = cm.tolist()
            else:
                metrics = {
                    'mae': round(float(mean_absolute_error(y_test, y_test_pred)), 4),
                    'mse': round(float(mean_squared_error(y_test, y_test_pred)), 4),
                    'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_test_pred))), 4),
                    'r2': round(float(r2_score(y_test, y_test_pred)), 4),
                    'train_r2': round(float(r2_score(y_train, y_train_pred)), 4),
                    'cv_mean': cv_results[name]['mean'],
                    'cv_std': cv_results[name]['std'],
                }
                primary_metric = metrics['r2']
            
            leaderboard.append({
                'rank': 0,  # Will be set after sorting
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
    
    # Sort leaderboard
    leaderboard.sort(key=lambda x: x['primary_metric'], reverse=True)
    for i, entry in enumerate(leaderboard):
        entry['rank'] = i + 1
    
    # Get best model name
    best_model_name = leaderboard[0]['model'] if leaderboard else None
    
    # Hyperparameter tuning for top 3
    if progress_callback:
        progress_callback('Tuning top models...', 80)
    
    top_n = min(3, len([l for l in leaderboard if l['primary_metric'] > -999]))
    tuned_results = []
    
    param_grids = CLASSIFICATION_PARAMS if problem_type == 'classification' else REGRESSION_PARAMS
    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
    
    for entry in leaderboard[:top_n]:
        name = entry['model']
        if name in param_grids and name in trained_models:
            try:
                model = get_models(problem_type)[name]
                search = RandomizedSearchCV(
                    model, param_grids[name],
                    n_iter=10, cv=3, scoring=scoring,
                    random_state=42, n_jobs=-1
                )
                search.fit(X_train, y_train)
                
                # Evaluate tuned model
                y_test_pred_tuned = search.best_estimator_.predict(X_test)
                
                if problem_type == 'classification':
                    tuned_score = round(float(accuracy_score(y_test, y_test_pred_tuned)), 4)
                else:
                    tuned_score = round(float(r2_score(y_test, y_test_pred_tuned)), 4)
                
                tuned_results.append({
                    'model': name,
                    'original_score': entry['primary_metric'],
                    'tuned_score': tuned_score,
                    'improvement': round(tuned_score - entry['primary_metric'], 4),
                    'best_params': {k: (int(v) if isinstance(v, (np.integer,)) else 
                                       float(v) if isinstance(v, (np.floating,)) else v) 
                                   for k, v in search.best_params_.items()},
                })
                
                # Update model if improved
                if tuned_score >= entry['primary_metric']:
                    trained_models[name] = search.best_estimator_
                    entry['primary_metric'] = tuned_score
                    entry['metrics']['tuned'] = True
                    # Also update train score for display
                    y_train_pred_tuned = search.best_estimator_.predict(X_train)
                    if problem_type == 'classification':
                        entry['metrics']['accuracy'] = tuned_score
                        entry['metrics']['train_accuracy'] = round(float(accuracy_score(y_train, y_train_pred_tuned)), 4)
                    else:
                        entry['metrics']['r2'] = tuned_score
                        entry['metrics']['train_r2'] = round(float(r2_score(y_train, y_train_pred_tuned)), 4)
            except Exception:
                pass
    
    # Re-sort after tuning
    leaderboard.sort(key=lambda x: x['primary_metric'], reverse=True)
    for i, entry in enumerate(leaderboard):
        entry['rank'] = i + 1
    
    best_model_name = leaderboard[0]['model'] if leaderboard else None
    
    # Feature importance
    feature_importance = _get_feature_importance(trained_models.get(best_model_name), X.columns.tolist())
    
    # Save best model
    best_model_path = None
    if best_model_name and best_model_name in trained_models:
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, 'best_model.pkl')
        joblib.dump(trained_models[best_model_name], best_model_path)
    
    if progress_callback:
        progress_callback('Training complete!', 100)
    
    # Training context for recommender
    training_context = {
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'n_features': X.shape[1],
        'cv_results': cv_results,
        'leaderboard': leaderboard,
        'problem_type': problem_type,
        'tuned_results': tuned_results,
        'trained_models': trained_models,
        'feature_importance': feature_importance,
    }
    
    results = {
        'leaderboard': leaderboard,
        'best_model': best_model_name,
        'best_score': leaderboard[0]['primary_metric'] if leaderboard else None,
        'primary_metric_name': 'accuracy' if problem_type == 'classification' else 'r2',
        'problem_type': problem_type,
        'feature_importance': feature_importance,
        'tuned_results': tuned_results,
        'best_model_path': best_model_path,
        'training_context': training_context,
    }
    
    return results


def _get_feature_importance(model, feature_names):
    """Extract feature importance from the best model."""
    if model is None:
        return []
    
    importance = None
    
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_).flatten()
        if len(importance) != len(feature_names):
            importance = None
    
    if importance is not None:
        # Normalize
        total = importance.sum()
        if total > 0:
            importance = importance / total
        
        fi = [{'feature': name, 'importance': round(float(imp), 4)} 
              for name, imp in zip(feature_names, importance)]
        fi.sort(key=lambda x: x['importance'], reverse=True)
        return fi[:20]  # Top 20
    
    return []
