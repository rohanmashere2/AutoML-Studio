"""
AutoML Problem Solver - Model Trainer
Trains multiple ML models, cross-validates, tunes hyperparameters, and returns a leaderboard.
"""

import logging
import time
import pandas as pd
import numpy as np
import joblib
import os
import warnings
import copy
warnings.filterwarnings('ignore')

from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.naive_bayes import GaussianNB
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

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


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
    'CatBoost': {
        'iterations': [100, 200, 500],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7],
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [500, 1000],
    },
    'Extra Trees': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'HistGBM': {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10, None],
        'max_leaf_nodes': [15, 31, 63, 127, None],
        'min_samples_leaf': [5, 10, 20, 50],
        'l2_regularization': [0.0, 0.01, 0.1, 1.0],
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
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
    'CatBoost': {
        'iterations': [100, 200, 500],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7],
    },
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100],
    },
    'Lasso': {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
    },
    'Extra Trees': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
    },
    'ElasticNet': {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    },
    'HistGBM': {
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_iter': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 10, None],
        'max_leaf_nodes': [15, 31, 63, 127, None],
        'min_samples_leaf': [5, 10, 20, 50],
        'l2_regularization': [0.0, 0.01, 0.1, 1.0],
    },
    'AdaBoost': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0],
    },
}


def _detect_gpu():
    """Check if CUDA GPU is available for accelerated training."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

HAS_GPU = _detect_gpu()


def get_models(problem_type, class_weight_balanced=False, n_samples=None):
    """Get an OrderedDict of model_name -> model_instance, ordered fast-first.
    
    Args:
        problem_type: 'classification' or 'regression'
        class_weight_balanced: Use balanced class weights for supported models
        n_samples: Number of training samples (used to skip slow models on large data)
    """
    # GPU-accelerated params for boosters
    xgb_extra = {'tree_method': 'gpu_hist', 'device': 'cuda'} if HAS_GPU else {}
    lgbm_extra = {'device': 'gpu'} if HAS_GPU else {}

    if problem_type == 'classification':
        # Ordered: fast baselines → medium ensembles → slow boosters → very slow
        models = OrderedDict([
            ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
            ('Naive Bayes', GaussianNB()),
            ('KNN', KNeighborsClassifier()),
            ('AdaBoost', AdaBoostClassifier(n_estimators=100, random_state=42)),
            ('HistGBM', HistGradientBoostingClassifier(max_iter=100, random_state=42)),
            ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('Extra Trees', ExtraTreesClassifier(n_estimators=100, random_state=42)),
            ('Gradient Boosting', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ])
        if class_weight_balanced:
            models['Logistic Regression'] = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
            models['Random Forest'] = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        
        if HAS_LGBM:
            models['LightGBM'] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1, **lgbm_extra)
        if HAS_XGB:
            models['XGBoost'] = XGBClassifier(n_estimators=100, random_state=42, verbosity=0, **xgb_extra)
        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostClassifier(iterations=100, random_state=42, verbose=0)
        
        # Skip SVM on large datasets (O(n²) complexity)
        if n_samples is None or n_samples <= 10000:
            if class_weight_balanced:
                models['SVM'] = SVC(probability=True, random_state=42, max_iter=5000, class_weight='balanced')
            else:
                models['SVM'] = SVC(probability=True, random_state=42, max_iter=5000)
    else:
        models = OrderedDict([
            ('Linear Regression', LinearRegression()),
            ('Ridge', Ridge(random_state=42)),
            ('Lasso', Lasso(random_state=42)),
            ('ElasticNet', ElasticNet(random_state=42)),
            ('KNN', KNeighborsRegressor()),
            ('AdaBoost', AdaBoostRegressor(n_estimators=100, random_state=42)),
            ('HistGBM', HistGradientBoostingRegressor(max_iter=100, random_state=42)),
            ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
            ('Extra Trees', ExtraTreesRegressor(n_estimators=100, random_state=42)),
            ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, random_state=42)),
        ])
        if HAS_LGBM:
            models['LightGBM'] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1, **lgbm_extra)
        if HAS_XGB:
            models['XGBoost'] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0, **xgb_extra)
        if HAS_CATBOOST:
            models['CatBoost'] = CatBoostRegressor(iterations=100, random_state=42, verbose=0)
        
        # Skip SVR on large datasets
        if n_samples is None or n_samples <= 10000:
            models['SVR'] = SVR()
    
    return models


def _validate_training_data(X, y, problem_type):
    """Validate data before training. Returns list of error strings."""
    errors = []
    if X.empty or X.shape[1] == 0:
        errors.append("No features available for training after preprocessing.")
    if y is None or len(y) == 0:
        errors.append("Target column is empty.")
    elif y.nunique() == 1:
        errors.append(f"Target has only 1 unique value ({y.iloc[0]}). Cannot train.")
    if len(X) < 10:
        errors.append(f"Only {len(X)} samples. Need at least 10 for training.")
    if X.isnull().any().any():
        # Auto-fix: fill remaining NaN with 0
        X.fillna(0, inplace=True)
    if problem_type == 'classification' and y.nunique() > 0.5 * len(y) and y.nunique() > 50:
        errors.append(f"Target has {y.nunique()} unique values out of {len(y)} rows. This looks like regression, not classification.")
    return errors


def train_models(df, profile, transform_metadata, output_dir, progress_callback=None, time_budget_seconds=None):
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
    
    # Fallback-encode any remaining non-numeric columns instead of dropping
    remaining_cats = X.select_dtypes(include=['object', 'category']).columns.tolist()
    if remaining_cats:
        logger.warning(f"Fallback-encoding {len(remaining_cats)} remaining non-numeric columns: {remaining_cats}")
        for col in remaining_cats:
            X[col] = X[col].astype('category').cat.codes

    # Now ensure all numeric (should be after encoding)
    X = X.select_dtypes(include=[np.number])

    # Clean inf/NaN — XGBoost crashes on these
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # For classification: ensure labels are contiguous integers (0, 1, 2...)
    # XGBoost requires this — raw labels like [1, 3, 5] will crash it
    if problem_type == 'classification':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    
    if X.empty:
        return {'error': 'No numeric features available for training.'}
    
    # Pre-training validation
    validation_errors = _validate_training_data(X, y, problem_type)
    if validation_errors:
        return {'error': '; '.join(validation_errors)}
    
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
    
    # Scale features AFTER split to prevent data leakage
    from ml_engine.transformer import fit_scaler, apply_scaler
    numeric_cols_to_scale = transform_metadata.get('numeric_cols_to_scale', [])
    fitted_scaler = fit_scaler(X_train, numeric_cols_to_scale)
    if fitted_scaler is not None:
        X_train = apply_scaler(X_train, fitted_scaler, numeric_cols_to_scale)
        X_test = apply_scaler(X_test, fitted_scaler, numeric_cols_to_scale)
        # Store fitted scaler in transform_metadata for inference
        transform_metadata['scaler'] = fitted_scaler
    
    # Force consistent dtypes — prevents XGBoost DataFrame.dtypes errors
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    X_train.columns = [str(c) for c in X_train.columns]
    X_test.columns = [str(c) for c in X_test.columns]
    
    # Get models (pass n_samples to skip slow models on large datasets)
    models = get_models(problem_type, n_samples=len(X_train))
    
    # Time budget tracking
    training_start_time = time.time()
    
    # Train and evaluate all models
    leaderboard = []
    trained_models = {}
    cv_results = {}
    
    total_models = len(models)
    
    # Smart CV strategy
    n_samples = len(X_train)
    if n_samples < 100:
        n_folds = min(10, n_samples)
    elif n_samples > 50000:
        n_folds = 3
    else:
        n_folds = 5

    if problem_type == 'classification':
        cv_strategy = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        cv_strategy = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for idx, (name, model) in enumerate(models.items()):
        # Check time budget before each model
        if time_budget_seconds is not None:
            elapsed = time.time() - training_start_time
            remaining = time_budget_seconds - elapsed
            if remaining < 30:
                logger.info(f"Time budget exhausted after {idx}/{total_models} models ({elapsed:.0f}s). Skipping remaining.")
                break
        
        if progress_callback:
            progress_callback(f'Training {name}...', int((idx / total_models) * 100))
        
        model_start_time = time.time()
        try:
            # Use early stopping for boosting models (try plain fit first as safety)
            if name in ('XGBoost', 'LightGBM'):
                try:
                    # Step 1: Plain fit (should always work)
                    plain_model = get_models(problem_type).get(name)
                    plain_model.fit(X_train, y_train)
                    model = plain_model
                    # Step 2: Try early stopping as an upgrade
                    try:
                        es_model = get_models(problem_type).get(name)
                        es_model.set_params(early_stopping_rounds=20)
                        es_model.fit(X_train, y_train,
                                     eval_set=[(X_test, y_test)],
                                     verbose=False)
                        model = es_model
                    except Exception:
                        pass  # Keep the plain-fit model
                except Exception as fit_err:
                    logger.warning(f"{name} plain fit failed: {fit_err}")
                    raise  # Re-raise to be caught by outer except
            elif name == 'CatBoost':
                try:
                    model.fit(X_train, y_train,
                              eval_set=(X_test, y_test),
                              early_stopping_rounds=20,
                              verbose=False)
                except Exception:
                    model = get_models(problem_type).get(name)
                    model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            trained_models[name] = model
            
            # Predict
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Cross-validation
            scoring = 'accuracy' if problem_type == 'classification' else 'r2'
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring=scoring)
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
                
                # ROC-AUC (binary + multiclass)
                try:
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test)
                        if is_binary:
                            metrics['roc_auc'] = round(float(roc_auc_score(y_test, y_proba[:, 1])), 4)
                        else:
                            metrics['roc_auc'] = round(float(roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')), 4)
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
                # Additional regression metrics
                metrics['mape'] = round(float(np.mean(np.abs((y_test - y_test_pred) / np.where(y_test == 0, 1, y_test))) * 100), 2)
                metrics['max_error'] = round(float(np.max(np.abs(y_test - y_test_pred))), 4)
                n = len(y_test)
                p = X_test.shape[1]
                r2_val = metrics['r2']
                metrics['adjusted_r2'] = round(float(1 - (1 - r2_val) * (n - 1) / max(n - p - 1, 1)), 4)
                metrics['explained_variance'] = round(float(1 - np.var(y_test - y_test_pred) / max(np.var(y_test), 1e-10)), 4)
                primary_metric = metrics['r2']
            
            train_time = round(time.time() - model_start_time, 2)
            leaderboard.append({
                'rank': 0,  # Will be set after sorting
                'model': name,
                'primary_metric': round(float(primary_metric), 4),
                'metrics': metrics,
                'training_time': train_time,
            })
        except Exception as e:
            logger.warning(f"Model {name} failed: {e}")
            leaderboard.append({
                'rank': 0,
                'model': name,
                'primary_metric': -999,
                'metrics': {'error': str(e)},
                'failed': True,
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
                tuned_model, best_params, tuned_score = _optuna_tune(
                    name, problem_type, param_grids[name],
                    X_train, y_train, X_test, y_test, scoring, n_trials=100
                )
                
                tuned_results.append({
                    'model': name,
                    'original_score': entry['primary_metric'],
                    'tuned_score': tuned_score,
                    'improvement': round(tuned_score - entry['primary_metric'], 4),
                    'best_params': best_params,
                    'method': 'optuna' if HAS_OPTUNA else 'random',
                })
                
                # Update model if improved
                if tuned_score >= entry['primary_metric'] and tuned_model is not None:
                    trained_models[name] = tuned_model
                    entry['primary_metric'] = tuned_score
                    entry['metrics']['tuned'] = True
                    y_train_pred_tuned = tuned_model.predict(X_train)
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
    
    # Build stacking ensemble from top models
    try:
        top_model_names = [e['model'] for e in leaderboard[:3] if e['primary_metric'] > -999 and e['model'] in trained_models]
        if len(top_model_names) >= 2:
            from sklearn.ensemble import StackingClassifier, StackingRegressor
            estimators = [(name, trained_models[name]) for name in top_model_names]
            if problem_type == 'classification':
                stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=-1, passthrough=False)
            else:
                stack = StackingRegressor(estimators=estimators, final_estimator=Ridge(), cv=3, n_jobs=-1, passthrough=False)
            stack.fit(X_train, y_train)
            y_stack_pred = stack.predict(X_test)
            if problem_type == 'classification':
                stack_score = round(float(accuracy_score(y_test, y_stack_pred)), 4)
                stack_metrics = {
                    'accuracy': stack_score,
                    'precision': round(float(precision_score(y_test, y_stack_pred, average='weighted', zero_division=0)), 4),
                    'recall': round(float(recall_score(y_test, y_stack_pred, average='weighted', zero_division=0)), 4),
                    'f1': round(float(f1_score(y_test, y_stack_pred, average='weighted', zero_division=0)), 4),
                }
            else:
                stack_score = round(float(r2_score(y_test, y_stack_pred)), 4)
                stack_metrics = {
                    'r2': stack_score,
                    'mae': round(float(mean_absolute_error(y_test, y_stack_pred)), 4),
                    'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_stack_pred))), 4),
                }
            leaderboard.append({
                'rank': 0,
                'model': 'Stacked Ensemble',
                'primary_metric': stack_score,
                'metrics': stack_metrics,
            })
            trained_models['Stacked Ensemble'] = stack
    except Exception as e:
        logger.warning(f"Stacking ensemble failed: {e}")
    
    # Build weighted ensemble from top models
    try:
        top_model_names = [e['model'] for e in leaderboard[:3] if e['primary_metric'] > -999 and e['model'] in trained_models]
        if len(top_model_names) >= 2 and HAS_OPTUNA:
            w_ensemble, w_score = _build_weighted_ensemble(
                trained_models, top_model_names, X_test, y_test, problem_type
            )
            if w_ensemble is not None:
                if problem_type == 'classification':
                    w_metrics = {'accuracy': w_score, 'method': 'weighted_average'}
                else:
                    w_metrics = {'r2': w_score, 'method': 'weighted_average'}
                leaderboard.append({
                    'rank': 0, 'model': 'Weighted Ensemble',
                    'primary_metric': w_score, 'metrics': w_metrics,
                })
                trained_models['Weighted Ensemble'] = w_ensemble
    except Exception as e:
        logger.warning(f"Weighted ensemble failed: {e}")
    
    # Model bagging for best model (free accuracy boost)
    try:
        best_name = leaderboard[0]['model'] if leaderboard else None
        if best_name and best_name in trained_models and best_name not in ('Stacked Ensemble', 'Weighted Ensemble'):
            bagged = _bag_best_model(
                trained_models[best_name], X_train, y_train, X_test, y_test,
                problem_type, n_bags=5
            )
            if bagged is not None:
                bag_name = f'Bagged {best_name}'
                bag_pred = bagged['model'].predict(X_test)
                if problem_type == 'classification':
                    bag_score = round(float(accuracy_score(y_test, bag_pred)), 4)
                    bag_metrics = {'accuracy': bag_score, 'method': f'5-bag bootstrap of {best_name}'}
                else:
                    bag_score = round(float(r2_score(y_test, bag_pred)), 4)
                    bag_metrics = {'r2': bag_score, 'method': f'5-bag bootstrap of {best_name}'}
                leaderboard.append({
                    'rank': 0, 'model': bag_name,
                    'primary_metric': bag_score, 'metrics': bag_metrics,
                })
                trained_models[bag_name] = bagged['model']
    except Exception as e:
        logger.warning(f"Model bagging failed: {e}")
    
    # Final re-sort after ensemble addition
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


def _optuna_tune(model_name, problem_type, param_grid, X_train, y_train,
                  X_test, y_test, scoring, n_trials=50):
    """Tune a model using Optuna Bayesian optimization (or RandomizedSearchCV fallback).
    
    Returns:
        tuple: (best_model, best_params_dict, test_score)
    """
    if HAS_OPTUNA:
        return _optuna_tune_inner(model_name, problem_type, param_grid,
                                  X_train, y_train, X_test, y_test, scoring, n_trials)
    else:
        return _random_tune_fallback(model_name, problem_type, param_grid,
                                     X_train, y_train, X_test, y_test, scoring)


def _optuna_tune_inner(model_name, problem_type, param_grid, X_train, y_train,
                       X_test, y_test, scoring, n_trials):
    """Optuna Bayesian search with MedianPruner."""
    best = {'score': -np.inf, 'model': None, 'params': {}}

    def objective(trial):
        params = {}
        for k, values in param_grid.items():
            if all(isinstance(v, int) for v in values):
                params[k] = trial.suggest_int(k, min(values), max(values))
            elif all(isinstance(v, float) for v in values):
                params[k] = trial.suggest_float(k, min(values), max(values))
            else:
                params[k] = trial.suggest_categorical(k, values)

        model = get_models(problem_type).get(model_name)
        if model is None:
            return -np.inf
        model.set_params(**params)

        try:
            cv_scores = cross_val_score(model, X_train, y_train, cv=3,
                                        scoring=scoring, n_jobs=-1)
            score = float(cv_scores.mean())
        except Exception:
            return -np.inf

        if score > best['score']:
            model.fit(X_train, y_train)
            best['score'] = score
            best['model'] = model
            best['params'] = params
        return score

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=3,
            min_early_stopping_rate=0,
        )
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if best['model'] is not None:
        y_pred = best['model'].predict(X_test)
        if problem_type == 'classification':
            test_score = round(float(accuracy_score(y_test, y_pred)), 4)
        else:
            test_score = round(float(r2_score(y_test, y_pred)), 4)
        clean_params = {k: (int(v) if isinstance(v, (np.integer,)) else
                            float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in best['params'].items()}
        return best['model'], clean_params, test_score

    return None, {}, -999


def _random_tune_fallback(model_name, problem_type, param_grid,
                          X_train, y_train, X_test, y_test, scoring):
    """Fallback to RandomizedSearchCV when Optuna is not available."""
    from sklearn.model_selection import RandomizedSearchCV
    model = get_models(problem_type).get(model_name)
    if model is None:
        return None, {}, -999
    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=3,
                                scoring=scoring, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)
    y_pred = search.best_estimator_.predict(X_test)
    if problem_type == 'classification':
        test_score = round(float(accuracy_score(y_test, y_pred)), 4)
    else:
        test_score = round(float(r2_score(y_test, y_pred)), 4)
    clean_params = {k: (int(v) if isinstance(v, (np.integer,)) else
                        float(v) if isinstance(v, (np.floating,)) else v)
                    for k, v in search.best_params_.items()}
    return search.best_estimator_, clean_params, test_score


# ---------------------------------------------------------------------------
# Phase 4: Advanced ensemble and bagging utilities
# ---------------------------------------------------------------------------

class BaggedPredictor:
    """Wrapper that averages predictions from multiple bootstrap model copies."""
    
    def __init__(self, models, problem_type='classification'):
        self.models = models
        self.problem_type = problem_type
    
    def predict(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        if self.problem_type == 'classification':
            from scipy.stats import mode as scipy_mode
            result = scipy_mode(preds, axis=0, keepdims=False)
            return result.mode.flatten()
        return preds.mean(axis=0)
    
    def predict_proba(self, X):
        probas = []
        for m in self.models:
            if hasattr(m, 'predict_proba'):
                probas.append(m.predict_proba(X))
        if probas:
            return np.mean(probas, axis=0)
        return None
    
    @property
    def feature_importances_(self):
        importances = [m.feature_importances_ for m in self.models if hasattr(m, 'feature_importances_')]
        if importances:
            return np.mean(importances, axis=0)
        return None


class WeightedEnsemble:
    """Ensemble that blends predictions with optimized weights."""
    
    def __init__(self, models, weights, problem_type='classification'):
        self.models = models
        self.weights = weights
        self.problem_type = problem_type
    
    def predict(self, X):
        if self.problem_type == 'classification':
            # Use weighted probability voting
            proba = self.predict_proba(X)
            if proba is not None:
                return proba.argmax(axis=1)
            # Fallback: weighted hard vote
            preds = np.array([m.predict(X) for m in self.models])
            from scipy.stats import mode as scipy_mode
            result = scipy_mode(preds, axis=0, keepdims=False)
            return result.mode.flatten()
        else:
            weighted_sum = sum(w * m.predict(X) for w, m in zip(self.weights, self.models))
            return weighted_sum
    
    def predict_proba(self, X):
        probas = []
        for m in self.models:
            if hasattr(m, 'predict_proba'):
                probas.append(m.predict_proba(X))
        if probas and len(probas) == len(self.models):
            return sum(w * p for w, p in zip(self.weights, probas))
        return None


def _bag_best_model(model, X_train, y_train, X_test, y_test, problem_type, n_bags=5):
    """Train multiple copies of a model on bootstrap samples and bag them.
    
    Returns:
        dict with 'model' (BaggedPredictor) or None if bagging fails.
    """
    from sklearn.base import clone
    
    bagged_models = []
    n = len(y_train)
    
    for i in range(n_bags):
        try:
            m = clone(model)
            rng = np.random.RandomState(i + 42)
            idx = rng.choice(n, n, replace=True)
            if isinstance(X_train, pd.DataFrame):
                X_boot = X_train.iloc[idx]
            else:
                X_boot = X_train[idx]
            y_boot = y_train.iloc[idx] if hasattr(y_train, 'iloc') else y_train[idx]
            m.fit(X_boot, y_boot)
            bagged_models.append(m)
        except Exception:
            continue
    
    if len(bagged_models) < 2:
        return None
    
    return {'model': BaggedPredictor(bagged_models, problem_type)}


def _build_weighted_ensemble(trained_models, top_names, X_test, y_test, problem_type):
    """Find optimal blending weights for top models using Optuna.
    
    Returns:
        tuple: (WeightedEnsemble, score) or (None, 0)
    """
    if not HAS_OPTUNA:
        return None, 0
    
    models_list = [trained_models[n] for n in top_names]
    
    # Pre-compute predictions to avoid redundant work
    model_preds = [m.predict(X_test) for m in models_list]
    model_probas = []
    for m in models_list:
        if hasattr(m, 'predict_proba'):
            try:
                model_probas.append(m.predict_proba(X_test))
            except Exception:
                model_probas.append(None)
        else:
            model_probas.append(None)
    
    def objective(trial):
        weights = [trial.suggest_float(f'w_{i}', 0.01, 1.0) for i in range(len(top_names))]
        total = sum(weights)
        weights = [w / total for w in weights]
        
        if problem_type == 'classification' and all(p is not None for p in model_probas):
            blended = sum(w * p for w, p in zip(weights, model_probas))
            y_pred = blended.argmax(axis=1)
            return float(accuracy_score(y_test, y_pred))
        elif problem_type != 'classification':
            y_pred = sum(w * p for w, p in zip(weights, model_preds))
            return float(r2_score(y_test, y_pred))
        else:
            return -np.inf
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    
    best_weights = []
    for i in range(len(top_names)):
        best_weights.append(study.best_params[f'w_{i}'])
    total = sum(best_weights)
    best_weights = [w / total for w in best_weights]
    
    ensemble = WeightedEnsemble(models_list, best_weights, problem_type)
    best_score = round(float(study.best_value), 4)
    
    return ensemble, best_score


def multi_seed_evaluate(model, params, X_train, y_train, X_test, y_test,
                        problem_type='classification', seeds=None):
    """Evaluate model with multiple random seeds for robust scoring.
    
    Returns:
        dict with 'mean_score', 'std_score', 'scores', 'seeds'
    """
    from sklearn.base import clone
    
    if seeds is None:
        seeds = [42, 123, 456, 789, 101]
    
    scores = []
    for seed in seeds:
        try:
            m = clone(model)
            if hasattr(m, 'random_state'):
                m.set_params(random_state=seed)
            m.fit(X_train, y_train)
            y_pred = m.predict(X_test)
            if problem_type == 'classification':
                score = float(accuracy_score(y_test, y_pred))
            else:
                score = float(r2_score(y_test, y_pred))
            scores.append(score)
        except Exception:
            continue
    
    if not scores:
        return {'mean_score': 0, 'std_score': 0, 'scores': [], 'seeds': seeds}
    
    return {
        'mean_score': round(float(np.mean(scores)), 4),
        'std_score': round(float(np.std(scores)), 4),
        'scores': [round(s, 4) for s in scores],
        'seeds': seeds[:len(scores)],
    }
