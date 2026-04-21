"""
AutoML Problem Solver - Time Series Trainer
Trains multiple forecasting models with walk-forward validation.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def train_timeseries_models(df, target_col, datetime_col=None, forecast_horizon=None,
                             output_dir=None, progress_callback=None):
    """
    Train multiple forecasting models using walk-forward validation.
    
    Args:
        df: DataFrame with features (lag/rolling already created)
        target_col: Name of the target column to forecast
        datetime_col: Name of datetime column (if exists)
        forecast_horizon: Number of periods to forecast ahead
        output_dir: Directory to save models
        progress_callback: Progress reporting function
    
    Returns:
        dict: Training results with leaderboard, forecasts, and metrics
    """
    if forecast_horizon is None:
        forecast_horizon = min(max(len(df) // 10, 5), 30)
    
    # Prepare features
    feature_cols = [c for c in df.columns if c != target_col and c != datetime_col]
    numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_features:
        return {'error': 'No numeric features for time series training'}
    
    X = df[numeric_features].values
    y = df[target_col].values
    
    # Walk-forward split (time-aware, no shuffling)
    test_size = min(forecast_horizon, len(df) // 5)
    train_size = len(df) - test_size
    
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Models
    models = _get_ts_models()
    
    leaderboard = []
    trained_models = {}
    forecasts = {}
    total = len(models)
    
    for idx, (name, model) in enumerate(models.items()):
        if progress_callback:
            progress_callback(f'Training {name}...', int((idx / total) * 80))
        
        try:
            model.fit(X_train, y_train)
            trained_models[name] = model
            
            y_pred = model.predict(X_test)
            forecasts[name] = y_pred.tolist()
            
            # Walk-forward CV metrics
            cv_metrics = _walk_forward_cv(model, X, y, n_splits=min(5, train_size // 20))
            
            metrics = {
                'mae': round(float(mean_absolute_error(y_test, y_pred)), 4),
                'mse': round(float(mean_squared_error(y_test, y_pred)), 4),
                'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
                'r2': round(float(r2_score(y_test, y_pred)), 4),
                'mape': round(float(_mape(y_test, y_pred)), 4),
                'cv_mae': cv_metrics.get('mae', None),
                'cv_rmse': cv_metrics.get('rmse', None),
            }
            
            leaderboard.append({
                'rank': 0,
                'model': name,
                'primary_metric': metrics['mae'],
                'metrics': metrics,
            })
        except Exception as e:
            leaderboard.append({
                'rank': 0,
                'model': name,
                'primary_metric': 999999,
                'metrics': {'error': str(e)},
            })
    
    # Sort by MAE (lower is better for time series)
    leaderboard.sort(key=lambda x: x['primary_metric'])
    for i, entry in enumerate(leaderboard):
        entry['rank'] = i + 1
    
    best_model_name = leaderboard[0]['model'] if leaderboard else None
    
    # Feature importance from best model
    feature_importance = []
    if best_model_name in trained_models:
        best = trained_models[best_model_name]
        if hasattr(best, 'feature_importances_'):
            importances = best.feature_importances_
            total_imp = importances.sum()
            if total_imp > 0:
                importances = importances / total_imp
            for fname, imp in zip(numeric_features, importances):
                feature_importance.append({
                    'feature': fname,
                    'importance': round(float(imp), 4),
                })
            feature_importance.sort(key=lambda x: x['importance'], reverse=True)
            feature_importance = feature_importance[:20]
    
    # Generate confidence intervals for best model's forecast
    confidence_intervals = _bootstrap_confidence(
        trained_models.get(best_model_name), X_train, y_train, X_test
    ) if best_model_name in trained_models else None
    
    # Forecast visualization data
    forecast_viz = {
        'actual_train': y_train[-min(100, len(y_train)):].tolist(),
        'actual_test': y_test.tolist(),
        'best_forecast': forecasts.get(best_model_name, []),
        'n_train_shown': min(100, len(y_train)),
        'confidence_intervals': confidence_intervals,
    }
    
    # Save best model
    best_model_path = None
    if output_dir and best_model_name in trained_models:
        import os, joblib
        os.makedirs(output_dir, exist_ok=True)
        best_model_path = os.path.join(output_dir, 'best_model.pkl')
        joblib.dump(trained_models[best_model_name], best_model_path)
    
    if progress_callback:
        progress_callback('Forecasting complete!', 100)
    
    return {
        'leaderboard': leaderboard,
        'best_model': best_model_name,
        'best_score': leaderboard[0]['primary_metric'],
        'primary_metric_name': 'mae',
        'problem_type': 'forecasting',
        'feature_importance': feature_importance,
        'forecast_viz': forecast_viz,
        'forecasts': {k: v for k, v in list(forecasts.items())[:5]},
        'forecast_horizon': test_size,
        'best_model_path': best_model_path,
        'training_context': {
            'n_features': len(numeric_features),
            'feature_names': numeric_features,
            'X_train_shape': (X_train.shape[0], X_train.shape[1]),
            'X_test_shape': (X_test.shape[0], X_test.shape[1]),
            'trained_models': trained_models,
            'leaderboard': leaderboard,
            'problem_type': 'forecasting',
        },
    }


def _get_ts_models():
    """Get time series models."""
    models = {
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1, max_iter=5000),
        'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42
        ),
    }
    
    if HAS_XGB:
        models['XGBoost'] = XGBRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbosity=0
        )
    if HAS_LGBM:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1
        )
    
    return models


def _walk_forward_cv(model, X, y, n_splits=5):
    """Walk-forward cross-validation."""
    try:
        n = len(X)
        min_train = max(n // 3, 20)
        fold_size = (n - min_train) // n_splits
        
        if fold_size < 5:
            return {}
        
        mae_scores = []
        rmse_scores = []
        
        for i in range(n_splits):
            train_end = min_train + i * fold_size
            test_end = min(train_end + fold_size, n)
            
            if test_end <= train_end:
                break
            
            X_tr, X_te = X[:train_end], X[train_end:test_end]
            y_tr, y_te = y[:train_end], y[train_end:test_end]
            
            from sklearn.base import clone
            m = clone(model)
            m.fit(X_tr, y_tr)
            y_pred = m.predict(X_te)
            
            mae_scores.append(mean_absolute_error(y_te, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        
        return {
            'mae': round(float(np.mean(mae_scores)), 4) if mae_scores else None,
            'rmse': round(float(np.mean(rmse_scores)), 4) if rmse_scores else None,
        }
    except Exception:
        return {}


def _bootstrap_confidence(model, X_train, y_train, X_test, n_iterations=50, confidence=0.95):
    """Generate confidence intervals via bootstrap."""
    if model is None:
        return None
    
    try:
        from sklearn.base import clone
        predictions = []
        n = len(X_train)
        
        for _ in range(n_iterations):
            idx = np.random.randint(0, n, size=n)
            m = clone(model)
            m.fit(X_train[idx], y_train[idx])
            predictions.append(m.predict(X_test))
        
        predictions = np.array(predictions)
        lower = np.percentile(predictions, (1 - confidence) / 2 * 100, axis=0)
        upper = np.percentile(predictions, (1 + confidence) / 2 * 100, axis=0)
        
        return {
            'lower': lower.tolist(),
            'upper': upper.tolist(),
            'confidence': confidence,
        }
    except Exception:
        return None


def _mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
