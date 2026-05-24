"""
AutoML Studio — Centralised Hyperparameter Search Spaces
Single source of truth consumed by trainer.py (grid/random search) and
hyperopt_engine.py (Bayesian / Optuna search).
"""

import numpy as np

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ── Discrete grids for GridSearchCV / RandomizedSearchCV ─────

CLASSIFICATION_PARAMS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
    },
    'Logistic Regression': {
        'C': [0.01, 0.1, 1, 10],
        'max_iter': [500, 1000],
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    },
    'Decision Tree': {
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
}

REGRESSION_PARAMS = {
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    },
    'Ridge': {
        'alpha': [0.01, 0.1, 1, 10, 100],
    },
    'Lasso': {
        'alpha': [0.001, 0.01, 0.1, 1, 10],
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    },
    'Decision Tree': {
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
}

# Conditionally add boosting libraries
if HAS_XGB:
    _xgb_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    }
    CLASSIFICATION_PARAMS['XGBoost'] = _xgb_grid
    REGRESSION_PARAMS['XGBoost'] = _xgb_grid.copy()

if HAS_LGB:
    _lgb_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [-1, 3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'num_leaves': [15, 31, 63],
    }
    CLASSIFICATION_PARAMS['LightGBM'] = _lgb_grid
    REGRESSION_PARAMS['LightGBM'] = _lgb_grid.copy()

if HAS_CATBOOST:
    _cat_grid = {
        'iterations': [100, 200, 500],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7],
    }
    CLASSIFICATION_PARAMS['CatBoost'] = _cat_grid
    REGRESSION_PARAMS['CatBoost'] = _cat_grid.copy()


# ── Static grids keyed by canonical name (for hyperopt_engine.py) ─

_CANONICAL_GRIDS = {
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
    },
    'KNeighbors': {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    },
    'DecisionTree': {
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    },
}

if HAS_XGB:
    _CANONICAL_GRIDS['XGBoost'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }

if HAS_LGB:
    _CANONICAL_GRIDS['LightGBM'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [-1, 5, 10],
        'num_leaves': [15, 31, 63],
    }

if HAS_CATBOOST:
    _CANONICAL_GRIDS['CatBoost'] = {
        'iterations': [100, 200, 500],
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'l2_leaf_reg': [1, 3, 5, 7],
    }


def get_param_grids(problem_type):
    """Return discrete param grids keyed by display name (for trainer.py)."""
    return CLASSIFICATION_PARAMS if problem_type == 'classification' else REGRESSION_PARAMS


def get_canonical_param_grids(problem_type=None):
    """Return discrete param grids keyed by canonical name (for hyperopt_engine.py)."""
    return dict(_CANONICAL_GRIDS)


def build_dynamic_search_space(model_name, X, y):
    """Build dataset-adaptive continuous search space for Optuna Bayesian search.
    
    Args:
        model_name: Canonical model name (e.g. 'RandomForest', 'XGBoost')
        X: Feature matrix
        y: Target vector
    
    Returns:
        dict mapping model names to param ranges (tuples for continuous, lists for categorical)
    """
    n_samples, n_features = X.shape
    is_high_dim = n_features > 50

    if n_samples < 1000:
        n_est_range = (20, 1000)
        depth_range = (2, 50)
        lr_range = (0.001, 0.5)
    elif n_samples < 10000:
        n_est_range = (50, 500)
        depth_range = (3, 20)
        lr_range = (0.005, 0.3)
    else:
        n_est_range = (100, 300)
        depth_range = (5, 15)
        lr_range = (0.01, 0.2)

    colsample_range = (0.3, 0.8) if is_high_dim else (0.6, 1.0)
    min_samples_range = (2, 20) if n_samples > 500 else (2, 10)

    spaces = {}

    spaces['RandomForest'] = {
        'n_estimators': n_est_range,
        'max_depth': depth_range,
        'min_samples_split': min_samples_range,
        'min_samples_leaf': (1, max(2, n_samples // 100)),
    }

    spaces['GradientBoosting'] = {
        'n_estimators': n_est_range,
        'learning_rate': lr_range,
        'max_depth': (2, min(depth_range[1], 10)),
        'subsample': (0.5, 1.0),
    }

    if HAS_XGB:
        spaces['XGBoost'] = {
            'n_estimators': n_est_range,
            'learning_rate': lr_range,
            'max_depth': (2, 12),
            'subsample': (0.5, 1.0),
            'colsample_bytree': colsample_range,
            'reg_alpha': (1e-8, 10.0),
            'reg_lambda': (1e-8, 10.0),
        }

    if HAS_LGB:
        spaces['LightGBM'] = {
            'n_estimators': n_est_range,
            'learning_rate': lr_range,
            'max_depth': (-1, 15),
            'num_leaves': (15, 127),
            'min_child_samples': (5, min(100, n_samples // 10)),
        }

    spaces['KNeighbors'] = {
        'n_neighbors': (1, min(30, n_samples // 5)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
    }

    spaces['DecisionTree'] = {
        'max_depth': depth_range,
        'min_samples_split': min_samples_range,
        'min_samples_leaf': (1, max(2, n_samples // 100)),
    }

    return spaces
