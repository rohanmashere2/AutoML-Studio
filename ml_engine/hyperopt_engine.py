"""
AutoML Studio — Hyperparameter Optimization Engine
Supports Grid Search, Random Search, and Bayesian Optimization (Optuna).
Features: dynamic search spaces, early stopping with pruning, optimization history,
and parameter importance analysis.
"""

import time
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    ExtraTreesClassifier, ExtraTreesRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

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


# ── Dynamic Search Space ─────────────────────────────────────

def _build_dynamic_search_space(model_name, X, y):
    """Build search space that adapts to dataset size, features, and complexity."""
    n_samples, n_features = X.shape
    is_large = n_samples > 10000
    is_medium = n_samples > 1000
    is_high_dim = n_features > 50

    # Adaptive ranges based on dataset size
    if n_samples < 1000:
        # Small dataset → wider exploration
        n_est_range = (20, 1000)
        depth_range = (2, 50)
        lr_range = (0.001, 0.5)
    elif n_samples < 10000:
        # Medium dataset
        n_est_range = (50, 500)
        depth_range = (3, 20)
        lr_range = (0.005, 0.3)
    else:
        # Large dataset → tighter, faster
        n_est_range = (100, 300)
        depth_range = (5, 15)
        lr_range = (0.01, 0.2)

    # Feature count affects sampling
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


def _get_param_grids(problem_type):
    """Return default parameter grids per model (for grid/random search)."""
    is_clf = problem_type == 'classification'
    grids = {}

    grids['RandomForest'] = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    grids['GradientBoosting'] = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 1.0],
    }

    if HAS_XGB:
        grids['XGBoost'] = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
        }

    if HAS_LGB:
        grids['LightGBM'] = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [-1, 5, 10],
            'num_leaves': [15, 31, 63],
        }

    grids['KNeighbors'] = {
        'n_neighbors': [3, 5, 7, 11, 15],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan'],
    }

    grids['DecisionTree'] = {
        'max_depth': [3, 5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

    return grids


def _get_model_instance(model_name, problem_type):
    """Get a fresh model instance by name."""
    is_clf = problem_type == 'classification'
    mapping = {
        'RandomForest': (RandomForestClassifier, RandomForestRegressor),
        'GradientBoosting': (GradientBoostingClassifier, GradientBoostingRegressor),
        'KNeighbors': (KNeighborsClassifier, KNeighborsRegressor),
        'DecisionTree': (DecisionTreeClassifier, DecisionTreeRegressor),
        'AdaBoost': (AdaBoostClassifier, AdaBoostRegressor),
        'ExtraTrees': (ExtraTreesClassifier, ExtraTreesRegressor),
    }

    if HAS_XGB:
        mapping['XGBoost'] = (XGBClassifier, XGBRegressor)
    if HAS_LGB:
        mapping['LightGBM'] = (LGBMClassifier, LGBMRegressor)

    pair = mapping.get(model_name)
    if not pair:
        return None
    cls = pair[0] if is_clf else pair[1]
    # Build kwargs safely — check what params the model actually accepts
    default_params = cls().get_params()
    kwargs = {}
    if 'random_state' in default_params:
        kwargs['random_state'] = 42
    if 'n_jobs' in default_params:
        kwargs['n_jobs'] = -1
    if model_name in ('XGBoost',):
        kwargs['verbosity'] = 0
        kwargs['use_label_encoder'] = False
    if model_name in ('LightGBM',):
        kwargs['verbose'] = -1
    return cls(**kwargs)


# ── Grid Search ──────────────────────────────────────────────

def grid_search(model, param_grid, X, y, cv=5, scoring=None):
    """Exhaustive grid search."""
    start = time.time()
    gs = GridSearchCV(
        model, param_grid, cv=cv, scoring=scoring,
        n_jobs=-1, refit=True, return_train_score=True
    )
    gs.fit(X, y)
    elapsed = time.time() - start

    return {
        'method': 'grid_search',
        'best_params': gs.best_params_,
        'best_score': round(gs.best_score_, 4),
        'best_estimator': gs.best_estimator_,
        'n_candidates': len(gs.cv_results_['params']),
        'time_seconds': round(elapsed, 2),
        'all_results': [
            {
                'params': gs.cv_results_['params'][i],
                'mean_test_score': round(gs.cv_results_['mean_test_score'][i], 4),
                'std_test_score': round(gs.cv_results_['std_test_score'][i], 4),
                'mean_train_score': round(gs.cv_results_['mean_train_score'][i], 4),
            }
            for i in range(min(20, len(gs.cv_results_['params'])))
        ]
    }


# ── Random Search ────────────────────────────────────────────

def random_search(model, param_distributions, X, y, n_iter=50, cv=5, scoring=None):
    """Randomized search with budget control."""
    start = time.time()
    rs = RandomizedSearchCV(
        model, param_distributions, n_iter=n_iter, cv=cv, scoring=scoring,
        n_jobs=-1, refit=True, return_train_score=True, random_state=42
    )
    rs.fit(X, y)
    elapsed = time.time() - start

    return {
        'method': 'random_search',
        'best_params': rs.best_params_,
        'best_score': round(rs.best_score_, 4),
        'best_estimator': rs.best_estimator_,
        'n_candidates': n_iter,
        'time_seconds': round(elapsed, 2),
        'all_results': [
            {
                'params': rs.cv_results_['params'][i],
                'mean_test_score': round(rs.cv_results_['mean_test_score'][i], 4),
                'std_test_score': round(rs.cv_results_['std_test_score'][i], 4),
            }
            for i in range(min(20, len(rs.cv_results_['params'])))
        ]
    }


# ── Bayesian Optimization (Optuna) with Pruning ──────────────

def bayesian_search(model_name, problem_type, X, y, n_trials=50, cv=5, scoring=None):
    """Optuna-based Bayesian optimization with early stopping (MedianPruner)."""
    if not HAS_OPTUNA:
        return {'error': 'optuna not installed. Run: pip install optuna'}

    start = time.time()
    is_clf = problem_type == 'classification'
    default_scoring = 'accuracy' if is_clf else 'r2'
    scoring = scoring or default_scoring

    best_result = {'score': -np.inf, 'params': {}, 'estimator': None}
    trial_history = []
    best_over_time = []
    pruned_count = [0]

    # Build dynamic search space
    dynamic_spaces = _build_dynamic_search_space(model_name, X, y)

    def objective(trial):
        params = _suggest_params_dynamic(trial, model_name, dynamic_spaces)
        model = _get_model_instance(model_name, problem_type)
        if model is None:
            return -np.inf
        model.set_params(**params)

        try:
            # Use cross_val_score but report intermediate fold scores for pruning
            from sklearn.model_selection import StratifiedKFold, KFold
            if is_clf:
                kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            else:
                kf = KFold(n_splits=cv, shuffle=True, random_state=42)

            fold_scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y)):
                X_fold_train, X_fold_val = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx], X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx], y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

                model_clone = _get_model_instance(model_name, problem_type)
                model_clone.set_params(**params)
                model_clone.fit(X_fold_train, y_fold_train)

                from sklearn.metrics import get_scorer
                scorer = get_scorer(scoring)
                fold_score = scorer(model_clone, X_fold_val, y_fold_val)
                fold_scores.append(fold_score)

                # Report intermediate value for pruning
                trial.report(np.mean(fold_scores), fold_idx)

                # Check if trial should be pruned
                if trial.should_prune():
                    pruned_count[0] += 1
                    raise optuna.exceptions.TrialPruned()

            score = np.mean(fold_scores)
        except optuna.exceptions.TrialPruned:
            raise
        except Exception:
            return -np.inf

        trial_history.append({
            'trial': trial.number,
            'params': params,
            'score': round(score, 4),
        })

        if score > best_result['score']:
            best_result['score'] = score
            best_result['params'] = params
            model.fit(X, y)
            best_result['estimator'] = model

        best_over_time.append(round(best_result['score'], 4))

        return score

    # Create study with MedianPruner for early stopping
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(direction='maximize', pruner=pruner)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    elapsed = time.time() - start

    # Parameter importance (if enough trials completed)
    param_importance = {}
    try:
        if len(study.trials) > 10:
            importances = optuna.importance.get_param_importances(study)
            param_importance = {k: round(v, 4) for k, v in importances.items()}
    except Exception:
        pass

    return {
        'method': 'bayesian_search',
        'best_params': best_result['params'],
        'best_score': round(best_result['score'], 4),
        'best_estimator': best_result['estimator'],
        'n_trials': n_trials,
        'time_seconds': round(elapsed, 2),
        'convergence': [{'trial': t['trial'], 'score': t['score']} for t in trial_history],
        'all_results': trial_history[:20],
        'optimization_history': {
            'best_score_over_time': best_over_time,
            'param_importance': param_importance,
            'pruned_count': pruned_count[0],
            'completed_trials': len(trial_history),
            'total_time': round(elapsed, 2),
        },
    }


def _suggest_params_dynamic(trial, model_name, dynamic_spaces):
    """Suggest hyperparameters using dynamic search spaces."""
    space = dynamic_spaces.get(model_name, {})

    if model_name == 'RandomForest':
        return {
            'n_estimators': trial.suggest_int('n_estimators', *space.get('n_estimators', (50, 500))),
            'max_depth': trial.suggest_int('max_depth', *space.get('max_depth', (3, 30))),
            'min_samples_split': trial.suggest_int('min_samples_split', *space.get('min_samples_split', (2, 20))),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *space.get('min_samples_leaf', (1, 10))),
        }
    elif model_name == 'GradientBoosting':
        lr = space.get('learning_rate', (0.001, 0.3))
        return {
            'n_estimators': trial.suggest_int('n_estimators', *space.get('n_estimators', (50, 500))),
            'learning_rate': trial.suggest_float('learning_rate', lr[0], lr[1], log=True),
            'max_depth': trial.suggest_int('max_depth', *space.get('max_depth', (2, 10))),
            'subsample': trial.suggest_float('subsample', *space.get('subsample', (0.5, 1.0))),
        }
    elif model_name == 'XGBoost' and HAS_XGB:
        lr = space.get('learning_rate', (0.001, 0.3))
        return {
            'n_estimators': trial.suggest_int('n_estimators', *space.get('n_estimators', (50, 500))),
            'learning_rate': trial.suggest_float('learning_rate', lr[0], lr[1], log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 12),
            'subsample': trial.suggest_float('subsample', *space.get('subsample', (0.5, 1.0))),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *space.get('colsample_bytree', (0.5, 1.0))),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        }
    elif model_name == 'LightGBM' and HAS_LGB:
        lr = space.get('learning_rate', (0.001, 0.3))
        return {
            'n_estimators': trial.suggest_int('n_estimators', *space.get('n_estimators', (50, 500))),
            'learning_rate': trial.suggest_float('learning_rate', lr[0], lr[1], log=True),
            'max_depth': trial.suggest_int('max_depth', *space.get('max_depth', (-1, 15))),
            'num_leaves': trial.suggest_int('num_leaves', *space.get('num_leaves', (15, 127))),
            'min_child_samples': trial.suggest_int('min_child_samples', *space.get('min_child_samples', (5, 100))),
        }
    elif model_name == 'KNeighbors':
        return {
            'n_neighbors': trial.suggest_int('n_neighbors', *space.get('n_neighbors', (1, 30))),
            'weights': trial.suggest_categorical('weights', space.get('weights', ['uniform', 'distance'])),
            'metric': trial.suggest_categorical('metric', space.get('metric', ['euclidean', 'manhattan', 'minkowski'])),
        }
    elif model_name == 'DecisionTree':
        return {
            'max_depth': trial.suggest_int('max_depth', *space.get('max_depth', (2, 30))),
            'min_samples_split': trial.suggest_int('min_samples_split', *space.get('min_samples_split', (2, 20))),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', *space.get('min_samples_leaf', (1, 10))),
        }
    else:
        return {}


# ── Auto Optimize ────────────────────────────────────────────

def auto_optimize(trained_models, X, y, problem_type, method='auto', budget=50,
                  progress_callback=None):
    """
    Auto-optimize hyperparameters for all trained models.

    Args:
        trained_models: dict of {name: model_instance}
        X: feature matrix
        y: target vector
        problem_type: 'classification' or 'regression'
        method: 'grid', 'random', 'bayesian', or 'auto'
        budget: number of iterations/trials
        progress_callback: fn(message, progress_pct)

    Returns:
        dict with optimization results per model
    """
    n_samples, n_features = X.shape
    is_clf = problem_type == 'classification'
    scoring = 'accuracy' if is_clf else 'r2'

    # Auto-select method based on dataset size
    if method == 'auto':
        total_combos = n_samples * n_features
        if total_combos < 5000:
            method = 'grid'
        elif HAS_OPTUNA:
            method = 'bayesian'
        else:
            method = 'random'

    param_grids = _get_param_grids(problem_type)
    results = {}
    total = len(trained_models)

    for idx, (name, model) in enumerate(trained_models.items()):
        if progress_callback:
            pct = int((idx / total) * 100)
            progress_callback(f'Optimizing {name} ({method})...', pct)

        # Skip models we don't have grids for
        canonical_name = _canonical_model_name(name)
        if canonical_name not in param_grids:
            results[name] = {
                'skipped': True,
                'reason': f'No param grid for {name}',
            }
            continue

        grid = param_grids[canonical_name]
        base_model = _get_model_instance(canonical_name, problem_type)
        if base_model is None:
            results[name] = {'skipped': True, 'reason': 'Model not available'}
            continue

        try:
            if method == 'grid':
                # Limit grid size
                limited_grid = {k: v[:3] for k, v in grid.items()}
                res = grid_search(base_model, limited_grid, X, y, cv=3, scoring=scoring)
            elif method == 'random':
                from scipy.stats import randint, uniform
                res = random_search(base_model, grid, X, y, n_iter=min(budget, 30), cv=3, scoring=scoring)
            elif method == 'bayesian':
                res = bayesian_search(canonical_name, problem_type, X, y, n_trials=min(budget, 30), cv=3, scoring=scoring)
            else:
                res = random_search(base_model, grid, X, y, n_iter=min(budget, 30), cv=3, scoring=scoring)

            # Get original score for comparison
            try:
                orig_scores = cross_val_score(model, X, y, cv=3, scoring=scoring, n_jobs=-1)
                orig_score = round(orig_scores.mean(), 4)
            except Exception:
                orig_score = None

            results[name] = {
                'method': res['method'],
                'original_score': orig_score,
                'optimized_score': res['best_score'],
                'improvement': round(res['best_score'] - (orig_score or 0), 4) if orig_score else None,
                'best_params': _serialize_params(res['best_params']),
                'n_candidates': res.get('n_candidates', res.get('n_trials', 0)),
                'time_seconds': res['time_seconds'],
                'convergence': res.get('convergence', []),
                'optimization_history': res.get('optimization_history', {}),
                'best_estimator': res.get('best_estimator'),
            }
        except Exception as e:
            results[name] = {'skipped': True, 'reason': str(e)}

    if progress_callback:
        progress_callback('Optimization complete!', 100)

    # Summary
    optimized = {k: v for k, v in results.items() if not v.get('skipped')}
    best_name = max(optimized, key=lambda k: optimized[k].get('optimized_score', 0)) if optimized else None

    return {
        'method': method,
        'total_models': total,
        'optimized_count': len(optimized),
        'skipped_count': len(results) - len(optimized),
        'best_model': best_name,
        'best_score': optimized[best_name]['optimized_score'] if best_name else None,
        'models': {k: {kk: vv for kk, vv in v.items() if kk != 'best_estimator'} for k, v in results.items()},
        'best_estimators': {k: v.get('best_estimator') for k, v in results.items() if v.get('best_estimator')},
    }


def _canonical_model_name(name):
    """Map trained model names to canonical names for param grids."""
    mapping = {
        'Random Forest': 'RandomForest',
        'Gradient Boosting': 'GradientBoosting',
        'XGBoost': 'XGBoost',
        'LightGBM': 'LightGBM',
        'K-Nearest Neighbors': 'KNeighbors',
        'KNN': 'KNeighbors',
        'Decision Tree': 'DecisionTree',
        'AdaBoost': 'AdaBoost',
        'Extra Trees': 'ExtraTrees',
    }
    return mapping.get(name, name)


def _serialize_params(params):
    """Ensure params are JSON-serializable."""
    out = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
