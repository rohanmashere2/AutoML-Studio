"""
Stacking Ensemble Builder — Builds a meta-learner on top of the best individual models.
Combines top-N leaderboard models into a StackingClassifier/Regressor.
"""

import numpy as np
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, r2_score


def build_stacking_ensemble(trained_models, leaderboard, X_train, y_train,
                            X_test, y_test, problem_type, top_n=3):
    """
    Build a stacking ensemble from the top-N models on the leaderboard.

    Args:
        trained_models: dict {name: fitted_model}
        leaderboard: list of leaderboard dicts (sorted by rank)
        X_train, y_train, X_test, y_test: train/test splits
        problem_type: 'classification' or 'regression'
        top_n: number of base models to stack

    Returns:
        dict with ensemble model, score, comparison, and metadata
    """
    # Pick top-N models that actually trained successfully
    valid = [e for e in leaderboard
             if e.get('primary_metric', -999) > -999 and e['model'] in trained_models]
    if len(valid) < 2:
        return {'error': 'Need at least 2 successfully trained models to build an ensemble.'}

    selected = valid[:min(top_n, len(valid))]
    estimators = [(entry['model'], trained_models[entry['model']]) for entry in selected]

    try:
        if problem_type == 'classification':
            meta = LogisticRegression(max_iter=1000, random_state=42)
            stack = StackingClassifier(
                estimators=estimators, final_estimator=meta,
                cv=3, n_jobs=-1, passthrough=False
            )
        else:
            meta = RidgeCV()
            stack = StackingRegressor(
                estimators=estimators, final_estimator=meta,
                cv=3, n_jobs=-1, passthrough=False
            )

        stack.fit(X_train, y_train)
        y_pred = stack.predict(X_test)

        if problem_type == 'classification':
            ensemble_score = round(float(accuracy_score(y_test, y_pred)), 4)
            scoring = 'accuracy'
        else:
            ensemble_score = round(float(r2_score(y_test, y_pred)), 4)
            scoring = 'r2'

        # Cross-validate the ensemble
        cv_scores = cross_val_score(stack, X_train, y_train, cv=3,
                                     scoring=scoring, n_jobs=-1)

        # Compare to best single model
        best_single = selected[0]
        improvement = round(ensemble_score - best_single['primary_metric'], 4)

        return {
            'ensemble_model': stack,
            'ensemble_score': ensemble_score,
            'cv_mean': round(float(cv_scores.mean()), 4),
            'cv_std': round(float(cv_scores.std()), 4),
            'base_models': [{'model': e['model'], 'score': e['primary_metric']} for e in selected],
            'meta_learner': 'LogisticRegression' if problem_type == 'classification' else 'RidgeCV',
            'best_single_model': best_single['model'],
            'best_single_score': best_single['primary_metric'],
            'improvement': improvement,
            'improvement_pct': f"{improvement / max(abs(best_single['primary_metric']), 0.001) * 100:.1f}%",
            'beats_best': ensemble_score > best_single['primary_metric'],
        }
    except Exception as e:
        return {'error': f'Ensemble building failed: {str(e)}'}
