"""
AutoML Studio — Model Complexity vs Performance Analyzer (Feature #21)
Shows trade-off between model complexity and performance to help
pick the simplest "good enough" model.
"""

import numpy as np
import time


def analyze_complexity(trained_models, X_test, y_test, leaderboard,
                        problem_type='classification'):
    """
    Analyze complexity vs performance for all trained models.

    Returns:
        dict with per-model complexity metrics and value recommendation
    """
    from sklearn.metrics import accuracy_score, r2_score
    is_clf = problem_type == 'classification'
    results = []

    for entry in leaderboard:
        name = entry.get('model', '')
        score = entry.get('primary_metric', 0)
        if score <= -999 or name not in trained_models:
            continue
        model = trained_models[name]

        # Count parameters
        n_params = _count_params(model)

        # Measure inference speed
        try:
            start = time.perf_counter()
            for _ in range(3):
                model.predict(X_test)
            elapsed = (time.perf_counter() - start) / 3
            inference_ms = round(elapsed * 1000, 1)
        except Exception:
            inference_ms = 0

        # Model size estimate (bytes)
        import sys
        try:
            import joblib, io
            buf = io.BytesIO()
            joblib.dump(model, buf)
            size_bytes = buf.tell()
            size_kb = round(size_bytes / 1024, 1)
        except Exception:
            size_kb = 0

        # Complexity rating
        if n_params <= 50:
            complexity = 1
            label = 'Simple'
            icon = '⭐'
        elif n_params <= 500:
            complexity = 2
            label = 'Moderate'
            icon = '⭐⭐'
        elif n_params <= 5000:
            complexity = 3
            label = 'Complex'
            icon = '⭐⭐⭐'
        else:
            complexity = 4
            label = 'Heavy'
            icon = '⭐⭐⭐⭐'

        results.append({
            'model': name,
            'score': round(float(score), 4),
            'n_params': n_params,
            'inference_ms': inference_ms,
            'size_kb': size_kb,
            'complexity_rating': complexity,
            'complexity_label': label,
            'complexity_icon': icon,
        })

    if not results:
        return {'error': 'No valid models to analyze'}

    results.sort(key=lambda x: x['score'], reverse=True)

    # Find best value: highest score-to-complexity ratio
    best = results[0]
    best_value = None
    for r in results:
        if r['score'] >= best['score'] * 0.97 and r['complexity_rating'] < best['complexity_rating']:
            best_value = r
            break

    if best_value:
        value_msg = (
            f'"{best_value["model"]}" achieves {best_value["score"]:.1%} '
            f'({best_value["complexity_label"]}) — only '
            f'{abs(best["score"] - best_value["score"]):.1%} less than the best '
            f'"{best["model"]}" ({best["complexity_label"]}). '
            f'Best value pick for production.'
        )
    else:
        value_msg = f'"{best["model"]}" is both the best performing and recommended.'
        best_value = best

    return {
        'models': results,
        'best_performer': best['model'],
        'best_value': best_value['model'],
        'value_recommendation': value_msg,
        'metric': 'accuracy' if is_clf else 'r2',
    }


def _count_params(model):
    """Estimate number of learnable parameters in a model."""
    n = 0
    # Tree-based: count nodes across all trees
    if hasattr(model, 'estimators_'):
        try:
            trees = model.estimators_
            if hasattr(trees, '__len__'):
                if hasattr(trees[0], 'tree_'):
                    n = sum(t.tree_.node_count for t in trees)
                elif hasattr(trees[0], '__len__'):
                    n = sum(t.tree_.node_count for row in trees for t in row if hasattr(t, 'tree_'))
                else:
                    n = len(trees) * 100
            return n
        except Exception:
            return len(model.estimators_) * 100

    # Linear: count coefficients
    if hasattr(model, 'coef_'):
        coef = model.coef_
        n = np.prod(coef.shape) if hasattr(coef, 'shape') else len(coef)
        if hasattr(model, 'intercept_'):
            n += np.prod(np.array(model.intercept_).shape) if hasattr(model.intercept_, 'shape') else 1
        return int(n)

    # Single tree
    if hasattr(model, 'tree_'):
        return model.tree_.node_count

    # XGBoost/LightGBM
    if hasattr(model, 'get_booster'):
        try:
            return len(model.get_booster().get_dump())
        except Exception:
            pass
    if hasattr(model, 'booster_'):
        try:
            return model.booster_.num_trees()
        except Exception:
            pass

    # KNN: no parameters
    if hasattr(model, 'n_neighbors'):
        return model.n_neighbors

    return 0
