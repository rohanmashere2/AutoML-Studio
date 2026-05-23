"""
AutoML Studio — Prediction Autopsy / Decision Debugger (Feature #5)
For any single prediction, generates a complete human-readable "autopsy report"
showing the step-by-step decision process.
"""

import numpy as np
import pandas as pd


def autopsy_prediction(model, sample, X_train, y_train, feature_names,
                        problem_type='classification', shap_values=None):
    """
    Generate a detailed autopsy for a single prediction.

    Args:
        model: trained model
        sample: single sample (dict, Series, or 1-row DataFrame)
        X_train: training features for neighbour lookup
        y_train: training labels
        feature_names: list of feature names
        problem_type: 'classification' or 'regression'
        shap_values: optional pre-computed SHAP values for this sample

    Returns:
        dict with decision path, similar examples, flip conditions, narrative
    """
    # Prepare sample
    if isinstance(sample, dict):
        sample_df = pd.DataFrame([sample])[feature_names]
    elif isinstance(sample, pd.Series):
        sample_df = pd.DataFrame([sample])[feature_names]
    else:
        sample_df = sample

    sample_arr = sample_df.values.reshape(1, -1)
    is_clf = problem_type == 'classification'

    # 1. Point prediction
    prediction = model.predict(sample_arr)[0]
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(sample_arr)[0]
        confidence = round(float(np.max(proba)), 4)
        class_probs = {str(i): round(float(p), 4) for i, p in enumerate(proba)}
    else:
        class_probs = {}

    # 2. Decision factors (SHAP-based or coefficient-based)
    factors = _extract_factors(model, sample_df, feature_names, shap_values)

    # 3. Similar training examples
    similar = _find_similar(sample_arr, X_train, y_train, feature_names, k=10)

    # 4. Counterfactual: what would flip the prediction?
    flip_conditions = _find_flip_conditions(
        model, sample_df, feature_names, prediction, is_clf, X_train
    )

    # 5. Confidence assessment
    assessment = _assess_confidence(similar, prediction, confidence, is_clf)

    # 6. Build narrative
    narrative = _build_narrative(
        prediction, confidence, factors, similar, flip_conditions,
        assessment, is_clf, feature_names
    )

    return {
        'prediction': _safe(prediction),
        'confidence': confidence,
        'class_probabilities': class_probs,
        'decision_factors': factors[:10],
        'similar_examples': similar[:5],
        'flip_conditions': flip_conditions[:5],
        'confidence_assessment': assessment,
        'narrative': narrative,
        'problem_type': problem_type,
    }


def _extract_factors(model, sample_df, feature_names, shap_values=None):
    """Extract decision factors ranked by importance."""
    factors = []

    # Try SHAP values first
    if shap_values is not None:
        vals = np.array(shap_values).flatten()
        for i, fname in enumerate(feature_names):
            if i >= len(vals):
                break
            factors.append({
                'feature': fname,
                'value': round(float(sample_df.iloc[0][fname]), 4) if fname in sample_df.columns else None,
                'impact': round(float(vals[i]), 4),
                'direction': 'positive' if vals[i] > 0 else 'negative',
            })
        factors.sort(key=lambda x: abs(x['impact']), reverse=True)
        return factors

    # Fallback: use feature_importances_ or coef_
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        for i, fname in enumerate(feature_names):
            if i >= len(importances):
                break
            val = float(sample_df.iloc[0][fname]) if fname in sample_df.columns else 0
            factors.append({
                'feature': fname,
                'value': round(val, 4),
                'impact': round(float(importances[i]), 4),
                'direction': 'contributor',
            })
    elif hasattr(model, 'coef_'):
        coefs = np.array(model.coef_).flatten()
        for i, fname in enumerate(feature_names):
            if i >= len(coefs):
                break
            val = float(sample_df.iloc[0][fname]) if fname in sample_df.columns else 0
            impact = val * coefs[i]
            factors.append({
                'feature': fname,
                'value': round(val, 4),
                'impact': round(float(impact), 4),
                'direction': 'positive' if impact > 0 else 'negative',
            })

    factors.sort(key=lambda x: abs(x['impact']), reverse=True)
    return factors


def _find_similar(sample, X_train, y_train, feature_names, k=10):
    """Find k nearest training examples."""
    from sklearn.neighbors import NearestNeighbors
    X_arr = np.array(X_train)
    y_arr = np.array(y_train)

    try:
        nn = NearestNeighbors(n_neighbors=min(k, len(X_arr)), metric='euclidean')
        nn.fit(X_arr)
        distances, indices = nn.kneighbors(sample)

        similar = []
        for dist, idx in zip(distances[0], indices[0]):
            similar.append({
                'index': int(idx),
                'distance': round(float(dist), 4),
                'true_label': _safe(y_arr[idx]),
            })
        return similar
    except Exception:
        return []


def _find_flip_conditions(model, sample_df, feature_names, current_pred,
                           is_clf, X_train):
    """Find minimal feature changes that flip the prediction."""
    flips = []
    sample = sample_df.iloc[0].copy()

    for fname in feature_names[:15]:
        if fname not in sample_df.columns:
            continue
        original_val = sample[fname]
        if not pd.api.types.is_numeric_dtype(pd.Series([original_val])):
            continue

        # Try +/- perturbations
        col_data = X_train[fname] if hasattr(X_train, 'columns') and fname in X_train.columns else None
        if col_data is None:
            continue

        std = float(col_data.std())
        if std < 1e-8:
            continue

        for delta_factor in [0.5, 1.0, 2.0, 3.0]:
            for direction in [1, -1]:
                new_val = float(original_val) + direction * delta_factor * std
                test_sample = sample.copy()
                test_sample[fname] = new_val
                test_df = pd.DataFrame([test_sample])[feature_names]
                new_pred = model.predict(test_df.values.reshape(1, -1))[0]

                if is_clf and new_pred != current_pred:
                    flips.append({
                        'feature': fname,
                        'original_value': round(float(original_val), 4),
                        'new_value': round(new_val, 4),
                        'change': round(new_val - float(original_val), 4),
                        'new_prediction': _safe(new_pred),
                        'description': f'If {fname} changes from {float(original_val):.2f} to {new_val:.2f} → prediction flips to {new_pred}',
                    })
                    break
                elif not is_clf:
                    change_pct = abs(new_pred - current_pred) / max(abs(current_pred), 1e-8) * 100
                    if change_pct > 10:
                        flips.append({
                            'feature': fname,
                            'original_value': round(float(original_val), 4),
                            'new_value': round(new_val, 4),
                            'change': round(new_val - float(original_val), 4),
                            'new_prediction': round(float(new_pred), 4),
                            'change_pct': round(change_pct, 1),
                            'description': f'If {fname} changes to {new_val:.2f} → prediction changes by {change_pct:.0f}%',
                        })
                        break
            if len(flips) >= 5:
                break

    return flips


def _assess_confidence(similar, prediction, confidence, is_clf):
    """Assess prediction reliability based on neighbourhood."""
    if not similar:
        return {'level': 'UNKNOWN', 'message': 'No similar training examples found.'}

    if is_clf:
        agreeing = sum(1 for s in similar if s['true_label'] == prediction)
        agreement_rate = agreeing / len(similar)
        if agreement_rate >= 0.8 and (confidence is None or confidence >= 0.8):
            return {'level': 'HIGH', 'icon': '🟢',
                    'message': f'{agreeing}/{len(similar)} similar training examples had the same label. Prediction is well-supported.'}
        elif agreement_rate >= 0.5:
            return {'level': 'MEDIUM', 'icon': '🟡',
                    'message': f'{agreeing}/{len(similar)} similar examples agree. Some uncertainty exists.'}
        else:
            return {'level': 'LOW', 'icon': '🔴',
                    'message': f'Only {agreeing}/{len(similar)} similar examples agree. Prediction may be unreliable.'}
    else:
        neighbor_vals = [s['true_label'] for s in similar if isinstance(s['true_label'], (int, float))]
        if neighbor_vals:
            neighbor_std = float(np.std(neighbor_vals))
            neighbor_mean = float(np.mean(neighbor_vals))
            return {'level': 'HIGH' if neighbor_std < abs(neighbor_mean) * 0.2 else 'MEDIUM',
                    'icon': '🟢' if neighbor_std < abs(neighbor_mean) * 0.2 else '🟡',
                    'message': f'Neighbours have mean={neighbor_mean:.2f}, std={neighbor_std:.2f}'}
        return {'level': 'UNKNOWN', 'message': 'Cannot assess.'}


def _build_narrative(prediction, confidence, factors, similar, flips,
                      assessment, is_clf, feature_names):
    """Build human-readable narrative."""
    lines = [f'📋 PREDICTION: {_safe(prediction)}']
    if confidence:
        lines.append(f'   Confidence: {confidence:.0%}')
    lines.append('')
    lines.append('🧠 KEY DECISION FACTORS:')
    for i, f in enumerate(factors[:5], 1):
        lines.append(f'   {i}. {f["feature"]} = {f["value"]} (impact: {f["impact"]:.4f})')
    lines.append('')
    if similar:
        agreeing = sum(1 for s in similar[:5] if s.get('true_label') == prediction)
        lines.append(f'📊 SIMILAR EXAMPLES: {agreeing}/{min(5, len(similar))} nearest neighbours agree')
    if flips:
        lines.append('')
        lines.append('🔄 WHAT WOULD CHANGE THE PREDICTION:')
        for f in flips[:3]:
            lines.append(f'   • {f["description"]}')
    lines.append('')
    lines.append(f'⚠️ CONFIDENCE: {assessment["level"]} — {assessment["message"]}')
    return '\n'.join(lines)


def _safe(v):
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 4)
    return v
