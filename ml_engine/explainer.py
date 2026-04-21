"""
AutoML Studio - SHAP Explainability Engine
Provides global and local model explanations, counterfactual explanations,
and partial dependence plots.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def explain_model(model, X_train, X_test, feature_names, problem_type='classification'):
    """
    Generate SHAP-based explanations for the best model.

    Returns:
        dict: Global explanations, feature interactions, summary stats
    """
    if not HAS_SHAP:
        return {'error': 'SHAP library not installed. Run: pip install shap'}

    try:
        shap_values, explainer = _compute_shap_values(model, X_train, X_test)
    except Exception as e:
        return {'error': f'SHAP computation failed: {str(e)}'}

    # Limit to manageable size
    max_samples = min(200, X_test.shape[0])
    X_explain = X_test.iloc[:max_samples] if isinstance(X_test, pd.DataFrame) else X_test[:max_samples]

    if isinstance(shap_values, list):
        # Multi-class: use the first class or aggregate
        sv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    else:
        sv = shap_values

    sv = sv[:max_samples]

    # Global feature importance from SHAP
    global_importance = _global_importance(sv, feature_names)

    # Feature interaction strengths
    interactions = _feature_interactions(sv, feature_names)

    # Summary statistics
    summary = {
        'n_samples_explained': max_samples,
        'n_features': len(feature_names),
        'problem_type': problem_type,
        'shap_available': True,
    }

    # Store raw SHAP values for local explanations (as lists for JSON)
    shap_data = {
        'values': sv.tolist() if hasattr(sv, 'tolist') else sv,
        'base_value': _get_base_value(explainer, shap_values),
        'feature_names': list(feature_names),
        'feature_data': X_explain.values.tolist() if isinstance(X_explain, pd.DataFrame) else X_explain.tolist(),
    }

    return {
        'global_importance': global_importance,
        'interactions': interactions,
        'summary': summary,
        'shap_data': shap_data,
    }


def explain_single_row(model, X_train, row_data, feature_names, problem_type='classification'):
    """
    Generate local explanation for a single prediction.

    Args:
        row_data: dict or list of feature values

    Returns:
        dict: Per-feature contributions, prediction breakdown
    """
    if not HAS_SHAP:
        return {'error': 'SHAP library not installed'}

    try:
        # Prepare row
        if isinstance(row_data, dict):
            row_df = pd.DataFrame([row_data])
        elif isinstance(row_data, list):
            row_df = pd.DataFrame([row_data], columns=feature_names)
        else:
            row_df = row_data

        shap_values, explainer = _compute_shap_values(model, X_train, row_df)

        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        else:
            sv = shap_values

        sv = sv[0]  # Single row
        base_value = _get_base_value(explainer, shap_values)

        # Prediction
        prediction = model.predict(row_df)[0]
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(row_df)[0].tolist()
        else:
            probabilities = None

        # Build contribution breakdown
        contributions = []
        for i, fname in enumerate(feature_names):
            contributions.append({
                'feature': fname,
                'value': float(row_df.iloc[0, i]) if isinstance(row_df, pd.DataFrame) else float(row_data[i]),
                'shap_value': float(sv[i]),
                'direction': 'positive' if sv[i] > 0 else 'negative',
            })

        # Sort by absolute SHAP value
        contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)

        return {
            'prediction': float(prediction) if not isinstance(prediction, str) else prediction,
            'probabilities': probabilities,
            'base_value': base_value,
            'contributions': contributions,
            'sum_shap': float(np.sum(sv)),
        }
    except Exception as e:
        return {'error': f'Local explanation failed: {str(e)}'}


def whatif_analysis(model, X_train, row_data, feature_name, new_value, feature_names, problem_type='classification'):
    """
    What-if analysis: change one feature value and see impact on prediction.
    """
    if not HAS_SHAP:
        return {'error': 'SHAP library not installed'}

    try:
        if isinstance(row_data, dict):
            row_df = pd.DataFrame([row_data])
            modified_row = row_data.copy()
            modified_row[feature_name] = new_value
            modified_df = pd.DataFrame([modified_row])
        else:
            row_df = pd.DataFrame([row_data], columns=feature_names)
            modified_data = list(row_data)
            idx = list(feature_names).index(feature_name)
            modified_data[idx] = new_value
            modified_df = pd.DataFrame([modified_data], columns=feature_names)

        # Original prediction
        orig_pred = model.predict(row_df)[0]
        orig_proba = model.predict_proba(row_df)[0].tolist() if hasattr(model, 'predict_proba') else None

        # Modified prediction
        new_pred = model.predict(modified_df)[0]
        new_proba = model.predict_proba(modified_df)[0].tolist() if hasattr(model, 'predict_proba') else None

        return {
            'feature_changed': feature_name,
            'original_value': float(row_data.get(feature_name, 0)) if isinstance(row_data, dict) else float(row_data[list(feature_names).index(feature_name)]),
            'new_value': float(new_value),
            'original_prediction': float(orig_pred) if not isinstance(orig_pred, str) else orig_pred,
            'new_prediction': float(new_pred) if not isinstance(new_pred, str) else new_pred,
            'original_probabilities': orig_proba,
            'new_probabilities': new_proba,
            'prediction_changed': bool(orig_pred != new_pred),
        }
    except Exception as e:
        return {'error': f'What-if analysis failed: {str(e)}'}


# ── Counterfactual Explanations ──────────────────────────────

def generate_counterfactuals(model, X_instance, feature_names, desired_outcome,
                              X_train, y_train=None, n_counterfactuals=3):
    """
    Generate diverse counterfactual explanations.
    Uses DiCE if available, otherwise falls back to manual perturbation.

    Args:
        model: trained model
        X_instance: single row (DataFrame or dict)
        feature_names: list of feature names
        desired_outcome: the desired prediction outcome
        X_train: training data for reference
        n_counterfactuals: how many counterfactuals to generate

    Returns:
        dict with original prediction, desired outcome, and list of counterfactuals
    """
    if isinstance(X_instance, dict):
        instance_df = pd.DataFrame([X_instance])
    elif isinstance(X_instance, pd.DataFrame):
        instance_df = X_instance
    else:
        instance_df = pd.DataFrame([X_instance], columns=feature_names)

    original_pred = model.predict(instance_df)[0]
    original_pred = float(original_pred) if not isinstance(original_pred, str) else original_pred

    # Try DiCE first
    try:
        cfs = _dice_counterfactuals(model, instance_df, feature_names, desired_outcome,
                                     X_train, y_train, n_counterfactuals)
        if cfs:
            return {
                'original_prediction': original_pred,
                'desired_outcome': desired_outcome,
                'method': 'DiCE',
                'counterfactuals': cfs,
            }
    except Exception:
        pass

    # Fallback to manual perturbation
    cfs = _manual_counterfactuals(model, instance_df, feature_names, desired_outcome,
                                   X_train, n_counterfactuals)
    return {
        'original_prediction': original_pred,
        'desired_outcome': desired_outcome,
        'method': 'perturbation',
        'counterfactuals': cfs,
    }


def _dice_counterfactuals(model, instance_df, feature_names, desired_outcome,
                           X_train, y_train, n_cfs):
    """Use DiCE library for diverse counterfactual explanations."""
    import dice_ml

    # Prepare training data with target
    continuous_features = list(X_train.select_dtypes(include='number').columns)

    train_with_target = X_train.copy()
    train_with_target['__target__'] = y_train

    d = dice_ml.Data(
        dataframe=train_with_target,
        continuous_features=continuous_features,
        outcome_name='__target__'
    )
    m = dice_ml.Model(model=model, backend='sklearn')
    exp = dice_ml.Dice(d, m, method='random')

    cf = exp.generate_counterfactuals(instance_df, total_CFs=n_cfs, desired_class=desired_outcome)
    cf_df = cf.cf_examples_list[0].final_cfs_df

    counterfactuals = []
    for _, row in cf_df.iterrows():
        changes = []
        for feat in feature_names:
            if feat in instance_df.columns and feat in cf_df.columns:
                orig_val = instance_df[feat].values[0]
                new_val = row[feat]
                if orig_val != new_val:
                    changes.append({
                        'feature': feat,
                        'from': float(orig_val) if isinstance(orig_val, (int, float, np.number)) else str(orig_val),
                        'to': float(new_val) if isinstance(new_val, (int, float, np.number)) else str(new_val),
                    })
        if changes:
            new_pred = row.get('__target__', desired_outcome)
            counterfactuals.append({
                'changes': changes,
                'new_prediction': float(new_pred) if isinstance(new_pred, (int, float, np.number)) else str(new_pred),
                'n_changes': len(changes),
                'feasibility': 'high' if len(changes) <= 2 else 'medium' if len(changes) <= 4 else 'low',
            })

    return counterfactuals


def _manual_counterfactuals(model, instance_df, feature_names, desired_outcome,
                             X_train, n_cfs):
    """Manual perturbation-based counterfactual generation."""
    counterfactuals = []
    numeric_features = instance_df.select_dtypes(include='number').columns.tolist()

    if not numeric_features:
        return counterfactuals

    # For each numeric feature, try perturbing it toward values that change the prediction
    for feat in numeric_features[:10]:  # Limit features to search
        orig_val = float(instance_df[feat].values[0])
        train_vals = X_train[feat].dropna() if feat in X_train.columns else pd.Series([])

        if train_vals.empty:
            continue

        # Try percentile values from training data
        for pctile in [10, 25, 50, 75, 90]:
            target_val = float(train_vals.quantile(pctile / 100))
            if abs(target_val - orig_val) < 1e-6:
                continue

            modified = instance_df.copy()
            modified[feat] = target_val
            new_pred = model.predict(modified)[0]

            new_pred_val = float(new_pred) if isinstance(new_pred, (int, float, np.number)) else str(new_pred)
            desired = float(desired_outcome) if isinstance(desired_outcome, (int, float)) else str(desired_outcome)

            if str(new_pred_val) == str(desired):
                changes = [{
                    'feature': feat,
                    'from': round(orig_val, 4),
                    'to': round(target_val, 4),
                    'change': f'{target_val - orig_val:+.4f}',
                }]

                # Check confidence
                confidence = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(modified)[0]
                    confidence = round(float(max(proba)), 4)

                counterfactuals.append({
                    'changes': changes,
                    'new_prediction': new_pred_val,
                    'confidence': confidence,
                    'n_changes': 1,
                    'feasibility': 'high',
                })

                if len(counterfactuals) >= n_cfs:
                    return counterfactuals

    return counterfactuals


# ── Partial Dependence Plots ─────────────────────────────────

def compute_partial_dependence(model, X_train, feature_names, top_n=3):
    """
    Compute Partial Dependence Plot data for the top-N most important features.

    Returns:
        list of dicts with grid_values and avg_predictions per feature
    """
    try:
        from sklearn.inspection import partial_dependence
    except ImportError:
        return {'error': 'sklearn.inspection not available'}

    # Get top features by model importance
    top_features = _get_top_feature_indices(model, feature_names, top_n)

    if not top_features:
        return []

    pdp_data = []
    for feat_idx in top_features:
        try:
            feat_name = feature_names[feat_idx] if isinstance(feat_idx, int) else feat_idx

            # Compute partial dependence
            result = partial_dependence(
                model, X_train, [feat_idx],
                kind='average',
                grid_resolution=50
            )

            pdp_entry = {
                'feature': feat_name,
                'grid_values': result['values'][0].tolist(),
                'avg_predictions': result['average'][0].tolist(),
            }

            # Also compute ICE curves (individual conditional expectation) for a subsample
            try:
                ice_result = partial_dependence(
                    model, X_train.iloc[:50] if hasattr(X_train, 'iloc') else X_train[:50],
                    [feat_idx], kind='individual', grid_resolution=30
                )
                pdp_entry['individual_predictions'] = ice_result['individual'][0].tolist()
            except Exception:
                pass

            pdp_data.append(pdp_entry)
        except Exception:
            continue

    return pdp_data


def _get_top_feature_indices(model, feature_names, top_n):
    """Get indices of top-N most important features."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            top_idx = np.argsort(importances)[::-1][:top_n]
            return top_idx.tolist()
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).flatten()
            top_idx = np.argsort(importances)[::-1][:top_n]
            return top_idx.tolist()
    except Exception:
        pass

    # Fallback: just use first N features
    return list(range(min(top_n, len(feature_names))))


# ── Internal Helpers ─────────────────────────────────────────

def _compute_shap_values(model, X_train, X_test):
    """Compute SHAP values using the appropriate explainer."""
    # Subsample training data for background
    max_bg = min(100, X_train.shape[0])
    if isinstance(X_train, pd.DataFrame):
        background = X_train.sample(n=max_bg, random_state=42)
    else:
        idx = np.random.RandomState(42).choice(X_train.shape[0], max_bg, replace=False)
        background = X_train[idx]

    # Try TreeExplainer first (faster for tree models)
    model_type = type(model).__name__.lower()
    tree_models = ['randomforest', 'gradientboosting', 'xgb', 'lgbm', 'lightgbm',
                   'extratrees', 'decisiontree', 'catboost']

    is_tree = any(t in model_type for t in tree_models)

    if is_tree:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            return shap_values, explainer
        except Exception:
            pass

    # Try LinearExplainer
    linear_models = ['logistic', 'linear', 'ridge', 'lasso', 'elasticnet', 'sgd']
    is_linear = any(t in model_type for t in linear_models)

    if is_linear:
        try:
            explainer = shap.LinearExplainer(model, background)
            shap_values = explainer.shap_values(X_test)
            return shap_values, explainer
        except Exception:
            pass

    # Fallback: KernelExplainer (works for any model, but slow)
    try:
        if hasattr(model, 'predict_proba'):
            explainer = shap.KernelExplainer(model.predict_proba, background)
        else:
            explainer = shap.KernelExplainer(model.predict, background)

        max_explain = min(50, X_test.shape[0])
        X_subset = X_test.iloc[:max_explain] if isinstance(X_test, pd.DataFrame) else X_test[:max_explain]
        shap_values = explainer.shap_values(X_subset, nsamples=100)
        return shap_values, explainer
    except Exception as e:
        raise RuntimeError(f'All SHAP explainers failed: {str(e)}')


def _get_base_value(explainer, shap_values):
    """Extract base value from explainer."""
    try:
        if hasattr(explainer, 'expected_value'):
            ev = explainer.expected_value
            if isinstance(ev, (list, np.ndarray)):
                return float(ev[1]) if len(ev) == 2 else float(ev[0])
            return float(ev)
    except Exception:
        pass
    return 0.0


def _global_importance(shap_values, feature_names):
    """Compute mean absolute SHAP value per feature."""
    if isinstance(shap_values, np.ndarray):
        mean_abs = np.abs(shap_values).mean(axis=0)
    else:
        mean_abs = np.abs(np.array(shap_values)).mean(axis=0)

    # Normalize
    total = mean_abs.sum()
    if total > 0:
        mean_abs = mean_abs / total

    importance = []
    for i, name in enumerate(feature_names):
        if i < len(mean_abs):
            importance.append({
                'feature': name,
                'importance': round(float(mean_abs[i]), 4),
                'mean_abs_shap': round(float(np.abs(shap_values[:, i]).mean()), 6) if i < shap_values.shape[1] else 0,
            })

    importance.sort(key=lambda x: x['importance'], reverse=True)
    return importance[:25]


def _feature_interactions(shap_values, feature_names, top_n=10):
    """Estimate feature interaction strengths from SHAP value correlations."""
    if not isinstance(shap_values, np.ndarray):
        shap_values = np.array(shap_values)

    n_features = min(shap_values.shape[1], 15)  # Limit for performance
    interactions = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            # Correlation between SHAP values as proxy for interaction
            try:
                corr = abs(float(np.corrcoef(shap_values[:, i], shap_values[:, j])[0, 1]))
                if not np.isnan(corr) and corr > 0.1:
                    interactions.append({
                        'feature_1': feature_names[i],
                        'feature_2': feature_names[j],
                        'interaction_strength': round(corr, 4),
                    })
            except Exception:
                pass

    interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
    return interactions[:top_n]
