"""
AutoML Problem Solver - Data Transformer
Handles encoding, scaling, feature selection, class imbalance, and NLP text processing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold

from ml_engine.text_processor import detect_text_columns, process_text_columns


def transform_dataset(df, profile):
    """
    Automatically transform the dataset for ML training.
    
    Returns:
        tuple: (transformed_df, transform_report, transform_metadata)
    """
    report = {
        'steps': [],
        'summary': {},
    }
    
    target_col = profile.get('target_column')
    problem_type = profile.get('problem_type', 'classification')
    
    # Separate features and target
    if target_col and target_col in df.columns:
        X = df.drop(columns=[target_col])
        y = df[target_col].copy()
    else:
        X = df.copy()
        y = None
    
    # Step 1: Encode the target variable (if categorical)
    target_encoder = None
    if y is not None and y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), name=target_col)
        target_encoder = le
        report['steps'].append({
            'name': 'Encode Target',
            'icon': '🏷️',
            'description': f'Label-encoded target column "{target_col}" ({len(le.classes_)} classes)',
            'count': len(le.classes_),
            'mapping': {str(cls): int(i) for i, cls in enumerate(le.classes_)},
            'applied': True,
        })
    
    # Step 2: Process text columns (NLP)
    text_columns = profile.get('text_columns', [])
    # Also detect from current X in case profile didn't detect
    if not text_columns:
        text_columns = detect_text_columns(X)
    
    text_report = None
    if text_columns:
        text_cols_in_X = [c for c in text_columns if c in X.columns]
        if text_cols_in_X:
            X, text_report = process_text_columns(X, text_cols_in_X)
            report['steps'].append({
                'name': 'Process Text Columns (NLP)',
                'icon': '📝',
                'description': f'Processed {len(text_cols_in_X)} text columns: {", ".join(text_cols_in_X)}. '
                               f'Added {text_report.get("features_added", 0)} NLP features '
                               f'(TF-IDF, sentiment, statistics)',
                'count': text_report.get('features_added', 0),
                'methods': text_report.get('methods_used', []),
                'applied': True,
            })
    
    # Step 3: Encode categorical features
    X, step = _encode_categoricals(X)
    report['steps'].append(step)
    
    # Step 4: Remove low-variance features
    X, step = _remove_low_variance(X)
    report['steps'].append(step)
    
    # Step 5: Remove highly correlated features
    X, step = _remove_high_correlation(X, threshold=0.95)
    report['steps'].append(step)
    
    # Step 6: Scale numeric features
    X, step, numeric_cols_to_scale = _scale_features(X)
    report['steps'].append(step)
    
    # Step 7: Handle class imbalance (classification only)
    imbalance_applied = False
    imbalance_ratio = None
    if problem_type == 'classification' and y is not None:
        X, y, step, imbalance_ratio = _handle_imbalance(X, y)
        report['steps'].append(step)
        imbalance_applied = step.get('applied', False)
    
    # Recombine
    if y is not None:
        transformed_df = X.copy()
        transformed_df[target_col] = y.values if len(y) == len(X) else y.reset_index(drop=True).values
    else:
        transformed_df = X.copy()
    
    # Summary
    report['summary'] = {
        'original_features': profile.get('n_cols', 0) - 1,
        'final_features': X.shape[1],
        'features_added': max(0, X.shape[1] - (profile.get('n_cols', 0) - 1)),
        'features_removed': max(0, (profile.get('n_cols', 0) - 1) - X.shape[1]),
        'total_rows': len(transformed_df),
        'imbalance_applied': imbalance_applied,
        'imbalance_ratio': imbalance_ratio,
        'text_features_added': text_report.get('features_added', 0) if text_report else 0,
    }
    
    # Store metadata for potential retrain
    metadata = {
        'target_encoder': target_encoder,
        'scaler': None,  # Scaler will be fitted after train/test split
        'numeric_cols_to_scale': numeric_cols_to_scale,
        'target_column': target_col,
        'problem_type': problem_type,
        'feature_names': list(X.columns),
        'text_columns_processed': text_columns,
    }
    
    return transformed_df, report, metadata


def _encode_categoricals(X):
    """Encode categorical features using appropriate strategies."""
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    encodings = {}
    
    cols_to_drop = []
    new_dfs = []
    
    for col in categorical_cols:
        n_unique = X[col].nunique()
        
        if n_unique == 2:
            # Binary: Label encode
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encodings[col] = f'label-encoded (binary, {n_unique} values)'
        elif n_unique <= 10:
            # Low cardinality: One-hot encode
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True, dtype=int)
            new_dfs.append(dummies)
            cols_to_drop.append(col)
            encodings[col] = f'one-hot encoded ({n_unique} values → {len(dummies.columns)} features)'
        else:
            # High cardinality: Frequency encoding
            freq_map = X[col].value_counts(normalize=True).to_dict()
            X[col] = X[col].map(freq_map).fillna(0)
            encodings[col] = f'frequency-encoded ({n_unique} unique values)'
    
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
    
    if new_dfs:
        X = pd.concat([X] + new_dfs, axis=1)
    
    step = {
        'name': 'Encode Categoricals',
        'icon': '🔤',
        'description': f'Encoded {len(categorical_cols)} categorical columns' if categorical_cols else 'No categorical columns to encode',
        'count': len(categorical_cols),
        'encodings': encodings,
        'applied': len(categorical_cols) > 0,
    }
    return X, step


def _remove_low_variance(X):
    """Remove features with very low variance."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        return X, {
            'name': 'Remove Low Variance',
            'icon': '📉',
            'description': 'No numeric columns to check',
            'count': 0,
            'applied': False,
        }
    
    try:
        selector = VarianceThreshold(threshold=0.01)
        X_numeric = X[numeric_cols]
        selector.fit(X_numeric)
        
        low_var_mask = ~selector.get_support()
        low_var_cols = [numeric_cols[i] for i in range(len(numeric_cols)) if low_var_mask[i]]
        
        if low_var_cols:
            X = X.drop(columns=low_var_cols)
    except Exception:
        low_var_cols = []
    
    step = {
        'name': 'Remove Low Variance',
        'icon': '📉',
        'description': f'Removed {len(low_var_cols)} low-variance features: {", ".join(low_var_cols)}' if low_var_cols else 'All features have sufficient variance',
        'count': len(low_var_cols),
        'columns': low_var_cols,
        'applied': len(low_var_cols) > 0,
    }
    return X, step


def _remove_high_correlation(X, threshold=0.95):
    """Remove highly correlated features."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return X, {
            'name': 'Remove High Correlation',
            'icon': '🔗',
            'description': 'Not enough numeric columns to check',
            'count': 0,
            'applied': False,
        }
    
    corr_matrix = X[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    
    if to_drop:
        X = X.drop(columns=to_drop)
    
    step = {
        'name': 'Remove High Correlation',
        'icon': '🔗',
        'description': f'Removed {len(to_drop)} highly correlated features (>{threshold*100}%): {", ".join(to_drop)}' if to_drop else 'No highly correlated features found',
        'count': len(to_drop),
        'columns': to_drop,
        'applied': len(to_drop) > 0,
    }
    return X, step


def _scale_features(X):
    """Identify numeric features for scaling (scaling is deferred to after train/test split)."""
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    step = {
        'name': 'Scale Features',
        'icon': '⚖️',
        'description': f'Identified {len(numeric_cols)} numeric features for StandardScaler (applied after train/test split to prevent data leakage)' if numeric_cols else 'No numeric features to scale',
        'count': len(numeric_cols),
        'applied': len(numeric_cols) > 0,
    }
    return X, step, numeric_cols


def _handle_imbalance(X, y):
    """Handle class imbalance using the best available strategy."""
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    
    value_counts = y.value_counts()
    majority = value_counts.max()
    minority = value_counts.min()
    ratio = majority / max(minority, 1)
    
    if ratio <= 2.0 or len(y) <= 50 or minority < 6:
        step = {
            'name': 'Handle Class Imbalance',
            'icon': '⚖️',
            'description': f'Classes are balanced enough (ratio {ratio:.1f}:1). No resampling needed.' if ratio <= 2 else 'Insufficient samples for resampling.',
            'count': 0,
            'imbalance_ratio': round(ratio, 2),
            'applied': False,
        }
        return X, y, step, round(ratio, 2)
    
    # Auto-select strategy based on severity and dataset size
    k_neighbors = min(5, minority - 1)
    strategies = []
    
    # For extreme imbalance (>10:1) or very large datasets, try undersampling first
    if ratio > 10 and len(y) > 5000:
        strategies.append(('RandomUnderSampler', RandomUnderSampler(random_state=42)))
    
    # ADASYN for moderate-to-high imbalance (adapts to boundary regions)
    if minority >= 10 and k_neighbors >= 2:
        strategies.append(('ADASYN', ADASYN(random_state=42, n_neighbors=k_neighbors)))
    
    # BorderlineSMOTE for borderline-focused oversampling
    if minority >= 10 and k_neighbors >= 2:
        strategies.append(('BorderlineSMOTE', BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors)))
    
    # Standard SMOTE as reliable fallback
    strategies.append(('SMOTE', SMOTE(random_state=42, k_neighbors=k_neighbors)))
    
    # Try strategies in order, use first that succeeds
    for strategy_name, sampler in strategies:
        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            X_out = pd.DataFrame(X_resampled, columns=X.columns)
            y_out = pd.Series(y_resampled, name=y.name)
            
            step = {
                'name': 'Handle Class Imbalance',
                'icon': '⚖️',
                'description': f'Applied {strategy_name} (imbalance ratio was {ratio:.1f}:1). Rows: {len(y)} → {len(y_out)}',
                'count': abs(len(y_out) - len(y)),
                'original_distribution': {str(k): int(v) for k, v in value_counts.items()},
                'new_distribution': {str(k): int(v) for k, v in y_out.value_counts().items()},
                'imbalance_ratio': round(ratio, 2),
                'strategy': strategy_name,
                'applied': True,
            }
            return X_out, y_out, step, round(ratio, 2)
        except Exception:
            continue
    
    # All strategies failed
    step = {
        'name': 'Handle Class Imbalance',
        'icon': '⚖️',
        'description': f'All resampling strategies failed (ratio {ratio:.1f}:1). Will use class_weight instead during training.',
        'count': 0,
        'imbalance_ratio': round(ratio, 2),
        'applied': False,
    }
    return X, y, step, round(ratio, 2)


def fit_scaler(X_train, numeric_cols):
    """Fit StandardScaler on training data only to prevent data leakage.
    
    Args:
        X_train: Training feature DataFrame
        numeric_cols: List of numeric column names to scale
    
    Returns:
        Fitted StandardScaler instance, or None if no numeric columns
    """
    if not numeric_cols:
        return None
    cols_present = [c for c in numeric_cols if c in X_train.columns]
    if not cols_present:
        return None
    scaler = StandardScaler()
    scaler.fit(X_train[cols_present])
    return scaler


def apply_scaler(X, scaler, numeric_cols):
    """Apply a pre-fitted scaler to a dataset split.
    
    Args:
        X: Feature DataFrame (train or test split)
        scaler: Fitted StandardScaler instance
        numeric_cols: List of numeric column names that were scaled
    
    Returns:
        DataFrame with scaled numeric columns
    """
    if scaler is None or not numeric_cols:
        return X
    X = X.copy()
    cols_present = [c for c in numeric_cols if c in X.columns]
    if cols_present:
        X[cols_present] = scaler.transform(X[cols_present])
    return X
