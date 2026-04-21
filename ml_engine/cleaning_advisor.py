"""
AutoML Studio — AI-Based Data Cleaning Advisor
Generates cleaning suggestions with impact benchmarking.
Each suggestion is measured against a quick baseline to show delta in model performance.
"""

import numpy as np
import pandas as pd


def generate_cleaning_suggestions(df, profile=None):
    """
    Analyze dataset and generate cleaning suggestions.

    Returns:
        list of suggestion dicts, each with:
        - id, type, column, title, description, impact, rationale, auto_applicable
    """
    suggestions = []
    sid = 0

    # 1. Duplicate rows
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        pct = round(n_dup / len(df) * 100, 1)
        suggestions.append({
            'id': sid,
            'type': 'duplicates',
            'column': '__all__',
            'title': f'Remove {n_dup} duplicate rows',
            'description': f'{n_dup} duplicate rows found ({pct}% of data). Removing them prevents data leakage and reduces training bias.',
            'impact': 'high' if pct > 5 else 'medium',
            'rationale': 'Duplicate rows can bias model training and inflate accuracy metrics.',
            'auto_applicable': True,
            'icon': '🔄',
        })
        sid += 1

    # 2. Missing values per column
    for col in df.columns:
        missing = df[col].isnull().sum()
        if missing == 0:
            continue
        pct = round(missing / len(df) * 100, 1)

        if pct > 60:
            # Suggest drop column
            suggestions.append({
                'id': sid,
                'type': 'drop_column',
                'column': col,
                'title': f'Drop column "{col}" ({pct}% missing)',
                'description': f'Column has {pct}% missing values. Too much missing data to impute reliably.',
                'impact': 'high',
                'rationale': f'Columns with >{60}% missing values add noise rather than signal.',
                'auto_applicable': True,
                'icon': '🗑️',
            })
        elif pct > 0:
            # Suggest fill strategy
            if df[col].dtype in ('float64', 'int64', 'float32', 'int32'):
                skew = abs(df[col].skew()) if df[col].nunique() > 2 else 0
                if skew > 1:
                    strategy = 'median'
                    reason = f'Column is skewed (skewness={skew:.2f}), median is more robust than mean.'
                else:
                    strategy = 'mean'
                    reason = f'Column is approximately normal, mean imputation is appropriate.'
            else:
                strategy = 'mode'
                reason = 'Categorical column — filling with most frequent value.'

            suggestions.append({
                'id': sid,
                'type': 'fill_missing',
                'column': col,
                'title': f'Fill {missing} missing values in "{col}" with {strategy}',
                'description': f'{missing} missing values ({pct}%). Suggested: fill with {strategy}.',
                'impact': 'high' if pct > 20 else 'medium' if pct > 5 else 'low',
                'rationale': reason,
                'strategy': strategy,
                'auto_applicable': True,
                'icon': '🩹',
            })
        sid += 1

    # 3. Outliers (numeric columns)
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        q1 = vals.quantile(0.25)
        q3 = vals.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((vals < lower) | (vals > upper)).sum()
        if n_outliers > 0:
            pct = round(n_outliers / len(vals) * 100, 1)
            if pct > 0.5:
                suggestions.append({
                    'id': sid,
                    'type': 'remove_outliers',
                    'column': col,
                    'title': f'Remove {n_outliers} outliers from "{col}"',
                    'description': f'{n_outliers} outliers detected ({pct}%) using IQR method. Values outside [{lower:.2f}, {upper:.2f}].',
                    'impact': 'medium' if pct < 5 else 'high',
                    'rationale': 'Outliers can skew model training and reduce accuracy, especially for distance-based algorithms.',
                    'bounds': {'lower': round(lower, 4), 'upper': round(upper, 4)},
                    'auto_applicable': True,
                    'icon': '📊',
                })
                sid += 1

    # 4. Constant/near-constant columns
    for col in df.columns:
        nunique = df[col].nunique()
        if nunique <= 1:
            suggestions.append({
                'id': sid,
                'type': 'drop_column',
                'column': col,
                'title': f'Drop constant column "{col}"',
                'description': f'Column has only {nunique} unique value(s). Provides no information for modeling.',
                'impact': 'medium',
                'rationale': 'Constant columns add no predictive power and waste computation.',
                'auto_applicable': True,
                'icon': '🗑️',
            })
            sid += 1
        elif nunique == len(df) and df[col].dtype == 'object':
            suggestions.append({
                'id': sid,
                'type': 'drop_column',
                'column': col,
                'title': f'Drop high-cardinality ID column "{col}"',
                'description': f'Column has {nunique} unique values (all unique). Likely an ID column.',
                'impact': 'medium',
                'rationale': 'ID-like columns with all unique values cannot help prediction.',
                'auto_applicable': True,
                'icon': '🆔',
            })
            sid += 1

    # 5. Data type suggestions
    for col in df.select_dtypes(include='object').columns:
        # Check if it's actually numeric
        try:
            converted = pd.to_numeric(df[col], errors='coerce')
            valid_pct = converted.notna().sum() / len(df) * 100
            if valid_pct > 90:
                suggestions.append({
                    'id': sid,
                    'type': 'fix_type',
                    'column': col,
                    'title': f'Convert "{col}" from text to numeric',
                    'description': f'{valid_pct:.0f}% of values in "{col}" are numeric but stored as text.',
                    'impact': 'medium',
                    'rationale': 'Numeric features stored as text prevent proper mathematical operations.',
                    'auto_applicable': True,
                    'icon': '🔢',
                })
                sid += 1
        except Exception:
            pass

    # 6. High correlation suggestion
    if len(numeric_cols) > 1:
        try:
            corr = df[numeric_cols].corr().abs()
            upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            high_corr = [(col, row, corr.loc[row, col])
                         for col in upper_tri.columns
                         for row in upper_tri.index
                         if upper_tri.loc[row, col] > 0.95]
            for col1, col2, corr_val in high_corr[:5]:
                suggestions.append({
                    'id': sid,
                    'type': 'drop_correlated',
                    'column': col2,
                    'title': f'Drop "{col2}" (95%+ correlated with "{col1}")',
                    'description': f'Correlation = {corr_val:.3f}. Keeping both adds multicollinearity.',
                    'impact': 'medium',
                    'rationale': 'Highly correlated features are redundant and can cause instability.',
                    'auto_applicable': True,
                    'icon': '🔗',
                })
                sid += 1
        except Exception:
            pass

    return suggestions


# ── Impact Benchmarking ──────────────────────────────────────

def benchmark_cleaning_impact(df, target_col, suggestions, problem_type='classification'):
    """
    Run quick benchmarks to measure the impact of each major cleaning suggestion.
    Uses a fast RandomForest with 3-fold CV for before/after comparison.

    Args:
        df: the DataFrame
        target_col: target column name
        suggestions: list of suggestion dicts
        problem_type: 'classification' or 'regression'

    Returns:
        list of suggestions enriched with 'measured_impact' field
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    if target_col not in df.columns:
        return suggestions

    # Quick model setup
    if problem_type == 'classification':
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        scoring = 'accuracy'
    else:
        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        scoring = 'r2'

    # Compute baseline score
    try:
        X_base, y_base = _prepare_quick_train(df, target_col)
        if X_base is None or len(X_base) < 20:
            return suggestions
        base_scores = cross_val_score(model, X_base, y_base, cv=3, scoring=scoring, n_jobs=-1)
        base_score = float(base_scores.mean())
    except Exception:
        return suggestions

    # Benchmark each suggestion
    enriched = []
    for s in suggestions:
        s = dict(s)  # copy
        try:
            df_modified = _apply_single_suggestion(df.copy(), s)
            X_mod, y_mod = _prepare_quick_train(df_modified, target_col)
            if X_mod is not None and len(X_mod) >= 20:
                mod_scores = cross_val_score(model, X_mod, y_mod, cv=3, scoring=scoring, n_jobs=-1)
                mod_score = float(mod_scores.mean())
                delta = mod_score - base_score

                s['measured_impact'] = {
                    'before_score': round(base_score, 4),
                    'after_score': round(mod_score, 4),
                    'delta': round(delta, 4),
                    'delta_pct': f'{delta * 100:+.2f}%',
                    'recommendation': 'apply' if delta >= -0.005 else 'skip',
                    'metric': scoring,
                }
        except Exception:
            s['measured_impact'] = {'error': 'Could not benchmark this suggestion'}

        enriched.append(s)

    return enriched


def _prepare_quick_train(df, target_col):
    """Prepare a quick training set (numeric features only)."""
    try:
        if target_col not in df.columns:
            return None, None
        y = df[target_col]
        X = df.drop(columns=[target_col]).select_dtypes(include='number')
        # Drop columns with all NaN
        X = X.dropna(axis=1, how='all')
        # Fill remaining NaN with median
        X = X.fillna(X.median())
        # Remove rows where target is NaN
        mask = y.notna()
        return X.loc[mask], y.loc[mask]
    except Exception:
        return None, None


def _apply_single_suggestion(df, suggestion):
    """Apply a single cleaning suggestion to a DataFrame copy."""
    s_type = suggestion.get('type')
    col = suggestion.get('column')

    if s_type == 'duplicates':
        return df.drop_duplicates()

    elif s_type == 'drop_column':
        if col in df.columns:
            return df.drop(columns=[col])

    elif s_type == 'drop_correlated':
        if col in df.columns:
            return df.drop(columns=[col])

    elif s_type == 'fill_missing':
        if col in df.columns:
            strategy = suggestion.get('strategy', 'mean')
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif strategy == 'mode':
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')

    elif s_type == 'remove_outliers':
        bounds = suggestion.get('bounds', {})
        if col in df.columns and bounds:
            df = df[(df[col] >= bounds['lower']) & (df[col] <= bounds['upper'])]

    elif s_type == 'fix_type':
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


# ── Apply Suggestions ────────────────────────────────────────

def apply_suggestions(df, suggestions, accepted_ids):
    """
    Apply accepted cleaning suggestions to the dataframe.

    Args:
        df: original DataFrame
        suggestions: list of suggestion dicts
        accepted_ids: list of suggestion IDs to apply

    Returns:
        (cleaned_df, applied_report)
    """
    df = df.copy()
    applied = []

    # Index suggestions by ID
    by_id = {s['id']: s for s in suggestions}

    for sid in accepted_ids:
        s = by_id.get(sid)
        if not s:
            continue

        try:
            if s['type'] == 'duplicates':
                before = len(df)
                df = df.drop_duplicates()
                applied.append({**s, 'result': f'Removed {before - len(df)} duplicate rows'})

            elif s['type'] == 'drop_column':
                col = s['column']
                if col in df.columns:
                    df = df.drop(columns=[col])
                    applied.append({**s, 'result': f'Dropped column {col}'})

            elif s['type'] == 'drop_correlated':
                col = s['column']
                if col in df.columns:
                    df = df.drop(columns=[col])
                    applied.append({**s, 'result': f'Dropped correlated column {col}'})

            elif s['type'] == 'fill_missing':
                col = s['column']
                strategy = s.get('strategy', 'mean')
                if col in df.columns:
                    before = df[col].isnull().sum()
                    if strategy == 'mean':
                        df[col] = df[col].fillna(df[col].mean())
                    elif strategy == 'median':
                        df[col] = df[col].fillna(df[col].median())
                    elif strategy == 'mode':
                        mode_val = df[col].mode()
                        df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
                    applied.append({**s, 'result': f'Filled {before} missing values with {strategy}'})

            elif s['type'] == 'remove_outliers':
                col = s['column']
                bounds = s.get('bounds', {})
                if col in df.columns and bounds:
                    before = len(df)
                    df = df[(df[col] >= bounds['lower']) & (df[col] <= bounds['upper'])]
                    applied.append({**s, 'result': f'Removed {before - len(df)} outlier rows'})

            elif s['type'] == 'fix_type':
                col = s['column']
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    applied.append({**s, 'result': f'Converted {col} to numeric'})

        except Exception as e:
            applied.append({**s, 'result': f'Error: {str(e)}'})

    return df, {
        'n_applied': len(applied),
        'applied': [{k: v for k, v in a.items() if k != 'auto_applicable'} for a in applied],
        'final_shape': {'rows': len(df), 'columns': len(df.columns)},
    }
