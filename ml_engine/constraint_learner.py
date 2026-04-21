"""
AutoML Studio — Data Constraint Learner
Automatically infers business rules from training data and validates new inputs.
"""

import numpy as np
import pandas as pd


def learn_constraints(df, confidence_threshold=0.95):
    """
    Infer data constraints from the training dataset.

    Returns:
        list of constraint dicts with type, column, rule, confidence
    """
    constraints = []

    for col in df.columns:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue

        # 1. Nullability constraints
        null_count = df[col].isnull().sum()
        if null_count == 0:
            constraints.append({
                'type': 'not_null',
                'column': col,
                'rule': f'"{col}" must not be null',
                'confidence': 1.0,
                'icon': '🔒',
            })

        # 2. Range constraints (numeric)
        if pd.api.types.is_numeric_dtype(df[col]):
            min_val = float(vals.min())
            max_val = float(vals.max())
            constraints.append({
                'type': 'range',
                'column': col,
                'rule': f'"{col}" must be between {min_val:.4g} and {max_val:.4g}',
                'min': min_val,
                'max': max_val,
                'confidence': 1.0,
                'icon': '📏',
            })

            # Integer-only constraint
            if pd.api.types.is_integer_dtype(df[col]) or (vals == vals.astype(int)).all():
                constraints.append({
                    'type': 'integer_only',
                    'column': col,
                    'rule': f'"{col}" must be a whole number',
                    'confidence': 1.0,
                    'icon': '🔢',
                })

            # Non-negative constraint
            if min_val >= 0:
                constraints.append({
                    'type': 'non_negative',
                    'column': col,
                    'rule': f'"{col}" must be ≥ 0',
                    'confidence': 1.0,
                    'icon': '➕',
                })

        # 3. Categorical domain constraints
        elif df[col].dtype == 'object':
            unique_vals = vals.unique().tolist()
            if len(unique_vals) <= 50:  # Only for reasonable cardinality
                constraints.append({
                    'type': 'allowed_values',
                    'column': col,
                    'rule': f'"{col}" must be one of {len(unique_vals)} known values',
                    'values': unique_vals[:50],
                    'confidence': 1.0,
                    'icon': '🏷️',
                })

    # 4. Conditional constraints (column A is never null when column B = X)
    conditional = _learn_conditional_constraints(df)
    constraints.extend(conditional)

    # 5. Uniqueness constraints
    for col in df.columns:
        if df[col].nunique() == len(df) and df[col].notna().all():
            constraints.append({
                'type': 'unique',
                'column': col,
                'rule': f'"{col}" must have unique values',
                'confidence': 1.0,
                'icon': '🆔',
            })

    return constraints


def _learn_conditional_constraints(df, max_pairs=20):
    """Learn conditional constraints between column pairs."""
    constraints = []
    cat_cols = df.select_dtypes(include='object').columns.tolist()[:10]

    for col_a in df.columns:
        null_a = df[col_a].isnull()
        if null_a.sum() == 0 or null_a.sum() == len(df):
            continue

        for col_b in cat_cols:
            if col_a == col_b:
                continue

            for val_b in df[col_b].dropna().unique()[:5]:
                mask = df[col_b] == val_b
                if mask.sum() < 10:
                    continue

                null_rate_when_b = df.loc[mask, col_a].isnull().mean()
                if null_rate_when_b == 0:
                    constraints.append({
                        'type': 'conditional_not_null',
                        'column': col_a,
                        'condition_column': col_b,
                        'condition_value': str(val_b),
                        'rule': f'"{col_a}" is never null when "{col_b}" = "{val_b}"',
                        'confidence': 1.0,
                        'icon': '🔗',
                    })

            if len(constraints) >= max_pairs:
                return constraints

    return constraints


def validate_against_constraints(new_data, constraints):
    """
    Validate new data rows against learned constraints.

    Returns:
        dict with violation summary and per-row details
    """
    if isinstance(new_data, dict):
        new_data = pd.DataFrame([new_data])

    violations = []
    row_violations = {}

    for idx, row in new_data.iterrows():
        row_viols = []

        for c in constraints:
            col = c.get('column')
            if col not in new_data.columns:
                continue

            val = row.get(col)

            if c['type'] == 'not_null' and pd.isna(val):
                row_viols.append({
                    'constraint': c['rule'],
                    'column': col,
                    'value': None,
                    'severity': 'high',
                })

            elif c['type'] == 'range' and not pd.isna(val):
                try:
                    num_val = float(val)
                    if num_val < c['min'] or num_val > c['max']:
                        row_viols.append({
                            'constraint': c['rule'],
                            'column': col,
                            'value': num_val,
                            'severity': 'high' if (num_val < c['min'] * 0.5 or num_val > c['max'] * 2) else 'medium',
                        })
                except (ValueError, TypeError):
                    pass

            elif c['type'] == 'integer_only' and not pd.isna(val):
                try:
                    if float(val) != int(float(val)):
                        row_viols.append({
                            'constraint': c['rule'],
                            'column': col,
                            'value': val,
                            'severity': 'low',
                        })
                except (ValueError, TypeError):
                    pass

            elif c['type'] == 'allowed_values' and not pd.isna(val):
                if str(val) not in [str(v) for v in c.get('values', [])]:
                    row_viols.append({
                        'constraint': c['rule'],
                        'column': col,
                        'value': str(val),
                        'severity': 'medium',
                    })

            elif c['type'] == 'non_negative' and not pd.isna(val):
                try:
                    if float(val) < 0:
                        row_viols.append({
                            'constraint': c['rule'],
                            'column': col,
                            'value': val,
                            'severity': 'high',
                        })
                except (ValueError, TypeError):
                    pass

        if row_viols:
            row_violations[int(idx)] = row_viols
            violations.extend(row_viols)

    return {
        'total_violations': len(violations),
        'rows_with_violations': len(row_violations),
        'total_rows': len(new_data),
        'violation_rate': round(len(row_violations) / max(len(new_data), 1) * 100, 2),
        'violations_by_row': row_violations,
        'summary_by_column': _summarize_violations(violations),
        'is_clean': len(violations) == 0,
    }


def _summarize_violations(violations):
    """Summarize violations by column."""
    from collections import Counter
    col_counts = Counter(v['column'] for v in violations)
    return [{'column': col, 'count': count} for col, count in col_counts.most_common()]
