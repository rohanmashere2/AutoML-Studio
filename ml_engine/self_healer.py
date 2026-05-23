"""
AutoML Studio — Self-Healing Pipeline (Feature #6)
Wraps pipeline stages with automatic error diagnosis, fix, and retry logic.
"""

import numpy as np
import pandas as pd
import traceback


class SelfHealer:
    """Diagnose pipeline errors and auto-apply fixes."""

    def __init__(self):
        self.healing_log = []

    def heal_training_error(self, error, X_train, y_train, X_test, y_test,
                             model_name, problem_type):
        """Diagnose and fix training errors."""
        err_str = str(error).lower()
        fix_applied = None

        # NaN in training data
        if 'nan' in err_str or 'missing' in err_str or 'null' in err_str:
            X_train = pd.DataFrame(X_train).fillna(pd.DataFrame(X_train).median())
            X_test = pd.DataFrame(X_test).fillna(pd.DataFrame(X_test).median())
            fix_applied = 'Filled NaN values with column medians'

        # Infinite values
        elif 'inf' in err_str:
            X_train = pd.DataFrame(X_train).replace([np.inf, -np.inf], np.nan)
            X_train = X_train.fillna(X_train.median())
            X_test = pd.DataFrame(X_test).replace([np.inf, -np.inf], np.nan)
            X_test = X_test.fillna(X_test.median())
            fix_applied = 'Replaced infinite values with medians'

        # Memory error
        elif 'memory' in err_str:
            n = len(X_train)
            sample_n = min(n, 50000)
            idx = np.random.RandomState(42).choice(n, sample_n, replace=False)
            X_train = X_train.iloc[idx] if hasattr(X_train, 'iloc') else X_train[idx]
            y_train = y_train.iloc[idx] if hasattr(y_train, 'iloc') else np.array(y_train)[idx]
            fix_applied = f'Downsampled from {n} to {sample_n} rows (memory limit)'

        # Single class
        elif 'single class' in err_str or 'only one class' in err_str:
            fix_applied = 'ABORT: Target has only one class — cannot train classifier'

        # Convergence
        elif 'converge' in err_str:
            fix_applied = 'Increased max_iter. Retry with relaxed convergence.'

        if fix_applied:
            self.healing_log.append({
                'stage': 'training',
                'model': model_name,
                'error': str(error)[:200],
                'fix': fix_applied,
                'auto_fixed': 'ABORT' not in fix_applied,
            })

        return X_train, y_train, X_test, y_test, fix_applied

    def heal_cleaning_error(self, error, df):
        """Fix cleaning stage errors."""
        err_str = str(error).lower()
        fix_applied = None

        if 'memory' in err_str:
            if len(df) > 100000:
                df = df.sample(100000, random_state=42)
                fix_applied = 'Downsampled to 100K rows for cleaning'
        elif 'dtype' in err_str or 'type' in err_str:
            for col in df.select_dtypes(include='object').columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception:
                    pass
            fix_applied = 'Attempted numeric conversion on object columns'

        if fix_applied:
            self.healing_log.append({
                'stage': 'cleaning', 'error': str(error)[:200],
                'fix': fix_applied, 'auto_fixed': True,
            })
        return df, fix_applied

    def heal_transform_error(self, error, df, target_col):
        """Fix transformation stage errors."""
        err_str = str(error).lower()
        fix_applied = None

        # High cardinality one-hot explosion
        if 'memory' in err_str or 'too many' in err_str:
            for col in df.select_dtypes(include='object').columns:
                if col == target_col:
                    continue
                if df[col].nunique() > 50:
                    freq = df[col].value_counts(normalize=True)
                    df[col] = df[col].map(freq).fillna(0)
            fix_applied = 'Applied frequency encoding to high-cardinality columns'

        if fix_applied:
            self.healing_log.append({
                'stage': 'transform', 'error': str(error)[:200],
                'fix': fix_applied, 'auto_fixed': True,
            })
        return df, fix_applied

    def diagnose_poor_performance(self, leaderboard, X_train, y_train,
                                   problem_type):
        """Diagnose when all models perform badly."""
        if not leaderboard:
            return []

        best_score = leaderboard[0].get('primary_metric', 0)
        is_clf = problem_type == 'classification'
        diagnoses = []

        # Worse than random
        threshold = 0.55 if is_clf else 0.1
        if best_score < threshold:
            # Check target column
            y = np.array(y_train)
            n_unique = len(np.unique(y))
            if n_unique == 1:
                diagnoses.append({
                    'issue': 'Constant target',
                    'severity': 'critical',
                    'message': 'Target column has only 1 unique value. Cannot build a predictive model.',
                    'fix': 'Verify target column selection.',
                    'icon': '🔴',
                })
            elif is_clf and n_unique > len(y) * 0.5:
                diagnoses.append({
                    'issue': 'Target is likely continuous',
                    'severity': 'high',
                    'message': f'Target has {n_unique} unique values — looks like regression, not classification.',
                    'fix': 'Switch problem_type to regression.',
                    'icon': '🟠',
                })

            # Check feature quality
            X = np.array(X_train)
            constant_cols = np.sum(np.std(X, axis=0) < 1e-8)
            if constant_cols > X.shape[1] * 0.5:
                diagnoses.append({
                    'issue': 'Too many constant features',
                    'severity': 'high',
                    'message': f'{constant_cols}/{X.shape[1]} features are constant. No predictive signal.',
                    'fix': 'Drop constant features and engineer new ones.',
                    'icon': '🟠',
                })

        # Overfitting check
        if leaderboard:
            best = leaderboard[0].get('metrics', {})
            train_key = 'train_accuracy' if is_clf else 'train_r2'
            test_key = 'accuracy' if is_clf else 'r2'
            train_s = best.get(train_key, 0)
            test_s = best.get(test_key, 0)
            if train_s - test_s > 0.15:
                diagnoses.append({
                    'issue': 'Severe overfitting',
                    'severity': 'high',
                    'message': f'Train={train_s:.1%}, Test={test_s:.1%}. Gap of {train_s-test_s:.1%}.',
                    'fix': 'Add regularisation, reduce features, or collect more data.',
                    'icon': '🟠',
                    'auto_fix': 'reduce_features',
                })

        return diagnoses

    def get_healing_report(self):
        """Return full healing log."""
        return {
            'total_heals': len(self.healing_log),
            'auto_fixed': sum(1 for h in self.healing_log if h.get('auto_fixed')),
            'log': self.healing_log,
        }
