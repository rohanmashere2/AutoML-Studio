"""
AutoML Studio — Column Fate Tracker
After every transform, display a full table showing every original column's
fate (kept / dropped / encoded / scaled), the reason, importance score,
missing %, and whether it can be restored.
"""

import numpy as np
import pandas as pd


class ColumnFateTracker:
    """Tracks what happened to each column through the pipeline."""

    def track(self, original_df, transformed_df, clean_report, transform_report,
              profile=None, journal=None):
        """
        Build the column fate table.

        Args:
            original_df: the original uploaded DataFrame
            transformed_df: the final transformed DataFrame
            clean_report: cleaning step report
            transform_report: transformation step report
            profile: dataset profile dict
            journal: transformation journal entries (list of dicts)

        Returns:
            list of column fate dicts
        """
        target_col = profile.get('target_column') if profile else None
        original_columns = list(original_df.columns)
        transformed_columns = list(transformed_df.columns) if transformed_df is not None else []

        # Build lookup sets
        transformed_set = set(transformed_columns)

        # Extract operation details from reports
        dropped_clean = self._extract_dropped(clean_report, 'Drop High-Missing Columns')
        dropped_lowvar = self._extract_dropped(transform_report, 'Remove Low Variance')
        dropped_corr = self._extract_dropped(transform_report, 'Remove High Correlation')
        encoded_cols = self._extract_encoded(transform_report)
        scaled_info = self._extract_scaled(transform_report)

        # Compute importance scores
        importance_scores = self._compute_importance(original_df, target_col)

        fate_table = []

        for col in original_columns:
            missing_count = int(original_df[col].isnull().sum())
            missing_pct = round(missing_count / max(len(original_df), 1) * 100, 2)
            importance = importance_scores.get(col, 0.0)

            # Determine fate
            if col == target_col:
                fate = 'target'
                reason = 'Used as prediction target'
                operation = 'Label-encoded' if col not in transformed_set else 'Kept as target'
                restorable = False

            elif col in dropped_clean:
                fate = 'dropped'
                reason = f'Dropped during cleaning: {dropped_clean[col]}'
                operation = 'Removed'
                restorable = True

            elif col in dropped_lowvar:
                fate = 'dropped'
                reason = f'Low variance: {dropped_lowvar[col]}'
                operation = 'Removed'
                restorable = True

            elif col in dropped_corr:
                fate = 'dropped'
                reason = f'High correlation: {dropped_corr[col]}'
                operation = 'Removed'
                restorable = True

            elif col in encoded_cols:
                fate = 'encoded'
                reason = f'Categorical encoding applied'
                operation = encoded_cols[col]
                restorable = False

            elif col in transformed_set:
                if col in scaled_info:
                    fate = 'scaled'
                    reason = 'StandardScaler applied'
                    operation = 'Scaled (mean=0, std=1)'
                else:
                    fate = 'kept'
                    reason = 'Passed through unchanged'
                    operation = 'No transformation'
                restorable = False

            else:
                # Column disappeared — check if it was one-hot encoded
                prefix_matches = [tc for tc in transformed_columns
                                  if tc.startswith(f'{col}_')]
                if prefix_matches:
                    fate = 'encoded'
                    reason = f'One-hot encoded into {len(prefix_matches)} columns'
                    operation = f'One-hot → {", ".join(prefix_matches[:3])}'
                    if len(prefix_matches) > 3:
                        operation += f' + {len(prefix_matches) - 3} more'
                    restorable = False
                else:
                    fate = 'dropped'
                    reason = 'Removed during transformation (reason unknown)'
                    operation = 'Removed'
                    restorable = True

            # Find journal step if available
            journal_step = None
            if journal:
                for idx, entry in enumerate(journal):
                    if col in entry.get('columns_affected', []):
                        journal_step = idx
                        break

            fate_table.append({
                'column': col,
                'fate': fate,
                'reason': reason,
                'operation': operation,
                'importance_score': round(importance, 4),
                'missing_pct': missing_pct,
                'missing_count': missing_count,
                'dtype': str(original_df[col].dtype),
                'n_unique': int(original_df[col].nunique()),
                'restorable': restorable,
                'journal_step': journal_step,
            })

        # Sort: dropped first, then by importance (descending)
        fate_order = {'dropped': 0, 'encoded': 1, 'scaled': 2, 'kept': 3, 'target': 4}
        fate_table.sort(key=lambda x: (fate_order.get(x['fate'], 5),
                                       -x['importance_score']))

        return fate_table

    def get_summary(self, fate_table):
        """Get a summary of column fates."""
        fates = [f['fate'] for f in fate_table]
        return {
            'total_columns': len(fate_table),
            'kept': fates.count('kept'),
            'dropped': fates.count('dropped'),
            'encoded': fates.count('encoded'),
            'scaled': fates.count('scaled'),
            'target': fates.count('target'),
            'restorable_count': sum(1 for f in fate_table if f['restorable']),
        }

    # ── Internal Helpers ─────────────────────────────────────

    def _compute_importance(self, df, target_col):
        """Compute mutual information importance for each column."""
        scores = {}

        if not target_col or target_col not in df.columns:
            return scores

        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

            target = df[target_col].copy()
            numeric_cols = df.select_dtypes(include='number').columns.tolist()

            if target_col in numeric_cols:
                numeric_cols.remove(target_col)

            if not numeric_cols:
                return scores

            X = df[numeric_cols].fillna(df[numeric_cols].median())
            y = target.fillna(target.mode()[0] if not target.mode().empty else 0)

            # Determine if classification or regression
            if y.dtype == 'object' or y.nunique() <= 20:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_encoded = le.fit_transform(y.astype(str))
                mi = mutual_info_classif(X, y_encoded, random_state=42)
            else:
                mi = mutual_info_regression(X, y, random_state=42)

            # Normalize
            max_mi = max(mi) if max(mi) > 0 else 1
            for col_name, score in zip(numeric_cols, mi):
                scores[col_name] = round(float(score / max_mi), 4)

            # For categorical columns, use a simple heuristic
            cat_cols = df.select_dtypes(include=['object', 'category']).columns
            for col_name in cat_cols:
                if col_name == target_col:
                    continue
                # Use number of unique values as a proxy
                nunique = df[col_name].nunique()
                if nunique <= 1:
                    scores[col_name] = 0.0
                elif nunique <= 20:
                    scores[col_name] = 0.3  # moderate default for low-cardinality
                else:
                    scores[col_name] = 0.1  # low default for high-cardinality

        except Exception:
            pass

        return scores

    def _extract_dropped(self, report, step_name):
        """Extract dropped columns from a report step."""
        if not report:
            return {}

        dropped = {}
        for step in report.get('steps', []):
            if step.get('name') == step_name:
                for col in step.get('columns', []):
                    dropped[col] = step.get('description', 'Dropped')
        return dropped

    def _extract_encoded(self, report):
        """Extract encoding info from transform report."""
        if not report:
            return {}

        encoded = {}
        for step in report.get('steps', []):
            if step.get('name') == 'Encode Categoricals':
                for col, desc in step.get('encodings', {}).items():
                    encoded[col] = desc
        return encoded

    def _extract_scaled(self, report):
        """Extract scaled columns info."""
        if not report:
            return set()

        for step in report.get('steps', []):
            if step.get('name') == 'Scale Features':
                return set()  # All numeric cols are scaled, tracked at column level
        return set()
