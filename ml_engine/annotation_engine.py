"""
AutoML Studio — Collaborative Annotation Engine
Allows labeling predictions as correct/incorrect and discovers failure patterns.
"""

import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime


class AnnotationEngine:
    """Collaborative annotation of predictions with failure pattern discovery."""

    def __init__(self, db_path):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        """Create annotations table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS annotations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    row_index INTEGER,
                    label TEXT,
                    notes TEXT DEFAULT '',
                    created_at TEXT,
                    features_json TEXT DEFAULT '{}'
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_annot_exp ON annotations(experiment_id)')

    def annotate(self, experiment_id, row_index, label, notes='', features=None):
        """Store a user annotation for a prediction."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO annotations (experiment_id, row_index, label, notes, created_at, features_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id, row_index, label, notes,
                datetime.utcnow().isoformat(),
                json.dumps(features or {}, default=str)
            ))
        return {'success': True}

    def get_annotations(self, experiment_id):
        """Get all annotations for an experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                'SELECT * FROM annotations WHERE experiment_id = ? ORDER BY created_at DESC',
                (experiment_id,)
            ).fetchall()

        annotations = []
        for r in rows:
            annotations.append({
                'id': r['id'],
                'row_index': r['row_index'],
                'label': r['label'],
                'notes': r['notes'],
                'created_at': r['created_at'],
                'features': json.loads(r['features_json']) if r['features_json'] else {},
            })

        # Summary
        labels = [a['label'] for a in annotations]
        return {
            'annotations': annotations,
            'total': len(annotations),
            'correct': labels.count('correct'),
            'incorrect': labels.count('incorrect'),
            'uncertain': labels.count('uncertain'),
            'accuracy': round(labels.count('correct') / max(len(labels), 1) * 100, 1),
        }

    def find_failure_patterns(self, experiment_id, model, X_test, feature_names):
        """
        Analyze incorrect predictions to find systematic failure patterns.

        Returns:
            dict with patterns and retraining suggestions
        """
        annots = self.get_annotations(experiment_id)
        incorrect = [a for a in annots['annotations'] if a['label'] == 'incorrect']

        if len(incorrect) < 3:
            return {
                'message': 'Not enough incorrect annotations to find patterns (need at least 3).',
                'patterns': [],
            }

        # Get row indices
        incorrect_indices = [a['row_index'] for a in incorrect]

        if isinstance(X_test, pd.DataFrame):
            valid_indices = [i for i in incorrect_indices if i < len(X_test)]
            if not valid_indices:
                return {'message': 'No valid row indices found.', 'patterns': []}

            incorrect_data = X_test.iloc[valid_indices]
            all_data = X_test
        else:
            return {'message': 'X_test must be a DataFrame.', 'patterns': []}

        patterns = []

        # For each feature, check if incorrect predictions cluster around certain values
        for feat in feature_names[:15]:
            if feat not in all_data.columns:
                continue

            if pd.api.types.is_numeric_dtype(all_data[feat]):
                inc_mean = incorrect_data[feat].mean()
                all_mean = all_data[feat].mean()
                inc_std = incorrect_data[feat].std()
                all_std = all_data[feat].std()

                # Check if the mean of incorrect predictions differs significantly
                if all_std > 0:
                    z_score = abs(inc_mean - all_mean) / all_std
                    if z_score > 1.5:
                        direction = 'higher' if inc_mean > all_mean else 'lower'
                        patterns.append({
                            'feature': feat,
                            'type': 'value_cluster',
                            'description': (
                                f'Model failures have {direction} "{feat}" than average '
                                f'(mean={inc_mean:.2f} vs overall={all_mean:.2f}). '
                                f'The model struggles when "{feat}" is {"high" if direction == "higher" else "low"}.'
                            ),
                            'strength': round(z_score, 2),
                            'incorrect_mean': round(inc_mean, 4),
                            'overall_mean': round(all_mean, 4),
                        })
            else:
                # Categorical feature — check if failures cluster in certain categories
                inc_dist = incorrect_data[feat].value_counts(normalize=True)
                all_dist = all_data[feat].value_counts(normalize=True)

                for val in inc_dist.index[:5]:
                    inc_pct = inc_dist.get(val, 0)
                    all_pct = all_dist.get(val, 0)
                    if inc_pct > all_pct * 1.5 and inc_pct > 0.2:
                        patterns.append({
                            'feature': feat,
                            'type': 'category_cluster',
                            'value': str(val),
                            'description': (
                                f'{inc_pct*100:.0f}% of failures have "{feat}"="{val}" '
                                f'(vs {all_pct*100:.0f}% in overall data). '
                                f'The model consistently fails for this category.'
                            ),
                            'strength': round(inc_pct / max(all_pct, 0.01), 2),
                        })

        patterns.sort(key=lambda x: x.get('strength', 0), reverse=True)

        suggestions = []
        if len(patterns) > 0:
            top = patterns[0]
            suggestions.append({
                'action': f'Collect more training data where {top["feature"]} has extreme values.',
                'priority': 'high',
            })
            suggestions.append({
                'action': f'Try creating interaction features involving {top["feature"]}.',
                'priority': 'medium',
            })

        return {
            'n_incorrect': len(incorrect),
            'patterns': patterns[:10],
            'retraining_suggestions': suggestions,
        }
