"""
AutoML Studio — Dataset Fingerprinting & Similarity Search
Computes statistical fingerprints of datasets and finds similar past experiments
to pre-populate recommendations.
"""

import numpy as np
import pandas as pd
import json


class DatasetFingerprinter:
    """Compute and compare dataset fingerprints."""

    def compute_fingerprint(self, df, target_column=None):
        """Generate a feature vector describing the dataset's characteristics."""
        n_rows, n_cols = df.shape
        numeric_cols = df.select_dtypes(include='number').columns
        cat_cols = df.select_dtypes(include='object').columns

        fp = {
            'n_rows': n_rows,
            'n_cols': n_cols,
            'numeric_ratio': round(len(numeric_cols) / max(n_cols, 1), 4),
            'categorical_ratio': round(len(cat_cols) / max(n_cols, 1), 4),
            'avg_missing_pct': round(float(df.isnull().mean().mean() * 100), 2),
            'avg_cardinality_ratio': round(float(df.nunique().mean() / max(n_rows, 1)), 4),
            'duplicated_pct': round(float(df.duplicated().mean() * 100), 2),
        }

        # Numeric summary stats
        if len(numeric_cols) > 0:
            num_df = df[numeric_cols]
            fp['avg_skewness'] = round(float(num_df.skew().abs().mean()), 4)
            fp['avg_kurtosis'] = round(float(num_df.kurtosis().abs().mean()), 4)

            # Average correlation
            if len(numeric_cols) > 1:
                corr = num_df.corr().abs()
                mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
                fp['avg_correlation'] = round(float(corr.where(mask).mean().mean()), 4)
            else:
                fp['avg_correlation'] = 0

        # Target info
        if target_column and target_column in df.columns:
            target = df[target_column]
            if pd.api.types.is_numeric_dtype(target):
                n_unique = target.nunique()
                if n_unique <= 20:
                    fp['target_type'] = 'classification'
                    fp['n_classes'] = n_unique
                    vc = target.value_counts(normalize=True)
                    fp['target_imbalance'] = round(float(vc.max() / max(vc.min(), 0.01)), 2)
                else:
                    fp['target_type'] = 'regression'
                    fp['target_skew'] = round(float(target.skew()), 4)
            else:
                fp['target_type'] = 'classification'
                fp['n_classes'] = target.nunique()
        else:
            fp['target_type'] = 'unknown'

        # Size category
        if n_rows < 500:
            fp['size_category'] = 'tiny'
        elif n_rows < 5000:
            fp['size_category'] = 'small'
        elif n_rows < 50000:
            fp['size_category'] = 'medium'
        else:
            fp['size_category'] = 'large'

        return fp

    def to_vector(self, fingerprint):
        """Convert fingerprint dict to a numeric vector for similarity computation."""
        keys = [
            'n_rows', 'n_cols', 'numeric_ratio', 'categorical_ratio',
            'avg_missing_pct', 'avg_cardinality_ratio',
            'avg_skewness', 'avg_kurtosis', 'avg_correlation',
        ]
        vec = []
        for k in keys:
            v = fingerprint.get(k, 0)
            if isinstance(v, (int, float)):
                vec.append(float(v))
            else:
                vec.append(0.0)

        # Normalize n_rows and n_cols to log scale
        vec[0] = np.log1p(vec[0])
        vec[1] = np.log1p(vec[1])

        return np.array(vec)

    def compute_similarity(self, fp1, fp2):
        """Compute cosine similarity between two fingerprints."""
        v1 = self.to_vector(fp1)
        v2 = self.to_vector(fp2)

        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return round(float(dot / (norm1 * norm2)), 4)

    def find_similar(self, fingerprint, stored_fingerprints, top_k=3):
        """Find the most similar past datasets.

        Args:
            fingerprint: current dataset fingerprint
            stored_fingerprints: list of (experiment_id, fingerprint_dict) tuples

        Returns:
            list of (experiment_id, similarity_score) tuples
        """
        similarities = []
        for exp_id, stored_fp in stored_fingerprints:
            if isinstance(stored_fp, str):
                stored_fp = json.loads(stored_fp)
            score = self.compute_similarity(fingerprint, stored_fp)
            similarities.append((exp_id, score))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_recommended_settings(self, similar_experiments):
        """Pre-populate recommendations from the best past results.

        Args:
            similar_experiments: list of experiment dicts from the store

        Returns:
            dict with recommended algorithms, cleaning, and hyperparams
        """
        if not similar_experiments:
            return {'confidence': 0, 'message': 'No similar past experiments found.'}

        # Aggregate best models
        best_models = []
        best_scores = []
        for exp in similar_experiments:
            if exp.get('best_model'):
                best_models.append(exp['best_model'])
            if exp.get('best_score'):
                best_scores.append(exp['best_score'])

        from collections import Counter
        model_counts = Counter(best_models)
        recommended = [m for m, _ in model_counts.most_common(3)]

        return {
            'recommended_algorithms': recommended,
            'avg_best_score': round(float(np.mean(best_scores)), 4) if best_scores else None,
            'based_on_experiments': len(similar_experiments),
            'confidence': min(0.95, len(similar_experiments) * 0.3),
            'message': f'Based on {len(similar_experiments)} similar past experiments.',
        }
