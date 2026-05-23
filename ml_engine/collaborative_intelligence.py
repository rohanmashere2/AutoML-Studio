"""
AutoML Studio — Collaborative Intelligence Network (Feature #8)
Stores anonymised dataset meta-features and best-model outcomes to build
a shared knowledge base. New datasets benefit from past experiments.
"""

import json
import os
import numpy as np
from pathlib import Path


class CollaborativeIntelligence:
    """Shared meta-knowledge network across experiments."""

    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'meta_knowledge.json'
        )
        self.knowledge = self._load()

    def contribute(self, dataset_dna, best_model, best_score, problem_type,
                    preprocessing_summary=None):
        """
        Contribute anonymised meta-knowledge from a completed experiment.
        Only meta-features are stored — no actual data values.
        """
        entry = {
            'dna': _sanitise_dna(dataset_dna),
            'best_model': best_model,
            'best_score': round(float(best_score), 4),
            'problem_type': problem_type,
            'preprocessing': preprocessing_summary or {},
            'timestamp': str(np.datetime64('now')),
        }
        self.knowledge.append(entry)
        self._save()
        return {'status': 'contributed', 'total_experiments': len(self.knowledge)}

    def recommend(self, dataset_dna, problem_type, top_k=5):
        """
        Find similar past experiments and recommend models/settings.
        """
        if not self.knowledge:
            return {'recommendations': [], 'message': 'No past experiments in knowledge base.'}

        # Filter by problem type
        relevant = [k for k in self.knowledge if k.get('problem_type') == problem_type]
        if not relevant:
            return {'recommendations': [], 'message': f'No past {problem_type} experiments.'}

        # Compute similarity
        scored = []
        for entry in relevant:
            sim = _dna_similarity(dataset_dna, entry.get('dna', {}))
            scored.append((entry, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        recommendations = []
        model_votes = {}
        for entry, sim in top:
            model = entry.get('best_model', 'Unknown')
            model_votes[model] = model_votes.get(model, 0) + sim
            recommendations.append({
                'similarity': round(sim, 4),
                'best_model': model,
                'best_score': entry.get('best_score', 0),
                'n_rows': entry['dna'].get('n_rows', 0),
                'n_features': entry['dna'].get('n_features', 0),
            })

        # Most recommended model
        if model_votes:
            recommended_model = max(model_votes, key=model_votes.get)
            confidence = round(model_votes[recommended_model] / max(sum(model_votes.values()), 1), 2)
        else:
            recommended_model = 'Random Forest'
            confidence = 0.5

        avg_score = np.mean([r['best_score'] for r in recommendations]) if recommendations else 0

        return {
            'recommendations': recommendations,
            'recommended_model': recommended_model,
            'confidence': confidence,
            'estimated_score': round(float(avg_score), 4),
            'total_experiments_in_db': len(self.knowledge),
            'relevant_experiments': len(relevant),
            'message': (
                f'Based on {len(relevant)} past experiments, {recommended_model} '
                f'is recommended (confidence: {confidence:.0%}). '
                f'Estimated score: ~{avg_score:.1%}.'
            ),
        }

    def get_stats(self):
        """Return knowledge base statistics."""
        if not self.knowledge:
            return {'total': 0}

        models = [k.get('best_model', '') for k in self.knowledge]
        model_counts = {}
        for m in models:
            model_counts[m] = model_counts.get(m, 0) + 1

        return {
            'total_experiments': len(self.knowledge),
            'model_distribution': model_counts,
            'avg_score': round(float(np.mean([k.get('best_score', 0) for k in self.knowledge])), 4),
        }

    def _load(self):
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with open(self.db_path, 'w') as f:
                json.dump(self.knowledge, f, indent=2, default=str)
        except Exception:
            pass


def _sanitise_dna(dna):
    """Keep only safe meta-features — no actual data values."""
    safe_keys = [
        'n_rows', 'n_features', 'n_numeric', 'n_categorical', 'dimensionality_ratio',
        'missing_pct', 'duplicate_pct', 'mean_skewness', 'mean_kurtosis',
        'mean_abs_correlation', 'outlier_density', 'n_classes', 'class_entropy',
        'imbalance_ratio', 'mean_cardinality', 'categorical_ratio',
        'target_skewness', 'target_cv',
    ]
    return {k: v for k, v in dna.items() if k in safe_keys}


def _dna_similarity(dna_a, dna_b):
    """Compute similarity between two dataset DNAs."""
    common_keys = set(dna_a.keys()) & set(dna_b.keys())
    if not common_keys:
        return 0.0

    similarities = []
    for key in common_keys:
        va, vb = dna_a[key], dna_b[key]
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            max_val = max(abs(va), abs(vb), 1e-8)
            sim = 1 - abs(va - vb) / max_val
            similarities.append(max(0, sim))

    return float(np.mean(similarities)) if similarities else 0.0
