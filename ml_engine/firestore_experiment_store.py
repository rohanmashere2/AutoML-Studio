"""
AutoML Studio - Firestore Experiment Store Backend
Cloud-native experiment tracking using Firebase Firestore.
Falls back to SQLite when Firestore is not configured.

Firestore structure:
    users/{user_id}/experiments/{exp_id}         — experiment doc
    users/{user_id}/experiments/{exp_id}/results  — step results subcollection
    users/{user_id}/experiments/{exp_id}/models   — model results subcollection
"""

import json
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

_firestore_db = None
_firestore_available = False


def _get_firestore():
    """Lazy-init Firestore client via Firebase Admin SDK."""
    global _firestore_db, _firestore_available
    if _firestore_db is not None:
        return _firestore_db

    try:
        import firebase_admin
        from firebase_admin import firestore

        # Ensure Firebase app is initialized
        try:
            firebase_admin.get_app()
        except ValueError:
            # Not initialized yet — let auth.py handle it
            from auth import _get_firebase_app
            if _get_firebase_app() is None:
                _firestore_available = False
                return None

        _firestore_db = firestore.client()
        _firestore_available = True
        logger.info("Firestore client initialized successfully")
        return _firestore_db
    except Exception as e:
        logger.warning(f"Firestore init failed: {e}. Using SQLite fallback.")
        _firestore_available = False
        return None


def is_firestore_available():
    """Check if Firestore backend is available."""
    _get_firestore()
    return _firestore_available


class FirestoreExperimentStore:
    """Firestore-backed experiment storage with the same API as SQLite ExperimentStore."""

    def __init__(self):
        self.db = _get_firestore()

    def _exp_ref(self, user_id, exp_id):
        """Get reference to an experiment document."""
        return self.db.collection('users').document(user_id).collection('experiments').document(exp_id)

    def _exp_collection(self, user_id):
        """Get reference to a user's experiments collection."""
        return self.db.collection('users').document(user_id).collection('experiments')

    # ── Create ────────────────────────────────────────────────

    def create_experiment(self, name=None, description='', dataset_name='',
                          target_column='', problem_type='', n_rows=0, n_cols=0,
                          session_id=None, tags=None, user_id=None, problem_statement=''):
        """Create a new experiment record in Firestore."""
        exp_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat()
        user_id = user_id or 'anonymous'

        if name is None:
            name = f'Experiment_{now[:10]}_{exp_id[:6]}'

        doc = {
            'id': exp_id,
            'name': name,
            'description': description,
            'tags': tags or [],
            'dataset_name': dataset_name,
            'target_column': target_column,
            'problem_type': problem_type,
            'n_rows': n_rows,
            'n_cols': n_cols,
            'status': 'created',
            'created_at': now,
            'updated_at': now,
            'best_model': '',
            'best_score': None,
            'primary_metric': '',
            'session_id': session_id,
            'notes': '',
            'user_id': user_id,
            'problem_statement': problem_statement or '',
        }

        self._exp_ref(user_id, exp_id).set(doc)
        return exp_id

    def update_experiment(self, exp_id, **kwargs):
        """Update experiment fields in Firestore."""
        allowed_fields = {
            'name', 'description', 'tags', 'status', 'best_model',
            'best_score', 'primary_metric', 'notes', 'target_column',
            'problem_type', 'problem_statement',
        }
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return

        updates['updated_at'] = datetime.utcnow().isoformat()

        # Find the experiment across users if user_id not known
        user_id = self._find_experiment_owner(exp_id)
        if user_id:
            self._exp_ref(user_id, exp_id).update(updates)

    def save_step_result(self, exp_id, step, result_data):
        """Save a pipeline step result as a subcollection doc."""
        now = datetime.utcnow().isoformat()

        if isinstance(result_data, dict):
            data_str = json.dumps(result_data, default=str)
        else:
            data_str = str(result_data)

        user_id = self._find_experiment_owner(exp_id)
        if user_id:
            self._exp_ref(user_id, exp_id).collection('results').add({
                'step': step,
                'result_data': data_str,
                'created_at': now,
            })

    def save_model_result(self, exp_id, model_name, model_type, primary_metric,
                          metrics, is_best=False, hyperparameters=None,
                          feature_importance=None):
        """Save a trained model result."""
        now = datetime.utcnow().isoformat()

        user_id = self._find_experiment_owner(exp_id)
        if user_id:
            self._exp_ref(user_id, exp_id).collection('models').add({
                'model_name': model_name,
                'model_type': model_type,
                'primary_metric': primary_metric,
                'metrics': json.dumps(metrics, default=str),
                'is_best': is_best,
                'created_at': now,
                'hyperparameters': json.dumps(hyperparameters or {}, default=str),
                'feature_importance': json.dumps(feature_importance or [], default=str),
            })

    # ── Fingerprints ─────────────────────────────────────────

    def save_fingerprint(self, exp_id, fingerprint):
        """Save a dataset fingerprint."""
        user_id = self._find_experiment_owner(exp_id)
        if user_id:
            self._exp_ref(user_id, exp_id).update({
                'fingerprint': json.dumps(fingerprint, default=str),
                'fingerprint_at': datetime.utcnow().isoformat(),
            })

    def get_all_fingerprints(self):
        """Get all stored fingerprints — scans all users (admin operation)."""
        results = []
        try:
            users = self.db.collection('users').stream()
            for user_doc in users:
                exps = self.db.collection('users').document(user_doc.id).collection('experiments').stream()
                for exp_doc in exps:
                    data = exp_doc.to_dict()
                    if 'fingerprint' in data:
                        results.append((data['id'], data['fingerprint']))
        except Exception as e:
            logger.error(f"Error fetching fingerprints: {e}")
        return results

    # ── Read Operations ──────────────────────────────────────

    def get_experiment(self, exp_id, user_id=None):
        """Get a single experiment with its results."""
        if user_id:
            doc = self._exp_ref(user_id, exp_id).get()
            if not doc.exists:
                return None
            exp = doc.to_dict()
        else:
            # Search across all users (fallback)
            exp = self._find_experiment_doc(exp_id)
            if not exp:
                return None
            user_id = exp.get('user_id', 'anonymous')

        exp['tags'] = exp.get('tags', [])

        # Get step results
        results_ref = self._exp_ref(user_id, exp_id).collection('results')
        results = results_ref.order_by('created_at').stream()
        exp['step_results'] = []
        for r in results:
            rd = r.to_dict()
            try:
                data = json.loads(rd.get('result_data', '{}'))
            except (json.JSONDecodeError, TypeError):
                data = rd.get('result_data', {})
            exp['step_results'].append({
                'step': rd['step'],
                'data': data,
                'timestamp': rd['created_at'],
            })

        # Get model results
        models_ref = self._exp_ref(user_id, exp_id).collection('models')
        models = models_ref.order_by('primary_metric', direction='DESCENDING').stream()
        exp['models'] = []
        for m in models:
            md = m.to_dict()
            exp['models'].append({
                'model_name': md['model_name'],
                'model_type': md['model_type'],
                'primary_metric': md['primary_metric'],
                'metrics': json.loads(md.get('metrics', '{}')),
                'is_best': bool(md.get('is_best')),
                'hyperparameters': json.loads(md.get('hyperparameters', '{}')),
                'feature_importance': json.loads(md.get('feature_importance', '[]')),
            })

        return exp

    def get_experiment_by_session_id(self, session_id, user_id=None):
        """Get the most recent experiment by session_id."""
        if not user_id:
            return None

        query = (self._exp_collection(user_id)
                 .where('session_id', '==', session_id)
                 .order_by('created_at', direction='DESCENDING')
                 .limit(1))

        docs = list(query.stream())
        if not docs:
            return None

        exp = docs[0].to_dict()
        exp['tags'] = exp.get('tags', [])
        return exp

    def list_experiments(self, limit=50, offset=0, search=None, tag=None,
                         sort_by='created_at', sort_order='desc', user_id=None):
        """List experiments with optional filtering."""
        if not user_id:
            user_id = 'anonymous'

        from google.cloud.firestore_v1 import Query

        direction = Query.DESCENDING if sort_order.lower() == 'desc' else Query.ASCENDING
        allowed_sort = {'created_at', 'updated_at', 'best_score', 'name', 'dataset_name'}
        if sort_by not in allowed_sort:
            sort_by = 'created_at'

        query = self._exp_collection(user_id).order_by(sort_by, direction=direction)

        if tag:
            query = query.where('tags', 'array_contains', tag)

        # Firestore doesn't support LIKE queries, so we fetch and filter in-memory for search
        all_docs = list(query.stream())

        # Apply search filter in-memory
        if search:
            search_lower = search.lower()
            all_docs = [
                d for d in all_docs
                if search_lower in (d.to_dict().get('name', '') or '').lower()
                or search_lower in (d.to_dict().get('description', '') or '').lower()
                or search_lower in (d.to_dict().get('dataset_name', '') or '').lower()
            ]

        total = len(all_docs)

        # Apply pagination
        paginated = all_docs[offset:offset + limit]

        experiments = []
        for doc in paginated:
            exp = doc.to_dict()
            exp['tags'] = exp.get('tags', [])
            experiments.append(exp)

        return {
            'experiments': experiments,
            'total': total,
            'limit': limit,
            'offset': offset,
        }

    # ── Comparison ────────────────────────────────────────────

    def compare_experiments(self, exp_ids):
        """Compare multiple experiments with structured diff."""
        experiments = []
        for eid in exp_ids:
            exp = self.get_experiment(eid)
            if exp:
                experiments.append(exp)

        if len(experiments) < 2:
            return {'error': 'Need at least 2 experiments to compare'}

        comparison = {
            'experiments': [],
            'metrics_comparison': [],
            'feature_comparison': [],
        }

        for exp in experiments:
            summary = {
                'id': exp['id'],
                'name': exp['name'],
                'dataset': exp['dataset_name'],
                'target': exp['target_column'],
                'problem_type': exp['problem_type'],
                'best_model': exp.get('best_model'),
                'best_score': exp.get('best_score'),
                'n_rows': exp.get('n_rows'),
                'n_cols': exp.get('n_cols'),
                'created_at': exp.get('created_at'),
            }
            comparison['experiments'].append(summary)

            for model in exp.get('models', []):
                comparison['metrics_comparison'].append({
                    'experiment': exp['name'],
                    'experiment_id': exp['id'],
                    'model': model['model_name'],
                    'score': model['primary_metric'],
                    'is_best': model['is_best'],
                    'metrics': model.get('metrics', {}),
                    'hyperparameters': model.get('hyperparameters', {}),
                })

        comparison['structured_diff'] = self._build_structured_diff(experiments)
        return comparison

    def _build_structured_diff(self, experiments):
        """Build a structured diff between experiments."""
        if len(experiments) < 2:
            return {}

        diff = {
            'metric_deltas': [],
            'hyperparam_diff': [],
            'feature_importance_diff': [],
            'summary': '',
        }

        exp_scores = {}
        for exp in experiments:
            best_model = None
            for m in exp.get('models', []):
                if m.get('is_best'):
                    best_model = m
                    break
            if not best_model and exp.get('models'):
                best_model = exp['models'][0]

            if best_model:
                exp_scores[exp['id']] = {
                    'model': best_model.get('model_name', ''),
                    'score': best_model.get('primary_metric', 0),
                    'metrics': best_model.get('metrics', {}),
                    'hyperparameters': best_model.get('hyperparameters', {}),
                    'feature_importance': best_model.get('feature_importance', []),
                }

        if len(exp_scores) >= 2:
            exp_ids = list(exp_scores.keys())
            e1, e2 = exp_scores[exp_ids[0]], exp_scores[exp_ids[1]]

            all_metric_keys = set(e1.get('metrics', {}).keys()) | set(e2.get('metrics', {}).keys())
            for metric in all_metric_keys:
                v1 = e1.get('metrics', {}).get(metric)
                v2 = e2.get('metrics', {}).get(metric)
                if v1 is not None and v2 is not None:
                    try:
                        v1_f, v2_f = float(v1), float(v2)
                        delta = v2_f - v1_f
                        diff['metric_deltas'].append({
                            'metric': metric,
                            'exp_1': round(v1_f, 4),
                            'exp_2': round(v2_f, 4),
                            'delta': round(delta, 4),
                            'delta_pct': f'{delta/max(abs(v1_f),0.001)*100:+.1f}%',
                            'winner': exp_ids[1] if delta > 0 else exp_ids[0] if delta < 0 else 'tie',
                        })
                    except (ValueError, TypeError):
                        pass

            all_params = set(e1.get('hyperparameters', {}).keys()) | set(e2.get('hyperparameters', {}).keys())
            for param in all_params:
                p1 = e1.get('hyperparameters', {}).get(param)
                p2 = e2.get('hyperparameters', {}).get(param)
                diff['hyperparam_diff'].append({
                    'param': param, 'exp_1': p1, 'exp_2': p2, 'changed': p1 != p2,
                })

            fi1 = {f['feature']: f for f in e1.get('feature_importance', []) if isinstance(f, dict)}
            fi2 = {f['feature']: f for f in e2.get('feature_importance', []) if isinstance(f, dict)}
            for feat in list(set(fi1.keys()) | set(fi2.keys()))[:15]:
                f1, f2 = fi1.get(feat, {}), fi2.get(feat, {})
                diff['feature_importance_diff'].append({
                    'feature': feat,
                    'exp_1_score': f1.get('importance', 0),
                    'exp_2_score': f2.get('importance', 0),
                    'exp_1_rank': list(fi1.keys()).index(feat) + 1 if feat in fi1 else None,
                    'exp_2_rank': list(fi2.keys()).index(feat) + 1 if feat in fi2 else None,
                })

            score_delta = (e2.get('score', 0) or 0) - (e1.get('score', 0) or 0)
            n_changed = sum(1 for p in diff['hyperparam_diff'] if p['changed'])
            diff['summary'] = (
                f"Experiment 2 {'improved' if score_delta > 0 else 'decreased'} score by {abs(score_delta):.4f}. "
                f"{n_changed} hyperparameters changed. "
                f"Best models: {e1.get('model', '?')} vs {e2.get('model', '?')}."
            )

        return diff

    # ── Delete & Stats ───────────────────────────────────────

    def delete_experiment(self, exp_id, user_id=None):
        """Delete an experiment and all its subcollection data."""
        if not user_id:
            user_id = self._find_experiment_owner(exp_id)
        if not user_id:
            return False

        ref = self._exp_ref(user_id, exp_id)

        # Delete subcollections
        for subcol_name in ('results', 'models'):
            for doc in ref.collection(subcol_name).stream():
                doc.reference.delete()

        ref.delete()
        return True

    def get_stats(self, user_id=None):
        """Get overall experiment statistics."""
        if not user_id:
            user_id = 'anonymous'

        docs = list(self._exp_collection(user_id).stream())

        total = len(docs)
        completed = 0
        scores = []
        problem_types = {}

        for doc in docs:
            data = doc.to_dict()
            if data.get('status') == 'complete':
                completed += 1
            if data.get('best_score') is not None:
                scores.append(float(data['best_score']))
            pt = data.get('problem_type')
            if pt:
                problem_types[pt] = problem_types.get(pt, 0) + 1

        avg_score = round(sum(scores) / len(scores), 4) if scores else None

        return {
            'total_experiments': total,
            'completed': completed,
            'avg_best_score': avg_score,
            'problem_types': problem_types,
        }

    # ── Internal Helpers ─────────────────────────────────────

    def _find_experiment_owner(self, exp_id):
        """Find which user owns an experiment by scanning (cached in doc)."""
        try:
            users = self.db.collection('users').stream()
            for user_doc in users:
                doc = self._exp_ref(user_doc.id, exp_id).get()
                if doc.exists:
                    return user_doc.id
        except Exception as e:
            logger.error(f"Error finding experiment owner: {e}")
        return None

    def _find_experiment_doc(self, exp_id):
        """Find an experiment doc across all users."""
        try:
            users = self.db.collection('users').stream()
            for user_doc in users:
                doc = self._exp_ref(user_doc.id, exp_id).get()
                if doc.exists:
                    return doc.to_dict()
        except Exception as e:
            logger.error(f"Error finding experiment: {e}")
        return None
