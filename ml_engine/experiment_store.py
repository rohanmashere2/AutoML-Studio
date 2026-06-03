"""
AutoML Studio - Experiment Store (v3)
Auto-selects Firestore (cloud) or SQLite (local fallback) backend.
SQLite backend includes structured diffing, hyperparameter storage,
feature importance tracking, and fingerprint support.
"""

import os
import json
import sqlite3
import uuid
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


DB_NAME = 'automl_experiments.db'


def ExperimentStore(db_path=None):
    """Factory: returns Firestore backend if available, else SQLite fallback."""
    try:
        from ml_engine.firestore_experiment_store import is_firestore_available, FirestoreExperimentStore
        if is_firestore_available():
            logger.info("Using Firestore experiment store (cloud)")
            return FirestoreExperimentStore()
    except Exception as e:
        logger.debug(f"Firestore check failed: {e}")

    logger.info("Using SQLite experiment store (local fallback)")
    return _SQLiteExperimentStore(db_path=db_path)


class _SQLiteExperimentStore:
    """SQLite-backed experiment storage (local fallback)."""

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), DB_NAME)
        self.db_path = db_path
        self._init_db()
        self._migrate_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._connect() as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    description TEXT,
                    tags TEXT DEFAULT '[]',
                    dataset_name TEXT,
                    target_column TEXT,
                    problem_type TEXT,
                    n_rows INTEGER,
                    n_cols INTEGER,
                    status TEXT DEFAULT 'created',
                    created_at TEXT,
                    updated_at TEXT,
                    best_model TEXT,
                    best_score REAL,
                    primary_metric TEXT,
                    session_id TEXT,
                    notes TEXT DEFAULT '',
                    user_id TEXT DEFAULT 'anonymous',
                    problem_statement TEXT DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS experiment_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    step TEXT,
                    result_data TEXT,
                    created_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS experiment_models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT,
                    model_name TEXT,
                    model_type TEXT,
                    primary_metric REAL,
                    metrics TEXT,
                    is_best INTEGER DEFAULT 0,
                    created_at TEXT,
                    hyperparameters TEXT DEFAULT '{}',
                    feature_importance TEXT DEFAULT '[]',
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE TABLE IF NOT EXISTS fingerprints (
                    experiment_id TEXT PRIMARY KEY,
                    fingerprint TEXT,
                    created_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);
                CREATE INDEX IF NOT EXISTS idx_exp_dataset ON experiments(dataset_name);
                CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
                CREATE INDEX IF NOT EXISTS idx_exp_user ON experiments(user_id);
                CREATE INDEX IF NOT EXISTS idx_results_exp ON experiment_results(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_models_exp ON experiment_models(experiment_id);
            ''')

    def _migrate_db(self):
        """Add new columns to existing tables if they don't exist."""
        migrations = [
            ('experiment_models', 'hyperparameters', "TEXT DEFAULT '{}'"),
            ('experiment_models', 'feature_importance', "TEXT DEFAULT '[]'"),
            ('experiments', 'user_id', "TEXT DEFAULT 'anonymous'"),
            ('experiments', 'problem_statement', "TEXT DEFAULT ''"),
        ]
        with self._connect() as conn:
            for table, column, col_type in migrations:
                try:
                    conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {col_type}')
                except sqlite3.OperationalError:
                    pass  # Column already exists

            # Backfill existing rows that have NULL user_id
            conn.execute("UPDATE experiments SET user_id = 'anonymous' WHERE user_id IS NULL")

            # Create fingerprints table if not exists
            conn.execute('''
                CREATE TABLE IF NOT EXISTS fingerprints (
                    experiment_id TEXT PRIMARY KEY,
                    fingerprint TEXT,
                    created_at TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            ''')

            # Create index on user_id if not exists
            try:
                conn.execute('CREATE INDEX IF NOT EXISTS idx_exp_user ON experiments(user_id)')
            except sqlite3.OperationalError:
                pass

    def _connect(self):
        """Get a database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_experiment(self, name=None, description='', dataset_name='',
                          target_column='', problem_type='', n_rows=0, n_cols=0,
                          session_id=None, tags=None, user_id=None, problem_statement=''):
        """Create a new experiment record."""
        exp_id = str(uuid.uuid4())[:12]
        now = datetime.utcnow().isoformat()
        user_id = user_id or 'anonymous'

        if name is None:
            name = f'Experiment_{now[:10]}_{exp_id[:6]}'

        with self._connect() as conn:
            conn.execute('''
                INSERT INTO experiments (id, name, description, tags, dataset_name,
                    target_column, problem_type, n_rows, n_cols, status,
                    created_at, updated_at, session_id, user_id, problem_statement)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'created', ?, ?, ?, ?, ?)
            ''', (exp_id, name, description, json.dumps(tags or []),
                  dataset_name, target_column, problem_type, n_rows, n_cols,
                  now, now, session_id, user_id, problem_statement or ''))

        return exp_id

    def update_experiment(self, exp_id, **kwargs):
        """Update experiment fields."""
        allowed_fields = {
            'name', 'description', 'tags', 'status', 'best_model',
            'best_score', 'primary_metric', 'notes', 'target_column',
            'problem_type', 'problem_statement',
        }

        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        if not updates:
            return

        updates['updated_at'] = datetime.utcnow().isoformat()

        if 'tags' in updates and isinstance(updates['tags'], list):
            updates['tags'] = json.dumps(updates['tags'])

        set_clause = ', '.join(f'{k} = ?' for k in updates.keys())
        values = list(updates.values()) + [exp_id]

        with self._connect() as conn:
            conn.execute(
                f'UPDATE experiments SET {set_clause} WHERE id = ?',
                values
            )

    def save_step_result(self, exp_id, step, result_data):
        """Save a pipeline step result."""
        now = datetime.utcnow().isoformat()

        # Serialize result data
        if isinstance(result_data, dict):
            data_str = json.dumps(result_data, default=str)
        else:
            data_str = str(result_data)

        with self._connect() as conn:
            conn.execute('''
                INSERT INTO experiment_results (experiment_id, step, result_data, created_at)
                VALUES (?, ?, ?, ?)
            ''', (exp_id, step, data_str, now))

    def save_model_result(self, exp_id, model_name, model_type, primary_metric,
                          metrics, is_best=False, hyperparameters=None,
                          feature_importance=None):
        """Save a trained model result with hyperparameters and feature importance."""
        now = datetime.utcnow().isoformat()

        with self._connect() as conn:
            conn.execute('''
                INSERT INTO experiment_models
                (experiment_id, model_name, model_type, primary_metric, metrics,
                 is_best, created_at, hyperparameters, feature_importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (exp_id, model_name, model_type, primary_metric,
                  json.dumps(metrics, default=str), 1 if is_best else 0, now,
                  json.dumps(hyperparameters or {}, default=str),
                  json.dumps(feature_importance or [], default=str)))

    # ── Fingerprints ─────────────────────────────────────────

    def save_fingerprint(self, exp_id, fingerprint):
        """Save a dataset fingerprint for an experiment."""
        with self._connect() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO fingerprints (experiment_id, fingerprint, created_at)
                VALUES (?, ?, ?)
            ''', (exp_id, json.dumps(fingerprint, default=str),
                  datetime.utcnow().isoformat()))

    def get_all_fingerprints(self):
        """Get all stored fingerprints for similarity search."""
        with self._connect() as conn:
            rows = conn.execute('SELECT experiment_id, fingerprint FROM fingerprints').fetchall()
        return [(r['experiment_id'], r['fingerprint']) for r in rows]

    # ── Read Operations ──────────────────────────────────────

    def get_experiment(self, exp_id, user_id=None):
        """Get a single experiment with its results."""
        with self._connect() as conn:
            if user_id:
                row = conn.execute(
                    'SELECT * FROM experiments WHERE id = ? AND user_id = ?', (exp_id, user_id)
                ).fetchone()
            else:
                row = conn.execute(
                    'SELECT * FROM experiments WHERE id = ?', (exp_id,)
                ).fetchone()

            if not row:
                return None

            exp = dict(row)
            exp['tags'] = json.loads(exp.get('tags', '[]'))

            # Get step results
            results = conn.execute(
                'SELECT step, result_data, created_at FROM experiment_results WHERE experiment_id = ? ORDER BY created_at',
                (exp_id,)
            ).fetchall()
            exp['step_results'] = [
                {'step': r['step'], 'data': json.loads(r['result_data']), 'timestamp': r['created_at']}
                for r in results
            ]

            # Get model results
            models = conn.execute(
                'SELECT * FROM experiment_models WHERE experiment_id = ? ORDER BY primary_metric DESC',
                (exp_id,)
            ).fetchall()
            exp['models'] = [
                {
                    'model_name': m['model_name'],
                    'model_type': m['model_type'],
                    'primary_metric': m['primary_metric'],
                    'metrics': json.loads(m['metrics']),
                    'is_best': bool(m['is_best']),
                    'hyperparameters': json.loads(m['hyperparameters'] or '{}'),
                    'feature_importance': json.loads(m['feature_importance'] or '[]'),
                }
                for m in models
            ]

            return exp

    def get_experiment_by_session_id(self, session_id, user_id=None):
        """Get the most recent experiment associated with a session id."""
        with self._connect() as conn:
            if user_id:
                row = conn.execute(
                    'SELECT * FROM experiments WHERE session_id = ? AND user_id = ? ORDER BY created_at DESC LIMIT 1',
                    (session_id, user_id)
                ).fetchone()
            else:
                row = conn.execute(
                    'SELECT * FROM experiments WHERE session_id = ? ORDER BY created_at DESC LIMIT 1',
                    (session_id,)
                ).fetchone()

        if not row:
            return None

        exp = dict(row)
        exp['tags'] = json.loads(exp.get('tags', '[]'))

        return exp

    def list_experiments(self, limit=50, offset=0, search=None, tag=None,
                         sort_by='created_at', sort_order='desc', user_id=None):
        """List experiments with optional filtering."""
        query = 'SELECT * FROM experiments'
        params = []
        conditions = []

        if user_id:
            conditions.append('user_id = ?')
            params.append(user_id)

        if search:
            conditions.append('(name LIKE ? OR description LIKE ? OR dataset_name LIKE ?)')
            params.extend([f'%{search}%'] * 3)

        if tag:
            conditions.append('tags LIKE ?')
            params.append(f'%"{tag}"%')

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)

        allowed_sort = {'created_at', 'updated_at', 'best_score', 'name', 'dataset_name'}
        if sort_by not in allowed_sort:
            sort_by = 'created_at'
        sort_dir = 'DESC' if sort_order.lower() == 'desc' else 'ASC'
        query += f' ORDER BY {sort_by} {sort_dir}'

        query += ' LIMIT ? OFFSET ?'
        params.extend([limit, offset])

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

            # Get total count
            count_query = 'SELECT COUNT(*) as cnt FROM experiments'
            if conditions:
                count_query += ' WHERE ' + ' AND '.join(conditions)
            total = conn.execute(count_query, params[:-2] if params else []).fetchone()['cnt']

        experiments = []
        for row in rows:
            exp = dict(row)
            exp['tags'] = json.loads(exp.get('tags', '[]'))
            experiments.append(exp)

        return {
            'experiments': experiments,
            'total': total,
            'limit': limit,
            'offset': offset,
        }

    # ── Structured Comparison ────────────────────────────────

    def compare_experiments(self, exp_ids):
        """Compare multiple experiments with structured diff."""
        experiments = []
        for eid in exp_ids:
            exp = self.get_experiment(eid)
            if exp:
                experiments.append(exp)

        if len(experiments) < 2:
            return {'error': 'Need at least 2 experiments to compare'}

        # Build basic comparison
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

            # Metrics from models
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

        # Build structured diff
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

        # Compare best models' metrics
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

            # Metric deltas
            all_metric_keys = set(e1.get('metrics', {}).keys()) | set(e2.get('metrics', {}).keys())
            for metric in all_metric_keys:
                v1 = e1.get('metrics', {}).get(metric)
                v2 = e2.get('metrics', {}).get(metric)
                if v1 is not None and v2 is not None:
                    try:
                        v1_f = float(v1)
                        v2_f = float(v2)
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

            # Hyperparameter diff
            all_params = set(e1.get('hyperparameters', {}).keys()) | set(e2.get('hyperparameters', {}).keys())
            for param in all_params:
                p1 = e1.get('hyperparameters', {}).get(param)
                p2 = e2.get('hyperparameters', {}).get(param)
                diff['hyperparam_diff'].append({
                    'param': param,
                    'exp_1': p1,
                    'exp_2': p2,
                    'changed': p1 != p2,
                })

            # Feature importance diff
            fi1 = {f['feature']: f for f in e1.get('feature_importance', []) if isinstance(f, dict)}
            fi2 = {f['feature']: f for f in e2.get('feature_importance', []) if isinstance(f, dict)}
            all_features = set(fi1.keys()) | set(fi2.keys())
            fi_entries = list(all_features)[:15]

            for feat in fi_entries:
                f1 = fi1.get(feat, {})
                f2 = fi2.get(feat, {})
                diff['feature_importance_diff'].append({
                    'feature': feat,
                    'exp_1_score': f1.get('importance', 0),
                    'exp_2_score': f2.get('importance', 0),
                    'exp_1_rank': list(fi1.keys()).index(feat) + 1 if feat in fi1 else None,
                    'exp_2_rank': list(fi2.keys()).index(feat) + 1 if feat in fi2 else None,
                })

            # Summary text
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
        """Delete an experiment and all its data."""
        with self._connect() as conn:
            # Verify ownership if user_id is provided
            if user_id:
                row = conn.execute(
                    'SELECT id FROM experiments WHERE id = ? AND user_id = ?', (exp_id, user_id)
                ).fetchone()
                if not row:
                    return False

            conn.execute('DELETE FROM experiment_results WHERE experiment_id = ?', (exp_id,))
            conn.execute('DELETE FROM experiment_models WHERE experiment_id = ?', (exp_id,))
            conn.execute('DELETE FROM fingerprints WHERE experiment_id = ?', (exp_id,))
            conn.execute('DELETE FROM experiments WHERE id = ?', (exp_id,))
        return True

    def get_stats(self, user_id=None):
        """Get overall experiment statistics."""
        user_filter = ' WHERE user_id = ?' if user_id else ''
        user_params = [user_id] if user_id else []

        with self._connect() as conn:
            total = conn.execute(
                f'SELECT COUNT(*) as cnt FROM experiments{user_filter}', user_params
            ).fetchone()['cnt']
            completed = conn.execute(
                f"SELECT COUNT(*) as cnt FROM experiments WHERE status = 'complete'" +
                (' AND user_id = ?' if user_id else ''),
                user_params
            ).fetchone()['cnt']
            avg_score = conn.execute(
                f'SELECT AVG(best_score) as avg FROM experiments WHERE best_score IS NOT NULL' +
                (' AND user_id = ?' if user_id else ''),
                user_params
            ).fetchone()['avg']

            # Problem type distribution
            types = conn.execute(
                f'SELECT problem_type, COUNT(*) as cnt FROM experiments{user_filter} GROUP BY problem_type',
                user_params
            ).fetchall()

            return {
                'total_experiments': total,
                'completed': completed,
                'avg_best_score': round(float(avg_score), 4) if avg_score else None,
                'problem_types': {r['problem_type']: r['cnt'] for r in types if r['problem_type']},
            }
