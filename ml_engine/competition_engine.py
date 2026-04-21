"""
AutoML Studio — Competition Engine (Local Leaderboard)
Compares model performance across experiments on similar dataset types.
"""

import json
import sqlite3
from datetime import datetime


class CompetitionEngine:
    """Local competition leaderboard comparing experiments."""

    def __init__(self, db_path):
        self.db_path = db_path
        self._init_table()

    def _init_table(self):
        """Create leaderboard table."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS leaderboard (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT UNIQUE,
                    model_name TEXT,
                    score REAL,
                    metric TEXT,
                    dataset_type TEXT,
                    n_rows INTEGER,
                    n_features INTEGER,
                    problem_type TEXT,
                    submitted_at TEXT,
                    hyperparams TEXT DEFAULT '{}'
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_lb_score ON leaderboard(score DESC)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_lb_type ON leaderboard(dataset_type)')

    def submit(self, experiment_id, model_name, score, metric, dataset_type='general',
               n_rows=0, n_features=0, problem_type='classification', hyperparams=None):
        """Submit a model result to the leaderboard."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO leaderboard
                (experiment_id, model_name, score, metric, dataset_type,
                 n_rows, n_features, problem_type, submitted_at, hyperparams)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experiment_id, model_name, score, metric, dataset_type,
                n_rows, n_features, problem_type,
                datetime.utcnow().isoformat(),
                json.dumps(hyperparams or {}, default=str)
            ))

        rank = self._get_rank(experiment_id, dataset_type)
        return {
            'success': True,
            'rank': rank['rank'],
            'total': rank['total'],
            'message': f'Submitted! Rank #{rank["rank"]} of {rank["total"]}.',
        }

    def get_leaderboard(self, dataset_type=None, problem_type=None, limit=20):
        """Get ranked leaderboard entries."""
        query = 'SELECT * FROM leaderboard'
        params = []
        conditions = []

        if dataset_type:
            conditions.append('dataset_type = ?')
            params.append(dataset_type)
        if problem_type:
            conditions.append('problem_type = ?')
            params.append(problem_type)

        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        query += ' ORDER BY score DESC LIMIT ?'
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        entries = []
        for i, r in enumerate(rows, 1):
            entries.append({
                'rank': i,
                'experiment_id': r['experiment_id'],
                'model_name': r['model_name'],
                'score': round(r['score'], 4),
                'metric': r['metric'],
                'dataset_type': r['dataset_type'],
                'problem_type': r['problem_type'],
                'n_rows': r['n_rows'],
                'n_features': r['n_features'],
                'submitted_at': r['submitted_at'],
                'medal': '🥇' if i == 1 else '🥈' if i == 2 else '🥉' if i == 3 else '',
            })

        return {
            'entries': entries,
            'total': len(entries),
        }

    def get_rank(self, experiment_id):
        """Get the rank and stats for a specific experiment."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT * FROM leaderboard WHERE experiment_id = ?',
                (experiment_id,)
            ).fetchone()

        if not row:
            return {'error': 'Experiment not found in leaderboard'}

        rank_info = self._get_rank(experiment_id, row['dataset_type'])

        # Get top score
        with sqlite3.connect(self.db_path) as conn:
            top = conn.execute(
                'SELECT MAX(score) as top FROM leaderboard WHERE dataset_type = ?',
                (row['dataset_type'],)
            ).fetchone()

        return {
            'rank': rank_info['rank'],
            'total': rank_info['total'],
            'percentile': round((1 - rank_info['rank'] / max(rank_info['total'], 1)) * 100),
            'score': round(row['score'], 4),
            'top_score': round(top['top'], 4) if top['top'] else None,
            'gap_to_top': round(top['top'] - row['score'], 4) if top['top'] else None,
            'medal': '🥇' if rank_info['rank'] == 1 else '🥈' if rank_info['rank'] == 2 else '🥉' if rank_info['rank'] == 3 else '',
        }

    def _get_rank(self, experiment_id, dataset_type):
        """Get rank of an experiment within its dataset type."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                'SELECT experiment_id FROM leaderboard WHERE dataset_type = ? ORDER BY score DESC',
                (dataset_type,)
            ).fetchall()

        total = len(rows)
        rank = total  # Default to last
        for i, r in enumerate(rows, 1):
            if r[0] == experiment_id:
                rank = i
                break

        return {'rank': rank, 'total': total}
