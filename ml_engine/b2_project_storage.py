"""
AutoML Studio — Backblaze B2 Project Storage
Save/load projects to B2 cloud storage.
"""

import json
import os
import pickle
import tempfile
from datetime import datetime

from ml_engine.b2_storage import (
    upload_bytes,
    download_bytes,
    upload_file,
    download_file,
    key_exists,
    list_prefix,
    delete_prefix,
)


def _safe_name(name):
    """Sanitize a project name for use as a B2 key component."""
    return ''.join(c for c in name if c.isalnum() or c in ('_', '-', ' ')).strip()


def _b2_prefix(user_id, safe_name):
    """Return the B2 key prefix for a project."""
    return f'users/{user_id or "anonymous"}/projects/{safe_name}'


def save_project(session, name=None, user_id=None):
    """
    Serialize a PipelineSession and upload to B2.

    Args:
        session: PipelineSession object
        name: project name (defaults to session_id)
        user_id: optional user identifier

    Returns:
        dict with save result
    """
    name = name or session.session_id
    safe = _safe_name(name)
    prefix = _b2_prefix(user_id, safe)

    metadata = {
        'name': name,
        'session_id': session.session_id,
        'saved_at': datetime.now().isoformat(),
        'status': session.status,
        'current_step': session.current_step,
        'is_timeseries': session.is_timeseries,
        'experiment_id': session.experiment_id,
        'user_id': user_id or 'anonymous',
    }

    # Save profile
    if session.profile:
        metadata['profile'] = {k: v for k, v in session.profile.items()
                               if k != 'preview'}
        metadata['has_profile'] = True

    # Save training results (serializable parts)
    if session.training_results:
        try:
            tr = {}
            for k, v in session.training_results.items():
                if k == 'training_context':
                    tr[k] = {kk: vv for kk, vv in v.items() if kk != 'trained_models'}
                elif k in ('best_model_path',):
                    pass
                else:
                    json.dumps(v)  # test serializable
                    tr[k] = v
            metadata['training_results'] = tr
        except (TypeError, ValueError):
            pass

    if session.retrain_results:
        try:
            json.dumps(session.retrain_results)
            metadata['retrain_results'] = session.retrain_results
        except (TypeError, ValueError):
            pass

    if session.recommendations:
        try:
            json.dumps(session.recommendations)
            metadata['recommendations'] = session.recommendations
        except (TypeError, ValueError):
            pass

    # Upload metadata JSON
    try:
        meta_bytes = json.dumps(metadata, indent=2, default=str).encode('utf-8')
        upload_bytes(f'{prefix}/metadata.json', meta_bytes, content_type='application/json')
    except Exception as exc:
        return {'error': f'Failed to upload metadata: {exc}'}

    # Upload original dataframe
    if session.original_df is not None:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            try:
                session.original_df.to_csv(tmp.name, index=False)
                upload_file(f'{prefix}/original_data.csv', tmp.name)
            finally:
                os.unlink(tmp.name)
        except Exception:
            pass

    # Upload transformed dataframe
    if session.transformed_df is not None:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
            try:
                session.transformed_df.to_csv(tmp.name, index=False)
                upload_file(f'{prefix}/transformed_data.csv', tmp.name)
            finally:
                os.unlink(tmp.name)
        except Exception:
            pass

    # Upload best model
    if session.best_model is not None:
        try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.pkl')
            try:
                with open(tmp.name, 'wb') as f:
                    pickle.dump(session.best_model, f)
                upload_file(f'{prefix}/best_model.pkl', tmp.name)
            finally:
                os.unlink(tmp.name)
        except Exception:
            pass

    return {
        'success': True,
        'project_name': name,
        'saved_at': metadata['saved_at'],
    }


def load_project(name, user_id=None):
    """
    Load a project from B2 and return data to restore a session.

    Returns:
        dict with project data, or {'error': ...}
    """
    safe = _safe_name(name)
    prefix = _b2_prefix(user_id, safe)
    meta_key = f'{prefix}/metadata.json'

    # Download metadata
    try:
        if not key_exists(meta_key):
            return {'error': f'Project "{name}" not found'}
        raw = download_bytes(meta_key)
        metadata = json.loads(raw.decode('utf-8'))
    except Exception as exc:
        return {'error': f'Failed to load project metadata: {exc}'}

    original_key = f'{prefix}/original_data.csv'
    transformed_key = f'{prefix}/transformed_data.csv'
    model_key = f'{prefix}/best_model.pkl'

    has_original = key_exists(original_key)
    has_transformed = key_exists(transformed_key)
    has_model = key_exists(model_key)

    tmp_dir = tempfile.mkdtemp(prefix='automl_project_')

    original_path = None
    if has_original:
        try:
            original_path = os.path.join(tmp_dir, 'original_data.csv')
            download_file(original_key, original_path)
        except Exception:
            original_path = None
            has_original = False

    transformed_path = None
    if has_transformed:
        try:
            transformed_path = os.path.join(tmp_dir, 'transformed_data.csv')
            download_file(transformed_key, transformed_path)
        except Exception:
            transformed_path = None
            has_transformed = False

    model_path = None
    if has_model:
        try:
            model_path = os.path.join(tmp_dir, 'best_model.pkl')
            download_file(model_key, model_path)
        except Exception:
            model_path = None
            has_model = False

    return {
        'metadata': metadata,
        'has_original_data': has_original,
        'has_transformed_data': has_transformed,
        'has_model': has_model,
        'original_data_path': original_path,
        'transformed_data_path': transformed_path,
        'model_path': model_path,
    }


def list_projects(user_id=None):
    """
    List all saved projects in B2.

    Returns:
        list of project summary dicts
    """
    prefix = f'users/{user_id or "anonymous"}/projects/'
    projects = []

    try:
        keys = list_prefix(prefix)
    except Exception:
        return projects

    # Discover unique project names from key prefixes
    project_names = set()
    for key in keys:
        # key looks like  users/<uid>/projects/<safe_name>/...
        remainder = key[len(prefix):]
        parts = remainder.split('/')
        if parts and parts[0]:
            project_names.add(parts[0])

    for proj_name in sorted(project_names):
        meta_key = f'{prefix}{proj_name}/metadata.json'
        try:
            raw = download_bytes(meta_key)
            meta = json.loads(raw.decode('utf-8'))

            projects.append({
                'name': meta.get('name', proj_name),
                'session_id': meta.get('session_id', ''),
                'saved_at': meta.get('saved_at', ''),
                'status': meta.get('status', 'unknown'),
                'current_step': meta.get('current_step', ''),
                'has_model': key_exists(f'{prefix}{proj_name}/best_model.pkl'),
                'has_data': key_exists(f'{prefix}{proj_name}/original_data.csv'),
                'profile_summary': {
                    'problem_type': meta.get('profile', {}).get('problem_type', ''),
                    'n_rows': meta.get('profile', {}).get('n_rows', 0),
                    'n_cols': meta.get('profile', {}).get('n_cols', 0),
                    'target': meta.get('profile', {}).get('target_column', ''),
                },
            })
        except Exception:
            continue

    return sorted(projects, key=lambda p: p.get('saved_at', ''), reverse=True)


def delete_project(name, user_id=None):
    """Delete a saved project from B2."""
    safe = _safe_name(name)
    prefix = _b2_prefix(user_id, safe)

    try:
        keys = list_prefix(f'{prefix}/')
        if not keys:
            return {'error': 'Project not found'}
        delete_prefix(f'{prefix}/')
        return {'success': True}
    except Exception as exc:
        return {'error': f'Failed to delete project: {exc}'}
