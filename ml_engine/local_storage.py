"""
AutoML Studio — Local Project Storage
Save/load projects to the local filesystem without login.
"""

import os
import json
import shutil
import pickle
import time
from datetime import datetime


PROJECTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'projects')


def _ensure_dir():
    os.makedirs(PROJECTS_DIR, exist_ok=True)


def save_project(session, name=None):
    """
    Serialize a PipelineSession to disk.

    Args:
        session: PipelineSession object
        name: project name (defaults to session_id)

    Returns:
        dict with save result
    """
    _ensure_dir()
    name = name or session.session_id
    safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-', ' ')).strip()
    project_dir = os.path.join(PROJECTS_DIR, safe_name)
    os.makedirs(project_dir, exist_ok=True)

    metadata = {
        'name': name,
        'session_id': session.session_id,
        'saved_at': datetime.now().isoformat(),
        'status': session.status,
        'current_step': session.current_step,
        'is_timeseries': session.is_timeseries,
        'experiment_id': session.experiment_id,
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

    # Save metadata JSON
    with open(os.path.join(project_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Save original dataframe
    if session.original_df is not None:
        try:
            session.original_df.to_csv(
                os.path.join(project_dir, 'original_data.csv'), index=False
            )
        except Exception:
            pass

    # Save transformed dataframe
    if session.transformed_df is not None:
        try:
            session.transformed_df.to_csv(
                os.path.join(project_dir, 'transformed_data.csv'), index=False
            )
        except Exception:
            pass

    # Save best model
    if session.best_model is not None:
        try:
            with open(os.path.join(project_dir, 'best_model.pkl'), 'wb') as f:
                pickle.dump(session.best_model, f)
        except Exception:
            pass

    # Copy any output files
    if session.output_dir and os.path.exists(session.output_dir):
        out_dest = os.path.join(project_dir, 'outputs')
        if os.path.exists(out_dest):
            shutil.rmtree(out_dest)
        try:
            shutil.copytree(session.output_dir, out_dest)
        except Exception:
            pass

    return {
        'success': True,
        'project_name': name,
        'project_dir': project_dir,
        'saved_at': metadata['saved_at'],
    }


def load_project(name):
    """
    Load a project from disk and return data to restore a session.

    Returns:
        dict with project data, or {'error': ...}
    """
    _ensure_dir()
    safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-', ' ')).strip()
    project_dir = os.path.join(PROJECTS_DIR, safe_name)

    if not os.path.exists(project_dir):
        return {'error': f'Project "{name}" not found'}

    meta_path = os.path.join(project_dir, 'metadata.json')
    if not os.path.exists(meta_path):
        return {'error': 'Project metadata missing'}

    with open(meta_path, 'r') as f:
        metadata = json.load(f)

    # Load dataframes
    original_path = os.path.join(project_dir, 'original_data.csv')
    transformed_path = os.path.join(project_dir, 'transformed_data.csv')

    result = {
        'metadata': metadata,
        'has_original_data': os.path.exists(original_path),
        'has_transformed_data': os.path.exists(transformed_path),
        'has_model': os.path.exists(os.path.join(project_dir, 'best_model.pkl')),
        'project_dir': project_dir,
        'original_data_path': original_path if os.path.exists(original_path) else None,
        'transformed_data_path': transformed_path if os.path.exists(transformed_path) else None,
        'model_path': os.path.join(project_dir, 'best_model.pkl') if os.path.exists(os.path.join(project_dir, 'best_model.pkl')) else None,
    }

    return result


def list_projects():
    """
    List all saved projects.

    Returns:
        list of project summary dicts
    """
    _ensure_dir()
    projects = []

    for item in sorted(os.listdir(PROJECTS_DIR)):
        project_dir = os.path.join(PROJECTS_DIR, item)
        if not os.path.isdir(project_dir):
            continue

        meta_path = os.path.join(project_dir, 'metadata.json')
        if not os.path.exists(meta_path):
            continue

        try:
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            
            # Calculate project size
            total_size = 0
            for root, dirs, files in os.walk(project_dir):
                for fn in files:
                    total_size += os.path.getsize(os.path.join(root, fn))

            projects.append({
                'name': meta.get('name', item),
                'session_id': meta.get('session_id', ''),
                'saved_at': meta.get('saved_at', ''),
                'status': meta.get('status', 'unknown'),
                'current_step': meta.get('current_step', ''),
                'has_model': os.path.exists(os.path.join(project_dir, 'best_model.pkl')),
                'has_data': os.path.exists(os.path.join(project_dir, 'original_data.csv')),
                'size_mb': round(total_size / 1024 / 1024, 2),
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


def delete_project(name):
    """Delete a saved project."""
    _ensure_dir()
    safe_name = "".join(c for c in name if c.isalnum() or c in ('_', '-', ' ')).strip()
    project_dir = os.path.join(PROJECTS_DIR, safe_name)

    if os.path.exists(project_dir):
        shutil.rmtree(project_dir)
        return {'success': True}
    return {'error': 'Project not found'}
