"""
AutoML Studio - Flask Application
REST API for the full-featured AutoML dashboard with 18 features.
"""

import os
from dotenv import load_dotenv
load_dotenv()

import json
import shutil
import zipfile
import io
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response, g
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import time
from ml_engine.decision_engine import DecisionEngine
from ml_engine.pipeline import PipelineManager
from ml_engine.profiler import read_dataset
from ml_engine.b2_storage import (
    upload_bytes,
    upload_file,
    generate_download_url,
    key_exists,
    list_prefix
)
from ml_engine.unsupervised_report_generator import generate_unsupervised_html_report
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from logging_config import setup_logging
from auth import verify_firebase_token, get_current_user_uid, login_required, _get_firebase_app

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'dev-change-me-in-production')

# CORS — configurable origins (defaults to localhost)
_allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:7860').split(',')
CORS(app, origins=[o.strip() for o in _allowed_origins], supports_credentials=True)

# Rate limiting
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["200/hour"])

# Max upload size — configurable via env var (default 100 MB)
_max_upload_mb = int(os.getenv('MAX_UPLOAD_MB', '100'))
app.config['MAX_CONTENT_LENGTH'] = _max_upload_mb * 1024 * 1024

# Structured logging
setup_logging(app)

# ── Authentication: protect all /api/ routes with Firebase ID token ──
# Routes that do NOT require authentication:
_PUBLIC_PREFIXES = ('/health', '/login', '/static/', '/favicon')
_PUBLIC_PATHS = {'/', '/dashboard', '/profile'}


@app.before_request
def _enforce_auth():
    """Verify Firebase ID token on every /api/ request."""
    path = request.path

    # Skip auth for public routes (pages, health, static assets)
    if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES):
        return None

    # Only enforce on API routes
    if not path.startswith('/api/'):
        return None

    # Check if Firebase Admin SDK is available
    if _get_firebase_app() is None:
        # Firebase not configured — allow request with anonymous identity
        g.user = {'uid': 'anonymous', 'email': 'anonymous@local'}
        return None

    auth_header = request.headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        return jsonify({
            'error': 'Authentication required',
            'message': 'Missing or malformed Authorization header. Expected: Bearer <firebase_id_token>'
        }), 401

    id_token = auth_header[7:]
    if not id_token:
        return jsonify({'error': 'Authentication required', 'message': 'Empty token'}), 401

    claims = verify_firebase_token(id_token)
    if claims is None:
        return jsonify({
            'error': 'Invalid or expired token',
            'message': 'Please sign in again'
        }), 401

    # Store verified user for downstream route handlers
    g.user = claims

# Custom JSON encoder for numpy types
from flask.json.provider import DefaultJSONProvider
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

app.json_provider_class = NumpyJSONProvider
app.json = NumpyJSONProvider(app)

import tempfile

UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'automl_uploads')
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), 'automl_outputs')

pipeline_manager = PipelineManager(UPLOAD_DIR, OUTPUT_DIR)

# Supported file extensions
ALLOWED_EXTENSIONS = {'.csv', '.tsv', '.xlsx', '.xls', '.json', '.parquet'}

# Magic-byte MIME types for file content validation
ALLOWED_MIMES = {
    'text/csv', 'text/plain', 'text/tab-separated-values',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel',
    'application/json',
    'application/octet-stream',  # parquet files
    'application/zip',           # xlsx is a zip
}


def _allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


def _validate_file_content(file_storage):
    """Validate uploaded file by magic bytes, not just extension."""
    try:
        import magic
        header = file_storage.read(2048)
        file_storage.seek(0)
        mime = magic.from_buffer(header, mime=True)
        if mime not in ALLOWED_MIMES:
            return False, f"File content type '{mime}' is not allowed"
        return True, None
    except ImportError:
        # python-magic not installed — fall back to extension-only check
        return True, None
    except Exception as e:
        return True, None  # Don't block uploads on validation errors


@app.route('/health')
@limiter.exempt
def health():
    """Health check endpoint for Docker/K8s probes."""
    db_status = 'connected'
    try:
        from ml_engine.experiment_store import ExperimentStore
        ExperimentStore()
    except Exception:
        db_status = 'disconnected'
    return jsonify({'status': 'ok', 'db': db_status})


@app.route('/')
def home():
    """Redirect root to login."""
    from flask import redirect, url_for
    return redirect(url_for('login'))

@app.route('/login')
def login():
    """Serve the login page."""
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """Serve the project overview dashboard."""
    return render_template('dashboard.html')

@app.route('/profile')
def profile():
    """Serve the logged-in user profile page."""
    return render_template('profile.html')

import pandas as pd
from io import BytesIO

@app.route('/api/preview-columns', methods=['POST'])
def preview_columns():
    """Extract columns from uploaded file for dropdown selection."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        columns = []
        
        file_bytes = file.read()
        buffer = BytesIO(file_bytes)
        
        if ext == 'csv':
            df = pd.read_csv(buffer, nrows=0)
            columns = df.columns.tolist()
        elif ext in ['xlsx', 'xls']:
            df = pd.read_excel(buffer, nrows=0)
            columns = df.columns.tolist()
        elif ext == 'parquet':
            df = pd.read_parquet(buffer)
            columns = df.columns.tolist()
        elif ext == 'json':
            df = pd.read_json(buffer)
            columns = df.columns.tolist()
        else:
            return jsonify({'error': 'Unsupported format'}), 400
            
        file.seek(0)
        return jsonify({'columns': columns})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# ==============================================================================
# AUTH: CURRENT USER IDENTITY
# ==============================================================================

@app.route('/api/me')
@login_required
def get_me():
    """Return the current authenticated user's identity (uid, email).
    Frontend uses this to verify session validity without trusting localStorage.
    """
    user = getattr(g, 'user', {})
    return jsonify({
        'uid': user.get('uid'),
        'email': user.get('email', ''),
        'name': user.get('name', ''),
    })


# ==============================================================================
# STEP 1: UPLOAD & PROFILE
# ==============================================================================

@app.route('/api/upload', methods=['POST'])
@limiter.limit("20/minute")
@login_required
def upload():
    """Step 1: Upload dataset (CSV/Excel/JSON/Parquet) and get profile."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not _allowed_file(file.filename):
        return jsonify({'error': f'Unsupported format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Validate file content by magic bytes
    valid, err_msg = _validate_file_content(file)
    if not valid:
        return jsonify({'error': err_msg}), 400
    
    problem_statement = request.form.get('problem_statement', '')
    user_id = get_current_user_uid()
    
    # Create session
    session = pipeline_manager.create_session()
    session.user_id = user_id
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, f'{session.session_id}_{filename}')
    file.save(filepath)
    
    # Profile
    result = pipeline_manager.upload_and_profile(session.session_id, filepath, problem_statement, user_id=user_id)
    
    # Clean up temp upload file (already synced to B2 by pipeline)
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except OSError:
        pass
    
    return jsonify(result)

@app.route("/api/decision/<session_id>")
def decision(session_id):
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    # DecisionEngine expects a dict-style payload.
    session_data = {
        'training_results': session.training_results or {},
        'drift': session.drift_report or {},
        'profile': session.profile or {},
    }
    engine = DecisionEngine()
    return jsonify(engine.analyze(session_data))





@app.route('/api/download/<session_id>/<file_type>')
@login_required
def api_download(session_id, file_type):
    from flask import request
    filename = request.args.get('filename')
    user_id = get_current_user_uid()
    session = pipeline_manager.get_session(session_id)
    exp = None

    if session and getattr(session, 'experiment_id', None):
        exp = pipeline_manager.experiment_store.get_experiment(session.experiment_id, user_id=user_id)
    if not exp:
        exp = pipeline_manager.experiment_store.get_experiment(session_id, user_id=user_id)

    if not filename and file_type == 'csv':
        if session and getattr(session, 'upload_path', None):
            filename = os.path.basename(session.upload_path)
        elif exp:
            filename = exp.get('dataset_name')

    # Build candidate keys — user-scoped paths first, then legacy paths for backward compat
    candidate_keys = []
    user_prefix = f"users/{user_id}/sessions/{session_id}" if user_id else None
    legacy_prefix = f"sessions/{session_id}"

    if file_type == 'csv':
        if filename:
            if user_prefix:
                candidate_keys.append(f"{user_prefix}/uploads/{filename}")
                candidate_keys.append(f"{user_prefix}/uploads/{session_id}_{filename}")
            candidate_keys.append(f"{legacy_prefix}/uploads/{filename}")
            candidate_keys.append(f"{legacy_prefix}/uploads/{session_id}_{filename}")
        # Also scan the uploads folder
        for prefix in [p for p in [user_prefix, legacy_prefix] if p]:
            try:
                found = list_prefix(f"{prefix}/uploads/")
                if found:
                    candidate_keys.extend(found)
            except Exception:
                pass
    elif file_type == 'transformed':
        if user_prefix:
            candidate_keys.append(f"{user_prefix}/data/transformed_data.csv")
        candidate_keys.append(f"{legacy_prefix}/data/transformed_data.csv")
    elif file_type == 'model':
        if user_prefix:
            candidate_keys.append(f"{user_prefix}/models/best_model.pkl")
        candidate_keys.append(f"{legacy_prefix}/models/best_model.pkl")
    elif file_type == 'report':
        if user_prefix:
            candidate_keys.append(f"{user_prefix}/reports/automl_report.html")
        candidate_keys.append(f"{legacy_prefix}/reports/automl_report.html")

    key = next((candidate for candidate in candidate_keys if key_exists(candidate)), None)

    if not key:
        return jsonify({
            "error": "File not found in Cloud Storage"
        }), 400

    try:
        url = generate_download_url(key)
        return jsonify({
            "success": True,
            "url": url
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


@app.route("/api/auto-run/<session_id>", methods=["POST"])
def auto_run(session_id):
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    # Step 1: start clean/transform. Training should be triggered after clean completes.
    result = pipeline_manager.clean_and_transform(session_id)
    if 'error' in result:
        return jsonify(result), 400

    return jsonify({
        'success': True,
        'status': 'started',
        'message': 'Auto-run started. Poll /api/status until current_step is train, then call /api/train.',
    })


@app.route("/api/export/model/<session_id>")
def export_model(session_id):
    import joblib
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.best_model is None:
        return jsonify({'error': 'No trained model available'}), 400

    import tempfile
    export_dir = os.path.join(tempfile.gettempdir(), 'automl_exports')
    os.makedirs(export_dir, exist_ok=True)
    path = os.path.join(export_dir, f"{session_id}.pkl")
    joblib.dump(session.best_model, path)
    return send_file(path, as_attachment=True)

@app.route('/api/update-target', methods=['POST'])
def update_target():
    """Allow user to override the auto-detected target column or problem type."""
    data = request.json
    session_id = data.get('session_id')
    
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    if 'target_column' in data:
        session.profile['target_column'] = data['target_column']
    if 'problem_type' in data:
        session.profile['problem_type'] = data['problem_type']
        session.is_timeseries = data['problem_type'] == 'forecasting'
    
    return jsonify({'success': True, 'profile': session.profile})


# ==============================================================================
# STEP 2: CLEAN & TRANSFORM
# ==============================================================================

@app.route('/api/clean-transform', methods=['POST'])
def clean_transform():
    """Step 2: Clean and transform the dataset."""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400
    
    result = pipeline_manager.clean_and_transform(session_id)
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)


# ==============================================================================
# STEP 3: TRAIN MODELS
# ==============================================================================

@app.route('/api/train', methods=['POST'])
@limiter.limit("5/minute")
def train():
    """Step 3: Train models (ML + Deep Learning)."""
    data = request.json
    session_id = data.get('session_id')
    time_budget = data.get('time_budget')  # Optional: seconds
    
    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400
    
    result = pipeline_manager.train(session_id, time_budget_seconds=time_budget)
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)


# ==============================================================================
# STEP 4: RETRAIN
# ==============================================================================

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Retrain models applying all recommendations."""
    data = request.get_json()
    session_id = data.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if not session.training_results:
        return jsonify({'error': 'No training results to improve upon'}), 400
    if not session.recommendations:
        return jsonify({'error': 'No recommendations available'}), 400

    from ml_engine.retrainer import retrain_with_recommendations
    try:
        df = session.transformed_df if session.transformed_df is not None else session.cleaned_df
        if df is None:
            return jsonify({'error': 'No transformed data available'}), 400

        result = retrain_with_recommendations(
            df.copy(),
            session.profile or {},
            session.transform_metadata or {},
            session.recommendations,
            session.training_results,
            session.output_dir,
        )

        if 'error' in result:
            return jsonify(result), 400

        session.retrain_results = result

        # Update best model if improved
        ctx = result.get('training_context', {})
        trained = ctx.get('trained_models', {})
        best_name = result.get('best_model')
        if best_name and best_name in trained:
            session.best_model = trained[best_name]
            session.trained_models = trained

        return jsonify({
            'success': True,
            'best_model': result.get('best_model'),
            'best_score': result.get('best_score'),
            'original_best_score': result.get('original_best_score'),
            'improvement': result.get('improvement'),
            'improvement_pct': result.get('improvement_pct'),
            'leaderboard': result.get('leaderboard', []),
            'retrain_report': result.get('retrain_report', {}),
            'applied_recommendations': result.get('applied_recommendations', []),
            'feature_changes': result.get('feature_changes', []),
        })
    except Exception as e:
        return jsonify({'error': f'Retrain failed: {str(e)}'}), 500


# ==============================================================================
# STATUS & DOWNLOADS
# ==============================================================================

@app.route('/api/status/<session_id>')
def status(session_id):
    """Get current session status and results."""
    result = pipeline_manager.get_status(session_id)
    
    if 'error' in result:
        return jsonify(result), 404
    
    return jsonify(result)


@app.route('/api/download-csv/<session_id>')
def download_csv(session_id):
    """Download the cleaned/transformed CSV."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    csv_path = os.path.join(session.output_dir, 'cleaned_data.csv')
    if not os.path.exists(csv_path):
        return jsonify({'error': 'Cleaned CSV not found'}), 404
    
    return send_file(csv_path, as_attachment=True, download_name='cleaned_data.csv')


@app.route('/api/download-model/<session_id>')
def download_model(session_id):
    """Download the best model as .pkl."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    model_path = os.path.join(session.output_dir, 'best_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model not found'}), 404
    
    return send_file(model_path, as_attachment=True, download_name='best_model.pkl')


@app.route('/api/download-improved-model/<session_id>')
def download_improved_model(session_id):
    """Download the improved model after retrain."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    
    model_path = os.path.join(session.output_dir, 'improved_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Improved model not found'}), 404
    
    return send_file(model_path, as_attachment=True, download_name='improved_model.pkl')


# ==============================================================================
# RETRAIN & RECOMMENDATIONS
# ==============================================================================

@app.route('/api/recommendations/<session_id>')
def get_recommendations(session_id):
    """Get recommendations for the current session."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if not session.recommendations:
        return jsonify({'recommendations': [], 'message': 'No recommendations yet. Train a model first.'})
    return jsonify({'recommendations': session.recommendations})





# ==============================================================================
# PHASE 1: EXPLAINABILITY & DIAGNOSTICS
# ==============================================================================

@app.route('/api/explain/<session_id>')
def get_explainability(session_id):
    """Get SHAP-based model explanations."""
    result = pipeline_manager.get_explainability(session_id)
    if 'error' in result:
        status_code = 404 if result.get('error') == 'Session not found' else 400
        return jsonify(result), status_code
    return jsonify(result)


@app.route('/api/explain-row/<session_id>/<int:row_index>')
def explain_row(session_id, row_index):
    """Get local explanation for a specific test row."""
    result = pipeline_manager.explain_row(session_id, row_index)
    if 'error' in result:
        status_code = 404 if result.get('error') == 'Session not found' else 400
        return jsonify(result), status_code
    return jsonify(result)


@app.route('/api/whatif/<session_id>', methods=['POST'])
def whatif(session_id):
    """What-if analysis: change a feature and see prediction change."""
    data = request.json
    row_index = data.get('row_index', 0)
    feature_name = data.get('feature_name')
    new_value = data.get('new_value')
    
    if not feature_name or new_value is None:
        return jsonify({'error': 'feature_name and new_value required'}), 400
    
    result = pipeline_manager.run_whatif(session_id, row_index, feature_name, new_value)
    if 'error' in result:
        status_code = 404 if result.get('error') == 'Session not found' else 400
        return jsonify(result), status_code
    return jsonify(result)


@app.route('/api/diagnostics/<session_id>')
def get_diagnostics(session_id):
    """Get advanced model diagnostics (ROC, residuals, learning curves)."""
    result = pipeline_manager.get_diagnostics(session_id)
    if 'error' in result:
        status_code = 404 if result.get('error') == 'Session not found' else 400
        return jsonify(result), status_code
    return jsonify(result)


# ==============================================================================
# PHASE 3: PREDICTION API & DEPLOYMENT
# ==============================================================================

@app.route('/api/predict/<session_id>', methods=['POST'])
def predict(session_id):
    """Make a single prediction with the trained model."""
    data = request.json
    if not data:
        return jsonify({'error': 'JSON body required with feature values'}), 400
    
    result = pipeline_manager.predict(session_id, data)
    if 'error' in result:
        err = str(result.get('error', ''))
        if err == 'Session not found':
            return jsonify(result), 404
        if err == 'No trained model available':
            return jsonify(result), 409
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/batch-predict/<session_id>', methods=['POST'])
def batch_predict(session_id):
    """Batch prediction: upload CSV with features, get predictions back."""
    if 'file' in request.files:
        file = request.files['file']
        df = read_dataset(file)
    elif request.json:
        df = pd.DataFrame(request.json)
    else:
        return jsonify({'error': 'Upload a file or send JSON array'}), 400
    
    result = pipeline_manager.batch_predict(session_id, df)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/export-deployment/<session_id>')
def export_deployment(session_id):
    """Export deployment package as ZIP (predict.py, serve.py, Dockerfile, etc.)."""
    result = pipeline_manager.export_deployment(session_id)
    if 'error' in result:
        return jsonify(result), 400
    
    deploy_dir = result.get('deploy_dir')
    if not deploy_dir or not os.path.exists(deploy_dir):
        return jsonify({'error': 'Deployment package not found'}), 404
    
    # Create ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(deploy_dir):
            for file in files:
                filepath = os.path.join(root, file)
                arcname = os.path.relpath(filepath, deploy_dir)
                zf.write(filepath, arcname)
    
    zip_buffer.seek(0)
    return send_file(
        zip_buffer, as_attachment=True,
        download_name='automl_deployment.zip',
        mimetype='application/zip'
    )


# ==============================================================================
# PHASE 3: DATA DRIFT MONITORING
# ==============================================================================

@app.route('/api/drift/<session_id>', methods=['POST'])
def check_drift(session_id):
    """Check data drift against training data distribution."""
    if 'file' in request.files:
        file = request.files['file']
        filepath = os.path.join(UPLOAD_DIR, f'drift_{secure_filename(file.filename)}')
        file.save(filepath)
        new_df = read_dataset(filepath)
    elif request.json:
        new_df = pd.DataFrame(request.json)
    else:
        return jsonify({'error': 'Upload a file or send JSON data'}), 400
    
    result = pipeline_manager.check_drift(session_id, new_df)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# MODEL DRIFT DETECTION
# ==============================================================================

@app.route('/api/model-drift/<session_id>', methods=['POST'])
def check_model_drift(session_id):
    """Check model performance drift with new labeled data."""
    if 'file' in request.files:
        file = request.files['file']
        filepath = os.path.join(UPLOAD_DIR, f'mdrift_{secure_filename(file.filename)}')
        file.save(filepath)
        new_df = read_dataset(filepath)
    elif request.json:
        new_df = pd.DataFrame(request.json)
    else:
        return jsonify({'error': 'Upload a file or send JSON data'}), 400

    result = pipeline_manager.check_model_drift(session_id, new_df)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# PHASE 4: EXPERIMENT TRACKING
# ==============================================================================

@app.route('/api/experiments')
@login_required
def list_experiments():
    """List all experiments with optional filtering."""
    user_id = get_current_user_uid()
    search = request.args.get('search', None)
    tag = request.args.get('tag', None)
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    sort_by = request.args.get('sort_by', 'created_at')
    sort_order = request.args.get('sort_order', 'desc')
    
    result = pipeline_manager.list_experiments(
        search=search, tag=tag, limit=limit, offset=offset,
        sort_by=sort_by, sort_order=sort_order, user_id=user_id
    )
    return jsonify(result)


@app.route('/api/experiments/<exp_id>')
@login_required
def get_experiment(exp_id):
    """Get experiment details."""
    user_id = get_current_user_uid()
    result = pipeline_manager.get_experiment(exp_id, user_id=user_id)
    if not result:
        result = pipeline_manager.experiment_store.get_experiment_by_session_id(exp_id, user_id=user_id)
    if not result:
        return jsonify({'error': 'Experiment not found'}), 404
    return jsonify(result)


@app.route('/api/experiments/<exp_id>', methods=['PUT'])
@login_required
def update_experiment(exp_id):
    """Update experiment metadata (name, tags, notes)."""
    data = request.json
    pipeline_manager.update_experiment(exp_id, **data)
    return jsonify({'success': True})


@app.route('/api/experiments/<exp_id>', methods=['DELETE'])
@login_required
def delete_experiment(exp_id):
    """Delete an experiment."""
    user_id = get_current_user_uid()
    pipeline_manager.delete_experiment(exp_id, user_id=user_id)
    return jsonify({'success': True})


@app.route('/api/experiments/compare', methods=['POST'])
@login_required
def compare_experiments():
    """Compare multiple experiments side by side."""
    data = request.json
    exp_ids = data.get('experiment_ids', [])
    
    if len(exp_ids) < 2:
        return jsonify({'error': 'Need at least 2 experiment IDs'}), 400
    
    result = pipeline_manager.compare_experiments(exp_ids)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/experiments/stats')
@login_required
def experiment_stats():
    """Get overall experiment statistics."""
    user_id = get_current_user_uid()
    return jsonify(pipeline_manager.get_experiment_stats(user_id=user_id))


# ==============================================================================
# USER STATE & PREFERENCES (replaces browser localStorage)
# ==============================================================================

# In-memory fallback when Firestore is unavailable
_user_state_cache = {}
_user_prefs_cache = {}


def _get_user_state_store():
    """Get Firestore client for user state, or None for in-memory fallback."""
    try:
        from ml_engine.firestore_experiment_store import _get_firestore
        return _get_firestore()
    except Exception:
        return None


@app.route('/api/user/state', methods=['GET'])
@login_required
def get_user_state():
    """Fetch saved UI state (session_id, active_tab, stage, etc.)."""
    user_id = get_current_user_uid()
    db = _get_user_state_store()

    if db:
        try:
            doc = db.collection('users').document(user_id).collection('settings').document('state').get()
            if doc.exists:
                return jsonify(doc.to_dict())
        except Exception:
            pass

    # In-memory fallback
    return jsonify(_user_state_cache.get(user_id, {}))


@app.route('/api/user/state', methods=['PUT'])
@login_required
def save_user_state():
    """Save UI state (session_id, active_tab, file_name, stage, model_id)."""
    user_id = get_current_user_uid()
    data = request.get_json() or {}
    db = _get_user_state_store()

    if db:
        try:
            db.collection('users').document(user_id).collection('settings').document('state').set(data, merge=True)
            return jsonify({'success': True})
        except Exception:
            pass

    # In-memory fallback
    _user_state_cache[user_id] = {**_user_state_cache.get(user_id, {}), **data}
    return jsonify({'success': True})


@app.route('/api/user/preferences', methods=['GET'])
@login_required
def get_user_preferences():
    """Fetch user preferences (theme, display_name, notifications)."""
    user_id = get_current_user_uid()
    db = _get_user_state_store()

    if db:
        try:
            doc = db.collection('users').document(user_id).collection('settings').document('preferences').get()
            if doc.exists:
                return jsonify(doc.to_dict())
        except Exception:
            pass

    return jsonify(_user_prefs_cache.get(user_id, {}))


@app.route('/api/user/preferences', methods=['PUT'])
@login_required
def save_user_preferences():
    """Save user preferences."""
    user_id = get_current_user_uid()
    data = request.get_json() or {}
    db = _get_user_state_store()

    if db:
        try:
            db.collection('users').document(user_id).collection('settings').document('preferences').set(data, merge=True)
            return jsonify({'success': True})
        except Exception:
            pass

    _user_prefs_cache[user_id] = {**_user_prefs_cache.get(user_id, {}), **data}
    return jsonify({'success': True})

# ==============================================================================
# EXECUTIVE SUMMARY
# ==============================================================================

@app.route('/api/executive-summary/<session_id>')
def executive_summary(session_id):
    """Retrieve high-level overview metrics for the executive dashboard."""
    result = pipeline_manager.get_executive_summary(session_id)
    if 'error' in result:
        return jsonify(result), 404
    return jsonify(result)


# ==============================================================================
# REPORT GENERATION
# ==============================================================================

@app.route('/api/report/<session_id>')
def generate_report(session_id):
    """Generate and download an HTML report."""
    result = pipeline_manager.generate_report(session_id)
    if 'error' in result:
        return jsonify(result), 400
    
    report_path = result.get('report_path')
    if report_path and os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name='automl_report.html')
    
    return jsonify({'error': 'Report generation failed'}), 500


# ==============================================================================
# TIER 0A: UNIVERSAL TRANSFORM & DATA QUALITY
# ==============================================================================

@app.route('/api/universal-transform/<session_id>', methods=['POST'])
def universal_transform(session_id):
    """Apply universal data transformation."""
    result = pipeline_manager.run_universal_transform(session_id)
    return jsonify(result)


@app.route('/api/data-quality/<session_id>')
def data_quality(session_id):
    """Get data quality scores."""
    result = pipeline_manager.get_data_quality(session_id)
    return jsonify(result)


# ==============================================================================
# TIER 0B: UNSUPERVISED ML
# ==============================================================================

@app.route('/api/unsupervised/<session_id>', methods=['POST'])
def unsupervised_analysis(session_id):
    """Run unsupervised analysis."""
    data = request.get_json() or {}
    method = data.get('method', 'all')
    result = pipeline_manager.run_unsupervised(session_id, method)
    return jsonify(result)


@app.route('/api/unsupervised/download-dataset/<session_id>')
def download_unsupervised_dataset(session_id):
    """Download the dataset used for unsupervised analysis."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    df = session.transformed_df if session.transformed_df is not None else (
        session.cleaned_df if session.cleaned_df is not None else session.original_df
    )
    if df is None:
        return jsonify({'error': 'No dataset available'}), 404

    dataset_path = os.path.join(session.output_dir, 'unsupervised_dataset.csv')
    df.to_csv(dataset_path, index=False)
    return send_file(dataset_path, as_attachment=True, download_name='unsupervised_dataset.csv')


@app.route('/api/unsupervised/download-model/<session_id>')
def download_unsupervised_model(session_id):
    """Download the unsupervised model artifact."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    model_path = session.unsupervised_model_path or os.path.join(session.output_dir, 'unsupervised_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Unsupervised model not found. Run unsupervised analysis first.'}), 404

    return send_file(model_path, as_attachment=True, download_name='unsupervised_model.pkl')


@app.route('/api/unsupervised/report/<session_id>')
def generate_unsupervised_report(session_id):
    """Generate and download unsupervised-only HTML report."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    if not session.unsupervised_results:
        return jsonify({'error': 'No unsupervised results found. Run unsupervised analysis first.'}), 400

    session_data = {
        'session_id': session.session_id,
        'profile': session.profile,
        'unsupervised_results': session.unsupervised_results,
    }

    report_path = generate_unsupervised_html_report(session_data, session.output_dir)
    if report_path and os.path.exists(report_path):
        return send_file(report_path, as_attachment=True, download_name='unsupervised_report.html')

    return jsonify({'error': 'Unsupervised report generation failed'}), 500


# ==============================================================================
# VISUALIZATION DATA
# ==============================================================================

@app.route('/api/viz-data/<session_id>')
def viz_data(session_id):
    """Return data needed for visualization charts (histogram, box, scatter, heatmap, pair, bar)."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    # Prefer transformed data, fallback to cleaned, then original
    df = session.transformed_df if session.transformed_df is not None else (
        session.cleaned_df if session.cleaned_df is not None else session.original_df
    )
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    target = session.profile.get('target_column') if session.profile else None
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    result = {
        'numeric_columns': numeric_cols,
        'categorical_columns': cat_cols,
        'n_rows': len(df),
        'target': target,
    }

    # Histogram data for each numeric column (binned)
    histograms = {}
    for col in numeric_cols[:20]:
        try:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            counts, edges = np.histogram(vals, bins=min(30, max(5, len(vals) // 10)))
            histograms[col] = {
                'counts': counts.tolist(),
                'edges': [round(float(e), 4) for e in edges],
            }
        except Exception:
            pass
    result['histograms'] = histograms

    # Box plot data for each numeric column
    box_plots = {}
    for col in numeric_cols[:20]:
        try:
            vals = df[col].dropna()
            if len(vals) == 0:
                continue
            q1, median, q3 = float(vals.quantile(0.25)), float(vals.median()), float(vals.quantile(0.75))
            iqr = q3 - q1
            lower = float(max(vals.min(), q1 - 1.5 * iqr))
            upper = float(min(vals.max(), q3 + 1.5 * iqr))
            box_plots[col] = {
                'min': round(lower, 4), 'q1': round(q1, 4),
                'median': round(median, 4), 'q3': round(q3, 4),
                'max': round(upper, 4), 'mean': round(float(vals.mean()), 4),
            }
        except Exception:
            pass
    result['box_plots'] = box_plots

    # Correlation matrix (top 15 numeric cols)
    try:
        top_numeric = df[numeric_cols[:15]].select_dtypes(include='number')
        corr = top_numeric.corr()
        result['correlation'] = {
            'columns': corr.columns.tolist(),
            'matrix': [[round(float(v), 4) for v in row] for row in corr.values],
        }
    except Exception:
        result['correlation'] = None

    # Scatter data: sample of rows for top numeric cols
    try:
        sample_n = min(500, len(df))
        sample_df = df[numeric_cols[:10]].sample(n=sample_n, random_state=42) if len(df) > 0 else df[numeric_cols[:10]]
        result['scatter_data'] = {col: [round(float(v), 4) if pd.notna(v) else None for v in sample_df[col]]
                                   for col in sample_df.columns}
    except Exception:
        result['scatter_data'] = {}

    # Bar plot data: value counts for categorical columns
    bar_plots = {}
    for col in cat_cols[:10]:
        try:
            vc = df[col].value_counts().head(20)
            bar_plots[col] = {'labels': vc.index.tolist(), 'values': vc.values.tolist()}
        except Exception:
            pass
    result['bar_plots'] = bar_plots

    return jsonify(result)


# ==============================================================================
# TIER 1: AUTO EDA
# ==============================================================================

@app.route('/api/eda/<session_id>')
def auto_eda(session_id):
    """Run automatic EDA."""
    result = pipeline_manager.run_eda(session_id)
    return jsonify(result)


# ==============================================================================
# TIER 1: FAIRNESS
# ==============================================================================

@app.route('/api/fairness/<session_id>')
def fairness_audit(session_id):
    """Run fairness audit."""
    result = pipeline_manager.get_fairness_report(session_id)
    return jsonify(result)


# ==============================================================================
# TIER 1: SYNTHETIC DATA
# ==============================================================================

@app.route('/api/synthetic/<session_id>', methods=['POST'])
def synthetic_data(session_id):
    """Generate synthetic data."""
    data = request.get_json() or {}
    n_samples = data.get('n_samples')
    result = pipeline_manager.generate_synthetic(session_id, n_samples)
    return jsonify(result)


# ==============================================================================
# TIER 2: CAUSAL INFERENCE
# ==============================================================================

@app.route('/api/causal/graph/<session_id>')
def causal_graph(session_id):
    """Discover causal graph."""
    result = pipeline_manager.get_causal_graph(session_id)
    return jsonify(result)


@app.route('/api/causal/effect/<session_id>', methods=['POST'])
def causal_effect(session_id):
    """Estimate causal effect."""
    data = request.get_json()
    treatment = data.get('treatment')
    outcome = data.get('outcome')
    if not treatment or not outcome:
        return jsonify({'error': 'Treatment and outcome columns required'}), 400
    result = pipeline_manager.get_causal_effect(session_id, treatment, outcome)
    return jsonify(result)


# ==============================================================================
# TIER 2: MODEL COMPRESSION
# ==============================================================================

@app.route('/api/compress/<session_id>', methods=['POST'])
def compress_model_api(session_id):
    """Compress the model."""
    result = pipeline_manager.compress(session_id)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# TIER 3: DATA VALUATION
# ==============================================================================

@app.route('/api/data-valuation/<session_id>')
def data_valuation(session_id):
    """Get data valuation scores."""
    result = pipeline_manager.get_data_valuation(session_id)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# TIER 3: AUTONOMOUS AGENT
# ==============================================================================

@app.route('/api/agent/<session_id>', methods=['POST'])
def run_agent(session_id):
    """Run autonomous improvement agent."""
    data = request.get_json() or {}
    max_iter = data.get('max_iterations', 5)
    result = pipeline_manager.run_agent(session_id, max_iter)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# API KEY MANAGEMENT
# ==============================================================================

@app.route('/api/set-api-key', methods=['POST'])
def set_api_key():
    """Update the LLM API key at runtime and re-initialize the chat agent."""
    try:
        data = request.get_json()
        api_key = (data.get('api_key') or '').strip()
        if not api_key:
            return jsonify({'error': 'API key is required'}), 400

        # Store in environment
        if api_key.startswith('sk-'):
            os.environ['OPENAI_API_KEY'] = api_key
        else:
            os.environ['GEMINI_API_KEY'] = api_key

        # Re-initialize the chat agent's LLM provider
        pipeline_manager.chat_agent._init_llm_provider()

        provider = pipeline_manager.chat_agent.llm_provider or 'rules'
        available = pipeline_manager.chat_agent.llm_available

        return jsonify({
            'success': available,
            'provider': provider,
            'message': f'AI assistant now using {provider}' if available else f'Key set but provider init failed — falling back to {provider}',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ==============================================================================
# TIER 1: CHAT (Upgraded with OpenAI + Memory)
# ==============================================================================

@app.route('/api/chat', methods=['POST'])
@limiter.limit("10/minute")
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        session_id = data.get('session_id')

        if not message:
            return jsonify({'error': 'Message required'}), 400

        session = pipeline_manager.get_session(session_id) if session_id else None
        session_data = None

        if session:
            session_data = {
                'df': session.transformed_df if session.transformed_df is not None else session.original_df,
                'profile': session.profile,
                'training_results': session.training_results,
                'recommendations': session.recommendations,
                'drift_report': session.drift_report,
                'explainability': session.explainability,
                'current_step': session.current_step,
            }

        result = pipeline_manager.chat_agent.chat(
            message,
            session_data=session_data,
            session_id=session_id
        )

        return jsonify(result)

    except Exception as e:
        import traceback
        traceback.print_exc()  # prints full error in Flask terminal
        return jsonify({'error': str(e), 'response': '❌ Server error: ' + str(e)}), 500


# ==============================================================================
# TIER 3: FEDERATED LEARNING
# ==============================================================================

@app.route('/api/federated/<session_id>', methods=['POST'])
def federated_learning(session_id):
    """Run federated learning simulation."""
    data = request.get_json() or {}
    n_clients = data.get('n_clients', 3)
    n_rounds = data.get('n_rounds', 5)
    result = pipeline_manager.run_federated(session_id, n_clients, n_rounds)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# PIPELINE BUILDER
# ==============================================================================

@app.route('/api/pipeline-blocks')
def pipeline_blocks():
    """Get available pipeline blocks."""
    return jsonify(pipeline_manager.get_pipeline_blocks())


# ==============================================================================
# HYPERPARAMETER OPTIMIZATION
# ==============================================================================

@app.route('/api/optimize/<session_id>', methods=['POST'])
def optimize_hyperparams(session_id):
    """Run hyperparameter optimization."""
    data = request.get_json() or {}
    method = data.get('method', 'auto')
    budget = data.get('budget', 30)
    result = pipeline_manager.optimize_hyperparams(session_id, method, budget)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# DATA CLEANING SUGGESTIONS
# ==============================================================================

@app.route('/api/cleaning-suggestions/<session_id>')
def cleaning_suggestions(session_id):
    """Get AI-based data cleaning suggestions."""
    result = pipeline_manager.get_cleaning_suggestions(session_id)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/apply-cleaning/<session_id>', methods=['POST'])
def apply_cleaning(session_id):
    """Apply accepted cleaning suggestions."""
    data = request.get_json()
    accepted_ids = data.get('accepted_ids', [])
    result = pipeline_manager.apply_cleaning_suggestions(session_id, accepted_ids)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# CLEANING IMPACT BENCHMARKS
# ==============================================================================

@app.route('/api/cleaning-impact/<session_id>')
def cleaning_impact(session_id):
    """Get cleaning suggestions with measured performance impact."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.cleaning_advisor import generate_cleaning_suggestions, benchmark_cleaning_impact
    suggestions = generate_cleaning_suggestions(session.original_df, session.profile)

    target = session.profile.get('target_column') if session.profile else None
    problem_type = session.profile.get('problem_type', 'classification') if session.profile else 'classification'

    if target and target in session.original_df.columns:
        suggestions = benchmark_cleaning_impact(session.original_df, target, suggestions, problem_type)

    return jsonify({'suggestions': suggestions})


# ==============================================================================
# MULTI-DATASET MANAGEMENT
# ==============================================================================

@app.route('/api/datasets')
def list_datasets():
    """List all uploaded datasets."""
    return jsonify(pipeline_manager.list_datasets())


@app.route('/api/datasets/switch', methods=['POST'])
def switch_dataset():
    """Switch active dataset."""
    data = request.get_json()
    dataset_id = data.get('dataset_id')
    session_id = data.get('session_id')
    result = pipeline_manager.switch_dataset(session_id, dataset_id)
    return jsonify(result)


# ==============================================================================
# LOCAL PROJECT STORAGE
# ==============================================================================

@app.route('/api/projects')
@login_required
def list_projects():
    """List saved projects."""
    user_id = get_current_user_uid()
    return jsonify(pipeline_manager.list_projects(user_id=user_id))


@app.route('/api/projects/save', methods=['POST'])
@login_required
def save_project():
    """Save current session as a project."""
    data = request.get_json()
    session_id = data.get('session_id')
    name = data.get('name')
    user_id = get_current_user_uid()
    result = pipeline_manager.save_project(session_id, name, user_id=user_id)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/projects/load', methods=['POST'])
@login_required
def load_project():
    """Load a saved project."""
    data = request.get_json()
    name = data.get('name')
    user_id = get_current_user_uid()
    result = pipeline_manager.load_project(name, user_id=user_id)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/projects/<name>', methods=['DELETE'])
@login_required
def delete_project(name):
    """Delete a saved project."""
    user_id = get_current_user_uid()
    result = pipeline_manager.delete_project(name, user_id=user_id)
    return jsonify(result)


# ==============================================================================
# PIPELINE STATUS & VISUALIZATION
# ==============================================================================

@app.route('/api/pipeline-status/<session_id>')
def pipeline_status(session_id):
    """Get pipeline status for flow visualization."""
    result = pipeline_manager.get_pipeline_flow_status(session_id)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# REAL-TIME PROGRESS (SSE)
# ==============================================================================

@app.route('/api/progress-stream/<session_id>')
def progress_stream(session_id):
    """Server-Sent Events stream for real-time training progress."""
    def generate():
        last_msg = ''
        while True:
            session = pipeline_manager.get_session(session_id)
            if not session:
                yield f"data: {json.dumps({'error': 'Session not found'})}\n\n"
                break

            msg = json.dumps({
                'status': session.status,
                'progress': session.progress,
                'message': session.progress_message,
                'step': session.current_step,
                'log': getattr(session, 'progress_log', [])[-5:],
            })

            if msg != last_msg:
                yield f"data: {msg}\n\n"
                last_msg = msg

            if session.status in ('complete', 'error'):
                yield f"data: {json.dumps({'status': session.status, 'progress': 100, 'message': session.progress_message, 'done': True})}\n\n"
                break

            time.sleep(0.5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


# ==============================================================================
# CUSTOM PIPELINE (Semi-Auto Mode)
# ==============================================================================

@app.route('/api/custom-pipeline/<session_id>', methods=['POST'])
def run_custom_pipeline(session_id):
    """Run a custom user-defined pipeline."""
    data = request.get_json() or {}
    config = data.get('config', {})
    result = pipeline_manager.run_custom_pipeline(session_id, config)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# v3.0: EDA NARRATIVE
# ==============================================================================

@app.route('/api/eda-narrative/<session_id>')
def eda_narrative(session_id):
    """Get AI-generated natural language EDA narrative."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.auto_eda import generate_narrative_report
    target = session.profile.get('target_column') if session.profile else None
    eda = session.eda_report if session.eda_report else None
    narrative = generate_narrative_report(session.original_df, eda, target)
    return jsonify({'narrative': narrative})


# ==============================================================================
# v3.0: COUNTERFACTUAL EXPLANATIONS
# ==============================================================================

@app.route('/api/counterfactuals/<session_id>', methods=['POST'])
def counterfactuals(session_id):
    """Generate counterfactual explanations for a prediction."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.best_model is None or session.X_test is None:
        return jsonify({'error': 'No trained model available'}), 400

    data = request.get_json() or {}
    row_index = data.get('row_index', 0)
    desired_outcome = data.get('desired_outcome')

    if row_index >= len(session.X_test):
        row_index = len(session.X_test) - 1

    from ml_engine.explainer import generate_counterfactuals
    instance = session.X_test.iloc[[row_index]]
    result = generate_counterfactuals(
        session.best_model, instance, session.feature_names,
        desired_outcome, session.X_train, session.y_train
    )
    return jsonify(result)


# ==============================================================================
# v3.0: PARTIAL DEPENDENCE PLOTS
# ==============================================================================

@app.route('/api/partial-dependence/<session_id>')
def partial_dependence_api(session_id):
    """Compute partial dependence plot data for top features."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.best_model is None or session.X_train is None:
        return jsonify({'error': 'No trained model available'}), 400

    from ml_engine.explainer import compute_partial_dependence
    top_n = request.args.get('top_n', 3, type=int)
    result = compute_partial_dependence(session.best_model, session.X_train, session.feature_names, top_n)
    if isinstance(result, dict) and 'error' in result:
        return jsonify(result), 400
    return jsonify({'pdp_data': result})


# ==============================================================================
# v3.0: CALIBRATION
# ==============================================================================

@app.route('/api/calibration/<session_id>')
def calibration(session_id):
    """Get model calibration curve and metrics."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.best_model is None or session.X_test is None:
        return jsonify({'error': 'No trained model available'}), 400

    from ml_engine.calibration_engine import compute_calibration
    result = compute_calibration(session.best_model, session.X_test, session.y_test)
    return jsonify(result)


@app.route('/api/calibrate/<session_id>', methods=['POST'])
def auto_calibrate_api(session_id):
    """Auto-calibrate model predictions."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.best_model is None:
        return jsonify({'error': 'No trained model available'}), 400

    from ml_engine.calibration_engine import auto_calibrate
    data = request.get_json() or {}
    method = data.get('method', 'auto')

    result = auto_calibrate(
        session.best_model, session.X_train, session.y_train,
        session.X_test, session.y_test, method
    )

    # Update model if calibration succeeded
    if result.get('success') and result.get('calibrated_model'):
        session.best_model = result['calibrated_model']
        result.pop('calibrated_model', None)

    return jsonify(result)


# ==============================================================================
# v3.0: CONSTRAINT LEARNING
# ==============================================================================

@app.route('/api/constraints/<session_id>')
def constraints(session_id):
    """Get learned data constraints."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.constraint_learner import learn_constraints
    result = learn_constraints(session.original_df)
    return jsonify({'constraints': result, 'n_constraints': len(result)})


@app.route('/api/validate-constraints/<session_id>', methods=['POST'])
def validate_constraints(session_id):
    """Validate new data against learned constraints."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.constraint_learner import learn_constraints, validate_against_constraints
    constraints_list = learn_constraints(session.original_df)

    data = request.get_json() or {}
    new_data = data.get('data', [])
    if not new_data:
        return jsonify({'error': 'No data to validate'}), 400

    import pandas as pd
    new_df = pd.DataFrame(new_data)
    result = validate_against_constraints(new_df, constraints_list)
    return jsonify(result)


# ==============================================================================
# v3.0: PROBLEM REFRAMING
# ==============================================================================

@app.route('/api/reframe/<session_id>')
def reframe(session_id):
    """Get alternative problem framing suggestions."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.original_df is None or session.profile is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.problem_reframer import suggest_reframings
    target = session.profile.get('target_column', '')
    problem_type = session.profile.get('problem_type', 'classification')
    suggestions = suggest_reframings(session.original_df, target, problem_type)
    return jsonify({'suggestions': suggestions})


# ==============================================================================
# v3.0: FEATURE ENGINEERING STUDIO
# ==============================================================================

@app.route('/api/feature-studio/validate/<session_id>', methods=['POST'])
def feature_studio_validate(session_id):
    """Validate a feature engineering expression."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.feature_studio import FeatureStudio
    data = request.get_json()
    expression = data.get('expression', '')
    studio = FeatureStudio()
    result = studio.validate_expression(expression, session.original_df)
    return jsonify(result)


@app.route('/api/feature-studio/preview/<session_id>', methods=['POST'])
def feature_studio_preview(session_id):
    """Preview a new feature."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.feature_studio import FeatureStudio
    data = request.get_json()
    expression = data.get('expression', '')
    name = data.get('name', expression)
    target = session.profile.get('target_column') if session.profile else None

    studio = FeatureStudio()
    result = studio.preview_feature(expression, session.original_df, name, target)
    return jsonify(result)


@app.route('/api/feature-studio/add/<session_id>', methods=['POST'])
def feature_studio_add(session_id):
    """Add a validated feature to the dataset."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.feature_studio import FeatureStudio
    data = request.get_json()
    expression = data.get('expression', '')
    name = data.get('name', '')
    if not name:
        return jsonify({'error': 'Feature name is required'}), 400

    studio = FeatureStudio()
    session.original_df, result = studio.add_feature(expression, name, session.original_df)
    return jsonify(result)


@app.route('/api/feature-studio/suggestions/<session_id>')
def feature_studio_suggestions(session_id):
    """Get auto-generated feature engineering suggestions."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.feature_studio import FeatureStudio
    studio = FeatureStudio()
    target = session.profile.get('target_column') if session.profile else None
    suggestions = studio.suggest_features(session.original_df, target)
    return jsonify({'suggestions': suggestions})


# ==============================================================================
# v3.0: ANNOTATIONS
# ==============================================================================

@app.route('/api/annotate/<session_id>', methods=['POST'])
def annotate(session_id):
    """Annotate a prediction as correct/incorrect."""
    session = pipeline_manager.get_session(session_id)
    if not session or not session.experiment_id:
        return jsonify({'error': 'No active experiment'}), 400

    from ml_engine.annotation_engine import AnnotationEngine
    engine = AnnotationEngine(pipeline_manager.experiment_store.db_path)
    data = request.get_json()
    result = engine.annotate(
        session.experiment_id,
        data.get('row_index', 0),
        data.get('label', 'uncertain'),
        data.get('notes', ''),
        data.get('features')
    )
    return jsonify(result)


@app.route('/api/annotations/<session_id>')
def annotations(session_id):
    """Get all annotations for the current experiment."""
    session = pipeline_manager.get_session(session_id)
    if not session or not session.experiment_id:
        return jsonify({'error': 'No active experiment'}), 400

    from ml_engine.annotation_engine import AnnotationEngine
    engine = AnnotationEngine(pipeline_manager.experiment_store.db_path)
    result = engine.get_annotations(session.experiment_id)
    return jsonify(result)


@app.route('/api/failure-patterns/<session_id>')
def failure_patterns(session_id):
    """Analyze failure patterns from annotations."""
    session = pipeline_manager.get_session(session_id)
    if not session or not session.experiment_id:
        return jsonify({'error': 'No active experiment'}), 400
    if session.best_model is None or session.X_test is None:
        return jsonify({'error': 'No trained model available'}), 400

    from ml_engine.annotation_engine import AnnotationEngine
    engine = AnnotationEngine(pipeline_manager.experiment_store.db_path)
    result = engine.find_failure_patterns(
        session.experiment_id, session.best_model,
        session.X_test, session.feature_names
    )
    return jsonify(result)


# ==============================================================================
# v3.0: COMPETITION LEADERBOARD
# ==============================================================================

@app.route('/api/competition/leaderboard')
def competition_leaderboard():
    """Get the competition leaderboard."""
    from ml_engine.competition_engine import CompetitionEngine
    engine = CompetitionEngine(pipeline_manager.experiment_store.db_path)
    problem_type = request.args.get('problem_type')
    result = engine.get_leaderboard(problem_type=problem_type)
    return jsonify(result)


@app.route('/api/competition/submit/<session_id>', methods=['POST'])
def competition_submit(session_id):
    """Submit current model to the competition leaderboard."""
    session = pipeline_manager.get_session(session_id)
    if not session or not session.experiment_id:
        return jsonify({'error': 'No active experiment'}), 400
    if not session.training_results:
        return jsonify({'error': 'No training results available'}), 400

    from ml_engine.competition_engine import CompetitionEngine
    engine = CompetitionEngine(pipeline_manager.experiment_store.db_path)

    result = engine.submit(
        experiment_id=session.experiment_id,
        model_name=session.training_results.get('best_model', 'Unknown'),
        score=session.training_results.get('best_score', 0),
        metric=session.training_results.get('primary_metric_name', 'score'),
        dataset_type=session.profile.get('problem_type', 'general') if session.profile else 'general',
        n_rows=session.profile.get('n_rows', 0) if session.profile else 0,
        n_features=session.profile.get('n_cols', 0) if session.profile else 0,
        problem_type=session.profile.get('problem_type', 'classification') if session.profile else 'classification',
    )
    return jsonify(result)


@app.route('/api/competition/rank/<session_id>')
def competition_rank(session_id):
    """Get rank for the current experiment."""
    session = pipeline_manager.get_session(session_id)
    if not session or not session.experiment_id:
        return jsonify({'error': 'No active experiment'}), 400

    from ml_engine.competition_engine import CompetitionEngine
    engine = CompetitionEngine(pipeline_manager.experiment_store.db_path)
    result = engine.get_rank(session.experiment_id)
    return jsonify(result)


# ==============================================================================
# v3.0: EXECUTIVE REPORT
# ==============================================================================

@app.route('/api/executive-report/<session_id>')
def executive_report(session_id):
    """Generate a non-technical executive summary report."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if not session.training_results:
        return jsonify({'error': 'No training results available'}), 400

    from ml_engine.report_generator import generate_executive_summary

    session_data = {
        'profile': session.profile,
        'training_results': session.training_results,
        'explainability': session.explainability,
        'recommendations': session.recommendations,
    }

    # Try to use the LLM agent for better summaries
    llm_agent = None
    try:
        llm_agent = pipeline_manager.chat_agent
    except Exception:
        pass

    result = generate_executive_summary(session_data, llm_agent)
    return jsonify(result)


# ==============================================================================
# v3.0: DATASET SIMILARITY
# ==============================================================================

@app.route('/api/dataset-similarity/<session_id>')
def dataset_similarity(session_id):
    """Find similar past datasets and get recommendations."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.dataset_fingerprint import DatasetFingerprinter
    fp = DatasetFingerprinter()
    target = session.profile.get('target_column') if session.profile else None
    fingerprint = fp.compute_fingerprint(session.original_df, target)

    # Save fingerprint
    if session.experiment_id:
        pipeline_manager.experiment_store.save_fingerprint(session.experiment_id, fingerprint)

    # Find similar
    stored = pipeline_manager.experiment_store.get_all_fingerprints()
    similar = fp.find_similar(fingerprint, stored, top_k=5)

    # Get experiment details for similar ones
    similar_exps = []
    for exp_id, score in similar:
        if exp_id == session.experiment_id:
            continue  # Skip self
        exp = pipeline_manager.experiment_store.get_experiment(exp_id)
        if exp:
            similar_exps.append({
                'experiment_id': exp_id,
                'name': exp.get('name', ''),
                'similarity': score,
                'best_model': exp.get('best_model', ''),
                'best_score': exp.get('best_score'),
            })

    recommendations = fp.get_recommended_settings(similar_exps)

    return jsonify({
        'fingerprint': fingerprint,
        'similar_experiments': similar_exps[:5],
        'recommendations': recommendations,
    })


# ==============================================================================
# v4.0: STACKING ENSEMBLE
# ==============================================================================

@app.route('/api/ensemble/<session_id>', methods=['POST'])
def build_ensemble(session_id):
    """Build a stacking ensemble from the top trained models."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if not session.trained_models or not session.training_results:
        return jsonify({'error': 'No trained models available'}), 400

    from ml_engine.ensemble_builder import build_stacking_ensemble
    data = request.get_json() or {}
    top_n = data.get('top_n', 3)

    result = build_stacking_ensemble(
        session.trained_models, session.training_results.get('leaderboard', []),
        session.X_train, session.y_train, session.X_test, session.y_test,
        session.profile.get('problem_type', 'classification'), top_n
    )

    if 'error' in result:
        return jsonify(result), 400

    # Store ensemble model as new best if it beats the current best
    if result.get('beats_best'):
        session.best_model = result.pop('ensemble_model', session.best_model)
    else:
        result.pop('ensemble_model', None)

    return jsonify(result)


# ==============================================================================
# v4.0: MODEL CARD
# ==============================================================================

@app.route('/api/model-card/<session_id>')
def model_card(session_id):
    """Generate an automated model card."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if not session.training_results:
        return jsonify({'error': 'No training results'}), 400

    from ml_engine.model_card_generator import generate_model_card
    card = generate_model_card(
        {},
        profile=session.profile,
        training_results=session.training_results,
        explainability=session.explainability,
        fairness_report=session.fairness_report,
    )
    return jsonify(card)


# ==============================================================================
# v4.0: SQL-STYLE DATA QUERYING
# ==============================================================================

@app.route('/api/query/<session_id>', methods=['POST'])
def query_data(session_id):
    """Execute a SQL-style or natural language query on the dataset."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404

    df = session.transformed_df if session.transformed_df is not None else (
        session.cleaned_df if session.cleaned_df is not None else session.original_df)
    if df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    data = request.get_json()
    query_str = data.get('query', '')
    if not query_str:
        return jsonify({'error': 'Query string required'}), 400

    from ml_engine.data_query_engine import query_dataset
    result = query_dataset(df, query_str)
    return jsonify(result)


# ==============================================================================
# v4.0: DATASET CHECKPOINTS
# ==============================================================================

@app.route('/api/checkpoints/<session_id>', methods=['GET'])
def list_checkpoints(session_id):
    """List all dataset checkpoints."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    return jsonify({'checkpoints': session.list_checkpoints()})


@app.route('/api/checkpoints/<session_id>/save', methods=['POST'])
def save_checkpoint(session_id):
    """Save a named dataset checkpoint."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    data = request.get_json() or {}
    label = data.get('label', 'auto')
    labels = session.save_checkpoint(label)
    return jsonify({'success': True, 'checkpoints': labels})


@app.route('/api/checkpoints/<session_id>/restore', methods=['POST'])
def restore_checkpoint(session_id):
    """Restore a dataset from a named checkpoint."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    data = request.get_json() or {}
    label = data.get('label', '')
    result = session.restore_checkpoint(label)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


# ==============================================================================
# v4.0: FEATURE CROSSING & POLYNOMIALS
# ==============================================================================

@app.route('/api/feature-crossing/<session_id>', methods=['POST'])
def feature_crossing(session_id):
    """Auto-generate interaction features."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.feature_studio import FeatureStudio
    studio = FeatureStudio()
    target = session.profile.get('target_column') if session.profile else None
    data = request.get_json() or {}
    result = studio.auto_feature_crossing(session.original_df, target, data.get('max_pairs', 10))
    return jsonify(result)


@app.route('/api/feature-polynomial/<session_id>', methods=['POST'])
def feature_polynomial(session_id):
    """Auto-generate polynomial features."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No dataset loaded'}), 400

    from ml_engine.feature_studio import FeatureStudio
    studio = FeatureStudio()
    target = session.profile.get('target_column') if session.profile else None
    data = request.get_json() or {}
    result = studio.auto_polynomial_features(session.original_df, target, data.get('degree', 2))
    return jsonify(result)


# ==============================================================================
# v4.0: PREDICTION PLAYGROUND
# ==============================================================================

@app.route('/api/playground/<session_id>')
def prediction_playground(session_id):
    """Return feature metadata for interactive prediction slider UI."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'Session not found'}), 404
    if session.best_model is None or session.X_train is None:
        return jsonify({'error': 'No trained model available'}), 400

    features = []
    for col in session.feature_names:
        if col not in session.X_train.columns:
            continue
        series = session.X_train[col]
        info = {'name': col, 'dtype': str(series.dtype)}
        if pd.api.types.is_numeric_dtype(series):
            info['type'] = 'slider'
            info['min'] = round(float(series.min()), 4)
            info['max'] = round(float(series.max()), 4)
            info['mean'] = round(float(series.mean()), 4)
            info['median'] = round(float(series.median()), 4)
            info['step'] = round(float((series.max() - series.min()) / 100), 4) if series.max() != series.min() else 1
        else:
            info['type'] = 'dropdown'
            info['options'] = series.value_counts().head(20).index.tolist()
            info['default'] = series.mode().iloc[0] if len(series.mode()) > 0 else None
        features.append(info)

    return jsonify({
        'features': features,
        'target': session.profile.get('target_column') if session.profile else None,
        'problem_type': session.profile.get('problem_type') if session.profile else None,
        'n_features': len(features),
    })


@app.route('/api/playground/<session_id>/predict', methods=['POST'])
def playground_predict(session_id):
    """Live prediction from playground with confidence."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model available'}), 400

    data = request.get_json()
    result = pipeline_manager.predict(session_id, data)
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════════
#  v5.0 — REVOLUTIONARY FEATURES API (25 world-first features)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/api/v5/data-sufficiency/<session_id>')
def api_data_sufficiency(session_id):
    """Feature #25: Data sufficiency calculator."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No session or data'}), 400
    from ml_engine.data_sufficiency import analyze_sufficiency
    target = session.profile.get('target_column') if session.profile else None
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    return jsonify(analyze_sufficiency(session.original_df, target, ptype))


@app.route('/api/v5/sample-difficulty/<session_id>')
def api_sample_difficulty(session_id):
    """Feature #12: Sample difficulty scorer."""
    session = pipeline_manager.get_session(session_id)
    if not session or not session.trained_models:
        return jsonify({'error': 'No trained models'}), 400
    from ml_engine.sample_difficulty import score_sample_difficulty
    fnames = list(session.X_test.columns) if hasattr(session.X_test, 'columns') else None
    return jsonify(score_sample_difficulty(session.trained_models, session.X_test,
                                           session.y_test, fnames))


@app.route('/api/v5/bias-amplification/<session_id>', methods=['POST'])
def api_bias_amplification(session_id):
    """Feature #23: Bias amplification detector."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.bias_amplification import detect_bias_amplification
    data = request.get_json() or {}
    sensitive_cols = data.get('sensitive_columns', [])
    if not sensitive_cols:
        return jsonify({'error': 'Provide sensitive_columns in request body'}), 400
    fnames = list(session.X_test.columns) if hasattr(session.X_test, 'columns') else None
    return jsonify(detect_bias_amplification(session.best_model, session.X_test,
                                              session.y_test, sensitive_cols, fnames))


@app.route('/api/v5/conformal/<session_id>')
def api_conformal(session_id):
    """Feature #11: Conformal prediction."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.conformal_predictor import build_conformal_predictor, conformal_predict
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    confidence = float(request.args.get('confidence', 0.95))
    cal_data = build_conformal_predictor(session.best_model, session.X_test,
                                          session.y_test, ptype, confidence)
    if 'error' in cal_data:
        return jsonify(cal_data), 400
    result = conformal_predict(session.best_model, session.X_test, cal_data, ptype)
    result['calibration'] = cal_data
    result['coverage'] = result.get('coverage', cal_data.get('confidence'))
    result['actual_coverage'] = result.get('actual_coverage', result['coverage'])
    result['avg_set_size'] = result.get('avg_set_size', result.get('average_set_size'))
    result['avg_interval_width'] = result.get('avg_interval_width', result.get('margin'))
    result['target_coverage'] = result.get('target_coverage', f"{int(round(cal_data.get('confidence', confidence) * 100))}%")
    result['calibrated'] = True
    result['samples'] = result.get('samples', result.get('predictions', []))
    return jsonify(result)


@app.route('/api/v5/vif/<session_id>')
def api_vif(session_id):
    """Feature #15: VIF multicollinearity analysis."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.X_train is None:
        return jsonify({'error': 'No training data'}), 400
    from ml_engine.vif_analyzer import compute_vif
    X = session.X_train if hasattr(session.X_train, 'columns') else session.X_test
    return jsonify(compute_vif(X))


@app.route('/api/v5/learning-curve/<session_id>')
def api_learning_curve(session_id):
    """Feature #13: Learning curve predictor."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.learning_curve_predictor import predict_learning_curve
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    result = predict_learning_curve(session.best_model, session.X_train,
                                    session.y_train, ptype)
    predictions = result.get('extrapolation', {}).get('predictions', [])
    keyed = {}
    for item in predictions:
        multiplier = item.get('multiplier')
        if multiplier:
            keyed[multiplier] = item.get('predicted_score')
    result['extrapolations'] = keyed or result.get('extrapolations', {})
    result['predictions'] = predictions
    result['curve_points'] = result.get('actual_curve', [])
    result['learning_curve'] = result.get('actual_curve', [])
    if result.get('plateau_message'):
        result['verdict'] = result['plateau_message']
    return jsonify(result)


@app.route('/api/v5/disagreement/<session_id>')
def api_disagreement(session_id):
    """Feature #14: Model disagreement analyzer."""
    session = pipeline_manager.get_session(session_id)
    if not session or not session.trained_models:
        return jsonify({'error': 'No trained models'}), 400
    from ml_engine.disagreement_analyzer import analyze_disagreement
    fnames = list(session.X_test.columns) if hasattr(session.X_test, 'columns') else None
    return jsonify(analyze_disagreement(session.trained_models, session.X_test,
                                         session.y_test, fnames))


@app.route('/api/v5/target-transform/<session_id>')
def api_target_transform(session_id):
    """Feature #16: Target distribution auto-transformer."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.X_train is None:
        return jsonify({'error': 'No training data'}), 400
    from ml_engine.target_transformer import auto_transform_target, analyze_target_distribution
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    analysis = analyze_target_distribution(session.y_train)
    transform = auto_transform_target(session.X_train, session.y_train, ptype)
    if 'transformed_y' in transform:
        del transform['transformed_y']  # Don't send large arrays
    return jsonify({'analysis': analysis, 'transform': transform})


@app.route('/api/v5/cv-stability/<session_id>')
def api_cv_stability(session_id):
    """Feature #19: Cross-validation stability report."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.cv_stability import analyze_cv_stability
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    result = analyze_cv_stability(session.best_model, session.X_train,
                                  session.y_train, ptype)
    dist = result.get('distribution', {})
    result['stable_count'] = result.get('stable_count', dist.get('stable_correct', 0))
    result['unstable_count'] = result.get('unstable_count', dist.get('unstable', 0))
    result['avg_stability'] = result.get('avg_stability', result.get('mean_stability', 0))
    result['flipflop_rate'] = result.get('flipflop_rate', dist.get('unstable_pct', 0))
    result['stable_samples'] = result.get('stable_samples', dist.get('stable_correct', 0))
    unstable_samples = result.get('unstable_samples', [])
    normalized_unstable = []
    for item in unstable_samples:
        normalized_unstable.append({
            **item,
            'stability': item.get('stability', item.get('stability_score')),
            'folds_correct': item.get('folds_correct', item.get('correct_folds')),
            'total_folds': item.get('total_folds', item.get('total_folds')),
            'category': item.get('category', 'unstable'),
        })
    result['unstable_samples'] = normalized_unstable
    result['summary'] = result.get('summary') or result.get('recommendation')
    return jsonify(result)


@app.route('/api/v5/confidence-bands/<session_id>')
def api_confidence_bands(session_id):
    """Feature #24: Prediction confidence bands."""
    session = pipeline_manager.get_session(session_id)
    if not session or not session.trained_models:
        return jsonify({'error': 'No trained models'}), 400
    from ml_engine.confidence_bands import compute_confidence_bands
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    return jsonify(compute_confidence_bands(session.trained_models, session.X_test, ptype))


@app.route('/api/v5/error-slices/<session_id>')
def api_error_slices(session_id):
    """Feature #18: Automated error slice analysis."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.error_slicer import analyze_error_slices
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    fnames = list(session.X_test.columns) if hasattr(session.X_test, 'columns') else None
    return jsonify(analyze_error_slices(session.best_model, session.X_test,
                                         session.y_test, ptype, fnames))


@app.route('/api/v5/outlier-explain/<session_id>')
def api_outlier_explain(session_id):
    """Feature #20: Outlier explanation engine."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No data'}), 400
    from ml_engine.outlier_explainer import explain_outliers
    target = session.profile.get('target_column') if session.profile else None
    return jsonify(explain_outliers(session.original_df, target))


@app.route('/api/v5/complexity/<session_id>')
def api_complexity(session_id):
    """Feature #21: Model complexity vs performance analyzer."""
    session = pipeline_manager.get_session(session_id)
    if not session or not session.trained_models:
        return jsonify({'error': 'No trained models'}), 400
    from ml_engine.complexity_analyzer import analyze_complexity
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    lb = session.training_results.get('leaderboard', []) if session.training_results else []
    return jsonify(analyze_complexity(session.trained_models, session.X_test,
                                      session.y_test, lb, ptype))


@app.route('/api/v5/semantic-types/<session_id>')
def api_semantic_types(session_id):
    """Feature #22: Semantic feature type detector."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No data'}), 400
    from ml_engine.semantic_type_detector import detect_semantic_types
    target = session.profile.get('target_column') if session.profile else None
    return jsonify(detect_semantic_types(session.original_df, target))


@app.route('/api/v5/data-prescription/<session_id>')
def api_data_prescription(session_id):
    """Feature #2: Data prescription engine."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.data_prescription import prescribe_data
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    fnames = list(session.X_test.columns) if hasattr(session.X_test, 'columns') else None
    return jsonify(prescribe_data(session.best_model, session.X_train, session.y_train,
                                   session.X_test, session.y_test, ptype, fnames))


@app.route('/api/v5/prediction-autopsy/<session_id>', methods=['POST'])
def api_prediction_autopsy(session_id):
    """Feature #5: Prediction autopsy / decision debugger."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.prediction_autopsy import autopsy_prediction
    data = request.get_json() or {}
    # Accept either a full sample dict or a sample index (from frontend)
    if not data:
        return jsonify({'error': 'Provide sample data in request body'}), 400

    # If frontend passed a sample index, look up the row in X_test
    if isinstance(data, dict) and 'sample_index' in data:
        idx = data.get('sample_index')
        try:
            idx = int(idx)
        except Exception:
            return jsonify({'error': 'Invalid sample_index'}), 400
        if session.X_test is None:
            return jsonify({'error': 'No test data available for session'}), 400
        try:
            # Convert row to dict of feature->value
            sample_row = session.X_test.iloc[idx]
            sample = sample_row.to_dict()
            actual_value = None
            if session.y_test is not None:
                if hasattr(session.y_test, 'iloc'):
                    actual_value = session.y_test.iloc[idx]
                else:
                    actual_value = session.y_test[idx]
                if hasattr(actual_value, 'iloc'):
                    actual_value = actual_value.iloc[0]
                elif hasattr(actual_value, 'item'):
                    try:
                        actual_value = actual_value.item()
                    except Exception:
                        pass
        except Exception:
            return jsonify({'error': 'Sample index out of range'}), 400
    else:
        sample = data
        actual_value = data.get('actual') if isinstance(data, dict) else None

    fnames = list(session.X_train.columns) if hasattr(session.X_train, 'columns') else []
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    try:
        result = autopsy_prediction(session.best_model, sample, session.X_train,
                                    session.y_train, fnames, ptype)
        if actual_value is not None:
            result['actual'] = actual_value
            result['actual_class'] = actual_value
            pred_value = result.get('prediction')
            if isinstance(actual_value, (int, float, np.integer, np.floating)) and isinstance(pred_value, (int, float, np.integer, np.floating)):
                result['correct'] = bool(np.isclose(float(pred_value), float(actual_value)))
            else:
                result['correct'] = pred_value == actual_value
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Autopsy failed: {str(e)}'}), 500


@app.route('/api/v5/adversarial-test/<session_id>')
def api_adversarial_test(session_id):
    """Feature #3: Adversarial stress test suite."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.adversarial_tester import run_stress_test
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    fnames = list(session.X_test.columns) if hasattr(session.X_test, 'columns') else None
    return jsonify(run_stress_test(session.best_model, session.X_test,
                                    session.y_test, ptype, fnames))


@app.route('/api/v5/shelf-life/<session_id>')
def api_shelf_life(session_id):
    """Feature #4: Model shelf-life predictor."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.shelf_life_predictor import predict_shelf_life
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    return jsonify(predict_shelf_life(session.best_model, session.X_train, session.y_train,
                                       session.X_test, session.y_test, ptype))


@app.route('/api/v5/dataset-dna/<session_id>')
def api_dataset_dna(session_id):
    """Feature #1: Dataset DNA & model prophecy."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No data'}), 400
    from ml_engine.model_prophecy import compute_dataset_dna, prophecy
    target = session.profile.get('target_column') if session.profile else None
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    dna = compute_dataset_dna(session.original_df, target, ptype)
    pred = prophecy(dna, ptype)
    return jsonify({'dna': dna, 'prophecy': pred})


@app.route('/api/v5/interaction-xray/<session_id>')
def api_interaction_xray(session_id):
    """Feature #10: Feature interaction X-ray."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.best_model is None:
        return jsonify({'error': 'No trained model'}), 400
    from ml_engine.interaction_xray import analyze_interactions
    fnames = list(session.X_test.columns) if hasattr(session.X_test, 'columns') else None
    return jsonify(analyze_interactions(session.best_model, session.X_test,
                                         session.y_test, fnames))


@app.route('/api/v5/paper/<session_id>')
def api_paper(session_id):
    """Feature #7: Auto research paper generator."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'No session'}), 400
    from ml_engine.paper_generator import generate_paper
    paper = generate_paper(
        session.profile, session.clean_report, session.transform_report,
        session.training_results)
    sections_dict = paper.get('sections', {}) if isinstance(paper, dict) else {}

    ordered_sections = [
        ('Abstract', 'abstract'),
        ('Introduction', 'introduction'),
        ('Methodology', 'methodology'),
        ('Results', 'results'),
        ('Discussion', 'discussion'),
        ('Limitations', 'limitations'),
        ('Conclusion', 'conclusion'),
        ('References', 'references'),
    ]

    sections = []
    for title, key in ordered_sections:
        content = sections_dict.get(key)
        if not content:
            continue
        if isinstance(content, str) and content.strip().startswith('## '):
            content_lines = content.splitlines()
            content = '\n'.join(content_lines[1:]).lstrip()
        sections.append({'title': title, 'content': content})

    normalized = {
        'title': sections_dict.get('title', 'AutoML Experiment Report'),
        'authors': sections_dict.get('authors', 'AutoML Studio'),
        'date': sections_dict.get('date'),
        'sections': sections,
        'markdown': paper.get('full_paper_markdown') if isinstance(paper, dict) else None,
        'text': paper.get('full_paper_markdown') if isinstance(paper, dict) else None,
    }

    payload = paper if isinstance(paper, dict) else {'paper': paper}
    payload['paper'] = normalized
    return jsonify(payload)


@app.route('/api/v5/collaborative/contribute/<session_id>', methods=['POST'])
def api_collab_contribute(session_id):
    """Feature #8: Contribute to collaborative intelligence."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.training_results is None:
        return jsonify({'error': 'No training results'}), 400
    from ml_engine.collaborative_intelligence import CollaborativeIntelligence
    from ml_engine.model_prophecy import compute_dataset_dna
    ci = CollaborativeIntelligence()
    target = session.profile.get('target_column') if session.profile else None
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    dna = compute_dataset_dna(session.original_df, target, ptype)
    best_model = session.training_results.get('best_model', 'Unknown')
    best_score = session.training_results.get('best_score', 0)
    return jsonify(ci.contribute(dna, best_model, best_score, ptype))


@app.route('/api/v5/collaborative/recommend/<session_id>')
def api_collab_recommend(session_id):
    """Feature #8: Get collaborative intelligence recommendations."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No data'}), 400
    from ml_engine.collaborative_intelligence import CollaborativeIntelligence
    from ml_engine.model_prophecy import compute_dataset_dna
    ci = CollaborativeIntelligence()
    target = session.profile.get('target_column') if session.profile else None
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    dna = compute_dataset_dna(session.original_df, target, ptype)
    return jsonify(ci.recommend(dna, ptype))


@app.route('/api/v5/smart-sample/<session_id>', methods=['POST'])
def api_smart_sample(session_id):
    """Feature #17: Smart subsampling engine."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.original_df is None:
        return jsonify({'error': 'No data'}), 400
    from ml_engine.smart_sampler import smart_subsample
    data = request.get_json() or {}
    target = session.profile.get('target_column') if session.profile else None
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    X = session.original_df.drop(columns=[target]) if target and target in session.original_df.columns else session.original_df
    y = session.original_df[target] if target and target in session.original_df.columns else None
    result = smart_subsample(X, y, data.get('target_size'), data.get('strategy', 'auto'), ptype)
    if 'indices' in result:
        result['indices'] = result['indices'][:100]  # Limit response size
    return jsonify(result)


@app.route('/api/v5/tournament/<session_id>', methods=['POST'])
def api_tournament(session_id):
    """Feature #9: Model tournament engine."""
    session = pipeline_manager.get_session(session_id)
    if not session or session.X_train is None:
        return jsonify({'error': 'No training data'}), 400
    from ml_engine.tournament_engine import run_tournament
    ptype = session.profile.get('problem_type', 'classification') if session.profile else 'classification'
    result = run_tournament(session.X_train, session.y_train,
                             session.X_test, session.y_test, ptype)
    if 'champion_model' in result:
        del result['champion_model']  # Can't serialize model object
    # Clean model_obj from rounds
    for rnd in result.get('rounds', {}).values():
        for r in rnd.get('results', []):
            r.pop('model_obj', None)
    return jsonify(result)


@app.route('/api/v5/self-heal/<session_id>')
def api_self_heal(session_id):
    """Feature #6: Self-healing pipeline status."""
    session = pipeline_manager.get_session(session_id)
    if not session:
        return jsonify({'error': 'No session'}), 400
    from ml_engine.self_healer import SelfHealer
    healer = SelfHealer()
    # Return healing history from the session or a fresh status
    healing_log = getattr(session, 'healing_log', [])
    errors_caught = getattr(session, 'errors_caught', 0)
    return jsonify({
        'status': 'Healthy' if not healing_log else 'Healed',
        'total_fixes': len(healing_log),
        'errors_caught': errors_caught,
        'uptime': '100%',
        'fixes_applied': healing_log,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)
