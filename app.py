"""
AutoML Studio - Flask Application
REST API for the full-featured AutoML dashboard with 18 features.
"""

import os
import json
import shutil
import zipfile
import io
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import time

from ml_engine.pipeline import PipelineManager
from ml_engine.profiler import read_dataset

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

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

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

pipeline_manager = PipelineManager(UPLOAD_DIR, OUTPUT_DIR)

# Supported file extensions
ALLOWED_EXTENSIONS = {'.csv', '.tsv', '.xlsx', '.xls', '.json', '.parquet'}


def _allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Serve the main dashboard UI."""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Serve the project overview dashboard."""
    return render_template('dashboard.html')


# ==============================================================================
# STEP 1: UPLOAD & PROFILE
# ==============================================================================

@app.route('/api/upload', methods=['POST'])
def upload():
    """Step 1: Upload dataset (CSV/Excel/JSON/Parquet) and get profile."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not _allowed_file(file.filename):
        return jsonify({'error': f'Unsupported format. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    problem_statement = request.form.get('problem_statement', '')
    
    # Create session
    session = pipeline_manager.create_session()
    
    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_DIR, f'{session.session_id}_{filename}')
    file.save(filepath)
    
    # Profile
    result = pipeline_manager.upload_and_profile(session.session_id, filepath, problem_statement)
    
    return jsonify(result)


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
def train():
    """Step 3: Train models (ML + Deep Learning)."""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400
    
    result = pipeline_manager.train(session_id)
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)


# ==============================================================================
# STEP 4: RETRAIN
# ==============================================================================

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Step 4: Retrain with recommendations."""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id:
        return jsonify({'error': 'Session ID required'}), 400
    
    result = pipeline_manager.retrain(session_id)
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)


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
# PHASE 1: EXPLAINABILITY & DIAGNOSTICS
# ==============================================================================

@app.route('/api/explain/<session_id>')
def get_explainability(session_id):
    """Get SHAP-based model explanations."""
    result = pipeline_manager.get_explainability(session_id)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/explain-row/<session_id>/<int:row_index>')
def explain_row(session_id, row_index):
    """Get local explanation for a specific test row."""
    result = pipeline_manager.explain_row(session_id, row_index)
    if 'error' in result:
        return jsonify(result), 400
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
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/diagnostics/<session_id>')
def get_diagnostics(session_id):
    """Get advanced model diagnostics (ROC, residuals, learning curves)."""
    result = pipeline_manager.get_diagnostics(session_id)
    if 'error' in result:
        return jsonify(result), 400
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
def list_experiments():
    """List all experiments with optional filtering."""
    search = request.args.get('search', None)
    tag = request.args.get('tag', None)
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    sort_by = request.args.get('sort_by', 'created_at')
    sort_order = request.args.get('sort_order', 'desc')
    
    result = pipeline_manager.list_experiments(
        search=search, tag=tag, limit=limit, offset=offset,
        sort_by=sort_by, sort_order=sort_order
    )
    return jsonify(result)


@app.route('/api/experiments/<exp_id>')
def get_experiment(exp_id):
    """Get experiment details."""
    result = pipeline_manager.get_experiment(exp_id)
    if not result:
        return jsonify({'error': 'Experiment not found'}), 404
    return jsonify(result)


@app.route('/api/experiments/<exp_id>', methods=['PUT'])
def update_experiment(exp_id):
    """Update experiment metadata (name, tags, notes)."""
    data = request.json
    pipeline_manager.update_experiment(exp_id, **data)
    return jsonify({'success': True})


@app.route('/api/experiments/<exp_id>', methods=['DELETE'])
def delete_experiment(exp_id):
    """Delete an experiment."""
    pipeline_manager.delete_experiment(exp_id)
    return jsonify({'success': True})


@app.route('/api/experiments/compare', methods=['POST'])
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
def experiment_stats():
    """Get overall experiment statistics."""
    return jsonify(pipeline_manager.get_experiment_stats())

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
# TIER 1: CHAT (Upgraded with OpenAI + Memory)
# ==============================================================================

@app.route('/api/chat/<session_id>', methods=['POST'])
def chat(session_id):
    """Process chat message with full context injection and conversation memory."""
    data = request.get_json()
    message = data.get('message', '')
    if not message:
        return jsonify({'error': 'Message required'}), 400
    result = pipeline_manager.chat(session_id, message)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


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
def list_projects():
    """List saved projects."""
    return jsonify(pipeline_manager.list_projects())


@app.route('/api/projects/save', methods=['POST'])
def save_project():
    """Save current session as a project."""
    data = request.get_json()
    session_id = data.get('session_id')
    name = data.get('name')
    result = pipeline_manager.save_project(session_id, name)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/projects/load', methods=['POST'])
def load_project():
    """Load a saved project."""
    data = request.get_json()
    name = data.get('name')
    result = pipeline_manager.load_project(name)
    if 'error' in result:
        return jsonify(result), 400
    return jsonify(result)


@app.route('/api/projects/<name>', methods=['DELETE'])
def delete_project(name):
    """Delete a saved project."""
    result = pipeline_manager.delete_project(name)
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



if __name__ == '__main__':
    app.run(debug=True, port=5000)

