"""
AutoML Problem Solver - Pipeline Orchestrator
Manages the step-by-step pipeline state and coordinates all ML operations.
Full integration: profiling, cleaning, transformation, training, recommendations,
retraining, explainability, diagnostics, time series, deployment, monitoring,
deep learning, experiment tracking, report generation, universal transform,
unsupervised ML, AutoEDA, fairness, synthetic data, causal inference,
model compression, data valuation, autonomous agent, chat, federated learning.
"""

import os
import uuid
import json
import pickle
import threading
import numpy as np
import pandas as pd

from ml_engine.profiler import profile_dataset, read_dataset
from ml_engine.cleaner import clean_dataset
from ml_engine.transformer import transform_dataset
from ml_engine.trainer import train_models
from ml_engine.recommender import generate_recommendations
from ml_engine.retrainer import retrain_with_recommendations
from ml_engine.explainer import explain_model, explain_single_row, whatif_analysis
from ml_engine.diagnostics import generate_diagnostics
from ml_engine.timeseries_detector import detect_timeseries, extract_temporal_features, create_lag_features
from ml_engine.timeseries_trainer import train_timeseries_models
from ml_engine.deployer import predict_single, predict_batch, export_deployment_package, prepare_single_row
from ml_engine.monitor import compute_drift, generate_drift_report, detect_model_drift
from ml_engine.experiment_store import ExperimentStore
from ml_engine.report_generator import generate_html_report
from ml_engine.unsupervised_report_generator import generate_unsupervised_html_report

# New Tier 0-3 imports
from ml_engine.universal_transformer import UniversalDataTransformer
from ml_engine.data_quality import compute_data_quality
from ml_engine.clustering_engine import auto_cluster
from ml_engine.anomaly_engine import detect_anomalies
from ml_engine.dim_reduction import reduce_dimensions
from ml_engine.association_engine import mine_association_rules
from ml_engine.topic_engine import discover_topics
from ml_engine.fairness import detect_sensitive_columns, audit_fairness
from ml_engine.synthetic_generator import generate_synthetic_data
from ml_engine.auto_eda import run_auto_eda
from ml_engine.causal_engine import discover_causal_graph, estimate_causal_effect
from ml_engine.compressor import compress_model
from ml_engine.data_valuation import valuate_data
from ml_engine.autonomous_agent import run_autonomous_agent
from ml_engine.llm_agent import AutoMLChatAgent
from ml_engine.federated_engine import simulate_federated
from ml_engine.pipeline_builder import PipelineDAG

# New Studio imports
from ml_engine.hyperopt_engine import auto_optimize
from ml_engine.cleaning_advisor import generate_cleaning_suggestions, apply_suggestions
# Project storage: B2 cloud first, local fallback
try:
    from ml_engine.b2_project_storage import save_project as _save_project, load_project as _load_project, list_projects as _list_projects, delete_project as _delete_project
except Exception:
    from ml_engine.local_storage import save_project as _save_project, load_project as _load_project, list_projects as _list_projects, delete_project as _delete_project
from ml_engine.b2_storage import upload_file


class PipelineSession:
    """Manages the state of a single AutoML session."""
    
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.status = 'idle'
        self.progress = 0
        self.progress_message = ''
        self.current_step = 'upload'
        
        # Data storage
        self.original_df = None
        self.cleaned_df = None
        self.transformed_df = None
        
        # Results storage
        self.profile = None
        self.clean_report = None
        self.transform_report = None
        self.transform_metadata = None
        self.training_results = None
        self.recommendations = None
        self.retrain_results = None
        
        # New: Phase 1 - Explainability & Diagnostics
        self.explainability = None
        self.diagnostics = None
        
        # New: Phase 2 - Time Series
        self.is_timeseries = False
        self.ts_training_results = None
        
        # New: Phase 3 - Deployment
        self.deployment_package = None
        self.drift_report = None
        
        # New: Phase 4 - Experiments
        self.experiment_id = None
        self.user_id = None
        
        # Train/test data (kept for explainability & deployment)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.trained_models = None
        self.best_model = None
        self.feature_names = None
        
        # Innovation features state
        self.universal_transform_report = None
        self.data_quality = None
        self.eda_report = None
        self.unsupervised_results = None
        self.fairness_report = None
        self.causal_graph = None
        self.unsupervised_model_path = None
        self.unsupervised_report_path = None
        self.is_unsupervised = False
        
        # Studio features
        self.progress_log = []
        self.cleaning_suggestions = None
        self.hyperopt_results = None
        
        # Dataset versioning / checkpoints
        self._checkpoints = {}  # label -> (df_copy, metadata)
        
        # File paths
        self.upload_path = None
        self.output_dir = None
    
    def update_progress(self, message, progress):
        self.progress_message = message
        self.progress = progress
    
    def save_checkpoint(self, label='auto'):
        """Save a named snapshot of the current dataset."""
        import time
        df = self.transformed_df if self.transformed_df is not None else (
            self.cleaned_df if self.cleaned_df is not None else self.original_df)
        if df is not None:
            self._checkpoints[label] = {
                'df': df.copy(),
                'timestamp': time.time(),
                'shape': df.shape,
                'step': self.current_step,
            }
        return list(self._checkpoints.keys())
    
    def restore_checkpoint(self, label):
        """Restore dataset from a named checkpoint."""
        if label not in self._checkpoints:
            return {'error': f'Checkpoint "{label}" not found. Available: {list(self._checkpoints.keys())}'}
        cp = self._checkpoints[label]
        self.original_df = cp['df'].copy()
        self.cleaned_df = None
        self.transformed_df = None
        self.current_step = 'upload'
        return {'success': True, 'restored': label, 'shape': list(cp['shape'])}
    
    def list_checkpoints(self):
        """Return metadata for all saved checkpoints."""
        return [
            {'label': k, 'shape': list(v['shape']), 'step': v['step'],
             'timestamp': v['timestamp']}
            for k, v in self._checkpoints.items()
        ]
    
    def to_dict(self):
        return {
            'session_id': self.session_id,
            'status': self.status,
            'progress': self.progress,
            'progress_message': self.progress_message,
            'current_step': self.current_step,
            'is_timeseries': self.is_timeseries,
            'experiment_id': self.experiment_id,
            'upload_path': self.upload_path,
        }


class PipelineManager:
    """Manages multiple pipeline sessions."""
    
    def __init__(self, upload_dir='uploads', output_dir='outputs'):
        from session_manager import SessionManager
        self.sessions = SessionManager(maxsize=1000, ttl=7200)
        self.upload_dir = upload_dir
        self.output_dir = output_dir
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize experiment store
        self.experiment_store = ExperimentStore()
        
        # Multi-dataset management
        self.datasets = {}  # dataset_id -> {name, path, profile, uploaded_at}
        
        # Initialize LLM Chat Agent
        self.chat_agent = AutoMLChatAgent()
    
    def create_session(self):
        session = PipelineSession()
        session.output_dir = os.path.join(self.output_dir, session.session_id)
        os.makedirs(session.output_dir, exist_ok=True)
        self.sessions.set(session.session_id, session)
        return session
    
    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def _b2_key(self, session, suffix):
        user_id = getattr(session, 'user_id', None) or 'anonymous'
        return f"users/{user_id}/sessions/{session.session_id}/{suffix}"

    def _upload_to_b2(self, session, local_path, key_suffix):
        if not local_path or not os.path.exists(local_path):
            return
        try:
            upload_file(self._b2_key(session, key_suffix), local_path)
        except Exception:
            pass
    
    def upload_and_profile(self, session_id, filepath, problem_statement, user_id=None):
        """Step 1: Upload file and profile the dataset."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        session.user_id = user_id
        session.status = 'processing'
        session.current_step = 'upload'
        session.update_progress('Analyzing dataset...', 10)
        
        try:
            # Read the data (multi-format)
            session.upload_path = filepath
            session.original_df = read_dataset(filepath)
            
            # Profile (includes TS detection, NLP detection, multi-format)
            session.update_progress('Detecting target column and problem type...', 50)
            session.profile = profile_dataset(filepath, problem_statement)
            
            # Check if time series
            session.is_timeseries = session.profile.get('is_timeseries', False)
            
            # Create experiment record
            dataset_name = os.path.basename(filepath)
            self._upload_to_b2(session, filepath, f"uploads/{dataset_name}")
            session.experiment_id = self.experiment_store.create_experiment(
                dataset_name=dataset_name,
                target_column=session.profile.get('target_column', ''),
                problem_type=session.profile.get('problem_type', ''),
                n_rows=session.profile.get('n_rows', 0),
                n_cols=session.profile.get('n_cols', 0),
                session_id=session_id,
                user_id=user_id,
                problem_statement=problem_statement or '',
            )

            if session.experiment_id:
                pass
            
            # Save profile to experiment
            self.experiment_store.save_step_result(
                session.experiment_id, 'profile',
                {k: v for k, v in session.profile.items() if k != 'preview'}
            )
            
            session.update_progress('Profiling complete!', 100)
            session.status = 'complete'
            session.current_step = 'clean'
            
            return {
                'success': True,
                'session_id': session_id,
                'profile': session.profile,
                'experiment_id': session.experiment_id,
            }
        except Exception as e:
            session.status = 'error'
            return {'error': str(e)}
    
    def clean_and_transform(self, session_id):
        """Step 2: Clean and transform the dataset."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.original_df is None:
            return {'error': 'No dataset uploaded'}
        
        session.status = 'processing'
        session.current_step = 'clean'
        
        def _run():
            try:
                # Clean
                session.update_progress('Removing duplicates...', 10)
                session.cleaned_df, session.clean_report = clean_dataset(
                    session.original_df.copy(), session.profile
                )
                
                # Handle time series data
                if session.is_timeseries:
                    session.update_progress('Extracting temporal features...', 30)
                    dt_col = session.profile.get('datetime_column')
                    target_col = session.profile.get('target_column')
                    
                    if dt_col and dt_col in session.cleaned_df.columns:
                        session.cleaned_df = extract_temporal_features(session.cleaned_df, dt_col)
                        # Sort by datetime
                        session.cleaned_df = session.cleaned_df.sort_values(dt_col).reset_index(drop=True)
                        # Create lag features
                        if target_col and target_col in session.cleaned_df.columns:
                            session.cleaned_df = create_lag_features(session.cleaned_df, target_col)
                        # Drop datetime column for training
                        session.cleaned_df = session.cleaned_df.drop(columns=[dt_col], errors='ignore')
                
                session.update_progress('Transforming features...', 50)
                
                # Transform (includes NLP text processing)
                session.transformed_df, session.transform_report, session.transform_metadata = transform_dataset(
                    session.cleaned_df.copy(), session.profile
                )
                
                # Save cleaned CSV
                csv_path = os.path.join(session.output_dir, 'cleaned_data.csv')
                session.transformed_df.to_csv(csv_path, index=False)

                transformed_path = os.path.join(session.output_dir, 'transformed_data.csv')
                session.transformed_df.to_csv(transformed_path, index=False)
                self._upload_to_b2(session, transformed_path, 'data/transformed_data.csv')
                
                # Save to experiment
                if session.experiment_id:
                    self.experiment_store.save_step_result(
                        session.experiment_id, 'clean_transform',
                        {
                            'clean_summary': session.clean_report.get('summary', {}),
                            'transform_summary': session.transform_report.get('summary', {}),
                        }
                    )
                
                session.update_progress('Cleaning & transformation complete!', 100)
                session.status = 'complete'
                session.current_step = 'train'
                
            except Exception as e:
                session.status = 'error'
                session.progress_message = str(e)
        
        thread = threading.Thread(target=_run)
        thread.start()
        
        return {
            'success': True,
            'session_id': session_id,
            'message': 'Cleaning and transformation started',
        }
    
    def train(self, session_id):
        """Step 3: Train models (standard + deep learning + optional time series)."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.transformed_df is None:
            return {'error': 'Dataset not yet cleaned and transformed'}
        
        session.status = 'processing'
        session.current_step = 'train'
        
        def _run():
            try:
                session.update_progress('Starting model training...', 5)
                
                if session.is_timeseries:
                    # Time Series training
                    target_col = session.transform_metadata.get('target_column') or session.profile.get('target_column')
                    session.ts_training_results = train_timeseries_models(
                        session.transformed_df.copy(),
                        target_col,
                        output_dir=session.output_dir,
                        progress_callback=lambda msg, prog: session.update_progress(msg, int(prog * 0.6))
                    )
                    session.training_results = session.ts_training_results
                else:
                    # Standard ML training
                    session.training_results = train_models(
                        session.transformed_df.copy(),
                        session.profile,
                        session.transform_metadata,
                        session.output_dir,
                        progress_callback=lambda msg, prog: session.update_progress(msg, int(prog * 0.5))
                    )
                
                # Store train/test data for later use
                target_col = session.transform_metadata.get('target_column') or session.profile.get('target_column')
                X = session.transformed_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
                y = session.transformed_df[target_col]
                
                from sklearn.model_selection import train_test_split
                stratify = y if session.profile.get('problem_type') == 'classification' else None
                try:
                    session.X_train, session.X_test, session.y_train, session.y_test = train_test_split(
                        X, y, test_size=0.25, random_state=42, stratify=stratify
                    )
                except ValueError:
                    session.X_train, session.X_test, session.y_train, session.y_test = train_test_split(
                        X, y, test_size=0.25, random_state=42
                    )
                
                # Apply scaling to stored train/test data (matching what trainer.py does)
                from ml_engine.transformer import fit_scaler, apply_scaler
                numeric_cols_to_scale = (session.transform_metadata or {}).get('numeric_cols_to_scale', [])
                stored_scaler = fit_scaler(session.X_train, numeric_cols_to_scale)
                if stored_scaler is not None:
                    session.X_train = apply_scaler(session.X_train, stored_scaler, numeric_cols_to_scale)
                    session.X_test = apply_scaler(session.X_test, stored_scaler, numeric_cols_to_scale)
                
                session.feature_names = list(X.columns)
                
                # Get trained models from context
                ctx = session.training_results.get('training_context', {})
                session.trained_models = ctx.get('trained_models', {})
                
                # Get best model object
                best_name = session.training_results.get('best_model')
                if best_name and best_name in session.trained_models:
                    session.best_model = session.trained_models[best_name]
                
                # Generate diagnostics
                session.update_progress('Generating model diagnostics...', 75)
                try:
                    problem_type = session.profile.get('problem_type', 'classification')
                    if problem_type != 'forecasting':
                        session.diagnostics = generate_diagnostics(
                            session.trained_models,
                            session.X_train, session.X_test,
                            session.y_train, session.y_test,
                            problem_type
                        )
                except Exception:
                    pass
                
                # Generate SHAP explanations
                session.update_progress('Computing model explanations (SHAP)...', 82)
                try:
                    if session.best_model is not None and session.X_train is not None:
                        problem_type = session.profile.get('problem_type', 'classification')
                        session.explainability = explain_model(
                            session.best_model,
                            session.X_train,
                            session.X_test,
                            session.feature_names,
                            problem_type
                        )
                except Exception:
                    pass
                
                # Auto-calibration (classification only)
                session.update_progress('Running confidence calibration...', 88)
                try:
                    problem_type = session.profile.get('problem_type', 'classification')
                    if problem_type == 'classification' and session.best_model is not None:
                        from ml_engine.calibration_engine import compute_calibration, auto_calibrate
                        cal = compute_calibration(session.best_model, session.X_test, session.y_test)
                        session.training_results['calibration'] = cal
                        # Auto-calibrate if ECE > 0.05
                        if cal.get('ece', 0) > 0.05:
                            cal_result = auto_calibrate(
                                session.best_model, session.X_train, session.y_train,
                                session.X_test, session.y_test, 'auto'
                            )
                            if cal_result.get('success') and cal_result.get('calibrated_model'):
                                session.best_model = cal_result['calibrated_model']
                                session.training_results['calibration']['auto_calibrated'] = True
                                session.training_results['calibration']['calibration_method'] = cal_result.get('method', 'auto')
                except Exception:
                    pass
                
                # Generate recommendations
                session.update_progress('Analyzing results for recommendations...', 90)
                if not session.is_timeseries:
                    session.recommendations = generate_recommendations(
                        session.profile,
                        session.clean_report,
                        session.transform_report,
                        session.training_results,
                    )
                
                # Save to experiment store
                if session.experiment_id:
                    self.experiment_store.update_experiment(
                        session.experiment_id,
                        status='trained',
                        best_model=session.training_results.get('best_model', ''),
                        best_score=session.training_results.get('best_score', 0),
                        primary_metric=session.training_results.get('primary_metric_name', ''),
                    )
                    
                    # Save model results
                    for entry in session.training_results.get('leaderboard', []):
                        self.experiment_store.save_model_result(
                            session.experiment_id,
                            entry['model'],
                            'ml',
                            entry['primary_metric'],
                            entry.get('metrics', {}),
                            is_best=(entry['rank'] == 1)
                        )

                best_model_path = session.training_results.get('best_model_path') if session.training_results else None
                if not best_model_path:
                    best_model_path = os.path.join(session.output_dir, 'best_model.pkl')
                self._upload_to_b2(session, best_model_path, 'models/best_model.pkl')
                
                # Automatically generate and upload report
                session.update_progress('Generating comprehensive report...', 95)
                try:
                    self.generate_report(session_id)
                except Exception as e:
                    pass
                
                session.update_progress('Training complete!', 100)
                session.status = 'complete'
                session.current_step = 'results'
                
            except Exception as e:
                session.status = 'error'
                session.progress_message = str(e)
        
        thread = threading.Thread(target=_run)
        thread.start()
        
        return {
            'success': True,
            'session_id': session_id,
            'message': 'Model training started',
        }
    
    def retrain(self, session_id):
        """Step 4: Retrain with recommendations."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.training_results is None or session.recommendations is None:
            return {'error': 'Must complete training first'}
        
        session.status = 'processing'
        session.current_step = 'retrain'
        
        def _run():
            try:
                session.update_progress('Applying recommendations...', 5)
                
                session.retrain_results = retrain_with_recommendations(
                    session.transformed_df.copy(),
                    session.profile,
                    session.transform_metadata,
                    session.recommendations,
                    session.training_results,
                    session.output_dir,
                    progress_callback=lambda msg, prog: session.update_progress(msg, prog)
                )
                
                # Update experiment
                if session.experiment_id:
                    self.experiment_store.update_experiment(
                        session.experiment_id,
                        status='complete',
                        best_model=session.retrain_results.get('best_model', ''),
                        best_score=session.retrain_results.get('best_score', 0),
                    )
                    self.experiment_store.save_step_result(
                        session.experiment_id, 'retrain',
                        {
                            'improvement': session.retrain_results.get('improvement', 0),
                            'best_model': session.retrain_results.get('best_model', ''),
                        }
                    )

                improved_path = session.retrain_results.get('best_model_path') if session.retrain_results else None
                if not improved_path:
                    improved_path = os.path.join(session.output_dir, 'improved_model.pkl')
                self._upload_to_b2(session, improved_path, 'models/improved_model.pkl')
                
                # Regenerate report to reflect retrain results
                session.update_progress('Updating comprehensive report...', 95)
                try:
                    self.generate_report(session_id)
                except Exception as e:
                    pass
                
                session.update_progress('Retraining complete!', 100)
                session.status = 'complete'
                session.current_step = 'retrain_done'
                
            except Exception as e:
                session.status = 'error'
                session.progress_message = str(e)
        
        thread = threading.Thread(target=_run)
        thread.start()
        
        return {
            'success': True,
            'session_id': session_id,
            'message': 'Retraining started',
        }
    
    # ========== Phase 1: Explainability ==========

    def _build_explainability_fallback(self, session):
        """Fallback explainability using model-native feature importance when SHAP is unavailable."""
        if not session or not session.training_results:
            return None

        fi = session.training_results.get('feature_importance')
        if not fi:
            return None

        return {
            'feature_importance': fi,
            'global_importance': fi,
            'shap_summary': 'SHAP unavailable; showing model feature importance fallback.',
            'summary': {
                'source': 'feature_importance_fallback',
                'shap_available': False,
                'n_features': len(fi),
            },
        }

    def _normalize_explainability_payload(self, payload):
        """Ensure backward-compatible keys consumed by multiple dashboard variants."""
        if not isinstance(payload, dict):
            return payload

        if 'feature_importance' not in payload and 'global_importance' in payload:
            payload['feature_importance'] = payload['global_importance']

        if 'shap_summary' not in payload:
            summary = payload.get('summary')
            if isinstance(summary, dict):
                n_features = summary.get('n_features')
                n_samples = summary.get('n_samples_explained')
                payload['shap_summary'] = (
                    f"Explained {n_samples} samples across {n_features} features."
                    if n_features is not None and n_samples is not None
                    else 'SHAP explainability generated.'
                )

        return payload
    
    def get_explainability(self, session_id):
        """Get SHAP explainability data."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.explainability and 'error' not in session.explainability:
            return self._normalize_explainability_payload(session.explainability)
        
        if session.explainability and 'error' in session.explainability:
            fallback = self._build_explainability_fallback(session)
            if fallback:
                session.explainability = fallback
                return self._normalize_explainability_payload(session.explainability)
        
        # Compute on demand
        if session.best_model and session.X_train is not None:
            problem_type = session.profile.get('problem_type', 'classification')
            session.explainability = explain_model(
                session.best_model, session.X_train, session.X_test,
                session.feature_names, problem_type
            )
            if 'error' in session.explainability:
                fallback = self._build_explainability_fallback(session)
                if fallback:
                    session.explainability = fallback
            return self._normalize_explainability_payload(session.explainability)
        
        fallback = self._build_explainability_fallback(session)
        if fallback:
            session.explainability = fallback
            return self._normalize_explainability_payload(session.explainability)

        return {'error': 'No trained model available for explanation'}
    
    def explain_row(self, session_id, row_index):
        """Get local explanation for a specific row."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.best_model is None or session.X_test is None:
            return {'error': 'No trained model available'}
        
        if row_index >= len(session.X_test):
            row_index = len(session.X_test) - 1
        
        row_data = session.X_test.iloc[[row_index]]
        problem_type = session.profile.get('problem_type', 'classification')
        
        return explain_single_row(
            session.best_model, session.X_train, row_data,
            session.feature_names, problem_type
        )
    
    def run_whatif(self, session_id, row_index, feature_name, new_value):
        """Run what-if analysis."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.best_model is None or session.X_test is None:
            return {'error': 'No trained model available'}
        
        if row_index >= len(session.X_test):
            row_index = len(session.X_test) - 1
        
        row_data = session.X_test.iloc[row_index].to_dict()
        
        return whatif_analysis(
            session.best_model, session.X_train, row_data,
            feature_name, float(new_value), session.feature_names,
            session.profile.get('problem_type', 'classification')
        )
    
    def get_diagnostics(self, session_id):
        """Get model diagnostics."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.diagnostics:
            return session.diagnostics
        
        return {'error': 'Diagnostics not computed yet'}
    
    # ========== Phase 3: Deployment ==========
    
    def predict(self, session_id, features):
        """Make a single prediction."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.best_model is None:
            return {'error': 'No trained model available'}
        
        result = predict_single(
            session.best_model, features, session.feature_names,
            session.transform_metadata
        )

        if result.get('error'):
            return result

        if session.X_train is None or not session.feature_names:
            return result

        try:
            row_df, _ = prepare_single_row(
                features, session.feature_names, session.transform_metadata
            )
            local = explain_single_row(
                session.best_model,
                session.X_train,
                row_df,
                session.feature_names,
                session.profile.get('problem_type', 'classification')
            )
            if local and not local.get('error'):
                result['feature_contributions'] = local.get('contributions')
        except Exception:
            pass

        return result
    
    def batch_predict(self, session_id, df):
        """Make batch predictions."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.best_model is None:
            return {'error': 'No trained model available'}
        
        return predict_batch(
            session.best_model, df, session.feature_names,
            session.transform_metadata
        )
    
    def export_deployment(self, session_id):
        """Export deployment package."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        model_path = os.path.join(session.output_dir, 'best_model.pkl')
        improved_path = os.path.join(session.output_dir, 'improved_model.pkl')
        
        # Prefer improved model
        use_path = improved_path if os.path.exists(improved_path) else model_path
        
        if not os.path.exists(use_path):
            return {'error': 'No model file found'}
        
        problem_type = session.profile.get('problem_type', 'classification')
        best_model_name = session.training_results.get('best_model', 'Unknown')
        
        result = export_deployment_package(
            use_path, session.feature_names, problem_type,
            best_model_name, session.output_dir
        )
        
        session.deployment_package = result
        return result
    
    # ========== Phase 3: Monitoring ==========
    
    def check_drift(self, session_id, new_df):
        """Check data drift against training data."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.X_train is None:
            return {'error': 'No training data available'}
        
        drift = compute_drift(session.X_train, new_df, session.feature_names)
        report = generate_drift_report(drift)
        session.drift_report = report
        
        return report
    
    # ========== Phase 4: Experiments ==========
    
    def list_experiments(self, **kwargs):
        """List all experiments."""
        return self.experiment_store.list_experiments(**kwargs)
    
    def get_experiment(self, exp_id, user_id=None):
        """Get experiment details."""
        return self.experiment_store.get_experiment(exp_id, user_id=user_id)
    
    def compare_experiments(self, exp_ids):
        """Compare experiments."""
        return self.experiment_store.compare_experiments(exp_ids)
    
    def delete_experiment(self, exp_id, user_id=None):
        """Delete an experiment."""
        return self.experiment_store.delete_experiment(exp_id, user_id=user_id)
    
    def get_experiment_stats(self, user_id=None):
        """Get experiment statistics."""
        return self.experiment_store.get_stats(user_id=user_id)
    
    def update_experiment(self, exp_id, **kwargs):
        """Update experiment metadata."""
        self.experiment_store.update_experiment(exp_id, **kwargs)
        return {'success': True}
    
    # ========== Report Generation ==========
    
    def generate_report(self, session_id):
        """Generate an HTML report for the session."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        session_data = {
            'session_id': session.session_id,
            'profile': session.profile,
            'clean_report': session.clean_report,
            'transform_report': session.transform_report,
            'training_results': _serialize_training_results(session.training_results) if session.training_results else None,
            'retrain_results': session.retrain_results,
            'explainability': session.explainability,
            'diagnostics': session.diagnostics,
            'eda_report': session.eda_report,
            'drift_report': session.drift_report,
            'fairness_report': session.fairness_report,
            'unsupervised_results': session.unsupervised_results,
            'causal_graph': session.causal_graph,
        }
        
        if session_data['training_results'] and session.recommendations:
            session_data['training_results']['recommendations'] = session.recommendations
        
        report_path = generate_html_report(session_data, session.output_dir)
        self._upload_to_b2(session, report_path, 'reports/automl_report.html')
        if session.experiment_id:
            try:
                self.experiment_store.save_step_result(
                    session.experiment_id,
                    'report',
                    {'report_path': report_path}
                )
            except Exception:
                pass
        return {'report_path': report_path}
    
    # ========== Tier 0A: Universal Transform ==========
    
    def run_universal_transform(self, session_id):
        """Apply universal data transformation."""
        session = self.get_session(session_id)
        if not session or session.original_df is None:
            return {'error': 'No dataset loaded'}
        
        transformer = UniversalDataTransformer()
        session.original_df, session.universal_transform_report = transformer.transform(session.original_df)
        return session.universal_transform_report
    
    def get_data_quality(self, session_id):
        """Compute data quality scores."""
        session = self.get_session(session_id)
        if not session or session.original_df is None:
            return {'error': 'No dataset loaded'}
        
        target = session.profile.get('target_column') if session.profile else None
        session.data_quality = compute_data_quality(session.original_df, target)
        return session.data_quality
    
    # ========== Tier 0B: Unsupervised ML ==========
    
    def run_unsupervised(self, session_id, method='clustering'):
        """Run unsupervised analysis."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        df = session.transformed_df if session.transformed_df is not None else session.original_df
        if df is None:
            return {'error': 'No data available'}
        
        numeric = df.select_dtypes(include='number')
        results = {}
        
        if method in ('clustering', 'all'):
            results['clustering'] = auto_cluster(numeric)
        
        if method in ('anomaly', 'all'):
            results['anomaly'] = detect_anomalies(numeric)
        
        if method in ('dim_reduction', 'all'):
            labels = results.get('clustering', {}).get('best_labels')
            results['dim_reduction'] = reduce_dimensions(numeric, labels)
        
        if method in ('association', 'all'):
            results['association'] = mine_association_rules(df)
        
        if method in ('topics', 'all'):
            text_cols = df.select_dtypes(include='object').columns
            if len(text_cols) > 0:
                longest_col = max(text_cols, key=lambda c: df[c].astype(str).str.len().mean())
                results['topics'] = discover_topics(df[longest_col])
        
        session.unsupervised_results = results

        if session.experiment_id:
            clustering = results.get('clustering') or {}
            anomaly = results.get('anomaly') or {}
            dim_reduction_result = results.get('dim_reduction') or {}
            association_result = results.get('association') or {}
            topics_result = results.get('topics') or {}

            unsupervised_summary = {
                'method': method,
                'clustering': {
                    'best_algorithm': clustering.get('best_algorithm'),
                    'best_silhouette': clustering.get('best_silhouette'),
                    'optimal_k': clustering.get('optimal_k'),
                },
                'anomaly': {
                    'n_anomalies': anomaly.get('n_anomalies'),
                    'anomaly_pct': anomaly.get('anomaly_pct'),
                },
                'dim_reduction': {
                    'method': dim_reduction_result.get('method'),
                },
                'association_rules': len(association_result.get('rules') or []),
                'topics': len(topics_result.get('topics') or []),
            }

            self.experiment_store.save_step_result(
                session.experiment_id,
                'unsupervised',
                {
                    'summary': unsupervised_summary,
                    'results': results,
                    'dataset_path': 'data/unsupervised_dataset.csv',
                    'model_path': 'models/unsupervised_model.pkl',
                    'report_path': 'reports/unsupervised_report.html',
                }
            )

            update_kwargs = {
                'status': 'complete',
                'notes': 'Unsupervised analysis complete',
            }
            if isinstance(clustering.get('best_silhouette'), (int, float)):
                update_kwargs['best_model'] = clustering.get('best_algorithm') or ''
                update_kwargs['best_score'] = float(clustering.get('best_silhouette'))

            self.experiment_store.update_experiment(session.experiment_id, **update_kwargs)
            self.experiment_store.save_model_result(
                session.experiment_id,
                clustering.get('best_algorithm') or 'Unsupervised Analysis',
                'unsupervised',
                float(clustering.get('best_silhouette')) if isinstance(clustering.get('best_silhouette'), (int, float)) else 0,
                {
                    'summary': unsupervised_summary,
                    'algorithm_scores': clustering.get('algorithms', {}),
                },
                is_best=True,
                hyperparameters={'method': method},
                feature_importance=[]
            )

        # Save dataset used for unsupervised run and upload to B2.
        try:
            unsup_csv_path = os.path.join(session.output_dir, 'unsupervised_dataset.csv')
            df.to_csv(unsup_csv_path, index=False)
            self._upload_to_b2(session, unsup_csv_path, 'data/unsupervised_dataset.csv')
        except Exception:
            pass

        # Save an unsupervised artifact so users can download the trained unsupervised model bundle.
        try:
            model_artifact = {
                'session_id': session.session_id,
                'method': method,
                'results': results,
                'feature_columns': numeric.columns.tolist(),
            }
            unsup_model_path = os.path.join(session.output_dir, 'unsupervised_model.pkl')
            with open(unsup_model_path, 'wb') as f:
                pickle.dump(model_artifact, f)
            session.unsupervised_model_path = unsup_model_path
            self._upload_to_b2(session, unsup_model_path, 'models/unsupervised_model.pkl')
        except Exception:
            pass

        # Generate unsupervised-only report and upload to B2.
        try:
            unsup_report_path = generate_unsupervised_html_report(
                {
                    'session_id': session.session_id,
                    'profile': session.profile,
                    'unsupervised_results': results,
                },
                session.output_dir,
            )
            session.unsupervised_report_path = unsup_report_path
            self._upload_to_b2(session, unsup_report_path, 'reports/unsupervised_report.html')
        except Exception:
            pass

        return results
    
    # ========== Tier 1: AutoEDA ==========
    
    def run_eda(self, session_id):
        """Run automatic EDA."""
        session = self.get_session(session_id)
        if not session or session.original_df is None:
            return {'error': 'No dataset loaded'}
        
        target = session.profile.get('target_column') if session.profile else None
        session.eda_report = run_auto_eda(session.original_df, target)
        return session.eda_report
    
    # ========== Tier 1: Fairness ==========
    
    def get_fairness_report(self, session_id):
        """Run fairness audit."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.best_model is None or session.X_test is None:
            return {'error': 'No trained model available'}
        
        # Detect sensitive columns in original data
        df = session.original_df if session.original_df is not None else session.transformed_df
        sensitive = detect_sensitive_columns(df)
        
        if not sensitive:
            return {'message': 'No sensitive columns auto-detected', 'reports': [], 'overall_fairness_score': 100}
        
        # Get predictions
        y_pred = session.best_model.predict(session.X_test)
        
        # Build sensitive features matrix
        sens_cols = [s['column'] for s in sensitive if s['column'] in df.columns]
        if not sens_cols:
            return {'message': 'Sensitive columns not in processed data', 'reports': [], 'overall_fairness_score': 100}
        
        sens_data = df.loc[session.X_test.index, sens_cols].values if all(c in df.columns for c in sens_cols) else None
        
        if sens_data is None:
            return {'message': 'Could not align sensitive columns', 'reports': [], 'overall_fairness_score': 100}
        
        session.fairness_report = audit_fairness(session.y_test, y_pred, sens_data, sens_cols)
        session.fairness_report['sensitive_columns'] = sensitive
        return session.fairness_report
    
    # ========== Tier 1: Synthetic Data ==========
    
    def generate_synthetic(self, session_id, n_samples=None):
        """Generate synthetic data."""
        session = self.get_session(session_id)
        if not session or session.original_df is None:
            return {'error': 'No dataset loaded'}
        
        target = session.profile.get('target_column') if session.profile else None
        result = generate_synthetic_data(session.original_df, n_samples=n_samples, class_column=target)
        
        # Save synthetic data
        if 'synthetic_data' in result:
            syn_path = os.path.join(session.output_dir, 'synthetic_data.csv')
            result['synthetic_data'].to_csv(syn_path, index=False)
            result['file_path'] = syn_path
            result['preview'] = result['synthetic_data'].head(10).to_dict('records')
            del result['synthetic_data']  # Don't send full data in JSON
        
        return result
    
    # ========== Tier 2: Causal Inference ==========
    
    def get_causal_graph(self, session_id):
        """Discover causal relationships."""
        session = self.get_session(session_id)
        if not session or session.original_df is None:
            return {'error': 'No dataset loaded'}
        
        target = session.profile.get('target_column') if session.profile else None
        session.causal_graph = discover_causal_graph(session.original_df, target)
        return session.causal_graph
    
    def get_causal_effect(self, session_id, treatment, outcome):
        """Estimate causal effect."""
        session = self.get_session(session_id)
        if not session or session.original_df is None:
            return {'error': 'No dataset loaded'}
        
        return estimate_causal_effect(session.original_df, treatment, outcome)
    
    # ========== Tier 2: Model Compression ==========
    
    def compress(self, session_id):
        """Compress the best model."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.best_model is None or session.X_train is None:
            return {'error': 'No trained model available'}
        
        problem_type = session.profile.get('problem_type', 'classification')
        return compress_model(
            session.best_model,
            session.X_train.values if hasattr(session.X_train, 'values') else session.X_train,
            session.y_train.values if hasattr(session.y_train, 'values') else session.y_train,
            session.X_test.values if hasattr(session.X_test, 'values') else session.X_test,
            session.y_test.values if hasattr(session.y_test, 'values') else session.y_test,
            problem_type, session.output_dir
        )
    
    # ========== Tier 3: Data Valuation ==========
    
    def get_data_valuation(self, session_id):
        """Compute data point values."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.best_model is None or session.X_train is None:
            return {'error': 'No trained model available'}
        
        problem_type = session.profile.get('problem_type', 'classification')
        return valuate_data(
            session.best_model, session.X_train, session.y_train,
            session.X_test, session.y_test, problem_type
        )
    
    # ========== Tier 3: Autonomous Agent ==========
    
    def run_agent(self, session_id, max_iterations=5):
        """Run autonomous improvement agent."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.best_model is None or session.X_train is None:
            return {'error': 'No trained model available'}
        
        problem_type = session.profile.get('problem_type', 'classification')
        return run_autonomous_agent(
            session.best_model, session.X_train, session.y_train,
            session.X_test, session.y_test, problem_type, max_iterations
        )
    
    # ========== Tier 1: Chat ==========
    
    def chat(self, session_id, message):
        """Process chat message."""
        session = self.get_session(session_id)
        agent = AutoMLChatAgent(self)
        session_data = {'df': session.original_df} if session else {}
        return agent.chat(message, session_data)
    
    # ========== Tier 3: Federated Learning ==========
    
    def run_federated(self, session_id, n_clients=3, n_rounds=5):
        """Simulate federated learning."""
        session = self.get_session(session_id)
        if not session or session.X_train is None:
            return {'error': 'No training data available'}
        
        problem_type = session.profile.get('problem_type', 'classification')
        X = session.X_train.values if hasattr(session.X_train, 'values') else session.X_train
        y = session.y_train.values if hasattr(session.y_train, 'values') else session.y_train
        
        return simulate_federated(X, y, n_clients, n_rounds, problem_type)
    
    # ========== Pipeline Builder ==========
    
    def get_pipeline_blocks(self):
        """Get available pipeline blocks."""
        dag = PipelineDAG()
        return dag.get_block_registry()
    
    def get_status(self, session_id):
        """Get current session status."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        result = session.to_dict()
        
        # Always expose profile and step outputs if they exist, regardless of overall status
        # (so the UI can restore its partial state while a later step is processing).
        if session.profile:
            result['profile'] = session.profile

        if session.clean_report:
            result['clean_report'] = session.clean_report
        if session.transform_report:
            result['transform_report'] = session.transform_report

        if session.training_results:
            result['training_results'] = _serialize_training_results(session.training_results)
        if session.recommendations:
            result['recommendations'] = session.recommendations

        if session.explainability and 'error' not in session.explainability:
            result['has_explainability'] = True
        if session.diagnostics:
            result['has_diagnostics'] = True

        if session.retrain_results:
            result['retrain_results'] = session.retrain_results
        
        return result

    def get_executive_summary(self, session_id):
        """Aggregate high-level metrics for the Executive Dashboard."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
            
        summary = {
            'dataset': {},
            'model': {},
            'metrics': {}
        }
        
        # Dataset Health
        if session.profile:
            summary['dataset'] = {
                'name': os.path.basename(session.upload_path) if session.upload_path else 'Unknown',
                'rows': session.profile.get('n_rows', 0),
                'columns': session.profile.get('n_cols', 0),
                'problem_type': session.profile.get('problem_type', 'unknown'),
                'target': session.profile.get('target_column', 'unknown'),
                'missing_pct': session.profile.get('total_missing_pct', 0)
            }
        
        # Model Performance
        res = session.retrain_results or session.training_results
        if res:
            summary['model'] = {
                'best_model': res.get('best_model', 'N/A'),
                'best_score': res.get('best_score', 0),
                'metric': res.get('primary_metric_name', 'score'),
                'improvement': session.retrain_results.get('improvement', 0) if session.retrain_results else 0
            }
            
        # Data Quality (Tier 0A)
        if session.data_quality:
            summary['metrics']['quality_score'] = session.data_quality.get('overall_score', 0)
            summary['metrics']['quality_grade'] = session.data_quality.get('overall_grade', 'N/A')
            
        # Fairness (Tier 1)
        if session.fairness_report:
            summary['metrics']['fairness_score'] = session.fairness_report.get('overall_fairness_score', 0)
            summary['metrics']['fairness_grade'] = session.fairness_report.get('overall_grade', 'N/A')
            summary['metrics']['has_bias'] = session.fairness_report.get('has_bias', False)
            
        # Drift (Phase 3)
        if session.drift_report:
            summary['metrics']['drift_status'] = session.drift_report.get('status', 'unknown')
        
        return summary

    # ========== Hyperparameter Optimization ==========
    
    def optimize_hyperparams(self, session_id, method='auto', budget=30):
        """Run hyperparameter optimization on trained models."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        if session.trained_models is None or session.X_train is None:
            return {'error': 'No trained model available. Train models first.'}
        
        problem_type = session.profile.get('problem_type', 'classification')
        session.current_step = 'tune'
        
        try:
            result = auto_optimize(
                session.trained_models,
                session.X_train, session.y_train,
                problem_type, method, budget,
                progress_callback=lambda msg, pct: session.update_progress(msg, pct)
            )
            session.hyperopt_results = result
            
            # Update best model if improved
            best_estimators = result.get('best_estimators', {})
            best_name = result.get('best_model')
            if best_name and best_name in best_estimators:
                session.best_model = best_estimators[best_name]
                session.trained_models[best_name] = best_estimators[best_name]
            
            return {k: v for k, v in result.items() if k != 'best_estimators'}
        except Exception as e:
            return {'error': str(e)}
    
    # ========== Cleaning Suggestions ==========
    
    def get_cleaning_suggestions(self, session_id):
        """Get AI-based data cleaning suggestions."""
        session = self.get_session(session_id)
        if not session or session.original_df is None:
            return {'error': 'No dataset loaded'}
        
        session.cleaning_suggestions = generate_cleaning_suggestions(
            session.original_df, session.profile
        )
        return {'suggestions': session.cleaning_suggestions}
    
    def apply_cleaning_suggestions(self, session_id, accepted_ids):
        """Apply accepted cleaning suggestions."""
        session = self.get_session(session_id)
        if not session or session.original_df is None:
            return {'error': 'No dataset loaded'}
        if not session.cleaning_suggestions:
            return {'error': 'No suggestions generated yet'}
        
        session.original_df, report = apply_suggestions(
            session.original_df, session.cleaning_suggestions, accepted_ids
        )
        # Re-profile after cleaning
        session.profile['n_rows'] = len(session.original_df)
        session.profile['n_cols'] = len(session.original_df.columns)
        
        return report
    
    # ========== Multi-Dataset Management ==========
    
    def list_datasets(self):
        """List all uploaded datasets."""
        datasets = []
        for did, info in self.datasets.items():
            datasets.append({
                'id': did,
                'name': info.get('name', ''),
                'rows': info.get('rows', 0),
                'columns': info.get('columns', 0),
                'uploaded_at': info.get('uploaded_at', ''),
                'session_id': info.get('session_id', ''),
            })
        return {'datasets': datasets}
    
    def switch_dataset(self, session_id, dataset_id):
        """Switch active dataset for a session."""
        if dataset_id not in self.datasets:
            return {'error': 'Dataset not found'}
        
        info = self.datasets[dataset_id]
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        # Reload the dataset 
        filepath = info.get('path')
        if filepath and os.path.exists(filepath):
            session.original_df = read_dataset(filepath)
            session.profile = None
            session.cleaned_df = None
            session.transformed_df = None
            session.training_results = None
            return {'success': True, 'dataset': info}
        
        return {'error': 'Dataset file not found'}
    
    # ========== Local Project Storage ==========
    
    def save_project(self, session_id, name=None, user_id=None):
        """Save current session as a project."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        return _save_project(session, name, user_id=user_id)
    
    def load_project(self, name, user_id=None):
        """Load a saved project."""
        result = _load_project(name, user_id=user_id)
        if 'error' in result:
            return result
        
        # Restore session
        session = self.create_session()
        session.user_id = user_id
        meta = result.get('metadata', {})
        
        if result.get('original_data_path'):
            session.original_df = pd.read_csv(result['original_data_path'])
        if result.get('transformed_data_path'):
            session.transformed_df = pd.read_csv(result['transformed_data_path'])
        if result.get('model_path'):
            import pickle
            with open(result['model_path'], 'rb') as f:
                session.best_model = pickle.load(f)
        
        session.profile = meta.get('profile')
        session.training_results = meta.get('training_results')
        session.retrain_results = meta.get('retrain_results')
        session.recommendations = meta.get('recommendations')
        session.status = meta.get('status', 'idle')
        session.current_step = meta.get('current_step', 'upload')
        
        return {
            'success': True,
            'session_id': session.session_id,
            'metadata': meta,
        }
    
    def list_projects(self, user_id=None):
        """List saved projects."""
        return {'projects': _list_projects(user_id=user_id)}
    
    def delete_project(self, name, user_id=None):
        """Delete a saved project."""
        return _delete_project(name, user_id=user_id)
    
    # ========== Model Drift Detection ==========
    
    def check_model_drift(self, session_id, new_df):
        """Check model performance drift with new labeled data."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        if session.best_model is None:
            return {'error': 'No trained model available'}
        
        target_col = session.profile.get('target_column') if session.profile else None
        if not target_col or target_col not in new_df.columns:
            return {'error': f'Target column "{target_col}" not found in uploaded data'}
        
        X_new = new_df.drop(columns=[target_col]).select_dtypes(include=['number'])
        y_new = new_df[target_col]
        
        # Align features
        if session.feature_names:
            missing = set(session.feature_names) - set(X_new.columns)
            for col in missing:
                X_new[col] = 0
            X_new = X_new[session.feature_names]
        
        original_score = 0
        if session.training_results:
            original_score = session.training_results.get('best_score', 0)
        if session.retrain_results:
            original_score = session.retrain_results.get('best_score', original_score)
        
        problem_type = session.profile.get('problem_type', 'classification')
        return detect_model_drift(
            session.best_model, X_new, y_new, original_score, problem_type
        )
    
    # ========== Pipeline Flow Status ==========
    
    def get_pipeline_flow_status(self, session_id):
        """Get pipeline status for flow visualization."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        
        steps = [
            {'id': 'upload', 'name': 'Upload Dataset', 'icon': '📤'},
            {'id': 'clean', 'name': 'Data Cleaning', 'icon': '🧹'},
            {'id': 'transform', 'name': 'Feature Engineering', 'icon': '🔄'},
            {'id': 'train', 'name': 'Model Training', 'icon': '🤖'},
            {'id': 'evaluate', 'name': 'Evaluation', 'icon': '📊'},
            {'id': 'deploy', 'name': 'Output / Deploy', 'icon': '📦'},
        ]
        
        step_order = ['upload', 'clean', 'transform', 'train', 'evaluate', 'deploy']
        current_idx = step_order.index(session.current_step) if session.current_step in step_order else 0
        
        for i, step in enumerate(steps):
            if i < current_idx:
                step['status'] = 'complete'
            elif i == current_idx:
                step['status'] = 'active' if session.status == 'processing' else 'complete' if session.status == 'complete' else 'pending'
            else:
                step['status'] = 'pending'
        
        return {'steps': steps, 'current_step': session.current_step}
    
    # ========== Custom Pipeline (Semi-Auto) ==========
    
    def run_custom_pipeline(self, session_id, config):
        """Run a custom user-defined pipeline."""
        session = self.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}
        if session.original_df is None:
            return {'error': 'No dataset loaded'}
        
        # Config: {models: [...], preprocessing: [...], features: [...]}
        selected_models = config.get('models', [])
        preprocessing = config.get('preprocessing', [])
        
        # Store custom config
        if session.profile:
            session.profile['custom_models'] = selected_models
            session.profile['custom_preprocessing'] = preprocessing
        
        return {
            'success': True,
            'config_applied': config,
            'message': 'Custom pipeline configured. Run training to use these settings.'
        }


def _serialize_training_results(results):
    """Remove non-serializable objects from training results."""
    if not results:
        return results
    
    serializable = {}
    for key, value in results.items():
        if key == 'training_context':
            # Exclude trained_models (not JSON serializable)
            ctx = {}
            for k, v in value.items():
                if k != 'trained_models':
                    try:
                        json.dumps(v)
                        ctx[k] = v
                    except (TypeError, ValueError):
                        ctx[k] = str(v)
            serializable[key] = ctx
        elif key in ('best_model_path', 'trained_models'):
            if key == 'best_model_path':
                serializable[key] = value
        else:
            try:
                json.dumps(value)
                serializable[key] = value
            except (TypeError, ValueError):
                serializable[key] = str(value)
    return serializable
