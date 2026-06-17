"""
AutoML Studio — Python SDK
==========================

A pip-installable client library that wraps the AutoML Studio REST API,
providing a clean Pythonic interface for automated machine learning.

Usage:
    from automl_studio import AutoML

    automl = AutoML(server="http://localhost:7860")
    results = automl.fit("data.csv", target="price", time_budget=300)
    print(results.leaderboard)
    predictions = automl.predict(new_data)
    automl.explain()
    automl.export("docker")
"""

import os
import time
import json
import logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)


class AutoMLResults:
    """Container for training results with convenient accessors."""
    
    def __init__(self, data):
        self._data = data
    
    @property
    def leaderboard(self):
        """Return leaderboard as a DataFrame."""
        lb = self._data.get('leaderboard', [])
        if lb:
            return pd.DataFrame(lb)
        return pd.DataFrame()
    
    @property
    def best_model(self):
        return self._data.get('best_model')
    
    @property
    def best_score(self):
        return self._data.get('best_score')
    
    @property
    def problem_type(self):
        return self._data.get('problem_type')
    
    @property
    def feature_importance(self):
        fi = self._data.get('feature_importance', [])
        if fi:
            return pd.DataFrame(fi)
        return pd.DataFrame()
    
    @property
    def timing(self):
        """Return pipeline timing breakdown."""
        return self._data.get('timing', {})
    
    def __repr__(self):
        return (
            f"AutoMLResults(best_model='{self.best_model}', "
            f"best_score={self.best_score}, "
            f"n_models={len(self._data.get('leaderboard', []))})"
        )


class AutoML:
    """Python SDK client for AutoML Studio.
    
    Args:
        server: Base URL of the AutoML Studio server.
        timeout: Request timeout in seconds.
    """
    
    def __init__(self, server="http://localhost:7860", timeout=30):
        self.server = server.rstrip('/')
        self.timeout = timeout
        self.session_id = None
        self._results = None
        self._profile = None
    
    def _url(self, path):
        return f"{self.server}/api/{path}"
    
    def _post(self, path, data=None, files=None):
        """Make a POST request and return JSON response."""
        try:
            resp = requests.post(
                self._url(path), json=data, files=files, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"AutoML Studio API error: {e}") from e
    
    def _get(self, path, params=None):
        """Make a GET request and return JSON response."""
        try:
            resp = requests.get(
                self._url(path), params=params, timeout=self.timeout
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"AutoML Studio API error: {e}") from e
    
    def _wait_for_completion(self, poll_interval=2, max_wait=3600):
        """Poll session status until complete or error."""
        start = time.time()
        while time.time() - start < max_wait:
            status = self._get('status', {'session_id': self.session_id})
            state = status.get('status', 'unknown')
            progress = status.get('progress', 0)
            msg = status.get('progress_message', '')
            
            if state == 'complete':
                logger.info("Pipeline step complete: %s", msg)
                return status
            elif state == 'error':
                raise RuntimeError(f"Pipeline error: {status.get('progress_message')}")
            
            logger.debug("Progress: %d%% - %s", progress, msg)
            time.sleep(poll_interval)
        
        raise TimeoutError(f"Pipeline did not complete within {max_wait}s")
    
    def fit(self, data, target=None, mode='balanced', time_budget=None, problem_statement=None):
        """
        Upload data and run the full AutoML pipeline.
        
        Args:
            data: Path to CSV/Excel/JSON/Parquet file, or pandas DataFrame.
            target: Name of the target column (auto-detected if not specified).
            mode: Training mode - 'quick' (3 models), 'balanced' (7), 'full' (14+).
            time_budget: Maximum training time in seconds.
            problem_statement: Natural language description of the problem.
        
        Returns:
            AutoMLResults with leaderboard, best model, feature importance.
        """
        # Step 1: Create session
        session_resp = self._post('session/create')
        self.session_id = session_resp.get('session_id')
        if not self.session_id:
            raise RuntimeError("Failed to create session")
        
        # Step 2: Upload
        if isinstance(data, pd.DataFrame):
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
                data.to_csv(f.name, index=False)
                filepath = f.name
        elif isinstance(data, str) and os.path.exists(data):
            filepath = data
        else:
            raise ValueError("data must be a file path or pandas DataFrame")
        
        upload_data = {
            'session_id': self.session_id,
            'problem_statement': problem_statement or '',
        }
        if target:
            upload_data['target_column'] = target
        
        # Upload file
        with open(filepath, 'rb') as f:
            self._post('upload', data=upload_data)
        
        self._wait_for_completion()
        
        # Step 3: Clean and transform
        self._post('clean', {'session_id': self.session_id})
        self._wait_for_completion()
        
        # Step 4: Train
        train_data = {'session_id': self.session_id, 'mode': mode}
        if time_budget:
            train_data['time_budget'] = time_budget
        
        self._post('train', train_data)
        self._wait_for_completion(poll_interval=5)
        
        # Get results
        results = self._get('results', {'session_id': self.session_id})
        self._results = AutoMLResults(results)
        
        return self._results
    
    def predict(self, data):
        """
        Make predictions using the best trained model.
        
        Args:
            data: pandas DataFrame or dict of feature values.
        
        Returns:
            Predictions as a list or numpy array.
        """
        if not self.session_id:
            raise RuntimeError("No trained model. Call fit() first.")
        
        if isinstance(data, pd.DataFrame):
            records = data.to_dict(orient='records')
        elif isinstance(data, dict):
            records = [data]
        elif isinstance(data, list):
            records = data
        else:
            raise ValueError("data must be a DataFrame, dict, or list of dicts")
        
        results = []
        for record in records:
            resp = self._post('predict', {
                'session_id': self.session_id,
                'features': record,
            })
            results.append(resp.get('prediction'))
        
        return results if len(results) > 1 else results[0]
    
    def explain(self):
        """
        Get SHAP explanations for the best model.
        
        Returns:
            dict with global_importance, summary_plot info.
        """
        if not self.session_id:
            raise RuntimeError("No trained model. Call fit() first.")
        
        return self._get('explainability', {'session_id': self.session_id})
    
    def export(self, format='script'):
        """
        Export the trained model as a deployment package.
        
        Args:
            format: 'script', 'docker', 'api', or 'pickle'.
        
        Returns:
            dict with download URL or package contents.
        """
        if not self.session_id:
            raise RuntimeError("No trained model. Call fit() first.")
        
        return self._post('export', {
            'session_id': self.session_id,
            'format': format,
        })
    
    @property
    def results(self):
        """Access the most recent training results."""
        return self._results
    
    def chat(self, message):
        """
        Chat with the AI assistant about the current session.
        
        Args:
            message: Natural language question or instruction.
        
        Returns:
            dict with response text and optional suggested_action.
        """
        return self._post('chat', {
            'session_id': self.session_id or '',
            'message': message,
        })
    
    def __repr__(self):
        if self.session_id:
            return f"AutoML(server='{self.server}', session='{self.session_id}')"
        return f"AutoML(server='{self.server}')"
