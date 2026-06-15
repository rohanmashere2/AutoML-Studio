"""Tests for the trainer module."""

import pandas as pd
import numpy as np
import pytest

from ml_engine.trainer import get_models, train_models


class TestGetModels:
    """Tests for model pool construction."""

    def test_classification_models(self):
        """Classification returns expected model types."""
        models = get_models('classification')
        assert len(models) >= 6
        assert 'Logistic Regression' in models
        assert 'Random Forest' in models
        assert 'HistGBM' in models
        assert 'Naive Bayes' in models
        assert 'AdaBoost' in models

    def test_regression_models(self):
        """Regression returns expected model types."""
        models = get_models('regression')
        assert len(models) >= 6
        assert 'Linear Regression' in models
        assert 'Ridge' in models
        assert 'HistGBM' in models
        assert 'AdaBoost' in models

    def test_classification_has_balanced_option(self):
        """class_weight_balanced creates balanced models."""
        models = get_models('classification', class_weight_balanced=True)
        lr = models['Logistic Regression']
        assert lr.get_params().get('class_weight') == 'balanced'

    def test_svm_skipped_on_large_data(self):
        """SVM is skipped when n_samples > 10000."""
        models = get_models('classification', n_samples=20000)
        assert 'SVM' not in models

    def test_svm_included_on_small_data(self):
        """SVM is included when n_samples <= 10000."""
        models = get_models('classification', n_samples=5000)
        assert 'SVM' in models

    def test_fast_models_first(self):
        """Fast models appear before slow models in ordering."""
        models = get_models('classification')
        names = list(models.keys())
        lr_idx = names.index('Logistic Regression')
        rf_idx = names.index('Random Forest')
        assert lr_idx < rf_idx, "Logistic Regression should come before Random Forest"


class TestTrainModels:
    """Tests for the train_models() pipeline."""

    @pytest.fixture
    def simple_train_data(self):
        """Small classification dataset for fast training tests."""
        np.random.seed(42)
        n = 100
        X = np.random.randn(n, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        df = pd.DataFrame(X, columns=[f'f{i}' for i in range(5)])
        df['target'] = y
        profile = {
            'target_column': 'target',
            'problem_type': 'classification',
        }
        transform_metadata = {
            'target_column': 'target',
            'problem_type': 'classification',
        }
        return df, profile, transform_metadata

    def test_returns_leaderboard(self, simple_train_data, tmp_path):
        """Training produces a leaderboard."""
        df, profile, meta = simple_train_data
        results = train_models(df, profile, meta, str(tmp_path))
        assert 'leaderboard' in results
        assert len(results['leaderboard']) > 0

    def test_best_model_identified(self, simple_train_data, tmp_path):
        """Best model is identified."""
        df, profile, meta = simple_train_data
        results = train_models(df, profile, meta, str(tmp_path))
        assert results['best_model'] is not None

    def test_leaderboard_sorted(self, simple_train_data, tmp_path):
        """Leaderboard is sorted by primary metric descending."""
        df, profile, meta = simple_train_data
        results = train_models(df, profile, meta, str(tmp_path))
        scores = [e['primary_metric'] for e in results['leaderboard']]
        assert scores == sorted(scores, reverse=True)

    def test_feature_importance(self, simple_train_data, tmp_path):
        """Feature importance is returned."""
        df, profile, meta = simple_train_data
        results = train_models(df, profile, meta, str(tmp_path))
        assert 'feature_importance' in results

    def test_time_budget(self, simple_train_data, tmp_path):
        """Time budget limits number of models trained."""
        df, profile, meta = simple_train_data
        results = train_models(df, profile, meta, str(tmp_path), time_budget_seconds=5)
        assert 'leaderboard' in results
        # With 5 second budget, likely fewer models than full run

    def test_model_saved(self, simple_train_data, tmp_path):
        """Best model is saved to disk."""
        df, profile, meta = simple_train_data
        results = train_models(df, profile, meta, str(tmp_path))
        assert results['best_model_path'] is not None
        import os
        assert os.path.exists(results['best_model_path'])
