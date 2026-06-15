"""Tests for the profiler module."""

import os
import pandas as pd
import numpy as np
import pytest

from ml_engine.profiler import profile_dataset, read_dataset


class TestReadDataset:
    """Tests for read_dataset() file loading."""

    def test_read_csv(self, tmp_csv):
        """CSV files are loaded correctly."""
        df = read_dataset(tmp_csv)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert 'target' in df.columns

    def test_read_nonexistent_file(self):
        """Non-existent file raises or returns None."""
        result = read_dataset("/nonexistent/path.csv")
        assert result is None or (isinstance(result, pd.DataFrame) and result.empty)


class TestProfileDataset:
    """Tests for profile_dataset() analysis."""

    def test_profile_returns_required_keys(self, classification_df):
        """Profile dict contains all expected keys."""
        profile = profile_dataset(classification_df)
        assert isinstance(profile, dict)
        required = ['n_rows', 'n_cols', 'target_column', 'problem_type']
        for key in required:
            assert key in profile, f"Missing key: {key}"

    def test_target_detection_classification(self, classification_df):
        """Target column is detected for classification datasets."""
        profile = profile_dataset(classification_df)
        assert profile.get('target_column') == 'target'

    def test_problem_type_classification(self, classification_df):
        """Binary target is detected as classification."""
        profile = profile_dataset(classification_df)
        assert profile.get('problem_type') == 'classification'

    def test_problem_type_regression(self, regression_df):
        """Continuous target is detected as regression."""
        profile = profile_dataset(regression_df)
        assert profile.get('problem_type') == 'regression'

    def test_row_col_counts(self, classification_df):
        """Row and column counts are accurate."""
        profile = profile_dataset(classification_df)
        assert profile['n_rows'] == 200
        assert profile['n_cols'] == 11
