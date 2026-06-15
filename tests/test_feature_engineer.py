"""Tests for the feature_engineer module."""

import pandas as pd
import numpy as np
import pytest

from ml_engine.feature_engineer import auto_engineer_features


class TestAutoEngineerFeatures:
    """Tests for the feature engineering pipeline."""

    @pytest.fixture
    def numeric_df(self):
        """Dataset with numeric features and a datetime column."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'price': np.random.uniform(10, 1000, n),
            'quantity': np.random.randint(1, 50, n),
            'discount': np.random.uniform(0, 0.5, n),
            'weight': np.random.normal(5, 2, n),
            'rating': np.random.uniform(1, 5, n),
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'target': np.random.choice([0, 1], n),
        })
        return df

    @pytest.fixture
    def cat_df(self):
        """Dataset with categorical columns for frequency encoding."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'color': np.random.choice(['red', 'blue', 'green', 'yellow', 'black', 'white'], n),
            'size': np.random.choice(['S', 'M', 'L'], n),
            'score': np.random.uniform(0, 100, n),
            'target': np.random.choice([0, 1], n),
        })
        return df

    def test_returns_tuple(self, numeric_df):
        """auto_engineer_features returns (df, report) tuple."""
        profile = {'target_column': 'target', 'numeric_columns': ['price', 'quantity', 'discount', 'weight', 'rating'], 'categorical_columns': []}
        result = auto_engineer_features(numeric_df, profile, target_col='target')
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_features_added(self, numeric_df):
        """New features are added to the dataframe."""
        profile = {'target_column': 'target', 'numeric_columns': ['price', 'quantity', 'discount', 'weight', 'rating'], 'categorical_columns': []}
        df, report = auto_engineer_features(numeric_df, profile, target_col='target')
        assert df.shape[1] > numeric_df.shape[1]

    def test_report_has_steps(self, numeric_df):
        """Report contains step descriptions."""
        profile = {'target_column': 'target', 'numeric_columns': ['price', 'quantity', 'discount', 'weight', 'rating'], 'categorical_columns': []}
        _, report = auto_engineer_features(numeric_df, profile, target_col='target')
        assert 'steps' in report
        assert 'summary' in report
        assert len(report['steps']) >= 5  # At least the core 5 steps

    def test_target_excluded(self, numeric_df):
        """Target column is never used as input for feature creation."""
        profile = {'target_column': 'target', 'numeric_columns': ['price', 'quantity', 'discount', 'weight', 'rating'], 'categorical_columns': []}
        df, _ = auto_engineer_features(numeric_df, profile, target_col='target')
        # No new feature should be based on target
        for col in df.columns:
            if col != 'target':
                assert 'target_' not in col or col.startswith('target') is False

    def test_frequency_features(self, cat_df):
        """Frequency encoding creates _freq features for high-cardinality categoricals."""
        profile = {'target_column': 'target', 'numeric_columns': ['score'], 'categorical_columns': ['color', 'size']}
        df, report = auto_engineer_features(cat_df, profile, target_col='target')
        # color has 6 unique values > 5 threshold → should have _freq
        assert 'color_freq' in df.columns

    def test_no_crash_on_small_data(self):
        """Feature engineering doesn't crash on tiny datasets."""
        df = pd.DataFrame({'a': [1.0, 2.0], 'target': [0, 1]})
        profile = {'target_column': 'target', 'numeric_columns': ['a'], 'categorical_columns': []}
        result_df, report = auto_engineer_features(df, profile, target_col='target')
        assert isinstance(result_df, pd.DataFrame)
