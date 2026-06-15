"""Tests for the cleaner module."""

import pandas as pd
import numpy as np
import pytest

from ml_engine.cleaner import clean_dataset


class TestCleanDataset:
    """Tests for the clean_dataset() orchestrator."""

    def test_returns_tuple(self, mixed_df):
        """clean_dataset returns (df, report) tuple."""
        profile = {'target_column': 'target'}
        result = clean_dataset(mixed_df, profile)
        assert isinstance(result, tuple)
        assert len(result) == 2
        df, report = result
        assert isinstance(df, pd.DataFrame)
        assert isinstance(report, dict)

    def test_no_missing_after_clean(self, mixed_df):
        """All missing values are filled after cleaning."""
        profile = {'target_column': 'target'}
        df, _ = clean_dataset(mixed_df, profile)
        assert df.isnull().sum().sum() == 0

    def test_duplicates_removed(self):
        """Duplicate rows are removed."""
        df = pd.DataFrame({
            'a': [1, 1, 2, 3], 'b': [4, 4, 5, 6], 'target': [0, 0, 1, 1]
        })
        cleaned, report = clean_dataset(df, {'target_column': 'target'})
        assert len(cleaned) == 3

    def test_report_has_steps(self, mixed_df):
        """Cleaning report contains step entries."""
        profile = {'target_column': 'target'}
        _, report = clean_dataset(mixed_df, profile)
        assert 'steps' in report
        assert len(report['steps']) > 0
        assert 'summary' in report

    def test_summary_counts(self, mixed_df):
        """Summary correctly counts rows/cols changes."""
        profile = {'target_column': 'target'}
        _, report = clean_dataset(mixed_df, profile)
        summary = report['summary']
        assert summary['original_rows'] == 150
        assert summary['cleaned_rows'] <= 150

    def test_imputation_knn(self, mixed_df):
        """KNN imputation strategy works."""
        profile = {'target_column': 'target'}
        df, report = clean_dataset(mixed_df, profile, imputation_strategy='knn')
        assert df.isnull().sum().sum() == 0

    def test_imputation_simple(self, mixed_df):
        """Simple imputation strategy works."""
        profile = {'target_column': 'target'}
        df, report = clean_dataset(mixed_df, profile, imputation_strategy='simple')
        assert df.isnull().sum().sum() == 0

    def test_all_nan_columns_dropped(self, all_nan_df):
        """Columns that are entirely NaN are dropped."""
        profile = {'target_column': 'target'}
        df, report = clean_dataset(all_nan_df, profile)
        # Columns a and b are 100% NaN → should be dropped (threshold 70%)
        assert 'a' not in df.columns
        assert 'b' not in df.columns


class TestOutlierHandling:
    """Tests for outlier detection and capping."""

    def test_outliers_capped(self):
        """Extreme outliers are capped."""
        np.random.seed(42)
        df = pd.DataFrame({
            'normal': np.random.normal(50, 10, 100),
            'target': np.random.choice([0, 1], 100),
        })
        df.loc[0, 'normal'] = 1000  # extreme outlier
        profile = {'target_column': 'target'}
        cleaned, _ = clean_dataset(df, profile)
        assert cleaned['normal'].max() < 1000
