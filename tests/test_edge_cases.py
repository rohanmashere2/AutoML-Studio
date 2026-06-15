"""Tests for edge cases across the AutoML pipeline."""

import pandas as pd
import numpy as np
import pytest

from ml_engine.profiler import profile_dataset
from ml_engine.cleaner import clean_dataset


class TestEdgeCases:
    """Tests for unusual or extreme inputs."""

    def test_single_feature_dataset(self):
        """Pipeline handles dataset with only one feature + target."""
        df = pd.DataFrame({
            'x': np.random.randn(50),
            'target': np.random.choice([0, 1], 50),
        })
        profile = profile_dataset(df)
        assert profile['n_cols'] == 2

    def test_constant_feature(self):
        """Dataset with a constant feature column is handled."""
        df = pd.DataFrame({
            'const': [1.0] * 50,
            'real': np.random.randn(50),
            'target': np.random.choice([0, 1], 50),
        })
        profile = profile_dataset(df)
        cleaned, _ = clean_dataset(df, profile)
        assert isinstance(cleaned, pd.DataFrame)

    def test_all_same_target(self):
        """All-same target is caught during profiling."""
        df = pd.DataFrame({
            'x': np.random.randn(50),
            'target': [1] * 50,
        })
        profile = profile_dataset(df)
        # Profile should still work but the target has only 1 unique value
        assert profile is not None

    def test_extremely_imbalanced(self):
        """Highly imbalanced dataset (100:1 ratio) is handled."""
        np.random.seed(42)
        n = 202
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'target': [0] * 200 + [1] * 2,
        })
        profile = profile_dataset(df)
        cleaned, report = clean_dataset(df, profile)
        assert isinstance(cleaned, pd.DataFrame)
        assert len(cleaned) > 0

    def test_mixed_missing_patterns(self):
        """Dataset with different missing patterns per column."""
        np.random.seed(42)
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0, np.nan, 5.0] * 10,
            'b': [np.nan, 2.0, np.nan, 4.0, np.nan] * 10,
            'c': [1.0, 2.0, 3.0, 4.0, 5.0] * 10,
            'target': [0, 1] * 25,
        })
        profile = {'target_column': 'target'}
        cleaned, report = clean_dataset(df, profile)
        assert cleaned.isnull().sum().sum() == 0

    def test_unicode_column_names(self):
        """Columns with unicode names are handled."""
        df = pd.DataFrame({
            'prêço': [1.0, 2.0, 3.0],
            'größe': [4.0, 5.0, 6.0],
            'target': [0, 1, 0],
        })
        profile = profile_dataset(df)
        assert isinstance(profile, dict)

    def test_very_wide_dataset(self):
        """Dataset with many columns (100+) is profiled."""
        np.random.seed(42)
        n_cols = 100
        df = pd.DataFrame(
            np.random.randn(50, n_cols),
            columns=[f'col_{i}' for i in range(n_cols)],
        )
        df['target'] = np.random.choice([0, 1], 50)
        profile = profile_dataset(df)
        assert profile['n_cols'] == n_cols + 1
