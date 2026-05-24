"""
Tests for ml_engine/transformer.py — verifying no data leakage.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from ml_engine.transformer import transform_dataset, fit_scaler, apply_scaler


def _make_profile(df, target='target'):
    return {
        'target_column': target,
        'problem_type': 'classification',
        'n_cols': len(df.columns),
        'text_columns': [],
    }


class TestNoDataLeakage:
    """Ensure scaling is deferred and applied only on training data."""

    def test_transform_does_not_scale(self, small_dataset):
        """transform_dataset should NOT scale features anymore."""
        df = small_dataset.copy()
        profile = _make_profile(df)
        transformed_df, report, metadata = transform_dataset(df, profile)

        # Scaler should be None (scaling deferred to trainer)
        assert metadata['scaler'] is None
        # numeric_cols_to_scale should list the numeric columns
        assert len(metadata['numeric_cols_to_scale']) > 0

    def test_fit_scaler_only_on_train(self, small_dataset):
        """fit_scaler should learn statistics only from training data."""
        df = small_dataset.copy()
        profile = _make_profile(df)
        transformed_df, _, metadata = transform_dataset(df, profile)

        target_col = metadata['target_column']
        X = transformed_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = transformed_df[target_col]

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25, random_state=42)

        numeric_cols = metadata['numeric_cols_to_scale']
        cols_present = [c for c in numeric_cols if c in X_train.columns]
        scaler = fit_scaler(X_train, cols_present)

        assert scaler is not None
        # Scaler mean should match training data mean, NOT full data mean
        train_means = X_train[cols_present].mean().values
        np.testing.assert_array_almost_equal(scaler.mean_, train_means, decimal=5)

    def test_scaled_train_has_zero_mean(self, small_dataset):
        """After applying the scaler, training data should have ~zero mean."""
        df = small_dataset.copy()
        profile = _make_profile(df)
        transformed_df, _, metadata = transform_dataset(df, profile)

        target_col = metadata['target_column']
        X = transformed_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = transformed_df[target_col]

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25, random_state=42)

        numeric_cols = metadata['numeric_cols_to_scale']
        cols_present = [c for c in numeric_cols if c in X_train.columns]
        scaler = fit_scaler(X_train, cols_present)
        X_train_scaled = apply_scaler(X_train, scaler, cols_present)

        # Training data means should be approximately zero
        means = X_train_scaled[cols_present].mean()
        np.testing.assert_array_almost_equal(means.values, 0, decimal=5)

    def test_test_set_has_nonzero_mean(self, small_dataset):
        """After applying the scaler, test data should NOT have exact zero mean."""
        df = small_dataset.copy()
        profile = _make_profile(df)
        transformed_df, _, metadata = transform_dataset(df, profile)

        target_col = metadata['target_column']
        X = transformed_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        y = transformed_df[target_col]

        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.25, random_state=42)

        numeric_cols = metadata['numeric_cols_to_scale']
        cols_present = [c for c in numeric_cols if c in X_train.columns]
        scaler = fit_scaler(X_train, cols_present)
        X_test_scaled = apply_scaler(X_test, scaler, cols_present)

        # Test data means should generally NOT be exactly zero
        # (different distribution from train)
        means = X_test_scaled[cols_present].mean()
        assert not np.allclose(means.values, 0, atol=1e-5), (
            "Test set means are suspiciously close to zero — possible data leakage"
        )
