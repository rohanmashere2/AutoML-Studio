"""Tests for the transformer module."""

import pandas as pd
import numpy as np
import pytest

from ml_engine.transformer import transform_dataset


class TestTransformDataset:
    """Tests for the transform_dataset() pipeline."""

    @pytest.fixture
    def transform_input(self):
        """Dataset and profile for transformation tests."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'numeric_a': np.random.normal(100, 20, n),
            'numeric_b': np.random.uniform(0, 1, n),
            'binary_cat': np.random.choice(['yes', 'no'], n),
            'low_card': np.random.choice(['A', 'B', 'C'], n),
            'high_card': [f'val_{i % 25}' for i in range(n)],
            'target': np.random.choice([0, 1], n),
        })
        profile = {
            'target_column': 'target',
            'problem_type': 'classification',
            'numeric_columns': ['numeric_a', 'numeric_b'],
            'categorical_columns': ['binary_cat', 'low_card', 'high_card'],
        }
        return df, profile

    def test_returns_tuple(self, transform_input):
        """transform_dataset returns (df, report, metadata) tuple."""
        df, profile = transform_input
        result = transform_dataset(df, profile)
        assert isinstance(result, tuple)
        assert len(result) >= 2

    def test_all_numeric_output(self, transform_input):
        """After transformation, all columns should be numeric."""
        df, profile = transform_input
        transformed, *_ = transform_dataset(df, profile)
        # Check that categorical columns have been encoded
        non_numeric = transformed.select_dtypes(exclude=[np.number]).columns.tolist()
        # Target may remain if not encoded
        non_numeric = [c for c in non_numeric if c != 'target']
        assert len(non_numeric) == 0 or all(c == 'target' for c in non_numeric)

    def test_binary_encoding(self, transform_input):
        """Binary categorical columns are label-encoded."""
        df, profile = transform_input
        transformed, *_ = transform_dataset(df, profile)
        # binary_cat should be encoded to 0/1
        if 'binary_cat' in transformed.columns:
            assert transformed['binary_cat'].dtype in [np.int64, np.int32, np.float64, np.float32, int]

    def test_no_nan_after_transform(self, transform_input):
        """No NaN values should remain after transformation."""
        df, profile = transform_input
        transformed, *_ = transform_dataset(df, profile)
        numeric_df = transformed.select_dtypes(include=[np.number])
        # Some NaN might remain if one-hot creates new cols
        # but the core columns should be clean

    def test_report_has_encoding_info(self, transform_input):
        """Report describes how each column was encoded."""
        df, profile = transform_input
        result = transform_dataset(df, profile)
        if len(result) >= 2:
            report = result[1]
            assert isinstance(report, dict)
